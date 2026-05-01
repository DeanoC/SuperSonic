use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("malformed model dir: {0}")]
    Malformed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Bf16,
    F16,
    F32,
    F64,
    I64,
    I32,
    U8,
    Bool,
}

impl ScalarKind {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }

    pub fn from_safetensors(d: Dtype) -> Result<Self, LoadError> {
        Ok(match d {
            Dtype::BF16 => Self::Bf16,
            Dtype::F16 => Self::F16,
            Dtype::F32 => Self::F32,
            Dtype::F64 => Self::F64,
            Dtype::I64 => Self::I64,
            Dtype::I32 => Self::I32,
            Dtype::U8 => Self::U8,
            Dtype::BOOL => Self::Bool,
            other => return Err(LoadError::UnsupportedDtype(format!("{other:?}"))),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: ScalarKind,
}

impl TensorMeta {
    pub fn elem_count(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64
    }

    pub fn byte_size(&self) -> u64 {
        self.elem_count() * self.dtype.size_bytes() as u64
    }
}

/// Read-only safetensors weight loader. Enumerates the BF16 HF download (or any
/// safetensors-compatible model dir). Built for the dry-run and shape-validation
/// path; the GPU-allocating path lands in PR 4 alongside the kernel work.
pub struct WeightLoader {
    shards: Vec<Arc<Mmap>>,
    /// tensor name -> shard index
    index: BTreeMap<String, usize>,
}

impl WeightLoader {
    pub fn from_dir(dir: &Path) -> Result<Self, LoadError> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::from_sharded(dir, &index_path)
        } else {
            let single = dir.join("model.safetensors");
            if single.exists() {
                Self::from_single(&single)
            } else {
                Err(LoadError::Malformed(format!(
                    "no safetensors files found in {}",
                    dir.display()
                )))
            }
        }
    }

    fn from_single(path: &Path) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let tensors = SafeTensors::deserialize(&mmap[..])?;
        let mut index = BTreeMap::new();
        for name in tensors.names() {
            index.insert(name.to_string(), 0);
        }
        Ok(Self {
            shards: vec![mmap],
            index,
        })
    }

    fn from_sharded(dir: &Path, index_path: &Path) -> Result<Self, LoadError> {
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let weight_map = raw["weight_map"].as_object().ok_or_else(|| {
            LoadError::Malformed("missing weight_map in safetensors index".into())
        })?;
        let mut shard_files: Vec<PathBuf> = Vec::new();
        let mut shard_idx_map: BTreeMap<String, usize> = BTreeMap::new();
        for filename in weight_map.values() {
            let filename = filename.as_str().unwrap_or("").to_string();
            if !shard_idx_map.contains_key(&filename) {
                shard_idx_map.insert(filename.clone(), shard_files.len());
                shard_files.push(dir.join(&filename));
            }
        }
        let mut shards = Vec::with_capacity(shard_files.len());
        for path in &shard_files {
            let file = File::open(path)?;
            shards.push(Arc::new(unsafe { Mmap::map(&file)? }));
        }
        let mut index = BTreeMap::new();
        for (tensor_name, filename) in weight_map {
            let filename = filename.as_str().unwrap_or("");
            if let Some(&shard_idx) = shard_idx_map.get(filename) {
                index.insert(tensor_name.clone(), shard_idx);
            }
        }
        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Total number of tensors across all shards.
    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    /// All tensor names known to this loader, sorted lexicographically.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn meta(&self, name: &str) -> Result<TensorMeta, LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx][..])?;
        let view = tensors.tensor(name)?;
        Ok(TensorMeta {
            shape: view.shape().to_vec(),
            dtype: ScalarKind::from_safetensors(view.dtype())?,
        })
    }

    /// Sum of byte sizes for the listed tensors. Reports any missing names so the
    /// caller can flag schema drift before allocating.
    pub fn accumulate_bytes<I, S>(&self, names: I) -> Result<u64, LoadError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut total: u64 = 0;
        for name in names {
            let m = self.meta(name.as_ref())?;
            total = total.saturating_add(m.byte_size());
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_dir_reports_malformed() {
        let dir = std::env::temp_dir().join("supersonic-qwen36-moe-no-such-dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = WeightLoader::from_dir(&dir)
            .err()
            .expect("expected error from empty dir");
        assert!(
            matches!(err, LoadError::Malformed(_)),
            "expected Malformed, got {err:?}"
        );
    }

    #[test]
    fn scalar_kind_size_table() {
        assert_eq!(ScalarKind::Bf16.size_bytes(), 2);
        assert_eq!(ScalarKind::F16.size_bytes(), 2);
        assert_eq!(ScalarKind::F32.size_bytes(), 4);
        assert_eq!(ScalarKind::I64.size_bytes(), 8);
        assert_eq!(ScalarKind::U8.size_bytes(), 1);
    }
}
