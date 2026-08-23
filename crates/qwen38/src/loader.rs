use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use memmap2::Mmap;
use safetensors::SafeTensors;

/// A safetensors weight loader that mmaps shard files and copies tensors to GPU.
pub struct WeightLoader {
    shards: Vec<Mmap>,
    /// Maps tensor name → shard index.
    index: BTreeMap<String, usize>,
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
    Gpu(GpuError),
    NotFound(String),
    UnsupportedDtype(String),
    Json(serde_json::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors error: {e}"),
            Self::Gpu(e) => write!(f, "GPU error: {e}"),
            Self::NotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedDtype(msg) => write!(f, "unsupported dtype: {msg}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(e)
    }
}
impl From<GpuError> for LoadError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}
impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl WeightLoader {
    /// Open a model directory containing either a single `model.safetensors` or
    /// sharded files with `model.safetensors.index.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, LoadError> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::from_sharded(dir, &index_path)
        } else {
            // Try single-shard naming variants
            let single = dir.join("model.safetensors");
            if single.exists() {
                Self::from_single(&single)
            } else {
                // Try the -00001-of-00001 variant
                let pattern = dir.join("model.safetensors-00001-of-00001.safetensors");
                if pattern.exists() {
                    Self::from_single(&pattern)
                } else {
                    Err(LoadError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no safetensors files found in {}", dir.display()),
                    )))
                }
            }
        }
    }

    fn from_single(path: &Path) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
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
        let index_text = std::fs::read_to_string(index_path)?;
        let raw: serde_json::Value = serde_json::from_str(&index_text)?;
        let weight_map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| LoadError::NotFound("weight_map key in index.json".into()))?;

        // Collect unique shard filenames in order
        let mut shard_files: Vec<String> = Vec::new();
        let mut shard_idx_map: BTreeMap<String, usize> = BTreeMap::new();
        for filename in weight_map.values() {
            let filename = filename.as_str().unwrap_or("").to_string();
            if !shard_idx_map.contains_key(&filename) {
                shard_idx_map.insert(filename.clone(), shard_files.len());
                shard_files.push(filename);
            }
        }

        // Mmap each shard
        let mut shards = Vec::with_capacity(shard_files.len());
        for filename in &shard_files {
            let path = dir.join(filename);
            let file = File::open(&path)?;
            shards.push(unsafe { Mmap::map(&file)? });
        }

        // Build tensor → shard index map
        let mut index = BTreeMap::new();
        for (tensor_name, filename) in weight_map {
            let filename = filename.as_str().unwrap_or("");
            if let Some(&shard_idx) = shard_idx_map.get(filename) {
                index.insert(tensor_name.clone(), shard_idx);
            }
        }

        Ok(Self { shards, index })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Load a tensor directly to GPU device memory.
    pub fn load_to_gpu(&self, name: &str, ordinal: usize) -> Result<GpuBuffer, LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = tensors.tensor(name)?;
        let dtype = ScalarType::from_safetensors(view.dtype())
            .ok_or_else(|| LoadError::UnsupportedDtype(format!("{:?}", view.dtype())))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &shape, view.data())?;
        Ok(buf)
    }

    /// Load one BF16 row from a 2D tensor and expand it to F32 on CPU.
    pub fn load_bf16_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = tensors.tensor(name)?;
        if view.dtype() != safetensors::Dtype::BF16 {
            return Err(LoadError::UnsupportedDtype(format!(
                "{name}: expected BF16, got {:?}",
                view.dtype()
            )));
        }
        let shape = view.shape();
        if shape.len() != 2 {
            return Err(LoadError::UnsupportedDtype(format!(
                "{name}: expected rank-2 tensor, got shape {shape:?}"
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        if row >= rows {
            return Err(LoadError::NotFound(format!(
                "{name}: row {row} out of range for {rows} rows"
            )));
        }
        let row_bytes = cols * 2;
        let start = row * row_bytes;
        let end = start + row_bytes;
        let data = &view.data()[start..end];
        Ok(data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}
