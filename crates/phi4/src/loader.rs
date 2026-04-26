use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use memmap2::Mmap;
use safetensors::SafeTensors;

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

/// Raw-safetensors loader used for the `--no-bake` code path. Mirrors
/// `qwen35::loader::WeightLoader`; kept per-family to avoid cross-crate coupling.
/// Fused `qkv_proj` / `gate_up_proj` tensors are split on the fly when requested
/// via [`WeightLoader::load_split_to_gpu`].
pub struct WeightLoader {
    shards: Vec<Mmap>,
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
                Err(LoadError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no safetensors files found in {}", dir.display()),
                )))
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
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let weight_map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| LoadError::NotFound("weight_map key in index.json".into()))?;
        let mut shard_files: Vec<String> = Vec::new();
        let mut shard_idx_map: BTreeMap<String, usize> = BTreeMap::new();
        for filename in weight_map.values() {
            let filename = filename.as_str().unwrap_or("").to_string();
            if !shard_idx_map.contains_key(&filename) {
                shard_idx_map.insert(filename.clone(), shard_files.len());
                shard_files.push(filename);
            }
        }
        let mut shards = Vec::with_capacity(shard_files.len());
        for filename in &shard_files {
            let file = File::open(dir.join(filename))?;
            shards.push(unsafe { Mmap::map(&file)? });
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

    /// Load a contiguous row-slice `[start_row..start_row + rows, cols]` of a 2D
    /// tensor directly to GPU. Used to split Phi's fused `qkv_proj` / `gate_up_proj`
    /// without materializing an intermediate host copy of the whole tensor.
    pub fn load_row_slice_to_gpu(
        &self,
        name: &str,
        start_row: usize,
        rows: usize,
        ordinal: usize,
    ) -> Result<GpuBuffer, LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = tensors.tensor(name)?;
        let dtype = ScalarType::from_safetensors(view.dtype())
            .ok_or_else(|| LoadError::UnsupportedDtype(format!("{:?}", view.dtype())))?;
        let shape = view.shape();
        if shape.len() != 2 {
            return Err(LoadError::UnsupportedDtype(format!(
                "row_slice expects 2D tensor, got shape {shape:?}"
            )));
        }
        if start_row + rows > shape[0] {
            return Err(LoadError::NotFound(format!(
                "row_slice out of bounds: start={start_row} rows={rows} shape={shape:?}"
            )));
        }
        let dtype_bytes = dtype.size_in_bytes();
        let cols = shape[1];
        let row_stride = cols * dtype_bytes;
        let data = view.data();
        let start_byte = start_row * row_stride;
        let end_byte = (start_row + rows) * row_stride;
        let slice = &data[start_byte..end_byte];
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &[rows, cols], slice)?;
        Ok(buf)
    }
}
