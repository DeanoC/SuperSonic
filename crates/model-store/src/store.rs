use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use gpu_hal::{GpuBuffer, ScalarType};
use memmap2::Mmap;

use crate::manifest::{Manifest, TensorMeta};
use crate::Error;

/// A memory-mapped baked weight store for fast GPU loading.
pub struct BakedStore {
    _mmap: Mmap,
    data: *const u8,
    data_len: usize,
    index: HashMap<String, TensorMeta>,
}

// Safety: the mmap is immutable and lives as long as BakedStore.
unsafe impl Send for BakedStore {}
unsafe impl Sync for BakedStore {}

fn parse_dtype(name: &str) -> Result<ScalarType, Error> {
    ScalarType::from_name(name)
        .ok_or_else(|| Error::UnsupportedDtype(name.to_string()))
}

impl BakedStore {
    /// Open a baked package from a bake directory.
    /// Reads manifest.json and mmaps weights.bin.
    pub fn open(bake_dir: &Path) -> Result<Self, Error> {
        let manifest_text = std::fs::read_to_string(crate::manifest_path(bake_dir))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)?;

        let weights_file = File::open(crate::weights_bin_path(bake_dir))?;
        let mmap = unsafe { Mmap::map(&weights_file)? };

        let data = mmap.as_ptr();
        let data_len = mmap.len();

        let mut index = HashMap::with_capacity(manifest.tensors.len());
        for entry in manifest.tensors {
            index.insert(entry.name.clone(), entry);
        }

        Ok(Self {
            _mmap: mmap,
            data,
            data_len,
            index,
        })
    }

    /// Check if a tensor exists in the store.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Get the shape of a tensor without loading it.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|m| m.shape.as_slice())
    }

    /// Return the raw mmap-backed bytes of a tensor. Useful for tensors that
    /// are too large to upload to GPU in full (e.g. Gemma 4's
    /// `embed_tokens_per_layer`, which is row-accessed per-token). The slice
    /// lives as long as the `BakedStore`'s mmap.
    pub fn raw_bytes(&self, name: &str) -> Option<&[u8]> {
        let meta = self.index.get(name)?;
        let start = meta.offset as usize;
        let end = start + meta.byte_len as usize;
        if end > self.data_len {
            return None;
        }
        let slice = unsafe {
            std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize)
        };
        Some(slice)
    }

    /// Load a tensor from the baked store directly to GPU memory.
    /// One memcpy (H2D), zero parsing or transformation.
    pub fn load_to_gpu(&self, name: &str, ordinal: usize) -> Result<GpuBuffer, Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;

        let start = meta.offset as usize;
        let end = start + meta.byte_len as usize;
        if end > self.data_len {
            return Err(Error::Other(format!(
                "tensor '{}' extends past end of weights.bin (offset={}, len={}, file_len={})",
                name, meta.offset, meta.byte_len, self.data_len,
            )));
        }

        let slice = unsafe { std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize) };
        let dtype = parse_dtype(&meta.dtype)?;
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &meta.shape, slice)?;
        Ok(buf)
    }
}
