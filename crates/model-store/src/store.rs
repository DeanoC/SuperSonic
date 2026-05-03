use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use gpu_hal::{
    copy_h2d, sync, GpuBuffer, ScalarType, VirtualAllocationRole, VirtualArena, VirtualBacking,
};
use memmap2::Mmap;

use crate::manifest::{LayoutTag, Manifest, TensorMeta};
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
    ScalarType::from_name(name).ok_or_else(|| Error::UnsupportedDtype(name.to_string()))
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

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name)
    }

    pub fn layout(&self, name: &str) -> Option<&LayoutTag> {
        self.index.get(name).map(|m| &m.layout)
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
        let slice =
            unsafe { std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize) };
        Some(slice)
    }

    pub fn raw_byte_range(
        &self,
        name: &str,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<&[u8], Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let tensor_end = (meta.offset as usize)
            .checked_add(meta.byte_len as usize)
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{name}' byte range overflows (offset={}, len={})",
                    meta.offset, meta.byte_len
                ))
            })?;
        if tensor_end > self.data_len {
            return Err(Error::Other(format!(
                "tensor '{}' extends past end of weights.bin (offset={}, len={}, file_len={})",
                name, meta.offset, meta.byte_len, self.data_len,
            )));
        }
        let range_end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' raw range overflows: offset={byte_offset} len={byte_len}"
            ))
        })?;
        if range_end > meta.byte_len as usize {
            return Err(Error::Other(format!(
                "tensor '{name}' raw range [{byte_offset}, {range_end}) exceeds byte_len={}",
                meta.byte_len
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.data.add(meta.offset as usize).add(byte_offset),
                byte_len,
            )
        })
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

        let slice =
            unsafe { std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize) };
        let dtype = parse_dtype(&meta.dtype)?;
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &meta.shape, slice)?;
        Ok(buf)
    }

    /// Load a tensor into a role-tagged virtual allocation.
    ///
    /// This is the low-level entry point for virtual weights and MoE expert
    /// islands: the allocation keeps a stable virtual address, exposes logical
    /// versus page-resident accounting through `VirtualArena`, and can later be
    /// evicted/restored by the HAL policy layer.
    pub fn load_to_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        name: &str,
        role: VirtualAllocationRole,
    ) -> Result<usize, Error> {
        let id = self.reserve_virtual_arena(arena, name, role)?;
        let len = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?
            .byte_len as usize;
        self.load_range_to_virtual_arena(arena, id, name, 0, len)?;
        Ok(id)
    }

    /// Reserve a stable virtual address range for one baked tensor without
    /// making any physical pages resident yet.
    pub fn reserve_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        name: &str,
        role: VirtualAllocationRole,
    ) -> Result<usize, Error> {
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

        let slice =
            unsafe { std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize) };
        let dtype = parse_dtype(&meta.dtype)?;
        let expected_len = meta
            .shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{}' shape {:?} overflows byte-size calculation",
                    name, meta.shape
                ))
            })?;
        if slice.len() != expected_len {
            return Err(Error::Other(format!(
                "tensor '{}' manifest byte_len={} does not match dtype={} shape={:?} expected_bytes={}",
                name,
                slice.len(),
                meta.dtype,
                meta.shape,
                expected_len,
            )));
        }
        arena
            .reserve(name.to_string(), role, dtype, &meta.shape)
            .map_err(Error::from)
    }

    /// Upload a byte range from a baked tensor into an existing virtual
    /// allocation. The allocation must have been reserved from the same tensor.
    pub fn load_range_to_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        allocation_id: usize,
        name: &str,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let tensor_end = (meta.offset as usize)
            .checked_add(meta.byte_len as usize)
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{name}' byte range overflows (offset={}, len={})",
                    meta.offset, meta.byte_len
                ))
            })?;
        if tensor_end > self.data_len {
            return Err(Error::Other(format!(
                "tensor '{}' extends past end of weights.bin (offset={}, len={}, file_len={})",
                name, meta.offset, meta.byte_len, self.data_len,
            )));
        }
        let range_end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' load range overflows: offset={byte_offset} len={byte_len}"
            ))
        })?;
        if range_end > meta.byte_len as usize {
            return Err(Error::Other(format!(
                "tensor '{name}' load range [{byte_offset}, {range_end}) exceeds byte_len={}",
                meta.byte_len
            )));
        }

        let dtype = parse_dtype(&meta.dtype)?;
        let ordinal = arena.device_ordinal();
        let allocation = arena.allocation_mut(allocation_id).ok_or_else(|| {
            Error::Other(format!("virtual allocation id {allocation_id} missing"))
        })?;
        let buffer = allocation.buffer_mut();
        if buffer.dtype() != dtype || buffer.shape() != meta.shape.as_slice() {
            return Err(Error::Other(format!(
                "virtual allocation id {allocation_id} does not match tensor '{name}' \
                 (allocation dtype={:?} shape={:?}, tensor dtype={} shape={:?})",
                buffer.dtype(),
                buffer.shape(),
                meta.dtype,
                meta.shape
            )));
        }
        buffer.map_range_bytes(byte_offset, byte_len)?;
        let src = unsafe { self.data.add(meta.offset as usize).add(byte_offset) as *const _ };
        copy_h2d(ordinal, buffer.offset_mut_ptr(byte_offset), src, byte_len)?;
        sync(ordinal)?;
        Ok(())
    }

    /// Convenience constructor for a virtual arena using this store's common
    /// CPU-backup residency policy.
    pub fn virtual_weight_arena(ordinal: usize) -> VirtualArena {
        VirtualArena::new(ordinal, VirtualBacking::CpuBackup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{LayoutTag, Manifest, TensorMeta, FORMAT_VERSION};

    /// End-to-end loadability test against a real Qwen3.6-MoE bake.
    /// Skipped when `SUPERSONIC_QWEN36_MOE_BAKE_DIR` is unset so CI / non-bake
    /// machines stay green. Exercises mmap, manifest parse, and per-tensor
    /// shape/layout/byte-range invariants the runtime relies on.
    #[test]
    fn qwen36_moe_bake_loadable() {
        let Ok(bake_dir_str) = std::env::var("SUPERSONIC_QWEN36_MOE_BAKE_DIR") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_MOE_BAKE_DIR not set. Point it at a bake \
                 directory like .supersonic/v2-int4-gptq to validate end-to-end \
                 loadability of a real Qwen3.6-MoE INT4 GPTQ bake."
            );
            return;
        };
        let bake_dir = Path::new(&bake_dir_str);
        let store = BakedStore::open(bake_dir).expect("open bake");

        // 1. Vocab/output sanity. lm_head is INT4-packed so its column count
        //    is hidden/2 (2 nibbles per byte). Companion scale + zero must
        //    both be present in the index.
        let lm = store
            .meta("lm_head.weight")
            .expect("lm_head.weight missing");
        assert_eq!(
            lm.layout,
            LayoutTag::Int4Quantized,
            "lm_head should be INT4 in this bake"
        );
        assert_eq!(lm.shape.len(), 2, "lm_head shape should be 2D");
        assert_eq!(
            lm.shape[1],
            2048 / 2,
            "lm_head INT4 column count = hidden/2 (2 nibbles per byte)"
        );
        assert!(
            store.contains("lm_head.weight_int4_scale"),
            "lm_head.weight_int4_scale missing"
        );
        assert!(
            store.contains("lm_head.weight_int4_zero"),
            "lm_head.weight_int4_zero missing"
        );

        // 2. Per-layer MoE expert presence. The bake must have all 40 layers
        //    of fused expert weight, each with packed nibbles + scale + zero.
        for li in 0..40 {
            let lp = format!("model.language_model.layers.{li}.mlp.experts");
            for kind in ["gate_up_proj", "down_proj"] {
                let base = format!("{lp}.{kind}");
                let scale = format!("{base}_int4_scale");
                let zero = format!("{base}_int4_zero");
                assert!(store.contains(&base), "missing fused expert tensor: {base}");
                assert!(store.contains(&scale), "missing scale sidecar: {scale}");
                assert!(store.contains(&zero), "missing zero sidecar: {zero}");
                let m = store.meta(&base).unwrap();
                assert_eq!(
                    m.layout,
                    LayoutTag::Int4Quantized,
                    "{base} should be Int4Quantized"
                );
                assert_eq!(m.shape.len(), 3, "{base} should be 3D [E, rows, cols/2]");
                assert_eq!(m.shape[0], 256, "{base} num_experts must be 256");
            }
        }

        // 3. Norm + gate raw tensors per layer.
        for li in 0..40 {
            let lp = format!("model.language_model.layers.{li}");
            for n in [
                format!("{lp}.input_layernorm.weight"),
                format!("{lp}.post_attention_layernorm.weight"),
                format!("{lp}.mlp.gate.weight"),
                format!("{lp}.mlp.shared_expert_gate.weight"),
            ] {
                let m = store
                    .meta(&n)
                    .unwrap_or_else(|| panic!("missing raw tensor: {n}"));
                assert_eq!(m.layout, LayoutTag::Raw, "{n} should be Raw layout");
                assert_eq!(m.dtype, "bf16", "{n} should be bf16");
            }
        }

        // 4. Each tensor's [offset, offset+byte_len) must lie strictly
        //    within weights.bin and never exceed it. Catches an
        //    integer-overflow / off-by-one in the writer.
        // 4a. Pull the file size via a normal stat to avoid relying on the
        //     internal mmap accessor.
        let weights_path = crate::weights_bin_path(bake_dir);
        let weights_len = std::fs::metadata(&weights_path)
            .expect("stat weights.bin")
            .len();
        for (name, _) in store.index.iter().take(20) {
            let m = store.meta(name).unwrap();
            let end = m.offset + m.byte_len;
            assert!(
                end <= weights_len,
                "{name}: offset+len {end} > weights.bin len {weights_len}"
            );
            // raw_bytes() should succeed and have the right length.
            let bytes = store
                .raw_bytes(name)
                .unwrap_or_else(|| panic!("raw_bytes returned None for {name}"));
            assert_eq!(
                bytes.len() as u64,
                m.byte_len,
                "{name}: raw_bytes length disagrees with manifest"
            );
        }

        // 5. Quick bake-quality smoke: lm_head's INT4 scale must not be all
        //    zero. A run that died mid-quant could leave us with a stub
        //    scale tensor that would silently produce zero logits.
        let scale_bytes = store
            .raw_bytes("lm_head.weight_int4_scale")
            .expect("lm_head scale bytes");
        let nonzero = scale_bytes.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > scale_bytes.len() / 4,
            "lm_head scale looks suspicious: {nonzero}/{} bytes nonzero",
            scale_bytes.len()
        );

        eprintln!(
            "[bake-validate] OK — {} tensors, weights.bin {} MiB",
            store.index.len(),
            weights_len / (1024 * 1024),
        );
    }

    #[test]
    fn virtual_arena_loads_baked_weight_and_expert_tensors() {
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        if !gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        let weights = (0..8192)
            .map(|idx| (idx as u8).wrapping_mul(13).wrapping_add(7))
            .collect::<Vec<_>>();
        std::fs::write(crate::weights_bin_path(bake_dir), &weights).expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            tensors: vec![
                TensorMeta {
                    name: "lm_head.weight".to_string(),
                    shape: vec![4096],
                    dtype: "u8".to_string(),
                    layout: LayoutTag::Int4Quantized,
                    offset: 0,
                    byte_len: 4096,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
                    shape: vec![4096],
                    dtype: "u8".to_string(),
                    layout: LayoutTag::Int4Quantized,
                    offset: 4096,
                    byte_len: 4096,
                },
            ],
        };
        std::fs::write(
            crate::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let store = BakedStore::open(bake_dir).expect("open synthetic bake");
        let mut arena = BakedStore::virtual_weight_arena(0);
        let weight_id = store
            .load_to_virtual_arena(
                &mut arena,
                "lm_head.weight",
                gpu_hal::VirtualAllocationRole::Weights,
            )
            .expect("load virtual weight");
        let expert_id = store
            .load_to_virtual_arena(
                &mut arena,
                "model.layers.0.mlp.experts.gate_up_proj",
                gpu_hal::VirtualAllocationRole::MoeExpert,
            )
            .expect("load virtual expert");

        let stats = arena.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.logical_bytes, 8192);
        assert_eq!(stats.logical_resident_bytes, 8192);
        assert!(stats.resident_bytes >= stats.logical_resident_bytes);

        let weight = arena
            .allocation(weight_id)
            .expect("weight allocation")
            .buffer()
            .to_host_bytes()
            .expect("read virtual weight");
        let expert = arena
            .allocation(expert_id)
            .expect("expert allocation")
            .buffer()
            .to_host_bytes()
            .expect("read virtual expert");
        assert_eq!(weight, weights[..4096]);
        assert_eq!(expert, weights[4096..]);
    }

    #[test]
    fn virtual_arena_rejects_manifest_byte_len_mismatch() {
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        if !gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        std::fs::write(crate::weights_bin_path(bake_dir), vec![0u8; 4]).expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            tensors: vec![TensorMeta {
                name: "bad.weight".to_string(),
                shape: vec![2],
                dtype: "bf16".to_string(),
                layout: LayoutTag::Raw,
                offset: 0,
                byte_len: 2,
            }],
        };
        std::fs::write(
            crate::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let store = BakedStore::open(bake_dir).expect("open synthetic bake");
        let mut arena = BakedStore::virtual_weight_arena(0);
        let err = store
            .load_to_virtual_arena(
                &mut arena,
                "bad.weight",
                gpu_hal::VirtualAllocationRole::Weights,
            )
            .expect_err("byte length mismatch should be rejected");
        assert!(
            err.to_string().contains("does not match"),
            "unexpected error: {err}"
        );
        assert_eq!(arena.stats().allocations, 0);
    }
}
