use std::ffi::c_void;
use std::mem;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use kernel_ffi::DecodeLayerDesc;

pub const PERSISTENT_4B_TIMING_SLOTS_PER_LAYER: usize = 43;
pub const PERSISTENT_SYNC_COUNTER_BYTES: usize = 24;
/// Required floats in the kernel's `attn_scratch` region for a given model
/// and context ceiling. The 4B persistent decode kernel lays out
/// `saved_q [nh*hd] + saved_gate [nh*hd] + saved_pre_gate [nh*hd] +
/// saved_scores [nh*kv_max_t]` per batch item. The caller must size the
/// workspace so the largest `kv_max_t` reached during the run fits.
pub fn required_attn_scratch_floats(
    num_attention_heads: usize,
    head_dim: usize,
    max_context_tokens: usize,
    kv_chunk_size: usize,
) -> usize {
    let aligned_kv_t = max_context_tokens.div_ceil(kv_chunk_size.max(1)) * kv_chunk_size.max(1);
    3 * num_attention_heads * head_dim + num_attention_heads * aligned_kv_t
}

/// Pre-allocated device scratch buffers for the persistent decode kernel.
/// Avoids per-token hipMalloc/hipFree overhead.
pub struct PersistentDecodeScratch {
    ordinal: usize,
    /// F32 workspace for projections, MLP, attention scratch.
    pub workspace: GpuBuffer,
    /// Sync region: counters[4×u32=16B] + barrier_counter[u32=4B] +
    /// barrier_flag[u32=4B] plus per-layer persistent 4B timing slots.
    pub sync_buf: GpuBuffer,
    /// Device copy of Vec<DecodeLayerDesc>.
    pub desc_device: GpuBuffer,
    desc_capacity_bytes: usize,
}

impl PersistentDecodeScratch {
    pub fn new(
        ordinal: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
    ) -> Result<Self, GpuError> {
        // Workspace layout matches the kernel expectation.
        let workspace_floats = hidden_dim
            + hidden_dim
            + intermediate_size * 2
            + hidden_dim
            + hidden_dim
            + attn_scratch_floats
            + proj_buf_floats;
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])?;

        let sync_bytes = PERSISTENT_SYNC_COUNTER_BYTES
            + num_layers * PERSISTENT_4B_TIMING_SLOTS_PER_LAYER * std::mem::size_of::<u64>();
        let sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[sync_bytes])?;

        let desc_bytes = num_layers * mem::size_of::<DecodeLayerDesc>();
        let desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[desc_bytes])?;

        Ok(Self {
            ordinal,
            workspace,
            sync_buf,
            desc_device,
            desc_capacity_bytes: desc_bytes,
        })
    }

    /// Upload layer descriptors to device memory.
    pub fn upload_descs(&mut self, descs: &[DecodeLayerDesc]) -> Result<(), GpuError> {
        let bytes = descs.len() * mem::size_of::<DecodeLayerDesc>();
        if bytes > self.desc_capacity_bytes {
            self.desc_device = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &[bytes])?;
            self.desc_capacity_bytes = bytes;
        }
        gpu_hal::copy_h2d(
            self.ordinal,
            self.desc_device.as_ptr() as *mut c_void,
            descs.as_ptr() as *const c_void,
            bytes,
        )
    }

    /// Reset sync counters to zero (needed before first kernel launch of a sequence).
    pub fn reset_sync(&mut self) -> Result<(), GpuError> {
        gpu_hal::memset_zeros(
            self.ordinal,
            self.sync_buf.as_mut_ptr(),
            self.sync_buf.len_bytes(),
        )
    }
}
