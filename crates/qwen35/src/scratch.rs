use std::ffi::c_void;
use std::mem;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use kernel_ffi::{DecodeLayerDesc, KVCacheFp8Desc, BatchSeqDesc};

pub const PERSISTENT_4B_TIMING_SLOTS_PER_LAYER: usize = 37;
pub const PERSISTENT_SYNC_COUNTER_BYTES: usize = 24;

/// Pre-allocated device scratch buffers for the persistent decode kernel.
/// Avoids per-token hipMalloc/hipFree overhead.
pub struct PersistentDecodeScratch {
    ordinal: usize,
    /// F32 workspace for projections, MLP, attention scratch.
    pub workspace: GpuBuffer,
    /// Sync region: counters[4×u32=16B] + barrier_counter[u32=4B] + barrier_flag[u32=4B]
    /// plus persistent 4B timing slots [6×u64].
    pub sync_buf: GpuBuffer,
    /// Device copy of Vec<DecodeLayerDesc>.
    pub desc_device: GpuBuffer,
    desc_capacity_bytes: usize,
    /// Device copy of Vec<KVCacheFp8Desc> (if FP8 KV enabled).
    pub kv_fp8_desc_device: Option<GpuBuffer>,
    kv_fp8_desc_capacity_bytes: usize,
    /// Device copy of Vec<BatchSeqDesc> (if batch_size > 1).
    pub batch_seq_desc_device: Option<GpuBuffer>,
    batch_seq_desc_capacity_bytes: usize,
}

impl PersistentDecodeScratch {
    pub fn new(
        ordinal: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        attn_scratch_floats: usize,
        saved_gate_floats: usize,
        batch_size: usize,
    ) -> Result<Self, GpuError> {
        // Workspace layout matches the kernel expectation.
        // Each section is multiplied by batch_size for batched decode.
        let b = batch_size;
        let workspace_floats = (hidden_dim
            + hidden_dim
            + intermediate_size * 2
            + hidden_dim
            + hidden_dim
            + attn_scratch_floats
            + saved_gate_floats) * b;
        let workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[workspace_floats],
        )?;

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
            kv_fp8_desc_device: None,
            kv_fp8_desc_capacity_bytes: 0,
            batch_seq_desc_device: None,
            batch_seq_desc_capacity_bytes: 0,
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

    /// Upload KV cache FP8 scale descriptors to device memory.
    /// Must be re-uploaded each step since scale buffer pointers change on KV cache growth.
    pub fn upload_kv_fp8_descs(&mut self, descs: &[KVCacheFp8Desc]) -> Result<(), GpuError> {
        let bytes = descs.len() * mem::size_of::<KVCacheFp8Desc>();
        if bytes > self.kv_fp8_desc_capacity_bytes {
            self.kv_fp8_desc_device =
                Some(GpuBuffer::zeros(self.ordinal, ScalarType::U8, &[bytes])?);
            self.kv_fp8_desc_capacity_bytes = bytes;
        }
        if let Some(ref buf) = self.kv_fp8_desc_device {
            gpu_hal::copy_h2d(
                self.ordinal,
                buf.as_ptr() as *mut c_void,
                descs.as_ptr() as *const c_void,
                bytes,
            )?;
        }
        Ok(())
    }

    /// Upload batch sequence descriptors to device memory.
    /// Must be re-uploaded each step since per-sequence state pointers may change.
    pub fn upload_batch_seq_descs(&mut self, descs: &[BatchSeqDesc]) -> Result<(), GpuError> {
        let bytes = descs.len() * mem::size_of::<BatchSeqDesc>();
        if bytes > self.batch_seq_desc_capacity_bytes {
            self.batch_seq_desc_device =
                Some(GpuBuffer::zeros(self.ordinal, ScalarType::U8, &[bytes])?);
            self.batch_seq_desc_capacity_bytes = bytes;
        }
        if let Some(ref buf) = self.batch_seq_desc_device {
            gpu_hal::copy_h2d(
                self.ordinal,
                buf.as_ptr() as *mut std::ffi::c_void,
                descs.as_ptr() as *const std::ffi::c_void,
                bytes,
            )?;
        }
        Ok(())
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
