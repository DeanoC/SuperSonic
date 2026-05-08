use std::ffi::c_void;
use std::mem;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen3_moe::{Qwen3MoeDecodeLayerDesc, Qwen3MoeInt4ScaleDesc};

/// Scratch/upload buffers for the Qwen3-MoE persistent decode path.
///
/// The current HIP kernel surface is a descriptor-walk stub, but these buffers
/// are sized for the eventual one-token decode kernel: F32 workspace,
/// descriptor arrays, INT4 sidecar descriptors, and a small work-stealing sync
/// region.
pub struct Qwen3MoeScratch {
    ordinal: usize,
    pub workspace: GpuBuffer,
    pub sync_buf: GpuBuffer,
    pub desc_device: GpuBuffer,
    desc_capacity_bytes: usize,
    pub int4_desc_device: GpuBuffer,
    int4_desc_capacity_bytes: usize,
}

impl Qwen3MoeScratch {
    pub fn new(
        ordinal: usize,
        num_layers: usize,
        workspace_floats: usize,
    ) -> Result<Self, GpuError> {
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats.max(4)])?;
        let sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])?;
        let desc_bytes = num_layers * mem::size_of::<Qwen3MoeDecodeLayerDesc>();
        let desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[desc_bytes.max(1)])?;
        let int4_desc_bytes = num_layers * mem::size_of::<Qwen3MoeInt4ScaleDesc>();
        let int4_desc_device =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[int4_desc_bytes.max(1)])?;
        Ok(Self {
            ordinal,
            workspace,
            sync_buf,
            desc_device,
            desc_capacity_bytes: desc_bytes,
            int4_desc_device,
            int4_desc_capacity_bytes: int4_desc_bytes,
        })
    }

    pub fn upload_descs(&mut self, descs: &[Qwen3MoeDecodeLayerDesc]) -> Result<(), GpuError> {
        let bytes = descs.len() * mem::size_of::<Qwen3MoeDecodeLayerDesc>();
        if bytes > self.desc_capacity_bytes {
            self.desc_device = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &[bytes])?;
            self.desc_capacity_bytes = bytes;
        }
        gpu_hal::copy_h2d(
            self.ordinal,
            self.desc_device.as_mut_ptr() as *mut c_void,
            descs.as_ptr() as *const c_void,
            bytes,
        )
    }

    pub fn upload_int4_descs(&mut self, descs: &[Qwen3MoeInt4ScaleDesc]) -> Result<(), GpuError> {
        let bytes = descs.len() * mem::size_of::<Qwen3MoeInt4ScaleDesc>();
        if bytes > self.int4_desc_capacity_bytes {
            self.int4_desc_device = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &[bytes])?;
            self.int4_desc_capacity_bytes = bytes;
        }
        gpu_hal::copy_h2d(
            self.ordinal,
            self.int4_desc_device.as_mut_ptr() as *mut c_void,
            descs.as_ptr() as *const c_void,
            bytes,
        )
    }

    pub fn reset_sync(&mut self) -> Result<(), GpuError> {
        gpu_hal::memset_zeros(
            self.ordinal,
            self.sync_buf.as_mut_ptr(),
            self.sync_buf.len_bytes(),
        )
    }
}
