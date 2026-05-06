use crate::gpu::GpuProfileError;
use crate::schema::GpuProfile;

pub struct CudaProfiler;
impl crate::gpu::GpuProfiler for CudaProfiler {
    fn profile(&self) -> Result<Vec<GpuProfile>, GpuProfileError> {
        Err(GpuProfileError::NotImplemented("CUDA"))
    }
}
