use crate::gpu::GpuProfileError;
use crate::schema::GpuProfile;

pub struct MetalProfiler;
impl crate::gpu::GpuProfiler for MetalProfiler {
    fn profile(&self) -> Result<Vec<GpuProfile>, GpuProfileError> {
        Err(GpuProfileError::NotImplemented("Metal"))
    }
}
