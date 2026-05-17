pub mod hip_ffi;

#[cfg(supersonic_backend_hip)]
pub mod hip;

pub mod cuda;
pub mod metal;

use crate::schema::GpuProfile;

#[derive(Debug, thiserror::Error)]
pub enum GpuProfileError {
    #[error("backend not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("hip error: {0}")]
    Hip(String),
    #[error("metal error: {0}")]
    Metal(String),
}

pub trait GpuProfiler {
    fn profile(&self) -> Result<Vec<GpuProfile>, GpuProfileError>;
}

pub fn run_all() -> Vec<GpuProfile> {
    let mut out = Vec::new();
    #[cfg(supersonic_backend_hip)]
    {
        match hip::HipProfiler.profile() {
            Ok(p) => out.extend(p),
            Err(e) => eprintln!("HIP profiler failed: {e}"),
        }
    }
    #[cfg(supersonic_backend_metal)]
    {
        match metal::MetalProfiler.profile() {
            Ok(p) => out.extend(p),
            Err(e) => eprintln!("Metal profiler failed: {e}"),
        }
    }
    out
}
