pub mod hip_ffi;

#[cfg(supersonic_backend_hip)]
pub mod hip;

#[derive(Debug, thiserror::Error)]
pub enum GpuProfileError {
    #[error("backend not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("hip error: {0}")]
    Hip(String),
}
