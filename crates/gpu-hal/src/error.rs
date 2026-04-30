use std::fmt;

use crate::backend::Backend;

#[derive(Debug)]
pub enum GpuError {
    Backend { backend: Backend, message: String },
    InvalidArg(String),
}

impl GpuError {
    pub fn backend(backend: Backend, message: String) -> Self {
        Self::Backend { backend, message }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend { backend, message } => write!(f, "{backend} error: {message}"),
            Self::InvalidArg(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

pub(crate) fn backend_error(backend: Backend, op: &str, status: i32) -> GpuError {
    GpuError::backend(backend, format!("{op} failed with status {status}"))
}
