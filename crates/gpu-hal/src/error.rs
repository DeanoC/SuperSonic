use std::fmt;

#[derive(Debug)]
pub enum GpuError {
    Hip(String),
    InvalidArg(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(msg) => write!(f, "HIP error: {msg}"),
            Self::InvalidArg(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

pub(crate) fn hip_error(op: &str, status: i32) -> GpuError {
    GpuError::Hip(format!("{op} failed with status {status}"))
}
