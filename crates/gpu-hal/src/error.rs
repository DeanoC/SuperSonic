use std::fmt;

use crate::backend::Backend;

#[derive(Debug)]
pub enum GpuError {
    Backend {
        backend: Backend,
        message: String,
    },
    BackendStatus {
        backend: Backend,
        operation: String,
        status: i32,
    },
    DeviceLost {
        backend: Backend,
        operation: String,
        status: i32,
    },
    InvalidArg(String),
    Unsupported(String),
}

impl GpuError {
    pub fn backend(backend: Backend, message: String) -> Self {
        Self::Backend { backend, message }
    }

    pub fn backend_status(backend: Backend, operation: impl Into<String>, status: i32) -> Self {
        let operation = operation.into();
        if is_device_loss_status(backend, status) {
            Self::DeviceLost {
                backend,
                operation,
                status,
            }
        } else {
            Self::BackendStatus {
                backend,
                operation,
                status,
            }
        }
    }

    pub fn is_device_lost(&self) -> bool {
        matches!(self, Self::DeviceLost { .. })
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend { backend, message } => write!(f, "{backend} error: {message}"),
            Self::BackendStatus {
                backend,
                operation,
                status,
            }
            | Self::DeviceLost {
                backend,
                operation,
                status,
            } => write!(
                f,
                "{backend} error: {operation} failed with status {status}"
            ),
            Self::InvalidArg(msg) => write!(f, "invalid argument: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

pub(crate) fn backend_error(backend: Backend, op: &str, status: i32) -> GpuError {
    GpuError::backend_status(backend, op, status)
}

fn is_device_loss_status(backend: Backend, status: i32) -> bool {
    match backend {
        // HIP names 709 hipErrorContextIsDestroyed. CUDA runtime and driver
        // APIs use 709 for a destroyed/lost context; the driver also exposes
        // 46 as CUDA_ERROR_DEVICE_UNAVAILABLE.
        Backend::Hip => status == 709,
        Backend::Cuda => matches!(status, 46 | 709),
        Backend::Metal => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, GpuError};

    #[test]
    fn backend_status_preserves_operation_and_maps_device_loss() {
        let hip = GpuError::backend_status(Backend::Hip, "hipDeviceSynchronize", 709);
        assert!(matches!(
            hip,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                ref operation,
                status: 709,
            } if operation == "hipDeviceSynchronize"
        ));

        let cuda = GpuError::backend_status(Backend::Cuda, "cudaMemcpy(D2H)", 709);
        assert!(matches!(
            cuda,
            GpuError::DeviceLost {
                backend: Backend::Cuda,
                ref operation,
                status: 709,
            } if operation == "cudaMemcpy(D2H)"
        ));

        let ordinary = GpuError::backend_status(Backend::Hip, "hipMalloc", 2);
        assert!(matches!(
            ordinary,
            GpuError::BackendStatus {
                backend: Backend::Hip,
                ref operation,
                status: 2,
            } if operation == "hipMalloc"
        ));
    }
}
