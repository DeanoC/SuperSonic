use std::fmt;

use crate::backend::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendApi {
    Runtime,
    Driver,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    Backend {
        backend: Backend,
        message: String,
    },
    BackendStatus {
        backend: Backend,
        api: BackendApi,
        operation: String,
        status: i32,
    },
    DeviceLost {
        backend: Backend,
        api: BackendApi,
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
        Self::backend_status_in(backend, BackendApi::Runtime, operation, status)
    }

    pub fn backend_status_in(
        backend: Backend,
        api: BackendApi,
        operation: impl Into<String>,
        status: i32,
    ) -> Self {
        let operation = operation.into();
        if is_device_loss_status(backend, api, status) {
            Self::DeviceLost {
                backend,
                api,
                operation,
                status,
            }
        } else {
            Self::BackendStatus {
                backend,
                api,
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
                api: _,
                operation,
                status,
            }
            | Self::DeviceLost {
                backend,
                api: _,
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

fn is_device_loss_status(backend: Backend, api: BackendApi, status: i32) -> bool {
    let _ = api;
    let _ = backend;
    status == 709
}

#[cfg(test)]
mod tests {
    use super::{Backend, BackendApi, GpuError};

    #[test]
    fn backend_status_preserves_operation_domain_and_maps_device_loss() {
        let hip = GpuError::backend_status_in(
            Backend::Hip,
            BackendApi::Runtime,
            "hipDeviceSynchronize",
            709,
        );
        assert!(matches!(
            hip,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                api: BackendApi::Runtime,
                ref operation,
                status: 709,
            } if operation == "hipDeviceSynchronize"
        ));

        let ordinary = GpuError::backend_status(Backend::Hip, "hipMalloc", 2);
        assert!(matches!(
            ordinary,
            GpuError::BackendStatus {
                backend: Backend::Hip,
                api: BackendApi::Runtime,
                ref operation,
                status: 2,
            } if operation == "hipMalloc"
        ));
    }
}
