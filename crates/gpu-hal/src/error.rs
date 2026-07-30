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

#[cfg(supersonic_backend_cuda)]
pub(crate) fn backend_driver_error(backend: Backend, op: &str, status: i32) -> GpuError {
    GpuError::backend_status_in(backend, BackendApi::Driver, op, status)
}

fn is_device_loss_status(backend: Backend, api: BackendApi, status: i32) -> bool {
    match backend {
        Backend::Hip => status == 709,
        Backend::Cuda => match api {
            BackendApi::Runtime => {
                matches!(
                    status,
                    226 | 700 | 702 | 709 | 710 | 714..=719 | 721 | 810 | 911
                )
            }
            BackendApi::Driver => {
                matches!(
                    status,
                    226 | 700 | 702 | 709 | 710 | 714..=719 | 721 | 810 | 911
                )
            }
        },
        Backend::Metal => false,
    }
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

        for api in [BackendApi::Runtime, BackendApi::Driver] {
            let cuda = GpuError::backend_status_in(Backend::Cuda, api, "cuda operation", 709);
            assert!(matches!(
                cuda,
                GpuError::DeviceLost {
                    backend: Backend::Cuda,
                    api: actual,
                    ref operation,
                    status: 709,
                } if actual == api && operation == "cuda operation"
            ));
        }

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

    #[test]
    fn cuda_unavailable_status_is_request_local_in_both_api_domains() {
        for api in [BackendApi::Runtime, BackendApi::Driver] {
            let error = GpuError::backend_status_in(Backend::Cuda, api, "availability", 46);
            assert!(matches!(
                error,
                GpuError::BackendStatus {
                    backend: Backend::Cuda,
                    api: actual,
                    status: 46,
                    ..
                } if actual == api
            ));
        }
    }

    #[test]
    fn cuda_fatal_statuses_are_integrity_loss_in_their_native_domain() {
        for status in [
            226, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 721, 810, 911,
        ] {
            let error = GpuError::backend_status_in(
                Backend::Cuda,
                BackendApi::Runtime,
                "runtime fatal operation",
                status,
            );
            assert!(
                error.is_device_lost(),
                "runtime status {status} must make the CUDA process unusable"
            );
        }

        for status in [
            226, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 721, 810, 911,
        ] {
            let error = GpuError::backend_status_in(
                Backend::Cuda,
                BackendApi::Driver,
                "driver fatal operation",
                status,
            );
            assert!(
                error.is_device_lost(),
                "driver status {status} must make the CUDA context unusable"
            );
        }
    }
}
