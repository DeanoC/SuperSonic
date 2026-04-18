use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use crate::backend::{current_backend, Backend, DeviceInfo};
#[cfg(supersonic_backend_cuda)]
use crate::cuda_sys::*;
use crate::error::{cuda_error, hip_error, GpuError, Result};
#[cfg(supersonic_backend_hip)]
use crate::hip_sys::*;
use crate::scalar_type::ScalarType;

fn with_device_impl<T>(
    backend: Backend,
    ordinal: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let mut prev = 0;
    match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            {
                let status = unsafe { hipGetDevice(&mut prev) };
                if status != 0 {
                    return Err(hip_error("hipGetDevice", status));
                }
            }
            #[cfg(not(supersonic_backend_hip))]
            return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let status = unsafe { cudaGetDevice(&mut prev) };
                if status != 0 {
                    return Err(cuda_error("cudaGetDevice", status));
                }
            }
            #[cfg(not(supersonic_backend_cuda))]
            return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
        }
    }
    let restore = if prev != ordinal_i32 {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipSetDevice(ordinal_i32)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaSetDevice(ordinal_i32)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipSetDevice", status),
                Backend::Cuda => cuda_error("cudaSetDevice", status),
            });
        }
        Some(prev)
    } else {
        None
    };
    let result = f();
    if let Some(prev) = restore {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipSetDevice(prev)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaSetDevice(prev)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipSetDevice(restore)", status),
                Backend::Cuda => cuda_error("cudaSetDevice(restore)", status),
            });
        }
    }
    result
}

pub fn set_device(ordinal: usize) -> Result<()> {
    let backend = current_backend();
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                hipSetDevice(ordinal_i32)
            }
            #[cfg(not(supersonic_backend_hip))]
            1
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                cudaSetDevice(ordinal_i32)
            }
            #[cfg(not(supersonic_backend_cuda))]
            1
        }
    };
    if status != 0 {
        return Err(match backend {
            Backend::Hip => hip_error("hipSetDevice", status),
            Backend::Cuda => cuda_error("cudaSetDevice", status),
        });
    }
    Ok(())
}

/// Allocate `len_bytes` of device memory, returning a non-null pointer.
pub fn alloc(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    let backend = current_backend();
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg("allocation size must be > 0".into()));
    }
    with_device_impl(backend, ordinal, || {
        let mut ptr = std::ptr::null_mut();
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipMalloc(&mut ptr, len_bytes)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaMalloc(&mut ptr, len_bytes)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMalloc", status),
                Backend::Cuda => cuda_error("cudaMalloc", status),
            });
        }
        NonNull::new(ptr).ok_or_else(|| match backend {
            Backend::Hip => GpuError::Hip("hipMalloc returned null".into()),
            Backend::Cuda => GpuError::Cuda("cudaMalloc returned null".into()),
        })
    })
}

/// Allocate `len_bytes` of device memory, zeroed.
pub fn alloc_zeros(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    let ptr = alloc(ordinal, len_bytes)?;
    memset_zeros(ordinal, ptr.as_ptr(), len_bytes)?;
    Ok(ptr)
}

/// Free device memory. No-op on null.
pub fn free(backend: Backend, ordinal: usize, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let _ = with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipFree(ptr)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaFree(ptr)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipFree", status),
                Backend::Cuda => cuda_error("cudaFree", status),
            });
        }
        Ok(())
    });
}

/// Copy from host memory to device memory.
pub fn copy_h2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    let backend = current_backend();
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_h2d: null pointer or zero len".into()));
    }
    with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipMemcpy(dst, src, len, HIP_MEMCPY_HOST_TO_DEVICE)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaMemcpy(dst, src, len, CUDA_MEMCPY_HOST_TO_DEVICE)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(H2D)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(H2D)", status),
            });
        }
        Ok(())
    })
}

/// Copy from device memory to host memory.
pub fn copy_d2h(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    let backend = current_backend();
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_d2h: null pointer or zero len".into()));
    }
    with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_HOST)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaMemcpy(dst, src, len, CUDA_MEMCPY_DEVICE_TO_HOST)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(D2H)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(D2H)", status),
            });
        }
        Ok(())
    })
}

/// Copy from device memory to device memory.
pub fn copy_d2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    let backend = current_backend();
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_d2d: null pointer or zero len".into()));
    }
    with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_DEVICE)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaMemcpy(dst, src, len, CUDA_MEMCPY_DEVICE_TO_DEVICE)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(D2D)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(D2D)", status),
            });
        }
        Ok(())
    })
}

/// Set device memory to zero.
pub fn memset_zeros(ordinal: usize, dst: *mut c_void, len: usize) -> Result<()> {
    let backend = current_backend();
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("memset_zeros: null pointer or zero len".into()));
    }
    with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipMemset(dst, 0, len)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaMemset(dst, 0, len)
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemset", status),
                Backend::Cuda => cuda_error("cudaMemset", status),
            });
        }
        Ok(())
    })
}

/// Synchronize the device (block until all pending work completes).
pub fn sync(ordinal: usize) -> Result<()> {
    let backend = current_backend();
    with_device_impl(backend, ordinal, || {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipDeviceSynchronize()
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(supersonic_backend_cuda)]
                unsafe {
                    cudaDeviceSynchronize()
                }
                #[cfg(not(supersonic_backend_cuda))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipDeviceSynchronize", status),
                Backend::Cuda => cuda_error("cudaDeviceSynchronize", status),
            });
        }
        Ok(())
    })
}

pub fn query_device_info(backend: Backend, ordinal: usize) -> Result<DeviceInfo> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    match backend {
        Backend::Hip => Err(GpuError::InvalidArg(
            "HIP device query is provided by the HIP kernel bridge, not gpu-hal".into(),
        )),
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let mut props = unsafe { std::mem::zeroed::<CudaDeviceProp>() };
                let status = unsafe { cudaGetDeviceProperties(&mut props, ordinal_i32) };
                if status != 0 {
                    return Err(cuda_error("cudaGetDeviceProperties", status));
                }
                let arch_name = format!("sm{}{}", props.major, props.minor);
                Ok(DeviceInfo {
                    arch_name,
                    total_vram_bytes: props.totalGlobalMem as u64,
                    warp_size: props.warpSize as u32,
                })
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    }
}

/// Dtype-aware element count from shape.
pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Byte size for a given dtype and element count.
pub fn byte_len(dtype: ScalarType, elems: usize) -> usize {
    elems * dtype.size_in_bytes()
}
