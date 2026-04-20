use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use crate::backend::{current_backend, Backend, DeviceInfo};
#[cfg(supersonic_backend_cuda)]
use crate::cuda_sys::*;
use crate::error::{cuda_error, hip_error, metal_error, GpuError, Result};
#[cfg(supersonic_backend_hip)]
use crate::hip_sys::*;
#[cfg(supersonic_backend_metal)]
use crate::metal_sys::*;
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
        Backend::Metal => {
            if ordinal != 0 {
                return Err(GpuError::InvalidArg(
                    "Metal backend currently supports only device ordinal 0".into(),
                ));
            }
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
            Backend::Metal => 0,
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipSetDevice", status),
                Backend::Cuda => cuda_error("cudaSetDevice", status),
                Backend::Metal => metal_error("metalSetDevice", status),
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
            Backend::Metal => 0,
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipSetDevice(restore)", status),
                Backend::Cuda => cuda_error("cudaSetDevice(restore)", status),
                Backend::Metal => metal_error("metalSetDevice(restore)", status),
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
        Backend::Metal => {
            if ordinal == 0 {
                0
            } else {
                1
            }
        }
    };
    if status != 0 {
        return Err(match backend {
            Backend::Hip => hip_error("hipSetDevice", status),
            Backend::Cuda => cuda_error("cudaSetDevice", status),
            Backend::Metal => GpuError::InvalidArg(
                "Metal backend currently supports only device ordinal 0".into(),
            ),
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
            Backend::Metal => {
                #[cfg(supersonic_backend_metal)]
                unsafe {
                    supersonic_metal_alloc(len_bytes, &mut ptr)
                }
                #[cfg(not(supersonic_backend_metal))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMalloc", status),
                Backend::Cuda => cuda_error("cudaMalloc", status),
                Backend::Metal => metal_error("metalAlloc", status),
            });
        }
        NonNull::new(ptr).ok_or_else(|| match backend {
            Backend::Hip => GpuError::Hip("hipMalloc returned null".into()),
            Backend::Cuda => GpuError::Cuda("cudaMalloc returned null".into()),
            Backend::Metal => GpuError::Metal("metalAlloc returned null".into()),
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
            Backend::Metal => {
                #[cfg(supersonic_backend_metal)]
                unsafe {
                    supersonic_metal_free(ptr)
                }
                #[cfg(not(supersonic_backend_metal))]
                1
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipFree", status),
                Backend::Cuda => cuda_error("cudaFree", status),
                Backend::Metal => metal_error("metalFree", status),
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
            Backend::Metal => {
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
                }
                0
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(H2D)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(H2D)", status),
                Backend::Metal => metal_error("metalMemcpy(H2D)", status),
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
            Backend::Metal => {
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
                }
                0
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(D2H)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(D2H)", status),
                Backend::Metal => metal_error("metalMemcpy(D2H)", status),
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
            Backend::Metal => {
                unsafe {
                    std::ptr::copy(src as *const u8, dst as *mut u8, len);
                }
                0
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemcpy(D2D)", status),
                Backend::Cuda => cuda_error("cudaMemcpy(D2D)", status),
                Backend::Metal => metal_error("metalMemcpy(D2D)", status),
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
            Backend::Metal => {
                unsafe {
                    std::ptr::write_bytes(dst as *mut u8, 0, len);
                }
                0
            }
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipMemset", status),
                Backend::Cuda => cuda_error("cudaMemset", status),
                Backend::Metal => metal_error("metalMemset", status),
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
            Backend::Metal => 0,
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => hip_error("hipDeviceSynchronize", status),
                Backend::Cuda => cuda_error("cudaDeviceSynchronize", status),
                Backend::Metal => metal_error("metalDeviceSynchronize", status),
            });
        }
        Ok(())
    })
}

/// RAII wrapper around a backend timing event.
///
/// Timing events are currently implemented only for HIP. On CUDA builds this
/// returns an explicit error until the matching runtime bindings are added.
pub struct GpuEvent {
    backend: Backend,
    ordinal: usize,
    raw: *mut c_void,
}

impl GpuEvent {
    pub fn new(ordinal: usize) -> Result<Self> {
        let backend = current_backend();
        let mut raw: *mut c_void = std::ptr::null_mut();
        with_device_impl(backend, ordinal, || match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventCreate(&mut raw) };
                    if status != 0 {
                        return Err(hip_error("hipEventCreate", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })?;
        Ok(Self {
            backend,
            ordinal,
            raw,
        })
    }

    pub fn record(&self) -> Result<()> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventRecord(self.raw, std::ptr::null_mut()) };
                    if status != 0 {
                        return Err(hip_error("hipEventRecord", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventSynchronize(self.raw) };
                    if status != 0 {
                        return Err(hip_error("hipEventSynchronize", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn elapsed_ms(start: &GpuEvent, end: &GpuEvent) -> Result<f32> {
        if start.backend != end.backend || start.ordinal != end.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuEvent::elapsed_ms requires matching backend/device".into(),
            ));
        }
        match start.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let mut ms: f32 = 0.0;
                    with_device_impl(start.backend, start.ordinal, || {
                        let status = unsafe { hipEventElapsedTime(&mut ms, start.raw, end.raw) };
                        if status != 0 {
                            return Err(hip_error("hipEventElapsedTime", status));
                        }
                        Ok(())
                    })?;
                    Ok(ms)
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        }
    }
}

impl Drop for GpuEvent {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let _ = with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventDestroy(self.raw) };
                    if status != 0 {
                        return Err(hip_error("hipEventDestroy", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Ok(()),
            Backend::Metal => Ok(()),
        });
    }
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
                    clock_rate_khz: props.clockRate as u32,
                })
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Metal => {
            #[cfg(supersonic_backend_metal)]
            {
                let mut arch_name = vec![0i8; 64];
                let mut total_vram_bytes = 0u64;
                let mut warp_size = 0u32;
                let mut clock_rate_khz = 0u32;
                let status = unsafe {
                    supersonic_metal_query_device_info(
                        ordinal,
                        arch_name.as_mut_ptr(),
                        arch_name.len(),
                        &mut total_vram_bytes,
                        &mut warp_size,
                        &mut clock_rate_khz,
                    )
                };
                if status != 0 {
                    return Err(metal_error("metalQueryDeviceInfo", status));
                }
                let nul_pos = arch_name
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(arch_name.len());
                let arch_name = String::from_utf8_lossy(
                    &arch_name[..nul_pos]
                        .iter()
                        .map(|&c| c as u8)
                        .collect::<Vec<_>>(),
                )
                .to_string();
                Ok(DeviceInfo {
                    arch_name,
                    total_vram_bytes,
                    warp_size,
                    clock_rate_khz,
                })
            }
            #[cfg(not(supersonic_backend_metal))]
            {
                Err(GpuError::InvalidArg("Metal backend not compiled".into()))
            }
        }
    }
}

#[cfg(supersonic_backend_metal)]
fn metal_runtime_compile_smoke() -> Result<()> {
    let status = unsafe { supersonic_metal_compile_shader_smoke() };
    if status != 0 {
        return Err(metal_error("metalCompileShaderSmoke", status));
    }
    Ok(())
}

/// Dtype-aware element count from shape.
pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Byte size for a given dtype and element count.
pub fn byte_len(dtype: ScalarType, elems: usize) -> usize {
    elems * dtype.size_in_bytes()
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use crate::{set_backend, Backend, GpuBuffer, ScalarType};

    fn use_metal_backend() {
        set_backend(Backend::Metal);
    }

    #[test]
    fn metal_device_info_reports_expected_shape() {
        use_metal_backend();
        let info = query_device_info(Backend::Metal, 0).expect("query metal device info");
        assert!(
            info.arch_name.contains("apple"),
            "unexpected metal arch name: {}",
            info.arch_name
        );
        assert!(info.total_vram_bytes > 0, "missing working set budget");
        assert_eq!(info.warp_size, 32);
    }

    #[test]
    fn metal_buffer_round_trip_copy_zero_and_sync() {
        use_metal_backend();
        let ordinal = 0usize;
        let host = [1.0f32, -2.5, 3.25, 4.5];
        let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let src = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[host.len()], &host_bytes)
            .expect("upload source buffer");
        let mut dst = GpuBuffer::zeros(ordinal, ScalarType::F32, &[host.len()])
            .expect("allocate zero destination");

        copy_d2d(ordinal, dst.as_mut_ptr(), src.as_ptr(), src.len_bytes()).expect("copy_d2d");
        sync(ordinal).expect("sync after copy_d2d");
        let copied = dst.to_host_bytes().expect("download copied bytes");
        assert_eq!(copied, host_bytes);

        memset_zeros(ordinal, dst.as_mut_ptr(), dst.len_bytes()).expect("memset zeros");
        sync(ordinal).expect("sync after memset");
        let zeroed = dst.to_host_bytes().expect("download zeroed bytes");
        assert!(zeroed.iter().all(|&b| b == 0), "destination buffer not zeroed");
    }

    #[test]
    fn metal_runtime_shader_compile_smoke_succeeds() {
        use_metal_backend();
        metal_runtime_compile_smoke().expect("runtime Metal shader compilation should succeed");
    }

    #[test]
    fn metal_rejects_nonzero_ordinal() {
        use_metal_backend();
        let err = alloc(1, 16).expect_err("metal ordinal 1 should be rejected");
        assert!(
            err.to_string().contains("ordinal 0"),
            "unexpected error: {err}"
        );
    }
}
