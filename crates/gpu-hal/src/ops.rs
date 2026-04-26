use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::backend::{current_backend, Backend, DeviceInfo};
#[cfg(supersonic_backend_cuda)]
use crate::cuda_sys::*;
use crate::error::{cuda_error, hip_error, metal_error, GpuError, Result};
#[cfg(supersonic_backend_hip)]
use crate::hip_sys::*;
#[cfg(supersonic_backend_metal)]
use crate::metal_sys::*;
use crate::scalar_type::ScalarType;

static HAL_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static HAL_PROFILE: OnceLock<Mutex<HalProfileAccumulator>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct HalProfileEntry {
    pub op: String,
    pub calls: u64,
    pub total_ms: f64,
    pub max_ms: f64,
    pub total_bytes: u64,
}

impl HalProfileEntry {
    pub fn mean_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ms / self.calls as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HalProfileSnapshot {
    pub total_calls: u64,
    pub total_ms: f64,
    pub alloc_calls: u64,
    pub alloc_bytes: u64,
    pub free_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub d2d_bytes: u64,
    pub memset_bytes: u64,
    pub sync_calls: u64,
    pub entries: Vec<HalProfileEntry>,
}

#[derive(Debug, Default)]
struct HalProfileAccumulator {
    entries: BTreeMap<String, HalProfileEntry>,
}

pub fn hal_profile_set_enabled(enabled: bool) {
    HAL_PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn hal_profile_enabled() -> bool {
    HAL_PROFILE_ENABLED.load(Ordering::Relaxed)
        || std::env::var_os("SUPERSONIC_HAL_PROFILE").is_some()
}

pub fn hal_profile_reset() {
    if let Some(profile) = HAL_PROFILE.get() {
        profile
            .lock()
            .expect("HAL profile mutex poisoned")
            .entries
            .clear();
    }
}

pub fn hal_profile_snapshot() -> HalProfileSnapshot {
    let mut snapshot = HalProfileSnapshot::default();
    let Some(profile) = HAL_PROFILE.get() else {
        return snapshot;
    };
    let mut entries: Vec<_> = profile
        .lock()
        .expect("HAL profile mutex poisoned")
        .entries
        .values()
        .cloned()
        .collect();
    entries.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.op.cmp(&rhs.op))
    });
    for entry in &entries {
        snapshot.total_calls += entry.calls;
        snapshot.total_ms += entry.total_ms;
        match entry.op.as_str() {
            "alloc" => {
                snapshot.alloc_calls += entry.calls;
                snapshot.alloc_bytes += entry.total_bytes;
            }
            "free" => {
                snapshot.free_calls += entry.calls;
            }
            "copy_h2d" => snapshot.h2d_bytes += entry.total_bytes,
            "copy_d2h" => snapshot.d2h_bytes += entry.total_bytes,
            "copy_d2d" => snapshot.d2d_bytes += entry.total_bytes,
            "memset_zeros" => snapshot.memset_bytes += entry.total_bytes,
            "sync" => snapshot.sync_calls += entry.calls,
            _ => {}
        }
    }
    snapshot.entries = entries;
    snapshot
}

fn hal_profile_time<T, F>(op: &'static str, bytes: usize, f: F) -> T
where
    F: FnOnce() -> T,
{
    if !hal_profile_enabled() {
        return f();
    }
    let start = Instant::now();
    let result = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let profile = HAL_PROFILE.get_or_init(|| Mutex::new(HalProfileAccumulator::default()));
    let mut profile = profile.lock().expect("HAL profile mutex poisoned");
    let entry = profile
        .entries
        .entry(op.to_string())
        .or_insert_with(|| HalProfileEntry {
            op: op.to_string(),
            calls: 0,
            total_ms: 0.0,
            max_ms: 0.0,
            total_bytes: 0,
        });
    entry.calls += 1;
    entry.total_ms += elapsed_ms;
    entry.max_ms = entry.max_ms.max(elapsed_ms);
    entry.total_bytes += bytes as u64;
    result
}

fn with_device_impl<T>(
    backend: Backend,
    ordinal: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    #[allow(unused_mut)]
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
        let _ = prev;
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
    let _ = ordinal_i32;
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
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg("allocation size must be > 0".into()));
    }
    hal_profile_time("alloc", len_bytes, || {
        let backend = current_backend();
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
    })
}

/// Allocate host memory suitable for fast host-to-device page-in.
pub fn alloc_host_pinned(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    let backend = current_backend();
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg(
            "host allocation size must be > 0".into(),
        ));
    }
    match backend {
        Backend::Cuda => with_device_impl(backend, ordinal, || {
            #[cfg(supersonic_backend_cuda)]
            {
                let mut ptr = std::ptr::null_mut();
                const CUDA_HOST_ALLOC_MAPPED: u32 = 0x02;
                let status = unsafe { cudaHostAlloc(&mut ptr, len_bytes, CUDA_HOST_ALLOC_MAPPED) };
                if status != 0 {
                    return Err(cuda_error("cudaHostAlloc", status));
                }
                NonNull::new(ptr)
                    .ok_or_else(|| GpuError::Cuda("cudaHostAlloc returned null".into()))
            }
            #[cfg(not(supersonic_backend_cuda))]
            Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
        }),
        Backend::Hip => with_device_impl(backend, ordinal, || {
            #[cfg(supersonic_backend_hip)]
            {
                let mut ptr = std::ptr::null_mut();
                let status = unsafe { hipHostMalloc(&mut ptr, len_bytes, 0) };
                if status != 0 {
                    return Err(hip_error("hipHostMalloc", status));
                }
                NonNull::new(ptr).ok_or_else(|| GpuError::Hip("hipHostMalloc returned null".into()))
            }
            #[cfg(not(supersonic_backend_hip))]
            Err(GpuError::InvalidArg("HIP backend not compiled".into()))
        }),
        Backend::Metal => {
            let layout = Layout::from_size_align(len_bytes, 64)
                .map_err(|e| GpuError::InvalidArg(format!("host allocation layout failed: {e}")))?;
            let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
            NonNull::new(ptr).ok_or_else(|| GpuError::Metal("host allocation returned null".into()))
        }
    }
}

/// Return the device-visible pointer for mapped pinned host memory.
pub fn host_pinned_device_ptr(
    backend: Backend,
    ordinal: usize,
    ptr: *mut c_void,
) -> Result<NonNull<c_void>> {
    if ptr.is_null() {
        return Err(GpuError::InvalidArg(
            "host_pinned_device_ptr: null host pointer".into(),
        ));
    }
    match backend {
        Backend::Cuda => with_device_impl(backend, ordinal, || {
            #[cfg(supersonic_backend_cuda)]
            {
                let mut device_ptr = std::ptr::null_mut();
                let status = unsafe { cudaHostGetDevicePointer(&mut device_ptr, ptr, 0) };
                if status != 0 {
                    return Err(cuda_error("cudaHostGetDevicePointer", status));
                }
                NonNull::new(device_ptr)
                    .ok_or_else(|| GpuError::Cuda("cudaHostGetDevicePointer returned null".into()))
            }
            #[cfg(not(supersonic_backend_cuda))]
            Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
        }),
        Backend::Hip | Backend::Metal => NonNull::new(ptr)
            .ok_or_else(|| GpuError::InvalidArg("host_pinned_device_ptr returned null".into())),
    }
}

/// Free host memory allocated by `alloc_host_pinned`.
pub fn free_host_pinned(backend: Backend, ordinal: usize, ptr: *mut c_void, len_bytes: usize) {
    if ptr.is_null() {
        return;
    }
    match backend {
        Backend::Cuda => {
            let _: Result<()> = with_device_impl(backend, ordinal, || {
                #[cfg(supersonic_backend_cuda)]
                {
                    let status = unsafe { cudaFreeHost(ptr) };
                    if status != 0 {
                        return Err(cuda_error("cudaFreeHost", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_cuda))]
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            });
        }
        Backend::Hip => {
            let _: Result<()> = with_device_impl(backend, ordinal, || {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipHostFree(ptr) };
                    if status != 0 {
                        return Err(hip_error("hipHostFree", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            });
        }
        Backend::Metal => {
            if let Ok(layout) = Layout::from_size_align(len_bytes, 64) {
                unsafe { dealloc(ptr as *mut u8, layout) };
            }
        }
    }
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
    hal_profile_time("free", 0, || {
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
    });
}

/// Copy from host memory to device memory.
pub fn copy_h2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_h2d: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_h2d", len, || {
        let backend = current_backend();
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
    })
}

/// Copy from device memory to host memory.
pub fn copy_d2h(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_d2h: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_d2h", len, || {
        let backend = current_backend();
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
    })
}

/// Copy from device memory to device memory.
pub fn copy_d2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_d2d: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_d2d", len, || {
        let backend = current_backend();
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
    })
}

/// Set device memory to zero.
pub fn memset_zeros(ordinal: usize, dst: *mut c_void, len: usize) -> Result<()> {
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "memset_zeros: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("memset_zeros", len, || {
        let backend = current_backend();
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
    })
}

/// Synchronize the device (block until all pending work completes).
pub fn sync(ordinal: usize) -> Result<()> {
    hal_profile_time("sync", 0, || {
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
        #[allow(unused_mut)]
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
    let _ = ordinal_i32;
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
        assert!(
            zeroed.iter().all(|&b| b == 0),
            "destination buffer not zeroed"
        );
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
