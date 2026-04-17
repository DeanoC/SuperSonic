use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use crate::error::{hip_error, GpuError, Result};
use crate::hip_sys::*;
use crate::scalar_type::ScalarType;

/// Set the active HIP device, execute `f`, then restore the previous device.
pub(crate) fn with_device<T>(ordinal: usize, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let mut prev = 0;
    let status = unsafe { hipGetDevice(&mut prev) };
    if status != 0 {
        return Err(hip_error("hipGetDevice", status));
    }
    let restore = if prev != ordinal_i32 {
        let status = unsafe { hipSetDevice(ordinal_i32) };
        if status != 0 {
            return Err(hip_error("hipSetDevice", status));
        }
        Some(prev)
    } else {
        None
    };
    let result = f();
    if let Some(prev) = restore {
        let status = unsafe { hipSetDevice(prev) };
        if status != 0 {
            return Err(hip_error("hipSetDevice(restore)", status));
        }
    }
    result
}

/// Set the active HIP device.
pub fn set_device(ordinal: usize) -> Result<()> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let status = unsafe { hipSetDevice(ordinal_i32) };
    if status != 0 {
        return Err(hip_error("hipSetDevice", status));
    }
    Ok(())
}

/// Allocate `len_bytes` of device memory, returning a non-null pointer.
pub fn alloc(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg("allocation size must be > 0".into()));
    }
    with_device(ordinal, || {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { hipMalloc(&mut ptr, len_bytes) };
        if status != 0 {
            return Err(hip_error("hipMalloc", status));
        }
        NonNull::new(ptr).ok_or_else(|| GpuError::Hip("hipMalloc returned null".into()))
    })
}

/// Allocate `len_bytes` of device memory, zeroed.
pub fn alloc_zeros(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    let ptr = alloc(ordinal, len_bytes)?;
    memset_zeros(ordinal, ptr.as_ptr(), len_bytes)?;
    Ok(ptr)
}

/// Free device memory. No-op on null.
pub fn free(ordinal: usize, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let _ = with_device(ordinal, || {
        let status = unsafe { hipFree(ptr) };
        if status != 0 {
            return Err(hip_error("hipFree", status));
        }
        Ok(())
    });
}

/// Copy from host memory to device memory.
pub fn copy_h2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_h2d: null pointer or zero len".into()));
    }
    with_device(ordinal, || {
        let status = unsafe { hipMemcpy(dst, src, len, HIP_MEMCPY_HOST_TO_DEVICE) };
        if status != 0 {
            return Err(hip_error("hipMemcpy(H2D)", status));
        }
        Ok(())
    })
}

/// Copy from device memory to host memory.
pub fn copy_d2h(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_d2h: null pointer or zero len".into()));
    }
    with_device(ordinal, || {
        let status = unsafe { hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_HOST) };
        if status != 0 {
            return Err(hip_error("hipMemcpy(D2H)", status));
        }
        Ok(())
    })
}

/// Copy from device memory to device memory.
pub fn copy_d2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("copy_d2d: null pointer or zero len".into()));
    }
    with_device(ordinal, || {
        let status = unsafe { hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_DEVICE) };
        if status != 0 {
            return Err(hip_error("hipMemcpy(D2D)", status));
        }
        Ok(())
    })
}

/// Set device memory to zero.
pub fn memset_zeros(ordinal: usize, dst: *mut c_void, len: usize) -> Result<()> {
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg("memset_zeros: null pointer or zero len".into()));
    }
    with_device(ordinal, || {
        let status = unsafe { hipMemset(dst, 0, len) };
        if status != 0 {
            return Err(hip_error("hipMemset", status));
        }
        Ok(())
    })
}

/// Synchronize the device (block until all pending work completes).
pub fn sync(ordinal: usize) -> Result<()> {
    with_device(ordinal, || {
        let status = unsafe { hipDeviceSynchronize() };
        if status != 0 {
            return Err(hip_error("hipDeviceSynchronize", status));
        }
        Ok(())
    })
}

/// RAII wrapper around `hipEvent_t`. Records and measures GPU wall time.
pub struct GpuEvent {
    ordinal: usize,
    raw: *mut c_void,
}

impl GpuEvent {
    pub fn new(ordinal: usize) -> Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        with_device(ordinal, || {
            let status = unsafe { hipEventCreate(&mut raw) };
            if status != 0 {
                return Err(hip_error("hipEventCreate", status));
            }
            Ok(())
        })?;
        Ok(Self { ordinal, raw })
    }

    pub fn record(&self) -> Result<()> {
        with_device(self.ordinal, || {
            let status = unsafe { hipEventRecord(self.raw, std::ptr::null_mut()) };
            if status != 0 {
                return Err(hip_error("hipEventRecord", status));
            }
            Ok(())
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        with_device(self.ordinal, || {
            let status = unsafe { hipEventSynchronize(self.raw) };
            if status != 0 {
                return Err(hip_error("hipEventSynchronize", status));
            }
            Ok(())
        })
    }

    /// Elapsed milliseconds between `start` and `end`. Both events must have
    /// been recorded and `end` must have been synchronized first.
    pub fn elapsed_ms(start: &GpuEvent, end: &GpuEvent) -> Result<f32> {
        let mut ms: f32 = 0.0;
        with_device(start.ordinal, || {
            let status = unsafe { hipEventElapsedTime(&mut ms, start.raw, end.raw) };
            if status != 0 {
                return Err(hip_error("hipEventElapsedTime", status));
            }
            Ok(())
        })?;
        Ok(ms)
    }
}

impl Drop for GpuEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = with_device(self.ordinal, || {
                let status = unsafe { hipEventDestroy(self.raw) };
                if status != 0 {
                    return Err(hip_error("hipEventDestroy", status));
                }
                Ok(())
            });
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
