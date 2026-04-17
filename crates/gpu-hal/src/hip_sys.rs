use std::ffi::{c_int, c_uint, c_void};

pub(crate) const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub(crate) const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub(crate) const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
#[allow(dead_code)]
pub(crate) const HIP_HOST_REGISTER_MAPPED: c_uint = 0x2;

#[link(name = "amdhip64")]
unsafe extern "C" {
    pub(crate) fn hipGetDevice(device: *mut c_int) -> c_int;
    pub(crate) fn hipSetDevice(device: c_int) -> c_int;
    pub(crate) fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub(crate) fn hipFree(ptr: *mut c_void) -> c_int;
    pub(crate) fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> c_int;
    pub(crate) fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    pub(crate) fn hipDeviceSynchronize() -> c_int;
    pub(crate) fn hipEventCreate(event: *mut *mut c_void) -> c_int;
    pub(crate) fn hipEventDestroy(event: *mut c_void) -> c_int;
    pub(crate) fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    pub(crate) fn hipEventSynchronize(event: *mut c_void) -> c_int;
    pub(crate) fn hipEventElapsedTime(
        ms: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> c_int;
}
