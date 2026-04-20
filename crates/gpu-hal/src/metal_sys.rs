use std::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    pub fn supersonic_metal_alloc(len_bytes: usize, ptr_out: *mut *mut c_void) -> c_int;
    pub fn supersonic_metal_free(ptr: *mut c_void) -> c_int;
    pub fn supersonic_metal_lookup_buffer(
        ptr: *const c_void,
        buffer_out: *mut *mut c_void,
        offset_out: *mut usize,
    ) -> c_int;
    pub fn supersonic_metal_query_device_info(
        ordinal: usize,
        arch_name_out: *mut c_char,
        arch_name_len: usize,
        total_vram_out: *mut u64,
        warp_size_out: *mut u32,
        clock_rate_khz_out: *mut u32,
    ) -> c_int;
}
