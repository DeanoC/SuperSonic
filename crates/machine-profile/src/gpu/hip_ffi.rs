#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn mp_lds_bandwidth_run(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_hbm_bandwidth_read(device: i32, bytes: u64) -> f64;
    pub fn mp_hbm_bandwidth_write(device: i32, bytes: u64) -> f64;
    pub fn mp_hbm_bandwidth_copy(device: i32, bytes: u64) -> f64;
    pub fn mp_wmma_peak_f16(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_wmma_peak_bf16(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_wmma_peak_i8(device: i32, cu_count: u32, iters: u64) -> f64;
}
