#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn mp_lds_bandwidth_run(device: i32, cu_count: u32, iters: u64) -> f64;
}
