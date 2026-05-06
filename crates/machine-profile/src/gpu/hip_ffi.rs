#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn mp_query_device_info(
        device: i32,
        arch_name_out: *mut u8,
        arch_name_len: u32,
        total_vram_bytes_out: *mut u64,
        warp_size_out: *mut u32,
        clock_rate_khz_out: *mut u32,
        pci_device_id_out: *mut u32,
    ) -> i32;

    pub fn mp_lds_bandwidth_run(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_hbm_bandwidth_read(device: i32, bytes: u64) -> f64;
    pub fn mp_hbm_bandwidth_write(device: i32, bytes: u64) -> f64;
    pub fn mp_hbm_bandwidth_copy(device: i32, bytes: u64) -> f64;
    pub fn mp_wmma_peak_f16(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_wmma_peak_bf16(device: i32, cu_count: u32, iters: u64) -> f64;
    pub fn mp_wmma_peak_i8(device: i32, cu_count: u32, iters: u64) -> f64;

    /// Probe the device's per-dtype WMMA support. Writes 0/1 flags into
    /// `flags_out_3[0..3]` (f16, bf16, i8). Returns 0 on success.
    pub fn mp_wmma_probe(device: i32, flags_out_3: *mut i32) -> i32;
}

#[cfg(supersonic_backend_hip)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MpTransferSample {
    pub bytes: u64,
    pub gb_s: f64,
}

#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn mp_pcie_h2d(device: i32, out: *mut MpTransferSample, max_samples: i32) -> i32;
    pub fn mp_pcie_d2h(device: i32, out: *mut MpTransferSample, max_samples: i32) -> i32;
    pub fn mp_pcie_duplex(device: i32, bytes: u64) -> f64;
}
