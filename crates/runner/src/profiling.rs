/// RAII scope that enables Metal/HAL profiling when SUPERSONIC_METAL_PROFILE
/// is set in the environment, and dumps a per-op breakdown to stderr when
/// the scope drops. Used to investigate Metal v2 decode hot paths without
/// adding a permanent profiling cost.
pub(crate) struct MetalProfileScope {
    active: bool,
}

impl MetalProfileScope {
    pub(crate) fn new() -> Self {
        let active = std::env::var_os("SUPERSONIC_METAL_PROFILE").is_some();
        if active {
            kernel_ffi::prefill_ffi::metal_profile_set_enabled(true);
            gpu_hal::hal_profile_set_enabled(true);
            kernel_ffi::prefill_ffi::metal_profile_reset();
            gpu_hal::hal_profile_reset();
        }
        Self { active }
    }
}

impl Drop for MetalProfileScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let metal = kernel_ffi::prefill_ffi::metal_profile_snapshot();
        let hal = gpu_hal::hal_profile_snapshot();
        eprintln!();
        eprintln!("=== Metal native/host op profile ===");
        eprintln!(
            "calls={} total_ms={:.3} (native={:.3} ms / host={:.3} ms)",
            metal.total_calls, metal.total_ms, metal.native_ms, metal.host_ms
        );
        eprintln!(
            "{:<48} {:>10} {:>10} {:>12} {:>12}",
            "op (path)", "calls", "mean_ms", "total_ms", "max_ms"
        );
        for entry in metal.entries.iter().take(40) {
            let mean_ms = if entry.calls > 0 {
                entry.total_ms / entry.calls as f64
            } else {
                0.0
            };
            eprintln!(
                "{:<48} {:>10} {:>10.4} {:>12.3} {:>12.3}",
                format!("{} ({})", entry.op, entry.path),
                entry.calls,
                mean_ms,
                entry.total_ms,
                entry.max_ms
            );
        }
        eprintln!();
        eprintln!("=== HAL op profile (gpu_hal level) ===");
        eprintln!(
            "calls={} total_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            hal.total_calls,
            hal.total_ms,
            hal.alloc_calls,
            hal.alloc_bytes,
            hal.h2d_bytes,
            hal.d2h_bytes,
            hal.d2d_bytes,
            hal.memset_bytes,
            hal.sync_calls,
        );
        eprintln!(
            "{:<32} {:>10} {:>10} {:>12} {:>12} {:>14}",
            "op", "calls", "mean_ms", "total_ms", "max_ms", "total_bytes"
        );
        for entry in hal.entries.iter().take(20) {
            let mean_ms = if entry.calls > 0 {
                entry.total_ms / entry.calls as f64
            } else {
                0.0
            };
            eprintln!(
                "{:<32} {:>10} {:>10.4} {:>12.3} {:>12.3} {:>14}",
                entry.op, entry.calls, mean_ms, entry.total_ms, entry.max_ms, entry.total_bytes
            );
        }
        kernel_ffi::prefill_ffi::metal_profile_set_enabled(false);
        gpu_hal::hal_profile_set_enabled(false);
    }
}
