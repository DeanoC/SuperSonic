/// Optional host-side profiling for the retained Qwen3.8 prefill path.
///
/// Profiling is intentionally opt-in and remains a narrow measurement aid for
/// the retained product path.
pub(crate) struct PrefillProfileScope<'a> {
    active: bool,
    json_path: Option<&'a std::path::Path>,
    family: &'a str,
    model: &'a str,
    backend: &'a str,
    prompt_tokens: usize,
    start: std::time::Instant,
}

impl<'a> PrefillProfileScope<'a> {
    pub(crate) fn new(
        enabled: bool,
        json_path: Option<&'a std::path::Path>,
        family: &'a str,
        model: &'a str,
        backend: &'a str,
        prompt_tokens: usize,
    ) -> Self {
        let active = enabled || json_path.is_some();
        if active {
            gpu_hal::hal_profile_set_enabled(true);
            kernel_ffi::prefill_ffi::ffi_profile_set_enabled(true);
            gpu_hal::hal_profile_reset();
            kernel_ffi::prefill_ffi::ffi_profile_reset();
        }
        Self {
            active,
            json_path,
            family,
            model,
            backend,
            prompt_tokens,
            start: std::time::Instant::now(),
        }
    }

    pub(crate) fn finish(mut self) -> anyhow::Result<()> {
        if !self.active {
            return Ok(());
        }
        let wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let hal = gpu_hal::hal_profile_snapshot();
        let ffi = kernel_ffi::prefill_ffi::ffi_profile_snapshot();
        eprintln!(
            "[prefill-profile] family={} model={} backend={} prompt_tokens={} wall_ms={:.3} ffi_calls={} ffi_ms={:.3} hal_calls={} hal_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            self.family,
            self.model,
            self.backend,
            self.prompt_tokens,
            wall_ms,
            ffi.total_calls,
            ffi.total_ms,
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
        for entry in ffi.entries.iter().take(30) {
            eprintln!(
                "[prefill-profile] ffi_op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
            );
        }
        for entry in hal.entries.iter().take(20) {
            eprintln!(
                "[prefill-profile] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} bytes={}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
            );
        }
        disable_prefill_profiles();
        self.active = false;
        if let Some(path) = self.json_path {
            let ffi_entries: Vec<_> = ffi
                .entries
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "op": entry.op,
                        "calls": entry.calls,
                        "mean_ms": entry.mean_ms(),
                        "total_ms": entry.total_ms,
                        "max_ms": entry.max_ms,
                    })
                })
                .collect();
            let hal_entries: Vec<_> = hal
                .entries
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "op": entry.op,
                        "calls": entry.calls,
                        "mean_ms": entry.mean_ms(),
                        "total_ms": entry.total_ms,
                        "max_ms": entry.max_ms,
                        "total_bytes": entry.total_bytes,
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "family": self.family,
                "model": self.model,
                "backend": self.backend,
                "prompt_tokens": self.prompt_tokens,
                "wall_ms": wall_ms,
                "ffi": {
                    "total_calls": ffi.total_calls,
                    "total_ms": ffi.total_ms,
                    "entries": ffi_entries,
                },
                "hal": {
                    "total_calls": hal.total_calls,
                    "total_ms": hal.total_ms,
                    "alloc_calls": hal.alloc_calls,
                    "alloc_bytes": hal.alloc_bytes,
                    "free_calls": hal.free_calls,
                    "h2d_bytes": hal.h2d_bytes,
                    "d2h_bytes": hal.d2h_bytes,
                    "d2d_bytes": hal.d2d_bytes,
                    "memset_bytes": hal.memset_bytes,
                    "sync_calls": hal.sync_calls,
                    "entries": hal_entries,
                },
            });
            std::fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
        }
        Ok(())
    }
}

pub(crate) struct DflashProfileScope {
    active: bool,
    start: std::time::Instant,
}

impl DflashProfileScope {
    pub(crate) fn new(enabled: bool) -> Self {
        if enabled {
            gpu_hal::hal_profile_set_enabled(true);
            kernel_ffi::prefill_ffi::ffi_profile_set_enabled(true);
            gpu_hal::hal_profile_reset();
            kernel_ffi::prefill_ffi::ffi_profile_reset();
        }
        Self {
            active: enabled,
            start: std::time::Instant::now(),
        }
    }

    pub(crate) fn finish(mut self) {
        if !self.active {
            return;
        }
        let wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let hal = gpu_hal::hal_profile_snapshot();
        let ffi = kernel_ffi::prefill_ffi::ffi_profile_snapshot();
        eprintln!(
            "[dflash-profile] wall_ms={wall_ms:.3} ffi_calls={} ffi_ms={:.3} hal_calls={} hal_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            ffi.total_calls,
            ffi.total_ms,
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
        for entry in ffi.entries.iter().take(40) {
            eprintln!(
                "[dflash-profile] ffi_op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
            );
        }
        for entry in hal.entries.iter().take(25) {
            eprintln!(
                "[dflash-profile] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} bytes={}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
            );
        }
        disable_prefill_profiles();
        self.active = false;
    }
}

impl Drop for DflashProfileScope {
    fn drop(&mut self) {
        if self.active {
            disable_prefill_profiles();
        }
    }
}

impl Drop for PrefillProfileScope<'_> {
    fn drop(&mut self) {
        if self.active {
            disable_prefill_profiles();
        }
    }
}

fn disable_prefill_profiles() {
    kernel_ffi::prefill_ffi::ffi_profile_set_enabled(false);
    gpu_hal::hal_profile_set_enabled(false);
}

#[cfg(test)]
mod dflash_profile_tests {
    use super::DflashProfileScope;

    #[test]
    fn dflash_profile_scope_toggles_gpu_accumulators() {
        let scope = DflashProfileScope::new(true);
        assert!(kernel_ffi::prefill_ffi::ffi_profile_enabled());
        assert!(gpu_hal::hal_profile_enabled());
        scope.finish();
        assert!(!kernel_ffi::prefill_ffi::ffi_profile_enabled());
        assert!(!gpu_hal::hal_profile_enabled());
    }
}
