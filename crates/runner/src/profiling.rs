/// Optional host-side profiling for the retained Qwen3.8 prefill/decode paths.
///
/// Profiling is intentionally opt-in and remains a narrow measurement aid for
/// the retained product path.
use std::path::Path;

pub(crate) struct PrefillProfileScope<'a> {
    inner: Option<NativeProfileScope<'a>>,
}

pub(crate) struct DecodeProfileScope<'a> {
    inner: Option<NativeProfileScope<'a>>,
}

struct NativeProfileScope<'a> {
    active: bool,
    label: &'static str,
    json_path: Option<&'a Path>,
    family: &'a str,
    model: &'a str,
    backend: &'a str,
    token_count: usize,
    start: std::time::Instant,
}

impl<'a> PrefillProfileScope<'a> {
    pub(crate) fn new(
        enabled: bool,
        json_path: Option<&'a Path>,
        family: &'a str,
        model: &'a str,
        backend: &'a str,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            inner: Some(NativeProfileScope::new(
                enabled,
                json_path,
                "prefill",
                family,
                model,
                backend,
                prompt_tokens,
            )),
        }
    }

    pub(crate) fn finish(mut self) -> anyhow::Result<()> {
        if let Some(inner) = self.inner.take() {
            inner.finish()
        } else {
            Ok(())
        }
    }
}

impl<'a> DecodeProfileScope<'a> {
    pub(crate) fn new(
        enabled: bool,
        json_path: Option<&'a Path>,
        family: &'a str,
        model: &'a str,
        backend: &'a str,
        decode_steps: usize,
    ) -> Self {
        Self {
            inner: Some(NativeProfileScope::new(
                enabled,
                json_path,
                "decode",
                family,
                model,
                backend,
                decode_steps,
            )),
        }
    }

    pub(crate) fn finish(mut self) -> anyhow::Result<()> {
        if let Some(inner) = self.inner.take() {
            inner.finish()
        } else {
            Ok(())
        }
    }

    pub(crate) fn finish_with_steps(mut self, decode_steps: usize) -> anyhow::Result<()> {
        if let Some(mut inner) = self.inner.take() {
            inner.token_count = decode_steps;
            inner.finish()
        } else {
            Ok(())
        }
    }
}

impl<'a> NativeProfileScope<'a> {
    fn new(
        enabled: bool,
        json_path: Option<&'a Path>,
        label: &'static str,
        family: &'a str,
        model: &'a str,
        backend: &'a str,
        token_count: usize,
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
            label,
            json_path,
            family,
            model,
            backend,
            token_count,
            start: std::time::Instant::now(),
        }
    }

    fn finish(mut self) -> anyhow::Result<()> {
        if !self.active {
            return Ok(());
        }
        let wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let hal = gpu_hal::hal_profile_snapshot();
        let ffi = kernel_ffi::prefill_ffi::ffi_profile_snapshot();
        let per_token_ms = if self.token_count == 0 {
            0.0
        } else {
            wall_ms / self.token_count as f64
        };
        eprintln!(
            "[{label}-profile] family={} model={} backend={} tokens={} wall_ms={:.3} ms_per_token={:.3} ffi_calls={} ffi_ms={:.3} hal_calls={} hal_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            self.family,
            self.model,
            self.backend,
            self.token_count,
            wall_ms,
            per_token_ms,
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
            label = self.label,
        );
        for entry in ffi.entries.iter().take(40) {
            eprintln!(
                "[{label}-profile] ffi_op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
                label = self.label,
            );
        }
        for entry in hal.entries.iter().take(20) {
            eprintln!(
                "[{label}-profile] hal_op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} bytes={}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
                label = self.label,
            );
        }
        disable_native_profiles();
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
                "phase": self.label,
                "family": self.family,
                "model": self.model,
                "backend": self.backend,
                "tokens": self.token_count,
                "wall_ms": wall_ms,
                "ms_per_token": per_token_ms,
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

impl Drop for NativeProfileScope<'_> {
    fn drop(&mut self) {
        if self.active {
            disable_native_profiles();
        }
    }
}

fn disable_native_profiles() {
    kernel_ffi::prefill_ffi::ffi_profile_set_enabled(false);
    gpu_hal::hal_profile_set_enabled(false);
}
