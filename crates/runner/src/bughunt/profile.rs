use anyhow::Result;

use super::report::{
    HalProfileOpReport, HalProfileReport, MetalProfileOpReport, MetalProfileReport,
};

#[derive(Debug, Default)]
pub(crate) struct ProfileReports {
    pub(crate) metal: Option<MetalProfileReport>,
    pub(crate) hal: Option<HalProfileReport>,
}

pub(crate) fn collect_profiles<F>(f: F) -> Result<ProfileReports>
where
    F: FnOnce() -> Result<()>,
{
    let _guard = ProfileGuard::new();
    reset_profiles();
    f()?;
    snapshot_profiles()
}

pub(crate) fn reset_profiles() {
    kernel_ffi::prefill_ffi::metal_profile_reset();
    gpu_hal::hal_profile_reset();
}

pub(crate) fn snapshot_profiles() -> Result<ProfileReports> {
    Ok(ProfileReports {
        metal: Some(metal_profile_report(
            kernel_ffi::prefill_ffi::metal_profile_snapshot(),
        )),
        hal: Some(hal_profile_report(gpu_hal::hal_profile_snapshot())),
    })
}

pub(crate) fn metal_profile_report(
    snapshot: kernel_ffi::prefill_ffi::MetalProfileSnapshot,
) -> MetalProfileReport {
    MetalProfileReport {
        total_calls: snapshot.total_calls,
        native_calls: snapshot.native_calls,
        host_calls: snapshot.host_calls,
        total_ms: snapshot.total_ms,
        native_ms: snapshot.native_ms,
        host_ms: snapshot.host_ms,
        entries: snapshot
            .entries
            .into_iter()
            .map(|entry| MetalProfileOpReport {
                mean_ms: entry.mean_ms(),
                op: entry.op,
                path: entry.path,
                calls: entry.calls,
                total_ms: entry.total_ms,
                max_ms: entry.max_ms,
            })
            .collect(),
    }
}

pub(crate) fn hal_profile_report(snapshot: gpu_hal::HalProfileSnapshot) -> HalProfileReport {
    HalProfileReport {
        total_calls: snapshot.total_calls,
        total_ms: snapshot.total_ms,
        alloc_calls: snapshot.alloc_calls,
        alloc_bytes: snapshot.alloc_bytes,
        free_calls: snapshot.free_calls,
        h2d_bytes: snapshot.h2d_bytes,
        d2h_bytes: snapshot.d2h_bytes,
        d2d_bytes: snapshot.d2d_bytes,
        memset_bytes: snapshot.memset_bytes,
        sync_calls: snapshot.sync_calls,
        entries: snapshot
            .entries
            .into_iter()
            .map(|entry| HalProfileOpReport {
                mean_ms: entry.mean_ms(),
                op: entry.op,
                calls: entry.calls,
                total_ms: entry.total_ms,
                max_ms: entry.max_ms,
                total_bytes: entry.total_bytes,
            })
            .collect(),
    }
}

pub(crate) struct ProfileGuard;

impl ProfileGuard {
    pub(crate) fn new() -> Self {
        kernel_ffi::prefill_ffi::metal_profile_set_enabled(true);
        gpu_hal::hal_profile_set_enabled(true);
        Self
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        kernel_ffi::prefill_ffi::metal_profile_set_enabled(false);
        gpu_hal::hal_profile_set_enabled(false);
    }
}

pub(crate) fn print_profile_summary(prompt_name: &str, phase: &str, profile: &MetalProfileReport) {
    println!(
        "PROFILE prompt={} phase={} total_calls={} native_calls={} host_calls={} total_ms={:.1} native_ms={:.1} host_ms={:.1}",
        prompt_name,
        phase,
        profile.total_calls,
        profile.native_calls,
        profile.host_calls,
        profile.total_ms,
        profile.native_ms,
        profile.host_ms,
    );
    for entry in profile.entries.iter().take(8) {
        println!(
            "PROFILE_OP prompt={} phase={} op={} path={} calls={} total_ms={:.1} mean_ms={:.3} max_ms={:.3}",
            prompt_name,
            phase,
            entry.op,
            entry.path,
            entry.calls,
            entry.total_ms,
            entry.mean_ms,
            entry.max_ms,
        );
    }
}

pub(crate) fn print_hal_profile_summary(
    prompt_name: &str,
    phase: &str,
    profile: &HalProfileReport,
) {
    println!(
        "HAL_PROFILE prompt={} phase={} total_calls={} total_ms={:.1} alloc_calls={} alloc_mb={:.1} free_calls={} h2d_mb={:.1} d2h_mb={:.1} d2d_mb={:.1} memset_mb={:.1} sync_calls={}",
        prompt_name,
        phase,
        profile.total_calls,
        profile.total_ms,
        profile.alloc_calls,
        bytes_to_mb(profile.alloc_bytes),
        profile.free_calls,
        bytes_to_mb(profile.h2d_bytes),
        bytes_to_mb(profile.d2h_bytes),
        bytes_to_mb(profile.d2d_bytes),
        bytes_to_mb(profile.memset_bytes),
        profile.sync_calls,
    );
    for entry in profile.entries.iter().take(8) {
        println!(
            "HAL_OP prompt={} phase={} op={} calls={} total_ms={:.1} mean_ms={:.3} max_ms={:.3} total_mb={:.1}",
            prompt_name,
            phase,
            entry.op,
            entry.calls,
            entry.total_ms,
            entry.mean_ms,
            entry.max_ms,
            bytes_to_mb(entry.total_bytes),
        );
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_profile_report_preserves_dispatch_summary() {
        let report = metal_profile_report(kernel_ffi::prefill_ffi::MetalProfileSnapshot {
            total_calls: 3,
            native_calls: 2,
            host_calls: 1,
            total_ms: 4.0,
            native_ms: 3.0,
            host_ms: 1.0,
            entries: vec![kernel_ffi::prefill_ffi::MetalProfileEntry {
                op: "cast".to_string(),
                path: "native".to_string(),
                calls: 2,
                total_ms: 3.0,
                max_ms: 2.0,
            }],
        });
        assert_eq!(report.total_calls, 3);
        assert_eq!(report.native_calls, 2);
        assert_eq!(report.host_calls, 1);
        assert_eq!(report.entries[0].mean_ms, 1.5);
    }

    #[test]
    fn hal_profile_report_preserves_memory_summary() {
        let report = hal_profile_report(gpu_hal::HalProfileSnapshot {
            total_calls: 2,
            total_ms: 5.0,
            alloc_calls: 1,
            alloc_bytes: 4096,
            free_calls: 1,
            h2d_bytes: 128,
            d2h_bytes: 256,
            d2d_bytes: 512,
            memset_bytes: 1024,
            sync_calls: 1,
            entries: vec![gpu_hal::HalProfileEntry {
                op: "alloc".to_string(),
                calls: 1,
                total_ms: 4.0,
                max_ms: 4.0,
                total_bytes: 4096,
            }],
        });
        assert_eq!(report.alloc_calls, 1);
        assert_eq!(report.alloc_bytes, 4096);
        assert_eq!(report.entries[0].mean_ms, 4.0);
    }
}
