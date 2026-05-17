//! Machine profiling — measure CPU + GPU hardware capabilities.

pub mod catalog;
pub mod cpu;
pub mod fingerprint;
pub mod gpu;
pub mod schema;
pub mod store;

pub use schema::Profile;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend not implemented: {0}")]
    BackendNotImplemented(&'static str),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn measure() -> Profile {
    use chrono::Utc;
    use schema::*;

    let mut warnings = Vec::new();

    let cpu_id = cpu::detect_cpu_id();
    let topology = cpu::topology::detect();
    let cache = cpu::cache::detect();
    let vector_peak = cpu::vector_kernels::measure(topology.cores_p.max(1));

    // Skip DRAM in startup-mode measurement (allocates ~768 MiB); the CLI
    // can opt in via a future flag. For now we record `null` and warn.
    let dram = schema::DramBandwidth {
        single_thread_read_gb_s: None,
        stream_read_gb_s: None,
        stream_write_gb_s: None,
        stream_copy_gb_s: None,
        theoretical_peak_gb_s: None,
        ratio_copy: None,
    };
    warnings.push(Warning {
        component: "cpu.dram".into(),
        reason: "DRAM measurement skipped in default measure() path".into(),
    });

    let cpu_profile = CpuProfile {
        vendor: cpu_id.vendor.clone(),
        model: cpu_id.model.clone(),
        stepping: cpu_id.stepping,
        microcode: cpu_id.microcode.clone(),
        isa: cpu_id.isa.clone(),
        topology,
        cache,
        vector_peak,
        dram,
    };

    let mut gpus = gpu::run_all();
    apply_catalog(&mut gpus, &cpu_profile);

    let driver = detect_driver();
    let fp_components = FingerprintComponents {
        cpu: format!(
            "{} stepping={} microcode={}",
            cpu_profile.model,
            cpu_profile.stepping,
            cpu_profile.microcode.as_deref().unwrap_or("?")
        ),
        gpus: gpus.iter().map(gpu_fingerprint_descriptor).collect(),
        driver: driver.clone(),
        isa: cpu_profile.isa.clone(),
    };
    let fp = fingerprint::compute(&fp_components);

    Profile {
        schema_version: 2,
        profile_version: "machine-profile/0.2.0".into(),
        fingerprint: fp,
        fingerprint_components: fp_components,
        captured_at: Utc::now().to_rfc3339(),
        warnings,
        cpu: Some(cpu_profile),
        gpus,
        system: SystemInfo {
            ram_bytes: read_total_ram().unwrap_or(0),
            os: read_uname_release(),
            kernel_driver: Some(driver),
        },
    }
}

fn gpu_fingerprint_descriptor(g: &schema::GpuProfile) -> String {
    if g.backend == "Metal" {
        return format!("Metal:{}:{}", g.arch_name, g.cu_count);
    }
    format!(
        "{}:{}:{}",
        g.backend,
        g.arch_name,
        g.pci_id.as_deref().unwrap_or("?")
    )
}

/// Compute the fingerprint without running any microkernels.
///
/// This is a small, fast subset of `measure()` — CPU `/proc/cpuinfo` parsing,
/// GPU enumeration via `hipGetDeviceProperties`, ISA detection. Suitable for
/// runtime cache invalidation checks where you only need to know if the
/// machine is the same as the cached profile.
pub fn fingerprint_only() -> (String, schema::FingerprintComponents) {
    use schema::*;

    let cpu_id = cpu::detect_cpu_id();
    let mut gpu_descriptors = Vec::new();
    #[cfg(supersonic_backend_hip)]
    {
        // Enumerate HIP devices for fingerprint inputs only — no kernel launches.
        if let Ok(infos) = gpu::hip::enumerate_for_fingerprint() {
            for info in infos {
                gpu_descriptors.push(format!(
                    "HIP:{}:0x{:04x}",
                    info.arch_name, info.pci_device_id
                ));
            }
        }
    }
    #[cfg(supersonic_backend_metal)]
    {
        if let Ok(infos) = gpu::metal::enumerate_for_fingerprint() {
            for info in infos {
                gpu_descriptors.push(format!("Metal:{}:{}", info.arch_name, info.core_count));
            }
        }
    }
    let driver = detect_driver();

    let components = FingerprintComponents {
        cpu: format!(
            "{} stepping={} microcode={}",
            cpu_id.model,
            cpu_id.stepping,
            cpu_id.microcode.as_deref().unwrap_or("?")
        ),
        gpus: gpu_descriptors,
        driver,
        isa: cpu_id.isa,
    };
    let fp = fingerprint::compute(&components);
    (fp, components)
}

fn apply_catalog(gpus: &mut [schema::GpuProfile], cpu: &schema::CpuProfile) {
    for g in gpus.iter_mut() {
        if let Some(peaks) = catalog::lookup_gpu(&g.arch_name, g.pci_id.as_deref()) {
            g.vram_bw.theoretical_peak_gb_s = Some(peaks.theoretical_hbm_gb_s);
            if let Some(read) = g.vram_bw.read_gb_s {
                g.vram_bw.ratio_read = Some(read / peaks.theoretical_hbm_gb_s);
            }
            if let Some(m) = g.mma_peak.f16.as_mut() {
                m.theoretical_tflops = Some(peaks.theoretical_f16_tflops);
                m.ratio = Some(m.measured_tflops / peaks.theoretical_f16_tflops);
            }
            if let Some(m) = g.mma_peak.bf16.as_mut() {
                m.theoretical_tflops = Some(peaks.theoretical_bf16_tflops);
                m.ratio = Some(m.measured_tflops / peaks.theoretical_bf16_tflops);
            }
        }
    }
    let _ = cpu; // CPU theoretical peaks are filled lazily — see follow-up.
}

fn read_total_ram() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        if let Some(v) = sysctl_string("hw.memsize").and_then(|s| s.parse().ok()) {
            return Some(v);
        }
    }
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

fn read_uname_release() -> String {
    #[cfg(target_os = "macos")]
    {
        let product = sysctl_string("kern.osproductversion")
            .or_else(|| sysctl_string("kern.osrelease"))
            .unwrap_or_else(|| "unknown".into());
        format!("macos {product}")
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| format!("linux {}", s.trim()))
            .unwrap_or_else(|_| "unknown".into())
    }
}

fn sysctl_string(name: &str) -> Option<String> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Detect a stable driver-version string for fingerprinting.
///
/// Combines the kernel release (changes on amdgpu module updates) with the
/// HIP runtime version (changes on ROCm userspace updates). On non-HIP
/// builds or when HIP is unavailable, falls back to kernel release alone.
/// `KFD_DRIVER_VERSION` is honoured as an explicit override for users who
/// want to pin a specific value.
fn detect_driver() -> String {
    if let Ok(env) = std::env::var("KFD_DRIVER_VERSION") {
        return env;
    }
    let kernel = read_uname_release();
    #[cfg(supersonic_backend_hip)]
    {
        let mut version: u32 = 0;
        let status = unsafe { gpu::hip_ffi::mp_hip_runtime_version(&mut version) };
        if status == 0 && version != 0 {
            // 60020322 → "rocm 6.2.2"
            let major = version / 10_000_000;
            let minor = (version / 100_000) % 100;
            let patch = (version / 1_000) % 100;
            return format!("rocm {major}.{minor}.{patch} {kernel}");
        }
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `fingerprint_only()` produces the same fingerprint hash as
    /// `measure()`. This ensures the fast path is bit-exact with the full path
    /// so cache invalidation decisions are consistent.
    ///
    /// This test calls `measure()` which runs GPU microkernels; mark it
    /// `#[ignore]` so it only runs on hardware CI where a GPU is present.
    #[test]
    #[ignore]
    fn fingerprint_only_matches_measure() {
        let (fp_fast, _components) = fingerprint_only();
        let profile = measure();
        assert_eq!(
            fp_fast, profile.fingerprint,
            "fingerprint_only() hash '{}' != measure() hash '{}'",
            fp_fast, profile.fingerprint
        );
    }
}
