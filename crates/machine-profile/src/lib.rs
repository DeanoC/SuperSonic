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
    let vector_peak = cpu::vector_kernels::measure(topology.cores_p);

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

    let driver = std::env::var("KFD_DRIVER_VERSION").unwrap_or_else(|_| "unknown".into());
    let fp_components = FingerprintComponents {
        cpu: format!("{} stepping={} microcode={}",
            cpu_profile.model,
            cpu_profile.stepping,
            cpu_profile.microcode.as_deref().unwrap_or("?")),
        gpus: gpus.iter()
            .map(|g| format!("{}:{}:{}", g.backend, g.arch_name,
                             g.pci_id.as_deref().unwrap_or("?")))
            .collect(),
        driver: driver.clone(),
        isa: cpu_profile.isa.clone(),
    };
    let fp = fingerprint::compute(&fp_components);

    Profile {
        schema_version: 1,
        profile_version: "machine-profile/0.1.0".into(),
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
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| format!("linux {}", s.trim()))
        .unwrap_or_else(|_| "unknown".into())
}
