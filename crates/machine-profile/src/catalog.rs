//! Static table of theoretical peaks for known CPUs and GPUs.

#[derive(Debug, Clone, Copy)]
pub struct GpuPeaks {
    pub theoretical_hbm_gb_s: f64,
    pub theoretical_f16_tflops: f64,
    pub theoretical_bf16_tflops: f64,
    pub theoretical_fp8_tflops: Option<f64>,
    pub theoretical_i8_tops: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuPeaks {
    pub theoretical_dram_gb_s: f64,
    pub theoretical_fp32_gflops_per_core: f64,
    pub theoretical_bf16_gflops_per_core: Option<f64>,
}

struct GpuEntry {
    arch: &'static str,
    pci_id: Option<&'static str>,
    peaks: GpuPeaks,
}

struct CpuEntry {
    pattern: &'static str,
    peaks: CpuPeaks,
}

const GPUS: &[GpuEntry] = &[
    GpuEntry {
        arch: "gfx1100",
        pci_id: Some("0x744c"),
        peaks: GpuPeaks {
            theoretical_hbm_gb_s: 800.0,
            theoretical_f16_tflops: 122.8,
            theoretical_bf16_tflops: 122.8,
            theoretical_fp8_tflops: None,
            theoretical_i8_tops: Some(245.6),
        },
    },
    GpuEntry {
        arch: "gfx1150",
        pci_id: None,
        peaks: GpuPeaks {
            theoretical_hbm_gb_s: 89.6,
            theoretical_f16_tflops: 18.0,
            theoretical_bf16_tflops: 18.0,
            theoretical_fp8_tflops: None,
            theoretical_i8_tops: Some(36.0),
        },
    },
    GpuEntry {
        arch: "gfx90a",
        pci_id: None,
        peaks: GpuPeaks {
            theoretical_hbm_gb_s: 1638.0,
            theoretical_f16_tflops: 383.0,
            theoretical_bf16_tflops: 383.0,
            theoretical_fp8_tflops: None,
            theoretical_i8_tops: Some(383.0),
        },
    },
];

const CPUS: &[CpuEntry] = &[
    CpuEntry {
        pattern: "Ryzen 9 7950X",
        peaks: CpuPeaks {
            theoretical_dram_gb_s: 83.2,
            theoretical_fp32_gflops_per_core: 200.0,
            theoretical_bf16_gflops_per_core: Some(400.0),
        },
    },
    CpuEntry {
        pattern: "Ryzen 7 7840U",
        peaks: CpuPeaks {
            theoretical_dram_gb_s: 89.6,
            theoretical_fp32_gflops_per_core: 200.0,
            theoretical_bf16_gflops_per_core: Some(400.0),
        },
    },
];

pub fn lookup_gpu(arch: &str, pci_id: Option<&str>) -> Option<GpuPeaks> {
    // HIP/ROCm reports CDNA target IDs as e.g. "gfx90a:sramecc+:xnack-".
    // Strip the feature suffix so "gfx90a" in the catalog still matches.
    let base_arch = arch.split(':').next().unwrap_or(arch);
    GPUS.iter()
        .find(|e| e.arch == base_arch && (e.pci_id.is_none() || e.pci_id == pci_id))
        .map(|e| e.peaks)
}

pub fn lookup_cpu(model: &str) -> Option<CpuPeaks> {
    CPUS.iter().find(|e| model.contains(e.pattern)).map(|e| e.peaks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1100_lookup_returns_known_peaks() {
        let peaks = lookup_gpu("gfx1100", Some("0x744c")).expect("should hit");
        assert!((peaks.theoretical_hbm_gb_s - 800.0).abs() < 1e-3);
        assert!((peaks.theoretical_bf16_tflops - 122.8).abs() < 1e-3);
    }

    #[test]
    fn unknown_gpu_returns_none() {
        assert!(lookup_gpu("gfx99999", None).is_none());
    }

    #[test]
    fn cdna_target_id_with_feature_suffix_matches_base_arch() {
        // ROCm reports CDNA targets like "gfx90a:sramecc+:xnack-".
        // The catalog entry is keyed by "gfx90a" alone.
        let peaks = lookup_gpu("gfx90a:sramecc+:xnack-", None)
            .expect("should match base arch after stripping feature suffix");
        assert!((peaks.theoretical_hbm_gb_s - 1638.0).abs() < 1e-3);
    }

    #[test]
    fn cpu_lookup_matches_substring() {
        let peaks = lookup_cpu("AMD Ryzen 9 7950X 16-Core Processor").expect("should hit");
        assert!((peaks.theoretical_dram_gb_s - 83.2).abs() < 1e-3);
    }
}
