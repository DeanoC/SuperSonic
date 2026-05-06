use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Profile {
    pub schema_version: u32,
    pub profile_version: String,
    pub fingerprint: String,
    pub fingerprint_components: FingerprintComponents,
    pub captured_at: String,
    #[serde(default)]
    pub warnings: Vec<Warning>,
    #[serde(default)]
    pub cpu: Option<CpuProfile>,
    #[serde(default)]
    pub gpus: Vec<GpuProfile>,
    pub system: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FingerprintComponents {
    pub cpu: String,
    pub gpus: Vec<String>,
    pub driver: String,
    pub isa: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Warning {
    pub component: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemInfo {
    pub ram_bytes: u64,
    pub os: String,
    pub kernel_driver: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuProfile {
    pub vendor: String,
    pub model: String,
    pub stepping: u32,
    pub microcode: Option<String>,
    pub isa: Vec<String>,
    pub topology: CpuTopology,
    pub cache: CacheHierarchy,
    pub vector_peak: VectorPeak,
    pub dram: DramBandwidth,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuTopology {
    pub sockets: u32,
    pub cores_total: u32,
    pub cores_p: u32,
    pub cores_e: u32,
    pub threads_per_core: u32,
    pub numa_nodes: Vec<NumaNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumaNode {
    pub id: u32,
    pub cpus: Vec<u32>,
    pub ram_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheHierarchy {
    pub l1d: Option<CacheLevel>,
    pub l2: Option<CacheLevel>,
    pub l3: Option<CacheLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheLevel {
    pub size_bytes: u64,
    pub line_bytes: u32,
    pub ways: Option<u32>,
    pub measured_lat_ns: Option<f64>,
    pub measured_bw_gb_s: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VectorPeak {
    pub fp32: Option<MeasuredVsTheoretical>,
    pub fp16: Option<MeasuredVsTheoretical>,
    pub bf16: Option<MeasuredVsTheoretical>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasuredVsTheoretical {
    pub measured_per_unit: Option<f64>,
    pub measured_aggregate: f64,
    pub theoretical_aggregate: Option<f64>,
    pub ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DramBandwidth {
    pub single_thread_read_gb_s: Option<f64>,
    pub stream_read_gb_s: Option<f64>,
    pub stream_write_gb_s: Option<f64>,
    pub stream_copy_gb_s: Option<f64>,
    pub theoretical_peak_gb_s: Option<f64>,
    pub ratio_copy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuProfile {
    pub backend: String,
    pub device_index: u32,
    pub arch_name: String,
    pub pci_id: Option<String>,
    pub uuid: Option<String>,
    pub memory_arch: String,
    pub total_vram_bytes: u64,
    pub cu_count: u32,
    pub wave_size: u32,
    pub lds_per_cu_bytes: u64,
    pub lds_bw_per_cu_gb_s: Option<f64>,
    pub lds_bw_aggregate_gb_s: Option<f64>,
    pub vram_bw: VramBandwidth,
    pub mma_peak: MmaPeak,
    pub pcie: PcieProfile,
    pub clock_rate_khz_measured: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VramBandwidth {
    pub read_gb_s: Option<f64>,
    pub write_gb_s: Option<f64>,
    pub copy_gb_s: Option<f64>,
    pub theoretical_peak_gb_s: Option<f64>,
    pub ratio_read: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MmaPeak {
    pub f16: Option<MmaMeasurement>,
    pub bf16: Option<MmaMeasurement>,
    pub fp8_e4m3: Option<MmaMeasurement>,
    pub i8: Option<MmaMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MmaMeasurement {
    pub measured_tflops: f64,
    pub theoretical_tflops: Option<f64>,
    pub ratio: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PcieProfile {
    pub generation: Option<String>,
    pub h2d_gb_s_by_size: Vec<TransferSample>,
    pub d2h_gb_s_by_size: Vec<TransferSample>,
    pub duplex_gb_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransferSample {
    pub bytes: u64,
    pub gb_s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> Profile {
        Profile {
            schema_version: 1,
            profile_version: "machine-profile/0.1.0".into(),
            fingerprint: "blake3:test".into(),
            fingerprint_components: FingerprintComponents {
                cpu: "AMD Ryzen 9 7950X".into(),
                gpus: vec!["HIP:gfx1100:0x744c".into()],
                driver: "amdgpu 6.10".into(),
                isa: vec!["AVX2".into()],
            },
            captured_at: "2026-05-06T12:34:56Z".into(),
            warnings: vec![],
            cpu: None,
            gpus: vec![],
            system: SystemInfo {
                ram_bytes: 64_000_000_000,
                os: "linux 6.19.14".into(),
                kernel_driver: Some("amdgpu 6.10".into()),
            },
        }
    }

    #[test]
    fn profile_round_trips_through_json() {
        let p = sample_profile();
        let s = serde_json::to_string(&p).unwrap();
        let back: Profile = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }
}
