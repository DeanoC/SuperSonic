use std::time::Duration;

use gpu_hal::Backend;

use crate::registry::{ModelFamily, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlmTransferBackend {
    PageableH2d,
    GpuDirectStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlmDirectProfile {
    pub required_weights: usize,
    pub raw_dense_weights: usize,
    pub native_int4_direct_weights: usize,
    pub bf16_fallback_weights: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlmSourceOpenDurations {
    pub total: Duration,
    pub store_open: Duration,
    pub config: Duration,
    pub direct_plan: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlmStartupDurations {
    pub total: Duration,
    pub source_open: FlmSourceOpenDurations,
    pub tokenizer: Duration,
    pub descriptor: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlmLoadWindowProfileDurations {
    pub allocation_api: Duration,
    pub upload_api: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingFeatures {
    pub plain_prefill_decode: bool,
    pub native_dflash_generate: bool,
    pub prefix_snapshot: bool,
    pub disk_prefix_snapshot: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlmLoadEvidence {
    pub source_file: String,
    pub architecture_id: u32,
    pub model_id: u16,
    pub storage_abi_ids: Vec<u16>,
    pub direct_profile: FlmDirectProfile,
    pub transfer_backend: FlmTransferBackend,
    pub source_bytes: u64,
    pub device_upload_bytes: u64,
    pub startup: FlmStartupDurations,
    pub load_window_profile: FlmLoadWindowProfileDurations,
    pub load_sequence: u64,
    pub source_open_count: u64,
    pub resident_allocation_count: u64,
    pub features: ServingFeatures,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub family: ModelFamily,
    pub backend: Backend,
    pub batch_decode: bool,
    pub int4: bool,
    pub fp8_runtime: bool,
    pub kv_fp8: bool,
    pub serve_status: ServeStatus,
    pub flm: Option<FlmLoadEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeStatus {
    Ready,
    CliOnly(&'static str),
}

impl ServeStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

pub fn capabilities_for_variant(
    variant: &ModelVariant,
    backend: Backend,
    int4: bool,
    fp8_runtime: bool,
    kv_fp8: bool,
) -> ModelCapabilities {
    let family = variant.family();
    let serve_status = match variant {
        ModelVariant::Qwen3_5_0_8B
        | ModelVariant::Qwen3_5_2B
        | ModelVariant::Qwen3_5_4B
        | ModelVariant::Qwen3_5_9B
        | ModelVariant::Qwen3_6_27B
        | ModelVariant::Qwen3_8_27B
        | ModelVariant::Gemma4_E2B
        | ModelVariant::Gemma4_E4B
        | ModelVariant::Qwen3_6_35B_A3B => ServeStatus::Ready,
        ModelVariant::Qwen3_5_35B_A3B => {
            ServeStatus::CliOnly("Qwen3.5 MoE runtime is still wired through the CLI flow")
        }
        ModelVariant::Qwen3_30B_A3B => {
            ServeStatus::CliOnly("Qwen3 MoE runtime is still wired through the CLI flow")
        }
        ModelVariant::Phi4_Mini => {
            ServeStatus::CliOnly("Phi-4 runtime is still wired through the CLI flow")
        }
        ModelVariant::Llama3_1_8B => {
            ServeStatus::CliOnly("Llama 3.1 runtime is still wired through the CLI flow")
        }
    };
    ModelCapabilities {
        family,
        backend,
        batch_decode: false,
        int4,
        fp8_runtime,
        kv_fp8,
        serve_status,
        flm: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_cover_every_model_variant() {
        let cases = [
            (ModelVariant::Qwen3_5_0_8B, ServeStatus::Ready),
            (ModelVariant::Qwen3_5_2B, ServeStatus::Ready),
            (ModelVariant::Qwen3_5_4B, ServeStatus::Ready),
            (ModelVariant::Qwen3_5_9B, ServeStatus::Ready),
            (
                ModelVariant::Qwen3_5_35B_A3B,
                ServeStatus::CliOnly("Qwen3.5 MoE runtime is still wired through the CLI flow"),
            ),
            (
                ModelVariant::Qwen3_30B_A3B,
                ServeStatus::CliOnly("Qwen3 MoE runtime is still wired through the CLI flow"),
            ),
            (ModelVariant::Qwen3_6_27B, ServeStatus::Ready),
            (ModelVariant::Qwen3_6_35B_A3B, ServeStatus::Ready),
            (ModelVariant::Gemma4_E2B, ServeStatus::Ready),
            (ModelVariant::Gemma4_E4B, ServeStatus::Ready),
            (
                ModelVariant::Phi4_Mini,
                ServeStatus::CliOnly("Phi-4 runtime is still wired through the CLI flow"),
            ),
            (
                ModelVariant::Llama3_1_8B,
                ServeStatus::CliOnly("Llama 3.1 runtime is still wired through the CLI flow"),
            ),
        ];
        for (variant, status) in cases {
            let caps = capabilities_for_variant(&variant, Backend::Cuda, false, false, false);
            assert_eq!(caps.family, variant.family());
            assert_eq!(caps.serve_status, status);
        }
    }
}
