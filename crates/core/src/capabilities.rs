use gpu_hal::Backend;

use crate::registry::{ModelFamily, ModelVariant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub family: ModelFamily,
    pub backend: Backend,
    pub batch_decode: bool,
    pub int4: bool,
    pub fp8_runtime: bool,
    pub kv_fp8: bool,
    pub serve_status: ServeStatus,
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
    let serve_status = match family {
        ModelFamily::Qwen35 | ModelFamily::Gemma4 => ServeStatus::Ready,
        ModelFamily::Qwen36Moe => {
            ServeStatus::CliOnly("Qwen3.6 MoE runtime is still wired through the CLI flow")
        }
        ModelFamily::Phi4 => {
            ServeStatus::CliOnly("Phi-4 runtime is still wired through the CLI flow")
        }
        ModelFamily::Llama31 => {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_cover_every_model_family() {
        let cases = [
            (ModelVariant::Qwen3_5_0_8B, ServeStatus::Ready),
            (ModelVariant::Gemma4_E2B, ServeStatus::Ready),
            (
                ModelVariant::Qwen3_6_35B_A3B,
                ServeStatus::CliOnly("Qwen3.6 MoE runtime is still wired through the CLI flow"),
            ),
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
