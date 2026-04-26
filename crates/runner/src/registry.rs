use std::fmt;

pub use gpu_hal::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen35,
    Qwen36Moe,
    Gemma4,
    Phi4,
    Llama31,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen35 => write!(f, "qwen3.5"),
            Self::Qwen36Moe => write!(f, "qwen3.6-moe"),
            Self::Gemma4 => write!(f, "gemma4"),
            Self::Phi4 => write!(f, "phi4"),
            Self::Llama31 => write!(f, "llama3.1"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelVariant {
    Qwen3_5_0_8B,
    Qwen3_5_2B,
    Qwen3_5_4B,
    Qwen3_5_9B,
    Qwen3_6_27B,
    Qwen3_6_35B_A3B,
    Gemma4_E2B,
    Gemma4_E4B,
    Phi4_Mini,
    Llama3_1_8B,
}

impl ModelVariant {
    pub fn from_cli_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "qwen3.5-0.8b" | "qwen35-0.8b" | "0.8b" => Some(Self::Qwen3_5_0_8B),
            "qwen3.5-2b" | "qwen35-2b" | "2b" => Some(Self::Qwen3_5_2B),
            "qwen3.5-4b" | "qwen35-4b" | "4b" => Some(Self::Qwen3_5_4B),
            "qwen3.5-9b" | "qwen35-9b" | "9b" => Some(Self::Qwen3_5_9B),
            "qwen3.6-27b" | "qwen36-27b" | "qwen3.6-27b-fp8" | "qwen36-27b-fp8" => {
                Some(Self::Qwen3_6_27B)
            }
            "qwen3.6-35b-a3b" | "qwen36-35b-a3b" | "qwen3.6-35b-a3b-fp8" | "qwen36-35b-a3b-fp8" => {
                Some(Self::Qwen3_6_35B_A3B)
            }
            "gemma4-e2b" | "gemma-4-e2b" | "e2b" => Some(Self::Gemma4_E2B),
            "gemma4-e4b" | "gemma-4-e4b" | "e4b" => Some(Self::Gemma4_E4B),
            "phi4-mini" | "phi-4-mini" | "phi4mini" => Some(Self::Phi4_Mini),
            "llama3.1-8b" | "llama31-8b" | "meta-llama-3.1-8b" => Some(Self::Llama3_1_8B),
            _ => None,
        }
    }

    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::Qwen3_5_0_8B => "Qwen/Qwen3.5-0.8B",
            Self::Qwen3_5_2B => "Qwen/Qwen3.5-2B",
            Self::Qwen3_5_4B => "Qwen/Qwen3.5-4B",
            Self::Qwen3_5_9B => "Qwen/Qwen3.5-9B",
            Self::Qwen3_6_27B => "Qwen/Qwen3.6-27B-FP8",
            Self::Qwen3_6_35B_A3B => "Qwen/Qwen3.6-35B-A3B-FP8",
            Self::Gemma4_E2B => "google/gemma-4-E2B",
            Self::Gemma4_E4B => "google/gemma-4-E4B",
            Self::Phi4_Mini => "microsoft/Phi-4-mini-instruct",
            Self::Llama3_1_8B => "NousResearch/Meta-Llama-3.1-8B",
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen3_5_0_8B
            | Self::Qwen3_5_2B
            | Self::Qwen3_5_4B
            | Self::Qwen3_5_9B
            | Self::Qwen3_6_27B => ModelFamily::Qwen35,
            Self::Qwen3_6_35B_A3B => ModelFamily::Qwen36Moe,
            Self::Gemma4_E2B | Self::Gemma4_E4B => ModelFamily::Gemma4,
            Self::Phi4_Mini => ModelFamily::Phi4,
            Self::Llama3_1_8B => ModelFamily::Llama31,
        }
    }
}

impl fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3_5_0_8B => write!(f, "qwen3.5-0.8b"),
            Self::Qwen3_5_2B => write!(f, "qwen3.5-2b"),
            Self::Qwen3_5_4B => write!(f, "qwen3.5-4b"),
            Self::Qwen3_5_9B => write!(f, "qwen3.5-9b"),
            Self::Qwen3_6_27B => write!(f, "qwen3.6-27b"),
            Self::Qwen3_6_35B_A3B => write!(f, "qwen3.6-35b-a3b"),
            Self::Gemma4_E2B => write!(f, "gemma4-e2b"),
            Self::Gemma4_E4B => write!(f, "gemma4-e4b"),
            Self::Phi4_Mini => write!(f, "phi4-mini"),
            Self::Llama3_1_8B => write!(f, "llama3.1-8b"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArch {
    Gfx1150,
    Sm86,
    AppleM4,
    Unknown(String),
}

impl GpuArch {
    pub fn from_backend_name(backend: &Backend, name: &str) -> Self {
        match backend {
            Backend::Hip => match name.trim() {
                "gfx1150" => Self::Gfx1150,
                other => Self::Unknown(other.to_owned()),
            },
            Backend::Cuda => match name.trim() {
                "sm86" => Self::Sm86,
                other => Self::Unknown(other.to_owned()),
            },
            Backend::Metal => match name.trim() {
                "apple-m4" => Self::AppleM4,
                other => Self::Unknown(other.to_owned()),
            },
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gfx1150 => write!(f, "gfx1150"),
            Self::Sm86 => write!(f, "sm86"),
            Self::AppleM4 => write!(f, "apple-m4"),
            Self::Unknown(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Qwen35KernelParams {
    pub proj_buf_floats: usize,
    pub attn_scratch_floats: usize,
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
    pub use_4b_kernel: bool,
    /// Per-model launch preset for the HIP 4B persistent decode kernel.
    /// `None` (default) keeps the non-cooperative 2x multiProcessorCount
    /// grid — empirically safe across every tested variant. `Some((blocks,
    /// cooperative))` installs a different grid size + cooperative-launch
    /// flag via `kernel_ffi::set_qwen35_4b_launch_preset`; this is how
    /// models opt into the larger grids that only stay hang-free when
    /// co-residence is enforced by `hipLaunchCooperativeKernel`. User env
    /// vars `SUPERSONIC_QWEN4B_BLOCKS` / `_COOP` still override any preset.
    pub hip_launch_preset: Option<(i32, bool)>,
}

pub struct Gemma4KernelParams {
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
}

#[derive(Clone, Copy)]
pub struct Phi4KernelParams {
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
}

#[derive(Clone, Copy)]
pub struct Llama31KernelParams {
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
}

pub enum FamilyParams {
    Qwen35(Qwen35KernelParams),
    Gemma4(Gemma4KernelParams),
    Phi4(Phi4KernelParams),
    Llama31(Llama31KernelParams),
}

pub struct VramBudget {
    pub fixed_bytes: u64,
    pub overhead_factor: f64,
}

impl VramBudget {
    pub fn estimate_total(&self, context_tokens: usize, kv_bytes_per_token: u64) -> u64 {
        let kv_bytes = kv_bytes_per_token * context_tokens as u64;
        ((self.fixed_bytes + kv_bytes) as f64 * self.overhead_factor) as u64
    }
}

pub struct RegistryEntry {
    pub model: ModelVariant,
    pub backend: Backend,
    pub arch: GpuArch,
    pub vram: VramBudget,
    pub params: FamilyParams,
}

const GIB: u64 = 1024 * 1024 * 1024;

static REGISTRY: &[RegistryEntry] = &[
    RegistryEntry {
        model: ModelVariant::Qwen3_5_0_8B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 2 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            // 4B kernel's attn_scratch holds saved_q+saved_gate+saved_pre_gate+saved_scores.
            // Needs 3*nh*hd + nh*kv_max_t floats; keep in sync with other 4B-kernel entries.
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            // 0.8B on HIP permanently runs through the 4B persistent megakernel.
            // The dedicated 0.8B kernel (full_attention.hip) was deleted: it had
            // no INT4/FP8 path, was ~2.8x slower than the 4B-routed path, and
            // the BF16 page-fault + hipcc codegen sensitivity warnings were
            // both found stale in the 2026-04-20 diagnostic pass.
            use_4b_kernel: true,
            // Cooperative launch at 32 blocks caps conservatively at 24 on
            // 0.8B's 14 KB LDS and runs at 77 ms/tok vs. the non-coop 2x
            // default's 91 ms/tok (measured 2026-04-20). Other variants
            // see no gain because their higher LDS usage caps the coop
            // grid at or below 2x — they stay on the default path.
            hip_launch_preset: Some((32, true)),
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_2B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 5 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_4B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 10 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_9B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 18 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_0_8B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 2 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: false,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_2B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 5 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_0_8B,
        backend: Backend::Metal,
        arch: GpuArch::AppleM4,
        vram: VramBudget {
            fixed_bytes: 4 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: false,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_2B,
        backend: Backend::Metal,
        arch: GpuArch::AppleM4,
        vram: VramBudget {
            fixed_bytes: 5 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: false,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_4B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 10 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_9B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 18 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 16384,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_6_27B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 60 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            // Qwen3.6-27B: qkv 8192 + z 6144 + b/a 48 each.
            proj_buf_floats: 16480,
            // Floor for 3*nh*hd + nh*aligned_kv_t at short contexts.
            // The runner still expands this from --context-size.
            attn_scratch_floats: 24576,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
            hip_launch_preset: None,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Gemma4_E2B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 11 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Gemma4(Gemma4KernelParams {
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Gemma4_E4B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 10 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Gemma4(Gemma4KernelParams {
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
        }),
    },
    // Phi-4-mini: 3.8B dense, full-attention all 32 layers, LongRoPE.
    // BF16 weights ≈ 7.6 GiB; KV cache at 4K ctx ≈ 520 MiB (32 layers × 2 × 8 kv_heads × 128 head_dim × 2 B).
    // Fits gfx1150 with headroom. Weight prefix is bare `model` (Phi stores tensors as `model.*`).
    RegistryEntry {
        model: ModelVariant::Phi4_Mini,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 8 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Phi4(Phi4KernelParams {
            weight_prefix: "model",
            kv_chunk_size: 256,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Llama3_1_8B,
        backend: Backend::Cuda,
        arch: GpuArch::Sm86,
        vram: VramBudget {
            fixed_bytes: 18 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Llama31(Llama31KernelParams {
            weight_prefix: "model",
            kv_chunk_size: 256,
        }),
    },
];

pub fn lookup(
    model: &ModelVariant,
    backend: &Backend,
    arch: &GpuArch,
) -> Option<&'static RegistryEntry> {
    REGISTRY
        .iter()
        .find(|e| e.model == *model && e.backend == *backend && e.arch == *arch)
}

pub fn supported_models_list() -> Vec<&'static str> {
    let mut models: Vec<&str> = REGISTRY
        .iter()
        .map(|e| match &e.model {
            ModelVariant::Qwen3_5_0_8B => "qwen3.5-0.8b",
            ModelVariant::Qwen3_5_2B => "qwen3.5-2b",
            ModelVariant::Qwen3_5_4B => "qwen3.5-4b",
            ModelVariant::Qwen3_5_9B => "qwen3.5-9b",
            ModelVariant::Qwen3_6_27B => "qwen3.6-27b",
            ModelVariant::Qwen3_6_35B_A3B => "qwen3.6-35b-a3b",
            ModelVariant::Gemma4_E2B => "gemma4-e2b",
            ModelVariant::Gemma4_E4B => "gemma4-e4b",
            ModelVariant::Phi4_Mini => "phi4-mini",
            ModelVariant::Llama3_1_8B => "llama3.1-8b",
        })
        .collect();
    models.sort_unstable();
    models.dedup();
    models
}

pub fn supported_archs_for(model: &ModelVariant, backend: &Backend) -> Vec<String> {
    REGISTRY
        .iter()
        .filter(|e| e.model == *model && e.backend == *backend)
        .map(|e| e.arch.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_sm86_qwen_registry_includes_2b_and_9b() {
        assert!(lookup(&ModelVariant::Qwen3_5_2B, &Backend::Cuda, &GpuArch::Sm86,).is_some());
        assert!(lookup(&ModelVariant::Qwen3_5_9B, &Backend::Cuda, &GpuArch::Sm86,).is_some());
    }

    #[test]
    fn cuda_sm86_qwen36_27b_registry_params_match_geometry() {
        let entry = lookup(&ModelVariant::Qwen3_6_27B, &Backend::Cuda, &GpuArch::Sm86)
            .expect("qwen3.6-27b CUDA sm86 registry entry");
        match entry.params {
            FamilyParams::Qwen35(params) => {
                assert_eq!(params.weight_prefix, "model.language_model");
                assert!(params.use_4b_kernel);
                assert_eq!(params.proj_buf_floats, 16_480);
                assert_eq!(params.attn_scratch_floats, 24_576);
            }
            _ => panic!("qwen3.6-27b must use the Qwen hybrid-attention engine"),
        }
    }

    #[test]
    fn qwen36_aliases_are_public_and_canonical() {
        assert_eq!(
            ModelVariant::from_cli_str("qwen36-27b-fp8"),
            Some(ModelVariant::Qwen3_6_27B)
        );
        assert_eq!(
            ModelVariant::from_cli_str("qwen36-35b-a3b-fp8"),
            Some(ModelVariant::Qwen3_6_35B_A3B)
        );
        assert_eq!(ModelVariant::Qwen3_6_27B.to_string(), "qwen3.6-27b");
        assert_eq!(
            ModelVariant::Qwen3_6_35B_A3B.family(),
            ModelFamily::Qwen36Moe
        );
        assert_eq!(ModelVariant::Qwen3_6_35B_A3B.to_string(), "qwen3.6-35b-a3b");
        assert!(supported_models_list().contains(&"qwen3.6-27b"));
    }

    #[test]
    fn metal_apple_m4_registry_includes_08b_and_2b() {
        let e08b = lookup(
            &ModelVariant::Qwen3_5_0_8B,
            &Backend::Metal,
            &GpuArch::AppleM4,
        );
        let e2b = lookup(
            &ModelVariant::Qwen3_5_2B,
            &Backend::Metal,
            &GpuArch::AppleM4,
        );
        assert!(e08b.is_some());
        assert!(e2b.is_some());
        let p08b = match &e08b.unwrap().params {
            FamilyParams::Qwen35(p) => p,
            _ => panic!("wrong family"),
        };
        let p2b = match &e2b.unwrap().params {
            FamilyParams::Qwen35(p) => p,
            _ => panic!("wrong family"),
        };
        assert!(!p08b.use_4b_kernel);
        assert!(!p2b.use_4b_kernel);
    }
}
