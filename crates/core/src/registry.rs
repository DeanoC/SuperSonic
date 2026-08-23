#![allow(non_camel_case_types)]

use std::fmt;

pub use gpu_hal::{AllocStrategy, Backend, BufferPolicy, MemoryArchitecture};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen38,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen38 => write!(f, "qwen3.8"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureFamily {
    QwenHybridDense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    Qwen3_8_27B,
}

impl ModelVariant {
    pub fn from_cli_str(s: &str) -> Option<Self> {
        match s {
            "qwen3.8-27b" => Some(Self::Qwen3_8_27B),
            _ => None,
        }
    }

    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::Qwen3_8_27B => "Qwen/Qwen3.8-27B",
        }
    }

    pub fn family(&self) -> ModelFamily {
        ModelFamily::Qwen38
    }

    pub fn architecture_family(&self) -> ArchitectureFamily {
        ArchitectureFamily::QwenHybridDense
    }
}

impl fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3_8_27B => write!(f, "qwen3.8-27b"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArch {
    Gfx1100,
    Gfx1201,
    Unknown(String),
}

impl GpuArch {
    pub fn from_backend_name(backend: &Backend, name: &str) -> Self {
        match backend {
            Backend::Hip => match name.trim().split(':').next().unwrap_or(name.trim()) {
                "gfx1100" => Self::Gfx1100,
                "gfx1201" => Self::Gfx1201,
                other => Self::Unknown(other.to_owned()),
            },
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gfx1100 => write!(f, "gfx1100"),
            Self::Gfx1201 => write!(f, "gfx1201"),
            Self::Unknown(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArchProfile {
    pub memory: MemoryArchitecture,
    pub buffer_policy: BufferPolicy,
}

impl ArchProfile {
    pub fn for_arch(_arch: &GpuArch) -> Self {
        Self {
            memory: MemoryArchitecture::Discrete,
            buffer_policy: BufferPolicy::all_default(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Qwen38KernelParams {
    pub proj_buf_floats: usize,
    pub attn_scratch_floats: usize,
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
    pub use_4b_kernel: bool,
}

pub enum FamilyParams {
    Qwen38(Qwen38KernelParams),
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
        model: ModelVariant::Qwen3_8_27B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1100,
        vram: VramBudget {
            fixed_bytes: 22 * GIB,
            overhead_factor: 1.05,
        },
        params: FamilyParams::Qwen38(Qwen38KernelParams {
            proj_buf_floats: 16_480,
            attn_scratch_floats: 24_576,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_8_27B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1201,
        vram: VramBudget {
            fixed_bytes: 22 * GIB,
            overhead_factor: 1.05,
        },
        params: FamilyParams::Qwen38(Qwen38KernelParams {
            proj_buf_floats: 16_480,
            attn_scratch_floats: 24_576,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
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
        .find(|entry| entry.model == *model && entry.backend == *backend && entry.arch == *arch)
}

pub fn supported_models_list() -> Vec<&'static str> {
    vec!["qwen3.8-27b"]
}

pub fn supported_archs_for(model: &ModelVariant, backend: &Backend) -> Vec<String> {
    let mut archs: Vec<_> = REGISTRY
        .iter()
        .filter(|entry| entry.model == *model && entry.backend == *backend)
        .map(|entry| entry.arch.to_string())
        .collect();
    archs.sort();
    archs.dedup();
    archs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_registry_accepts_only_canonical_model_name() {
        let model = ModelVariant::from_cli_str("qwen3.8-27b").unwrap();
        assert_eq!(model, ModelVariant::Qwen3_8_27B);
        for alias in [
            "qwen38-27b",
            "qwen3.8-27b-fp8",
            "qwen38-27b-fp8",
            "Qwen3.8-27B",
            "qwen3.5-0.8b",
        ] {
            assert_eq!(
                ModelVariant::from_cli_str(alias),
                None,
                "unsupported model spelling must be rejected: {alias}"
            );
        }
        assert_eq!(format!("{:?}", model.family()), "Qwen38");
        assert_eq!(
            model.architecture_family(),
            ArchitectureFamily::QwenHybridDense
        );
        assert_eq!(supported_models_list(), vec!["qwen3.8-27b"]);
        assert_eq!(
            supported_archs_for(&model, &Backend::Hip),
            vec!["gfx1100", "gfx1201"]
        );
    }

    #[test]
    fn qwen38_registry_rows_are_hip_only() {
        assert!(REGISTRY.iter().all(|entry| {
            entry.backend == Backend::Hip
                && entry.model == ModelVariant::Qwen3_8_27B
                && format!("{:?}", entry.model.family()) == "Qwen38"
        }));

        let mut archs: Vec<_> = REGISTRY.iter().map(|entry| entry.arch.to_string()).collect();
        archs.sort();
        archs.dedup();
        assert_eq!(archs, vec!["gfx1100", "gfx1201"]);
        for unsupported in ["gfx1150", "gfx942", "sm86"] {
            assert_eq!(
                GpuArch::from_backend_name(&Backend::Hip, unsupported),
                GpuArch::Unknown(unsupported.to_owned())
            );
        }
    }
}
