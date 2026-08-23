#![allow(non_camel_case_types)]

use std::fmt;

pub use gpu_hal::{AllocStrategy, Backend, BufferPolicy, MemoryArchitecture};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen35,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen35 => write!(f, "qwen3.8"),
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
        match s.to_ascii_lowercase().as_str() {
            "qwen3.8-27b" | "qwen38-27b" | "qwen3.8-27b-fp8" | "qwen38-27b-fp8" => {
                Some(Self::Qwen3_8_27B)
            }
            _ => None,
        }
    }

    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::Qwen3_8_27B => "Qwen/Qwen3.8-27B",
        }
    }

    pub fn family(&self) -> ModelFamily {
        ModelFamily::Qwen35
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
    Gfx1150,
    Gfx1201,
    Gfx942,
    Unknown(String),
}

impl GpuArch {
    pub fn from_backend_name(backend: &Backend, name: &str) -> Self {
        match backend {
            Backend::Hip => match name.trim().split(':').next().unwrap_or(name.trim()) {
                "gfx1100" => Self::Gfx1100,
                "gfx1150" => Self::Gfx1150,
                "gfx1201" => Self::Gfx1201,
                "gfx942" => Self::Gfx942,
                other => Self::Unknown(other.to_owned()),
            },
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gfx1100 => write!(f, "gfx1100"),
            Self::Gfx1150 => write!(f, "gfx1150"),
            Self::Gfx1201 => write!(f, "gfx1201"),
            Self::Gfx942 => write!(f, "gfx942"),
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
    pub fn for_arch(arch: &GpuArch) -> Self {
        let memory = match arch {
            GpuArch::Gfx1150 => MemoryArchitecture::Unified,
            _ => MemoryArchitecture::Discrete,
        };
        let buffer_policy = match arch {
            GpuArch::Gfx1150 => BufferPolicy {
                persistent: AllocStrategy::Default,
                scratch: AllocStrategy::HostMapped,
            },
            _ => BufferPolicy::all_default(),
        };
        Self {
            memory,
            buffer_policy,
        }
    }
}

pub fn qwen35_4b_launch_preset(arch: &GpuArch, model: &ModelVariant) -> Option<(i32, bool)> {
    match (arch, model) {
        (GpuArch::Gfx1150, ModelVariant::Qwen3_8_27B) => Some((32, true)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub struct Qwen35KernelParams {
    pub proj_buf_floats: usize,
    pub attn_scratch_floats: usize,
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
    pub use_4b_kernel: bool,
}

pub enum FamilyParams {
    Qwen35(Qwen35KernelParams),
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
        params: FamilyParams::Qwen35(Qwen35KernelParams {
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
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            fixed_bytes: 22 * GIB,
            overhead_factor: 1.05,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
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
        params: FamilyParams::Qwen35(Qwen35KernelParams {
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
        arch: GpuArch::Gfx942,
        vram: VramBudget {
            fixed_bytes: 22 * GIB,
            overhead_factor: 1.05,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
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
    fn qwen38_registry_is_hip_only() {
        let model = ModelVariant::from_cli_str("qwen38-27b").unwrap();
        assert_eq!(model, ModelVariant::Qwen3_8_27B);
        assert_eq!(model.family(), ModelFamily::Qwen35);
        assert_eq!(
            model.architecture_family(),
            ArchitectureFamily::QwenHybridDense
        );
        assert_eq!(supported_models_list(), vec!["qwen3.8-27b"]);
        assert_eq!(
            supported_archs_for(&model, &Backend::Hip),
            vec!["gfx1100", "gfx1150", "gfx1201", "gfx942"]
        );
    }
}
