use std::fmt;

/// Identifies a specific model variant with a known optimized megakernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelVariant {
    Qwen3_5_0_8B,
    Qwen3_5_4B,
}

impl ModelVariant {
    /// Parse from CLI string (case-insensitive).
    pub fn from_cli_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "qwen3.5-0.8b" | "qwen35-0.8b" | "0.8b" => Some(Self::Qwen3_5_0_8B),
            "qwen3.5-4b" | "qwen35-4b" | "4b" => Some(Self::Qwen3_5_4B),
            _ => None,
        }
    }

    /// Canonical HuggingFace model ID (used as oracle default).
    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::Qwen3_5_0_8B => "Qwen/Qwen3.5-0.8B",
            Self::Qwen3_5_4B => "Qwen/Qwen3.5-4B",
        }
    }
}

impl fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3_5_0_8B => write!(f, "qwen3.5-0.8b"),
            Self::Qwen3_5_4B => write!(f, "qwen3.5-4b"),
        }
    }
}

/// Compute backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    Hip,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip => write!(f, "HIP"),
        }
    }
}

/// GPU architecture (must match for kernel dispatch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArch {
    Gfx1150,
    Unknown(String),
}

impl GpuArch {
    /// Parse from the gcnArchName string returned by hipGetDeviceProperties.
    pub fn from_rocm_name(name: &str) -> Self {
        match name.trim() {
            "gfx1150" => Self::Gfx1150,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gfx1150 => write!(f, "gfx1150"),
            Self::Unknown(s) => write!(f, "{s}"),
        }
    }
}

/// Kernel-specific parameters tied to a (model, backend, arch) combination.
pub struct KernelParams {
    /// Max projection output buffer size in floats (kernel: proj_buf).
    pub proj_buf_floats: usize,
    /// Attention/recurrent scratch buffer size in floats (kernel: attn_scratch).
    pub attn_scratch_floats: usize,
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
    /// Use the 4B kernel variant (separate compilation for hipcc compatibility).
    pub use_4b_kernel: bool,
}

/// VRAM budget for a specific (model, backend, arch) combination.
pub struct VramBudget {
    /// Fixed VRAM cost in bytes: weights + scratch buffers + activations + overhead.
    /// This is measured/calculated per registry entry and includes everything
    /// except the KV cache (which scales with context length).
    pub fixed_bytes: u64,
    /// Safety margin multiplier (e.g. 1.1 for 10% headroom).
    pub overhead_factor: f64,
}

impl VramBudget {
    /// Estimate total VRAM needed for a given context size.
    /// `kv_bytes_per_token` is computed at runtime from the loaded model config.
    pub fn estimate_total(&self, context_tokens: usize, kv_bytes_per_token: u64) -> u64 {
        let kv_bytes = kv_bytes_per_token * context_tokens as u64;
        ((self.fixed_bytes + kv_bytes) as f64 * self.overhead_factor) as u64
    }
}

/// One supported (model, backend, arch) combination.
pub struct RegistryEntry {
    pub model: ModelVariant,
    pub backend: Backend,
    pub arch: GpuArch,
    pub vram: VramBudget,
    pub params: KernelParams,
}

const GIB: u64 = 1024 * 1024 * 1024;

static REGISTRY: &[RegistryEntry] = &[
    RegistryEntry {
        model: ModelVariant::Qwen3_5_0_8B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            // ~1.6 GiB weights (BF16) + scratch + activations + buffers
            fixed_bytes: 2 * GIB,
            overhead_factor: 1.1,
        },
        params: KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 2048,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: false,
        },
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_4B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            // ~8.8 GiB weights (BF16) + scratch + activations + buffers
            fixed_bytes: 10 * GIB,
            overhead_factor: 1.1,
        },
        params: KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 4096,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
        },
    },
];

/// Find a registry entry for the given combination. Returns None if unsupported.
pub fn lookup(
    model: &ModelVariant,
    backend: &Backend,
    arch: &GpuArch,
) -> Option<&'static RegistryEntry> {
    REGISTRY
        .iter()
        .find(|e| e.model == *model && e.backend == *backend && e.arch == *arch)
}

/// List all supported model names for error messages.
pub fn supported_models_list() -> Vec<&'static str> {
    // Deduplicate model names from registry
    let mut models: Vec<&str> = REGISTRY
        .iter()
        .map(|e| match &e.model {
            ModelVariant::Qwen3_5_0_8B => "qwen3.5-0.8b",
            ModelVariant::Qwen3_5_4B => "qwen3.5-4b",
        })
        .collect();
    models.dedup();
    models
}

/// List all supported arch names for a given model + backend.
pub fn supported_archs_for(model: &ModelVariant, backend: &Backend) -> Vec<String> {
    REGISTRY
        .iter()
        .filter(|e| e.model == *model && e.backend == *backend)
        .map(|e| e.arch.to_string())
        .collect()
}
