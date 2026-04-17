use std::fmt;

/// Family of architectures handled by the same code path.
/// Used to dispatch between per-family config parsers, weight loaders, and kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen35,
    Gemma4,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen35 => write!(f, "qwen3.5"),
            Self::Gemma4 => write!(f, "gemma4"),
        }
    }
}

/// Identifies a specific model variant with a known optimized megakernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelVariant {
    Qwen3_5_0_8B,
    Qwen3_5_2B,
    Qwen3_5_4B,
    Qwen3_5_9B,
    Gemma4_E2B,
    Gemma4_E4B,
}

impl ModelVariant {
    /// Parse from CLI string (case-insensitive).
    pub fn from_cli_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "qwen3.5-0.8b" | "qwen35-0.8b" | "0.8b" => Some(Self::Qwen3_5_0_8B),
            "qwen3.5-2b" | "qwen35-2b" | "2b" => Some(Self::Qwen3_5_2B),
            "qwen3.5-4b" | "qwen35-4b" | "4b" => Some(Self::Qwen3_5_4B),
            "qwen3.5-9b" | "qwen35-9b" | "9b" => Some(Self::Qwen3_5_9B),
            "gemma4-e2b" | "gemma-4-e2b" | "e2b" => Some(Self::Gemma4_E2B),
            "gemma4-e4b" | "gemma-4-e4b" | "e4b" => Some(Self::Gemma4_E4B),
            _ => None,
        }
    }

    /// Canonical HuggingFace model ID (used as oracle default).
    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::Qwen3_5_0_8B => "Qwen/Qwen3.5-0.8B",
            Self::Qwen3_5_2B => "Qwen/Qwen3.5-2B",
            Self::Qwen3_5_4B => "Qwen/Qwen3.5-4B",
            Self::Qwen3_5_9B => "Qwen/Qwen3.5-9B",
            Self::Gemma4_E2B => "google/gemma-4-E2B",
            Self::Gemma4_E4B => "google/gemma-4-E4B",
        }
    }

    /// Which family this variant belongs to.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen3_5_0_8B | Self::Qwen3_5_2B | Self::Qwen3_5_4B | Self::Qwen3_5_9B => {
                ModelFamily::Qwen35
            }
            Self::Gemma4_E2B | Self::Gemma4_E4B => ModelFamily::Gemma4,
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
            Self::Gemma4_E2B => write!(f, "gemma4-e2b"),
            Self::Gemma4_E4B => write!(f, "gemma4-e4b"),
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

/// Kernel-specific parameters for Qwen3.5-family variants.
pub struct Qwen35KernelParams {
    /// Max projection output buffer size in floats (kernel: proj_buf).
    pub proj_buf_floats: usize,
    /// Attention/recurrent scratch buffer size in floats (kernel: attn_scratch).
    pub attn_scratch_floats: usize,
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
    /// Use the 4B kernel variant (separate compilation for hipcc compatibility).
    pub use_4b_kernel: bool,
}

/// Kernel-specific parameters for Gemma 4 dense variants (E2B, E4B).
/// Fields will grow as the kernel design stabilizes; for scaffolding we carry
/// only what the weight loader and (future) bake path will need.
pub struct Gemma4KernelParams {
    pub weight_prefix: &'static str,
    pub kv_chunk_size: usize,
}

/// Per-family parameter bundle. Code paths downstream match on the variant
/// and extract only the parameters relevant to their family.
pub enum FamilyParams {
    Qwen35(Qwen35KernelParams),
    Gemma4(Gemma4KernelParams),
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
    pub params: FamilyParams,
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
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 2048,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: false,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_2B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            // ~3.7 GiB weights (BF16) + scratch + activations + buffers
            fixed_bytes: 5 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 8224,
            attn_scratch_floats: 2048,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
        }),
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
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 4096,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
        }),
    },
    RegistryEntry {
        model: ModelVariant::Qwen3_5_9B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            // ~16.8 GiB weights (BF16) + scratch + activations + buffers
            // Only fits with --fp8-runtime (~9.4 GiB effective)
            fixed_bytes: 18 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Qwen35(Qwen35KernelParams {
            proj_buf_floats: 12352,
            attn_scratch_floats: 4096,
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
            use_4b_kernel: true,
        }),
    },
    // --- Gemma 4 ---
    // Scaffolding-only: kernel not yet implemented. Registry entry exists so
    // the CLI can parse the variant, config loading can be exercised, and
    // VRAM bookkeeping has a place to live when the kernel lands.
    RegistryEntry {
        model: ModelVariant::Gemma4_E2B,
        backend: Backend::Hip,
        arch: GpuArch::Gfx1150,
        vram: VramBudget {
            // ~10 GiB weights (BF16, includes ~4.7 GiB PLE pathway) + scratch + activations
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
            // ~16 GiB weights (BF16) — does not fit in BF16, requires INT4 (~4 GiB)
            // or FP8 (~8 GiB). Left at BF16 sizing so VRAM check flags the fit
            // problem until INT4/FP8 paths land for Gemma 4.
            fixed_bytes: 16 * GIB,
            overhead_factor: 1.1,
        },
        params: FamilyParams::Gemma4(Gemma4KernelParams {
            weight_prefix: "model.language_model",
            kv_chunk_size: 256,
        }),
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
            ModelVariant::Qwen3_5_2B => "qwen3.5-2b",
            ModelVariant::Qwen3_5_4B => "qwen3.5-4b",
            ModelVariant::Qwen3_5_9B => "qwen3.5-9b",
            ModelVariant::Gemma4_E2B => "gemma4-e2b",
            ModelVariant::Gemma4_E4B => "gemma4-e4b",
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
