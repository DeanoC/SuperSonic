use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Hip,
    Cuda,
    Metal,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hip" => Some(Self::Hip),
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Hip => 1,
            Self::Cuda => 2,
            Self::Metal => 3,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Hip),
            2 => Some(Self::Cuda),
            3 => Some(Self::Metal),
            _ => None,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip => write!(f, "HIP"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Metal => write!(f, "Metal"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub arch_name: String,
    pub total_vram_bytes: u64,
    pub warp_size: u32,
    pub clock_rate_khz: u32,
}

/// How a GPU's memory is wired relative to host RAM.
///
/// Drives allocation/copy policy in `gpu-hal`. `Discrete` (default) keeps the
/// classic `hipMalloc` / `cudaMalloc` device-pointer path. `Unified` switches
/// HIP allocations to mapped+coherent host pages so host and device address
/// the same physical bytes — the right shape for APUs (gfx1150) and Apple
/// M-series, where there is no separate VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryArchitecture {
    Discrete,
    Unified,
}

impl MemoryArchitecture {
    fn code(self) -> u8 {
        match self {
            Self::Discrete => 1,
            Self::Unified => 2,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            2 => Self::Unified,
            // 0 (unset) and 1 both mean Discrete — preserves pre-wiring behavior.
            _ => Self::Discrete,
        }
    }
}

/// Caller intent passed to `GpuBuffer::*_with_kind` constructors.
///
/// `Persistent` covers anything the GPU re-reads across kernel launches —
/// weights, KV cache, activations, layer descriptor arrays, etc. These need
/// GPU L2 cache to engage; the active `BufferPolicy` resolves them to a
/// strategy that preserves cacheability (always `Default` today, on every
/// platform).
///
/// `Scratch` covers one-shot staging memory the GPU touches once and
/// discards. On platforms where the GPU genuinely caches host-mapped memory
/// (Apple silicon, possibly future RDNA4 laptops), the policy may map both
/// kinds to the same strategy. On gfx1150 (RDNA3.5 APU), `Scratch` maps to
/// `HostMapped` to skip the H2D copy — the L2 bypass cost doesn't matter
/// when there's no reuse to lose. See `docs/gfx1150-l2-bypass.md`.
///
/// Default for any caller that doesn't care: `Persistent`. Mis-tagging a
/// re-read buffer as `Scratch` is a perf foot-gun on gfx1150 and a no-op
/// elsewhere; mis-tagging a write-once buffer as `Persistent` is correct
/// but slightly wasteful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BufferKind {
    #[default]
    Persistent,
    Scratch,
}

/// Mechanism a `BufferPolicy` resolves a `BufferKind` to.
///
/// `Default` is always-safe: classic `hipMalloc` / `cudaMalloc` / metal
/// device memory. `HostMapped` is HIP-only and means
/// `hipHostMalloc(MAPPED) + hipHostGetDevicePointer` — saves the H2D copy
/// on APUs but bypasses GPU L2 on RDNA3.5 (see investigation doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocStrategy {
    #[default]
    Default,
    HostMapped,
}

impl AllocStrategy {
    fn code(self) -> u8 {
        match self {
            Self::Default => 1,
            Self::HostMapped => 2,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            2 => Self::HostMapped,
            // 0 (unset) and 1 both map to Default. Preserves classic alloc
            // behavior for any path that runs before startup wiring.
            _ => Self::Default,
        }
    }
}

/// Per-platform table mapping `BufferKind` to `AllocStrategy`.
///
/// Owned by the runner registry's `ArchProfile` and installed once at
/// startup via `set_buffer_policy`. Default is `{Default, Default}` — every
/// kind routes through the classic allocator. Platforms where a non-default
/// strategy actually wins flip the relevant entry.
///
/// Today's table:
///   - gfx1150 (RDNA3.5 APU)            : `{Persistent: Default, Scratch: HostMapped}`
///   - gfx1100 (RDNA3 dGPU), sm86       : `{Default, Default}` — no host-mapped path benefit
///   - apple-m4 (Metal)                 : `{Default, Default}` — Metal owns the unified-memory wiring; HIP enums don't apply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferPolicy {
    pub persistent: AllocStrategy,
    pub scratch: AllocStrategy,
}

impl BufferPolicy {
    pub const fn all_default() -> Self {
        Self {
            persistent: AllocStrategy::Default,
            scratch: AllocStrategy::Default,
        }
    }

    pub fn strategy_for(self, kind: BufferKind) -> AllocStrategy {
        match kind {
            BufferKind::Persistent => self.persistent,
            BufferKind::Scratch => self.scratch,
        }
    }
}

static DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(0);
static DEFAULT_MEMORY_ARCHITECTURE: AtomicU8 = AtomicU8::new(0);
static POLICY_PERSISTENT: AtomicU8 = AtomicU8::new(0);
static POLICY_SCRATCH: AtomicU8 = AtomicU8::new(0);

pub fn compiled_backends() -> Vec<Backend> {
    let mut backends = Vec::new();
    #[cfg(supersonic_backend_hip)]
    backends.push(Backend::Hip);
    #[cfg(supersonic_backend_cuda)]
    backends.push(Backend::Cuda);
    #[cfg(supersonic_backend_metal)]
    backends.push(Backend::Metal);
    backends
}

pub fn is_backend_compiled(backend: Backend) -> bool {
    compiled_backends().contains(&backend)
}

pub fn set_backend(backend: Backend) {
    DEFAULT_BACKEND.store(backend.code(), Ordering::Relaxed);
}

pub fn current_backend() -> Backend {
    if let Some(backend) = Backend::from_code(DEFAULT_BACKEND.load(Ordering::Relaxed)) {
        return backend;
    }
    compiled_backends()
        .into_iter()
        .next()
        .expect("no GPU backends compiled")
}

/// Set the active memory architecture. Called once at startup after
/// `set_backend`, typically from `ArchProfile::for_arch(...).memory`.
pub fn set_memory_architecture(arch: MemoryArchitecture) {
    DEFAULT_MEMORY_ARCHITECTURE.store(arch.code(), Ordering::Relaxed);
}

/// Read the active memory architecture. Defaults to `Discrete` until a
/// `set_memory_architecture` call lands — preserves classic alloc behavior
/// for any code path that runs before startup wiring.
pub fn current_memory_architecture() -> MemoryArchitecture {
    MemoryArchitecture::from_code(DEFAULT_MEMORY_ARCHITECTURE.load(Ordering::Relaxed))
}

/// Install the active `BufferPolicy`. Called once at startup from the runner
/// after `ArchProfile::for_arch(...)` has resolved the per-arch table.
pub fn set_buffer_policy(policy: BufferPolicy) {
    POLICY_PERSISTENT.store(policy.persistent.code(), Ordering::Relaxed);
    POLICY_SCRATCH.store(policy.scratch.code(), Ordering::Relaxed);
}

/// Read the active `BufferPolicy`. Defaults to `BufferPolicy::all_default()`
/// (every kind → `Default`) until `set_buffer_policy` has been called.
pub fn current_buffer_policy() -> BufferPolicy {
    BufferPolicy {
        persistent: AllocStrategy::from_code(POLICY_PERSISTENT.load(Ordering::Relaxed)),
        scratch: AllocStrategy::from_code(POLICY_SCRATCH.load(Ordering::Relaxed)),
    }
}

/// Resolve a `BufferKind` to its `AllocStrategy` using the currently
/// installed policy. Convenience for hot-path callers in `ops::alloc`.
pub fn current_strategy_for(kind: BufferKind) -> AllocStrategy {
    current_buffer_policy().strategy_for(kind)
}
