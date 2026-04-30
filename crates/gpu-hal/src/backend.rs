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

static DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(0);
static DEFAULT_MEMORY_ARCHITECTURE: AtomicU8 = AtomicU8::new(0);

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
