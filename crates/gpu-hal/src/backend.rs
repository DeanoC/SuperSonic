use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Hip,
    Cuda,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hip" => Some(Self::Hip),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Hip => 1,
            Self::Cuda => 2,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Hip),
            2 => Some(Self::Cuda),
            _ => None,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip => write!(f, "HIP"),
            Self::Cuda => write!(f, "CUDA"),
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

static DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(0);

pub fn compiled_backends() -> Vec<Backend> {
    let mut backends = Vec::new();
    #[cfg(supersonic_backend_hip)]
    backends.push(Backend::Hip);
    #[cfg(supersonic_backend_cuda)]
    backends.push(Backend::Cuda);
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
