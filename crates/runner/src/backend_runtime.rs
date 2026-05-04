use anyhow::Result;
use supersonic_core::backend::{compiled_backends_display, BackendChoice};

use crate::registry::{self, Backend, GpuArch, ModelVariant, RegistryEntry};

pub(crate) struct GpuInfo {
    #[allow(dead_code)]
    pub(crate) arch_name: String,
    pub(crate) gpu_arch: GpuArch,
    #[cfg_attr(feature = "bughunt", allow(dead_code))]
    pub(crate) total_vram: u64,
}

pub(crate) fn resolve_backend(choice: BackendChoice, ordinal: usize) -> Result<Backend> {
    match choice {
        BackendChoice::Explicit(backend) => {
            if !gpu_hal::is_backend_compiled(backend) {
                anyhow::bail!(
                    "Requested backend {backend} is not compiled into this build. Compiled backends: [{}]",
                    compiled_backends_display()
                );
            }
            Ok(backend)
        }
        BackendChoice::Auto => {
            if gpu_hal::is_backend_compiled(Backend::Cuda)
                && gpu_hal::query_device_info(Backend::Cuda, ordinal).is_ok()
            {
                return Ok(Backend::Cuda);
            }
            if gpu_hal::is_backend_compiled(Backend::Hip)
                && kernel_ffi::query_gpu_info(ordinal).is_ok()
            {
                return Ok(Backend::Hip);
            }
            if gpu_hal::is_backend_compiled(Backend::Metal)
                && gpu_hal::query_device_info(Backend::Metal, ordinal).is_ok()
            {
                return Ok(Backend::Metal);
            }
            anyhow::bail!(
                "No usable GPU backend available for device {ordinal}. Compiled backends: [{}]",
                compiled_backends_display()
            )
        }
    }
}

pub(crate) fn resolve_oracle_device(spec: &str, backend: Backend, ordinal: usize) -> String {
    match spec.trim().to_ascii_lowercase().as_str() {
        "auto" => match backend {
            Backend::Cuda => format!("cuda:{ordinal}"),
            Backend::Hip => "cpu".to_string(),
            Backend::Metal => "cpu".to_string(),
        },
        other => other.to_string(),
    }
}

pub(crate) fn query_gpu_info(backend: Backend, ordinal: usize) -> Result<GpuInfo> {
    let (arch_name, total_vram, warp_size) = match backend {
        Backend::Hip => {
            let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            let base_arch = arch_name.split(':').next().unwrap_or(&arch_name);
            let warp_size = if base_arch.starts_with("gfx9") {
                64
            } else {
                32
            };
            (arch_name, total_vram, warp_size)
        }
        Backend::Cuda | Backend::Metal => {
            let info = gpu_hal::query_device_info(backend, ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
    };
    let gpu_arch = GpuArch::from_backend_name(&backend, &arch_name);
    eprintln!(
        "[gpu] backend={backend} device={ordinal} arch={arch_name} warp={} vram={:.1}GiB",
        warp_size,
        total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(GpuInfo {
        arch_name,
        gpu_arch,
        total_vram,
    })
}

pub(crate) fn lookup_registry_entry(
    model_variant: &ModelVariant,
    backend: Backend,
    gpu_arch: &GpuArch,
    allow_untested_gpu: Option<&str>,
) -> Result<&'static RegistryEntry> {
    if let Some(entry) = registry::lookup(model_variant, &backend, gpu_arch) {
        return Ok(entry);
    }

    if let Some(override_arch) = allow_untested_gpu {
        let reuse_arch = GpuArch::from_backend_name(&backend, override_arch);
        let entry = registry::lookup(model_variant, &backend, &reuse_arch).ok_or_else(|| {
            let supported_archs = registry::supported_archs_for(model_variant, &backend);
            anyhow::anyhow!(
                "--allow-untested-gpu={override_arch}: no registry entry for \
                 model={model_variant} backend={backend} arch={reuse_arch}. \
                 Pass one of: [{}]",
                supported_archs.join(", ")
            )
        })?;
        eprintln!(
            "[gpu] WARNING: detected arch={gpu_arch} has no registry entry; \
             reusing {reuse_arch} kernel as requested by --allow-untested-gpu. \
             Correctness is not guaranteed."
        );
        return Ok(entry);
    }

    let supported_archs = registry::supported_archs_for(model_variant, &backend);
    anyhow::bail!(
        "No optimized kernel for model={model_variant} backend={backend} arch={gpu_arch}. \
         Supported GPU architectures for this model: [{}]. \
         To force-reuse another arch's kernel, pass --allow-untested-gpu=<arch>.",
        supported_archs.join(", ")
    );
}

#[cfg_attr(feature = "bughunt", allow(dead_code))]
pub(crate) fn install_arch_profile(entry: &RegistryEntry) {
    // Install per-arch policy so gpu_hal::alloc dispatches correctly.
    // `MemoryArchitecture` is informational (used downstream for VRAM
    // budgeting on APUs); `BufferPolicy` maps caller-side `BufferKind`
    // intent to the actual `AllocStrategy`. Persistent always uses the
    // classic device allocator (GPU-cacheable); Scratch may opt into
    // host-mapped on arches where that's a win - today only gfx1150.
    // Must be set before any GpuBuffer::alloc, which starts during weight
    // loading below.
    let arch_profile = registry::ArchProfile::for_arch(&entry.arch);
    gpu_hal::set_memory_architecture(arch_profile.memory);
    gpu_hal::set_buffer_policy(arch_profile.buffer_policy);
    eprintln!(
        "[gpu] memory={:?}, buffer_policy: persistent={} scratch={}",
        arch_profile.memory,
        strategy_label(arch_profile.buffer_policy.persistent),
        strategy_label(arch_profile.buffer_policy.scratch),
    );
}

#[cfg_attr(feature = "bughunt", allow(dead_code))]
fn strategy_label(s: gpu_hal::AllocStrategy) -> &'static str {
    match s {
        gpu_hal::AllocStrategy::Default => "hipMalloc / cudaMalloc / metal",
        gpu_hal::AllocStrategy::HostMapped => "hipHostMalloc(MAPPED) + GetDevicePointer",
    }
}
