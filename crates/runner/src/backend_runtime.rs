use anyhow::Result;

use crate::registry::{self, Backend, GpuArch, ModelVariant, RegistryEntry};

pub(crate) struct GpuInfo {
    pub(crate) gpu_arch: GpuArch,
}

/// Query the selected AMD device. HIP is part of the product contract and is
/// therefore deliberately not selected through a CLI/backend fallback.
pub(crate) fn query_gpu_info(ordinal: usize) -> Result<GpuInfo> {
    let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
        .map_err(|e| anyhow::anyhow!("HIP GPU query failed for device {ordinal}: {e}"))?;
    let base_arch = arch_name.split(':').next().unwrap_or(&arch_name);
    let gpu_arch = GpuArch::from_backend_name(&Backend::Hip, base_arch);
    eprintln!(
        "[gpu] backend=hip device={ordinal} arch={arch_name} vram={:.1}GiB",
        total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(GpuInfo { gpu_arch })
}

pub(crate) fn lookup_registry_entry(gpu_arch: &GpuArch) -> Result<&'static RegistryEntry> {
    registry::lookup(&ModelVariant::Qwen3_8_27B, &Backend::Hip, gpu_arch).ok_or_else(|| {
        let supported_archs = registry::supported_archs_for(
            &ModelVariant::Qwen3_8_27B,
            &Backend::Hip,
        );
        anyhow::anyhow!(
            "No optimized HIP kernel for model=qwen3.8-27b arch={gpu_arch}. Supported GPU architectures: [{}]",
            supported_archs.join(", ")
        )
    })
}

/// Install the per-architecture allocation policy before loading weights.
pub(crate) fn install_arch_profile(entry: &RegistryEntry) {
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

fn strategy_label(strategy: gpu_hal::AllocStrategy) -> &'static str {
    match strategy {
        gpu_hal::AllocStrategy::Default => "hipMalloc",
        gpu_hal::AllocStrategy::HostMapped => "hipHostMalloc(MAPPED) + GetDevicePointer",
    }
}
