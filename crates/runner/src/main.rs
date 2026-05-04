#![recursion_limit = "512"]

mod bakes;
mod backend_runtime;
mod certified_kv;
mod cli;
mod decode_engine;
mod gemma4_engine;
mod gemma4_int4_engine;
mod gemma4_runtime;
mod llama31_engine;
mod model_files;
mod oracle;
mod phi4_engine;
mod policy;
mod prefill_engine;
mod profiling;
mod qwen35_dflash_engine;
mod qwen35_kv_trace;
mod qwen35_runtime;
mod qwen35_trace;
mod qwen35_trace_utils;
#[path = "qwen36_moe/mod.rs"]
mod qwen36_moe_cli;
#[path = "qwen36_moe/decode.rs"]
mod qwen36_moe_decode;
#[path = "qwen36_moe/logits.rs"]
mod qwen36_moe_logits;
#[path = "qwen36_moe/mtp.rs"]
mod qwen36_moe_mtp;
#[path = "qwen36_moe/persistent_decode.rs"]
mod qwen36_moe_persistent_decode;
#[path = "qwen36_moe/residency.rs"]
mod qwen36_moe_residency;
#[path = "qwen36_moe/residency_pages.rs"]
mod qwen36_moe_residency_pages;
#[path = "qwen36_moe/residency_types.rs"]
mod qwen36_moe_residency_types;
#[path = "qwen36_moe/speculative.rs"]
mod qwen36_moe_speculative;
#[path = "qwen36_moe/state.rs"]
mod qwen36_moe_state;
#[path = "qwen36_moe/telemetry.rs"]
mod qwen36_moe_telemetry;
#[path = "qwen36_moe/types.rs"]
mod qwen36_moe_types;
mod registry;
mod specprefill;
mod specprefill_engine;
mod tensor_bytes;
mod validate;

use anyhow::Result;
use clap::Parser;

pub(crate) use cli::Cli;
pub(crate) use bakes::{should_fetch_exact_bake, try_download_bake};
use backend_runtime::resolve_backend;
use gemma4_runtime::run_gemma4;
pub(crate) use model_files::{load_tokenizer, resolve_prompt_token_ids};
pub(crate) use backend_runtime::resolve_oracle_device;
use policy::{
    q4km_like, validate_dflash_flags, validate_gfx942_policy, validate_global_flags,
    validate_specprefill_flags,
};
use profiling::MetalProfileScope;
use qwen35_runtime::run_qwen35;
use registry::{Backend, GpuArch, ModelFamily, ModelVariant};
use supersonic_core::backend::{BackendChoice, BACKEND_CHOICES};
fn main() -> Result<()> {
    let cli = Cli::parse();
    let _metal_profile_scope = MetalProfileScope::new();
    let ordinal = cli.device;
    let backend_choice = BackendChoice::parse(&cli.backend).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown backend '{}'. Expected one of: {}",
            cli.backend,
            BACKEND_CHOICES
        )
    })?;
    let backend = resolve_backend(backend_choice, ordinal)?;
    gpu_hal::set_backend(backend);

    // 1. Parse model variant
    let model_variant = ModelVariant::from_cli_str(&cli.model).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model '{}'. Supported models: {}",
            cli.model,
            registry::supported_models_list().join(", ")
        )
    })?;

    validate_global_flags(&cli, &model_variant, backend)?;
    let q4km_like = q4km_like(&cli);
    // 2. Detect GPU
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
        Backend::Cuda => {
            let info = gpu_hal::query_device_info(backend, ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
        Backend::Metal => {
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

    // 3. Registry lookup
    let entry = match registry::lookup(&model_variant, &backend, &gpu_arch) {
        Some(e) => e,
        None => {
            if let Some(override_arch) = cli.allow_untested_gpu.as_deref() {
                let reuse_arch = GpuArch::from_backend_name(&backend, override_arch);
                let e =
                    registry::lookup(&model_variant, &backend, &reuse_arch).ok_or_else(|| {
                        let supported_archs =
                            registry::supported_archs_for(&model_variant, &backend);
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
                e
            } else {
                let supported_archs = registry::supported_archs_for(&model_variant, &backend);
                anyhow::bail!(
                    "No optimized kernel for model={model_variant} backend={backend} arch={gpu_arch}. \
                     Supported GPU architectures for this model: [{}]. \
                     To force-reuse another arch's kernel, pass --allow-untested-gpu=<arch>.",
                    supported_archs.join(", ")
                );
            }
        }
    };
    validate_gfx942_policy(&cli, &model_variant, backend, &gpu_arch)?;

    // Install per-arch policy so gpu_hal::alloc dispatches correctly.
    // `MemoryArchitecture` is informational (used downstream for VRAM
    // budgeting on APUs); `BufferPolicy` maps caller-side `BufferKind`
    // intent to the actual `AllocStrategy`. Persistent always uses the
    // classic device allocator (GPU-cacheable); Scratch may opt into
    // host-mapped on arches where that's a win — today only gfx1150.
    // Must be set before any GpuBuffer::alloc, which starts during weight
    // loading below.
    let arch_profile = registry::ArchProfile::for_arch(&entry.arch);
    gpu_hal::set_memory_architecture(arch_profile.memory);
    gpu_hal::set_buffer_policy(arch_profile.buffer_policy);
    fn strategy_label(s: gpu_hal::AllocStrategy) -> &'static str {
        match s {
            gpu_hal::AllocStrategy::Default => "hipMalloc / cudaMalloc / metal",
            gpu_hal::AllocStrategy::HostMapped => "hipHostMalloc(MAPPED) + GetDevicePointer",
        }
    }
    eprintln!(
        "[gpu] memory={:?}, buffer_policy: persistent={} scratch={}",
        arch_profile.memory,
        strategy_label(arch_profile.buffer_policy.persistent),
        strategy_label(arch_profile.buffer_policy.scratch),
    );

    // Run before family dispatch so DFlash flags are not silently ignored by
    // non-Qwen branches.
    validate_dflash_flags(&cli, &model_variant)?;
    validate_specprefill_flags(&cli, &model_variant, backend)?;

    match model_variant.family() {
        ModelFamily::Gemma4 => run_gemma4(&cli, &model_variant, entry, ordinal, total_vram),
        ModelFamily::Phi4 => phi4_engine::run_phi4(&cli, &model_variant, entry, ordinal, total_vram),
        ModelFamily::Llama31 => {
            llama31_engine::run_llama31(&cli, &model_variant, entry, ordinal, total_vram)
        }
        ModelFamily::Qwen36Moe => qwen36_moe_cli::run(&cli, entry, total_vram),
        ModelFamily::Qwen35 => run_qwen35(
            &cli,
            &model_variant,
            entry,
            backend,
            gpu_arch,
            ordinal,
            total_vram,
            q4km_like,
        ),
    }
}
