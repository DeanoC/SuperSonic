#![recursion_limit = "512"]

mod backend_runtime;
mod bakes;
mod certified_kv;
mod cli;
mod decode_engine;
mod dflash_ddtree;
mod flm_model_source;
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
mod qwen35_alt_runtime;
mod qwen35_component_decode;
mod qwen35_decode_batch;
mod qwen35_decode_loop;
mod qwen35_decode_modes;
mod qwen35_decode_report;
mod qwen35_decode_single;
mod qwen35_decode_traces;
mod qwen35_decode_util;
mod qwen35_decode_validation;
mod qwen35_dflash_engine;
mod qwen35_engine_setup;
mod qwen35_kv_trace;
mod qwen35_oracle_prefill_trace;
mod qwen35_prefill;
mod qwen35_prefill_validation_report;
mod qwen35_runtime;
mod qwen35_startup;
mod qwen35_trace;
mod qwen35_trace_utils;
mod qwen35_validation;
mod qwen35_virtual_kv;
mod qwen35_vram;
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
mod qwen3_moe;
mod registry;
mod specprefill;
mod specprefill_engine;
mod tensor_bytes;
mod validate;

use anyhow::Result;
use clap::Parser;

pub(crate) use backend_runtime::resolve_oracle_device;
use backend_runtime::{
    install_arch_profile, lookup_registry_entry, query_gpu_info, resolve_backend,
};
pub(crate) use bakes::{should_fetch_exact_bake, try_download_bake};
pub(crate) use cli::Cli;
use gemma4_runtime::run_gemma4;
pub(crate) use model_files::{load_tokenizer, resolve_prompt_token_ids};
use policy::{
    q4km_like, validate_dflash_flags, validate_gfx942_policy, validate_global_flags,
    validate_specprefill_flags,
};
use profiling::MetalProfileScope;
use qwen35_runtime::run_qwen35;
use registry::{ModelFamily, ModelVariant};
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
    let gpu = query_gpu_info(backend, ordinal)?;

    // 3. Registry lookup
    let entry = lookup_registry_entry(
        &model_variant,
        backend,
        &gpu.gpu_arch,
        cli.allow_untested_gpu.as_deref(),
    )?;
    validate_gfx942_policy(&cli, &model_variant, backend, &gpu.gpu_arch)?;
    install_arch_profile(entry);

    // Run before family dispatch so DFlash flags are not silently ignored by
    // non-Qwen branches.
    validate_dflash_flags(&cli, &model_variant)?;
    validate_specprefill_flags(&cli, &model_variant, backend)?;

    // Teacher-forced is currently implemented for Llama 3.1, Qwen3.5, Gemma 4, and Phi-4.
    // Bail early for unsupported families so the user gets a clear message.
    if cli.teacher_forced {
        match model_variant.family() {
            ModelFamily::Llama31
            | ModelFamily::Qwen35
            | ModelFamily::Gemma4
            | ModelFamily::Phi4 => {} // supported; handled inside engine
            ModelFamily::Qwen3Moe => {
                anyhow::bail!("Qwen3-MoE teacher-forced decode is out of v1 scope")
            }
            ModelFamily::Qwen36Moe => {
                anyhow::bail!("Qwen3.6-MoE teacher-forced out of Phase 1 scope")
            }
        }
    }

    match model_variant.family() {
        ModelFamily::Gemma4 => run_gemma4(&cli, &model_variant, entry, ordinal, gpu.total_vram),
        ModelFamily::Phi4 => {
            phi4_engine::run_phi4(&cli, &model_variant, entry, ordinal, gpu.total_vram)
        }
        ModelFamily::Llama31 => {
            llama31_engine::run_llama31(&cli, &model_variant, entry, ordinal, gpu.total_vram)
        }
        ModelFamily::Qwen3Moe => qwen3_moe::run(&cli, entry, gpu.total_vram),
        ModelFamily::Qwen36Moe => {
            // Route Qwen3.6-MoE through the SpecPrefill orchestrator when the
            // drafter flag is set, so the cross-family R1 path
            // (Qwen3.5-0.8B drafter -> Qwen3.6-35B-A3B target) actually engages.
            // Without this branch `--specprefill-draft-dir` would be silently
            // ignored — the dense MoE path runs and the user gets no signal.
            if cli.specprefill_draft_dir.is_some() {
                specprefill_engine::run_specprefill(
                    &cli,
                    &model_variant,
                    entry,
                    ordinal,
                    gpu.total_vram,
                )
            } else {
                qwen36_moe_cli::run(&cli, entry, gpu.total_vram)
            }
        }
        ModelFamily::Qwen35 => run_qwen35(
            &cli,
            &model_variant,
            entry,
            backend,
            gpu.gpu_arch,
            ordinal,
            gpu.total_vram,
            q4km_like,
        ),
    }
}
