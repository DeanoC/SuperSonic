//! Qwen3.6-MoE runtime engine.
//!
//! Owns the CLI-facing Qwen3.6-MoE flow: bake selection, dry-run/budget
//! reporting, prompt setup, layer loading, session allocation, prefill,
//! generation, optional speculative extension, and final telemetry. The
//! GPU launch details live in the lower-level chain, persistent-decode,
//! generation, and spec-verify modules.

use std::{io::Write as _, path::Path, ptr};

use anyhow::{Context, Result};
use gpu_hal::{set_backend, Backend};
use kernel_ffi::qwen36_moe::{
    Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights,
};
use model_store::manifest::QuantProfile;
use model_store::BakedStore;
use supersonic_runtime::qwen36_moe_config::qwen36_kv_vmm_mode_from_env_value;

use crate::bakes::{
    effective_flm_source, flm_source_open_options, validate_effective_flm_source_model,
    validate_flm_weight_source_options,
};
use crate::profiling::{PrefillProfileScope, Qwen36DecodeProfileScope};
use crate::qwen36_moe_cli::bake::{ensure_qwen36_bake, select_decode_bake};
use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::dry_run::{
    print_report, run_qwen36_moe_dry_run, run_qwen36_moe_dry_run_with_config, ContextSizeSource,
    DryRunReport,
};
use crate::qwen36_moe_cli::flm_source::{open_qwen36_moe_flm_source, Qwen36MoeFlmSource};
use crate::qwen36_moe_cli::generation::{run_generation_step, Qwen36GenerationStep};
use crate::qwen36_moe_cli::geom::build_multi_layer_geom;
use crate::qwen36_moe_cli::host::{lookup_embed_row, lookup_embed_row_timed};
use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::qwen36_moe_cli::output::{
    print_decode_stream_start, print_decoded_token, print_generation_summary,
    print_last_logits_if_requested, print_runtime_engine_load_evidence, print_sampling_summary,
};
use crate::qwen36_moe_cli::policy::{
    max_speculative_tokens_for_backend, metal_mtp_experiment_enabled, resolve_context_size,
    validate_cuda_v1_flags, validate_decode_backend, validate_metal_v1_flags,
    validate_persistent_kv_fp8_flags,
};
use crate::qwen36_moe_cli::prompt::{
    prepare_prompt, prepare_prompt_with_tokenizer, print_prompt_summary,
    validate_speculative_sampling,
};
use crate::qwen36_moe_cli::session::{prepare_decode_session, Qwen36DecodeSession};
use crate::qwen36_moe_cli::spec_verify::{run_speculative_extension, Qwen36SpeculativeExtension};
use crate::qwen36_moe_cli::timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_cli::vmm::{
    load_decode_layers_with_vmm_strategy, print_virtual_kv_stats_if_active,
    virtual_kv_stats_for_layers, Qwen36LayerLoadTimings,
};
use crate::qwen36_moe_cli::vmm_config::{
    effective_moe_expert_vmm_mode_for_transfer_backend, prepare_moe_runtime_config,
    should_use_qwen36_kv_vmm,
};
use crate::qwen36_moe_cli::{Qwen36MoeEngine, Qwen36MoeLoadConfig, Qwen36MoeLoadPolicy};
use crate::qwen36_moe_logits::{f32_to_bf16_bytes, sample_bf16_logits, XorshiftRng};
use crate::qwen36_moe_speculative::SpeculativeStepResult;
use crate::qwen36_moe_telemetry::{print_and_write_moe_residency_summary, MoeRouteRuntime};
use crate::qwen36_moe_types::{LayerBuffers, PositionPair};
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};

fn prewarm_qwen36_mps_static_topn_if_requested(
    ordinal: usize,
    backend: Backend,
    geom: &crate::qwen36_moe_types::MultiLayerGeom,
    layers: &mut [LayerBuffers],
) -> Result<std::time::Duration> {
    if backend != Backend::Metal
        || std::env::var_os("SUPERSONIC_METAL_PREWARM_QWEN36_FFN_MPS_STATIC_TOPN").is_none()
    {
        return Ok(std::time::Duration::ZERO);
    }

    let started = std::time::Instant::now();
    let mut attempted_layers = 0usize;
    let mut warmed_layers = 0usize;
    let mut allocations = 0usize;
    let mut copied_bytes = 0usize;
    let mut resident_capacity = 0usize;

    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        let ffn = &mut layer.ffn;
        let Some(int4) = &ffn.int4 else {
            continue;
        };
        attempted_layers += 1;
        let params = Qwen36MoeFfnStepParams {
            stage: 5,
            layer_idx: layer_idx as i32,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weights = Qwen36MoeFfnStepWeights {
            input_hidden: ptr::null(),
            post_attn_norm_w: ffn.post_attn_norm_w.as_ptr(),
            gate_w: ffn.gate_w.as_ptr(),
            gate_up_proj_w: ffn.gate_up_proj_w.as_ptr(),
            down_proj_w: ffn.down_proj_w.as_ptr(),
            shared_gate_proj_w: ffn.shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: ffn.shared_up_proj_w.as_ptr(),
            shared_down_proj_w: ffn.shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: ffn.shared_expert_gate_w.as_ptr(),
        };
        let fp8 = int4.group_size < 0;
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size: int4.group_size,
            gate_up_proj_type: int4.gate_up_proj_type,
            gate_up_proj_scale: int4.gate_up_proj_scale.as_ptr(),
            gate_up_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.gate_up_proj_zero.as_ptr()
            },
            down_proj_type: int4.down_proj_type,
            down_proj_scale: int4.down_proj_scale.as_ptr(),
            down_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.down_proj_zero.as_ptr()
            },
            shared_gate_proj_type: int4.shared_gate_proj_type,
            shared_gate_proj_scale: int4.shared_gate_proj_scale.as_ptr(),
            shared_gate_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_gate_proj_zero.as_ptr()
            },
            shared_up_proj_type: int4.shared_up_proj_type,
            shared_up_proj_scale: int4.shared_up_proj_scale.as_ptr(),
            shared_up_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_up_proj_zero.as_ptr()
            },
            shared_down_proj_type: int4.shared_down_proj_type,
            shared_down_proj_scale: int4.shared_down_proj_scale.as_ptr(),
            shared_down_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_down_proj_zero.as_ptr()
            },
        };
        if let Some(stats) = kernel_ffi::qwen36_moe::qwen36_prewarm_mps_static_topn_rhs_for_metal(
            ordinal, params, &weights, &int4_ptrs,
        )
        .with_context(|| format!("prewarm Qwen3.6 MPS static top-N RHS layer {layer_idx}"))?
        {
            warmed_layers += 1;
            allocations += usize::from(stats.allocated);
            copied_bytes += stats.copied_bytes;
            resident_capacity = resident_capacity.max(stats.resident_capacity);
        }
    }

    let elapsed = started.elapsed();
    eprintln!(
        "[qwen36-moe ffn-prewarm] mode=mps-static-topn status=ok attempted_layers={} warmed_layers={} allocations={} resident_capacity={} copied_bytes={} elapsed_ms={:.3}",
        attempted_layers,
        warmed_layers,
        allocations,
        resident_capacity,
        copied_bytes,
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(elapsed)
}

fn validate_qwen36_decode_weight_mode(
    weight_mode: Qwen36WeightMode,
    backend: Backend,
    source_label: &str,
) -> Result<()> {
    if weight_mode.is_int4() {
        return Ok(());
    }
    match backend {
        Backend::Cuda => anyhow::bail!(
            "Qwen3.6-35B-A3B CUDA v1 requires an INT4-compatible source; \
             selected {} from {}",
            weight_mode.display_name(),
            source_label,
        ),
        Backend::Metal => anyhow::bail!(
            "Qwen3.6-35B-A3B Metal v1 requires an INT4-GPTQ-compatible source; \
             selected {} from {}",
            weight_mode.display_name(),
            source_label,
        ),
        _ => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Qwen36StartupTimings {
    flm_source_open: std::time::Duration,
    flm_store_open: std::time::Duration,
    flm_config: std::time::Duration,
    flm_tokenizer: std::time::Duration,
    flm_tokenizer_assets: std::time::Duration,
    flm_tokenizer_parse: std::time::Duration,
    flm_tokenizer_parse_vocab: std::time::Duration,
    flm_tokenizer_parse_vocab_ids: std::time::Duration,
    flm_tokenizer_parse_merges: std::time::Duration,
    flm_tokenizer_parse_added_tokens: std::time::Duration,
    flm_tokenizer_parse_regex: std::time::Duration,
    flm_tokenizer_build: std::time::Duration,
    flm_direct_plan: std::time::Duration,
    bake_prepare: std::time::Duration,
    dry_run: std::time::Duration,
}

impl Qwen36StartupTimings {
    fn pre_decode_total(self) -> std::time::Duration {
        self.flm_source_open + self.bake_prepare + self.dry_run
    }

    fn print_if_requested(self, emit_stage_timings: bool) {
        if !emit_stage_timings {
            return;
        }
        eprintln!(
            "[qwen36-moe startup-timings] {}",
            format_qwen36_startup_timings(&self)
        );
    }
}

fn qwen36_duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn format_qwen36_startup_timings(timings: &Qwen36StartupTimings) -> String {
    format!(
        "flm_source_open_ms={:.3} flm_store_open_ms={:.3} \
         flm_config_ms={:.3} flm_tokenizer_ms={:.3} \
         flm_tokenizer_assets_ms={:.3} flm_tokenizer_parse_ms={:.3} \
         flm_tokenizer_parse_vocab_ms={:.3} \
         flm_tokenizer_parse_vocab_ids_ms={:.3} \
         flm_tokenizer_parse_merges_ms={:.3} \
         flm_tokenizer_parse_added_tokens_ms={:.3} \
         flm_tokenizer_parse_regex_ms={:.3} \
         flm_tokenizer_build_ms={:.3} flm_direct_plan_ms={:.3} \
         bake_prepare_ms={:.3} \
         dry_run_ms={:.3} pre_decode_total_ms={:.3}",
        qwen36_duration_ms(timings.flm_source_open),
        qwen36_duration_ms(timings.flm_store_open),
        qwen36_duration_ms(timings.flm_config),
        qwen36_duration_ms(timings.flm_tokenizer),
        qwen36_duration_ms(timings.flm_tokenizer_assets),
        qwen36_duration_ms(timings.flm_tokenizer_parse),
        qwen36_duration_ms(timings.flm_tokenizer_parse_vocab),
        qwen36_duration_ms(timings.flm_tokenizer_parse_vocab_ids),
        qwen36_duration_ms(timings.flm_tokenizer_parse_merges),
        qwen36_duration_ms(timings.flm_tokenizer_parse_added_tokens),
        qwen36_duration_ms(timings.flm_tokenizer_parse_regex),
        qwen36_duration_ms(timings.flm_tokenizer_build),
        qwen36_duration_ms(timings.flm_direct_plan),
        qwen36_duration_ms(timings.bake_prepare),
        qwen36_duration_ms(timings.dry_run),
        qwen36_duration_ms(timings.pre_decode_total()),
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct Qwen36LifecycleTimings {
    prompt_setup: std::time::Duration,
    flm_tokenizer: std::time::Duration,
    flm_tokenizer_assets: std::time::Duration,
    flm_tokenizer_parse: std::time::Duration,
    flm_tokenizer_parse_vocab: std::time::Duration,
    flm_tokenizer_parse_vocab_ids: std::time::Duration,
    flm_tokenizer_parse_merges: std::time::Duration,
    flm_tokenizer_parse_added_tokens: std::time::Duration,
    flm_tokenizer_parse_regex: std::time::Duration,
    flm_tokenizer_build: std::time::Duration,
    model_source: std::time::Duration,
    layer_load: std::time::Duration,
    layer_load_profile: Qwen36LayerLoadTimings,
    session: std::time::Duration,
    prefill_steps: usize,
    prefill_embed: std::time::Duration,
    prefill_chain: std::time::Duration,
    generation_wall: Option<f64>,
    total_wall: std::time::Duration,
}

impl Qwen36LifecycleTimings {
    fn prefill_total(self) -> std::time::Duration {
        self.prefill_embed + self.prefill_chain
    }
}

fn format_qwen36_lifecycle_timings(timings: &Qwen36LifecycleTimings) -> String {
    let duration = |available: bool, value: std::time::Duration| {
        available
            .then(|| format!("{:.3}", qwen36_duration_ms(value)))
            .unwrap_or_else(|| "unavailable".to_string())
    };
    let bytes = |available: bool, value: u64| {
        available
            .then(|| value.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    };
    format!(
        "prompt_setup_ms={:.3} flm_tokenizer_ms={:.3} \
         flm_tokenizer_assets_ms={:.3} flm_tokenizer_parse_ms={:.3} \
         flm_tokenizer_parse_vocab_ms={:.3} \
         flm_tokenizer_parse_vocab_ids_ms={:.3} \
         flm_tokenizer_parse_merges_ms={:.3} \
         flm_tokenizer_parse_added_tokens_ms={:.3} \
         flm_tokenizer_parse_regex_ms={:.3} \
         flm_tokenizer_build_ms={:.3} model_source_ms={:.3} \
         layer_load_ms={:.3} layer_load_buffers_ms={} \
         layer_load_vmm_setup_ms={} layer_load_prewarm_ms={} \
         layer_load_hal_ms={} layer_load_alloc_ms={} \
         layer_load_copy_h_to_d_ms={} layer_load_memset_ms={} \
         layer_load_vmm_ms={} layer_load_alloc_bytes={} \
         layer_load_copy_h_to_d_bytes={} layer_load_memset_bytes={} \
         layer_load_vmm_bytes={} session_ms={:.3} prefill_steps={} \
         prefill_embed_ms={:.3} prefill_chain_ms={:.3} \
         prefill_total_ms={:.3} generation_wall_ms={:.3} total_wall_ms={:.3}",
        qwen36_duration_ms(timings.prompt_setup),
        qwen36_duration_ms(timings.flm_tokenizer),
        qwen36_duration_ms(timings.flm_tokenizer_assets),
        qwen36_duration_ms(timings.flm_tokenizer_parse),
        qwen36_duration_ms(timings.flm_tokenizer_parse_vocab),
        qwen36_duration_ms(timings.flm_tokenizer_parse_vocab_ids),
        qwen36_duration_ms(timings.flm_tokenizer_parse_merges),
        qwen36_duration_ms(timings.flm_tokenizer_parse_added_tokens),
        qwen36_duration_ms(timings.flm_tokenizer_parse_regex),
        qwen36_duration_ms(timings.flm_tokenizer_build),
        qwen36_duration_ms(timings.model_source),
        qwen36_duration_ms(timings.layer_load),
        duration(
            timings.layer_load_profile.detail_available,
            timings.layer_load_profile.buffers
        ),
        duration(
            timings.layer_load_profile.detail_available,
            timings.layer_load_profile.vmm_setup
        ),
        duration(
            timings.layer_load_profile.detail_available,
            timings.layer_load_profile.prewarm
        ),
        duration(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_total
        ),
        duration(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_alloc
        ),
        duration(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_copy_h_to_d
        ),
        duration(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_memset
        ),
        duration(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_vmm
        ),
        bytes(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_alloc_bytes
        ),
        bytes(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_copy_h_to_d_bytes
        ),
        bytes(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_memset_bytes
        ),
        bytes(
            timings.layer_load_profile.hal_available,
            timings.layer_load_profile.hal_vmm_bytes
        ),
        qwen36_duration_ms(timings.session),
        timings.prefill_steps,
        qwen36_duration_ms(timings.prefill_embed),
        qwen36_duration_ms(timings.prefill_chain),
        qwen36_duration_ms(timings.prefill_total()),
        timings.generation_wall.unwrap_or(0.0),
        qwen36_duration_ms(timings.total_wall),
    )
}

fn qwen36_layer_load_hal_timings(snapshot: &gpu_hal::HalProfileSnapshot) -> Qwen36LayerLoadTimings {
    let mut timings = Qwen36LayerLoadTimings {
        hal_available: true,
        hal_total: std::time::Duration::from_secs_f64(snapshot.total_ms / 1000.0),
        hal_alloc_bytes: snapshot.alloc_bytes,
        hal_copy_h_to_d_bytes: snapshot.h2d_bytes,
        hal_memset_bytes: snapshot.memset_bytes,
        ..Default::default()
    };

    for entry in &snapshot.entries {
        let duration = std::time::Duration::from_secs_f64(entry.total_ms / 1000.0);
        match entry.op.as_str() {
            "alloc" => timings.hal_alloc += duration,
            "copy_h2d" => timings.hal_copy_h_to_d += duration,
            "memset_zeros" => timings.hal_memset += duration,
            op if op.starts_with("vmm_") => {
                timings.hal_vmm += duration;
                timings.hal_vmm_bytes += entry.total_bytes;
            }
            _ => {}
        }
    }

    timings
}

fn qwen36_should_profile_layer_load_hal(
    emit_stage_timings: bool,
    external_hal_profile_active: bool,
) -> bool {
    emit_stage_timings && !external_hal_profile_active
}

fn qwen36_external_hal_profile_env_active() -> bool {
    std::env::var_os("SUPERSONIC_HAL_PROFILE").is_some()
        || std::env::var_os("SUPERSONIC_METAL_PROFILE").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
    use clap::Parser;

    #[test]
    fn cuda_decode_rejects_non_int4_flm_source_weight_mode_with_source_label() {
        let err =
            validate_qwen36_decode_weight_mode(Qwen36WeightMode::Bf16, Backend::Cuda, "model.flm")
                .unwrap_err()
                .to_string();

        assert!(err.contains("CUDA"), "{err}");
        assert!(err.contains("INT4"), "{err}");
        assert!(err.contains("model.flm"), "{err}");
    }

    #[test]
    fn hip_decode_allows_non_int4_weight_mode_for_development_path() {
        validate_qwen36_decode_weight_mode(Qwen36WeightMode::Bf16, Backend::Hip, "model.flm")
            .expect("HIP development path can still choose BF16/FP8");
    }

    #[test]
    fn formats_startup_timings_for_machine_parsing() {
        let timings = Qwen36StartupTimings {
            flm_source_open: std::time::Duration::from_micros(1_500),
            flm_store_open: std::time::Duration::from_micros(500),
            flm_config: std::time::Duration::from_micros(250),
            flm_tokenizer: std::time::Duration::from_micros(625),
            flm_tokenizer_assets: std::time::Duration::from_micros(25),
            flm_tokenizer_parse: std::time::Duration::from_micros(275),
            flm_tokenizer_parse_vocab: std::time::Duration::from_micros(100),
            flm_tokenizer_parse_vocab_ids: std::time::Duration::from_micros(25),
            flm_tokenizer_parse_merges: std::time::Duration::from_micros(75),
            flm_tokenizer_parse_added_tokens: std::time::Duration::from_micros(50),
            flm_tokenizer_parse_regex: std::time::Duration::from_micros(25),
            flm_tokenizer_build: std::time::Duration::from_micros(325),
            flm_direct_plan: std::time::Duration::from_micros(125),
            bake_prepare: std::time::Duration::ZERO,
            dry_run: std::time::Duration::from_micros(2_250),
        };

        assert_eq!(
            format_qwen36_startup_timings(&timings),
            "flm_source_open_ms=1.500 flm_store_open_ms=0.500 \
             flm_config_ms=0.250 flm_tokenizer_ms=0.625 \
             flm_tokenizer_assets_ms=0.025 flm_tokenizer_parse_ms=0.275 \
             flm_tokenizer_parse_vocab_ms=0.100 \
             flm_tokenizer_parse_vocab_ids_ms=0.025 \
             flm_tokenizer_parse_merges_ms=0.075 \
             flm_tokenizer_parse_added_tokens_ms=0.050 \
             flm_tokenizer_parse_regex_ms=0.025 \
             flm_tokenizer_build_ms=0.325 flm_direct_plan_ms=0.125 \
             bake_prepare_ms=0.000 \
             dry_run_ms=2.250 pre_decode_total_ms=3.750"
        );
    }

    #[test]
    fn formats_lifecycle_timings_with_lazy_flm_tokenizer_breakdown() {
        let timings = Qwen36LifecycleTimings {
            prompt_setup: std::time::Duration::from_micros(5_000),
            flm_tokenizer: std::time::Duration::from_micros(625),
            flm_tokenizer_assets: std::time::Duration::from_micros(25),
            flm_tokenizer_parse: std::time::Duration::from_micros(275),
            flm_tokenizer_parse_vocab: std::time::Duration::from_micros(100),
            flm_tokenizer_parse_vocab_ids: std::time::Duration::from_micros(25),
            flm_tokenizer_parse_merges: std::time::Duration::from_micros(75),
            flm_tokenizer_parse_added_tokens: std::time::Duration::from_micros(50),
            flm_tokenizer_parse_regex: std::time::Duration::from_micros(25),
            flm_tokenizer_build: std::time::Duration::from_micros(325),
            model_source: std::time::Duration::from_micros(1_500),
            layer_load: std::time::Duration::from_micros(2_500),
            layer_load_profile: Qwen36LayerLoadTimings {
                detail_available: true,
                hal_available: true,
                buffers: std::time::Duration::from_micros(1_100),
                vmm_setup: std::time::Duration::from_micros(200),
                prewarm: std::time::Duration::from_micros(300),
                hal_total: std::time::Duration::from_micros(1_000),
                hal_alloc: std::time::Duration::from_micros(400),
                hal_copy_h_to_d: std::time::Duration::from_micros(500),
                hal_memset: std::time::Duration::from_micros(50),
                hal_vmm: std::time::Duration::from_micros(50),
                hal_alloc_bytes: 1_024,
                hal_copy_h_to_d_bytes: 2_048,
                hal_memset_bytes: 512,
                hal_vmm_bytes: 4_096,
            },
            session: std::time::Duration::from_micros(3_500),
            prefill_steps: 2,
            prefill_embed: std::time::Duration::from_micros(100),
            prefill_chain: std::time::Duration::from_micros(200),
            generation_wall: Some(4.5),
            total_wall: std::time::Duration::from_micros(9_000),
        };

        assert_eq!(
            format_qwen36_lifecycle_timings(&timings),
            "prompt_setup_ms=5.000 flm_tokenizer_ms=0.625 \
             flm_tokenizer_assets_ms=0.025 flm_tokenizer_parse_ms=0.275 \
             flm_tokenizer_parse_vocab_ms=0.100 \
             flm_tokenizer_parse_vocab_ids_ms=0.025 \
             flm_tokenizer_parse_merges_ms=0.075 \
             flm_tokenizer_parse_added_tokens_ms=0.050 \
             flm_tokenizer_parse_regex_ms=0.025 \
             flm_tokenizer_build_ms=0.325 model_source_ms=1.500 \
             layer_load_ms=2.500 layer_load_buffers_ms=1.100 \
             layer_load_vmm_setup_ms=0.200 layer_load_prewarm_ms=0.300 \
             layer_load_hal_ms=1.000 layer_load_alloc_ms=0.400 \
             layer_load_copy_h_to_d_ms=0.500 layer_load_memset_ms=0.050 \
             layer_load_vmm_ms=0.050 layer_load_alloc_bytes=1024 \
             layer_load_copy_h_to_d_bytes=2048 layer_load_memset_bytes=512 \
             layer_load_vmm_bytes=4096 session_ms=3.500 prefill_steps=2 \
             prefill_embed_ms=0.100 prefill_chain_ms=0.200 \
             prefill_total_ms=0.300 generation_wall_ms=4.500 \
             total_wall_ms=9.000"
        );
    }

    #[test]
    fn lifecycle_timings_mark_unmeasured_layer_load_subphases_unavailable() {
        let formatted = format_qwen36_lifecycle_timings(&Qwen36LifecycleTimings {
            layer_load_profile: Qwen36LayerLoadTimings {
                hal_available: true,
                hal_total: std::time::Duration::from_micros(1_000),
                hal_alloc: std::time::Duration::from_micros(400),
                hal_alloc_bytes: 1_024,
                ..Default::default()
            },
            ..Default::default()
        });

        assert!(formatted.contains("layer_load_buffers_ms=unavailable"));
        assert!(formatted.contains("layer_load_vmm_setup_ms=unavailable"));
        assert!(formatted.contains("layer_load_prewarm_ms=unavailable"));
        assert!(formatted.contains("layer_load_hal_ms=1.000"));
        assert!(formatted.contains("layer_load_alloc_bytes=1024"));
    }

    #[test]
    fn layer_load_hal_timings_group_transfer_and_vmm_ops() {
        let snapshot = gpu_hal::HalProfileSnapshot {
            total_calls: 5,
            total_ms: 7.0,
            alloc_calls: 1,
            alloc_bytes: 1_024,
            free_calls: 0,
            h2d_bytes: 2_048,
            d2h_bytes: 0,
            d2d_bytes: 0,
            memset_bytes: 512,
            sync_calls: 0,
            entries: vec![
                gpu_hal::HalProfileEntry {
                    op: "alloc".to_string(),
                    calls: 1,
                    total_ms: 1.5,
                    max_ms: 1.5,
                    total_bytes: 1_024,
                },
                gpu_hal::HalProfileEntry {
                    op: "copy_h2d".to_string(),
                    calls: 1,
                    total_ms: 2.5,
                    max_ms: 2.5,
                    total_bytes: 2_048,
                },
                gpu_hal::HalProfileEntry {
                    op: "memset_zeros".to_string(),
                    calls: 1,
                    total_ms: 0.75,
                    max_ms: 0.75,
                    total_bytes: 512,
                },
                gpu_hal::HalProfileEntry {
                    op: "vmm_reserve".to_string(),
                    calls: 1,
                    total_ms: 1.0,
                    max_ms: 1.0,
                    total_bytes: 3_000,
                },
                gpu_hal::HalProfileEntry {
                    op: "vmm_map".to_string(),
                    calls: 1,
                    total_ms: 1.25,
                    max_ms: 1.25,
                    total_bytes: 1_096,
                },
            ],
        };

        let timings = qwen36_layer_load_hal_timings(&snapshot);

        assert_eq!(timings.hal_total, std::time::Duration::from_micros(7_000));
        assert_eq!(timings.hal_alloc, std::time::Duration::from_micros(1_500));
        assert_eq!(
            timings.hal_copy_h_to_d,
            std::time::Duration::from_micros(2_500)
        );
        assert_eq!(timings.hal_memset, std::time::Duration::from_micros(750));
        assert_eq!(timings.hal_vmm, std::time::Duration::from_micros(2_250));
        assert_eq!(timings.hal_alloc_bytes, 1_024);
        assert_eq!(timings.hal_copy_h_to_d_bytes, 2_048);
        assert_eq!(timings.hal_memset_bytes, 512);
        assert_eq!(timings.hal_vmm_bytes, 4_096);
    }

    #[test]
    fn layer_load_hal_profile_only_enables_for_isolated_stage_timings() {
        assert!(qwen36_should_profile_layer_load_hal(true, false));
        assert!(!qwen36_should_profile_layer_load_hal(false, false));
        assert!(!qwen36_should_profile_layer_load_hal(true, true));
    }

    #[test]
    fn multi_token_runtime_prefill_accounting_does_not_double_count_final_production() {
        let accounting = runtime_prefill_accounting(
            3,
            std::time::Duration::from_millis(12),
            std::time::Duration::from_millis(5),
        );

        assert_eq!(accounting.prefill_steps, 2);
        assert_eq!(
            accounting.prefill_chain,
            std::time::Duration::from_millis(12)
        );
        assert_eq!(
            accounting.first_generation_inference,
            std::time::Duration::from_millis(5)
        );
        assert_eq!(
            accounting.prefill_chain + accounting.first_generation_inference,
            std::time::Duration::from_millis(17)
        );
    }

    #[test]
    fn runtime_request_adapter_delegates_one_load_reset_and_request_sequence() {
        #[derive(Default)]
        struct FakeEngine {
            reset_calls: usize,
            prefill_calls: usize,
            decode_calls: usize,
        }

        let mut load_calls = 0;
        let mut request = Qwen36RuntimeRequestAdapter::load(
            || {
                load_calls += 1;
                Ok(FakeEngine::default())
            },
            |engine| {
                engine.reset_calls += 1;
                Ok(())
            },
        )
        .expect("load fake runtime request");

        request
            .prefill(|engine| {
                engine.prefill_calls += 1;
                Ok(())
            })
            .expect("prefill fake runtime request");
        request
            .decode_step(|engine| {
                engine.decode_calls += 1;
                Ok(())
            })
            .expect("first fake decode");
        request
            .decode_step(|engine| {
                engine.decode_calls += 1;
                Ok(())
            })
            .expect("second fake decode");

        assert_eq!(load_calls, 1);
        let engine = request.into_inner();
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(engine.prefill_calls, 1);
        assert_eq!(engine.decode_calls, 2);
    }

    #[test]
    fn plain_flm_selector_deterministically_routes_only_compatible_cli_requests() {
        let cli = crate::Cli::parse_from([
            "supersonic",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            "/tmp/prevalidated.flm",
            "--backend",
            "hip",
            "--prompt",
            "Hello",
        ]);
        let entry = crate::registry::lookup(
            &ModelVariant::Qwen3_6_35B_A3B,
            &Backend::Hip,
            &crate::registry::GpuArch::Gfx1100,
        )
        .expect("Qwen3.6 HIP gfx1100 registry entry");
        let compatible = crate::qwen36_moe_cli::options::Qwen36RunnerModeOptions::default();

        assert!(qwen36_plain_flm_runtime_path(
            &cli,
            entry,
            None,
            &compatible
        ));

        let segmented = crate::qwen36_moe_cli::options::Qwen36RunnerModeOptions {
            segmented_profile: true,
            ..Default::default()
        };
        assert!(!qwen36_plain_flm_runtime_path(
            &cli, entry, None, &segmented
        ));
    }
}

enum DecodeStore<'a> {
    Borrowed(&'a BakedStore),
    Owned(BakedStore),
}

impl<'a> DecodeStore<'a> {
    fn as_store(&self) -> &BakedStore {
        match self {
            DecodeStore::Borrowed(store) => store,
            DecodeStore::Owned(store) => store,
        }
    }
}

/// Compute the `(rope, cache)` PositionPair for one step of the
/// decode loop. In dense mode the rope and cache agree; in
/// SpecPrefill mode the rope tracks the absolute prompt-token
/// position (during prefill of kept tokens) or the absolute
/// generation position (after prefill ends) while the cache slot
/// is the compact `loop_state_position`.
pub(crate) fn current_position(
    step: usize,
    loop_state_position: i32,
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    full_prompt_len: usize,
) -> PositionPair {
    match keep_mask {
        None => PositionPair::dense(loop_state_position),
        Some(_) => {
            let rope = if step < effective_prompt_len {
                kept_positions[step] as i32
            } else {
                let gen_off = step - effective_prompt_len;
                (full_prompt_len + gen_off) as i32
            };
            PositionPair::split(rope, loop_state_position)
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen36MtpAcceptanceStats {
    mode: &'static str,
    steps: usize,
    drafted_tokens: usize,
    accepted_tokens: usize,
    emitted_tokens: usize,
    base_steps: usize,
    replay_steps: usize,
    full_accept_steps: usize,
    zero_accept_steps: usize,
    max_accept: usize,
}

impl Qwen36MtpAcceptanceStats {
    fn new(batched_spec_verify: bool) -> Self {
        Self {
            mode: if batched_spec_verify {
                "batched"
            } else {
                "sequential"
            },
            steps: 0,
            drafted_tokens: 0,
            accepted_tokens: 0,
            emitted_tokens: 0,
            base_steps: 0,
            replay_steps: 0,
            full_accept_steps: 0,
            zero_accept_steps: 0,
            max_accept: 0,
        }
    }

    fn record(&mut self, result: &SpeculativeStepResult) {
        self.steps += 1;
        self.drafted_tokens += result.n_drafted;
        self.accepted_tokens += result.n_accepted;
        self.emitted_tokens += result.emitted_tokens.len();
        self.base_steps += result.base_steps;
        self.replay_steps += result.replay_steps;
        self.max_accept = self.max_accept.max(result.n_accepted);
        if result.n_drafted > 0 && result.n_accepted == result.n_drafted {
            self.full_accept_steps += 1;
        }
        if result.n_accepted == 0 {
            self.zero_accept_steps += 1;
        }
    }

    fn print_if_requested(&self, enabled: bool) {
        if !enabled || self.steps == 0 {
            return;
        }
        let acceptance_rate = if self.drafted_tokens > 0 {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        } else {
            0.0
        };
        let emitted_per_step = self.emitted_tokens as f64 / self.steps as f64;
        let target_steps = self.base_steps + self.replay_steps;
        let target_steps_per_emitted = if self.emitted_tokens > 0 {
            target_steps as f64 / self.emitted_tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "[qwen36-mtp-acceptance] mode={} steps={} drafted_tokens={} \
             accepted_tokens={} acceptance_rate={:.6} emitted_tokens={} \
             emitted_per_step={:.6} base_steps={} replay_steps={} \
             target_steps_per_emitted={:.6} full_accept_steps={} \
             zero_accept_steps={} max_accept={}",
            self.mode,
            self.steps,
            self.drafted_tokens,
            self.accepted_tokens,
            acceptance_rate,
            self.emitted_tokens,
            emitted_per_step,
            self.base_steps,
            self.replay_steps,
            target_steps_per_emitted,
            self.full_accept_steps,
            self.zero_accept_steps,
            self.max_accept,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimePrefillAccounting {
    prefill_steps: usize,
    prefill_chain: std::time::Duration,
    first_generation_inference: std::time::Duration,
}

struct Qwen36RuntimeRequestAdapter<E> {
    engine: E,
    prefilled: bool,
}

impl<E> Qwen36RuntimeRequestAdapter<E> {
    fn load(
        load: impl FnOnce() -> Result<E>,
        reset: impl FnOnce(&mut E) -> Result<()>,
    ) -> Result<Self> {
        let mut engine = load()?;
        reset(&mut engine)?;
        Ok(Self {
            engine,
            prefilled: false,
        })
    }

    fn engine(&self) -> &E {
        &self.engine
    }

    fn prefill<T>(&mut self, prefill: impl FnOnce(&mut E) -> Result<T>) -> Result<T> {
        if self.prefilled {
            anyhow::bail!("Qwen3.6 runtime CLI request prefill may only run once");
        }
        let output = prefill(&mut self.engine)?;
        self.prefilled = true;
        Ok(output)
    }

    fn decode_step<T>(&mut self, decode: impl FnOnce(&mut E) -> Result<T>) -> Result<T> {
        if !self.prefilled {
            anyhow::bail!("Qwen3.6 runtime CLI request decode requires prefill");
        }
        decode(&mut self.engine)
    }

    #[cfg(test)]
    fn into_inner(self) -> E {
        self.engine
    }
}

fn runtime_prefill_accounting(
    prompt_len: usize,
    prefix_duration: std::time::Duration,
    final_production_duration: std::time::Duration,
) -> RuntimePrefillAccounting {
    RuntimePrefillAccounting {
        prefill_steps: prompt_len.saturating_sub(1),
        prefill_chain: prefix_duration,
        first_generation_inference: final_production_duration,
    }
}

fn qwen36_plain_flm_runtime_path(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    keep_mask: Option<&Vec<bool>>,
    modes: &crate::qwen36_moe_cli::options::Qwen36RunnerModeOptions,
) -> bool {
    !cli.dry_run
        && entry.backend == Backend::Hip
        && effective_flm_source(cli).is_some()
        && !cli.speculative_decode
        && keep_mask.is_none()
        && !cli.no_persistent_decode
        && cli.batch_size.max(1) == 1
        && modes.runtime_engine_compatible()
}

#[allow(clippy::too_many_arguments)]
fn run_with_runtime_engine(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    flm_path: &Path,
    context_size: usize,
    context_size_source: ContextSizeSource,
    total_vram: u64,
    sampling: SamplingParams,
    execution_options: supersonic_runtime::qwen36_moe::decode::Qwen36ExecutionOptions,
) -> Result<()> {
    validate_decode_backend(entry)?;
    validate_effective_flm_source_model(cli, &ModelVariant::Qwen3_6_35B_A3B)?;
    validate_flm_weight_source_options(cli, crate::policy::q4km_like(cli))?;
    let source_options = flm_source_open_options(cli)?;
    let kernel_params = match entry.params {
        FamilyParams::Qwen36Moe(params) => params,
        _ => anyhow::bail!("registry entry is not Qwen36Moe family"),
    };

    let mut moe_runtime = prepare_moe_runtime_config(
        false,
        true,
        entry.backend,
        kernel_params.top_k as usize,
        cli.flm_virtual_transfer_backend.as_deref(),
    )?;
    moe_runtime.vmm_mode = effective_moe_expert_vmm_mode_for_transfer_backend(
        moe_runtime.vmm_mode,
        moe_runtime.island_cap_experts,
        moe_runtime.virtual_transfer_backend,
    )?;
    let virtual_transfer_backend = moe_runtime.virtual_transfer_backend;
    let moe_policy = (*moe_runtime).clone();
    let _sparse_telemetry = moe_runtime.sparse_telemetry.take();
    let kv_vmm_env = std::env::var("SUPERSONIC_VMM_KV").ok();
    let kv_vmm = qwen36_kv_vmm_mode_from_env_value(kv_vmm_env.as_deref(), entry.backend)?;
    let load_config = Qwen36MoeLoadConfig {
        flm_path: flm_path.to_owned(),
        backend: entry.backend,
        device_ordinal: cli.device,
        max_context_len: context_size,
        policy: Qwen36MoeLoadPolicy {
            persistent_decode: true,
            kv_fp8: cli.kv_fp8,
            kv_vmm,
            moe: moe_policy,
            virtual_transfer_backend,
        },
        verify_block_hashes: source_options.verify_block_hashes,
        execution_options,
        accurate_stage_timings: cli.emit_stage_timings,
    };

    let runtime_wall_start = std::time::Instant::now();
    let progress_interval = (cli.progress_heartbeat_seconds > 0.0)
        .then(|| std::time::Duration::from_secs_f64(cli.progress_heartbeat_seconds));
    let mut last_progress = runtime_wall_start
        .checked_sub(progress_interval.unwrap_or(std::time::Duration::ZERO))
        .unwrap_or(runtime_wall_start);
    let mut progress = |phase: &str, detail: String, force: bool| {
        let Some(interval) = progress_interval else {
            return;
        };
        let now = std::time::Instant::now();
        if force || now.duration_since(last_progress) >= interval {
            eprintln!(
                "[qwen36-moe progress] phase={phase} elapsed_ms={} {detail}",
                now.duration_since(runtime_wall_start).as_millis()
            );
            last_progress = now;
        }
    };

    eprintln!(
        "[flm] opening model source at {}{}{}",
        flm_path.display(),
        if source_options.int4_runtime {
            " (FLM logical INT4 aliases enabled)"
        } else {
            ""
        },
        if source_options.verify_block_hashes {
            " (BLAKE3 hash verification enabled)"
        } else {
            ""
        }
    );
    eprintln!("[qwen36-moe] loading config from FLM runtime descriptor");
    eprintln!("[qwen36-moe] loading tokenizer from FLM assets");
    progress("runtime_engine_load", "start".to_string(), true);
    let mut reset_elapsed = std::time::Duration::ZERO;
    let mut request = Qwen36RuntimeRequestAdapter::load(
        || Qwen36MoeEngine::load(load_config).context("load Qwen3.6 runtime engine"),
        |engine| {
            let reset_start = std::time::Instant::now();
            engine
                .reset()
                .context("reset Qwen3.6 runtime engine for CLI request")?;
            reset_elapsed = reset_start.elapsed();
            Ok(())
        },
    )?;
    let evidence = request.engine().load_evidence().clone();
    if evidence.source_open_count != 1 {
        anyhow::bail!(
            "Qwen3.6 CLI runtime engine must own exactly one FLM source open, observed {}",
            evidence.source_open_count
        );
    }
    progress(
        "runtime_engine_load",
        format!(
            "done load_sequence={} source_open_count={} elapsed_ms={}",
            evidence.load_sequence,
            evidence.source_open_count,
            evidence.total_duration.as_millis()
        ),
        true,
    );

    let dry_run_start = std::time::Instant::now();
    let dry_run_report = run_qwen36_moe_dry_run_with_config(
        &cli.model_dir,
        Some(flm_path),
        Some(evidence.direct_profile),
        evidence
            .config
            .clone()
            .context("Qwen3.6 runtime load evidence did not retain the FLM config")?,
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
        cli.device,
    )?;
    print_report(&dry_run_report);
    let dry_run_elapsed = dry_run_start.elapsed();
    print_runtime_engine_load_evidence(&evidence);
    let tokenizer_timings = evidence.tokenizer_timings;
    Qwen36StartupTimings {
        flm_source_open: evidence.source_open_duration,
        flm_tokenizer: evidence.tokenizer_duration,
        flm_tokenizer_assets: tokenizer_timings.asset_lookup,
        flm_tokenizer_parse: tokenizer_timings.parse,
        flm_tokenizer_parse_vocab: tokenizer_timings.parse_vocab,
        flm_tokenizer_parse_vocab_ids: tokenizer_timings.parse_vocab_ids,
        flm_tokenizer_parse_merges: tokenizer_timings.parse_merges,
        flm_tokenizer_parse_added_tokens: tokenizer_timings.parse_added_tokens,
        flm_tokenizer_parse_regex: tokenizer_timings.parse_regex,
        flm_tokenizer_build: tokenizer_timings.build,
        flm_direct_plan: evidence.plan_duration,
        dry_run: dry_run_elapsed,
        ..Default::default()
    }
    .print_if_requested(cli.emit_stage_timings);

    println!();
    println!("=== Decode (Qwen3.6-MoE) ===");
    let prompt_setup_start = std::time::Instant::now();
    let tokenizer = request.engine().tokenizer().clone();
    let prompt_ids = if cli.prompt.is_empty() {
        vec![0]
    } else {
        let encoding = tokenizer
            .encode(cli.prompt.as_str(), true)
            .map_err(|err| anyhow::anyhow!("tokenize prompt: {err}"))?;
        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            vec![0]
        } else {
            ids
        }
    };
    let eos_id = if cli.ignore_eos {
        None
    } else {
        request.engine().eos_ids().first().copied()
    };
    let prompt_setup_elapsed = prompt_setup_start.elapsed();
    print_prompt_summary(&cli.prompt, &prompt_ids);
    print_decode_stream_start(Some(&tokenizer), &prompt_ids, cli.max_new_tokens.max(1));
    print_sampling_summary(sampling);

    let backend_label = format!("{:?}", entry.backend);
    let mut prefill_profile = Some(PrefillProfileScope::new(
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        "qwen3.6-moe",
        &cli.model,
        &backend_label,
        prompt_ids.len(),
    ));
    let mut generation_wall_start = None;
    let mut decode_profile = None;
    progress(
        "prefill",
        format!("start prompt_tokens={}", prompt_ids.len()),
        true,
    );
    let prefill_start = std::time::Instant::now();
    let prefill_output = request
        .prefill(|engine| engine.prefill_with_boundaries(&prompt_ids, |boundary| {
            if boundary
                == supersonic_runtime::qwen36_moe::engine::Qwen36MoePrefillBoundary::FinalProductionStarted
            {
                if let Some(profile) = prefill_profile.take() {
                    profile.finish()?;
                }
                generation_wall_start = Some(std::time::Instant::now());
                decode_profile = Some(Qwen36DecodeProfileScope::new_from_env());
            }
            Ok(())
        }))
        .context("prefill Qwen3.6 runtime engine")?;
    let prefill_elapsed = prefill_start.elapsed();
    if let Some(profile) = prefill_profile.take() {
        profile.finish()?;
    }
    let accounting = runtime_prefill_accounting(
        prompt_ids.len(),
        prefill_output.prefix_duration,
        prefill_output.final_production_duration,
    );
    let mut logits = prefill_output.logits;
    progress(
        "prefill",
        format!(
            "done prompt_tokens={} elapsed_ms={}",
            prompt_ids.len(),
            prefill_elapsed.as_millis()
        ),
        true,
    );

    let mut rng = XorshiftRng::new(sampling.seed);
    let mut generated_ids = Vec::with_capacity(cli.max_new_tokens.max(1));
    let mut last_logits_bytes = Vec::new();
    let mut pending_inference = accounting.first_generation_inference;
    let mut stage_timings = Qwen36StageTimingTotals::default();
    let max_new = cli.max_new_tokens.max(1);

    for gen_index in 0..max_new {
        progress(
            "generate",
            format!(
                "step={} generated={} absolute_position={}",
                gen_index,
                generated_ids.len(),
                prompt_ids.len() + gen_index
            ),
            false,
        );
        let sample_start = std::time::Instant::now();
        let logits_bytes = f32_to_bf16_bytes(&logits);
        let next_token = sample_bf16_logits(
            &logits_bytes,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            &mut rng,
        );
        let sample_elapsed = sample_start.elapsed();
        if cli.dump_last_logits {
            last_logits_bytes = logits_bytes;
        }
        generated_ids.push(next_token);

        let detok_start = std::time::Instant::now();
        print_decoded_token(Some(&tokenizer), next_token);
        std::io::stdout().flush().ok();
        let detok_elapsed = detok_start.elapsed();
        stage_timings.record_runtime_generation_step(
            pending_inference,
            sample_elapsed,
            detok_elapsed,
        );

        if Some(next_token) == eos_id || gen_index + 1 == max_new {
            break;
        }

        let decode_start = std::time::Instant::now();
        logits = request
            .decode_step(|engine| engine.decode_step(next_token, prompt_ids.len() + gen_index))
            .with_context(|| {
                format!(
                    "decode Qwen3.6 runtime engine at absolute position {}",
                    prompt_ids.len() + gen_index
                )
            })?;
        pending_inference = decode_start.elapsed();
    }

    if let Some(profile) = decode_profile.take() {
        profile.finish();
    }
    print_last_logits_if_requested(cli.dump_last_logits, &last_logits_bytes);
    let generation_wall_ms = generation_wall_start
        .as_ref()
        .map(|start| start.elapsed().as_secs_f64() * 1000.0);
    print_generation_summary(&generated_ids, prompt_ids.len(), eos_id, generation_wall_ms);
    stage_timings.print_if_requested(cli.emit_stage_timings);
    if cli.emit_stage_timings {
        let layer_load_elapsed = evidence
            .total_duration
            .saturating_sub(evidence.source_open_duration)
            .saturating_sub(evidence.tokenizer_duration);
        let lifecycle_timings = Qwen36LifecycleTimings {
            prompt_setup: prompt_setup_elapsed,
            flm_tokenizer: evidence.tokenizer_duration,
            flm_tokenizer_assets: tokenizer_timings.asset_lookup,
            flm_tokenizer_parse: tokenizer_timings.parse,
            flm_tokenizer_parse_vocab: tokenizer_timings.parse_vocab,
            flm_tokenizer_parse_vocab_ids: tokenizer_timings.parse_vocab_ids,
            flm_tokenizer_parse_merges: tokenizer_timings.parse_merges,
            flm_tokenizer_parse_added_tokens: tokenizer_timings.parse_added_tokens,
            flm_tokenizer_parse_regex: tokenizer_timings.parse_regex,
            flm_tokenizer_build: tokenizer_timings.build,
            model_source: evidence.source_open_duration,
            layer_load: layer_load_elapsed,
            layer_load_profile: qwen36_layer_load_hal_timings(&evidence.hal_profile),
            session: reset_elapsed,
            prefill_steps: accounting.prefill_steps,
            prefill_chain: accounting.prefill_chain,
            generation_wall: generation_wall_ms,
            total_wall: runtime_wall_start.elapsed(),
            ..Default::default()
        };
        eprintln!(
            "[qwen36-moe lifecycle-timings] {}",
            format_qwen36_lifecycle_timings(&lifecycle_timings),
        );
    }
    eprintln!(
        "[qwen36-moe runtime telemetry] resident_allocations={} mapped_virtual_ranges={} source_bytes={} device_upload_bytes={}",
        evidence.resident_allocation_count,
        evidence.mapped_virtual_ranges.len(),
        evidence.source_bytes,
        evidence.device_upload_bytes
    );

    Ok(())
}

pub fn run(cli: &crate::Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    run_inner(cli, entry, total_vram, None)
}

/// SpecPrefill sparse-prefill variant. `keep_mask[i] == true` means the
/// drafter selected prompt token `i` to be included in the target's
/// prefill; pruned positions are skipped entirely. The mask must be the
/// same length as the tokenized prompt (validated downstream); the
/// drafter side guarantees the last prompt token is kept (its logits
/// produce the first generation token). Inside the prefill loop, kept
/// tokens use their original prompt position for RoPE rotation but land
/// in compact KV-cache slots via `Qwen36MoeAttnStepParams::cache_pos`,
/// the same kernel-side split MTP already uses for draft-step rotation.
pub fn run_with_sparse_prefill(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    total_vram: u64,
    keep_mask: Vec<bool>,
) -> Result<()> {
    run_inner(cli, entry, total_vram, Some(keep_mask))
}

fn run_inner(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    total_vram: u64,
    keep_mask: Option<Vec<bool>>,
) -> Result<()> {
    let (context_size, context_size_source) = resolve_context_size(cli);
    validate_persistent_kv_fp8_flags(cli)?;
    validate_cuda_v1_flags(cli, entry)?;
    validate_metal_v1_flags(cli, entry)?;

    let sampling = SamplingParams {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        seed: cli.sampling_seed,
    };
    let runtime_modes = crate::qwen36_moe_cli::options::runner_mode_options_from_environment();
    if qwen36_plain_flm_runtime_path(cli, entry, keep_mask.as_ref(), &runtime_modes) {
        let flm_path = effective_flm_source(cli)
            .expect("plain FLM runtime path requires an effective FLM source")
            .to_owned();
        return run_with_runtime_engine(
            cli,
            entry,
            &flm_path,
            context_size,
            context_size_source,
            total_vram,
            sampling,
            runtime_modes.execution,
        );
    }

    let mut startup_timings = Qwen36StartupTimings::default();
    let flm_source_open_start = std::time::Instant::now();
    let flm_source = open_qwen36_moe_flm_source(cli)?;
    if let Some(flm) = flm_source.as_ref() {
        startup_timings.flm_source_open = flm_source_open_start.elapsed();
        startup_timings.flm_store_open = flm.timings.store_open;
        startup_timings.flm_config = flm.timings.config;
        startup_timings.flm_tokenizer = flm.timings.tokenizer;
        startup_timings.flm_tokenizer_assets = flm.timings.tokenizer_assets;
        startup_timings.flm_tokenizer_parse = flm.timings.tokenizer_parse;
        startup_timings.flm_tokenizer_build = flm.timings.tokenizer_build;
        startup_timings.flm_direct_plan = flm.timings.direct_plan;
    }
    if flm_source.is_none() {
        let bake_prepare_start = std::time::Instant::now();
        ensure_qwen36_bake(cli, entry)?;
        startup_timings.bake_prepare = bake_prepare_start.elapsed();
    }

    let dry_run_start = std::time::Instant::now();
    let report = if let Some(flm) = flm_source.as_ref() {
        run_qwen36_moe_dry_run_with_config(
            &cli.model_dir,
            Some(&flm.source.path),
            Some(flm.direct_profile),
            flm.config.clone(),
            entry,
            total_vram,
            context_size,
            context_size_source,
            cli.batch_size.max(1),
            cli.kv_fp8,
            cli.no_bake,
            cli.device,
        )?
    } else {
        run_qwen36_moe_dry_run(
            &cli.model_dir,
            entry,
            total_vram,
            context_size,
            context_size_source,
            cli.batch_size.max(1),
            cli.kv_fp8,
            cli.no_bake,
            cli.device,
        )?
    };
    startup_timings.dry_run = dry_run_start.elapsed();
    print_report(&report);
    startup_timings.print_if_requested(cli.emit_stage_timings);
    if cli.dry_run {
        return Ok(());
    }

    validate_decode_backend(entry)?;

    println!();
    println!("=== Decode (Qwen3.6-MoE) ===");
    let requires_int4_bake = cli.int4 || matches!(entry.backend, Backend::Cuda | Backend::Metal);
    decode_text(
        &cli.model_dir,
        &report,
        &cli.prompt,
        cli.max_new_tokens.max(1),
        sampling,
        cli.emit_stage_timings,
        cli.speculative_decode,
        crate::bakes::effective_quant_profile(cli)?,
        requires_int4_bake,
        cli.batched_spec_verify,
        entry.backend,
        cli.device,
        // Phase 3e.4: persistent decode is now the default. The legacy
        // `--persistent-decode` flag is a hidden no-op (kept for harness
        // back-compat); `--no-persistent-decode` is the documented
        // opt-out for A/B comparison or bisecting megakernel regressions.
        entry.backend != Backend::Metal && !cli.no_persistent_decode,
        cli.kv_fp8,
        flm_source.as_ref(),
        cli.dump_last_logits,
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        &cli.model,
        cli.ignore_eos,
        keep_mask,
        cli.progress_heartbeat_seconds,
        cli.flm_virtual_transfer_backend.as_deref(),
    )?;
    Ok(())
}

/// Tokenize the prompt and run the multi-token decode loop end-to-end:
/// prefill the prompt one token at a time, then generate `max_new`
/// tokens via the configured sampling policy. Streams decoded text to stdout
/// as each token arrives.
///
/// State persistence across decode steps:
///  - Linear-attn `conv_state` + `recurrent_state` mutated in place by
///    the kernel.
///  - Full-attn KV cache: per-layer `[kv_max_t, Hkv*d]` buffers; the kernel
///    writes the current step's K/V at slot `position` and attends over
///    `kv_len = position + 1` past tokens. `kv_max_t` is sized for
///    `prompt_len + max_new` here.
///  - Persistent decode is the default path. The host-orchestrated chained
///    path remains available behind `--no-persistent-decode` for parity and
///    regression isolation.
///  - When self-speculative decode is enabled, each generation iteration can
///    append extra accepted MTP drafts after the regular base-model sample.
fn decode_text(
    model_dir: &Path,
    report: &DryRunReport,
    prompt: &str,
    max_new: usize,
    sampling: SamplingParams,
    emit_stage_timings: bool,
    speculative_decode: bool,
    quant_profile: QuantProfile,
    int4_runtime: bool,
    batched_spec_verify: bool,
    backend: Backend,
    ordinal: usize,
    persistent_decode: bool,
    kv_fp8: bool,
    flm_source: Option<&Qwen36MoeFlmSource>,
    dump_last_logits: bool,
    profile_prefill: bool,
    profile_prefill_json: Option<&Path>,
    model_name: &str,
    ignore_eos: bool,
    keep_mask: Option<Vec<bool>>,
    progress_heartbeat_seconds: f64,
    flm_virtual_transfer_backend_cli: Option<&str>,
) -> Result<()> {
    let runtime_options = crate::qwen36_moe_cli::options::execution_options_from_environment();

    validate_speculative_sampling(speculative_decode, sampling)?;

    if keep_mask.is_some() && speculative_decode {
        eprintln!(
            "[specprefill+mtp] composed run: rope on absolute prompt timeline, \
             cache on compact KV slot. See \
             docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md."
        );
    }

    let decode_wall_start = std::time::Instant::now();
    let progress_interval = (progress_heartbeat_seconds > 0.0)
        .then(|| std::time::Duration::from_secs_f64(progress_heartbeat_seconds));
    let mut last_progress = decode_wall_start
        .checked_sub(progress_interval.unwrap_or(std::time::Duration::ZERO))
        .unwrap_or(decode_wall_start);
    let mut progress = |phase: &str, detail: String, force: bool| {
        let Some(interval) = progress_interval else {
            return;
        };
        let now = std::time::Instant::now();
        if force || now.duration_since(last_progress) >= interval {
            eprintln!(
                "[qwen36-moe progress] phase={phase} elapsed_ms={} {detail}",
                now.duration_since(decode_wall_start).as_millis()
            );
            last_progress = now;
        }
    };
    let weight_prefix = report.kernel_params.weight_prefix;

    progress("prompt_setup", "start".to_string(), true);
    let prompt_setup_start = std::time::Instant::now();
    let mut flm_tokenizer_elapsed = std::time::Duration::ZERO;
    let mut flm_tokenizer_timings = crate::flm_tokenizer::QwenBpeTokenizerTimings::default();
    let prompt_setup = if let Some(flm) = flm_source {
        eprintln!("[qwen36-moe] loading tokenizer from FLM assets");
        let tokenizer_start = std::time::Instant::now();
        let tokenizer_load = flm.load_tokenizer_timed()?;
        flm_tokenizer_elapsed = tokenizer_start.elapsed();
        flm_tokenizer_timings = tokenizer_load.timings;
        prepare_prompt_with_tokenizer(
            Some(tokenizer_load.tokenizer),
            &report.config.text_config,
            prompt,
        )?
    } else {
        prepare_prompt(model_dir, &report.config.text_config, prompt)?
    };
    let prompt_setup_elapsed = prompt_setup_start.elapsed();
    let tokenizer = prompt_setup.tokenizer;
    let prompt_ids = prompt_setup.prompt_ids;
    let eos_id = if ignore_eos {
        None
    } else {
        prompt_setup.eos_id
    };
    print_prompt_summary(prompt, &prompt_ids);

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(backend);

    // KV cache size: needs to fit prompt_len + max_new past tokens. Sized
    // generously here since per-layer KV is small (10 full-attn layers ×
    // [kv_max_t, Hkv*d=512] BF16 = 10 KiB per token of context).
    let kv_max_t = prompt_ids.len() + max_new;

    let kv_vmm = should_use_qwen36_kv_vmm(backend, ordinal)?;

    progress(
        "prompt_setup",
        format!("done prompt_tokens={}", prompt_ids.len()),
        true,
    );
    let source_phase = if flm_source.is_some() {
        "flm_source"
    } else {
        "bake_open"
    };
    progress(source_phase, "start".to_string(), true);
    let source_open_start = std::time::Instant::now();
    let (decode_store, weight_mode, source_label) = if let Some(flm) = flm_source {
        let source_label = flm.source.path.display().to_string();
        validate_qwen36_decode_weight_mode(flm.weight_mode, backend, &source_label)?;
        println!(
            "[qwen36-moe] loading weights from already-open FLM source at {} ({})",
            source_label, flm.weight_mode_label,
        );
        (
            DecodeStore::Borrowed(&flm.source.store),
            flm.weight_mode,
            source_label,
        )
    } else {
        let bake = select_decode_bake(model_dir, quant_profile, int4_runtime)?;
        let source_label = bake.bake_dir.display().to_string();
        validate_qwen36_decode_weight_mode(bake.weight_mode, backend, &source_label)?;
        println!(
            "  loading from bake: {} ({})",
            source_label,
            bake.weight_mode.display_name(),
        );
        let store = BakedStore::open(&bake.bake_dir)
            .with_context(|| format!("open BakedStore at {}", bake.bake_dir.display()))?;
        (DecodeStore::Owned(store), bake.weight_mode, source_label)
    };
    let store = decode_store.as_store();
    let model_source_elapsed = source_open_start.elapsed();
    progress(source_phase, format!("done source={source_label}"), true);
    let persistent_decode = if persistent_decode && !weight_mode.supports_persistent_decode() {
        eprintln!(
            "  persistent decode is unavailable for {}; routing through runtime chained decode",
            weight_mode.display_name()
        );
        false
    } else {
        persistent_decode
    };
    let mut moe_runtime = prepare_moe_runtime_config(
        speculative_decode,
        persistent_decode,
        backend,
        geom.top_k as usize,
        flm_virtual_transfer_backend_cli,
    )?;

    println!(
        "  loading {} layers ({} INT4 sidecar sets, KV cache cap = {} tokens)…",
        geom.num_layers,
        if weight_mode.is_int4() {
            geom.num_layers
        } else {
            0
        },
        kv_max_t,
    );

    progress(
        "layer_load",
        format!(
            "start layers={} kv_max_t={} sparse_vmm={:?}",
            geom.num_layers, kv_max_t, moe_runtime.vmm_mode
        ),
        true,
    );
    let layer_load_start = std::time::Instant::now();
    let profile_layer_load_hal = qwen36_should_profile_layer_load_hal(
        emit_stage_timings,
        qwen36_external_hal_profile_env_active(),
    );
    if profile_layer_load_hal {
        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
    }
    let effective_moe_vmm_mode = effective_moe_expert_vmm_mode_for_transfer_backend(
        moe_runtime.vmm_mode,
        moe_runtime.island_cap_experts,
        moe_runtime.virtual_transfer_backend,
    )?;
    let loaded_layers = match load_decode_layers_with_vmm_strategy(
        &store,
        ordinal,
        backend,
        &geom,
        &report.config.text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        effective_moe_vmm_mode,
        moe_runtime.island_cap_experts,
        moe_runtime.protected_experts,
        moe_runtime.fixed_hot_experts,
        moe_runtime.prefetch_mode,
        moe_runtime.prefetch_ranks,
        moe_runtime.transition_min_observations,
        moe_runtime.async_prefetch,
        moe_runtime.async_staging_pages,
        moe_runtime.prefetch_evict,
        moe_runtime.prefetch_evict_min_probability,
        moe_runtime.virtual_transfer_backend,
        persistent_decode,
    ) {
        Ok(loaded_layers) => loaded_layers,
        Err(err) => {
            if profile_layer_load_hal {
                gpu_hal::hal_profile_set_enabled(false);
            }
            return Err(err);
        }
    };
    let mut layer_load_profile = loaded_layers.timings;
    let mut loaded_layers = loaded_layers.loaded;
    let ffn_prewarm_elapsed = match prewarm_qwen36_mps_static_topn_if_requested(
        ordinal,
        backend,
        &geom,
        loaded_layers.layers_mut_before_persistent()?,
    ) {
        Ok(elapsed) => elapsed,
        Err(err) => {
            if profile_layer_load_hal {
                gpu_hal::hal_profile_set_enabled(false);
            }
            return Err(err);
        }
    };
    layer_load_profile.prewarm = ffn_prewarm_elapsed;
    if profile_layer_load_hal {
        let hal_timings = qwen36_layer_load_hal_timings(&gpu_hal::hal_profile_snapshot());
        layer_load_profile = Qwen36LayerLoadTimings {
            hal_total: hal_timings.hal_total,
            hal_alloc: hal_timings.hal_alloc,
            hal_copy_h_to_d: hal_timings.hal_copy_h_to_d,
            hal_memset: hal_timings.hal_memset,
            hal_vmm: hal_timings.hal_vmm,
            hal_alloc_bytes: hal_timings.hal_alloc_bytes,
            hal_copy_h_to_d_bytes: hal_timings.hal_copy_h_to_d_bytes,
            hal_memset_bytes: hal_timings.hal_memset_bytes,
            hal_vmm_bytes: hal_timings.hal_vmm_bytes,
            ..layer_load_profile
        };
        gpu_hal::hal_profile_set_enabled(false);
    }
    let layer_load_elapsed = layer_load_start.elapsed();
    let virtual_kv_stats = virtual_kv_stats_for_layers(loaded_layers.layers());
    print_virtual_kv_stats_if_active(virtual_kv_stats, kv_fp8, backend, ordinal);
    progress(
        "layer_load",
        format!("done elapsed_ms={}", layer_load_elapsed.as_millis()),
        true,
    );
    progress("session", "start".to_string(), true);
    let session_start = std::time::Instant::now();
    let max_speculative_tokens = max_speculative_tokens_for_backend(backend);
    if speculative_decode && backend == Backend::Metal && metal_mtp_experiment_enabled() {
        eprintln!(
            "[qwen36-mtp-metal-experiment] enabled=1 max_drafts={} verify=sequential status=experimental",
            max_speculative_tokens
        );
    }
    let session = prepare_decode_session(
        &store,
        ordinal,
        &geom,
        &report.config.text_config,
        weight_prefix,
        kv_max_t,
        speculative_decode,
        batched_spec_verify,
        persistent_decode,
        max_speculative_tokens,
        &mut loaded_layers,
    )?;
    let session_elapsed = session_start.elapsed();
    progress(
        "session",
        format!("done elapsed_ms={}", session_elapsed.as_millis()),
        true,
    );
    let Qwen36DecodeSession {
        final_norm_w_buf,
        lm_head_w_buf,
        mut logits_buf,
        mut counter_buf,
        mut final_hidden_buf,
        mut mtp_buffers,
        mut mtp_forward_scratch,
        mut mtp_chain_scratch,
        embed_w_buf,
        mut linear_attn_snapshot,
    } = session;

    print_decode_stream_start(tokenizer.as_ref(), &prompt_ids, max_new);

    // Sparse-prefill setup. `kept_positions[i]` holds the original prompt
    // position of the i-th kept token; the prefill loop iterates over
    // these positions instead of every prompt token. In the dense case
    // (keep_mask=None) it's just `0..prompt_ids.len()` and the loop is
    // bit-equal to before. The drafter side (run_specprefill_qwen36_moe)
    // guarantees `keep_mask.last() == true` and `keep_mask.len() ==
    // prompt_ids.len()`; we re-validate as a defence against future
    // mis-wiring.
    let kept_positions: Vec<usize> = match &keep_mask {
        Some(mask) => {
            if mask.len() != prompt_ids.len() {
                anyhow::bail!(
                    "sparse-prefill: keep_mask.len()={} != prompt_ids.len()={}",
                    mask.len(),
                    prompt_ids.len(),
                );
            }
            let kept: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| k.then_some(i))
                .collect();
            if kept.is_empty() {
                anyhow::bail!("sparse-prefill: keep_mask kept zero positions");
            }
            if *kept.last().unwrap() != prompt_ids.len() - 1 {
                anyhow::bail!(
                    "sparse-prefill: last prompt position must be kept (got last kept={})",
                    kept.last().unwrap()
                );
            }
            kept
        }
        None => (0..prompt_ids.len()).collect(),
    };
    let effective_prompt_len = kept_positions.len();
    if keep_mask.is_some() {
        eprintln!(
            "[specprefill] sparse prefill: {}/{} prompt tokens kept",
            effective_prompt_len,
            prompt_ids.len(),
        );
    }
    if kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_enabled() {
        kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_configure(
            loaded_layers.len(),
            geom.top_k as usize,
            geom.num_experts as usize,
            crate::qwen36_moe_cli::batched_prefill::PREFILL_CHUNK_SIZE_WMMA_FULL,
            effective_prompt_len.saturating_sub(1),
        );
    }
    let backend_label = format!("{backend:?}");
    let mut prefill_profile = Some(PrefillProfileScope::new(
        profile_prefill,
        profile_prefill_json,
        "qwen3.6-moe",
        model_name,
        &backend_label,
        effective_prompt_len,
    ));

    // `Qwen36DecodeLoopState::new` assumes dense (every position kept).
    // For sparse, override the initial token to be the first *kept*
    // prompt token and shrink `total_steps` to `effective_prompt_len +
    // max_new - 1`. `position` (the loop's compact KV-slot counter) and
    // `current_token` advance per chain-step iteration.
    let mut loop_state = Qwen36DecodeLoopState::new(&prompt_ids, max_new);
    if keep_mask.is_some() {
        loop_state.current_token = prompt_ids[kept_positions[0]];
        loop_state.total_steps = effective_prompt_len + max_new - 1;
    }
    let mut rng = XorshiftRng::new(sampling.seed);
    print_sampling_summary(sampling);

    // Per-stage wall-clock accumulators. Aggregated across generation steps
    // only (prefill steps run the chain but skip the lm_head/sample stages,
    // so timing prefill mixed with gen would distort the per-token average).
    // `chain_ms` includes the GPU work + the D2H copy of `final_hidden_bytes`
    // — `run_chained_decode` syncs before returning, so the wall-clock here
    // is a real GPU+sync measurement. CPU-side stages (embed lookup, lm_head
    // GEMV, sampling, detokenize) are pure host work.
    let mut stage_timings = Qwen36StageTimingTotals::default();
    let mtp_acceptance_profile =
        std::env::var_os("SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE").is_some();
    let mut mtp_acceptance_stats =
        speculative_decode.then(|| Qwen36MtpAcceptanceStats::new(batched_spec_verify));
    let mut prefill_steps = 0usize;
    let mut prefill_embed_elapsed = std::time::Duration::ZERO;
    let mut prefill_chain_elapsed = std::time::Duration::ZERO;
    let mut generation_wall_start = None;
    let mut decode_profile = None;
    let mut moe_routes = MoeRouteRuntime::new(
        geom.num_layers as usize,
        geom.top_k as usize,
        moe_runtime.sparse_requested,
        moe_runtime.prefetch_mode,
        moe_runtime.transition_min_observations,
        moe_runtime.hot_protect_min_hits,
        moe_runtime.fixed_hot_min_hits,
    );
    // Batched-Q prefill opt-in. Read once. When set the new chunked
    // host orchestrator drives the prefill range
    // `[0, effective_prompt_len - 1)` instead of the engine's main
    // per-step loop — see
    // docs/superpowers/plans/2026-05-05-qwen36-moe-batched-prefill-phase1.md.
    // M13: batched-prefill is the DEFAULT for Qwen 3.6 MoE. Bench at
    // 4K context (gfx1100, qwen3.6-35b-a3b INT4) shows 1.79x prefill
    // speedup vs the per-token persistent megakernel. Set
    // SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0 to revert to the legacy
    // per-token path (kept as a bisect/escape hatch).
    let batched_prefill_disabled = std::env::var("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL")
        .map(|v| v == "0")
        .unwrap_or(false);

    let mut start_step = 0usize;
    let dense_prefill_token_loop =
        std::env::var_os("SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP").is_some();
    if batched_prefill_disabled
        && dense_prefill_token_loop
        && keep_mask.is_none()
        && !loaded_layers.has_sparse_expert_residency()
        && effective_prompt_len > 1
    {
        if let Some(embed_w) = embed_w_buf
            .as_ref()
            .filter(|_| loaded_layers.persistent_enabled())
        {
            let dense_prefill_count = effective_prompt_len - 1;
            let t_prefill = loaded_layers
                .run_dense_prefill_tokens_from_device_embedding(
                    ordinal,
                    embed_w,
                    &prompt_ids[..dense_prefill_count],
                    0,
                    0,
                )
                .context("persistent dense prefill token loop")?;
            start_step = dense_prefill_count;
            prefill_steps += dense_prefill_count;
            prefill_chain_elapsed += t_prefill;
            loop_state.position += dense_prefill_count as i32;
            loop_state.current_token = prompt_ids[dense_prefill_count];
            eprintln!(
                "[qwen36-moe prefill-progress] mode=dense-token-loop variant=legacy \
                 chunks=1 tokens={} prefill_tokens={} last_context={} embed_ms={:.3} \
                 chain_ms={:.3} elapsed_ms={:.3}",
                dense_prefill_count,
                dense_prefill_count,
                dense_prefill_count,
                0.0,
                t_prefill.as_secs_f64() * 1000.0,
                t_prefill.as_secs_f64() * 1000.0,
            );
        }
    }

    if start_step == 0 && !batched_prefill_disabled && effective_prompt_len > 1 {
        let timings = crate::qwen36_moe_cli::batched_prefill::run_batched_prefill_stub(
            ordinal,
            &geom,
            &store,
            weight_prefix,
            &mut loaded_layers,
            &mut moe_runtime,
            &mut moe_routes,
            &mut loop_state,
            &prompt_ids,
            keep_mask.as_ref(),
            &kept_positions,
            effective_prompt_len,
            emit_stage_timings,
            &runtime_options,
        )?;
        eprintln!(
            "[qwen36-moe batched-prefill] chunks={} tokens={} embed_ms={:.1} chain_ms={:.1}",
            timings.chunks,
            timings.tokens,
            timings.embed_total.as_secs_f64() * 1000.0,
            timings.chain_total.as_secs_f64() * 1000.0,
        );
        prefill_steps += timings.tokens;
        prefill_embed_elapsed += timings.embed_total;
        prefill_chain_elapsed += timings.chain_total;
        // After the orchestrator processes prefill steps
        // [0, effective_prompt_len - 1), the engine's main loop must
        // resume at the FIRST generation step (where logits are
        // computed). At that point `loop_state.position ==
        // effective_prompt_len - 1` (incremented once per processed
        // token) and `loop_state.current_token` is the LAST prompt
        // token (the one to fold into logits in the gen step).
        start_step = effective_prompt_len - 1;
    }

    for step in start_step..loop_state.total_steps {
        // When speculative decode is on, each iteration can commit
        // multiple tokens (up to K+1), so the standard `total_steps =
        // prompt_len + max_new - 1` count over-shoots. Break here once
        // we've already committed `max_new` tokens — otherwise the
        // next regular chain call would request a cache slot beyond
        // `kv_max_t = prompt_len + max_new` (status 120). Plain decode
        // stays bit-identical because it always emits exactly one
        // token per iteration.
        if loop_state.reached_max_new() {
            break;
        }
        let is_gen_step = step + 1 >= effective_prompt_len;
        if is_gen_step && generation_wall_start.is_none() {
            if let Some(profile) = prefill_profile.take() {
                profile.finish()?;
            }
            generation_wall_start = Some(std::time::Instant::now());
            decode_profile = Some(Qwen36DecodeProfileScope::new_from_env());
        }
        // Per-step (rope, cache) pair. Dense mode: rope == cache.
        // SpecPrefill mode: rope on absolute prompt timeline, cache
        // on compact slot count. See `current_position` above.
        let position = current_position(
            step,
            loop_state.position,
            keep_mask.as_ref(),
            &kept_positions,
            effective_prompt_len,
            prompt_ids.len(),
        );
        progress(
            if is_gen_step { "generate" } else { "prefill" },
            format!(
                "step={} total_steps={} rope_position={} cache_position={} generated={} current_token={}",
                step,
                loop_state.total_steps,
                position.rope,
                position.cache,
                loop_state.generated_ids.len(),
                loop_state.current_token
            ),
            false,
        );
        if batched_prefill_disabled
            && dense_prefill_token_loop
            && !is_gen_step
            && keep_mask.is_none()
            && !loaded_layers.has_sparse_expert_residency()
        {
            if let Some(embed_w) = embed_w_buf
                .as_ref()
                .filter(|_| loaded_layers.persistent_enabled())
            {
                let t_chain_step = loaded_layers
                    .run_from_device_embedding_no_download(
                        ordinal,
                        embed_w,
                        loop_state.current_token,
                        position.rope,
                        position.cache,
                    )
                    .with_context(|| {
                        format!(
                            "persistent dense prefill from device embedding \
                             (step {}, rope {}, cache {})",
                            step, position.rope, position.cache
                        )
                    })?;
                loop_state.position += 1;
                prefill_steps += 1;
                prefill_chain_elapsed += t_chain_step;
                loop_state.current_token = prompt_ids[kept_positions[step + 1]];
                continue;
            }
        }

        // Embed lookup for the current token.
        let t0 = std::time::Instant::now();
        let (initial_hidden, embed_lookup_timing) = if emit_stage_timings {
            let (row, timing) = lookup_embed_row_timed(
                &store,
                weight_prefix,
                loop_state.current_token as usize,
                geom.hidden as usize,
            )
            .with_context(|| {
                format!(
                    "embed lookup token {} (step {step})",
                    loop_state.current_token
                )
            })?;
            (row, Some(timing))
        } else {
            let row = lookup_embed_row(
                &store,
                weight_prefix,
                loop_state.current_token as usize,
                geom.hidden as usize,
            )
            .with_context(|| {
                format!(
                    "embed lookup token {} (step {step})",
                    loop_state.current_token
                )
            })?;
            (row, None)
        };
        let t_embed_step = t0.elapsed();

        // Run the chain. Linear-attn state mutates in `layers` in place.
        // `run_chained_decode_fast` skips the per-layer D2H sync chain
        // (~80 GPU syncs/token on 35B-A3B) — `decode_text` only consumes
        // `final_hidden_bytes`. The multilayer parity test still calls
        // the legacy `run_chained_decode` which captures per-layer.
        let t1 = std::time::Instant::now();
        // When `--emit-stage-timings` is set, sync after each step launch
        // so the per-stage `kernel_*_us` accumulators in `outputs` reflect
        // GPU compute time. Without it, PR #80's async dispatch path
        // would record host queue time instead — fast but useless for
        // stage-level perf attribution. The total `chain_ms` measured by
        // the wall-clock around this call stays correct either way
        // because `run_chained_decode_fast` ends with a D2H copy that
        // implicitly drains the queue.
        // Phase 3f: on generation steps when the persistent path is
        // active, fold final RMSnorm + lm_head GEMV into the
        // megakernel — saves the separate `lm_head_launch` (one launch
        // + ~30 µs) and the H2D round-trip that staged final_hidden
        // into final_hidden_buf. The host then D2Hs `logits_buf`
        // directly. On prefill steps logits aren't needed; on the
        // chained path the explicit lm_head_launch path stays.
        let disable_folded_lm_head =
            std::env::var_os("SUPERSONIC_QWEN36_DISABLE_FOLDED_LM_HEAD").is_some();
        let folded_top1_enabled = matches!(logits_buf.backend(), Backend::Metal | Backend::Hip)
            && (sampling.temperature <= 0.0 || sampling.top_k == 1)
            && !dump_last_logits
            && std::env::var_os("SUPERSONIC_QWEN36_DUMP_LOGITS").is_none()
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LM_HEAD_GPU_ARGMAX").is_none();
        let fold = if is_gen_step && !disable_folded_lm_head {
            if folded_top1_enabled {
                Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                    final_norm_w: &final_norm_w_buf,
                    lm_head_w: &lm_head_w_buf,
                    logits_out: None,
                    top1_out: Some(&mut counter_buf),
                    vocab: geom.vocab,
                })
            } else {
                Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                    final_norm_w: &final_norm_w_buf,
                    lm_head_w: &lm_head_w_buf,
                    logits_out: Some(&mut logits_buf),
                    top1_out: None,
                    vocab: geom.vocab,
                })
            }
        } else {
            None
        };
        let final_hidden_observer_enabled = std::env::var_os("SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN")
            .is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_FINAL_HIDDEN_TAP").is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP").is_some();
        let download_final_hidden = !is_gen_step
            || fold.is_none()
            || final_hidden_observer_enabled
            || emit_stage_timings
            || mtp_buffers.is_some();
        let chain_step = run_chain_step(Qwen36ChainStep {
            ordinal,
            geom: &geom,
            store: &store,
            loaded_layers: &mut loaded_layers,
            moe_runtime: &mut moe_runtime,
            moe_routes: &mut moe_routes,
            initial_hidden: &initial_hidden,
            position,
            step,
            is_gen_step,
            emit_stage_timings,
            fold,
            download_final_hidden,
            execution: &runtime_options,
        })?;
        let outputs = chain_step.outputs;
        let lm_head_folded = chain_step.lm_head_folded;
        let lm_head_folded_top1 = chain_step.lm_head_folded_top1;
        let t_chain_step = t1.elapsed();
        loop_state.position += 1;

        // KV-FP8 sidecar descriptors stay fixed across decode. The
        // persistent kernel computes the rolling covered range from
        // `position` and `kv_shadow_window`, so no descriptor re-upload is
        // needed when old sidecar slots roll over.

        // Prefill steps: feed the next prompt token without computing logits.
        // For sparse-prefill, the next "prompt token" is the next *kept*
        // prompt token (`kept_positions[step + 1]` indexes into the
        // original prompt).
        if step + 1 < effective_prompt_len {
            prefill_steps += 1;
            prefill_embed_elapsed += t_embed_step;
            prefill_chain_elapsed += t_chain_step;
            loop_state.current_token = prompt_ids[kept_positions[step + 1]];
            continue;
        }

        let next_token = run_generation_step(Qwen36GenerationStep {
            ordinal,
            geom: &geom,
            step,
            lm_head_folded,
            lm_head_folded_top1,
            dump_last_logits,
            tokenizer: tokenizer.as_ref(),
            sampling,
            t_embed_step,
            embed_lookup_timing,
            t_chain_step,
            outputs: &outputs,
            execution: &runtime_options,
            final_norm_w_buf: &final_norm_w_buf,
            lm_head_w_buf: &lm_head_w_buf,
            final_hidden_buf: &mut final_hidden_buf,
            logits_buf: &mut logits_buf,
            counter_buf: &mut counter_buf,
            loop_state: &mut loop_state,
            rng: &mut rng,
            stage_timings: &mut stage_timings,
        })?;
        if let Some(profile) = prefill_profile.take() {
            profile.finish()?;
        }

        if Some(next_token) == eos_id {
            break;
        }
        loop_state.current_token = next_token;

        if let (Some(mtp), Some(fwd_scratch), Some(chain_scratch), Some(embed_w)) = (
            mtp_buffers.as_mut(),
            mtp_forward_scratch.as_mut(),
            mtp_chain_scratch.as_mut(),
            embed_w_buf.as_ref(),
        ) {
            if loop_state.reached_max_new() {
                break;
            }

            let h_base = outputs.final_hidden_bytes.clone();
            // Runs either batched or sequential speculative verify depending on
            // whether session setup allocated a linear-attn snapshot.
            let result = unsafe {
                loaded_layers.with_experimental_parts(|layers, persistent_scratch| {
                    run_speculative_extension(Qwen36SpeculativeExtension {
                        ordinal,
                        geom: &geom,
                        store: &store,
                        weight_prefix,
                        layers,
                        execution: &runtime_options,
                        persistent_scratch,
                        mtp,
                        forward_scratch: fwd_scratch,
                        chain_scratch,
                        embed_w,
                        final_norm_w: &final_norm_w_buf,
                        lm_head_w: &lm_head_w_buf,
                        final_hidden: &mut final_hidden_buf,
                        logits: &mut logits_buf,
                        counter: &mut counter_buf,
                        linear_attn_snapshot: linear_attn_snapshot.as_mut(),
                        loop_state: &loop_state,
                        base_position: position,
                        h_base_in: &h_base,
                        first_token: next_token,
                        stage_timings: &mut stage_timings,
                        emit_stage_timings,
                        max_drafts: max_speculative_tokens,
                    })
                })?
            };

            if let Some(stats) = mtp_acceptance_stats.as_mut() {
                stats.record(&result);
            }
            if loop_state.append_speculative_emissions(&result, tokenizer.as_ref(), eos_id) {
                break;
            }
        }
    }

    if let Some(profile) = prefill_profile.take() {
        profile.finish()?;
    }
    if let Some(profile) = decode_profile.take() {
        profile.finish();
    }

    print_last_logits_if_requested(dump_last_logits, &loop_state.last_logits_bytes);
    let generation_wall_ms = generation_wall_start
        .as_ref()
        .map(|start| start.elapsed().as_secs_f64() * 1000.0);
    print_generation_summary(
        &loop_state.generated_ids,
        prompt_ids.len(),
        eos_id,
        generation_wall_ms,
    );
    if let Some(manager) = loaded_layers.sparse_expert_residency() {
        print_and_write_moe_residency_summary(
            manager,
            virtual_kv_stats,
            &loop_state.generated_ids,
            moe_routes.route_telemetry.as_ref(),
            moe_runtime.sparse_telemetry.as_ref(),
        )?;
    }
    stage_timings.print_if_requested(emit_stage_timings);
    if emit_stage_timings {
        let lifecycle_timings = Qwen36LifecycleTimings {
            prompt_setup: prompt_setup_elapsed,
            flm_tokenizer: flm_tokenizer_elapsed,
            flm_tokenizer_assets: flm_tokenizer_timings.asset_lookup,
            flm_tokenizer_parse: flm_tokenizer_timings.parse,
            flm_tokenizer_parse_vocab: flm_tokenizer_timings.parse_vocab,
            flm_tokenizer_parse_vocab_ids: flm_tokenizer_timings.parse_vocab_ids,
            flm_tokenizer_parse_merges: flm_tokenizer_timings.parse_merges,
            flm_tokenizer_parse_added_tokens: flm_tokenizer_timings.parse_added_tokens,
            flm_tokenizer_parse_regex: flm_tokenizer_timings.parse_regex,
            flm_tokenizer_build: flm_tokenizer_timings.build,
            model_source: model_source_elapsed,
            layer_load: layer_load_elapsed,
            layer_load_profile,
            session: session_elapsed,
            prefill_steps,
            prefill_embed: prefill_embed_elapsed,
            prefill_chain: prefill_chain_elapsed,
            generation_wall: generation_wall_ms,
            total_wall: decode_wall_start.elapsed(),
        };
        eprintln!(
            "[qwen36-moe lifecycle-timings] {}",
            format_qwen36_lifecycle_timings(&lifecycle_timings),
        );
    }
    if let Some(stats) = mtp_acceptance_stats.as_ref() {
        stats.print_if_requested(mtp_acceptance_profile || emit_stage_timings);
    }
    emit_mpp_pilot_if_requested(emit_stage_timings);
    emit_mps_expert_pilot_if_requested(emit_stage_timings);

    Ok(())
}

fn emit_mpp_pilot_if_requested(emit_stage_timings: bool) {
    if !emit_stage_timings || std::env::var_os("SUPERSONIC_METAL_QWEN36_MPP_PILOT").is_none() {
        return;
    }
    let size = std::env::var("SUPERSONIC_METAL_QWEN36_MPP_PILOT_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(2048);
    let iterations = std::env::var("SUPERSONIC_METAL_QWEN36_MPP_PILOT_ITERS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(5);
    match kernel_ffi::qwen36_moe::metal_mpp_tile_gemm_f16_tflops(size, iterations) {
        Ok(tflops) => eprintln!(
            "[qwen36-moe mpp-pilot] status=ok size={} iterations={} tile_m=64 tile_n=32 tile_k=64 tflops={:.3}",
            size, iterations, tflops
        ),
        Err(err) => eprintln!(
            "[qwen36-moe mpp-pilot] status=error size={} iterations={} tflops=0.000 error={}",
            size, iterations, err
        ),
    }
}

fn emit_mps_expert_pilot_if_requested(emit_stage_timings: bool) {
    if !emit_stage_timings || std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT").is_none()
    {
        return;
    }
    let hidden = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_HIDDEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2048);
    let moe_intermediate = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_MOE_INTERMEDIATE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let top_k = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let iterations = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_ITERS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(100);
    match kernel_ffi::qwen36_moe::metal_mps_expert_f16_probe(
        hidden,
        moe_intermediate,
        top_k,
        iterations,
    ) {
        Ok(probe) => eprintln!(
            "[qwen36-moe mps-expert-pilot] status=ok hidden={} moe_intermediate={} top_k={} iterations={} gate_up_ms={:.3} down_ms={:.3} gate_up_tflops={:.3} down_tflops={:.3}",
            hidden,
            moe_intermediate,
            top_k,
            iterations,
            probe.gate_up_ms,
            probe.down_ms,
            probe.gate_up_tflops,
            probe.down_tflops,
        ),
        Err(err) => eprintln!(
            "[qwen36-moe mps-expert-pilot] status=error hidden={} moe_intermediate={} top_k={} iterations={} gate_up_ms=0.000 down_ms=0.000 gate_up_tflops=0.000 down_tflops=0.000 error={}",
            hidden, moe_intermediate, top_k, iterations, err
        ),
    }
}
