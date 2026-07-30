use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use gpu_hal::Backend;
use model_store::BakedStore;

use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::engine::current_position;
use crate::qwen36_moe_cli::host::lookup_embed_row;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_telemetry::MoeRouteRuntime;
use crate::qwen36_moe_types::MultiLayerGeom;
use supersonic_runtime::qwen36_moe::layers::LoadedQwen36Layers;

pub(crate) use supersonic_runtime::qwen36_moe::prefill::{
    BatchedPrefillTimings, PREFILL_CHUNK_SIZE_WMMA_FULL,
};

fn batched_prefill_variant_label(backend: Backend) -> String {
    let force_host_native = std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_some();
    let mut parts = if backend == Backend::Metal {
        if std::env::var_os("SUPERSONIC_QWEN36_MOE_METAL_BATCHED_PREFILL_PROTOTYPE").is_some() {
            vec!["metal-prototype"]
        } else {
            vec!["metal-default"]
        }
    } else {
        vec!["batched"]
    };
    if force_host_native {
        parts.push("force-host-native");
    }

    let metal_native_enabled = backend == Backend::Metal && !force_host_native;
    let full_attn_tmajor_enabled = metal_native_enabled
        && std::env::var("SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_TMAJOR")
            .map(|v| v != "0")
            .unwrap_or(false);
    let full_attn_vec_enabled = metal_native_enabled
        && !full_attn_tmajor_enabled
        && std::env::var("SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_VEC")
            .map(|v| v != "0")
            .unwrap_or(true);
    let shared_expert_batch_enabled = metal_native_enabled
        && std::env::var("SUPERSONIC_QWEN36_MOE_METAL_SHARED_EXPERT_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true);

    if full_attn_tmajor_enabled {
        parts.push("full-attn-tmajor");
    }
    if full_attn_vec_enabled {
        parts.push("full-attn-vec");
    }
    if metal_native_enabled
        && std::env::var("SUPERSONIC_QWEN36_MOE_METAL_ROUTER_TOPK")
            .map(|v| v != "0")
            .unwrap_or(false)
    {
        parts.push("router-topk");
    }
    if shared_expert_batch_enabled {
        parts.push("shared-expert-batch");
    }
    if shared_expert_batch_enabled
        && std::env::var("SUPERSONIC_QWEN36_MOE_METAL_FUSED_FFN_RESIDUAL")
            .map(|v| v != "0")
            .unwrap_or(true)
    {
        parts.push("fused-residual");
    }
    parts.join("+")
}

fn emit_prefill_progress(
    mode: &str,
    variant: &str,
    timings: &BatchedPrefillTimings,
    prefill_tokens: usize,
    last_context: usize,
    elapsed: Duration,
) {
    eprintln!(
        "[qwen36-moe prefill-progress] mode={mode} variant={variant} \
         chunks={} tokens={} prefill_tokens={} last_context={} embed_ms={:.3} \
         chain_ms={:.3} elapsed_ms={:.3}",
        timings.chunks,
        timings.tokens,
        prefill_tokens,
        last_context,
        timings.embed_total.as_secs_f64() * 1000.0,
        timings.chain_total.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}

/// CLI adapter: translate loop state and optional SpecPrefill positions into
/// the runtime's explicit token/position contract. Sparse residency,
/// per-step telemetry, and profiling callbacks remain runner-owned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_batched_prefill_stub(
    ordinal: usize,
    geom: &MultiLayerGeom,
    store: &BakedStore,
    weight_prefix: &str,
    loaded_layers: &mut LoadedQwen36Layers,
    moe_runtime: &mut MoeRuntimeConfig,
    moe_routes: &mut MoeRouteRuntime,
    loop_state: &mut Qwen36DecodeLoopState,
    prompt_ids: &[u32],
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    emit_stage_timings: bool,
) -> Result<BatchedPrefillTimings> {
    let prefill_count = effective_prompt_len.saturating_sub(1);
    if prefill_count == 0 {
        return Ok(BatchedPrefillTimings::default());
    }

    let start_position = loop_state.position;
    let mut tokens = Vec::with_capacity(prefill_count);
    let mut positions = Vec::with_capacity(prefill_count);
    for step in 0..prefill_count {
        let token = if step == 0 {
            loop_state.current_token
        } else {
            prompt_ids[kept_positions[step]]
        };
        tokens.push(token);
        positions.push(current_position(
            step,
            start_position + step as i32,
            keep_mask,
            kept_positions,
            effective_prompt_len,
            prompt_ids.len(),
        ));
    }

    let mut fallback =
        |callback_layers: &mut LoadedQwen36Layers,
         step: usize,
         token: u32,
         position|
         -> Result<supersonic_runtime::qwen36_moe::prefill::PrefillTokenTimings> {
            let embed_start = Instant::now();
            let initial_hidden =
                lookup_embed_row(store, weight_prefix, token as usize, geom.hidden as usize)
                    .with_context(|| {
                        format!("embed lookup token {token} (batched prefill step {step})")
                    })?;
            let embed = embed_start.elapsed();
            let chain_start = Instant::now();
            run_chain_step(Qwen36ChainStep {
                ordinal,
                geom,
                store,
                loaded_layers: callback_layers,
                moe_runtime,
                moe_routes,
                initial_hidden: &initial_hidden,
                position,
                step,
                is_gen_step: false,
                emit_stage_timings,
                fold: None,
                download_final_hidden: true,
            })?;
            Ok(
                supersonic_runtime::qwen36_moe::prefill::PrefillTokenTimings {
                    embed,
                    chain: chain_start.elapsed(),
                },
            )
        };
    let variant = batched_prefill_variant_label(gpu_hal::current_backend());
    let mut progress = |timings: &BatchedPrefillTimings, prefill_tokens, last_context, elapsed| {
        emit_prefill_progress(
            "batched",
            &variant,
            timings,
            prefill_tokens,
            last_context,
            elapsed,
        );
    };

    let timings = supersonic_runtime::qwen36_moe::prefill::run_batched_prefill(
        ordinal,
        geom,
        store,
        weight_prefix,
        loaded_layers,
        &tokens,
        &positions,
        emit_stage_timings,
        Some(&mut fallback),
        Some(&mut progress),
    )?;

    loop_state.position = start_position + prefill_count as i32;
    loop_state.current_token = prompt_ids[kept_positions[prefill_count]];
    Ok(timings)
}
