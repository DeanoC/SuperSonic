use anyhow::{bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use qwen35::state::ModelState;

use super::args::BughuntLayerKind;
use super::runtime::QwenBughuntRuntime;
use super::util::encode_bf16_le;
use crate::decode_engine::DecodeEngine;
use crate::oracle;
use crate::prefill_engine;

pub(crate) fn run_native_prefill(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
) -> Result<prefill_engine::PrefillResult> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("native prefill model state init: {e}"))?;
    prefill_engine::prefill(
        &runtime.weights,
        &mut state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        false,
        None,
    )
}

pub(crate) fn run_native_prefill_greedy_token_with_state(
    runtime: &QwenBughuntRuntime,
    state: &mut ModelState,
    prompt_ids: &[u32],
) -> Result<u32> {
    state.reset_for_prefill_reuse();
    let result = prefill_engine::prefill(
        &runtime.weights,
        state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        false,
        None,
    )?;
    Ok(DecodeEngine::greedy_sample(&result.logits))
}

pub(crate) fn run_native_prefill_with_trace(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<prefill_engine::PrefillResult> {
    let _ = trace_position;
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("native traced prefill model state init: {e}"))?;
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    if debug_full_layer.is_some() || debug_mlp_layer.is_some() {
        bail!("native full-attention/MLP debug traces are not available in the current prefill API");
    }
    prefill_engine::prefill(
        &runtime.weights,
        &mut state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        true,
        debug_linear_layer,
    )
}

pub(crate) fn run_tail_replay_with_trace(
    runtime: &QwenBughuntRuntime,
    hidden_bf16: &[u8],
    start_layer: usize,
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<prefill_engine::PrefillResult> {
    let _ = (
        runtime,
        hidden_bf16,
        start_layer,
        trace_position,
        debug_layer,
        debug_kind,
    );
    bail!("tail replay tracing is not available in the current prefill API")
}

pub(crate) fn run_trace_oracle(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<oracle::Qwen35TraceOutput> {
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    oracle::run_qwen35_trace_oracle(
        &runtime.qwen35_trace_script,
        runtime.model_variant.hf_model_id(),
        prompt_ids,
        0,
        "bf16",
        &runtime.oracle_device,
        debug_linear_layer,
        debug_full_layer,
        debug_mlp_layer,
        trace_position,
    )
}

pub(crate) fn compute_qwen_logits_from_hidden_row(
    runtime: &QwenBughuntRuntime,
    hidden_row: &[f32],
) -> Result<Vec<f32>> {
    let hidden_dim = runtime.weights.config.hidden_size;
    if hidden_row.len() != hidden_dim {
        bail!(
            "hidden row length {} did not match hidden size {}",
            hidden_row.len(),
            hidden_dim
        );
    }
    let hidden_bf16 = encode_bf16_le(hidden_row);
    let hidden_gpu = GpuBuffer::from_host_bytes(
        runtime.ordinal,
        ScalarType::BF16,
        &[1, hidden_dim],
        &hidden_bf16,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row upload: {e}"))?;
    kernel_ffi::qwen_rms_norm_standalone_matvec_host_f32(
        runtime.ordinal,
        ScalarType::BF16,
        &hidden_gpu,
        &runtime.weights.norm_weight,
        runtime.weights.config.rms_norm_eps as f32,
        &runtime.weights.lm_head,
        hidden_dim,
        runtime.weights.config.vocab_size,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row logits: {e}"))
}

fn debug_layer_flags(
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    match (debug_layer, debug_kind) {
        (Some(layer), Some(BughuntLayerKind::Linear)) => (Some(layer), None, None),
        (Some(layer), Some(BughuntLayerKind::Full)) => (None, Some(layer), None),
        (Some(layer), Some(BughuntLayerKind::Mlp)) => (None, None, Some(layer)),
        _ => (None, None, None),
    }
}
