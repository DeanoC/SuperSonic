use anyhow::Result;
use qwen35::state::ModelState;

use crate::decode_engine::{ComponentLayerTrace, ComponentLinearTrace, DecodeEngine};
use crate::prefill_engine;
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le, f32_to_bf16_bytes,
};
use crate::validate;
use crate::qwen35_trace_utils::{build_linear_decode_v_reference, fp8_e4m3_to_f32_host};

pub(crate) fn trace_component_input_layer(
    engine: &DecodeEngine,
    native_hidden: &[u8],
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component input trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = if trace_layer == 0 {
        None
    } else {
        replay
            .layer_hidden_trace
            .as_ref()
            .and_then(|layers| layers.get(trace_layer - 1))
    };
    if let Some(replay_hidden) = replay_hidden {
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(replay_hidden);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!("[trace-component-input] layer={trace_layer} hidden_delta={delta:.6}");
    } else {
        eprintln!(
            "[trace-component-input] layer={trace_layer} has no replay previous-layer hidden reference"
        );
    }
    Ok(())
}

pub(crate) fn trace_persistent_input_layer(
    engine: &DecodeEngine,
    native_hidden: &[u8],
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("persistent input trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = if trace_layer == 0 {
        None
    } else {
        replay
            .layer_hidden_trace
            .as_ref()
            .and_then(|layers| layers.get(trace_layer - 1))
    };
    if let Some(replay_hidden) = replay_hidden {
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(replay_hidden);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!("[trace-persistent-input] layer={trace_layer} hidden_delta={delta:.6}");
    } else {
        eprintln!(
            "[trace-persistent-input] layer={trace_layer} has no replay previous-layer hidden reference"
        );
    }
    Ok(())
}

pub(crate) fn trace_persistent_linear_state_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent linear trace replay state init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
    )?;

    let native_state = engine.state_for_batch(0);
    let native_layer = native_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("native layer {trace_layer} out of range"))?;
    let replay_layer = replay_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} out of range"))?;

    let (conv_delta, first_conv_mismatch) =
        match (&native_layer.conv_state, &replay_layer.conv_state) {
            (Some(native), Some(replay)) => {
                let native_vals = decode_bf16_le(
                    &native
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("native persistent conv trace D2H: {e}"))?,
                );
                let replay_vals = decode_bf16_le(
                    &replay
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("replay persistent conv trace D2H: {e}"))?,
                );
                let delta = validate::max_abs_delta(&native_vals, &replay_vals);
                let first = native_vals
                    .iter()
                    .zip(replay_vals.iter())
                    .enumerate()
                    .find(|(_, (n, r))| (*n - *r).abs() > 0.0)
                    .map(|(idx, (n, r))| (idx, *n, *r));
                (delta, first)
            }
            _ => (0.0, None),
        };
    let (rec_delta, first_rec_mismatch, max_rec_mismatch) =
        match (&native_layer.recurrent_state, &replay_layer.recurrent_state) {
            (Some(native), Some(replay)) => {
                let native_vals =
                    decode_f32_le(&native.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("native persistent recurrent trace D2H: {e}")
                    })?);
                let replay_vals =
                    decode_f32_le(&replay.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("replay persistent recurrent trace D2H: {e}")
                    })?);
                let delta = validate::max_abs_delta(&native_vals, &replay_vals);
                let first = native_vals
                    .iter()
                    .zip(replay_vals.iter())
                    .enumerate()
                    .find(|(_, (n, r))| (*n - *r).abs() > 0.0)
                    .map(|(idx, (n, r))| (idx, *n, *r));
                let max_entry = native_vals
                    .iter()
                    .zip(replay_vals.iter())
                    .enumerate()
                    .max_by(|(_, (na, ra)), (_, (nb, rb))| {
                        (*na - *ra)
                            .abs()
                            .partial_cmp(&(*nb - *rb).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, (n, r))| (idx, *n, *r, (*n - *r).abs()));
                (delta, first, max_entry)
            }
            _ => (0.0, None, None),
        };
    eprintln!(
        "[trace-persistent-linear-state] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={rec_delta:.6}{}{}{}",
        first_conv_mismatch
            .map(|(idx, native, replay)| format!(
                " first_conv_mismatch=(idx={idx},native={native:.9},replay={replay:.9})"
            ))
            .unwrap_or_default(),
        first_rec_mismatch
            .map(|(idx, native, replay)| format!(
                " first_recurrent_mismatch=(idx={idx},native={native:.9},replay={replay:.9})"
            ))
            .unwrap_or_default(),
        max_rec_mismatch
            .map(|(idx, native, replay, delta)| format!(
                " max_recurrent_mismatch=(idx={idx},native={native:.9},replay={replay:.9},delta={delta:.9})"
            ))
            .unwrap_or_default()
    );
    Ok(())
}

pub(crate) fn trace_persistent_full_attn_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    token_ids: &[u32],
    trace_tokens: &[u32],
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let text_config = engine.weights().config.clone();
    anyhow::ensure!(
        text_config.is_full_attention(trace_layer),
        "layer {trace_layer} is not a full-attention layer"
    );
    anyhow::ensure!(
        trace_layer > 0,
        "trace layer must be > 0 for full-attention input tracing"
    );

    let prefix_ids = token_ids
        .get(..token_ids.len().saturating_sub(1))
        .ok_or_else(|| {
            anyhow::anyhow!("missing prefix token ids for persistent full-attn trace")
        })?;
    engine.rebuild_prefill_state(prefix_ids, true)?;

    let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer,
        0,
    )?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let _ = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer + 1,
        0,
    )?;
    let native_gated = engine.trace_persistent_full_attention_gated_after_layers(0)?;
    let native_q = engine.trace_persistent_full_attention_q_after_layers(0)?;
    let native_saved_gate = engine.trace_persistent_full_attention_saved_gate_after_layers(0)?;
    let native_pre_gate = engine.trace_persistent_full_attention_pre_gate_after_layers(0)?;
    let native_scores =
        engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?;
    let (_, _, _, native_token_mixer) =
        engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_component = engine.trace_full_attention_stages_from_hidden(
        trace_layer,
        &native_hidden,
        seqlen_offset,
    )?;
    let native_component_layer = engine
        .trace_full_attention_layer_output_from_hidden_current_state(
            trace_layer,
            0,
            &native_hidden,
            seqlen_offset,
        )?;

    let mut replay_prefix_state = ModelState::new(&text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent full-attn replay prefix state init: {e}"))?;
    let _ = prefill_engine::prefill(
        engine.weights(),
        &mut replay_prefix_state,
        engine.rotary(),
        prefix_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let mut replay_state = ModelState::new(&text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent full-attn replay state init: {e}"))?;
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer - 1))
        .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?;
    let replay_component = engine.trace_full_attention_stages_from_hidden(
        trace_layer,
        replay_hidden,
        seqlen_offset,
    )?;
    let replay_cache_component_layer = engine.trace_full_attention_layer_output_from_hidden_state(
        &replay_prefix_state,
        trace_layer,
        &native_hidden,
        seqlen_offset,
    )?;

    engine.rebuild_prefill_state(prefix_ids, true)?;
    let _ = engine.decode_step_batch(trace_tokens, seqlen_offset)?;
    let native_hidden_f32 = decode_bf16_le(&native_hidden);
    let replay_hidden_f32 = decode_bf16_le(replay_hidden);
    let replay_attn_hidden = replay
        .layer_attn_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay attn trace for layer {trace_layer}"))?;
    let replay_attn_hidden_f32 = decode_bf16_le(replay_attn_hidden);
    let native_normed_f32 = decode_bf16_le(&native_component.normed);
    let replay_normed_f32 = decode_bf16_le(&replay_component.normed);
    let native_q_proj_f32 = decode_bf16_le(&native_component.q_proj);
    let replay_q_proj_f32 = decode_bf16_le(&replay_component.q_proj);
    let native_gate_proj_f32 = decode_bf16_le(&native_component.gate_proj);
    let replay_gate_proj_f32 = decode_bf16_le(&replay_component.gate_proj);
    let native_k_proj_f32 = decode_bf16_le(&native_component.k_proj);
    let replay_k_proj_f32 = decode_bf16_le(&replay_component.k_proj);
    let native_v_proj_f32 = decode_bf16_le(&native_component.v_proj);
    let replay_v_proj_f32 = decode_bf16_le(&replay_component.v_proj);
    let native_q_rope_f32 = decode_bf16_le(&native_component.q_rope);
    let replay_q_rope_f32 = decode_bf16_le(&replay_component.q_rope);
    let native_q_f32 = decode_f32_le(&native_q);
    let native_comp_k_f32 = decode_bf16_le(&native_component.k_rope);
    let native_comp_v_f32 = decode_bf16_le(&native_component.v_proj);
    let replay_comp_k_f32 = decode_bf16_le(&replay_component.k_rope);
    let replay_comp_v_f32 = decode_bf16_le(&replay_component.v_proj);
    let hidden_delta = validate::max_abs_delta(&native_hidden_f32, &replay_hidden_f32);
    let normed_delta = validate::max_abs_delta(&native_normed_f32, &replay_normed_f32);
    let q_proj_delta = validate::max_abs_delta(&native_q_proj_f32, &replay_q_proj_f32);
    let gate_proj_delta = validate::max_abs_delta(&native_gate_proj_f32, &replay_gate_proj_f32);
    let k_proj_delta = validate::max_abs_delta(&native_k_proj_f32, &replay_k_proj_f32);
    let v_proj_delta = validate::max_abs_delta(&native_v_proj_f32, &replay_v_proj_f32);
    let q_rope_delta = validate::max_abs_delta(&native_q_rope_f32, &replay_q_rope_f32);
    let native_vs_component_q = validate::max_abs_delta(&native_q_f32, &native_q_rope_f32);
    let native_vs_replay_k = validate::max_abs_delta(&native_comp_k_f32, &replay_comp_k_f32);
    let native_vs_replay_v = validate::max_abs_delta(&native_comp_v_f32, &replay_comp_v_f32);
    let native_gated_f32 = decode_f32_le(&native_gated);
    let full_weights = engine.weights().layers[trace_layer]
        .full
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing full-attention weights"))?;
    let q_dim = native_gated_f32.len();
    let native_gated_gpu = gpu_hal::GpuBuffer::from_host_bytes(
        ordinal,
        gpu_hal::ScalarType::BF16,
        &[1, q_dim],
        &f32_to_bf16_bytes(&native_gated_f32),
    )
    .map_err(|e| anyhow::anyhow!("trace native gated H2D: {e}"))?;
    let mut native_o_proj_gpu = gpu_hal::GpuBuffer::zeros(
        ordinal,
        gpu_hal::ScalarType::BF16,
        &[1, text_config.hidden_size],
    )
    .map_err(|e| anyhow::anyhow!("trace native o_proj alloc: {e}"))?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        gpu_hal::ScalarType::BF16,
        1,
        1,
        text_config.hidden_size,
        q_dim,
        &native_gated_gpu,
        &full_weights.o_proj_w,
        &mut native_o_proj_gpu,
    )
    .map_err(|e| anyhow::anyhow!("trace native o_proj matmul: {e}"))?;
    let native_host_o_proj_f32 = decode_bf16_le(
        &native_o_proj_gpu
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace native o_proj D2H: {e}"))?,
    );
    let native_saved_gate_f32 = decode_f32_le(&native_saved_gate);
    let native_pre_gate_f32 = decode_f32_le(&native_pre_gate);
    let native_scores_f32 = decode_f32_le(&native_scores);
    let native_comp_gated_f32 = decode_bf16_le(&native_component_layer.gated);
    let native_comp_pre_gate_f32 = decode_bf16_le(&native_component_layer.pre_gate);
    let native_token_mixer_f32 = decode_f32_le(&native_token_mixer);
    let native_comp_token_mixer_f32 = decode_bf16_le(&native_component_layer.attn_hidden);
    let replay_cache_token_mixer_f32 = decode_bf16_le(&replay_cache_component_layer.attn_hidden);
    let mut kv_vs_bf16_pre_gate = None;
    let mut kv_vs_bf16_gated = None;
    let mut kv_vs_bf16_attn_hidden = None;
    let mut kv_vs_bf16_scores = None;
    let mut kv_vs_bf16_scores_heads = None;
    let mut kv_vs_bf16_hidden = None;
    let mut kv_vs_bf16_q = None;
    let mut kv_vs_bf16_saved_gate = None;
    let mut kv_vs_bf16_cache_k = None;
    let mut kv_vs_bf16_cache_v = None;
    if engine.kv_fp8_enabled() {
        let (native_cache_k_bf16, native_cache_v_bf16, _) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        engine.set_kv_fp8_for_trace(false);
        engine.rebuild_prefill_state(prefix_ids, true)?;
        let (bf16_cache_k_bf16, bf16_cache_v_bf16, _) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        let bf16_hidden = decode_bf16_le(&engine.decode_step_batch_trace_hidden_after_layers(
            trace_tokens,
            seqlen_offset,
            trace_layer,
            0,
        )?);
        let _ = engine.decode_step_batch_trace_hidden_after_layers(
            trace_tokens,
            seqlen_offset,
            trace_layer + 1,
            0,
        )?;
        let bf16_q = decode_f32_le(&engine.trace_persistent_full_attention_q_after_layers(0)?);
        let bf16_saved_gate =
            decode_f32_le(&engine.trace_persistent_full_attention_saved_gate_after_layers(0)?);
        let bf16_gated =
            decode_f32_le(&engine.trace_persistent_full_attention_gated_after_layers(0)?);
        let bf16_pre_gate =
            decode_f32_le(&engine.trace_persistent_full_attention_pre_gate_after_layers(0)?);
        let bf16_scores = decode_f32_le(
            &engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?,
        );
        let (_, _, _, bf16_token_mixer) =
            engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
        let bf16_token_mixer_f32 = decode_f32_le(&bf16_token_mixer);
        kv_vs_bf16_pre_gate = Some(validate::max_abs_delta(
            &native_pre_gate_f32,
            &bf16_pre_gate,
        ));
        kv_vs_bf16_gated = Some(validate::max_abs_delta(&native_gated_f32, &bf16_gated));
        kv_vs_bf16_attn_hidden = Some(validate::max_abs_delta(
            &native_token_mixer_f32,
            &bf16_token_mixer_f32,
        ));
        kv_vs_bf16_scores = Some(validate::max_abs_delta(&native_scores_f32, &bf16_scores));
        kv_vs_bf16_hidden = Some(validate::max_abs_delta(&native_hidden_f32, &bf16_hidden));
        kv_vs_bf16_q = Some(validate::max_abs_delta(&native_q_f32, &bf16_q));
        kv_vs_bf16_saved_gate = Some(validate::max_abs_delta(
            &native_saved_gate_f32,
            &bf16_saved_gate,
        ));
        kv_vs_bf16_cache_k = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_cache_k_bf16),
            &decode_bf16_le(&bf16_cache_k_bf16),
        ));
        kv_vs_bf16_cache_v = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_cache_v_bf16),
            &decode_bf16_le(&bf16_cache_v_bf16),
        ));
        let score_cols = seqlen_offset + 1;
        kv_vs_bf16_scores_heads = Some(
            (0..text_config.num_attention_heads)
                .map(|h| {
                    let start = h * score_cols;
                    let end = start + score_cols;
                    validate::max_abs_delta(
                        &native_scores_f32[start..end],
                        &bf16_scores[start..end],
                    )
                })
                .collect::<Vec<_>>(),
        );
        engine.set_kv_fp8_for_trace(true);
        engine.rebuild_prefill_state(prefix_ids, true)?;
    }
    let native_state = engine.state_for_batch(0);
    let native_layer = native_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native layer {trace_layer}"))?;
    let native_vs_component_attn_hidden =
        validate::max_abs_delta(&native_token_mixer_f32, &native_comp_token_mixer_f32);
    let native_vs_host_o_proj =
        validate::max_abs_delta(&native_token_mixer_f32, &native_host_o_proj_f32);
    let native_vs_component_gated =
        validate::max_abs_delta(&native_gated_f32, &native_comp_gated_f32);
    let native_vs_component_saved_gate =
        validate::max_abs_delta(&native_saved_gate_f32, &native_gate_proj_f32);
    let native_vs_component_pre_gate =
        validate::max_abs_delta(&native_pre_gate_f32, &native_comp_pre_gate_f32);
    let head_dim = engine.weights().config.head_dim;
    let num_q_heads = engine.weights().config.num_attention_heads;
    let per_head_pre_gate = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            validate::max_abs_delta(
                &native_pre_gate_f32[start..end],
                &native_comp_pre_gate_f32[start..end],
            )
        })
        .collect::<Vec<_>>();
    let per_head_pre_gate_str = per_head_pre_gate
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let pre_gate_best_match = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            let native_head = &native_pre_gate_f32[start..end];
            let (best_idx, best_delta) = (0..num_q_heads)
                .map(|cand| {
                    let cand_start = cand * head_dim;
                    let cand_end = cand_start + head_dim;
                    (
                        cand,
                        validate::max_abs_delta(
                            native_head,
                            &native_comp_pre_gate_f32[cand_start..cand_end],
                        ),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((h, f32::INFINITY));
            format!("{h}->{best_idx}:{best_delta:.6}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let per_head_q = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            validate::max_abs_delta(&native_q_f32[start..end], &native_q_rope_f32[start..end])
        })
        .collect::<Vec<_>>();
    let per_head_q_str = per_head_q
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let q_best_match = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            let native_head = &native_q_f32[start..end];
            let (best_idx, best_delta) = (0..num_q_heads)
                .map(|cand| {
                    let cand_start = cand * head_dim;
                    let cand_end = cand_start + head_dim;
                    (
                        cand,
                        validate::max_abs_delta(
                            native_head,
                            &native_q_rope_f32[cand_start..cand_end],
                        ),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((h, f32::INFINITY));
            format!("{h}->{best_idx}:{best_delta:.6}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let (score_row_delta, per_head_score_str) = if let (Some(scale_k), Some(k_cache)) = (
        native_layer.kv_scale_k.as_ref(),
        native_layer.kv_cache_k.as_ref(),
    ) {
        let hd = engine.weights().config.head_dim;
        let num_q_heads = engine.weights().config.num_attention_heads;
        let num_kv_heads = engine.weights().config.num_key_value_heads;
        let max_t = k_cache.shape()[2];
        let k_bytes = k_cache
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace native K cache D2H: {e}"))?;
        let k_scales = decode_f32_le(
            &scale_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("trace native K scale D2H: {e}"))?,
        );
        let kv_groups = num_q_heads / num_kv_heads;
        let mut host_scores = Vec::with_capacity(num_q_heads * (seqlen_offset + 1));
        let mut per_head_score = Vec::with_capacity(num_q_heads);
        for qh in 0..num_q_heads {
            let kvh = qh / kv_groups;
            let q_head = &native_q_f32[qh * hd..(qh + 1) * hd];
            let row_start = host_scores.len();
            for t in 0..=seqlen_offset {
                let scale_val = k_scales[kvh * max_t + t];
                let base = (kvh * max_t + t) * hd;
                let mut acc = 0.0f32;
                for d in 0..hd {
                    let k_val =
                        half::bf16::from_f32(fp8_e4m3_to_f32_host(k_bytes[base + d]) * scale_val)
                            .to_f32();
                    acc += q_head[d] * k_val;
                }
                host_scores.push(acc / (hd as f32).sqrt());
            }
            let row_end = host_scores.len();
            per_head_score.push(validate::max_abs_delta(
                &native_scores_f32[row_start..row_end],
                &host_scores[row_start..row_end],
            ));
        }
        (
            validate::max_abs_delta(&native_scores_f32, &host_scores),
            per_head_score
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    } else {
        (0.0, String::new())
    };
    let native_vs_replay_attn_hidden =
        validate::max_abs_delta(&native_token_mixer_f32, &replay_attn_hidden_f32);
    let native_cache_vs_replay_cache_attn_hidden =
        validate::max_abs_delta(&native_comp_token_mixer_f32, &replay_cache_token_mixer_f32);
    let component_vs_replay_attn_hidden =
        validate::max_abs_delta(&native_comp_token_mixer_f32, &replay_attn_hidden_f32);

    if let (Some(scale_k), Some(scale_v), Some(k_cache), Some(_v_cache)) = (
        native_layer.kv_scale_k.as_ref(),
        native_layer.kv_scale_v.as_ref(),
        native_layer.kv_cache_k.as_ref(),
        native_layer.kv_cache_v.as_ref(),
    ) {
        let nkv = engine.weights().config.num_key_value_heads;
        let hd = engine.weights().config.head_dim;
        let max_t = k_cache.shape()[2];

        let src_k = gpu_hal::GpuBuffer::from_host_bytes(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &[nkv, 1, hd],
            &native_component.k_rope,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp K H2D: {e}"))?;
        let src_v = gpu_hal::GpuBuffer::from_host_bytes(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &[nkv, 1, hd],
            &native_component.v_proj,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp V H2D: {e}"))?;
        let mut tmp_k_fp8 =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::U8, &[nkv, max_t, hd])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp K cache alloc: {e}"))?;
        let mut tmp_v_fp8 =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::U8, &[nkv, max_t, hd])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp V cache alloc: {e}"))?;
        let mut tmp_k_scale =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::F32, &[nkv, max_t])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp K scale alloc: {e}"))?;
        let mut tmp_v_scale =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::F32, &[nkv, max_t])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp V scale alloc: {e}"))?;
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &src_k,
            &mut tmp_k_fp8,
            &mut tmp_k_scale,
            nkv,
            1,
            hd,
            max_t,
            seqlen_offset,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 quantize K: {e}"))?;
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &src_v,
            &mut tmp_v_fp8,
            &mut tmp_v_scale,
            nkv,
            1,
            hd,
            max_t,
            seqlen_offset,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 quantize V: {e}"))?;

        let tmp_k_bytes = tmp_k_fp8
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp K D2H: {e}"))?;
        let tmp_v_bytes = tmp_v_fp8
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp V D2H: {e}"))?;
        let tmp_k_scale_bytes = tmp_k_scale
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp K scale D2H: {e}"))?;
        let tmp_v_scale_bytes = tmp_v_scale
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp V scale D2H: {e}"))?;
        let native_k_cache_bytes = k_cache
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native K cache D2H: {e}"))?;
        let native_v_cache_bytes = native_layer
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing native V cache layer {trace_layer}"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native V cache D2H: {e}"))?;
        let native_k_scale_bytes = scale_k
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native K scale D2H: {e}"))?;
        let native_v_scale_bytes = scale_v
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native V scale D2H: {e}"))?;

        let head_span = max_t * hd;
        let kv_groups = num_q_heads / nkv;
        let mut native_k_step = Vec::with_capacity(nkv * hd);
        let mut native_v_step = Vec::with_capacity(nkv * hd);
        let mut quant_k_step = Vec::with_capacity(nkv * hd);
        let mut quant_v_step = Vec::with_capacity(nkv * hd);
        for h in 0..nkv {
            let base = h * head_span + seqlen_offset * hd;
            native_k_step.extend_from_slice(&native_k_cache_bytes[base..base + hd]);
            native_v_step.extend_from_slice(&native_v_cache_bytes[base..base + hd]);
            quant_k_step.extend_from_slice(&tmp_k_bytes[base..base + hd]);
            quant_v_step.extend_from_slice(&tmp_v_bytes[base..base + hd]);
        }
        let native_k_scales = decode_f32_le(&native_k_scale_bytes);
        let native_v_scales = decode_f32_le(&native_v_scale_bytes);
        let quant_k_scales = decode_f32_le(&tmp_k_scale_bytes);
        let quant_v_scales = decode_f32_le(&tmp_v_scale_bytes);
        let mut native_k_scale_step = Vec::with_capacity(nkv);
        let mut native_v_scale_step = Vec::with_capacity(nkv);
        let mut quant_k_scale_step = Vec::with_capacity(nkv);
        let mut quant_v_scale_step = Vec::with_capacity(nkv);
        for h in 0..nkv {
            native_k_scale_step.push(native_k_scales[h * max_t + seqlen_offset]);
            native_v_scale_step.push(native_v_scales[h * max_t + seqlen_offset]);
            quant_k_scale_step.push(quant_k_scales[h * max_t + seqlen_offset]);
            quant_v_scale_step.push(quant_v_scales[h * max_t + seqlen_offset]);
        }
        let cache_vs_quant_k = native_k_step
            .iter()
            .zip(quant_k_step.iter())
            .filter(|(n, q)| n != q)
            .count();
        let cache_vs_quant_v = native_v_step
            .iter()
            .zip(quant_v_step.iter())
            .filter(|(n, q)| n != q)
            .count();
        let scale_vs_quant_k = validate::max_abs_delta(&native_k_scale_step, &quant_k_scale_step);
        let scale_vs_quant_v = validate::max_abs_delta(&native_v_scale_step, &quant_v_scale_step);
        let mut host_pre_gate = vec![0.0f32; num_q_heads * hd];
        for qh in 0..num_q_heads {
            let kvh = qh / kv_groups;
            let row = &native_scores_f32[qh * (seqlen_offset + 1)..(qh + 1) * (seqlen_offset + 1)];
            let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; row.len()];
            for (idx, score) in row.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let scale_val = native_v_scales[kvh * max_t + t];
                    let base = (kvh * max_t + t) * hd + d;
                    let v_val = half::bf16::from_f32(
                        fp8_e4m3_to_f32_host(native_v_cache_bytes[base]) * scale_val,
                    )
                    .to_f32();
                    acc += w * v_val;
                }
                host_pre_gate[qh * hd + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let native_vs_host_pre_gate = validate::max_abs_delta(&native_pre_gate_f32, &host_pre_gate);
        let per_head_host_pre_gate = (0..num_q_heads)
            .map(|h| {
                let start = h * hd;
                let end = start + hd;
                validate::max_abs_delta(
                    &native_pre_gate_f32[start..end],
                    &host_pre_gate[start..end],
                )
            })
            .collect::<Vec<_>>();
        let per_head_host_pre_gate_str = per_head_host_pre_gate
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        let host_gated = host_pre_gate
            .iter()
            .zip(native_saved_gate_f32.iter())
            .map(|(x, g)| x / (1.0 + (-g).exp()))
            .collect::<Vec<_>>();
        let native_vs_host_gated = validate::max_abs_delta(&native_gated_f32, &host_gated);
        let kv_vs_bf16_pre_gate = kv_vs_bf16_pre_gate.unwrap_or(0.0);
        let kv_vs_bf16_gated = kv_vs_bf16_gated.unwrap_or(0.0);
        let kv_vs_bf16_attn_hidden = kv_vs_bf16_attn_hidden.unwrap_or(0.0);
        let kv_vs_bf16_scores = kv_vs_bf16_scores.unwrap_or(0.0);
        let kv_vs_bf16_hidden = kv_vs_bf16_hidden.unwrap_or(0.0);
        let kv_vs_bf16_q = kv_vs_bf16_q.unwrap_or(0.0);
        let kv_vs_bf16_saved_gate = kv_vs_bf16_saved_gate.unwrap_or(0.0);
        let kv_vs_bf16_cache_k = kv_vs_bf16_cache_k.unwrap_or(0.0);
        let kv_vs_bf16_cache_v = kv_vs_bf16_cache_v.unwrap_or(0.0);
        let kv_vs_bf16_scores_heads_str = kv_vs_bf16_scores_heads
            .as_ref()
            .map(|vals| {
                vals.iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        eprintln!(
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] q_best_match=[{q_best_match}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_vs_host_pre_gate={native_vs_host_pre_gate:.6} kv_vs_bf16_hidden={kv_vs_bf16_hidden:.6} kv_vs_bf16_cache_k={kv_vs_bf16_cache_k:.6} kv_vs_bf16_cache_v={kv_vs_bf16_cache_v:.6} kv_vs_bf16_q={kv_vs_bf16_q:.6} kv_vs_bf16_saved_gate={kv_vs_bf16_saved_gate:.6} kv_vs_bf16_scores={kv_vs_bf16_scores:.6} kv_vs_bf16_scores_heads=[{kv_vs_bf16_scores_heads_str}] kv_vs_bf16_pre_gate={kv_vs_bf16_pre_gate:.6} per_head_host_pre_gate=[{per_head_host_pre_gate_str}] native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_host_gated={native_vs_host_gated:.6} kv_vs_bf16_gated={kv_vs_bf16_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} kv_vs_bf16_attn_hidden={kv_vs_bf16_attn_hidden:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] pre_gate_best_match=[{pre_gate_best_match}] cache_vs_quant_k_mismatches={cache_vs_quant_k} cache_vs_quant_v_mismatches={cache_vs_quant_v} cache_vs_quant_k_scale_delta={scale_vs_quant_k:.6} cache_vs_quant_v_scale_delta={scale_vs_quant_v:.6}"
        );
    } else {
        let native_cache = engine.full_attention_cache_step_bytes(trace_layer, 0, seqlen_offset)?;
        let native_cache_k_f32 = decode_bf16_le(&native_cache.0);
        let native_cache_v_f32 = decode_bf16_le(&native_cache.1);
        let cache_vs_component_k = validate::max_abs_delta(&native_cache_k_f32, &native_comp_k_f32);
        let cache_vs_component_v = validate::max_abs_delta(&native_cache_v_f32, &native_comp_v_f32);
        let cache_vs_replay_k = validate::max_abs_delta(&native_cache_k_f32, &replay_comp_k_f32);
        let cache_vs_replay_v = validate::max_abs_delta(&native_cache_v_f32, &replay_comp_v_f32);
        eprintln!(
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] q_best_match=[{q_best_match}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] pre_gate_best_match=[{pre_gate_best_match}] cache_vs_component_k={cache_vs_component_k:.6} cache_vs_component_v={cache_vs_component_v:.6} cache_vs_replay_k={cache_vs_replay_k:.6} cache_vs_replay_v={cache_vs_replay_v:.6}"
        );
    }
    Ok(())
}


pub(crate) fn trace_component_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    native: &ComponentLayerTrace,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component layer trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let attn = replay
        .layer_attn_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay attn trace for layer {trace_layer}"))?;
    let post = replay
        .layer_post_attn_norm_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay post-attn trace for layer {trace_layer}"))?;
    let mlp = replay
        .layer_mlp_out_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay mlp trace for layer {trace_layer}"))?;
    let hidden = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?;
    let attn_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.attn_hidden), &decode_bf16_le(attn));
    let post_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.post_attn_norm),
        &decode_bf16_le(post),
    );
    let mlp_delta = validate::max_abs_delta(&decode_bf16_le(&native.mlp_out), &decode_bf16_le(mlp));
    let hidden_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.layer_hidden),
        &decode_bf16_le(hidden),
    );
    eprintln!(
        "[trace-component-layer] layer={trace_layer} attn_delta={attn_delta:.6} post_norm_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}"
    );
    Ok(())
}

pub(crate) fn trace_component_linear_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    native: &ComponentLinearTrace,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component linear trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        Some(trace_layer),
    )?;
    let replay = replay
        .linear_debug_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing replay linear trace for layer {trace_layer}"))?;
    let qkv_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.qkv), &decode_bf16_le(&replay.qkv));
    let z_delta = validate::max_abs_delta(&decode_bf16_le(&native.z), &decode_bf16_le(&replay.z));
    let packed_native = decode_f32_le(&native.packed);
    let packed_replay = decode_f32_le(&replay.packed);
    let packed_delta = validate::max_abs_delta(&packed_native, &packed_replay);
    let cfg = &engine.weights().config;
    let nv = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let packed_width = 2 * khd + vhd + 2;
    let mut q_delta = 0.0f32;
    let mut k_delta = 0.0f32;
    let mut v_delta = 0.0f32;
    let mut beta_delta = 0.0f32;
    let mut gexp_delta = 0.0f32;
    let v_ref = build_linear_decode_v_reference(engine, trace_layer, &native.qkv)?;
    let mut v_ref_native_delta = 0.0f32;
    let mut v_ref_replay_delta = 0.0f32;
    let mut state_vs_tail_delta = 0.0f32;
    if !replay.qkv_tail.is_empty() {
        let state = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing state for layer {trace_layer}"))?;
        let conv_state = decode_bf16_le(
            &state
                .conv_state
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing conv_state"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("trace conv_state D2H: {e}"))?,
        );
        let qkv_tail = decode_bf16_le(&replay.qkv_tail);
        let qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let state_len = cfg.linear_conv_kernel_dim - 1;
        let mut expected = vec![0.0f32; qkv_dim * state_len];
        for t in 0..state_len {
            for c in 0..qkv_dim {
                expected[c * state_len + t] = qkv_tail[t * qkv_dim + c];
            }
        }
        state_vs_tail_delta = validate::max_abs_delta(&conv_state, &expected);
    }
    for h in 0..nv {
        let base = h * packed_width;
        q_delta = q_delta.max(validate::max_abs_delta(
            &packed_native[base..base + khd],
            &packed_replay[base..base + khd],
        ));
        k_delta = k_delta.max(validate::max_abs_delta(
            &packed_native[base + khd..base + 2 * khd],
            &packed_replay[base + khd..base + 2 * khd],
        ));
        v_delta = v_delta.max(validate::max_abs_delta(
            &packed_native[base + 2 * khd..base + 2 * khd + vhd],
            &packed_replay[base + 2 * khd..base + 2 * khd + vhd],
        ));
        let v_ref_base = h * vhd;
        v_ref_native_delta = v_ref_native_delta.max(validate::max_abs_delta(
            &packed_native[base + 2 * khd..base + 2 * khd + vhd],
            &v_ref[v_ref_base..v_ref_base + vhd],
        ));
        v_ref_replay_delta = v_ref_replay_delta.max(validate::max_abs_delta(
            &packed_replay[base + 2 * khd..base + 2 * khd + vhd],
            &v_ref[v_ref_base..v_ref_base + vhd],
        ));
        beta_delta = beta_delta
            .max((packed_native[base + 2 * khd + vhd] - packed_replay[base + 2 * khd + vhd]).abs());
        gexp_delta = gexp_delta.max(
            (packed_native[base + 2 * khd + vhd + 1] - packed_replay[base + 2 * khd + vhd + 1])
                .abs(),
        );
    }
    let rec_apply_delta = validate::max_abs_delta(
        &decode_f32_le(&native.rec_apply),
        &decode_f32_le(&replay.rec_apply),
    );
    let attn_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.attn), &decode_bf16_le(&replay.attn));
    let gated_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.gated),
        &decode_bf16_le(&replay.gated),
    );
    let proj_out_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.proj_out),
        &decode_bf16_le(&replay.proj_out),
    );
    eprintln!(
        "[trace-component-linear] layer={trace_layer} qkv_delta={qkv_delta:.6} z_delta={z_delta:.6} packed_delta={packed_delta:.6} q_delta={q_delta:.6} k_delta={k_delta:.6} v_delta={v_delta:.6} state_vs_tail_delta={state_vs_tail_delta:.6} v_ref_native_delta={v_ref_native_delta:.6} v_ref_replay_delta={v_ref_replay_delta:.6} beta_delta={beta_delta:.6} gexp_delta={gexp_delta:.6} rec_apply_delta={rec_apply_delta:.6} attn_delta={attn_delta:.6} gated_delta={gated_delta:.6} proj_out_delta={proj_out_delta:.6}"
    );
    Ok(())
}

pub(crate) fn trace_component_linear_state_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    history_token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let native_layer = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native layer {trace_layer}"))?;
    let native_conv = native_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} has no conv_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native conv_state D2H: {e}"))?;
    let native_rec = native_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} has no recurrent_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native recurrent_state D2H: {e}"))?;

    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("component linear state replay init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        history_token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
    )?;
    let replay_layer = replay_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing replay layer {trace_layer}"))?;
    let replay_conv = replay_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} has no conv_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay conv_state D2H: {e}"))?;
    let replay_rec = replay_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} has no recurrent_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay recurrent_state D2H: {e}"))?;

    let conv_delta =
        validate::max_abs_delta(&decode_bf16_le(&native_conv), &decode_bf16_le(&replay_conv));
    let rec_delta =
        validate::max_abs_delta(&decode_f32_le(&native_rec), &decode_f32_le(&replay_rec));
    eprintln!(
        "[trace-component-linear-state] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={rec_delta:.6}"
    );
    Ok(())
}
