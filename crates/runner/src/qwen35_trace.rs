use anyhow::Result;
use qwen35::state::ModelState;

use crate::decode_engine::{ComponentLayerTrace, ComponentLinearTrace, DecodeEngine};
use crate::prefill_engine;
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le,
};
use crate::validate;
use crate::qwen35_trace_utils::build_linear_decode_v_reference;

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
