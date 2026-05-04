use anyhow::Result;
use qwen35::state::ModelState;

use crate::decode_engine::DecodeEngine;
use crate::prefill_engine;
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le,
};
use crate::validate;

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
