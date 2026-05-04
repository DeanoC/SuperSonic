use anyhow::Result;
use qwen35::state::{LayerState, ModelState};

use crate::decode_engine::DecodeEngine;
use crate::prefill_engine;
use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le,
};

pub(crate) fn trace_kv_cache(
    engine: &DecodeEngine,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    kv_fp8: bool,
    batch_size: usize,
    step: usize,
) -> Result<()> {
    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("kv-fp8 trace replay state init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        false,
        None,
    )?;

    for batch_index in 0..batch_size {
        let native_state = engine.state_for_batch(batch_index);
        let mut first_bad = None;
        for (layer_idx, (native_layer, replay_layer)) in native_state
            .layers
            .iter()
            .zip(replay_state.layers.iter())
            .enumerate()
        {
            if !matches!(native_layer.kind, qwen35::weights::LayerKind::Full) {
                continue;
            }
            let diff = compare_kv_layer(native_layer, replay_layer)?;
            if first_bad.is_none()
                && (diff.k_mismatches > 0
                    || diff.v_mismatches > 0
                    || diff.max_scale_k_delta > 0.0
                    || diff.max_scale_v_delta > 0.0)
            {
                first_bad = Some((layer_idx, diff));
            }
        }
        if let Some((layer_idx, diff)) = first_bad {
            eprintln!(
                "[trace-kv-cache] step={step} batch={batch_index} first_bad_layer={layer_idx} filled={} dtype={} k_mismatches={} v_mismatches={} max_k_delta={:.6} max_v_delta={:.6} max_scale_k_delta={:.6} max_scale_v_delta={:.6}{}{}",
                diff.filled,
                diff.dtype,
                diff.k_mismatches,
                diff.v_mismatches,
                diff.max_k_delta,
                diff.max_v_delta,
                diff.max_scale_k_delta,
                diff.max_scale_v_delta,
                diff.first_k_mismatch
                    .map(|(h, t, d, native, replay)| format!(
                        " first_k_mismatch=(h={h},t={t},d={d},native={native},replay={replay})"
                    ))
                    .unwrap_or_default(),
                diff.first_v_mismatch
                    .map(|(h, t, d, native, replay)| format!(
                        " first_v_mismatch=(h={h},t={t},d={d},native={native},replay={replay})"
                    ))
                    .unwrap_or_default(),
            );
        } else {
            eprintln!(
                "[trace-kv-cache] step={step} batch={batch_index} all_full_attention_layers_match"
            );
        }
    }

    Ok(())
}

struct KvFp8LayerDiff {
    filled: usize,
    dtype: &'static str,
    k_mismatches: usize,
    v_mismatches: usize,
    max_k_delta: f32,
    max_v_delta: f32,
    max_scale_k_delta: f32,
    max_scale_v_delta: f32,
    first_k_mismatch: Option<(usize, usize, usize, u8, u8)>,
    first_v_mismatch: Option<(usize, usize, usize, u8, u8)>,
}

fn compare_kv_layer(native: &LayerState, replay: &LayerState) -> Result<KvFp8LayerDiff> {
    let filled = native.kv_filled.min(replay.kv_filled);
    let kv_dtype = native
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_k missing"))?
        .dtype();
    let mut diff = KvFp8LayerDiff {
        filled,
        dtype: if matches!(kv_dtype, gpu_hal::ScalarType::U8) {
            "fp8"
        } else {
            "bf16"
        },
        k_mismatches: 0,
        v_mismatches: 0,
        max_k_delta: 0.0,
        max_v_delta: 0.0,
        max_scale_k_delta: 0.0,
        max_scale_v_delta: 0.0,
        first_k_mismatch: None,
        first_v_mismatch: None,
    };
    if filled == 0 {
        return Ok(diff);
    }

    let native_k = native
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_k missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native kv_cache_k D2H: {e}"))?;
    let replay_k = replay
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay kv_cache_k missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay kv_cache_k D2H: {e}"))?;
    let native_v = native
        .kv_cache_v
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_v missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native kv_cache_v D2H: {e}"))?;
    let replay_v = replay
        .kv_cache_v
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay kv_cache_v missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay kv_cache_v D2H: {e}"))?;

    let native_k_shape = native.kv_cache_k.as_ref().unwrap().shape();
    let replay_k_shape = replay.kv_cache_k.as_ref().unwrap().shape();
    let nkv = native_k_shape[1].min(replay_k_shape[1]);
    let hd = native_k_shape[3].min(replay_k_shape[3]);
    let native_cap = native_k_shape[2];
    let replay_cap = replay_k_shape[2];

    if matches!(kv_dtype, gpu_hal::ScalarType::U8) {
        let native_scale_shape = native.kv_scale_k.as_ref().unwrap().shape();
        let replay_scale_shape = replay.kv_scale_k.as_ref().unwrap().shape();
        for h in 0..nkv {
            for t in 0..filled {
                for d in 0..hd {
                    let native_idx = (h * native_cap + t) * hd + d;
                    let replay_idx = (h * replay_cap + t) * hd + d;
                    let nk = native_k[native_idx];
                    let rk = replay_k[replay_idx];
                    if nk != rk {
                        diff.k_mismatches += 1;
                        if diff.first_k_mismatch.is_none() {
                            diff.first_k_mismatch = Some((h, t, d, nk, rk));
                        }
                    }
                    let nv = native_v[native_idx];
                    let rv = replay_v[replay_idx];
                    if nv != rv {
                        diff.v_mismatches += 1;
                        if diff.first_v_mismatch.is_none() {
                            diff.first_v_mismatch = Some((h, t, d, nv, rv));
                        }
                    }
                }
            }
        }

        let native_scale_k = decode_f32_le(
            &native
                .kv_scale_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("native kv_scale_k missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("native kv_scale_k D2H: {e}"))?,
        );
        let replay_scale_k = decode_f32_le(
            &replay
                .kv_scale_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("replay kv_scale_k missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("replay kv_scale_k D2H: {e}"))?,
        );
        let native_scale_v = decode_f32_le(
            &native
                .kv_scale_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("native kv_scale_v missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("native kv_scale_v D2H: {e}"))?,
        );
        let replay_scale_v = decode_f32_le(
            &replay
                .kv_scale_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("replay kv_scale_v missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("replay kv_scale_v D2H: {e}"))?,
        );

        let native_scale_cap = native_scale_shape[1];
        let replay_scale_cap = replay_scale_shape[1];
        for h in 0..native_scale_shape[0].min(replay_scale_shape[0]) {
            for t in 0..filled {
                let nk = native_scale_k[h * native_scale_cap + t];
                let rk = replay_scale_k[h * replay_scale_cap + t];
                diff.max_scale_k_delta = diff.max_scale_k_delta.max((nk - rk).abs());
                let nv = native_scale_v[h * native_scale_cap + t];
                let rv = replay_scale_v[h * replay_scale_cap + t];
                diff.max_scale_v_delta = diff.max_scale_v_delta.max((nv - rv).abs());
            }
        }
    } else {
        let native_k_f32 = decode_bf16_le(&native_k);
        let replay_k_f32 = decode_bf16_le(&replay_k);
        let native_v_f32 = decode_bf16_le(&native_v);
        let replay_v_f32 = decode_bf16_le(&replay_v);
        for h in 0..nkv {
            for t in 0..filled {
                for d in 0..hd {
                    let native_idx = (h * native_cap + t) * hd + d;
                    let replay_idx = (h * replay_cap + t) * hd + d;
                    let nk = native_k_f32[native_idx];
                    let rk = replay_k_f32[replay_idx];
                    let kd = (nk - rk).abs();
                    diff.max_k_delta = diff.max_k_delta.max(kd);
                    if kd > 0.0 {
                        diff.k_mismatches += 1;
                        if diff.first_k_mismatch.is_none() {
                            diff.first_k_mismatch =
                                Some((h, t, d, native_k[native_idx * 2], replay_k[replay_idx * 2]));
                        }
                    }
                    let nv = native_v_f32[native_idx];
                    let rv = replay_v_f32[replay_idx];
                    let vd = (nv - rv).abs();
                    diff.max_v_delta = diff.max_v_delta.max(vd);
                    if vd > 0.0 {
                        diff.v_mismatches += 1;
                        if diff.first_v_mismatch.is_none() {
                            diff.first_v_mismatch =
                                Some((h, t, d, native_v[native_idx * 2], replay_v[replay_idx * 2]));
                        }
                    }
                }
            }
        }
    }

    Ok(diff)
}
