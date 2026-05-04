use anyhow::Result;

use crate::decode_engine::DecodeEngine;
use crate::tensor_bytes::bf16_bytes_to_f32 as decode_bf16_le;

pub(crate) fn bf16_residual_sum(lhs_bf16: &[u8], rhs_bf16: &[u8]) -> Vec<f32> {
    lhs_bf16
        .chunks_exact(2)
        .zip(rhs_bf16.chunks_exact(2))
        .map(|(l, r)| {
            let sum = half::bf16::from_le_bytes([l[0], l[1]]).to_f32()
                + half::bf16::from_le_bytes([r[0], r[1]]).to_f32();
            half::bf16::from_f32(sum).to_f32()
        })
        .collect()
}

pub(crate) fn fp8_e4m3_to_f32_host(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;
    if byte == 0x7F || byte == 0xFF {
        return 0.0;
    }
    let val = if exp == 0 {
        f32::from(mantissa) / 8.0 * 1.52587890625e-2
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    if sign != 0 {
        -val
    } else {
        val
    }
}

pub(crate) fn build_linear_decode_v_reference(
    engine: &DecodeEngine,
    trace_layer: usize,
    qkv_bytes: &[u8],
) -> Result<Vec<f32>> {
    let cfg = &engine.weights().config;
    let layer = engine
        .weights()
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing weights for layer {trace_layer}"))?
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} is not linear"))?;
    let state = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing state for layer {trace_layer}"))?;

    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let vhd = cfg.linear_value_head_dim;
    let state_len = cfg.linear_conv_kernel_dim - 1;
    let kernel_size = cfg.linear_conv_kernel_dim;
    let key_dim = nk * cfg.linear_key_head_dim;

    let qkv = decode_bf16_le(qkv_bytes);
    let conv_state = decode_bf16_le(
        &state
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("conv_state D2H: {e}"))?,
    );
    let conv_w = decode_bf16_le(
        &layer
            .conv1d_w
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("conv1d_w D2H: {e}"))?,
    );
    let conv_channel = |channel: usize| -> f32 {
        let weight_base = channel * kernel_size;
        let state_base = channel * state_len;
        let mut acc = 0.0f32;
        for tap in 0..kernel_size {
            let x = if tap + 1 == kernel_size {
                qkv[channel]
            } else if tap < state_len {
                conv_state[state_base + tap]
            } else {
                0.0
            };
            acc += x * conv_w[weight_base + tap];
        }
        bf16_round(acc * sigmoid_fast(acc))
    };

    let mut v = vec![0.0f32; nv * vhd];
    for v_head in 0..nv {
        let v_base = key_dim * 2 + v_head * vhd;
        for i in 0..vhd {
            v[v_head * vhd + i] = conv_channel(v_base + i);
        }
    }
    Ok(v)
}

pub(crate) fn sigmoid_fast(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn bf16_round(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}
