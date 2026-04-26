use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use base64::Engine as _;
use gpu_hal::{GpuBuffer, HostBuffer, ScalarType};

use qwen35::config::TextConfig;
use qwen35::desc_builder::{
    build_batch_seq_descs, build_fp8_scale_descs, build_int4_scale_descs, build_kv_fp8_descs,
    build_layer_descs,
};
use qwen35::rotary::RotaryTables;
use qwen35::scratch::{
    PersistentDecodeScratch, PERSISTENT_4B_TIMING_SLOTS_PER_LAYER, PERSISTENT_SYNC_COUNTER_BYTES,
};
use qwen35::state::{kv_fp8_bf16_sidecar_enabled, kv_fp8_bf16_sidecar_window_tokens, ModelState};
use qwen35::weights::{LayerKind, Qwen35Weights};

use crate::oracle::OracleOutput;
use crate::prefill_engine;

const CERTIFIED_KV_MAX_RANKING_BOUNDARY_REPAIR_BLOCKS: usize = 0;

/// Decode a byte slice of little-endian `f32` values into a host `Vec<f32>`.
/// Shared helper used across decode/validate paths.
pub fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_u32_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_bf16_le_host(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn decode_f16_le_host(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn logsumexp(values: &[f32]) -> f32 {
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_v.is_finite() {
        return max_v;
    }
    let sum: f32 = values.iter().map(|v| (*v - max_v).exp()).sum();
    max_v + sum.ln()
}

fn certified_kv_score_delta_from_channel_max(
    query_f32_all: &[f32],
    key_scale_channel_max_host: &[f32],
    q_head: usize,
    kv_head: usize,
    head_dim: usize,
    q_scale: f32,
) -> f32 {
    let query = &query_f32_all[q_head * head_dim..(q_head + 1) * head_dim];
    let scale_base = kv_head * head_dim;
    let mut weighted_scale_sum = 0.0f32;
    for dim in 0..head_dim {
        weighted_scale_sum += query[dim].abs() * key_scale_channel_max_host[scale_base + dim];
    }
    0.5 * q_scale * weighted_scale_sum
}

fn certified_kv_vmax_for_head(
    value_norm_host: &[f32],
    kv_head: usize,
    num_blocks: usize,
    value_error_stride_blocks: usize,
    selected_flags: &[bool],
) -> f32 {
    let base = kv_head * value_error_stride_blocks;
    let mut vmax = 0.0f32;
    for block in 0..num_blocks {
        if selected_flags[block] {
            continue;
        }
        vmax = vmax.max(value_norm_host[base + block]);
    }
    vmax
}

fn certified_kv_key_error_bound(vmax: f32, true_tail_bound: f32, delta_tail: f32) -> f32 {
    2.0 * vmax * true_tail_bound * ((2.0 * delta_tail).exp() - 1.0)
}

fn certified_kv_select_blocks_from_scores(
    block_max: &[f32],
    block_sum: &[f32],
    delta_blocks: Option<&[f32]>,
    tau_cov: f32,
    k_min: usize,
    k_max: usize,
    rung1_threshold: f32,
    rung1_multiplier: f32,
    delta_guard_factor: f32,
) -> Result<(usize, f32, bool, Vec<f32>)> {
    let (selected, tail, rung1, probs, _) = certified_kv_select_block_indices_from_scores(
        block_max,
        block_sum,
        delta_blocks,
        tau_cov,
        k_min,
        k_max,
        rung1_threshold,
        rung1_multiplier,
        delta_guard_factor,
    )?;
    Ok((selected, tail, rung1, probs))
}

fn certified_kv_select_block_indices_from_scores(
    block_max: &[f32],
    block_sum: &[f32],
    delta_blocks: Option<&[f32]>,
    tau_cov: f32,
    k_min: usize,
    k_max: usize,
    rung1_threshold: f32,
    rung1_multiplier: f32,
    delta_guard_factor: f32,
) -> Result<(usize, f32, bool, Vec<f32>, Vec<usize>)> {
    let num_blocks = block_max.len();
    if num_blocks == 0 || block_sum.len() != num_blocks {
        return Ok((0, 0.0, false, Vec::new(), Vec::new()));
    }
    if let Some(delta) = delta_blocks {
        if delta.len() != num_blocks {
            return Err(anyhow::anyhow!(
                "certified KV selector delta block length {} != block count {}",
                delta.len(),
                num_blocks
            ));
        }
    }
    let mut log_mass = Vec::with_capacity(num_blocks);
    for (&m, &s) in block_max.iter().zip(block_sum.iter()) {
        if !m.is_finite() || !s.is_finite() || s <= 0.0 {
            return Err(anyhow::anyhow!(
                "certified KV selector received invalid block score max={} sum={}",
                m,
                s
            ));
        }
        log_mass.push(m + s.ln());
    }
    let global = logsumexp(&log_mass);
    if !global.is_finite() {
        return Err(anyhow::anyhow!(
            "certified KV selector global logsumexp is not finite"
        ));
    }
    let mut probs: Vec<f32> = log_mass.iter().map(|m| (*m - global).exp()).collect();
    for p in &mut probs {
        if !p.is_finite() || *p < 0.0 {
            return Err(anyhow::anyhow!(
                "certified KV selector produced invalid probability {}",
                *p
            ));
        }
    }
    let mut order: Vec<usize> = (0..num_blocks).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = 0usize;
    let mut covered = 0.0f32;
    for &idx in &order {
        covered += probs[idx];
        selected += 1;
        if covered >= tau_cov {
            break;
        }
    }
    let min_k = k_min.min(num_blocks);
    let max_k = k_max.min(num_blocks).max(min_k);
    selected = selected.max(min_k).min(max_k);
    covered = order.iter().take(selected).map(|&idx| probs[idx]).sum();
    let certified_tail = |selected: usize, covered: f32| -> f32 {
        let raw_tail = (1.0 - covered).max(0.0);
        let tail_delta = if let Some(delta) = delta_blocks {
            order
                .iter()
                .skip(selected)
                .map(|&idx| delta[idx])
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        (delta_guard_factor * tail_delta).exp() * raw_tail
    };
    let mut tail = certified_tail(selected, covered);
    let mut rung1 = false;
    if tail > rung1_threshold && selected < num_blocks {
        let expanded = ((selected as f32) * rung1_multiplier).ceil() as usize;
        selected = expanded.max(selected + 1).min(num_blocks);
        covered = order.iter().take(selected).map(|&idx| probs[idx]).sum();
        tail = certified_tail(selected, covered);
        rung1 = true;
    }
    let selected_indices = order.into_iter().take(selected).collect();
    Ok((selected, tail, rung1, probs, selected_indices))
}

fn certified_kv_block_log_mass(max_v: f32, sum_v: f32) -> Option<f32> {
    if max_v.is_finite() && sum_v.is_finite() && sum_v > 0.0 {
        Some(max_v + sum_v.ln())
    } else {
        None
    }
}

fn flush_metal_batch_for_host_boundary(buffer: &GpuBuffer, label: &str) -> Result<()> {
    if buffer.backend() == gpu_hal::Backend::Metal {
        kernel_ffi::prefill_ffi::flush_metal_batch()
            .map_err(|e| anyhow::anyhow!("{label} Metal flush: {e}"))?;
    }
    Ok(())
}

fn metal_profile_flush_layers_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_PROFILE_FLUSH_LAYERS").is_some()
}

fn metal_matmul_lm_head_argmax_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_MATMUL_LM_HEAD_ARGMAX").is_some()
}

fn metal_full_attention_decode_kernel_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FULL_ATTENTION_DECODE_KERNEL").is_some()
}

fn copy_d2d_ordered(
    ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    backend_hint: &GpuBuffer,
    label: &str,
) -> Result<()> {
    if backend_hint.backend() == gpu_hal::Backend::Metal {
        kernel_ffi::prefill_ffi::metal_copy_d2d(src, dst, bytes)
            .map_err(|e| anyhow::anyhow!("{label} Metal blit copy: {e}"))?;
    } else {
        gpu_hal::copy_d2d(ordinal, dst, src, bytes).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    }
    Ok(())
}

fn certified_kv_selected_block_fp16_log_masses_from_tier2(
    query_f32_all: &[f32],
    tier2_key: &[u8],
    q_head: usize,
    kv_head: usize,
    selected_blocks: &[usize],
    cap: usize,
    block_size: usize,
    head_dim: usize,
    q_scale: f32,
) -> Vec<(usize, f32)> {
    let query = &query_f32_all[q_head * head_dim..(q_head + 1) * head_dim];
    let mut out = Vec::with_capacity(selected_blocks.len());
    for &block_id in selected_blocks {
        let mut scores = Vec::with_capacity(block_size);
        for token in 0..block_size {
            let token_base = ((kv_head * cap + block_id * block_size + token) * head_dim) * 2;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                let byte_idx = token_base + d * 2;
                let k = half::bf16::from_le_bytes([tier2_key[byte_idx], tier2_key[byte_idx + 1]])
                    .to_f32();
                dot += query[d] * k;
            }
            scores.push(dot * q_scale);
        }
        let max_v = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_v: f32 = scores.iter().map(|s| (*s - max_v).exp()).sum();
        out.push((block_id, max_v + sum_v.ln()));
    }
    out
}

fn certified_kv_ranking_boundary_violators(
    int8_block_max: &[f32],
    int8_block_sum: &[f32],
    delta_blocks: &[f32],
    fp16_selected_log_masses: &[(usize, f32)],
    selected_blocks: &[usize],
    ranking_r: usize,
) -> Vec<usize> {
    if ranking_r == 0 || selected_blocks.is_empty() {
        return Vec::new();
    }
    let top_r = ranking_r.min(selected_blocks.len());
    let mut fp16_ranked = fp16_selected_log_masses.to_vec();
    fp16_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let promoted_boundary = fp16_ranked
        .get(top_r.saturating_sub(1))
        .map(|(_, log_mass)| *log_mass)
        .unwrap_or(f32::NEG_INFINITY);
    let mut selected_flags = vec![false; int8_block_max.len()];
    for &block in selected_blocks {
        selected_flags[block] = true;
    }
    let mut out = Vec::new();
    for block_id in 0..int8_block_max.len() {
        if selected_flags[block_id] {
            continue;
        }
        if let Some(int8_log_mass) =
            certified_kv_block_log_mass(int8_block_max[block_id], int8_block_sum[block_id])
        {
            if int8_log_mass + delta_blocks[block_id] > promoted_boundary {
                out.push(block_id);
            }
        }
    }
    out.sort_by(|&a, &b| {
        let la = certified_kv_block_log_mass(int8_block_max[a], int8_block_sum[a])
            .unwrap_or(f32::NEG_INFINITY);
        let lb = certified_kv_block_log_mass(int8_block_max[b], int8_block_sum[b])
            .unwrap_or(f32::NEG_INFINITY);
        lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn certified_kv_ranking_mismatch(
    int8_block_max: &[f32],
    int8_block_sum: &[f32],
    delta_blocks: &[f32],
    fp16_selected_log_masses: &[(usize, f32)],
    selected_blocks: &[usize],
    ranking_r: usize,
) -> bool {
    if ranking_r == 0 || selected_blocks.is_empty() {
        return false;
    }
    let top_r = ranking_r.min(selected_blocks.len());
    let mut selected_flags = vec![false; int8_block_max.len()];
    for &block in selected_blocks {
        selected_flags[block] = true;
    }
    if top_r == 1 {
        let fp16_best = fp16_selected_log_masses
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let int8_best = selected_blocks
            .iter()
            .filter_map(|&block_id| {
                certified_kv_block_log_mass(int8_block_max[block_id], int8_block_sum[block_id])
                    .map(|log_mass| (block_id, log_mass))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let Some((fp16_block, promoted_boundary)) = fp16_best else {
            return true;
        };
        let Some((int8_block, _)) = int8_best else {
            return true;
        };
        if fp16_block != int8_block {
            return true;
        }
        for block_id in 0..int8_block_max.len() {
            if selected_flags[block_id] {
                continue;
            }
            if let Some(int8_log_mass) =
                certified_kv_block_log_mass(int8_block_max[block_id], int8_block_sum[block_id])
            {
                if int8_log_mass + delta_blocks[block_id] > promoted_boundary {
                    return true;
                }
            }
        }
        return false;
    }
    let mut fp16_ranked = fp16_selected_log_masses.to_vec();
    fp16_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut int8_selected = Vec::with_capacity(selected_blocks.len());
    for &block_id in selected_blocks {
        if let Some(int8_log_mass) =
            certified_kv_block_log_mass(int8_block_max[block_id], int8_block_sum[block_id])
        {
            int8_selected.push((block_id, int8_log_mass));
        }
    }
    int8_selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let fp16_top: Vec<usize> = fp16_ranked
        .iter()
        .take(top_r)
        .map(|(block_id, _)| *block_id)
        .collect();
    let int8_top: Vec<usize> = int8_selected
        .iter()
        .take(top_r)
        .map(|(block_id, _)| *block_id)
        .collect();
    if fp16_top != int8_top {
        return true;
    }
    let promoted_boundary = fp16_ranked
        .get(top_r.saturating_sub(1))
        .map(|(_, log_mass)| *log_mass)
        .unwrap_or(f32::NEG_INFINITY);
    for block_id in 0..int8_block_max.len() {
        if selected_flags[block_id] {
            continue;
        }
        if let Some(int8_log_mass) =
            certified_kv_block_log_mass(int8_block_max[block_id], int8_block_sum[block_id])
        {
            if int8_log_mass + delta_blocks[block_id] > promoted_boundary {
                return true;
            }
        }
    }
    false
}

fn matmul_proj(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    scale: Option<&GpuBuffer>,
    int8_scale: Option<&GpuBuffer>,
    block_size: usize,
    out: &mut GpuBuffer,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<()> {
    if let (Some(sc), Some(zr)) = (int4_scale, int4_zero) {
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            batch,
            m,
            n,
            k,
            lhs,
            weight,
            sc,
            zr,
            int4_group_size,
            out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
    } else if let Some(sc) = int8_scale {
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int8(
            ordinal, batch, m, n, k, lhs, weight, sc, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"))
    } else {
        match scale {
            Some(s) => kernel_ffi::prefill_ffi::matmul_rhs_transposed_fp8(
                ordinal, batch, m, n, k, lhs, weight, s, block_size, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_fp8: {e}")),
            None => kernel_ffi::prefill_ffi::matmul_rhs_transposed(
                ordinal,
                ScalarType::BF16,
                batch,
                m,
                n,
                k,
                lhs,
                weight,
                out,
            )
            .map_err(|e| anyhow::anyhow!("matmul: {e}")),
        }
    }
}

fn llama31_int8_late_full_mixed_layers() -> usize {
    env::var("SUPERSONIC_LLAMA31_INT8_LATE_FULL_MIXED_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(32)
}

fn llama31_int8_late_full_mixed_component_enabled(component: &str) -> bool {
    env::var("SUPERSONIC_LLAMA31_INT8_LATE_FULL_MIXED_COMPONENTS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|piece| piece.trim().to_ascii_lowercase())
                .any(|piece| piece == component)
        })
        .unwrap_or(matches!(component, "q" | "k" | "v"))
}

fn residual_add(
    ordinal: usize,
    total_elems: usize,
    dst: &mut GpuBuffer,
    src: &GpuBuffer,
) -> Result<()> {
    let lhs: &GpuBuffer = unsafe { &*(dst as *const GpuBuffer) };
    kernel_ffi::prefill_ffi::element_add(ordinal, ScalarType::BF16, total_elems, lhs, src, dst)
        .map_err(|e| anyhow::anyhow!("residual_add failed: {e}"))?;
    Ok(())
}

fn metal_fused_residual_projection_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_RESIDUAL_PROJ").is_some()
}

fn metal_fused_mlp_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_MLP").is_some()
}

fn metal_fused_mlp_gate_up_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_MLP_GATE_UP").is_none()
}

fn metal_fused_mlp_gate_up_swiglu_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_MLP_GATE_UP_SWIGLU").is_none()
}

fn metal_fused_full_projection_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_FULL_PROJ").is_some()
}

fn metal_fused_full_qk_prep_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_FULL_QK_PREP").is_some()
}

fn metal_fused_full_attention_gate_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_FULL_ATTENTION_GATE").is_some()
}

fn metal_fused_linear_out_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_LINEAR_OUT").is_some()
}

fn metal_fused_linear_out_bf16_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_LINEAR_OUT_BF16").is_some()
}

fn metal_fused_linear_decode_apply_inplace_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_LINEAR_DECODE_APPLY_INPLACE").is_some()
}

fn metal_matmul_residual_add_bf16(
    input_dim: usize,
    output_dim: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    residual_out: &mut GpuBuffer,
) -> Result<()> {
    let residual: &GpuBuffer = unsafe { &*(residual_out as *const GpuBuffer) };
    kernel_ffi::prefill_ffi::metal_matmul_rhs_transposed_residual_bf16(
        1,
        1,
        output_dim,
        input_dim,
        input,
        weight,
        residual,
        residual_out,
    )
    .map_err(|e| anyhow::anyhow!("fused residual projection failed: {e}"))?;
    Ok(())
}

fn rms_norm_rows_model(
    config: &TextConfig,
    ordinal: usize,
    rows: usize,
    cols: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    output: &mut GpuBuffer,
    label: &str,
) -> Result<()> {
    let op = if config.rms_norm_add_unit_offset {
        kernel_ffi::prefill_ffi::rms_norm_rows
    } else {
        kernel_ffi::prefill_ffi::rms_norm_rows_plain
    };
    op(
        ordinal,
        ScalarType::BF16,
        rows,
        cols,
        config.rms_norm_eps as f32,
        input,
        weight,
        output,
    )
    .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    Ok(())
}

fn maybe_attn_rms_norm_rows(
    config: &TextConfig,
    ordinal: usize,
    rows: usize,
    cols: usize,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    output: &mut GpuBuffer,
    label: &str,
) -> Result<()> {
    if let Some(weight) = weight {
        let op = if config.rms_norm_add_unit_offset {
            kernel_ffi::prefill_ffi::rms_norm_rows
        } else {
            kernel_ffi::prefill_ffi::rms_norm_rows_plain
        };
        op(
            ordinal,
            ScalarType::BF16,
            rows,
            cols,
            1e-6,
            input,
            weight,
            output,
        )
        .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    } else {
        gpu_hal::copy_d2d(
            ordinal,
            output.as_mut_ptr(),
            input.as_ptr(),
            rows * cols * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("{label} copy-through: {e}"))?;
    }
    Ok(())
}

fn fp8_e4m3_to_f32_host(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;
    if byte == 0x7F || byte == 0xFF {
        return 0.0;
    }
    let val = if exp == 0 {
        mantissa as f32 / 8.0 * 1.52587890625e-2
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign != 0 {
        -val
    } else {
        val
    }
}

fn f32_to_bf16_bytes_host(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

fn is_qwen35_4b_shape(config: &TextConfig) -> bool {
    config.hidden_size == 2560
        && config.intermediate_size == 9216
        && config.num_hidden_layers == 32
        && config.num_attention_heads == 16
        && config.num_key_value_heads == 4
}

fn qwen35_4b_cuda_hero_enabled() -> bool {
    match env::var("SUPERSONIC_DISABLE_QWEN35_4B_CUDA_HERO") {
        Ok(value) => {
            let value = value.trim();
            value.is_empty() || value == "0"
        }
        Err(_) => true,
    }
}

// Keep the early split for the CUDA Qwen3.5-4B path. It remains part of the
// validated long-context lane even after removing the default component
// fallback.
const QWEN35_4B_CUDA_SPLIT_LAYER: usize = 5;
const QWEN35_4B_CUDA_COMPONENT_FALLBACK_TOKENS: usize = 512;

fn qwen35_4b_cuda_long_context_component_fallback_enabled() -> bool {
    match env::var("SUPERSONIC_ENABLE_QWEN35_4B_CUDA_LONG_FALLBACK") {
        Ok(value) => {
            let value = value.trim();
            value.is_empty() || value == "1"
        }
        Err(_) => false,
    }
}

fn qwen35_4b_cuda_dump_layer_timings_topn() -> Option<usize> {
    match env::var("SUPERSONIC_QWEN35_4B_CUDA_DUMP_LAYER_TIMINGS") {
        Ok(value) => {
            let value = value.trim();
            if value.is_empty() {
                Some(8)
            } else {
                value.parse::<usize>().ok().filter(|&topn| topn > 0)
            }
        }
        Err(_) => None,
    }
}

fn qwen35_4b_cuda_split_windows(total_layers: usize) -> Vec<(usize, usize)> {
    fn default_windows(total_layers: usize) -> Vec<(usize, usize)> {
        if total_layers <= QWEN35_4B_CUDA_SPLIT_LAYER {
            return vec![(0, total_layers)];
        }
        vec![
            (0, QWEN35_4B_CUDA_SPLIT_LAYER),
            (
                QWEN35_4B_CUDA_SPLIT_LAYER,
                total_layers - QWEN35_4B_CUDA_SPLIT_LAYER,
            ),
        ]
    }

    let raw = match env::var("SUPERSONIC_QWEN35_4B_CUDA_SPLIT_POINTS") {
        Ok(value) => value,
        Err(_) => return default_windows(total_layers),
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return default_windows(total_layers);
    }

    let mut split_points = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Ok(point) = part.parse::<usize>() else {
            return default_windows(total_layers);
        };
        if point == 0 || point >= total_layers {
            return default_windows(total_layers);
        }
        split_points.push(point);
    }
    if split_points.is_empty() {
        return default_windows(total_layers);
    }

    split_points.sort_unstable();
    split_points.dedup();

    let mut windows = Vec::with_capacity(split_points.len() + 1);
    let mut start = 0usize;
    for point in split_points {
        if point > start {
            windows.push((start, point - start));
            start = point;
        }
    }
    if start < total_layers {
        windows.push((start, total_layers - start));
    }
    if windows.is_empty() {
        default_windows(total_layers)
    } else {
        windows
    }
}

pub struct DecodeEngine {
    weights: Qwen35Weights,
    state: ModelState,
    /// Extra model states for batch items 1..batch_size-1.
    extra_states: Vec<ModelState>,
    scratch: PersistentDecodeScratch,
    rotary: RotaryTables,
    hidden_io: GpuBuffer,
    normed_buf: GpuBuffer,
    logits_buf: GpuBuffer,
    argmax_buf: GpuBuffer,
    lm_head_block_best_vals: GpuBuffer,
    lm_head_block_best_idxs: GpuBuffer,
    target_nll_token: GpuBuffer,
    target_nll_value: GpuBuffer,
    target_nll_accum: GpuBuffer,
    matvec_counter: GpuBuffer,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
    /// FP8 scale descriptors on GPU (None for BF16 weights).
    fp8_scale_device: Option<GpuBuffer>,
    /// INT4 scale descriptors on GPU (None for non-INT4 weights).
    int4_scale_device: Option<GpuBuffer>,
    /// Prefill chunk size (0 = no chunking).
    prefill_chunk_size: usize,
    /// Use FP8 E4M3 KV cache with dynamic per-head scaling.
    kv_fp8: bool,
    /// Optional total decode context reservation. Certified KV drops the dense
    /// GPU KV cache after offload, so it must reserve Tier-1/Tier-2 capacity up
    /// front instead of relying on later dense-cache growth.
    decode_context_limit: Option<usize>,
    /// Batch size (1 = single-sequence, default).
    batch_size: usize,
    /// Cached DFlash tap scratch: `(tap_layers, workspace [num_taps,
    /// hidden_dim] BF16, layer_ids i32-as-u8 buffer)`. `decode_step_
    /// with_taps_kernel` reuses these across calls with the same
    /// tap_layers list. Avoids a per-call GpuBuffer::zeros + upload of
    /// a small i32 vec — ~100ms savings per call at 9B INT4.
    dflash_tap_cache: Option<(Vec<usize>, GpuBuffer, GpuBuffer)>,
    /// Cached workspace for `verify_block_fused_decode` (DFlash M4.3).
    /// The fused verify path runs the persistent 4B megakernel with
    /// `batch_size = B` while the live engine is constructed with
    /// `batch_size = 1`; the cache owns a B-sized workspace + IO buffers
    /// + batch-seq desc table so the per-round allocation cost is paid
    /// only once per fused-verify call chain. Re-allocated if the block
    /// size changes between calls.
    dflash_fused_verify_cache: Option<DFlashFusedVerifyCache>,
    /// Reusable fixed-size scratch for component full-attention decode.
    /// Avoids per-layer-per-step allocation churn on single-sequence
    /// component decode paths.
    component_full_attn_scratch: Option<ComponentFullAttentionScratch>,
    /// Reusable fixed-size scratch for component MLP decode.
    component_mlp_scratch: Option<ComponentMlpScratch>,
    /// Reusable scratch for Metal v2 incremental decode. Lazily allocated on
    /// the first Metal decode step; carries the BF16 inter-chunk linear-attention
    /// buffers across decode steps.
    metal_v2_scratch: Option<prefill_engine::MetalV2DecodeScratch>,
}

/// Per-call workspace for `DecodeEngine::verify_block_fused_decode`.
///
/// The fused verify path needs a B-sized workspace (F32 projection +
/// attention scratch, multi-row hidden_io / normed_buf / logits_buf) and
/// a `BatchSeqDesc` table, sized independently from `DecodeEngine`'s
/// `batch_size = 1` scratch. The cache is populated lazily on first
/// fused-verify call and reused thereafter via the take/put pattern that
/// `decode_step_with_taps_kernel` uses for `dflash_tap_cache`.
struct DFlashFusedVerifyCache {
    /// Block size the cache is sized for. A change in `--dflash-block`
    /// between calls triggers a full re-allocation.
    block_size: usize,
    /// F32 scratch for projections + MLP + attention. Sized to the same
    /// per-item layout as `PersistentDecodeScratch::workspace` at
    /// `batch_size = block_size`.
    workspace: GpuBuffer,
    /// BF16 hidden I/O, shape `[block_size, 1, hidden_size]`.
    hidden_io: GpuBuffer,
    /// BF16 RMSNorm output, shape `[block_size, 1, hidden_size]`.
    normed_buf: GpuBuffer,
    /// BF16 logits output, shape `[block_size, 1, vocab_size]`.
    logits_buf: GpuBuffer,
    /// Device copy of `Vec<BatchSeqDesc>` (one per layer), re-uploaded
    /// each fused-verify call.
    batch_desc_device: GpuBuffer,
}

struct ComponentFullAttentionScratch {
    q_full: GpuBuffer,
    query_buf: GpuBuffer,
    gate_buf: GpuBuffer,
    k_buf: GpuBuffer,
    v_buf: GpuBuffer,
    q_normed: GpuBuffer,
    k_normed: GpuBuffer,
    attn_q: GpuBuffer,
    attn_k_step: GpuBuffer,
    attn_v_step: GpuBuffer,
    attn_out_f32: GpuBuffer,
    attn_out_bf16: GpuBuffer,
    attn_flat: GpuBuffer,
    gated: GpuBuffer,
    proj_out: GpuBuffer,
    certified_score_scratch: Option<GpuBuffer>,
    certified_softmax_stats: Option<GpuBuffer>,
    certified_final_block_mass: Option<GpuBuffer>,
    certified_value_promotion_counters: Option<GpuBuffer>,
    certified_value_promotion_any: Option<GpuBuffer>,
    certified_value_promotion_head_flags: Option<GpuBuffer>,
    certified_value_e_val_by_head: Option<GpuBuffer>,
    certified_ranking_fallback_head_flags: Option<GpuBuffer>,
    certified_block_max: Option<GpuBuffer>,
    certified_block_sum: Option<GpuBuffer>,
    certified_key_scale_norm: Option<GpuBuffer>,
    certified_delta_blocks: Option<GpuBuffer>,
    certified_selected_blocks: Option<GpuBuffer>,
    certified_selected_counts: Option<GpuBuffer>,
    certified_key_gather_index: Option<GpuBuffer>,
    certified_key_cache_counters: Option<GpuBuffer>,
    certified_selector_e_key: Option<GpuBuffer>,
    certified_selector_delta_tail: Option<GpuBuffer>,
    certified_selector_vmax: Option<GpuBuffer>,
    certified_selector_true_tail: Option<GpuBuffer>,
    certified_promote_index: Option<GpuBuffer>,
    certified_promoted_key_bf16: Option<GpuBuffer>,
    certified_value_promote_index: Option<GpuBuffer>,
    certified_promoted_value_bf16: Option<GpuBuffer>,
    certified_score_consistency_flags: Option<GpuBuffer>,
    certified_selected_fp16_log_masses: Option<GpuBuffer>,
    certified_ranking_fallback_heads: Option<GpuBuffer>,
    certified_ranking_fallback_kv_slots: Option<GpuBuffer>,
    certified_ranking_fallback_kv_heads: Option<GpuBuffer>,
    certified_ranking_fallback_score: Option<GpuBuffer>,
    certified_tail_key_compact: Option<GpuBuffer>,
    certified_tail_value_compact: Option<GpuBuffer>,
}

impl ComponentFullAttentionScratch {
    fn alloc(weights: &Qwen35Weights, ordinal: usize) -> Result<Self> {
        let config = &weights.config;
        let head_dim = config.head_dim;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let max_q_proj_dim = weights
            .layers
            .iter()
            .filter_map(|layer| layer.full.as_ref())
            .map(|fw| fw.q_proj_w.shape()[0])
            .max()
            .unwrap_or(q_dim);
        Ok(Self {
            q_full: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, max_q_proj_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn q_full alloc: {e}"))?,
            query_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn query alloc: {e}"))?,
            gate_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn gate alloc: {e}"))?,
            k_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn k alloc: {e}"))?,
            v_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn v alloc: {e}"))?,
            q_normed: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn q_normed alloc: {e}"))?,
            k_normed: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn k_normed alloc: {e}"))?,
            attn_q: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn attn_q alloc: {e}"))?,
            attn_k_step: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn attn_k alloc: {e}"))?,
            attn_v_step: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn attn_v alloc: {e}"))?,
            attn_out_f32: GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
                .map_err(|e| {
                anyhow::anyhow!("component full-attn attn_out alloc: {e}")
            })?,
            attn_out_bf16: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn attn_out bf16 alloc: {e}"))?,
            attn_flat: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn attn_flat alloc: {e}"))?,
            gated: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
                .map_err(|e| anyhow::anyhow!("component full-attn gated alloc: {e}"))?,
            proj_out: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.hidden_size])
                .map_err(|e| anyhow::anyhow!("component full-attn proj_out alloc: {e}"))?,
            certified_score_scratch: None,
            certified_softmax_stats: None,
            certified_final_block_mass: None,
            certified_value_promotion_counters: None,
            certified_value_promotion_any: None,
            certified_value_promotion_head_flags: None,
            certified_value_e_val_by_head: None,
            certified_ranking_fallback_head_flags: None,
            certified_block_max: None,
            certified_block_sum: None,
            certified_key_scale_norm: None,
            certified_delta_blocks: None,
            certified_selected_blocks: None,
            certified_selected_counts: None,
            certified_key_gather_index: None,
            certified_key_cache_counters: None,
            certified_selector_e_key: None,
            certified_selector_delta_tail: None,
            certified_selector_vmax: None,
            certified_selector_true_tail: None,
            certified_promote_index: None,
            certified_promoted_key_bf16: None,
            certified_value_promote_index: None,
            certified_promoted_value_bf16: None,
            certified_score_consistency_flags: None,
            certified_selected_fp16_log_masses: None,
            certified_ranking_fallback_heads: None,
            certified_ranking_fallback_kv_slots: None,
            certified_ranking_fallback_kv_heads: None,
            certified_ranking_fallback_score: None,
            certified_tail_key_compact: None,
            certified_tail_value_compact: None,
        })
    }
}

struct ComponentMlpScratch {
    gate: GpuBuffer,
    up: GpuBuffer,
    mlp: GpuBuffer,
    down: GpuBuffer,
}

impl ComponentMlpScratch {
    fn alloc(config: &TextConfig, ordinal: usize) -> Result<Self> {
        Ok(Self {
            gate: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.intermediate_size])
                .map_err(|e| anyhow::anyhow!("component mlp gate alloc: {e}"))?,
            up: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.intermediate_size])
                .map_err(|e| anyhow::anyhow!("component mlp up alloc: {e}"))?,
            mlp: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.intermediate_size])
                .map_err(|e| anyhow::anyhow!("component mlp act alloc: {e}"))?,
            down: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.hidden_size])
                .map_err(|e| anyhow::anyhow!("component mlp down alloc: {e}"))?,
        })
    }
}

impl DFlashFusedVerifyCache {
    #[allow(clippy::too_many_arguments)]
    fn alloc(
        ordinal: usize,
        block_size: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
    ) -> Result<Self> {
        // Layout matches `PersistentDecodeScratch::new` — see
        // crates/qwen35/src/scratch.rs. Per-item segments:
        //   [hidden] input/output, [hidden] normed, [inter*2] gate+up,
        //   [hidden] down-proj slab × 2, [proj_buf_floats] proj, and
        //   [attn_scratch_floats] attention saved_q/gate/pre_gate/scores.
        let per_item_floats = hidden_dim
            + hidden_dim
            + intermediate_size * 2
            + hidden_dim
            + hidden_dim
            + proj_buf_floats
            + attn_scratch_floats;
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[per_item_floats * block_size])
            .map_err(|e| anyhow::anyhow!("fused verify workspace alloc: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("fused verify hidden_io alloc: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("fused verify normed_buf alloc: {e}"))?;
        let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, vocab_size])
            .map_err(|e| anyhow::anyhow!("fused verify logits_buf alloc: {e}"))?;
        let batch_desc_bytes = num_layers * std::mem::size_of::<kernel_ffi::BatchSeqDesc>();
        let batch_desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[batch_desc_bytes])
            .map_err(|e| anyhow::anyhow!("fused verify batch desc alloc: {e}"))?;
        Ok(Self {
            block_size,
            workspace,
            hidden_io,
            normed_buf,
            logits_buf,
            batch_desc_device,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeSamplingMode {
    HostLogits,
    CudaFastGreedy,
    CudaHeroFusedLmHead,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeStageTimings {
    pub persistent_ms: f64,
    pub rms_norm_ms: f64,
    pub lm_head_ms: f64,
    pub logits_d2h_ms: f64,
    pub host_sampling_ms: f64,
    pub gpu_argmax_ms: f64,
    pub token_d2h_ms: f64,
    pub persistent_full_attn_ms: f64,
    pub persistent_full_attn_proj_ms: f64,
    pub persistent_full_attn_core_ms: f64,
    pub certified_kv_key_quantize_ms: f64,
    pub certified_kv_value_quantize_ms: f64,
    pub certified_kv_score_ms: f64,
    pub certified_kv_selector_ms: f64,
    pub certified_kv_gather_ms: f64,
    pub certified_kv_score_consistency_ms: f64,
    pub certified_kv_rank_log_ms: f64,
    pub certified_kv_ranking_cpu_ms: f64,
    pub certified_kv_attend_ms: f64,
    pub certified_kv_cast_ms: f64,
    pub certified_kv_value_escalation_heads: usize,
    pub certified_kv_ranking_fallback_heads: usize,
    pub certified_kv_dense_fallback_layers: usize,
    pub certified_kv_promoted_key_h2d_bytes: usize,
    pub certified_kv_promoted_value_h2d_bytes: usize,
    pub certified_kv_promoted_key_cache_hits: usize,
    pub certified_kv_promoted_key_cache_misses: usize,
    pub certified_kv_promoted_key_cache_overflows: usize,
    pub certified_kv_promoted_value_cache_hits: usize,
    pub certified_kv_promoted_value_cache_misses: usize,
    pub certified_kv_promoted_value_cache_overflows: usize,
    pub certified_kv_ranking_prefix_cache_hits: usize,
    pub certified_kv_ranking_prefix_cache_misses: usize,
    pub certified_kv_ranking_prefix_h2d_bytes: usize,
    pub certified_kv_ranking_prefix_reuse_bytes: usize,
    pub certified_kv_e_key_max: f32,
    pub certified_kv_e_val_max: f32,
    pub certified_kv_bound_total_max: f32,
    pub certified_kv_delta_tail_max: f32,
    pub certified_kv_vmax_max: f32,
    pub certified_kv_true_tail_bound_max: f32,
    pub certified_kv_uncertified_tail_heads: usize,
    pub certified_kv_score_consistency_violations: usize,
    pub persistent_full_attn_out_ms: f64,
    pub persistent_linear_proj_ms: f64,
    pub persistent_linear_core_ms: f64,
    pub persistent_linear_core_conv_ms: f64,
    pub persistent_linear_core_recurrent_ms: f64,
    pub persistent_linear_core_post_ms: f64,
    pub persistent_linear_out_ms: f64,
    pub persistent_mlp_gate_up_ms: f64,
    pub persistent_mlp_down_ms: f64,
}

impl DecodeStageTimings {
    pub fn add_assign(&mut self, rhs: Self) {
        self.persistent_ms += rhs.persistent_ms;
        self.rms_norm_ms += rhs.rms_norm_ms;
        self.lm_head_ms += rhs.lm_head_ms;
        self.logits_d2h_ms += rhs.logits_d2h_ms;
        self.host_sampling_ms += rhs.host_sampling_ms;
        self.gpu_argmax_ms += rhs.gpu_argmax_ms;
        self.token_d2h_ms += rhs.token_d2h_ms;
        self.persistent_full_attn_ms += rhs.persistent_full_attn_ms;
        self.persistent_full_attn_proj_ms += rhs.persistent_full_attn_proj_ms;
        self.persistent_full_attn_core_ms += rhs.persistent_full_attn_core_ms;
        self.certified_kv_key_quantize_ms += rhs.certified_kv_key_quantize_ms;
        self.certified_kv_value_quantize_ms += rhs.certified_kv_value_quantize_ms;
        self.certified_kv_score_ms += rhs.certified_kv_score_ms;
        self.certified_kv_selector_ms += rhs.certified_kv_selector_ms;
        self.certified_kv_gather_ms += rhs.certified_kv_gather_ms;
        self.certified_kv_score_consistency_ms += rhs.certified_kv_score_consistency_ms;
        self.certified_kv_rank_log_ms += rhs.certified_kv_rank_log_ms;
        self.certified_kv_ranking_cpu_ms += rhs.certified_kv_ranking_cpu_ms;
        self.certified_kv_attend_ms += rhs.certified_kv_attend_ms;
        self.certified_kv_cast_ms += rhs.certified_kv_cast_ms;
        self.certified_kv_value_escalation_heads += rhs.certified_kv_value_escalation_heads;
        self.certified_kv_ranking_fallback_heads += rhs.certified_kv_ranking_fallback_heads;
        self.certified_kv_dense_fallback_layers += rhs.certified_kv_dense_fallback_layers;
        self.certified_kv_promoted_key_h2d_bytes += rhs.certified_kv_promoted_key_h2d_bytes;
        self.certified_kv_promoted_value_h2d_bytes += rhs.certified_kv_promoted_value_h2d_bytes;
        self.certified_kv_promoted_key_cache_hits += rhs.certified_kv_promoted_key_cache_hits;
        self.certified_kv_promoted_key_cache_misses += rhs.certified_kv_promoted_key_cache_misses;
        self.certified_kv_promoted_key_cache_overflows +=
            rhs.certified_kv_promoted_key_cache_overflows;
        self.certified_kv_promoted_value_cache_hits += rhs.certified_kv_promoted_value_cache_hits;
        self.certified_kv_promoted_value_cache_misses +=
            rhs.certified_kv_promoted_value_cache_misses;
        self.certified_kv_promoted_value_cache_overflows +=
            rhs.certified_kv_promoted_value_cache_overflows;
        self.certified_kv_ranking_prefix_cache_hits += rhs.certified_kv_ranking_prefix_cache_hits;
        self.certified_kv_ranking_prefix_cache_misses +=
            rhs.certified_kv_ranking_prefix_cache_misses;
        self.certified_kv_ranking_prefix_h2d_bytes += rhs.certified_kv_ranking_prefix_h2d_bytes;
        self.certified_kv_ranking_prefix_reuse_bytes += rhs.certified_kv_ranking_prefix_reuse_bytes;
        self.certified_kv_e_key_max = self.certified_kv_e_key_max.max(rhs.certified_kv_e_key_max);
        self.certified_kv_e_val_max = self.certified_kv_e_val_max.max(rhs.certified_kv_e_val_max);
        self.certified_kv_bound_total_max = self
            .certified_kv_bound_total_max
            .max(rhs.certified_kv_bound_total_max);
        self.certified_kv_delta_tail_max = self
            .certified_kv_delta_tail_max
            .max(rhs.certified_kv_delta_tail_max);
        self.certified_kv_vmax_max = self.certified_kv_vmax_max.max(rhs.certified_kv_vmax_max);
        self.certified_kv_true_tail_bound_max = self
            .certified_kv_true_tail_bound_max
            .max(rhs.certified_kv_true_tail_bound_max);
        self.certified_kv_uncertified_tail_heads += rhs.certified_kv_uncertified_tail_heads;
        self.certified_kv_score_consistency_violations +=
            rhs.certified_kv_score_consistency_violations;
        self.persistent_full_attn_out_ms += rhs.persistent_full_attn_out_ms;
        self.persistent_linear_proj_ms += rhs.persistent_linear_proj_ms;
        self.persistent_linear_core_ms += rhs.persistent_linear_core_ms;
        self.persistent_linear_core_conv_ms += rhs.persistent_linear_core_conv_ms;
        self.persistent_linear_core_recurrent_ms += rhs.persistent_linear_core_recurrent_ms;
        self.persistent_linear_core_post_ms += rhs.persistent_linear_core_post_ms;
        self.persistent_linear_out_ms += rhs.persistent_linear_out_ms;
        self.persistent_mlp_gate_up_ms += rhs.persistent_mlp_gate_up_ms;
        self.persistent_mlp_down_ms += rhs.persistent_mlp_down_ms;
    }

    pub fn total_ms(&self) -> f64 {
        self.persistent_ms
            + self.rms_norm_ms
            + self.lm_head_ms
            + self.logits_d2h_ms
            + self.host_sampling_ms
            + self.gpu_argmax_ms
            + self.token_d2h_ms
    }
}

pub struct DecodeStepOutput {
    pub logits: Option<Vec<f32>>,
    pub sampled_token: u32,
    pub timings: DecodeStageTimings,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CertifiedKvMemoryStats {
    pub full_attention_layers: usize,
    pub tier1_compressed_vram_bytes: usize,
    pub tier2_host_pinned_bytes: usize,
    pub tail_bf16_vram_bytes: usize,
    pub promoted_key_cache_vram_bytes: usize,
    pub promoted_value_cache_vram_bytes: usize,
    pub ranking_prefix_scratch_vram_bytes: usize,
    pub dense_bf16_kv_vram_bytes: usize,
}

impl CertifiedKvMemoryStats {
    pub fn total_certified_vram_bytes(&self) -> usize {
        self.tier1_compressed_vram_bytes
            + self.tail_bf16_vram_bytes
            + self.promoted_key_cache_vram_bytes
            + self.promoted_value_cache_vram_bytes
            + self.ranking_prefix_scratch_vram_bytes
            + self.dense_bf16_kv_vram_bytes
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CertifiedKvDecodeParams {
    pub block_size: usize,
    pub value_group_size: usize,
    pub bf16_values: bool,
    pub tau_cov: f32,
    pub k_min: usize,
    pub k_max: usize,
    pub v_tol: f32,
    pub value_cache_blocks: usize,
    pub ranking_r: usize,
    pub rung1_threshold: f32,
    pub rung1_multiplier: f32,
    pub key_cache_blocks: usize,
    pub delta_guard_factor: f32,
    pub score_exploration_rate: f32,
    pub require_certified_tail_bound: bool,
    pub eps_guard: f32,
}

impl CertifiedKvDecodeParams {
    fn trace_default(block_size: usize, value_group_size: usize, bf16_values: bool) -> Self {
        Self {
            block_size,
            value_group_size,
            bf16_values,
            tau_cov: 0.995,
            k_min: 2,
            k_max: 128,
            v_tol: 0.05,
            value_cache_blocks: 128,
            ranking_r: 1,
            rung1_threshold: 0.005,
            rung1_multiplier: 2.0,
            key_cache_blocks: 256,
            delta_guard_factor: 3.0,
            score_exploration_rate: 0.01,
            require_certified_tail_bound: true,
            eps_guard: 0.0001,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CertifiedKvShadowStats {
    pub layers: usize,
    pub aligned_tokens: usize,
    pub compressed_vram_bytes: usize,
    pub max_value_error: f32,
    pub quantize_ms: f64,
    pub score_layers: usize,
    pub score_ms: f64,
    pub max_score_ref_delta: f32,
    pub selector_heads: usize,
    pub selector_selected_blocks: usize,
    pub selector_max_tail_mass: f32,
    pub selector_rung1_heads: usize,
    pub value_bound_heads: usize,
    pub value_bound_max: f32,
    pub value_escalation_blocks: usize,
    pub attend_layers: usize,
    pub attend_ms: f64,
    pub attend_max_abs: f32,
    pub attend_ref_max_delta: f32,
    pub attend_bf16_value_ms: f64,
    pub attend_bf16_value_ref_max_delta: f32,
}

pub struct ComponentLayerTrace {
    pub attn_hidden: Vec<u8>,
    pub post_attn_norm: Vec<u8>,
    pub mlp_swiglu: Vec<u8>,
    pub mlp_out: Vec<u8>,
    pub layer_hidden: Vec<u8>,
}

pub struct ComponentMlpTrace {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub swiglu: Vec<u8>,
    pub down: Vec<u8>,
}

pub struct ComponentLinearTrace {
    pub qkv: Vec<u8>,
    pub z: Vec<u8>,
    pub b: Vec<u8>,
    pub a: Vec<u8>,
    pub packed: Vec<u8>,
    pub rec_apply: Vec<u8>,
    pub attn: Vec<u8>,
    pub gated: Vec<u8>,
    pub proj_out: Vec<u8>,
}

pub struct FullAttentionStageTrace {
    pub normed: Vec<u8>,
    pub q_proj: Vec<u8>,
    pub gate_proj: Vec<u8>,
    pub k_proj: Vec<u8>,
    pub v_proj: Vec<u8>,
    pub q_rope: Vec<u8>,
    pub k_rope: Vec<u8>,
}

pub struct FullAttentionLayerOutputTrace {
    pub pre_gate: Vec<u8>,
    pub gated: Vec<u8>,
    pub attn_hidden: Vec<u8>,
    pub key_only_pre_gate: Option<Vec<u8>>,
    pub value_only_pre_gate: Option<Vec<u8>>,
}

pub struct ComponentFullAttentionTrace {
    pub q_proj: Vec<u8>,
    pub gate_proj: Vec<u8>,
    pub k_proj: Vec<u8>,
    pub v_proj: Vec<u8>,
    pub q_rope: Vec<u8>,
    pub k_rope: Vec<u8>,
    pub pre_gate: Vec<u8>,
    pub gated: Vec<u8>,
    pub proj_out: Vec<u8>,
    pub attn_hidden: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
struct Persistent4BLayerTiming {
    full_attn_ms: f64,
    full_attn_core_ms: f64,
    full_attn_hero_prep_ms: f64,
    full_attn_hero_loop_ms: f64,
    full_attn_hero_merge_ms: f64,
    full_attn_hero_gate_ms: f64,
    linear_proj_ms: f64,
    linear_core_ms: f64,
    linear_core_recurrent_ms: f64,
    linear_out_ms: f64,
    mlp_gate_up_ms: f64,
    mlp_down_ms: f64,
}

impl Persistent4BLayerTiming {
    fn add_assign(&mut self, rhs: &Self) {
        self.full_attn_ms += rhs.full_attn_ms;
        self.full_attn_core_ms += rhs.full_attn_core_ms;
        self.full_attn_hero_prep_ms += rhs.full_attn_hero_prep_ms;
        self.full_attn_hero_loop_ms += rhs.full_attn_hero_loop_ms;
        self.full_attn_hero_merge_ms += rhs.full_attn_hero_merge_ms;
        self.full_attn_hero_gate_ms += rhs.full_attn_hero_gate_ms;
        self.linear_proj_ms += rhs.linear_proj_ms;
        self.linear_core_ms += rhs.linear_core_ms;
        self.linear_core_recurrent_ms += rhs.linear_core_recurrent_ms;
        self.linear_out_ms += rhs.linear_out_ms;
        self.mlp_gate_up_ms += rhs.mlp_gate_up_ms;
        self.mlp_down_ms += rhs.mlp_down_ms;
    }

    fn total_ms(&self) -> f64 {
        self.full_attn_ms
            + self.linear_proj_ms
            + self.linear_core_ms
            + self.linear_out_ms
            + self.mlp_gate_up_ms
            + self.mlp_down_ms
    }

    fn mlp_total_ms(&self) -> f64 {
        self.mlp_gate_up_ms + self.mlp_down_ms
    }
}

static QWEN35_4B_CUDA_LAYER_TIMINGS_DUMPED: AtomicBool = AtomicBool::new(false);

fn maybe_dump_qwen35_4b_layer_timings(
    layer_timings: &[Persistent4BLayerTiming],
    topn: usize,
    seqlen_offset: usize,
    batch_size: usize,
) {
    if layer_timings.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..layer_timings.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        layer_timings[rhs]
            .total_ms()
            .partial_cmp(&layer_timings[lhs].total_ms())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let emit = topn.min(order.len());
    eprintln!(
        "[layer-timings] seqlen_offset={} batch_size={} top_layers={} total_layers={}",
        seqlen_offset,
        batch_size,
        emit,
        layer_timings.len(),
    );
    for layer in order.into_iter().take(emit) {
        let timing = &layer_timings[layer];
        eprintln!(
            "[layer-timings] layer={} total_ms={:.3} full_attn_ms={:.3} full_attn_core_ms={:.3} full_attn_hero_prep_ms={:.3} full_attn_hero_loop_ms={:.3} full_attn_hero_merge_ms={:.3} full_attn_hero_gate_ms={:.3} linear_proj_ms={:.3} linear_core_ms={:.3} linear_core_recurrent_ms={:.3} linear_out_ms={:.3} mlp_ms={:.3} mlp_gate_up_ms={:.3} mlp_down_ms={:.3}",
            layer,
            timing.total_ms(),
            timing.full_attn_ms,
            timing.full_attn_core_ms,
            timing.full_attn_hero_prep_ms,
            timing.full_attn_hero_loop_ms,
            timing.full_attn_hero_merge_ms,
            timing.full_attn_hero_gate_ms,
            timing.linear_proj_ms,
            timing.linear_core_ms,
            timing.linear_core_recurrent_ms,
            timing.linear_out_ms,
            timing.mlp_total_ms(),
            timing.mlp_gate_up_ms,
            timing.mlp_down_ms,
        );
    }
}

const PERSISTENT_4B_TIMING_FULL_ATTN: usize = 0;
const PERSISTENT_4B_TIMING_FULL_ATTN_PROJ: usize = 1;
const PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE: usize = 2;
const PERSISTENT_4B_TIMING_FULL_ATTN_HERO_PREP: usize = 3;
const PERSISTENT_4B_TIMING_FULL_ATTN_HERO_LOOP: usize = 5;
const PERSISTENT_4B_TIMING_FULL_ATTN_HERO_MERGE: usize = 6;
const PERSISTENT_4B_TIMING_FULL_ATTN_HERO_GATE: usize = 7;
const PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE: usize = 10;
const PERSISTENT_4B_TIMING_LINEAR_PROJ: usize = 18;
const PERSISTENT_4B_TIMING_LINEAR_CORE_BASE: usize = 19;
const PERSISTENT_4B_TIMING_LINEAR_OUT_BASE: usize = 27;
const PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE: usize = 35;
const PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE: usize = 37;
const PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE: usize = 39;
const PERSISTENT_4B_TIMING_MLP_GATE_UP: usize = 41;
const PERSISTENT_4B_TIMING_MLP_DOWN: usize = 42;

fn persistent_4b_clock_cycles_to_ms(cycles: u64, clock_rate_khz: u32) -> f64 {
    if cycles == 0 || clock_rate_khz == 0 {
        0.0
    } else {
        cycles as f64 / clock_rate_khz as f64
    }
}

/// HIP `clock64()` on gfx1150 does not tick at the rate reported by
/// `hipDeviceProp_t::clockRate`, so the raw `cycles / clock_rate_khz`
/// conversion is off by a large, unknown factor. The cycle counts
/// themselves are self-consistent — per-section ratios match reality —
/// so we calibrate the scale against the wall-clock `persistent_ms`
/// measured on the Rust side.
///
/// `total_cycles` is the sum of per-section cycle counts across all
/// non-overlapping subsections (proj + core + out + MLP), i.e. the
/// cycle budget that should add up to wall-clock. The returned scale
/// converts one cycle into one millisecond such that applying it to
/// each section's cycles redistributes `persistent_ms` proportionally.
///
/// Returns 0.0 when calibration is impossible (no cycles or no wall
/// time captured); the caller then gets all-zero section ms, which is
/// correct — it surfaces the missing data instead of hiding it behind
/// a plausible-looking but wrong number.
fn persistent_4b_wall_clock_scale(persistent_ms: f64, total_cycles: u64) -> f64 {
    if persistent_ms <= 0.0 || total_cycles == 0 {
        0.0
    } else {
        persistent_ms / total_cycles as f64
    }
}

fn persistent_4b_scaled_ms(cycles: u64, ms_per_cycle: f64) -> f64 {
    cycles as f64 * ms_per_cycle
}

/// Calibration source for converting persistent-kernel cycle counts to
/// milliseconds. The CUDA path has a reliable `clockRate` so we use it
/// directly. The HIP path's `clock64()` doesn't tick at the reported rate
/// on gfx1150, so we anchor on the Rust-side wall-clock `persistent_ms`
/// instead and redistribute it proportionally across non-overlapping
/// subsection cycle counts.
#[derive(Clone, Copy)]
enum PersistentTimingCalibration {
    ClockRateKhz(u32),
    WallClockMs(f64),
}

fn decode_persistent_4b_timing_slots(
    sync_bytes: &[u8],
    num_layers: usize,
    batch_size: usize,
    calibration: PersistentTimingCalibration,
    mut layer_timings: Option<&mut [Persistent4BLayerTiming]>,
    layer_offset: usize,
) -> DecodeStageTimings {
    let timing_bytes =
        num_layers * PERSISTENT_4B_TIMING_SLOTS_PER_LAYER * std::mem::size_of::<u64>();
    let start = PERSISTENT_SYNC_COUNTER_BYTES;
    let end = start + timing_bytes;
    if sync_bytes.len() < end {
        return DecodeStageTimings::default();
    }

    // HIP `clock64()` on gfx1150 is not reliably monotonic — `clock64() -
    // start` occasionally wraps when the second sample lands lower than the
    // first (suspected wave-migration / counter-rollover). Once any block in
    // a layer hits that wrap, the slot's `atomicMax` locks in a near-2^64
    // value for the rest of the run. Treat those clearly-bogus reads as zero
    // so they don't poison the calibration. A real cycle count even at a 5
    // GHz tick rate over a full 1-second slot would be ~5e9 cycles; a
    // 60-bit ceiling (~1.15e18) is many orders of magnitude beyond anything
    // physical and unambiguously catches the wraps we've observed without
    // risking false positives on legitimate counts.
    const WRAP_FILTER_CEILING: u64 = 1 << 60;
    let load_slot = |idx: usize| -> u64 {
        let byte_start = start + idx * std::mem::size_of::<u64>();
        let byte_end = byte_start + std::mem::size_of::<u64>();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&sync_bytes[byte_start..byte_end]);
        let v = u64::from_le_bytes(raw);
        if v >= WRAP_FILTER_CEILING {
            0
        } else {
            v
        }
    };

    let mut full_attn_cycles = 0u64;
    let mut full_attn_proj_cycles = 0u64;
    let mut full_attn_core_cycles = 0u64;
    let mut full_attn_out_cycles = 0u64;
    let mut linear_proj_cycles = 0u64;
    let mut linear_core_cycles = 0u64;
    let mut linear_core_conv_cycles = 0u64;
    let mut linear_core_recurrent_cycles = 0u64;
    let mut linear_core_post_cycles = 0u64;
    let mut linear_out_cycles = 0u64;
    let mut mlp_gate_up_cycles = 0u64;
    let mut mlp_down_cycles = 0u64;
    let section_batches = batch_size.min(8);
    let split_batches = batch_size.min(2);
    for layer in 0..num_layers {
        let layer_base = layer * PERSISTENT_4B_TIMING_SLOTS_PER_LAYER;
        let layer_full_attn_cycles = load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN);
        let layer_full_attn_proj_cycles =
            load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_PROJ);
        let layer_full_attn_hero_prep_cycles =
            load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_HERO_PREP);
        let layer_full_attn_hero_loop_cycles =
            load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_HERO_LOOP);
        let layer_full_attn_hero_merge_cycles =
            load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_HERO_MERGE);
        let layer_full_attn_hero_gate_cycles =
            load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_HERO_GATE);
        let layer_linear_proj_cycles = load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_PROJ);
        let layer_mlp_gate_up_cycles = load_slot(layer_base + PERSISTENT_4B_TIMING_MLP_GATE_UP);
        let layer_mlp_down_cycles = load_slot(layer_base + PERSISTENT_4B_TIMING_MLP_DOWN);
        full_attn_cycles += layer_full_attn_cycles;
        full_attn_proj_cycles += layer_full_attn_proj_cycles;
        linear_proj_cycles += layer_linear_proj_cycles;
        mlp_gate_up_cycles += layer_mlp_gate_up_cycles;
        mlp_down_cycles += layer_mlp_down_cycles;
        let mut layer_full_attn_core_cycles = 0u64;
        let mut layer_linear_core_cycles = 0u64;
        let mut layer_linear_out_cycles = 0u64;
        let mut layer_linear_core_recurrent_cycles = 0u64;
        for b in 0..section_batches {
            let full_attn_core =
                load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE + b);
            let linear_core = load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_BASE + b);
            let linear_out = load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_OUT_BASE + b);
            layer_full_attn_core_cycles += full_attn_core;
            layer_linear_core_cycles += linear_core;
            layer_linear_out_cycles += linear_out;
            full_attn_core_cycles += full_attn_core;
            full_attn_out_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE + b);
            linear_core_cycles += linear_core;
            linear_out_cycles += linear_out;
        }
        for b in 0..split_batches {
            linear_core_conv_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE + b);
            let linear_core_recurrent =
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE + b);
            layer_linear_core_recurrent_cycles += linear_core_recurrent;
            linear_core_recurrent_cycles += linear_core_recurrent;
            linear_core_post_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE + b);
        }
        let cvt: Box<dyn Fn(u64) -> f64> = match calibration {
            PersistentTimingCalibration::ClockRateKhz(khz) => {
                Box::new(move |cycles| persistent_4b_clock_cycles_to_ms(cycles, khz))
            }
            PersistentTimingCalibration::WallClockMs(ms) => {
                let total_non_overlapping: u64 = full_attn_proj_cycles
                    .saturating_add(full_attn_core_cycles)
                    .saturating_add(full_attn_out_cycles)
                    .saturating_add(linear_proj_cycles)
                    .saturating_add(linear_core_cycles)
                    .saturating_add(linear_out_cycles)
                    .saturating_add(mlp_gate_up_cycles)
                    .saturating_add(mlp_down_cycles);
                let ms_per_cycle = persistent_4b_wall_clock_scale(ms, total_non_overlapping);
                Box::new(move |cycles| persistent_4b_scaled_ms(cycles, ms_per_cycle))
            }
        };
        if let Some(layer_timings) = layer_timings.as_mut() {
            let layer_timing = Persistent4BLayerTiming {
                full_attn_ms: cvt(layer_full_attn_cycles),
                full_attn_core_ms: cvt(layer_full_attn_core_cycles),
                full_attn_hero_prep_ms: cvt(layer_full_attn_hero_prep_cycles),
                full_attn_hero_loop_ms: cvt(layer_full_attn_hero_loop_cycles),
                full_attn_hero_merge_ms: cvt(layer_full_attn_hero_merge_cycles),
                full_attn_hero_gate_ms: cvt(layer_full_attn_hero_gate_cycles),
                linear_proj_ms: cvt(layer_linear_proj_cycles),
                linear_core_ms: cvt(layer_linear_core_cycles),
                linear_core_recurrent_ms: cvt(layer_linear_core_recurrent_cycles),
                linear_out_ms: cvt(layer_linear_out_cycles),
                mlp_gate_up_ms: cvt(layer_mlp_gate_up_cycles),
                mlp_down_ms: cvt(layer_mlp_down_cycles),
            };
            layer_timings[layer_offset + layer].add_assign(&layer_timing);
        }
    }

    // Pick a cycles→ms conversion. CUDA uses the reported clockRate
    // directly; HIP anchors on wall-clock persistent_ms and redistributes
    // it proportionally across the non-overlapping subsection cycle
    // totals. The umbrella FULL_ATTN slot overlaps with PROJ/CORE/OUT, so
    // exclude it from the calibration denominator to avoid double-count.
    let cvt: Box<dyn Fn(u64) -> f64> = match calibration {
        PersistentTimingCalibration::ClockRateKhz(khz) => {
            Box::new(move |cycles| persistent_4b_clock_cycles_to_ms(cycles, khz))
        }
        PersistentTimingCalibration::WallClockMs(ms) => {
            let total_non_overlapping: u64 = full_attn_proj_cycles
                .saturating_add(full_attn_core_cycles)
                .saturating_add(full_attn_out_cycles)
                .saturating_add(linear_proj_cycles)
                .saturating_add(linear_core_cycles)
                .saturating_add(linear_out_cycles)
                .saturating_add(mlp_gate_up_cycles)
                .saturating_add(mlp_down_cycles);
            let ms_per_cycle = persistent_4b_wall_clock_scale(ms, total_non_overlapping);
            Box::new(move |cycles| persistent_4b_scaled_ms(cycles, ms_per_cycle))
        }
    };

    DecodeStageTimings {
        persistent_full_attn_ms: cvt(full_attn_cycles),
        persistent_full_attn_proj_ms: cvt(full_attn_proj_cycles),
        persistent_full_attn_core_ms: cvt(full_attn_core_cycles),
        persistent_full_attn_out_ms: cvt(full_attn_out_cycles),
        persistent_linear_proj_ms: cvt(linear_proj_cycles),
        persistent_linear_core_ms: cvt(linear_core_cycles),
        persistent_linear_core_conv_ms: cvt(linear_core_conv_cycles),
        persistent_linear_core_recurrent_ms: cvt(linear_core_recurrent_cycles),
        persistent_linear_core_post_ms: cvt(linear_core_post_cycles),
        persistent_linear_out_ms: cvt(linear_out_cycles),
        persistent_mlp_gate_up_ms: cvt(mlp_gate_up_cycles),
        persistent_mlp_down_ms: cvt(mlp_down_cycles),
        ..DecodeStageTimings::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35::scratch::PERSISTENT_4B_TIMING_SLOTS_PER_LAYER;

    #[test]
    fn persistent_4b_timing_ranges_do_not_overlap() {
        let full_attn_core =
            PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE..PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE + 8;
        let full_attn_out =
            PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE..PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE + 8;
        let linear_core =
            PERSISTENT_4B_TIMING_LINEAR_CORE_BASE..PERSISTENT_4B_TIMING_LINEAR_CORE_BASE + 8;
        let linear_out =
            PERSISTENT_4B_TIMING_LINEAR_OUT_BASE..PERSISTENT_4B_TIMING_LINEAR_OUT_BASE + 8;
        let linear_core_conv = PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE + 2;
        let linear_core_recurrent = PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE + 2;
        let linear_core_post = PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE + 2;
        let singleton_slots = [
            PERSISTENT_4B_TIMING_FULL_ATTN,
            PERSISTENT_4B_TIMING_FULL_ATTN_PROJ,
            PERSISTENT_4B_TIMING_LINEAR_PROJ,
            PERSISTENT_4B_TIMING_MLP_GATE_UP,
            PERSISTENT_4B_TIMING_MLP_DOWN,
        ];

        let mut used = [false; PERSISTENT_4B_TIMING_SLOTS_PER_LAYER];
        for slot in singleton_slots {
            assert!(!used[slot], "slot {slot} overlaps");
            used[slot] = true;
        }
        for range in [
            full_attn_core,
            full_attn_out,
            linear_core,
            linear_out,
            linear_core_conv,
            linear_core_recurrent,
            linear_core_post,
        ] {
            for slot in range {
                assert!(slot < PERSISTENT_4B_TIMING_SLOTS_PER_LAYER);
                assert!(!used[slot], "slot {slot} overlaps");
                used[slot] = true;
            }
        }
    }
}

impl DecodeEngine {
    pub fn scratch_debug_ptr(&self) -> usize {
        self.scratch.workspace.as_ptr() as usize
    }

    pub fn certified_kv_shadow_quantize_probe(
        &self,
        block_size: usize,
        value_group_size: usize,
        tau_cov: f32,
        k_min: usize,
        k_max: usize,
        v_tol: f32,
        rung1_threshold: f32,
        rung1_multiplier: f32,
    ) -> Result<CertifiedKvShadowStats> {
        let mut stats = CertifiedKvShadowStats::default();
        for (layer_idx, layer_state) in self.state.layers.iter().enumerate() {
            if !self.weights.config.is_full_attention(layer_idx) || layer_state.kv_filled == 0 {
                continue;
            }
            let cache_k = layer_state
                .kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
            let cache_v = layer_state
                .kv_cache_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
            let aligned =
                kernel_ffi::certified_kv::aligned_tokens(layer_state.kv_filled, block_size);
            if aligned == 0 {
                continue;
            }
            let num_kv_heads = cache_k.shape()[1];
            let head_dim = cache_k.shape()[3];
            let (
                key_i8_shape,
                key_scale_shape,
                value_i4_shape,
                value_meta_shape,
                value_error_shape,
            ) = kernel_ffi::certified_kv::quantized_shapes(
                num_kv_heads,
                layer_state.kv_filled,
                head_dim,
                block_size,
                value_group_size,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV shapes: {e}"))?;

            let mut key_i8 = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &key_i8_shape)
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV key_i8 alloc: {e}"))?;
            let mut key_scale = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape)
                .map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV key_scale alloc: {e}")
            })?;
            let mut key_zero = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} certified KV key_zero alloc: {e}")
                })?;
            let mut value_i4 = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &value_i4_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} certified KV value_i4 alloc: {e}")
                })?;
            let mut value_scale =
                GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape).map_err(
                    |e| anyhow::anyhow!("layer {layer_idx} certified KV value_scale alloc: {e}"),
                )?;
            let mut value_zero = GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} certified KV value_zero alloc: {e}")
                })?;
            let mut value_error =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape).map_err(
                    |e| anyhow::anyhow!("layer {layer_idx} certified KV value_error alloc: {e}"),
                )?;
            let mut value_norm =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape).map_err(
                    |e| anyhow::anyhow!("layer {layer_idx} certified KV value_norm alloc: {e}"),
                )?;

            let start = Instant::now();
            kernel_ffi::certified_kv::quantize_bf16_cache(
                self.ordinal,
                cache_k,
                cache_v,
                layer_state.kv_filled,
                block_size,
                value_group_size,
                &mut key_i8,
                &mut key_scale,
                &mut key_zero,
                &mut value_i4,
                &mut value_scale,
                &mut value_zero,
                &mut value_error,
                &mut value_norm,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV quantize: {e}"))?;
            stats.quantize_ms += start.elapsed().as_secs_f64() * 1000.0;

            let errors = decode_f32_le(&value_error.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV value_error D2H: {e}")
            })?);
            let layer_max = errors.iter().copied().fold(0.0f32, f32::max);
            stats.max_value_error = stats.max_value_error.max(layer_max);
            stats.layers += 1;
            stats.aligned_tokens = stats.aligned_tokens.max(aligned);
            stats.compressed_vram_bytes += key_i8.len_bytes()
                + key_scale.len_bytes()
                + value_i4.len_bytes()
                + value_scale.len_bytes()
                + value_zero.len_bytes()
                + value_error.len_bytes();

            let num_q_heads = self.weights.config.num_attention_heads;
            if num_q_heads % num_kv_heads != 0 {
                return Err(anyhow::anyhow!(
                    "layer {layer_idx} certified KV score probe q_heads={} not divisible by kv_heads={}",
                    num_q_heads,
                    num_kv_heads
                ));
            }
            let gqa_group = num_q_heads / num_kv_heads;
            let num_blocks = aligned / block_size;
            let cache_k_host = cache_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV K cache D2H: {e}"))?;
            let cache_v_host = cache_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV V cache D2H: {e}"))?;
            let mut query_host =
                vec![0u8; num_q_heads * head_dim * ScalarType::BF16.size_in_bytes()];
            let max_t = cache_k.shape()[2];
            let elem_bytes = ScalarType::BF16.size_in_bytes();
            for qh in 0..num_q_heads {
                let kvh = qh / gqa_group;
                let src = (kvh * max_t * head_dim) * elem_bytes;
                let dst = (qh * head_dim) * elem_bytes;
                let bytes = head_dim * elem_bytes;
                query_host[dst..dst + bytes].copy_from_slice(&cache_k_host[src..src + bytes]);
            }
            let query = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_q_heads, head_dim],
                &query_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV score query H2D: {e}"))?;
            let mut block_max =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, num_blocks])
                    .map_err(|e| {
                        anyhow::anyhow!("layer {layer_idx} certified KV block_max alloc: {e}")
                    })?;
            let mut block_sum =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, num_blocks])
                    .map_err(|e| {
                        anyhow::anyhow!("layer {layer_idx} certified KV block_sum alloc: {e}")
                    })?;
            let score_start = Instant::now();
            kernel_ffi::certified_kv::score_blocks_int8(
                self.ordinal,
                &query,
                &key_i8,
                &key_scale,
                &key_zero,
                block_size,
                gqa_group,
                (head_dim as f32).powf(-0.5),
                &mut block_max,
                &mut block_sum,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV score blocks: {e}"))?;
            stats.score_ms += score_start.elapsed().as_secs_f64() * 1000.0;

            let block_max_host = decode_f32_le(&block_max.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV block_max D2H: {e}")
            })?);
            let block_sum_host = decode_f32_le(&block_sum.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV block_sum D2H: {e}")
            })?);
            for (idx, (&m, &s)) in block_max_host.iter().zip(block_sum_host.iter()).enumerate() {
                if !m.is_finite() || !s.is_finite() || s < 1.0 || s > block_size as f32 + 0.001 {
                    return Err(anyhow::anyhow!(
                        "layer {layer_idx} certified KV score output invalid at {}: max={} sum={}",
                        idx,
                        m,
                        s
                    ));
                }
            }
            for qh in 0..num_q_heads {
                let start = qh * num_blocks;
                let end = start + num_blocks;
                let (selected, tail, rung1, probs) = certified_kv_select_blocks_from_scores(
                    &block_max_host[start..end],
                    &block_sum_host[start..end],
                    None,
                    tau_cov,
                    k_min,
                    k_max,
                    rung1_threshold,
                    rung1_multiplier,
                    0.0,
                )
                .map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} q_head {qh} certified KV selector: {e}")
                })?;
                stats.selector_heads += 1;
                stats.selector_selected_blocks += selected;
                stats.selector_max_tail_mass = stats.selector_max_tail_mass.max(tail);
                if rung1 {
                    stats.selector_rung1_heads += 1;
                }
                let kvh = qh / gqa_group;
                let mut value_bound = 0.0f32;
                for (block, &prob) in probs.iter().enumerate() {
                    let eta = errors[kvh * num_blocks + block];
                    let contribution = prob * eta;
                    value_bound += contribution;
                    if contribution > v_tol {
                        stats.value_escalation_blocks += 1;
                    }
                }
                stats.value_bound_heads += 1;
                stats.value_bound_max = stats.value_bound_max.max(value_bound);
            }

            let mut attn_score_scratch =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, aligned]).map_err(
                    |e| anyhow::anyhow!("layer {layer_idx} certified KV attend score alloc: {e}"),
                )?;
            let mut attn_output =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, head_dim]).map_err(
                    |e| anyhow::anyhow!("layer {layer_idx} certified KV attend output alloc: {e}"),
                )?;
            let attend_start = Instant::now();
            kernel_ffi::certified_kv::attend_int8_int4(
                self.ordinal,
                &query,
                &key_i8,
                &key_scale,
                &key_zero,
                &value_i4,
                &value_scale,
                &value_zero,
                block_size,
                value_group_size,
                gqa_group,
                (head_dim as f32).powf(-0.5),
                &mut attn_score_scratch,
                &mut attn_output,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV attend: {e}"))?;
            stats.attend_ms += attend_start.elapsed().as_secs_f64() * 1000.0;
            let attn_host = decode_f32_le(&attn_output.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV attend output D2H: {e}")
            })?);
            for (idx, value) in attn_host.iter().enumerate() {
                if !value.is_finite() {
                    return Err(anyhow::anyhow!(
                        "layer {layer_idx} certified KV attend output invalid at {}: {}",
                        idx,
                        *value
                    ));
                }
                stats.attend_max_abs = stats.attend_max_abs.max(value.abs());
            }
            let mut value_bf16_host = vec![0u8; num_kv_heads * aligned * head_dim * elem_bytes];
            let aligned_row_bytes = aligned * head_dim * elem_bytes;
            for kvh in 0..num_kv_heads {
                let src = kvh * max_t * head_dim * elem_bytes;
                let dst = kvh * aligned_row_bytes;
                value_bf16_host[dst..dst + aligned_row_bytes]
                    .copy_from_slice(&cache_v_host[src..src + aligned_row_bytes]);
            }
            let value_bf16 = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_kv_heads, aligned, head_dim],
                &value_bf16_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV BF16-value H2D: {e}"))?;
            let mut bf16_value_score_scratch = GpuBuffer::zeros(
                self.ordinal,
                ScalarType::F32,
                &[num_q_heads, aligned],
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV BF16-value score alloc: {e}")
            })?;
            let mut bf16_value_output = GpuBuffer::zeros(
                self.ordinal,
                ScalarType::F32,
                &[num_q_heads, head_dim],
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV BF16-value output alloc: {e}")
            })?;
            let bf16_value_start = Instant::now();
            kernel_ffi::certified_kv::attend_int8_bf16_values(
                self.ordinal,
                &query,
                &key_i8,
                &key_scale,
                &key_zero,
                &value_bf16,
                None,
                block_size,
                gqa_group,
                (head_dim as f32).powf(-0.5),
                &mut bf16_value_score_scratch,
                &mut bf16_value_output,
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV BF16-value attend: {e}")
            })?;
            stats.attend_bf16_value_ms += bf16_value_start.elapsed().as_secs_f64() * 1000.0;
            let bf16_value_host =
                decode_f32_le(&bf16_value_output.to_host_bytes().map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} certified KV BF16-value output D2H: {e}")
                })?);
            for (idx, value) in bf16_value_host.iter().enumerate() {
                if !value.is_finite() {
                    return Err(anyhow::anyhow!(
                        "layer {layer_idx} certified KV BF16-value output invalid at {}: {}",
                        idx,
                        *value
                    ));
                }
            }
            let query_f32_all = decode_bf16_le_host(&query_host);
            let mut dense_scores = vec![0.0f32; aligned];
            let q_scale = (head_dim as f32).powf(-0.5);
            for qh in 0..num_q_heads {
                let kvh = qh / gqa_group;
                let query0 = qh * head_dim;
                let kv_base = kvh * max_t * head_dim * elem_bytes;
                for t in 0..aligned {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        let k_offset = kv_base + (t * head_dim + d) * elem_bytes;
                        let k = half::bf16::from_le_bytes([
                            cache_k_host[k_offset],
                            cache_k_host[k_offset + 1],
                        ])
                        .to_f32();
                        acc += query_f32_all[query0 + d] * k;
                    }
                    dense_scores[t] = acc * q_scale;
                }
                let dense_max = dense_scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let dense_denom: f32 = dense_scores.iter().map(|s| (*s - dense_max).exp()).sum();
                for d in 0..head_dim {
                    let mut dense_out = 0.0f32;
                    for (t, score) in dense_scores.iter().enumerate() {
                        let v_offset = kv_base + (t * head_dim + d) * elem_bytes;
                        let v = half::bf16::from_le_bytes([
                            cache_v_host[v_offset],
                            cache_v_host[v_offset + 1],
                        ])
                        .to_f32();
                        dense_out += ((*score - dense_max).exp() / dense_denom) * v;
                    }
                    let attn_idx = qh * head_dim + d;
                    stats.attend_ref_max_delta = stats
                        .attend_ref_max_delta
                        .max((attn_host[attn_idx] - dense_out).abs());
                    stats.attend_bf16_value_ref_max_delta = stats
                        .attend_bf16_value_ref_max_delta
                        .max((bf16_value_host[attn_idx] - dense_out).abs());
                }
            }
            stats.attend_layers += 1;

            let query_f32 = query_f32_all[0..head_dim].to_vec();
            let key_i8_host = key_i8
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} certified KV key_i8 D2H: {e}"))?;
            let key_scale_host = decode_f32_le(&key_scale.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV key_scale D2H: {e}")
            })?);
            let key_zero_host = decode_f32_le(&key_zero.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {layer_idx} certified KV key_zero D2H: {e}")
            })?);
            let mut ref_scores = Vec::with_capacity(block_size);
            for t in 0..block_size {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    let kq = key_i8_host[t * head_dim + d] as i8 as f32;
                    let ks = key_scale_host[d];
                    let kz = key_zero_host[d];
                    acc += query_f32[d] * (kq * ks + kz);
                }
                ref_scores.push(acc * q_scale);
            }
            let ref_max = ref_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ref_sum: f32 = ref_scores.iter().map(|s| (*s - ref_max).exp()).sum();
            let delta = (block_max_host[0] - ref_max)
                .abs()
                .max((block_sum_host[0] - ref_sum).abs());
            stats.max_score_ref_delta = stats.max_score_ref_delta.max(delta);
            stats.score_layers += 1;
        }
        Ok(stats)
    }

    // Rebuild the BF16 sidecar from the current prefix cache when a KV-FP8 state
    // grows after prefill or is cloned for batched decode.
    fn load_kv_shadow_for_state_static(
        config: &TextConfig,
        ordinal: usize,
        state: &mut ModelState,
    ) -> Result<()> {
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        for layer_idx in 0..state.layers.len() {
            if !config.is_full_attention(layer_idx) {
                continue;
            }
            let should_populate = {
                let ls = &state.layers[layer_idx];
                ls.kv_shadow_k.is_some()
                    && ls.kv_shadow_v.is_some()
                    && ls.kv_shadow_start == ls.kv_filled
            };
            if !should_populate {
                continue;
            }

            let (prefix_k_host, prefix_v_host, prefix_len) =
                Self::assemble_full_attention_prefix_cache_bf16_host_static(
                    config, state, layer_idx,
                )?;
            if prefix_len == 0 {
                state.layers[layer_idx].kv_shadow_start = 0;
                continue;
            }

            let ls = &mut state.layers[layer_idx];
            let shadow_k = ls
                .kv_shadow_k
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K shadow"))?;
            let shadow_v = ls
                .kv_shadow_v
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V shadow"))?;
            let cap = shadow_k.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let contig_stride = prefix_len * head_dim * elem_bytes;

            let tmp_k = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_k_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow K H2D: {e}"))?;
            let tmp_v = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_v_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow V H2D: {e}"))?;

            for h in 0..num_kv_heads {
                gpu_hal::copy_d2d(
                    ordinal,
                    shadow_k.offset_ptr(h * cap_stride) as *mut c_void,
                    tmp_k.offset_ptr(h * contig_stride),
                    contig_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow K copy h={h}: {e}"))?;
                gpu_hal::copy_d2d(
                    ordinal,
                    shadow_v.offset_ptr(h * cap_stride) as *mut c_void,
                    tmp_v.offset_ptr(h * contig_stride),
                    contig_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow V copy h={h}: {e}"))?;
            }
            ls.kv_shadow_start = kv_fp8_bf16_sidecar_window_tokens()
                .map(|window| prefix_len.saturating_sub(window))
                .unwrap_or(0);
        }

        Ok(())
    }

    fn assemble_full_attention_prefix_cache_bf16_host_static(
        config: &TextConfig,
        state: &ModelState,
        layer_idx: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        let ls = state
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let prefix_len = ls.kv_filled;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let mut out_k = vec![0u8; num_kv_heads * prefix_len * head_dim * elem_bytes];
        let mut out_v = vec![0u8; num_kv_heads * prefix_len * head_dim * elem_bytes];
        if prefix_len == 0 {
            return Ok((out_k, out_v, prefix_len));
        }

        let cache_k = ls
            .kv_cache_k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
        let cache_v = ls
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
        let cap = cache_k.shape()[2];

        if let (Some(scale_k), Some(scale_v)) = (ls.kv_scale_k.as_ref(), ls.kv_scale_v.as_ref()) {
            let k_bytes = cache_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 K cache D2H: {e}"))?;
            let v_bytes = cache_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 V cache D2H: {e}"))?;
            let k_scales = decode_f32_le(
                &scale_k
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 K scale D2H: {e}"))?,
            );
            let v_scales = decode_f32_le(
                &scale_v
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 V scale D2H: {e}"))?,
            );

            let mut deq_k = Vec::with_capacity(num_kv_heads * prefix_len * head_dim);
            let mut deq_v = Vec::with_capacity(num_kv_heads * prefix_len * head_dim);
            for h in 0..num_kv_heads {
                for t in 0..prefix_len {
                    let scale_k_val = k_scales[h * cap + t];
                    let scale_v_val = v_scales[h * cap + t];
                    let base = (h * cap + t) * head_dim;
                    for d in 0..head_dim {
                        deq_k.push(fp8_e4m3_to_f32_host(k_bytes[base + d]) * scale_k_val);
                        deq_v.push(fp8_e4m3_to_f32_host(v_bytes[base + d]) * scale_v_val);
                    }
                }
            }
            out_k = f32_to_bf16_bytes_host(deq_k);
            out_v = f32_to_bf16_bytes_host(deq_v);
        } else {
            let k_bytes = cache_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} BF16 K cache D2H: {e}"))?;
            let v_bytes = cache_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} BF16 V cache D2H: {e}"))?;
            let src_head_stride = cap * head_dim * elem_bytes;
            let dst_head_stride = prefix_len * head_dim * elem_bytes;
            let copy_bytes = prefix_len * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                out_k[dst..dst + copy_bytes].copy_from_slice(&k_bytes[src..src + copy_bytes]);
                out_v[dst..dst + copy_bytes].copy_from_slice(&v_bytes[src..src + copy_bytes]);
            }
        }

        Ok((out_k, out_v, prefix_len))
    }

    fn assemble_full_attention_prefix_cache_bf16_host_for_state(
        &self,
        state: &ModelState,
        layer_idx: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        Self::assemble_full_attention_prefix_cache_bf16_host_static(
            &self.weights.config,
            state,
            layer_idx,
        )
    }

    pub fn full_attention_prefix_cache_bf16_host(
        &self,
        layer_idx: usize,
        batch_index: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        let state = self.state_for_batch(batch_index);
        self.assemble_full_attention_prefix_cache_bf16_host_for_state(state, layer_idx)
    }

    pub fn trace_full_attention_stages_from_hidden(
        &self,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionStageTrace> {
        let config = self.weights.config.clone();
        let fw = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = fw.q_proj_w.shape()[0];
        let has_attn_gate = match q_proj_dim {
            dim if dim == q_dim => false,
            dim if dim == q_dim * 2 => true,
            dim => {
                return Err(anyhow::anyhow!(
                    "layer {idx}: unsupported full-attention q_proj rows {dim}, expected {q_dim} or {}",
                    q_dim * 2
                ));
            }
        };
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = config.rotary_dim();
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let hidden_buf = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} hidden trace H2D: {e}"))?;
        let mut normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace normed alloc: {e}"))?;
        rms_norm_rows_model(
            &config,
            self.ordinal,
            1,
            hidden_dim,
            &hidden_buf,
            &self.weights.layers[idx].input_norm_w,
            &mut normed,
            &format!("layer {idx} trace input rms_norm"),
        )?;

        let mut q_full = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q_full alloc: {e}"))?;
        let mut query_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace query alloc: {e}"))?;
        let mut gate_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace gate alloc: {e}"))?;
        let mut k_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace k alloc: {e}"))?;
        let mut v_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace v alloc: {e}"))?;
        let mut q_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace q_normed alloc: {e}"))?;
        let mut k_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace k_normed alloc: {e}"))?;

        matmul_proj(
            self.ordinal,
            1,
            1,
            q_proj_dim,
            hidden_dim,
            &normed,
            &fw.q_proj_w,
            fw.q_proj_scale.as_ref(),
            fw.q_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut q_full,
            fw.q_proj_int4_scale.as_ref(),
            fw.q_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        if has_attn_gate {
            kernel_ffi::prefill_ffi::split_qgate(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_q_heads,
                head_dim,
                &q_full,
                &mut query_buf,
                &mut gate_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace split qgate: {e}"))?;
        } else {
            gpu_hal::copy_d2d(
                self.ordinal,
                query_buf.as_ptr() as *mut c_void,
                q_full.as_ptr(),
                q_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q copy: {e}"))?;
        }

        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.k_proj_w,
            fw.k_proj_scale.as_ref(),
            fw.k_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut k_buf,
            fw.k_proj_int4_scale.as_ref(),
            fw.k_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.v_proj_w,
            fw.v_proj_scale.as_ref(),
            fw.v_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut v_buf,
            fw.v_proj_int4_scale.as_ref(),
            fw.v_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;

        maybe_attn_rms_norm_rows(
            &config,
            self.ordinal,
            num_q_heads,
            head_dim,
            &query_buf,
            fw.q_norm_w.as_ref(),
            &mut q_normed,
            &format!("layer {idx} trace q norm"),
        )?;
        gpu_hal::copy_d2d(
            self.ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            q_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q norm copy: {e}"))?;

        maybe_attn_rms_norm_rows(
            &config,
            self.ordinal,
            num_kv_heads,
            head_dim,
            &k_buf,
            fw.k_norm_w.as_ref(),
            &mut k_normed,
            &format!("layer {idx} trace k norm"),
        )?;
        gpu_hal::copy_d2d(
            self.ordinal,
            k_buf.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            kv_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace k norm copy: {e}"))?;

        let q_proj = query_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q pre-rope D2H: {e}"))?;
        let gate_proj = gate_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace gate pre-rope D2H: {e}"))?;
        let k_proj = k_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace k pre-rope D2H: {e}"))?;
        let v_proj = v_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace v D2H: {e}"))?;

        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace k rope: {e}"))?;

        Ok(FullAttentionStageTrace {
            normed: normed
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace normed D2H: {e}"))?,
            q_proj,
            gate_proj,
            k_proj,
            v_proj,
            q_rope: query_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace q rope D2H: {e}"))?,
            k_rope: k_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace k rope D2H: {e}"))?,
        })
    }

    fn trace_full_attention_layer_output_from_hidden_with_state(
        &self,
        state: &ModelState,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
        certified_kv: Option<CertifiedKvDecodeParams>,
    ) -> Result<FullAttentionLayerOutputTrace> {
        let config = self.weights.config.clone();
        let fw = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = fw.q_proj_w.shape()[0];
        let has_attn_gate = match q_proj_dim {
            dim if dim == q_dim => false,
            dim if dim == q_dim * 2 => true,
            dim => {
                return Err(anyhow::anyhow!(
                    "layer {idx}: unsupported full-attention q_proj rows {dim}, expected {q_dim} or {}",
                    q_dim * 2
                ));
            }
        };
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = config.rotary_dim();
        let kv_len = seqlen_offset + 1;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let hidden_in = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer hidden H2D: {e}"))?;
        let mut normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer normed alloc: {e}"))?;
        rms_norm_rows_model(
            &config,
            self.ordinal,
            1,
            hidden_dim,
            &hidden_in,
            &self.weights.layers[idx].input_norm_w,
            &mut normed,
            &format!("layer {idx} trace full layer input rms_norm"),
        )?;

        let mut q_full = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q_full alloc: {e}"))?;
        let mut query_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer query alloc: {e}"))?;
        let mut gate_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate alloc: {e}"))?;
        let mut k_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k alloc: {e}"))?;
        let mut v_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer v alloc: {e}"))?;
        let mut q_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q_normed alloc: {e}"))?;
        let mut k_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k_normed alloc: {e}"))?;
        let mut attn_out_f32 =
            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_out alloc: {e}"))?;
        let mut attn_out_bf16 =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim]).map_err(
                |e| anyhow::anyhow!("layer {idx} trace full layer attn_out bf16 alloc: {e}"),
            )?;
        let mut gated = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gated alloc: {e}"))?;
        let mut proj_out = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer proj_out alloc: {e}"))?;
        let mut hidden_out = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer hidden copy H2D: {e}"))?;
        let mut key_only_pre_gate = None;
        let mut value_only_pre_gate = None;

        matmul_proj(
            self.ordinal,
            1,
            1,
            q_proj_dim,
            hidden_dim,
            &normed,
            &fw.q_proj_w,
            fw.q_proj_scale.as_ref(),
            fw.q_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut q_full,
            fw.q_proj_int4_scale.as_ref(),
            fw.q_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        if has_attn_gate {
            kernel_ffi::prefill_ffi::split_qgate(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_q_heads,
                head_dim,
                &q_full,
                &mut query_buf,
                &mut gate_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer split qgate: {e}"))?;
        } else {
            gpu_hal::copy_d2d(
                self.ordinal,
                query_buf.as_ptr() as *mut c_void,
                q_full.as_ptr(),
                q_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q copy: {e}"))?;
        }
        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.k_proj_w,
            fw.k_proj_scale.as_ref(),
            fw.k_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut k_buf,
            fw.k_proj_int4_scale.as_ref(),
            fw.k_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.v_proj_w,
            fw.v_proj_scale.as_ref(),
            fw.v_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut v_buf,
            fw.v_proj_int4_scale.as_ref(),
            fw.v_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        maybe_attn_rms_norm_rows(
            &config,
            self.ordinal,
            num_q_heads,
            head_dim,
            &query_buf,
            fw.q_norm_w.as_ref(),
            &mut q_normed,
            &format!("layer {idx} trace full layer q norm"),
        )?;
        gpu_hal::copy_d2d(
            self.ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q norm copy: {e}"))?;
        maybe_attn_rms_norm_rows(
            &config,
            self.ordinal,
            num_kv_heads,
            head_dim,
            &k_buf,
            fw.k_norm_w.as_ref(),
            &mut k_normed,
            &format!("layer {idx} trace full layer k norm"),
        )?;
        gpu_hal::copy_d2d(
            self.ordinal,
            k_buf.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            kv_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k norm copy: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k rope: {e}"))?;

        let (prefix_k_host, prefix_v_host, prefix_len) =
            self.assemble_full_attention_prefix_cache_bf16_host_for_state(state, idx)?;
        anyhow::ensure!(
            prefix_len == seqlen_offset,
            "layer {idx} prefix_len {} != seqlen_offset {}",
            prefix_len,
            seqlen_offset
        );
        let q_rope_bytes = query_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q rope D2H: {e}"))?;
        let k_step_bytes = k_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k rope D2H: {e}"))?;
        let v_step_bytes = v_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer v step D2H: {e}"))?;
        let mut full_k_bytes = vec![0u8; num_kv_heads * kv_len * head_dim * elem_bytes];
        let mut full_v_bytes = vec![0u8; num_kv_heads * kv_len * head_dim * elem_bytes];
        let prefix_row_bytes = prefix_len * head_dim * elem_bytes;
        let kv_row_bytes = kv_len * head_dim * elem_bytes;
        let step_row_bytes = head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            let prefix_src = h * prefix_row_bytes;
            let full_dst = h * kv_row_bytes;
            let step_src = h * step_row_bytes;
            full_k_bytes[full_dst..full_dst + prefix_row_bytes]
                .copy_from_slice(&prefix_k_host[prefix_src..prefix_src + prefix_row_bytes]);
            full_v_bytes[full_dst..full_dst + prefix_row_bytes]
                .copy_from_slice(&prefix_v_host[prefix_src..prefix_src + prefix_row_bytes]);
            full_k_bytes[full_dst + prefix_row_bytes..full_dst + kv_row_bytes]
                .copy_from_slice(&k_step_bytes[step_src..step_src + step_row_bytes]);
            full_v_bytes[full_dst + prefix_row_bytes..full_dst + kv_row_bytes]
                .copy_from_slice(&v_step_bytes[step_src..step_src + step_row_bytes]);
        }
        if let Some(certified_kv) = certified_kv {
            let block_size = certified_kv.block_size;
            let value_group_size = certified_kv.value_group_size;
            let bf16_values = certified_kv.bf16_values;
            let aligned = kernel_ffi::certified_kv::aligned_tokens(kv_len, block_size);
            let tail_len = kv_len - aligned;
            anyhow::ensure!(
                aligned > 0,
                "layer {idx} certified KV trace requires at least one complete block, kv_len={kv_len} block_size={block_size}"
            );
            let kv_k_cert = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[1, num_kv_heads, kv_len, head_dim],
                &full_k_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV K H2D: {e}"))?;
            let kv_v_cert = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[1, num_kv_heads, kv_len, head_dim],
                &full_v_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV V H2D: {e}"))?;
            let (
                key_i8_shape,
                key_scale_shape,
                value_i4_shape,
                value_meta_shape,
                value_error_shape,
            ) = kernel_ffi::certified_kv::quantized_shapes(
                num_kv_heads,
                kv_len,
                head_dim,
                block_size,
                value_group_size,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV shapes: {e}"))?;
            let mut key_i8 = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &key_i8_shape)
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV key_i8 alloc: {e}"))?;
            let mut key_scale = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape)
                .map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV key_scale alloc: {e}")
            })?;
            let mut key_zero = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} trace certified KV key_zero alloc: {e}")
                })?;
            let mut value_i4 = GpuBuffer::zeros(self.ordinal, ScalarType::U8, &value_i4_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} trace certified KV value_i4 alloc: {e}")
                })?;
            let mut value_scale =
                GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape).map_err(
                    |e| anyhow::anyhow!("layer {idx} trace certified KV value_scale alloc: {e}"),
                )?;
            let mut value_zero = GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape)
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} trace certified KV value_zero alloc: {e}")
                })?;
            let mut value_error =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape).map_err(
                    |e| anyhow::anyhow!("layer {idx} trace certified KV value_error alloc: {e}"),
                )?;
            let mut value_norm =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape).map_err(
                    |e| anyhow::anyhow!("layer {idx} trace certified KV value_norm alloc: {e}"),
                )?;
            kernel_ffi::certified_kv::quantize_bf16_cache(
                self.ordinal,
                &kv_k_cert,
                &kv_v_cert,
                kv_len,
                block_size,
                value_group_size,
                &mut key_i8,
                &mut key_scale,
                &mut key_zero,
                &mut value_i4,
                &mut value_scale,
                &mut value_zero,
                &mut value_error,
                &mut value_norm,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV quantize: {e}"))?;
            let q_f32 = decode_bf16_le_host(&q_rope_bytes);
            let full_k_f32 = decode_bf16_le_host(&full_k_bytes);
            let full_v_f32 = decode_bf16_le_host(&full_v_bytes);
            let key_i8_host = key_i8
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV key_i8 D2H: {e}"))?;
            let key_scale_host = decode_f32_le(&key_scale.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV key_scale D2H: {e}")
            })?);
            let key_zero_host = decode_f32_le(&key_zero.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV key_zero D2H: {e}")
            })?);
            let value_i4_host = value_i4
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV value_i4 D2H: {e}"))?;
            let value_scale_host =
                decode_f16_le_host(&value_scale.to_host_bytes().map_err(|e| {
                    anyhow::anyhow!("layer {idx} trace certified KV value_scale D2H: {e}")
                })?);
            let value_zero_host = decode_f16_le_host(&value_zero.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV value_zero D2H: {e}")
            })?);
            let host_attention = |quant_keys: bool, quant_values: bool| -> Vec<f32> {
                let mut out = vec![0.0f32; num_q_heads * head_dim];
                let q_scale = 1.0 / (head_dim as f32).sqrt();
                let num_blocks = aligned / block_size;
                let groups = head_dim / value_group_size;
                for qh in 0..num_q_heads {
                    let kvh = qh / (num_q_heads / num_kv_heads);
                    let mut scores = vec![0.0f32; kv_len];
                    for t in 0..kv_len {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            let q = q_f32[qh * head_dim + d];
                            let k = if quant_keys && t < aligned {
                                let block_id = t / block_size;
                                let qk =
                                    key_i8_host[(kvh * aligned + t) * head_dim + d] as i8 as f32;
                                let scale =
                                    key_scale_host[(kvh * num_blocks + block_id) * head_dim + d];
                                let zero =
                                    key_zero_host[(kvh * num_blocks + block_id) * head_dim + d];
                                qk * scale + zero
                            } else {
                                full_k_f32[(kvh * kv_len + t) * head_dim + d]
                            };
                            score += q * k;
                        }
                        scores[t] = score * q_scale;
                    }
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denom: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for t in 0..kv_len {
                            let value = if quant_values && t < aligned {
                                let packed =
                                    value_i4_host[(kvh * aligned + t) * (head_dim / 2) + d / 2];
                                let qv_u8 = if (d & 1) == 0 {
                                    packed & 0x0f
                                } else {
                                    (packed >> 4) & 0x0f
                                };
                                let qv = qv_u8 as f32;
                                let group = d / value_group_size;
                                let meta = (kvh * aligned + t) * groups + group;
                                qv * value_scale_host[meta] + value_zero_host[meta]
                            } else {
                                full_v_f32[(kvh * kv_len + t) * head_dim + d]
                            };
                            let weight = (scores[t] - max_score).exp() / denom;
                            acc += weight * value;
                        }
                        out[qh * head_dim + d] = acc;
                    }
                }
                out
            };
            value_only_pre_gate = Some(f32_to_bf16_bytes_host(host_attention(false, true)));
            let query_cert = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_q_heads, head_dim],
                &q_rope_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV query H2D: {e}"))?;
            let mut score_scratch =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, kv_len]).map_err(
                    |e| anyhow::anyhow!("layer {idx} trace certified KV score alloc: {e}"),
                )?;
            let mut cert_attn_out =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, head_dim]).map_err(
                    |e| anyhow::anyhow!("layer {idx} trace certified KV attn alloc: {e}"),
                )?;
            let full_v_cert = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &full_v_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV BF16 value H2D: {e}"))?;
            let tail_k_for_bf16_values = if tail_len > 0 {
                let mut tail_k_bytes = vec![0u8; num_kv_heads * tail_len * head_dim * elem_bytes];
                let tail_row_bytes = tail_len * head_dim * elem_bytes;
                for h in 0..num_kv_heads {
                    let full_src = h * kv_row_bytes + aligned * head_dim * elem_bytes;
                    let tail_dst = h * tail_row_bytes;
                    tail_k_bytes[tail_dst..tail_dst + tail_row_bytes]
                        .copy_from_slice(&full_k_bytes[full_src..full_src + tail_row_bytes]);
                }
                Some(
                    GpuBuffer::from_host_bytes(
                        self.ordinal,
                        ScalarType::BF16,
                        &[num_kv_heads, tail_len, head_dim],
                        &tail_k_bytes,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} trace certified KV BF16-value tail K H2D: {e}")
                    })?,
                )
            } else {
                None
            };
            let mut key_only_score_scratch = GpuBuffer::zeros(
                self.ordinal,
                ScalarType::F32,
                &[num_q_heads, kv_len],
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV BF16-value score alloc: {e}")
            })?;
            let mut key_only_attn_out =
                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, head_dim]).map_err(
                    |e| {
                        anyhow::anyhow!("layer {idx} trace certified KV BF16-value attn alloc: {e}")
                    },
                )?;
            kernel_ffi::certified_kv::attend_int8_bf16_values(
                self.ordinal,
                &query_cert,
                &key_i8,
                &key_scale,
                &key_zero,
                &full_v_cert,
                tail_k_for_bf16_values.as_ref(),
                block_size,
                num_q_heads / num_kv_heads,
                1.0 / (head_dim as f32).sqrt(),
                &mut key_only_score_scratch,
                &mut key_only_attn_out,
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV BF16-value attention: {e}")
            })?;
            let mut key_only_attn_out_bf16 =
                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} trace certified KV BF16-value cast alloc: {e}")
                    })?;
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                num_q_heads * head_dim,
                &key_only_attn_out,
                &mut key_only_attn_out_bf16,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV BF16-value cast: {e}"))?;
            key_only_pre_gate = Some(key_only_attn_out_bf16.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace certified KV BF16-value D2H: {e}")
            })?);
            if bf16_values {
                cert_attn_out = key_only_attn_out;
            } else if tail_len > 0 {
                let mut tail_k_bytes = vec![0u8; num_kv_heads * tail_len * head_dim * elem_bytes];
                let mut tail_v_bytes = vec![0u8; num_kv_heads * tail_len * head_dim * elem_bytes];
                let tail_row_bytes = tail_len * head_dim * elem_bytes;
                for h in 0..num_kv_heads {
                    let full_src = h * kv_row_bytes + aligned * head_dim * elem_bytes;
                    let tail_dst = h * tail_row_bytes;
                    tail_k_bytes[tail_dst..tail_dst + tail_row_bytes]
                        .copy_from_slice(&full_k_bytes[full_src..full_src + tail_row_bytes]);
                    tail_v_bytes[tail_dst..tail_dst + tail_row_bytes]
                        .copy_from_slice(&full_v_bytes[full_src..full_src + tail_row_bytes]);
                }
                let tail_k = GpuBuffer::from_host_bytes(
                    self.ordinal,
                    ScalarType::BF16,
                    &[num_kv_heads, tail_len, head_dim],
                    &tail_k_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV tail K H2D: {e}"))?;
                let tail_v = GpuBuffer::from_host_bytes(
                    self.ordinal,
                    ScalarType::BF16,
                    &[num_kv_heads, tail_len, head_dim],
                    &tail_v_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV tail V H2D: {e}"))?;
                kernel_ffi::certified_kv::attend_int8_int4_with_bf16_tail(
                    self.ordinal,
                    &query_cert,
                    &key_i8,
                    &key_scale,
                    &key_zero,
                    &value_i4,
                    &value_scale,
                    &value_zero,
                    &tail_k,
                    &tail_v,
                    block_size,
                    value_group_size,
                    num_q_heads / num_kv_heads,
                    1.0 / (head_dim as f32).sqrt(),
                    &mut score_scratch,
                    &mut cert_attn_out,
                )
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} trace certified KV hybrid attention: {e}")
                })?;
            } else {
                kernel_ffi::certified_kv::attend_int8_int4(
                    self.ordinal,
                    &query_cert,
                    &key_i8,
                    &key_scale,
                    &key_zero,
                    &value_i4,
                    &value_scale,
                    &value_zero,
                    block_size,
                    value_group_size,
                    num_q_heads / num_kv_heads,
                    1.0 / (head_dim as f32).sqrt(),
                    &mut score_scratch,
                    &mut cert_attn_out,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV attention: {e}"))?;
            }
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                num_q_heads * head_dim,
                &cert_attn_out,
                &mut attn_out_bf16,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace certified KV attn cast: {e}"))?;
        } else {
            let attn_q = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_q_heads, 1, head_dim],
                &q_rope_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_q H2D: {e}"))?;
            let kv_k_contig = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &full_k_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv_k_contig H2D: {e}"))?;
            let kv_v_contig = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &full_v_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv_v_contig H2D: {e}"))?;

            kernel_ffi::prefill_ffi::full_attention_prefill(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_q_heads,
                num_kv_heads,
                1,
                kv_len,
                head_dim,
                1.0 / (head_dim as f32).sqrt(),
                seqlen_offset,
                &attn_q,
                &kv_k_contig,
                &kv_v_contig,
                &mut attn_out_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attention: {e}"))?;
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                num_q_heads * head_dim,
                &attn_out_f32,
                &mut attn_out_bf16,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn cast: {e}"))?;
        }
        let attn_flat = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, q_dim],
            &attn_out_bf16
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn bf16 D2H: {e}"))?,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_flat H2D: {e}"))?;
        if has_attn_gate {
            kernel_ffi::prefill_ffi::sigmoid_mul(
                self.ordinal,
                ScalarType::BF16,
                q_dim,
                &attn_flat,
                &gate_buf,
                &mut gated,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate apply: {e}"))?;
        } else {
            gpu_hal::copy_d2d(
                self.ordinal,
                gated.as_ptr() as *mut c_void,
                attn_flat.as_ptr(),
                q_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate bypass copy: {e}"))?;
        }
        matmul_proj(
            self.ordinal,
            1,
            1,
            hidden_dim,
            q_dim,
            &gated,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            fw.o_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut proj_out,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        residual_add(self.ordinal, hidden_dim, &mut hidden_out, &proj_out)?;
        Ok(FullAttentionLayerOutputTrace {
            pre_gate: attn_flat
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer pre_gate D2H: {e}"))?,
            gated: gated
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gated D2H: {e}"))?,
            attn_hidden: hidden_out.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace full layer attn_hidden D2H: {e}")
            })?,
            key_only_pre_gate,
            value_only_pre_gate,
        })
    }

    pub fn trace_full_attention_layer_output_from_hidden_current_state(
        &self,
        idx: usize,
        batch_index: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_full_attention_layer_output_from_hidden_with_state(
            self.state_for_batch(batch_index),
            idx,
            hidden_bytes,
            seqlen_offset,
            None,
        )
    }

    pub fn trace_full_attention_layer_output_from_hidden_state(
        &self,
        state: &ModelState,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_full_attention_layer_output_from_hidden_with_state(
            state,
            idx,
            hidden_bytes,
            seqlen_offset,
            None,
        )
    }

    pub fn trace_certified_kv_full_attention_layer_output_from_hidden_state(
        &self,
        state: &ModelState,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
        block_size: usize,
        value_group_size: usize,
        bf16_values: bool,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_full_attention_layer_output_from_hidden_with_state(
            state,
            idx,
            hidden_bytes,
            seqlen_offset,
            Some(CertifiedKvDecodeParams::trace_default(
                block_size,
                value_group_size,
                bf16_values,
            )),
        )
    }

    pub fn trace_certified_kv_full_attention_layer_output_from_hidden_current_state(
        &self,
        idx: usize,
        batch_index: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
        block_size: usize,
        value_group_size: usize,
        bf16_values: bool,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_certified_kv_full_attention_layer_output_from_hidden_state(
            self.state_for_batch(batch_index),
            idx,
            hidden_bytes,
            seqlen_offset,
            block_size,
            value_group_size,
            bf16_values,
        )
    }

    pub fn set_hidden_from_bytes(&mut self, hidden_bytes: &[u8]) -> Result<()> {
        let row = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, self.weights.config.hidden_size],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("set hidden row from bytes: {e}"))?;
        let hidden_io = GpuBuffer::zeros(
            self.ordinal,
            ScalarType::BF16,
            &[self.batch_size, 1, self.weights.config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("alloc hidden_io for trace: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            hidden_io.as_ptr() as *mut c_void,
            row.as_ptr(),
            hidden_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("copy hidden trace row into hidden_io: {e}"))?;
        self.hidden_io = hidden_io;
        Ok(())
    }

    pub fn component_trace_linear_layer_from_current_hidden(
        &mut self,
        idx: usize,
    ) -> Result<(ComponentLinearTrace, Vec<u8>, Vec<u8>, Vec<u8>)> {
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            self.weights.config.hidden_size,
            &self.hidden_io,
            &self.weights.layers[idx].input_norm_w,
            &mut self.normed_buf,
            &format!("layer {idx} component trace input rms_norm"),
        )?;
        let trace = self
            .component_decode_linear_attention_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear trace output"))?;
        let ls = &self.state.layers[idx];
        let conv = ls
            .conv_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("layer {idx}: missing conv state after component trace")
            })?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component conv D2H: {e}"))?;
        let recurrent = ls
            .recurrent_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("layer {idx}: missing recurrent state after component trace")
            })?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component recurrent D2H: {e}"))?;
        let hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component hidden D2H: {e}"))?;
        let row_bytes = self.weights.config.hidden_size * ScalarType::BF16.size_in_bytes();
        Ok((trace, conv, recurrent, hidden[..row_bytes].to_vec()))
    }

    pub fn component_trace_full_layer_from_current_hidden(
        &mut self,
        idx: usize,
    ) -> Result<ComponentLayerTrace> {
        self.component_trace_full_layer_from_current_hidden_with_seqlen(idx, 0)
    }

    pub fn component_trace_full_layer_from_current_hidden_with_seqlen(
        &mut self,
        idx: usize,
        seqlen_offset: usize,
    ) -> Result<ComponentLayerTrace> {
        let row_bytes = self.weights.config.hidden_size * ScalarType::BF16.size_in_bytes();
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            self.weights.config.hidden_size,
            &self.hidden_io,
            &self.weights.layers[idx].input_norm_w,
            &mut self.normed_buf,
            &format!("layer {idx} component full-layer input rms_norm"),
        )?;

        if self.weights.config.is_full_attention(idx) {
            let hidden_in = self
                .hidden_io
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer hidden D2H: {e}"))?;
            let attn_trace = self.trace_full_attention_layer_output_from_hidden_current_state(
                idx,
                0,
                &hidden_in[..row_bytes],
                seqlen_offset,
            )?;
            self.hidden_io = GpuBuffer::from_host_bytes(
                self.ordinal,
                ScalarType::BF16,
                &[1, self.weights.config.hidden_size],
                &attn_trace.attn_hidden,
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {idx} component full-layer attn hidden H2D: {e}")
            })?;
        } else {
            self.component_decode_linear_attention_layer(idx, false)?;
        }
        let attn_hidden = self.hidden_io.to_host_bytes().map_err(|e| {
            anyhow::anyhow!("layer {idx} component full-layer attn hidden D2H: {e}")
        })?[..row_bytes]
            .to_vec();

        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            self.weights.config.hidden_size,
            &self.hidden_io,
            &self.weights.layers[idx].post_attn_norm_w,
            &mut self.normed_buf,
            &format!("layer {idx} component full-layer post rms_norm"),
        )?;
        let post_attn_norm =
            self.normed_buf.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} component full-layer post norm D2H: {e}")
            })?[..row_bytes]
                .to_vec();

        let mlp_trace = self
            .component_decode_mlp_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx} component full-layer missing mlp trace"))?;
        let layer_hidden = self.hidden_io.to_host_bytes().map_err(|e| {
            anyhow::anyhow!("layer {idx} component full-layer final hidden D2H: {e}")
        })?[..row_bytes]
            .to_vec();

        Ok(ComponentLayerTrace {
            attn_hidden,
            post_attn_norm,
            mlp_swiglu: mlp_trace.swiglu,
            mlp_out: mlp_trace.down,
            layer_hidden,
        })
    }

    pub fn component_trace_full_attention_from_current_hidden_with_seqlen(
        &mut self,
        idx: usize,
        seqlen_offset: usize,
    ) -> Result<ComponentFullAttentionTrace> {
        anyhow::ensure!(
            self.weights.config.is_full_attention(idx),
            "layer {idx} is not full attention"
        );
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            self.weights.config.hidden_size,
            &self.hidden_io,
            &self.weights.layers[idx].input_norm_w,
            &mut self.normed_buf,
            &format!("layer {idx} component full-attn input rms_norm"),
        )?;
        self.component_decode_full_attention_layer(idx, seqlen_offset, true, None, None)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx} missing full-attention trace"))
    }

    pub fn full_attention_cache_step_bytes(
        &self,
        layer_idx: usize,
        batch_index: usize,
        seq_pos: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let config = self.weights.config.clone();
        let ls = self
            .state_for_batch(batch_index)
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let cache_k = ls
            .kv_cache_k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
        let cache_v = ls
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let step_k = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k alloc: {e}"))?;
        let step_v = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v alloc: {e}"))?;

        let cap = cache_k.shape()[2];
        let cap_stride = cap * head_dim * elem_bytes;
        let src_stride = head_dim * elem_bytes;
        let dst_stride = head_dim * elem_bytes;
        let src_offset = seq_pos * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                self.ordinal,
                step_k.offset_ptr(h * dst_stride) as *mut c_void,
                cache_k.offset_ptr(h * cap_stride + src_offset),
                src_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k copy h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                self.ordinal,
                step_v.offset_ptr(h * dst_stride) as *mut c_void,
                cache_v.offset_ptr(h * cap_stride + src_offset),
                src_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v copy h={h}: {e}"))?;
        }

        Ok((
            step_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k D2H: {e}"))?,
            step_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v D2H: {e}"))?,
        ))
    }

    fn component_decode_step_4b_impl(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_input_layer: Option<usize>,
        trace_layer: Option<usize>,
        trace_linear_layer: Option<usize>,
        cuda_greedy: bool,
        certified_kv_decode: Option<CertifiedKvDecodeParams>,
        target_nll_token: Option<u32>,
        accumulate_target_nll: bool,
        mut timings: Option<&mut DecodeStageTimings>,
    ) -> Result<(
        Option<Vec<f32>>,
        Option<u32>,
        Option<f32>,
        Option<Vec<u8>>,
        Option<ComponentLayerTrace>,
        Option<ComponentLinearTrace>,
    )> {
        let hidden_dim = self.weights.config.hidden_size;
        let vocab_size = self.weights.config.vocab_size;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let collect_timings = timings.is_some();

        let row_bytes = hidden_dim * elem_bytes;
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        let layer_count = self.state.layers.len();
        let mut traced_hidden = None;
        let mut traced_layer = None;
        let mut traced_linear = None;
        for i in 0..layer_count {
            if trace_input_layer == Some(i) {
                traced_hidden = Some(
                    self.hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} hidden trace D2H: {e}"))?,
                );
            }
            let input_norm_start = Instant::now();
            rms_norm_rows_model(
                &self.weights.config,
                self.ordinal,
                1,
                hidden_dim,
                &self.hidden_io,
                &self.weights.layers[i].input_norm_w,
                &mut self.normed_buf,
                &format!("layer {i} input rms_norm"),
            )?;
            self.sync_stage_if_requested(collect_timings, &format!("layer {i} input rms_norm"))?;
            if let Some(t) = timings.as_mut() {
                t.rms_norm_ms += input_norm_start.elapsed().as_secs_f64() * 1000.0;
            }

            if self.weights.config.is_full_attention(i) {
                let full_attn_start = Instant::now();
                let _ = self.component_decode_full_attention_layer(
                    i,
                    seqlen_offset,
                    false,
                    certified_kv_decode,
                    timings.as_deref_mut(),
                )?;
                self.sync_stage_if_requested(
                    collect_timings,
                    &format!("layer {i} full attention"),
                )?;
                let elapsed = full_attn_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(t) = timings.as_mut() {
                    t.persistent_ms += elapsed;
                    t.persistent_full_attn_ms += elapsed;
                }
            } else {
                let linear_start = Instant::now();
                if let Some(trace) =
                    self.component_decode_linear_attention_layer(i, trace_linear_layer == Some(i))?
                {
                    traced_linear = Some(trace);
                }
                self.sync_stage_if_requested(
                    collect_timings,
                    &format!("layer {i} linear attention"),
                )?;
                let elapsed = linear_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(t) = timings.as_mut() {
                    t.persistent_ms += elapsed;
                    t.persistent_linear_core_ms += elapsed;
                }
            }

            let mut trace_attn_hidden = None;
            let mut trace_post_attn_norm = None;
            if trace_layer == Some(i) {
                trace_attn_hidden = Some(
                    self.hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} attn hidden trace D2H: {e}"))?,
                );
            }

            let post_attn_norm_start = Instant::now();
            rms_norm_rows_model(
                &self.weights.config,
                self.ordinal,
                1,
                hidden_dim,
                &self.hidden_io,
                &self.weights.layers[i].post_attn_norm_w,
                &mut self.normed_buf,
                &format!("layer {i} post-attn rms_norm"),
            )?;
            self.sync_stage_if_requested(
                collect_timings,
                &format!("layer {i} post-attn rms_norm"),
            )?;
            if let Some(t) = timings.as_mut() {
                t.rms_norm_ms += post_attn_norm_start.elapsed().as_secs_f64() * 1000.0;
            }

            if trace_layer == Some(i) {
                trace_post_attn_norm = Some(
                    self.normed_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} post-attn norm trace D2H: {e}"))?,
                );
            }

            let mlp_start = Instant::now();
            let maybe_mlp = self.component_decode_mlp_layer(i, trace_layer == Some(i))?;
            self.sync_stage_if_requested(collect_timings, &format!("layer {i} mlp"))?;
            let mlp_elapsed = mlp_start.elapsed().as_secs_f64() * 1000.0;
            if let Some(t) = timings.as_mut() {
                t.persistent_ms += mlp_elapsed;
                t.persistent_mlp_down_ms += mlp_elapsed;
            }
            if trace_layer == Some(i) {
                let mlp_trace =
                    maybe_mlp.ok_or_else(|| anyhow::anyhow!("missing mlp trace for layer {i}"))?;
                traced_layer = Some(ComponentLayerTrace {
                    attn_hidden: trace_attn_hidden
                        .ok_or_else(|| anyhow::anyhow!("missing attn trace for layer {i}"))?,
                    post_attn_norm: trace_post_attn_norm.ok_or_else(|| {
                        anyhow::anyhow!("missing post-attn norm trace for layer {i}")
                    })?,
                    mlp_swiglu: mlp_trace.swiglu,
                    mlp_out: mlp_trace.down,
                    layer_hidden: self
                        .hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} final hidden trace D2H: {e}"))?,
                });
            }
        }

        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if self.weights.config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        let final_norm_start = Instant::now();
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            hidden_dim,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
            "final rms_norm",
        )?;
        self.sync_stage_if_requested(collect_timings, "final rms_norm")?;
        if let Some(t) = timings.as_mut() {
            t.rms_norm_ms += final_norm_start.elapsed().as_secs_f64() * 1000.0;
        }

        if cuda_greedy {
            let lm_head_start = Instant::now();
            kernel_ffi::cuda_lm_head_argmax_bf16(
                self.ordinal,
                &self.normed_buf,
                &*self.weights.lm_head,
                &mut self.lm_head_block_best_vals,
                &mut self.lm_head_block_best_idxs,
                &mut self.argmax_buf,
                hidden_dim,
                vocab_size,
            )
            .map_err(|e| anyhow::anyhow!("cuda fused lm_head argmax 4b: {e}"))?;
            self.sync_stage_if_requested(collect_timings, "cuda fused lm_head argmax 4b")?;
            if let Some(t) = timings.as_mut() {
                t.lm_head_ms += lm_head_start.elapsed().as_secs_f64() * 1000.0;
            }
            let token_d2h_start = Instant::now();
            let token_bytes = self
                .argmax_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("argmax D2H: {e}"))?;
            if let Some(t) = timings.as_mut() {
                t.token_d2h_ms += token_d2h_start.elapsed().as_secs_f64() * 1000.0;
            }
            let sampled_token = u32::from_le_bytes(
                token_bytes[..4]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("argmax D2H returned truncated token buffer"))?,
            );
            return Ok((
                None,
                Some(sampled_token),
                None,
                traced_hidden,
                traced_layer,
                traced_linear,
            ));
        }

        let lm_head_start = Instant::now();
        if self.logits_buf.backend() == gpu_hal::Backend::Cuda
            && std::env::var_os("SUPERSONIC_LLAMA31_DISABLE_CUBLAS_LM_HEAD").is_none()
        {
            kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                self.ordinal,
                &mut self.logits_buf,
                &self.normed_buf,
                &*self.weights.lm_head,
                hidden_dim,
                vocab_size,
            )
            .map_err(|e| anyhow::anyhow!("cuBLAS lm_head matvec 4b: {e}"))?;
        } else {
            kernel_ffi::standalone_matvec_4b(
                self.ordinal,
                ScalarType::BF16,
                &mut self.logits_buf,
                &self.normed_buf,
                &*self.weights.lm_head,
                hidden_dim,
                vocab_size,
                &mut self.matvec_counter,
            )
            .map_err(|e| anyhow::anyhow!("lm_head matvec: {e}"))?;
        }
        self.sync_stage_if_requested(collect_timings, "lm_head matvec 4b")?;
        if let Some(t) = timings.as_mut() {
            t.lm_head_ms += lm_head_start.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some(target) = target_nll_token {
            if target as usize >= vocab_size {
                anyhow::bail!("target token {target} outside vocab size {vocab_size}");
            }
            let nll_start = Instant::now();
            if accumulate_target_nll {
                kernel_ffi::cuda_accumulate_target_nll_bf16(
                    self.ordinal,
                    &self.logits_buf,
                    target,
                    &mut self.target_nll_accum,
                    vocab_size,
                )
                .map_err(|e| anyhow::anyhow!("target NLL accumulate kernel: {e}"))?;
                if let Some(t) = timings.as_mut() {
                    t.logits_d2h_ms += nll_start.elapsed().as_secs_f64() * 1000.0;
                }
                return Ok((
                    None,
                    None,
                    Some(0.0),
                    traced_hidden,
                    traced_layer,
                    traced_linear,
                ));
            }
            let target_bytes = target.to_le_bytes();
            gpu_hal::copy_h2d(
                self.ordinal,
                self.target_nll_token.as_mut_ptr(),
                target_bytes.as_ptr() as *const c_void,
                target_bytes.len(),
            )
            .map_err(|e| anyhow::anyhow!("target NLL target H2D: {e}"))?;
            kernel_ffi::cuda_target_nll_bf16(
                self.ordinal,
                &self.logits_buf,
                &self.target_nll_token,
                &mut self.target_nll_value,
                1,
                vocab_size,
            )
            .map_err(|e| anyhow::anyhow!("target NLL kernel: {e}"))?;
            let nll_bytes = self
                .target_nll_value
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("target NLL D2H: {e}"))?;
            if let Some(t) = timings.as_mut() {
                t.logits_d2h_ms += nll_start.elapsed().as_secs_f64() * 1000.0;
            }
            let nll = f32::from_le_bytes(
                nll_bytes[..4]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("target NLL D2H returned truncated buffer"))?,
            );
            return Ok((
                None,
                None,
                Some(nll),
                traced_hidden,
                traced_layer,
                traced_linear,
            ));
        }

        let logits_d2h_start = Instant::now();
        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
        if let Some(t) = timings.as_mut() {
            t.logits_d2h_ms += logits_d2h_start.elapsed().as_secs_f64() * 1000.0;
        }
        Ok((
            Some(
                logits_bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
            ),
            None,
            None,
            traced_hidden,
            traced_layer,
            traced_linear,
        ))
    }

    fn component_decode_step_4b(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<Vec<f32>> {
        let (logits, _, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            None,
            None,
            false,
            None,
        )?;
        logits.ok_or_else(|| anyhow::anyhow!("component decode missing logits"))
    }

    fn component_decode_step_4b_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        let mut timings = DecodeStageTimings::default();
        let (logits, _, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            None,
            None,
            false,
            Some(&mut timings),
        )?;
        let logits =
            logits.ok_or_else(|| anyhow::anyhow!("component decode timings missing logits"))?;
        let sampling_start = Instant::now();
        let _ = Self::greedy_sample(&logits);
        timings.host_sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
        Ok((logits, timings))
    }

    pub fn component_decode_step_4b_cuda_fast_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        collect_timings: bool,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("component_decode_step_4b_cuda_fast_greedy requires CUDA backend");
        }
        let mut timings = DecodeStageTimings::default();
        let timing_slot = if collect_timings {
            Some(&mut timings)
        } else {
            None
        };
        let (_, sampled_token, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            true,
            None,
            None,
            false,
            timing_slot,
        )?;
        let sampled_token = sampled_token
            .ok_or_else(|| anyhow::anyhow!("component CUDA fast greedy missing sampled token"))?;
        Ok((sampled_token, timings))
    }

    pub fn component_decode_step_4b_certified_kv(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        certified_kv_decode: CertifiedKvDecodeParams,
    ) -> Result<Vec<f32>> {
        let (logits, _, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            Some(certified_kv_decode),
            None,
            false,
            None,
        )?;
        logits.ok_or_else(|| anyhow::anyhow!("certified KV decode missing logits"))
    }

    pub fn component_decode_step_4b_certified_kv_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        certified_kv_decode: CertifiedKvDecodeParams,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        let mut timings = DecodeStageTimings::default();
        let (logits, _, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            Some(certified_kv_decode),
            None,
            false,
            Some(&mut timings),
        )?;
        let logits = logits.ok_or_else(|| anyhow::anyhow!("certified KV decode missing logits"))?;
        Ok((logits, timings))
    }

    pub fn component_decode_step_4b_target_nll(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        target_token: u32,
        certified_kv_decode: Option<CertifiedKvDecodeParams>,
        collect_timings: bool,
    ) -> Result<(f32, DecodeStageTimings)> {
        let mut timings = DecodeStageTimings::default();
        let timing_slot = if collect_timings {
            Some(&mut timings)
        } else {
            None
        };
        let (_, _, nll, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            certified_kv_decode,
            Some(target_token),
            false,
            timing_slot,
        )?;
        let nll = nll.ok_or_else(|| anyhow::anyhow!("component target NLL missing output"))?;
        Ok((nll, timings))
    }

    pub fn reset_target_nll_accum(&mut self) -> Result<()> {
        gpu_hal::memset_zeros(
            self.ordinal,
            self.target_nll_accum.as_mut_ptr(),
            self.target_nll_accum.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("target NLL accumulator reset: {e}"))
    }

    pub fn read_target_nll_accum(&self) -> Result<f32> {
        let bytes = self
            .target_nll_accum
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("target NLL accumulator D2H: {e}"))?;
        Ok(f32::from_le_bytes(bytes[..4].try_into().map_err(|_| {
            anyhow::anyhow!("target NLL accumulator D2H truncated")
        })?))
    }

    pub fn component_decode_step_4b_accumulate_target_nll(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        target_token: u32,
        certified_kv_decode: Option<CertifiedKvDecodeParams>,
        collect_timings: bool,
    ) -> Result<DecodeStageTimings> {
        let mut timings = DecodeStageTimings::default();
        let timing_slot = if collect_timings {
            Some(&mut timings)
        } else {
            None
        };
        let (_, _, nll, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            false,
            certified_kv_decode,
            Some(target_token),
            true,
            timing_slot,
        )?;
        nll.ok_or_else(|| anyhow::anyhow!("component accumulated target NLL missing marker"))?;
        Ok(timings)
    }

    pub fn component_decode_step_4b_certified_kv_cuda_fast_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        certified_kv_decode: CertifiedKvDecodeParams,
        collect_timings: bool,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("certified KV fast greedy requires CUDA backend");
        }
        let mut timings = DecodeStageTimings::default();
        let timing_slot = if collect_timings {
            Some(&mut timings)
        } else {
            None
        };
        let (_, sampled_token, _, _, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            None,
            true,
            Some(certified_kv_decode),
            None,
            false,
            timing_slot,
        )?;
        let sampled_token = sampled_token
            .ok_or_else(|| anyhow::anyhow!("certified KV fast greedy missing sampled token"))?;
        Ok((sampled_token, timings))
    }

    pub fn component_decode_step_4b_traced(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_input_layer: usize,
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        let (logits, _, _, trace, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            Some(trace_input_layer),
            None,
            None,
            false,
            None,
            None,
            false,
            None,
        )?;
        let logits = logits.ok_or_else(|| anyhow::anyhow!("component trace missing logits"))?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing trace for layer {trace_input_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLayerTrace)> {
        let (logits, _, _, _, trace, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            Some(trace_layer),
            None,
            false,
            None,
            None,
            false,
            None,
        )?;
        let logits =
            logits.ok_or_else(|| anyhow::anyhow!("component layer trace missing logits"))?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing stage trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_linear_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLinearTrace)> {
        let (logits, _, _, _, _, trace) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            Some(trace_layer),
            false,
            None,
            None,
            false,
            None,
        )?;
        let logits =
            logits.ok_or_else(|| anyhow::anyhow!("component linear trace missing logits"))?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing linear trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    fn component_decode_full_attention_layer(
        &mut self,
        idx: usize,
        seqlen_offset: usize,
        trace_output: bool,
        certified_kv_decode: Option<CertifiedKvDecodeParams>,
        mut timings: Option<&mut DecodeStageTimings>,
    ) -> Result<Option<ComponentFullAttentionTrace>> {
        let hidden_dim = self.weights.config.hidden_size;
        let num_q_heads = self.weights.config.num_attention_heads;
        let num_kv_heads = self.weights.config.num_key_value_heads;
        let head_dim = self.weights.config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?
            .q_proj_w
            .shape()[0];
        let has_attn_gate = match q_proj_dim {
            dim if dim == q_dim => false,
            dim if dim == q_dim * 2 => true,
            dim => {
                return Err(anyhow::anyhow!(
                    "layer {idx}: unsupported full-attention q_proj rows {dim}, expected {q_dim} or {}",
                    q_dim * 2
                ));
            }
        };
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = self.weights.config.rotary_dim();
        let kv_len = seqlen_offset + 1;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let late_mixed_layers = llama31_int8_late_full_mixed_layers();
        let use_late_decode_mixed = self.weights.is_int8
            && idx + late_mixed_layers >= self.weights.config.num_hidden_layers;
        let use_late_q_mixed =
            use_late_decode_mixed && llama31_int8_late_full_mixed_component_enabled("q");
        let use_late_k_mixed =
            use_late_decode_mixed && llama31_int8_late_full_mixed_component_enabled("k");
        let use_late_v_mixed =
            use_late_decode_mixed && llama31_int8_late_full_mixed_component_enabled("v");
        let collect_timings = timings.is_some();
        let mut q_proj_trace = None;
        let mut gate_proj_trace = None;
        let mut k_proj_trace = None;
        let mut v_proj_trace = None;
        let mut q_rope_trace = None;
        let mut k_rope_trace = None;
        let mut pre_gate_trace = None;
        let mut gated_trace = None;
        let mut proj_out_trace = None;
        let mut scratch = self
            .component_full_attn_scratch
            .take()
            .ok_or_else(|| anyhow::anyhow!("component full-attn scratch missing"))?;
        let result = (|| -> Result<Option<ComponentFullAttentionTrace>> {
            let config = &self.weights.config;
            let fw = self.weights.layers[idx]
                .full
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
            let q_full = &mut scratch.q_full;
            let query_buf = &mut scratch.query_buf;
            let gate_buf = &mut scratch.gate_buf;
            let k_buf = &mut scratch.k_buf;
            let v_buf = &mut scratch.v_buf;
            let q_normed = &mut scratch.q_normed;
            let k_normed = &mut scratch.k_normed;
            let attn_q = &mut scratch.attn_q;
            let attn_k_step = &mut scratch.attn_k_step;
            let attn_v_step = &mut scratch.attn_v_step;
            let attn_out_f32 = &mut scratch.attn_out_f32;
            let attn_out_bf16 = &mut scratch.attn_out_bf16;
            let attn_flat = &mut scratch.attn_flat;
            let gated = &mut scratch.gated;
            let proj_out = &mut scratch.proj_out;
            let use_cublas_full_proj = self.normed_buf.backend() == gpu_hal::Backend::Cuda
                && std::env::var_os("SUPERSONIC_LLAMA31_DISABLE_CUBLAS_FULL_ATTN_PROJ").is_none()
                && !use_late_q_mixed
                && !use_late_k_mixed
                && !use_late_v_mixed
                && fw.q_proj_scale.is_none()
                && fw.q_proj_int8_scale.is_none()
                && fw.q_proj_int4_scale.is_none()
                && fw.k_proj_scale.is_none()
                && fw.k_proj_int8_scale.is_none()
                && fw.k_proj_int4_scale.is_none()
                && fw.v_proj_scale.is_none()
                && fw.v_proj_int8_scale.is_none()
                && fw.v_proj_int4_scale.is_none()
                && fw.o_proj_scale.is_none()
                && fw.o_proj_int8_scale.is_none()
                && fw.o_proj_int4_scale.is_none();
            let mixed_lhs = if !use_cublas_full_proj
                && (use_late_q_mixed || use_late_k_mixed || use_late_v_mixed)
            {
                prefill_engine::prepare_int8_mixed_lhs(
                    self.ordinal,
                    1,
                    1,
                    hidden_dim,
                    &self.normed_buf,
                    &self.weights,
                )?
            } else {
                None
            };

            let proj_start = Instant::now();
            if use_cublas_full_proj {
                kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                    self.ordinal,
                    q_full,
                    &self.normed_buf,
                    &fw.q_proj_w,
                    hidden_dim,
                    q_proj_dim,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS q_proj: {e}"))?;
            } else if use_late_q_mixed {
                if let Some(sc) = fw.q_proj_int8_scale.as_ref() {
                    prefill_engine::matmul_int8_mixed_prepared_host(
                        self.ordinal,
                        1,
                        1,
                        q_proj_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &self.weights,
                        &format!(
                            "{}.layers.{idx}.self_attn.q_proj.weight",
                            self.weights.weight_prefix
                        ),
                        &fw.q_proj_w,
                        sc,
                        q_full,
                        mixed_lhs.as_ref(),
                    )?;
                } else {
                    matmul_proj(
                        self.ordinal,
                        1,
                        1,
                        q_proj_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &fw.q_proj_w,
                        fw.q_proj_scale.as_ref(),
                        fw.q_proj_int8_scale.as_ref(),
                        self.weights.fp8_block_size,
                        q_full,
                        fw.q_proj_int4_scale.as_ref(),
                        fw.q_proj_int4_zero.as_ref(),
                        self.weights.int4_group_size,
                    )?;
                }
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    q_proj_dim,
                    hidden_dim,
                    &self.normed_buf,
                    &fw.q_proj_w,
                    fw.q_proj_scale.as_ref(),
                    fw.q_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    q_full,
                    fw.q_proj_int4_scale.as_ref(),
                    fw.q_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }
            if has_attn_gate {
                kernel_ffi::prefill_ffi::split_qgate(
                    self.ordinal,
                    ScalarType::BF16,
                    1,
                    num_q_heads,
                    head_dim,
                    &q_full,
                    query_buf,
                    gate_buf,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} split qgate: {e}"))?;
            } else {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    query_buf.as_ptr() as *mut c_void,
                    q_full.as_ptr(),
                    q_dim * elem_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} q copy: {e}"))?;
            }

            if use_cublas_full_proj {
                kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                    self.ordinal,
                    k_buf,
                    &self.normed_buf,
                    &fw.k_proj_w,
                    hidden_dim,
                    kv_dim,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS k_proj: {e}"))?;
            } else if use_late_k_mixed {
                if let Some(sc) = fw.k_proj_int8_scale.as_ref() {
                    prefill_engine::matmul_int8_mixed_prepared_host(
                        self.ordinal,
                        1,
                        1,
                        kv_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &self.weights,
                        &format!(
                            "{}.layers.{idx}.self_attn.k_proj.weight",
                            self.weights.weight_prefix
                        ),
                        &fw.k_proj_w,
                        sc,
                        k_buf,
                        mixed_lhs.as_ref(),
                    )?;
                } else {
                    matmul_proj(
                        self.ordinal,
                        1,
                        1,
                        kv_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &fw.k_proj_w,
                        fw.k_proj_scale.as_ref(),
                        fw.k_proj_int8_scale.as_ref(),
                        self.weights.fp8_block_size,
                        k_buf,
                        fw.k_proj_int4_scale.as_ref(),
                        fw.k_proj_int4_zero.as_ref(),
                        self.weights.int4_group_size,
                    )?;
                }
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    kv_dim,
                    hidden_dim,
                    &self.normed_buf,
                    &fw.k_proj_w,
                    fw.k_proj_scale.as_ref(),
                    fw.k_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    k_buf,
                    fw.k_proj_int4_scale.as_ref(),
                    fw.k_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }
            if use_cublas_full_proj {
                kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                    self.ordinal,
                    v_buf,
                    &self.normed_buf,
                    &fw.v_proj_w,
                    hidden_dim,
                    kv_dim,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS v_proj: {e}"))?;
            } else if use_late_v_mixed {
                if let Some(sc) = fw.v_proj_int8_scale.as_ref() {
                    prefill_engine::matmul_int8_mixed_prepared_host(
                        self.ordinal,
                        1,
                        1,
                        kv_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &self.weights,
                        &format!(
                            "{}.layers.{idx}.self_attn.v_proj.weight",
                            self.weights.weight_prefix
                        ),
                        &fw.v_proj_w,
                        sc,
                        v_buf,
                        mixed_lhs.as_ref(),
                    )?;
                } else {
                    matmul_proj(
                        self.ordinal,
                        1,
                        1,
                        kv_dim,
                        hidden_dim,
                        &self.normed_buf,
                        &fw.v_proj_w,
                        fw.v_proj_scale.as_ref(),
                        fw.v_proj_int8_scale.as_ref(),
                        self.weights.fp8_block_size,
                        v_buf,
                        fw.v_proj_int4_scale.as_ref(),
                        fw.v_proj_int4_zero.as_ref(),
                        self.weights.int4_group_size,
                    )?;
                }
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    kv_dim,
                    hidden_dim,
                    &self.normed_buf,
                    &fw.v_proj_w,
                    fw.v_proj_scale.as_ref(),
                    fw.v_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    v_buf,
                    fw.v_proj_int4_scale.as_ref(),
                    fw.v_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }

            maybe_attn_rms_norm_rows(
                config,
                self.ordinal,
                num_q_heads,
                head_dim,
                &query_buf,
                fw.q_norm_w.as_ref(),
                q_normed,
                &format!("layer {idx} q norm"),
            )?;
            gpu_hal::copy_d2d(
                self.ordinal,
                query_buf.as_ptr() as *mut c_void,
                q_normed.as_ptr(),
                q_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} q norm copy: {e}"))?;

            maybe_attn_rms_norm_rows(
                config,
                self.ordinal,
                num_kv_heads,
                head_dim,
                &k_buf,
                fw.k_norm_w.as_ref(),
                k_normed,
                &format!("layer {idx} k norm"),
            )?;
            gpu_hal::copy_d2d(
                self.ordinal,
                k_buf.as_ptr() as *mut c_void,
                k_normed.as_ptr(),
                kv_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} k norm copy: {e}"))?;
            self.sync_stage_if_requested(
                collect_timings,
                &format!("layer {idx} full attention proj"),
            )?;
            if let Some(t) = timings.as_mut() {
                t.persistent_full_attn_proj_ms += proj_start.elapsed().as_secs_f64() * 1000.0;
            }
            if trace_output {
                q_proj_trace = Some(
                    query_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} q proj trace D2H: {e}"))?,
                );
                gate_proj_trace = Some(
                    gate_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} gate proj trace D2H: {e}"))?,
                );
                k_proj_trace = Some(
                    k_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} k proj trace D2H: {e}"))?,
                );
                v_proj_trace = Some(
                    v_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} v proj trace D2H: {e}"))?,
                );
            }

            kernel_ffi::prefill_ffi::apply_rope_prefill(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_q_heads,
                head_dim,
                rotary_dim,
                &self.rotary.cos,
                &self.rotary.sin,
                seqlen_offset,
                query_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} q rope: {e}"))?;
            kernel_ffi::prefill_ffi::apply_rope_prefill(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_kv_heads,
                head_dim,
                rotary_dim,
                &self.rotary.cos,
                &self.rotary.sin,
                seqlen_offset,
                k_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} k rope: {e}"))?;
            if trace_output {
                q_rope_trace = Some(
                    query_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} q rope trace D2H: {e}"))?,
                );
                k_rope_trace = Some(
                    k_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} k rope trace D2H: {e}"))?,
                );
            }

            kernel_ffi::prefill_ffi::transpose_shd_hsd(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_q_heads,
                head_dim,
                query_buf,
                attn_q,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} q transpose: {e}"))?;
            kernel_ffi::prefill_ffi::transpose_shd_hsd(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_kv_heads,
                head_dim,
                k_buf,
                attn_k_step,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} k transpose: {e}"))?;
            kernel_ffi::prefill_ffi::transpose_shd_hsd(
                self.ordinal,
                ScalarType::BF16,
                1,
                num_kv_heads,
                head_dim,
                v_buf,
                attn_v_step,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} v transpose: {e}"))?;

            let ls = &mut self.state.layers[idx];
            let use_certified_cuda = certified_kv_decode
                .map(|cfg| kernel_ffi::certified_kv::aligned_tokens(kv_len, cfg.block_size) > 0)
                .unwrap_or(false);
            let capacity_offset = if use_certified_cuda {
                self.decode_context_limit
                    .unwrap_or(seqlen_offset + 1)
                    .saturating_sub(1)
                    .max(seqlen_offset)
            } else {
                seqlen_offset
            };
            if !(use_certified_cuda && ls.certified_kv_gpu_tail_only) {
                ls.ensure_kv_capacity(
                    capacity_offset,
                    self.ordinal,
                    config,
                    self.kv_chunk_size,
                    self.kv_fp8,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} kv alloc: {e}"))?;
                if let (Some(cache_k), Some(cache_v)) =
                    (ls.kv_cache_k.as_mut(), ls.kv_cache_v.as_mut())
                {
                    kernel_ffi::certified_kv::copy_step_bf16(
                        self.ordinal,
                        attn_k_step,
                        attn_v_step,
                        cache_k,
                        cache_v,
                        seqlen_offset,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} cache KV step write: {e}"))?;
                }
            } else {
                let tail_slot = seqlen_offset % certified_kv_decode.unwrap().block_size;
                let tail_k = ls.certified_kv_tail_k.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("layer {idx} certified tail key buffer missing")
                })?;
                let tail_v = ls.certified_kv_tail_v.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("layer {idx} certified tail value buffer missing")
                })?;
                kernel_ffi::certified_kv::copy_step_bf16(
                    self.ordinal,
                    attn_k_step,
                    attn_v_step,
                    tail_k,
                    tail_v,
                    tail_slot,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} certified tail KV step write: {e}"))?;
            }

            let cap = if use_certified_cuda && ls.certified_kv_gpu_tail_only {
                ls.certified_kv_host_k
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("layer {idx} certified host key buffer missing")
                    })?
                    .shape()[2]
            } else {
                ls.kv_cache_k.as_ref().unwrap().shape()[2]
            };

            let core_start = Instant::now();
            if use_certified_cuda {
                let cfg = certified_kv_decode
                    .expect("certified KV decode config is present when enabled");
                let mut certified_attn_already_flat = false;
                let block_size = cfg.block_size;
                let value_group_size = cfg.value_group_size;
                let bf16_values = cfg.bf16_values;
                let aligned = kernel_ffi::certified_kv::aligned_tokens(kv_len, block_size);
                let tail_len = kv_len - aligned;
                let cap_aligned = kernel_ffi::certified_kv::aligned_tokens(cap, block_size);
                let key_i8_shape = [num_kv_heads, cap_aligned, head_dim];
                let key_scale_shape = [num_kv_heads, cap_aligned / block_size, head_dim];
                let needs_key_alloc = ls
                    .certified_kv_key_i8
                    .as_ref()
                    .map(|buf| buf.shape() != key_i8_shape)
                    .unwrap_or(true)
                    || ls
                        .certified_kv_key_scale
                        .as_ref()
                        .map(|buf| buf.shape() != key_scale_shape)
                        .unwrap_or(true)
                    || ls
                        .certified_kv_key_zero
                        .as_ref()
                        .map(|buf| buf.shape() != key_scale_shape)
                        .unwrap_or(true);
                if needs_key_alloc {
                    ls.certified_kv_key_i8 = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::U8, &key_i8_shape).map_err(
                            |e| {
                                anyhow::anyhow!("layer {idx} certified KV decode key_i8 alloc: {e}")
                            },
                        )?,
                    );
                    ls.certified_kv_key_scale = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape).map_err(
                            |e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode key_scale alloc: {e}"
                                )
                            },
                        )?,
                    );
                    ls.certified_kv_key_zero = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::F32, &key_scale_shape).map_err(
                            |e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode key_zero alloc: {e}"
                                )
                            },
                        )?,
                    );
                    ls.certified_kv_key_tokens = 0;
                }
                if ls.certified_kv_key_tokens < aligned {
                    let key_i8 = ls.certified_kv_key_i8.as_mut().unwrap();
                    let key_scale = ls.certified_kv_key_scale.as_mut().unwrap();
                    let key_zero = ls.certified_kv_key_zero.as_mut().unwrap();
                    let start_block = ls.certified_kv_key_tokens / block_size;
                    let block_count = (aligned - ls.certified_kv_key_tokens) / block_size;
                    if timings.is_some() {
                        gpu_hal::sync(self.ordinal).map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV key quantize pre-sync: {e}")
                        })?;
                    }
                    let key_quantize_start = Instant::now();
                    if !ls.certified_kv_gpu_tail_only {
                        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
                        kernel_ffi::certified_kv::quantize_bf16_keys_range(
                            self.ordinal,
                            cache_k_ref,
                            start_block,
                            block_count,
                            block_size,
                            key_i8,
                            key_scale,
                            key_zero,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV decode key quantize: {e}")
                        })?;
                    } else {
                        if block_count != 1 {
                            anyhow::bail!(
                                "layer {idx} certified tail key quantize expects single completed block, got {block_count}"
                            );
                        }
                        let tail_k = ls.certified_kv_tail_k.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("layer {idx} certified tail key missing")
                        })?;
                        let quant_k = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::BF16,
                            &[1, num_kv_heads, block_size, head_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} certified quant_k alloc: {e}"))?;
                        gpu_hal::copy_d2d(
                            self.ordinal,
                            quant_k.as_ptr() as *mut c_void,
                            tail_k.as_ptr(),
                            tail_k.len_bytes(),
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} certified quant_k copy: {e}"))?;
                        let mut tmp_key_i8 = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::U8,
                            &[num_kv_heads, block_size, head_dim],
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified tmp key_i8 alloc: {e}")
                        })?;
                        let mut tmp_key_scale = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::F32,
                            &[num_kv_heads, 1, head_dim],
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified tmp key_scale alloc: {e}")
                        })?;
                        let mut tmp_key_zero = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::F32,
                            &[num_kv_heads, 1, head_dim],
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified tmp key_zero alloc: {e}")
                        })?;
                        kernel_ffi::certified_kv::quantize_bf16_keys(
                            self.ordinal,
                            &quant_k,
                            block_size,
                            block_size,
                            &mut tmp_key_i8,
                            &mut tmp_key_scale,
                            &mut tmp_key_zero,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified tail key quantize: {e}")
                        })?;
                        let dst_block = start_block;
                        let dst_key_stride = key_i8.shape()[1] * head_dim;
                        let src_key_stride = block_size * head_dim;
                        let key_block_bytes = src_key_stride * ScalarType::U8.size_in_bytes();
                        let dst_scale_stride = key_scale.shape()[1] * head_dim;
                        let scale_block_bytes = head_dim * ScalarType::F32.size_in_bytes();
                        for h in 0..num_kv_heads {
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                key_i8.offset_ptr(
                                    (h * dst_key_stride + dst_block * block_size * head_dim)
                                        * ScalarType::U8.size_in_bytes(),
                                ) as *mut c_void,
                                tmp_key_i8.offset_ptr(
                                    h * src_key_stride * ScalarType::U8.size_in_bytes(),
                                ),
                                key_block_bytes,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified key_i8 block copy h={h}: {e}"
                                )
                            })?;
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                key_scale.offset_ptr(
                                    (h * dst_scale_stride + dst_block * head_dim)
                                        * ScalarType::F32.size_in_bytes(),
                                ) as *mut c_void,
                                tmp_key_scale
                                    .offset_ptr(h * head_dim * ScalarType::F32.size_in_bytes()),
                                scale_block_bytes,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified key_scale block copy h={h}: {e}"
                                )
                            })?;
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                key_zero.offset_ptr(
                                    (h * dst_scale_stride + dst_block * head_dim)
                                        * ScalarType::F32.size_in_bytes(),
                                ) as *mut c_void,
                                tmp_key_zero
                                    .offset_ptr(h * head_dim * ScalarType::F32.size_in_bytes()),
                                scale_block_bytes,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified key_zero block copy h={h}: {e}"
                                )
                            })?;
                        }
                    }
                    if let Some(t) = timings.as_mut() {
                        gpu_hal::sync(self.ordinal).map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV key quantize synchronize: {e}"
                            )
                        })?;
                        t.certified_kv_key_quantize_ms +=
                            key_quantize_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    ls.certified_kv_key_tokens = aligned;
                }

                let host_shape = [1, num_kv_heads, cap, head_dim];
                let needs_host_alloc = ls
                    .certified_kv_host_k
                    .as_ref()
                    .map(|buf| buf.shape() != host_shape)
                    .unwrap_or(true)
                    || ls
                        .certified_kv_host_v
                        .as_ref()
                        .map(|buf| buf.shape() != host_shape)
                        .unwrap_or(true);
                if needs_host_alloc {
                    ls.certified_kv_host_k = Some(
                        HostBuffer::zeros(self.ordinal, ScalarType::BF16, &host_shape).map_err(
                            |e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV Tier-2 host key alloc: {e}"
                                )
                            },
                        )?,
                    );
                    ls.certified_kv_host_v = Some(
                        HostBuffer::zeros(self.ordinal, ScalarType::BF16, &host_shape).map_err(
                            |e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV Tier-2 host value alloc: {e}"
                                )
                            },
                        )?,
                    );
                    ls.certified_kv_host_tokens = 0;
                }
                if ls.certified_kv_host_tokens < kv_len {
                    let host_k = ls.certified_kv_host_k.as_mut().unwrap();
                    let host_v = ls.certified_kv_host_v.as_mut().unwrap();
                    if !ls.certified_kv_gpu_tail_only {
                        let start_token = ls.certified_kv_host_tokens;
                        let token_count = kv_len - start_token;
                        let byte_count = token_count * head_dim * elem_bytes;
                        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
                        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
                        for h in 0..num_kv_heads {
                            let offset = (h * cap + start_token) * head_dim * elem_bytes;
                            gpu_hal::copy_d2h(
                                self.ordinal,
                                host_k.offset_mut_ptr(offset),
                                cache_k_ref.offset_ptr(offset),
                                byte_count,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV Tier-2 key D2H h={h}: {e}"
                                )
                            })?;
                            gpu_hal::copy_d2h(
                                self.ordinal,
                                host_v.offset_mut_ptr(offset),
                                cache_v_ref.offset_ptr(offset),
                                byte_count,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV Tier-2 value D2H h={h}: {e}"
                                )
                            })?;
                        }
                    } else {
                        let token_offset = kv_len - 1;
                        let src_stride = head_dim * elem_bytes;
                        for h in 0..num_kv_heads {
                            let dst = (h * cap + token_offset) * head_dim * elem_bytes;
                            gpu_hal::copy_d2h(
                                self.ordinal,
                                host_k.offset_mut_ptr(dst),
                                attn_k_step.offset_ptr(h * src_stride),
                                src_stride,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified Tier-2 key step D2H h={h}: {e}"
                                )
                            })?;
                            gpu_hal::copy_d2h(
                                self.ordinal,
                                host_v.offset_mut_ptr(dst),
                                attn_v_step.offset_ptr(h * src_stride),
                                src_stride,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified Tier-2 value step D2H h={h}: {e}"
                                )
                            })?;
                        }
                    }
                    ls.certified_kv_host_tokens = kv_len;
                }

                if !ls.certified_kv_gpu_tail_only {
                    let value_i4_shape_preoffload = [num_kv_heads, cap_aligned, head_dim / 2];
                    let value_meta_shape_preoffload =
                        [num_kv_heads, cap_aligned, head_dim / value_group_size];
                    let value_error_shape_preoffload = [num_kv_heads, cap_aligned / block_size];
                    let needs_value_alloc_preoffload = ls
                        .certified_kv_value_i4
                        .as_ref()
                        .map(|buf| buf.shape() != value_i4_shape_preoffload)
                        .unwrap_or(true)
                        || ls
                            .certified_kv_value_scale
                            .as_ref()
                            .map(|buf| buf.shape() != value_meta_shape_preoffload)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_zero
                            .as_ref()
                            .map(|buf| buf.shape() != value_meta_shape_preoffload)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_error
                            .as_ref()
                            .map(|buf| buf.shape() != value_error_shape_preoffload)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_norm
                            .as_ref()
                            .map(|buf| buf.shape() != value_error_shape_preoffload)
                            .unwrap_or(true);
                    if needs_value_alloc_preoffload {
                        ls.certified_kv_value_i4 = Some(
                            GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::U8,
                                &value_i4_shape_preoffload,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value_i4 alloc: {e}"
                                )
                            })?,
                        );
                        ls.certified_kv_value_scale = Some(
                            GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F16,
                                &value_meta_shape_preoffload,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value_scale alloc: {e}"
                                )
                            })?,
                        );
                        ls.certified_kv_value_zero = Some(
                            GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F16,
                                &value_meta_shape_preoffload,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value_zero alloc: {e}"
                                )
                            })?,
                        );
                        ls.certified_kv_value_error = Some(
                            GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F32,
                                &value_error_shape_preoffload,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value_error alloc: {e}"
                                )
                            })?,
                        );
                        ls.certified_kv_value_norm = Some(
                            GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F32,
                                &value_error_shape_preoffload,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value_norm alloc: {e}"
                                )
                            })?,
                        );
                        ls.certified_kv_value_tokens = 0;
                    }
                    if ls.certified_kv_value_tokens < aligned {
                        let start_block = ls.certified_kv_value_tokens / block_size;
                        let block_count = (aligned - ls.certified_kv_value_tokens) / block_size;
                        if timings.is_some() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value quantize pre-sync: {e}"
                                )
                            })?;
                        }
                        let value_quantize_start = Instant::now();
                        kernel_ffi::certified_kv::quantize_bf16_values_range(
                            self.ordinal,
                            ls.kv_cache_v.as_ref().unwrap(),
                            start_block,
                            block_count,
                            block_size,
                            value_group_size,
                            ls.certified_kv_value_i4.as_mut().unwrap(),
                            ls.certified_kv_value_scale.as_mut().unwrap(),
                            ls.certified_kv_value_zero.as_mut().unwrap(),
                            ls.certified_kv_value_error.as_mut().unwrap(),
                            ls.certified_kv_value_norm.as_mut().unwrap(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV decode value quantize: {e}")
                        })?;
                        if let Some(t) = timings.as_mut() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value quantize synchronize: {e}"
                                )
                            })?;
                            t.certified_kv_value_quantize_ms +=
                                value_quantize_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        ls.certified_kv_value_tokens = aligned;
                    }

                    let tail_shape = [num_kv_heads, block_size, head_dim];
                    let needs_tail_alloc = ls
                        .certified_kv_tail_k
                        .as_ref()
                        .map(|buf| buf.shape() != tail_shape)
                        .unwrap_or(true)
                        || ls
                            .certified_kv_tail_v
                            .as_ref()
                            .map(|buf| buf.shape() != tail_shape)
                            .unwrap_or(true);
                    if needs_tail_alloc {
                        ls.certified_kv_tail_k = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &tail_shape).map_err(
                                |e| anyhow::anyhow!("layer {idx} certified tail k alloc: {e}"),
                            )?,
                        );
                        ls.certified_kv_tail_v = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &tail_shape).map_err(
                                |e| anyhow::anyhow!("layer {idx} certified tail v alloc: {e}"),
                            )?,
                        );
                    }
                    if tail_len > 0 {
                        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
                        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
                        let tail_k = ls.certified_kv_tail_k.as_mut().unwrap();
                        let tail_v = ls.certified_kv_tail_v.as_mut().unwrap();
                        kernel_ffi::certified_kv::copy_token_range_bf16(
                            self.ordinal,
                            cache_k_ref,
                            cache_v_ref,
                            tail_k,
                            tail_v,
                            aligned,
                            0,
                            tail_len,
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} certified tail KV seed: {e}"))?;
                    }
                    ls.kv_cache_k = None;
                    ls.kv_cache_v = None;
                    ls.certified_kv_gpu_tail_only = true;
                }

                let key_i8 = ls.certified_kv_key_i8.as_ref().unwrap();
                let key_scale = ls.certified_kv_key_scale.as_ref().unwrap();
                let key_zero = ls.certified_kv_key_zero.as_ref().unwrap();
                let full_value_gpu_ref = ls.kv_cache_v.as_ref();
                let tail_key_bf16 = if tail_len > 0 {
                    ls.certified_kv_tail_k.as_ref()
                } else {
                    None
                };
                let tail_value_bf16 = if tail_len > 0 {
                    ls.certified_kv_tail_v.as_ref()
                } else {
                    None
                };
                let tail_compact_shape = [num_kv_heads, tail_len, head_dim];
                if tail_len > 0 {
                    if scratch
                        .certified_tail_key_compact
                        .as_ref()
                        .map(|buf| buf.shape() != tail_compact_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_tail_key_compact = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &tail_compact_shape)
                                .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV compact tail key alloc: {e}"
                                )
                            })?,
                        );
                    }
                    if scratch
                        .certified_tail_value_compact
                        .as_ref()
                        .map(|buf| buf.shape() != tail_compact_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_tail_value_compact = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &tail_compact_shape)
                                .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV compact tail value alloc: {e}"
                                )
                            })?,
                        );
                    }
                    let src_tail_key = tail_key_bf16.ok_or_else(|| {
                        anyhow::anyhow!("layer {idx} certified KV tail key missing")
                    })?;
                    let src_tail_value = tail_value_bf16.ok_or_else(|| {
                        anyhow::anyhow!("layer {idx} certified KV tail value missing")
                    })?;
                    let dst_tail_key = scratch.certified_tail_key_compact.as_mut().unwrap();
                    let dst_tail_value = scratch.certified_tail_value_compact.as_mut().unwrap();
                    let src_stride_bytes = src_tail_key.shape()[1] * head_dim * elem_bytes;
                    let dst_stride_bytes = tail_len * head_dim * elem_bytes;
                    for h in 0..num_kv_heads {
                        gpu_hal::copy_d2d(
                            self.ordinal,
                            dst_tail_key.offset_ptr(h * dst_stride_bytes) as *mut c_void,
                            src_tail_key.offset_ptr(h * src_stride_bytes),
                            dst_stride_bytes,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV compact tail key copy h={h}: {e}"
                            )
                        })?;
                        gpu_hal::copy_d2d(
                            self.ordinal,
                            dst_tail_value.offset_ptr(h * dst_stride_bytes) as *mut c_void,
                            src_tail_value.offset_ptr(h * src_stride_bytes),
                            dst_stride_bytes,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV compact tail value copy h={h}: {e}"
                            )
                        })?;
                    }
                }
                let tail_key_kernel_ref = if tail_len > 0 {
                    scratch.certified_tail_key_compact.as_ref()
                } else {
                    None
                };
                let tail_value_kernel_ref = if tail_len > 0 {
                    scratch.certified_tail_value_compact.as_ref()
                } else {
                    None
                };
                let score_shape = [num_q_heads, cap];
                if scratch
                    .certified_score_scratch
                    .as_ref()
                    .map(|buf| buf.shape() != score_shape)
                    .unwrap_or(true)
                {
                    scratch.certified_score_scratch = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::F32, &score_shape).map_err(
                            |e| anyhow::anyhow!("layer {idx} certified KV decode score alloc: {e}"),
                        )?,
                    );
                }
                let softmax_stats_shape = [num_q_heads, 2];
                if scratch
                    .certified_softmax_stats
                    .as_ref()
                    .map(|buf| buf.shape() != softmax_stats_shape)
                    .unwrap_or(true)
                {
                    scratch.certified_softmax_stats = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::F32, &softmax_stats_shape)
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV softmax stats alloc: {e}")
                            })?,
                    );
                }
                let final_block_mass_shape = [num_q_heads, aligned / block_size];
                if aligned > 0
                    && scratch
                        .certified_final_block_mass
                        .as_ref()
                        .map(|buf| buf.shape() != final_block_mass_shape)
                        .unwrap_or(true)
                {
                    scratch.certified_final_block_mass = Some(
                        GpuBuffer::zeros(self.ordinal, ScalarType::F32, &final_block_mass_shape)
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV final block mass alloc: {e}"
                                )
                            })?,
                    );
                }
                let score_scratch = scratch.certified_score_scratch.as_mut().unwrap();
                if bf16_values {
                    let full_value_gpu_ref = full_value_gpu_ref.ok_or_else(|| {
                        anyhow::anyhow!(
                            "layer {idx} certified BF16-values debug path requires full GPU BF16 values"
                        )
                    })?;
                    if timings.is_some() {
                        gpu_hal::sync(self.ordinal).map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV attention pre-sync: {e}")
                        })?;
                    }
                    let attend_start = Instant::now();
                    kernel_ffi::certified_kv::attend_int8_bf16_values_strided(
                        self.ordinal,
                        attn_q,
                        key_i8,
                        key_scale,
                        key_zero,
                        full_value_gpu_ref,
                        tail_key_kernel_ref,
                        kv_len,
                        block_size,
                        num_q_heads / num_kv_heads,
                        1.0 / (head_dim as f32).sqrt(),
                        score_scratch,
                        attn_out_bf16,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} certified KV decode attention: {e}")
                    })?;
                    if let Some(t) = timings.as_mut() {
                        gpu_hal::sync(self.ordinal).map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV attention synchronize: {e}")
                        })?;
                        t.certified_kv_attend_ms += attend_start.elapsed().as_secs_f64() * 1000.0;
                    }
                } else {
                    let value_i4_shape = [num_kv_heads, cap_aligned, head_dim / 2];
                    let value_meta_shape = [num_kv_heads, cap_aligned, head_dim / value_group_size];
                    let value_error_shape = [num_kv_heads, cap_aligned / block_size];
                    let needs_value_alloc = ls
                        .certified_kv_value_i4
                        .as_ref()
                        .map(|buf| buf.shape() != value_i4_shape)
                        .unwrap_or(true)
                        || ls
                            .certified_kv_value_scale
                            .as_ref()
                            .map(|buf| buf.shape() != value_meta_shape)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_zero
                            .as_ref()
                            .map(|buf| buf.shape() != value_meta_shape)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_error
                            .as_ref()
                            .map(|buf| buf.shape() != value_error_shape)
                            .unwrap_or(true)
                        || ls
                            .certified_kv_value_norm
                            .as_ref()
                            .map(|buf| buf.shape() != value_error_shape)
                            .unwrap_or(true);
                    if needs_value_alloc {
                        ls.certified_kv_value_i4 = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U8, &value_i4_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV decode value_i4 alloc: {e}"
                                    )
                                })?,
                        );
                        ls.certified_kv_value_scale = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV decode value_scale alloc: {e}"
                                    )
                                })?,
                        );
                        ls.certified_kv_value_zero = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F16, &value_meta_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV decode value_zero alloc: {e}"
                                    )
                                })?,
                        );
                        ls.certified_kv_value_error = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV decode value_error alloc: {e}"
                                    )
                                })?,
                        );
                        ls.certified_kv_value_norm = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &value_error_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV decode value_norm alloc: {e}"
                                    )
                                })?,
                        );
                        ls.certified_kv_value_tokens = 0;
                    }
                    if ls.certified_kv_value_tokens < aligned {
                        let start_block = ls.certified_kv_value_tokens / block_size;
                        let block_count = (aligned - ls.certified_kv_value_tokens) / block_size;
                        if timings.is_some() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value quantize pre-sync: {e}"
                                )
                            })?;
                        }
                        let value_quantize_start = Instant::now();
                        if !ls.certified_kv_gpu_tail_only {
                            let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
                            kernel_ffi::certified_kv::quantize_bf16_values_range(
                                self.ordinal,
                                cache_v_ref,
                                start_block,
                                block_count,
                                block_size,
                                value_group_size,
                                ls.certified_kv_value_i4.as_mut().unwrap(),
                                ls.certified_kv_value_scale.as_mut().unwrap(),
                                ls.certified_kv_value_zero.as_mut().unwrap(),
                                ls.certified_kv_value_error.as_mut().unwrap(),
                                ls.certified_kv_value_norm.as_mut().unwrap(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV decode value quantize: {e}"
                                )
                            })?;
                        } else {
                            if block_count != 1 {
                                anyhow::bail!(
                                    "layer {idx} certified tail value quantize expects single completed block, got {block_count}"
                                );
                            }
                            let tail_v = ls.certified_kv_tail_v.as_ref().ok_or_else(|| {
                                anyhow::anyhow!("layer {idx} certified tail value missing")
                            })?;
                            let quant_v = GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::BF16,
                                &[1, num_kv_heads, block_size, head_dim],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified quant_v alloc: {e}")
                            })?;
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                quant_v.as_ptr() as *mut c_void,
                                tail_v.as_ptr(),
                                tail_v.len_bytes(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified quant_v copy: {e}")
                            })?;
                            let groups = head_dim / value_group_size;
                            let mut tmp_value_i4 = GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::U8,
                                &[num_kv_heads, block_size, head_dim / 2],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified tmp value_i4 alloc: {e}")
                            })?;
                            let mut tmp_value_scale = GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F16,
                                &[num_kv_heads, block_size, groups],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified tmp value_scale alloc: {e}")
                            })?;
                            let mut tmp_value_zero = GpuBuffer::zeros(
                                self.ordinal,
                                ScalarType::F16,
                                &[num_kv_heads, block_size, groups],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified tmp value_zero alloc: {e}")
                            })?;
                            let mut tmp_value_error =
                                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_kv_heads, 1])
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified tmp value_error alloc: {e}"
                                        )
                                    })?;
                            let mut tmp_value_norm =
                                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_kv_heads, 1])
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified tmp value_norm alloc: {e}"
                                        )
                                    })?;
                            kernel_ffi::certified_kv::quantize_bf16_values_range(
                                self.ordinal,
                                &quant_v,
                                0,
                                1,
                                block_size,
                                value_group_size,
                                &mut tmp_value_i4,
                                &mut tmp_value_scale,
                                &mut tmp_value_zero,
                                &mut tmp_value_error,
                                &mut tmp_value_norm,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified tail value quantize: {e}")
                            })?;
                            let dst_block = start_block;
                            let dst_value = ls.certified_kv_value_i4.as_mut().unwrap();
                            let dst_scale = ls.certified_kv_value_scale.as_mut().unwrap();
                            let dst_zero = ls.certified_kv_value_zero.as_mut().unwrap();
                            let dst_error = ls.certified_kv_value_error.as_mut().unwrap();
                            let dst_norm = ls.certified_kv_value_norm.as_mut().unwrap();
                            let dst_value_stride = dst_value.shape()[1] * (head_dim / 2);
                            let src_value_stride = block_size * (head_dim / 2);
                            let value_block_bytes = src_value_stride;
                            let meta_block_bytes =
                                block_size * groups * ScalarType::F16.size_in_bytes();
                            let dst_meta_stride = dst_scale.shape()[1] * groups;
                            let error_bytes = ScalarType::F32.size_in_bytes();
                            for h in 0..num_kv_heads {
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    dst_value.offset_ptr(
                                        (h * dst_value_stride
                                            + dst_block * block_size * (head_dim / 2))
                                            * ScalarType::U8.size_in_bytes(),
                                    ) as *mut c_void,
                                    tmp_value_i4.offset_ptr(h * src_value_stride),
                                    value_block_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified value_i4 block copy h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    dst_scale.offset_ptr(
                                        (h * dst_meta_stride + dst_block * block_size * groups)
                                            * ScalarType::F16.size_in_bytes(),
                                    ) as *mut c_void,
                                    tmp_value_scale.offset_ptr(
                                        h * block_size * groups * ScalarType::F16.size_in_bytes(),
                                    ),
                                    meta_block_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified value_scale block copy h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    dst_zero.offset_ptr(
                                        (h * dst_meta_stride + dst_block * block_size * groups)
                                            * ScalarType::F16.size_in_bytes(),
                                    ) as *mut c_void,
                                    tmp_value_zero.offset_ptr(
                                        h * block_size * groups * ScalarType::F16.size_in_bytes(),
                                    ),
                                    meta_block_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified value_zero block copy h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    dst_error.offset_ptr(
                                        (h * dst_error.shape()[1] + dst_block) * error_bytes,
                                    ) as *mut c_void,
                                    tmp_value_error.offset_ptr(h * error_bytes),
                                    error_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified value_error block copy h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    dst_norm.offset_ptr(
                                        (h * dst_norm.shape()[1] + dst_block) * error_bytes,
                                    ) as *mut c_void,
                                    tmp_value_norm.offset_ptr(h * error_bytes),
                                    error_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified value_norm block copy h={h}: {e}"
                                    )
                                })?;
                            }
                        }
                        if let Some(t) = timings.as_mut() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value quantize synchronize: {e}"
                                )
                            })?;
                            t.certified_kv_value_quantize_ms +=
                                value_quantize_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        ls.certified_kv_value_tokens = aligned;
                    }
                    let num_blocks = aligned / block_size;
                    let gqa_group = num_q_heads / num_kv_heads;
                    let q_scale = 1.0 / (head_dim as f32).sqrt();
                    let key_stride_tokens = key_i8.shape()[1];
                    let key_scale_stride_blocks = key_scale.shape()[1];
                    let value_error_stride_blocks =
                        ls.certified_kv_value_error.as_ref().unwrap().shape()[1];
                    // Paper contract: every BF16 key use must be driven by the
                    // selector/page path. Do not bypass scoring/selection or
                    // silently materialise an all-key mirror in VRAM, even when
                    // the selected set happens to cover the whole prefix.
                    let use_device_selector = cfg.ranking_r <= 1 && num_blocks <= 2048;
                    let mut block_max_host = Vec::new();
                    let mut block_sum_host = Vec::new();
                    let mut query_f32_all = Vec::new();
                    let block_score_shape = [num_q_heads, num_blocks];
                    if scratch
                        .certified_block_max
                        .as_ref()
                        .map(|buf| buf.shape() != block_score_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_block_max = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &block_score_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV block_max alloc: {e}")
                                })?,
                        );
                    }
                    if scratch
                        .certified_block_sum
                        .as_ref()
                        .map(|buf| buf.shape() != block_score_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_block_sum = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &block_score_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV block_sum alloc: {e}")
                                })?,
                        );
                    }
                    let cert_score_start = Instant::now();
                    kernel_ffi::certified_kv::score_blocks_int8(
                        self.ordinal,
                        attn_q,
                        key_i8,
                        key_scale,
                        key_zero,
                        block_size,
                        gqa_group,
                        q_scale,
                        scratch.certified_block_max.as_mut().unwrap(),
                        scratch.certified_block_sum.as_mut().unwrap(),
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} certified KV score blocks: {e}"))?;
                    if !use_device_selector {
                        block_max_host = decode_f32_le(
                            &scratch
                                .certified_block_max
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV block_max D2H: {e}")
                                })?,
                        );
                        block_sum_host = decode_f32_le(
                            &scratch
                                .certified_block_sum
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV block_sum D2H: {e}")
                                })?,
                        );
                        query_f32_all =
                            decode_bf16_le_host(&attn_q.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV query D2H: {e}")
                            })?);
                    }
                    if let Some(t) = timings.as_mut() {
                        t.certified_kv_score_ms +=
                            cert_score_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    let need_meta_refresh = !use_device_selector
                        && (ls.certified_kv_host_meta_blocks != num_blocks
                            || ls.certified_kv_host_meta_key_stride_tokens != key_stride_tokens
                            || ls.certified_kv_host_meta_key_scale_stride_blocks
                                != key_scale_stride_blocks
                            || ls.certified_kv_host_meta_value_error_stride_blocks
                                != value_error_stride_blocks);
                    if need_meta_refresh {
                        ls.certified_kv_host_value_error_cache = decode_f32_le(
                            &ls.certified_kv_value_error
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV value_error D2H: {e}")
                                })?,
                        );
                        ls.certified_kv_host_value_norm_cache = decode_f32_le(
                            &ls.certified_kv_value_norm
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!("layer {idx} certified KV value_norm D2H: {e}")
                                })?,
                        );
                        ls.certified_kv_host_key_scale_cache =
                            decode_f32_le(&key_scale.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV key_scale D2H: {e}")
                            })?);
                        ls.certified_kv_host_key_scale_channel_max_cache =
                            vec![0.0; num_kv_heads * head_dim];
                        for kvh in 0..num_kv_heads {
                            for block in 0..num_blocks {
                                let scale_base = (kvh * key_scale_stride_blocks + block) * head_dim;
                                let max_base = kvh * head_dim;
                                for dim in 0..head_dim {
                                    let scale =
                                        ls.certified_kv_host_key_scale_cache[scale_base + dim];
                                    let max_slot = &mut ls
                                        .certified_kv_host_key_scale_channel_max_cache
                                        [max_base + dim];
                                    *max_slot = (*max_slot).max(scale);
                                }
                            }
                        }
                        ls.certified_kv_host_meta_blocks = num_blocks;
                        ls.certified_kv_host_meta_key_stride_tokens = key_stride_tokens;
                        ls.certified_kv_host_meta_key_scale_stride_blocks = key_scale_stride_blocks;
                        ls.certified_kv_host_meta_value_error_stride_blocks =
                            value_error_stride_blocks;
                    }
                    let mut promote_index_host = vec![u32::MAX; num_q_heads * num_blocks];
                    let mut value_promote_index_host = vec![u32::MAX; num_kv_heads * num_blocks];
                    let mut selected_by_head = Vec::with_capacity(num_q_heads);
                    let mut max_promoted_blocks = 1usize;
                    let mut value_promote_flags = vec![false; num_kv_heads * num_blocks];
                    let key_cache_capacity = cfg.key_cache_blocks.max(1).min(num_blocks.max(1));
                    // The paper requires promoted BF16 keys to go through the
                    // selector/page-cache path, not a silent all-key mirror.
                    // Misses page BF16 blocks from Tier-2 into bounded scratch
                    // slots; hits reuse those slots.
                    let use_key_page_cache = cfg.key_cache_blocks > 0;
                    // Recoverable certificate failures are handled by the Rung-3
                    // per-head dense path. Keep Rung-4 available for future
                    // unrecoverable layer-wide failures without making common
                    // tail-bound misses recompute every query head.
                    let force_dense_layer_fallback = false;
                    let mut dense_fallback_qhead_flags = vec![false; num_q_heads];
                    let tier2_key = ls.certified_kv_host_k.as_ref().unwrap().as_bytes();
                    let mut delta_blocks_by_head = Vec::with_capacity(num_q_heads);
                    let mut e_key_by_head = Vec::with_capacity(num_q_heads);
                    let mut delta_tail_by_head = Vec::with_capacity(num_q_heads);
                    let mut vmax_by_head = Vec::with_capacity(num_q_heads);
                    let mut true_tail_bound_by_head = Vec::with_capacity(num_q_heads);
                    let collect_cert_host_telemetry =
                        std::env::var_os("SUPERSONIC_CERTIFIED_HOST_TELEMETRY").is_some();
                    let trace_cert_phases =
                        std::env::var_os("SUPERSONIC_CERTIFIED_TRACE_PHASES").is_some();
                    let cert_selector_start = Instant::now();
                    if use_device_selector {
                        let max_device_promoted_blocks = cfg
                            .k_max
                            .max(cfg.k_min)
                            .min(num_blocks)
                            .saturating_mul(cfg.rung1_multiplier.ceil().max(1.0) as usize)
                            .clamp(1, num_blocks);
                        max_promoted_blocks = max_device_promoted_blocks;
                        let promote_shape = [num_q_heads, num_blocks];
                        let value_promote_shape = [num_kv_heads, num_blocks];
                        let selected_shape = [num_q_heads, max_device_promoted_blocks];
                        let head_shape = [num_q_heads];
                        if scratch
                            .certified_promote_index
                            .as_ref()
                            .map(|buf| buf.shape() != promote_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_promote_index = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::U32, &promote_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV promote index alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        if scratch
                            .certified_value_promote_index
                            .as_ref()
                            .map(|buf| buf.shape() != value_promote_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_value_promote_index = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::U32,
                                    &value_promote_shape,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV value promote index alloc: {e}"
                                    )
                                })?,
                            );
                        }
                        if scratch
                            .certified_selected_blocks
                            .as_ref()
                            .map(|buf| buf.shape() != selected_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_selected_blocks = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::U32, &selected_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV selected block alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        if scratch
                            .certified_selected_counts
                            .as_ref()
                            .map(|buf| buf.shape() != head_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_selected_counts = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::U32, &head_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV selected count alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        if scratch
                            .certified_ranking_fallback_head_flags
                            .as_ref()
                            .map(|buf| buf.shape() != head_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_ranking_fallback_head_flags = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::U32, &head_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV fallback flag alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        let score_shape = [num_q_heads, num_blocks];
                        if scratch
                            .certified_delta_blocks
                            .as_ref()
                            .map(|buf| buf.shape() != score_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_delta_blocks = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &score_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV delta block alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        let scale_norm_shape = [num_kv_heads, num_blocks];
                        if scratch
                            .certified_key_scale_norm
                            .as_ref()
                            .map(|buf| buf.shape() != scale_norm_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_key_scale_norm = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::F32, &scale_norm_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV key-scale norm alloc: {e}"
                                        )
                                    })?,
                            );
                        }
                        let refresh_key_scale_norm =
                            ls.certified_kv_device_meta_key_scale_norm_blocks != num_blocks
                                || ls.certified_kv_device_meta_key_scale_stride_blocks
                                    != key_scale_stride_blocks;
                        if refresh_key_scale_norm {
                            kernel_ffi::certified_kv::key_scale_norms(
                                self.ordinal,
                                key_scale,
                                scratch.certified_key_scale_norm.as_mut().unwrap(),
                                num_blocks,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV key-scale norm prepass: {e}"
                                )
                            })?;
                            ls.certified_kv_device_meta_key_scale_norm_blocks = num_blocks;
                            ls.certified_kv_device_meta_key_scale_stride_blocks =
                                key_scale_stride_blocks;
                        }
                        for (slot, name) in [
                            (&mut scratch.certified_selector_e_key, "e-key"),
                            (&mut scratch.certified_selector_delta_tail, "delta-tail"),
                            (&mut scratch.certified_selector_vmax, "vmax"),
                            (&mut scratch.certified_selector_true_tail, "true-tail"),
                        ] {
                            if slot
                                .as_ref()
                                .map(|buf| buf.shape() != head_shape)
                                .unwrap_or(true)
                            {
                                *slot = Some(
                                    GpuBuffer::zeros(self.ordinal, ScalarType::F32, &head_shape)
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "layer {idx} certified KV selector {name} alloc: {e}"
                                            )
                                        })?,
                                );
                            }
                        }
                        kernel_ffi::certified_kv::select_blocks_device(
                            self.ordinal,
                            attn_q,
                            scratch.certified_key_scale_norm.as_ref().unwrap(),
                            scratch.certified_block_max.as_ref().unwrap(),
                            scratch.certified_block_sum.as_ref().unwrap(),
                            ls.certified_kv_value_norm.as_ref().unwrap(),
                            scratch.certified_promote_index.as_mut().unwrap(),
                            scratch.certified_value_promote_index.as_mut().unwrap(),
                            scratch.certified_selected_blocks.as_mut().unwrap(),
                            scratch.certified_selected_counts.as_mut().unwrap(),
                            scratch
                                .certified_ranking_fallback_head_flags
                                .as_mut()
                                .unwrap(),
                            scratch.certified_delta_blocks.as_mut().unwrap(),
                            scratch.certified_selector_e_key.as_mut().unwrap(),
                            scratch.certified_selector_delta_tail.as_mut().unwrap(),
                            scratch.certified_selector_vmax.as_mut().unwrap(),
                            scratch.certified_selector_true_tail.as_mut().unwrap(),
                            gqa_group,
                            cfg.k_min,
                            cfg.k_max,
                            max_device_promoted_blocks,
                            q_scale,
                            cfg.tau_cov,
                            cfg.rung1_threshold,
                            cfg.rung1_multiplier,
                            cfg.delta_guard_factor,
                            cfg.score_exploration_rate,
                            cfg.require_certified_tail_bound,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV device selector: {e}")
                        })?;
                        if trace_cert_phases {
                            eprintln!("cert-kv layer {idx}: device selector done");
                        }
                        selected_by_head.resize_with(num_q_heads, Vec::new);
                        if collect_cert_host_telemetry {
                            dense_fallback_qhead_flags = decode_u32_le(
                                &scratch
                                    .certified_ranking_fallback_head_flags
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV fallback flags D2H: {e}"
                                        )
                                    })?,
                            )
                            .into_iter()
                            .map(|flag| flag != 0)
                            .collect();
                            if let Some(t) = timings.as_mut() {
                                t.certified_kv_uncertified_tail_heads += dense_fallback_qhead_flags
                                    .iter()
                                    .filter(|&&flag| flag)
                                    .count();
                            }
                            e_key_by_head = decode_f32_le(
                                &scratch
                                    .certified_selector_e_key
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!("layer {idx} certified KV e-key D2H: {e}")
                                    })?,
                            );
                            delta_tail_by_head = decode_f32_le(
                                &scratch
                                    .certified_selector_delta_tail
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV delta-tail D2H: {e}"
                                        )
                                    })?,
                            );
                            vmax_by_head = decode_f32_le(
                                &scratch
                                    .certified_selector_vmax
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!("layer {idx} certified KV vmax D2H: {e}")
                                    })?,
                            );
                            true_tail_bound_by_head = decode_f32_le(
                                &scratch
                                    .certified_selector_true_tail
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV true-tail D2H: {e}"
                                        )
                                    })?,
                            );
                        } else {
                            e_key_by_head.resize(num_q_heads, 0.0);
                            delta_tail_by_head.resize(num_q_heads, 0.0);
                            vmax_by_head.resize(num_q_heads, 0.0);
                            true_tail_bound_by_head.resize(num_q_heads, 0.0);
                        }
                        if use_key_page_cache {
                            let selected_counts = decode_u32_le(
                                &scratch
                                    .certified_selected_counts
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV selected counts D2H: {e}"
                                        )
                                    })?,
                            );
                            let selected_blocks = decode_u32_le(
                                &scratch
                                    .certified_selected_blocks
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV selected blocks D2H: {e}"
                                        )
                                    })?,
                            );
                            selected_by_head.clear();
                            selected_by_head.reserve(num_q_heads);
                            for qh in 0..num_q_heads {
                                let count = selected_counts
                                    .get(qh)
                                    .copied()
                                    .unwrap_or(0)
                                    .min(max_device_promoted_blocks as u32)
                                    as usize;
                                let base = qh * max_device_promoted_blocks;
                                selected_by_head.push(
                                    selected_blocks[base..base + count]
                                        .iter()
                                        .map(|&block| block as usize)
                                        .filter(|&block| block < num_blocks)
                                        .collect(),
                                );
                            }
                        }
                    } else {
                        for qh in 0..num_q_heads {
                            let score_start = qh * num_blocks;
                            let score_end = score_start + num_blocks;
                            let kvh = qh / gqa_group;
                            let value_error_host = &ls.certified_kv_host_value_error_cache;
                            let value_norm_host = &ls.certified_kv_host_value_norm_cache;
                            let key_scale_channel_max_host =
                                &ls.certified_kv_host_key_scale_channel_max_cache;
                            let delta_global = certified_kv_score_delta_from_channel_max(
                                &query_f32_all,
                                key_scale_channel_max_host,
                                qh,
                                kvh,
                                head_dim,
                                q_scale,
                            );
                            let delta_blocks = vec![delta_global; num_blocks];
                            let (_, _selector_true_tail_bound, _, probs, mut selected_indices) =
                                certified_kv_select_block_indices_from_scores(
                                    &block_max_host[score_start..score_end],
                                    &block_sum_host[score_start..score_end],
                                    Some(&delta_blocks),
                                    cfg.tau_cov,
                                    cfg.k_min,
                                    cfg.k_max,
                                    cfg.rung1_threshold,
                                    cfg.rung1_multiplier,
                                    cfg.delta_guard_factor,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} q_head {qh} certified KV selector: {e}"
                                    )
                                })?;
                            for (block, prob) in probs.iter().enumerate() {
                                let value_contribution = prob
                                    * value_error_host[kvh * value_error_stride_blocks + block];
                                if value_contribution > cfg.v_tol {
                                    value_promote_flags[kvh * num_blocks + block] = true;
                                }
                            }
                            if cfg.ranking_r > 0
                                && CERTIFIED_KV_MAX_RANKING_BOUNDARY_REPAIR_BLOCKS > 0
                            {
                                let mut selected_set: std::collections::HashSet<usize> =
                                    selected_indices.iter().copied().collect();
                                let mut repaired_blocks = 0usize;
                                loop {
                                    let fp16_selected_log_masses =
                                        certified_kv_selected_block_fp16_log_masses_from_tier2(
                                            &query_f32_all,
                                            tier2_key,
                                            qh,
                                            kvh,
                                            &selected_indices,
                                            cap,
                                            block_size,
                                            head_dim,
                                            q_scale,
                                        );
                                    let violators = certified_kv_ranking_boundary_violators(
                                        &block_max_host[score_start..score_end],
                                        &block_sum_host[score_start..score_end],
                                        &delta_blocks,
                                        &fp16_selected_log_masses,
                                        &selected_indices,
                                        cfg.ranking_r,
                                    );
                                    if violators.is_empty()
                                        || selected_indices.len() >= num_blocks
                                        || repaired_blocks
                                            >= CERTIFIED_KV_MAX_RANKING_BOUNDARY_REPAIR_BLOCKS
                                    {
                                        break;
                                    }
                                    for block in violators {
                                        if repaired_blocks
                                            >= CERTIFIED_KV_MAX_RANKING_BOUNDARY_REPAIR_BLOCKS
                                        {
                                            break;
                                        }
                                        if selected_set.insert(block) {
                                            selected_indices.push(block);
                                            repaired_blocks += 1;
                                        }
                                    }
                                }
                            }
                            let mut selected_flags = vec![false; num_blocks];
                            for &block in &selected_indices {
                                selected_flags[block] = true;
                            }
                            if cfg.score_exploration_rate > 0.0
                                && selected_indices.len() < num_blocks
                            {
                                let period =
                                    (1.0 / cfg.score_exploration_rate).ceil().max(1.0) as usize;
                                for block in 0..num_blocks {
                                    if selected_flags[block] {
                                        continue;
                                    }
                                    if ((block.wrapping_mul(1_103_515_245) ^ qh) % period) == 0 {
                                        selected_flags[block] = true;
                                        selected_indices.push(block);
                                    }
                                }
                            }
                            let tail_mass = (0..num_blocks)
                                .filter(|&block| !selected_flags[block])
                                .map(|block| probs[block])
                                .sum::<f32>()
                                .max(0.0);
                            let delta_tail = (0..num_blocks)
                                .filter(|&block| !selected_flags[block])
                                .map(|block| delta_blocks[block])
                                .fold(0.0f32, f32::max);
                            let true_tail_bound =
                                (cfg.delta_guard_factor * delta_tail).exp() * tail_mass;
                            if cfg.require_certified_tail_bound
                                && true_tail_bound > cfg.rung1_threshold
                            {
                                dense_fallback_qhead_flags[qh] = true;
                                if let Some(t) = timings.as_mut() {
                                    t.certified_kv_uncertified_tail_heads += 1;
                                }
                            }
                            let vmax = certified_kv_vmax_for_head(
                                &value_norm_host,
                                kvh,
                                num_blocks,
                                value_error_stride_blocks,
                                &selected_flags,
                            );
                            let e_key =
                                certified_kv_key_error_bound(vmax, true_tail_bound, delta_tail);
                            e_key_by_head.push(e_key);
                            delta_tail_by_head.push(delta_tail);
                            vmax_by_head.push(vmax);
                            true_tail_bound_by_head.push(true_tail_bound);
                            delta_blocks_by_head.push(delta_blocks);
                            max_promoted_blocks = max_promoted_blocks.max(selected_indices.len());
                            for (slot, &block) in selected_indices.iter().enumerate() {
                                promote_index_host[score_start + block] = slot as u32;
                            }
                            selected_by_head.push(selected_indices);
                        }
                    }
                    let mut selected_value_blocks_by_kvh = Vec::with_capacity(num_kv_heads);
                    let mut max_promoted_value_blocks = 1usize;
                    for kvh in 0..num_kv_heads {
                        let mut selected_blocks = Vec::new();
                        for block in 0..num_blocks {
                            if value_promote_flags[kvh * num_blocks + block] {
                                value_promote_index_host[kvh * num_blocks + block] =
                                    selected_blocks.len() as u32;
                                selected_blocks.push(block);
                            }
                        }
                        max_promoted_value_blocks =
                            max_promoted_value_blocks.max(selected_blocks.len());
                        selected_value_blocks_by_kvh.push(selected_blocks);
                    }
                    let has_value_promotions = selected_value_blocks_by_kvh
                        .iter()
                        .any(|blocks| !blocks.is_empty());
                    let value_cache_capacity = cfg.value_cache_blocks.max(1);
                    let initial_value_cache_fits_step = selected_value_blocks_by_kvh
                        .iter()
                        .all(|blocks| blocks.len() <= value_cache_capacity);
                    let use_initial_value_cache = cfg.value_cache_blocks > 0
                        && has_value_promotions
                        && initial_value_cache_fits_step;
                    let mut value_gather_index_host = value_promote_index_host.clone();
                    let mut value_cache_hits = 0usize;
                    let mut value_cache_misses = 0usize;
                    let mut value_cache_overflows = 0usize;
                    let value_block_bytes = block_size * head_dim * elem_bytes;
                    let selected_value_block_count = selected_value_blocks_by_kvh
                        .iter()
                        .map(Vec::len)
                        .sum::<usize>();
                    if use_initial_value_cache {
                        let tag_len = num_kv_heads * value_cache_capacity;
                        let cache_shape =
                            [num_kv_heads, value_cache_capacity, block_size, head_dim];
                        let reset_cache = ls
                            .certified_kv_promoted_value_cache
                            .as_ref()
                            .map(|buf| buf.shape() != cache_shape)
                            .unwrap_or(true)
                            || ls.certified_kv_promoted_value_cache_capacity
                                != value_cache_capacity
                            || ls.certified_kv_promoted_value_cache_tags.len() != tag_len;
                        if reset_cache {
                            ls.certified_kv_promoted_value_cache = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &cache_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV promoted value cache alloc: {e}"
                                        )
                                    })?,
                            );
                            ls.certified_kv_promoted_value_cache_capacity = value_cache_capacity;
                            ls.certified_kv_promoted_value_cache_tags = vec![usize::MAX; tag_len];
                            ls.certified_kv_promoted_value_cache_lru = vec![0; tag_len];
                            ls.certified_kv_promoted_value_cache_tick = 0;
                        }
                        value_promote_index_host.fill(u32::MAX);
                        value_gather_index_host.fill(u32::MAX);
                        for kvh in 0..num_kv_heads {
                            for &block in &selected_value_blocks_by_kvh[kvh] {
                                let base = kvh * value_cache_capacity;
                                ls.certified_kv_promoted_value_cache_tick =
                                    ls.certified_kv_promoted_value_cache_tick.wrapping_add(1);
                                let tick = ls.certified_kv_promoted_value_cache_tick;
                                let mut slot = (0..value_cache_capacity).find(|&s| {
                                    ls.certified_kv_promoted_value_cache_tags[base + s] == block
                                });
                                if let Some(s) = slot {
                                    value_cache_hits += 1;
                                    ls.certified_kv_promoted_value_cache_lru[base + s] = tick;
                                } else {
                                    value_cache_misses += 1;
                                    slot = (0..value_cache_capacity).find(|&s| {
                                        ls.certified_kv_promoted_value_cache_tags[base + s]
                                            == usize::MAX
                                    });
                                    let s = if let Some(s) = slot {
                                        s
                                    } else {
                                        value_cache_overflows += 1;
                                        (0..value_cache_capacity)
                                            .min_by_key(|&s| {
                                                ls.certified_kv_promoted_value_cache_lru[base + s]
                                            })
                                            .unwrap_or(0)
                                    };
                                    ls.certified_kv_promoted_value_cache_tags[base + s] = block;
                                    ls.certified_kv_promoted_value_cache_lru[base + s] = tick;
                                    value_gather_index_host[kvh * num_blocks + block] = s as u32;
                                    slot = Some(s);
                                }
                                value_promote_index_host[kvh * num_blocks + block] =
                                    slot.unwrap() as u32;
                            }
                        }
                        max_promoted_value_blocks = value_cache_capacity;
                    }
                    let mut key_gather_index_host = promote_index_host.clone();
                    let mut key_cache_hits = 0usize;
                    let mut key_cache_misses = 0usize;
                    let mut key_cache_overflows = 0usize;
                    let key_block_bytes = block_size * head_dim * elem_bytes;
                    let selected_key_block_count = if use_device_selector {
                        // Keep telemetry nonzero without forcing selector D2H on
                        // the fast path. This is an upper bound when the GPU
                        // selector chooses fewer than its slot budget.
                        num_q_heads * max_promoted_blocks
                    } else {
                        selected_by_head.iter().map(Vec::len).sum::<usize>()
                    };
                    if use_key_page_cache {
                        let tag_len = num_q_heads * key_cache_capacity;
                        let cache_shape = [num_q_heads, key_cache_capacity, block_size, head_dim];
                        let reset_cache = ls
                            .certified_kv_promoted_key_cache
                            .as_ref()
                            .map(|buf| buf.shape() != cache_shape)
                            .unwrap_or(true)
                            || ls
                                .certified_kv_promoted_key_cache_tags_gpu
                                .as_ref()
                                .map(|buf| buf.shape() != [num_q_heads, key_cache_capacity])
                                .unwrap_or(true)
                            || ls
                                .certified_kv_promoted_key_cache_lru_gpu
                                .as_ref()
                                .map(|buf| buf.shape() != [num_q_heads, key_cache_capacity])
                                .unwrap_or(true)
                            || ls.certified_kv_promoted_key_cache_capacity != key_cache_capacity
                            || ls.certified_kv_promoted_key_cache_tags.len() != tag_len;
                        if reset_cache {
                            ls.certified_kv_promoted_key_cache = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &cache_shape)
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV promoted key cache alloc: {e}"
                                        )
                                    })?,
                            );
                            ls.certified_kv_promoted_key_cache_tags_gpu = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::U32,
                                    &[num_q_heads, key_cache_capacity],
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV promoted key cache tags alloc: {e}"
                                    )
                                })?,
                            );
                            ls.certified_kv_promoted_key_cache_lru_gpu = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::U32,
                                    &[num_q_heads, key_cache_capacity],
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV promoted key cache lru alloc: {e}"
                                    )
                                })?,
                            );
                            kernel_ffi::certified_kv::init_key_page_cache(
                                self.ordinal,
                                ls.certified_kv_promoted_key_cache_tags_gpu
                                    .as_mut()
                                    .unwrap(),
                                ls.certified_kv_promoted_key_cache_lru_gpu.as_mut().unwrap(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV promoted key cache init: {e}"
                                )
                            })?;
                            ls.certified_kv_promoted_key_cache_capacity = key_cache_capacity;
                            ls.certified_kv_promoted_key_cache_tags = vec![usize::MAX; tag_len];
                            ls.certified_kv_promoted_key_cache_lru = vec![0; tag_len];
                            ls.certified_kv_promoted_key_cache_tick = 0;
                        }
                        if scratch
                            .certified_key_gather_index
                            .as_ref()
                            .map(|buf| buf.shape() != [num_q_heads, num_blocks])
                            .unwrap_or(true)
                        {
                            scratch.certified_key_gather_index = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::U32,
                                    &[num_q_heads, num_blocks],
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV key gather index alloc: {e}"
                                    )
                                })?,
                            );
                        }
                        if scratch
                            .certified_key_cache_counters
                            .as_ref()
                            .map(|buf| buf.shape() != [3])
                            .unwrap_or(true)
                        {
                            scratch.certified_key_cache_counters = Some(
                                GpuBuffer::zeros(self.ordinal, ScalarType::U32, &[3]).map_err(
                                    |e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV key cache counters alloc: {e}"
                                        )
                                    },
                                )?,
                            );
                        }
                        if use_device_selector {
                            ls.certified_kv_promoted_key_cache_tick = ls
                                .certified_kv_promoted_key_cache_tick
                                .wrapping_add(max_promoted_blocks as u64 + 1);
                            kernel_ffi::certified_kv::resolve_key_page_cache(
                                self.ordinal,
                                scratch.certified_selected_blocks.as_ref().unwrap(),
                                scratch.certified_selected_counts.as_ref().unwrap(),
                                ls.certified_kv_promoted_key_cache_tags_gpu
                                    .as_mut()
                                    .unwrap(),
                                ls.certified_kv_promoted_key_cache_lru_gpu.as_mut().unwrap(),
                                scratch.certified_promote_index.as_mut().unwrap(),
                                scratch.certified_key_gather_index.as_mut().unwrap(),
                                scratch.certified_key_cache_counters.as_mut().unwrap(),
                                ls.certified_kv_promoted_key_cache_tick as u32,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV key page-cache resolve: {e}"
                                )
                            })?;
                            let collect_key_cache_counters =
                                collect_cert_host_telemetry || trace_cert_phases;
                            if collect_key_cache_counters {
                                let counter_bytes = scratch
                                    .certified_key_cache_counters
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV key cache counters D2H: {e}"
                                        )
                                    })?;
                                let counters = decode_u32_le(&counter_bytes);
                                key_cache_hits = counters.first().copied().unwrap_or(0) as usize;
                                key_cache_misses = counters.get(1).copied().unwrap_or(0) as usize;
                                key_cache_overflows =
                                    counters.get(2).copied().unwrap_or(0) as usize;
                            }
                        } else {
                            promote_index_host.fill(u32::MAX);
                            key_gather_index_host.fill(u32::MAX);
                            for qh in 0..num_q_heads {
                                for &block in &selected_by_head[qh] {
                                    let base = qh * key_cache_capacity;
                                    ls.certified_kv_promoted_key_cache_tick =
                                        ls.certified_kv_promoted_key_cache_tick.wrapping_add(1);
                                    let tick = ls.certified_kv_promoted_key_cache_tick;
                                    let mut slot = (0..key_cache_capacity).find(|&s| {
                                        ls.certified_kv_promoted_key_cache_tags[base + s] == block
                                    });
                                    if let Some(s) = slot {
                                        key_cache_hits += 1;
                                        ls.certified_kv_promoted_key_cache_lru[base + s] = tick;
                                    } else {
                                        key_cache_misses += 1;
                                        slot = (0..key_cache_capacity).find(|&s| {
                                            ls.certified_kv_promoted_key_cache_tags[base + s]
                                                == usize::MAX
                                        });
                                        let s = if let Some(s) = slot {
                                            s
                                        } else {
                                            key_cache_overflows += 1;
                                            (0..key_cache_capacity)
                                                .min_by_key(|&s| {
                                                    ls.certified_kv_promoted_key_cache_lru[base + s]
                                                })
                                                .unwrap_or(0)
                                        };
                                        ls.certified_kv_promoted_key_cache_tags[base + s] = block;
                                        ls.certified_kv_promoted_key_cache_lru[base + s] = tick;
                                        key_gather_index_host[qh * num_blocks + block] = s as u32;
                                        slot = Some(s);
                                    }
                                    promote_index_host[qh * num_blocks + block] =
                                        slot.unwrap() as u32;
                                }
                            }
                        }
                        max_promoted_blocks = key_cache_capacity;
                    }
                    if trace_cert_phases {
                        eprintln!(
                            "cert-kv layer {idx}: cache resolved key_hits={} key_misses={} key_overflows={} max_key_slots={}",
                            key_cache_hits,
                            key_cache_misses,
                            key_cache_overflows,
                            max_promoted_blocks
                        );
                    }
                    if let Some(t) = timings.as_mut() {
                        t.certified_kv_selector_ms +=
                            cert_selector_start.elapsed().as_secs_f64() * 1000.0;
                        t.certified_kv_promoted_key_cache_hits += key_cache_hits;
                        t.certified_kv_promoted_key_cache_misses += key_cache_misses;
                        t.certified_kv_promoted_key_cache_overflows += key_cache_overflows;
                        t.certified_kv_promoted_key_h2d_bytes += if use_key_page_cache {
                            if collect_cert_host_telemetry || trace_cert_phases {
                                key_cache_misses * key_block_bytes
                            } else {
                                selected_key_block_count * key_block_bytes
                            }
                        } else {
                            selected_key_block_count * key_block_bytes
                        };
                        t.certified_kv_promoted_value_cache_hits += value_cache_hits;
                        t.certified_kv_promoted_value_cache_misses += value_cache_misses;
                        t.certified_kv_promoted_value_cache_overflows += value_cache_overflows;
                        t.certified_kv_promoted_value_h2d_bytes += if use_initial_value_cache {
                            value_cache_misses * value_block_bytes
                        } else {
                            selected_value_block_count * value_block_bytes
                        };
                    }
                    let promoted_key_shape =
                        [num_q_heads, max_promoted_blocks, block_size, head_dim];
                    let promoted_value_shape = [
                        num_kv_heads,
                        max_promoted_value_blocks,
                        block_size,
                        head_dim,
                    ];
                    let promote_shape = [num_q_heads, num_blocks];
                    if scratch
                        .certified_promote_index
                        .as_ref()
                        .map(|buf| buf.shape() != promote_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_promote_index = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &promote_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV promote index alloc: {e}"
                                    )
                                })?,
                        );
                    }
                    let value_promote_shape = [num_kv_heads, num_blocks];
                    if scratch
                        .certified_value_promote_index
                        .as_ref()
                        .map(|buf| buf.shape() != value_promote_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_value_promote_index = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &value_promote_shape)
                                .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value promote index alloc: {e}"
                                )
                            })?,
                        );
                    }
                    if scratch
                        .certified_value_promotion_counters
                        .as_ref()
                        .map(|buf| buf.shape() != [num_kv_heads])
                        .unwrap_or(true)
                    {
                        scratch.certified_value_promotion_counters = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &[num_kv_heads])
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV value promotion counters alloc: {e}"
                                    )
                                })?,
                        );
                    }
                    if scratch
                        .certified_value_promotion_any
                        .as_ref()
                        .map(|buf| buf.shape() != [1])
                        .unwrap_or(true)
                    {
                        scratch.certified_value_promotion_any = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &[1]).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value promotion flag alloc: {e}"
                                )
                            })?,
                        );
                    }
                    if scratch
                        .certified_value_promotion_head_flags
                        .as_ref()
                        .map(|buf| buf.shape() != [num_q_heads])
                        .unwrap_or(true)
                    {
                        scratch.certified_value_promotion_head_flags = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &[num_q_heads])
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV value promotion head flags alloc: {e}"
                                    )
                                })?,
                        );
                    }
                    if scratch
                        .certified_value_e_val_by_head
                        .as_ref()
                        .map(|buf| buf.shape() != [num_q_heads])
                        .unwrap_or(true)
                    {
                        scratch.certified_value_e_val_by_head = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads])
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV e_val-by-head alloc: {e}"
                                    )
                                })?,
                        );
                    }
                    if !use_key_page_cache
                        && scratch
                            .certified_promoted_key_bf16
                            .as_ref()
                            .map(|buf| buf.shape() != promoted_key_shape)
                            .unwrap_or(true)
                    {
                        scratch.certified_promoted_key_bf16 = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &promoted_key_shape)
                                .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV promoted key alloc: {e}")
                            })?,
                        );
                    }
                    if !use_initial_value_cache {
                        if scratch
                            .certified_promoted_value_bf16
                            .as_ref()
                            .map(|buf| buf.shape() != promoted_value_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_promoted_value_bf16 = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::BF16,
                                    &promoted_value_shape,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV promoted value alloc: {e}"
                                    )
                                })?,
                            );
                        }
                    }
                    let cert_gather_start = Instant::now();
                    if !use_device_selector {
                        let promote_index_upload = if use_key_page_cache {
                            &key_gather_index_host
                        } else {
                            &promote_index_host
                        };
                        let promote_index_bytes = unsafe {
                            std::slice::from_raw_parts(
                                promote_index_upload.as_ptr() as *const u8,
                                promote_index_upload.len() * std::mem::size_of::<u32>(),
                            )
                        };
                        gpu_hal::copy_h2d(
                            self.ordinal,
                            scratch
                                .certified_promote_index
                                .as_mut()
                                .unwrap()
                                .as_mut_ptr(),
                            promote_index_bytes.as_ptr() as *const c_void,
                            promote_index_bytes.len(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV promote index H2D: {e}")
                        })?;
                        let value_promote_upload = if use_initial_value_cache {
                            &value_gather_index_host
                        } else {
                            &value_promote_index_host
                        };
                        if !use_device_selector || use_initial_value_cache {
                            let value_promote_index_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    value_promote_upload.as_ptr() as *const u8,
                                    value_promote_upload.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_value_promote_index
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                value_promote_index_bytes.as_ptr() as *const c_void,
                                value_promote_index_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV value promote index H2D: {e}"
                                )
                            })?;
                        }
                    }
                    let tier2_value_device_ptr = ls
                        .certified_kv_host_v
                        .as_ref()
                        .unwrap()
                        .device_ptr()
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV Tier-2 value map: {e}")
                        })?;
                    {
                        let tier2_key_device_ptr = ls
                            .certified_kv_host_k
                            .as_ref()
                            .unwrap()
                            .device_ptr()
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV Tier-2 key map: {e}")
                            })?;
                        if use_initial_value_cache {
                            let key_gather_index_ref = if use_key_page_cache && use_device_selector
                            {
                                scratch.certified_key_gather_index.as_ref().unwrap()
                            } else {
                                scratch.certified_promote_index.as_ref().unwrap()
                            };
                            kernel_ffi::certified_kv::gather_promoted_bf16_from_tier2(
                                self.ordinal,
                                tier2_key_device_ptr,
                                tier2_value_device_ptr,
                                key_gather_index_ref,
                                scratch.certified_value_promote_index.as_ref().unwrap(),
                                if use_key_page_cache {
                                    ls.certified_kv_promoted_key_cache.as_mut().unwrap()
                                } else {
                                    scratch.certified_promoted_key_bf16.as_mut().unwrap()
                                },
                                ls.certified_kv_promoted_value_cache.as_mut().unwrap(),
                                block_size,
                                cap,
                                max_promoted_blocks,
                                max_promoted_value_blocks,
                                gqa_group,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV promoted BF16 gather through value cache: {e}"
                                )
                            })?;
                            let value_promote_index_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    value_promote_index_host.as_ptr() as *const u8,
                                    value_promote_index_host.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_value_promote_index
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                value_promote_index_bytes.as_ptr() as *const c_void,
                                value_promote_index_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV cached value promote index H2D: {e}"
                                )
                            })?;
                        } else {
                            let key_gather_index_ref = if use_key_page_cache && use_device_selector
                            {
                                scratch.certified_key_gather_index.as_ref().unwrap()
                            } else {
                                scratch.certified_promote_index.as_ref().unwrap()
                            };
                            kernel_ffi::certified_kv::gather_promoted_bf16_from_tier2(
                                self.ordinal,
                                tier2_key_device_ptr,
                                tier2_value_device_ptr,
                                key_gather_index_ref,
                                scratch.certified_value_promote_index.as_ref().unwrap(),
                                if use_key_page_cache {
                                    ls.certified_kv_promoted_key_cache.as_mut().unwrap()
                                } else {
                                    scratch.certified_promoted_key_bf16.as_mut().unwrap()
                                },
                                scratch.certified_promoted_value_bf16.as_mut().unwrap(),
                                block_size,
                                cap,
                                max_promoted_blocks,
                                max_promoted_value_blocks,
                                gqa_group,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV promoted BF16 gather: {e}"
                                )
                            })?;
                        }
                        if use_key_page_cache && !use_device_selector {
                            let promote_index_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    promote_index_host.as_ptr() as *const u8,
                                    promote_index_host.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_promote_index
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                promote_index_bytes.as_ptr() as *const c_void,
                                promote_index_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV cached key promote index H2D: {e}"
                                )
                            })?;
                        }
                    }
                    if trace_cert_phases {
                        eprintln!("cert-kv layer {idx}: gather done");
                    }
                    if timings.is_some() {
                        gpu_hal::sync(self.ordinal).map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV gather synchronize: {e}")
                        })?;
                    }
                    if let Some(t) = timings.as_mut() {
                        t.certified_kv_gather_ms +=
                            cert_gather_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    let mut score_consistency_violations = 0usize;
                    let score_flags_shape = [num_q_heads];
                    if scratch
                        .certified_score_consistency_flags
                        .as_ref()
                        .map(|buf| buf.shape() != score_flags_shape)
                        .unwrap_or(true)
                    {
                        scratch.certified_score_consistency_flags = Some(
                            GpuBuffer::zeros(self.ordinal, ScalarType::U32, &score_flags_shape)
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                    "layer {idx} certified KV score-consistency flags alloc: {e}"
                                )
                                })?,
                        );
                    }
                    let cert_score_consistency_start = Instant::now();
                    let score_consistency_flags_ref = if use_device_selector {
                        scratch
                            .certified_ranking_fallback_head_flags
                            .as_mut()
                            .unwrap()
                    } else {
                        scratch.certified_score_consistency_flags.as_mut().unwrap()
                    };
                    kernel_ffi::certified_kv::score_consistency(
                        self.ordinal,
                        attn_q,
                        key_i8,
                        key_scale,
                        key_zero,
                        if use_key_page_cache {
                            ls.certified_kv_promoted_key_cache.as_ref().unwrap()
                        } else {
                            scratch.certified_promoted_key_bf16.as_ref().unwrap()
                        },
                        scratch.certified_promote_index.as_ref().unwrap(),
                        score_consistency_flags_ref,
                        block_size,
                        max_promoted_blocks,
                        gqa_group,
                        q_scale,
                        cfg.eps_guard,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} certified KV score consistency: {e}")
                    })?;
                    if trace_cert_phases {
                        eprintln!("cert-kv layer {idx}: score consistency done");
                    }
                    if !use_device_selector {
                        let score_consistency_flags = scratch
                            .certified_score_consistency_flags
                            .as_ref()
                            .unwrap()
                            .to_host_bytes()
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV score-consistency flags D2H: {e}"
                                )
                            })?;
                        for (qh, chunk) in score_consistency_flags
                            .chunks_exact(std::mem::size_of::<u32>())
                            .enumerate()
                        {
                            if u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) != 0 {
                                score_consistency_violations += 1;
                                dense_fallback_qhead_flags[qh] = true;
                            }
                        }
                    }
                    if let Some(t) = timings.as_mut() {
                        t.certified_kv_score_consistency_ms +=
                            cert_score_consistency_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    if cfg.ranking_r > 0 {
                        let selected_log_mass_shape = [num_q_heads, max_promoted_blocks];
                        if scratch
                            .certified_selected_fp16_log_masses
                            .as_ref()
                            .map(|buf| buf.shape() != selected_log_mass_shape)
                            .unwrap_or(true)
                        {
                            scratch.certified_selected_fp16_log_masses = Some(
                                GpuBuffer::zeros(
                                    self.ordinal,
                                    ScalarType::F32,
                                    &selected_log_mass_shape,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV selected FP16 log-mass alloc: {e}"
                                    )
                                })?,
                            );
                        }
                        let cert_rank_log_start = Instant::now();
                        kernel_ffi::certified_kv::selected_fp16_log_masses(
                            self.ordinal,
                            attn_q,
                            if use_key_page_cache {
                                ls.certified_kv_promoted_key_cache.as_ref().unwrap()
                            } else {
                                scratch.certified_promoted_key_bf16.as_ref().unwrap()
                            },
                            scratch.certified_promote_index.as_ref().unwrap(),
                            scratch.certified_selected_fp16_log_masses.as_mut().unwrap(),
                            block_size,
                            max_promoted_blocks,
                            q_scale,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV selected FP16 log-masses: {e}"
                            )
                        })?;
                        if trace_cert_phases {
                            eprintln!("cert-kv layer {idx}: selected fp16 log masses done");
                        }
                        if use_device_selector {
                            kernel_ffi::certified_kv::ranking_flags_device(
                                self.ordinal,
                                scratch.certified_block_max.as_ref().unwrap(),
                                scratch.certified_block_sum.as_ref().unwrap(),
                                scratch.certified_delta_blocks.as_ref().unwrap(),
                                scratch.certified_selected_fp16_log_masses.as_ref().unwrap(),
                                scratch.certified_promote_index.as_ref().unwrap(),
                                scratch
                                    .certified_ranking_fallback_head_flags
                                    .as_mut()
                                    .unwrap(),
                                max_promoted_blocks,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV device ranking check: {e}"
                                )
                            })?;
                            if trace_cert_phases {
                                gpu_hal::sync(self.ordinal).map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV trace sync after device ranking: {e}"
                                    )
                                })?;
                                eprintln!("cert-kv layer {idx}: device ranking check done");
                            }
                            if collect_cert_host_telemetry {
                                let device_fallback_flags = decode_u32_le(
                                    &scratch
                                        .certified_ranking_fallback_head_flags
                                        .as_ref()
                                        .unwrap()
                                        .to_host_bytes()
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "layer {idx} certified KV fallback flags D2H: {e}"
                                            )
                                        })?,
                                );
                                for (qh, flag) in device_fallback_flags.into_iter().enumerate() {
                                    if flag != 0 {
                                        dense_fallback_qhead_flags[qh] = true;
                                    }
                                }
                            }
                        } else {
                            let selected_fp16_log_mass_host = decode_f32_le(
                                &scratch
                                    .certified_selected_fp16_log_masses
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV selected FP16 log-masses D2H: {e}"
                                        )
                                    })?,
                            );
                            let cert_ranking_cpu_start = Instant::now();
                            for qh in 0..num_q_heads {
                                let score_start = qh * num_blocks;
                                let score_end = score_start + num_blocks;
                                let fp16_selected_log_masses: Vec<(usize, f32)> = selected_by_head
                                    [qh]
                                    .iter()
                                    .filter_map(|&block| {
                                        let slot = promote_index_host[score_start + block] as usize;
                                        (slot < max_promoted_blocks).then(|| {
                                            (
                                                block,
                                                selected_fp16_log_mass_host
                                                    [qh * max_promoted_blocks + slot],
                                            )
                                        })
                                    })
                                    .collect();
                                if certified_kv_ranking_mismatch(
                                    &block_max_host[score_start..score_end],
                                    &block_sum_host[score_start..score_end],
                                    &delta_blocks_by_head[qh],
                                    &fp16_selected_log_masses,
                                    &selected_by_head[qh],
                                    cfg.ranking_r,
                                ) {
                                    dense_fallback_qhead_flags[qh] = true;
                                }
                            }
                            if let Some(t) = timings.as_mut() {
                                t.certified_kv_ranking_cpu_ms +=
                                    cert_ranking_cpu_start.elapsed().as_secs_f64() * 1000.0;
                            }
                        }
                        if let Some(t) = timings.as_mut() {
                            t.certified_kv_rank_log_ms +=
                                cert_rank_log_start.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                    let ranking_fallback_qheads: Vec<usize> = dense_fallback_qhead_flags
                        .iter()
                        .enumerate()
                        .filter_map(|(qh, &fallback)| fallback.then_some(qh))
                        .collect();
                    let ranking_fallback_heads = ranking_fallback_qheads.len();
                    let ranking_fallback_qhead_flags = dense_fallback_qhead_flags;
                    let mut ranking_fallback_kv_slots_by_kvh = vec![usize::MAX; num_kv_heads];
                    let mut ranking_fallback_kv_heads = Vec::new();
                    let mut ranking_fallback_qhead_kv_slots = Vec::new();
                    for &qh in &ranking_fallback_qheads {
                        let kvh = qh / (num_q_heads / num_kv_heads);
                        let mut slot = ranking_fallback_kv_slots_by_kvh[kvh];
                        if slot == usize::MAX {
                            slot = ranking_fallback_kv_heads.len();
                            ranking_fallback_kv_slots_by_kvh[kvh] = slot;
                            ranking_fallback_kv_heads.push(kvh);
                        }
                        ranking_fallback_qhead_kv_slots.push(slot as u32);
                    }
                    if let Some(t) = timings.as_mut() {
                        t.certified_kv_ranking_fallback_heads += ranking_fallback_heads;
                        t.certified_kv_score_consistency_violations += score_consistency_violations;
                        if force_dense_layer_fallback {
                            t.certified_kv_dense_fallback_layers += 1;
                        }
                    }
                    if force_dense_layer_fallback {
                        let tier2_key = ls.certified_kv_host_k.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV dense fallback missing Tier-2 key buffer"
                            )
                        })?;
                        let tier2_value = ls.certified_kv_host_v.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV dense fallback missing Tier-2 value buffer"
                            )
                        })?;
                        let kv_k_contig = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::BF16,
                            &[1, num_kv_heads, kv_len, head_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_k_contig alloc: {e}"))?;
                        let kv_v_contig = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::BF16,
                            &[1, num_kv_heads, kv_len, head_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_v_contig alloc: {e}"))?;
                        let contig_stride = kv_len * head_dim * elem_bytes;
                        let prefix_tokens = aligned.min(kv_len);
                        let prefix_bytes = prefix_tokens * head_dim * elem_bytes;
                        let tail_bytes = tail_len * head_dim * elem_bytes;
                        let tail_offset = aligned * head_dim * elem_bytes;
                        let tail_k = if tail_len > 0 {
                            Some(ls.certified_kv_tail_k.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV dense fallback missing tail key buffer"
                                )
                            })?)
                        } else {
                            None
                        };
                        let tail_v = if tail_len > 0 {
                            Some(ls.certified_kv_tail_v.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV dense fallback missing tail value buffer"
                                )
                            })?)
                        } else {
                            None
                        };
                        let tail_stride = block_size * head_dim * elem_bytes;
                        for h in 0..num_kv_heads {
                            let src_bytes = h * cap * head_dim * elem_bytes;
                            let dst_bytes = h * contig_stride;
                            if prefix_bytes > 0 {
                                gpu_hal::copy_h2d(
                                    self.ordinal,
                                    kv_k_contig.offset_ptr(dst_bytes) as *mut c_void,
                                    tier2_key.offset_ptr(src_bytes),
                                    prefix_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified dense fallback key prefix H2D h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_h2d(
                                    self.ordinal,
                                    kv_v_contig.offset_ptr(dst_bytes) as *mut c_void,
                                    tier2_value.offset_ptr(src_bytes),
                                    prefix_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified dense fallback value prefix H2D h={h}: {e}"
                                    )
                                })?;
                            }
                            if tail_bytes > 0 {
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    kv_k_contig.offset_ptr(dst_bytes + tail_offset) as *mut c_void,
                                    tail_k.unwrap().offset_ptr(h * tail_stride),
                                    tail_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified dense fallback tail key copy h={h}: {e}"
                                    )
                                })?;
                                gpu_hal::copy_d2d(
                                    self.ordinal,
                                    kv_v_contig.offset_ptr(dst_bytes + tail_offset) as *mut c_void,
                                    tail_v.unwrap().offset_ptr(h * tail_stride),
                                    tail_bytes,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified dense fallback tail value copy h={h}: {e}"
                                    )
                                })?;
                            }
                        }
                        if has_attn_gate {
                            kernel_ffi::prefill_ffi::full_attention_prefill(
                                self.ordinal,
                                ScalarType::BF16,
                                1,
                                num_q_heads,
                                num_kv_heads,
                                1,
                                kv_len,
                                head_dim,
                                q_scale,
                                seqlen_offset,
                                attn_q,
                                &kv_k_contig,
                                &kv_v_contig,
                                attn_out_f32,
                            )
                            .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

                            kernel_ffi::prefill_ffi::cast(
                                self.ordinal,
                                ScalarType::F32,
                                ScalarType::BF16,
                                num_q_heads * head_dim,
                                attn_out_f32,
                                attn_out_bf16,
                            )
                            .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
                        } else {
                            kernel_ffi::prefill_ffi::full_attention_decode_flat(
                                self.ordinal,
                                ScalarType::BF16,
                                1,
                                num_q_heads,
                                num_kv_heads,
                                kv_len,
                                head_dim,
                                q_scale,
                                attn_q,
                                &kv_k_contig,
                                &kv_v_contig,
                                gated,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("layer {idx} decode attention flat: {e}")
                            })?;
                            certified_attn_already_flat = true;
                        }
                    } else {
                        if timings.is_some() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!("layer {idx} certified KV attention pre-sync: {e}")
                            })?;
                        }
                        let attend_start = Instant::now();
                        let promoted_value_ref = if use_initial_value_cache {
                            ls.certified_kv_promoted_value_cache.as_ref().unwrap()
                        } else {
                            scratch.certified_promoted_value_bf16.as_ref().unwrap()
                        };
                        kernel_ffi::certified_kv::attend_mixed_key_int4_with_bf16_tail_strided(
                            self.ordinal,
                            attn_q,
                            key_i8,
                            key_scale,
                            key_zero,
                            if use_key_page_cache {
                                ls.certified_kv_promoted_key_cache.as_ref().unwrap()
                            } else {
                                scratch.certified_promoted_key_bf16.as_ref().unwrap()
                            },
                            scratch.certified_promote_index.as_ref().unwrap(),
                            promoted_value_ref,
                            scratch.certified_value_promote_index.as_ref().unwrap(),
                            ls.certified_kv_value_i4.as_ref().unwrap(),
                            ls.certified_kv_value_scale.as_ref().unwrap(),
                            ls.certified_kv_value_zero.as_ref().unwrap(),
                            tail_key_kernel_ref,
                            tail_value_kernel_ref,
                            kv_len,
                            block_size,
                            value_group_size,
                            num_q_heads / num_kv_heads,
                            q_scale,
                            score_scratch,
                            attn_out_bf16,
                            None,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV mixed-key INT4 attention: {e}"
                            )
                        })?;
                        if trace_cert_phases {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV trace sync after initial attend: {e}"
                                )
                            })?;
                            eprintln!("cert-kv layer {idx}: initial attend done");
                        }
                        kernel_ffi::certified_kv::block_masses_from_token_probs(
                            self.ordinal,
                            score_scratch,
                            scratch.certified_final_block_mass.as_mut().unwrap(),
                            block_size,
                            None,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV final block mass reduction: {e}"
                            )
                        })?;
                        if trace_cert_phases {
                            eprintln!("cert-kv layer {idx}: final block masses done");
                        }
                        let ranking_fallback_head_flags_ref = if use_device_selector {
                            scratch.certified_ranking_fallback_head_flags.as_ref()
                        } else if ranking_fallback_qheads.is_empty() {
                            None
                        } else {
                            let fallback_flag_shape = [num_q_heads];
                            if scratch
                                .certified_ranking_fallback_head_flags
                                .as_ref()
                                .map(|buf| buf.shape() != fallback_flag_shape)
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_head_flags = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::U32,
                                        &fallback_flag_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV fallback head flags alloc: {e}"
                                        )
                                    })?,
                                );
                            }
                            let fallback_flags_u32: Vec<u32> = ranking_fallback_qhead_flags
                                .iter()
                                .map(|&flag| if flag { 1 } else { 0 })
                                .collect();
                            let fallback_flags_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    fallback_flags_u32.as_ptr() as *const u8,
                                    fallback_flags_u32.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_ranking_fallback_head_flags
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                fallback_flags_bytes.as_ptr() as *const c_void,
                                fallback_flags_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV fallback head flags H2D: {e}"
                                )
                            })?;
                            scratch.certified_ranking_fallback_head_flags.as_ref()
                        };
                        kernel_ffi::certified_kv::value_promotions_from_block_masses(
                            self.ordinal,
                            scratch.certified_final_block_mass.as_ref().unwrap(),
                            ls.certified_kv_value_error.as_ref().unwrap(),
                            ranking_fallback_head_flags_ref,
                            scratch.certified_value_promote_index.as_mut().unwrap(),
                            scratch.certified_value_promotion_counters.as_mut().unwrap(),
                            scratch.certified_value_promotion_any.as_mut().unwrap(),
                            scratch
                                .certified_value_promotion_head_flags
                                .as_mut()
                                .unwrap(),
                            scratch.certified_value_e_val_by_head.as_mut().unwrap(),
                            gqa_group,
                            cfg.v_tol,
                            None,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} certified KV final value-promotion evaluation: {e}"
                            )
                        })?;
                        if trace_cert_phases {
                            eprintln!("cert-kv layer {idx}: final value promotion eval done");
                        }
                        let added_value_promotions = if use_device_selector {
                            true
                        } else {
                            scratch
                                .certified_value_promotion_any
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV value promotion flag D2H: {e}"
                                    )
                                })?
                                .chunks_exact(std::mem::size_of::<u32>())
                                .next()
                                .map(|chunk| {
                                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                        != 0
                                })
                                .unwrap_or(false)
                        };
                        if added_value_promotions {
                            let final_promoted_by_kvh: Vec<Vec<usize>>;
                            if use_device_selector {
                                final_promoted_by_kvh = Vec::new();
                            } else {
                                let final_value_index_bytes = scratch
                                    .certified_value_promote_index
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV final value index D2H: {e}"
                                        )
                                    })?;
                                let final_value_index_host: Vec<u32> = final_value_index_bytes
                                    .chunks_exact(std::mem::size_of::<u32>())
                                    .map(|chunk| {
                                        u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                    })
                                    .collect();
                                final_promoted_by_kvh = (0..num_kv_heads)
                                    .map(|kvh| {
                                        (0..num_blocks)
                                            .filter(|&block| {
                                                final_value_index_host[kvh * num_blocks + block]
                                                    != u32::MAX
                                            })
                                            .collect()
                                    })
                                    .collect();
                            }
                            let final_value_cache_fits_step = !use_device_selector
                                && final_promoted_by_kvh
                                    .iter()
                                    .all(|blocks| blocks.len() <= value_cache_capacity);
                            let use_final_value_cache = !use_device_selector
                                && cfg.value_cache_blocks > 0
                                && final_value_cache_fits_step;
                            let final_promoted_value_ref: &GpuBuffer;
                            if use_final_value_cache {
                                let tag_len = num_kv_heads * value_cache_capacity;
                                let cache_shape =
                                    [num_kv_heads, value_cache_capacity, block_size, head_dim];
                                let reset_cache = ls
                                    .certified_kv_promoted_value_cache
                                    .as_ref()
                                    .map(|buf| buf.shape() != cache_shape)
                                    .unwrap_or(true)
                                    || ls.certified_kv_promoted_value_cache_capacity
                                        != value_cache_capacity
                                    || ls.certified_kv_promoted_value_cache_tags.len() != tag_len;
                                if reset_cache {
                                    ls.certified_kv_promoted_value_cache = Some(
                                        GpuBuffer::zeros(
                                            self.ordinal,
                                            ScalarType::BF16,
                                            &cache_shape,
                                        )
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "layer {idx} certified KV promoted value cache alloc: {e}"
                                            )
                                        })?,
                                    );
                                    ls.certified_kv_promoted_value_cache_capacity =
                                        value_cache_capacity;
                                    ls.certified_kv_promoted_value_cache_tags =
                                        vec![usize::MAX; tag_len];
                                    ls.certified_kv_promoted_value_cache_lru = vec![0; tag_len];
                                    ls.certified_kv_promoted_value_cache_tick = 0;
                                }

                                let mut final_value_promote_index_host =
                                    vec![u32::MAX; num_kv_heads * num_blocks];
                                let mut final_value_gather_index_host =
                                    vec![u32::MAX; num_kv_heads * num_blocks];
                                let mut final_cache_hits = 0usize;
                                let mut final_cache_misses = 0usize;
                                let mut final_cache_overflows = 0usize;
                                for kvh in 0..num_kv_heads {
                                    for &block in &final_promoted_by_kvh[kvh] {
                                        let base = kvh * value_cache_capacity;
                                        ls.certified_kv_promoted_value_cache_tick = ls
                                            .certified_kv_promoted_value_cache_tick
                                            .wrapping_add(1);
                                        let tick = ls.certified_kv_promoted_value_cache_tick;
                                        let mut slot = (0..value_cache_capacity).find(|&s| {
                                            ls.certified_kv_promoted_value_cache_tags[base + s]
                                                == block
                                        });
                                        if let Some(s) = slot {
                                            final_cache_hits += 1;
                                            ls.certified_kv_promoted_value_cache_lru[base + s] =
                                                tick;
                                        } else {
                                            final_cache_misses += 1;
                                            slot = (0..value_cache_capacity).find(|&s| {
                                                ls.certified_kv_promoted_value_cache_tags[base + s]
                                                    == usize::MAX
                                            });
                                            let s = if let Some(s) = slot {
                                                s
                                            } else {
                                                final_cache_overflows += 1;
                                                (0..value_cache_capacity)
                                                    .min_by_key(|&s| {
                                                        ls.certified_kv_promoted_value_cache_lru
                                                            [base + s]
                                                    })
                                                    .unwrap_or(0)
                                            };
                                            ls.certified_kv_promoted_value_cache_tags[base + s] =
                                                block;
                                            ls.certified_kv_promoted_value_cache_lru[base + s] =
                                                tick;
                                            final_value_gather_index_host
                                                [kvh * num_blocks + block] = s as u32;
                                            slot = Some(s);
                                        }
                                        final_value_promote_index_host[kvh * num_blocks + block] =
                                            slot.unwrap() as u32;
                                    }
                                }
                                let final_gather_index_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        final_value_gather_index_host.as_ptr() as *const u8,
                                        final_value_gather_index_host.len()
                                            * std::mem::size_of::<u32>(),
                                    )
                                };
                                gpu_hal::copy_h2d(
                                    self.ordinal,
                                    scratch
                                        .certified_value_promote_index
                                        .as_mut()
                                        .unwrap()
                                        .as_mut_ptr(),
                                    final_gather_index_bytes.as_ptr() as *const c_void,
                                    final_gather_index_bytes.len(),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV final value cache gather index H2D: {e}"
                                    )
                                })?;
                                kernel_ffi::certified_kv::gather_promoted_values_bf16_from_tier2(
                                    self.ordinal,
                                    tier2_value_device_ptr,
                                    scratch.certified_value_promote_index.as_ref().unwrap(),
                                    ls.certified_kv_promoted_value_cache.as_mut().unwrap(),
                                    block_size,
                                    cap,
                                    value_cache_capacity,
                                    Some(scratch.certified_value_promotion_any.as_ref().unwrap()),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV exact promoted BF16 value cache gather: {e}"
                                    )
                                })?;
                                let final_promote_index_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        final_value_promote_index_host.as_ptr() as *const u8,
                                        final_value_promote_index_host.len()
                                            * std::mem::size_of::<u32>(),
                                    )
                                };
                                gpu_hal::copy_h2d(
                                    self.ordinal,
                                    scratch
                                        .certified_value_promote_index
                                        .as_mut()
                                        .unwrap()
                                        .as_mut_ptr(),
                                    final_promote_index_bytes.as_ptr() as *const c_void,
                                    final_promote_index_bytes.len(),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV final cached value promote index H2D: {e}"
                                    )
                                })?;
                                if let Some(t) = timings.as_mut() {
                                    t.certified_kv_promoted_value_cache_hits += final_cache_hits;
                                    t.certified_kv_promoted_value_cache_misses +=
                                        final_cache_misses;
                                    t.certified_kv_promoted_value_cache_overflows +=
                                        final_cache_overflows;
                                    t.certified_kv_promoted_value_h2d_bytes +=
                                        final_cache_misses * value_block_bytes;
                                }
                                final_promoted_value_ref =
                                    ls.certified_kv_promoted_value_cache.as_ref().unwrap();
                            } else {
                                max_promoted_value_blocks = num_blocks;
                                let promoted_value_shape = [
                                    num_kv_heads,
                                    max_promoted_value_blocks,
                                    block_size,
                                    head_dim,
                                ];
                                if scratch
                                    .certified_promoted_value_bf16
                                    .as_ref()
                                    .map(|buf| buf.shape() != promoted_value_shape)
                                    .unwrap_or(true)
                                {
                                    scratch.certified_promoted_value_bf16 = Some(
                                        GpuBuffer::zeros(
                                            self.ordinal,
                                            ScalarType::BF16,
                                            &promoted_value_shape,
                                        )
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "layer {idx} certified KV promoted value realloc: {e}"
                                            )
                                        })?,
                                    );
                                }
                                kernel_ffi::certified_kv::gather_promoted_values_bf16_from_tier2(
                                    self.ordinal,
                                    tier2_value_device_ptr,
                                    scratch.certified_value_promote_index.as_ref().unwrap(),
                                    scratch.certified_promoted_value_bf16.as_mut().unwrap(),
                                    block_size,
                                    cap,
                                    max_promoted_value_blocks,
                                    Some(scratch.certified_value_promotion_any.as_ref().unwrap()),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV exact promoted BF16 value gather: {e}"
                                    )
                                })?;
                                if collect_cert_host_telemetry {
                                    if let Some(t) = timings.as_mut() {
                                        let final_value_counter_bytes = scratch
                                        .certified_value_promotion_counters
                                        .as_ref()
                                        .unwrap()
                                        .to_host_bytes()
                                        .map_err(|e| {
                                            anyhow::anyhow!(
                                                "layer {idx} certified KV final value counters D2H: {e}"
                                            )
                                        })?;
                                        let final_value_blocks = final_value_counter_bytes
                                            .chunks_exact(std::mem::size_of::<u32>())
                                            .map(|chunk| {
                                                u32::from_le_bytes([
                                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                                ])
                                                    as usize
                                            })
                                            .sum::<usize>();
                                        t.certified_kv_promoted_value_h2d_bytes +=
                                            final_value_blocks * value_block_bytes;
                                    }
                                }
                                final_promoted_value_ref =
                                    scratch.certified_promoted_value_bf16.as_ref().unwrap();
                            }
                            kernel_ffi::certified_kv::attend_mixed_key_int4_with_bf16_tail_strided(
                                self.ordinal,
                                attn_q,
                                key_i8,
                                key_scale,
                                key_zero,
                                if use_key_page_cache {
                                    ls.certified_kv_promoted_key_cache.as_ref().unwrap()
                                } else {
                                    scratch.certified_promoted_key_bf16.as_ref().unwrap()
                                },
                                scratch.certified_promote_index.as_ref().unwrap(),
                                final_promoted_value_ref,
                                scratch.certified_value_promote_index.as_ref().unwrap(),
                                ls.certified_kv_value_i4.as_ref().unwrap(),
                                ls.certified_kv_value_scale.as_ref().unwrap(),
                                ls.certified_kv_value_zero.as_ref().unwrap(),
                                tail_key_kernel_ref,
                                tail_value_kernel_ref,
                                kv_len,
                                block_size,
                                value_group_size,
                                num_q_heads / num_kv_heads,
                                q_scale,
                                score_scratch,
                                attn_out_bf16,
                                Some(scratch.certified_value_promotion_any.as_ref().unwrap()),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV exact mixed-key INT4 attention: {e}"
                                )
                            })?;
                            kernel_ffi::certified_kv::block_masses_from_token_probs(
                                self.ordinal,
                                score_scratch,
                                scratch.certified_final_block_mass.as_mut().unwrap(),
                                block_size,
                                Some(scratch.certified_value_promotion_any.as_ref().unwrap()),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV exact final block mass reduction: {e}"
                                )
                            })?;
                            kernel_ffi::certified_kv::value_promotions_from_block_masses(
                                self.ordinal,
                                scratch.certified_final_block_mass.as_ref().unwrap(),
                                ls.certified_kv_value_error.as_ref().unwrap(),
                                ranking_fallback_head_flags_ref,
                                scratch.certified_value_promote_index.as_mut().unwrap(),
                                scratch.certified_value_promotion_counters.as_mut().unwrap(),
                                scratch.certified_value_promotion_any.as_mut().unwrap(),
                                scratch
                                    .certified_value_promotion_head_flags
                                    .as_mut()
                                    .unwrap(),
                                scratch.certified_value_e_val_by_head.as_mut().unwrap(),
                                gqa_group,
                                cfg.v_tol,
                                None,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV exact final value-promotion evaluation: {e}"
                                )
                            })?;
                        }
                        if collect_cert_host_telemetry {
                            let final_value_head_flags = scratch
                                .certified_value_promotion_head_flags
                                .as_ref()
                                .unwrap()
                                .to_host_bytes()
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV value promotion head flags D2H: {e}"
                                    )
                                })?;
                            let final_value_escalation_heads = final_value_head_flags
                                .chunks_exact(std::mem::size_of::<u32>())
                                .filter(|chunk| {
                                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                        != 0
                                })
                                .count();
                            let e_val_by_head = decode_f32_le(
                                &scratch
                                    .certified_value_e_val_by_head
                                    .as_ref()
                                    .unwrap()
                                    .to_host_bytes()
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV e_val-by-head D2H: {e}"
                                        )
                                    })?,
                            );
                            if let Some(t) = timings.as_mut() {
                                t.certified_kv_value_escalation_heads +=
                                    final_value_escalation_heads;
                                for qh in 0..num_q_heads {
                                    let mut e_val = e_val_by_head[qh];
                                    let e_key = if ranking_fallback_qhead_flags[qh] {
                                        0.0
                                    } else {
                                        e_key_by_head[qh]
                                    };
                                    if ranking_fallback_qhead_flags[qh] {
                                        e_val = 0.0;
                                    }
                                    t.certified_kv_e_key_max = t.certified_kv_e_key_max.max(e_key);
                                    t.certified_kv_delta_tail_max = t
                                        .certified_kv_delta_tail_max
                                        .max(if ranking_fallback_qhead_flags[qh] {
                                            0.0
                                        } else {
                                            delta_tail_by_head[qh]
                                        });
                                    t.certified_kv_vmax_max = t.certified_kv_vmax_max.max(
                                        if ranking_fallback_qhead_flags[qh] {
                                            0.0
                                        } else {
                                            vmax_by_head[qh]
                                        },
                                    );
                                    t.certified_kv_true_tail_bound_max = t
                                        .certified_kv_true_tail_bound_max
                                        .max(if ranking_fallback_qhead_flags[qh] {
                                            0.0
                                        } else {
                                            true_tail_bound_by_head[qh]
                                        });
                                    t.certified_kv_e_val_max = t.certified_kv_e_val_max.max(e_val);
                                    t.certified_kv_bound_total_max =
                                        t.certified_kv_bound_total_max.max(e_key + e_val);
                                }
                            }
                        }
                        if use_device_selector && cfg.ranking_r > 0 {
                            let prefix_tokens = aligned.min(kv_len);
                            let prefix_shape = [num_kv_heads, cap, head_dim];
                            let reset_prefix_cache = ls
                                .certified_kv_ranking_prefix_k
                                .as_ref()
                                .map(|buf| buf.shape() != prefix_shape)
                                .unwrap_or(true)
                                || ls
                                    .certified_kv_ranking_prefix_v
                                    .as_ref()
                                    .map(|buf| buf.shape() != prefix_shape)
                                    .unwrap_or(true)
                                || ls.certified_kv_ranking_prefix_tokens > prefix_tokens;
                            if reset_prefix_cache {
                                ls.certified_kv_ranking_prefix_k = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::BF16,
                                        &prefix_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV full fallback prefix key alloc: {e}"
                                        )
                                    })?,
                                );
                                ls.certified_kv_ranking_prefix_v = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::BF16,
                                        &prefix_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV full fallback prefix value alloc: {e}"
                                        )
                                    })?,
                                );
                                ls.certified_kv_ranking_prefix_tokens = 0;
                                ls.certified_kv_ranking_prefix_kv_heads.clear();
                            }
                            if ls.certified_kv_ranking_prefix_tokens < prefix_tokens {
                                let old_tokens = ls.certified_kv_ranking_prefix_tokens;
                                let new_tokens = prefix_tokens - old_tokens;
                                let token_offset_bytes = old_tokens * head_dim * elem_bytes;
                                let copy_bytes = new_tokens * head_dim * elem_bytes;
                                let dst_stride_bytes = cap * head_dim * elem_bytes;
                                let tier2_key = ls.certified_kv_host_k.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV flagged fallback missing Tier-2 key buffer"
                                    )
                                })?;
                                let tier2_value = ls.certified_kv_host_v.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV flagged fallback missing Tier-2 value buffer"
                                    )
                                })?;
                                let fallback_key =
                                    ls.certified_kv_ranking_prefix_k.as_mut().unwrap();
                                let fallback_value =
                                    ls.certified_kv_ranking_prefix_v.as_mut().unwrap();
                                for kvh in 0..num_kv_heads {
                                    let src_bytes = kvh * dst_stride_bytes + token_offset_bytes;
                                    let dst_bytes = kvh * dst_stride_bytes + token_offset_bytes;
                                    gpu_hal::copy_h2d(
                                        self.ordinal,
                                        fallback_key.offset_ptr(dst_bytes) as *mut c_void,
                                        tier2_key.offset_ptr(src_bytes),
                                        copy_bytes,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified flagged fallback key H2D kvh={kvh}: {e}"
                                        )
                                    })?;
                                    gpu_hal::copy_h2d(
                                        self.ordinal,
                                        fallback_value.offset_ptr(dst_bytes) as *mut c_void,
                                        tier2_value.offset_ptr(src_bytes),
                                        copy_bytes,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified flagged fallback value H2D kvh={kvh}: {e}"
                                        )
                                    })?;
                                }
                                ls.certified_kv_ranking_prefix_tokens = prefix_tokens;
                                if let Some(t) = timings.as_mut() {
                                    t.certified_kv_ranking_prefix_cache_misses += num_kv_heads;
                                    t.certified_kv_ranking_prefix_h2d_bytes +=
                                        2 * num_kv_heads * copy_bytes;
                                }
                            } else if let Some(t) = timings.as_mut() {
                                t.certified_kv_ranking_prefix_cache_hits += num_kv_heads;
                                t.certified_kv_ranking_prefix_reuse_bytes +=
                                    2 * num_kv_heads * prefix_tokens * head_dim * elem_bytes;
                            }
                            let tail_k = if tail_len > 0 {
                                Some(ls.certified_kv_tail_k.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV flagged fallback missing tail key buffer"
                                    )
                                })?)
                            } else {
                                None
                            };
                            let tail_v = if tail_len > 0 {
                                Some(ls.certified_kv_tail_v.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV flagged fallback missing tail value buffer"
                                    )
                                })?)
                            } else {
                                None
                            };

                            let fallback_score_shape = [num_q_heads, kv_len];
                            if scratch
                                .certified_ranking_fallback_score
                                .as_ref()
                                .map(|buf| buf.shape() != fallback_score_shape)
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_score = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::F32,
                                        &fallback_score_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV flagged fallback score alloc: {e}"
                                        )
                                    })?,
                                );
                            }
                            kernel_ffi::certified_kv::dense_flagged_heads_out_bf16(
                                self.ordinal,
                                attn_q,
                                scratch
                                    .certified_ranking_fallback_head_flags
                                    .as_ref()
                                    .unwrap(),
                                ls.certified_kv_ranking_prefix_k.as_ref().unwrap(),
                                ls.certified_kv_ranking_prefix_v.as_ref().unwrap(),
                                prefix_tokens,
                                tail_k,
                                tail_v,
                                kv_len,
                                scratch.certified_ranking_fallback_score.as_mut().unwrap(),
                                attn_out_bf16,
                                q_scale,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV flagged-head fallback: {e}"
                                )
                            })?;
                        } else if !ranking_fallback_qheads.is_empty() {
                            let fallback_count = ranking_fallback_qheads.len();
                            let fallback_kv_count = ranking_fallback_kv_heads.len();
                            let fallback_heads_shape = [fallback_count];
                            let fallback_score_shape = [fallback_count, kv_len];
                            if scratch
                                .certified_ranking_fallback_heads
                                .as_ref()
                                .map(|buf| buf.shape() != fallback_heads_shape)
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_heads = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::U32,
                                        &fallback_heads_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback heads alloc: {e}"
                                        )
                                    })?,
                                );
                            }
                            if scratch
                                .certified_ranking_fallback_kv_slots
                                .as_ref()
                                .map(|buf| buf.shape() != fallback_heads_shape)
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_kv_slots = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::U32,
                                        &fallback_heads_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback kv slots alloc: {e}"
                                        )
                                    })?,
                                );
                            }
                            if scratch
                                .certified_ranking_fallback_kv_heads
                                .as_ref()
                                .map(|buf| buf.shape() != [fallback_kv_count])
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_kv_heads = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::U32,
                                        &[fallback_kv_count],
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback kv heads alloc: {e}"
                                        )
                                    })?,
                                );
                            }
                            if scratch
                                .certified_ranking_fallback_score
                                .as_ref()
                                .map(|buf| buf.shape() != fallback_score_shape)
                                .unwrap_or(true)
                            {
                                scratch.certified_ranking_fallback_score = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::F32,
                                        &fallback_score_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback score alloc: {e}"
                                        )
                                    })?,
                                );
                            }

                            let fallback_heads_u32: Vec<u32> = ranking_fallback_qheads
                                .iter()
                                .map(|&qh| qh as u32)
                                .collect();
                            let fallback_heads_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    fallback_heads_u32.as_ptr() as *const u8,
                                    fallback_heads_u32.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_ranking_fallback_heads
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                fallback_heads_bytes.as_ptr() as *const c_void,
                                fallback_heads_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking fallback heads H2D: {e}"
                                )
                            })?;
                            let fallback_kv_slots_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    ranking_fallback_qhead_kv_slots.as_ptr() as *const u8,
                                    ranking_fallback_qhead_kv_slots.len()
                                        * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_ranking_fallback_kv_slots
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                fallback_kv_slots_bytes.as_ptr() as *const c_void,
                                fallback_kv_slots_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking fallback kv slots H2D: {e}"
                                )
                            })?;
                            let fallback_kv_heads_u32: Vec<u32> = ranking_fallback_kv_heads
                                .iter()
                                .map(|&kvh| kvh as u32)
                                .collect();
                            let fallback_kv_heads_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    fallback_kv_heads_u32.as_ptr() as *const u8,
                                    fallback_kv_heads_u32.len() * std::mem::size_of::<u32>(),
                                )
                            };
                            gpu_hal::copy_h2d(
                                self.ordinal,
                                scratch
                                    .certified_ranking_fallback_kv_heads
                                    .as_mut()
                                    .unwrap()
                                    .as_mut_ptr(),
                                fallback_kv_heads_bytes.as_ptr() as *const c_void,
                                fallback_kv_heads_bytes.len(),
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking fallback kv heads H2D: {e}"
                                )
                            })?;

                            let tier2_key = ls.certified_kv_host_k.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking fallback missing Tier-2 key buffer"
                                )
                            })?;
                            let tier2_value = ls.certified_kv_host_v.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking fallback missing Tier-2 value buffer"
                                )
                            })?;
                            let prefix_tokens = aligned.min(kv_len);
                            let prefix_bytes = prefix_tokens * head_dim * elem_bytes;
                            let prefix_shape = [fallback_kv_count, prefix_tokens, head_dim];
                            let tail_k = if tail_len > 0 {
                                Some(ls.certified_kv_tail_k.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV ranking fallback missing tail key buffer"
                                    )
                                })?)
                            } else {
                                None
                            };
                            let tail_v = if tail_len > 0 {
                                Some(ls.certified_kv_tail_v.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "layer {idx} certified KV ranking fallback missing tail value buffer"
                                    )
                                })?)
                            } else {
                                None
                            };
                            let cache_valid = ls.certified_kv_ranking_prefix_tokens
                                == prefix_tokens
                                && ls.certified_kv_ranking_prefix_kv_heads
                                    == ranking_fallback_kv_heads
                                && ls
                                    .certified_kv_ranking_prefix_k
                                    .as_ref()
                                    .map(|buf| buf.shape() == prefix_shape)
                                    .unwrap_or(false)
                                && ls
                                    .certified_kv_ranking_prefix_v
                                    .as_ref()
                                    .map(|buf| buf.shape() == prefix_shape)
                                    .unwrap_or(false);
                            let ranking_prefix_bytes = 2 * fallback_kv_count * prefix_bytes;
                            if let Some(t) = timings.as_mut() {
                                if cache_valid {
                                    t.certified_kv_ranking_prefix_cache_hits += fallback_kv_count;
                                    t.certified_kv_ranking_prefix_reuse_bytes +=
                                        ranking_prefix_bytes;
                                } else {
                                    t.certified_kv_ranking_prefix_cache_misses += fallback_kv_count;
                                    t.certified_kv_ranking_prefix_h2d_bytes += ranking_prefix_bytes;
                                }
                            }
                            if !cache_valid {
                                ls.certified_kv_ranking_prefix_k = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::BF16,
                                        &prefix_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback prefix key alloc: {e}"
                                        )
                                    })?,
                                );
                                ls.certified_kv_ranking_prefix_v = Some(
                                    GpuBuffer::zeros(
                                        self.ordinal,
                                        ScalarType::BF16,
                                        &prefix_shape,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified KV ranking fallback prefix value alloc: {e}"
                                        )
                                    })?,
                                );
                                ls.certified_kv_ranking_prefix_tokens = prefix_tokens;
                                ls.certified_kv_ranking_prefix_kv_heads =
                                    ranking_fallback_kv_heads.clone();
                                let fallback_key =
                                    ls.certified_kv_ranking_prefix_k.as_mut().unwrap();
                                let fallback_value =
                                    ls.certified_kv_ranking_prefix_v.as_mut().unwrap();
                                let fallback_stride = prefix_tokens * head_dim * elem_bytes;
                                for (slot, &kvh) in ranking_fallback_kv_heads.iter().enumerate() {
                                    let src_bytes = kvh * cap * head_dim * elem_bytes;
                                    let dst_bytes = slot * fallback_stride;
                                    if prefix_bytes == 0 {
                                        continue;
                                    }
                                    gpu_hal::copy_h2d(
                                        self.ordinal,
                                        fallback_key.offset_ptr(dst_bytes) as *mut c_void,
                                        tier2_key.offset_ptr(src_bytes),
                                        prefix_bytes,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified ranking fallback key H2D kvh={kvh}: {e}"
                                        )
                                    })?;
                                    gpu_hal::copy_h2d(
                                        self.ordinal,
                                        fallback_value.offset_ptr(dst_bytes) as *mut c_void,
                                        tier2_value.offset_ptr(src_bytes),
                                        prefix_bytes,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "layer {idx} certified ranking fallback value H2D kvh={kvh}: {e}"
                                        )
                                    })?;
                                }
                            }

                            kernel_ffi::certified_kv::dense_selected_heads_out_bf16(
                                self.ordinal,
                                attn_q,
                                scratch.certified_ranking_fallback_heads.as_ref().unwrap(),
                                scratch
                                    .certified_ranking_fallback_kv_slots
                                    .as_ref()
                                    .unwrap(),
                                scratch
                                    .certified_ranking_fallback_kv_heads
                                    .as_ref()
                                    .unwrap(),
                                ls.certified_kv_ranking_prefix_k.as_ref().unwrap(),
                                ls.certified_kv_ranking_prefix_v.as_ref().unwrap(),
                                tail_k,
                                tail_v,
                                kv_len,
                                scratch.certified_ranking_fallback_score.as_mut().unwrap(),
                                attn_out_bf16,
                                q_scale,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV ranking selected-head fallback: {e}"
                                )
                            })?;
                        }
                        if let Some(t) = timings.as_mut() {
                            gpu_hal::sync(self.ordinal).map_err(|e| {
                                anyhow::anyhow!(
                                    "layer {idx} certified KV INT4 attention synchronize: {e}"
                                )
                            })?;
                            t.certified_kv_attend_ms +=
                                attend_start.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                if has_attn_gate {
                    kernel_ffi::prefill_ffi::transpose_shd_hsd(
                        self.ordinal,
                        ScalarType::BF16,
                        num_q_heads,
                        1,
                        head_dim,
                        attn_out_bf16,
                        attn_flat,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} certified KV decode transpose back: {e}")
                    })?;
                    if trace_output {
                        pre_gate_trace =
                            Some(attn_flat.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} pre-gate trace D2H: {e}")
                            })?);
                    }
                    kernel_ffi::prefill_ffi::sigmoid_mul(
                        self.ordinal,
                        ScalarType::BF16,
                        q_dim,
                        attn_flat,
                        gate_buf,
                        gated,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} gate apply: {e}"))?;
                } else {
                    if !certified_attn_already_flat {
                        kernel_ffi::prefill_ffi::transpose_shd_hsd(
                            self.ordinal,
                            ScalarType::BF16,
                            num_q_heads,
                            1,
                            head_dim,
                            attn_out_bf16,
                            gated,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("layer {idx} certified KV decode transpose back: {e}")
                        })?;
                    }
                    if trace_output {
                        pre_gate_trace =
                            Some(gated.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} pre-gate trace D2H: {e}")
                            })?);
                    }
                }
            } else {
                let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
                let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
                if has_attn_gate {
                    let kv_k_contig;
                    let kv_v_contig;
                    let attn_k_ref;
                    let attn_v_ref;
                    if cap == kv_len {
                        attn_k_ref = cache_k_ref;
                        attn_v_ref = cache_v_ref;
                    } else {
                        kv_k_contig = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, kv_len, head_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_k_contig alloc: {e}"))?;
                        kv_v_contig = GpuBuffer::zeros(
                            self.ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, kv_len, head_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_v_contig alloc: {e}"))?;
                        let cap_stride = cap * head_dim * elem_bytes;
                        let contig_stride = kv_len * head_dim * elem_bytes;
                        let copy_bytes = kv_len * head_dim * elem_bytes;
                        for h in 0..num_kv_heads {
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                kv_k_contig.offset_ptr(h * contig_stride) as *mut c_void,
                                cache_k_ref.offset_ptr(h * cap_stride),
                                copy_bytes,
                            )
                            .map_err(|e| anyhow::anyhow!("layer {idx} kv assemble k h={h}: {e}"))?;
                            gpu_hal::copy_d2d(
                                self.ordinal,
                                kv_v_contig.offset_ptr(h * contig_stride) as *mut c_void,
                                cache_v_ref.offset_ptr(h * cap_stride),
                                copy_bytes,
                            )
                            .map_err(|e| anyhow::anyhow!("layer {idx} kv assemble v h={h}: {e}"))?;
                        }
                        attn_k_ref = &kv_k_contig;
                        attn_v_ref = &kv_v_contig;
                    }
                    kernel_ffi::prefill_ffi::full_attention_prefill(
                        self.ordinal,
                        ScalarType::BF16,
                        1,
                        num_q_heads,
                        num_kv_heads,
                        1,
                        kv_len,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                        seqlen_offset,
                        attn_q,
                        attn_k_ref,
                        attn_v_ref,
                        attn_out_f32,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

                    kernel_ffi::prefill_ffi::cast(
                        self.ordinal,
                        ScalarType::F32,
                        ScalarType::BF16,
                        num_q_heads * head_dim,
                        attn_out_f32,
                        attn_out_bf16,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
                    kernel_ffi::prefill_ffi::transpose_shd_hsd(
                        self.ordinal,
                        ScalarType::BF16,
                        num_q_heads,
                        1,
                        head_dim,
                        attn_out_bf16,
                        attn_flat,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} attn transpose back: {e}"))?;
                    if trace_output {
                        pre_gate_trace =
                            Some(attn_flat.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} pre-gate trace D2H: {e}")
                            })?);
                    }
                    kernel_ffi::prefill_ffi::sigmoid_mul(
                        self.ordinal,
                        ScalarType::BF16,
                        q_dim,
                        attn_flat,
                        gate_buf,
                        gated,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} gate apply: {e}"))?;
                } else if cap == kv_len {
                    kernel_ffi::prefill_ffi::full_attention_decode_flat(
                        self.ordinal,
                        ScalarType::BF16,
                        1,
                        num_q_heads,
                        num_kv_heads,
                        kv_len,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                        attn_q,
                        cache_k_ref,
                        cache_v_ref,
                        gated,
                    )
                    .map_err(|e| anyhow::anyhow!("layer {idx} decode attention flat: {e}"))?;
                    if trace_output {
                        pre_gate_trace =
                            Some(gated.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} pre-gate trace D2H: {e}")
                            })?);
                    }
                } else {
                    kernel_ffi::prefill_ffi::full_attention_decode_flat_strided(
                        self.ordinal,
                        ScalarType::BF16,
                        1,
                        num_q_heads,
                        num_kv_heads,
                        kv_len,
                        cap,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                        attn_q,
                        cache_k_ref,
                        cache_v_ref,
                        gated,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("layer {idx} decode attention flat strided: {e}")
                    })?;
                    if trace_output {
                        pre_gate_trace =
                            Some(gated.to_host_bytes().map_err(|e| {
                                anyhow::anyhow!("layer {idx} pre-gate trace D2H: {e}")
                            })?);
                    }
                }
            }
            self.sync_stage_if_requested(
                collect_timings,
                &format!("layer {idx} full attention core"),
            )?;
            if let Some(t) = timings.as_mut() {
                t.persistent_full_attn_core_ms += core_start.elapsed().as_secs_f64() * 1000.0;
            }
            if trace_output {
                gated_trace = Some(
                    gated
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} gated trace D2H: {e}"))?,
                );
            }

            let out_start = Instant::now();
            if use_cublas_full_proj {
                kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                    self.ordinal,
                    proj_out,
                    gated,
                    &fw.o_proj_w,
                    q_dim,
                    hidden_dim,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS o_proj: {e}"))?;
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    hidden_dim,
                    q_dim,
                    gated,
                    &fw.o_proj_w,
                    fw.o_proj_scale.as_ref(),
                    fw.o_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    proj_out,
                    fw.o_proj_int4_scale.as_ref(),
                    fw.o_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }
            if trace_output {
                proj_out_trace = Some(
                    proj_out
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} proj_out trace D2H: {e}"))?,
                );
            }
            residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &proj_out)?;
            self.sync_stage_if_requested(
                collect_timings,
                &format!("layer {idx} full attention out"),
            )?;
            if let Some(t) = timings.as_mut() {
                t.persistent_full_attn_out_ms += out_start.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(if trace_output {
                Some(ComponentFullAttentionTrace {
                    q_proj: q_proj_trace.unwrap_or_default(),
                    gate_proj: gate_proj_trace.unwrap_or_default(),
                    k_proj: k_proj_trace.unwrap_or_default(),
                    v_proj: v_proj_trace.unwrap_or_default(),
                    q_rope: q_rope_trace.unwrap_or_default(),
                    k_rope: k_rope_trace.unwrap_or_default(),
                    pre_gate: pre_gate_trace.unwrap_or_default(),
                    gated: gated_trace.unwrap_or_default(),
                    proj_out: proj_out_trace.unwrap_or_default(),
                    attn_hidden: self
                        .hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} attn hidden trace D2H: {e}"))?,
                })
            } else {
                None
            })
        })();
        self.component_full_attn_scratch = Some(scratch);
        result
    }

    fn component_decode_linear_attention_layer(
        &mut self,
        idx: usize,
        trace_output: bool,
    ) -> Result<Option<ComponentLinearTrace>> {
        let config = &self.weights.config;
        let lw = self.weights.layers[idx]
            .linear
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear attention weights"))?;
        let hidden_dim = config.hidden_size;
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;
        let key_dim = nk * khd;
        let val_dim = nv * vhd;
        let qkv_dim = key_dim * 2 + val_dim;
        let head_repeat = nv / nk;

        let mut qkv = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, qkv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} qkv alloc: {e}"))?;
        let mut z = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} z alloc: {e}"))?;
        let mut a = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("layer {idx} a alloc: {e}"))?;
        let mut b = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("layer {idx} b alloc: {e}"))?;
        let a_beta_raw = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv * 2])
            .map_err(|e| anyhow::anyhow!("layer {idx} a_beta alloc: {e}"))?;
        let mut rec_apply = GpuBuffer::zeros(
            self.ordinal,
            ScalarType::F32,
            &[1, val_dim + nv * khd * vhd],
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} rec_apply alloc: {e}"))?;
        let mut attn_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_bf16 alloc: {e}"))?;
        let mut norm_w_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} norm_w alloc: {e}"))?;
        let mut gated = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} gated alloc: {e}"))?;
        let mut proj_out = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} proj_out alloc: {e}"))?;

        matmul_proj(
            self.ordinal,
            1,
            1,
            qkv_dim,
            hidden_dim,
            &self.normed_buf,
            &lw.qkv_proj_w,
            lw.qkv_proj_scale.as_ref(),
            lw.qkv_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut qkv,
            lw.qkv_proj_int4_scale.as_ref(),
            lw.qkv_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        let qkv_trace = if trace_output {
            Some(
                qkv.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} qkv trace D2H: {e}"))?,
            )
        } else {
            None
        };
        matmul_proj(
            self.ordinal,
            1,
            1,
            val_dim,
            hidden_dim,
            &self.normed_buf,
            &lw.z_proj_w,
            lw.z_proj_scale.as_ref(),
            lw.z_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut z,
            lw.z_proj_int4_scale.as_ref(),
            lw.z_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        let z_trace = if trace_output {
            Some(
                z.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} z trace D2H: {e}"))?,
            )
        } else {
            None
        };
        matmul_proj(
            self.ordinal,
            1,
            1,
            nv,
            hidden_dim,
            &self.normed_buf,
            &lw.a_proj_w,
            lw.a_proj_scale.as_ref(),
            lw.a_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut a,
            None,
            None,
            self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            nv,
            hidden_dim,
            &self.normed_buf,
            &lw.b_proj_w,
            lw.b_proj_scale.as_ref(),
            lw.b_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut b,
            None,
            None,
            self.weights.int4_group_size,
        )?;

        let ab_bytes = nv * ScalarType::BF16.size_in_bytes();
        gpu_hal::copy_d2d(
            self.ordinal,
            a_beta_raw.as_ptr() as *mut c_void,
            a.as_ptr(),
            ab_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} copy A: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            a_beta_raw.offset_ptr(ab_bytes) as *mut c_void,
            b.as_ptr(),
            ab_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} copy B: {e}"))?;
        let a_trace = if trace_output {
            Some(
                a.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} a trace D2H: {e}"))?,
            )
        } else {
            None
        };
        let b_trace = if trace_output {
            Some(
                b.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} b trace D2H: {e}"))?,
            )
        } else {
            None
        };

        let ls = &mut self.state.layers[idx];
        let conv_state = ls
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing conv state"))?;
        let recurrent_state = ls
            .recurrent_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing recurrent state"))?;

        let mut conv_pack = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, qkv_dim + nv])
            .map_err(|e| anyhow::anyhow!("layer {idx} conv_pack alloc: {e}"))?;
        kernel_ffi::prefill_ffi::linear_stateful_conv_value_decay_4b(
            self.ordinal,
            ScalarType::BF16,
            1,
            qkv_dim,
            1,
            config.linear_conv_kernel_dim - 1,
            config.linear_conv_kernel_dim,
            nv,
            &qkv,
            conv_state,
            &lw.conv1d_w,
            &a,
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut conv_pack,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} linear conv/value_decay: {e}"))?;

        let mut q_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_linear alloc: {e}"))?;
        let mut k_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_linear alloc: {e}"))?;
        let mut v_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear alloc: {e}"))?;
        kernel_ffi::prefill_ffi::split_qkv(
            self.ordinal,
            ScalarType::BF16,
            1,
            key_dim,
            val_dim,
            &conv_pack,
            &mut q_linear,
            &mut k_linear,
            &mut v_linear,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} split_qkv: {e}"))?;

        let mut q_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_linear_f32 alloc: {e}"))?;
        let mut k_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_linear_f32 alloc: {e}"))?;
        let mut v_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear_f32 alloc: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            key_dim,
            &q_linear,
            &mut q_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q cast: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            key_dim,
            &k_linear,
            &mut k_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k cast: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            val_dim,
            &v_linear,
            &mut v_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} v cast: {e}"))?;

        let mut q_normed = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_normed alloc: {e}"))?;
        let mut q_scaled = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_scaled alloc: {e}"))?;
        let mut k_normed = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_normed alloc: {e}"))?;
        kernel_ffi::prefill_ffi::l2norm(
            self.ordinal,
            ScalarType::F32,
            nk,
            khd,
            1e-6,
            &q_linear_f32,
            &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q l2norm: {e}"))?;
        kernel_ffi::prefill_ffi::mul_scalar(
            self.ordinal,
            ScalarType::F32,
            key_dim,
            (khd as f32).sqrt().recip(),
            &q_normed,
            &mut q_scaled,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q scale: {e}"))?;
        kernel_ffi::prefill_ffi::l2norm(
            self.ordinal,
            ScalarType::F32,
            nk,
            khd,
            1e-6,
            &k_linear_f32,
            &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k l2norm: {e}"))?;

        let q_scaled_host = q_scaled
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} q_scaled D2H: {e}"))?;
        let k_normed_host = k_normed
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} k_normed D2H: {e}"))?;
        let v_linear_host = v_linear_f32
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear D2H: {e}"))?;
        let a_host = a
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} a D2H: {e}"))?;
        let b_host = b
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} b D2H: {e}"))?;
        let dt_bias_host = lw
            .dt_bias
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} dt_bias D2H: {e}"))?;
        let a_log_exp_host = lw
            .a_log_exp
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} a_log_exp D2H: {e}"))?;
        let q_scaled_f32: Vec<f32> = q_scaled_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let k_normed_f32: Vec<f32> = k_normed_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v_linear_f32_host: Vec<f32> = v_linear_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let a_bf16: Vec<f32> = a_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let b_bf16: Vec<f32> = b_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let dt_bias_bf16: Vec<f32> = dt_bias_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let a_log_exp_bf16: Vec<f32> = a_log_exp_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let packed_width = 2 * khd + vhd + 2;
        let mut packed_host = vec![0f32; nv * packed_width];
        for v_head in 0..nv {
            let k_head = v_head / head_repeat;
            let out_base = v_head * packed_width;
            let q_base = k_head * khd;
            let k_base = k_head * khd;
            let v_base = v_head * vhd;
            for i in 0..khd {
                packed_host[out_base + i] = q_scaled_f32[q_base + i];
                packed_host[out_base + khd + i] = k_normed_f32[k_base + i];
            }
            for i in 0..vhd {
                packed_host[out_base + 2 * khd + i] = v_linear_f32_host[v_base + i];
            }
            packed_host[out_base + 2 * khd + vhd] = 1.0f32 / (1.0f32 + (-b_bf16[v_head]).exp());
            let softplus = (1.0f32 + (a_bf16[v_head] + dt_bias_bf16[v_head]).exp()).ln();
            packed_host[out_base + 2 * khd + vhd + 1] = (-softplus * a_log_exp_bf16[v_head]).exp();
        }
        let packed = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::F32,
            &[nv, packed_width],
            &packed_host
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} packed H2D: {e}"))?;

        kernel_ffi::prefill_ffi::linear_decode_apply_4b(
            self.ordinal,
            1,
            nv,
            khd,
            vhd,
            &packed,
            recurrent_state,
            &mut rec_apply,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} linear decode apply: {e}"))?;

        let state_len = config.linear_conv_kernel_dim - 1;
        let state_bytes = ScalarType::BF16.size_in_bytes();
        let new_conv_state =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[qkv_dim, state_len])
                .map_err(|e| anyhow::anyhow!("layer {idx} new_conv_state alloc: {e}"))?;
        for c in 0..qkv_dim {
            let channel_base = c * state_len * state_bytes;
            if state_len > 1 {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    new_conv_state.offset_ptr(channel_base) as *mut c_void,
                    conv_state.offset_ptr(channel_base + state_bytes),
                    (state_len - 1) * state_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} conv shift c={c}: {e}"))?;
            }
            gpu_hal::copy_d2d(
                self.ordinal,
                new_conv_state.offset_ptr(channel_base + (state_len - 1) * state_bytes)
                    as *mut c_void,
                qkv.offset_ptr(c * state_bytes),
                state_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} conv append c={c}: {e}"))?;
        }
        gpu_hal::copy_d2d(
            self.ordinal,
            conv_state.as_ptr() as *mut c_void,
            new_conv_state.as_ptr(),
            qkv_dim * state_len * state_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} conv state update copy: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            val_dim,
            &rec_apply,
            &mut attn_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
        let attn_trace = if trace_output {
            Some(
                attn_bf16
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} attn trace D2H: {e}"))?,
            )
        } else {
            None
        };
        let rec_apply_trace = if trace_output {
            Some(
                rec_apply
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} rec_apply trace D2H: {e}"))?,
            )
        } else {
            None
        };
        gpu_hal::copy_d2d(
            self.ordinal,
            recurrent_state.as_ptr() as *mut c_void,
            rec_apply.offset_ptr(val_dim * ScalarType::F32.size_in_bytes()),
            nv * khd * vhd * ScalarType::F32.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} recurrent update copy: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            vhd,
            &lw.norm_w,
            &mut norm_w_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} norm_w cast: {e}"))?;
        kernel_ffi::prefill_ffi::rms_norm_gated(
            self.ordinal,
            ScalarType::BF16,
            nv,
            vhd,
            config.rms_norm_eps as f32,
            &attn_bf16,
            &z,
            &norm_w_bf16,
            &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;
        let gated_trace = if trace_output {
            Some(
                gated
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} gated trace D2H: {e}"))?,
            )
        } else {
            None
        };
        matmul_proj(
            self.ordinal,
            1,
            1,
            hidden_dim,
            val_dim,
            &gated,
            &lw.out_proj_w,
            lw.out_proj_scale.as_ref(),
            lw.out_proj_int8_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut proj_out,
            lw.out_proj_int4_scale.as_ref(),
            lw.out_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        let proj_trace = if trace_output {
            Some(
                proj_out
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} proj trace D2H: {e}"))?,
            )
        } else {
            None
        };
        let packed_trace = if trace_output {
            Some(
                packed
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} packed trace D2H: {e}"))?,
            )
        } else {
            None
        };
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &proj_out)?;
        Ok(if trace_output {
            Some(ComponentLinearTrace {
                qkv: qkv_trace.unwrap_or_default(),
                z: z_trace.unwrap_or_default(),
                b: b_trace.unwrap_or_default(),
                a: a_trace.unwrap_or_default(),
                packed: packed_trace.unwrap_or_default(),
                rec_apply: rec_apply_trace.unwrap_or_default(),
                attn: attn_trace.unwrap_or_default(),
                gated: gated_trace.unwrap_or_default(),
                proj_out: proj_trace.unwrap_or_default(),
            })
        } else {
            None
        })
    }

    fn component_decode_mlp_layer(
        &mut self,
        idx: usize,
        trace_output: bool,
    ) -> Result<Option<ComponentMlpTrace>> {
        let config = &self.weights.config;
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut scratch = self
            .component_mlp_scratch
            .take()
            .map(Ok)
            .unwrap_or_else(|| ComponentMlpScratch::alloc(config, self.ordinal))?;
        let lw = &self.weights.layers[idx];
        let use_cublas_mlp = self.normed_buf.backend() == gpu_hal::Backend::Cuda
            && std::env::var_os("SUPERSONIC_LLAMA31_DISABLE_CUBLAS_MLP").is_none()
            && lw.gate_proj_scale.is_none()
            && lw.gate_proj_int8_scale.is_none()
            && lw.gate_proj_int4_scale.is_none()
            && lw.up_proj_scale.is_none()
            && lw.up_proj_int8_scale.is_none()
            && lw.up_proj_int4_scale.is_none()
            && lw.down_proj_scale.is_none()
            && lw.down_proj_int8_scale.is_none()
            && lw.down_proj_int4_scale.is_none();
        let gate_up_mixed_lhs = if !use_cublas_mlp
            && (lw.gate_proj_int8_scale.is_some() || lw.up_proj_int8_scale.is_some())
        {
            prefill_engine::prepare_int8_mixed_lhs(
                self.ordinal,
                1,
                1,
                hidden_dim,
                &self.normed_buf,
                &self.weights,
            )?
        } else {
            None
        };

        if use_cublas_mlp {
            kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                self.ordinal,
                &mut scratch.gate,
                &self.normed_buf,
                &lw.gate_proj_w,
                hidden_dim,
                intermediate,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS MLP gate_proj: {e}"))?;
            kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                self.ordinal,
                &mut scratch.up,
                &self.normed_buf,
                &lw.up_proj_w,
                hidden_dim,
                intermediate,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS MLP up_proj: {e}"))?;
        } else {
            if let Some(sc) = lw.gate_proj_int8_scale.as_ref() {
                prefill_engine::matmul_int8_mixed_prepared_host(
                    self.ordinal,
                    1,
                    1,
                    intermediate,
                    hidden_dim,
                    &self.normed_buf,
                    &self.weights,
                    &format!(
                        "{}.layers.{idx}.mlp.gate_proj.weight",
                        self.weights.weight_prefix
                    ),
                    &lw.gate_proj_w,
                    sc,
                    &mut scratch.gate,
                    gate_up_mixed_lhs.as_ref(),
                )?;
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    intermediate,
                    hidden_dim,
                    &self.normed_buf,
                    &lw.gate_proj_w,
                    lw.gate_proj_scale.as_ref(),
                    lw.gate_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    &mut scratch.gate,
                    lw.gate_proj_int4_scale.as_ref(),
                    lw.gate_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }

            if lw.gate_proj_int8_scale.is_some() || lw.up_proj_int8_scale.is_some() {
                let _ = gate_up_mixed_lhs.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("layer {idx} missing prepared INT8 mixed lhs")
                })?;
            }

            if let Some(sc) = lw.up_proj_int8_scale.as_ref() {
                prefill_engine::matmul_int8_mixed_prepared_host(
                    self.ordinal,
                    1,
                    1,
                    intermediate,
                    hidden_dim,
                    &self.normed_buf,
                    &self.weights,
                    &format!(
                        "{}.layers.{idx}.mlp.up_proj.weight",
                        self.weights.weight_prefix
                    ),
                    &lw.up_proj_w,
                    sc,
                    &mut scratch.up,
                    gate_up_mixed_lhs.as_ref(),
                )?;
            } else {
                matmul_proj(
                    self.ordinal,
                    1,
                    1,
                    intermediate,
                    hidden_dim,
                    &self.normed_buf,
                    &lw.up_proj_w,
                    lw.up_proj_scale.as_ref(),
                    lw.up_proj_int8_scale.as_ref(),
                    self.weights.fp8_block_size,
                    &mut scratch.up,
                    lw.up_proj_int4_scale.as_ref(),
                    lw.up_proj_int4_zero.as_ref(),
                    self.weights.int4_group_size,
                )?;
            }
        }
        kernel_ffi::prefill_ffi::swiglu_mul(
            self.ordinal,
            ScalarType::BF16,
            intermediate,
            &scratch.gate,
            &scratch.up,
            &mut scratch.mlp,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} swiglu: {e}"))?;
        if use_cublas_mlp {
            kernel_ffi::cuda_lm_head_bf16_gemm_4b(
                self.ordinal,
                &mut scratch.down,
                &scratch.mlp,
                &lw.down_proj_w,
                intermediate,
                hidden_dim,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} cuBLAS MLP down_proj: {e}"))?;
        } else if let Some(sc) = lw.down_proj_int8_scale.as_ref() {
            prefill_engine::matmul_int8_mixed_host(
                self.ordinal,
                1,
                1,
                hidden_dim,
                intermediate,
                &scratch.mlp,
                &self.weights,
                &format!(
                    "{}.layers.{idx}.mlp.down_proj.weight",
                    self.weights.weight_prefix
                ),
                &lw.down_proj_w,
                sc,
                &mut scratch.down,
            )?;
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                hidden_dim,
                intermediate,
                &scratch.mlp,
                &lw.down_proj_w,
                lw.down_proj_scale.as_ref(),
                lw.down_proj_int8_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut scratch.down,
                lw.down_proj_int4_scale.as_ref(),
                lw.down_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
        }
        let trace = if trace_output {
            Some(ComponentMlpTrace {
                gate: scratch
                    .gate
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp gate trace D2H: {e}"))?,
                up: scratch
                    .up
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp up trace D2H: {e}"))?,
                swiglu: scratch
                    .mlp
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp swiglu trace D2H: {e}"))?,
                down: scratch
                    .down
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp down trace D2H: {e}"))?,
            })
        } else {
            None
        };
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &scratch.down)?;
        self.component_mlp_scratch = Some(scratch);
        Ok(trace)
    }

    pub fn component_trace_mlp_from_post_attn_norm(
        &mut self,
        idx: usize,
        attn_hidden_bytes: &[u8],
        post_attn_norm_bytes: &[u8],
    ) -> Result<ComponentMlpTrace> {
        let hidden_dim = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let intermediate_bytes = intermediate * ScalarType::BF16.size_in_bytes();
        anyhow::ensure!(
            attn_hidden_bytes.len() == hidden_bytes,
            "layer {idx} attn_hidden_bytes {} != expected {}",
            attn_hidden_bytes.len(),
            hidden_bytes,
        );
        anyhow::ensure!(
            post_attn_norm_bytes.len() == hidden_bytes,
            "layer {idx} post_attn_norm_bytes {} != expected {}",
            post_attn_norm_bytes.len(),
            hidden_bytes,
        );
        self.hidden_io = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            attn_hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} mlp trace attn_hidden H2D: {e}"))?;
        self.normed_buf = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            post_attn_norm_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} mlp trace post_norm H2D: {e}"))?;
        let trace = self
            .component_decode_mlp_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx} component mlp trace missing"))?;
        anyhow::ensure!(
            trace.gate.len() == intermediate_bytes
                && trace.up.len() == intermediate_bytes
                && trace.swiglu.len() == intermediate_bytes
                && trace.down.len() == hidden_bytes,
            "layer {idx} mlp trace returned unexpected sizes",
        );
        Ok(trace)
    }

    fn apply_oracle_hidden(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;

        let hidden_b64 = oracle
            .prefill_hidden
            .as_ref()
            .context("oracle output missing prefill_hidden (use --emit-state)")?;
        let hidden_bytes = b64
            .decode(hidden_b64)
            .context("decode prefill_hidden base64")?;
        let hidden_shape = oracle
            .prefill_hidden_shape
            .as_ref()
            .context("missing prefill_hidden_shape")?;
        // Oracle's tensor_to_b64 may return the full underlying storage (all tokens)
        // instead of just the last token. Take only the last token's worth of bytes.
        let expected_bytes: usize =
            hidden_shape.iter().product::<usize>() * ScalarType::BF16.size_in_bytes();
        let actual_hidden = if hidden_bytes.len() > expected_bytes {
            &hidden_bytes[hidden_bytes.len() - expected_bytes..]
        } else {
            &hidden_bytes
        };
        self.hidden_io =
            GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, hidden_shape, actual_hidden)
                .map_err(|e| anyhow::anyhow!("load prefill hidden: {e}"))?;
        Ok(())
    }

    fn apply_oracle_full_attention_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let kv_caches = oracle
            .kv_caches
            .as_ref()
            .context("oracle output missing kv_caches")?;
        for kv in kv_caches {
            let k_bytes = b64.decode(&kv.k).context("decode KV k base64")?;
            let v_bytes = b64.decode(&kv.v).context("decode KV v base64")?;
            let ls = &mut self.state.layers[kv.layer];
            ls.kv_cache_k = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &kv.k_shape, &k_bytes)
                    .map_err(|e| anyhow::anyhow!("load KV k layer {}: {e}", kv.layer))?,
            );
            ls.kv_cache_v = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &kv.v_shape, &v_bytes)
                    .map_err(|e| anyhow::anyhow!("load KV v layer {}: {e}", kv.layer))?,
            );
            ls.kv_filled = kv.k_shape[2];
        }
        Ok(())
    }

    fn apply_oracle_linear_attention_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        self.apply_oracle_conv_state(oracle)?;
        self.apply_oracle_recurrent_state(oracle)?;
        Ok(())
    }

    fn apply_oracle_conv_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let conv_states = oracle
            .conv_states
            .as_ref()
            .context("oracle output missing conv_states")?;
        for cs in conv_states {
            let bytes = b64.decode(&cs.data).context("decode conv_state base64")?;
            let ls = &mut self.state.layers[cs.layer];
            ls.conv_state = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &cs.shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("load conv_state layer {}: {e}", cs.layer))?,
            );
        }
        Ok(())
    }

    fn apply_oracle_recurrent_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let rec_states = oracle
            .recurrent_states
            .as_ref()
            .context("oracle output missing recurrent_states")?;
        for rs in rec_states {
            let bytes = b64
                .decode(&rs.data)
                .context("decode recurrent_state base64")?;
            let ls = &mut self.state.layers[rs.layer];
            ls.recurrent_state = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::F32, &rs.shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("load recurrent_state layer {}: {e}", rs.layer))?,
            );
        }
        Ok(())
    }

    pub fn new(
        weights: Qwen35Weights,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
        prefill_chunk_size: usize,
        kv_fp8: bool,
        batch_size: usize,
    ) -> Result<Self> {
        let config = &weights.config;
        let rotary =
            RotaryTables::build(config, ordinal).map_err(|e| anyhow::anyhow!("rotary: {e}"))?;
        Self::new_with_rotary(
            weights,
            rotary,
            ordinal,
            proj_buf_floats,
            attn_scratch_floats,
            kv_chunk_size,
            use_4b_kernel,
            prefill_chunk_size,
            kv_fp8,
            batch_size,
        )
    }

    pub fn new_with_rotary(
        weights: Qwen35Weights,
        rotary: RotaryTables,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
        prefill_chunk_size: usize,
        kv_fp8: bool,
        batch_size: usize,
    ) -> Result<Self> {
        let config = &weights.config;
        let state = ModelState::new(config, ordinal)
            .map_err(|e| anyhow::anyhow!("model state init: {e}"))?;

        // Create extra model states for batch items 1..batch_size
        let mut extra_states = Vec::new();
        for b in 1..batch_size {
            extra_states.push(
                ModelState::new(config, ordinal)
                    .map_err(|e| anyhow::anyhow!("model state init (batch {b}): {e}"))?,
            );
        }

        let scratch = PersistentDecodeScratch::new(
            ordinal,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            proj_buf_floats,
            attn_scratch_floats,
            batch_size,
        )
        .map_err(|e| anyhow::anyhow!("scratch init: {e}"))?;
        let hidden_io = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("hidden_io: {e}"))?;
        let normed_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("normed_buf: {e}"))?;
        let logits_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.vocab_size],
        )
        .map_err(|e| anyhow::anyhow!("logits_buf: {e}"))?;
        let argmax_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("argmax_buf: {e}"))?;
        let lm_head_block_best_vals = GpuBuffer::zeros(ordinal, ScalarType::F32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_vals: {e}"))?;
        let lm_head_block_best_idxs = GpuBuffer::zeros(ordinal, ScalarType::U32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_idxs: {e}"))?;
        let target_nll_token = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("target_nll_token: {e}"))?;
        let target_nll_value = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1])
            .map_err(|e| anyhow::anyhow!("target_nll_value: {e}"))?;
        let target_nll_accum = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1])
            .map_err(|e| anyhow::anyhow!("target_nll_accum: {e}"))?;
        let matvec_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("matvec_counter: {e}"))?;

        let fp8_scale_device = if let Some(fp8_descs) = build_fp8_scale_descs(&weights) {
            let desc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    fp8_descs.as_ptr() as *const u8,
                    fp8_descs.len() * std::mem::size_of::<kernel_ffi::FP8ScaleDesc>(),
                )
            };
            let buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[desc_bytes.len()],
                desc_bytes,
            )
            .map_err(|e| anyhow::anyhow!("upload fp8 scale descs: {e}"))?;
            Some(buf)
        } else {
            None
        };

        let int4_scale_device = if let Some(int4_descs) = build_int4_scale_descs(&weights) {
            let desc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    int4_descs.as_ptr() as *const u8,
                    int4_descs.len() * std::mem::size_of::<kernel_ffi::INT4ScaleDesc>(),
                )
            };
            let buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[desc_bytes.len()],
                desc_bytes,
            )
            .map_err(|e| anyhow::anyhow!("upload int4 scale descs: {e}"))?;
            Some(buf)
        } else {
            None
        };
        let component_full_attn_scratch =
            if use_4b_kernel && weights.layers.iter().any(|layer| layer.full.is_some()) {
                Some(ComponentFullAttentionScratch::alloc(&weights, ordinal)?)
            } else {
                None
            };
        let component_mlp_scratch = if use_4b_kernel {
            Some(ComponentMlpScratch::alloc(config, ordinal)?)
        } else {
            None
        };

        Ok(Self {
            weights,
            state,
            extra_states,
            scratch,
            rotary,
            hidden_io,
            normed_buf,
            logits_buf,
            argmax_buf,
            lm_head_block_best_vals,
            lm_head_block_best_idxs,
            target_nll_token,
            target_nll_value,
            target_nll_accum,
            matvec_counter,
            ordinal,
            kv_chunk_size,
            use_4b_kernel,
            proj_buf_floats,
            attn_scratch_floats,
            fp8_scale_device,
            int4_scale_device,
            prefill_chunk_size,
            kv_fp8,
            decode_context_limit: None,
            batch_size,
            dflash_tap_cache: None,
            dflash_fused_verify_cache: None,
            component_full_attn_scratch,
            component_mlp_scratch,
            metal_v2_scratch: None,
        })
    }

    pub fn weights(&self) -> &Qwen35Weights {
        &self.weights
    }

    pub fn set_decode_context_limit(&mut self, context_tokens: usize) {
        self.decode_context_limit = Some(context_tokens.max(1));
    }

    pub fn kv_fp8_enabled(&self) -> bool {
        self.kv_fp8
    }

    /// Verify the engine's attn_scratch budget covers the current largest
    /// `kv_max_t` across all full-attention layers (of every batch item).
    /// The 4B persistent decode kernel writes `saved_q+gate+pre_gate+scores`
    /// into attn_scratch; `saved_scores` is indexed `[qh * kv_max_b + t]`.
    fn check_attn_scratch_budget(&self) -> Result<()> {
        if !self.use_4b_kernel {
            return Ok(());
        }
        let config = &self.weights.config;
        let nh = config.num_attention_heads;
        let hd = config.head_dim;
        let base = 3 * nh * hd;
        let mut max_kv = 0usize;
        for st in std::iter::once(&self.state).chain(self.extra_states.iter()) {
            for ls in &st.layers {
                max_kv = max_kv.max(ls.kv_capacity());
            }
        }
        let required = base + nh * max_kv;
        if required > self.attn_scratch_floats {
            anyhow::bail!(
                "attn_scratch_floats={} too small for kv_max_t={} \
                 (need {} = 3*{nh}*{hd} + {nh}*{max_kv}). \
                 Pass --context-size to budget the run's max context.",
                self.attn_scratch_floats,
                max_kv,
                required,
            );
        }
        Ok(())
    }

    pub fn set_kv_fp8_for_trace(&mut self, enabled: bool) {
        self.kv_fp8 = enabled;
    }

    pub fn rotary(&self) -> &RotaryTables {
        &self.rotary
    }

    pub fn state_for_batch(&self, batch_index: usize) -> &ModelState {
        if batch_index == 0 {
            &self.state
        } else {
            &self.extra_states[batch_index - 1]
        }
    }

    pub fn certified_kv_memory_stats(&self, batch_index: usize) -> CertifiedKvMemoryStats {
        let mut stats = CertifiedKvMemoryStats::default();
        for layer in &self.state_for_batch(batch_index).layers {
            if !matches!(layer.kind, LayerKind::Full) {
                continue;
            }
            stats.full_attention_layers += 1;
            for buf in [
                layer.certified_kv_key_i8.as_ref(),
                layer.certified_kv_key_scale.as_ref(),
                layer.certified_kv_key_zero.as_ref(),
                layer.certified_kv_value_i4.as_ref(),
                layer.certified_kv_value_scale.as_ref(),
                layer.certified_kv_value_zero.as_ref(),
                layer.certified_kv_value_error.as_ref(),
                layer.certified_kv_value_norm.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                stats.tier1_compressed_vram_bytes += buf.len_bytes();
            }
            for buf in [
                layer.certified_kv_host_k.as_ref(),
                layer.certified_kv_host_v.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                stats.tier2_host_pinned_bytes += buf.len_bytes();
            }
            for buf in [
                layer.certified_kv_tail_k.as_ref(),
                layer.certified_kv_tail_v.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                stats.tail_bf16_vram_bytes += buf.len_bytes();
            }
            if let Some(buf) = layer.certified_kv_promoted_key_cache.as_ref() {
                stats.promoted_key_cache_vram_bytes += buf.len_bytes();
            }
            for buf in [
                layer.certified_kv_promoted_key_cache_tags_gpu.as_ref(),
                layer.certified_kv_promoted_key_cache_lru_gpu.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                stats.promoted_key_cache_vram_bytes += buf.len_bytes();
            }
            if let Some(buf) = layer.certified_kv_promoted_value_cache.as_ref() {
                stats.promoted_value_cache_vram_bytes += buf.len_bytes();
            }
            for buf in [
                layer.certified_kv_ranking_prefix_k.as_ref(),
                layer.certified_kv_ranking_prefix_v.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                stats.ranking_prefix_scratch_vram_bytes += buf.len_bytes();
            }
            for buf in [layer.kv_cache_k.as_ref(), layer.kv_cache_v.as_ref()]
                .into_iter()
                .flatten()
            {
                stats.dense_bf16_kv_vram_bytes += buf.len_bytes();
            }
        }
        stats
    }

    /// Load prefill state from oracle output into GPU buffers.
    pub fn load_prefill_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        self.apply_oracle_hidden(oracle)?;
        self.apply_oracle_full_attention_state(oracle)?;
        self.apply_oracle_linear_attention_state(oracle)?;

        // Convert BF16 KV caches to FP8 if requested
        if self.kv_fp8 {
            prefill_engine::convert_kv_caches_to_fp8(
                &mut self.state,
                &self.weights.config,
                self.ordinal,
            )?;
        }

        // Reset sync counters for fresh kernel launch sequence
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync: {e}"))?;

        Ok(())
    }

    /// Reset per-session state so the engine is ready for a fresh prompt.
    /// Weights, rotary tables, scratch allocations, and quantization scales are
    /// untouched — only KV caches, conv/recurrent state, and the sync counters
    /// are cleared. Used by the HTTP server between requests.
    pub fn reset(&mut self) -> Result<()> {
        self.state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("reset model state: {e}"))?;
        for es in &mut self.extra_states {
            *es = ModelState::new(&self.weights.config, self.ordinal)
                .map_err(|e| anyhow::anyhow!("reset extra state: {e}"))?;
        }
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync: {e}"))?;
        Ok(())
    }

    /// Run native GPU prefill on the prompt, returning logits for the last token.
    /// Fills KV caches, conv states, and recurrent states for subsequent decode.
    pub fn prefill_native(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        let result = prefill_engine::prefill(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            false,
            None,
        )?;

        // Reset sync counters for the decode kernel
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result.logits)
    }

    pub fn prefill_native_with_final_norm(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            false,
            None,
        )?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;
        Ok(result)
    }

    pub fn prefill_native_with_target_nll(
        &mut self,
        prompt_ids: &[u32],
        score_hidden_start: usize,
        score_targets: &[u32],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill_with_target_nll(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            score_hidden_start,
            score_targets,
            self.ordinal,
            self.kv_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
        )?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after scored prefill: {e}"))?;
        Ok(result)
    }

    /// Rebuild sequence-0 state from scratch by replaying native GPU prefill
    /// over the provided token history. Optionally replicates that state across
    /// extra batch slots for lockstep batch decoding.
    pub fn rebuild_prefill_state(
        &mut self,
        token_ids: &[u32],
        replicate_batch: bool,
    ) -> Result<Vec<f32>> {
        self.state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("rebuild model state init: {e}"))?;
        let logits = self.prefill_native(token_ids)?;
        if replicate_batch && self.batch_size > 1 {
            self.replicate_state_to_batch()?;
        }
        Ok(logits)
    }

    /// DFlash prefill: runs `prefill_with_taps` against the engine's own
    /// target state + weights, returning the regular PrefillResult with
    /// its `tap_hiddens` populated for the layers in `tap_layers`.
    pub fn prefill_native_with_taps(
        &mut self,
        prompt_ids: &[u32],
        tap_layers: &[usize],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill_with_taps(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            false,
            None,
            tap_layers,
        )?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after dflash prefill: {e}"))?;
        Ok(result)
    }

    pub fn prefill_native_with_trace(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            true,
            None,
        )?;

        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result)
    }

    fn sync_stage_if_requested(&self, enabled: bool, stage: &str) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        gpu_hal::sync(self.ordinal).map_err(|e| anyhow::anyhow!("{stage} synchronize: {e}"))
    }

    fn decode_step_non_4b(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        sampling_mode: DecodeSamplingMode,
        sync_for_timing: bool,
    ) -> Result<DecodeStepOutput> {
        // Metal: dispatch to per-op v2 incremental decode. The persistent_decode
        // megakernel below is HIP/CUDA only.
        if self.hidden_io.backend() == gpu_hal::Backend::Metal {
            return self.decode_step_non_4b_metal(token_id, seqlen_offset);
        }

        let config = &self.weights.config;
        let mut timings = DecodeStageTimings::default();

        // 1. Embedding lookup: copy one row from embed_tokens into hidden_io
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        // 2. Ensure KV capacity for full-attention layers
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.ensure_kv_capacity(
                    seqlen_offset,
                    self.ordinal,
                    config,
                    self.kv_chunk_size,
                    self.kv_fp8,
                )
                .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {i}: {e}"))?;
            }
        }
        self.check_attn_scratch_budget()?;
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(
                &self.weights.config,
                self.ordinal,
                &mut self.state,
            )?;
        }

        // 3. Build layer descriptors
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);

        // 4. Upload descriptors to device
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload descs: {e}"))?;

        // 4b. Upload KV FP8 scale descriptors (pointers may change on KV cache growth)
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("clear decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset decode sync: {e}"))?;

        // 5. Launch persistent decode kernel (dispatch by model variant)
        let start = Instant::now();
        let persist_result = if sampling_mode == DecodeSamplingMode::CudaHeroFusedLmHead {
            kernel_ffi::persistent_decode_qwen08_sm86_specialized(
                self.ordinal,
                ScalarType::BF16,
                config.num_hidden_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
            )
        } else {
            kernel_ffi::persistent_decode(
                self.ordinal,
                ScalarType::BF16,
                config.num_hidden_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
            )
        };
        persist_result.map_err(|e| anyhow::anyhow!("persistent_decode kernel: {e}"))?;
        self.sync_stage_if_requested(sync_for_timing, "persistent_decode")?;
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;

        // 6. Update KV filled counts
        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        // 7. Final RMSNorm
        let start = Instant::now();
        kernel_ffi::rms_norm(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            config.hidden_size,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm: {e}"))?;
        self.sync_stage_if_requested(sync_for_timing, "final rms_norm")?;
        timings.rms_norm_ms = start.elapsed().as_secs_f64() * 1000.0;

        match sampling_mode {
            DecodeSamplingMode::CudaHeroFusedLmHead => {
                let start = Instant::now();
                kernel_ffi::cuda_lm_head_argmax_bf16(
                    self.ordinal,
                    &self.normed_buf,
                    &*self.weights.lm_head,
                    &mut self.lm_head_block_best_vals,
                    &mut self.lm_head_block_best_idxs,
                    &mut self.argmax_buf,
                    config.hidden_size,
                    config.vocab_size,
                )
                .map_err(|e| anyhow::anyhow!("cuda fused lm_head argmax: {e}"))?;
                self.sync_stage_if_requested(sync_for_timing, "cuda fused lm_head argmax")?;
                timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

                let start = Instant::now();
                let token_bytes = self
                    .argmax_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("argmax D2H: {e}"))?;
                timings.token_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
                let sampled_token =
                    u32::from_le_bytes(token_bytes[..4].try_into().map_err(|_| {
                        anyhow::anyhow!("argmax D2H returned truncated token buffer")
                    })?);

                Ok(DecodeStepOutput {
                    logits: None,
                    sampled_token,
                    timings,
                })
            }
            DecodeSamplingMode::CudaFastGreedy | DecodeSamplingMode::HostLogits => {
                // 8. lm_head projection → logits (work-stealing matvec)
                let start = Instant::now();
                kernel_ffi::standalone_matvec(
                    self.ordinal,
                    ScalarType::BF16,
                    &mut self.logits_buf,
                    &self.normed_buf,
                    &*self.weights.lm_head,
                    config.hidden_size,
                    config.vocab_size,
                    &mut self.matvec_counter,
                )
                .map_err(|e| anyhow::anyhow!("lm_head matvec: {e}"))?;
                self.sync_stage_if_requested(sync_for_timing, "lm_head matvec")?;
                timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

                if sampling_mode == DecodeSamplingMode::CudaFastGreedy {
                    let start = Instant::now();
                    kernel_ffi::cuda_argmax_bf16(
                        self.ordinal,
                        &self.logits_buf,
                        &mut self.argmax_buf,
                        config.vocab_size,
                    )
                    .map_err(|e| anyhow::anyhow!("cuda argmax: {e}"))?;
                    self.sync_stage_if_requested(sync_for_timing, "cuda argmax")?;
                    timings.gpu_argmax_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let start = Instant::now();
                    let token_bytes = self
                        .argmax_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("argmax D2H: {e}"))?;
                    timings.token_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let sampled_token =
                        u32::from_le_bytes(token_bytes[..4].try_into().map_err(|_| {
                            anyhow::anyhow!("argmax D2H returned truncated token buffer")
                        })?);

                    return Ok(DecodeStepOutput {
                        logits: None,
                        sampled_token,
                        timings,
                    });
                }

                // 9. Copy logits to CPU and convert BF16 → F32
                let start = Instant::now();
                let logits_bytes = self
                    .logits_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
                timings.logits_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;

                let start = Instant::now();
                let logits_f32: Vec<f32> = logits_bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                let sampled_token = Self::greedy_sample(&logits_f32);
                timings.host_sampling_ms = start.elapsed().as_secs_f64() * 1000.0;

                Ok(DecodeStepOutput {
                    logits: Some(logits_f32),
                    sampled_token,
                    timings,
                })
            }
        }
    }

    /// Run one decode step and return logits on CPU. Stage timings are only
    /// populated for the non-4B native decode path.
    pub fn decode_step_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        if self.use_4b_kernel {
            return self.component_decode_step_4b_with_timings(token_id, seqlen_offset);
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::HostLogits,
            true,
        )?;
        let logits = out
            .logits
            .ok_or_else(|| anyhow::anyhow!("decode_step_with_timings missing logits"))?;
        Ok((logits, out.timings))
    }

    /// CUDA-only fast greedy path for the non-4B single-sequence decode path.
    /// Returns the sampled token without copying full logits to the host.
    pub fn decode_step_cuda_fast_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.use_4b_kernel {
            anyhow::bail!("decode_step_cuda_fast_greedy only supports the non-4B path");
        }
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("decode_step_cuda_fast_greedy requires CUDA backend");
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::CudaFastGreedy,
            false,
        )?;
        Ok((out.sampled_token, out.timings))
    }

    /// CUDA-only sm86/qwen3.5-0.8b hero path for the non-4B single-sequence decode path.
    /// Returns the sampled token without materializing logits on the host.
    pub fn decode_step_cuda_08b_hero(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.use_4b_kernel {
            anyhow::bail!("decode_step_cuda_08b_hero only supports the non-4B path");
        }
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("decode_step_cuda_08b_hero requires CUDA backend");
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::CudaHeroFusedLmHead,
            false,
        )?;
        Ok((out.sampled_token, out.timings))
    }

    /// Run one decode step. Returns logits as Vec<f32> on CPU.
    /// Metal v2 incremental decode: one length-1 forward pass per generated
    /// token, mutating the engine's `state` in place. Replaces Metal v1's
    /// O(N²) replay-prefill path. Always returns logits and a host-computed
    /// argmax; sampling_mode and sync_for_timing in the caller are ignored
    /// because v2 doesn't have GPU-side fused argmax yet (that's v3).
    fn decode_step_non_4b_metal(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<DecodeStepOutput> {
        let config = &self.weights.config;
        if self.metal_v2_scratch.is_none() {
            self.metal_v2_scratch = Some(prefill_engine::MetalV2DecodeScratch::new(
                config,
                self.ordinal,
            )?);
        }
        let scratch = self
            .metal_v2_scratch
            .as_mut()
            .expect("metal v2 scratch was just initialized");
        let logits = prefill_engine::metal_v2_decode_step(
            &self.weights,
            &mut self.state,
            &self.rotary,
            scratch,
            token_id,
            seqlen_offset,
            self.ordinal,
            self.kv_chunk_size,
        )?;
        let sampled_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        Ok(DecodeStepOutput {
            logits: Some(logits),
            sampled_token,
            timings: DecodeStageTimings::default(),
        })
    }

    pub fn decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
        if self.use_4b_kernel {
            return self.component_decode_step_4b(token_id, seqlen_offset);
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::HostLogits,
            false,
        )?;
        out.logits
            .ok_or_else(|| anyhow::anyhow!("decode_step missing logits"))
    }

    /// Backend the engine is running on. Used by callers (the server) that
    /// need to pick between the incremental [`Self::decode_step`] path and
    /// the replay-prefill path Metal v1 requires.
    pub fn backend(&self) -> gpu_hal::Backend {
        self.hidden_io.backend()
    }

    /// Replay-prefill decode: runs prefill from scratch over the full
    /// `token_history` (prompt + everything emitted so far, including the
    /// freshly sampled token whose logits we need next), and returns the
    /// last-position logits. O(N²) per generated token but reuses the
    /// validated prefill pipeline — the v1 path Metal must take because it
    /// has no megakernel and no per-op decode pipeline yet. Non-destructive
    /// to engine state (allocates a throwaway `ModelState`).
    pub fn decode_step_replay(&self, token_history: &[u32]) -> Result<Vec<f32>> {
        prefill_engine::gpu_reference_replay_step(
            &self.weights,
            &self.rotary,
            token_history,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.use_4b_kernel,
        )
    }

    /// Forced single-sequence 4B kernel path with native stage timings.
    pub fn decode_step_4b_single_kernel_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        anyhow::ensure!(
            self.use_4b_kernel,
            "decode_step_4b_single_kernel_with_timings requires 4B kernel"
        );
        anyhow::ensure!(
            self.batch_size == 1,
            "decode_step_4b_single_kernel_with_timings requires batch_size == 1"
        );

        let (mut batch_logits, mut timings) =
            self.decode_step_batch_impl(&[token_id], seqlen_offset, true)?;
        let logits = batch_logits
            .pop()
            .ok_or_else(|| anyhow::anyhow!("single-sequence 4B kernel timings missing logits"))?;
        let sampling_start = Instant::now();
        let _ = Self::greedy_sample(&logits);
        timings.host_sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
        Ok((logits, timings))
    }

    /// One decode step via the 4B persistent megakernel, capturing DFlash
    /// hidden-state taps for the specified target layers.
    ///
    /// Returns `(logits_f32, tap_hiddens_bf16_bytes)`:
    /// * `logits_f32` — `[vocab_size]` F32 logits for the next position.
    /// * `tap_hiddens_bf16_bytes` — raw BF16 bytes of shape
    ///   `[num_taps, hidden_dim]`, one row per entry in `tap_layers`.
    ///
    /// Requires `use_4b_kernel=true`, `batch_size=1`, and a non-empty
    /// `tap_layers`. Every element of `tap_layers` must be in
    /// `0..num_hidden_layers`. The tap values match what the persistent
    /// megakernel writes for each listed layer — the post-MLP residual
    /// hidden state, i.e. the same data point captured by
    /// `prefill_with_taps` / `layer_hidden_trace`.
    pub fn decode_step_with_taps_kernel(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        tap_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        if !self.use_4b_kernel {
            anyhow::bail!("decode_step_with_taps_kernel requires use_4b_kernel");
        }
        if self.batch_size != 1 {
            anyhow::bail!("decode_step_with_taps_kernel requires batch_size=1");
        }
        if tap_layers.is_empty() {
            anyhow::bail!("decode_step_with_taps_kernel requires at least one tap layer");
        }
        let config = &self.weights.config;
        let num_layers = config.num_hidden_layers;
        for &li in tap_layers {
            if li >= num_layers {
                anyhow::bail!("tap layer {li} out of range (num_hidden_layers={num_layers})");
            }
        }

        // 1) Tap workspace + i32-layer-indices: reuse the cache if tap_layers
        //    hasn't changed, otherwise (re)allocate once. DFlash calls this
        //    in a tight loop with a fixed tap_layers list, so the second+
        //    call pays zero allocation / upload cost here.
        //
        // Take the cache out of `self` into locals for the kernel call
        // (split-borrow through `Option::as_mut` conflicts with the many
        // other `&self` / `&mut self.*` borrows persistent_decode_4b needs);
        // put it back after a successful kernel launch. Kernel error paths
        // drop the cache, which is fine — next call re-allocates.
        let hidden_dim = config.hidden_size;
        let num_taps = tap_layers.len();
        let (mut tap_workspace, tap_layers_buf) = match self.dflash_tap_cache.take() {
            Some((cached, ws, lb)) if cached.as_slice() == tap_layers => (ws, lb),
            _ => {
                let workspace =
                    GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_taps, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("alloc tap_workspace: {e}"))?;
                let tap_ints: Vec<i32> = tap_layers.iter().map(|&li| li as i32).collect();
                let tap_ints_bytes: Vec<u8> =
                    tap_ints.iter().flat_map(|v| v.to_le_bytes()).collect();
                let layers_buf = GpuBuffer::from_host_bytes(
                    self.ordinal,
                    ScalarType::U8,
                    &[tap_ints_bytes.len()],
                    &tap_ints_bytes,
                )
                .map_err(|e| anyhow::anyhow!("upload tap_layers: {e}"))?;
                (workspace, layers_buf)
            }
        };

        // 2) Embedding lookup → hidden_io.
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps embedding: {e}"))?;

        // 3) Ensure KV capacity on full-attention layers.
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.ensure_kv_capacity(
                    seqlen_offset,
                    self.ordinal,
                    config,
                    self.kv_chunk_size,
                    self.kv_fp8,
                )
                .map_err(|e| anyhow::anyhow!("dflash-taps ensure KV layer {i}: {e}"))?;
            }
        }
        self.check_attn_scratch_budget()?;
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(
                &self.weights.config,
                self.ordinal,
                &mut self.state,
            )?;
        }

        // 4) Build + upload layer descs (pointers + kv_len change each step).
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("dflash-taps upload descs: {e}"))?;
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("dflash-taps upload KV FP8 descs: {e}"))?;
        }
        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("dflash-taps reset sync: {e}"))?;

        // 5) Launch the 4B megakernel with taps enabled.
        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            hidden_dim,
            config.intermediate_size,
            seqlen_offset,
            &self.scratch.desc_device,
            &mut self.hidden_io,
            &mut self.scratch.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            self.fp8_scale_device.as_ref(),
            self.scratch.kv_fp8_desc_device.as_ref(),
            1, // batch_size=1
            None,
            self.int4_scale_device.as_ref(),
            false, // enable_timing_slots
            false, // enable_attention_trace
            Some(&mut tap_workspace),
            Some(&tap_layers_buf),
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps persistent_decode_4b: {e}"))?;

        // 6) Advance kv_filled on every full-attention layer.
        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        // 7) Final RMSNorm + lm_head → logits F32.
        kernel_ffi::rms_norm_4b(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            hidden_dim,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps final rms_norm: {e}"))?;
        kernel_ffi::standalone_matvec_4b(
            self.ordinal,
            ScalarType::BF16,
            &mut self.logits_buf,
            &self.normed_buf,
            &*self.weights.lm_head,
            hidden_dim,
            config.vocab_size,
            &mut self.matvec_counter,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps lm_head matvec: {e}"))?;
        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("dflash-taps logits D2H: {e}"))?;
        let logits_f32: Vec<f32> = logits_bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();

        // 8) D2H the tap workspace, then put the workspace + layer-ids
        // back into the cache so subsequent calls with the same tap_layers
        // avoid the allocation.
        let tap_host = tap_workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("dflash-taps D2H: {e}"))?;
        self.dflash_tap_cache = Some((tap_layers.to_vec(), tap_workspace, tap_layers_buf));

        Ok((logits_f32, tap_host))
    }

    /// Mutable access to the engine's primary `ModelState`. Used by the
    /// DFlash speculative engine to snapshot/restore linear-attention state.
    pub fn state_mut(&mut self) -> &mut ModelState {
        &mut self.state
    }

    /// Device ordinal carried by the engine. Used by the DFlash engine when
    /// invoking free-function helpers (e.g. `ModelState::restore_linear`).
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Rewind every full-attention layer's `kv_filled` cursor to `new_len`
    /// (no-op if already at or below). The physical K/V beyond the cursor is
    /// untouched and will be harmlessly overwritten by subsequent decodes —
    /// used by the DFlash engine after a partial-acceptance verify to roll
    /// the cache logically back to the committed length.
    pub fn rewind_full_kv_filled(&mut self, new_len: usize) {
        let config = &self.weights.config;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) && ls.kv_filled > new_len {
                ls.set_kv_filled(new_len);
            }
        }
    }

    /// DFlash M4.3 fused verify: single `persistent_decode_4b` megakernel
    /// launch over all `tokens.len()` consecutive positions starting at
    /// `pos_offset`. Returns per-position logits `[tokens.len()][vocab]`.
    ///
    /// The megakernel's batched path already runs `B` batch elements
    /// sequentially on `blockIdx.x == 0` within a single layer iteration
    /// (see `kernels/full_attention_4b.hip` ~4165). Feeding it a
    /// `BatchSeqDesc` whose slots alias one sequence's KV cache with
    /// `seqlen_offset[b] = pos_offset + b` yields the correct causal
    /// in-sequence verify — each position reads the cache written by
    /// prior positions within the same launch.
    ///
    /// Requirements:
    /// * `use_4b_kernel = true` and `batch_size = 1` (engine construction
    ///   is not mutated; a verify-local B-sized cache is used instead).
    /// * `kv_fp8 = false` — fused verify uses BF16 KV like
    ///   `verify_block_prefill`.
    /// * `tokens.len()` must be in `1..=MAX_BATCH_SIZE` (kernel limit).
    ///
    /// Semantics match `verify_block_prefill`: full-attention K/V is
    /// written at positions `[pos_offset, pos_offset + tokens.len())`
    /// but `kv_filled` is NOT advanced on any layer — the DFlash engine
    /// owns rollback via `rewind_full_kv_filled` + `restore_linear`.
    /// Linear-attention `conv_state` / `recurrent_state` are mutated in
    /// place (shared across all B slots via pointer aliasing), so the
    /// caller MUST snapshot linear state before this call and restore
    /// it after the accept decision — same snapshot/restore contract
    /// the existing verify paths already require.
    pub fn verify_block_fused_decode(
        &mut self,
        tokens: &[u32],
        pos_offset: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if !self.use_4b_kernel {
            anyhow::bail!("verify_block_fused_decode requires use_4b_kernel");
        }
        if self.batch_size != 1 {
            anyhow::bail!("verify_block_fused_decode requires engine batch_size=1");
        }
        if self.kv_fp8 {
            anyhow::bail!("verify_block_fused_decode does not support kv_fp8");
        }
        if tokens.is_empty() {
            anyhow::bail!("verify_block_fused_decode: tokens must be non-empty");
        }
        let b = tokens.len();
        if b > kernel_ffi::MAX_BATCH_SIZE {
            anyhow::bail!(
                "verify_block_fused_decode: block size {b} > MAX_BATCH_SIZE {}",
                kernel_ffi::MAX_BATCH_SIZE,
            );
        }

        // Copy out primitive config values up front so the later
        // `self.state.layers.iter_mut()` borrow doesn't fight with
        // `&self.weights.config` reads.
        let (hidden_dim, intermediate_size, vocab_size, num_layers, rms_norm_eps) = {
            let c = &self.weights.config;
            (
                c.hidden_size,
                c.intermediate_size,
                c.vocab_size,
                c.num_hidden_layers,
                c.rms_norm_eps as f32,
            )
        };

        // The 4B megakernel's shared-memory footprint per workgroup is
        //   (block_size + max(B × hidden_dim, intermediate_size) + fp8_lut) × sizeof(f32)
        // with kernel block_size = 256 and fp8_lut = 256. gfx1150 caps
        // LDS at 64 KiB per workgroup → 16384 floats total. Reserve 2
        // KiB (512 floats) for block_size + fp8_lut, leaving 15872
        // floats for the input cache. 9B (hidden=4096) tops out at B=3;
        // 4B (hidden=2048) tops out at B=7. If a user passes a larger
        // --dflash-block the launch fails with HIP status 254 and a
        // confusing error — fail fast here instead with the math
        // spelled out.
        const MAX_INPUT_CACHE_FLOATS: usize = 15872;
        let input_cache = (b * hidden_dim).max(intermediate_size);
        if input_cache > MAX_INPUT_CACHE_FLOATS {
            anyhow::bail!(
                "verify_block_fused_decode: shared-memory budget exceeded \
                 (B={b} × hidden_dim={hidden_dim} = {}, intermediate={intermediate_size}; \
                 cap = {MAX_INPUT_CACHE_FLOATS} floats). \
                 Lower --dflash-block to ≤ {}.",
                b * hidden_dim,
                MAX_INPUT_CACHE_FLOATS.min(b * hidden_dim) / hidden_dim.max(1),
            );
        }

        let max_pos = pos_offset + b - 1;

        // Ensure KV capacity on every full-attention layer for the
        // highest position this launch will write.
        {
            let config = &self.weights.config;
            for (i, ls) in self.state.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        max_pos,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| anyhow::anyhow!("fused verify ensure KV layer {i}: {e}"))?;
                }
            }
        }
        self.check_attn_scratch_budget()?;

        // Take the cached workspace if it matches the current block
        // size, otherwise allocate fresh. Put it back at the end.
        let mut cache = match self.dflash_fused_verify_cache.take() {
            Some(c) if c.block_size == b => c,
            _ => DFlashFusedVerifyCache::alloc(
                self.ordinal,
                b,
                hidden_dim,
                intermediate_size,
                vocab_size,
                num_layers,
                self.proj_buf_floats,
                self.attn_scratch_floats,
            )?,
        };

        // Layer descs (state pointers are ignored by the kernel when
        // `batch_descs` is non-null — weights + norm pointers still
        // matter). Reuse `self.scratch.desc_device` to avoid a second
        // device allocation; the scratch is not otherwise touched by
        // this method.
        let descs = build_layer_descs(&self.weights, &self.state, pos_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("fused verify upload layer descs: {e}"))?;

        // Shared-cache batch-seq descriptors: all B slots point at
        // `self.state`'s per-layer buffers; `seqlen_offset[b] =
        // pos_offset + b` gives the kernel the unique per-position
        // offset for RoPE + KV append + causal read.
        let state_refs: Vec<&ModelState> = (0..b).map(|_| &self.state).collect();
        let seqlen_offsets: Vec<usize> = (0..b).map(|bi| pos_offset + bi).collect();
        let batch_descs =
            build_batch_seq_descs(&state_refs, &seqlen_offsets, /* kv_fp8 */ false).ok_or_else(
                || anyhow::anyhow!("fused verify: build_batch_seq_descs returned None for B={b}"),
            )?;
        let desc_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                batch_descs.as_ptr() as *const u8,
                batch_descs.len() * std::mem::size_of::<kernel_ffi::BatchSeqDesc>(),
            )
        };
        gpu_hal::copy_h2d(
            self.ordinal,
            cache.batch_desc_device.as_mut_ptr(),
            desc_bytes.as_ptr() as *const c_void,
            desc_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("fused verify upload batch-seq descs: {e}"))?;

        // Embedding lookup: gather each token's row into
        // cache.hidden_io[b, 0, :].
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in tokens.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (cache.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("fused verify embedding slot {bi}: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            cache.workspace.as_mut_ptr(),
            cache.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("fused verify clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("fused verify reset sync: {e}"))?;

        // Launch the fused megakernel. `pos_offset` as the kernel's
        // `seqlen_offset` arg is ignored because `batch_descs` is
        // non-null; pass it through for consistency with the batched
        // call site.
        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            hidden_dim,
            intermediate_size,
            pos_offset,
            &self.scratch.desc_device,
            &mut cache.hidden_io,
            &mut cache.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            self.fp8_scale_device.as_ref(),
            None, // kv_fp8_descs: fused verify disallows kv_fp8
            b,
            Some(&cache.batch_desc_device),
            self.int4_scale_device.as_ref(),
            false, // enable_timing_slots
            false, // enable_attention_trace
            None,  // tap_workspace: verify doesn't capture taps — re-decode does
            None,  // tap_layers: ignored when tap_workspace is None
        )
        .map_err(|e| anyhow::anyhow!("fused verify persistent_decode_4b: {e}"))?;

        // Deliberately do NOT advance `kv_filled` on any layer. The
        // DFlash engine rolls the K/V cursor back via
        // `rewind_full_kv_filled` and the linear state via
        // `restore_linear` after the accept decision.

        // Final RMSNorm (multirow) + tiled lm_head over all B hiddens.
        kernel_ffi::rms_norm_4b_multirow(
            self.ordinal,
            ScalarType::BF16,
            b,
            hidden_dim,
            rms_norm_eps,
            &cache.hidden_io,
            &self.weights.norm_weight,
            &mut cache.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("fused verify final rms_norm: {e}"))?;

        kernel_ffi::matmul_rhs_transposed_4b(
            self.ordinal,
            ScalarType::BF16,
            1,
            b,
            vocab_size,
            hidden_dim,
            &cache.normed_buf,
            &*self.weights.lm_head,
            &mut cache.logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("fused verify lm_head matmul: {e}"))?;

        let logits_host = cache
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("fused verify logits D2H: {e}"))?;
        let row_stride_bytes = vocab_size * ScalarType::BF16.size_in_bytes();
        let mut logits_per_pos = Vec::with_capacity(b);
        for bi in 0..b {
            let start = bi * row_stride_bytes;
            let end = start + row_stride_bytes;
            let row: Vec<f32> = logits_host[start..end]
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            logits_per_pos.push(row);
        }

        self.dflash_fused_verify_cache = Some(cache);
        Ok(logits_per_pos)
    }

    /// Greedy argmax over logits.
    pub fn greedy_sample(logits: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_idx = idx;
                best_val = val;
            }
        }
        best_idx as u32
    }

    pub fn last_normed_host_f32(&self) -> Result<Vec<f32>> {
        let bytes = self
            .normed_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("normed D2H: {e}"))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }

    /// Copy prefill state from sequence 0 to all extra batch sequences.
    /// Call after load_prefill_state() or prefill_native() to initialize batch items.
    pub fn replicate_state_to_batch(&mut self) -> Result<()> {
        for b in 0..self.extra_states.len() {
            self.extra_states[b] = self
                .state
                .clone_gpu()
                .map_err(|e| anyhow::anyhow!("clone state to batch {}: {e}", b + 1))?;
        }
        Ok(())
    }

    /// Run one batched decode step. Returns per-sequence logits.
    /// `token_ids`: one token per batch item.
    /// `seqlen_offset`: shared sequence position (all sequences advance in lockstep).
    pub fn decode_step_batch(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (all_logits, _) = self.decode_step_batch_impl(token_ids, seqlen_offset, false)?;
        Ok(all_logits)
    }

    /// Run one batched decode step and return per-sequence logits plus native
    /// stage timings for the persistent batch path.
    pub fn decode_step_batch_with_timings(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        self.decode_step_batch_impl(token_ids, seqlen_offset, true)
    }

    fn decode_step_batch_impl(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        enable_timing_slots: bool,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        assert_eq!(token_ids.len(), self.batch_size);
        assert!(self.use_4b_kernel, "batched decode requires 4b kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        let use_qwen35_4b_cuda_long_context_component_fallback = self.hidden_io.backend()
            == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8
            && qwen35_4b_cuda_long_context_component_fallback_enabled()
            && seqlen_offset >= QWEN35_4B_CUDA_COMPONENT_FALLBACK_TOKENS;
        if use_qwen35_4b_cuda_long_context_component_fallback {
            let logits = self.component_decode_step_4b(token_ids[0], seqlen_offset)?;
            return Ok((vec![logits], DecodeStageTimings::default()));
        }
        let mut timings = DecodeStageTimings::default();
        let dump_layer_timings_topn = if enable_timing_slots
            && self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
        {
            qwen35_4b_cuda_dump_layer_timings_topn().and_then(|topn| {
                QWEN35_4B_CUDA_LAYER_TIMINGS_DUMPED
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .ok()
                    .map(|_| topn)
            })
        } else {
            None
        };
        let mut persistent_layer_timings = dump_layer_timings_topn
            .map(|_| vec![Persistent4BLayerTiming::default(); config.num_hidden_layers]);

        // 1. Embedding lookup: place each sequence's embedding at offset b * hidden_size
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in token_ids.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("embedding lookup batch {bi}: {e}"))?;
        }

        // 2. Ensure KV capacity for all batch items
        let seqlen_offsets: Vec<usize> = vec![seqlen_offset; b];
        for bi in 0..b {
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        seqlen_offset,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity batch {bi} layer {i}: {e}"))?;
                }
            }
        }
        self.check_attn_scratch_budget()?;
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(
                &self.weights.config,
                self.ordinal,
                &mut self.state,
            )?;
            for bi in 0..self.extra_states.len() {
                Self::load_kv_shadow_for_state_static(
                    &self.weights.config,
                    self.ordinal,
                    &mut self.extra_states[bi],
                )?;
            }
        }

        // 3. Build layer descriptors (weights only, per-sequence state in batch descs)
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload descs: {e}"))?;

        // 4. Build and upload batch sequence descriptors
        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8)
        {
            self.scratch
                .upload_batch_seq_descs(&batch_descs)
                .map_err(|e| anyhow::anyhow!("upload batch seq descs: {e}"))?;
        }

        // 4b. Upload KV FP8 scale descriptors
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("clear batched decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset batched decode sync: {e}"))?;

        // 5. Launch batched persistent decode kernel
        let timing_calibration = if enable_timing_slots {
            Some(match self.hidden_io.backend() {
                gpu_hal::Backend::Cuda => {
                    let khz = gpu_hal::query_device_info(gpu_hal::Backend::Cuda, self.ordinal)
                        .map_err(|e| anyhow::anyhow!("query CUDA device clock rate: {e}"))?
                        .clock_rate_khz;
                    PersistentTimingCalibration::ClockRateKhz(khz)
                }
                gpu_hal::Backend::Hip => PersistentTimingCalibration::WallClockMs(0.0),
                gpu_hal::Backend::Metal => PersistentTimingCalibration::WallClockMs(0.0),
            })
        } else {
            None
        };
        // The CUDA single-stream hero specialization is only validated for the
        // exact Qwen3.5-4B geometry. 2B/9B share the generic persistent path,
        // but routing them through the specialized launch changes numerics.
        let use_qwen35_4b_cuda_hero = self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
            && qwen35_4b_cuda_hero_enabled()
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8;
        let use_qwen35_4b_cuda_split = self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8
            && config.num_hidden_layers > QWEN35_4B_CUDA_SPLIT_LAYER;
        let mut persistent_kernel_ms = 0.0;
        if use_qwen35_4b_cuda_split {
            let windows = qwen35_4b_cuda_split_windows(config.num_hidden_layers);
            for (window_start, window_layers) in windows {
                self.scratch
                    .upload_descs(&descs[window_start..window_start + window_layers])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "upload descs for split window [{window_start}, {}): {e}",
                            window_start + window_layers
                        )
                    })?;
                gpu_hal::memset_zeros(
                    self.ordinal,
                    self.scratch.workspace.as_mut_ptr(),
                    self.scratch.workspace.len_bytes(),
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "clear split decode workspace [{window_start}, {}): {e}",
                        window_start + window_layers
                    )
                })?;
                self.scratch.reset_sync().map_err(|e| {
                    anyhow::anyhow!(
                        "reset split decode sync [{window_start}, {}): {e}",
                        window_start + window_layers
                    )
                })?;

                let window_launch_start = Instant::now();
                let window_result = if use_qwen35_4b_cuda_hero {
                    kernel_ffi::persistent_decode_4b_qwen35_sm86_specialized(
                        self.ordinal,
                        ScalarType::BF16,
                        window_layers,
                        config.hidden_size,
                        config.intermediate_size,
                        seqlen_offset,
                        &self.scratch.desc_device,
                        &mut self.hidden_io,
                        &mut self.scratch.workspace,
                        &mut self.scratch.sync_buf,
                        &self.rotary.cos,
                        &self.rotary.sin,
                        self.rotary.rotary_dim,
                        self.proj_buf_floats,
                        self.attn_scratch_floats,
                        self.fp8_scale_device.as_ref(),
                        None,
                        1,
                        None,
                        self.int4_scale_device.as_ref(),
                        enable_timing_slots,
                        false,
                        None,
                        None,
                    )
                } else {
                    kernel_ffi::persistent_decode_4b(
                        self.ordinal,
                        ScalarType::BF16,
                        window_layers,
                        config.hidden_size,
                        config.intermediate_size,
                        seqlen_offset,
                        &self.scratch.desc_device,
                        &mut self.hidden_io,
                        &mut self.scratch.workspace,
                        &mut self.scratch.sync_buf,
                        &self.rotary.cos,
                        &self.rotary.sin,
                        self.rotary.rotary_dim,
                        self.proj_buf_floats,
                        self.attn_scratch_floats,
                        self.fp8_scale_device.as_ref(),
                        None,
                        1,
                        None,
                        self.int4_scale_device.as_ref(),
                        enable_timing_slots,
                        false,
                        None,
                        None,
                    )
                };
                window_result.map_err(|e| {
                    anyhow::anyhow!(
                        "persistent_decode_4b split window [{window_start}, {}): {e}",
                        window_start + window_layers
                    )
                })?;
                persistent_kernel_ms += window_launch_start.elapsed().as_secs_f64() * 1000.0;

                if let Some(PersistentTimingCalibration::ClockRateKhz(clock_rate_khz)) =
                    timing_calibration
                {
                    let sync_bytes = self.scratch.sync_buf.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!(
                            "split timing slots D2H [{window_start}, {}): {e}",
                            window_start + window_layers
                        )
                    })?;
                    timings.add_assign(decode_persistent_4b_timing_slots(
                        &sync_bytes,
                        window_layers,
                        1,
                        PersistentTimingCalibration::ClockRateKhz(clock_rate_khz),
                        persistent_layer_timings.as_deref_mut(),
                        window_start,
                    ));
                }
            }
            self.scratch
                .upload_descs(&descs)
                .map_err(|e| anyhow::anyhow!("restore full descs after split decode: {e}"))?;
        } else {
            let persist_start = Instant::now();
            let persist_result = if use_qwen35_4b_cuda_hero {
                kernel_ffi::persistent_decode_4b_qwen35_sm86_specialized(
                    self.ordinal,
                    ScalarType::BF16,
                    config.num_hidden_layers,
                    config.hidden_size,
                    config.intermediate_size,
                    seqlen_offset,
                    &self.scratch.desc_device,
                    &mut self.hidden_io,
                    &mut self.scratch.workspace,
                    &mut self.scratch.sync_buf,
                    &self.rotary.cos,
                    &self.rotary.sin,
                    self.rotary.rotary_dim,
                    self.proj_buf_floats,
                    self.attn_scratch_floats,
                    self.fp8_scale_device.as_ref(),
                    self.scratch.kv_fp8_desc_device.as_ref(),
                    b,
                    self.scratch.batch_seq_desc_device.as_ref(),
                    self.int4_scale_device.as_ref(),
                    enable_timing_slots,
                    false,
                    None, // tap_workspace: DFlash-only, off in batched decode
                    None, // tap_layers: DFlash-only, off in batched decode
                )
            } else {
                kernel_ffi::persistent_decode_4b(
                    self.ordinal,
                    ScalarType::BF16,
                    config.num_hidden_layers,
                    config.hidden_size,
                    config.intermediate_size,
                    seqlen_offset,
                    &self.scratch.desc_device,
                    &mut self.hidden_io,
                    &mut self.scratch.workspace,
                    &mut self.scratch.sync_buf,
                    &self.rotary.cos,
                    &self.rotary.sin,
                    self.rotary.rotary_dim,
                    self.proj_buf_floats,
                    self.attn_scratch_floats,
                    self.fp8_scale_device.as_ref(),
                    self.scratch.kv_fp8_desc_device.as_ref(),
                    b,
                    self.scratch.batch_seq_desc_device.as_ref(),
                    self.int4_scale_device.as_ref(),
                    enable_timing_slots,
                    false,
                    None, // tap_workspace: DFlash-only, off in batched decode
                    None, // tap_layers: DFlash-only, off in batched decode
                )
            };
            persist_result
                .map_err(|e| anyhow::anyhow!("persistent_decode_4b batch kernel: {e}"))?;
            persistent_kernel_ms = persist_start.elapsed().as_secs_f64() * 1000.0;

            if let Some(calibration) = timing_calibration {
                let calibration = match calibration {
                    PersistentTimingCalibration::ClockRateKhz(khz) => {
                        PersistentTimingCalibration::ClockRateKhz(khz)
                    }
                    PersistentTimingCalibration::WallClockMs(_) => {
                        PersistentTimingCalibration::WallClockMs(persistent_kernel_ms)
                    }
                };
                let sync_bytes = self
                    .scratch
                    .sync_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("persistent timing slots D2H: {e}"))?;
                timings.add_assign(decode_persistent_4b_timing_slots(
                    &sync_bytes,
                    config.num_hidden_layers,
                    b,
                    calibration,
                    persistent_layer_timings.as_deref_mut(),
                    0,
                ));
            }
        }
        if let (Some(topn), Some(layer_timings)) =
            (dump_layer_timings_topn, persistent_layer_timings.as_deref())
        {
            maybe_dump_qwen35_4b_layer_timings(layer_timings, topn, seqlen_offset, b);
        }
        timings.persistent_ms = persistent_kernel_ms;

        // 6. Update KV filled counts for all batch items
        let filled = seqlen_offset + 1;
        for bi in 0..b {
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.set_kv_filled(filled);
                }
            }
        }

        // 7-9. Final multi-row RMSNorm + tiled lm_head matmul, then one D2H.
        let start = Instant::now();
        kernel_ffi::rms_norm_4b_multirow(
            self.ordinal,
            ScalarType::BF16,
            b,
            config.hidden_size,
            config.rms_norm_eps as f32,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm batch rows: {e}"))?;
        timings.rms_norm_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        kernel_ffi::matmul_rhs_transposed_4b(
            self.ordinal,
            ScalarType::BF16,
            1,
            b,
            config.vocab_size,
            config.hidden_size,
            &self.normed_buf,
            &*self.weights.lm_head,
            &mut self.logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("tiled lm_head batch matmul: {e}"))?;
        timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let logits_host = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H batch rows: {e}"))?;
        timings.logits_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
        let row_bytes = config.vocab_size * ScalarType::BF16.size_in_bytes();
        let mut all_logits = Vec::with_capacity(b);
        for bi in 0..b {
            let start = bi * row_bytes;
            let end = start + row_bytes;
            let logits_f32: Vec<f32> = logits_host[start..end]
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect();
            all_logits.push(logits_f32);
        }

        Ok((all_logits, timings))
    }

    /// Debug-only: run the real batched 4B persistent kernel for the first `num_layers`
    /// layers and return one batch row of the resulting hidden state as BF16 bytes.
    /// This mutates the decode state; callers should rebuild state afterwards if they
    /// need to continue from the pre-trace state.
    pub fn decode_step_batch_trace_hidden_after_layers(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        num_layers: usize,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        assert_eq!(token_ids.len(), self.batch_size);
        assert!(self.use_4b_kernel, "persistent trace requires 4b kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        anyhow::ensure!(
            num_layers <= config.num_hidden_layers,
            "trace layer count {} exceeds model layers {}",
            num_layers,
            config.num_hidden_layers
        );
        anyhow::ensure!(
            batch_index < b,
            "trace batch index {} out of range for batch {}",
            batch_index,
            b
        );

        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in token_ids.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("trace embedding lookup batch {bi}: {e}"))?;
        }

        let seqlen_offsets: Vec<usize> = vec![seqlen_offset; b];
        for bi in 0..b {
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        seqlen_offset,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("trace ensure KV capacity batch {bi} layer {i}: {e}")
                    })?;
                }
            }
        }

        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("trace upload descs: {e}"))?;

        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8)
        {
            self.scratch
                .upload_batch_seq_descs(&batch_descs)
                .map_err(|e| anyhow::anyhow!("trace upload batch seq descs: {e}"))?;
        }

        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("trace upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("trace clear batched decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("trace reset batched decode sync: {e}"))?;

        let use_qwen35_4b_cuda_hero = self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
            && qwen35_4b_cuda_hero_enabled()
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8;
        let persist_result = if use_qwen35_4b_cuda_hero {
            kernel_ffi::persistent_decode_4b_qwen35_sm86_specialized(
                self.ordinal,
                ScalarType::BF16,
                num_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
                self.proj_buf_floats,
                self.attn_scratch_floats,
                self.fp8_scale_device.as_ref(),
                self.scratch.kv_fp8_desc_device.as_ref(),
                b,
                self.scratch.batch_seq_desc_device.as_ref(),
                self.int4_scale_device.as_ref(),
                false,
                true,
                None, // tap_workspace: DFlash-only, off in trace path
                None, // tap_layers: DFlash-only, off in trace path
            )
        } else {
            kernel_ffi::persistent_decode_4b(
                self.ordinal,
                ScalarType::BF16,
                num_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
                self.proj_buf_floats,
                self.attn_scratch_floats,
                self.fp8_scale_device.as_ref(),
                self.scratch.kv_fp8_desc_device.as_ref(),
                b,
                self.scratch.batch_seq_desc_device.as_ref(),
                self.int4_scale_device.as_ref(),
                false,
                true,
                None, // tap_workspace: DFlash-only, off in trace path
                None, // tap_layers: DFlash-only, off in trace path
            )
        };
        persist_result
            .map_err(|e| anyhow::anyhow!("trace persistent_decode_4b batch kernel: {e}"))?;

        let hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace hidden D2H: {e}"))?;
        let start = batch_index * row_bytes;
        let end = start + row_bytes;
        Ok(hidden[start..end].to_vec())
    }

    /// Debug-only: run a BF16 4B persistent-kernel window over a sliced layer
    /// descriptor range starting from a caller-supplied hidden row.
    pub fn debug_decode_window_from_hidden_bf16(
        &mut self,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
        start_layer: usize,
        num_layers: usize,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(self.batch_size == 1, "debug window requires batch_size=1");
        anyhow::ensure!(self.use_4b_kernel, "debug window requires 4b kernel");
        anyhow::ensure!(self.fp8_scale_device.is_none(), "debug window is BF16-only");
        anyhow::ensure!(
            self.int4_scale_device.is_none(),
            "debug window is BF16-only"
        );
        anyhow::ensure!(!self.kv_fp8, "debug window does not support kv_fp8");

        let config = self.weights.config.clone();
        anyhow::ensure!(
            start_layer < config.num_hidden_layers,
            "debug window start_layer {} exceeds model layers {}",
            start_layer,
            config.num_hidden_layers
        );
        anyhow::ensure!(
            start_layer + num_layers <= config.num_hidden_layers,
            "debug window [{}, {}) exceeds model layers {}",
            start_layer,
            start_layer + num_layers,
            config.num_hidden_layers
        );
        anyhow::ensure!(
            hidden_bytes.len() == config.hidden_size * ScalarType::BF16.size_in_bytes(),
            "debug window hidden len {} != hidden row bytes {}",
            hidden_bytes.len(),
            config.hidden_size * ScalarType::BF16.size_in_bytes()
        );
        anyhow::ensure!(
            batch_index < self.batch_size,
            "debug window batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );

        self.set_hidden_from_bytes(hidden_bytes)?;

        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        let window_descs = descs
            .get(start_layer..start_layer + num_layers)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "debug window desc slice [{start_layer}, {}) missing",
                    start_layer + num_layers
                )
            })?;
        self.scratch
            .upload_descs(window_descs)
            .map_err(|e| anyhow::anyhow!("debug window upload descs: {e}"))?;

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("debug window clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("debug window reset sync: {e}"))?;

        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            config.hidden_size,
            config.intermediate_size,
            seqlen_offset,
            &self.scratch.desc_device,
            &mut self.hidden_io,
            &mut self.scratch.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            None,
            None,
            self.batch_size,
            None,
            None,
            false,
            false,
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("debug window persistent_decode_4b: {e}"))?;

        let hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("debug window hidden D2H: {e}"))?;
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        let start = batch_index * row_bytes;
        let end = start + row_bytes;
        Ok(hidden[start..end].to_vec())
    }

    /// Debug-only: run the real 4B persistent decode kernel for the first
    /// `num_layers` layers and export one selected linear layer/channel's
    /// post-Step-B conv-state taps as F32 bytes.
    ///
    /// Output layout is `[state_len taps..., qkv_bf16, conv_out_bf16]`, all
    /// widened to F32 by the kernel.
    pub fn trace_persistent_linear_step_b_after_layers(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        num_layers: usize,
        trace_layer: usize,
        trace_channel: usize,
    ) -> Result<Vec<u8>> {
        assert_eq!(token_ids.len(), self.batch_size);
        assert!(self.use_4b_kernel, "persistent trace requires 4b kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        anyhow::ensure!(
            num_layers <= config.num_hidden_layers,
            "trace layer count {} exceeds model layers {}",
            num_layers,
            config.num_hidden_layers
        );
        anyhow::ensure!(
            trace_layer < num_layers,
            "trace layer {} out of range for num_layers {}",
            trace_layer,
            num_layers
        );
        anyhow::ensure!(
            !config.is_full_attention(trace_layer),
            "trace layer {} is not linear-attention",
            trace_layer
        );

        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        anyhow::ensure!(
            trace_channel < qkv_dim,
            "trace channel {} out of range for qkv_dim {}",
            trace_channel,
            qkv_dim
        );

        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in token_ids.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("linear step-b trace embedding lookup batch {bi}: {e}"))?;
        }

        let seqlen_offsets: Vec<usize> = vec![seqlen_offset; b];
        for bi in 0..b {
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        seqlen_offset,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "linear step-b trace ensure KV capacity batch {bi} layer {i}: {e}"
                        )
                    })?;
                }
            }
        }

        let state_len = config.linear_conv_kernel_dim - 1;
        let mut debug_buf = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[state_len + 2])
            .map_err(|e| anyhow::anyhow!("alloc linear step-b debug buffer: {e}"))?;

        let mut descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        descs[trace_layer].debug_linear_trace_out = debug_buf.as_mut_ptr();
        descs[trace_layer].debug_linear_trace_channel = trace_channel as i32;
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("linear step-b trace upload descs: {e}"))?;

        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8)
        {
            self.scratch
                .upload_batch_seq_descs(&batch_descs)
                .map_err(|e| anyhow::anyhow!("linear step-b trace upload batch seq descs: {e}"))?;
        }

        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("linear step-b trace upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("linear step-b trace clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("linear step-b trace reset sync: {e}"))?;

        let use_qwen35_4b_cuda_hero = self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_shape(config)
            && qwen35_4b_cuda_hero_enabled()
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8;
        let persist_result = if use_qwen35_4b_cuda_hero {
            kernel_ffi::persistent_decode_4b_qwen35_sm86_specialized(
                self.ordinal,
                ScalarType::BF16,
                num_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
                self.proj_buf_floats,
                self.attn_scratch_floats,
                self.fp8_scale_device.as_ref(),
                self.scratch.kv_fp8_desc_device.as_ref(),
                b,
                self.scratch.batch_seq_desc_device.as_ref(),
                self.int4_scale_device.as_ref(),
                false,
                false,
                None,
                None,
            )
        } else {
            kernel_ffi::persistent_decode_4b(
                self.ordinal,
                ScalarType::BF16,
                num_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
                self.proj_buf_floats,
                self.attn_scratch_floats,
                self.fp8_scale_device.as_ref(),
                self.scratch.kv_fp8_desc_device.as_ref(),
                b,
                self.scratch.batch_seq_desc_device.as_ref(),
                self.int4_scale_device.as_ref(),
                false,
                false,
                None,
                None,
            )
        };
        persist_result
            .map_err(|e| anyhow::anyhow!("linear step-b trace persistent_decode_4b kernel: {e}"))?;

        debug_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("linear step-b trace D2H: {e}"))
    }

    pub fn trace_persistent_linear_proj_buf_after_layers(
        &self,
        batch_index: usize,
        qkv_dim: usize,
        z_dim: usize,
        nv: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let prefix_floats = self.weights.config.hidden_size
            + self.weights.config.hidden_size
            + self.weights.config.intermediate_size * 2
            + self.weights.config.hidden_size
            + self.weights.config.hidden_size;
        let start_floats = prefix_floats * self.batch_size + batch_index * self.proj_buf_floats;
        let total_floats = qkv_dim + z_dim + nv + nv;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = start_floats * ScalarType::F32.size_in_bytes();
        let end = start + total_floats * ScalarType::F32.size_in_bytes();
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent projection slice out of bounds"
        );
        let slice = &bytes[start..end];
        let qkv_end = qkv_dim * 4;
        let z_end = qkv_end + z_dim * 4;
        let b_end = z_end + nv * 4;
        Ok((
            slice[..qkv_end].to_vec(),
            slice[qkv_end..z_end].to_vec(),
            slice[z_end..b_end].to_vec(),
            slice[b_end..].to_vec(),
        ))
    }

    pub fn trace_persistent_mlp_stage_after_layers(
        &self,
        batch_index: usize,
        intermediate: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let normed_start = (normed_base + batch_index * hidden) * 4;
        let normed_end = normed_start + hidden * 4;
        let gate_start = (gate_up_base + batch_index * intermediate * 2) * 4;
        let gate_end = gate_start + intermediate * 4;
        let mlp_out_start = (mlp_out_base + batch_index * hidden) * 4;
        let mlp_out_end = mlp_out_start + hidden * 4;
        let token_out_start = (token_out_base + batch_index * hidden) * 4;
        let token_out_end = token_out_start + hidden * 4;
        anyhow::ensure!(
            token_out_end <= bytes.len(),
            "persistent MLP slice out of bounds"
        );
        Ok((
            bytes[normed_start..normed_end].to_vec(),
            bytes[gate_start..gate_end].to_vec(),
            bytes[mlp_out_start..mlp_out_end].to_vec(),
            bytes[token_out_start..token_out_end].to_vec(),
        ))
    }

    pub fn trace_persistent_linear_gated_after_layers(
        &self,
        batch_index: usize,
        value_dim: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats) * 4;
        let end = start + value_dim * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent gated slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_gated_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (proj_buf_base + batch_index * self.proj_buf_floats) * 4;
        let end = start + hidden * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn gated slice out of bounds"
        );
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_saved_gate_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn saved_gate slice out of bounds"
        );
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_q_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn q slice out of bounds"
        );
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_pre_gate_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim * 2) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn pre_gate slice out of bounds"
        );
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_scores_after_layers(
        &self,
        batch_index: usize,
        kv_len: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let score_cols = self.attn_scratch_floats.saturating_sub(q_dim * 3)
            / self.weights.config.num_attention_heads;
        anyhow::ensure!(
            kv_len <= score_cols,
            "persistent full-attn scores kv_len {} exceeds scratch score columns {}",
            kv_len,
            score_cols,
        );
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim * 3) * 4;
        let end = start + self.weights.config.num_attention_heads * score_cols * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn scores slice out of bounds"
        );
        let full = &bytes[start..end];
        let mut out = Vec::with_capacity(self.weights.config.num_attention_heads * kv_len * 4);
        let stride = score_cols * 4;
        for h in 0..self.weights.config.num_attention_heads {
            let row = h * stride;
            out.extend_from_slice(&full[row..row + kv_len * 4]);
        }
        Ok(out)
    }
}
