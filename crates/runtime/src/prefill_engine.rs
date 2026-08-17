//! Native GPU prefill engine — replaces the Python oracle.
//!
//! Orchestrates component kernels (embedding, matmul, attention, conv, recurrence,
//! norms, MLP) to process the entire prompt sequence through the model on GPU.

use std::{env, ffi::c_void};

use anyhow::Result;
use gpu_hal::{copy_h2d, Backend, GpuBuffer, ScalarType};
use half::{bf16, f16};

use qwen35::config::TextConfig;
use qwen35::rotary::RotaryTables;
use qwen35::state::{kv_fp8_bf16_sidecar_enabled, kv_fp8_bf16_sidecar_window_tokens, ModelState};
use qwen35::weights::Qwen35Weights;

use crate::tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le,
    f32_to_bf16_bytes as encode_bf16_le, f32_to_f32_bytes as encode_f32_le,
};
use kernel_ffi::prefill_ffi;

/// D2D copy that stays correctly sequenced with pending GPU work when an
/// open Metal batch is active. Without an open batch (or on HIP/CUDA), uses
/// the cheap host memcpy path; with a batch open on Metal, uses a Metal blit
/// encoded into the shared command buffer.
///
/// Use this in any prefill helper that may run inside `metal_v2_decode_step`'s
/// `MetalBatchGuard` scope. Outside that scope, behavior is unchanged.
fn copy_d2d_batched(
    ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
) -> Result<()> {
    if prefill_ffi::metal_batch_is_active() {
        prefill_ffi::metal_copy_d2d(src, dst, bytes)
            .map_err(|e| anyhow::anyhow!("metal blit copy: {e}"))
    } else {
        gpu_hal::copy_d2d(ordinal, dst, src, bytes).map_err(|e| anyhow::anyhow!("d2d copy: {e}"))
    }
}

fn copy_tap_rows_to_gpu_history(
    ordinal: usize,
    sink: &mut PrefillAppendGpuTapSink<'_>,
    tap_slot: usize,
    tap_count: usize,
    hidden: &GpuBuffer,
    chunk_len: usize,
    hidden_dim: usize,
) -> Result<()> {
    if chunk_len == 0 {
        return Ok(());
    }
    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
    let expected_row_bytes = tap_count * hidden_bytes;
    if sink.row_bytes != expected_row_bytes {
        return Err(anyhow::anyhow!(
            "GPU tap sink row_bytes {} != expected {} (tap_count={} hidden_dim={})",
            sink.row_bytes,
            expected_row_bytes,
            tap_count,
            hidden_dim
        ));
    }
    if tap_slot >= tap_count {
        return Err(anyhow::anyhow!(
            "GPU tap sink slot {tap_slot} out of range tap_count={tap_count}"
        ));
    }
    let first_dst = sink.start_row * sink.row_bytes + tap_slot * hidden_bytes;
    let last_dst = first_dst + (chunk_len - 1) * sink.row_bytes;
    if last_dst + hidden_bytes > sink.buffer.len_bytes() {
        return Err(anyhow::anyhow!(
            "GPU tap sink write exceeds buffer: offset {} + len {} > {}",
            last_dst,
            hidden_bytes,
            sink.buffer.len_bytes()
        ));
    }
    let needed_src = chunk_len * hidden_bytes;
    if needed_src > hidden.len_bytes() {
        return Err(anyhow::anyhow!(
            "GPU tap sink source hidden too small: need {} bytes, buffer has {}",
            needed_src,
            hidden.len_bytes()
        ));
    }
    for row in 0..chunk_len {
        let dst_off = first_dst + row * sink.row_bytes;
        let src_off = row * hidden_bytes;
        let dst = unsafe { (sink.buffer.as_mut_ptr() as *mut u8).add(dst_off) as *mut c_void };
        copy_d2d_batched(ordinal, dst, hidden.offset_ptr(src_off), hidden_bytes)
            .map_err(|e| anyhow::anyhow!("GPU tap sink copy row {row} slot {tap_slot}: {e}"))?;
    }
    Ok(())
}

fn dflash_bf16_rollback_trace_enabled() -> bool {
    gpu_hal::current_backend() == Backend::Hip
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_BF16_ROLLBACK_TRACE").is_none()
}

fn dflash_q8_rollback_trace_enabled() -> bool {
    gpu_hal::current_backend() == Backend::Hip
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_Q8_ROLLBACK_TRACE").is_none()
}

fn dflash_rollback_trace_dtype() -> ScalarType {
    if dflash_q8_rollback_trace_enabled() {
        ScalarType::U8
    } else if dflash_bf16_rollback_trace_enabled() {
        ScalarType::BF16
    } else {
        ScalarType::F32
    }
}

fn dflash_q8_trace_bytes(num_v_heads: usize, chunk_len: usize, khd: usize, vhd: usize) -> usize {
    let q8_block_bytes = 34usize;
    let q8_block = 32usize;
    num_v_heads * chunk_len * vhd * ((khd + q8_block - 1) / q8_block) * q8_block_bytes
}

fn encode_u32_le(values: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    out
}

fn detect_outlier_cols(lhs_bf16: &[f32], rows: usize, cols: usize, threshold: f32) -> Vec<usize> {
    let mut flags = vec![false; cols];
    for r in 0..rows {
        let row = &lhs_bf16[r * cols..(r + 1) * cols];
        for c in 0..cols {
            if f16::from_f32(row[c]).to_f32().abs() >= threshold {
                flags[c] = true;
            }
        }
    }
    flags
        .into_iter()
        .enumerate()
        .filter_map(|(idx, hit)| hit.then_some(idx))
        .collect()
}

fn host_bf16_addmm_inplace(
    base: &mut [f32],
    suba: &[f32],
    rows: usize,
    sub_cols: usize,
    subb_t: &[f32],
    out_dim: usize,
) {
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for kk in 0..sub_cols {
                acc += suba[r * sub_cols + kk] * subb_t[o * sub_cols + kk];
            }
            base[r * out_dim + o] = bf16::from_f32(base[r * out_dim + o] + acc).to_f32();
        }
    }
}

pub(crate) struct Int8MixedLhs {
    rows: usize,
    k: usize,
    lhs_host: Vec<f32>,
    outlier_cols: Vec<usize>,
    lhs_zeroed_gpu: Option<GpuBuffer>,
}

pub(crate) fn prepare_int8_mixed_lhs(
    ordinal: usize,
    batch: usize,
    m: usize,
    k: usize,
    lhs: &GpuBuffer,
    weights: &Qwen35Weights,
) -> Result<Option<Int8MixedLhs>> {
    if weights.int8_baked_store.is_none() || weights.int8_outlier_threshold <= 0.0 {
        return Ok(None);
    }

    let rows = batch * m;
    let lhs_host = decode_bf16_le(
        &lhs.to_host_bytes()
            .map_err(|e| anyhow::anyhow!("int8 mixed lhs D2H: {e}"))?,
    );
    let outlier_cols = detect_outlier_cols(&lhs_host, rows, k, weights.int8_outlier_threshold);
    if outlier_cols.is_empty() {
        return Ok(Some(Int8MixedLhs {
            rows,
            k,
            lhs_host,
            outlier_cols,
            lhs_zeroed_gpu: None,
        }));
    }

    let mut lhs_zeroed = lhs_host.clone();
    for r in 0..rows {
        for &col in &outlier_cols {
            lhs_zeroed[r * k + col] = 0.0;
        }
    }
    let lhs_zeroed_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        lhs.shape(),
        &encode_bf16_le(&lhs_zeroed),
    )
    .map_err(|e| anyhow::anyhow!("int8 mixed lhs_zeroed H2D: {e}"))?;

    Ok(Some(Int8MixedLhs {
        rows,
        k,
        lhs_host,
        outlier_cols,
        lhs_zeroed_gpu: Some(lhs_zeroed_gpu),
    }))
}

pub(crate) fn matmul_int8_mixed_host(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weights: &Qwen35Weights,
    weight_name: &str,
    weight: &GpuBuffer,
    int8_scale: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<()> {
    matmul_int8_mixed_prepared_host(
        ordinal,
        batch,
        m,
        n,
        k,
        lhs,
        weights,
        weight_name,
        weight,
        int8_scale,
        out,
        None,
    )
}

pub(crate) fn matmul_int8_mixed_prepared_host(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weights: &Qwen35Weights,
    weight_name: &str,
    weight: &GpuBuffer,
    int8_scale: &GpuBuffer,
    out: &mut GpuBuffer,
    prepared_lhs: Option<&Int8MixedLhs>,
) -> Result<()> {
    let Some(store) = weights.int8_baked_store.as_ref() else {
        return prefill_ffi::matmul_rhs_transposed_int8(
            ordinal, batch, m, n, k, lhs, weight, int8_scale, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
    };
    if weights.int8_outlier_threshold <= 0.0 {
        return prefill_ffi::matmul_rhs_transposed_int8(
            ordinal, batch, m, n, k, lhs, weight, int8_scale, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
    }

    let owned_prepared;
    let prepared_lhs = if let Some(prepared_lhs) = prepared_lhs {
        prepared_lhs
    } else {
        owned_prepared = prepare_int8_mixed_lhs(ordinal, batch, m, k, lhs, weights)?;
        let Some(prepared_lhs) = owned_prepared.as_ref() else {
            return prefill_ffi::matmul_rhs_transposed_int8(
                ordinal, batch, m, n, k, lhs, weight, int8_scale, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
        };
        prepared_lhs
    };

    let rows = batch * m;
    if prepared_lhs.rows != rows || prepared_lhs.k != k {
        return Err(anyhow::anyhow!(
            "int8 mixed prepared lhs shape mismatch: got rows={} k={}, want rows={} k={}",
            prepared_lhs.rows,
            prepared_lhs.k,
            rows,
            k
        ));
    }
    let Some(lhs_zeroed_gpu) = prepared_lhs.lhs_zeroed_gpu.as_ref() else {
        return prefill_ffi::matmul_rhs_transposed_int8(
            ordinal, batch, m, n, k, lhs, weight, int8_scale, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
    };
    let outlier_cols = &prepared_lhs.outlier_cols;
    prefill_ffi::matmul_rhs_transposed_int8(
        ordinal,
        batch,
        m,
        n,
        k,
        lhs_zeroed_gpu,
        weight,
        int8_scale,
        out,
    )
    .map_err(|e| anyhow::anyhow!("matmul_int8_zeroed: {e}"))?;

    if env::var_os("SUPERSONIC_LLAMA31_INT8_HOST_OUTLIER_CORRECTION").is_none() {
        let mut outlier_vals = Vec::with_capacity(rows * outlier_cols.len());
        for r in 0..rows {
            for &col in outlier_cols {
                outlier_vals.push(prepared_lhs.lhs_host[r * k + col]);
            }
        }
        let outlier_cols_gpu = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U32,
            &[outlier_cols.len()],
            &encode_u32_le(outlier_cols),
        )
        .map_err(|e| anyhow::anyhow!("int8 mixed outlier_cols H2D: {e}"))?;
        let outlier_vals_gpu = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[rows, outlier_cols.len()],
            &encode_f32_le(&outlier_vals),
        )
        .map_err(|e| anyhow::anyhow!("int8 mixed outlier_vals H2D: {e}"))?;
        return prefill_ffi::int8_outlier_add(
            ordinal,
            rows,
            n,
            k,
            outlier_cols.len(),
            weight,
            int8_scale,
            &outlier_cols_gpu,
            &outlier_vals_gpu,
            out,
        )
        .map_err(|e| anyhow::anyhow!("int8 mixed outlier add: {e}"));
    }

    let scb_name = weight_name.replace(".weight", ".SCB");
    let rhs_i8 = store
        .raw_bytes(weight_name)
        .ok_or_else(|| anyhow::anyhow!("missing baked raw bytes for {weight_name}"))?;
    let scb = decode_f32_le(
        store
            .raw_bytes(&scb_name)
            .ok_or_else(|| anyhow::anyhow!("missing baked raw bytes for {scb_name}"))?,
    );
    let mut base_host = decode_bf16_le(
        &out.to_host_bytes()
            .map_err(|e| anyhow::anyhow!("int8 mixed base D2H: {e}"))?,
    );

    let sub_cols = outlier_cols.len();
    let mut suba = vec![0.0f32; rows * sub_cols];
    for r in 0..rows {
        for (j, &col) in outlier_cols.iter().enumerate() {
            suba[r * sub_cols + j] = prepared_lhs.lhs_host[r * k + col];
        }
    }
    let mut subb_t = vec![0.0f32; n * sub_cols];
    let inv_127 = 1.0f32 / 127.0;
    for o in 0..n {
        let row_scale = scb[o];
        let row_base = o * k;
        for (j, &col) in outlier_cols.iter().enumerate() {
            let q = rhs_i8[row_base + col] as i8 as f32;
            subb_t[o * sub_cols + j] = bf16::from_f32(q * row_scale * inv_127).to_f32();
        }
    }
    host_bf16_addmm_inplace(&mut base_host, &suba, rows, sub_cols, &subb_t, n);
    let final_bytes = encode_bf16_le(&base_host);
    copy_h2d(
        ordinal,
        out.as_mut_ptr(),
        final_bytes.as_ptr() as *const std::ffi::c_void,
        final_bytes.len(),
    )
    .map_err(|e| anyhow::anyhow!("int8 mixed final H2D: {e}"))?;
    Ok(())
}

/// Dispatch matmul to either BF16 or FP8 dequant path.
/// When `scale` is Some, uses FP8 dequant matmul; otherwise standard BF16 matmul.
/// Projection matmul with INT4, FP8, or BF16 dispatch.
/// Priority: INT4 > FP8 > BF16.
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
    int4_awq_inv_scale: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<()> {
    let qtype = qwen35::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    if qwen35::weights::is_gqh_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("GQH matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen35::weights::matmul_gqh(ordinal, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_gqh: {e}"));
    }
    if qtype != 0 {
        let sc = int4_scale.unwrap_or(weight);
        let zr = int4_zero.unwrap_or(weight);
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            batch,
            m,
            n,
            k,
            lhs,
            weight,
            sc,
            zr,
            int4_awq_inv_scale,
            int4_group_size,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
    } else if let Some(sc) = int8_scale {
        prefill_ffi::matmul_rhs_transposed_int8(ordinal, batch, m, n, k, lhs, weight, sc, out)
            .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"))
    } else {
        match scale {
            Some(s) => prefill_ffi::matmul_rhs_transposed_fp8(
                ordinal, batch, m, n, k, lhs, weight, s, block_size, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_fp8: {e}")),
            None => prefill_ffi::matmul_rhs_transposed(
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

fn prefill_lm_head_lowbit(
    ordinal: usize,
    count: usize,
    vocab_size: usize,
    hidden_dim: usize,
    lhs: &GpuBuffer,
    weights: &Qwen35Weights,
    out: &mut GpuBuffer,
    label: &str,
) -> Result<bool> {
    let Some((qtype, scale, zero)) = weights.lm_head_lowbit_params(hidden_dim) else {
        return Ok(false);
    };
    if qwen35::weights::is_gqh_qtype(qtype) {
        qwen35::weights::matmul_gqh(
            ordinal,
            vocab_size,
            hidden_dim,
            lhs,
            &*weights.lm_head,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} gqh: {e}"))?;
        return Ok(true);
    }
    if !maybe_matmul_q6_k_mmq_lm_head(
        ordinal,
        1,
        count,
        vocab_size,
        hidden_dim,
        qtype,
        weights.lm_head_awq_inv_scale.as_ref(),
        lhs,
        &*weights.lm_head,
        out,
    )? {
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            count,
            vocab_size,
            hidden_dim,
            lhs,
            &*weights.lm_head,
            scale,
            zero,
            weights.lm_head_awq_inv_scale.as_ref(),
            weights.int4_group_size,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} int4: {e}"))?;
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn matmul_proj_residual_add_inplace(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    scale: Option<&GpuBuffer>,
    int8_scale: Option<&GpuBuffer>,
    out_residual: &mut GpuBuffer,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    int4_awq_inv_scale: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<bool> {
    if env::var_os("SUPERSONIC_DFLASH_DISABLE_TREE_RESIDUAL_FUSED_MATMUL").is_some()
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m != 16
        || scale.is_some()
        || int8_scale.is_some()
        || int4_awq_inv_scale.is_some()
    {
        return Ok(false);
    }

    let qtype = qwen35::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    let raw_ggml = matches!(
        qtype,
        qwen35::weights::LOWBIT_GGML_Q8_0
            | qwen35::weights::LOWBIT_GGML_Q4_K
            | qwen35::weights::LOWBIT_GGML_Q5_K
            | qwen35::weights::LOWBIT_GGML_Q6_K
    );
    if !raw_ggml {
        return Ok(false);
    }

    let sc = int4_scale.unwrap_or(weight);
    let zr = int4_zero.unwrap_or(weight);
    let residual: &GpuBuffer = unsafe { &*(out_residual as *const GpuBuffer) };
    prefill_ffi::matmul_rhs_transposed_int4_residual_add(
        ordinal,
        batch,
        m,
        n,
        k,
        lhs,
        weight,
        sc,
        zr,
        int4_awq_inv_scale,
        int4_group_size,
        qtype,
        residual,
        out_residual,
    )
    .map_err(|e| anyhow::anyhow!("matmul_int4 residual add: {e}"))
}

fn mmq_q8_1_workspace_bytes(batch: usize, m: usize, k: usize) -> usize {
    const Q8_BLOCK: usize = 128;
    const Q8_BLOCK_BYTES: usize = 144;
    batch * ((k + Q8_BLOCK - 1) / Q8_BLOCK) * m * Q8_BLOCK_BYTES
}

fn ensure_mmq_q8_1_workspace<'a>(
    workspace: &'a mut Option<GpuBuffer>,
    ordinal: usize,
    batch: usize,
    m: usize,
    k: usize,
    label: &str,
) -> Result<&'a mut GpuBuffer> {
    let q8_bytes = mmq_q8_1_workspace_bytes(batch, m, k).max(1);
    let needs_alloc = workspace
        .as_ref()
        .map(|buf| buf.len_bytes() < q8_bytes)
        .unwrap_or(true);
    if needs_alloc {
        *workspace = Some(
            GpuBuffer::alloc(ordinal, ScalarType::U8, &[q8_bytes])
                .map_err(|e| anyhow::anyhow!("{label}: {e}"))?,
        );
    }
    workspace
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("{label}: missing workspace after allocation"))
}

pub(crate) fn q6_k_mmq_lm_head_enabled(m: usize) -> bool {
    if env::var_os("SUPERSONIC_DISABLE_Q6_K_MMQ_LM_HEAD").is_some() {
        return false;
    }
    env::var_os("SUPERSONIC_ENABLE_Q6_K_MMQ_LM_HEAD").is_some() || m == 8
}

fn q6_k_lm_head_argmax_fused_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_Q6_K_LM_HEAD_ARGMAX_FUSED").is_none()
}

fn q6_k_mmq_mlp_down_enabled() -> bool {
    env::var_os("SUPERSONIC_DISABLE_Q6_K_MMQ_MLP_DOWN").is_none()
}

fn q6_k_mmq_mlp_down_residual_fused_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_Q6_K_MMQ_MLP_DOWN_RESIDUAL_FUSED").is_none()
}

fn ggml_mlp_gate_up_pair_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_GGML_MLP_GATE_UP_PAIR").is_none()
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_GGML_PAIR_M16_QTYPE").is_none()
}

fn ggml_mlp_gate_up_swiglu_fused_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_GGML_MLP_GATE_UP_SWIGLU_FUSED").is_none()
}

fn raw_ggml_qtype(qtype: i32) -> bool {
    matches!(
        qtype,
        qwen35::weights::LOWBIT_GGML_Q8_0
            | qwen35::weights::LOWBIT_GGML_Q4_K
            | qwen35::weights::LOWBIT_GGML_Q5_K
            | qwen35::weights::LOWBIT_GGML_Q6_K
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_matmul_q6_k_mmq_lm_head(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    qtype: i32,
    awq_inv_scale: Option<&GpuBuffer>,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<bool> {
    if !q6_k_mmq_lm_head_enabled(m)
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || qtype != qwen35::weights::LOWBIT_GGML_Q6_K
        || awq_inv_scale.is_some()
        || k % 256 != 0
    {
        return Ok(false);
    }

    if !prefill_ffi::device_supports_wmma_i8(ordinal)
        .map_err(|e| anyhow::anyhow!("q6_k_mmq lm_head arch probe: {e}"))?
    {
        return Ok(false);
    }

    let q8_bytes = mmq_q8_1_workspace_bytes(batch, m, k);
    let mut q8_workspace = GpuBuffer::alloc(ordinal, ScalarType::U8, &[q8_bytes.max(1)])
        .map_err(|e| anyhow::anyhow!("q6_k_mmq lm_head q8 workspace: {e}"))?;
    prefill_ffi::quantize_mmq_q8_1(
        ordinal,
        batch,
        m,
        k,
        lhs,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        &mut q8_workspace,
    )
    .map_err(|e| anyhow::anyhow!("q6_k_mmq lm_head quantize q8_1: {e}"))?;
    prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, batch, m, n, k, &q8_workspace, weight, out)
        .map_err(|e| anyhow::anyhow!("q6_k_mmq lm_head matmul: {e}"))?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn maybe_matmul_q6_k_lm_head_argmax(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    qtype: i32,
    awq_inv_scale: Option<&GpuBuffer>,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    block_best_vals: &mut GpuBuffer,
    block_best_indices: &mut GpuBuffer,
    out_indices: &mut GpuBuffer,
) -> Result<bool> {
    if !q6_k_lm_head_argmax_fused_enabled()
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m != 16
        || n == 0
        || k == 0
        || n % 16 != 0
        || k % 256 != 0
        || qtype != qwen35::weights::LOWBIT_GGML_Q6_K
        || awq_inv_scale.is_some()
    {
        return Ok(false);
    }

    prefill_ffi::matmul_q6_k_m16_argmax(
        ordinal,
        batch,
        m,
        n,
        k,
        lhs,
        weight,
        block_best_vals,
        block_best_indices,
        out_indices,
    )
    .map_err(|e| anyhow::anyhow!("q6_k lm_head fused argmax: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn maybe_matmul_ggml_mlp_gate_up_pair(
    ordinal: usize,
    batch: usize,
    m: usize,
    n_each: usize,
    k: usize,
    lhs: &GpuBuffer,
    gate_weight: &GpuBuffer,
    gate_scale: Option<&GpuBuffer>,
    gate_int8_scale: Option<&GpuBuffer>,
    gate_int4_scale: Option<&GpuBuffer>,
    gate_int4_zero: Option<&GpuBuffer>,
    gate_awq_inv_scale: Option<&GpuBuffer>,
    up_weight: &GpuBuffer,
    up_scale: Option<&GpuBuffer>,
    up_int8_scale: Option<&GpuBuffer>,
    up_int4_scale: Option<&GpuBuffer>,
    up_int4_zero: Option<&GpuBuffer>,
    up_awq_inv_scale: Option<&GpuBuffer>,
    packed_gate_up: &mut GpuBuffer,
    swiglu_out: &mut GpuBuffer,
) -> Result<bool> {
    if !ggml_mlp_gate_up_pair_enabled()
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m != 16
        || n_each == 0
        || k == 0
        || n_each % 16 != 0
        || k % 256 != 0
        || gate_scale.is_some()
        || gate_int8_scale.is_some()
        || gate_int4_scale.is_some()
        || gate_int4_zero.is_some()
        || gate_awq_inv_scale.is_some()
        || up_scale.is_some()
        || up_int8_scale.is_some()
        || up_int4_scale.is_some()
        || up_int4_zero.is_some()
        || up_awq_inv_scale.is_some()
    {
        return Ok(false);
    }

    let gate_qtype = qwen35::weights::infer_lowbit_type(gate_weight, k, false);
    let up_qtype = qwen35::weights::infer_lowbit_type(up_weight, k, false);
    if gate_qtype != up_qtype || !raw_ggml_qtype(gate_qtype) {
        return Ok(false);
    }

    if ggml_mlp_gate_up_swiglu_fused_enabled()
        && prefill_ffi::matmul_rhs_transposed_ggml_pair_swiglu(
            ordinal,
            batch,
            m,
            n_each,
            k,
            lhs,
            gate_weight,
            up_weight,
            gate_qtype,
            swiglu_out,
        )
        .map_err(|e| anyhow::anyhow!("ggml MLP gate/up fused SwiGLU: {e}"))?
    {
        return Ok(true);
    }

    prefill_ffi::matmul_rhs_transposed_ggml_pair(
        ordinal,
        batch,
        m,
        n_each,
        k,
        lhs,
        gate_weight,
        up_weight,
        gate_qtype,
        packed_gate_up,
    )
    .map_err(|e| anyhow::anyhow!("ggml MLP gate/up pair matmul: {e}"))?;
    prefill_ffi::swiglu_mul_split(
        ordinal,
        ScalarType::BF16,
        m,
        n_each,
        packed_gate_up,
        swiglu_out,
    )
    .map_err(|e| anyhow::anyhow!("ggml MLP gate/up split SwiGLU: {e}"))?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn maybe_matmul_q6_k_mmq_mlp_down(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    qtype: i32,
    scale: Option<&GpuBuffer>,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    awq_inv_scale: Option<&GpuBuffer>,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
    q8_workspace: &mut Option<GpuBuffer>,
) -> Result<bool> {
    if !q6_k_mmq_mlp_down_enabled()
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m == 0
        || n == 0
        || k == 0
        || qtype != qwen35::weights::LOWBIT_GGML_Q6_K
        || scale.is_some()
        || int4_scale.is_some()
        || int4_zero.is_some()
        || awq_inv_scale.is_some()
        || k % 256 != 0
    {
        return Ok(false);
    }

    if !prefill_ffi::device_supports_wmma_i8(ordinal)
        .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down arch probe: {e}"))?
    {
        return Ok(false);
    }

    let q8_workspace = ensure_mmq_q8_1_workspace(
        q8_workspace,
        ordinal,
        batch,
        m,
        k,
        "q6_k_mmq MLP down q8 workspace",
    )?;
    prefill_ffi::quantize_mmq_q8_1(
        ordinal,
        batch,
        m,
        k,
        lhs,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        q8_workspace,
    )
    .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down quantize q8_1: {e}"))?;
    prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, batch, m, n, k, q8_workspace, weight, out)
        .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down matmul: {e}"))?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn maybe_matmul_q6_k_mmq_mlp_down_residual_add(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    qtype: i32,
    scale: Option<&GpuBuffer>,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    awq_inv_scale: Option<&GpuBuffer>,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    out_residual: &mut GpuBuffer,
    q8_workspace: &mut Option<GpuBuffer>,
) -> Result<bool> {
    if !q6_k_mmq_mlp_down_enabled()
        || !q6_k_mmq_mlp_down_residual_fused_enabled()
        || gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m == 0
        || n == 0
        || k == 0
        || qtype != qwen35::weights::LOWBIT_GGML_Q6_K
        || scale.is_some()
        || int4_scale.is_some()
        || int4_zero.is_some()
        || awq_inv_scale.is_some()
        || k % 256 != 0
    {
        return Ok(false);
    }

    if !prefill_ffi::device_supports_wmma_i8(ordinal)
        .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down residual arch probe: {e}"))?
    {
        return Ok(false);
    }

    let q8_workspace = ensure_mmq_q8_1_workspace(
        q8_workspace,
        ordinal,
        batch,
        m,
        k,
        "q6_k_mmq MLP down residual q8 workspace",
    )?;
    prefill_ffi::quantize_mmq_q8_1(
        ordinal,
        batch,
        m,
        k,
        lhs,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        q8_workspace,
    )
    .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down residual quantize q8_1: {e}"))?;
    let residual: &GpuBuffer = unsafe { &*(out_residual as *const GpuBuffer) };
    prefill_ffi::matmul_mmq_q8_1_q6_k_residual_add(
        ordinal,
        batch,
        m,
        n,
        k,
        q8_workspace,
        weight,
        residual,
        out_residual,
    )
    .map_err(|e| anyhow::anyhow!("q6_k_mmq MLP down residual matmul: {e}"))?;
    Ok(true)
}

/// In-place residual add: dst += src.
/// Uses unsafe to work around the borrow checker since the GPU kernel
/// reads src[i] and writes dst[i] independently per element.
fn residual_add(
    ordinal: usize,
    total_elems: usize,
    dst: &mut GpuBuffer,
    src: &GpuBuffer,
) -> Result<()> {
    let lhs: &GpuBuffer = unsafe { &*(dst as *const GpuBuffer) };
    prefill_ffi::element_add(ordinal, ScalarType::BF16, total_elems, lhs, src, dst)
        .map_err(|e| anyhow::anyhow!("residual_add failed: {e}"))?;
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
        prefill_ffi::rms_norm_rows
    } else {
        prefill_ffi::rms_norm_rows_plain
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
            prefill_ffi::rms_norm_rows
        } else {
            prefill_ffi::rms_norm_rows_plain
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
        copy_d2d_batched(
            ordinal,
            output.as_mut_ptr(),
            input.as_ptr(),
            rows * cols * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("{label} copy-through: {e}"))?;
    }
    Ok(())
}

fn maybe_attn_rms_norm_rows_inplace(
    config: &TextConfig,
    ordinal: usize,
    rows: usize,
    cols: usize,
    data: &mut GpuBuffer,
    weight: Option<&GpuBuffer>,
    label: &str,
) -> Result<bool> {
    if gpu_hal::current_backend() != Backend::Hip
        || config.rms_norm_add_unit_offset
        || env::var_os("SUPERSONIC_DFLASH_DISABLE_INPLACE_ATTN_QK_NORM").is_some()
    {
        return Ok(false);
    }
    let Some(weight) = weight else {
        return Ok(true);
    };
    prefill_ffi::rms_norm_rows_plain_inplace(
        ordinal,
        ScalarType::BF16,
        rows,
        cols,
        1e-6,
        data,
        weight,
    )
    .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn maybe_split_qgate_norm_bf16(
    config: &TextConfig,
    ordinal: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    q_norm_w: Option<&GpuBuffer>,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
    label: &str,
) -> Result<bool> {
    let Some(q_norm_w) = q_norm_w else {
        return Ok(false);
    };
    if gpu_hal::current_backend() != Backend::Hip
        || config.rms_norm_add_unit_offset
        || src.dtype() != ScalarType::BF16
        || q_norm_w.dtype() != ScalarType::BF16
        || env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_QGATE_QNORM").is_some()
    {
        return Ok(false);
    }
    prefill_ffi::split_qgate_norm_bf16(
        ordinal, seq_len, num_heads, head_dim, 1e-6, src, q_norm_w, query_out, gate_out,
    )
    .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    Ok(true)
}

/// Compute per-position logits for a contiguous range of the hidden-state
/// buffer.
///
/// * `hidden`: `[seq_len, hidden_dim]` BF16 (typically `scratch.hidden` after
///   a prefill pass).
/// * `start`, `count`: logical range `[start..start+count]`.
///
/// Returns `(logits_per_pos, normed)` where `logits_per_pos.len() == count`
/// and each inner vec has `vocab_size` F32 entries. `normed` is the BF16
/// `[count, hidden_dim]` buffer produced by the final RMSNorm before
/// `lm_head` — kept available so the caller can emit a final-norm trace
/// without re-running the norm.
///
/// Allocates scratch buffers locally; the verify path is called at most
/// once per speculative round and once per prefill, so the cost is
/// amortized. Hot-path prefill also goes through here with `count=1`.
pub fn compute_logits_for_range(
    hidden: &GpuBuffer,
    weights: &Qwen35Weights,
    config: &TextConfig,
    start: usize,
    count: usize,
    _use_4b_kernel: bool,
    ordinal: usize,
) -> Result<(Vec<Vec<f32>>, GpuBuffer)> {
    if count == 0 {
        return Err(anyhow::anyhow!(
            "compute_logits_for_range: count must be > 0"
        ));
    }
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    // D2D slice [start..start+count] of the hidden-state buffer. The append
    // verifier asks for the leading range, so borrow it directly in that hot
    // path and avoid an allocation/copy.
    let slice_storage;
    let slice = if start == 0 {
        hidden
    } else {
        let mut buf = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[count, hidden_dim])
            .map_err(|e| anyhow::anyhow!("range slice alloc: {e}"))?;
        let src_offset = start * hidden_dim * elem_bytes;
        copy_d2d_batched(
            ordinal,
            buf.as_mut_ptr(),
            hidden.offset_ptr(src_offset),
            count * hidden_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("range slice copy: {e}"))?;
        slice_storage = buf;
        &slice_storage
    };

    // Final RMSNorm → BF16 [count, hidden_dim]. Qwen3.5 uses add_unit_offset=1.
    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range normed alloc: {e}"))?;
    rms_norm_rows_model(
        config,
        ordinal,
        count,
        hidden_dim,
        slice,
        &weights.norm_weight,
        &mut normed,
        "range final norm",
    )?;

    // lm_head projection. For count=1, prefer the standalone matvec even on
    // 4B-capable models: it avoids the tiled matmul path's extra packing and
    // keeps the single-row score path numerically aligned with decode.
    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
        .map_err(|e| anyhow::anyhow!("range logits alloc: {e}"))?;
    // INT4 lm_head: when the baked package quantized lm_head weights to GPTQ
    // INT4, dispatch through the INT4 dequant matmul. Saves ~4x device memory
    // bandwidth on what is the dominant matmul on small models.
    if prefill_lm_head_lowbit(
        ordinal,
        count,
        vocab_size,
        hidden_dim,
        &normed,
        weights,
        &mut logits_buf,
        "range lm_head",
    )? {
    } else if count > 1 {
        kernel_ffi::matmul_rhs_transposed_4b(
            ordinal,
            ScalarType::BF16,
            1,          // batch
            count,      // m
            vocab_size, // n
            hidden_dim, // k
            &normed,
            &*weights.lm_head,
            &mut logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head tiled: {e}"))?;
    } else {
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("range matvec counter: {e}"))?;
        kernel_ffi::standalone_matvec(
            ordinal,
            ScalarType::BF16,
            &mut logits_buf,
            &normed,
            &*weights.lm_head,
            hidden_dim,
            vocab_size,
            &mut counter,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head matvec: {e}"))?;
    }

    // If a Metal batch is open (Metal v2 incremental decode wraps the entire
    // step in one), commit + wait so the lm_head GPU work is visible to the
    // host memcpy below. No-op when no batch is active or on non-Metal builds.
    if prefill_ffi::metal_batch_is_active() {
        prefill_ffi::flush_metal_batch()
            .map_err(|e| anyhow::anyhow!("range logits batch flush: {e}"))?;
    }

    // D2H + split into one Vec<f32> per position.
    let host_bytes = logits_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("range logits D2H: {e}"))?;
    let row_elems = vocab_size;
    let mut logits_per_pos: Vec<Vec<f32>> = Vec::with_capacity(count);
    for row in 0..count {
        let start_byte = row * row_elems * elem_bytes;
        let end_byte = start_byte + row_elems * elem_bytes;
        let row_vec: Vec<f32> = host_bytes[start_byte..end_byte]
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        logits_per_pos.push(row_vec);
    }

    Ok((logits_per_pos, normed))
}

fn compute_logits_for_range_f32_hidden(
    hidden_f32: &GpuBuffer,
    weights: &Qwen35Weights,
    config: &TextConfig,
    start: usize,
    count: usize,
    _use_4b_kernel: bool,
    ordinal: usize,
) -> Result<(Vec<Vec<f32>>, GpuBuffer)> {
    if count == 0 {
        return Err(anyhow::anyhow!(
            "compute_logits_for_range_f32_hidden: count must be > 0"
        ));
    }
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let elem_bytes_f32 = ScalarType::F32.size_in_bytes();
    let elem_bytes_bf16 = ScalarType::BF16.size_in_bytes();

    let slice_storage;
    let slice = if start == 0 {
        hidden_f32
    } else {
        let mut buf = GpuBuffer::alloc(ordinal, ScalarType::F32, &[count, hidden_dim])
            .map_err(|e| anyhow::anyhow!("range F32 slice alloc: {e}"))?;
        let src_offset = start * hidden_dim * elem_bytes_f32;
        copy_d2d_batched(
            ordinal,
            buf.as_mut_ptr(),
            hidden_f32.offset_ptr(src_offset),
            count * hidden_dim * elem_bytes_f32,
        )
        .map_err(|e| anyhow::anyhow!("range F32 slice copy: {e}"))?;
        slice_storage = buf;
        &slice_storage
    };

    let mut norm_weight_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[hidden_dim])
        .map_err(|e| anyhow::anyhow!("range final norm weight F32 alloc: {e}"))?;
    prefill_ffi::cast(
        ordinal,
        weights.norm_weight.dtype(),
        ScalarType::F32,
        hidden_dim,
        &weights.norm_weight,
        &mut norm_weight_f32,
    )
    .map_err(|e| anyhow::anyhow!("range final norm weight F32 cast: {e}"))?;

    let mut normed_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range final normed F32 alloc: {e}"))?;
    let op = if config.rms_norm_add_unit_offset {
        prefill_ffi::rms_norm_rows
    } else {
        prefill_ffi::rms_norm_rows_plain
    };
    op(
        ordinal,
        ScalarType::F32,
        count,
        hidden_dim,
        config.rms_norm_eps as f32,
        slice,
        &norm_weight_f32,
        &mut normed_f32,
    )
    .map_err(|e| anyhow::anyhow!("range final norm F32: {e}"))?;

    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range final normed BF16 alloc: {e}"))?;
    prefill_ffi::cast(
        ordinal,
        ScalarType::F32,
        ScalarType::BF16,
        count * hidden_dim,
        &normed_f32,
        &mut normed,
    )
    .map_err(|e| anyhow::anyhow!("range final norm F32->BF16 cast: {e}"))?;

    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
        .map_err(|e| anyhow::anyhow!("range logits alloc: {e}"))?;
    if prefill_lm_head_lowbit(
        ordinal,
        count,
        vocab_size,
        hidden_dim,
        &normed,
        weights,
        &mut logits_buf,
        "range lm_head",
    )? {
    } else if count > 1 {
        kernel_ffi::matmul_rhs_transposed_4b(
            ordinal,
            ScalarType::BF16,
            1,
            count,
            vocab_size,
            hidden_dim,
            &normed,
            &*weights.lm_head,
            &mut logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head tiled: {e}"))?;
    } else {
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("range matvec counter: {e}"))?;
        kernel_ffi::standalone_matvec(
            ordinal,
            ScalarType::BF16,
            &mut logits_buf,
            &normed,
            &*weights.lm_head,
            hidden_dim,
            vocab_size,
            &mut counter,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head matvec: {e}"))?;
    }

    if prefill_ffi::metal_batch_is_active() {
        prefill_ffi::flush_metal_batch()
            .map_err(|e| anyhow::anyhow!("range logits batch flush: {e}"))?;
    }

    let host_bytes = logits_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("range logits D2H: {e}"))?;
    let row_elems = vocab_size;
    let mut logits_per_pos: Vec<Vec<f32>> = Vec::with_capacity(count);
    for row in 0..count {
        let start_byte = row * row_elems * elem_bytes_bf16;
        let end_byte = start_byte + row_elems * elem_bytes_bf16;
        let row_vec: Vec<f32> = host_bytes[start_byte..end_byte]
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        logits_per_pos.push(row_vec);
    }

    Ok((logits_per_pos, normed))
}

pub fn compute_greedy_for_range(
    hidden: &GpuBuffer,
    weights: &Qwen35Weights,
    config: &TextConfig,
    start: usize,
    count: usize,
    use_4b_kernel: bool,
    ordinal: usize,
) -> Result<(Vec<u32>, GpuBuffer)> {
    if gpu_hal::current_backend() != Backend::Hip {
        let (logits, normed) = compute_logits_for_range(
            hidden,
            weights,
            config,
            start,
            count,
            use_4b_kernel,
            ordinal,
        )?;
        let ids = logits
            .iter()
            .map(|row| {
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &val) in row.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_idx = idx as u32;
                    }
                }
                best_idx
            })
            .collect();
        return Ok((ids, normed));
    }

    if count == 0 {
        return Err(anyhow::anyhow!(
            "compute_greedy_for_range: count must be > 0"
        ));
    }
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    let slice_storage;
    let slice = if start == 0 {
        hidden
    } else {
        let mut buf = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[count, hidden_dim])
            .map_err(|e| anyhow::anyhow!("range greedy slice alloc: {e}"))?;
        let src_offset = start * hidden_dim * elem_bytes;
        copy_d2d_batched(
            ordinal,
            buf.as_mut_ptr(),
            hidden.offset_ptr(src_offset),
            count * hidden_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("range greedy slice copy: {e}"))?;
        slice_storage = buf;
        &slice_storage
    };

    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range greedy normed alloc: {e}"))?;
    rms_norm_rows_model(
        config,
        ordinal,
        count,
        hidden_dim,
        slice,
        &weights.norm_weight,
        &mut normed,
        "range greedy final norm",
    )?;

    let mut out_index = GpuBuffer::zeros(ordinal, ScalarType::U32, &[count])
        .map_err(|e| anyhow::anyhow!("range greedy argmax alloc: {e}"))?;
    let mut fused_argmax = false;
    if let Some((qtype, _, _)) = weights.lm_head_lowbit_params(hidden_dim) {
        if count == 16 {
            let lm_head_tiles = (vocab_size + 15) / 16;
            let mut block_best_vals =
                GpuBuffer::alloc(ordinal, ScalarType::F32, &[count, lm_head_tiles])
                    .map_err(|e| anyhow::anyhow!("range greedy block-best vals alloc: {e}"))?;
            let mut block_best_indices =
                GpuBuffer::alloc(ordinal, ScalarType::U32, &[count, lm_head_tiles])
                    .map_err(|e| anyhow::anyhow!("range greedy block-best indices alloc: {e}"))?;
            fused_argmax = maybe_matmul_q6_k_lm_head_argmax(
                ordinal,
                1,
                count,
                vocab_size,
                hidden_dim,
                qtype,
                weights.lm_head_awq_inv_scale.as_ref(),
                &normed,
                &*weights.lm_head,
                &mut block_best_vals,
                &mut block_best_indices,
                &mut out_index,
            )?;
        }
        if !fused_argmax {
            let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
                .map_err(|e| anyhow::anyhow!("range greedy logits alloc: {e}"))?;
            if !prefill_lm_head_lowbit(
                ordinal,
                count,
                vocab_size,
                hidden_dim,
                &normed,
                weights,
                &mut logits_buf,
                "range greedy lm_head",
            )? {
                unreachable!("lowbit lm_head params were Some");
            }
            prefill_ffi::argmax_bf16_rows(ordinal, count, vocab_size, &logits_buf, &mut out_index)
                .map_err(|e| anyhow::anyhow!("range greedy argmax: {e}"))?;
        }
    } else {
        let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
            .map_err(|e| anyhow::anyhow!("range greedy logits alloc: {e}"))?;
        if count > 1 {
            kernel_ffi::matmul_rhs_transposed_4b(
                ordinal,
                ScalarType::BF16,
                1,
                count,
                vocab_size,
                hidden_dim,
                &normed,
                &*weights.lm_head,
                &mut logits_buf,
            )
            .map_err(|e| anyhow::anyhow!("range greedy lm_head tiled: {e}"))?;
        } else {
            let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
                .map_err(|e| anyhow::anyhow!("range greedy matvec counter: {e}"))?;
            kernel_ffi::standalone_matvec(
                ordinal,
                ScalarType::BF16,
                &mut logits_buf,
                &normed,
                &*weights.lm_head,
                hidden_dim,
                vocab_size,
                &mut counter,
            )
            .map_err(|e| anyhow::anyhow!("range greedy lm_head matvec: {e}"))?;
        }
        prefill_ffi::argmax_bf16_rows(ordinal, count, vocab_size, &logits_buf, &mut out_index)
            .map_err(|e| anyhow::anyhow!("range greedy argmax: {e}"))?;
    }

    let ids_bytes = out_index
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("range greedy ids D2H: {e}"))?;
    let ids = ids_bytes
        .chunks_exact(4)
        .take(count)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((ids, normed))
}

fn compute_greedy_for_acceptance(
    hidden: &GpuBuffer,
    weights: &Qwen35Weights,
    config: &TextConfig,
    count: usize,
    compare_tokens: &[u32],
    use_4b_kernel: bool,
    ordinal: usize,
) -> Result<Vec<u32>> {
    if compare_tokens.len() < count {
        return Err(anyhow::anyhow!(
            "greedy acceptance compare tokens {} shorter than count {count}",
            compare_tokens.len()
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let scan_chunk = std::env::var("SUPERSONIC_DFLASH_GREEDY_SCAN_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16);
    let mut ids = Vec::with_capacity(count);

    while ids.len() < count {
        let start = ids.len();
        let step = scan_chunk.min(count - start);
        let (mut next, _normed) =
            compute_greedy_for_range(hidden, weights, config, start, step, use_4b_kernel, ordinal)?;
        ids.append(&mut next);

        let mut accept_n = 1usize;
        while accept_n < count && accept_n <= ids.len() {
            if ids[accept_n - 1] == compare_tokens[accept_n] {
                accept_n += 1;
            } else {
                return Ok(ids);
            }
        }
    }

    Ok(ids)
}

fn compute_target_nll_for_range(
    hidden: &GpuBuffer,
    weights: &Qwen35Weights,
    config: &TextConfig,
    start: usize,
    targets: &[u32],
    ordinal: usize,
) -> Result<Vec<f32>> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let count = targets.len();
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    for &target in targets {
        if target as usize >= vocab_size {
            return Err(anyhow::anyhow!(
                "target token {target} outside vocab size {vocab_size}"
            ));
        }
    }

    let slice = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("target NLL slice alloc: {e}"))?;
    let src_offset = start * hidden_dim * elem_bytes;
    gpu_hal::copy_d2d(
        ordinal,
        slice.as_ptr() as *mut c_void,
        hidden.offset_ptr(src_offset),
        count * hidden_dim * elem_bytes,
    )
    .map_err(|e| anyhow::anyhow!("target NLL slice copy: {e}"))?;

    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("target NLL normed alloc: {e}"))?;
    rms_norm_rows_model(
        config,
        ordinal,
        count,
        hidden_dim,
        &slice,
        &weights.norm_weight,
        &mut normed,
        "target NLL final norm",
    )?;

    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
        .map_err(|e| anyhow::anyhow!("target NLL logits alloc: {e}"))?;
    kernel_ffi::matmul_rhs_transposed_4b(
        ordinal,
        ScalarType::BF16,
        1,
        count,
        vocab_size,
        hidden_dim,
        &normed,
        &*weights.lm_head,
        &mut logits_buf,
    )
    .map_err(|e| anyhow::anyhow!("target NLL lm_head tiled: {e}"))?;

    let mut target_bytes = Vec::with_capacity(count * 4);
    for &target in targets {
        target_bytes.extend_from_slice(&target.to_le_bytes());
    }
    let targets_gpu = GpuBuffer::zeros(ordinal, ScalarType::U32, &[count])
        .map_err(|e| anyhow::anyhow!("target NLL target alloc: {e}"))?;
    copy_h2d(
        ordinal,
        targets_gpu.as_ptr() as *mut c_void,
        target_bytes.as_ptr() as *const c_void,
        target_bytes.len(),
    )
    .map_err(|e| anyhow::anyhow!("target NLL target H2D: {e}"))?;

    let mut nll_gpu = GpuBuffer::zeros(ordinal, ScalarType::F32, &[count])
        .map_err(|e| anyhow::anyhow!("target NLL output alloc: {e}"))?;
    kernel_ffi::cuda_target_nll_bf16(
        ordinal,
        &logits_buf,
        &targets_gpu,
        &mut nll_gpu,
        count,
        vocab_size,
    )
    .map_err(|e| anyhow::anyhow!("target NLL kernel: {e}"))?;

    let nll_bytes = nll_gpu
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("target NLL D2H: {e}"))?;
    Ok(nll_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[derive(Debug, Clone)]
pub struct PrefillTargetNll {
    pub total_nll: f64,
    pub scored_tokens: usize,
}

/// Result of a prefill pass.
pub struct PrefillResult {
    /// Logits for the last token position [vocab_size] as F32 on CPU.
    pub logits: Vec<f32>,
    /// Optional BF16 last-token dump after final RMSNorm and before lm_head.
    pub final_norm_trace: Option<Vec<u8>>,
    /// Optional BF16 last-token hidden dump after token-mixer residual for each layer.
    pub layer_attn_trace: Option<Vec<Vec<u8>>>,
    /// Optional BF16 last-token dump after post-attention RMSNorm for each layer.
    pub layer_post_attn_norm_trace: Option<Vec<Vec<u8>>>,
    /// Optional BF16 last-token dump after SwiGLU and before MLP down-proj.
    pub layer_mlp_swiglu_trace: Option<Vec<Vec<u8>>>,
    /// Optional BF16 last-token dump after MLP down-proj and before residual add.
    pub layer_mlp_out_trace: Option<Vec<Vec<u8>>>,
    /// Optional BF16 last-token hidden dump after each decoder layer.
    pub layer_hidden_trace: Option<Vec<Vec<u8>>>,
    /// DFlash hidden-state taps. When `tap_layers` is supplied to `prefill`, this
    /// vector is 1:1 with `tap_layers`: each entry is a BF16-encoded `[hidden_dim]`
    /// blob holding the post-MLP residual hidden state of the LAST token of the
    /// final chunk for that layer. Always None when `tap_layers` was None.
    pub tap_hiddens: Option<Vec<Vec<u8>>>,
    /// DFlash hidden-state tap history for every prefill token. When present,
    /// this is 1:1 with `tap_layers`; each entry is BF16 `[seq_len, hidden_dim]`
    /// in token order for one tap layer.
    pub tap_hiddens_all: Option<Vec<Vec<u8>>>,
    /// Optional last-token debug trace for one selected linear-attention layer.
    pub linear_debug_trace: Option<LinearLayerDebugTrace>,
    /// Optional GPU-computed next-token NLL summary for a requested hidden range.
    pub target_nll: Option<PrefillTargetNll>,
}

/// Result from the DFlash prefill-append verifier.
pub struct PrefillAppendVerifyResult {
    pub logits: Vec<Vec<f32>>,
    pub target_next: Option<Vec<u32>>,
    pub tap_hiddens_all: Option<Vec<Vec<u8>>>,
    pub rollback: Option<PrefillAppendRollback>,
}

pub struct PrefillAppendGpuTapSink<'a> {
    pub buffer: &'a mut GpuBuffer,
    pub start_row: usize,
    pub row_bytes: usize,
}

/// Rollback material captured while verifying a candidate append block.
pub struct PrefillAppendRollback {
    pub pos_offset: usize,
    pub chunk_len: usize,
    pub per_layer: Vec<Option<PrefillAppendLayerRollback>>,
}

pub struct PrefillAppendLayerRollback {
    conv_input: GpuBuffer,
    recurrent_trace: GpuBuffer,
}

/// Result from the opt-in DFlash DDTree target verifier.
pub struct PrefillTreeVerifyResult {
    pub target_next: Vec<u32>,
    pub tap_hiddens_all: Option<Vec<Vec<u8>>>,
    pub tap_hiddens_gpu: bool,
    pub rollback: Option<PrefillTreeRollback>,
}

pub struct PrefillTreeRollback {
    prefix_len: usize,
    tree_len: usize,
    per_layer: Vec<Option<PrefillTreeLayerRollback>>,
}

enum PrefillTreeLayerRollback {
    Full {
        tree_k: GpuBuffer,
        tree_v: GpuBuffer,
    },
    Linear {
        conv_input: GpuBuffer,
        recurrent_trace: GpuBuffer,
    },
}

pub struct LinearLayerDebugTrace {
    #[allow(dead_code)]
    pub normed: Vec<u8>,
    pub qkv: Vec<u8>,
    pub qkv_tail: Vec<u8>,
    #[allow(dead_code)]
    pub conv_window: Vec<u8>,
    #[allow(dead_code)]
    pub post_conv: Vec<u8>,
    pub z: Vec<u8>,
    pub packed: Vec<u8>,
    pub rec_apply: Vec<u8>,
    pub attn: Vec<u8>,
    pub gated: Vec<u8>,
    pub proj_out: Vec<u8>,
}

/// Scratch buffers for prefill (larger than decode — seq_len > 1).
struct PrefillScratch {
    /// [seq_len, hidden_dim] BF16 — main hidden state
    hidden: GpuBuffer,
    /// [seq_len, hidden_dim] F32 — optional residual source of truth for
    /// Lucebox-default precision diagnostics.
    hidden_f32: Option<GpuBuffer>,
    /// [seq_len, hidden_dim] BF16 — normed activations
    normed: GpuBuffer,
    /// [seq_len, hidden_dim] F32 — optional normalization staging buffer.
    normed_f32: Option<GpuBuffer>,
    /// [hidden_dim] F32 — optional casted RMSNorm weight staging buffer.
    norm_weight_f32: Option<GpuBuffer>,
    /// [seq_len, max_proj_dim] BF16 — projection output buffer
    proj_buf: GpuBuffer,
    /// [seq_len, max_proj_dim] BF16 — second projection buffer (for gate/up)
    proj_buf2: GpuBuffer,
    /// [seq_len, hidden_dim] F32 — optional casted residual staging buffer.
    residual_f32: Option<GpuBuffer>,
    /// [seq_len, intermediate_size] BF16 — MLP intermediate
    mlp_buf: GpuBuffer,
    /// Grow-only Q8_1 activation workspace for optional Q6_K MMQ paths.
    q6_k_mmq_q8_workspace: Option<GpuBuffer>,
    /// [1, vocab_size] BF16 — logits output
    #[allow(dead_code)]
    logits_buf: GpuBuffer,
    // Full attention scratch:
    /// [num_q_heads, seq_len, head_dim] BF16 — transposed Q for attention
    attn_q: GpuBuffer,
    /// [num_kv_heads, seq_len, head_dim] BF16 — transposed K
    attn_k: GpuBuffer,
    /// [num_kv_heads, seq_len, head_dim] BF16 — transposed V
    attn_v: GpuBuffer,
    /// [num_q_heads, seq_len, head_dim] F32 — attention output
    attn_out_f32: GpuBuffer,
    /// [seq_len, max full-attention Q projection dim] BF16
    full_q_buf: GpuBuffer,
    /// [seq_len, num_q_heads * head_dim] BF16
    full_query_buf: GpuBuffer,
    /// [seq_len, num_q_heads * head_dim] BF16
    full_gate_buf: GpuBuffer,
    /// [seq_len, num_kv_heads * head_dim] BF16
    full_v_buf: GpuBuffer,
    // Linear attention scratch:
    /// [qkv_dim, seq_len + kern - 1] BF16 — padded conv input
    conv_input: GpuBuffer,
    /// [qkv_dim, kern - 1] BF16 — post-chunk convolution tail
    linear_new_tail: GpuBuffer,
    /// [seq_len, 2 * linear_num_value_heads] BF16 — fused B/A projection
    linear_ba_buf: GpuBuffer,
    /// [seq_len, linear_num_value_heads] BF16 — separate B projection fallback
    linear_b_buf: GpuBuffer,
    /// [seq_len, linear_num_value_heads] BF16 — separate A projection fallback
    linear_a_buf: GpuBuffer,
    /// [seq_len, linear_key_dim] F32 — split/cast Q
    linear_q_f32: GpuBuffer,
    /// [seq_len, linear_key_dim] F32 — split/cast K
    linear_k_f32: GpuBuffer,
    /// [seq_len, linear_value_dim] F32 — split/cast V
    linear_v_f32: GpuBuffer,
    /// [seq_len * linear_num_key_heads, linear_key_head_dim] F32
    linear_q_normed: GpuBuffer,
    /// [seq_len, linear_key_dim] F32
    linear_q_scaled: GpuBuffer,
    /// [seq_len * linear_num_key_heads, linear_key_head_dim] F32
    linear_k_normed: GpuBuffer,
    /// [linear_num_value_heads, seq_len] F32
    linear_beta: GpuBuffer,
    /// [linear_num_value_heads, seq_len] F32
    linear_g: GpuBuffer,
    /// [linear_num_value_heads, seq_len, linear_key_head_dim] F32
    linear_q_trans: GpuBuffer,
    /// [linear_num_value_heads, seq_len, linear_key_head_dim] F32
    linear_k_trans: GpuBuffer,
    /// [linear_num_value_heads, seq_len, linear_value_head_dim] F32
    linear_v_trans: GpuBuffer,
    /// [linear_num_value_heads, seq_len + linear_key_head_dim, linear_value_head_dim] F32
    linear_delta_out: GpuBuffer,
    /// [linear_num_value_heads, seq_len, linear_value_head_dim] BF16
    linear_attn_output: GpuBuffer,
    /// [linear_num_value_heads, seq_len, linear_value_head_dim] BF16
    linear_z_trans: GpuBuffer,
    /// [linear_num_value_heads * seq_len, linear_value_head_dim] BF16
    linear_gated_out: GpuBuffer,
    /// [seq_len, linear_value_dim] BF16
    linear_gated_s_first: GpuBuffer,
    /// [linear_num_value_heads, linear_key_head_dim, linear_value_head_dim] F32
    linear_dummy_state: GpuBuffer,
    /// [tree_len, num_taps * hidden_dim] BF16, used by cached DDTree verify only.
    tree_tap_capture_gpu: Option<GpuBuffer>,
    tree_tap_capture_row_bytes: usize,
    tree_tap_capture_rows: usize,
}

impl PrefillScratch {
    fn new(config: &TextConfig, seq_len: usize, ordinal: usize) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kern = config.linear_conv_kernel_dim;
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;

        // Max projection dim across all layer types and MLP
        let max_proj = [
            // Full attention: q_proj (doubled for gate)
            num_q_heads * head_dim * 2,
            // Linear attention: qkv_out
            config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim,
            // MLP fused gate/up projection when GGML low-bit rows are available
            intermediate * 2,
            // MLP: intermediate_size (gate/up projection output)
            intermediate,
        ]
        .into_iter()
        .max()
        .unwrap();

        let key_dim = nk * khd;
        let val_dim = nv * vhd;
        let qkv_dim = key_dim * 2 + val_dim;
        let pad = kern - 1;
        let conv_total_len = seq_len + kern - 1;
        let f32_activation_carry = prefill_f32_activation_carry_enabled();

        Ok(Self {
            hidden: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill hidden: {e}"))?,
            hidden_f32: if f32_activation_carry {
                Some(
                    GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("prefill hidden_f32: {e}"))?,
                )
            } else {
                None
            },
            normed: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill normed: {e}"))?,
            normed_f32: if f32_activation_carry {
                Some(
                    GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("prefill normed_f32: {e}"))?,
                )
            } else {
                None
            },
            norm_weight_f32: if f32_activation_carry {
                Some(
                    GpuBuffer::alloc(ordinal, ScalarType::F32, &[hidden_dim])
                        .map_err(|e| anyhow::anyhow!("prefill norm_weight_f32: {e}"))?,
                )
            } else {
                None
            },
            proj_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf: {e}"))?,
            proj_buf2: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf2: {e}"))?,
            residual_f32: if f32_activation_carry {
                Some(
                    GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("prefill residual_f32: {e}"))?,
                )
            } else {
                None
            },
            mlp_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, intermediate])
                .map_err(|e| anyhow::anyhow!("prefill mlp_buf: {e}"))?,
            q6_k_mmq_q8_workspace: None,
            logits_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[1, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("prefill logits: {e}"))?,
            attn_q: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[num_q_heads, seq_len, head_dim])
                .map_err(|e| anyhow::anyhow!("prefill attn_q: {e}"))?,
            attn_k: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_k: {e}"))?,
            attn_v: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_v: {e}"))?,
            attn_out_f32: GpuBuffer::alloc(
                ordinal,
                ScalarType::F32,
                &[num_q_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_out_f32: {e}"))?,
            full_q_buf: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[seq_len, num_q_heads * head_dim * 2],
            )
            .map_err(|e| anyhow::anyhow!("prefill full_q_buf: {e}"))?,
            full_query_buf: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[seq_len, num_q_heads * head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill full_query_buf: {e}"))?,
            full_gate_buf: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[seq_len, num_q_heads * head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill full_gate_buf: {e}"))?,
            full_v_buf: GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[seq_len, num_kv_heads * head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill full_v_buf: {e}"))?,
            conv_input: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, conv_total_len])
                .map_err(|e| anyhow::anyhow!("prefill conv_input: {e}"))?,
            linear_new_tail: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, pad])
                .map_err(|e| anyhow::anyhow!("prefill linear_new_tail: {e}"))?,
            linear_ba_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, 2 * nv])
                .map_err(|e| anyhow::anyhow!("prefill linear_ba_buf: {e}"))?,
            linear_b_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, nv])
                .map_err(|e| anyhow::anyhow!("prefill linear_b_buf: {e}"))?,
            linear_a_buf: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, nv])
                .map_err(|e| anyhow::anyhow!("prefill linear_a_buf: {e}"))?,
            linear_q_f32: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, key_dim])
                .map_err(|e| anyhow::anyhow!("prefill linear_q_f32: {e}"))?,
            linear_k_f32: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, key_dim])
                .map_err(|e| anyhow::anyhow!("prefill linear_k_f32: {e}"))?,
            linear_v_f32: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, val_dim])
                .map_err(|e| anyhow::anyhow!("prefill linear_v_f32: {e}"))?,
            linear_q_normed: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len * nk, khd])
                .map_err(|e| anyhow::anyhow!("prefill linear_q_normed: {e}"))?,
            linear_q_scaled: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len, key_dim])
                .map_err(|e| anyhow::anyhow!("prefill linear_q_scaled: {e}"))?,
            linear_k_normed: GpuBuffer::alloc(ordinal, ScalarType::F32, &[seq_len * nk, khd])
                .map_err(|e| anyhow::anyhow!("prefill linear_k_normed: {e}"))?,
            linear_beta: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len])
                .map_err(|e| anyhow::anyhow!("prefill linear_beta: {e}"))?,
            linear_g: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len])
                .map_err(|e| anyhow::anyhow!("prefill linear_g: {e}"))?,
            linear_q_trans: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len, khd])
                .map_err(|e| anyhow::anyhow!("prefill linear_q_trans: {e}"))?,
            linear_k_trans: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len, khd])
                .map_err(|e| anyhow::anyhow!("prefill linear_k_trans: {e}"))?,
            linear_v_trans: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_v_trans: {e}"))?,
            linear_delta_out: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, seq_len + khd, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_delta_out: {e}"))?,
            linear_attn_output: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[nv, seq_len, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_attn_output: {e}"))?,
            linear_z_trans: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[nv, seq_len, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_z_trans: {e}"))?,
            linear_gated_out: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[nv * seq_len, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_gated_out: {e}"))?,
            linear_gated_s_first: GpuBuffer::alloc(ordinal, ScalarType::BF16, &[seq_len, val_dim])
                .map_err(|e| anyhow::anyhow!("prefill linear_gated_s_first: {e}"))?,
            linear_dummy_state: GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, khd, vhd])
                .map_err(|e| anyhow::anyhow!("prefill linear_dummy_state: {e}"))?,
            tree_tap_capture_gpu: None,
            tree_tap_capture_row_bytes: 0,
            tree_tap_capture_rows: 0,
        })
    }
}

#[derive(Clone, Copy)]
enum ResidualSource {
    ProjBuf,
    ProjBuf2,
}

fn prefill_f32_activation_carry_enabled() -> bool {
    env::var_os("SUPERSONIC_QWEN35_PREFILL_F32_ACTIVATION_CARRY").is_some()
}

impl PrefillScratch {
    fn has_f32_activation_carry(&self) -> bool {
        self.hidden_f32.is_some()
    }

    fn seed_f32_from_hidden(&mut self, ordinal: usize, elems: usize, label: &str) -> Result<()> {
        if let Some(hidden_f32) = self.hidden_f32.as_mut() {
            prefill_ffi::cast(
                ordinal,
                ScalarType::BF16,
                ScalarType::F32,
                elems,
                &self.hidden,
                hidden_f32,
            )
            .map_err(|e| anyhow::anyhow!("{label} seed hidden_f32: {e}"))?;
        }
        Ok(())
    }

    fn materialize_hidden_bf16_from_f32(
        &mut self,
        ordinal: usize,
        elems: usize,
        label: &str,
    ) -> Result<()> {
        if let Some(hidden_f32) = self.hidden_f32.as_ref() {
            prefill_ffi::cast(
                ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                elems,
                hidden_f32,
                &mut self.hidden,
            )
            .map_err(|e| anyhow::anyhow!("{label} materialize hidden BF16: {e}"))?;
        }
        Ok(())
    }

    fn rms_norm_hidden_to_normed_model(
        &mut self,
        config: &TextConfig,
        ordinal: usize,
        rows: usize,
        cols: usize,
        weight: &GpuBuffer,
        label: &str,
    ) -> Result<()> {
        if !self.has_f32_activation_carry() {
            return rms_norm_rows_model(
                config,
                ordinal,
                rows,
                cols,
                &self.hidden,
                weight,
                &mut self.normed,
                label,
            );
        }

        let op = if config.rms_norm_add_unit_offset {
            prefill_ffi::rms_norm_rows
        } else {
            prefill_ffi::rms_norm_rows_plain
        };
        let hidden_f32 = self
            .hidden_f32
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{label}: missing hidden_f32"))?;
        let normed_f32 = self
            .normed_f32
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("{label}: missing normed_f32"))?;
        let norm_weight_f32 = self
            .norm_weight_f32
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("{label}: missing norm_weight_f32"))?;

        prefill_ffi::cast(
            ordinal,
            weight.dtype(),
            ScalarType::F32,
            cols,
            weight,
            norm_weight_f32,
        )
        .map_err(|e| anyhow::anyhow!("{label} cast norm weight to F32: {e}"))?;
        op(
            ordinal,
            ScalarType::F32,
            rows,
            cols,
            config.rms_norm_eps as f32,
            hidden_f32,
            norm_weight_f32,
            normed_f32,
        )
        .map_err(|e| anyhow::anyhow!("{label} F32: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            rows * cols,
            normed_f32,
            &mut self.normed,
        )
        .map_err(|e| anyhow::anyhow!("{label} cast normed to BF16: {e}"))?;
        Ok(())
    }

    fn residual_add_from_source(
        &mut self,
        ordinal: usize,
        total_elems: usize,
        source: ResidualSource,
        label: &str,
    ) -> Result<()> {
        if !self.has_f32_activation_carry() {
            let src = match source {
                ResidualSource::ProjBuf => &self.proj_buf,
                ResidualSource::ProjBuf2 => &self.proj_buf2,
            };
            return residual_add(ordinal, total_elems, &mut self.hidden, src)
                .map_err(|e| anyhow::anyhow!("{label}: {e}"));
        }

        let src = match source {
            ResidualSource::ProjBuf => &self.proj_buf,
            ResidualSource::ProjBuf2 => &self.proj_buf2,
        };
        let residual_f32 = self
            .residual_f32
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("{label}: missing residual_f32"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            total_elems,
            src,
            residual_f32,
        )
        .map_err(|e| anyhow::anyhow!("{label} cast residual to F32: {e}"))?;

        let hidden_f32 = self
            .hidden_f32
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("{label}: missing hidden_f32"))?;
        let lhs: &GpuBuffer = unsafe { &*(hidden_f32 as *const GpuBuffer) };
        prefill_ffi::element_add(
            ordinal,
            ScalarType::F32,
            total_elems,
            lhs,
            residual_f32,
            hidden_f32,
        )
        .map_err(|e| anyhow::anyhow!("{label} F32 residual add: {e}"))?;
        self.materialize_hidden_bf16_from_f32(ordinal, total_elems, label)?;
        Ok(())
    }

    fn ensure_tree_tap_capture(
        &mut self,
        ordinal: usize,
        tree_len: usize,
        tap_count: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        if tree_len == 0 || tap_count == 0 {
            self.tree_tap_capture_gpu = None;
            self.tree_tap_capture_row_bytes = 0;
            self.tree_tap_capture_rows = 0;
            return Ok(());
        }
        let row_elems = tap_count * hidden_dim;
        let row_bytes = row_elems * ScalarType::BF16.size_in_bytes();
        let needs_alloc = self
            .tree_tap_capture_gpu
            .as_ref()
            .map(|buf| buf.len_bytes() < tree_len * row_bytes)
            .unwrap_or(true)
            || self.tree_tap_capture_row_bytes != row_bytes
            || self.tree_tap_capture_rows < tree_len;
        if needs_alloc {
            self.tree_tap_capture_gpu = Some(
                GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, row_elems])
                    .map_err(|e| anyhow::anyhow!("tree tap capture alloc: {e}"))?,
            );
            self.tree_tap_capture_row_bytes = row_bytes;
            self.tree_tap_capture_rows = tree_len;
        }
        Ok(())
    }

    fn tree_tap_capture_sink(
        &mut self,
        start_row: usize,
        row_bytes: usize,
    ) -> Result<PrefillAppendGpuTapSink<'_>> {
        let buffer = self
            .tree_tap_capture_gpu
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("tree tap capture requested before allocation"))?;
        Ok(PrefillAppendGpuTapSink {
            buffer,
            start_row,
            row_bytes,
        })
    }

    fn copy_hidden_to_tree_tap_capture(
        &mut self,
        ordinal: usize,
        tap_slot: usize,
        tap_count: usize,
        tree_len: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        let row_bytes = self.tree_tap_capture_row_bytes;
        let hidden: &GpuBuffer = unsafe { &*(&self.hidden as *const GpuBuffer) };
        let mut sink = self.tree_tap_capture_sink(0, row_bytes)?;
        copy_tap_rows_to_gpu_history(
            ordinal, &mut sink, tap_slot, tap_count, hidden, tree_len, hidden_dim,
        )
    }

    fn copy_tree_tap_capture_to_gpu_history(
        &self,
        ordinal: usize,
        accepted_indices: &[usize],
        commit_len: usize,
        tap_history_gpu: &mut GpuBuffer,
        start_row: usize,
        row_bytes: usize,
    ) -> Result<()> {
        if commit_len == 0 {
            return Ok(());
        }
        if accepted_indices.len() < commit_len {
            return Err(anyhow::anyhow!(
                "tree tap gather has {} accepted indices, need {commit_len}",
                accepted_indices.len()
            ));
        }
        if row_bytes == 0 || self.tree_tap_capture_row_bytes != row_bytes {
            return Err(anyhow::anyhow!(
                "tree tap capture row_bytes {} != destination row_bytes {}",
                self.tree_tap_capture_row_bytes,
                row_bytes
            ));
        }
        let capture = self
            .tree_tap_capture_gpu
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tree tap capture buffer is missing"))?;
        let dst_start = start_row * row_bytes;
        if dst_start + commit_len * row_bytes > tap_history_gpu.len_bytes() {
            return Err(anyhow::anyhow!(
                "tree tap gather destination exceeds buffer: offset {} + len {} > {}",
                dst_start,
                commit_len * row_bytes,
                tap_history_gpu.len_bytes()
            ));
        }
        for (out_row, &tree_row) in accepted_indices.iter().take(commit_len).enumerate() {
            if tree_row >= self.tree_tap_capture_rows {
                return Err(anyhow::anyhow!(
                    "tree tap gather row {tree_row} out of range {}",
                    self.tree_tap_capture_rows
                ));
            }
            let dst_off = dst_start + out_row * row_bytes;
            let src_off = tree_row * row_bytes;
            let dst =
                unsafe { (tap_history_gpu.as_mut_ptr() as *mut u8).add(dst_off) as *mut c_void };
            copy_d2d_batched(ordinal, dst, capture.offset_ptr(src_off), row_bytes)
                .map_err(|e| anyhow::anyhow!("tree tap gather row {out_row}: {e}"))?;
        }
        Ok(())
    }
}

pub struct PrefillAppendVerifyCache {
    chunk_len: usize,
    ordinal: usize,
    scratch: PrefillScratch,
    chunk_conv_tail: Vec<Option<GpuBuffer>>,
    token_ids_gpu: GpuBuffer,
    rollback: Option<PrefillAppendRollback>,
}

pub struct PrefillTreeVerifyCache {
    tree_len: usize,
    ordinal: usize,
    scratch: PrefillScratch,
    token_ids_gpu: GpuBuffer,
    positions_gpu: GpuBuffer,
    parent_ids_gpu: GpuBuffer,
    conv_source_cols_gpu: GpuBuffer,
    conv_source_cols_stride: usize,
    visibility_gpu: GpuBuffer,
    greedy_logits_gpu: GpuBuffer,
    greedy_indices_gpu: GpuBuffer,
    greedy_block_best_vals_gpu: GpuBuffer,
    greedy_block_best_indices_gpu: GpuBuffer,
    token_id_bytes: Vec<u8>,
    position_bytes: Vec<u8>,
    parent_id_bytes: Vec<u8>,
    conv_source_col_bytes: Vec<u8>,
    rollback: Option<PrefillTreeRollback>,
}

impl PrefillTreeVerifyCache {
    pub fn new(config: &TextConfig, tree_len: usize, ordinal: usize) -> Result<Self> {
        let scratch = PrefillScratch::new(config, tree_len, ordinal)?;
        let token_ids_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[tree_len])
            .map_err(|e| anyhow::anyhow!("tree cache token ids alloc: {e}"))?;
        let positions_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[tree_len])
            .map_err(|e| anyhow::anyhow!("tree cache positions alloc: {e}"))?;
        let parent_ids_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[tree_len])
            .map_err(|e| anyhow::anyhow!("tree cache parent ids alloc: {e}"))?;
        let conv_source_cols_stride = config.linear_conv_kernel_dim.max(1);
        let conv_source_cols_gpu = GpuBuffer::alloc(
            ordinal,
            ScalarType::U32,
            &[tree_len, conv_source_cols_stride],
        )
        .map_err(|e| anyhow::anyhow!("tree cache conv source cols alloc: {e}"))?;
        let visibility_gpu = GpuBuffer::alloc(ordinal, ScalarType::U8, &[tree_len, tree_len])
            .map_err(|e| anyhow::anyhow!("tree cache visibility alloc: {e}"))?;
        let greedy_logits_gpu =
            GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("tree cache greedy logits alloc: {e}"))?;
        let greedy_indices_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[tree_len])
            .map_err(|e| anyhow::anyhow!("tree cache greedy indices alloc: {e}"))?;
        let lm_head_tiles = (config.vocab_size + 15) / 16;
        let greedy_block_best_vals_gpu =
            GpuBuffer::alloc(ordinal, ScalarType::F32, &[tree_len, lm_head_tiles])
                .map_err(|e| anyhow::anyhow!("tree cache greedy block-best vals alloc: {e}"))?;
        let greedy_block_best_indices_gpu =
            GpuBuffer::alloc(ordinal, ScalarType::U32, &[tree_len, lm_head_tiles])
                .map_err(|e| anyhow::anyhow!("tree cache greedy block-best indices alloc: {e}"))?;

        Ok(Self {
            tree_len,
            ordinal,
            scratch,
            token_ids_gpu,
            positions_gpu,
            parent_ids_gpu,
            conv_source_cols_gpu,
            conv_source_cols_stride,
            visibility_gpu,
            greedy_logits_gpu,
            greedy_indices_gpu,
            greedy_block_best_vals_gpu,
            greedy_block_best_indices_gpu,
            token_id_bytes: Vec::with_capacity(tree_len * 4),
            position_bytes: Vec::with_capacity(tree_len * 4),
            parent_id_bytes: Vec::with_capacity(tree_len * 4),
            conv_source_col_bytes: Vec::with_capacity(tree_len * conv_source_cols_stride * 4),
            rollback: None,
        })
    }

    fn matches(&self, tree_len: usize, ordinal: usize) -> bool {
        self.tree_len == tree_len && self.ordinal == ordinal
    }

    fn take_rollback(
        &mut self,
        config: &TextConfig,
        prefix_len: usize,
    ) -> Result<PrefillTreeRollback> {
        if let Some(mut rollback) = self.rollback.take() {
            if tree_rollback_matches(config, &rollback, self.tree_len, self.ordinal) {
                rollback.prefix_len = prefix_len;
                rollback.tree_len = self.tree_len;
                return Ok(rollback);
            }
        }
        alloc_tree_rollback(config, self.tree_len, prefix_len, self.ordinal)
    }

    pub fn recycle_rollback(&mut self, rollback: PrefillTreeRollback) {
        if rollback.tree_len == self.tree_len {
            self.rollback = Some(rollback);
        }
    }

    fn upload_inputs(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        parent_ids: &[i32],
        visibility: &[u8],
    ) -> Result<()> {
        self.token_id_bytes.clear();
        self.token_id_bytes.reserve(token_ids.len() * 4);
        for &id in token_ids {
            self.token_id_bytes.extend_from_slice(&id.to_le_bytes());
        }
        copy_h2d(
            self.ordinal,
            self.token_ids_gpu.as_mut_ptr(),
            self.token_id_bytes.as_ptr() as *const c_void,
            self.token_id_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("tree verify upload token IDs: {e}"))?;

        self.position_bytes.clear();
        self.position_bytes.reserve(positions.len() * 4);
        for &pos in positions {
            self.position_bytes
                .extend_from_slice(&(pos as u32).to_le_bytes());
        }
        copy_h2d(
            self.ordinal,
            self.positions_gpu.as_mut_ptr(),
            self.position_bytes.as_ptr() as *const c_void,
            self.position_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("tree verify upload positions: {e}"))?;

        self.parent_id_bytes.clear();
        self.parent_id_bytes.reserve(parent_ids.len() * 4);
        for &parent in parent_ids {
            self.parent_id_bytes
                .extend_from_slice(&parent.to_le_bytes());
        }
        copy_h2d(
            self.ordinal,
            self.parent_ids_gpu.as_mut_ptr(),
            self.parent_id_bytes.as_ptr() as *const c_void,
            self.parent_id_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("tree verify upload parent IDs: {e}"))?;

        self.conv_source_col_bytes.clear();
        self.conv_source_col_bytes
            .reserve(parent_ids.len() * self.conv_source_cols_stride * 4);
        for t in 0..parent_ids.len() {
            for tap in 0..self.conv_source_cols_stride {
                let source_col =
                    tree_conv_source_col(parent_ids, t, tap, self.conv_source_cols_stride)?;
                self.conv_source_col_bytes
                    .extend_from_slice(&(source_col as u32).to_le_bytes());
            }
        }
        copy_h2d(
            self.ordinal,
            self.conv_source_cols_gpu.as_mut_ptr(),
            self.conv_source_col_bytes.as_ptr() as *const c_void,
            self.conv_source_col_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("tree verify upload conv source columns: {e}"))?;

        copy_h2d(
            self.ordinal,
            self.visibility_gpu.as_mut_ptr(),
            visibility.as_ptr() as *const c_void,
            visibility.len(),
        )
        .map_err(|e| anyhow::anyhow!("tree verify upload visibility: {e}"))?;

        Ok(())
    }

    pub fn copy_tap_capture_to_gpu_history(
        &self,
        accepted_indices: &[usize],
        commit_len: usize,
        tap_history_gpu: &mut GpuBuffer,
        start_row: usize,
        row_bytes: usize,
    ) -> Result<()> {
        self.scratch.copy_tree_tap_capture_to_gpu_history(
            self.ordinal,
            accepted_indices,
            commit_len,
            tap_history_gpu,
            start_row,
            row_bytes,
        )
    }

    fn compute_greedy_ids(
        &mut self,
        weights: &Qwen35Weights,
        config: &TextConfig,
        use_4b_kernel: bool,
    ) -> Result<Vec<u32>> {
        if gpu_hal::current_backend() != Backend::Hip {
            let (ids, _normed) = compute_greedy_for_range(
                &self.scratch.hidden,
                weights,
                config,
                0,
                self.tree_len,
                use_4b_kernel,
                self.ordinal,
            )?;
            return Ok(ids);
        }

        let hidden_dim = config.hidden_size;
        let vocab_size = config.vocab_size;
        rms_norm_rows_model(
            config,
            self.ordinal,
            self.tree_len,
            hidden_dim,
            &self.scratch.hidden,
            &weights.norm_weight,
            &mut self.scratch.normed,
            "tree cache greedy final norm",
        )?;

        let mut fused_argmax = false;
        if let Some((qtype, scale, zero)) = weights.lm_head_lowbit_params(hidden_dim) {
            fused_argmax = maybe_matmul_q6_k_lm_head_argmax(
                self.ordinal,
                1,
                self.tree_len,
                vocab_size,
                hidden_dim,
                qtype,
                weights.lm_head_awq_inv_scale.as_ref(),
                &self.scratch.normed,
                &*weights.lm_head,
                &mut self.greedy_block_best_vals_gpu,
                &mut self.greedy_block_best_indices_gpu,
                &mut self.greedy_indices_gpu,
            )?;
            if !fused_argmax {
                if !maybe_matmul_q6_k_mmq_lm_head(
                    self.ordinal,
                    1,
                    self.tree_len,
                    vocab_size,
                    hidden_dim,
                    qtype,
                    weights.lm_head_awq_inv_scale.as_ref(),
                    &self.scratch.normed,
                    &*weights.lm_head,
                    &mut self.greedy_logits_gpu,
                )? {
                    prefill_ffi::matmul_rhs_transposed_int4(
                        self.ordinal,
                        1,
                        self.tree_len,
                        vocab_size,
                        hidden_dim,
                        &self.scratch.normed,
                        &*weights.lm_head,
                        scale,
                        zero,
                        weights.lm_head_awq_inv_scale.as_ref(),
                        weights.int4_group_size,
                        qtype,
                        &mut self.greedy_logits_gpu,
                    )
                    .map_err(|e| anyhow::anyhow!("tree cache greedy lm_head int4: {e}"))?;
                }
            }
        } else if self.tree_len > 1 {
            kernel_ffi::matmul_rhs_transposed_4b(
                self.ordinal,
                ScalarType::BF16,
                1,
                self.tree_len,
                vocab_size,
                hidden_dim,
                &self.scratch.normed,
                &*weights.lm_head,
                &mut self.greedy_logits_gpu,
            )
            .map_err(|e| anyhow::anyhow!("tree cache greedy lm_head tiled: {e}"))?;
        } else {
            let mut counter = GpuBuffer::zeros(self.ordinal, ScalarType::U32, &[1])
                .map_err(|e| anyhow::anyhow!("tree cache greedy matvec counter: {e}"))?;
            kernel_ffi::standalone_matvec(
                self.ordinal,
                ScalarType::BF16,
                &mut self.greedy_logits_gpu,
                &self.scratch.normed,
                &*weights.lm_head,
                hidden_dim,
                vocab_size,
                &mut counter,
            )
            .map_err(|e| anyhow::anyhow!("tree cache greedy lm_head matvec: {e}"))?;
        }

        if !fused_argmax {
            prefill_ffi::argmax_bf16_rows(
                self.ordinal,
                self.tree_len,
                vocab_size,
                &self.greedy_logits_gpu,
                &mut self.greedy_indices_gpu,
            )
            .map_err(|e| anyhow::anyhow!("tree cache greedy argmax: {e}"))?;
        }
        let ids_bytes = self
            .greedy_indices_gpu
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("tree cache greedy ids D2H: {e}"))?;
        Ok(ids_bytes
            .chunks_exact(4)
            .take(self.tree_len)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

impl PrefillAppendVerifyCache {
    pub fn new(config: &TextConfig, chunk_len: usize, ordinal: usize) -> Result<Self> {
        let kern = config.linear_conv_kernel_dim;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;
        let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
        let scratch = PrefillScratch::new(config, chunk_len, ordinal)?;
        let chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_full_attention(i) {
                    Ok(None)
                } else {
                    GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                        .map(Some)
                        .map_err(|e| anyhow::anyhow!("append cache conv_tail alloc: {e}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[chunk_len])
            .map_err(|e| anyhow::anyhow!("append cache token ids alloc: {e}"))?;

        Ok(Self {
            chunk_len,
            ordinal,
            scratch,
            chunk_conv_tail,
            token_ids_gpu,
            rollback: None,
        })
    }

    fn matches(&self, chunk_len: usize, ordinal: usize) -> bool {
        self.chunk_len == chunk_len && self.ordinal == ordinal
    }

    fn take_rollback(
        &mut self,
        config: &TextConfig,
        pos_offset: usize,
        ordinal: usize,
    ) -> Result<PrefillAppendRollback> {
        if let Some(mut rollback) = self.rollback.take() {
            if append_rollback_matches(config, &rollback, self.chunk_len, ordinal) {
                rollback.pos_offset = pos_offset;
                rollback.chunk_len = self.chunk_len;
                return Ok(rollback);
            }
        }
        alloc_append_rollback(config, self.chunk_len, pos_offset, ordinal)
    }

    pub fn recycle_rollback(&mut self, rollback: PrefillAppendRollback) {
        if rollback.chunk_len == self.chunk_len {
            self.rollback = Some(rollback);
        }
    }
}

fn alloc_tree_rollback(
    config: &TextConfig,
    tree_len: usize,
    prefix_len: usize,
    ordinal: usize,
) -> Result<PrefillTreeRollback> {
    let kern = config.linear_conv_kernel_dim;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
    let recurrent_trace_dtype = dflash_rollback_trace_dtype();
    let mut per_layer = Vec::with_capacity(config.num_hidden_layers);
    for idx in 0..config.num_hidden_layers {
        if config.is_full_attention(idx) {
            let tree_k = GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[config.num_key_value_heads, tree_len, config.head_dim],
            )
            .map_err(|e| anyhow::anyhow!("tree rollback K layer {idx}: {e}"))?;
            let tree_v = GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[config.num_key_value_heads, tree_len, config.head_dim],
            )
            .map_err(|e| anyhow::anyhow!("tree rollback V layer {idx}: {e}"))?;
            per_layer.push(Some(PrefillTreeLayerRollback::Full { tree_k, tree_v }));
        } else {
            let conv_input =
                GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1 + tree_len])
                    .map_err(|e| anyhow::anyhow!("tree rollback conv_input layer {idx}: {e}"))?;
            let recurrent_trace = if recurrent_trace_dtype == ScalarType::U8 {
                GpuBuffer::alloc(
                    ordinal,
                    recurrent_trace_dtype,
                    &[dflash_q8_trace_bytes(
                        config.linear_num_value_heads,
                        tree_len,
                        khd,
                        vhd,
                    )],
                )
            } else {
                GpuBuffer::alloc(
                    ordinal,
                    recurrent_trace_dtype,
                    &[config.linear_num_value_heads, tree_len, khd, vhd],
                )
            }
            .map_err(|e| anyhow::anyhow!("tree rollback recurrent trace layer {idx}: {e}"))?;
            per_layer.push(Some(PrefillTreeLayerRollback::Linear {
                conv_input,
                recurrent_trace,
            }));
        }
    }
    Ok(PrefillTreeRollback {
        prefix_len,
        tree_len,
        per_layer,
    })
}

fn tree_rollback_matches(
    config: &TextConfig,
    rollback: &PrefillTreeRollback,
    tree_len: usize,
    ordinal: usize,
) -> bool {
    if rollback.tree_len != tree_len || rollback.per_layer.len() != config.num_hidden_layers {
        return false;
    }
    let kern = config.linear_conv_kernel_dim;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
    let expected_trace_dtype = dflash_rollback_trace_dtype();
    let expected_trace_bytes = if expected_trace_dtype == ScalarType::U8 {
        dflash_q8_trace_bytes(config.linear_num_value_heads, tree_len, khd, vhd)
    } else {
        config.linear_num_value_heads * tree_len * khd * vhd * expected_trace_dtype.size_in_bytes()
    };
    let expected_conv_bytes = qkv_dim * (kern - 1 + tree_len) * ScalarType::BF16.size_in_bytes();
    let expected_kv_bytes =
        config.num_key_value_heads * tree_len * config.head_dim * ScalarType::BF16.size_in_bytes();

    for (idx, layer) in rollback.per_layer.iter().enumerate() {
        if config.is_full_attention(idx) {
            let Some(PrefillTreeLayerRollback::Full { tree_k, tree_v }) = layer else {
                return false;
            };
            if tree_k.device_ordinal() != ordinal
                || tree_v.device_ordinal() != ordinal
                || tree_k.dtype() != ScalarType::BF16
                || tree_v.dtype() != ScalarType::BF16
                || tree_k.len_bytes() < expected_kv_bytes
                || tree_v.len_bytes() < expected_kv_bytes
            {
                return false;
            }
        } else {
            let Some(PrefillTreeLayerRollback::Linear {
                conv_input,
                recurrent_trace,
            }) = layer
            else {
                return false;
            };
            if conv_input.device_ordinal() != ordinal
                || recurrent_trace.device_ordinal() != ordinal
                || conv_input.dtype() != ScalarType::BF16
                || recurrent_trace.dtype() != expected_trace_dtype
                || conv_input.len_bytes() < expected_conv_bytes
                || recurrent_trace.len_bytes() < expected_trace_bytes
            {
                return false;
            }
        }
    }
    true
}

fn alloc_append_rollback(
    config: &TextConfig,
    chunk_len: usize,
    pos_offset: usize,
    ordinal: usize,
) -> Result<PrefillAppendRollback> {
    let kern = config.linear_conv_kernel_dim;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
    let recurrent_trace_dtype = dflash_rollback_trace_dtype();
    let mut per_layer = Vec::with_capacity(config.num_hidden_layers);
    for idx in 0..config.num_hidden_layers {
        if config.is_full_attention(idx) {
            per_layer.push(None);
        } else {
            let conv_input =
                GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1 + chunk_len])
                    .map_err(|e| anyhow::anyhow!("append rollback conv_input layer {idx}: {e}"))?;
            let recurrent_trace = if recurrent_trace_dtype == ScalarType::U8 {
                GpuBuffer::alloc(
                    ordinal,
                    recurrent_trace_dtype,
                    &[dflash_q8_trace_bytes(
                        config.linear_num_value_heads,
                        chunk_len,
                        khd,
                        vhd,
                    )],
                )
            } else {
                GpuBuffer::alloc(
                    ordinal,
                    recurrent_trace_dtype,
                    &[config.linear_num_value_heads, chunk_len, khd, vhd],
                )
            }
            .map_err(|e| anyhow::anyhow!("append rollback recurrent trace layer {idx}: {e}"))?;
            per_layer.push(Some(PrefillAppendLayerRollback {
                conv_input,
                recurrent_trace,
            }));
        }
    }
    Ok(PrefillAppendRollback {
        pos_offset,
        chunk_len,
        per_layer,
    })
}

fn append_rollback_matches(
    config: &TextConfig,
    rollback: &PrefillAppendRollback,
    chunk_len: usize,
    ordinal: usize,
) -> bool {
    if rollback.chunk_len != chunk_len || rollback.per_layer.len() != config.num_hidden_layers {
        return false;
    }
    let kern = config.linear_conv_kernel_dim;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
    let expected_conv_bytes = qkv_dim * (kern - 1 + chunk_len) * ScalarType::BF16.size_in_bytes();
    let expected_trace_dtype = dflash_rollback_trace_dtype();
    let expected_trace_bytes = if expected_trace_dtype == ScalarType::U8 {
        dflash_q8_trace_bytes(config.linear_num_value_heads, chunk_len, khd, vhd)
    } else {
        config.linear_num_value_heads * chunk_len * khd * vhd * expected_trace_dtype.size_in_bytes()
    };
    for (idx, layer) in rollback.per_layer.iter().enumerate() {
        match (config.is_full_attention(idx), layer) {
            (true, None) => {}
            (false, Some(layer)) => {
                if layer.conv_input.device_ordinal() != ordinal
                    || layer.conv_input.dtype() != ScalarType::BF16
                    || layer.conv_input.len_bytes() != expected_conv_bytes
                    || layer.recurrent_trace.device_ordinal() != ordinal
                    || layer.recurrent_trace.dtype() != expected_trace_dtype
                    || layer.recurrent_trace.len_bytes() != expected_trace_bytes
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Run native prefill on GPU, returning logits and leaving state filled.
/// When `prefill_chunk_size > 0`, processes the prompt in chunks to reduce activation VRAM.
pub fn prefill(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
) -> Result<PrefillResult> {
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        trace_layers,
        debug_linear_layer,
        None,
        None,
        None,
        None, // last_layer
    )
}

/// DFlash variant of `prefill`. Identical behavior plus selective per-layer
/// hidden-state capture: when `tap_layers` is supplied, the returned
/// `PrefillResult.tap_hiddens` carries one BF16 `[hidden_dim]` blob per tap
/// (the post-MLP residual hidden state at the LAST token of the final chunk).
pub fn prefill_with_taps(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
    tap_layers: &[usize],
) -> Result<PrefillResult> {
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        trace_layers,
        debug_linear_layer,
        Some(tap_layers),
        None,
        None,
        None, // last_layer
    )
}

pub fn prefill_with_target_nll(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    score_hidden_start: usize,
    score_targets: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
) -> Result<PrefillResult> {
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        0,
        kv_fp8,
        use_4b_kernel,
        false,
        None,
        None,
        Some((score_hidden_start, score_targets)),
        None,
        None, // last_layer
    )
}

/// SpecPrefill (arXiv 2502.02789) target sparse prefill: like `prefill`,
/// but consumes a sorted ascending `kept_positions` slice. The compacted
/// embedding sequence is `prompt_ids[kept_positions[i]] for i in 0..len`,
/// each token rotates by its ORIGINAL prompt position via the
/// RoPE-indirect kernel (Phase B), and the lower-triangular causal mask
/// over the compacted sequence is exactly the right semantics — Phase B
/// parity tests pin this.
///
/// Post-condition: `kv_filled` on every full-attention layer equals
/// `kept_positions.len()`. The caller's decode-position cursor must
/// nonetheless start at `prompt_ids.len()` (the original prompt's last
/// position + 1), NOT `kept_positions.len()`.
pub fn prefill_kept(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    kept_positions: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
) -> Result<PrefillResult> {
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        false,                // trace_layers
        None,                 // debug_linear_layer
        None,                 // tap_layers
        None,                 // target_nll
        Some(kept_positions), // kept_positions
        None,                 // last_layer
    )
}

#[allow(dead_code)]
fn copy_bf16_row(
    ordinal: usize,
    source: &GpuBuffer,
    row: usize,
    cols: usize,
    label: &str,
) -> Result<Vec<u8>> {
    let bytes = cols * ScalarType::BF16.size_in_bytes();
    let row_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, cols])
        .map_err(|e| anyhow::anyhow!("{label} alloc: {e}"))?;
    copy_d2d_batched(
        ordinal,
        row_buf.as_ptr() as *mut c_void,
        source.offset_ptr(row * bytes),
        bytes,
    )
    .map_err(|e| anyhow::anyhow!("{label} copy: {e}"))?;
    row_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("{label} D2H: {e}"))
}

#[allow(dead_code)]
pub fn prefill_tail_from_hidden_with_trace_position(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    hidden_bf16: &[u8],
    start_layer: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    _prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
    debug_full_layer: Option<usize>,
    debug_mlp_layer: Option<usize>,
    trace_position: Option<usize>,
) -> Result<PrefillResult> {
    let config = &weights.config;
    let hidden_dim = config.hidden_size;
    let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
    if row_bytes == 0 || hidden_bf16.is_empty() || hidden_bf16.len() % row_bytes != 0 {
        anyhow::bail!(
            "tail replay hidden bytes length {} is not a non-empty multiple of hidden row bytes {}",
            hidden_bf16.len(),
            row_bytes
        );
    }
    if start_layer >= config.num_hidden_layers {
        anyhow::bail!(
            "tail replay start_layer {} out of range (num_hidden_layers={})",
            start_layer,
            config.num_hidden_layers
        );
    }
    if debug_linear_layer.is_some() || debug_full_layer.is_some() || debug_mlp_layer.is_some() {
        anyhow::bail!(
            "tail replay debug stage traces are not available in the current prefill API"
        );
    }

    let seq_len = hidden_bf16.len() / row_bytes;
    let trace_row = trace_position.unwrap_or(seq_len - 1);
    if trace_row >= seq_len {
        anyhow::bail!(
            "tail replay trace position {} out of range for sequence length {}",
            trace_row,
            seq_len
        );
    }

    let mut scratch = PrefillScratch::new(config, seq_len, ordinal)?;
    copy_h2d(
        ordinal,
        scratch.hidden.as_ptr() as *mut c_void,
        hidden_bf16.as_ptr() as *const c_void,
        hidden_bf16.len(),
    )
    .map_err(|e| anyhow::anyhow!("tail replay hidden upload: {e}"))?;
    scratch.seed_f32_from_hidden(ordinal, seq_len * hidden_dim, "tail replay")?;

    let mut layer_attn_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers - start_layer))
    } else {
        None
    };
    let mut layer_post_attn_norm_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers - start_layer))
    } else {
        None
    };
    let mut layer_mlp_swiglu_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers - start_layer))
    } else {
        None
    };
    let mut layer_mlp_out_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers - start_layer))
    } else {
        None
    };
    let mut layer_hidden_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers - start_layer))
    } else {
        None
    };

    let kern = config.linear_conv_kernel_dim;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let mut chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
        .map(|i| {
            if config.is_full_attention(i) {
                Ok(None)
            } else {
                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("tail replay conv_tail alloc: {e}"))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    for idx in start_layer..config.num_hidden_layers {
        scratch.rms_norm_hidden_to_normed_model(
            config,
            ordinal,
            seq_len,
            hidden_dim,
            &weights.layers[idx].input_norm_w,
            &format!("tail replay layer {idx} input norm"),
        )?;

        if config.is_full_attention(idx) {
            prefill_full_attention_layer(
                weights,
                state,
                rotary,
                &mut scratch,
                config,
                idx,
                seq_len,
                0,
                ordinal,
                kv_chunk_size,
                /* commit_kv_filled */ true,
                None,
            )?;
        } else {
            let mut no_debug_trace = None;
            prefill_linear_attention_layer(
                weights,
                state,
                &mut scratch,
                config,
                idx,
                seq_len,
                0,
                true,
                chunk_conv_tail[idx].as_mut().unwrap(),
                ordinal,
                false,
                &mut no_debug_trace,
                None,
            )?;
        }

        if let Some(trace) = layer_attn_trace.as_mut() {
            trace.push(copy_bf16_row(
                ordinal,
                &scratch.hidden,
                trace_row,
                hidden_dim,
                &format!("tail replay attn trace layer {idx}"),
            )?);
        }

        scratch.rms_norm_hidden_to_normed_model(
            config,
            ordinal,
            seq_len,
            hidden_dim,
            &weights.layers[idx].post_attn_norm_w,
            &format!("tail replay layer {idx} post-attn norm"),
        )?;

        if let Some(trace) = layer_post_attn_norm_trace.as_mut() {
            trace.push(copy_bf16_row(
                ordinal,
                &scratch.normed,
                trace_row,
                hidden_dim,
                &format!("tail replay post-attn norm trace layer {idx}"),
            )?);
        }

        prefill_mlp_layer(weights, &mut scratch, config, idx, seq_len, ordinal)?;

        if let Some(trace) = layer_mlp_swiglu_trace.as_mut() {
            trace.push(copy_bf16_row(
                ordinal,
                &scratch.mlp_buf,
                trace_row,
                config.intermediate_size,
                &format!("tail replay mlp swiglu trace layer {idx}"),
            )?);
        }
        if let Some(trace) = layer_mlp_out_trace.as_mut() {
            trace.push(copy_bf16_row(
                ordinal,
                &scratch.proj_buf,
                trace_row,
                hidden_dim,
                &format!("tail replay mlp out trace layer {idx}"),
            )?);
        }
        if let Some(trace) = layer_hidden_trace.as_mut() {
            trace.push(copy_bf16_row(
                ordinal,
                &scratch.hidden,
                trace_row,
                hidden_dim,
                &format!("tail replay hidden trace layer {idx}"),
            )?);
        }
    }

    let (mut logits_per_pos, normed_last) = if let Some(hidden_f32) = scratch.hidden_f32.as_ref() {
        compute_logits_for_range_f32_hidden(
            hidden_f32,
            weights,
            config,
            seq_len - 1,
            1,
            use_4b_kernel,
            ordinal,
        )?
    } else {
        compute_logits_for_range(
            &scratch.hidden,
            weights,
            config,
            seq_len - 1,
            1,
            use_4b_kernel,
            ordinal,
        )?
    };
    let logits = logits_per_pos
        .pop()
        .expect("count=1 produces exactly one row");
    let final_norm_trace = Some(
        normed_last
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("tail replay final norm D2H: {e}"))?,
    );

    if kv_fp8 {
        convert_kv_caches_to_fp8(state, config, ordinal)?;
    }

    Ok(PrefillResult {
        logits,
        final_norm_trace,
        layer_attn_trace,
        layer_post_attn_norm_trace,
        layer_mlp_swiglu_trace,
        layer_mlp_out_trace,
        layer_hidden_trace,
        tap_hiddens: None,
        tap_hiddens_all: None,
        linear_debug_trace: None,
        target_nll: None,
    })
}

/// SpecPrefill cosine fast path: drafter prefill that only writes K/V
/// caches for layers `0..=last_layer` and then returns. Skips final
/// norm + lm_head + target_nll + BF16→FP8 KV conversion. Caller reads
/// the K cache from `state.layers[i]` for each scored layer `i`.
///
/// Required: `last_layer < num_hidden_layers`.
pub fn prefill_kv_through(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    last_layer: usize,
) -> Result<()> {
    if last_layer >= weights.config.num_hidden_layers {
        return Err(anyhow::anyhow!(
            "prefill_kv_through: last_layer {last_layer} out of range (num_hidden_layers={})",
            weights.config.num_hidden_layers
        ));
    }
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        false, // trace_layers
        None,  // debug_linear_layer
        None,  // tap_layers
        None,  // target_nll
        None,  // kept_positions
        Some(last_layer),
    )?;
    Ok(())
}

// ---- SpecPrefill Phase C: speculator-side prefill with attention export ----

/// Per-layer attention score tensor returned from
/// `prefill_with_lookahead_attention`: F32 with shape
/// `[q_heads, lookahead_count, kv_len]` flattened, where `kv_len ==
/// prompt_len` (the kernel attends to the prompt context only).
pub type LookaheadLayerScores = Vec<f32>;

pub struct PrefillWithLookaheadResult {
    /// The dense prefill's last-step logits + traces (currently unused
    /// downstream; preserved for symmetry with `prefill`).
    #[allow(dead_code)]
    pub base: PrefillResult,
    /// Per full-attention layer (in source-layer-index ascending order):
    /// F32 scores `[q_heads, lookahead_count, kv_len]` flattened. The
    /// number of vec entries equals the number of full-attention layers
    /// in the speculator.
    pub layer_scores: Vec<LookaheadLayerScores>,
    /// The number of query rows scored (passed-in `lookahead_count`; typically `paper_N + 1`).
    #[allow(dead_code)]
    pub lookahead_count: usize,
}

// ---- End SpecPrefill Phase C ----
// Implementation: see `DecodeEngine::prefill_with_lookahead_attention` in decode_engine.rs.

fn prefill_inner(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
    tap_layers: Option<&[usize]>,
    target_nll: Option<(usize, &[u32])>,
    kept_positions: Option<&[u32]>,
    // SpecPrefill cosine fast path: when Some(N), the per-chunk layer
    // loop stops after layer N (KV caches for layers 0..=N are written;
    // layers N+1.. are skipped). Final norm + lm_head + target_nll +
    // kv_fp8 conversion are all skipped — caller reads the KV caches
    // directly from `state`. The returned PrefillResult has empty
    // logits and None traces in this mode. Required: `tap_layers`,
    // `target_nll`, `debug_linear_layer`, `trace_layers`, and
    // `kept_positions` must all be unused (None / false) — early-exit
    // mode is K-only.
    last_layer: Option<usize>,
) -> Result<PrefillResult> {
    let config = &weights.config;
    let seq_len = if let Some(kept) = kept_positions {
        if kept.is_empty() {
            return Err(anyhow::anyhow!("prefill_inner: kept_positions is empty"));
        }
        let max_pos = *kept.iter().max().unwrap() as usize;
        if max_pos >= prompt_ids.len() {
            return Err(anyhow::anyhow!(
                "prefill_inner: kept_positions max {} out of range (prompt_len={})",
                max_pos,
                prompt_ids.len()
            ));
        }
        // Strict ascending uniqueness — the selection layer already
        // guarantees this; we re-check defensively so future callers
        // that hand-build kept lists fail loudly on bad input.
        for w in kept.windows(2) {
            if w[0] >= w[1] {
                return Err(anyhow::anyhow!(
                    "prefill_inner: kept_positions must be strictly ascending"
                ));
            }
        }
        kept.len()
    } else {
        prompt_ids.len()
    };
    let hidden_dim = config.hidden_size;

    // Determine effective chunk size: 0 = no chunking (full seq_len).
    // Minimum chunk size is conv kernel size (typically 4) to ensure
    // extract_conv_state has enough rows to read. We also ensure the
    // last chunk won't be smaller than kern by absorbing remaining tokens.
    let min_chunk = config.linear_conv_kernel_dim;
    let eff_chunk = if prefill_chunk_size == 0 || prefill_chunk_size >= seq_len {
        seq_len
    } else {
        prefill_chunk_size.max(min_chunk)
    };
    // Ensure the last chunk won't be too small: if remainder < min_chunk,
    // the last chunk absorbs into the previous one. E.g., 10 tokens with chunk=4:
    // remainder=2 < 4, so last chunk becomes 4+2=6 instead of 4,2.
    // This is handled in the loop by making the second-to-last chunk larger.

    // Allocate scratch buffers sized to max possible chunk (may absorb up to min_chunk-1 extra)
    let max_chunk = if eff_chunk < seq_len {
        eff_chunk + min_chunk - 1
    } else {
        seq_len
    };
    let mut scratch = PrefillScratch::new(config, max_chunk, ordinal)?;
    let mut layer_attn_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers))
    } else {
        None
    };
    let mut layer_post_attn_norm_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers))
    } else {
        None
    };
    let mut layer_mlp_swiglu_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers))
    } else {
        None
    };
    let mut layer_mlp_out_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers))
    } else {
        None
    };
    let mut layer_hidden_trace = if trace_layers {
        Some(Vec::with_capacity(config.num_hidden_layers))
    } else {
        None
    };
    let mut linear_debug_trace = None;

    // DFlash hidden-state taps: pre-allocate one slot per requested layer.
    // Validate indices up front so we fail loudly before doing prefill work.
    let mut tap_hiddens: Option<Vec<Vec<u8>>> = if let Some(tap) = tap_layers {
        for &li in tap {
            if li >= config.num_hidden_layers {
                return Err(anyhow::anyhow!(
                    "tap layer index {li} out of range (num_hidden_layers={})",
                    config.num_hidden_layers
                ));
            }
        }
        Some(vec![Vec::new(); tap.len()])
    } else {
        None
    };
    let mut tap_hiddens_all: Option<Vec<Vec<u8>>> =
        tap_layers.map(|tap| vec![Vec::with_capacity(seq_len * hidden_dim * 2); tap.len()]);

    // Per-layer inter-chunk state for linear attention layers. The F32 recurrent
    // state lives on the layer (`state.layers[idx].recurrent_state`) and is
    // updated in place at the end of every chunk — the BF16 sidecar that used
    // to live here was a precision bug (silently quantized the state at chunk
    // boundaries) and has been removed.
    let kern = config.linear_conv_kernel_dim;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;

    // Allocate inter-chunk conv tail buffers (last kern-1 QKV tokens per linear layer)
    let mut chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
        .map(|i| {
            if config.is_full_attention(i) {
                Ok(None)
            } else {
                // [qkv_dim, kern-1] BF16
                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("chunk conv_tail alloc: {e}"))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Process prompt in chunks
    let mut chunk_start = 0;
    let mut last_chunk_len = 0;
    while chunk_start < seq_len {
        let remaining = seq_len - chunk_start;
        // If the remaining tokens after this chunk would be too small (< kern),
        // absorb them into this chunk to avoid the small-chunk edge case.
        let chunk_len = if remaining > eff_chunk && remaining - eff_chunk < min_chunk {
            remaining // absorb the small remainder
        } else {
            std::cmp::min(eff_chunk, remaining)
        };
        let is_last_chunk = chunk_start + chunk_len >= seq_len;
        last_chunk_len = chunk_len;

        // Upload token IDs for this chunk. When `kept_positions` is set,
        // chunk_start/chunk_len index into the compacted kept sequence;
        // each compacted slot's actual token ID is prompt_ids[kept_positions[slot]].
        let chunk_ids_storage: Vec<u32>;
        let chunk_ids: &[u32] = if let Some(kept) = kept_positions {
            chunk_ids_storage = kept[chunk_start..chunk_start + chunk_len]
                .iter()
                .map(|&p| prompt_ids[p as usize])
                .collect();
            &chunk_ids_storage
        } else {
            &prompt_ids[chunk_start..chunk_start + chunk_len]
        };
        let id_bytes: Vec<u8> = chunk_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let token_ids_gpu =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[chunk_len], &id_bytes)
                .map_err(|e| anyhow::anyhow!("upload token IDs chunk: {e}"))?;

        // Embedding lookup: token IDs → hidden [chunk_len, hidden_dim]
        prefill_ffi::embedding_lookup(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            config.vocab_size,
            hidden_dim,
            &weights.embed_tokens,
            &token_ids_gpu,
            &mut scratch.hidden,
        )?;
        scratch.seed_f32_from_hidden(ordinal, chunk_len * hidden_dim, "prefill embedding")?;

        // Layer loop (all layers for this chunk, or 0..=last_layer when
        // the SpecPrefill cosine fast path requests early exit).
        let last_layer_idx = last_layer
            .map(|n| n.min(config.num_hidden_layers - 1))
            .unwrap_or(config.num_hidden_layers - 1);
        for idx in 0..=last_layer_idx {
            // Input RMSNorm
            scratch.rms_norm_hidden_to_normed_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &weights.layers[idx].input_norm_w,
                &format!("layer {idx} input norm"),
            )?;

            if config.is_full_attention(idx) {
                prefill_full_attention_layer(
                    weights,
                    state,
                    rotary,
                    &mut scratch,
                    config,
                    idx,
                    chunk_len,
                    chunk_start,
                    ordinal,
                    kv_chunk_size,
                    /* commit_kv_filled */ true,
                    kept_positions.map(|k| &k[chunk_start..chunk_start + chunk_len]), // NEW
                )?;
            } else {
                let mut no_debug_trace = None;
                let debug_trace_slot = if debug_linear_layer == Some(idx) && is_last_chunk {
                    &mut linear_debug_trace
                } else {
                    &mut no_debug_trace
                };
                let trace_linear_debug = debug_linear_layer == Some(idx) && is_last_chunk;
                prefill_linear_attention_layer(
                    weights,
                    state,
                    &mut scratch,
                    config,
                    idx,
                    chunk_len,
                    chunk_start,
                    is_last_chunk,
                    chunk_conv_tail[idx].as_mut().unwrap(),
                    ordinal,
                    trace_linear_debug,
                    debug_trace_slot,
                    None,
                )?;
            }

            if is_last_chunk {
                if let Some(trace) = layer_attn_trace.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * hidden_bytes;
                    let last_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                        .map_err(|e| {
                            anyhow::anyhow!("trace attn last_hidden alloc layer {idx}: {e}")
                        })?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_hidden.as_ptr() as *mut c_void,
                        scratch.hidden.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace attn last_hidden copy layer {idx}: {e}"))?;
                    trace.push(last_hidden.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("trace attn last_hidden D2H layer {idx}: {e}")
                    })?);
                }
            }

            // Post-attention RMSNorm
            scratch.rms_norm_hidden_to_normed_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &weights.layers[idx].post_attn_norm_w,
                &format!("layer {idx} post-attn norm"),
            )?;

            if is_last_chunk {
                if let Some(trace) = layer_post_attn_norm_trace.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * hidden_bytes;
                    let last_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                        .map_err(|e| {
                            anyhow::anyhow!("trace post-attn norm alloc layer {idx}: {e}")
                        })?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_normed.as_ptr() as *mut c_void,
                        scratch.normed.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace post-attn norm copy layer {idx}: {e}"))?;
                    trace.push(last_normed.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("trace post-attn norm D2H layer {idx}: {e}")
                    })?);
                }
            }

            // MLP
            prefill_mlp_layer(weights, &mut scratch, config, idx, chunk_len, ordinal)?;

            if is_last_chunk {
                if let Some(trace) = layer_mlp_swiglu_trace.as_mut() {
                    let swiglu_dim = config.intermediate_size;
                    let row_bytes = swiglu_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * row_bytes;
                    let last_swiglu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, swiglu_dim])
                        .map_err(|e| anyhow::anyhow!("trace mlp swiglu alloc layer {idx}: {e}"))?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_swiglu.as_ptr() as *mut c_void,
                        scratch.mlp_buf.offset_ptr(last_token_offset),
                        row_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace mlp swiglu copy layer {idx}: {e}"))?;
                    trace.push(
                        last_swiglu.to_host_bytes().map_err(|e| {
                            anyhow::anyhow!("trace mlp swiglu D2H layer {idx}: {e}")
                        })?,
                    );
                }
                if let Some(trace) = layer_mlp_out_trace.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * hidden_bytes;
                    let last_mlp = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("trace mlp out alloc layer {idx}: {e}"))?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_mlp.as_ptr() as *mut c_void,
                        scratch.proj_buf.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace mlp out copy layer {idx}: {e}"))?;
                    trace.push(
                        last_mlp
                            .to_host_bytes()
                            .map_err(|e| anyhow::anyhow!("trace mlp out D2H layer {idx}: {e}"))?,
                    );
                }
            }

            if is_last_chunk {
                if let Some(trace) = layer_hidden_trace.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * hidden_bytes;
                    let last_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("trace last_hidden alloc layer {idx}: {e}"))?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_hidden.as_ptr() as *mut c_void,
                        scratch.hidden.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace last_hidden copy layer {idx}: {e}"))?;
                    trace.push(
                        last_hidden.to_host_bytes().map_err(|e| {
                            anyhow::anyhow!("trace last_hidden D2H layer {idx}: {e}")
                        })?,
                    );
                }

                // DFlash tap: same data point as layer_hidden_trace (post-MLP residual,
                // last token of the final chunk) but captured selectively for the
                // requested tap layers only — avoids per-layer D2H cost when only a
                // few layers are needed.
                if let (Some(tap), Some(out)) = (tap_layers, tap_hiddens.as_mut()) {
                    for (slot, &target_layer) in tap.iter().enumerate().map(|(s, t)| (s, t)) {
                        if target_layer == idx {
                            let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                            let last_token_offset = (chunk_len - 1) * hidden_bytes;
                            let last_hidden =
                                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                                    .map_err(|e| {
                                        anyhow::anyhow!("dflash tap alloc layer {idx}: {e}")
                                    })?;
                            gpu_hal::copy_d2d(
                                ordinal,
                                last_hidden.as_ptr() as *mut c_void,
                                scratch.hidden.offset_ptr(last_token_offset),
                                hidden_bytes,
                            )
                            .map_err(|e| anyhow::anyhow!("dflash tap copy layer {idx}: {e}"))?;
                            out[slot] = last_hidden
                                .to_host_bytes()
                                .map_err(|e| anyhow::anyhow!("dflash tap D2H layer {idx}: {e}"))?;
                        }
                    }
                }
            }
            if let (Some(tap), Some(out_all)) = (tap_layers, tap_hiddens_all.as_mut()) {
                for (slot, &target_layer) in tap.iter().enumerate().map(|(s, t)| (s, t)) {
                    if target_layer == idx {
                        let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                        let chunk_bytes = chunk_len * hidden_bytes;
                        let host = scratch.hidden.to_host_bytes().map_err(|e| {
                            anyhow::anyhow!("dflash tap history D2H layer {idx}: {e}")
                        })?;
                        out_all[slot].extend_from_slice(&host[..chunk_bytes]);
                    }
                }
            }
        }

        chunk_start += chunk_len;
    }

    // SpecPrefill cosine fast path: caller asked for KV-only prefill
    // through `last_layer`. Skip final norm + lm_head + target_nll + the
    // BF16→FP8 KV conversion. Caller reads K caches directly from
    // `state` for the layers it scored.
    if last_layer.is_some() {
        return Ok(PrefillResult {
            logits: Vec::new(),
            final_norm_trace: None,
            layer_attn_trace,
            layer_post_attn_norm_trace,
            layer_mlp_swiglu_trace,
            layer_mlp_out_trace,
            layer_hidden_trace,
            tap_hiddens,
            tap_hiddens_all,
            linear_debug_trace,
            target_nll: None,
        });
    }

    // Extract logits for the last token of the final chunk. Refactored out
    // into `compute_logits_for_range` so the DFlash verify path can request
    // count=B and walk the block argmax in one shot (M3; see docs/dflash.md §6).
    let target_nll = if let Some((score_hidden_start, score_targets)) = target_nll {
        let score_end = score_hidden_start
            .checked_add(score_targets.len())
            .ok_or_else(|| anyhow::anyhow!("target NLL range overflow"))?;
        if score_end > last_chunk_len {
            return Err(anyhow::anyhow!(
                "target NLL range [{score_hidden_start}, {score_end}) is outside unchunked hidden buffer of {last_chunk_len} rows"
            ));
        }
        let nll = compute_target_nll_for_range(
            &scratch.hidden,
            weights,
            config,
            score_hidden_start,
            score_targets,
            ordinal,
        )?;
        Some(PrefillTargetNll {
            total_nll: nll.iter().map(|&x| x as f64).sum(),
            scored_tokens: nll.len(),
        })
    } else {
        None
    };

    let (mut logits_per_pos, normed_last) = if let Some(hidden_f32) = scratch.hidden_f32.as_ref() {
        compute_logits_for_range_f32_hidden(
            hidden_f32,
            weights,
            config,
            last_chunk_len - 1,
            1,
            use_4b_kernel,
            ordinal,
        )?
    } else {
        compute_logits_for_range(
            &scratch.hidden,
            weights,
            config,
            last_chunk_len - 1,
            1,
            use_4b_kernel,
            ordinal,
        )?
    };
    let logits = logits_per_pos
        .pop()
        .expect("count=1 produces exactly one row");
    let final_norm_trace = Some(
        normed_last
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("final norm D2H: {e}"))?,
    );

    // Post-prefill: convert BF16 KV caches to FP8 if requested.
    // During prefill we use BF16 KV so the attention kernel can read them directly.
    // Now convert to FP8 for subsequent decode steps.
    if kv_fp8 {
        convert_kv_caches_to_fp8(state, config, ordinal)?;
    }

    Ok(PrefillResult {
        logits,
        final_norm_trace,
        layer_attn_trace,
        layer_post_attn_norm_trace,
        layer_mlp_swiglu_trace,
        layer_mlp_out_trace,
        layer_hidden_trace,
        tap_hiddens,
        tap_hiddens_all,
        linear_debug_trace,
        target_nll,
    })
}

/// Convert all full-attention KV caches from BF16 to FP8 E4M3 in-place.
/// Allocates new FP8 cache + scale buffers, quantizes, replaces the BF16 caches.
pub fn convert_kv_caches_to_fp8(
    state: &mut ModelState,
    config: &TextConfig,
    ordinal: usize,
) -> Result<()> {
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    for (idx, ls) in state.layers.iter_mut().enumerate() {
        if !config.is_full_attention(idx) {
            continue;
        }
        let kv_len = ls.kv_filled;
        if kv_len == 0 {
            continue;
        }

        // Source: BF16 cache [1, nkv, cap, hd]. Preserve it as the exact BF16
        // sidecar used by KV-FP8 decode, and quantize from a contiguous view.
        let bf16_k = ls.kv_cache_k.take().unwrap();
        let bf16_v = ls.kv_cache_v.take().unwrap();
        let cap = bf16_k.shape()[2];

        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let k_contig =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("kv fp8 convert K contig layer {idx}: {e}"))?;
        let v_contig =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("kv fp8 convert V contig layer {idx}: {e}"))?;
        let cap_stride = cap * head_dim * elem_bytes;
        let contig_stride = kv_len * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                ordinal,
                k_contig.offset_ptr(h * contig_stride) as *mut std::ffi::c_void,
                bf16_k.offset_ptr(h * cap_stride),
                kv_len * head_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("kv fp8 convert K assemble h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                ordinal,
                v_contig.offset_ptr(h * contig_stride) as *mut std::ffi::c_void,
                bf16_v.offset_ptr(h * cap_stride),
                kv_len * head_dim * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("kv fp8 convert V assemble h={h}: {e}"))?;
        }

        // Allocate FP8 cache and scale buffers with same capacity
        let fp8_cap = cap; // keep same capacity for alignment
        let mut fp8_k = GpuBuffer::zeros(
            ordinal,
            ScalarType::U8,
            &[1, num_kv_heads, fp8_cap, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("fp8 K alloc layer {idx}: {e}"))?;
        let mut fp8_v = GpuBuffer::zeros(
            ordinal,
            ScalarType::U8,
            &[1, num_kv_heads, fp8_cap, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("fp8 V alloc layer {idx}: {e}"))?;
        let mut scale_k = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, fp8_cap])
            .map_err(|e| anyhow::anyhow!("scale K alloc layer {idx}: {e}"))?;
        let mut scale_v = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, fp8_cap])
            .map_err(|e| anyhow::anyhow!("scale V alloc layer {idx}: {e}"))?;

        // Quantize using GPU kernel
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            ScalarType::BF16,
            &k_contig,
            &mut fp8_k,
            &mut scale_k,
            num_kv_heads,
            kv_len,
            head_dim,
            fp8_cap,
            0,
        )
        .map_err(|e| anyhow::anyhow!("fp8 K quant layer {idx}: {e}"))?;

        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            ScalarType::BF16,
            &v_contig,
            &mut fp8_v,
            &mut scale_v,
            num_kv_heads,
            kv_len,
            head_dim,
            fp8_cap,
            0,
        )
        .map_err(|e| anyhow::anyhow!("fp8 V quant layer {idx}: {e}"))?;

        ls.kv_cache_k = Some(fp8_k);
        ls.kv_cache_v = Some(fp8_v);
        ls.kv_scale_k = Some(scale_k);
        ls.kv_scale_v = Some(scale_v);
        if kv_fp8_bf16_sidecar_enabled() {
            ls.kv_shadow_k = Some(bf16_k);
            ls.kv_shadow_v = Some(bf16_v);
            ls.kv_shadow_start = kv_fp8_bf16_sidecar_window_tokens()
                .map(|window| kv_len.saturating_sub(window))
                .unwrap_or(0);
        } else {
            ls.kv_shadow_k = None;
            ls.kv_shadow_v = None;
            ls.kv_shadow_start = usize::MAX;
        }
    }
    Ok(())
}

/// Replay the full token history through the validated GPU prefill path and
/// return last-token logits. Slower than incremental decode, but much closer to
/// the native path than the experimental component decode oracle.
pub fn gpu_reference_replay_step(
    weights: &Qwen35Weights,
    rotary: &RotaryTables,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<Vec<f32>> {
    let mut replay_state = ModelState::new(&weights.config, ordinal)
        .map_err(|e| anyhow::anyhow!("gpu replay state init: {e}"))?;
    let result = prefill(
        weights,
        &mut replay_state,
        rotary,
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
    )?;
    Ok(result.logits)
}

/// DFlash variant of `gpu_reference_replay_step` that additionally returns the
/// post-MLP residual hidden state at the LAST token of the input sequence for
/// each layer in `tap_layers`. The taps are 1:1 with `tap_layers` (BF16 bytes,
/// length `hidden_dim` each). Used by the DFlash speculative decoder to feed
/// fused multi-layer target context into the small bidirectional draft model.
#[allow(dead_code)]
pub fn gpu_reference_replay_step_with_taps(
    weights: &Qwen35Weights,
    rotary: &RotaryTables,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: &[usize],
) -> Result<(Vec<f32>, Vec<Vec<u8>>)> {
    let mut replay_state = ModelState::new(&weights.config, ordinal)
        .map_err(|e| anyhow::anyhow!("gpu replay state init: {e}"))?;
    let result = prefill_with_taps(
        weights,
        &mut replay_state,
        rotary,
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
        tap_layers,
    )?;
    let taps = result.tap_hiddens.ok_or_else(|| {
        anyhow::anyhow!("internal: tap_hiddens missing despite tap_layers being supplied")
    })?;
    Ok((result.logits, taps))
}

/// Append a contiguous DFlash verify block to the live target state using the
/// prefill component kernels, returning logits for every appended position.
///
/// Unlike `prefill_with_taps`, this does not assume position zero. The caller
/// supplies the absolute `pos_offset`; full-attention KV is written at
/// `[pos_offset, pos_offset + token_ids.len())` without advancing `kv_filled`,
/// while linear-attention state is mutated in place. The DFlash driver must
/// snapshot/restore linear state around this call, just as it does for the
/// persistent fused verifier.
pub fn prefill_append_logits(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<Vec<Vec<f32>>> {
    Ok(prefill_append_verify(
        weights,
        state,
        rotary,
        token_ids,
        pos_offset,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        None,
        false,
        false,
        None,
    )?
    .logits)
}

pub fn prefill_append_verify(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    capture_rollback: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
) -> Result<PrefillAppendVerifyResult> {
    prefill_append_verify_impl(
        weights,
        state,
        rotary,
        token_ids,
        pos_offset,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        tap_layers,
        capture_rollback,
        greedy_only,
        greedy_compare_tokens,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_append_verify_cached(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    capture_rollback: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: &mut PrefillAppendVerifyCache,
) -> Result<PrefillAppendVerifyResult> {
    prefill_append_verify_impl(
        weights,
        state,
        rotary,
        token_ids,
        pos_offset,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        tap_layers,
        capture_rollback,
        greedy_only,
        greedy_compare_tokens,
        Some(cache),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_append_verify_cached_with_gpu_taps(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    capture_rollback: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: &mut PrefillAppendVerifyCache,
    gpu_tap_sink: Option<&mut PrefillAppendGpuTapSink<'_>>,
) -> Result<PrefillAppendVerifyResult> {
    prefill_append_verify_impl(
        weights,
        state,
        rotary,
        token_ids,
        pos_offset,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        tap_layers,
        capture_rollback,
        greedy_only,
        greedy_compare_tokens,
        Some(cache),
        gpu_tap_sink,
    )
}

#[allow(clippy::too_many_arguments)]
fn prefill_append_verify_impl(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    capture_rollback: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: Option<&mut PrefillAppendVerifyCache>,
    mut gpu_tap_sink: Option<&mut PrefillAppendGpuTapSink<'_>>,
) -> Result<PrefillAppendVerifyResult> {
    if token_ids.is_empty() {
        return Err(anyhow::anyhow!("prefill_append_logits: token_ids is empty"));
    }

    let config = &weights.config;
    let chunk_len = token_ids.len();
    let hidden_dim = config.hidden_size;
    let profile = std::env::var_os("SUPERSONIC_DFLASH_PROFILE_APPEND").is_some();
    let mut ms_seed = 0.0_f64;
    let mut ms_embed = 0.0_f64;
    let mut ms_input_norm = 0.0_f64;
    let mut ms_full_attn = 0.0_f64;
    let mut ms_linear_attn = 0.0_f64;
    let mut ms_post_norm = 0.0_f64;
    let mut ms_mlp = 0.0_f64;
    let mut ms_logits = 0.0_f64;
    let kern = config.linear_conv_kernel_dim;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
    let conv_tail_bytes = qkv_dim * (kern - 1) * ScalarType::BF16.size_in_bytes();

    let mut local_cache;
    let cache = match cache {
        Some(cache) => cache,
        None => {
            local_cache = PrefillAppendVerifyCache::new(config, chunk_len, ordinal)?;
            &mut local_cache
        }
    };
    if !cache.matches(chunk_len, ordinal) {
        *cache = PrefillAppendVerifyCache::new(config, chunk_len, ordinal)?;
    }
    let mut rollback: Option<PrefillAppendRollback> = if capture_rollback {
        Some(cache.take_rollback(config, pos_offset, ordinal)?)
    } else {
        None
    };
    let scratch = &mut cache.scratch;
    let chunk_conv_tail = &mut cache.chunk_conv_tail;
    let token_ids_gpu = &mut cache.token_ids_gpu;

    let tap_count = tap_layers.map(|tap| tap.len()).unwrap_or(0);
    let mut tap_hiddens_all: Option<Vec<Vec<u8>>> =
        if tap_layers.is_some() && gpu_tap_sink.is_none() {
            Some(vec![
                Vec::with_capacity(
                    chunk_len * hidden_dim * ScalarType::BF16.size_in_bytes()
                );
                tap_count
            ])
        } else {
            None
        };

    for idx in 0..config.num_hidden_layers {
        if config.is_full_attention(idx) {
            continue;
        }
        let t_seed = std::time::Instant::now();
        if let (Some(conv_state), Some(chunk_tail)) = (
            state.layers[idx].conv_state.as_ref(),
            chunk_conv_tail[idx].as_mut(),
        ) {
            copy_d2d_batched(
                ordinal,
                chunk_tail.as_ptr() as *mut c_void,
                conv_state.as_ptr(),
                conv_tail_bytes,
            )
            .map_err(|e| anyhow::anyhow!("append seed conv tail layer {idx}: {e}"))?;
        }
        if profile {
            ms_seed += t_seed.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let t_embed = std::time::Instant::now();
    let id_bytes: Vec<u8> = token_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    copy_h2d(
        ordinal,
        token_ids_gpu.as_mut_ptr(),
        id_bytes.as_ptr() as *const c_void,
        id_bytes.len(),
    )
    .map_err(|e| anyhow::anyhow!("append upload token IDs: {e}"))?;
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        config.vocab_size,
        hidden_dim,
        &weights.embed_tokens,
        token_ids_gpu,
        &mut scratch.hidden,
    )
    .map_err(|e| anyhow::anyhow!("append embedding lookup: {e}"))?;
    scratch.seed_f32_from_hidden(ordinal, chunk_len * hidden_dim, "append embedding")?;
    if profile {
        ms_embed += t_embed.elapsed().as_secs_f64() * 1000.0;
    }

    for idx in 0..config.num_hidden_layers {
        let t_input_norm = std::time::Instant::now();
        if scratch.has_f32_activation_carry() {
            scratch.rms_norm_hidden_to_normed_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &weights.layers[idx].input_norm_w,
                "append input norm",
            )?;
        } else {
            rms_norm_rows_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &scratch.hidden,
                &weights.layers[idx].input_norm_w,
                &mut scratch.normed,
                "append input norm",
            )?;
        }
        if profile {
            ms_input_norm += t_input_norm.elapsed().as_secs_f64() * 1000.0;
        }

        if config.is_full_attention(idx) {
            let t_full = std::time::Instant::now();
            prefill_full_attention_layer(
                weights,
                state,
                rotary,
                scratch,
                config,
                idx,
                chunk_len,
                pos_offset,
                ordinal,
                kv_chunk_size,
                /* commit_kv_filled */ false,
                None,
            )?;
            if profile {
                ms_full_attn += t_full.elapsed().as_secs_f64() * 1000.0;
            }
        } else {
            let t_linear = std::time::Instant::now();
            let mut no_debug_trace = None;
            let chunk_tail = chunk_conv_tail[idx]
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("append missing conv tail for layer {idx}"))?;
            let append_capture = rollback.as_mut().and_then(|r| r.per_layer[idx].as_mut());
            prefill_linear_attention_layer(
                weights,
                state,
                scratch,
                config,
                idx,
                chunk_len,
                pos_offset,
                /* is_last_chunk */ true,
                chunk_tail,
                ordinal,
                false,
                &mut no_debug_trace,
                append_capture,
            )?;
            if profile {
                ms_linear_attn += t_linear.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let t_post_norm = std::time::Instant::now();
        if scratch.has_f32_activation_carry() {
            scratch.rms_norm_hidden_to_normed_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &weights.layers[idx].post_attn_norm_w,
                "append post-attn norm",
            )?;
        } else {
            rms_norm_rows_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &scratch.hidden,
                &weights.layers[idx].post_attn_norm_w,
                &mut scratch.normed,
                "append post-attn norm",
            )?;
        }
        if profile {
            ms_post_norm += t_post_norm.elapsed().as_secs_f64() * 1000.0;
        }

        let t_mlp = std::time::Instant::now();
        prefill_mlp_layer(weights, scratch, config, idx, chunk_len, ordinal)?;
        if profile {
            ms_mlp += t_mlp.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(tap) = tap_layers {
            for (slot, &target_layer) in tap.iter().enumerate().map(|(s, t)| (s, t)) {
                if target_layer != idx {
                    continue;
                }
                if let Some(sink) = gpu_tap_sink.as_deref_mut() {
                    copy_tap_rows_to_gpu_history(
                        ordinal,
                        sink,
                        slot,
                        tap.len(),
                        &scratch.hidden,
                        chunk_len,
                        hidden_dim,
                    )
                    .map_err(|e| anyhow::anyhow!("append GPU tap history copy layer {idx}: {e}"))?;
                } else if let Some(out_all) = tap_hiddens_all.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let chunk_bytes = chunk_len * hidden_bytes;
                    let host = scratch.hidden.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("append dflash tap history D2H layer {idx}: {e}")
                    })?;
                    out_all[slot].extend_from_slice(&host[..chunk_bytes]);
                }
            }
        }
    }

    let t_logits = std::time::Instant::now();
    let (logits_per_pos, target_next) = if greedy_only {
        let ids = if let Some(compare_tokens) = greedy_compare_tokens {
            compute_greedy_for_acceptance(
                &scratch.hidden,
                weights,
                config,
                chunk_len,
                compare_tokens,
                use_4b_kernel,
                ordinal,
            )?
        } else {
            let (ids, _normed) = compute_greedy_for_range(
                &scratch.hidden,
                weights,
                config,
                0,
                chunk_len,
                use_4b_kernel,
                ordinal,
            )?;
            ids
        };
        (Vec::new(), Some(ids))
    } else {
        let (logits, _normed) = if let Some(hidden_f32) = scratch.hidden_f32.as_ref() {
            compute_logits_for_range_f32_hidden(
                hidden_f32,
                weights,
                config,
                0,
                chunk_len,
                use_4b_kernel,
                ordinal,
            )?
        } else {
            compute_logits_for_range(
                &scratch.hidden,
                weights,
                config,
                0,
                chunk_len,
                use_4b_kernel,
                ordinal,
            )?
        };
        (logits, None)
    };
    if profile {
        ms_logits += t_logits.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dflash-profile] prefill_append B={} pos={} seed={:.2}ms embed={:.2}ms input_norm={:.2}ms full_attn={:.2}ms linear_attn={:.2}ms post_norm={:.2}ms mlp={:.2}ms logits={:.2}ms",
            chunk_len,
            pos_offset,
            ms_seed,
            ms_embed,
            ms_input_norm,
            ms_full_attn,
            ms_linear_attn,
            ms_post_norm,
            ms_mlp,
            ms_logits,
        );
    }
    Ok(PrefillAppendVerifyResult {
        logits: logits_per_pos,
        target_next,
        tap_hiddens_all,
        rollback,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_tree_verify(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    positions: &[usize],
    parent_ids: &[i32],
    visibility: &[u8],
    prefix_len: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    greedy_only: bool,
    capture_rollback: bool,
) -> Result<PrefillTreeVerifyResult> {
    prefill_tree_verify_impl(
        weights,
        state,
        rotary,
        token_ids,
        positions,
        parent_ids,
        visibility,
        prefix_len,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        tap_layers,
        greedy_only,
        capture_rollback,
        false,
        None,
    )
}

fn tree_conv_source_col(
    parent_ids: &[i32],
    t: usize,
    tap: usize,
    kernel_size: usize,
) -> Result<usize> {
    let state_len = kernel_size.saturating_sub(1);
    let steps = state_len.saturating_sub(tap);
    if steps == 0 {
        return Ok(state_len + t);
    }

    let mut node = t;
    let mut walked = 0usize;
    while walked < steps {
        let parent = *parent_ids
            .get(node)
            .ok_or_else(|| anyhow::anyhow!("tree conv source node {node} out of range"))?;
        if parent < 0 {
            break;
        }
        let parent = usize::try_from(parent)
            .map_err(|_| anyhow::anyhow!("tree conv source invalid parent id {parent}"))?;
        if parent >= parent_ids.len() {
            return Err(anyhow::anyhow!(
                "tree conv source parent id {parent} out of range {}",
                parent_ids.len()
            ));
        }
        node = parent;
        walked += 1;
    }

    if walked == steps {
        Ok(state_len + node)
    } else {
        Ok(tap + walked)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_tree_verify_cached(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    positions: &[usize],
    parent_ids: &[i32],
    visibility: &[u8],
    prefix_len: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    greedy_only: bool,
    capture_rollback: bool,
    capture_gpu_taps: bool,
    cache: &mut PrefillTreeVerifyCache,
) -> Result<PrefillTreeVerifyResult> {
    prefill_tree_verify_impl(
        weights,
        state,
        rotary,
        token_ids,
        positions,
        parent_ids,
        visibility,
        prefix_len,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        tap_layers,
        greedy_only,
        capture_rollback,
        capture_gpu_taps,
        Some(cache),
    )
}

#[allow(clippy::too_many_arguments)]
fn prefill_tree_verify_impl(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    positions: &[usize],
    parent_ids: &[i32],
    visibility: &[u8],
    prefix_len: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    tap_layers: Option<&[usize]>,
    greedy_only: bool,
    capture_rollback: bool,
    capture_gpu_taps: bool,
    cache: Option<&mut PrefillTreeVerifyCache>,
) -> Result<PrefillTreeVerifyResult> {
    if token_ids.is_empty() {
        return Err(anyhow::anyhow!("prefill_tree_verify: token_ids is empty"));
    }
    let tree_len = token_ids.len();
    if positions.len() != tree_len {
        return Err(anyhow::anyhow!(
            "prefill_tree_verify: positions len {} != tree_len {tree_len}",
            positions.len()
        ));
    }
    if parent_ids.len() != tree_len {
        return Err(anyhow::anyhow!(
            "prefill_tree_verify: parent_ids len {} != tree_len {tree_len}",
            parent_ids.len()
        ));
    }
    if visibility.len() != tree_len * tree_len {
        return Err(anyhow::anyhow!(
            "prefill_tree_verify: visibility len {} != tree_len^2 {}",
            visibility.len(),
            tree_len * tree_len
        ));
    }

    let config = &weights.config;
    let hidden_dim = config.hidden_size;
    let profile = std::env::var_os("SUPERSONIC_DFLASH_PROFILE_VERIFY").is_some();
    let mut ms_setup = 0.0_f64;
    let mut ms_embed = 0.0_f64;
    let mut ms_input_norm = 0.0_f64;
    let mut ms_full_attn = 0.0_f64;
    let mut ms_linear_attn = 0.0_f64;
    let mut ms_post_norm = 0.0_f64;
    let mut ms_mlp = 0.0_f64;
    let mut ms_taps = 0.0_f64;
    let mut ms_logits = 0.0_f64;

    let t_setup = std::time::Instant::now();
    let mut local_cache;
    let cache = match cache {
        Some(cache) => cache,
        None => {
            local_cache = PrefillTreeVerifyCache::new(config, tree_len, ordinal)?;
            &mut local_cache
        }
    };
    if !cache.matches(tree_len, ordinal) {
        *cache = PrefillTreeVerifyCache::new(config, tree_len, ordinal)?;
    }
    cache.upload_inputs(token_ids, positions, parent_ids, visibility)?;
    if profile {
        ms_setup += t_setup.elapsed().as_secs_f64() * 1000.0;
    }

    let tap_count = tap_layers.map(|tap| tap.len()).unwrap_or(0);
    let use_gpu_tap_capture = capture_gpu_taps && tap_count > 0;
    if use_gpu_tap_capture {
        cache
            .scratch
            .ensure_tree_tap_capture(ordinal, tree_len, tap_count, hidden_dim)?;
    }

    let mut rollback = if capture_rollback {
        Some(cache.take_rollback(config, prefix_len)?)
    } else {
        None
    };
    let scratch = &mut cache.scratch;
    let mut tap_hiddens_all: Option<Vec<Vec<u8>>> = if use_gpu_tap_capture {
        None
    } else {
        tap_layers.map(|tap| vec![Vec::with_capacity(tree_len * hidden_dim * 2); tap.len()])
    };

    let t_embed = std::time::Instant::now();
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        tree_len,
        config.vocab_size,
        hidden_dim,
        &weights.embed_tokens,
        &cache.token_ids_gpu,
        &mut scratch.hidden,
    )
    .map_err(|e| anyhow::anyhow!("tree verify embedding lookup: {e}"))?;
    if profile {
        ms_embed += t_embed.elapsed().as_secs_f64() * 1000.0;
    }

    for idx in 0..config.num_hidden_layers {
        let t_input_norm = std::time::Instant::now();
        rms_norm_rows_model(
            config,
            ordinal,
            tree_len,
            hidden_dim,
            &scratch.hidden,
            &weights.layers[idx].input_norm_w,
            &mut scratch.normed,
            &format!("tree layer {idx} input norm"),
        )?;
        if profile {
            ms_input_norm += t_input_norm.elapsed().as_secs_f64() * 1000.0;
        }

        if config.is_full_attention(idx) {
            let capture_slot = rollback.as_mut().map(|r| &mut r.per_layer[idx]);
            let t_attn = std::time::Instant::now();
            prefill_tree_full_attention_layer(
                weights,
                state,
                rotary,
                &mut *scratch,
                config,
                idx,
                tree_len,
                prefix_len,
                ordinal,
                kv_chunk_size,
                &cache.positions_gpu,
                &cache.visibility_gpu,
                capture_slot,
            )?;
            if profile {
                ms_full_attn += t_attn.elapsed().as_secs_f64() * 1000.0;
            }
        } else {
            let capture_slot = rollback.as_mut().map(|r| &mut r.per_layer[idx]);
            let t_attn = std::time::Instant::now();
            prefill_tree_linear_attention_layer(
                weights,
                state,
                &mut *scratch,
                config,
                idx,
                tree_len,
                prefix_len,
                ordinal,
                &cache.parent_ids_gpu,
                &cache.conv_source_cols_gpu,
                cache.conv_source_cols_stride,
                capture_slot,
            )?;
            if profile {
                ms_linear_attn += t_attn.elapsed().as_secs_f64() * 1000.0;
            }
        }

        let t_post_norm = std::time::Instant::now();
        rms_norm_rows_model(
            config,
            ordinal,
            tree_len,
            hidden_dim,
            &scratch.hidden,
            &weights.layers[idx].post_attn_norm_w,
            &mut scratch.normed,
            &format!("tree layer {idx} post-attn norm"),
        )?;
        if profile {
            ms_post_norm += t_post_norm.elapsed().as_secs_f64() * 1000.0;
        }

        let t_mlp = std::time::Instant::now();
        prefill_mlp_layer(weights, &mut *scratch, config, idx, tree_len, ordinal)?;
        if profile {
            ms_mlp += t_mlp.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(tap) = tap_layers {
            let t_taps = std::time::Instant::now();
            for (slot, &target_layer) in tap.iter().enumerate() {
                if target_layer == idx {
                    if use_gpu_tap_capture {
                        scratch.copy_hidden_to_tree_tap_capture(
                            ordinal, slot, tap_count, tree_len, hidden_dim,
                        )?;
                    } else if let Some(out_all) = tap_hiddens_all.as_mut() {
                        let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                        let tree_bytes = tree_len * hidden_bytes;
                        let host = scratch.hidden.to_host_bytes().map_err(|e| {
                            anyhow::anyhow!("tree verify dflash tap history D2H layer {idx}: {e}")
                        })?;
                        out_all[slot].extend_from_slice(&host[..tree_bytes]);
                    }
                }
            }
            if profile {
                ms_taps += t_taps.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    let t_logits = std::time::Instant::now();
    let target_next = if greedy_only {
        cache.compute_greedy_ids(weights, config, use_4b_kernel)?
    } else {
        let (logits, normed) = compute_logits_for_range(
            &cache.scratch.hidden,
            weights,
            config,
            0,
            tree_len,
            use_4b_kernel,
            ordinal,
        )?;
        let ids = logits
            .iter()
            .map(|row| {
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &val) in row.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_idx = idx as u32;
                    }
                }
                best_idx
            })
            .collect();
        let _ = normed;
        ids
    };
    if profile {
        ms_logits += t_logits.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dflash-profile] tree_verify len={} prefix={} setup/upload={:.2}ms embed={:.2}ms input_norm={:.2}ms full_attn={:.2}ms linear_attn={:.2}ms post_norm={:.2}ms mlp={:.2}ms taps={:.2}ms logits/greedy={:.2}ms",
            tree_len,
            prefix_len,
            ms_setup,
            ms_embed,
            ms_input_norm,
            ms_full_attn,
            ms_linear_attn,
            ms_post_norm,
            ms_mlp,
            ms_taps,
            ms_logits,
        );
    }

    Ok(PrefillTreeVerifyResult {
        target_next,
        tap_hiddens_all,
        tap_hiddens_gpu: use_gpu_tap_capture,
        rollback,
    })
}

pub fn apply_prefill_append_rollback(
    state: &mut ModelState,
    config: &TextConfig,
    result: &PrefillAppendVerifyResult,
    commit_len: usize,
    ordinal: usize,
) -> Result<()> {
    let rollback = result
        .rollback
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prefill append result did not capture rollback state"))?;
    if commit_len == 0 || commit_len > rollback.chunk_len {
        return Err(anyhow::anyhow!(
            "prefill append rollback commit_len {commit_len} outside 1..={}",
            rollback.chunk_len
        ));
    }
    if rollback.per_layer.len() != state.layers.len() {
        return Err(anyhow::anyhow!(
            "prefill append rollback layer count {} != state layer count {}",
            rollback.per_layer.len(),
            state.layers.len()
        ));
    }

    let pad = config.linear_conv_kernel_dim - 1;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let needs_q8_trace_sync = rollback.per_layer.iter().any(|layer| {
        layer
            .as_ref()
            .is_some_and(|layer_rb| layer_rb.recurrent_trace.dtype() == ScalarType::U8)
    });
    if needs_q8_trace_sync {
        gpu_hal::sync(ordinal).map_err(|e| anyhow::anyhow!("dflash Q8 trace sync: {e}"))?;
    }

    for idx in 0..state.layers.len() {
        if config.is_full_attention(idx) {
            state.layers[idx].set_kv_filled(rollback.pos_offset + commit_len);
            continue;
        }

        let layer_rb = rollback.per_layer[idx]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill append rollback missing linear layer {idx}"))?;
        let layer_state = &mut state.layers[idx];
        let conv_dst = layer_state.conv_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!("prefill append rollback layer {idx} missing conv_state")
        })?;
        let rec_dst = layer_state.recurrent_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!("prefill append rollback layer {idx} missing recurrent_state")
        })?;

        if layer_rb.recurrent_trace.dtype() == ScalarType::U8 {
            prefill_ffi::dflash_apply_rollback_q8_trace(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.chunk_len,
                rollback.chunk_len,
                commit_len,
                nv,
                khd,
                vhd,
                &layer_rb.conv_input,
                conv_dst,
                &layer_rb.recurrent_trace,
                rec_dst,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} dflash Q8-trace rollback apply: {e}"))?;
        } else if layer_rb.recurrent_trace.dtype() == ScalarType::BF16 {
            prefill_ffi::dflash_apply_rollback_bf16_trace(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.chunk_len,
                rollback.chunk_len,
                commit_len,
                nv,
                khd,
                vhd,
                &layer_rb.conv_input,
                conv_dst,
                &layer_rb.recurrent_trace,
                rec_dst,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} dflash BF16-trace rollback apply: {e}"))?;
        } else {
            prefill_ffi::dflash_apply_rollback(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.chunk_len,
                rollback.chunk_len,
                commit_len,
                nv,
                khd,
                vhd,
                &layer_rb.conv_input,
                conv_dst,
                &layer_rb.recurrent_trace,
                rec_dst,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} dflash rollback apply: {e}"))?;
        }
    }

    Ok(())
}

pub fn apply_prefill_tree_rollback(
    state: &mut ModelState,
    config: &TextConfig,
    result: &PrefillTreeVerifyResult,
    accepted_indices: &[usize],
    commit_len: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<()> {
    let rollback = result
        .rollback
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prefill tree result did not capture rollback state"))?;
    if commit_len == 0 || commit_len > rollback.tree_len || commit_len > accepted_indices.len() {
        return Err(anyhow::anyhow!(
            "prefill tree rollback commit_len {commit_len} outside accepted/tree bounds accepted={} tree={}",
            accepted_indices.len(),
            rollback.tree_len
        ));
    }
    if rollback.per_layer.len() != state.layers.len() {
        return Err(anyhow::anyhow!(
            "prefill tree rollback layer count {} != state layer count {}",
            rollback.per_layer.len(),
            state.layers.len()
        ));
    }
    for &idx in accepted_indices.iter().take(commit_len) {
        if idx >= rollback.tree_len {
            return Err(anyhow::anyhow!(
                "prefill tree rollback accepted index {idx} >= tree_len {}",
                rollback.tree_len
            ));
        }
    }

    let accepted_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::U32,
        &[commit_len],
        &encode_u32_le(&accepted_indices[..commit_len]),
    )
    .map_err(|e| anyhow::anyhow!("prefill tree rollback upload accepted indices: {e}"))?;

    let pad = config.linear_conv_kernel_dim - 1;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let needs_q8_trace_sync = rollback.per_layer.iter().any(|layer| {
        layer.as_ref().is_some_and(|layer_rb| match layer_rb {
            PrefillTreeLayerRollback::Linear {
                recurrent_trace, ..
            } => recurrent_trace.dtype() == ScalarType::U8,
            PrefillTreeLayerRollback::Full { .. } => false,
        })
    });
    if needs_q8_trace_sync {
        gpu_hal::sync(ordinal).map_err(|e| anyhow::anyhow!("dflash tree Q8 trace sync: {e}"))?;
    }
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    for idx in 0..state.layers.len() {
        let layer_rb = rollback.per_layer[idx]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill tree rollback missing layer {idx}"))?;

        if config.is_full_attention(idx) {
            let (tree_k, tree_v) = match layer_rb {
                PrefillTreeLayerRollback::Full { tree_k, tree_v } => (tree_k, tree_v),
                PrefillTreeLayerRollback::Linear { .. } => {
                    return Err(anyhow::anyhow!(
                        "prefill tree rollback layer {idx} expected full-attention material"
                    ));
                }
            };
            let layer_state = &mut state.layers[idx];
            let final_pos = rollback.prefix_len + commit_len - 1;
            layer_state
                .ensure_kv_capacity(final_pos, ordinal, config, kv_chunk_size, false)
                .map_err(|e| anyhow::anyhow!("tree rollback ensure KV layer {idx}: {e}"))?;
            let cap = layer_state.kv_capacity();
            let num_kv_heads = config.num_key_value_heads;
            let head_dim = config.head_dim;
            let tree_stride = rollback.tree_len * head_dim * elem_bytes;
            let cap_stride = cap * head_dim * elem_bytes;
            let row_bytes = head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                for (commit_pos, &tree_idx) in accepted_indices.iter().take(commit_len).enumerate()
                {
                    let dst_pos = rollback.prefix_len + commit_pos;
                    let src_off = h * tree_stride + tree_idx * row_bytes;
                    let dst_off = h * cap_stride + dst_pos * row_bytes;
                    let dst_k = layer_state
                        .kv_cache_k_offset_ptr(dst_off)
                        .ok_or_else(|| anyhow::anyhow!("tree rollback layer {idx} missing K"))?;
                    let dst_v = layer_state
                        .kv_cache_v_offset_ptr(dst_off)
                        .ok_or_else(|| anyhow::anyhow!("tree rollback layer {idx} missing V"))?;
                    copy_d2d_batched(
                        ordinal,
                        dst_k as *mut c_void,
                        tree_k.offset_ptr(src_off),
                        row_bytes,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("tree rollback layer {idx} K h={h} pos={commit_pos}: {e}")
                    })?;
                    copy_d2d_batched(
                        ordinal,
                        dst_v as *mut c_void,
                        tree_v.offset_ptr(src_off),
                        row_bytes,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("tree rollback layer {idx} V h={h} pos={commit_pos}: {e}")
                    })?;
                }
            }
            layer_state.set_kv_filled(rollback.prefix_len + commit_len);
            continue;
        }

        let (conv_input, recurrent_trace) = match layer_rb {
            PrefillTreeLayerRollback::Linear {
                conv_input,
                recurrent_trace,
            } => (conv_input, recurrent_trace),
            PrefillTreeLayerRollback::Full { .. } => {
                return Err(anyhow::anyhow!(
                    "prefill tree rollback layer {idx} expected linear material"
                ));
            }
        };
        let layer_state = &mut state.layers[idx];
        let conv_dst = layer_state.conv_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!("prefill tree rollback layer {idx} missing conv_state")
        })?;
        let rec_dst = layer_state.recurrent_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!("prefill tree rollback layer {idx} missing recurrent_state")
        })?;

        if recurrent_trace.dtype() == ScalarType::U8 {
            prefill_ffi::dflash_apply_tree_rollback_q8_trace(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.tree_len,
                rollback.tree_len,
                commit_len,
                nv,
                khd,
                vhd,
                conv_input,
                &accepted_gpu,
                conv_dst,
                recurrent_trace,
                rec_dst,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} dflash tree Q8-trace rollback apply: {e}"))?;
        } else if recurrent_trace.dtype() == ScalarType::BF16 {
            prefill_ffi::dflash_apply_tree_rollback_bf16_trace(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.tree_len,
                rollback.tree_len,
                commit_len,
                nv,
                khd,
                vhd,
                conv_input,
                &accepted_gpu,
                conv_dst,
                recurrent_trace,
                rec_dst,
            )
            .map_err(|e| {
                anyhow::anyhow!("layer {idx} dflash tree BF16-trace rollback apply: {e}")
            })?;
        } else {
            prefill_ffi::dflash_apply_tree_rollback(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + rollback.tree_len,
                rollback.tree_len,
                commit_len,
                nv,
                khd,
                vhd,
                conv_input,
                &accepted_gpu,
                conv_dst,
                recurrent_trace,
                rec_dst,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} dflash tree rollback apply: {e}"))?;
        }
    }

    Ok(())
}

/// Reusable scratch for Metal v2 incremental decode. Sized for chunk_len=1 and
/// allocated once on the engine; carries the BF16 inter-chunk linear-attention
/// buffers across decode steps so we don't re-zero them each call.
pub struct MetalV2DecodeScratch {
    scratch: PrefillScratch,
    chunk_conv_tail: Vec<Option<GpuBuffer>>,
    token_id_buf: GpuBuffer,
}

impl MetalV2DecodeScratch {
    pub fn new(config: &TextConfig, ordinal: usize) -> Result<Self> {
        let scratch = PrefillScratch::new(config, 1, ordinal)?;
        let kern = config.linear_conv_kernel_dim;
        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;

        let chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_full_attention(i) {
                    Ok(None)
                } else {
                    GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                        .map(Some)
                        .map_err(|e| anyhow::anyhow!("metal v2 chunk conv tail alloc: {e}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let token_id_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("metal v2 token id buf: {e}"))?;

        Ok(Self {
            scratch,
            chunk_conv_tail,
            token_id_buf,
        })
    }
}

/// Per-token forward pass body shared by `metal_v2_decode_step` (full-logits)
/// and `metal_v2_decode_step_greedy` (fused argmax). Performs token embed +
/// 24-layer transformer pass, leaving the post-final-layer hidden state in
/// `scratch.scratch.hidden`. Caller is responsible for the final RMSNorm +
/// lm_head and for owning a `MetalBatchGuard` around the call.
fn metal_v2_decode_step_body(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MetalV2DecodeScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<()> {
    let config = &weights.config;
    let hidden_dim = config.hidden_size;
    let chunk_len = 1usize;
    let chunk_start = seqlen_offset;

    // Seed the inter-chunk conv tail buffers from the persistent layer state.
    // The recurrent state lives on the layer in F32 and `prefill_linear_attention_layer`
    // reads/writes it directly, so no separate seeding is needed for it.
    let kern = config.linear_conv_kernel_dim;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    for idx in 0..config.num_hidden_layers {
        if config.is_full_attention(idx) {
            continue;
        }
        let chunk_tail = scratch.chunk_conv_tail[idx]
            .as_mut()
            .expect("metal v2 chunk conv tail missing for linear layer");
        if let Some(conv_state) = state.layers[idx].conv_state.as_ref() {
            let bytes = qkv_dim * (kern - 1) * ScalarType::BF16.size_in_bytes();
            copy_d2d_batched(
                ordinal,
                chunk_tail.as_ptr() as *mut c_void,
                conv_state.as_ptr(),
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("metal v2 layer {idx} seed conv tail: {e}"))?;
        }
    }

    // Upload the single token id and embed it into hidden[0].
    let id_bytes = token_id.to_le_bytes();
    copy_h2d(
        ordinal,
        scratch.token_id_buf.as_ptr() as *mut c_void,
        id_bytes.as_ptr() as *const c_void,
        4,
    )
    .map_err(|e| anyhow::anyhow!("metal v2 token id upload: {e}"))?;
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        config.vocab_size,
        hidden_dim,
        &weights.embed_tokens,
        &scratch.token_id_buf,
        &mut scratch.scratch.hidden,
    )?;
    scratch.scratch.seed_f32_from_hidden(
        ordinal,
        chunk_len * hidden_dim,
        "decode-loop embedding",
    )?;

    for idx in 0..config.num_hidden_layers {
        scratch.scratch.rms_norm_hidden_to_normed_model(
            config,
            ordinal,
            chunk_len,
            hidden_dim,
            &weights.layers[idx].input_norm_w,
            &format!("layer {idx} input norm"),
        )?;

        if config.is_full_attention(idx) {
            prefill_full_attention_layer(
                weights,
                state,
                rotary,
                &mut scratch.scratch,
                config,
                idx,
                chunk_len,
                chunk_start,
                ordinal,
                kv_chunk_size,
                /* commit_kv_filled */ true,
                None, // NEW: metal v2 decode never sparsifies
            )?;
        } else {
            let mut no_debug_trace = None;
            let chunk_tail = scratch.chunk_conv_tail[idx].as_mut().unwrap();
            prefill_linear_attention_layer(
                weights,
                state,
                &mut scratch.scratch,
                config,
                idx,
                chunk_len,
                chunk_start,
                /* is_last_chunk */ true,
                chunk_tail,
                ordinal,
                false,
                &mut no_debug_trace,
                None,
            )?;
        }

        scratch.scratch.rms_norm_hidden_to_normed_model(
            config,
            ordinal,
            chunk_len,
            hidden_dim,
            &weights.layers[idx].post_attn_norm_w,
            &format!("layer {idx} post-attn norm"),
        )?;

        prefill_mlp_layer(
            weights,
            &mut scratch.scratch,
            config,
            idx,
            chunk_len,
            ordinal,
        )?;
    }

    Ok(())
}

/// Single-token incremental decode step for Metal. Mirrors `prefill_inner`'s
/// chunk-loop body with `chunk_len=1, chunk_start=seqlen_offset, is_last_chunk=true`,
/// reading and writing the persistent layer state in place. Replaces Metal v1's
/// O(N²) replay-prefill path with O(N)-per-step proper incremental decode.
///
/// Returns the full BF16→f32 logits row over the vocabulary. Use
/// `metal_v2_decode_step_greedy` to skip the 250k-element D2H + host argmax
/// when only the sampled token is needed.
pub fn metal_v2_decode_step(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MetalV2DecodeScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<Vec<f32>> {
    // Open one Metal batch around the entire per-token forward pass so all
    // ~800 kernel dispatches end up in a single command buffer rather than
    // committing and waiting individually.
    let _metal_batch = prefill_ffi::MetalBatchGuard::begin()
        .map_err(|e| anyhow::anyhow!("metal v2 batch begin: {e}"))?;

    metal_v2_decode_step_body(
        weights,
        state,
        rotary,
        scratch,
        token_id,
        seqlen_offset,
        ordinal,
        kv_chunk_size,
    )?;

    let config = &weights.config;
    let (mut logits_per_pos, _normed) = compute_logits_for_range(
        &scratch.scratch.hidden,
        weights,
        config,
        0,
        1,
        false,
        ordinal,
    )?;
    let logits = logits_per_pos
        .pop()
        .expect("count=1 produces exactly one row");
    Ok(logits)
}

/// Same forward pass as `metal_v2_decode_step`, but uses the fused
/// lm_head + argmax kernel and returns just the sampled token id. Skips the
/// per-token 250k-element BF16 D2H, the bf16→f32 conversion, and the host
/// argmax loop. Use this when full logits aren't needed (no validation,
/// no rescore, plain greedy decode).
pub fn metal_v2_decode_step_greedy(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MetalV2DecodeScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    let _metal_batch = prefill_ffi::MetalBatchGuard::begin()
        .map_err(|e| anyhow::anyhow!("metal v2 batch begin: {e}"))?;

    metal_v2_decode_step_body(
        weights,
        state,
        rotary,
        scratch,
        token_id,
        seqlen_offset,
        ordinal,
        kv_chunk_size,
    )?;

    let config = &weights.config;
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    // Final RMSNorm + fused lm_head argmax. For chunk_len=1, scratch.hidden is
    // already a single row, so we apply RMSNorm in-place via a temp slice.
    let slice = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
        .map_err(|e| anyhow::anyhow!("greedy slice alloc: {e}"))?;
    copy_d2d_batched(
        ordinal,
        slice.as_ptr() as *mut c_void,
        scratch.scratch.hidden.as_ptr(),
        hidden_dim * elem_bytes,
    )
    .map_err(|e| anyhow::anyhow!("greedy slice copy: {e}"))?;

    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
        .map_err(|e| anyhow::anyhow!("greedy normed alloc: {e}"))?;
    rms_norm_rows_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &slice,
        &weights.norm_weight,
        &mut normed,
        "greedy final norm",
    )?;

    let mut out_index = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .map_err(|e| anyhow::anyhow!("greedy out_index alloc: {e}"))?;
    if weights.lm_head_lowbit_params(hidden_dim).is_some() {
        // INT4/GQH lm_head: there is no fused matmul + argmax kernel, so do
        // them separately. Both run inside the open Metal batch — the cost of
        // the extra argmax dispatch is dwarfed by the 4x bandwidth win on the
        // matmul.
        let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, vocab_size])
            .map_err(|e| anyhow::anyhow!("greedy int4 logits alloc: {e}"))?;
        prefill_lm_head_lowbit(
            ordinal,
            1,
            vocab_size,
            hidden_dim,
            &normed,
            weights,
            &mut logits_buf,
            "greedy lm_head",
        )?;
        kernel_ffi::metal_argmax_bf16_into(&logits_buf, &mut out_index, vocab_size)
            .map_err(|e| anyhow::anyhow!("greedy int4 argmax: {e}"))?;
    } else {
        kernel_ffi::metal_lm_head_argmax_bf16_into(
            &normed,
            &*weights.lm_head,
            &mut out_index,
            hidden_dim,
            vocab_size,
        )
        .map_err(|e| anyhow::anyhow!("greedy fused lm_head argmax: {e}"))?;
    }

    // Flush before D2H so the GPU work is visible to the host memcpy.
    if prefill_ffi::metal_batch_is_active() {
        prefill_ffi::flush_metal_batch().map_err(|e| anyhow::anyhow!("greedy batch flush: {e}"))?;
    }

    let bytes = out_index
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("greedy token D2H: {e}"))?;
    let token = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    Ok(token)
}

/// Per-layer full-attention prefill step.
///
/// `commit_kv_filled`: when false, K/V are written to the cache at positions
/// `[chunk_start, chunk_start + chunk_len)` but `ls.kv_filled` is NOT
/// advanced. That's the DFlash verify path per docs/dflash.md §6 — the
/// speculative engine owns the post-acceptance `set_kv_filled(L + k + 1)`
/// call and harmlessly overwrites the tail on the next round. Normal
/// prefill passes `true`.
fn prefill_full_attention_layer(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    chunk_len: usize,
    chunk_start: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    commit_kv_filled: bool,
    kept_positions_chunk: Option<&[u32]>, // NEW
) -> Result<()> {
    let fw = weights.layers[idx]
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
    let kv_len = chunk_start + chunk_len; // total KV length after this chunk

    let pos_ids_buf: Option<GpuBuffer> = if let Some(kept) = kept_positions_chunk {
        if kept.len() != chunk_len {
            return Err(anyhow::anyhow!(
                "layer {idx}: kept_positions_chunk has {} entries but chunk_len is {chunk_len}",
                kept.len()
            ));
        }
        let mut buf = GpuBuffer::alloc(ordinal, ScalarType::U32, &[kept.len()])
            .map_err(|e| anyhow::anyhow!("layer {idx} pos_ids alloc: {e}"))?;
        let bytes =
            unsafe { std::slice::from_raw_parts(kept.as_ptr() as *const u8, kept.len() * 4) };
        copy_h2d(
            ordinal,
            buf.as_mut_ptr(),
            bytes.as_ptr() as *const _,
            bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} pos_ids upload: {e}"))?;
        Some(buf)
    } else {
        None
    };

    // 1. Q projection
    let mut q_full = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, q_proj_dim])
        .map_err(|e| anyhow::anyhow!("q_full alloc: {e}"))?;
    matmul_proj(
        ordinal,
        1,
        chunk_len,
        q_proj_dim,
        hidden_dim,
        &scratch.normed,
        &fw.q_proj_w,
        fw.q_proj_scale.as_ref(),
        fw.q_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut q_full,
        fw.q_proj_int4_scale.as_ref(),
        fw.q_proj_int4_zero.as_ref(),
        fw.q_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    // 2. Split Q into query and gate when present. Llama-style full attention
    // uses an ungated q_proj whose row count matches q_dim exactly.
    let mut query_buf = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
        .map_err(|e| anyhow::anyhow!("query_buf alloc: {e}"))?;
    let mut gate_buf = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
        .map_err(|e| anyhow::anyhow!("gate_buf alloc: {e}"))?;
    let q_norm_done = if has_attn_gate {
        if maybe_split_qgate_norm_bf16(
            config,
            ordinal,
            chunk_len,
            num_q_heads,
            head_dim,
            &q_full,
            fw.q_norm_w.as_ref(),
            &mut query_buf,
            &mut gate_buf,
            &format!("layer {idx} fused Q split+norm"),
        )? {
            true
        } else {
            prefill_ffi::split_qgate(
                ordinal,
                ScalarType::BF16,
                chunk_len,
                num_q_heads,
                head_dim,
                &q_full,
                &mut query_buf,
                &mut gate_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} Q split: {e}"))?;
            false
        }
    } else {
        copy_d2d_batched(
            ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_full.as_ptr(),
            chunk_len * q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q copy: {e}"))?;
        false
    };

    // 3. K projection
    matmul_proj(
        ordinal,
        1,
        chunk_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.k_proj_w,
        fw.k_proj_scale.as_ref(),
        fw.k_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut scratch.proj_buf2,
        fw.k_proj_int4_scale.as_ref(),
        fw.k_proj_int4_zero.as_ref(),
        fw.k_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    // 4. Q normalization
    if !q_norm_done
        && !maybe_attn_rms_norm_rows_inplace(
            config,
            ordinal,
            chunk_len * num_q_heads,
            head_dim,
            &mut query_buf,
            fw.q_norm_w.as_ref(),
            &format!("layer {idx} Q norm inplace"),
        )?
    {
        let mut q_normed = GpuBuffer::alloc(
            ordinal,
            ScalarType::BF16,
            &[chunk_len * num_q_heads, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("q_normed alloc: {e}"))?;
        maybe_attn_rms_norm_rows(
            config,
            ordinal,
            chunk_len * num_q_heads,
            head_dim,
            &query_buf,
            fw.q_norm_w.as_ref(),
            &mut q_normed,
            &format!("layer {idx} Q norm"),
        )?;
        copy_d2d_batched(
            ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            chunk_len * q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q norm copy: {e}"))?;
    }

    // 5. K normalization
    if !maybe_attn_rms_norm_rows_inplace(
        config,
        ordinal,
        chunk_len * num_kv_heads,
        head_dim,
        &mut scratch.proj_buf2,
        fw.k_norm_w.as_ref(),
        &format!("layer {idx} K norm inplace"),
    )? {
        let mut k_normed = GpuBuffer::alloc(
            ordinal,
            ScalarType::BF16,
            &[chunk_len * num_kv_heads, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("k_normed alloc: {e}"))?;
        maybe_attn_rms_norm_rows(
            config,
            ordinal,
            chunk_len * num_kv_heads,
            head_dim,
            &scratch.proj_buf2,
            fw.k_norm_w.as_ref(),
            &mut k_normed,
            &format!("layer {idx} K norm"),
        )?;
        copy_d2d_batched(
            ordinal,
            scratch.proj_buf2.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            chunk_len * kv_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K norm copy: {e}"))?;
    }

    // 6. RoPE on query - use pos_offset = chunk_start for the dense path,
    //    or apply_rope_prefill_indirect with the kept positions for SpecPrefill.
    if let Some(pos_ids) = pos_ids_buf.as_ref() {
        prefill_ffi::apply_rope_prefill_indirect(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_q_heads,
            head_dim,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_ids,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE (indirect): {e}"))?;
    } else {
        prefill_ffi::apply_rope_prefill(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_q_heads,
            head_dim,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            chunk_start,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;
    }
    if let Some(pos_ids) = pos_ids_buf.as_ref() {
        prefill_ffi::apply_rope_prefill_indirect(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_ids,
            &mut scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K RoPE (indirect): {e}"))?;
    } else {
        prefill_ffi::apply_rope_prefill(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            chunk_start,
            &mut scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K RoPE: {e}"))?;
    }

    // 7. V projection
    let mut v_buf = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, kv_dim])
        .map_err(|e| anyhow::anyhow!("v_buf alloc: {e}"))?;
    matmul_proj(
        ordinal,
        1,
        chunk_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.v_proj_w,
        fw.v_proj_scale.as_ref(),
        fw.v_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut v_buf,
        fw.v_proj_int4_scale.as_ref(),
        fw.v_proj_int4_zero.as_ref(),
        fw.v_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    // 8/9. Write this chunk's K/V to KV cache BEFORE attention (so attention can read from it).
    //      The HIP fast path transposes directly into the persistent cache; fallback keeps the
    //      scratch transpose plus per-head copy path for virtual caches and debug A/B runs.
    let mut kv_capacity_prepared = false;
    let kv_cache_written = if gpu_hal::current_backend() == Backend::Hip
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_KV_CACHE_WRITE").is_none()
    {
        let ls = &mut state.layers[idx];
        ls.ensure_kv_capacity(kv_len - 1, ordinal, config, kv_chunk_size, false)
            .map_err(|e| anyhow::anyhow!("layer {idx} KV alloc: {e}"))?;
        kv_capacity_prepared = true;
        if !ls.has_virtual_kv_cache() && ls.kv_cache_k.is_some() && ls.kv_cache_v.is_some() {
            let cap = ls.kv_capacity();
            let k_cache = ls
                .kv_cache_k
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing K cache"))?;
            prefill_ffi::transpose_shd_to_cache_bf16(
                ordinal,
                chunk_len,
                num_kv_heads,
                head_dim,
                cap,
                chunk_start,
                &scratch.proj_buf2,
                k_cache,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused KV cache K write: {e}"))?;
            let v_cache = ls
                .kv_cache_v
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing V cache"))?;
            prefill_ffi::transpose_shd_to_cache_bf16(
                ordinal,
                chunk_len,
                num_kv_heads,
                head_dim,
                cap,
                chunk_start,
                &v_buf,
                v_cache,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused KV cache V write: {e}"))?;
            true
        } else {
            false
        }
    } else {
        false
    };

    if !kv_cache_written {
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_kv_heads,
            head_dim,
            &scratch.proj_buf2,
            &mut scratch.attn_k,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K transpose: {e}"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            num_kv_heads,
            head_dim,
            &v_buf,
            &mut scratch.attn_v,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} V transpose: {e}"))?;

        let ls = &mut state.layers[idx];
        if !kv_capacity_prepared {
            ls.ensure_kv_capacity(kv_len - 1, ordinal, config, kv_chunk_size, false)
                .map_err(|e| anyhow::anyhow!("layer {idx} KV alloc: {e}"))?;
        }

        if ls.kv_cache_k_ptr().is_some() {
            let bytes_per_chunk_head = chunk_len * head_dim * elem_bytes;
            let cap = ls.kv_capacity();
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = chunk_len * head_dim * elem_bytes;
            let dst_pos_offset = chunk_start * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                let dst = ls
                    .kv_cache_k_offset_ptr(h * cap_stride + dst_pos_offset)
                    .ok_or_else(|| anyhow::anyhow!("layer {idx} missing K cache"))?;
                copy_d2d_batched(
                    ordinal,
                    dst as *mut c_void,
                    scratch.attn_k.offset_ptr(h * src_stride),
                    bytes_per_chunk_head,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} KV cache K write h={h}: {e}"))?;
            }
        }
        if ls.kv_cache_v_ptr().is_some() {
            let bytes_per_chunk_head = chunk_len * head_dim * elem_bytes;
            let cap = ls.kv_capacity();
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = chunk_len * head_dim * elem_bytes;
            let dst_pos_offset = chunk_start * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                let dst = ls
                    .kv_cache_v_offset_ptr(h * cap_stride + dst_pos_offset)
                    .ok_or_else(|| anyhow::anyhow!("layer {idx} missing V cache"))?;
                copy_d2d_batched(
                    ordinal,
                    dst as *mut c_void,
                    scratch.attn_v.offset_ptr(h * src_stride),
                    bytes_per_chunk_head,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} KV cache V write h={h}: {e}"))?;
            }
        }
    }
    if commit_kv_filled {
        state.layers[idx].set_kv_filled(kv_len);
    }

    // 10. Transpose Q to [H, chunk_len, D]
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        num_q_heads,
        head_dim,
        &query_buf,
        &mut scratch.attn_q,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q transpose: {e}"))?;

    // 11. Causal attention — Q: [q_heads, chunk_len, hd], K/V: [kv_heads, kv_len, hd]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let ls = &mut state.layers[idx];
    let cap = ls.kv_capacity();
    let kv_k_contig;
    let kv_v_contig;
    let attn_k_ref: &GpuBuffer;
    let attn_v_ref: &GpuBuffer;

    if !ls.has_virtual_kv_cache() && cap == kv_len {
        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
        // No padding - cache is already contiguous, use directly.
        attn_k_ref = cache_k_ref;
        attn_v_ref = cache_v_ref;
    } else {
        // Capacity > kv_len - copy each head's kv_len entries into contiguous buffers.
        // Virtual KV also uses this path because the prefill attention FFI takes
        // `GpuBuffer` wrappers, while the virtual cache is represented by raw VA.
        kv_k_contig =
            GpuBuffer::alloc(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("kv_k_contig alloc: {e}"))?;
        kv_v_contig =
            GpuBuffer::alloc(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("kv_v_contig alloc: {e}"))?;
        let cap_stride = cap * head_dim * elem_bytes;
        let contig_stride = kv_len * head_dim * elem_bytes;
        let copy_bytes = kv_len * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            let src_k = ls
                .kv_cache_k_offset_ptr(h * cap_stride)
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing K cache"))?;
            let src_v = ls
                .kv_cache_v_offset_ptr(h * cap_stride)
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing V cache"))?;
            copy_d2d_batched(
                ordinal,
                kv_k_contig.offset_ptr(h * contig_stride) as *mut c_void,
                src_k,
                copy_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV assemble K h={h}: {e}"))?;
            copy_d2d_batched(
                ordinal,
                kv_v_contig.offset_ptr(h * contig_stride) as *mut c_void,
                src_v,
                copy_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV assemble V h={h}: {e}"))?;
        }
        attn_k_ref = &kv_k_contig;
        attn_v_ref = &kv_v_contig;
    }

    prefill_ffi::full_attention_prefill(
        ordinal,
        ScalarType::BF16,
        1,
        num_q_heads,
        num_kv_heads,
        chunk_len,
        kv_len,
        head_dim,
        scale,
        chunk_start,
        &scratch.attn_q,
        attn_k_ref,
        attn_v_ref,
        &mut scratch.attn_out_f32,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

    let fused_attn_gate_prep = has_attn_gate
        && gpu_hal::current_backend() == Backend::Hip
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_ATTN_GATE_PREP").is_none();
    if fused_attn_gate_prep {
        prefill_ffi::cast_transpose_gate_hsd_to_shd_bf16(
            ordinal,
            chunk_len,
            num_q_heads,
            head_dim,
            &scratch.attn_out_f32,
            &gate_buf,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused attn gate prep: {e}"))?;
    } else {
        // 12. Cast F32 → BF16
        prefill_ffi::cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            num_q_heads * chunk_len * head_dim,
            &scratch.attn_out_f32,
            &mut scratch.attn_q,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;

        // 13. Transpose back [H, chunk_len, D] → [chunk_len, H, D] = [chunk_len, q_dim]
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            num_q_heads,
            chunk_len,
            head_dim,
            &scratch.attn_q,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn transpose back: {e}"))?;

        // 14. Apply attention gate only for gated-Q attention models (Qwen).
        if has_attn_gate {
            if gpu_hal::current_backend() == Backend::Hip
                && env::var_os("SUPERSONIC_DFLASH_DISABLE_INPLACE_ATTN_GATE").is_none()
            {
                prefill_ffi::sigmoid_mul_inplace(
                    ordinal,
                    ScalarType::BF16,
                    chunk_len * q_dim,
                    &mut scratch.proj_buf,
                    &gate_buf,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} gate inplace: {e}"))?;
            } else {
                let mut gated = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
                    .map_err(|e| anyhow::anyhow!("gated alloc: {e}"))?;
                prefill_ffi::sigmoid_mul(
                    ordinal,
                    ScalarType::BF16,
                    chunk_len * q_dim,
                    &scratch.proj_buf,
                    &gate_buf,
                    &mut gated,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} gate: {e}"))?;
                copy_d2d_batched(
                    ordinal,
                    scratch.proj_buf.as_ptr() as *mut c_void,
                    gated.as_ptr(),
                    chunk_len * q_dim * elem_bytes,
                )
                .map_err(|e| anyhow::anyhow!("gated copy: {e}"))?;
            }
        }
    }

    // 15-16. O projection + residual
    let fused_residual = !scratch.has_f32_activation_carry()
        && matmul_proj_residual_add_inplace(
            ordinal,
            1,
            chunk_len,
            hidden_dim,
            q_dim,
            &scratch.proj_buf,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            fw.o_proj_int8_scale.as_ref(),
            &mut scratch.hidden,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            fw.o_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    if !fused_residual {
        matmul_proj(
            ordinal,
            1,
            chunk_len,
            hidden_dim,
            q_dim,
            &scratch.proj_buf,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            fw.o_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            fw.o_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;

        scratch.residual_add_from_source(
            ordinal,
            chunk_len * hidden_dim,
            ResidualSource::ProjBuf2,
            "attention residual",
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prefill_tree_full_attention_layer(
    weights: &Qwen35Weights,
    state: &ModelState,
    rotary: &RotaryTables,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    tree_len: usize,
    prefix_len: usize,
    ordinal: usize,
    _kv_chunk_size: usize,
    positions_gpu: &GpuBuffer,
    visibility_gpu: &GpuBuffer,
    mut capture_slot: Option<&mut Option<PrefillTreeLayerRollback>>,
) -> Result<()> {
    let fw = weights.layers[idx]
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

    matmul_proj(
        ordinal,
        1,
        tree_len,
        q_proj_dim,
        hidden_dim,
        &scratch.normed,
        &fw.q_proj_w,
        fw.q_proj_scale.as_ref(),
        fw.q_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut scratch.full_q_buf,
        fw.q_proj_int4_scale.as_ref(),
        fw.q_proj_int4_zero.as_ref(),
        fw.q_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    let q_norm_done = if has_attn_gate {
        if maybe_split_qgate_norm_bf16(
            config,
            ordinal,
            tree_len,
            num_q_heads,
            head_dim,
            &scratch.full_q_buf,
            fw.q_norm_w.as_ref(),
            &mut scratch.full_query_buf,
            &mut scratch.full_gate_buf,
            &format!("tree layer {idx} fused Q split+norm"),
        )? {
            true
        } else {
            prefill_ffi::split_qgate(
                ordinal,
                ScalarType::BF16,
                tree_len,
                num_q_heads,
                head_dim,
                &scratch.full_q_buf,
                &mut scratch.full_query_buf,
                &mut scratch.full_gate_buf,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} Q split: {e}"))?;
            false
        }
    } else {
        copy_d2d_batched(
            ordinal,
            scratch.full_query_buf.as_ptr() as *mut c_void,
            scratch.full_q_buf.as_ptr(),
            tree_len * q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Q copy: {e}"))?;
        false
    };

    matmul_proj(
        ordinal,
        1,
        tree_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.k_proj_w,
        fw.k_proj_scale.as_ref(),
        fw.k_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut scratch.proj_buf2,
        fw.k_proj_int4_scale.as_ref(),
        fw.k_proj_int4_zero.as_ref(),
        fw.k_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    if !q_norm_done
        && !maybe_attn_rms_norm_rows_inplace(
            config,
            ordinal,
            tree_len * num_q_heads,
            head_dim,
            &mut scratch.full_query_buf,
            fw.q_norm_w.as_ref(),
            &format!("tree layer {idx} Q norm inplace"),
        )?
    {
        let mut q_normed = GpuBuffer::alloc(
            ordinal,
            ScalarType::BF16,
            &[tree_len * num_q_heads, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("tree q_normed alloc: {e}"))?;
        maybe_attn_rms_norm_rows(
            config,
            ordinal,
            tree_len * num_q_heads,
            head_dim,
            &scratch.full_query_buf,
            fw.q_norm_w.as_ref(),
            &mut q_normed,
            &format!("tree layer {idx} Q norm"),
        )?;
        copy_d2d_batched(
            ordinal,
            scratch.full_query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            tree_len * q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Q norm copy: {e}"))?;
    }

    if !maybe_attn_rms_norm_rows_inplace(
        config,
        ordinal,
        tree_len * num_kv_heads,
        head_dim,
        &mut scratch.proj_buf2,
        fw.k_norm_w.as_ref(),
        &format!("tree layer {idx} K norm inplace"),
    )? {
        let mut k_normed = GpuBuffer::alloc(
            ordinal,
            ScalarType::BF16,
            &[tree_len * num_kv_heads, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("tree k_normed alloc: {e}"))?;
        maybe_attn_rms_norm_rows(
            config,
            ordinal,
            tree_len * num_kv_heads,
            head_dim,
            &scratch.proj_buf2,
            fw.k_norm_w.as_ref(),
            &mut k_normed,
            &format!("tree layer {idx} K norm"),
        )?;
        copy_d2d_batched(
            ordinal,
            scratch.proj_buf2.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            tree_len * kv_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} K norm copy: {e}"))?;
    }

    prefill_ffi::apply_rope_prefill_indirect(
        ordinal,
        ScalarType::BF16,
        tree_len,
        num_q_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        positions_gpu,
        &mut scratch.full_query_buf,
    )
    .map_err(|e| anyhow::anyhow!("tree layer {idx} Q RoPE: {e}"))?;
    prefill_ffi::apply_rope_prefill_indirect(
        ordinal,
        ScalarType::BF16,
        tree_len,
        num_kv_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        positions_gpu,
        &mut scratch.proj_buf2,
    )
    .map_err(|e| anyhow::anyhow!("tree layer {idx} K RoPE: {e}"))?;

    matmul_proj(
        ordinal,
        1,
        tree_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.v_proj_w,
        fw.v_proj_scale.as_ref(),
        fw.v_proj_int8_scale.as_ref(),
        weights.fp8_block_size,
        &mut scratch.full_v_buf,
        fw.v_proj_int4_scale.as_ref(),
        fw.v_proj_int4_zero.as_ref(),
        fw.v_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;

    let use_fused_tree_full_kv_transpose = gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_TREE_FULL_KV_TRANSPOSE").is_none();
    if use_fused_tree_full_kv_transpose {
        prefill_ffi::transpose_shd_hsd_pair(
            ordinal,
            ScalarType::BF16,
            tree_len,
            num_kv_heads,
            head_dim,
            &scratch.proj_buf2,
            &scratch.full_v_buf,
            &mut scratch.attn_k,
            &mut scratch.attn_v,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused K/V transpose: {e}"))?;
    } else {
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            tree_len,
            num_kv_heads,
            head_dim,
            &scratch.proj_buf2,
            &mut scratch.attn_k,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} K transpose: {e}"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            tree_len,
            num_kv_heads,
            head_dim,
            &scratch.full_v_buf,
            &mut scratch.attn_v,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} V transpose: {e}"))?;
    }
    if let Some(slot) = capture_slot.as_deref_mut() {
        let bytes = num_kv_heads * tree_len * head_dim * elem_bytes;
        let needs_alloc = !matches!(
            slot.as_ref(),
            Some(PrefillTreeLayerRollback::Full { tree_k, tree_v })
                if tree_k.device_ordinal() == ordinal
                    && tree_v.device_ordinal() == ordinal
                    && tree_k.dtype() == ScalarType::BF16
                    && tree_v.dtype() == ScalarType::BF16
                    && tree_k.len_bytes() >= bytes
                    && tree_v.len_bytes() >= bytes
        );
        if needs_alloc {
            let tree_k = GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, tree_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} rollback K alloc: {e}"))?;
            let tree_v = GpuBuffer::alloc(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, tree_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} rollback V alloc: {e}"))?;
            *slot = Some(PrefillTreeLayerRollback::Full { tree_k, tree_v });
        }
        let Some(PrefillTreeLayerRollback::Full { tree_k, tree_v }) = slot.as_mut() else {
            unreachable!("tree full-attention rollback slot was just initialized");
        };
        copy_d2d_batched(ordinal, tree_k.as_mut_ptr(), scratch.attn_k.as_ptr(), bytes)
            .map_err(|e| anyhow::anyhow!("tree layer {idx} rollback K capture: {e}"))?;
        copy_d2d_batched(ordinal, tree_v.as_mut_ptr(), scratch.attn_v.as_ptr(), bytes)
            .map_err(|e| anyhow::anyhow!("tree layer {idx} rollback V capture: {e}"))?;
    }
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        tree_len,
        num_q_heads,
        head_dim,
        &scratch.full_query_buf,
        &mut scratch.attn_q,
    )
    .map_err(|e| anyhow::anyhow!("tree layer {idx} Q transpose: {e}"))?;

    let ls = &state.layers[idx];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let use_strided_tree_prefix = prefix_len > 0
        && gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_ENABLE_TREE_FULL_PREFIX_STRIDED").is_some()
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_TREE_FULL_PREFIX_STRIDED").is_none();
    if use_strided_tree_prefix {
        let cap = ls.kv_capacity();
        if cap < prefix_len {
            return Err(anyhow::anyhow!(
                "tree layer {idx}: KV capacity {cap} < prefix_len {prefix_len}"
            ));
        }
        let prefix_k = ls
            .kv_cache_k_offset_ptr(0)
            .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing K cache"))?;
        let prefix_v = ls
            .kv_cache_v_offset_ptr(0)
            .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing V cache"))?;
        prefill_ffi::full_attention_tree_prefill_strided_raw(
            ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            num_kv_heads,
            tree_len,
            prefix_len,
            cap,
            head_dim,
            scale,
            &scratch.attn_q,
            prefix_k,
            prefix_v,
            &scratch.attn_k,
            &scratch.attn_v,
            visibility_gpu,
            &mut scratch.attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} strided attention: {e}"))?;
    } else {
        let prefix_k_storage;
        let prefix_v_storage;
        let zero_prefix_k;
        let zero_prefix_v;
        let prefix_k_ref: &GpuBuffer;
        let prefix_v_ref: &GpuBuffer;
        if prefix_len == 0 {
            zero_prefix_k =
                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                    .map_err(|e| anyhow::anyhow!("tree zero prefix K alloc: {e}"))?;
            zero_prefix_v =
                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                    .map_err(|e| anyhow::anyhow!("tree zero prefix V alloc: {e}"))?;
            prefix_k_ref = &zero_prefix_k;
            prefix_v_ref = &zero_prefix_v;
        } else {
            let cap = ls.kv_capacity();
            if cap < prefix_len {
                return Err(anyhow::anyhow!(
                    "tree layer {idx}: KV capacity {cap} < prefix_len {prefix_len}"
                ));
            }
            if !ls.has_virtual_kv_cache() && cap == prefix_len {
                prefix_k_ref = ls
                    .kv_cache_k
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing K cache"))?;
                prefix_v_ref = ls
                    .kv_cache_v
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing V cache"))?;
            } else {
                prefix_k_storage = GpuBuffer::alloc(
                    ordinal,
                    ScalarType::BF16,
                    &[num_kv_heads, prefix_len, head_dim],
                )
                .map_err(|e| anyhow::anyhow!("tree prefix K alloc: {e}"))?;
                prefix_v_storage = GpuBuffer::alloc(
                    ordinal,
                    ScalarType::BF16,
                    &[num_kv_heads, prefix_len, head_dim],
                )
                .map_err(|e| anyhow::anyhow!("tree prefix V alloc: {e}"))?;
                let cap_stride = cap * head_dim * elem_bytes;
                let contig_stride = prefix_len * head_dim * elem_bytes;
                let copy_bytes = prefix_len * head_dim * elem_bytes;
                for h in 0..num_kv_heads {
                    let src_k = ls
                        .kv_cache_k_offset_ptr(h * cap_stride)
                        .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing K cache"))?;
                    let src_v = ls
                        .kv_cache_v_offset_ptr(h * cap_stride)
                        .ok_or_else(|| anyhow::anyhow!("tree layer {idx} missing V cache"))?;
                    copy_d2d_batched(
                        ordinal,
                        prefix_k_storage.offset_ptr(h * contig_stride) as *mut c_void,
                        src_k,
                        copy_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("tree layer {idx} prefix K copy h={h}: {e}"))?;
                    copy_d2d_batched(
                        ordinal,
                        prefix_v_storage.offset_ptr(h * contig_stride) as *mut c_void,
                        src_v,
                        copy_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("tree layer {idx} prefix V copy h={h}: {e}"))?;
                }
                prefix_k_ref = &prefix_k_storage;
                prefix_v_ref = &prefix_v_storage;
            }
        }

        prefill_ffi::full_attention_tree_prefill(
            ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            num_kv_heads,
            tree_len,
            prefix_len,
            head_dim,
            scale,
            &scratch.attn_q,
            prefix_k_ref,
            prefix_v_ref,
            &scratch.attn_k,
            &scratch.attn_v,
            visibility_gpu,
            &mut scratch.attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} attention: {e}"))?;
    }

    let fused_attn_gate_prep = has_attn_gate
        && gpu_hal::current_backend() == Backend::Hip
        && env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_ATTN_GATE_PREP").is_none();
    if fused_attn_gate_prep {
        prefill_ffi::cast_transpose_gate_hsd_to_shd_bf16(
            ordinal,
            tree_len,
            num_q_heads,
            head_dim,
            &scratch.attn_out_f32,
            &scratch.full_gate_buf,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused attn gate prep: {e}"))?;
    } else {
        prefill_ffi::cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            num_q_heads * tree_len * head_dim,
            &scratch.attn_out_f32,
            &mut scratch.attn_q,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} attn cast: {e}"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            num_q_heads,
            tree_len,
            head_dim,
            &scratch.attn_q,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} attn transpose back: {e}"))?;

        if has_attn_gate {
            if gpu_hal::current_backend() == Backend::Hip
                && env::var_os("SUPERSONIC_DFLASH_DISABLE_INPLACE_ATTN_GATE").is_none()
            {
                prefill_ffi::sigmoid_mul_inplace(
                    ordinal,
                    ScalarType::BF16,
                    tree_len * q_dim,
                    &mut scratch.proj_buf,
                    &scratch.full_gate_buf,
                )
                .map_err(|e| anyhow::anyhow!("tree layer {idx} gate inplace: {e}"))?;
            } else {
                let mut gated = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, q_dim])
                    .map_err(|e| anyhow::anyhow!("tree gated alloc: {e}"))?;
                prefill_ffi::sigmoid_mul(
                    ordinal,
                    ScalarType::BF16,
                    tree_len * q_dim,
                    &scratch.proj_buf,
                    &scratch.full_gate_buf,
                    &mut gated,
                )
                .map_err(|e| anyhow::anyhow!("tree layer {idx} gate: {e}"))?;
                copy_d2d_batched(
                    ordinal,
                    scratch.proj_buf.as_ptr() as *mut c_void,
                    gated.as_ptr(),
                    tree_len * q_dim * elem_bytes,
                )
                .map_err(|e| anyhow::anyhow!("tree gated copy: {e}"))?;
            }
        }
    }

    let fused_residual = !scratch.has_f32_activation_carry()
        && matmul_proj_residual_add_inplace(
            ordinal,
            1,
            tree_len,
            hidden_dim,
            q_dim,
            &scratch.proj_buf,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            fw.o_proj_int8_scale.as_ref(),
            &mut scratch.hidden,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            fw.o_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    if !fused_residual {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            hidden_dim,
            q_dim,
            &scratch.proj_buf,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            fw.o_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            fw.o_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;

        residual_add(
            ordinal,
            tree_len * hidden_dim,
            &mut scratch.hidden,
            &scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} attention residual: {e}"))?;
    }

    Ok(())
}

fn prefill_linear_attention_layer(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    chunk_len: usize,
    chunk_start: usize,
    is_last_chunk: bool,
    chunk_conv_tail: &mut GpuBuffer,
    ordinal: usize,
    trace_linear_debug: bool,
    linear_debug_trace: &mut Option<LinearLayerDebugTrace>,
    mut append_capture: Option<&mut PrefillAppendLayerRollback>,
) -> Result<()> {
    let lw = weights.layers[idx]
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear attention weights"))?;

    let hidden_dim = config.hidden_size;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let kern = config.linear_conv_kernel_dim;
    let key_dim = nk * khd; // Q and K share this dimension
    let val_dim = nv * vhd;
    let qkv_dim = key_dim * 2 + val_dim;
    let z_dim = val_dim;

    let fused_qkvz_enabled = if env::var_os("SUPERSONIC_DISABLE_FUSED_QKVZ").is_some() {
        false
    } else {
        env::var_os("SUPERSONIC_ENABLE_FUSED_QKVZ").is_some()
            || (config.hidden_size == 5120 && config.num_hidden_layers == 64)
    };
    let use_fused_qkvz =
        fused_qkvz_enabled && scratch.normed.backend() == Backend::Hip && lw.qkvz_proj_w.is_some();

    // 1. QKV projection: normed [chunk, hidden] -> [chunk, qkv_dim].
    // Raw GGML fused [QKV; Z] avoids a second low-bit matmul, then splits
    // back into the old contiguous buffers.
    if use_fused_qkvz {
        matmul_proj(
            ordinal,
            1,
            chunk_len,
            qkv_dim + z_dim,
            hidden_dim,
            &scratch.normed,
            lw.qkvz_proj_w.as_ref().expect("checked fused QKVZ weight"),
            None,
            None,
            weights.fp8_block_size,
            &mut scratch.mlp_buf,
            None,
            None,
            None,
            0,
        )?;
        prefill_ffi::split_qkvz_bf16(
            ordinal,
            chunk_len,
            qkv_dim,
            z_dim,
            &scratch.mlp_buf,
            &mut scratch.proj_buf,
            &mut scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused QKVZ split: {e}"))?;
    } else {
        matmul_proj(
            ordinal,
            1,
            chunk_len,
            qkv_dim,
            hidden_dim,
            &scratch.normed,
            &lw.qkv_proj_w,
            lw.qkv_proj_scale.as_ref(),
            lw.qkv_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf,
            lw.qkv_proj_int4_scale.as_ref(),
            lw.qkv_proj_int4_zero.as_ref(),
            lw.qkv_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    }
    if trace_linear_debug {
        let normed_bytes = scratch
            .normed
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug normed D2H: {e}"))?;
        let qkv_bytes = scratch
            .proj_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug qkv D2H: {e}"))?;
        let normed_row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let row_bytes = qkv_dim * ScalarType::BF16.size_in_bytes();
        let normed_start = (chunk_len - 1) * normed_row_bytes;
        let qkv_start = (chunk_len - 1) * row_bytes;
        *linear_debug_trace = Some(LinearLayerDebugTrace {
            normed: normed_bytes[normed_start..normed_start + normed_row_bytes].to_vec(),
            qkv: qkv_bytes[qkv_start..qkv_start + row_bytes].to_vec(),
            qkv_tail: Vec::new(),
            conv_window: Vec::new(),
            post_conv: Vec::new(),
            z: Vec::new(),
            packed: Vec::new(),
            rec_apply: Vec::new(),
            attn: Vec::new(),
            gated: Vec::new(),
            proj_out: Vec::new(),
        });
    }

    // Compute the post-this-chunk conv tail in a temporary buffer. We DEFER
    // writing it back to `chunk_conv_tail` / `state.conv_state` until after
    // the conv1d, because the conv_input padding for this chunk reads from
    // `chunk_conv_tail` — that source must still be the PREVIOUS chunk's tail
    // when the conv1d runs. Updating in place here was a cross-platform bug
    // exposed by chunk_len=1 incremental decode (Metal v2): the new tail
    // mixed with the current chunk's QKV got fed back into this same chunk's
    // conv1d window, shifting the inputs.
    let pad = kern - 1;
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let use_fused_conv_prep = gpu_hal::current_backend() == Backend::Hip
        && !trace_linear_debug
        && chunk_start > 0
        && chunk_len >= pad
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_CONV_PREP").is_none();
    if use_fused_conv_prep {
        // The fused helper prepares conv_input and linear_new_tail together
        // below, after the independent Z/B/A projections are issued.
    } else if chunk_len >= pad {
        if trace_linear_debug {
            let trace = linear_debug_trace
                .as_mut()
                .expect("linear debug trace missing");
            let bytes = scratch
                .proj_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} debug qkv tail D2H: {e}"))?;
            let row_bytes = qkv_dim * ScalarType::BF16.size_in_bytes();
            let start = (chunk_len - pad) * row_bytes;
            trace.qkv_tail = bytes[start..start + pad * row_bytes].to_vec();
        }
        prefill_ffi::extract_conv_state(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            qkv_dim,
            pad,
            &scratch.proj_buf,
            &mut scratch.linear_new_tail,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} extract conv state: {e}"))?;
    } else {
        // chunk_len < pad — assemble from previous conv_tail + current chunk's QKV.
        let keep_old = pad - chunk_len;
        if chunk_start == 0 {
            gpu_hal::memset_zeros(
                ordinal,
                scratch.linear_new_tail.as_mut_ptr(),
                qkv_dim * pad * elem_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} new_tail zero: {e}"))?;
        }
        let tail_stride = pad * elem_bytes;
        for ch in 0..qkv_dim {
            // Keep last keep_old entries from old tail
            if keep_old > 0 && chunk_start > 0 {
                let src_off = ch * tail_stride + chunk_len * elem_bytes;
                let dst_off = ch * tail_stride;
                copy_d2d_batched(
                    ordinal,
                    scratch.linear_new_tail.offset_ptr(dst_off) as *mut c_void,
                    chunk_conv_tail.offset_ptr(src_off),
                    keep_old * elem_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} conv tail shift ch={ch}: {e}"))?;
            }
            // Append new QKV values
            for t in 0..chunk_len {
                let src_off = t * qkv_dim * elem_bytes + ch * elem_bytes;
                let dst_off = ch * tail_stride + (keep_old + t) * elem_bytes;
                copy_d2d_batched(
                    ordinal,
                    scratch.linear_new_tail.offset_ptr(dst_off) as *mut c_void,
                    scratch.proj_buf.offset_ptr(src_off),
                    elem_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} conv tail append ch={ch} t={t}: {e}"))?;
            }
        }
    }

    // 2. Z projection: normed [chunk, hidden] -> [chunk, z_dim]
    if !use_fused_qkvz {
        matmul_proj(
            ordinal,
            1,
            chunk_len,
            z_dim,
            hidden_dim,
            &scratch.normed,
            &lw.z_proj_w,
            lw.z_proj_scale.as_ref(),
            lw.z_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            lw.z_proj_int4_scale.as_ref(),
            lw.z_proj_int4_zero.as_ref(),
            lw.z_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    }
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .proj_buf2
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug z D2H: {e}"))?;
        let row_bytes = z_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.z = bytes[start..start + row_bytes].to_vec();
    }

    // 3/4. B and A projections. Baked packages carry an optional fused
    // [B; A] weight so verify can do one BF16 projection instead of two tiny
    // launches. Other paths fall back to the original separate projections.
    let use_fused_ba =
        lw.ba_proj_w.is_some() && scratch.normed.backend() != gpu_hal::Backend::Metal;
    let (ba_buf, b_buf, a_buf): (Option<&GpuBuffer>, Option<&GpuBuffer>, Option<&GpuBuffer>) =
        if use_fused_ba {
            matmul_proj(
                ordinal,
                1,
                chunk_len,
                2 * nv,
                hidden_dim,
                &scratch.normed,
                lw.ba_proj_w.as_ref().expect("checked fused BA weight"),
                None,
                None,
                weights.fp8_block_size,
                &mut scratch.linear_ba_buf,
                None,
                None,
                None,
                0,
            )?;
            (Some(&scratch.linear_ba_buf), None, None)
        } else {
            matmul_proj(
                ordinal,
                1,
                chunk_len,
                nv,
                hidden_dim,
                &scratch.normed,
                &lw.b_proj_w,
                lw.b_proj_scale.as_ref(),
                lw.b_proj_int8_scale.as_ref(),
                weights.fp8_block_size,
                &mut scratch.linear_b_buf,
                None,
                None,
                None,
                0,
            )?;

            matmul_proj(
                ordinal,
                1,
                chunk_len,
                nv,
                hidden_dim,
                &scratch.normed,
                &lw.a_proj_w,
                lw.a_proj_scale.as_ref(),
                lw.a_proj_int8_scale.as_ref(),
                weights.fp8_block_size,
                &mut scratch.linear_a_buf,
                None,
                None,
                None,
                0,
            )?;
            (
                None,
                Some(&scratch.linear_b_buf),
                Some(&scratch.linear_a_buf),
            )
        };

    // 5. Transpose QKV [chunk, qkv_dim] -> [qkv_dim, pad+chunk] for conv input.
    //    DFlash append blocks can prepare the previous tail, transposed rows,
    //    and next tail in one HIP helper; other paths keep the generic helpers.
    if use_fused_conv_prep {
        prefill_ffi::prepare_conv_input_tail(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            qkv_dim,
            pad,
            &scratch.proj_buf,
            chunk_conv_tail,
            &mut scratch.conv_input,
            &mut scratch.linear_new_tail,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused conv prepare: {e}"))?;
    } else {
        prefill_ffi::transpose_pad_conv(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            qkv_dim,
            pad,
            &scratch.proj_buf,
            &mut scratch.conv_input,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} conv transpose+pad: {e}"))?;

        // If not the first chunk, overwrite the zero padding with conv_tail from previous chunk.
        if chunk_start > 0 {
            // conv_input is [qkv_dim, pad+chunk_len] in transposed layout.
            // The first `pad` columns of each row need the previous [qkv_dim, pad] tail.
            prefill_ffi::fill_conv_tail(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + chunk_len,
                chunk_conv_tail,
                &mut scratch.conv_input,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} conv pad fill: {e}"))?;
        }
    }
    if let Some(capture) = append_capture.as_mut() {
        copy_d2d_batched(
            ordinal,
            capture.conv_input.as_ptr() as *mut c_void,
            scratch.conv_input.as_ptr(),
            qkv_dim * (pad + chunk_len) * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} rollback conv_input capture: {e}"))?;
    }
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .conv_input
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug conv_window D2H: {e}"))?;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let total_len = chunk_len + pad;
        let window_start = chunk_len - 1;
        trace.conv_window = Vec::with_capacity(qkv_dim * kern * elem_bytes);
        for ch in 0..qkv_dim {
            let start = (ch * total_len + window_start) * elem_bytes;
            let end = start + kern * elem_bytes;
            trace.conv_window.extend_from_slice(&bytes[start..end]);
        }
    }

    // 6. Conv1d + SiLU: [qkv_dim, pad+chunk] → [chunk, qkv_dim]
    let total_len = chunk_len + pad;
    prefill_ffi::linear_prefill_conv_pack(
        ordinal,
        ScalarType::BF16,
        1, // batch_size
        qkv_dim,
        total_len,
        chunk_len,
        kern,
        &scratch.conv_input,
        &lw.conv1d_w,
        &mut scratch.proj_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} conv: {e}"))?;
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .proj_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug post_conv D2H: {e}"))?;
        let row_bytes = qkv_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.post_conv = bytes[start..start + row_bytes].to_vec();
    }

    // Now (post-conv1d) update the inter-chunk and persistent conv tail
    // buffers from the new_tail we computed earlier. Order matters: this MUST
    // happen after `linear_prefill_conv_pack` reads from `chunk_conv_tail`,
    // otherwise the conv1d window would mix this chunk's QKV with itself.
    let total_tail_bytes = qkv_dim * pad * elem_bytes;
    if is_last_chunk {
        if let Some(ref mut conv_state) = state.layers[idx].conv_state {
            copy_d2d_batched(
                ordinal,
                conv_state.as_ptr() as *mut c_void,
                scratch.linear_new_tail.as_ptr(),
                total_tail_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} conv state writeback: {e}"))?;
        }
    }
    copy_d2d_batched(
        ordinal,
        chunk_conv_tail.as_ptr() as *mut c_void,
        scratch.linear_new_tail.as_ptr(),
        total_tail_bytes,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} chunk_conv_tail writeback: {e}"))?;

    // 7. Split conv output [S, qkv_dim] into Q [S, key_dim], K [S, key_dim], V [S, val_dim]
    //    Layout within qkv_dim: [Q(key_dim) | K(key_dim) | V(val_dim)]

    // 8. L2-normalize Q and K per head
    //    Q: treat [S, key_dim] as [S*nk, khd], normalize each row
    //    K: treat [S, key_dim] starting at offset key_dim as [S*nk, khd]
    //    The l2norm function normalizes each row independently.
    //
    //    However, the megakernel applies Q_norm = Q / ||Q|| * rsqrt(khd)
    //    and K_norm = K / ||K||. The l2norm kernel does x / ||x|| with eps.
    //    For Q, we need an extra * rsqrt(khd) scaling.
    //
    //    Strategy: normalize both Q and K via l2norm, then scale Q by rsqrt(khd).

    let use_fused_qkv_prepare = gpu_hal::current_backend() == Backend::Hip
        && !trace_linear_debug
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_QKV_PREP").is_none();
    if use_fused_qkv_prepare {
        let q_scale = 1.0 / (khd as f32).sqrt();
        prefill_ffi::split_norm_transpose_qkv_bf16(
            ordinal,
            chunk_len,
            nk,
            nv,
            khd,
            vhd,
            q_scale,
            1e-6,
            &scratch.proj_buf,
            &mut scratch.linear_q_trans,
            &mut scratch.linear_k_trans,
            &mut scratch.linear_v_trans,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused QKV prepare: {e}"))?;
    } else if gpu_hal::current_backend() == Backend::Hip {
        prefill_ffi::split_qkv_bf16_to_f32(
            ordinal,
            chunk_len,
            key_dim,
            val_dim,
            &scratch.proj_buf,
            &mut scratch.linear_q_f32,
            &mut scratch.linear_k_f32,
            &mut scratch.linear_v_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} QKV split+cast: {e}"))?;
    } else {
        // Split interleaved QKV [chunk, qkv_dim] -> Q, K, V.
        let mut q_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, key_dim])
            .map_err(|e| anyhow::anyhow!("q_linear alloc: {e}"))?;
        let mut k_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, key_dim])
            .map_err(|e| anyhow::anyhow!("k_linear alloc: {e}"))?;
        let mut v_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[chunk_len, val_dim])
            .map_err(|e| anyhow::anyhow!("v_linear alloc: {e}"))?;
        prefill_ffi::split_qkv(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            key_dim,
            val_dim,
            &scratch.proj_buf,
            &mut q_linear,
            &mut k_linear,
            &mut v_linear,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} QKV split: {e}"))?;

        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            chunk_len * key_dim,
            &q_linear,
            &mut scratch.linear_q_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            chunk_len * key_dim,
            &k_linear,
            &mut scratch.linear_k_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            chunk_len * val_dim,
            &v_linear,
            &mut scratch.linear_v_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} V cast: {e}"))?;
    }

    if !use_fused_qkv_prepare {
        prefill_ffi::l2norm(
            ordinal,
            ScalarType::F32,
            chunk_len * nk,
            khd,
            1e-6,
            &scratch.linear_q_f32,
            &mut scratch.linear_q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q l2norm: {e}"))?;

        let q_scale = 1.0 / (khd as f32).sqrt();
        prefill_ffi::mul_scalar(
            ordinal,
            ScalarType::F32,
            chunk_len * key_dim,
            q_scale,
            &scratch.linear_q_normed,
            &mut scratch.linear_q_scaled,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q scale: {e}"))?;

        prefill_ffi::l2norm(
            ordinal,
            ScalarType::F32,
            chunk_len * nk,
            khd,
            1e-6,
            &scratch.linear_k_f32,
            &mut scratch.linear_k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K l2norm: {e}"))?;
    }

    // 9. Compute beta and g on GPU
    //    beta[h, t] = sigmoid(B[t, h]) → [nv, chunk_len]
    //    g[h, t] = -softplus(A[t, h] + dt_bias[h]) * a_log_exp[h] → [nv, chunk_len]
    if let Some(ba) = ba_buf {
        prefill_ffi::compute_beta_g_ba_bf16(
            ordinal,
            chunk_len,
            nv,
            ba,
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut scratch.linear_beta,
            &mut scratch.linear_g,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused beta/g: {e}"))?;
    } else {
        let a_buf = a_buf.expect("separate A buffer initialized");
        let b_buf = b_buf.expect("separate B buffer initialized");
        let mut a_buf_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[chunk_len, nv])
            .map_err(|e| anyhow::anyhow!("a_buf_f32 alloc: {e}"))?;
        let mut b_buf_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[chunk_len, nv])
            .map_err(|e| anyhow::anyhow!("b_buf_f32 alloc: {e}"))?;
        let mut dt_bias_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv])
            .map_err(|e| anyhow::anyhow!("dt_bias_f32 alloc: {e}"))?;
        let mut a_log_exp_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv])
            .map_err(|e| anyhow::anyhow!("a_log_exp_f32 alloc: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            chunk_len * nv,
            a_buf,
            &mut a_buf_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} A cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            chunk_len * nv,
            b_buf,
            &mut b_buf_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} B cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            nv,
            &lw.dt_bias,
            &mut dt_bias_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} dt_bias cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            nv,
            &lw.a_log_exp,
            &mut a_log_exp_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} a_log_exp cast: {e}"))?;
        prefill_ffi::compute_beta_g(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nv,
            &b_buf_f32,
            &a_buf_f32,
            &dt_bias_f32,
            &a_log_exp_f32,
            &mut scratch.linear_beta,
            &mut scratch.linear_g,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} beta/g: {e}"))?;
    }

    // 10. Transpose Q [S, nk, khd] → [nk, S, khd] and K, V similarly
    //     If nk != nv, repeat Q and K heads to match nv (like GQA head expansion)
    let head_repeat = nv / nk;

    if !use_fused_qkv_prepare {
        if head_repeat == 1 {
            prefill_ffi::transpose_shd_hsd(
                ordinal,
                ScalarType::F32,
                chunk_len,
                nk,
                khd,
                &scratch.linear_q_scaled,
                &mut scratch.linear_q_trans,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} Q linear transpose: {e}"))?;
        } else {
            prefill_ffi::repeat_interleave_transpose_hsd(
                ordinal,
                ScalarType::F32,
                chunk_len,
                nk,
                khd,
                head_repeat,
                &scratch.linear_q_scaled,
                &mut scratch.linear_q_trans,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} Q repeat+transpose: {e}"))?;
        }

        if head_repeat == 1 {
            prefill_ffi::transpose_shd_hsd(
                ordinal,
                ScalarType::F32,
                chunk_len,
                nk,
                khd,
                &scratch.linear_k_normed,
                &mut scratch.linear_k_trans,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} K linear transpose: {e}"))?;
        } else {
            prefill_ffi::repeat_interleave_transpose_hsd(
                ordinal,
                ScalarType::F32,
                chunk_len,
                nk,
                khd,
                head_repeat,
                &scratch.linear_k_normed,
                &mut scratch.linear_k_trans,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} K repeat+transpose: {e}"))?;
        }

        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nv,
            vhd,
            &scratch.linear_v_f32,
            &mut scratch.linear_v_trans,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} V linear transpose: {e}"))?;
    }

    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let q_scaled_bytes = scratch
            .linear_q_scaled
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug q_scaled D2H: {e}"))?;
        let k_normed_bytes = scratch
            .linear_k_normed
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug k_normed D2H: {e}"))?;
        let v_linear_bytes = scratch
            .linear_v_f32
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug v_linear D2H: {e}"))?;
        let beta_bytes = scratch
            .linear_beta
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug beta D2H: {e}"))?;
        let g_bytes = scratch
            .linear_g
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug g D2H: {e}"))?;
        let q_scaled_f32: Vec<f32> = q_scaled_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let k_normed_f32: Vec<f32> = k_normed_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v_linear_f32_host: Vec<f32> = v_linear_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let beta_f32: Vec<f32> = beta_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let g_f32: Vec<f32> = g_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let packed_width = 2 * khd + vhd + 2;
        let mut packed_equiv = vec![0f32; nv * packed_width];
        let last_t = chunk_len - 1;
        for v_head in 0..nv {
            let k_head = v_head % nk;
            let out_base = v_head * packed_width;
            let q_base = last_t * key_dim + k_head * khd;
            let k_base = (last_t * nk + k_head) * khd;
            let v_base = last_t * val_dim + v_head * vhd;
            for i in 0..khd {
                packed_equiv[out_base + i] = q_scaled_f32[q_base + i];
                packed_equiv[out_base + khd + i] = k_normed_f32[k_base + i];
            }
            for i in 0..vhd {
                packed_equiv[out_base + 2 * khd + i] = v_linear_f32_host[v_base + i];
            }
            packed_equiv[out_base + 2 * khd + vhd] = beta_f32[v_head * chunk_len + last_t];
            packed_equiv[out_base + 2 * khd + vhd + 1] = g_f32[v_head * chunk_len + last_t].exp();
        }
        trace.packed = packed_equiv
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
    }

    // 11. Delta recurrent prefill
    // The recurrent state is the F32 source of truth across chunk boundaries
    // (and across incremental decode steps). The delta kernel only reads the
    // initial state, so use the persistent state buffer directly when present
    // and write the updated state back after the kernel returns.
    let state_elems = nv * khd * vhd;
    let elem_bytes_f32 = ScalarType::F32.size_in_bytes();
    let out_rows = chunk_len + khd;
    let mut recurrent_attn_direct = false;
    if !trace_linear_debug {
        if let Some(capture) = append_capture.as_mut() {
            if capture.recurrent_trace.dtype() == ScalarType::U8 {
                if let Some(rec_state) = state.layers[idx].recurrent_state.as_mut() {
                    recurrent_attn_direct =
                        prefill_ffi::delta_recurrent_prefill_capture_q8_trace_attn(
                            ordinal,
                            ScalarType::F32,
                            nv, // batch_heads
                            chunk_len,
                            khd,
                            vhd,
                            rec_state,
                            &scratch.linear_q_trans,
                            &scratch.linear_k_trans,
                            &scratch.linear_v_trans,
                            &scratch.linear_beta,
                            &scratch.linear_g,
                            &mut scratch.linear_attn_output,
                            &mut capture.recurrent_trace,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "layer {idx} delta recurrent Q8 trace direct attention: {e}"
                            )
                        })?;
                }
            }
        }
    }

    if !recurrent_attn_direct {
        let zero_recurrent;
        let recurrent_initial = if let Some(rec_state) = state.layers[idx].recurrent_state.as_ref()
        {
            rec_state
        } else {
            zero_recurrent = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, khd, vhd])
                .map_err(|e| anyhow::anyhow!("zero recurrent alloc: {e}"))?;
            &zero_recurrent
        };

        if let Some(capture) = append_capture.as_mut() {
            if capture.recurrent_trace.dtype() == ScalarType::U8 {
                prefill_ffi::delta_recurrent_prefill_capture_q8_trace(
                    ordinal,
                    ScalarType::F32,
                    nv, // batch_heads
                    chunk_len,
                    khd,
                    vhd,
                    recurrent_initial,
                    &scratch.linear_q_trans,
                    &scratch.linear_k_trans,
                    &scratch.linear_v_trans,
                    &scratch.linear_beta,
                    &scratch.linear_g,
                    &mut scratch.linear_delta_out,
                    &mut capture.recurrent_trace,
                )
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} delta recurrent Q8 trace capture: {e}")
                })?;
            } else if capture.recurrent_trace.dtype() == ScalarType::BF16 {
                prefill_ffi::delta_recurrent_prefill_capture_bf16_trace(
                    ordinal,
                    ScalarType::F32,
                    nv, // batch_heads
                    chunk_len,
                    khd,
                    vhd,
                    recurrent_initial,
                    &scratch.linear_q_trans,
                    &scratch.linear_k_trans,
                    &scratch.linear_v_trans,
                    &scratch.linear_beta,
                    &scratch.linear_g,
                    &mut scratch.linear_delta_out,
                    &mut capture.recurrent_trace,
                )
                .map_err(|e| {
                    anyhow::anyhow!("layer {idx} delta recurrent BF16 trace capture: {e}")
                })?;
            } else {
                prefill_ffi::delta_recurrent_prefill_capture(
                    ordinal,
                    ScalarType::F32,
                    nv, // batch_heads
                    chunk_len,
                    khd,
                    vhd,
                    recurrent_initial,
                    &scratch.linear_q_trans,
                    &scratch.linear_k_trans,
                    &scratch.linear_v_trans,
                    &scratch.linear_beta,
                    &scratch.linear_g,
                    &mut scratch.linear_delta_out,
                    &mut capture.recurrent_trace,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} delta recurrent capture: {e}"))?;
            }
        } else {
            prefill_ffi::delta_recurrent_prefill(
                ordinal,
                ScalarType::F32,
                nv, // batch_heads
                chunk_len,
                khd,
                vhd,
                recurrent_initial,
                &scratch.linear_q_trans,
                &scratch.linear_k_trans,
                &scratch.linear_v_trans,
                &scratch.linear_beta,
                &scratch.linear_g,
                &mut scratch.linear_delta_out,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} delta recurrent: {e}"))?;
        }
    }

    // 12. Extract recurrent state from delta_out and write F32 directly back to
    //     state.layers[idx].recurrent_state. Always update — between chunks of
    //     a multi-chunk prefill the previous behavior gated this on
    //     `is_last_chunk` and used a BF16 sidecar (`chunk_recurrent`) for
    //     handoff, which silently quantized the recurrent state at every
    //     boundary. Updating in F32 every chunk preserves precision and makes
    //     incremental decode (chunk_len=1 across calls) produce identical
    //     state to single-chunk prefill of the same token history.
    let mut state_bytes_debug: Option<Vec<u8>> = None;
    let mut attn_output_f32_debug: Option<Vec<u8>> = None;

    if !trace_linear_debug {
        if recurrent_attn_direct {
            // Direct recurrent capture already wrote BF16 attention output and
            // updated the persistent F32 recurrent state.
        } else if let Some(ref mut rec_state) = state.layers[idx].recurrent_state {
            prefill_ffi::dflash_extract_recurrent_attn(
                ordinal,
                nv,
                chunk_len,
                khd,
                vhd,
                &scratch.linear_delta_out,
                rec_state,
                &mut scratch.linear_attn_output,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} recurrent/attn extract: {e}"))?;
        } else {
            return Err(anyhow::anyhow!(
                "layer {idx} missing recurrent state for linear attention"
            ));
        }
    } else {
        let state_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, khd, vhd])
            .map_err(|e| anyhow::anyhow!("state_f32 alloc: {e}"))?;

        let state_bytes_per_head = khd * vhd * elem_bytes_f32;
        let out_stride = out_rows * vhd * elem_bytes_f32;
        let attn_offset = chunk_len * vhd * elem_bytes_f32;
        for h in 0..nv {
            let src_off = h * out_stride + attn_offset;
            let dst_off = h * state_bytes_per_head;
            copy_d2d_batched(
                ordinal,
                state_f32.offset_ptr(dst_off) as *mut c_void,
                scratch.linear_delta_out.offset_ptr(src_off),
                state_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state extract h={h}: {e}"))?;
        }

        if let Some(ref mut rec_state) = state.layers[idx].recurrent_state {
            copy_d2d_batched(
                ordinal,
                rec_state.as_ptr() as *mut c_void,
                state_f32.as_ptr(),
                state_elems * elem_bytes_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state writeback: {e}"))?;
        }
        state_bytes_debug = Some(
            state_f32
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} debug state_f32 D2H: {e}"))?,
        );

        // 13. Extract attention output: [nv, chunk_len, vhd] from delta_out.
        let attn_output_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv, chunk_len, vhd])
            .map_err(|e| anyhow::anyhow!("attn_output_f32 alloc: {e}"))?;
        let attn_bytes_per_head = chunk_len * vhd * ScalarType::F32.size_in_bytes();
        let out_stride = out_rows * vhd * ScalarType::F32.size_in_bytes();
        for h in 0..nv {
            let src_off = h * out_stride;
            let dst_off = h * attn_bytes_per_head;
            copy_d2d_batched(
                ordinal,
                attn_output_f32.offset_ptr(dst_off) as *mut c_void,
                scratch.linear_delta_out.offset_ptr(src_off),
                attn_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} attn output extract h={h}: {e}"))?;
        }
        prefill_ffi::cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            nv * chunk_len * vhd,
            &attn_output_f32,
            &mut scratch.linear_attn_output,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn output cast: {e}"))?;
        attn_output_f32_debug = Some(
            attn_output_f32
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} debug attn_output_f32 D2H: {e}"))?,
        );
    }
    let _ = is_last_chunk; // recurrent state is now always written; flag still gates conv_state above.
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let attn_out_bytes = attn_output_f32_debug
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx} missing debug attn_output_f32 bytes"))?;
        let mut rec_apply_equiv =
            Vec::with_capacity((val_dim + nv * khd * vhd) * ScalarType::F32.size_in_bytes());
        let elem_bytes = ScalarType::F32.size_in_bytes();
        let head_stride = chunk_len * vhd * elem_bytes;
        let tok_off = (chunk_len - 1) * vhd * elem_bytes;
        let row_bytes = vhd * elem_bytes;
        for h in 0..nv {
            let start = h * head_stride + tok_off;
            rec_apply_equiv.extend_from_slice(&attn_out_bytes[start..start + row_bytes]);
        }
        let state_bytes = state_bytes_debug
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx} missing debug state_f32 bytes"))?;
        rec_apply_equiv.extend_from_slice(state_bytes);
        trace.rec_apply = rec_apply_equiv;
    }
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .linear_attn_output
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug attn D2H: {e}"))?;
        let mut last = Vec::with_capacity(nv * vhd * ScalarType::BF16.size_in_bytes());
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let head_stride = chunk_len * vhd * elem_bytes;
        let tok_off = (chunk_len - 1) * vhd * elem_bytes;
        let row_bytes = vhd * elem_bytes;
        for h in 0..nv {
            let start = h * head_stride + tok_off;
            last.extend_from_slice(&bytes[start..start + row_bytes]);
        }
        trace.attn = last;
    }

    // 14. Gated RMSNorm: out = rms_norm(attn_output) * norm_w * silu(Z)
    //     attn_output is [nv, S, vhd]; Z (proj_buf2) is [S, val_dim] = [S, nv*vhd]
    //     Need Z in [nv, S, vhd] layout
    let use_fused_gated_epilogue = gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_GATED_EPILOGUE").is_none();
    if use_fused_gated_epilogue {
        prefill_ffi::rms_norm_gated_sfirst_bf16(
            ordinal,
            chunk_len,
            nv,
            vhd,
            config.rms_norm_eps as f32,
            &scratch.linear_attn_output,
            &scratch.proj_buf2,
            &lw.norm_w_bf16,
            &mut scratch.linear_gated_s_first,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} fused gated epilogue: {e}"))?;
    } else {
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            chunk_len,
            nv,
            vhd,
            &scratch.proj_buf2,
            &mut scratch.linear_z_trans,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Z transpose: {e}"))?;

        prefill_ffi::rms_norm_gated(
            ordinal,
            ScalarType::BF16,
            nv * chunk_len,
            vhd,
            config.rms_norm_eps as f32,
            &scratch.linear_attn_output,
            &scratch.linear_z_trans,
            &lw.norm_w_bf16,
            &mut scratch.linear_gated_out,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;

        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            nv,
            chunk_len,
            vhd,
            &scratch.linear_gated_out,
            &mut scratch.linear_gated_s_first,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gated transpose: {e}"))?;
    }
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .linear_gated_s_first
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug gated D2H: {e}"))?;
        let row_bytes = val_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.gated = bytes[start..start + row_bytes].to_vec();
    }

    // 16-17. O projection + residual:
    // [S, val_dim] × out_proj_w [hidden, val_dim]^T → hidden += [S, hidden].
    let fused_residual = !trace_linear_debug
        && !scratch.has_f32_activation_carry()
        && matmul_proj_residual_add_inplace(
            ordinal,
            1,
            chunk_len,
            hidden_dim,
            val_dim,
            &scratch.linear_gated_s_first,
            &lw.out_proj_w,
            lw.out_proj_scale.as_ref(),
            lw.out_proj_int8_scale.as_ref(),
            &mut scratch.hidden,
            lw.out_proj_int4_scale.as_ref(),
            lw.out_proj_int4_zero.as_ref(),
            lw.out_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    if !fused_residual {
        matmul_proj(
            ordinal,
            1,
            chunk_len,
            hidden_dim,
            val_dim,
            &scratch.linear_gated_s_first,
            &lw.out_proj_w,
            lw.out_proj_scale.as_ref(),
            lw.out_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            lw.out_proj_int4_scale.as_ref(),
            lw.out_proj_int4_zero.as_ref(),
            lw.out_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    }
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let bytes = scratch
            .proj_buf2
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug proj_out D2H: {e}"))?;
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.proj_out = bytes[start..start + row_bytes].to_vec();
    }

    if !fused_residual {
        scratch.residual_add_from_source(
            ordinal,
            chunk_len * hidden_dim,
            ResidualSource::ProjBuf2,
            "linear attn residual",
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn delta_recurrent_tree_prefill_capture_with_trace(
    ordinal: usize,
    idx: usize,
    nv: usize,
    tree_len: usize,
    khd: usize,
    vhd: usize,
    recurrent_initial: &GpuBuffer,
    linear_q_trans: &GpuBuffer,
    linear_k_trans: &GpuBuffer,
    linear_v_trans: &GpuBuffer,
    linear_beta: &GpuBuffer,
    linear_g: &GpuBuffer,
    parent_ids_gpu: &GpuBuffer,
    linear_delta_out: &mut GpuBuffer,
    recurrent_trace: &mut GpuBuffer,
    linear_attn_output: Option<&mut GpuBuffer>,
) -> Result<bool> {
    if recurrent_trace.dtype() == ScalarType::U8 {
        if gpu_hal::current_backend() == Backend::Hip
            && env::var_os("SUPERSONIC_DFLASH_DISABLE_TREE_DIRECT_ATTENTION").is_none()
        {
            if let Some(attn_output) = linear_attn_output {
                prefill_ffi::delta_recurrent_tree_prefill_capture_q8_trace_attn(
                    ordinal,
                    ScalarType::F32,
                    nv,
                    tree_len,
                    khd,
                    vhd,
                    recurrent_initial,
                    linear_q_trans,
                    linear_k_trans,
                    linear_v_trans,
                    linear_beta,
                    linear_g,
                    parent_ids_gpu,
                    attn_output,
                    recurrent_trace,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "tree layer {idx} delta recurrent Q8 trace direct attention: {e}"
                    )
                })?;
                return Ok(true);
            }
        }

        prefill_ffi::delta_recurrent_tree_prefill_capture_q8_trace(
            ordinal,
            ScalarType::F32,
            nv,
            tree_len,
            khd,
            vhd,
            recurrent_initial,
            linear_q_trans,
            linear_k_trans,
            linear_v_trans,
            linear_beta,
            linear_g,
            parent_ids_gpu,
            linear_delta_out,
            recurrent_trace,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} delta recurrent Q8 trace capture: {e}"))?;
    } else if recurrent_trace.dtype() == ScalarType::BF16 {
        prefill_ffi::delta_recurrent_tree_prefill_capture_bf16_trace(
            ordinal,
            ScalarType::F32,
            nv,
            tree_len,
            khd,
            vhd,
            recurrent_initial,
            linear_q_trans,
            linear_k_trans,
            linear_v_trans,
            linear_beta,
            linear_g,
            parent_ids_gpu,
            linear_delta_out,
            recurrent_trace,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} delta recurrent BF16 trace capture: {e}"))?;
    } else {
        prefill_ffi::delta_recurrent_tree_prefill_capture(
            ordinal,
            ScalarType::F32,
            nv,
            tree_len,
            khd,
            vhd,
            recurrent_initial,
            linear_q_trans,
            linear_k_trans,
            linear_v_trans,
            linear_beta,
            linear_g,
            parent_ids_gpu,
            linear_delta_out,
            recurrent_trace,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} delta recurrent capture: {e}"))?;
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn prefill_tree_linear_attention_layer(
    weights: &Qwen35Weights,
    state: &ModelState,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    tree_len: usize,
    prefix_len: usize,
    ordinal: usize,
    parent_ids_gpu: &GpuBuffer,
    conv_source_cols_gpu: &GpuBuffer,
    conv_source_cols_stride: usize,
    mut capture_slot: Option<&mut Option<PrefillTreeLayerRollback>>,
) -> Result<()> {
    let lw = weights.layers[idx]
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear attention weights"))?;

    let hidden_dim = config.hidden_size;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let kern = config.linear_conv_kernel_dim;
    let key_dim = nk * khd;
    let val_dim = nv * vhd;
    let qkv_dim = key_dim * 2 + val_dim;
    let z_dim = val_dim;

    let fused_qkvz_enabled = if env::var_os("SUPERSONIC_DISABLE_FUSED_QKVZ").is_some() {
        false
    } else {
        env::var_os("SUPERSONIC_ENABLE_FUSED_QKVZ").is_some()
            || (config.hidden_size == 5120 && config.num_hidden_layers == 64)
    };
    let use_fused_qkvz =
        fused_qkvz_enabled && scratch.normed.backend() == Backend::Hip && lw.qkvz_proj_w.is_some();

    if use_fused_qkvz {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            qkv_dim + z_dim,
            hidden_dim,
            &scratch.normed,
            lw.qkvz_proj_w.as_ref().expect("checked fused QKVZ weight"),
            None,
            None,
            weights.fp8_block_size,
            &mut scratch.mlp_buf,
            None,
            None,
            None,
            0,
        )?;
        prefill_ffi::split_qkvz_bf16(
            ordinal,
            tree_len,
            qkv_dim,
            z_dim,
            &scratch.mlp_buf,
            &mut scratch.proj_buf,
            &mut scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused QKVZ split: {e}"))?;
    } else {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            qkv_dim,
            hidden_dim,
            &scratch.normed,
            &lw.qkv_proj_w,
            lw.qkv_proj_scale.as_ref(),
            lw.qkv_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf,
            lw.qkv_proj_int4_scale.as_ref(),
            lw.qkv_proj_int4_zero.as_ref(),
            lw.qkv_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;

        matmul_proj(
            ordinal,
            1,
            tree_len,
            z_dim,
            hidden_dim,
            &scratch.normed,
            &lw.z_proj_w,
            lw.z_proj_scale.as_ref(),
            lw.z_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            lw.z_proj_int4_scale.as_ref(),
            lw.z_proj_int4_zero.as_ref(),
            lw.z_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    }

    let use_fused_ba =
        lw.ba_proj_w.is_some() && scratch.normed.backend() != gpu_hal::Backend::Metal;
    // Experimental scalar fused path. Current profiling shows the generic BF16 matmul plus
    // beta/g epilogue is faster, so keep this opt-in until it gets an MFMA implementation.
    let use_direct_ba = use_fused_ba
        && gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_ENABLE_FUSED_BA_DIRECT").is_some()
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_BA_DIRECT").is_none();
    if use_direct_ba {
        prefill_ffi::project_ba_compute_beta_g_bf16(
            ordinal,
            tree_len,
            hidden_dim,
            nv,
            &scratch.normed,
            lw.ba_proj_w.as_ref().expect("checked fused BA weight"),
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut scratch.linear_beta,
            &mut scratch.linear_g,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} direct BA beta/g: {e}"))?;
    } else if use_fused_ba {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            2 * nv,
            hidden_dim,
            &scratch.normed,
            lw.ba_proj_w.as_ref().expect("checked fused BA weight"),
            None,
            None,
            weights.fp8_block_size,
            &mut scratch.linear_ba_buf,
            None,
            None,
            None,
            0,
        )?;
    } else {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            nv,
            hidden_dim,
            &scratch.normed,
            &lw.b_proj_w,
            lw.b_proj_scale.as_ref(),
            lw.b_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.linear_b_buf,
            None,
            None,
            None,
            0,
        )?;

        matmul_proj(
            ordinal,
            1,
            tree_len,
            nv,
            hidden_dim,
            &scratch.normed,
            &lw.a_proj_w,
            lw.a_proj_scale.as_ref(),
            lw.a_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.linear_a_buf,
            None,
            None,
            None,
            0,
        )?;
    }

    let pad = kern - 1;
    let use_fused_tree_conv_prep = gpu_hal::current_backend() == Backend::Hip
        && prefix_len > 0
        && tree_len >= pad
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_CONV_PREP").is_none();
    if use_fused_tree_conv_prep {
        let conv_state = state.layers[idx].conv_state.as_ref().ok_or_else(|| {
            anyhow::anyhow!("tree layer {idx} missing conv_state for prefix_len={prefix_len}")
        })?;
        prefill_ffi::prepare_conv_input_tail(
            ordinal,
            ScalarType::BF16,
            tree_len,
            qkv_dim,
            pad,
            &scratch.proj_buf,
            conv_state,
            &mut scratch.conv_input,
            &mut scratch.linear_new_tail,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused conv prepare: {e}"))?;
    } else {
        prefill_ffi::transpose_pad_conv(
            ordinal,
            ScalarType::BF16,
            tree_len,
            qkv_dim,
            pad,
            &scratch.proj_buf,
            &mut scratch.conv_input,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} conv transpose+pad: {e}"))?;

        if prefix_len > 0 {
            let conv_state = state.layers[idx].conv_state.as_ref().ok_or_else(|| {
                anyhow::anyhow!("tree layer {idx} missing conv_state for prefix_len={prefix_len}")
            })?;
            prefill_ffi::fill_conv_tail(
                ordinal,
                ScalarType::BF16,
                qkv_dim,
                pad,
                pad + tree_len,
                conv_state,
                &mut scratch.conv_input,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} conv pad fill: {e}"))?;
        }
    }
    let recurrent_trace_dtype = dflash_rollback_trace_dtype();
    if let Some(slot) = capture_slot.as_deref_mut() {
        let conv_bytes = qkv_dim * (pad + tree_len) * ScalarType::BF16.size_in_bytes();
        let trace_bytes = if recurrent_trace_dtype == ScalarType::U8 {
            dflash_q8_trace_bytes(nv, tree_len, khd, vhd)
        } else {
            nv * tree_len * khd * vhd * recurrent_trace_dtype.size_in_bytes()
        };
        let needs_alloc = !matches!(
            slot.as_ref(),
            Some(PrefillTreeLayerRollback::Linear {
                conv_input,
                recurrent_trace,
            }) if conv_input.device_ordinal() == ordinal
                && recurrent_trace.device_ordinal() == ordinal
                && conv_input.dtype() == ScalarType::BF16
                && recurrent_trace.dtype() == recurrent_trace_dtype
                && conv_input.len_bytes() >= conv_bytes
                && recurrent_trace.len_bytes() >= trace_bytes
        );
        if needs_alloc {
            let conv_input =
                GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, pad + tree_len]).map_err(
                    |e| anyhow::anyhow!("tree layer {idx} rollback conv_input alloc: {e}"),
                )?;
            let recurrent_trace = if recurrent_trace_dtype == ScalarType::U8 {
                GpuBuffer::alloc(
                    ordinal,
                    recurrent_trace_dtype,
                    &[dflash_q8_trace_bytes(nv, tree_len, khd, vhd)],
                )
            } else {
                GpuBuffer::alloc(ordinal, recurrent_trace_dtype, &[nv, tree_len, khd, vhd])
            }
            .map_err(|e| anyhow::anyhow!("tree recurrent trace alloc: {e}"))?;
            *slot = Some(PrefillTreeLayerRollback::Linear {
                conv_input,
                recurrent_trace,
            });
        }
        let Some(PrefillTreeLayerRollback::Linear { conv_input, .. }) = slot.as_mut() else {
            unreachable!("tree linear rollback slot was just initialized");
        };
        copy_d2d_batched(
            ordinal,
            conv_input.as_mut_ptr(),
            scratch.conv_input.as_ptr(),
            conv_bytes,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} rollback conv_input capture: {e}"))?;
    }

    let use_indexed_tree_conv = gpu_hal::current_backend() == Backend::Hip
        && conv_source_cols_stride >= kern
        && conv_source_cols_gpu.elem_count() >= tree_len * conv_source_cols_stride
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_TREE_CONV_SOURCE_MAP").is_none();
    if use_indexed_tree_conv {
        prefill_ffi::linear_tree_conv_pack_indexed(
            ordinal,
            ScalarType::BF16,
            1,
            qkv_dim,
            pad + tree_len,
            tree_len,
            kern,
            conv_source_cols_stride,
            &scratch.conv_input,
            &lw.conv1d_w,
            conv_source_cols_gpu,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} indexed conv: {e}"))?;
    } else {
        prefill_ffi::linear_tree_conv_pack(
            ordinal,
            ScalarType::BF16,
            1,
            qkv_dim,
            pad + tree_len,
            tree_len,
            kern,
            &scratch.conv_input,
            &lw.conv1d_w,
            parent_ids_gpu,
            &mut scratch.proj_buf,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} conv: {e}"))?;
    }

    let use_fused_qkv_prepare = gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_QKV_PREP").is_none();
    if use_fused_qkv_prepare {
        let q_scale = 1.0 / (khd as f32).sqrt();
        prefill_ffi::split_norm_transpose_qkv_bf16(
            ordinal,
            tree_len,
            nk,
            nv,
            khd,
            vhd,
            q_scale,
            1e-6,
            &scratch.proj_buf,
            &mut scratch.linear_q_trans,
            &mut scratch.linear_k_trans,
            &mut scratch.linear_v_trans,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused QKV prepare: {e}"))?;
    } else if gpu_hal::current_backend() == Backend::Hip {
        prefill_ffi::split_qkv_bf16_to_f32(
            ordinal,
            tree_len,
            key_dim,
            val_dim,
            &scratch.proj_buf,
            &mut scratch.linear_q_f32,
            &mut scratch.linear_k_f32,
            &mut scratch.linear_v_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} QKV split+cast: {e}"))?;
    } else {
        let mut q_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, key_dim])
            .map_err(|e| anyhow::anyhow!("tree q_linear alloc: {e}"))?;
        let mut k_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, key_dim])
            .map_err(|e| anyhow::anyhow!("tree k_linear alloc: {e}"))?;
        let mut v_linear = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[tree_len, val_dim])
            .map_err(|e| anyhow::anyhow!("tree v_linear alloc: {e}"))?;
        prefill_ffi::split_qkv(
            ordinal,
            ScalarType::BF16,
            tree_len,
            key_dim,
            val_dim,
            &scratch.proj_buf,
            &mut q_linear,
            &mut k_linear,
            &mut v_linear,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} QKV split: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            tree_len * key_dim,
            &q_linear,
            &mut scratch.linear_q_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Q cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            tree_len * key_dim,
            &k_linear,
            &mut scratch.linear_k_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} K cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            tree_len * val_dim,
            &v_linear,
            &mut scratch.linear_v_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} V cast: {e}"))?;
    }

    if !use_fused_qkv_prepare {
        prefill_ffi::l2norm(
            ordinal,
            ScalarType::F32,
            tree_len * nk,
            khd,
            1e-6,
            &scratch.linear_q_f32,
            &mut scratch.linear_q_normed,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Q l2norm: {e}"))?;
        let q_scale = 1.0 / (khd as f32).sqrt();
        prefill_ffi::mul_scalar(
            ordinal,
            ScalarType::F32,
            tree_len * key_dim,
            q_scale,
            &scratch.linear_q_normed,
            &mut scratch.linear_q_scaled,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Q scale: {e}"))?;

        prefill_ffi::l2norm(
            ordinal,
            ScalarType::F32,
            tree_len * nk,
            khd,
            1e-6,
            &scratch.linear_k_f32,
            &mut scratch.linear_k_normed,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} K l2norm: {e}"))?;
    }

    if use_direct_ba {
        // beta/g were produced directly from the fused BA projection above.
    } else if use_fused_ba {
        prefill_ffi::compute_beta_g_ba_bf16(
            ordinal,
            tree_len,
            nv,
            &scratch.linear_ba_buf,
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut scratch.linear_beta,
            &mut scratch.linear_g,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused beta/g: {e}"))?;
    } else {
        let mut a_buf_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[tree_len, nv])
            .map_err(|e| anyhow::anyhow!("tree a_buf_f32 alloc: {e}"))?;
        let mut b_buf_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[tree_len, nv])
            .map_err(|e| anyhow::anyhow!("tree b_buf_f32 alloc: {e}"))?;
        let mut dt_bias_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv])
            .map_err(|e| anyhow::anyhow!("tree dt_bias_f32 alloc: {e}"))?;
        let mut a_log_exp_f32 = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nv])
            .map_err(|e| anyhow::anyhow!("tree a_log_exp_f32 alloc: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            tree_len * nv,
            &scratch.linear_a_buf,
            &mut a_buf_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} A cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            tree_len * nv,
            &scratch.linear_b_buf,
            &mut b_buf_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} B cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            nv,
            &lw.dt_bias,
            &mut dt_bias_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} dt_bias cast: {e}"))?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            nv,
            &lw.a_log_exp,
            &mut a_log_exp_f32,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} a_log_exp cast: {e}"))?;
        prefill_ffi::compute_beta_g(
            ordinal,
            ScalarType::F32,
            tree_len,
            nv,
            &b_buf_f32,
            &a_buf_f32,
            &dt_bias_f32,
            &a_log_exp_f32,
            &mut scratch.linear_beta,
            &mut scratch.linear_g,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} beta/g: {e}"))?;
    }

    if !use_fused_qkv_prepare {
        let head_repeat = nv / nk;
        if head_repeat == 1 {
            prefill_ffi::transpose_shd_hsd(
                ordinal,
                ScalarType::F32,
                tree_len,
                nk,
                khd,
                &scratch.linear_q_scaled,
                &mut scratch.linear_q_trans,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} Q linear transpose: {e}"))?;
        } else {
            prefill_ffi::repeat_interleave_transpose_hsd(
                ordinal,
                ScalarType::F32,
                tree_len,
                nk,
                khd,
                head_repeat,
                &scratch.linear_q_scaled,
                &mut scratch.linear_q_trans,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} Q repeat+transpose: {e}"))?;
        };
        if head_repeat == 1 {
            prefill_ffi::transpose_shd_hsd(
                ordinal,
                ScalarType::F32,
                tree_len,
                nk,
                khd,
                &scratch.linear_k_normed,
                &mut scratch.linear_k_trans,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} K linear transpose: {e}"))?;
        } else {
            prefill_ffi::repeat_interleave_transpose_hsd(
                ordinal,
                ScalarType::F32,
                tree_len,
                nk,
                khd,
                head_repeat,
                &scratch.linear_k_normed,
                &mut scratch.linear_k_trans,
            )
            .map_err(|e| anyhow::anyhow!("tree layer {idx} K repeat+transpose: {e}"))?;
        };
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            tree_len,
            nv,
            vhd,
            &scratch.linear_v_f32,
            &mut scratch.linear_v_trans,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} V linear transpose: {e}"))?;
    }

    let zero_recurrent;
    let recurrent_initial = if let Some(rec_state) = state.layers[idx].recurrent_state.as_ref() {
        rec_state
    } else {
        zero_recurrent = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, khd, vhd])
            .map_err(|e| anyhow::anyhow!("tree zero recurrent alloc: {e}"))?;
        &zero_recurrent
    };

    let direct_attn = if let Some(slot) = capture_slot.as_deref_mut() {
        let Some(PrefillTreeLayerRollback::Linear {
            recurrent_trace, ..
        }) = slot.as_mut()
        else {
            unreachable!("tree linear rollback slot was initialized before recurrent capture");
        };
        delta_recurrent_tree_prefill_capture_with_trace(
            ordinal,
            idx,
            nv,
            tree_len,
            khd,
            vhd,
            recurrent_initial,
            &scratch.linear_q_trans,
            &scratch.linear_k_trans,
            &scratch.linear_v_trans,
            &scratch.linear_beta,
            &scratch.linear_g,
            parent_ids_gpu,
            &mut scratch.linear_delta_out,
            recurrent_trace,
            Some(&mut scratch.linear_attn_output),
        )?
    } else {
        let mut recurrent_trace = if recurrent_trace_dtype == ScalarType::U8 {
            GpuBuffer::alloc(
                ordinal,
                recurrent_trace_dtype,
                &[dflash_q8_trace_bytes(nv, tree_len, khd, vhd)],
            )
        } else {
            GpuBuffer::alloc(ordinal, recurrent_trace_dtype, &[nv, tree_len, khd, vhd])
        }
        .map_err(|e| anyhow::anyhow!("tree recurrent trace alloc: {e}"))?;
        delta_recurrent_tree_prefill_capture_with_trace(
            ordinal,
            idx,
            nv,
            tree_len,
            khd,
            vhd,
            recurrent_initial,
            &scratch.linear_q_trans,
            &scratch.linear_k_trans,
            &scratch.linear_v_trans,
            &scratch.linear_beta,
            &scratch.linear_g,
            parent_ids_gpu,
            &mut scratch.linear_delta_out,
            &mut recurrent_trace,
            Some(&mut scratch.linear_attn_output),
        )?
    };

    if !direct_attn {
        prefill_ffi::dflash_extract_recurrent_attn(
            ordinal,
            nv,
            tree_len,
            khd,
            vhd,
            &scratch.linear_delta_out,
            &mut scratch.linear_dummy_state,
            &mut scratch.linear_attn_output,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} recurrent/attn extract: {e}"))?;
    }

    let use_fused_gated_epilogue = gpu_hal::current_backend() == Backend::Hip
        && std::env::var_os("SUPERSONIC_DFLASH_DISABLE_FUSED_GATED_EPILOGUE").is_none();
    if use_fused_gated_epilogue {
        prefill_ffi::rms_norm_gated_sfirst_bf16(
            ordinal,
            tree_len,
            nv,
            vhd,
            config.rms_norm_eps as f32,
            &scratch.linear_attn_output,
            &scratch.proj_buf2,
            &lw.norm_w_bf16,
            &mut scratch.linear_gated_s_first,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} fused gated epilogue: {e}"))?;
    } else {
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            tree_len,
            nv,
            vhd,
            &scratch.proj_buf2,
            &mut scratch.linear_z_trans,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} Z transpose: {e}"))?;

        prefill_ffi::rms_norm_gated(
            ordinal,
            ScalarType::BF16,
            nv * tree_len,
            vhd,
            config.rms_norm_eps as f32,
            &scratch.linear_attn_output,
            &scratch.linear_z_trans,
            &lw.norm_w_bf16,
            &mut scratch.linear_gated_out,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} gated norm: {e}"))?;

        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::BF16,
            nv,
            tree_len,
            vhd,
            &scratch.linear_gated_out,
            &mut scratch.linear_gated_s_first,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} gated transpose: {e}"))?;
    }

    let fused_residual = !scratch.has_f32_activation_carry()
        && matmul_proj_residual_add_inplace(
            ordinal,
            1,
            tree_len,
            hidden_dim,
            val_dim,
            &scratch.linear_gated_s_first,
            &lw.out_proj_w,
            lw.out_proj_scale.as_ref(),
            lw.out_proj_int8_scale.as_ref(),
            &mut scratch.hidden,
            lw.out_proj_int4_scale.as_ref(),
            lw.out_proj_int4_zero.as_ref(),
            lw.out_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;
    if !fused_residual {
        matmul_proj(
            ordinal,
            1,
            tree_len,
            hidden_dim,
            val_dim,
            &scratch.linear_gated_s_first,
            &lw.out_proj_w,
            lw.out_proj_scale.as_ref(),
            lw.out_proj_int8_scale.as_ref(),
            weights.fp8_block_size,
            &mut scratch.proj_buf2,
            lw.out_proj_int4_scale.as_ref(),
            lw.out_proj_int4_zero.as_ref(),
            lw.out_proj_awq_inv_scale.as_ref(),
            weights.int4_group_size,
        )?;

        residual_add(
            ordinal,
            tree_len * hidden_dim,
            &mut scratch.hidden,
            &scratch.proj_buf2,
        )
        .map_err(|e| anyhow::anyhow!("tree layer {idx} linear attn residual: {e}"))?;
    }

    Ok(())
}

fn prefill_mlp_layer(
    weights: &Qwen35Weights,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    seq_len: usize,
    ordinal: usize,
) -> Result<()> {
    let lw = &weights.layers[idx];
    let hidden_dim = config.hidden_size;
    let intermediate = config.intermediate_size;

    let paired_gate_up = maybe_matmul_ggml_mlp_gate_up_pair(
        ordinal,
        1,
        seq_len,
        intermediate,
        hidden_dim,
        &scratch.normed,
        &lw.gate_proj_w,
        lw.gate_proj_scale.as_ref(),
        lw.gate_proj_int8_scale.as_ref(),
        lw.gate_proj_int4_scale.as_ref(),
        lw.gate_proj_int4_zero.as_ref(),
        lw.gate_proj_awq_inv_scale.as_ref(),
        &lw.up_proj_w,
        lw.up_proj_scale.as_ref(),
        lw.up_proj_int8_scale.as_ref(),
        lw.up_proj_int4_scale.as_ref(),
        lw.up_proj_int4_zero.as_ref(),
        lw.up_proj_awq_inv_scale.as_ref(),
        &mut scratch.proj_buf,
        &mut scratch.mlp_buf,
    )?;

    if !paired_gate_up {
        // gate_proj: normed [seq, hidden] x gate_w [intermediate, hidden]^T -> [seq, intermediate]
        if let Some(sc) = lw.gate_proj_int8_scale.as_ref() {
            matmul_int8_mixed_host(
                ordinal,
                1,
                seq_len,
                intermediate,
                hidden_dim,
                &scratch.normed,
                weights,
                &format!(
                    "{}.layers.{idx}.mlp.gate_proj.weight",
                    weights.weight_prefix
                ),
                &lw.gate_proj_w,
                sc,
                &mut scratch.proj_buf,
            )?;
        } else {
            matmul_proj(
                ordinal,
                1,
                seq_len,
                intermediate,
                hidden_dim,
                &scratch.normed,
                &lw.gate_proj_w,
                lw.gate_proj_scale.as_ref(),
                lw.gate_proj_int8_scale.as_ref(),
                weights.fp8_block_size,
                &mut scratch.proj_buf,
                lw.gate_proj_int4_scale.as_ref(),
                lw.gate_proj_int4_zero.as_ref(),
                lw.gate_proj_awq_inv_scale.as_ref(),
                weights.int4_group_size,
            )?;
        }

        // up_proj: normed [seq, hidden] x up_w [intermediate, hidden]^T -> [seq, intermediate]
        if let Some(sc) = lw.up_proj_int8_scale.as_ref() {
            matmul_int8_mixed_host(
                ordinal,
                1,
                seq_len,
                intermediate,
                hidden_dim,
                &scratch.normed,
                weights,
                &format!("{}.layers.{idx}.mlp.up_proj.weight", weights.weight_prefix),
                &lw.up_proj_w,
                sc,
                &mut scratch.proj_buf2,
            )?;
        } else {
            matmul_proj(
                ordinal,
                1,
                seq_len,
                intermediate,
                hidden_dim,
                &scratch.normed,
                &lw.up_proj_w,
                lw.up_proj_scale.as_ref(),
                lw.up_proj_int8_scale.as_ref(),
                weights.fp8_block_size,
                &mut scratch.proj_buf2,
                lw.up_proj_int4_scale.as_ref(),
                lw.up_proj_int4_zero.as_ref(),
                lw.up_proj_awq_inv_scale.as_ref(),
                weights.int4_group_size,
            )?;
        }

        // SwiGLU: out = silu(gate) * up
        prefill_ffi::swiglu_mul(
            ordinal,
            ScalarType::BF16,
            seq_len * intermediate,
            &scratch.proj_buf,
            &scratch.proj_buf2,
            &mut scratch.mlp_buf,
        )?;
    }

    // down_proj: mlp_buf [seq, intermediate] x down_w [hidden, intermediate]^T -> [seq, hidden]
    if let Some(sc) = lw.down_proj_int8_scale.as_ref() {
        matmul_int8_mixed_host(
            ordinal,
            1,
            seq_len,
            hidden_dim,
            intermediate,
            &scratch.mlp_buf,
            weights,
            &format!(
                "{}.layers.{idx}.mlp.down_proj.weight",
                weights.weight_prefix
            ),
            &lw.down_proj_w,
            sc,
            &mut scratch.proj_buf,
        )?;
        scratch.residual_add_from_source(
            ordinal,
            seq_len * hidden_dim,
            ResidualSource::ProjBuf,
            "MLP residual",
        )?;
    } else {
        let down_qtype = qwen35::weights::infer_lowbit_type(
            &lw.down_proj_w,
            intermediate,
            lw.down_proj_int4_scale.is_some(),
        );
        let fused_down_residual = down_qtype != qwen35::weights::LOWBIT_GGML_Q6_K
            && env::var_os("SUPERSONIC_DFLASH_DISABLE_MLP_DOWN_RESIDUAL_FUSED_MATMUL").is_none()
            && !scratch.has_f32_activation_carry()
            && matmul_proj_residual_add_inplace(
                ordinal,
                1,
                seq_len,
                hidden_dim,
                intermediate,
                &scratch.mlp_buf,
                &lw.down_proj_w,
                lw.down_proj_scale.as_ref(),
                lw.down_proj_int8_scale.as_ref(),
                &mut scratch.hidden,
                lw.down_proj_int4_scale.as_ref(),
                lw.down_proj_int4_zero.as_ref(),
                lw.down_proj_awq_inv_scale.as_ref(),
                weights.int4_group_size,
            )?;
        let q6_fused_down_residual = !fused_down_residual
            && !scratch.has_f32_activation_carry()
            && maybe_matmul_q6_k_mmq_mlp_down_residual_add(
                ordinal,
                1,
                seq_len,
                hidden_dim,
                intermediate,
                down_qtype,
                lw.down_proj_scale.as_ref(),
                lw.down_proj_int4_scale.as_ref(),
                lw.down_proj_int4_zero.as_ref(),
                lw.down_proj_awq_inv_scale.as_ref(),
                &scratch.mlp_buf,
                &lw.down_proj_w,
                &mut scratch.hidden,
                &mut scratch.q6_k_mmq_q8_workspace,
            )?;
        if !fused_down_residual && !q6_fused_down_residual {
            if !maybe_matmul_q6_k_mmq_mlp_down(
                ordinal,
                1,
                seq_len,
                hidden_dim,
                intermediate,
                down_qtype,
                lw.down_proj_scale.as_ref(),
                lw.down_proj_int4_scale.as_ref(),
                lw.down_proj_int4_zero.as_ref(),
                lw.down_proj_awq_inv_scale.as_ref(),
                &scratch.mlp_buf,
                &lw.down_proj_w,
                &mut scratch.proj_buf,
                &mut scratch.q6_k_mmq_q8_workspace,
            )? {
                matmul_proj(
                    ordinal,
                    1,
                    seq_len,
                    hidden_dim,
                    intermediate,
                    &scratch.mlp_buf,
                    &lw.down_proj_w,
                    lw.down_proj_scale.as_ref(),
                    lw.down_proj_int8_scale.as_ref(),
                    weights.fp8_block_size,
                    &mut scratch.proj_buf,
                    lw.down_proj_int4_scale.as_ref(),
                    lw.down_proj_int4_zero.as_ref(),
                    lw.down_proj_awq_inv_scale.as_ref(),
                    weights.int4_group_size,
                )?;
            }
            scratch.residual_add_from_source(
                ordinal,
                seq_len * hidden_dim,
                ResidualSource::ProjBuf,
                "MLP residual",
            )?;
        }
    }

    Ok(())
}
