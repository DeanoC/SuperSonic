//! Native GPU prefill engine — replaces the Python oracle.
//!
//! Orchestrates component kernels (embedding, matmul, attention, conv, recurrence,
//! norms, MLP) to process the entire prompt sequence through the model on GPU.

use std::{env, ffi::c_void};

use anyhow::Result;
use gpu_hal::{copy_h2d, Backend, GpuBuffer, ScalarType};

use qwen38::config::TextConfig;
use qwen38::rotary::RotaryTables;
use qwen38::state::ModelState;
use qwen38::weights::Qwen38Weights;

use crate::mtp::{MtpPrefillAppendCache, MtpVerifyScratch};
use crate::tensor_bytes::{bf16_bytes_to_f32 as decode_bf16_le, f32_to_f32_bytes as encode_f32_le};
use kernel_ffi::prefill_ffi;
use kernel_ffi;

/// D2D copy helper shared by the retained component paths.
fn copy_d2d_batched(
    ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
) -> Result<()> {
    gpu_hal::copy_d2d(ordinal, dst, src, bytes).map_err(|e| anyhow::anyhow!("d2d copy: {e}"))
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
    let qtype = qwen38::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    if qwen38::weights::is_gqh_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("GQH matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen38::weights::matmul_gqh(ordinal, m, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_gqh: {e}"));
    }
    if qwen38::weights::is_mix_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("mix matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen38::weights::matmul_mix(ordinal, m, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_mix: {e}"));
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
    } else if int8_scale.is_some() {
        anyhow::bail!("integer activation scales are not supported by the HIP Qwen3.8 path")
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
    weights: &Qwen38Weights,
    out: &mut GpuBuffer,
    label: &str,
) -> Result<bool> {
    let Some((qtype, scale, zero)) = weights.lm_head_lowbit_params(hidden_dim) else {
        return Ok(false);
    };
    if qwen38::weights::is_gqh_qtype(qtype) {
        qwen38::weights::matmul_gqh(
            ordinal,
            count,
            vocab_size,
            hidden_dim,
            lhs,
            weights.lm_head(),
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} gqh: {e}"))?;
        return Ok(true);
    }
    if qwen38::weights::is_mix_qtype(qtype) {
        qwen38::weights::matmul_mix(
            ordinal,
            count,
            vocab_size,
            hidden_dim,
            lhs,
            weights.lm_head(),
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} mix: {e}"))?;
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
        weights.lm_head(),
        out,
    )? {
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            count,
            vocab_size,
            hidden_dim,
            lhs,
            weights.lm_head(),
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
    if gpu_hal::current_backend() != Backend::Hip
        || batch != 1
        || m != 16
        || scale.is_some()
        || int8_scale.is_some()
        || int4_awq_inv_scale.is_some()
    {
        return Ok(false);
    }

    let qtype = qwen38::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    let raw_ggml = matches!(
        qtype,
        qwen38::weights::LOWBIT_GGML_Q8_0
            | qwen38::weights::LOWBIT_GGML_Q4_K
            | qwen38::weights::LOWBIT_GGML_Q5_K
            | qwen38::weights::LOWBIT_GGML_Q6_K
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
    true
}

fn q6_k_mmq_mlp_down_enabled() -> bool {
    env::var_os("SUPERSONIC_DISABLE_Q6_K_MMQ_MLP_DOWN").is_none()
}

fn q6_k_mmq_mlp_down_residual_fused_enabled() -> bool {
    true
}

fn ggml_mlp_gate_up_pair_enabled() -> bool {
    true
}

fn ggml_mlp_gate_up_swiglu_fused_enabled() -> bool {
    true
}

fn raw_ggml_qtype(qtype: i32) -> bool {
    matches!(
        qtype,
        qwen38::weights::LOWBIT_GGML_Q8_0
            | qwen38::weights::LOWBIT_GGML_Q4_K
            | qwen38::weights::LOWBIT_GGML_Q5_K
            | qwen38::weights::LOWBIT_GGML_Q6_K
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
        || qtype != qwen38::weights::LOWBIT_GGML_Q6_K
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
        qwen38::weights::LOWBIT_GGML_Q6_K,
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
        || qtype != qwen38::weights::LOWBIT_GGML_Q6_K
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

    let gate_qtype = qwen38::weights::infer_lowbit_type(gate_weight, k, false);
    let up_qtype = qwen38::weights::infer_lowbit_type(up_weight, k, false);
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
        || qtype != qwen38::weights::LOWBIT_GGML_Q6_K
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
        qwen38::weights::LOWBIT_GGML_Q6_K,
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
        || qtype != qwen38::weights::LOWBIT_GGML_Q6_K
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
        qwen38::weights::LOWBIT_GGML_Q6_K,
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
    if config.rms_norm_add_unit_offset || gpu_hal::current_backend() != Backend::Hip {
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
    weights: &Qwen38Weights,
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

    // Final RMSNorm → BF16 [count, hidden_dim]. Qwen3.8 uses add_unit_offset=1.
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
            weights.lm_head(),
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
            weights.lm_head(),
            hidden_dim,
            vocab_size,
            &mut counter,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head matvec: {e}"))?;
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
    weights: &Qwen38Weights,
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
            weights.lm_head(),
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
            weights.lm_head(),
            hidden_dim,
            vocab_size,
            &mut counter,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head matvec: {e}"))?;
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
    weights: &Qwen38Weights,
    config: &TextConfig,
    start: usize,
    count: usize,
    _use_4b_kernel: bool,
    ordinal: usize,
) -> Result<(Vec<u32>, GpuBuffer)> {
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
                weights.lm_head(),
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
                weights.lm_head(),
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
                weights.lm_head(),
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
    weights: &Qwen38Weights,
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

    let scan_chunk = std::env::var("SUPERSONIC_QWEN38_MTP_GREEDY_SCAN_CHUNK")
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
    /// Optional last-token debug trace for one selected linear-attention layer.
    pub linear_debug_trace: Option<LinearLayerDebugTrace>,
}

/// Result from the Qwen3.8 MTP prefill-append verifier.
pub struct PrefillAppendVerifyResult {
    pub logits: Vec<Vec<f32>>,
    pub target_next: Option<Vec<u32>>,
    /// BF16 `[chunk_len, hidden]` after final RMSNorm (embeddings_nextn).
    /// Filled on the greedy-only path used by Qwen3.8 NextN verify.
    pub normed_rows: Option<Vec<u8>>,
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
pub(crate) struct PrefillScratch {
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
    /// Grow-only `[num_kv_heads, kv_len, head_dim]` assemble workspace so
    /// padded KV caches are compacted without a per-layer hipMalloc.
    kv_assemble_k: Option<GpuBuffer>,
    kv_assemble_v: Option<GpuBuffer>,
}

impl PrefillScratch {
    pub(crate) fn new(config: &TextConfig, seq_len: usize, ordinal: usize) -> Result<Self> {
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
            kv_assemble_k: None,
            kv_assemble_v: None,
        })
    }

    pub(crate) fn copy_hidden_to(&self, ordinal: usize, dst: &mut GpuBuffer) -> Result<()> {
        let bytes = self.hidden.len_bytes().min(dst.len_bytes());
        gpu_hal::copy_d2d(ordinal, dst.as_mut_ptr(), self.hidden.as_ptr(), bytes)
            .map_err(|e| anyhow::anyhow!("copy MTP residual: {e}"))
    }
}

#[derive(Clone, Copy)]
enum ResidualSource {
    ProjBuf,
    ProjBuf2,
}

fn prefill_f32_activation_carry_enabled() -> bool {
    env::var_os("SUPERSONIC_QWEN38_PREFILL_F32_ACTIVATION_CARRY").is_some()
}

impl PrefillScratch {
    fn has_f32_activation_carry(&self) -> bool {
        self.hidden_f32.is_some()
    }

    fn take_kv_assemble(
        &mut self,
        ordinal: usize,
        num_kv_heads: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        let need = num_kv_heads
            .checked_mul(kv_len)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| anyhow::anyhow!("kv assemble size overflow"))?;
        let alloc_kv_len = kv_len.next_power_of_two().max(32);
        let shape = [num_kv_heads, alloc_kv_len, head_dim];
        let take_or_alloc = |slot: &mut Option<GpuBuffer>, label: &str| -> Result<GpuBuffer> {
            if let Some(buf) = slot.take() {
                if buf.elem_count() >= need {
                    return Ok(buf);
                }
            }
            GpuBuffer::alloc(ordinal, ScalarType::BF16, &shape)
                .map_err(|e| anyhow::anyhow!("{label}: {e}"))
        };
        Ok((
            take_or_alloc(&mut self.kv_assemble_k, "kv_assemble_k")?,
            take_or_alloc(&mut self.kv_assemble_v, "kv_assemble_v")?,
        ))
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

    pub(crate) fn normed(&self) -> &GpuBuffer {
        &self.normed
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
}

/// Run native prefill on GPU, returning logits and leaving state filled.
/// When `prefill_chunk_size > 0`, processes the prompt in chunks to reduce activation VRAM.
pub fn prefill(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
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
        use_4b_kernel,
        trace_layers,
        debug_linear_layer,
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

fn decode_hidden_dump_dir_and_pos() -> Option<(std::path::PathBuf, usize)> {
    let dir = std::env::var_os("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_DIR")?;
    let pos = std::env::var("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_POS")
        .ok()?
        .parse::<usize>()
        .ok()?;
    Some((std::path::PathBuf::from(dir), pos))
}

fn dump_decode_hidden_f32(
    dir: &std::path::Path,
    kind: &str,
    layer: usize,
    ordinal: usize,
    source: &GpuBuffer,
    hidden_dim: usize,
) -> Result<()> {
    dump_named_bf16_as_f32(
        dir,
        &format!("{kind}_{layer:02}"),
        ordinal,
        source,
        hidden_dim,
    )
}

fn dump_named_bf16_as_f32(
    dir: &std::path::Path,
    name: &str,
    ordinal: usize,
    source: &GpuBuffer,
    cols: usize,
) -> Result<()> {
    let bytes = copy_bf16_row(ordinal, source, 0, cols, &format!("decode dump {name}"))?;
    let f32s = decode_bf16_le(&bytes);
    if f32s.len() > 3994 {
        eprintln!("[dump] {name} n={} dim3994={:.6}", f32s.len(), f32s[3994]);
    }
    std::fs::write(dir.join(format!("{name}.f32")), encode_f32_le(&f32s))
        .map_err(|e| anyhow::anyhow!("write decode {name}: {e}"))
}

fn dump_named_f32_row(
    dir: &std::path::Path,
    name: &str,
    source: &GpuBuffer,
    cols: usize,
) -> Result<()> {
    let bytes = source
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("decode dump {name} D2H: {e}"))?;
    let mut f32s: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if f32s.len() > cols {
        f32s.truncate(cols);
    }
    if f32s.len() > 3994 {
        eprintln!("[dump] {name} n={} dim3994={:.6}", f32s.len(), f32s[3994]);
    }
    std::fs::write(dir.join(format!("{name}.f32")), encode_f32_le(&f32s))
        .map_err(|e| anyhow::anyhow!("write decode {name}: {e}"))
}

fn linear_layer_dump_dir(idx: usize) -> Option<std::path::PathBuf> {
    let want = std::env::var("SUPERSONIC_QWEN38_DUMP_LINEAR_LAYER")
        .ok()?
        .parse::<usize>()
        .ok()?;
    if want != idx {
        return None;
    }
    std::env::var_os("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_DIR").map(std::path::PathBuf::from)
}

#[allow(dead_code)]
fn prefill_inner(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
) -> Result<PrefillResult> {
    let config = &weights.config;
    let seq_len = prompt_ids.len();
    if seq_len == 0 {
        return Err(anyhow::anyhow!("prefill_inner: prompt is empty"));
    }
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

        // Upload token IDs for this chunk.
        let chunk_ids = &prompt_ids[chunk_start..chunk_start + chunk_len];
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
            weights.embed_tokens(),
            &token_ids_gpu,
            &mut scratch.hidden,
        )?;
        if chunk_start == 0 {
            if let Ok(path) = std::env::var("SUPERSONIC_QWEN38_DUMP_EMBED") {
                let n = (chunk_len * hidden_dim).min(hidden_dim);
                let bytes = copy_bf16_row(ordinal, &scratch.hidden, 0, n, "prefill embed dump")?;
                std::fs::write(&path, bytes)
                    .map_err(|e| anyhow::anyhow!("write embed dump {path}: {e}"))?;
                eprintln!("[prefill] dumped embed row0 n={n} to {path}");
            }
        }
        scratch.seed_f32_from_hidden(ordinal, chunk_len * hidden_dim, "prefill embedding")?;

        for idx in 0..config.num_hidden_layers {
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
                    weights.layers[idx]
                        .full
                        .as_ref()
                        .expect("full attn weights"),
                    &mut state.layers[idx],
                    rotary,
                    &mut scratch,
                    config,
                    idx,
                    chunk_len,
                    chunk_start,
                    ordinal,
                    kv_chunk_size,
                    /* commit_kv_filled */ true,
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
            prefill_mlp_layer(
                weights,
                &weights.layers[idx],
                &mut scratch,
                config,
                idx,
                chunk_len,
                ordinal,
            )?;

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
            }
        }

        chunk_start += chunk_len;
    }

    // Extract logits for the last token of the final chunk.
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

    Ok(PrefillResult {
        logits,
        final_norm_trace,
        layer_attn_trace,
        layer_post_attn_norm_trace,
        layer_mlp_swiglu_trace,
        layer_mlp_out_trace,
        layer_hidden_trace,
        linear_debug_trace,
    })
}

/// Append a contiguous Qwen3.8 MTP verify block to the live target state using
/// the prefill component kernels.
///
/// Unlike the position-zero helper, this does not assume position zero. The caller
/// supplies the absolute `pos_offset`; full-attention KV is written at
/// `[pos_offset, pos_offset + token_ids.len())` without advancing `kv_filled`,
/// while linear-attention state is mutated in place. The Qwen3.8 MTP driver
/// snapshots/restores linear state around this call, just as it does for the
/// persistent fused verifier.
#[allow(clippy::too_many_arguments)]
pub fn prefill_append_verify_cached(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: &mut MtpPrefillAppendCache,
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
        greedy_only,
        greedy_compare_tokens,
        Some(cache),
    )
}

#[allow(clippy::too_many_arguments)]
fn prefill_append_verify_impl(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: Option<&mut MtpPrefillAppendCache>,
) -> Result<PrefillAppendVerifyResult> {
    if token_ids.is_empty() {
        return Err(anyhow::anyhow!("prefill_append_verify: token_ids is empty"));
    }

    let config = &weights.config;
    let chunk_len = token_ids.len();
    let hidden_dim = config.hidden_size;
    let profile = std::env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE_APPEND").is_some();
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
            local_cache = MtpPrefillAppendCache::new(config, chunk_len, ordinal)?;
            &mut local_cache
        }
    };
    if !cache.matches(chunk_len, ordinal) {
        let can_reuse_larger = cache.ordinal == ordinal && cache.chunk_len >= chunk_len;
        if !can_reuse_larger {
            *cache = MtpPrefillAppendCache::new(config, chunk_len, ordinal)?;
        }
    }
    let scratch = &mut cache.scratch;
    let chunk_conv_tail = &mut cache.chunk_conv_tail;
    let token_ids_gpu = &mut cache.token_ids_gpu;

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
        weights.embed_tokens(),
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
                weights.layers[idx]
                    .full
                    .as_ref()
                    .expect("full attn weights"),
                &mut state.layers[idx],
                rotary,
                scratch,
                config,
                idx,
                chunk_len,
                pos_offset,
                ordinal,
                kv_chunk_size,
                /* commit_kv_filled */ false,
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
        prefill_mlp_layer(
            weights,
            &weights.layers[idx],
            scratch,
            config,
            idx,
            chunk_len,
            ordinal,
        )?;
        if profile {
            ms_mlp += t_mlp.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let t_logits = std::time::Instant::now();
    let mut normed_rows = None;
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
            let (ids, normed) = compute_greedy_for_range(
                &scratch.hidden,
                weights,
                config,
                0,
                chunk_len,
                use_4b_kernel,
                ordinal,
            )?;
            normed_rows = Some(
                normed
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("append greedy normed D2H: {e}"))?,
            );
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
            "[qwen38-mtp-profile] prefill_append B={} pos={} seed={:.2}ms embed={:.2}ms input_norm={:.2}ms full_attn={:.2}ms linear_attn={:.2}ms post_norm={:.2}ms mlp={:.2}ms logits={:.2}ms",
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
        normed_rows,
    })
}

/// Per-token forward pass body shared by `mtp_decode_step` (full-logits)
/// and `mtp_decode_step_greedy` (fused argmax). Performs token embed +
/// 24-layer transformer pass, leaving the post-final-layer hidden state in
/// `scratch.scratch.hidden`. Caller is responsible for the final RMSNorm +
/// lm_head and for owning the batch guard around the call.
fn mtp_decode_step_body(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
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
            .expect("mtp chunk conv tail missing for linear layer");
        if let Some(conv_state) = state.layers[idx].conv_state.as_ref() {
            let bytes = qkv_dim * (kern - 1) * ScalarType::BF16.size_in_bytes();
            copy_d2d_batched(
                ordinal,
                chunk_tail.as_ptr() as *mut c_void,
                conv_state.as_ptr(),
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("mtp layer {idx} seed conv tail: {e}"))?;
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
    .map_err(|e| anyhow::anyhow!("mtp token id upload: {e}"))?;
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        config.vocab_size,
        hidden_dim,
        weights.embed_tokens(),
        &scratch.token_id_buf,
        &mut scratch.scratch.hidden,
    )?;
    scratch.scratch.seed_f32_from_hidden(
        ordinal,
        chunk_len * hidden_dim,
        "decode-loop embedding",
    )?;

    let dump = decode_hidden_dump_dir_and_pos()
        .filter(|(_, pos)| *pos == seqlen_offset)
        .map(|(dir, _)| dir);

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
                weights.layers[idx]
                    .full
                    .as_ref()
                    .expect("full attn weights"),
                &mut state.layers[idx],
                rotary,
                &mut scratch.scratch,
                config,
                idx,
                chunk_len,
                chunk_start,
                ordinal,
                kv_chunk_size,
                /* commit_kv_filled */ true,
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
            )?;
        }

        if let Some(dir) = dump.as_ref() {
            dump_decode_hidden_f32(
                dir,
                "attn",
                idx,
                ordinal,
                &scratch.scratch.hidden,
                hidden_dim,
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
            &weights.layers[idx],
            &mut scratch.scratch,
            config,
            idx,
            chunk_len,
            ordinal,
        )?;

        if let Some(dir) = dump.as_ref() {
            dump_decode_hidden_f32(
                dir,
                "hidden",
                idx,
                ordinal,
                &scratch.scratch.hidden,
                hidden_dim,
            )?;
        }
    }
    if let Some(dir) = dump.as_ref() {
        eprintln!(
            "[dump] component decode hiddens pos={seqlen_offset} layers={} dir={}",
            config.num_hidden_layers,
            dir.display()
        );
    }

    Ok(())
}

/// Single-token incremental decode step. Mirrors `prefill_inner`'s
/// chunk-loop body with `chunk_len=1, chunk_start=seqlen_offset, is_last_chunk=true`,
/// reading and writing the persistent layer state in place. Replaces the old
/// O(N²) replay-prefill path with O(N)-per-step proper incremental decode.
///
/// Returns the full BF16→f32 logits row over the vocabulary. Use
/// `mtp_decode_step_greedy` to skip the 250k-element D2H + host argmax
/// when only the sampled token is needed.
pub fn mtp_decode_step(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<Vec<f32>> {
    mtp_decode_step_body(
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

/// Same forward pass as `mtp_decode_step`, returning the host greedy token.
pub fn mtp_decode_step_greedy(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    mtp_decode_step_body(
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
    scratch.scratch.rms_norm_hidden_to_normed_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &weights.norm_weight,
        "decode greedy final norm",
    )?;

    if weights.lm_head_lowbit_params(hidden_dim).is_some() {
        if !prefill_lm_head_lowbit(
            ordinal,
            1,
            vocab_size,
            hidden_dim,
            scratch.scratch.normed(),
            weights,
            &mut scratch.mtp_logits,
            "decode greedy lm_head",
        )? {
            unreachable!("lowbit lm_head params were Some");
        }
    } else {
        kernel_ffi::standalone_matvec(
            ordinal,
            ScalarType::BF16,
            &mut scratch.mtp_logits,
            scratch.scratch.normed(),
            weights.lm_head(),
            hidden_dim,
            vocab_size,
            &mut scratch.mtp_counter,
        )
        .map_err(|e| anyhow::anyhow!("decode greedy lm_head matvec: {e}"))?;
    }

    prefill_ffi::argmax_bf16_rows(
        ordinal,
        1,
        vocab_size,
        &scratch.mtp_logits,
        &mut scratch.mtp_argmax,
    )
    .map_err(|e| anyhow::anyhow!("decode greedy argmax: {e}"))?;

    gpu_hal::sync(ordinal).map_err(|e| anyhow::anyhow!("decode greedy sync: {e}"))?;

    let ids_bytes = scratch
        .mtp_argmax
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("decode greedy argmax D2H: {e}"))?;
    Ok(u32::from_le_bytes([
        ids_bytes[0],
        ids_bytes[1],
        ids_bytes[2],
        ids_bytes[3],
    ]))
}

/// Per-layer full-attention prefill step.
///
/// `commit_kv_filled` controls whether the full-attention cache cursor advances.
fn prefill_full_attention_layer(
    weights: &Qwen38Weights,
    fw: &qwen38::weights::FullWeights,
    ls: &mut qwen38::state::LayerState,
    rotary: &RotaryTables,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    chunk_len: usize,
    chunk_start: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    commit_kv_filled: bool,
) -> Result<()> {
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

    // 1. Q projection
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
        &mut scratch.full_q_buf,
        fw.q_proj_int4_scale.as_ref(),
        fw.q_proj_int4_zero.as_ref(),
        fw.q_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;
    if chunk_len == 1 {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_bf16_as_f32(&dir, "l5_q", ordinal, &scratch.full_q_buf, q_proj_dim)?;
        }
    }

    // 2. Split Q into query and gate when present. Llama-style full attention
    // uses an ungated q_proj whose row count matches q_dim exactly.
    let q_norm_done = if has_attn_gate {
        if maybe_split_qgate_norm_bf16(
            config,
            ordinal,
            chunk_len,
            num_q_heads,
            head_dim,
            &scratch.full_q_buf,
            fw.q_norm_w.as_ref(),
            &mut scratch.full_query_buf,
            &mut scratch.full_gate_buf,
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
                &scratch.full_q_buf,
                &mut scratch.full_query_buf,
                &mut scratch.full_gate_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} Q split: {e}"))?;
            false
        }
    } else {
        copy_d2d_batched(
            ordinal,
            scratch.full_query_buf.as_ptr() as *mut c_void,
            scratch.full_q_buf.as_ptr(),
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
    if chunk_len == 1 {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_bf16_as_f32(&dir, "l5_k", ordinal, &scratch.proj_buf2, kv_dim)?;
        }
    }

    // 4. Q normalization
    if !q_norm_done
        && !maybe_attn_rms_norm_rows_inplace(
            config,
            ordinal,
            chunk_len * num_q_heads,
            head_dim,
            &mut scratch.full_query_buf,
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
            &scratch.full_query_buf,
            fw.q_norm_w.as_ref(),
            &mut q_normed,
            &format!("layer {idx} Q norm"),
        )?;
        copy_d2d_batched(
            ordinal,
            scratch.full_query_buf.as_ptr() as *mut c_void,
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

    // 6. RoPE on query and key.
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
        &mut scratch.full_query_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;
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
    if chunk_len == 1 {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_bf16_as_f32(&dir, "l5_qrope", ordinal, &scratch.full_query_buf, q_dim)?;
        }
    }

    // 7. V projection
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
        &mut scratch.full_v_buf,
        fw.v_proj_int4_scale.as_ref(),
        fw.v_proj_int4_zero.as_ref(),
        fw.v_proj_awq_inv_scale.as_ref(),
        weights.int4_group_size,
    )?;
    if chunk_len == 1 {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_bf16_as_f32(&dir, "l5_v", ordinal, &scratch.full_v_buf, kv_dim)?;
        }
    }

    // 8/9. Write this chunk's K/V to KV cache BEFORE attention (so attention can read from it).
    //      The HIP fast path transposes directly into the persistent cache; fallback keeps the
    //      scratch transpose plus per-head copy path for debug A/B runs.
    let mut kv_capacity_prepared = false;
    let kv_cache_written = if gpu_hal::current_backend() == Backend::Hip {
        ls.ensure_kv_capacity(kv_len - 1, ordinal, config, kv_chunk_size)
            .map_err(|e| anyhow::anyhow!("layer {idx} KV alloc: {e}"))?;
        kv_capacity_prepared = true;
        if ls.kv_cache_k.is_some() && ls.kv_cache_v.is_some() {
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
                &scratch.full_v_buf,
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
            &scratch.full_v_buf,
            &mut scratch.attn_v,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} V transpose: {e}"))?;

        if !kv_capacity_prepared {
            ls.ensure_kv_capacity(kv_len - 1, ordinal, config, kv_chunk_size)
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
        ls.set_kv_filled(kv_len);
    }

    // 10. Transpose Q to [H, chunk_len, D]
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        num_q_heads,
        head_dim,
        &scratch.full_query_buf,
        &mut scratch.attn_q,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q transpose: {e}"))?;

    // 11. Causal attention — Q: [q_heads, chunk_len, hd], K/V: [kv_heads, kv_len, hd]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let cap = ls.kv_capacity();
    let mut assembled_k = None;
    let mut assembled_v = None;
    let attn_k_ref: &GpuBuffer;
    let attn_v_ref: &GpuBuffer;

    if cap == kv_len {
        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
        // No padding - cache is already contiguous, use directly.
        attn_k_ref = cache_k_ref;
        attn_v_ref = cache_v_ref;
    } else {
        // Capacity > kv_len - copy each head's kv_len entries into contiguous buffers.
        let (kv_k_contig, kv_v_contig) =
            scratch.take_kv_assemble(ordinal, num_kv_heads, kv_len, head_dim)?;
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
        assembled_k = Some(kv_k_contig);
        assembled_v = Some(kv_v_contig);
        attn_k_ref = assembled_k.as_ref().unwrap();
        attn_v_ref = assembled_v.as_ref().unwrap();
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
    let _ = (attn_k_ref, attn_v_ref);
    if let (Some(k), Some(v)) = (assembled_k, assembled_v) {
        scratch.kv_assemble_k = Some(k);
        scratch.kv_assemble_v = Some(v);
    }

    let fused_attn_gate_prep = has_attn_gate && gpu_hal::current_backend() == Backend::Hip;
    if fused_attn_gate_prep {
        prefill_ffi::cast_transpose_gate_hsd_to_shd_bf16(
            ordinal,
            chunk_len,
            num_q_heads,
            head_dim,
            &scratch.attn_out_f32,
            &scratch.full_gate_buf,
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
            if gpu_hal::current_backend() == Backend::Hip {
                prefill_ffi::sigmoid_mul_inplace(
                    ordinal,
                    ScalarType::BF16,
                    chunk_len * q_dim,
                    &mut scratch.proj_buf,
                    &scratch.full_gate_buf,
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
                    &scratch.full_gate_buf,
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
    if chunk_len == 1 {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_bf16_as_f32(&dir, "l5_attn", ordinal, &scratch.proj_buf, q_dim)?;
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
fn prefill_linear_attention_layer(
    weights: &Qwen38Weights,
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(&dir, "l5_in", ordinal, &scratch.hidden, hidden_dim)?;
        dump_named_bf16_as_f32(&dir, "l5_normed", ordinal, &scratch.normed, hidden_dim)?;
    }

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
    // exposed by chunk_len=1 incremental decode: the new tail
    // mixed with the current chunk's QKV got fed back into this same chunk's
    // conv1d window, shifting the inputs.
    let pad = kern - 1;
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let use_fused_conv_prep = gpu_hal::current_backend() == Backend::Hip
        && !trace_linear_debug
        && chunk_start > 0
        && chunk_len >= pad;
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(&dir, "l5_qkv", ordinal, &scratch.proj_buf, qkv_dim)?;
    }
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(&dir, "l5_z", ordinal, &scratch.proj_buf2, z_dim)?;
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
    let use_fused_ba = lw.ba_proj_w.is_some();
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        if let Some(b) = b_buf {
            dump_named_bf16_as_f32(&dir, "l5_b", ordinal, b, nv)?;
        }
        if let Some(a) = a_buf {
            dump_named_bf16_as_f32(&dir, "l5_a", ordinal, a, nv)?;
        }
    }

    // 5. Transpose QKV [chunk, qkv_dim] -> [qkv_dim, pad+chunk] for conv input.
    //    The HIP helper prepares the previous tail, transposed rows, and next
    //    tail together; the generic path remains available for diagnostics.
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(&dir, "l5_conv", ordinal, &scratch.proj_buf, qkv_dim)?;
    }
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

    let use_fused_qkv_prepare = gpu_hal::current_backend() == Backend::Hip && !trace_linear_debug;
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
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_f32_row(&dir, "l5_qnorm", &scratch.linear_q_trans, key_dim)?;
            dump_named_f32_row(&dir, "l5_knorm", &scratch.linear_k_trans, key_dim)?;
        }
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

    // 11. Delta recurrent prefill.
    let state_elems = nv * khd * vhd;
    let elem_bytes_f32 = ScalarType::F32.size_in_bytes();
    let out_rows = chunk_len + khd;
    let zero_recurrent;
    let recurrent_initial = if let Some(rec_state) = state.layers[idx].recurrent_state.as_ref() {
        if let Some(dir) = linear_layer_dump_dir(idx) {
            dump_named_f32_row(&dir, "l5_rec_in", rec_state, state_elems)?;
        }
        rec_state
    } else {
        zero_recurrent = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, khd, vhd])
            .map_err(|e| anyhow::anyhow!("zero recurrent alloc: {e}"))?;
        &zero_recurrent
    };
    prefill_ffi::delta_recurrent_prefill(
        ordinal,
        ScalarType::F32,
        nv,
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

    // 12. Extract recurrent state and attention output from the F32 result.
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
    if let Some(rec_state) = state.layers[idx].recurrent_state.as_mut() {
        copy_d2d_batched(
            ordinal,
            rec_state.as_ptr() as *mut c_void,
            state_f32.as_ptr(),
            state_elems * elem_bytes_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state writeback: {e}"))?;
    }
    let state_bytes_debug = state_f32
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("layer {idx} debug state_f32 D2H: {e}"))?;

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
    let attn_output_f32_debug = attn_output_f32
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("layer {idx} debug attn_output_f32 D2H: {e}"))?;
    let _ = is_last_chunk; // recurrent state is now always written; flag still gates conv_state above.
    if trace_linear_debug {
        let trace = linear_debug_trace
            .as_mut()
            .expect("linear debug trace missing");
        let attn_out_bytes = &attn_output_f32_debug;
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
        let state_bytes = &state_bytes_debug;
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

    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(
            &dir,
            "l5_rec",
            ordinal,
            &scratch.linear_attn_output,
            val_dim,
        )?;
    }

    // 14. Gated RMSNorm: out = rms_norm(attn_output) * norm_w * silu(Z)
    //     attn_output is [nv, S, vhd]; Z (proj_buf2) is [S, val_dim] = [S, nv*vhd]
    //     Need Z in [nv, S, vhd] layout
    let use_fused_gated_epilogue = gpu_hal::current_backend() == Backend::Hip;
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        dump_named_bf16_as_f32(
            &dir,
            "l5_gated",
            ordinal,
            &scratch.linear_gated_s_first,
            val_dim,
        )?;
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
    if let Some(dir) = linear_layer_dump_dir(idx) {
        if !fused_residual {
            dump_named_bf16_as_f32(&dir, "l5_proj", ordinal, &scratch.proj_buf2, hidden_dim)?;
        }
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
fn prefill_mlp_layer(
    weights: &Qwen38Weights,
    lw: &qwen38::weights::LayerWeights,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    _idx: usize,
    seq_len: usize,
    ordinal: usize,
) -> Result<()> {
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

        // up_proj: normed [seq, hidden] x up_w [intermediate, hidden]^T -> [seq, intermediate]
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
    let down_qtype = qwen38::weights::infer_lowbit_type(
        &lw.down_proj_w,
        intermediate,
        lw.down_proj_int4_scale.is_some(),
    );
    let fused_down_residual = down_qtype != qwen38::weights::LOWBIT_GGML_Q6_K
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

    Ok(())
}

/// One NextN/MTP draft step. Matches llama.cpp `graph_mtp`.
///
/// `h` is either a trunk last-layer residual (`h_is_nextn=false`, RMS with
/// `output_norm` first) or an embeddings_nextn / previous MTP `t_h_nextn`
/// row (`h_is_nextn=true`). Compact MTP KV is written at `ls.kv_filled`;
/// RoPE uses `abs_pos`. Writes shared-head hidden into `out_h`.
pub fn mtp_forward(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    h: &GpuBuffer,
    h_is_nextn: bool,
    token_id: u32,
    _abs_pos: usize,
    out_h: &mut GpuBuffer,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    let mtp = weights
        .mtp
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Qwen3.8 MTP: blk.64 weights were not loaded"))?;
    let ls = state
        .mtp
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Qwen3.8 MTP: missing MTP KV state"))?;
    let fw = mtp
        .layer
        .full
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Qwen3.8 MTP: expected a full-attention block"))?;
    let config = &weights.config;
    let hidden_dim = config.hidden_size;
    let elem = ScalarType::BF16.size_in_bytes();
    anyhow::ensure!(
        scratch.scratch.mlp_buf.shape().get(1).copied().unwrap_or(0) >= hidden_dim * 2,
        "mtp concat needs mlp_buf cols >= {}, got {:?}",
        hidden_dim * 2,
        scratch.scratch.mlp_buf.shape()
    );

    copy_d2d_batched(
        ordinal,
        scratch.mtp_residual.as_mut_ptr(),
        h.as_ptr(),
        hidden_dim * elem,
    )
    .map_err(|e| anyhow::anyhow!("mtp residual copy: {e}"))?;
    if h_is_nextn {
        copy_d2d_batched(
            ordinal,
            scratch.scratch.proj_buf2.as_mut_ptr(),
            scratch.mtp_residual.as_ptr(),
            hidden_dim * elem,
        )
        .map_err(|e| anyhow::anyhow!("mtp h_nextn copy: {e}"))?;
    } else {
        rms_norm_rows_model(
            config,
            ordinal,
            1,
            hidden_dim,
            &scratch.mtp_residual,
            &weights.norm_weight,
            &mut scratch.scratch.proj_buf2,
            "mtp output_norm / h_nextn",
        )?;
    }

    let id_bytes = token_id.to_le_bytes();
    copy_h2d(
        ordinal,
        scratch.token_id_buf.as_ptr() as *mut c_void,
        id_bytes.as_ptr() as *const c_void,
        4,
    )
    .map_err(|e| anyhow::anyhow!("mtp token id upload: {e}"))?;
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        1,
        config.vocab_size,
        hidden_dim,
        weights.embed_tokens(),
        &scratch.token_id_buf,
        &mut scratch.scratch.hidden,
    )?;

    rms_norm_rows_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &scratch.scratch.hidden,
        &mtp.enorm_w,
        &mut scratch.scratch.proj_buf,
        "mtp enorm",
    )?;
    rms_norm_rows_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &scratch.scratch.proj_buf2,
        &mtp.hnorm_w,
        &mut scratch.mtp_residual,
        "mtp hnorm",
    )?;
    copy_d2d_batched(
        ordinal,
        scratch.scratch.mlp_buf.as_mut_ptr(),
        scratch.scratch.proj_buf.as_ptr(),
        hidden_dim * elem,
    )
    .map_err(|e| anyhow::anyhow!("mtp concat e: {e}"))?;
    copy_d2d_batched(
        ordinal,
        scratch.scratch.mlp_buf.offset_ptr(hidden_dim * elem) as *mut c_void,
        scratch.mtp_residual.as_ptr(),
        hidden_dim * elem,
    )
    .map_err(|e| anyhow::anyhow!("mtp concat h: {e}"))?;

    matmul_proj(
        ordinal,
        1,
        1,
        hidden_dim,
        hidden_dim * 2,
        &scratch.scratch.mlp_buf,
        &mtp.eh_proj_w,
        None,
        None,
        0,
        &mut scratch.scratch.hidden,
        None,
        None,
        None,
        0,
    )?;

    scratch.scratch.rms_norm_hidden_to_normed_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &mtp.layer.input_norm_w,
        "mtp attn norm",
    )?;
    let compact = ls.kv_filled;
    prefill_full_attention_layer(
        weights,
        fw,
        ls,
        rotary,
        &mut scratch.scratch,
        config,
        64,
        1,
        compact,
        ordinal,
        kv_chunk_size,
        /* commit_kv_filled */ true,
    )?;
    scratch.scratch.rms_norm_hidden_to_normed_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &mtp.layer.post_attn_norm_w,
        "mtp post-attn norm",
    )?;
    prefill_mlp_layer(
        weights,
        &mtp.layer,
        &mut scratch.scratch,
        config,
        64,
        1,
        ordinal,
    )?;

    rms_norm_rows_model(
        config,
        ordinal,
        1,
        hidden_dim,
        &scratch.scratch.hidden,
        &mtp.shared_head_norm_w,
        out_h,
        "mtp shared_head_norm",
    )?;
    let vocab_size = config.vocab_size;
    if !prefill_lm_head_lowbit(
        ordinal,
        1,
        vocab_size,
        hidden_dim,
        out_h,
        weights,
        &mut scratch.mtp_logits,
        "mtp lm_head",
    )? {
        kernel_ffi::standalone_matvec(
            ordinal,
            ScalarType::BF16,
            &mut scratch.mtp_logits,
            out_h,
            weights.lm_head(),
            hidden_dim,
            vocab_size,
            &mut scratch.mtp_counter,
        )
        .map_err(|e| anyhow::anyhow!("mtp lm_head matvec: {e}"))?;
    }
    prefill_ffi::argmax_bf16_rows(
        ordinal,
        1,
        vocab_size,
        &scratch.mtp_logits,
        &mut scratch.mtp_argmax,
    )
    .map_err(|e| anyhow::anyhow!("mtp argmax: {e}"))?;
    let mut id = [0u8; 4];
    gpu_hal::copy_d2h(
        ordinal,
        id.as_mut_ptr() as *mut c_void,
        scratch.mtp_argmax.as_ptr(),
        4,
    )
    .map_err(|e| anyhow::anyhow!("mtp argmax d2h: {e}"))?;
    Ok(u32::from_le_bytes(id))
}

/// Diagnostic wrapper: trunk residual in, greedy draft token out.
pub fn mtp_draft_greedy(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    h: &GpuBuffer,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    let hidden_dim = weights.config.hidden_size;
    let mut out_h = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
        .map_err(|e| anyhow::anyhow!("mtp draft out_h alloc: {e}"))?;
    mtp_forward(
        weights,
        state,
        rotary,
        scratch,
        h,
        false,
        token_id,
        seqlen_offset,
        &mut out_h,
        ordinal,
        kv_chunk_size,
    )
}
