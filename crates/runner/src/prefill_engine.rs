//! Native GPU prefill engine — replaces the Python oracle.
//!
//! Orchestrates component kernels (embedding, matmul, attention, conv, recurrence,
//! norms, MLP) to process the entire prompt sequence through the model on GPU.

use std::ffi::c_void;

use anyhow::Result;
use gpu_hal::{copy_h2d, GpuBuffer, ScalarType};
use half::{bf16, f16};

use qwen35::config::TextConfig;
use qwen35::rotary::RotaryTables;
use qwen35::state::{kv_fp8_bf16_sidecar_enabled, kv_fp8_bf16_sidecar_window_tokens, ModelState};
use qwen35::weights::Qwen35Weights;

use kernel_ffi::prefill_ffi;

fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2)
        .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn encode_bf16_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
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
    flags.into_iter()
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
        ordinal, batch, m, n, k, lhs, weights, weight_name, weight, int8_scale, out, None,
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
        return prefill_ffi::matmul_rhs_transposed_int8(ordinal, batch, m, n, k, lhs, weight, int8_scale, out)
            .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
    };
    if weights.int8_outlier_threshold <= 0.0 {
        return prefill_ffi::matmul_rhs_transposed_int8(ordinal, batch, m, n, k, lhs, weight, int8_scale, out)
            .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"));
    }

    let owned_prepared;
    let prepared_lhs = if let Some(prepared_lhs) = prepared_lhs {
        prepared_lhs
    } else {
        owned_prepared = prepare_int8_mixed_lhs(ordinal, batch, m, k, lhs, weights)?;
        let Some(prepared_lhs) = owned_prepared.as_ref() else {
            return prefill_ffi::matmul_rhs_transposed_int8(ordinal, batch, m, n, k, lhs, weight, int8_scale, out)
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
        return prefill_ffi::matmul_rhs_transposed_int8(ordinal, batch, m, n, k, lhs, weight, int8_scale, out)
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
    int4_group_size: usize,
) -> Result<()> {
    if let (Some(sc), Some(zr)) = (int4_scale, int4_zero) {
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal, batch, m, n, k, lhs, weight, sc, zr, int4_group_size, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
    } else if let Some(sc) = int8_scale {
        prefill_ffi::matmul_rhs_transposed_int8(
            ordinal, batch, m, n, k, lhs, weight, sc, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int8: {e}"))
    } else {
        match scale {
            Some(s) => prefill_ffi::matmul_rhs_transposed_fp8(
                ordinal, batch, m, n, k, lhs, weight, s, block_size, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_fp8: {e}")),
            None => prefill_ffi::matmul_rhs_transposed(
                ordinal, ScalarType::BF16, batch, m, n, k, lhs, weight, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul: {e}")),
        }
    }
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
    prefill_ffi::element_add(
        ordinal,
        ScalarType::BF16,
        total_elems,
        lhs,
        src,
        dst,
    )
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
        gpu_hal::copy_d2d(ordinal, output.as_mut_ptr(), input.as_ptr(), rows * cols * ScalarType::BF16.size_in_bytes())
            .map_err(|e| anyhow::anyhow!("{label} copy-through: {e}"))?;
    }
    Ok(())
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
        return Err(anyhow::anyhow!("compute_logits_for_range: count must be > 0"));
    }
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;
    let elem_bytes = ScalarType::BF16.size_in_bytes();

    // D2D slice [start..start+count] of the hidden-state buffer.
    let slice = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range slice alloc: {e}"))?;
    let src_offset = start * hidden_dim * elem_bytes;
    gpu_hal::copy_d2d(
        ordinal,
        slice.as_ptr() as *mut c_void,
        hidden.offset_ptr(src_offset),
        count * hidden_dim * elem_bytes,
    )
    .map_err(|e| anyhow::anyhow!("range slice copy: {e}"))?;

    // Final RMSNorm → BF16 [count, hidden_dim]. Qwen3.5 uses add_unit_offset=1.
    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, hidden_dim])
        .map_err(|e| anyhow::anyhow!("range normed alloc: {e}"))?;
    rms_norm_rows_model(
        config,
        ordinal,
        count,
        hidden_dim,
        &slice,
        &weights.norm_weight,
        &mut normed,
        "range final norm",
    )?;

    // lm_head projection. For count=1, prefer the standalone matvec even on
    // 4B-capable models: it avoids the tiled matmul path's extra packing and
    // keeps the single-row score path numerically aligned with decode.
    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[count, vocab_size])
        .map_err(|e| anyhow::anyhow!("range logits alloc: {e}"))?;
    if count > 1 {
        kernel_ffi::matmul_rhs_transposed_4b(
            ordinal, ScalarType::BF16,
            1,         // batch
            count,     // m
            vocab_size, // n
            hidden_dim, // k
            &normed, &*weights.lm_head, &mut logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("range lm_head tiled: {e}"))?;
    } else {
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("range matvec counter: {e}"))?;
        kernel_ffi::standalone_matvec(
            ordinal, ScalarType::BF16,
            &mut logits_buf, &normed, &*weights.lm_head,
            hidden_dim, vocab_size, &mut counter,
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
    /// Optional last-token debug trace for one selected linear-attention layer.
    pub linear_debug_trace: Option<LinearLayerDebugTrace>,
}

pub struct LinearLayerDebugTrace {
    pub qkv: Vec<u8>,
    pub qkv_tail: Vec<u8>,
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
    /// [seq_len, hidden_dim] BF16 — normed activations
    normed: GpuBuffer,
    /// [seq_len, max_proj_dim] BF16 — projection output buffer
    proj_buf: GpuBuffer,
    /// [seq_len, max_proj_dim] BF16 — second projection buffer (for gate/up)
    proj_buf2: GpuBuffer,
    /// [seq_len, intermediate_size] BF16 — MLP intermediate
    mlp_buf: GpuBuffer,
    /// [1, vocab_size] BF16 — logits output
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
    // Linear attention scratch:
    /// [qkv_dim, seq_len + kern - 1] BF16 — padded conv input
    conv_input: GpuBuffer,
}

impl PrefillScratch {
    fn new(config: &TextConfig, seq_len: usize, ordinal: usize) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kern = config.linear_conv_kernel_dim;

        // Max projection dim across all layer types and MLP
        let max_proj = [
            // Full attention: q_proj (doubled for gate)
            num_q_heads * head_dim * 2,
            // Linear attention: qkv_out
            config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim,
            // MLP: intermediate_size (gate/up projection output)
            intermediate,
        ]
        .into_iter()
        .max()
        .unwrap();

        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_total_len = seq_len + kern - 1;

        Ok(Self {
            hidden: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill hidden: {e}"))?,
            normed: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill normed: {e}"))?,
            proj_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf: {e}"))?,
            proj_buf2: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf2: {e}"))?,
            mlp_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, intermediate])
                .map_err(|e| anyhow::anyhow!("prefill mlp_buf: {e}"))?,
            logits_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("prefill logits: {e}"))?,
            attn_q: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_q_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_q: {e}"))?,
            attn_k: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_k: {e}"))?,
            attn_v: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_v: {e}"))?,
            attn_out_f32: GpuBuffer::zeros(
                ordinal,
                ScalarType::F32,
                &[num_q_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_out_f32: {e}"))?,
            conv_input: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[qkv_dim, conv_total_len],
            )
            .map_err(|e| anyhow::anyhow!("prefill conv_input: {e}"))?,
        })
    }
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
        weights, state, rotary, prompt_ids, ordinal, kv_chunk_size,
        prefill_chunk_size, kv_fp8, use_4b_kernel, trace_layers,
        debug_linear_layer, None,
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
        weights, state, rotary, prompt_ids, ordinal, kv_chunk_size,
        prefill_chunk_size, kv_fp8, use_4b_kernel, trace_layers,
        debug_linear_layer, Some(tap_layers),
    )
}

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
) -> Result<PrefillResult> {
    let config = &weights.config;
    let seq_len = prompt_ids.len();
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

    // Per-layer inter-chunk state for linear attention layers
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let kern = config.linear_conv_kernel_dim;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + nv * vhd;

    // Allocate inter-chunk recurrent states (one per linear layer)
    // These carry the delta recurrent state between chunks.
    // Must be BF16 because the delta_recurrent_prefill kernel reads initial_state as T (BF16).
    let mut chunk_recurrent: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
        .map(|i| {
            if config.is_full_attention(i) {
                Ok(None)
            } else {
                GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, khd, vhd])
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("chunk recurrent alloc: {e}"))
            }
        })
        .collect::<Result<Vec<_>>>()?;

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

        // Upload token IDs for this chunk
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
            &weights.embed_tokens,
            &token_ids_gpu,
            &mut scratch.hidden,
        )?;

        // Layer loop (all layers for this chunk)
        for idx in 0..config.num_hidden_layers {
            // Input RMSNorm
            rms_norm_rows_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &scratch.hidden,
                &weights.layers[idx].input_norm_w,
                &mut scratch.normed,
                &format!("layer {idx} input norm"),
            )?;

            if config.is_full_attention(idx) {
                prefill_full_attention_layer(
                    weights, state, rotary, &mut scratch, config, idx,
                    chunk_len, chunk_start, ordinal, kv_chunk_size,
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
                    weights, state, &mut scratch, config, idx,
                    chunk_len, chunk_start, is_last_chunk,
                    chunk_recurrent[idx].as_mut().unwrap(),
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
                        .map_err(|e| anyhow::anyhow!("trace attn last_hidden alloc layer {idx}: {e}"))?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_hidden.as_ptr() as *mut c_void,
                        scratch.hidden.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace attn last_hidden copy layer {idx}: {e}"))?;
                    trace.push(
                        last_hidden
                            .to_host_bytes()
                            .map_err(|e| anyhow::anyhow!("trace attn last_hidden D2H layer {idx}: {e}"))?,
                    );
                }
            }

            // Post-attention RMSNorm
            rms_norm_rows_model(
                config,
                ordinal,
                chunk_len,
                hidden_dim,
                &scratch.hidden,
                &weights.layers[idx].post_attn_norm_w,
                &mut scratch.normed,
                &format!("layer {idx} post-attn norm"),
            )?;

            if is_last_chunk {
                if let Some(trace) = layer_post_attn_norm_trace.as_mut() {
                    let hidden_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
                    let last_token_offset = (chunk_len - 1) * hidden_bytes;
                    let last_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("trace post-attn norm alloc layer {idx}: {e}"))?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        last_normed.as_ptr() as *mut c_void,
                        scratch.normed.offset_ptr(last_token_offset),
                        hidden_bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("trace post-attn norm copy layer {idx}: {e}"))?;
                    trace.push(
                        last_normed
                            .to_host_bytes()
                            .map_err(|e| anyhow::anyhow!("trace post-attn norm D2H layer {idx}: {e}"))?,
                    );
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
                        last_swiglu
                            .to_host_bytes()
                            .map_err(|e| anyhow::anyhow!("trace mlp swiglu D2H layer {idx}: {e}"))?,
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
                    let last_token_offset =
                        (chunk_len - 1) * hidden_bytes;
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
                        last_hidden
                            .to_host_bytes()
                            .map_err(|e| anyhow::anyhow!("trace last_hidden D2H layer {idx}: {e}"))?,
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
                            let last_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
                                .map_err(|e| anyhow::anyhow!("dflash tap alloc layer {idx}: {e}"))?;
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

        }

        chunk_start += chunk_len;
    }

    // Extract logits for the last token of the final chunk. Refactored out
    // into `compute_logits_for_range` so the DFlash verify path can request
    // count=B and walk the block argmax in one shot (M3; see docs/dflash.md §6).
    let (mut logits_per_pos, normed_last) = compute_logits_for_range(
        &scratch.hidden,
        weights,
        config,
        last_chunk_len - 1,
        1,
        use_4b_kernel,
        ordinal,
    )?;
    let logits = logits_per_pos.pop().expect("count=1 produces exactly one row");
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
        linear_debug_trace,
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
        let k_contig = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
            .map_err(|e| anyhow::anyhow!("kv fp8 convert K contig layer {idx}: {e}"))?;
        let v_contig = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
            .map_err(|e| anyhow::anyhow!("kv fp8 convert V contig layer {idx}: {e}"))?;
        let cap_stride = cap * head_dim * elem_bytes;
        let contig_stride = kv_len * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                ordinal,
                k_contig.offset_ptr(h * contig_stride) as *mut std::ffi::c_void,
                bf16_k.offset_ptr(h * cap_stride),
                kv_len * head_dim * elem_bytes,
            ).map_err(|e| anyhow::anyhow!("kv fp8 convert K assemble h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                ordinal,
                v_contig.offset_ptr(h * contig_stride) as *mut std::ffi::c_void,
                bf16_v.offset_ptr(h * cap_stride),
                kv_len * head_dim * elem_bytes,
            ).map_err(|e| anyhow::anyhow!("kv fp8 convert V assemble h={h}: {e}"))?;
        }

        // Allocate FP8 cache and scale buffers with same capacity
        let fp8_cap = cap; // keep same capacity for alignment
        let mut fp8_k = GpuBuffer::zeros(ordinal, ScalarType::U8, &[1, num_kv_heads, fp8_cap, head_dim])
            .map_err(|e| anyhow::anyhow!("fp8 K alloc layer {idx}: {e}"))?;
        let mut fp8_v = GpuBuffer::zeros(ordinal, ScalarType::U8, &[1, num_kv_heads, fp8_cap, head_dim])
            .map_err(|e| anyhow::anyhow!("fp8 V alloc layer {idx}: {e}"))?;
        let mut scale_k = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, fp8_cap])
            .map_err(|e| anyhow::anyhow!("scale K alloc layer {idx}: {e}"))?;
        let mut scale_v = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, fp8_cap])
            .map_err(|e| anyhow::anyhow!("scale V alloc layer {idx}: {e}"))?;

        // Quantize using GPU kernel
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal, ScalarType::BF16,
            &k_contig, &mut fp8_k, &mut scale_k,
            num_kv_heads, kv_len, head_dim, fp8_cap, 0,
        ).map_err(|e| anyhow::anyhow!("fp8 K quant layer {idx}: {e}"))?;

        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal, ScalarType::BF16,
            &v_contig, &mut fp8_v, &mut scale_v,
            num_kv_heads, kv_len, head_dim, fp8_cap, 0,
        ).map_err(|e| anyhow::anyhow!("fp8 V quant layer {idx}: {e}"))?;

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

    // 1. Q projection
    let mut q_full = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, q_proj_dim])
        .map_err(|e| anyhow::anyhow!("q_full alloc: {e}"))?;
    matmul_proj(
        ordinal, 1, chunk_len, q_proj_dim, hidden_dim,
        &scratch.normed, &fw.q_proj_w, fw.q_proj_scale.as_ref(), fw.q_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut q_full,
        fw.q_proj_int4_scale.as_ref(), fw.q_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;

    // 2. Split Q into query and gate when present. Llama-style full attention
    // uses an ungated q_proj whose row count matches q_dim exactly.
    let mut query_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
        .map_err(|e| anyhow::anyhow!("query_buf alloc: {e}"))?;
    let mut gate_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
        .map_err(|e| anyhow::anyhow!("gate_buf alloc: {e}"))?;
    if has_attn_gate {
        prefill_ffi::split_qgate(ordinal, ScalarType::BF16, chunk_len, num_q_heads, head_dim, &q_full, &mut query_buf, &mut gate_buf)
            .map_err(|e| anyhow::anyhow!("layer {idx} Q split: {e}"))?;
    } else {
        gpu_hal::copy_d2d(
            ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_full.as_ptr(),
            chunk_len * q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q copy: {e}"))?;
    }

    // 3. K projection
    matmul_proj(
        ordinal, 1, chunk_len, kv_dim, hidden_dim,
        &scratch.normed, &fw.k_proj_w, fw.k_proj_scale.as_ref(), fw.k_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf2,
        fw.k_proj_int4_scale.as_ref(), fw.k_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;

    // 4. Q normalization
    {
        let mut q_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len * num_q_heads, head_dim])
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
        gpu_hal::copy_d2d(ordinal, query_buf.as_ptr() as *mut c_void, q_normed.as_ptr(), chunk_len * q_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} Q norm copy: {e}"))?;
    }

    // 5. K normalization
    {
        let mut k_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len * num_kv_heads, head_dim])
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
        gpu_hal::copy_d2d(ordinal, scratch.proj_buf2.as_ptr() as *mut c_void, k_normed.as_ptr(), chunk_len * kv_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} K norm copy: {e}"))?;
    }

    // 6. RoPE on query and K — use pos_offset = chunk_start for correct position indexing
    prefill_ffi::apply_rope_prefill(
        ordinal, ScalarType::BF16, chunk_len, num_q_heads, head_dim, rotary_dim,
        &rotary.cos, &rotary.sin, chunk_start, &mut query_buf,
    ).map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;
    prefill_ffi::apply_rope_prefill(
        ordinal, ScalarType::BF16, chunk_len, num_kv_heads, head_dim, rotary_dim,
        &rotary.cos, &rotary.sin, chunk_start, &mut scratch.proj_buf2,
    ).map_err(|e| anyhow::anyhow!("layer {idx} K RoPE: {e}"))?;

    // 7. V projection
    let mut v_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, kv_dim])
        .map_err(|e| anyhow::anyhow!("v_buf alloc: {e}"))?;
    matmul_proj(
        ordinal, 1, chunk_len, kv_dim, hidden_dim,
        &scratch.normed, &fw.v_proj_w, fw.v_proj_scale.as_ref(), fw.v_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut v_buf,
        fw.v_proj_int4_scale.as_ref(), fw.v_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;

    // 8. Transpose K and V to [H, chunk_len, D] for KV cache write
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, chunk_len, num_kv_heads, head_dim, &scratch.proj_buf2, &mut scratch.attn_k)
        .map_err(|e| anyhow::anyhow!("layer {idx} K transpose: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, chunk_len, num_kv_heads, head_dim, &v_buf, &mut scratch.attn_v)
        .map_err(|e| anyhow::anyhow!("layer {idx} V transpose: {e}"))?;

    // 9. Write this chunk's K/V to KV cache BEFORE attention (so attention can read from it)
    //    Always use BF16 during prefill (attention kernel expects BF16).
    //    FP8 conversion happens post-prefill via convert_kv_to_fp8().
    let ls = &mut state.layers[idx];
    ls.ensure_kv_capacity(kv_len - 1, ordinal, config, kv_chunk_size, false)
        .map_err(|e| anyhow::anyhow!("layer {idx} KV alloc: {e}"))?;

    if let Some(ref mut cache_k) = ls.kv_cache_k {
        let bytes_per_chunk_head = chunk_len * head_dim * elem_bytes;
        let cap = cache_k.shape()[2];
        let cap_stride = cap * head_dim * elem_bytes;
        let src_stride = chunk_len * head_dim * elem_bytes;
        let dst_pos_offset = chunk_start * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                ordinal,
                cache_k.offset_ptr(h * cap_stride + dst_pos_offset) as *mut c_void,
                scratch.attn_k.offset_ptr(h * src_stride),
                bytes_per_chunk_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV cache K write h={h}: {e}"))?;
        }
    }
    if let Some(ref mut cache_v) = ls.kv_cache_v {
        let bytes_per_chunk_head = chunk_len * head_dim * elem_bytes;
        let cap = cache_v.shape()[2];
        let cap_stride = cap * head_dim * elem_bytes;
        let src_stride = chunk_len * head_dim * elem_bytes;
        let dst_pos_offset = chunk_start * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                ordinal,
                cache_v.offset_ptr(h * cap_stride + dst_pos_offset) as *mut c_void,
                scratch.attn_v.offset_ptr(h * src_stride),
                bytes_per_chunk_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV cache V write h={h}: {e}"))?;
        }
    }
    if commit_kv_filled {
        ls.set_kv_filled(kv_len);
    }

    // 10. Transpose Q to [H, chunk_len, D]
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, chunk_len, num_q_heads, head_dim, &query_buf, &mut scratch.attn_q)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q transpose: {e}"))?;

    // 11. Causal attention — Q: [q_heads, chunk_len, hd], K/V: [kv_heads, kv_len, hd]
    //     The KV cache has layout [1, nkv, capacity, hd] where capacity >= kv_len.
    //     The attention kernel expects contiguous [nkv, kv_len, hd].
    //     Assemble contiguous K/V from the KV cache (handles both single and multi-chunk).
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_k_contig;
    let kv_v_contig;
    let attn_k_ref;
    let attn_v_ref;

    let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
    let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
    let cap = cache_k_ref.shape()[2];

    if cap == kv_len {
        // No padding — cache is already contiguous, use directly
        attn_k_ref = cache_k_ref;
        attn_v_ref = cache_v_ref;
    } else {
        // Capacity > kv_len — copy each head's kv_len entries into contiguous buffers
        kv_k_contig = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
            .map_err(|e| anyhow::anyhow!("kv_k_contig alloc: {e}"))?;
        kv_v_contig = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
            .map_err(|e| anyhow::anyhow!("kv_v_contig alloc: {e}"))?;
        let cap_stride = cap * head_dim * elem_bytes;
        let contig_stride = kv_len * head_dim * elem_bytes;
        let copy_bytes = kv_len * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                ordinal,
                kv_k_contig.offset_ptr(h * contig_stride) as *mut c_void,
                cache_k_ref.offset_ptr(h * cap_stride),
                copy_bytes,
            ).map_err(|e| anyhow::anyhow!("layer {idx} KV assemble K h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                ordinal,
                kv_v_contig.offset_ptr(h * contig_stride) as *mut c_void,
                cache_v_ref.offset_ptr(h * cap_stride),
                copy_bytes,
            ).map_err(|e| anyhow::anyhow!("layer {idx} KV assemble V h={h}: {e}"))?;
        }
        attn_k_ref = &kv_k_contig;
        attn_v_ref = &kv_v_contig;
    }

    prefill_ffi::full_attention_prefill(
        ordinal, ScalarType::BF16, 1, num_q_heads, num_kv_heads,
        chunk_len, kv_len, head_dim, scale, chunk_start,
        &scratch.attn_q, attn_k_ref, attn_v_ref, &mut scratch.attn_out_f32,
    ).map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

    // 12. Cast F32 → BF16
    prefill_ffi::cast(ordinal, ScalarType::F32, ScalarType::BF16, num_q_heads * chunk_len * head_dim, &scratch.attn_out_f32, &mut scratch.attn_q)
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;

    // 13. Transpose back [H, chunk_len, D] → [chunk_len, H, D] = [chunk_len, q_dim]
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, num_q_heads, chunk_len, head_dim, &scratch.attn_q, &mut scratch.proj_buf)
        .map_err(|e| anyhow::anyhow!("layer {idx} attn transpose back: {e}"))?;

    // 14. Apply attention gate only for gated-Q attention models (Qwen).
    if has_attn_gate {
        let mut gated = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, q_dim])
            .map_err(|e| anyhow::anyhow!("gated alloc: {e}"))?;
        prefill_ffi::sigmoid_mul(ordinal, ScalarType::BF16, chunk_len * q_dim, &scratch.proj_buf, &gate_buf, &mut gated)
            .map_err(|e| anyhow::anyhow!("layer {idx} gate: {e}"))?;
        gpu_hal::copy_d2d(ordinal, scratch.proj_buf.as_ptr() as *mut c_void, gated.as_ptr(), chunk_len * q_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("gated copy: {e}"))?;
    }

    // 15. O projection
    matmul_proj(
        ordinal, 1, chunk_len, hidden_dim, q_dim,
        &scratch.proj_buf, &fw.o_proj_w, fw.o_proj_scale.as_ref(), fw.o_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf2,
        fw.o_proj_int4_scale.as_ref(), fw.o_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;

    // 16. Residual
    residual_add(ordinal, chunk_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf2)
        .map_err(|e| anyhow::anyhow!("layer {idx} attention residual: {e}"))?;

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
    chunk_recurrent: &mut GpuBuffer,
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
    let key_dim = nk * khd;     // Q and K share this dimension
    let val_dim = nv * vhd;
    let qkv_dim = key_dim * 2 + val_dim;
    let z_dim = val_dim;

    // 1. QKV projection: normed [chunk, hidden] → [chunk, qkv_dim]
    matmul_proj(
        ordinal, 1, chunk_len, qkv_dim, hidden_dim,
        &scratch.normed, &lw.qkv_proj_w, lw.qkv_proj_scale.as_ref(), lw.qkv_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf,
        lw.qkv_proj_int4_scale.as_ref(), lw.qkv_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;
    if trace_linear_debug {
        let bytes = scratch
            .proj_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug qkv D2H: {e}"))?;
        let row_bytes = qkv_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        *linear_debug_trace = Some(LinearLayerDebugTrace {
            qkv: bytes[start..start + row_bytes].to_vec(),
            qkv_tail: Vec::new(),
            z: Vec::new(),
            packed: Vec::new(),
            rec_apply: Vec::new(),
            attn: Vec::new(),
            gated: Vec::new(),
            proj_out: Vec::new(),
        });
    }

    // Save last kern-1 QKV rows for conv state (inter-chunk or final decode state).
    // When chunk_len >= kern-1, use extract_conv_state directly.
    // When chunk_len < kern-1, assemble from conv_tail + current chunk's QKV.
    let pad = kern - 1;
    if chunk_len >= pad {
        // Enough rows in this chunk to extract directly
        if trace_linear_debug {
            let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
            let bytes = scratch
                .proj_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} debug qkv tail D2H: {e}"))?;
            let row_bytes = qkv_dim * ScalarType::BF16.size_in_bytes();
            let start = (chunk_len - pad) * row_bytes;
            trace.qkv_tail = bytes[start..start + pad * row_bytes].to_vec();
        }
        if is_last_chunk {
            if let Some(ref mut conv_state) = state.layers[idx].conv_state {
                prefill_ffi::extract_conv_state(
                    ordinal, ScalarType::BF16, chunk_len, qkv_dim, pad,
                    &scratch.proj_buf, conv_state,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} extract conv state: {e}"))?;
            }
        }
        if !is_last_chunk {
            prefill_ffi::extract_conv_state(
                ordinal, ScalarType::BF16, chunk_len, qkv_dim, pad,
                &scratch.proj_buf, chunk_conv_tail,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} extract conv tail: {e}"))?;
        }
    } else {
        // chunk_len < pad — assemble from previous conv_tail + current chunk's QKV.
        // Use a temp buffer to avoid aliasing issues.
        let keep_old = pad - chunk_len;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let tail_stride = pad * elem_bytes;

        let new_tail = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, pad])
            .map_err(|e| anyhow::anyhow!("new_tail alloc: {e}"))?;

        for ch in 0..qkv_dim {
            // Keep last keep_old entries from old tail
            if keep_old > 0 && chunk_start > 0 {
                let src_off = ch * tail_stride + chunk_len * elem_bytes;
                let dst_off = ch * tail_stride;
                gpu_hal::copy_d2d(
                    ordinal,
                    new_tail.offset_ptr(dst_off) as *mut c_void,
                    chunk_conv_tail.offset_ptr(src_off),
                    keep_old * elem_bytes,
                ).map_err(|e| anyhow::anyhow!("layer {idx} conv tail shift ch={ch}: {e}"))?;
            }
            // Append new QKV values
            for t in 0..chunk_len {
                let src_off = t * qkv_dim * elem_bytes + ch * elem_bytes;
                let dst_off = ch * tail_stride + (keep_old + t) * elem_bytes;
                gpu_hal::copy_d2d(
                    ordinal,
                    new_tail.offset_ptr(dst_off) as *mut c_void,
                    scratch.proj_buf.offset_ptr(src_off),
                    elem_bytes,
                ).map_err(|e| anyhow::anyhow!("layer {idx} conv tail append ch={ch} t={t}: {e}"))?;
            }
        }

        // Copy assembled tail to destination
        let total_bytes = qkv_dim * pad * elem_bytes;
        if is_last_chunk {
            if let Some(ref mut conv_state) = state.layers[idx].conv_state {
                gpu_hal::copy_d2d(ordinal, conv_state.as_ptr() as *mut c_void, new_tail.as_ptr(), total_bytes)
                    .map_err(|e| anyhow::anyhow!("layer {idx} conv state final: {e}"))?;
            }
        }
        gpu_hal::copy_d2d(ordinal, chunk_conv_tail.as_ptr() as *mut c_void, new_tail.as_ptr(), total_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} conv tail update: {e}"))?;
    }

    // 2. Z projection: normed [chunk, hidden] → [chunk, z_dim]
    matmul_proj(
        ordinal, 1, chunk_len, z_dim, hidden_dim,
        &scratch.normed, &lw.z_proj_w, lw.z_proj_scale.as_ref(), lw.z_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf2,
        lw.z_proj_int4_scale.as_ref(), lw.z_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;
    if trace_linear_debug {
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let bytes = scratch
            .proj_buf2
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug z D2H: {e}"))?;
        let row_bytes = z_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.z = bytes[start..start + row_bytes].to_vec();
    }

    // 3. B projection: normed [chunk, hidden] → [chunk, nv] (kept BF16, too small for INT4)
    let mut b_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, nv])
        .map_err(|e| anyhow::anyhow!("b_buf alloc: {e}"))?;
    matmul_proj(
        ordinal, 1, chunk_len, nv, hidden_dim,
        &scratch.normed, &lw.b_proj_w, lw.b_proj_scale.as_ref(), lw.b_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut b_buf,
        None, None, 0,
    )?;

    // 4. A projection: normed [chunk, hidden] → [chunk, nv] (kept BF16, too small for INT4)
    let mut a_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, nv])
        .map_err(|e| anyhow::anyhow!("a_buf alloc: {e}"))?;
    matmul_proj(
        ordinal, 1, chunk_len, nv, hidden_dim,
        &scratch.normed, &lw.a_proj_w, lw.a_proj_scale.as_ref(), lw.a_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut a_buf,
        None, None, 0,
    )?;

    // 5. Transpose QKV [chunk, qkv_dim] → [qkv_dim, pad+chunk] for conv input
    //    For chunk 0: pad with zeros. For chunk N>0: pad with chunk_conv_tail from previous chunk.
    let pad = kern - 1;
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

    // If not the first chunk, overwrite the zero padding with conv_tail from previous chunk
    if chunk_start > 0 {
        // conv_input is [qkv_dim, pad+chunk_len] in transposed layout
        // The first `pad` columns of each row need to be filled from chunk_conv_tail [qkv_dim, pad]
        // chunk_conv_tail is already in [qkv_dim, pad] transposed layout from extract_conv_state
        let pad_bytes = pad * ScalarType::BF16.size_in_bytes();
        let conv_input_stride = (pad + chunk_len) * ScalarType::BF16.size_in_bytes();
        let tail_stride = pad * ScalarType::BF16.size_in_bytes();
        for ch in 0..qkv_dim {
            gpu_hal::copy_d2d(
                ordinal,
                scratch.conv_input.offset_ptr(ch * conv_input_stride) as *mut c_void,
                chunk_conv_tail.offset_ptr(ch * tail_stride),
                pad_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} conv pad fill ch={ch}: {e}"))?;
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

    // Split interleaved QKV [chunk, qkv_dim] → Q, K, V
    let mut q_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, key_dim])
        .map_err(|e| anyhow::anyhow!("q_linear alloc: {e}"))?;
    let mut k_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, key_dim])
        .map_err(|e| anyhow::anyhow!("k_linear alloc: {e}"))?;
    let mut v_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, val_dim])
        .map_err(|e| anyhow::anyhow!("v_linear alloc: {e}"))?;
    prefill_ffi::split_qkv(ordinal, ScalarType::BF16, chunk_len, key_dim, val_dim, &scratch.proj_buf, &mut q_linear, &mut k_linear, &mut v_linear)
        .map_err(|e| anyhow::anyhow!("layer {idx} QKV split: {e}"))?;

    let mut q_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, key_dim])
        .map_err(|e| anyhow::anyhow!("q_linear_f32 alloc: {e}"))?;
    let mut k_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, key_dim])
        .map_err(|e| anyhow::anyhow!("k_linear_f32 alloc: {e}"))?;
    let mut v_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, val_dim])
        .map_err(|e| anyhow::anyhow!("v_linear_f32 alloc: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, chunk_len * key_dim, &q_linear, &mut q_linear_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q cast: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, chunk_len * key_dim, &k_linear, &mut k_linear_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} K cast: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, chunk_len * val_dim, &v_linear, &mut v_linear_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} V cast: {e}"))?;

    let mut q_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("q_normed alloc: {e}"))?;
    prefill_ffi::l2norm(ordinal, ScalarType::F32, chunk_len * nk, khd, 1e-6, &q_linear_f32, &mut q_normed)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q l2norm: {e}"))?;

    let q_scale = 1.0 / (khd as f32).sqrt();
    let mut q_scaled = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, key_dim])
        .map_err(|e| anyhow::anyhow!("q_scaled alloc: {e}"))?;
    prefill_ffi::mul_scalar(ordinal, ScalarType::F32, chunk_len * key_dim, q_scale, &q_normed, &mut q_scaled)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q scale: {e}"))?;

    let mut k_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("k_normed alloc: {e}"))?;
    prefill_ffi::l2norm(ordinal, ScalarType::F32, chunk_len * nk, khd, 1e-6, &k_linear_f32, &mut k_normed)
        .map_err(|e| anyhow::anyhow!("layer {idx} K l2norm: {e}"))?;

    // 9. Compute beta and g on GPU
    //    beta[h, t] = sigmoid(B[t, h]) → [nv, chunk_len]
    //    g[h, t] = -softplus(A[t, h] + dt_bias[h]) * a_log_exp[h] → [nv, chunk_len]
    let mut a_buf_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, nv])
        .map_err(|e| anyhow::anyhow!("a_buf_f32 alloc: {e}"))?;
    let mut b_buf_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, nv])
        .map_err(|e| anyhow::anyhow!("b_buf_f32 alloc: {e}"))?;
    let mut dt_bias_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv])
        .map_err(|e| anyhow::anyhow!("dt_bias_f32 alloc: {e}"))?;
    let mut a_log_exp_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv])
        .map_err(|e| anyhow::anyhow!("a_log_exp_f32 alloc: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, chunk_len * nv, &a_buf, &mut a_buf_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} A cast: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, chunk_len * nv, &b_buf, &mut b_buf_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} B cast: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, nv, &lw.dt_bias, &mut dt_bias_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} dt_bias cast: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, nv, &lw.a_log_exp, &mut a_log_exp_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} a_log_exp cast: {e}"))?;
    let mut beta_gpu = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len])
        .map_err(|e| anyhow::anyhow!("beta alloc: {e}"))?;
    let mut g_gpu = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len])
        .map_err(|e| anyhow::anyhow!("g alloc: {e}"))?;
    prefill_ffi::compute_beta_g(
        ordinal, ScalarType::F32, chunk_len, nv,
        &b_buf_f32, &a_buf_f32, &dt_bias_f32, &a_log_exp_f32,
        &mut beta_gpu, &mut g_gpu,
    ).map_err(|e| anyhow::anyhow!("layer {idx} beta/g: {e}"))?;

    // 10. Transpose Q [S, nk, khd] → [nk, S, khd] and K, V similarly
    //     If nk != nv, repeat Q and K heads to match nv (like GQA head expansion)
    let head_repeat = nv / nk;

    let q_trans = if head_repeat == 1 {
        let mut buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nk, chunk_len, khd])
            .map_err(|e| anyhow::anyhow!("q_trans alloc: {e}"))?;
        prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::F32, chunk_len, nk, khd, &q_scaled, &mut buf)
            .map_err(|e| anyhow::anyhow!("layer {idx} Q linear transpose: {e}"))?;
        buf
    } else {
        let mut expanded_shd = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, nv, khd])
            .map_err(|e| anyhow::anyhow!("q_expanded_shd alloc: {e}"))?;
        prefill_ffi::repeat_interleave_heads(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nk,
            khd,
            head_repeat,
            &q_scaled,
            &mut expanded_shd,
        )
            .map_err(|e| anyhow::anyhow!("layer {idx} Q repeat: {e}"))?;
        let mut expanded = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len, khd])
            .map_err(|e| anyhow::anyhow!("q_expanded alloc: {e}"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nv,
            khd,
            &expanded_shd,
            &mut expanded,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q expanded transpose: {e}"))?;
        expanded
    };

    let k_trans = if head_repeat == 1 {
        let mut buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nk, chunk_len, khd])
            .map_err(|e| anyhow::anyhow!("k_trans alloc: {e}"))?;
        prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::F32, chunk_len, nk, khd, &k_normed, &mut buf)
            .map_err(|e| anyhow::anyhow!("layer {idx} K linear transpose: {e}"))?;
        buf
    } else {
        let mut expanded_shd = GpuBuffer::zeros(ordinal, ScalarType::F32, &[chunk_len, nv, khd])
            .map_err(|e| anyhow::anyhow!("k_expanded_shd alloc: {e}"))?;
        prefill_ffi::repeat_interleave_heads(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nk,
            khd,
            head_repeat,
            &k_normed,
            &mut expanded_shd,
        )
            .map_err(|e| anyhow::anyhow!("layer {idx} K repeat: {e}"))?;
        let mut expanded = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len, khd])
            .map_err(|e| anyhow::anyhow!("k_expanded alloc: {e}"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            chunk_len,
            nv,
            khd,
            &expanded_shd,
            &mut expanded,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K expanded transpose: {e}"))?;
        expanded
    };

    let k_trans_mut = k_trans;

    let mut v_trans = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len, vhd])
        .map_err(|e| anyhow::anyhow!("v_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::F32, chunk_len, nv, vhd, &v_linear_f32, &mut v_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} V linear transpose: {e}"))?;

    if trace_linear_debug {
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let q_scaled_bytes = q_scaled
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug q_scaled D2H: {e}"))?;
        let k_normed_bytes = k_normed
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug k_normed D2H: {e}"))?;
        let v_linear_bytes = v_linear_f32
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug v_linear D2H: {e}"))?;
        let beta_bytes = beta_gpu
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug beta D2H: {e}"))?;
        let g_bytes = g_gpu
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
            let k_head = v_head / head_repeat;
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
            packed_equiv[out_base + 2 * khd + vhd + 1] =
                g_f32[v_head * chunk_len + last_t].exp();
        }
        trace.packed = packed_equiv
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
    }

    // 11. Delta recurrent prefill
    // Keep the recurrent scan in F32 so the decode-time recurrent state matches
    // the decode kernel's native F32 state more closely on larger models.
    let out_rows = chunk_len + khd;
    let mut recurrent_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, khd, vhd])
        .map_err(|e| anyhow::anyhow!("recurrent_f32 alloc: {e}"))?;
    let mut delta_out =
        GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, out_rows, vhd])
            .map_err(|e| anyhow::anyhow!("delta_out alloc: {e}"))?;

    prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, nv * khd * vhd, chunk_recurrent, &mut recurrent_f32)
        .map_err(|e| anyhow::anyhow!("layer {idx} recurrent init cast: {e}"))?;

    prefill_ffi::delta_recurrent_prefill(
        ordinal,
        ScalarType::F32,
        nv,       // batch_heads
        chunk_len,
        khd,
        vhd,
        &recurrent_f32,
        &q_trans,
        &k_trans_mut,
        &v_trans,
        &beta_gpu,
        &g_gpu,
        &mut delta_out,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} delta recurrent: {e}"))?;

    // 12. Extract recurrent state from delta_out into chunk_recurrent (always)
    //     and into state.recurrent_state (on last chunk only, for decode).
    let mut state_bytes_debug: Option<Vec<u8>> = None;
    {
        let state_elems = nv * khd * vhd;
        let state_f32 =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, khd, vhd])
                .map_err(|e| anyhow::anyhow!("state_f32 alloc: {e}"))?;

        let elem_bytes_f32 = ScalarType::F32.size_in_bytes();
        let state_bytes_per_head = khd * vhd * elem_bytes_f32;
        let out_stride = out_rows * vhd * elem_bytes_f32;
        let attn_offset = chunk_len * vhd * elem_bytes_f32;
        for h in 0..nv {
            let src_off = h * out_stride + attn_offset;
            let dst_off = h * state_bytes_per_head;
            gpu_hal::copy_d2d(
                ordinal,
                state_f32.offset_ptr(dst_off) as *mut c_void,
                delta_out.offset_ptr(src_off),
                state_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state extract h={h}: {e}"))?;
        }

        // Always update chunk_recurrent (BF16) for the next chunk.
        prefill_ffi::cast(ordinal, ScalarType::F32, ScalarType::BF16, state_elems, &state_f32, chunk_recurrent)
            .map_err(|e| anyhow::anyhow!("layer {idx} chunk recurrent update: {e}"))?;

        // On the last chunk, keep the decode-time recurrent state in F32.
        if is_last_chunk {
            if let Some(ref mut rec_state) = state.layers[idx].recurrent_state {
                gpu_hal::copy_d2d(
                    ordinal,
                    rec_state.as_ptr() as *mut c_void,
                    state_f32.as_ptr(),
                    state_elems * elem_bytes_f32,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state final copy: {e}"))?;
            }
        }
        if trace_linear_debug {
            state_bytes_debug = Some(
                state_f32
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} debug state_f32 D2H: {e}"))?,
            );
        }
    }

    // 13. Extract attention output: [nv, chunk_len, vhd] from delta_out
    //     Transpose [nv, S, vhd] → [S, nv, vhd] = [S, val_dim]
    let attn_output_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, chunk_len, vhd])
        .map_err(|e| anyhow::anyhow!("attn_output_f32 alloc: {e}"))?;
    let mut attn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, chunk_len, vhd])
        .map_err(|e| anyhow::anyhow!("attn_output alloc: {e}"))?;
    // Copy only the first chunk_len rows per head
    {
        let attn_bytes_per_head = chunk_len * vhd * ScalarType::F32.size_in_bytes();
        let out_stride = out_rows * vhd * ScalarType::F32.size_in_bytes();
        for h in 0..nv {
            let src_off = h * out_stride;
            let dst_off = h * attn_bytes_per_head;
            gpu_hal::copy_d2d(
                ordinal,
                attn_output_f32.offset_ptr(dst_off) as *mut c_void,
                delta_out.offset_ptr(src_off),
                attn_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} attn output extract h={h}: {e}"))?;
        }
    }
    prefill_ffi::cast(ordinal, ScalarType::F32, ScalarType::BF16, nv * chunk_len * vhd, &attn_output_f32, &mut attn_output)
        .map_err(|e| anyhow::anyhow!("layer {idx} attn output cast: {e}"))?;
    if trace_linear_debug {
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let attn_out_bytes = attn_output_f32
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug attn_output_f32 D2H: {e}"))?;
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
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let bytes = attn_output
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
    let mut z_trans = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, chunk_len, vhd])
        .map_err(|e| anyhow::anyhow!("z_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, chunk_len, nv, vhd, &scratch.proj_buf2, &mut z_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} Z transpose: {e}"))?;

    let mut norm_w_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[vhd])
        .map_err(|e| anyhow::anyhow!("norm_w_bf16 alloc: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::F32, ScalarType::BF16, vhd, &lw.norm_w, &mut norm_w_bf16)
        .map_err(|e| anyhow::anyhow!("layer {idx} norm_w cast: {e}"))?;

    let mut gated_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv * chunk_len, vhd])
        .map_err(|e| anyhow::anyhow!("gated_out alloc: {e}"))?;
    prefill_ffi::rms_norm_gated(
        ordinal,
        ScalarType::BF16,
        nv * chunk_len,
        vhd,
        config.rms_norm_eps as f32,
        &attn_output,
        &z_trans,
        &norm_w_bf16,
        &mut gated_out,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;

    let mut gated_s_first = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[chunk_len, val_dim])
        .map_err(|e| anyhow::anyhow!("gated_s_first alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, nv, chunk_len, vhd, &gated_out, &mut gated_s_first)
        .map_err(|e| anyhow::anyhow!("layer {idx} gated transpose: {e}"))?;
    if trace_linear_debug {
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let bytes = gated_s_first
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug gated D2H: {e}"))?;
        let row_bytes = val_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.gated = bytes[start..start + row_bytes].to_vec();
    }

    // 16. O projection: [S, val_dim] × out_proj_w [hidden, val_dim]^T → [S, hidden]
    matmul_proj(
        ordinal, 1, chunk_len, hidden_dim, val_dim,
        &gated_s_first, &lw.out_proj_w, lw.out_proj_scale.as_ref(), lw.out_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf2,
        lw.out_proj_int4_scale.as_ref(), lw.out_proj_int4_zero.as_ref(), weights.int4_group_size,
    )?;
    if trace_linear_debug {
        let trace = linear_debug_trace.as_mut().expect("linear debug trace missing");
        let bytes = scratch
            .proj_buf2
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} debug proj_out D2H: {e}"))?;
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let start = (chunk_len - 1) * row_bytes;
        trace.proj_out = bytes[start..start + row_bytes].to_vec();
    }

    // 17. Residual: hidden += O projection output
    residual_add(ordinal, chunk_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf2)
        .map_err(|e| anyhow::anyhow!("layer {idx} linear attn residual: {e}"))?;

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

    // gate_proj: normed [seq, hidden] × gate_w [intermediate, hidden]^T → [seq, intermediate]
    if let Some(sc) = lw.gate_proj_int8_scale.as_ref() {
        matmul_int8_mixed_host(
            ordinal,
            1,
            seq_len,
            intermediate,
            hidden_dim,
            &scratch.normed,
            weights,
            &format!("{}.layers.{idx}.mlp.gate_proj.weight", weights.weight_prefix),
            &lw.gate_proj_w,
            sc,
            &mut scratch.proj_buf,
        )?;
    } else {
        matmul_proj(
            ordinal, 1, seq_len, intermediate, hidden_dim,
            &scratch.normed, &lw.gate_proj_w, lw.gate_proj_scale.as_ref(), lw.gate_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf,
            lw.gate_proj_int4_scale.as_ref(), lw.gate_proj_int4_zero.as_ref(), weights.int4_group_size,
        )?;
    }

    // up_proj: normed [seq, hidden] × up_w [intermediate, hidden]^T → [seq, intermediate]
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
            ordinal, 1, seq_len, intermediate, hidden_dim,
            &scratch.normed, &lw.up_proj_w, lw.up_proj_scale.as_ref(), lw.up_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf2,
            lw.up_proj_int4_scale.as_ref(), lw.up_proj_int4_zero.as_ref(), weights.int4_group_size,
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

    // down_proj: mlp_buf [seq, intermediate] × down_w [hidden, intermediate]^T → [seq, hidden]
    if let Some(sc) = lw.down_proj_int8_scale.as_ref() {
        matmul_int8_mixed_host(
            ordinal,
            1,
            seq_len,
            hidden_dim,
            intermediate,
            &scratch.mlp_buf,
            weights,
            &format!("{}.layers.{idx}.mlp.down_proj.weight", weights.weight_prefix),
            &lw.down_proj_w,
            sc,
            &mut scratch.proj_buf,
        )?;
    } else {
        matmul_proj(
            ordinal, 1, seq_len, hidden_dim, intermediate,
            &scratch.mlp_buf, &lw.down_proj_w, lw.down_proj_scale.as_ref(), lw.down_proj_int8_scale.as_ref(), weights.fp8_block_size, &mut scratch.proj_buf,
            lw.down_proj_int4_scale.as_ref(), lw.down_proj_int4_zero.as_ref(), weights.int4_group_size,
        )?;
    }

    // Residual: hidden += down_proj output
    residual_add(ordinal, seq_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf)
        .map_err(|e| anyhow::anyhow!("layer {idx} MLP residual: {e}"))?;

    Ok(())
}
