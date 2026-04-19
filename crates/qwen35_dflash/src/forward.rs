//! DFlash draft forward pass.
//!
//! One call per speculative round, per the protocol in `docs/dflash.md` §5.
//! Ownership of the per-round KV cache lifecycle (append + crop rollback)
//! is M3's problem; this function takes any `past_len` and assumes the
//! caller has arranged the inputs correctly.
//!
//! The forward returns a reference to `scratch.final_hidden`
//! `[1, q_len, hidden]`. Applying `lm_head` is the caller's responsibility,
//! because the draft does not own that tensor (see §7).
//!
//! M2 supports `past_len = 0`. The signature accepts `past_len` so M3 can
//! drive it without a refactor.

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use kernel_ffi::{dflash, prefill_ffi};

use crate::rotary::RotaryTables;
use crate::state::DFlashScratch;
use crate::weights::DFlashWeights;

pub struct ForwardParams {
    /// Number of already-cached positions in the draft KV (M3; 0 for M2).
    pub past_len: usize,
    /// Number of ctx (tap) positions in the current round, 1..block_size.
    pub ctx_len: usize,
    /// Number of draft query positions (= block_size).
    pub q_len: usize,
    /// First absolute position id of the contiguous slice covering ctx+noise.
    /// This is `past_len` for M3's monotonic arange layout, or 0 for the
    /// very first round.
    pub pos_offset: usize,
}

/// Run one DFlash draft forward pass.
///
/// * `noise_embedding: [1, q_len, hidden]` — already embedded
///   `[bonus_seed, MASK, MASK, ...]`. Caller applies target.embed_tokens.
/// * `target_hidden_raw: [1, ctx_len, num_taps * hidden]` — per-ctx-position
///   concat-along-hidden of the tapped target hiddens. Un-normed.
///
/// Returns `&scratch.final_hidden` `[1, q_len, hidden]`.
pub fn forward<'a>(
    weights: &DFlashWeights,
    scratch: &'a mut DFlashScratch,
    rotary: &RotaryTables,
    noise_embedding: &GpuBuffer,
    target_hidden_raw: &GpuBuffer,
    params: ForwardParams,
) -> Result<&'a GpuBuffer, GpuError> {
    let cfg = &weights.config;
    let ordinal = scratch.ordinal;
    let dtype = ScalarType::BF16;
    let bf16_bytes = 2_usize;

    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let q_out = cfg.q_out_dim();
    let kv_out = cfg.kv_out_dim();
    let eps = cfg.rms_norm_eps as f32;
    let scale = 1.0_f32 / (hd as f32).sqrt();

    let ForwardParams { past_len, ctx_len, q_len, pos_offset } = params;
    if past_len != 0 {
        return Err(GpuError::InvalidArg(
            "dflash::forward: past_len > 0 is M3 (requires draft KV cache); not yet wired".into(),
        ));
    }
    if ctx_len == 0 || q_len == 0 {
        return Err(GpuError::InvalidArg(
            "dflash::forward: ctx_len and q_len must both be > 0".into(),
        ));
    }
    if q_len > scratch.block_size || ctx_len > scratch.block_size {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: ctx_len={ctx_len} or q_len={q_len} exceeds scratch block_size={}",
            scratch.block_size
        )));
    }
    let seq = past_len + ctx_len + q_len;
    if pos_offset + seq > rotary.max_position {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: pos_offset+seq = {} exceeds RoPE table max_position = {}",
            pos_offset + seq,
            rotary.max_position,
        )));
    }

    // ----- Per-round fuser (runs once, reused by every layer) -----
    //
    // target_hidden_raw  [1, ctx_len, num_taps*hidden]
    //   -- fc matmul -->  target_hidden_ctx  [1, ctx_len, hidden]
    //   -- hidden_norm -> target_hidden_ctx_norm  [1, ctx_len, hidden]
    prefill_ffi::matmul_rhs_transposed(
        ordinal,
        dtype,
        1,
        ctx_len,
        hidden,
        cfg.fuser_in_dim(),
        target_hidden_raw,
        &weights.fc_w,
        &mut scratch.target_hidden_ctx,
    )?;
    prefill_ffi::rms_norm_rows_plain(
        ordinal,
        dtype,
        ctx_len,
        hidden,
        eps,
        &scratch.target_hidden_ctx,
        &weights.hidden_norm_w,
        &mut scratch.target_hidden_ctx_norm,
    )?;

    // ----- Initial hidden = noise_embedding (D2D copy) -----
    let hidden_bytes = q_len * hidden * bf16_bytes;
    gpu_hal::copy_d2d(
        ordinal,
        scratch.hidden_a.as_mut_ptr(),
        noise_embedding.as_ptr(),
        hidden_bytes,
    )?;

    // ----- Per-layer loop -----
    for layer in weights.layers.iter() {
        // 1) input_layernorm (noise side only).
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            dtype,
            q_len,
            hidden,
            eps,
            &scratch.hidden_a,
            &layer.input_norm_w,
            &mut scratch.hidden_norm,
        )?;

        // 2) Concat [target_hidden_ctx_norm; hidden_norm] into norm_concat.
        //    Ctx-input already post-fuser-norm; noise-input post-input_layernorm.
        let ctx_bytes   = ctx_len * hidden * bf16_bytes;
        let noise_bytes = q_len   * hidden * bf16_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            scratch.norm_concat.as_mut_ptr(),
            scratch.target_hidden_ctx_norm.as_ptr(),
            ctx_bytes,
        )?;
        let concat_noise_dst = unsafe {
            (scratch.norm_concat.as_mut_ptr() as *mut u8).add(ctx_bytes) as *mut std::ffi::c_void
        };
        gpu_hal::copy_d2d(
            ordinal,
            concat_noise_dst,
            scratch.hidden_norm.as_ptr(),
            noise_bytes,
        )?;

        let kv_seq = ctx_len + q_len;

        // 3) Q from draft-only; K/V from concat (shared k_proj/v_proj weights).
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            q_len, q_out, hidden,
            &scratch.hidden_norm, &layer.q_proj_w, &mut scratch.q_proj,
        )?;
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            kv_seq, kv_out, hidden,
            &scratch.norm_concat, &layer.k_proj_w, &mut scratch.k_concat,
        )?;
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            kv_seq, kv_out, hidden,
            &scratch.norm_concat, &layer.v_proj_w, &mut scratch.v_concat,
        )?;

        // 4) Per-head q_norm / k_norm. These are RMSNorm over head_dim=128
        //    applied to each (position, head) slice. Layout is contiguous
        //    [positions, heads, head_dim], so row count = positions*heads,
        //    col count = head_dim. In-place is fine — the kernel reads each
        //    row's sq-sum before writing that row's output.
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal, dtype,
            q_len * nh, hd, eps,
            &mut scratch.q_proj, &layer.q_norm_w,
        )?;
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal, dtype,
            kv_seq * nkv, hd, eps,
            &mut scratch.k_concat, &layer.k_norm_w,
        )?;

        // 5) RoPE — full-dim rotary (rotary_dim = head_dim).
        //    K sees all kv_seq positions starting at pos_offset.
        //    Q sees only the last q_len positions (after ctx) —
        //    dflash.py line 24 uses cos[..., -q_len:, :].
        prefill_ffi::apply_rope_prefill(
            ordinal, dtype,
            kv_seq, nkv, hd, rotary.rotary_dim,
            &rotary.cos, &rotary.sin,
            pos_offset,
            &mut scratch.k_concat,
        )?;
        prefill_ffi::apply_rope_prefill(
            ordinal, dtype,
            q_len, nh, hd, rotary.rotary_dim,
            &rotary.cos, &rotary.sin,
            pos_offset + ctx_len,
            &mut scratch.q_proj,
        )?;

        // 6) Bidirectional attention.
        dflash::bidir_attention(
            ordinal, dtype,
            q_len, kv_seq, nh, nkv, hd, scale,
            &scratch.q_proj, &scratch.k_concat, &scratch.v_concat,
            &mut scratch.attn_out,
        )?;

        // 7) o_proj into hidden_b, residual-add into hidden_a.
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            q_len, hidden, q_out,
            &scratch.attn_out, &layer.o_proj_w, &mut scratch.hidden_b,
        )?;
        let hidden_elems = q_len * hidden;
        prefill_ffi::element_add_inplace(
            ordinal, dtype, hidden_elems,
            &mut scratch.hidden_a, &scratch.hidden_b,
        )?;

        // 8) post_attention_layernorm → gate + up → SwiGLU → down.
        prefill_ffi::rms_norm_rows_plain(
            ordinal, dtype,
            q_len, hidden, eps,
            &scratch.hidden_a, &layer.post_attn_norm_w, &mut scratch.post_attn_norm,
        )?;
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            q_len, intermediate, hidden,
            &scratch.post_attn_norm, &layer.gate_proj_w, &mut scratch.gate,
        )?;
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            q_len, intermediate, hidden,
            &scratch.post_attn_norm, &layer.up_proj_w, &mut scratch.up,
        )?;
        prefill_ffi::swiglu_mul(
            ordinal, dtype,
            q_len * intermediate,
            &scratch.gate, &scratch.up, &mut scratch.swiglu_out,
        )?;
        prefill_ffi::matmul_rhs_transposed(
            ordinal, dtype, 1,
            q_len, hidden, intermediate,
            &scratch.swiglu_out, &layer.down_proj_w, &mut scratch.hidden_b,
        )?;
        prefill_ffi::element_add_inplace(
            ordinal, dtype, hidden_elems,
            &mut scratch.hidden_a, &scratch.hidden_b,
        )?;
    }

    // ----- Final norm (before lm_head) -----
    prefill_ffi::rms_norm_rows_plain(
        ordinal,
        dtype,
        q_len,
        hidden,
        eps,
        &scratch.hidden_a,
        &weights.norm_w,
        &mut scratch.final_hidden,
    )?;

    Ok(&scratch.final_hidden)
}
