//! DFlash draft forward pass.
//!
//! One call per speculative round, per the protocol in `docs/dflash.md` §5.
//! Appends `ctx_len + q_len` positions to every layer's KV cache; the engine
//! must call [`DFlashState::crop`] after acceptance to roll back the
//! unused tail. See `DFlashState` docs for the full lifecycle.
//!
//! The forward returns a reference to `scratch.final_hidden`
//! `[1, q_len, hidden]`. Applying `lm_head` is the caller's responsibility,
//! because the draft does not own that tensor (see `docs/dflash.md` §7).

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use kernel_ffi::{dflash, prefill_ffi};

use crate::rotary::RotaryTables;
use crate::state::{DFlashScratch, DFlashState};
use crate::weights::DFlashWeights;

pub struct ForwardParams {
    /// Number of ctx (tap) positions this round, 1..block_size.
    pub ctx_len: usize,
    /// Number of draft query positions (= block_size).
    pub q_len: usize,
    /// First absolute position id of the contiguous slice covering ctx+noise.
    /// Typically equals `state.kv_filled` — the draft's contiguous arange
    /// layout (docs/dflash.md §5.3).
    pub pos_offset: usize,
}

/// Run one DFlash draft forward pass.
///
/// Inputs:
/// * `state` — persistent KV caches. Pre-call `state.kv_filled` is the count
///   of positions already cached from prior rounds; forward appends
///   `ctx_len + q_len` more, so post-call `kv_filled = pre + ctx_len + q_len`.
/// * `scratch` — per-round transient buffers.
/// * `rotary` — precomputed RoPE tables sized to cover at least
///   `pos_offset + ctx_len + q_len` positions.
/// * `noise_embedding: [1, q_len, hidden]` — already-embedded
///   `[bonus_seed, MASK, MASK, ...]`. Caller applies target.embed_tokens.
/// * `target_hidden_raw: [1, ctx_len, num_taps * hidden]` — per-ctx-position
///   concatenation of the tapped target hiddens, un-normed.
///
/// Returns `&scratch.final_hidden` `[1, q_len, hidden]`.
pub fn forward<'a>(
    weights: &DFlashWeights,
    state: &mut DFlashState,
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

    let ForwardParams { ctx_len, q_len, pos_offset } = params;
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
    if state.layers.len() != weights.layers.len() {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: state has {} layers but weights have {}",
            state.layers.len(),
            weights.layers.len(),
        )));
    }

    let past_len = state.kv_filled;
    let kv_seq = ctx_len + q_len;
    let full_seq = past_len + kv_seq;
    if full_seq > state.max_ctx {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: full_seq={full_seq} exceeds DFlashState.max_ctx={}",
            state.max_ctx,
        )));
    }
    if pos_offset + kv_seq > rotary.max_position {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: pos_offset+kv_seq = {} exceeds RoPE table max_position = {}",
            pos_offset + kv_seq,
            rotary.max_position,
        )));
    }

    // ----- Per-round fuser (runs once, reused by every layer) -----
    prefill_ffi::matmul_rhs_transposed(
        ordinal, dtype, 1,
        ctx_len, hidden, cfg.fuser_in_dim(),
        target_hidden_raw, &weights.fc_w, &mut scratch.target_hidden_ctx,
    )?;
    prefill_ffi::rms_norm_rows_plain(
        ordinal, dtype, ctx_len, hidden, eps,
        &scratch.target_hidden_ctx, &weights.hidden_norm_w,
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

    // Byte stride of one cache row: nKV * head_dim * bf16.
    let cache_row_bytes = nkv * hd * bf16_bytes;
    let append_bytes = kv_seq * cache_row_bytes;
    let past_byte_offset = past_len * cache_row_bytes;

    // ----- Per-layer loop -----
    for (idx, layer) in weights.layers.iter().enumerate() {
        let layer_kv = &mut state.layers[idx];

        // 1) input_layernorm (noise side only).
        prefill_ffi::rms_norm_rows_plain(
            ordinal, dtype, q_len, hidden, eps,
            &scratch.hidden_a, &layer.input_norm_w, &mut scratch.hidden_norm,
        )?;

        // 2) Concat [target_hidden_ctx_norm; hidden_norm] into norm_concat.
        let ctx_bytes = ctx_len * hidden * bf16_bytes;
        let noise_bytes_copy = q_len * hidden * bf16_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            scratch.norm_concat.as_mut_ptr(),
            scratch.target_hidden_ctx_norm.as_ptr(),
            ctx_bytes,
        )?;
        let concat_noise_dst = unsafe {
            (scratch.norm_concat.as_mut_ptr() as *mut u8).add(ctx_bytes)
                as *mut std::ffi::c_void
        };
        gpu_hal::copy_d2d(
            ordinal,
            concat_noise_dst,
            scratch.hidden_norm.as_ptr(),
            noise_bytes_copy,
        )?;

        // 3) Q from draft-only; K/V from concat (shared k_proj/v_proj).
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

        // 4) Per-head q_norm / k_norm (in-place over head_dim).
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal, dtype, q_len * nh, hd, eps,
            &mut scratch.q_proj, &layer.q_norm_w,
        )?;
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal, dtype, kv_seq * nkv, hd, eps,
            &mut scratch.k_concat, &layer.k_norm_w,
        )?;

        // 5) RoPE — full-dim rotary. Q at pos_offset + ctx_len; K across full
        //    kv_seq starting at pos_offset. V is not rotated (dflash.py).
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

        // 6) Append this round's K/V to the per-layer cache.
        //    Dst offset = past_len * row_bytes. Source is the current round's
        //    k_concat / v_concat buffers (post k_norm, post RoPE).
        let cache_k_dst = unsafe {
            (layer_kv.cache_k.as_mut_ptr() as *mut u8).add(past_byte_offset)
                as *mut std::ffi::c_void
        };
        gpu_hal::copy_d2d(
            ordinal,
            cache_k_dst,
            scratch.k_concat.as_ptr(),
            append_bytes,
        )?;
        let cache_v_dst = unsafe {
            (layer_kv.cache_v.as_mut_ptr() as *mut u8).add(past_byte_offset)
                as *mut std::ffi::c_void
        };
        gpu_hal::copy_d2d(
            ordinal,
            cache_v_dst,
            scratch.v_concat.as_ptr(),
            append_bytes,
        )?;

        // 7) Bidirectional attention reads the cache up to full_seq rows.
        //    Physical cache is [max_ctx, nKV, hd]; kernel only touches
        //    [0..full_seq, nKV, hd]. The stride (nKV*hd) is identical either
        //    way, so passing the cache pointer with seq_len=full_seq is safe.
        dflash::bidir_attention(
            ordinal, dtype,
            q_len, full_seq, nh, nkv, hd, scale,
            &scratch.q_proj, &layer_kv.cache_k, &layer_kv.cache_v,
            &mut scratch.attn_out,
        )?;

        // 8) o_proj into hidden_b, residual-add into hidden_a.
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

        // 9) post_attention_layernorm → gate + up → SwiGLU → down → residual.
        prefill_ffi::rms_norm_rows_plain(
            ordinal, dtype, q_len, hidden, eps,
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
            ordinal, dtype, q_len * intermediate,
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

    // Advance the fill cursor after all layers succeed.
    state.kv_filled = full_seq;

    // ----- Final norm (before lm_head) -----
    prefill_ffi::rms_norm_rows_plain(
        ordinal, dtype, q_len, hidden, eps,
        &scratch.hidden_a, &weights.norm_w, &mut scratch.final_hidden,
    )?;

    Ok(&scratch.final_hidden)
}
