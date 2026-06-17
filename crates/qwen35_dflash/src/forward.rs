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
use crate::weights::{DFlashWeights, LinearWeight};

pub struct ForwardParams {
    /// Number of ctx (tap) positions this round, 1..scratch.ctx_capacity.
    pub ctx_len: usize,
    /// Number of draft query positions (= block_size).
    pub q_len: usize,
    /// First relative position id of the contiguous slice covering ctx+noise.
    /// Lucebox's draft graph is stateless and uses positions
    /// `[0, ctx_len + q_len)`, so callers normally pass 0.
    pub pos_offset: usize,
}

#[allow(clippy::too_many_arguments)]
fn draft_matmul_rhs_transposed(
    ordinal: usize,
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &LinearWeight,
    dummy_lowbit_scale: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if rhs.logical_rows != n || rhs.logical_cols != k {
        return Err(GpuError::InvalidArg(format!(
            "dflash draft matmul shape mismatch: requested n={n} k={k}, weight has n={} k={}",
            rhs.logical_rows, rhs.logical_cols
        )));
    }
    if rhs.is_lowbit() {
        return prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            batch_elems,
            m,
            n,
            k,
            lhs,
            &rhs.weight,
            dummy_lowbit_scale,
            dummy_lowbit_scale,
            None,
            128,
            rhs.quant_type,
            out,
        );
    }
    prefill_ffi::matmul_rhs_transposed(ordinal, dtype, batch_elems, m, n, k, lhs, &rhs.weight, out)
}

/// Run one DFlash draft forward pass.
///
/// Inputs:
/// * `state` — persistent KV caches. Pre-call `state.kv_filled` is the count
///   of positions already cached from prior rounds. The current Lucebox-compatible
///   path treats the cache as scratch and overwrites it from slot 0 each call.
/// * `scratch` — per-round transient buffers.
/// * `rotary` — precomputed RoPE tables sized to cover at least
///   `pos_offset + ctx_len + q_len` positions.
/// * `noise_embedding: [1, q_len, hidden]` — already-embedded
///   `[bonus_seed, MASK, MASK, ...]`. Caller applies target.embed_tokens.
/// * `scratch.fuser_input: [1, ctx_capacity, num_taps * hidden]` — caller-filled
///   per-ctx-position concatenation of the tapped target hiddens, un-normed.
///
/// Returns `&scratch.final_hidden` `[1, q_len, hidden]`.
pub fn forward<'a>(
    weights: &DFlashWeights,
    state: &mut DFlashState,
    scratch: &'a mut DFlashScratch,
    rotary: &RotaryTables,
    noise_embedding: &GpuBuffer,
    params: ForwardParams,
) -> Result<&'a GpuBuffer, GpuError> {
    let trace = std::env::var_os("SUPERSONIC_DFLASH_TRACE").is_some();
    let profile = std::env::var_os("SUPERSONIC_DFLASH_PROFILE_DRAFT").is_some();
    let mut ms_fuser_matmul = 0.0_f64;
    let mut ms_fuser_norm = 0.0_f64;
    let mut ms_hidden_copy = 0.0_f64;
    let mut ms_input_norm = 0.0_f64;
    let mut ms_concat = 0.0_f64;
    let mut ms_q_proj = 0.0_f64;
    let mut ms_kv_proj = 0.0_f64;
    let mut ms_qk_norm = 0.0_f64;
    let mut ms_rope = 0.0_f64;
    let mut ms_cache_copy = 0.0_f64;
    let mut ms_attn = 0.0_f64;
    let mut ms_o_proj = 0.0_f64;
    let mut ms_attn_resid = 0.0_f64;
    let mut ms_post_norm = 0.0_f64;
    let mut ms_gate_up = 0.0_f64;
    let mut ms_swiglu = 0.0_f64;
    let mut ms_down = 0.0_f64;
    let mut ms_mlp_resid = 0.0_f64;
    let mut ms_final_norm = 0.0_f64;
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

    let ForwardParams {
        ctx_len,
        q_len,
        pos_offset,
    } = params;
    if ctx_len == 0 || q_len == 0 {
        return Err(GpuError::InvalidArg(
            "dflash::forward: ctx_len and q_len must both be > 0".into(),
        ));
    }
    if q_len > scratch.block_size || ctx_len > scratch.ctx_capacity {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: ctx_len={ctx_len} exceeds scratch ctx_capacity={} or q_len={q_len} exceeds block_size={}",
            scratch.ctx_capacity, scratch.block_size
        )));
    }
    if state.layers.len() != weights.layers.len() {
        return Err(GpuError::InvalidArg(format!(
            "dflash::forward: state has {} layers but weights have {}",
            state.layers.len(),
            weights.layers.len(),
        )));
    }

    let past_len = 0usize;
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
    if trace {
        eprintln!(
            "[dflash-forward] fuser ctx_len={ctx_len} q_len={q_len} hidden={hidden} fuser_in={}",
            cfg.fuser_in_dim()
        );
    }
    let t_fuser_matmul = std::time::Instant::now();
    draft_matmul_rhs_transposed(
        ordinal,
        dtype,
        1,
        ctx_len,
        hidden,
        cfg.fuser_in_dim(),
        &scratch.fuser_input,
        &weights.fc_w,
        &weights.dummy_lowbit_scale,
        &mut scratch.target_hidden_ctx,
    )?;
    if profile {
        ms_fuser_matmul += t_fuser_matmul.elapsed().as_secs_f64() * 1000.0;
    }
    if trace {
        eprintln!("[dflash-forward] fuser matmul ok");
    }
    let t_fuser_norm = std::time::Instant::now();
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
    if profile {
        ms_fuser_norm += t_fuser_norm.elapsed().as_secs_f64() * 1000.0;
    }
    if trace {
        eprintln!("[dflash-forward] fuser norm ok");
    }

    // ----- Initial hidden = noise_embedding (D2D copy) -----
    let hidden_bytes = q_len * hidden * bf16_bytes;
    let t_hidden_copy = std::time::Instant::now();
    gpu_hal::copy_d2d(
        ordinal,
        scratch.hidden_a.as_mut_ptr(),
        noise_embedding.as_ptr(),
        hidden_bytes,
    )?;
    if profile {
        ms_hidden_copy += t_hidden_copy.elapsed().as_secs_f64() * 1000.0;
    }

    // The context half of norm_concat is the same for every draft layer in
    // this round; only the noise half changes after each layer input norm.
    let ctx_bytes = ctx_len * hidden * bf16_bytes;
    let noise_bytes_copy = q_len * hidden * bf16_bytes;
    let t_concat = std::time::Instant::now();
    gpu_hal::copy_d2d(
        ordinal,
        scratch.norm_concat.as_mut_ptr(),
        scratch.target_hidden_ctx_norm.as_ptr(),
        ctx_bytes,
    )?;
    if profile {
        ms_concat += t_concat.elapsed().as_secs_f64() * 1000.0;
    }

    // Byte stride of one cache row: nKV * head_dim * bf16.
    let cache_row_bytes = nkv * hd * bf16_bytes;
    let append_bytes = kv_seq * cache_row_bytes;
    let past_byte_offset = past_len * cache_row_bytes;

    // ----- Per-layer loop -----
    for (idx, layer) in weights.layers.iter().enumerate() {
        if trace {
            eprintln!("[dflash-forward] layer {idx} start");
        }
        let layer_kv = &mut state.layers[idx];

        // 1) input_layernorm (noise side only).
        let t_input_norm = std::time::Instant::now();
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
        if profile {
            ms_input_norm += t_input_norm.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} input norm ok");
        }

        // 2) Concat [target_hidden_ctx_norm; hidden_norm] into norm_concat.
        let t_concat = std::time::Instant::now();
        let concat_noise_dst = unsafe {
            (scratch.norm_concat.as_mut_ptr() as *mut u8).add(ctx_bytes) as *mut std::ffi::c_void
        };
        gpu_hal::copy_d2d(
            ordinal,
            concat_noise_dst,
            scratch.hidden_norm.as_ptr(),
            noise_bytes_copy,
        )?;
        if profile {
            ms_concat += t_concat.elapsed().as_secs_f64() * 1000.0;
        }

        // 3) Q from draft-only; K/V from concat (shared k_proj/v_proj).
        let t_q_proj = std::time::Instant::now();
        draft_matmul_rhs_transposed(
            ordinal,
            dtype,
            1,
            q_len,
            q_out,
            hidden,
            &scratch.hidden_norm,
            &layer.q_proj_w,
            &weights.dummy_lowbit_scale,
            &mut scratch.q_proj,
        )?;
        if profile {
            ms_q_proj += t_q_proj.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} q proj ok");
        }
        let use_fused_kv = std::env::var_os("SUPERSONIC_DFLASH_DISABLE_DRAFT_FUSED_KV").is_none();
        if use_fused_kv {
            let t_kv_proj = std::time::Instant::now();
            draft_matmul_rhs_transposed(
                ordinal,
                dtype,
                1,
                kv_seq,
                2 * kv_out,
                hidden,
                &scratch.norm_concat,
                &layer.kv_proj_w,
                &weights.dummy_lowbit_scale,
                &mut scratch.kv_concat,
            )?;
            prefill_ffi::split_kv_bf16(
                ordinal,
                kv_seq,
                kv_out,
                &scratch.kv_concat,
                &mut scratch.k_concat,
                &mut scratch.v_concat,
            )?;
            if profile {
                ms_kv_proj += t_kv_proj.elapsed().as_secs_f64() * 1000.0;
            }
            if trace {
                eprintln!("[dflash-forward] layer {idx} fused kv proj ok");
            }
        } else {
            let t_kv_proj = std::time::Instant::now();
            draft_matmul_rhs_transposed(
                ordinal,
                dtype,
                1,
                kv_seq,
                kv_out,
                hidden,
                &scratch.norm_concat,
                &layer.k_proj_w,
                &weights.dummy_lowbit_scale,
                &mut scratch.k_concat,
            )?;
            if trace {
                eprintln!("[dflash-forward] layer {idx} k proj ok");
            }
            draft_matmul_rhs_transposed(
                ordinal,
                dtype,
                1,
                kv_seq,
                kv_out,
                hidden,
                &scratch.norm_concat,
                &layer.v_proj_w,
                &weights.dummy_lowbit_scale,
                &mut scratch.v_concat,
            )?;
            if profile {
                ms_kv_proj += t_kv_proj.elapsed().as_secs_f64() * 1000.0;
            }
            if trace {
                eprintln!("[dflash-forward] layer {idx} v proj ok");
            }
        }

        // 4) Per-head q_norm / k_norm (in-place over head_dim).
        let t_qk_norm = std::time::Instant::now();
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal,
            dtype,
            q_len * nh,
            hd,
            eps,
            &mut scratch.q_proj,
            &layer.q_norm_w,
        )?;
        if trace {
            eprintln!("[dflash-forward] layer {idx} q norm ok");
        }
        prefill_ffi::rms_norm_rows_plain_inplace(
            ordinal,
            dtype,
            kv_seq * nkv,
            hd,
            eps,
            &mut scratch.k_concat,
            &layer.k_norm_w,
        )?;
        if profile {
            ms_qk_norm += t_qk_norm.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} k norm ok");
        }

        // 5) RoPE — full-dim rotary. Q at pos_offset + ctx_len; K across full
        //    kv_seq starting at pos_offset. V is not rotated (dflash.py).
        let t_rope = std::time::Instant::now();
        prefill_ffi::apply_rope_prefill(
            ordinal,
            dtype,
            kv_seq,
            nkv,
            hd,
            rotary.rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_offset,
            &mut scratch.k_concat,
        )?;
        if trace {
            eprintln!("[dflash-forward] layer {idx} k rope ok");
        }
        prefill_ffi::apply_rope_prefill(
            ordinal,
            dtype,
            q_len,
            nh,
            hd,
            rotary.rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_offset + ctx_len,
            &mut scratch.q_proj,
        )?;
        if profile {
            ms_rope += t_rope.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} q rope ok");
        }

        // 6) Copy this round's K/V to the per-layer scratch cache.
        //    Lucebox rebuilds the draft graph statelessly each step; matching
        //    that, we overwrite from slot 0 rather than appending old draft rows.
        let cache_k_dst = unsafe {
            (layer_kv.cache_k.as_mut_ptr() as *mut u8).add(past_byte_offset)
                as *mut std::ffi::c_void
        };
        let t_cache_copy = std::time::Instant::now();
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
        if profile {
            ms_cache_copy += t_cache_copy.elapsed().as_secs_f64() * 1000.0;
        }

        // 7) Bidirectional attention reads the cache up to full_seq rows.
        //    Physical cache is [max_ctx, nKV, hd]; kernel only touches
        //    [0..full_seq, nKV, hd]. The stride (nKV*hd) is identical either
        //    way, so passing the cache pointer with seq_len=full_seq is safe.
        let t_attn = std::time::Instant::now();
        dflash::bidir_attention(
            ordinal,
            dtype,
            q_len,
            full_seq,
            nh,
            nkv,
            hd,
            scale,
            &scratch.q_proj,
            &layer_kv.cache_k,
            &layer_kv.cache_v,
            &mut scratch.attn_out,
        )?;
        if profile {
            ms_attn += t_attn.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} attention ok");
        }

        // 8) o_proj into hidden_b, residual-add into hidden_a.
        let t_o_proj = std::time::Instant::now();
        draft_matmul_rhs_transposed(
            ordinal,
            dtype,
            1,
            q_len,
            hidden,
            q_out,
            &scratch.attn_out,
            &layer.o_proj_w,
            &weights.dummy_lowbit_scale,
            &mut scratch.hidden_b,
        )?;
        if profile {
            ms_o_proj += t_o_proj.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} o proj ok");
        }
        let hidden_elems = q_len * hidden;
        let t_attn_resid = std::time::Instant::now();
        prefill_ffi::element_add_inplace(
            ordinal,
            dtype,
            hidden_elems,
            &mut scratch.hidden_a,
            &scratch.hidden_b,
        )?;
        if profile {
            ms_attn_resid += t_attn_resid.elapsed().as_secs_f64() * 1000.0;
        }

        // 9) post_attention_layernorm → gate + up → SwiGLU → down → residual.
        let t_post_norm = std::time::Instant::now();
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            dtype,
            q_len,
            hidden,
            eps,
            &scratch.hidden_a,
            &layer.post_attn_norm_w,
            &mut scratch.post_attn_norm,
        )?;
        if profile {
            ms_post_norm += t_post_norm.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} post norm ok");
        }
        let t_gate_up = std::time::Instant::now();
        draft_matmul_rhs_transposed(
            ordinal,
            dtype,
            1,
            q_len,
            intermediate,
            hidden,
            &scratch.post_attn_norm,
            &layer.gate_proj_w,
            &weights.dummy_lowbit_scale,
            &mut scratch.gate,
        )?;
        if trace {
            eprintln!("[dflash-forward] layer {idx} gate proj ok");
        }
        draft_matmul_rhs_transposed(
            ordinal,
            dtype,
            1,
            q_len,
            intermediate,
            hidden,
            &scratch.post_attn_norm,
            &layer.up_proj_w,
            &weights.dummy_lowbit_scale,
            &mut scratch.up,
        )?;
        if profile {
            ms_gate_up += t_gate_up.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} up proj ok");
        }
        let t_swiglu = std::time::Instant::now();
        prefill_ffi::swiglu_mul(
            ordinal,
            dtype,
            q_len * intermediate,
            &scratch.gate,
            &scratch.up,
            &mut scratch.swiglu_out,
        )?;
        if profile {
            ms_swiglu += t_swiglu.elapsed().as_secs_f64() * 1000.0;
        }
        let t_down = std::time::Instant::now();
        draft_matmul_rhs_transposed(
            ordinal,
            dtype,
            1,
            q_len,
            hidden,
            intermediate,
            &scratch.swiglu_out,
            &layer.down_proj_w,
            &weights.dummy_lowbit_scale,
            &mut scratch.hidden_b,
        )?;
        if profile {
            ms_down += t_down.elapsed().as_secs_f64() * 1000.0;
        }
        if trace {
            eprintln!("[dflash-forward] layer {idx} down proj ok");
        }
        let t_mlp_resid = std::time::Instant::now();
        prefill_ffi::element_add_inplace(
            ordinal,
            dtype,
            hidden_elems,
            &mut scratch.hidden_a,
            &scratch.hidden_b,
        )?;
        if profile {
            ms_mlp_resid += t_mlp_resid.elapsed().as_secs_f64() * 1000.0;
        }
    }

    // The draft cache is scratch in the Lucebox-compatible path.
    state.kv_filled = 0;

    // ----- Final norm (before lm_head) -----
    let t_final_norm = std::time::Instant::now();
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
    if profile {
        ms_final_norm += t_final_norm.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dflash-profile] draft forward ctx={} q={} fuser_matmul={:.2}ms fuser_norm={:.2}ms hidden_copy={:.2}ms input_norm={:.2}ms concat={:.2}ms q_proj={:.2}ms kv_proj={:.2}ms qk_norm={:.2}ms rope={:.2}ms cache_copy={:.2}ms attn={:.2}ms o_proj={:.2}ms attn_resid={:.2}ms post_norm={:.2}ms gate_up={:.2}ms swiglu={:.2}ms down={:.2}ms mlp_resid={:.2}ms final_norm={:.2}ms",
            ctx_len,
            q_len,
            ms_fuser_matmul,
            ms_fuser_norm,
            ms_hidden_copy,
            ms_input_norm,
            ms_concat,
            ms_q_proj,
            ms_kv_proj,
            ms_qk_norm,
            ms_rope,
            ms_cache_copy,
            ms_attn,
            ms_o_proj,
            ms_attn_resid,
            ms_post_norm,
            ms_gate_up,
            ms_swiglu,
            ms_down,
            ms_mlp_resid,
            ms_final_norm,
        );
    }
    if trace {
        eprintln!("[dflash-forward] final norm ok");
    }

    Ok(&scratch.final_hidden)
}
