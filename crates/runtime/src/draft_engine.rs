//! DFlash2 draft-model forward engine (GPU).
//!
//! Runs the DFlash2 draft forward pass on the GPU by orchestrating the
//! existing Q8_0 GEMM, RMSNorm, RoPE, GQA attention, and SwiGLU kernels —
//! the draft's geometry (32 Q heads, 8 KV heads, 128 head_dim, hidden 5120)
//! is handled by the same parameterized kernels the target prefill uses.
//!
//! The draft has no KV cache of its own: each forward recomputes the ctx+noise
//! attention from scratch (stateless), matching the geo-lucebox `build_draft_graph`
//! semantics. The draft shares the target's lm_head; this engine produces the
//! post-norm hidden state, and the caller projects through the target lm_head
//! and argmaxes to get draft tokens.
//!
//! Validated against the CPU reference (`model_store::dflash_ref::draft_forward`).

use std::sync::Mutex;

use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi;
use model_store::dflash::{DraftConfig, DraftGpuLayer, DraftGpuWeights};
use qwen38::rotary::RotaryTables;

/// Scratch buffers for one draft forward pass. Grown to the max shapes seen
/// and reused across calls. The draft is stateless (no KV cache), so these
/// hold only the transient activations of a single forward.
struct DraftScratch {
    // Feature fusion fc output [ctx_len, hidden] F32 (pre-RMSNorm).
    fc_out: GpuBuffer,
    // h (the running hidden state): [nq, hidden] F32.
    h: GpuBuffer,
    // normed: [nq, hidden] F32 (RMSNorm output).
    normed: GpuBuffer,
    // Q/K/V projections: [nq, q_dim] / [total_k, kv_dim] F32 (in [S,H,D] order).
    q_buf: GpuBuffer,
    k_buf: GpuBuffer,
    v_buf: GpuBuffer,
    // Transposed [H,S,D] layouts for attention.
    attn_q: GpuBuffer,
    attn_k: GpuBuffer,
    attn_v: GpuBuffer,
    attn_out_f32: GpuBuffer,
    // FFN: separate F32 gate, up, and SwiGLU intermediates ([nq, inter]).
    // Keeping these F32 (not BF16) is what matches the upstream F32 compute
    // type and prevents the k=inter down-projection from amplifying BF16
    // rounding error in the SwiGLU output.
    gate_buf: GpuBuffer,
    up_buf: GpuBuffer,
    gu_buf: GpuBuffer,
    // Per-head norm temp (q_norm, k_norm).
    q_normed: GpuBuffer,
    k_normed: GpuBuffer,
    // DFlash2 dynamic conv: per-block dyn projection outputs ([nq, 2*K*groups])
    // and a disjoint conv output temp ([nq, hidden]); out must not alias x.
    // dyn stays F32 (the conv kernel reads F32 dyn coefficients); conv_buf
    // is F32 (x and out are F32).
    conv_dyn_attn: GpuBuffer,
    conv_dyn_ffn: GpuBuffer,
    conv_buf: GpuBuffer,
    // F32 lhs cast temp for the fc matmul (target_hidden arrives BF16).
    mm_lhs_f32: GpuBuffer,
}

impl DraftScratch {
    fn new(ordinal: usize) -> Result<Self> {
        // Allocated lazily / grown in `ensure`. Start at size 1 to avoid
        // zero-element buffers; the ensure() calls resize as needed.
        let one = |dtype: ScalarType| -> Result<GpuBuffer> {
            GpuBuffer::alloc(ordinal, dtype, &[1]).context("draft scratch alloc")
        };
        Ok(Self {
            fc_out: one(ScalarType::F32)?,
            h: one(ScalarType::F32)?,
            normed: one(ScalarType::F32)?,
            q_buf: one(ScalarType::F32)?,
            k_buf: one(ScalarType::F32)?,
            v_buf: one(ScalarType::F32)?,
            attn_q: one(ScalarType::F32)?,
            attn_k: one(ScalarType::F32)?,
            attn_v: one(ScalarType::F32)?,
            attn_out_f32: one(ScalarType::F32)?,
            gate_buf: one(ScalarType::F32)?,
            up_buf: one(ScalarType::F32)?,
            gu_buf: one(ScalarType::F32)?,
            q_normed: one(ScalarType::F32)?,
            k_normed: one(ScalarType::F32)?,
            conv_dyn_attn: one(ScalarType::F32)?,
            conv_dyn_ffn: one(ScalarType::F32)?,
            conv_buf: one(ScalarType::F32)?,
            mm_lhs_f32: one(ScalarType::F32)?,
        })
    }
}

fn ensure(buf: &mut GpuBuffer, ordinal: usize, dtype: ScalarType, elems: usize) -> Result<()> {
    if buf.elem_count() < elems {
        *buf = GpuBuffer::alloc(ordinal, dtype, &[elems])
            .with_context(|| format!("draft ensure alloc {dtype:?} {elems}"))?;
    }
    Ok(())
}

/// The DFlash2 draft forward engine.
pub struct DraftEngine {
    ordinal: usize,
    weights: DraftGpuWeights,
    rotary: RotaryTables,
    scratch: Mutex<DraftScratch>,
    /// Selector predecessor codebook, Q8_0-packed on host for CPU dequant
    /// during chain tracing. `[vocab, row_bytes(rank)]` U8, or empty when the
    /// drafter GGUF ships no selector.
    pred_cb_host: Vec<u8>,
    /// Selector successor codebook, Q8_0-packed on host.
    succ_cb_host: Vec<u8>,
}

impl DraftEngine {
    /// Build a draft engine from loaded+uploaded GPU weights.
    ///
    /// `max_pos` caps the RoPE table size (the draft's context_length is large;
    /// callers pass the actual generation bound to keep the table small).
    pub fn new(weights: DraftGpuWeights, ordinal: usize, max_pos: usize) -> Result<Self> {
        let cfg = &weights.config;
        let rotary = RotaryTables::build_with_params_f32(
            cfg.head_dim,
            max_pos,
            cfg.rope_freq_base as f64,
            ordinal,
        )
        .context("build draft rotary tables")?;
        let scratch = DraftScratch::new(ordinal)?;
        // Download selector codebooks to host for CPU dequant during chain
        // tracing. The codebooks are Q8_0 [rank, vocab] packed U8 on the GPU;
        // the selector chain dequants individual rows on the fly. Empty when
        // the drafter GGUF ships no selector (rank == 0).
        let (pred_cb_host, succ_cb_host) = if cfg.selector_rank > 0 {
            (
                weights
                    .selector_pred_cb
                    .to_host_bytes()
                    .context("draft pred_cb D2H")?,
                weights
                    .selector_succ_cb
                    .to_host_bytes()
                    .context("draft succ_cb D2H")?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            ordinal,
            weights,
            rotary,
            scratch: Mutex::new(scratch),
            pred_cb_host,
            succ_cb_host,
        })
    }

    pub fn config(&self) -> &DraftConfig {
        &self.weights.config
    }

    /// DFlash2 selector chain: produce draft tokens using the selector's
    /// bigram correction instead of pure argmax.
    ///
    /// Matches the upstream `dflash2_select_chain`: for each drafted
    /// position (1..q_len-1) the target lm_head logits are reduced to the
    /// selector's top-K candidates, then a greedy path is traced scoring each
    /// candidate as:
    ///   score(c) = logp(c) + <pred_cb[prev] . hproj(h_pos), succ_cb[c]>
    /// starting from the seed `last_tok`. Fills `draft_tok = [last_tok,
    /// tok_1, ..., tok_{q_len-1}]`.
    ///
    /// `draft_hidden` is the post-final-norm hidden states `[q_len, hidden]`
    /// F32 on GPU (the same buffer the caller projects through lm_head).
    /// `logits` is `[q_len][vocab]` F32 on host (from the lm_head projection).
    ///
    /// Returns `None` when the drafter ships no selector (rank == 0) so the
    /// caller can fall back to pure argmax.
    pub fn select_chain(
        &self,
        draft_hidden: &GpuBuffer,
        logits: &[Vec<f32>],
        last_tok: u32,
        q_len: usize,
    ) -> Result<Option<Vec<u32>>> {
        let cfg = &self.weights.config;
        let rank = cfg.selector_rank;
        if rank == 0 || self.pred_cb_host.is_empty() || self.succ_cb_host.is_empty() {
            return Ok(None);
        }
        let top_k = cfg.selector_top_k;
        let hidden = cfg.hidden;
        let ordinal = self.ordinal;
        if q_len < 2 {
            return Ok(Some(vec![last_tok]));
        }
        let n_cand = q_len - 1;

        // 1. hproj projection on GPU: draft_hidden [q_len, hidden] @ hproj
        //    [rank, hidden] -> [q_len, rank] F32. The hproj weight is Q8_0
        //    packed [rank, hidden]; draft_matmul_q8_f32 dequantizes on device
        //    with F32 accumulation.
        let mut hproj_buf = GpuBuffer::alloc(ordinal, ScalarType::F32, &[q_len * rank])
            .context("selector hproj out alloc")?;
        self.draft_matmul_q8_f32(
            draft_hidden,
            &self.weights.selector_hproj,
            q_len,
            rank,
            hidden,
            &mut hproj_buf,
        )
        .context("selector hproj matmul")?;
        let hproj_bytes = hproj_buf.to_host_bytes().context("selector hproj D2H")?;
        let hproj_f32: Vec<f32> = hproj_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // 2. Top-K candidates + log-probs per position (positions 1..q_len-1).
        let mut cand_ids: Vec<Vec<u32>> = Vec::with_capacity(n_cand);
        let mut cand_logp: Vec<Vec<f32>> = Vec::with_capacity(n_cand);
        for pos in 1..q_len {
            let row = &logits[pos];
            // Partial selection of the top-K indices (descending by logit).
            let mut idx: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            let k = top_k.min(idx.len());
            if k > 0 && k < idx.len() {
                idx.select_nth_unstable_by(k, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            idx[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Log-softmax over the full vocab row (temperature = 1.0).
            let max_val = idx[0].1;
            let sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
            let lse = max_val + sum_exp.ln();
            cand_ids.push(idx[..k].iter().map(|&(i, _)| i as u32).collect());
            cand_logp.push(idx[..k].iter().map(|&(_, v)| v - lse).collect());
        }

        // 3. Pre-dequant succ_cb rows for all candidates.
        let row_bytes = model_store::q8_0::row_bytes(rank)
            .map_err(|e| anyhow::anyhow!("selector q8_0 row_bytes: {e}"))?;
        let mut succ_per_pos: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_cand);
        let mut succ_tmp = vec![0.0f32; rank];
        for i in 0..n_cand {
            let mut rows = Vec::with_capacity(cand_ids[i].len());
            for &cand in &cand_ids[i] {
                let off = cand as usize * row_bytes;
                model_store::q8_0::decode_row(
                    &self.succ_cb_host[off..off + row_bytes],
                    rank,
                    &mut succ_tmp,
                )
                .map_err(|e| anyhow::anyhow!("selector succ_cb dequant: {e}"))?;
                rows.push(succ_tmp.clone());
            }
            succ_per_pos.push(rows);
        }

        // 4. Greedy chain tracing.
        let mut draft_tok = vec![last_tok; q_len];
        let mut prev_tok = last_tok;
        let mut pred_row = vec![0.0f32; rank];
        for i in 0..n_cand {
            let pos = i + 1;
            let hp = &hproj_f32[pos * rank..(pos + 1) * rank];
            // Dequant pred_cb[prev_tok].
            let off = prev_tok as usize * row_bytes;
            model_store::q8_0::decode_row(
                &self.pred_cb_host[off..off + row_bytes],
                rank,
                &mut pred_row,
            )
            .map_err(|e| anyhow::anyhow!("selector pred_cb dequant: {e}"))?;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_k = 0usize;
            for k in 0..cand_ids[i].len() {
                let succ = &succ_per_pos[i][k];
                let mut dot = 0.0f32;
                for r in 0..rank {
                    dot += pred_row[r] * hp[r] * succ[r];
                }
                let score = cand_logp[i][k] + dot;
                if score > best_score {
                    best_score = score;
                    best_k = k;
                }
            }
            let chosen = cand_ids[i][best_k];
            draft_tok[pos] = chosen;
            prev_tok = chosen;
        }

        Ok(Some(draft_tok))
    }

    /// Run one draft forward pass.
    ///
    /// Inputs (all on GPU):
    /// - `target_hidden`: `[ctx_len, n_target_layers * hidden]` BF16 — the
    ///   concatenated target hidden states captured at `target_layer_ids`.
    /// - `noise_embed`: `[nq, hidden]` BF16 — the draft noise embedding.
    /// - `positions_q`: `[nq]` absolute positions for the query (noise) tokens.
    /// - `positions_k`: `[ctx_len + nq]` absolute positions for the keys.
    ///
    /// Returns the post-final-norm hidden states `[nq, hidden]` F32 on GPU.
    pub fn forward(
        &self,
        target_hidden: &GpuBuffer,
        noise_embed: &GpuBuffer,
        positions_q: &[usize],
        positions_k: &[usize],
    ) -> Result<GpuBuffer> {
        let cfg = &self.weights.config;
        let hidden = cfg.hidden;
        let nq = positions_q.len();
        let ctx_len = positions_k.len().saturating_sub(nq);
        let total_k = positions_k.len();
        let eps = cfg.rms_eps;
        let ordinal = self.ordinal;
        let mut sc = self.scratch.lock().expect("draft scratch lock");

        // ── 1. Feature fusion: fc @ target_hidden -> rms_norm * hidden_norm.
        // target_hidden arrives as BF16 [ctx_len, ntl*hidden] (captured from the
        // target prefill/verify post-MLP residual). Cast it to F32 and run the
        // F32 matmul path (scalar kernel, F32 accumulation, F32 output) so the
        // fc output never passes through a BF16 truncation — matching the
        // upstream ggml F32 compute type.
        let s: &mut DraftScratch = &mut sc;
        // fc: [hidden, ntl*hidden] packed, lhs [ctx_len, ntl*hidden] F32 -> F32 out.
        ensure(
            &mut s.mm_lhs_f32,
            ordinal,
            ScalarType::F32,
            ctx_len * cfg.n_target_layers * hidden,
        )?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            ctx_len * cfg.n_target_layers * hidden,
            target_hidden,
            &mut s.mm_lhs_f32,
        )
        .context("draft fc lhs cast bf16->f32")?;
        self.draft_matmul_q8_f32(
            &s.mm_lhs_f32,
            &self.weights.fc,
            ctx_len,
            hidden,
            cfg.n_target_layers * hidden,
            &mut s.fc_out,
        )?;
        // RMSNorm per ctx row + hidden_norm scale (F32).
        ensure(&mut s.normed, ordinal, ScalarType::F32, ctx_len * hidden)?;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            ctx_len,
            hidden,
            eps,
            &s.fc_out,
            &self.weights.hidden_norm,
            &mut s.normed,
        )
        .context("draft fc rms_norm")?;
        // target_feat (normed) for K/V projections — F32 [ctx_len, hidden].
        let target_feat = dup_buffer(ordinal, &s.normed)?;
        // ── 2. h = noise_embed (cast BF16 -> F32 into scratch h).
        ensure(&mut sc.h, ordinal, ScalarType::F32, nq * hidden)?;
        prefill_ffi::cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            nq * hidden,
            noise_embed,
            &mut sc.h,
        )
        .context("draft h init cast bf16->f32")?;

        // ── 3. Decoder layers.
        for (il, layer) in self.weights.layers.iter().enumerate() {
            self.draft_layer(
                &mut sc,
                layer,
                &target_feat,
                ctx_len,
                nq,
                total_k,
                positions_q,
                positions_k,
                il,
            )?;
        }

        // ── 4. Final norm: out = rms_norm(h) * output_norm (F32).
        ensure(&mut sc.normed, ordinal, ScalarType::F32, nq * hidden)?;
        let s: &mut DraftScratch = &mut sc;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            nq,
            hidden,
            eps,
            &s.h,
            &self.weights.output_norm,
            &mut s.normed,
        )
        .context("draft final norm")?;
        // Return only the first `nq * hidden` elements. `ensure` only grows
        // `normed` (never shrinks), so after the fc RMSNorm (ctx_len * hidden)
        // the buffer may be larger than `nq * hidden`; copying the whole
        // buffer would include stale fc-norm rows.
        let f32_elem = ScalarType::F32.size_in_bytes();
        let mut out = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * hidden])
            .context("draft out alloc")?;
        gpu_hal::copy_d2d(
            ordinal,
            out.as_mut_ptr(),
            sc.normed.as_ptr(),
            nq * hidden * f32_elem,
        )
        .context("draft out copy")?;
        Ok(out)
    }

    /// A draft decoder layer: attention + SwiGLU FFN with residuals.
    fn draft_layer(
        &self,
        sc: &mut DraftScratch,
        layer: &DraftGpuLayer,
        target_feat: &GpuBuffer,
        ctx_len: usize,
        nq: usize,
        total_k: usize,
        positions_q: &[usize],
        positions_k: &[usize],
        il: usize,
    ) -> Result<()> {
        let cfg = &self.weights.config;
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let inter = cfg.intermediate;
        let eps = cfg.rms_eps;
        let ordinal = self.ordinal;
        let f32_elem = ScalarType::F32.size_in_bytes();
        let scale = 1.0 / (hd as f32).sqrt();
        let conv_k = cfg.conv_kernel_size;
        let conv_gs = cfg.conv_group_size;
        let conv_on = conv_k > 0 && conv_gs > 0 && (hidden % conv_gs == 0);
        let conv_dyn_cols = if conv_on {
            2 * conv_k * (hidden / conv_gs)
        } else {
            0
        };

        // Attention pre-norm: hn = rms_norm(h) * attn_norm (F32)
        ensure(&mut sc.normed, ordinal, ScalarType::F32, nq * hidden)?;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            nq,
            hidden,
            eps,
            &sc.h,
            &layer.attn_norm,
            &mut sc.normed,
        )
        .with_context(|| format!("draft layer {il} attn rms_norm"))?;

        // DFlash2 dynamic conv "prepare" tap (s=0) on hn. The per-group dyn
        // coefficients are projected from the pre-conv hn and reused for the
        // finish tap after the output projection. conv_buf holds the prepared
        // hn (disjoint from normed, since the k>=1 taps read un-shifted rows).
        ensure(&mut sc.conv_buf, ordinal, ScalarType::F32, nq * hidden)?;
        if conv_on {
            ensure(
                &mut sc.conv_dyn_attn,
                ordinal,
                ScalarType::F32,
                nq * conv_dyn_cols,
            )?;
            // dyn = proj @ normed (F32 lhs -> F32 out, matching upstream ggml).
            self.draft_matmul_q8_f32(
                &sc.normed,
                &layer.attn_conv_proj,
                nq,
                conv_dyn_cols,
                hidden,
                &mut sc.conv_dyn_attn,
            )
            .with_context(|| format!("draft layer {il} attn conv dyn"))?;
            prefill_ffi::dflash_dyn_conv(
                ordinal,
                ScalarType::F32,
                hidden,
                nq,
                conv_k,
                conv_gs,
                0,
                &sc.normed,
                &layer.attn_conv_base,
                &sc.conv_dyn_attn,
                &mut sc.conv_buf,
            )
            .with_context(|| format!("draft layer {il} attn conv prepare"))?;
        } else {
            gpu_hal::copy_d2d(
                ordinal,
                sc.conv_buf.as_mut_ptr() as *mut std::ffi::c_void,
                sc.normed.as_ptr(),
                nq * hidden * f32_elem,
            )
            .with_context(|| format!("draft layer {il} attn hn copy"))?;
        }
        // Q = wq @ hn(prepared) -> [nq, q_dim] F32 (conv_buf is F32 lhs).
        ensure(&mut sc.q_buf, ordinal, ScalarType::F32, nq * q_dim)?;
        self.draft_matmul_q8_f32(&sc.conv_buf, &layer.q, nq, q_dim, hidden, &mut sc.q_buf)?;
        // K/V from target_feat (ctx) and hn (noise), concatenated along sequence.
        // K_ctx/V_ctx: wk/wv @ target_feat -> [ctx_len, kv_dim] F32
        // K_noise/V_noise: wk/wv @ hn -> [nq, kv_dim] F32
        ensure(&mut sc.k_buf, ordinal, ScalarType::F32, total_k * kv_dim)?;
        ensure(&mut sc.v_buf, ordinal, ScalarType::F32, total_k * kv_dim)?;
        // ctx part: target_feat is F32 lhs directly (F32 matmul path).
        self.draft_matmul_q8_f32(
            target_feat,
            &layer.k,
            ctx_len,
            kv_dim,
            hidden,
            &mut sc.k_buf,
        )?;
        self.draft_matmul_q8_f32(
            target_feat,
            &layer.v,
            ctx_len,
            kv_dim,
            hidden,
            &mut sc.v_buf,
        )?;
        // noise part (appended after ctx). Write into the tail of k_buf/v_buf.
        // The noise K/V use hn (the conv-prepared normed hidden = conv_buf),
        // not target_feat. conv_buf is F32 lhs directly (F32 matmul path).
        let noise_k_off = ctx_len * kv_dim;
        let mut k_noise = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * kv_dim])
            .with_context(|| format!("draft layer {il} k_noise alloc"))?;
        let mut v_noise = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * kv_dim])
            .with_context(|| format!("draft layer {il} v_noise alloc"))?;
        self.draft_matmul_q8_f32(&sc.conv_buf, &layer.k, nq, kv_dim, hidden, &mut k_noise)?;
        self.draft_matmul_q8_f32(&sc.conv_buf, &layer.v, nq, kv_dim, hidden, &mut v_noise)?;
        gpu_hal::copy_d2d(
            ordinal,
            sc.k_buf.offset_ptr(noise_k_off * f32_elem) as *mut std::ffi::c_void,
            k_noise.as_ptr(),
            nq * kv_dim * f32_elem,
        )
        .with_context(|| format!("draft layer {il} k_noise copy"))?;
        gpu_hal::copy_d2d(
            ordinal,
            sc.v_buf.offset_ptr(noise_k_off * f32_elem) as *mut std::ffi::c_void,
            v_noise.as_ptr(),
            nq * kv_dim * f32_elem,
        )
        .with_context(|| format!("draft layer {il} v_noise copy"))?;

        // Per-head Q/K RMSNorm + scale (F32). rms_norm_rows over [n_rows=nq*n_heads, hd].
        ensure(&mut sc.q_normed, ordinal, ScalarType::F32, nq * q_dim)?;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            nq * n_heads,
            hd,
            eps,
            &sc.q_buf,
            &layer.q_norm,
            &mut sc.q_normed,
        )
        .with_context(|| format!("draft layer {il} q_norm"))?;
        ensure(&mut sc.k_normed, ordinal, ScalarType::F32, total_k * kv_dim)?;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            total_k * n_kv,
            hd,
            eps,
            &sc.k_buf,
            &layer.k_norm,
            &mut sc.k_normed,
        )
        .with_context(|| format!("draft layer {il} k_norm"))?;

        // RoPE: Q over positions_q, K over positions_k (F32).
        let q_pos0 = positions_q.first().copied().unwrap_or(0);
        prefill_ffi::apply_rope_prefill(
            ordinal,
            ScalarType::F32,
            nq,
            n_heads,
            hd,
            hd,
            &self.rotary.cos,
            &self.rotary.sin,
            q_pos0,
            &mut sc.q_normed,
        )
        .with_context(|| format!("draft layer {il} q rope"))?;
        let k_pos0 = positions_k.first().copied().unwrap_or(0);
        prefill_ffi::apply_rope_prefill(
            ordinal,
            ScalarType::F32,
            total_k,
            n_kv,
            hd,
            hd,
            &self.rotary.cos,
            &self.rotary.sin,
            k_pos0,
            &mut sc.k_normed,
        )
        .with_context(|| format!("draft layer {il} k rope"))?;

        // Transpose Q/K/V from [S,H,D] to [H,S,D] for the attention kernel (F32).
        ensure(&mut sc.attn_q, ordinal, ScalarType::F32, nq * q_dim)?;
        ensure(&mut sc.attn_k, ordinal, ScalarType::F32, total_k * kv_dim)?;
        ensure(&mut sc.attn_v, ordinal, ScalarType::F32, total_k * kv_dim)?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            nq,
            n_heads,
            hd,
            &sc.q_normed,
            &mut sc.attn_q,
        )
        .with_context(|| format!("draft layer {il} q transpose"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            total_k,
            n_kv,
            hd,
            &sc.k_normed,
            &mut sc.attn_k,
        )
        .with_context(|| format!("draft layer {il} k transpose"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            total_k,
            n_kv,
            hd,
            &sc.v_buf,
            &mut sc.attn_v,
        )
        .with_context(|| format!("draft layer {il} v transpose"))?;

        // Attention: bidirectional (non-causal) within the block (F32 Q/K/V).
        ensure(&mut sc.attn_out_f32, ordinal, ScalarType::F32, nq * q_dim)?;
        prefill_ffi::full_attention_prefill(
            ordinal,
            ScalarType::F32,
            1,
            n_heads,
            n_kv,
            nq,
            total_k,
            hd,
            scale,
            q_pos0 + nq - 1,
            &sc.attn_q,
            &sc.attn_k,
            &sc.attn_v,
            &mut sc.attn_out_f32,
        )
        .with_context(|| format!("draft layer {il} attention"))?;
        // Transpose attn_out [H,S,D] -> [S,H,D] = [nq, q_dim] (F32, no cast needed).
        let mut attn_shd = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * q_dim])
            .with_context(|| format!("draft layer {il} attn_shd alloc"))?;
        prefill_ffi::transpose_shd_hsd(
            ordinal,
            ScalarType::F32,
            n_heads,
            nq,
            hd,
            &sc.attn_out_f32,
            &mut attn_shd,
        )
        .with_context(|| format!("draft layer {il} attn transpose back"))?;

        // Output projection: wo @ attn -> [nq, hidden] F32 (attn_shd is F32 lhs).
        let mut attn_out = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * hidden])
            .with_context(|| format!("draft layer {il} attn_out alloc"))?;
        self.draft_matmul_q8_f32(&attn_shd, &layer.output, nq, hidden, q_dim, &mut attn_out)?;
        // DFlash2 dynamic conv "finish" tap (s=1) on attn_out (F32), reusing dyn_attn.
        if conv_on {
            prefill_ffi::dflash_dyn_conv(
                ordinal,
                ScalarType::F32,
                hidden,
                nq,
                conv_k,
                conv_gs,
                1,
                &attn_out,
                &layer.attn_conv_base,
                &sc.conv_dyn_attn,
                &mut sc.conv_buf,
            )
            .with_context(|| format!("draft layer {il} attn conv finish"))?;
        } else {
            gpu_hal::copy_d2d(
                ordinal,
                sc.conv_buf.as_mut_ptr() as *mut std::ffi::c_void,
                attn_out.as_ptr(),
                nq * hidden * f32_elem,
            )
            .with_context(|| format!("draft layer {il} attn_out copy"))?;
        }
        // h += finished attn_out (F32 residual add)
        residual_add(ordinal, nq * hidden, &mut sc.h, &sc.conv_buf)
            .with_context(|| format!("draft layer {il} attn residual"))?;
        // ── FFN: h += down(silu(gate(hn)) * up(hn)), hn = rms_norm(h) * ffn_norm (F32)
        ensure(&mut sc.normed, ordinal, ScalarType::F32, nq * hidden)?;
        prefill_ffi::rms_norm_rows_plain(
            ordinal,
            ScalarType::F32,
            nq,
            hidden,
            eps,
            &sc.h,
            &layer.ffn_norm,
            &mut sc.normed,
        )
        .with_context(|| format!("draft layer {il} ffn rms_norm"))?;
        // DFlash2 dynamic conv "prepare" tap (s=0) on hf (F32).
        ensure(&mut sc.conv_buf, ordinal, ScalarType::F32, nq * hidden)?;
        if conv_on {
            ensure(
                &mut sc.conv_dyn_ffn,
                ordinal,
                ScalarType::F32,
                nq * conv_dyn_cols,
            )?;
            // dyn = proj @ normed (F32 lhs -> F32 out, matching upstream ggml).
            self.draft_matmul_q8_f32(
                &sc.normed,
                &layer.ffn_conv_proj,
                nq,
                conv_dyn_cols,
                hidden,
                &mut sc.conv_dyn_ffn,
            )
            .with_context(|| format!("draft layer {il} ffn conv dyn"))?;
            prefill_ffi::dflash_dyn_conv(
                ordinal,
                ScalarType::F32,
                hidden,
                nq,
                conv_k,
                conv_gs,
                0,
                &sc.normed,
                &layer.ffn_conv_base,
                &sc.conv_dyn_ffn,
                &mut sc.conv_buf,
            )
            .with_context(|| format!("draft layer {il} ffn conv prepare"))?;
        } else {
            gpu_hal::copy_d2d(
                ordinal,
                sc.conv_buf.as_mut_ptr() as *mut std::ffi::c_void,
                sc.normed.as_ptr(),
                nq * hidden * f32_elem,
            )
            .with_context(|| format!("draft layer {il} ffn hf copy"))?;
        }
        // SwiGLU FFN in F32: gate = w_gate @ hf, up = w_up @ hf (both F32
        // matmul output via the scalar F32 path), gu = silu(gate) * up
        // (F32 element-wise), then down = w_down @ gu (F32 lhs → F32 output).
        // Keeping gate, up, gu, and the down lhs at full F32 precision — the
        // upstream ggml F32 compute type — prevents the k=inter
        // down-projection from amplifying BF16 rounding error in gu.
        ensure(&mut sc.gate_buf, ordinal, ScalarType::F32, nq * inter)?;
        ensure(&mut sc.up_buf, ordinal, ScalarType::F32, nq * inter)?;
        ensure(&mut sc.gu_buf, ordinal, ScalarType::F32, nq * inter)?;
        self.draft_matmul_q8_f32(
            &sc.conv_buf,
            &layer.ffn_gate,
            nq,
            inter,
            hidden,
            &mut sc.gate_buf,
        )
        .with_context(|| format!("draft layer {il} ffn gate"))?;
        self.draft_matmul_q8_f32(
            &sc.conv_buf,
            &layer.ffn_up,
            nq,
            inter,
            hidden,
            &mut sc.up_buf,
        )
        .with_context(|| format!("draft layer {il} ffn up"))?;
        prefill_ffi::swiglu_mul(
            ordinal,
            ScalarType::F32,
            nq * inter,
            &sc.gate_buf,
            &sc.up_buf,
            &mut sc.gu_buf,
        )
        .with_context(|| format!("draft layer {il} ffn swiglu"))?;
        // down @ gu -> [nq, hidden] F32 (gu is F32 lhs, output F32).
        let mut ffn_out = GpuBuffer::alloc(ordinal, ScalarType::F32, &[nq * hidden])
            .with_context(|| format!("draft layer {il} ffn_out alloc"))?;
        self.draft_matmul_q8_f32(&sc.gu_buf, &layer.ffn_down, nq, hidden, inter, &mut ffn_out)
            .with_context(|| format!("draft layer {il} ffn down"))?;
        // DFlash2 dynamic conv "finish" tap (s=1) on ffn_out (F32), reusing dyn_ffn.
        if conv_on {
            prefill_ffi::dflash_dyn_conv(
                ordinal,
                ScalarType::F32,
                hidden,
                nq,
                conv_k,
                conv_gs,
                1,
                &ffn_out,
                &layer.ffn_conv_base,
                &sc.conv_dyn_ffn,
                &mut sc.conv_buf,
            )
            .with_context(|| format!("draft layer {il} ffn conv finish"))?;
        } else {
            gpu_hal::copy_d2d(
                ordinal,
                sc.conv_buf.as_mut_ptr() as *mut std::ffi::c_void,
                ffn_out.as_ptr(),
                nq * hidden * f32_elem,
            )
            .with_context(|| format!("draft layer {il} ffn_out copy"))?;
        }
        residual_add(ordinal, nq * hidden, &mut sc.h, &sc.conv_buf)
            .with_context(|| format!("draft layer {il} ffn residual"))?;
        Ok(())
    }

    /// Q8_0 dequant matmul for the DFlash2 draft forward: F32 lhs → F32
    /// output with F32 accumulation, dispatching the scalar Q8_0
    /// dequant-matmul kernel (the WMMA path stores BF16 only). This matches
    /// the upstream ggml F32 compute type — the draft activations never
    /// pass through a BF16 truncation. `lhs` must be F32; the FFI rejects a
    /// dtype mismatch.
    fn draft_matmul_q8_f32(
        &self,
        lhs_f32: &GpuBuffer,
        rhs: &GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        out: &mut GpuBuffer,
    ) -> Result<()> {
        let qtype = qwen38::weights::LOWBIT_GGML_Q8_0;
        let ordinal = self.ordinal;
        ensure(out, ordinal, ScalarType::F32, m * n)?;
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal, 1, m, n, k, lhs_f32, rhs, rhs, rhs, None, 0, qtype, out,
        )
        .context("draft_matmul_q8_f32")?;
        Ok(())
    }
}

/// In-place residual add: `dst[i] += src[i]` via the GPU `element_add` kernel.
/// Aliases `dst` as both the lhs input and the output — safe because the
/// kernel reads `lhs[i]` and `src[i]` before writing `out[i]` per element.
/// Runs at F32 to avoid BF16 truncation at each residual add.
fn residual_add(
    ordinal: usize,
    total_elems: usize,
    dst: &mut GpuBuffer,
    src: &GpuBuffer,
) -> Result<()> {
    let lhs: &GpuBuffer = unsafe { &*(dst as *const GpuBuffer) };
    prefill_ffi::element_add(ordinal, ScalarType::F32, total_elems, lhs, src, dst)
        .map_err(|e| anyhow::anyhow!("draft residual_add: {e}"))?;
    Ok(())
}

/// Allocate a new GPU buffer and copy `src` into it (device-to-device).
fn dup_buffer(ordinal: usize, src: &GpuBuffer) -> Result<GpuBuffer> {
    let mut dst =
        GpuBuffer::alloc(ordinal, src.dtype(), src.shape()).context("dup_buffer alloc")?;
    gpu_hal::copy_d2d(ordinal, dst.as_mut_ptr(), src.as_ptr(), src.len_bytes())
        .context("dup_buffer copy")?;
    Ok(dst)
}
