//! CPU reference forward pass for the DFlash2 draft model.
//!
//! Pure-Rust reference of the draft forward (matching geo-lucebox's
//! `build_draft_graph`): fc feature fusion + RMSNorm, then 5 decoder layers of
//! GQA attention (per-head q/k RMSNorm, NEOX RoPE, causal/SWA masking) and a
//! SwiGLU MLP, then final RMSNorm. The draft shares the target's lm_head, so
//! this stops at the post-norm hidden state — the caller projects through the
//! target lm_head and argmaxes to produce draft tokens.
//!
//! This is the correctness oracle the GPU draft forward engine kernels are
//! validated against. It is intentionally simple (row-major f32, no tiling)
//! and CPU-only; it never runs in the inference hot path.

use crate::dflash::{DraftConfig, DraftWeights};
use crate::q8_0;
use crate::Error;

/// RMSNorm: `out = x / rms(x) * weight`, `rms(x) = sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<(), Error> {
    let n = x.len();
    if n == 0 || weight.len() != n || out.len() != n {
        return Err(Error::Other(format!(
            "rms_norm len mismatch: x={} w={} out={}",
            x.len(),
            weight.len(),
            out.len()
        )));
    }
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let inv = 1.0 / ((ss / n as f64).sqrt() + eps as f64);
    for i in 0..n {
        out[i] = (x[i] as f64 * inv) as f32 * weight[i];
    }
    Ok(())
}

/// Matrix-vector product: `out[i] = sum_j w[i*cols + j] * x[j]`.
pub fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        let mut acc = 0.0f64;
        let row = &w[i * cols..(i + 1) * cols];
        for j in 0..cols {
            acc += row[j] as f64 * x[j] as f64;
        }
        out[i] = acc as f32;
    }
}

/// DFlash2 dynamic depthwise conv over the nq-position draft block. For tap
/// `s` (0 = prepare, 1 = finish):
///   out[l,e] = sum_k (base[s,k,e] + dyn[s,k,e/gs,l]) * x[l-k,e]   (0 for l<k)
/// `base` is F32 `[K*K, hidden]`, `dyn` is `[nq, 2*K*groups]`, `x` is `[nq, hidden]`.
fn dyn_conv_apply(
    base: &[f32],
    dyn_: &[f32],
    x: &[f32],
    hidden: usize,
    nq: usize,
    k: usize,
    gs: usize,
    s: usize,
) -> Vec<f32> {
    let groups = hidden / gs;
    let two_kg = 2 * k * groups;
    let mut out = vec![0.0f32; nq * hidden];
    for l in 0..nq {
        for e in 0..hidden {
            let g = e / gs;
            let mut acc = 0.0f64;
            for kk in 0..k {
                let coef = base[(s * k + kk) * hidden + e] as f64
                    + dyn_[l * two_kg + (s * k + kk) * groups + g] as f64;
                let xval = if l >= kk {
                    x[(l - kk) * hidden + e] as f64
                } else {
                    0.0
                };
                acc += coef * xval;
            }
            out[l * hidden + e] = acc as f32;
        }
    }
    out
}

/// Dequantize a Q8_0 weight tensor to row-major f32. `ne0`/`ne1` follow the
/// GGUF layout: the packed rows iterate over `ne1` (slow axis), each holding
/// `ne0` weights.
pub fn dequant_weight(
    weights: &DraftWeights,
    name: &str,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>, Error> {
    let packed = weights.tensor_bytes(name)?;
    let rb = q8_0::row_bytes(ne0)?;
    if packed.len() < ne1 * rb {
        return Err(Error::Other(format!(
            "dflash dequant {name}: packed {} < {ne1} rows * {rb} B",
            packed.len()
        )));
    }
    let mut out = vec![0.0f32; ne1 * ne0];
    q8_0::decode_matrix(packed, ne1, ne0, &mut out)?;
    Ok(out)
}

/// Dequantize a rank-1 f32 tensor (a norm weight) as a copied f32 slice.
pub fn f32_weight(weights: &DraftWeights, name: &str, n: usize) -> Result<Vec<f32>, Error> {
    let bytes = weights.tensor_bytes(name)?;
    if bytes.len() != n * 4 {
        return Err(Error::Other(format!(
            "dflash f32 {name}: {} B != {n}*4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// NEOX-style RoPE over `head_dim` consecutive elements of a head.
/// `pos` is the absolute position; `freq_base` is theta. Applies the
/// interleaved (x_even, x_odd) rotation used by `GGML_ROPE_TYPE_NEOX`.
fn apply_rope_neox(vec: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    // GPT-NeoX split-half RoPE (GGML_ROPE_TYPE_NEOX): the head dimension is
    // split into [0..half) and [half..head_dim); pair i rotates (x[i], x[i+half]).
    // This matches the GPU `pfx_apply_rope_prefill_kernel` and geo-lucebox's
    // `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX, ...)`.
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f64) / (freq_base as f64).powf((2 * i) as f64 / head_dim as f64);
        let freq = theta.cos() as f32;
        let sin = theta.sin() as f32;
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * freq - x1 * sin;
        vec[i + half] = x0 * sin + x1 * freq;
    }
}

/// Run the full CPU reference draft forward pass.
///
/// Inputs:
/// - `target_hidden`: the `n_target_layers` target hidden states captured at
///   `target_layer_ids`, concatenated feature-wise: `[ctx_len][n_target_layers
///   * hidden]` f32 (row-major).
/// - `noise_embed`: `[nq][hidden]` f32 — the draft's noise embedding (the
///   target token embedding broadcast over the draft block).
/// - `positions_q`: `[nq]` absolute positions for the query (noise) tokens.
/// - `positions_k`: `[ctx_len + nq]` absolute positions for the keys.
///
/// Returns the post-final-norm hidden states `[nq][hidden]`.
pub fn draft_forward(
    w: &DraftWeights,
    cfg: &DraftConfig,
    target_hidden: &[f32],
    noise_embed: &[f32],
    positions_q: &[usize],
    positions_k: &[usize],
) -> Result<Vec<f32>, Error> {
    let hidden = cfg.hidden;
    let nq = noise_embed.len() / hidden;
    let ctx_len = if target_hidden.is_empty() {
        0
    } else {
        target_hidden.len() / (cfg.n_target_layers * hidden)
    };
    if nq == 0 || noise_embed.len() != nq * hidden {
        return Err(Error::Other(format!(
            "draft_forward: noise_embed len {} not nq*hidden (nq={nq}, hidden={hidden})",
            noise_embed.len()
        )));
    }
    if ctx_len * cfg.n_target_layers * hidden != target_hidden.len() && !target_hidden.is_empty() {
        return Err(Error::Other(format!(
            "draft_forward: target_hidden len {} != ctx_len*{}*{hidden}",
            target_hidden.len(),
            cfg.n_target_layers
        )));
    }
    if positions_q.len() != nq {
        return Err(Error::Other(format!(
            "draft_forward: positions_q len {} != nq {nq}",
            positions_q.len()
        )));
    }
    if positions_k.len() != ctx_len + nq {
        return Err(Error::Other(format!(
            "draft_forward: positions_k len {} != ctx_len+nq ({ctx_len}+{nq})",
            positions_k.len()
        )));
    }

    // Feature fusion: fc @ target_hidden -> rms_norm * hidden_norm -> target_feat
    let fc = dequant_weight(w, "dflash.fc.weight", cfg.n_target_layers * hidden, hidden)?;
    let hidden_norm = f32_weight(w, "dflash.hidden_norm.weight", hidden)?;
    let mut target_feat = vec![0.0f32; ctx_len * hidden];
    for ci in 0..ctx_len {
        matvec(
            &fc,
            &target_hidden
                [ci * cfg.n_target_layers * hidden..(ci + 1) * cfg.n_target_layers * hidden],
            hidden,
            cfg.n_target_layers * hidden,
            &mut target_feat[ci * hidden..(ci + 1) * hidden],
        );
        let mut nrm = vec![0.0f32; hidden];
        rms_norm(
            &target_feat[ci * hidden..(ci + 1) * hidden],
            &hidden_norm,
            cfg.rms_eps,
            &mut nrm,
        )?;
        target_feat[ci * hidden..(ci + 1) * hidden].copy_from_slice(&nrm);
    }

    let mut h = noise_embed.to_vec();
    for (i, layer) in w.layers.iter().enumerate() {
        draft_layer_indexed(
            cfg,
            w,
            i,
            layer,
            &mut h,
            &target_feat,
            ctx_len,
            nq,
            positions_q,
            positions_k,
        )?;
    }

    // Final norm
    let out_norm = f32_weight(w, "output_norm.weight", hidden)?;
    let mut out = vec![0.0f32; nq * hidden];
    for qi in 0..nq {
        rms_norm(
            &h[qi * hidden..(qi + 1) * hidden],
            &out_norm,
            cfg.rms_eps,
            &mut out[qi * hidden..(qi + 1) * hidden],
        )?;
    }
    Ok(out)
}

fn draft_layer_indexed(
    cfg: &DraftConfig,
    w: &DraftWeights,
    idx: usize,
    _layer: &crate::dflash::DraftLayer,
    h: &mut [f32],
    target_feat: &[f32],
    ctx_len: usize,
    nq: usize,
    positions_q: &[usize],
    positions_k: &[usize],
) -> Result<(), Error> {
    let hidden = cfg.hidden;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let inter = cfg.intermediate;
    let eps = cfg.rms_eps;
    let rope_base = cfg.rope_freq_base;
    let nm = |s: &str| format!("blk.{idx}.{s}");
    let conv_k = cfg.conv_kernel_size;
    let conv_gs = cfg.conv_group_size;
    let conv_on = conv_k > 0 && conv_gs > 0 && (hidden % conv_gs == 0);
    let conv_dyn_cols = if conv_on {
        2 * conv_k * (hidden / conv_gs)
    } else {
        0
    };

    let attn_norm = f32_weight(w, &nm("attn_norm.weight"), hidden)?;
    let mut hn = vec![0.0f32; nq * hidden];
    for qi in 0..nq {
        rms_norm(
            &h[qi * hidden..(qi + 1) * hidden],
            &attn_norm,
            eps,
            &mut hn[qi * hidden..(qi + 1) * hidden],
        )?;
    }

    // DFlash2 dynamic conv "prepare" tap (s=0) on hn. dyn_attn is projected
    // from the pre-conv hn and reused for the finish tap after the output proj.
    let mut attn_conv_base: Vec<f32> = Vec::new();
    let mut dyn_attn: Vec<f32> = Vec::new();
    if conv_on {
        attn_conv_base = f32_weight(w, &nm("attn_conv.base"), hidden * conv_k * conv_k)?;
        let attn_conv_proj =
            dequant_weight(w, &nm("attn_conv.proj.weight"), hidden, conv_dyn_cols)?;
        dyn_attn = vec![0.0f32; nq * conv_dyn_cols];
        for qi in 0..nq {
            matvec(
                &attn_conv_proj,
                &hn[qi * hidden..(qi + 1) * hidden],
                conv_dyn_cols,
                hidden,
                &mut dyn_attn[qi * conv_dyn_cols..(qi + 1) * conv_dyn_cols],
            );
        }
        hn = dyn_conv_apply(
            &attn_conv_base,
            &dyn_attn,
            &hn,
            hidden,
            nq,
            conv_k,
            conv_gs,
            0,
        );
    }
    let wq = dequant_weight(w, &nm("attn_q.weight"), hidden, q_dim)?;
    let mut q = vec![0.0f32; nq * q_dim];
    let q_norm = f32_weight(w, &nm("attn_q_norm.weight"), hd)?;
    for qi in 0..nq {
        matvec(
            &wq,
            &hn[qi * hidden..(qi + 1) * hidden],
            q_dim,
            hidden,
            &mut q[qi * q_dim..(qi + 1) * q_dim],
        );
        for h_i in 0..n_heads {
            let s = qi * q_dim + h_i * hd;
            let mut nrm = vec![0.0f32; hd];
            rms_norm(&q[s..s + hd], &q_norm, eps, &mut nrm)?;
            q[s..s + hd].copy_from_slice(&nrm);
        }
    }

    let wk = dequant_weight(w, &nm("attn_k.weight"), hidden, kv_dim)?;
    let wv = dequant_weight(w, &nm("attn_v.weight"), hidden, kv_dim)?;
    let total_k = ctx_len + nq;
    let mut k = vec![0.0f32; total_k * kv_dim];
    let mut v = vec![0.0f32; total_k * kv_dim];
    for ci in 0..ctx_len {
        matvec(
            &wk,
            &target_feat[ci * hidden..(ci + 1) * hidden],
            kv_dim,
            hidden,
            &mut k[ci * kv_dim..(ci + 1) * kv_dim],
        );
        matvec(
            &wv,
            &target_feat[ci * hidden..(ci + 1) * hidden],
            kv_dim,
            hidden,
            &mut v[ci * kv_dim..(ci + 1) * kv_dim],
        );
    }
    for ni in 0..nq {
        matvec(
            &wk,
            &hn[ni * hidden..(ni + 1) * hidden],
            kv_dim,
            hidden,
            &mut k[(ctx_len + ni) * kv_dim..],
        );
        matvec(
            &wv,
            &hn[ni * hidden..(ni + 1) * hidden],
            kv_dim,
            hidden,
            &mut v[(ctx_len + ni) * kv_dim..],
        );
    }
    let k_norm = f32_weight(w, &nm("attn_k_norm.weight"), hd)?;
    for ki in 0..total_k {
        for h_i in 0..n_kv {
            let s = ki * kv_dim + h_i * hd;
            let mut nrm = vec![0.0f32; hd];
            rms_norm(&k[s..s + hd], &k_norm, eps, &mut nrm)?;
            k[s..s + hd].copy_from_slice(&nrm);
        }
    }
    for qi in 0..nq {
        for h_i in 0..n_heads {
            let s = qi * q_dim + h_i * hd;
            apply_rope_neox(&mut q[s..s + hd], positions_q[qi], hd, rope_base);
        }
    }
    for ki in 0..total_k {
        for h_i in 0..n_kv {
            let s = ki * kv_dim + h_i * hd;
            apply_rope_neox(&mut k[s..s + hd], positions_k[ki], hd, rope_base);
        }
    }

    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut attn_out = vec![0.0f32; nq * q_dim];
    for qi in 0..nq {
        for h_i in 0..n_heads {
            let kvh = h_i / (n_heads / n_kv); // contiguous GQA (matches GPU + Qwen/Llama)
            let qs = qi * q_dim + h_i * hd;
            let mut scores = vec![0.0f64; total_k];
            let mut max_score = f64::MIN;
            for ki in 0..total_k {
                let ks = ki * kv_dim + kvh * hd;
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += q[qs + d] as f64 * k[ks + d] as f64;
                }
                let sc = dot * scale as f64;
                // Bidirectional (non-causal) attention: every query attends to
                // all keys (ctx + all noise), matching PR #35 pad_mask_full for
                // full-attention layers. The draft config has no SWA.
                scores[ki] = sc;
                if sc > max_score {
                    max_score = sc;
                }
            }
            if max_score == f64::NEG_INFINITY {
                max_score = 0.0;
            }
            let mut sumexp = 0.0f64;
            for ki in 0..total_k {
                scores[ki] = (scores[ki] - max_score).exp();
                sumexp += scores[ki];
            }
            for d in 0..hd {
                let mut acc = 0.0f64;
                for ki in 0..total_k {
                    let vs = ki * kv_dim + kvh * hd;
                    acc += scores[ki] * v[vs + d] as f64;
                }
                attn_out[qs + d] = (acc / sumexp) as f32;
            }
        }
    }
    let wo = dequant_weight(w, &nm("attn_output.weight"), q_dim, hidden)?;
    let mut attn_proj = vec![0.0f32; nq * hidden];
    for qi in 0..nq {
        matvec(
            &wo,
            &attn_out[qi * q_dim..(qi + 1) * q_dim],
            hidden,
            q_dim,
            &mut attn_proj[qi * hidden..(qi + 1) * hidden],
        );
    }
    let attn_res = if conv_on {
        dyn_conv_apply(
            &attn_conv_base,
            &dyn_attn,
            &attn_proj,
            hidden,
            nq,
            conv_k,
            conv_gs,
            1,
        )
    } else {
        attn_proj
    };
    for qi in 0..nq {
        for c in 0..hidden {
            h[qi * hidden + c] += attn_res[qi * hidden + c];
        }
    }
    let ffn_norm = f32_weight(w, &nm("ffn_norm.weight"), hidden)?;
    let w_gate = dequant_weight(w, &nm("ffn_gate.weight"), hidden, inter)?;
    let w_up = dequant_weight(w, &nm("ffn_up.weight"), hidden, inter)?;
    let w_down = dequant_weight(w, &nm("ffn_down.weight"), inter, hidden)?;
    // hf = rms_norm(h) * ffn_norm for all nq rows, then conv prepare (block).
    let mut hf = vec![0.0f32; nq * hidden];
    for qi in 0..nq {
        rms_norm(
            &h[qi * hidden..(qi + 1) * hidden],
            &ffn_norm,
            eps,
            &mut hf[qi * hidden..(qi + 1) * hidden],
        )?;
    }
    let mut ffn_conv_base: Vec<f32> = Vec::new();
    let mut dyn_ffn: Vec<f32> = Vec::new();
    if conv_on {
        ffn_conv_base = f32_weight(w, &nm("ffn_conv.base"), hidden * conv_k * conv_k)?;
        let ffn_conv_proj = dequant_weight(w, &nm("ffn_conv.proj.weight"), hidden, conv_dyn_cols)?;
        dyn_ffn = vec![0.0f32; nq * conv_dyn_cols];
        for qi in 0..nq {
            matvec(
                &ffn_conv_proj,
                &hf[qi * hidden..(qi + 1) * hidden],
                conv_dyn_cols,
                hidden,
                &mut dyn_ffn[qi * conv_dyn_cols..(qi + 1) * conv_dyn_cols],
            );
        }
        hf = dyn_conv_apply(
            &ffn_conv_base,
            &dyn_ffn,
            &hf,
            hidden,
            nq,
            conv_k,
            conv_gs,
            0,
        );
    }
    let mut ffn_out = vec![0.0f32; nq * hidden];
    for qi in 0..nq {
        let mut g = vec![0.0f32; inter];
        matvec(
            &w_gate,
            &hf[qi * hidden..(qi + 1) * hidden],
            inter,
            hidden,
            &mut g,
        );
        let mut u = vec![0.0f32; inter];
        matvec(
            &w_up,
            &hf[qi * hidden..(qi + 1) * hidden],
            inter,
            hidden,
            &mut u,
        );
        let mut gu = vec![0.0f32; inter];
        for c in 0..inter {
            let silu = 1.0 / (1.0 + (-g[c]).exp());
            gu[c] = silu * g[c] * u[c];
        }
        matvec(
            &w_down,
            &gu,
            hidden,
            inter,
            &mut ffn_out[qi * hidden..(qi + 1) * hidden],
        );
    }
    let ffn_res = if conv_on {
        dyn_conv_apply(
            &ffn_conv_base,
            &dyn_ffn,
            &ffn_out,
            hidden,
            nq,
            conv_k,
            conv_gs,
            1,
        )
    } else {
        ffn_out
    };
    for qi in 0..nq {
        for c in 0..hidden {
            h[qi * hidden + c] += ffn_res[qi * hidden + c];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dflash::load_draft;
    use std::path::PathBuf;

    fn require_artifacts() -> bool {
        std::env::var_os("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").is_some()
    }

    fn draft_path() -> Option<PathBuf> {
        let value = std::env::var_os("SUPERSONIC_DFLASH_DRAFT_GGUF")?;
        let path = PathBuf::from(value);
        if !path.is_file() {
            if require_artifacts() {
                panic!(
                    "SUPERSONIC_DFLASH_DRAFT_GGUF points to a missing drafter: {}",
                    path.display()
                );
            }
            return None;
        }
        Some(path)
    }

    /// An end-to-end forward pass must run without error and produce finite,
    /// non-all-zero hidden states. Covers single-token (nq=1) and multi-token
    /// (nq=2) cases, exercising the full pipeline: fc fusion, 5 decoder layers
    /// (GQA attention with q/k norms, RoPE, causal mask, SwiGLU), and final
    /// norm. The multi-token case pins per-row RMSNorm of the nq-row hidden
    /// state and cross-query causal masking.
    #[test]
    fn draft_forward_runs_and_produces_finite_output() {
        let Some(path) = draft_path() else {
            eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
            return;
        };
        let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));
        let cfg = weights.config.clone();
        let hidden = cfg.hidden;

        // Cover both single-token (nq=1) and multi-token (nq=2) cases. The
        // multi-token case exercises per-row RMSNorm of the nq-row hidden state
        // (the attention pre-norm) and the causal mask across query rows,
        // neither of which a single-token case can reach.
        for (ctx_len, nq) in [(1usize, 1usize), (2usize, 2usize)] {
            let target_hidden: Vec<f32> = (0..ctx_len * cfg.n_target_layers * hidden)
                .map(|i| (((i % 13) as f32) - 6.0) / 11.0)
                .collect();
            let noise_embed: Vec<f32> = (0..nq * hidden)
                .map(|i| (((i % 7) as f32) - 3.0) / 5.0)
                .collect();
            let positions_q: Vec<usize> = (0..nq).map(|i| ctx_len + i).collect();
            let positions_k: Vec<usize> = (0..ctx_len + nq).map(|i| i).collect();

            let out = draft_forward(
                &weights,
                &cfg,
                &target_hidden,
                &noise_embed,
                &positions_q,
                &positions_k,
            )
            .unwrap_or_else(|e| panic!("draft_forward(ctx={ctx_len},nq={nq}): {e}"));

            assert_eq!(
                out.len(),
                nq * hidden,
                "output length (ctx={ctx_len},nq={nq})"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "output not finite (ctx={ctx_len},nq={nq})"
            );
            let energy: f64 = out.iter().map(|v| (*v as f64).powi(2)).sum();
            assert!(
                energy > 0.0,
                "draft forward produced all zeros (ctx={ctx_len},nq={nq})"
            );
            let amax = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            assert!(
                amax < 100.0,
                "output amax {amax} implausibly large (ctx={ctx_len},nq={nq})"
            );
            eprintln!(
                "dflash_ref forward: nq={nq} ctx={ctx_len} hidden={hidden} energy={energy:.4} amax={amax:.4}"
            );
        }
    }

    /// RMSNorm of a zero vector with a unit weight is all-zero (inv_rms is
    /// finite because of eps), and a constant vector normalizes to the
    /// weight. This pins the reference math independently of the artifact.
    #[test]
    fn rms_norm_reference_is_correct() {
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        // constant 3.0 -> rms = 3.0 -> inv = 1/3 -> out = 1.0 * weight
        rms_norm(&[3.0, 3.0, 3.0, 3.0], &w, 1e-6, &mut out).unwrap();
        for v in &out {
            assert!((v - 1.0).abs() < 1e-5, "rms_norm constant {v}");
        }
        // zero input -> rms=0 -> inv=1/eps -> out = 0 (x*inv = 0)
        rms_norm(&[0.0, 0.0, 0.0, 0.0], &w, 1e-6, &mut out).unwrap();
        for v in &out {
            assert!(v.abs() < 1e-30, "rms_norm zero {v}");
        }
    }

    /// NEOX RoPE at position 0 is the identity (theta=0 -> cos=1, sin=0).
    #[test]
    fn rope_neox_position_zero_is_identity() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = v.clone();
        apply_rope_neox(&mut v, 0, 8, 10000000.0);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "rope pos0 changed value {a} vs {b}");
        }
    }
}
