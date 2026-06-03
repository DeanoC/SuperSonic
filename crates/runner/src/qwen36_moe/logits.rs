//! Host-side Qwen3.6-MoE logit utilities.
//!
//! The decode launch orchestration lives in `qwen36_moe_decode`; this module
//! owns CPU-side BF16 conversion helpers, host final RMSnorm + lm_head math,
//! INT4 host dequant used by bake-loading fallbacks, and token sampling.

use crate::tensor_bytes;

/// Decode a stream of BF16 little-endian bytes into F32. The oracle stores
/// BF16 as raw int16 -> bytes; this is the inverse.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "BF16 bytes must be even");
    tensor_bytes::bf16_bytes_to_f32(bytes)
}

/// Round an F32 to BF16 (RNE), returning the 16 raw bits. Matches PyTorch's
/// `.to(torch.bfloat16)` rounding: nearest-even on the lopped-off mantissa
/// bits, with the standard NaN-quieting convention.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN: keep top 16 with quiet bit set.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFFu32 + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

/// Encode a slice of F32 values to BF16 little-endian bytes (RNE).
pub fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    tensor_bytes::f32_to_bf16_bytes(vals)
}

/// Apply RMSnorm + lm_head GEMV on the host. Mirrors the multi-layer
/// oracle's tail:
///   final_normed = rms_norm(hidden, final_norm_w, eps)   # (1+w) offset
///   logits       = final_normed.to(F32) @ lm_head_w.to(F32).T
///   logits       = logits.to(BF16)
///
/// The RMSnorm uses the HuggingFace `Qwen3_5MoeRMSNorm` convention with
/// the `(1.0 + weight)` unit offset: `output * (1.0 + self.weight.float())`.
///
/// All inputs are BF16 little-endian byte streams; output is BF16
/// little-endian bytes for `vocab` logit channels.
pub fn host_final_norm_lm_head(
    hidden_bytes: &[u8],
    final_norm_w_bytes: &[u8],
    lm_head_w_bytes: &[u8],
    hidden: usize,
    vocab: usize,
    eps: f32,
) -> Vec<u8> {
    assert_eq!(hidden_bytes.len(), hidden * 2, "hidden bytes mismatch");
    assert_eq!(
        final_norm_w_bytes.len(),
        hidden * 2,
        "norm_w bytes mismatch"
    );
    assert_eq!(
        lm_head_w_bytes.len(),
        vocab * hidden * 2,
        "lm_head bytes mismatch"
    );

    // BF16-input convenience wrapper. For multi-token decode loops use
    // `host_final_norm_lm_head_f32` directly with cached F32 lm_head + norm
    // weight to avoid re-converting the large lm_head matrix per step.
    let w_f32 = bf16_bytes_to_f32(final_norm_w_bytes);
    let lm_f32 = bf16_bytes_to_f32(lm_head_w_bytes);
    host_final_norm_lm_head_f32(hidden_bytes, &w_f32, &lm_f32, hidden, vocab, eps)
}

/// Same math as [`host_final_norm_lm_head`] but with `final_norm_w` and
/// `lm_head_w` already converted to F32 by the caller.
pub fn host_final_norm_lm_head_f32(
    hidden_bytes: &[u8],
    final_norm_w_f32: &[f32],
    lm_head_w_f32: &[f32],
    hidden: usize,
    vocab: usize,
    eps: f32,
) -> Vec<u8> {
    assert_eq!(hidden_bytes.len(), hidden * 2, "hidden bytes mismatch");
    assert_eq!(final_norm_w_f32.len(), hidden, "norm_w f32 len mismatch");
    assert_eq!(
        lm_head_w_f32.len(),
        vocab * hidden,
        "lm_head f32 len mismatch"
    );

    let h_f32 = bf16_bytes_to_f32(hidden_bytes);

    // RMSnorm with the HF Qwen convention: F32 mean of squares ->
    // rsqrt(var+eps) -> elementwise mul by `(1.0 + w)`.
    let mean_sq: f32 = h_f32.iter().map(|&x| x * x).sum::<f32>() / hidden as f32;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();
    let normed: Vec<f32> = h_f32
        .iter()
        .zip(final_norm_w_f32.iter())
        .map(|(&x, &w)| x * rsqrt * (1.0 + w))
        .collect();

    let mut logits = vec![0f32; vocab];
    for v in 0..vocab {
        let row_start = v * hidden;
        let mut acc = 0f64;
        for h in 0..hidden {
            acc += lm_head_w_f32[row_start + h] as f64 * normed[h] as f64;
        }
        logits[v] = acc as f32;
    }

    f32_to_bf16_bytes(&logits)
}

/// Dequantize an INT4-packed weight slab back to BF16 bytes. Mirrors the
/// kernel's scalar path: split nibbles, apply the BF16 scale/zero tile, then
/// write a single-rounded BF16 value.
pub fn dequant_int4_to_bf16_bytes(
    packed: &[u8],
    scale_bf16: &[u8],
    zero_bf16: &[u8],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<u8> {
    assert_eq!(packed.len(), out_dim * in_dim / 2, "packed size mismatch");
    assert_eq!(
        in_dim % group_size,
        0,
        "in_dim must be divisible by group_size"
    );
    assert_eq!(
        out_dim % group_size,
        0,
        "out_dim must be divisible by group_size"
    );
    let n_row_tiles = out_dim / group_size;
    let n_col_tiles = in_dim / group_size;
    assert_eq!(
        scale_bf16.len(),
        n_row_tiles * n_col_tiles * 2,
        "scale size mismatch"
    );
    assert_eq!(
        zero_bf16.len(),
        n_row_tiles * n_col_tiles * 2,
        "zero size mismatch"
    );

    let scale = bf16_bytes_to_f32(scale_bf16);
    let zero = bf16_bytes_to_f32(zero_bf16);
    let mut out = Vec::with_capacity(out_dim * in_dim * 2);
    let half_in = in_dim / 2;
    for o in 0..out_dim {
        let row_tile = o / group_size;
        let row_base = o * half_in;
        for i in 0..in_dim {
            let col_tile = i / group_size;
            let tile_idx = row_tile * n_col_tiles + col_tile;
            let s = scale[tile_idx];
            let z = zero[tile_idx];
            let byte = packed[row_base + (i / 2)];
            let nib = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let q = nib as f32;
            let bf = f32_to_bf16_bits(q * s - z * s);
            out.push((bf & 0xFF) as u8);
            out.push((bf >> 8) as u8);
        }
    }
    out
}

fn f16_le_to_f32(bytes: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    half::f16::from_bits(bits).to_f32()
}

fn ggml_k_row_bytes(qtype: i32, cols: usize) -> usize {
    assert_eq!(cols % 256, 0, "GGML K-block cols must be divisible by 256");
    let blocks = cols / 256;
    match qtype {
        12 => blocks * 144,
        13 => blocks * 176,
        14 => blocks * 210,
        _ => panic!("unsupported GGML K-block qtype {qtype}"),
    }
}

fn ggml_q4_k_scale_min(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        ((q[j] & 63) as i32, (q[j + 4] & 63) as i32)
    } else {
        (
            ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4)) as i32,
            (((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) & 63) as i32,
        )
    }
}

fn ggml_k_dequant_scalar(packed: &[u8], qtype: i32, row: usize, col: usize, cols: usize) -> f32 {
    let block = col / 256;
    let inb = col - block * 256;
    let row_bytes = ggml_k_row_bytes(qtype, cols);
    let b = row * row_bytes
        + match qtype {
            12 => block * 144,
            13 => block * 176,
            14 => block * 210,
            _ => unreachable!(),
        };
    match qtype {
        12 => {
            let d = f16_le_to_f32(packed, b);
            let dmin = f16_le_to_f32(packed, b + 2);
            let sc = &packed[b + 4..b + 16];
            let qs = &packed[b + 16..b + 144];
            let g = inb / 64;
            let sub = (inb % 64) / 32;
            let (scale, minv) = ggml_q4_k_scale_min(2 * g + sub, sc);
            let qbyte = qs[g * 32 + (inb % 32)];
            let q = if sub != 0 {
                ((qbyte >> 4) & 0x0f) as i32
            } else {
                (qbyte & 0x0f) as i32
            };
            d * scale as f32 * q as f32 - dmin * minv as f32
        }
        13 => {
            let d = f16_le_to_f32(packed, b);
            let dmin = f16_le_to_f32(packed, b + 2);
            let sc = &packed[b + 4..b + 16];
            let qh = &packed[b + 16..b + 48];
            let ql = &packed[b + 48..b + 176];
            let g = inb / 64;
            let sub = (inb % 64) / 32;
            let idx = inb % 32;
            let (scale, minv) = ggml_q4_k_scale_min(2 * g + sub, sc);
            let qbyte = ql[g * 32 + idx];
            let lo = if sub != 0 {
                ((qbyte >> 4) & 0x0f) as i32
            } else {
                (qbyte & 0x0f) as i32
            };
            let high_mask = if sub != 0 {
                2u8 << (2 * g)
            } else {
                1u8 << (2 * g)
            };
            let hi = if qh[idx] & high_mask != 0 { 16 } else { 0 };
            d * scale as f32 * (lo + hi) as f32 - dmin * minv as f32
        }
        14 => {
            let ql = &packed[b..b + 128];
            let qh = &packed[b + 128..b + 192];
            let sc = &packed[b + 192..b + 208];
            let d = f16_le_to_f32(packed, b + 208);
            let half_idx = inb / 128;
            let pos = inb - half_idx * 128;
            let idx = pos % 32;
            let qh_byte = qh[half_idx * 32 + idx];
            let (q, scale_idx) = if pos < 32 {
                (
                    (ql[half_idx * 64 + idx] & 0x0f) as i32 | (((qh_byte >> 0) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16,
                )
            } else if pos < 64 {
                (
                    (ql[half_idx * 64 + 32 + idx] & 0x0f) as i32
                        | (((qh_byte >> 2) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 2,
                )
            } else if pos < 96 {
                (
                    ((ql[half_idx * 64 + idx] >> 4) & 0x0f) as i32
                        | (((qh_byte >> 4) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 4,
                )
            } else {
                (
                    ((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0f) as i32
                        | (((qh_byte >> 6) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 6,
                )
            };
            d * sc[scale_idx] as i8 as f32 * (q - 32) as f32
        }
        _ => unreachable!(),
    }
}

/// Dequantize raw GGML Q4_K/Q5_K/Q6_K rows to BF16 bytes for host-upload
/// paths such as the standalone Metal lm_head.
pub fn dequant_ggml_k_to_bf16_bytes(
    packed: &[u8],
    qtype: i32,
    out_dim: usize,
    in_dim: usize,
) -> Vec<u8> {
    let row_bytes = ggml_k_row_bytes(qtype, in_dim);
    assert_eq!(
        packed.len(),
        out_dim * row_bytes,
        "GGML K-block size mismatch"
    );
    let mut out = Vec::with_capacity(out_dim * in_dim * 2);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let bf = f32_to_bf16_bits(ggml_k_dequant_scalar(packed, qtype, row, col, in_dim));
            out.push((bf & 0xFF) as u8);
            out.push((bf >> 8) as u8);
        }
    }
    out
}

/// Tiny dependency-free xorshift64 RNG. Deterministic given the seed.
pub struct XorshiftRng(u64);

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        // Xorshift requires a non-zero state.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform `f32` in `[0, 1)`. 24 random bits -> IEEE single mantissa.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// Sample one token from BF16 logits with optional temperature, top-K,
/// and top-P (nucleus) filters. `temperature <= 0` falls through to
/// greedy argmax. `top_k == 0` means "no top-K cap"; `top_p >= 1.0`
/// means "no nucleus truncation".
pub fn sample_bf16_logits(
    logits_bytes: &[u8],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut XorshiftRng,
) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        return argmax_bf16_logits(logits_bytes);
    }
    let logits = bf16_bytes_to_f32(logits_bytes);
    let inv_t = 1.0 / temperature;

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v * inv_t))
        .collect();
    let k = if top_k == 0 || top_k > indexed.len() {
        indexed.len()
    } else {
        let _ = indexed.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        top_k
    };
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_logit = indexed[0].1;
    let mut exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return indexed[0].0 as u32;
    }
    for e in &mut exps {
        *e /= sum;
    }

    let nucleus_size = if top_p >= 1.0 {
        exps.len()
    } else {
        let mut cum = 0.0f32;
        let mut n = exps.len();
        for (i, &p) in exps.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                n = i + 1;
                break;
            }
        }
        n.max(1)
    };

    let nucleus_sum: f32 = exps[..nucleus_size].iter().sum();
    let r: f32 = rng.next_f32() * nucleus_sum;
    let mut cum = 0.0f32;
    for i in 0..nucleus_size {
        cum += exps[i];
        if cum >= r {
            return indexed[i].0 as u32;
        }
    }
    indexed[0].0 as u32
}

/// Greedy argmax over a BF16 logits buffer.
pub fn argmax_bf16_logits(logits_bytes: &[u8]) -> u32 {
    let logits = bf16_bytes_to_f32(logits_bytes);
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i, v)
            } else {
                (best_i, best_v)
            }
        })
        .0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_preserves_normal_values() {
        let bf16_clean = [0.0f32, 1.0, 2.0, 0.5, -3.25, 1024.0];
        for &v in &bf16_clean {
            let bytes = f32_to_bf16_bytes(&[v]);
            let roundtrip = bf16_bytes_to_f32(&bytes);
            assert_eq!(roundtrip[0], v, "bf16 roundtrip drift at {v}");
        }
    }

    #[test]
    fn rms_norm_then_lm_head_matches_naive_computation() {
        let hidden = 3usize;
        let vocab = 1usize;
        let h_bytes = f32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let w_bytes = f32_to_bf16_bytes(&[1.0, 1.0, 1.0]);
        let lm_bytes = f32_to_bf16_bytes(&[1.0, 0.0, 0.0]);
        let logits = host_final_norm_lm_head(&h_bytes, &w_bytes, &lm_bytes, hidden, vocab, 0.0);
        let logit = bf16_bytes_to_f32(&logits)[0];
        let expected = 2.0 * (3.0f32 / 14.0).sqrt();
        assert!(
            (logit - expected).abs() < 1e-2,
            "logit {logit} far from expected {expected}"
        );
    }

    #[test]
    fn rms_norm_unit_offset_zero_weight_is_identity_scale() {
        let hidden = 3usize;
        let vocab = 1usize;
        let h_bytes = f32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let w_bytes = f32_to_bf16_bytes(&[0.0, 0.0, 0.0]);
        let lm_bytes = f32_to_bf16_bytes(&[1.0, 0.0, 0.0]);
        let logits = host_final_norm_lm_head(&h_bytes, &w_bytes, &lm_bytes, hidden, vocab, 0.0);
        let logit = bf16_bytes_to_f32(&logits)[0];
        let expected = (3.0f32 / 14.0).sqrt();
        assert!(
            (logit - expected).abs() < 1e-2,
            "logit {logit} far from expected {expected}"
        );
    }

    #[test]
    fn dequant_ggml_q4_k_to_bf16_bytes_decodes_uniform_block() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
        for j in 0..4 {
            block[4 + j] = 1;
            block[8 + j] = 1;
        }
        for j in 8..12 {
            block[4 + j] = 1;
        }
        for b in &mut block[16..144] {
            *b = 0x33;
        }

        let bf16 = dequant_ggml_k_to_bf16_bytes(&block, 12, 1, 256);
        let vals = bf16_bytes_to_f32(&bf16);

        assert_eq!(vals.len(), 256);
        assert!(vals.iter().all(|v| (*v - 1.5).abs() < 1e-6));
    }
}
