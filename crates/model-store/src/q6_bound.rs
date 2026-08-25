//! Offline helpers for studying conservative Q6_K/Q8_1 output-head bounds.
//!
//! These routines deliberately do not participate in model loading or runtime
//! dispatch. They expose the two weight interpretations relevant to the bound:
//! the raw Q6_K scale expression and that value truncated to BF16 as consumed
//! by SuperSonic's baseline small-M WMMA path.

use std::collections::BTreeSet;

use half::{bf16, f16};

pub const Q6_K_VALUES: usize = 256;
pub const Q6_K_BYTES: usize = 210;
pub const Q8_1_VALUES: usize = 32;
pub const SCALAR_HEAD_ACCUMULATION_DEPTH: usize = 165;

#[derive(Debug, Clone)]
pub struct DecodedQ6Block {
    pub d: f32,
    pub quants: [i8; Q6_K_VALUES],
    pub scales: [i8; Q6_K_VALUES],
    pub raw: [f32; Q6_K_VALUES],
    pub baseline_bf16: [f32; Q6_K_VALUES],
}

#[derive(Debug, Clone, Copy)]
pub struct WeightBlockNorms {
    pub w_l2: f64,
    pub d_l2: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ActivationBlockNorms {
    pub e_l2: f64,
    pub a_l2: f64,
    pub x_l2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactTileSelection {
    pub winner: usize,
    pub rows_not_excludable: usize,
    pub exact_tiles_required: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileCountSummary {
    pub p50: usize,
    pub p95: usize,
    pub p99: usize,
    pub max: usize,
    pub fallback_count: usize,
}

fn bf16_trunc(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xffff_0000)
}

fn upward_f32(value: f64) -> f32 {
    let mut encoded = value as f32;
    if f64::from(encoded) < value {
        let bits = encoded.to_bits();
        encoded = if encoded.is_sign_negative() {
            f32::from_bits(bits - 1)
        } else {
            f32::from_bits(bits + 1)
        };
    }
    encoded
}

fn upper_bf16(center: f64, radius: f64) -> f32 {
    bf16::from_f32(upward_f32(center + radius)).to_f32()
}

/// Project exact row tiles for one correction pass after the proposal tile.
///
/// The proposal winner's tile is recomputed first. Every other tile containing
/// a row that its interval cannot exclude against that exact tile winner is
/// then included. BF16 argmax keeps the lower row on ties, so a lower row is
/// excluded only when its upper endpoint is strictly lower; a higher row may
/// also be excluded when its endpoint is equal.
pub fn required_exact_tiles(
    exact_logits: &[f32],
    center: &[f64],
    radius: &[f64],
    tile_rows: usize,
) -> Result<ExactTileSelection, String> {
    if exact_logits.is_empty()
        || exact_logits.len() != center.len()
        || exact_logits.len() != radius.len()
        || tile_rows == 0
    {
        return Err("tile selection requires equal non-empty rows and a non-zero tile size".into());
    }
    if exact_logits.iter().any(|value| !value.is_finite())
        || center.iter().any(|value| !value.is_finite())
        || radius
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("tile selection inputs must be finite with non-negative radii".into());
    }
    let winner = exact_logits
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .0;
    let proposal_winner = center
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, candidate| {
            let value = bf16::from_f32(candidate.1 as f32).to_f32();
            if value > best.1 {
                (candidate.0, value)
            } else {
                best
            }
        })
        .0;
    let proposal_tile = proposal_winner / tile_rows;
    let tile_start = proposal_tile * tile_rows;
    let tile_end = (tile_start + tile_rows).min(exact_logits.len());
    let (initial_winner, initial_winner_value) = exact_logits[tile_start..tile_end]
        .iter()
        .copied()
        .enumerate()
        .fold((tile_start, f32::NEG_INFINITY), |best, candidate| {
            let candidate = (tile_start + candidate.0, candidate.1);
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        });
    let mut tiles = BTreeSet::from([proposal_tile]);
    let mut rows_not_excludable = 0usize;
    for row in 0..exact_logits.len() {
        if row / tile_rows == proposal_tile {
            continue;
        }
        let upper = upper_bf16(center[row], radius[row]);
        let excluded = if row < initial_winner {
            upper < initial_winner_value
        } else {
            upper <= initial_winner_value
        };
        if !excluded {
            rows_not_excludable += 1;
            tiles.insert(row / tile_rows);
        }
    }
    Ok(ExactTileSelection {
        winner,
        rows_not_excludable,
        exact_tiles_required: tiles.len(),
    })
}

/// Summarize exact-tile counts with the nearest-rank percentile definition.
pub fn summarize_tile_counts(
    counts: &[usize],
    fallback_limit: usize,
) -> Result<TileCountSummary, String> {
    if counts.is_empty() {
        return Err("tile counts must be non-empty".into());
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let percentile = |fraction: f64| {
        let rank = (sorted.len() as f64 * fraction).ceil() as usize;
        sorted[rank.saturating_sub(1)]
    };
    Ok(TileCountSummary {
        p50: percentile(0.50),
        p95: percentile(0.95),
        p99: percentile(0.99),
        max: *sorted.last().expect("non-empty"),
        fallback_count: counts
            .iter()
            .filter(|&&count| count > fallback_limit)
            .count(),
    })
}

/// Decode one canonical 210-byte GGML Q6_K block in logical K order.
pub fn decode_q6_k_block(block: &[u8]) -> Result<DecodedQ6Block, String> {
    if block.len() != Q6_K_BYTES {
        return Err(format!(
            "Q6_K block must contain {Q6_K_BYTES} bytes, got {}",
            block.len()
        ));
    }
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
    let mut raw = [0.0f32; Q6_K_VALUES];
    let mut baseline_bf16 = [0.0f32; Q6_K_VALUES];
    let mut quants = [0i8; Q6_K_VALUES];
    let mut value_scales = [0i8; Q6_K_VALUES];

    for inb in (0..Q6_K_VALUES).step_by(16) {
        let half_idx = inb / 128;
        let pos = inb - half_idx * 128;
        let idx = pos % 32;
        let mut scale_idx = half_idx * 8 + idx / 16;
        let mut ql_offset = half_idx * 64 + idx;
        let (qh_shift, high_nibble) = if pos < 32 {
            (0, false)
        } else if pos < 64 {
            ql_offset += 32;
            scale_idx += 2;
            (2, false)
        } else if pos < 96 {
            scale_idx += 4;
            (4, true)
        } else {
            ql_offset += 32;
            scale_idx += 6;
            (6, true)
        };
        let qh_offset = 128 + half_idx * 32 + idx;
        let scale = block[192 + scale_idx] as i8;
        let ds = d * f32::from(scale);
        for lane in 0..16 {
            let packed = block[ql_offset + lane];
            let lo = if high_nibble {
                (packed >> 4) & 0x0f
            } else {
                packed & 0x0f
            };
            let hi = ((block[qh_offset + lane] >> qh_shift) & 3) << 4;
            let quant = (lo | hi) as i8 - 32;
            let value = ds * f32::from(quant);
            quants[inb + lane] = quant;
            value_scales[inb + lane] = scale;
            raw[inb + lane] = value;
            baseline_bf16[inb + lane] = bf16_trunc(value);
        }
    }
    Ok(DecodedQ6Block {
        d,
        quants,
        scales: value_scales,
        raw,
        baseline_bf16,
    })
}

#[inline(never)]
fn rn_mul(left: f32, right: f32) -> f32 {
    left * right
}

pub fn raw_q6_scalar_row_f32(row: &[u8], activation_bf16: &[u16]) -> Result<f32, String> {
    if row.len() != 20 * Q6_K_BYTES || activation_bf16.len() != 5120 {
        return Err("raw Q6 scalar row requires 4200 weight bytes and 5120 BF16 values".into());
    }
    let mut lanes = [0.0f32; 32];
    for block_index in 0..20 {
        let block =
            decode_q6_k_block(&row[block_index * Q6_K_BYTES..(block_index + 1) * Q6_K_BYTES])?;
        for lane in 0..32 {
            for t in 0..8 {
                let coordinate = lane + 32 * t;
                let weight = rn_mul(
                    rn_mul(block.d, f32::from(block.scales[coordinate])),
                    f32::from(block.quants[coordinate]),
                );
                let x = bf16::from_bits(activation_bf16[block_index * 256 + coordinate]).to_f32();
                lanes[lane] = weight.mul_add(x, lanes[lane]);
            }
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let before = lanes;
        for lane in 0..32 {
            lanes[lane] = before[lane] + before[lane ^ offset];
        }
    }
    lanes[0]
        .is_finite()
        .then_some(lanes[0])
        .ok_or_else(|| "scalar row is non-finite".into())
}

pub fn f32_to_bf16_rne_finite(value: f32) -> Result<u16, String> {
    if !value.is_finite() {
        return Err("BF16 RNE conversion requires a finite F32 value".into());
    }
    let bits = value.to_bits();
    let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
    Ok(((bits + rounding_bias) >> 16) as u16)
}

pub fn argmax_f32_as_bf16(logits: &[f32]) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("BF16 argmax requires a non-empty F32 slice".into());
    }
    let first = f32::from_bits(u32::from(f32_to_bf16_rne_finite(logits[0])?) << 16);
    let mut winner = 0usize;
    let mut best = first;
    for (index, &logit) in logits.iter().enumerate().skip(1) {
        let value = f32::from_bits(u32::from(f32_to_bf16_rne_finite(logit)?) << 16);
        if value > best {
            winner = index;
            best = value;
        }
    }
    Ok(winner)
}

pub fn weight_block_norms(block: &DecodedQ6Block) -> WeightBlockNorms {
    let mut w2 = 0.0f64;
    let mut d2 = 0.0f64;
    for (&raw, &baseline) in block.raw.iter().zip(&block.baseline_bf16) {
        let w = f64::from(baseline);
        let delta = w - f64::from(raw);
        w2 += w * w;
        d2 += delta * delta;
    }
    WeightBlockNorms {
        w_l2: w2.sqrt(),
        d_l2: d2.sqrt(),
    }
}

/// Encode a finite non-negative norm as FP16 without rounding it downward.
pub fn upward_f16_norm(value: f64) -> Result<f32, String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("norm must be finite and non-negative, got {value}"));
    }
    if value > f64::from(f16::MAX.to_f32()) {
        return Err(format!("norm {value} overflows finite FP16"));
    }
    let mut value_f32 = value as f32;
    if f64::from(value_f32) < value {
        value_f32 = f32::from_bits(value_f32.to_bits() + 1);
    }
    let mut encoded = f16::from_f32(value_f32);
    if encoded.to_f32() < value_f32 {
        let bits = encoded.to_bits();
        if bits >= 0x7bff {
            return Err(format!("norm {value} overflows finite FP16"));
        }
        encoded = f16::from_bits(bits + 1);
    }
    Ok(encoded.to_f32())
}

/// Reconstruct the activation values consumed by canonical 32-value Q8_1.
pub fn q8_1_reconstruct(x: &[f32]) -> Result<Vec<f32>, String> {
    if x.is_empty() || x.len() % Q8_1_VALUES != 0 {
        return Err(format!(
            "Q8_1 input length must be a non-zero multiple of {Q8_1_VALUES}, got {}",
            x.len()
        ));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err("Q8_1 input must be finite".into());
    }
    let mut out = vec![0.0f32; x.len()];
    for (input, reconstructed) in x
        .chunks_exact(Q8_1_VALUES)
        .zip(out.chunks_exact_mut(Q8_1_VALUES))
    {
        let amax = input.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        if amax == 0.0 {
            continue;
        }
        let d = amax / 127.0;
        let stored_d = f16::from_f32(d).to_f32();
        for (&value, output) in input.iter().zip(reconstructed) {
            let q = (value / d).round().clamp(-127.0, 127.0);
            *output = stored_d * q;
        }
    }
    Ok(out)
}

pub fn activation_block_norms(
    x: &[f32],
    reconstructed: &[f32],
) -> Result<ActivationBlockNorms, String> {
    if x.len() != Q6_K_VALUES || reconstructed.len() != Q6_K_VALUES {
        return Err(format!(
            "activation norm block requires {Q6_K_VALUES} values"
        ));
    }
    let mut e2 = 0.0f64;
    let mut a2 = 0.0f64;
    let mut x2 = 0.0f64;
    for (&x, &a) in x.iter().zip(reconstructed) {
        let x = f64::from(x);
        let a = f64::from(a);
        let e = x - a;
        e2 += e * e;
        a2 += a * a;
        x2 += x * x;
    }
    Ok(ActivationBlockNorms {
        e_l2: e2.sqrt(),
        a_l2: a2.sqrt(),
        x_l2: x2.sqrt(),
    })
}
