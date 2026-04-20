use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::{Phi4Config, RopeScaling};

/// Pre-computed Phi-4-mini LongRoPE cos/sin tables on GPU.
///
/// Phi-4-mini uses distinct per-dimension frequency scaling factors for short vs long context:
/// - `short_*` tables apply when `kv_len <= original_max_position_embeddings` (4096).
/// - `long_*` tables apply beyond that boundary.
/// Both are pre-scaled by `mscale`, so the kernel multiplies cos/sin directly.
pub struct Phi4LongRope {
    pub cos_short: GpuBuffer,
    pub sin_short: GpuBuffer,
    pub cos_long: GpuBuffer,
    pub sin_long: GpuBuffer,
    pub rotary_dim: usize,
    pub original_max_position_embeddings: usize,
    pub mscale: f64,
}

impl Phi4LongRope {
    pub fn build(config: &Phi4Config, ordinal: usize) -> Result<Self, GpuError> {
        let rotary_dim = config.rotary_dim();
        let max_pos = config.max_position_embeddings;
        let mscale = config.mscale();

        let half_dim = rotary_dim / 2;
        let (short_factor, long_factor) = match &config.rope_scaling {
            Some(RopeScaling::Longrope { short_factor, long_factor }) => {
                (short_factor.as_slice(), long_factor.as_slice())
            }
            None => {
                let ones = vec![1.0_f64; half_dim];
                let ones_clone = ones.clone();
                return build_from_factors(
                    ordinal,
                    max_pos,
                    rotary_dim,
                    config.rope_theta,
                    &ones,
                    &ones_clone,
                    1.0,
                    config.original_max_position_embeddings,
                );
            }
        };

        build_from_factors(
            ordinal,
            max_pos,
            rotary_dim,
            config.rope_theta,
            short_factor,
            long_factor,
            mscale,
            config.original_max_position_embeddings,
        )
    }

    /// Return (cos, sin) pair to use at a given kv_len, matching HF's selection rule.
    pub fn tables_for_kv_len(&self, kv_len: usize) -> (&GpuBuffer, &GpuBuffer) {
        if kv_len > self.original_max_position_embeddings {
            (&self.cos_long, &self.sin_long)
        } else {
            (&self.cos_short, &self.sin_short)
        }
    }
}

fn build_from_factors(
    ordinal: usize,
    max_pos: usize,
    rotary_dim: usize,
    theta: f64,
    short_factor: &[f64],
    long_factor: &[f64],
    mscale: f64,
    original_max_position_embeddings: usize,
) -> Result<Phi4LongRope, GpuError> {
    let (cos_short_bytes, sin_short_bytes) = build_tables(max_pos, rotary_dim, theta, short_factor, mscale);
    let (cos_long_bytes, sin_long_bytes) = build_tables(max_pos, rotary_dim, theta, long_factor, mscale);

    let half_dim = rotary_dim / 2;
    let shape = [max_pos, half_dim];
    let cos_short = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &shape, &cos_short_bytes)?;
    let sin_short = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &shape, &sin_short_bytes)?;
    let cos_long = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &shape, &cos_long_bytes)?;
    let sin_long = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &shape, &sin_long_bytes)?;

    Ok(Phi4LongRope {
        cos_short,
        sin_short,
        cos_long,
        sin_long,
        rotary_dim,
        original_max_position_embeddings,
        mscale,
    })
}

/// Build cos/sin tables (each `[max_pos, rotary_dim/2]` BF16) for a single factor set.
///
/// inv_freq[i] = 1 / (factor[i] * theta^(2i/rotary_dim))
/// angle[pos, i] = pos * inv_freq[i]
/// cos_table[pos, i] = cos(angle) * mscale
/// sin_table[pos, i] = sin(angle) * mscale
fn build_tables(
    max_pos: usize,
    rotary_dim: usize,
    theta: f64,
    factor: &[f64],
    mscale: f64,
) -> (Vec<u8>, Vec<u8>) {
    let half_dim = rotary_dim / 2;
    assert_eq!(factor.len(), half_dim, "factor length must equal rotary_dim/2");

    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| {
            let base = theta.powf(2.0 * i as f64 / rotary_dim as f64);
            1.0 / (factor[i] * base)
        })
        .collect();

    let mut cos_data = Vec::with_capacity(max_pos * half_dim * 2);
    let mut sin_data = Vec::with_capacity(max_pos * half_dim * 2);

    for pos in 0..max_pos {
        for i in 0..half_dim {
            let angle = pos as f64 * inv_freq[i];
            let c = half::bf16::from_f64(angle.cos() * mscale);
            let s = half::bf16::from_f64(angle.sin() * mscale);
            cos_data.extend_from_slice(&c.to_le_bytes());
            sin_data.extend_from_slice(&s.to_le_bytes());
        }
    }

    (cos_data, sin_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phi4_mini_test_config() -> Phi4Config {
        Phi4Config {
            vocab_size: 200064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            original_max_position_embeddings: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.75,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            rope_scaling: Some(RopeScaling::Longrope {
                short_factor: vec![1.0; 48],
                long_factor: (0..48).map(|i| 1.0 + i as f64 * 0.1).collect(),
            }),
            eos_token_id: None,
            bos_token_id: None,
        }
    }

    #[test]
    fn cpu_tables_match_hf_formula_at_pos_zero() {
        let config = phi4_mini_test_config();
        let half_dim = config.rotary_dim() / 2;
        let short = vec![1.0_f64; half_dim];
        let (cos, sin) = build_tables(4096, config.rotary_dim(), config.rope_theta, &short, config.mscale());
        // At pos=0, cos = 1 * mscale, sin = 0.
        let m = config.mscale();
        let c0 = half::bf16::from_le_bytes([cos[0], cos[1]]).to_f64();
        let s0 = half::bf16::from_le_bytes([sin[0], sin[1]]).to_f64();
        assert!((c0 - m).abs() < 1e-2, "cos[0,0] = {c0}, mscale = {m}");
        assert!(s0.abs() < 1e-2);
    }

    #[test]
    fn long_factor_reduces_frequency() {
        // With long_factor[i] > 1, the angle at a given position is smaller than with short_factor=1.
        let rotary_dim = 96;
        let half_dim = rotary_dim / 2;
        let theta = 10000.0_f64;
        let short = vec![1.0_f64; half_dim];
        let long: Vec<f64> = (0..half_dim).map(|i| 1.0 + i as f64).collect();

        // pos=100, dim i=10
        let i = 10;
        let pos = 100.0;
        let base = theta.powf(2.0 * i as f64 / rotary_dim as f64);
        let angle_short = pos / base;
        let angle_long = pos / (long[i] * base);
        assert!(angle_long < angle_short);
        // Make sure our builder would give the same ratio.
        let (cos_short, _) = build_tables(200, rotary_dim, theta, &short, 1.0);
        let (cos_long, _) = build_tables(200, rotary_dim, theta, &long, 1.0);
        let offs = (100 * half_dim + i) * 2;
        let cs = half::bf16::from_le_bytes([cos_short[offs], cos_short[offs + 1]]).to_f64();
        let cl = half::bf16::from_le_bytes([cos_long[offs], cos_long[offs + 1]]).to_f64();
        // cos(small angle) > cos(large angle) for angles in (0, pi/2). Long factor yields smaller angle.
        assert!(cl >= cs - 1e-3, "cos_long ({cl}) expected >= cos_short ({cs})");
    }

    #[test]
    fn mscale_is_one_when_no_scaling() {
        let mut config = phi4_mini_test_config();
        config.max_position_embeddings = config.original_max_position_embeddings;
        assert!((config.mscale() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mscale_greater_than_one_for_long_context() {
        let config = phi4_mini_test_config();
        // Phi-4-mini: max=131072, original=4096 → scale=32, mscale = sqrt(1 + ln(32)/ln(4096)) ≈ 1.19.
        assert!(config.mscale() > 1.15);
        assert!(config.mscale() < 1.25);
    }
}
