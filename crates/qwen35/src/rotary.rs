use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::TextConfig;

/// Pre-computed RoPE cos/sin tables on GPU.
pub struct RotaryTables {
    pub cos: GpuBuffer,
    pub sin: GpuBuffer,
    pub rotary_dim: usize,
}

impl RotaryTables {
    /// Build RoPE cos/sin tables and upload to GPU.
    /// Shape: [max_position, rotary_dim/2] in BF16.
    pub fn build(config: &TextConfig, ordinal: usize) -> Result<Self, GpuError> {
        let rotary_dim = config.rotary_dim();
        let half_dim = rotary_dim / 2;
        let max_pos = config.max_position_embeddings;
        let theta = config.rope_theta();

        // Compute on CPU in F32, convert to BF16
        let mut cos_data = Vec::with_capacity(max_pos * half_dim * 2);
        let mut sin_data = Vec::with_capacity(max_pos * half_dim * 2);

        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                let c = half::bf16::from_f64(angle.cos());
                let s = half::bf16::from_f64(angle.sin());
                cos_data.extend_from_slice(&c.to_le_bytes());
                sin_data.extend_from_slice(&s.to_le_bytes());
            }
        }

        let cos =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half_dim], &cos_data)?;
        let sin =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half_dim], &sin_data)?;

        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }
}
