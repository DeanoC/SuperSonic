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

    /// Build RoPE tables with explicit parameters (for non-Qwen3.8 models that
    /// share the same NEOX RoPE kernel, e.g. the DFlash2 draft model with
    /// theta=10M and head_dim=128).
    pub fn build_with_params(
        rotary_dim: usize,
        max_pos: usize,
        theta: f64,
        ordinal: usize,
    ) -> Result<Self, GpuError> {
        let half_dim = rotary_dim / 2;
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

    /// Build F32 RoPE tables with explicit parameters. The DFlash2 draft
    /// forward runs RoPE at F32 (matching the upstream ggml F32 compute
    /// type), so its cos/sin tables must be F32 — the RoPE kernel reads the
    /// table with the same dtype as the activation, and a BF16 table read as
    /// F32 would reinterpret the bytes as garbage.
    pub fn build_with_params_f32(
        rotary_dim: usize,
        max_pos: usize,
        theta: f64,
        ordinal: usize,
    ) -> Result<Self, GpuError> {
        let half_dim = rotary_dim / 2;
        let mut cos_data = Vec::with_capacity(max_pos * half_dim * 4);
        let mut sin_data = Vec::with_capacity(max_pos * half_dim * 4);
        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                cos_data.extend_from_slice(&(angle.cos() as f32).to_le_bytes());
                sin_data.extend_from_slice(&(angle.sin() as f32).to_le_bytes());
            }
        }
        let cos =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[max_pos, half_dim], &cos_data)?;
        let sin =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[max_pos, half_dim], &sin_data)?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }
}
