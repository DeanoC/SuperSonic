//! RoPE cos/sin tables for the DFlash draft.
//!
//! Full-dim rotary (rotary_dim = head_dim = 128). Layout matches the
//! `apply_rope_prefill` kernel: `[max_position, half_rot]` BF16 — stride
//! `half_rot * 2` bytes per position. Mirrors `qwen35::rotary::RotaryTables`
//! but lives in the draft crate so it can be built from `DFlashConfig`
//! without pulling in `TextConfig`.

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::DFlashConfig;

pub struct RotaryTables {
    pub cos: GpuBuffer,
    pub sin: GpuBuffer,
    pub rotary_dim: usize,
    pub max_position: usize,
}

impl RotaryTables {
    /// Build cos/sin tables sized for `max_position` positions. Pass
    /// `config.max_position_embeddings` for full coverage, or cap smaller
    /// at test time to save VRAM.
    pub fn build(
        config: &DFlashConfig,
        ordinal: usize,
        max_position: usize,
    ) -> Result<Self, GpuError> {
        let rotary_dim = config.head_dim;
        let half_dim = rotary_dim / 2;
        let theta = config.rope_theta;

        let mut cos_data = Vec::with_capacity(max_position * half_dim * 2);
        let mut sin_data = Vec::with_capacity(max_position * half_dim * 2);

        for pos in 0..max_position {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                let c = half::bf16::from_f64(angle.cos());
                let s = half::bf16::from_f64(angle.sin());
                cos_data.extend_from_slice(&c.to_le_bytes());
                sin_data.extend_from_slice(&s.to_le_bytes());
            }
        }

        let cos = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[max_position, half_dim],
            &cos_data,
        )?;
        let sin = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[max_position, half_dim],
            &sin_data,
        )?;

        Ok(Self {
            cos,
            sin,
            rotary_dim,
            max_position,
        })
    }
}
