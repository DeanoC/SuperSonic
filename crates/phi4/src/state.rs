use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::Phi4Config;

/// Per-layer mutable state for Phi-4. Only full-attention KV caches — no
/// conv/recurrent state because Phi-4-mini has no linear attention layers.
///
/// When `kv_fp8` is set at allocation time, `kv_cache_k`/`kv_cache_v` hold
/// FP8-E4M3 bytes (`ScalarType::U8`) instead of BF16, and `kv_scale_k`/
/// `kv_scale_v` hold the per-head, per-position F32 absmax scale used by the
/// kernel for dequantization.
pub struct Phi4LayerState {
    pub kv_cache_k: Option<GpuBuffer>,
    pub kv_cache_v: Option<GpuBuffer>,
    pub kv_scale_k: Option<GpuBuffer>,
    pub kv_scale_v: Option<GpuBuffer>,
    pub kv_filled: usize,
}

impl Phi4LayerState {
    pub fn new() -> Self {
        Self {
            kv_cache_k: None,
            kv_cache_v: None,
            kv_scale_k: None,
            kv_scale_v: None,
            kv_filled: 0,
        }
    }

    /// Allocate or grow KV caches to hold `needed` positions. Caches grow in
    /// chunks of `kv_chunk_size` tokens to amortize reallocation. When
    /// `kv_fp8` is true the cache itself is FP8 (u8) and a parallel
    /// `[num_kv_heads, max_T]` F32 scale buffer is grown alongside.
    pub fn ensure_kv_capacity(
        &mut self,
        needed: usize,
        ordinal: usize,
        config: &Phi4Config,
        kv_chunk_size: usize,
        kv_fp8: bool,
    ) -> Result<(), GpuError> {
        let needed = needed + 1;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim();
        let kv_dtype = if kv_fp8 {
            ScalarType::U8
        } else {
            ScalarType::BF16
        };
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2];
            if current_cap >= needed {
                return Ok(());
            }
            let new_cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            self.kv_cache_k = Some(k.grow_seq_dim(2, new_cap)?);
            self.kv_cache_v = Some(v.grow_seq_dim(2, new_cap)?);
            if kv_fp8 {
                if let (Some(ref sk), Some(ref sv)) = (&self.kv_scale_k, &self.kv_scale_v) {
                    self.kv_scale_k = Some(sk.grow_seq_dim(1, new_cap)?);
                    self.kv_scale_v = Some(sv.grow_seq_dim(1, new_cap)?);
                }
            }
        } else {
            let cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            self.kv_cache_k = Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            self.kv_cache_v = Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            if kv_fp8 {
                self.kv_scale_k = Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
                self.kv_scale_v = Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
            }
        }
        Ok(())
    }

    pub fn set_kv_filled(&mut self, filled: usize) {
        self.kv_filled = filled;
    }

    pub fn kv_capacity(&self) -> usize {
        self.kv_cache_k.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }
}

impl Default for Phi4LayerState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Phi4ModelState {
    pub layers: Vec<Phi4LayerState>,
}

impl Phi4ModelState {
    pub fn new(config: &Phi4Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Phi4LayerState::new());
        }
        Self { layers }
    }
}
