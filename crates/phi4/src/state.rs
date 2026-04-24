use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::Phi4Config;

/// Per-layer mutable state for Phi-4. Only full-attention KV caches — no
/// conv/recurrent state because Phi-4-mini has no linear attention layers.
pub struct Phi4LayerState {
    pub kv_cache_k: Option<GpuBuffer>,
    pub kv_cache_v: Option<GpuBuffer>,
    pub kv_filled: usize,
}

impl Phi4LayerState {
    pub fn new() -> Self {
        Self {
            kv_cache_k: None,
            kv_cache_v: None,
            kv_filled: 0,
        }
    }

    /// Allocate or grow KV caches to hold `needed` positions. Caches grow in
    /// chunks of `kv_chunk_size` tokens to amortize reallocation.
    pub fn ensure_kv_capacity(
        &mut self,
        needed: usize,
        ordinal: usize,
        config: &Phi4Config,
        kv_chunk_size: usize,
    ) -> Result<(), GpuError> {
        let needed = needed + 1;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim();
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2];
            if current_cap >= needed {
                return Ok(());
            }
            let new_cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            self.kv_cache_k = Some(k.grow_seq_dim(2, new_cap)?);
            self.kv_cache_v = Some(v.grow_seq_dim(2, new_cap)?);
        } else {
            let cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            self.kv_cache_k = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, nkv, cap, hd],
            )?);
            self.kv_cache_v = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, nkv, cap, hd],
            )?);
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
