use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::TextConfig;
use crate::weights::LayerKind;

/// Mutable per-layer state (KV cache, conv state, recurrent state).
pub struct LayerState {
    pub kind: LayerKind,
    // Full attention
    pub kv_cache_k: Option<GpuBuffer>,
    pub kv_cache_v: Option<GpuBuffer>,
    pub kv_filled: usize,
    // Linear attention
    pub conv_state: Option<GpuBuffer>,
    pub recurrent_state: Option<GpuBuffer>,
}

impl LayerState {
    pub fn new_linear(ordinal: usize, config: &TextConfig) -> Result<Self, GpuError> {
        // Conv state: BF16 [qkv_out_dim, conv_kernel_size - 1] = [6144, 3]
        let qkv_out_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[qkv_out_dim, config.linear_conv_kernel_dim - 1],
        )?;
        // Recurrent state: F32 [num_v_heads, head_k_dim, head_v_dim]
        let recurrent_state = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ],
        )?;
        Ok(Self {
            kind: LayerKind::Linear,
            kv_cache_k: None,
            kv_cache_v: None,
            kv_filled: 0,
            conv_state: Some(conv_state),
            recurrent_state: Some(recurrent_state),
        })
    }

    pub fn new_full(_ordinal: usize) -> Self {
        Self {
            kind: LayerKind::Full,
            kv_cache_k: None,
            kv_cache_v: None,
            kv_filled: 0,
            conv_state: None,
            recurrent_state: None,
        }
    }

    /// Ensure KV cache has capacity for `needed` positions.
    /// Pre-allocates in chunks of `kv_chunk_size`.
    pub fn ensure_kv_capacity(
        &mut self,
        needed: usize,
        ordinal: usize,
        config: &TextConfig,
        kv_chunk_size: usize,
    ) -> Result<(), GpuError> {
        let needed = needed + 1; // need room for position `seqlen_offset`
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2]; // [1, nkv, seq, hd]
            if current_cap >= needed {
                return Ok(());
            }
            let new_cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            let new_k = k.grow_seq_dim(2, new_cap)?;
            let new_v = v.grow_seq_dim(2, new_cap)?;
            self.kv_cache_k = Some(new_k);
            self.kv_cache_v = Some(new_v);
        } else {
            // First allocation: create cache with chunked capacity
            let cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            let nkv = config.num_key_value_heads;
            let hd = config.head_dim;
            self.kv_cache_k =
                Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, cap, hd])?);
            self.kv_cache_v =
                Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, cap, hd])?);
        }
        Ok(())
    }

    /// Record actual filled KV length (no reallocation).
    pub fn set_kv_filled(&mut self, filled: usize) {
        self.kv_filled = filled;
    }

    /// Get KV cache capacity (allocated seq dim).
    pub fn kv_capacity(&self) -> usize {
        self.kv_cache_k
            .as_ref()
            .map(|k| k.shape()[2])
            .unwrap_or(0)
    }
}

/// All mutable state for the model.
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

impl ModelState {
    pub fn new(config: &TextConfig, ordinal: usize) -> Result<Self, GpuError> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            if config.is_full_attention(idx) {
                layers.push(LayerState::new_full(ordinal));
            } else {
                layers.push(LayerState::new_linear(ordinal, config)?);
            }
        }
        Ok(Self { layers })
    }
}
