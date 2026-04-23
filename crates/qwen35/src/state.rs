use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::TextConfig;
use crate::weights::LayerKind;

pub fn kv_fp8_bf16_sidecar_enabled() -> bool {
    std::env::var_os("SUPERSONIC_DEBUG_DISABLE_KV_FP8_BF16_SIDECAR").is_none()
}

pub fn kv_fp8_bf16_sidecar_window_tokens() -> Option<usize> {
    std::env::var("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Mutable per-layer state (KV cache, conv state, recurrent state).
pub struct LayerState {
    pub kind: LayerKind,
    // Full attention
    pub kv_cache_k: Option<GpuBuffer>,
    pub kv_cache_v: Option<GpuBuffer>,
    pub kv_filled: usize,
    // FP8 KV cache scales (per-head-per-position absmax)
    pub kv_scale_k: Option<GpuBuffer>,
    pub kv_scale_v: Option<GpuBuffer>,
    // BF16 sidecar cache used by the 4B KV-FP8 decode path for parity-sensitive
    // reads and debug tracing.
    pub kv_shadow_k: Option<GpuBuffer>,
    pub kv_shadow_v: Option<GpuBuffer>,
    pub kv_shadow_start: usize,
    // Certified KV experimental decode cache: INT8 post-RoPE keys and per-block
    // scales for the contiguous prefix covered by complete blocks.
    pub certified_kv_key_i8: Option<GpuBuffer>,
    pub certified_kv_key_scale: Option<GpuBuffer>,
    pub certified_kv_key_tokens: usize,
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
            kv_scale_k: None,
            kv_scale_v: None,
            kv_shadow_k: None,
            kv_shadow_v: None,
            kv_shadow_start: usize::MAX,
            certified_kv_key_i8: None,
            certified_kv_key_scale: None,
            certified_kv_key_tokens: 0,
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
            kv_scale_k: None,
            kv_scale_v: None,
            kv_shadow_k: None,
            kv_shadow_v: None,
            kv_shadow_start: usize::MAX,
            certified_kv_key_i8: None,
            certified_kv_key_scale: None,
            certified_kv_key_tokens: 0,
            conv_state: None,
            recurrent_state: None,
        }
    }

    /// Ensure KV cache has capacity for `needed` positions.
    /// Pre-allocates in chunks of `kv_chunk_size`.
    /// When `kv_fp8` is true, KV caches use FP8 E4M3 (U8) with per-head-per-position
    /// F32 absmax scale buffers, halving KV cache VRAM.
    pub fn ensure_kv_capacity(
        &mut self,
        needed: usize,
        ordinal: usize,
        config: &TextConfig,
        kv_chunk_size: usize,
        kv_fp8: bool,
    ) -> Result<(), GpuError> {
        let needed = needed + 1; // need room for position `seqlen_offset`
        let kv_dtype = if kv_fp8 { ScalarType::U8 } else { ScalarType::BF16 };
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2]; // [1, nkv, seq, hd]
            if kv_fp8
                && kv_fp8_bf16_sidecar_enabled()
                && (self.kv_shadow_k.is_none() || self.kv_shadow_v.is_none())
            {
                let nkv = config.num_key_value_heads;
                let hd = config.head_dim;
                self.kv_shadow_k =
                    Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, current_cap, hd])?);
                self.kv_shadow_v =
                    Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, current_cap, hd])?);
                self.kv_shadow_start = self.kv_filled;
            }
            if current_cap >= needed {
                return Ok(());
            }
            let new_cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            let new_k = k.grow_seq_dim(2, new_cap)?;
            let new_v = v.grow_seq_dim(2, new_cap)?;
            self.kv_cache_k = Some(new_k);
            self.kv_cache_v = Some(new_v);
            // Grow scale buffers alongside KV caches
            if kv_fp8 {
                if let (Some(ref sk), Some(ref sv)) = (&self.kv_scale_k, &self.kv_scale_v) {
                    let new_sk = sk.grow_seq_dim(1, new_cap)?;
                    let new_sv = sv.grow_seq_dim(1, new_cap)?;
                    self.kv_scale_k = Some(new_sk);
                    self.kv_scale_v = Some(new_sv);
                }
                if let (Some(ref shadow_k), Some(ref shadow_v)) = (&self.kv_shadow_k, &self.kv_shadow_v) {
                    let new_shadow_k = shadow_k.grow_seq_dim(2, new_cap)?;
                    let new_shadow_v = shadow_v.grow_seq_dim(2, new_cap)?;
                    self.kv_shadow_k = Some(new_shadow_k);
                    self.kv_shadow_v = Some(new_shadow_v);
                }
            }
        } else {
            // First allocation: create cache with chunked capacity
            let cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            let nkv = config.num_key_value_heads;
            let hd = config.head_dim;
            self.kv_cache_k =
                Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            self.kv_cache_v =
                Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            if kv_fp8 {
                // Scale buffers: [nkv, cap] of F32 — one scale per head per position
                self.kv_scale_k =
                    Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
                self.kv_scale_v =
                    Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
                if kv_fp8_bf16_sidecar_enabled() {
                    self.kv_shadow_k =
                        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, cap, hd])?);
                    self.kv_shadow_v =
                        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nkv, cap, hd])?);
                    self.kv_shadow_start = self.kv_filled;
                }
            }
        }
        Ok(())
    }

    /// Record actual filled KV length (no reallocation).
    pub fn set_kv_filled(&mut self, filled: usize) {
        self.kv_filled = filled;
        if filled < self.certified_kv_key_tokens {
            self.certified_kv_key_tokens = 0;
        }
        if self.kv_shadow_k.is_some() && self.kv_shadow_v.is_some() {
            self.kv_shadow_start = kv_fp8_bf16_sidecar_window_tokens()
                .map(|window| filled.saturating_sub(window))
                .unwrap_or(0);
        }
    }

    /// Get KV cache capacity (allocated seq dim).
    pub fn kv_capacity(&self) -> usize {
        self.kv_cache_k
            .as_ref()
            .map(|k| k.shape()[2])
            .unwrap_or(0)
    }
}

impl LayerState {
    /// Deep-copy all GPU buffers to create an independent clone.
    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let clone_opt = |opt: &Option<GpuBuffer>| -> Result<Option<GpuBuffer>, GpuError> {
            match opt {
                Some(buf) => Ok(Some(buf.clone_device()?)),
                None => Ok(None),
            }
        };
        Ok(Self {
            kind: self.kind,
            kv_cache_k: clone_opt(&self.kv_cache_k)?,
            kv_cache_v: clone_opt(&self.kv_cache_v)?,
            kv_filled: self.kv_filled,
            kv_scale_k: clone_opt(&self.kv_scale_k)?,
            kv_scale_v: clone_opt(&self.kv_scale_v)?,
            kv_shadow_k: clone_opt(&self.kv_shadow_k)?,
            kv_shadow_v: clone_opt(&self.kv_shadow_v)?,
            kv_shadow_start: self.kv_shadow_start,
            certified_kv_key_i8: clone_opt(&self.certified_kv_key_i8)?,
            certified_kv_key_scale: clone_opt(&self.certified_kv_key_scale)?,
            certified_kv_key_tokens: self.certified_kv_key_tokens,
            conv_state: clone_opt(&self.conv_state)?,
            recurrent_state: clone_opt(&self.recurrent_state)?,
        })
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

    /// Deep-copy all layer states to create an independent clone.
    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            layers.push(ls.clone_gpu()?);
        }
        Ok(Self { layers })
    }

    /// Capture `(conv_state, recurrent_state)` for every linear-attention
    /// layer into a sidecar. Full-attention slots carry `None` so the inner
    /// `Vec` is indexed 1:1 with `self.layers`.
    ///
    /// Used by the DFlash speculative engine to roll back linear state after
    /// a partial-acceptance verify — full-attention K/V uses the separate
    /// counter-flip (commit_kv_filled=false) path, per docs/dflash.md §6.1.
    /// Cost on Qwen3.5-9B: ~1 MiB/layer × 24 linear layers = ~25 MiB.
    pub fn snapshot_linear(&self) -> Result<LinearStateSnapshot, GpuError> {
        let mut per_layer = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            match (ls.kind, &ls.conv_state, &ls.recurrent_state) {
                (LayerKind::Linear, Some(conv), Some(rec)) => {
                    per_layer.push(Some((conv.clone_device()?, rec.clone_device()?)));
                }
                _ => per_layer.push(None),
            }
        }
        Ok(LinearStateSnapshot { per_layer })
    }

    /// Restore every linear layer's `(conv_state, recurrent_state)` from
    /// `snap` via D2D copies into the existing buffers. Shapes/dtypes must
    /// match what `snapshot_linear` captured — this is a tight invariant
    /// because the sidecar originated from the same `ModelState::new`.
    pub fn restore_linear(
        &mut self,
        snap: &LinearStateSnapshot,
        ordinal: usize,
    ) -> Result<(), GpuError> {
        if snap.per_layer.len() != self.layers.len() {
            return Err(GpuError::InvalidArg(format!(
                "restore_linear: snapshot has {} layers, state has {}",
                snap.per_layer.len(),
                self.layers.len(),
            )));
        }
        for (i, ls) in self.layers.iter_mut().enumerate() {
            match (ls.kind, &snap.per_layer[i]) {
                (LayerKind::Linear, Some((conv_src, rec_src))) => {
                    let conv_dst = ls.conv_state.as_mut().ok_or_else(|| {
                        GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} missing conv_state"
                        ))
                    })?;
                    let rec_dst = ls.recurrent_state.as_mut().ok_or_else(|| {
                        GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} missing recurrent_state"
                        ))
                    })?;
                    if conv_dst.len_bytes() != conv_src.len_bytes()
                        || rec_dst.len_bytes() != rec_src.len_bytes()
                    {
                        return Err(GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} size mismatch (conv dst={} src={}, rec dst={} src={})",
                            conv_dst.len_bytes(),
                            conv_src.len_bytes(),
                            rec_dst.len_bytes(),
                            rec_src.len_bytes(),
                        )));
                    }
                    gpu_hal::copy_d2d(
                        ordinal,
                        conv_dst.as_mut_ptr(),
                        conv_src.as_ptr(),
                        conv_src.len_bytes(),
                    )?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        rec_dst.as_mut_ptr(),
                        rec_src.as_ptr(),
                        rec_src.len_bytes(),
                    )?;
                }
                (LayerKind::Full, None) => {}
                (LayerKind::Linear, None) => {
                    return Err(GpuError::InvalidArg(format!(
                        "restore_linear: layer {i} is Linear but snapshot slot is None"
                    )));
                }
                (LayerKind::Full, Some(_)) => {
                    return Err(GpuError::InvalidArg(format!(
                        "restore_linear: layer {i} is Full but snapshot slot is Some"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Sidecar holding `(conv_state, recurrent_state)` for every linear-attention
/// layer at some earlier logical position. Produced by
/// [`ModelState::snapshot_linear`] and consumed by
/// [`ModelState::restore_linear`]. Full-attention layers store `None` so
/// slot indices line up 1:1 with [`ModelState::layers`].
pub struct LinearStateSnapshot {
    pub per_layer: Vec<Option<(GpuBuffer, GpuBuffer)>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;

    fn tiny_config() -> TextConfig {
        TextConfig {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            hidden_act: Activation::default(),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rms_norm_add_unit_offset: true,
            tie_word_embeddings: false,
            eos_token_id: None,
            head_dim: 16,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            layer_types: vec![],
            rope_parameters: None,
        }
        .normalized()
    }

    fn random_bytes(count: usize, seed: u64) -> Vec<u8> {
        let mut s: u64 = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push(((s >> 33) & 0xFF) as u8);
        }
        out
    }

    /// Bit-exact roundtrip: fill linear state with bytes A, snapshot, overwrite
    /// with bytes B, restore, assert we read back bytes A everywhere.
    ///
    /// `#[ignore]` because it needs a HIP/CUDA runtime. Run with:
    ///   cargo test -p qwen35 -- --ignored linear_snapshot_roundtrip_bit_exact
    #[test]
    #[ignore = "requires a GPU runtime"]
    fn linear_snapshot_roundtrip_bit_exact() {
        let ordinal = 0_usize;
        let config = tiny_config();
        assert_eq!(config.layer_types.len(), config.num_hidden_layers);
        assert!(!config.is_full_attention(0));
        assert!(config.is_full_attention(3));

        let mut state = ModelState::new(&config, ordinal).expect("alloc ModelState");

        // Write bytes-A into every linear layer's (conv_state, recurrent_state).
        let mut expected_per_layer: Vec<Option<(Vec<u8>, Vec<u8>)>> =
            Vec::with_capacity(state.layers.len());
        for (i, ls) in state.layers.iter_mut().enumerate() {
            match (ls.kind, ls.conv_state.as_mut(), ls.recurrent_state.as_mut()) {
                (LayerKind::Linear, Some(conv), Some(rec)) => {
                    let conv_a = random_bytes(conv.len_bytes(), 0xC07A + i as u64);
                    let rec_a = random_bytes(rec.len_bytes(), 0x8EC0 + i as u64);
                    gpu_hal::copy_h2d(
                        ordinal,
                        conv.as_mut_ptr(),
                        conv_a.as_ptr() as *const _,
                        conv_a.len(),
                    )
                    .expect("h2d conv A");
                    gpu_hal::copy_h2d(
                        ordinal,
                        rec.as_mut_ptr(),
                        rec_a.as_ptr() as *const _,
                        rec_a.len(),
                    )
                    .expect("h2d rec A");
                    expected_per_layer.push(Some((conv_a, rec_a)));
                }
                _ => expected_per_layer.push(None),
            }
        }

        // Snapshot at bytes-A.
        let snap = state.snapshot_linear().expect("snapshot_linear");
        assert_eq!(snap.per_layer.len(), state.layers.len());

        // Overwrite with bytes-B (different seed).
        for (i, ls) in state.layers.iter_mut().enumerate() {
            if let (LayerKind::Linear, Some(conv), Some(rec)) =
                (ls.kind, ls.conv_state.as_mut(), ls.recurrent_state.as_mut())
            {
                let conv_b = random_bytes(conv.len_bytes(), 0xBBBB + i as u64);
                let rec_b = random_bytes(rec.len_bytes(), 0xCCCC + i as u64);
                gpu_hal::copy_h2d(
                    ordinal,
                    conv.as_mut_ptr(),
                    conv_b.as_ptr() as *const _,
                    conv_b.len(),
                )
                .expect("h2d conv B");
                gpu_hal::copy_h2d(
                    ordinal,
                    rec.as_mut_ptr(),
                    rec_b.as_ptr() as *const _,
                    rec_b.len(),
                )
                .expect("h2d rec B");
            }
        }

        // Restore from the bytes-A snapshot.
        state
            .restore_linear(&snap, ordinal)
            .expect("restore_linear");

        // Read back and assert bit-exact equality with bytes-A.
        for (i, ls) in state.layers.iter().enumerate() {
            match (ls.kind, &ls.conv_state, &ls.recurrent_state, &expected_per_layer[i]) {
                (LayerKind::Linear, Some(conv), Some(rec), Some((conv_a, rec_a))) => {
                    let conv_rb = conv.to_host_bytes().expect("d2h conv restored");
                    let rec_rb = rec.to_host_bytes().expect("d2h rec restored");
                    assert_eq!(
                        &conv_rb, conv_a,
                        "layer {i}: conv_state mismatch after restore"
                    );
                    assert_eq!(
                        &rec_rb, rec_a,
                        "layer {i}: recurrent_state mismatch after restore"
                    );
                }
                (LayerKind::Full, None, None, None) => {}
                _ => panic!(
                    "layer {i}: kind/state/snapshot-slot inconsistency after restore"
                ),
            }
        }

        // Sanity: a second snapshot + restore round-trip on the restored state
        // is a no-op.
        let snap2 = state.snapshot_linear().expect("snapshot_linear 2nd");
        state.restore_linear(&snap2, ordinal).expect("restore_linear 2nd");
        for (i, ls) in state.layers.iter().enumerate() {
            if let (LayerKind::Linear, Some(conv), Some(rec), Some((conv_a, rec_a))) =
                (ls.kind, &ls.conv_state, &ls.recurrent_state, &expected_per_layer[i])
            {
                let conv_rb = conv.to_host_bytes().expect("d2h conv 2nd");
                let rec_rb = rec.to_host_bytes().expect("d2h rec 2nd");
                assert_eq!(&conv_rb, conv_a, "layer {i}: conv drift on 2nd roundtrip");
                assert_eq!(&rec_rb, rec_a, "layer {i}: rec drift on 2nd roundtrip");
            }
        }
    }
}
