use std::ffi::c_void;

use gpu_hal::{
    copy_h2d, sync, GpuBuffer, GpuError, HostBuffer, ScalarType, VirtualBacking, VirtualBuffer,
};

use crate::config::TextConfig;
use crate::weights::LayerKind;

#[derive(Debug, Clone, Copy, Default)]
pub struct VirtualKvMemoryStats {
    pub layers: usize,
    pub logical_bytes: usize,
    pub reserved_bytes: usize,
    pub resident_bytes: usize,
    pub logical_resident_bytes: usize,
    pub logical_backup_bytes: usize,
    pub mappings: usize,
}

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
    pub virtual_kv_cache_k: Option<VirtualBuffer>,
    pub virtual_kv_guard: Option<VirtualBuffer>,
    pub virtual_kv_cache_v: Option<VirtualBuffer>,
    pub virtual_kv_max_t: Option<usize>,
    virtual_kv_full_backup: Option<VirtualKvFullBackup>,
    virtual_kv_logical_backup: Option<VirtualKvLogicalBackup>,
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
    pub certified_kv_key_zero: Option<GpuBuffer>,
    pub certified_kv_key_tokens: usize,
    pub certified_kv_value_i4: Option<GpuBuffer>,
    pub certified_kv_value_scale: Option<GpuBuffer>,
    pub certified_kv_value_zero: Option<GpuBuffer>,
    pub certified_kv_value_error: Option<GpuBuffer>,
    pub certified_kv_value_norm: Option<GpuBuffer>,
    pub certified_kv_value_tokens: usize,
    pub certified_kv_host_k: Option<HostBuffer>,
    pub certified_kv_host_v: Option<HostBuffer>,
    pub certified_kv_host_tokens: usize,
    pub certified_kv_host_meta_blocks: usize,
    pub certified_kv_host_meta_key_stride_tokens: usize,
    pub certified_kv_host_meta_key_scale_stride_blocks: usize,
    pub certified_kv_host_meta_value_error_stride_blocks: usize,
    pub certified_kv_device_meta_key_scale_norm_blocks: usize,
    pub certified_kv_device_meta_key_scale_stride_blocks: usize,
    pub certified_kv_host_key_i8_cache: Vec<u8>,
    pub certified_kv_host_key_scale_cache: Vec<f32>,
    pub certified_kv_host_key_scale_channel_max_cache: Vec<f32>,
    pub certified_kv_host_key_zero_cache: Vec<f32>,
    pub certified_kv_host_value_error_cache: Vec<f32>,
    pub certified_kv_host_value_norm_cache: Vec<f32>,
    pub certified_kv_tail_k: Option<GpuBuffer>,
    pub certified_kv_tail_v: Option<GpuBuffer>,
    pub certified_kv_gpu_tail_only: bool,
    pub certified_kv_promoted_key_cache: Option<GpuBuffer>,
    pub certified_kv_promoted_key_cache_tags_gpu: Option<GpuBuffer>,
    pub certified_kv_promoted_key_cache_lru_gpu: Option<GpuBuffer>,
    pub certified_kv_promoted_key_cache_capacity: usize,
    pub certified_kv_promoted_key_cache_tags: Vec<usize>,
    pub certified_kv_promoted_key_cache_lru: Vec<u64>,
    pub certified_kv_promoted_key_cache_tick: u64,
    pub certified_kv_promoted_value_cache: Option<GpuBuffer>,
    pub certified_kv_promoted_value_cache_capacity: usize,
    pub certified_kv_promoted_value_cache_tags: Vec<usize>,
    pub certified_kv_promoted_value_cache_lru: Vec<u64>,
    pub certified_kv_promoted_value_cache_tick: u64,
    pub certified_kv_ranking_prefix_k: Option<GpuBuffer>,
    pub certified_kv_ranking_prefix_v: Option<GpuBuffer>,
    pub certified_kv_ranking_prefix_tokens: usize,
    pub certified_kv_ranking_prefix_kv_heads: Vec<usize>,
    // Linear attention
    pub conv_state: Option<GpuBuffer>,
    pub recurrent_state: Option<GpuBuffer>,
}

struct VirtualKvLogicalBackup {
    k: Vec<u8>,
    v: Vec<u8>,
    prefix_len: usize,
    cap: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

struct VirtualKvFullBackup {
    k: Vec<u8>,
    v: Option<Vec<u8>>,
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
            virtual_kv_cache_k: None,
            virtual_kv_guard: None,
            virtual_kv_cache_v: None,
            virtual_kv_max_t: None,
            virtual_kv_full_backup: None,
            virtual_kv_logical_backup: None,
            kv_filled: 0,
            kv_scale_k: None,
            kv_scale_v: None,
            kv_shadow_k: None,
            kv_shadow_v: None,
            kv_shadow_start: usize::MAX,
            certified_kv_key_i8: None,
            certified_kv_key_scale: None,
            certified_kv_key_zero: None,
            certified_kv_key_tokens: 0,
            certified_kv_value_i4: None,
            certified_kv_value_scale: None,
            certified_kv_value_zero: None,
            certified_kv_value_error: None,
            certified_kv_value_norm: None,
            certified_kv_value_tokens: 0,
            certified_kv_host_k: None,
            certified_kv_host_v: None,
            certified_kv_host_tokens: 0,
            certified_kv_host_meta_blocks: 0,
            certified_kv_host_meta_key_stride_tokens: 0,
            certified_kv_host_meta_key_scale_stride_blocks: 0,
            certified_kv_host_meta_value_error_stride_blocks: 0,
            certified_kv_device_meta_key_scale_norm_blocks: 0,
            certified_kv_device_meta_key_scale_stride_blocks: 0,
            certified_kv_host_key_i8_cache: Vec::new(),
            certified_kv_host_key_scale_cache: Vec::new(),
            certified_kv_host_key_scale_channel_max_cache: Vec::new(),
            certified_kv_host_key_zero_cache: Vec::new(),
            certified_kv_host_value_error_cache: Vec::new(),
            certified_kv_host_value_norm_cache: Vec::new(),
            certified_kv_tail_k: None,
            certified_kv_tail_v: None,
            certified_kv_gpu_tail_only: false,
            certified_kv_promoted_key_cache: None,
            certified_kv_promoted_key_cache_tags_gpu: None,
            certified_kv_promoted_key_cache_lru_gpu: None,
            certified_kv_promoted_key_cache_capacity: 0,
            certified_kv_promoted_key_cache_tags: Vec::new(),
            certified_kv_promoted_key_cache_lru: Vec::new(),
            certified_kv_promoted_key_cache_tick: 0,
            certified_kv_promoted_value_cache: None,
            certified_kv_promoted_value_cache_capacity: 0,
            certified_kv_promoted_value_cache_tags: Vec::new(),
            certified_kv_promoted_value_cache_lru: Vec::new(),
            certified_kv_promoted_value_cache_tick: 0,
            certified_kv_ranking_prefix_k: None,
            certified_kv_ranking_prefix_v: None,
            certified_kv_ranking_prefix_tokens: 0,
            certified_kv_ranking_prefix_kv_heads: Vec::new(),
            conv_state: Some(conv_state),
            recurrent_state: Some(recurrent_state),
        })
    }

    pub fn new_full(_ordinal: usize) -> Self {
        Self {
            kind: LayerKind::Full,
            kv_cache_k: None,
            kv_cache_v: None,
            virtual_kv_cache_k: None,
            virtual_kv_guard: None,
            virtual_kv_cache_v: None,
            virtual_kv_max_t: None,
            virtual_kv_full_backup: None,
            virtual_kv_logical_backup: None,
            kv_filled: 0,
            kv_scale_k: None,
            kv_scale_v: None,
            kv_shadow_k: None,
            kv_shadow_v: None,
            kv_shadow_start: usize::MAX,
            certified_kv_key_i8: None,
            certified_kv_key_scale: None,
            certified_kv_key_zero: None,
            certified_kv_key_tokens: 0,
            certified_kv_value_i4: None,
            certified_kv_value_scale: None,
            certified_kv_value_zero: None,
            certified_kv_value_error: None,
            certified_kv_value_norm: None,
            certified_kv_value_tokens: 0,
            certified_kv_host_k: None,
            certified_kv_host_v: None,
            certified_kv_host_tokens: 0,
            certified_kv_host_meta_blocks: 0,
            certified_kv_host_meta_key_stride_tokens: 0,
            certified_kv_host_meta_key_scale_stride_blocks: 0,
            certified_kv_host_meta_value_error_stride_blocks: 0,
            certified_kv_device_meta_key_scale_norm_blocks: 0,
            certified_kv_device_meta_key_scale_stride_blocks: 0,
            certified_kv_host_key_i8_cache: Vec::new(),
            certified_kv_host_key_scale_cache: Vec::new(),
            certified_kv_host_key_scale_channel_max_cache: Vec::new(),
            certified_kv_host_key_zero_cache: Vec::new(),
            certified_kv_host_value_error_cache: Vec::new(),
            certified_kv_host_value_norm_cache: Vec::new(),
            certified_kv_tail_k: None,
            certified_kv_tail_v: None,
            certified_kv_gpu_tail_only: false,
            certified_kv_promoted_key_cache: None,
            certified_kv_promoted_key_cache_tags_gpu: None,
            certified_kv_promoted_key_cache_lru_gpu: None,
            certified_kv_promoted_key_cache_capacity: 0,
            certified_kv_promoted_key_cache_tags: Vec::new(),
            certified_kv_promoted_key_cache_lru: Vec::new(),
            certified_kv_promoted_key_cache_tick: 0,
            certified_kv_promoted_value_cache: None,
            certified_kv_promoted_value_cache_capacity: 0,
            certified_kv_promoted_value_cache_tags: Vec::new(),
            certified_kv_promoted_value_cache_lru: Vec::new(),
            certified_kv_promoted_value_cache_tick: 0,
            certified_kv_ranking_prefix_k: None,
            certified_kv_ranking_prefix_v: None,
            certified_kv_ranking_prefix_tokens: 0,
            certified_kv_ranking_prefix_kv_heads: Vec::new(),
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
        let kv_dtype = if kv_fp8 {
            ScalarType::U8
        } else {
            ScalarType::BF16
        };
        if !kv_fp8 {
            if let Some(max_t) = self.virtual_kv_max_t {
                let cap = ((max_t.max(needed) + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
                let nkv = config.num_key_value_heads;
                let hd = config.head_dim;
                if self.virtual_kv_cache_k.is_none() {
                    self.kv_cache_k = None;
                    self.kv_cache_v = None;
                    self.virtual_kv_cache_k = Some(VirtualBuffer::reserve(
                        ordinal,
                        ScalarType::BF16,
                        &[1, nkv, cap, hd],
                        VirtualBacking::CpuBackup,
                    )?);
                    self.virtual_kv_guard = Some(VirtualBuffer::reserve(
                        ordinal,
                        ScalarType::U8,
                        &[1],
                        VirtualBacking::Discard,
                    )?);
                    self.virtual_kv_cache_v = Some(VirtualBuffer::reserve(
                        ordinal,
                        ScalarType::BF16,
                        &[1, nkv, cap, hd],
                        VirtualBacking::CpuBackup,
                    )?);
                }
                if let Some(kv) = self.virtual_kv_cache_k.as_mut() {
                    kv.map_prefix_bytes(kv.len_bytes())
                        .map_err(|e| GpuError::InvalidArg(format!("virtual K map: {e}")))?;
                }
                if let Some(kv) = self.virtual_kv_cache_v.as_mut() {
                    kv.map_prefix_bytes(kv.len_bytes())
                        .map_err(|e| GpuError::InvalidArg(format!("virtual V map: {e}")))?;
                }
                return Ok(());
            }
        }
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2]; // [1, nkv, seq, hd]
            if kv_fp8
                && kv_fp8_bf16_sidecar_enabled()
                && (self.kv_shadow_k.is_none() || self.kv_shadow_v.is_none())
            {
                let nkv = config.num_key_value_heads;
                let hd = config.head_dim;
                self.kv_shadow_k = Some(GpuBuffer::zeros(
                    ordinal,
                    ScalarType::BF16,
                    &[1, nkv, current_cap, hd],
                )?);
                self.kv_shadow_v = Some(GpuBuffer::zeros(
                    ordinal,
                    ScalarType::BF16,
                    &[1, nkv, current_cap, hd],
                )?);
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
                if let (Some(ref shadow_k), Some(ref shadow_v)) =
                    (&self.kv_shadow_k, &self.kv_shadow_v)
                {
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
            self.kv_cache_k = Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            self.kv_cache_v = Some(GpuBuffer::zeros(ordinal, kv_dtype, &[1, nkv, cap, hd])?);
            if kv_fp8 {
                // Scale buffers: [nkv, cap] of F32 — one scale per head per position
                self.kv_scale_k = Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
                self.kv_scale_v = Some(GpuBuffer::zeros(ordinal, ScalarType::F32, &[nkv, cap])?);
                if kv_fp8_bf16_sidecar_enabled() {
                    self.kv_shadow_k = Some(GpuBuffer::zeros(
                        ordinal,
                        ScalarType::BF16,
                        &[1, nkv, cap, hd],
                    )?);
                    self.kv_shadow_v = Some(GpuBuffer::zeros(
                        ordinal,
                        ScalarType::BF16,
                        &[1, nkv, cap, hd],
                    )?);
                    self.kv_shadow_start = self.kv_filled;
                }
            }
        }
        Ok(())
    }

    /// Record actual filled KV length (no reallocation).
    pub fn set_kv_filled(&mut self, filled: usize) {
        self.kv_filled = filled;
        if filled < self.certified_kv_key_tokens || filled < self.certified_kv_value_tokens {
            self.certified_kv_key_tokens = 0;
            self.certified_kv_value_tokens = 0;
            self.certified_kv_host_meta_blocks = 0;
            self.certified_kv_host_meta_key_stride_tokens = 0;
            self.certified_kv_host_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_meta_value_error_stride_blocks = 0;
            self.certified_kv_device_meta_key_scale_norm_blocks = 0;
            self.certified_kv_device_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_key_i8_cache.clear();
            self.certified_kv_host_key_scale_cache.clear();
            self.certified_kv_host_key_scale_channel_max_cache.clear();
            self.certified_kv_host_key_zero_cache.clear();
            self.certified_kv_host_value_error_cache.clear();
            self.certified_kv_host_value_norm_cache.clear();
        }
        if filled < self.certified_kv_host_tokens {
            self.certified_kv_host_tokens = 0;
        }
        if filled == 0 {
            self.certified_kv_promoted_key_cache = None;
            self.certified_kv_promoted_key_cache_tags_gpu = None;
            self.certified_kv_promoted_key_cache_lru_gpu = None;
            self.certified_kv_promoted_key_cache_capacity = 0;
            self.certified_kv_promoted_key_cache_tags.clear();
            self.certified_kv_promoted_key_cache_lru.clear();
            self.certified_kv_promoted_key_cache_tick = 0;
        }
        if filled == 0 {
            self.certified_kv_promoted_value_cache = None;
            self.certified_kv_promoted_value_cache_capacity = 0;
            self.certified_kv_promoted_value_cache_tags.clear();
            self.certified_kv_promoted_value_cache_lru.clear();
            self.certified_kv_promoted_value_cache_tick = 0;
        }
        if filled < self.certified_kv_host_meta_blocks {
            self.certified_kv_host_meta_blocks = 0;
            self.certified_kv_host_meta_key_stride_tokens = 0;
            self.certified_kv_host_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_meta_value_error_stride_blocks = 0;
            self.certified_kv_device_meta_key_scale_norm_blocks = 0;
            self.certified_kv_device_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_key_i8_cache.clear();
            self.certified_kv_host_key_scale_cache.clear();
            self.certified_kv_host_key_scale_channel_max_cache.clear();
            self.certified_kv_host_key_zero_cache.clear();
            self.certified_kv_host_value_error_cache.clear();
            self.certified_kv_host_value_norm_cache.clear();
        }
        if filled < self.certified_kv_ranking_prefix_tokens {
            self.certified_kv_ranking_prefix_k = None;
            self.certified_kv_ranking_prefix_v = None;
            self.certified_kv_ranking_prefix_tokens = 0;
            self.certified_kv_ranking_prefix_kv_heads.clear();
        }
        if filled == 0 {
            self.certified_kv_gpu_tail_only = false;
            self.certified_kv_host_meta_blocks = 0;
            self.certified_kv_host_meta_key_stride_tokens = 0;
            self.certified_kv_host_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_meta_value_error_stride_blocks = 0;
            self.certified_kv_device_meta_key_scale_norm_blocks = 0;
            self.certified_kv_device_meta_key_scale_stride_blocks = 0;
            self.certified_kv_host_key_i8_cache.clear();
            self.certified_kv_host_key_scale_cache.clear();
            self.certified_kv_host_key_scale_channel_max_cache.clear();
            self.certified_kv_host_key_zero_cache.clear();
            self.certified_kv_host_value_error_cache.clear();
            self.certified_kv_host_value_norm_cache.clear();
            self.certified_kv_promoted_key_cache = None;
            self.certified_kv_promoted_key_cache_tags_gpu = None;
            self.certified_kv_promoted_key_cache_lru_gpu = None;
            self.certified_kv_promoted_key_cache_capacity = 0;
            self.certified_kv_promoted_key_cache_tags.clear();
            self.certified_kv_promoted_key_cache_lru.clear();
            self.certified_kv_promoted_key_cache_tick = 0;
            self.certified_kv_promoted_value_cache = None;
            self.certified_kv_promoted_value_cache_capacity = 0;
            self.certified_kv_promoted_value_cache_tags.clear();
            self.certified_kv_promoted_value_cache_lru.clear();
            self.certified_kv_promoted_value_cache_tick = 0;
            self.certified_kv_ranking_prefix_k = None;
            self.certified_kv_ranking_prefix_v = None;
            self.certified_kv_ranking_prefix_tokens = 0;
            self.certified_kv_ranking_prefix_kv_heads.clear();
        }
        if self.kv_shadow_k.is_some() && self.kv_shadow_v.is_some() {
            self.kv_shadow_start = kv_fp8_bf16_sidecar_window_tokens()
                .map(|window| filled.saturating_sub(window))
                .unwrap_or(0);
        }
    }

    /// Get KV cache capacity (allocated seq dim).
    pub fn kv_capacity(&self) -> usize {
        self.virtual_kv_cache_k
            .as_ref()
            .map(|k| k.shape()[2])
            .or_else(|| self.kv_cache_k.as_ref().map(|k| k.shape()[2]))
            .unwrap_or(0)
    }

    pub fn enable_virtual_bf16_kv(&mut self, max_t: usize) {
        self.virtual_kv_max_t = Some(max_t.max(1));
    }

    pub fn disable_virtual_bf16_kv(&mut self) {
        self.virtual_kv_max_t = None;
        self.virtual_kv_cache_k = None;
        self.virtual_kv_guard = None;
        self.virtual_kv_cache_v = None;
        self.virtual_kv_full_backup = None;
        self.virtual_kv_logical_backup = None;
    }

    pub fn kv_cache_k_ptr(&self) -> Option<*mut c_void> {
        self.virtual_kv_cache_k
            .as_ref()
            .map(|b| b.as_ptr() as *mut c_void)
            .or_else(|| self.kv_cache_k.as_ref().map(|b| b.as_ptr() as *mut c_void))
    }

    pub fn kv_cache_v_ptr(&self) -> Option<*mut c_void> {
        self.virtual_kv_cache_v
            .as_ref()
            .map(|b| b.as_ptr() as *mut c_void)
            .or_else(|| {
                self.virtual_kv_cache_k
                    .as_ref()
                    .map(|b| b.offset_ptr(ops_byte_len_half(b)) as *mut c_void)
            })
            .or_else(|| self.kv_cache_v.as_ref().map(|b| b.as_ptr() as *mut c_void))
    }

    pub fn kv_cache_k_offset_ptr(&self, byte_offset: usize) -> Option<*const c_void> {
        self.virtual_kv_cache_k
            .as_ref()
            .map(|b| b.offset_ptr(byte_offset))
            .or_else(|| self.kv_cache_k.as_ref().map(|b| b.offset_ptr(byte_offset)))
    }

    pub fn kv_cache_v_offset_ptr(&self, byte_offset: usize) -> Option<*const c_void> {
        self.virtual_kv_cache_v
            .as_ref()
            .map(|b| b.offset_ptr(byte_offset))
            .or_else(|| {
                self.virtual_kv_cache_k
                    .as_ref()
                    .map(|b| b.offset_ptr(ops_byte_len_half(b) + byte_offset))
            })
            .or_else(|| self.kv_cache_v.as_ref().map(|b| b.offset_ptr(byte_offset)))
    }

    pub fn virtual_kv_cache_k_range_to_host(
        &self,
        byte_offset: usize,
        len: usize,
    ) -> Result<Option<Vec<u8>>, GpuError> {
        self.virtual_kv_cache_k
            .as_ref()
            .map(|b| b.to_host_range_bytes(byte_offset, len))
            .transpose()
    }

    pub fn virtual_kv_cache_v_range_to_host(
        &self,
        byte_offset: usize,
        len: usize,
    ) -> Result<Option<Vec<u8>>, GpuError> {
        self.virtual_kv_cache_v
            .as_ref()
            .map(|b| b.to_host_range_bytes(byte_offset, len))
            .or_else(|| {
                self.virtual_kv_cache_k
                    .as_ref()
                    .map(|b| b.to_host_range_bytes(ops_byte_len_half(b) + byte_offset, len))
            })
            .transpose()
    }

    pub fn has_virtual_kv_cache(&self) -> bool {
        self.virtual_kv_cache_k.is_some() || self.virtual_kv_cache_v.is_some()
    }

    pub fn virtual_kv_memory_stats(&self) -> Option<VirtualKvMemoryStats> {
        let k = self.virtual_kv_cache_k.as_ref()?;
        let v = self.virtual_kv_cache_v.as_ref();
        let k_stats = k.stats();
        let v_stats = v.map(VirtualBuffer::stats);
        let state_backup_bytes = self
            .virtual_kv_full_backup
            .as_ref()
            .map(|backup| backup.k.len() + backup.v.as_ref().map(Vec::len).unwrap_or(0))
            .or_else(|| {
                self.virtual_kv_logical_backup
                    .as_ref()
                    .map(|backup| backup.k.len() + backup.v.len())
            })
            .unwrap_or(0);
        Some(VirtualKvMemoryStats {
            layers: 1,
            logical_bytes: k_stats.logical_bytes
                + v_stats.map(|stats| stats.logical_bytes).unwrap_or(0),
            reserved_bytes: k_stats.reserved_bytes
                + v_stats.map(|stats| stats.reserved_bytes).unwrap_or(0),
            resident_bytes: k_stats.resident_bytes
                + v_stats.map(|stats| stats.resident_bytes).unwrap_or(0),
            logical_resident_bytes: k_stats.logical_resident_bytes
                + v_stats
                    .map(|stats| stats.logical_resident_bytes)
                    .unwrap_or(0),
            logical_backup_bytes: state_backup_bytes.max(
                k_stats.logical_backup_bytes
                    + v_stats.map(|stats| stats.logical_backup_bytes).unwrap_or(0),
            ),
            mappings: k_stats.mapping_count + v_stats.map(|stats| stats.mapping_count).unwrap_or(0),
        })
    }

    pub fn backup_virtual_kv_logical_prefix(
        &mut self,
        config: &TextConfig,
    ) -> Result<(), GpuError> {
        let Some(kv) = self.virtual_kv_cache_k.as_ref() else {
            return Ok(());
        };
        let prefix_len = self.kv_filled;
        if prefix_len == 0 {
            self.virtual_kv_logical_backup = Some(VirtualKvLogicalBackup {
                k: Vec::new(),
                v: Vec::new(),
                prefix_len: 0,
                cap: kv.shape()[2],
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
            });
            return Ok(());
        }

        let cap = kv.shape()[2];
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let src_head_stride = cap * head_dim * elem_bytes;
        let dst_head_stride = prefix_len * head_dim * elem_bytes;
        let copy_bytes = prefix_len * head_dim * elem_bytes;
        let mut k = vec![0u8; num_kv_heads * dst_head_stride];
        let mut v = vec![0u8; num_kv_heads * dst_head_stride];
        for h in 0..num_kv_heads {
            let src = h * src_head_stride;
            let dst = h * dst_head_stride;
            let k_head = kv.to_host_range_bytes(src, copy_bytes)?;
            let v_head = kv.to_host_range_bytes(ops_byte_len_half(kv) + src, copy_bytes)?;
            k[dst..dst + copy_bytes].copy_from_slice(&k_head);
            v[dst..dst + copy_bytes].copy_from_slice(&v_head);
        }
        self.virtual_kv_logical_backup = Some(VirtualKvLogicalBackup {
            k,
            v,
            prefix_len,
            cap,
            num_kv_heads,
            head_dim,
        });
        Ok(())
    }

    pub fn set_virtual_kv_logical_backup(
        &mut self,
        config: &TextConfig,
        k: Vec<u8>,
        v: Vec<u8>,
        prefix_len: usize,
    ) -> Result<(), GpuError> {
        let Some(kv) = self.virtual_kv_cache_k.as_ref() else {
            return Ok(());
        };
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let expected = config.num_key_value_heads * prefix_len * config.head_dim * elem_bytes;
        if k.len() != expected || v.len() != expected {
            return Err(GpuError::InvalidArg(format!(
                "virtual KV logical backup length mismatch: k={} v={} expected={expected}",
                k.len(),
                v.len()
            )));
        }
        self.virtual_kv_logical_backup = Some(VirtualKvLogicalBackup {
            k,
            v,
            prefix_len,
            cap: kv.shape()[2],
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        });
        self.virtual_kv_full_backup = None;
        Ok(())
    }

    pub fn restore_virtual_kv_logical_prefix(&mut self) -> Result<(), GpuError> {
        let Some(backup) = self.virtual_kv_logical_backup.as_ref() else {
            return Ok(());
        };
        let Some(kv) = self.virtual_kv_cache_k.as_mut() else {
            return Ok(());
        };
        kv.map_prefix_bytes(kv.len_bytes())?;
        if let Some(v_cache) = self.virtual_kv_cache_v.as_mut() {
            v_cache.map_prefix_bytes(v_cache.len_bytes())?;
        }
        if backup.prefix_len == 0 {
            return Ok(());
        }

        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let src_head_stride = backup.prefix_len * backup.head_dim * elem_bytes;
        let dst_head_stride = backup.cap * backup.head_dim * elem_bytes;
        let copy_bytes = src_head_stride;
        if let Some(v_cache) = self.virtual_kv_cache_v.as_mut() {
            let mut k_image = vec![0u8; kv.len_bytes()];
            let mut v_image = vec![0u8; v_cache.len_bytes()];
            for h in 0..backup.num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                k_image[dst..dst + copy_bytes].copy_from_slice(&backup.k[src..src + copy_bytes]);
                v_image[dst..dst + copy_bytes].copy_from_slice(&backup.v[src..src + copy_bytes]);
            }
            restore_virtual_kv_image(kv, &k_image, "K logical")?;
            restore_virtual_kv_image(v_cache, &v_image, "V logical")?;
        } else {
            let v_base = ops_byte_len_half(kv);
            let mut packed = vec![0u8; kv.len_bytes()];
            for h in 0..backup.num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                packed[dst..dst + copy_bytes].copy_from_slice(&backup.k[src..src + copy_bytes]);
                packed[v_base + dst..v_base + dst + copy_bytes]
                    .copy_from_slice(&backup.v[src..src + copy_bytes]);
            }
            restore_virtual_kv_image(kv, &packed, "packed logical")?;
        }
        sync(kv.device_ordinal())?;
        Ok(())
    }

    pub fn map_virtual_kv_logical_prefix_restore(&mut self) -> Result<(), GpuError> {
        if self.virtual_kv_logical_backup.is_none() {
            return Ok(());
        }
        if let Some(kv) = self.virtual_kv_cache_k.as_mut() {
            kv.map_prefix_bytes(kv.len_bytes())?;
        }
        if let Some(v_cache) = self.virtual_kv_cache_v.as_mut() {
            v_cache.map_prefix_bytes(v_cache.len_bytes())?;
        }
        Ok(())
    }

    pub fn copy_virtual_kv_logical_prefix_restore_mapped(&mut self) -> Result<(), GpuError> {
        let Some(backup) = self.virtual_kv_logical_backup.take() else {
            return Ok(());
        };
        let Some(kv) = self.virtual_kv_cache_k.as_mut() else {
            return Ok(());
        };
        if backup.prefix_len == 0 {
            sync(kv.device_ordinal())?;
            return Ok(());
        }

        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let src_head_stride = backup.prefix_len * backup.head_dim * elem_bytes;
        let dst_head_stride = backup.cap * backup.head_dim * elem_bytes;
        let copy_bytes = src_head_stride;
        if let Some(v_cache) = self.virtual_kv_cache_v.as_mut() {
            let mut k_image = vec![0u8; kv.len_bytes()];
            let mut v_image = vec![0u8; v_cache.len_bytes()];
            for h in 0..backup.num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                k_image[dst..dst + copy_bytes].copy_from_slice(&backup.k[src..src + copy_bytes]);
                v_image[dst..dst + copy_bytes].copy_from_slice(&backup.v[src..src + copy_bytes]);
            }
            restore_virtual_kv_image_mapped(kv, &k_image, "K logical")?;
            restore_virtual_kv_image_mapped(v_cache, &v_image, "V logical")?;
        } else {
            let v_base = ops_byte_len_half(kv);
            let mut packed = vec![0u8; kv.len_bytes()];
            for h in 0..backup.num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                packed[dst..dst + copy_bytes].copy_from_slice(&backup.k[src..src + copy_bytes]);
                packed[v_base + dst..v_base + dst + copy_bytes]
                    .copy_from_slice(&backup.v[src..src + copy_bytes]);
            }
            restore_virtual_kv_image_mapped(kv, &packed, "packed logical")?;
        }
        sync(kv.device_ordinal())?;
        Ok(())
    }

    pub fn restore_virtual_kv_logical_prefix_dense(&mut self) -> Result<(), GpuError> {
        let Some(backup) = self.virtual_kv_logical_backup.take() else {
            return Ok(());
        };
        let Some(kv) = self.virtual_kv_cache_k.as_ref() else {
            return Ok(());
        };
        let ordinal = kv.device_ordinal();
        if backup.prefix_len == 0 {
            self.kv_cache_k = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, backup.num_kv_heads, backup.cap, backup.head_dim],
            )?);
            self.kv_cache_v = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, backup.num_kv_heads, backup.cap, backup.head_dim],
            )?);
            self.virtual_kv_cache_k = None;
            self.virtual_kv_cache_v = None;
            self.virtual_kv_guard = None;
            self.virtual_kv_max_t = None;
            return Ok(());
        }

        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let src_head_stride = backup.prefix_len * backup.head_dim * elem_bytes;
        let dst_head_stride = backup.cap * backup.head_dim * elem_bytes;
        let copy_bytes = src_head_stride;
        let image_len = backup.num_kv_heads * dst_head_stride;
        let mut k_image = vec![0u8; image_len];
        let mut v_image = vec![0u8; image_len];
        for h in 0..backup.num_kv_heads {
            let src = h * src_head_stride;
            let dst = h * dst_head_stride;
            k_image[dst..dst + copy_bytes].copy_from_slice(&backup.k[src..src + copy_bytes]);
            v_image[dst..dst + copy_bytes].copy_from_slice(&backup.v[src..src + copy_bytes]);
        }

        let mut k_dense = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[1, backup.num_kv_heads, backup.cap, backup.head_dim],
        )?;
        let mut v_dense = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[1, backup.num_kv_heads, backup.cap, backup.head_dim],
        )?;
        copy_h2d(
            ordinal,
            k_dense.as_mut_ptr(),
            k_image.as_ptr() as *const c_void,
            k_image.len(),
        )?;
        copy_h2d(
            ordinal,
            v_dense.as_mut_ptr(),
            v_image.as_ptr() as *const c_void,
            v_image.len(),
        )?;
        sync(ordinal)?;
        self.kv_cache_k = Some(k_dense);
        self.kv_cache_v = Some(v_dense);
        self.virtual_kv_cache_k = None;
        self.virtual_kv_cache_v = None;
        self.virtual_kv_guard = None;
        self.virtual_kv_max_t = None;
        self.virtual_kv_full_backup = None;
        Ok(())
    }

    pub fn backup_virtual_kv_to_host(&mut self) -> Result<(), GpuError> {
        let Some(k_cache) = self.virtual_kv_cache_k.as_ref() else {
            return Ok(());
        };
        let k = k_cache.to_host_bytes()?;
        let v = self
            .virtual_kv_cache_v
            .as_ref()
            .map(VirtualBuffer::to_host_bytes)
            .transpose()?;
        self.virtual_kv_full_backup = Some(VirtualKvFullBackup { k, v });
        Ok(())
    }

    pub fn discard_virtual_kv_mapping(&mut self) -> Result<(), GpuError> {
        if let Some(k) = self.virtual_kv_cache_k.as_mut() {
            k.evict_discard()?;
        }
        if let Some(v) = self.virtual_kv_cache_v.as_mut() {
            v.evict_discard()?;
        }
        Ok(())
    }

    pub fn map_virtual_kv_full(&mut self) -> Result<(), GpuError> {
        if let Some(k) = self.virtual_kv_cache_k.as_mut() {
            k.map_prefix_bytes(k.len_bytes())?;
        }
        if let Some(v) = self.virtual_kv_cache_v.as_mut() {
            v.map_prefix_bytes(v.len_bytes())?;
        }
        Ok(())
    }

    pub fn evict_virtual_kv_to_host(&mut self, _config: &TextConfig) -> Result<(), GpuError> {
        self.backup_virtual_kv_to_host()?;
        self.discard_virtual_kv_mapping()
    }

    pub fn restore_virtual_kv_from_host(&mut self) -> Result<(), GpuError> {
        self.map_virtual_kv_restore_from_host()?;
        self.copy_virtual_kv_restore_from_host()
    }

    pub fn map_virtual_kv_restore_from_host(&mut self) -> Result<(), GpuError> {
        if self.virtual_kv_logical_backup.is_some() || self.virtual_kv_full_backup.is_none() {
            return Ok(());
        }
        if let Some(k) = self.virtual_kv_cache_k.as_mut() {
            k.map_prefix_bytes(k.len_bytes())?;
        }
        if let Some(v) = self.virtual_kv_cache_v.as_mut() {
            v.map_prefix_bytes(v.len_bytes())?;
        }
        Ok(())
    }

    pub fn copy_virtual_kv_restore_from_host(&mut self) -> Result<(), GpuError> {
        if self.virtual_kv_logical_backup.is_some() {
            return self.restore_virtual_kv_logical_prefix_dense();
        }
        let Some(backup) = self.virtual_kv_full_backup.take() else {
            return Ok(());
        };
        if let Some(k) = self.virtual_kv_cache_k.as_mut() {
            restore_virtual_kv_image_mapped(k, &backup.k, "K")?;
        }
        if let (Some(v), Some(v_backup)) = (self.virtual_kv_cache_v.as_mut(), backup.v.as_ref()) {
            restore_virtual_kv_image_mapped(v, v_backup, "V")?;
        }
        Ok(())
    }
}

fn ops_byte_len_half(buf: &VirtualBuffer) -> usize {
    buf.len_bytes() / 2
}

fn restore_virtual_kv_image(
    buf: &mut VirtualBuffer,
    image: &[u8],
    label: &'static str,
) -> Result<(), GpuError> {
    buf.map_prefix_bytes(buf.len_bytes())?;
    restore_virtual_kv_image_mapped(buf, image, label)
}

fn restore_virtual_kv_image_mapped(
    buf: &mut VirtualBuffer,
    image: &[u8],
    label: &'static str,
) -> Result<(), GpuError> {
    if image.len() != buf.len_bytes() {
        return Err(GpuError::InvalidArg(format!(
            "virtual KV {label} backup length {} does not match buffer length {}",
            image.len(),
            buf.len_bytes()
        )));
    }
    let mut restored = false;
    for _attempt in 0..3 {
        copy_h2d(
            buf.device_ordinal(),
            buf.as_mut_ptr(),
            image.as_ptr() as *const c_void,
            image.len(),
        )?;
        sync(buf.device_ordinal())?;
        let verify = buf.to_host_bytes()?;
        if verify == image {
            restored = true;
            break;
        }
    }
    if !restored {
        let verify = buf.to_host_bytes()?;
        let first_diff = verify
            .iter()
            .zip(image.iter())
            .position(|(got, expected)| got != expected);
        return Err(GpuError::InvalidArg(format!(
            "virtual KV {label} restore verification failed first_diff={first_diff:?}"
        )));
    }
    Ok(())
}

impl LayerState {
    pub fn resident_gpu_bytes(&self) -> usize {
        let mut total = 0usize;
        let mut add = |buf: &Option<GpuBuffer>| {
            if let Some(buf) = buf {
                total = total.saturating_add(buf.len_bytes());
            }
        };
        add(&self.kv_cache_k);
        add(&self.kv_cache_v);
        add(&self.kv_scale_k);
        add(&self.kv_scale_v);
        add(&self.kv_shadow_k);
        add(&self.kv_shadow_v);
        add(&self.certified_kv_key_i8);
        add(&self.certified_kv_key_scale);
        add(&self.certified_kv_key_zero);
        add(&self.certified_kv_value_i4);
        add(&self.certified_kv_value_scale);
        add(&self.certified_kv_value_zero);
        add(&self.certified_kv_value_error);
        add(&self.certified_kv_value_norm);
        add(&self.certified_kv_tail_k);
        add(&self.certified_kv_tail_v);
        add(&self.certified_kv_promoted_key_cache);
        add(&self.certified_kv_promoted_key_cache_tags_gpu);
        add(&self.certified_kv_promoted_key_cache_lru_gpu);
        add(&self.certified_kv_promoted_value_cache);
        add(&self.certified_kv_ranking_prefix_k);
        add(&self.certified_kv_ranking_prefix_v);
        add(&self.conv_state);
        add(&self.recurrent_state);
        total
    }

    /// Deep-copy all GPU buffers to create an independent clone.
    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let clone_opt = |opt: &Option<GpuBuffer>| -> Result<Option<GpuBuffer>, GpuError> {
            match opt {
                Some(buf) => Ok(Some(buf.clone_device()?)),
                None => Ok(None),
            }
        };
        let clone_host_opt = |opt: &Option<HostBuffer>| -> Result<Option<HostBuffer>, GpuError> {
            match opt {
                Some(buf) => Ok(Some(buf.clone_host()?)),
                None => Ok(None),
            }
        };
        Ok(Self {
            kind: self.kind,
            kv_cache_k: clone_opt(&self.kv_cache_k)?,
            kv_cache_v: clone_opt(&self.kv_cache_v)?,
            virtual_kv_cache_k: None,
            virtual_kv_guard: None,
            virtual_kv_cache_v: None,
            virtual_kv_max_t: if self.has_virtual_kv_cache() {
                return Err(GpuError::Unsupported(
                    "clone_gpu is not implemented for virtual KV cache state".into(),
                ));
            } else {
                self.virtual_kv_max_t
            },
            virtual_kv_full_backup: None,
            virtual_kv_logical_backup: None,
            kv_filled: self.kv_filled,
            kv_scale_k: clone_opt(&self.kv_scale_k)?,
            kv_scale_v: clone_opt(&self.kv_scale_v)?,
            kv_shadow_k: clone_opt(&self.kv_shadow_k)?,
            kv_shadow_v: clone_opt(&self.kv_shadow_v)?,
            kv_shadow_start: self.kv_shadow_start,
            certified_kv_key_i8: clone_opt(&self.certified_kv_key_i8)?,
            certified_kv_key_scale: clone_opt(&self.certified_kv_key_scale)?,
            certified_kv_key_zero: clone_opt(&self.certified_kv_key_zero)?,
            certified_kv_key_tokens: self.certified_kv_key_tokens,
            certified_kv_value_i4: clone_opt(&self.certified_kv_value_i4)?,
            certified_kv_value_scale: clone_opt(&self.certified_kv_value_scale)?,
            certified_kv_value_zero: clone_opt(&self.certified_kv_value_zero)?,
            certified_kv_value_error: clone_opt(&self.certified_kv_value_error)?,
            certified_kv_value_norm: clone_opt(&self.certified_kv_value_norm)?,
            certified_kv_value_tokens: self.certified_kv_value_tokens,
            certified_kv_host_k: clone_host_opt(&self.certified_kv_host_k)?,
            certified_kv_host_v: clone_host_opt(&self.certified_kv_host_v)?,
            certified_kv_host_tokens: self.certified_kv_host_tokens,
            certified_kv_host_meta_blocks: self.certified_kv_host_meta_blocks,
            certified_kv_host_meta_key_stride_tokens: self.certified_kv_host_meta_key_stride_tokens,
            certified_kv_host_meta_key_scale_stride_blocks: self
                .certified_kv_host_meta_key_scale_stride_blocks,
            certified_kv_host_meta_value_error_stride_blocks: self
                .certified_kv_host_meta_value_error_stride_blocks,
            certified_kv_device_meta_key_scale_norm_blocks: self
                .certified_kv_device_meta_key_scale_norm_blocks,
            certified_kv_device_meta_key_scale_stride_blocks: self
                .certified_kv_device_meta_key_scale_stride_blocks,
            certified_kv_host_key_i8_cache: self.certified_kv_host_key_i8_cache.clone(),
            certified_kv_host_key_scale_cache: self.certified_kv_host_key_scale_cache.clone(),
            certified_kv_host_key_scale_channel_max_cache: self
                .certified_kv_host_key_scale_channel_max_cache
                .clone(),
            certified_kv_host_key_zero_cache: self.certified_kv_host_key_zero_cache.clone(),
            certified_kv_host_value_error_cache: self.certified_kv_host_value_error_cache.clone(),
            certified_kv_host_value_norm_cache: self.certified_kv_host_value_norm_cache.clone(),
            certified_kv_tail_k: clone_opt(&self.certified_kv_tail_k)?,
            certified_kv_tail_v: clone_opt(&self.certified_kv_tail_v)?,
            certified_kv_gpu_tail_only: self.certified_kv_gpu_tail_only,
            certified_kv_promoted_key_cache: clone_opt(&self.certified_kv_promoted_key_cache)?,
            certified_kv_promoted_key_cache_tags_gpu: clone_opt(
                &self.certified_kv_promoted_key_cache_tags_gpu,
            )?,
            certified_kv_promoted_key_cache_lru_gpu: clone_opt(
                &self.certified_kv_promoted_key_cache_lru_gpu,
            )?,
            certified_kv_promoted_key_cache_capacity: self.certified_kv_promoted_key_cache_capacity,
            certified_kv_promoted_key_cache_tags: self.certified_kv_promoted_key_cache_tags.clone(),
            certified_kv_promoted_key_cache_lru: self.certified_kv_promoted_key_cache_lru.clone(),
            certified_kv_promoted_key_cache_tick: self.certified_kv_promoted_key_cache_tick,
            certified_kv_promoted_value_cache: clone_opt(&self.certified_kv_promoted_value_cache)?,
            certified_kv_promoted_value_cache_capacity: self
                .certified_kv_promoted_value_cache_capacity,
            certified_kv_promoted_value_cache_tags: self
                .certified_kv_promoted_value_cache_tags
                .clone(),
            certified_kv_promoted_value_cache_lru: self
                .certified_kv_promoted_value_cache_lru
                .clone(),
            certified_kv_promoted_value_cache_tick: self.certified_kv_promoted_value_cache_tick,
            certified_kv_ranking_prefix_k: clone_opt(&self.certified_kv_ranking_prefix_k)?,
            certified_kv_ranking_prefix_v: clone_opt(&self.certified_kv_ranking_prefix_v)?,
            certified_kv_ranking_prefix_tokens: self.certified_kv_ranking_prefix_tokens,
            certified_kv_ranking_prefix_kv_heads: self.certified_kv_ranking_prefix_kv_heads.clone(),
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

    /// Prepare existing buffers for a from-scratch prefill replay without
    /// reallocating caches that can safely be overwritten in place.
    pub fn reset_for_prefill_reuse(&mut self) {
        for ls in &mut self.layers {
            ls.kv_filled = 0;
            ls.kv_shadow_start = usize::MAX;
            ls.kv_scale_k = None;
            ls.kv_scale_v = None;
            ls.kv_shadow_k = None;
            ls.kv_shadow_v = None;

            let reusable_bf16_cache = ls
                .kv_cache_k
                .as_ref()
                .zip(ls.kv_cache_v.as_ref())
                .is_some_and(|(k, v)| {
                    k.dtype() == ScalarType::BF16 && v.dtype() == ScalarType::BF16
                });
            if !reusable_bf16_cache && !ls.has_virtual_kv_cache() {
                ls.kv_cache_k = None;
                ls.kv_cache_v = None;
            }
        }
    }

    pub fn enable_virtual_bf16_kv(&mut self, config: &TextConfig, max_t: usize) {
        for (idx, ls) in self.layers.iter_mut().enumerate() {
            if config.is_full_attention(idx) {
                ls.enable_virtual_bf16_kv(max_t);
            }
        }
    }

    pub fn virtual_kv_memory_stats(&self) -> VirtualKvMemoryStats {
        self.layers
            .iter()
            .filter_map(LayerState::virtual_kv_memory_stats)
            .fold(VirtualKvMemoryStats::default(), |mut acc, stats| {
                acc.layers += stats.layers;
                acc.logical_bytes += stats.logical_bytes;
                acc.reserved_bytes += stats.reserved_bytes;
                acc.resident_bytes += stats.resident_bytes;
                acc.logical_resident_bytes += stats.logical_resident_bytes;
                acc.logical_backup_bytes += stats.logical_backup_bytes;
                acc.mappings += stats.mappings;
                acc
            })
    }

    pub fn virtual_kv_memory_stats_by_layer(&self) -> Vec<(usize, VirtualKvMemoryStats)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(idx, layer)| layer.virtual_kv_memory_stats().map(|stats| (idx, stats)))
            .collect()
    }

    pub fn evict_virtual_kv_to_host(&mut self, _config: &TextConfig) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            layer.backup_virtual_kv_to_host()?;
        }
        for layer in &mut self.layers {
            layer.discard_virtual_kv_mapping()?;
        }
        Ok(())
    }

    pub fn evict_virtual_kv_to_host_from_snapshots(
        &mut self,
        config: &TextConfig,
        snapshots: Vec<(usize, Vec<u8>, Vec<u8>, usize)>,
    ) -> Result<(), GpuError> {
        for (idx, k, v, prefix_len) in snapshots {
            if let Some(layer) = self.layers.get_mut(idx) {
                layer.set_virtual_kv_logical_backup(config, k, v, prefix_len)?;
            }
        }
        for layer in &mut self.layers {
            layer.discard_virtual_kv_mapping()?;
        }
        Ok(())
    }

    pub fn restore_virtual_kv_from_host(&mut self) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            layer.map_virtual_kv_restore_from_host()?;
        }
        for layer in &mut self.layers {
            layer.copy_virtual_kv_restore_from_host()?;
        }
        Ok(())
    }

    pub fn restore_virtual_kv_from_host_to_vmm(&mut self) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            if layer.virtual_kv_logical_backup.is_some() {
                layer.map_virtual_kv_logical_prefix_restore()?;
            } else {
                layer.map_virtual_kv_restore_from_host()?;
            }
        }
        for layer in &mut self.layers {
            if layer.virtual_kv_logical_backup.is_some() {
                layer.copy_virtual_kv_logical_prefix_restore_mapped()?;
            } else {
                layer.copy_virtual_kv_restore_from_host()?;
            }
        }
        Ok(())
    }

    /// Deep-copy all layer states to create an independent clone.
    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            layers.push(ls.clone_gpu()?);
        }
        Ok(Self { layers })
    }

    pub fn resident_gpu_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(LayerState::resident_gpu_bytes)
            .fold(0usize, usize::saturating_add)
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
            match (
                ls.kind,
                &ls.conv_state,
                &ls.recurrent_state,
                &expected_per_layer[i],
            ) {
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
                _ => panic!("layer {i}: kind/state/snapshot-slot inconsistency after restore"),
            }
        }

        // Sanity: a second snapshot + restore round-trip on the restored state
        // is a no-op.
        let snap2 = state.snapshot_linear().expect("snapshot_linear 2nd");
        state
            .restore_linear(&snap2, ordinal)
            .expect("restore_linear 2nd");
        for (i, ls) in state.layers.iter().enumerate() {
            if let (LayerKind::Linear, Some(conv), Some(rec), Some((conv_a, rec_a))) = (
                ls.kind,
                &ls.conv_state,
                &ls.recurrent_state,
                &expected_per_layer[i],
            ) {
                let conv_rb = conv.to_host_bytes().expect("d2h conv 2nd");
                let rec_rb = rec.to_host_bytes().expect("d2h rec 2nd");
                assert_eq!(&conv_rb, conv_a, "layer {i}: conv drift on 2nd roundtrip");
                assert_eq!(&rec_rb, rec_a, "layer {i}: rec drift on 2nd roundtrip");
            }
        }
    }
}
