//! Pre-allocated per-round scratch buffers + per-layer persistent KV caches
//! for the DFlash draft forward pass.
//!
//! Split responsibility:
//!   - [`DFlashScratch`] — per-round transient buffers (norm_concat, q/k/v
//!     projection outputs, attention output, MLP intermediates, final hidden,
//!     logits). Overwritten every `forward()` call.
//!   - [`DFlashState`] — persistent across rounds. Per-layer K/V caches that
//!     get appended on each forward and cropped after each speculative round
//!     (see docs/dflash.md §5.4).
//!
//! Batch-outer = 1 (one speculative round at a time). `q_len` is the draft's
//! block size (16); `ctx_len` varies per round in 1..block_size.

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

use crate::config::DFlashConfig;

pub struct DFlashScratch {
    pub ordinal: usize,
    pub block_size: usize,

    // Fuser path: input is tap-vectors concatenated along hidden dim.
    // Shape [1, block_size, num_taps * hidden] before fc; re-used as
    // [1, block_size, hidden] afterwards by `target_hidden_ctx`.
    pub fuser_input: GpuBuffer,
    pub target_hidden_ctx: GpuBuffer,
    pub target_hidden_ctx_norm: GpuBuffer,

    // Per-layer activations. Two hidden buffers so forward can swap them
    // without an intra-kernel copy during the residual add.
    pub hidden_a: GpuBuffer,
    pub hidden_b: GpuBuffer,
    pub hidden_norm: GpuBuffer,
    pub post_attn_norm: GpuBuffer,

    // Attention projections.
    //   q:        [1, q_len, q_out_dim]
    //   norm_concat: [1, ctx_len + q_len, hidden]  — [ctx_norm || noise_norm]
    //   k/v concat: [1, ctx_len + q_len, kv_out_dim]
    pub q_proj: GpuBuffer,
    pub norm_concat: GpuBuffer,
    pub k_concat: GpuBuffer,
    pub v_concat: GpuBuffer,

    // Attention output: [1, q_len, q_out_dim].
    pub attn_out: GpuBuffer,

    // MLP intermediates: [1, q_len, intermediate].
    pub gate: GpuBuffer,
    pub up: GpuBuffer,
    pub swiglu_out: GpuBuffer,

    // Final hidden (pre-lm_head) and logits.
    pub final_hidden: GpuBuffer,
    pub logits: GpuBuffer,

    // Small scratch for the work-stealing matvec counter.
    pub matvec_counter: GpuBuffer,
}

impl DFlashScratch {
    pub fn new(ordinal: usize, config: &DFlashConfig) -> Result<Self, GpuError> {
        let q_len = config.block_size;
        let ctx_len = config.block_size;
        let hidden = config.hidden_size;
        let q_out = config.q_out_dim();
        let kv_out = config.kv_out_dim();
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;
        let num_taps_hidden = config.fuser_in_dim();

        Ok(Self {
            ordinal,
            block_size: config.block_size,

            fuser_input: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, ctx_len, num_taps_hidden])?,
            target_hidden_ctx: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, ctx_len, hidden])?,
            target_hidden_ctx_norm: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, ctx_len, hidden],
            )?,

            hidden_a: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, hidden])?,
            hidden_b: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, hidden])?,
            hidden_norm: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, hidden])?,
            post_attn_norm: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, hidden])?,

            q_proj: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, q_out])?,
            norm_concat: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, ctx_len + q_len, hidden])?,
            k_concat: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, ctx_len + q_len, kv_out])?,
            v_concat: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, ctx_len + q_len, kv_out])?,

            attn_out: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, q_out])?,

            gate: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, intermediate])?,
            up: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, intermediate])?,
            swiglu_out: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, intermediate])?,

            final_hidden: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_len, hidden])?,
            logits: GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, q_len, vocab])?,

            matvec_counter: GpuBuffer::zeros(ordinal, ScalarType::F32, &[1])?,
        })
    }
}

/// Persistent per-layer KV cache for the DFlash draft.
///
/// Physical layout is `[max_ctx, num_kv_heads, head_dim]` BF16, matching the
/// SHD layout expected by the bidirectional attention helper. The kernel
/// only reads `[0..kv_filled + seq]` rows on each forward, so unused slots
/// beyond the cursor are harmless.
pub struct DFlashLayerKv {
    pub cache_k: GpuBuffer,
    pub cache_v: GpuBuffer,
}

/// All persistent (across-round) state the draft owns.
///
/// `kv_filled` is the number of real positions currently stored in every
/// layer's cache — all layers move in lockstep. `max_ctx` is the physical
/// capacity.
///
/// Lifecycle per speculative round (see docs/dflash.md §5.4):
///   1. forward() appends `ctx_len + q_len` positions to every layer.
///   2. After acceptance, the engine calls [`DFlashState::crop`] with the
///      new committed length, rolling back the unused ctx+noise tail.
///   3. Next round starts from `kv_filled = committed_length`.
pub struct DFlashState {
    pub ordinal: usize,
    pub layers: Vec<DFlashLayerKv>,
    pub kv_filled: usize,
    pub max_ctx: usize,
}

impl DFlashState {
    pub fn new(
        ordinal: usize,
        config: &DFlashConfig,
        max_ctx: usize,
    ) -> Result<Self, GpuError> {
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DFlashLayerKv {
                cache_k: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[max_ctx, nkv, hd])?,
                cache_v: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[max_ctx, nkv, hd])?,
            });
        }
        Ok(Self { ordinal, layers, kv_filled: 0, max_ctx })
    }

    /// Roll back the logical fill cursor. Physical memory beyond the cursor
    /// is untouched — it will be overwritten on the next append. Per the
    /// canonical `past_key_values_draft.crop(start)` pattern.
    pub fn crop(&mut self, keep: usize) {
        if keep < self.kv_filled {
            self.kv_filled = keep;
        }
    }

    /// Discard all cached positions. Equivalent to starting a fresh sequence.
    pub fn reset(&mut self) {
        self.kv_filled = 0;
    }
}
