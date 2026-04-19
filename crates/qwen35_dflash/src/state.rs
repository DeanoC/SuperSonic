//! Pre-allocated per-round scratch buffers for the DFlash draft forward pass.
//!
//! Batch-outer = 1 (one speculative round at a time). `q_len` is the draft's
//! block size (16). The attention's effective K/V sequence length is
//! `ctx_len + q_len` where `ctx_len = block_size` too (the fused tap vector
//! is tiled to match `q_len`). See docs/dflash.md §3, §4.

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
