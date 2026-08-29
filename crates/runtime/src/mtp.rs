//! Qwen3.8 NextN/MTP verification state and reusable workspaces.
//!
//! This module owns the allocations that are shared by the retained MTP
//! verify paths.  The decode engine still owns the surrounding acceptance
//! policy; only the reusable MTP buffers and the state-restore boundary live
//! here.

use anyhow::Result;
use gpu_hal::{GpuBuffer, ScalarType};
use qwen38::config::TextConfig;
use qwen38::rotary::RotaryTables;
use qwen38::state::{LinearStateSnapshot, ModelState};
use qwen38::weights::Qwen38Weights;

use crate::prefill_engine::{PrefillAppendVerifyResult, PrefillScratch};

/// Reusable workspace for fused multi-token MTP verification.
///
/// The live decode engine remains a single-sequence engine.  This cache owns
/// the B-sized buffers needed by one fused verification launch and is reused
/// when consecutive rounds use the same block size.
pub struct MtpVerifyCache {
    pub(crate) block_size: usize,
    pub(crate) workspace: GpuBuffer,
    pub(crate) hidden_io: GpuBuffer,
    pub(crate) normed_buf: GpuBuffer,
    pub(crate) logits_buf: GpuBuffer,
    pub(crate) argmax_buf: GpuBuffer,
    pub(crate) batch_desc_device: GpuBuffer,
}

impl MtpVerifyCache {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alloc(
        ordinal: usize,
        block_size: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
    ) -> Result<Self> {
        // Keep this layout in lockstep with PersistentDecodeScratch.  The
        // fused kernel indexes one per-item segment for each verify row.
        let per_item_floats = hidden_dim
            + hidden_dim
            + intermediate_size * 2
            + hidden_dim
            + hidden_dim
            + proj_buf_floats
            + attn_scratch_floats;
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[per_item_floats * block_size])
            .map_err(|e| anyhow::anyhow!("mtp verify workspace alloc: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("mtp verify hidden_io alloc: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("mtp verify normed_buf alloc: {e}"))?;
        let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, vocab_size])
            .map_err(|e| anyhow::anyhow!("mtp verify logits_buf alloc: {e}"))?;
        let argmax_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[block_size])
            .map_err(|e| anyhow::anyhow!("mtp verify argmax_buf alloc: {e}"))?;
        let batch_desc_bytes = num_layers * std::mem::size_of::<kernel_ffi::BatchSeqDesc>();
        let batch_desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[batch_desc_bytes])
            .map_err(|e| anyhow::anyhow!("mtp verify batch desc alloc: {e}"))?;
        Ok(Self {
            block_size,
            workspace,
            hidden_io,
            normed_buf,
            logits_buf,
            argmax_buf,
            batch_desc_device,
        })
    }
}

/// Reusable one-token scratch used by the MTP draft/verify component path.
///
/// The name describes its retained consumer rather than the historical
/// backend implementation.  The buffers are backend-neutral at this layer;
/// the active HIP kernel path is selected by the prefill helpers.
pub struct MtpVerifyScratch {
    pub(crate) scratch: PrefillScratch,
    pub(crate) chunk_conv_tail: Vec<Option<GpuBuffer>>,
    pub(crate) token_id_buf: GpuBuffer,
    pub(crate) mtp_residual: GpuBuffer,
    pub(crate) mtp_logits: GpuBuffer,
    pub(crate) mtp_argmax: GpuBuffer,
    pub(crate) mtp_counter: GpuBuffer,
}

impl MtpVerifyScratch {
    pub fn new(config: &TextConfig, ordinal: usize) -> Result<Self> {
        let scratch = PrefillScratch::new(config, 1, ordinal)?;
        let kern = config.linear_conv_kernel_dim;
        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;

        let chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_full_attention(i) {
                    Ok(None)
                } else {
                    GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                        .map(Some)
                        .map_err(|e| anyhow::anyhow!("mtp verify conv tail layer {i} alloc: {e}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let token_id_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("mtp verify token id buffer: {e}"))?;
        let mtp_residual = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("mtp verify residual scratch: {e}"))?;
        let mtp_logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.vocab_size])
            .map_err(|e| anyhow::anyhow!("mtp verify logits scratch: {e}"))?;
        let mtp_argmax = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("mtp verify argmax scratch: {e}"))?;
        let mtp_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("mtp verify counter scratch: {e}"))?;

        Ok(Self {
            scratch,
            chunk_conv_tail,
            token_id_buf,
            mtp_residual,
            mtp_logits,
            mtp_argmax,
            mtp_counter,
        })
    }

    /// Copy the post-layer residual into a caller-owned MTP hidden buffer.
    pub fn copy_last_residual_to(&self, ordinal: usize, dst: &mut GpuBuffer) -> Result<()> {
        self.scratch.copy_hidden_to(ordinal, dst)
    }
}

/// Cached prefill-append buffers used by the Qwen3.8 MTP verifier.
pub struct MtpPrefillAppendCache {
    pub(crate) chunk_len: usize,
    pub(crate) ordinal: usize,
    pub(crate) scratch: PrefillScratch,
    pub(crate) chunk_conv_tail: Vec<Option<GpuBuffer>>,
    pub(crate) token_ids_gpu: GpuBuffer,
}

impl MtpPrefillAppendCache {
    pub fn new(config: &TextConfig, chunk_len: usize, ordinal: usize) -> Result<Self> {
        let kern = config.linear_conv_kernel_dim;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;
        let qkv_dim = config.linear_num_key_heads * khd * 2 + config.linear_num_value_heads * vhd;
        let scratch = PrefillScratch::new(config, chunk_len, ordinal)?;
        let chunk_conv_tail: Vec<Option<GpuBuffer>> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_full_attention(i) {
                    Ok(None)
                } else {
                    GpuBuffer::alloc(ordinal, ScalarType::BF16, &[qkv_dim, kern - 1])
                        .map(Some)
                        .map_err(|e| anyhow::anyhow!("mtp append conv tail layer {i}: {e}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids_gpu = GpuBuffer::alloc(ordinal, ScalarType::U32, &[chunk_len])
            .map_err(|e| anyhow::anyhow!("mtp append token ids alloc: {e}"))?;

        Ok(Self {
            chunk_len,
            ordinal,
            scratch,
            chunk_conv_tail,
            token_ids_gpu,
        })
    }

    pub(crate) fn matches(&self, chunk_len: usize, ordinal: usize) -> bool {
        self.chunk_len == chunk_len && self.ordinal == ordinal
    }
}

/// Restore linear-attention state after a partial MTP acceptance decision.
pub(crate) fn restore_linear_state(
    state: &mut ModelState,
    snapshot: &LinearStateSnapshot,
    ordinal: usize,
) -> Result<()> {
    state
        .restore_linear(snapshot, ordinal)
        .map_err(|e| anyhow::anyhow!("mtp restore linear state: {e}"))
}

/// Ask the HIP bridge to restore its captured linear prefix, when available.
/// The external symbol spelling remains stable at this boundary.
pub(crate) fn restore_linear_prefix(
    ordinal: usize,
    layers: &GpuBuffer,
    commit_len: usize,
) -> Result<bool> {
    kernel_ffi::mtp_restore_linear_prefix(ordinal, layers, commit_len)
        .map_err(|e| anyhow::anyhow!("mtp restore linear prefix: {e}"))
}

/// MTP-owned entry point for the cached prefill-append verifier.
#[allow(clippy::too_many_arguments)]
pub fn prefill_append_verify_cached(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    token_ids: &[u32],
    pos_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    greedy_only: bool,
    greedy_compare_tokens: Option<&[u32]>,
    cache: &mut MtpPrefillAppendCache,
    dflash_capture: Option<&mut crate::prefill_engine::DflashTargetCapture>,
    rollback_capture: Option<&mut crate::prefill_engine::DflashRollbackCapture>,
) -> Result<PrefillAppendVerifyResult> {
    crate::prefill_engine::prefill_append_verify_cached(
        weights,
        state,
        rotary,
        token_ids,
        pos_offset,
        ordinal,
        kv_chunk_size,
        use_4b_kernel,
        greedy_only,
        greedy_compare_tokens,
        cache,
        dflash_capture,
        rollback_capture,
    )
}

/// MTP-owned entry point for the NextN forward helper.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    h: &GpuBuffer,
    h_is_nextn: bool,
    token_id: u32,
    abs_pos: usize,
    out_h: &mut GpuBuffer,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    crate::prefill_engine::mtp_forward(
        weights,
        state,
        rotary,
        scratch,
        h,
        h_is_nextn,
        token_id,
        abs_pos,
        out_h,
        ordinal,
        kv_chunk_size,
    )
}

/// MTP-owned entry point for the diagnostic one-step draft helper.
pub fn mtp_draft_greedy(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    h: &GpuBuffer,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    crate::prefill_engine::mtp_draft_greedy(
        weights,
        state,
        rotary,
        scratch,
        h,
        token_id,
        seqlen_offset,
        ordinal,
        kv_chunk_size,
    )
}

/// MTP-owned entry point for one-token component decode with full logits.
#[allow(clippy::too_many_arguments)]
pub fn mtp_decode_step(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<Vec<f32>> {
    crate::prefill_engine::mtp_decode_step(
        weights,
        state,
        rotary,
        scratch,
        token_id,
        seqlen_offset,
        ordinal,
        kv_chunk_size,
    )
}

/// MTP-owned entry point for one-token component decode with fused argmax.
pub fn mtp_decode_step_greedy(
    weights: &Qwen38Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut MtpVerifyScratch,
    token_id: u32,
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<u32> {
    crate::prefill_engine::mtp_decode_step_greedy(
        weights,
        state,
        rotary,
        scratch,
        token_id,
        seqlen_offset,
        ordinal,
        kv_chunk_size,
    )
}
