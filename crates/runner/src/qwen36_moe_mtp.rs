//! Qwen3.6-MoE multi-token-prediction (MTP) forward pass — Phase 6.2c.2.
//!
//! Builds on top of the Phase 6.2b weight loader (`MtpLayerBuffers`) and
//! the Phase 6.2c.1 pre-fusion kernel. Drives one MTP draft step's
//! single-layer transformer block (full-attention + MoE FFN) by
//! reusing the existing per-block FFI launchers
//! [`kernel_ffi::qwen36_moe::attn_step_launch`] and
//! [`kernel_ffi::qwen36_moe::ffn_step_launch`] — the same kernels the
//! base-model decode chain calls. The MTP block is structurally one
//! Qwen3.6 full-attention layer, so the only differences from a base
//! step are:
//!
//!   1. The cache is per-MTP-session (fresh per draft chain).
//!   2. RoPE rotates at absolute position `base_seq_len + k` while the
//!      cache slot is just `k` — handled by the `cache_pos` parameter
//!      added to [`Qwen36MoeAttnStepParams`] in this PR.
//!
//! The pre-fusion kernel (Phase 6.2c.1) feeds `fused_in` into this
//! function; the post-norm + lm_head (Phase 6.2c.3) consumes the output
//! and produces the next draft token. Nothing in the production decode
//! path calls this module today — wiring lands in the speculative driver
//! (Phase 6.3).

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2d, GpuBuffer, ScalarType};
use kernel_ffi::qwen36_moe::{
    attn_step_launch, ffn_step_launch, Qwen36MoeAttnStepInt4, Qwen36MoeAttnStepParams,
    Qwen36MoeAttnStepWeights, Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights,
};
use std::ptr;

use crate::qwen36_moe_decode::{
    ffn_output_elems, ffn_workspace_floats, full_attn_output_elems, full_attn_workspace_floats,
    reset_sync_buf, MtpLayerBuffers, MultiLayerGeom,
};

/// Pre-allocated GPU scratch buffers for one MTP forward call. The MTP
/// session reuses the same scratch across all `K` draft steps — allocate
/// once at the start of a speculative pass, drop when the pass ends.
///
/// All buffers live on `ordinal`; the caller must not mix scratches
/// across devices.
pub struct MtpForwardScratch {
    /// Stage-5 attn output. Sized for the wider of (full, linear) attn
    /// staged intermediates so the same allocator can serve both base
    /// and MTP-style chains; the MTP path always uses the full-attn
    /// half. BF16, `[full_attn_output_elems(geom)]` elements.
    pub attn_output: GpuBuffer,

    /// F32 scratch consumed by the attn kernel. Sized for the largest
    /// stage with optional KV-cache score region (`H * kv_max_t`).
    pub attn_workspace: GpuBuffer,

    /// Mid-layer residual (input to the FFN). The attn kernel publishes
    /// `input + o_proj(...)` into the leading `hidden` BF16 elements of
    /// `attn_output`; we D2D-copy that into `attn_residual` before
    /// calling the FFN, exactly mirroring the base-decode chain's
    /// hidden_a/hidden_b ping-pong (only here we ping-pong via two
    /// dedicated buffers since the MTP pass writes its final output
    /// into the caller's `out`).
    pub attn_residual: GpuBuffer,

    /// Stage-5 FFN output. BF16, `[ffn_output_elems(geom)]` elements.
    pub ffn_output: GpuBuffer,

    /// FFN top-k indices buffer. U32, `[top_k]` elements.
    pub ffn_output_idx: GpuBuffer,

    /// FFN F32 workspace, sized for the per-stage footprint.
    pub ffn_workspace: GpuBuffer,

    /// 96-byte zeroed scratch for the cooperative-launch counters +
    /// barrier state. Must be reset between attn and FFN launches.
    pub sync_buf: GpuBuffer,
}

/// Allocate a fresh [`MtpForwardScratch`] sized for `geom`. `kv_max_t` is
/// the MTP cache capacity in tokens (typically `K`, the
/// `num_speculative_tokens` in the speculative driver). It only affects
/// the F32 attn workspace size — the cache buffers themselves live
/// inside `MtpLayerBuffers::kv_cache`, allocated by the Phase 6.2b
/// loader.
pub fn alloc_mtp_forward_scratch(
    ordinal: usize,
    geom: &MultiLayerGeom,
    kv_max_t: usize,
) -> Result<MtpForwardScratch> {
    let attn_extra = if kv_max_t > 0 {
        (geom.num_attention_heads as usize) * kv_max_t
    } else {
        0
    };
    let attn_output = GpuBuffer::zeros(
        ordinal,
        ScalarType::BF16,
        &[full_attn_output_elems(geom)],
    )
    .context("alloc mtp attn_output")?;
    let attn_workspace = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[full_attn_workspace_floats(geom) + attn_extra],
    )
    .context("alloc mtp attn_workspace")?;
    let attn_residual = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.hidden as usize])
        .context("alloc mtp attn_residual")?;
    let ffn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_output_elems(geom)])
        .context("alloc mtp ffn_output")?;
    let ffn_output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
        .context("alloc mtp ffn_output_idx")?;
    let ffn_workspace = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[ffn_workspace_floats(geom)],
    )
    .context("alloc mtp ffn_workspace")?;
    let sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])
        .context("alloc mtp sync_buf")?;

    Ok(MtpForwardScratch {
        attn_output,
        attn_workspace,
        attn_residual,
        ffn_output,
        ffn_output_idx,
        ffn_workspace,
        sync_buf,
    })
}

/// Run one MTP-layer forward step.
///
/// Computes the byte-equivalent of the vLLM
/// `Qwen3NextMultiTokenPredictor` post-fusion stage:
///
/// ```text
/// out = mtp.layers.0(fused_in, position=base_seq_len+k, kv=mtp_kv)
/// ```
///
/// Internally:
///   1. `attn_step_launch` (stage 5, full-attn path) on `fused_in` →
///      `attn_residual = fused_in + o_proj(...)`. Writes K/V at MTP
///      cache slot `cache_pos = k`, attends over `kv_len = k + 1`.
///   2. `ffn_step_launch` (stage 5) on `attn_residual` →
///      `out = attn_residual + moe_out + shared_out`.
///
/// `position` is the absolute RoPE position (`base_seq_len + k`).
/// `cache_pos` is the MTP cache slot for this draft step (`k` in 0..K).
/// `out` is filled with the layer's final BF16 hidden state, matching
/// the oracle's `attn_out` field per draft step.
#[allow(clippy::too_many_arguments)]
pub fn run_mtp_layer_step(
    ordinal: usize,
    geom: &MultiLayerGeom,
    mtp: &mut MtpLayerBuffers,
    position: i32,
    cache_pos: i32,
    fused_in: &GpuBuffer,
    out: &mut GpuBuffer,
    scratch: &mut MtpForwardScratch,
) -> Result<()> {
    if cache_pos < 0 {
        return Err(anyhow!(
            "run_mtp_layer_step: cache_pos must be ≥ 0 (got {cache_pos}); \
             the kernel sentinel `-1` means \"inherit position\" which \
             defeats MTP's per-session cache semantics."
        ));
    }
    let hidden = geom.hidden as usize;

    // ---- attention ----------------------------------------------------
    reset_sync_buf(ordinal, &mut scratch.sync_buf)
        .context("reset sync_buf (mtp attn)")?;
    let (kv_k_ptr, kv_v_ptr, kv_max_t) = match &mut mtp.kv_cache {
        Some(c) => (c.k.as_mut_ptr(), c.v.as_mut_ptr(), c.kv_max_t),
        None => (ptr::null_mut(), ptr::null_mut(), 0),
    };
    let attn_params = Qwen36MoeAttnStepParams {
        stage: 5,
        hidden: geom.hidden,
        num_heads: geom.num_attention_heads,
        num_kv_heads: geom.num_kv_heads,
        head_dim: geom.head_dim,
        rotary_dim: geom.rotary_dim,
        rope_theta: geom.rope_theta,
        rms_norm_eps: geom.rms_norm_eps,
        position,
        cache_pos,
    };
    let attn_weights = Qwen36MoeAttnStepWeights {
        input_hidden: fused_in.as_ptr(),
        input_norm_w: mtp.input_norm_w.as_ptr(),
        q_proj_w: mtp.q_proj_w.as_ptr(),
        k_proj_w: mtp.k_proj_w.as_ptr(),
        v_proj_w: mtp.v_proj_w.as_ptr(),
        q_norm_w: mtp.q_norm_w.as_ptr(),
        k_norm_w: mtp.k_norm_w.as_ptr(),
        o_proj_w: mtp.o_proj_w.as_ptr(),
        kv_cache_k: kv_k_ptr,
        kv_cache_v: kv_v_ptr,
        kv_max_t,
    };
    attn_step_launch(
        ordinal,
        ScalarType::BF16,
        attn_params,
        &attn_weights,
        // BF16 path only — the production-bake INT4 path is a
        // follow-up; the MTP weights are emitted as raw BF16 by the
        // current `oracle/bake_int4.py` pass-through, and even if we
        // bake INT4 later, that's a parallel sidecar struct switch.
        &Qwen36MoeAttnStepInt4::disabled(),
        &mut scratch.attn_output,
        &mut scratch.attn_workspace,
        &mut scratch.sync_buf,
    )
    .context("mtp attn_step_launch")?;

    // attn_output[..hidden] is `fused_in + o_proj(...)`. Copy into the
    // dedicated mid-residual buffer so the FFN reads contiguous BF16
    // (attn_output is sized for the wider of full/linear stages, the
    // FFN expects exactly `hidden` elements).
    copy_d2d(
        ordinal,
        scratch.attn_residual.as_mut_ptr(),
        scratch.attn_output.as_ptr(),
        hidden * 2,
    )
    .context("mtp d2d attn_output -> attn_residual")?;

    // ---- FFN ----------------------------------------------------------
    reset_sync_buf(ordinal, &mut scratch.sync_buf)
        .context("reset sync_buf (mtp ffn)")?;
    let ffn_params = Qwen36MoeFfnStepParams {
        stage: 5,
        hidden: geom.hidden,
        num_experts: geom.num_experts,
        moe_intermediate: geom.moe_intermediate,
        shared_intermediate: geom.shared_intermediate,
        top_k: geom.top_k,
        rms_norm_eps: geom.rms_norm_eps,
    };
    let ffn_weights = Qwen36MoeFfnStepWeights {
        input_hidden: scratch.attn_residual.as_ptr(),
        post_attn_norm_w: mtp.post_attn_norm_w.as_ptr(),
        gate_w: mtp.gate_w.as_ptr(),
        gate_up_proj_w: mtp.gate_up_proj_w.as_ptr(),
        down_proj_w: mtp.down_proj_w.as_ptr(),
        shared_gate_proj_w: mtp.shared_gate_proj_w.as_ptr(),
        shared_up_proj_w: mtp.shared_up_proj_w.as_ptr(),
        shared_down_proj_w: mtp.shared_down_proj_w.as_ptr(),
        shared_expert_gate_w: mtp.shared_expert_gate_w.as_ptr(),
    };
    ffn_step_launch(
        ordinal,
        ScalarType::BF16,
        ffn_params,
        &ffn_weights,
        &Qwen36MoeFfnStepInt4::disabled(),
        &mut scratch.ffn_output,
        &mut scratch.ffn_output_idx,
        &mut scratch.ffn_workspace,
        &mut scratch.sync_buf,
    )
    .context("mtp ffn_step_launch")?;

    // ffn_output[..hidden] is `attn_residual + moe_out + shared_out` —
    // the layer's final residual output. Hand it back via `out`.
    copy_d2d(
        ordinal,
        out.as_mut_ptr(),
        scratch.ffn_output.as_ptr(),
        hidden * 2,
    )
    .context("mtp d2d ffn_output -> out")?;
    Ok(())
}
