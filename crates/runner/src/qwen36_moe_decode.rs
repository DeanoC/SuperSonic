//! Host-orchestrated multi-launch decode for Qwen3.6-MoE.
//!
//! Walks the hybrid pattern (every 4th layer full-attn at indices 3/7/11/...,
//! every other layer linear-attn) calling the per-block FFI launchers
//! ([`kernel_ffi::qwen36_moe::attn_step_launch`] / `linear_step_launch` /
//! `ffn_step_launch`) at stage 5 (full-layer output). One HIP launch per
//! block per layer per token. The persistent megakernel is the production
//! path, but this chained implementation remains the reviewable oracle path
//! for parity tests, fallback runs, and isolated stage diagnostics.
//!
//! The decode core in [`run_chained_decode`] takes pre-allocated
//! per-layer weight + state buffers and an initial hidden vector, runs the
//! chain, and returns the final hidden plus per-layer post-attn / post-FFN
//! intermediates (used by the parity test in
//! `crates/runner/tests/qwen36_moe_multilayer_parity.rs` to gate
//! correctness against the multi-layer Python oracle).
//!
//! Host-side final RMSnorm, lm_head helpers, INT4 dequant, and sampling live
//! in `qwen36_moe_logits`.
//!
//! Both the parity test (synthetic weights from oracle JSON) and the
//! engine's real-decode path (weights from the bake) call into the same
//! [`run_chained_decode`] core — the only difference is how the
//! [`LayerBuffers`] vec gets populated.
#![allow(dead_code)]

use std::ffi::c_void;
use std::ptr;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2h, memset_zeros, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    attn_step_launch, ffn_step_launch, linear_step_launch, Qwen36MoeAttnStepInt4,
    Qwen36MoeAttnStepParams, Qwen36MoeAttnStepWeights, Qwen36MoeFfnStepInt4,
    Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights, Qwen36MoeLinearStepInt4,
    Qwen36MoeLinearStepParams, Qwen36MoeLinearStepWeights,
};

use crate::qwen36_moe_logits::bf16_bytes_to_f32;

/// Hybrid pattern: every 4th layer is full attention. Indices 3, 7, 11, ...
/// are full; everything else is linear. Matches Qwen3.6-MoE 35B-A3B.
pub const HYBRID_FULL_ATTN_STRIDE: i32 = 4;

/// `true` when `layer_idx + 1` is a multiple of [`HYBRID_FULL_ATTN_STRIDE`].
pub fn is_full_attn_layer(layer_idx: i32) -> bool {
    (layer_idx + 1) % HYBRID_FULL_ATTN_STRIDE == 0
}

/// Geometry the chained decoder needs at every layer + the lm_head.
/// Mirrors the synthetic + production cases.
#[derive(Debug, Clone, Copy)]
pub struct MultiLayerGeom {
    pub hidden: i32,
    pub vocab: i32,
    pub num_layers: i32,
    pub rms_norm_eps: f32,

    // Full-attention (read iff a layer's `attn` is `Full`).
    pub num_attention_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub rope_theta: f32,

    // Linear-attention (read iff a layer's `attn` is `Linear`).
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,

    // MoE FFN (every layer).
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
}

/// Per-layer attention weight buffers. The two variants are mutually
/// exclusive: a layer is full xor linear. Selection happens at populate time
/// via [`is_full_attn_layer`]. When [`AttnLayerBuffers::Full::int4`] /
/// [`AttnLayerBuffers::Linear::int4`] is `Some`, the matching weight buffer
/// holds INT4 packed nibbles (`u8`, `[out, in/2]`) instead of BF16; the
/// sidecar carries `(scale, zero)` BF16 tiles `[out/gs, in/gs]` and the
/// active group_size.
pub enum AttnLayerBuffers {
    Full {
        input_norm_w: GpuBuffer,
        q_proj_w: GpuBuffer,
        k_proj_w: GpuBuffer,
        v_proj_w: GpuBuffer,
        q_norm_w: GpuBuffer,
        k_norm_w: GpuBuffer,
        o_proj_w: GpuBuffer,
        int4: Option<FullAttnInt4Sidecars>,
        /// PR 4d KV cache for this full-attention layer. When `Some`, the
        /// kernel writes the current step's K/V at slot `position` and
        /// attends over `kv_len = position + 1` past tokens. When `None`
        /// (parity tests, single-token decode), back-compat kv_len=1.
        kv_cache: Option<FullAttnKvCache>,
    },
    Linear {
        input_norm_w: GpuBuffer,
        in_proj_qkv_w: GpuBuffer,
        in_proj_z_w: GpuBuffer,
        in_proj_a_w: GpuBuffer,
        in_proj_b_w: GpuBuffer,
        conv1d_w: GpuBuffer,
        conv1d_bias: Option<GpuBuffer>,
        dt_bias: GpuBuffer,
        a_log: GpuBuffer,
        norm_w: GpuBuffer,
        out_proj_w: GpuBuffer,
        // Conv state ([qkv_dim, kernel-1] BF16) and recurrent state
        // ([V*K*Vd] F32), both mutated in place by the kernel.
        conv_state: GpuBuffer,
        recurrent_state: GpuBuffer,
        int4: Option<LinearAttnInt4Sidecars>,
    },
}

/// PR 4d KV cache for a full-attention layer. `[kv_max_t, num_kv_heads *
/// head_dim]` BF16 each (or U8 FP8 bytes when `kv_fp8` is on), mutated
/// by the kernel: at decode position `p` it writes the current step's
/// K/V at slot `p` then attends over `kv_len = p + 1` past tokens.
/// Lifetime tied to one decode session.
///
/// When KV-FP8 is active (`kv_scale_k.is_some()`):
///   - `k` and `v` are dtype `U8` with the same shape (or `None` when
///     VMM-backed and `virtual_kv_cache_k` is `Some`); the kernel
///     reinterprets the pointer in place to read/write FP8 E4M3 bytes.
///   - `kv_scale_k` / `kv_scale_v` are F32 `[num_kv_heads, kv_max_t]`
///     per-(head, position) absmax scales.
///   - `kv_shadow_k` / `kv_shadow_v` are an optional BF16 sidecar at
///     shape `[num_kv_heads, sidecar_window, head_dim]` for
///     parity-sensitive recent reads.
///   - `kv_shadow_start` is the first absolute position covered by the
///     sidecar (`-1` when the sidecar is disabled).
///   - `kv_shadow_window` is the sidecar's rolling capacity in positions
///     (`0` when disabled). The kernel computes the active start from the
///     current decode position, so descriptors do not need per-step updates.
///
/// Exactly one of `k` / `virtual_kv_cache_k` is `Some` (same for V).
/// `virtual_kv_cache_k` is `Some` when Qwen3.6 KV VMM was selected
/// (default on HIP when supported, or forced with `SUPERSONIC_VMM_KV=1`).
/// It can back either BF16 KV or FP8 KV.
pub struct FullAttnKvCache {
    /// `Some` when dense KV backing was selected. `None` when
    /// `virtual_kv_cache_k` is `Some` (VMM-backed). Exactly one of `k` /
    /// `virtual_kv_cache_k` is `Some`. Same for V.
    pub k: Option<GpuBuffer>,
    pub v: Option<GpuBuffer>,
    pub kv_max_t: i32,
    /// Some only when KV-FP8 is on.
    pub kv_scale_k: Option<GpuBuffer>,
    pub kv_scale_v: Option<GpuBuffer>,
    /// Some only when KV-FP8 is on AND the sidecar is enabled.
    pub kv_shadow_k: Option<GpuBuffer>,
    pub kv_shadow_v: Option<GpuBuffer>,
    /// First absolute KV position covered by the sidecar (`-1` when
    /// the sidecar is disabled).
    pub kv_shadow_start: i32,
    /// Rolling sidecar capacity in positions (`0` when disabled).
    pub kv_shadow_window: i32,
    /// `Some` when Qwen3.6 KV VMM was selected at allocation time AND the
    /// backend supports VMM. Mutually exclusive with `k` (exactly one is
    /// `Some`).
    pub virtual_kv_cache_k: Option<gpu_hal::VirtualBuffer>,
    pub virtual_kv_cache_v: Option<gpu_hal::VirtualBuffer>,
    /// The VMM reservation's logical max_T (matches `kv_max_t` above).
    /// Carried separately for symmetry with the qwen35 KV-VMM pattern.
    pub virtual_kv_max_t: Option<usize>,
}

impl FullAttnKvCache {
    /// Device pointer to the K cache, regardless of VMM vs dense backing.
    pub fn k_device_ptr(&mut self) -> *mut std::ffi::c_void {
        if let Some(vk) = self.virtual_kv_cache_k.as_mut() {
            vk.as_mut_ptr()
        } else if let Some(b) = self.k.as_mut() {
            b.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        }
    }

    /// Device pointer to the V cache, regardless of VMM vs dense backing.
    pub fn v_device_ptr(&mut self) -> *mut std::ffi::c_void {
        if let Some(vv) = self.virtual_kv_cache_v.as_mut() {
            vv.as_mut_ptr()
        } else if let Some(b) = self.v.as_mut() {
            b.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        }
    }
}

/// INT4 sidecars for a full-attention layer. Mirrors the per-block FFI
/// struct [`Qwen36MoeAttnStepInt4`]; only the four projection weights
/// (q/k/v/o) are quantizable — norms stay BF16. Group size is pinned to
/// 128 across the runtime + bake.
pub struct FullAttnInt4Sidecars {
    pub group_size: i32,
    pub q_proj_scale: GpuBuffer,
    pub q_proj_zero: GpuBuffer,
    pub k_proj_scale: GpuBuffer,
    pub k_proj_zero: GpuBuffer,
    pub v_proj_scale: GpuBuffer,
    pub v_proj_zero: GpuBuffer,
    pub o_proj_scale: GpuBuffer,
    pub o_proj_zero: GpuBuffer,
}

/// INT4 sidecars for a linear-attention layer. Mirrors
/// [`Qwen36MoeLinearStepInt4`]; only `in_proj_qkv`, `in_proj_z`, `out_proj`
/// are quantized — `in_proj_a/b`, conv1d, dt_bias, A_log, norms all stay BF16.
pub struct LinearAttnInt4Sidecars {
    pub group_size: i32,
    pub in_proj_qkv_scale: GpuBuffer,
    pub in_proj_qkv_zero: GpuBuffer,
    pub in_proj_z_scale: GpuBuffer,
    pub in_proj_z_zero: GpuBuffer,
    pub out_proj_scale: GpuBuffer,
    pub out_proj_zero: GpuBuffer,
}

/// INT4 sidecars for an MoE FFN block. Mirrors [`Qwen36MoeFfnStepInt4`];
/// the router (`gate_w`) and the scalar `shared_expert_gate` stay BF16.
pub struct FfnInt4Sidecars {
    pub group_size: i32,
    pub gate_up_proj_scale: GpuBuffer,
    pub gate_up_proj_zero: GpuBuffer,
    pub down_proj_scale: GpuBuffer,
    pub down_proj_zero: GpuBuffer,
    pub shared_gate_proj_scale: GpuBuffer,
    pub shared_gate_proj_zero: GpuBuffer,
    pub shared_up_proj_scale: GpuBuffer,
    pub shared_up_proj_zero: GpuBuffer,
    pub shared_down_proj_scale: GpuBuffer,
    pub shared_down_proj_zero: GpuBuffer,
}

/// GPU-resident weight pointer used by decode descriptors.
///
/// Most Qwen3.6-MoE weights are ordinary `GpuBuffer`s. Routed expert slabs can
/// also live in a `VirtualArena`; in that case the engine owns the arena and
/// this wrapper carries the stable virtual pointer plus shape metadata.
#[allow(dead_code)]
pub enum ResidentWeight {
    Dense(GpuBuffer),
    Virtual {
        allocation_id: usize,
        ptr: *const c_void,
        dtype: ScalarType,
        shape: Vec<usize>,
        len_bytes: usize,
    },
}

unsafe impl Send for ResidentWeight {}
unsafe impl Sync for ResidentWeight {}

#[allow(dead_code)]
impl ResidentWeight {
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            Self::Dense(buf) => buf.as_ptr(),
            Self::Virtual { ptr, .. } => *ptr,
        }
    }

    pub fn len_bytes(&self) -> usize {
        match self {
            Self::Dense(buf) => buf.len_bytes(),
            Self::Virtual { len_bytes, .. } => *len_bytes,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(buf) => buf.shape(),
            Self::Virtual { shape, .. } => shape.as_slice(),
        }
    }

    pub fn dtype(&self) -> ScalarType {
        match self {
            Self::Dense(buf) => buf.dtype(),
            Self::Virtual { dtype, .. } => *dtype,
        }
    }

    pub fn allocation_id(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::Virtual { allocation_id, .. } => Some(*allocation_id),
        }
    }
}

impl From<GpuBuffer> for ResidentWeight {
    fn from(value: GpuBuffer) -> Self {
        Self::Dense(value)
    }
}

/// Per-layer MoE FFN weight buffers. Always present (every layer has an
/// FFN block). When `int4` is `Some`, every `*_proj_w` field carries
/// packed nibbles instead of BF16 weights.
pub struct FfnLayerBuffers {
    pub post_attn_norm_w: GpuBuffer,
    pub gate_w: GpuBuffer,
    pub gate_up_proj_w: ResidentWeight,
    pub down_proj_w: ResidentWeight,
    pub shared_gate_proj_w: GpuBuffer,
    pub shared_up_proj_w: GpuBuffer,
    pub shared_down_proj_w: GpuBuffer,
    pub shared_expert_gate_w: GpuBuffer,
    pub int4: Option<FfnInt4Sidecars>,
}

/// One layer's worth of GPU-resident weight + state buffers.
pub struct LayerBuffers {
    pub attn: AttnLayerBuffers,
    pub ffn: FfnLayerBuffers,
}

/// Multi-token-prediction (MTP) head weights for Qwen3.6-MoE
/// self-speculative decode (Phase 6 of the perf roadmap). The MTP block
/// is structurally one full-attention MoE layer plus three small
/// "fusion" RMSNorms (`pre_fc_norm_hidden`, `pre_fc_norm_embedding`,
/// `norm`) and a fusion linear (`fc`), sharing `embed_tokens` and
/// `lm_head` with the base model. See `oracle/qwen36_moe_mtp_oracle.py`
/// for the full forward equation.
///
/// All 19 tensors are BF16 in the published FP8 release (no FP8
/// dequant or INT4 calibration this round — the MTP block is one
/// layer's worth of compute and BF16 vs INT4 is a wash). The total
/// weight footprint is ~1.6 GiB BF16; comfortably fits alongside the
/// 17 GiB INT4 base bake on a 24 GiB 7900 XTX.
pub struct MtpLayerBuffers {
    /// Fusion linears + norms (top-level `mtp.*` tensors).
    pub pre_fc_norm_hidden_w: GpuBuffer, // [hidden]
    pub pre_fc_norm_embedding_w: GpuBuffer, // [hidden]
    pub fc_w: GpuBuffer,                    // [hidden, 2*hidden]
    pub norm_w: GpuBuffer,                  // [hidden]

    /// Single-layer full-attn block (`mtp.layers.0.*`).
    pub input_norm_w: GpuBuffer, // [hidden]
    pub post_attn_norm_w: GpuBuffer, // [hidden]
    pub q_proj_w: GpuBuffer,         // [2*H*d, hidden]
    pub k_proj_w: GpuBuffer,         // [Hkv*d, hidden]
    pub v_proj_w: GpuBuffer,         // [Hkv*d, hidden]
    pub o_proj_w: GpuBuffer,         // [hidden, H*d]
    pub q_norm_w: GpuBuffer,         // [head_dim]
    pub k_norm_w: GpuBuffer,         // [head_dim]

    /// MoE FFN sub-block (`mtp.layers.0.mlp.*`).
    pub gate_w: GpuBuffer, // [num_experts, hidden]
    pub gate_up_proj_w: GpuBuffer,       // [num_experts, 2*I, hidden]
    pub down_proj_w: GpuBuffer,          // [num_experts, hidden, I]
    pub shared_gate_proj_w: GpuBuffer,   // [Is, hidden]
    pub shared_up_proj_w: GpuBuffer,     // [Is, hidden]
    pub shared_down_proj_w: GpuBuffer,   // [hidden, Is]
    pub shared_expert_gate_w: GpuBuffer, // [1, hidden]

    /// Per-step KV cache for the MTP layer's self-attention. Separate
    /// from the base layers' KV caches per the vLLM reference (each
    /// MTP draft step appends to this buffer at increasing positions;
    /// step k attends to K/V from steps 0..k-1).
    pub kv_cache: Option<FullAttnKvCache>,
}

impl LayerBuffers {
    pub fn is_full_attn(&self) -> bool {
        matches!(self.attn, AttnLayerBuffers::Full { .. })
    }
}

/// Captured intermediates from a chained decode pass. The per-layer hiddens
/// are useful for granular parity diagnostics; `final_hidden_bytes` is what
/// the host-side final RMSnorm + lm_head consumes.
pub struct DecodeOutputs {
    /// `[hidden]` BF16 little-endian — the residual after the last layer's
    /// FFN, before the final RMSnorm.
    pub final_hidden_bytes: Vec<u8>,
    /// `[num_layers][hidden]` BF16. `output_after_attn[i]` is layer `i`'s
    /// post-attention residual (input to that layer's FFN).
    pub per_layer_attn_out: Vec<Vec<u8>>,
    /// `[num_layers][hidden]` BF16. `output_after_ffn[i]` is layer `i`'s
    /// post-FFN residual (input to layer `i+1`).
    pub per_layer_ffn_out: Vec<Vec<u8>>,
    /// Wall-clock breakdown of the kernel launches inside this chain.
    /// `*_us` are sums-across-layers in microseconds. The launches are
    /// internally synchronous (`hipDeviceSynchronize` in the bridge), so
    /// the host wall-clock here measures real GPU + sync time.
    pub kernel_full_attn_us: u64,
    pub kernel_linear_attn_us: u64,
    pub kernel_ffn_us: u64,
}

/// Workspace floats sufficient for the full-attn parity launcher's stage 5
/// (the largest stage). Mirrors `parity_workspace_floats` in the per-block
/// test file: 6*H*d + 4*Hkv*d + hidden.
pub fn full_attn_workspace_floats(geom: &MultiLayerGeom) -> usize {
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let d = geom.head_dim as usize;
    6 * h * d + 4 * hkv * d + geom.hidden as usize
}

/// BF16 elements sufficient for the full-attn parity launcher's largest
/// stage (stage 3 publishes q_rot || k_rot, the widest output).
pub(crate) fn full_attn_output_elems(geom: &MultiLayerGeom) -> usize {
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let d = geom.head_dim as usize;
    h * d + hkv * d
}

/// Workspace floats for the linear-attn parity launcher's stage 5. Mirrors
/// the size used in the per-block test file. Only the linear-specific terms
/// matter — the larger of the two attn workspaces drives the shared scratch
/// allocation in `run_chained_decode`.
pub fn linear_attn_workspace_floats(geom: &MultiLayerGeom) -> usize {
    let k = geom.num_k_heads as usize;
    let v = geom.num_v_heads as usize;
    let kd = geom.head_k_dim as usize;
    let vd = geom.head_v_dim as usize;
    let key_dim = k * kd;
    let val_dim = v * vd;
    let qkv_dim = 2 * key_dim + val_dim;
    let v_kdim = v * kd;
    let v_vdim = v * vd;
    qkv_dim + val_dim + 2 * v + 2 * key_dim + 2 * v_kdim + v + v + v_vdim
}

/// BF16 elements for the linear-attn parity launcher's widest stage.
fn linear_attn_output_elems(geom: &MultiLayerGeom) -> usize {
    let v = geom.num_v_heads as usize;
    let kd = geom.head_k_dim as usize;
    let vd = geom.head_v_dim as usize;
    // Stage 5 publishes [hidden]; earlier stages publish wider intermediates.
    // The per-block linear test uses `2 * V*Kd + V*Vd` as the upper bound.
    2 * v * kd + v * vd
}

/// FFN parity launcher workspace floats — copied from the per-block test
/// file's `ffn_parity_workspace_floats`. See its docstring for the
/// per-stage layout (`OFF_H_NORM` in `kernels/qwen36_moe.hip`).
///
/// The per-expert scratch slabs (`EXPERT_GU`, `EXPERT_MID`) are sized
/// `k * 2*I` and `k * I` respectively so all `top_k` experts can run
/// G/H/I concurrently (one block-group per expert).
pub fn ffn_workspace_floats(geom: &MultiLayerGeom) -> usize {
    let hidden = geom.hidden as usize;
    let e = geom.num_experts as usize;
    let k = geom.top_k as usize;
    let is_dim = geom.shared_intermediate as usize;
    let i_dim = geom.moe_intermediate as usize;
    3 * hidden + 2 * e + 2 * k + 1 + 3 * is_dim + k * 3 * i_dim + k * hidden
}

/// FFN output BF16 elements — stage 5 publishes `[hidden]`, which is also
/// the upper bound (stages 2..=4 fit in a strict subset).
pub(crate) fn ffn_output_elems(geom: &MultiLayerGeom) -> usize {
    geom.hidden as usize
}

/// Reset the 96-byte cooperative-launch sync buffer between launches. The
/// kernels use it for atomic counters + the grid barrier; failure to reset
/// would cause the next launch's barrier to hang or skip work.
///
/// Layout (matches the four qwen36_moe FFI wrappers):
///   counters[0..16]      bytes  0..63  (16 work-stealing slots, used by the
///                                       FFN concurrent-experts dispatch)
///   barrier_counter      bytes 64..67
///   barrier_flag         bytes 68..71
///   (padding to 96 bytes for alignment headroom)
pub(crate) fn reset_sync_buf(ordinal: usize, sync_buf: &mut GpuBuffer) -> Result<(), GpuError> {
    memset_zeros(ordinal, sync_buf.as_mut_ptr(), 96)
}

/// Copy `[hidden]` BF16 elements out of a GPU buffer into a freshly
/// allocated host vec. Convenience wrapper that respects the buffer's full
/// size — kernels publish into the leading `hidden` elements but the
/// allocation may be wider (stage-3 q_rot||k_rot etc).
fn download_hidden_bf16(
    ordinal: usize,
    src: &GpuBuffer,
    hidden: usize,
) -> Result<Vec<u8>, GpuError> {
    let bytes = hidden * 2;
    let mut out = vec![0u8; bytes];
    copy_d2h(
        ordinal,
        out.as_mut_ptr() as *mut std::ffi::c_void,
        src.as_ptr(),
        bytes,
    )?;
    Ok(out)
}

/// Run one full decode step across `layers.len()` layers, returning the
/// per-layer hidden states + the final hidden (input to the lm_head).
///
/// Kernel state contract:
///  - The full-attn launcher's stage 5 publishes `output_hidden = input +
///    o_proj(...)` into the leading `hidden` elements of its output buffer.
///  - The linear-attn launcher's stage 5 publishes the same residual; it
///    also mutates the layer's `conv_state` / `recurrent_state` in place.
///  - The FFN launcher's stage 5 publishes `output_hidden = input + moe_out
///    + shared_out` into its output buffer's leading `hidden` elements.
///
/// Buffers reused across all layers within one step:
///  - `hidden_a` / `hidden_b`: BF16 residual ping-pong. The current input
///    lives in one, the just-published output in the other; we swap pointers
///    by indexing rather than `mem::swap` so the fixed `as_ptr()` references
///    we hand the kernel stay valid through the call.
///  - `attn_workspace`: F32 scratch sized for `max(full, linear)` workspaces.
///  - `attn_output`: BF16 scratch sized for the wider of the two attn
///    output footprints.
///  - `ffn_output`, `ffn_output_idx`, `ffn_workspace`: same idea per-FFN.
///  - `sync_buf`: a single 32-byte cooperative-launch counter, zero-reset
///    before each kernel call.
///
/// Per-layer state (lives in `LayerBuffers`, NOT shared):
///  - Linear-attn `conv_state` + `recurrent_state` — mutated in place.
///  - (Full-attn: no KV cache here. The single-block kernels treat each
///    call as `kv_len=1` self-attention; KV-cache extension is a PR 4d
///    follow-up.)
/// Knobs for `run_chained_decode`'s per-layer diagnostics.
///
/// Two costly options that the production decode loop doesn't need:
///
///   - `capture_per_layer`: when true, D2H-downloads each layer's
///     post-attn and post-ffn hidden into `DecodeOutputs.per_layer_*`.
///     The multilayer parity test consumes these; the engine doesn't.
///     Each download forces a full GPU sync, so on 35B-A3B (40 layers)
///     the unconditional path is ~80 syncs/token of pure overhead. Off
///     by default — turn on only for tests / parity diagnostics.
///   - `trace_norms`: when true, prints per-layer L2 norms for spotting
///     signal blow-up / collapse / NaN at production scale. Implies
///     `capture_per_layer` (we need the data to compute the norm).
///     Defaults from the `SUPERSONIC_QWEN36_DEBUG_TRACE_NORMS` env var
///     — see `ChainedDecodeOptions::from_env`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainedDecodeOptions {
    pub capture_per_layer: bool,
    pub trace_norms: bool,
    /// When true, call `gpu_hal::sync(ordinal)` after each step launch so
    /// the per-stage `kernel_*_us` accumulators reflect GPU execution time
    /// rather than host dispatch queue time. Set by `--emit-stage-timings`
    /// — production runs leave it false, paying ~80 syncs/token to keep
    /// the chain breakdown accurate when the user asks for it.
    /// (Codex review #80: PR #80 made the bridge step launchers async,
    /// which silently turned the stage-breakdown numbers into host-queue
    /// times — this flag preserves their original "GPU compute time"
    /// semantics for users opting into the breakdown.)
    pub accurate_stage_timings: bool,
}

impl ChainedDecodeOptions {
    /// Default flags + the legacy `SUPERSONIC_QWEN36_DEBUG_TRACE_NORMS`
    /// env var as the trace-norms enable. Production callers use this.
    pub fn from_env() -> Self {
        let trace_norms = std::env::var("SUPERSONIC_QWEN36_DEBUG_TRACE_NORMS")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        Self {
            capture_per_layer: false,
            trace_norms,
            accurate_stage_timings: false,
        }
    }
}

pub fn run_chained_decode(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
) -> Result<DecodeOutputs> {
    run_chained_decode_with_options(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        ChainedDecodeOptions {
            // Existing behaviour: parity tests rely on `per_layer_*` —
            // they call this entry point directly. The fast engine path
            // routes through `run_chained_decode_fast` (defined below).
            capture_per_layer: true,
            trace_norms: ChainedDecodeOptions::from_env().trace_norms,
            // The legacy capture path D2H-syncs every step anyway, so the
            // explicit per-step sync is redundant here (kept off).
            accurate_stage_timings: false,
        },
    )
}

/// Production decode entry point: skips the per-layer D2H downloads
/// (which force ~80 GPU syncs/token on 35B-A3B and which neither the
/// engine nor sampling consume — only the multilayer parity test
/// reads `per_layer_*`). Reuses `run_chained_decode_with_options`'s
/// implementation so parity guarantees are unchanged.
///
/// `accurate_stage_timings`: when true, synchronizes between step
/// launches so the per-stage `kernel_*_us` numbers in `DecodeOutputs`
/// reflect GPU compute time (not host queue time). Set this when
/// `--emit-stage-timings` is requested. Default false in production
/// to keep the chain async.
pub fn run_chained_decode_fast(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    accurate_stage_timings: bool,
) -> Result<DecodeOutputs> {
    let mut options = ChainedDecodeOptions::from_env();
    options.accurate_stage_timings = accurate_stage_timings;
    run_chained_decode_with_options(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        options,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPrefetchPhase {
    Lookahead,
    Demand,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertRoute {
    pub rank: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

pub type ExpertPrefetchCallback<'a> =
    dyn FnMut(ExpertPrefetchPhase, usize, &[ExpertRoute]) -> Result<()> + 'a;

/// Production chained decode with a host-side hook between router top-k and
/// routed expert GEMVs. The hook is intended for sparse VMM MoE residency:
/// stage 1 computes top-k indices, the callback pins those expert slices, and
/// the normal stage-5 FFN launch then consumes the same stable expert pointers.
pub fn run_chained_decode_fast_with_expert_prefetch(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    accurate_stage_timings: bool,
    expert_prefetch: &mut ExpertPrefetchCallback<'_>,
) -> Result<DecodeOutputs> {
    let mut options = ChainedDecodeOptions::from_env();
    options.accurate_stage_timings = accurate_stage_timings;
    run_chained_decode_impl(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        options,
        Some(expert_prefetch),
    )
}

pub fn run_chained_decode_with_options(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    options: ChainedDecodeOptions,
) -> Result<DecodeOutputs> {
    run_chained_decode_impl(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        options,
        None,
    )
}

fn run_chained_decode_impl(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    options: ChainedDecodeOptions,
    mut expert_prefetch: Option<&mut ExpertPrefetchCallback<'_>>,
) -> Result<DecodeOutputs> {
    let hidden = geom.hidden as usize;
    if initial_hidden_bytes.len() != hidden * 2 {
        return Err(anyhow!(
            "initial_hidden_bytes len {} != expected {} (hidden*2 BF16 bytes)",
            initial_hidden_bytes.len(),
            hidden * 2,
        ));
    }
    if layers.len() as i32 != geom.num_layers {
        return Err(anyhow!(
            "layers.len() {} != geom.num_layers {}",
            layers.len(),
            geom.num_layers,
        ));
    }

    // Residual ping-pong. Two buffers, each sized [hidden] BF16. We index
    // them by `front` so the buffer the kernel reads is well-defined for
    // every launch; alternating `front = 1 - front` after each launch puts
    // the just-written buffer into "input" position for the next call.
    let mut hidden_a =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[hidden], initial_hidden_bytes)
            .context("alloc hidden_a")?;
    let mut hidden_b =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).context("alloc hidden_b")?;

    // PR 4d: when any full-attn layer carries a KV cache, the kernel uses
    // an additional `[H, kv_max_t]` F32 region (OFF_SCORES) for per-head
    // attention scores. Size workspace for the largest kv_max_t any layer
    // declares.
    let max_kv_t = layers
        .iter()
        .filter_map(|l| match &l.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(c), ..
            } => Some(c.kv_max_t as usize),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let attn_extra = if max_kv_t > 0 {
        geom.num_attention_heads as usize * max_kv_t
    } else {
        0
    };

    // Shared attention scratch: sized for the larger of (full, linear).
    let attn_ws_floats =
        full_attn_workspace_floats(geom).max(linear_attn_workspace_floats(geom)) + attn_extra;
    let attn_out_elems = full_attn_output_elems(geom).max(linear_attn_output_elems(geom));
    let mut attn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[attn_out_elems])
        .context("alloc attn_output")?;
    let mut attn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[attn_ws_floats])
        .context("alloc attn_workspace")?;

    // Shared FFN scratch.
    let mut ffn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_output_elems(geom)])
        .context("alloc ffn_output")?;
    let mut ffn_output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
        .context("alloc ffn_output_idx")?;
    let mut ffn_workspace =
        GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_workspace_floats(geom)])
            .context("alloc ffn_workspace")?;

    let mut sync_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).context("alloc sync_buf")?;

    let mut per_layer_attn_out: Vec<Vec<u8>> = Vec::with_capacity(layers.len());
    let mut per_layer_ffn_out: Vec<Vec<u8>> = Vec::with_capacity(layers.len());

    // Per-kernel-class wall-clock accumulators. Reported back via
    // `DecodeOutputs.kernel_*_us`; the engine surfaces them under
    // `--emit-stage-timings` so we can see whether attn / linear / ffn
    // dominates the chain time.
    let mut t_full_attn = std::time::Duration::ZERO;
    let mut t_linear_attn = std::time::Duration::ZERO;
    let mut t_ffn = std::time::Duration::ZERO;

    // `capture` ⇔ "I need the per-layer hidden bytes on the host".
    // True if the caller asked for them OR if `trace_norms` is on (norm
    // computation reads the BF16 bytes). Otherwise we skip the D2H
    // copies entirely, which on 35B-A3B drops 80 GPU syncs/token.
    let trace_norms = options.trace_norms;
    let capture = options.capture_per_layer || trace_norms;
    if trace_norms {
        let init_norm = bf16_bytes_to_f32(initial_hidden_bytes)
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        eprintln!("[trace] step pos={position} init_hidden L2={init_norm:.4}");
    }

    // `front` indexes which of (hidden_a, hidden_b) holds the current
    // "input to next launch". Starts at 0 (initial_hidden was uploaded
    // into hidden_a). After each launch we swap.
    let mut front: usize = 0;

    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        // ---- Attention ----
        reset_sync_buf(ordinal, &mut sync_buf).context("reset sync_buf (attn)")?;
        // Capture the *const input pointer + *mut output pointer based on
        // current `front`. Borrowing both `hidden_a` and `hidden_b`
        // mutably at the same time isn't possible; pointer arithmetic is.
        let (input_ptr, output_buf): (_, &mut GpuBuffer) = if front == 0 {
            (hidden_a.as_ptr(), &mut hidden_b)
        } else {
            (hidden_b.as_ptr(), &mut hidden_a)
        };

        match &mut layer.attn {
            AttnLayerBuffers::Full {
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
                int4,
                kv_cache,
            } => {
                let params = Qwen36MoeAttnStepParams {
                    stage: 5,
                    hidden: geom.hidden,
                    num_heads: geom.num_attention_heads,
                    num_kv_heads: geom.num_kv_heads,
                    head_dim: geom.head_dim,
                    rotary_dim: geom.rotary_dim,
                    rope_theta: geom.rope_theta,
                    rms_norm_eps: geom.rms_norm_eps,
                    position,
                    cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
                };
                let (kv_k_ptr, kv_v_ptr, kv_max_t) = match kv_cache {
                    Some(c) => (c.k_device_ptr(), c.v_device_ptr(), c.kv_max_t),
                    None => (ptr::null_mut(), ptr::null_mut(), 0),
                };
                let weights = Qwen36MoeAttnStepWeights {
                    input_hidden: input_ptr,
                    input_norm_w: input_norm_w.as_ptr(),
                    q_proj_w: q_proj_w.as_ptr(),
                    k_proj_w: k_proj_w.as_ptr(),
                    v_proj_w: v_proj_w.as_ptr(),
                    q_norm_w: q_norm_w.as_ptr(),
                    k_norm_w: k_norm_w.as_ptr(),
                    o_proj_w: o_proj_w.as_ptr(),
                    kv_cache_k: kv_k_ptr,
                    kv_cache_v: kv_v_ptr,
                    kv_max_t,
                };
                let int4_ptrs = match int4 {
                    Some(s) => {
                        let fp8 = s.group_size < 0;
                        Qwen36MoeAttnStepInt4 {
                            group_size: s.group_size,
                            q_proj_scale: s.q_proj_scale.as_ptr(),
                            q_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.q_proj_zero.as_ptr()
                            },
                            k_proj_scale: s.k_proj_scale.as_ptr(),
                            k_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.k_proj_zero.as_ptr()
                            },
                            v_proj_scale: s.v_proj_scale.as_ptr(),
                            v_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.v_proj_zero.as_ptr()
                            },
                            o_proj_scale: s.o_proj_scale.as_ptr(),
                            o_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.o_proj_zero.as_ptr()
                            },
                        }
                    }
                    None => Qwen36MoeAttnStepInt4::disabled(),
                };
                let t_k = std::time::Instant::now();
                attn_step_launch(
                    ordinal,
                    ScalarType::BF16,
                    params,
                    &weights,
                    &int4_ptrs,
                    &mut attn_output,
                    &mut attn_workspace,
                    &mut sync_buf,
                )
                .with_context(|| format!("attn_step_launch (layer {layer_idx}, full)"))?;
                if options.accurate_stage_timings {
                    gpu_hal::sync(ordinal)
                        .context("sync_after_attn_step (accurate_stage_timings)")?;
                }
                t_full_attn += t_k.elapsed();
            }
            AttnLayerBuffers::Linear {
                input_norm_w,
                in_proj_qkv_w,
                in_proj_z_w,
                in_proj_a_w,
                in_proj_b_w,
                conv1d_w,
                conv1d_bias,
                dt_bias,
                a_log,
                norm_w,
                out_proj_w,
                conv_state,
                recurrent_state,
                int4,
            } => {
                let params = Qwen36MoeLinearStepParams {
                    stage: 5,
                    hidden: geom.hidden,
                    num_k_heads: geom.num_k_heads,
                    num_v_heads: geom.num_v_heads,
                    head_k_dim: geom.head_k_dim,
                    head_v_dim: geom.head_v_dim,
                    conv_kernel_dim: geom.conv_kernel_dim,
                    rms_norm_eps: geom.rms_norm_eps,
                };
                let weights = Qwen36MoeLinearStepWeights {
                    input_hidden: input_ptr,
                    input_norm_w: input_norm_w.as_ptr(),
                    in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
                    in_proj_z_w: in_proj_z_w.as_ptr(),
                    in_proj_a_w: in_proj_a_w.as_ptr(),
                    in_proj_b_w: in_proj_b_w.as_ptr(),
                    conv1d_w: conv1d_w.as_ptr(),
                    conv1d_bias: conv1d_bias
                        .as_ref()
                        .map(|b| b.as_ptr())
                        .unwrap_or(ptr::null()),
                    dt_bias: dt_bias.as_ptr(),
                    a_log: a_log.as_ptr(),
                    norm_w: norm_w.as_ptr(),
                    out_proj_w: out_proj_w.as_ptr(),
                    conv_state: conv_state.as_mut_ptr(),
                    recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
                };
                let int4_ptrs = match int4 {
                    Some(s) => {
                        let fp8 = s.group_size < 0;
                        Qwen36MoeLinearStepInt4 {
                            group_size: s.group_size,
                            in_proj_qkv_scale: s.in_proj_qkv_scale.as_ptr(),
                            in_proj_qkv_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.in_proj_qkv_zero.as_ptr()
                            },
                            in_proj_z_scale: s.in_proj_z_scale.as_ptr(),
                            in_proj_z_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.in_proj_z_zero.as_ptr()
                            },
                            out_proj_scale: s.out_proj_scale.as_ptr(),
                            out_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.out_proj_zero.as_ptr()
                            },
                        }
                    }
                    None => Qwen36MoeLinearStepInt4::disabled(),
                };
                let t_k = std::time::Instant::now();
                linear_step_launch(
                    ordinal,
                    ScalarType::BF16,
                    params,
                    &weights,
                    &int4_ptrs,
                    &mut attn_output,
                    &mut attn_workspace,
                    &mut sync_buf,
                )
                .with_context(|| format!("linear_step_launch (layer {layer_idx})"))?;
                if options.accurate_stage_timings {
                    gpu_hal::sync(ordinal)
                        .context("sync_after_linear_step (accurate_stage_timings)")?;
                }
                t_linear_attn += t_k.elapsed();
            }
        }

        // attn_output[..hidden] now holds output_after_attn. Copy it into
        // the front buffer so the FFN reads it as input. We use a D2D
        // copy through the kernel for simplicity rather than juggling a
        // third hidden buffer.
        gpu_hal::copy_d2d(
            ordinal,
            output_buf.as_mut_ptr(),
            attn_output.as_ptr(),
            hidden * 2,
        )
        .context("d2d attn_output -> residual")?;

        // Swap front: the just-published value is now the "current input".
        front = 1 - front;

        if capture {
            let attn_out_bytes = download_hidden_bf16(ordinal, output_buf, hidden)
                .context("download per-layer attn output")?;
            if trace_norms {
                let v = bf16_bytes_to_f32(&attn_out_bytes);
                let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nan = v.iter().any(|x| !x.is_finite());
                let kind = if matches!(layer.attn, AttnLayerBuffers::Full { .. }) {
                    "full"
                } else {
                    "lin "
                };
                eprintln!(
                    "[trace]   layer {layer_idx:2} {kind} attn  L2={l2:.4}{}",
                    if nan { " NaN!" } else { "" }
                );
            }
            per_layer_attn_out.push(attn_out_bytes);
        }

        // ---- FFN ----
        reset_sync_buf(ordinal, &mut sync_buf).context("reset sync_buf (ffn)")?;
        let (input_ptr, output_buf): (_, &mut GpuBuffer) = if front == 0 {
            (hidden_a.as_ptr(), &mut hidden_b)
        } else {
            (hidden_b.as_ptr(), &mut hidden_a)
        };

        let ffn = &layer.ffn;
        let params_stage5 = Qwen36MoeFfnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let ffn_weights = Qwen36MoeFfnStepWeights {
            input_hidden: input_ptr,
            post_attn_norm_w: ffn.post_attn_norm_w.as_ptr(),
            gate_w: ffn.gate_w.as_ptr(),
            gate_up_proj_w: ffn.gate_up_proj_w.as_ptr(),
            down_proj_w: ffn.down_proj_w.as_ptr(),
            shared_gate_proj_w: ffn.shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: ffn.shared_up_proj_w.as_ptr(),
            shared_down_proj_w: ffn.shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: ffn.shared_expert_gate_w.as_ptr(),
        };
        let ffn_int4_ptrs = match &ffn.int4 {
            Some(s) => {
                let fp8 = s.group_size < 0;
                Qwen36MoeFfnStepInt4 {
                    group_size: s.group_size,
                    gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                    gate_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.gate_up_proj_zero.as_ptr()
                    },
                    down_proj_scale: s.down_proj_scale.as_ptr(),
                    down_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.down_proj_zero.as_ptr()
                    },
                    shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                    shared_gate_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_gate_proj_zero.as_ptr()
                    },
                    shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                    shared_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_up_proj_zero.as_ptr()
                    },
                    shared_down_proj_scale: s.shared_down_proj_scale.as_ptr(),
                    shared_down_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_down_proj_zero.as_ptr()
                    },
                }
            }
            None => Qwen36MoeFfnStepInt4::disabled(),
        };
        if let Some(prefetch) = expert_prefetch.as_mut() {
            prefetch(ExpertPrefetchPhase::Lookahead, layer_idx, &[]).with_context(|| {
                format!("lookahead prefetch routed experts (layer {layer_idx})")
            })?;
            let params_stage1 = Qwen36MoeFfnStepParams {
                stage: 1,
                ..params_stage5
            };
            let t_router = std::time::Instant::now();
            ffn_step_launch(
                ordinal,
                ScalarType::BF16,
                params_stage1,
                &ffn_weights,
                &ffn_int4_ptrs,
                &mut ffn_output,
                &mut ffn_output_idx,
                &mut ffn_workspace,
                &mut sync_buf,
            )
            .with_context(|| format!("ffn_step_launch router prefetch (layer {layer_idx})"))?;
            if options.accurate_stage_timings {
                gpu_hal::sync(ordinal)
                    .context("sync_after_ffn_router_prefetch (accurate_stage_timings)")?;
            }
            t_ffn += t_router.elapsed();

            let routes = download_topk_routes(&ffn_output_idx, &ffn_output, geom.top_k as usize)
                .with_context(|| format!("download FFN top-k routes (layer {layer_idx})"))?;
            prefetch(ExpertPrefetchPhase::Demand, layer_idx, &routes)
                .with_context(|| format!("prefetch routed experts (layer {layer_idx})"))?;
            reset_sync_buf(ordinal, &mut sync_buf)
                .context("reset sync_buf (ffn after prefetch)")?;
        }
        let t_k = std::time::Instant::now();
        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params_stage5,
            &ffn_weights,
            &ffn_int4_ptrs,
            &mut ffn_output,
            &mut ffn_output_idx,
            &mut ffn_workspace,
            &mut sync_buf,
        )
        .with_context(|| format!("ffn_step_launch (layer {layer_idx})"))?;
        if options.accurate_stage_timings {
            gpu_hal::sync(ordinal).context("sync_after_ffn_step (accurate_stage_timings)")?;
        }
        t_ffn += t_k.elapsed();

        // Same D2D + swap as the attn step.
        gpu_hal::copy_d2d(
            ordinal,
            output_buf.as_mut_ptr(),
            ffn_output.as_ptr(),
            hidden * 2,
        )
        .context("d2d ffn_output -> residual")?;
        front = 1 - front;

        if capture {
            let ffn_out_bytes = download_hidden_bf16(ordinal, output_buf, hidden)
                .context("download per-layer ffn output")?;
            if trace_norms {
                let v = bf16_bytes_to_f32(&ffn_out_bytes);
                let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nan = v.iter().any(|x| !x.is_finite());
                eprintln!(
                    "[trace]   layer {layer_idx:2}      ffn   L2={l2:.4}{}",
                    if nan { " NaN!" } else { "" }
                );
            }
            per_layer_ffn_out.push(ffn_out_bytes);
        }
    }

    let final_buf = if front == 0 { &hidden_a } else { &hidden_b };
    let final_hidden_bytes =
        download_hidden_bf16(ordinal, final_buf, hidden).context("download final hidden")?;

    Ok(DecodeOutputs {
        final_hidden_bytes,
        per_layer_attn_out,
        per_layer_ffn_out,
        kernel_full_attn_us: t_full_attn.as_micros() as u64,
        kernel_linear_attn_us: t_linear_attn.as_micros() as u64,
        kernel_ffn_us: t_ffn.as_micros() as u64,
    })
}

fn download_topk_indices(buf: &GpuBuffer, top_k: usize) -> Result<Vec<usize>> {
    let bytes = buf.to_host_bytes().context("d2h top-k indices")?;
    let needed = top_k * std::mem::size_of::<u32>();
    if bytes.len() < needed {
        return Err(anyhow!(
            "top-k index buffer has {} bytes, need {needed}",
            bytes.len()
        ));
    }
    Ok(bytes[..needed]
        .chunks_exact(4)
        .map(|chunk| {
            let raw = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            raw as usize
        })
        .collect())
}

fn download_topk_routes(
    idx_buf: &GpuBuffer,
    weight_buf: &GpuBuffer,
    top_k: usize,
) -> Result<Vec<ExpertRoute>> {
    let idx = download_topk_indices(idx_buf, top_k)?;
    let needed = top_k * std::mem::size_of::<u16>();
    let mut weight_bytes = vec![0u8; needed];
    copy_d2h(
        weight_buf.device_ordinal(),
        weight_bytes.as_mut_ptr() as *mut _,
        weight_buf.as_ptr(),
        needed,
    )
    .context("d2h top-k route weights")?;
    let weights = bf16_bytes_to_f32(&weight_bytes);
    Ok(idx
        .into_iter()
        .zip(weights)
        .enumerate()
        .map(|(rank, (expert_idx, weight))| ExpertRoute {
            rank,
            expert_idx,
            weight,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_pattern_marks_every_fourth_layer_full() {
        for li in 0..40 {
            let expect_full = (li + 1) % 4 == 0;
            assert_eq!(is_full_attn_layer(li), expect_full, "layer {li}");
        }
    }
}
