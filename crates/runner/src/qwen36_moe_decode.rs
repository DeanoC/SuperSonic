//! Host-orchestrated multi-launch decode for Qwen3.6-MoE — PR 4c step 2.
//!
//! Walks the hybrid pattern (every 4th layer full-attn at indices 3/7/11/...,
//! every other layer linear-attn) calling the per-block FFI launchers
//! ([`kernel_ffi::qwen36_moe::attn_step_launch`] / `linear_step_launch` /
//! `ffn_step_launch`) at stage 5 (full-layer output). One HIP launch per
//! block per layer per token — high launch overhead, but correct,
//! reviewable, and unblocks end-to-end testing. The persistent megakernel
//! (PR 4c step 4) folds these N×3 launches down to 1.
//!
//! The decode core in [`run_chained_decode`] takes pre-allocated
//! per-layer weight + state buffers and an initial hidden vector, runs the
//! chain, and returns the final hidden plus per-layer post-attn / post-FFN
//! intermediates (used by the parity test in
//! `crates/runner/tests/qwen36_moe_multilayer_parity.rs` to gate
//! correctness against the multi-layer Python oracle).
//!
//! [`host_final_norm_lm_head`] applies the final RMSnorm + lm_head GEMV on
//! the host. The plan recommends host-side for PR 4c (~1ms slower per token
//! vs a kernel; lifting it to GPU is PR 4d).
//!
//! Both the parity test (synthetic weights from oracle JSON) and the
//! engine's real-decode path (weights from the bake) call into the same
//! [`run_chained_decode`] core — the only difference is how the
//! [`LayerBuffers`] vec gets populated.

use std::ptr;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2h, memset_zeros, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    attn_step_launch, ffn_step_launch, linear_step_launch, Qwen36MoeAttnStepInt4,
    Qwen36MoeAttnStepParams, Qwen36MoeAttnStepWeights, Qwen36MoeFfnStepInt4,
    Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights, Qwen36MoeLinearStepInt4,
    Qwen36MoeLinearStepParams, Qwen36MoeLinearStepWeights,
};

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
/// head_dim]` BF16 each, mutated by the kernel: at decode position `p` it
/// writes the current step's K/V at slot `p` then attends over
/// `kv_len = p + 1` past tokens. Lifetime tied to one decode session.
pub struct FullAttnKvCache {
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub kv_max_t: i32,
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

/// Per-layer MoE FFN weight buffers. Always present (every layer has an
/// FFN block). When `int4` is `Some`, every `*_proj_w` field carries
/// packed nibbles instead of BF16 weights.
pub struct FfnLayerBuffers {
    pub post_attn_norm_w: GpuBuffer,
    pub gate_w: GpuBuffer,
    pub gate_up_proj_w: GpuBuffer,
    pub down_proj_w: GpuBuffer,
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
fn full_attn_workspace_floats(geom: &MultiLayerGeom) -> usize {
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let d = geom.head_dim as usize;
    6 * h * d + 4 * hkv * d + geom.hidden as usize
}

/// BF16 elements sufficient for the full-attn parity launcher's largest
/// stage (stage 3 publishes q_rot || k_rot, the widest output).
fn full_attn_output_elems(geom: &MultiLayerGeom) -> usize {
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let d = geom.head_dim as usize;
    h * d + hkv * d
}

/// Workspace floats for the linear-attn parity launcher's stage 5. Mirrors
/// the size used in the per-block test file. Only the linear-specific terms
/// matter — the larger of the two attn workspaces drives the shared scratch
/// allocation in `run_chained_decode`.
fn linear_attn_workspace_floats(geom: &MultiLayerGeom) -> usize {
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
fn ffn_workspace_floats(geom: &MultiLayerGeom) -> usize {
    let hidden = geom.hidden as usize;
    let e = geom.num_experts as usize;
    let k = geom.top_k as usize;
    let is_dim = geom.shared_intermediate as usize;
    let i_dim = geom.moe_intermediate as usize;
    3 * hidden + 2 * e + 2 * k + 1 + 3 * is_dim + 3 * i_dim + k * hidden
}

/// FFN output BF16 elements — stage 5 publishes `[hidden]`, which is also
/// the upper bound (stages 2..=4 fit in a strict subset).
fn ffn_output_elems(geom: &MultiLayerGeom) -> usize {
    geom.hidden as usize
}

/// Reset the 32-byte cooperative-launch sync buffer between launches. The
/// kernels use it for atomic counters + the grid barrier; failure to reset
/// would cause the next launch's barrier to hang or skip work.
fn reset_sync_buf(ordinal: usize, sync_buf: &mut GpuBuffer) -> Result<(), GpuError> {
    memset_zeros(ordinal, sync_buf.as_mut_ptr(), 32)
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
        ordinal, geom, layers, initial_hidden_bytes, position,
        ChainedDecodeOptions {
            // Existing behaviour: parity tests rely on `per_layer_*` —
            // they call this entry point directly. The fast engine path
            // routes through `run_chained_decode_fast` (defined below).
            capture_per_layer: true,
            trace_norms: ChainedDecodeOptions::from_env().trace_norms,
        },
    )
}

/// Production decode entry point: skips the per-layer D2H downloads
/// (which force ~80 GPU syncs/token on 35B-A3B and which neither the
/// engine nor sampling consume — only the multilayer parity test
/// reads `per_layer_*`). Reuses `run_chained_decode_with_options`'s
/// implementation so parity guarantees are unchanged.
pub fn run_chained_decode_fast(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
) -> Result<DecodeOutputs> {
    run_chained_decode_with_options(
        ordinal, geom, layers, initial_hidden_bytes, position,
        ChainedDecodeOptions::from_env(),
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
    let mut hidden_a = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[hidden],
        initial_hidden_bytes,
    )
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
    let attn_ws_floats = full_attn_workspace_floats(geom).max(linear_attn_workspace_floats(geom)) + attn_extra;
    let attn_out_elems = full_attn_output_elems(geom).max(linear_attn_output_elems(geom));
    let mut attn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[attn_out_elems])
        .context("alloc attn_output")?;
    let mut attn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[attn_ws_floats])
        .context("alloc attn_workspace")?;

    // Shared FFN scratch.
    let mut ffn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_output_elems(geom)])
        .context("alloc ffn_output")?;
    let mut ffn_output_idx =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
            .context("alloc ffn_output_idx")?;
    let mut ffn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_workspace_floats(geom)])
        .context("alloc ffn_workspace")?;

    let mut sync_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U8, &[32]).context("alloc sync_buf")?;

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
            .iter().map(|x| x * x).sum::<f32>().sqrt();
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
                };
                let (kv_k_ptr, kv_v_ptr, kv_max_t) = match kv_cache {
                    Some(c) => (c.k.as_mut_ptr(), c.v.as_mut_ptr(), c.kv_max_t),
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
                    Some(s) => Qwen36MoeAttnStepInt4 {
                        group_size: s.group_size,
                        q_proj_scale: s.q_proj_scale.as_ptr(),
                        q_proj_zero: s.q_proj_zero.as_ptr(),
                        k_proj_scale: s.k_proj_scale.as_ptr(),
                        k_proj_zero: s.k_proj_zero.as_ptr(),
                        v_proj_scale: s.v_proj_scale.as_ptr(),
                        v_proj_zero: s.v_proj_zero.as_ptr(),
                        o_proj_scale: s.o_proj_scale.as_ptr(),
                        o_proj_zero: s.o_proj_zero.as_ptr(),
                    },
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
                    Some(s) => Qwen36MoeLinearStepInt4 {
                        group_size: s.group_size,
                        in_proj_qkv_scale: s.in_proj_qkv_scale.as_ptr(),
                        in_proj_qkv_zero: s.in_proj_qkv_zero.as_ptr(),
                        in_proj_z_scale: s.in_proj_z_scale.as_ptr(),
                        in_proj_z_zero: s.in_proj_z_zero.as_ptr(),
                        out_proj_scale: s.out_proj_scale.as_ptr(),
                        out_proj_zero: s.out_proj_zero.as_ptr(),
                    },
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
                let kind = if matches!(layer.attn, AttnLayerBuffers::Full { .. }) { "full" } else { "lin " };
                eprintln!("[trace]   layer {layer_idx:2} {kind} attn  L2={l2:.4}{}",
                    if nan { " NaN!" } else { "" });
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
        let params = Qwen36MoeFfnStepParams {
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
            Some(s) => Qwen36MoeFfnStepInt4 {
                group_size: s.group_size,
                gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                gate_up_proj_zero: s.gate_up_proj_zero.as_ptr(),
                down_proj_scale: s.down_proj_scale.as_ptr(),
                down_proj_zero: s.down_proj_zero.as_ptr(),
                shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                shared_gate_proj_zero: s.shared_gate_proj_zero.as_ptr(),
                shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                shared_up_proj_zero: s.shared_up_proj_zero.as_ptr(),
                shared_down_proj_scale: s.shared_down_proj_scale.as_ptr(),
                shared_down_proj_zero: s.shared_down_proj_zero.as_ptr(),
            },
            None => Qwen36MoeFfnStepInt4::disabled(),
        };
        let t_k = std::time::Instant::now();
        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &ffn_weights,
            &ffn_int4_ptrs,
            &mut ffn_output,
            &mut ffn_output_idx,
            &mut ffn_workspace,
            &mut sync_buf,
        )
        .with_context(|| format!("ffn_step_launch (layer {layer_idx})"))?;
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
                eprintln!("[trace]   layer {layer_idx:2}      ffn   L2={l2:.4}{}",
                    if nan { " NaN!" } else { "" });
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

// ---------------------------------------------------------------------------
// Host-side final RMSnorm + lm_head GEMV. The plan recommends host execution
// for PR 4c; lifting to a dedicated kernel is PR 4d.
// ---------------------------------------------------------------------------

/// Decode a stream of BF16 little-endian bytes into F32. The oracle stores
/// BF16 as raw int16 → bytes; this is the inverse.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "BF16 bytes must be even");
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
            f32::from_bits(bits << 16)
        })
        .collect()
}

/// Round an F32 to BF16 (RNE), returning the 16 raw bits. Matches PyTorch's
/// `.to(torch.bfloat16)` rounding: nearest-even on the lopped-off mantissa
/// bits, with the standard NaN-quieting convention.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN: keep top 16 with quiet bit set.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFFu32 + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

/// Encode a slice of F32 values to BF16 little-endian bytes (RNE).
pub fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = f32_to_bf16_bits(v);
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

/// Apply RMSnorm + lm_head GEMV on the host. Mirrors the multi-layer
/// oracle's tail:
///   final_normed = rms_norm(hidden, final_norm_w, eps)   # (1+w) offset
///   logits       = final_normed.to(F32) @ lm_head_w.to(F32).T
///   logits       = logits.to(BF16)
///
/// The RMSnorm uses the HuggingFace `Qwen3_5MoeRMSNorm` convention with
/// the `(1.0 + weight)` unit offset — `model.norm` in the HF text model
/// (line 1354 of `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`)
/// is an instance of `Qwen3_5MoeRMSNorm` whose forward computes
/// `output * (1.0 + self.weight.float())`.
///
/// All inputs are BF16 little-endian byte streams; output is BF16
/// little-endian bytes for `vocab` logit channels.
pub fn host_final_norm_lm_head(
    hidden_bytes: &[u8],
    final_norm_w_bytes: &[u8],
    lm_head_w_bytes: &[u8],
    hidden: usize,
    vocab: usize,
    eps: f32,
) -> Vec<u8> {
    assert_eq!(hidden_bytes.len(), hidden * 2, "hidden bytes mismatch");
    assert_eq!(final_norm_w_bytes.len(), hidden * 2, "norm_w bytes mismatch");
    assert_eq!(
        lm_head_w_bytes.len(),
        vocab * hidden * 2,
        "lm_head bytes mismatch"
    );

    // BF16-input convenience wrapper. For multi-token decode loops use
    // `host_final_norm_lm_head_f32` directly with cached F32 lm_head + norm
    // weight to avoid re-converting the (multi-GiB) lm_head matrix per step.
    let w_f32 = bf16_bytes_to_f32(final_norm_w_bytes);
    let lm_f32 = bf16_bytes_to_f32(lm_head_w_bytes);
    host_final_norm_lm_head_f32(hidden_bytes, &w_f32, &lm_f32, hidden, vocab, eps)
}

/// Same math as [`host_final_norm_lm_head`] but with `final_norm_w` and
/// `lm_head_w` already converted to F32 by the caller. Hot-loop variant —
/// the BF16→F32 conversion of the lm_head matrix is by far the dominant
/// cost on the 35B-A3B geometry (~1 GiB BF16 → ~2 GiB F32 per call), and
/// it never changes across tokens, so the engine converts it once and
/// passes the F32 view here per-step.
pub fn host_final_norm_lm_head_f32(
    hidden_bytes: &[u8],
    final_norm_w_f32: &[f32],
    lm_head_w_f32: &[f32],
    hidden: usize,
    vocab: usize,
    eps: f32,
) -> Vec<u8> {
    assert_eq!(hidden_bytes.len(), hidden * 2, "hidden bytes mismatch");
    assert_eq!(final_norm_w_f32.len(), hidden, "norm_w f32 len mismatch");
    assert_eq!(lm_head_w_f32.len(), vocab * hidden, "lm_head f32 len mismatch");

    let h_f32 = bf16_bytes_to_f32(hidden_bytes);

    // RMSnorm with the HF `Qwen3_5MoeRMSNorm` convention: F32 mean of
    // squares -> rsqrt(var+eps) -> elementwise mul by `(1.0 + w)`. The
    // stored weight is a small delta around zero (initialized to zeros
    // in HF — line 810 of modeling_qwen3_5_moe.py) and the effective
    // scale factor is `1 + w`.
    let mean_sq: f32 =
        h_f32.iter().map(|&x| x * x).sum::<f32>() / hidden as f32;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();
    let normed: Vec<f32> = h_f32
        .iter()
        .zip(final_norm_w_f32.iter())
        .map(|(&x, &w)| x * rsqrt * (1.0 + w))
        .collect();

    // GEMV: lm_head [vocab, hidden] @ normed [hidden] -> logits [vocab].
    // F64 accumulator matches the oracle's torch.float32 reduction order
    // closely enough for cos_sim parity at vocab+hidden ≤ a few thousand.
    let mut logits = vec![0f32; vocab];
    for v in 0..vocab {
        let row_start = v * hidden;
        let mut acc = 0f64;
        for h in 0..hidden {
            acc += lm_head_w_f32[row_start + h] as f64 * normed[h] as f64;
        }
        logits[v] = acc as f32;
    }

    f32_to_bf16_bytes(&logits)
}

/// Dequantize an INT4-packed weight slab back to BF16 bytes. Mirrors the
/// kernel's `int4_dequant_scalar`: for each output row + input column the
/// nibble is split off the byte (even col → low nibble, odd col → high),
/// converted to F32, then `bf16(q*s - z*s)` written using the matching
/// `[out/gs, in/gs]` BF16 scale/zero tile.
///
/// Used by the host-side lm_head path when the bake quantizes lm_head.
/// `out_dim` × `in_dim` BF16 result = ~1 GiB for 35B-A3B (vocab 248K ×
/// hidden 2048); fine for one-token smoke, kernel-side lm_head is PR 4d.
pub fn dequant_int4_to_bf16_bytes(
    packed: &[u8],
    scale_bf16: &[u8],
    zero_bf16: &[u8],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<u8> {
    assert_eq!(packed.len(), out_dim * in_dim / 2, "packed size mismatch");
    assert_eq!(in_dim % group_size, 0, "in_dim must be divisible by group_size");
    assert_eq!(out_dim % group_size, 0, "out_dim must be divisible by group_size");
    let n_row_tiles = out_dim / group_size;
    let n_col_tiles = in_dim / group_size;
    assert_eq!(scale_bf16.len(), n_row_tiles * n_col_tiles * 2, "scale size mismatch");
    assert_eq!(zero_bf16.len(), n_row_tiles * n_col_tiles * 2, "zero size mismatch");

    let scale = bf16_bytes_to_f32(scale_bf16);
    let zero = bf16_bytes_to_f32(zero_bf16);
    let mut out = Vec::with_capacity(out_dim * in_dim * 2);
    let half_in = in_dim / 2;
    for o in 0..out_dim {
        let row_tile = o / group_size;
        let row_base = o * half_in;
        for i in 0..in_dim {
            let col_tile = i / group_size;
            let tile_idx = row_tile * n_col_tiles + col_tile;
            let s = scale[tile_idx];
            let z = zero[tile_idx];
            let byte = packed[row_base + (i / 2)];
            let nib = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let q = nib as f32;
            // Single-rounding bf16(q*s - z*s) — matches the kernel's
            // `bf16_round_rne_f32_finite(n*s - zs)`.
            let bf = f32_to_bf16_bits(q * s - z * s);
            out.push((bf & 0xFF) as u8);
            out.push((bf >> 8) as u8);
        }
    }
    out
}

/// Tiny dependency-free xorshift64 RNG. Deterministic given the seed —
/// the engine's `--sampling-seed` plus identical prompt + model + sampling
/// params produces bit-identical generation.
pub struct XorshiftRng(u64);

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        // Xorshift requires a non-zero state.
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform `f32` in `[0, 1)`. 24 random bits → IEEE single mantissa.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// Sample one token from BF16 logits with optional temperature, top-K,
/// and top-P (nucleus) filters. `temperature <= 0` falls through to
/// greedy argmax (the deterministic, reproducible default — same as
/// [`argmax_bf16_logits`]). `top_k == 0` means "no top-K cap" (full
/// vocab); `top_p >= 1.0` means "no nucleus truncation".
///
/// Greedy is bit-identical to argmax_bf16_logits (same iteration order,
/// no temperature scaling).
pub fn sample_bf16_logits(
    logits_bytes: &[u8],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut XorshiftRng,
) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        return argmax_bf16_logits(logits_bytes);
    }
    let logits = bf16_bytes_to_f32(logits_bytes);
    let inv_t = 1.0 / temperature;

    // Top-K: pick the K highest-logit indices, descending. For top_k==0
    // sort the entire vocab (slow but only paid when sampling — vocab≈248K
    // takes ~few ms per token, negligible vs the chained decode).
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v * inv_t)).collect();
    let k = if top_k == 0 || top_k > indexed.len() {
        indexed.len()
    } else {
        // Partial sort: select_nth_unstable + sort the head saves the tail.
        let _ = indexed.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        top_k
    };
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Softmax with max-stabilisation over the kept indices.
    let max_logit = indexed[0].1;
    let mut exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return indexed[0].0 as u32;
    }
    for e in &mut exps {
        *e /= sum;
    }

    // Top-P nucleus: smallest prefix whose cumulative prob ≥ top_p.
    let nucleus_size = if top_p >= 1.0 {
        exps.len()
    } else {
        let mut cum = 0.0f32;
        let mut n = exps.len();
        for (i, &p) in exps.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                n = i + 1;
                break;
            }
        }
        n.max(1)
    };

    // Sample from the (renormalised) nucleus.
    let nucleus_sum: f32 = exps[..nucleus_size].iter().sum();
    let r: f32 = rng.next_f32() * nucleus_sum;
    let mut cum = 0.0f32;
    for i in 0..nucleus_size {
        cum += exps[i];
        if cum >= r {
            return indexed[i].0 as u32;
        }
    }
    indexed[0].0 as u32
}

/// Greedy argmax over a BF16 logits buffer. Returns the highest-scoring
/// token id. Used by the runner's first-token smoke path.
pub fn argmax_bf16_logits(logits_bytes: &[u8]) -> u32 {
    let logits = bf16_bytes_to_f32(logits_bytes);
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i, v)
            } else {
                (best_i, best_v)
            }
        })
        .0 as u32
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

    #[test]
    fn bf16_roundtrip_preserves_normal_values() {
        // PyTorch's BF16 rounding round-trips most "round" F32 values
        // bit-exactly. Sample a few that have all-zero low mantissa bits
        // so rounding is a no-op.
        let bf16_clean = [0.0f32, 1.0, 2.0, 0.5, -3.25, 1024.0];
        for &v in &bf16_clean {
            let bytes = f32_to_bf16_bytes(&[v]);
            let roundtrip = bf16_bytes_to_f32(&bytes);
            assert_eq!(roundtrip[0], v, "bf16 roundtrip drift at {v}");
        }
    }

    #[test]
    fn rms_norm_then_lm_head_matches_naive_computation() {
        // Tiny hand-checked case. RMS-normalised (1, 2, 3) has mean_sq=14/3
        // → rsqrt=sqrt(3/14). With norm_w=(1,1,1) and the HF `(1+w)` unit
        // offset, the effective scale per channel is 2. Times lm_head row
        // [1, 0, 0] yields `1 * sqrt(3/14) * 2` ≈ 0.9258. BF16 rounding
        // is loose enough that we check within 1e-2.
        let hidden = 3usize;
        let vocab = 1usize;
        let h_bytes = f32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let w_bytes = f32_to_bf16_bytes(&[1.0, 1.0, 1.0]);
        let lm_bytes = f32_to_bf16_bytes(&[1.0, 0.0, 0.0]);
        let logits = host_final_norm_lm_head(&h_bytes, &w_bytes, &lm_bytes, hidden, vocab, 0.0);
        let logit = bf16_bytes_to_f32(&logits)[0];
        let expected = 2.0 * (3.0f32 / 14.0).sqrt();
        assert!(
            (logit - expected).abs() < 1e-2,
            "logit {logit} far from expected {expected}"
        );
    }

    #[test]
    fn rms_norm_unit_offset_zero_weight_is_identity_scale() {
        // With norm_w = zeros, the HF `(1+w)` convention degrades to
        // a plain RMSnorm (effective scale = 1). Locks the offset in
        // semantics — same input as the test above with weight=1 should
        // give exactly half the logit when weight=0.
        let hidden = 3usize;
        let vocab = 1usize;
        let h_bytes = f32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let w_bytes = f32_to_bf16_bytes(&[0.0, 0.0, 0.0]);
        let lm_bytes = f32_to_bf16_bytes(&[1.0, 0.0, 0.0]);
        let logits = host_final_norm_lm_head(&h_bytes, &w_bytes, &lm_bytes, hidden, vocab, 0.0);
        let logit = bf16_bytes_to_f32(&logits)[0];
        let expected = (3.0f32 / 14.0).sqrt();
        assert!(
            (logit - expected).abs() < 1e-2,
            "logit {logit} far from expected {expected}"
        );
    }
}
