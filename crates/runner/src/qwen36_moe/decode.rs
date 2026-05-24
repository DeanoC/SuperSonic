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

use std::ptr;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2h, memset_zeros, Backend, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    attn_step_launch, attn_step_stage5_metal_host_into,
    ffn_expert_direct_gather_defer_wait_enabled,
    ffn_expert_direct_gather_stage5_metal_native_supported, ffn_stage5_router_defer_wait_enabled,
    ffn_stage5_router_metal_native_supported, ffn_step_launch, linear_step_launch,
    linear_step_stage5_metal_native_into, Qwen36MoeAttnStepInt4, Qwen36MoeAttnStepParams,
    Qwen36MoeAttnStepWeights, Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams,
    Qwen36MoeFfnStepWeights, Qwen36MoeLinearStepInt4, Qwen36MoeLinearStepParams,
    Qwen36MoeLinearStepWeights,
};

use crate::qwen36_moe_logits::bf16_bytes_to_f32;
use crate::qwen36_moe_types::{
    AttnLayerBuffers, DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom,
};

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

fn metal_linear_decode_direct_enabled(int4: &Qwen36MoeLinearStepInt4) -> bool {
    int4.group_size == 128
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT").is_none()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5").is_none()
}

fn metal_full_attn_decode_direct_enabled(int4: &Qwen36MoeAttnStepInt4) -> bool {
    int4.group_size == 128
        && std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_DECODE_DIRECT").is_some()
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_FULL_ATTN_DECODE_DIRECT").is_none()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
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

fn qwen36_metal_decode_batch_enabled(
    capture: bool,
    accurate_stage_timings: bool,
    has_expert_prefetch: bool,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH").is_some()
        && !capture
        && !accurate_stage_timings
        && !has_expert_prefetch
        && std::env::var_os("SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES").is_none()
        && std::env::var_os("SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP").is_none()
}

fn qwen36_metal_decode_batch_profile_phases_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES").is_some()
}

fn flush_active_metal_decode_batch(label: &str) -> Result<bool> {
    if !kernel_ffi::prefill_ffi::metal_batch_is_active() {
        return Ok(false);
    }
    kernel_ffi::prefill_ffi::set_metal_batch_label("qwen36_decode_batch")
        .map_err(|e| anyhow!("{label} Metal batch label: {e}"))?;
    kernel_ffi::prefill_ffi::flush_metal_batch()
        .map_err(|e| anyhow!("{label} Metal batch flush: {e}"))?;
    Ok(true)
}

fn flush_metal_decode_batch_profile_phase(label: &str, profile_label: &str) -> Result<()> {
    if !qwen36_metal_decode_batch_profile_phases_enabled()
        || !kernel_ffi::prefill_ffi::metal_batch_is_active()
    {
        return Ok(());
    }
    kernel_ffi::prefill_ffi::set_metal_batch_label(profile_label)
        .map_err(|e| anyhow!("{label} Metal batch label: {e}"))?;
    kernel_ffi::prefill_ffi::flush_metal_batch()
        .map_err(|e| anyhow!("{label} Metal batch flush: {e}"))?;
    Ok(())
}

fn sync_metal_queue_for_host_boundary(backend: Backend, label: &str) -> Result<()> {
    if backend == Backend::Metal {
        if flush_active_metal_decode_batch(label)? {
            return Ok(());
        }
        kernel_ffi::prefill_ffi::sync_metal_queue()
            .map_err(|e| anyhow!("{label} Metal queue sync: {e}"))?;
    }
    Ok(())
}

fn sync_metal_queue_for_host_read(buffer: &GpuBuffer, label: &str) -> Result<()> {
    sync_metal_queue_for_host_boundary(buffer.backend(), label)
}

fn copy_d2d_decode(
    ordinal: usize,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
) -> Result<(), GpuError> {
    if kernel_ffi::prefill_ffi::metal_batch_is_active() {
        kernel_ffi::prefill_ffi::metal_copy_d2d(src, dst, bytes)
    } else {
        gpu_hal::copy_d2d(ordinal, dst, src, bytes)
    }
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
    expert_prefetch: Option<&mut ExpertPrefetchCallback<'_>>,
) -> Result<DecodeOutputs> {
    run_chained_decode_impl_with_cache_pos(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        options,
        expert_prefetch,
    )
}

/// Sparse-prefill entry point: like `run_chained_decode_fast` but with an
/// explicit `cache_pos` (KV-cache slot index) decoupled from `position`
/// (RoPE rotation). Used when the prompt has been pruned by SpecPrefill —
/// kept tokens land in compact KV slots (`cache_pos = compacted_idx`) but
/// rotate to their original prompt positions (`position = original_pos`).
/// Pass `Qwen36MoeAttnStepParams::CACHE_POS_INHERIT` for `cache_pos` to
/// reproduce the dense `run_chained_decode_fast` behavior bit-equally.
pub fn run_chained_decode_fast_with_cache_pos(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    cache_pos: i32,
    accurate_stage_timings: bool,
) -> Result<DecodeOutputs> {
    let mut options = ChainedDecodeOptions::from_env();
    options.accurate_stage_timings = accurate_stage_timings;
    run_chained_decode_impl_with_cache_pos(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        cache_pos,
        options,
        None,
    )
}

/// Sparse-prefill + MoE-residency variant. Combines
/// `run_chained_decode_fast_with_cache_pos` with the host-side expert
/// prefetch hook from `run_chained_decode_fast_with_expert_prefetch`.
pub fn run_chained_decode_fast_with_expert_prefetch_and_cache_pos(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    cache_pos: i32,
    accurate_stage_timings: bool,
    expert_prefetch: &mut ExpertPrefetchCallback<'_>,
) -> Result<DecodeOutputs> {
    let mut options = ChainedDecodeOptions::from_env();
    options.accurate_stage_timings = accurate_stage_timings;
    run_chained_decode_impl_with_cache_pos(
        ordinal,
        geom,
        layers,
        initial_hidden_bytes,
        position,
        cache_pos,
        options,
        Some(expert_prefetch),
    )
}

fn run_chained_decode_impl_with_cache_pos(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layers: &mut [LayerBuffers],
    initial_hidden_bytes: &[u8],
    position: i32,
    cache_pos: i32,
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

    let metal_decode_batch = if hidden_a.backend() == Backend::Metal
        && qwen36_metal_decode_batch_enabled(
            capture,
            options.accurate_stage_timings,
            expert_prefetch.is_some(),
        ) {
        let guard = kernel_ffi::prefill_ffi::MetalBatchGuard::begin()
            .map_err(|e| anyhow!("qwen36 Metal decode batch begin: {e}"))?;
        kernel_ffi::prefill_ffi::set_metal_batch_label("qwen36_decode_batch")
            .map_err(|e| anyhow!("qwen36 Metal decode batch label: {e}"))?;
        Some(guard)
    } else {
        None
    };
    let metal_decode_batch_active = metal_decode_batch.is_some();

    // `front` indexes which of (hidden_a, hidden_b) holds the current
    // "input to next launch". Starts at 0 (initial_hidden was uploaded
    // into hidden_a). After each launch we swap.
    let mut front: usize = 0;
    let mut metal_queue_dirty = false;

    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        // ---- Attention ----
        if metal_queue_dirty && matches!(layer.attn, AttnLayerBuffers::Full { .. }) {
            sync_metal_queue_for_host_boundary(
                if front == 0 {
                    hidden_a.backend()
                } else {
                    hidden_b.backend()
                },
                "sync before full-attn host read",
            )?;
            metal_queue_dirty = false;
        }
        // Capture the *const input pointer + *mut output pointer based on
        // current `front`. Borrowing both `hidden_a` and `hidden_b`
        // mutably at the same time isn't possible; pointer arithmetic is.
        let input_backend = if front == 0 {
            hidden_a.backend()
        } else {
            hidden_b.backend()
        };
        let (input_ptr, output_buf): (_, &mut GpuBuffer) = if front == 0 {
            (hidden_a.as_ptr(), &mut hidden_b)
        } else {
            (hidden_b.as_ptr(), &mut hidden_a)
        };
        reset_sync_buf(ordinal, &mut sync_buf).context("reset sync_buf (attn)")?;
        let mut attn_output_published = false;
        let defer_layer_ffn_router = {
            let ffn = &layer.ffn;
            let params = Qwen36MoeFfnStepParams {
                stage: 5,
                layer_idx: layer_idx as i32,
                hidden: geom.hidden,
                num_experts: geom.num_experts,
                moe_intermediate: geom.moe_intermediate,
                shared_intermediate: geom.shared_intermediate,
                top_k: geom.top_k,
                rms_norm_eps: geom.rms_norm_eps,
            };
            let weights = Qwen36MoeFfnStepWeights {
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
            let int4 = match &ffn.int4 {
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
            output_buf.backend() == Backend::Metal
                && !capture
                && expert_prefetch.is_none()
                && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT")
                    .is_none()
                && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5").is_none()
                && ffn_stage5_router_defer_wait_enabled()
                && ffn_stage5_router_metal_native_supported(params, &weights, &int4)
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
                    cache_pos,
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
                let use_metal_direct = output_buf.backend() == Backend::Metal
                    && metal_full_attn_decode_direct_enabled(&int4_ptrs);
                if metal_decode_batch_active {
                    sync_metal_queue_for_host_boundary(
                        input_backend,
                        "sync before full-attn host read",
                    )?;
                }
                if use_metal_direct {
                    unsafe {
                        attn_step_stage5_metal_host_into(
                            params,
                            &weights,
                            &int4_ptrs,
                            &mut attn_output,
                            &mut attn_workspace,
                            output_buf.as_mut_ptr(),
                            hidden,
                        )
                    }
                    .with_context(|| {
                        format!("attn_step_stage5_metal_host_into (layer {layer_idx})")
                    })?;
                    attn_output_published = true;
                } else {
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
                }
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
                let use_metal_direct = output_buf.backend() == Backend::Metal
                    && metal_linear_decode_direct_enabled(&int4_ptrs);
                if (metal_queue_dirty || metal_decode_batch_active) && !use_metal_direct {
                    sync_metal_queue_for_host_boundary(
                        input_backend,
                        "sync before linear-attn host fallback",
                    )?;
                    metal_queue_dirty = false;
                }
                if use_metal_direct {
                    let wait_for_completion = !defer_layer_ffn_router;
                    unsafe {
                        linear_step_stage5_metal_native_into(
                            params,
                            &weights,
                            &int4_ptrs,
                            &mut attn_output,
                            &mut attn_workspace,
                            output_buf.as_mut_ptr(),
                            hidden,
                            wait_for_completion,
                        )
                    }
                    .with_context(|| {
                        format!("linear_step_stage5_metal_native_into (layer {layer_idx})")
                    })?;
                    attn_output_published = true;
                } else {
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
                }
                if options.accurate_stage_timings {
                    gpu_hal::sync(ordinal)
                        .context("sync_after_linear_step (accurate_stage_timings)")?;
                }
                if use_metal_direct {
                    flush_metal_decode_batch_profile_phase(
                        "profile flush after linear-attn",
                        "qwen36_decode_batch_linear_attn",
                    )?;
                }
                t_linear_attn += t_k.elapsed();
            }
        }

        if !attn_output_published {
            // attn_output[..hidden] now holds output_after_attn. Copy it into
            // the front buffer so the FFN reads it as input. Linear Metal INT4
            // can publish directly to this buffer and skips this D2D copy.
            copy_d2d_decode(
                ordinal,
                output_buf.as_mut_ptr(),
                attn_output.as_ptr(),
                hidden * 2,
            )
            .context("d2d attn_output -> residual")?;
        }

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
            layer_idx: layer_idx as i32,
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
        let defer_layer_ffn_direct_gather = output_buf.backend() == Backend::Metal
            && !capture
            && expert_prefetch.is_none()
            && ffn_expert_direct_gather_defer_wait_enabled()
            && ffn_expert_direct_gather_stage5_metal_native_supported(
                params_stage5,
                &ffn_weights,
                &ffn_int4_ptrs,
            );
        let ffn_uses_router_native = output_buf.backend() == Backend::Metal
            && ffn_stage5_router_metal_native_supported(
                params_stage5,
                &ffn_weights,
                &ffn_int4_ptrs,
            );
        if (metal_queue_dirty || metal_decode_batch_active) && !ffn_uses_router_native {
            sync_metal_queue_for_host_boundary(input_backend, "sync before ffn host router")?;
        }
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
        let ffn_stage5_output = output_buf;
        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params_stage5,
            &ffn_weights,
            &ffn_int4_ptrs,
            ffn_stage5_output,
            &mut ffn_output_idx,
            &mut ffn_workspace,
            &mut sync_buf,
        )
        .with_context(|| format!("ffn_step_launch (layer {layer_idx})"))?;
        metal_queue_dirty = defer_layer_ffn_router || defer_layer_ffn_direct_gather;
        if options.accurate_stage_timings {
            gpu_hal::sync(ordinal).context("sync_after_ffn_step (accurate_stage_timings)")?;
        }
        if ffn_uses_router_native {
            flush_metal_decode_batch_profile_phase(
                "profile flush after ffn",
                "qwen36_decode_batch_ffn",
            )?;
        }
        t_ffn += t_k.elapsed();

        front = 1 - front;

        if capture {
            sync_metal_queue_for_host_read(ffn_stage5_output, "download per-layer ffn output")?;
            let ffn_out_bytes = download_hidden_bf16(ordinal, ffn_stage5_output, hidden)
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
    sync_metal_queue_for_host_read(final_buf, "download final hidden")?;
    let final_hidden_bytes =
        download_hidden_bf16(ordinal, final_buf, hidden).context("download final hidden")?;
    if let Some(batch) = metal_decode_batch {
        batch
            .finish()
            .map_err(|e| anyhow!("qwen36 Metal decode batch finish: {e}"))?;
    }

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
