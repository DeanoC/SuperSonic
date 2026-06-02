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
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2h, memset_zeros, Backend, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    attn_step_launch, attn_step_stage5_metal_host_into, attn_step_stage5_metal_native_into,
    attn_step_stage5_metal_native_supported, emit_decode_batch_routed_stage5_parity_tap_from_host,
    emit_decode_batch_shared_stage5_parity_tap_from_host,
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
    AttnLayerBuffers, DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, FullAttnKvCache,
    LayerBuffers, MultiLayerGeom,
};

static QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_DECODE_BATCH_SHARED_PARITY_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_DECODE_BATCH_ROUTED_PARITY_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_LAYER_OUTPUT_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_LAYER_OUTPUT_DELTA_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_DECODE_BATCH_FFN_COMMIT_CALLS: AtomicUsize = AtomicUsize::new(0);

const LAYER_OUTPUT_TAP_PHASE_COUNT: usize = 2;
const LAYER_OUTPUT_TAP_PHASES: [&str; 2] = ["attn", "ffn"];

struct DecodeBatchSharedParitySnapshots {
    input: GpuBuffer,
    workspace: GpuBuffer,
    output: GpuBuffer,
    output_idx: GpuBuffer,
    captured: Vec<bool>,
    workspace_floats: usize,
}

struct LayerOutputSnapshots {
    output: GpuBuffer,
    captured: Vec<[bool; LAYER_OUTPUT_TAP_PHASE_COUNT]>,
}

struct DecodeRotaryRow {
    cos: GpuBuffer,
    sin: GpuBuffer,
}

impl DecodeRotaryRow {
    fn build(ordinal: usize, position: i32, rotary_dim: usize, theta: f32) -> Result<Self> {
        let half = rotary_dim / 2;
        let mut cos_data = Vec::with_capacity(half * std::mem::size_of::<u16>());
        let mut sin_data = Vec::with_capacity(half * std::mem::size_of::<u16>());
        let theta_log = theta.ln();
        for i in 0..half {
            let exponent = (i as f32 / half as f32) * theta_log;
            let freq = position as f32 * (-exponent).exp();
            let c = half::bf16::from_f32(freq.cos());
            let s = half::bf16::from_f32(freq.sin());
            cos_data.extend_from_slice(&c.to_le_bytes());
            sin_data.extend_from_slice(&s.to_le_bytes());
        }
        let cos = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[half], &cos_data)
            .context("alloc qwen36 decode rope cos row")?;
        let sin = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[half], &sin_data)
            .context("alloc qwen36 decode rope sin row")?;
        Ok(Self { cos, sin })
    }
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

fn metal_linear_decode_direct_enabled(int4: &Qwen36MoeLinearStepInt4) -> bool {
    int4.group_size == 128
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT").is_none()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5").is_none()
}

fn metal_full_attn_decode_direct_enabled(int4: &Qwen36MoeAttnStepInt4) -> bool {
    int4.group_size == 128
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
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DISABLE_DECODE_BATCH").is_none()
        && !capture
        && !accurate_stage_timings
        && !has_expert_prefetch
        && (std::env::var_os("SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES").is_none()
            || qwen36_metal_decode_batch_ffn_profile_phases_enabled())
        && std::env::var_os("SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP").is_none()
}

fn qwen36_metal_decode_batch_profile_phases_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES").is_some()
}

fn qwen36_metal_decode_batch_profile_phases_deferred_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED").is_some()
}

fn qwen36_metal_decode_batch_deferred_commits_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SYNC_PHASES").is_none()
}

fn qwen36_metal_decode_batch_ffn_commit_interval() -> usize {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
}

fn qwen36_metal_decode_batch_ffn_profile_phases_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES").is_some()
}

fn qwen36_metal_decode_batch_route_snapshot_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT").is_some()
}

fn qwen36_metal_decode_batch_shared_stage5_parity_tap_enabled() -> bool {
    (std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP").is_some()
        || std::env::var_os("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP").is_some())
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
}

fn qwen36_metal_decode_batch_routed_stage5_parity_tap_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP").is_some()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
}

fn qwen36_metal_decode_batch_shared_stage5_parity_tap_max_calls() -> usize {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_MAX_CALLS")
        .or_else(|_| {
            std::env::var("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS")
        })
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(40)
}

fn qwen36_metal_decode_batch_shared_stage5_parity_tap_position() -> Option<i32> {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_POSITION")
        .or_else(|_| std::env::var("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_POSITION"))
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

fn qwen36_metal_decode_batch_shared_stage5_parity_tap_layer() -> Option<usize> {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_LAYER")
        .or_else(|_| std::env::var("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_LAYER"))
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn qwen36_metal_decode_batch_shared_stage5_parity_tap_matches(
    position: i32,
    layer_idx: usize,
) -> bool {
    if let Some(wanted) = qwen36_metal_decode_batch_shared_stage5_parity_tap_position() {
        if position != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_metal_decode_batch_shared_stage5_parity_tap_layer() {
        if layer_idx != wanted {
            return false;
        }
    }
    true
}

fn qwen36_metal_decode_batch_routed_stage5_parity_tap_max_calls() -> usize {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_MAX_CALLS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(40)
}

fn qwen36_metal_decode_batch_routed_stage5_parity_tap_position() -> Option<i32> {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_POSITION")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

fn qwen36_metal_decode_batch_routed_stage5_parity_tap_layer() -> Option<usize> {
    std::env::var("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_LAYER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn qwen36_metal_decode_batch_routed_stage5_parity_tap_matches(
    position: i32,
    layer_idx: usize,
) -> bool {
    if let Some(wanted) = qwen36_metal_decode_batch_routed_stage5_parity_tap_position() {
        if position != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_metal_decode_batch_routed_stage5_parity_tap_layer() {
        if layer_idx != wanted {
            return false;
        }
    }
    true
}

fn qwen36_layer_output_tap_enabled() -> bool {
    std::env::var_os("SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP").is_some()
        || qwen36_layer_output_delta_tap_enabled()
}

fn qwen36_layer_output_tap_should_snapshot(
    position: i32,
    layer_idx: usize,
    phase_idx: usize,
) -> bool {
    if std::env::var_os("SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP").is_some() {
        return true;
    }
    qwen36_layer_output_delta_tap_matches(position, layer_idx, phase_idx)
}

fn qwen36_layer_output_delta_tap_enabled() -> bool {
    std::env::var_os("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP").is_some()
}

fn qwen36_layer_output_delta_tap_position() -> Option<i32> {
    std::env::var("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

fn qwen36_layer_output_delta_tap_layer() -> Option<usize> {
    std::env::var("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn qwen36_layer_output_delta_tap_phase() -> Option<usize> {
    let raw = std::env::var("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE").ok()?;
    LAYER_OUTPUT_TAP_PHASES
        .iter()
        .position(|phase| phase.eq_ignore_ascii_case(raw.trim()))
}

fn qwen36_layer_output_delta_tap_matches(
    position: i32,
    layer_idx: usize,
    phase_idx: usize,
) -> bool {
    if !qwen36_layer_output_delta_tap_enabled() {
        return false;
    }
    if let Some(wanted) = qwen36_layer_output_delta_tap_position() {
        if position != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_layer_output_delta_tap_layer() {
        if layer_idx != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_layer_output_delta_tap_phase() {
        if phase_idx != wanted {
            return false;
        }
    }
    true
}

fn qwen36_metal_router_stage5_simd_env_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD").is_some()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
}

fn qwen36_metal_router_stage5_exact_simd_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_STAGE5_EXACT_SIMD").is_none()
        && !qwen36_metal_router_stage5_simd_env_enabled()
        && std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_none()
}

fn qwen36_full_attn_native_layer_limit() -> Option<usize> {
    std::env::var("SUPERSONIC_METAL_QWEN36_FULL_ATTN_NATIVE_MAX_LAYER")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn qwen36_full_attn_native_enabled_for_layer(layer_idx: usize) -> bool {
    qwen36_full_attn_native_layer_limit().map_or(true, |max_layer| layer_idx <= max_layer)
}

fn qwen36_metal_decode_batch_router_path_label() -> &'static str {
    if qwen36_metal_router_stage5_simd_env_enabled() {
        "simd"
    } else if qwen36_metal_router_stage5_exact_simd_enabled() {
        "exact-simd"
    } else {
        "serial"
    }
}

fn qwen36_metal_decode_batch_phase_profile_enabled() -> bool {
    qwen36_metal_decode_batch_ffn_profile_phases_enabled()
        || qwen36_metal_decode_batch_profile_phases_enabled()
        || qwen36_metal_decode_batch_profile_phases_deferred_enabled()
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
    if !kernel_ffi::prefill_ffi::metal_batch_is_active() {
        return Ok(());
    }
    if qwen36_metal_decode_batch_deferred_commits_enabled()
        || qwen36_metal_decode_batch_profile_phases_deferred_enabled()
    {
        let ffn_commit_interval = qwen36_metal_decode_batch_ffn_commit_interval();
        if profile_label == "qwen36_decode_batch_ffn" && ffn_commit_interval > 1 {
            let call_idx = QWEN36_DECODE_BATCH_FFN_COMMIT_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if call_idx % ffn_commit_interval != 0 {
                return Ok(());
            }
        }
        kernel_ffi::prefill_ffi::commit_metal_batch_current(profile_label)
            .map_err(|e| anyhow!("{label} Metal batch deferred commit: {e}"))?;
        return Ok(());
    }
    if !qwen36_metal_decode_batch_profile_phases_enabled() {
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

fn sync_decode_queue(ordinal: usize, backend: Backend, label: &str) -> Result<()> {
    if backend == Backend::Metal {
        sync_metal_queue_for_host_boundary(backend, label)
    } else {
        gpu_hal::sync(ordinal).with_context(|| format!("{label} (accurate_stage_timings)"))
    }
}

fn copy_d2d_decode(
    ordinal: usize,
    backend: Backend,
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
) -> Result<(), GpuError> {
    if backend == Backend::Metal {
        kernel_ffi::prefill_ffi::metal_copy_d2d(src, dst, bytes)
    } else {
        gpu_hal::copy_d2d(ordinal, dst, src, bytes)
    }
}

fn decode_batch_route_snapshot_checksum(
    routes: &[u32],
    captured_layers: &[bool],
    top_k: usize,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (layer, &captured) in captured_layers.iter().enumerate() {
        if !captured {
            continue;
        }
        hash ^= layer as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        for rank in 0..top_k {
            let expert = routes[layer * top_k + rank] as u64;
            hash ^= ((rank as u64) << 32) ^ expert;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    hash
}

fn format_decode_batch_route_snapshot(
    routes: &[u32],
    captured_layers: &[bool],
    top_k: usize,
) -> String {
    captured_layers
        .iter()
        .enumerate()
        .map(|(layer, &captured)| {
            if !captured {
                return "-".to_string();
            }
            let start = layer * top_k;
            routes[start..start + top_k]
                .iter()
                .map(|expert| expert.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn decode_batch_route_snapshot_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn emit_decode_batch_route_snapshot(
    position: i32,
    cache_pos: i32,
    top_k: usize,
    captured_layers: &[bool],
    route_snapshot: &GpuBuffer,
) -> Result<()> {
    let bytes = route_snapshot
        .to_host_bytes()
        .context("d2h qwen36 decode-batch route snapshot")?;
    let routes = decode_batch_route_snapshot_u32(&bytes);
    let captured_count = captured_layers.iter().filter(|&&captured| captured).count();
    let entries = captured_count * top_k;
    let first_layer = captured_layers
        .iter()
        .position(|&captured| captured)
        .map(|layer| layer as i32)
        .unwrap_or(-1);
    let last_layer = captured_layers
        .iter()
        .rposition(|&captured| captured)
        .map(|layer| layer as i32)
        .unwrap_or(-1);
    let checksum = decode_batch_route_snapshot_checksum(&routes, captured_layers, top_k);
    let route_path = qwen36_metal_decode_batch_router_path_label();
    let phase_profile = qwen36_metal_decode_batch_phase_profile_enabled();
    let call = QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT_CALLS.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "[qwen36-decode-batch-route-snapshot] call={} position={} cache_pos={} router_path={} phase_profile={} layers={} top_k={} captured_layers={} entries={} first_layer={} last_layer={} checksum={} routes={}",
        call,
        position,
        cache_pos,
        route_path,
        phase_profile as u8,
        captured_layers.len(),
        top_k,
        captured_count,
        entries,
        first_layer,
        last_layer,
        checksum,
        format_decode_batch_route_snapshot(&routes, captured_layers, top_k),
    );
    Ok(())
}

fn decode_batch_shared_snapshot_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn decode_batch_shared_snapshot_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_batch_shared_snapshot_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn bf16_head_hex(bytes: &[u8], elems: usize) -> String {
    bytes
        .chunks_exact(2)
        .take(elems)
        .map(|chunk| format!("{:02x}{:02x}", chunk[1], chunk[0]))
        .collect::<Vec<_>>()
        .join(",")
}

fn bf16_full_hex(bytes: &[u8]) -> String {
    bytes
        .chunks_exact(2)
        .map(|chunk| format!("{:02x}{:02x}", chunk[1], chunk[0]))
        .collect::<Vec<_>>()
        .join(",")
}

fn byte_head_hex(bytes: &[u8], elems: usize) -> String {
    bytes
        .iter()
        .take(elems)
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn f32_word_head_hex(bytes: &[u8], elems: usize) -> String {
    bytes
        .chunks_exact(4)
        .take(elems)
        .map(|chunk| {
            let word = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            format!("{word:08x}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn qwen36_full_attn_workspace_tap_word_count() -> Option<usize> {
    std::env::var("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_WORDS")
        .ok()
        .map(|raw| raw.trim().parse::<usize>().unwrap_or(16).max(1))
}

#[allow(clippy::too_many_arguments)]
fn emit_full_attn_workspace_tap_words(
    position: i32,
    cache_pos: i32,
    path: &str,
    layer_idx: usize,
    region: &str,
    bytes: &[u8],
    elems: usize,
) {
    if let Some(count) = qwen36_full_attn_workspace_tap_word_count() {
        eprintln!(
            "[qwen36-full-attn-workspace-tap-words] position={} cache_pos={} path={} layer={} region={} elems={} words={}",
            position,
            cache_pos,
            path,
            layer_idx,
            region,
            elems,
            f32_word_head_hex(bytes, count),
        );
    }
}

fn snapshot_layer_output(
    ordinal: usize,
    snapshots: &mut LayerOutputSnapshots,
    layer_idx: usize,
    phase_idx: usize,
    src: *const std::ffi::c_void,
    hidden: usize,
) -> Result<()> {
    if phase_idx >= LAYER_OUTPUT_TAP_PHASES.len() {
        return Err(anyhow!("invalid qwen36 layer-output tap phase {phase_idx}"));
    }
    let bytes = hidden * std::mem::size_of::<u16>();
    let row = layer_idx * LAYER_OUTPUT_TAP_PHASES.len() + phase_idx;
    let dst = unsafe { (snapshots.output.as_mut_ptr() as *mut u8).add(row * bytes) }
        as *mut std::ffi::c_void;
    copy_d2d_decode(ordinal, snapshots.output.backend(), dst, src, bytes)
        .with_context(|| format!("snapshot qwen36 layer output tap layer={layer_idx}"))?;
    if let Some(captured) = snapshots.captured.get_mut(layer_idx) {
        captured[phase_idx] = true;
    }
    Ok(())
}

fn qwen36_full_attn_kv_cache_tap_enabled() -> bool {
    std::env::var_os("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP").is_some()
}

fn qwen36_full_attn_kv_cache_tap_position() -> Option<i32> {
    std::env::var("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP_POSITION")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

fn qwen36_full_attn_kv_cache_tap_layer() -> Option<usize> {
    std::env::var("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP_LAYER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn qwen36_full_attn_kv_cache_tap_matches(position: i32, layer_idx: usize) -> bool {
    if !qwen36_full_attn_kv_cache_tap_enabled() {
        return false;
    }
    if let Some(wanted) = qwen36_full_attn_kv_cache_tap_position() {
        if position != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_full_attn_kv_cache_tap_layer() {
        if layer_idx != wanted {
            return false;
        }
    }
    true
}

fn qwen36_full_attn_workspace_tap_enabled() -> bool {
    std::env::var_os("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP").is_some()
}

fn qwen36_full_attn_workspace_tap_position() -> Option<i32> {
    std::env::var("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_POSITION")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

fn qwen36_full_attn_workspace_tap_layer() -> Option<usize> {
    std::env::var("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_LAYER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn qwen36_full_attn_workspace_tap_matches(position: i32, layer_idx: usize) -> bool {
    if !qwen36_full_attn_workspace_tap_enabled() {
        return false;
    }
    if let Some(wanted) = qwen36_full_attn_workspace_tap_position() {
        if position != wanted {
            return false;
        }
    }
    if let Some(wanted) = qwen36_full_attn_workspace_tap_layer() {
        if layer_idx != wanted {
            return false;
        }
    }
    true
}

fn download_full_attn_kv_prefix(
    ordinal: usize,
    cache: &FullAttnKvCache,
    elems: usize,
) -> Result<(Vec<u8>, Vec<u8>, ScalarType)> {
    let dtype = cache
        .k
        .as_ref()
        .map(|b| b.dtype())
        .or_else(|| cache.v.as_ref().map(|b| b.dtype()))
        .unwrap_or_else(|| {
            if cache.kv_scale_k.is_some() || cache.kv_scale_v.is_some() {
                ScalarType::U8
            } else {
                ScalarType::BF16
            }
        });
    let bytes = elems
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| anyhow!("qwen36 full-attn KV cache tap byte count overflow"))?;
    if bytes == 0 {
        return Ok((Vec::new(), Vec::new(), dtype));
    }
    let k_bytes = if let Some(k) = cache.k.as_ref() {
        if bytes > k.len_bytes() {
            return Err(anyhow!(
                "qwen36 full-attn K cache tap prefix out of bounds: {bytes}/{} bytes",
                k.len_bytes()
            ));
        }
        let mut out = vec![0u8; bytes];
        copy_d2h(
            ordinal,
            out.as_mut_ptr() as *mut std::ffi::c_void,
            k.as_ptr(),
            bytes,
        )
        .context("d2h qwen36 full-attn K cache tap")?;
        out
    } else if let Some(k) = cache.virtual_kv_cache_k.as_ref() {
        k.to_host_prefix_bytes(bytes)
            .context("d2h qwen36 virtual full-attn K cache tap")?
    } else {
        return Err(anyhow!(
            "qwen36 full-attn K cache tap has no backing buffer"
        ));
    };
    let v_bytes = if let Some(v) = cache.v.as_ref() {
        if bytes > v.len_bytes() {
            return Err(anyhow!(
                "qwen36 full-attn V cache tap prefix out of bounds: {bytes}/{} bytes",
                v.len_bytes()
            ));
        }
        let mut out = vec![0u8; bytes];
        copy_d2h(
            ordinal,
            out.as_mut_ptr() as *mut std::ffi::c_void,
            v.as_ptr(),
            bytes,
        )
        .context("d2h qwen36 full-attn V cache tap")?;
        out
    } else if let Some(v) = cache.virtual_kv_cache_v.as_ref() {
        v.to_host_prefix_bytes(bytes)
            .context("d2h qwen36 virtual full-attn V cache tap")?
    } else {
        return Err(anyhow!(
            "qwen36 full-attn V cache tap has no backing buffer"
        ));
    };
    Ok((k_bytes, v_bytes, dtype))
}

fn emit_full_attn_workspace_tap(
    ordinal: usize,
    position: i32,
    cache_pos: i32,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    workspace: &GpuBuffer,
    path: &str,
) -> Result<()> {
    let num_heads = geom.num_attention_heads as usize;
    let num_kv_heads = geom.num_kv_heads as usize;
    let head_dim = geom.head_dim as usize;
    let hidden = geom.hidden as usize;
    let q_dim = num_heads * head_dim;
    let q_out_dim = 2 * q_dim;
    let kv_dim = num_kv_heads * head_dim;
    let off_q_raw = 0usize;
    let off_k_raw = q_out_dim;
    let off_v_raw = off_k_raw + kv_dim;
    let off_q_normed = off_v_raw + kv_dim;
    let off_k_normed = off_q_normed + q_dim;
    let off_q_rot = off_k_normed + kv_dim;
    let off_k_rot = off_q_rot + q_dim;
    let off_attn = off_k_rot + kv_dim;
    let off_gated = off_attn + q_dim;
    let off_o_out = off_gated + q_dim;
    let eff_cache_pos = if cache_pos >= 0 { cache_pos } else { position };
    let kv_len = (eff_cache_pos + 1).max(0) as usize;
    let score_tap_elems =
        if std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_SCORE_TAP").is_some() {
            num_heads * kv_len
        } else {
            0
        };
    let prob_tap_elems = if std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_PROB_TAP").is_some()
    {
        num_heads * kv_len
    } else {
        0
    };
    let exp_tap_elems = if std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXP_TAP").is_some() {
        num_heads * kv_len
    } else {
        0
    };
    let denom_tap_elems =
        if std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_DENOM_TAP").is_some() {
            num_heads
        } else {
            0
        };
    let workspace_len =
        off_o_out + hidden + score_tap_elems + prob_tap_elems + exp_tap_elems + denom_tap_elems;
    let bytes = workspace_len * std::mem::size_of::<f32>();
    if bytes > workspace.len_bytes() {
        return Err(anyhow!(
            "qwen36 full-attn workspace tap out of bounds: {bytes}/{} bytes",
            workspace.len_bytes()
        ));
    }
    let mut all = vec![0u8; bytes];
    copy_d2h(
        ordinal,
        all.as_mut_ptr() as *mut std::ffi::c_void,
        workspace.as_ptr(),
        bytes,
    )
    .context("d2h qwen36 full-attn workspace tap")?;
    let regions = [
        ("q_raw", off_q_raw, q_out_dim),
        ("k_raw", off_k_raw, kv_dim),
        ("v_raw", off_v_raw, kv_dim),
        ("q_normed", off_q_normed, q_dim),
        ("k_normed", off_k_normed, kv_dim),
        ("q_rot", off_q_rot, q_dim),
        ("k_rot", off_k_rot, kv_dim),
        ("attn", off_attn, q_dim),
        ("gated", off_gated, q_dim),
        ("o_out", off_o_out, hidden),
    ];
    for (name, start, elems) in regions {
        let start_b = start * std::mem::size_of::<f32>();
        let end_b = start_b + elems * std::mem::size_of::<f32>();
        let bytes = &all[start_b..end_b];
        eprintln!(
            "[qwen36-full-attn-workspace-tap] position={} cache_pos={} path={} layer={} region={} elems={} checksum={:016x} head16={}",
            position,
            cache_pos,
            path,
            layer_idx,
            name,
            elems,
            fnv1a64_bytes(bytes),
            byte_head_hex(bytes, 16),
        );
    }
    if score_tap_elems > 0 {
        let start = off_o_out + hidden;
        let start_b = start * std::mem::size_of::<f32>();
        let end_b = start_b + score_tap_elems * std::mem::size_of::<f32>();
        let bytes = &all[start_b..end_b];
        eprintln!(
            "[qwen36-full-attn-workspace-tap] position={} cache_pos={} path={} layer={} region=score_tap elems={} checksum={:016x} head16={}",
            position,
            cache_pos,
            path,
            layer_idx,
            score_tap_elems,
            fnv1a64_bytes(bytes),
            byte_head_hex(bytes, 16),
        );
        emit_full_attn_workspace_tap_words(
            position,
            cache_pos,
            path,
            layer_idx,
            "score_tap",
            bytes,
            score_tap_elems,
        );
    }
    if prob_tap_elems > 0 {
        let start = off_o_out + hidden + score_tap_elems;
        let start_b = start * std::mem::size_of::<f32>();
        let end_b = start_b + prob_tap_elems * std::mem::size_of::<f32>();
        let bytes = &all[start_b..end_b];
        eprintln!(
            "[qwen36-full-attn-workspace-tap] position={} cache_pos={} path={} layer={} region=prob_tap elems={} checksum={:016x} head16={}",
            position,
            cache_pos,
            path,
            layer_idx,
            prob_tap_elems,
            fnv1a64_bytes(bytes),
            byte_head_hex(bytes, 16),
        );
        emit_full_attn_workspace_tap_words(
            position,
            cache_pos,
            path,
            layer_idx,
            "prob_tap",
            bytes,
            prob_tap_elems,
        );
    }
    if exp_tap_elems > 0 {
        let start = off_o_out + hidden + score_tap_elems + prob_tap_elems;
        let start_b = start * std::mem::size_of::<f32>();
        let end_b = start_b + exp_tap_elems * std::mem::size_of::<f32>();
        let bytes = &all[start_b..end_b];
        eprintln!(
            "[qwen36-full-attn-workspace-tap] position={} cache_pos={} path={} layer={} region=exp_tap elems={} checksum={:016x} head16={}",
            position,
            cache_pos,
            path,
            layer_idx,
            exp_tap_elems,
            fnv1a64_bytes(bytes),
            byte_head_hex(bytes, 16),
        );
        emit_full_attn_workspace_tap_words(
            position,
            cache_pos,
            path,
            layer_idx,
            "exp_tap",
            bytes,
            exp_tap_elems,
        );
    }
    if denom_tap_elems > 0 {
        let start = off_o_out + hidden + score_tap_elems + prob_tap_elems + exp_tap_elems;
        let start_b = start * std::mem::size_of::<f32>();
        let end_b = start_b + denom_tap_elems * std::mem::size_of::<f32>();
        let bytes = &all[start_b..end_b];
        eprintln!(
            "[qwen36-full-attn-workspace-tap] position={} cache_pos={} path={} layer={} region=denom_tap elems={} checksum={:016x} head16={}",
            position,
            cache_pos,
            path,
            layer_idx,
            denom_tap_elems,
            fnv1a64_bytes(bytes),
            byte_head_hex(bytes, 16),
        );
        emit_full_attn_workspace_tap_words(
            position,
            cache_pos,
            path,
            layer_idx,
            "denom_tap",
            bytes,
            denom_tap_elems,
        );
    }
    Ok(())
}

fn emit_full_attn_kv_cache_tap(
    ordinal: usize,
    position: i32,
    cache_pos: i32,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    cache: &FullAttnKvCache,
    path: &str,
) -> Result<()> {
    let eff_cache_pos = if cache_pos >= 0 { cache_pos } else { position };
    let kv_len = (eff_cache_pos + 1).max(0) as usize;
    let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
    let elems = kv_len
        .checked_mul(kv_dim)
        .ok_or_else(|| anyhow!("qwen36 full-attn KV cache tap element count overflow"))?;
    let (k_bytes, v_bytes, dtype) = download_full_attn_kv_prefix(ordinal, cache, elems)?;
    let (k_head, v_head) = if dtype == ScalarType::BF16 {
        (bf16_head_hex(&k_bytes, 8), bf16_head_hex(&v_bytes, 8))
    } else {
        (byte_head_hex(&k_bytes, 16), byte_head_hex(&v_bytes, 16))
    };
    eprintln!(
        "[qwen36-full-attn-kv-cache-tap] position={} cache_pos={} eff_cache_pos={} path={} layer={} kv_len={} kv_dim={} dtype={:?} k_bytes={} k_checksum={:016x} k_head={} v_bytes={} v_checksum={:016x} v_head={}",
        position,
        cache_pos,
        eff_cache_pos,
        path,
        layer_idx,
        kv_len,
        kv_dim,
        dtype,
        k_bytes.len(),
        fnv1a64_bytes(&k_bytes),
        k_head,
        v_bytes.len(),
        fnv1a64_bytes(&v_bytes),
        v_head,
    );
    Ok(())
}

fn emit_layer_output_taps(
    position: i32,
    cache_pos: i32,
    path: &str,
    phase_profile: bool,
    geom: &MultiLayerGeom,
    snapshots: &LayerOutputSnapshots,
) -> Result<()> {
    let bytes = snapshots
        .output
        .to_host_bytes()
        .context("d2h qwen36 layer-output tap snapshots")?;
    let hidden = geom.hidden as usize;
    let row_bytes = hidden * std::mem::size_of::<u16>();

    for (layer_idx, captured) in snapshots.captured.iter().enumerate() {
        for (phase_idx, &is_captured) in captured.iter().enumerate() {
            if !is_captured {
                continue;
            }
            let row = layer_idx * LAYER_OUTPUT_TAP_PHASES.len() + phase_idx;
            let start = row * row_bytes;
            let end = start + row_bytes;
            if end > bytes.len() {
                return Err(anyhow!(
                    "qwen36 layer-output tap snapshot out of bounds: layer={layer_idx} phase={} end={end}/{}",
                    LAYER_OUTPUT_TAP_PHASES[phase_idx],
                    bytes.len()
                ));
            }
            let row_bytes = &bytes[start..end];
            let hidden_f32 = bf16_bytes_to_f32(row_bytes);
            let mut l2 = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut max_abs_idx = 0usize;
            for (idx, &value) in hidden_f32.iter().enumerate() {
                l2 += (value as f64) * (value as f64);
                let abs = value.abs();
                if abs > max_abs {
                    max_abs = abs;
                    max_abs_idx = idx;
                }
            }
            let call = QWEN36_LAYER_OUTPUT_TAP_CALLS.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[qwen36-layer-output-tap] call={} position={} cache_pos={} path={} phase_profile={} layer={} phase={} elems={} checksum={:016x} l2={:.8e} max_abs={:.8e} max_abs_idx={} head8={}",
                call,
                position,
                cache_pos,
                path,
                phase_profile as u8,
                layer_idx,
                LAYER_OUTPUT_TAP_PHASES[phase_idx],
                hidden_f32.len(),
                fnv1a64_bytes(row_bytes),
                l2.sqrt(),
                max_abs,
                max_abs_idx,
                bf16_head_hex(row_bytes, 8),
            );
            if qwen36_layer_output_delta_tap_matches(position, layer_idx, phase_idx) {
                let delta_call =
                    QWEN36_LAYER_OUTPUT_DELTA_TAP_CALLS.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[qwen36-layer-output-delta-tap] call={} position={} cache_pos={} path={} phase_profile={} layer={} phase={} elems={} checksum={:016x} bf16={}",
                    delta_call,
                    position,
                    cache_pos,
                    path,
                    phase_profile as u8,
                    layer_idx,
                    LAYER_OUTPUT_TAP_PHASES[phase_idx],
                    hidden_f32.len(),
                    fnv1a64_bytes(row_bytes),
                    bf16_full_hex(row_bytes),
                );
            }
        }
    }
    Ok(())
}

fn emit_decode_batch_shared_parity_taps(
    position: i32,
    cache_pos: i32,
    geom: &MultiLayerGeom,
    layers: &[LayerBuffers],
    snapshots: &DecodeBatchSharedParitySnapshots,
) -> Result<()> {
    let input_bytes = snapshots
        .input
        .to_host_bytes()
        .context("d2h qwen36 decode-batch shared parity inputs")?;
    let workspace_bytes = snapshots
        .workspace
        .to_host_bytes()
        .context("d2h qwen36 decode-batch shared parity workspaces")?;
    let hidden = geom.hidden as usize;
    let input_stride_bytes = hidden * std::mem::size_of::<u16>();
    let workspace_stride_bytes = snapshots.workspace_floats * std::mem::size_of::<f32>();
    let router_path = qwen36_metal_decode_batch_router_path_label();
    let phase_profile = qwen36_metal_decode_batch_phase_profile_enabled();
    let max_calls = qwen36_metal_decode_batch_shared_stage5_parity_tap_max_calls();

    for (layer_idx, layer) in layers.iter().enumerate() {
        if !snapshots.captured.get(layer_idx).copied().unwrap_or(false) {
            continue;
        }
        if !qwen36_metal_decode_batch_shared_stage5_parity_tap_matches(position, layer_idx) {
            continue;
        }
        let call = QWEN36_DECODE_BATCH_SHARED_PARITY_TAP_CALLS.fetch_add(1, Ordering::Relaxed);
        if call >= max_calls {
            continue;
        }

        let input_start = layer_idx * input_stride_bytes;
        let input_end = input_start + input_stride_bytes;
        let workspace_start = layer_idx * workspace_stride_bytes;
        let workspace_end = workspace_start + workspace_stride_bytes;
        if input_end > input_bytes.len() || workspace_end > workspace_bytes.len() {
            return Err(anyhow!(
                "qwen36 decode-batch shared parity snapshot out of bounds: layer={layer_idx} input_end={input_end}/{} workspace_end={workspace_end}/{}",
                input_bytes.len(),
                workspace_bytes.len()
            ));
        }

        let input = decode_batch_shared_snapshot_u16(&input_bytes[input_start..input_end]);
        let workspace =
            decode_batch_shared_snapshot_f32(&workspace_bytes[workspace_start..workspace_end]);
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
            input_hidden: input.as_ptr() as *const std::ffi::c_void,
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
                    gate_up_proj_type: s.gate_up_proj_type,
                    gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                    gate_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.gate_up_proj_zero.as_ptr()
                    },
                    down_proj_type: s.down_proj_type,
                    down_proj_scale: s.down_proj_scale.as_ptr(),
                    down_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.down_proj_zero.as_ptr()
                    },
                    shared_gate_proj_type: s.shared_gate_proj_type,
                    shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                    shared_gate_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_gate_proj_zero.as_ptr()
                    },
                    shared_up_proj_type: s.shared_up_proj_type,
                    shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                    shared_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_up_proj_zero.as_ptr()
                    },
                    shared_down_proj_type: s.shared_down_proj_type,
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

        emit_decode_batch_shared_stage5_parity_tap_from_host(
            call,
            position,
            cache_pos,
            layer_idx as i32,
            router_path,
            phase_profile,
            &input,
            &workspace,
            params,
            &weights,
            &int4,
        )
        .with_context(|| format!("emit decode-batch shared parity tap (layer {layer_idx})"))?;
    }

    Ok(())
}

fn emit_decode_batch_routed_parity_taps(
    position: i32,
    cache_pos: i32,
    geom: &MultiLayerGeom,
    layers: &[LayerBuffers],
    snapshots: &DecodeBatchSharedParitySnapshots,
) -> Result<()> {
    let input_bytes = snapshots
        .input
        .to_host_bytes()
        .context("d2h qwen36 decode-batch routed parity inputs")?;
    let workspace_bytes = snapshots
        .workspace
        .to_host_bytes()
        .context("d2h qwen36 decode-batch routed parity workspaces")?;
    let output_bytes = snapshots
        .output
        .to_host_bytes()
        .context("d2h qwen36 decode-batch routed parity outputs")?;
    let output_idx_bytes = snapshots
        .output_idx
        .to_host_bytes()
        .context("d2h qwen36 decode-batch routed parity output_idx")?;
    let hidden = geom.hidden as usize;
    let top_k = geom.top_k as usize;
    let input_stride_bytes = hidden * std::mem::size_of::<u16>();
    let workspace_stride_bytes = snapshots.workspace_floats * std::mem::size_of::<f32>();
    let output_stride_bytes = hidden * std::mem::size_of::<u16>();
    let output_idx_stride_bytes = top_k * std::mem::size_of::<u32>();
    let router_path = qwen36_metal_decode_batch_router_path_label();
    let phase_profile = qwen36_metal_decode_batch_phase_profile_enabled();
    let max_calls = qwen36_metal_decode_batch_routed_stage5_parity_tap_max_calls();

    for (layer_idx, layer) in layers.iter().enumerate() {
        if !snapshots.captured.get(layer_idx).copied().unwrap_or(false) {
            continue;
        }
        if !qwen36_metal_decode_batch_routed_stage5_parity_tap_matches(position, layer_idx) {
            continue;
        }
        let call = QWEN36_DECODE_BATCH_ROUTED_PARITY_TAP_CALLS.fetch_add(1, Ordering::Relaxed);
        if call >= max_calls {
            continue;
        }

        let input_start = layer_idx * input_stride_bytes;
        let input_end = input_start + input_stride_bytes;
        let workspace_start = layer_idx * workspace_stride_bytes;
        let workspace_end = workspace_start + workspace_stride_bytes;
        let output_start = layer_idx * output_stride_bytes;
        let output_end = output_start + output_stride_bytes;
        let output_idx_start = layer_idx * output_idx_stride_bytes;
        let output_idx_end = output_idx_start + output_idx_stride_bytes;
        if input_end > input_bytes.len()
            || workspace_end > workspace_bytes.len()
            || output_end > output_bytes.len()
            || output_idx_end > output_idx_bytes.len()
        {
            return Err(anyhow!(
                "qwen36 decode-batch routed parity snapshot out of bounds: layer={layer_idx} input_end={input_end}/{} workspace_end={workspace_end}/{} output_end={output_end}/{} output_idx_end={output_idx_end}/{}",
                input_bytes.len(),
                workspace_bytes.len(),
                output_bytes.len(),
                output_idx_bytes.len()
            ));
        }

        let input = decode_batch_shared_snapshot_u16(&input_bytes[input_start..input_end]);
        let workspace =
            decode_batch_shared_snapshot_f32(&workspace_bytes[workspace_start..workspace_end]);
        let output = decode_batch_shared_snapshot_u16(&output_bytes[output_start..output_end]);
        let output_idx =
            decode_batch_shared_snapshot_u32(&output_idx_bytes[output_idx_start..output_idx_end]);
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
            input_hidden: input.as_ptr() as *const std::ffi::c_void,
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
                    gate_up_proj_type: s.gate_up_proj_type,
                    gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                    gate_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.gate_up_proj_zero.as_ptr()
                    },
                    down_proj_type: s.down_proj_type,
                    down_proj_scale: s.down_proj_scale.as_ptr(),
                    down_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.down_proj_zero.as_ptr()
                    },
                    shared_gate_proj_type: s.shared_gate_proj_type,
                    shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                    shared_gate_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_gate_proj_zero.as_ptr()
                    },
                    shared_up_proj_type: s.shared_up_proj_type,
                    shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                    shared_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_up_proj_zero.as_ptr()
                    },
                    shared_down_proj_type: s.shared_down_proj_type,
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

        emit_decode_batch_routed_stage5_parity_tap_from_host(
            call,
            position,
            cache_pos,
            layer_idx as i32,
            router_path,
            phase_profile,
            &input,
            &workspace,
            &output,
            &output_idx,
            params,
            &weights,
            &int4,
        )
        .with_context(|| format!("emit decode-batch routed parity tap (layer {layer_idx})"))?;
    }

    Ok(())
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
    let full_attn_tap_regions =
        usize::from(std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_SCORE_TAP").is_some())
            + usize::from(std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_PROB_TAP").is_some())
            + usize::from(std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXP_TAP").is_some())
            + usize::from(
                std::env::var_os("SUPERSONIC_METAL_QWEN36_FULL_ATTN_DENOM_TAP").is_some(),
            );
    let attn_extra_regions = full_attn_tap_regions.max(1);
    let attn_extra = if max_kv_t > 0 {
        geom.num_attention_heads as usize * max_kv_t * attn_extra_regions
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
    let ffn_ws_floats = ffn_workspace_floats(geom);
    let mut ffn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_ws_floats])
        .context("alloc ffn_workspace")?;

    let native_full_attn_rope = if hidden_a.backend() == Backend::Metal
        && std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE").is_some()
    {
        Some(
            DecodeRotaryRow::build(ordinal, position, geom.rotary_dim as usize, geom.rope_theta)
                .context("build qwen36 decode RoPE row")?,
        )
    } else {
        None
    };

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

    let metal_decode_batch_requested = hidden_a.backend() == Backend::Metal
        && qwen36_metal_decode_batch_enabled(
            capture,
            options.accurate_stage_timings,
            expert_prefetch.is_some(),
        );
    let mut decode_batch_route_snapshot =
        if metal_decode_batch_requested && qwen36_metal_decode_batch_route_snapshot_enabled() {
            Some(
                GpuBuffer::zeros(
                    ordinal,
                    ScalarType::U32,
                    &[layers.len(), geom.top_k as usize],
                )
                .context("alloc qwen36 decode-batch route snapshot")?,
            )
        } else {
            None
        };
    let mut decode_batch_route_snapshot_captured = vec![false; layers.len()];
    let decode_batch_shared_parity_tap =
        qwen36_metal_decode_batch_shared_stage5_parity_tap_enabled();
    let decode_batch_routed_parity_tap =
        qwen36_metal_decode_batch_routed_stage5_parity_tap_enabled();
    let mut decode_batch_shared_parity = if metal_decode_batch_requested
        && (decode_batch_shared_parity_tap || decode_batch_routed_parity_tap)
    {
        Some(DecodeBatchSharedParitySnapshots {
            input: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[layers.len(), hidden])
                .context("alloc qwen36 decode-batch shared parity input snapshots")?,
            workspace: GpuBuffer::zeros(ordinal, ScalarType::F32, &[layers.len(), ffn_ws_floats])
                .context("alloc qwen36 decode-batch shared parity workspace snapshots")?,
            output: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[layers.len(), hidden])
                .context("alloc qwen36 decode-batch routed parity output snapshots")?,
            output_idx: GpuBuffer::zeros(
                ordinal,
                ScalarType::U32,
                &[layers.len(), geom.top_k as usize],
            )
            .context("alloc qwen36 decode-batch routed parity output_idx snapshots")?,
            captured: vec![false; layers.len()],
            workspace_floats: ffn_ws_floats,
        })
    } else {
        None
    };
    let mut layer_output_tap = if qwen36_layer_output_tap_enabled() {
        Some(LayerOutputSnapshots {
            output: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[layers.len(), LAYER_OUTPUT_TAP_PHASE_COUNT, hidden],
            )
            .context("alloc qwen36 layer-output tap snapshots")?,
            captured: vec![[false; LAYER_OUTPUT_TAP_PHASE_COUNT]; layers.len()],
        })
    } else {
        None
    };

    let metal_decode_batch = if metal_decode_batch_requested {
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
                        gate_up_proj_type: s.gate_up_proj_type,
                        gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                        gate_up_proj_zero: if fp8 {
                            ptr::null()
                        } else {
                            s.gate_up_proj_zero.as_ptr()
                        },
                        down_proj_type: s.down_proj_type,
                        down_proj_scale: s.down_proj_scale.as_ptr(),
                        down_proj_zero: if fp8 {
                            ptr::null()
                        } else {
                            s.down_proj_zero.as_ptr()
                        },
                        shared_gate_proj_type: s.shared_gate_proj_type,
                        shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                        shared_gate_proj_zero: if fp8 {
                            ptr::null()
                        } else {
                            s.shared_gate_proj_zero.as_ptr()
                        },
                        shared_up_proj_type: s.shared_up_proj_type,
                        shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                        shared_up_proj_zero: if fp8 {
                            ptr::null()
                        } else {
                            s.shared_up_proj_zero.as_ptr()
                        },
                        shared_down_proj_type: s.shared_down_proj_type,
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
                && (metal_decode_batch_active || ffn_stage5_router_defer_wait_enabled())
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
                            q_proj_type: s.q_proj_type,
                            q_proj_scale: s.q_proj_scale.as_ptr(),
                            q_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.q_proj_zero.as_ptr()
                            },
                            k_proj_type: s.k_proj_type,
                            k_proj_scale: s.k_proj_scale.as_ptr(),
                            k_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.k_proj_zero.as_ptr()
                            },
                            v_proj_type: s.v_proj_type,
                            v_proj_scale: s.v_proj_scale.as_ptr(),
                            v_proj_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.v_proj_zero.as_ptr()
                            },
                            o_proj_type: s.o_proj_type,
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
                let use_metal_native = output_buf.backend() == Backend::Metal
                    && attn_step_stage5_metal_native_supported(
                        params, &weights, &int4_ptrs, hidden,
                    )
                    && qwen36_full_attn_native_enabled_for_layer(layer_idx);
                let use_metal_direct = !use_metal_native
                    && output_buf.backend() == Backend::Metal
                    && metal_full_attn_decode_direct_enabled(&int4_ptrs);
                if (metal_queue_dirty || metal_decode_batch_active) && !use_metal_native {
                    sync_metal_queue_for_host_boundary(
                        input_backend,
                        "sync before full-attn host read",
                    )?;
                    metal_queue_dirty = false;
                }
                if use_metal_native {
                    unsafe {
                        attn_step_stage5_metal_native_into(
                            params,
                            &weights,
                            &int4_ptrs,
                            native_full_attn_rope
                                .as_ref()
                                .map(|row| row.cos.as_ptr())
                                .unwrap_or(ptr::null()),
                            native_full_attn_rope
                                .as_ref()
                                .map(|row| row.sin.as_ptr())
                                .unwrap_or(ptr::null()),
                            &mut attn_output,
                            &mut attn_workspace,
                            output_buf.as_mut_ptr(),
                            hidden,
                            !metal_decode_batch_active,
                        )
                    }
                    .with_context(|| {
                        format!("attn_step_stage5_metal_native_into (layer {layer_idx})")
                    })?;
                    attn_output_published = true;
                } else if use_metal_direct {
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
                    reset_sync_buf(ordinal, &mut sync_buf).context("reset sync_buf (attn)")?;
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
                if qwen36_full_attn_kv_cache_tap_matches(position, layer_idx) {
                    sync_metal_queue_for_host_boundary(
                        output_buf.backend(),
                        "sync before full-attn KV cache tap",
                    )?;
                    if let Some(cache) = kv_cache.as_ref() {
                        let path = if use_metal_native {
                            "native"
                        } else if use_metal_direct {
                            "direct"
                        } else {
                            "kernel"
                        };
                        emit_full_attn_kv_cache_tap(
                            ordinal, position, cache_pos, layer_idx, geom, cache, path,
                        )?;
                    }
                }
                if qwen36_full_attn_workspace_tap_matches(position, layer_idx) {
                    sync_metal_queue_for_host_boundary(
                        output_buf.backend(),
                        "sync before full-attn workspace tap",
                    )?;
                    let path = if use_metal_native {
                        "native"
                    } else if use_metal_direct {
                        "direct"
                    } else {
                        "kernel"
                    };
                    emit_full_attn_workspace_tap(
                        ordinal,
                        position,
                        cache_pos,
                        layer_idx,
                        geom,
                        &attn_workspace,
                        path,
                    )?;
                }
                if options.accurate_stage_timings {
                    sync_decode_queue(ordinal, output_buf.backend(), "sync_after_attn_step")?;
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
                            in_proj_qkv_type: s.in_proj_qkv_type,
                            in_proj_qkv_scale: s.in_proj_qkv_scale.as_ptr(),
                            in_proj_qkv_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.in_proj_qkv_zero.as_ptr()
                            },
                            in_proj_z_type: s.in_proj_z_type,
                            in_proj_z_scale: s.in_proj_z_scale.as_ptr(),
                            in_proj_z_zero: if fp8 {
                                ptr::null()
                            } else {
                                s.in_proj_z_zero.as_ptr()
                            },
                            out_proj_type: s.out_proj_type,
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
                    reset_sync_buf(ordinal, &mut sync_buf)
                        .context("reset sync_buf (linear-attn)")?;
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
                    sync_decode_queue(ordinal, output_buf.backend(), "sync_after_linear_step")?;
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
                output_buf.backend(),
                output_buf.as_mut_ptr(),
                attn_output.as_ptr(),
                hidden * 2,
            )
            .context("d2d attn_output -> residual")?;
        }
        if let Some(snapshots) = layer_output_tap.as_mut() {
            if qwen36_layer_output_tap_should_snapshot(position, layer_idx, 0) {
                snapshot_layer_output(
                    ordinal,
                    snapshots,
                    layer_idx,
                    0,
                    output_buf.as_ptr(),
                    hidden,
                )?;
            }
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
                    gate_up_proj_type: s.gate_up_proj_type,
                    gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                    gate_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.gate_up_proj_zero.as_ptr()
                    },
                    down_proj_type: s.down_proj_type,
                    down_proj_scale: s.down_proj_scale.as_ptr(),
                    down_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.down_proj_zero.as_ptr()
                    },
                    shared_gate_proj_type: s.shared_gate_proj_type,
                    shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                    shared_gate_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_gate_proj_zero.as_ptr()
                    },
                    shared_up_proj_type: s.shared_up_proj_type,
                    shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                    shared_up_proj_zero: if fp8 {
                        ptr::null()
                    } else {
                        s.shared_up_proj_zero.as_ptr()
                    },
                    shared_down_proj_type: s.shared_down_proj_type,
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
            reset_sync_buf(ordinal, &mut sync_buf)
                .context("reset sync_buf (ffn router prefetch)")?;
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
                sync_decode_queue(
                    ordinal,
                    ffn_output.backend(),
                    "sync_after_ffn_router_prefetch",
                )?;
            }
            t_ffn += t_router.elapsed();

            let routes = download_topk_routes(&ffn_output_idx, &ffn_output, geom.top_k as usize)
                .with_context(|| format!("download FFN top-k routes (layer {layer_idx})"))?;
            prefetch(ExpertPrefetchPhase::Demand, layer_idx, &routes)
                .with_context(|| format!("prefetch routed experts (layer {layer_idx})"))?;
        }
        if ffn_uses_router_native {
            if let Some(snapshots) = decode_batch_shared_parity.as_mut() {
                let should_snapshot_shared = decode_batch_shared_parity_tap
                    && qwen36_metal_decode_batch_shared_stage5_parity_tap_matches(
                        position, layer_idx,
                    );
                let should_snapshot_routed = decode_batch_routed_parity_tap
                    && qwen36_metal_decode_batch_routed_stage5_parity_tap_matches(
                        position, layer_idx,
                    );
                if should_snapshot_shared || should_snapshot_routed {
                    let bytes = hidden * std::mem::size_of::<u16>();
                    let dst = unsafe {
                        (snapshots.input.as_mut_ptr() as *mut u8).add(layer_idx * bytes)
                            as *mut std::ffi::c_void
                    };
                    copy_d2d_decode(ordinal, snapshots.input.backend(), dst, input_ptr, bytes)
                        .with_context(|| {
                            format!("snapshot decode-batch FFN shared input (layer {layer_idx})")
                        })?;
                }
            }
        }
        let t_k = std::time::Instant::now();
        let ffn_stage5_output = output_buf;
        if !ffn_uses_router_native {
            reset_sync_buf(ordinal, &mut sync_buf).context("reset sync_buf (ffn)")?;
        }
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
        if let Some(snapshots) = layer_output_tap.as_mut() {
            if qwen36_layer_output_tap_should_snapshot(position, layer_idx, 1) {
                snapshot_layer_output(
                    ordinal,
                    snapshots,
                    layer_idx,
                    1,
                    ffn_stage5_output.as_ptr(),
                    hidden,
                )?;
            }
        }
        if ffn_uses_router_native {
            if let Some(snapshot) = decode_batch_route_snapshot.as_mut() {
                let bytes = geom.top_k as usize * std::mem::size_of::<u32>();
                let dst = unsafe {
                    (snapshot.as_mut_ptr() as *mut u8).add(layer_idx * bytes)
                        as *mut std::ffi::c_void
                };
                copy_d2d_decode(
                    ordinal,
                    snapshot.backend(),
                    dst,
                    ffn_output_idx.as_ptr(),
                    bytes,
                )
                .with_context(|| {
                    format!("snapshot decode-batch FFN top-k routes (layer {layer_idx})")
                })?;
                decode_batch_route_snapshot_captured[layer_idx] = true;
            }
            if let Some(snapshots) = decode_batch_shared_parity.as_mut() {
                let should_snapshot_shared = decode_batch_shared_parity_tap
                    && qwen36_metal_decode_batch_shared_stage5_parity_tap_matches(
                        position, layer_idx,
                    );
                let should_snapshot_routed = decode_batch_routed_parity_tap
                    && qwen36_metal_decode_batch_routed_stage5_parity_tap_matches(
                        position, layer_idx,
                    );
                if should_snapshot_shared || should_snapshot_routed {
                    let bytes = snapshots.workspace_floats * std::mem::size_of::<f32>();
                    let dst = unsafe {
                        (snapshots.workspace.as_mut_ptr() as *mut u8).add(layer_idx * bytes)
                            as *mut std::ffi::c_void
                    };
                    copy_d2d_decode(
                        ordinal,
                        snapshots.workspace.backend(),
                        dst,
                        ffn_workspace.as_ptr(),
                        bytes,
                    )
                    .with_context(|| {
                        format!("snapshot decode-batch FFN shared workspace (layer {layer_idx})")
                    })?;
                }
                if should_snapshot_routed {
                    let output_bytes = hidden * std::mem::size_of::<u16>();
                    let output_dst = unsafe {
                        (snapshots.output.as_mut_ptr() as *mut u8).add(layer_idx * output_bytes)
                            as *mut std::ffi::c_void
                    };
                    copy_d2d_decode(
                        ordinal,
                        snapshots.output.backend(),
                        output_dst,
                        ffn_stage5_output.as_ptr(),
                        output_bytes,
                    )
                    .with_context(|| {
                        format!("snapshot decode-batch FFN routed output (layer {layer_idx})")
                    })?;

                    let idx_bytes = geom.top_k as usize * std::mem::size_of::<u32>();
                    let idx_dst = unsafe {
                        (snapshots.output_idx.as_mut_ptr() as *mut u8).add(layer_idx * idx_bytes)
                            as *mut std::ffi::c_void
                    };
                    copy_d2d_decode(
                        ordinal,
                        snapshots.output_idx.backend(),
                        idx_dst,
                        ffn_output_idx.as_ptr(),
                        idx_bytes,
                    )
                    .with_context(|| {
                        format!("snapshot decode-batch FFN routed output_idx (layer {layer_idx})")
                    })?;
                }
                if should_snapshot_shared || should_snapshot_routed {
                    snapshots.captured[layer_idx] = true;
                }
            }
        }
        metal_queue_dirty = defer_layer_ffn_router || defer_layer_ffn_direct_gather;
        if options.accurate_stage_timings {
            sync_decode_queue(ordinal, ffn_stage5_output.backend(), "sync_after_ffn_step")?;
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
    if let Some(snapshot) = decode_batch_route_snapshot.as_ref() {
        emit_decode_batch_route_snapshot(
            position,
            cache_pos,
            geom.top_k as usize,
            &decode_batch_route_snapshot_captured,
            snapshot,
        )?;
    }
    if decode_batch_shared_parity_tap {
        if let Some(snapshots) = decode_batch_shared_parity.as_ref() {
            emit_decode_batch_shared_parity_taps(position, cache_pos, geom, layers, snapshots)?;
        }
    }
    if decode_batch_routed_parity_tap {
        if let Some(snapshots) = decode_batch_shared_parity.as_ref() {
            emit_decode_batch_routed_parity_taps(position, cache_pos, geom, layers, snapshots)?;
        }
    }
    if let Some(snapshots) = layer_output_tap.as_ref() {
        let path = if metal_decode_batch_active {
            "decode_batch"
        } else {
            "chained"
        };
        emit_layer_output_taps(
            position,
            cache_pos,
            path,
            qwen36_metal_decode_batch_phase_profile_enabled(),
            geom,
            snapshots,
        )?;
    }
    if let Some(batch) = metal_decode_batch {
        batch
            .finish()
            .map_err(|e| anyhow!("qwen36 Metal decode batch finish: {e}"))?;
    }

    Ok(DecodeOutputs {
        path_label: if metal_decode_batch_active {
            "decode_batch"
        } else {
            "chained"
        },
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
    use super::{
        decode_batch_route_snapshot_checksum, decode_batch_route_snapshot_u32,
        decode_batch_shared_snapshot_f32, decode_batch_shared_snapshot_u16,
        format_decode_batch_route_snapshot,
    };

    #[test]
    fn decode_batch_route_snapshot_formats_captured_layers() {
        let routes = vec![1, 2, 3, 4, 5, 6];
        let captured = vec![true, false, true];

        assert_eq!(
            format_decode_batch_route_snapshot(&routes, &captured, 2),
            "1,2;-;5,6"
        );
        assert_ne!(
            decode_batch_route_snapshot_checksum(&routes, &captured, 2),
            decode_batch_route_snapshot_checksum(&routes, &[true, true, true], 2)
        );
    }

    #[test]
    fn decode_batch_route_snapshot_decodes_u32_bytes() {
        let bytes = [7u32.to_ne_bytes(), 11u32.to_ne_bytes()].concat();

        assert_eq!(decode_batch_route_snapshot_u32(&bytes), vec![7, 11]);
    }

    #[test]
    fn decode_batch_shared_parity_snapshot_decodes_rows() {
        let u16_bytes = [3u16.to_ne_bytes(), 9u16.to_ne_bytes()].concat();
        let f32_bytes = [1.5f32.to_ne_bytes(), (-2.25f32).to_ne_bytes()].concat();

        assert_eq!(decode_batch_shared_snapshot_u16(&u16_bytes), vec![3, 9]);
        assert_eq!(
            decode_batch_shared_snapshot_f32(&f32_bytes),
            vec![1.5, -2.25]
        );
    }
}
