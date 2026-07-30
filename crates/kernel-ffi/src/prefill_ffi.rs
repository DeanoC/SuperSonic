//! FFI bindings for prefill kernels.
//! These are component kernels (not megakernels) — the prefill engine
//! orchestrates them layer by layer.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::{metal_host, metal_native};
use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

static METAL_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static METAL_PROFILE: OnceLock<Mutex<MetalProfileAccumulator>> = OnceLock::new();
static FFI_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static FFI_PROFILE: OnceLock<Mutex<FfiProfileAccumulator>> = OnceLock::new();

#[derive(Clone, Default)]
pub struct PrefillFfiLaunchOptions {
    pub force_host_native: bool,
    pub force_host_rms_norm: bool,
    pub force_host_matmul: bool,
    pub force_host_element_add: bool,
    pub force_host_cast: bool,
    pub force_host_transpose_shd_hsd: bool,
    pub force_host_split_qgate: bool,
    pub disable_gemv_m1: bool,
    pub disable_gemv_m1_tiled: bool,
    pub disable_int4_gemv_m1: bool,
    pub disable_int4_gemv_m1_tiled: bool,
    pub metal_profile: bool,
    pub metal_profile_qwen36_ffn_phases: bool,
    pub ffi_profile_shapes: bool,
}

#[cfg(test)]
mod explicit_options_tests {
    use super::*;

    #[test]
    fn prefill_kernel_policy_is_selected_from_explicit_typed_options() {
        let options = PrefillFfiLaunchOptions {
            force_host_native: true,
            force_host_matmul: true,
            disable_gemv_m1: true,
            metal_profile: true,
            ..PrefillFfiLaunchOptions::default()
        };

        assert!(prefill_native_disabled(Some(&options)));
        assert!(prefill_launch_flag(
            Some(&options),
            |options| options.force_host_matmul,
            || false,
        ));
        assert!(prefill_launch_flag(
            Some(&options),
            |options| options.disable_gemv_m1,
            || false,
        ));
        assert!(!prefill_profile_shapes(Some(&options)));
    }

    #[test]
    fn lm_head_metal_policy_ignores_opposite_ambient_flags() {
        const FLAGS: [&str; 5] = [
            "SUPERSONIC_METAL_FORCE_HOST_NATIVE",
            "SUPERSONIC_METAL_FORCE_HOST_RMS_NORM",
            "SUPERSONIC_METAL_FORCE_HOST_MATMUL",
            "SUPERSONIC_METAL_DISABLE_GEMV_M1",
            "SUPERSONIC_METAL_DISABLE_GEMV_M1_TILED",
        ];
        let previous: Vec<_> = FLAGS
            .iter()
            .map(|&name| (name, std::env::var_os(name)))
            .collect();
        for &name in &FLAGS {
            unsafe {
                std::env::set_var(name, "1");
            }
        }

        let explicit_off = PrefillFfiLaunchOptions::default();
        assert!(!prefill_native_disabled(Some(&explicit_off)));
        assert!(!prefill_launch_flag(
            Some(&explicit_off),
            |options| options.force_host_rms_norm,
            metal_force_host_rms_norm,
        ));
        assert!(!prefill_launch_flag(
            Some(&explicit_off),
            |options| options.force_host_matmul,
            metal_force_host_matmul,
        ));
        assert!(!prefill_launch_flag(
            Some(&explicit_off),
            |options| options.disable_gemv_m1,
            || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1").is_some(),
        ));
        assert!(!prefill_launch_flag(
            Some(&explicit_off),
            |options| options.disable_gemv_m1_tiled,
            || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1_TILED").is_some(),
        ));

        for &name in &FLAGS {
            unsafe {
                std::env::remove_var(name);
            }
        }
        let explicit_on = PrefillFfiLaunchOptions {
            force_host_native: true,
            force_host_rms_norm: true,
            force_host_matmul: true,
            disable_gemv_m1: true,
            disable_gemv_m1_tiled: true,
            ..PrefillFfiLaunchOptions::default()
        };
        assert!(prefill_native_disabled(Some(&explicit_on)));
        assert!(prefill_launch_flag(
            Some(&explicit_on),
            |options| options.force_host_rms_norm,
            metal_force_host_rms_norm,
        ));
        assert!(prefill_launch_flag(
            Some(&explicit_on),
            |options| options.force_host_matmul,
            metal_force_host_matmul,
        ));
        assert!(prefill_launch_flag(
            Some(&explicit_on),
            |options| options.disable_gemv_m1,
            || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1").is_some(),
        ));
        assert!(prefill_launch_flag(
            Some(&explicit_on),
            |options| options.disable_gemv_m1_tiled,
            || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1_TILED").is_some(),
        ));

        for &(name, ref value) in &previous {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn prefill_native_disabled(options: Option<&PrefillFfiLaunchOptions>) -> bool {
    options
        .map(|options| options.force_host_native)
        .unwrap_or_else(metal_native::disabled_by_env)
}

fn prefill_launch_flag(
    options: Option<&PrefillFfiLaunchOptions>,
    field: impl FnOnce(&PrefillFfiLaunchOptions) -> bool,
    legacy: impl FnOnce() -> bool,
) -> bool {
    options.map(field).unwrap_or_else(legacy)
}

fn prefill_profile_shapes(options: Option<&PrefillFfiLaunchOptions>) -> bool {
    options
        .map(|options| options.ffi_profile_shapes)
        .unwrap_or_else(|| std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some())
}

#[derive(Debug, Clone)]
pub struct MetalProfileEntry {
    pub op: String,
    pub path: String,
    pub calls: u64,
    pub total_ms: f64,
    pub max_ms: f64,
}

impl MetalProfileEntry {
    pub fn mean_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ms / self.calls as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetalProfileSnapshot {
    pub total_calls: u64,
    pub native_calls: u64,
    pub host_calls: u64,
    pub total_ms: f64,
    pub native_ms: f64,
    pub host_ms: f64,
    pub entries: Vec<MetalProfileEntry>,
}

#[derive(Debug, Default)]
struct MetalProfileAccumulator {
    entries: BTreeMap<(String, String), MetalProfileEntry>,
}

#[derive(Debug, Clone)]
pub struct FfiProfileEntry {
    pub op: String,
    pub calls: u64,
    pub total_ms: f64,
    pub max_ms: f64,
}

impl FfiProfileEntry {
    pub fn mean_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ms / self.calls as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FfiProfileSnapshot {
    pub total_calls: u64,
    pub total_ms: f64,
    pub entries: Vec<FfiProfileEntry>,
}

#[derive(Debug, Default)]
struct FfiProfileAccumulator {
    entries: BTreeMap<String, FfiProfileEntry>,
}

pub fn metal_profile_set_enabled(enabled: bool) {
    METAL_PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn metal_profile_enabled() -> bool {
    METAL_PROFILE_ENABLED.load(Ordering::Relaxed)
}

pub fn metal_profile_reset() {
    if let Some(profile) = METAL_PROFILE.get() {
        profile
            .lock()
            .expect("metal profile mutex poisoned")
            .entries
            .clear();
    }
}

pub fn metal_profile_snapshot() -> MetalProfileSnapshot {
    let mut snapshot = MetalProfileSnapshot::default();
    let Some(profile) = METAL_PROFILE.get() else {
        return snapshot;
    };
    let mut entries: Vec<_> = profile
        .lock()
        .expect("metal profile mutex poisoned")
        .entries
        .values()
        .cloned()
        .collect();
    entries.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.op.cmp(&rhs.op))
            .then_with(|| lhs.path.cmp(&rhs.path))
    });
    for entry in &entries {
        snapshot.total_calls += entry.calls;
        snapshot.total_ms += entry.total_ms;
        match entry.path.as_str() {
            "native" => {
                snapshot.native_calls += entry.calls;
                snapshot.native_ms += entry.total_ms;
            }
            "host" => {
                snapshot.host_calls += entry.calls;
                snapshot.host_ms += entry.total_ms;
            }
            _ => {}
        }
    }
    snapshot.entries = entries;
    snapshot
}

pub(crate) fn metal_profile_time<T, F>(op: &'static str, path: &'static str, f: F) -> T
where
    F: FnOnce() -> T,
{
    metal_profile_time_explicit(metal_profile_enabled(), op, path, f)
}

pub(crate) fn metal_profile_time_explicit<T, F>(
    enabled: bool,
    op: &'static str,
    path: &'static str,
    f: F,
) -> T
where
    F: FnOnce() -> T,
{
    if !enabled {
        return f();
    }
    let start = Instant::now();
    let result = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let profile = METAL_PROFILE.get_or_init(|| Mutex::new(MetalProfileAccumulator::default()));
    let mut profile = profile.lock().expect("metal profile mutex poisoned");
    let entry = profile
        .entries
        .entry((op.to_string(), path.to_string()))
        .or_insert_with(|| MetalProfileEntry {
            op: op.to_string(),
            path: path.to_string(),
            calls: 0,
            total_ms: 0.0,
            max_ms: 0.0,
        });
    entry.calls += 1;
    entry.total_ms += elapsed_ms;
    entry.max_ms = entry.max_ms.max(elapsed_ms);
    result
}

fn prefill_metal_profile_time<T, F>(
    options: Option<&PrefillFfiLaunchOptions>,
    op: &'static str,
    path: &'static str,
    f: F,
) -> T
where
    F: FnOnce() -> T,
{
    match options {
        Some(options) => metal_profile_time_explicit(options.metal_profile, op, path, f),
        None => metal_profile_time(op, path, f),
    }
}

pub fn ffi_profile_set_enabled(enabled: bool) {
    FFI_PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn ffi_profile_enabled() -> bool {
    FFI_PROFILE_ENABLED.load(Ordering::Relaxed)
}

pub fn ffi_profile_reset() {
    if let Some(profile) = FFI_PROFILE.get() {
        profile
            .lock()
            .expect("ffi profile mutex poisoned")
            .entries
            .clear();
    }
}

pub fn ffi_profile_snapshot() -> FfiProfileSnapshot {
    let mut snapshot = FfiProfileSnapshot::default();
    let Some(profile) = FFI_PROFILE.get() else {
        return snapshot;
    };
    let mut entries: Vec<_> = profile
        .lock()
        .expect("ffi profile mutex poisoned")
        .entries
        .values()
        .cloned()
        .collect();
    entries.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.op.cmp(&rhs.op))
    });
    for entry in &entries {
        snapshot.total_calls += entry.calls;
        snapshot.total_ms += entry.total_ms;
    }
    snapshot.entries = entries;
    snapshot
}

pub(crate) fn ffi_profile_time_result<T, F>(
    op: &'static str,
    ordinal: usize,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce() -> Result<T, GpuError>,
{
    ffi_profile_time_result_key(op.to_string(), ordinal, f)
}

pub(crate) fn ffi_profile_time_result_key<T, F>(
    op: String,
    ordinal: usize,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce() -> Result<T, GpuError>,
{
    if !ffi_profile_enabled() {
        return f();
    }

    gpu_hal::sync(ordinal)?;
    let start = Instant::now();
    let result = f();
    let sync_result = gpu_hal::sync(ordinal);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let profile = FFI_PROFILE.get_or_init(|| Mutex::new(FfiProfileAccumulator::default()));
    let mut profile = profile.lock().expect("ffi profile mutex poisoned");
    let entry = profile
        .entries
        .entry(op.clone())
        .or_insert_with(|| FfiProfileEntry {
            op,
            calls: 0,
            total_ms: 0.0,
            max_ms: 0.0,
        });
    entry.calls += 1;
    entry.total_ms += elapsed_ms;
    entry.max_ms = entry.max_ms.max(elapsed_ms);

    let value = result?;
    sync_result?;
    Ok(value)
}

#[no_mangle]
pub extern "C" fn supersonic_metal_profile_record(
    op: *const c_char,
    path: *const c_char,
    elapsed_ms: f64,
) {
    if !metal_profile_enabled() || op.is_null() || path.is_null() || !elapsed_ms.is_finite() {
        return;
    }
    record_metal_profile_ffi_sample(op, path, elapsed_ms);
}

#[no_mangle]
pub extern "C" fn supersonic_metal_profile_record_explicit(
    enabled: c_int,
    op: *const c_char,
    path: *const c_char,
    elapsed_ms: f64,
) {
    if enabled == 0 || op.is_null() || path.is_null() || !elapsed_ms.is_finite() {
        return;
    }
    record_metal_profile_ffi_sample(op, path, elapsed_ms);
}

fn record_metal_profile_ffi_sample(op: *const c_char, path: *const c_char, elapsed_ms: f64) {
    let op = unsafe { CStr::from_ptr(op) }.to_string_lossy().into_owned();
    let path = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();
    let profile = METAL_PROFILE.get_or_init(|| Mutex::new(MetalProfileAccumulator::default()));
    let mut profile = profile.lock().expect("metal profile mutex poisoned");
    let entry = profile
        .entries
        .entry((op.clone(), path.clone()))
        .or_insert_with(|| MetalProfileEntry {
            op,
            path,
            calls: 0,
            total_ms: 0.0,
            max_ms: 0.0,
        });
    entry.calls += 1;
    entry.total_ms += elapsed_ms.max(0.0);
    entry.max_ms = entry.max_ms.max(elapsed_ms.max(0.0));
}

fn metal_profile_host_time<T, F>(op: &'static str, f: F) -> Result<T, GpuError>
where
    F: FnOnce() -> Result<T, GpuError>,
{
    metal_native::flush_batch()?;
    metal_profile_time(op, "host", f)
}

fn prefill_metal_profile_host_time<T, F>(
    options: Option<&PrefillFfiLaunchOptions>,
    op: &'static str,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce() -> Result<T, GpuError>,
{
    metal_native::flush_batch()?;
    prefill_metal_profile_time(options, op, "host", f)
}

pub fn flush_metal_batch() -> Result<(), GpuError> {
    metal_native::flush_batch()
}

pub fn sync_metal_queue() -> Result<(), GpuError> {
    metal_native::queue_sync()
}

/// True when a Metal batch is currently open (i.e. ops will be accumulated
/// into the shared command buffer rather than committing one-by-one).
/// Always false on non-Metal builds.
pub fn metal_batch_is_active() -> bool {
    metal_native::batch_is_active()
}

pub fn set_metal_batch_label(label: &str) -> Result<(), GpuError> {
    metal_native::set_batch_label(label)
}

pub fn record_metal_profile_sample(op: &str, path: &str, elapsed_ms: f64) -> Result<(), GpuError> {
    let op = CString::new(op)
        .map_err(|_| GpuError::InvalidArg("metal profile op contains NUL byte".to_string()))?;
    let path = CString::new(path)
        .map_err(|_| GpuError::InvalidArg("metal profile path contains NUL byte".to_string()))?;
    supersonic_metal_profile_record(op.as_ptr(), path.as_ptr(), elapsed_ms);
    Ok(())
}

pub fn commit_metal_batch_current(label: &str) -> Result<(), GpuError> {
    metal_native::commit_batch_current(label)
}

pub fn metal_copy_d2d(src: *const c_void, dst: *mut c_void, bytes: usize) -> Result<(), GpuError> {
    metal_native::copy_d2d(src, dst, bytes)
}

pub fn metal_linear_decode_apply_parts_f32(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    q_scaled: &GpuBuffer,
    k_normed: &GpuBuffer,
    v_linear: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_native::linear_decode_apply_parts_f32(
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        q_scaled,
        k_normed,
        v_linear,
        a,
        b,
        dt_bias,
        a_log_exp,
        initial_state,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_prep_bf16_f32(
    key_dim: usize,
    val_dim: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    conv_pack: &GpuBuffer,
    q_bf16: &mut GpuBuffer,
    k_bf16: &mut GpuBuffer,
    v_bf16: &mut GpuBuffer,
    q_f32: &mut GpuBuffer,
    k_f32: &mut GpuBuffer,
    v_f32: &mut GpuBuffer,
    q_normed: &mut GpuBuffer,
    q_scaled: &mut GpuBuffer,
    k_normed: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_prep", "native", || {
        metal_native::qwen_linear_prep_bf16_f32(
            key_dim,
            val_dim,
            num_key_heads,
            key_head_dim,
            conv_pack,
            q_bf16,
            k_bf16,
            v_bf16,
            q_f32,
            k_f32,
            v_f32,
            q_normed,
            q_scaled,
            k_normed,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_prep_decode_apply_bf16_f32(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_pack: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_prep_decode_apply", "native", || {
        metal_native::qwen_linear_prep_decode_apply_bf16_f32(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_pack,
            a,
            b,
            dt_bias,
            a_log_exp,
            initial_state,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_decode_apply_inplace_bf16(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_pack: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    state: &mut GpuBuffer,
    attn_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_decode_apply_inplace", "native", || {
        metal_native::qwen_linear_decode_apply_inplace_bf16(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_pack,
            a,
            b,
            dt_bias,
            a_log_exp,
            state,
            attn_out,
        )
    })
}

pub fn metal_conv_state_update_bf16(
    channels: usize,
    state_len: usize,
    qkv: &GpuBuffer,
    state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_native::conv_state_update_bf16(channels, state_len, qkv, state)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_linear_conv_value_decay_update_bf16(
    conv_dim: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
    mixed_qkv: &GpuBuffer,
    state: &mut GpuBuffer,
    weights: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("linear_conv_value_decay_update", "native", || {
        metal_native::linear_conv_value_decay_update_bf16(
            conv_dim,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv,
            state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_projections_bf16(
    hidden_dim: usize,
    qkv_dim: usize,
    val_dim: usize,
    num_value_heads: usize,
    input: &GpuBuffer,
    qkv_weight: &GpuBuffer,
    z_weight: &GpuBuffer,
    a_weight: &GpuBuffer,
    b_weight: &GpuBuffer,
    qkv_out: &mut GpuBuffer,
    z_out: &mut GpuBuffer,
    a_out: &mut GpuBuffer,
    b_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_projections", "native", || {
        metal_native::qwen_linear_projections_bf16(
            hidden_dim,
            qkv_dim,
            val_dim,
            num_value_heads,
            input,
            qkv_weight,
            z_weight,
            a_weight,
            b_weight,
            qkv_out,
            z_out,
            a_out,
            b_out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_mlp_gate_up_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    input: &GpuBuffer,
    gate_weight: &GpuBuffer,
    up_weight: &GpuBuffer,
    gate_out: &mut GpuBuffer,
    up_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_mlp_gate_up", "native", || {
        metal_native::qwen_mlp_gate_up_bf16(
            hidden_dim,
            intermediate_dim,
            input,
            gate_weight,
            up_weight,
            gate_out,
            up_out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_mlp_gate_up_swiglu_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    input: &GpuBuffer,
    gate_weight: &GpuBuffer,
    up_weight: &GpuBuffer,
    mlp_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_mlp_gate_up_swiglu", "native", || {
        metal_native::qwen_mlp_gate_up_swiglu_bf16(
            hidden_dim,
            intermediate_dim,
            input,
            gate_weight,
            up_weight,
            mlp_out,
        )
    })
}

pub fn metal_full_attention_gate_bf16(
    total_elems: usize,
    attn_f32: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("full_attention_gate", "native", || {
        metal_native::full_attention_gate_bf16(total_elems, attn_f32, gate, out)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_mlp_down_residual_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    down_weight: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_mlp_down_residual", "native", || {
        metal_native::qwen_mlp_down_residual_bf16(
            hidden_dim,
            intermediate_dim,
            gate,
            up,
            down_weight,
            residual,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_out_residual_f32_bf16(
    hidden_dim: usize,
    num_rows: usize,
    row_dim: usize,
    eps: f32,
    attn: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out_proj: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_out_residual", "native", || {
        metal_native::qwen_linear_out_residual_f32_bf16(
            hidden_dim, num_rows, row_dim, eps, attn, gate, weight, out_proj, residual, out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_linear_out_residual_bf16_bf16(
    hidden_dim: usize,
    num_rows: usize,
    row_dim: usize,
    eps: f32,
    attn: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out_proj: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_linear_out_residual_bf16", "native", || {
        metal_native::qwen_linear_out_residual_bf16_bf16(
            hidden_dim, num_rows, row_dim, eps, attn, gate, weight, out_proj, residual, out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_rms_norm_rope_rows_bf16(
    n_rows: usize,
    n_cols: usize,
    rotary_dim: usize,
    eps: f32,
    pos_offset: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    cos: &GpuBuffer,
    sin: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("rms_norm_rope_rows", "native", || {
        metal_native::rms_norm_rope_rows_bf16(
            n_rows, n_cols, rotary_dim, eps, pos_offset, input, weight, cos, sin, out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_full_projections_bf16(
    hidden_dim: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    input: &GpuBuffer,
    q_weight: &GpuBuffer,
    k_weight: &GpuBuffer,
    v_weight: &GpuBuffer,
    q_out: &mut GpuBuffer,
    k_out: &mut GpuBuffer,
    v_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("qwen_full_projections", "native", || {
        metal_native::qwen_full_projections_bf16(
            hidden_dim, q_proj_dim, kv_dim, input, q_weight, k_weight, v_weight, q_out, k_out,
            v_out,
        )
    })
}

pub struct MetalBatchGuard {
    inner: Option<metal_native::MetalBatchGuard>,
}

impl MetalBatchGuard {
    pub fn begin() -> Result<Self, GpuError> {
        Ok(Self {
            inner: Some(metal_native::MetalBatchGuard::begin()?),
        })
    }

    pub fn finish(mut self) -> Result<(), GpuError> {
        if let Some(inner) = self.inner.take() {
            inner.finish()?;
        }
        Ok(())
    }
}

fn metal_force_host_rms_norm() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_RMS_NORM").is_some()
}

fn metal_force_host_rms_norm_gated() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_RMS_NORM_GATED").is_some()
}

fn metal_force_host_matmul() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_MATMUL").is_some()
}

fn metal_force_host_full_attention() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_FULL_ATTENTION").is_some()
}

fn metal_force_host_linear_conv_pack() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_LINEAR_CONV_PACK").is_some()
}

fn metal_force_host_element_add() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_ELEMENT_ADD").is_some()
}

fn metal_force_host_cast() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_CAST").is_some()
}

fn metal_force_host_mul_scalar() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_MUL_SCALAR").is_some()
}

fn metal_force_host_l2norm() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_L2NORM").is_some()
}

fn metal_force_host_transpose_shd_hsd() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_TRANSPOSE_SHD_HSD").is_some()
}

fn metal_force_host_split_qkv() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_SPLIT_QKV").is_some()
}

fn metal_force_host_split_qgate() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_SPLIT_QGATE").is_some()
}

fn ffi_error(msg: String) -> GpuError {
    match gpu_hal::current_backend() {
        Backend::Hip => GpuError::backend(Backend::Hip, msg),
        Backend::Cuda => GpuError::backend(Backend::Cuda, msg),
        Backend::Metal => GpuError::backend(Backend::Metal, msg),
    }
}

unsafe extern "C" {
    // ---- Existing bridge functions (from full_attention_bridge.cpp) ----

    fn supersonic_qwen35_hip_rms_norm(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        add_unit_offset: c_int,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_cast(
        input_dtype: c_int,
        output_dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        xs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    // ---- Prefill helper bridge functions (from prefill_helpers_bridge.cpp) ----

    fn supersonic_qwen35_hip_element_add(
        dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_argmax_bf16_rows(
        device_ordinal: usize,
        rows: usize,
        cols: usize,
        logits: *const c_void,
        out_index: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_apply_rope_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        half_rot: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        data: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_apply_rope_prefill_indirect(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        half_rot: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        pos_ids: *const c_int,
        data: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_lookahead_attention_scores(
        dtype: c_int,
        device_ordinal: usize,
        q_heads: usize,
        kv_heads: usize,
        lookahead_count: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        scores: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_pflash_cosine_score(
        dtype: c_int,
        device_ordinal: usize,
        n_pos: usize,
        kv_heads: usize,
        cap: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
        k_cache: *const c_void,
        scores: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_transpose_shd_hsd(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        h: usize,
        d: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_transpose_shd_hsd_pair(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        h: usize,
        d: usize,
        src_a: *const c_void,
        src_b: *const c_void,
        dst_a: *mut c_void,
        dst_b: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_sigmoid_mul(
        dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        data: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_transpose_shd_to_cache_bf16(
        device_ordinal: usize,
        s: usize,
        h: usize,
        d: usize,
        cache_len: usize,
        dst_pos: usize,
        src: *const c_void,
        cache: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_cast_transpose_gate_bf16(
        device_ordinal: usize,
        s: usize,
        heads: usize,
        head_dim: usize,
        attn_hsd: *const c_void,
        gate_shd: *const c_void,
        out_shd: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_compute_beta_g(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        nv: usize,
        b: *const c_void,
        a: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        beta: *mut c_void,
        g: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_compute_beta_g_ba_bf16(
        device_ordinal: usize,
        seq_len: usize,
        nv: usize,
        ba: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        beta: *mut c_void,
        g: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_project_ba_compute_beta_g_bf16(
        device_ordinal: usize,
        seq_len: usize,
        hidden_dim: usize,
        nv: usize,
        hidden: *const c_void,
        ba_weight: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        beta: *mut c_void,
        g: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_qgate(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        num_heads: usize,
        head_dim: usize,
        src: *const c_void,
        query_out: *mut c_void,
        gate_out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_qgate_norm_bf16(
        device_ordinal: usize,
        s: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
        src: *const c_void,
        norm_w: *const c_void,
        query_out: *mut c_void,
        gate_out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_qkv(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_qkv_bf16_to_f32(
        device_ordinal: usize,
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_kv_bf16(
        device_ordinal: usize,
        s: usize,
        kv_dim: usize,
        src: *const c_void,
        k: *mut c_void,
        v: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_norm_transpose_qkv_bf16(
        device_ordinal: usize,
        s: usize,
        nk: usize,
        nv: usize,
        khd: usize,
        vhd: usize,
        q_scale: f32,
        eps: f32,
        src: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_split_qkvz_bf16(
        device_ordinal: usize,
        s: usize,
        qkv_dim: usize,
        z_dim: usize,
        src: *const c_void,
        qkv: *mut c_void,
        z: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_repeat_interleave_heads(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        n_heads: usize,
        head_dim: usize,
        repeats: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_repeat_interleave_transpose_hsd(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        n_heads: usize,
        head_dim: usize,
        repeats: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_transpose_pad_conv(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        c: usize,
        pad: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_extract_conv_state(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        c: usize,
        kern_minus_1: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_prepare_conv_input_tail(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        c: usize,
        pad: usize,
        src: *const c_void,
        old_tail: *const c_void,
        conv_input: *mut c_void,
        new_tail: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_fused_rms_norm_linear(
        dtype: c_int,
        device_ordinal: usize,
        hidden_dim: usize,
        out_dim: usize,
        eps: f32,
        add_unit_offset: c_int,
        hidden: *const c_void,
        norm_weight: *const c_void,
        proj_weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    // Tiled BF16 matmul: out = lhs × rhs^T (rhs stored [n, k])
    fn supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    // FP8 dequant matmul: out = lhs (BF16) × dequant(rhs_fp8)^T
    fn supersonic_qwen35_4b_hip_matmul_fp8_dequant(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_fp8: *const c_void,
        scale: *const c_void,
        block_size: c_int,
        out: *mut c_void,
    ) -> c_int;

    // INT4 dequant matmul: out = lhs (BF16) × dequant(rhs_int4)^T
    fn supersonic_qwen35_4b_hip_matmul_int4_dequant(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_int4: *const c_void,
        scale: *const c_void,
        zero: *const c_void,
        awq_inv_scale: *const c_void,
        group_size: c_int,
        quant_type: c_int,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_int4: *const c_void,
        scale: *const c_void,
        zero: *const c_void,
        awq_inv_scale: *const c_void,
        group_size: c_int,
        quant_type: c_int,
        residual: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n_each: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_first: *const c_void,
        rhs_second: *const c_void,
        quant_type: c_int,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n_each: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_gate: *const c_void,
        rhs_up: *const c_void,
        quant_type: c_int,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_quantize_mmq_q8_1(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        k: c_int,
        lhs: *const c_void,
        quant_type: c_int,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        q8: *const c_void,
        rhs_q6: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k_residual_add(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        q8: *const c_void,
        rhs_q6: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_q6_k_m16_argmax(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_q6: *const c_void,
        block_best_vals: *mut c_void,
        block_best_indices: *mut c_void,
        out_indices: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_device_supports_wmma_i8(
        device_ordinal: usize,
        out_supported: *mut c_int,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    fn supersonic_qwen35_4b_hip_matmul_int8(
        dtype: c_int,
        device_ordinal: usize,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs: *const c_void,
        rhs_int8: *const c_void,
        scale: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    fn supersonic_qwen35_4b_hip_int8_outlier_add(
        dtype: c_int,
        device_ordinal: usize,
        rows: c_int,
        n: c_int,
        k: c_int,
        sub_cols: c_int,
        rhs_int8: *const c_void,
        scale: *const c_void,
        outlier_cols: *const c_void,
        outlier_vals: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_int4_sparse_outlier_add(
        dtype: c_int,
        device_ordinal: usize,
        rows: c_int,
        n: c_int,
        k: c_int,
        sub_cols: c_int,
        lhs: *const c_void,
        outlier_cols: *const c_void,
        outlier_delta: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    // BF16 → FP8 KV cache quantization
    fn supersonic_qwen35_4b_hip_quantize_kv_to_fp8(
        dtype: c_int,
        device_ordinal: usize,
        src: *const c_void,
        dst_fp8: *mut c_void,
        dst_scale: *mut c_void,
        num_kv_heads: c_int,
        seq_len: c_int,
        head_dim: c_int,
        max_T: c_int,
        pos_offset: c_int,
    ) -> c_int;

    // ---- Original prefill kernel declarations ----

    fn supersonic_qwen35_hip_embedding_lookup(
        dtype: c_int,
        index_dtype: c_int,
        device_ordinal: usize,
        token_count: usize,
        vocab_size: usize,
        hidden_size: usize,
        embeddings: *const c_void,
        indexes: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_batched_matmul(
        dtype: c_int,
        device_ordinal: usize,
        batch_rank: c_int,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs_batch_dims: *const c_int,
        rhs_batch_dims: *const c_int,
        out_batch_dims: *const c_int,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_full_attention_prefill(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        q_heads: usize,
        kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        num_kv_groups: usize,
        scale: f32,
        seqlen_offset: usize,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_full_attention_tree_prefill(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        q_heads: usize,
        kv_heads: usize,
        tree_len: usize,
        prefix_len: usize,
        head_dim: usize,
        num_kv_groups: usize,
        scale: f32,
        query: *const c_void,
        prefix_key: *const c_void,
        prefix_value: *const c_void,
        tree_key: *const c_void,
        tree_value: *const c_void,
        visibility: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_full_attention_tree_prefill_strided(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        q_heads: usize,
        kv_heads: usize,
        tree_len: usize,
        prefix_len: usize,
        prefix_stride: usize,
        head_dim: usize,
        num_kv_groups: usize,
        scale: f32,
        query: *const c_void,
        prefix_key: *const c_void,
        prefix_value: *const c_void,
        tree_key: *const c_void,
        tree_value: *const c_void,
        visibility: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_full_attention_decode_flat(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        q_heads: usize,
        kv_heads: usize,
        kv_len: usize,
        head_dim: usize,
        num_kv_groups: usize,
        scale: f32,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    fn supersonic_qwen35_hip_full_attention_decode_flat_strided(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        q_heads: usize,
        kv_heads: usize,
        kv_len: usize,
        kv_stride: usize,
        head_dim: usize,
        num_kv_groups: usize,
        scale: f32,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_linear_prefill_conv_pack(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        conv_dim: usize,
        total_len: usize,
        seq_len: usize,
        kernel_size: usize,
        mixed_qkv: *const c_void,
        weights: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_linear_tree_conv_pack(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        conv_dim: usize,
        total_len: usize,
        tree_len: usize,
        kernel_size: usize,
        mixed_qkv: *const c_void,
        weights: *const c_void,
        parent_ids: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_linear_tree_conv_pack_indexed(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        conv_dim: usize,
        total_len: usize,
        tree_len: usize,
        kernel_size: usize,
        source_stride: usize,
        mixed_qkv: *const c_void,
        weights: *const c_void,
        source_cols: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_linear_decode_prepare(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        state_len: usize,
        kernel_size: usize,
        head_repeat: usize,
        mixed_qkv: *const c_void,
        prev_conv_state: *const c_void,
        weights: *const c_void,
        a_beta_raw: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_linear_decode_apply(
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        packed: *const c_void,
        initial_state: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_linear_decode_prepare(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        state_len: usize,
        kernel_size: usize,
        head_repeat: usize,
        mixed_qkv: *const c_void,
        prev_conv_state: *const c_void,
        weights: *const c_void,
        a_beta_raw: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_linear_decode_apply(
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        packed: *const c_void,
        initial_state: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_linear_stateful_conv_value_decay(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: usize,
        conv_dim: usize,
        seq_len: usize,
        state_len: usize,
        kernel_size: usize,
        num_heads: usize,
        mixed_qkv: *const c_void,
        prev_state: *const c_void,
        weights: *const c_void,
        a: *const c_void,
        dt_bias: *const c_void,
        a_log_exp: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_prefill(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_prefill_capture(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_prefill_capture_bf16_trace(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_prefill_capture_q8_trace(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_prefill_capture_q8_trace_attn(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        recurrent_state: *mut c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        attn_output: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        parent_ids: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_bf16_trace(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        parent_ids: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_q8_trace(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        parent_ids: *const c_void,
        out: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_q8_trace_attn(
        dtype: c_int,
        device_ordinal: usize,
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state: *const c_void,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        parent_ids: *const c_void,
        attn_output: *mut c_void,
        state_trace: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_rollback(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        chunk_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_rollback_bf16_trace(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        chunk_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_rollback_q8_trace(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        chunk_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_tree_rollback(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        tree_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        accepted_indices: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_tree_rollback_bf16_trace(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        tree_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        accepted_indices: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_apply_tree_rollback_q8_trace(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        conv_state_len: usize,
        conv_input_len: usize,
        tree_len: usize,
        commit_len: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_input: *const c_void,
        accepted_indices: *const c_void,
        conv_state: *mut c_void,
        recurrent_trace: *const c_void,
        recurrent_state: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_fill_conv_tail(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        pad: usize,
        total_len: usize,
        tail: *const c_void,
        conv_input: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_dflash_extract_recurrent_attn(
        device_ordinal: usize,
        num_v_heads: usize,
        chunk_len: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        delta_out: *const c_void,
        recurrent_state: *mut c_void,
        attn_output: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_l2norm(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_swiglu_mul(
        dtype: c_int,
        device_ordinal: usize,
        elem_count: usize,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_swiglu_mul_split(
        dtype: c_int,
        device_ordinal: usize,
        rows: usize,
        cols: usize,
        gate_up: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_rms_norm_gated(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        hidden: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_rms_norm_gated_sfirst_bf16(
        device_ordinal: usize,
        s: usize,
        nv: usize,
        vhd: usize,
        eps: f32,
        hidden_hsd: *const c_void,
        gate_sfirst: *const c_void,
        weight: *const c_void,
        out_sfirst: *mut c_void,
    ) -> c_int;

    fn supersonic_qwen35_hip_mul_scalar(
        dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        scalar: f32,
        xs: *const c_void,
        out: *mut c_void,
    ) -> c_int;
}

// --- Safe wrappers ---

/// Embedding lookup: token IDs → hidden states.
/// indexes: U32 device buffer of token IDs
/// out: [token_count, hidden_size] in dtype
pub fn embedding_lookup(
    ordinal: usize,
    dtype: ScalarType,
    token_count: usize,
    vocab_size: usize,
    hidden_size: usize,
    embeddings: &GpuBuffer,
    indexes: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if embeddings.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16 && !metal_native::disabled_by_env() {
            let result = metal_profile_time("embedding_lookup", "native", || {
                metal_native::embedding_lookup_bf16(
                    token_count,
                    vocab_size,
                    hidden_size,
                    embeddings,
                    indexes,
                    out,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("embedding_lookup", || {
            metal_host::embedding_lookup(
                token_count,
                vocab_size,
                hidden_size,
                embeddings,
                indexes,
                out,
            )
        });
    }
    ffi_profile_time_result("qwen.embedding_lookup", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_embedding_lookup(
                dtype.kernel_dtype_code(),
                1, // index_dtype=1 → uint32
                ordinal,
                token_count,
                vocab_size,
                hidden_size,
                embeddings.as_ptr(),
                indexes.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("embedding_lookup failed: {status}")));
        }
        Ok(())
    })
}

/// Batched matrix multiply: lhs [batch, m, k] × rhs [batch, k, n] → out [batch, m, n].
/// For weight projections: lhs = activations, rhs = weights^T (or transposed layout).
pub fn batched_matmul(
    ordinal: usize,
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        return metal_profile_host_time("batched_matmul", || {
            metal_host::batched_matmul(dtype, batch_elems, m, n, k, lhs, rhs, out)
        });
    }
    // Simple rank-1 batch (no broadcasting)
    let batch_dims = [batch_elems as c_int];
    ffi_profile_time_result("qwen.batched_matmul", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_batched_matmul(
                dtype.kernel_dtype_code(),
                ordinal,
                1, // batch_rank
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                batch_dims.as_ptr(),
                batch_dims.as_ptr(),
                batch_dims.as_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("batched_matmul failed: {status}")));
        }
        Ok(())
    })
}

/// Full causal attention for prefill.
pub fn full_attention_prefill(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && batch_size == 1
            && !metal_native::disabled_by_env()
            && !metal_force_host_full_attention()
        {
            let result = metal_profile_time("full_attention_prefill", "native", || {
                metal_native::full_attention_prefill_bf16_f32(
                    q_heads,
                    kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    seqlen_offset,
                    query,
                    key,
                    value,
                    out,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("full_attention_prefill", || {
            metal_host::full_attention_prefill(
                dtype,
                batch_size,
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                scale,
                seqlen_offset,
                query,
                key,
                value,
                out,
            )
        });
    }
    let num_kv_groups = q_heads / kv_heads;
    ffi_profile_time_result("qwen.full_attention_prefill", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_full_attention_prefill(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                num_kv_groups,
                scale,
                seqlen_offset,
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "full_attention_prefill failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Full attention over a tree-structured verify batch.
///
/// Queries and tree K/V use `[batch, heads, tree_len, head_dim]` layout.
/// Prefix K/V use `[batch, kv_heads, prefix_len, head_dim]`. Every tree query
/// attends all prefix positions and only the tree positions marked nonzero in
/// the row-major `visibility[tree_len, tree_len]` mask.
#[allow(clippy::too_many_arguments)]
pub fn full_attention_tree_prefill(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    tree_len: usize,
    prefix_len: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    prefix_key: &GpuBuffer,
    prefix_value: &GpuBuffer,
    tree_key: &GpuBuffer,
    tree_value: &GpuBuffer,
    visibility: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "full_attention_tree_prefill is not implemented for Metal".into(),
        ));
    }
    let num_kv_groups = q_heads / kv_heads;
    ffi_profile_time_result("qwen.full_attention_tree_prefill", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_full_attention_tree_prefill(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                q_heads,
                kv_heads,
                tree_len,
                prefix_len,
                head_dim,
                num_kv_groups,
                scale,
                query.as_ptr(),
                prefix_key.as_ptr(),
                prefix_value.as_ptr(),
                tree_key.as_ptr(),
                tree_value.as_ptr(),
                visibility.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "full_attention_tree_prefill failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Full tree attention where prefix K/V are laid out as
/// `[batch, kv_heads, prefix_stride, head_dim]` and only the first
/// `prefix_len` rows per head are live.
#[allow(clippy::too_many_arguments)]
pub fn full_attention_tree_prefill_strided_raw(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    tree_len: usize,
    prefix_len: usize,
    prefix_stride: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    prefix_key: *const c_void,
    prefix_value: *const c_void,
    tree_key: &GpuBuffer,
    tree_value: &GpuBuffer,
    visibility: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "full_attention_tree_prefill_strided_raw is not implemented for Metal".into(),
        ));
    }
    if prefix_key.is_null() || prefix_value.is_null() {
        return Err(ffi_error(
            "full_attention_tree_prefill_strided_raw got null prefix K/V".into(),
        ));
    }
    if prefix_stride < prefix_len {
        return Err(ffi_error(format!(
            "full_attention_tree_prefill_strided_raw prefix_stride {prefix_stride} < prefix_len {prefix_len}"
        )));
    }
    let num_kv_groups = q_heads / kv_heads;
    ffi_profile_time_result("qwen.full_attention_tree_prefill_strided", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_full_attention_tree_prefill_strided(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                q_heads,
                kv_heads,
                tree_len,
                prefix_len,
                prefix_stride,
                head_dim,
                num_kv_groups,
                scale,
                query.as_ptr(),
                prefix_key,
                prefix_value,
                tree_key.as_ptr(),
                tree_value.as_ptr(),
                visibility.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "full_attention_tree_prefill_strided failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_full_attention_prefill_strided_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    kv_stride: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_full_attention_prefill_strided_bf16_f32 requires a Metal output buffer".into(),
        ));
    }
    metal_profile_time("full_attention_prefill_strided", "native", || {
        metal_native::full_attention_prefill_strided_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            kv_stride,
            head_dim,
            scale,
            seqlen_offset,
            query,
            key,
            value,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_full_attention_prefill_tmajor_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key_ptr: *const c_void,
    value_ptr: *const c_void,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_full_attention_prefill_tmajor_bf16_f32 requires a Metal output buffer".into(),
        ));
    }
    metal_profile_time("full_attention_prefill_tmajor", "native", || unsafe {
        metal_native::full_attention_prefill_tmajor_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
            seqlen_offset,
            query,
            key_ptr,
            value_ptr,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_full_attention_prefill_tmajor_bf16_f32_with_options(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key_ptr: *const c_void,
    value_ptr: *const c_void,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_full_attention_prefill_tmajor_bf16_f32_with_options requires a Metal output buffer"
                .into(),
        ));
    }
    prefill_metal_profile_time(
        Some(options),
        "full_attention_prefill_tmajor",
        "native",
        || unsafe {
            metal_native::full_attention_prefill_tmajor_bf16_f32(
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                scale,
                seqlen_offset,
                query,
                key_ptr,
                value_ptr,
                out,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_full_attention_prefill_tmajor_vec_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key_ptr: *const c_void,
    value_ptr: *const c_void,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_full_attention_prefill_tmajor_vec_bf16_f32 requires a Metal output buffer"
                .into(),
        ));
    }
    metal_profile_time("full_attention_prefill_tmajor_vec", "native", || unsafe {
        metal_native::full_attention_prefill_tmajor_vec_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
            seqlen_offset,
            query,
            key_ptr,
            value_ptr,
            out,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_full_attention_prefill_tmajor_vec_bf16_f32_with_options(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key_ptr: *const c_void,
    value_ptr: *const c_void,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_full_attention_prefill_tmajor_vec_bf16_f32_with_options requires a Metal output buffer"
                .into(),
        ));
    }
    prefill_metal_profile_time(
        Some(options),
        "full_attention_prefill_tmajor_vec",
        "native",
        || unsafe {
            metal_native::full_attention_prefill_tmajor_vec_bf16_f32(
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                scale,
                seqlen_offset,
                query,
                key_ptr,
                value_ptr,
                out,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn metal_full_attention_decode_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    kv_stride: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("full_attention_decode", "native", || {
        metal_native::full_attention_decode_bf16_f32(
            q_heads, kv_heads, kv_len, kv_stride, head_dim, scale, query, key, value, out,
        )
    })
}

/// Decode-only full attention for q_len=1. Writes BF16/FP16 flat [batch, q_heads * head_dim].
pub fn full_attention_decode_flat(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let num_kv_groups = q_heads / kv_heads;
    let status = unsafe {
        supersonic_qwen35_hip_full_attention_decode_flat(
            dtype.kernel_dtype_code(),
            ordinal,
            batch_size,
            q_heads,
            kv_heads,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            query.as_ptr(),
            key.as_ptr(),
            value.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "full_attention_decode_flat failed: {status}"
        )));
    }
    Ok(())
}

/// Decode-only full attention for q_len=1 over a KV cache whose physical
/// capacity stride can exceed the live `kv_len`. Writes BF16/FP16 flat
/// `[batch, q_heads * head_dim]`.
pub fn full_attention_decode_flat_strided(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    kv_stride: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    #[cfg(not(supersonic_backend_cuda))]
    {
        return Err(ffi_error(
            "full_attention_decode_flat_strided requires the CUDA backend".to_string(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        if gpu_hal::current_backend() != Backend::Cuda {
            return Err(ffi_error(
                "full_attention_decode_flat_strided requires the CUDA backend".to_string(),
            ));
        }

        let num_kv_groups = q_heads / kv_heads;
        let status = unsafe {
            supersonic_qwen35_hip_full_attention_decode_flat_strided(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                q_heads,
                kv_heads,
                kv_len,
                kv_stride,
                head_dim,
                num_kv_groups,
                scale,
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "full_attention_decode_flat_strided failed: {status}"
            )));
        }
        Ok(())
    }
}

/// Linear attention conv1d + SiLU for prefill.
pub fn linear_prefill_conv_pack(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
    mixed_qkv: &GpuBuffer,
    weights: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && batch_size == 1
            && !metal_native::disabled_by_env()
            && !metal_force_host_linear_conv_pack()
        {
            let result = metal_profile_time("linear_prefill_conv_pack", "native", || {
                metal_native::linear_prefill_conv_pack_bf16(
                    conv_dim,
                    total_len,
                    seq_len,
                    kernel_size,
                    mixed_qkv,
                    weights,
                    out,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("linear_prefill_conv_pack", || {
            metal_host::linear_prefill_conv_pack(
                dtype,
                batch_size,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                mixed_qkv,
                weights,
                out,
            )
        });
    }
    ffi_profile_time_result("qwen.linear_prefill_conv_pack", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_linear_prefill_conv_pack(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                mixed_qkv.as_ptr(),
                weights.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "linear_prefill_conv_pack failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn linear_tree_conv_pack(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    tree_len: usize,
    kernel_size: usize,
    mixed_qkv: &GpuBuffer,
    weights: &GpuBuffer,
    parent_ids: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "linear_tree_conv_pack is not implemented for Metal".into(),
        ));
    }
    if parent_ids.dtype() != ScalarType::U32 || parent_ids.elem_count() < tree_len {
        return Err(GpuError::InvalidArg(format!(
            "linear_tree_conv_pack parent_ids must be U32[{tree_len}], got {:?}[{}]",
            parent_ids.dtype(),
            parent_ids.elem_count()
        )));
    }
    ffi_profile_time_result("qwen.linear_tree_conv_pack", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_linear_tree_conv_pack(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                conv_dim,
                total_len,
                tree_len,
                kernel_size,
                mixed_qkv.as_ptr(),
                weights.as_ptr(),
                parent_ids.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("linear_tree_conv_pack failed: {status}")));
        }
        Ok(())
    })
}

pub fn linear_tree_conv_pack_indexed(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    tree_len: usize,
    kernel_size: usize,
    source_stride: usize,
    mixed_qkv: &GpuBuffer,
    weights: &GpuBuffer,
    source_cols: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "linear_tree_conv_pack_indexed is not implemented for Metal".into(),
        ));
    }
    if source_stride < kernel_size {
        return Err(GpuError::InvalidArg(format!(
            "linear_tree_conv_pack_indexed source_stride {source_stride} < kernel_size {kernel_size}"
        )));
    }
    if source_cols.dtype() != ScalarType::U32 || source_cols.elem_count() < tree_len * source_stride
    {
        return Err(GpuError::InvalidArg(format!(
            "linear_tree_conv_pack_indexed source_cols must be U32[tree_len * source_stride], got {:?}[{}] for {tree_len} * {source_stride}",
            source_cols.dtype(),
            source_cols.elem_count()
        )));
    }
    ffi_profile_time_result("qwen.linear_tree_conv_pack_indexed", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_linear_tree_conv_pack_indexed(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_size,
                conv_dim,
                total_len,
                tree_len,
                kernel_size,
                source_stride,
                mixed_qkv.as_ptr(),
                weights.as_ptr(),
                source_cols.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "linear_tree_conv_pack_indexed failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Linear attention single-step decode prep.
pub fn linear_decode_prepare(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    state_len: usize,
    kernel_size: usize,
    head_repeat: usize,
    mixed_qkv: &GpuBuffer,
    prev_conv_state: &GpuBuffer,
    weights: &GpuBuffer,
    a_beta_raw: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_qwen35_hip_linear_decode_prepare(
            dtype.kernel_dtype_code(),
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_len,
            kernel_size,
            head_repeat,
            mixed_qkv.as_ptr(),
            prev_conv_state.as_ptr(),
            weights.as_ptr(),
            a_beta_raw.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!("linear_decode_prepare failed: {status}")));
    }
    Ok(())
}

/// Linear attention single-step recurrent apply.
pub fn linear_decode_apply(
    ordinal: usize,
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    packed: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_qwen35_hip_linear_decode_apply(
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            packed.as_ptr(),
            initial_state.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!("linear_decode_apply failed: {status}")));
    }
    Ok(())
}

/// 4B linear attention single-step decode prep.
pub fn linear_decode_prepare_4b(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    state_len: usize,
    kernel_size: usize,
    head_repeat: usize,
    mixed_qkv: &GpuBuffer,
    prev_conv_state: &GpuBuffer,
    weights: &GpuBuffer,
    a_beta_raw: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_qwen35_4b_hip_linear_decode_prepare(
            dtype.kernel_dtype_code(),
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_len,
            kernel_size,
            head_repeat,
            mixed_qkv.as_ptr(),
            prev_conv_state.as_ptr(),
            weights.as_ptr(),
            a_beta_raw.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "linear_decode_prepare_4b failed: {status}"
        )));
    }
    Ok(())
}

/// 4B linear attention single-step recurrent apply.
pub fn linear_decode_apply_4b(
    ordinal: usize,
    batch_size: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    packed: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        return metal_profile_host_time("linear_decode_apply_4b", || {
            metal_host::linear_decode_apply(
                batch_size,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                packed,
                initial_state,
                out,
            )
        });
    }
    let status = unsafe {
        supersonic_qwen35_4b_hip_linear_decode_apply(
            ordinal,
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            packed.as_ptr(),
            initial_state.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "linear_decode_apply_4b failed: {status}"
        )));
    }
    Ok(())
}

pub fn linear_stateful_conv_value_decay_4b(
    ordinal: usize,
    dtype: ScalarType,
    batch_size: usize,
    conv_dim: usize,
    seq_len: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
    mixed_qkv: &GpuBuffer,
    prev_state: &GpuBuffer,
    weights: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && batch_size == 1
            && seq_len == 1
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_NATIVE_LINEAR_CONV_VALUE_DECAY").is_none()
        {
            return metal_profile_time("linear_stateful_conv_value_decay_4b", "native", || {
                metal_native::linear_conv_value_decay_bf16(
                    conv_dim,
                    state_len,
                    kernel_size,
                    num_heads,
                    mixed_qkv,
                    prev_state,
                    weights,
                    a,
                    dt_bias,
                    a_log_exp,
                    out,
                )
            });
        }
        return metal_profile_host_time("linear_stateful_conv_value_decay_4b", || {
            metal_host::linear_stateful_conv_value_decay(
                dtype,
                batch_size,
                conv_dim,
                seq_len,
                state_len,
                kernel_size,
                num_heads,
                mixed_qkv,
                prev_state,
                weights,
                a,
                dt_bias,
                a_log_exp,
                out,
            )
        });
    }
    let status = unsafe {
        supersonic_qwen35_4b_hip_linear_stateful_conv_value_decay(
            dtype.kernel_dtype_code(),
            ordinal,
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv.as_ptr(),
            prev_state.as_ptr(),
            weights.as_ptr(),
            a.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "linear_stateful_conv_value_decay_4b failed: {status}"
        )));
    }
    Ok(())
}

/// Delta recurrent state accumulation for linear attention prefill.
pub fn delta_recurrent_prefill(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::F32 && !metal_native::disabled_by_env() {
            let result = metal_profile_time("delta_recurrent_prefill", "native", || {
                metal_native::delta_recurrent_prefill_f32(
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state,
                    query,
                    key,
                    value,
                    beta,
                    g,
                    out,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("delta_recurrent_prefill", || {
            metal_host::delta_recurrent_prefill(
                dtype,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state,
                query,
                key,
                value,
                beta,
                g,
                out,
            )
        });
    }
    ffi_profile_time_result("qwen.delta_recurrent_prefill", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_delta_recurrent_prefill(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state.as_ptr(),
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                beta.as_ptr(),
                g.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "delta_recurrent_prefill failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Delta recurrent state accumulation for append verification, with
/// per-token recurrent states captured as
/// `[batch_heads, seq_len, k_head_dim, v_head_dim]`.
pub fn delta_recurrent_prefill_capture(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_prefill_capture is not implemented for Metal".into(),
        ));
    }
    ffi_profile_time_result("qwen.delta_recurrent_prefill_capture", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_delta_recurrent_prefill_capture(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state.as_ptr(),
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                beta.as_ptr(),
                g.as_ptr(),
                out.as_mut_ptr(),
                state_trace.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "delta_recurrent_prefill_capture failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Delta recurrent append verification with per-token recurrent states stored
/// as BF16 while accumulation and the final recurrent state remain F32.
#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_prefill_capture_bf16_trace(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_prefill_capture_bf16_trace is not implemented for Metal".into(),
        ));
    }
    if dtype != ScalarType::F32
        || state_trace.dtype() != ScalarType::BF16
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Err(ffi_error(
            "delta_recurrent_prefill_capture_bf16_trace requires F32 state, BF16 trace, k/v head dim 128"
                .into(),
        ));
    }
    ffi_profile_time_result(
        "qwen.delta_recurrent_prefill_capture_bf16_trace",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_delta_recurrent_prefill_capture_bf16_trace(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state.as_ptr(),
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    beta.as_ptr(),
                    g.as_ptr(),
                    out.as_mut_ptr(),
                    state_trace.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "delta_recurrent_prefill_capture_bf16_trace failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_prefill_capture_q8_trace(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_prefill_capture_q8_trace is not implemented for Metal".into(),
        ));
    }
    if dtype != ScalarType::F32
        || state_trace.dtype() != ScalarType::U8
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Err(ffi_error(
            "delta_recurrent_prefill_capture_q8_trace requires F32 state, U8 trace, k/v head dim 128"
                .into(),
        ));
    }
    ffi_profile_time_result(
        "qwen.delta_recurrent_prefill_capture_q8_trace",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_delta_recurrent_prefill_capture_q8_trace(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state.as_ptr(),
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    beta.as_ptr(),
                    g.as_ptr(),
                    out.as_mut_ptr(),
                    state_trace.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "delta_recurrent_prefill_capture_q8_trace failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_prefill_capture_q8_trace_attn(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    recurrent_state: &mut GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    attn_output: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<bool, GpuError> {
    if std::env::var_os("SUPERSONIC_DFLASH_DISABLE_APPEND_RECURRENT_WARP32").is_some() {
        return Ok(false);
    }
    if recurrent_state.backend() == Backend::Metal {
        return Ok(false);
    }
    if dtype != ScalarType::F32
        || recurrent_state.dtype() != ScalarType::F32
        || attn_output.dtype() != ScalarType::BF16
        || state_trace.dtype() != ScalarType::U8
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Ok(false);
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.delta_recurrent_prefill_capture_q8_trace_attn[bh={} s={} k={} v={}]",
            batch_heads, seq_len, k_head_dim, v_head_dim
        )
    } else {
        "qwen.delta_recurrent_prefill_capture_q8_trace_attn".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_delta_recurrent_prefill_capture_q8_trace_attn(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                recurrent_state.as_mut_ptr(),
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                beta.as_ptr(),
                g.as_ptr(),
                attn_output.as_mut_ptr(),
                state_trace.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "delta_recurrent_prefill_capture_q8_trace_attn failed: {status}"
            )));
        }
        Ok(())
    })?;
    Ok(true)
}

/// Parent-aware DeltaNet accumulation for tree-structured DFlash verify.
///
/// `parent_ids[t]` is `-1` for a node whose parent is the already-committed
/// root state, otherwise it indexes a previous node in `[0, seq_len)`.
/// Currently implemented for HIP F32 K/V head dim 128, matching the optimized
/// append-capture kernel used by Qwen3.6 DFlash.
pub fn delta_recurrent_tree_prefill_capture(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    parent_ids: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture is not implemented for Metal".into(),
        ));
    }
    if dtype != ScalarType::F32 || k_head_dim != 128 || v_head_dim != 128 {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture requires F32 k_head_dim=128 v_head_dim=128"
                .into(),
        ));
    }
    ffi_profile_time_result("qwen.delta_recurrent_tree_prefill_capture", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state.as_ptr(),
                query.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
                beta.as_ptr(),
                g.as_ptr(),
                parent_ids.as_ptr(),
                out.as_mut_ptr(),
                state_trace.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "delta_recurrent_tree_prefill_capture failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_tree_prefill_capture_bf16_trace(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    parent_ids: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_bf16_trace is not implemented for Metal".into(),
        ));
    }
    if dtype != ScalarType::F32
        || state_trace.dtype() != ScalarType::BF16
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_bf16_trace requires F32 state, BF16 trace, k/v head dim 128"
                .into(),
        ));
    }
    ffi_profile_time_result(
        "qwen.delta_recurrent_tree_prefill_capture_bf16_trace",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_bf16_trace(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state.as_ptr(),
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    beta.as_ptr(),
                    g.as_ptr(),
                    parent_ids.as_ptr(),
                    out.as_mut_ptr(),
                    state_trace.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "delta_recurrent_tree_prefill_capture_bf16_trace failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_tree_prefill_capture_q8_trace(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    parent_ids: &GpuBuffer,
    out: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_q8_trace is not implemented for Metal".into(),
        ));
    }
    if dtype != ScalarType::F32
        || state_trace.dtype() != ScalarType::U8
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_q8_trace requires F32 state, U8 trace, k/v head dim 128"
                .into(),
        ));
    }
    ffi_profile_time_result(
        "qwen.delta_recurrent_tree_prefill_capture_q8_trace",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_q8_trace(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state.as_ptr(),
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    beta.as_ptr(),
                    g.as_ptr(),
                    parent_ids.as_ptr(),
                    out.as_mut_ptr(),
                    state_trace.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "delta_recurrent_tree_prefill_capture_q8_trace failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrent_tree_prefill_capture_q8_trace_attn(
    ordinal: usize,
    dtype: ScalarType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    parent_ids: &GpuBuffer,
    attn_output: &mut GpuBuffer,
    state_trace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if attn_output.backend() == Backend::Metal {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_q8_trace_attn is not implemented for Metal"
                .into(),
        ));
    }
    if dtype != ScalarType::F32
        || attn_output.dtype() != ScalarType::BF16
        || state_trace.dtype() != ScalarType::U8
        || k_head_dim != 128
        || v_head_dim != 128
    {
        return Err(ffi_error(
            "delta_recurrent_tree_prefill_capture_q8_trace_attn requires F32 state, BF16 attention output, U8 trace, k/v head dim 128"
                .into(),
        ));
    }
    ffi_profile_time_result(
        "qwen.delta_recurrent_tree_prefill_capture_q8_trace_attn",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_delta_recurrent_tree_prefill_capture_q8_trace_attn(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    batch_heads,
                    seq_len,
                    k_head_dim,
                    v_head_dim,
                    initial_state.as_ptr(),
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    beta.as_ptr(),
                    g.as_ptr(),
                    parent_ids.as_ptr(),
                    attn_output.as_mut_ptr(),
                    state_trace.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "delta_recurrent_tree_prefill_capture_q8_trace_attn failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

pub fn dflash_apply_rollback(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    chunk_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_rollback is not implemented for Metal".into(),
        ));
    }
    ffi_profile_time_result("qwen.dflash_apply_rollback", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_apply_rollback(
                conv_dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                conv_state_len,
                conv_input_len,
                chunk_len,
                commit_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_input.as_ptr(),
                conv_state.as_mut_ptr(),
                recurrent_trace.as_ptr(),
                recurrent_state.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("dflash_apply_rollback failed: {status}")));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_apply_rollback_bf16_trace(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    chunk_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_rollback_bf16_trace is not implemented for Metal".into(),
        ));
    }
    if recurrent_trace.dtype() != ScalarType::BF16 {
        return Err(ffi_error(
            "dflash_apply_rollback_bf16_trace requires BF16 recurrent_trace".into(),
        ));
    }
    ffi_profile_time_result("qwen.dflash_apply_rollback_bf16_trace", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_apply_rollback_bf16_trace(
                conv_dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                conv_state_len,
                conv_input_len,
                chunk_len,
                commit_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_input.as_ptr(),
                conv_state.as_mut_ptr(),
                recurrent_trace.as_ptr(),
                recurrent_state.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "dflash_apply_rollback_bf16_trace failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_apply_rollback_q8_trace(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    chunk_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_rollback_q8_trace is not implemented for Metal".into(),
        ));
    }
    if recurrent_trace.dtype() != ScalarType::U8 {
        return Err(ffi_error(
            "dflash_apply_rollback_q8_trace requires U8 recurrent_trace".into(),
        ));
    }
    ffi_profile_time_result("qwen.dflash_apply_rollback_q8_trace", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_apply_rollback_q8_trace(
                conv_dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                conv_state_len,
                conv_input_len,
                chunk_len,
                commit_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_input.as_ptr(),
                conv_state.as_mut_ptr(),
                recurrent_trace.as_ptr(),
                recurrent_state.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "dflash_apply_rollback_q8_trace failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_apply_tree_rollback(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    tree_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    accepted_indices: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_tree_rollback is not implemented for Metal".into(),
        ));
    }
    if accepted_indices.dtype() != ScalarType::U32 || accepted_indices.elem_count() < commit_len {
        return Err(GpuError::InvalidArg(format!(
            "dflash_apply_tree_rollback accepted_indices must be U32[{commit_len}], got {:?}[{}]",
            accepted_indices.dtype(),
            accepted_indices.elem_count()
        )));
    }
    ffi_profile_time_result("qwen.dflash_apply_tree_rollback", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_apply_tree_rollback(
                conv_dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                conv_state_len,
                conv_input_len,
                tree_len,
                commit_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_input.as_ptr(),
                accepted_indices.as_ptr(),
                conv_state.as_mut_ptr(),
                recurrent_trace.as_ptr(),
                recurrent_state.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "dflash_apply_tree_rollback failed: {status}"
            )));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_apply_tree_rollback_bf16_trace(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    tree_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    accepted_indices: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_tree_rollback_bf16_trace is not implemented for Metal".into(),
        ));
    }
    if recurrent_trace.dtype() != ScalarType::BF16 {
        return Err(ffi_error(
            "dflash_apply_tree_rollback_bf16_trace requires BF16 recurrent_trace".into(),
        ));
    }
    if accepted_indices.dtype() != ScalarType::U32 || accepted_indices.elem_count() < commit_len {
        return Err(GpuError::InvalidArg(format!(
            "dflash_apply_tree_rollback_bf16_trace accepted_indices must be U32[{commit_len}], got {:?}[{}]",
            accepted_indices.dtype(),
            accepted_indices.elem_count()
        )));
    }
    ffi_profile_time_result(
        "qwen.dflash_apply_tree_rollback_bf16_trace",
        ordinal,
        || {
            let status = unsafe {
                supersonic_qwen35_hip_dflash_apply_tree_rollback_bf16_trace(
                    conv_dtype.kernel_dtype_code(),
                    ordinal,
                    qkv_dim,
                    conv_state_len,
                    conv_input_len,
                    tree_len,
                    commit_len,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    conv_input.as_ptr(),
                    accepted_indices.as_ptr(),
                    conv_state.as_mut_ptr(),
                    recurrent_trace.as_ptr(),
                    recurrent_state.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "dflash_apply_tree_rollback_bf16_trace failed: {status}"
                )));
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_apply_tree_rollback_q8_trace(
    ordinal: usize,
    conv_dtype: ScalarType,
    qkv_dim: usize,
    conv_state_len: usize,
    conv_input_len: usize,
    tree_len: usize,
    commit_len: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_input: &GpuBuffer,
    accepted_indices: &GpuBuffer,
    conv_state: &mut GpuBuffer,
    recurrent_trace: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "dflash_apply_tree_rollback_q8_trace is not implemented for Metal".into(),
        ));
    }
    if recurrent_trace.dtype() != ScalarType::U8 {
        return Err(ffi_error(
            "dflash_apply_tree_rollback_q8_trace requires U8 recurrent_trace".into(),
        ));
    }
    if accepted_indices.dtype() != ScalarType::U32 || accepted_indices.elem_count() < commit_len {
        return Err(GpuError::InvalidArg(format!(
            "dflash_apply_tree_rollback_q8_trace accepted_indices must be U32[{commit_len}], got {:?}[{}]",
            accepted_indices.dtype(),
            accepted_indices.elem_count()
        )));
    }
    ffi_profile_time_result("qwen.dflash_apply_tree_rollback_q8_trace", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_apply_tree_rollback_q8_trace(
                conv_dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                conv_state_len,
                conv_input_len,
                tree_len,
                commit_len,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_input.as_ptr(),
                accepted_indices.as_ptr(),
                conv_state.as_mut_ptr(),
                recurrent_trace.as_ptr(),
                recurrent_state.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "dflash_apply_tree_rollback_q8_trace failed: {status}"
            )));
        }
        Ok(())
    })
}

pub fn fill_conv_tail(
    ordinal: usize,
    dtype: ScalarType,
    qkv_dim: usize,
    pad: usize,
    total_len: usize,
    tail: &GpuBuffer,
    conv_input: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "fill_conv_tail is not implemented for Metal".into(),
        ));
    }
    ffi_profile_time_result("qwen.fill_conv_tail", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_fill_conv_tail(
                dtype.kernel_dtype_code(),
                ordinal,
                qkv_dim,
                pad,
                total_len,
                tail.as_ptr(),
                conv_input.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("fill_conv_tail failed: {status}")));
        }
        Ok(())
    })
}

pub fn dflash_extract_recurrent_attn(
    ordinal: usize,
    num_v_heads: usize,
    chunk_len: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    delta_out: &GpuBuffer,
    recurrent_state: &mut GpuBuffer,
    attn_output: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if delta_out.backend() != Backend::Hip {
        return Err(ffi_error(
            "dflash_extract_recurrent_attn is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.dflash_extract_recurrent_attn", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_dflash_extract_recurrent_attn(
                ordinal,
                num_v_heads,
                chunk_len,
                head_k_dim,
                head_v_dim,
                delta_out.as_ptr(),
                recurrent_state.as_mut_ptr(),
                attn_output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "dflash_extract_recurrent_attn failed: {status}"
            )));
        }
        Ok(())
    })
}

/// L2 normalization per row.
pub fn l2norm(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_l2norm() {
            let result = metal_profile_time("l2norm", "native", || {
                metal_native::l2norm(dtype, n_rows, n_cols, eps, input, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("l2norm", || {
            metal_host::l2norm(dtype, n_rows, n_cols, eps, input, out)
        });
    }
    ffi_profile_time_result("qwen.l2norm", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_l2norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_rows,
                n_cols,
                eps,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("l2norm failed: {status}")));
        }
        Ok(())
    })
}

/// SwiGLU: out = silu(gate) * up, element-wise.
pub fn swiglu_mul(
    ordinal: usize,
    dtype: ScalarType,
    elem_count: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    swiglu_mul_impl(ordinal, dtype, elem_count, gate, up, out, None)
}

pub fn swiglu_mul_with_options(
    ordinal: usize,
    dtype: ScalarType,
    elem_count: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    swiglu_mul_impl(ordinal, dtype, elem_count, gate, up, out, Some(options))
}

fn swiglu_mul_impl(
    ordinal: usize,
    dtype: ScalarType,
    elem_count: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options) {
            let result = prefill_metal_profile_time(options, "swiglu_mul", "native", || {
                metal_native::swiglu_mul(dtype, elem_count, gate, up, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "swiglu_mul", || {
            metal_host::swiglu_mul(dtype, elem_count, gate, up, out)
        });
    }
    ffi_profile_time_result("qwen.swiglu_mul", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_swiglu_mul(
                dtype.kernel_dtype_code(),
                ordinal,
                elem_count,
                gate.as_ptr(),
                up.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("swiglu_mul failed: {status}")));
        }
        Ok(())
    })
}

/// SwiGLU from a packed `[rows, 2 * cols]` buffer where each row is
/// `[gate..., up...]`.
pub fn swiglu_mul_split(
    ordinal: usize,
    dtype: ScalarType,
    rows: usize,
    cols: usize,
    gate_up: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "swiglu_mul_split is only implemented for HIP/CUDA backends".to_string(),
        ));
    }
    ffi_profile_time_result("qwen.swiglu_mul_split", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_swiglu_mul_split(
                dtype.kernel_dtype_code(),
                ordinal,
                rows,
                cols,
                gate_up.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("swiglu_mul_split failed: {status}")));
        }
        Ok(())
    })
}

/// RMSNorm with SiLU gating: out = rms_norm(hidden) * weight * silu(gate).
pub fn rms_norm_gated(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    hidden: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm_gated()
        {
            let result = metal_profile_time("rms_norm_gated", "native", || {
                metal_native::rms_norm_gated_bf16(n_rows, n_cols, eps, hidden, gate, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm_gated()
        {
            let result = metal_profile_time("rms_norm_gated", "native", || {
                metal_native::rms_norm_gated_f32(n_rows, n_cols, eps, hidden, gate, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("rms_norm_gated", || {
            metal_host::rms_norm_gated(dtype, n_rows, n_cols, eps, hidden, gate, weight, out)
        });
    }
    ffi_profile_time_result("qwen.rms_norm_gated", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_rms_norm_gated(
                dtype.kernel_dtype_code(),
                ordinal,
                n_rows,
                n_cols,
                eps,
                hidden.as_ptr(),
                gate.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("rms_norm_gated failed: {status}")));
        }
        Ok(())
    })
}

pub fn rms_norm_gated_sfirst_bf16(
    ordinal: usize,
    s: usize,
    nv: usize,
    vhd: usize,
    eps: f32,
    hidden_hsd: &GpuBuffer,
    gate_sfirst: &GpuBuffer,
    weight: &GpuBuffer,
    out_sfirst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out_sfirst.backend() != Backend::Hip {
        return Err(ffi_error(
            "rms_norm_gated_sfirst_bf16 is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.rms_norm_gated_sfirst_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_rms_norm_gated_sfirst_bf16(
                ordinal,
                s,
                nv,
                vhd,
                eps,
                hidden_hsd.as_ptr(),
                gate_sfirst.as_ptr(),
                weight.as_ptr(),
                out_sfirst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "rms_norm_gated_sfirst_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Multiply all elements by a scalar: out = xs * scalar.
pub fn mul_scalar(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    scalar: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_mul_scalar() {
            let result = metal_profile_time("mul_scalar", "native", || {
                metal_native::mul_scalar(dtype, total_elems, scalar, input, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("mul_scalar", || {
            metal_host::mul_scalar(dtype, total_elems, scalar, input, out)
        });
    }
    ffi_profile_time_result("qwen.mul_scalar", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_mul_scalar(
                dtype.kernel_dtype_code(),
                ordinal,
                total_elems,
                scalar,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("mul_scalar failed: {status}")));
        }
        Ok(())
    })
}

// ---- Fused RMSNorm + linear projection (F32 intermediate) ----

/// Fused RMSNorm → linear projection for multiple rows.
/// Keeps normed intermediate in F32 to avoid BF16 precision loss.
/// hidden: [n_rows, hidden_dim], norm_weight: [hidden_dim], proj_weight: [out_dim, hidden_dim]
/// out: [n_rows, out_dim]
pub fn fused_rms_norm_linear_rows(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    hidden_dim: usize,
    out_dim: usize,
    eps: f32,
    hidden: &GpuBuffer,
    norm_weight: &GpuBuffer,
    proj_weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        return metal_profile_host_time("fused_rms_norm_linear_rows", || {
            metal_host::fused_rms_norm_linear_rows(
                dtype,
                n_rows,
                hidden_dim,
                out_dim,
                eps,
                hidden,
                norm_weight,
                proj_weight,
                out,
            )
        });
    }
    let row_bytes = hidden_dim * dtype.size_in_bytes();
    let out_row_bytes = out_dim * dtype.size_in_bytes();
    for row in 0..n_rows {
        let hidden_ptr = hidden.offset_ptr(row * row_bytes);
        let out_ptr = unsafe {
            (out.as_mut_ptr() as *mut u8).add(row * out_row_bytes) as *mut std::ffi::c_void
        };
        let status = unsafe {
            supersonic_qwen35_hip_fused_rms_norm_linear(
                dtype.kernel_dtype_code(),
                ordinal,
                hidden_dim,
                out_dim,
                eps,
                1, // add_unit_offset (Qwen3.5 uses w + 1.0)
                hidden_ptr,
                norm_weight.as_ptr(),
                proj_weight.as_ptr(),
                out_ptr,
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "fused_rms_norm_linear row {row} failed: {status}"
            )));
        }
    }
    Ok(())
}

// ---- Matmul with transposed rhs (y = x @ W^T) ----

/// Matrix multiply with transposed rhs: out [m, n] = lhs [m, k] × rhs^T where rhs is [n, k].
/// This is the standard linear projection: y = x @ W.T where W is [out_dim, in_dim].
/// Uses a tiled kernel for performance.
pub fn matmul_rhs_transposed(
    ordinal: usize,
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    matmul_rhs_transposed_impl(ordinal, dtype, batch_elems, m, n, k, lhs, rhs, out, None)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_rhs_transposed_with_options(
    ordinal: usize,
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    matmul_rhs_transposed_impl(
        ordinal,
        dtype,
        batch_elems,
        m,
        n,
        k,
        lhs,
        rhs,
        out,
        Some(options),
    )
}

#[allow(clippy::too_many_arguments)]
fn matmul_rhs_transposed_impl(
    ordinal: usize,
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        // M=1 batch=1 GEMV: SIMD-group cooperative reduction. Used by every
        // decode-step projection (q/k/v/o/gate/up/down/lm_head). Opt-out for
        // bring-up/bisect via SUPERSONIC_METAL_DISABLE_GEMV_M1.
        //
        // Prefer the tiled variant when K fits in 16 KB of threadgroup memory
        // (K <= 4096 floats). Reuses `lhs` across 32 output cols per
        // threadgroup → 32× fewer device reads. Biggest win on lm_head where
        // N is huge. Falls back to the per-column GEMV when K is too large
        // (e.g. down_proj K=8960 for qwen3.5-0.8b).
        if dtype == ScalarType::BF16
            && batch_elems == 1
            && m == 1
            && !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_matmul,
                metal_force_host_matmul,
            )
            && !prefill_launch_flag(
                options,
                |options| options.disable_gemv_m1,
                || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1").is_some(),
            )
        {
            let tiled_disabled = prefill_launch_flag(
                options,
                |options| options.disable_gemv_m1_tiled,
                || std::env::var_os("SUPERSONIC_METAL_DISABLE_GEMV_M1_TILED").is_some(),
            );
            if !tiled_disabled && k <= 4096 {
                let result = prefill_metal_profile_time(
                    options,
                    "matmul_rhs_transposed_gemv_m1_tiled",
                    "native",
                    || metal_native::matmul_rhs_transposed_bf16_gemv_m1_tiled(n, k, lhs, rhs, out),
                );
                if result.is_ok() {
                    return result;
                }
            }
            let result = prefill_metal_profile_time(
                options,
                "matmul_rhs_transposed_gemv_m1",
                "native",
                || metal_native::matmul_rhs_transposed_bf16_gemv_m1(n, k, lhs, rhs, out),
            );
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::BF16
            && !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_matmul,
                metal_force_host_matmul,
            )
        {
            let result =
                prefill_metal_profile_time(options, "matmul_rhs_transposed", "native", || {
                    metal_native::matmul_rhs_transposed_bf16(batch_elems, m, n, k, lhs, rhs, out)
                });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_matmul,
                metal_force_host_matmul,
            )
        {
            let result =
                prefill_metal_profile_time(options, "matmul_rhs_transposed", "native", || {
                    metal_native::matmul_rhs_transposed_f32(batch_elems, m, n, k, lhs, rhs, out)
                });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "matmul_rhs_transposed", || {
            metal_host::matmul_rhs_transposed(dtype, batch_elems, m, n, k, lhs, rhs, out)
        });
    }
    let profile_key = if prefill_profile_shapes(options) {
        format!(
            "qwen.matmul_rhs_transposed[b={} m={} n={} k={} dtype={:?}]",
            batch_elems, m, n, k, dtype
        )
    } else {
        "qwen.matmul_rhs_transposed".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
                dtype.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("matmul_rhs_transposed failed: {status}")));
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn metal_matmul_rhs_transposed_residual_bf16(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_profile_time("matmul_rhs_transposed_residual", "native", || {
        metal_native::matmul_rhs_transposed_residual_bf16(
            batch_elems,
            m,
            n,
            k,
            lhs,
            rhs,
            residual,
            out,
        )
    })
}

/// FP8 dequant matmul: out [batch, m, n] = lhs [batch, m, k] × dequant(rhs_fp8 [batch, n, k])^T
/// rhs_fp8 is FP8 E4M3 weights, scale is BF16 scale_inv [n/block, k/block].
pub fn matmul_rhs_transposed_fp8(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_fp8: &GpuBuffer,
    scale: &GpuBuffer,
    block_size: usize,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        return metal_profile_host_time("matmul_rhs_transposed_fp8", || {
            metal_host::matmul_rhs_transposed_fp8(
                batch_elems,
                m,
                n,
                k,
                lhs,
                rhs_fp8,
                scale,
                block_size,
                out,
            )
        });
    }
    ffi_profile_time_result("qwen.matmul_rhs_transposed_fp8", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_fp8_dequant(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_fp8.as_ptr(),
                scale.as_ptr(),
                block_size as c_int,
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "matmul_rhs_transposed_fp8 failed: {status}"
            )));
        }
        Ok(())
    })
}

/// INT4 dequant matmul: out [batch, m, n] = lhs [batch, m, k] × dequant(rhs_int4 [batch, n, k/2])^T
/// rhs_int4 is packed INT4 (2 nibbles per byte), scale/zero are BF16 [n/group, k/group].
pub fn matmul_rhs_transposed_int4(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_int4: &GpuBuffer,
    scale: &GpuBuffer,
    zero: &GpuBuffer,
    awq_inv_scale: Option<&GpuBuffer>,
    group_size: usize,
    quant_type: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    matmul_rhs_transposed_int4_impl(
        ordinal,
        batch_elems,
        m,
        n,
        k,
        lhs,
        rhs_int4,
        scale,
        zero,
        awq_inv_scale,
        group_size,
        quant_type,
        out,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_rhs_transposed_int4_with_options(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_int4: &GpuBuffer,
    scale: &GpuBuffer,
    zero: &GpuBuffer,
    awq_inv_scale: Option<&GpuBuffer>,
    group_size: usize,
    quant_type: i32,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    matmul_rhs_transposed_int4_impl(
        ordinal,
        batch_elems,
        m,
        n,
        k,
        lhs,
        rhs_int4,
        scale,
        zero,
        awq_inv_scale,
        group_size,
        quant_type,
        out,
        Some(options),
    )
}

#[allow(clippy::too_many_arguments)]
fn matmul_rhs_transposed_int4_impl(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_int4: &GpuBuffer,
    scale: &GpuBuffer,
    zero: &GpuBuffer,
    awq_inv_scale: Option<&GpuBuffer>,
    group_size: usize,
    quant_type: i32,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        // M=1 batch=1 GEMV: SIMD-group cooperative reduction with on-the-fly
        // nibble dequant. Used by every decode-step projection. Disable via
        // SUPERSONIC_METAL_DISABLE_INT4_GEMV_M1 for bring-up/bisect.
        //
        // Prefer the tiled variant when K fits in 16 KB of threadgroup memory
        // (K <= 4096 floats). Reuses lhs across 32 cols per threadgroup → ~32×
        // fewer device reads of the lhs vector. Biggest win on lm_head where N
        // is large. Falls back to the per-column INT4 GEMV for larger K (e.g.
        // down_proj K=8960).
        if batch_elems == 1
            && m == 1
            && !prefill_launch_flag(
                options,
                |options| options.disable_int4_gemv_m1,
                || std::env::var_os("SUPERSONIC_METAL_DISABLE_INT4_GEMV_M1").is_some(),
            )
        {
            let tiled_disabled = prefill_launch_flag(
                options,
                |options| options.disable_int4_gemv_m1_tiled,
                || std::env::var_os("SUPERSONIC_METAL_DISABLE_INT4_GEMV_M1_TILED").is_some(),
            );
            if !tiled_disabled && k <= 4096 {
                let result = prefill_metal_profile_time(
                    options,
                    "matmul_rhs_transposed_int4_gemv_m1_tiled",
                    "native",
                    || {
                        metal_native::matmul_rhs_transposed_int4_bf16_gemv_m1_tiled(
                            n, k, group_size, lhs, rhs_int4, scale, zero, out,
                        )
                    },
                );
                if result.is_ok() {
                    return result;
                }
            }
            let result = prefill_metal_profile_time(
                options,
                "matmul_rhs_transposed_int4_gemv_m1",
                "native",
                || {
                    metal_native::matmul_rhs_transposed_int4_bf16_gemv_m1(
                        n, k, group_size, lhs, rhs_int4, scale, zero, out,
                    )
                },
            );
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_time(options, "matmul_rhs_transposed_int4", "native", || {
            metal_native::matmul_rhs_transposed_int4_bf16(
                batch_elems,
                m,
                n,
                k,
                group_size,
                lhs,
                rhs_int4,
                scale,
                zero,
                out,
            )
        });
    }
    let profile_key = if prefill_profile_shapes(options) {
        format!(
            "qwen.matmul_rhs_transposed_int4[b={} m={} n={} k={} g={} qt={}]",
            batch_elems, m, n, k, group_size, quant_type
        )
    } else {
        "qwen.matmul_rhs_transposed_int4".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_int4_dequant(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_int4.as_ptr(),
                scale.as_ptr(),
                zero.as_ptr(),
                awq_inv_scale
                    .map(|buf| buf.as_ptr())
                    .unwrap_or(std::ptr::null()),
                group_size as c_int,
                quant_type as c_int,
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "matmul_rhs_transposed_int4 failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Raw GGML low-bit m16 matmul with a BF16 residual-add epilogue.
///
/// Returns `Ok(false)` when the HIP kernel does not handle the shape/qtype, so
/// callers can fall back to `matmul_rhs_transposed_int4` plus `element_add`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_rhs_transposed_int4_residual_add(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_int4: &GpuBuffer,
    scale: &GpuBuffer,
    zero: &GpuBuffer,
    awq_inv_scale: Option<&GpuBuffer>,
    group_size: usize,
    quant_type: i32,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<bool, GpuError> {
    if out.backend() == Backend::Metal {
        return Ok(false);
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.matmul_rhs_transposed_int4_residual_add[b={} m={} n={} k={} g={} qt={}]",
            batch_elems, m, n, k, group_size, quant_type
        )
    } else {
        "qwen.matmul_rhs_transposed_int4_residual_add".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_int4.as_ptr(),
                scale.as_ptr(),
                zero.as_ptr(),
                awq_inv_scale
                    .map(|buf| buf.as_ptr())
                    .unwrap_or(std::ptr::null()),
                group_size as c_int,
                quant_type as c_int,
                residual.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        match status {
            0 => Ok(true),
            312..=316 | 319 => Ok(false),
            _ => Err(ffi_error(format!(
                "matmul_rhs_transposed_int4_residual_add failed: {status}"
            ))),
        }
    })
}

/// GGML low-bit pair matmul: out `[batch, m, 2 * n_each]` contains
/// `[lhs * rhs_first^T, lhs * rhs_second^T]` per row.
pub fn matmul_rhs_transposed_ggml_pair(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n_each: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_first: &GpuBuffer,
    rhs_second: &GpuBuffer,
    quant_type: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "matmul_rhs_transposed_ggml_pair is only implemented for HIP/CUDA backends".to_string(),
        ));
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.matmul_rhs_transposed_ggml_pair[b={} m={} n_each={} k={} qt={}]",
            batch_elems, m, n_each, k, quant_type
        )
    } else {
        "qwen.matmul_rhs_transposed_ggml_pair".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n_each as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_first.as_ptr(),
                rhs_second.as_ptr(),
                quant_type as c_int,
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "matmul_rhs_transposed_ggml_pair failed: {status}"
            )));
        }
        Ok(())
    })
}

/// GGML low-bit pair matmul plus SwiGLU epilogue:
/// out `[batch, m, n_each] = silu(lhs * rhs_gate^T) * (lhs * rhs_up^T)`.
pub fn matmul_rhs_transposed_ggml_pair_swiglu(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n_each: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_gate: &GpuBuffer,
    rhs_up: &GpuBuffer,
    quant_type: i32,
    out: &mut GpuBuffer,
) -> Result<bool, GpuError> {
    if out.backend() == Backend::Metal {
        return Ok(false);
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.matmul_rhs_transposed_ggml_pair_swiglu[b={} m={} n_each={} k={} qt={}]",
            batch_elems, m, n_each, k, quant_type
        )
    } else {
        "qwen.matmul_rhs_transposed_ggml_pair_swiglu".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n_each as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_gate.as_ptr(),
                rhs_up.as_ptr(),
                quant_type as c_int,
                out.as_mut_ptr(),
            )
        };
        match status {
            0 => Ok(true),
            322..=324 | 327 => Ok(false),
            _ => Err(ffi_error(format!(
                "matmul_rhs_transposed_ggml_pair_swiglu failed: {status}"
            ))),
        }
    })
}

/// Harness-facing Q8_1/MMQ activation quantization.
///
/// The output layout uses 144 bytes per `[128]` activation block, matching
/// llama.cpp's `block_q8_1_mmq` size. Q4_K/Q5_K store DS4 metadata; Q6_K
/// stores D4 metadata. This is not wired into runtime dispatch.
pub fn quantize_mmq_q8_1(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    k: usize,
    lhs: &GpuBuffer,
    quant_type: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "quantize_mmq_q8_1 is only implemented for HIP/CUDA backends".to_string(),
        ));
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.quantize_mmq_q8_1[b={} m={} k={} qt={}]",
            batch_elems, m, k, quant_type
        )
    } else {
        "qwen.quantize_mmq_q8_1".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_quantize_mmq_q8_1(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                k as c_int,
                lhs.as_ptr(),
                quant_type as c_int,
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("quantize_mmq_q8_1 failed: {status}")));
        }
        Ok(())
    })
}

/// Harness-facing Q6_K MMQ consumer for the Q8_1 activation workspace.
///
/// This is intentionally not wired into runtime dispatch yet; it exists to
/// validate the Lucebox/llama-style `mmq_y=128` Q6_K tile path in isolation.
pub fn matmul_mmq_q8_1_q6_k(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    q8: &GpuBuffer,
    rhs_q6: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "matmul_mmq_q8_1_q6_k is only implemented for HIP/CUDA backends".to_string(),
        ));
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!("qwen.matmul_mmq_q8_1_q6_k[b={batch_elems} m={m} n={n} k={k}]")
    } else {
        "qwen.matmul_mmq_q8_1_q6_k".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                q8.as_ptr(),
                rhs_q6.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("matmul_mmq_q8_1_q6_k failed: {status}")));
        }
        Ok(())
    })
}

pub fn matmul_mmq_q8_1_q6_k_residual_add(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    q8: &GpuBuffer,
    rhs_q6: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        return Err(ffi_error(
            "matmul_mmq_q8_1_q6_k_residual_add is only implemented for HIP/CUDA backends"
                .to_string(),
        ));
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!("qwen.matmul_mmq_q8_1_q6_k_residual_add[b={batch_elems} m={m} n={n} k={k}]")
    } else {
        "qwen.matmul_mmq_q8_1_q6_k_residual_add".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k_residual_add(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                q8.as_ptr(),
                rhs_q6.as_ptr(),
                residual.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "matmul_mmq_q8_1_q6_k_residual_add failed: {status}"
            )));
        }
        Ok(())
    })
}

pub fn matmul_q6_k_m16_argmax(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_q6: &GpuBuffer,
    block_best_vals: &mut GpuBuffer,
    block_best_indices: &mut GpuBuffer,
    out_indices: &mut GpuBuffer,
) -> Result<bool, GpuError> {
    if out_indices.backend() == Backend::Metal {
        return Ok(false);
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!("qwen.matmul_q6_k_m16_argmax[b={batch_elems} m={m} n={n} k={k}]")
    } else {
        "qwen.matmul_q6_k_m16_argmax".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_matmul_q6_k_m16_argmax(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                batch_elems,
                m as c_int,
                n as c_int,
                k as c_int,
                lhs.as_ptr(),
                rhs_q6.as_ptr(),
                block_best_vals.as_mut_ptr(),
                block_best_indices.as_mut_ptr(),
                out_indices.as_mut_ptr(),
            )
        };
        match status {
            0 => Ok(true),
            340..=349 => Ok(false),
            _ => Err(ffi_error(format!(
                "matmul_q6_k_m16_argmax failed: {status}"
            ))),
        }
    })
}

pub fn device_supports_wmma_i8(ordinal: usize) -> Result<bool, GpuError> {
    if gpu_hal::current_backend() != Backend::Hip {
        return Ok(false);
    }
    let mut supported: c_int = 0;
    let status =
        unsafe { supersonic_qwen35_4b_hip_device_supports_wmma_i8(ordinal, &mut supported) };
    if status != 0 {
        return Err(ffi_error(format!(
            "device_supports_wmma_i8 failed: {status}"
        )));
    }
    Ok(supported != 0)
}

pub fn matmul_rhs_transposed_int8(
    ordinal: usize,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs_int8: &GpuBuffer,
    scale: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    #[cfg(not(supersonic_backend_cuda))]
    {
        return Err(ffi_error(
            "matmul_rhs_transposed_int8 requires the CUDA backend".to_string(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        if gpu_hal::current_backend() != Backend::Cuda {
            return Err(ffi_error(
                "matmul_rhs_transposed_int8 requires the CUDA backend".to_string(),
            ));
        }

        ffi_profile_time_result("qwen.matmul_rhs_transposed_int8", ordinal, || {
            let status = unsafe {
                supersonic_qwen35_4b_hip_matmul_int8(
                    ScalarType::BF16.kernel_dtype_code(),
                    ordinal,
                    batch_elems,
                    m as c_int,
                    n as c_int,
                    k as c_int,
                    lhs.as_ptr(),
                    rhs_int8.as_ptr(),
                    scale.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!(
                    "matmul_rhs_transposed_int8 failed: {status}"
                )));
            }
            Ok(())
        })
    }
}

pub fn int8_outlier_add(
    ordinal: usize,
    rows: usize,
    n: usize,
    k: usize,
    sub_cols: usize,
    rhs_int8: &GpuBuffer,
    scale: &GpuBuffer,
    outlier_cols: &GpuBuffer,
    outlier_vals: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    #[cfg(not(supersonic_backend_cuda))]
    {
        return Err(ffi_error(
            "int8_outlier_add requires the CUDA backend".to_string(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        if gpu_hal::current_backend() != Backend::Cuda {
            return Err(ffi_error(
                "int8_outlier_add requires the CUDA backend".to_string(),
            ));
        }

        ffi_profile_time_result("qwen.int8_outlier_add", ordinal, || {
            let status = unsafe {
                supersonic_qwen35_4b_hip_int8_outlier_add(
                    ScalarType::BF16.kernel_dtype_code(),
                    ordinal,
                    rows as c_int,
                    n as c_int,
                    k as c_int,
                    sub_cols as c_int,
                    rhs_int8.as_ptr(),
                    scale.as_ptr(),
                    outlier_cols.as_ptr(),
                    outlier_vals.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(ffi_error(format!("int8_outlier_add failed: {status}")));
            }
            Ok(())
        })
    }
}

pub fn int4_sparse_outlier_add(
    ordinal: usize,
    rows: usize,
    n: usize,
    k: usize,
    sub_cols: usize,
    lhs: &GpuBuffer,
    outlier_cols: &GpuBuffer,
    outlier_delta: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if gpu_hal::current_backend() != Backend::Hip && gpu_hal::current_backend() != Backend::Cuda {
        return Err(ffi_error(
            "int4_sparse_outlier_add requires the HIP or CUDA backend".to_string(),
        ));
    }

    ffi_profile_time_result("qwen.int4_sparse_outlier_add", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_int4_sparse_outlier_add(
                ScalarType::BF16.kernel_dtype_code(),
                ordinal,
                rows as c_int,
                n as c_int,
                k as c_int,
                sub_cols as c_int,
                lhs.as_ptr(),
                outlier_cols.as_ptr(),
                outlier_delta.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "int4_sparse_outlier_add failed: {status}"
            )));
        }
        Ok(())
    })
}

// ---- Multi-row RMSNorm (for prefill — n_rows > 1) ----

/// RMSNorm on multiple rows. Each row is independently normalized.
/// Qwen3.5 uses add_unit_offset=1 (weight applied as (w + 1.0) * x).
pub fn rms_norm_rows(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    rms_norm_rows_impl(
        ordinal, dtype, n_rows, n_cols, eps, input, weight, out, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rows_with_options(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    rms_norm_rows_impl(
        ordinal,
        dtype,
        n_rows,
        n_cols,
        eps,
        input,
        weight,
        out,
        Some(options),
    )
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_rows_impl(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_rms_norm,
                metal_force_host_rms_norm,
            )
        {
            let result = prefill_metal_profile_time(options, "rms_norm_rows", "native", || {
                metal_native::rms_norm_rows_bf16(n_rows, n_cols, eps, true, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_rms_norm,
                metal_force_host_rms_norm,
            )
        {
            let result = prefill_metal_profile_time(options, "rms_norm_rows", "native", || {
                metal_native::rms_norm_rows_f32(n_rows, n_cols, eps, true, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "rms_norm_rows", || {
            metal_host::rms_norm_rows(dtype, n_rows, n_cols, eps, true, input, weight, out)
        });
    }
    let profile_key = if prefill_profile_shapes(options) {
        format!(
            "qwen.rms_norm_rows[rows={} cols={} dtype={:?}]",
            n_rows, n_cols, dtype
        )
    } else {
        "qwen.rms_norm_rows".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_rows,
                n_cols,
                eps,
                1, // add_unit_offset
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("rms_norm_rows failed: {status}")));
        }
        Ok(())
    })
}

/// Multi-row RMSNorm WITHOUT add_unit_offset. Qwen3 (the dflash draft base)
/// uses plain `x * rms * w`, not `x * rms * (w + 1)`. The underlying kernel
/// already supports this via a template flag; this wrapper just passes 0.
pub fn rms_norm_rows_plain(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows_plain", "native", || {
                metal_native::rms_norm_rows_bf16(n_rows, n_cols, eps, false, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows_plain", "native", || {
                metal_native::rms_norm_rows_f32(n_rows, n_cols, eps, false, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("rms_norm_rows_plain", || {
            metal_host::rms_norm_rows(dtype, n_rows, n_cols, eps, false, input, weight, out)
        });
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.rms_norm_rows_plain[rows={} cols={} dtype={:?}]",
            n_rows, n_cols, dtype
        )
    } else {
        "qwen.rms_norm_rows_plain".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_rows,
                n_cols,
                eps,
                0,
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("rms_norm_rows_plain failed: {status}")));
        }
        Ok(())
    })
}

/// In-place variant of [`rms_norm_rows_plain`]. The underlying kernel reads
/// each row before writing it, so aliasing input/output is safe. Avoids the
/// borrow-checker dance at callsites that normalize into their own buffer.
pub fn rms_norm_rows_plain_inplace(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    data: &mut GpuBuffer,
    weight: &GpuBuffer,
) -> Result<(), GpuError> {
    if data.backend() == Backend::Metal {
        let mut input = GpuBuffer::zeros(ordinal, dtype, data.shape())?;
        metal_native::flush_batch()?;
        gpu_hal::copy_d2d(ordinal, input.as_mut_ptr(), data.as_ptr(), data.len_bytes())?;
        if dtype == ScalarType::BF16
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows_plain_inplace", "native", || {
                metal_native::rms_norm_rows_bf16(n_rows, n_cols, eps, false, &input, weight, data)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows_plain_inplace", "native", || {
                metal_native::rms_norm_rows_f32(n_rows, n_cols, eps, false, &input, weight, data)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("rms_norm_rows_plain_inplace", || {
            metal_host::rms_norm_rows(dtype, n_rows, n_cols, eps, false, &input, weight, data)
        });
    }
    let profile_key = if std::env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES").is_some() {
        format!(
            "qwen.rms_norm_rows_plain_inplace[rows={} cols={} dtype={:?}]",
            n_rows, n_cols, dtype
        )
    } else {
        "qwen.rms_norm_rows_plain_inplace".to_string()
    };
    ffi_profile_time_result_key(profile_key, ordinal, || {
        let ptr = data.as_mut_ptr();
        let status = unsafe {
            supersonic_qwen35_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_rows,
                n_cols,
                eps,
                0,
                ptr,
                weight.as_ptr(),
                ptr,
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "rms_norm_rows_plain_inplace failed: {status}"
            )));
        }
        Ok(())
    })
}

/// In-place `lhs += rhs`. Aliasing lhs and out in the underlying kernel is
/// fine — it reads both operands into registers before the store.
pub fn element_add_inplace(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    lhs_out: &mut GpuBuffer,
    rhs: &GpuBuffer,
) -> Result<(), GpuError> {
    element_add_inplace_impl(ordinal, dtype, total_elems, lhs_out, rhs, None)
}

pub fn element_add_inplace_with_options(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    lhs_out: &mut GpuBuffer,
    rhs: &GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    element_add_inplace_impl(ordinal, dtype, total_elems, lhs_out, rhs, Some(options))
}

fn element_add_inplace_impl(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    lhs_out: &mut GpuBuffer,
    rhs: &GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if lhs_out.backend() == Backend::Metal {
        let mut lhs = GpuBuffer::zeros(ordinal, dtype, lhs_out.shape())?;
        metal_native::flush_batch()?;
        gpu_hal::copy_d2d(
            ordinal,
            lhs.as_mut_ptr(),
            lhs_out.as_ptr(),
            lhs_out.len_bytes(),
        )?;
        if !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_element_add,
                metal_force_host_element_add,
            )
        {
            let result =
                prefill_metal_profile_time(options, "element_add_inplace", "native", || {
                    metal_native::element_add(dtype, total_elems, &lhs, rhs, lhs_out)
                });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "element_add_inplace", || {
            metal_host::element_add(dtype, total_elems, &lhs, rhs, lhs_out)
        });
    }
    let ptr = lhs_out.as_mut_ptr();
    let status = unsafe {
        supersonic_qwen35_hip_element_add(
            dtype.kernel_dtype_code(),
            ordinal,
            total_elems,
            ptr,
            rhs.as_ptr(),
            ptr,
        )
    };
    if status != 0 {
        return Err(ffi_error(format!("element_add_inplace failed: {status}")));
    }
    Ok(())
}

pub fn qwen36_router_softmax_topk_bf16(
    n_tokens: usize,
    num_experts: usize,
    top_k: usize,
    logits: &GpuBuffer,
    topk_idx: &mut GpuBuffer,
    topk_weight: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if logits.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_router_softmax_topk_bf16 requires Metal logits".into(),
        ));
    }
    metal_profile_time(
        "qwen36_batched_prefill_router_softmax_topk",
        "native",
        || {
            metal_native::qwen36_router_softmax_topk_bf16(
                n_tokens,
                num_experts,
                top_k,
                logits,
                topk_idx,
                topk_weight,
            )
        },
    )
}

pub fn qwen36_router_softmax_topk_bf16_with_options(
    n_tokens: usize,
    num_experts: usize,
    top_k: usize,
    logits: &GpuBuffer,
    topk_idx: &mut GpuBuffer,
    topk_weight: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    if logits.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_router_softmax_topk_bf16_with_options requires Metal logits".into(),
        ));
    }
    prefill_metal_profile_time(
        Some(options),
        "qwen36_batched_prefill_router_softmax_topk",
        "native",
        || {
            metal_native::qwen36_router_softmax_topk_bf16(
                n_tokens,
                num_experts,
                top_k,
                logits,
                topk_idx,
                topk_weight,
            )
        },
    )
}

pub fn qwen36_ffn_residual_add_bf16(
    total_elems: usize,
    residual: &mut GpuBuffer,
    combined: &GpuBuffer,
    shared: &GpuBuffer,
) -> Result<(), GpuError> {
    if residual.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_ffn_residual_add_bf16 requires a Metal residual buffer".into(),
        ));
    }
    metal_profile_time("qwen36_batched_prefill_ffn_residual_add", "native", || {
        metal_native::qwen36_ffn_residual_add_bf16(total_elems, residual, combined, shared)
    })
}

pub fn qwen36_ffn_residual_add_bf16_with_options(
    total_elems: usize,
    residual: &mut GpuBuffer,
    combined: &GpuBuffer,
    shared: &GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    if residual.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_ffn_residual_add_bf16_with_options requires a Metal residual buffer".into(),
        ));
    }
    prefill_metal_profile_time(
        Some(options),
        "qwen36_batched_prefill_ffn_residual_add",
        "native",
        || metal_native::qwen36_ffn_residual_add_bf16(total_elems, residual, combined, shared),
    )
}

// ---- Cast between dtypes ----

/// Cast all elements from one dtype to another on GPU.
pub fn cast(
    ordinal: usize,
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    cast_impl(
        ordinal,
        input_dtype,
        output_dtype,
        total_elems,
        input,
        out,
        None,
    )
}

pub fn cast_with_options(
    ordinal: usize,
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    cast_impl(
        ordinal,
        input_dtype,
        output_dtype,
        total_elems,
        input,
        out,
        Some(options),
    )
}

fn cast_impl(
    ordinal: usize,
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_cast,
                metal_force_host_cast,
            )
        {
            let result = prefill_metal_profile_time(options, "cast", "native", || {
                metal_native::cast(input_dtype, output_dtype, total_elems, input, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "cast", || {
            metal_host::cast(input_dtype, output_dtype, total_elems, input, out)
        });
    }
    ffi_profile_time_result("qwen.cast", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_cast(
                input_dtype.kernel_dtype_code(),
                output_dtype.kernel_dtype_code(),
                ordinal,
                total_elems,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("cast failed: {status}")));
        }
        Ok(())
    })
}

// ---- Element-wise add ----

/// Element-wise addition: out[i] = lhs[i] + rhs[i].
pub fn element_add(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_element_add() {
            let result = metal_profile_time("element_add", "native", || {
                metal_native::element_add(dtype, total_elems, lhs, rhs, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("element_add", || {
            metal_host::element_add(dtype, total_elems, lhs, rhs, out)
        });
    }
    ffi_profile_time_result("qwen.element_add", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_element_add(
                dtype.kernel_dtype_code(),
                ordinal,
                total_elems,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("element_add failed: {status}")));
        }
        Ok(())
    })
}

pub fn argmax_bf16_rows(
    ordinal: usize,
    rows: usize,
    cols: usize,
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if gpu_hal::current_backend() != Backend::Hip {
        return Err(ffi_error(
            "argmax_bf16_rows requires the HIP backend".to_string(),
        ));
    }
    if rows == 0 || cols == 0 {
        return Err(GpuError::InvalidArg(
            "argmax_bf16_rows requires non-zero rows and cols".into(),
        ));
    }
    if logits.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "argmax_bf16_rows logits must be BF16, got {:?}",
            logits.dtype()
        )));
    }
    if logits.elem_count() < rows.saturating_mul(cols) {
        return Err(GpuError::InvalidArg(format!(
            "argmax_bf16_rows logits has {} elems, need {}",
            logits.elem_count(),
            rows.saturating_mul(cols)
        )));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() < rows {
        return Err(GpuError::InvalidArg(format!(
            "argmax_bf16_rows output must be U32[{rows}], got {:?}[{}]",
            out_index.dtype(),
            out_index.elem_count()
        )));
    }

    ffi_profile_time_result("qwen.argmax_bf16_rows", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_argmax_bf16_rows(
                ordinal,
                rows,
                cols,
                logits.as_ptr(),
                out_index.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("argmax_bf16_rows failed: {status}")));
        }
        Ok(())
    })
}

// ---- RoPE for prefill ----

/// Apply RoPE in-place on tensor [seq_len, num_heads, head_dim].
/// Only the first rotary_dim dimensions of each head are rotated.
/// Apply rotary position embeddings to data in-place.
/// `pos_offset`: starting position index (0 for first chunk, chunk_start for subsequent chunks).
/// The kernel reads cos/sin from position pos_offset..pos_offset+seq_len.
pub fn apply_rope_prefill(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    pos_offset: usize,
    data: &mut GpuBuffer,
) -> Result<(), GpuError> {
    apply_rope_prefill_impl(
        ordinal, dtype, seq_len, num_heads, head_dim, rotary_dim, cos_table, sin_table, pos_offset,
        data, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rope_prefill_with_options(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    pos_offset: usize,
    data: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    apply_rope_prefill_impl(
        ordinal,
        dtype,
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        cos_table,
        sin_table,
        pos_offset,
        data,
        Some(options),
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_rope_prefill_impl(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    pos_offset: usize,
    data: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if data.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options) {
            let result =
                prefill_metal_profile_time(options, "apply_rope_prefill", "native", || {
                    metal_native::apply_rope_prefill(
                        dtype, seq_len, num_heads, head_dim, rotary_dim, cos_table, sin_table,
                        pos_offset, data,
                    )
                });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "apply_rope_prefill", || {
            metal_host::apply_rope_prefill(
                dtype, seq_len, num_heads, head_dim, rotary_dim, cos_table, sin_table, pos_offset,
                data,
            )
        });
    }
    let half_rot = rotary_dim / 2;
    // Offset cos/sin table pointers by pos_offset positions.
    // Table layout: [max_positions, half_rot] BF16 → stride = half_rot * 2 bytes per position
    let table_byte_offset = pos_offset * half_rot * dtype.size_in_bytes();
    let cos_ptr = cos_table.offset_ptr(table_byte_offset);
    let sin_ptr = sin_table.offset_ptr(table_byte_offset);
    ffi_profile_time_result("qwen.apply_rope_prefill", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_apply_rope_prefill(
                dtype.kernel_dtype_code(),
                ordinal,
                seq_len,
                num_heads,
                head_dim,
                half_rot,
                cos_ptr,
                sin_ptr,
                data.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("apply_rope_prefill failed: {status}")));
        }
        Ok(())
    })
}

/// SpecPrefill (arXiv 2502.02789): apply RoPE in-place using a per-token
/// position-ID array. `pos_ids[slot]` selects the cos/sin table row each
/// token gets rotated by, instead of using `slot` directly.
///
/// Layout:
///   - `data`: `[seq_len, num_heads, head_dim]` of `dtype` — modified in place.
///   - `cos_table` / `sin_table`: `[max_positions, half_rot]` of `dtype`.
///     Pre-built for the full prompt's position range; the kernel just
///     gathers from row `pos_ids[slot]`.
///   - `pos_ids`: `[seq_len]` `i32` — the original prompt position each
///     compacted slot belongs to. All values must be in `[0, max_positions)`.
///
/// Parity guarantee: when `pos_ids = [0, 1, ..., seq_len-1]`, output is
/// byte-identical to [`apply_rope_prefill`] with `pos_offset=0`. See
/// `crates/runner/tests/specprefill_rope_indirect_parity.rs`.
///
/// Supported on HIP and CUDA. Metal stubs return -1.
pub fn apply_rope_prefill_indirect(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    pos_ids: &GpuBuffer,
    data: &mut GpuBuffer,
) -> Result<(), GpuError> {
    // Position IDs are stored as `ScalarType::U32` (4-byte unsigned) on the
    // host side because gpu-hal doesn't carry an i32 variant; the kernel
    // reinterprets the buffer as `const int*` (signed). This is safe
    // because position IDs are always non-negative and well below 2^31.
    if pos_ids.dtype() != ScalarType::U32 {
        return Err(ffi_error(format!(
            "apply_rope_prefill_indirect: pos_ids must be ScalarType::U32 (treated as i32 \
             by the kernel), got {:?}",
            pos_ids.dtype()
        )));
    }
    if pos_ids.elem_count() < seq_len {
        return Err(ffi_error(format!(
            "apply_rope_prefill_indirect: pos_ids has {} elements but seq_len is {}",
            pos_ids.elem_count(),
            seq_len
        )));
    }
    let half_rot = rotary_dim / 2;
    let status = unsafe {
        supersonic_qwen35_hip_apply_rope_prefill_indirect(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_heads,
            head_dim,
            half_rot,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            pos_ids.as_ptr() as *const c_int,
            data.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "apply_rope_prefill_indirect failed: {status}"
        )));
    }
    Ok(())
}

/// SpecPrefill (arXiv 2502.02789): per-row softmax(Q · Kᵀ) for the last
/// `lookahead_count` query rows of a single full-attention layer.
/// Computes the importance signal the host-side selection consumes.
///
/// Layouts:
/// - `q`: BF16 `[lookahead_count, q_heads, head_dim]` (post-RoPE)
/// - `k`: BF16 `[kv_heads, kv_len, head_dim]`
/// - `scores`: F32 `[q_heads, lookahead_count, kv_len]` (output)
///
/// `q_heads` must be a multiple of `kv_heads` (GQA broadcasting handled
/// inside the kernel via `num_kv_groups = q_heads / kv_heads`).
pub fn lookahead_attention_scores(
    ordinal: usize,
    dtype: ScalarType,
    q_heads: usize,
    kv_heads: usize,
    lookahead_count: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    scores: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if !matches!(dtype, ScalarType::BF16 | ScalarType::F16) {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: dtype must be ScalarType::BF16 or ScalarType::F16, got {:?}",
            dtype
        )));
    }
    if q_heads == 0 || kv_heads == 0 || lookahead_count == 0 || kv_len == 0 || head_dim == 0 {
        return Err(ffi_error(
            "lookahead_attention_scores: all dimensions must be > 0".into(),
        ));
    }
    if q_heads % kv_heads != 0 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: q_heads ({q_heads}) must be a multiple of kv_heads ({kv_heads})"
        )));
    }
    // gfx1100 LDS is 64 KiB per block; the kernel allocates kv_len * f32
    // for the per-row exponentials. We cap at 32 KiB (8000 tokens) to
    // leave room for compiler-allocated shared memory. Long prompts
    // beyond this need a tiled/online-softmax kernel — Phase D work.
    const MAX_KV_LEN_LOOKAHEAD: usize = 8 * 1024;
    if kv_len > MAX_KV_LEN_LOOKAHEAD {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: kv_len ({kv_len}) exceeds the LDS-bounded \
             maximum of {MAX_KV_LEN_LOOKAHEAD}. Long prompts need a tiled scoring \
             kernel — currently a Phase C limitation. Reduce prompt length or use \
             a smaller --specprefill-keep-ratio so the speculator's K cache fits."
        )));
    }
    let expected_q = lookahead_count * q_heads * head_dim;
    if q.elem_count() < expected_q {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: q has {} elems, expected >= {}",
            q.elem_count(),
            expected_q
        )));
    }
    let expected_k = kv_heads * kv_len * head_dim;
    if k.elem_count() < expected_k {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: k has {} elems, expected >= {}",
            k.elem_count(),
            expected_k
        )));
    }
    if scores.dtype() != ScalarType::F32 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: scores must be ScalarType::F32, got {:?}",
            scores.dtype()
        )));
    }
    let expected_scores = q_heads * lookahead_count * kv_len;
    if scores.elem_count() < expected_scores {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: scores has {} elems, expected >= {}",
            scores.elem_count(),
            expected_scores
        )));
    }
    let status = unsafe {
        supersonic_qwen35_hip_lookahead_attention_scores(
            dtype.kernel_dtype_code(),
            ordinal,
            q_heads,
            kv_heads,
            lookahead_count,
            kv_len,
            head_dim,
            scale,
            q.as_ptr(),
            k.as_ptr(),
            scores.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores failed: {status}"
        )));
    }
    Ok(())
}

/// SpecPrefill (Phase D PFlash-style): per-block cosine similarity
/// between the block-mean K vector and the K at the last prompt
/// position, computed from the drafter's K cache after dense prefill.
/// Replaces the lookahead-attention scoring of Phase C with a single
/// kernel pass that doesn't need decode steps.
///
/// Layout: `k_cache` is the drafter's full-attention K cache for one
/// layer, BF16 with shape `[1, kv_heads, cap, head_dim]` (the standard
/// `qwen35::state::LayerState::kv_cache_k` allocation).
///
/// `last_pos` must be in `[0, n_pos)` and `cap >= n_pos`. The kernel
/// reads positions `[0, n_pos)` only — the prompt context, not any
/// decode-side appends.
///
/// Output: `scores` is F32 of length `n_blocks` where
/// `n_blocks = (n_pos + block_size - 1) / block_size`. Each cell is
/// the cosine in `[-1, 1]`.
pub fn pflash_cosine_score(
    ordinal: usize,
    dtype: ScalarType,
    n_pos: usize,
    kv_heads: usize,
    cap: usize,
    head_dim: usize,
    block_size: usize,
    last_pos: usize,
    k_cache: &GpuBuffer,
    scores: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if !matches!(dtype, ScalarType::BF16 | ScalarType::F16) {
        return Err(ffi_error(format!(
            "pflash_cosine_score: dtype must be ScalarType::BF16 or ScalarType::F16, got {:?}",
            dtype
        )));
    }
    if n_pos == 0 || kv_heads == 0 || cap == 0 || head_dim == 0 || block_size == 0 {
        return Err(ffi_error(
            "pflash_cosine_score: all dimensions must be > 0".into(),
        ));
    }
    if last_pos >= n_pos {
        return Err(ffi_error(format!(
            "pflash_cosine_score: last_pos ({last_pos}) must be < n_pos ({n_pos})"
        )));
    }
    if cap < n_pos {
        return Err(ffi_error(format!(
            "pflash_cosine_score: cap ({cap}) must be >= n_pos ({n_pos})"
        )));
    }
    if scores.dtype() != ScalarType::F32 {
        return Err(ffi_error(format!(
            "pflash_cosine_score: scores must be ScalarType::F32, got {:?}",
            scores.dtype()
        )));
    }
    let n_blocks = (n_pos + block_size - 1) / block_size;
    if scores.elem_count() < n_blocks {
        return Err(ffi_error(format!(
            "pflash_cosine_score: scores has {} elems, expected >= {}",
            scores.elem_count(),
            n_blocks
        )));
    }
    let expected_k = kv_heads * cap * head_dim;
    if k_cache.elem_count() < expected_k {
        return Err(ffi_error(format!(
            "pflash_cosine_score: k_cache has {} elems, expected >= {} ([1, {}, {}, {}])",
            k_cache.elem_count(),
            expected_k,
            kv_heads,
            cap,
            head_dim
        )));
    }
    let status = unsafe {
        supersonic_qwen35_hip_pflash_cosine_score(
            dtype.kernel_dtype_code(),
            ordinal,
            n_pos,
            kv_heads,
            cap,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
            k_cache.as_ptr(),
            scores.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!("pflash_cosine_score failed: {status}")));
    }
    Ok(())
}

// ---- Transpose [S,H,D] <-> [H,S,D] ----

/// Transpose tensor from [S, H, D] layout to [H, S, D] layout.
pub fn transpose_shd_hsd(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    transpose_shd_hsd_impl(ordinal, dtype, s, h, d, src, dst, None)
}

#[allow(clippy::too_many_arguments)]
pub fn transpose_shd_hsd_with_options(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    transpose_shd_hsd_impl(ordinal, dtype, s, h, d, src, dst, Some(options))
}

#[allow(clippy::too_many_arguments)]
fn transpose_shd_hsd_impl(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if dst.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_transpose_shd_hsd,
                metal_force_host_transpose_shd_hsd,
            )
        {
            let result = prefill_metal_profile_time(options, "transpose_shd_hsd", "native", || {
                metal_native::transpose_shd_hsd(dtype, s, h, d, src, dst)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "transpose_shd_hsd", || {
            metal_host::transpose_shd_hsd(dtype, s, h, d, src, dst)
        });
    }
    ffi_profile_time_result("qwen.transpose_shd_hsd", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_transpose_shd_hsd(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                h,
                d,
                src.as_ptr(),
                dst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("transpose_shd_hsd failed: {status}")));
        }
        Ok(())
    })
}

/// Transpose two tensors from [S, H, D] layout to [H, S, D] layout in one HIP launch.
pub fn transpose_shd_hsd_pair(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src_a: &GpuBuffer,
    src_b: &GpuBuffer,
    dst_a: &mut GpuBuffer,
    dst_b: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dst_a.backend() == Backend::Metal || dst_b.backend() == Backend::Metal {
        transpose_shd_hsd(ordinal, dtype, s, h, d, src_a, dst_a)?;
        return transpose_shd_hsd(ordinal, dtype, s, h, d, src_b, dst_b);
    }
    ffi_profile_time_result("qwen.transpose_shd_hsd_pair", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_transpose_shd_hsd_pair(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                h,
                d,
                src_a.as_ptr(),
                src_b.as_ptr(),
                dst_a.as_mut_ptr(),
                dst_b.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "transpose_shd_hsd_pair failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Transpose BF16 from [S, H, D] directly into a cache laid out as
/// [H, cache_len, D] at rows [dst_pos, dst_pos + S).
pub fn transpose_shd_to_cache_bf16(
    ordinal: usize,
    s: usize,
    h: usize,
    d: usize,
    cache_len: usize,
    dst_pos: usize,
    src: &GpuBuffer,
    cache: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if cache.backend() == Backend::Metal {
        return Err(ffi_error(
            "transpose_shd_to_cache_bf16 is currently implemented only for HIP/CUDA".into(),
        ));
    }
    if src.dtype() != ScalarType::BF16 || cache.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "transpose_shd_to_cache_bf16 expects BF16 buffers; got src={:?} cache={:?}",
            src.dtype(),
            cache.dtype()
        )));
    }
    if dst_pos.checked_add(s).is_none_or(|end| end > cache_len) {
        return Err(GpuError::InvalidArg(format!(
            "transpose_shd_to_cache_bf16 dst range [{dst_pos}, {}) exceeds cache_len {cache_len}",
            dst_pos + s
        )));
    }
    ffi_profile_time_result("qwen.transpose_shd_to_cache_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_transpose_shd_to_cache_bf16(
                ordinal,
                s,
                h,
                d,
                cache_len,
                dst_pos,
                src.as_ptr(),
                cache.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "transpose_shd_to_cache_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

// ---- Transpose + pad for conv input ----

/// Transpose [S, C] -> [C, pad + S] with zero-padding on the left.
/// Used to prepare QKV projection output for causal conv1d.
pub fn transpose_pad_conv(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    c: usize,
    pad: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dst.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("transpose_pad_conv", "native", || {
                metal_native::transpose_pad_conv(dtype, s, c, pad, src, dst)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("transpose_pad_conv", || {
            metal_host::transpose_pad_conv(dtype, s, c, pad, src, dst)
        });
    }
    ffi_profile_time_result("qwen.transpose_pad_conv", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_transpose_pad_conv(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                c,
                pad,
                src.as_ptr(),
                dst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("transpose_pad_conv failed: {status}")));
        }
        Ok(())
    })
}

// ---- Extract conv state after prefill ----

/// Extract the last (kern-1) values per channel from [S, C] into [C, kern-1].
pub fn extract_conv_state(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    c: usize,
    kern_minus_1: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dst.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("extract_conv_state", "native", || {
                metal_native::extract_conv_state(dtype, s, c, kern_minus_1, src, dst)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("extract_conv_state", || {
            metal_host::extract_conv_state(dtype, s, c, kern_minus_1, src, dst)
        });
    }
    ffi_profile_time_result("qwen.extract_conv_state", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_extract_conv_state(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                c,
                kern_minus_1,
                src.as_ptr(),
                dst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("extract_conv_state failed: {status}")));
        }
        Ok(())
    })
}

/// Prepare `[C, pad + S]` conv input from an existing tail and current `[S, C]`
/// QKV rows, while also extracting the next `[C, pad]` tail.
pub fn prepare_conv_input_tail(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    c: usize,
    pad: usize,
    src: &GpuBuffer,
    old_tail: &GpuBuffer,
    conv_input: &mut GpuBuffer,
    new_tail: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_input.backend() == Backend::Metal {
        return Err(ffi_error(
            "prepare_conv_input_tail is only implemented for HIP/CUDA backends".to_string(),
        ));
    }
    ffi_profile_time_result("qwen.prepare_conv_input_tail", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_prepare_conv_input_tail(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                c,
                pad,
                src.as_ptr(),
                old_tail.as_ptr(),
                conv_input.as_mut_ptr(),
                new_tail.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "prepare_conv_input_tail failed: {status}"
            )));
        }
        Ok(())
    })
}

// ---- Sigmoid-gate multiply ----

/// out[i] = data[i] * sigmoid(gate[i]). Fused for gated attention.
pub fn sigmoid_mul(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    sigmoid_mul_impl(ordinal, dtype, total_elems, data, gate, out, None)
}

pub fn sigmoid_mul_with_options(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    sigmoid_mul_impl(ordinal, dtype, total_elems, data, gate, out, Some(options))
}

fn sigmoid_mul_impl(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options) {
            let result = prefill_metal_profile_time(options, "sigmoid_mul", "native", || {
                metal_native::sigmoid_mul(dtype, total_elems, data, gate, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "sigmoid_mul", || {
            metal_host::sigmoid_mul(dtype, total_elems, data, gate, out)
        });
    }
    ffi_profile_time_result("qwen.sigmoid_mul", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_sigmoid_mul(
                dtype.kernel_dtype_code(),
                ordinal,
                total_elems,
                data.as_ptr(),
                gate.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("sigmoid_mul failed: {status}")));
        }
        Ok(())
    })
}

/// In-place variant of [`sigmoid_mul`]. Safe for HIP because each lane reads
/// and writes only its own `data[idx]` element.
pub fn sigmoid_mul_inplace(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    data: &mut GpuBuffer,
    gate: &GpuBuffer,
) -> Result<(), GpuError> {
    if data.backend() == Backend::Metal {
        return Err(ffi_error(
            "sigmoid_mul_inplace is currently implemented only for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.sigmoid_mul_inplace", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_sigmoid_mul(
                dtype.kernel_dtype_code(),
                ordinal,
                total_elems,
                data.as_ptr(),
                gate.as_ptr(),
                data.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("sigmoid_mul_inplace failed: {status}")));
        }
        Ok(())
    })
}

/// Fused full-attention post processing for HIP:
/// F32 `[heads, S, D]` attention output -> BF16 `[S, heads, D]`, then gate.
pub fn cast_transpose_gate_hsd_to_shd_bf16(
    ordinal: usize,
    s: usize,
    heads: usize,
    head_dim: usize,
    attn_hsd: &GpuBuffer,
    gate_shd: &GpuBuffer,
    out_shd: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out_shd.backend() == Backend::Metal {
        return Err(ffi_error(
            "cast_transpose_gate_hsd_to_shd_bf16 is currently implemented only for HIP".into(),
        ));
    }
    if attn_hsd.dtype() != ScalarType::F32
        || gate_shd.dtype() != ScalarType::BF16
        || out_shd.dtype() != ScalarType::BF16
    {
        return Err(ffi_error(format!(
            "cast_transpose_gate_hsd_to_shd_bf16 expects F32 attention, BF16 gate/out; got {:?}, {:?}, {:?}",
            attn_hsd.dtype(),
            gate_shd.dtype(),
            out_shd.dtype()
        )));
    }
    ffi_profile_time_result("qwen.cast_transpose_gate_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_cast_transpose_gate_bf16(
                ordinal,
                s,
                heads,
                head_dim,
                attn_hsd.as_ptr(),
                gate_shd.as_ptr(),
                out_shd.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "cast_transpose_gate_hsd_to_shd_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

/// BF16 row-wise scalar gate: `out[row, col] = data[row, col] * sigmoid(row_gate[row])`.
pub fn sigmoid_mul_row_scalar_bf16(
    ordinal: usize,
    rows: usize,
    cols: usize,
    data: &GpuBuffer,
    row_gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    sigmoid_mul_row_scalar_bf16_impl(rows, cols, data, row_gate, out, None)
}

pub fn sigmoid_mul_row_scalar_bf16_with_options(
    ordinal: usize,
    rows: usize,
    cols: usize,
    data: &GpuBuffer,
    row_gate: &GpuBuffer,
    out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    let _ = ordinal;
    sigmoid_mul_row_scalar_bf16_impl(rows, cols, data, row_gate, out, Some(options))
}

fn sigmoid_mul_row_scalar_bf16_impl(
    rows: usize,
    cols: usize,
    data: &GpuBuffer,
    row_gate: &GpuBuffer,
    out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if out.backend() != Backend::Metal {
        return Err(GpuError::backend(
            out.backend(),
            "sigmoid_mul_row_scalar_bf16 is currently implemented only for Metal".into(),
        ));
    }
    if !prefill_native_disabled(options) {
        let result =
            prefill_metal_profile_time(options, "sigmoid_mul_row_scalar", "native", || {
                metal_native::sigmoid_mul_row_scalar_bf16(rows, cols, data, row_gate, out)
            });
        if result.is_ok() {
            return result;
        }
    }
    prefill_metal_profile_host_time(options, "sigmoid_mul_row_scalar", || {
        metal_host::sigmoid_mul_row_scalar_bf16(rows, cols, data, row_gate, out)
    })
}

// ---- Compute beta/g for delta recurrent ----

/// Compute beta = sigmoid(B) and g = -softplus(A + dt_bias) * a_log_exp.
/// Inputs: B [S, nv], A [S, nv] in dtype; dt_bias [nv], a_log_exp [nv] in dtype.
/// Outputs: beta [nv, S], g [nv, S] in dtype (transposed for delta recurrent).
pub fn compute_beta_g(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    nv: usize,
    b: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    beta: &mut GpuBuffer,
    g: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if beta.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::F32 && !metal_native::disabled_by_env() {
            let result = metal_profile_time("compute_beta_g", "native", || {
                metal_native::compute_beta_g_f32(seq_len, nv, b, a, dt_bias, a_log_exp, beta, g)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("compute_beta_g", || {
            metal_host::compute_beta_g(dtype, seq_len, nv, b, a, dt_bias, a_log_exp, beta, g)
        });
    }
    ffi_profile_time_result("qwen.compute_beta_g", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_compute_beta_g(
                dtype.kernel_dtype_code(),
                ordinal,
                seq_len,
                nv,
                b.as_ptr(),
                a.as_ptr(),
                dt_bias.as_ptr(),
                a_log_exp.as_ptr(),
                beta.as_mut_ptr(),
                g.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("compute_beta_g failed: {status}")));
        }
        Ok(())
    })
}

pub fn compute_beta_g_ba_bf16(
    ordinal: usize,
    seq_len: usize,
    nv: usize,
    ba: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    beta: &mut GpuBuffer,
    g: &mut GpuBuffer,
) -> Result<(), GpuError> {
    ffi_profile_time_result("qwen.compute_beta_g_ba_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_compute_beta_g_ba_bf16(
                ordinal,
                seq_len,
                nv,
                ba.as_ptr(),
                dt_bias.as_ptr(),
                a_log_exp.as_ptr(),
                beta.as_mut_ptr(),
                g.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "compute_beta_g_ba_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

pub fn project_ba_compute_beta_g_bf16(
    ordinal: usize,
    seq_len: usize,
    hidden_dim: usize,
    nv: usize,
    hidden: &GpuBuffer,
    ba_weight: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    beta: &mut GpuBuffer,
    g: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if hidden.backend() == Backend::Metal {
        return Err(ffi_error(
            "project_ba_compute_beta_g_bf16 is only implemented for HIP/CUDA backends".into(),
        ));
    }
    if hidden.dtype() != ScalarType::BF16
        || ba_weight.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
        || beta.dtype() != ScalarType::F32
        || g.dtype() != ScalarType::F32
    {
        return Err(ffi_error(format!(
            "project_ba_compute_beta_g_bf16 expects BF16 hidden/weights/biases and F32 beta/g, got hidden={:?} ba={:?} dt={:?} a_log={:?} beta={:?} g={:?}",
            hidden.dtype(),
            ba_weight.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            beta.dtype(),
            g.dtype()
        )));
    }
    ffi_profile_time_result("qwen.project_ba_compute_beta_g_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_project_ba_compute_beta_g_bf16(
                ordinal,
                seq_len,
                hidden_dim,
                nv,
                hidden.as_ptr(),
                ba_weight.as_ptr(),
                dt_bias.as_ptr(),
                a_log_exp.as_ptr(),
                beta.as_mut_ptr(),
                g.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "project_ba_compute_beta_g_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

// ---- Split gated Q projection ----

/// Split [S, num_heads, 2*head_dim] into query [S, num_heads, head_dim] and gate [S, num_heads, head_dim].
pub fn split_qgate(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    split_qgate_impl(
        ordinal, dtype, s, num_heads, head_dim, src, query_out, gate_out, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn split_qgate_with_options(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
    options: &PrefillFfiLaunchOptions,
) -> Result<(), GpuError> {
    split_qgate_impl(
        ordinal,
        dtype,
        s,
        num_heads,
        head_dim,
        src,
        query_out,
        gate_out,
        Some(options),
    )
}

#[allow(clippy::too_many_arguments)]
fn split_qgate_impl(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
    options: Option<&PrefillFfiLaunchOptions>,
) -> Result<(), GpuError> {
    if query_out.backend() == Backend::Metal {
        let _ = ordinal;
        if !prefill_native_disabled(options)
            && !prefill_launch_flag(
                options,
                |options| options.force_host_split_qgate,
                metal_force_host_split_qgate,
            )
        {
            let result = prefill_metal_profile_time(options, "split_qgate", "native", || {
                metal_native::split_qgate(dtype, s, num_heads, head_dim, src, query_out, gate_out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return prefill_metal_profile_host_time(options, "split_qgate", || {
            metal_host::split_qgate(dtype, s, num_heads, head_dim, src, query_out, gate_out)
        });
    }
    ffi_profile_time_result("qwen.split_qgate", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_qgate(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                num_heads,
                head_dim,
                src.as_ptr(),
                query_out.as_mut_ptr(),
                gate_out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_qgate failed: {status}")));
        }
        Ok(())
    })
}

/// Split gated full-attention Q projection and plain-RMSNorm the query rows.
///
/// Inputs are BF16 `[S, heads, 2*head_dim]`; outputs are BF16 query and gate
/// buffers in `[S, heads, head_dim]`. This matches `split_qgate` followed by
/// `rms_norm_rows_plain(..., rows=S*heads, cols=head_dim)` on the query side.
pub fn split_qgate_norm_bf16(
    ordinal: usize,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    src: &GpuBuffer,
    norm_w: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_out.backend() == Backend::Metal {
        return Err(ffi_error(
            "split_qgate_norm_bf16 is currently implemented only for HIP/CUDA".into(),
        ));
    }
    if src.dtype() != ScalarType::BF16
        || norm_w.dtype() != ScalarType::BF16
        || query_out.dtype() != ScalarType::BF16
        || gate_out.dtype() != ScalarType::BF16
    {
        return Err(ffi_error(format!(
            "split_qgate_norm_bf16 expects BF16 buffers; got src={:?} norm={:?} query={:?} gate={:?}",
            src.dtype(),
            norm_w.dtype(),
            query_out.dtype(),
            gate_out.dtype()
        )));
    }
    ffi_profile_time_result("qwen.split_qgate_norm_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_qgate_norm_bf16(
                ordinal,
                s,
                num_heads,
                head_dim,
                eps,
                src.as_ptr(),
                norm_w.as_ptr(),
                query_out.as_mut_ptr(),
                gate_out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_qgate_norm_bf16 failed: {status}")));
        }
        Ok(())
    })
}

// ---- Split interleaved QKV ----

/// Split [S, qkv_dim] where qkv_dim = [Q(key_dim) | K(key_dim) | V(val_dim)]
/// into separate Q [S, key_dim], K [S, key_dim], V [S, val_dim].
pub fn split_qkv(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    key_dim: usize,
    val_dim: usize,
    src: &GpuBuffer,
    q: &mut GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if q.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_split_qkv() {
            let result = metal_profile_time("split_qkv", "native", || {
                metal_native::split_qkv(dtype, s, key_dim, val_dim, src, q, k, v)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("split_qkv", || {
            metal_host::split_qkv(dtype, s, key_dim, val_dim, src, q, k, v)
        });
    }
    ffi_profile_time_result("qwen.split_qkv", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_qkv(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_qkv failed: {status}")));
        }
        Ok(())
    })
}

pub fn split_qkv_bf16_to_f32(
    ordinal: usize,
    s: usize,
    key_dim: usize,
    val_dim: usize,
    src: &GpuBuffer,
    q: &mut GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if q.backend() != Backend::Hip {
        return Err(ffi_error(
            "split_qkv_bf16_to_f32 is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.split_qkv_bf16_to_f32", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_qkv_bf16_to_f32(
                ordinal,
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_qkv_bf16_to_f32 failed: {status}")));
        }
        Ok(())
    })
}

pub fn split_kv_bf16(
    ordinal: usize,
    s: usize,
    kv_dim: usize,
    src: &GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if k.backend() != Backend::Hip {
        return Err(ffi_error(
            "split_kv_bf16 is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.split_kv_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_kv_bf16(
                ordinal,
                s,
                kv_dim,
                src.as_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_kv_bf16 failed: {status}")));
        }
        Ok(())
    })
}

pub fn split_norm_transpose_qkv_bf16(
    ordinal: usize,
    s: usize,
    nk: usize,
    nv: usize,
    khd: usize,
    vhd: usize,
    q_scale: f32,
    eps: f32,
    src: &GpuBuffer,
    q: &mut GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if q.backend() != Backend::Hip {
        return Err(ffi_error(
            "split_norm_transpose_qkv_bf16 is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.split_norm_transpose_qkv_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_norm_transpose_qkv_bf16(
                ordinal,
                s,
                nk,
                nv,
                khd,
                vhd,
                q_scale,
                eps,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "split_norm_transpose_qkv_bf16 failed: {status}"
            )));
        }
        Ok(())
    })
}

pub fn split_qkvz_bf16(
    ordinal: usize,
    s: usize,
    qkv_dim: usize,
    z_dim: usize,
    src: &GpuBuffer,
    qkv: &mut GpuBuffer,
    z: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if qkv.backend() != Backend::Hip {
        return Err(ffi_error(
            "split_qkvz_bf16 is only implemented for HIP".into(),
        ));
    }
    ffi_profile_time_result("qwen.split_qkvz_bf16", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_split_qkvz_bf16(
                ordinal,
                s,
                qkv_dim,
                z_dim,
                src.as_ptr(),
                qkv.as_mut_ptr(),
                z.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("split_qkvz_bf16 failed: {status}")));
        }
        Ok(())
    })
}

// ---- Repeat interleave heads ----

/// Repeat each head `repeats` times: [S, n_heads, head_dim] → [S, n_heads * repeats, head_dim].
/// Used for GQA-style head expansion in linear attention when nk != nv.
pub fn repeat_interleave_heads(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    n_heads: usize,
    head_dim: usize,
    repeats: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dst.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("repeat_interleave_heads", "native", || {
                metal_native::repeat_interleave_heads(
                    dtype, s, n_heads, head_dim, repeats, src, dst,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("repeat_interleave_heads", || {
            metal_host::repeat_interleave_heads(dtype, s, n_heads, head_dim, repeats, src, dst)
        });
    }
    ffi_profile_time_result("qwen.repeat_interleave_heads", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_repeat_interleave_heads(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                n_heads,
                head_dim,
                repeats,
                src.as_ptr(),
                dst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "repeat_interleave_heads failed: {status}"
            )));
        }
        Ok(())
    })
}

pub fn repeat_interleave_transpose_hsd(
    ordinal: usize,
    dtype: ScalarType,
    s: usize,
    n_heads: usize,
    head_dim: usize,
    repeats: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dst.backend() == Backend::Metal {
        let mut expanded = GpuBuffer::zeros(ordinal, dtype, &[s, n_heads * repeats, head_dim])?;
        repeat_interleave_heads(
            ordinal,
            dtype,
            s,
            n_heads,
            head_dim,
            repeats,
            src,
            &mut expanded,
        )?;
        return transpose_shd_hsd(
            ordinal,
            dtype,
            s,
            n_heads * repeats,
            head_dim,
            &expanded,
            dst,
        );
    }
    ffi_profile_time_result("qwen.repeat_interleave_transpose_hsd", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_repeat_interleave_transpose_hsd(
                dtype.kernel_dtype_code(),
                ordinal,
                s,
                n_heads,
                head_dim,
                repeats,
                src.as_ptr(),
                dst.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error(format!(
                "repeat_interleave_transpose_hsd failed: {status}"
            )));
        }
        Ok(())
    })
}

/// Quantize BF16 K or V tensor to FP8 E4M3 KV cache with per-head-per-position absmax scaling.
/// src: contiguous [num_kv_heads, seq_len, head_dim] BF16
/// dst_fp8: KV cache [num_kv_heads, max_T, head_dim] U8 (written at positions pos_offset..pos_offset+seq_len)
/// dst_scale: scale buffer [num_kv_heads, max_T] F32 (written at same positions)
pub fn quantize_kv_to_fp8(
    ordinal: usize,
    dtype: ScalarType,
    src: &GpuBuffer,
    dst_fp8: &mut GpuBuffer,
    dst_scale: &mut GpuBuffer,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    max_t: usize,
    pos_offset: usize,
) -> Result<(), GpuError> {
    ffi_profile_time_result("qwen.quantize_kv_to_fp8", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_4b_hip_quantize_kv_to_fp8(
                dtype.kernel_dtype_code(),
                ordinal,
                src.as_ptr(),
                dst_fp8.as_mut_ptr(),
                dst_scale.as_mut_ptr() as *mut c_void,
                num_kv_heads as c_int,
                seq_len as c_int,
                head_dim as c_int,
                max_t as c_int,
                pos_offset as c_int,
            )
        };
        if status != 0 {
            return Err(ffi_error(format!("quantize_kv_to_fp8 failed: {status}")));
        }
        Ok(())
    })
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};
    use half::bf16;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16 buffer");
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    #[test]
    fn metal_prefill_rms_norm_rows_applies_qwen_unit_offset() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2],
            &bf16_bytes(&[3.0, 4.0]),
        )
        .expect("upload input");
        let weight =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2], &bf16_bytes(&[0.5, -0.5]))
                .expect("upload weight");
        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2]).expect("alloc output");

        rms_norm_rows(
            ordinal,
            ScalarType::BF16,
            1,
            2,
            0.0,
            &input,
            &weight,
            &mut output,
        )
        .expect("run rms_norm_rows");

        let actual = read_bf16(&output);
        let inv_rms = 1.0f32 / ((25.0f32 / 2.0).sqrt());
        let expected = vec![3.0 * inv_rms * 1.5, 4.0 * inv_rms * 0.5];
        for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (got - want).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {want}, got {got}, delta {delta}"
            );
        }
    }
}
