//! FFI bindings for prefill kernels.
//! These are component kernels (not megakernels) — the prefill engine
//! orchestrates them layer by layer.

use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::{metal_host, metal_native};
use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

static METAL_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static METAL_PROFILE: OnceLock<Mutex<MetalProfileAccumulator>> = OnceLock::new();

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

pub fn metal_profile_set_enabled(enabled: bool) {
    METAL_PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn metal_profile_enabled() -> bool {
    METAL_PROFILE_ENABLED.load(Ordering::Relaxed)
        || std::env::var_os("SUPERSONIC_METAL_PROFILE").is_some()
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
    if !metal_profile_enabled() {
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

fn metal_profile_host_time<T, F>(op: &'static str, f: F) -> Result<T, GpuError>
where
    F: FnOnce() -> Result<T, GpuError>,
{
    metal_native::flush_batch()?;
    metal_profile_time(op, "host", f)
}

pub fn flush_metal_batch() -> Result<(), GpuError> {
    metal_native::flush_batch()
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

pub fn metal_conv_state_update_bf16(
    channels: usize,
    state_len: usize,
    qkv: &GpuBuffer,
    state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    metal_native::conv_state_update_bf16(channels, state_len, qkv, state)
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
        Backend::Hip => GpuError::Hip(msg),
        Backend::Cuda => GpuError::Cuda(msg),
        Backend::Metal => GpuError::Metal(msg),
    }
}

unsafe extern "C" {
    // ---- Existing bridge functions (from full_attention_bridge.cpp) ----

    fn dotcache_qwen35_hip_rms_norm(
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

    fn dotcache_qwen35_hip_cast(
        input_dtype: c_int,
        output_dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        xs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    // ---- Prefill helper bridge functions (from prefill_helpers_bridge.cpp) ----

    fn dotcache_qwen35_hip_element_add(
        dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_apply_rope_prefill(
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

    fn dotcache_qwen35_hip_transpose_shd_hsd(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        h: usize,
        d: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_sigmoid_mul(
        dtype: c_int,
        device_ordinal: usize,
        total_elems: usize,
        data: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_compute_beta_g(
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

    fn dotcache_qwen35_hip_split_qgate(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        num_heads: usize,
        head_dim: usize,
        src: *const c_void,
        query_out: *mut c_void,
        gate_out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_split_qkv(
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

    fn dotcache_qwen35_hip_repeat_interleave_heads(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        n_heads: usize,
        head_dim: usize,
        repeats: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_transpose_pad_conv(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        c: usize,
        pad: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_extract_conv_state(
        dtype: c_int,
        device_ordinal: usize,
        s: usize,
        c: usize,
        kern_minus_1: usize,
        src: *const c_void,
        dst: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_fused_rms_norm_linear(
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
    fn dotcache_qwen35_4b_hip_matmul_rhs_transposed_tiled(
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
    fn dotcache_qwen35_4b_hip_matmul_fp8_dequant(
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
    fn dotcache_qwen35_4b_hip_matmul_int4_dequant(
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
        group_size: c_int,
        out: *mut c_void,
    ) -> c_int;

    // BF16 → FP8 KV cache quantization
    fn dotcache_qwen35_4b_hip_quantize_kv_to_fp8(
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

    fn dotcache_qwen35_hip_embedding_lookup(
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

    fn dotcache_qwen35_hip_batched_matmul(
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

    fn dotcache_qwen35_hip_full_attention_prefill(
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

    fn dotcache_qwen35_hip_linear_prefill_conv_pack(
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

    fn dotcache_qwen35_hip_linear_decode_prepare(
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

    fn dotcache_qwen35_hip_linear_decode_apply(
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        packed: *const c_void,
        initial_state: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_4b_hip_linear_decode_prepare(
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

    fn dotcache_qwen35_4b_hip_linear_decode_apply(
        device_ordinal: usize,
        batch_size: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        packed: *const c_void,
        initial_state: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_4b_hip_linear_stateful_conv_value_decay(
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

    fn dotcache_qwen35_hip_delta_recurrent_prefill(
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

    fn dotcache_qwen35_hip_l2norm(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_swiglu_mul(
        dtype: c_int,
        device_ordinal: usize,
        elem_count: usize,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_hip_rms_norm_gated(
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

    fn dotcache_qwen35_hip_mul_scalar(
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
    let status = unsafe {
        dotcache_qwen35_hip_embedding_lookup(
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
    let status = unsafe {
        dotcache_qwen35_hip_batched_matmul(
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
    let status = unsafe {
        dotcache_qwen35_hip_full_attention_prefill(
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
    let status = unsafe {
        dotcache_qwen35_hip_linear_prefill_conv_pack(
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
        dotcache_qwen35_hip_linear_decode_prepare(
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
        dotcache_qwen35_hip_linear_decode_apply(
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
        dotcache_qwen35_4b_hip_linear_decode_prepare(
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
        dotcache_qwen35_4b_hip_linear_decode_apply(
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
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_NATIVE_LINEAR_CONV_VALUE_DECAY")
                .is_none()
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
        dotcache_qwen35_4b_hip_linear_stateful_conv_value_decay(
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
    let status = unsafe {
        dotcache_qwen35_hip_delta_recurrent_prefill(
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
    let status = unsafe {
        dotcache_qwen35_hip_l2norm(
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
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("swiglu_mul", "native", || {
                metal_native::swiglu_mul(dtype, elem_count, gate, up, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("swiglu_mul", || {
            metal_host::swiglu_mul(dtype, elem_count, gate, up, out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_swiglu_mul(
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
    let status = unsafe {
        dotcache_qwen35_hip_rms_norm_gated(
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
    let status = unsafe {
        dotcache_qwen35_hip_mul_scalar(
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
            dotcache_qwen35_hip_fused_rms_norm_linear(
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
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && !metal_native::disabled_by_env()
            && !metal_force_host_matmul()
        {
            let result = metal_profile_time("matmul_rhs_transposed", "native", || {
                metal_native::matmul_rhs_transposed_bf16(batch_elems, m, n, k, lhs, rhs, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !metal_native::disabled_by_env()
            && !metal_force_host_matmul()
        {
            let result = metal_profile_time("matmul_rhs_transposed", "native", || {
                metal_native::matmul_rhs_transposed_f32(batch_elems, m, n, k, lhs, rhs, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("matmul_rhs_transposed", || {
            metal_host::matmul_rhs_transposed(dtype, batch_elems, m, n, k, lhs, rhs, out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_4b_hip_matmul_rhs_transposed_tiled(
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
    let status = unsafe {
        dotcache_qwen35_4b_hip_matmul_fp8_dequant(
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
    group_size: usize,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_qwen35_4b_hip_matmul_int4_dequant(
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
            group_size as c_int,
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "matmul_rhs_transposed_int4 failed: {status}"
        )));
    }
    Ok(())
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
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows", "native", || {
                metal_native::rms_norm_rows_bf16(n_rows, n_cols, eps, true, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        if dtype == ScalarType::F32
            && !metal_native::disabled_by_env()
            && !metal_force_host_rms_norm()
        {
            let result = metal_profile_time("rms_norm_rows", "native", || {
                metal_native::rms_norm_rows_f32(n_rows, n_cols, eps, true, input, weight, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("rms_norm_rows", || {
            metal_host::rms_norm_rows(dtype, n_rows, n_cols, eps, true, input, weight, out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_rms_norm(
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
    let status = unsafe {
        dotcache_qwen35_hip_rms_norm(
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
    let ptr = data.as_mut_ptr();
    let status = unsafe {
        dotcache_qwen35_hip_rms_norm(
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
    if lhs_out.backend() == Backend::Metal {
        let mut lhs = GpuBuffer::zeros(ordinal, dtype, lhs_out.shape())?;
        metal_native::flush_batch()?;
        gpu_hal::copy_d2d(
            ordinal,
            lhs.as_mut_ptr(),
            lhs_out.as_ptr(),
            lhs_out.len_bytes(),
        )?;
        if !metal_native::disabled_by_env() && !metal_force_host_element_add() {
            let result = metal_profile_time("element_add_inplace", "native", || {
                metal_native::element_add(dtype, total_elems, &lhs, rhs, lhs_out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("element_add_inplace", || {
            metal_host::element_add(dtype, total_elems, &lhs, rhs, lhs_out)
        });
    }
    let ptr = lhs_out.as_mut_ptr();
    let status = unsafe {
        dotcache_qwen35_hip_element_add(
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
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_cast() {
            let result = metal_profile_time("cast", "native", || {
                metal_native::cast(input_dtype, output_dtype, total_elems, input, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("cast", || {
            metal_host::cast(input_dtype, output_dtype, total_elems, input, out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_cast(
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
    let status = unsafe {
        dotcache_qwen35_hip_element_add(
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
    if data.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("apply_rope_prefill", "native", || {
                metal_native::apply_rope_prefill(
                    dtype, seq_len, num_heads, head_dim, rotary_dim, cos_table, sin_table,
                    pos_offset, data,
                )
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("apply_rope_prefill", || {
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
    let status = unsafe {
        dotcache_qwen35_hip_apply_rope_prefill(
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
    if dst.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_transpose_shd_hsd() {
            let result = metal_profile_time("transpose_shd_hsd", "native", || {
                metal_native::transpose_shd_hsd(dtype, s, h, d, src, dst)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("transpose_shd_hsd", || {
            metal_host::transpose_shd_hsd(dtype, s, h, d, src, dst)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_transpose_shd_hsd(
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
    let status = unsafe {
        dotcache_qwen35_hip_transpose_pad_conv(
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
    let status = unsafe {
        dotcache_qwen35_hip_extract_conv_state(
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
    if out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() {
            let result = metal_profile_time("sigmoid_mul", "native", || {
                metal_native::sigmoid_mul(dtype, total_elems, data, gate, out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("sigmoid_mul", || {
            metal_host::sigmoid_mul(dtype, total_elems, data, gate, out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_sigmoid_mul(
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
    let status = unsafe {
        dotcache_qwen35_hip_compute_beta_g(
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
    if query_out.backend() == Backend::Metal {
        let _ = ordinal;
        if !metal_native::disabled_by_env() && !metal_force_host_split_qgate() {
            let result = metal_profile_time("split_qgate", "native", || {
                metal_native::split_qgate(dtype, s, num_heads, head_dim, src, query_out, gate_out)
            });
            if result.is_ok() {
                return result;
            }
        }
        return metal_profile_host_time("split_qgate", || {
            metal_host::split_qgate(dtype, s, num_heads, head_dim, src, query_out, gate_out)
        });
    }
    let status = unsafe {
        dotcache_qwen35_hip_split_qgate(
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
    let status = unsafe {
        dotcache_qwen35_hip_split_qkv(
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
    let status = unsafe {
        dotcache_qwen35_hip_repeat_interleave_heads(
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
    let status = unsafe {
        dotcache_qwen35_4b_hip_quantize_kv_to_fp8(
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
