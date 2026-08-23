//! FFI bindings for prefill kernels.
//! These are component kernels (not megakernels) — the prefill engine
//! orchestrates them layer by layer.

use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

fn ffi_error(msg: String) -> GpuError {
    GpuError::backend(Backend::Hip, msg)
}

static FFI_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static FFI_PROFILE: OnceLock<Mutex<FfiProfileAccumulator>> = OnceLock::new();

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

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefillBridgeStatus(i32);

impl PrefillBridgeStatus {
    // Native failures set bit 31, keep the project status in bits 16..30,
    // and preserve the backend's 16-bit runtime status in bits 0..15.
    const BACKEND_FAILURE_BIT: u32 = 1 << 31;

    const fn project_status(self) -> i32 {
        let raw = self.0 as u32;
        if raw & Self::BACKEND_FAILURE_BIT == 0 {
            self.0
        } else {
            ((raw >> 16) & 0x7fff) as i32
        }
    }

    const fn native_status(self) -> i32 {
        let raw = self.0 as u32;
        if raw & Self::BACKEND_FAILURE_BIT == 0 {
            0
        } else {
            (raw & 0xffff) as i32
        }
    }
}

fn prefill_bridge_result(
    backend: Backend,
    operation: &str,
    raw_status: i32,
) -> Result<(), GpuError> {
    let status = PrefillBridgeStatus(raw_status);
    let native_status = status.native_status();
    if native_status != 0 {
        return Err(GpuError::backend_status_in(
            backend,
            gpu_hal::BackendApi::Runtime,
            operation,
            native_status,
        ));
    }
    let project_status = status.project_status();
    if project_status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("{operation} failed with project status {project_status}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod typed_bridge_status_tests {
    use super::*;

    unsafe extern "C" {
        fn supersonic_prefill_encode_bridge_status(
            project_status: c_int,
            native_status: c_int,
        ) -> c_int;

        fn supersonic_qwen35_4b_bf16_matmul_bridge_status(
            project_status: c_int,
            native_status: c_int,
        ) -> c_int;
    }

    fn encoded_status(project_status: i32, native_status: i32) -> i32 {
        unsafe { supersonic_prefill_encode_bridge_status(project_status, native_status) }
    }

    #[test]
    fn generic_prefill_status_conversion_preserves_native_and_project_failures() {
        let native = prefill_bridge_result(Backend::Hip, "rms_norm_rows", encoded_status(302, 709))
            .unwrap_err();
        assert!(matches!(
            native,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 709,
            } if operation == "rms_norm_rows"
        ));

        let ordinary = prefill_bridge_result(Backend::Hip, "rms_norm_rows", encoded_status(301, 1))
            .unwrap_err();
        assert!(matches!(
            ordinary,
            GpuError::BackendStatus {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 1,
            } if operation == "rms_norm_rows"
        ));

        let validation =
            prefill_bridge_result(Backend::Hip, "rms_norm_rows", encoded_status(340, 0))
                .unwrap_err();
        assert!(matches!(
            validation,
            GpuError::Backend {
                backend: Backend::Hip,
                ref message,
            } if message == "rms_norm_rows failed with project status 340"
        ));
    }

    #[test]
    fn bf16_matmul_production_branch_status_preserves_native_and_project_failures() {
        let branch_status = |project_status, native_status| unsafe {
            supersonic_qwen35_4b_bf16_matmul_bridge_status(project_status, native_status)
        };

        let device_lost = prefill_bridge_result(
            Backend::Hip,
            "matmul_rhs_transposed",
            branch_status(280, 709),
        )
        .unwrap_err();
        assert!(matches!(
            device_lost,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 709,
            } if operation == "matmul_rhs_transposed"
        ));

        let ordinary =
            prefill_bridge_result(Backend::Hip, "matmul_rhs_transposed", branch_status(275, 1))
                .unwrap_err();
        assert!(matches!(
            ordinary,
            GpuError::BackendStatus {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 1,
            } if operation == "matmul_rhs_transposed"
        ));

        let project =
            prefill_bridge_result(Backend::Hip, "matmul_rhs_transposed", branch_status(272, 0))
                .unwrap_err();
        assert!(matches!(
            project,
            GpuError::Backend {
                backend: Backend::Hip,
                ref message,
            } if message == "matmul_rhs_transposed failed with project status 272"
        ));
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

    fn supersonic_qwen35_hip_argmax_f32_as_bf16_rows(
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

    fn supersonic_qwen35_hip_fill_conv_tail(
        dtype: c_int,
        device_ordinal: usize,
        qkv_dim: usize,
        pad: usize,
        total_len: usize,
        tail: *const c_void,
        conv_input: *mut c_void,
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
    crate::gqh::gemm_flush();
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
    crate::gqh::gemm_flush();
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
    crate::gqh::gemm_flush();
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
pub fn fill_conv_tail(
    ordinal: usize,
    dtype: ScalarType,
    qkv_dim: usize,
    pad: usize,
    total_len: usize,
    tail: &GpuBuffer,
    conv_input: &mut GpuBuffer,
) -> Result<(), GpuError> {
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
    crate::gqh::gemm_flush();
    swiglu_mul_impl(ordinal, dtype, elem_count, gate, up, out)
}

fn swiglu_mul_impl(
    ordinal: usize,
    dtype: ScalarType,
    elem_count: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "swiglu_mul", status)?;
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "rms_norm_gated_sfirst_bf16",
            status,
        )?;
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
    matmul_rhs_transposed_impl(ordinal, dtype, batch_elems, m, n, k, lhs, rhs, out)
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
) -> Result<(), GpuError> {
    ffi_profile_time_result("qwen.matmul_rhs_transposed", ordinal, || {
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
        prefill_bridge_result(gpu_hal::current_backend(), "matmul_rhs_transposed", status)?;
        Ok(())
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
) -> Result<(), GpuError> {
    ffi_profile_time_result("qwen.matmul_rhs_transposed_int4", ordinal, || {
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "matmul_rhs_transposed_int4",
            status,
        )?;
        Ok(())
    })
}

/// GQH fused dequant-matmul used by the Qwen3.8 GGUF path.
///
/// `lhs` is BF16/F32 `[>=ncols, k]` (or a rank-1 `[k]` vector). `out` is
/// BF16/F32 `[>=ncols, n]`. Only the first `ncols` rows are read/written so
/// oversized prefill scratch is safe. Activations are cast to f32 for the
/// kernel, which must not reassociate the GQH scale products.
///
/// For `ncols > 8` (llama.cpp `MMVQ_MAX_BATCH_SIZE`) the fused per-token
/// matvec rereads the weight matrix once per token. Prefill instead
/// dequantizes once and uses the tiled BF16 GEMM, matching ggml's
/// dequant→hipBLAS fallback. Opt out with `SUPERSONIC_GQH_FORCE_FUSED=1`.
const GQH_FUSED_MAX_COLS: usize = 8;
const GQH_DEQUANT_GEMM_MAX_WEIGHT_BYTES: usize = 768 * 1024 * 1024;

fn gqh_force_fused() -> bool {
    std::env::var_os("SUPERSONIC_GQH_FORCE_FUSED").is_some()
}

pub fn matmul_rhs_transposed_gqh(
    ordinal: usize,
    ncols: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    tensor_scale: f32,
    grid_code: u8,
    rung: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16 && lhs.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matmul lhs must be bf16/f32, got {:?}",
            lhs.dtype()
        )));
    }
    if out.dtype() != ScalarType::BF16 && out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matmul out must be bf16/f32, got {:?}",
            out.dtype()
        )));
    }
    if ncols == 0 || k == 0 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matmul ncols={ncols} k={k} must be positive"
        )));
    }
    if lhs.elem_count() < ncols * k {
        return Err(GpuError::InvalidArg(format!(
            "gqh matmul lhs has {} elems, need {ncols}*{k}",
            lhs.elem_count()
        )));
    }
    if out.elem_count() < ncols * n {
        return Err(GpuError::InvalidArg(format!(
            "gqh matmul out has {} elems, need {}",
            out.elem_count(),
            ncols * n
        )));
    }
    let dequant_bytes = n.saturating_mul(k).saturating_mul(4);
    if ncols > GQH_FUSED_MAX_COLS
        && !gqh_force_fused()
        && dequant_bytes > 0
        && dequant_bytes <= GQH_DEQUANT_GEMM_MAX_WEIGHT_BYTES
    {
        return matmul_gqh_dequant_gemm(
            ordinal,
            ncols,
            n,
            k,
            lhs,
            rhs,
            tensor_scale,
            grid_code,
            rung,
            out,
        );
    }
    thread_local! {
        static GQH_X: std::cell::RefCell<Option<GpuBuffer>> = const { std::cell::RefCell::new(None) };
        static GQH_Y: std::cell::RefCell<Option<GpuBuffer>> = const { std::cell::RefCell::new(None) };
    }
    let take_scratch =
        |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
         elems: usize|
         -> Result<GpuBuffer, GpuError> {
            slot.with(|cell| {
                let mut held = cell.borrow_mut();
                if let Some(buf) = held.as_ref() {
                    if buf.elem_count() >= elems {
                        return Ok(held.take().expect("scratch present"));
                    }
                }
                held.take();
                GpuBuffer::zeros(ordinal, ScalarType::F32, &[elems])
            })
        };
    let put_scratch =
        |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
         buf: GpuBuffer| {
            slot.with(|cell| {
                *cell.borrow_mut() = Some(buf);
            });
        };

    let x_f32 = if lhs.dtype() == ScalarType::F32 {
        None
    } else {
        let mut buf = take_scratch(&GQH_X, ncols * k)?;
        cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            ncols * k,
            lhs,
            &mut buf,
        )?;
        Some(buf)
    };
    let x_ref = x_f32.as_ref().unwrap_or(lhs);
    let mut y_f32 = if out.dtype() == ScalarType::F32 {
        None
    } else {
        Some(take_scratch(&GQH_Y, ncols * n)?)
    };
    let y_ref = y_f32.as_mut().unwrap_or(out);
    crate::gqh::matvec(
        ordinal,
        rung,
        rhs,
        x_ref,
        y_ref,
        k,
        n,
        ncols,
        k,
        n,
        tensor_scale,
        grid_code,
    )?;
    if let Some(y) = y_f32.as_ref() {
        cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            ncols * n,
            y,
            out,
        )?;
    }
    if let Some(buf) = x_f32 {
        put_scratch(&GQH_X, buf);
    }
    if let Some(buf) = y_f32 {
        put_scratch(&GQH_Y, buf);
    }
    Ok(())
}

fn matmul_gqh_dequant_gemm(
    ordinal: usize,
    ncols: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    tensor_scale: f32,
    grid_code: u8,
    rung: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    thread_local! {
        static W_F32: std::cell::RefCell<Option<GpuBuffer>> =
            const { std::cell::RefCell::new(None) };
        static W_BF16: std::cell::RefCell<Option<GpuBuffer>> =
            const { std::cell::RefCell::new(None) };
    }
    let take = |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
                dtype: ScalarType,
                elems: usize|
     -> Result<GpuBuffer, GpuError> {
        slot.with(|cell| {
            let mut held = cell.borrow_mut();
            if let Some(buf) = held.as_ref() {
                if buf.dtype() == dtype && buf.elem_count() >= elems {
                    return Ok(held.take().expect("scratch present"));
                }
            }
            held.take();
            GpuBuffer::zeros(ordinal, dtype, &[elems])
        })
    };
    let put = |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
               buf: GpuBuffer| {
        slot.with(|cell| {
            *cell.borrow_mut() = Some(buf);
        });
    };

    let status = if lhs.dtype() == ScalarType::BF16 && out.dtype() == ScalarType::BF16 {
        crate::gqh::dequant_gemm_bf16(
            ordinal,
            rung,
            rhs,
            tensor_scale,
            grid_code,
            lhs,
            out,
            ncols,
            n,
            k,
        )
    } else {
        thread_local! {
            static X: std::cell::RefCell<Option<GpuBuffer>> =
                const { std::cell::RefCell::new(None) };
            static Y: std::cell::RefCell<Option<GpuBuffer>> =
                const { std::cell::RefCell::new(None) };
        }
        let mut w_f32 = take(&W_F32, ScalarType::F32, n * k)?;
        crate::gqh::decode(
            ordinal,
            rung,
            rhs,
            tensor_scale,
            grid_code,
            &mut w_f32,
            n,
            k,
        )?;
        let x_owned = if lhs.dtype() == ScalarType::F32 {
            None
        } else {
            let mut buf = take(&X, ScalarType::F32, ncols * k)?;
            cast(
                ordinal,
                ScalarType::BF16,
                ScalarType::F32,
                ncols * k,
                lhs,
                &mut buf,
            )?;
            Some(buf)
        };
        let x_ref = x_owned.as_ref().unwrap_or(lhs);
        let mut y_owned = if out.dtype() == ScalarType::F32 {
            None
        } else {
            Some(take(&Y, ScalarType::F32, ncols * n)?)
        };
        let y_ref = y_owned.as_mut().unwrap_or(out);
        let gemm = matmul_rhs_transposed(
            ordinal,
            ScalarType::F32,
            1,
            ncols,
            n,
            k,
            x_ref,
            &w_f32,
            y_ref,
        );
        if let Some(y) = y_owned.as_ref() {
            cast(
                ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                ncols * n,
                y,
                out,
            )?;
        }
        if let Some(buf) = x_owned {
            put(&X, buf);
        }
        if let Some(buf) = y_owned {
            put(&Y, buf);
        }
        put(&W_F32, w_f32);
        gemm
    };
    status
}

/// Mix qtype 105/106 fused dequant-matmul. Same scratch/cast convention as GQH.
pub fn matmul_rhs_transposed_mix(
    ordinal: usize,
    ncols: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    mode: i32,
    lut: &[f32; 16],
    qtype: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16 && lhs.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "mix matmul lhs must be bf16/f32, got {:?}",
            lhs.dtype()
        )));
    }
    if out.dtype() != ScalarType::BF16 && out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "mix matmul out must be bf16/f32, got {:?}",
            out.dtype()
        )));
    }
    if ncols == 0 || k == 0 {
        return Err(GpuError::InvalidArg(format!(
            "mix matmul ncols={ncols} k={k} must be positive"
        )));
    }
    thread_local! {
        static MIX_X: std::cell::RefCell<Option<GpuBuffer>> = const { std::cell::RefCell::new(None) };
        static MIX_Y: std::cell::RefCell<Option<GpuBuffer>> = const { std::cell::RefCell::new(None) };
    }
    let take_scratch =
        |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
         elems: usize|
         -> Result<GpuBuffer, GpuError> {
            slot.with(|cell| {
                let mut held = cell.borrow_mut();
                if let Some(buf) = held.as_ref() {
                    if buf.elem_count() >= elems {
                        return Ok(held.take().expect("scratch present"));
                    }
                }
                held.take();
                GpuBuffer::zeros(ordinal, ScalarType::F32, &[elems])
            })
        };
    let put_scratch =
        |slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<GpuBuffer>>>,
         buf: GpuBuffer| {
            slot.with(|cell| {
                *cell.borrow_mut() = Some(buf);
            });
        };
    let x_f32 = if lhs.dtype() == ScalarType::F32 {
        None
    } else {
        let mut buf = take_scratch(&MIX_X, ncols * k)?;
        cast(
            ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            ncols * k,
            lhs,
            &mut buf,
        )?;
        Some(buf)
    };
    let x_ref = x_f32.as_ref().unwrap_or(lhs);
    let mut y_f32 = if out.dtype() == ScalarType::F32 {
        None
    } else {
        Some(take_scratch(&MIX_Y, ncols * n)?)
    };
    let y_ref = y_f32.as_mut().unwrap_or(out);
    crate::gqh::mix_matvec(
        ordinal, qtype, rhs, x_ref, y_ref, k, n, ncols, false, mode, lut,
    )?;
    if let Some(y) = y_f32.as_ref() {
        cast(
            ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            ncols * n,
            y,
            out,
        )?;
    }
    if let Some(buf) = x_f32 {
        put_scratch(&MIX_X, buf);
    }
    if let Some(buf) = y_f32 {
        put_scratch(&MIX_Y, buf);
    }
    Ok(())
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
    crate::gqh::gemm_flush();
    rms_norm_rows_impl(ordinal, dtype, n_rows, n_cols, eps, input, weight, out)
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
) -> Result<(), GpuError> {
    ffi_profile_time_result("qwen.rms_norm_rows", ordinal, || {
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
        prefill_bridge_result(gpu_hal::current_backend(), "rms_norm_rows", status)?;
        Ok(())
    })
}

/// Multi-row RMSNorm WITHOUT add_unit_offset. Qwen3.8 recurrent path
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
    crate::gqh::gemm_flush();
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
        prefill_bridge_result(gpu_hal::current_backend(), "rms_norm_rows_plain", status)?;
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
    crate::gqh::gemm_flush();
    let profile_key = if std::env::var_os("SUPERSONIC_FFI_PROFILE_SHAPES").is_some() {
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "rms_norm_rows_plain_inplace",
            status,
        )?;
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
    element_add_inplace_impl(ordinal, dtype, total_elems, lhs_out, rhs)
}

fn element_add_inplace_impl(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    lhs_out: &mut GpuBuffer,
    rhs: &GpuBuffer,
) -> Result<(), GpuError> {
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
    prefill_bridge_result(gpu_hal::current_backend(), "element_add_inplace", status)?;
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
    crate::gqh::gemm_flush();
    cast_impl(ordinal, input_dtype, output_dtype, total_elems, input, out)
}

fn cast_impl(
    ordinal: usize,
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "cast", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(gpu_hal::current_backend(), "element_add", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "argmax_bf16_rows", status)?;
        Ok(())
    })
}

pub fn argmax_f32_as_bf16_rows(
    ordinal: usize,
    rows: usize,
    cols: usize,
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if gpu_hal::current_backend() != Backend::Hip {
        return Err(ffi_error(
            "argmax_f32_as_bf16_rows requires the HIP backend".to_string(),
        ));
    }
    if rows == 0 || cols == 0 {
        return Err(GpuError::InvalidArg(
            "argmax_f32_as_bf16_rows requires non-zero rows and cols".into(),
        ));
    }
    if logits.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "argmax_f32_as_bf16_rows logits must be F32, got {:?}",
            logits.dtype()
        )));
    }
    if logits.elem_count() < rows.saturating_mul(cols) {
        return Err(GpuError::InvalidArg(format!(
            "argmax_f32_as_bf16_rows logits has {} elems, need {}",
            logits.elem_count(),
            rows.saturating_mul(cols)
        )));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() < rows {
        return Err(GpuError::InvalidArg(format!(
            "argmax_f32_as_bf16_rows output must be U32[{rows}], got {:?}[{}]",
            out_index.dtype(),
            out_index.elem_count()
        )));
    }

    ffi_profile_time_result("qwen.argmax_f32_as_bf16_rows", ordinal, || {
        let status = unsafe {
            supersonic_qwen35_hip_argmax_f32_as_bf16_rows(
                ordinal,
                rows,
                cols,
                logits.as_ptr(),
                out_index.as_mut_ptr(),
            )
        };
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "argmax_f32_as_bf16_rows",
            status,
        )?;
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
    crate::gqh::gemm_flush();
    apply_rope_prefill_impl(
        ordinal, dtype, seq_len, num_heads, head_dim, rotary_dim, cos_table, sin_table, pos_offset,
        data,
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
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "apply_rope_prefill", status)?;
        Ok(())
    })
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
    crate::gqh::gemm_flush();
    transpose_shd_hsd_impl(ordinal, dtype, s, h, d, src, dst)
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
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "transpose_shd_hsd", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "transpose_shd_hsd_pair", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "transpose_shd_to_cache_bf16",
            status,
        )?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(gpu_hal::current_backend(), "transpose_pad_conv", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "extract_conv_state", status)?;
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "prepare_conv_input_tail",
            status,
        )?;
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
    sigmoid_mul_impl(ordinal, dtype, total_elems, data, gate, out)
}

fn sigmoid_mul_impl(
    ordinal: usize,
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "sigmoid_mul", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "sigmoid_mul_inplace", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "cast_transpose_gate_hsd_to_shd_bf16",
            status,
        )?;
        Ok(())
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
        prefill_bridge_result(gpu_hal::current_backend(), "compute_beta_g", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "compute_beta_g_ba_bf16", status)?;
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "project_ba_compute_beta_g_bf16",
            status,
        )?;
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
    crate::gqh::gemm_flush();
    split_qgate_impl(
        ordinal, dtype, s, num_heads, head_dim, src, query_out, gate_out,
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
) -> Result<(), GpuError> {
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_qgate", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_qgate_norm_bf16", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_qkv", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_qkv_bf16_to_f32", status)?;
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_kv_bf16", status)?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "split_norm_transpose_qkv_bf16",
            status,
        )?;
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
    crate::gqh::gemm_flush();
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
        prefill_bridge_result(gpu_hal::current_backend(), "split_qkvz_bf16", status)?;
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "repeat_interleave_heads",
            status,
        )?;
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
        prefill_bridge_result(
            gpu_hal::current_backend(),
            "repeat_interleave_transpose_hsd",
            status,
        )?;
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
