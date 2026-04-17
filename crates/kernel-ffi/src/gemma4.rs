//! FFI surface for the Gemma 4 decode primitives (`kernels/gemma4.hip`).
//!
//! Kept separate from the Qwen FFI (`ffi.rs`) because Gemma 4 has a different
//! layer shape (four RMSNorms per block, dual RoPE tables, optional PLE) and
//! its kernels are in their own compilation unit for hipcc codegen stability.
//!
//! Every wrapper here is intentionally a single-kernel launch — the goal for
//! the first correctness milestone is layer-by-layer validation against the
//! PyTorch oracle, not a fused persistent megakernel. Fusion comes later.

use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

unsafe extern "C" {
    fn dotcache_gemma4_hip_rms_norm(
        dtype: c_int,
        device_ordinal: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    fn dotcache_gemma4_hip_gelu_tanh_gate_mul(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_rope_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        x: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_swa_attn_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_t: usize,
        sliding_window: c_int,
        scale: f32,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        scores_scratch: *mut c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_kv_append(
        dtype: c_int,
        device_ordinal: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos: usize,
        max_t: usize,
        k_in: *const c_void,
        v_in: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
    ) -> c_int;
}

/// Gemma-variant RMSNorm — plain `weight * (x / sqrt(mean(x^2) + eps))` with
/// no `(w + 1)` offset (unlike Qwen's). Pass a null `weight` pointer via
/// `weight.is_none()` for `with_scale=False` (used by Gemma 4's `v_norm`).
pub fn rms_norm(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        dotcache_gemma4_hip_rms_norm(
            dtype.kernel_dtype_code(),
            ordinal,
            n_cols,
            eps,
            input.as_ptr(),
            weight_ptr,
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 rms_norm failed with status {status}"
        )));
    }
    Ok(())
}

/// Single-token matvec: `out = W @ x` where `W` is `[out_dim, in_dim]` row-major.
/// `counter_buf` must hold ≥4 mutable bytes — it's reset to 0 inside the call
/// and drives the work-stealing row assignment.
pub fn matvec(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_matvec(
            dtype.kernel_dtype_code(),
            ordinal,
            in_dim,
            out_dim,
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
            counter_buf.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 matvec failed with status {status}"
        )));
    }
    Ok(())
}

/// Elementwise `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]`. Matches Gemma 4's
/// `hidden_activation = "gelu_pytorch_tanh"` exactly.
pub fn gelu_tanh_gate_mul(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_gelu_tanh_gate_mul(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            gate.as_ptr(),
            up.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 gelu_tanh_gate_mul failed with status {status}"
        )));
    }
    Ok(())
}

/// Apply Gemma-style (split-half) RoPE to a single decode token's Q or K tensor
/// of shape `[num_heads, head_dim]`, using position `pos` from the cos/sin table.
/// `rotary_dim` may be less than `head_dim` for partial rotation (Gemma 4 full-
/// attention layers use `partial_rotary_factor=0.25`).
pub fn rope_decode(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_rope_decode(
            dtype.kernel_dtype_code(),
            ordinal,
            num_heads,
            head_dim,
            rotary_dim,
            position,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 rope_decode failed with status {status}"
        )));
    }
    Ok(())
}

/// Run sliding-window attention for one decode token.
///
/// The caller is responsible for having already appended the current token's
/// K and V into the caches (see [`kv_append`]) so that `kv_len` entries are
/// valid. `scores_scratch` must hold at least `num_q_heads * max_t * 4` bytes
/// of F32 storage; it is written and read inside the kernel but its contents
/// are not meaningful to the caller afterwards.
///
/// Pass `sliding_window <= 0` to attend over the whole cache (behaves as full
/// attention). Gemma 4 uses `scale = 1.0` (no 1/sqrt(d_k)).
#[allow(clippy::too_many_arguments)]
pub fn swa_attn_decode(
    ordinal: usize,
    dtype: ScalarType,
    q: &GpuBuffer,
    k_cache: &GpuBuffer,
    v_cache: &GpuBuffer,
    scores_scratch: &mut GpuBuffer,
    out: &mut GpuBuffer,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    max_t: usize,
    sliding_window: i32,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_swa_attn_decode(
            dtype.kernel_dtype_code(),
            ordinal,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            max_t,
            sliding_window as c_int,
            scale,
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            scores_scratch.as_mut_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 swa_attn_decode failed with status {status}"
        )));
    }
    Ok(())
}

/// Append a single decode token's K and V into pre-allocated caches of layout
/// `[num_kv_heads, max_T, head_dim]`. `pos` is the absolute position in the
/// cache (= token index in the sequence so far). Does not bounds-check pos
/// against `max_T`; caller must ensure capacity.
pub fn kv_append(
    ordinal: usize,
    dtype: ScalarType,
    k_in: &GpuBuffer,
    v_in: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_t: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_kv_append(
            dtype.kernel_dtype_code(),
            ordinal,
            num_kv_heads,
            head_dim,
            pos,
            max_t,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 kv_append failed with status {status}"
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Gemma4DecodeLayerDesc — a Rust-side descriptor of one decoder layer's
// weight pointers + per-layer state. Kept as plain data (no lifetimes) and
// `#[repr(C)]` so that when a future persistent megakernel wants to consume
// an array of these from the device, the Rust and HIP sides agree on layout.
//
// For this session the descriptor is populated by the caller and its fields
// are passed one primitive at a time to the single-kernel FFI above; no
// kernel consumes the struct yet.
// -----------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4DecodeLayerDesc {
    /// 0 = sliding_attention, 1 = full_attention.
    pub layer_type: c_int,
    /// MLP intermediate size (`intermediate_size` for dense layers, or
    /// `2 * intermediate_size` on `use_double_wide_mlp` variants).
    pub intermediate_size: c_int,
    /// `hidden_size` of the model (for convenience at kernel-call time).
    pub hidden_size: c_int,
    /// `head_dim` for this layer (256 for SWA, 512 for full on E2B).
    pub head_dim: c_int,
    pub num_q_heads: c_int,
    pub num_kv_heads: c_int,
    /// Number of `head_dim` columns that receive rotary (== head_dim for
    /// sliding layers, 0.25 * head_dim for full layers on E2B).
    pub rotary_dim: c_int,
    /// Sliding window size in tokens (only used for sliding layers).
    pub sliding_window: c_int,

    // --- Norms (all plain-Gemma, no `(w+1)` offset) ---
    pub input_norm_w: *const c_void,
    pub post_attn_norm_w: *const c_void,
    pub pre_ff_norm_w: *const c_void,
    pub post_ff_norm_w: *const c_void,
    pub q_norm_w: *const c_void,       // [head_dim]
    pub k_norm_w: *const c_void,       // [head_dim]
    // `v_norm` has no weight parameter (with_scale=False); caller passes null.
    pub norm_eps: f32,

    // --- Attention projections ---
    pub q_proj_w: *const c_void,       // [num_q_heads*head_dim, hidden]
    pub k_proj_w: *const c_void,       // [num_kv_heads*head_dim, hidden]
    pub v_proj_w: *const c_void,       // [num_kv_heads*head_dim, hidden]
    pub o_proj_w: *const c_void,       // [hidden, num_q_heads*head_dim]

    // --- MLP projections ---
    pub gate_proj_w: *const c_void,    // [intermediate_size, hidden]
    pub up_proj_w: *const c_void,      // [intermediate_size, hidden]
    pub down_proj_w: *const c_void,    // [hidden, intermediate_size]

    // --- KV cache state ---
    pub kv_cache_k: *mut c_void,       // [num_kv_heads, max_T, head_dim]
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,                 // current cache length (pre-append)
    pub kv_max_t: c_int,               // allocated T dimension
}

unsafe impl Send for Gemma4DecodeLayerDesc {}
unsafe impl Sync for Gemma4DecodeLayerDesc {}

impl Default for Gemma4DecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
