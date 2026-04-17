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

    fn dotcache_gemma4_hip_rms_norm_rows(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_matvec_batched(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;

    fn dotcache_gemma4_hip_rope_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_base: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        x: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_kv_append_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_base: usize,
        max_t: usize,
        k_in: *const c_void,
        v_in: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_attn_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_base: usize,
        max_t: usize,
        sliding_window: c_int,
        scale: f32,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        scores_scratch: *mut c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_add_residual(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_add_scaled_residual(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        scalar: f32,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_scalar_mul_inplace(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        scalar: f32,
        x: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_fused_attn_block(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        max_t: usize,
        sliding_window: c_int,
        shared_kv: c_int,
        eps: f32,
        scale: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        input_norm_w: *const c_void,
        q_proj_w: *const c_void,
        k_proj_w: *const c_void,
        v_proj_w: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_w: *const c_void,
        post_attn_norm_w: *const c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    fn dotcache_gemma4_hip_fused_mlp_ple(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        intermediate_size: usize,
        ple_hidden: usize,
        eps: f32,
        layer_scalar: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        pre_ff_norm_w: *const c_void,
        gate_proj_w: *const c_void,
        up_proj_w: *const c_void,
        down_proj_w: *const c_void,
        post_ff_norm_w: *const c_void,
        per_layer_input: *const c_void,
        per_layer_input_gate_w: *const c_void,
        per_layer_projection_w: *const c_void,
        post_per_layer_input_norm_w: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    fn dotcache_gemma4_hip_gather_layer_slice(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_layers: usize,
        ple_hidden: usize,
        layer_idx: usize,
        src: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    fn dotcache_gemma4_hip_embed_gather_scaled(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        scale: f32,
        token_ids: *const c_uint,
        table: *const c_void,
        out: *mut c_void,
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

/// Apply the Gemma RMSNorm to each of `num_rows` rows of a packed
/// `[num_rows, n_cols]` tensor independently, using the same `weight` vector
/// (or `None` for with_scale=False) for every row.
///
/// Implemented as a serial launch loop — one kernel launch per row, each
/// handling `n_cols` scalars. The underlying kernel is a single-block launch
/// that normalizes exactly one row, so this helper advances the input/output
/// pointers by `n_cols * sizeof(T)` bytes per iteration. Fine for the Gemma
/// 4 layer-0 validator (num_heads ≤ 8, n_cols=256), correctness before fusion.
pub fn rms_norm_per_row(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    num_rows: usize,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let row_bytes = n_cols * dtype.size_in_bytes();
    let in_base = input.as_ptr() as *const u8;
    let out_base = output.as_mut_ptr() as *mut u8;
    for row in 0..num_rows {
        let offset = row * row_bytes;
        let in_ptr = unsafe { in_base.add(offset) as *const c_void };
        let out_ptr = unsafe { out_base.add(offset) as *mut c_void };
        let status = unsafe {
            dotcache_gemma4_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_cols,
                eps,
                in_ptr,
                weight_ptr,
                out_ptr,
            )
        };
        if status != 0 {
            return Err(GpuError::Hip(format!(
                "gemma4 rms_norm_per_row failed at row {row} with status {status}"
            )));
        }
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

// =============================================================================
// Prefill / batched primitives (Step 13).
//
// Each wrapper below mirrors its single-token counterpart above but takes a
// `seq_len` (or `n_rows`) parameter so the kernel launch processes the whole
// batch in one shot. Only `gemma4_e2e_validate`'s Phase A consumes them today;
// the decode path in `gemma4_decode_validate` continues to use the
// single-token primitives unchanged.
// =============================================================================

/// Multi-row Gemma RMSNorm. Normalizes each row of a `[n_rows, n_cols]` tensor
/// independently using the same `weight[n_cols]` (or `None` for
/// `with_scale=False`). Dispatches to a single kernel launch (grid=n_rows).
pub fn rms_norm_rows(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        dotcache_gemma4_hip_rms_norm_rows(
            dtype.kernel_dtype_code(),
            ordinal,
            n_rows,
            n_cols,
            eps,
            input.as_ptr(),
            weight_ptr,
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 rms_norm_rows failed with status {status}"
        )));
    }
    Ok(())
}

/// Batched matvec `out[s, r] = dot(W[r, :], in[s, :])` computed for all (s, r).
/// Single kernel launch with work-stealing over `seq_len * out_dim` items.
pub fn matvec_batched(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_matvec_batched(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
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
            "gemma4 matvec_batched failed with status {status}"
        )));
    }
    Ok(())
}

/// Apply Gemma split-half RoPE to every token in a `[seq_len, num_heads,
/// head_dim]` tensor, with token `s` using position `pos_base + s` into the
/// shared cos/sin tables.
#[allow(clippy::too_many_arguments)]
pub fn rope_prefill(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos_base: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_rope_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_heads,
            head_dim,
            rotary_dim,
            pos_base,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 rope_prefill failed with status {status}"
        )));
    }
    Ok(())
}

/// Write a `[seq_len, num_kv_heads, head_dim]` K/V tensor into cache slots
/// `[pos_base, pos_base+seq_len)` of a `[num_kv_heads, max_t, head_dim]` cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_prefill(
    ordinal: usize,
    dtype: ScalarType,
    k_in: &GpuBuffer,
    v_in: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_base: usize,
    max_t: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_kv_append_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_kv_heads,
            head_dim,
            pos_base,
            max_t,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 kv_append_prefill failed with status {status}"
        )));
    }
    Ok(())
}

/// Prefill-style SWA/full attention over `seq_len` query tokens. The cache
/// must already contain `pos_base + seq_len` valid entries (fill via
/// `kv_append_prefill`). `scores_scratch` needs at least
/// `seq_len * num_q_heads * max_t * 4` bytes of F32 storage.
///
/// Output layout: `[seq_len, num_q_heads, head_dim]`.
/// Pass `sliding_window <= 0` for full attention.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    ordinal: usize,
    dtype: ScalarType,
    q: &GpuBuffer,
    k_cache: &GpuBuffer,
    v_cache: &GpuBuffer,
    scores_scratch: &mut GpuBuffer,
    out: &mut GpuBuffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_base: usize,
    max_t: usize,
    sliding_window: i32,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_attn_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_base,
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
            "gemma4 attn_prefill failed with status {status}"
        )));
    }
    Ok(())
}

/// Elementwise residual add `out[i] = a[i] + b[i]` over `n` scalars.
pub fn add_residual(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_add_residual(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 add_residual failed with status {status}"
        )));
    }
    Ok(())
}

/// Elementwise `out[i] = (a[i] + b[i]) * scalar` for the Gemma 4 PLE residual
/// "(h_pre_ple + normed) * layer_scalar" step.
pub fn add_scaled_residual(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    scalar: f32,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_add_scaled_residual(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            scalar,
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 add_scaled_residual failed with status {status}"
        )));
    }
    Ok(())
}

/// In-place scalar multiply `x[i] *= scalar`. Use after a matvec when the
/// output needs a constant BF16-rounded multiplier applied (mirrors HF's
/// Python-float times BF16-tensor rounding — the caller should pass the
/// scalar pre-rounded to BF16 on the host).
pub fn scalar_mul_inplace(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    scalar: f32,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_scalar_mul_inplace(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            scalar,
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 scalar_mul_inplace failed with status {status}"
        )));
    }
    Ok(())
}

/// Required F32 workspace (elements) for `fused_attn_block`. Layout matches
/// the kernel: hidden + normed + proj (q+2kv) + scores (nq*max_t) + attn_out
/// (q_dim) + oproj + oproj_normed.
pub fn fused_attn_block_workspace_elems(
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_t: usize,
) -> usize {
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    2 * hidden_size + q_dim + 2 * kv_dim + num_q_heads * max_t + q_dim + 2 * hidden_size
}

/// Run one Gemma 4 decoder layer's attention half (input_norm → QKV → qk/v
/// norm → RoPE → kv_append → SWA/full attention → o_proj → post_attn_norm
/// → residual) in a single kernel launch. `shared_kv` skips K/V generation
/// and assumes the cache at slot `position` already holds the inherited K/V.
///
/// `workspace` must be at least `fused_attn_block_workspace_elems(...) * 4`
/// bytes (F32). The three counter buffers are each ≥4 bytes of U32 storage;
/// the kernel clears the barrier pair before launch and reuses
/// `matvec_counter` internally between the QKV and o_proj phases.
#[allow(clippy::too_many_arguments)]
pub fn fused_attn_block(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    input_norm_w: &GpuBuffer,
    q_proj_w: &GpuBuffer,
    k_proj_w: Option<&GpuBuffer>,
    v_proj_w: Option<&GpuBuffer>,
    q_norm_w: &GpuBuffer,
    k_norm_w: Option<&GpuBuffer>,
    o_proj_w: &GpuBuffer,
    post_attn_norm_w: &GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    sliding_window: i32,
    position: usize,
    max_t: usize,
    shared_kv: bool,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    let null = std::ptr::null();
    let k_proj_ptr = k_proj_w.map(|b| b.as_ptr()).unwrap_or(null);
    let v_proj_ptr = v_proj_w.map(|b| b.as_ptr()).unwrap_or(null);
    let k_norm_ptr = k_norm_w.map(|b| b.as_ptr()).unwrap_or(null);
    let status = unsafe {
        dotcache_gemma4_hip_fused_attn_block(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            position,
            max_t,
            sliding_window as c_int,
            if shared_kv { 1 } else { 0 },
            eps,
            scale,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            input_norm_w.as_ptr(),
            q_proj_w.as_ptr(),
            k_proj_ptr,
            v_proj_ptr,
            q_norm_w.as_ptr(),
            k_norm_ptr,
            o_proj_w.as_ptr(),
            post_attn_norm_w.as_ptr(),
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 fused_attn_block failed with status {status}"
        )));
    }
    Ok(())
}

/// Required F32 workspace (elements) for `fused_mlp_ple`. Matches the
/// kernel's layout: 7 * hidden + 3 * intermediate + 2 * ple_hidden.
pub fn fused_mlp_ple_workspace_elems(
    hidden_size: usize,
    intermediate_size: usize,
    ple_hidden: usize,
) -> usize {
    7 * hidden_size + 3 * intermediate_size + 2 * ple_hidden
}

/// Run one Gemma 4 decoder layer's MLP + PLE half (pre_ff_norm → gate/up
/// proj → gelu*up → down_proj → post_ff_norm → residual → per_layer_input_gate
/// → gelu*per_layer_input → per_layer_projection → post_per_layer_input_norm
/// → (+)*layer_scalar) in a single kernel launch. `hidden_in` is `h_mid`
/// (output of `fused_attn_block`); `hidden_out` is the new `h_running`.
#[allow(clippy::too_many_arguments)]
pub fn fused_mlp_ple(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    pre_ff_norm_w: &GpuBuffer,
    gate_proj_w: &GpuBuffer,
    up_proj_w: &GpuBuffer,
    down_proj_w: &GpuBuffer,
    post_ff_norm_w: &GpuBuffer,
    per_layer_input: &GpuBuffer,
    per_layer_input_gate_w: &GpuBuffer,
    per_layer_projection_w: &GpuBuffer,
    post_per_layer_input_norm_w: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    intermediate_size: usize,
    ple_hidden: usize,
    eps: f32,
    layer_scalar: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_fused_mlp_ple(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            intermediate_size,
            ple_hidden,
            eps,
            layer_scalar,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            pre_ff_norm_w.as_ptr(),
            gate_proj_w.as_ptr(),
            up_proj_w.as_ptr(),
            down_proj_w.as_ptr(),
            post_ff_norm_w.as_ptr(),
            per_layer_input.as_ptr(),
            per_layer_input_gate_w.as_ptr(),
            per_layer_projection_w.as_ptr(),
            post_per_layer_input_norm_w.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 fused_mlp_ple failed with status {status}"
        )));
    }
    Ok(())
}

/// Extract one layer's `[seq_len, ple_hidden]` slice from a batched
/// `[seq_len, num_layers, ple_hidden]` PLI table. The output buffer is
/// contiguous so the existing elementwise primitives can consume it.
#[allow(clippy::too_many_arguments)]
pub fn gather_layer_slice(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    src: &GpuBuffer,
    seq_len: usize,
    num_layers: usize,
    ple_hidden: usize,
    layer_idx: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_gather_layer_slice(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_layers,
            ple_hidden,
            layer_idx,
            src.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 gather_layer_slice failed with status {status}"
        )));
    }
    Ok(())
}

/// Gather rows from an embedding table and scale by a host-side multiplier.
/// `token_ids` is a device-resident `[seq_len]` u32 buffer; `table` is a
/// `[vocab_size, hidden_size]` row-major embedding. The multiplier is applied
/// in FP32 after the BF16 load (mirroring HF's `embed * sqrt(hidden_size)`).
#[allow(clippy::too_many_arguments)]
pub fn embed_gather_scaled(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    token_ids: &GpuBuffer,
    table: &GpuBuffer,
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        dotcache_gemma4_hip_embed_gather_scaled(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            hidden_size,
            vocab_size,
            scale,
            token_ids.as_ptr() as *const c_uint,
            table.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "gemma4 embed_gather_scaled failed with status {status}"
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
