//! FFI bindings for prefill kernels.
//! These are component kernels (not megakernels) — the prefill engine
//! orchestrates them layer by layer.

use std::ffi::{c_int, c_void};
use gpu_hal::{GpuBuffer, GpuError, ScalarType};

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

    fn dotcache_qwen35_hip_batched_matmul_view(
        dtype: c_int,
        device_ordinal: usize,
        batch_rank: c_int,
        batch_elems: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        lhs_batch_strides: *const c_int,
        rhs_batch_strides: *const c_int,
        out_batch_dims: *const c_int,
        lhs_row_stride: c_int,
        lhs_k_stride: c_int,
        rhs_k_stride: c_int,
        rhs_col_stride: c_int,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
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
        return Err(GpuError::Hip(format!("embedding_lookup failed: {status}")));
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
        return Err(GpuError::Hip(format!("batched_matmul failed: {status}")));
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
        return Err(GpuError::Hip(format!("full_attention_prefill failed: {status}")));
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
        return Err(GpuError::Hip(format!("linear_prefill_conv_pack failed: {status}")));
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
        return Err(GpuError::Hip(format!("delta_recurrent_prefill failed: {status}")));
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
        return Err(GpuError::Hip(format!("l2norm failed: {status}")));
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
        return Err(GpuError::Hip(format!("swiglu_mul failed: {status}")));
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
        return Err(GpuError::Hip(format!("rms_norm_gated failed: {status}")));
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
        return Err(GpuError::Hip(format!("mul_scalar failed: {status}")));
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
    let row_bytes = hidden_dim * dtype.size_in_bytes();
    let out_row_bytes = out_dim * dtype.size_in_bytes();
    for row in 0..n_rows {
        let hidden_ptr = hidden.offset_ptr(row * row_bytes);
        let out_ptr = unsafe { (out.as_mut_ptr() as *mut u8).add(row * out_row_bytes) as *mut std::ffi::c_void };
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
            return Err(GpuError::Hip(format!("fused_rms_norm_linear row {row} failed: {status}")));
        }
    }
    Ok(())
}

// ---- Matmul with transposed rhs (y = x @ W^T) ----

/// Matrix multiply with transposed rhs: out [m, n] = lhs [m, k] × rhs^T where rhs is [n, k].
/// This is the standard linear projection: y = x @ W.T where W is [out_dim, in_dim].
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
    let batch_dims = [batch_elems as c_int];
    let batch_strides = [1 as c_int]; // no batching
    let status = unsafe {
        dotcache_qwen35_hip_batched_matmul_view(
            dtype.kernel_dtype_code(),
            ordinal,
            1, // batch_rank
            batch_elems,
            m as c_int,
            n as c_int,
            k as c_int,
            batch_strides.as_ptr(), // lhs_batch_strides
            batch_strides.as_ptr(), // rhs_batch_strides
            batch_dims.as_ptr(),    // out_batch_dims
            k as c_int,            // lhs_row_stride = k (standard row-major)
            1,                      // lhs_k_stride = 1 (contiguous)
            1,                      // rhs_k_stride = 1 (k dim contiguous in rhs)
            k as c_int,            // rhs_col_stride = k (virtually transpose: rhs[kk,col] reads rhs_data[col*k + kk])
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("matmul_rhs_transposed failed: {status}")));
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
        return Err(GpuError::Hip(format!("rms_norm_rows failed: {status}")));
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
        return Err(GpuError::Hip(format!("cast failed: {status}")));
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
        return Err(GpuError::Hip(format!("element_add failed: {status}")));
    }
    Ok(())
}

// ---- RoPE for prefill ----

/// Apply RoPE in-place on tensor [seq_len, num_heads, head_dim].
/// Only the first rotary_dim dimensions of each head are rotated.
pub fn apply_rope_prefill(
    ordinal: usize,
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    data: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let half_rot = rotary_dim / 2;
    let status = unsafe {
        dotcache_qwen35_hip_apply_rope_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_heads,
            head_dim,
            half_rot,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            data.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("apply_rope_prefill failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_transpose_shd_hsd(
            dtype.kernel_dtype_code(),
            ordinal,
            s, h, d,
            src.as_ptr(),
            dst.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("transpose_shd_hsd failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_transpose_pad_conv(
            dtype.kernel_dtype_code(),
            ordinal,
            s, c, pad,
            src.as_ptr(),
            dst.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("transpose_pad_conv failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_extract_conv_state(
            dtype.kernel_dtype_code(),
            ordinal,
            s, c, kern_minus_1,
            src.as_ptr(),
            dst.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("extract_conv_state failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_sigmoid_mul(
            dtype.kernel_dtype_code(), ordinal, total_elems,
            data.as_ptr(), gate.as_ptr(), out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("sigmoid_mul failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_compute_beta_g(
            dtype.kernel_dtype_code(), ordinal, seq_len, nv,
            b.as_ptr(), a.as_ptr(), dt_bias.as_ptr(), a_log_exp.as_ptr(),
            beta.as_mut_ptr(), g.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("compute_beta_g failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_split_qgate(
            dtype.kernel_dtype_code(), ordinal, s, num_heads, head_dim,
            src.as_ptr(), query_out.as_mut_ptr(), gate_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("split_qgate failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_split_qkv(
            dtype.kernel_dtype_code(), ordinal, s, key_dim, val_dim,
            src.as_ptr(), q.as_mut_ptr(), k.as_mut_ptr(), v.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("split_qkv failed: {status}")));
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
    let status = unsafe {
        dotcache_qwen35_hip_repeat_interleave_heads(
            dtype.kernel_dtype_code(), ordinal,
            s, n_heads, head_dim, repeats,
            src.as_ptr(), dst.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!("repeat_interleave_heads failed: {status}")));
    }
    Ok(())
}
