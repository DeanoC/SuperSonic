use std::ffi::c_void;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use half::bf16;

fn unsupported(op: &str, detail: impl Into<String>) -> GpuError {
    GpuError::InvalidArg(format!("metal host {op}: {}", detail.into()))
}

#[inline(always)]
unsafe fn bf16_slice<'a>(ptr: *const c_void, len: usize) -> &'a [u16] {
    std::slice::from_raw_parts(ptr as *const u16, len)
}

#[inline(always)]
unsafe fn bf16_slice_mut<'a>(ptr: *mut c_void, len: usize) -> &'a mut [u16] {
    std::slice::from_raw_parts_mut(ptr as *mut u16, len)
}

#[inline(always)]
unsafe fn f32_slice<'a>(ptr: *const c_void, len: usize) -> &'a [f32] {
    std::slice::from_raw_parts(ptr as *const f32, len)
}

#[inline(always)]
unsafe fn f32_slice_mut<'a>(ptr: *mut c_void, len: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(ptr as *mut f32, len)
}

#[inline(always)]
unsafe fn u32_slice<'a>(ptr: *const c_void, len: usize) -> &'a [u32] {
    std::slice::from_raw_parts(ptr as *const u32, len)
}

#[inline(always)]
unsafe fn u32_slice_mut<'a>(ptr: *mut c_void, len: usize) -> &'a mut [u32] {
    std::slice::from_raw_parts_mut(ptr as *mut u32, len)
}

#[inline(always)]
fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

#[inline(always)]
fn f32_to_bf16_bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
fn use_fast_silu() -> bool {
    std::env::var_os("SUPERSONIC_METAL_USE_FAST_SILU").is_some()
}

#[inline(always)]
fn sigmoid_fast(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline(always)]
fn silu_fast(x: f32) -> f32 {
    x * sigmoid_fast(x)
}

#[inline(always)]
fn qwen_silu(x: f32) -> f32 {
    if use_fast_silu() {
        silu_fast(x)
    } else {
        x / (1.0 + (-x).exp())
    }
}

#[inline(always)]
fn read_weight_f32(weight: &GpuBuffer, idx: usize) -> Result<f32, GpuError> {
    match weight.dtype() {
        ScalarType::BF16 => {
            let values = unsafe { bf16_slice(weight.as_ptr(), weight.elem_count()) };
            Ok(bf16_to_f32(values[idx]))
        }
        ScalarType::F32 => {
            let values = unsafe { f32_slice(weight.as_ptr(), weight.elem_count()) };
            Ok(values[idx])
        }
        other => Err(unsupported(
            "weight read",
            format!("unsupported dtype {other:?}"),
        )),
    }
}

pub(crate) fn embedding_lookup(
    token_count: usize,
    vocab_size: usize,
    hidden_size: usize,
    embeddings: &GpuBuffer,
    indexes: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if embeddings.dtype() != ScalarType::BF16 || out.dtype() != ScalarType::BF16 {
        return Err(unsupported(
            "embedding_lookup",
            format!(
                "expected BF16 embeddings/out, got {:?}/{:?}",
                embeddings.dtype(),
                out.dtype()
            ),
        ));
    }
    if indexes.dtype() != ScalarType::U32 {
        return Err(unsupported(
            "embedding_lookup",
            format!("expected U32 indexes, got {:?}", indexes.dtype()),
        ));
    }
    let embedding_elems = vocab_size * hidden_size;
    let out_elems = token_count * hidden_size;
    let embeddings = unsafe { bf16_slice(embeddings.as_ptr(), embedding_elems) };
    let indexes = unsafe { u32_slice(indexes.as_ptr(), token_count) };
    let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), out_elems) };
    for token_idx in 0..token_count {
        let vocab_idx = indexes[token_idx] as usize;
        if vocab_idx >= vocab_size {
            return Err(unsupported(
                "embedding_lookup",
                format!("token index {vocab_idx} out of range for vocab {vocab_size}"),
            ));
        }
        let src = &embeddings[vocab_idx * hidden_size..(vocab_idx + 1) * hidden_size];
        let dst = &mut out[token_idx * hidden_size..(token_idx + 1) * hidden_size];
        dst.copy_from_slice(src);
    }
    Ok(())
}

pub(crate) fn batched_matmul(
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let lhs = unsafe { bf16_slice(lhs.as_ptr(), batch_elems * m * k) };
            let rhs = unsafe { bf16_slice(rhs.as_ptr(), batch_elems * k * n) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), batch_elems * m * n) };
            for b in 0..batch_elems {
                for row in 0..m {
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let lhs_idx = b * m * k + row * k + kk;
                            let rhs_idx = b * k * n + kk * n + col;
                            acc += bf16_to_f32(lhs[lhs_idx]) * bf16_to_f32(rhs[rhs_idx]);
                        }
                        out[b * m * n + row * n + col] = f32_to_bf16_bits(acc);
                    }
                }
            }
        }
        ScalarType::F32 => {
            let lhs = unsafe { f32_slice(lhs.as_ptr(), batch_elems * m * k) };
            let rhs = unsafe { f32_slice(rhs.as_ptr(), batch_elems * k * n) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), batch_elems * m * n) };
            for b in 0..batch_elems {
                for row in 0..m {
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let lhs_idx = b * m * k + row * k + kk;
                            let rhs_idx = b * k * n + kk * n + col;
                            acc += lhs[lhs_idx] * rhs[rhs_idx];
                        }
                        out[b * m * n + row * n + col] = acc;
                    }
                }
            }
        }
        other => {
            return Err(unsupported(
                "batched_matmul",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn matmul_rhs_transposed(
    dtype: ScalarType,
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let lhs = unsafe { bf16_slice(lhs.as_ptr(), batch_elems * m * k) };
            let rhs = unsafe { bf16_slice(rhs.as_ptr(), n * k) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), batch_elems * m * n) };
            for batch in 0..batch_elems {
                let lhs_batch_base = batch * m * k;
                let out_batch_base = batch * m * n;
                for row in 0..m {
                    let lhs_row_base = lhs_batch_base + row * k;
                    let out_row_base = out_batch_base + row * n;
                    for col in 0..n {
                        let rhs_row_base = col * k;
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += bf16_to_f32(lhs[lhs_row_base + kk])
                                * bf16_to_f32(rhs[rhs_row_base + kk]);
                        }
                        out[out_row_base + col] = f32_to_bf16_bits(acc);
                    }
                }
            }
        }
        ScalarType::F32 => {
            let lhs = unsafe { f32_slice(lhs.as_ptr(), batch_elems * m * k) };
            let rhs = unsafe { f32_slice(rhs.as_ptr(), n * k) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), batch_elems * m * n) };
            for batch in 0..batch_elems {
                let lhs_batch_base = batch * m * k;
                let out_batch_base = batch * m * n;
                for row in 0..m {
                    let lhs_row_base = lhs_batch_base + row * k;
                    let out_row_base = out_batch_base + row * n;
                    for col in 0..n {
                        let rhs_row_base = col * k;
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += lhs[lhs_row_base + kk] * rhs[rhs_row_base + kk];
                        }
                        out[out_row_base + col] = acc;
                    }
                }
            }
        }
        other => {
            return Err(unsupported(
                "matmul_rhs_transposed",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn standalone_matvec(
    dtype: ScalarType,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    output: &mut GpuBuffer,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), GpuError> {
    matmul_rhs_transposed(dtype, 1, 1, out_dim, in_dim, input, weight, output)
}

pub(crate) fn fused_rms_norm_linear_rows(
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
    match dtype {
        ScalarType::BF16 => {
            if hidden.dtype() != ScalarType::BF16 || out.dtype() != ScalarType::BF16 {
                return Err(unsupported(
                    "fused_rms_norm_linear_rows",
                    format!(
                        "expected BF16 hidden/out buffers, got {:?}/{:?}",
                        hidden.dtype(),
                        out.dtype()
                    ),
                ));
            }
            let hidden = unsafe { bf16_slice(hidden.as_ptr(), n_rows * hidden_dim) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), n_rows * out_dim) };
            let mut normed = vec![0.0f32; hidden_dim];
            for row in 0..n_rows {
                let hidden_base = row * hidden_dim;
                let out_base = row * out_dim;
                let mut mean_sq = 0.0f32;
                for col in 0..hidden_dim {
                    let value = bf16_to_f32(hidden[hidden_base + col]);
                    mean_sq += value * value;
                    normed[col] = value;
                }
                let inv_rms = 1.0f32 / ((mean_sq / hidden_dim as f32) + eps).sqrt();
                for col in 0..hidden_dim {
                    let scale = read_weight_f32(norm_weight, col)? + 1.0f32;
                    normed[col] *= inv_rms * scale;
                }
                for out_col in 0..out_dim {
                    let weight_base = out_col * hidden_dim;
                    let mut acc = 0.0f32;
                    for col in 0..hidden_dim {
                        acc += normed[col] * read_weight_f32(proj_weight, weight_base + col)?;
                    }
                    out[out_base + out_col] = f32_to_bf16_bits(acc);
                }
            }
        }
        ScalarType::F32 => {
            if hidden.dtype() != ScalarType::F32 || out.dtype() != ScalarType::F32 {
                return Err(unsupported(
                    "fused_rms_norm_linear_rows",
                    format!(
                        "expected F32 hidden/out buffers, got {:?}/{:?}",
                        hidden.dtype(),
                        out.dtype()
                    ),
                ));
            }
            let hidden = unsafe { f32_slice(hidden.as_ptr(), n_rows * hidden_dim) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), n_rows * out_dim) };
            let mut normed = vec![0.0f32; hidden_dim];
            for row in 0..n_rows {
                let hidden_base = row * hidden_dim;
                let out_base = row * out_dim;
                let mut mean_sq = 0.0f32;
                for col in 0..hidden_dim {
                    let value = hidden[hidden_base + col];
                    mean_sq += value * value;
                    normed[col] = value;
                }
                let inv_rms = 1.0f32 / ((mean_sq / hidden_dim as f32) + eps).sqrt();
                for col in 0..hidden_dim {
                    let scale = read_weight_f32(norm_weight, col)? + 1.0f32;
                    normed[col] *= inv_rms * scale;
                }
                for out_col in 0..out_dim {
                    let weight_base = out_col * hidden_dim;
                    let mut acc = 0.0f32;
                    for col in 0..hidden_dim {
                        acc += normed[col] * read_weight_f32(proj_weight, weight_base + col)?;
                    }
                    out[out_base + out_col] = acc;
                }
            }
        }
        other => {
            return Err(unsupported(
                "fused_rms_norm_linear_rows",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn standalone_matvec_host_f32(
    dtype: ScalarType,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let input = unsafe { bf16_slice(input.as_ptr(), in_dim) };
            let weight = unsafe { bf16_slice(weight.as_ptr(), out_dim * in_dim) };
            let mut out = vec![0.0f32; out_dim];
            for row in 0..out_dim {
                let weight_row_base = row * in_dim;
                let mut acc = 0.0f32;
                for col in 0..in_dim {
                    acc += bf16_to_f32(input[col]) * bf16_to_f32(weight[weight_row_base + col]);
                }
                out[row] = acc;
            }
            Ok(out)
        }
        ScalarType::F32 => {
            let input = unsafe { f32_slice(input.as_ptr(), in_dim) };
            let weight = unsafe { f32_slice(weight.as_ptr(), out_dim * in_dim) };
            let mut out = vec![0.0f32; out_dim];
            for row in 0..out_dim {
                let weight_row_base = row * in_dim;
                let mut acc = 0.0f32;
                for col in 0..in_dim {
                    acc += input[col] * weight[weight_row_base + col];
                }
                out[row] = acc;
            }
            Ok(out)
        }
        other => Err(unsupported(
            "standalone_matvec_host_f32",
            format!("unsupported dtype {other:?}"),
        )),
    }
}

pub(crate) fn qwen_rms_norm_standalone_matvec_host_f32(
    dtype: ScalarType,
    input: &GpuBuffer,
    norm_weight: &GpuBuffer,
    eps: f32,
    weight: &GpuBuffer,
    hidden_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let input = unsafe { bf16_slice(input.as_ptr(), hidden_dim) };
            let weight = unsafe { bf16_slice(weight.as_ptr(), out_dim * hidden_dim) };
            let mut mean_sq = 0.0f32;
            for &value_bits in input.iter() {
                let value = bf16_to_f32(value_bits);
                mean_sq += value * value;
            }
            let inv_rms = 1.0 / ((mean_sq / hidden_dim as f32) + eps).sqrt();

            let mut normed = vec![0.0f32; hidden_dim];
            for col in 0..hidden_dim {
                let scale = read_weight_f32(norm_weight, col)? + 1.0;
                normed[col] = bf16_to_f32(input[col]) * inv_rms * scale;
            }

            let mut out = vec![0.0f32; out_dim];
            for row in 0..out_dim {
                let weight_row_base = row * hidden_dim;
                let mut acc = 0.0f32;
                for col in 0..hidden_dim {
                    acc += normed[col] * bf16_to_f32(weight[weight_row_base + col]);
                }
                out[row] = acc;
            }
            Ok(out)
        }
        ScalarType::F32 => {
            let input = unsafe { f32_slice(input.as_ptr(), hidden_dim) };
            let weight = unsafe { f32_slice(weight.as_ptr(), out_dim * hidden_dim) };
            let mut mean_sq = 0.0f32;
            for &value in input.iter() {
                mean_sq += value * value;
            }
            let inv_rms = 1.0 / ((mean_sq / hidden_dim as f32) + eps).sqrt();

            let mut normed = vec![0.0f32; hidden_dim];
            for col in 0..hidden_dim {
                let scale = read_weight_f32(norm_weight, col)? + 1.0;
                normed[col] = input[col] * inv_rms * scale;
            }

            let mut out = vec![0.0f32; out_dim];
            for row in 0..out_dim {
                let weight_row_base = row * hidden_dim;
                let mut acc = 0.0f32;
                for col in 0..hidden_dim {
                    acc += normed[col] * weight[weight_row_base + col];
                }
                out[row] = acc;
            }
            Ok(out)
        }
        other => Err(unsupported(
            "qwen_rms_norm_standalone_matvec_host_f32",
            format!("unsupported dtype {other:?}"),
        )),
    }
}

pub(crate) fn rms_norm_rows(
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    add_unit_offset: bool,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let input = unsafe { bf16_slice(input.as_ptr(), n_rows * n_cols) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let row_base = row * n_cols;
                let mut mean_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = bf16_to_f32(input[row_base + col]);
                    mean_sq += value * value;
                }
                let inv_rms = 1.0 / ((mean_sq / n_cols as f32) + eps).sqrt();
                for col in 0..n_cols {
                    let value = bf16_to_f32(input[row_base + col]);
                    let scale =
                        read_weight_f32(weight, col)? + if add_unit_offset { 1.0 } else { 0.0 };
                    out[row_base + col] = f32_to_bf16_bits(value * inv_rms * scale);
                }
            }
        }
        ScalarType::F32 => {
            let input = unsafe { f32_slice(input.as_ptr(), n_rows * n_cols) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let row_base = row * n_cols;
                let mut mean_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = input[row_base + col];
                    mean_sq += value * value;
                }
                let inv_rms = 1.0 / ((mean_sq / n_cols as f32) + eps).sqrt();
                for col in 0..n_cols {
                    let scale =
                        read_weight_f32(weight, col)? + if add_unit_offset { 1.0 } else { 0.0 };
                    out[row_base + col] = input[row_base + col] * inv_rms * scale;
                }
            }
        }
        other => {
            return Err(unsupported(
                "rms_norm_rows",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn cast(
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match (input_dtype, output_dtype) {
        (ScalarType::BF16, ScalarType::BF16) => {
            let src = unsafe { bf16_slice(input.as_ptr(), total_elems) };
            let dst = unsafe { bf16_slice_mut(out.as_mut_ptr(), total_elems) };
            dst.copy_from_slice(src);
        }
        (ScalarType::F32, ScalarType::F32) => {
            let src = unsafe { f32_slice(input.as_ptr(), total_elems) };
            let dst = unsafe { f32_slice_mut(out.as_mut_ptr(), total_elems) };
            dst.copy_from_slice(src);
        }
        (ScalarType::U32, ScalarType::U32) => {
            let src = unsafe { u32_slice(input.as_ptr(), total_elems) };
            let dst = unsafe { u32_slice_mut(out.as_mut_ptr(), total_elems) };
            dst.copy_from_slice(src);
        }
        (ScalarType::BF16, ScalarType::F32) => {
            let src = unsafe { bf16_slice(input.as_ptr(), total_elems) };
            let dst = unsafe { f32_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                dst[i] = bf16_to_f32(src[i]);
            }
        }
        (ScalarType::F32, ScalarType::BF16) => {
            let src = unsafe { f32_slice(input.as_ptr(), total_elems) };
            let dst = unsafe { bf16_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                dst[i] = f32_to_bf16_bits(src[i]);
            }
        }
        other => return Err(unsupported("cast", format!("unsupported cast {other:?}"))),
    }
    Ok(())
}

pub(crate) fn element_add(
    dtype: ScalarType,
    total_elems: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let lhs = unsafe { bf16_slice(lhs.as_ptr(), total_elems) };
            let rhs = unsafe { bf16_slice(rhs.as_ptr(), total_elems) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = f32_to_bf16_bits(bf16_to_f32(lhs[i]) + bf16_to_f32(rhs[i]));
            }
        }
        ScalarType::F32 => {
            let lhs = unsafe { f32_slice(lhs.as_ptr(), total_elems) };
            let rhs = unsafe { f32_slice(rhs.as_ptr(), total_elems) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = lhs[i] + rhs[i];
            }
        }
        other => {
            return Err(unsupported(
                "element_add",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn apply_rope_prefill(
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
    let cos = unsafe { bf16_slice(cos_table.as_ptr(), cos_table.elem_count()) };
    let sin = unsafe { bf16_slice(sin_table.as_ptr(), sin_table.elem_count()) };
    match dtype {
        ScalarType::BF16 => {
            let data = unsafe { bf16_slice_mut(data.as_mut_ptr(), seq_len * num_heads * head_dim) };
            for pos in 0..seq_len {
                let table_base = (pos_offset + pos) * half_rot;
                for head in 0..num_heads {
                    let base = (pos * num_heads + head) * head_dim;
                    for i in 0..half_rot {
                        let c = bf16_to_f32(cos[table_base + i]);
                        let s = bf16_to_f32(sin[table_base + i]);
                        let x0 = bf16_to_f32(data[base + i]);
                        let x1 = bf16_to_f32(data[base + i + half_rot]);
                        data[base + i] = f32_to_bf16_bits(x0 * c - x1 * s);
                        data[base + i + half_rot] = f32_to_bf16_bits(x1 * c + x0 * s);
                    }
                }
            }
        }
        ScalarType::F32 => {
            let data = unsafe { f32_slice_mut(data.as_mut_ptr(), seq_len * num_heads * head_dim) };
            for pos in 0..seq_len {
                let table_base = (pos_offset + pos) * half_rot;
                for head in 0..num_heads {
                    let base = (pos * num_heads + head) * head_dim;
                    for i in 0..half_rot {
                        let c = bf16_to_f32(cos[table_base + i]);
                        let s = bf16_to_f32(sin[table_base + i]);
                        let x0 = data[base + i];
                        let x1 = data[base + i + half_rot];
                        data[base + i] = x0 * c - x1 * s;
                        data[base + i + half_rot] = x1 * c + x0 * s;
                    }
                }
            }
        }
        other => {
            return Err(unsupported(
                "apply_rope_prefill",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn transpose_shd_hsd(
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = s * h * d;
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), total) };
            let dst = unsafe { bf16_slice_mut(dst.as_mut_ptr(), total) };
            for si in 0..s {
                for hi in 0..h {
                    let src_base = (si * h + hi) * d;
                    let dst_base = (hi * s + si) * d;
                    dst[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
                }
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), total) };
            let dst = unsafe { f32_slice_mut(dst.as_mut_ptr(), total) };
            for si in 0..s {
                for hi in 0..h {
                    let src_base = (si * h + hi) * d;
                    let dst_base = (hi * s + si) * d;
                    dst[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
                }
            }
        }
        other => {
            return Err(unsupported(
                "transpose_shd_hsd",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn transpose_pad_conv(
    dtype: ScalarType,
    s: usize,
    c: usize,
    pad: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total_dst = c * (pad + s);
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), s * c) };
            let dst = unsafe { bf16_slice_mut(dst.as_mut_ptr(), total_dst) };
            dst.fill(0);
            for row in 0..s {
                for ch in 0..c {
                    dst[ch * (pad + s) + pad + row] = src[row * c + ch];
                }
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), s * c) };
            let dst = unsafe { f32_slice_mut(dst.as_mut_ptr(), total_dst) };
            dst.fill(0.0);
            for row in 0..s {
                for ch in 0..c {
                    dst[ch * (pad + s) + pad + row] = src[row * c + ch];
                }
            }
        }
        other => {
            return Err(unsupported(
                "transpose_pad_conv",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn extract_conv_state(
    dtype: ScalarType,
    s: usize,
    c: usize,
    kern_minus_1: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), s * c) };
            let dst = unsafe { bf16_slice_mut(dst.as_mut_ptr(), c * kern_minus_1) };
            dst.fill(0);
            let copy = s.min(kern_minus_1);
            let start = s.saturating_sub(copy);
            let dst_start = kern_minus_1 - copy;
            for ch in 0..c {
                for i in 0..copy {
                    dst[ch * kern_minus_1 + dst_start + i] = src[(start + i) * c + ch];
                }
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), s * c) };
            let dst = unsafe { f32_slice_mut(dst.as_mut_ptr(), c * kern_minus_1) };
            dst.fill(0.0);
            let copy = s.min(kern_minus_1);
            let start = s.saturating_sub(copy);
            let dst_start = kern_minus_1 - copy;
            for ch in 0..c {
                for i in 0..copy {
                    dst[ch * kern_minus_1 + dst_start + i] = src[(start + i) * c + ch];
                }
            }
        }
        other => {
            return Err(unsupported(
                "extract_conv_state",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn sigmoid_mul(
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let data = unsafe { bf16_slice(data.as_ptr(), total_elems) };
            let gate = unsafe { bf16_slice(gate.as_ptr(), total_elems) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = f32_to_bf16_bits(bf16_to_f32(data[i]) * sigmoid(bf16_to_f32(gate[i])));
            }
        }
        ScalarType::F32 => {
            let data = unsafe { f32_slice(data.as_ptr(), total_elems) };
            let gate = unsafe { f32_slice(gate.as_ptr(), total_elems) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = data[i] * sigmoid(gate[i]);
            }
        }
        other => {
            return Err(unsupported(
                "sigmoid_mul",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn compute_beta_g(
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
    match dtype {
        ScalarType::F32 => {
            let b = unsafe { f32_slice(b.as_ptr(), seq_len * nv) };
            let a = unsafe { f32_slice(a.as_ptr(), seq_len * nv) };
            let dt_bias = unsafe { f32_slice(dt_bias.as_ptr(), nv) };
            let a_log_exp = unsafe { f32_slice(a_log_exp.as_ptr(), nv) };
            let beta = unsafe { f32_slice_mut(beta.as_mut_ptr(), nv * seq_len) };
            let g = unsafe { f32_slice_mut(g.as_mut_ptr(), nv * seq_len) };
            for t in 0..seq_len {
                for h in 0..nv {
                    let src_idx = t * nv + h;
                    let dst_idx = h * seq_len + t;
                    beta[dst_idx] = sigmoid(b[src_idx]);
                    g[dst_idx] = -softplus(a[src_idx] + dt_bias[h]) * a_log_exp[h];
                }
            }
        }
        ScalarType::BF16 => {
            let b = unsafe { bf16_slice(b.as_ptr(), seq_len * nv) };
            let a = unsafe { bf16_slice(a.as_ptr(), seq_len * nv) };
            let dt_bias = unsafe { bf16_slice(dt_bias.as_ptr(), nv) };
            let a_log_exp = unsafe { bf16_slice(a_log_exp.as_ptr(), nv) };
            let beta = unsafe { bf16_slice_mut(beta.as_mut_ptr(), nv * seq_len) };
            let g = unsafe { bf16_slice_mut(g.as_mut_ptr(), nv * seq_len) };
            for t in 0..seq_len {
                for h in 0..nv {
                    let src_idx = t * nv + h;
                    let dst_idx = h * seq_len + t;
                    beta[dst_idx] = f32_to_bf16_bits(sigmoid(bf16_to_f32(b[src_idx])));
                    let g_value = -softplus(bf16_to_f32(a[src_idx]) + bf16_to_f32(dt_bias[h]))
                        * bf16_to_f32(a_log_exp[h]);
                    g[dst_idx] = f32_to_bf16_bits(g_value);
                }
            }
        }
        other => {
            return Err(unsupported(
                "compute_beta_g",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn split_qgate(
    dtype: ScalarType,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let src_stride = num_heads * head_dim * 2;
    let dst_stride = num_heads * head_dim;
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), s * src_stride) };
            let query_out = unsafe { bf16_slice_mut(query_out.as_mut_ptr(), s * dst_stride) };
            let gate_out = unsafe { bf16_slice_mut(gate_out.as_mut_ptr(), s * dst_stride) };
            for row in 0..s {
                for head in 0..num_heads {
                    let src_base = row * src_stride + head * head_dim * 2;
                    let dst_base = row * dst_stride + head * head_dim;
                    query_out[dst_base..dst_base + head_dim]
                        .copy_from_slice(&src[src_base..src_base + head_dim]);
                    gate_out[dst_base..dst_base + head_dim]
                        .copy_from_slice(&src[src_base + head_dim..src_base + 2 * head_dim]);
                }
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), s * src_stride) };
            let query_out = unsafe { f32_slice_mut(query_out.as_mut_ptr(), s * dst_stride) };
            let gate_out = unsafe { f32_slice_mut(gate_out.as_mut_ptr(), s * dst_stride) };
            for row in 0..s {
                for head in 0..num_heads {
                    let src_base = row * src_stride + head * head_dim * 2;
                    let dst_base = row * dst_stride + head * head_dim;
                    query_out[dst_base..dst_base + head_dim]
                        .copy_from_slice(&src[src_base..src_base + head_dim]);
                    gate_out[dst_base..dst_base + head_dim]
                        .copy_from_slice(&src[src_base + head_dim..src_base + 2 * head_dim]);
                }
            }
        }
        other => {
            return Err(unsupported(
                "split_qgate",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn split_qkv(
    dtype: ScalarType,
    s: usize,
    key_dim: usize,
    val_dim: usize,
    src: &GpuBuffer,
    q: &mut GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let src_stride = key_dim * 2 + val_dim;
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), s * src_stride) };
            let q = unsafe { bf16_slice_mut(q.as_mut_ptr(), s * key_dim) };
            let k = unsafe { bf16_slice_mut(k.as_mut_ptr(), s * key_dim) };
            let v = unsafe { bf16_slice_mut(v.as_mut_ptr(), s * val_dim) };
            for row in 0..s {
                let src_base = row * src_stride;
                q[row * key_dim..(row + 1) * key_dim]
                    .copy_from_slice(&src[src_base..src_base + key_dim]);
                k[row * key_dim..(row + 1) * key_dim]
                    .copy_from_slice(&src[src_base + key_dim..src_base + key_dim * 2]);
                v[row * val_dim..(row + 1) * val_dim].copy_from_slice(
                    &src[src_base + key_dim * 2..src_base + key_dim * 2 + val_dim],
                );
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), s * src_stride) };
            let q = unsafe { f32_slice_mut(q.as_mut_ptr(), s * key_dim) };
            let k = unsafe { f32_slice_mut(k.as_mut_ptr(), s * key_dim) };
            let v = unsafe { f32_slice_mut(v.as_mut_ptr(), s * val_dim) };
            for row in 0..s {
                let src_base = row * src_stride;
                q[row * key_dim..(row + 1) * key_dim]
                    .copy_from_slice(&src[src_base..src_base + key_dim]);
                k[row * key_dim..(row + 1) * key_dim]
                    .copy_from_slice(&src[src_base + key_dim..src_base + key_dim * 2]);
                v[row * val_dim..(row + 1) * val_dim].copy_from_slice(
                    &src[src_base + key_dim * 2..src_base + key_dim * 2 + val_dim],
                );
            }
        }
        other => {
            return Err(unsupported(
                "split_qkv",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn repeat_interleave_heads(
    dtype: ScalarType,
    s: usize,
    n_heads: usize,
    head_dim: usize,
    repeats: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let src = unsafe { bf16_slice(src.as_ptr(), s * n_heads * head_dim) };
            let dst = unsafe { bf16_slice_mut(dst.as_mut_ptr(), s * n_heads * repeats * head_dim) };
            for row in 0..s {
                for head in 0..n_heads {
                    let src_base = (row * n_heads + head) * head_dim;
                    for rep in 0..repeats {
                        let dst_base =
                            (row * (n_heads * repeats) + head * repeats + rep) * head_dim;
                        dst[dst_base..dst_base + head_dim]
                            .copy_from_slice(&src[src_base..src_base + head_dim]);
                    }
                }
            }
        }
        ScalarType::F32 => {
            let src = unsafe { f32_slice(src.as_ptr(), s * n_heads * head_dim) };
            let dst = unsafe { f32_slice_mut(dst.as_mut_ptr(), s * n_heads * repeats * head_dim) };
            for row in 0..s {
                for head in 0..n_heads {
                    let src_base = (row * n_heads + head) * head_dim;
                    for rep in 0..repeats {
                        let dst_base =
                            (row * (n_heads * repeats) + head * repeats + rep) * head_dim;
                        dst[dst_base..dst_base + head_dim]
                            .copy_from_slice(&src[src_base..src_base + head_dim]);
                    }
                }
            }
        }
        other => {
            return Err(unsupported(
                "repeat_interleave_heads",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn linear_prefill_conv_pack(
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
    if batch_size != 1 {
        return Err(unsupported(
            "linear_prefill_conv_pack",
            format!("batch_size={batch_size} is not supported"),
        ));
    }
    match dtype {
        ScalarType::BF16 => {
            if mixed_qkv.dtype() != ScalarType::BF16
                || weights.dtype() != ScalarType::BF16
                || out.dtype() != ScalarType::BF16
            {
                return Err(unsupported(
                    "linear_prefill_conv_pack",
                    format!(
                        "expected BF16 buffers, got mixed={:?} weights={:?} out={:?}",
                        mixed_qkv.dtype(),
                        weights.dtype(),
                        out.dtype()
                    ),
                ));
            }
            let mixed = unsafe { bf16_slice(mixed_qkv.as_ptr(), conv_dim * total_len) };
            let weights = unsafe { bf16_slice(weights.as_ptr(), conv_dim * kernel_size) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), seq_len * conv_dim) };
            for t in 0..seq_len {
                for ch in 0..conv_dim {
                    let mixed_base = ch * total_len + t;
                    let weight_base = ch * kernel_size;
                    let mut acc = 0.0f32;
                    for k_idx in 0..kernel_size {
                        acc += bf16_to_f32(mixed[mixed_base + k_idx])
                            * bf16_to_f32(weights[weight_base + k_idx]);
                    }
                    out[t * conv_dim + ch] = f32_to_bf16_bits(qwen_silu(acc));
                }
            }
            Ok(())
        }
        ScalarType::F32 => {
            if mixed_qkv.dtype() != ScalarType::F32
                || weights.dtype() != ScalarType::F32
                || out.dtype() != ScalarType::F32
            {
                return Err(unsupported(
                    "linear_prefill_conv_pack",
                    format!(
                        "expected F32 buffers, got mixed={:?} weights={:?} out={:?}",
                        mixed_qkv.dtype(),
                        weights.dtype(),
                        out.dtype()
                    ),
                ));
            }
            let mixed = unsafe { f32_slice(mixed_qkv.as_ptr(), conv_dim * total_len) };
            let weights = unsafe { f32_slice(weights.as_ptr(), conv_dim * kernel_size) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), seq_len * conv_dim) };
            for t in 0..seq_len {
                for ch in 0..conv_dim {
                    let mixed_base = ch * total_len + t;
                    let weight_base = ch * kernel_size;
                    let mut acc = 0.0f32;
                    for k_idx in 0..kernel_size {
                        acc += mixed[mixed_base + k_idx] * weights[weight_base + k_idx];
                    }
                    out[t * conv_dim + ch] = qwen_silu(acc);
                }
            }
            Ok(())
        }
        other => Err(unsupported(
            "linear_prefill_conv_pack",
            format!("unsupported dtype {other:?}"),
        )),
    }
}

pub(crate) fn l2norm(
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::F32 => {
            let input = unsafe { f32_slice(input.as_ptr(), n_rows * n_cols) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let base = row * n_cols;
                let mut norm_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = input[base + col];
                    norm_sq += value * value;
                }
                let inv_norm = 1.0 / (norm_sq + eps).sqrt();
                for col in 0..n_cols {
                    out[base + col] = input[base + col] * inv_norm;
                }
            }
        }
        ScalarType::BF16 => {
            let input = unsafe { bf16_slice(input.as_ptr(), n_rows * n_cols) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let base = row * n_cols;
                let mut norm_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = bf16_to_f32(input[base + col]);
                    norm_sq += value * value;
                }
                let inv_norm = 1.0 / (norm_sq + eps).sqrt();
                for col in 0..n_cols {
                    out[base + col] = f32_to_bf16_bits(bf16_to_f32(input[base + col]) * inv_norm);
                }
            }
        }
        other => {
            return Err(unsupported(
                "l2norm",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn mul_scalar(
    dtype: ScalarType,
    total_elems: usize,
    scalar: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::F32 => {
            let input = unsafe { f32_slice(input.as_ptr(), total_elems) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = input[i] * scalar;
            }
        }
        ScalarType::BF16 => {
            let input = unsafe { bf16_slice(input.as_ptr(), total_elems) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), total_elems) };
            for i in 0..total_elems {
                out[i] = f32_to_bf16_bits(bf16_to_f32(input[i]) * scalar);
            }
        }
        other => {
            return Err(unsupported(
                "mul_scalar",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn swiglu_mul(
    dtype: ScalarType,
    elem_count: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let gate = unsafe { bf16_slice(gate.as_ptr(), elem_count) };
            let up = unsafe { bf16_slice(up.as_ptr(), elem_count) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), elem_count) };
            for i in 0..elem_count {
                out[i] = f32_to_bf16_bits(qwen_silu(bf16_to_f32(gate[i])) * bf16_to_f32(up[i]));
            }
        }
        ScalarType::F32 => {
            let gate = unsafe { f32_slice(gate.as_ptr(), elem_count) };
            let up = unsafe { f32_slice(up.as_ptr(), elem_count) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), elem_count) };
            for i in 0..elem_count {
                out[i] = qwen_silu(gate[i]) * up[i];
            }
        }
        other => {
            return Err(unsupported(
                "swiglu_mul",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn rms_norm_gated(
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    hidden: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    match dtype {
        ScalarType::BF16 => {
            let hidden = unsafe { bf16_slice(hidden.as_ptr(), n_rows * n_cols) };
            let gate = unsafe { bf16_slice(gate.as_ptr(), n_rows * n_cols) };
            let out = unsafe { bf16_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let base = row * n_cols;
                let mut mean_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = bf16_to_f32(hidden[base + col]);
                    mean_sq += value * value;
                }
                let inv_rms = 1.0 / ((mean_sq / n_cols as f32) + eps).sqrt();
                for col in 0..n_cols {
                    let scale = read_weight_f32(weight, col)?;
                    let value = bf16_to_f32(hidden[base + col]) * inv_rms * scale;
                    out[base + col] =
                        f32_to_bf16_bits(value * qwen_silu(bf16_to_f32(gate[base + col])));
                }
            }
        }
        ScalarType::F32 => {
            let hidden = unsafe { f32_slice(hidden.as_ptr(), n_rows * n_cols) };
            let gate = unsafe { f32_slice(gate.as_ptr(), n_rows * n_cols) };
            let out = unsafe { f32_slice_mut(out.as_mut_ptr(), n_rows * n_cols) };
            for row in 0..n_rows {
                let base = row * n_cols;
                let mut mean_sq = 0.0f32;
                for col in 0..n_cols {
                    let value = hidden[base + col];
                    mean_sq += value * value;
                }
                let inv_rms = 1.0 / ((mean_sq / n_cols as f32) + eps).sqrt();
                for col in 0..n_cols {
                    let scale = read_weight_f32(weight, col)?;
                    out[base + col] =
                        hidden[base + col] * inv_rms * scale * qwen_silu(gate[base + col]);
                }
            }
        }
        other => {
            return Err(unsupported(
                "rms_norm_gated",
                format!("unsupported dtype {other:?}"),
            ))
        }
    }
    Ok(())
}

pub(crate) fn full_attention_prefill(
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
    if batch_size != 1 {
        return Err(unsupported(
            "full_attention_prefill",
            format!("batch_size={batch_size} is not supported"),
        ));
    }
    if dtype != ScalarType::BF16 {
        return Err(unsupported(
            "full_attention_prefill",
            format!("expected BF16 query/key/value, got {dtype:?}"),
        ));
    }
    if out.dtype() != ScalarType::F32 {
        return Err(unsupported(
            "full_attention_prefill",
            format!("expected F32 output, got {:?}", out.dtype()),
        ));
    }
    let num_kv_groups = q_heads / kv_heads;
    let query = unsafe { bf16_slice(query.as_ptr(), q_heads * q_len * head_dim) };
    let key = unsafe { bf16_slice(key.as_ptr(), kv_heads * kv_len * head_dim) };
    let value = unsafe { bf16_slice(value.as_ptr(), kv_heads * kv_len * head_dim) };
    let out = unsafe { f32_slice_mut(out.as_mut_ptr(), q_heads * q_len * head_dim) };
    let mut scores = vec![0.0f32; kv_len];
    let mut probs = vec![0.0f32; kv_len];
    for q_head in 0..q_heads {
        let kv_head = q_head / num_kv_groups;
        for q_pos in 0..q_len {
            let q_base = (q_head * q_len + q_pos) * head_dim;
            let max_attend = (seqlen_offset + q_pos + 1).min(kv_len);
            let mut max_score = f32::NEG_INFINITY;
            for kv_pos in 0..max_attend {
                let k_base = (kv_head * kv_len + kv_pos) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += bf16_to_f32(query[q_base + d]) * bf16_to_f32(key[k_base + d]);
                }
                let score = dot * scale;
                scores[kv_pos] = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0f32;
            for kv_pos in 0..max_attend {
                let prob = (scores[kv_pos] - max_score).exp();
                probs[kv_pos] = prob;
                denom += prob;
            }
            let out_base = (q_head * q_len + q_pos) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for kv_pos in 0..max_attend {
                    let v_base = (kv_head * kv_len + kv_pos) * head_dim;
                    acc += (probs[kv_pos] / denom) * bf16_to_f32(value[v_base + d]);
                }
                out[out_base + d] = acc;
            }
        }
    }
    Ok(())
}

pub(crate) fn delta_recurrent_prefill(
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
    if dtype != ScalarType::F32 {
        return Err(unsupported(
            "delta_recurrent_prefill",
            format!("expected F32 dtype, got {dtype:?}"),
        ));
    }
    let initial_state = unsafe {
        f32_slice(
            initial_state.as_ptr(),
            batch_heads * k_head_dim * v_head_dim,
        )
    };
    let query = unsafe { f32_slice(query.as_ptr(), batch_heads * seq_len * k_head_dim) };
    let key = unsafe { f32_slice(key.as_ptr(), batch_heads * seq_len * k_head_dim) };
    let value = unsafe { f32_slice(value.as_ptr(), batch_heads * seq_len * v_head_dim) };
    let beta = unsafe { f32_slice(beta.as_ptr(), batch_heads * seq_len) };
    let g = unsafe { f32_slice(g.as_ptr(), batch_heads * seq_len) };
    let out_rows = seq_len + k_head_dim;
    let out = unsafe { f32_slice_mut(out.as_mut_ptr(), batch_heads * out_rows * v_head_dim) };

    for head in 0..batch_heads {
        let mut state = vec![0.0f32; k_head_dim * v_head_dim];
        let state_base = head * k_head_dim * v_head_dim;
        state.copy_from_slice(&initial_state[state_base..state_base + k_head_dim * v_head_dim]);

        for t in 0..seq_len {
            let decay = g[head * seq_len + t].exp();
            for cell in &mut state {
                *cell *= decay;
            }

            let q_base = (head * seq_len + t) * k_head_dim;
            let k_base = (head * seq_len + t) * k_head_dim;
            let v_base = (head * seq_len + t) * v_head_dim;
            let beta_t = beta[head * seq_len + t];

            let mut kv_mem = vec![0.0f32; v_head_dim];
            for kk in 0..k_head_dim {
                let key_t = key[k_base + kk];
                let state_row = &state[kk * v_head_dim..(kk + 1) * v_head_dim];
                for vv in 0..v_head_dim {
                    kv_mem[vv] = state_row[vv].mul_add(key_t, kv_mem[vv]);
                }
            }

            let mut delta = vec![0.0f32; v_head_dim];
            for vv in 0..v_head_dim {
                delta[vv] = (value[v_base + vv] - kv_mem[vv]) * beta_t;
            }

            for kk in 0..k_head_dim {
                let key_t = key[k_base + kk];
                let state_row = &mut state[kk * v_head_dim..(kk + 1) * v_head_dim];
                for vv in 0..v_head_dim {
                    state_row[vv] = key_t.mul_add(delta[vv], state_row[vv]);
                }
            }

            let out_base = (head * out_rows + t) * v_head_dim;
            for vv in 0..v_head_dim {
                let mut acc = 0.0f32;
                for kk in 0..k_head_dim {
                    acc = state[kk * v_head_dim + vv].mul_add(query[q_base + kk], acc);
                }
                out[out_base + vv] = acc;
            }
        }

        for kk in 0..k_head_dim {
            let out_base = (head * out_rows + seq_len + kk) * v_head_dim;
            let state_base = kk * v_head_dim;
            out[out_base..out_base + v_head_dim]
                .copy_from_slice(&state[state_base..state_base + v_head_dim]);
        }
    }

    Ok(())
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};

    fn use_metal_backend() {
        set_backend(Backend::Metal);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16 buffer");
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    fn read_f32(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download f32 buffer");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= tol,
                "idx {idx}: expected {e}, got {a}, delta {delta} > tol {tol}"
            );
        }
    }

    #[test]
    fn metal_host_matmul_rhs_transposed_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 3],
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .expect("upload lhs");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 0.0, 1.0, 0.5, -1.0, 2.0]),
        )
        .expect("upload rhs");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2, 2]).expect("allocate out");

        matmul_rhs_transposed(ScalarType::BF16, 1, 2, 2, 3, &lhs, &rhs, &mut out)
            .expect("run matmul_rhs_transposed");

        let actual = read_bf16(&out);
        let expected = vec![4.0, 4.5, 10.0, 9.0];
        assert_close(&actual, &expected, 0.02);
    }

    #[test]
    fn metal_host_full_attention_prefill_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .expect("upload query");
        let key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .expect("upload key");
        let value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[10.0, 1.0, 1.0, 20.0]),
        )
        .expect("upload value");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 2]).expect("allocate out");

        full_attention_prefill(
            ScalarType::BF16,
            1,
            1,
            1,
            2,
            2,
            2,
            1.0,
            0,
            &query,
            &key,
            &value,
            &mut out,
        )
        .expect("run full_attention_prefill");

        let actual = read_f32(&out);
        let prob0 = 1.0f32 / (1.0 + 1.0f32.exp());
        let prob1 = 1.0 - prob0;
        let expected = vec![
            10.0,
            1.0,
            prob0 * 10.0 + prob1 * 1.0,
            prob0 * 1.0 + prob1 * 20.0,
        ];
        assert_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn metal_host_fused_rms_norm_linear_rows_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 2.0, 2.0, 2.0, 0.0, 2.0]),
        )
        .expect("upload hidden");
        let norm_weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[0.0, 0.5, -0.5]),
        )
        .expect("upload norm weight");
        let proj_weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 0.0, -1.0, 0.5, 1.0, 0.5]),
        )
        .expect("upload proj weight");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("allocate out");

        fused_rms_norm_linear_rows(
            ScalarType::BF16,
            2,
            3,
            2,
            1e-5,
            &hidden,
            &norm_weight,
            &proj_weight,
            &mut out,
        )
        .expect("run fused rms norm linear");

        let actual = read_bf16(&out);
        let row0_inv = 1.0f32 / ((3.0f32 + 1e-5).sqrt());
        let row1_inv = 1.0f32 / (((8.0f32 / 3.0f32) + 1e-5).sqrt());
        let row0 = [
            1.0 * row0_inv * 1.0,
            2.0 * row0_inv * 1.5,
            2.0 * row0_inv * 0.5,
        ];
        let row1 = [2.0 * row1_inv * 1.0, 0.0, 2.0 * row1_inv * 0.5];
        let expected = vec![
            row0[0] * 1.0 + row0[1] * 0.0 + row0[2] * -1.0,
            row0[0] * 0.5 + row0[1] * 1.0 + row0[2] * 0.5,
            row1[0] * 1.0 + row1[1] * 0.0 + row1[2] * -1.0,
            row1[0] * 0.5 + row1[1] * 1.0 + row1[2] * 0.5,
        ];
        assert_close(&actual, &expected, 0.03);
    }

    #[test]
    fn metal_host_delta_recurrent_prefill_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let initial_state =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[1, 1, 1], &f32_bytes(&[0.5]))
                .expect("upload initial_state");
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 1],
            &f32_bytes(&[2.0, 3.0]),
        )
        .expect("upload query");
        let key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 1],
            &f32_bytes(&[1.0, 2.0]),
        )
        .expect("upload key");
        let value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 1],
            &f32_bytes(&[1.0, 4.0]),
        )
        .expect("upload value");
        let beta =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[1, 2], &f32_bytes(&[0.25, 0.5]))
                .expect("upload beta");
        let g =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[1, 2], &f32_bytes(&[0.0, 0.0]))
                .expect("upload g");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3, 1]).expect("allocate out");

        delta_recurrent_prefill(
            ScalarType::F32,
            1,
            2,
            1,
            1,
            &initial_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            &mut out,
        )
        .expect("run delta_recurrent_prefill");

        let actual = read_f32(&out);
        let expected = vec![1.25, 10.125, 3.375];
        assert_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn metal_host_standalone_matvec_host_f32_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[1.0, -2.0, 0.5]),
        )
        .expect("upload input");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[2.0, 1.0, -1.0, -3.0, 0.5, 4.0]),
        )
        .expect("upload weight");

        let actual = standalone_matvec_host_f32(ScalarType::BF16, &input, &weight, 3, 2)
            .expect("run standalone_matvec_host_f32");
        let expected = vec![-0.5, -2.0];
        assert_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn metal_host_qwen_rms_norm_standalone_matvec_host_f32_matches_reference() {
        use_metal_backend();
        let ordinal = 0usize;
        let input =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2], &bf16_bytes(&[3.0, 4.0]))
                .expect("upload input");
        let norm_weight =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2], &bf16_bytes(&[0.5, -0.5]))
                .expect("upload norm weight");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 2],
            &bf16_bytes(&[2.0, -1.0, 0.25, 3.0]),
        )
        .expect("upload weight");

        let actual = qwen_rms_norm_standalone_matvec_host_f32(
            ScalarType::BF16,
            &input,
            &norm_weight,
            0.0,
            &weight,
            2,
            2,
        )
        .expect("run qwen_rms_norm_standalone_matvec_host_f32");

        let inv_rms = 1.0f32 / ((25.0f32 / 2.0).sqrt());
        let normed = [3.0 * inv_rms * 1.5, 4.0 * inv_rms * 0.5];
        let expected = vec![
            normed[0] * 2.0 + normed[1] * -1.0,
            normed[0] * 0.25 + normed[1] * 3.0,
        ];
        assert_close(&actual, &expected, 1e-6);
    }
}
