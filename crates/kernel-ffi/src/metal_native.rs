use std::ffi::{c_int, c_void};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

pub(crate) fn disabled_by_env() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_some()
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
unsafe extern "C" {
    fn supersonic_metal_matmul_rhs_transposed_bf16(
        batch_elems: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_full_attention_prefill_bf16_f32(
        q_heads: usize,
        kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        seqlen_offset: usize,
        query_ptr: *const c_void,
        key_ptr: *const c_void,
        value_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_rows_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        add_unit_offset: bool,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_linear_prefill_conv_pack_bf16(
        conv_dim: usize,
        total_len: usize,
        seq_len: usize,
        kernel_size: usize,
        mixed_ptr: *const c_void,
        weights_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_element_add_bf16(
        total_elems: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_element_add_f32(
        total_elems: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_bf16_to_bf16(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_f32_to_f32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_u32_to_u32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_bf16_to_f32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_f32_to_bf16(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_mul_scalar_bf16(
        total_elems: usize,
        scalar: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_mul_scalar_f32(
        total_elems: usize,
        scalar: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_shd_hsd_bf16(
        s: usize,
        h: usize,
        d: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_shd_hsd_f32(
        s: usize,
        h: usize,
        d: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qkv_bf16(
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src_ptr: *const c_void,
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qkv_f32(
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src_ptr: *const c_void,
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
    ) -> c_int;
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn matmul_rhs_transposed_bf16(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16
        || rhs.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_matmul_rhs_transposed_bf16(
            batch_elems,
            m,
            n,
            k,
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native matmul_rhs_transposed_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn full_attention_prefill_bf16_f32(
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
    if query.dtype() != ScalarType::BF16
        || key.dtype() != ScalarType::BF16
        || value.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_prefill expects BF16 query/key/value, got {:?}/{:?}/{:?}",
            query.dtype(),
            key.dtype(),
            value.dtype()
        )));
    }
    if out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_prefill expects F32 output, got {:?}",
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_full_attention_prefill_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
            seqlen_offset,
            query.as_ptr(),
            key.as_ptr(),
            value.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native full_attention_prefill_bf16_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn rms_norm_rows_bf16(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    add_unit_offset: bool,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if input.dtype() != ScalarType::BF16
        || weight.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_rows_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_rms_norm_rows_bf16(
            n_rows,
            n_cols,
            eps,
            add_unit_offset,
            input.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_rows_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn linear_prefill_conv_pack_bf16(
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
    mixed_qkv: &GpuBuffer,
    weights: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if mixed_qkv.dtype() != ScalarType::BF16
        || weights.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_prefill_conv_pack_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            mixed_qkv.dtype(),
            weights.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_linear_prefill_conv_pack_bf16(
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
        return Err(GpuError::Metal(format!(
            "metal native linear_prefill_conv_pack_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn element_add(
    dtype: ScalarType,
    total_elems: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native element_add supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if lhs.dtype() != dtype || rhs.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native element_add expects matching dtype {dtype:?}, got {:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_element_add_bf16(
                total_elems,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_element_add_f32(
                total_elems,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native element_add does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native element_add failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn cast(
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native cast supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if input.dtype() != input_dtype || out.dtype() != output_dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native cast expects buffer dtypes {input_dtype:?}->{output_dtype:?}, got {:?}->{:?}",
            input.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        match (input_dtype, output_dtype) {
            (ScalarType::BF16, ScalarType::BF16) => {
                supersonic_metal_cast_bf16_to_bf16(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::F32, ScalarType::F32) => {
                supersonic_metal_cast_f32_to_f32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::U32, ScalarType::U32) => {
                supersonic_metal_cast_u32_to_u32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::BF16, ScalarType::F32) => {
                supersonic_metal_cast_bf16_to_f32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::F32, ScalarType::BF16) => {
                supersonic_metal_cast_f32_to_bf16(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native cast does not support {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native cast failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn mul_scalar(
    dtype: ScalarType,
    total_elems: usize,
    scalar: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native mul_scalar supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if input.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native mul_scalar expects dtype {dtype:?}, got {:?}->{:?}",
            input.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_mul_scalar_bf16(
                total_elems,
                scalar,
                input.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_mul_scalar_f32(
                total_elems,
                scalar,
                input.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native mul_scalar does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native mul_scalar failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn transpose_shd_hsd(
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = s
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native transpose_shd_hsd shape overflows: {s}x{h}x{d}"
            ))
        })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || h > u32::MAX as usize
        || d > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native transpose_shd_hsd supports u32-sized shapes, got {s}x{h}x{d}"
        )));
    }
    if src.dtype() != dtype || dst.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native transpose_shd_hsd expects dtype {dtype:?}, got {:?}->{:?}",
            src.dtype(),
            dst.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => {
                supersonic_metal_transpose_shd_hsd_bf16(s, h, d, src.as_ptr(), dst.as_mut_ptr())
            }
            ScalarType::F32 => {
                supersonic_metal_transpose_shd_hsd_f32(s, h, d, src.as_ptr(), dst.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native transpose_shd_hsd does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native transpose_shd_hsd failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
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
    let src_stride = key_dim
        .checked_mul(2)
        .and_then(|v| v.checked_add(val_dim))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native split_qkv shape overflows: s={s} key_dim={key_dim} val_dim={val_dim}"
            ))
        })?;
    let total = s.checked_mul(src_stride).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native split_qkv total overflows: s={s} stride={src_stride}"
        ))
    })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || key_dim > u32::MAX as usize
        || val_dim > u32::MAX as usize
        || src_stride > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qkv supports u32-sized shapes, got s={s} key_dim={key_dim} val_dim={val_dim}"
        )));
    }
    if src.dtype() != dtype || q.dtype() != dtype || k.dtype() != dtype || v.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qkv expects dtype {dtype:?}, got src={:?} q={:?} k={:?} v={:?}",
            src.dtype(),
            q.dtype(),
            k.dtype(),
            v.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_split_qkv_bf16(
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_split_qkv_f32(
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native split_qkv does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native split_qkv failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn matmul_rhs_transposed_bf16(
    _batch_elems: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native matmul_rhs_transposed_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn full_attention_prefill_bf16_f32(
    _q_heads: usize,
    _kv_heads: usize,
    _q_len: usize,
    _kv_len: usize,
    _head_dim: usize,
    _scale: f32,
    _seqlen_offset: usize,
    _query: &GpuBuffer,
    _key: &GpuBuffer,
    _value: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native full_attention_prefill_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn rms_norm_rows_bf16(
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _add_unit_offset: bool,
    _input: &GpuBuffer,
    _weight: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_rows_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn linear_prefill_conv_pack_bf16(
    _conv_dim: usize,
    _total_len: usize,
    _seq_len: usize,
    _kernel_size: usize,
    _mixed_qkv: &GpuBuffer,
    _weights: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native linear_prefill_conv_pack_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn element_add(
    _dtype: ScalarType,
    _total_elems: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native element_add is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn cast(
    _input_dtype: ScalarType,
    _output_dtype: ScalarType,
    _total_elems: usize,
    _input: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal("metal native cast is not compiled".into()))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn mul_scalar(
    _dtype: ScalarType,
    _total_elems: usize,
    _scalar: f32,
    _input: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native mul_scalar is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn transpose_shd_hsd(
    _dtype: ScalarType,
    _s: usize,
    _h: usize,
    _d: usize,
    _src: &GpuBuffer,
    _dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native transpose_shd_hsd is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn split_qkv(
    _dtype: ScalarType,
    _s: usize,
    _key_dim: usize,
    _val_dim: usize,
    _src: &GpuBuffer,
    _q: &mut GpuBuffer,
    _k: &mut GpuBuffer,
    _v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native split_qkv is not compiled".into(),
    ))
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};
    use half::bf16;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
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

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_u32(buffer: &GpuBuffer) -> Vec<u32> {
        let bytes = buffer.to_host_bytes().expect("download u32 buffer");
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn metal_native_matmul_rhs_transposed_matches_reference() {
        set_backend(Backend::Metal);
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

        matmul_rhs_transposed_bf16(1, 2, 2, 3, &lhs, &rhs, &mut out).expect("run native matmul");

        let actual = read_bf16(&out);
        let expected = [4.0f32, 4.5, 10.0, 9.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_full_attention_prefill_matches_reference() {
        set_backend(Backend::Metal);
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

        full_attention_prefill_bf16_f32(1, 1, 2, 2, 2, 1.0, 0, &query, &key, &value, &mut out)
            .expect("run native full attention");

        let actual = read_f32(&out);
        let prob0 = 1.0f32 / (1.0 + 1.0f32.exp());
        let prob1 = 1.0 - prob0;
        let expected = [
            10.0f32,
            1.0,
            prob0 * 10.0 + prob1 * 1.0,
            prob0 * 1.0 + prob1 * 20.0,
        ];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-4,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_rms_norm_rows_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 2.0, 2.0, 2.0, 0.0, 2.0]),
        )
        .expect("upload input");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[0.0, 0.5, -0.5]),
        )
        .expect("upload weight");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("allocate out");

        rms_norm_rows_bf16(2, 3, 1e-5, true, &input, &weight, &mut out)
            .expect("run native rms norm");

        let actual = read_bf16(&out);
        let row0_inv = 1.0f32 / ((3.0f32 + 1e-5).sqrt());
        let row1_inv = 1.0f32 / (((8.0f32 / 3.0f32) + 1e-5).sqrt());
        let expected = [
            1.0 * row0_inv * 1.0,
            2.0 * row0_inv * 1.5,
            2.0 * row0_inv * 0.5,
            2.0 * row1_inv * 1.0,
            0.0,
            2.0 * row1_inv * 0.5,
        ];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_linear_prefill_conv_pack_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let cases = [
            (
                2usize,
                4usize,
                2usize,
                3usize,
                vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, -2.0],
                vec![1.0, 0.5, -1.0, -1.0, 0.5, 1.0],
            ),
            (
                96usize,
                11usize,
                8usize,
                4usize,
                (0..(96 * 11))
                    .map(|idx| {
                        let centered = (idx % 23) as f32 - 11.0;
                        centered / 7.0
                    })
                    .collect::<Vec<_>>(),
                (0..(96 * 4))
                    .map(|idx| {
                        let centered = (idx % 17) as f32 - 8.0;
                        centered / 5.0
                    })
                    .collect::<Vec<_>>(),
            ),
        ];

        for (case_idx, (conv_dim, total_len, seq_len, kernel_size, mixed_vals, weight_vals)) in
            cases.into_iter().enumerate()
        {
            let mixed = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[conv_dim, total_len],
                &bf16_bytes(&mixed_vals),
            )
            .expect("upload mixed");
            let weights = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[conv_dim, kernel_size],
                &bf16_bytes(&weight_vals),
            )
            .expect("upload weights");
            let mut expected = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, conv_dim])
                .expect("allocate expected");
            let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, conv_dim])
                .expect("allocate out");

            crate::metal_host::linear_prefill_conv_pack(
                ScalarType::BF16,
                1,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                &mixed,
                &weights,
                &mut expected,
            )
            .expect("run host conv");

            linear_prefill_conv_pack_bf16(
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                &mixed,
                &weights,
                &mut out,
            )
            .expect("run native conv");

            let expected = read_bf16(&expected);
            let actual = read_bf16(&out);
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 0.02,
                    "case {case_idx} idx {idx}: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_element_add_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[1.0, -2.0, 0.5, 10.0]),
        )
        .expect("upload lhs bf16");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[0.25, 3.0, -0.25, -1.5]),
        )
        .expect("upload rhs bf16");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate bf16 out");
        element_add(ScalarType::BF16, 4, &lhs, &rhs, &mut out).expect("run bf16 add");
        let actual = read_bf16(&out);
        let expected = [1.25f32, 1.0, 0.25, 8.5];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.25, -4.0, 8.0]),
        )
        .expect("upload lhs f32");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[-0.25, 2.0, 0.5]),
        )
        .expect("upload rhs f32");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        element_add(ScalarType::F32, 3, &lhs, &rhs, &mut out).expect("run f32 add");
        assert_eq!(read_f32(&out), vec![1.0, -2.0, 8.5]);
    }

    #[test]
    fn metal_native_cast_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[1.0, -2.5, 0.25]),
        )
        .expect("upload bf16 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        cast(ScalarType::BF16, ScalarType::F32, 3, &input, &mut out).expect("run bf16->f32 cast");
        assert_eq!(read_f32(&out), vec![1.0, -2.5, 0.25]);

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.0, -2.25, 3.5]),
        )
        .expect("upload f32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3]).expect("allocate bf16 out");
        cast(ScalarType::F32, ScalarType::BF16, 3, &input, &mut out).expect("run f32->bf16 cast");
        let actual = read_bf16(&out);
        for (idx, (a, e)) in actual.iter().zip([1.0f32, -2.25, 3.5].iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "f32->bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U32,
            &[3],
            &u32_bytes(&[7, 42, u32::MAX - 1]),
        )
        .expect("upload u32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::U32, &[3]).expect("allocate u32 out");
        cast(ScalarType::U32, ScalarType::U32, 3, &input, &mut out).expect("run u32 copy cast");
        assert_eq!(read_u32(&out), vec![7, 42, u32::MAX - 1]);
    }

    #[test]
    fn metal_native_mul_scalar_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[1.0, -2.0, 0.5, 8.0]),
        )
        .expect("upload bf16 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate bf16 out");
        mul_scalar(ScalarType::BF16, 4, -0.5, &input, &mut out).expect("run bf16 mul_scalar");
        let actual = read_bf16(&out);
        let expected = [-0.5f32, 1.0, -0.25, -4.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.25, -4.0, 8.0]),
        )
        .expect("upload f32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        mul_scalar(ScalarType::F32, 3, 2.0, &input, &mut out).expect("run f32 mul_scalar");
        assert_eq!(read_f32(&out), vec![2.5, -8.0, 16.0]);
    }

    #[test]
    fn metal_native_transpose_shd_hsd_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input_vals = (0..12).map(|v| v as f32).collect::<Vec<_>>();
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3, 2],
            &bf16_bytes(&input_vals),
        )
        .expect("upload bf16 input");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3, 2, 2]).expect("allocate bf16 out");
        transpose_shd_hsd(ScalarType::BF16, 2, 3, 2, &input, &mut out).expect("run bf16 transpose");
        assert_eq!(
            read_bf16(&out),
            vec![0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]
        );

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 2, 2],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .expect("upload f32 input");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2, 2]).expect("allocate f32 out");
        transpose_shd_hsd(ScalarType::F32, 2, 2, 2, &input, &mut out).expect("run f32 transpose");
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn metal_native_split_qkv_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 5],
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        .expect("upload bf16 input");
        let mut q = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("allocate bf16 q");
        let mut k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("allocate bf16 k");
        let mut v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 1]).expect("allocate bf16 v");
        split_qkv(ScalarType::BF16, 2, 2, 1, &input, &mut q, &mut k, &mut v)
            .expect("run bf16 split_qkv");
        assert_eq!(read_bf16(&q), vec![1.0, 2.0, 10.0, 20.0]);
        assert_eq!(read_bf16(&k), vec![3.0, 4.0, 30.0, 40.0]);
        assert_eq!(read_bf16(&v), vec![5.0, 50.0]);

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 7],
            &f32_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            ]),
        )
        .expect("upload f32 input");
        let mut q = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("allocate f32 q");
        let mut k = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("allocate f32 k");
        let mut v = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate f32 v");
        split_qkv(ScalarType::F32, 2, 2, 3, &input, &mut q, &mut k, &mut v)
            .expect("run f32 split_qkv");
        assert_eq!(read_f32(&q), vec![1.0, 2.0, 11.0, 12.0]);
        assert_eq!(read_f32(&k), vec![3.0, 4.0, 13.0, 14.0]);
        assert_eq!(read_f32(&v), vec![5.0, 6.0, 7.0, 15.0, 16.0, 17.0]);
    }
}
