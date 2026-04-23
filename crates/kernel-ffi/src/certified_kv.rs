use std::ffi::{c_int, c_void};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

fn certified_kv_error(backend: Backend, msg: String) -> GpuError {
    match backend {
        Backend::Hip => GpuError::Hip(msg),
        Backend::Cuda => GpuError::Cuda(msg),
        Backend::Metal => GpuError::Metal(msg),
    }
}

#[cfg(supersonic_backend_cuda)]
unsafe extern "C" {
    fn dotcache_llama31_certified_kv_quantize_bf16(
        device_ordinal: usize,
        key_bf16: *const c_void,
        value_bf16: *const c_void,
        key_int8: *mut c_void,
        key_scale: *mut c_void,
        value_int4: *mut c_void,
        value_scale: *mut c_void,
        value_zero: *mut c_void,
        value_error: *mut c_void,
        num_kv_heads: c_int,
        seq_len: c_int,
        max_t: c_int,
        head_dim: c_int,
        block_size: c_int,
        value_group_size: c_int,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_quantize_keys_bf16(
        device_ordinal: usize,
        key_bf16: *const c_void,
        key_int8: *mut c_void,
        key_scale: *mut c_void,
        num_kv_heads: c_int,
        seq_len: c_int,
        max_t: c_int,
        head_dim: c_int,
        block_size: c_int,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_score_blocks_int8(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        block_max: *mut c_void,
        block_sum: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        head_dim: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_attend_int8_int4(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        value_int4: *const c_void,
        value_scale: *const c_void,
        value_zero: *const c_void,
        score_scratch: *mut c_void,
        output_f32: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        value_int4: *const c_void,
        value_scale: *const c_void,
        value_zero: *const c_void,
        tail_key_bf16: *const c_void,
        tail_value_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_f32: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_attend_int8_bf16_values_strided(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        value_bf16: *const c_void,
        tail_key_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_f32: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        score_stride_tokens: c_int,
        value_stride_tokens: c_int,
        head_dim: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
}

pub fn aligned_tokens(seq_len: usize, block_size: usize) -> usize {
    if block_size == 0 {
        0
    } else {
        (seq_len / block_size) * block_size
    }
}

pub fn quantized_shapes(
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
    value_group_size: usize,
) -> Result<([usize; 3], [usize; 3], [usize; 3], [usize; 3], [usize; 2]), GpuError> {
    if block_size == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV block_size must be > 0".into(),
        ));
    }
    if value_group_size == 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_group_size={value_group_size} must divide head_dim={head_dim}"
        )));
    }
    if head_dim % 2 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV head_dim={head_dim} must be even for INT4 packing"
        )));
    }
    let aligned = aligned_tokens(seq_len, block_size);
    let blocks = aligned / block_size;
    let value_groups = head_dim / value_group_size;
    Ok((
        [num_kv_heads, aligned, head_dim],
        [num_kv_heads, blocks, head_dim],
        [num_kv_heads, aligned, head_dim / 2],
        [num_kv_heads, aligned, value_groups],
        [num_kv_heads, blocks],
    ))
}

pub fn quantize_bf16_cache(
    ordinal: usize,
    key_bf16: &GpuBuffer,
    value_bf16: &GpuBuffer,
    seq_len: usize,
    block_size: usize,
    value_group_size: usize,
    key_int8: &mut GpuBuffer,
    key_scale: &mut GpuBuffer,
    value_int4: &mut GpuBuffer,
    value_scale: &mut GpuBuffer,
    value_zero: &mut GpuBuffer,
    value_error: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if key_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV quantization is currently CUDA-only".into(),
        ));
    }
    if key_bf16.dtype() != ScalarType::BF16 || value_bf16.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV quantization expects BF16 key/value, got {:?}/{:?}",
            key_bf16.dtype(),
            value_bf16.dtype()
        )));
    }
    if key_bf16.shape().len() != 4 || value_bf16.shape() != key_bf16.shape() {
        return Err(GpuError::InvalidArg(format!(
            "certified KV quantization expects matching [1,nkv,max_t,hd] caches, got {:?}/{:?}",
            key_bf16.shape(),
            value_bf16.shape()
        )));
    }
    if key_bf16.shape()[0] != 1 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV quantization currently supports batch=1 caches, got shape {:?}",
            key_bf16.shape()
        )));
    }
    let num_kv_heads = key_bf16.shape()[1];
    let max_t = key_bf16.shape()[2];
    let head_dim = key_bf16.shape()[3];
    if seq_len > max_t {
        return Err(GpuError::InvalidArg(format!(
            "certified KV seq_len={seq_len} exceeds cache capacity max_t={max_t}"
        )));
    }
    let (key_i8_shape, key_scale_shape, value_i4_shape, value_meta_shape, value_error_shape) =
        quantized_shapes(
            num_kv_heads,
            seq_len,
            head_dim,
            block_size,
            value_group_size,
        )?;
    if key_int8.dtype() != ScalarType::U8 || key_int8.shape() != key_i8_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_int8 expects U8 {:?}, got {:?} {:?}",
            key_i8_shape,
            key_int8.dtype(),
            key_int8.shape()
        )));
    }
    if key_scale.dtype() != ScalarType::F32 || key_scale.shape() != key_scale_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_scale expects F32 {:?}, got {:?} {:?}",
            key_scale_shape,
            key_scale.dtype(),
            key_scale.shape()
        )));
    }
    if value_int4.dtype() != ScalarType::U8 || value_int4.shape() != value_i4_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_int4 expects U8 {:?}, got {:?} {:?}",
            value_i4_shape,
            value_int4.dtype(),
            value_int4.shape()
        )));
    }
    if value_scale.dtype() != ScalarType::F16 || value_scale.shape() != value_meta_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_scale expects F16 {:?}, got {:?} {:?}",
            value_meta_shape,
            value_scale.dtype(),
            value_scale.shape()
        )));
    }
    if value_zero.dtype() != ScalarType::F16 || value_zero.shape() != value_meta_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_zero expects F16 {:?}, got {:?} {:?}",
            value_meta_shape,
            value_zero.dtype(),
            value_zero.shape()
        )));
    }
    if value_error.dtype() != ScalarType::F32 || value_error.shape() != value_error_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_error expects F32 {:?}, got {:?} {:?}",
            value_error_shape,
            value_error.dtype(),
            value_error.shape()
        )));
    }

    let backend = key_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let status = unsafe {
                    dotcache_llama31_certified_kv_quantize_bf16(
                        ordinal,
                        key_bf16.as_ptr(),
                        value_bf16.as_ptr(),
                        key_int8.as_mut_ptr(),
                        key_scale.as_mut_ptr(),
                        value_int4.as_mut_ptr(),
                        value_scale.as_mut_ptr(),
                        value_zero.as_mut_ptr(),
                        value_error.as_mut_ptr(),
                        num_kv_heads as c_int,
                        seq_len as c_int,
                        max_t as c_int,
                        head_dim as c_int,
                        block_size as c_int,
                        value_group_size as c_int,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        backend,
                        format!("certified KV CUDA quantize failed: {status}"),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV quantization is currently CUDA-only".into(),
        )),
    }
}

pub fn quantize_bf16_keys(
    ordinal: usize,
    key_bf16: &GpuBuffer,
    seq_len: usize,
    block_size: usize,
    key_int8: &mut GpuBuffer,
    key_scale: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if key_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV key quantization is currently CUDA-only".into(),
        ));
    }
    if key_bf16.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key quantization expects BF16 key cache, got {:?}",
            key_bf16.dtype()
        )));
    }
    if key_bf16.shape().len() != 4 || key_bf16.shape()[0] != 1 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key quantization expects [1,nkv,max_t,hd] cache, got {:?}",
            key_bf16.shape()
        )));
    }
    if block_size == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV key quantization block_size must be > 0".into(),
        ));
    }
    let num_kv_heads = key_bf16.shape()[1];
    let max_t = key_bf16.shape()[2];
    let head_dim = key_bf16.shape()[3];
    if seq_len > max_t {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key quantization seq_len={seq_len} exceeds cache capacity max_t={max_t}"
        )));
    }
    let aligned = aligned_tokens(seq_len, block_size);
    let key_i8_shape = [num_kv_heads, aligned, head_dim];
    let key_scale_shape = [num_kv_heads, aligned / block_size, head_dim];
    if key_int8.dtype() != ScalarType::U8 || key_int8.shape() != key_i8_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_int8 expects U8 {:?}, got {:?} {:?}",
            key_i8_shape,
            key_int8.dtype(),
            key_int8.shape()
        )));
    }
    if key_scale.dtype() != ScalarType::F32 || key_scale.shape() != key_scale_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_scale expects F32 {:?}, got {:?} {:?}",
            key_scale_shape,
            key_scale.dtype(),
            key_scale.shape()
        )));
    }

    match key_bf16.backend() {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                let status = dotcache_llama31_certified_kv_quantize_keys_bf16(
                    ordinal,
                    key_bf16.as_ptr(),
                    key_int8.as_mut_ptr(),
                    key_scale.as_mut_ptr(),
                    num_kv_heads as c_int,
                    seq_len as c_int,
                    max_t as c_int,
                    head_dim as c_int,
                    block_size as c_int,
                );
                if status != 0 {
                    return Err(certified_kv_error(
                        Backend::Cuda,
                        format!("certified KV CUDA key quantization failed: {status}"),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV key quantization is currently CUDA-only".into(),
        )),
    }
}

pub fn score_blocks_int8(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    block_size: usize,
    gqa_group: usize,
    q_scale: f32,
    block_max: &mut GpuBuffer,
    block_sum: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV INT8 scoring is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects BF16 query, got {:?}",
            query_bf16.dtype()
        )));
    }
    if key_int8.dtype() != ScalarType::U8 || key_scale.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects U8/F32 key buffers, got {:?}/{:?}",
            key_int8.dtype(),
            key_scale.dtype()
        )));
    }
    if block_max.dtype() != ScalarType::F32 || block_sum.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects F32 outputs, got {:?}/{:?}",
            block_max.dtype(),
            block_sum.dtype()
        )));
    }
    if query_bf16.shape().len() != 2 || key_int8.shape().len() != 3 || key_scale.shape().len() != 3
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects query [qh,hd], key_int8 [kvh,t,hd], key_scale [kvh,b,hd], got {:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape()
        )));
    }
    if block_size == 0 || block_size > 256 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring block_size={block_size} must be in 1..=256"
        )));
    }
    if gqa_group == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV INT8 scoring gqa_group must be > 0".into(),
        ));
    }
    let q_heads = query_bf16.shape()[0];
    let head_dim = query_bf16.shape()[1];
    let kv_heads = key_int8.shape()[0];
    let aligned_tokens = key_int8.shape()[1];
    if key_int8.shape()[2] != head_dim {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring query head_dim={head_dim} does not match key_int8 shape {:?}",
            key_int8.shape()
        )));
    }
    if aligned_tokens % block_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring aligned_tokens={aligned_tokens} must divide block_size={block_size}"
        )));
    }
    let num_blocks = aligned_tokens / block_size;
    if q_heads != kv_heads * gqa_group {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring q_heads={q_heads} must equal kv_heads={kv_heads} * gqa_group={gqa_group}"
        )));
    }
    if key_scale.shape() != [kv_heads, num_blocks, head_dim] {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring key_scale expects {:?}, got {:?}",
            [kv_heads, num_blocks, head_dim],
            key_scale.shape()
        )));
    }
    if block_max.shape() != [q_heads, num_blocks] || block_sum.shape() != [q_heads, num_blocks] {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring outputs expect {:?}, got {:?}/{:?}",
            [q_heads, num_blocks],
            block_max.shape(),
            block_sum.shape()
        )));
    }

    let backend = query_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let status = unsafe {
                    dotcache_llama31_certified_kv_score_blocks_int8(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        block_max.as_mut_ptr(),
                        block_sum.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        head_dim as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        backend,
                        format!("certified KV CUDA INT8 scoring failed: {status}"),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV INT8 scoring is currently CUDA-only".into(),
        )),
    }
}

pub fn attend_int8_int4(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    block_size: usize,
    value_group_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV INT8/INT4 attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || value_int4.dtype() != ScalarType::U8
        || value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || score_scratch.dtype() != ScalarType::F32
        || output_f32.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention dtypes must be BF16/U8/F32/U8/F16/F16/F32/F32, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.dtype(),
            key_int8.dtype(),
            key_scale.dtype(),
            value_int4.dtype(),
            value_scale.dtype(),
            value_zero.dtype(),
            score_scratch.dtype(),
            output_f32.dtype()
        )));
    }
    if query_bf16.shape().len() != 2 || key_int8.shape().len() != 3 || key_scale.shape().len() != 3
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention expects query [qh,hd], key_int8 [kvh,t,hd], key_scale [kvh,b,hd], got {:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape()
        )));
    }
    if block_size == 0 || block_size > 256 || value_group_size == 0 || gqa_group == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention invalid block_size={block_size} value_group_size={value_group_size} gqa_group={gqa_group}"
        )));
    }
    let q_heads = query_bf16.shape()[0];
    let head_dim = query_bf16.shape()[1];
    let kv_heads = key_int8.shape()[0];
    let aligned_tokens = key_int8.shape()[1];
    if head_dim % 2 != 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention head_dim={head_dim} must be even and divisible by value_group_size={value_group_size}"
        )));
    }
    if key_int8.shape()[2] != head_dim || aligned_tokens % block_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention key shape {:?} incompatible with head_dim={head_dim} block_size={block_size}",
            key_int8.shape()
        )));
    }
    let num_blocks = aligned_tokens / block_size;
    if q_heads != kv_heads * gqa_group {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention q_heads={q_heads} must equal kv_heads={kv_heads} * gqa_group={gqa_group}"
        )));
    }
    let value_groups = head_dim / value_group_size;
    if key_scale.shape() != [kv_heads, num_blocks, head_dim]
        || value_int4.shape() != [kv_heads, aligned_tokens, head_dim / 2]
        || value_scale.shape() != [kv_heads, aligned_tokens, value_groups]
        || value_zero.shape() != [kv_heads, aligned_tokens, value_groups]
        || score_scratch.shape() != [q_heads, aligned_tokens]
        || output_f32.shape() != [q_heads, head_dim]
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV attention shape mismatch key_scale={:?} value_int4={:?} value_scale={:?} value_zero={:?} score_scratch={:?} output={:?}",
            key_scale.shape(),
            value_int4.shape(),
            value_scale.shape(),
            value_zero.shape(),
            score_scratch.shape(),
            output_f32.shape()
        )));
    }

    let backend = query_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let status = unsafe {
                    dotcache_llama31_certified_kv_attend_int8_int4(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        value_int4.as_ptr(),
                        value_scale.as_ptr(),
                        value_zero.as_ptr(),
                        score_scratch.as_mut_ptr(),
                        output_f32.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        head_dim as c_int,
                        value_group_size as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        backend,
                        format!("certified KV CUDA INT8/INT4 attention failed: {status}"),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV INT8/INT4 attention is currently CUDA-only".into(),
        )),
    }
}

pub fn attend_int8_int4_with_bf16_tail(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    tail_key_bf16: &GpuBuffer,
    tail_value_bf16: &GpuBuffer,
    block_size: usize,
    value_group_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV hybrid attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || value_int4.dtype() != ScalarType::U8
        || value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || tail_key_bf16.dtype() != ScalarType::BF16
        || tail_value_bf16.dtype() != ScalarType::BF16
        || score_scratch.dtype() != ScalarType::F32
        || output_f32.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid dtypes must be BF16/U8/F32/U8/F16/F16/BF16/BF16/F32/F32, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.dtype(),
            key_int8.dtype(),
            key_scale.dtype(),
            value_int4.dtype(),
            value_scale.dtype(),
            value_zero.dtype(),
            tail_key_bf16.dtype(),
            tail_value_bf16.dtype(),
            score_scratch.dtype(),
            output_f32.dtype()
        )));
    }
    if query_bf16.shape().len() != 2
        || key_int8.shape().len() != 3
        || key_scale.shape().len() != 3
        || tail_key_bf16.shape().len() != 3
        || tail_value_bf16.shape() != tail_key_bf16.shape()
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid expects query [qh,hd], compressed key/value, tail [kvh,tail,hd], got {:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape(),
            tail_key_bf16.shape(),
            tail_value_bf16.shape()
        )));
    }
    if block_size == 0 || block_size > 256 || value_group_size == 0 || gqa_group == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid invalid block_size={block_size} value_group_size={value_group_size} gqa_group={gqa_group}"
        )));
    }
    let q_heads = query_bf16.shape()[0];
    let head_dim = query_bf16.shape()[1];
    let kv_heads = key_int8.shape()[0];
    let aligned_tokens = key_int8.shape()[1];
    let tail_len = tail_key_bf16.shape()[1];
    if tail_len == 0 {
        return attend_int8_int4(
            ordinal,
            query_bf16,
            key_int8,
            key_scale,
            value_int4,
            value_scale,
            value_zero,
            block_size,
            value_group_size,
            gqa_group,
            q_scale,
            score_scratch,
            output_f32,
        );
    }
    if head_dim % 2 != 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid head_dim={head_dim} must be even and divisible by value_group_size={value_group_size}"
        )));
    }
    if key_int8.shape()[2] != head_dim
        || aligned_tokens % block_size != 0
        || aligned_tokens == 0
        || tail_key_bf16.shape()[0] != kv_heads
        || tail_key_bf16.shape()[2] != head_dim
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid key/tail shapes incompatible: key={:?} tail={:?} head_dim={head_dim} block_size={block_size}",
            key_int8.shape(),
            tail_key_bf16.shape()
        )));
    }
    let num_blocks = aligned_tokens / block_size;
    if q_heads != kv_heads * gqa_group {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid q_heads={q_heads} must equal kv_heads={kv_heads} * gqa_group={gqa_group}"
        )));
    }
    let value_groups = head_dim / value_group_size;
    if key_scale.shape() != [kv_heads, num_blocks, head_dim]
        || value_int4.shape() != [kv_heads, aligned_tokens, head_dim / 2]
        || value_scale.shape() != [kv_heads, aligned_tokens, value_groups]
        || value_zero.shape() != [kv_heads, aligned_tokens, value_groups]
        || score_scratch.shape() != [q_heads, aligned_tokens + tail_len]
        || output_f32.shape() != [q_heads, head_dim]
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV hybrid shape mismatch key_scale={:?} value_int4={:?} value_scale={:?} value_zero={:?} score_scratch={:?} output={:?}",
            key_scale.shape(),
            value_int4.shape(),
            value_scale.shape(),
            value_zero.shape(),
            score_scratch.shape(),
            output_f32.shape()
        )));
    }

    let backend = query_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let status = unsafe {
                    dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        value_int4.as_ptr(),
                        value_scale.as_ptr(),
                        value_zero.as_ptr(),
                        tail_key_bf16.as_ptr(),
                        tail_value_bf16.as_ptr(),
                        score_scratch.as_mut_ptr(),
                        output_f32.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        tail_len as c_int,
                        head_dim as c_int,
                        value_group_size as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        backend,
                        format!("certified KV CUDA hybrid attention failed: {status}"),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV hybrid attention is currently CUDA-only".into(),
        )),
    }
}

pub fn attend_int8_bf16_values(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    value_bf16: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    block_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if value_bf16.shape().len() != 3 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8/BF16-value contiguous wrapper expects value [kvh,total,hd], got {:?}",
            value_bf16.shape()
        )));
    }
    attend_int8_bf16_values_strided(
        ordinal,
        query_bf16,
        key_int8,
        key_scale,
        value_bf16,
        tail_key_bf16,
        value_bf16.shape()[1],
        block_size,
        gqa_group,
        q_scale,
        score_scratch,
        output_f32,
    )
}

pub fn attend_int8_bf16_values_strided(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    value_bf16: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    total_tokens: usize,
    block_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV strided INT8/BF16-value attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || value_bf16.dtype() != ScalarType::BF16
        || score_scratch.dtype() != ScalarType::F32
        || output_f32.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value dtypes must be BF16/U8/F32/BF16/F32/F32, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.dtype(),
            key_int8.dtype(),
            key_scale.dtype(),
            value_bf16.dtype(),
            score_scratch.dtype(),
            output_f32.dtype()
        )));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if !query_is_2d
        && !query_is_3d
        || key_int8.shape().len() != 3
        || key_scale.shape().len() != 3
        || (value_bf16.shape().len() != 3
            && !(value_bf16.shape().len() == 4 && value_bf16.shape()[0] == 1))
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value expects query [qh,hd] or [qh,1,hd], key_int8 [kvh,aligned,hd], key_scale [kvh,b,hd], value [kvh,stride,hd] or [1,kvh,stride,hd], got {:?}/{:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape(),
            value_bf16.shape()
        )));
    }
    if block_size == 0 || block_size > 256 || gqa_group == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value invalid block_size={block_size} gqa_group={gqa_group}"
        )));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let kv_heads = key_int8.shape()[0];
    let aligned_tokens = key_int8.shape()[1];
    if key_int8.shape()[2] != head_dim || aligned_tokens == 0 || aligned_tokens % block_size != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value key shape {:?} incompatible with head_dim={head_dim} block_size={block_size}",
            key_int8.shape()
        )));
    }
    let num_blocks = aligned_tokens / block_size;
    let value_shape = value_bf16.shape();
    let (value_kv_heads, value_stride_tokens, value_head_dim) = if value_shape.len() == 3 {
        (value_shape[0], value_shape[1], value_shape[2])
    } else {
        (value_shape[1], value_shape[2], value_shape[3])
    };
    if value_kv_heads != kv_heads || value_head_dim != head_dim {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value value shape {:?} incompatible with kv_heads={kv_heads} head_dim={head_dim}",
            value_bf16.shape()
        )));
    }
    if total_tokens < aligned_tokens || value_stride_tokens < total_tokens {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value needs aligned_tokens={aligned_tokens} <= total_tokens={total_tokens} <= value_stride_tokens={value_stride_tokens}"
        )));
    }
    let tail_len = total_tokens - aligned_tokens;
    if q_heads != kv_heads * gqa_group {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value q_heads={q_heads} must equal kv_heads={kv_heads} * gqa_group={gqa_group}"
        )));
    }
    let output_shape = output_f32.shape();
    let output_is_2d = output_shape == [q_heads, head_dim];
    let output_is_3d = output_shape == [q_heads, 1, head_dim];
    let score_shape = score_scratch.shape();
    let score_stride_tokens = if score_shape.len() == 2 && score_shape[0] == q_heads {
        score_shape[1]
    } else {
        0
    };
    if key_scale.shape() != [kv_heads, num_blocks, head_dim]
        || score_stride_tokens < total_tokens
        || (!output_is_2d && !output_is_3d)
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value shape mismatch key_scale={:?} score_scratch={:?} output={:?}",
            key_scale.shape(),
            score_scratch.shape(),
            output_f32.shape()
        )));
    }
    if let Some(tail_key) = tail_key_bf16 {
        if tail_key.dtype() != ScalarType::BF16
            || tail_key.shape() != [kv_heads, tail_len, head_dim]
        {
            return Err(GpuError::InvalidArg(format!(
                "certified KV strided INT8/BF16-value tail key expects BF16 [{kv_heads}, {tail_len}, {head_dim}], got {:?} {:?}",
                tail_key.dtype(),
                tail_key.shape()
            )));
        }
    } else if tail_len != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value needs tail key for tail_len={tail_len}"
        )));
    }

    let backend = query_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let tail_key_ptr = tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr);
                let status = unsafe {
                    dotcache_llama31_certified_kv_attend_int8_bf16_values_strided(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        value_bf16.as_ptr(),
                        tail_key_ptr,
                        score_scratch.as_mut_ptr(),
                        output_f32.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        tail_len as c_int,
                        score_stride_tokens as c_int,
                        value_stride_tokens as c_int,
                        head_dim as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        backend,
                        format!(
                            "certified KV CUDA strided INT8/BF16-value attention failed: {status}"
                        ),
                    ));
                }
                Ok(())
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip | Backend::Metal => Err(GpuError::InvalidArg(
            "certified KV strided INT8/BF16-value attention is currently CUDA-only".into(),
        )),
    }
}

#[cfg(all(test, supersonic_backend_cuda))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};
    use half::{bf16, f16};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn f16s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    #[test]
    fn cuda_quantize_bf16_cache_matches_simple_oracle_shapes() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let num_kv_heads = 1usize;
        let max_t = 4usize;
        let head_dim = 4usize;
        let block_size = 2usize;
        let value_group_size = 2usize;
        let seq_len = 4usize;
        let key_values = [
            1.0, -2.0, 0.5, -0.5, 2.0, -1.0, 1.5, -1.5, -4.0, 3.0, -2.0, 1.0, 1.0, -3.0, 2.0, -1.0,
        ];
        let value_values = [
            0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, -1.0, 1.0, -2.0, 2.0, 0.0, 2.0, -1.0, 3.0,
        ];
        let key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, num_kv_heads, max_t, head_dim],
            &bf16_bytes(&key_values),
        )
        .expect("upload key");
        let value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, num_kv_heads, max_t, head_dim],
            &bf16_bytes(&value_values),
        )
        .expect("upload value");

        let (key_i8_shape, key_scale_shape, value_i4_shape, value_meta_shape, value_error_shape) =
            quantized_shapes(
                num_kv_heads,
                seq_len,
                head_dim,
                block_size,
                value_group_size,
            )
            .expect("shapes");
        let mut key_i8 = GpuBuffer::zeros(ordinal, ScalarType::U8, &key_i8_shape).expect("key_i8");
        let mut key_scale =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &key_scale_shape).expect("key_scale");
        let mut value_i4 =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &value_i4_shape).expect("value_i4");
        let mut value_scale =
            GpuBuffer::zeros(ordinal, ScalarType::F16, &value_meta_shape).expect("value_scale");
        let mut value_zero =
            GpuBuffer::zeros(ordinal, ScalarType::F16, &value_meta_shape).expect("value_zero");
        let mut value_error =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &value_error_shape).expect("value_error");

        quantize_bf16_cache(
            ordinal,
            &key,
            &value,
            seq_len,
            block_size,
            value_group_size,
            &mut key_i8,
            &mut key_scale,
            &mut value_i4,
            &mut value_scale,
            &mut value_zero,
            &mut value_error,
        )
        .expect("quantize");

        let key_scales = f32s(&key_scale.to_host_bytes().expect("download key_scale"));
        assert!((key_scales[0] - (2.0 / 127.0)).abs() < 1.0e-6);
        assert!((key_scales[4] - (4.0 / 127.0)).abs() < 1.0e-6);
        let key_q = key_i8.to_host_bytes().expect("download key_i8");
        assert_eq!(key_q[0] as i8, 64);
        assert_eq!(key_q[1] as i8, -127);

        let value_q = value_i4.to_host_bytes().expect("download value_i4");
        assert_eq!(value_q[0], 0xf0);
        assert_eq!(value_q[1], 0xf0);
        let value_scales = f16s(&value_scale.to_host_bytes().expect("download value_scale"));
        let value_zeros = f16s(&value_zero.to_host_bytes().expect("download value_zero"));
        assert!((value_scales[0] - (1.0 / 15.0)).abs() < 0.001);
        assert_eq!(value_zeros[0], 0.0);
        let errors = f32s(&value_error.to_host_bytes().expect("download value_error"));
        assert!(
            errors[0] <= 0.02,
            "unexpected block 0 value error {}",
            errors[0]
        );

        let mut key_only_i8 =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &key_i8_shape).expect("key_only_i8");
        let mut key_only_scale =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &key_scale_shape).expect("key_only_scale");
        quantize_bf16_keys(
            ordinal,
            &key,
            seq_len,
            block_size,
            &mut key_only_i8,
            &mut key_only_scale,
        )
        .expect("quantize keys");
        assert_eq!(
            key_only_i8.to_host_bytes().expect("download key_only_i8"),
            key_i8.to_host_bytes().expect("download key_i8 again")
        );
        assert_eq!(
            key_only_scale
                .to_host_bytes()
                .expect("download key_only_scale"),
            key_scale.to_host_bytes().expect("download key_scale again")
        );
    }

    #[test]
    fn cuda_score_blocks_int8_matches_hand_computed_logits() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 4],
            &bf16_bytes(&[1.0, 0.5, -1.0, 2.0, -1.0, 1.0, 0.25, 0.5]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 4],
            &[
                10i8 as u8,
                -20i8 as u8,
                30i8 as u8,
                -40i8 as u8,
                -5i8 as u8,
                15i8 as u8,
                -25i8 as u8,
                35i8 as u8,
            ],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 1, 4],
            &f32_bytes(&[0.1, 0.2, 0.05, 0.25]),
        )
        .expect("upload key_scale");
        let mut block_max = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 1]).expect("block_max");
        let mut block_sum = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 1]).expect("block_sum");

        score_blocks_int8(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            2,
            2,
            1.0,
            &mut block_max,
            &mut block_sum,
        )
        .expect("score blocks");

        let maxes = f32s(&block_max.to_host_bytes().expect("download block_max"));
        let sums = f32s(&block_sum.to_host_bytes().expect("download block_sum"));
        let expected_maxes = [19.75_f32, 7.5625_f32];
        let expected_sums = [
            (-22.5_f32 - 19.75_f32).exp() + 1.0,
            (-9.625_f32 - 7.5625_f32).exp() + 1.0,
        ];
        for (got, expected) in maxes.iter().zip(expected_maxes) {
            assert!(
                (got - expected).abs() < 0.02,
                "unexpected block max got={got} expected={expected}"
            );
        }
        for (got, expected) in sums.iter().zip(expected_sums) {
            assert!(
                (got - expected).abs() < 0.001,
                "unexpected block sum got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn cuda_attend_int8_int4_matches_dominant_token_reference() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 4],
            &bf16_bytes(&[1.0, 0.5, -1.0, 2.0, -1.0, 1.0, 0.25, 0.5]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 4],
            &[
                10i8 as u8,
                -20i8 as u8,
                30i8 as u8,
                -40i8 as u8,
                -5i8 as u8,
                15i8 as u8,
                -25i8 as u8,
                35i8 as u8,
            ],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 1, 4],
            &f32_bytes(&[0.1, 0.2, 0.05, 0.25]),
        )
        .expect("upload key_scale");
        let value_i4 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 2],
            &[0xf0, 0xf0, 0xf0, 0xf0],
        )
        .expect("upload value_i4");
        let value_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 2, 2],
            &f16_bytes(&[1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0]),
        )
        .expect("upload value_scale");
        let value_zero = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 2, 2],
            &f16_bytes(&[0.0, 2.0, 4.0, 6.0]),
        )
        .expect("upload value_zero");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 4]).expect("output");

        attend_int8_int4(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &value_i4,
            &value_scale,
            &value_zero,
            2,
            2,
            2,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("attend");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for row in 0..2 {
            let got = &out[row * 4..row * 4 + 4];
            for (value, expected) in got.iter().zip([4.0_f32, 5.0, 6.0, 7.0]) {
                assert!(
                    (value - expected).abs() < 0.01,
                    "row {row} unexpected value got={value} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn cuda_hybrid_attend_includes_bf16_tail_in_softmax() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 4],
            &bf16_bytes(&[1.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 4],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 1, 4],
            &f32_bytes(&[1.0, 1.0, 1.0, 1.0]),
        )
        .expect("upload key_scale");
        let value_i4 =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[1, 2, 2], &[0, 0, 0, 0])
                .expect("upload value_i4");
        let value_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 2, 2],
            &f16_bytes(&[1.0, 1.0, 1.0, 1.0]),
        )
        .expect("upload value_scale");
        let value_zero = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 2, 2],
            &f16_bytes(&[0.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload value_zero");
        let tail_key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4],
            &bf16_bytes(&[10.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload tail_key");
        let tail_value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4],
            &bf16_bytes(&[7.0, 8.0, 9.0, 10.0]),
        )
        .expect("upload tail_value");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("output");

        attend_int8_int4_with_bf16_tail(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &value_i4,
            &value_scale,
            &value_zero,
            &tail_key,
            &tail_value,
            2,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("hybrid attend");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for (value, expected) in out.iter().zip([7.0_f32, 8.0, 9.0, 10.0]) {
            assert!(
                (value - expected).abs() < 0.01,
                "tail should dominate hybrid softmax: got={value} expected={expected}"
            );
        }
    }

    #[test]
    fn cuda_int8_key_bf16_value_attend_uses_full_precision_values() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 4],
            &bf16_bytes(&[1.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 4],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 1, 4],
            &f32_bytes(&[1.0, 1.0, 1.0, 1.0]),
        )
        .expect("upload key_scale");
        let values = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 3, 4],
            &bf16_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
        )
        .expect("upload values");
        let tail_key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4],
            &bf16_bytes(&[10.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload tail_key");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("output");

        attend_int8_bf16_values(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &values,
            Some(&tail_key),
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("attend int8 keys with bf16 values");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for (value, expected) in out.iter().zip([9.0_f32, 10.0, 11.0, 12.0]) {
            assert!(
                (value - expected).abs() < 0.01,
                "tail should dominate INT8-key/BF16-value softmax: got={value} expected={expected}"
            );
        }
    }

    #[test]
    fn cuda_int8_key_bf16_value_attend_accepts_strided_values() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 4],
            &bf16_bytes(&[1.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 2, 4],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 1, 4],
            &f32_bytes(&[1.0, 1.0, 1.0, 1.0]),
        )
        .expect("upload key_scale");
        let values = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 5, 4],
            &bf16_bytes(&[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                -100.0, -100.0, -100.0, -100.0,
                -200.0, -200.0, -200.0, -200.0,
            ]),
        )
        .expect("upload values");
        let tail_key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4],
            &bf16_bytes(&[10.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload tail_key");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("output");

        attend_int8_bf16_values_strided(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &values,
            Some(&tail_key),
            3,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("attend int8 keys with strided bf16 values");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for (value, expected) in out.iter().zip([9.0_f32, 10.0, 11.0, 12.0]) {
            assert!(
                (value - expected).abs() < 0.01,
                "tail should dominate strided INT8-key/BF16-value softmax: got={value} expected={expected}"
            );
        }
    }
}
