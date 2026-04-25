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
        key_zero: *mut c_void,
        value_int4: *mut c_void,
        value_scale: *mut c_void,
        value_zero: *mut c_void,
        value_error: *mut c_void,
        value_norm: *mut c_void,
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
        key_zero: *mut c_void,
        num_kv_heads: c_int,
        seq_len: c_int,
        max_t: c_int,
        head_dim: c_int,
        block_size: c_int,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_quantize_keys_bf16_range(
        device_ordinal: usize,
        key_bf16: *const c_void,
        key_int8: *mut c_void,
        key_scale: *mut c_void,
        key_zero: *mut c_void,
        num_kv_heads: c_int,
        max_t: c_int,
        key_stride_tokens: c_int,
        scale_stride_blocks: c_int,
        start_block: c_int,
        block_count: c_int,
        head_dim: c_int,
        block_size: c_int,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_quantize_values_bf16_range(
        device_ordinal: usize,
        value_bf16: *const c_void,
        value_int4: *mut c_void,
        value_scale: *mut c_void,
        value_zero: *mut c_void,
        value_error: *mut c_void,
        value_norm: *mut c_void,
        num_kv_heads: c_int,
        max_t: c_int,
        value_stride_tokens: c_int,
        value_error_stride_blocks: c_int,
        start_block: c_int,
        block_count: c_int,
        head_dim: c_int,
        block_size: c_int,
        value_group_size: c_int,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_score_blocks_int8(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        block_max: *mut c_void,
        block_sum: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        head_dim: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_score_consistency(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        promoted_key_bf16: *const c_void,
        promote_index: *const c_void,
        violation_flags: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        max_promoted_blocks: c_int,
        head_dim: c_int,
        gqa_group: c_int,
        q_scale: f32,
        eps_guard: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_gather_promoted_bf16(
        device_ordinal: usize,
        tier2_key_bf16: *const c_void,
        tier2_value_bf16: *const c_void,
        promote_index: *const c_void,
        value_promote_index: *const c_void,
        promoted_key_bf16: *mut c_void,
        promoted_value_bf16: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        cap_tokens: c_int,
        promoted_key_heads: c_int,
        max_promoted_blocks: c_int,
        max_promoted_value_blocks: c_int,
        head_dim: c_int,
        gqa_group: c_int,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_gather_promoted_values_bf16(
        device_ordinal: usize,
        tier2_value_bf16: *const c_void,
        value_promote_index: *const c_void,
        promoted_value_bf16: *mut c_void,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        cap_tokens: c_int,
        max_promoted_value_blocks: c_int,
        head_dim: c_int,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_selected_fp16_log_masses(
        device_ordinal: usize,
        query_bf16: *const c_void,
        promoted_key_bf16: *const c_void,
        promote_index: *const c_void,
        out_log_masses: *mut c_void,
        q_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        max_promoted_blocks: c_int,
        head_dim: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_attend_int8_int4(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
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
        key_zero: *const c_void,
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
    fn dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
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
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        value_stride_tokens: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        tail_value_start_tokens: c_int,
        tail_value_stride_tokens: c_int,
        score_stride_tokens: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided_out_bf16(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        value_int4: *const c_void,
        value_scale: *const c_void,
        value_zero: *const c_void,
        tail_key_bf16: *const c_void,
        tail_value_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_bf16: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        value_stride_tokens: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        tail_value_start_tokens: c_int,
        tail_value_stride_tokens: c_int,
        score_stride_tokens: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_attend_mixed_key_int4_bf16_tail_strided_out_bf16(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        promoted_key_bf16: *const c_void,
        promote_index: *const c_void,
        promoted_value_bf16: *const c_void,
        value_promote_index: *const c_void,
        value_int4: *const c_void,
        value_scale: *const c_void,
        value_zero: *const c_void,
        tail_key_bf16: *const c_void,
        tail_value_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_bf16: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        promoted_key_heads: c_int,
        max_promoted_blocks: c_int,
        max_promoted_value_blocks: c_int,
        value_stride_tokens: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        tail_value_start_tokens: c_int,
        tail_value_stride_tokens: c_int,
        score_stride_tokens: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_attend_all_promoted_int4_bf16_tail_out_bf16(
        device_ordinal: usize,
        query_bf16: *const c_void,
        promoted_key_bf16: *const c_void,
        promoted_value_bf16: *const c_void,
        value_promote_index: *const c_void,
        value_int4: *const c_void,
        value_scale: *const c_void,
        value_zero: *const c_void,
        tail_key_bf16: *const c_void,
        tail_value_bf16: *const c_void,
        score_scratch: *mut c_void,
        softmax_stats: *mut c_void,
        output_bf16: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        max_promoted_value_blocks: c_int,
        value_stride_tokens: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        tail_value_start_tokens: c_int,
        tail_value_stride_tokens: c_int,
        score_stride_tokens: c_int,
        head_dim: c_int,
        value_group_size: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_dense_selected_heads_out_bf16(
        device_ordinal: usize,
        query_bf16: *const c_void,
        fallback_heads: *const c_void,
        fallback_kv_slots: *const c_void,
        fallback_kv_heads: *const c_void,
        fallback_key_bf16: *const c_void,
        fallback_value_bf16: *const c_void,
        tail_key_bf16: *const c_void,
        tail_value_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_bf16: *mut c_void,
        q_heads: c_int,
        fallback_count: c_int,
        fallback_kv_count: c_int,
        prefix_tokens: c_int,
        tail_len: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        tail_value_start_tokens: c_int,
        tail_value_stride_tokens: c_int,
        score_stride_tokens: c_int,
        head_dim: c_int,
        q_scale: f32,
    ) -> c_int;

    fn dotcache_llama31_certified_kv_attend_int8_bf16_values_strided(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        value_bf16: *const c_void,
        tail_key_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_f32: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
        score_stride_tokens: c_int,
        value_stride_tokens: c_int,
        head_dim: c_int,
        gqa_group: c_int,
        q_scale: f32,
    ) -> c_int;
    fn dotcache_llama31_certified_kv_attend_int8_bf16_values_strided_out_bf16(
        device_ordinal: usize,
        query_bf16: *const c_void,
        key_int8: *const c_void,
        key_scale: *const c_void,
        key_zero: *const c_void,
        value_bf16: *const c_void,
        tail_key_bf16: *const c_void,
        score_scratch: *mut c_void,
        output_bf16: *mut c_void,
        q_heads: c_int,
        kv_heads: c_int,
        num_blocks: c_int,
        block_size: c_int,
        tail_len: c_int,
        key_stride_tokens: c_int,
        key_scale_stride_blocks: c_int,
        tail_key_start_tokens: c_int,
        tail_key_stride_tokens: c_int,
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
    key_zero: &mut GpuBuffer,
    value_int4: &mut GpuBuffer,
    value_scale: &mut GpuBuffer,
    value_zero: &mut GpuBuffer,
    value_error: &mut GpuBuffer,
    value_norm: &mut GpuBuffer,
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
    if key_zero.dtype() != ScalarType::F32 || key_zero.shape() != key_scale_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_zero expects F32 {:?}, got {:?} {:?}",
            key_scale_shape,
            key_zero.dtype(),
            key_zero.shape()
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
    if value_norm.dtype() != ScalarType::F32 || value_norm.shape() != value_error_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value_norm expects F32 {:?}, got {:?} {:?}",
            value_error_shape,
            value_norm.dtype(),
            value_norm.shape()
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
                        key_zero.as_mut_ptr(),
                        value_int4.as_mut_ptr(),
                        value_scale.as_mut_ptr(),
                        value_zero.as_mut_ptr(),
                        value_error.as_mut_ptr(),
                        value_norm.as_mut_ptr(),
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
    key_zero: &mut GpuBuffer,
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
    if key_zero.dtype() != ScalarType::F32 || key_zero.shape() != key_scale_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key_zero expects F32 {:?}, got {:?} {:?}",
            key_scale_shape,
            key_zero.dtype(),
            key_zero.shape()
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
                    key_zero.as_mut_ptr(),
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

pub fn quantize_bf16_keys_range(
    ordinal: usize,
    key_bf16: &GpuBuffer,
    start_block: usize,
    block_count: usize,
    block_size: usize,
    key_int8: &mut GpuBuffer,
    key_scale: &mut GpuBuffer,
    key_zero: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if key_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV key range quantization is currently CUDA-only".into(),
        ));
    }
    if key_bf16.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization expects BF16 key cache, got {:?}",
            key_bf16.dtype()
        )));
    }
    if key_bf16.shape().len() != 4 || key_bf16.shape()[0] != 1 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization expects [1,nkv,max_t,hd] cache, got {:?}",
            key_bf16.shape()
        )));
    }
    if block_size == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV key range quantization block_size must be > 0".into(),
        ));
    }
    let num_kv_heads = key_bf16.shape()[1];
    let max_t = key_bf16.shape()[2];
    let head_dim = key_bf16.shape()[3];
    let key_shape = key_int8.shape();
    let scale_shape = key_scale.shape();
    if key_int8.dtype() != ScalarType::U8
        || key_shape.len() != 3
        || key_shape[0] != num_kv_heads
        || key_shape[2] != head_dim
        || key_shape[1] % block_size != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization key_int8 expects U8 [{num_kv_heads}, stride, {head_dim}], got {:?} {:?}",
            key_int8.dtype(),
            key_int8.shape()
        )));
    }
    if key_scale.dtype() != ScalarType::F32
        || scale_shape.len() != 3
        || scale_shape[0] != num_kv_heads
        || scale_shape[2] != head_dim
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization key_scale expects F32 [{num_kv_heads}, blocks, {head_dim}], got {:?} {:?}",
            key_scale.dtype(),
            key_scale.shape()
        )));
    }
    if key_zero.dtype() != ScalarType::F32 || key_zero.shape() != scale_shape {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization key_zero expects F32 {:?}, got {:?} {:?}",
            scale_shape,
            key_zero.dtype(),
            key_zero.shape()
        )));
    }
    let end_block = start_block.checked_add(block_count).ok_or_else(|| {
        GpuError::InvalidArg("certified KV key range quantization block range overflow".into())
    })?;
    let key_stride_tokens = key_shape[1];
    let scale_stride_blocks = scale_shape[1];
    if end_block > scale_stride_blocks
        || end_block * block_size > key_stride_tokens
        || end_block * block_size > max_t
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV key range quantization range start_block={start_block} block_count={block_count} exceeds key_stride={key_stride_tokens} scale_blocks={scale_stride_blocks} max_t={max_t}"
        )));
    }

    match key_bf16.backend() {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                let status = dotcache_llama31_certified_kv_quantize_keys_bf16_range(
                    ordinal,
                    key_bf16.as_ptr(),
                    key_int8.as_mut_ptr(),
                    key_scale.as_mut_ptr(),
                    key_zero.as_mut_ptr(),
                    num_kv_heads as c_int,
                    max_t as c_int,
                    key_stride_tokens as c_int,
                    scale_stride_blocks as c_int,
                    start_block as c_int,
                    block_count as c_int,
                    head_dim as c_int,
                    block_size as c_int,
                );
                if status != 0 {
                    return Err(certified_kv_error(
                        Backend::Cuda,
                        format!("certified KV CUDA key range quantization failed: {status}"),
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
            "certified KV key range quantization is currently CUDA-only".into(),
        )),
    }
}

pub fn quantize_bf16_values_range(
    ordinal: usize,
    value_bf16: &GpuBuffer,
    start_block: usize,
    block_count: usize,
    block_size: usize,
    value_group_size: usize,
    value_int4: &mut GpuBuffer,
    value_scale: &mut GpuBuffer,
    value_zero: &mut GpuBuffer,
    value_error: &mut GpuBuffer,
    value_norm: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if value_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV value range quantization is currently CUDA-only".into(),
        ));
    }
    if value_bf16.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization expects BF16 value cache, got {:?}",
            value_bf16.dtype()
        )));
    }
    if value_bf16.shape().len() != 4 || value_bf16.shape()[0] != 1 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization expects [1,nkv,max_t,hd] cache, got {:?}",
            value_bf16.shape()
        )));
    }
    if block_size == 0 || value_group_size == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization invalid block_size={block_size} value_group_size={value_group_size}"
        )));
    }
    let num_kv_heads = value_bf16.shape()[1];
    let max_t = value_bf16.shape()[2];
    let head_dim = value_bf16.shape()[3];
    if head_dim % 2 != 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization head_dim={head_dim} incompatible with value_group_size={value_group_size}"
        )));
    }
    let groups = head_dim / value_group_size;
    let value_shape = value_int4.shape();
    let scale_shape = value_scale.shape();
    if value_int4.dtype() != ScalarType::U8
        || value_shape.len() != 3
        || value_shape[0] != num_kv_heads
        || value_shape[2] != head_dim / 2
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization value_int4 expects U8 [{num_kv_heads}, stride, {}], got {:?} {:?}",
            head_dim / 2,
            value_int4.dtype(),
            value_int4.shape()
        )));
    }
    if value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || scale_shape.len() != 3
        || scale_shape[0] != num_kv_heads
        || scale_shape[2] != groups
        || value_zero.shape() != scale_shape
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization value_scale/value_zero expect F16 [{num_kv_heads}, stride, {groups}], got {:?} {:?} / {:?} {:?}",
            value_scale.dtype(),
            value_scale.shape(),
            value_zero.dtype(),
            value_zero.shape()
        )));
    }
    if value_error.dtype() != ScalarType::F32
        || value_error.shape().len() != 2
        || value_error.shape()[0] != num_kv_heads
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization value_error expects F32 [{num_kv_heads}, blocks], got {:?} {:?}",
            value_error.dtype(),
            value_error.shape()
        )));
    }
    if value_norm.dtype() != ScalarType::F32 || value_norm.shape() != value_error.shape() {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization value_norm expects F32 {:?}, got {:?} {:?}",
            value_error.shape(),
            value_norm.dtype(),
            value_norm.shape()
        )));
    }
    let value_stride_tokens = value_shape[1];
    if scale_shape[1] != value_stride_tokens {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization value/meta strides differ: value={value_stride_tokens} meta={}",
            scale_shape[1]
        )));
    }
    let end_block = start_block.checked_add(block_count).ok_or_else(|| {
        GpuError::InvalidArg("certified KV value range quantization block range overflow".into())
    })?;
    let value_error_stride_blocks = value_error.shape()[1];
    if end_block > value_error_stride_blocks
        || end_block * block_size > value_stride_tokens
        || end_block * block_size > max_t
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV value range quantization range start_block={start_block} block_count={block_count} exceeds value_stride={value_stride_tokens} error_blocks={value_error_stride_blocks} max_t={max_t}"
        )));
    }

    match value_bf16.backend() {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                let status = dotcache_llama31_certified_kv_quantize_values_bf16_range(
                    ordinal,
                    value_bf16.as_ptr(),
                    value_int4.as_mut_ptr(),
                    value_scale.as_mut_ptr(),
                    value_zero.as_mut_ptr(),
                    value_error.as_mut_ptr(),
                    value_norm.as_mut_ptr(),
                    num_kv_heads as c_int,
                    max_t as c_int,
                    value_stride_tokens as c_int,
                    value_error_stride_blocks as c_int,
                    start_block as c_int,
                    block_count as c_int,
                    head_dim as c_int,
                    block_size as c_int,
                    value_group_size as c_int,
                );
                if status != 0 {
                    return Err(certified_kv_error(
                        Backend::Cuda,
                        format!("certified KV CUDA value range quantization failed: {status}"),
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
            "certified KV value range quantization is currently CUDA-only".into(),
        )),
    }
}

pub fn score_blocks_int8(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
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
    if key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || key_zero.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects U8/F32/F32 key buffers, got {:?}/{:?}/{:?}",
            key_int8.dtype(),
            key_scale.dtype(),
            key_zero.dtype()
        )));
    }
    if block_max.dtype() != ScalarType::F32 || block_sum.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects F32 outputs, got {:?}/{:?}",
            block_max.dtype(),
            block_sum.dtype()
        )));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if (!query_is_2d && !query_is_3d) || key_int8.shape().len() != 3 || key_scale.shape().len() != 3
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring expects query [qh,hd] or [qh,1,hd], key_int8 [kvh,t,hd], key_scale/key_zero [kvh,b,hd], got {:?}/{:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape(),
            key_zero.shape()
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
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let kv_heads = key_int8.shape()[0];
    let stride_tokens = key_int8.shape()[1];
    if key_int8.shape()[2] != head_dim {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring query head_dim={head_dim} does not match key_int8 shape {:?}",
            key_int8.shape()
        )));
    }
    if stride_tokens % block_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring stride_tokens={stride_tokens} must divide block_size={block_size}"
        )));
    }
    let stride_blocks = stride_tokens / block_size;
    if q_heads != kv_heads * gqa_group {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring q_heads={q_heads} must equal kv_heads={kv_heads} * gqa_group={gqa_group}"
        )));
    }
    if block_max.shape().len() != 2
        || block_sum.shape().len() != 2
        || block_max.shape()[0] != q_heads
        || block_sum.shape()[0] != q_heads
        || block_max.shape()[1] != block_sum.shape()[1]
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring outputs expect matching [q_heads, active_blocks], got {:?}/{:?}",
            block_max.shape(),
            block_sum.shape()
        )));
    }
    let num_blocks = block_max.shape()[1];
    if num_blocks == 0 || num_blocks > stride_blocks {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring active blocks={num_blocks} must be in 1..={stride_blocks}"
        )));
    }
    if key_scale.shape().len() != 3
        || key_scale.shape()[0] != kv_heads
        || key_scale.shape()[1] < num_blocks
        || key_scale.shape()[2] != head_dim
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring key_scale expects [kv_heads, >=active_blocks, head_dim] = [{kv_heads}, >={num_blocks}, {head_dim}], got {:?}",
            key_scale.shape()
        )));
    }
    if key_zero.shape() != key_scale.shape() {
        return Err(GpuError::InvalidArg(format!(
            "certified KV INT8 scoring key_zero expects same shape as key_scale {:?}, got {:?}",
            key_scale.shape(),
            key_zero.shape()
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
                        key_zero.as_ptr(),
                        block_max.as_mut_ptr(),
                        block_sum.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        stride_tokens as c_int,
                        stride_blocks as c_int,
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

pub fn score_consistency(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
    promoted_key_bf16: &GpuBuffer,
    promote_index: &GpuBuffer,
    violation_flags: &mut GpuBuffer,
    block_size: usize,
    max_promoted_blocks: usize,
    gqa_group: usize,
    q_scale: f32,
    eps_guard: f32,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV score consistency is currently CUDA-only".into(),
        ));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if !query_is_2d && !query_is_3d {
        return Err(GpuError::InvalidArg(format!(
            "certified KV score consistency expects query [qh,hd] or [qh,1,hd], got {:?}",
            query_bf16.shape()
        )));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || key_zero.dtype() != ScalarType::F32
        || promoted_key_bf16.dtype() != ScalarType::BF16
        || promote_index.dtype() != ScalarType::U32
        || violation_flags.dtype() != ScalarType::U32
    {
        return Err(GpuError::InvalidArg(
            "certified KV score consistency dtype mismatch".into(),
        ));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let kv_heads = key_int8.shape()[0];
    let key_stride_tokens = key_int8.shape()[1];
    let key_scale_stride_blocks = key_scale.shape()[1];
    let num_blocks = promote_index.shape()[1];
    if block_size == 0
        || max_promoted_blocks == 0
        || gqa_group == 0
        || q_heads != kv_heads * gqa_group
        || key_int8.shape().len() != 3
        || key_int8.shape()[2] != head_dim
        || key_scale.shape().len() != 3
        || key_scale.shape()[0] != kv_heads
        || key_scale.shape()[2] != head_dim
        || key_zero.shape() != key_scale.shape()
        || promote_index.shape().len() != 2
        || promote_index.shape()[0] != q_heads
        || promoted_key_bf16.shape().len() != 4
        || promoted_key_bf16.shape()[0] != q_heads
        || promoted_key_bf16.shape()[1] < max_promoted_blocks
        || promoted_key_bf16.shape()[2] != block_size
        || promoted_key_bf16.shape()[3] != head_dim
        || violation_flags.elem_count() < q_heads
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV score consistency shape mismatch: query={:?} key={:?} scale={:?} promoted={:?} index={:?} flags={:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape(),
            promoted_key_bf16.shape(),
            promote_index.shape(),
            violation_flags.shape()
        )));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_score_consistency(
            ordinal,
            query_bf16.as_ptr(),
            key_int8.as_ptr(),
            key_scale.as_ptr(),
            key_zero.as_ptr(),
            promoted_key_bf16.as_ptr(),
            promote_index.as_ptr(),
            violation_flags.as_mut_ptr(),
            q_heads as c_int,
            kv_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            key_stride_tokens as c_int,
            key_scale_stride_blocks as c_int,
            max_promoted_blocks as c_int,
            head_dim as c_int,
            gqa_group as c_int,
            q_scale,
            eps_guard,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA score consistency failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            query_bf16,
            key_int8,
            key_scale,
            key_zero,
            promoted_key_bf16,
            promote_index,
            violation_flags,
            block_size,
            max_promoted_blocks,
            gqa_group,
            q_scale,
            eps_guard,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gather_promoted_bf16_from_tier2(
    ordinal: usize,
    tier2_key_bf16_device_ptr: *const c_void,
    tier2_value_bf16_device_ptr: *const c_void,
    promote_index: &GpuBuffer,
    value_promote_index: &GpuBuffer,
    promoted_key_bf16: &mut GpuBuffer,
    promoted_value_bf16: &mut GpuBuffer,
    block_size: usize,
    cap_tokens: usize,
    max_promoted_blocks: usize,
    max_promoted_value_blocks: usize,
    gqa_group: usize,
) -> Result<(), GpuError> {
    if promote_index.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 gather is currently CUDA-only".into(),
        ));
    }
    if tier2_key_bf16_device_ptr.is_null() || tier2_value_bf16_device_ptr.is_null() {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 gather received null Tier-2 device pointer".into(),
        ));
    }
    if promote_index.dtype() != ScalarType::U32
        || value_promote_index.dtype() != ScalarType::U32
        || promoted_key_bf16.dtype() != ScalarType::BF16
        || promoted_value_bf16.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 gather dtype mismatch".into(),
        ));
    }
    let promote_shape = promote_index.shape();
    let value_promote_shape = value_promote_index.shape();
    let promoted_key_shape = promoted_key_bf16.shape();
    let promoted_value_shape = promoted_value_bf16.shape();
    if promote_shape.len() != 2
        || value_promote_shape.len() != 2
        || promoted_key_shape.len() != 4
        || promoted_value_shape.len() != 4
        || block_size == 0
        || cap_tokens == 0
        || max_promoted_blocks == 0
        || max_promoted_value_blocks == 0
        || gqa_group == 0
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV promoted BF16 gather invalid shapes: index={:?} value_index={:?} key={:?} value={:?}",
            promote_shape, value_promote_shape, promoted_key_shape, promoted_value_shape
        )));
    }
    let q_heads = promote_shape[0];
    let num_blocks = promote_shape[1];
    let kv_heads = value_promote_shape[0];
    let promoted_key_heads = promoted_key_shape[0];
    let head_dim = promoted_key_shape[3];
    let compact_all_key_layout =
        promoted_key_heads == kv_heads && max_promoted_blocks == num_blocks;
    if value_promote_shape[1] != num_blocks
        || q_heads != kv_heads * gqa_group
        || (promoted_key_heads != q_heads && !compact_all_key_layout)
        || promoted_key_shape[1] < max_promoted_blocks
        || promoted_key_shape[2] != block_size
        || promoted_value_shape[0] != kv_heads
        || promoted_value_shape[1] < max_promoted_value_blocks
        || promoted_value_shape[2] != block_size
        || promoted_value_shape[3] != head_dim
        || cap_tokens < num_blocks * block_size
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV promoted BF16 gather shape mismatch: index={:?} value_index={:?} key={:?} value={:?} cap={cap_tokens} block={block_size} max_key={max_promoted_blocks} max_value={max_promoted_value_blocks} gqa={gqa_group}",
            promote_shape, value_promote_shape, promoted_key_shape, promoted_value_shape
        )));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_gather_promoted_bf16(
            ordinal,
            tier2_key_bf16_device_ptr,
            tier2_value_bf16_device_ptr,
            promote_index.as_ptr(),
            value_promote_index.as_ptr(),
            promoted_key_bf16.as_mut_ptr(),
            promoted_value_bf16.as_mut_ptr(),
            q_heads as c_int,
            kv_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            cap_tokens as c_int,
            promoted_key_heads as c_int,
            max_promoted_blocks as c_int,
            max_promoted_value_blocks as c_int,
            head_dim as c_int,
            gqa_group as c_int,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA promoted BF16 gather failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            tier2_key_bf16_device_ptr,
            tier2_value_bf16_device_ptr,
            promote_index,
            value_promote_index,
            promoted_key_bf16,
            promoted_value_bf16,
            block_size,
            cap_tokens,
            max_promoted_blocks,
            max_promoted_value_blocks,
            gqa_group,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gather_promoted_values_bf16_from_tier2(
    ordinal: usize,
    tier2_value_bf16_device_ptr: *const c_void,
    value_promote_index: &GpuBuffer,
    promoted_value_bf16: &mut GpuBuffer,
    block_size: usize,
    cap_tokens: usize,
    max_promoted_value_blocks: usize,
) -> Result<(), GpuError> {
    if value_promote_index.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 value gather is currently CUDA-only".into(),
        ));
    }
    if tier2_value_bf16_device_ptr.is_null() {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 value gather received null Tier-2 device pointer".into(),
        ));
    }
    if value_promote_index.dtype() != ScalarType::U32
        || promoted_value_bf16.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(
            "certified KV promoted BF16 value gather dtype mismatch".into(),
        ));
    }
    let value_promote_shape = value_promote_index.shape();
    let promoted_value_shape = promoted_value_bf16.shape();
    if value_promote_shape.len() != 2
        || promoted_value_shape.len() != 4
        || block_size == 0
        || cap_tokens == 0
        || max_promoted_value_blocks == 0
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV promoted BF16 value gather invalid shapes: value_index={:?} value={:?}",
            value_promote_shape, promoted_value_shape
        )));
    }
    let kv_heads = value_promote_shape[0];
    let num_blocks = value_promote_shape[1];
    let head_dim = promoted_value_shape[3];
    if promoted_value_shape[0] != kv_heads
        || promoted_value_shape[1] < max_promoted_value_blocks
        || promoted_value_shape[2] != block_size
        || cap_tokens < num_blocks * block_size
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV promoted BF16 value gather shape mismatch: value_index={:?} value={:?} cap={cap_tokens} block={block_size} max_value={max_promoted_value_blocks}",
            value_promote_shape, promoted_value_shape
        )));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_gather_promoted_values_bf16(
            ordinal,
            tier2_value_bf16_device_ptr,
            value_promote_index.as_ptr(),
            promoted_value_bf16.as_mut_ptr(),
            kv_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            cap_tokens as c_int,
            max_promoted_value_blocks as c_int,
            head_dim as c_int,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA promoted BF16 value gather failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            tier2_value_bf16_device_ptr,
            value_promote_index,
            promoted_value_bf16,
            block_size,
            cap_tokens,
            max_promoted_value_blocks,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

pub fn selected_fp16_log_masses(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    promoted_key_bf16: &GpuBuffer,
    promote_index: &GpuBuffer,
    out_log_masses: &mut GpuBuffer,
    block_size: usize,
    max_promoted_blocks: usize,
    q_scale: f32,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV selected FP16 log-masses is currently CUDA-only".into(),
        ));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if !query_is_2d && !query_is_3d {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected FP16 log-masses expects query [qh,hd] or [qh,1,hd], got {:?}",
            query_shape
        )));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || promoted_key_bf16.dtype() != ScalarType::BF16
        || promote_index.dtype() != ScalarType::U32
        || out_log_masses.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(
            "certified KV selected FP16 log-masses dtype mismatch".into(),
        ));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let num_blocks = promote_index.shape().get(1).copied().unwrap_or(0);
    if block_size == 0
        || max_promoted_blocks == 0
        || promote_index.shape() != [q_heads, num_blocks]
        || promoted_key_bf16.shape() != [q_heads, max_promoted_blocks, block_size, head_dim]
        || out_log_masses.shape() != [q_heads, max_promoted_blocks]
        || max_promoted_blocks > num_blocks
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected FP16 log-masses shape mismatch: query={:?} promoted={:?} index={:?} out={:?} block={block_size} max={max_promoted_blocks}",
            query_bf16.shape(),
            promoted_key_bf16.shape(),
            promote_index.shape(),
            out_log_masses.shape()
        )));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_selected_fp16_log_masses(
            ordinal,
            query_bf16.as_ptr(),
            promoted_key_bf16.as_ptr(),
            promote_index.as_ptr(),
            out_log_masses.as_mut_ptr(),
            q_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            max_promoted_blocks as c_int,
            head_dim as c_int,
            q_scale,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA selected FP16 log-masses failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            query_bf16,
            promoted_key_bf16,
            promote_index,
            out_log_masses,
            block_size,
            max_promoted_blocks,
            q_scale,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

pub fn attend_int8_int4(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
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
                        key_zero.as_ptr(),
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
    key_zero: &GpuBuffer,
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
            key_zero,
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
                        key_zero.as_ptr(),
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

pub fn attend_int8_int4_with_bf16_tail_strided(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    tail_value_bf16: Option<&GpuBuffer>,
    total_tokens: usize,
    block_size: usize,
    value_group_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV strided hybrid attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || value_int4.dtype() != ScalarType::U8
        || value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || score_scratch.dtype() != ScalarType::F32
        || (output_f32.dtype() != ScalarType::F32 && output_f32.dtype() != ScalarType::BF16)
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid dtypes must be BF16/U8/F32/U8/F16/F16/F32/F32-or-BF16, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
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
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if (!query_is_2d && !query_is_3d) || key_int8.shape().len() != 3 || key_scale.shape().len() != 3
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid expects query [qh,hd] or [qh,1,hd], key_int8 [kvh,stride,hd], key_scale [kvh,b,hd], got {:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            key_scale.shape()
        )));
    }
    if block_size == 0 || block_size > 256 || value_group_size == 0 || gqa_group == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid invalid block_size={block_size} value_group_size={value_group_size} gqa_group={gqa_group}"
        )));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let kv_heads = key_int8.shape()[0];
    let active_aligned_tokens = aligned_tokens(total_tokens, block_size);
    if active_aligned_tokens == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV strided hybrid needs at least one complete block".into(),
        ));
    }
    if head_dim % 2 != 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid head_dim={head_dim} must be even and divisible by value_group_size={value_group_size}"
        )));
    }
    let key_stride_tokens = key_int8.shape()[1];
    if key_int8.shape()[2] != head_dim || key_stride_tokens < active_aligned_tokens {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid key shape {:?} incompatible with active_aligned={active_aligned_tokens} head_dim={head_dim}",
            key_int8.shape()
        )));
    }
    let num_blocks = active_aligned_tokens / block_size;
    let key_scale_stride_blocks =
        if key_scale.shape()[0] == kv_heads && key_scale.shape()[2] == head_dim {
            key_scale.shape()[1]
        } else {
            0
        };
    let value_groups = head_dim / value_group_size;
    let value_stride_tokens = if value_int4.shape().len() == 3
        && value_int4.shape()[0] == kv_heads
        && value_int4.shape()[2] == head_dim / 2
    {
        value_int4.shape()[1]
    } else {
        0
    };
    let output_shape = output_f32.shape();
    let output_is_2d = output_shape == [q_heads, head_dim];
    let output_is_3d = output_shape == [q_heads, 1, head_dim];
    if value_scale.shape() != [kv_heads, value_stride_tokens, value_groups]
        || value_zero.shape() != [kv_heads, value_stride_tokens, value_groups]
        || key_scale_stride_blocks < num_blocks
        || value_stride_tokens < active_aligned_tokens
        || score_scratch.shape().len() != 2
        || score_scratch.shape()[0] != q_heads
        || score_scratch.shape()[1] < total_tokens
        || (!output_is_2d && !output_is_3d)
        || q_heads != kv_heads * gqa_group
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided hybrid shape mismatch key_scale={:?} value_int4={:?} value_scale={:?} value_zero={:?} score_scratch={:?} output={:?}",
            key_scale.shape(),
            value_int4.shape(),
            value_scale.shape(),
            value_zero.shape(),
            score_scratch.shape(),
            output_f32.shape()
        )));
    }
    let tail_len = total_tokens - active_aligned_tokens;
    let tail_layout = |tail: &GpuBuffer, name: &str| -> Result<(usize, usize), GpuError> {
        let shape = tail.shape();
        let compact_3d = shape == [kv_heads, tail_len, head_dim];
        let strided_3d = shape.len() == 3 && shape[0] == kv_heads && shape[2] == head_dim;
        let strided_4d =
            shape.len() == 4 && shape[0] == 1 && shape[1] == kv_heads && shape[3] == head_dim;
        if tail.dtype() != ScalarType::BF16 || (!compact_3d && !strided_3d && !strided_4d) {
            return Err(GpuError::InvalidArg(format!(
                "certified KV strided hybrid {name} expects BF16 compact or strided tail, got {:?} {:?}",
                tail.dtype(),
                tail.shape()
            )));
        }
        if compact_3d {
            Ok((0, tail_len))
        } else {
            let stride = if shape.len() == 3 { shape[1] } else { shape[2] };
            if stride < total_tokens {
                return Err(GpuError::InvalidArg(format!(
                    "certified KV strided hybrid {name} stride={stride} must cover total_tokens={total_tokens}"
                )));
            }
            Ok((active_aligned_tokens, stride))
        }
    };
    let (tail_key_start, tail_key_stride, tail_value_start, tail_value_stride) = if tail_len == 0 {
        (0, 0, 0, 0)
    } else {
        let tail_key = tail_key_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV strided hybrid needs tail key for tail_len={tail_len}"
            ))
        })?;
        let tail_value = tail_value_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV strided hybrid needs tail value for tail_len={tail_len}"
            ))
        })?;
        let (key_start, key_stride) = tail_layout(tail_key, "tail key")?;
        let (value_start, value_stride) = tail_layout(tail_value, "tail value")?;
        (key_start, key_stride, value_start, value_stride)
    };

    match query_bf16.backend() {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                let status = if output_f32.dtype() == ScalarType::BF16 {
                    dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided_out_bf16(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        key_zero.as_ptr(),
                        value_int4.as_ptr(),
                        value_scale.as_ptr(),
                        value_zero.as_ptr(),
                        tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
                        tail_value_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
                        score_scratch.as_mut_ptr(),
                        output_f32.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        tail_len as c_int,
                        key_stride_tokens as c_int,
                        key_scale_stride_blocks as c_int,
                        value_stride_tokens as c_int,
                        tail_key_start as c_int,
                        tail_key_stride as c_int,
                        tail_value_start as c_int,
                        tail_value_stride as c_int,
                        score_scratch.shape()[1] as c_int,
                        head_dim as c_int,
                        value_group_size as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                } else {
                    dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided(
                        ordinal,
                        query_bf16.as_ptr(),
                        key_int8.as_ptr(),
                        key_scale.as_ptr(),
                        key_zero.as_ptr(),
                        value_int4.as_ptr(),
                        value_scale.as_ptr(),
                        value_zero.as_ptr(),
                        tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
                        tail_value_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
                        score_scratch.as_mut_ptr(),
                        output_f32.as_mut_ptr(),
                        q_heads as c_int,
                        kv_heads as c_int,
                        num_blocks as c_int,
                        block_size as c_int,
                        tail_len as c_int,
                        key_stride_tokens as c_int,
                        key_scale_stride_blocks as c_int,
                        value_stride_tokens as c_int,
                        tail_key_start as c_int,
                        tail_key_stride as c_int,
                        tail_value_start as c_int,
                        tail_value_stride as c_int,
                        score_scratch.shape()[1] as c_int,
                        head_dim as c_int,
                        value_group_size as c_int,
                        gqa_group as c_int,
                        q_scale,
                    )
                };
                if status != 0 {
                    return Err(certified_kv_error(
                        Backend::Cuda,
                        format!("certified KV CUDA strided hybrid attention failed: {status}"),
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
            "certified KV strided hybrid attention is currently CUDA-only".into(),
        )),
    }
}

pub fn attend_mixed_key_int4_with_bf16_tail_strided(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
    promoted_key_bf16: &GpuBuffer,
    promote_index: &GpuBuffer,
    promoted_value_bf16: &GpuBuffer,
    value_promote_index: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    tail_value_bf16: Option<&GpuBuffer>,
    total_tokens: usize,
    block_size: usize,
    value_group_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    output_bf16: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV mixed-key attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || key_int8.dtype() != ScalarType::U8
        || key_scale.dtype() != ScalarType::F32
        || key_zero.dtype() != ScalarType::F32
        || promoted_key_bf16.dtype() != ScalarType::BF16
        || promote_index.dtype() != ScalarType::U32
        || promoted_value_bf16.dtype() != ScalarType::BF16
        || value_promote_index.dtype() != ScalarType::U32
        || value_int4.dtype() != ScalarType::U8
        || value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || score_scratch.dtype() != ScalarType::F32
        || output_bf16.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV mixed-key dtypes must be BF16/U8/F32/F32/BF16/U32/BF16/U32/U8/F16/F16/F32/BF16, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.dtype(),
            key_int8.dtype(),
            key_scale.dtype(),
            key_zero.dtype(),
            promoted_key_bf16.dtype(),
            promote_index.dtype(),
            promoted_value_bf16.dtype(),
            value_promote_index.dtype(),
            value_int4.dtype(),
            value_scale.dtype(),
            value_zero.dtype(),
            score_scratch.dtype(),
            output_bf16.dtype()
        )));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if (!query_is_2d && !query_is_3d)
        || key_int8.shape().len() != 3
        || key_scale.shape().len() != 3
        || key_zero.shape().len() != 3
        || value_int4.shape().len() != 3
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV mixed-key expects query [qh,hd] or [qh,1,hd], key/value packed 3D, got {:?}/{:?}/{:?}",
            query_bf16.shape(),
            key_int8.shape(),
            value_int4.shape()
        )));
    }
    if block_size == 0 || block_size > 256 || value_group_size == 0 || gqa_group == 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV mixed-key invalid block_size={block_size} value_group_size={value_group_size} gqa_group={gqa_group}"
        )));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let kv_heads = key_int8.shape()[0];
    let active_aligned_tokens = aligned_tokens(total_tokens, block_size);
    if active_aligned_tokens == 0 {
        return Err(GpuError::InvalidArg(
            "certified KV mixed-key needs at least one complete block".into(),
        ));
    }
    if head_dim % 2 != 0 || head_dim % value_group_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV mixed-key head_dim={head_dim} must be even and divisible by value_group_size={value_group_size}"
        )));
    }
    let key_stride_tokens = key_int8.shape()[1];
    let num_blocks = active_aligned_tokens / block_size;
    let key_scale_stride_blocks = if key_scale.shape()[0] == kv_heads
        && key_scale.shape()[2] == head_dim
        && key_zero.shape() == key_scale.shape()
    {
        key_scale.shape()[1]
    } else {
        0
    };
    let promoted_key_shape = promoted_key_bf16.shape();
    let promoted_key_heads = promoted_key_shape.first().copied().unwrap_or(0);
    let max_promoted_blocks = if promoted_key_shape.len() == 4
        && (promoted_key_shape[0] == q_heads || promoted_key_shape[0] == kv_heads)
        && promoted_key_shape[2] == block_size
        && promoted_key_shape[3] == head_dim
    {
        promoted_key_shape[1]
    } else {
        0
    };
    let promoted_value_shape = promoted_value_bf16.shape();
    let max_promoted_value_blocks = if promoted_value_shape.len() == 4
        && promoted_value_shape[0] == kv_heads
        && promoted_value_shape[2] == block_size
        && promoted_value_shape[3] == head_dim
    {
        promoted_value_shape[1]
    } else {
        0
    };
    let value_groups = head_dim / value_group_size;
    let value_stride_tokens =
        if value_int4.shape()[0] == kv_heads && value_int4.shape()[2] == head_dim / 2 {
            value_int4.shape()[1]
        } else {
            0
        };
    let output_shape = output_bf16.shape();
    let output_is_2d = output_shape == [q_heads, head_dim];
    let output_is_3d = output_shape == [q_heads, 1, head_dim];
    if key_int8.shape()[2] != head_dim
        || key_stride_tokens < active_aligned_tokens
        || key_scale_stride_blocks < num_blocks
        || max_promoted_blocks == 0
        || (promoted_key_heads == kv_heads && max_promoted_blocks != num_blocks)
        || promote_index.shape() != [q_heads, num_blocks]
        || max_promoted_value_blocks == 0
        || value_promote_index.shape() != [kv_heads, num_blocks]
        || value_scale.shape() != [kv_heads, value_stride_tokens, value_groups]
        || value_zero.shape() != [kv_heads, value_stride_tokens, value_groups]
        || value_stride_tokens < active_aligned_tokens
        || score_scratch.shape().len() != 2
        || score_scratch.shape()[0] != q_heads
        || score_scratch.shape()[1] < total_tokens
        || (!output_is_2d && !output_is_3d)
        || q_heads != kv_heads * gqa_group
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV mixed-key shape mismatch key_int8={:?} key_scale={:?} full_key={:?} promote_mask={:?} promoted_value={:?} value_promote_index={:?} value_int4={:?} value_scale={:?} score_scratch={:?} output={:?}",
            key_int8.shape(),
            key_scale.shape(),
            promoted_key_bf16.shape(),
            promote_index.shape(),
            promoted_value_bf16.shape(),
            value_promote_index.shape(),
            value_int4.shape(),
            value_scale.shape(),
            score_scratch.shape(),
            output_bf16.shape()
        )));
    }
    let tail_len = total_tokens - active_aligned_tokens;
    let tail_layout = |tail: &GpuBuffer, name: &str| -> Result<(usize, usize), GpuError> {
        let shape = tail.shape();
        let compact_3d = shape == [kv_heads, tail_len, head_dim];
        let strided_3d = shape.len() == 3 && shape[0] == kv_heads && shape[2] == head_dim;
        let strided_4d =
            shape.len() == 4 && shape[0] == 1 && shape[1] == kv_heads && shape[3] == head_dim;
        if tail.dtype() != ScalarType::BF16 || (!compact_3d && !strided_3d && !strided_4d) {
            return Err(GpuError::InvalidArg(format!(
                "certified KV mixed-key {name} expects BF16 compact or strided tail, got {:?} {:?}",
                tail.dtype(),
                tail.shape()
            )));
        }
        if compact_3d {
            Ok((0, tail_len))
        } else {
            let stride = if shape.len() == 3 { shape[1] } else { shape[2] };
            if stride < total_tokens {
                return Err(GpuError::InvalidArg(format!(
                    "certified KV mixed-key {name} stride={stride} must cover total_tokens={total_tokens}"
                )));
            }
            Ok((active_aligned_tokens, stride))
        }
    };
    let (tail_key_start, tail_key_stride, tail_value_start, tail_value_stride) = if tail_len == 0 {
        (0, 0, 0, 0)
    } else {
        let tail_key = tail_key_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV mixed-key needs tail key for tail_len={tail_len}"
            ))
        })?;
        let tail_value = tail_value_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV mixed-key needs tail value for tail_len={tail_len}"
            ))
        })?;
        let (key_start, key_stride) = tail_layout(tail_key, "tail key")?;
        let (value_start, value_stride) = tail_layout(tail_value, "tail value")?;
        (key_start, key_stride, value_start, value_stride)
    };

    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_attend_mixed_key_int4_bf16_tail_strided_out_bf16(
            ordinal,
            query_bf16.as_ptr(),
            key_int8.as_ptr(),
            key_scale.as_ptr(),
            key_zero.as_ptr(),
            promoted_key_bf16.as_ptr(),
            promote_index.as_ptr(),
            promoted_value_bf16.as_ptr(),
            value_promote_index.as_ptr(),
            value_int4.as_ptr(),
            value_scale.as_ptr(),
            value_zero.as_ptr(),
            tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
            tail_value_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
            score_scratch.as_mut_ptr(),
            output_bf16.as_mut_ptr(),
            q_heads as c_int,
            kv_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            tail_len as c_int,
            key_stride_tokens as c_int,
            key_scale_stride_blocks as c_int,
            promoted_key_heads as c_int,
            max_promoted_blocks as c_int,
            max_promoted_value_blocks as c_int,
            value_stride_tokens as c_int,
            tail_key_start as c_int,
            tail_key_stride as c_int,
            tail_value_start as c_int,
            tail_value_stride as c_int,
            score_scratch.shape()[1] as c_int,
            head_dim as c_int,
            value_group_size as c_int,
            gqa_group as c_int,
            q_scale,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA mixed-key attention failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn attend_all_promoted_int4_with_bf16_tail(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    promoted_key_bf16: &GpuBuffer,
    promoted_value_bf16: &GpuBuffer,
    value_promote_index: &GpuBuffer,
    value_int4: &GpuBuffer,
    value_scale: &GpuBuffer,
    value_zero: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    tail_value_bf16: Option<&GpuBuffer>,
    total_tokens: usize,
    block_size: usize,
    value_group_size: usize,
    gqa_group: usize,
    q_scale: f32,
    score_scratch: &mut GpuBuffer,
    softmax_stats: &mut GpuBuffer,
    output_bf16: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV all-promoted attention is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || promoted_key_bf16.dtype() != ScalarType::BF16
        || promoted_value_bf16.dtype() != ScalarType::BF16
        || value_promote_index.dtype() != ScalarType::U32
        || value_int4.dtype() != ScalarType::U8
        || value_scale.dtype() != ScalarType::F16
        || value_zero.dtype() != ScalarType::F16
        || score_scratch.dtype() != ScalarType::F32
        || softmax_stats.dtype() != ScalarType::F32
        || output_bf16.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(
            "certified KV all-promoted attention dtype mismatch".into(),
        ));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if !query_is_2d && !query_is_3d {
        return Err(GpuError::InvalidArg(format!(
            "certified KV all-promoted query expects [qh,hd] or [qh,1,hd], got {:?}",
            query_shape
        )));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let key_shape = promoted_key_bf16.shape();
    if key_shape.len() != 4 || key_shape[2] != block_size || key_shape[3] != head_dim {
        return Err(GpuError::InvalidArg(format!(
            "certified KV all-promoted key shape mismatch: {:?}",
            key_shape
        )));
    }
    let kv_heads = key_shape[0];
    let num_blocks = key_shape[1];
    let active_aligned_tokens = num_blocks * block_size;
    if total_tokens < active_aligned_tokens {
        return Err(GpuError::InvalidArg(format!(
            "certified KV all-promoted total_tokens={total_tokens} < aligned={active_aligned_tokens}"
        )));
    }
    let tail_len = total_tokens - active_aligned_tokens;
    let value_groups = head_dim / value_group_size;
    let value_stride_tokens = if value_int4.shape().len() == 3 && value_int4.shape()[0] == kv_heads
    {
        value_int4.shape()[1]
    } else {
        0
    };
    let promoted_value_shape = promoted_value_bf16.shape();
    let max_promoted_value_blocks = if promoted_value_shape.len() == 4
        && promoted_value_shape[0] == kv_heads
        && promoted_value_shape[2] == block_size
        && promoted_value_shape[3] == head_dim
    {
        promoted_value_shape[1]
    } else {
        0
    };
    let output_shape = output_bf16.shape();
    let output_is_2d = output_shape == [q_heads, head_dim];
    let output_is_3d = output_shape == [q_heads, 1, head_dim];
    if block_size == 0
        || block_size > 256
        || head_dim == 0
        || head_dim > 128
        || value_group_size == 0
        || head_dim % value_group_size != 0
        || gqa_group == 0
        || q_heads != kv_heads * gqa_group
        || max_promoted_value_blocks == 0
        || value_promote_index.shape() != [kv_heads, num_blocks]
        || value_int4.shape() != [kv_heads, value_stride_tokens, head_dim / 2]
        || value_scale.shape() != [kv_heads, value_stride_tokens, value_groups]
        || value_zero.shape() != [kv_heads, value_stride_tokens, value_groups]
        || value_stride_tokens < active_aligned_tokens
        || score_scratch.shape().len() != 2
        || score_scratch.shape()[0] != q_heads
        || score_scratch.shape()[1] < total_tokens
        || softmax_stats.shape() != [q_heads, 2]
        || (!output_is_2d && !output_is_3d)
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV all-promoted shape mismatch query={:?} key={:?} value_index={:?} value={:?} value_scale={:?} score={:?} stats={:?} output={:?}",
            query_bf16.shape(),
            promoted_key_bf16.shape(),
            value_promote_index.shape(),
            value_int4.shape(),
            value_scale.shape(),
            score_scratch.shape(),
            softmax_stats.shape(),
            output_bf16.shape()
        )));
    }
    let tail_layout = |tail: &GpuBuffer, name: &str| -> Result<(usize, usize), GpuError> {
        let shape = tail.shape();
        let compact_3d = shape == [kv_heads, tail_len, head_dim];
        let strided_3d = shape.len() == 3 && shape[0] == kv_heads && shape[2] == head_dim;
        let strided_4d =
            shape.len() == 4 && shape[0] == 1 && shape[1] == kv_heads && shape[3] == head_dim;
        if tail.dtype() != ScalarType::BF16 || (!compact_3d && !strided_3d && !strided_4d) {
            return Err(GpuError::InvalidArg(format!(
                "certified KV all-promoted {name} expects BF16 compact or strided tail, got {:?} {:?}",
                tail.dtype(),
                tail.shape()
            )));
        }
        if compact_3d {
            Ok((0, tail_len))
        } else {
            let stride = if shape.len() == 3 { shape[1] } else { shape[2] };
            if stride < total_tokens {
                return Err(GpuError::InvalidArg(format!(
                    "certified KV all-promoted {name} stride={stride} must cover total_tokens={total_tokens}"
                )));
            }
            Ok((active_aligned_tokens, stride))
        }
    };
    let (tail_key_start, tail_key_stride, tail_value_start, tail_value_stride) = if tail_len == 0 {
        (0, 0, 0, 0)
    } else {
        let tail_key = tail_key_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV all-promoted needs tail key for tail_len={tail_len}"
            ))
        })?;
        let tail_value = tail_value_bf16.ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "certified KV all-promoted needs tail value for tail_len={tail_len}"
            ))
        })?;
        let (key_start, key_stride) = tail_layout(tail_key, "tail key")?;
        let (value_start, value_stride) = tail_layout(tail_value, "tail value")?;
        (key_start, key_stride, value_start, value_stride)
    };

    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_attend_all_promoted_int4_bf16_tail_out_bf16(
            ordinal,
            query_bf16.as_ptr(),
            promoted_key_bf16.as_ptr(),
            promoted_value_bf16.as_ptr(),
            value_promote_index.as_ptr(),
            value_int4.as_ptr(),
            value_scale.as_ptr(),
            value_zero.as_ptr(),
            tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
            tail_value_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr),
            score_scratch.as_mut_ptr(),
            softmax_stats.as_mut_ptr(),
            output_bf16.as_mut_ptr(),
            q_heads as c_int,
            kv_heads as c_int,
            num_blocks as c_int,
            block_size as c_int,
            tail_len as c_int,
            max_promoted_value_blocks as c_int,
            value_stride_tokens as c_int,
            tail_key_start as c_int,
            tail_key_stride as c_int,
            tail_value_start as c_int,
            tail_value_stride as c_int,
            score_scratch.shape()[1] as c_int,
            head_dim as c_int,
            value_group_size as c_int,
            gqa_group as c_int,
            q_scale,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA all-promoted attention failed: {status}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            query_bf16,
            promoted_key_bf16,
            promoted_value_bf16,
            value_promote_index,
            value_int4,
            value_scale,
            value_zero,
            tail_key_bf16,
            tail_value_bf16,
            total_tokens,
            block_size,
            value_group_size,
            gqa_group,
            q_scale,
            score_scratch,
            softmax_stats,
            output_bf16,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

pub fn attend_int8_bf16_values(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
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
        key_zero,
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

pub fn dense_selected_heads_out_bf16(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    fallback_heads: &GpuBuffer,
    fallback_kv_slots: &GpuBuffer,
    fallback_kv_heads: &GpuBuffer,
    fallback_key_bf16: &GpuBuffer,
    fallback_value_bf16: &GpuBuffer,
    tail_key_bf16: Option<&GpuBuffer>,
    tail_value_bf16: Option<&GpuBuffer>,
    total_tokens: usize,
    score_scratch: &mut GpuBuffer,
    output_bf16: &mut GpuBuffer,
    q_scale: f32,
) -> Result<(), GpuError> {
    if query_bf16.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "certified KV selected-head dense fallback is currently CUDA-only".into(),
        ));
    }
    if query_bf16.dtype() != ScalarType::BF16
        || fallback_heads.dtype() != ScalarType::U32
        || fallback_kv_slots.dtype() != ScalarType::U32
        || fallback_kv_heads.dtype() != ScalarType::U32
        || fallback_key_bf16.dtype() != ScalarType::BF16
        || fallback_value_bf16.dtype() != ScalarType::BF16
        || score_scratch.dtype() != ScalarType::F32
        || output_bf16.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected-head fallback dtypes must be BF16/U32/U32/U32/BF16/BF16/F32/BF16, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            query_bf16.dtype(),
            fallback_heads.dtype(),
            fallback_kv_slots.dtype(),
            fallback_kv_heads.dtype(),
            fallback_key_bf16.dtype(),
            fallback_value_bf16.dtype(),
            score_scratch.dtype(),
            output_bf16.dtype()
        )));
    }
    let query_shape = query_bf16.shape();
    let query_is_2d = query_shape.len() == 2;
    let query_is_3d = query_shape.len() == 3 && query_shape[1] == 1;
    if !query_is_2d && !query_is_3d {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected-head fallback expects query [qh,hd] or [qh,1,hd], got {:?}",
            query_shape
        )));
    }
    let q_heads = query_shape[0];
    let head_dim = if query_is_2d {
        query_shape[1]
    } else {
        query_shape[2]
    };
    let fallback_shape = fallback_key_bf16.shape();
    let prefix_tokens = fallback_shape.get(1).copied().unwrap_or(0);
    let tail_len = total_tokens.saturating_sub(prefix_tokens);
    if fallback_shape.len() != 3
        || fallback_value_bf16.shape() != fallback_shape
        || fallback_heads.shape().len() != 1
        || fallback_kv_slots.shape() != fallback_heads.shape()
        || fallback_kv_heads.shape().len() != 1
        || fallback_kv_heads.shape()[0] != fallback_shape[0]
        || fallback_shape[2] != head_dim
        || score_scratch.shape().len() != 2
        || score_scratch.shape()[0] != fallback_heads.shape()[0]
        || score_scratch.shape()[1] < total_tokens
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected-head fallback shape mismatch query={:?} heads={:?} kv_slots={:?} kv_heads={:?} key={:?} value={:?} score={:?} total_tokens={total_tokens}",
            query_shape,
            fallback_heads.shape(),
            fallback_kv_slots.shape(),
            fallback_kv_heads.shape(),
            fallback_key_bf16.shape(),
            fallback_value_bf16.shape(),
            score_scratch.shape()
        )));
    }
    if tail_len > 0 {
        let tail_k = tail_key_bf16.ok_or_else(|| {
            GpuError::InvalidArg(
                "certified KV selected-head fallback requires tail key when tail_len > 0".into(),
            )
        })?;
        let tail_v = tail_value_bf16.ok_or_else(|| {
            GpuError::InvalidArg(
                "certified KV selected-head fallback requires tail value when tail_len > 0".into(),
            )
        })?;
        if tail_k.dtype() != ScalarType::BF16
            || tail_v.dtype() != ScalarType::BF16
            || tail_k.shape().len() != 3
            || tail_v.shape().len() != 3
            || tail_k.shape()[2] != head_dim
            || tail_v.shape()[2] != head_dim
            || tail_k.shape()[1] < tail_len
            || tail_v.shape()[1] < tail_len
        {
            return Err(GpuError::InvalidArg(format!(
                "certified KV selected-head fallback tail shape mismatch tail_k={:?} tail_v={:?} tail_len={tail_len} head_dim={head_dim}",
                tail_k.shape(),
                tail_v.shape()
            )));
        }
    }
    let output_shape = output_bf16.shape();
    let output_is_2d = output_shape == [q_heads, head_dim];
    let output_is_3d = output_shape == [q_heads, 1, head_dim];
    if !output_is_2d && !output_is_3d {
        return Err(GpuError::InvalidArg(format!(
            "certified KV selected-head fallback output shape mismatch {:?}",
            output_shape
        )));
    }
    let fallback_count = fallback_heads.shape()[0];
    let fallback_kv_count = fallback_shape[0];
    if fallback_count == 0 || fallback_kv_count == 0 || total_tokens == 0 {
        return Ok(());
    }
    let tail_key_ptr = tail_key_bf16
        .map(|buf| buf.as_ptr())
        .unwrap_or(std::ptr::null());
    let tail_value_ptr = tail_value_bf16
        .map(|buf| buf.as_ptr())
        .unwrap_or(std::ptr::null());
    let tail_key_stride = tail_key_bf16.map(|buf| buf.shape()[1]).unwrap_or(0);
    let tail_value_stride = tail_value_bf16.map(|buf| buf.shape()[1]).unwrap_or(0);

    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_llama31_certified_kv_dense_selected_heads_out_bf16(
            ordinal,
            query_bf16.as_ptr(),
            fallback_heads.as_ptr(),
            fallback_kv_slots.as_ptr(),
            fallback_kv_heads.as_ptr(),
            fallback_key_bf16.as_ptr(),
            fallback_value_bf16.as_ptr(),
            tail_key_ptr,
            tail_value_ptr,
            score_scratch.as_mut_ptr(),
            output_bf16.as_mut_ptr(),
            q_heads as c_int,
            fallback_count as c_int,
            fallback_kv_count as c_int,
            prefix_tokens as c_int,
            tail_len as c_int,
            0,
            tail_key_stride as c_int,
            0,
            tail_value_stride as c_int,
            score_scratch.shape()[1] as c_int,
            head_dim as c_int,
            q_scale,
        );
        if status != 0 {
            return Err(certified_kv_error(
                Backend::Cuda,
                format!("certified KV CUDA selected-head fallback failed: {status}"),
            ));
        }
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            query_bf16,
            fallback_heads,
            fallback_kv_slots,
            fallback_kv_heads,
            fallback_key_bf16,
            fallback_value_bf16,
            tail_key_bf16,
            tail_value_bf16,
            total_tokens,
            score_scratch,
            output_bf16,
            q_scale,
        );
        return Err(GpuError::InvalidArg(
            "certified KV selected-head fallback requires CUDA support".into(),
        ));
    }
    Ok(())
}

pub fn attend_int8_bf16_values_strided(
    ordinal: usize,
    query_bf16: &GpuBuffer,
    key_int8: &GpuBuffer,
    key_scale: &GpuBuffer,
    key_zero: &GpuBuffer,
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
        || (output_f32.dtype() != ScalarType::F32 && output_f32.dtype() != ScalarType::BF16)
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value dtypes must be BF16/U8/F32/BF16/F32/F32-or-BF16, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
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
    if !query_is_2d && !query_is_3d
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
    let key_stride_tokens = key_int8.shape()[1];
    let active_aligned_tokens = aligned_tokens(total_tokens, block_size);
    if key_int8.shape()[2] != head_dim
        || active_aligned_tokens == 0
        || key_stride_tokens < active_aligned_tokens
        || key_stride_tokens % block_size != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value key shape {:?} incompatible with active_aligned={active_aligned_tokens} head_dim={head_dim} block_size={block_size}",
            key_int8.shape()
        )));
    }
    let num_blocks = active_aligned_tokens / block_size;
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
    if value_stride_tokens < total_tokens {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value needs total_tokens={total_tokens} <= value_stride_tokens={value_stride_tokens}"
        )));
    }
    let tail_len = total_tokens - active_aligned_tokens;
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
    let key_scale_shape = key_scale.shape();
    let key_scale_stride_blocks = if key_scale_shape.len() == 3
        && key_scale_shape[0] == kv_heads
        && key_scale_shape[2] == head_dim
    {
        key_scale_shape[1]
    } else {
        0
    };
    if key_scale_stride_blocks < num_blocks
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
    let (tail_key_start_tokens, tail_key_stride_tokens) = if let Some(tail_key) = tail_key_bf16 {
        let tail_shape = tail_key.shape();
        let compact_3d = tail_shape == [kv_heads, tail_len, head_dim];
        let strided_3d =
            tail_shape.len() == 3 && tail_shape[0] == kv_heads && tail_shape[2] == head_dim;
        let strided_4d = tail_shape.len() == 4
            && tail_shape[0] == 1
            && tail_shape[1] == kv_heads
            && tail_shape[3] == head_dim;
        if tail_key.dtype() != ScalarType::BF16 || (!compact_3d && !strided_3d && !strided_4d) {
            return Err(GpuError::InvalidArg(format!(
                "certified KV strided INT8/BF16-value tail key expects BF16 [{kv_heads}, {tail_len}, {head_dim}], [{kv_heads}, stride, {head_dim}], or [1, {kv_heads}, stride, {head_dim}], got {:?} {:?}",
                tail_key.dtype(),
                tail_key.shape()
            )));
        }
        if compact_3d {
            (0, tail_len)
        } else {
            let stride = if tail_shape.len() == 3 {
                tail_shape[1]
            } else {
                tail_shape[2]
            };
            if stride < total_tokens {
                return Err(GpuError::InvalidArg(format!(
                    "certified KV strided INT8/BF16-value full tail key stride={stride} must cover total_tokens={total_tokens}"
                )));
            }
            (active_aligned_tokens, stride)
        }
    } else if tail_len != 0 {
        return Err(GpuError::InvalidArg(format!(
            "certified KV strided INT8/BF16-value needs tail key for tail_len={tail_len}"
        )));
    } else {
        (0, 0)
    };

    let backend = query_bf16.backend();
    match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            {
                let tail_key_ptr = tail_key_bf16.map_or(std::ptr::null(), GpuBuffer::as_ptr);
                let status = unsafe {
                    if output_f32.dtype() == ScalarType::BF16 {
                        dotcache_llama31_certified_kv_attend_int8_bf16_values_strided_out_bf16(
                            ordinal,
                            query_bf16.as_ptr(),
                            key_int8.as_ptr(),
                            key_scale.as_ptr(),
                            key_zero.as_ptr(),
                            value_bf16.as_ptr(),
                            tail_key_ptr,
                            score_scratch.as_mut_ptr(),
                            output_f32.as_mut_ptr(),
                            q_heads as c_int,
                            kv_heads as c_int,
                            num_blocks as c_int,
                            block_size as c_int,
                            tail_len as c_int,
                            key_stride_tokens as c_int,
                            key_scale_stride_blocks as c_int,
                            tail_key_start_tokens as c_int,
                            tail_key_stride_tokens as c_int,
                            score_stride_tokens as c_int,
                            value_stride_tokens as c_int,
                            head_dim as c_int,
                            gqa_group as c_int,
                            q_scale,
                        )
                    } else {
                        dotcache_llama31_certified_kv_attend_int8_bf16_values_strided(
                            ordinal,
                            query_bf16.as_ptr(),
                            key_int8.as_ptr(),
                            key_scale.as_ptr(),
                            key_zero.as_ptr(),
                            value_bf16.as_ptr(),
                            tail_key_ptr,
                            score_scratch.as_mut_ptr(),
                            output_f32.as_mut_ptr(),
                            q_heads as c_int,
                            kv_heads as c_int,
                            num_blocks as c_int,
                            block_size as c_int,
                            tail_len as c_int,
                            key_stride_tokens as c_int,
                            key_scale_stride_blocks as c_int,
                            tail_key_start_tokens as c_int,
                            tail_key_stride_tokens as c_int,
                            score_stride_tokens as c_int,
                            value_stride_tokens as c_int,
                            head_dim as c_int,
                            gqa_group as c_int,
                            q_scale,
                        )
                    }
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

    fn bf16s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
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
        let mut key_zero =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &key_scale_shape).expect("key_zero");
        let mut value_i4 =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &value_i4_shape).expect("value_i4");
        let mut value_scale =
            GpuBuffer::zeros(ordinal, ScalarType::F16, &value_meta_shape).expect("value_scale");
        let mut value_zero =
            GpuBuffer::zeros(ordinal, ScalarType::F16, &value_meta_shape).expect("value_zero");
        let mut value_error =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &value_error_shape).expect("value_error");
        let mut value_norm =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &value_error_shape).expect("value_norm");

        quantize_bf16_cache(
            ordinal,
            &key,
            &value,
            seq_len,
            block_size,
            value_group_size,
            &mut key_i8,
            &mut key_scale,
            &mut key_zero,
            &mut value_i4,
            &mut value_scale,
            &mut value_zero,
            &mut value_error,
            &mut value_norm,
        )
        .expect("quantize");

        let key_scales = f32s(&key_scale.to_host_bytes().expect("download key_scale"));
        assert!((key_scales[0] - (1.0 / 255.0)).abs() < 1.0e-6);
        assert!((key_scales[4] - (5.0 / 255.0)).abs() < 1.0e-6);
        let key_zeros = f32s(&key_zero.to_host_bytes().expect("download key_zero"));
        assert!((key_zeros[0] - (1.0 + 128.0 / 255.0)).abs() < 1.0e-6);
        assert!((key_zeros[4] - (-4.0 + 128.0 * 5.0 / 255.0)).abs() < 1.0e-6);
        let key_q = key_i8.to_host_bytes().expect("download key_i8");
        assert_eq!(key_q[0] as i8, -128);
        assert_eq!(key_q[1] as i8, -128);

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
        let mut key_only_zero =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &key_scale_shape).expect("key_only_zero");
        quantize_bf16_keys(
            ordinal,
            &key,
            seq_len,
            block_size,
            &mut key_only_i8,
            &mut key_only_scale,
            &mut key_only_zero,
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
        assert_eq!(
            key_only_zero
                .to_host_bytes()
                .expect("download key_only_zero"),
            key_zero.to_host_bytes().expect("download key_zero again")
        );

        let mut key_range_i8 =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[num_kv_heads, 6, head_dim])
                .expect("key_range_i8");
        let mut key_range_scale =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, 3, head_dim])
                .expect("key_range_scale");
        let mut key_range_zero =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, 3, head_dim])
                .expect("key_range_zero");
        quantize_bf16_keys_range(
            ordinal,
            &key,
            0,
            1,
            block_size,
            &mut key_range_i8,
            &mut key_range_scale,
            &mut key_range_zero,
        )
        .expect("quantize key range 0");
        quantize_bf16_keys_range(
            ordinal,
            &key,
            1,
            1,
            block_size,
            &mut key_range_i8,
            &mut key_range_scale,
            &mut key_range_zero,
        )
        .expect("quantize key range 1");
        let key_range_q = key_range_i8.to_host_bytes().expect("download key_range_i8");
        assert_eq!(
            &key_range_q[..num_kv_heads * seq_len * head_dim],
            &key_i8.to_host_bytes().expect("download key_i8 range cmp")[..]
        );
        let key_range_scales = key_range_scale
            .to_host_bytes()
            .expect("download key_range_scale");
        assert_eq!(
            &key_range_scales[..num_kv_heads * (seq_len / block_size) * head_dim * 4],
            &key_scale
                .to_host_bytes()
                .expect("download key_scale range cmp")[..]
        );
        let key_range_zeros = key_range_zero
            .to_host_bytes()
            .expect("download key_range_zero");
        assert_eq!(
            &key_range_zeros[..num_kv_heads * (seq_len / block_size) * head_dim * 4],
            &key_zero
                .to_host_bytes()
                .expect("download key_zero range cmp")[..]
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
        let mut block_max = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 1]).expect("block_max");
        let mut block_sum = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 1]).expect("block_sum");

        score_blocks_int8(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &key_zero,
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
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
            &key_zero,
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
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
            &key_zero,
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
    fn cuda_hybrid_attend_accepts_strided_int4_cache_and_tail() {
        set_backend(Backend::Cuda);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4],
            &bf16_bytes(&[1.0, 0.0, 0.0, 0.0]),
        )
        .expect("upload query");
        let key_i8 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 4, 4],
            &[0, 0, 0, 0, 0, 0, 0, 0, 99, 99, 99, 99, 88, 88, 88, 88],
        )
        .expect("upload key_i8");
        let key_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 4],
            &f32_bytes(&[1.0, 1.0, 1.0, 1.0, 99.0, 99.0, 99.0, 99.0]),
        )
        .expect("upload key_scale");
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 4]).expect("key_zero");
        let value_i4 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[1, 4, 2],
            &[0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff],
        )
        .expect("upload value_i4");
        let value_scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 4, 2],
            &f16_bytes(&[1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0]),
        )
        .expect("upload value_scale");
        let value_zero = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F16,
            &[1, 4, 2],
            &f16_bytes(&[0.0, 0.0, 0.0, 0.0, -100.0, -100.0, -100.0, -100.0]),
        )
        .expect("upload value_zero");
        let tail_key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4, 4],
            &bf16_bytes(&[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0,
            ]),
        )
        .expect("upload tail_key");
        let tail_value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 4, 4],
            &bf16_bytes(&[
                -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, 7.0, 8.0, 9.0, 10.0, -3.0, -3.0,
                -3.0, -3.0,
            ]),
        )
        .expect("upload tail_value");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("output");

        attend_int8_int4_with_bf16_tail_strided(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &key_zero,
            &value_i4,
            &value_scale,
            &value_zero,
            Some(&tail_key),
            Some(&tail_value),
            3,
            2,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("strided hybrid attend");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for (value, expected) in out.iter().zip([7.0_f32, 8.0, 9.0, 10.0]) {
            assert!(
                (value - expected).abs() < 0.01,
                "tail should dominate strided INT4 hybrid softmax: got={value} expected={expected}"
            );
        }

        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("score_scratch bf16");
        let mut output_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, 4]).expect("output bf16");
        attend_int8_int4_with_bf16_tail_strided(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &key_zero,
            &value_i4,
            &value_scale,
            &value_zero,
            Some(&tail_key),
            Some(&tail_value),
            3,
            2,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output_bf16,
        )
        .expect("strided hybrid attend bf16 output");
        let out = bf16s(&output_bf16.to_host_bytes().expect("download bf16 output"));
        for (value, expected) in out.iter().zip([7.0_f32, 8.0, 9.0, 10.0]) {
            assert!(
                (value - expected).abs() < 0.02,
                "tail should dominate strided INT4 hybrid BF16 output: got={value} expected={expected}"
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
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
            &key_zero,
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
        let values = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 5, 4],
            &bf16_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -100.0, -100.0,
                -100.0, -100.0, -200.0, -200.0, -200.0, -200.0,
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
            &key_zero,
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

        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3]).expect("score_scratch bf16");
        let mut output_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 4]).expect("output bf16");
        attend_int8_bf16_values_strided(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &key_zero,
            &values,
            Some(&tail_key),
            3,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output_bf16,
        )
        .expect("attend int8 keys with strided bf16 values and bf16 output");
        let out = bf16s(&output_bf16.to_host_bytes().expect("download bf16 output"));
        for (value, expected) in out.iter().zip([9.0_f32, 10.0, 11.0, 12.0]) {
            assert!(
                (value - expected).abs() < 0.02,
                "tail should dominate strided INT8-key/BF16-value BF16 output: got={value} expected={expected}"
            );
        }
    }

    #[test]
    fn cuda_int8_key_bf16_value_attend_accepts_strided_tail_key() {
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
        let key_zero = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 4]).expect("key_zero");
        let values = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 5, 4],
            &bf16_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -100.0, -100.0,
                -100.0, -100.0, -200.0, -200.0, -200.0, -200.0,
            ]),
        )
        .expect("upload values");
        let tail_key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 5, 4],
            &bf16_bytes(&[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ]),
        )
        .expect("upload tail_key");
        let mut score_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 5]).expect("score_scratch");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("output");

        attend_int8_bf16_values_strided(
            ordinal,
            &query,
            &key_i8,
            &key_scale,
            &key_zero,
            &values,
            Some(&tail_key),
            3,
            2,
            1,
            1.0,
            &mut score_scratch,
            &mut output,
        )
        .expect("attend int8 keys with strided bf16 values and tail key");

        let out = f32s(&output.to_host_bytes().expect("download output"));
        for (value, expected) in out.iter().zip([9.0_f32, 10.0, 11.0, 12.0]) {
            assert!(
                (value - expected).abs() < 0.01,
                "tail should dominate strided tail-key softmax: got={value} expected={expected}"
            );
        }
    }
}
