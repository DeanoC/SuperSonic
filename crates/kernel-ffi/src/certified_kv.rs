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
    }
}
