use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

#[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
unsafe extern "C" {
    fn dotcache_qwen35_hip_persistent_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        seqlen_offset: usize,
        layers: *const c_void,
        hidden_io: *mut c_void,
        workspace: *mut c_void,
        counters: *mut c_void,
        barrier_counter: *mut c_void,
        barrier_flag: *mut c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        rotary_dim: usize,
    ) -> c_int;

    fn dotcache_qwen35_cuda_persistent_decode_qwen08_sm86_specialized(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        seqlen_offset: usize,
        layers: *const c_void,
        hidden_io: *mut c_void,
        workspace: *mut c_void,
        counters: *mut c_void,
        barrier_counter: *mut c_void,
        barrier_flag: *mut c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        rotary_dim: usize,
    ) -> c_int;

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

    fn dotcache_qwen35_hip_standalone_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    // 4B kernel variant (separate compilation to avoid hipcc codegen issues)
    fn dotcache_qwen35_4b_hip_persistent_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        seqlen_offset: usize,
        layers: *const c_void,
        hidden_io: *mut c_void,
        workspace: *mut c_void,
        counters: *mut c_void,
        barrier_counter: *mut c_void,
        barrier_flag: *mut c_void,
        timing_slots: *mut c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        rotary_dim: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        fp8_scales: *const c_void,  // nullptr for BF16, pointer to FP8ScaleDesc[] for FP8
        kv_fp8_descs: *const c_void,  // nullptr for BF16 KV, pointer to KVCacheFp8Desc[] for FP8 KV
        batch_size: usize,            // 1 for single-sequence (default), >1 for batched
        batch_descs: *const c_void,   // nullptr for single-sequence, BatchSeqDesc[] for batched
        int4_scales: *const c_void,   // nullptr for non-INT4, pointer to INT4ScaleDesc[] for INT4
    ) -> c_int;

    fn dotcache_qwen35_4b_hip_rms_norm(
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

    fn dotcache_qwen35_4b_hip_standalone_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

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

    fn dotcache_query_gpu_info(
        device_ordinal: c_int,
        arch_name_out: *mut u8,
        arch_name_len: usize,
        total_vram_out: *mut u64,
    ) -> c_int;
}

#[cfg(supersonic_backend_cuda)]
unsafe extern "C" {
    fn dotcache_qwen35_cuda_argmax_bf16(
        device_ordinal: usize,
        n: usize,
        logits: *const c_void,
        out_index: *mut c_void,
    ) -> c_int;

    fn dotcache_qwen35_cuda_lm_head_argmax_bf16(
        device_ordinal: usize,
        hidden_dim: usize,
        vocab_size: usize,
        hidden: *const c_void,
        weight: *const c_void,
        block_best_vals: *mut c_void,
        block_best_idxs: *mut c_void,
        out_index: *mut c_void,
    ) -> c_int;
}

fn backend_error(backend: Backend, what: &str, status: c_int) -> GpuError {
    match backend {
        Backend::Hip => GpuError::Hip(format!("{what} failed with status {status}")),
        Backend::Cuda => GpuError::Cuda(format!("{what} failed with status {status}")),
    }
}

fn ffi_error(msg: String) -> GpuError {
    match gpu_hal::current_backend() {
        Backend::Hip => GpuError::Hip(msg),
        Backend::Cuda => GpuError::Cuda(msg),
    }
}

/// Safe wrapper around the persistent decode kernel.
pub fn persistent_decode(
    ordinal: usize,
    dtype: ScalarType,
    num_layers: usize,
    hidden_dim: usize,
    intermediate_size: usize,
    seqlen_offset: usize,
    layer_descs_device: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    rotary_dim: usize,
) -> Result<(), GpuError> {
    let backend = layer_descs_device.backend();
    // sync_buf layout: counters[16 bytes] + barrier_counter[4 bytes] + barrier_flag[4 bytes]
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_hip_persistent_decode(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_descs_device.as_ptr(),
                    hidden_io.as_mut_ptr(),
                    workspace.as_mut_ptr(),
                    counters,
                    barrier_counter,
                    barrier_flag,
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    rotary_dim,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_hip_persistent_decode(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_descs_device.as_ptr(),
                    hidden_io.as_mut_ptr(),
                    workspace.as_mut_ptr(),
                    counters,
                    barrier_counter,
                    barrier_flag,
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    rotary_dim,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "persistent_decode kernel", status));
    }
    Ok(())
}

pub fn persistent_decode_qwen08_sm86_specialized(
    ordinal: usize,
    dtype: ScalarType,
    num_layers: usize,
    hidden_dim: usize,
    intermediate_size: usize,
    seqlen_offset: usize,
    layer_descs_device: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    rotary_dim: usize,
) -> Result<(), GpuError> {
    let backend = layer_descs_device.backend();
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };

    let status = match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_cuda_persistent_decode_qwen08_sm86_specialized(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_descs_device.as_ptr(),
                    hidden_io.as_mut_ptr(),
                    workspace.as_mut_ptr(),
                    counters,
                    barrier_counter,
                    barrier_flag,
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    rotary_dim,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Hip => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_qwen08_sm86_specialized is CUDA-only".into(),
            ))
        }
    };

    if status != 0 {
        return Err(backend_error(
            backend,
            "persistent_decode_qwen08_sm86_specialized kernel",
            status,
        ));
    }
    Ok(())
}

/// Safe wrapper around the 4B persistent decode kernel (separate compilation).
/// `fp8_scale_descs`: when Some, contains FP8ScaleDesc array on GPU for runtime FP8 dequant.
/// `kv_fp8_descs`: when Some, contains KVCacheFp8Desc array on GPU for FP8 KV cache.
/// `batch_size`: number of sequences (1 = single-sequence, default).
/// `batch_descs`: when Some, contains BatchSeqDesc array on GPU for per-sequence state.
/// `int4_scale_descs`: when Some, contains INT4ScaleDesc array on GPU for INT4 dequant.
pub fn persistent_decode_4b(
    ordinal: usize,
    dtype: ScalarType,
    num_layers: usize,
    hidden_dim: usize,
    intermediate_size: usize,
    seqlen_offset: usize,
    layer_descs_device: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    rotary_dim: usize,
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
    fp8_scale_descs: Option<&GpuBuffer>,
    kv_fp8_descs: Option<&GpuBuffer>,
    batch_size: usize,
    batch_descs: Option<&GpuBuffer>,
    int4_scale_descs: Option<&GpuBuffer>,
    enable_timing_slots: bool,
) -> Result<(), GpuError> {
    let backend = layer_descs_device.backend();
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };
    let timing_slots = if enable_timing_slots {
        unsafe { (counters as *mut u8).add(24) as *mut c_void }
    } else {
        std::ptr::null_mut()
    };

    let fp8_scales_ptr = fp8_scale_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    let kv_fp8_ptr = kv_fp8_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    let batch_descs_ptr = batch_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    let int4_scales_ptr = int4_scale_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_4b_hip_persistent_decode(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_descs_device.as_ptr(),
                    hidden_io.as_mut_ptr(),
                    workspace.as_mut_ptr(),
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    fp8_scales_ptr,
                    kv_fp8_ptr,
                    batch_size,
                    batch_descs_ptr,
                    int4_scales_ptr,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_4b_hip_persistent_decode(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_descs_device.as_ptr(),
                    hidden_io.as_mut_ptr(),
                    workspace.as_mut_ptr(),
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    fp8_scales_ptr,
                    kv_fp8_ptr,
                    batch_size,
                    batch_descs_ptr,
                    int4_scales_ptr,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "persistent_decode_4b kernel", status));
    }
    Ok(())
}

pub fn cuda_argmax_bf16(
    ordinal: usize,
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    if logits.backend() != Backend::Cuda || out_index.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "cuda_argmax_bf16 requires CUDA buffers".into(),
        ));
    }
    if logits.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "cuda_argmax_bf16 requires BF16 logits, got {:?}",
            logits.dtype()
        )));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() != 1 {
        return Err(GpuError::InvalidArg(
            "cuda_argmax_bf16 requires a U32[1] output buffer".into(),
        ));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_qwen35_cuda_argmax_bf16(
            ordinal,
            n,
            logits.as_ptr(),
            out_index.as_mut_ptr(),
        );
        if status != 0 {
            return Err(GpuError::Cuda(format!(
                "cuda_argmax_bf16 failed with status {status}"
            )));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (ordinal, logits, out_index, n);
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

pub fn cuda_lm_head_argmax_bf16(
    ordinal: usize,
    hidden: &GpuBuffer,
    weight: &GpuBuffer,
    block_best_vals: &mut GpuBuffer,
    block_best_idxs: &mut GpuBuffer,
    out_index: &mut GpuBuffer,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<(), GpuError> {
    if hidden.backend() != Backend::Cuda
        || weight.backend() != Backend::Cuda
        || block_best_vals.backend() != Backend::Cuda
        || block_best_idxs.backend() != Backend::Cuda
        || out_index.backend() != Backend::Cuda
    {
        return Err(GpuError::InvalidArg(
            "cuda_lm_head_argmax_bf16 requires CUDA buffers".into(),
        ));
    }
    if hidden.dtype() != ScalarType::BF16 || weight.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "cuda_lm_head_argmax_bf16 requires BF16 hidden/weight, got {:?}/{:?}",
            hidden.dtype(),
            weight.dtype()
        )));
    }
    if block_best_vals.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(
            "cuda_lm_head_argmax_bf16 requires F32 block_best_vals".into(),
        ));
    }
    if block_best_idxs.dtype() != ScalarType::U32 {
        return Err(GpuError::InvalidArg(
            "cuda_lm_head_argmax_bf16 requires U32 block_best_idxs".into(),
        ));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() != 1 {
        return Err(GpuError::InvalidArg(
            "cuda_lm_head_argmax_bf16 requires a U32[1] output buffer".into(),
        ));
    }
    #[cfg(supersonic_backend_cuda)]
    unsafe {
        let status = dotcache_qwen35_cuda_lm_head_argmax_bf16(
            ordinal,
            hidden_dim,
            vocab_size,
            hidden.as_ptr(),
            weight.as_ptr(),
            block_best_vals.as_mut_ptr(),
            block_best_idxs.as_mut_ptr(),
            out_index.as_mut_ptr(),
        );
        if status != 0 {
            return Err(GpuError::Cuda(format!(
                "cuda_lm_head_argmax_bf16 failed with status {status}"
            )));
        }
        Ok(())
    }
    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            hidden,
            weight,
            block_best_vals,
            block_best_idxs,
            out_index,
            hidden_dim,
            vocab_size,
        );
        Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
    }
}

/// 4B variant of RMSNorm (same logic, separate compilation).
pub fn rms_norm_4b(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
    hidden_dim: usize,
) -> Result<(), GpuError> {
    let backend = output.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_4b_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    1,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_4b_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    1,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "rms_norm_4b kernel", status));
    }
    Ok(())
}

/// 4B variant of standalone matvec (same logic, separate compilation).
pub fn standalone_matvec_4b(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = output.backend();
    gpu_hal::memset_zeros(ordinal, counter_buf.as_mut_ptr(), 4)?;
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_4b_hip_standalone_matvec(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    in_dim,
                    out_dim,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    counter_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_4b_hip_standalone_matvec(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    in_dim,
                    out_dim,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    counter_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "standalone_matvec_4b kernel", status));
    }
    Ok(())
}

/// 4B RMSNorm over multiple contiguous rows.
pub fn rms_norm_4b_multirow(
    ordinal: usize,
    dtype: ScalarType,
    n_rows: usize,
    hidden_dim: usize,
    eps: f32,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = out.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_4b_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    n_rows,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_4b_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    n_rows,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "rms_norm_4b_multirow kernel", status));
    }
    Ok(())
}

/// 4B tiled BF16 matmul with transposed rhs: out [batch, m, n] = lhs [batch, m, k] × rhs^T.
pub fn matmul_rhs_transposed_4b(
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
    let backend = out.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
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
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
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
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(
            backend,
            "matmul_rhs_transposed_4b kernel",
            status,
        ));
    }
    Ok(())
}

/// Apply RMSNorm on device. Qwen3.5 uses add_unit_offset=1 (weight + 1.0).
pub fn rms_norm(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
    hidden_dim: usize,
) -> Result<(), GpuError> {
    let backend = output.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    1,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_hip_rms_norm(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    1,
                    hidden_dim,
                    eps,
                    1,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "rms_norm kernel", status));
    }
    Ok(())
}

/// Matrix-vector multiply for output projection (lm_head).
/// Uses work-stealing matvec kernel. `counter_buf` is a small device buffer
/// for the atomic row counter (4 bytes, reset to 0 before each call).
pub fn standalone_matvec(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = output.backend();
    // Reset the atomic row counter to 0
    gpu_hal::memset_zeros(ordinal, counter_buf.as_mut_ptr(), 4)?;

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dotcache_qwen35_hip_standalone_matvec(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    in_dim,
                    out_dim,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    counter_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
        }
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_hip_standalone_matvec(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    in_dim,
                    out_dim,
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    counter_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "standalone_matvec kernel", status));
    }
    Ok(())
}

/// Query GPU architecture name and total VRAM for a given device ordinal.
pub fn query_gpu_info(ordinal: usize) -> Result<(String, u64), GpuError> {
    let mut arch_buf = [0u8; 64];
    let mut total_vram: u64 = 0;
    let status = unsafe {
        dotcache_query_gpu_info(
            ordinal as c_int,
            arch_buf.as_mut_ptr(),
            arch_buf.len(),
            &mut total_vram,
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "query_gpu_info failed with status {status}"
        )));
    }
    let arch_name = std::ffi::CStr::from_bytes_until_nul(&arch_buf)
        .map_err(|_| ffi_error("query_gpu_info: arch name not null-terminated".into()))?
        .to_string_lossy()
        .into_owned();
    Ok((arch_name, total_vram))
}
