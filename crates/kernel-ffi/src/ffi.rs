use std::ffi::{c_int, c_uint, c_void};

use crate::{metal_host, metal_native};
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
        enable_attention_trace: c_int,
        fp8_scales: *const c_void, // nullptr for BF16, pointer to FP8ScaleDesc[] for FP8
        kv_fp8_descs: *const c_void, // nullptr for BF16 KV, pointer to KVCacheFp8Desc[] for FP8 KV
        batch_size: usize,         // 1 for single-sequence (default), >1 for batched
        batch_descs: *const c_void, // nullptr for single-sequence, BatchSeqDesc[] for batched
        int4_scales: *const c_void, // nullptr for non-INT4, pointer to INT4ScaleDesc[] for INT4
        tap_workspace: *mut c_void, // nullptr for non-DFlash, [num_taps * hidden_dim] T for DFlash
        tap_layers: *const c_int,  // nullptr when tap_workspace is nullptr
        num_taps: usize,           // 0 when tap_workspace is nullptr
    ) -> c_int;

    fn dotcache_qwen35_cuda_persistent_decode_qwen35_4b_sm86_specialized(
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
        enable_attention_trace: c_int,
        fp8_scales: *const c_void,
        kv_fp8_descs: *const c_void,
        batch_size: usize,
        batch_descs: *const c_void,
        int4_scales: *const c_void,
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

// HIP-only: the clock-rate bridge lives in `full_attention_bridge.cpp` and has
// no CUDA counterpart (CUDA timing paths get clockRate via
// `cudaGetDeviceProperties` on the Rust side).
#[cfg(supersonic_backend_hip)]
unsafe extern "C" {
    fn dotcache_hip_device_clock_khz(device_ordinal: c_int, clock_rate_khz_out: *mut u32) -> c_int;

    // Per-model launch preset for the qwen4b persistent decode kernel.
    // `blocks=0` clears the preset (falls back to the hardcoded gfx11xx 2x
    // default). `coop != 0` opts into `hipLaunchCooperativeKernel` for that
    // preset, which is safe at higher block counts but caps conservatively
    // based on `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    fn dotcache_qwen35_4b_hip_set_launch_preset(blocks: c_int, coop: c_int);
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
        Backend::Metal => GpuError::Metal(format!("{what} failed with status {status}")),
    }
}

fn ffi_error(msg: String) -> GpuError {
    match gpu_hal::current_backend() {
        Backend::Hip => GpuError::Hip(msg),
        Backend::Cuda => GpuError::Cuda(msg),
        Backend::Metal => GpuError::Metal(msg),
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "persistent_decode is not supported on Metal v1".into(),
            ))
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Hip => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_qwen08_sm86_specialized is CUDA-only".into(),
            ))
        }
        Backend::Metal => {
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
/// `tap_workspace` / `tap_layers`: DFlash hidden-state taps. When `tap_workspace` is Some,
///   the kernel mirrors the per-layer post-MLP residual hidden state for each layer in
///   `tap_layers` into `tap_workspace` at offset `i * hidden_dim` (i = tap index, not layer
///   index). Both must be Some together or both None. The kernel body short-circuits the
///   tap write when `tap_workspace` is null to preserve gfx1150 codegen on the hot path.
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
    enable_attention_trace: bool,
    tap_workspace: Option<&mut GpuBuffer>,
    tap_layers: Option<&GpuBuffer>,
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

    let kv_fp8_ptr = kv_fp8_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

    let batch_descs_ptr = batch_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

    let int4_scales_ptr = int4_scale_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    // num_taps is derived from tap_layers length (4-byte ints). Both must be Some or both None.
    let (tap_ws_ptr, tap_layers_ptr, num_taps) = match (tap_workspace, tap_layers) {
        (Some(ws), Some(layers)) => (
            ws.as_mut_ptr(),
            layers.as_ptr() as *const c_int,
            layers.len_bytes() / std::mem::size_of::<c_int>(),
        ),
        (None, None) => (std::ptr::null_mut(), std::ptr::null::<c_int>(), 0),
        _ => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_4b: tap_workspace and tap_layers must both be Some or both None"
                    .into(),
            ));
        }
    };

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
                    if enable_attention_trace { 1 } else { 0 },
                    fp8_scales_ptr,
                    kv_fp8_ptr,
                    batch_size,
                    batch_descs_ptr,
                    int4_scales_ptr,
                    tap_ws_ptr,
                    tap_layers_ptr,
                    num_taps,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                    if enable_attention_trace { 1 } else { 0 },
                    fp8_scales_ptr,
                    kv_fp8_ptr,
                    batch_size,
                    batch_descs_ptr,
                    int4_scales_ptr,
                    tap_ws_ptr,
                    tap_layers_ptr,
                    num_taps,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_4b is not supported on Metal v1".into(),
            ))
        }
    };
    if status != 0 {
        return Err(backend_error(
            backend,
            "persistent_decode_4b kernel",
            status,
        ));
    }
    Ok(())
}

pub fn persistent_decode_4b_qwen35_sm86_specialized(
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
    enable_attention_trace: bool,
    tap_workspace: Option<&mut GpuBuffer>,
    tap_layers: Option<&GpuBuffer>,
) -> Result<(), GpuError> {
    if tap_workspace.is_some() || tap_layers.is_some() {
        return Err(GpuError::InvalidArg(
            "persistent_decode_4b_qwen35_sm86_specialized does not support DFlash taps".into(),
        ));
    }
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
    let kv_fp8_descs_ptr = kv_fp8_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let batch_descs_ptr = batch_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let int4_scales_ptr = int4_scale_descs
        .map(|b| b.as_ptr())
        .unwrap_or(std::ptr::null());

    let status = match backend {
        Backend::Cuda => {
            #[cfg(supersonic_backend_cuda)]
            unsafe {
                dotcache_qwen35_cuda_persistent_decode_qwen35_4b_sm86_specialized(
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
                    if enable_attention_trace { 1 } else { 0 },
                    fp8_scales_ptr,
                    kv_fp8_descs_ptr,
                    batch_size,
                    batch_descs_ptr,
                    int4_scales_ptr,
                )
            }
            #[cfg(not(supersonic_backend_cuda))]
            {
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Hip => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_4b_qwen35_sm86_specialized is CUDA-only".into(),
            ))
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "persistent_decode_4b_qwen35_sm86_specialized is CUDA-only".into(),
            ))
        }
    };

    if status != 0 {
        return Err(backend_error(
            backend,
            "persistent_decode_4b_qwen35_sm86_specialized kernel",
            status,
        ));
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
        let status =
            dotcache_qwen35_cuda_argmax_bf16(ordinal, n, logits.as_ptr(), out_index.as_mut_ptr());
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

pub fn metal_lm_head_argmax_bf16(
    ordinal: usize,
    hidden: &GpuBuffer,
    weight: &GpuBuffer,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<u32, GpuError> {
    if hidden.backend() != Backend::Metal || weight.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "metal_lm_head_argmax_bf16 requires Metal buffers".into(),
        ));
    }
    if hidden.dtype() != ScalarType::BF16 || weight.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "metal_lm_head_argmax_bf16 requires BF16 hidden/weight, got {:?}/{:?}",
            hidden.dtype(),
            weight.dtype()
        )));
    }
    let mut out_index = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])?;
    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    {
        crate::prefill_ffi::metal_profile_time("lm_head_argmax", "native", || {
            crate::metal_native::lm_head_argmax_bf16(
                hidden,
                weight,
                &mut out_index,
                hidden_dim,
                vocab_size,
            )
        })?;
        crate::metal_native::flush_batch()?;
        let bytes = out_index.to_host_bytes()?;
        let token = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok(token)
    }
    #[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
    {
        let _ = (hidden, weight, hidden_dim, vocab_size, out_index);
        Err(GpuError::InvalidArg("Metal backend not compiled".into()))
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "rms_norm_4b is not supported on Metal v1".into(),
            ))
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "rms_norm_4b kernel", status));
    }
    Ok(())
}

/// Maximum `in_dim` the standalone matvec kernels support. The kernels allocate
/// a fixed `__shared__ float shared_input[STANDALONE_MATVEC_MAX_IN_DIM]` for the
/// F32-cached input vector; anything larger overruns LDS and faults the same
/// way the 2B attn_scratch crash did (c.f. 338b939). Keep in sync with the
/// `shared_input[...]` declaration in every `full_attention*.hip`/`.cuh`.
pub const STANDALONE_MATVEC_MAX_IN_DIM: usize = 4096;

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
    if in_dim > STANDALONE_MATVEC_MAX_IN_DIM {
        return Err(GpuError::InvalidArg(format!(
            "standalone_matvec_4b in_dim={in_dim} exceeds kernel LDS bound \
             STANDALONE_MATVEC_MAX_IN_DIM={STANDALONE_MATVEC_MAX_IN_DIM}. \
             Raise `shared_input[...]` in full_attention_4b.hip + \
             full_attention_4b_cuda.cuh and bump the constant."
        )));
    }
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "standalone_matvec_4b is not supported on Metal v1".into(),
            ))
        }
    };
    if status != 0 {
        return Err(backend_error(
            backend,
            "standalone_matvec_4b kernel",
            status,
        ));
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "rms_norm_4b_multirow is not supported on Metal v1".into(),
            ))
        }
    };
    if status != 0 {
        return Err(backend_error(
            backend,
            "rms_norm_4b_multirow kernel",
            status,
        ));
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "matmul_rhs_transposed_4b is not supported on Metal v1".into(),
            ))
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
    if backend == Backend::Metal {
        let _ = ordinal;
        if dtype == ScalarType::BF16 && !metal_native::disabled_by_env() {
            if metal_native::rms_norm_rows_bf16(1, hidden_dim, eps, true, input, weight, output)
                .is_ok()
            {
                return Ok(());
            }
        }
        return metal_host::rms_norm_rows(dtype, 1, hidden_dim, eps, true, input, weight, output);
    }
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => unreachable!("handled above"),
    };
    if status != 0 {
        return Err(backend_error(backend, "rms_norm kernel", status));
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};
    use half::bf16;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16 buffer");
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    #[test]
    fn metal_rms_norm_applies_qwen_unit_offset() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2], &bf16_bytes(&[3.0, 4.0]))
                .expect("upload input");
        let weight =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2], &bf16_bytes(&[0.5, -0.5]))
                .expect("upload weight");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc output");

        rms_norm(
            ordinal,
            ScalarType::BF16,
            &mut output,
            &input,
            &weight,
            0.0,
            2,
        )
        .expect("run rms_norm");

        let actual = read_bf16(&output);
        let inv_rms = 1.0f32 / ((25.0f32 / 2.0).sqrt());
        let expected = vec![3.0 * inv_rms * 1.5, 4.0 * inv_rms * 0.5];
        for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (got - want).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {want}, got {got}, delta {delta}"
            );
        }
    }
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
    if in_dim > STANDALONE_MATVEC_MAX_IN_DIM {
        return Err(GpuError::InvalidArg(format!(
            "standalone_matvec in_dim={in_dim} exceeds kernel LDS bound \
             STANDALONE_MATVEC_MAX_IN_DIM={STANDALONE_MATVEC_MAX_IN_DIM}. \
             Raise `shared_input[...]` in full_attention.hip + \
             full_attention_cuda.cuh and bump the constant."
        )));
    }
    let backend = output.backend();
    if backend == Backend::Metal {
        let _ = (ordinal, counter_buf);
        if crate::metal_native::disabled_by_env()
            || std::env::var_os("SUPERSONIC_METAL_DISABLE_NATIVE_STANDALONE_MATVEC").is_some()
        {
            crate::metal_native::flush_batch()?;
            return crate::prefill_ffi::metal_profile_time("standalone_matvec", "host", || {
                metal_host::standalone_matvec(dtype, input, weight, output, in_dim, out_dim)
            });
        }
        return crate::prefill_ffi::metal_profile_time("standalone_matvec", "native", || {
            match dtype {
                ScalarType::BF16 => crate::metal_native::matmul_rhs_transposed_bf16(
                    1, 1, out_dim, in_dim, input, weight, output,
                ),
                ScalarType::F32 => crate::metal_native::matmul_rhs_transposed_f32(
                    1, 1, out_dim, in_dim, input, weight, output,
                ),
                other => Err(GpuError::InvalidArg(format!(
                    "standalone_matvec unsupported Metal dtype {other:?}"
                ))),
            }
        });
    }
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
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
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
                return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
            }
        }
        Backend::Metal => unreachable!("handled above"),
    };
    if status != 0 {
        return Err(backend_error(backend, "standalone_matvec kernel", status));
    }
    Ok(())
}

pub fn standalone_matvec_host_f32(
    ordinal: usize,
    dtype: ScalarType,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, GpuError> {
    let backend = input.backend();
    if backend == Backend::Metal {
        let _ = ordinal;
        crate::metal_native::flush_batch()?;
        return crate::prefill_ffi::metal_profile_time(
            "standalone_matvec_host_f32",
            "host",
            || metal_host::standalone_matvec_host_f32(dtype, input, weight, in_dim, out_dim),
        );
    }

    let mut output = GpuBuffer::zeros(ordinal, dtype, &[out_dim])?;
    let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])?;
    standalone_matvec(
        ordinal,
        dtype,
        &mut output,
        input,
        weight,
        in_dim,
        out_dim,
        &mut counter,
    )?;
    let bytes = output.to_host_bytes()?;
    match dtype {
        ScalarType::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect()),
        ScalarType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        other => Err(GpuError::InvalidArg(format!(
            "standalone_matvec_host_f32 unsupported dtype {other:?}"
        ))),
    }
}

pub fn qwen_rms_norm_standalone_matvec_host_f32(
    ordinal: usize,
    dtype: ScalarType,
    input: &GpuBuffer,
    norm_weight: &GpuBuffer,
    eps: f32,
    weight: &GpuBuffer,
    hidden_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, GpuError> {
    let backend = input.backend();
    if backend == Backend::Metal {
        let _ = ordinal;
        crate::metal_native::flush_batch()?;
        return crate::prefill_ffi::metal_profile_time(
            "qwen_rms_norm_standalone_matvec_host_f32",
            "host",
            || {
                metal_host::qwen_rms_norm_standalone_matvec_host_f32(
                    dtype,
                    input,
                    norm_weight,
                    eps,
                    weight,
                    hidden_dim,
                    out_dim,
                )
            },
        );
    }

    let mut normed = GpuBuffer::zeros(ordinal, dtype, &[hidden_dim])?;
    rms_norm(
        ordinal,
        dtype,
        &mut normed,
        input,
        norm_weight,
        eps,
        hidden_dim,
    )?;
    standalone_matvec_host_f32(ordinal, dtype, &normed, weight, hidden_dim, out_dim)
}

/// Query GPU architecture name and total VRAM for a given device ordinal.
pub fn query_gpu_info(ordinal: usize) -> Result<(String, u64), GpuError> {
    let backend = gpu_hal::current_backend();
    let info = gpu_hal::query_device_info(backend, ordinal)?;
    Ok((info.arch_name, info.total_vram_bytes))
}

/// Query the HIP device clock rate (kHz) for cycle→ms conversion in `--emit-stage-timings`.
///
/// On non-HIP builds this returns `GpuError::InvalidArg`; the caller is
/// responsible for only invoking it when the active backend is HIP.
pub fn query_hip_device_clock_khz(ordinal: usize) -> Result<u32, GpuError> {
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = ordinal;
        return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
    }
    #[cfg(supersonic_backend_hip)]
    {
        let mut clock_khz: u32 = 0;
        let status = unsafe { dotcache_hip_device_clock_khz(ordinal as c_int, &mut clock_khz) };
        if status != 0 {
            return Err(ffi_error(format!(
                "hip_device_clock_khz failed with status {status}"
            )));
        }
        Ok(clock_khz)
    }
}

/// Install a per-model launch preset for the qwen4b persistent decode
/// kernel on HIP. `blocks == 0` clears any active preset; a positive value
/// makes the bridge pick that grid size when no `SUPERSONIC_QWEN4B_BLOCKS`
/// env var is set. `coop == true` opts the preset into cooperative launch
/// (recommended whenever the preset exceeds the safe non-coop 2x default).
///
/// No-op on non-HIP builds — CUDA has its own separate bridge.
pub fn set_qwen35_4b_launch_preset(blocks: i32, coop: bool) {
    #[cfg(supersonic_backend_hip)]
    unsafe {
        dotcache_qwen35_4b_hip_set_launch_preset(blocks as c_int, if coop { 1 } else { 0 });
    }
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (blocks, coop);
    }
}
