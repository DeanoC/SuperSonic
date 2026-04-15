use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

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
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
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

    fn dotcache_query_gpu_info(
        device_ordinal: c_int,
        arch_name_out: *mut u8,
        arch_name_len: usize,
        total_vram_out: *mut u64,
    ) -> c_int;
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
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
) -> Result<(), GpuError> {
    // sync_buf layout: counters[16 bytes] + barrier_counter[4 bytes] + barrier_flag[4 bytes]
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };

    let status = unsafe {
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
            proj_buf_floats,
            attn_scratch_floats,
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "persistent_decode kernel failed with status {status}"
        )));
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
    let status = unsafe {
        dotcache_qwen35_hip_rms_norm(
            dtype.kernel_dtype_code(),
            ordinal,
            1,          // n_rows (single token)
            hidden_dim, // n_cols
            eps,
            1, // add_unit_offset: weight applied as (w + 1.0) * x
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "rms_norm kernel failed with status {status}"
        )));
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
    // Reset the atomic row counter to 0
    gpu_hal::memset_zeros(ordinal, counter_buf.as_mut_ptr(), 4)?;

    let status = unsafe {
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
    };
    if status != 0 {
        return Err(GpuError::Hip(format!(
            "standalone_matvec kernel failed with status {status}"
        )));
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
        return Err(GpuError::Hip(format!(
            "query_gpu_info failed with status {status}"
        )));
    }
    let arch_name = std::ffi::CStr::from_bytes_until_nul(&arch_buf)
        .map_err(|_| GpuError::Hip("query_gpu_info: arch name not null-terminated".into()))?
        .to_string_lossy()
        .into_owned();
    Ok((arch_name, total_vram))
}
