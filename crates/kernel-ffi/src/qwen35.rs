use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

#[cfg(supersonic_backend_hip)]
unsafe extern "C" {
    fn supersonic_qwen35_hip_persistent_decode(
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

    fn supersonic_qwen35_hip_rms_norm(
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

    fn supersonic_qwen35_hip_standalone_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_persistent_decode(
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

    fn supersonic_qwen35_4b_hip_rms_norm(
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

    fn supersonic_qwen35_4b_hip_standalone_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
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

    fn supersonic_query_gpu_info(
        device_ordinal: c_int,
        arch_name_out: *mut u8,
        arch_name_len: usize,
        total_vram_out: *mut u64,
    ) -> c_int;

    fn supersonic_hip_device_clock_khz(
        device_ordinal: c_int,
        clock_rate_khz_out: *mut u32,
    ) -> c_int;

    fn supersonic_qwen35_4b_hip_set_launch_preset(blocks: c_int, coop: c_int);
    fn supersonic_qwen35_4b_hip_set_gqh_prepare_only(on: c_int);

    #[link_name = "supersonic_qwen35_hip_mtp_restore_linear_prefix"]
    fn qwen38_mtp_restore_linear_prefix(commit_len: c_int) -> c_int;
}

fn hip_error(what: &str, status: c_int) -> GpuError {
    GpuError::backend(
        gpu_hal::Backend::Hip,
        format!("{what} failed with status {status}"),
    )
}

fn hip_not_compiled() -> GpuError {
    GpuError::InvalidArg("HIP backend not compiled".into())
}

/// Invoke the dense Qwen3.8 persistent decode kernel.
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
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_hip_persistent_decode(
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
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (
            ordinal,
            dtype,
            num_layers,
            hidden_dim,
            intermediate_size,
            seqlen_offset,
            layer_descs_device,
            hidden_io,
            workspace,
            sync_buf,
            cos_table,
            sin_table,
            rotary_dim,
        );
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("persistent_decode kernel", status));
    }
    Ok(())
}

/// Restore the linear recurrent state after a fused speculative verification.
pub fn mtp_restore_linear_prefix(commit_len: usize) -> Result<bool, GpuError> {
    #[cfg(supersonic_backend_hip)]
    {
        let status = unsafe { qwen38_mtp_restore_linear_prefix(commit_len as c_int) };
        return match status {
            0 => Ok(true),
            1 => Ok(false),
            status => Err(GpuError::InvalidArg(format!(
                "mtp_restore_linear_prefix failed: {status}"
            ))),
        };
    }
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = commit_len;
        Err(hip_not_compiled())
    }
}

pub fn set_hip_gqh_prepare_only(on: bool) {
    #[cfg(supersonic_backend_hip)]
    unsafe {
        supersonic_qwen35_4b_hip_set_gqh_prepare_only(if on { 1 } else { 0 });
    }
    #[cfg(not(supersonic_backend_hip))]
    let _ = on;
}

/// Invoke the 4B persistent decode kernel.
#[allow(clippy::too_many_arguments)]
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
) -> Result<(), GpuError> {
    let counters = sync_buf.as_mut_ptr();
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_void };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_void };
    let timing_slots = if enable_timing_slots {
        unsafe { (counters as *mut u8).add(24) as *mut c_void }
    } else {
        std::ptr::null_mut()
    };
    let fp8_scales_ptr = fp8_scale_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
    let kv_fp8_ptr = kv_fp8_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
    let batch_descs_ptr = batch_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
    let int4_scales_ptr = int4_scale_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_4b_hip_persistent_decode(
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
        )
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (
            ordinal,
            dtype,
            num_layers,
            hidden_dim,
            intermediate_size,
            seqlen_offset,
            layer_descs_device,
            hidden_io,
            workspace,
            sync_buf,
            cos_table,
            sin_table,
            rotary_dim,
            proj_buf_floats,
            attn_scratch_floats,
            fp8_scale_descs,
            kv_fp8_descs,
            batch_size,
            batch_descs,
            int4_scale_descs,
            enable_timing_slots,
            enable_attention_trace,
        );
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("persistent_decode_4b kernel", status));
    }
    Ok(())
}

/// Maximum input dimension supported by the shared-memory matvec kernels.
pub const STANDALONE_MATVEC_MAX_IN_DIM: usize = 8192;

pub fn rms_norm_4b(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
    hidden_dim: usize,
) -> Result<(), GpuError> {
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_4b_hip_rms_norm(
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
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (ordinal, dtype, output, input, weight, eps, hidden_dim);
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("rms_norm_4b kernel", status));
    }
    Ok(())
}

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
            "standalone_matvec_4b in_dim={in_dim} exceeds shared-memory bound {STANDALONE_MATVEC_MAX_IN_DIM}"
        )));
    }
    gpu_hal::memset_zeros(ordinal, counter_buf.as_mut_ptr(), 4)?;
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_4b_hip_standalone_matvec(
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
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (
            ordinal,
            dtype,
            output,
            input,
            weight,
            in_dim,
            out_dim,
            counter_buf,
        );
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("standalone_matvec_4b kernel", status));
    }
    Ok(())
}

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
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_4b_hip_rms_norm(
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
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (ordinal, dtype, n_rows, hidden_dim, eps, input, weight, out);
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("rms_norm_4b_multirow kernel", status));
    }
    Ok(())
}

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
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
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
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (ordinal, dtype, batch_elems, m, n, k, lhs, rhs, out);
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("matmul_rhs_transposed_4b kernel", status));
    }
    Ok(())
}

pub fn rms_norm(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
    hidden_dim: usize,
) -> Result<(), GpuError> {
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_hip_rms_norm(
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
    };
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (ordinal, dtype, output, input, weight, eps, hidden_dim);
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("rms_norm kernel", status));
    }
    Ok(())
}

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
            "standalone_matvec in_dim={in_dim} exceeds shared-memory bound {STANDALONE_MATVEC_MAX_IN_DIM}"
        )));
    }
    gpu_hal::memset_zeros(ordinal, counter_buf.as_mut_ptr(), 4)?;
    #[cfg(supersonic_backend_hip)]
    let status = unsafe {
        supersonic_qwen35_hip_standalone_matvec(
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
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (
            ordinal,
            dtype,
            output,
            input,
            weight,
            in_dim,
            out_dim,
            counter_buf,
        );
        return Err(hip_not_compiled());
    }
    if status != 0 {
        return Err(hip_error("standalone_matvec kernel", status));
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

pub fn query_gpu_info(ordinal: usize) -> Result<(String, u64), GpuError> {
    #[cfg(supersonic_backend_hip)]
    {
        let mut arch_name = vec![0u8; 64];
        let mut total_vram = 0u64;
        let status = unsafe {
            supersonic_query_gpu_info(
                ordinal as c_int,
                arch_name.as_mut_ptr(),
                arch_name.len(),
                &mut total_vram,
            )
        };
        if status != 0 {
            return Err(hip_error("supersonic_query_gpu_info", status));
        }
        let nul_pos = arch_name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(arch_name.len());
        Ok((
            String::from_utf8_lossy(&arch_name[..nul_pos]).into_owned(),
            total_vram,
        ))
    }
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = ordinal;
        Err(hip_not_compiled())
    }
}

pub fn query_hip_device_clock_khz(ordinal: usize) -> Result<u32, GpuError> {
    #[cfg(supersonic_backend_hip)]
    {
        let mut clock_khz = 0;
        let status = unsafe { supersonic_hip_device_clock_khz(ordinal as c_int, &mut clock_khz) };
        if status != 0 {
            return Err(hip_error("hip_device_clock_khz", status));
        }
        Ok(clock_khz)
    }
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = ordinal;
        Err(hip_not_compiled())
    }
}

pub fn set_qwen35_4b_launch_preset(blocks: i32, coop: bool) {
    #[cfg(supersonic_backend_hip)]
    unsafe {
        supersonic_qwen35_4b_hip_set_launch_preset(blocks as c_int, if coop { 1 } else { 0 });
    }
    #[cfg(not(supersonic_backend_hip))]
    let _ = (blocks, coop);
}
