//! Qwen3.8 HIP FFI wrappers.
//!
//! The `supersonic_qwen35_*` identifiers in the `extern "C"` block are
//! historical symbols exported by the retained HIP bridge. They are an
//! external ABI contract, so only those link names remain legacy-spelled;
//! Rust-facing wrappers use Qwen3.8 names.

use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

pub const PERSISTENT_4B_TIMING_SLOT_COUNT: usize = 43;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Persistent4BTimingSlots {
    pub full_attn: u64,
    pub full_attn_proj: u64,
    pub full_attn_core: [u64; 8],
    pub full_attn_out: [u64; 8],
    pub linear_proj: u64,
    pub linear_core: [u64; 8],
    pub linear_out: [u64; 8],
    pub linear_core_conv_reserved: [u64; 2],
    pub linear_core_recurrent_reserved: [u64; 2],
    pub linear_core_post_reserved: [u64; 2],
    pub mlp_gate_up: u64,
    pub mlp_down: u64,
}

pub fn parse_persistent_4b_timing_slots(slots: &[u64]) -> Result<Persistent4BTimingSlots, String> {
    if slots.len() != PERSISTENT_4B_TIMING_SLOT_COUNT {
        return Err(format!(
            "persistent 4B timing slot buffer must contain exactly {PERSISTENT_4B_TIMING_SLOT_COUNT} slots, got {}",
            slots.len()
        ));
    }
    Ok(Persistent4BTimingSlots {
        full_attn: slots[0],
        full_attn_proj: slots[1],
        full_attn_core: slots[2..10].try_into().unwrap(),
        full_attn_out: slots[10..18].try_into().unwrap(),
        linear_proj: slots[18],
        linear_core: slots[19..27].try_into().unwrap(),
        linear_out: slots[27..35].try_into().unwrap(),
        linear_core_conv_reserved: slots[35..37].try_into().unwrap(),
        linear_core_recurrent_reserved: slots[37..39].try_into().unwrap(),
        linear_core_post_reserved: slots[39..41].try_into().unwrap(),
        mlp_gate_up: slots[41],
        mlp_down: slots[42],
    })
}

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
        // ABI-reserved slot; Qwen3.8 always passes null for dynamic KV data.
        _reserved_kv_descs: *const c_void,
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

    fn supersonic_qwen35_4b_hip_set_gqh_prepare_only(on: c_int);

    #[link_name = "supersonic_qwen35_hip_mtp_restore_linear_prefix"]
    fn qwen38_mtp_restore_linear_prefix(
        device_ordinal: c_int,
        layers: *const c_void,
        commit_len: c_int,
    ) -> c_int;
}

fn hip_error(what: &str, status: c_int) -> GpuError {
    GpuError::backend(
        gpu_hal::Backend::Hip,
        format!("{what} failed with status {status}"),
    )
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
    if status != 0 {
        return Err(hip_error("persistent_decode kernel", status));
    }
    Ok(())
}

/// Restore the linear recurrent state after a fused speculative verification.
pub fn mtp_restore_linear_prefix(
    ordinal: usize,
    layers: &GpuBuffer,
    commit_len: usize,
) -> Result<bool, GpuError> {
    {
        let status = unsafe {
            qwen38_mtp_restore_linear_prefix(ordinal as c_int, layers.as_ptr(), commit_len as c_int)
        };
        return match status {
            0 => Ok(true),
            1 => Ok(false),
            status => Err(GpuError::InvalidArg(format!(
                "mtp_restore_linear_prefix failed: {status}"
            ))),
        };
    }
}

pub fn set_hip_gqh_prepare_only(on: bool) {
    unsafe {
        supersonic_qwen35_4b_hip_set_gqh_prepare_only(if on { 1 } else { 0 });
    }
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
    let batch_descs_ptr = batch_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
    let int4_scales_ptr = int4_scale_descs
        .map(|buffer| buffer.as_ptr())
        .unwrap_or(std::ptr::null());
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
            std::ptr::null(),
            batch_size,
            batch_descs_ptr,
            int4_scales_ptr,
        )
    };
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
}

pub fn query_hip_device_clock_khz(ordinal: usize) -> Result<u32, GpuError> {
    {
        let mut clock_khz = 0;
        let status = unsafe { supersonic_hip_device_clock_khz(ordinal as c_int, &mut clock_khz) };
        if status != 0 {
            return Err(hip_error("hip_device_clock_khz", status));
        }
        Ok(clock_khz)
    }
}

#[cfg(test)]
mod timing_slot_tests {
    use super::{parse_persistent_4b_timing_slots, PERSISTENT_4B_TIMING_SLOT_COUNT};

    #[test]
    fn parses_the_current_43_slot_abi() {
        assert_eq!(PERSISTENT_4B_TIMING_SLOT_COUNT, 43);
        let slots: Vec<u64> = (0..43).map(|value| value as u64).collect();
        let parsed = parse_persistent_4b_timing_slots(&slots).expect("current slot ABI");
        assert_eq!(parsed.full_attn, 0);
        assert_eq!(parsed.full_attn_proj, 1);
        assert_eq!(parsed.full_attn_core, [2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(parsed.full_attn_out, [10, 11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(parsed.linear_proj, 18);
        assert_eq!(parsed.linear_core, [19, 20, 21, 22, 23, 24, 25, 26]);
        assert_eq!(parsed.linear_out, [27, 28, 29, 30, 31, 32, 33, 34]);
        assert_eq!(parsed.linear_core_conv_reserved, [35, 36]);
        assert_eq!(parsed.linear_core_recurrent_reserved, [37, 38]);
        assert_eq!(parsed.linear_core_post_reserved, [39, 40]);
        assert_eq!(parsed.mlp_gate_up, 41);
        assert_eq!(parsed.mlp_down, 42);
    }

    #[test]
    fn rejects_non_43_slot_buffers() {
        let error = parse_persistent_4b_timing_slots(&[0; 42]).expect_err("wrong ABI length");
        assert!(error.to_string().contains("exactly 43"));
    }
}
