//! FFI bridge for the Phi-4-mini persistent decode megakernel.
//!
//! Phi-4-mini runs 32 pure full-attention layers (no sliding, no hybrid linear),
//! GQA 3:1, SwiGLU MLP, partial RoPE with `rot_dim = 96` driven per-layer via
//! [`Phi4DecodeLayerDesc::rot_dim`]. LongRoPE is handled runner-side: the
//! runner picks the short- or long-factor cos/sin pair by `kv_len` and passes
//! the chosen pair to the kernel. Mscale is baked into the tables, so the
//! kernel needs no scaling logic of its own.
//!
//! BF16 is the only wired dtype at launch. [`Phi4FP8ScaleDesc`],
//! [`Phi4INT4ScaleDesc`], and [`Phi4KVCacheFp8Desc`] are kept as parallel
//! structs so the kernel-side helpers (which are already compiled into
//! `phi4.hip`) can be enabled by flipping `--fp8-runtime`, `--int4`, or
//! `--kv-fp8` once the bakes and engine paths are wired.

use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

use crate::layer_desc::MAX_BATCH_SIZE;

/// Mirrors `Phi4DecodeLayerDesc` in `kernels/phi4.hip`. Field order and
/// natural x86_64 alignment must match the C++ struct exactly.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Phi4DecodeLayerDesc {
    pub intermediate_size: c_int,
    pub rot_dim: c_int,
    // --- RMSNorm ---
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,
    // --- MLP (SwiGLU) ---
    pub gate_proj_w: *const c_void,
    pub up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,
    // --- Full attention (GQA) ---
    pub q_proj_w: *const c_void,
    pub q_out_dim: c_int,
    pub k_proj_w: *const c_void,
    pub k_out_dim: c_int,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub attn_head_dim: c_int,
    pub attn_num_heads: c_int,
    pub attn_num_kv_heads: c_int,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,
    pub kv_shadow_k: *mut c_void,
    pub kv_shadow_v: *mut c_void,
    pub kv_shadow_start: c_int,
}

unsafe impl Send for Phi4DecodeLayerDesc {}
unsafe impl Sync for Phi4DecodeLayerDesc {}

impl Default for Phi4DecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// FP8 scale pointers for Phi-4 weight dequant. Parallel-struct to
/// [`Phi4DecodeLayerDesc`]; pass a nullptr (`None` → null layer array) when
/// running BF16.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Phi4FP8ScaleDesc {
    pub gate_proj_scale: *const c_void,
    pub up_proj_scale: *const c_void,
    pub down_proj_scale: *const c_void,
    pub q_proj_scale: *const c_void,
    pub k_proj_scale: *const c_void,
    pub v_proj_scale: *const c_void,
    pub o_proj_scale: *const c_void,
    pub block_size: c_int,
}

unsafe impl Send for Phi4FP8ScaleDesc {}
unsafe impl Sync for Phi4FP8ScaleDesc {}

impl Default for Phi4FP8ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-layer FP8 KV cache scale pointers for Phi-4. Parallel-struct to
/// [`Phi4DecodeLayerDesc`].
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Phi4KVCacheFp8Desc {
    pub kv_scale_k: *mut c_void,
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for Phi4KVCacheFp8Desc {}
unsafe impl Sync for Phi4KVCacheFp8Desc {}

impl Default for Phi4KVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// INT4 scale + zero pointers for Phi-4 weight dequant. Parallel-struct to
/// [`Phi4DecodeLayerDesc`]; the main desc's `*_w` slots reinterpret as packed
/// INT4 (2 nibbles per byte) when `--int4` is active.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Phi4INT4ScaleDesc {
    pub gate_proj_scale: *const c_void,
    pub gate_proj_zero: *const c_void,
    pub up_proj_scale: *const c_void,
    pub up_proj_zero: *const c_void,
    pub down_proj_scale: *const c_void,
    pub down_proj_zero: *const c_void,
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
    pub group_size: c_int,
}

unsafe impl Send for Phi4INT4ScaleDesc {}
unsafe impl Sync for Phi4INT4ScaleDesc {}

impl Default for Phi4INT4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-sequence batched decode state. One per layer, parallel to
/// [`Phi4DecodeLayerDesc`]. Only the first `batch_size` slots are read.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Phi4BatchSeqDesc {
    pub seqlen_offset: [c_int; MAX_BATCH_SIZE],
    pub kv_cache_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_cache_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_len: [c_int; MAX_BATCH_SIZE],
    pub kv_max_t: [c_int; MAX_BATCH_SIZE],
    pub kv_shadow_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_shadow_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_shadow_start: [c_int; MAX_BATCH_SIZE],
    pub kv_scale_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_scale_v: [*mut c_void; MAX_BATCH_SIZE],
}

unsafe impl Send for Phi4BatchSeqDesc {}
unsafe impl Sync for Phi4BatchSeqDesc {}

impl Default for Phi4BatchSeqDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

extern "C" {
    /// Standalone BF16 RMSNorm without add_unit_offset. Used for the final
    /// model norm (`model.norm.weight`) before lm_head. Phi-4 norms use
    /// plain `x * inv_rms * w`; Qwen's `rms_norm_4b` bakes `+1.0f` into the
    /// weight term which would give the wrong output on Phi-4 weights.
    pub fn phi4_hip_rms_norm(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    /// Launch the Phi-4-mini persistent decode megakernel. `dtype` follows
    /// the Qwen/Gemma bridges (0 = half, 2 = bf16); only bf16 is wired.
    ///
    /// `fp8_scales`, `kv_fp8_descs`, and `int4_scales` must be null at BF16
    /// launch. `batch_descs` must be null when `batch_size == 1`.
    ///
    /// Prefer [`persistent_decode`] — this extern is exposed for the engine
    /// layer and should not be called from outside `kernel_ffi`.
    pub fn phi4_hip_persistent_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        seqlen_offset: usize,
        layers: *const Phi4DecodeLayerDesc,
        hidden_io: *mut c_void,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
        cos_table: *const c_void,
        sin_table: *const c_void,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        fp8_scales: *const Phi4FP8ScaleDesc,
        kv_fp8_descs: *const Phi4KVCacheFp8Desc,
        batch_size: usize,
        batch_descs: *const Phi4BatchSeqDesc,
        int4_scales: *const Phi4INT4ScaleDesc,
    ) -> c_int;
}

/// Single-row standalone RMSNorm (no add_unit_offset) — suitable for the
/// final `model.norm.weight` step before the lm_head matvec. BF16 only.
pub fn rms_norm(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    eps: f32,
    hidden_dim: usize,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "phi4::rms_norm: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = output.backend();
    let status = match backend {
        Backend::Hip => unsafe {
            phi4_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                1,
                hidden_dim,
                eps,
                input.as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
            )
        },
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "phi4::rms_norm: CUDA backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(phi4_backend_error(backend, "phi4 rms_norm kernel", status));
    }
    Ok(())
}

fn phi4_backend_error(backend: Backend, what: &str, status: c_int) -> GpuError {
    match backend {
        Backend::Hip => GpuError::Hip(format!("{what} failed with status {status}")),
        Backend::Cuda => GpuError::Cuda(format!("{what} failed with status {status}")),
    }
}

/// Safe wrapper over [`phi4_hip_persistent_decode`]. The engine must
/// pre-allocate `sync_buf` as a 32-byte zeroed GPU scratch, which the kernel
/// uses for the work-stealing counter (offset 0) and the grid barrier
/// counter + flag (offsets 16 and 20 — identical layout to
/// [`crate::persistent_decode_4b`]). BF16 is the only wired dtype at launch;
/// `fp8_scale_descs`, `kv_fp8_descs`, and `int4_scale_descs` should be `None`.
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
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
    fp8_scale_descs: Option<&GpuBuffer>,
    kv_fp8_descs: Option<&GpuBuffer>,
    batch_size: usize,
    batch_descs: Option<&GpuBuffer>,
    int4_scale_descs: Option<&GpuBuffer>,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "phi4::persistent_decode: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = layer_descs_device.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_uint };

    let fp8_scales_ptr = fp8_scale_descs
        .map(|b| b.as_ptr() as *const Phi4FP8ScaleDesc)
        .unwrap_or(std::ptr::null());
    let kv_fp8_ptr = kv_fp8_descs
        .map(|b| b.as_ptr() as *const Phi4KVCacheFp8Desc)
        .unwrap_or(std::ptr::null());
    let batch_descs_ptr = batch_descs
        .map(|b| b.as_ptr() as *const Phi4BatchSeqDesc)
        .unwrap_or(std::ptr::null());
    let int4_ptr = int4_scale_descs
        .map(|b| b.as_ptr() as *const Phi4INT4ScaleDesc)
        .unwrap_or(std::ptr::null());

    let status = match backend {
        Backend::Hip => unsafe {
            phi4_hip_persistent_decode(
                dtype.kernel_dtype_code(),
                ordinal,
                num_layers,
                hidden_dim,
                intermediate_size,
                seqlen_offset,
                layer_descs_device.as_ptr() as *const Phi4DecodeLayerDesc,
                hidden_io.as_mut_ptr(),
                workspace.as_mut_ptr() as *mut f32,
                counters,
                barrier_counter,
                barrier_flag,
                cos_table.as_ptr(),
                sin_table.as_ptr(),
                proj_buf_floats,
                attn_scratch_floats,
                fp8_scales_ptr,
                kv_fp8_ptr,
                batch_size,
                batch_descs_ptr,
                int4_ptr,
            )
        },
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "phi4::persistent_decode: CUDA backend not yet wired for Phi-4".into(),
            ));
        }
    };
    if status != 0 {
        return Err(phi4_backend_error(backend, "phi4 persistent decode kernel", status));
    }
    Ok(())
}
