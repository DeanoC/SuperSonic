//! FFI bindings for the DFlash draft's bidirectional attention helper.
//!
//! Everything else the draft needs (RMSNorm, matmul, RoPE, SwiGLU) is
//! reused from the existing Qwen3.5 primitives exposed by `prefill_ffi`.
//! This file houses only what's genuinely new for DFlash M2.

use std::ffi::{c_int, c_void};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

#[cfg(supersonic_backend_hip)]
unsafe extern "C" {
    fn dflash_draft_hip_bidir_attention(
        device_ordinal: c_int,
        q_len: c_int,
        seq_len: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
    ) -> c_int;
}

fn backend_error(backend: Backend, what: &str, status: c_int) -> GpuError {
    match backend {
        Backend::Hip => GpuError::backend(Backend::Hip, format!("{what} failed with status {status}")),
        Backend::Cuda => GpuError::backend(Backend::Cuda, format!("{what} failed with status {status}")),
        Backend::Metal => GpuError::backend(Backend::Metal, format!("{what} failed with status {status}")),
    }
}

/// Bidirectional attention for DFlash draft layers.
///
/// Shape contract (all BF16, SHD layout):
///   Q:   [q_len, num_q_heads, head_dim]
///   K:   [seq_len, num_kv_heads, head_dim]   seq_len = past + ctx + q_len
///   V:   [seq_len, num_kv_heads, head_dim]
///   out: [q_len, num_q_heads, head_dim]
///
/// `scale` is typically `1.0 / sqrt(head_dim)`. The kernel is fully
/// bidirectional (no causal mask, no sliding window).
pub fn bidir_attention(
    ordinal: usize,
    dtype: ScalarType,
    q_len: usize,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "dflash::bidir_attention: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = out.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                dflash_draft_hip_bidir_attention(
                    ordinal as c_int,
                    q_len as c_int,
                    seq_len as c_int,
                    num_q_heads as c_int,
                    num_kv_heads as c_int,
                    head_dim as c_int,
                    scale,
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "dflash::bidir_attention: CUDA backend not wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "dflash::bidir_attention: Metal backend not wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(backend_error(
            backend,
            "dflash_draft bidir attention",
            status,
        ));
    }
    Ok(())
}
