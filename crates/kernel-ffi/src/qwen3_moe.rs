//! FFI bridge for the Qwen3-MoE HIP kernel family.
//!
//! This is deliberately separate from `qwen36_moe`: Qwen3-30B-A3B is a
//! pure-attention MoE model with different geometry and no Qwen3.6 hybrid
//! linear-attention/shared-expert path.

#![cfg_attr(not(supersonic_backend_hip), allow(unused_variables))]

use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen3MoeDecodeLayerDesc {
    pub layer_idx: c_int,
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,

    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub rope_theta: f32,
    pub head_dim: c_int,
    pub num_heads: c_int,
    pub num_kv_heads: c_int,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,

    pub router_w: *const c_void,
    pub experts_gate_up_w: *const c_void,
    pub experts_down_w: *const c_void,
    pub num_experts: c_int,
    pub top_k: c_int,
    pub moe_intermediate_size: c_int,
    pub norm_topk_prob: c_int,
}

unsafe impl Send for Qwen3MoeDecodeLayerDesc {}
unsafe impl Sync for Qwen3MoeDecodeLayerDesc {}

impl Default for Qwen3MoeDecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen3MoeInt4ScaleDesc {
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
    pub experts_gate_up_scale: *const c_void,
    pub experts_gate_up_zero: *const c_void,
    pub experts_down_scale: *const c_void,
    pub experts_down_zero: *const c_void,
    pub group_size: c_int,
}

unsafe impl Send for Qwen3MoeInt4ScaleDesc {}
unsafe impl Sync for Qwen3MoeInt4ScaleDesc {}

impl Default for Qwen3MoeInt4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn qwen3_moe_hip_stub_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        layers: *const Qwen3MoeDecodeLayerDesc,
        workspace: *mut f32,
        counters: *mut c_uint,
    ) -> c_int;

    pub fn qwen3_moe_hip_decode_layer_launch(
        dtype: c_int,
        device_ordinal: usize,
        layer: *const Qwen3MoeDecodeLayerDesc,
        int4: *const Qwen3MoeInt4ScaleDesc,
        hidden: c_int,
        position: c_int,
        input_hidden: *const c_void,
        output_hidden: *mut c_void,
        workspace: *mut f32,
    ) -> c_int;

    pub fn qwen3_moe_hip_lm_head_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;
}

pub fn stub_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer_descs_device: &GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    num_layers: usize,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::stub_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = layer_descs_device.backend();
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_stub_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    layer_descs_device.as_ptr() as *const Qwen3MoeDecodeLayerDesc,
                    workspace.as_mut_ptr() as *mut f32,
                    sync_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::stub_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::stub_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe stub launch failed with status {status}"),
        ));
    }
    Ok(())
}

pub fn decode_layer_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer: &Qwen3MoeDecodeLayerDesc,
    int4: &Qwen3MoeInt4ScaleDesc,
    hidden: i32,
    position: i32,
    input_hidden: &GpuBuffer,
    output_hidden: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = input_hidden.backend();
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_decode_layer_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    layer as *const Qwen3MoeDecodeLayerDesc,
                    int4 as *const Qwen3MoeInt4ScaleDesc,
                    hidden,
                    position,
                    input_hidden.as_ptr(),
                    output_hidden.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    layer,
                    int4,
                    hidden,
                    position,
                    output_hidden,
                    workspace,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::decode_layer_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::decode_layer_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe decode layer launch failed with status {status}"),
        ));
    }
    Ok(())
}

pub fn lm_head_launch(
    ordinal: usize,
    dtype: ScalarType,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    logits: &mut GpuBuffer,
    counter: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::lm_head_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = final_hidden.backend();
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_lm_head_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_hidden.as_ptr(),
                    final_norm_w.as_ptr(),
                    lm_head_w.as_ptr(),
                    logits.as_mut_ptr(),
                    counter.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_norm_w,
                    lm_head_w,
                    logits,
                    counter,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::lm_head_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::lm_head_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe lm_head launch failed with status {status}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_sizes_match_bridge_static_asserts() {
        assert_eq!(std::mem::size_of::<Qwen3MoeDecodeLayerDesc>(), 168);
        assert_eq!(std::mem::size_of::<Qwen3MoeInt4ScaleDesc>(), 104);
    }
}
