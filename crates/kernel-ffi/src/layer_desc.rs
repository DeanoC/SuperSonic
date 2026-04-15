use std::ffi::{c_int, c_void};

/// Rust mirror of `Qwen35DecodeLayerDesc` in full_attention.hip.
/// Describes one decoder layer for the persistent decode megakernel.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DecodeLayerDesc {
    // --- Common fields ---
    pub layer_type: c_int,          // 0 = linear_attention, 1 = full_attention
    pub intermediate_size: c_int,   // MLP intermediate dim
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,
    pub gate_proj_w: *const c_void,
    pub up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,

    // --- Linear attention (layer_type == 0) ---
    pub qkv_proj_w: *const c_void,
    pub qkv_out_dim: c_int,
    pub z_proj_w: *const c_void,
    pub z_out_dim: c_int,
    pub b_proj_w: *const c_void,
    pub a_proj_w: *const c_void,
    pub conv1d_w: *const c_void,
    pub conv_kernel_size: c_int,
    pub linear_out_proj_w: *const c_void,
    pub linear_value_dim: c_int,
    pub linear_num_v_heads: c_int,
    pub linear_head_k_dim: c_int,
    pub linear_head_v_dim: c_int,
    pub dt_bias_w: *const c_void,
    pub a_log_exp_w: *const c_void,
    pub linear_norm_w: *const c_void,
    pub linear_norm_eps: f32,
    pub conv_state: *mut c_void,
    pub recurrent_state: *mut c_void,

    // --- Full attention (layer_type == 1) ---
    pub q_proj_w: *const c_void,
    pub q_out_dim: c_int,
    pub k_proj_w: *const c_void,
    pub k_out_dim: c_int,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub attn_head_dim: c_int,
    pub attn_num_heads: c_int,
    pub attn_num_kv_heads: c_int,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub q_norm_eps: f32,
    pub k_norm_eps: f32,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,
}

unsafe impl Send for DecodeLayerDesc {}
unsafe impl Sync for DecodeLayerDesc {}

impl Default for DecodeLayerDesc {
    fn default() -> Self {
        // Safety: all-zeros is valid for this #[repr(C)] struct (null pointers, zero ints/floats).
        unsafe { std::mem::zeroed() }
    }
}
