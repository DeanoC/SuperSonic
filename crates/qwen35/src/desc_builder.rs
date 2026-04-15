use kernel_ffi::DecodeLayerDesc;

use crate::state::ModelState;
use crate::weights::{Qwen35Weights, LayerKind};

/// Build the array of layer descriptors for the persistent decode kernel.
/// Must be called each decode step (kv_len changes).
pub fn build_layer_descs(
    weights: &Qwen35Weights,
    state: &ModelState,
    seqlen_offset: usize,
) -> Vec<DecodeLayerDesc> {
    let config = &weights.config;
    let mut descs = Vec::with_capacity(config.num_hidden_layers);

    for (idx, lw) in weights.layers.iter().enumerate() {
        let ls = &state.layers[idx];
        let mut d = DecodeLayerDesc::default();

        // Common fields
        d.layer_type = match lw.kind {
            LayerKind::Linear => 0,
            LayerKind::Full => 1,
        };
        d.intermediate_size = config.intermediate_size as i32;
        d.input_norm_w = lw.input_norm_w.as_ptr();
        d.input_norm_eps = config.rms_norm_eps as f32;
        d.post_attn_norm_w = lw.post_attn_norm_w.as_ptr();
        d.post_attn_norm_eps = config.rms_norm_eps as f32;
        d.gate_proj_w = lw.gate_proj_w.as_ptr();
        d.up_proj_w = lw.up_proj_w.as_ptr();
        d.down_proj_w = lw.down_proj_w.as_ptr();

        match lw.kind {
            LayerKind::Linear => {
                let lin = lw.linear.as_ref().unwrap();
                d.qkv_proj_w = lin.qkv_proj_w.as_ptr();
                d.qkv_out_dim = lin.qkv_proj_w.shape()[0] as i32;
                d.z_proj_w = lin.z_proj_w.as_ptr();
                d.z_out_dim = lin.z_proj_w.shape()[0] as i32;
                d.b_proj_w = lin.b_proj_w.as_ptr();
                d.a_proj_w = lin.a_proj_w.as_ptr();
                d.conv1d_w = lin.conv1d_w.as_ptr();
                d.conv_kernel_size = config.linear_conv_kernel_dim as i32;
                d.linear_out_proj_w = lin.out_proj_w.as_ptr();
                d.linear_value_dim = config.linear_value_dim() as i32;
                d.linear_num_v_heads = config.linear_num_value_heads as i32;
                d.linear_num_k_heads = config.linear_num_key_heads as i32;
                d.linear_head_k_dim = config.linear_key_head_dim as i32;
                d.linear_head_v_dim = config.linear_value_head_dim as i32;
                d.dt_bias_w = lin.dt_bias.as_ptr();
                d.a_log_exp_w = lin.a_log_exp.as_ptr();
                d.linear_norm_w = lin.norm_w.as_ptr();
                d.linear_norm_eps = config.rms_norm_eps as f32;
                if let Some(ref cs) = ls.conv_state {
                    d.conv_state = cs.as_ptr() as *mut _;
                }
                if let Some(ref rs) = ls.recurrent_state {
                    d.recurrent_state = rs.as_ptr() as *mut _;
                }
            }
            LayerKind::Full => {
                let fa = lw.full.as_ref().unwrap();
                d.q_proj_w = fa.q_proj_w.as_ptr();
                d.q_out_dim = fa.q_proj_w.shape()[0] as i32;
                d.k_proj_w = fa.k_proj_w.as_ptr();
                d.k_out_dim = fa.k_proj_w.shape()[0] as i32;
                d.v_proj_w = fa.v_proj_w.as_ptr();
                d.o_proj_w = fa.o_proj_w.as_ptr();
                d.attn_head_dim = config.head_dim as i32;
                d.attn_num_heads = config.num_attention_heads as i32;
                d.attn_num_kv_heads = config.num_key_value_heads as i32;
                d.q_norm_w = fa.q_norm_w.as_ptr();
                d.k_norm_w = fa.k_norm_w.as_ptr();
                d.q_norm_eps = config.rms_norm_eps as f32;
                d.k_norm_eps = config.rms_norm_eps as f32;
                d.kv_len = seqlen_offset as i32;
                if let Some(ref k) = ls.kv_cache_k {
                    d.kv_cache_k = k.as_ptr() as *mut _;
                    d.kv_max_t = k.shape()[2] as i32;
                }
                if let Some(ref v) = ls.kv_cache_v {
                    d.kv_cache_v = v.as_ptr() as *mut _;
                }
            }
        }

        descs.push(d);
    }
    descs
}
