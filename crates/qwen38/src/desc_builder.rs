use kernel_ffi::{
    BatchSeqDesc, DecodeLayerDesc, FP8ScaleDesc, INT4ScaleDesc, KVCacheFp8Desc, MAX_BATCH_SIZE,
};

use crate::state::ModelState;
use crate::weights::{infer_lowbit_type, LayerKind, Qwen38Weights};

/// Build the array of layer descriptors for the persistent decode kernel.
/// Must be called each decode step (kv_len changes).
pub fn build_layer_descs(
    weights: &Qwen38Weights,
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
        d.rms_norm_add_unit_offset = if config.rms_norm_add_unit_offset {
            1
        } else {
            0
        };
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
                d.q_norm_w = fa
                    .q_norm_w
                    .as_ref()
                    .map(|w| w.as_ptr())
                    .unwrap_or(std::ptr::null());
                d.k_norm_w = fa
                    .k_norm_w
                    .as_ref()
                    .map(|w| w.as_ptr())
                    .unwrap_or(std::ptr::null());
                d.q_norm_eps = if fa.q_norm_w.is_some() {
                    config.rms_norm_eps as f32
                } else {
                    0.0
                };
                d.k_norm_eps = if fa.k_norm_w.is_some() {
                    config.rms_norm_eps as f32
                } else {
                    0.0
                };
                d.kv_len = seqlen_offset as i32;
                if let Some(ptr) = ls.kv_cache_k_ptr() {
                    d.kv_cache_k = ptr;
                    d.kv_max_t = ls.kv_capacity() as i32;
                }
                if let Some(ptr) = ls.kv_cache_v_ptr() {
                    d.kv_cache_v = ptr;
                }
                if let Some(ref shadow_k) = ls.kv_shadow_k {
                    d.kv_shadow_k = shadow_k.as_ptr() as *mut _;
                }
                if let Some(ref shadow_v) = ls.kv_shadow_v {
                    d.kv_shadow_v = shadow_v.as_ptr() as *mut _;
                }
                d.kv_shadow_start = if ls.kv_shadow_start == usize::MAX {
                    -1
                } else {
                    ls.kv_shadow_start as i32
                };
            }
        }

        descs.push(d);
    }
    descs
}

/// Build FP8 scale descriptors (parallel to layer descs) for runtime FP8 dequant.
/// Returns None if weights are not FP8.
pub fn build_fp8_scale_descs(weights: &Qwen38Weights) -> Option<Vec<FP8ScaleDesc>> {
    if !weights.is_fp8 {
        return None;
    }

    let scale_ptr = |opt: &Option<gpu_hal::GpuBuffer>| -> *const std::ffi::c_void {
        opt.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null())
    };

    let mut descs = Vec::with_capacity(weights.layers.len());
    for lw in &weights.layers {
        let mut d = FP8ScaleDesc::default();
        d.block_size = weights.fp8_block_size as i32;

        // Common MLP scales
        d.gate_proj_scale = scale_ptr(&lw.gate_proj_scale);
        d.up_proj_scale = scale_ptr(&lw.up_proj_scale);
        d.down_proj_scale = scale_ptr(&lw.down_proj_scale);

        match lw.kind {
            LayerKind::Linear => {
                let lin = lw.linear.as_ref().unwrap();
                d.qkv_proj_scale = scale_ptr(&lin.qkv_proj_scale);
                d.z_proj_scale = scale_ptr(&lin.z_proj_scale);
                d.b_proj_scale = scale_ptr(&lin.b_proj_scale);
                d.a_proj_scale = scale_ptr(&lin.a_proj_scale);
                d.linear_out_proj_scale = scale_ptr(&lin.out_proj_scale);
            }
            LayerKind::Full => {
                let fa = lw.full.as_ref().unwrap();
                d.q_proj_scale = scale_ptr(&fa.q_proj_scale);
                d.k_proj_scale = scale_ptr(&fa.k_proj_scale);
                d.v_proj_scale = scale_ptr(&fa.v_proj_scale);
                d.o_proj_scale = scale_ptr(&fa.o_proj_scale);
            }
        }

        descs.push(d);
    }
    Some(descs)
}

/// Build INT4 / ggml-K / GQH scale descriptors (parallel to layer descs).
///
/// The kernel-facing descriptor keeps its historical INT4 name for ABI
/// stability. For Qwen3.8 GQH tensors, each non-null `*_proj_scale` slot is a
/// pointer to GQH `{tensor_scale, grid_code}` metadata; headerless GQH2_C has
/// no such sidecar, and these pointers are not legacy GPTQ scales.
/// Returns None if the megakernel has no low-bit weights to dequant.
pub fn build_int4_scale_descs(weights: &Qwen38Weights) -> Option<Vec<INT4ScaleDesc>> {
    if !weights.is_int4 && weights.gqh_headers.is_empty() {
        return None;
    }

    let ptr = |opt: &Option<gpu_hal::GpuBuffer>| -> *const std::ffi::c_void {
        opt.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null())
    };
    let scale_or_gqh_sidecar =
        |role: &str, int4: &Option<gpu_hal::GpuBuffer>| -> *const std::ffi::c_void {
            weights
                .gqh_sidecars
                .get(role)
                .map(|buf| buf.as_ptr())
                .unwrap_or_else(|| ptr(int4))
        };

    let mut descs = Vec::with_capacity(weights.layers.len());
    for (idx, lw) in weights.layers.iter().enumerate() {
        let mut d = INT4ScaleDesc::default();
        // GQH GGUF has no native INT4 group size; keep this non-zero so the
        // megakernel's INT4-oriented `row / group_size` prep is defined.
        d.group_size = if weights.int4_group_size == 0 {
            128
        } else {
            weights.int4_group_size as i32
        };

        let gate_role = format!("layers.{idx}.mlp.gate_proj");
        let up_role = format!("layers.{idx}.mlp.up_proj");
        let down_role = format!("layers.{idx}.mlp.down_proj");

        // Common MLP scales/zeros
        d.gate_proj_scale = scale_or_gqh_sidecar(&gate_role, &lw.gate_proj_int4_scale);
        d.gate_proj_zero = ptr(&lw.gate_proj_int4_zero);
        d.gate_proj_awq_inv_scale = ptr(&lw.gate_proj_awq_inv_scale);
        d.up_proj_scale = scale_or_gqh_sidecar(&up_role, &lw.up_proj_int4_scale);
        d.up_proj_zero = ptr(&lw.up_proj_int4_zero);
        d.up_proj_awq_inv_scale = ptr(&lw.up_proj_awq_inv_scale);
        d.down_proj_scale = scale_or_gqh_sidecar(&down_role, &lw.down_proj_int4_scale);
        d.down_proj_zero = ptr(&lw.down_proj_int4_zero);
        d.down_proj_awq_inv_scale = ptr(&lw.down_proj_awq_inv_scale);
        d.gate_proj_type = infer_lowbit_type(
            &lw.gate_proj_w,
            weights.config.hidden_size,
            lw.gate_proj_int4_scale.is_some(),
        );
        d.up_proj_type = infer_lowbit_type(
            &lw.up_proj_w,
            weights.config.hidden_size,
            lw.up_proj_int4_scale.is_some(),
        );
        d.down_proj_type = infer_lowbit_type(
            &lw.down_proj_w,
            weights.config.intermediate_size,
            lw.down_proj_int4_scale.is_some(),
        );

        match lw.kind {
            LayerKind::Linear => {
                let lin = lw.linear.as_ref().unwrap();
                d.qkv_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.linear_attn.in_proj_qkv"),
                    &lin.qkv_proj_int4_scale,
                );
                d.qkv_proj_zero = ptr(&lin.qkv_proj_int4_zero);
                d.qkv_proj_awq_inv_scale = ptr(&lin.qkv_proj_awq_inv_scale);
                d.z_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.linear_attn.in_proj_z"),
                    &lin.z_proj_int4_scale,
                );
                d.z_proj_zero = ptr(&lin.z_proj_int4_zero);
                d.z_proj_awq_inv_scale = ptr(&lin.z_proj_awq_inv_scale);
                d.linear_out_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.linear_attn.out_proj"),
                    &lin.out_proj_int4_scale,
                );
                d.linear_out_proj_zero = ptr(&lin.out_proj_int4_zero);
                d.linear_out_proj_awq_inv_scale = ptr(&lin.out_proj_awq_inv_scale);
                d.qkv_proj_type = infer_lowbit_type(
                    &lin.qkv_proj_w,
                    weights.config.hidden_size,
                    lin.qkv_proj_int4_scale.is_some(),
                );
                d.z_proj_type = infer_lowbit_type(
                    &lin.z_proj_w,
                    weights.config.hidden_size,
                    lin.z_proj_int4_scale.is_some(),
                );
                d.linear_out_proj_type = infer_lowbit_type(
                    &lin.out_proj_w,
                    weights.config.linear_num_value_heads * weights.config.linear_value_head_dim,
                    lin.out_proj_int4_scale.is_some(),
                );
                d.a_proj_type = infer_lowbit_type(&lin.a_proj_w, weights.config.hidden_size, false);
                d.b_proj_type = infer_lowbit_type(&lin.b_proj_w, weights.config.hidden_size, false);
            }
            LayerKind::Full => {
                let fa = lw.full.as_ref().unwrap();
                d.q_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.self_attn.q_proj"),
                    &fa.q_proj_int4_scale,
                );
                d.q_proj_zero = ptr(&fa.q_proj_int4_zero);
                d.q_proj_awq_inv_scale = ptr(&fa.q_proj_awq_inv_scale);
                d.k_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.self_attn.k_proj"),
                    &fa.k_proj_int4_scale,
                );
                d.k_proj_zero = ptr(&fa.k_proj_int4_zero);
                d.k_proj_awq_inv_scale = ptr(&fa.k_proj_awq_inv_scale);
                d.v_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.self_attn.v_proj"),
                    &fa.v_proj_int4_scale,
                );
                d.v_proj_zero = ptr(&fa.v_proj_int4_zero);
                d.v_proj_awq_inv_scale = ptr(&fa.v_proj_awq_inv_scale);
                d.o_proj_scale = scale_or_gqh_sidecar(
                    &format!("layers.{idx}.self_attn.o_proj"),
                    &fa.o_proj_int4_scale,
                );
                d.o_proj_zero = ptr(&fa.o_proj_int4_zero);
                d.o_proj_awq_inv_scale = ptr(&fa.o_proj_awq_inv_scale);
                d.q_proj_type = infer_lowbit_type(
                    &fa.q_proj_w,
                    weights.config.hidden_size,
                    fa.q_proj_int4_scale.is_some(),
                );
                d.k_proj_type = infer_lowbit_type(
                    &fa.k_proj_w,
                    weights.config.hidden_size,
                    fa.k_proj_int4_scale.is_some(),
                );
                d.v_proj_type = infer_lowbit_type(
                    &fa.v_proj_w,
                    weights.config.hidden_size,
                    fa.v_proj_int4_scale.is_some(),
                );
                d.o_proj_type = infer_lowbit_type(
                    &fa.o_proj_w,
                    weights.config.num_attention_heads * weights.config.head_dim,
                    fa.o_proj_int4_scale.is_some(),
                );
            }
        }

        descs.push(d);
    }
    Some(descs)
}

/// Build batch sequence descriptors for batched decode.
/// One BatchSeqDesc per layer, containing per-sequence state pointers for all batch items.
/// `states`: slice of ModelStates (one per batch item).
/// `seqlen_offsets`: per-sequence position offsets.
/// Returns None if batch_size <= 1.
pub fn build_batch_seq_descs(
    states: &[&ModelState],
    seqlen_offsets: &[usize],
    kv_fp8: bool,
) -> Option<Vec<BatchSeqDesc>> {
    let batch_size = states.len();
    if batch_size <= 1 {
        return None;
    }
    assert!(
        batch_size <= MAX_BATCH_SIZE,
        "batch_size {} exceeds MAX_BATCH_SIZE {}",
        batch_size,
        MAX_BATCH_SIZE
    );
    assert_eq!(states.len(), seqlen_offsets.len());

    let num_layers = states[0].layers.len();
    let mut descs = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let mut d = BatchSeqDesc::default();
        for b in 0..batch_size {
            d.seqlen_offset[b] = seqlen_offsets[b] as i32;
            let ls = &states[b].layers[layer_idx];
            match ls.kind {
                LayerKind::Full => {
                    if let Some(ref k) = ls.kv_cache_k {
                        d.kv_cache_k[b] = k.as_ptr() as *mut _;
                        d.kv_max_t[b] = k.shape()[2] as i32;
                    }
                    if let Some(ref v) = ls.kv_cache_v {
                        d.kv_cache_v[b] = v.as_ptr() as *mut _;
                    }
                    d.kv_len[b] = seqlen_offsets[b] as i32;
                    if kv_fp8 {
                        if let Some(ref sk) = ls.kv_scale_k {
                            d.kv_scale_k[b] = sk.as_ptr() as *mut _;
                        }
                        if let Some(ref sv) = ls.kv_scale_v {
                            d.kv_scale_v[b] = sv.as_ptr() as *mut _;
                        }
                        if let Some(ref shadow_k) = ls.kv_shadow_k {
                            d.kv_shadow_k[b] = shadow_k.as_ptr() as *mut _;
                        }
                        if let Some(ref shadow_v) = ls.kv_shadow_v {
                            d.kv_shadow_v[b] = shadow_v.as_ptr() as *mut _;
                        }
                        d.kv_shadow_start[b] = if ls.kv_shadow_start == usize::MAX {
                            -1
                        } else {
                            ls.kv_shadow_start as i32
                        };
                    }
                }
                LayerKind::Linear => {
                    if let Some(ref cs) = ls.conv_state {
                        d.conv_state[b] = cs.as_ptr() as *mut _;
                    }
                    if let Some(ref rs) = ls.recurrent_state {
                        d.recurrent_state[b] = rs.as_ptr() as *mut _;
                    }
                }
            }
        }
        descs.push(d);
    }
    Some(descs)
}

/// Build KV cache FP8 scale descriptors (parallel to layer descs).
/// Returns None if `kv_fp8` is false.
pub fn build_kv_fp8_descs(state: &ModelState, kv_fp8: bool) -> Option<Vec<KVCacheFp8Desc>> {
    if !kv_fp8 {
        return None;
    }

    let mut descs = Vec::with_capacity(state.layers.len());
    for ls in &state.layers {
        let mut d = KVCacheFp8Desc::default();
        if let Some(ref sk) = ls.kv_scale_k {
            d.kv_scale_k = sk.as_ptr() as *mut _;
        }
        if let Some(ref sv) = ls.kv_scale_v {
            d.kv_scale_v = sv.as_ptr() as *mut _;
        }
        descs.push(d);
    }
    Some(descs)
}
