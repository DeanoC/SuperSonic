use std::ffi::c_void;

use kernel_ffi::qwen3_moe::{Qwen3MoeDecodeLayerDesc, Qwen3MoeInt4ScaleDesc};
use thiserror::Error;

use crate::config::TextConfig;

#[derive(Debug, Error)]
pub enum DescBuildError {
    #[error("expected {expected} layer pointer records, got {got}")]
    LayerCount { expected: usize, got: usize },
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3MoeLayerPtrs {
    pub input_norm_w: *const c_void,
    pub post_attn_norm_w: *const c_void,
    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub router_w: *const c_void,
    pub experts_gate_up_w: *const c_void,
    pub experts_down_w: *const c_void,
}

impl Default for Qwen3MoeLayerPtrs {
    fn default() -> Self {
        Self {
            input_norm_w: std::ptr::null(),
            post_attn_norm_w: std::ptr::null(),
            q_proj_w: std::ptr::null(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
            q_norm_w: std::ptr::null(),
            k_norm_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            router_w: std::ptr::null(),
            experts_gate_up_w: std::ptr::null(),
            experts_down_w: std::ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3MoeInt4ScalePtrs {
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
}

impl Default for Qwen3MoeInt4ScalePtrs {
    fn default() -> Self {
        Self {
            q_proj_scale: std::ptr::null(),
            q_proj_zero: std::ptr::null(),
            k_proj_scale: std::ptr::null(),
            k_proj_zero: std::ptr::null(),
            v_proj_scale: std::ptr::null(),
            v_proj_zero: std::ptr::null(),
            o_proj_scale: std::ptr::null(),
            o_proj_zero: std::ptr::null(),
            experts_gate_up_scale: std::ptr::null(),
            experts_gate_up_zero: std::ptr::null(),
            experts_down_scale: std::ptr::null(),
            experts_down_zero: std::ptr::null(),
        }
    }
}

pub fn build_layer_descs(
    config: &TextConfig,
    ptrs: &[Qwen3MoeLayerPtrs],
    kv_len: usize,
    kv_max_t: usize,
) -> Result<Vec<Qwen3MoeDecodeLayerDesc>, DescBuildError> {
    if ptrs.len() != config.num_hidden_layers {
        return Err(DescBuildError::LayerCount {
            expected: config.num_hidden_layers,
            got: ptrs.len(),
        });
    }

    let mut out = Vec::with_capacity(config.num_hidden_layers);
    for (layer_idx, p) in ptrs.iter().enumerate() {
        let mut d = Qwen3MoeDecodeLayerDesc::default();
        d.layer_idx = layer_idx as i32;
        d.input_norm_w = p.input_norm_w;
        d.input_norm_eps = config.rms_norm_eps as f32;
        d.post_attn_norm_w = p.post_attn_norm_w;
        d.post_attn_norm_eps = config.rms_norm_eps as f32;
        d.q_proj_w = p.q_proj_w;
        d.k_proj_w = p.k_proj_w;
        d.v_proj_w = p.v_proj_w;
        d.o_proj_w = p.o_proj_w;
        d.q_norm_w = p.q_norm_w;
        d.k_norm_w = p.k_norm_w;
        d.rope_theta = config.rope_theta as f32;
        d.head_dim = config.head_dim as i32;
        d.num_heads = config.num_attention_heads as i32;
        d.num_kv_heads = config.num_key_value_heads as i32;
        d.kv_cache_k = p.kv_cache_k;
        d.kv_cache_v = p.kv_cache_v;
        d.kv_len = kv_len as i32;
        d.kv_max_t = kv_max_t as i32;
        d.router_w = p.router_w;
        d.experts_gate_up_w = p.experts_gate_up_w;
        d.experts_down_w = p.experts_down_w;
        d.num_experts = config.num_experts as i32;
        d.top_k = config.top_k() as i32;
        d.moe_intermediate_size = config.moe_intermediate_size as i32;
        d.norm_topk_prob = i32::from(config.norm_topk_prob);
        out.push(d);
    }
    Ok(out)
}

pub fn build_int4_scale_descs(
    ptrs: &[Qwen3MoeInt4ScalePtrs],
    group_size: usize,
) -> Vec<Qwen3MoeInt4ScaleDesc> {
    ptrs.iter()
        .map(|p| {
            let mut d = Qwen3MoeInt4ScaleDesc::default();
            d.q_proj_scale = p.q_proj_scale;
            d.q_proj_zero = p.q_proj_zero;
            d.k_proj_scale = p.k_proj_scale;
            d.k_proj_zero = p.k_proj_zero;
            d.v_proj_scale = p.v_proj_scale;
            d.v_proj_zero = p.v_proj_zero;
            d.o_proj_scale = p.o_proj_scale;
            d.o_proj_zero = p.o_proj_zero;
            d.experts_gate_up_scale = p.experts_gate_up_scale;
            d.experts_gate_up_zero = p.experts_gate_up_zero;
            d.experts_down_scale = p.experts_down_scale;
            d.experts_down_zero = p.experts_down_zero;
            d.group_size = group_size as i32;
            d
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn qwen3_30b_text_config() -> TextConfig {
        let json = r#"{
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "norm_topk_prob": true
        }"#;
        serde_json::from_str::<Config>(json).unwrap().text_config
    }

    #[test]
    fn qwen3_desc_builder_sets_full_attention_moe_geometry() {
        let cfg = qwen3_30b_text_config();
        let ptrs = vec![Qwen3MoeLayerPtrs::default(); cfg.num_hidden_layers];
        let descs = build_layer_descs(&cfg, &ptrs, 7, 256).unwrap();
        assert_eq!(descs.len(), 48);
        assert_eq!(descs[12].layer_idx, 12);
        assert_eq!(descs[12].head_dim, 128);
        assert_eq!(descs[12].num_heads, 32);
        assert_eq!(descs[12].num_kv_heads, 4);
        assert_eq!(descs[12].kv_len, 7);
        assert_eq!(descs[12].kv_max_t, 256);
        assert_eq!(descs[12].num_experts, 128);
        assert_eq!(descs[12].top_k, 8);
        assert_eq!(descs[12].moe_intermediate_size, 768);
        assert_eq!(descs[12].norm_topk_prob, 1);
    }

    #[test]
    fn qwen3_desc_builder_rejects_wrong_layer_count() {
        let cfg = qwen3_30b_text_config();
        let err = build_layer_descs(&cfg, &[], 0, 0).unwrap_err();
        assert!(matches!(
            err,
            DescBuildError::LayerCount {
                expected: 48,
                got: 0
            }
        ));
    }

    #[test]
    fn qwen3_int4_desc_builder_sets_group_size() {
        let ptrs = vec![Qwen3MoeInt4ScalePtrs::default(); 48];
        let descs = build_int4_scale_descs(&ptrs, 128);
        assert_eq!(descs.len(), 48);
        assert_eq!(descs[0].group_size, 128);
    }
}
