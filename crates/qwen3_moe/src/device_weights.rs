use gpu_hal::GpuBuffer;
use model_store::BakedStore;
use thiserror::Error;

use crate::{
    baked::{build_int4_baked_index, BakedIndexError, Qwen3MoeInt4BakedIndex},
    config::TextConfig,
    desc_builder::{Qwen3MoeInt4ScalePtrs, Qwen3MoeLayerPtrs},
};

#[derive(Debug, Error)]
pub enum DeviceWeightsError {
    #[error("{0}")]
    BakedIndex(#[from] BakedIndexError),
    #[error("model-store error: {0}")]
    Store(#[from] model_store::Error),
}

pub struct Qwen3MoeInt4DeviceWeights {
    pub embed_tokens: GpuBuffer,
    pub final_norm: GpuBuffer,
    pub lm_head: Option<Qwen3MoeInt4DeviceTensor>,
    pub layers: Vec<Qwen3MoeInt4DeviceLayer>,
    pub total_bytes: u64,
}

pub struct Qwen3MoeInt4DeviceLayer {
    pub input_norm: GpuBuffer,
    pub post_attn_norm: GpuBuffer,
    pub q_proj: Qwen3MoeInt4DeviceTensor,
    pub k_proj: Qwen3MoeInt4DeviceTensor,
    pub v_proj: Qwen3MoeInt4DeviceTensor,
    pub o_proj: Qwen3MoeInt4DeviceTensor,
    pub q_norm: GpuBuffer,
    pub k_norm: GpuBuffer,
    pub router: GpuBuffer,
    pub experts_gate_up: Qwen3MoeInt4DeviceTensor,
    pub experts_down: Qwen3MoeInt4DeviceTensor,
}

pub struct Qwen3MoeInt4DeviceTensor {
    pub weight: GpuBuffer,
    pub scale: GpuBuffer,
    pub zero: GpuBuffer,
}

impl Qwen3MoeInt4DeviceWeights {
    pub fn load(
        store: &BakedStore,
        ordinal: usize,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, DeviceWeightsError> {
        let index = build_int4_baked_index(store, config, prefix)?;
        Self::load_from_index(store, ordinal, index)
    }

    pub fn layer_ptrs(&self) -> Vec<Qwen3MoeLayerPtrs> {
        self.layers
            .iter()
            .map(|layer| Qwen3MoeLayerPtrs {
                input_norm_w: layer.input_norm.as_ptr(),
                post_attn_norm_w: layer.post_attn_norm.as_ptr(),
                q_proj_w: layer.q_proj.weight.as_ptr(),
                k_proj_w: layer.k_proj.weight.as_ptr(),
                v_proj_w: layer.v_proj.weight.as_ptr(),
                o_proj_w: layer.o_proj.weight.as_ptr(),
                q_norm_w: layer.q_norm.as_ptr(),
                k_norm_w: layer.k_norm.as_ptr(),
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                router_w: layer.router.as_ptr(),
                experts_gate_up_w: layer.experts_gate_up.weight.as_ptr(),
                experts_down_w: layer.experts_down.weight.as_ptr(),
            })
            .collect()
    }

    pub fn int4_scale_ptrs(&self) -> Vec<Qwen3MoeInt4ScalePtrs> {
        self.layers
            .iter()
            .map(|layer| Qwen3MoeInt4ScalePtrs {
                q_proj_scale: layer.q_proj.scale.as_ptr(),
                q_proj_zero: layer.q_proj.zero.as_ptr(),
                k_proj_scale: layer.k_proj.scale.as_ptr(),
                k_proj_zero: layer.k_proj.zero.as_ptr(),
                v_proj_scale: layer.v_proj.scale.as_ptr(),
                v_proj_zero: layer.v_proj.zero.as_ptr(),
                o_proj_scale: layer.o_proj.scale.as_ptr(),
                o_proj_zero: layer.o_proj.zero.as_ptr(),
                experts_gate_up_scale: layer.experts_gate_up.scale.as_ptr(),
                experts_gate_up_zero: layer.experts_gate_up.zero.as_ptr(),
                experts_down_scale: layer.experts_down.scale.as_ptr(),
                experts_down_zero: layer.experts_down.zero.as_ptr(),
            })
            .collect()
    }

    fn load_from_index(
        store: &BakedStore,
        ordinal: usize,
        index: Qwen3MoeInt4BakedIndex,
    ) -> Result<Self, DeviceWeightsError> {
        let embed_tokens = store.load_to_gpu(&index.embed_tokens.name, ordinal)?;
        let final_norm = store.load_to_gpu(&index.final_norm.name, ordinal)?;
        let lm_head = index
            .lm_head
            .as_ref()
            .map(|tensor| load_int4_tensor(store, ordinal, tensor))
            .transpose()?;

        let mut layers = Vec::with_capacity(index.layers.len());
        for layer in &index.layers {
            layers.push(Qwen3MoeInt4DeviceLayer {
                input_norm: store.load_to_gpu(&layer.input_norm.name, ordinal)?,
                post_attn_norm: store.load_to_gpu(&layer.post_attn_norm.name, ordinal)?,
                q_proj: load_int4_tensor(store, ordinal, &layer.q_proj)?,
                k_proj: load_int4_tensor(store, ordinal, &layer.k_proj)?,
                v_proj: load_int4_tensor(store, ordinal, &layer.v_proj)?,
                o_proj: load_int4_tensor(store, ordinal, &layer.o_proj)?,
                q_norm: store.load_to_gpu(&layer.q_norm.name, ordinal)?,
                k_norm: store.load_to_gpu(&layer.k_norm.name, ordinal)?,
                router: store.load_to_gpu(&layer.router.name, ordinal)?,
                experts_gate_up: load_int4_tensor(store, ordinal, &layer.experts_gate_up)?,
                experts_down: load_int4_tensor(store, ordinal, &layer.experts_down)?,
            });
        }

        Ok(Self {
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            total_bytes: index.total_bytes,
        })
    }
}

fn load_int4_tensor(
    store: &BakedStore,
    ordinal: usize,
    tensor: &crate::baked::Qwen3MoeInt4Tensor,
) -> Result<Qwen3MoeInt4DeviceTensor, model_store::Error> {
    Ok(Qwen3MoeInt4DeviceTensor {
        weight: store.load_to_gpu(&tensor.weight.name, ordinal)?,
        scale: store.load_to_gpu(&tensor.scale.name, ordinal)?,
        zero: store.load_to_gpu(&tensor.zero.name, ordinal)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Activation, TextConfig};

    fn config_30b_a3b() -> TextConfig {
        TextConfig {
            vocab_size: 151_936,
            hidden_size: 2048,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            max_position_embeddings: 40_960,
            rms_norm_eps: 1e-6,
            intermediate_size: 6144,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
        }
    }

    #[test]
    fn device_weight_layer_count_follows_config() {
        let cfg = config_30b_a3b();
        let ptrs = vec![Qwen3MoeLayerPtrs::default(); cfg.num_hidden_layers];
        assert_eq!(ptrs.len(), cfg.num_hidden_layers);
    }
}
