use crate::config::TextConfig;

pub struct StateAccount {
    pub kv_bytes: u64,
    pub moe_scratch_bytes: u64,
    pub activation_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct StateLayout {
    pub kv_dtype_bytes: u64,
    pub context_tokens: u64,
    pub batch_size: u64,
}

impl StateLayout {
    pub fn new(context_tokens: usize, batch_size: usize, kv_fp8: bool) -> Self {
        Self {
            kv_dtype_bytes: if kv_fp8 { 1 } else { 2 },
            context_tokens: context_tokens as u64,
            batch_size: batch_size as u64,
        }
    }
}

impl StateAccount {
    pub fn from_config(config: &TextConfig, layout: StateLayout) -> Self {
        let hidden = config.hidden_size as u64;
        let head_dim = config.head_dim as u64;
        let kv_heads = config.num_key_value_heads as u64;
        let layers = config.num_hidden_layers as u64;
        let kv_bytes = layers
            * 2
            * kv_heads
            * head_dim
            * layout.kv_dtype_bytes
            * layout.context_tokens
            * layout.batch_size;

        let top_k = config.num_experts_per_tok as u64;
        let num_experts = config.num_experts as u64;
        let router_logits = 4 * num_experts * layout.batch_size;
        let topk_indices = 4 * top_k * layout.batch_size;
        let topk_weights = 4 * top_k * layout.batch_size;
        let expert_acc = 2 * hidden * layout.batch_size;
        let moe_scratch_bytes = router_logits + topk_indices + topk_weights + expert_acc;
        let activation_bytes = 2 * hidden * layout.batch_size * 8;
        let total_bytes = kv_bytes + moe_scratch_bytes + activation_bytes;
        Self {
            kv_bytes,
            moe_scratch_bytes,
            activation_bytes,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;

    #[test]
    fn qwen3_state_account_counts_all_layers_as_full_attention() {
        let cfg = TextConfig {
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
        };
        let acct = StateAccount::from_config(&cfg, StateLayout::new(1024, 1, false));
        assert_eq!(acct.kv_bytes, 48 * 2 * 4 * 128 * 2 * 1024);
    }
}
