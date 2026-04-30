use crate::config::TextConfig;

/// VRAM accounting for the runtime-mutable state buffers (KV caches, linear
/// attention conv/recurrent state, MoE scratch). PR 3 only computes sizes;
/// the actual GPU allocation lands when the kernel is wired up.
pub struct StateAccount {
    pub full_kv_bytes: u64,
    pub linear_conv_state_bytes: u64,
    pub linear_recurrent_state_bytes: u64,
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
        let n_full = config.num_full_attention_layers() as u64;
        let n_linear = config.num_linear_attention_layers() as u64;

        let kv_bytes_per_layer_per_token = 2 * kv_heads * head_dim * layout.kv_dtype_bytes;
        let full_kv_bytes =
            n_full * kv_bytes_per_layer_per_token * layout.context_tokens * layout.batch_size;

        let lin_qkv_dim = (config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim) as u64;
        let conv_state_bytes_per_layer =
            2 * lin_qkv_dim * (config.linear_conv_kernel_dim as u64).saturating_sub(1);
        let recurrent_state_bytes_per_layer = 4
            * config.linear_num_value_heads as u64
            * config.linear_value_head_dim as u64
            * config.linear_key_head_dim as u64;
        let linear_conv_state_bytes = n_linear * conv_state_bytes_per_layer * layout.batch_size;
        let linear_recurrent_state_bytes =
            n_linear * recurrent_state_bytes_per_layer * layout.batch_size;

        let top_k = config.num_experts_per_tok as u64;
        let num_experts = config.num_experts as u64;
        let router_logits = 4 * num_experts * layout.batch_size;
        let topk_indices = 4 * top_k * layout.batch_size;
        let topk_weights = 4 * top_k * layout.batch_size;
        let work_units = 16 * (top_k + 1) * layout.batch_size;
        let expert_acc = 2 * hidden * layout.batch_size;
        let moe_scratch_bytes =
            router_logits + topk_indices + topk_weights + work_units + expert_acc;

        let activation_bytes = 2 * hidden * layout.batch_size * 8;

        let total_bytes = full_kv_bytes
            + linear_conv_state_bytes
            + linear_recurrent_state_bytes
            + moe_scratch_bytes
            + activation_bytes;
        Self {
            full_kv_bytes,
            linear_conv_state_bytes,
            linear_recurrent_state_bytes,
            moe_scratch_bytes,
            activation_bytes,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_35b_a3b() -> TextConfig {
        let layer_types: Vec<String> = (0..40)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();
        TextConfig {
            vocab_size: 248_320,
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            hidden_act: crate::config::Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 256,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            layer_types,
            rope_parameters: None,
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
    }

    #[test]
    fn kv_cache_sizing_at_4k_context_under_5gib() {
        let cfg = config_35b_a3b();
        let layout = StateLayout::new(4096, 1, false);
        let acct = StateAccount::from_config(&cfg, layout);
        // 10 full layers × 2 KV heads × 256 head_dim × 4096 ctx × 2 BF16 × 2 (K+V)
        let expected_kv = 10u64 * 2 * 256 * 4096 * 2 * 2;
        assert_eq!(acct.full_kv_bytes, expected_kv);
        let gib = 1024.0_f64.powi(3);
        assert!(acct.full_kv_bytes as f64 / gib < 1.0);
    }

    #[test]
    fn moe_scratch_is_well_under_a_megabyte() {
        let cfg = config_35b_a3b();
        let layout = StateLayout::new(4096, 1, false);
        let acct = StateAccount::from_config(&cfg, layout);
        // scratch is per-step, batch=1: ~router(1024) + topk(80) + work(144) +
        // expert acc (4096) ≈ tiny.
        assert!(acct.moe_scratch_bytes < 1 << 20);
    }

    #[test]
    fn fp8_kv_halves_full_kv_bytes() {
        let cfg = config_35b_a3b();
        let bf16 = StateAccount::from_config(&cfg, StateLayout::new(4096, 1, false)).full_kv_bytes;
        let fp8 = StateAccount::from_config(&cfg, StateLayout::new(4096, 1, true)).full_kv_bytes;
        assert_eq!(bf16, 2 * fp8);
    }
}
