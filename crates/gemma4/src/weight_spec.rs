//! Expected-tensor spec for Gemma 4 dense language models.
//!
//! Given a `TextConfig`, `layer_tensors()` / `global_tensors()` enumerate every
//! tensor we expect to find in `model.safetensors` under the language-model
//! prefix, with its BF16 shape.
//!
//! Findings confirmed against the real `google/gemma-4-E2B` checkpoint:
//!   * All 35 layers own their `k_proj` / `v_proj` weights — "shared KV" is a
//!     runtime KV-cache optimization, not a weight-storage one.
//!   * `use_double_wide_mlp: true` doubles the MLP intermediate size for the
//!     last `num_kv_shared_layers` layers (E2B: 0-14 → 6144, 15-34 → 12288).
//!   * Sliding and full attention use different `head_dim` (256 vs 512 on E2B),
//!     and the accompanying `q_norm` / `k_norm` norms are sized per-kind.
//!   * `embed_tokens_per_layer.weight` is a single [vocab, num_layers * ple_dim]
//!     table, not per-layer — sliced at compute time.

use crate::config::TextConfig;

/// A single expected tensor: path relative to the weight prefix, and BF16 shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Tensor name with `{prefix}.` already prepended.
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    fn new(name: String, shape: Vec<usize>) -> Self {
        Self { name, shape }
    }
}

/// Expected global (non-layer) tensors.
pub fn global_tensors(cfg: &TextConfig, prefix: &str) -> Vec<TensorSpec> {
    let ple_table_dim = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
    vec![
        TensorSpec::new(
            format!("{prefix}.embed_tokens.weight"),
            vec![cfg.vocab_size, cfg.hidden_size],
        ),
        TensorSpec::new(
            format!("{prefix}.embed_tokens_per_layer.weight"),
            vec![cfg.vocab_size, ple_table_dim],
        ),
        TensorSpec::new(format!("{prefix}.norm.weight"), vec![cfg.hidden_size]),
        TensorSpec::new(
            format!("{prefix}.per_layer_model_projection.weight"),
            vec![ple_table_dim, cfg.hidden_size],
        ),
        TensorSpec::new(
            format!("{prefix}.per_layer_projection_norm.weight"),
            vec![cfg.hidden_size_per_layer_input],
        ),
    ]
}

/// Whether layer `L` uses the double-wide MLP. When `use_double_wide_mlp` is
/// true, layers in the shared-KV tail get 2× intermediate size; front layers
/// stay at the config's `intermediate_size`. When false, every layer uses
/// `intermediate_size`.
pub fn mlp_intermediate(cfg: &TextConfig, layer_idx: usize) -> usize {
    let shared_start = cfg.num_kv_owning_layers();
    if cfg.use_double_wide_mlp && layer_idx >= shared_start {
        cfg.intermediate_size * 2
    } else {
        cfg.intermediate_size
    }
}

/// Expected per-layer tensors for a single decoder layer.
pub fn layer_tensors(cfg: &TextConfig, prefix: &str, layer_idx: usize) -> Vec<TensorSpec> {
    let kind = cfg
        .attn_kind(layer_idx)
        .expect("caller must validate layer_types before calling layer_tensors");
    let head_dim = cfg.head_dim_for(kind);
    let q_dim = cfg.num_attention_heads * head_dim;
    let kv_dim = cfg.num_key_value_heads * head_dim;
    let imm = mlp_intermediate(cfg, layer_idx);
    let h = cfg.hidden_size;
    let ple = cfg.hidden_size_per_layer_input;
    let p = format!("{prefix}.layers.{layer_idx}");

    vec![
        TensorSpec::new(format!("{p}.input_layernorm.weight"), vec![h]),
        TensorSpec::new(format!("{p}.layer_scalar"), vec![1]),
        TensorSpec::new(format!("{p}.mlp.down_proj.weight"), vec![h, imm]),
        TensorSpec::new(format!("{p}.mlp.gate_proj.weight"), vec![imm, h]),
        TensorSpec::new(format!("{p}.mlp.up_proj.weight"), vec![imm, h]),
        TensorSpec::new(format!("{p}.per_layer_input_gate.weight"), vec![ple, h]),
        TensorSpec::new(format!("{p}.per_layer_projection.weight"), vec![h, ple]),
        TensorSpec::new(format!("{p}.post_attention_layernorm.weight"), vec![h]),
        TensorSpec::new(format!("{p}.post_feedforward_layernorm.weight"), vec![h]),
        TensorSpec::new(format!("{p}.post_per_layer_input_norm.weight"), vec![h]),
        TensorSpec::new(format!("{p}.pre_feedforward_layernorm.weight"), vec![h]),
        TensorSpec::new(format!("{p}.self_attn.k_norm.weight"), vec![head_dim]),
        TensorSpec::new(format!("{p}.self_attn.k_proj.weight"), vec![kv_dim, h]),
        TensorSpec::new(format!("{p}.self_attn.o_proj.weight"), vec![h, q_dim]),
        TensorSpec::new(format!("{p}.self_attn.q_norm.weight"), vec![head_dim]),
        TensorSpec::new(format!("{p}.self_attn.q_proj.weight"), vec![q_dim, h]),
        TensorSpec::new(format!("{p}.self_attn.v_proj.weight"), vec![kv_dim, h]),
    ]
}

/// Every expected language-model tensor: globals + all per-layer tensors.
pub fn all_tensors(cfg: &TextConfig, prefix: &str) -> Vec<TensorSpec> {
    let mut out = global_tensors(cfg, prefix);
    for i in 0..cfg.num_hidden_layers {
        out.extend(layer_tensors(cfg, prefix, i));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    const E2B_SHAPE: &str = include_str!("../tests/e2b_config.json");

    fn e2b_config() -> Config {
        serde_json::from_str(E2B_SHAPE).expect("parse e2b_config.json")
    }

    #[test]
    fn e2b_global_shapes() {
        let cfg = e2b_config();
        let t = &cfg.text_config;
        let globals = global_tensors(t, "model.language_model");
        // embed_tokens.weight [262144, 1536]
        assert_eq!(globals[0].shape, vec![262144, 1536]);
        // embed_tokens_per_layer.weight [262144, 35*256=8960]
        assert_eq!(globals[1].shape, vec![262144, 8960]);
        // norm.weight [1536]
        assert_eq!(globals[2].shape, vec![1536]);
        // per_layer_model_projection.weight [8960, 1536]
        assert_eq!(globals[3].shape, vec![8960, 1536]);
        // per_layer_projection_norm.weight [256]
        assert_eq!(globals[4].shape, vec![256]);
    }

    #[test]
    fn e2b_sliding_layer_0() {
        let cfg = e2b_config();
        let t = &cfg.text_config;
        let ts = layer_tensors(t, "model.language_model", 0);
        let by_name: std::collections::HashMap<_, _> = ts
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        assert_eq!(
            by_name["model.language_model.layers.0.self_attn.q_proj.weight"],
            vec![2048, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.0.self_attn.k_proj.weight"],
            vec![256, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.0.self_attn.v_proj.weight"],
            vec![256, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.0.self_attn.o_proj.weight"],
            vec![1536, 2048]
        );
        assert_eq!(
            by_name["model.language_model.layers.0.self_attn.k_norm.weight"],
            vec![256]
        );
        assert_eq!(
            by_name["model.language_model.layers.0.mlp.gate_proj.weight"],
            vec![6144, 1536]
        );
    }

    #[test]
    fn e2b_full_layer_4_uses_global_head_dim() {
        let cfg = e2b_config();
        let t = &cfg.text_config;
        let ts = layer_tensors(t, "model.language_model", 4);
        let by_name: std::collections::HashMap<_, _> = ts
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        assert_eq!(
            by_name["model.language_model.layers.4.self_attn.q_proj.weight"],
            vec![4096, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.4.self_attn.k_proj.weight"],
            vec![512, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.4.self_attn.v_proj.weight"],
            vec![512, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.4.self_attn.o_proj.weight"],
            vec![1536, 4096]
        );
        assert_eq!(
            by_name["model.language_model.layers.4.self_attn.k_norm.weight"],
            vec![512]
        );
    }

    #[test]
    fn e2b_shared_kv_layer_has_double_mlp() {
        let cfg = e2b_config();
        let t = &cfg.text_config;
        // Layer 15 is the first shared-KV layer (35 - 20 = 15).
        let ts = layer_tensors(t, "model.language_model", 15);
        let by_name: std::collections::HashMap<_, _> = ts
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        assert_eq!(
            by_name["model.language_model.layers.15.mlp.gate_proj.weight"],
            vec![12288, 1536]
        );
        assert_eq!(
            by_name["model.language_model.layers.15.mlp.down_proj.weight"],
            vec![1536, 12288]
        );
    }

    #[test]
    fn e2b_total_tensor_count() {
        let cfg = e2b_config();
        let t = &cfg.text_config;
        let total = all_tensors(t, "model.language_model").len();
        // 5 global + 17 per layer * 35 layers = 600
        assert_eq!(total, 600);
    }
}
