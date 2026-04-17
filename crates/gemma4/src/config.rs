//! Gemma 4 config.json parser.
//!
//! Gemma 4's HF schema differs from Qwen3.5 in several ways that make a shared
//! parser awkward:
//!   - `rope_parameters` is a nested object keyed by attention type
//!     (`full_attention` vs `sliding_attention`) rather than a flat config.
//!   - Full-attention layers have a different head_dim (`global_head_dim`) than
//!     sliding layers (`head_dim`).
//!   - New fields with no Qwen3.5 analog: `sliding_window`, `num_kv_shared_layers`,
//!     `final_logit_softcapping`, `hidden_size_per_layer_input`,
//!     `use_double_wide_mlp`, `enable_moe_block`.
//!
//! Vision and audio configs are parsed as opaque `serde_json::Value` for now —
//! text-only is the initial target.

use serde::Deserialize;

fn default_rope_theta() -> f64 { 10_000.0 }
fn default_partial_rotary_factor() -> f64 { 1.0 }
fn default_rope_type() -> String { "default".to_string() }

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GemmaRope {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Fraction of head_dim that receives rotary embedding.
    /// Absent for sliding-attention blocks (full rotation); present for full-attention.
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for GemmaRope {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

/// Gemma 4 splits RoPE config by attention type. Both blocks may be present;
/// either may be absent if the model doesn't use that attention variant.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GemmaRopeParameters {
    #[serde(default)]
    pub full_attention: Option<GemmaRope>,
    #[serde(default)]
    pub sliding_attention: Option<GemmaRope>,
}

/// The per-layer attention shape. Sliding layers use `head_dim`; full layers
/// use `global_head_dim`. Keeping them as an explicit enum avoids silent dim
/// mismatches downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    Sliding,
    Full,
}

impl AttnKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sliding_attention" => Some(Self::Sliding),
            "full_attention" => Some(Self::Full),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// head_dim for full-attention (global) layers. Different from `head_dim`
    /// which is used by sliding layers.
    pub global_head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Window size for sliding_attention layers (token count).
    pub sliding_window: usize,

    /// Last N layers don't project their own K/V; they reuse K/V from the
    /// last non-shared layer of the same attention type.
    #[serde(default)]
    pub num_kv_shared_layers: usize,

    /// Gemma-style logit softcap: logits = cap * tanh(logits / cap) pre-sampling.
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,

    /// Per-layer embedding pathway. Each layer gets a small conditioning vector
    /// of this dimension, looked up per token from a separate vocab-sized table.
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,

    /// E2B doubles the MLP gate/up projections (gate has 2x intermediate).
    #[serde(default)]
    pub use_double_wide_mlp: bool,

    /// Activation string ("gelu_pytorch_tanh" for Gemma 4).
    pub hidden_activation: String,

    /// "sliding_attention" or "full_attention" per layer.
    pub layer_types: Vec<String>,

    #[serde(default)]
    pub rope_parameters: GemmaRopeParameters,

    /// Present for 26B-A4B MoE variant; `false`/absent for dense E2B/E4B/31B.
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub expert_intermediate_size: Option<usize>,

    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl TextConfig {
    pub fn attn_kind(&self, layer_idx: usize) -> Option<AttnKind> {
        self.layer_types.get(layer_idx).and_then(|s| AttnKind::parse(s))
    }

    pub fn head_dim_for(&self, kind: AttnKind) -> usize {
        match kind {
            AttnKind::Sliding => self.head_dim,
            AttnKind::Full => self.global_head_dim,
        }
    }

    pub fn rope_for(&self, kind: AttnKind) -> GemmaRope {
        match kind {
            AttnKind::Sliding => self.rope_parameters.sliding_attention.clone().unwrap_or_default(),
            AttnKind::Full => self.rope_parameters.full_attention.clone().unwrap_or_default(),
        }
    }

    pub fn num_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.attn_kind(i) == Some(AttnKind::Full))
            .count()
    }

    pub fn num_sliding_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.attn_kind(i) == Some(AttnKind::Sliding))
            .count()
    }

    /// Count of layers that actually own their K/V projections.
    /// The last `num_kv_shared_layers` reuse K/V from earlier layers.
    pub fn num_kv_owning_layers(&self) -> usize {
        self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers)
    }

    /// EOS token IDs (may be a single ID or a list, matching Qwen's config shape).
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    /// Top-level tie_word_embeddings (multimodal wrapper may override text_config's).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    /// Kept as raw JSON; text-only inference ignores these.
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub audio_config: Option<serde_json::Value>,
}

impl Config {
    /// Whether this checkpoint is a pure-text dense model we can handle.
    /// Returns Err with a reason if we don't support the variant yet.
    pub fn check_supported(&self) -> Result<(), String> {
        if self.text_config.enable_moe_block {
            return Err("Gemma4 MoE variant (26B-A4B) is not supported".into());
        }
        for (i, t) in self.text_config.layer_types.iter().enumerate() {
            if AttnKind::parse(t).is_none() {
                return Err(format!("layer {i}: unrecognized layer_type '{t}'"));
            }
        }
        if self.text_config.layer_types.len() != self.text_config.num_hidden_layers {
            return Err(format!(
                "layer_types length ({}) != num_hidden_layers ({})",
                self.text_config.layer_types.len(),
                self.text_config.num_hidden_layers
            ));
        }
        Ok(())
    }
}

pub fn load_config(model_dir: &std::path::Path) -> Result<Config, String> {
    let config_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let config: Config =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    config.check_supported()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const E2B_CONFIG: &str = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "vocab_size": 262144,
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": true,
            "sliding_window": 512,
            "num_kv_shared_layers": 2,
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "use_double_wide_mlp": true,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            },
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0
        }
    }"#;

    #[test]
    fn parses_e2b_shaped_config() {
        let cfg: Config = serde_json::from_str(E2B_CONFIG).expect("parse");
        cfg.check_supported().expect("supported");
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 4);
        assert_eq!(t.head_dim, 256);
        assert_eq!(t.global_head_dim, 512);
        assert_eq!(t.attn_kind(0), Some(AttnKind::Sliding));
        assert_eq!(t.attn_kind(3), Some(AttnKind::Full));
        assert_eq!(t.head_dim_for(AttnKind::Sliding), 256);
        assert_eq!(t.head_dim_for(AttnKind::Full), 512);
        assert_eq!(t.num_sliding_attention_layers(), 3);
        assert_eq!(t.num_full_attention_layers(), 1);
        assert_eq!(t.num_kv_owning_layers(), 2);
        assert!(t.use_double_wide_mlp);
        assert_eq!(t.final_logit_softcapping, Some(30.0));
        let full_rope = t.rope_for(AttnKind::Full);
        assert_eq!(full_rope.rope_type, "proportional");
        assert!((full_rope.rope_theta - 1_000_000.0).abs() < 1e-3);
        assert!((full_rope.partial_rotary_factor - 0.25).abs() < 1e-6);
        let slide_rope = t.rope_for(AttnKind::Sliding);
        assert_eq!(slide_rope.rope_type, "default");
        assert_eq!(t.eos_token_ids(), vec![1]);
    }

    #[test]
    fn rejects_moe_variant() {
        let mut v: serde_json::Value = serde_json::from_str(E2B_CONFIG).unwrap();
        v["text_config"]["enable_moe_block"] = serde_json::Value::Bool(true);
        let cfg: Config = serde_json::from_value(v).unwrap();
        assert!(cfg.check_supported().is_err());
    }
}
