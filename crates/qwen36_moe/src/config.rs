use serde::Deserialize;

fn default_head_dim() -> usize {
    256
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}
fn default_linear_key_head_dim() -> usize {
    128
}
fn default_linear_value_head_dim() -> usize {
    128
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_num_value_heads() -> usize {
    32
}
fn default_partial_rotary_factor() -> f64 {
    0.25
}
fn default_rope_theta() -> f64 {
    10_000_000.0
}
fn default_rope_type() -> String {
    "default".to_string()
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_router_aux_loss_coef() -> f64 {
    0.001
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
            mrope_interleaved: false,
            mrope_section: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Silu,
    Gelu,
    Swiglu,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,

    #[serde(default)]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,

    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub attn_output_gate: bool,

    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,

    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f64,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub decoder_sparse_step: Option<usize>,
}

impl TextConfig {
    pub fn normalized(mut self) -> Self {
        if self.layer_types.is_empty() {
            let interval = self.full_attention_interval.max(1);
            self.layer_types = (0..self.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        self
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.partial_rotary_factor)
            .unwrap_or_else(default_partial_rotary_factor)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()) as usize
    }

    pub fn top_k(&self) -> usize {
        self.num_experts_per_tok
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "full_attention")
            .unwrap_or(false)
    }

    pub fn is_dense_mlp_layer(&self, layer_idx: usize) -> bool {
        self.mlp_only_layers.contains(&layer_idx)
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn num_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_full_attention(i))
            .count()
    }

    pub fn num_linear_attention_layers(&self) -> usize {
        self.num_hidden_layers - self.num_full_attention_layers()
    }

    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        let per_layer = 2 * self.num_key_value_heads * self.head_dim * dtype_bytes;
        (self.num_full_attention_layers() * per_layer) as u64
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        extract_token_ids(&self.eos_token_id)
    }

    pub fn bos_token_ids(&self) -> Vec<u32> {
        extract_token_ids(&self.bos_token_id)
    }
}

fn extract_token_ids(value: &Option<serde_json::Value>) -> Vec<u32> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|v| vec![v as u32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as u32))
            .collect(),
        _ => vec![],
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

impl Config {
    pub fn normalized(mut self) -> Self {
        self.text_config = self.text_config.normalized();
        self
    }
}

pub fn load_config(model_dir: &std::path::Path) -> Result<Config, String> {
    let config_path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config: Config =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    let normalized = config.normalized();
    validate(&normalized)?;
    Ok(normalized)
}

fn validate(config: &Config) -> Result<(), String> {
    let t = &config.text_config;

    if t.num_hidden_layers == 0 {
        return Err("num_hidden_layers must be > 0".to_string());
    }
    if t.layer_types.len() != t.num_hidden_layers {
        return Err(format!(
            "layer_types length {} != num_hidden_layers {}",
            t.layer_types.len(),
            t.num_hidden_layers
        ));
    }
    for (i, kind) in t.layer_types.iter().enumerate() {
        if kind != "full_attention" && kind != "linear_attention" {
            return Err(format!(
                "layer_types[{i}] = {kind:?}, expected full_attention or linear_attention"
            ));
        }
    }

    let interval = t.full_attention_interval.max(1);
    let expected_full: usize = (0..t.num_hidden_layers)
        .filter(|&i| (i + 1) % interval == 0)
        .count();
    if t.num_full_attention_layers() != expected_full {
        return Err(format!(
            "hybrid pattern broken: got {} full-attention layers, expected {} (interval {}, layers {})",
            t.num_full_attention_layers(),
            expected_full,
            interval,
            t.num_hidden_layers,
        ));
    }

    if t.num_attention_heads == 0 || t.num_key_value_heads == 0 {
        return Err("attention head counts must be > 0".to_string());
    }
    if !t.num_attention_heads.is_multiple_of(t.num_key_value_heads) {
        return Err(format!(
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            t.num_attention_heads, t.num_key_value_heads
        ));
    }
    let rot_dim = t.rotary_dim();
    if rot_dim == 0 || !rot_dim.is_multiple_of(2) {
        return Err(format!("rotary_dim {rot_dim} must be even and > 0"));
    }
    if t.rotary_dim() > t.head_dim {
        return Err(format!(
            "rotary_dim {} exceeds head_dim {}",
            t.rotary_dim(),
            t.head_dim
        ));
    }

    if t.num_experts == 0 {
        return Err("num_experts must be > 0 for MoE family".to_string());
    }
    if t.num_experts_per_tok == 0 || t.num_experts_per_tok > t.num_experts {
        return Err(format!(
            "num_experts_per_tok {} must be in 1..={}",
            t.num_experts_per_tok, t.num_experts
        ));
    }
    if t.moe_intermediate_size == 0 {
        return Err("moe_intermediate_size must be > 0".to_string());
    }
    if t.shared_expert_intermediate_size == 0 {
        return Err("shared_expert_intermediate_size must be > 0".to_string());
    }
    for &idx in &t.mlp_only_layers {
        if idx >= t.num_hidden_layers {
            return Err(format!(
                "mlp_only_layers entry {idx} >= num_hidden_layers {}",
                t.num_hidden_layers
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_qwen36_35b_a3b_config_json() -> String {
        let mut layers = Vec::with_capacity(40);
        for idx in 0..40 {
            if (idx + 1) % 4 == 0 {
                layers.push("\"full_attention\"");
            } else {
                layers.push("\"linear_attention\"");
            }
        }
        let layer_types = layers.join(",");
        format!(
            r#"{{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "tie_word_embeddings": false,
                "text_config": {{
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "attn_output_gate": true,
                    "bos_token_id": 248044,
                    "dtype": "bfloat16",
                    "eos_token_id": 248044,
                    "full_attention_interval": 4,
                    "head_dim": 256,
                    "hidden_act": "silu",
                    "hidden_size": 2048,
                    "initializer_range": 0.02,
                    "layer_types": [{layer_types}],
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_value_head_dim": 128,
                    "max_position_embeddings": 262144,
                    "model_type": "qwen3_5_moe_text",
                    "moe_intermediate_size": 512,
                    "num_attention_heads": 16,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "num_hidden_layers": 40,
                    "num_key_value_heads": 2,
                    "output_router_logits": false,
                    "partial_rotary_factor": 0.25,
                    "rms_norm_eps": 1e-06,
                    "rope_parameters": {{
                        "mrope_interleaved": true,
                        "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000,
                        "rope_type": "default"
                    }},
                    "router_aux_loss_coef": 0.001,
                    "shared_expert_intermediate_size": 512,
                    "tie_word_embeddings": false,
                    "use_cache": true,
                    "vocab_size": 248320
                }}
            }}"#
        )
    }

    #[test]
    fn parses_real_qwen36_35b_a3b_config() {
        let json = real_qwen36_35b_a3b_config_json();
        let config: Config = serde_json::from_str(&json).expect("parse");
        let config = config.normalized();
        validate(&config).expect("validate");
        let t = &config.text_config;

        assert_eq!(config.architectures, vec!["Qwen3_5MoeForConditionalGeneration"]);
        assert_eq!(config.model_type.as_deref(), Some("qwen3_5_moe"));

        assert_eq!(t.vocab_size, 248320);
        assert_eq!(t.hidden_size, 2048);
        assert_eq!(t.num_hidden_layers, 40);
        assert_eq!(t.num_attention_heads, 16);
        assert_eq!(t.num_key_value_heads, 2);
        assert_eq!(t.head_dim, 256);
        assert_eq!(t.max_position_embeddings, 262144);
        assert_eq!(t.full_attention_interval, 4);
        assert!(t.attn_output_gate);

        assert_eq!(t.linear_num_key_heads, 16);
        assert_eq!(t.linear_num_value_heads, 32);
        assert_eq!(t.linear_key_head_dim, 128);
        assert_eq!(t.linear_value_head_dim, 128);
        assert_eq!(t.linear_conv_kernel_dim, 4);
        assert_eq!(t.linear_value_dim(), 32 * 128);

        assert_eq!(t.rotary_dim(), 64);
        assert_eq!(t.rope_theta(), 10_000_000.0);
        let rp = t.rope_parameters.as_ref().expect("rope_parameters");
        assert!(rp.mrope_interleaved);
        assert_eq!(rp.mrope_section, vec![11, 11, 10]);

        assert_eq!(t.num_experts, 256);
        assert_eq!(t.num_experts_per_tok, 8);
        assert_eq!(t.top_k(), 8);
        assert_eq!(t.moe_intermediate_size, 512);
        assert_eq!(t.shared_expert_intermediate_size, 512);
        assert!(t.norm_topk_prob);
        assert!((t.router_aux_loss_coef - 0.001).abs() < 1e-9);
        assert!(t.mlp_only_layers.is_empty());
        assert!(t.decoder_sparse_step.is_none());

        assert_eq!(t.num_full_attention_layers(), 10);
        assert_eq!(t.num_linear_attention_layers(), 30);
        assert!(t.is_full_attention(3));
        assert!(!t.is_full_attention(0));

        assert_eq!(t.eos_token_ids(), vec![248044]);
        assert_eq!(t.bos_token_ids(), vec![248044]);

        assert_eq!(t.kv_bytes_per_token(2), (2 * 2 * 256 * 2 * 10) as u64);
    }

    #[test]
    fn synthesizes_layer_types_when_missing() {
        let json = r#"{
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-06,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let config = config.normalized();
        validate(&config).unwrap();
        let t = &config.text_config;
        assert_eq!(t.layer_types.len(), 8);
        assert_eq!(t.num_full_attention_layers(), 2);
        assert!(t.is_full_attention(3));
        assert!(t.is_full_attention(7));
    }

    #[test]
    fn rejects_layer_type_count_mismatch() {
        let json = r#"{
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-06,
                "layer_types": ["full_attention", "linear_attention"],
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let err = validate(&config.normalized()).unwrap_err();
        assert!(err.contains("layer_types length"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_layer_type() {
        let json = r#"{
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-06,
                "layer_types": ["full_attention", "sliding_attention"],
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let err = validate(&config.normalized()).unwrap_err();
        assert!(err.contains("sliding_attention"), "got: {err}");
    }

    #[test]
    fn rejects_top_k_above_num_experts() {
        let json = r#"{
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-06,
                "num_experts": 4,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let err = validate(&config.normalized()).unwrap_err();
        assert!(err.contains("num_experts_per_tok"), "got: {err}");
    }

    #[test]
    fn rejects_hybrid_pattern_break() {
        let mut layers = (0..8)
            .map(|idx| {
                if (idx + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>();
        layers[0] = "\"full_attention\"";
        let layer_types = layers.join(",");
        let json = format!(
            r#"{{
                "text_config": {{
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 4096,
                    "rms_norm_eps": 1e-06,
                    "full_attention_interval": 4,
                    "layer_types": [{layer_types}],
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64
                }}
            }}"#
        );
        let config: Config = serde_json::from_str(&json).unwrap();
        let err = validate(&config.normalized()).unwrap_err();
        assert!(err.contains("hybrid pattern broken"), "got: {err}");
    }
}
