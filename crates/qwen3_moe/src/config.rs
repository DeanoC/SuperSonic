use serde::Deserialize;

fn default_head_dim() -> usize {
    128
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_norm_topk_prob() -> bool {
    true
}

fn default_router_aux_loss_coef() -> f64 {
    0.001
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Silu,
    Gelu,
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
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,

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
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f64,
}

impl TextConfig {
    pub fn rotary_dim(&self) -> usize {
        self.head_dim
    }

    pub fn top_k(&self) -> usize {
        self.num_experts_per_tok
    }

    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        let per_layer = 2 * self.num_key_value_heads * self.head_dim * dtype_bytes;
        (self.num_hidden_layers * per_layer) as u64
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
    #[serde(flatten)]
    pub text_config: TextConfig,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

pub fn load_config(model_dir: &std::path::Path) -> Result<Config, String> {
    let config_path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config: Config =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    validate(&config)?;
    Ok(config)
}

fn validate(config: &Config) -> Result<(), String> {
    let t = &config.text_config;
    if config.model_type.as_deref() != Some("qwen3_moe") {
        return Err(format!(
            "model_type must be qwen3_moe for Qwen3-MoE, got {:?}",
            config.model_type
        ));
    }
    if !config
        .architectures
        .iter()
        .any(|a| a == "Qwen3MoeForCausalLM")
    {
        return Err("architectures must include Qwen3MoeForCausalLM".to_string());
    }
    if t.num_hidden_layers == 0 {
        return Err("num_hidden_layers must be > 0".to_string());
    }
    if t.hidden_size == 0 || t.num_attention_heads == 0 {
        return Err("hidden_size and num_attention_heads must be > 0".to_string());
    }
    if t.head_dim == 0 {
        return Err("head_dim must be > 0".to_string());
    }
    if t.num_key_value_heads == 0 || t.num_attention_heads % t.num_key_value_heads != 0 {
        return Err("num_attention_heads must be divisible by num_key_value_heads".to_string());
    }
    if t.num_experts == 0 || t.num_experts_per_tok == 0 {
        return Err("num_experts and num_experts_per_tok must be > 0".to_string());
    }
    if t.num_experts_per_tok > t.num_experts {
        return Err("num_experts_per_tok must be <= num_experts".to_string());
    }
    if t.moe_intermediate_size == 0 {
        return Err("moe_intermediate_size must be > 0".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_30b_geometry_sanity() {
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
            "rms_norm_eps": 1e-6
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        validate(&cfg).unwrap();
        assert_eq!(cfg.text_config.kv_bytes_per_token(2), 48 * 2 * 4 * 128 * 2);
    }
}
