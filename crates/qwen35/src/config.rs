use serde::Deserialize;

fn default_head_dim() -> usize {
    256
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
    10_000.0
}
fn default_rope_type() -> String {
    "default".to_string()
}
fn default_rms_norm_add_unit_offset() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

/// Activation function identifier (only used for config parsing).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Gelu,
    Silu,
    Swiglu,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rms_norm_add_unit_offset")]
    pub rms_norm_add_unit_offset: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
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
}

impl TextConfig {
    pub fn normalized(mut self) -> Self {
        if self.layer_types.is_empty() {
            self.layer_types = (0..self.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % 4 == 0 {
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

    /// Get EOS token IDs (may be a single ID or a list).
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
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

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "full_attention")
            .unwrap_or(false)
    }

    /// Value dimension for linear attention output projection.
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Number of full-attention layers in the model.
    pub fn num_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_full_attention(i))
            .count()
    }

    /// KV cache bytes per token across all full-attention layers.
    /// Each full-attention layer stores K and V: 2 × num_kv_heads × head_dim × dtype_bytes.
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        let per_layer = 2 * self.num_key_value_heads * self.head_dim * dtype_bytes;
        (self.num_full_attention_layers() * per_layer) as u64
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

impl Config {
    pub fn normalized(mut self) -> Self {
        self.text_config = self.text_config.normalized();
        self
    }
}

/// Load config.json from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<Config, String> {
    let config_path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config: Config =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    Ok(config.normalized())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen36_27b_text_config_geometry() {
        let layer_types = (0..64)
            .map(|idx| {
                if (idx + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{
                "text_config": {{
                    "vocab_size": 248320,
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "hidden_act": "silu",
                    "max_position_embeddings": 262144,
                    "rms_norm_eps": 1e-06,
                    "tie_word_embeddings": false,
                    "eos_token_id": 248044,
                    "head_dim": 256,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "layer_types": [{layer_types}],
                    "rope_parameters": {{
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000,
                        "rope_type": "default",
                        "mrope_interleaved": true,
                        "mrope_section": [11, 11, 10]
                    }}
                }},
                "vision_config": {{"hidden_size": 1152}}
            }}"#
        );

        let config: Config = serde_json::from_str(&json).unwrap();
        let text = config.normalized().text_config;
        assert_eq!(text.num_hidden_layers, 64);
        assert_eq!(text.num_full_attention_layers(), 16);
        assert_eq!(text.hidden_size, 5120);
        assert_eq!(text.intermediate_size, 17408);
        assert_eq!(text.num_attention_heads, 24);
        assert_eq!(text.num_key_value_heads, 4);
        assert_eq!(text.linear_value_dim(), 6144);
        assert_eq!(text.rotary_dim(), 64);
        assert_eq!(text.rope_theta(), 10_000_000.0);
        assert_eq!(text.kv_bytes_per_token(2), 65_536);
        assert!(text.rms_norm_add_unit_offset);
    }
}
