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
    false
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

    pub fn from_flm_qwen36_dense(flm: &model_store::FlmQwen36DenseConfig) -> Self {
        Self::try_from_flm_qwen36_dense(flm)
            .unwrap_or_else(|e| panic!("invalid FLM Qwen3.6 dense config: {e}"))
    }

    pub fn try_from_flm_qwen36_dense(
        flm: &model_store::FlmQwen36DenseConfig,
    ) -> Result<Self, String> {
        validate_positive("vocab_size", flm.vocab_size)?;
        validate_positive("hidden_size", flm.hidden_size)?;
        validate_positive("intermediate_size", flm.intermediate_size)?;
        validate_positive("num_hidden_layers", flm.num_hidden_layers)?;
        validate_positive("num_attention_heads", flm.num_attention_heads)?;
        validate_positive("num_key_value_heads", flm.num_key_value_heads)?;
        validate_positive("head_dim", flm.head_dim)?;
        validate_positive("max_position_embeddings", flm.max_position_embeddings)?;
        validate_positive("linear_conv_kernel_dim", flm.linear_conv_kernel_dim)?;
        validate_positive("linear_key_head_dim", flm.linear_key_head_dim)?;
        validate_positive("linear_value_head_dim", flm.linear_value_head_dim)?;
        validate_positive("linear_num_key_heads", flm.linear_num_key_heads)?;
        validate_positive("linear_num_value_heads", flm.linear_num_value_heads)?;
        validate_positive_finite("rms_norm_eps", flm.rms_norm_eps)?;
        validate_positive_finite("rope_theta", flm.rope_theta)?;
        validate_positive_finite("partial_rotary_factor", flm.partial_rotary_factor)?;
        if flm.partial_rotary_factor > 1.0 {
            return Err(format!(
                "partial_rotary_factor must be <= 1.0, got {}",
                flm.partial_rotary_factor
            ));
        }
        let attention_hidden = flm
            .num_attention_heads
            .checked_mul(flm.head_dim)
            .ok_or_else(|| {
                format!(
                    "num_attention_heads {} * head_dim {} overflows",
                    flm.num_attention_heads, flm.head_dim
                )
            })?;
        if flm.hidden_size != attention_hidden {
            return Err(format!(
                "hidden_size {} must equal num_attention_heads {} * head_dim {}",
                flm.hidden_size, flm.num_attention_heads, flm.head_dim
            ));
        }
        if flm.num_key_value_heads > flm.num_attention_heads {
            return Err(format!(
                "num_key_value_heads {} must be <= num_attention_heads {}",
                flm.num_key_value_heads, flm.num_attention_heads
            ));
        }

        let hidden_act = activation_from_flm_id(flm.activation_id)?;
        let mut layer_types = vec!["linear_attention".to_string(); flm.num_hidden_layers];
        for &idx in &flm.full_attention_layers {
            let slot = layer_types.get_mut(idx).ok_or_else(|| {
                format!(
                    "full_attention_layers contains {idx}, but num_hidden_layers is {}",
                    flm.num_hidden_layers
                )
            })?;
            *slot = "full_attention".to_string();
        }

        Ok(Self {
            text_config: TextConfig {
                vocab_size: flm.vocab_size,
                hidden_size: flm.hidden_size,
                intermediate_size: flm.intermediate_size,
                num_hidden_layers: flm.num_hidden_layers,
                num_attention_heads: flm.num_attention_heads,
                num_key_value_heads: flm.num_key_value_heads,
                hidden_act,
                max_position_embeddings: flm.max_position_embeddings,
                rms_norm_eps: flm.rms_norm_eps,
                rms_norm_add_unit_offset: false,
                tie_word_embeddings: flm.tie_word_embeddings,
                eos_token_id: Some(serde_json::Value::Array(
                    flm.eos_token_ids
                        .iter()
                        .map(|&id| serde_json::json!(id))
                        .collect(),
                )),
                head_dim: flm.head_dim,
                linear_conv_kernel_dim: flm.linear_conv_kernel_dim,
                linear_key_head_dim: flm.linear_key_head_dim,
                linear_value_head_dim: flm.linear_value_head_dim,
                linear_num_key_heads: flm.linear_num_key_heads,
                linear_num_value_heads: flm.linear_num_value_heads,
                layer_types,
                rope_parameters: Some(RopeParameters {
                    rope_type: "default".to_string(),
                    rope_theta: flm.rope_theta,
                    partial_rotary_factor: flm.partial_rotary_factor,
                }),
            },
        })
    }
}

fn validate_positive(field: &str, value: usize) -> Result<(), String> {
    if value == 0 {
        Err(format!("{field} must be non-zero"))
    } else {
        Ok(())
    }
}

fn validate_positive_finite(field: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        Err(format!("{field} must be finite"))
    } else if value <= 0.0 {
        Err(format!("{field} must be positive"))
    } else {
        Ok(())
    }
}

fn activation_from_flm_id(activation_id: u8) -> Result<Activation, String> {
    match activation_id {
        0 => Ok(Activation::Gelu),
        1 => Ok(Activation::Silu),
        2 => Ok(Activation::Swiglu),
        _ => Err(format!("unknown activation_id {activation_id}")),
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

    fn flm_qwen36_dense_descriptor() -> model_store::FlmQwen36DenseConfig {
        model_store::FlmQwen36DenseConfig {
            vocab_size: 248320,
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            num_attention_heads: 20,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 262144,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            partial_rotary_factor: 0.25,
            activation_id: 1,
            tie_word_embeddings: false,
            eos_token_ids: vec![248044],
            full_attention_layers: (3..64).step_by(4).collect(),
        }
    }

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
        assert!(!text.rms_norm_add_unit_offset);
    }

    #[test]
    fn builds_text_config_from_flm_qwen36_dense_descriptor() {
        let flm = flm_qwen36_dense_descriptor();

        let config = Config::try_from_flm_qwen36_dense(&flm)
            .unwrap()
            .normalized();

        assert_eq!(config.text_config.hidden_size, 5120);
        assert_eq!(config.text_config.num_full_attention_layers(), 16);
        assert_eq!(config.text_config.linear_value_dim(), 6144);
        assert_eq!(config.text_config.rope_theta(), 10000000.0);
        assert_eq!(config.text_config.eos_token_ids(), vec![248044]);
    }

    #[test]
    fn rejects_unknown_activation_id_from_flm_qwen36_dense_descriptor() {
        let mut flm = flm_qwen36_dense_descriptor();
        flm.activation_id = 99;

        let err = Config::try_from_flm_qwen36_dense(&flm).unwrap_err();

        assert!(err.contains("activation_id"), "{err}");
        assert!(err.contains("99"), "{err}");
    }

    #[test]
    fn rejects_out_of_range_full_attention_layer_from_flm_qwen36_dense_descriptor() {
        let mut flm = flm_qwen36_dense_descriptor();
        flm.full_attention_layers.push(64);

        let err = Config::try_from_flm_qwen36_dense(&flm).unwrap_err();

        assert!(err.contains("full_attention_layers"), "{err}");
        assert!(err.contains("64"), "{err}");
        assert!(err.contains("num_hidden_layers"), "{err}");
    }

    #[test]
    fn rejects_hidden_size_head_geometry_mismatch_from_flm_qwen36_dense_descriptor() {
        let mut flm = flm_qwen36_dense_descriptor();
        flm.head_dim = 128;

        let err = Config::try_from_flm_qwen36_dense(&flm).unwrap_err();

        assert!(err.contains("hidden_size"), "{err}");
        assert!(err.contains("num_attention_heads"), "{err}");
        assert!(err.contains("head_dim"), "{err}");
    }

    #[test]
    fn rejects_more_kv_heads_than_attention_heads_from_flm_qwen36_dense_descriptor() {
        let mut flm = flm_qwen36_dense_descriptor();
        flm.num_key_value_heads = 25;

        let err = Config::try_from_flm_qwen36_dense(&flm).unwrap_err();

        assert!(err.contains("num_key_value_heads"), "{err}");
        assert!(err.contains("num_attention_heads"), "{err}");
    }
}
