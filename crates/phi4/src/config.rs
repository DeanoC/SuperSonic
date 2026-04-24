use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RopeScaling {
    Longrope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Phi4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,
}

impl Phi4Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim() as f64) * self.partial_rotary_factor) as usize
    }

    pub fn mscale(&self) -> f64 {
        let scale =
            self.max_position_embeddings as f64 / self.original_max_position_embeddings as f64;
        if scale <= 1.0 {
            1.0
        } else {
            (1.0 + scale.ln() / (self.original_max_position_embeddings as f64).ln()).sqrt()
        }
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        extract_token_ids(&self.eos_token_id)
    }

    pub fn bos_token_ids(&self) -> Vec<u32> {
        extract_token_ids(&self.bos_token_id)
    }

    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        let per_layer = 2 * self.num_key_value_heads * self.head_dim() * dtype_bytes;
        (self.num_hidden_layers * per_layer) as u64
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

pub fn load_config(model_dir: &std::path::Path) -> Result<Phi4Config, String> {
    let config_path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config: Phi4Config =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    validate(&config)?;
    Ok(config)
}

fn validate(config: &Phi4Config) -> Result<(), String> {
    if config.hidden_size % config.num_attention_heads != 0 {
        return Err(format!(
            "hidden_size {} not divisible by num_attention_heads {}",
            config.hidden_size, config.num_attention_heads
        ));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err(format!(
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            config.num_attention_heads, config.num_key_value_heads
        ));
    }
    let rot_dim = config.rotary_dim();
    if rot_dim % 2 != 0 {
        return Err(format!("rotary_dim {rot_dim} must be even"));
    }
    let expected_factor_len = rot_dim / 2;
    if let Some(RopeScaling::Longrope {
        short_factor,
        long_factor,
    }) = &config.rope_scaling
    {
        if short_factor.len() != expected_factor_len {
            return Err(format!(
                "short_factor length {} != rotary_dim/2 {}",
                short_factor.len(),
                expected_factor_len
            ));
        }
        if long_factor.len() != expected_factor_len {
            return Err(format!(
                "long_factor length {} != rotary_dim/2 {}",
                long_factor.len(),
                expected_factor_len
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phi4_mini_config() {
        let json = r#"{
            "vocab_size": 200064,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.75,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "long_factor": [1.0, 1.118, 1.25, 1.398, 1.564, 1.749, 1.956, 2.187, 2.446, 2.735, 3.059, 3.421, 3.826, 4.279, 4.785, 5.351, 5.984, 6.693, 7.485, 8.37, 9.361, 10.468, 11.707, 13.092, 14.641, 16.374, 18.311, 20.478, 22.901, 25.61, 28.641, 32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71, 32.93, 32.97, 33.28, 33.49, 33.5, 44.16, 47.77]
            },
            "eos_token_id": 199999
        }"#;
        let config: Phi4Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.rotary_dim(), 96);
        assert_eq!(config.eos_token_ids(), vec![199999]);
        assert!(config.mscale() > 1.0);
        validate(&config).unwrap();
    }

    #[test]
    fn rejects_mismatched_factor_length() {
        let config = Phi4Config {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 2048,
            original_max_position_embeddings: 512,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            rope_scaling: Some(RopeScaling::Longrope {
                short_factor: vec![1.0; 10],
                long_factor: vec![1.0; 32],
            }),
            eos_token_id: None,
            bos_token_id: None,
        };
        assert!(validate(&config).is_err());
    }
}
