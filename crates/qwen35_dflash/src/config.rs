//! DFlash draft config — parsed from the checkpoint's `config.json`.
//!
//! Nothing architectural is hardcoded; tap indices, block size, and mask
//! token id all come from the checkpoint. This lets us retarget to a future
//! Qwen-size DFlash draft without code changes.

use std::path::Path;

use serde::Deserialize;

/// DFlash-specific sub-config (nested under `dflash_config` in config.json).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DFlashSubConfig {
    /// Token id used to fill non-accepted positions in the draft's input
    /// block before forward. NOT a standard tokenizer special-token —
    /// trained in by DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices whose post-layer hidden states are tapped
    /// and fed to the draft's fuser. Length = number of taps (5 for 9B).
    pub target_layer_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DFlashConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub block_size: usize,
    /// Number of layers in the TARGET model (e.g. 32 for Qwen3.5-9B). The
    /// draft uses this only to validate tap indices at load time.
    pub num_target_layers: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub dflash_config: DFlashSubConfig,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl DFlashConfig {
    /// Number of tap layers (length of `target_layer_ids`). Determines
    /// `fc: [num_taps * hidden_size → hidden_size]` shape.
    pub fn num_taps(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    /// Input dim of the tap fuser: `num_taps * hidden_size`.
    pub fn fuser_in_dim(&self) -> usize {
        self.num_taps() * self.hidden_size
    }

    /// Q output dim: `num_attention_heads * head_dim`.
    pub fn q_out_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// K/V output dim: `num_key_value_heads * head_dim`.
    pub fn kv_out_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

pub fn load_config(model_dir: &Path) -> Result<DFlashConfig, String> {
    let config_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let config: DFlashConfig =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", config_path.display()))?;
    validate(&config)?;
    Ok(config)
}

fn validate(c: &DFlashConfig) -> Result<(), String> {
    if c.num_attention_heads == 0 || c.num_key_value_heads == 0 {
        return Err("num_attention_heads and num_key_value_heads must be > 0".into());
    }
    if c.num_attention_heads % c.num_key_value_heads != 0 {
        return Err(format!(
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            c.num_attention_heads, c.num_key_value_heads
        ));
    }
    if c.hidden_size % c.num_attention_heads != 0
        && c.head_dim * c.num_attention_heads != c.hidden_size
    {
        // Qwen3's head_dim can be independent of hidden_size / num_heads (true here:
        // 128 * 32 = 4096 which happens to match but don't assume it).
    }
    if c.block_size == 0 {
        return Err("block_size must be > 0".into());
    }
    let num_taps = c.dflash_config.target_layer_ids.len();
    if num_taps == 0 {
        return Err("dflash_config.target_layer_ids must not be empty".into());
    }
    for (i, &layer) in c.dflash_config.target_layer_ids.iter().enumerate() {
        if (layer as usize) >= c.num_target_layers {
            return Err(format!(
                "target_layer_ids[{i}]={layer} out of range for num_target_layers={}",
                c.num_target_layers
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen35_9b_dflash_config() {
        let json = r#"{
            "vocab_size": 248320,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rope_theta": 10000000,
            "rms_norm_eps": 1e-6,
            "block_size": 16,
            "num_target_layers": 32,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "dflash_config": {
                "mask_token_id": 248070,
                "target_layer_ids": [1, 8, 15, 22, 29]
            },
            "eos_token_id": 248044
        }"#;
        let c: DFlashConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.num_taps(), 5);
        assert_eq!(c.fuser_in_dim(), 20480);
        assert_eq!(c.q_out_dim(), 4096);
        assert_eq!(c.kv_out_dim(), 1024);
        assert_eq!(c.num_kv_groups(), 4);
        validate(&c).unwrap();
    }

    #[test]
    fn rejects_tap_out_of_range() {
        let mut c = DFlashConfig {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            block_size: 16,
            num_target_layers: 10,
            attention_bias: false,
            tie_word_embeddings: false,
            dflash_config: DFlashSubConfig {
                mask_token_id: 0,
                target_layer_ids: vec![1, 5, 15],
            },
            eos_token_id: None,
        };
        assert!(validate(&c).is_err());
        c.dflash_config.target_layer_ids = vec![1, 5, 9];
        validate(&c).unwrap();
    }
}
