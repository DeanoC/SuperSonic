use std::path::Path;

use anyhow::{anyhow, Result};
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_cli::timing::SamplingParams;

pub(crate) struct Qwen36PromptSetup {
    pub(crate) tokenizer: Option<tokenizers::Tokenizer>,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) eos_id: Option<u32>,
}

pub(crate) fn validate_speculative_sampling(
    speculative_decode: bool,
    sampling: SamplingParams,
) -> Result<()> {
    if !speculative_decode {
        return Ok(());
    }

    let is_greedy = sampling.temperature <= 0.0 || sampling.top_k == 1;
    if is_greedy {
        return Ok(());
    }

    anyhow::bail!(
        "--speculative-decode currently supports greedy sampling \
         only (temperature <= 0 or top_k == 1). Got temperature={}, \
         top_k={}, top_p={}. Phase 6.4 will add sampling-consistent \
         verification (rejection sampling); until then, re-run with \
         `--temperature 0` for speculative decode, or drop \
         `--speculative-decode` for non-greedy sampling.",
        sampling.temperature,
        sampling.top_k,
        sampling.top_p
    );
}

pub(crate) fn prepare_prompt(
    model_dir: &Path,
    text_config: &TextConfig,
    prompt: &str,
) -> Result<Qwen36PromptSetup> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path).ok();

    prepare_prompt_with_tokenizer(tokenizer, text_config, prompt)
}

pub(crate) fn prepare_prompt_with_tokenizer(
    tokenizer: Option<tokenizers::Tokenizer>,
    text_config: &TextConfig,
    prompt: &str,
) -> Result<Qwen36PromptSetup> {
    let bos_id = text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let eos_id = text_config
        .eos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let prompt_ids = match (&tokenizer, prompt.is_empty()) {
        (Some(tok), false) => {
            let enc = tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            if ids.is_empty() {
                vec![bos_id]
            } else {
                ids
            }
        }
        _ => vec![bos_id],
    };

    Ok(Qwen36PromptSetup {
        tokenizer,
        prompt_ids,
        eos_id,
    })
}

pub(crate) fn print_prompt_summary(prompt: &str, prompt_ids: &[u32]) {
    println!(
        "  prompt: {prompt:?} → {} token{} ({:?}{}…)",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        &prompt_ids[..prompt_ids.len().min(8)],
        if prompt_ids.len() > 8 { ", " } else { "" },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36_moe::config::{Activation, TextConfig};
    use tokenizers::models::bpe::BPE;

    fn text_config_with_bos_eos() -> TextConfig {
        TextConfig {
            vocab_size: 1024,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: Some(serde_json::json!(9)),
            bos_token_id: Some(serde_json::json!(7)),
            head_dim: 64,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            layer_types: Vec::new(),
            rope_parameters: None,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 64,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
        .normalized()
    }

    #[test]
    fn prepare_prompt_with_preloaded_tokenizer_keeps_tokenizer_and_empty_prompt_uses_bos() {
        let tokenizer = tokenizers::Tokenizer::new(BPE::default());

        let setup = prepare_prompt_with_tokenizer(Some(tokenizer), &text_config_with_bos_eos(), "")
            .expect("prompt setup");

        assert!(setup.tokenizer.is_some());
        assert_eq!(setup.prompt_ids, vec![7]);
        assert_eq!(setup.eos_id, Some(9));
    }
}
