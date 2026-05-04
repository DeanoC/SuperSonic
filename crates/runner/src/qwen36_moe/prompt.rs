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
