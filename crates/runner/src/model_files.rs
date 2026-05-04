use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::Cli;

pub(crate) fn load_tokenizer(tokenizer_path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))
}

pub(crate) fn resolve_prompt_token_ids(
    cli: &Cli,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt tokenization produced 0 tokens");
    }
    Ok(prompt_ids)
}

pub(crate) fn model_dir_has_raw_safetensors(model_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(model_dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
    })
}
