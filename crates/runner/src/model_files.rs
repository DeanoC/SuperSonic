use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::Cli;

pub(crate) fn load_tokenizer(tokenizer_path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))
}

/// Validate every file needed by the public Qwen3.8 GQH startup contract.
///
/// This function intentionally performs only host-side reads. Callers should
/// invoke it before selecting an accelerator or allocating any GPU buffers so
/// a typo in a model directory or artifact fails with an actionable message.
pub fn validate_input_contract(cli: &Cli) -> Result<()> {
    if cli.model != "qwen3.8-27b" {
        anyhow::bail!(
            "unsupported model {:?}; the startup contract requires --model qwen3.8-27b",
            cli.model
        );
    }
    if cli.temperature != 0.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--temperature 0)");
    }
    if cli.top_k != 0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-k 0)");
    }
    if cli.top_p != 1.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-p 1)");
    }

    let config_path = cli.model_dir.join("config.json");
    require_file(&config_path, "Qwen3.8 config.json")?;
    let config = qwen35::config::load_config(&cli.model_dir).map_err(|e| {
        anyhow::anyhow!("invalid Qwen3.8 config.json {}: {e}", config_path.display())
    })?;

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    require_file(&tokenizer_path, "Qwen3.8 tokenizer data")?;
    load_tokenizer(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "invalid Qwen3.8 tokenizer data {}: {e}",
            tokenizer_path.display()
        )
    })?;

    if cli.chat {
        let tokenizer_config_path = cli.model_dir.join("tokenizer_config.json");
        require_file(&tokenizer_config_path, "Qwen3.8 chat-template metadata")?;
        supersonic_runtime::chat_template::ChatTemplate::try_load(&cli.model_dir)
            .map_err(|e| {
                anyhow::anyhow!(
                    "invalid Qwen3.8 chat-template metadata {}: {e}",
                    tokenizer_config_path.display()
                )
            })?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.8 chat-template metadata {} does not define a chat template",
                    tokenizer_config_path.display()
                )
            })?;
    }

    let gguf_path = cli.gguf_file.as_deref().ok_or_else(|| {
        anyhow::anyhow!("missing Qwen3.8 GQH GGUF artifact: --gguf-file is required")
    })?;
    require_file(gguf_path, "Qwen3.8 GQH GGUF artifact")?;
    let gguf = model_store::gguf::GgufFile::open(gguf_path).map_err(|e| {
        anyhow::anyhow!(
            "invalid Qwen3.8 GQH GGUF artifact {}: {e}",
            gguf_path.display()
        )
    })?;

    if gguf.gqh_header_count() == 0 {
        anyhow::bail!(
            "GGUF {} is not a custom GQH artifact: required GQH headers are absent",
            gguf_path.display()
        );
    }
    match gguf.kv("general.architecture") {
        Some("qwen35") => {}
        Some(architecture) => anyhow::bail!(
            "GQH GGUF {} has unsupported architecture {:?}; expected qwen35 for Qwen3.8",
            gguf_path.display(),
            architecture
        ),
        None => anyhow::bail!(
            "GQH GGUF {} is missing required general.architecture metadata",
            gguf_path.display()
        ),
    }

    qwen35::gguf_ingest::check_mapping(&gguf, &config.text_config).map_err(|e| {
        anyhow::anyhow!(
            "GQH GGUF {} is incompatible with Qwen3.8 geometry or qtypes: {e}",
            gguf_path.display()
        )
    })?;
    Ok(())
}

fn require_file(path: &Path, role: &str) -> Result<()> {
    let metadata = fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("missing {role} at {}: {e}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{role} at {} is not a regular file", path.display());
    }
    fs::File::open(path)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{role} at {} is not readable: {e}", path.display()))
}
