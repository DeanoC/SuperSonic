use anyhow::Result;

use crate::model_files::load_tokenizer;
use crate::Cli;

/// Host-side startup state shared by the direct Qwen3.8 generation path.
pub(crate) struct Qwen35Startup {
    pub(crate) text_config: qwen35::config::TextConfig,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
}

pub(crate) fn load_qwen35_startup(cli: &Cli) -> Result<Qwen35Startup> {
    let config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading Qwen3.8 config.json: {e}"))?;
    let text_config = config.text_config;
    eprintln!(
        "[config] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let prompt_text = if cli.chat {
        let template = supersonic_runtime::chat_template::ChatTemplate::try_load(&cli.model_dir)?
            .ok_or_else(|| {
            anyhow::anyhow!(
                "--chat requires chat-template metadata in {}",
                cli.model_dir.join("tokenizer_config.json").display()
            )
        })?;
        let rendered = template.render(
            &[supersonic_runtime::chat_template::ChatMessage::text(
                "user",
                cli.prompt.as_str(),
            )],
            true,
        )?;
        eprintln!("[chat] rendered {} chars", rendered.len());
        rendered
    } else {
        cli.prompt.clone()
    };
    let encoding = tokenizer
        .encode(
            prompt_text.as_str(),
            !cli.prompt_no_special_tokens && !cli.chat,
        )
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    if context_tokens < prompt_ids.len() {
        anyhow::bail!(
            "--context-size={} is smaller than the {} prompt tokens",
            context_tokens,
            prompt_ids.len()
        );
    }

    Ok(Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    })
}
