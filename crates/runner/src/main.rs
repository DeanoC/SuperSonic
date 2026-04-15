mod decode_engine;
mod oracle;
mod validate;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use decode_engine::DecodeEngine;

#[derive(Parser)]
#[command(name = "qwen35-decode", about = "Persistent decode megakernel runner")]
struct Cli {
    /// Path to HuggingFace model directory (containing config.json + safetensors)
    #[arg(long)]
    model_dir: PathBuf,

    /// Text prompt (will be tokenized)
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "8")]
    max_new_tokens: usize,

    /// HIP device ordinal
    #[arg(long, default_value = "0")]
    device: usize,

    /// Run PyTorch oracle and compare logits
    #[arg(long)]
    validate: bool,

    /// Oracle dtype (bf16 or fp32)
    #[arg(long, default_value = "bf16")]
    oracle_dtype: String,

    /// HuggingFace model ID (for oracle; defaults to model_dir basename)
    #[arg(long)]
    model_id: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ordinal = cli.device;

    // Load config
    let config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
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

    // Tokenize
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());

    // Load weights
    let t0 = Instant::now();
    let weights = qwen35::weights::Qwen35Weights::load(&cli.model_dir, &text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("load weights: {e}"))?;
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    // Create decode engine
    let mut engine = DecodeEngine::new(weights, ordinal)?;

    // Run oracle for prefill (and optionally validation)
    let model_id = cli
        .model_id
        .clone()
        .unwrap_or_else(|| format!("Qwen/Qwen3.5-0.8B"));
    let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("oracle/run_oracle.py");

    let oracle_output = oracle::run_oracle(
        &oracle_script,
        &model_id,
        &prompt_ids,
        cli.max_new_tokens,
        &cli.oracle_dtype,
        true, // always emit state for prefill
    )?;

    // Load prefill state into GPU
    engine.load_prefill_state(&oracle_output)?;
    eprintln!("[engine] prefill state loaded to GPU");

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;

    // First token comes from prefill argmax
    let mut next_token = oracle_output.generated_token_ids[0];

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        let seqlen_offset = seqlen_start + step;
        let logits = engine.decode_step(next_token, seqlen_offset)?;

        // Validate against oracle if available
        if cli.validate && step < oracle_output.decode_logits.len() {
            let oracle_logits = &oracle_output.decode_logits[step];
            let delta = validate::max_abs_delta(&logits, oracle_logits);
            if delta > max_delta {
                max_delta = delta;
            }
            eprintln!(
                "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
            );
        }

        let sampled = DecodeEngine::greedy_sample(&logits);
        generated_ids.push(next_token);
        next_token = sampled;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    // Decode generated tokens to text
    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;

    println!("{text}");
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_tok={:.0} decode_max_delta={max_delta:.4}",
        prompt_ids.len(),
        generated_ids.len(),
        decode_ms / generated_ids.len() as f64,
    );

    Ok(())
}
