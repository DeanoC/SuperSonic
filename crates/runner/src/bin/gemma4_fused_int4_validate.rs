//! Fused-INT4 greedy-decode validator for Gemma 4 (Step 28).
//!
//! Drives the same `Gemma4Int4Engine` through two decode paths:
//!   1. `decode_step`           — attention block runs as one `fused_attn_block_int4`
//!                                launch (Step 28, the new hot path).
//!   2. `decode_step_primitive` — attention block runs as the pre-Step-28
//!                                10-primitive chain.
//!
//! Each path owns its own `Gemma4Int4Engine` so K/V caches don't interfere,
//! then both run the same prompt + greedy-decode loop. The only math
//! difference between the paths is *when* BF16 rounding happens on the
//! intermediate Q/K/V/O proj outputs — primitive-chain writes BF16 after
//! every op, fused keeps the intermediates in F32 through the attention
//! phase. That can shift marginal tokens by ≤2 ULP per op, so the harness
//! compares **token IDs** (not logit cosine similarity) and exits non-zero on
//! any mismatch; the expectation is bit-identical token sequences on the
//! validated bakes, same as the BF16 `gemma4_fused_decode_validate` passes.
//!
//! Usage:
//!   cargo run --release --bin gemma4_fused_int4_validate -- \
//!     --model-dir <checkpoint> [--prompt TEXT] [--max-new-tokens N]

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use tokenizers::Tokenizer;

#[path = "../gemma4_int4_engine.rs"]
mod gemma4_int4_engine;
use gemma4_int4_engine::{int4_bake_ok, Gemma4Int4Engine};

#[derive(Parser, Debug)]
#[command(about = "Validate Gemma 4 fused-INT4 decode against primitive-chain INT4")]
struct Cli {
    /// Path to a Gemma 4 checkpoint directory (config.json + tokenizer.json +
    /// `.supersonic/v1-int4-gptq/`).
    #[arg(long)]
    model_dir: PathBuf,
    /// Prompt to decode.
    #[arg(long, default_value = "Hello")]
    prompt: String,
    /// Number of new tokens to generate on each path.
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
    /// Weight prefix inside the bake (matches registry default).
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,
}

fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best as u32
}

fn run_path(
    label: &str,
    model_dir: &std::path::Path,
    weight_prefix: &str,
    max_t: usize,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    fused: bool,
) -> Result<Vec<u32>> {
    let start = std::time::Instant::now();
    let mut engine = Gemma4Int4Engine::load(model_dir, weight_prefix, max_t, 0)
        .with_context(|| format!("load engine ({label})"))?;
    eprintln!(
        "[{label}] engine loaded in {} ms",
        start.elapsed().as_millis()
    );

    let prefill_start = std::time::Instant::now();
    let prefill_logits = engine.prefill(prompt_tokens)?;
    eprintln!(
        "[{label}] prefill ({} tok) in {} ms",
        prompt_tokens.len(),
        prefill_start.elapsed().as_millis()
    );

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let tok0 = greedy(&prefill_logits);
    generated.push(tok0);

    let decode_start = std::time::Instant::now();
    let mut current = tok0;
    for step in 1..max_new_tokens {
        let pos = prompt_tokens.len() + step - 1;
        let logits = if fused {
            engine.decode_step(current, pos)?
        } else {
            engine.decode_step_primitive(current, pos)?
        };
        let next = greedy(&logits);
        generated.push(next);
        current = next;
    }
    let decode_ms = decode_start.elapsed().as_millis();
    let steps = (max_new_tokens - 1).max(1);
    eprintln!(
        "[{label}] decode {} step(s) in {} ms ({} ms/tok)",
        max_new_tokens - 1,
        decode_ms,
        decode_ms / (steps as u128),
    );

    Ok(generated)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !int4_bake_ok(&cli.model_dir) {
        bail!(
            "No INT4 bake at {}/.supersonic/. Run:\n  python oracle/bake_int4_gemma4.py --model-dir {}",
            cli.model_dir.display(),
            cli.model_dir.display()
        );
    }

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
    let encoded = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();
    if prompt_tokens.is_empty() {
        bail!("tokenizer produced zero tokens for prompt {:?}", cli.prompt);
    }
    let max_t = prompt_tokens.len() + cli.max_new_tokens + 2;

    eprintln!(
        "[prompt] {:?} -> {} token(s), max_new_tokens={}, max_t={}",
        cli.prompt,
        prompt_tokens.len(),
        cli.max_new_tokens,
        max_t
    );

    let fused_tokens = run_path(
        "fused", &cli.model_dir, &cli.weight_prefix,
        max_t, &prompt_tokens, cli.max_new_tokens, true,
    )?;
    let primitive_tokens = run_path(
        "primitive", &cli.model_dir, &cli.weight_prefix,
        max_t, &prompt_tokens, cli.max_new_tokens, false,
    )?;

    eprintln!("[fused]     tokens = {:?}", fused_tokens);
    eprintln!("[primitive] tokens = {:?}", primitive_tokens);

    let mismatches: Vec<usize> = fused_tokens
        .iter()
        .zip(primitive_tokens.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect();

    if mismatches.is_empty() {
        println!("[result] MATCH: {} token(s) identical", fused_tokens.len());
        Ok(())
    } else {
        for i in &mismatches {
            eprintln!(
                "  step {i}: fused={} primitive={}",
                fused_tokens[*i], primitive_tokens[*i]
            );
        }
        println!(
            "[result] MISMATCH at {} step(s) (of {})",
            mismatches.len(),
            fused_tokens.len()
        );
        bail!("fused-INT4 vs primitive-chain-INT4 token mismatch")
    }
}
