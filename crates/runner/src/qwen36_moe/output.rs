use std::io::Write as _;

use anyhow::{Context, Result};

use crate::qwen36_moe_cli::timing::SamplingParams;
use crate::qwen36_moe_logits::bf16_bytes_to_f32;

pub(crate) fn print_decode_stream_start(
    tokenizer: Option<&tokenizers::Tokenizer>,
    prompt_ids: &[u32],
    max_new: usize,
) {
    println!(
        "  decoding {} prompt token{} + generating ≤{} new token{}…",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        max_new,
        if max_new == 1 { "" } else { "s" },
    );
    println!();
    print!("> ");
    if let Some(tok) = tokenizer {
        if let Ok(prompt_text) = tok.decode(prompt_ids, false) {
            print!("{prompt_text}");
        }
    }
    std::io::stdout().flush().ok();
}

pub(crate) fn print_sampling_summary(sampling: SamplingParams) {
    println!(
        "  sampling: temp={} top_k={} top_p={} seed={}",
        sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed,
    );
}

pub(crate) fn print_decoded_token(tokenizer: Option<&tokenizers::Tokenizer>, token: u32) {
    if let Some(tok) = tokenizer {
        if let Ok(text) = tok.decode(&[token], false) {
            print!("{text}");
        }
    }
}

pub(crate) fn dump_final_hidden_if_requested(
    step: usize,
    position: i32,
    final_hidden_bytes: &[u8],
) -> Result<()> {
    let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN") else {
        return Ok(());
    };
    std::fs::write(&dump_path, final_hidden_bytes)
        .with_context(|| format!("write final_hidden dump to {dump_path}"))?;
    eprintln!(
        "[debug] dumped step={step} position={position} final_hidden ({} BF16 bytes) to {dump_path}",
        final_hidden_bytes.len()
    );
    Ok(())
}

pub(crate) fn dump_logits_if_requested(step: usize, logits: &[u8]) -> Result<()> {
    let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_LOGITS") else {
        return Ok(());
    };
    std::fs::write(&dump_path, logits)
        .with_context(|| format!("write logits dump to {dump_path}"))?;
    eprintln!(
        "[debug] dumped step={step} logits ({} BF16 bytes) to {dump_path}",
        logits.len()
    );
    Ok(())
}

pub(crate) fn print_last_logits_if_requested(dump_last_logits: bool, last_logits_bytes: &[u8]) {
    if !dump_last_logits || last_logits_bytes.is_empty() {
        return;
    }

    let logits_f32 = bf16_bytes_to_f32(last_logits_bytes);
    // Lead with `\n` so the marker lands at the start of its own line: the
    // streamed-token path uses `print!` without a trailing newline.
    print!("\nLAST_LOGITS: ");
    for (i, x) in logits_f32.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        // Display preserves enough precision for bit-exact smoke parsers.
        print!("{}", x);
    }
    println!();
    std::io::stdout().flush().ok();
}

pub(crate) fn print_generation_summary(
    generated_ids: &[u32],
    prompt_len: usize,
    eos_id: Option<u32>,
    decode_ms: Option<f64>,
) {
    println!();
    println!();
    println!(
        "Generated {} token{} ({} prompt + {} new). EOS: {}.",
        generated_ids.len(),
        if generated_ids.len() == 1 { "" } else { "s" },
        prompt_len,
        generated_ids.len(),
        if eos_id
            .map(|e| generated_ids.last() == Some(&e))
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no (max_new_tokens hit)"
        },
    );
    if !generated_ids.is_empty() {
        println!("  Generated ids: {generated_ids:?}");
    }
    if let Some(decode_ms) = decode_ms {
        let ms_per_step = if generated_ids.is_empty() {
            0.0
        } else {
            decode_ms / generated_ids.len() as f64
        };
        println!(
            "[result] prompt_tokens={} generated_tokens={} decode_ms={:.0} ms_per_step={:.1}",
            prompt_len,
            generated_ids.len(),
            decode_ms,
            ms_per_step,
        );
    } else {
        println!(
            "[result] prompt_tokens={} generated_tokens={}",
            prompt_len,
            generated_ids.len()
        );
    }
}
