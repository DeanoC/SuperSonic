use std::io::Write as _;

use anyhow::{Context, Result};
use supersonic_runtime::qwen36_moe::engine::Qwen36MoeLoadEvidence;

use crate::qwen36_moe_cli::timing::SamplingParams;
use crate::qwen36_moe_logits::bf16_bytes_to_f32;

pub(crate) fn print_runtime_engine_load_evidence(evidence: &Qwen36MoeLoadEvidence) {
    eprintln!("[qwen36-moe] FLM weight mode: INT4 native FLM");
    eprintln!("[qwen36-moe] FLM direct plans: {}", evidence.direct_profile);
    println!(
        "[qwen36-moe] loading weights from already-open FLM source at {} (INT4 native FLM)",
        evidence.flm_path.display()
    );
    println!(
        "[FLM runtime weights] ready-for-decode: YES (source={} direct_plans={})",
        evidence.flm_path.display(),
        evidence.direct_profile
    );
    println!("[runtime residency]");
    println!(
        "  resident allocations: {}",
        evidence.resident_allocation_count
    );
    println!(
        "  mapped virtual ranges: {}",
        evidence.mapped_virtual_ranges.len()
    );
    eprintln!(
        "[qwen36-moe] runtime engine ready: load_sequence={} source_open_count={}",
        evidence.load_sequence, evidence.source_open_count
    );
}

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

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn bf16_head_hex(bytes: &[u8], elems: usize) -> String {
    bytes
        .chunks_exact(2)
        .take(elems)
        .map(|chunk| format!("{:02x}{:02x}", chunk[1], chunk[0]))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn emit_final_hidden_tap_if_requested(
    step: usize,
    gen_index: usize,
    position: i32,
    path: &str,
    lm_head_folded: bool,
    final_hidden_bytes: &[u8],
) {
    if std::env::var_os("SUPERSONIC_QWEN36_FINAL_HIDDEN_TAP").is_none()
        && std::env::var_os("SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP").is_none()
    {
        return;
    }
    let hidden = bf16_bytes_to_f32(final_hidden_bytes);
    let mut l2 = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (idx, &value) in hidden.iter().enumerate() {
        l2 += (value as f64) * (value as f64);
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_idx = idx;
        }
    }
    eprintln!(
        "[qwen36-final-hidden-tap] step={} gen_index={} position={} path={} lm_head_folded={} elems={} checksum={:016x} l2={:.8e} max_abs={:.8e} max_abs_idx={} head8={}",
        step,
        gen_index,
        position,
        path,
        lm_head_folded as u8,
        hidden.len(),
        fnv1a64(final_hidden_bytes),
        l2.sqrt(),
        max_abs,
        max_abs_idx,
        bf16_head_hex(final_hidden_bytes, 8),
    );
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

pub(crate) fn emit_logits_tap_if_requested(
    step: usize,
    gen_index: usize,
    position: i32,
    path: &str,
    lm_head_folded: bool,
    logits: &[u8],
) {
    if std::env::var_os("SUPERSONIC_QWEN36_LOGITS_TAP").is_none()
        && std::env::var_os("SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP").is_none()
    {
        return;
    }
    let logits_f32 = bf16_bytes_to_f32(logits);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(5);
    for (idx, &value) in logits_f32.iter().enumerate() {
        let insert_at = top
            .iter()
            .position(|&(_, existing)| value > existing)
            .unwrap_or(top.len());
        if insert_at < 5 {
            top.insert(insert_at, (idx, value));
            top.truncate(5);
        }
    }
    let (top1_idx, top1_val) = top.first().copied().unwrap_or((usize::MAX, f32::NAN));
    let top5 = top
        .iter()
        .map(|(idx, value)| format!("{}:{:.8e}", idx, value))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[qwen36-logits-tap] step={} gen_index={} position={} path={} lm_head_folded={} elems={} checksum={:016x} top1_idx={} top1_val={:.8e} top5={}",
        step,
        gen_index,
        position,
        path,
        lm_head_folded as u8,
        logits_f32.len(),
        fnv1a64(logits),
        top1_idx,
        top1_val,
        top5,
    );
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
