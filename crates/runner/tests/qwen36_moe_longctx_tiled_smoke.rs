//! Long-context tiled full-attention smoke for Qwen3.6-35B-A3B.
//!
//! This forces the persistent full-attention path past the 384-token tile
//! threshold introduced in the long-context perf pass. It is intentionally
//! model-backed and skips when the local Qwen3.6 directory is unavailable.

use gpu_hal::Backend;
use std::process::Command;

fn longctx_prompt(context_tokens: usize, seed: usize) -> String {
    const TOKEN_CHARS: usize = 7;
    let needle = format!("SSB-NEEDLE-{:05}", seed % 100000);
    let intro = "You are reading a long diagnostic log. Remember the secret value if one appears. ";
    let question =
        "\nQuestion: What is the secret value? Answer with only the secret value.\nAnswer:";
    let target_chars = std::cmp::max(256, context_tokens * TOKEN_CHARS);
    let filler_words = [
        "cache",
        "route",
        "kernel",
        "tensor",
        "context",
        "expert",
        "prefetch",
        "latency",
        "resident",
        "decode",
        "attention",
        "wave",
    ];

    let mut chunks = vec![intro.to_string()];
    let mut chars = intro.len();
    let mut i = 0usize;
    let halfway = target_chars / 2;
    let mut inserted = false;
    while chars < target_chars.saturating_sub(question.len()) {
        let chunk = if !inserted && chars >= halfway {
            inserted = true;
            format!(" The secret value is {needle}. ")
        } else {
            let word = filler_words[(i + seed) % filler_words.len()];
            i += 1;
            format!("{word}-{:03} ", i.saturating_sub(1) % 997)
        };
        chars += chunk.len();
        chunks.push(chunk);
    }
    if !inserted {
        chunks.push(format!(" The secret value is {needle}. "));
    }
    chunks.push(question.to_string());
    chunks.concat()
}

fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn generated_ids(output: &str) -> Vec<u32> {
    let line = output
        .lines()
        .find(|line| line.trim_start().starts_with("Generated ids: "))
        .expect("supersonic output should contain generated ids");
    let (_, ids) = line
        .split_once(':')
        .expect("generated ids line should contain ':'");
    serde_json::from_str(ids.trim()).expect("generated ids should parse as JSON")
}

fn prefill_steps(output: &str) -> usize {
    let line = output
        .lines()
        .find(|line| line.contains("[qwen36-moe lifecycle-timings]"))
        .expect("supersonic output should contain qwen36 lifecycle timings");
    for part in line.split_whitespace() {
        if let Some(raw) = part.strip_prefix("prefill_steps=") {
            return raw
                .parse::<usize>()
                .expect("prefill_steps should parse as usize");
        }
    }
    panic!("prefill_steps missing from lifecycle line: {line}");
}

#[test]
fn qwen36_moe_longctx_tiled_attention_deterministic_ids() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_QWEN36_LONGCTX_TILED_SMOKE").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_QWEN36_LONGCTX_TILED_SMOKE=0");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };

    let prompt = longctx_prompt(512, 20260504);
    let out = Command::new(env!("CARGO_BIN_EXE_supersonic"))
        .env("SUPERSONIC_BACKENDS", "hip")
        .args([
            "--backend",
            "hip",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            model_dir.as_str(),
            "--int4",
            "--prompt",
            prompt.as_str(),
            "--context-size",
            "512",
            "--max-new-tokens",
            "4",
            "--temperature",
            "0",
            "--top-k",
            "1",
            "--sampling-seed",
            "20260504",
            "--no-download",
            "--emit-stage-timings",
        ])
        .output()
        .expect("run supersonic qwen36 tiled longctx smoke");

    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "qwen36 tiled longctx smoke failed with status {:?}:\n{}",
        out.status.code(),
        combined
    );
    assert!(
        prefill_steps(&combined) > 384,
        "prompt should cross the tiled full-attention threshold:\n{}",
        combined
    );
    assert_eq!(
        generated_ids(&combined),
        vec![271, 271, 1178, 33],
        "longctx tiled smoke generated unexpected token IDs"
    );
}
