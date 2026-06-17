//! End-to-end parity gate for the Qwen 3.6 MoE batched-Q prefill path.
//!
//! M13: batched prefill is now the default for Qwen 3.6 MoE. This test
//! runs `supersonic` twice on the same prompt — once with the LEGACY
//! per-token persistent-decode prefill loop forced via
//! `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0`, once with the default
//! (no env vars set, batched path active) — and compares the dumped
//! post-prefill last-token logits.
//!
//! Subprocess-spawn pattern lives at:
//!   crates/runner/tests/specprefill_qwen36_moe_cosine_parity.rs
//!
//! Skipped silently when:
//!   - HIP backend not compiled
//!   - SUPERSONIC_QWEN36_35B_A3B_DIR unset or path missing

use gpu_hal::Backend;
use std::process::Command;

fn run_supersonic_capture_logits(
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> anyhow::Result<Vec<f32>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args(args);
    cmd.arg("--dump-last-logits");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        anyhow::bail!(
            "supersonic exited {}: stderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8(out.stdout)?;
    let line = stdout
        .lines()
        .find(|l| l.starts_with("LAST_LOGITS:"))
        .ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found in stdout"))?;
    let csv = &line["LAST_LOGITS:".len()..];
    csv.trim()
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(Into::into))
        .collect()
}

fn cossim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
            if x > a.1 {
                (i, x)
            } else {
                a
            }
        })
        .0
}

#[test]
fn batched_prefill_matches_per_token() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let target = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset/missing");
            return;
        }
    };

    // ~80-token prompt — long enough to exercise multiple chunks at the
    // default chunk size while keeping per-run wall-clock < 60s. Same prompt
    // as specprefill_qwen36_moe_cosine_parity.rs for comparability.
    let prompt = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, allowing models to weigh the importance of different parts of the input sequence dynamically. Unlike recurrent networks, transformers can process all tokens in parallel during training, making them highly efficient on modern accelerator hardware. The overall result is";

    let common: Vec<&str> = vec![
        "--backend",
        "hip",
        "--model",
        "qwen3.6-35b-a3b",
        "--model-dir",
        &target,
        "--prompt",
        prompt,
        "--max-new-tokens",
        "1",
    ];

    // M13: explicitly disable all three stages to get the LEGACY per-token
    // persistent-decode behavior (the pre-PR path).
    let baseline = run_supersonic_capture_logits(
        &common,
        &[
            ("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL", "0"),
            ("SUPERSONIC_QWEN36_MOE_BATCHED_ATTN", "0"),
            ("SUPERSONIC_QWEN36_MOE_GROUPED_FFN", "0"),
        ],
    )
    .expect("baseline (per-token, env=0 forced)");
    // No env vars set → the new default: batched prefill + batched attn +
    // grouped FFN all active.
    let batched = run_supersonic_capture_logits(&common, &[]).expect("batched (default)");

    assert_eq!(
        baseline.len(),
        batched.len(),
        "logits length mismatch: baseline={} batched={}",
        baseline.len(),
        batched.len()
    );

    let cs = cossim(&baseline, &batched);
    let am_b = argmax(&baseline);
    let am_n = argmax(&batched);

    // Bar matches the codebase's INT4/BF16 noise floor for "different
    // fused-op shapes, same math" parity. cossim >= 0.999 is what
    // qwen36_moe_kv_fp8_parity uses for KV-FP8 vs BF16-KV; the M6.2
    // batched path lands ~0.9996 because the prefill primitives BF16-round
    // at slightly different points than the per-token persistent megakernel
    // (q_norm/k_norm intermediates, sigmoid·gate, RoPE table interpolation).
    // Argmax must still match — that's the load-bearing bar for greedy decode.
    assert!(
        cs >= 0.999,
        "cossim {:.6} < 0.999 (per-token vs batched diverged beyond INT4/BF16 noise)",
        cs
    );
    assert_eq!(
        am_b, am_n,
        "argmax mismatch: per-token={} batched={}",
        am_b, am_n
    );
}
