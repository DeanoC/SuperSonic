//! End-to-end parity gate for the Qwen 3.6 MoE batched-Q prefill path.
//!
//! Runs `supersonic` twice on the same prompt — once with the legacy
//! per-token persistent-decode prefill loop (env var unset), once with
//! `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=1` (the new chunked path).
//! Compares the dumped post-prefill last-token logits.
//!
//! While the new path is still a stub that delegates back to the
//! per-token kernels (Milestones 1–5), this test passes trivially —
//! it exists to lock in the regression gate before any real kernel
//! work lands. From Milestone 6 onward (when the orchestrator actually
//! runs the new batched kernels), this test becomes the load-bearing
//! correctness signal.
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
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
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
        "--backend", "hip",
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", &target,
        "--prompt", prompt,
        "--max-new-tokens", "1",
    ];

    let baseline = run_supersonic_capture_logits(&common, &[]).expect("baseline (per-token)");
    let batched = run_supersonic_capture_logits(
        &common,
        &[("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL", "1")],
    )
    .expect("batched");

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

    // Tight bars: while the env var is a no-op (M1 stub) these will be 1.0
    // and equal. From M6 onward the bars catch any divergence introduced by
    // the real batched kernels.
    assert!(
        cs >= 0.9999,
        "cossim {:.6} < 0.9999 (per-token vs batched diverged)",
        cs
    );
    assert_eq!(am_b, am_n, "argmax mismatch: per-token={} batched={}", am_b, am_n);
}
