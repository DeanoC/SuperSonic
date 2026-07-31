//! End-to-end parity gate for qualified Qwen 3.6 MoE HIP prefill dispatch.
//!
//! The HIP native-INT4 batched path remains available behind
//! `SUPERSONIC_QWEN36_ENABLE_HIP_OPTIMIZED_PREFILL=1`, but its real-model
//! `0.999 + argmax` gate is red. Default dispatch must therefore reproduce
//! the explicit per-token owner exactly.
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

#[test]
fn qualified_hip_prefill_matches_per_token() {
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

    let baseline =
        run_supersonic_capture_logits(&common, &[("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL", "0")])
            .expect("baseline (per-token, env=0 forced)");
    let qualified = run_supersonic_capture_logits(&common, &[]).expect("qualified HIP default");

    assert_eq!(
        baseline.len(),
        qualified.len(),
        "logits length mismatch: baseline={} qualified={}",
        baseline.len(),
        qualified.len()
    );
    assert_eq!(
        baseline, qualified,
        "qualified HIP default must be bit-exact to explicit per-token prefill"
    );
}
