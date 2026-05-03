//! Smoke test for `--fp8-runtime --kv-fp8` on Qwen3.6-35B-A3B / gfx942.
//! Skipped at runtime on every other arch (compiles on gfx1100).
//!
//! Asserts logits cossim ≥ 0.99 between INT4 + KV-FP8 and FP8 + KV-FP8
//! runs (same prompt, same args). Cossim is looser than the .999 used
//! for KV-only parity because both INT4 and FP8 are lossy weight
//! quantisations vs BF16; this is a self-consistency floor that
//! catches gross numerical bugs in the FP8-weights path.

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

fn live_arch_is_gfx942() -> bool {
    // Replace with whatever the runner already uses to detect arch at
    // startup if a cleaner helper is available. The shell-out is a
    // no-frills fallback that works in the test environment.
    match std::process::Command::new("rocminfo").output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).contains("gfx942"),
        Err(_) => false,
    }
}

#[test]
fn qwen36_moe_fp8_weights_kv_fp8_gfx942() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if !live_arch_is_gfx942() {
        eprintln!("skipped: not running on gfx942");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };

    let common = vec![
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", model_dir.as_str(),
        "--kv-fp8",
        "--prompt", "When the cooperative kernel grids meet,",
        "--max-new-tokens", "16",
    ];
    let mut int4 = common.clone();
    int4.push("--int4");
    let mut fp8 = common.clone();
    fp8.push("--fp8-runtime");

    let int4_logits = run_supersonic_capture_logits(&int4, &[]).expect("int4 + kv-fp8");
    let fp8_logits = run_supersonic_capture_logits(&fp8, &[]).expect("fp8 + kv-fp8");
    assert_eq!(int4_logits.len(), fp8_logits.len());
    let dot: f64 = int4_logits
        .iter()
        .zip(&fp8_logits)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = int4_logits
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = fp8_logits
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum::<f64>()
        .sqrt();
    let cossim = dot / (na * nb);
    eprintln!("[fp8-weights smoke gfx942] cossim INT4↔FP8 = {:.6}", cossim);
    assert!(cossim >= 0.99, "FP8 vs INT4 cossim {cossim} < 0.99");
}
