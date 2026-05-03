//! KV-FP8 vs BF16-KV self-parity for Qwen3.6-35B-A3B (--int4 weights, gfx1100).
//!
//! Runs the same prompt twice through the supersonic binary — once with
//! --kv-fp8, once without — and asserts the last-step logits cossim
//! >= 0.999. Skipped silently when:
//!  - HIP backend not compiled (gpu_hal::is_backend_compiled returns false)
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR not set or path missing
//!  - SUPERSONIC_QWEN36_KV_FP8_PARITY=0

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
fn qwen36_moe_kv_fp8_vs_bf16_self_parity() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_QWEN36_KV_FP8_PARITY").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_QWEN36_KV_FP8_PARITY=0");
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
        "--model",
        "qwen3.6-35b-a3b",
        "--model-dir",
        model_dir.as_str(),
        "--int4",
        "--prompt",
        "The cosine similarity of two parallel decoding streams is",
        "--max-new-tokens",
        "16",
    ];

    let bf16_kv: Vec<f32> =
        run_supersonic_capture_logits(&common, &[]).expect("BF16-KV decode");
    let mut fp8_args = common.clone();
    fp8_args.push("--kv-fp8");
    let fp8_kv: Vec<f32> =
        run_supersonic_capture_logits(&fp8_args, &[]).expect("FP8-KV decode");

    assert_eq!(
        bf16_kv.len(),
        fp8_kv.len(),
        "logits length mismatch ({} vs {})",
        bf16_kv.len(),
        fp8_kv.len(),
    );
    let dot: f64 = bf16_kv
        .iter()
        .zip(&fp8_kv)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = bf16_kv
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = fp8_kv
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum::<f64>()
        .sqrt();
    let cossim = dot / (na * nb);
    eprintln!("[kv-fp8 self-parity] cossim = {:.6}", cossim);
    assert!(
        cossim >= 0.999,
        "KV-FP8 vs BF16-KV cossim {cossim} < 0.999"
    );
}
