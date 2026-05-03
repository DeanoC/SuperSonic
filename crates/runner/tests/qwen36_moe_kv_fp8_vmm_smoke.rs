//! VMM-backed vs dense FP8 KV bit-exact smoke for Qwen3.6-35B-A3B
//! (--int4 weights + --kv-fp8, gfx1100).
//!
//! Same prompt, same args, run twice via the supersonic binary — once
//! with SUPERSONIC_VMM_KV=1, once with SUPERSONIC_VMM_KV=0. Last-step
//! logits must be bit-exact (not just cossim ≥ 0.999): VMM-backed KV
//! is a different *backing*, not a different *quantisation*, so kernel
//! reads should be identical to the byte.
//!
//! Skipped silently when:
//!  - HIP backend not compiled
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR unset or path missing
//!  - HIP backend doesn't support VMM on the live device
//!  - SUPERSONIC_QWEN36_KV_FP8_VMM_SMOKE=0

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
fn qwen36_moe_kv_fp8_vmm_dense_bit_exact() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_QWEN36_KV_FP8_VMM_SMOKE").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_QWEN36_KV_FP8_VMM_SMOKE=0");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };
    if !gpu_hal::vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skipped: VMM not supported on this device");
        return;
    }

    let args = vec![
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", model_dir.as_str(),
        "--int4", "--kv-fp8",
        "--prompt", "Hello, world.",
        "--max-new-tokens", "8",
    ];

    let dense = run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "0")])
        .expect("dense decode");
    let vmm = run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "1")])
        .expect("vmm decode");

    assert_eq!(dense.len(), vmm.len(), "logits length mismatch");
    for (i, (a, b)) in dense.iter().zip(&vmm).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "VMM-backed KV-FP8 logits diverged at index {i}: dense={a} vmm={b}",
        );
    }
}
