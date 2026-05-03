//! VMM-backed vs dense BF16 KV bit-exact smoke for Qwen3.6-35B-A3B.
//!
//! Same prompt, same args, run via the supersonic binary with dense KV
//! (`SUPERSONIC_VMM_KV=0`), forced VMM (`SUPERSONIC_VMM_KV=1`), and HIP
//! default VMM (`SUPERSONIC_VMM_KV` unset). Last-step logits must be
//! bit-exact because VMM changes only the cache backing.
//!
//! Skipped silently when:
//!  - HIP backend is not compiled
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR is unset or missing
//!  - HIP VMM is unsupported on the live device
//!  - SUPERSONIC_QWEN36_BF16_KV_VMM_SMOKE=0

use gpu_hal::Backend;
use std::process::Command;

struct RunResult {
    logits: Vec<f32>,
    combined_output: String,
}

fn run_supersonic_capture_logits(
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> anyhow::Result<RunResult> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args(args);
    cmd.arg("--dump-last-logits");
    cmd.env_remove("SUPERSONIC_VMM_KV");
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
    let stderr = String::from_utf8(out.stderr)?;
    let line = stdout
        .lines()
        .find(|l| l.starts_with("LAST_LOGITS:"))
        .ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found in stdout"))?;
    let csv = &line["LAST_LOGITS:".len()..];
    let logits = csv
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(Into::into))
        .collect::<anyhow::Result<Vec<f32>>>()?;
    Ok(RunResult {
        logits,
        combined_output: format!("{stdout}\n{stderr}"),
    })
}

#[test]
fn qwen36_moe_bf16_kv_vmm_dense_bit_exact() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_QWEN36_BF16_KV_VMM_SMOKE").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_QWEN36_BF16_KV_VMM_SMOKE=0");
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
        "--model",
        "qwen3.6-35b-a3b",
        "--model-dir",
        model_dir.as_str(),
        "--int4",
        "--prompt",
        "The stable virtual address for the key value cache is",
        "--max-new-tokens",
        "8",
    ];

    let dense =
        run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "0")]).expect("dense decode");
    let vmm =
        run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "1")]).expect("vmm decode");
    let auto_vmm = run_supersonic_capture_logits(&args, &[]).expect("auto vmm decode");

    assert!(
        vmm.combined_output
            .contains("[vmm] Qwen3.6-MoE BF16 KV active"),
        "VMM run did not report Qwen3.6 BF16 KV VMM activation:\n{}",
        vmm.combined_output
    );
    assert!(
        auto_vmm
            .combined_output
            .contains("[vmm] Qwen3.6-MoE BF16 KV active"),
        "unset-env HIP run did not auto-enable Qwen3.6 BF16 KV VMM:\n{}",
        auto_vmm.combined_output
    );
    assert_eq!(
        dense.logits.len(),
        vmm.logits.len(),
        "logits length mismatch"
    );
    assert_eq!(
        dense.logits.len(),
        auto_vmm.logits.len(),
        "auto-vmm logits length mismatch"
    );
    for (i, (a, b)) in dense.logits.iter().zip(&vmm.logits).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "VMM-backed BF16 KV logits diverged at index {i}: dense={a} vmm={b}",
        );
    }
    for (i, (a, b)) in dense.logits.iter().zip(&auto_vmm.logits).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "auto VMM-backed BF16 KV logits diverged at index {i}: dense={a} auto_vmm={b}",
        );
    }
}
