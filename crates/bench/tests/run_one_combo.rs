use std::path::PathBuf;
use supersonic_bench::perf::{run_one_combo, ComboInvocation, RunPolicy};

fn fake_supersonic_script(tmp: &std::path::Path, ms_per_step: f64) -> PathBuf {
    let path = tmp.join("supersonic");
    let body = format!(
        r#"#!/usr/bin/env bash
echo "[result] prompt_tokens=6 generated_tokens=16 decode_ms=128 ms_per_step={ms_per_step}"
"#
    );
    std::fs::write(&path, body).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    use std::os::unix::fs::PermissionsExt;
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

#[test]
fn run_one_combo_takes_median_of_three() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = fake_supersonic_script(tmp.path(), 12.5);
    let invocation = ComboInvocation {
        binary: bin,
        backend: None,
        model: "qwen3.5-0.8b".into(),
        model_dir: PathBuf::from("/nonexistent"),
        quant: "bf16".into(),
        specprefill_draft_dir: None,
        prompt: "The quick brown fox jumps over".into(),
        max_new_tokens: 16,
        warmup_tokens: 2,
    };
    let policy = RunPolicy {
        measurement_runs: 3,
        cooldown_seconds: 0,
    };
    let cell = run_one_combo(&invocation, &policy).unwrap();
    use supersonic_bench::runs::PerfStatus;
    match cell.status {
        PerfStatus::Ok {
            ms_per_step,
            samples,
            ..
        } => {
            assert_eq!(samples.len(), 3);
            assert!((ms_per_step - 12.5).abs() < 1e-6, "median should be 12.5");
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn run_one_combo_records_error_on_missing_binary() {
    let invocation = ComboInvocation {
        binary: PathBuf::from("/nonexistent/supersonic"),
        backend: None,
        model: "qwen3.5-0.8b".into(),
        model_dir: PathBuf::from("/nonexistent"),
        quant: "bf16".into(),
        specprefill_draft_dir: None,
        prompt: "x".into(),
        max_new_tokens: 1,
        warmup_tokens: 1,
    };
    let policy = RunPolicy {
        measurement_runs: 1,
        cooldown_seconds: 0,
    };
    let cell = run_one_combo(&invocation, &policy).unwrap();
    use supersonic_bench::runs::PerfStatus;
    assert!(
        matches!(cell.status, PerfStatus::Error { .. }),
        "expected Error status on missing binary, got {:?}",
        cell.status
    );
}
