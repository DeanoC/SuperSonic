use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[cfg(target_os = "macos")]
fn model_dir() -> Option<String> {
    std::env::var("SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR").ok()
}

#[cfg(target_os = "macos")]
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("runner crate should live under <repo>/crates/runner")
        .to_path_buf()
}

#[cfg(target_os = "macos")]
fn bughunt_manifest() -> PathBuf {
    repo_root().join("crates/runner/bughunt/qwen35_metal_manifest.json")
}

#[cfg(target_os = "macos")]
fn augmented_path() -> std::ffi::OsString {
    let repo_root = repo_root();
    let venv_bin = repo_root.join(".venv/bin");
    let mut path_entries = Vec::new();
    if venv_bin.exists() {
        path_entries.push(venv_bin);
    }
    path_entries.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    std::env::join_paths(path_entries).expect("join PATH entries")
}

#[cfg(target_os = "macos")]
fn temp_report_path(mode: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_millis();
    std::env::temp_dir().join(format!(
        "qwen35_bughunt_{}_{}_{}.json",
        mode,
        std::process::id(),
        timestamp
    ))
}

#[cfg(target_os = "macos")]
fn run_bughunt(mode: &str, extra_args: &[&str]) -> (std::process::Output, PathBuf) {
    let Some(model_dir) = model_dir() else {
        panic!("SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR must be set for ignored bughunt smoke tests");
    };

    let report_path = temp_report_path(mode);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qwen35_bughunt"));
    cmd.env("PATH", augmented_path()).args([
        "--model-dir",
        &model_dir,
        "--backend",
        "metal",
        "--oracle-device",
        "cpu",
        "--mode",
        mode,
        "--prompt-manifest",
        bughunt_manifest()
            .to_str()
            .expect("manifest path must be valid UTF-8"),
        "--prompt",
        "code_prompt",
        "--report-json",
        report_path
            .to_str()
            .expect("report path must be valid UTF-8"),
    ]);
    cmd.args(extra_args);

    let output = cmd.output().expect("run qwen35_bughunt");
    (output, report_path)
}

#[cfg(target_os = "macos")]
fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).expect("read report JSON"))
        .expect("parse report JSON")
}

#[cfg(target_os = "macos")]
fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn qwen35_bughunt_gate_reports_known_failure() {
    let (output, report_path) = run_bughunt("gate", &[]);
    let combined = combined_output(&output);
    assert!(
        !output.status.success(),
        "gate should fail on the known-bad code_prompt:\n{}",
        combined
    );
    assert!(report_path.exists(), "gate report was not written");

    let report = read_json(&report_path);
    assert_eq!(report["mode"], "gate");
    assert_eq!(report["gate"]["pass"], false);
    assert_eq!(report["gate"]["prompt_results"][0]["name"], "code_prompt");
    assert_eq!(report["gate"]["prompt_results"][0]["pass"], false);
    assert!(
        report["gate"]["prompt_results"][0]["prefill_logit_max_abs"]
            .as_f64()
            .unwrap_or(0.0)
            > 0.0
    );
    assert!(
        combined.contains("mode=gate"),
        "expected concise gate summary:\n{}",
        combined
    );
    let _ = fs::remove_file(report_path);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn qwen35_bughunt_localize_emits_checkpoint_summary() {
    let (output, report_path) = run_bughunt("localize", &[]);
    let combined = combined_output(&output);
    assert!(
        !output.status.success(),
        "localize should return non-zero on a failing prompt:\n{}",
        combined
    );
    assert!(report_path.exists(), "localize report was not written");

    let report = read_json(&report_path);
    assert_eq!(report["mode"], "localize");
    assert_eq!(report["localize"]["pass"], false);
    assert_eq!(report["localize"]["gate_prompt"]["name"], "code_prompt");
    assert!(
        report["localize"]["localization"]["initial_suspicious_layer"]
            .as_u64()
            .is_some()
    );
    assert!(report["localize"]["localization"]["chosen_traced_layer"]
        .as_u64()
        .is_some());
    assert!(
        combined.contains("mode=localize"),
        "expected concise localize summary:\n{}",
        combined
    );
    let _ = fs::remove_file(report_path);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn qwen35_bughunt_dump_emits_stage_metrics() {
    let (output, report_path) = run_bughunt(
        "dump",
        &["--position", "0", "--layer", "8", "--layer-kind", "linear"],
    );
    let combined = combined_output(&output);
    assert!(
        !output.status.success(),
        "dump inherits the failing prompt status for code_prompt:\n{}",
        combined
    );
    assert!(report_path.exists(), "dump report was not written");

    let report = read_json(&report_path);
    assert_eq!(report["mode"], "dump");
    assert_eq!(report["dump"]["pass"], false);
    assert_eq!(report["dump"]["dump"]["prompt_name"], "code_prompt");
    assert_eq!(report["dump"]["dump"]["position"], 0);
    assert_eq!(report["dump"]["dump"]["layer"], 8);
    assert_eq!(report["dump"]["dump"]["layer_kind"], "linear");
    assert!(report["dump"]["dump"]["traced_metrics"]["stages"]
        .as_array()
        .map(|stages| !stages.is_empty())
        .unwrap_or(false));
    assert!(
        combined.contains("stage="),
        "expected per-stage dump output:\n{}",
        combined
    );
    let _ = fs::remove_file(report_path);
}
