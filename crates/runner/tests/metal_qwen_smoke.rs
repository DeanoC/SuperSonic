use std::path::PathBuf;
use std::process::Command;

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn metal_qwen_smoke_runs_end_to_end() {
    let Some(model_dir) = std::env::var_os("SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR") else {
        eprintln!("skipping: SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR is not set");
        return;
    };

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("runner crate should live under <repo>/crates/runner");
    let venv_bin = repo_root.join(".venv/bin");
    let mut path_entries = Vec::new();
    if venv_bin.exists() {
        path_entries.push(venv_bin);
    }
    path_entries.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let path_value = std::env::join_paths(path_entries).expect("join PATH entries");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.env("PATH", path_value)
        .args([
            "--backend",
            "auto",
            "--model",
            "qwen3.5-0.8b",
            "--model-dir",
            model_dir.to_str().expect("model dir must be valid UTF-8"),
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "1",
        ]);

    if repo_root.join(".venv/bin/python3").exists() {
        cmd.args(["--validate", "--oracle-device", "cpu"]);
    }

    let output = cmd.output().expect("run supersonic metal smoke test");
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        output.status.success(),
        "supersonic metal smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("backend=Metal"),
        "expected auto backend resolution to pick Metal:\n{}",
        combined
    );
    assert!(
        combined.contains("Metal v1 replays native prefill"),
        "expected replay-prefill Metal decode path:\n{}",
        combined
    );
    assert!(
        combined.contains("[result]"),
        "expected result summary in output:\n{}",
        combined
    );
}
