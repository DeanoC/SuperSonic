#[cfg(target_os = "macos")]
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::process::Command;

#[cfg(target_os = "macos")]
fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[cfg(target_os = "macos")]
fn model_dir() -> Option<std::ffi::OsString> {
    std::env::var_os("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR")
        .or_else(|| std::env::var_os("SUPERSONIC_TEST_MODEL_DIR"))
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple M5 Max Metal and a local Qwen3.6-35B-A3B dir via SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR or SUPERSONIC_TEST_MODEL_DIR"]
fn qwen36_moe_metal_int4_smoke_runs_end_to_end() {
    let Some(model_dir) = model_dir() else {
        eprintln!(
            "skipping: SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR/SUPERSONIC_TEST_MODEL_DIR is not set"
        );
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
    cmd.env("PATH", path_value).args([
        "--backend",
        "metal",
        "--model",
        "qwen3.6-35b-a3b",
        "--model-dir",
        model_dir.to_str().expect("model dir must be valid UTF-8"),
        "--int4",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "1",
        "--emit-stage-timings",
    ]);

    let output = cmd.output().expect("run Qwen3.6-MoE Metal INT4 smoke test");
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "Qwen3.6-MoE Metal INT4 smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("backend=Metal"),
        "expected Metal backend selection:\n{}",
        combined
    );
    assert!(
        combined.contains("=== Decode (Qwen3.6-MoE) ==="),
        "expected Qwen3.6-MoE decode marker:\n{}",
        combined
    );
    assert!(
        combined.contains("INT4 GPTQ") || combined.contains("INT4 sidecar"),
        "expected INT4 bake/sidecar marker:\n{}",
        combined
    );
    assert!(
        combined.contains("[result]"),
        "expected result summary in output:\n{}",
        combined
    );
}
