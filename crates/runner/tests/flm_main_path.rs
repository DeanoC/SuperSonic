#[cfg(target_os = "linux")]
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::process::Command;

#[cfg(target_os = "linux")]
fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[cfg(target_os = "linux")]
#[test]
fn qwen36_dense_flm_model_dir_runs_without_hf_snapshot() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_27B_NO_HF_FLM") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_27B_NO_HF_FLM is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        eprintln!(
            "skipping: SUPERSONIC_QWEN36_27B_NO_HF_FLM path does not exist: {}",
            path.display()
        );
        return;
    }

    let backend =
        std::env::var("SUPERSONIC_FLM_MAIN_PATH_BACKEND").unwrap_or_else(|_| "hip".to_string());
    let device =
        std::env::var("SUPERSONIC_FLM_MAIN_PATH_DEVICE").unwrap_or_else(|_| "0".to_string());

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args([
        "--backend",
        backend.as_str(),
        "--device",
        device.as_str(),
        "--model",
        "qwen3.6-27b",
        "--model-dir",
    ]);
    cmd.arg(&path);
    cmd.args([
        "--int4",
        "--verify-flm-hashes",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "1",
        "--context-size",
        "16",
        "--emit-generated-json",
    ]);

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("run supersonic FLM main-path smoke: {e}"));
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "FLM main-path smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("[config] loading FLM runtime descriptor"),
        "config was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[tokenizer] loading FLM tokenizer assets"),
        "tokenizer was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[weights] loading FLM container"),
        "weights were not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("BLAKE3 hash verification enabled"),
        "--verify-flm-hashes was not threaded to FLM weight loading:\n{combined}"
    );
    assert!(
        combined.contains("[tokens] "),
        "decode did not emit generated token ids:\n{combined}"
    );
    assert!(
        combined.contains("[generated_json] "),
        "decode did not emit generated text JSON:\n{combined}"
    );

    for forbidden in ["[fetch]", "[bake]", "config.json", "tokenizer.json", ".supersonic"] {
        assert!(
            !combined.contains(forbidden),
            "FLM main path unexpectedly referenced {forbidden:?}:\n{combined}"
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn qwen36_dense_flm_model_dir_runs_without_hf_snapshot() {
    eprintln!("skipping: FLM main-path smoke is Linux/HIP-only");
}
