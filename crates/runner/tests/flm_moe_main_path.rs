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
fn occurrence_count(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

#[cfg(target_os = "linux")]
fn assert_moe_flm_main_path_contract(combined: &str) {
    assert!(
        combined.contains("[qwen36-moe] loading config from FLM runtime descriptor"),
        "config was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading tokenizer from FLM assets"),
        "tokenizer was not loaded from FLM:\n{combined}"
    );
    assert_eq!(
        occurrence_count(combined, "[flm] opening model source"),
        1,
        "FLM main path should open the source exactly once:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading weights from already-open FLM source"),
        "weights were not loaded from the already-open FLM source:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] FLM weight mode: INT4 native FLM"),
        "FLM weight mode was not reported:\n{combined}"
    );
    assert!(
        combined.contains("loading weights from already-open FLM source")
            && combined.contains("(INT4 native FLM)"),
        "native FLM source load was not labeled as native INT4:\n{combined}"
    );
    assert!(
        combined.contains("BLAKE3 hash verification enabled"),
        "--verify-flm-hashes was not threaded to the single FLM source open:\n{combined}"
    );
    assert!(
        combined.contains("Generated ids: "),
        "MoE decode did not emit generated token ids:\n{combined}"
    );
    assert!(
        combined.contains("[result] prompt_tokens="),
        "MoE decode did not emit the generation result summary:\n{combined}"
    );

    for forbidden in [
        "[fetch]",
        "[bake]",
        "config.json",
        "tokenizer.json",
        "safetensors",
        ".supersonic",
        "INT4 GPTQ",
    ] {
        assert!(
            !combined.contains(forbidden),
            "FLM MoE main path unexpectedly referenced {forbidden:?}:\n{combined}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn moe_flm_main_path_output_contract_accepts_expected_logs() {
    let combined = "\
[flm] opening model source at /tmp/qwen36-35b-a3b.flm (FLM logical INT4 aliases enabled) (BLAKE3 hash verification enabled)
[qwen36-moe] loading config from FLM runtime descriptor
[qwen36-moe] loading tokenizer from FLM assets
[qwen36-moe] FLM weight mode: INT4 native FLM
[qwen36-moe] loading weights from already-open FLM source at /tmp/qwen36-35b-a3b.flm (INT4 native FLM)
  Generated ids: [123]
[result] prompt_tokens=1 generated_tokens=1 decode_ms=1 ms_per_step=1.0
";

    assert_moe_flm_main_path_contract(combined);
}

#[cfg(target_os = "linux")]
#[test]
fn qwen36_moe_flm_model_dir_runs_without_hf_snapshot() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        panic!(
            "SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM is set but the path does not exist: {}",
            path.display()
        );
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
        "qwen3.6-35b-a3b",
        "--model-dir",
    ]);
    cmd.arg(&path);
    cmd.args([
        "--verify-flm-hashes",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "1",
        "--context-size",
        "16",
        "--no-download",
    ]);

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("run supersonic Qwen3.6-MoE FLM main-path smoke: {e}"));
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "Qwen3.6-MoE FLM main-path smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert_moe_flm_main_path_contract(&combined);
}

#[cfg(target_os = "linux")]
#[test]
fn qwen36_moe_ct_int4_flm_dry_run_consumes_source_without_hf_snapshot() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_35B_CT_INT4_FLM_DRY_RUN") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_35B_CT_INT4_FLM_DRY_RUN is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        panic!(
            "SUPERSONIC_QWEN36_35B_CT_INT4_FLM_DRY_RUN is set but the path does not exist: {}",
            path.display()
        );
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
        "qwen3.6-35b-a3b",
        "--model-dir",
    ]);
    cmd.arg(&path);
    cmd.args(["--context-size", "16", "--no-download", "--dry-run"]);

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("run supersonic Qwen3.6-MoE CT FLM dry-run smoke: {e}"));
    let combined = combined_output(&output);

    assert!(
        output.status.success(),
        "Qwen3.6-MoE CT FLM dry-run smoke failed with status {:?}:\n{}",
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("[qwen36-moe] loading config from FLM runtime descriptor"),
        "config was not loaded from FLM:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] loading tokenizer from FLM assets"),
        "tokenizer was not loaded from FLM:\n{combined}"
    );
    assert_eq!(
        occurrence_count(&combined, "[flm] opening model source"),
        1,
        "FLM dry-run should open the source exactly once:\n{combined}"
    );
    assert!(
        combined.contains("[qwen36-moe] FLM weight mode: BF16"),
        "CT INT4 FLM source did not select the BF16 fallback weight mode:\n{combined}"
    );
    assert!(
        combined.contains("[qwen3.6-moe] dry-run summary"),
        "dry-run summary was not emitted:\n{combined}"
    );
    assert!(
        combined.contains("[FLM runtime weights] ready-for-decode: YES"),
        "FLM runtime weights were not reported ready:\n{combined}"
    );
    assert!(
        !combined.contains("Generated ids: "),
        "dry-run should not decode tokens:\n{combined}"
    );

    for forbidden in [
        "[fetch]",
        "[bake]",
        "config.json",
        "tokenizer.json",
        "safetensors",
        ".supersonic",
    ] {
        assert!(
            !combined.contains(forbidden),
            "FLM CT dry-run unexpectedly referenced {forbidden:?}:\n{combined}"
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn qwen36_moe_flm_model_dir_runs_without_hf_snapshot() {
    eprintln!("skipping: Qwen3.6-MoE FLM main-path smoke is Linux/HIP-only");
}
