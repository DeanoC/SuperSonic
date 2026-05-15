#[cfg(target_os = "macos")]
mod support;

#[cfg(target_os = "macos")]
use std::process::Command;

#[cfg(target_os = "macos")]
use support::{GEMMA4_E2B, GEMMA4_E4B, PHI4_MINI, QWEN3_30B_A3B};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct SmokeCase {
    name: &'static str,
    model: &'static str,
    model_dir: support::TestModel,
    quant_flag: Option<&'static str>,
    family_marker: &'static str,
    quant_marker: Option<&'static str>,
}

#[cfg(target_os = "macos")]
const CASES: &[SmokeCase] = &[
    SmokeCase {
        name: "qwen3_30b_a3b_int4",
        model: "qwen3-30b-a3b",
        model_dir: QWEN3_30B_A3B,
        quant_flag: Some("--int4"),
        family_marker: "[qwen3-moe]",
        quant_marker: Some("INT4"),
    },
    SmokeCase {
        name: "gemma4_e2b_bf16",
        model: "gemma4-e2b",
        model_dir: GEMMA4_E2B,
        quant_flag: None,
        family_marker: "[gemma4]",
        quant_marker: None,
    },
    SmokeCase {
        name: "gemma4_e2b_int4",
        model: "gemma4-e2b",
        model_dir: GEMMA4_E2B,
        quant_flag: Some("--int4"),
        family_marker: "[gemma4]",
        quant_marker: Some("INT4 GPTQ"),
    },
    SmokeCase {
        name: "gemma4_e4b_bf16",
        model: "gemma4-e4b",
        model_dir: GEMMA4_E4B,
        quant_flag: None,
        family_marker: "[gemma4]",
        quant_marker: None,
    },
    SmokeCase {
        name: "gemma4_e4b_int4",
        model: "gemma4-e4b",
        model_dir: GEMMA4_E4B,
        quant_flag: Some("--int4"),
        family_marker: "[gemma4]",
        quant_marker: Some("INT4 GPTQ"),
    },
    SmokeCase {
        name: "phi4_mini_bf16",
        model: "phi4-mini",
        model_dir: PHI4_MINI,
        quant_flag: None,
        family_marker: "[phi4]",
        quant_marker: None,
    },
    SmokeCase {
        name: "phi4_mini_int4",
        model: "phi4-mini",
        model_dir: PHI4_MINI,
        quant_flag: Some("--int4"),
        family_marker: "[phi4]",
        quant_marker: Some("INT4 runtime dequant"),
    },
    SmokeCase {
        name: "phi4_mini_fp8_runtime",
        model: "phi4-mini",
        model_dir: PHI4_MINI,
        quant_flag: Some("--fp8-runtime"),
        family_marker: "[phi4]",
        quant_marker: Some("FP8 runtime dequant"),
    },
];

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple M5 Max Metal and local models via SUPERSONIC_TEST_MODEL_ROOT"]
fn metal_large_model_smokes_run_end_to_end() {
    for case in CASES {
        run_case(*case);
    }
}

#[cfg(target_os = "macos")]
fn run_case(case: SmokeCase) {
    let Some(model_dir) = support::resolve_model_dir(case.model_dir) else {
        eprintln!(
            "skipping {}: set SUPERSONIC_TEST_MODEL_ROOT with {} or {}",
            case.name, case.model_dir.canonical_subdir, case.model_dir.override_env
        );
        return;
    };

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.env("PATH", support::path_with_repo_venv()).args([
        "--backend",
        "metal",
        "--model",
        case.model,
        "--model-dir",
        model_dir.to_str().expect("model dir must be valid UTF-8"),
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "1",
        "--emit-stage-timings",
    ]);
    if let Some(flag) = case.quant_flag {
        cmd.arg(flag);
    }

    let output = cmd.output().unwrap_or_else(|e| {
        panic!("run {} Metal smoke: {e}", case.name);
    });
    let combined = support::combined_output(&output);

    assert!(
        output.status.success(),
        "{} Metal smoke failed with status {:?}:\n{}",
        case.name,
        output.status.code(),
        combined
    );
    assert!(
        combined.contains("backend=Metal"),
        "{} expected Metal backend selection:\n{}",
        case.name,
        combined
    );
    assert!(
        combined.contains(case.family_marker),
        "{} expected family marker {}:\n{}",
        case.name,
        case.family_marker,
        combined
    );
    if let Some(marker) = case.quant_marker {
        assert!(
            combined.contains(marker),
            "{} expected quant marker {}:\n{}",
            case.name,
            marker,
            combined
        );
    }
    assert!(
        combined.contains("[result]"),
        "{} expected result summary:\n{}",
        case.name,
        combined
    );
}
