use std::path::PathBuf;
use std::process::Command;

fn require_artifacts() -> bool {
    std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
}

fn artifact_path(name: &str) -> Option<PathBuf> {
    let Some(value) = std::env::var_os(name) else {
        if require_artifacts() {
            panic!("{name} is required for the Qwen3.8 decode boundary test");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if path.exists() {
        Some(path)
    } else if require_artifacts() {
        panic!("{name} points to a missing path: {}", path.display());
    } else {
        None
    }
}

fn field_value(line: &str, name: &str) -> Option<f64> {
    line.split_whitespace()
        .find_map(|field| field.strip_prefix(name)?.strip_prefix('='))
        .and_then(|value| value.parse().ok())
}

#[test]
#[ignore = "requires a configured Qwen3.8 GQH artifact and HIP device"]
fn ordinary_decode_ms_excludes_prefill_and_prepare() {
    let Some(model_dir) = artifact_path("SUPERSONIC_QWEN38_MODEL_DIR") else {
        return;
    };
    let Some(gguf_file) = artifact_path("SUPERSONIC_GQH_GGUF") else {
        return;
    };

    let output = Command::new(env!("CARGO_BIN_EXE_supersonic"))
        .arg("--model")
        .arg("qwen3.8-27b")
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--gguf-file")
        .arg(gguf_file)
        .arg("--prompt")
        .arg(
            "You are benchmarking a single-sequence greedy Qwen3.8 runner. \
             Explain how fixed prompts and exact artifacts keep timing comparable.",
        )
        .arg("--max-new-tokens")
        .arg("16")
        .arg("--ignore-eos")
        .arg("--context-size")
        .arg("32768")
        .arg("--emit-stage-timings")
        .arg("--device")
        .arg("0")
        .output()
        .expect("run supersonic");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "supersonic failed with status {}:\n{stdout}\n{stderr}",
        output.status
    );

    let combined = format!("{stdout}\n{stderr}");
    let result_lines: Vec<&str> = combined
        .lines()
        .filter(|line| line.starts_with("[result] "))
        .collect();
    let stage_lines: Vec<&str> = combined
        .lines()
        .filter(|line| line.starts_with("[stage-timings] "))
        .collect();
    assert_eq!(result_lines.len(), 1, "expected exactly one result line");
    assert_eq!(
        stage_lines.len(),
        1,
        "expected exactly one stage-timings line"
    );

    let decode_ms = field_value(result_lines[0], "decode_ms").expect("decode_ms must be numeric");
    let native_ms = field_value(stage_lines[0], "total_native_decode_ms")
        .expect("total_native_decode_ms must be numeric");
    let tolerance = native_ms * 0.05 + 50.0;
    assert!(
        decode_ms <= native_ms + tolerance,
        "decode_ms {decode_ms} includes prefill or prepare; native decode was {native_ms}"
    );
}
