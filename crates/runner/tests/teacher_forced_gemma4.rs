//! Smoke: --teacher-forced --model gemma4-e2b emits a [teacher_forced_json] line.
//!
//! This requires a real model dir + GPU. Skipped automatically when MODEL_DIR_GEMMA_E2B is unset.

#[test]
fn teacher_forced_gemma4_emits_json_line() {
    let model_dir = match std::env::var("MODEL_DIR_GEMMA_E2B") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("MODEL_DIR_GEMMA_E2B unset; skipping");
            return;
        }
    };
    let bin =
        std::env::var("SUPERSONIC_BIN").unwrap_or_else(|_| "./target/release/supersonic".into());
    let out = std::process::Command::new(&bin)
        .args([
            "--model",
            "gemma4-e2b",
            "--model-dir",
            &model_dir,
            "--prompt",
            "The quick brown fox jumps over the lazy dog",
            "--teacher-forced",
            "--max-new-tokens",
            "1",
        ])
        .output()
        .expect("spawn supersonic");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("[teacher_forced_json]"),
        "stdout missing teacher_forced_json line:\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
