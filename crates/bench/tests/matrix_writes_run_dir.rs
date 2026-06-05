use std::path::PathBuf;
use supersonic_bench::matrix::{run_matrix, BenchArch, MatrixConfig};
use supersonic_bench::runs::RunDir;

#[test]
fn matrix_writes_meta_and_at_least_one_perf_cell() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = MatrixConfig {
        arch: BenchArch::Gfx1100,
        models: vec!["qwen3.5-0.8b".into()],
        quants: vec!["bf16".into()],
        binary: PathBuf::from("/bin/echo"), // will produce no [result], so cells will be Error
        model_dir_resolver: Box::new(|_| Ok(PathBuf::from("/nonexistent"))),
        specprefill_draft_dir_resolver: Box::new(|_| None),
        prompt: "x".into(),
        max_new_tokens: 1,
        context_size: None,
        warmup_tokens: 1,
        measurement_runs: 1,
        cooldown_seconds: 0,
        collect_attribution: true,
        git_sha: "test".into(),
        git_dirty: false,
        git_dirty_paths: vec![],
        git_diff_hash: Some("clean".into()),
        runner_version: "test 0.0.0".into(),
    };
    let rd = RunDir::new(tmp.path().join("run-1"));
    run_matrix(&cfg, &rd).unwrap();
    assert!(rd.meta_path().exists(), "meta.json should be written");
    assert!(rd.perf_path("qwen3.5-0.8b", "bf16").exists());
    let meta_text = std::fs::read_to_string(rd.meta_path()).unwrap();
    assert!(meta_text.contains("\"arch\": \"gfx1100\""));
}

#[test]
fn matrix_skips_unsupported_combo_with_skipped_status() {
    // qwen3.6-35b-a3b only supports int4 and kv-fp8 on gfx1100; bf16 is not
    // a registered combo. run_matrix should write a Skipped cell rather than
    // running the binary and producing an Error cell.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = MatrixConfig {
        arch: BenchArch::Gfx1100,
        models: vec!["qwen3.6-35b-a3b".into()],
        quants: vec!["bf16".into()],
        // /bin/false would produce an Error cell if the supported-combo guard
        // weren't there; the test asserts the guard runs first.
        binary: PathBuf::from("/bin/false"),
        model_dir_resolver: Box::new(|_| Ok(PathBuf::from("/nonexistent"))),
        specprefill_draft_dir_resolver: Box::new(|_| None),
        prompt: "x".into(),
        max_new_tokens: 1,
        context_size: None,
        warmup_tokens: 1,
        measurement_runs: 1,
        cooldown_seconds: 0,
        collect_attribution: true,
        git_sha: "test".into(),
        git_dirty: false,
        git_dirty_paths: vec![],
        git_diff_hash: Some("clean".into()),
        runner_version: "test 0.0.0".into(),
    };
    let rd = RunDir::new(tmp.path().join("run-skip"));
    run_matrix(&cfg, &rd).unwrap();
    let cell_path = rd.perf_path("qwen3.6-35b-a3b", "bf16");
    assert!(
        cell_path.exists(),
        "expected skipped perf cell to be written"
    );
    let cell_text = std::fs::read_to_string(&cell_path).unwrap();
    assert!(
        cell_text.contains("\"status\": \"skipped\""),
        "expected skipped status, got: {cell_text}"
    );
    assert!(
        cell_text.contains("not in SUPPORTED_COMBOS"),
        "expected reason to mention SUPPORTED_COMBOS, got: {cell_text}"
    );
}

#[test]
fn matrix_runs_ad_hoc_sm86_specprefill_lane() {
    // `int4-spec070` is intentionally not in SUPPORTED_COMBOS because it is an
    // exploratory keep-ratio lane, but sm86 Qwen3.6 sweeps should still run it
    // when requested explicitly.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = MatrixConfig {
        arch: BenchArch::Sm86,
        models: vec!["qwen3.6-35b-a3b".into()],
        quants: vec!["int4-spec070".into()],
        binary: PathBuf::from("/bin/echo"),
        model_dir_resolver: Box::new(|_| Ok(PathBuf::from("/nonexistent"))),
        specprefill_draft_dir_resolver: Box::new(|_| Some(PathBuf::from("/draft"))),
        prompt: "x".into(),
        max_new_tokens: 1,
        context_size: None,
        warmup_tokens: 1,
        measurement_runs: 1,
        cooldown_seconds: 0,
        collect_attribution: true,
        git_sha: "test".into(),
        git_dirty: false,
        git_dirty_paths: vec![],
        git_diff_hash: Some("clean".into()),
        runner_version: "test 0.0.0".into(),
    };
    let rd = RunDir::new(tmp.path().join("run-ad-hoc-spec"));
    run_matrix(&cfg, &rd).unwrap();
    let cell_text =
        std::fs::read_to_string(rd.perf_path("qwen3.6-35b-a3b", "int4-spec070")).unwrap();
    assert!(
        cell_text.contains("\"status\": \"error\""),
        "expected the binary to be invoked and fail metric extraction, got: {cell_text}"
    );
    assert!(
        !cell_text.contains("\"status\": \"skipped\""),
        "ad hoc sm86 spec lane should not be filtered out: {cell_text}"
    );
}

#[test]
fn matrix_writes_skipped_lower_precision_candidate_with_artifact_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = MatrixConfig {
        arch: BenchArch::AppleM5Max,
        models: vec!["qwen3.5-0.8b".into()],
        quants: vec!["int2-4-mixed".into()],
        binary: PathBuf::from("/bin/false"),
        model_dir_resolver: Box::new(|_| Ok(PathBuf::from("/nonexistent"))),
        specprefill_draft_dir_resolver: Box::new(|_| None),
        prompt: "x".into(),
        max_new_tokens: 1,
        context_size: None,
        warmup_tokens: 1,
        measurement_runs: 1,
        cooldown_seconds: 0,
        collect_attribution: true,
        git_sha: "test".into(),
        git_dirty: false,
        git_dirty_paths: vec![],
        git_diff_hash: Some("clean".into()),
        runner_version: "test 0.0.0".into(),
    };
    let rd = RunDir::new(tmp.path().join("run-lower-precision-skip"));
    run_matrix(&cfg, &rd).unwrap();
    let cell_text = std::fs::read_to_string(rd.perf_path("qwen3.5-0.8b", "int2-4-mixed")).unwrap();
    assert!(
        cell_text.contains("\"status\": \"skipped\""),
        "expected skipped status, got: {cell_text}"
    );
    assert!(
        cell_text.contains("experimental lower-precision candidate"),
        "expected lower-precision reason, got: {cell_text}"
    );
    assert!(
        cell_text.contains("\"quant_artifact\""),
        "expected quant artifact metadata, got: {cell_text}"
    );
    assert!(
        cell_text.contains("\"profile\": \"autoround-int2-4-mixed\""),
        "expected AutoRound profile metadata, got: {cell_text}"
    );
}
