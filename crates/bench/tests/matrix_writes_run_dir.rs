use supersonic_bench::matrix::{run_matrix, BenchArch, MatrixConfig};
use supersonic_bench::runs::RunDir;
use std::path::PathBuf;

#[test]
fn matrix_writes_meta_and_at_least_one_perf_cell() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = MatrixConfig {
        arch: BenchArch::Gfx1100,
        models: vec!["qwen3.5-0.8b".into()],
        quants: vec!["bf16".into()],
        binary: PathBuf::from("/bin/echo"),  // will produce no [result], so cells will be Error
        model_dir_resolver: Box::new(|_| PathBuf::from("/nonexistent")),
        prompt: "x".into(),
        max_new_tokens: 1,
        warmup_tokens: 1,
        measurement_runs: 1,
        cooldown_seconds: 0,
        git_sha: "test".into(),
        runner_version: "test 0.0.0".into(),
    };
    let rd = RunDir::new(tmp.path().join("run-1"));
    run_matrix(&cfg, &rd).unwrap();
    assert!(rd.meta_path().exists(), "meta.json should be written");
    assert!(rd.perf_path("qwen3.5-0.8b", "bf16").exists());
    let meta_text = std::fs::read_to_string(rd.meta_path()).unwrap();
    assert!(meta_text.contains("\"arch\": \"gfx1100\""));
}
