use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::PathBuf;
use supersonic_bench::matrix::{run_matrix, BenchArch, MatrixConfig};
use supersonic_bench::runs::{allocate_run_dir, RunDir};

#[derive(Parser, Debug)]
#[command(name = "bench-perf")]
struct Cli {
    #[arg(long, default_value = "gfx1100")]
    arch: String,
    #[arg(long, default_value = "all")]
    models: String,
    #[arg(long, default_value = "all")]
    quants: String,
    #[arg(long, default_value = "The quick brown fox jumps over")]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: u32,
    #[arg(long, default_value_t = 2)]
    warmup_tokens: u32,
    #[arg(long, default_value_t = 3)]
    measurement_runs: u32,
    #[arg(long, default_value_t = 3)]
    cooldown_seconds: u32,
    #[arg(long, default_value = "./target/release/supersonic")]
    binary: PathBuf,
    #[arg(long, default_value = "./target/bench-runs")]
    run_root: PathBuf,
    /// Model dir override: KEY=PATH, repeatable. KEY is e.g. "qwen3.5-0.8b".
    #[arg(long = "model-dir", value_parser = parse_kv)]
    model_dirs: Vec<(String, PathBuf)>,
    /// SpecPrefill draft dir override: KEY=PATH, repeatable. KEY is the target
    /// model, e.g. "qwen3.6-35b-a3b=/models/Qwen3.5-0.8B".
    #[arg(long = "specprefill-draft-dir", value_parser = parse_kv)]
    specprefill_draft_dirs: Vec<(String, PathBuf)>,
}

fn parse_kv(s: &str) -> Result<(String, PathBuf), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| "expected KEY=PATH".to_string())?;
    Ok((k.to_string(), PathBuf::from(v)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let arch = BenchArch::parse(&cli.arch).ok_or_else(|| anyhow!("unknown arch: {}", cli.arch))?;
    let combos = supersonic_bench::matrix::combos_for_arch(arch.clone());
    let models = filter_csv(&cli.models, combos.iter().map(|c| c.model));
    let quants = filter_csv(&cli.quants, combos.iter().map(|c| c.quant));

    let git_sha = capture_git_sha();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    std::fs::create_dir_all(&cli.run_root)?;
    let run_path = allocate_run_dir(&cli.run_root, &git_sha, &today);
    let rd = RunDir::new(run_path.clone());

    let dir_map: std::collections::HashMap<_, _> = cli.model_dirs.into_iter().collect();
    let resolver: Box<dyn Fn(&str) -> PathBuf> = Box::new(move |m: &str| {
        dir_map
            .get(m)
            .cloned()
            .unwrap_or_else(|| PathBuf::from(format!("/mnt/data/models/{m}")))
    });
    let draft_dir_map: std::collections::HashMap<_, _> =
        cli.specprefill_draft_dirs.into_iter().collect();
    let draft_resolver: Box<dyn Fn(&str) -> Option<PathBuf>> =
        Box::new(move |m: &str| draft_dir_map.get(m).cloned());

    let binary_clone = cli.binary.clone();
    let runner_version = capture_runner_version(&cli.binary);
    let cfg = MatrixConfig {
        arch,
        models,
        quants,
        binary: binary_clone,
        model_dir_resolver: resolver,
        specprefill_draft_dir_resolver: draft_resolver,
        prompt: cli.prompt,
        max_new_tokens: cli.max_new_tokens,
        warmup_tokens: cli.warmup_tokens,
        measurement_runs: cli.measurement_runs,
        cooldown_seconds: cli.cooldown_seconds,
        git_sha,
        runner_version,
    };
    run_matrix(&cfg, &rd)?;
    println!("[bench-perf] wrote {}", run_path.display());
    Ok(())
}

fn filter_csv<'a>(spec: &str, available: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let unique: std::collections::BTreeSet<&str> = available.into_iter().collect();
    if spec == "all" {
        unique.into_iter().map(|s| s.to_string()).collect()
    } else {
        spec.split(',').map(|s| s.trim().to_string()).collect()
    }
}

fn capture_git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn capture_runner_version(binary: &PathBuf) -> String {
    std::process::Command::new(binary)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
