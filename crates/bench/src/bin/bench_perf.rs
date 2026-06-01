use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use supersonic_bench::matrix::{run_matrix, BenchArch, MatrixConfig};
use supersonic_bench::runs::{allocate_run_dir, RunDir};

#[derive(Parser, Debug)]
#[command(name = "bench-perf")]
struct Cli {
    #[arg(long, default_value = "gfx1100")]
    arch: String,
    /// Named benchmark preset. Presets overwrite arch/model/quant/token policy
    /// fields so the measured workload is reproducible.
    #[arg(long, value_enum)]
    preset: Option<PerfPreset>,
    #[arg(long, default_value = "all")]
    models: String,
    #[arg(long, default_value = "all")]
    quants: String,
    #[arg(long, default_value = "The quick brown fox jumps over")]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: u32,
    #[arg(long)]
    context_size: Option<u32>,
    #[arg(long, default_value_t = 2)]
    warmup_tokens: u32,
    #[arg(long, default_value_t = 3)]
    measurement_runs: u32,
    #[arg(long, default_value_t = 3)]
    cooldown_seconds: u32,
    /// Skip extra attribution/profile runs after measured samples.
    #[arg(long)]
    no_attribution: bool,
    #[arg(long, default_value = "./target/release/supersonic")]
    binary: PathBuf,
    #[arg(long, default_value = "./target/bench-runs")]
    run_root: PathBuf,
    /// Model dir override. Use KEY=PATH for a specific model, or PATH when a
    /// single model is selected.
    #[arg(long = "model-dir", value_parser = parse_model_dir)]
    model_dirs: Vec<ModelDirArg>,
    /// SpecPrefill draft dir override: KEY=PATH, repeatable. KEY is the target
    /// model, e.g. "qwen3.6-35b-a3b=/models/Qwen3.5-0.8B".
    #[arg(long = "specprefill-draft-dir", value_parser = parse_kv)]
    specprefill_draft_dirs: Vec<(String, PathBuf)>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum PerfPreset {
    /// Public llama.cpp-style M5 Max generation benchmark:
    /// qwen3.5-35b-a3b Q4_K_M, tg/gen 512, ctx1024, one warmup, five measured reps.
    Qwen35Q4kmM5MaxGen512,
}

fn parse_kv(s: &str) -> Result<(String, PathBuf), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| "expected KEY=PATH".to_string())?;
    Ok((k.to_string(), PathBuf::from(v)))
}

#[derive(Debug, Clone)]
enum ModelDirArg {
    Exact(String, PathBuf),
    Bare(PathBuf),
}

fn parse_model_dir(s: &str) -> Result<ModelDirArg, String> {
    if let Some((k, v)) = s.split_once('=') {
        Ok(ModelDirArg::Exact(k.to_string(), PathBuf::from(v)))
    } else {
        Ok(ModelDirArg::Bare(PathBuf::from(s)))
    }
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    apply_preset(&mut cli);
    let arch = BenchArch::parse(&cli.arch).ok_or_else(|| anyhow!("unknown arch: {}", cli.arch))?;
    let combos = supersonic_bench::matrix::combos_for_arch(arch.clone());
    let models = filter_csv(&cli.models, combos.iter().map(|c| c.model));
    let quants = filter_csv(&cli.quants, combos.iter().map(|c| c.quant));

    let git_sha = capture_git_sha();
    let git_status = capture_git_status_porcelain();
    let git_dirty_paths = parse_git_dirty_paths(&git_status);
    let git_dirty = !git_dirty_paths.is_empty();
    let git_diff_hash = capture_git_diff_hash(&git_status);
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    std::fs::create_dir_all(&cli.run_root)?;
    let run_path = allocate_run_dir(&cli.run_root, &git_sha, &today);
    let rd = RunDir::new(run_path.clone());

    let selected_models = models.clone();
    let model_root = std::env::var_os("SUPERSONIC_TEST_MODEL_ROOT").map(PathBuf::from);
    let (dir_map, bare_model_dir) = normalize_model_dirs(cli.model_dirs, &selected_models)?;
    let resolver: Box<dyn Fn(&str) -> Result<PathBuf>> = Box::new(move |m: &str| {
        let candidate = dir_map.get(m).cloned().or_else(|| bare_model_dir.clone());
        if let Some(path) = candidate {
            if !path.exists() {
                return Err(anyhow!(
                    "model dir for {m} does not exist: {}; pass --model-dir {m}=PATH or set SUPERSONIC_TEST_MODEL_ROOT",
                    path.display()
                ));
            }
            return Ok(path);
        }
        let Some(root) = model_root.as_ref() else {
            return Ok(PathBuf::from(format!("/mnt/data/models/{m}")));
        };
        let path = root.join(m);
        if !path.exists() {
            return Err(anyhow!(
                "model dir for {m} does not exist: {}; pass --model-dir {m}=PATH or set SUPERSONIC_TEST_MODEL_ROOT",
                path.display()
            ));
        }
        Ok(path)
    });
    let draft_dir_map: HashMap<_, _> = cli.specprefill_draft_dirs.into_iter().collect();
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
        context_size: cli.context_size,
        warmup_tokens: cli.warmup_tokens,
        measurement_runs: cli.measurement_runs,
        cooldown_seconds: cli.cooldown_seconds,
        collect_attribution: !cli.no_attribution,
        git_sha,
        git_dirty,
        git_dirty_paths,
        git_diff_hash,
        runner_version,
    };
    run_matrix(&cfg, &rd)?;
    println!("[bench-perf] wrote {}", run_path.display());
    Ok(())
}

fn apply_preset(cli: &mut Cli) {
    match cli.preset {
        Some(PerfPreset::Qwen35Q4kmM5MaxGen512) => {
            cli.arch = "apple-m5-max".into();
            cli.models = "qwen3.5-35b-a3b".into();
            cli.quants = "q4km".into();
            cli.prompt.clear();
            cli.max_new_tokens = 512;
            cli.context_size = Some(1024);
            cli.warmup_tokens = 512;
            cli.measurement_runs = 5;
            cli.cooldown_seconds = 0;
            cli.no_attribution = true;
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_preset, Cli, PerfPreset};
    use clap::Parser;

    #[test]
    fn qwen35_q4km_m5max_gen512_preset_cli_name_is_stable() {
        let cli = Cli::try_parse_from(["bench-perf", "--preset", "qwen35-q4km-m5-max-gen512"])
            .expect("preset should parse with the documented CLI spelling");

        assert_eq!(cli.preset, Some(PerfPreset::Qwen35Q4kmM5MaxGen512));
    }

    #[test]
    fn qwen35_q4km_m5max_gen512_preset_matches_public_generation_shape() {
        let mut cli = Cli {
            arch: "gfx1100".into(),
            preset: Some(PerfPreset::Qwen35Q4kmM5MaxGen512),
            models: "all".into(),
            quants: "all".into(),
            prompt: "The quick brown fox jumps over".into(),
            max_new_tokens: 16,
            context_size: None,
            warmup_tokens: 2,
            measurement_runs: 3,
            cooldown_seconds: 3,
            no_attribution: false,
            binary: "./target/release/supersonic".into(),
            run_root: "./target/bench-runs".into(),
            model_dirs: vec![],
            specprefill_draft_dirs: vec![],
        };

        apply_preset(&mut cli);

        assert_eq!(cli.arch, "apple-m5-max");
        assert_eq!(cli.models, "qwen3.5-35b-a3b");
        assert_eq!(cli.quants, "q4km");
        assert_eq!(cli.prompt, "");
        assert_eq!(cli.max_new_tokens, 512);
        assert_eq!(cli.context_size, Some(1024));
        assert_eq!(cli.warmup_tokens, 512);
        assert_eq!(cli.measurement_runs, 5);
        assert_eq!(cli.cooldown_seconds, 0);
        assert!(cli.no_attribution);
    }
}

fn normalize_model_dirs(
    args: Vec<ModelDirArg>,
    selected_models: &[String],
) -> Result<(HashMap<String, PathBuf>, Option<PathBuf>)> {
    let mut exact = HashMap::new();
    let mut bare = None;
    for arg in args {
        match arg {
            ModelDirArg::Exact(model, path) => {
                exact.insert(model, path);
            }
            ModelDirArg::Bare(path) => {
                if selected_models.len() != 1 {
                    return Err(anyhow!(
                        "--model-dir PATH is only valid when exactly one model is selected; use --model-dir KEY=PATH for matrix runs"
                    ));
                }
                bare = Some(path);
            }
        }
    }
    Ok((exact, bare))
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

fn capture_git_status_porcelain() -> String {
    std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=all"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

fn parse_git_dirty_paths(status: &str) -> Vec<String> {
    status
        .lines()
        .filter_map(|line| {
            if line.len() < 4 {
                return None;
            }
            let path = line[3..].trim();
            if path.is_empty() {
                return None;
            }
            Some(
                path.split_once(" -> ")
                    .map(|(_, new_path)| new_path)
                    .unwrap_or(path)
                    .trim_matches('"')
                    .to_string(),
            )
        })
        .collect()
}

fn capture_git_diff_hash(status: &str) -> Option<String> {
    let diff = std::process::Command::new("git")
        .args(["diff", "--binary", "HEAD"])
        .output()
        .ok()?;
    let mut hasher = Sha256::new();
    hasher.update(status.as_bytes());
    hasher.update(b"\0");
    hasher.update(&diff.stdout);
    Some(format!("{:x}", hasher.finalize()))
}

fn capture_runner_version(binary: &PathBuf) -> String {
    std::process::Command::new(binary)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
