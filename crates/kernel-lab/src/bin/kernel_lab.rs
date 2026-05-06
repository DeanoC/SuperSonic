use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use gpu_hal::Backend;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitStatus};
use supersonic_kernel_lab::run::{
    diff_exit_code, diff_runs, load_summary, render_diff_markdown, render_markdown, resolve_tasks,
    run_tasks, KernelLabConfig,
};

#[derive(Parser, Debug)]
#[command(name = "kernel-lab")]
#[command(about = "KernelBench-style harness for SuperSonic kernel candidates")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    List,
    Run {
        #[arg(long, default_value = "all")]
        tasks: String,
        #[arg(long, default_value = "hip")]
        backend: String,
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        #[arg(long, default_value = "target/kernel-lab-runs")]
        run_root: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Diff {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, default_value_t = 0.03)]
        max_regression: f64,
        #[arg(long, default_value_t = 1.02)]
        min_speedup: f64,
        #[arg(long)]
        markdown_out: Option<PathBuf>,
        #[arg(long)]
        github_summary: bool,
    },
    Baseline {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "crates/kernel-lab/baselines")]
        out_root: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        no_markdown: bool,
    },
    CompareRef {
        #[arg(long, default_value = "main")]
        baseline_ref: String,
        #[arg(long, default_value = "worktree")]
        candidate_ref: String,
        #[arg(long, default_value = "all")]
        tasks: String,
        #[arg(long, default_value = "hip")]
        backend: String,
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        #[arg(long, default_value = "target/kernel-lab-runs")]
        run_root: PathBuf,
        #[arg(long, default_value = "target/kernel-lab-worktrees")]
        worktree_root: PathBuf,
        #[arg(long, default_value_t = 0.03)]
        max_regression: f64,
        #[arg(long, default_value_t = 1.02)]
        min_speedup: f64,
        #[arg(long)]
        markdown_out: Option<PathBuf>,
        #[arg(long)]
        github_summary: bool,
    },
    Render {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::List => {
            for task in supersonic_kernel_lab::all_tasks() {
                println!(
                    "{}\tfamily={}\trequired={}\ttags={}",
                    task.id,
                    task.family,
                    task.required,
                    task.tags.join(",")
                );
            }
        }
        Command::Run {
            tasks,
            backend,
            device,
            warmup,
            iters,
            run_root,
            json,
        } => {
            let backend =
                Backend::parse(&backend).ok_or_else(|| anyhow!("unknown backend: {backend}"))?;
            let tasks = resolve_tasks(&tasks)?;
            let cfg = KernelLabConfig {
                backend,
                device,
                warmup,
                iters,
                run_root,
            };
            let (run_dir, summary) = run_tasks(&cfg, &tasks)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                println!(
                    "[kernel-lab] wrote {} (required {}/{})",
                    run_dir.display(),
                    summary.passed_required_tasks,
                    summary.required_task_count
                );
            }
            if run_has_failed_tasks(&summary) {
                std::process::exit(1);
            }
        }
        Command::Diff {
            baseline,
            candidate,
            max_regression,
            min_speedup,
            markdown_out,
            github_summary,
        } => {
            let baseline = load_summary(&baseline)?;
            let candidate = load_summary(&candidate)?;
            let diff = diff_runs(&baseline, &candidate, max_regression);
            println!("{}", serde_json::to_string_pretty(&diff)?);
            write_diff_markdown(&diff, min_speedup, markdown_out, github_summary)?;
            let code = diff_exit_code(&diff, min_speedup);
            if code != 0 {
                std::process::exit(code);
            }
        }
        Command::Baseline {
            run,
            out_root,
            name,
            no_markdown,
        } => {
            let summary = load_summary(&run)?;
            let path = write_baseline_artifact(&summary, &out_root, name.as_deref(), !no_markdown)?;
            println!("[kernel-lab] wrote baseline {}", path.display());
        }
        Command::CompareRef {
            baseline_ref,
            candidate_ref,
            tasks,
            backend,
            device,
            warmup,
            iters,
            run_root,
            worktree_root,
            max_regression,
            min_speedup,
            markdown_out,
            github_summary,
        } => {
            let baseline = run_ref(
                &baseline_ref,
                &tasks,
                &backend,
                device,
                warmup,
                iters,
                &run_root,
                &worktree_root,
            )?;
            let candidate = run_ref(
                &candidate_ref,
                &tasks,
                &backend,
                device,
                warmup,
                iters,
                &run_root,
                &worktree_root,
            )?;
            let baseline_summary = load_summary(&baseline)?;
            let candidate_summary = load_summary(&candidate)?;
            let diff = diff_runs(&baseline_summary, &candidate_summary, max_regression);
            println!("{}", serde_json::to_string_pretty(&diff)?);
            write_diff_markdown(&diff, min_speedup, markdown_out, github_summary)?;
            let code = diff_exit_code(&diff, min_speedup);
            if code != 0 {
                std::process::exit(code);
            }
        }
        Command::Render { run, out } => {
            let summary = load_summary(&run)?;
            std::fs::write(&out, render_markdown(&summary))?;
            println!("[kernel-lab] wrote {}", out.display());
        }
    }
    Ok(())
}

fn write_baseline_artifact(
    summary: &supersonic_kernel_lab::run::RunSummary,
    out_root: &Path,
    name: Option<&str>,
    write_markdown: bool,
) -> Result<PathBuf> {
    let arch = sanitize_label(&summary.meta.arch);
    let stem = name.map(sanitize_label).unwrap_or_else(|| {
        sanitize_label(&format!("{}-{}", summary.meta.backend, summary.meta.run_id))
    });
    let out_dir = out_root.join(arch);
    std::fs::create_dir_all(&out_dir)?;
    let summary_path = out_dir.join(format!("{stem}.summary.json"));
    std::fs::write(&summary_path, serde_json::to_string_pretty(summary)?)?;
    if write_markdown {
        std::fs::write(out_dir.join(format!("{stem}.md")), render_markdown(summary))?;
    }
    Ok(summary_path)
}

fn write_diff_markdown(
    diff: &supersonic_kernel_lab::run::DiffReport,
    min_speedup: f64,
    markdown_out: Option<PathBuf>,
    github_summary: bool,
) -> Result<()> {
    let markdown = render_diff_markdown(diff, min_speedup);
    if let Some(path) = markdown_out {
        std::fs::write(&path, &markdown)?;
        println!("[kernel-lab] wrote {}", path.display());
    }
    if github_summary {
        let path = std::env::var_os("GITHUB_STEP_SUMMARY")
            .ok_or_else(|| anyhow!("GITHUB_STEP_SUMMARY is not set"))?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(PathBuf::from(path))?;
        writeln!(file, "{markdown}")?;
    }
    Ok(())
}

fn run_ref(
    git_ref: &str,
    tasks: &str,
    backend: &str,
    device: usize,
    warmup: usize,
    iters: usize,
    run_root: &PathBuf,
    worktree_root: &PathBuf,
) -> Result<PathBuf> {
    if git_ref == "worktree" {
        let before = existing_run_dirs(run_root);
        let status = ProcessCommand::new("cargo")
            .args([
                "run",
                "--release",
                "-p",
                "supersonic-kernel-lab",
                "--bin",
                "kernel-lab",
                "--",
                "run",
                "--tasks",
                tasks,
                "--backend",
                backend,
                "--device",
                &device.to_string(),
                "--warmup",
                &warmup.to_string(),
                "--iters",
                &iters.to_string(),
                "--run-root",
                run_root.to_string_lossy().as_ref(),
            ])
            .status()?;
        return child_run_dir_after(status, run_root, &before, "candidate worktree");
    }

    std::fs::create_dir_all(worktree_root)?;
    let label = sanitize_ref(git_ref);
    let dir = worktree_root.join(label);
    if dir.exists() {
        run_checked(
            "git",
            &[
                "worktree",
                "remove",
                "--force",
                dir.to_string_lossy().as_ref(),
            ],
        )?;
    }
    run_checked(
        "git",
        &[
            "worktree",
            "add",
            "--detach",
            dir.to_string_lossy().as_ref(),
            git_ref,
        ],
    )?;
    overlay_harness_crate(&dir)?;
    let absolute_run_root = std::env::current_dir()?.join(run_root);
    let before = existing_run_dirs(&absolute_run_root);
    let status = ProcessCommand::new("cargo")
        .current_dir(&dir)
        .args([
            "run",
            "--release",
            "-p",
            "supersonic-kernel-lab",
            "--bin",
            "kernel-lab",
            "--",
            "run",
            "--tasks",
            tasks,
            "--backend",
            backend,
            "--device",
            &device.to_string(),
            "--warmup",
            &warmup.to_string(),
            "--iters",
            &iters.to_string(),
            "--run-root",
            absolute_run_root.to_string_lossy().as_ref(),
        ])
        .status()?;
    child_run_dir_after(
        status,
        &absolute_run_root,
        &before,
        &format!("ref {git_ref}"),
    )
}

fn run_has_failed_tasks(summary: &supersonic_kernel_lab::run::RunSummary) -> bool {
    summary.passed_tasks != summary.task_count
}

fn overlay_harness_crate(worktree: &Path) -> Result<()> {
    let source_root = std::env::current_dir()?;
    overlay_harness_crate_from(&source_root, worktree)
}

fn overlay_harness_crate_from(source_root: &Path, worktree: &Path) -> Result<()> {
    let src_crate = source_root.join("crates/kernel-lab");
    let dst_crate = worktree.join("crates/kernel-lab");
    if dst_crate.exists() {
        std::fs::remove_dir_all(&dst_crate)?;
    }
    copy_dir_recursive(&src_crate, &dst_crate)?;

    let workspace_toml = worktree.join("Cargo.toml");
    let mut cargo_toml = std::fs::read_to_string(&workspace_toml)?;
    if !cargo_toml.contains("\"crates/kernel-lab\"") {
        let marker = "members = [";
        let idx = cargo_toml
            .find(marker)
            .ok_or_else(|| anyhow!("workspace Cargo.toml has no members array"))?
            + marker.len();
        cargo_toml.insert_str(idx, "\n    \"crates/kernel-lab\",");
        std::fs::write(&workspace_toml, cargo_toml)?;
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn child_run_dir_after(
    status: ExitStatus,
    run_root: &PathBuf,
    before: &BTreeSet<PathBuf>,
    label: &str,
) -> Result<PathBuf> {
    match newest_run_dir_excluding(run_root, before) {
        Ok(path) => Ok(path),
        Err(err) if status.success() => Err(err),
        Err(_) => Err(anyhow!("{label} run failed before writing summary.json")),
    }
}

fn existing_run_dirs(run_root: &PathBuf) -> BTreeSet<PathBuf> {
    std::fs::read_dir(run_root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("summary.json").exists())
        .collect()
}

fn newest_run_dir_excluding(run_root: &PathBuf, before: &BTreeSet<PathBuf>) -> Result<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(run_root)?
        .filter_map(Result::ok)
        .filter(|entry| {
            let path = entry.path();
            path.join("summary.json").exists() && !before.contains(&path)
        })
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, entry.path()))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
        .pop()
        .map(|(_, path)| path)
        .ok_or_else(|| anyhow!("no new kernel-lab run found under {}", run_root.display()))
}

fn run_checked(cmd: &str, args: &[&str]) -> Result<()> {
    let status = ProcessCommand::new(cmd).args(args).status()?;
    if !status.success() {
        return Err(anyhow!("{cmd} failed"));
    }
    Ok(())
}

fn sanitize_ref(git_ref: &str) -> String {
    sanitize_label(git_ref)
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use supersonic_kernel_lab::run::{MetaJson, RunSummary, SCHEMA_VERSION};

    #[test]
    fn sanitize_ref_keeps_path_safe_characters_only() {
        assert_eq!(sanitize_ref("main"), "main");
        assert_eq!(
            sanitize_ref("feature/quant+bake@123"),
            "feature-quant-bake-123"
        );
        assert_eq!(sanitize_ref("origin/main~1"), "origin-main-1");
    }

    #[test]
    fn overlay_harness_crate_copies_crate_and_injects_workspace_member() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        let worktree = tmp.path().join("worktree");
        std::fs::create_dir_all(source.join("crates/kernel-lab/src")).unwrap();
        std::fs::create_dir_all(worktree.join("crates")).unwrap();

        std::fs::write(
            source.join("crates/kernel-lab/Cargo.toml"),
            "[package]\nname = \"supersonic-kernel-lab\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(
            source.join("crates/kernel-lab/src/lib.rs"),
            "pub fn marker() {}\n",
        )
        .unwrap();
        std::fs::write(
            worktree.join("Cargo.toml"),
            "[workspace]\nmembers = [\n    \"crates/gpu-hal\",\n]\n",
        )
        .unwrap();

        overlay_harness_crate_from(&source, &worktree).unwrap();

        let cargo_toml = std::fs::read_to_string(worktree.join("Cargo.toml")).unwrap();
        assert!(cargo_toml.contains("\"crates/kernel-lab\""));
        let lib = std::fs::read_to_string(worktree.join("crates/kernel-lab/src/lib.rs")).unwrap();
        assert!(lib.contains("marker"));

        overlay_harness_crate_from(&source, &worktree).unwrap();
        let cargo_toml = std::fs::read_to_string(worktree.join("Cargo.toml")).unwrap();
        assert_eq!(cargo_toml.matches("\"crates/kernel-lab\"").count(), 1);
    }

    #[test]
    fn run_has_failed_tasks_checks_all_selected_tasks() {
        let mut summary = test_summary(0, 0, 0, 0);
        assert!(!run_has_failed_tasks(&summary));

        summary = test_summary(1, 0, 0, 0);
        assert!(run_has_failed_tasks(&summary));

        summary = test_summary(2, 1, 1, 1);
        assert!(run_has_failed_tasks(&summary));

        summary = test_summary(2, 2, 1, 1);
        assert!(!run_has_failed_tasks(&summary));
    }

    #[test]
    fn newest_run_dir_excluding_returns_only_new_summaries() {
        let tmp = tempfile::tempdir().unwrap();
        let run_root = tmp.path().to_path_buf();
        let old = run_root.join("old");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("summary.json"), "{}").unwrap();

        let before = existing_run_dirs(&run_root);
        assert!(newest_run_dir_excluding(&run_root, &before).is_err());

        let new = run_root.join("new");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(new.join("summary.json"), "{}").unwrap();

        assert_eq!(newest_run_dir_excluding(&run_root, &before).unwrap(), new);
    }

    #[test]
    fn write_baseline_artifact_groups_by_arch_and_sanitizes_name() {
        let tmp = tempfile::tempdir().unwrap();
        let summary = test_summary(1, 1, 1, 1);
        let path = write_baseline_artifact(&summary, tmp.path(), Some("required/gfx+smoke"), true)
            .unwrap();

        assert_eq!(
            path.strip_prefix(tmp.path()).unwrap(),
            Path::new("gfx1100").join("required-gfx-smoke.summary.json")
        );
        assert!(path.exists());
        assert!(path.with_file_name("required-gfx-smoke.md").exists());
    }

    fn test_summary(
        task_count: usize,
        passed_tasks: usize,
        required_task_count: usize,
        passed_required_tasks: usize,
    ) -> RunSummary {
        RunSummary {
            schema_version: SCHEMA_VERSION,
            meta: MetaJson {
                schema_version: SCHEMA_VERSION,
                run_id: "test".into(),
                timestamp_utc: "2026-05-06T00:00:00Z".into(),
                git_sha: "test".into(),
                git_dirty: false,
                backend: "hip".into(),
                device: 0,
                arch: "gfx1100".into(),
                rocm_smi: String::new(),
            },
            task_count,
            passed_tasks,
            required_task_count,
            passed_required_tasks,
            tasks: Vec::new(),
        }
    }
}
