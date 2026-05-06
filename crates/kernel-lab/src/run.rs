use crate::registry::{all_tasks, TaskDef};
use anyhow::{anyhow, Context, Result};
use gpu_hal::Backend;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct KernelLabConfig {
    pub backend: Backend,
    pub device: usize,
    pub warmup: usize,
    pub iters: usize,
    pub run_root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaJson {
    pub schema_version: u32,
    pub run_id: String,
    pub timestamp_utc: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub backend: String,
    pub device: usize,
    pub arch: String,
    pub rocm_smi: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub name: String,
    pub shape: BTreeMap<String, usize>,
    pub correct: bool,
    pub max_abs: Option<f32>,
    pub max_rel: Option<f32>,
    pub min_cos: Option<f32>,
    pub exact: Option<bool>,
    pub warmup: usize,
    pub iters: usize,
    pub timing_source: String,
    pub median_us: f64,
    pub mean_us: f64,
    pub min_us: f64,
    pub p95_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub schema_version: u32,
    pub task_id: String,
    pub family: String,
    pub tags: Vec<String>,
    pub required: bool,
    pub correct: bool,
    pub cases: Vec<CaseResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub schema_version: u32,
    pub meta: MetaJson,
    pub task_count: usize,
    pub passed_tasks: usize,
    pub required_task_count: usize,
    pub passed_required_tasks: usize,
    pub tasks: Vec<TaskResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffTask {
    pub task_id: String,
    pub correct: bool,
    pub baseline_median_us: f64,
    pub candidate_median_us: f64,
    pub speedup: f64,
    pub regression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    pub correct: bool,
    pub fast_p: f64,
    pub geomean_speedup: f64,
    pub worst_regression: Option<DiffTask>,
    pub tasks: Vec<DiffTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub schema_version: u32,
    pub run_id: String,
    pub timestamp_utc: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub backend: String,
    pub device: usize,
    pub arch: String,
    pub required: String,
    pub tasks: BTreeMap<String, f64>,
}

pub fn resolve_tasks(spec: &str) -> Result<Vec<TaskDef>> {
    if spec == "all" {
        return Ok(all_tasks()
            .iter()
            .filter(|task| task.required)
            .copied()
            .collect());
    }
    if spec == "everything" {
        return Ok(all_tasks().to_vec());
    }
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for raw in spec.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        if let Some(tag) = item.strip_prefix("tag:") {
            let mut matched = false;
            for task in all_tasks()
                .iter()
                .filter(|task| task.tags.iter().any(|task_tag| *task_tag == tag))
            {
                matched = true;
                if seen.insert(task.id.to_string()) {
                    out.push(*task);
                }
            }
            if !matched {
                return Err(anyhow!("unknown task tag: {tag}"));
            }
            continue;
        }

        let task = crate::find_task(item).ok_or_else(|| anyhow!("unknown task: {item}"))?;
        if seen.insert(task.id.to_string()) {
            out.push(*task);
        }
    }
    Ok(out)
}

pub fn run_tasks(cfg: &KernelLabConfig, tasks: &[TaskDef]) -> Result<(PathBuf, RunSummary)> {
    if !gpu_hal::is_backend_compiled(cfg.backend) {
        return Err(anyhow!("backend {} is not compiled", cfg.backend));
    }
    gpu_hal::set_backend(cfg.backend);
    gpu_hal::set_device(cfg.device).context("set GPU device")?;

    let git_sha = capture(["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let git_dirty = !capture(["status", "--porcelain", "--untracked-files=no"])
        .unwrap_or_default()
        .is_empty();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let run_dir = allocate_run_dir(&cfg.run_root, &git_sha, &date);
    std::fs::create_dir_all(run_dir.join("tasks"))?;

    let meta = MetaJson {
        schema_version: SCHEMA_VERSION,
        run_id: run_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "kernel-lab-run".into()),
        timestamp_utc: chrono::Utc::now().to_rfc3339(),
        git_sha,
        git_dirty,
        backend: cfg.backend.to_string(),
        device: cfg.device,
        arch: capture_cmd("rocminfo", &[])
            .and_then(|s| {
                s.lines()
                    .filter_map(|line| line.trim().strip_prefix("Name:"))
                    .map(str::trim)
                    .find(|name| name.starts_with("gfx") || name.starts_with("sm"))
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".into()),
        rocm_smi: capture_cmd("rocm-smi", &["--showuse"]).unwrap_or_else(|| "unavailable".into()),
    };

    std::fs::write(
        run_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;

    let mut results = Vec::new();
    for task in tasks {
        let mut result = (task.run)(cfg).unwrap_or_else(|err| TaskResult {
            schema_version: SCHEMA_VERSION,
            task_id: task.id.to_string(),
            family: task.family.to_string(),
            tags: task.tags.iter().map(|s| s.to_string()).collect(),
            required: task.required,
            correct: false,
            cases: Vec::new(),
            error: Some(err.to_string()),
        });
        result.correct = result.error.is_none() && result.cases.iter().all(|case| case.correct);
        std::fs::write(
            run_dir
                .join("tasks")
                .join(format!("{}.json", result.task_id.replace('.', "_"))),
            serde_json::to_string_pretty(&result)?,
        )?;
        results.push(result);
    }

    let summary = RunSummary {
        schema_version: SCHEMA_VERSION,
        meta,
        task_count: results.len(),
        passed_tasks: results.iter().filter(|task| task.correct).count(),
        required_task_count: results.iter().filter(|task| task.required).count(),
        passed_required_tasks: results
            .iter()
            .filter(|task| task.required && task.correct)
            .count(),
        tasks: results,
    };
    std::fs::write(
        run_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    append_history(&cfg.run_root.join("history.jsonl"), &summary)?;
    Ok((run_dir, summary))
}

pub fn load_summary(path: &Path) -> Result<RunSummary> {
    let summary_path = if path.is_dir() {
        path.join("summary.json")
    } else {
        path.to_path_buf()
    };
    let bytes =
        std::fs::read(&summary_path).with_context(|| format!("read {}", summary_path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn diff_runs(baseline: &RunSummary, candidate: &RunSummary, max_regression: f64) -> DiffReport {
    let mut candidate_by_id = BTreeMap::new();
    for task in &candidate.tasks {
        candidate_by_id.insert(task.task_id.as_str(), task);
    }

    let mut tasks = Vec::new();
    for base_task in baseline.tasks.iter().filter(|task| task.required) {
        let Some(cand_task) = candidate_by_id.get(base_task.task_id.as_str()) else {
            let base_med = task_median(base_task);
            tasks.push(DiffTask {
                task_id: base_task.task_id.clone(),
                correct: false,
                baseline_median_us: base_med,
                candidate_median_us: 0.0,
                speedup: 0.0,
                regression: true,
            });
            continue;
        };
        let base_med = task_median(base_task);
        let cand_med = task_median(cand_task);
        let speedup = if cand_med > 0.0 {
            base_med / cand_med
        } else {
            0.0
        };
        let correct = base_task.correct && cand_task.correct;
        let regression = !correct || cand_med > base_med * (1.0 + max_regression);
        tasks.push(DiffTask {
            task_id: base_task.task_id.clone(),
            correct,
            baseline_median_us: base_med,
            candidate_median_us: cand_med,
            speedup,
            regression,
        });
    }

    let valid: Vec<_> = tasks.iter().filter(|task| task.correct).collect();
    let fast_count = valid.iter().filter(|task| task.speedup > 1.0).count();
    let fast_p = if tasks.is_empty() {
        0.0
    } else {
        fast_count as f64 / tasks.len() as f64
    };
    let geomean_speedup = if valid.is_empty() {
        0.0
    } else {
        (valid
            .iter()
            .map(|task| task.speedup.max(1e-9).ln())
            .sum::<f64>()
            / valid.len() as f64)
            .exp()
    };
    let worst_regression = tasks
        .iter()
        .filter(|task| task.regression)
        .min_by(|a, b| a.speedup.total_cmp(&b.speedup))
        .cloned();

    DiffReport {
        correct: worst_regression.is_none(),
        fast_p,
        geomean_speedup,
        worst_regression,
        tasks,
    }
}

pub fn diff_exit_code(diff: &DiffReport, min_speedup: f64) -> i32 {
    if diff.tasks.iter().any(|task| !task.correct) {
        2
    } else if diff.tasks.iter().any(|task| task.regression) {
        3
    } else if diff.geomean_speedup < min_speedup {
        4
    } else {
        0
    }
}

pub fn render_markdown(summary: &RunSummary) -> String {
    let mut s = String::new();
    s.push_str("# SuperSonic Kernel Lab\n\n");
    s.push_str(&format!(
        "- run: `{}`\n- git: `{}`{}\n- backend: `{}` device `{}` arch `{}`\n- required: {}/{}\n\n",
        summary.meta.run_id,
        summary.meta.git_sha,
        if summary.meta.git_dirty {
            " (dirty)"
        } else {
            ""
        },
        summary.meta.backend,
        summary.meta.device,
        summary.meta.arch,
        summary.passed_required_tasks,
        summary.required_task_count
    ));
    s.push_str("| task | correct | timing | median us |\n| --- | --- | --- | ---: |\n");
    for task in &summary.tasks {
        let timing_source = task
            .cases
            .first()
            .map(|case| case.timing_source.as_str())
            .unwrap_or("unknown");
        s.push_str(&format!(
            "| `{}` | {} | `{}` | {:.3} |\n",
            task.task_id,
            if task.correct { "yes" } else { "no" },
            timing_source,
            task_median(task)
        ));
    }
    s
}

pub fn render_diff_markdown(diff: &DiffReport, min_speedup: f64) -> String {
    let mut s = String::new();
    s.push_str("# SuperSonic Kernel Lab Diff\n\n");
    s.push_str(&format!(
        "- correct: `{}`\n- fast_p: `{:.3}`\n- geomean speedup: `{:.3}`\n- min speedup: `{:.3}`\n\n",
        diff.correct, diff.fast_p, diff.geomean_speedup, min_speedup
    ));
    s.push_str("| task | correct | baseline us | candidate us | speedup | regression |\n");
    s.push_str("| --- | --- | ---: | ---: | ---: | --- |\n");
    for task in &diff.tasks {
        s.push_str(&format!(
            "| `{}` | {} | {:.3} | {:.3} | {:.3} | {} |\n",
            task.task_id,
            if task.correct { "yes" } else { "no" },
            task.baseline_median_us,
            task.candidate_median_us,
            task.speedup,
            if task.regression { "yes" } else { "no" }
        ));
    }
    s
}

pub fn task_median(task: &TaskResult) -> f64 {
    let mut samples: Vec<f64> = task.cases.iter().map(|case| case.median_us).collect();
    median(&mut samples)
}

pub fn summarize_times_us(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = median(&mut sorted.clone());
    let mean = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    let min = sorted.first().copied().unwrap_or(0.0);
    let p95 = if sorted.is_empty() {
        0.0
    } else {
        sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize]
    };
    (median, mean, min, p95)
}

fn append_history(path: &Path, summary: &RunSummary) -> Result<()> {
    let mut tasks = BTreeMap::new();
    for task in &summary.tasks {
        tasks.insert(task.task_id.clone(), task_median(task));
    }
    let entry = HistoryEntry {
        schema_version: SCHEMA_VERSION,
        run_id: summary.meta.run_id.clone(),
        timestamp_utc: summary.meta.timestamp_utc.clone(),
        git_sha: summary.meta.git_sha.clone(),
        git_dirty: summary.meta.git_dirty,
        backend: summary.meta.backend.clone(),
        device: summary.meta.device,
        arch: summary.meta.arch.clone(),
        required: format!(
            "{}/{}",
            summary.passed_required_tasks, summary.required_task_count
        ),
        tasks,
    };
    let line = serde_json::to_string(&entry)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn allocate_run_dir(parent: &Path, git_sha: &str, today: &str) -> PathBuf {
    let base = format!("{today}-{git_sha}");
    let mut candidate = parent.join(&base);
    let mut n = 2;
    while candidate.exists() {
        candidate = parent.join(format!("{base}-{n}"));
        n += 1;
    }
    candidate
}

fn capture<const N: usize>(args: [&str; N]) -> Option<String> {
    capture_cmd("git", &args)
}

fn capture_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, correct: bool, median: f64) -> TaskResult {
        TaskResult {
            schema_version: SCHEMA_VERSION,
            task_id: id.into(),
            family: "test".into(),
            tags: Vec::new(),
            required: true,
            correct,
            cases: vec![CaseResult {
                name: "case".into(),
                shape: BTreeMap::new(),
                correct,
                max_abs: None,
                max_rel: None,
                min_cos: None,
                exact: Some(correct),
                warmup: 0,
                iters: 1,
                timing_source: "wall_sync".into(),
                median_us: median,
                mean_us: median,
                min_us: median,
                p95_us: median,
            }],
            error: None,
        }
    }

    fn summary(tasks: Vec<TaskResult>) -> RunSummary {
        RunSummary {
            schema_version: SCHEMA_VERSION,
            meta: MetaJson {
                schema_version: SCHEMA_VERSION,
                run_id: "test".into(),
                timestamp_utc: "now".into(),
                git_sha: "sha".into(),
                git_dirty: false,
                backend: "HIP".into(),
                device: 0,
                arch: "gfx1100".into(),
                rocm_smi: String::new(),
            },
            task_count: tasks.len(),
            passed_tasks: tasks.iter().filter(|task| task.correct).count(),
            required_task_count: tasks.len(),
            passed_required_tasks: tasks.iter().filter(|task| task.correct).count(),
            tasks,
        }
    }

    #[test]
    fn diff_flags_latency_regression() {
        let base = summary(vec![task("a", true, 100.0)]);
        let cand = summary(vec![task("a", true, 110.0)]);
        let diff = diff_runs(&base, &cand, 0.03);
        assert!(!diff.correct);
        assert!(diff.tasks[0].regression);
    }

    #[test]
    fn diff_accepts_correct_speedup() {
        let base = summary(vec![task("a", true, 100.0), task("b", true, 200.0)]);
        let cand = summary(vec![task("a", true, 80.0), task("b", true, 160.0)]);
        let diff = diff_runs(&base, &cand, 0.03);
        assert!(diff.correct);
        assert_eq!(diff.fast_p, 1.0);
        assert!(diff.geomean_speedup > 1.24 && diff.geomean_speedup < 1.26);
        assert!(diff.worst_regression.is_none());
    }

    #[test]
    fn diff_flags_correctness_regression() {
        let base = summary(vec![task("a", true, 100.0)]);
        let cand = summary(vec![task("a", false, 80.0)]);
        let diff = diff_runs(&base, &cand, 0.03);
        assert!(!diff.correct);
        assert!(diff.tasks[0].regression);
        assert_eq!(diff.fast_p, 0.0);
    }

    #[test]
    fn diff_flags_missing_required_candidate_task() {
        let base = summary(vec![task("a", true, 100.0), task("b", true, 200.0)]);
        let cand = summary(vec![task("a", true, 80.0)]);
        let diff = diff_runs(&base, &cand, 0.03);
        assert!(!diff.correct);
        assert_eq!(diff.tasks.len(), 2);
        assert_eq!(diff.tasks[1].task_id, "b");
        assert!(diff.tasks[1].regression);
        assert_eq!(diff.tasks[1].candidate_median_us, 0.0);
    }

    #[test]
    fn diff_exit_code_distinguishes_failure_modes() {
        let base = summary(vec![task("a", true, 100.0)]);
        let bad_correctness = summary(vec![task("a", false, 80.0)]);
        assert_eq!(
            diff_exit_code(&diff_runs(&base, &bad_correctness, 0.03), 1.02),
            2
        );

        let slow = summary(vec![task("a", true, 110.0)]);
        assert_eq!(diff_exit_code(&diff_runs(&base, &slow, 0.03), 1.02), 3);

        let flat = summary(vec![task("a", true, 100.0)]);
        assert_eq!(diff_exit_code(&diff_runs(&base, &flat, 0.03), 1.02), 4);
    }

    #[test]
    fn load_summary_accepts_file_or_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let summary = summary(vec![task("a", true, 100.0)]);
        let path = tmp.path().join("summary.json");
        std::fs::write(&path, serde_json::to_string_pretty(&summary).unwrap()).unwrap();

        let from_file = load_summary(&path).unwrap();
        let from_dir = load_summary(tmp.path()).unwrap();
        assert_eq!(from_file.task_count, 1);
        assert_eq!(from_dir.tasks[0].task_id, "a");
    }

    #[test]
    fn render_markdown_includes_run_and_task_rows() {
        let summary = summary(vec![task("qwen35.full_attention_prefill", true, 123.4567)]);
        let md = render_markdown(&summary);
        assert!(md.contains("# SuperSonic Kernel Lab"));
        assert!(md.contains("arch `gfx1100`"));
        assert!(md.contains("`qwen35.full_attention_prefill`"));
        assert!(md.contains("`wall_sync`"));
        assert!(md.contains("123.457"));
    }

    #[test]
    fn render_diff_markdown_includes_task_rows() {
        let base = summary(vec![task("a", true, 100.0)]);
        let cand = summary(vec![task("a", true, 80.0)]);
        let md = render_diff_markdown(&diff_runs(&base, &cand, 0.03), 1.02);
        assert!(md.contains("# SuperSonic Kernel Lab Diff"));
        assert!(md.contains("`a`"));
        assert!(md.contains("1.250"));
    }

    #[test]
    fn summarize_times_handles_empty_and_even_samples() {
        assert_eq!(summarize_times_us(&[]), (0.0, 0.0, 0.0, 0.0));
        let (median, mean, min, p95) = summarize_times_us(&[4.0, 1.0, 2.0, 3.0]);
        assert_eq!(median, 2.5);
        assert_eq!(mean, 2.5);
        assert_eq!(min, 1.0);
        assert_eq!(p95, 4.0);
    }

    #[test]
    fn resolve_tasks_accepts_csv() {
        let tasks = resolve_tasks("qwen35.full_attention_prefill,qwen36.router_permute").unwrap();
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn resolve_tasks_all_means_required_suite() {
        let tasks = resolve_tasks("all").unwrap();
        assert_eq!(tasks.len(), 5);
        assert!(tasks.iter().all(|task| task.required));
    }

    #[test]
    fn resolve_tasks_accepts_tags_and_deduplicates() {
        let tasks = resolve_tasks("tag:prefill,qwen35.full_attention_prefill").unwrap();
        let ids: Vec<_> = tasks.iter().map(|task| task.id).collect();
        assert_eq!(
            ids,
            vec![
                "qwen35.full_attention_prefill",
                "qwen36.batched_prefill_attn_full"
            ]
        );
    }

    #[test]
    fn resolve_tasks_rejects_unknown_tag() {
        let err = resolve_tasks("tag:not-a-real-tag").unwrap_err();
        assert!(err.to_string().contains("unknown task tag"));
    }

    #[test]
    fn resolve_tasks_accepts_stress_tag() {
        let tasks = resolve_tasks("tag:stress").unwrap();
        assert_eq!(tasks.len(), 3);
        assert!(tasks.iter().all(|task| !task.required));
    }
}
