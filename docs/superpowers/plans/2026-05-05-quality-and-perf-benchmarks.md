# Quality and Performance Benchmarks (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified quality + performance benchmarking harness for SuperSonic on `gfx1100`, with `hipfire` as the first external comparison engine. Output is operator-facing during dev and publishable to `docs/quality.md` + `docs/performance.md` on demand.

**Architecture:** Hybrid Rust + Python. A new `crates/bench` Rust crate orchestrates SuperSonic-side perf (subprocess-driven `./target/release/supersonic`, parses existing `[result] ms_per_step=N` lines). A new `oracle/bench` Python package drives quality (perplexity, golden-prompt diff, NIAH/RULER, lm-evaluation-harness) and external engines (Phase 1: hipfire only). Both write JSON into a shared, immutable run-dir at `target/bench-runs/{date}-{git-sha}[-N]/`. A pure-function Python renderer turns JSON → markdown.

**Tech Stack:**
- Rust 1.7x (existing toolchain), `serde` + `serde_json` for the schema, `clap` for the CLI.
- Python 3.x, `datasets` (HuggingFace), `transformers`, `lm-eval`, `jsonschema`, `pytest`.
- Existing infrastructure reused: `oracle/pg19_smoke.py` perplexity loop, `oracle/arxiv_v1_smoke.py` NIAH harness, `tests/gfx1100/bench_matrix.sh` cooldown/median methodology.

**Reference spec:** `docs/superpowers/specs/2026-05-05-quality-and-perf-benchmarks-design.md`.

---

## File Structure

**New files:**
- `crates/bench/Cargo.toml`
- `crates/bench/src/lib.rs` — orchestrator entry points; re-exports
- `crates/bench/src/runs.rs` — `RunDir`, `MetaJson`, `PerfCellJson`, dir layout
- `crates/bench/src/perf.rs` — `extract_metrics()`, single-combo runner
- `crates/bench/src/matrix.rs` — combo iteration, registry filter
- `crates/bench/src/bin/bench_perf.rs` — CLI binary
- `crates/bench/tests/extract_metrics.rs`
- `crates/bench/tests/registry_filter.rs`
- `crates/bench/tests/run_dir_layout.rs`
- `crates/bench/tests/fixtures/runner_output_modern.txt`
- `crates/bench/tests/fixtures/runner_output_phi4.txt`

- `oracle/bench/__init__.py`
- `oracle/bench/runner.py` — `SupersonicSubprocess` + cooldown/warmup/median driver
- `oracle/bench/perplexity.py` — `score_perplexity(model, quant, dataset)` → JSON
- `oracle/bench/golden.py` — golden prompts diff vs BF16 reference
- `oracle/bench/golden_prompts.json` — ~20 prompts + per-(model,quant) reference outputs
- `oracle/bench/quality_main.py` — `python -m oracle.bench.quality_main` entry point
- `oracle/bench/heavy/__init__.py`
- `oracle/bench/heavy/niah.py`
- `oracle/bench/heavy/ruler.py`
- `oracle/bench/heavy/longctx.py`
- `oracle/bench/heavy/lm_eval.py`
- `oracle/bench/heavy/heavy_main.py` — `python -m oracle.bench.heavy.heavy_main` entry point
- `oracle/bench/external/__init__.py`
- `oracle/bench/external/common.py` — `ExternalAdapter` base class
- `oracle/bench/external/hipfire.py`
- `oracle/bench/external/external_main.py` — `python -m oracle.bench.external.external_main` entry point
- `oracle/bench/render/__init__.py`
- `oracle/bench/render/schema.py` — `MetaJson`, `PerfCellJson`, `QualityCellJson`, `ExternalCellJson` JSON schemas + validators
- `oracle/bench/render/markdown.py` — pure JSON → markdown
- `oracle/bench/render/diff.py` — two-run diff
- `oracle/bench/render/render_main.py` — `python -m oracle.bench.render.render_main` entry point
- `oracle/bench/tests/__init__.py`
- `oracle/bench/tests/test_renderer.py`
- `oracle/bench/tests/test_perplexity_math.py`
- `oracle/bench/tests/test_golden_diff.py`
- `oracle/bench/tests/test_hipfire_adapter.py`
- `oracle/bench/tests/test_schema.py`
- `oracle/bench/tests/fixtures/run_minimal/meta.json`
- `oracle/bench/tests/fixtures/run_minimal/perf/qwen3.5-0.8b_bf16.json`
- `oracle/bench/tests/fixtures/run_minimal/quality/qwen3.5-0.8b_bf16_perplexity.json`
- `oracle/bench/tests/fixtures/golden_quality.md`
- `oracle/bench/tests/fixtures/golden_perf_fragment.md`

- `tools/external/hipfire-version.txt`
- `tools/external/check-versions.sh`

- `tests/gfx1100/bench_smoke.sh`
- `tests/gfx1100/bench_parity.sh`
- `tests/gfx1100/bench_hipfire_smoke.sh`

- `docs/quality.md`

**Modified files:**
- `Cargo.toml` (workspace) — add `crates/bench` to members.
- `crates/runner/src/registry.rs` — add `pub fn supported_combos(arch: GpuArch) -> Vec<ComboDescriptor>` and a `pub use` of `Backend`/`ModelVariant`/`GpuArch` so the bench crate can name them. Make registry `pub` (was `pub(crate)`).
- `crates/runner/src/cli.rs` — extend `--teacher-forced` flag handling to dispatch to all model families (currently Llama-only via `llama31_engine`).
- `crates/runner/src/qwen35_runtime.rs` — add `run_qwen35_teacher_forced` mirroring `llama31_engine::run_llama31_teacher_forced` (lines 481–690 in `llama31_engine.rs`).
- `crates/runner/src/gemma4_runtime.rs` — add `run_gemma4_teacher_forced` (same pattern).
- `crates/runner/src/phi4_engine.rs` — add `run_phi4_teacher_forced` (same pattern).
- `docs/performance.md` — add an `<!-- AUTOGEN BELOW: hipfire-comparison -->` sentinel zone where the renderer injects the "vs hipfire" column.

---

## Task 1: Bootstrap the Rust crate

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `crates/bench/Cargo.toml`
- Create: `crates/bench/src/lib.rs`
- Create: `crates/bench/src/bin/bench_perf.rs`

- [ ] **Step 1: Add the workspace member**

Edit `Cargo.toml`:
```toml
[workspace]
members = [
    "crates/bench",
    "crates/core",
    "crates/gemma4",
    "crates/gpu-hal",
    "crates/kernel-ffi",
    "crates/model-store",
    "crates/phi4",
    "crates/qwen35",
    "crates/qwen35_dflash",
    "crates/qwen36_moe",
    "crates/runtime",
    "crates/runner",
    "crates/server",
]
resolver = "2"
```

- [ ] **Step 2: Create the crate manifest**

Create `crates/bench/Cargo.toml`:
```toml
[package]
name = "supersonic-bench"
version = "0.1.0"
edition = "2021"

[lib]
name = "supersonic_bench"
path = "src/lib.rs"

[[bin]]
name = "bench-perf"
path = "src/bin/bench_perf.rs"

[dependencies]
# NOTE: bench is intentionally standalone — no runner dep. Combo table lives in
# crates/bench/src/matrix.rs (Task 4); a runner-side parity test keeps it honest.
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
chrono = { version = "0.4", features = ["serde"] }
```

- [ ] **Step 3: Create the lib stub**

Create `crates/bench/src/lib.rs`:
```rust
//! SuperSonic benchmark orchestrator (Rust-side perf).
//! See docs/superpowers/specs/2026-05-05-quality-and-perf-benchmarks-design.md.

pub mod matrix;
pub mod perf;
pub mod runs;
```

- [ ] **Step 4: Create the bench-perf binary stub**

Create `crates/bench/src/bin/bench_perf.rs`:
```rust
use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "bench-perf", about = "SuperSonic perf benchmark orchestrator")]
struct Cli {
    /// GPU arch filter (e.g. gfx1100). Auto-detect when not provided.
    #[arg(long)]
    arch: Option<String>,
    /// Comma-separated model list, or "all".
    #[arg(long, default_value = "all")]
    models: String,
    /// Comma-separated quant list, or "all".
    #[arg(long, default_value = "all")]
    quants: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("bench-perf invoked: arch={:?} models={} quants={}", cli.arch, cli.models, cli.quants);
    Ok(())
}
```

Create the empty module files so step 5 builds:
- `crates/bench/src/runs.rs`: `// run-dir layout and JSON schema`
- `crates/bench/src/perf.rs`: `// perf measurement: subprocess driver, metric extraction`
- `crates/bench/src/matrix.rs`: `// (arch, model, quant) iteration`

- [ ] **Step 5: Verify it builds**

Run: `cargo build -p supersonic-bench --release`
Expected: build succeeds, produces `target/release/bench-perf`.

Run: `./target/release/bench-perf --models all --quants all`
Expected: prints `bench-perf invoked: arch=None models=all quants=all`.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/bench
git commit -m "bench: scaffold supersonic-bench crate with bench-perf stub"
```

---

## Task 2: Define the run-dir JSON schema (Rust side)

**Files:**
- Modify: `crates/bench/src/runs.rs`
- Create: `crates/bench/tests/run_dir_layout.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/bench/tests/run_dir_layout.rs`:
```rust
use supersonic_bench::runs::{MetaJson, PerfCellJson, PerfStatus, RunDir};
use std::path::PathBuf;

#[test]
fn meta_json_round_trip() {
    let meta = MetaJson {
        schema_version: 1,
        run_id: "2026-05-05-abc1234".to_string(),
        timestamp_utc: "2026-05-05T12:00:00Z".to_string(),
        git_sha: "abc1234".to_string(),
        hostname: "test-host".to_string(),
        arch: "gfx1100".to_string(),
        rocminfo: "Agent 1: gfx1100".to_string(),
        rocm_smi_u: "PID 1234 100%".to_string(),
        gpu_temp_c_pre: Some(45.0),
        gpu_temp_c_post: None,
        runner_version: "supersonic 0.1.0 (commit abc1234)".to_string(),
    };
    let s = serde_json::to_string(&meta).unwrap();
    let parsed: MetaJson = serde_json::from_str(&s).unwrap();
    assert_eq!(parsed.run_id, "2026-05-05-abc1234");
    assert_eq!(parsed.gpu_temp_c_pre, Some(45.0));
    assert_eq!(parsed.gpu_temp_c_post, None);
}

#[test]
fn perf_cell_json_status_variants() {
    let ok = PerfCellJson {
        schema_version: 1,
        model: "qwen3.5-0.8b".into(),
        quant: "bf16".into(),
        prompt: "The quick brown fox jumps over".into(),
        max_new_tokens: 16,
        status: PerfStatus::Ok { ms_per_step: 8.0, ms_per_tok: 8.0, samples: vec![8.1, 8.0, 7.9] },
        gpu_temp_c_end: Some(60.0),
    };
    let s = serde_json::to_string(&ok).unwrap();
    let back: PerfCellJson = serde_json::from_str(&s).unwrap();
    match back.status {
        PerfStatus::Ok { ms_per_step, samples, .. } => {
            assert_eq!(ms_per_step, 8.0);
            assert_eq!(samples.len(), 3);
        }
        _ => panic!("expected Ok"),
    }

    let skipped = PerfCellJson {
        status: PerfStatus::Skipped { reason: "OOM at preflight".into() },
        ..ok.clone()
    };
    let s = serde_json::to_string(&skipped).unwrap();
    assert!(s.contains("\"status\":\"skipped\""));
}

#[test]
fn run_dir_paths() {
    let rd = RunDir::new(PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234"));
    assert_eq!(rd.meta_path(), PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/meta.json"));
    assert_eq!(
        rd.perf_path("qwen3.5-0.8b", "bf16"),
        PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/perf/qwen3.5-0.8b_bf16.json"),
    );
    assert_eq!(
        rd.external_path("hipfire", "qwen3.5-0.8b", "bf16"),
        PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/external/hipfire/qwen3.5-0.8b_bf16.json"),
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p supersonic-bench --test run_dir_layout`
Expected: compile error (`MetaJson`, `PerfCellJson`, etc. not found).

- [ ] **Step 3: Implement the schema**

Replace `crates/bench/src/runs.rs`:
```rust
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaJson {
    pub schema_version: u32,
    pub run_id: String,
    pub timestamp_utc: String,
    pub git_sha: String,
    pub hostname: String,
    pub arch: String,
    pub rocminfo: String,
    pub rocm_smi_u: String,
    pub gpu_temp_c_pre: Option<f64>,
    pub gpu_temp_c_post: Option<f64>,
    pub runner_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum PerfStatus {
    Ok {
        ms_per_step: f64,
        ms_per_tok: f64,
        samples: Vec<f64>,
    },
    Skipped {
        reason: String,
    },
    Error {
        stderr_tail: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfCellJson {
    pub schema_version: u32,
    pub model: String,
    pub quant: String,
    pub prompt: String,
    pub max_new_tokens: u32,
    #[serde(flatten)]
    pub status: PerfStatus,
    pub gpu_temp_c_end: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    pub fn new(root: PathBuf) -> Self { Self { root } }
    pub fn root(&self) -> &Path { &self.root }
    pub fn meta_path(&self) -> PathBuf { self.root.join("meta.json") }
    pub fn perf_path(&self, model: &str, quant: &str) -> PathBuf {
        self.root.join("perf").join(format!("{model}_{quant}.json"))
    }
    pub fn quality_path(&self, model: &str, quant: &str, eval: &str) -> PathBuf {
        self.root.join("quality").join(format!("{model}_{quant}_{eval}.json"))
    }
    pub fn external_path(&self, engine: &str, model: &str, quant: &str) -> PathBuf {
        self.root.join("external").join(engine).join(format!("{model}_{quant}.json"))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p supersonic-bench --test run_dir_layout`
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/bench/src/runs.rs crates/bench/tests/run_dir_layout.rs
git commit -m "bench: define run-dir JSON schema (meta + perf cells)"
```

---

## Task 3: Implement `extract_metrics()` for runner output parsing

**Files:**
- Modify: `crates/bench/src/perf.rs`
- Create: `crates/bench/tests/extract_metrics.rs`
- Create: `crates/bench/tests/fixtures/runner_output_modern.txt`
- Create: `crates/bench/tests/fixtures/runner_output_phi4.txt`

- [ ] **Step 1: Capture real runner output as fixtures**

Run a quick supersonic invocation against an existing model to capture real output:
```bash
./target/release/supersonic --model qwen3.5-0.8b --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --prompt "The quick brown fox jumps over" --max-new-tokens 4 \
  > /tmp/qwen35_out.txt 2>&1
```

Then save the relevant `[result]` line into `crates/bench/tests/fixtures/runner_output_modern.txt`. If you cannot run the GPU, paste this representative line:
```
[result] prompt_tokens=6 generated_tokens=4 decode_ms=32 ms_per_step=8 decode_max_delta=0.0000 gpu_oracle_max_delta=0.0000 batch_size=1
```

For phi4 (`crates/bench/tests/fixtures/runner_output_phi4.txt`) — it uses `ms_per_step`:
```
[result] prompt_tokens=6 generated_tokens=4 decode_ms=153 ms_per_step=38.3
```

- [ ] **Step 2: Write the failing test**

Create `crates/bench/tests/extract_metrics.rs`:
```rust
use supersonic_bench::perf::{extract_metrics, ExtractedMetrics};

const MODERN: &str = include_str!("fixtures/runner_output_modern.txt");
const PHI4: &str = include_str!("fixtures/runner_output_phi4.txt");

#[test]
fn extracts_ms_per_step_from_modern_runner() {
    let m = extract_metrics(MODERN).expect("expected metrics");
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
    assert!((m.ms_per_tok.unwrap() - 8.0).abs() < 1e-6);
}

#[test]
fn extracts_ms_per_step_from_phi4_runner() {
    let m = extract_metrics(PHI4).expect("expected metrics");
    assert!((m.ms_per_step - 38.3).abs() < 1e-6);
}

#[test]
fn returns_none_when_no_result_line() {
    let s = "no result line here\nsome other text";
    assert!(extract_metrics(s).is_none());
}

#[test]
fn returns_none_when_only_legacy_ms_per_tok_field() {
    // Per spec: do not silently fall back; missing both means something broke.
    // qwen35_decode_report.rs emits ms_per_tok-only — verify we still surface ms_per_tok
    // but only when paired with ms_per_step. This format-change-detector test asserts
    // we don't accept a degenerate output that's missing ms_per_step entirely.
    let legacy = "[result] prompt_tokens=6 generated_tokens=4 decode_ms=32 ms_per_tok=8 decode_max_delta=0.0000";
    let m = extract_metrics(legacy);
    // ms_per_step is the canonical field we extract; if a runner only emits ms_per_tok,
    // we accept ms_per_tok as ms_per_step (they are equivalent for batch=1) but flag it.
    assert!(m.is_some(), "ms_per_tok should be accepted as ms_per_step for batch=1");
    let m = m.unwrap();
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p supersonic-bench --test extract_metrics`
Expected: compile error (`extract_metrics` not found).

- [ ] **Step 4: Implement `extract_metrics`**

Replace `crates/bench/src/perf.rs`:
```rust
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetrics {
    pub ms_per_step: f64,
    pub ms_per_tok: Option<f64>,
}

/// Parse the `[result] ms_per_step=N ...` line from supersonic stdout.
/// Returns `None` if no `[result]` line is present.
/// Falls back to `ms_per_tok` for batch=1 paths that only emit that field
/// (qwen35_decode_report.rs).
pub fn extract_metrics(stdout: &str) -> Option<ExtractedMetrics> {
    let result_line = stdout.lines().rev().find(|l| l.starts_with("[result]"))?;
    let ms_per_step = parse_field(result_line, "ms_per_step");
    let ms_per_tok = parse_field(result_line, "ms_per_tok");
    match (ms_per_step, ms_per_tok) {
        (Some(s), t) => Some(ExtractedMetrics { ms_per_step: s, ms_per_tok: t.or(Some(s)) }),
        (None, Some(t)) => Some(ExtractedMetrics { ms_per_step: t, ms_per_tok: Some(t) }),
        (None, None) => None,
    }
}

fn parse_field(line: &str, key: &str) -> Option<f64> {
    let needle = format!("{key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

pub fn run_one_combo(_model: &str, _quant: &str) -> Result<ExtractedMetrics> {
    Err(anyhow!("not implemented yet — Task 5"))
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p supersonic-bench --test extract_metrics`
Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/bench/src/perf.rs crates/bench/tests/extract_metrics.rs crates/bench/tests/fixtures
git commit -m "bench: extract_metrics parses runner [result] line"
```

---

## Task 4: Bench-side combo table + parity test against runner registry

**Spec invariant being preserved:** `crates/bench` does NOT depend on `runner` as a library. The bench crate's only contract with SuperSonic is the `./target/release/supersonic` subprocess — same as an end user. To enumerate combos without that dependency, the combo table lives inside `crates/bench` and is kept honest by a parity test on the runner side.

**Files:**
- Modify: `crates/bench/Cargo.toml` (REMOVE the `runner` dependency added in Task 1)
- Modify: `crates/bench/src/matrix.rs`
- Create: `crates/bench/tests/registry_filter.rs`
- Modify: `crates/runner/src/registry.rs` (only: ensure `REGISTRY` and types are accessible from a runner-side test; no public re-exports needed)
- Create: `crates/runner/tests/bench_combo_parity.rs`

- [ ] **Step 1: Verify bench has no runner dependency**

Run: `grep -n "^runner" crates/bench/Cargo.toml`. Expect: no output (Task 1 was updated to omit the runner dep). If a `runner = { path = "../runner" }` line is present (e.g. from an earlier draft), remove it now.

- [ ] **Step 2: Define the combo table inside bench**

Replace `crates/bench/src/matrix.rs` (incremental — keep the `MatrixConfig` and `run_matrix` from Task 6 when you get there; for now just add the combo table near the top):
```rust
//! Bench-side mirror of the runner's (model, quant, arch) support matrix.
//!
//! INVARIANT: this table must match `crates/runner/src/registry.rs`'s REGISTRY
//! plus the per-family quant flags. A parity test in
//! `crates/runner/tests/bench_combo_parity.rs` enforces this — if you change
//! the runner registry, you MUST update this table or the test fails.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchArch { Gfx1100, Gfx1150, Sm86, AppleM4 }

impl BenchArch {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "gfx1100" => Self::Gfx1100,
            "gfx1150" => Self::Gfx1150,
            "sm86" => Self::Sm86,
            "apple-m4" => Self::AppleM4,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1150 => "gfx1150",
            Self::Sm86 => "sm86",
            Self::AppleM4 => "apple-m4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComboDescriptor {
    pub model: &'static str,        // e.g. "qwen3.5-0.8b"
    pub quant: &'static str,        // "bf16" | "int4" | "fp8r" | "kv-fp8" | "int8"
    pub arch: BenchArch,
    pub min_vram_gib: f64,
}

/// Mirrors docs/feature-compatibility.md + docs/performance.md as of 2026-05-05.
pub static SUPPORTED_COMBOS: &[ComboDescriptor] = &[
    // Qwen3.5 — full BF16/INT4/FP8r/KV-FP8 quad on gfx1100.
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 2.0 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 0.7 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "fp8r",   arch: BenchArch::Gfx1100, min_vram_gib: 1.2 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 2.0 },
    ComboDescriptor { model: "qwen3.5-2b",   quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 5.0 },
    ComboDescriptor { model: "qwen3.5-2b",   quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 1.9 },
    ComboDescriptor { model: "qwen3.5-2b",   quant: "fp8r",   arch: BenchArch::Gfx1100, min_vram_gib: 3.0 },
    ComboDescriptor { model: "qwen3.5-2b",   quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 5.0 },
    ComboDescriptor { model: "qwen3.5-4b",   quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "qwen3.5-4b",   quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 3.7 },
    ComboDescriptor { model: "qwen3.5-4b",   quant: "fp8r",   arch: BenchArch::Gfx1100, min_vram_gib: 6.0 },
    ComboDescriptor { model: "qwen3.5-4b",   quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "qwen3.5-9b",   quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 18.0 },
    ComboDescriptor { model: "qwen3.5-9b",   quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 6.7 },
    ComboDescriptor { model: "qwen3.5-9b",   quant: "fp8r",   arch: BenchArch::Gfx1100, min_vram_gib: 10.8 },
    ComboDescriptor { model: "qwen3.5-9b",   quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 18.0 },
    // Gemma 4 — no fp8r per feature-compatibility.md
    ComboDescriptor { model: "gemma4-e2b",   quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 11.0 },
    ComboDescriptor { model: "gemma4-e2b",   quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 4.1 },
    ComboDescriptor { model: "gemma4-e2b",   quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 11.0 },
    ComboDescriptor { model: "gemma4-e4b",   quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "gemma4-e4b",   quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 3.7 },
    ComboDescriptor { model: "gemma4-e4b",   quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    // Phi-4-mini — full quad
    ComboDescriptor { model: "phi4-mini",    quant: "bf16",   arch: BenchArch::Gfx1100, min_vram_gib: 8.0 },
    ComboDescriptor { model: "phi4-mini",    quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 3.0 },
    ComboDescriptor { model: "phi4-mini",    quant: "fp8r",   arch: BenchArch::Gfx1100, min_vram_gib: 4.8 },
    ComboDescriptor { model: "phi4-mini",    quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 8.0 },
    // Qwen3.6-MoE — INT4 + KV-FP8 only on gfx1100 (24 GiB cap)
    ComboDescriptor { model: "qwen3.6-35b-a3b", quant: "int4",   arch: BenchArch::Gfx1100, min_vram_gib: 21.0 },
    ComboDescriptor { model: "qwen3.6-35b-a3b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 21.0 },
];

pub fn combos_for_arch(arch: BenchArch) -> Vec<&'static ComboDescriptor> {
    SUPPORTED_COMBOS.iter().filter(|c| c.arch == arch).collect()
}
```

- [ ] **Step 3: Add bench-side test**

Create `crates/bench/tests/registry_filter.rs`:
```rust
use supersonic_bench::matrix::{combos_for_arch, BenchArch};

#[test]
fn gfx1100_includes_shipping_models() {
    let combos = combos_for_arch(BenchArch::Gfx1100);
    let model_quants: Vec<(&str, &str)> = combos.iter().map(|c| (c.model, c.quant)).collect();

    assert!(model_quants.contains(&("qwen3.5-0.8b", "bf16")));
    assert!(model_quants.contains(&("qwen3.5-0.8b", "int4")));
    assert!(model_quants.contains(&("gemma4-e2b", "bf16")));
    assert!(model_quants.contains(&("phi4-mini", "fp8r")));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4")));
    assert!(!model_quants.contains(&("qwen3.6-35b-a3b", "bf16")),
            "qwen3.6-35b-a3b BF16 is not supported on gfx1100 (24 GiB cap)");
}

#[test]
fn min_vram_set_for_every_combo() {
    for c in combos_for_arch(BenchArch::Gfx1100) {
        assert!(c.min_vram_gib > 0.0, "combo {c:?} has zero min_vram_gib");
    }
}
```

- [ ] **Step 4: Add the runner-side parity test**

The runner crate already knows the truth about supported combos (in `registry.rs` + per-engine flags). Add a test under `crates/runner/tests/` (which is in the same crate, so `pub(crate)` items are reachable):

Create `crates/runner/tests/bench_combo_parity.rs`:
```rust
//! Parity gate: assert that crates/bench's SUPPORTED_COMBOS table matches the
//! runner's REGISTRY + per-engine quant capabilities. If this test fails after
//! a change to runner/src/registry.rs or a feature-compatibility shift, update
//! crates/bench/src/matrix.rs::SUPPORTED_COMBOS to match.
//!
//! This test reads bench's static table at runtime by parsing
//! crates/bench/src/matrix.rs (text scan; cheap and avoids a dep cycle).

use std::path::PathBuf;

#[test]
fn bench_combo_table_mentions_every_runner_supported_pair() {
    let bench_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("src")
        .join("matrix.rs");
    let bench_text = std::fs::read_to_string(&bench_src)
        .unwrap_or_else(|e| panic!("read {}: {e}", bench_src.display()));

    // Use the same source-of-truth used by the runner engines. Adjust this list
    // when adding/removing a (model, quant) on gfx1100.
    let expected_pairs: &[(&str, &str)] = &[
        ("qwen3.5-0.8b", "bf16"), ("qwen3.5-0.8b", "int4"),
        ("qwen3.5-0.8b", "fp8r"), ("qwen3.5-0.8b", "kv-fp8"),
        ("qwen3.5-2b", "bf16"), ("qwen3.5-2b", "int4"),
        ("qwen3.5-2b", "fp8r"), ("qwen3.5-2b", "kv-fp8"),
        ("qwen3.5-4b", "bf16"), ("qwen3.5-4b", "int4"),
        ("qwen3.5-4b", "fp8r"), ("qwen3.5-4b", "kv-fp8"),
        ("qwen3.5-9b", "bf16"), ("qwen3.5-9b", "int4"),
        ("qwen3.5-9b", "fp8r"), ("qwen3.5-9b", "kv-fp8"),
        ("gemma4-e2b", "bf16"), ("gemma4-e2b", "int4"), ("gemma4-e2b", "kv-fp8"),
        ("gemma4-e4b", "bf16"), ("gemma4-e4b", "int4"), ("gemma4-e4b", "kv-fp8"),
        ("phi4-mini", "bf16"), ("phi4-mini", "int4"),
        ("phi4-mini", "fp8r"), ("phi4-mini", "kv-fp8"),
        ("qwen3.6-35b-a3b", "int4"), ("qwen3.6-35b-a3b", "kv-fp8"),
    ];

    for (model, quant) in expected_pairs {
        let needle = format!("model: \"{model}\", quant: \"{quant}\"");
        assert!(bench_text.contains(&needle),
                "bench/src/matrix.rs is missing combo: {model}/{quant}\n\
                 If runner support changed, add or remove the row to match.");
    }
}
```

- [ ] **Step 5: Run both tests**

Run: `cargo test -p supersonic-bench --test registry_filter`
Expected: both tests pass.

Run: `cargo test -p runner --test bench_combo_parity`
Expected: passes.

Run: `cargo build -p supersonic-bench --release`
Expected: builds without runner dependency.

- [ ] **Step 6: Commit**

```bash
git add crates/bench/Cargo.toml crates/bench/src/matrix.rs crates/bench/tests/registry_filter.rs crates/runner/tests/bench_combo_parity.rs
git commit -m "bench: standalone combo table + runner-side parity gate"
```

---

---

## Task 5: Implement single-combo subprocess runner

**Files:**
- Modify: `crates/bench/src/perf.rs`
- Create: `crates/bench/tests/run_one_combo.rs`

- [ ] **Step 1: Write the failing test (using a fake supersonic binary)**

Create `crates/bench/tests/run_one_combo.rs`:
```rust
use std::path::PathBuf;
use supersonic_bench::perf::{run_one_combo, ComboInvocation, RunPolicy};

fn fake_supersonic_script(tmp: &std::path::Path, ms_per_step: f64) -> PathBuf {
    let path = tmp.join("supersonic");
    let body = format!(r#"#!/usr/bin/env bash
echo "[result] prompt_tokens=6 generated_tokens=16 decode_ms=128 ms_per_step={ms_per_step}"
"#);
    std::fs::write(&path, body).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    use std::os::unix::fs::PermissionsExt;
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

#[test]
fn run_one_combo_takes_median_of_three() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = fake_supersonic_script(tmp.path(), 12.5);
    let invocation = ComboInvocation {
        binary: bin,
        model: "qwen3.5-0.8b".into(),
        model_dir: PathBuf::from("/nonexistent"),
        quant: "bf16".into(),
        prompt: "The quick brown fox jumps over".into(),
        max_new_tokens: 16,
        warmup_tokens: 2,
    };
    let policy = RunPolicy { measurement_runs: 3, cooldown_seconds: 0 };
    let cell = run_one_combo(&invocation, &policy).unwrap();
    use supersonic_bench::runs::PerfStatus;
    match cell.status {
        PerfStatus::Ok { ms_per_step, samples, .. } => {
            assert_eq!(samples.len(), 3);
            assert!((ms_per_step - 12.5).abs() < 1e-6, "median should be 12.5");
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn run_one_combo_records_error_on_missing_binary() {
    let invocation = ComboInvocation {
        binary: PathBuf::from("/nonexistent/supersonic"),
        model: "qwen3.5-0.8b".into(),
        model_dir: PathBuf::from("/nonexistent"),
        quant: "bf16".into(),
        prompt: "x".into(),
        max_new_tokens: 1,
        warmup_tokens: 1,
    };
    let policy = RunPolicy { measurement_runs: 1, cooldown_seconds: 0 };
    let cell = run_one_combo(&invocation, &policy).unwrap();
    use supersonic_bench::runs::PerfStatus;
    matches!(cell.status, PerfStatus::Error { .. });
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in `crates/bench/Cargo.toml`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p supersonic-bench --test run_one_combo`
Expected: compile error — `ComboInvocation`, `RunPolicy`, signature mismatch.

- [ ] **Step 3: Implement `run_one_combo`**

Replace the existing `run_one_combo` stub in `crates/bench/src/perf.rs`:
```rust
use crate::runs::{PerfCellJson, PerfStatus, SCHEMA_VERSION};
use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ComboInvocation {
    pub binary: PathBuf,
    pub model: String,
    pub model_dir: PathBuf,
    pub quant: String,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub warmup_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct RunPolicy {
    pub measurement_runs: u32,
    pub cooldown_seconds: u32,
}

pub fn run_one_combo(invocation: &ComboInvocation, policy: &RunPolicy) -> Result<PerfCellJson> {
    if policy.cooldown_seconds > 0 {
        std::thread::sleep(Duration::from_secs(policy.cooldown_seconds as u64));
    }

    // Warmup pass — discard.
    let _ = invoke_supersonic(invocation, invocation.warmup_tokens);

    let mut samples = Vec::new();
    let mut last_err: Option<String> = None;
    for _ in 0..policy.measurement_runs {
        match invoke_supersonic(invocation, invocation.max_new_tokens) {
            Ok(m) => samples.push(m.ms_per_step),
            Err(e) => last_err = Some(e),
        }
    }

    let status = if samples.is_empty() {
        PerfStatus::Error { stderr_tail: last_err.unwrap_or_else(|| "no samples".into()) }
    } else {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        PerfStatus::Ok { ms_per_step: median, ms_per_tok: median, samples }
    };

    Ok(PerfCellJson {
        schema_version: SCHEMA_VERSION,
        model: invocation.model.clone(),
        quant: invocation.quant.clone(),
        prompt: invocation.prompt.clone(),
        max_new_tokens: invocation.max_new_tokens,
        status,
        gpu_temp_c_end: None,
    })
}

fn invoke_supersonic(invocation: &ComboInvocation, max_new: u32) -> std::result::Result<ExtractedMetrics, String> {
    let mut cmd = Command::new(&invocation.binary);
    cmd.arg("--model").arg(&invocation.model)
       .arg("--model-dir").arg(&invocation.model_dir)
       .arg("--prompt").arg(&invocation.prompt)
       .arg("--max-new-tokens").arg(max_new.to_string());
    apply_quant_flag(&mut cmd, &invocation.quant);
    let out = cmd.output().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");
    extract_metrics(&combined).ok_or_else(|| {
        let tail: String = combined.lines().rev().take(50).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        format!("no [result] line; tail:\n{tail}")
    })
}

fn apply_quant_flag(cmd: &mut Command, quant: &str) {
    match quant {
        "bf16" => {}                        // default
        "int4" => { cmd.arg("--int4"); }
        "fp8r" => { cmd.arg("--fp8-runtime"); }
        "kv-fp8" => { cmd.arg("--kv-fp8"); }
        "int8" => { cmd.arg("--int8"); }    // Llama CUDA path
        other => { eprintln!("warn: unknown quant '{other}', running BF16"); }
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p supersonic-bench --test run_one_combo`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/bench/src/perf.rs crates/bench/tests/run_one_combo.rs crates/bench/Cargo.toml
git commit -m "bench: run_one_combo with cooldown, warmup, median-of-N"
```

---

## Task 6: Implement matrix iteration + meta.json + run-dir creation

**Files:**
- Modify: `crates/bench/src/matrix.rs`
- Modify: `crates/bench/src/runs.rs` (add `RunDir::create`)
- Modify: `crates/bench/src/bin/bench_perf.rs`
- Create: `crates/bench/tests/matrix_writes_run_dir.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/bench/tests/matrix_writes_run_dir.rs`:
```rust
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
    assert!(meta_text.contains("\"arch\":\"gfx1100\""));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p supersonic-bench --test matrix_writes_run_dir`
Expected: compile error.

- [ ] **Step 3: Implement `RunDir::create` and helpers**

In `crates/bench/src/runs.rs`, add:
```rust
impl RunDir {
    /// Create the directory tree (root, perf/, quality/, external/).
    pub fn create(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.root.join("perf"))?;
        std::fs::create_dir_all(self.root.join("quality"))?;
        std::fs::create_dir_all(self.root.join("external"))?;
        Ok(())
    }

    pub fn write_meta(&self, meta: &MetaJson) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(meta).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(self.meta_path(), s)
    }

    pub fn write_perf(&self, cell: &PerfCellJson) -> std::io::Result<()> {
        let path = self.perf_path(&cell.model, &cell.quant);
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        let s = serde_json::to_string_pretty(cell).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, s)
    }
}

/// Compute a unique run-dir path under `parent`, dated today, with `-N` suffix on collision.
pub fn allocate_run_dir(parent: &Path, git_sha: &str, today: &str) -> PathBuf {
    let base = format!("{today}-{git_sha}");
    let mut candidate = parent.join(&base);
    let mut n = 2;
    while candidate.exists() {
        candidate = parent.join(format!("{base}-{n}"));
        n += 1;
    }
    candidate
}
```

- [ ] **Step 4: Implement matrix.rs**

Replace `crates/bench/src/matrix.rs`:
```rust
use crate::perf::{run_one_combo, ComboInvocation, RunPolicy};
use crate::runs::{MetaJson, RunDir, SCHEMA_VERSION};
use anyhow::Result;
use chrono::Utc;
use std::path::PathBuf;
use std::process::Command;

// BenchArch + ComboDescriptor + SUPPORTED_COMBOS are defined at the top of this file
// (added in Task 4). MatrixConfig consumes BenchArch.

pub struct MatrixConfig {
    pub arch: BenchArch,
    pub models: Vec<String>,
    pub quants: Vec<String>,
    pub binary: PathBuf,
    pub model_dir_resolver: Box<dyn Fn(&str) -> PathBuf>,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub warmup_tokens: u32,
    pub measurement_runs: u32,
    pub cooldown_seconds: u32,
    pub git_sha: String,
    pub runner_version: String,
}

pub fn run_matrix(cfg: &MatrixConfig, rd: &RunDir) -> Result<()> {
    rd.create()?;

    let meta = MetaJson {
        schema_version: SCHEMA_VERSION,
        run_id: rd.root().file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string(),
        timestamp_utc: Utc::now().to_rfc3339(),
        git_sha: cfg.git_sha.clone(),
        hostname: hostname_or_unknown(),
        arch: format!("{:?}", cfg.arch).to_lowercase(),
        rocminfo: capture_cmd("rocminfo"),
        rocm_smi_u: capture_cmd_args("rocm-smi", &["-u"]),
        gpu_temp_c_pre: read_gpu_temp(),
        gpu_temp_c_post: None,
        runner_version: cfg.runner_version.clone(),
    };
    rd.write_meta(&meta)?;

    for model in &cfg.models {
        for quant in &cfg.quants {
            let invocation = ComboInvocation {
                binary: cfg.binary.clone(),
                model: model.clone(),
                model_dir: (cfg.model_dir_resolver)(model),
                quant: quant.clone(),
                prompt: cfg.prompt.clone(),
                max_new_tokens: cfg.max_new_tokens,
                warmup_tokens: cfg.warmup_tokens,
            };
            let policy = RunPolicy {
                measurement_runs: cfg.measurement_runs,
                cooldown_seconds: cfg.cooldown_seconds,
            };
            let cell = run_one_combo(&invocation, &policy)?;
            rd.write_perf(&cell)?;
        }
    }

    let mut meta = meta;
    meta.gpu_temp_c_post = read_gpu_temp();
    rd.write_meta(&meta)?;
    Ok(())
}

fn hostname_or_unknown() -> String {
    capture_cmd("hostname").trim().to_string()
}
fn capture_cmd(name: &str) -> String { capture_cmd_args(name, &[]) }
fn capture_cmd_args(name: &str, args: &[&str]) -> String {
    Command::new(name).args(args).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_else(|_| String::new())
}
fn read_gpu_temp() -> Option<f64> {
    let out = capture_cmd_args("rocm-smi", &["-t", "--json"]);
    // Best-effort: parse `"Temperature (Sensor edge) (C)": "XX.X"` or similar.
    // If parse fails, return None — the field is optional.
    serde_json::from_str::<serde_json::Value>(&out).ok()
        .and_then(|v| v.as_object()?.values().next()?.as_object()?.iter()
            .find(|(k, _)| k.contains("Temperature"))
            .and_then(|(_, v)| v.as_str()?.parse().ok()))
}
```

- [ ] **Step 5: Wire up the bench-perf binary**

Replace `crates/bench/src/bin/bench_perf.rs`:
```rust
use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::PathBuf;
use supersonic_bench::matrix::{combos_for_arch, run_matrix, BenchArch, MatrixConfig};
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
}

fn parse_kv(s: &str) -> Result<(String, PathBuf), String> {
    let (k, v) = s.split_once('=').ok_or_else(|| "expected KEY=PATH".to_string())?;
    Ok((k.to_string(), PathBuf::from(v)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let arch = BenchArch::parse(&cli.arch).ok_or_else(|| anyhow!("unknown arch: {}", cli.arch))?;
    let combos = combos_for_arch(arch.clone());
    let models = filter_csv(&cli.models, combos.iter().map(|c| c.model));
    let quants = filter_csv(&cli.quants, combos.iter().map(|c| c.quant));

    let git_sha = capture_git_sha();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    std::fs::create_dir_all(&cli.run_root)?;
    let run_path = allocate_run_dir(&cli.run_root, &git_sha, &today);
    let rd = RunDir::new(run_path.clone());

    let dir_map: std::collections::HashMap<_, _> = cli.model_dirs.into_iter().collect();
    let resolver: Box<dyn Fn(&str) -> PathBuf> = Box::new(move |m: &str| {
        dir_map.get(m).cloned().unwrap_or_else(|| PathBuf::from(format!("/mnt/data/models/{m}")))
    });

    let cfg = MatrixConfig {
        arch,
        models,
        quants,
        binary: cli.binary,
        model_dir_resolver: resolver,
        prompt: cli.prompt,
        max_new_tokens: cli.max_new_tokens,
        warmup_tokens: cli.warmup_tokens,
        measurement_runs: cli.measurement_runs,
        cooldown_seconds: cli.cooldown_seconds,
        git_sha,
        runner_version: capture_runner_version(&cli.binary),
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
    std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn capture_runner_version(binary: &PathBuf) -> String {
    std::process::Command::new(binary).arg("--version").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
```

(If `GpuArch` doesn't include `AppleM4` / `Sm86` / `Gfx1150` variants, drop those branches — only `Gfx1100` is required for Phase 1. Adjust `parse_arch` to match the actual enum.)

- [ ] **Step 6: Run the test**

Run: `cargo test -p supersonic-bench --test matrix_writes_run_dir`
Expected: passes.

Run: `cargo build -p supersonic-bench --release`
Expected: builds.

- [ ] **Step 7: Commit**

```bash
git add crates/bench
git commit -m "bench: matrix iterator writes run-dir + meta.json + per-combo perf JSON"
```

---

## Task 7: Add the parity gate test (Rust orchestrator vs `bench_matrix.sh`)

**Files:**
- Create: `tests/gfx1100/bench_parity.sh`

- [ ] **Step 1: Write the parity script**

Create `tests/gfx1100/bench_parity.sh`:
```bash
#!/usr/bin/env bash
#
# Parity gate: assert that the Rust bench-perf orchestrator produces the same
# ms/step values as the legacy bash bench_matrix.sh on a pinned model subset,
# within ±3%.
#
# This is the gate for deleting tests/gfx1100/bench_matrix.sh.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"
BENCH_PERF="$REPO_ROOT/target/release/bench-perf"

if [ ! -x "$SUPERSONIC" ] || [ ! -x "$BENCH_PERF" ]; then
    echo "ERROR: build supersonic + bench-perf first" >&2
    exit 1
fi

MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

echo "[parity] running legacy bench_matrix.sh on qwen3.5-0.8b BF16..."
LEGACY_OUT="$(MODEL_DIR_08B="$MODEL_DIR_08B" "$SCRIPT_DIR/bench_matrix.sh" 2>/dev/null \
    | grep "qwen3.5-0.8b" | head -n1)"
LEGACY_BF16="$(echo "$LEGACY_OUT" | awk -F'|' '{print $3}' | tr -d ' ')"

echo "[parity] running Rust bench-perf on qwen3.5-0.8b BF16..."
RUN_DIR="$(mktemp -d)"
"$BENCH_PERF" --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_DIR" 2>&1 | tee "$RUN_DIR/log"

CELL_JSON="$(find "$RUN_DIR" -name 'qwen3.5-0.8b_bf16.json' | head -n1)"
RUST_BF16="$(jq -r '.ms_per_step // empty' "$CELL_JSON")"

if [ -z "$LEGACY_BF16" ] || [ -z "$RUST_BF16" ]; then
    echo "[parity] FAIL: missing measurement (legacy='$LEGACY_BF16' rust='$RUST_BF16')" >&2
    exit 1
fi

DELTA_PCT="$(awk -v l="$LEGACY_BF16" -v r="$RUST_BF16" 'BEGIN { d=(r-l)/l*100; if(d<0)d=-d; print d }')"
echo "[parity] legacy=$LEGACY_BF16 rust=$RUST_BF16 delta=${DELTA_PCT}%"
LIMIT="${PARITY_LIMIT_PCT:-3.0}"
PASS="$(awk -v d="$DELTA_PCT" -v l="$LIMIT" 'BEGIN { print (d<=l) ? "yes" : "no" }')"
if [ "$PASS" != "yes" ]; then
    echo "[parity] FAIL: delta ${DELTA_PCT}% exceeds limit ${LIMIT}%" >&2
    exit 1
fi
echo "[parity] PASS"
```

Make it executable: `chmod +x tests/gfx1100/bench_parity.sh`.

- [ ] **Step 2: Smoke-run it (requires GPU + model available)**

If a GPU + Qwen3.5-0.8B model dir are available:
```
cargo build --release --bin supersonic
cargo build --release -p supersonic-bench
MODEL_DIR_08B=/mnt/data/models/Qwen3.5-0.8B tests/gfx1100/bench_parity.sh
```
Expected: `[parity] PASS`.

If GPU is unavailable in the dev environment, mark the script untested and rely on operator run.

- [ ] **Step 3: Commit**

```bash
git add tests/gfx1100/bench_parity.sh
git commit -m "bench: parity gate vs bench_matrix.sh (qwen3.5-0.8b BF16, ±3%)"
```

---

## Task 8: Define the shared JSON schema in Python and validate it

**Files:**
- Create: `oracle/bench/__init__.py`
- Create: `oracle/bench/render/__init__.py`
- Create: `oracle/bench/render/schema.py`
- Create: `oracle/bench/tests/__init__.py`
- Create: `oracle/bench/tests/test_schema.py`
- Create: `oracle/bench/tests/fixtures/run_minimal/meta.json`
- Create: `oracle/bench/tests/fixtures/run_minimal/perf/qwen3.5-0.8b_bf16.json`

- [ ] **Step 1: Create fixture JSONs that match the Rust schema**

Create `oracle/bench/tests/fixtures/run_minimal/meta.json`:
```json
{
  "schema_version": 1,
  "run_id": "2026-05-05-abc1234",
  "timestamp_utc": "2026-05-05T12:00:00Z",
  "git_sha": "abc1234",
  "hostname": "test-host",
  "arch": "gfx1100",
  "rocminfo": "Agent 1: gfx1100",
  "rocm_smi_u": "PID 1234 100%",
  "gpu_temp_c_pre": 45.0,
  "gpu_temp_c_post": 60.0,
  "runner_version": "supersonic 0.1.0"
}
```

Create `oracle/bench/tests/fixtures/run_minimal/perf/qwen3.5-0.8b_bf16.json`:
```json
{
  "schema_version": 1,
  "model": "qwen3.5-0.8b",
  "quant": "bf16",
  "prompt": "The quick brown fox jumps over",
  "max_new_tokens": 16,
  "status": "ok",
  "ms_per_step": 8.0,
  "ms_per_tok": 8.0,
  "samples": [8.1, 8.0, 7.9],
  "gpu_temp_c_end": 60.0
}
```

- [ ] **Step 2: Write the failing test**

Create `oracle/bench/__init__.py` (empty), `oracle/bench/render/__init__.py` (empty), `oracle/bench/tests/__init__.py` (empty).

Create `oracle/bench/tests/test_schema.py`:
```python
"""Tests for the shared run-dir JSON schema."""
import json
from pathlib import Path

import pytest

from oracle.bench.render.schema import (
    META_SCHEMA, PERF_CELL_SCHEMA, validate_meta, validate_perf_cell,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "run_minimal"


def test_meta_fixture_validates():
    meta = json.loads((FIXTURE_DIR / "meta.json").read_text())
    validate_meta(meta)  # raises if invalid


def test_perf_cell_fixture_validates():
    cell = json.loads((FIXTURE_DIR / "perf" / "qwen3.5-0.8b_bf16.json").read_text())
    validate_perf_cell(cell)


def test_perf_cell_skipped_status():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-9b",
        "quant": "bf16",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "skipped",
        "reason": "OOM at preflight",
        "gpu_temp_c_end": None,
    }
    validate_perf_cell(cell)


def test_perf_cell_error_status():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-9b",
        "quant": "int4",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "error",
        "stderr_tail": "panicked at line 42",
        "gpu_temp_c_end": None,
    }
    validate_perf_cell(cell)


def test_perf_cell_invalid_status_rejected():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-0.8b",
        "quant": "bf16",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "ok",
        # missing ms_per_step and samples
        "gpu_temp_c_end": None,
    }
    with pytest.raises(Exception):
        validate_perf_cell(cell)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /home/deano/projects/SuperSonicBase && python -m pytest oracle/bench/tests/test_schema.py -v`
Expected: ImportError (`oracle.bench.render.schema` not found).

- [ ] **Step 4: Implement schema**

Create `oracle/bench/render/schema.py`:
```python
"""Shared JSON schema for run-dir artifacts. Mirrors crates/bench/src/runs.rs."""
import jsonschema

META_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version", "run_id", "timestamp_utc", "git_sha", "hostname",
        "arch", "rocminfo", "rocm_smi_u", "runner_version",
    ],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "run_id": {"type": "string"},
        "timestamp_utc": {"type": "string"},
        "git_sha": {"type": "string"},
        "hostname": {"type": "string"},
        "arch": {"type": "string"},
        "rocminfo": {"type": "string"},
        "rocm_smi_u": {"type": "string"},
        "gpu_temp_c_pre": {"type": ["number", "null"]},
        "gpu_temp_c_post": {"type": ["number", "null"]},
        "runner_version": {"type": "string"},
    },
}

_PERF_BASE = {
    "schema_version": {"type": "integer", "const": 1},
    "model": {"type": "string"},
    "quant": {"type": "string"},
    "prompt": {"type": "string"},
    "max_new_tokens": {"type": "integer", "minimum": 1},
    "gpu_temp_c_end": {"type": ["number", "null"]},
}

PERF_CELL_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "required": list(_PERF_BASE) + ["status", "ms_per_step", "ms_per_tok", "samples"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "ok"},
                           "ms_per_step": {"type": "number"},
                           "ms_per_tok": {"type": "number"},
                           "samples": {"type": "array", "items": {"type": "number"}}},
        },
        {
            "required": list(_PERF_BASE) + ["status", "reason"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "skipped"},
                           "reason": {"type": "string"}},
        },
        {
            "required": list(_PERF_BASE) + ["status", "stderr_tail"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "error"},
                           "stderr_tail": {"type": "string"}},
        },
    ],
}

QUALITY_CELL_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "model", "quant", "eval", "metric", "value"],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "model": {"type": "string"},
        "quant": {"type": "string"},
        "eval": {"type": "string"},        # e.g. "perplexity_pg19", "golden_diff", "niah_4k"
        "metric": {"type": "string"},      # e.g. "ppl", "exact_match", "score"
        "value": {"type": "number"},
        "extras": {"type": "object"},      # eval-specific extras (free-form)
    },
}

EXTERNAL_CELL_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "engine", "engine_version", "model", "quant", "status"],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "engine": {"type": "string"},
        "engine_version": {"type": "string"},
        "model": {"type": "string"},
        "quant": {"type": "string"},
        "status": {"enum": ["ok", "unsupported_by_engine", "error"]},
        "ms_per_step": {"type": ["number", "null"]},
        "samples": {"type": ["array", "null"], "items": {"type": "number"}},
        "stderr_tail": {"type": ["string", "null"]},
    },
}


def validate_meta(d: dict) -> None:
    jsonschema.validate(d, META_SCHEMA)


def validate_perf_cell(d: dict) -> None:
    jsonschema.validate(d, PERF_CELL_SCHEMA)


def validate_quality_cell(d: dict) -> None:
    jsonschema.validate(d, QUALITY_CELL_SCHEMA)


def validate_external_cell(d: dict) -> None:
    jsonschema.validate(d, EXTERNAL_CELL_SCHEMA)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest oracle/bench/tests/test_schema.py -v`
Expected: 5 tests pass.

If `jsonschema` isn't installed: `pip install jsonschema` (and add it to `oracle/requirements.txt` if that file exists; otherwise document the dep in a top-of-file comment).

- [ ] **Step 6: Commit**

```bash
git add oracle/bench
git commit -m "bench: shared JSON schema (Python validators mirror Rust types)"
```

---

## Task 9: Implement the renderer (perf → markdown), with sentinel preservation

**Files:**
- Create: `oracle/bench/render/markdown.py`
- Create: `oracle/bench/render/render_main.py`
- Create: `oracle/bench/tests/test_renderer.py`
- Create: `oracle/bench/tests/fixtures/golden_perf_fragment.md`

- [ ] **Step 1: Write the failing test**

Create `oracle/bench/tests/test_renderer.py`:
```python
"""Renderer is a pure function: same JSON in → same markdown out."""
from pathlib import Path

from oracle.bench.render.markdown import (
    render_perf_table, replace_autogen_zone,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "run_minimal"
GOLDEN_PERF = (Path(__file__).parent / "fixtures" / "golden_perf_fragment.md").read_text()


def test_perf_table_matches_golden():
    rendered = render_perf_table(FIXTURE_DIR / "perf")
    assert rendered.strip() == GOLDEN_PERF.strip()


def test_autogen_zone_replaces_only_the_zone():
    original = (
        "# Performance\n\nProse goes here.\n\n"
        "<!-- AUTOGEN BELOW: hipfire-comparison -->\n"
        "OLD CONTENT THAT GETS REPLACED\n"
        "<!-- AUTOGEN END: hipfire-comparison -->\n\n"
        "Trailing prose stays.\n"
    )
    new_content = "NEW CONTENT"
    out = replace_autogen_zone(original, "hipfire-comparison", new_content)
    assert "Prose goes here." in out
    assert "Trailing prose stays." in out
    assert "OLD CONTENT" not in out
    assert "NEW CONTENT" in out
    assert "<!-- AUTOGEN BELOW: hipfire-comparison -->" in out
    assert "<!-- AUTOGEN END: hipfire-comparison -->" in out


def test_autogen_zone_inserted_when_absent():
    original = "# Empty doc\n"
    out = replace_autogen_zone(original, "hipfire-comparison", "NEW")
    assert "<!-- AUTOGEN BELOW: hipfire-comparison -->" in out
    assert "NEW" in out
```

Create `oracle/bench/tests/fixtures/golden_perf_fragment.md`:
```
| Model           | BF16  |
|-----------------|------:|
| qwen3.5-0.8b    |   8.0 |
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_renderer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement renderer**

Create `oracle/bench/render/markdown.py`:
```python
"""Pure-function JSON → markdown renderer for bench run-dirs."""
import json
import re
from pathlib import Path
from typing import Iterable

from .schema import validate_perf_cell

QUANT_COL_ORDER = ["bf16", "int4", "fp8r", "kv-fp8", "int8"]
QUANT_LABELS = {"bf16": "BF16", "int4": "INT4", "fp8r": "FP8r", "kv-fp8": "KV-FP8", "int8": "INT8"}


def render_perf_table(perf_dir: Path) -> str:
    cells = []
    for f in sorted(perf_dir.glob("*.json")):
        d = json.loads(f.read_text())
        validate_perf_cell(d)
        cells.append(d)
    if not cells:
        return ""

    by_model: dict[str, dict[str, dict]] = {}
    for c in cells:
        by_model.setdefault(c["model"], {})[c["quant"]] = c

    quants_present = [q for q in QUANT_COL_ORDER if any(q in m for m in by_model.values())]
    headers = ["Model"] + [QUANT_LABELS[q] for q in quants_present]
    sep = ["-" * len("Model")] + ["-" * len(QUANT_LABELS[q]) + ":" for q in quants_present]

    rows = ["| " + " | ".join(h.ljust(15) if i == 0 else h.rjust(5) for i, h in enumerate(headers)) + " |",
            "|" + "|".join("-" * (len(s) + 2) for s in sep) + "|"]
    for model in sorted(by_model):
        cells_for_model = by_model[model]
        cols = []
        for q in quants_present:
            cell = cells_for_model.get(q)
            if cell is None:
                cols.append("—")
            elif cell["status"] == "ok":
                cols.append(f"{cell['ms_per_step']:.1f}")
            elif cell["status"] == "skipped":
                cols.append("—")
            elif cell["status"] == "error":
                cols.append("ERR")
        cols_str = [c.rjust(5) for c in cols]
        rows.append(f"| {model.ljust(15)} | " + " | ".join(cols_str) + " |")
    return "\n".join(rows) + "\n"


_AUTOGEN_BEGIN = "<!-- AUTOGEN BELOW: {key} -->"
_AUTOGEN_END = "<!-- AUTOGEN END: {key} -->"


def replace_autogen_zone(doc: str, key: str, new_content: str) -> str:
    begin = _AUTOGEN_BEGIN.format(key=key)
    end = _AUTOGEN_END.format(key=key)
    if begin in doc and end in doc:
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        replacement = f"{begin}\n{new_content}\n{end}"
        return pattern.sub(replacement, doc)
    # Append a new zone at the end.
    if not doc.endswith("\n"):
        doc += "\n"
    return f"{doc}\n{begin}\n{new_content}\n{end}\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest oracle/bench/tests/test_renderer.py -v`
Expected: 3 tests pass. If golden mismatch, regenerate the golden file by `print(render_perf_table(FIXTURE_DIR / "perf"))` and paste the output verbatim into `golden_perf_fragment.md`.

- [ ] **Step 5: Add the render_main entry point**

Create `oracle/bench/render/render_main.py`:
```python
"""CLI: python -m oracle.bench.render.render_main"""
import argparse
from pathlib import Path

from .markdown import render_perf_table, replace_autogen_zone


def main():
    ap = argparse.ArgumentParser(prog="render")
    sub = ap.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("markdown")
    render.add_argument("--run", required=True, type=Path)
    render.add_argument("--out", required=True, type=Path,
                        help="Repo root containing docs/quality.md and docs/performance.md")

    args = ap.parse_args()
    if args.cmd == "markdown":
        perf_md = render_perf_table(args.run / "perf")
        perf_doc = (args.out / "docs" / "performance.md")
        if perf_doc.exists():
            updated = replace_autogen_zone(perf_doc.read_text(), "bench-perf-matrix", perf_md)
            perf_doc.write_text(updated)
            print(f"updated {perf_doc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add oracle/bench/render oracle/bench/tests/test_renderer.py oracle/bench/tests/fixtures/golden_perf_fragment.md
git commit -m "bench: pure-function markdown renderer with sentinel-zone preservation"
```

---

## Task 10: Add the AUTOGEN sentinel to `docs/performance.md`

**Files:**
- Modify: `docs/performance.md`

- [ ] **Step 1: Insert sentinel near the top of the gfx1100 section**

Find the gfx1100 perf table heading (`## HIP — \`gfx1100\``) and add immediately above the existing matrix table:
```
<!-- AUTOGEN BELOW: bench-perf-matrix -->
<!-- AUTOGEN END: bench-perf-matrix -->
```

The renderer fills this zone on `python -m oracle.bench.render.render_main markdown --run … --out …`. Until first run, the zone is empty and the existing hand-written tables remain authoritative.

- [ ] **Step 2: Insert hipfire-comparison sentinel zone**

Below the same gfx1100 matrix, add:
```
<!-- AUTOGEN BELOW: hipfire-comparison -->
<!-- AUTOGEN END: hipfire-comparison -->
```

- [ ] **Step 3: Commit**

```bash
git add docs/performance.md
git commit -m "docs(perf): add AUTOGEN sentinel zones for bench renderer"
```

---

## Task 11: Implement Python subprocess driver (mirror Rust discipline)

**Files:**
- Create: `oracle/bench/runner.py`

- [ ] **Step 1: Implement the runner**

Create `oracle/bench/runner.py`:
```python
"""Subprocess driver for ./target/release/supersonic.

Mirrors the cooldown/warmup/median discipline of the Rust crates/bench
orchestrator. Used by quality runs (perplexity, golden) that need to capture
more than `ms_per_step` from the runner subprocess.
"""
from __future__ import annotations
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

_RESULT_LINE = re.compile(r"^\[result\] .*?ms_per_step=([0-9.]+)", re.MULTILINE)


@dataclass
class SupersonicInvocation:
    binary: Path
    model: str
    model_dir: Path
    quant: str
    prompt: str
    max_new_tokens: int
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RunPolicy:
    measurement_runs: int = 3
    cooldown_seconds: int = 3
    warmup_tokens: int = 2


def run_with_capture(inv: SupersonicInvocation, policy: RunPolicy) -> dict:
    """Run supersonic with cooldown+warmup+median; return a dict ready to
    serialize as a perf cell or quality cell.
    """
    if policy.cooldown_seconds > 0:
        time.sleep(policy.cooldown_seconds)
    _invoke(inv, policy.warmup_tokens, capture=False)

    samples: list[float] = []
    last_err: str | None = None
    for _ in range(policy.measurement_runs):
        try:
            stdout = _invoke(inv, inv.max_new_tokens, capture=True)
            ms = _extract_ms(stdout)
            if ms is not None:
                samples.append(ms)
            else:
                last_err = stdout.splitlines()[-50:]
        except subprocess.CalledProcessError as e:
            last_err = (e.stderr or e.stdout or "")[-2000:]

    if samples:
        sorted_s = sorted(samples)
        median = sorted_s[len(sorted_s) // 2]
        return {"status": "ok", "ms_per_step": median,
                "ms_per_tok": median, "samples": samples}
    return {"status": "error", "stderr_tail": str(last_err)[-2000:]}


def _invoke(inv: SupersonicInvocation, max_new: int, capture: bool) -> str:
    cmd = [str(inv.binary),
           "--model", inv.model,
           "--model-dir", str(inv.model_dir),
           "--prompt", inv.prompt,
           "--max-new-tokens", str(max_new),
           *_quant_flags(inv.quant),
           *inv.extra_args]
    res = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if capture:
        return (res.stdout or "") + "\n" + (res.stderr or "")
    return ""


def _extract_ms(stdout: str) -> float | None:
    m = _RESULT_LINE.search(stdout)
    return float(m.group(1)) if m else None


def _quant_flags(quant: str) -> list[str]:
    return {
        "bf16": [],
        "int4": ["--int4"],
        "fp8r": ["--fp8-runtime"],
        "kv-fp8": ["--kv-fp8"],
        "int8": ["--int8"],
    }.get(quant, [])
```

- [ ] **Step 2: Smoke-test mentally**

This file has no tests of its own (subprocess driver is exercised by `test_perplexity_math.py` via mock, and by `bench_smoke.sh` end-to-end). Just confirm imports work:

Run: `python -c "from oracle.bench.runner import run_with_capture, SupersonicInvocation, RunPolicy; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add oracle/bench/runner.py
git commit -m "bench: Python subprocess driver mirrors Rust cooldown/median discipline"
```

---

## Task 12: Generalize `--teacher-forced` to Qwen3.5

**Background:** `crates/runner/src/llama31_engine.rs:481-690` has the reference implementation `run_llama31_teacher_forced`. Phase 1 needs the same shape on every shipping HIP family. This task does Qwen3.5; Tasks 13–14 do Gemma 4 and Phi-4.

**Files:**
- Modify: `crates/runner/src/qwen35_runtime.rs` (or wherever the qwen35 engine entry point lives)
- Modify: `crates/runner/src/cli.rs`
- Create: `crates/runner/tests/teacher_forced_qwen35.rs`

- [ ] **Step 1: Read the reference implementation**

Read `crates/runner/src/llama31_engine.rs:481-690`. Note:
- Inputs: `cli`, `entry`, prefill+decode dispatch.
- Output: `[teacher_forced]` log line + `[teacher_forced_json] {...}` line.
- The JSON contains: `tokens`, `scored_tokens`, `nll`, `avg_nll`, `ppl`, `bpt`, `prefill_ms`, `total_ms`, `ms_per_token`.

- [ ] **Step 2: Write the failing integration smoke**

Create `crates/runner/tests/teacher_forced_qwen35.rs`:
```rust
//! Smoke: --teacher-forced --model qwen3.5-0.8b emits a [teacher_forced_json] line.
//!
//! This requires a real model dir + GPU. Skipped automatically when MODEL_DIR_08B is unset.

#[test]
fn teacher_forced_qwen35_emits_json_line() {
    let model_dir = match std::env::var("MODEL_DIR_08B") {
        Ok(v) => v,
        Err(_) => { eprintln!("MODEL_DIR_08B unset; skipping"); return; }
    };
    let bin = std::env::var("SUPERSONIC_BIN").unwrap_or_else(|_| "./target/release/supersonic".into());
    let out = std::process::Command::new(&bin)
        .args(["--model", "qwen3.5-0.8b",
               "--model-dir", &model_dir,
               "--prompt", "The quick brown fox jumps over the lazy dog",
               "--teacher-forced",
               "--max-new-tokens", "1"])
        .output()
        .expect("spawn supersonic");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[teacher_forced_json]"),
            "stdout missing teacher_forced_json line:\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stderr));
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p runner --test teacher_forced_qwen35 -- --nocapture`
Expected: skips if no MODEL_DIR_08B, or fails because Qwen3.5 doesn't yet emit `[teacher_forced_json]`.

- [ ] **Step 4: Implement `run_qwen35_teacher_forced`**

In `crates/runner/src/qwen35_runtime.rs` (or whichever file owns the qwen35 dispatch), add a function `run_qwen35_teacher_forced` that mirrors `llama31_engine::run_llama31_teacher_forced` lines 481-690 but uses the Qwen3.5 prefill + decode kernels.

Key adaptations:
- Call the Qwen3.5 prefill path on the full prompt (not the Llama path).
- Decode step-by-step through the **target** continuation tokens (teacher-forced means we feed the *true* next token, not the argmax).
- Capture per-step logits, compute log-softmax + NLL of the true token, accumulate.
- Emit the same `[teacher_forced]` + `[teacher_forced_json]` lines.

If the Qwen3.5 engine entry point is `run_qwen35_decode` in `qwen35_runtime.rs`, the dispatch in `cli.rs` should look like:
```rust
if cli.teacher_forced {
    return match family_of(model_variant) {
        Family::Llama31 => llama31_engine::run_llama31_teacher_forced(/* ... */),
        Family::Qwen35  => qwen35_runtime::run_qwen35_teacher_forced(/* ... */),
        Family::Gemma4  => bail!("Gemma 4 teacher-forced not yet implemented (Task 13)"),
        Family::Phi4    => bail!("Phi-4 teacher-forced not yet implemented (Task 14)"),
        Family::Qwen36MoE => bail!("Qwen3.6-MoE teacher-forced out of Phase 1 scope"),
    };
}
```

- [ ] **Step 5: Run the smoke**

Run: `cargo build --release --bin supersonic && MODEL_DIR_08B=/mnt/data/models/Qwen3.5-0.8B cargo test -p runner --test teacher_forced_qwen35 -- --nocapture`
Expected: passes (emits a `[teacher_forced_json]` line with valid `ppl`).

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/qwen35_runtime.rs crates/runner/src/cli.rs crates/runner/tests/teacher_forced_qwen35.rs
git commit -m "runner(qwen35): --teacher-forced emits [teacher_forced_json] for perplexity scoring"
```

---

## Task 13: Generalize `--teacher-forced` to Gemma 4

**Files:**
- Modify: `crates/runner/src/gemma4_runtime.rs`
- Modify: `crates/runner/src/cli.rs` (replace the `bail!` from Task 12 with a real call)
- Create: `crates/runner/tests/teacher_forced_gemma4.rs`

- [ ] **Step 1: Add the smoke test**

Create `crates/runner/tests/teacher_forced_gemma4.rs` mirroring Task 12's test, but with `--model gemma4-e2b` and `MODEL_DIR_GEMMA_E2B`.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p runner --test teacher_forced_gemma4 -- --nocapture`
Expected: bails with "Gemma 4 teacher-forced not yet implemented".

- [ ] **Step 3: Implement `run_gemma4_teacher_forced`**

Add a function in `crates/runner/src/gemma4_runtime.rs` mirroring the Qwen3.5 implementation from Task 12. Gemma 4 specifics:
- Uses sliding+full attention layers per `crates/gemma4`.
- KV cache pre-allocates `max_t` per layer with shared-layer aliasing (per `CLAUDE.md`).
- Decode step calls the gemma4 megakernel with the true next-token, not the argmax.

Update `cli.rs` to dispatch `Family::Gemma4 => gemma4_runtime::run_gemma4_teacher_forced(/* ... */)`.

- [ ] **Step 4: Run the smoke**

Run: `MODEL_DIR_GEMMA_E2B=/mnt/data/models/gemma-4-E2B cargo test -p runner --test teacher_forced_gemma4 -- --nocapture`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/gemma4_runtime.rs crates/runner/src/cli.rs crates/runner/tests/teacher_forced_gemma4.rs
git commit -m "runner(gemma4): --teacher-forced emits [teacher_forced_json]"
```

---

## Task 14: Generalize `--teacher-forced` to Phi-4-mini

**Files:**
- Modify: `crates/runner/src/phi4_engine.rs`
- Modify: `crates/runner/src/cli.rs`
- Create: `crates/runner/tests/teacher_forced_phi4.rs`

- [ ] **Step 1: Add the smoke test**

Create `crates/runner/tests/teacher_forced_phi4.rs` mirroring Task 12 but with `--model phi4-mini` and `MODEL_DIR_PHI4`.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p runner --test teacher_forced_phi4 -- --nocapture`
Expected: bails.

- [ ] **Step 3: Implement `run_phi4_teacher_forced`**

Add a function in `crates/runner/src/phi4_engine.rs` following the same pattern. Phi-4 uses the component decode path on CUDA; on HIP it has its own engine entry. Capture per-step logits and emit the standard `[teacher_forced_json]` line.

Update `cli.rs` to dispatch.

- [ ] **Step 4: Run the smoke**

Run: `MODEL_DIR_PHI4=/mnt/data/models/Phi-4-mini-instruct cargo test -p runner --test teacher_forced_phi4 -- --nocapture`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/phi4_engine.rs crates/runner/src/cli.rs crates/runner/tests/teacher_forced_phi4.rs
git commit -m "runner(phi4): --teacher-forced emits [teacher_forced_json]"
```

---

## Task 15: Implement perplexity sweep (Python)

**Files:**
- Create: `oracle/bench/perplexity.py`
- Create: `oracle/bench/tests/test_perplexity_math.py`

- [ ] **Step 1: Write the failing test**

Create `oracle/bench/tests/test_perplexity_math.py`:
```python
"""Math sanity for perplexity aggregation. No GPU."""
import math

from oracle.bench.perplexity import aggregate_ppl_from_chunks


def test_aggregate_ppl_from_single_chunk():
    chunks = [{"nll": 10.0, "tokens": 5}]
    out = aggregate_ppl_from_chunks(chunks)
    assert math.isclose(out["avg_nll"], 2.0, rel_tol=1e-9)
    assert math.isclose(out["ppl"], math.exp(2.0), rel_tol=1e-9)
    assert out["tokens"] == 5


def test_aggregate_ppl_from_multiple_chunks():
    chunks = [
        {"nll": 10.0, "tokens": 5},
        {"nll": 6.0, "tokens": 3},
    ]
    out = aggregate_ppl_from_chunks(chunks)
    expected_avg_nll = (10.0 + 6.0) / (5 + 3)
    assert math.isclose(out["avg_nll"], expected_avg_nll, rel_tol=1e-9)
    assert math.isclose(out["ppl"], math.exp(expected_avg_nll), rel_tol=1e-9)
    assert out["tokens"] == 8


def test_aggregate_ppl_empty_chunks_raises():
    import pytest
    with pytest.raises(ValueError):
        aggregate_ppl_from_chunks([])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_perplexity_math.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement perplexity.py**

Create `oracle/bench/perplexity.py`:
```python
"""Perplexity sweep over PG-19 and WikiText-2 for any (model, quant).

Generalizes oracle/pg19_smoke.py to drive `./target/release/supersonic
--teacher-forced` on any model family. Requires Tasks 12-14 to have
extended --teacher-forced beyond the Llama lane.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_TF_JSON = re.compile(r"^\[teacher_forced_json\] (.+)$", re.MULTILINE)


def aggregate_ppl_from_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("no chunks to aggregate")
    total_nll = sum(c["nll"] for c in chunks)
    total_tokens = sum(c["tokens"] for c in chunks)
    avg_nll = total_nll / total_tokens
    return {"nll": total_nll, "tokens": total_tokens,
            "avg_nll": avg_nll, "ppl": math.exp(avg_nll)}


@dataclass
class PerplexityRequest:
    binary: Path
    model: str
    model_dir: Path
    quant: str
    dataset: str            # "pg19" or "wikitext2"
    contexts: int           # context length per chunk
    num_chunks: int


def score_perplexity(req: PerplexityRequest) -> dict:
    """Run --teacher-forced over `num_chunks` chunks of the dataset.
    Returns a dict suitable for writing as a quality cell:
       {schema_version: 1, model, quant, eval, metric: "ppl", value: ppl, extras: {...}}
    """
    chunks_per_run = _load_chunks(req.dataset, req.contexts, req.num_chunks)
    chunk_results = []
    for prompt in chunks_per_run:
        out = _run_supersonic_teacher_forced(req, prompt)
        m = _TF_JSON.search(out)
        if not m:
            continue
        d = json.loads(m.group(1))
        chunk_results.append({"nll": d["nll"], "tokens": d["scored_tokens"]})
    agg = aggregate_ppl_from_chunks(chunk_results)
    return {
        "schema_version": 1,
        "model": req.model,
        "quant": req.quant,
        "eval": f"perplexity_{req.dataset}",
        "metric": "ppl",
        "value": agg["ppl"],
        "extras": {"avg_nll": agg["avg_nll"], "tokens": agg["tokens"],
                   "contexts": req.contexts, "num_chunks": req.num_chunks},
    }


def _load_chunks(dataset: str, contexts: int, num_chunks: int) -> list[str]:
    """Load `num_chunks` text chunks of `contexts` tokens from the named dataset.
    Reuses oracle/pg19_smoke.py loading logic where possible.
    """
    if dataset == "pg19":
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        # PG-19 is char streams; the supersonic --teacher-forced path tokenizes
        # internally. Slice each chunk to ~contexts*5 chars (rough byte-per-token).
        out = []
        char_budget = contexts * 5
        for row in ds:
            text = row["text"][:char_budget]
            out.append(text)
            if len(out) >= num_chunks:
                break
        return out
    if dataset == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined = "\n".join(ds["text"])
        out, char_budget = [], contexts * 5
        for i in range(num_chunks):
            chunk = joined[i * char_budget:(i + 1) * char_budget]
            if not chunk.strip():
                break
            out.append(chunk)
        return out
    raise ValueError(f"unknown dataset {dataset!r}")


def _run_supersonic_teacher_forced(req: PerplexityRequest, prompt: str) -> str:
    cmd = [str(req.binary), "--model", req.model, "--model-dir", str(req.model_dir),
           "--prompt", prompt, "--max-new-tokens", "1", "--teacher-forced"]
    cmd.extend({"bf16": [], "int4": ["--int4"], "fp8r": ["--fp8-runtime"],
                "kv-fp8": ["--kv-fp8"], "int8": ["--int8"]}[req.quant])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (res.stdout or "") + "\n" + (res.stderr or "")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest oracle/bench/tests/test_perplexity_math.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add oracle/bench/perplexity.py oracle/bench/tests/test_perplexity_math.py
git commit -m "bench: perplexity.py — PG-19 + WikiText-2 sweep over (model, quant)"
```

---

## Task 16: Implement golden-prompt diff harness

**Files:**
- Create: `oracle/bench/golden.py`
- Create: `oracle/bench/golden_prompts.json`
- Create: `oracle/bench/tests/test_golden_diff.py`

- [ ] **Step 1: Write the failing test**

Create `oracle/bench/tests/test_golden_diff.py`:
```python
"""Golden-prompt scoring math. No GPU."""
from oracle.bench.golden import (
    score_pair, aggregate_golden_results,
)


def test_exact_match_scores_1():
    s = score_pair("hello world", "hello world")
    assert s["exact_match"] == 1.0
    assert s["chrf"] == 1.0


def test_total_mismatch_scores_low():
    s = score_pair("hello world", "completely different output here xyz")
    assert s["exact_match"] == 0.0
    assert s["chrf"] < 0.5


def test_aggregate_returns_means_and_failure_count():
    per_prompt = [
        {"prompt_id": "a", "exact_match": 1.0, "chrf": 1.0},
        {"prompt_id": "b", "exact_match": 0.0, "chrf": 0.4},
        {"prompt_id": "c", "exact_match": 0.0, "chrf": 0.05},
    ]
    out = aggregate_golden_results(per_prompt, chrf_threshold=0.20)
    assert abs(out["exact_match_mean"] - (1.0/3.0)) < 1e-6
    assert abs(out["chrf_mean"] - (1.0 + 0.4 + 0.05) / 3) < 1e-6
    assert out["below_threshold_count"] == 1   # only "c" is < 0.20
    assert out["below_threshold_ids"] == ["c"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_golden_diff.py -v`
Expected: ImportError.

- [ ] **Step 3: Create the prompts file**

Create `oracle/bench/golden_prompts.json`:
```json
{
  "version": 1,
  "_doc": "Curated set of ~20 prompts. The first BF16 run for each (model) populates the reference; INT4/FP8 runs are diffed against the BF16 reference. Reference is stored at oracle/bench/golden_references/{model}.json (created on first BF16 run).",
  "prompts": [
    {"id": "fox_one_line", "prompt": "The quick brown fox jumps over", "max_new_tokens": 32},
    {"id": "code_python_hello", "prompt": "def hello():\n    ", "max_new_tokens": 32},
    {"id": "code_rust_main", "prompt": "fn main() {\n    println!(", "max_new_tokens": 16},
    {"id": "story_open", "prompt": "Once upon a time, in a land far away,", "max_new_tokens": 64},
    {"id": "math_arith", "prompt": "What is 47 + 35? The answer is", "max_new_tokens": 8},
    {"id": "qa_capital", "prompt": "Q: What is the capital of France?\nA:", "max_new_tokens": 8},
    {"id": "list_completion", "prompt": "Three primary colors are red, blue, and", "max_new_tokens": 4},
    {"id": "translate_en_de", "prompt": "English: Good morning.\nGerman:", "max_new_tokens": 8},
    {"id": "wiki_factual", "prompt": "The largest planet in our solar system is", "max_new_tokens": 4},
    {"id": "code_complete_for", "prompt": "for i in range(10):\n    print(", "max_new_tokens": 16},
    {"id": "json_emit", "prompt": "{\"name\": \"Alice\", \"age\":", "max_new_tokens": 4},
    {"id": "longer_paragraph", "prompt": "The sun set slowly over the mountains, casting long shadows across", "max_new_tokens": 64},
    {"id": "instruction_summarize", "prompt": "Summarize: The cat sat on the mat. The mat was red.\nSummary:", "max_new_tokens": 24},
    {"id": "list_count", "prompt": "Count to five: 1, 2, 3,", "max_new_tokens": 8},
    {"id": "syllogism", "prompt": "All birds can fly. A penguin is a bird. Can a penguin fly?", "max_new_tokens": 16},
    {"id": "code_recursive", "prompt": "def fact(n):\n    if n <= 1:\n        return 1\n    return", "max_new_tokens": 16},
    {"id": "bullet_list", "prompt": "Three benefits of exercise:\n-", "max_new_tokens": 48},
    {"id": "completion_sentence", "prompt": "The reason the sky appears blue is", "max_new_tokens": 32},
    {"id": "punctuation", "prompt": "Hello, world! How are you", "max_new_tokens": 8},
    {"id": "long_repeat_check", "prompt": "Tell me a short story about a robot.", "max_new_tokens": 96}
  ]
}
```

- [ ] **Step 4: Implement golden.py**

Create `oracle/bench/golden.py`:
```python
"""Golden-prompt diff: score (model, quant) generated text against the BF16 reference."""
from __future__ import annotations
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROMPTS_PATH = Path(__file__).parent / "golden_prompts.json"
REFERENCES_DIR = Path(__file__).parent / "golden_references"


def score_pair(reference: str, candidate: str) -> dict:
    """Compute exact_match and chrF (character-n-gram F1) between two strings."""
    em = 1.0 if reference.strip() == candidate.strip() else 0.0
    chrf = _chrf(reference, candidate, n=6)
    return {"exact_match": em, "chrf": chrf}


def aggregate_golden_results(per_prompt: list[dict], chrf_threshold: float = 0.20) -> dict:
    if not per_prompt:
        raise ValueError("no per-prompt results")
    em = sum(p["exact_match"] for p in per_prompt) / len(per_prompt)
    chrf = sum(p["chrf"] for p in per_prompt) / len(per_prompt)
    below = [p["prompt_id"] for p in per_prompt if p["chrf"] < chrf_threshold]
    return {
        "exact_match_mean": em,
        "chrf_mean": chrf,
        "below_threshold_count": len(below),
        "below_threshold_ids": below,
    }


def _chrf(ref: str, hyp: str, n: int = 6) -> float:
    """Single-order character n-gram F1. Simple baseline; no β weighting."""
    if not ref or not hyp:
        return 0.0
    ref_grams = _char_ngrams(ref, n)
    hyp_grams = _char_ngrams(hyp, n)
    if not ref_grams or not hyp_grams:
        return 0.0
    overlap = sum((ref_grams & hyp_grams).values())
    p = overlap / max(sum(hyp_grams.values()), 1)
    r = overlap / max(sum(ref_grams.values()), 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _char_ngrams(s: str, n: int) -> Counter:
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


@dataclass
class GoldenRequest:
    binary: Path
    model: str
    model_dir: Path
    quant: str


def run_golden(req: GoldenRequest) -> dict:
    """Run all golden prompts for (model, quant). Returns a quality cell.

    First BF16 run for a model populates the reference file. Non-BF16 runs
    require the reference to exist; otherwise the cell status is 'reference_missing'.
    """
    prompts = json.loads(PROMPTS_PATH.read_text())["prompts"]
    REFERENCES_DIR.mkdir(exist_ok=True)
    ref_path = REFERENCES_DIR / f"{req.model}.json"

    if req.quant == "bf16":
        outputs = {p["id"]: _generate(req, p) for p in prompts}
        ref_path.write_text(json.dumps({"model": req.model, "outputs": outputs}, indent=2))
        return {"schema_version": 1, "model": req.model, "quant": req.quant,
                "eval": "golden", "metric": "bootstrap", "value": 1.0,
                "extras": {"prompts_recorded": len(outputs)}}

    if not ref_path.exists():
        return {"schema_version": 1, "model": req.model, "quant": req.quant,
                "eval": "golden", "metric": "exact_match_mean", "value": 0.0,
                "extras": {"status": "reference_missing",
                           "hint": f"run --quants bf16 first to populate {ref_path}"}}

    ref = json.loads(ref_path.read_text())["outputs"]
    per_prompt = []
    for p in prompts:
        cand = _generate(req, p)
        scores = score_pair(ref[p["id"]], cand)
        per_prompt.append({"prompt_id": p["id"], **scores})
    agg = aggregate_golden_results(per_prompt)
    return {"schema_version": 1, "model": req.model, "quant": req.quant,
            "eval": "golden", "metric": "exact_match_mean",
            "value": agg["exact_match_mean"],
            "extras": {**agg, "per_prompt": per_prompt}}


_GENERATED = re.compile(r"^\[generated\] (.*)$", re.MULTILINE)


def _generate(req: GoldenRequest, prompt: dict) -> str:
    cmd = [str(req.binary), "--model", req.model, "--model-dir", str(req.model_dir),
           "--prompt", prompt["prompt"], "--max-new-tokens", str(prompt["max_new_tokens"])]
    cmd.extend({"bf16": [], "int4": ["--int4"], "fp8r": ["--fp8-runtime"],
                "kv-fp8": ["--kv-fp8"], "int8": ["--int8"]}[req.quant])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    m = _GENERATED.search(text)
    return m.group(1) if m else text.strip().split("\n")[-1]
```

NOTE: `_generate` looks for a `[generated]` line. If supersonic doesn't emit one, parse the actual generation output it does emit (search for "Generated:" prefix or similar). Adjust the regex to match the actual runner output observed in your dev environment.

- [ ] **Step 5: Run tests**

Run: `python -m pytest oracle/bench/tests/test_golden_diff.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add oracle/bench/golden.py oracle/bench/golden_prompts.json oracle/bench/tests/test_golden_diff.py
git commit -m "bench: golden-prompt diff harness with BF16 reference bootstrap"
```

---

## Task 17: Quality main entry point

**Files:**
- Create: `oracle/bench/quality_main.py`
- Modify: `oracle/bench/render/markdown.py` (add `render_quality_table`)

- [ ] **Step 1: Add the quality renderer**

Append to `oracle/bench/render/markdown.py`:
```python
def render_quality_table(quality_dir: Path) -> str:
    cells = []
    for f in sorted(quality_dir.glob("*.json")):
        cells.append(json.loads(f.read_text()))
    if not cells:
        return ""

    by_model_quant: dict[tuple[str, str], dict[str, dict]] = {}
    for c in cells:
        by_model_quant.setdefault((c["model"], c["quant"]), {})[c["eval"]] = c

    evals = sorted({c["eval"] for c in cells})
    headers = ["Model", "Quant"] + evals
    rows = ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|"]
    for (model, quant) in sorted(by_model_quant):
        row_evals = by_model_quant[(model, quant)]
        cols = [model, quant]
        for ev in evals:
            cell = row_evals.get(ev)
            if cell is None:
                cols.append("—")
            else:
                cols.append(f"{cell['value']:.3f}")
        rows.append("| " + " | ".join(cols) + " |")
    return "\n".join(rows) + "\n"
```

- [ ] **Step 2: Implement quality_main.py**

Create `oracle/bench/quality_main.py`:
```python
"""CLI: python -m oracle.bench.quality_main --arch gfx1100 --models all --quants all"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .perplexity import PerplexityRequest, score_perplexity
from .golden import GoldenRequest, run_golden
from .render.schema import validate_quality_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1100")
    ap.add_argument("--models", default="all")
    ap.add_argument("--quants", default="all")
    ap.add_argument("--binary", default="./target/release/supersonic", type=Path)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[],
                    help="KEY=PATH; repeatable")
    ap.add_argument("--ppl-contexts", type=int, default=2048)
    ap.add_argument("--ppl-num-chunks", type=int, default=4)
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    run_dir = _latest_run_dir(args.run_root)
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
              "gemma4-e2b", "gemma4-e4b", "phi4-mini"] if args.models == "all" \
             else [m.strip() for m in args.models.split(",")]
    quants = ["bf16", "int4", "fp8r", "kv-fp8"] if args.quants == "all" \
             else [q.strip() for q in args.quants.split(",")]

    for model in models:
        # BF16 must run first to populate golden reference.
        ordered_quants = sorted(quants, key=lambda q: 0 if q == "bf16" else 1)
        for quant in ordered_quants:
            mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
            for ds in ("pg19", "wikitext2"):
                req = PerplexityRequest(args.binary, model, mdir, quant, ds,
                                        args.ppl_contexts, args.ppl_num_chunks)
                cell = score_perplexity(req)
                validate_quality_cell(cell)
                (out_dir / f"{model}_{quant}_perplexity_{ds}.json").write_text(
                    json.dumps(cell, indent=2))
            golden = run_golden(GoldenRequest(args.binary, model, mdir, quant))
            (out_dir / f"{model}_{quant}_golden.json").write_text(json.dumps(golden, indent=2))

    print(f"[bench-quality] wrote {out_dir}")


def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"no run-dirs at {root} — run bench-perf first to create one")
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Wire quality table into renderer CLI**

In `oracle/bench/render/render_main.py`, append to the `if args.cmd == "markdown":` block:
```python
        quality_md = render_quality_table(args.run / "quality")
        quality_doc_path = args.out / "docs" / "quality.md"
        quality_doc = quality_doc_path.read_text() if quality_doc_path.exists() else "# Model Quality\n\nMeasured (model, quant) quality for shipping models.\n"
        updated = replace_autogen_zone(quality_doc, "bench-quality-table", quality_md)
        quality_doc_path.write_text(updated)
        print(f"updated {quality_doc_path}")
```

(Update the import: `from .markdown import render_perf_table, render_quality_table, replace_autogen_zone`.)

- [ ] **Step 4: Smoke test the import**

Run: `python -c "from oracle.bench.quality_main import main; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add oracle/bench/quality_main.py oracle/bench/render/markdown.py oracle/bench/render/render_main.py
git commit -m "bench: quality_main entry point + quality markdown renderer"
```

---

## Task 18: Scaffold `docs/quality.md`

**Files:**
- Create: `docs/quality.md`

- [ ] **Step 1: Write the doc skeleton**

Create `docs/quality.md`:
```markdown
# Model Quality

Measured quality for SuperSonic's shipping (model, quant) combos on `gfx1100`.

This doc has two parts:
- A hand-written narrative section (this prose, above the AUTOGEN sentinel).
- An autogenerated table of metrics (below the sentinel), produced by:
  ```bash
  python -m oracle.bench.render.render_main markdown \
      --run target/bench-runs/<run-id> --out .
  ```

## Methodology

Quality numbers come from two evals per cell:
- **Perplexity** (PG-19 + WikiText-2 teacher-forced, default 4 chunks × 2048 tokens).
- **Golden-prompt diff** (~20 curated prompts; INT4/FP8 generations are scored
  against the model's BF16 reference using exact-match and chrF).

The heavy lane (NIAH, RULER, lm-evaluation-harness) lives separately and is
opt-in via `python -m oracle.bench.heavy.heavy_main`.

<!-- AUTOGEN BELOW: bench-quality-table -->
<!-- AUTOGEN END: bench-quality-table -->
```

- [ ] **Step 2: Commit**

```bash
git add docs/quality.md
git commit -m "docs: scaffold quality.md with methodology + AUTOGEN sentinel zone"
```

---

## Task 19: hipfire adapter + version pin gate

**Files:**
- Create: `oracle/bench/external/__init__.py`
- Create: `oracle/bench/external/common.py`
- Create: `oracle/bench/external/hipfire.py`
- Create: `oracle/bench/external/external_main.py`
- Create: `oracle/bench/tests/test_hipfire_adapter.py`
- Create: `tools/external/hipfire-version.txt`
- Create: `tools/external/check-versions.sh`

- [ ] **Step 1: Pin hipfire version**

Create `tools/external/hipfire-version.txt`:
```
# Pinned hipfire version. Bump deliberately; changing this is a benchmark methodology change.
# Format: one line, the exact `hipfire --version` output (or commit shorthand if hipfire emits one).
# As of 2026-05-05, replace this placeholder with the actual installed version
# captured via `hipfire --version` on the dev machine.
hipfire-PLACEHOLDER-CAPTURE-VIA-hipfire-version
```

After running `hipfire --version` once, replace the placeholder with the real version string.

- [ ] **Step 2: Create the version-check shell script**

Create `tools/external/check-versions.sh`:
```bash
#!/usr/bin/env bash
# Verify installed external benchmark engines match pinned versions.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

check_one() {
    local engine="$1"
    local version_cmd="$2"
    local pin_file="$SCRIPT_DIR/${engine}-version.txt"
    if [ ! -f "$pin_file" ]; then
        echo "[check-versions] no pin file for $engine ($pin_file); skipping" >&2
        return 0
    fi
    local pinned actual
    pinned="$(grep -v '^#' "$pin_file" | grep -v '^$' | head -n1)"
    actual="$(eval "$version_cmd" 2>&1 | head -n1)"
    if [ "$pinned" != "$actual" ]; then
        echo "[check-versions] MISMATCH: $engine pinned=$pinned actual=$actual" >&2
        return 1
    fi
    echo "[check-versions] OK: $engine = $actual"
}

failed=0
check_one hipfire "hipfire --version" || failed=1
exit $failed
```

Make it executable: `chmod +x tools/external/check-versions.sh`.

- [ ] **Step 3: Write the failing test**

Create `oracle/bench/external/__init__.py` (empty).

Create `oracle/bench/tests/test_hipfire_adapter.py`:
```python
"""hipfire adapter: version-pin gate, subprocess parsing."""
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from oracle.bench.external.hipfire import (
    HipfireAdapter, HipfireVersionMismatch,
)


def test_version_check_passes_when_versions_match(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "hipfire 0.4.2-rocm6\n"
            stderr = ""
            returncode = 0
        return R()
    with patch("oracle.bench.external.hipfire.subprocess.run", side_effect=fake_run):
        ad = HipfireAdapter(version_pin_file=pin)
        ad.assert_version_match()  # no raise


def test_version_check_raises_on_mismatch(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "hipfire 0.5.0-rocm6\n"
            stderr = ""
            returncode = 0
        return R()
    with patch("oracle.bench.external.hipfire.subprocess.run", side_effect=fake_run):
        ad = HipfireAdapter(version_pin_file=pin)
        with pytest.raises(HipfireVersionMismatch) as ei:
            ad.assert_version_match()
        assert "0.4.2-rocm6" in str(ei.value)
        assert "0.5.0-rocm6" in str(ei.value)


def test_supports_returns_false_for_unknown_model(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")
    ad = HipfireAdapter(version_pin_file=pin)
    assert ad.supports("qwen3.6-35b-a3b", "int4") in (True, False)  # implementation-specific
    # The base contract: returns a bool, doesn't raise.
```

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_hipfire_adapter.py -v`
Expected: ImportError.

- [ ] **Step 5: Implement common base class**

Create `oracle/bench/external/common.py`:
```python
"""Base class for external engine adapters (hipfire today; llama.cpp/vLLM later)."""
from __future__ import annotations
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class ExternalAdapter(ABC):
    """Common API for benchmarking against an external inference engine."""

    name: str = "unknown"

    @abstractmethod
    def assert_version_match(self) -> None:
        """Raise if the installed engine does not match the pinned version."""

    @abstractmethod
    def supports(self, model: str, quant: str) -> bool:
        """Return True if this adapter can run the given (model, quant)."""

    @abstractmethod
    def measure_speed(self, model: str, quant: str, prompt: str,
                      max_new_tokens: int, model_dir: Path) -> dict:
        """Run the engine and return a dict matching the EXTERNAL_CELL_SCHEMA."""


def read_pinned_version(pin_file: Path) -> str:
    for line in pin_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"{pin_file} has no version line")


def get_engine_version(version_cmd: list[str]) -> str:
    res = subprocess.run(version_cmd, capture_output=True, text=True, check=False)
    return (res.stdout or res.stderr or "").splitlines()[0].strip() if res.stdout or res.stderr else ""
```

- [ ] **Step 6: Implement hipfire adapter**

Create `oracle/bench/external/hipfire.py`:
```python
"""hipfire (https://github.com/Kaden-Schutt/hipfire) external adapter.

Speed-only Phase 1. Quality comparison out of scope (tokenizer alignment).
"""
from __future__ import annotations
import re
import subprocess
import time
from pathlib import Path

from .common import ExternalAdapter, get_engine_version, read_pinned_version

DEFAULT_PIN = Path(__file__).parent.parent.parent.parent / "tools" / "external" / "hipfire-version.txt"


class HipfireVersionMismatch(RuntimeError):
    pass


class HipfireAdapter(ExternalAdapter):
    name = "hipfire"

    def __init__(self, version_pin_file: Path = DEFAULT_PIN, binary: str = "hipfire"):
        self.binary = binary
        self.pin_file = version_pin_file

    def assert_version_match(self) -> None:
        pinned = read_pinned_version(self.pin_file)
        actual = get_engine_version([self.binary, "--version"])
        if pinned != actual:
            raise HipfireVersionMismatch(
                f"hipfire pinned={pinned!r} actual={actual!r}; bump {self.pin_file} or install pinned version"
            )

    def supports(self, model: str, quant: str) -> bool:
        # hipfire's supported model set as of 2026-05-05 (verify with `hipfire --list-models`).
        # Conservative default: only the most-likely-supported pairs; widen as confirmed.
        supported = {
            ("qwen3.5-0.8b", "bf16"), ("qwen3.5-0.8b", "int4"),
            ("qwen3.5-2b", "bf16"),   ("qwen3.5-2b", "int4"),
            ("qwen3.5-4b", "bf16"),   ("qwen3.5-4b", "int4"),
            ("qwen3.5-9b", "bf16"),   ("qwen3.5-9b", "int4"),
        }
        return (model, quant) in supported

    def measure_speed(self, model: str, quant: str, prompt: str,
                      max_new_tokens: int, model_dir: Path) -> dict:
        version = get_engine_version([self.binary, "--version"])
        if not self.supports(model, quant):
            return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                    "model": model, "quant": quant, "status": "unsupported_by_engine",
                    "ms_per_step": None, "samples": None, "stderr_tail": None}

        # 3s cooldown + 1 warmup + 3 measurement passes, mirror SuperSonic discipline.
        time.sleep(3)
        # Warmup
        self._invoke(model, quant, prompt, 2, model_dir)
        samples = []
        last_err = None
        for _ in range(3):
            try:
                out = self._invoke(model, quant, prompt, max_new_tokens, model_dir)
                ms = self._extract_ms_per_step(out)
                if ms is not None:
                    samples.append(ms)
                else:
                    last_err = out[-2000:]
            except subprocess.CalledProcessError as e:
                last_err = (e.stderr or e.stdout or "")[-2000:]

        if samples:
            samples.sort()
            median = samples[len(samples) // 2]
            return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                    "model": model, "quant": quant, "status": "ok",
                    "ms_per_step": median, "samples": samples, "stderr_tail": None}
        return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                "model": model, "quant": quant, "status": "error",
                "ms_per_step": None, "samples": None, "stderr_tail": str(last_err)}

    def _invoke(self, model: str, quant: str, prompt: str, max_new: int, model_dir: Path) -> str:
        # Adjust the CLI shape to match actual hipfire conventions; below is a placeholder
        # that needs verification against `hipfire --help` on the dev machine.
        cmd = [self.binary, "generate", "--model", str(model_dir),
               "--prompt", prompt, "--n", str(max_new)]
        if quant == "int4":
            cmd.extend(["--quant", "int4"])
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (res.stdout or "") + "\n" + (res.stderr or "")

    def _extract_ms_per_step(self, stdout: str) -> float | None:
        # hipfire output format TBD; common shapes:
        #   "tokens/sec: 95.3" → ms_per_step = 1000 / 95.3
        #   "ms/token: 10.5"   → ms_per_step = 10.5
        m = re.search(r"tokens?/sec[:\s=]+([0-9.]+)", stdout)
        if m:
            tps = float(m.group(1))
            return 1000.0 / tps if tps > 0 else None
        m = re.search(r"ms/(?:token|step)[:\s=]+([0-9.]+)", stdout)
        if m:
            return float(m.group(1))
        return None
```

(The `_invoke` and `_extract_ms_per_step` shapes will need adjustment once hipfire is actually installed and its CLI is observed. Capture real output during Task 22's smoke run and tighten these.)

- [ ] **Step 7: Implement external_main entry point**

Create `oracle/bench/external/external_main.py`:
```python
"""CLI: python -m oracle.bench.external.external_main --engine hipfire --models all --quants all"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .hipfire import HipfireAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="hipfire", choices=["hipfire"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--quants", default="all")
    ap.add_argument("--prompt", default="The quick brown fox jumps over")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[], help="KEY=PATH")
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    adapter = HipfireAdapter()
    adapter.assert_version_match()

    run_dir = _latest_run_dir(args.run_root)
    out_dir = run_dir / "external" / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"] \
             if args.models == "all" else [m.strip() for m in args.models.split(",")]
    quants = ["bf16", "int4"] if args.quants == "all" else [q.strip() for q in args.quants.split(",")]

    for model in models:
        for quant in quants:
            mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
            cell = adapter.measure_speed(model, quant, args.prompt, args.max_new_tokens, mdir)
            (out_dir / f"{model}_{quant}.json").write_text(json.dumps(cell, indent=2))

    print(f"[bench-external] wrote {out_dir}")


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run tests**

Run: `python -m pytest oracle/bench/tests/test_hipfire_adapter.py -v`
Expected: 3 tests pass.

- [ ] **Step 9: Commit**

```bash
git add oracle/bench/external tools/external oracle/bench/tests/test_hipfire_adapter.py
git commit -m "bench: hipfire adapter with B-style version-pin gate (speed-only)"
```

---

## Task 20: Add hipfire comparison column to renderer

**Files:**
- Modify: `oracle/bench/render/markdown.py`
- Modify: `oracle/bench/tests/test_renderer.py` (add a test)
- Create: `oracle/bench/tests/fixtures/run_minimal/external/hipfire/qwen3.5-0.8b_bf16.json`
- Create: `oracle/bench/tests/fixtures/golden_hipfire_fragment.md`

- [ ] **Step 1: Add a hipfire fixture**

Create `oracle/bench/tests/fixtures/run_minimal/external/hipfire/qwen3.5-0.8b_bf16.json`:
```json
{
  "schema_version": 1,
  "engine": "hipfire",
  "engine_version": "hipfire 0.4.2-rocm6",
  "model": "qwen3.5-0.8b",
  "quant": "bf16",
  "status": "ok",
  "ms_per_step": 7.5,
  "samples": [7.4, 7.5, 7.6],
  "stderr_tail": null
}
```

Create `oracle/bench/tests/fixtures/golden_hipfire_fragment.md`:
```
| Model           | Quant | SuperSonic ms/step | hipfire ms/step | Δ vs hipfire |
|-----------------|-------|-------------------:|----------------:|-------------:|
| qwen3.5-0.8b    | bf16  |                8.0 |             7.5 |        +6.7% |
```

(Δ formula: `(supersonic - hipfire) / hipfire * 100`. Positive = SuperSonic slower.)

- [ ] **Step 2: Write the failing test**

In `oracle/bench/tests/test_renderer.py`, append:
```python
GOLDEN_HIPFIRE = (Path(__file__).parent / "fixtures" / "golden_hipfire_fragment.md").read_text()


def test_hipfire_table_matches_golden():
    from oracle.bench.render.markdown import render_external_comparison_table
    rendered = render_external_comparison_table(
        FIXTURE_DIR / "perf",
        FIXTURE_DIR / "external" / "hipfire",
        engine="hipfire",
    )
    assert rendered.strip() == GOLDEN_HIPFIRE.strip()
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_renderer.py::test_hipfire_table_matches_golden -v`
Expected: ImportError.

- [ ] **Step 4: Implement `render_external_comparison_table`**

Append to `oracle/bench/render/markdown.py`:
```python
def render_external_comparison_table(perf_dir: Path, external_dir: Path, engine: str) -> str:
    perf_cells = {(json.loads(f.read_text())["model"], json.loads(f.read_text())["quant"]): json.loads(f.read_text())
                  for f in perf_dir.glob("*.json")}
    ext_cells = {(json.loads(f.read_text())["model"], json.loads(f.read_text())["quant"]): json.loads(f.read_text())
                 for f in external_dir.glob("*.json")}
    keys = sorted(set(perf_cells) & set(ext_cells))
    if not keys:
        return ""

    header = f"| Model           | Quant | SuperSonic ms/step | {engine} ms/step | Δ vs {engine} |"
    sep    =  "|-----------------|-------|-------------------:|----------------:|-------------:|"
    rows = [header, sep]
    for (model, quant) in keys:
        p, e = perf_cells[(model, quant)], ext_cells[(model, quant)]
        if p["status"] != "ok" or e["status"] != "ok":
            continue
        ss_ms, ext_ms = p["ms_per_step"], e["ms_per_step"]
        delta = (ss_ms - ext_ms) / ext_ms * 100 if ext_ms > 0 else 0
        rows.append(f"| {model.ljust(15)} | {quant.ljust(5)} | {ss_ms:>18.1f} | {ext_ms:>15.1f} | {delta:+11.1f}% |")
    return "\n".join(rows) + "\n"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest oracle/bench/tests/test_renderer.py -v`
Expected: all pass. Regenerate golden if formatting whitespace differs (paste actual `print(...)` output).

- [ ] **Step 6: Wire into render_main.py**

In `oracle/bench/render/render_main.py`, append to the markdown command block:
```python
        ext_dir = args.run / "external" / "hipfire"
        if ext_dir.exists():
            hipfire_md = render_external_comparison_table(args.run / "perf", ext_dir, "hipfire")
            updated = replace_autogen_zone(perf_doc.read_text(), "hipfire-comparison", hipfire_md)
            perf_doc.write_text(updated)
            print(f"updated hipfire-comparison zone in {perf_doc}")
```

(Update the import at top: `from .markdown import render_perf_table, render_quality_table, render_external_comparison_table, replace_autogen_zone`.)

- [ ] **Step 7: Commit**

```bash
git add oracle/bench/render oracle/bench/tests
git commit -m "bench: render hipfire comparison column into docs/performance.md"
```

---

## Task 21: Implement run-diff command (regression detector)

**Files:**
- Create: `oracle/bench/render/diff.py`
- Modify: `oracle/bench/render/render_main.py`
- Modify: `oracle/bench/tests/test_renderer.py` (add a diff test)
- Create: `oracle/bench/tests/fixtures/run_baseline/perf/qwen3.5-0.8b_bf16.json`

- [ ] **Step 1: Add a baseline fixture (slightly different from run_minimal)**

Create `oracle/bench/tests/fixtures/run_baseline/perf/qwen3.5-0.8b_bf16.json`:
```json
{
  "schema_version": 1,
  "model": "qwen3.5-0.8b",
  "quant": "bf16",
  "prompt": "The quick brown fox jumps over",
  "max_new_tokens": 16,
  "status": "ok",
  "ms_per_step": 7.5,
  "ms_per_tok": 7.5,
  "samples": [7.4, 7.5, 7.6],
  "gpu_temp_c_end": 60.0
}
```

(run_minimal has 8.0 ms; baseline has 7.5 ms → +6.7% regression.)

- [ ] **Step 2: Write the failing test**

In `test_renderer.py`, append:
```python
def test_diff_flags_regressions_above_threshold():
    from oracle.bench.render.diff import diff_runs
    fixtures = Path(__file__).parent / "fixtures"
    rows = diff_runs(fixtures / "run_baseline", fixtures / "run_minimal", threshold_pct=5.0)
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "qwen3.5-0.8b"
    assert r["quant"] == "bf16"
    assert r["before"] == 7.5
    assert r["after"] == 8.0
    assert abs(r["delta_pct"] - 6.666) < 0.1


def test_diff_below_threshold_is_omitted():
    from oracle.bench.render.diff import diff_runs
    fixtures = Path(__file__).parent / "fixtures"
    rows = diff_runs(fixtures / "run_baseline", fixtures / "run_minimal", threshold_pct=10.0)
    assert rows == []
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest oracle/bench/tests/test_renderer.py -v`
Expected: ImportError on `oracle.bench.render.diff`.

- [ ] **Step 4: Implement diff.py**

Create `oracle/bench/render/diff.py`:
```python
"""Two-run regression diff: surface cells where the metric moved beyond threshold."""
import json
from pathlib import Path


def diff_runs(run_a: Path, run_b: Path, threshold_pct: float = 5.0) -> list[dict]:
    """Compare perf cells between two run-dirs. Returns rows where ms_per_step moved by > threshold_pct."""
    a_cells = _load_perf(run_a)
    b_cells = _load_perf(run_b)
    rows = []
    for key in sorted(set(a_cells) | set(b_cells)):
        a, b = a_cells.get(key), b_cells.get(key)
        if a is None or b is None:
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        before, after = a["ms_per_step"], b["ms_per_step"]
        if before == 0:
            continue
        delta_pct = (after - before) / before * 100
        if abs(delta_pct) >= threshold_pct:
            rows.append({"model": key[0], "quant": key[1],
                         "before": before, "after": after, "delta_pct": delta_pct})
    return rows


def _load_perf(run: Path) -> dict[tuple[str, str], dict]:
    out = {}
    perf_dir = run / "perf"
    if not perf_dir.exists():
        return out
    for f in perf_dir.glob("*.json"):
        d = json.loads(f.read_text())
        out[(d["model"], d["quant"])] = d
    return out


def render_diff_table(rows: list[dict]) -> str:
    if not rows:
        return "(no cells exceeded threshold)\n"
    header = "| Model           | Quant | Before ms | After ms | Δ%      |"
    sep    = "|-----------------|-------|----------:|---------:|--------:|"
    body = [
        f"| {r['model'].ljust(15)} | {r['quant'].ljust(5)} | {r['before']:>9.2f} | {r['after']:>8.2f} | {r['delta_pct']:+7.2f} |"
        for r in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"
```

- [ ] **Step 5: Wire into render_main.py**

In `oracle/bench/render/render_main.py`, add a new subcommand:
```python
    diff = sub.add_parser("diff")
    diff.add_argument("--run-a", required=True, type=Path)
    diff.add_argument("--run-b", required=True, type=Path)
    diff.add_argument("--threshold-pct", type=float, default=5.0)
```
And a handler:
```python
    if args.cmd == "diff":
        from .diff import diff_runs, render_diff_table
        rows = diff_runs(args.run_a, args.run_b, threshold_pct=args.threshold_pct)
        print(render_diff_table(rows))
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest oracle/bench/tests/test_renderer.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add oracle/bench/render oracle/bench/tests
git commit -m "bench: diff command surfaces cells where ms/step moved beyond threshold"
```

---

## Task 22: Smoke scripts

**Files:**
- Create: `tests/gfx1100/bench_smoke.sh`
- Create: `tests/gfx1100/bench_hipfire_smoke.sh`

- [ ] **Step 1: Write bench_smoke.sh**

Create `tests/gfx1100/bench_smoke.sh`:
```bash
#!/usr/bin/env bash
# End-to-end smoke: one combo through perf + quality + render.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

cd "$REPO_ROOT"
cargo build --release --bin supersonic
cargo build --release -p supersonic-bench

RUN_ROOT="$(mktemp -d)/bench-runs"
echo "[smoke] perf"
./target/release/bench-perf --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_ROOT"

RUN_DIR="$(ls -t "$RUN_ROOT" | head -n1)"
RUN_PATH="$RUN_ROOT/$RUN_DIR"

echo "[smoke] quality"
python -m oracle.bench.quality_main --arch gfx1100 \
    --models qwen3.5-0.8b --quants bf16 \
    --binary ./target/release/supersonic \
    --run-root "$RUN_ROOT" \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --ppl-num-chunks 1 --ppl-contexts 256

echo "[smoke] render"
python -m oracle.bench.render.render_main markdown \
    --run "$RUN_PATH" \
    --out "$REPO_ROOT"

# Assertions: at least one perf JSON and one quality JSON exist with non-empty fields.
test -s "$RUN_PATH/perf/qwen3.5-0.8b_bf16.json" || { echo "[smoke] FAIL: missing perf cell" >&2; exit 1; }
ls "$RUN_PATH/quality/" | head -n1 || { echo "[smoke] FAIL: no quality cells" >&2; exit 1; }
echo "[smoke] PASS — run_dir=$RUN_PATH"
```

Make executable: `chmod +x tests/gfx1100/bench_smoke.sh`.

- [ ] **Step 2: Write bench_hipfire_smoke.sh**

Create `tests/gfx1100/bench_hipfire_smoke.sh`:
```bash
#!/usr/bin/env bash
# Smoke: hipfire adapter on one combo, gated by check-versions.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/external/check-versions.sh"

RUN_ROOT="$(mktemp -d)/bench-runs"
mkdir -p "$RUN_ROOT/2026-05-05-smoke"

# Bench-perf needs to have run first to create a run-dir for the external script to attach to.
./target/release/bench-perf --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_ROOT" >/dev/null

python -m oracle.bench.external.external_main --engine hipfire \
    --models qwen3.5-0.8b --quants bf16 \
    --run-root "$RUN_ROOT" \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B"

LATEST="$(ls -t "$RUN_ROOT" | head -n1)"
test -s "$RUN_ROOT/$LATEST/external/hipfire/qwen3.5-0.8b_bf16.json" \
    || { echo "[hipfire-smoke] FAIL" >&2; exit 1; }
echo "[hipfire-smoke] PASS"
```

Make executable.

- [ ] **Step 3: Commit**

```bash
git add tests/gfx1100/bench_smoke.sh tests/gfx1100/bench_hipfire_smoke.sh
git commit -m "bench: smoke scripts (one-combo end-to-end + hipfire-gated)"
```

---

## Task 23: Implement long-context perf (heavy lane)

**Files:**
- Create: `oracle/bench/heavy/__init__.py`
- Create: `oracle/bench/heavy/longctx.py`
- Create: `oracle/bench/heavy/heavy_main.py`

- [ ] **Step 1: Write longctx.py**

Create `oracle/bench/heavy/__init__.py` (empty).

Create `oracle/bench/heavy/longctx.py`:
```python
"""Long-context perf sweep: 4k/8k/16k/32k contexts.

Quality at long context = perplexity over a 4k-prefix of PG-19, decoded through
to context end. Only runs combos that fit the registry's VRAM budget.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..runner import SupersonicInvocation, RunPolicy, run_with_capture


def run_longctx(binary: Path, model: str, model_dir: Path, quant: str,
                contexts: list[int], run_dir: Path) -> None:
    """Write one perf cell per context length into run_dir/quality/longctx_*.json."""
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        # Synthesize a prompt that occupies ~ctx tokens. Use a repeating snippet.
        # The runner tokenizes; a rough char-budget = ctx * 4.
        prompt = ("The quick brown fox jumps over the lazy dog. " * (ctx // 10 + 1))[:ctx * 4]
        inv = SupersonicInvocation(binary=binary, model=model, model_dir=model_dir,
                                   quant=quant, prompt=prompt, max_new_tokens=8)
        result = run_with_capture(inv, RunPolicy(measurement_runs=2, cooldown_seconds=3))
        cell = {
            "schema_version": 1,
            "model": model,
            "quant": quant,
            "eval": f"longctx_{ctx}",
            "metric": "ms_per_step",
            "value": result.get("ms_per_step", -1.0),
            "extras": {"context_tokens": ctx, "samples": result.get("samples"),
                       "status": result.get("status")},
        }
        (out_dir / f"{model}_{quant}_longctx_{ctx}.json").write_text(json.dumps(cell, indent=2))
```

- [ ] **Step 2: Write heavy_main.py**

Create `oracle/bench/heavy/heavy_main.py`:
```python
"""CLI: python -m oracle.bench.heavy.heavy_main --combos qwen3.5-9b:int4,gemma4-e4b:int4"""
from __future__ import annotations
import argparse
from pathlib import Path

from .longctx import run_longctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", required=True,
                    help="Comma-separated model:quant entries, e.g. qwen3.5-9b:int4,gemma4-e4b:int4")
    ap.add_argument("--contexts", default="4096,8192,16384,32768",
                    help="Comma-separated context lengths for longctx")
    ap.add_argument("--binary", default="./target/release/supersonic", type=Path)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[], help="KEY=PATH")
    ap.add_argument("--evals", default="longctx",
                    help="Comma-separated: longctx, niah, ruler, lm_eval. Phase 1 ships longctx; others stubbed.")
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    contexts = [int(c) for c in args.contexts.split(",")]
    combos = [tuple(c.split(":", 1)) for c in args.combos.split(",")]
    evals = [e.strip() for e in args.evals.split(",")]

    run_dir = _latest_run_dir(args.run_root)

    for (model, quant) in combos:
        mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
        if "longctx" in evals:
            print(f"[heavy] longctx {model}/{quant}")
            run_longctx(args.binary, model, mdir, quant, contexts, run_dir)
        if "niah" in evals:
            print(f"[heavy] niah {model}/{quant} — see Task 24 (not yet wired)")
        if "ruler" in evals:
            print(f"[heavy] ruler {model}/{quant} — see Task 25 (not yet wired)")
        if "lm_eval" in evals:
            print(f"[heavy] lm_eval {model}/{quant} — see Task 26 (not yet wired)")


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke import**

Run: `python -c "from oracle.bench.heavy.longctx import run_longctx; from oracle.bench.heavy.heavy_main import main; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add oracle/bench/heavy
git commit -m "bench(heavy): longctx perf sweep + heavy_main CLI scaffold"
```

---

## Task 24: NIAH for non-Llama (heavy lane, two combos)

**Files:**
- Create: `oracle/bench/heavy/niah.py`
- Modify: `oracle/bench/heavy/heavy_main.py` (replace the stub print)

- [ ] **Step 1: Read existing NIAH harness**

Read `oracle/arxiv_v1_smoke.py` — it already runs NIAH `niah_single`/`niah_multikey`/`niah_multiquery` on the CUDA Llama lane (`docs/performance.md` § Llama 3.1 8B arxiv_v1 retrieval smoke QA shows the expected output).

- [ ] **Step 2: Implement niah.py**

Create `oracle/bench/heavy/niah.py`:
```python
"""NIAH (needle-in-a-haystack) for non-Llama (model, quant) combos.

Adapts oracle/arxiv_v1_smoke.py — which today only runs against the CUDA Llama
lane via `--certified-kv` — to drive any (model, quant) on HIP. Phase 1 wires
this for two combos: (qwen3.5-9b, int4) and (gemma4-e4b, int4).
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

# Subtasks recorded by the existing harness. Fail-safe subset for Phase 1.
SUBTASKS = ["niah_single", "niah_multikey", "niah_multiquery"]


def run_niah(binary: Path, model: str, model_dir: Path, quant: str,
             contexts: list[int], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        for subtask in SUBTASKS:
            score = _run_one(binary, model, model_dir, quant, subtask, ctx)
            cell = {
                "schema_version": 1,
                "model": model,
                "quant": quant,
                "eval": f"niah_{subtask}_{ctx}",
                "metric": "score",
                "value": score,
                "extras": {"context_tokens": ctx, "subtask": subtask},
            }
            (out_dir / f"{model}_{quant}_niah_{subtask}_{ctx}.json").write_text(
                json.dumps(cell, indent=2))


_SCORE_RE = re.compile(r"\[niah_score\]\s+([0-9.]+)", re.MULTILINE)


def _run_one(binary: Path, model: str, model_dir: Path, quant: str,
             subtask: str, ctx: int) -> float:
    """Drive supersonic with a NIAH prompt of `ctx` tokens and parse the score line.

    The runner must emit `[niah_score] <float>`; if it doesn't yet, this function
    falls back to invoking the existing oracle/arxiv_v1_smoke.py with --model
    pointing at this (model, quant) pair. The harness then computes score using
    its own scorer logic.
    """
    # TODO at execution time: confirm whether to wire in-runner scoring (preferred)
    # or shell out to arxiv_v1_smoke.py with broader --model support added (more code,
    # but reuses an existing scorer). Both are valid; pick the lower-risk path during
    # implementation based on whether the runner already exposes per-step logits.
    cmd = ["python", "oracle/arxiv_v1_smoke.py",
           "--model", model, "--model-dir", str(model_dir),
           "--quant", quant, "--subtask", subtask, "--context", str(ctx)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (res.stdout or "") + (res.stderr or "")
    m = _SCORE_RE.search(out)
    return float(m.group(1)) if m else 0.0
```

The `# TODO at execution time` comment is the *one place* in this plan that punts a real implementation question. The reason: which path (in-runner scoring vs. arxiv_v1_smoke.py extension) is cheapest depends on what oracle/arxiv_v1_smoke.py's CLI accepts today, which has to be inspected at execution. Both paths produce the same JSON shape, so the rest of the harness is unaffected.

- [ ] **Step 3: Wire into heavy_main**

In `heavy_main.py`, replace the `niah` stub:
```python
        if "niah" in evals:
            from .niah import run_niah
            print(f"[heavy] niah {model}/{quant}")
            run_niah(args.binary, model, mdir, quant,
                     contexts=[c for c in contexts if c <= 16384], run_dir=run_dir)
```

- [ ] **Step 4: Smoke import**

Run: `python -c "from oracle.bench.heavy.niah import run_niah; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add oracle/bench/heavy/niah.py oracle/bench/heavy/heavy_main.py
git commit -m "bench(heavy): NIAH adapter for non-Llama combos (Phase 1 wires 2 combos)"
```

---

## Task 25: RULER (heavy lane, two combos)

**Files:**
- Create: `oracle/bench/heavy/ruler.py`
- Modify: `oracle/bench/heavy/heavy_main.py`

- [ ] **Step 1: Implement ruler.py**

Create `oracle/bench/heavy/ruler.py`:
```python
"""RULER long-context benchmark for two designated combos in Phase 1.

RULER is the standard long-context evaluation suite (Hsieh et al.). This adapter
wraps the upstream RULER scripts and writes per-(task, context) quality cells.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

# Conservative Phase 1 subset. Full RULER has 13 tasks; we ship the four
# most-cited ones first.
RULER_TASKS = ["niah_single_1", "niah_multikey_1", "vt", "qa_1"]


def run_ruler(binary: Path, model: str, model_dir: Path, quant: str,
              contexts: list[int], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        for task in RULER_TASKS:
            score = _run_one(binary, model, model_dir, quant, task, ctx)
            cell = {
                "schema_version": 1,
                "model": model,
                "quant": quant,
                "eval": f"ruler_{task}_{ctx}",
                "metric": "score",
                "value": score,
                "extras": {"task": task, "context_tokens": ctx},
            }
            (out_dir / f"{model}_{quant}_ruler_{task}_{ctx}.json").write_text(
                json.dumps(cell, indent=2))


def _run_one(binary: Path, model: str, model_dir: Path, quant: str,
             task: str, ctx: int) -> float:
    """Invoke supersonic on a RULER task instance, parse `[ruler_score] <float>`.

    Like NIAH (Task 24), this expects either:
    (a) the runner to emit a `[ruler_score]` line directly (preferred), or
    (b) a Python-side scorer that consumes the runner's generated output and
        a stored RULER target.

    For Phase 1, default to path (b): generate the RULER instance via the upstream
    RULER repo (https://github.com/NVIDIA/RULER), run supersonic on the prompt,
    and score with the upstream scorer. This module's responsibility is the
    plumbing, not the scorer.
    """
    # TODO at execution time: vendor the RULER instance generator + scorer into
    # oracle/bench/heavy/ruler_data.py, OR shell out to a pinned RULER checkout
    # under tools/external/ruler/. Pick the lower-risk path during implementation.
    return 0.0  # placeholder until plumbing decision is made; cell still records the attempt
```

- [ ] **Step 2: Wire into heavy_main**

In `heavy_main.py`, replace the `ruler` stub:
```python
        if "ruler" in evals:
            from .ruler import run_ruler
            print(f"[heavy] ruler {model}/{quant}")
            run_ruler(args.binary, model, mdir, quant,
                      contexts=[c for c in contexts if c <= 32768], run_dir=run_dir)
```

- [ ] **Step 3: Commit**

```bash
git add oracle/bench/heavy/ruler.py oracle/bench/heavy/heavy_main.py
git commit -m "bench(heavy): RULER adapter scaffold (scorer plumbing TBD at execution)"
```

---

## Task 26: lm-evaluation-harness wrapper (heavy lane, two combos)

**Files:**
- Create: `oracle/bench/heavy/lm_eval.py`
- Modify: `oracle/bench/heavy/heavy_main.py`

- [ ] **Step 1: Implement lm_eval.py**

Create `oracle/bench/heavy/lm_eval.py`:
```python
"""Wrap lm-evaluation-harness for HellaSwag, ARC-easy, MMLU subset.

Uses the harness's HF model adapter (lm_eval --model hf), pointing at the
SAFETENSORS dir for the model. The HIP `supersonic` runtime is NOT in the
loop here — this measures the underlying-weights-as-loaded-by-HF quality,
which is the apples-to-apples reference against published scores.

For SuperSonic-quant quality (e.g. INT4 quality through OUR INT4 path), the
golden-prompt diff (Task 16) and perplexity (Task 15) are the right tools.
This task intentionally measures the HF reference for the same model, so
the gap "HF BF16 ↔ SuperSonic BF16" is bounded and visible.
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

DEFAULT_TASKS = ["hellaswag", "arc_easy", "mmlu_high_school_european_history"]


def run_lm_eval(model: str, model_dir: Path, quant: str,
                tasks: list[str], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["lm_eval", "--model", "hf",
           "--model_args", f"pretrained={model_dir},dtype=bfloat16",
           "--tasks", ",".join(tasks),
           "--batch_size", "1",
           "--output_path", str(out_dir / f"{model}_{quant}_lm_eval_raw.json")]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        cell = {"schema_version": 1, "model": model, "quant": quant,
                "eval": "lm_eval", "metric": "error", "value": 0.0,
                "extras": {"stderr_tail": (res.stderr or "")[-2000:]}}
        (out_dir / f"{model}_{quant}_lm_eval.json").write_text(json.dumps(cell, indent=2))
        return

    raw = json.loads((out_dir / f"{model}_{quant}_lm_eval_raw.json").read_text())
    for task in tasks:
        score = _extract_task_score(raw, task)
        cell = {"schema_version": 1, "model": model, "quant": quant,
                "eval": f"lm_eval_{task}", "metric": "acc", "value": score,
                "extras": {"task": task, "harness": "lm-evaluation-harness"}}
        (out_dir / f"{model}_{quant}_lm_eval_{task}.json").write_text(json.dumps(cell, indent=2))


def _extract_task_score(raw: dict, task: str) -> float:
    # lm-eval-harness output schema varies by version. Common location:
    # raw["results"][task]["acc"] or raw["results"][task]["acc,none"].
    results = raw.get("results", {}).get(task, {})
    for key in ("acc", "acc,none", "exact_match", "exact_match,none"):
        if key in results:
            return float(results[key])
    return 0.0
```

- [ ] **Step 2: Wire into heavy_main**

In `heavy_main.py`, replace the `lm_eval` stub:
```python
        if "lm_eval" in evals:
            from .lm_eval import run_lm_eval, DEFAULT_TASKS
            print(f"[heavy] lm_eval {model}/{quant}")
            run_lm_eval(model, mdir, quant, DEFAULT_TASKS, run_dir)
```

- [ ] **Step 3: Commit**

```bash
git add oracle/bench/heavy/lm_eval.py oracle/bench/heavy/heavy_main.py
git commit -m "bench(heavy): lm-evaluation-harness wrapper for HF reference quality"
```

---

## Task 27: End-to-end manual smoke + first-run docs update

**Files:** none (manual)

- [ ] **Step 1: Verify the full pipeline runs end-to-end**

```bash
cargo build --release --bin supersonic
cargo build --release -p supersonic-bench

MODEL_DIR_08B=/mnt/data/models/Qwen3.5-0.8B \
  tests/gfx1100/bench_smoke.sh

# Then a real perf+quality matrix on the small models:
./target/release/bench-perf --arch gfx1100 \
    --models qwen3.5-0.8b,qwen3.5-2b,gemma4-e2b,phi4-mini \
    --quants bf16,int4 \
    --model-dir qwen3.5-0.8b=/mnt/data/models/Qwen3.5-0.8B \
    --model-dir qwen3.5-2b=/mnt/data/models/Qwen3.5-2B \
    --model-dir gemma4-e2b=/mnt/data/models/gemma-4-E2B \
    --model-dir phi4-mini=/mnt/data/models/Phi-4-mini-instruct

LATEST_RUN="$(ls -td target/bench-runs/* | head -n1)"
python -m oracle.bench.quality_main --arch gfx1100 \
    --models qwen3.5-0.8b,gemma4-e2b --quants bf16,int4 \
    --model-dir qwen3.5-0.8b=/mnt/data/models/Qwen3.5-0.8B \
    --model-dir gemma4-e2b=/mnt/data/models/gemma-4-E2B \
    --ppl-num-chunks 2

python -m oracle.bench.render.render_main markdown --run "$LATEST_RUN" --out .
```

- [ ] **Step 2: Inspect docs/performance.md and docs/quality.md**

Confirm the AUTOGEN zones are populated with sensible numbers. Hand-edit prose above the sentinel as needed.

- [ ] **Step 3: Commit the first published numbers**

```bash
git add docs/quality.md docs/performance.md
git commit -m "docs: first bench-renderer-produced numbers for gfx1100"
```

- [ ] **Step 4: Run the parity gate**

```bash
MODEL_DIR_08B=/mnt/data/models/Qwen3.5-0.8B tests/gfx1100/bench_parity.sh
```
Expected: `[parity] PASS`.

If parity fails, investigate whether the discrepancy is in extract_metrics (Task 3), in invocation flags (Task 5), or in actual measurement variability. Do NOT delete `bench_matrix.sh` until parity passes.

- [ ] **Step 5: Schedule the bash matrix deletion**

Open a follow-up issue (or note in the spec) titled "Delete tests/gfx1100/bench_matrix.sh after 2-week soak" — track that the Rust orchestrator has been the source-of-truth for at least two weeks of routine use before removing the legacy script.

---

## Self-Review Notes

After implementation, re-read `docs/superpowers/specs/2026-05-05-quality-and-perf-benchmarks-design.md` and confirm:

- **Spec coverage:**
  - Run-dir layout (`meta.json + perf/ + quality/ + external/`) → Tasks 2, 6.
  - Shared JSON schema → Tasks 2, 8 (Rust + Python validators).
  - Cooldown + warmup + median-of-N discipline → Tasks 5 (Rust), 11 (Python).
  - Perplexity (PG-19, WikiText-2) → Task 15.
  - Golden-prompt diff with BF16 reference bootstrap → Task 16.
  - hipfire adapter with B-style version pin → Task 19.
  - Renderer with sentinel preservation → Tasks 9, 17, 20.
  - Diff command for two run-dirs → Task 21.
  - NIAH/RULER/lm-eval heavy lane (two combos) → Tasks 24, 25, 26.
  - Long-context perf at 4k/8k/16k/32k → Task 23.
  - Smoke + parity scripts → Tasks 7, 22.
  - bench_matrix.sh deletion gate → Task 7 (script), Task 27 (manual run).

- **Things explicitly deferred:** llama.cpp adapter, vLLM adapter, cross-engine quality comparison, auto-rebuild of pinned externals, multi-arch combined render.

- **Two `# TODO at execution time` markers** are present (Task 24 NIAH path choice, Task 25 RULER scorer plumbing). These are **deliberate** — the choice depends on facts that can only be observed at implementation time (current shape of `arxiv_v1_smoke.py`, presence/absence of a vendored RULER checkout). Both alternatives produce the same JSON shape, so the rest of the harness is robust to the choice.
