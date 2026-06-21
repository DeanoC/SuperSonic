# SuperSonic Kernel Lab

`kernel-lab` is a KernelBench-style harness for isolated SuperSonic kernels.
It is intentionally in this repository because the tasks call the same
`gpu-hal` and `kernel-ffi` launch paths as the runtime and parity tests.
See the [repo architecture map](repo-architecture.md) and
[consolidation roadmap](consolidation-roadmap.md) for how kernel-lab fits into
the broader lab and validation tooling.

V1 evaluates candidate worktrees. Run it once on a baseline checkout and once
on a candidate checkout, then compare the two run directories.

```bash
cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- list

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- list --json

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- list --tags

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- describe tag:prefill

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- run \
  --tasks all \
  --backend hip \
  --device 0 \
  --warmup 5 \
  --iters 20

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- run \
  --tasks tag:prefill,tag:moe \
  --backend hip \
  --device 0

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- run \
  --tasks tag:functional \
  --backend hip \
  --device 0

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- diff \
  --baseline target/kernel-lab-runs/BASELINE \
  --candidate target/kernel-lab-runs/CANDIDATE \
  --markdown-out target/kernel-lab-runs/diff.md

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- compare-ref \
  --baseline-ref main \
  --candidate-ref worktree \
  --tasks all \
  --backend hip \
  --device 0

cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- baseline \
  --run target/kernel-lab-runs/CANDIDATE \
  --name required
```

The harness writes `meta.json`, `task_manifest.json`, one `tasks/*.json` file
per task, `summary.json`, and `summary.md` under
`target/kernel-lab-runs/<date>-<git-sha>/`.
Per-case timings use HIP events when available and record
`timing_source: "hip_event"` in JSON; unsupported timing backends fall back to
synchronized wall-clock timing.

The Rust task registry is the source of truth for task metadata. `list --json`
prints serializable task snapshots, `list --tags` summarizes tag membership, and
`describe <task-id|tag:tag> [--json]` shows the description, backend support,
tags, required/optional status, and correctness contract.

`--tasks` accepts `all` for the required suite, `everything` for every registry
entry, comma-separated task ids, or comma-separated `tag:<name>` selectors. Tag
expansion is de-duplicated in registry order. The required task set stays fast;
larger optional shapes are behind `--tasks tag:stress`, and primitive
correctness checks are behind `--tasks tag:functional`.

The `tag:functional` suite is a non-gating correctness corpus for compact
primitive-level and compound checks such as BF16 RMSNorm, BF16 RoPE, INT4
dequant matvec, and a Qwen3.6 MoE route -> grouped-expert -> combine pipeline.
These tasks are intended to catch wrong-answer regressions before expanding
performance coverage or promoting any subset into `all`.

V1 required tasks cover the current high-leverage Qwen kernel surface:

- `qwen35.full_attention_prefill`
- `qwen36.batched_prefill_attn_full`
- `qwen36.router_permute`
- `qwen36.grouped_expert_int4`
- `qwen36.unpermute_combine`

Every run appends one compact line to `target/kernel-lab-runs/history.jsonl`,
including metadata and per-task median latency. This gives a local regression
trail across candidate runs.

`compare-ref` checks out non-`worktree` refs into `target/kernel-lab-worktrees/`
and overlays only the current `crates/kernel-lab` harness crate into that
temporary checkout. The kernels and FFI crates are still built from the checked
out ref, so a baseline run measures baseline kernel code even before this
harness lands on `main`.

Scoring is conservative: a task is valid only if every correctness case passes.
`diff` reports per-task speedup, `fast_p`, geometric mean speedup, and fails on
correctness regressions or median latency regressions above `--max-regression`.
`--latency-floor-us` can make latency regressions non-blocking for very small
tasks whose baseline median is below the floor; correctness failures still fail
regardless of latency.
Diff JSON and markdown include machine-readable status/reason fields for each
required task, including missing candidate tasks, incorrect candidate tasks, and
latency regressions. The top-level diff status/reason also classifies geomean
speedup failures.
Exit code `2` means correctness failed, `3` means latency regression, and `4`
means the run was correct but missed `--min-speedup`. `--github-summary`
appends the markdown diff to `$GITHUB_STEP_SUMMARY` for Actions jobs.

## CI

`.github/workflows/kernel-lab.yml` runs on PRs that touch kernel-lab, kernel FFI,
GPU HAL, Cargo metadata, or HIP kernel sources. It targets a self-hosted ROCm
runner labelled `self-hosted`, `linux`, and `rocm`. The default PR suite is
`all` with short warmup/iteration counts; use `workflow_dispatch` to run broader
selectors such as `everything`, `tag:functional`, or `tag:stress` with custom
iteration counts.

Because the job executes candidate code on a persistent self-hosted GPU host,
the PR trigger is restricted to same-repository branches. Fork PRs must be run
through a trusted manual `workflow_dispatch` after review, or on disposable
isolated runners.

The workflow compares the checked-out candidate worktree against the PR base
commit using `compare-ref`, writes markdown to the GitHub step summary, and
uploads the run JSON/markdown artifacts.

Before building, the workflow waits for the ROCm device to report at most 10%
GPU use and at most 20% VRAM allocation for three consecutive 30-second
samples. If the device stays busy for one hour, the job fails with a clear
error. The default PR preset also uses a `75us` latency floor so tiny required
tasks remain correctness-gated without failing on incidental GPU timing noise.
timeout rather than running a noisy contended benchmark.

## Baselines

Reviewed baseline summaries live under
`crates/kernel-lab/baselines/<arch>/<name>.summary.json`. Generate one from a
run directory with:

```bash
cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- baseline \
  --run target/kernel-lab-runs/2026-05-06-abcdef0 \
  --name required
```

The command groups artifacts by the run's detected architecture, writes the
canonical JSON summary, and also writes a markdown sidecar unless
`--no-markdown` is passed. These checked-in summaries are intended for stable
release or lab-machine comparisons; PR gating should still prefer `compare-ref`
when a suitable ROCm runner is available.
