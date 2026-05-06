# SuperSonic Kernel Lab

`kernel-lab` is a KernelBench-style harness for isolated SuperSonic kernels.
It is intentionally in this repository because the tasks call the same
`gpu-hal` and `kernel-ffi` launch paths as the runtime and parity tests.

V1 evaluates candidate worktrees. Run it once on a baseline checkout and once
on a candidate checkout, then compare the two run directories.

```bash
cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- list

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
```

The harness writes `meta.json`, one `tasks/*.json` file per task, and a
`summary.json` under `target/kernel-lab-runs/<date>-<git-sha>/`.
Per-case timings use HIP events when available and record
`timing_source: "hip_event"` in JSON; unsupported timing backends fall back to
synchronized wall-clock timing.

`--tasks` accepts `all` for the required suite, `everything` for every registry
entry, comma-separated task ids, or comma-separated `tag:<name>` selectors. Tag
expansion is de-duplicated in registry order. The required task set stays fast;
larger optional shapes are behind `--tasks tag:stress`.

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
Exit code `2` means correctness failed, `3` means latency regression, and `4`
means the run was correct but missed `--min-speedup`. `--github-summary`
appends the markdown diff to `$GITHUB_STEP_SUMMARY` for Actions jobs.
