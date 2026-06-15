# Qwen3.6 Sparse VMM Long-Context Regression - 2026-05-04

Branch: `perf/qwen36-sparse-vmm-longctx-regression`  
Worktree: `/home/deano/projects/SuperSonicBase-qwen36-sparse-vmm-regression`  
Base: `535ed60` (`Merge pull request #206 from DeanoC/research/qwen36-longctx-vmm-fp8-profiles`)

## Starting Point

The previous profile showed this 8k result:

| Mode | Prefill s | Decode ms/tok | Full-attn ms/tok | Total resident GiB | MoE resident GiB |
|:---|---:|---:|---:|---:|---:|
| `int4-vmm` | 313.01 | 56.27 | 55.82 | 15.16 | 15.00 |
| `cap320` | 1135.73 | 179.79 | 164.66 | 1.41 | 1.25 |

The headline problem was that sparse `cap320` cut residency dramatically but
appeared to make full attention 3x slower.

## Finding

The sparse `cap320` path uses segmented persistent decode:

1. lookahead prefetch
2. router-only persistent launch
3. route D2H
4. demand page-in/remap
5. FFN-only persistent launch

Before this branch, `run_sparse_with_expert_prefetch` recorded the entire
segmented path wall time in `DecodeOutputs.kernel_full_attn_us`. That made
`full_attn_ms_avg` include router D2H, MoE residency page-in/remap, and FFN
resume work. The previous `full_attn_ms_avg=164.66` for `cap320` is therefore
not a valid full-attention measurement.

## Change

`run_sparse_with_expert_prefetch` now reports:

- `kernel_full_attn_us`: router-only launch plus top-k route download.
- `kernel_ffn_us`: lookahead prefetch, demand prefetch/page-in, and FFN-only
  launch time.

This does not make sparse faster by itself. It makes the stage timing useful
for the next pass, where the real question is how much time is in residency
page-in/remap versus FFN-only kernels.

The follow-up timing split adds:

- `lookahead_prefetch_ms_avg`
- `router_launch_ms_avg`
- `route_d2h_ms_avg`
- `demand_prefetch_ms_avg`
- `ffn_launch_ms_avg`

These are printed as `[qwen36-moe sparse-breakdown]` under
`--emit-stage-timings` and captured in the long-context benchmark JSON as
`sparse_breakdown`.

The long-context profiling harness now also exposes the existing sparse
runtime policy knobs:

- `--sparse-prefetch`
- `--sparse-prefetch-ranks`
- `--sparse-prefetch-transition-min-obs`
- `--sparse-protected-experts`
- `--sparse-async-prefetch`
- `--sparse-async-staging-pages`

The GPU-idle wrapper forwards those flags to each row so policy variants can be
run without hand-editing environment variables.

## Validation

- `rustfmt --check crates/runner/src/qwen36_moe/persistent_decode.rs`
- `cargo test -p runner qwen36_moe::vmm_config --lib`
- `cargo check -p runner --bin supersonic`
- `SUPERSONIC_BACKENDS=hip cargo build --release --bin supersonic`
- `python3 -m py_compile tests/gfx1100/bench_qwen36_longctx.py tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py`
- parser smoke for `[qwen36-moe sparse-breakdown]`
- 8k `cap320` repro under GPU-idle gating:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile sparse-vmm-fp8 \
  --contexts 8192 \
  --modes sparse \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --sparse-caps 320 \
  --max-new-tokens 4 \
  --timeout 3000 \
  --max-mem-use 8 \
  --out-dir target/qwen36_sparse_vmm_regression/cap320-attribution
```

Corrected 8k `cap320` attribution:

| Mode | Prefill s | Decode ms/tok | Chain ms/tok | Full/router ms/tok | FFN/residency ms/tok | Total resident GiB |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 1146.35 | 166.40 | 163.06 | 48.19 | 114.43 | 1.41 |

The corrected run confirms sparse decode is still slow, but it is not a
full-attention regression. The dominant bucket is FFN/residency. Compared with
the dense 8k profile (`56.27 ms/tok` total), sparse is still 2.96x slower, but
the next optimization target is page-in/remap and FFN-only segmented execution.

Sub-bucket 8k `cap320` repro:

| Mode | Decode ms/tok | Chain ms/tok | Router launch ms/tok | Route D2H ms/tok | Demand prefetch ms/tok | FFN launch ms/tok |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 182.94 | 169.49 | 46.91 | 1.85 | 108.10 | 12.14 |

This identifies the main sparse decode cost as demand page-in/remap, not the
FFN-only kernel. Lookahead prefetch was effectively zero (`0.04 ms/tok`) in
this run, which means the current transition prefetch policy is not hiding the
demand residency work for the measured generation tokens.

Residency telemetry explains why:

| Mode | Prefetch requests | Prefetch skipped | Prefetch uploaded bytes | Page misses | Uploaded bytes | Unmapped bytes |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 2,285,846 | 2,052,482 | 0 | 4,350,230 | 9,123,093,544,960 | 9,121,751,367,680 |

Transition lookahead requested many candidates, but the resident page budget
was already full. It therefore skipped prefetch uploads and left demand
page-in/remap to do roughly 9.1 TB of uploads/unmaps over the 8k profile.

Async transition prefetch repro:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile sparse-vmm-fp8 \
  --contexts 8192 \
  --modes sparse \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --sparse-caps 320 \
  --max-new-tokens 4 \
  --timeout 3000 \
  --max-mem-use 8 \
  --sparse-prefetch transition \
  --sparse-prefetch-ranks 4 \
  --sparse-async-prefetch \
  --sparse-async-staging-pages 32 \
  --out-dir target/qwen36_sparse_vmm_regression/cap320-transition-r4-async32
```

| Mode | Decode ms/tok | Chain ms/tok | Router launch ms/tok | Route D2H ms/tok | Demand prefetch ms/tok | FFN launch ms/tok | Async scheduled pages | Async capacity skips |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32` | 180.92 | 167.53 | 46.78 | 1.83 | 106.25 | 12.15 | 0 | 2,052,080 |

Async prefetch did not schedule any pages at `cap320` because the page budget
was full. This confirms that the next optimization target is prefetch capacity
or eviction policy, not merely enabling the async prefetch stream.

Repo-wide `cargo fmt --check` currently reports unrelated formatting drift in
other files, so this branch intentionally used targeted formatting validation.
