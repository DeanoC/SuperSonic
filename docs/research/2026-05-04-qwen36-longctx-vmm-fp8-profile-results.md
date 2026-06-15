# Qwen3.6 Long-Context VMM/KV-FP8 Profile Results - 2026-05-04

Branch: `research/qwen36-longctx-vmm-fp8-profiles`  
Worktree: `/home/deano/projects/SuperSonicBase-qwen36-longctx-vmm-fp8-profiles`  
Base: `3cd856d` (`Merge pull request #205 from DeanoC/perf/qwen36-longctx-full-attn`)

## Run

The primary profile was run after checking the GPU was idle with
`rocm-smi --showuse --showmemuse --showpidgpus`. The wrapper also checked for
idle GPU state before each row.

```bash
SUPERSONIC_BACKENDS=hip cargo build --release --bin supersonic

python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile vmm-fp8-large \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --max-new-tokens 4 \
  --timeout 2400 \
  --out-dir target/qwen36_longctx_profiles/vmm-fp8-large
```

Raw outputs:

- `target/qwen36_longctx_profiles/vmm-fp8-large/vmm-fp8-large_combined.json`
- `target/qwen36_longctx_profiles/vmm-fp8-large/vmm-fp8-large_*.json`
- `target/qwen36_longctx_profiles/vmm-fp8-large/vmm-fp8-large_*.md`
- `target/qwen36_longctx_profiles/sparse-vmm-fp8/sparse-vmm-fp8_8192_sparse.json`
- `target/qwen36_longctx_profiles/sparse-vmm-fp8/sparse-vmm-fp8_8192_sparse.md`

## Results

| Context | Mode | Prefill s | Decode ms/tok | Full-attn ms/tok | Tok/s | Total resident GiB | KV resident GiB | IDs match baseline | NIAH hit |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8192 | `int4-vmm` | 313.01 | 56.27 | 55.82 | 17.77 | 15.16 | 0.16 | yes | NO |
| 8192 | `int4-kv-fp8` | 313.67 | 56.35 | 55.86 | 17.75 | 15.08 | 0.08 | - | NO |
| 16384 | `int4-vmm` | 808.80 | 79.41 | 78.97 | 12.59 | 15.27 | 0.27 | yes | NO |
| 16384 | `int4-kv-fp8` | 814.04 | 80.69 | 80.25 | 12.39 | 15.16 | 0.16 | - | NO |

Partial sparse follow-up:

| Context | Mode | Prefill s | Decode ms/tok | Full-attn ms/tok | Tok/s | Total resident GiB | MoE resident GiB | KV resident GiB | NIAH hit |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8192 | `cap320` | 1135.73 | 179.79 | 164.66 | 5.56 | 1.41 | 1.25 | 0.16 | NO |

## Readout

KV-FP8 reduced resident KV memory as expected:

- 8k: 0.16 GiB to 0.08 GiB, saving 0.08 GiB.
- 16k: 0.27 GiB to 0.16 GiB, saving 0.12 GiB.

The decode tradeoff was small but not favorable in this run:

- 8k: KV-FP8 was +0.14% slower by decode ms/token.
- 16k: KV-FP8 was +1.62% slower by decode ms/token.

Full attention remains the measured decode bottleneck. `full_attn_ms_avg`
accounts for almost all `total_ms_avg` in every row, and FFN timing is not
visible in the per-token chain breakdown for these measured decode steps.

The sparse `cap320` row cut total residency from 15.16 GiB to 1.41 GiB at 8k,
but it was much slower: prefill was 3.63x dense and decode ms/token was 3.19x
dense. Follow-up work on `perf/qwen36-sparse-vmm-longctx-regression` found that
this row's `full_attn_ms_avg` was misattributed: segmented sparse persistent
decode had been reporting router, route D2H, residency page-in/remap, and FFN
resume work in the full-attention timing bucket. A corrected repro measured
`full_attn_ms_avg=48.19` and `ffn_ms_avg=114.43` for 8k `cap320`, so the sparse
regression is in FFN/residency rather than full attention.

The NIAH hit was false for all rows despite identical generated IDs across the
dense and KV-FP8 rows at each context. These runs used only 4 generated tokens,
so they are useful for performance and ID agreement, not for validating long
answer recovery.

## Next

Do not spend the next pass on 16k sparse timings until the sparse prefill and
FFN/residency regressions are understood. The higher-value next step is to
separate page-in/remap time from FFN-only kernel time, then rerun only the 8k
sparse and sparse+KV-FP8 rows.
