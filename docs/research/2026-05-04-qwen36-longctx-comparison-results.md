# Qwen3.6 Long-Context Comparison Results - 2026-05-04

This records the first `--preset comparison` run from
`tests/gfx1100/bench_qwen36_longctx.py` after PR #200 added the preset and
summary logic.

## Command

```bash
python3 tests/gfx1100/bench_qwen36_longctx.py \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --preset comparison \
  --out-json target/qwen36_longctx_comparison.json \
  --out-md target/qwen36_longctx_comparison.md
```

The binary was built with:

```bash
cargo build --release --bin supersonic
```

## Summary

Recommendation from the harness: prioritize full-attention/KV bandwidth work
for longer contexts.

| Context | best mode | baseline ms/tok | best ms/tok | best vs baseline | prefill s | gen wall ms | likely bottleneck |
|---:|:---|---:|---:|---:|---:|---:|:---|
| 512 | int4-vmm | 34.54 | 34.54 | +0.0% | 13.30 | 138.18 | full_attn |
| 2048 | int4-vmm | 52.74 | 52.74 | +0.0% | 73.51 | 210.98 | full_attn |
| 4096 | int4-vmm | 76.75 | 76.75 | +0.0% | 196.27 | 307.00 | full_attn |
| 8192 | int4-vmm | 124.94 | 124.94 | +0.0% | 576.28 | 499.77 | full_attn |

## Detail

| Context | Mode | wall s | prefill s | gen wall s | total ms/tok | tok/s | prompt tokens | total resident GiB | MoE resident GiB | KV resident GiB | generated ids match | NIAH hit |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 512 | int4-vmm | 16.16 | 13.30 | 0.14 | 34.54 | 28.95 | 418 | 15.04 | 15.00 | 0.04 | yes | NO |
| 512 | int4-kv-fp8 | 16.36 | 13.46 | 0.14 | 34.97 | 28.60 | 418 | 15.04 | 15.00 | 0.04 | yes | NO |
| 2048 | int4-vmm | 76.41 | 73.51 | 0.21 | 52.74 | 18.96 | 1763 | 15.04 | 15.00 | 0.04 | yes | NO |
| 2048 | int4-kv-fp8 | 77.88 | 74.95 | 0.21 | 53.64 | 18.64 | 1763 | 15.04 | 15.00 | 0.04 | yes | NO |
| 4096 | int4-vmm | 199.43 | 196.27 | 0.31 | 76.75 | 13.03 | 3553 | 15.08 | 15.00 | 0.08 | yes | NO |
| 4096 | int4-kv-fp8 | 200.81 | 197.73 | 0.31 | 78.36 | 12.76 | 3553 | 15.04 | 15.00 | 0.04 | yes | NO |
| 8192 | int4-vmm | 579.54 | 576.28 | 0.50 | 124.94 | 8.00 | 7143 | 15.16 | 15.00 | 0.16 | yes | NO |
| 8192 | int4-kv-fp8 | 591.05 | 587.78 | 0.51 | 128.08 | 7.81 | 7143 | 15.08 | 15.00 | 0.08 | yes | NO |

## Interpretation

`int4-vmm` was the best mode at every tested context.  KV-FP8 reduced reported
KV residency at 4096 and 8192 requested context tokens, but it did not improve
throughput or prefill time in this single-sequence comparison run.

The dominant scaling problem is prefill.  Requested context grew 16x from 512
to 8192, while prefill time grew from 13.30 seconds to 576.28 seconds.  Decode
tail also slowed from 34.54 ms/token to 124.94 ms/token, but it is not the main
wall-clock contributor for these short-generation long-context prompts.

The next performance PR should focus on Qwen3.6 full-attention/KV bandwidth in
the HIP path before more KV-FP8 policy work.  The result does not argue against
KV compression for larger contexts or batch serving; it only shows that the
current single-sequence KV-FP8 path is not a speed win through 8k requested
context on this machine.
