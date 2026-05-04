# HIP Qwen3.6 Long-Context Performance

This branch is measurement-first.  The target is Qwen3.6-35B-A3B on HIP,
using the local model at `/mnt/data/models/Qwen3.6-35B-A3B`, with larger
decode contexts than the short-prompt performance matrix currently covers.

## Local Measurement Harness

`tests/gfx1100/bench_qwen36_longctx.py` runs deterministic NIAH-style prompts
through the existing `supersonic` CLI and records:

- requested context size, prompt token count, generated ids, and NIAH substring hit
- wall-clock run time, so prefill-heavy long prompts are visible
- `[qwen36-moe stage-timings]` fields and derived tok/s
- dense or sparse VMM residency fields, including total, MoE, and KV bytes
- JSON plus Markdown summaries under `target/`

Default sweep:

```bash
.venv/bin/python tests/gfx1100/bench_qwen36_longctx.py \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --contexts 8192,16384,32768 \
  --modes int4-vmm,int4-kv-fp8 \
  --max-new-tokens 16
```

Optional sparse-MoE rows can be added with:

```bash
.venv/bin/python tests/gfx1100/bench_qwen36_longctx.py \
  --modes int4-vmm,int4-kv-fp8,sparse,sparse-kv-fp8 \
  --sparse-caps 320
```

## VMM Baseline

The harness sets `SUPERSONIC_VMM_MOE_ISLANDS=1` by default so the baseline is
the virtual-MoE path needed for long contexts.  Pass `--no-force-moe-vmm` only
when intentionally testing dense expert residency or fallback behavior.

## hipfire Reference Targets

`Kaden-Schutt/hipfire` is used as a reference, not vendored code.  The useful
comparison areas for follow-up work are:

- long-context fixtures: hipfire's NIAH-style benchmark shape gives a simple
  pass/fail correctness signal alongside throughput
- PFlash / tri-attention paths: candidate references if SuperSonic's
  Qwen3.6 attention stage becomes the long-context bottleneck
- asymmetric and low-bit KV attention kernels: candidate references if KV-FP8
  is memory-positive but throughput-neutral or slower
- RDNA-tuned GEMV/GEMM/HFQ kernels: candidate references only if stage timing
  shows routed expert matvec remains dominant after VMM policy choices

## First Optimization Gate

Do not port kernels on this branch until the harness identifies the limiting
stage at 8k, 16k, and 32k contexts.  The first likely follow-up is whichever
of these is supported by data:

- attention/KV bandwidth work if `attn_ms_avg` grows with context and dominates
- KV-FP8 sidecar or residency changes if memory improves without decode speed
- sparse-MoE prefetch/residency policy work if page misses dominate sparse rows
- GEMV/HFQ investigation if FFN or routed expert time dominates independent of
  context length
