# Qwen3.6 Long-Context Full-Attention Pass - 2026-05-04

This records the first dense full-attention/KV bandwidth pass after
`docs/research/2026-05-04-qwen36-longctx-comparison-results.md` identified
`full_attn` as the long-context bottleneck.

## Change

`kernels/qwen36_moe_persistent/full_attn_phase.cuh` Phase G now has two cached
KV read paths:

1. Short contexts stream KV through an online softmax accumulator for
   `kv_len > 1`. The old path wrote one score per `(query head, position)` to
   workspace, then ran a serial softmax pass and a separate V-weighted
   reduction. The online path keeps the running max, denominator, and per-lane V
   accumulator in registers and avoids the score workspace write/read.
2. Long contexts split each query head's KV history into 384-token tiles. Tile
   workers write `(tile max, tile sum, tile V partial[d])` into the existing
   per-head `OFF_SCORES` workspace, then a second pass combines those tile
   partials with max-stable softmax rescaling. This follows the same partial
   reduction shape used by hipfire-style tiled attention without vendoring
   hipfire code.

The tiled path is gated on `kv_cache_k != null`, `kv_len > 384`,
`head_dim <= block_size`, and `tile_count * (head_dim + 2) <= kv_max_t`, so it
reuses the already allocated `[H * kv_max_t]` score workspace. Cache writes, the
`kv_len == 1` path, BF16 KV, KV-FP8 scale reads, and the BF16 sidecar window are
intentionally unchanged.

## Measurements

Baseline is the prior report's `int4-vmm` row from
`2026-05-04-qwen36-longctx-comparison-results.md`. Candidate rows were run
without warmup, so these are directional single-process checks rather than a
full median gate.

| Context | baseline prefill s | candidate prefill s | prefill delta | baseline ms/tok | candidate ms/tok | ms/tok delta |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 13.30 | 13.30 | +0.0% | 34.54 | 33.99 | -1.6% |
| 8192 | 576.28 | 313.24 | -45.6% | 124.94 | 56.39 | -54.9% |

The final 8192 candidate row:

```text
prefill_ms=313238.480 total_ms=56.389 tok/s=17.734 generated_ids=[271,248068,271,248069]
```

The 512 row remains effectively unchanged. Depending on the exact allocated
`kv_max_t`, it either stays on the online path throughout or only reaches the
tiled path for the tail end of the 418-token measured prompt.

## Tuning Notes

The initial online-softmax-only pass was useful but stayed near the noise band.
The tiled path is the first variant that clearly moves the 8192 row.

| Variant | 8192 prefill s | 8192 ms/tok |
|:---|---:|---:|
| prior report baseline | 576.28 | 124.94 |
| online softmax, per-lane exp | 525.52 | 112.95 |
| tiled partials, 512-token tiles | 321.10 | 54.88 |
| tiled partials, 384-token tiles | 313.24 | 56.39 |

A follow-up micro-change moved the online-softmax exponentials to lane 0 and
broadcast the factors through shared memory. It regressed the 8192 row:

| Variant | prefill s | ms/tok |
|:---|---:|---:|
| online softmax, per-lane exp | 525.52 | 112.95 |
| lane-0 exp + shared broadcast | 544.72 | 116.74 |

The extra synchronization cost was worse than the redundant per-lane exponentials
on this kernel shape, so it was reverted.

The 384-token tiled variant has better 8192 prefill/wall time than 512-token
tiles, while 512-token tiles have slightly better generated-token latency. This
branch keeps 384 because the pass is targeted at long-context prefill/full-attn
wall time; the difference is small enough that a later median gate may choose a
different tile size.

## Commands

```bash
SUPERSONIC_BACKENDS=hip cargo build --release --bin supersonic

python3 tests/gfx1100/bench_qwen36_longctx.py \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --contexts 8192 \
  --modes int4-vmm \
  --max-new-tokens 4 \
  --no-warmup \
  --timeout 1800 \
  --out-json target/qwen36_longctx_8192_tiled384_candidate.json \
  --out-md target/qwen36_longctx_8192_tiled384_candidate.md
```

## Verification

```bash
python3 -m unittest tests/test_qwen36_longctx_bench.py
SUPERSONIC_BACKENDS=hip SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B \
  cargo test --release -p runner --test qwen36_moe_bf16_kv_vmm_smoke -- --nocapture

SUPERSONIC_BACKENDS=hip SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B \
  cargo test --release -p runner --test qwen36_moe_longctx_tiled_smoke -- --nocapture
```

Both checks passed after the online-softmax change.
They also passed after adding the tiled path and after changing the tile length
from 512 to 384. The long-context tiled smoke forces `prefill_steps > 384` and
asserts the deterministic generated IDs for that prompt.

## Interpretation

The tiled partial path produces a decisive single-run 8192 improvement relative
to the prior report and to the online-only candidate. It should still get a clean
baseline/candidate median gate before being claimed as a final benchmark number,
but the delta is far beyond the plus/minus 10-15% within-session noise band that
the prior hipfire comparison warned about.
