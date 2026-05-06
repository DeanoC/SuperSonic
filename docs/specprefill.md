# SpecPrefill

Speculator-driven sparse target prefill for long-prompt TTFT
optimization, based on [arXiv 2502.02789](https://arxiv.org/abs/2502.02789).
Currently shipping for Qwen3.5-9B target + Qwen3.5-0.8B draft on HIP
(gfx1100), plus CUDA Qwen3.6-35B-A3B cross-family cosine mode with a
Qwen3.5-0.8B drafter. Greedy decode only.

## When to use it

- Prompt is long (≥ 1k tokens) and TTFT (time to first token) matters.
- You're running greedy decode (`--max-new-tokens` ≥ 1, no top-p).
- Target is Qwen3.5-9B on HIP, or Qwen3.6-35B-A3B INT4 on CUDA sm86.

## When NOT to use it

- Sampling-based decode (top-p, temperature > 0). Top-5 stability is
  poor at low keep ratios — see Phase A2 measurements.
- Aggressive Qwen3.6 CUDA keep ratios for quality-sensitive generation.
  `0.50` is the balanced/perf lane and may change next-token argmax;
  use `0.75` for the conservative lane.
- Very long prompts (>8192 tokens). The legacy `lookahead` kernel is
  bounded by per-block LDS and trips a clear FFI error past that
  point. The default `cosine` path has no comparable bound and has
  been measured cleanly at 8k tokens (3.36× faster than dense — see
  Performance below); beyond 8k is untested.
- 24 GiB GPU + a model larger than Qwen3.5-9B in BF16. Doesn't fit.

## Flags

```bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/Qwen3.5-9B \
    --specprefill-draft-dir /path/to/Qwen3.5-0.8B \
    --specprefill-keep-ratio 0.50 \
    --prompt "..." --max-new-tokens 32
```

| Flag | Default | Notes |
|---|---|---|
| `--specprefill-draft-dir <path>` | (none) | Required. Same-family draft (Qwen3.5-0.8B). Presence enables SpecPrefill. |
| `--specprefill-keep-ratio <0.05..1.0>` | 0.50 generally; 0.75 for CUDA Qwen3.6 cross-family | Fraction of prompt tokens kept by chunked top-K selection. Qwen3.6 CUDA: `0.50` = balanced/perf, `0.75` = conservative. |
| `--specprefill-algorithm <cosine\|lookahead>` | cosine | Importance-scoring algorithm. See [Algorithm](#algorithm). |
| `--specprefill-chunk-size <int>` | 32 | Selection chunk size (paper §3.4). |
| `--specprefill-pool-window <odd int>` | 5 | 1-D smoothing window for importance scores. |
| `--specprefill-lookahead <1..16>` | 4 | Number of look-ahead decode steps on the draft (`lookahead` algorithm only). |
| `--specprefill-always-keep-prefix <int>` | 4 | Force-keep first N tokens (BOS / system). |
| `--specprefill-always-keep-suffix <int ≥ 1>` | 4 | Force-keep last N tokens. Must be ≥ 1 (the first decode logits come from this slot). |
| `--specprefill-unload-draft` | false | Free the draft weights between selection and target prefill (claws back ~1.6 GiB). |

## Algorithm

Two scoring algorithms are available; both feed the same chunked top-K
selection (`keep_ratio` × `chunk_size`) and produce the same kept-token
schedule downstream:

- `cosine` (default, Phase D): per-block `cosine(block_mean_K, last_K)`
  computed from one drafter K cache. The drafter prefill stops after
  the deepest scoring layer (early-exit through `prefill_kv_through`)
  — for `shallowest` mode on Qwen3.5-0.8B that prunes ~26 of 28 layers.
  The orchestrator then picks one full-attention layer and a HIP kernel
  emits per-block cosine scores in one launch. The default scoring
  layer is the **shallowest** full-attention layer (env var
  `SUPERSONIC_SPECPREFILL_SCORE_LAYER=N` overrides). The default
  aggregation mode is `shallowest`; set
  `SUPERSONIC_SPECPREFILL_LAYERS=all_max` to take the per-block max
  across all full-attention layers — `all_max` doesn't benefit from
  early-exit (the deepest full-attention layer is near the end of the
  drafter), so it ends up at roughly dense-baseline TTFT in our
  measurements (see Performance below).
- `lookahead` (legacy, Phase C): per-layer `softmax(Q·Kᵀ)` over
  look-ahead query rows from `--specprefill-lookahead` decode steps on
  the draft. Correctness-validated and structurally faithful to the
  paper, but the lookahead decode steps route through the component
  decode path on gfx1100 and end up NET SLOWER than dense prefill.
  Kept available for parity-test coverage and future research.

The cosine path is inspired by [hipfire](https://github.com/Kaden-Schutt/hipfire)'s
PFlash implementation. It does not require multi-layer attention
softmax, drops the lookahead decode entirely, and ships measurably
faster TTFT than dense.

## Performance

Measured on gfx1100 with Qwen3.5-9B target + Qwen3.5-0.8B draft,
warmup pass + 3 measurement runs (median reported),
`--specprefill-keep-ratio 0.50`. Three prompt lengths: 1.3k tokens
(the original measurement) plus 4k and 8k (the regimes SpecPrefill is
actually meant for):

| Mode | 1.3k tok | 4k tok | 8k tok |
|---|---:|---:|---:|
| dense (no SpecPrefill) | 4941 ms | 29374 ms | 124220 ms |
| `cosine`, `shallowest` (default) | **2385 ms** | **11157 ms** | **36939 ms** |
| `cosine`, `all_max` | 4144 ms | 20893 ms | 74948 ms |
| `lookahead` (legacy) | 7846 ms | 24950 ms | 80363 ms |

Speedup vs dense:

| Mode | 1.3k | 4k | 8k |
|---|---:|---:|---:|
| `cosine`, `shallowest` (default) | **2.07×** | **2.63×** | **3.36×** |
| `cosine`, `all_max` | 1.19× | 1.41× | 1.66× |
| `lookahead` (legacy) | 0.63× (slower) | 1.18× | 1.55× |

The default `cosine` + `shallowest` path's speedup *compounds with
prompt length*: 2.07× at 1.3k → 3.36× at 8k, saving ~87 seconds of
TTFT on a single 8k prefill. This is the regime SpecPrefill is
designed for. The legacy `lookahead` path is slower than dense at
short prompts but pulls ahead of dense at ≥4k tokens; it is still
beaten by `cosine` at every measured length.

Quality at `keep_ratio=0.50` on the 1353-token prompt (against the
dense reference; cossim is a regression backstop, argmax-match is the
primary correctness gate):

| Algorithm | argmax match | cossim |
|---|---|---|
| `cosine` (shallowest) | ✓ | 0.820 |
| `lookahead` | ✓ | 0.708 |

Cosine is the default for new SpecPrefill runs because it is both
faster *and* more accurate on this configuration. Multi-token
`keep=1.00` runs through cosine produce byte-equal text against dense.

## Quality

Phase A2 measured against the Qwen3.5-9B target with the 0.8B draft on
a 1354-token prompt:

| keep_ratio | argmax match | cossim | top-5 overlap |
|---|---|---|---|
| 0.10 | ✓ | 0.809 | 3/5 |
| 0.30 | ✓ | 0.684 | 3/5 |
| 0.50 | ✓ | 0.927 | 5/5 |
| 0.70 | ✓ | 0.948 | 4/5 |
| 0.90 | ✓ | 0.986 | 5/5 |

Argmax preservation is universal (greedy output's first token is
unchanged). Cossim is a regression backstop. Default `keep_ratio=0.50`
sits in the sweet spot.

For CUDA Qwen3.6-35B-A3B cross-family SpecPrefill, the sm86 logits gate
uses dense INT4 next-token logits as reference:

| lane | keep ratio | status | quality gate |
|---|---:|---|---|
| `int4-spec025` | 0.25 | exploratory | too aggressive on the standard prompt set |
| `int4-spec050` | 0.50 | balanced/perf | faster, but may drift argmax on some prompts |
| `int4-spec075` | 0.75 | conservative | current quality-preserving recommendation |

The runtime default for CUDA Qwen3.6 cross-family mode is therefore
`keep_ratio=0.75`; pass `--specprefill-keep-ratio 0.50` explicitly when
you want the faster balanced lane. `bench-perf` also accepts explicit sm86
Qwen3.6 exploratory lanes such as `int4-spec070`; see the May-06 sweep below
for why `0.70` was measured but not promoted.

## Reference docs

- Feasibility memo: [research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md).
- Phase A measurements (4B target): [research/2026-05-03-specprefill-phase-a-results.md](research/2026-05-03-specprefill-phase-a-results.md).
- Phase A2 measurements (9B target): [research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- CUDA Qwen3.6 keep-ratio sweep: [research/2026-05-06-qwen36-specprefill-keep-ratio-sweep.md](research/2026-05-06-qwen36-specprefill-keep-ratio-sweep.md).
- Original paper: [papers/SpecPrefill_arXiv_2502.02789.pdf](papers/SpecPrefill_arXiv_2502.02789.pdf).
