# SpecPrefill

Speculator-driven sparse target prefill for long-prompt TTFT
optimization, based on [arXiv 2502.02789](https://arxiv.org/abs/2502.02789).
Currently shipping for Qwen3.5-9B target + Qwen3.5-0.8B draft on HIP
(gfx1100). Greedy decode only.

## When to use it

- Prompt is long (≥ 1k tokens) and TTFT (time to first token) matters.
- You're running greedy decode (`--max-new-tokens` ≥ 1, no top-p).
- Target is Qwen3.5-9B (the only target validated this phase).
- Backend is HIP. CUDA / Metal stubs return errors.

## When NOT to use it

- Sampling-based decode (top-p, temperature > 0). Top-5 stability is
  poor at low keep ratios — see Phase A2 measurements.
- Cross-family draft (e.g. Qwen3.5-0.8B → Qwen3.6-MoE). Deferred —
  see [research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- Very long prompts (>8192 tokens). The look-ahead kernel is bounded
  by per-block LDS; longer prompts trip a clear FFI error today.
  (Cosine scoring has no comparable bound, but the rest of the
  pipeline has not been measured beyond ~1500 tokens yet.)
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
| `--specprefill-keep-ratio <0.05..1.0>` | 0.50 | Fraction of prompt tokens kept by chunked top-K selection. |
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

Measured on gfx1100 with Qwen3.5-9B target + Qwen3.5-0.8B draft, a
1353-token prompt, 4 generated tokens, warmup pass + 3 measurement
runs (median reported), `--specprefill-keep-ratio 0.50`:

| Mode | TTFT (ms) | vs dense |
|---|---|---|
| dense (no SpecPrefill) | 4941 | 1.00× |
| `--specprefill-algorithm cosine`, `shallowest` (default) | 2385 | **2.07× faster** |
| `--specprefill-algorithm cosine`, `all_max` | 4144 | 1.19× faster |
| `--specprefill-algorithm lookahead` | 7846 | 0.63× (slower than dense) |

Quality at the same `keep_ratio=0.50` (against the dense reference; same
1353-token prompt; cossim is a regression backstop, argmax-match is
the primary correctness gate):

| Algorithm | argmax match | cossim |
|---|---|---|
| `cosine` (shallowest) | ✓ | 0.820 |
| `lookahead` | ✓ | 0.708 |

Cosine is the default for new SpecPrefill runs because it is both
faster *and* more accurate on this configuration. The single-token
`keep_ratio=0.50` measurement is end-to-end; multi-token keep=1.00 runs
through cosine produce byte-equal text against dense.

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

## Reference docs

- Feasibility memo: [research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md).
- Phase A measurements (4B target): [research/2026-05-03-specprefill-phase-a-results.md](research/2026-05-03-specprefill-phase-a-results.md).
- Phase A2 measurements (9B target): [research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- Original paper: [papers/SpecPrefill_arXiv_2502.02789.pdf](papers/SpecPrefill_arXiv_2502.02789.pdf).
