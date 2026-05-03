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
- Cross-family draft (e.g. Qwen3.5-0.8B → Qwen3.6-MoE). Deferred to
  Phase D — see
  [research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- Very long prompts (>8192 tokens). The look-ahead kernel is bounded
  by per-block LDS; longer prompts trip a clear FFI error today.
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
| `--specprefill-chunk-size <int>` | 32 | Selection chunk size (paper §3.4). |
| `--specprefill-pool-window <odd int>` | 5 | 1-D smoothing window for importance scores. |
| `--specprefill-lookahead <1..16>` | 4 | Number of look-ahead decode steps on the draft. |
| `--specprefill-always-keep-prefix <int>` | 4 | Force-keep first N tokens (BOS / system). |
| `--specprefill-always-keep-suffix <int ≥ 1>` | 4 | Force-keep last N tokens. Must be ≥ 1 (the first decode logits come from this slot). |
| `--specprefill-unload-draft` | false | Free the draft weights between selection and target prefill (claws back ~1.6 GiB). |

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
