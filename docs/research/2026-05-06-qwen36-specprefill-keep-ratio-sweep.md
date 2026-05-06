# Qwen 3.6 CUDA SpecPrefill Keep-Ratio Sweep

**Date:** 2026-05-06
**Branch:** `codex/bench-arbitrary-specprefill-lanes`
**Hardware:** NVIDIA sm86 CUDA target
**Target:** `qwen3.6-35b-a3b` INT4
**Draft:** `qwen3.5-0.8b`

## Question

After sparse Qwen3.6 SpecPrefill moved onto the batched prefill path, the
`int4-spec075` lane became faster than dense instead of slower. We checked
whether an intermediate keep ratio could preserve the conservative quality
profile while reclaiming more TTFT than `keep=0.75`.

## Method

Quality used `oracle.bench.specprefill_quality` against the standard prompt
set, with dense INT4 next-token logits as the reference. Performance used a
single long synthetic prompt of about 800 target tokens with `max-new-tokens=1`
so the reported value is effectively target prefill wall time.

The benchmark harness now accepts explicit sm86 Qwen3.6 lanes of the form
`int4-specNNN` for ad hoc sweeps. The static matrix still lists only the
regular lanes (`025`, `050`, `075`), but `bench-perf --quants int4-spec070`
will run instead of being marked skipped.

## Results

Quality sweep:

| lane | keep ratio | argmax match | cosine min | top-5 min | notes |
|---|---:|---:|---:|---:|---|
| `int4-spec050` | 0.50 | 0.714 | 0.780312 | 2 | balanced/perf lane; fastest, but not argmax-preserving on the standard set |
| `int4-spec060` | 0.60 | 0.714 | 0.814386 | 3 | still misses argmax on two cases |
| `int4-spec065` | 0.65 | 0.857 | 0.829645 | 2 | improves argmax but still misses `json_completion` |
| `int4-spec070` | 0.70 | 1.000 | 0.881636 | 3 | argmax-preserving, but below the conservative cosine floor on two cases |
| `int4-spec075` | 0.75 | 1.000 | 0.911746 | 3 | conservative lane; passes current quality gate |

Performance on the about-800-token prompt after the sparse batched-prefill
optimization:

| lane | target prefill | speedup vs dense |
|---|---:|---:|
| dense INT4 | 22981.0 ms | 1.00x |
| `int4-spec050` | 13290.4 ms | 1.73x |
| `int4-spec070` | 18652.8 ms | 1.23x |
| `int4-spec075` | 19368.3 ms | 1.19x |

The direct `int4-spec070` runner measurement reported 579 kept tokens out of
800 prompt tokens (72.4%). Its target prefill was only about 3.7% faster than
`int4-spec075`, while weakening the cosine floor enough to fall below the
current conservative threshold.

## Recommendation

Keep the public CUDA Qwen3.6 cross-family default at `keep_ratio=0.75`. It is
now faster than dense on short, mid, and long prompts after the batched sparse
path, and it retains the argmax-preserving quality profile.

Keep `keep_ratio=0.50` as the explicit balanced/perf lane for users who accept
occasional first-token drift. Do not promote `0.70`: its speed gain over
`0.75` is too small for the quality regression.
