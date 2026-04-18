# Qwen3.5-4B CUDA `sm86` Optimization Notes

This document records the current `qwen3.5-4b` CUDA `sm86` hero lane, the
first warmed baseline captured on this machine, and the early experiments that
were tried and intentionally not kept.

It is the `4B` companion to [qwen35-sm86-optimization.md](/workspace/SuperSonic/docs/qwen35-sm86-optimization.md),
which remains the completed `0.8B` reference workflow.

## Hero Lane

The current `4B` CUDA optimization target is:

- backend: CUDA
- arch: `sm86`
- model: `qwen3.5-4b`
- dtype: BF16
- batch size: `2`
- mode: normal generation
- path: baked model load
- benchmark shape: warmed `pp533` / `tg128`

Why this lane:

- single-sequence CUDA `4B` on this box still defaults to replayed GPU prefill
  for correctness, so it is not the right performance target
- the validated fast CUDA path for `4B` is batched decode
- existing `sm86` coverage already treats `qwen3.5-4b --batch-size 2` as the
  checked CUDA batch lane

## Benchmark Harness

Added in commit:

- `151516c` `Add warmed qwen4b sm86 batch benchmark`

Command:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_batch.sh \
  /path/to/Qwen3.5-4B
```

Defaults:

- target prompt: about `pp520` and calibrates to `pp533` on this box
- generation: `tg128`
- batch size: `2`
- warmup runs: `10`
- timed runs: `20`
- stage timings enabled

## Current Baseline

Current warmed result on this RTX 3090-class `sm86` machine:

- prompt tokens: `533`
- generated tokens: `128`
- aggregate batch decode tokens: `256`
- prefill: `4488.4 ms` = `118.7 tok/s`
- decode: `26178.0 ms` = `9.8 tok/s`

Stage timing mean:

- persistent decode: `25431.576 ms`
- `rms_norm`: `2.380 ms`
- `lm_head`: `615.619 ms`
- logits D2H: `20.712 ms`
- host sampling: `60.224 ms`
- total native decode: `26130.511 ms`

What that means:

- persistent decode is about `97.4%` of native decode time
- `lm_head` is visible but not the main blocker
- host-side work is not the first thing to optimize for this lane

## Internal Persistent Split

Added a `4B`-local persistent-section timing breakdown on the warmed batch hero
lane.

Short sanity run (`pp533` / `tg16`) came in at:

- decode: about `3283 ms`
- persistent decode: about `3169 ms`
- persistent sections:
  - full attention: about `1632 ms`
  - linear projection: about `188 ms`
  - linear core: about `867 ms`
  - linear out: about `158 ms`
  - MLP gate+up: about `266 ms`
  - MLP down: about `427 ms`

Full warmed result (`pp533` / `tg128`) came in at:

- decode: `27548.9 ms`
- aggregate decode throughput: `9.3 tok/s`
- persistent decode: `26640.164 ms`
- persistent sections:
  - full attention: `14336.164 ms`
  - linear projection: `1503.813 ms`
  - linear core: `6909.271 ms`
  - linear out: `1260.057 ms`
  - MLP gate+up: `2122.330 ms`
  - MLP down: `3396.552 ms`

How to read it:

- the per-section totals are measured inside the persistent kernel and are close
  enough to the wall-clock persistent total to guide optimization choice
- full attention is the dominant subsection on this machine and lane
- the next bounded pass should target full-attention internals before returning
  to MLP or linear attention

## Bottleneck Read

The real bottleneck is the batched persistent kernel body, not sampling.

From the current kernel structure in
[full_attention_4b_cuda.cuh](/workspace/SuperSonic/kernels/full_attention_4b_cuda.cuh):

- full-attention Q/K/V projection work already uses warp-level row reductions
- the `4B` linear-attention projection family still uses block-wide reductions
- the MLP projection family also still uses block-wide reductions
- several large sections still process `batch_size=2` via explicit `for (int b = 0; b < B; b++)`
  loops inside the persistent body
- with the new internal split, full attention is now the first place to look
  inside the persistent kernel on `4B`

That combination makes the next bounded target clear:

1. keep the hero lane fixed at `batch_size=2`
2. use the warmed harness plus stage timings as the gate
3. attack one numerically safe, high-volume kernel subsection at a time

The highest-probability next areas are:

- full-attention internal body
- linear-attention recurrent/core body
- MLP down projection

## Experiments Tried And Reverted

These were tested against the `4B` batch hero lane and intentionally not kept.

- Warpized BF16 `qkv/z/b/a` linear projection subpath for `batch_size=2`.
  - Short benchmark improved from about `1625 ms` to about `1557 ms` decode for
    `tg8`.
  - Full warmed benchmark improved from `27407.5 ms` to `26178.0 ms` decode for
    `tg128`.
  - Reverted because the baked batch golden corpus regressed on
    `medium_context`.

- Fused `counter reset + grid barrier` setup in the `4B` batch kernel.
  - Numerically safe.
  - Short benchmark came in effectively flat at about `1628 ms` vs the
    `1625 ms` baseline.
  - Reverted because it did not clear the measurement gate.

- `B == 2` MLP `down_proj` row streaming.
  - Idea: load each `down_proj` row once and accumulate both batch items in
    lockstep, instead of serializing the two batch items through the same row.
  - Short benchmark on `pp533/tg16` improved from:
    - decode `3265.7 ms` to `3084.0 ms`
    - persistent decode `3171.989 ms` to `2990.370 ms`
    - aggregate decode throughput `9.8 tok/s` to `10.4 tok/s`
  - Reverted because the baked batch golden corpus regressed on
    `medium_context`, producing the same token mismatch pattern seen in the
    earlier linear-projection warpization attempt.

- Persistent-kernel launch block-count sweep.
  - Added a temporary bridge override and swept the short hero lane at block
    counts `82`, `64`, `48`, and `32`.
  - Result was flat within noise:
    - `82`: decode `3265.5 ms`, persistent `3171.874 ms`
    - `64`: decode `3266.0 ms`, persistent `3171.789 ms`
    - `48`: decode `3265.0 ms`, persistent `3171.527 ms`
    - `32`: decode `3265.5 ms`, persistent `3171.387 ms`
  - Reverted because there was no meaningful exact-safe launch win to keep.

## Validation Notes

One local environment detail matters on this machine:

- `/workspace/models/Qwen3.5-4B` currently contains the baked `.supersonic`
  package plus config/tokenizer files, but no raw `safetensors`

That means:

- baked CUDA validation paths work
- the checked `--no-bake` leg in `tests/sm86/run_batch.sh` fails locally before
  decode starts, due to missing source weights rather than a kernel regression

The baked batch path and baked batch golden corpus remain the relevant
correctness gates for this optimization lane on this machine.
