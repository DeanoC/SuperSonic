# Qwen3.5-4B CUDA `sm86` Optimization Notes

This document records the current `qwen3.5-4b` CUDA `sm86` hero lane, the
current warmed baselines captured on this machine, and the early experiments
that were tried and intentionally not kept.

It is the `4B` companion to [qwen35-sm86-optimization.md](/workspace/SuperSonic/docs/qwen35-sm86-optimization.md),
which remains the completed `0.8B` reference workflow.

## Throughput Lane

The current validated `4B` CUDA throughput target is:

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

## Single-Sequence Native Lane

To stage Lucebox-style single-stream work, this branch now also defines a
measured single-sequence native-kernel lane:

- backend: CUDA
- arch: `sm86`
- model: `qwen3.5-4b`
- dtype: BF16
- batch size: `1`
- mode: normal generation
- path: baked model load + `--force-kernel-decode`
- benchmark shape: warmed `pp533` / `tg128`

Why this lane exists:

- `0.8B`-style hero work is fundamentally a single-stream latency problem
- the default single-sequence `4B` path on this box replays prefill for
  correctness, which is the wrong surface for latency optimization
- forcing the native `4B` kernel gives a direct staging lane that is much
  closer in spirit to the `0.8B` hero process

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

Single-sequence native-kernel command:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_single.sh \
  /path/to/Qwen3.5-4B
```

Single-sequence harness defaults:

- target prompt: about `pp520` and calibrates to `pp533` on this box
- generation: `tg128`
- batch size: `1`
- forces native kernel decode with `--force-kernel-decode`
- warmup runs: `10`
- timed runs: `20`
- stage timings enabled

## Initial Baseline

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

## Single-Sequence Native Baseline

Committed baseline for the forced single-sequence native-kernel lane on this
RTX 3090-class `sm86` machine (`29450cc`):

- prompt tokens: `533`
- generated tokens: `128`
- prefill: `4470.4 ms` = `119.2 tok/s`
- decode: `14780.5 ms` = `8.7 tok/s`

Stage timing mean:

- persistent decode: `14045.437 ms`
- `rms_norm`: `1.854 ms`
- `lm_head`: `510.347 ms`
- logits D2H: `10.094 ms`
- host sampling: `0.002 ms`
- total native decode: `14567.735 ms`

Persistent subsection mean:

- full attention: `7197.903 ms`
- full-attention projection: `152.364 ms`
- full-attention core: `6984.718 ms`
- full-attention `o_proj`: `56.676 ms`
- linear projection: `1119.540 ms`
- linear core: `3481.079 ms`
- linear out: `634.208 ms`
- MLP gate+up: `1455.487 ms`
- MLP down: `1708.718 ms`

What that means:

- the native single-sequence `4B` path is now directly measurable instead of
  being hidden behind replayed prefill
- full-attention core is still the main bottleneck in the single-stream lane,
  just as it is in the batched lane
- the single-stream baseline is now close enough in shape to the `0.8B`
  workflow to justify using it as the next hero staging surface

## Early Single-Stream Experiments Not Kept

The first speculative `B == 1` packed-BF16 attention-core branch, copied in the
spirit of the `0.8B` hero path, was tested and reverted.

Short screening result (`pp533` / `tg16`):

- baseline decode: `1776.0 ms`
- experiment decode: `1777.3 ms`
- baseline full-attention core: `791.650 ms`
- experiment full-attention core: `792.208 ms`

Why it was dropped:

- the packed two-float inner loop was correct but flat to slightly worse
- it did not change the actual bottleneck enough to justify more validation
- this reinforced that `4B` should keep following measured bottlenecks, not
  copy `0.8B` structure blindly

## First Kept Single-Stream Pass

The first kept `4B` single-stream pass was narrower than the failed hero-copy
experiment:

- left the attention math unchanged
- kept `saved_gate`, which the decode path still needs after attention
- made `saved_q`, `saved_pre_gate`, and `saved_scores` opt-in instead of
  writing them unconditionally on every decode step
- enabled those scratch writes only for the persistent full-attention trace
  path used by the debug tooling in `main.rs`

Why this was the right next pass:

- those trace buffers are read by the trace/validation workflow, not by the
  normal decode or stage-timing lane
- the previous `4B` implementation was paying for trace-only global writes in
  the hottest part of single-stream attention
- this is exactly the kind of bounded, measurable cleanup worth doing before a
  larger structural kernel rewrite

Short screening result (`pp533` / `tg16`):

- decode improved from `1776.0 ms` to `1770.0 ms`
- persistent decode improved from `1683.655 ms` to `1678.152 ms`
- full attention improved from `818.310 ms` to `812.547 ms`
- full-attention core improved from `791.650 ms` to `785.903 ms`

Full warmed result (`pp533` / `tg128`):

- prefill moved from `4470.4 ms` to `4482.7 ms`
- decode improved from `14780.5 ms` to `14733.1 ms`
- native decode total improved from `14567.735 ms` to `14520.915 ms`
- persistent decode improved from `14045.437 ms` to `13998.467 ms`
- full attention improved from `7197.903 ms` to `7147.420 ms`
- full-attention core improved from `6984.718 ms` to `6934.247 ms`

Verification notes:

- `--validate --force-kernel-decode` passed with `decode_max_delta=0.3047`
- `--gpu-validate --force-kernel-decode` passed with matching tokens and
  `gpu_oracle_max_delta=0.3125`
- baked-path batch-2 validate still passed with `decode_max_delta=0.3047`
- `tests/sm86/run_4b.sh` baked path still passed; its `--no-bake` subtest still
  fails on this machine because `/workspace/models/Qwen3.5-4B` has no raw
  safetensors files
- `tests/sm86/run_4b_long.sh` still fails the `medium_context` golden case, but
  that same failure reproduces on baseline commit `29450cc`, so it is not
  introduced by this pass

What this changes about the next step:

- the single-stream hero lane still points at full-attention core as the main
  structural bottleneck
- the next pass should continue to target the single native-kernel lane first
- the next structural experiment should avoid reintroducing trace or debug
  work into the hot path

## Second Kept Single-Stream Pass

The next kept pass targeted the actual BF16 attention-core math in the
single-stream lane rather than the surrounding schedule:

- left the single-block `B == 1` decode schedule in place
- left FP8-KV and batched decode untouched
- staged each `q` head into shared memory as BF16
- switched the BF16 attention inner loop from scalar per-dimension loads to
  packed two-dimension BF16 loads for `q`, `k`, and `v`
- kept the existing online softmax structure and output semantics

Why this pass was chosen:

- after the trace-gating cleanup, `full_attn_core` was still the dominant
  single-stream hotspot by a large margin
- the first failed `B == 1` hero-copy experiment changed scheduling and math at
  the same time, which made it a poor template
- this pass isolates the likely useful part of the `0.8B` idea: packed BF16
  score/value work inside the existing `4B` single-stream control flow

Short screening result (`pp533` / `tg16`) against commit `d3a7ab3`:

- decode improved from `1770.0 ms` to `1647.7 ms`
- persistent decode improved from `1678.152 ms` to `1555.383 ms`
- full attention improved from `812.547 ms` to `670.579 ms`
- full-attention core improved from `785.903 ms` to `643.178 ms`

Full warmed result (`pp533` / `tg128`) against commit `d3a7ab3`:

- prefill moved from `4482.7 ms` to `4485.2 ms`
- decode improved from `14733.1 ms` to `13644.5 ms`
- native decode total improved from `14520.915 ms` to `13433.153 ms`
- persistent decode improved from `13998.467 ms` to `12910.028 ms`
- full attention improved from `7147.420 ms` to `5893.787 ms`
- full-attention core improved from `6934.247 ms` to `5673.589 ms`

Verification notes:

- `--validate --force-kernel-decode` passed with `decode_max_delta=0.2734`
- `--gpu-validate --force-kernel-decode` passed with matching tokens and
  `gpu_oracle_max_delta=0.3125`
- baked-path batch-2 validate still passed with `decode_max_delta=0.3047`

What this means now:

- the single-stream `4B` lane finally has a meaningful structural win, not
  just instrumentation cleanup
- the remaining `full_attn_core` cost is now much smaller in absolute terms,
  but it is still the largest single subsection
- the next bounded pass should break down whether the remaining cost is mostly
  in score reduction or in value accumulation / softmax rescaling

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

## Full-Attention Internal Split

Added one more level of timing inside the `4B` full-attention section so the
next optimization pass would not guess between projection, the attention core,
and `o_proj`.

Short sanity run (`pp533` / `tg16`) came in at:

- decode: `3286.7 ms`
- persistent decode: `3173.691 ms`
- full attention total: `1632.724 ms`
- full-attention subsections:
  - projection: `21.540 ms`
  - core: `1596.197 ms`
  - `o_proj`: `14.185 ms`

Full warmed result (`pp533` / `tg128`) came in at:

- decode: `27454.2 ms`
- aggregate decode throughput: `9.3 tok/s`
- persistent decode: `26551.106 ms`
- full attention total: `14375.092 ms`
- full-attention subsections:
  - projection: `172.364 ms`
  - core: `14082.755 ms`
  - `o_proj`: `113.577 ms`

What that clarified:

- the real `4B` full-attention bottleneck is the BF16 attention core, not the
  matmul work around it
- copying the `0.8B` projection or `o_proj` hero ideas first would not have
  matched the measured hotspot on this lane
- the first structural pass should target the per-head attention body itself

## First Kept Pass

Added a narrow `sm86` batch-2 full-attention core specialization for the baked
BF16 hero lane:

- fixed lane guard: `B == 2`, `bs == 256`, `hd == 256`, `nh == 8`, `nkv == 2`
- left `q_proj/k_proj/v_proj` and `o_proj` unchanged
- replaced the sequential block-0 full-attention core with head-parallel work
  across blocks for:
  - Q norm + RoPE + saved-gate staging
  - K norm + RoPE + KV append
  - attention + gate
- left `kv_fp8` on the old path; this pass is explicitly for the current BF16
  hero lane and does not widen to other modes yet

Short sanity result (`pp533` / `tg16`):

- decode improved from `3286.7 ms` to `3275.0 ms`
- persistent decode improved from `3173.691 ms` to `3161.851 ms`
- full-attention core improved from `1596.197 ms` to `1581.192 ms`

Full warmed result (`pp533` / `tg128`):

- decode improved from `27454.2 ms` to `27347.7 ms`
- aggregate decode throughput moved from `9.3 tok/s` to `9.4 tok/s`
- persistent decode improved from `26551.106 ms` to `26444.424 ms`
- full attention improved from `14375.092 ms` to `14241.680 ms`
- full-attention core improved from `14082.755 ms` to `13950.210 ms`

Why this one stayed:

- it is numerically safe on the baked CUDA path
- the warmed measurement moved in the expected direction, in the exact
  subsection it was meant to target
- the gain is modest, but it cleared the “measured win” bar without regressing
  correctness

## Bottleneck Read

The real bottleneck is the batched persistent kernel body, not sampling.

From the current kernel structure in
[full_attention_4b_cuda.cuh](/workspace/SuperSonic/kernels/full_attention_4b_cuda.cuh):

- full-attention Q/K/V projection work already uses warp-level row reductions
- the `4B` linear-attention projection family still uses block-wide reductions
- the MLP projection family also still uses block-wide reductions
- several large sections still process `batch_size=2` via explicit `for (int b = 0; b < B; b++)`
  loops inside the persistent body
- with the new internal split, full-attention core is now the first place to
  look inside the persistent kernel on `4B`

That combination makes the next bounded target clear:

1. keep the hero lane fixed at `batch_size=2`
2. use the warmed harness plus stage timings as the gate
3. attack one numerically safe, high-volume kernel subsection at a time

The highest-probability next areas are:

- full-attention core inner loop
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

Additional baked-path validation for the kept full-attention-core pass:

- single-sequence baked validate passed with `decode_max_delta=0.2812`
- single-sequence baked `--gpu-validate` passed with exact token match and
  `gpu_oracle_max_delta=0.0000`
- baked batch quick validate passed with `decode_max_delta=0.3047`
- baked batch golden corpus passed `11 / 11`
- single-sequence long-context golden corpus passed `4 / 4`

Additional validation for the single-sequence native-kernel staging lane:

- baked `--validate --force-kernel-decode` passed with `decode_max_delta=0.3047`
- baked `--gpu-validate --force-kernel-decode` passed with token agreement and
  `gpu_oracle_max_delta=0.3125`
