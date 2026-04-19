# Qwen3.5-4B CUDA `sm86` Optimization Notes

This document records the current `qwen3.5-4b` CUDA `sm86` hero lane, the
current warmed baselines captured on this machine, and the early experiments
that were tried and intentionally not kept.

It is the `4B` companion to [qwen35-sm86-optimization.md](/workspace/SuperSonic/docs/qwen35-sm86-optimization.md),
which remains the completed `0.8B` reference workflow.

## Production Throughput Lane

The current validated `4B` CUDA production-throughput target is:

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

## Hero Lane

The current `4B` CUDA `sm86` hero lane for single-stream optimization work is:

- backend: CUDA
- arch: `sm86`
- model: `qwen3.5-4b`
- dtype: BF16
- batch size: `1`
- mode: normal generation
- path: baked model load + `--force-kernel-decode`
- benchmark shape: warmed `pp533` / `tg128`
- hero attention guard: `B == 1 && bs == 256 && hd == 256`

Why this is the hero lane:

- `0.8B`-style hero work is fundamentally a single-stream latency problem
- the default single-sequence `4B` path on this box replays prefill for
  correctness, which is the wrong surface for latency optimization
- forcing the native `4B` kernel gives a direct staging lane that is much
  closer in spirit to the `0.8B` hero process
- the production-throughput CUDA lane for `4B` remains `--batch-size 2`, but
  that is not the right first surface for Lucebox-style single-stream work

## Single-Sequence Native Lane

This hero lane is measured with the warmed single-sequence native-kernel
harness below.

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

## Third Kept Single-Stream Pass

The next kept pass stayed inside the same packed BF16 single-stream attention
path and targeted the remaining reduction overhead:

- kept the single-block `B == 1` BF16 path introduced in the second kept pass
- left FP8-KV and batch-2 decode unchanged
- raised the packed BF16 inner loop from `2` dims per active thread to `4`
  dims per active thread
- hoisted the packed `q` loads out of the KV-token loop
- reduced the active attention waves in the score reduction from `4` to `2`
  while keeping the online softmax math unchanged

Why this pass was chosen:

- after `5915e54`, the packed BF16 path was still paying the old cross-wave
  reduction cost every token
- the remaining hotspot still looked like score reduction overhead more than
  projection or output work
- this was the smallest way to cut the reduction footprint again without
  changing the outer schedule or widening the optimization to other modes

Short screening result (`pp533` / `tg16`) against commit `5915e54`:

- decode improved from `1647.7 ms` to `1545.0 ms`
- persistent decode improved from `1555.383 ms` to `1453.126 ms`
- full attention improved from `670.579 ms` to `557.794 ms`
- full-attention core improved from `643.178 ms` to `529.680 ms`

Full warmed result (`pp533` / `tg128`) against commit `5915e54`:

- prefill moved from `4485.2 ms` to `4481.3 ms`
- decode improved from `13644.5 ms` to `12756.0 ms`
- native decode total improved from `13433.153 ms` to `12544.656 ms`
- persistent decode improved from `12910.028 ms` to `12021.549 ms`
- full attention improved from `5893.787 ms` to `4891.021 ms`
- full-attention core improved from `5673.589 ms` to `4672.824 ms`

Verification notes:

- `--validate --force-kernel-decode` passed with `decode_max_delta=0.2627`
- `--gpu-validate --force-kernel-decode` passed with matching tokens and
  `gpu_oracle_max_delta=0.3125`
- baked-path batch-2 validate still passed with `decode_max_delta=0.3047`

What this means now:

- the single-stream `4B` hero lane is still nowhere near the practical 3090
  ceiling, but it is now materially faster than the initial native-kernel lane
- `full_attn_core` remains the largest single subsection, but it is no longer
  overwhelmingly dominant the way it was at the start of the 4B single-stream
  work
- the next pass should explicitly separate score-reduction work from value
  accumulation / softmax rescaling before attempting another structural rewrite

## Fourth Kept Single-Stream Pass

The next kept pass stayed inside the same packed BF16 single-stream attention
path, but it stopped making the idle threads do useless score-side work:

- kept the single-block `B == 1` BF16 path introduced in the earlier kept
  passes
- left FP8-KV and batch-2 decode unchanged
- kept the existing `4` dims per active thread mapping and the same full-block
  synchronization points
- changed `q`-head BF16 staging so only the `64` active attention lanes write
  the staged shared-memory query buffer
- changed the per-token score path so only the active lanes execute the
  wave-level reduction and online-softmax scalar update, while the other
  threads only participate in the required block barriers

Why this pass was chosen:

- the one-warp subset-barrier experiment was faster but not safe enough to
  keep
- the earlier timing split still pointed at score-side work, not projection or
  output
- on the current `4B` hero branch, `192` threads were still spending cycles on
  zero-value shuffle/reduction and scalar softmax math every token
- this pass keeps the proven schedule and math ordering intact while trimming
  only work that does not contribute to the result

Short screening result (`pp533` / `tg16`) against commit `9cce7c2`:

- decode improved from `1544.3 ms` to `1539.0 ms`
- persistent decode improved from `1452.782 ms` to `1447.006 ms`
- full attention improved from `557.829 ms` to `548.709 ms`
- full-attention core improved from `529.602 ms` to `521.446 ms`

Full warmed result (`pp533` / `tg128`) against commit `9cce7c2`:

- prefill moved from `4481.3 ms` to `4496.5 ms`
- decode improved from `12756.0 ms` to `12685.8 ms`
- native decode total improved from `12544.656 ms` to `12475.365 ms`
- persistent decode improved from `12021.549 ms` to `11952.418 ms`
- full attention improved from `4891.021 ms` to `4817.504 ms`
- full-attention core improved from `4672.824 ms` to `4599.407 ms`

Verification notes:

- short-prompt baked `--validate --force-kernel-decode` matched the restored
  `9cce7c2` baseline token stream and deltas on this machine
- short-prompt baked `--gpu-validate --force-kernel-decode` also matched the
  restored baseline token stream and deltas
- baked batch-2 quick validate matched the restored baseline, and the batch
  path was not modified by this pass

What this means now:

- the remaining single-stream `4B` bottleneck is still the packed BF16
  attention core
- score-side work can still be reduced a little without changing the proven
  lane mapping or synchronization structure
- the next pass should still focus on the attention core, but it needs to buy
  more than this pass did to justify added complexity

## Linear-Core Internal Split

Before touching the next structural pass, the single-stream hero lane added a
temporary internal split inside the `4B` linear-attention core to measure where
`linear_core` time was actually going.

Short diagnostic run (`tg16`) on the current `6866040` baseline:

- total `linear_core`: `433.919 ms`
- conv front-end: `6.186 ms`
- recurrent update body: `423.504 ms`
- post/gating tail: `4.045 ms`

What that clarified:

- the next pass should not guess from the `0.8B` schedule copy attempts
- the measured `4B` hotspot was the recurrent-state walk, not conv or post
- a useful bounded pass should reduce recurrent-state traffic before trying
  another block-level remap

## Fifth Kept Single-Stream Pass

The next kept pass stayed inside the serial `4B` linear-attention block and
reduced how many times it walks recurrent state:

- left the single-sequence native-kernel hero lane unchanged:
  `CUDA + sm86 + qwen3.5-4b + BF16 + batch-size 1 + baked load + --force-kernel-decode`
- left batch-2 decode, FP8-KV, and the packed BF16 full-attention hero path
  untouched
- fused the linear-attention recurrent body from four state walks down to two:
  - fused decay with the `kv_mem` accumulation pass
  - fused state update with the final output accumulation pass
- kept the math order and output semantics unchanged

Why this pass was chosen:

- the temporary split showed the recurrent update body consuming essentially
  all of `linear_core`
- earlier attempts to copy the `0.8B` block/head mapping into `4B` linear
  attention were flat to worse on this machine
- reducing recurrent-state memory traffic is the smallest pass that matches the
  measured hotspot without broadening the optimization surface

Short screening result (`pp533` / `tg16`) against commit `6866040`:

- prefill moved from `4496.5 ms` to `4458.0 ms`
- decode improved from `1539.0 ms` to `1516.7 ms`
- native decode total improved from `1490.119 ms` to `1490.015 ms`
- persistent decode improved from `1447.006 ms` to `1424.597 ms`
- full attention stayed flat: `548.709 ms` to `548.653 ms`
- full-attention core stayed flat: `521.446 ms` to `521.454 ms`
- linear projection stayed flat: `140.272 ms` to `140.277 ms`
- linear core improved from `434.034 ms` to `409.137 ms`
- linear out stayed flat: `79.714 ms` to `79.690 ms`

Full warmed result (`pp533` / `tg128`) against commit `6866040`:

- prefill moved from `4496.5 ms` to `4482.9 ms`
- decode improved from `12685.8 ms` to `12500.6 ms`
- native decode total improved from `12475.365 ms` to `12288.288 ms`
- persistent decode improved from `11952.418 ms` to `11765.571 ms`
- full attention stayed flat: `4817.504 ms` to `4813.594 ms`
- full-attention core stayed flat: `4599.407 ms` to `4600.385 ms`
- linear projection stayed flat: `1122.219 ms` to `1121.687 ms`
- linear core improved from `3471.135 ms` to `3275.092 ms`
- linear out stayed flat: `637.595 ms` to `637.628 ms`

Verification notes:

- baked `--validate --force-kernel-decode` still passed with
  `decode_max_delta=0.7930` and matched the kept baseline token stream
- baked batch-2 validate still passed with `decode_max_delta=0.8086`
- `tests/sm86/run_4b.sh` baked path still passed; its `--no-bake` subtest still
  fails on this machine because `/workspace/models/Qwen3.5-4B` has no raw
  safetensors files
- `tests/sm86/run_batch.sh` baked path still passed; its `--no-bake` subtest
  fails for the same model-directory reason
- `tests/sm86/run_4b_long.sh` passed all `4/4` long-context corpus cases on
  this build

What this means now:

- the single-stream `4B` hero lane is no longer purely a full-attention story;
  `linear_core` is now the section that moved materially on the kept pass
- the right lesson from `0.8B` is still to copy the workflow, not the lane
  mapping
- the next bounded pass should keep targeting the no-batch hero lane first and
  should either split `linear_core` again on the final build or prove that a
  head-parallel recurrent remap beats the simpler fused serial path

## Sixth Kept Single-Stream Pass

The next kept pass stayed inside the same serial `4B` linear-attention body,
but cut redundant Q/K reloads from the recurrent loop:

- left the hero lane unchanged:
  `CUDA + sm86 + qwen3.5-4b + BF16 + batch-size 1 + baked load + --force-kernel-decode`
- left batch scheduling, FP8-KV, and the full-attention hero path untouched
- reused the front of `lds_input` as a small per-head staging buffer
- staged the normalized `q` and `k` vectors for each two-head recurrent
  iteration into shared memory once, then reused those shared values across all
  `128` value-lane threads for that head
- kept the recurrent math and writeback ordering unchanged

Why this pass was chosen:

- after the recurrent-pass fusion, `linear_core` was still a real bottleneck,
  but the remaining work was no longer dominated by recurrent-state traffic
  alone
- each thread in the recurrent body was still reloading the same normalized
  `q/k` vectors from global scratch on every pass over state
- staging those vectors once per head pair is the smallest way to reduce that
  redundant traffic without reopening the wider block/head scheduling question

Short screening result (`pp533` / `tg16`) against commit `44ab4bd`:

- prefill stayed flat: `4483.0 ms`
- decode improved from `1516.7 ms` to `1512.0 ms`
- native decode total improved from `1490.015 ms` to `1485.502 ms`
- persistent decode improved from `1424.597 ms` to `1420.061 ms`
- full attention stayed flat: `548.653 ms` to `548.610 ms`
- full-attention core stayed flat: `521.454 ms` to `521.495 ms`
- linear projection stayed flat: `140.277 ms` to `140.224 ms`
- linear core improved from `409.137 ms` to `404.261 ms`
- linear out stayed flat: `79.690 ms` to `79.682 ms`

Full warmed result (`pp533` / `tg128`) against commit `44ab4bd`:

- prefill improved from `4482.9 ms` to `4474.9 ms`
- decode improved from `12500.6 ms` to `12462.6 ms`
- native decode total improved from `12288.288 ms` to `12251.867 ms`
- persistent decode improved from `11765.571 ms` to `11729.533 ms`
- full attention stayed flat: `4813.594 ms` to `4813.412 ms`
- full-attention core stayed flat: `4600.385 ms` to `4600.225 ms`
- linear projection stayed flat: `1121.687 ms` to `1121.198 ms`
- linear core improved from `3275.092 ms` to `3235.868 ms`
- linear out stayed flat: `637.628 ms` to `637.556 ms`

Verification notes:

- baked `--validate --force-kernel-decode` still passed with
  `decode_max_delta=0.7930` and the same token stream as the kept baseline
- baked batch-2 validate still passed with `decode_max_delta=0.8086` and the
  same token stream as the kept baseline
- `tests/sm86/run_4b.sh` baked path still passed; its `--no-bake` subtest still
  fails on this machine because `/workspace/models/Qwen3.5-4B` has no raw
  safetensors files
- `tests/sm86/run_batch.sh` baked path still passed; its `--no-bake` subtest
  fails for the same model-directory reason
- `tests/sm86/run_4b_long.sh` again passed all `4/4` long-context corpus cases

What this means now:

- the current single-stream `4B` hero lane is still best treated as a
  no-batch latency lane first, with batch remaining a secondary surface
- the remaining easy wins inside the serial linear-attention body are getting
  smaller, which raises the bar for any further in-place tweaks
- the next bounded pass should likely compare this serial shared-staging path
  against a narrow head-parallel recurrent remap, not a broader `0.8B`-style
  schedule copy

## Attention-Core Bottleneck Read

Before changing the next structural pass, a temporary split was added inside
the `4B` single-stream hero attention core to separate score-side work from
value-side work. The instrumentation was measured and then reverted.

Measured on the hero lane with:

- `CUDA + sm86 + qwen3.5-4b + BF16 + batch-size 1 + baked load + --force-kernel-decode`
- prompt: warmed parity prompt, `tg16`

Two runs came in essentially identical:

- full-attention core total: about `45.8 ms`
- score-side work: about `32.7 ms`
- value-side work: about `5.0 ms`

What that clarified:

- the real remaining bottleneck inside the current single-stream hero
  attention core is score-side work, not value accumulation
- the next bounded pass should remove score-reduction overhead before trying a
  broader remap of the full attention body
- a narrow head-parallel recurrent remap for `linear_core` and a shared
  softmax-scalar experiment were both tested after this read and reverted,
  because they were flat to worse on the warmed lane

## Seventh Kept Single-Stream Pass

The next kept pass stayed inside the same single-stream `4B` full-attention
hero branch, but collapsed the score reduction down to one active wave:

- left the hero lane unchanged:
  `CUDA + sm86 + qwen3.5-4b + BF16 + batch-size 1 + baked load + --force-kernel-decode`
- left batch scheduling, FP8-KV, and the linear-attention recurrent path
  untouched
- changed the single-stream attention-core hero mapping from `64` active lanes
  x `4` dims to `32` active lanes x `8` dims
- kept the same BF16 math and output semantics, but removed the cross-wave
  score reduction and its inner-loop shared-memory synchronization from the
  hot per-token path
- widened the per-lane value accumulation to match the new `8`-dim lane shape

Why this pass was chosen:

- the temporary split showed score-side work dominating value-side work by
  roughly `6.6x`
- the existing hero branch was still paying for a two-wave score reduction on
  every token, even though the lane geometry could be collapsed into one wave
- this is a bounded structural pass in the exact hot section that measured as
  the bottleneck, without reopening batch behavior or the recurrent path

Short screening result (`pp533` / `tg16`) against commit `4197860`:

- prefill moved slightly and is not meaningful for this pass
- decode improved from `1512.0 ms` to `1469.3 ms`
- native decode total improved from `1485.502 ms` to `1443.121 ms`
- persistent decode improved from `1420.061 ms` to `1377.798 ms`
- full attention improved from `548.610 ms` to `499.054 ms`
- full-attention core improved from `521.495 ms` to `472.388 ms`
- linear projection stayed flat: `140.224 ms` to `140.532 ms`
- linear core stayed effectively flat: `404.261 ms` to `405.461 ms`
- linear out stayed flat: `79.682 ms` to `79.846 ms`

Full warmed result (`pp533` / `tg128`) against commit `4197860`:

- prefill moved from `4474.9 ms` to `4483.7 ms`
- decode improved from `12462.6 ms` to `12097.2 ms`
- native decode total improved from `12251.867 ms` to `11885.495 ms`
- persistent decode improved from `11729.533 ms` to `11362.940 ms`
- full attention improved from `4813.412 ms` to `4379.307 ms`
- full-attention core improved from `4600.225 ms` to `4165.966 ms`
- linear projection stayed flat: `1121.198 ms` to `1124.275 ms`
- linear core stayed effectively flat: `3235.868 ms` to `3243.006 ms`
- linear out stayed flat: `637.556 ms` to `638.729 ms`

Verification notes:

- baked `--validate --force-kernel-decode` still passed with the same token
  stream and `decode_max_delta=0.8359`
- a direct stage-timing spot check on the warmed hero lane moved in the same
  direction, with full-attention core dropping to about `38.8 ms` on `tg16`
- `tests/sm86/run_4b.sh` baked path still passed; its `--no-bake` subtest
  still fails on this machine because `/workspace/models/Qwen3.5-4B` has no
  raw safetensors files
- `tests/sm86/run_batch.sh` baked path still passed; its `--no-bake` subtest
  fails for the same model-directory reason
- `tests/sm86/run_4b_long.sh` passed all `4/4` long-context corpus cases on
  this build

What this means now:

- the measured single-stream `4B` bottleneck has moved back toward the
  attention core, but in a narrower form than the earlier full-attention
  hotspot
- the useful `0.8B` lesson remains to copy the workflow, not the exact lane
  mapping or packing assumptions
- the next bounded pass should instrument or target the remaining
  single-stream attention-core overhead carefully, because the broad
  cross-wave reduction tax is now gone

## Eighth Kept Single-Stream Pass

The next kept pass stayed off the hero attention branch and instead reduced
live projection state in the same single-stream native lane:

- left the hero lane unchanged:
  `CUDA + sm86 + qwen3.5-4b + BF16 + batch-size 1 + baked load + --force-kernel-decode`
- left the one-wave `32 x 8` attention-core hero mapping unchanged
- left FP8-KV, batch scheduling, and the linear-attention recurrent path
  untouched
- specialized the hot full-attention projection accumulation path for
  `B <= 2`
- replaced the generic `float p[MAX_BATCH_SIZE]` accumulator array in that hot
  path with scalar accumulators for the `B == 1` / `B == 2` cases, and kept
  the old generic path for larger batch counts

Why this pass was chosen:

- after the one-wave attention-core win, temporary timing splits showed the
  broad cross-wave score tax was already gone
- several follow-up attention-core experiments were correct but flat to worse,
  so the next bounded pass needed to reduce live state instead of changing the
  score/value math again
- static resource inspection on this machine is available through `cuobjdump`
  even though Nsight Compute counters are blocked by `ERR_NVGPUCTRPERM`
- the generic projection path was still carrying an unnecessary
  `MAX_BATCH_SIZE` accumulator array in the hot `B == 1` and `B == 2` cases

Static resource change:

- persistent kernel resource usage moved from `REG:170 STACK:128 SHARED:16`
  to `REG:164 STACK:128 SHARED:16`

Short screening result (`pp533` / `tg16`) against commit `a3c72eb`:

- prefill moved from `4429.7 ms` to `4454.7 ms`
- decode improved from `1469.3 ms` to `1468.7 ms`
- native decode total improved from `1443.121 ms` to `1442.123 ms`
- persistent decode improved from `1377.798 ms` to `1376.694 ms`
- full attention improved from `499.054 ms` to `497.901 ms`
- full-attention projection improved from `19.015 ms` to `8.959 ms`
- full-attention core regressed from `472.388 ms` to `480.818 ms`
- linear projection stayed flat: `140.532 ms` to `140.850 ms`
- linear core stayed flat-to-slightly-better: `405.461 ms` to `404.158 ms`

Full warmed result (`pp533` / `tg128`) against commit `a3c72eb`:

- prefill moved from `4483.7 ms` to `4484.7 ms`
- decode improved from `12097.2 ms` to `12095.4 ms`
- native decode total improved from `11885.495 ms` to `11884.882 ms`
- persistent decode improved from `11362.940 ms` to `11361.844 ms`
- full attention improved from `4379.307 ms` to `4377.807 ms`
- full-attention projection improved from `152.137 ms` to `71.659 ms`
- full-attention core regressed from `4165.966 ms` to `4240.442 ms`
- linear projection moved from `1124.275 ms` to `1126.750 ms`
- linear core improved from `3243.006 ms` to `3233.276 ms`
- linear out stayed flat: `638.729 ms` to `638.767 ms`

Why this one stayed:

- both the short screen and the full warmed screen stayed slightly positive on
  end-to-end decode
- the pass is numerically safe on the baked native-kernel lane
- the persistent kernel footprint dropped materially, which is useful headroom
  even though the wall-clock gain is small
- this is a bounded cleanup that does not widen the hero surface or lock in a
  speculative new attention schedule

Verification notes:

- baked `--validate --force-kernel-decode` still passed with the same token
  stream and `decode_max_delta=0.8359`
- `tests/sm86/run_4b.sh` baked path still passed; its `--no-bake` subtest
  still fails on this machine because `/workspace/models/Qwen3.5-4B` has no
  raw safetensors files
- `tests/sm86/run_batch.sh` baked path still passed; its `--no-bake` subtest
  fails for the same model-directory reason
- `tests/sm86/run_4b_long.sh` passed all `4/4` long-context corpus cases on
  this build

What this means now:

- the next real performance pass should still treat the single-stream `4B`
  lane as an attention-side problem first
- the score-reduction rewrite and this projection-state cleanup together make
  register pressure and live-range control a more credible next lever than
  another immediate math rewrite
- the next bounded experiment should target the remaining score/K-load side of
  the hero attention core or another provable live-range reduction inside that
  section

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
