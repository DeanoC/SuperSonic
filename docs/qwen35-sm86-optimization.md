# Qwen3.5 CUDA `sm86` Optimization Status

This document records the current optimization state of the CUDA `sm86`
Qwen3.5 decode paths, the benchmark targets, the commits that moved the
needle, and the experiments that were tried and discarded.

It is meant to be the handoff doc for applying the same process across the
supported CUDA Qwen3.5 models.

## Scope

Current hero lanes:

- backend: CUDA
- arch: `sm86`
- `qwen3.5-0.8b`: batch-1, BF16, normal generation
- `qwen3.5-4b`: batch-1, BF16, baked load, `--force-kernel-decode`

These lanes are intentionally specialized. Validation, tracing, replay-based
correctness, and other models still keep their fallback paths.

## Benchmark Target

External comparison point:

- Lucebox local run on this same RTX 3090-class `sm86` box:
  - `pp520`: about `8702 tok/s`
  - `tg129`: about `419 tok/s`

Internal parity harness:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen08.sh \
  /path/to/Qwen3.5-0.8B
```

Defaults:

- prompt target: about `pp520`
- generation: `tg128`
- warmup runs: `10`
- timed runs: `20`
- stage timings enabled

## Current Best

Best verified commit:

- `d929d78` `Fuse qwen0.8 hero attention setup barriers`

Current warmed result on this box:

- prefill: `552.2 tok/s`
- decode: `147.9 tok/s`
- persistent decode stage: `780.999 ms`
- `lm_head`: `81.048 ms`

What that means:

- versus Lucebox decode target `419 tok/s`, we are at about `35.3%`
- remaining decode gap is about `2.83x`
- versus the old legacy CUDA path (`18.7 tok/s`), current hero decode is about `7.9x` faster
- versus the first measured hero checkpoint (`21.4 tok/s`), current hero decode is about `6.9x` faster

## Progression

These are the commits that materially improved the hero lane.

| Commit | Change | Result |
| --- | --- | --- |
| `16b531e` | add the initial `qwen3.5-0.8b` `sm86` hero decode path | establishes the hero lane |
| `82add52` | first major persistent-kernel optimization pass | about `87.8 tok/s` |
| `c675b23` | tune hero launch shape and entrypoint plumbing | about `108.0 tok/s` |
| `d477e0a` | split linear recurrent work across stripes | about `109.1 tok/s` |
| `2afb464` | replace tree reductions with warp reductions | about `115.0 tok/s` |
| `7491d88` | vectorize the wide hero row-dot paths | about `115.3 tok/s` |
| `e881d08` | use BF16 shared caches for hero wide matvecs | about `116.4 tok/s` |
| `2df2897` | warpize hero projection families | about `133.8 tok/s` |
| `e9ea101` | pack the hero attention decode loop | about `147.4 tok/s` |
| `d929d78` | fuse hero attention setup barriers | about `147.9 tok/s` |

The biggest wins came from:

- making the hero lane truly model-specific instead of descriptor-generic
- moving wide matvecs and projection families to BF16 shared activations plus warp-per-row execution
- packing the full-attention decode loop so one active thread handles two head dimensions
- collapsing unnecessary grid barriers in the hero attention setup

## What Changed Architecturally

The current hero path differs from the older non-4B CUDA path in a few key ways:

- dedicated `qwen3.5-0.8b` `sm86` hero routing
- fused CUDA `lm_head + argmax` for normal generation
- BF16 shared activation caches on wide matvec families
- warp-per-row execution for hero projection families
- packed full-attention decode loop using BF16 pair lanes
- fewer grid barriers in the hero full-attention setup

The main files are:

- [kernels/full_attention_cuda.cuh](/workspace/SuperSonic/kernels/full_attention_cuda.cuh)
- [kernels/full_attention_bridge_cuda.cu](/workspace/SuperSonic/kernels/full_attention_bridge_cuda.cu)
- [kernels/prefill_helpers_bridge_cuda.cu](/workspace/SuperSonic/kernels/prefill_helpers_bridge_cuda.cu)
- [crates/runner/src/decode_engine.rs](/workspace/SuperSonic/crates/runner/src/decode_engine.rs)

## Things Tried And Not Kept

These experiments were run, debugged, and intentionally not kept.

- Extra full-attention block slicing per head.
  - `2` and `4` slices per head both regressed.
  - Reason: duplicated Q·K score work cost more than the added parallelism saved.

- A `4`-way DeltaNet recurrent split.
  - Stable but slower than the best committed path at the time.

- A dedicated `512`-thread hero launcher/body attempt.
  - Changed generated tokens on the smoke prompt and was reverted.

- Wider BF16 vectorization on the narrow `1024`-wide row-dot families.
  - Regressed; only the wider `o_proj` / `out_proj` / `down_proj` vectorization was kept.

- First-128-thread attention setup reductions.
  - Passed token regression but regressed the full warmed benchmark to about `144.9 tok/s`.

- DeltaNet subgroup reduction rewrite.
  - Passed token regression but came in at about `133.0-133.2 tok/s`, below the `133.8 tok/s` checkpoint.

- Packed KV append and gate save/apply cleanup on the attention path.
  - Correct but effectively flat on performance, so not worth carrying.

## Current Bottleneck Read

The main problem is still persistent decode, not sampling:

- host-side sampling and logits D2H are no longer the dominant issue
- `lm_head` is still a visible fixed cost at about `81 ms`, but it is not the primary blocker
- the remaining gap is still inside the persistent kernel body

After the latest attention-side work, the next likely sources of material gain are:

1. the linear-attention recurrent/update body
2. any remaining generic control-flow or barrier overhead in the hero path

The attention row-dot families and host path have already yielded most of the cheap wins.

## Recommended Process For Other CUDA-Supported Qwen3.5 Models

Use the same process we used for `qwen3.5-0.8b`.

1. Define one explicit hero lane.
   - Fix model, architecture, batch shape, and dtype.
   - Do not start from a generic “make everything faster” effort.

2. Add a warmed parity benchmark first.
   - Use a fixed prompt target, fixed generation length, warmups, and timed runs.
   - Keep stage timing output on every performance pass.

3. Route a specialized CUDA path behind strict gating.
   - Keep validation and debug paths on the old behavior if needed.

4. Remove host/logits waste first only if it is actually visible in stage timings.
   - For `0.8B`, that was necessary but not sufficient.

5. Move projection families to BF16 shared activations and warp-per-row execution.
   - This was one of the highest-yield structural changes.

6. Then attack the core persistent body.
   - On `0.8B`, the biggest later win came from packing the attention decode loop.

7. Keep only measured wins.
   - Token regression is required.
   - Full warmed benchmark is the gate.
   - If an idea is correct but flat or slightly worse, drop it immediately.

## Carry-Forward Notes

For the next CUDA-supported Qwen3.5 model, copy the workflow, not the exact kernel code.

Keep:

- hero-lane specialization
- warmed parity harness
- stage-timed benchmark loop
- BF16 shared activation strategy
- warp-per-row projection pattern
- strict token-regression gate before trusting benchmark deltas

Do not assume:

- the same head packing or lane mapping will be optimal
- the same failed experiments will become wins on a larger model
- a generic `512`-thread rewrite is automatically better

## 4B Hero Lane

Exact lane:

- CUDA
- `sm86`
- `qwen3.5-4b`
- BF16
- baked load
- `--force-kernel-decode`
- `--batch-size 1`
- warmed `pp533 / tg128`

Current best verified commit:

- `d8124c8` `cuda: parallelize qwen35 4b recurrent head pairs`

Current warmed result on this box:

- prefill: `119.0 tok/s`
- decode: `19.9 tok/s`
- decode wall time: `804.7 ms`
- persistent decode stage: `731.951 ms`
- `linear_core_recurrent`: `24.775 ms`

Validated behavior at this checkpoint:

- `tests/sm86/run_4b_long.sh`: `4/4` pass
- `tests/sm86/run_4b.sh`: baked path pass
- `tests/sm86/run_batch.sh`: baked path pass
- the local `--no-bake` legs still depend on raw safetensors existing under the
  model dir

Kept progression on the `4B` hero lane:

| Commit | Change | Result |
| --- | --- | --- |
| `db26c08` | parallelize the single-stream full-attention hero schedule across warps | established the first kept 4B attention-side win |
| `553c292` | trim recurrent-state traffic in the serial linear-attention core | reduced warmed `linear_core` and persistent time |
| `e5f244d` | add a true CUDA-only single-stream BF16 specialized 4B kernel entrypoint | improved warmed decode from about `14.5 tok/s` to `15.2 tok/s` |
| `d8124c8` | parallelize recurrent head-pair work across hero blocks | improved warmed decode to about `19.9 tok/s` and cut `linear_core_recurrent` to about `24.8 ms` |

What the latest kept pass changed structurally:

- moved the single-stream BF16 hero recurrent update from one block onto a
  head-pair-per-block schedule
- kept the generic block-0-only recurrent implementation as the fallback path
- disabled the decayless-store shortcut only in the cross-block recurrent hero
  split, where it was not numerically safe enough

Things tried on the `4B` hero lane and not kept:

- forcing higher block residency with register caps
- a noinline block-0 linear-core outline
- compile-time recurrent-loop rewrites that lowered register count but regressed
  wall-clock decode
- corrected single-stream BF16 MLP specializations that still regressed the
  short warmed lane

Current 4B bottleneck read:

- the dominant remaining cost is still inside the persistent kernel
- the recurrent bottleneck is no longer dominant after `d8124c8`
- the biggest remaining timing buckets are now `mlp_down`, `mlp_gate_up`,
  `full_attn_core`, `linear_proj`, and `linear_out`
- the next likely win is another hero-lane scheduling change on one of those
  wide row-dot families, not more surgery on the recurrent body

## Commands

Build:

```bash
cargo build --release --bin supersonic
```

Fast-greedy regression:

```bash
./tests/sm86/run_fast_greedy.sh /path/to/Qwen3.5-0.8B
```

Warmed parity benchmark:

```bash
COMPARE_LEGACY=0 ./tests/sm86/bench_qwen08.sh /path/to/Qwen3.5-0.8B
```
