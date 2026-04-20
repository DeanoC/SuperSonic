# Qwen3.5 CUDA `sm86` Optimization Status

This document records the current optimization state of the CUDA `sm86`
Qwen3.5 decode path, the benchmark target, the commits that moved the needle,
and the experiments that were tried and discarded.

It is meant to be the handoff doc for applying the same process to the other
CUDA-supported Qwen3.5 models.

## Scope

Current optimization work is focused on:

- backend: CUDA
- arch: `sm86`
- hero lane: `qwen3.5-0.8b`
- mode: batch-1, BF16, normal generation

The hero lane is intentionally specialized. Validation, tracing, and other
models still keep their fallback paths.

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
