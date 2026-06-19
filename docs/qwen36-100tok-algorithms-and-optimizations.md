# Qwen3.6 100 Tok/s Algorithms And Optimizations

Last updated: 2026-06-19

This note explains how SuperSonic reached `100 tok/s` mean throughput on the
Lucebox-style Qwen3.6-27B DFlash benchmark on RX 7900 XTX / `gfx1100`.
It is the readable companion to the raw working log in
`docs/qwen36-lucebox-parity-log.md`.

The headline result is:

- Benchmark: Lucebox HumanEval 10-prompt serving-mode suite, `n_gen=256`.
- Target: Qwen3.6-27B Q4_K_M.
- Draft: Q8 DFlash draft, 5 layers, block size 16.
- Hardware: RX 7900 XTX, `gfx1100`, HIP.
- Best artifact:
  `target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256.json`.
- Mean throughput: `100.86 tok/s`.
- Weighted throughput: `99.40 tok/s`.
- Repeat: `100.79 tok/s` mean, `99.38 tok/s` weighted.
- Generated tokens: `1654`, with all 10 prompts stopping early.
- Output stability: per-prompt generated-token counts stayed
  `[179,232,99,159,174,154,216,114,165,162]`, and the combined output hash
  stayed `032209e65467e8aa6c74025dc8b70b325f0ec767054ddff614e04550bb11f3bf`.

The careful wording matters: the explicit objective was `100 tok/s` mean across
the 10 prompts. Weighted throughput is still just below `100 tok/s`, so a future
phase can reasonably target weighted >= `100 tok/s`.

## Algorithm Shape

The final run is DFlash speculative decoding in Lucebox serving mode. It uses
the prefill-append verifier path, not the optional DDTree target verifier.
Several DDTree/tree-verify optimizations were important stepping stones and
share kernels with this path, but the last measured 100 tok/s artifacts say:

```text
[dflash] using prefill-append target verifier
```

One decode round works like this:

1. The target model has already prefetched the prompt and captured DFlash tap
   hidden states at selected target layers.
2. The DFlash draft consumes those target taps plus a 16-token noise block and
   produces draft hidden states for a candidate block.
3. The target lm-head scores the draft hidden rows and gives a proposed chain of
   candidate token IDs.
4. The target verifies that whole candidate chain as an append-style prefill.
   This is cheaper than decoding one token at a time because the target does
   one batched verifier pass over the 16 rows.
5. SuperSonic computes target greedy IDs for the verified rows and accepts the
   longest matching prefix.
6. Accepted state is committed, rejected state is rolled back, and the next
   round starts from the new committed sequence length.

Throughput is therefore controlled by two quantities:

- how many target tokens are accepted per round, and
- how much verifier work is required per round.

For the final `he_01` run, SuperSonic generated 179 tokens in 21 rounds, with
mean accepted-per-round around `8.48`. The speed work here mostly reduced the
cost of a verifier round while preserving the same acceptance shape and output.

## Baseline

The fresh post-PR #264 baseline for this continuation was:

| Artifact | Mean tok/s | Weighted tok/s | Min tok/s | Generated |
| --- | ---: | ---: | ---: | ---: |
| `post264_main_10x256.json` | 86.54 | 85.44 | 76.10 | 1654 |

The profile showed the next material target was not generic host overhead. The
hot area was the small-M verifier kernel train:

- Q4_K/Q5_K/Q6_K small-M matmuls at `m=16`.
- Q6_K MLP-down projection.
- recurrent linear-attention verify.
- full-attention verify.
- lm-head plus argmax paths.

Rollback, allocations, and helper cleanups were already too small to close a
15% mean-throughput gap by themselves.

## Optimization Ladder

The improvement came from several measured, exactness-preserving changes. No
single early cleanup was enough; the 100 tok/s result is the composition of
multiple small-to-medium wins, plus one final append recurrent kernel win.

| Step | Artifact | Mean tok/s | Weighted tok/s | What changed |
| --- | --- | ---: | ---: | --- |
| Baseline | `post264_main_10x256.json` | 86.54 | 85.44 | Post-PR #264 verifier baseline |
| Q6 hot exact | `q6_hot_exact_10x256.json` | 92.60 | 91.38 | Specialized Q6_K MMQ MLP-down hot shape |
| Tree recurrent warp32 | `tree_recurrent_warp32_default_10x256.json` | 95.63 | 94.36 | Wave32 recurrent tree attention |
| m16 launch bounds | `m16_qtype_launchbounds4_10x256.json` | 96.40 | 94.87 | Generic m16 qtype launch-bounds tuning |
| Target lm-head argmax | `q6_lm_head_argmax_range_fused_10x256.json` | 96.89 | 95.51 | Target verifier returns greedy IDs directly |
| Target+draft argmax | `q6_lm_head_argmax_target_draft_fused_10x256.json` | 96.94 | 95.57 | Draft generation also avoids full lm-head logits |
| Append O residual | `append_attn_residual_fused_10x256.json` | 97.12 | 95.75 | Append attention O projections use residual epilogue |
| Append recurrent direct | `append_recurrent_warp32_direct_10x256.json` | 100.86 | 99.40 | Append recurrent Q8 warp32 direct attention |

The table hides some kept smaller steps that were necessary foundations:

- fixed-qtype `m=16` GGML gate/up pair dispatch,
- fused gate/up + SwiGLU for the verifier MLP path,
- Q4_K and Q6_K residual-add epilogues,
- tree residual-add projection helpers,
- rollback gates and parity tests around every risky kernel path.

Those early steps moved the code to a shape where the later hot exact kernels
had enough impact to matter at suite scale.

## Key Algorithmic Optimizations

### 1. Fuse verifier MLP gate/up + SwiGLU

The verifier repeatedly evaluates MLP rows with `m=16`. The old path did:

```text
gate matmul -> up matmul -> BF16 intermediate -> SwiGLU launch
```

The kept path uses a fixed-qtype `m=16` pair kernel and then fuses the SwiGLU
epilogue:

```text
paired gate/up matmul + BF16-exact SwiGLU -> down input
```

The important correctness detail is that the fused kernel rounds the gate and
up accumulators to BF16 before applying `silu(gate) * up`, matching the old
observable order. This kept output hashes stable while removing intermediate
traffic and launches.

This was the first material win in the 100 tok/s phase:

- same-build full-suite A/B: `88.52` mean vs `86.17` mean with the rollback
  gate enabled.
- the output hash and token counts matched.

### 2. Specialize the Q6_K MMQ MLP-down hot shape

After gate/up+SwiGLU, the Q6_K MLP-down row became the dominant remaining MLP
cost. The hot shape was stable:

```text
m=16, n=5120, k=17408
```

The optimized Q6_K MMQ path:

- detects the exact hot shape,
- skips partial-row/partial-column checks that cannot fire for this shape,
- fixes redundant Q6 scale loading,
- keeps the same Q8 activation quantization and Q6_K tile math,
- preserves the residual-add BF16 rounding order.

This moved the suite from the high 80s to the low 90s:

- `q6_hot_exact_10x256.json`: `92.60` mean / `91.38` weighted.
- rollback-gated same-binary run: `88.35` mean / `87.17` weighted.

This was the point where the remaining gap stopped looking like generic MLP
cleanup and started looking like recurrent attention plus smaller projection
tail costs.

### 3. Use wave32 recurrent tree attention

The old recurrent tree attention used a 128-thread block and block-wide
reductions over K. On RDNA3 wave32 hardware that meant crossing four waves via
LDS and synchronization for reductions that are naturally `K=128`.

The kept wave32 design maps one wave to the recurrent reduction:

- one lane owns four K positions,
- it keeps four state values in registers,
- wave reductions compute the four partial sums,
- the final order is `(p0 + p2) + (p1 + p3)` to stay close to the existing
  block-reduction order,
- it writes BF16 attention rows directly,
- it still writes the exact Q8 rollback trace.

This is deliberately different from the rejected scalar-broadcast attempts:
those added synchronization without changing the core reduction structure. The
kept warp32 kernel removed cross-wave reduction cost.

Measured effect:

- recurrent FFI row fell from about `207 ms` to about `137 ms` on the profiled
  prompt.
- full-suite mean moved from `92.60` to `95.63 tok/s`.

### 4. Tune generic m16 qtype launch bounds

Once the specialized Q6 path and recurrent path improved, generic m16 qtype
matmuls were still frequent enough to matter. The kept tuning changes the
generic qtype WMMA kernel launch bounds from `(32, 8)` to `(32, 4)`.

The effect is not algorithmic in the mathematical sense; it gives the compiler
more room around register pressure for the small-M qtype kernels. The important
lesson was to apply it only to the generic m16 qtype path. Similar launch-bound
probes on the fused gate/up+SwiGLU and Q6_K MMQ paths were flat or worse and
were reverted.

Measured effect:

- `tree_recurrent_warp32_default_10x256.json`: `95.63` mean.
- `m16_qtype_launchbounds4_10x256.json`: `96.40` mean.

### 5. Fuse Q6_K lm-head argmax for target and draft

The verifier and draft path both need greedy token IDs. They do not need to
materialize a full `[16, vocab]` BF16 logits matrix just to immediately scan it.

The fused Q6_K m16 argmax path:

- computes BF16-rounded tile logits,
- reduces each tile to a row winner,
- reduces tile winners to token IDs,
- returns only IDs for the greedy path.

This first landed for target verifier greedy rows, then for the draft candidate
generator. The draft side keeps the old full-logits path when DDTree top-k
probing is active, because top-k probing genuinely needs more than argmax.

Measured effect:

- target verifier fused argmax: `96.89` mean.
- target + draft fused argmax: `96.94` mean.
- FFI shape profile showed the old full `m=16,n=248320` lm-head row removed in
  no-DDTree serving mode.

The suite-level gain was small, but it removed a structurally wasteful path.

### 6. Extend residual-add epilogues to append attention projections

Earlier residual-add projection helpers were used in tree verifier paths. The
final serving benchmark used the append verifier, so append full-attention and
append linear-attention O projections still had:

```text
projection matmul -> element_add
```

Extending the existing residual epilogue to append O projections removed the
separate `element_add` row from the hot FFI shape profile. The raw projection
row became slightly more expensive, but the launch and memory traffic saved by
removing `element_add` won at suite scale.

Measured effect:

- `q6_lm_head_argmax_target_draft_fused_10x256.json`: `96.94` mean.
- `append_attn_residual_fused_10x256.json`: `97.12` mean.

### 7. Add append recurrent warp32 direct attention

This was the final jump from `97.12` to `100.86` mean tok/s.

The append linear-attention recurrent path was still doing:

```text
delta_recurrent_prefill_capture_q8_trace
-> dflash_extract_recurrent_attn
```

The optimized path mirrors the accepted tree warp32 design but for append
verification:

- one wave per recurrent reduction,
- four K positions per lane,
- BF16 attention output written directly,
- persistent F32 recurrent state updated directly,
- exact Q8 rollback trace still written,
- separate recurrent-attention extraction pass removed.

Correctness was gated by an ignored GPU parity test that compares:

- BF16 attention bytes,
- Q8 trace bytes,
- final recurrent state,

against the old Q8 capture plus extract path.

Measured effect:

- old Phase 2U FFI profile had
  `delta_recurrent_prefill_capture_q8_trace` at 1008 calls / `199.52 ms` and
  `dflash_extract_recurrent_attn` at 1008 calls / `31.39 ms`.
- new profile had
  `delta_recurrent_prefill_capture_q8_trace_attn` at 1008 calls / `132.42 ms`,
  with both old rows gone.
- suite mean moved from `97.12` to `100.86 tok/s`.
- repeat suite mean was `100.79 tok/s`.

## Why This Worked

The core pattern was not "make one kernel heroic". It was:

1. Profile the actual serving path, not a stale benchmark mode.
2. Attack the highest repeated `m=16` verifier rows.
3. Preserve exact visible math by matching BF16 rounding points.
4. Avoid full intermediate materialization when only IDs or residual-added rows
   are needed.
5. Match RDNA3's wave32 execution model for recurrent reductions.
6. Keep rollback gates so each optimization can be A/B tested in the same
   binary.
7. Promote only changes that improve the full 10-prompt suite without changing
   generated-token counts or output hashes.

The last point is the most important engineering guardrail. Several ideas
looked good in a one-prompt smoke or microbench and were still rejected because
the full suite was flat or worse:

- DDTree budget/top-k/no-chain changes after the kernel wins,
- fast SwiGLU sigmoid,
- output-branch removal in the fused SwiGLU kernel,
- Q6_K MMQ Y64 tiling,
- hard-sync removal in full tree attention,
- scalar broadcast variants for recurrent attention,
- old generic dense gate/up pair helper,
- launch-bounds probes on the wrong kernels.

## Code Map

Main algorithm loop:

- `crates/runner/src/qwen35_dflash_engine.rs`
  - DFlash round loop, append verifier selection, optional DDTree controls,
    draft candidate generation.

Verifier/runtime implementation:

- `crates/runner/src/prefill_engine.rs`
  - prefill-append verifier, tree verifier, greedy ID paths, residual projection
    routing, cache-owned scratch.

FFI and dispatch:

- `crates/kernel-ffi/src/prefill_ffi.rs`
  - append/tree recurrent dispatch, Q6_K fused argmax dispatch, profiling rows.

HIP kernels:

- `kernels/full_attention.hip`
  - recurrent Q8 trace kernels and the append/tree wave32 direct-attention
    kernels.
- `kernels/full_attention_4b.hip`
  - Q4/Q5/Q6 small-M qtype kernels, fused gate/up+SwiGLU, Q6_K MMQ hot exact,
    fused lm-head argmax.
- `kernels/full_attention_bridge.cpp`
  - full-attention and recurrent HIP launch routing.
- `kernels/full_attention_bridge_4b.cpp`
  - low-bit matmul, residual epilogue, and argmax launch routing.

Benchmark and profiling history:

- `docs/qwen36-lucebox-parity-log.md`
  - authoritative run log, artifacts, keep/reject decisions.
- `docs/qwen36-lucebox-next-roofline.md`
  - roofline/profiling setup and earlier PR #264 performance baseline.

## Validation Summary

The final kept state was validated by:

- `HIP_ARCH=gfx1100 cargo build --release --bin supersonic`.
- ignored GPU test:
  `dflash_append_delta_q8_direct_attention_matches_extract_path`.
- one-prompt enabled/disabled A/B for the final append recurrent path.
- FFI shape profile confirming the direct recurrent row replaced the old
  capture plus extract rows.
- two full 10-prompt Lucebox serving-mode benchmark runs above `100 tok/s`
  mean.
- unchanged per-prompt generated-token counts and stable combined output hash.

The result is therefore not a hidden acceptance collapse. It is the same output
shape produced with a cheaper verifier round.
