# Qwen3.6-MoE persistent megakernel — Phase 3 plan

This document maps the proven Qwen3.5-4B persistent-decode pattern in
`kernels/full_attention_4b.hip::supersonic_qwen35_persistent_decode_kernel`
onto Qwen3.6-MoE's hybrid-attention + MoE-FFN architecture, with a
multi-file decomposition so the implementation lands in reviewable sub-PRs.

## Why a megakernel

Current chained-decode launches per token (35B-A3B, gfx1100):

  - 30 linear-attn kernels (1 per linear layer)
  - 10 full-attn kernels (1 per full-attn layer)
  - 40 FFN kernels (1 per layer)
  - 1 lm_head kernel (the GPU final-norm + GEMV)

Total: **81 launches/token**. At ~30 µs HIP launch overhead each, that's
~2.4 ms/token of pure host-side queue work — **~9% of the 27 ms/token
total chain time** on the local 7900 XTX (38 tok/s).

Folding everything into a single cooperative-launch kernel collapses this
to **1 launch/token**, recovering essentially all of the 2.4 ms/token.
Independent of MTP accept rate — helps plain decode and speculative
paths equally. Also lays groundwork for a future batched chain
(Phase 6.4d) since multi-query verify can fold into the same
descriptor-walk loop without re-doing the launch coordination.

Reference for the pattern:
`kernels/full_attention_4b.hip::supersonic_qwen35_persistent_decode_kernel`
(~1400 LoC, gfx1100/gfx1150-validated).

## Architectural mapping

Qwen3.5-4B (the reference) has uniform full-attention layers. Qwen3.6-MoE
diverges in three places:

1. **Hybrid attention pattern**: every 4th layer is full-attn, the rest
   are linear-attn. The qwen3.5-4B kernel branches on `L.layer_type`
   already (it has both prefill and decode variants); qwen3.6-MoE follows
   the same convention but with linear-attn as the dominant layer count
   (30/40 layers).
2. **MoE FFN with top-K=8 routing**: instead of qwen3.5-4B's dense FFN
   (`gate_proj` / `up_proj` / `down_proj`), qwen3.6-MoE has 256 experts
   per layer with per-token top-K=8 routing plus a shared expert. The
   FFN phase becomes a routing pass + per-selected-expert dispatch.
   Concurrent expert dispatch already lives in
   `qwen36_moe_ffn_step_kernel` (PR #74).
3. **MTP head**: when `--speculative-decode` is set, the MTP draft chain
   reuses the same per-block kernels via `run_mtp_layer_step`. The
   megakernel is base-model only — speculative wiring stays as it is
   (per-step closure or batched via Phase 6.4c.2). The MTP layer itself
   is structurally one full-attn block + MoE FFN, so if we ever want
   to fold MTP into a megakernel, the existing phase code reuses
   directly.

The qwen3.5-4B kernel is well-suited as a template; we adapt it rather
than write from scratch.

## File layout — multi-file decomposition

To keep individual files reviewable and side-step the
"3000-line monolithic kernel function" trap, split the megakernel
across header + impl files:

```
kernels/
├── qwen36_moe.hip                    [existing — keep step kernels for back-compat
│                                      + parity tests; bridge launchers point here
│                                      until the megakernel ships]
├── qwen36_moe_bridge.cpp             [existing — gains a new
│                                      `qwen36_moe_hip_persistent_decode_launch`
│                                      entry once the megakernel lands]
└── qwen36_moe_persistent/            [new folder for the megakernel work]
    ├── helpers.cuh                   [shared `__device__ inline` primitives:
    │                                  RMSNorm reduction, BF16 round, INT4 dequant,
    │                                  small wave-reduce helpers. Pulled out of
    │                                  qwen36_moe.hip, no behavior change]
    ├── full_attn_phase.cuh           [`__device__ inline` per-layer full-attn
    │                                  phase logic. Mirrors stages 1-5 of
    │                                  `qwen36_moe_attn_step_kernel` but called
    │                                  from inside the megakernel loop, with the
    │                                  layer's descriptor + workspace offsets
    │                                  passed as arguments]
    ├── linear_attn_phase.cuh         [same shape for linear-attn — stages
    │                                  mirror `qwen36_moe_linear_step_kernel`'s
    │                                  delta-rule + recurrent state update]
    ├── ffn_phase.cuh                 [MoE FFN: post-attn norm + router + top-K
    │                                  selection + concurrent expert dispatch +
    │                                  shared expert. Mirrors
    │                                  `qwen36_moe_ffn_step_kernel`. Uses
    │                                  `sync_buf`'s 16 work-stealing slots
    │                                  identically to PR #74.]
    └── persistent_decode.hip         [the `__global__` kernel that walks the
                                       layer descriptor array, calls each phase
                                       header, applies grid_barrier between
                                       phase boundaries. Plus the lm_head fold
                                       at the end]
```

`#include`-driven composition: `persistent_decode.hip` includes the four
headers; the bridge `qwen36_moe_bridge.cpp` includes
`qwen36_moe_persistent/persistent_decode.hip` to reach the kernel
template. Build script's `-I kernel_dir` flag already enables the
relative-path includes.

## Phase boundaries inside the megakernel

Per layer, the kernel runs these phases in order with a `grid_barrier`
between each (matches Qwen3.5-4B):

```
for layer in 0..num_layers:
    L = layers[layer]                          // descriptor read
    # === Token mixer ===
    if L.is_full_attention:
        full_attn_phase::run(L, hidden, ws, ...)  // RMSNorm + Q/K/V proj +
                                                   // q-gate + per-head Q/K norm +
                                                   // RoPE + GQA attn + o_proj
                                                   // + sigmoid(gate) + residual
    else:
        linear_attn_phase::run(L, hidden, ws, ...)// RMSNorm + in_proj_qkv + z +
                                                   // a/b + conv1d + delta-rule
                                                   // recurrent update + out_proj
                                                   // + residual
    grid_barrier()
    # === FFN ===
    ffn_phase::run(L, hidden, ws, ...)              // post-attn RMSNorm + router +
                                                    // top-K + concurrent expert
                                                    // dispatch + shared expert +
                                                    // residual
    grid_barrier()
# === Final norm + lm_head GEMV + argmax ===
final_phase::run(...)                                // currently a separate launch
                                                     // (lm_head_launch); folding it
                                                     // into the megakernel saves one
                                                     // more launch
```

`grid_barrier` uses the existing primitive in `qwen36_moe.hip` (already
proven correct via the descriptor walk stub on cooperative launch).

## Workspace + state lifecycle

Single-launch persistent kernel needs all per-token scratch reachable
without re-allocation across phases. Reuse the existing per-layer
buffer pool from chained decode:

  - `attn_workspace` F32, sized `max(full_attn, linear_attn)` per the
    multi-layer parity test's sizing helpers.
  - `attn_output` BF16, max staged-attn intermediate.
  - `ffn_workspace` F32 + `ffn_output` BF16 + `ffn_output_idx` U32.
  - `sync_buf` 96 bytes — work-stealing counters + grid barrier state.
    Same one-buffer-shared-across-phases pattern as the chained path.
  - Hidden ping-pong: `[hidden]` BF16 input + `[hidden]` BF16 output,
    swapped per layer. Persistent kernel maintains the swap via
    `blockIdx.x == 0` writeback to a single hidden_io buffer (same as
    qwen3.5-4B).

Linear-attn `conv_state` + `recurrent_state` are mutated in place per
layer — same lifetime as chained, just no inter-launch sync.

KV cache slots write at `position` for full-attn layers, identical to
the existing `kv_cache_k` / `kv_cache_v` semantics (Phase 6.2c.2 added
`cache_pos` decoupling for MTP, but the base-model megakernel uses the
default `cache_pos = position` path).

## Sub-PR plan

The work decomposes cleanly into reviewable PRs that each end in a
green build + parity guard:

### Phase 3a — helpers.cuh extraction

Pull the shared `__device__ inline` primitives currently inlined inside
`qwen36_moe.hip` into a header. Single source of truth so the existing
step kernels and the upcoming megakernel kernel both call the same
helpers (avoids a duplicate-and-drift maintenance tax).

  - `int4_dequant_8` / `int4_dequant_scalar`
  - `bf16_round_rne_f32`
  - `wave_sum`, `block_reduce_sum`
  - `grid_barrier`, `grid_barrier_reset_counter` (already extractable)
  - `wmma_int4_matvec_partial_16rows` (the WMMA tile helper from Phase 2)

Risk: low. Pure refactor — existing parity tests gate correctness.
Size: ~300 lines moved + the new header.

### Phase 3b — full_attn_phase.cuh

Extract full-attn stages 1-5 (`qwen36_moe_attn_step_kernel`'s body) into
a `__device__ inline run(...)` function in the header. Existing
`qwen36_moe_attn_step_kernel` becomes a thin `__global__` wrapper that
calls the device helper. All 6 attn parity tests stay green
(`SUPERSONIC_QWEN36_ORACLE_JSON`).

Risk: medium. The kernel is ~600 lines of phased compute with a
non-trivial WMMA inner loop; subtle shape changes are easy to miss.
Mitigation: keep the wrapper byte-for-byte equivalent — function
arguments are the same, just an extra layer of indirection.
Size: ~700 lines refactor.

### Phase 3c — linear_attn_phase.cuh

Same treatment for `qwen36_moe_linear_step_kernel`. State-bound delta-
rule recurrence + conv1d window state. 6 linear parity tests gate.
Size: ~700 lines refactor.

### Phase 3d — ffn_phase.cuh

Same for `qwen36_moe_ffn_step_kernel`. The biggest one: 5 stages,
concurrent-experts dispatch, shared expert path, top-K=8 routing.
8 FFN parity tests gate.
Size: ~1000 lines refactor.

### Phase 3e — persistent_decode.hip + bridge launcher

The new `__global__` kernel that walks `Qwen36MoeDecodeLayerDesc[]`,
calling the phase helpers in order with `grid_barrier` between phases.
New bridge entry `qwen36_moe_hip_persistent_decode_launch` and a
matching FFI safe wrapper. Engine routes through the new path when
WMMA is available.

End-to-end parity: chained-decode output bit-identical to
persistent-decode output on the local "quick brown fox" test prompt
(except for WMMA F32-accumulation-order drift, which the existing
multilayer parity test already tolerates at cos_sim ≥ 0.999).

Perf measurement: stage timings before vs after on 16-token decode.
Expected: chain_total_ms drops by ~2.4 ms (one-shot launch vs
80 chained); per-token total drops by ~10%.

Risk: medium. The descriptor walk + grid barrier infrastructure
(`qwen36_moe_descriptor_walk_stub`) is already proven on gfx1100;
the new kernel just fills in the actual compute body.
Size: ~600 lines new code + ~200 lines bridge/FFI/engine wiring.

### Phase 3f (optional) — fold lm_head into megakernel

The lm_head launch is currently separate from chained decode (~1.5 ms
each call); folding it into the persistent kernel saves another launch.
Smaller win (~30 µs) but cleaner end-to-end. Defer to follow-up if
3a-e are sufficient.

## Total scope estimate

  - 3a: 300 lines moved + header
  - 3b: 700 lines refactor
  - 3c: 700 lines refactor
  - 3d: 1000 lines refactor
  - 3e: 800 lines new + wiring
  - 3f: 200 lines wiring (optional)

**Total: ~3500 lines net** across ~5 sub-PRs. Each shippable
independently because the existing chained path stays as the
back-compat reference until 3e wires the megakernel into the engine.

## Validation strategy

Each sub-PR reuses the existing per-block parity tests
(`qwen36_moe_attn_step_*`, `qwen36_moe_linear_step_*`,
`qwen36_moe_ffn_step_*`). Phases 3b-d touch the wrappers, so the
existing tests gate correctness — no behavior change is the explicit
contract.

Phase 3e adds a new e2e parity test similar to
`crates/runner/tests/qwen36_moe_multilayer_parity.rs` but driving the
persistent kernel instead of the chained path. cos_sim ≥ 0.999 against
the multilayer Python oracle.

CLI smoke for 3e: `--max-new-tokens 16` produces bit-identical token
sequence to plain greedy on the local "quick brown fox" prompt.

## Open questions

  1. **Scratch sizing under coop launch.** The chained path allocates
     one `attn_workspace` sized for the worst phase. Persistent kernel
     amortizes across all 40 layers using the same buffer (sequential
     reuse, no aliasing). Verify total VRAM impact against the budget
     in `runner::registry::Qwen36MoeKernelParams`.
  2. **CU occupancy on gfx1100.** qwen3.5-4B's persistent kernel
     launches `dim3(96)` on 7900 XTX (= num_cus). Same shape applies
     here; just confirm that the larger LDS footprint of
     concurrent-experts FFN doesn't bump us off `2 WGs/CU`.
  3. **MoE FFN routing inside the megakernel.** The chained FFN uses
     a separate atomic-counter slot per top-K group (PR #74's
     concurrent-experts dispatch). Need to confirm the same 16-slot
     `sync_buf` layout fits cleanly across phases (FFN claims slots
     [0..2*top_k), grid barrier @ +64/+68 — same as today).
  4. **WMMA fallback path.** Non-WMMA hosts (gfx1150, CDNA bring-up)
     keep the chained path. Bridge launcher gates on
     `device_supports_wmma_bf16`.

These resolve naturally during 3a-e — none should reshape the design.

## Cross-reference

  - `kernels/full_attention_4b.hip::supersonic_qwen35_persistent_decode_kernel`
    — proven 1400-line reference for the descriptor-walk pattern.
  - `kernels/qwen36_moe.hip::qwen36_moe_descriptor_walk_stub` — existing
    skeleton with cooperative launch + grid barrier. Phase 3e replaces
    its body.
  - `docs/qwen36-moe-pr4c-plan.md::Step 4` — the original plan note that
    deferred this work; this doc supersedes it with a concrete
    decomposition.
