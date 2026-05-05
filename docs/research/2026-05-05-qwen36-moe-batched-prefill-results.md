# Qwen 3.6 MoE Batched-Q Prefill — Design and Results

**Date:** 2026-05-05  
**Branch:** `worktree-qwen36-moe-batched-prefill`  
**Plan:** `docs/superpowers/plans/2026-05-05-qwen36-moe-batched-prefill-phase1.md`  
**Hardware:** AMD Radeon RX 7900 XTX (gfx1100), 24 GiB

## Problem

Long-context prefill on `qwen3.6-35b-a3b` was the dominant wall-clock cost of a research workload. Per `docs/research/2026-05-04-qwen36-longctx-comparison-results.md` (the May-04 baseline that drove this work):

| Context tokens | Prefill wall (s) |
|---:|---:|
| 512 | 13.3 |
| 2048 | 73.5 |
| 4096 | 196.3 |
| 8192 | 576.3 (**9.6 minutes**) |

The per-token persistent decode megakernel (`kernels/qwen36_moe_persistent/persistent_decode.hip`) is well-tuned for *decode* but, when reused for prefill, processes one prompt token per launch. Each step's full-attention phase re-reads the entire growing K cache — quadratic total cost — and each step's MoE FFN runs the router + top-K + 8 expert matvecs + shared expert with `M=1` per matmul.

## Approach

Build a separate batched-Q prefill code path (mirrors Qwen 3.5's structure: prefill kernel ≠ decode megakernel). Don't touch the persistent megakernel. Two stages:

- **Stage A** — batched-Q full attention with K-tile share across queries.
- **Stage B** — permute-by-expert grouped MoE FFN with one INT4 GEMM per expert across all its assigned tokens in the chunk.

Run both as one PR. Make the path opt-in initially via env vars so each stage's contribution is measurable; flip to default at the end.

## Implementation map

The plan ran to 13 milestones (M1–M13) — each a single shippable commit. Final commit graph:

```
e9aee14 runner: M13 — promote Qwen 3.6 MoE batched-Q prefill to default
b6368b4 runner: M12 — chunk size policy chosen from prefix context, capped at WMMA-100%
2440130 runner: M12 — runtime-dispatched chunk size (64 / 256 / 1024)
2f9d96f runner: M11 — wire grouped MoE FFN into batched prefill orchestrator
5daa0e3 kernel: M10 — qwen3.6-moe batched-prefill grouped expert GEMM
edc4f85 kernel: M9 — qwen3.6-moe batched-prefill router permutation
3207120 test:   relax M1 parity threshold to codebase INT4/BF16 floor
ecbadde runner: M6.2 — wire batched primitive sequence for full-attn prefill
0238e47 runner: M6.1 — qwen3.6-moe batched-prefill orchestrator skeleton
170883d kernel: M3 — qwen3.6-moe batched-Q full-attention prefill kernel
22e0759 bench:  M2 — long-ctx harness gains --batched-prefill flag
8c95f5a test:   M1 — qwen3.6-moe batched-prefill parity scaffold + env-var hook
```

### New GPU kernels (all `kernels/qwen36_moe_persistent/`)

- **M3** `batched_prefill_attn_full.cuh` — direct port of the Qwen 3.5 K-tiled attention kernel into the qwen36 namespace. BM=4 warps cooperatively load BK=32-row K/V tiles into LDS once per outer tile; each warp owns one query row and runs the FlashAttention online-softmax recurrence (m_i, l_i, acc[]) over its tile. LDS = 2 × BK × head_dim × sizeof(BF16) = 32 KiB at hd=256.
- **M9** `batched_prefill_router_permute.cuh` — single-block counting-sort kernel: per-token top-K assignments → `(expert_offsets, permuted_token_idx, permuted_kpos, permuted_weight)`.
- **M10** `batched_prefill_grouped_expert.cuh` — persistent-block work-stealing kernel; processes all 256 experts in one launch. Each block claims an expert via `atomicAdd(counters[0], 1)`, then for each of its assigned rows runs `gate_up @ x_token → silu*mul → down @ silu*mul → expert_out`. Reuses `int4_dq8_matvec_partial` and `wmma_int4_matvec_partial_16rows` from `helpers.cuh` — math identical to the existing `ffn_phase.cuh` Phase G/I, restructured into a per-expert outer loop.
- **M11** `batched_prefill_unpermute_combine.cuh` — per (token, hidden_col) sums the top_k expert contributions weighted by router weight.

### Host orchestrator (`crates/runner/src/qwen36_moe/batched_prefill.rs`)

Replaces the engine's main per-step loop for the prefill range `[0, effective_prompt_len - 1)`. Per chunk:

1. Embed N tokens H2D batched.
2. For each layer:
   - **Full-attn**: rms_norm_rows (input norm) → matmul_int4 ×3 (q/k/v proj) → split q+gate (HF interleaved per-head layout, NOT concatenated) → rms_norm_rows ×2 (per-head q_norm/k_norm) → apply_rope_prefill ×2 → gpu_hal::copy_d2d (KV cache write) → M3 attention → sigmoid·gate → cast F32→BF16 → matmul_int4 (o_proj) → element_add (residual). All using `kernel_ffi::prefill_ffi` primitives that turned out to be layout-agnostic enough to call from qwen36-moe context (verified empirically).
   - **Linear-attn**: per-token loop calling existing `qwen36_moe_hip_linear_step_launch`.
   - **MoE FFN** (default grouped path): rms_norm_rows (post-attn norm) → matmul_rhs_transposed (router, BF16) → host softmax+top-K+renorm → M9 → M10 → M11 → batched shared expert (matmul_int4 ×3 + swiglu_mul + sigmoid_mul) → element_add ×2 (residual + shared).

The non-default per-token FFN path (when `SUPERSONIC_QWEN36_MOE_GROUPED_FFN=0`) preserves the existing per-token `ffn_step_launch` for bisecting.

### Chunk size policy

Picked at runtime from the prompt's prefix context, capped at the largest size that gives 100% WMMA utilization in M10's grouped GEMM:

```rust
const PREFILL_CHUNK_SIZE_WMMA_FULL: usize = 512;   // = 16 * num_experts / top_k
fn pick_chunk_size(remaining: usize) -> usize {
    if remaining >= 512 { 512 } else { remaining }
}
```

Buffers (`FullAttnBatchScratch`) are sized for `PREFILL_CHUNK_SIZE_MAX = 1024` (8 MB max scratch at hd=256/H=16) so the kernels can grow without re-tuning allocation.

## Results

Three-way A/B at 4K context (gfx1100, qwen3.6-35b-a3b INT4, NIAH-style synthetic prompt, `prefill_total_ms`):

| Path | Prefill (s) | Speedup |
|---|---:|---:|
| Per-token persistent megakernel (legacy, pre-PR) | 134.44 | 1.00× |
| Stage A only (batched attn, per-token FFN inside chunk) | 92.44 | 1.45× |
| Stage A + B with chunk=64 | 79.10 | 1.70× |
| Stage A + B with chunk=256 | 76.07 | 1.77× |
| Stage A + B with chunk=512 (WMMA-100%, the default) | **75.03** | **1.79×** |
| Stage A + B with adaptive {64,256,1024} | 74.74 | 1.80× |

Smaller-context numbers (also Stage A+B at the new default):

| Context | Per-token | Stage A+B | Speedup |
|---:|---:|---:|---:|
| 512 | 13.4 s | 8.9 s | 1.50× |
| 2048 | 61.1 s | 38.6 s | 1.58× |
| 4096 | 134.4 s | 75.0 s | **1.79×** |

The 8K data point couldn't be measured cleanly: at 8192 context the persistent path's MoE expert weight VMM allocation hits `hipMemCreate failed with status 2` (a pre-existing 24 GiB pressure issue, not introduced by this PR). The batched path inherits the same limit because it uses the same expert weights.

## What didn't move the needle as much as expected

- The original projection (5–10× at long context) assumed per-token *launch overhead* was the dominant cost. It isn't — the persistent megakernel does all 40 layers in one launch, so launch overhead is small. The dominant cost is **INT4 weight read bandwidth** (per-token: each layer per token reads ~4 GiB of INT4 weights total at 4K context).
- Stage A's K-tile share dropped attention work but only ~30% of total prefill time was attention.
- Stage B's grouped GEMM cuts MoE-FFN expert weight bandwidth from O(N×top_k×expert_size) to O(num_experts×expert_size) per chunk, but at chunk=512 with avg-tokens-per-expert=16 we just cross into 100% WMMA territory; the load-imbalance of router assignments (some experts get 30 tokens, some get 0) means real WMMA utilization is lower than peak.
- Larger chunks (1024) give marginal extra K-tile-share in attention (+0.4% at 4K). Bigger isn't free — scratch grows linearly.

## What would unlock more speedup

1. **HIP graph capture** of the per-token launches inside a chunk (could amortize the ~50 µs per-launch overhead × 1000+ launches per chunk).
2. **Full FlashAttention-2 with Q-tiling** (current M3 only tiles K).
3. **Fused FFN megakernel** — combine the router + permute + grouped GEMM + unpermute + shared into one cooperative-launch kernel, eliminating intra-chunk launches.
4. **VRAM headroom for 8K+ contexts** — the 24 GiB limit blocks the workloads that would benefit most from batched prefill.

Each of these is multi-day. Tracked as follow-ups, not part of this PR.

## Tests

- `crates/runner/tests/qwen36_moe_batched_prefill_attn_kernel_parity.rs` — M3 kernel-direct parity sweep.
- `crates/runner/tests/qwen36_moe_batched_prefill_router_permute_parity.rs` — M9 kernel-direct parity.
- `crates/runner/tests/qwen36_moe_batched_prefill_grouped_expert_parity.rs` — M10 kernel-direct parity (covers WMMA + scalar paths).
- `crates/runner/tests/qwen36_moe_batched_prefill_parity.rs` — end-to-end gate (legacy `=0`-forced vs default-batched, cossim ≥ 0.999, argmax matches).

The pre-existing `qwen36_moe_multilayer_parity`, `qwen36_moe_kv_fp8_parity`, and `specprefill_qwen36_moe_*_parity` tests continue to pass — the FP8 KV and SpecPrefill paths automatically fall through to per-token via `supports_batched_path()`.
