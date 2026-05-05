// Router permutation kernel for Qwen 3.6 MoE batched-Q prefill (Stage B / M9).
//
// Given the router's top-K outputs for a chunk of N tokens, build the data
// structure that lets M10 dispatch one grouped GEMM per expert across all
// the (token, k_pos) pairs assigned to that expert. The permutation groups
// all assignments by their target expert and produces a parallel set of
// arrays the FFN replacement path can index by expert.
//
// **Inputs (GPU-resident):**
//   topk_idx    : [N, top_k] i32  — for each token, the top_k expert ids in
//                                    `[0, num_experts)`.
//   topk_weight : [N, top_k] BF16 — softmax+top-K+renormalised routing
//                                    weights produced upstream.
//
// **Outputs (allocated by caller):**
//   expert_offsets     : [num_experts + 1] i32  — prefix-sum start offsets
//                                                  per expert.
//   permuted_token_idx : [N * top_k] i32  — original token index for each
//                                            (sorted) assignment.
//   permuted_kpos      : [N * top_k] i32  — original top_k slot index, kept
//                                            so unpermute can write back to
//                                            the right output position.
//   permuted_weight    : [N * top_k] BF16 — routing weight for the assignment.
//
// **Algorithm (single-block counting sort).** With `chunk_size <= 512` and
// `num_experts <= 256` the entire permutation builds in a single block:
//
//   Phase 1 — lane-cooperative count: every thread atomically increments
//             counts[topk_idx[entry]] for its strided slice of entries.
//   Phase 2 — single-thread inclusive prefix scan over counts → expert_offsets;
//             reset counts[] to zero so it can be reused as Phase-3 cursors.
//   Phase 3 — lane-cooperative scatter: for each entry, atomicAdd on the
//             expert's cursor yields the in-segment slot, then write
//             permuted_token_idx / kpos / weight at expert_offsets[e] +
//             cursor.
//
// **Stability note:** atomicAdd on the cursor is NOT a stable sort —
// the relative order of entries within an expert's segment depends on
// thread-scheduling. The CPU reference must compare per-expert as a SET
// of (token_idx, kpos, weight) triples, not positionally. The downstream
// grouped GEMM is permutation-invariant within a segment so this is OK.
//
// **Shared memory:** num_experts * sizeof(int). At num_experts=256 → 1 KiB.
// **Block:** 256 threads. **Grid:** 1 block.
//
// Same isolation rules as the rest of qwen36_moe (CLAUDE.md): keep this
// file inside `kernels/qwen36_moe_persistent/` and don't share with other
// model families' kernels.

#pragma once

#include "../qwen36_moe_cuda_prelude.cuh"

namespace qwen36_moe {

// Grid: 1 block. Block: 256 threads (1D).
//
// `MAX_EXPERTS` pins the LDS size at compile time. The bridge rejects
// num_experts > 256 (status 141) so we always have enough room.
template <int MAX_EXPERTS>
__global__ void qwen36_moe_batched_prefill_router_permute_kernel(
    int  n_tokens,
    int  top_k,
    int  num_experts,
    const int* __restrict__         topk_idx,
    const hip_bfloat16* __restrict__ topk_weight,
    int* __restrict__               expert_offsets,
    int* __restrict__               permuted_token_idx,
    int* __restrict__               permuted_kpos,
    hip_bfloat16* __restrict__    permuted_weight
) {
    __shared__ int counts[MAX_EXPERTS];

    const int tid    = threadIdx.x;
    const int nt     = blockDim.x;
    const int total  = n_tokens * top_k;

    // Phase 1 — zero counts then count per expert.
    for (int e = tid; e < num_experts; e += nt) {
        counts[e] = 0;
    }
    __syncthreads();

    for (int entry = tid; entry < total; entry += nt) {
        const int e = topk_idx[entry];
        atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    // Phase 2 — single-thread exclusive prefix scan to expert_offsets, then
    // reset counts[] to zero so Phase 3 can reuse them as write cursors.
    if (tid == 0) {
        int s = 0;
        for (int e = 0; e < num_experts; ++e) {
            expert_offsets[e] = s;
            s += counts[e];
        }
        expert_offsets[num_experts] = s;   // == n_tokens * top_k
        for (int e = 0; e < num_experts; ++e) {
            counts[e] = 0;
        }
    }
    __syncthreads();

    // Phase 3 — scatter. atomicAdd(&counts[e], 1) returns the local slot
    // inside expert e's segment; the global write index is base + slot.
    for (int entry = tid; entry < total; entry += nt) {
        const int e         = topk_idx[entry];
        const int token_idx = entry / top_k;
        const int kpos      = entry - token_idx * top_k;
        const int slot      = atomicAdd(&counts[e], 1);
        const int dst       = expert_offsets[e] + slot;
        permuted_token_idx[dst] = token_idx;
        permuted_kpos[dst]      = kpos;
        permuted_weight[dst]    = topk_weight[entry];
    }
}

}  // namespace qwen36_moe
