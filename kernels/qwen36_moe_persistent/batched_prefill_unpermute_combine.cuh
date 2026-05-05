// Unpermute + weighted combine kernel for Qwen 3.6 MoE batched-Q prefill (Stage B / M11).
//
// Inverts the M9 permutation and computes the weighted sum of expert
// outputs per token:
//
//     combined[token, col] = sum_kpos( permuted_weight[dst] * expert_out[dst, col] )
//
// where `dst = permuted_inverse[token * top_k + kpos]` is the index into
// the M10 output where the (token, kpos) pair landed. The host builds the
// inverse table once per layer per chunk by walking M9's permuted_token_idx
// + permuted_kpos arrays — O(N * top_k), trivial cost. Building it host-side
// avoids touching the M9/M10 kernels (which are proven correct) and keeps
// this kernel a simple gather + dot product.
//
// **Inputs (GPU-resident):**
//   permuted_inverse : [N * top_k] i32  — for the (token, kpos) pair at
//                                          logical entry `token * top_k +
//                                          kpos`, the row index in the
//                                          permuted_*[] arrays where that
//                                          entry landed (i.e. its `dst`
//                                          slot in M9's scatter output).
//   permuted_weight  : [N * top_k] BF16 — M9 output; routing weight per
//                                          permuted entry.
//   expert_out       : [N * top_k, hidden] BF16 — M10 output.
//
// **Output:**
//   combined         : [N, hidden] BF16 — weighted sum of the top_k expert
//                                          outputs per token, ready to be
//                                          residual-added into the chunk's
//                                          hidden state.
//
// **Algorithm:** one block per (token, hidden tile of `BLOCK` cols). Each
// thread accumulates F32 over the top_k entries for its (token, col) and
// stores BF16 once at the end. No cross-thread reduction needed — each
// thread owns one column for one token.
//
// **Block layout:** 256 threads × 1D. Grid = (ceil(hidden / 256), N).
//
// Same isolation rules as the rest of qwen36_moe (CLAUDE.md): keep this
// file inside `kernels/qwen36_moe_persistent/` and don't share with other
// model families' kernels.

#pragma once

#include "../qwen36_moe_cuda_prelude.cuh"
#include "helpers.cuh"

namespace qwen36_moe {

// Grid:  (ceil(hidden / BLOCK), n_tokens, 1)
// Block: BLOCK threads (1D)
template <typename T, int BLOCK>
__global__ void qwen36_moe_batched_prefill_unpermute_combine_kernel(
    int                                  n_tokens,
    int                                  top_k,
    int                                  hidden,
    const int*          __restrict__     permuted_inverse,    // [N * top_k]
    const hip_bfloat16* __restrict__   permuted_weight,     // [N * top_k]
    const T*            __restrict__     expert_out,          // [N * top_k, hidden]
    T*                  __restrict__     combined             // [N, hidden]
) {
    const int token = blockIdx.y;
    if (token >= n_tokens) return;
    const int col_base = blockIdx.x * BLOCK;
    const int col      = col_base + threadIdx.x;
    if (col >= hidden) return;

    float acc = 0.0f;
    const int inv_base = token * top_k;
    for (int kpos = 0; kpos < top_k; ++kpos) {
        const int dst = permuted_inverse[inv_base + kpos];
        const float w = __bfloat162float(permuted_weight[dst]);
        const float v = static_cast<float>(expert_out[
            static_cast<size_t>(dst) * hidden + col]);
        acc += w * v;
    }
    combined[static_cast<size_t>(token) * hidden + col] =
        static_cast<T>(bf16_round_rne_f32(acc));
}

}  // namespace qwen36_moe
