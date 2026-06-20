// Final RMSnorm + lm_head GEMV phase for the Qwen3.6-MoE persistent
// megakernel — Phase 3f.
//
// The persistent kernel runs all 40 layers in one cooperative launch
// (Phase 3e). At generation steps the host then has to launch
// `qwen36_moe_lm_head_kernel` separately to produce logits, which is one
// more launch (~30 µs) plus a D2H/H2D round trip on `final_hidden`.
// Folding the final RMSnorm + lm_head GEMV into the megakernel collapses
// that to zero extra launches.
//
// Design notes
// ------------
// The standalone `qwen36_moe_lm_head_kernel` and
// `qwen36_moe_lm_head_wmma_kernel` use launch geometries the persistent
// kernel can't share:
//   - Scalar: 96 blocks × 256 threads, work-stealing on `counter`.
//   - WMMA:   ceil(vocab/16) blocks × 32 threads (one block per
//             16-vocab tile, no work-stealing).
//
// The persistent kernel runs at 96 blocks × 256 threads = 768 waves on
// gfx1100. To reuse that shape for lm_head we need a work-stealing path
// for both scalar and WMMA. This file provides one device function that
// dispatches by `USE_WMMA`:
//
//   - Scalar path: each BLOCK claims one vocab row via atomicAdd on
//     `counters[0]`, reduces hidden into a single output. Same as the
//     standalone scalar kernel, just with the activation pre-staged in
//     LDS by the previous layer's residual instead of re-loaded from
//     `final_hidden`.
//
//   - WMMA path: each BLOCK claims 128 vocab rows via atomicAdd
//     (`counters[0] += 128`), then its 8 waves each handle one 16-row
//     WMMA tile in parallel. Same wmma_bf16 16x16x16 path the standalone
//     WMMA kernel uses, just with work-stealing on top so we use all 96
//     CUs / 768 waves at once instead of queueing 15k blocks × 1 wave
//     each.
//
// LDS layout matches the layer phases:
//   lds[0,        hidden):           x_norm_lds (BF16-rounded F32 view).
//   lds[hidden,   hidden+block_size): shared_scratch (block reduction).
// Total 9 KiB at hidden=2048, block_size=256 — same as full_attn /
// linear_attn / ffn phases, so no LDS budget change for the megakernel.

#pragma once

#include "qwen36_moe_persistent/helpers.cuh"

namespace qwen36_moe {

// Final RMSnorm + lm_head GEMV. Caller responsibilities:
//
//   - `input_hidden` is the previous layer's post-FFN residual; this
//     function applies the final RMSnorm to it (with the standard HF
//     `(1.0 + w)` unit offset) and uses the result as the matvec
//     activation.
//   - `final_norm_w` and `lm_head_w` are device-resident BF16 weights.
//     `lm_head_w` is `[vocab, hidden]` BF16 — the engine pre-dequants
//     INT4 lm_head once at startup (the bake's lm_head is INT4 but the
//     WMMA tile path expects BF16).
//   - `logits_out` is `[vocab]` BF16; the kernel writes one logit per
//     vocab row.
//   - `counters[0]` must be 0 on entry (caller does the reset_counters_16
//     before this phase).
//   - Caller invokes `grid_barrier` before this fn so the previous
//     phase's writes to `input_hidden` are visible to all blocks here.
//
// `USE_WMMA` is plumbed from the persistent kernel template; the
// activation-load + RMSnorm body is shared, the matvec path branches.
template <typename T, bool USE_WMMA = false>
__device__ inline void qwen36_moe_lm_head_device(
    int                            hidden,
    int                            vocab,
    float                          rms_norm_eps,
    const T* __restrict__          input_hidden,    // [hidden] BF16
    const T* __restrict__          final_norm_w,    // [hidden] BF16
    const T* __restrict__          lm_head_w,       // [vocab, hidden] BF16
    T*       __restrict__          logits_out,      // [vocab] BF16, nullable
    unsigned int* __restrict__     top1_out,        // [1] U32, nullable
    float*   __restrict__          top1_scratch,    // [2 * gridDim.x] F32
    unsigned int* __restrict__     counters,
    unsigned int* __restrict__     barrier_counter,
    unsigned int* __restrict__     barrier_flag,
    int                            num_blocks
) {
    const int tid        = threadIdx.x;
    const int block_size = static_cast<int>(blockDim.x);

    extern __shared__ float lds[];
    float* x_norm_lds    = lds;
    float* shared_scratch = lds + hidden;

    // -- Phase A: RMSnorm of input_hidden into LDS -------------------------
    // Mirrors qwen36_moe_lm_head_kernel's Phase A. Each block computes
    // the same RMSnorm result into its own LDS — redundant compute but
    // avoids a cross-block sync. The matvec that follows is the
    // dominant cost (~970 MiB BF16 weight read at vocab=248k, hidden=2048).
    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        const float v = load_as_float<T>(input_hidden, col);
        x_norm_lds[col] = v;
        partial_sq += v * v;
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // Apply (1.0 + w) unit offset, BF16-round, store F32-of-BF16.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(final_norm_w, col);
        x_norm_lds[col] =
            bf16_round_rne_f32(x_norm_lds[col] * inv_rms * (1.0f + w));
    }
    __syncthreads();

    // -- Phase B: work-stealing matvec over vocab rows ---------------------
    float local_best = -INFINITY;
    int local_best_idx = 0;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    if constexpr (USE_WMMA) {
        // Same wave32 layout the standalone WMMA kernel uses, dropped
        // into a work-stealing loop. Each block claims 128 vocab rows
        // per atomicAdd; its 8 waves each compute one 16-row WMMA tile
        // (32 lanes × 4 K-iters at hidden=2048 = standard m=1 WMMA GEMV
        // pattern from the standalone kernel).
        const int wave_id = tid >> 5;
        const int lane_in_wave = tid & 31;
        const int lane_row = lane_in_wave & 15;
        const int lane_half = lane_in_wave >> 4;
        const uint16_t* x_norm_lds_u16 =
            reinterpret_cast<const uint16_t*>(x_norm_lds);
        const uint16_t* lm_head_w_u16 =
            reinterpret_cast<const uint16_t*>(lm_head_w);

        for (;;) {
            __shared__ unsigned int row_base_s;
            if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
            __syncthreads();
            const int row_base = static_cast<int>(row_base_s);
            if (row_base >= vocab) break;

            const int rhs_row = row_base + wave_id * 16 + lane_row;
            const bool in_range = rhs_row < vocab;
            const size_t rhs_row_base =
                in_range ? static_cast<size_t>(rhs_row) * hidden : 0;

            qwen36_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const int k_main = hidden & ~15;
            for (int kk = 0; kk < k_main; kk += 16) {
                qwen36_short16 a;
                if (lane_row == 0) {
                    // x_norm is stored F32 in LDS but the top 16 bits of
                    // each F32 ARE the BF16 bit pattern (we BF16-rounded
                    // before storing in Phase A). Read the high half via
                    // the u16 alias so the WMMA A operand is BF16-shaped.
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        // Each F32 occupies 2 u16 slots (LE: low, high);
                        // the BF16 bits are the high u16 (offset
                        // 2*idx + 1).
                        a[i] = static_cast<short>(
                            x_norm_lds_u16[2 * (kk + i) + 1]);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) a[i] = 0;
                }

                qwen36_short16 b;
                if (in_range) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        b[i] = static_cast<short>(
                            lm_head_w_u16[rhs_row_base + kk + i]);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = 0;
                }

                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
            }

            // K tail (defensive — production hidden=2048 is /16-aligned).
            if (k_main != hidden) {
                qwen36_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                qwen36_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                if (lane_row == 0) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int kc = k_main + i;
                        if (kc < hidden) {
                            a[i] = static_cast<short>(
                                x_norm_lds_u16[2 * kc + 1]);
                        }
                    }
                }
                if (in_range) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int kc = k_main + i;
                        if (kc < hidden) {
                            b[i] = static_cast<short>(
                                lm_head_w_u16[rhs_row_base + kc]);
                        }
                    }
                }
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
            }

            // Lanes with lane_half==0 carry C[0, lane_row]. Each wave's
            // 16 such lanes write the wave's 16-row tile.
            if (lane_half == 0 && in_range) {
                const float logit = bf16_round_rne_f32(acc[0]);
                if (logits_out != nullptr) {
                    logits_out[rhs_row] = static_cast<T>(logit);
                }
                if (top1_out != nullptr &&
                    (logit > local_best ||
                     (logit == local_best && rhs_row < local_best_idx))) {
                    local_best = logit;
                    local_best_idx = rhs_row;
                }
            }
            __syncthreads();
        }
    } else
#endif
    {
        // Scalar work-stealing path. One block claims one vocab row per
        // atomicAdd, threads cooperate on the hidden-wide reduction.
        // Same body as qwen36_moe_lm_head_kernel's Phase B.
        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= vocab) break;

            const T* w_row = lm_head_w + static_cast<size_t>(my_row) * hidden;
            float partial = 0.0f;
            for (int col = tid; col < hidden; col += block_size) {
                partial += load_as_float<T>(w_row, col) * x_norm_lds[col];
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const float logit = bf16_round_rne_f32(shared_scratch[0]);
                if (logits_out != nullptr) {
                    logits_out[my_row] = static_cast<T>(logit);
                }
                if (top1_out != nullptr &&
                    (logit > local_best ||
                     (logit == local_best && my_row < local_best_idx))) {
                    local_best = logit;
                    local_best_idx = my_row;
                }
            }
            __syncthreads();
        }
    }

    if (top1_out != nullptr) {
        // Reduce each block's best vocab row. `x_norm_lds` is no longer
        // needed after the matvec, so reuse its first block_size entries as
        // integer scratch without increasing dynamic LDS.
        shared_scratch[tid] = local_best;
        x_norm_lds[tid] = __int_as_float(local_best_idx);
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const float a = shared_scratch[tid];
                const float b = shared_scratch[tid + s];
                const int ia = __float_as_int(x_norm_lds[tid]);
                const int ib = __float_as_int(x_norm_lds[tid + s]);
                if (b > a || (b == a && ib < ia)) {
                    shared_scratch[tid] = b;
                    x_norm_lds[tid] = __int_as_float(ib);
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            top1_scratch[blockIdx.x] = shared_scratch[0];
            top1_scratch[num_blocks + blockIdx.x] = x_norm_lds[0];
        }
        grid_barrier(barrier_counter, barrier_flag, num_blocks);

        // Final grid-wide reduction on block 0. Supports override grids larger
        // than block_size by first having each thread scan a stride.
        if (blockIdx.x == 0) {
            float best = -INFINITY;
            int best_idx = 0;
            for (int i = tid; i < num_blocks; i += block_size) {
                const float v = top1_scratch[i];
                const int idx = __float_as_int(top1_scratch[num_blocks + i]);
                if (v > best || (v == best && idx < best_idx)) {
                    best = v;
                    best_idx = idx;
                }
            }
            shared_scratch[tid] = best;
            x_norm_lds[tid] = __int_as_float(best_idx);
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const float a = shared_scratch[tid];
                    const float b = shared_scratch[tid + s];
                    const int ia = __float_as_int(x_norm_lds[tid]);
                    const int ib = __float_as_int(x_norm_lds[tid + s]);
                    if (b > a || (b == a && ib < ia)) {
                        shared_scratch[tid] = b;
                        x_norm_lds[tid] = __int_as_float(ib);
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                top1_out[0] = static_cast<unsigned int>(__float_as_int(x_norm_lds[0]));
            }
        }
    }
}

}  // namespace qwen36_moe
