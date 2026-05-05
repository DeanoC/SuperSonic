// Grouped-expert INT4 GEMM kernel for Qwen 3.6 MoE batched-Q prefill (Stage B / M10).
//
// Replaces per-token MoE FFN dispatch with a single launch that processes
// ALL `num_experts` experts in one go via persistent-block work-stealing:
// each block claims an expert id from an atomic counter, then iterates the
// (token, kpos) pairs assigned to that expert by the M9 router permutation
// kernel. For each pair the block runs gate_up + silu*mul + down INT4
// matmuls and writes the result into a `[N * top_k, hidden]` output buffer.
//
// **Inputs (GPU-resident):**
//   x_norm              : [N, hidden] BF16              — post-input-RMSnorm
//                                                         hidden states. The
//                                                         caller produces
//                                                         this; we read it
//                                                         indexed by
//                                                         permuted_token_idx.
//   expert_offsets      : [num_experts + 1] i32         — M9 output (prefix
//                                                         sum). Slice
//                                                         [start, end) of
//                                                         permuted_* arrays
//                                                         is expert e's
//                                                         segment.
//   permuted_token_idx  : [N * top_k] i32               — M9 output; original
//                                                         token id for each
//                                                         segment entry.
//   experts_gate_up_w   : [E, 2*I, hidden/2] u8         — INT4 packed
//                                                         gate+up fused
//                                                         projection.
//   experts_gate_up_s   : [E, 2*I/gs, hidden/gs] BF16   — INT4 scales.
//   experts_gate_up_z   : [E, 2*I/gs, hidden/gs] BF16   — INT4 zeros.
//   experts_down_w      : [E, hidden, I/2] u8           — INT4 packed down.
//   experts_down_s      : [E, hidden/gs, I/gs] BF16
//   experts_down_z      : [E, hidden/gs, I/gs] BF16
//
// **Output:**
//   expert_out          : [N * top_k, hidden] BF16      — for each permuted
//                                                         row holds
//                                                         down(silu(gate(x))
//                                                         * up(x)) for the
//                                                         (token, kpos)
//                                                         indexed at row.
//                                                         M11 unpermutes +
//                                                         combines.
//
// **Algorithm — persistent-block work-stealing on expert id.** Each block
// atomicAdd's `counters[0]` to claim an expert. For that expert it walks
// the segment `[start, end)` and runs the same per-row gate_up + silu_mul +
// down sequence the per-token FFN (`ffn_phase.cuh` Phases G/H/I) executes,
// just sourcing x from `x_norm[token_idx]` and writing to
// `expert_out[row * hidden]` instead of the persistent megakernel's
// per-top-k workspace stack.
//
// Because most experts have very few rows at chunk_size 64 (avg ~2,
// occasionally 0), running an entire block per row would waste compute.
// The block instead loops over rows serially and uses all 256 threads to
// cooperate on the per-row gate_up / down matmuls — matching the
// per-expert work-balance of the per-token FFN kernel without paying the
// per-row launch tax.
//
// **WMMA path:** for INT4 weights on RDNA3 we reuse
// `wmma_int4_matvec_partial_16rows` (helpers.cuh) the same way Phase G/I
// do. The path is selected at compile time via a template parameter; the
// bridge picks the instantiation based on `device_supports_wmma_bf16`.
//
// **LDS budget:** hidden F32 (x staging) + 2*I F32 (gate_up output) +
// I F32 (silu_mul). At hidden=2048, I=512 that's 8KiB + 4KiB + 2KiB
// = 14 KiB — under gfx1100's 64 KiB LDS/WG cap. No per-row reduction
// scratch is needed because both the WMMA and scalar matvec paths
// produce one full-row dot product per thread.
//
// Same isolation rules as the rest of qwen36_moe (CLAUDE.md): keep this
// file inside `kernels/qwen36_moe_persistent/` and don't share it with
// other model families' kernels.

#pragma once

#include "../qwen36_moe_cuda_prelude.cuh"
#include "helpers.cuh"

namespace qwen36_moe {

// Grid:  num_blocks = props.multiProcessorCount (one block per CU)
// Block: 256 threads
//
// Each block claims experts via `atomicAdd(counters, 1)`; per-expert it
// loops over its segment of permuted rows and computes the full FFN.
template <typename T, bool USE_WMMA>
__global__ void qwen36_moe_batched_prefill_grouped_expert_kernel(
    int                                  n_tokens,
    int                                  top_k,
    int                                  num_experts,
    int                                  hidden,
    int                                  moe_intermediate,    // I
    int                                  group_size,          // 128
    const T*            __restrict__     x_norm,              // [N, hidden]
    const int*          __restrict__     expert_offsets,      // [E+1]
    const int*          __restrict__     permuted_token_idx,  // [N*top_k]
    const uint8_t*      __restrict__     experts_gate_up_w,   // [E, 2I, hidden/2]
    const hip_bfloat16* __restrict__     experts_gate_up_s,
    const hip_bfloat16* __restrict__     experts_gate_up_z,
    const uint8_t*      __restrict__     experts_down_w,      // [E, hidden, I/2]
    const hip_bfloat16* __restrict__     experts_down_s,
    const hip_bfloat16* __restrict__     experts_down_z,
    T*                  __restrict__     expert_out,          // [N*top_k, hidden]
    unsigned int*       __restrict__     counters             // [1] expert-claim counter
) {
    const int tid        = threadIdx.x;
    const int block_size = static_cast<int>(blockDim.x);
    const int I_         = moe_intermediate;
    const int two_I      = 2 * I_;

    extern __shared__ float lds[];
    // Layout (F32 elements):
    //   x_lds          [hidden]                (post-norm activation as F32)
    //   gu_lds         [2 * I_]                (gate_up matmul output)
    //   silu_mul_lds   [I_]                    (silu(gate) * up)
    //
    // Both the WMMA and scalar matvec paths produce one full-row dot
    // product per thread (no cross-thread reduction needed inside a
    // row), so we don't need a `shared_scratch` slot like the per-token
    // FFN's Phase G/I.
    float* x_lds        = lds;
    float* gu_lds       = x_lds + hidden;
    float* silu_mul_lds = gu_lds + two_I;

    const int gsc_gu = hidden / group_size;       // gate_up scale columns
    const int gsr_gu = two_I  / group_size;       // gate_up scale rows
    const int gsc_dp = I_     / group_size;       // down  scale columns
    const int gsr_dp = hidden / group_size;       // down  scale rows

    // Persistent loop — claim experts via atomic counter.
    for (;;) {
        __shared__ unsigned int s_expert_id;
        if (tid == 0) {
            s_expert_id = atomicAdd(counters, 1u);
        }
        __syncthreads();
        const int e = static_cast<int>(s_expert_id);
        if (e >= num_experts) return;

        const int seg_start = expert_offsets[e];
        const int seg_end   = expert_offsets[e + 1];
        if (seg_start == seg_end) continue;   // empty segment — try next expert.

        // Pre-resolve this expert's weight slabs.
        const uint8_t* gu_slab_packed =
            experts_gate_up_w + static_cast<size_t>(e) * two_I * (hidden / 2);
        const hip_bfloat16* gu_slab_scale =
            experts_gate_up_s + static_cast<size_t>(e) * gsr_gu * gsc_gu;
        const hip_bfloat16* gu_slab_zero =
            experts_gate_up_z + static_cast<size_t>(e) * gsr_gu * gsc_gu;

        const uint8_t* dp_slab_packed =
            experts_down_w + static_cast<size_t>(e) * hidden * (I_ / 2);
        const hip_bfloat16* dp_slab_scale =
            experts_down_s + static_cast<size_t>(e) * gsr_dp * gsc_dp;
        const hip_bfloat16* dp_slab_zero =
            experts_down_z + static_cast<size_t>(e) * gsr_dp * gsc_dp;

        // Walk the assigned rows for this expert.
        for (int row = seg_start; row < seg_end; ++row) {
            const int token_idx = permuted_token_idx[row];
            const T* x_token = x_norm + static_cast<size_t>(token_idx) * hidden;
            T* out_row = expert_out + static_cast<size_t>(row) * hidden;

            // ---------------------------------------------------------------
            // Load x_token into LDS as F32, BF16-rounded so the matmul reads
            // the same bit pattern PyTorch would. Source is already a
            // post-RMSnorm BF16 tensor (caller produced it), so casting to
            // float and back is the identity in BF16 precision — the
            // explicit `bf16_round_rne_f32` keeps semantics aligned with
            // the per-token FFN's Phase A (which does its own RMSnorm in
            // F32 then BF16-rounds before storing into LDS).
            // ---------------------------------------------------------------
            for (int col = tid; col < hidden; col += block_size) {
                x_lds[col] = bf16_round_rne_f32(static_cast<float>(x_token[col]));
            }
            __syncthreads();

            // ---------------------------------------------------------------
            // Phase G port: gate_up_proj @ x → gu_lds[2*I].
            // Reuses int4_dq8_matvec_partial / wmma_int4_matvec_partial_16rows
            // exactly as ffn_phase.cuh does. Output rows are striped across
            // the block via a row-cursor counter scoped to this row's gate_up.
            //
            // Per-row counter strategy: rather than allocating an external
            // counter (the persistent FFN reuses `counters[group_id]`), we
            // walk rows explicitly with the block-size stride. That avoids
            // any cross-row dependency on the atomic counter and keeps the
            // outer expert-claim atomic at counters[0] sole-purpose.
            // ---------------------------------------------------------------
            bool wmma_handled_g = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
            if constexpr (USE_WMMA) {
                // 8 waves × 16 rows = 128 rows per pass. two_I (1024) /
                // 128 = 8 passes for production shape; the loop just keeps
                // walking until row_base >= two_I.
                const int wave_id   = tid >> 5;
                const int lane      = tid & 31;
                const int lane_row  = lane & 15;
                const int lane_half = lane >> 4;
                for (int row_base = 0; row_base < two_I; row_base += 128) {
                    const int rhs_row_idx = row_base + wave_id * 16 + lane_row;
                    const bool rhs_in_range = rhs_row_idx < two_I;
                    const uint8_t* slab_row = rhs_in_range
                        ? gu_slab_packed +
                          static_cast<size_t>(rhs_row_idx) * (hidden / 2)
                        : nullptr;

                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row, gu_slab_scale, gu_slab_zero,
                        rhs_row_idx, rhs_in_range,
                        x_lds, hidden,
                        gsc_gu, group_size, lane_row);

                    if (lane_half == 0 && rhs_in_range) {
                        gu_lds[rhs_row_idx] = acc[0];
                    }
                    __syncthreads();
                }
                wmma_handled_g = true;
            }
#endif
            if (!wmma_handled_g) {
                // Scalar path: each iteration does `block_size` rows in
                // parallel, one row per dot product fully reduced inside
                // the block. Production: two_I=1024, block_size=256 →
                // 4 iterations.
                for (int row_base = 0; row_base < two_I; row_base += block_size) {
                    const int my_row = row_base + tid;
                    float partial = 0.0f;
                    if (my_row < two_I) {
                        // int4_dq8_matvec_partial expects each lane to
                        // process its own row across all cols. Here we
                        // instead want every thread on the SAME row
                        // (one block per row), so we inline the col-stride
                        // dequant with one thread = one col.
                        const int byte_cols  = hidden / 2;
                        const int scale_cols = hidden / group_size;
                        const int scale_row  = my_row / group_size;
                        const size_t row_byte_off =
                            static_cast<size_t>(my_row) * byte_cols;
                        for (int col = 0; col < hidden; col += 8) {
                            // Single thread does this row → col iteration is
                            // serial within the lane. Caller-side: we'll
                            // strip-mine across threads on a row-major basis
                            // — see below for an alternate formulation.
                            const uint32_t pk = *reinterpret_cast<const uint32_t*>(
                                &gu_slab_packed[row_byte_off + col / 2]);
                            float dq[8];
                            int4_dequant_8(pk, gu_slab_scale, gu_slab_zero,
                                           scale_row, col, scale_cols,
                                           group_size, dq);
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                partial += dq[i] * x_lds[col + i];
                            }
                        }
                    }
                    if (my_row < two_I) {
                        gu_lds[my_row] = partial;
                    }
                    __syncthreads();
                }
            }

            // ---------------------------------------------------------------
            // Phase H port: silu(gate) * up → silu_mul_lds[I].
            // Trivial elementwise — block_size threads each do I/block_size
            // elements (production: 512/256=2 iters). Already in F32 from
            // gu_lds.
            //
            // **BF16-round on store**: the WMMA path in helpers.cuh treats
            // the activation as "BF16-rounded F32" (it constructs the WMMA
            // A-operand by taking the top 16 bits of each F32). If we
            // stored raw F32 here the WMMA tile would silently drop the
            // low 16 mantissa bits — which causes a noticeable bias at
            // small output magnitudes (random INT4 weight tests caught
            // this; the production FFN per-block parity tests check only
            // max_abs and so don't see it). Rounding at the store site
            // makes the scalar and WMMA paths bit-exact. M11's downstream
            // path will read the same BF16-quantised value either way so
            // there's no semantic loss.
            // ---------------------------------------------------------------
            for (int i = tid; i < I_; i += block_size) {
                const float gp = gu_lds[i];
                const float up = gu_lds[I_ + i];
                const float sigmoid_gp = 1.0f / (1.0f + expf(-gp));
                const float silu_gp    = gp * sigmoid_gp;
                silu_mul_lds[i] = bf16_round_rne_f32(silu_gp * up);
            }
            __syncthreads();

            // ---------------------------------------------------------------
            // Phase I port: down_proj @ silu_mul → out_row[hidden] BF16.
            // Same row-strided structure as Phase G; reduction dim is now
            // I (smaller). Final BF16-round on store matches the
            // per-token FFN's `output[my_row] = (T)bf16_round_rne_f32(val)`
            // when stage==3 and group_id==0.
            // ---------------------------------------------------------------
            bool wmma_handled_i = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
            if constexpr (USE_WMMA) {
                const int wave_id   = tid >> 5;
                const int lane      = tid & 31;
                const int lane_row  = lane & 15;
                const int lane_half = lane >> 4;
                for (int row_base = 0; row_base < hidden; row_base += 128) {
                    const int rhs_row_idx = row_base + wave_id * 16 + lane_row;
                    const bool rhs_in_range = rhs_row_idx < hidden;
                    const uint8_t* slab_row = rhs_in_range
                        ? dp_slab_packed +
                          static_cast<size_t>(rhs_row_idx) * (I_ / 2)
                        : nullptr;

                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row, dp_slab_scale, dp_slab_zero,
                        rhs_row_idx, rhs_in_range,
                        silu_mul_lds, I_,
                        gsc_dp, group_size, lane_row);

                    if (lane_half == 0 && rhs_in_range) {
                        const float val = acc[0];
                        out_row[rhs_row_idx] =
                            static_cast<T>(bf16_round_rne_f32(val));
                    }
                    __syncthreads();
                }
                wmma_handled_i = true;
            }
#endif
            if (!wmma_handled_i) {
                for (int row_base = 0; row_base < hidden; row_base += block_size) {
                    const int my_row = row_base + tid;
                    float partial = 0.0f;
                    if (my_row < hidden) {
                        const int byte_cols  = I_ / 2;
                        const int scale_cols = I_ / group_size;
                        const int scale_row  = my_row / group_size;
                        const size_t row_byte_off =
                            static_cast<size_t>(my_row) * byte_cols;
                        for (int col = 0; col < I_; col += 8) {
                            const uint32_t pk = *reinterpret_cast<const uint32_t*>(
                                &dp_slab_packed[row_byte_off + col / 2]);
                            float dq[8];
                            int4_dequant_8(pk, dp_slab_scale, dp_slab_zero,
                                           scale_row, col, scale_cols,
                                           group_size, dq);
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                partial += dq[i] * silu_mul_lds[col + i];
                            }
                        }
                    }
                    if (my_row < hidden) {
                        out_row[my_row] =
                            static_cast<T>(bf16_round_rne_f32(partial));
                    }
                    __syncthreads();
                }
            }

            // (No __syncthreads needed at the loop tail — the next row
            // start with x_lds load, which itself begins with __syncthreads
            // after the cooperative store.)
        }
    }
}

}  // namespace qwen36_moe
