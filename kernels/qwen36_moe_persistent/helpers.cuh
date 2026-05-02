// Shared __device__ primitives for the Qwen3.6-MoE persistent megakernel.
//
// Phase 3a refactor: pulled verbatim out of `kernels/qwen36_moe.hip` so
// that the upcoming phase-extraction headers (full_attn_phase.cuh,
// linear_attn_phase.cuh, ffn_phase.cuh) can `#include` exactly the same
// helpers without circular dependencies on the step kernels. No behavior
// changes — the existing 26+ per-block parity tests in
// `crates/kernel-ffi/src/qwen36_moe.rs::tests::*` cover correctness.
//
// CLAUDE.md is non-negotiable about isolation: hipcc on gfx11xx is
// fragile and merging Qwen3.6-MoE with full_attention*.hip caused codegen
// regressions. These helpers are cross-pollinated *by intent* from
// `kernels/full_attention_4b.hip` PRs #4b5/#77/#78 but live in their own
// namespace and translation unit; they are NOT shared with the qwen35
// kernels.
//
// The numerics match `oracle/bake_int4.py`'s reconstruction exactly:
//   recon = bf16_round_rne_f32(nibble * scale - zero * scale)
// (single rounding through BF16 at the end).

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_moe {

// Round an F32 value to BF16 precision (round-to-nearest-even) and widen
// back to F32. Used at every "BF16 store" point so that intermediate F32
// computation matches the BF16-quantised values PyTorch produces. Finite
// inputs only — qwen36_moe is downstream of upcasted-from-BF16 weights and
// inputs, so NaN preservation is not needed here.
__device__ __forceinline__ float bf16_round_rne_f32(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    float y;
    __builtin_memcpy(&y, &bits, 4);
    return y;
}

// ---------------------------------------------------------------------------
// INT4 group-quant dequant — ported from kernels/full_attention_4b.hip
// (PR 4b5). Per CLAUDE.md the qwen36_moe and full_attention_4b kernels are
// isolated compilation units (hipcc on gfx11xx is fragile), so the helpers
// live here as a copy rather than a shared header.
//
// Layout (matches the bake + the full_attention_4b versions):
//   weights : `[..., out, in/2]` u8   — 2 nibbles/byte, even col → low.
//   scale   : `[..., out/gs, in/gs]` BF16
//   zero    : `[..., out/gs, in/gs]` BF16

// Dequantize 8 INT4 weights from 4 packed bytes.
// `packed` carries 8 nibbles in the same nibble-order
// `int4_dequant_8` in full_attention_4b uses (low nibble = even col).
// `col` is the starting input column for the 8-element span (must be
// 2-aligned). `scale_row = row / gsz`, `scale_cols = (cols + gsz - 1) / gsz`,
// `sb = scale_row * scale_cols`. Fast path covers the common case where all
// 8 columns share a group (gs=128 + 8-element step ⇒ true except at group
// boundaries).
__device__ inline void int4_dequant_8(
    uint32_t packed,
    const hip_bfloat16* __restrict__ scales,
    const hip_bfloat16* __restrict__ zeros,
    int scale_row, int col, int scale_cols, int gsz,
    float out[8]
) {
    const int sb = scale_row * scale_cols;
    int n0 = (packed >>  0) & 0xF;
    int n1 = (packed >>  4) & 0xF;
    int n2 = (packed >>  8) & 0xF;
    int n3 = (packed >> 12) & 0xF;
    int n4 = (packed >> 16) & 0xF;
    int n5 = (packed >> 20) & 0xF;
    int n6 = (packed >> 24) & 0xF;
    int n7 = (packed >> 28) & 0xF;

    const int g0 = col / gsz;
    const int g7 = (col + 7) / gsz;
    if (g0 == g7) {
        float s = static_cast<float>(scales[sb + g0]);
        float zs = static_cast<float>(zeros[sb + g0]) * s;
        out[0] = bf16_round_rne_f32(static_cast<float>(n0) * s - zs);
        out[1] = bf16_round_rne_f32(static_cast<float>(n1) * s - zs);
        out[2] = bf16_round_rne_f32(static_cast<float>(n2) * s - zs);
        out[3] = bf16_round_rne_f32(static_cast<float>(n3) * s - zs);
        out[4] = bf16_round_rne_f32(static_cast<float>(n4) * s - zs);
        out[5] = bf16_round_rne_f32(static_cast<float>(n5) * s - zs);
        out[6] = bf16_round_rne_f32(static_cast<float>(n6) * s - zs);
        out[7] = bf16_round_rne_f32(static_cast<float>(n7) * s - zs);
    } else {
        float s0 = static_cast<float>(scales[sb + g0]);
        float zs0 = static_cast<float>(zeros[sb + g0]) * s0;
        float s1 = static_cast<float>(scales[sb + g7]);
        float zs1 = static_cast<float>(zeros[sb + g7]) * s1;
        #define I4DQ(idx, ni) do { \
            int gi = (col + idx) / gsz; \
            float si = (gi == g0) ? s0 : s1; \
            float zsi = (gi == g0) ? zs0 : zs1; \
            out[idx] = bf16_round_rne_f32(static_cast<float>(ni) * si - zsi); \
        } while(0)
        I4DQ(0, n0); I4DQ(1, n1); I4DQ(2, n2); I4DQ(3, n3);
        I4DQ(4, n4); I4DQ(5, n5); I4DQ(6, n6); I4DQ(7, n7);
        #undef I4DQ
    }
}

// RDNA3 WMMA BF16 gate. Detects gfx11xx variants where
// `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` is supported. Used by both
// the lm_head WMMA kernel and the FFN per-expert WMMA path.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__)
#define SUPERSONIC_QWEN36_HAS_WMMA_BF16 1
#endif

#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
typedef short  qwen36_short16 __attribute__((ext_vector_type(16)));
typedef float  qwen36_float8  __attribute__((ext_vector_type(8)));
#endif

// Work-stealing matvec inner loop for an INT4 slab `[out_rows, in_cols/2]`.
//
// Computes the partial dot product `sum_i (dequant(w[my_row, i]) * x[i])`
// for the cols this thread is responsible for, using the 8-wide
// `int4_dequant_8` helper instead of the scalar path.
//
// Caller pre-resolves the slab pointers (so the same helper drives both 2D
// dense and 3D fused-expert matmuls — for 3D, just pass the per-expert
// slab pointers). `cols` must be a multiple of 8 (the 8-element span unit)
// AND of `gsz` (so `scale_cols = cols / gsz` is well-defined). Each thread
// processes 8 cols per iteration starting at `tid * 8`, striding by
// `block_size * 8`. Caller must ensure `block_size * 8` divides cleanly
// (hidden=2048 with block_size=256 fits exactly with 1 iter/thread).
//
// 4-byte-aligned `pk` load is safe because:
//   - byte_cols = cols / 2 is a multiple of 4 (cols is a multiple of 8)
//   - col_start / 2 is a multiple of 4 (col_start is in steps of 8)
//   so the resulting `&packed[my_row * byte_cols + col_start/2]` is always
//   4-byte aligned.
__device__ inline float int4_dq8_matvec_partial(
    const uint8_t*      __restrict__ packed,
    const hip_bfloat16* __restrict__ scales,
    const hip_bfloat16* __restrict__ zeros,
    const float*        __restrict__ x,
    int my_row, int cols, int gsz,
    int tid, int block_size
) {
    const int byte_cols  = cols / 2;
    const int scale_cols = cols / gsz;
    const int scale_row  = my_row / gsz;
    const size_t row_byte_off = static_cast<size_t>(my_row) * byte_cols;

    float partial = 0.0f;
    for (int col_start = tid * 8; col_start < cols; col_start += block_size * 8) {
        const uint32_t pk = *reinterpret_cast<const uint32_t*>(
            &packed[row_byte_off + col_start / 2]);
        float dq[8];
        int4_dequant_8(pk, scales, zeros,
                       scale_row, col_start, scale_cols, gsz, dq);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            partial += dq[i] * x[col_start + i];
        }
    }
    return partial;
}

#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
// WMMA INT4 GEMV tile: each invocation runs one wave32's worth of work,
// computing 16 output rows of `acc[0]` (one row of the 16x16 WMMA C tile)
// across the full hidden dimension via `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.
//
// Caller responsibility: invoke from a wave32 (block_size must be a
// multiple of 32; this helper reads `lane = threadIdx.x & 31`,
// `lane_row = lane & 15`, `lane_half = lane >> 4`).
//
// Math (matches `int4_dq8_matvec_partial` numerically up to F32
// accumulation order):
//   For each k-chunk of 16 in [0, hidden):
//     A operand = [activation[kk..kk+16], 0...] (only M=0 row real,
//                 m=1 GEMV pads M=1..15 with zero)
//     B operand = [16 dequanted rows × 16 cols] of int4 weights at
//                 (rhs_row_idx + lane_row, kk..kk+16) — same dequant
//                 formula as int4_dequant_8 (`bf16_round_rne_f32(nibble*s - z*s)`)
//     acc      += A × B  (F32 accumulate, K=16 per WMMA)
//
// Returns: lanes 0..15 (lane_half==0) hold acc[0] = C[0, lane_row], the
// GEMV result for output row `rhs_row_idx`. Caller writes
// `workspace[out_off + rhs_row_idx] = acc[0]` for those lanes.
//
// `slab_packed`     : full INT4-packed row base for THIS lane's output
//                     row. Caller must compute
//                     `slab_packed_base + (size_t)rhs_row_idx * (hidden/2)`
//                     and pass it (or pass nullptr if rhs_in_range==false).
// `slab_scale`,
// `slab_zero`       : scale/zero slab origin shared across the tile (BF16).
//                     Indexed as `slab_scale[(rhs_row_idx/gs) * gsc + (kk/gs)]`.
// `gsc`             : scale columns = hidden / group_size (precomputed).
// `group_size`      : INT4 quant group size (e.g. 128).
// `rhs_in_range`    : false → A and B both zero (kept for divergence-free
//                     branches; result `acc[0]` will be 0).
// `x_norm_lds_f32`  : LDS-resident BF16-rounded F32 activation (top 16 bits
//                     of each float ARE the BF16 bit pattern).
// `hidden`          : reduction dim. Must be a multiple of 16 (caller checks).
__device__ inline qwen36_float8 wmma_int4_matvec_partial_16rows(
    const uint8_t*      __restrict__ slab_packed_row,    // [hidden/2] u8 (this lane's row, may be nullptr)
    const hip_bfloat16* __restrict__ slab_scale,         // [.] BF16 (slab origin)
    const hip_bfloat16* __restrict__ slab_zero,          // [.] BF16
    int                              rhs_row_idx,
    bool                             rhs_in_range,
    const float*        __restrict__ x_norm_lds_f32,
    int                              hidden,
    int                              gsc,
    int                              group_size,
    int                              lane_row
) {
    qwen36_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int gsr_idx = rhs_row_idx / group_size;

    for (int kk = 0; kk < hidden; kk += 16) {
        // A operand: lane_row==0 carries activation; rows 1..15 are zero
        // (m=1 GEMV — we waste 15/16 of the 16x16x16 muls, but the path is
        // bandwidth-bound on B so the wasted compute is free).
        qwen36_short16 a;
        if (lane_row == 0) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint32_t bits;
                float v = x_norm_lds_f32[kk + i];
                __builtin_memcpy(&bits, &v, 4);
                a[i] = static_cast<short>(bits >> 16);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        // B operand: each lane reads its own output row's 16-element K-slab,
        // dequanting INT4 (8 bytes -> 16 BF16) using one shared scale/zero
        // for the whole K-chunk (caller guarantees group_size is a multiple
        // of 16 so all 16 K elements live in the same group).
        qwen36_short16 b;
        if (rhs_in_range) {
            const int gsc_idx   = kk / group_size;
            const int scale_idx = gsr_idx * gsc + gsc_idx;
            const float s  = static_cast<float>(slab_scale[scale_idx]);
            const float z  = static_cast<float>(slab_zero[scale_idx]);
            const float zs = z * s;
            const int byte_base = kk >> 1;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint8_t pk = slab_packed_row[byte_base + i];
                int n0 = pk & 0xF;
                int n1 = (pk >> 4) & 0xF;
                float v0 = bf16_round_rne_f32(static_cast<float>(n0) * s - zs);
                float v1 = bf16_round_rne_f32(static_cast<float>(n1) * s - zs);
                uint32_t b0_bits, b1_bits;
                __builtin_memcpy(&b0_bits, &v0, 4);
                __builtin_memcpy(&b1_bits, &v1, 4);
                b[2 * i]     = static_cast<short>(b0_bits >> 16);
                b[2 * i + 1] = static_cast<short>(b1_bits >> 16);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    return acc;
}
#endif // SUPERSONIC_QWEN36_HAS_WMMA_BF16

// Single-element dequant for non-8-aligned tails. `cols` is the unpacked
// input dim of the 2D slab `(row, col)` indexes into. `scales`/`zeros`
// are at `[(row/gs) * ((cols + gs - 1)/gs) + col/gs]` — same as the 2D
// helpers in full_attention_4b.
__device__ inline float int4_dequant_scalar(
    const void* w_ptr, const void* scale_ptr, const void* zero_ptr,
    int row, int col, int cols, int group_size
) {
    const uint8_t* data = static_cast<const uint8_t*>(w_ptr);
    int byte_cols = cols / 2;
    uint8_t packed_byte = data[static_cast<size_t>(row) * byte_cols + col / 2];
    int nibble = (col & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    const hip_bfloat16* zeros = static_cast<const hip_bfloat16*>(zero_ptr);
    int si = (row / group_size) * ((cols + group_size - 1) / group_size)
           + col / group_size;
    float s = static_cast<float>(scales[si]);
    return bf16_round_rne_f32(
        static_cast<float>(nibble) * s - static_cast<float>(zeros[si]) * s);
}

__device__ inline float fp8_e4m3_to_float(uint8_t byte) {
    const int sign = (byte & 0x80) ? -1 : 1;
    const int exp = (byte >> 3) & 0x0F;
    const int mant = byte & 0x07;
    if (exp == 0) {
        if (mant == 0) return sign < 0 ? -0.0f : 0.0f;
        return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
    if (exp == 0x0F) {
        return sign * 448.0f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

__device__ inline float fp8_dequant_scalar(
    const void* w_ptr, const void* scale_ptr,
    int row, int col, int cols, int block_size
) {
    const uint8_t* data = static_cast<const uint8_t*>(w_ptr);
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    const int scale_cols = (cols + block_size - 1) / block_size;
    const int si = row * scale_cols + col / block_size;
    const uint8_t byte = data[static_cast<size_t>(row) * cols + col];
    return bf16_round_rne_f32(fp8_e4m3_to_float(byte) * static_cast<float>(scales[si]));
}

__device__ inline float fp8_matvec_partial(
    const void* w_ptr, const void* scale_ptr, const float* __restrict__ x,
    int row, int cols, int block_size,
    int tid, int block_threads
) {
    float partial = 0.0f;
    for (int col = tid; col < cols; col += block_threads) {
        partial += fp8_dequant_scalar(w_ptr, scale_ptr, row, col, cols, block_size) * x[col];
    }
    return partial;
}

// -- Grid barrier (verbatim from full_attention_4b.hip)
//
// Acquire/release ordering on `barrier_flag` makes this safe across CUs.
// Block 0..N-1 each atomicAdd `barrier_counter`; the last arrival increments
// the phase, others spin on a monotonic phase comparison. Coupled with
// the launch using `multiProcessorCount` blocks (i.e. one block per CU,
// all guaranteed concurrent on RDNA3), no cooperative-launch flag is
// needed — same as the production qwen35-4b path.

__device__ inline void grid_barrier(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __threadfence();
            __atomic_store_n(barrier_flag, phase + 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

__device__ inline void grid_barrier_reset_counter(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks,
    unsigned int* counter_to_reset
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            __atomic_store_n(counter_to_reset, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __threadfence();
            __atomic_store_n(barrier_flag, phase + 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

// Generic "load element idx from a typed pointer as F32". Used pervasively
// in step kernels to abstract over BF16 / hip_bfloat16 / FP32 weight types.
template <typename T>
__device__ inline float load_as_float(const T* p, int idx) {
    return static_cast<float>(p[idx]);
}

}  // namespace qwen36_moe
