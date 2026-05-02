// Qwen3.6-MoE persistent decode megakernel — PR 4 infrastructure stub.
//
// Status: descriptor layout + stub launch only. The real megakernel
// (~6500 LoC of attention + MoE compute) lands in follow-up PRs.
//
// What this file provides today:
// 1. C++ struct definitions that mirror the Rust FFI types in
//    `crates/kernel-ffi/src/qwen36_moe.rs`. A `static_assert` in the
//    bridge pins the size on x86_64.
// 2. A descriptor-validation kernel that walks the layer array, summarises
//    integrity facts, and writes them to the workspace at known offsets.
//    The host smoke test reads these back to confirm the Rust↔C++
//    descriptor layout is in sync.
// 3. The grid-barrier primitive ported verbatim from `full_attention_4b.hip`
//    (proven correct under cooperative launch on gfx1100). Same primitive
//    the real kernel will use for cross-layer synchronisation and for the
//    Option-C work-stealing dispatch.
//
// CLAUDE.md is non-negotiable about isolation: hipcc on gfx11xx is
// fragile, merging this with full_attention*.hip caused codegen
// regressions in the past. Keep this file standalone.

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
// live here as a copy rather than a shared header. The numerics match
// `oracle/bake_int4.py`'s reconstruction exactly:
//   recon = bf16_round_rne_f32(nibble * scale - zero * scale)
// (single rounding through BF16 at the end).
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

// -- Layer descriptor (mirror of Rust Qwen36MoeDecodeLayerDesc) ----------

struct DecodeLayerDesc {
    int  layer_idx;
    int  is_full_attention;

    // RMS norms
    const void* input_norm_w;
    float       input_norm_eps;
    const void* post_attn_norm_w;
    float       post_attn_norm_eps;

    // Full-attention slots (read iff is_full_attention == 1)
    const void* q_proj_w;
    int         q_proj_out_dim;
    int         attn_output_gate;
    const void* k_proj_w;
    const void* v_proj_w;
    const void* o_proj_w;
    const void* q_norm_w;
    const void* k_norm_w;
    int         attn_head_dim;
    int         attn_num_heads;
    int         attn_num_kv_heads;
    void*       kv_cache_k;
    void*       kv_cache_v;
    int         kv_len;
    int         kv_max_t;

    // Linear-attention slots (read iff is_full_attention == 0)
    const void* linear_in_proj_qkv_w;
    const void* linear_in_proj_z_w;
    const void* linear_in_proj_b_w;
    const void* linear_in_proj_a_w;
    const void* linear_out_proj_w;
    const void* linear_conv1d_w;
    const void* linear_dt_bias;
    const void* linear_a_log_exp;
    const void* linear_norm_w;
    int         linear_qkv_dim;
    int         linear_v_dim;
    int         linear_v_heads;
    int         linear_conv_kernel_dim;
    void*       linear_conv_state;
    void*       linear_recurrent_state;

    // MoE block (always read)
    const void* router_w;
    const void* experts_gate_up_w;
    const void* experts_down_w;
    const void* shared_expert_gate_proj_w;
    const void* shared_expert_up_proj_w;
    const void* shared_expert_down_proj_w;
    const void* shared_expert_gate_w;
    int         num_experts;
    int         top_k;
    int         moe_intermediate_size;
    int         shared_expert_intermediate_size;
    int         norm_topk_prob;
};

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

// -- Descriptor-validation stub kernel
//
// Each block grabs the next layer index from `counters[0]` (work-stealing)
// and contributes to global sentinel counters in `workspace`. Future kernel
// work replaces the inner loop body with actual decode compute; the
// outer iteration shape stays.

__global__ void qwen36_moe_descriptor_walk_stub(
    int                              num_layers,
    const DecodeLayerDesc* __restrict__ layers,
    float* __restrict__              workspace,
    unsigned int* __restrict__       counters,
    unsigned int* __restrict__       barrier_counter,
    unsigned int* __restrict__       barrier_flag) {
    const int num_blocks = static_cast<int>(gridDim.x);
    const int bid        = blockIdx.x;

    // Sentinel slots (only block 0 / thread 0 writes the initial values).
    // Match `kernel-ffi/src/qwen36_moe.rs::stub_launch` doc.
    if (bid == 0 && threadIdx.x == 0) {
        workspace[0] = 0.0f; // num_layers seen
        workspace[1] = 0.0f; // total experts
        workspace[2] = 0.0f; // total top_k
        workspace[3] = 1.0f; // hybrid pattern OK (zeroed by any failing block)
        workspace[4] = 1.0f; // attn_output_gate consistent
    }
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

    // Cooperative work-stealing over layers. Each block grabs the next
    // layer index from `counters[0]`. Same primitive the full kernel
    // will use to dispatch (token, expert) work units later. Shared-memory
    // broadcast of the claimed index is portable across any block_size.
    __shared__ unsigned int s_layer_idx;
    for (;;) {
        if (threadIdx.x == 0) {
            s_layer_idx = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const unsigned int layer_idx = s_layer_idx;
        if (layer_idx >= static_cast<unsigned int>(num_layers)) break;

        const DecodeLayerDesc& d = layers[layer_idx];

        if (threadIdx.x == 0) {
            atomicAdd(&workspace[0], 1.0f);
            atomicAdd(&workspace[1], static_cast<float>(d.num_experts));
            atomicAdd(&workspace[2], static_cast<float>(d.top_k));

            // Hybrid pattern check: layers (1..)%4==0 should be full.
            const int expect_full = (((d.layer_idx + 1) % 4) == 0) ? 1 : 0;
            if (d.is_full_attention != expect_full) {
                workspace[3] = 0.0f;
            }
            // attn_output_gate consistency: only meaningful on full layers.
            if (d.is_full_attention == 1 && d.attn_output_gate != 1) {
                workspace[4] = 0.0f;
            }
        }
        __syncthreads();
    }

    // Reset the work counter for any future barriers (idempotent here).
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
}

// =============================================================================
// Single-layer full-attention parity kernel (PR 4b2)
//
// This is the staged build-up of the real megakernel's attention path. Each
// stage adds one slice of compute — the host parity test downloads the
// matching intermediate buffer and compares it to the PyTorch oracle.
//
//   stage 1: x_norm + q_proj + q-gate split + per-head q_norm  → write q_normed
//   stage 2: ... + k_proj + v_proj + per-head k_norm           → write k_normed
//   stage 3: ... + partial-RoPE on q & k                       → write q_rot/k_rot
//   stage 4: ... + GQA self-attention (kv_len=1)               → write attn
//   stage 5: ... + sigmoid(out_gate)*attn + o_proj + residual  → write out
//
// PR 4b2 step 1 lands stage 1. Stages 2–5 ship in follow-up commits and add
// to this same kernel; the launcher's `stage` argument decides where to stop.
//
// Layout choices fixed up-front so later stages don't reshuffle:
//   block_size = 256  (4 waves on gfx1100; matches head_dim and divides
//                     hidden=2048 evenly)
//   grid_size  = num_cus  (one block per CU, all guaranteed concurrent on
//                          RDNA3 → grid_barrier safe without coop launch)
//
//   workspace[F32], staged-attention scratch (offsets in F32 elements):
//     Q_RAW    = 0                                [2*H*d]   q_proj output
//     K_RAW    = 2*H*d                            [Hkv*d]   k_proj output
//     V_RAW    = 2*H*d + Hkv*d                    [Hkv*d]   v_proj output
//     Q_NORMED = 2*H*d + 2*Hkv*d                  [H*d]     stage-1 result
//     K_NORMED = 2*H*d + 2*Hkv*d + H*d            [Hkv*d]   stage-2 result
//     Q_ROT    = 2*H*d + 2*Hkv*d + H*d + Hkv*d    [H*d]     stage-3 result
//     K_ROT    = 3*H*d + 2*Hkv*d + H*d + Hkv*d    [Hkv*d]   stage-3 result
//     ATTN     = 4*H*d + 4*Hkv*d                  [H*d]     stage-4 result
//     GATED    = 5*H*d + 4*Hkv*d                  [H*d]     sigmoid(gate)*attn
//     O_OUT    = 6*H*d + 4*Hkv*d                  [hidden]  o_proj output
//   Total stage-5 footprint: 6*H*d + 4*Hkv*d + hidden F32 elements.
//   For 35B-A3B (H=16, Hkv=2, d=256, hidden=2048) that's 28,672 F32 = 112 KiB.
//
//   counters[0]: work-stealing index, reused across phases. Reset by
//                grid_barrier_reset_counter at every phase boundary.
//
//   LDS layout (extern __shared__):
//     lds[0,        hidden):           x_norm_lds (BF16-rounded F32 view)
//     lds[hidden,   hidden+block_size): scratch (block reduction)
//   Total: (hidden + block_size) * 4 bytes = (2048+256)*4 = 9 KiB on 35B-A3B.
//   Comfortably under gfx1100's 64 KiB LDS/WG cap; ≥2 WGs/CU possible.

template <typename T>
__device__ inline float load_as_float(const T* p, int idx) {
    return static_cast<float>(p[idx]);
}

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase B (q_proj GEMV),
// Phase D (fused k_proj/v_proj GEMV — split into two WMMA sub-pools),
// and Phase I (o_proj GEMV + residual). Same `wmma_int4_matvec_partial_16rows`
// helper as PRs #77 and #78. Bridge picks the instantiation at launch
// time based on `device_supports_wmma_bf16(ordinal)` + dim divisibility.
template <typename T, bool USE_WMMA = false>
__global__ void qwen36_moe_attn_step_kernel(
    int                            stage,
    int                            hidden,
    int                            num_heads,
    int                            num_kv_heads,
    int                            head_dim,
    int                            rotary_dim,    // unused at stage 1, parsed at stage 3+
    float                          rope_theta,    // unused at stage 1
    float                          rms_norm_eps,
    int                            position,      // unused at stage 1
    // Optional override for the KV cache slot index. Negative ⇒ "same as
    // `position`" (the base-model decode case where RoPE position == cache
    // slot). Set to a non-negative value to decouple the two — used by
    // self-speculative MTP, where RoPE rotates at absolute position
    // `base_seq_len + k` but the per-MTP-session cache writes at slot
    // `k` (the cache is fresh per draft session). When `cache_pos >= 0`,
    // Phase G writes K/V at `cache_pos` and attends over
    // `kv_len = cache_pos + 1`; RoPE still uses `position`.
    int                            cache_pos,
    const T* __restrict__          input_hidden,  // [hidden]
    const T* __restrict__          input_norm_w,  // [hidden]
    const T* __restrict__          q_proj_w,      // [2*H*d, hidden]
    const T* __restrict__          k_proj_w,      // [Hkv*d, hidden]   (nullable for stage<2)
    const T* __restrict__          v_proj_w,      // [Hkv*d, hidden]   (nullable for stage<2)
    const T* __restrict__          q_norm_w,      // [d]
    const T* __restrict__          k_norm_w,      // [d]               (nullable for stage<2)
    const T* __restrict__          o_proj_w,      // [hidden, H*d]     (nullable for stage<5)
    // PR 4b6 INT4 sidecars. `int4_group_size == 0` ⇒ all weights are BF16
    // and every sidecar pointer must be null. When non-zero, each sidecar
    // pair (`*_scale`, `*_zero`) is consulted independently per tensor: a
    // null `*_scale` for a given tensor keeps it on the BF16 path, a
    // non-null pair switches that tensor to INT4. The kernel reinterprets
    // the matching weight pointer as `const uint8_t*` packed bytes (low
    // nibble = even col) and dequants per-element via `int4_dequant_scalar`.
    int                                  int4_group_size,
    const hip_bfloat16* __restrict__     q_proj_scale,
    const hip_bfloat16* __restrict__     q_proj_zero,
    const hip_bfloat16* __restrict__     k_proj_scale,
    const hip_bfloat16* __restrict__     k_proj_zero,
    const hip_bfloat16* __restrict__     v_proj_scale,
    const hip_bfloat16* __restrict__     v_proj_zero,
    const hip_bfloat16* __restrict__     o_proj_scale,
    const hip_bfloat16* __restrict__     o_proj_zero,
    T* __restrict__                output,        // BF16, sized for the largest staged intermediate
    float* __restrict__             workspace,    // F32 scratch, see header comment
    // PR 4d KV cache extension. When `kv_cache_k`/`kv_cache_v` are non-null
    // and `kv_max_t > 0`, Phase G writes the current step's K/V at slot
    // `eff_cache_pos = (cache_pos >= 0 ? cache_pos : position)` and attends
    // over `kv_len = eff_cache_pos + 1` past tokens. When both pointers are
    // null, falls back to the kv_len=1 self-attention path (PR 4b2 behavior —
    // back-compat for the parity tests).
    T* __restrict__                kv_cache_k,    // [kv_max_t, Hkv*d] BF16 (mut, nullable)
    T* __restrict__                kv_cache_v,    // [kv_max_t, Hkv*d] BF16 (mut, nullable)
    int                            kv_max_t,      // capacity bound; 0 ⇔ no cache
    unsigned int* __restrict__     counters,
    unsigned int* __restrict__     barrier_counter,
    unsigned int* __restrict__     barrier_flag) {
    (void)k_proj_w;
    (void)v_proj_w;
    (void)k_norm_w;
    (void)o_proj_w;
    (void)rotary_dim;
    (void)rope_theta;
    // `position` is now read in Phase G when KV cache is enabled.

    const int num_blocks = static_cast<int>(gridDim.x);
    const int tid        = threadIdx.x;
    const int block_size = static_cast<int>(blockDim.x);

    extern __shared__ float lds[];
    float* x_norm_lds    = lds;             // [hidden]
    float* shared_scratch = lds + hidden;   // [block_size]

    // -- Phase A: load input + RMS norm ------------------------------------
    // Every block computes x_norm into its own LDS. Redundant compute, but
    // avoids a pre-stage cross-block sync; the RMS reduction is cheap
    // compared to the matmul that follows.
    for (int col = tid; col < hidden; col += block_size) {
        x_norm_lds[col] = load_as_float<T>(input_hidden, col);
    }
    __syncthreads();

    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        partial_sq += x_norm_lds[col] * x_norm_lds[col];
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float inv_rms_input;
    if (tid == 0) {
        inv_rms_input = rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // BF16-round each x_norm element so the matmul reads what PyTorch reads
    // (its `out.to(in_dtype)` step in rms_norm). HF `Qwen3_5MoeRMSNorm`
    // applies the `(1.0 + weight)` unit offset (line 819 of
    // modeling_qwen3_5_moe.py); `input_layernorm` is one such instance.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(input_norm_w, col);
        x_norm_lds[col] = bf16_round_rne_f32(x_norm_lds[col] * inv_rms_input * (1.0f + w));
    }
    __syncthreads();

    // -- Phase B: q_raw = q_proj_w @ x_norm  (work-stealing matvec) --------
    // Output rows of q_raw are written BF16-rounded into workspace[0..2*H*d]
    // so that the per-head RMS norm in Phase C reads the same F32-of-BF16
    // values PyTorch's BF16 q_raw represents.
    // INT4 (when `q_proj_scale != nullptr`): treat `q_proj_w` as
    // `[2*H*d, hidden/2]` u8 and dequant per element. Same matvec stride.
    const int q_out_dim = 2 * num_heads * head_dim;
    const bool q_int4 = (q_proj_scale != nullptr);

    bool wmma_handled_qproj = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    if constexpr (USE_WMMA) {
        if (q_int4) {
            // WMMA tile path — 128 rows per atomicAdd, 8 waves × 16-row
            // tiles. Same `wmma_int4_matvec_partial_16rows` helper as the
            // FFN/linear-attn WMMA paths; output is BF16-rounded to match
            // the scalar path's `workspace[my_row] = bf16_round_rne_f32(...)`.
            const int gsc_q = hidden / int4_group_size;
            const int wave_id = tid >> 5;
            const int lane_in_wave = tid & 31;
            const int lane_row = lane_in_wave & 15;
            const int lane_half = lane_in_wave >> 4;
            const uint8_t* slab_packed =
                reinterpret_cast<const uint8_t*>(q_proj_w);
            for (;;) {
                __shared__ unsigned int row_base_s;
                if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                __syncthreads();
                const int row_base = static_cast<int>(row_base_s);
                if (row_base >= q_out_dim) break;

                const int rhs_row = row_base + wave_id * 16 + lane_row;
                const bool in_range = rhs_row < q_out_dim;
                const uint8_t* slab_row = in_range
                    ? slab_packed +
                      static_cast<size_t>(rhs_row) * (hidden / 2)
                    : nullptr;
                qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                    slab_row,
                    static_cast<const hip_bfloat16*>(q_proj_scale),
                    static_cast<const hip_bfloat16*>(q_proj_zero),
                    rhs_row, in_range,
                    x_norm_lds, hidden,
                    gsc_q, int4_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[rhs_row] = bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
            wmma_handled_qproj = true;
        }
    }
#endif
    if (!wmma_handled_qproj) {
        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) {
                my_row_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= q_out_dim) break;

            float partial = 0.0f;
            if (q_int4) {
                // 8-wide INT4 dequant + dot accumulate. ~2-4× faster than the
                // scalar path on RDNA3 because it amortizes scale/zero lookups
                // across the 8-element span and folds 4 nibble-byte reads into
                // a single 4-byte uint32 load.
                partial = int4_dq8_matvec_partial(
                    reinterpret_cast<const uint8_t*>(q_proj_w),
                    static_cast<const hip_bfloat16*>(q_proj_scale),
                    static_cast<const hip_bfloat16*>(q_proj_zero),
                    x_norm_lds,
                    my_row, hidden, int4_group_size,
                    tid, block_size);
            } else {
                const T* w_row = q_proj_w + static_cast<size_t>(my_row) * hidden;
                for (int col = tid; col < hidden; col += block_size) {
                    partial += load_as_float<T>(w_row, col) * x_norm_lds[col];
                }
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                workspace[my_row] = bf16_round_rne_f32(shared_scratch[0]);
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // Workspace offsets (F32 elements). Kept in locals so the layout is
    // documented next to the code that uses it.
    const int H        = num_heads;
    const int Hkv      = num_kv_heads;
    const int d        = head_dim;
    const int OFF_Q_RAW    = 0;
    const int OFF_K_RAW    = 2 * H * d;
    const int OFF_V_RAW    = 2 * H * d + Hkv * d;
    const int OFF_Q_NORMED = 2 * H * d + 2 * Hkv * d;
    const int OFF_K_NORMED = 2 * H * d + 2 * Hkv * d + H * d;
    const int OFF_Q_ROT    = 2 * H * d + 2 * Hkv * d + H * d + Hkv * d;
    const int OFF_K_ROT    = 3 * H * d + 2 * Hkv * d + H * d + Hkv * d;
    const int OFF_ATTN     = 4 * H * d + 4 * Hkv * d;
    const int OFF_GATED    = 5 * H * d + 4 * Hkv * d;
    const int OFF_O_OUT    = 6 * H * d + 4 * Hkv * d;
    // Scores for the attention reduction when kv_len > 1. Per Q head we
    // need [kv_len] F32 slots; allocate per-head slots [H, kv_max_t] so
    // blocks processing different heads don't race. When kv_cache_k is
    // null (back-compat kv_len=1 path) this region is unused — the engine
    // doesn't need to allocate space for it.
    const int OFF_SCORES   = 6 * H * d + 4 * Hkv * d + hidden;

    // -- Phase C: per-head RMS norm of q lanes ----------------------------
    //
    // The q_proj output at `workspace[OFF_Q_RAW..OFF_Q_RAW+2*H*d]` carries
    // BF16-rounded rows of `q_proj_w @ x_norm` in their natural HF layout:
    // per-head interleaved `[q_h | gate_h]` because HF reshapes via
    // `q_proj(x).view(..., H, head_dim*2).chunk(2, dim=-1)` (line 672 of
    // modeling_qwen3_5_moe.py). For head h the q values live at offset
    // `h*2*d + i` (i in [0, d)) and the output gate at `h*2*d + d + i`.
    //
    // Each block grabs a head index, reduces across head_dim, writes BF16
    // q_normed to a *tightly packed* `OFF_Q_NORMED + h*d` so the RoPE +
    // attention readers downstream see a `[H, d]` q array. If stage == 1
    // we also publish q_normed to the host-visible `output` buffer.
    {
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(head_s);
            if (h >= H) break;

            const int q_in_base  = h * 2 * d;   // HF interleaved q half
            const int q_out_base = h * d;       // tightly packed q_normed
            float partial = 0.0f;
            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_Q_RAW + q_in_base + i];
                partial += v * v;
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            __shared__ float head_inv_rms;
            if (tid == 0) {
                head_inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(d) + rms_norm_eps);
            }
            __syncthreads();

            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_Q_RAW + q_in_base + i];
                const float wv = load_as_float<T>(q_norm_w, i);
                // HF `Qwen3_5MoeRMSNorm` `(1.0 + weight)` unit offset.
                const float normed = bf16_round_rne_f32(v * head_inv_rms * (1.0f + wv));
                // Round-trip through BF16 representation: store BF16 in
                // workspace as F32 so the next-stage RoPE reader sees the
                // exact dtype-cast values PyTorch's `.to(bf16)` produces.
                workspace[OFF_Q_NORMED + q_out_base + i] = normed;
                if (stage == 1) {
                    output[q_out_base + i] = static_cast<T>(normed);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 2) {
        // Stage 1 stops here. The reset_counter call below would still be
        // safe but is unnecessary work for the parity test.
        return;
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase D: k_raw, v_raw via fused work-stealing matvec --------------
    // Two structural variants:
    //
    //   USE_WMMA=false (scalar): fused 2*Hkv*d-row work pool — every block
    //     stays busy regardless of which projection it claims. PyTorch's
    //     fused-QKV pattern.
    //
    //   USE_WMMA=true: split into k and v sub-pools with one grid barrier
    //     between. WMMA tiles need 16 contiguous output rows from one slab,
    //     and k/v live in different weight slabs (`k_proj_w` vs `v_proj_w`)
    //     with independent INT4 sidecars, so straddling tiles is unsafe.
    //     One extra grid_barrier_reset_counter (~5 µs × 10 full-attn layers
    //     = ~50 µs / token); negligible vs the matmul win.
    const int kv_total_rows = 2 * Hkv * d;

#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    if constexpr (USE_WMMA) {
        const int gsc_kv = hidden / int4_group_size;
        const int wave_id = tid >> 5;
        const int lane_in_wave = tid & 31;
        const int lane_row = lane_in_wave & 15;
        const int lane_half = lane_in_wave >> 4;

        // ------- Sub-pool 1: k_proj -------
        if (k_proj_scale != nullptr) {
            const uint8_t* slab_packed =
                reinterpret_cast<const uint8_t*>(k_proj_w);
            for (;;) {
                __shared__ unsigned int row_base_s;
                if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                __syncthreads();
                const int row_base = static_cast<int>(row_base_s);
                if (row_base >= Hkv * d) break;

                const int rhs_row = row_base + wave_id * 16 + lane_row;
                const bool in_range = rhs_row < Hkv * d;
                const uint8_t* slab_row = in_range
                    ? slab_packed +
                      static_cast<size_t>(rhs_row) * (hidden / 2)
                    : nullptr;
                qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                    slab_row,
                    static_cast<const hip_bfloat16*>(k_proj_scale),
                    static_cast<const hip_bfloat16*>(k_proj_zero),
                    rhs_row, in_range,
                    x_norm_lds, hidden,
                    gsc_kv, int4_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_K_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16 fallback for k_proj.
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= Hkv * d) break;
                const T* w_row =
                    k_proj_w + static_cast<size_t>(my_row) * hidden;
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
                    workspace[OFF_K_RAW + my_row] =
                        bf16_round_rne_f32(shared_scratch[0]);
                }
                __syncthreads();
            }
        }

        // Inter-barrier: publish k writes, reset counters[0] for v.
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);

        // ------- Sub-pool 2: v_proj -------
        if (v_proj_scale != nullptr) {
            const uint8_t* slab_packed =
                reinterpret_cast<const uint8_t*>(v_proj_w);
            for (;;) {
                __shared__ unsigned int row_base_s;
                if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                __syncthreads();
                const int row_base = static_cast<int>(row_base_s);
                if (row_base >= Hkv * d) break;

                const int rhs_row = row_base + wave_id * 16 + lane_row;
                const bool in_range = rhs_row < Hkv * d;
                const uint8_t* slab_row = in_range
                    ? slab_packed +
                      static_cast<size_t>(rhs_row) * (hidden / 2)
                    : nullptr;
                qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                    slab_row,
                    static_cast<const hip_bfloat16*>(v_proj_scale),
                    static_cast<const hip_bfloat16*>(v_proj_zero),
                    rhs_row, in_range,
                    x_norm_lds, hidden,
                    gsc_kv, int4_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_V_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16 fallback for v_proj.
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= Hkv * d) break;
                const T* w_row =
                    v_proj_w + static_cast<size_t>(my_row) * hidden;
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
                    workspace[OFF_V_RAW + my_row] =
                        bf16_round_rne_f32(shared_scratch[0]);
                }
                __syncthreads();
            }
        }
    } else
#endif
    {
        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) {
                my_row_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= kv_total_rows) break;

            // First Hkv*d rows belong to k_proj, next Hkv*d to v_proj.
            const bool is_v = my_row >= Hkv * d;
            const int proj_row = is_v ? (my_row - Hkv * d) : my_row;
            const T* w_slab = is_v ? v_proj_w : k_proj_w;
            const hip_bfloat16* i4_scale = is_v ? v_proj_scale : k_proj_scale;
            const hip_bfloat16* i4_zero  = is_v ? v_proj_zero  : k_proj_zero;
            const int dst_off = (is_v ? OFF_V_RAW : OFF_K_RAW) + proj_row;

            float partial = 0.0f;
            if (i4_scale != nullptr) {
                partial = int4_dq8_matvec_partial(
                    reinterpret_cast<const uint8_t*>(w_slab),
                    i4_scale, i4_zero, x_norm_lds,
                    proj_row, hidden, int4_group_size,
                    tid, block_size);
            } else {
                const T* w_row = w_slab + static_cast<size_t>(proj_row) * hidden;
                for (int col = tid; col < hidden; col += block_size) {
                    partial += load_as_float<T>(w_row, col) * x_norm_lds[col];
                }
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                workspace[dst_off] = bf16_round_rne_f32(shared_scratch[0]);
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase E: per-head RMS norm of K (Hkv heads) -----------------------
    // Same shape as Phase C but reduces over Hkv heads, sourcing from
    // workspace[K_RAW] and writing workspace[K_NORMED]. If stage == 2
    // we also publish k_normed to the host-visible `output` buffer.
    {
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(head_s);
            if (h >= Hkv) break;

            const int base = h * d;
            float partial = 0.0f;
            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_K_RAW + base + i];
                partial += v * v;
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            __shared__ float head_inv_rms;
            if (tid == 0) {
                head_inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(d) + rms_norm_eps);
            }
            __syncthreads();

            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_K_RAW + base + i];
                const float wv = load_as_float<T>(k_norm_w, i);
                // HF `Qwen3_5MoeRMSNorm` `(1.0 + weight)` unit offset.
                const float normed = bf16_round_rne_f32(v * head_inv_rms * (1.0f + wv));
                workspace[OFF_K_NORMED + base + i] = normed;
                if (stage == 2) {
                    output[base + i] = static_cast<T>(normed);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 3) return;

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase F: partial RoPE on Q and K ----------------------------------
    //
    // For each head, rotate the first `rotary_dim` channels using the
    // half-pair convention from oracle/qwen36_moe_oracle.py:
    //
    //   for i in [0, half):
    //     a, b = x[i], x[half + i]
    //     freq = position * theta^(-i/half)
    //     c = bf16(cos(freq))      s = bf16(sin(freq))
    //     out[i]        = bf16(bf16(a*c) - bf16(b*s))
    //     out[half + i] = bf16(bf16(b*c) + bf16(a*s))
    //   for i in [rotary_dim, d): out[i] = x[i]    (pass-through)
    //
    // Each "BF16-round" mirrors the BF16 dtype boundary in PyTorch — the
    // multiplication, the subtraction, and the cos/sin cast all happen in
    // BF16 in the oracle. Skipping any of these RNE rounds drifts the test.
    //
    // Work units = H + Hkv (one per attention head, Q and K combined).
    // Stage 3 publishes q_rot || k_rot to the host-visible output buffer:
    //   output[0..H*d)         = q_rot (BF16)
    //   output[H*d..H*d+Hkv*d) = k_rot (BF16)
    {
        const int half = rotary_dim / 2;
        const float theta_log = logf(rope_theta);
        const int total_heads = H + Hkv;

        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int head_idx = static_cast<int>(head_s);
            if (head_idx >= total_heads) break;

            const bool is_k = head_idx >= H;
            const int  h    = is_k ? (head_idx - H) : head_idx;
            const int  src_off = is_k ? OFF_K_NORMED : OFF_Q_NORMED;
            const int  dst_off = is_k ? OFF_K_ROT    : OFF_Q_ROT;
            // q_rot lives at output[0..H*d), k_rot at output[H*d..H*d+Hkv*d).
            const int  pub_off = head_idx * d;

            // Rotated half-pairs.
            for (int i = tid; i < half; i += block_size) {
                const float a = workspace[src_off + h * d + i];
                const float b = workspace[src_off + h * d + half + i];
                // freq = position * theta^(-i/half) = position / theta^(i/half).
                // Match PyTorch's `theta ** (i/half)` ≡ exp((i/half) * log(theta)).
                const float exponent = (static_cast<float>(i) / static_cast<float>(half)) * theta_log;
                const float inv_freq = expf(-exponent);
                const float freq = static_cast<float>(position) * inv_freq;
                const float c = bf16_round_rne_f32(cosf(freq));
                const float s = bf16_round_rne_f32(sinf(freq));
                const float ac = bf16_round_rne_f32(a * c);
                const float bs = bf16_round_rne_f32(b * s);
                const float bc = bf16_round_rne_f32(b * c);
                const float as_ = bf16_round_rne_f32(a * s);
                const float rot_a = bf16_round_rne_f32(ac - bs);
                const float rot_b = bf16_round_rne_f32(bc + as_);
                workspace[dst_off + h * d + i]        = rot_a;
                workspace[dst_off + h * d + half + i] = rot_b;
                if (stage == 3) {
                    output[pub_off + i]        = static_cast<T>(rot_a);
                    output[pub_off + half + i] = static_cast<T>(rot_b);
                }
            }

            // Pass-through tail [rotary_dim, head_dim).
            for (int i = rotary_dim + tid; i < d; i += block_size) {
                const float x = workspace[src_off + h * d + i];
                workspace[dst_off + h * d + i] = x;
                if (stage == 3) {
                    output[pub_off + i] = static_cast<T>(x);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 4) return;

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase G: GQA self-attention with optional KV cache -----------------
    //
    // Two modes selected by whether kv_cache_k/v are non-null:
    //
    // 1) Back-compat (kv_cache == null): kv_len = 1 self-attention. Softmax
    //    over a single score is 1.0; output[h] = V[h_kv]. Same path as the
    //    PR 4b2 oracle, kept bit-exact so the per-block parity tests don't
    //    regress.
    //
    // 2) KV cache enabled (kv_cache != null): write current (K_rot, V_raw)
    //    at slot `position`, then attend over kv_len = position + 1 past
    //    tokens with real softmax (max-stabilised, F32 exp). Per-Q-head
    //    work-stealing; per-head scores live at workspace[OFF_SCORES +
    //    hq * kv_max_t] so concurrent blocks computing different heads
    //    don't race. Cap: kv_max_t must accommodate position + 1.
    {
        const int rep   = H / Hkv;
        const float scale = rsqrtf(static_cast<float>(d));
        const bool use_kv_cache = (kv_cache_k != nullptr && kv_cache_v != nullptr);
        // Effective cache slot. `cache_pos < 0` ⇒ inherit from `position`
        // (base-model decode); `cache_pos >= 0` ⇒ MTP-style decoupled write.
        const int eff_cache_pos = (cache_pos >= 0) ? cache_pos : position;
        const int kv_len = use_kv_cache ? (eff_cache_pos + 1) : 1;

        // Step 1: write current K/V into the cache (if enabled). All blocks
        // cooperate via a flat work-stealing index since the writes are
        // independent across (h_kv, i).
        if (use_kv_cache) {
            const int slot_base = eff_cache_pos * Hkv * d;
            const int total = Hkv * d;
            for (int idx = blockIdx.x * block_size + tid;
                 idx < total;
                 idx += num_blocks * block_size) {
                const int h_kv = idx / d;
                const int i    = idx % d;
                const float kv = workspace[OFF_K_ROT + h_kv * d + i];
                const float vv = workspace[OFF_V_RAW + h_kv * d + i];
                kv_cache_k[slot_base + h_kv * d + i] = static_cast<T>(kv);
                kv_cache_v[slot_base + h_kv * d + i] = static_cast<T>(vv);
            }
            // Ensure all blocks' writes are visible before the read-side
            // attention reduction reads cache[0..=position].
            grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                       &counters[0]);
        }

        // Step 2: per-Q-head attention reduction. Each block claims one head
        // via atomicAdd on counters[0]; the write phase above already reset
        // it (via grid_barrier_reset_counter when use_kv_cache, or by leaving
        // it untouched when not).
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int hq = static_cast<int>(head_s);
            if (hq >= H) break;

            const int h_kv = hq / rep;

            if (kv_len == 1) {
                // Back-compat fast path — bit-exact match for PR 4b2 parity.
                float partial = 0.0f;
                for (int i = tid; i < d; i += block_size) {
                    const float q = workspace[OFF_Q_ROT + hq   * d + i];
                    const float k = workspace[OFF_K_ROT + h_kv * d + i];
                    partial += q * k;
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                __shared__ float head_score;
                if (tid == 0) {
                    head_score = shared_scratch[0] * scale;
                }
                __syncthreads();
                (void)head_score;
                // softmax([s]) = [1.0]; output collapses to V at this position.
                for (int i = tid; i < d; i += block_size) {
                    const float v = workspace[OFF_V_RAW + h_kv * d + i];
                    workspace[OFF_ATTN + hq * d + i] = v;
                    if (stage == 4) {
                        output[hq * d + i] = static_cast<T>(v);
                    }
                }
                __syncthreads();
                continue;
            }

            // KV-cache path. Compute scores[t] = (q · K_cache[t]) * scale
            // for t in 0..=position. Per-head scores live at
            // workspace[OFF_SCORES + hq * kv_max_t + t].
            const int score_base = OFF_SCORES + hq * kv_max_t;
            for (int t = 0; t < kv_len; t++) {
                float partial = 0.0f;
                for (int i = tid; i < d; i += block_size) {
                    const float q = workspace[OFF_Q_ROT + hq * d + i];
                    const float k = static_cast<float>(
                        kv_cache_k[t * Hkv * d + h_kv * d + i]);
                    partial += q * k;
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    workspace[score_base + t] = shared_scratch[0] * scale;
                }
                __syncthreads();
            }

            // Softmax with max-stabilisation. Done by tid==0 to keep the
            // reduction simple — kv_len is small (≤ context window) and the
            // remaining work (V-weighted reduction) is the bulk of the cost.
            __shared__ float max_score;
            __shared__ float exp_sum;
            if (tid == 0) {
                float m = workspace[score_base];
                for (int t = 1; t < kv_len; t++) {
                    m = fmaxf(m, workspace[score_base + t]);
                }
                max_score = m;
                float s = 0.0f;
                for (int t = 0; t < kv_len; t++) {
                    const float e = expf(workspace[score_base + t] - m);
                    workspace[score_base + t] = e;
                    s += e;
                }
                exp_sum = s;
            }
            __syncthreads();

            // Weighted sum of V over t. Each thread handles a slice of d.
            const float inv_sum = 1.0f / exp_sum;
            for (int i = tid; i < d; i += block_size) {
                float acc = 0.0f;
                for (int t = 0; t < kv_len; t++) {
                    const float w = workspace[score_base + t] * inv_sum;
                    const float v = static_cast<float>(
                        kv_cache_v[t * Hkv * d + h_kv * d + i]);
                    acc += w * v;
                }
                workspace[OFF_ATTN + hq * d + i] = acc;
                if (stage == 4) {
                    output[hq * d + i] = static_cast<T>(acc);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 5) return;

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase H: gated_attn = bf16(sigmoid(out_gate)) * attn --------------
    //
    // out_gate is the second-half lane of q_proj's per-head interleaved
    // output: for head h it lives at workspace[OFF_Q_RAW + h*2*d + d + i]
    // (HF `view(..., H, head_dim*2).chunk(2, -1)` convention). Each block
    // grabs one Q head and produces gated_attn[h, :] in workspace.
    // Oracle math (qwen36_moe_oracle.py `out_gate * attn`):
    //   sigmoid_bf16 = bf16(sigmoid(f32(out_gate)))
    //   gated_attn   = bf16(sigmoid_bf16 * attn)
    {
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(head_s);
            if (h >= H) break;

            for (int i = tid; i < d; i += block_size) {
                const float out_gate = workspace[OFF_Q_RAW + h * 2 * d + d + i];
                const float attn_v   = workspace[OFF_ATTN + h * d + i];
                const float sig      = 1.0f / (1.0f + expf(-out_gate));
                const float sig_bf   = bf16_round_rne_f32(sig);
                const float gated    = bf16_round_rne_f32(sig_bf * attn_v);
                workspace[OFF_GATED + h * d + i] = gated;
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase I: o_proj matmul + residual ---------------------------------
    //
    // output_hidden[i] = bf16(input_hidden[i] +
    //                        bf16(sum_j(o_proj_w[i, j] * gated_attn[j])))
    //
    // The o_proj output is BF16-rounded before adding the residual, then
    // the sum itself is BF16-rounded. That's two RNE rounds — one at the
    // matmul boundary (PyTorch's bf16 matmul), one at the add (bf16 + bf16).
    // Skipping either drifts the test.
    //
    // Each block grabs one row of o_proj_w via work-stealing, reduces over
    // qd = H*d in F32, and writes BF16 output_hidden to both workspace and
    // (when stage == 5) the host-visible output buffer.
    {
        const int qd = H * d;
        const bool o_int4 = (o_proj_scale != nullptr);

        bool wmma_handled_oproj = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
        if constexpr (USE_WMMA) {
            if (o_int4) {
                const int gsc_o = qd / int4_group_size;
                const int wave_id = tid >> 5;
                const int lane_in_wave = tid & 31;
                const int lane_row = lane_in_wave & 15;
                const int lane_half = lane_in_wave >> 4;
                const float* gated_lds_f32 = workspace + OFF_GATED;
                const uint8_t* slab_packed =
                    reinterpret_cast<const uint8_t*>(o_proj_w);
                for (;;) {
                    __shared__ unsigned int row_base_s;
                    if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                    __syncthreads();
                    const int row_base = static_cast<int>(row_base_s);
                    if (row_base >= hidden) break;

                    const int rhs_row = row_base + wave_id * 16 + lane_row;
                    const bool in_range = rhs_row < hidden;
                    const uint8_t* slab_row = in_range
                        ? slab_packed +
                          static_cast<size_t>(rhs_row) * (qd / 2)
                        : nullptr;
                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row,
                        static_cast<const hip_bfloat16*>(o_proj_scale),
                        static_cast<const hip_bfloat16*>(o_proj_zero),
                        rhs_row, in_range,
                        gated_lds_f32, qd,
                        gsc_o, int4_group_size, lane_row);
                    if (lane_half == 0 && in_range) {
                        const float o_out  = bf16_round_rne_f32(acc[0]);
                        const float in_f   =
                            load_as_float<T>(input_hidden, rhs_row);
                        const float result = bf16_round_rne_f32(in_f + o_out);
                        workspace[OFF_O_OUT + rhs_row] = result;
                        if (stage == 5) {
                            output[rhs_row] = static_cast<T>(result);
                        }
                    }
                    __syncthreads();
                }
                wmma_handled_oproj = true;
            }
        }
#endif
        if (!wmma_handled_oproj) {
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) {
                    my_row_s = atomicAdd(&counters[0], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= hidden) break;

                float partial = 0.0f;
                if (o_int4) {
                    partial = int4_dq8_matvec_partial(
                        reinterpret_cast<const uint8_t*>(o_proj_w),
                        static_cast<const hip_bfloat16*>(o_proj_scale),
                        static_cast<const hip_bfloat16*>(o_proj_zero),
                        workspace + OFF_GATED,
                        my_row, qd, int4_group_size,
                        tid, block_size);
                } else {
                    const T* w_row = o_proj_w + static_cast<size_t>(my_row) * qd;
                    for (int j = tid; j < qd; j += block_size) {
                        partial += load_as_float<T>(w_row, j) * workspace[OFF_GATED + j];
                    }
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    const float o_out  = bf16_round_rne_f32(shared_scratch[0]);
                    const float in_f   = load_as_float<T>(input_hidden, my_row);
                    const float result = bf16_round_rne_f32(in_f + o_out);
                    workspace[OFF_O_OUT + my_row] = result;
                    if (stage == 5) {
                        output[my_row] = static_cast<T>(result);
                    }
                }
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Single-layer LINEAR-attention parity kernel (PR 4b3)
//
// Mirrors the staged build-up pattern from PR 4b2's full-attention kernel:
// each stage adds one slice of compute, the host parity test downloads the
// matching intermediate buffer and compares it to the linear-attn oracle
// (oracle/qwen36_moe_linear_oracle.py).
//
//   stage 1: x_norm + in_proj_qkv + in_proj_z + in_proj_a + in_proj_b
//            → publish qkv_raw [qkv_dim] BF16
//   stage 2: ... + depthwise conv1d + silu
//            → publish silu_out [qkv_dim] BF16  (planned)
//   stage 3: ... + L2-norm Q/K + GQA repeat + Q scale
//            → publish q_scaled || k_rep || v_heads (planned)
//   stage 4: ... + delta-rule recurrent state update
//            → publish recurrent_out [V*v_dim] BF16  (planned)
//   stage 5: ... + per-head RMS norm + z-gate + out_proj + residual
//            → publish output_hidden [hidden] BF16  (planned)
//
// PR 4b3 step 2 lands stage 1. Stages 2-5 follow in their own commits.
//
// Workspace layout (F32 elements, offsets) — pinned now even though only
// the first four slots are written in stage 1, so later stages don't have
// to reshuffle the layout:
//
//   QKV_RAW    = 0                                        [qkv_dim]
//   Z_RAW      = qkv_dim                                  [V*v_dim]
//   A_RAW      = qkv_dim + V*v_dim                        [V]
//   B_RAW      = qkv_dim + V*v_dim + V                    [V]
//   Q_NORMED   = qkv_dim + V*v_dim + 2*V                  [K*k_dim]
//   K_NORMED   = ... + K*k_dim                            [K*k_dim]
//   Q_REP      = ... + K*k_dim                            [V*k_dim]
//   K_REP      = ... + V*k_dim                            [V*k_dim]
//   BETA       = ... + V*k_dim                            [V]
//   G          = ... + V                                  [V]
//   REC_OUT    = ... + V                                  [V*v_dim]
//   O_OUT      = ... + V*v_dim                            [hidden]
//
// For 35B-A3B (qkv_dim=8192, V=32, v_dim=128, K=16, k_dim=128, hidden=2048):
//   stage-1 footprint = 8192 + 4096 + 32 + 32 = 12,352 F32 = 48 KiB
//   stage-5 footprint = ~30,848 F32 = 120 KiB (still well under sane limits)
//
// Conv state and recurrent state live in dedicated state buffers (state
// is up to 2 MiB per layer at F32) — those are NOT in `workspace`.

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase B (the four
// in-projections — INT4 on qkv/z, BF16 on a/b) and Phase I (out_proj).
// When true:
//   - Phase B splits the unified work pool into four sub-loops with grid
//     barriers between them; qkv/z run WMMA tiles when their scales are
//     present, a/b stay scalar (always BF16, ≤V=32 rows so WMMA tiling
//     is overkill anyway).
//   - Phase I runs the same WMMA tile pattern as the FFN's Phase I.
// The bridge picks the instantiation at launch time based on
// `device_supports_wmma_bf16(ordinal)` plus dim divisibility checks.
template <typename T, bool USE_WMMA = false>
__global__ void qwen36_moe_linear_step_kernel(
    int                            stage,
    int                            hidden,
    int                            num_k_heads,         // K
    int                            num_v_heads,         // V
    int                            head_k_dim,          // k_dim
    int                            head_v_dim,          // v_dim
    int                            conv_kernel_dim,
    float                          rms_norm_eps,
    const T* __restrict__          input_hidden,        // [hidden]
    const T* __restrict__          input_norm_w,        // [hidden]
    const T* __restrict__          in_proj_qkv_w,       // [qkv_dim, hidden]
    const T* __restrict__          in_proj_z_w,         // [V*v_dim, hidden]
    const T* __restrict__          in_proj_a_w,         // [V, hidden]
    const T* __restrict__          in_proj_b_w,         // [V, hidden]
    const T* __restrict__          conv1d_w,            // unused at stage 1
    const T* __restrict__          conv1d_bias,         // unused at stage 1
    const T* __restrict__          dt_bias,             // unused at stage 1
    const T* __restrict__          a_log,               // unused at stage 1
    const T* __restrict__          norm_w,              // unused at stage 1
    const T* __restrict__          out_proj_w,          // unused at stage 1
    T* __restrict__                conv_state,          // unused at stage 1
    float* __restrict__            recurrent_state,     // unused at stage 1
    // PR 4b6 INT4 sidecars. Same null-pointer-as-mode discipline as the
    // full-attn kernel. Only the three projections that the bake quantizes
    // are wired (in_proj_qkv, in_proj_z, out_proj). in_proj_a/in_proj_b are
    // tiny per-V-head scalars and stay BF16.
    int                                  int4_group_size,
    const hip_bfloat16* __restrict__     in_proj_qkv_scale,
    const hip_bfloat16* __restrict__     in_proj_qkv_zero,
    const hip_bfloat16* __restrict__     in_proj_z_scale,
    const hip_bfloat16* __restrict__     in_proj_z_zero,
    const hip_bfloat16* __restrict__     out_proj_scale,
    const hip_bfloat16* __restrict__     out_proj_zero,
    T* __restrict__                output,
    float* __restrict__            workspace,
    unsigned int* __restrict__     counters,
    unsigned int* __restrict__     barrier_counter,
    unsigned int* __restrict__     barrier_flag) {
    (void)conv1d_w;
    (void)conv1d_bias;
    (void)dt_bias;
    (void)a_log;
    (void)norm_w;
    (void)out_proj_w;
    (void)conv_state;
    (void)recurrent_state;

    const int num_blocks = static_cast<int>(gridDim.x);
    const int tid        = threadIdx.x;
    const int block_size = static_cast<int>(blockDim.x);

    extern __shared__ float lds[];
    float* x_norm_lds    = lds;             // [hidden]
    float* shared_scratch = lds + hidden;   // [block_size]

    // Workspace offsets (computed once; same convention as the full-attn
    // kernel's OFF_* locals).
    const int K       = num_k_heads;
    const int V       = num_v_heads;
    const int k_dim   = head_k_dim;
    const int v_dim   = head_v_dim;
    const int key_dim = K * k_dim;
    const int val_dim = V * v_dim;
    const int qkv_dim = 2 * key_dim + val_dim;
    const int OFF_QKV_RAW = 0;
    const int OFF_Z_RAW   = qkv_dim;
    const int OFF_A_RAW   = qkv_dim + val_dim;
    const int OFF_B_RAW   = qkv_dim + val_dim + V;

    // -- Phase A: load input_hidden + RMS norm into LDS --------------------
    // Same idiom as the full-attn kernel's Phase A.
    for (int col = tid; col < hidden; col += block_size) {
        x_norm_lds[col] = static_cast<float>(input_hidden[col]);
    }
    __syncthreads();

    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        partial_sq += x_norm_lds[col] * x_norm_lds[col];
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float inv_rms_input;
    if (tid == 0) {
        inv_rms_input = rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // HF `Qwen3_5MoeRMSNorm` `(1.0 + weight)` unit offset for
    // `input_layernorm` ahead of the linear-attn block.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = static_cast<float>(input_norm_w[col]);
        x_norm_lds[col] = bf16_round_rne_f32(x_norm_lds[col] * inv_rms_input * (1.0f + w));
    }
    __syncthreads();

    // -- Phase B: four in-projections (qkv / z / a / b) -------------------
    //
    // qkv_raw, z_raw, a_raw, b_raw all read the same x_norm and have
    // independent output rows. Two structural variants:
    //
    //   USE_WMMA=false (scalar): one fused work pool of total_rows =
    //     qkv_dim + V*v_dim + 2*V rows. Each block claims the next row,
    //     branches on row range to pick the projection, computes the dot
    //     product (INT4 dq8 or scalar BF16), writes BF16-rounded F32.
    //
    //   USE_WMMA=true: the qkv and z projections — both INT4 in
    //     production — split into their own WMMA tile sub-loops (128
    //     rows per atomicAdd, 8 waves × 16 rows each). a and b stay on
    //     the scalar path (always BF16, ≤V rows each, small enough that
    //     a single fused 2*V scalar mini-pool wins). Two extra grid
    //     barriers gate the sub-loops.
    //
    // Row-id range layout (matches the workspace offsets):
    //   [0,                       qkv_dim):                qkv (in_proj_qkv)
    //   [qkv_dim,                 qkv_dim + V*v_dim):      z   (in_proj_z)
    //   [qkv_dim + V*v_dim,       qkv_dim + V*v_dim + V):  a   (in_proj_a)
    //   [qkv_dim + V*v_dim + V,   qkv_dim + V*v_dim + 2V): b   (in_proj_b)
    const int total_rows = qkv_dim + val_dim + 2 * V;

#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    if constexpr (USE_WMMA) {
        const int wave_id  = tid >> 5;
        const int lane_in_wave = tid & 31;
        const int lane_row = lane_in_wave & 15;
        const int lane_half = lane_in_wave >> 4;

        // ------- Sub-pool 1: in_proj_qkv -------
        if (in_proj_qkv_scale != nullptr) {
            const int gsc_q = hidden / int4_group_size;
            const uint8_t* slab_packed =
                reinterpret_cast<const uint8_t*>(in_proj_qkv_w);
            for (;;) {
                __shared__ unsigned int row_base_s;
                if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                __syncthreads();
                const int row_base = static_cast<int>(row_base_s);
                if (row_base >= qkv_dim) break;

                const int rhs_row = row_base + wave_id * 16 + lane_row;
                const bool in_range = rhs_row < qkv_dim;
                const uint8_t* slab_row = in_range
                    ? slab_packed +
                      static_cast<size_t>(rhs_row) * (hidden / 2)
                    : nullptr;
                qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                    slab_row, in_proj_qkv_scale, in_proj_qkv_zero,
                    rhs_row, in_range,
                    x_norm_lds, hidden,
                    gsc_q, int4_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_QKV_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16 fallback for qkv (parity-test path; bake is INT4).
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= qkv_dim) break;
                const T* w_row =
                    in_proj_qkv_w + static_cast<size_t>(my_row) * hidden;
                float partial = 0.0f;
                for (int col = tid; col < hidden; col += block_size) {
                    partial += static_cast<float>(w_row[col]) * x_norm_lds[col];
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    workspace[OFF_QKV_RAW + my_row] =
                        bf16_round_rne_f32(shared_scratch[0]);
                }
                __syncthreads();
            }
        }

        // Inter-barrier: publish qkv writes, reset counters[0] for z.
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);

        // ------- Sub-pool 2: in_proj_z -------
        if (in_proj_z_scale != nullptr) {
            const int gsc_z = hidden / int4_group_size;
            const uint8_t* slab_packed =
                reinterpret_cast<const uint8_t*>(in_proj_z_w);
            for (;;) {
                __shared__ unsigned int row_base_s;
                if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                __syncthreads();
                const int row_base = static_cast<int>(row_base_s);
                if (row_base >= val_dim) break;

                const int rhs_row = row_base + wave_id * 16 + lane_row;
                const bool in_range = rhs_row < val_dim;
                const uint8_t* slab_row = in_range
                    ? slab_packed +
                      static_cast<size_t>(rhs_row) * (hidden / 2)
                    : nullptr;
                qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                    slab_row, in_proj_z_scale, in_proj_z_zero,
                    rhs_row, in_range,
                    x_norm_lds, hidden,
                    gsc_z, int4_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_Z_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16 fallback for z.
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= val_dim) break;
                const T* w_row =
                    in_proj_z_w + static_cast<size_t>(my_row) * hidden;
                float partial = 0.0f;
                for (int col = tid; col < hidden; col += block_size) {
                    partial += static_cast<float>(w_row[col]) * x_norm_lds[col];
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    workspace[OFF_Z_RAW + my_row] =
                        bf16_round_rne_f32(shared_scratch[0]);
                }
                __syncthreads();
            }
        }

        // Inter-barrier: publish z writes, reset counters[0] for a/b.
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);

        // ------- Sub-pool 3: in_proj_a + in_proj_b (always BF16, V each) -
        // 2*V rows total — small enough that a single fused scalar pool
        // beats WMMA tiling overhead. Branches on `is_b` to pick slab
        // and dst offset.
        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= 2 * V) break;

            const bool is_b = my_row >= V;
            const int  local = is_b ? (my_row - V) : my_row;
            const T* w_row = is_b
                ? in_proj_b_w + static_cast<size_t>(local) * hidden
                : in_proj_a_w + static_cast<size_t>(local) * hidden;
            const int dst_off = is_b ? (OFF_B_RAW + local) : (OFF_A_RAW + local);

            float partial = 0.0f;
            for (int col = tid; col < hidden; col += block_size) {
                partial += static_cast<float>(w_row[col]) * x_norm_lds[col];
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                workspace[dst_off] = bf16_round_rne_f32(shared_scratch[0]);
            }
            __syncthreads();
        }
    } else
#endif
    {
        // Scalar unified pool (USE_WMMA=false, also the parity-test path).
        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) {
                my_row_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= total_rows) break;

            // Pick projection + per-projection row-index + workspace dst.
            // BF16 path tracks `w_row`; INT4 path tracks (slab base, scale, zero,
            // local row index) so the helper indexes within the per-projection
            // 2D weight regardless of which projection the row belongs to.
            const T* w_row = nullptr;
            const void* i4_slab = nullptr;
            const hip_bfloat16* i4_scale = nullptr;
            const hip_bfloat16* i4_zero  = nullptr;
            int i4_row = 0;
            int dst_off;
            if (my_row < qkv_dim) {
                if (in_proj_qkv_scale != nullptr) {
                    i4_slab  = reinterpret_cast<const void*>(in_proj_qkv_w);
                    i4_scale = in_proj_qkv_scale;
                    i4_zero  = in_proj_qkv_zero;
                    i4_row   = my_row;
                } else {
                    w_row = in_proj_qkv_w + static_cast<size_t>(my_row) * hidden;
                }
                dst_off = OFF_QKV_RAW + my_row;
            } else if (my_row < qkv_dim + val_dim) {
                const int r = my_row - qkv_dim;
                if (in_proj_z_scale != nullptr) {
                    i4_slab  = reinterpret_cast<const void*>(in_proj_z_w);
                    i4_scale = in_proj_z_scale;
                    i4_zero  = in_proj_z_zero;
                    i4_row   = r;
                } else {
                    w_row = in_proj_z_w + static_cast<size_t>(r) * hidden;
                }
                dst_off = OFF_Z_RAW + r;
            } else if (my_row < qkv_dim + val_dim + V) {
                // in_proj_a stays BF16 — bake excludes it.
                const int r = my_row - (qkv_dim + val_dim);
                w_row   = in_proj_a_w + static_cast<size_t>(r) * hidden;
                dst_off = OFF_A_RAW + r;
            } else {
                // in_proj_b stays BF16 — bake excludes it.
                const int r = my_row - (qkv_dim + val_dim + V);
                w_row   = in_proj_b_w + static_cast<size_t>(r) * hidden;
                dst_off = OFF_B_RAW + r;
            }

            float partial = 0.0f;
            if (i4_scale != nullptr) {
                partial = int4_dq8_matvec_partial(
                    static_cast<const uint8_t*>(i4_slab),
                    i4_scale, i4_zero, x_norm_lds,
                    i4_row, hidden, int4_group_size,
                    tid, block_size);
            } else {
                for (int col = tid; col < hidden; col += block_size) {
                    partial += static_cast<float>(w_row[col]) * x_norm_lds[col];
                }
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                workspace[dst_off] = bf16_round_rne_f32(shared_scratch[0]);
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // Stage 1 publishes qkv_raw to the host-visible output buffer. Later
    // stages publish their own intermediates and keep qkv_raw in workspace
    // (where Phase C — depthwise conv1d — will read it).
    if (stage == 1) {
        for (int i = tid + blockIdx.x * block_size;
             i < qkv_dim;
             i += block_size * num_blocks) {
            output[i] = static_cast<T>(workspace[OFF_QKV_RAW + i]);
        }
        return;
    }

    // No barrier here: post-Phase-B barrier above already published Phase B's
    // writes (qkv_raw etc.), and Phase C uses strided thread-per-channel
    // distribution — it never touches counters[0]. The pre-D barrier below
    // resets counters[0] for Phase D. (Earlier revisions had a redundant
    // grid_barrier_reset_counter here; it was pure vestige.)

    // -- Phase C: depthwise conv1d + silu + state update -------------------
    //
    // For each output channel ch in [0, qkv_dim):
    //   conv_in[t] = conv_state_before[ch, t]   for t in [0, kernel-1)
    //   conv_in[kernel-1] = bf16-rounded qkv_raw[ch]   (already in workspace)
    //   conv_out_bf = bf16(sum_t(conv_in[t] * conv1d_w[ch, t]) + bias[ch])
    //   silu_out    = bf16(conv_out_bf * sigmoid(conv_out_bf))
    //   conv_state_after[ch, 0..kernel-2] = conv_in[1..kernel-1]
    //
    // Output channels are independent and the per-channel arithmetic is
    // small (kernel multiplies + one sigmoid), so we use a strided
    // thread-per-channel loop instead of block-per-channel work-stealing —
    // ~256 channels processed per block per stride, much less overhead.
    //
    // KERNEL_MAX bounds the local conv_in array; 8 fits 35B-A3B's
    // conv_kernel_dim=4 with margin and avoids dynamic LDS for the tiny
    // per-thread scratch. Bigger kernels would just need a bump here.
    constexpr int KERNEL_MAX = 8;
    const int kernel = conv_kernel_dim;
    const int kstate = kernel - 1;
    const int total_threads = num_blocks * block_size;
    for (int ch = blockIdx.x * block_size + tid;
         ch < qkv_dim;
         ch += total_threads) {
        // Load conv_state_before (BF16) into a per-thread F32 register window.
        float conv_in[KERNEL_MAX];
        for (int t = 0; t < kstate; ++t) {
            conv_in[t] = static_cast<float>(conv_state[ch * kstate + t]);
        }
        // Tail slot is the freshly-projected qkv_raw[ch] (already
        // BF16-rounded in workspace by Phase B).
        const float new_qkv = workspace[OFF_QKV_RAW + ch];
        conv_in[kstate] = new_qkv;

        // Depthwise conv1d: per-channel dot product over the kernel window.
        float sum = 0.0f;
        for (int t = 0; t < kernel; ++t) {
            sum += conv_in[t] * static_cast<float>(conv1d_w[ch * kernel + t]);
        }
        // Bias is optional (None on 35B-A3B; the oracle handles both).
        if (conv1d_bias != nullptr) {
            sum += static_cast<float>(conv1d_bias[ch]);
        }
        // Match PyTorch BF16 conv: the matmul-then-bias result lands in BF16
        // before silu reads it.
        const float conv_out_bf = bf16_round_rne_f32(sum);

        // SiLU: x * sigmoid(x) in F32, then BF16-round at the dtype
        // boundary, exactly as `silu()` in oracle/qwen36_moe_linear_oracle.py.
        const float sig = 1.0f / (1.0f + expf(-conv_out_bf));
        const float silu_out = bf16_round_rne_f32(conv_out_bf * sig);

        // Overwrite QKV_RAW slot with SILU_OUT — qkv_raw is no longer
        // needed after this phase, and the channel layout is identical.
        workspace[OFF_QKV_RAW + ch] = silu_out;

        // Shift conv_state_before left by one and append the new qkv_raw
        // at position kstate-1, store back as BF16. Each thread owns its
        // channel's row — no cross-thread races.
        for (int t = 0; t < kstate - 1; ++t) {
            conv_state[ch * kstate + t] =
                static_cast<T>(conv_in[t + 1]);
        }
        conv_state[ch * kstate + (kstate - 1)] = static_cast<T>(new_qkv);

        // Stage 2 publishes silu_out to the host-visible output buffer.
        if (stage == 2) {
            output[ch] = static_cast<T>(silu_out);
        }
    }

    if (stage < 3) return;

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // Workspace offsets reserved by the layout comment for stages 3+.
    const int OFF_Q_NORMED = qkv_dim + val_dim + 2 * V;
    const int OFF_K_NORMED = OFF_Q_NORMED + K * k_dim;
    const int OFF_Q_REP    = OFF_K_NORMED + K * k_dim;
    const int OFF_K_REP    = OFF_Q_REP + V * k_dim;

    // -- Phase D: L2-normalize Q and K per K-head --------------------------
    //
    // q_raw lives at workspace[OFF_QKV_RAW : OFF_QKV_RAW + key_dim] (the
    // first slice of silu_out from Phase C); k_raw is the next key_dim
    // slice. Each block grabs one head out of a 2*K work pool — first K
    // heads are Q, next K are K.
    //
    // L2 norm matches `F.normalize(p=2, dim=-1, eps=1e-6)`:
    //   norm = sqrt(sum(x^2))                    in F32
    //   denom = max(bf16(norm), eps)             clamped, BF16-rounded
    //   y[i]  = bf16(x[i] / bf16(denom))         BF16-rounded division
    {
        const int total_heads = 2 * K;
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int head_idx = static_cast<int>(head_s);
            if (head_idx >= total_heads) break;

            const bool is_k = head_idx >= K;
            const int  h    = is_k ? (head_idx - K) : head_idx;
            const int  src_off = (is_k ? OFF_QKV_RAW + key_dim
                                       : OFF_QKV_RAW) + h * k_dim;
            const int  dst_off = (is_k ? OFF_K_NORMED : OFF_Q_NORMED) + h * k_dim;

            // Sum of squares (F32 — the reduction is the precision-critical
            // part; doing it in BF16 would drift visibly).
            float partial_ss = 0.0f;
            for (int i = tid; i < k_dim; i += block_size) {
                const float v = workspace[src_off + i];
                partial_ss += v * v;
            }
            shared_scratch[tid] = partial_ss;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            __shared__ float head_denom;
            if (tid == 0) {
                const float norm  = bf16_round_rne_f32(sqrtf(shared_scratch[0]));
                head_denom        = bf16_round_rne_f32(fmaxf(norm, 1e-6f));
            }
            __syncthreads();

            for (int i = tid; i < k_dim; i += block_size) {
                const float v = workspace[src_off + i];
                workspace[dst_off + i] = bf16_round_rne_f32(v / head_denom);
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase E: GQA fan-out + Q scale ------------------------------------
    //
    // For each V-head v in [0, V):
    //   src_kh = v // rep
    //   q_rep[v, :]   = bf16(q_normed[src_kh, :] * 1/sqrt(k_dim))
    //   k_rep[v, :]   =      k_normed[src_kh, :]            (just broadcast)
    //
    // Stage 3 publishes q_scaled || k_rep || v_heads concatenated:
    //   output[0..V*k_dim)                            = q_scaled
    //   output[V*k_dim..2*V*k_dim)                    = k_rep
    //   output[2*V*k_dim..2*V*k_dim + V*v_dim)        = v_heads
    {
        const int rep = V / K;
        const float q_scale = rsqrtf(static_cast<float>(k_dim));

        for (;;) {
            __shared__ unsigned int vhead_s;
            if (tid == 0) {
                vhead_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int vhead = static_cast<int>(vhead_s);
            if (vhead >= V) break;

            const int src_kh = vhead / rep;
            const int q_src  = OFF_Q_NORMED + src_kh * k_dim;
            const int k_src  = OFF_K_NORMED + src_kh * k_dim;
            const int q_dst  = OFF_Q_REP    + vhead  * k_dim;
            const int k_dst  = OFF_K_REP    + vhead  * k_dim;

            for (int i = tid; i < k_dim; i += block_size) {
                const float qn = workspace[q_src + i];
                const float kn = workspace[k_src + i];
                const float qs = bf16_round_rne_f32(qn * q_scale);
                workspace[q_dst + i] = qs;
                workspace[k_dst + i] = kn;
                if (stage == 3) {
                    output[vhead * k_dim + i]               = static_cast<T>(qs);
                    output[V * k_dim + vhead * k_dim + i]   = static_cast<T>(kn);
                }
            }
            // V is reshape-only of v_raw (silu_out's third partition); no
            // arithmetic, just publish for stage 3 verification.
            if (stage == 3) {
                const int v_src = OFF_QKV_RAW + 2 * key_dim + vhead * v_dim;
                for (int i = tid; i < v_dim; i += block_size) {
                    const float vv = workspace[v_src + i];
                    output[2 * V * k_dim + vhead * v_dim + i] = static_cast<T>(vv);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 4) return;

    const int OFF_BETA    = OFF_K_REP + V * k_dim;
    const int OFF_G       = OFF_BETA + V;
    const int OFF_REC_OUT = OFF_G + V;

    // -- Phase F: beta + g per V-head ---------------------------------------
    // beta[h] = sigmoid(b_raw[h])                                  (F32)
    // g[h]    = -softplus(a_raw[h] + dt_bias[h]) * exp(A_log[h])   (F32)
    //
    // V is small (32 on 35B-A3B) so we do this on block 0 only — every
    // other block parks at the post-F grid_barrier below.
    //
    // No pre-F barrier: Phase F reads only `a_raw`, `b_raw` (Phase B writes,
    // long since published) and the host-uploaded BF16 tables `dt_bias` /
    // `a_log`. Phase F doesn't touch counters[0] (block-0-only, no atomic
    // claim). counters[0] is left at V from Phase E's work-stealing; the
    // post-F barrier (below) resets it to 0 in time for Phase G.
    if (blockIdx.x == 0) {
        for (int h = tid; h < V; h += block_size) {
            const float a_v       = workspace[OFF_A_RAW + h];
            const float b_v       = workspace[OFF_B_RAW + h];
            const float dt_b      = static_cast<float>(dt_bias[h]);
            const float a_log_v   = static_cast<float>(a_log[h]);
            const float a_log_exp = expf(a_log_v);
            const float arg       = a_v + dt_b;
            const float softplus  = log1pf(expf(arg));
            workspace[OFF_BETA + h] = 1.0f / (1.0f + expf(-b_v));
            workspace[OFF_G + h]    = -softplus * a_log_exp;
        }
    }
    // Post-F barrier: publishes Phase F (beta, g) AND transitively Phase E's
    // writes (q_rep, k_rep) to all blocks, AND resets counters[0] (left at V
    // by Phase E) for Phase G's per-V-head work-stealing.
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase G: delta-rule recurrent update --------------------------------
    //
    // Per V-head (work-stolen):
    //   1. state[h, :, :] *= exp(g[h])                          (decay)
    //   2. kv_mem[j] = sum_i(state[h, i, j] * k_rep[h, i])     for j ∈ [0, v_dim)
    //   3. delta[j]  = (v_heads[h, j] - kv_mem[j]) * beta[h]
    //   4. state[h, i, j] += k_rep[h, i] * delta[j]            (outer product)
    //   5. rec_out[h, j] = sum_i(state[h, i, j] * q_scaled[h, i])  (BF16-round on store)
    //
    // All math in F32; recurrent state stays F32 between decode steps so
    // there's no precision loss across the sequence. rec_out is BF16-rounded
    // when written to workspace because Phase H (per-head RMS norm) reads it
    // as a BF16 input.
    //
    // LDS scratch (per block):
    //   kv_mem_lds[v_dim]  - kv_mem reduction outputs (F32)
    //   delta_lds[v_dim]   - delta values (F32)
    //   v_dim is bounded by 256 (35B-A3B uses 128); a static LDS array
    //   sized 256 fits all current configs and avoids dynamic LDS arithmetic.
    constexpr int V_DIM_MAX = 256;
    __shared__ float kv_mem_lds[V_DIM_MAX];
    __shared__ float delta_lds[V_DIM_MAX];
    __shared__ float head_beta;
    __shared__ float head_gstep;

    {
        for (;;) {
            __shared__ unsigned int vh_s;
            if (tid == 0) {
                vh_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(vh_s);
            if (h >= V) break;

            if (tid == 0) {
                head_beta  = workspace[OFF_BETA + h];
                head_gstep = expf(workspace[OFF_G + h]);
            }
            __syncthreads();

            const int state_off = h * k_dim * v_dim;
            const int kv_off    = OFF_K_REP + h * k_dim;
            const int qv_off    = OFF_Q_REP + h * k_dim;
            const int v_off     = OFF_QKV_RAW + 2 * key_dim + h * v_dim;
            const int total_state = k_dim * v_dim;

            // Step 1: decay.
            for (int e = tid; e < total_state; e += block_size) {
                recurrent_state[state_off + e] *= head_gstep;
            }
            __syncthreads();

            // Step 2: kv_mem[j] = sum_i(state[h, i, j] * k_rep[h, i]).
            // Loop over j; for each j do a block-wide reduction over i.
            for (int j = 0; j < v_dim; ++j) {
                float partial = 0.0f;
                for (int i = tid; i < k_dim; i += block_size) {
                    partial += recurrent_state[state_off + i * v_dim + j] *
                               workspace[kv_off + i];
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    kv_mem_lds[j] = shared_scratch[0];
                }
                __syncthreads();
            }

            // Step 3: delta[j] = (v_heads[h, j] - kv_mem[j]) * beta[h].
            for (int j = tid; j < v_dim; j += block_size) {
                const float v_val = workspace[v_off + j];
                delta_lds[j] = (v_val - kv_mem_lds[j]) * head_beta;
            }
            __syncthreads();

            // Step 4: state[h, i, j] += k_rep[h, i] * delta[j].
            for (int e = tid; e < total_state; e += block_size) {
                const int i = e / v_dim;
                const int j = e - i * v_dim;
                recurrent_state[state_off + e] +=
                    workspace[kv_off + i] * delta_lds[j];
            }
            __syncthreads();

            // Step 5: recurrent_out[h, j] = sum_i(state[h, i, j] * q_scaled[h, i]).
            for (int j = 0; j < v_dim; ++j) {
                float partial = 0.0f;
                for (int i = tid; i < k_dim; i += block_size) {
                    partial += recurrent_state[state_off + i * v_dim + j] *
                               workspace[qv_off + i];
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    // Match oracle's `recurrent_out.to(dtype)` — F32 → BF16
                    // cast happens here so Phase H reads BF16-precision
                    // values regardless of stage.
                    const float r = bf16_round_rne_f32(shared_scratch[0]);
                    workspace[OFF_REC_OUT + h * v_dim + j] = r;
                    if (stage == 4) {
                        output[h * v_dim + j] = static_cast<T>(r);
                    }
                }
                __syncthreads();
            }
        }
    }

    if (stage < 5) return;

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase H: per-V-head RMS norm + z-gate + multiply --------------------
    //
    // For each V-head h:
    //   out_normed[h, :] = rms_norm(rec_out[h, :], norm_w, eps)        (per head)
    //   z_silu[h, j]     = bf16(z_raw[h, j] * sigmoid(z_raw[h, j]))    (silu)
    //   out_gated[h, j]  = bf16(out_normed[h, j] * z_silu[h, j])
    //
    // Fuse the three into one per-(h, j) computation, writing out_gated
    // back into the OFF_REC_OUT slot — rec_out is no longer needed after
    // this point. Block-per-head; V=32 work units fit comfortably across
    // num_cus on gfx1100.
    {
        for (;;) {
            __shared__ unsigned int vh_s;
            if (tid == 0) {
                vh_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(vh_s);
            if (h >= V) break;

            const int rec_off = OFF_REC_OUT + h * v_dim;
            const int z_off   = OFF_Z_RAW   + h * v_dim;

            // RMS norm reduction over v_dim — F32 partial then block-reduce.
            float partial_sq = 0.0f;
            for (int j = tid; j < v_dim; j += block_size) {
                const float v = workspace[rec_off + j];
                partial_sq += v * v;
            }
            shared_scratch[tid] = partial_sq;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            __shared__ float head_inv_rms_norm;
            if (tid == 0) {
                head_inv_rms_norm =
                    rsqrtf(shared_scratch[0] / static_cast<float>(v_dim) + rms_norm_eps);
            }
            __syncthreads();

            // Per-element fused: out_normed * z_silu, BF16-rounded at every
            // dtype boundary so the path matches the oracle byte-for-byte.
            for (int j = tid; j < v_dim; j += block_size) {
                const float r       = workspace[rec_off + j];
                const float n_w     = static_cast<float>(norm_w[j]);
                const float on      = bf16_round_rne_f32(r * head_inv_rms_norm * n_w);
                const float z_val   = workspace[z_off + j];
                const float z_sig   = 1.0f / (1.0f + expf(-z_val));
                const float z_silu_val = bf16_round_rne_f32(z_val * z_sig);
                const float gated   = bf16_round_rne_f32(on * z_silu_val);
                workspace[rec_off + j] = gated;   // overwrites rec_out in-place
            }
            __syncthreads();
        }
    }

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // -- Phase I: out_proj matmul + residual --------------------------------
    //
    // out_gated lives at workspace[OFF_REC_OUT : OFF_REC_OUT + V*v_dim].
    // out_proj_w shape: [hidden, V*v_dim].
    //
    //   o_out[i]          = bf16(sum_j(out_proj_w[i, j] * out_gated[j]))
    //   output_hidden[i]  = bf16(input_hidden[i] + o_out[i])
    //
    // Two BF16 RNE rounds — one at the matmul boundary, one at the
    // residual add — exactly mirroring `o_out = out_flat @ out_proj.T`
    // and `output_hidden = input_hidden + o_out` in the oracle.
    {
        const int qd = V * v_dim;
        const bool out_int4 = (out_proj_scale != nullptr);

        // WMMA path: claim 128 output rows per atomicAdd, 8 waves × 16-row
        // tiles each. Same shape as FFN's Phase I (PR #77). The reduction
        // dim is `qd = V*v_dim` and the activation lives in
        // `workspace[OFF_REC_OUT..]` rather than `h_norm_lds`. INT4 only;
        // BF16 falls through to the scalar path below.
        bool wmma_handled_oi = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
        if constexpr (USE_WMMA) {
            if (out_int4) {
                const int gsc_o = qd / int4_group_size;
                const int wave_id = tid >> 5;
                const int lane_in_wave = tid & 31;
                const int lane_row = lane_in_wave & 15;
                const int lane_half = lane_in_wave >> 4;
                const float* mid_lds_f32 = workspace + OFF_REC_OUT;
                const uint8_t* slab_packed =
                    reinterpret_cast<const uint8_t*>(out_proj_w);
                for (;;) {
                    __shared__ unsigned int row_base_s;
                    if (tid == 0) row_base_s = atomicAdd(&counters[0], 128u);
                    __syncthreads();
                    const int row_base = static_cast<int>(row_base_s);
                    if (row_base >= hidden) break;

                    const int rhs_row = row_base + wave_id * 16 + lane_row;
                    const bool in_range = rhs_row < hidden;
                    const uint8_t* slab_row = in_range
                        ? slab_packed +
                          static_cast<size_t>(rhs_row) * (qd / 2)
                        : nullptr;
                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row,
                        static_cast<const hip_bfloat16*>(out_proj_scale),
                        static_cast<const hip_bfloat16*>(out_proj_zero),
                        rhs_row, in_range,
                        mid_lds_f32, qd,
                        gsc_o, int4_group_size, lane_row);
                    if (lane_half == 0 && in_range) {
                        const float o_out =
                            bf16_round_rne_f32(acc[0]);
                        const float in_f =
                            static_cast<float>(input_hidden[rhs_row]);
                        const float result =
                            bf16_round_rne_f32(in_f + o_out);
                        if (stage == 5) {
                            output[rhs_row] = static_cast<T>(result);
                        }
                    }
                    __syncthreads();
                }
                wmma_handled_oi = true;
            }
        }
#endif
        if (!wmma_handled_oi) {
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) {
                    my_row_s = atomicAdd(&counters[0], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= hidden) break;

                float partial = 0.0f;
                if (out_int4) {
                    partial = int4_dq8_matvec_partial(
                        reinterpret_cast<const uint8_t*>(out_proj_w),
                        static_cast<const hip_bfloat16*>(out_proj_scale),
                        static_cast<const hip_bfloat16*>(out_proj_zero),
                        workspace + OFF_REC_OUT,
                        my_row, qd, int4_group_size,
                        tid, block_size);
                } else {
                    const T* w_row = out_proj_w + static_cast<size_t>(my_row) * qd;
                    for (int j = tid; j < qd; j += block_size) {
                        partial += static_cast<float>(w_row[j]) * workspace[OFF_REC_OUT + j];
                    }
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    const float o_out  = bf16_round_rne_f32(shared_scratch[0]);
                    const float in_f   = static_cast<float>(input_hidden[my_row]);
                    const float result = bf16_round_rne_f32(in_f + o_out);
                    if (stage == 5) {
                        output[my_row] = static_cast<T>(result);
                    }
                }
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Single-block MoE FFN parity kernel (PR 4b4)
//
// Lives alongside the attn step kernel; together they cover the two halves
// of one Qwen3.6-MoE layer (attn = pre-residual, FFN = post-residual).
// Same staged build-up convention as PR 4b2/4b3 — each stage adds one slice
// of compute and the host parity test reads the matching intermediate.
//
//   stage 1: post_attn rmsnorm + router gate + softmax + topk + renorm → write topk_weights, topk_idx
//   stage 2: ... + shared expert (gate/up/silu/down + sigmoid_gate)    → write shared_out
//   stage 3: ... + ONE expert FFN (top-1 dispatch via topk_idx[0])     → write expert_0_out
//   stage 4: ... + topk-weighted sum across all k experts              → write moe_out
//   stage 5: ... + residual add (input_hidden + moe_out + shared_out)  → write output_hidden
//
// PR 4b4 step 1 lands stage 1. Stages 2–5 add to this kernel; the launcher's
// `stage` argument decides where to stop.
//
// Layout choices fixed up-front:
//   block_size = 256  (4 waves on gfx1100; matches the attn kernel for symmetry)
//   grid_size  = num_cus  (one block per CU, all guaranteed concurrent on RDNA3
//                          → grid_barrier safe without coop launch — same as
//                          the attn kernel and full_attention_4b.hip)
//
//   workspace[F32], staged-FFN scratch (offsets in F32 elements):
//     H_NORM        = 0                                              [hidden]    rms-normed input
//     ROUTER_LOGITS = hidden                                         [E]
//     ROUTER_PROBS  = hidden + E                                     [E]
//     TOPK_VAL      = hidden + 2*E                                   [k]   renormed
//     TOPK_IDX      = hidden + 2*E + k                               [k]   bit-cast u32 in float slot
//     SG_SCALAR     = hidden + 2*E + 2*k                             [1]   sigmoid(shared_expert_gate @ h_norm)
//     SGP           = hidden + 2*E + 2*k + 1                         [Is]  shared_gate_proj output (F32, no BF16 round)
//     SUP           = hidden + 2*E + 2*k + 1 + Is                    [Is]  shared_up_proj output
//     SHARED_MID    = hidden + 2*E + 2*k + 1 + 2*Is                  [Is]  silu(sgp) * sup (F32)
//     SHARED_OUT    = hidden + 2*E + 2*k + 1 + 3*Is                  [hidden] shared expert path output (BF16-rounded F32)
//     EXPERT_GU     = ... + hidden                                   [k * 2*I] per-expert gate_up_proj outputs (F32)
//                                                                    one [2*I] slab per concurrent expert group
//     EXPERT_MID    = ... + k*2*I                                    [k * I]   per-expert silu(gu[:I]) * gu[I:] (F32)
//                                                                    one [I] slab per concurrent expert group
//     EXPERT_STACK  = ... + k*I                                      [k * hidden] per-expert outs (F32, no BF16 round)
//     MOE_OUT       = ... + k*hidden                                 [hidden] BF16-rounded F32
//     OUTPUT_HIDDEN = ... + hidden                                   [hidden]
//   For 35B-A3B (hidden=2048, E=256, k=8, Is=512, I=512) stage-5 footprint
//   is ~38,929 F32 = ~152 KiB after EXPERT_GU/MID grew K_top-wide so all
//   `top_k` experts can run G/H/I concurrently (one block-group per expert,
//   `num_blocks / k` blocks per group on gfx1100 = 96/8 = 12). Still well
//   within the persistent-decode budget.
//
//   Numerics: every per-expert intermediate (EXPERT_GU, EXPERT_MID,
//   EXPERT_STACK) stays raw F32 to mirror the oracle's all-F32 chain
//   (`mid @ down_proj.T` accumulates F32, then `topk_w * expert_stack`
//   is summed in F32). The only BF16 rounds in the per-expert path are
//   at (a) stage 3's test-output write where we expose one expert's
//   output BF16-cast just like the oracle's `expert_stack.to(dtype)`,
//   and (b) stage 4's moe_out where the topk-weighted sum is finally
//   rounded — matching `moe_out.sum(0).to(dtype)` in the oracle.
//
// Parity output convention (BF16, sized for the largest staged intermediate):
//   stage 1: output[0..k] = topk_weights  (BF16-rounded; renormed across k)
//            output_idx[0..k] = topk_idx  (int32, separate buffer)
//   stage 2..5: output written by the matching final phase (see comments
//   on each stage in subsequent commits).

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase G (per-expert
// gate_up_proj GEMV) and Phase I (per-expert down_proj GEMV). When true,
// each block claims 128 rows per atomicAdd and dispatches them to its 8
// waves (16 rows per wave) running `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`
// against INT4-dequant-on-the-fly weight tiles. Phase D/F (shared expert)
// stays on the scalar path in both variants — it's ~1/8 the per-token
// matmul cost and will be a follow-up. The bridge picks the instantiation
// at launch time based on `device_supports_wmma_bf16(ordinal)` and dim
// divisibility checks.
template <typename T, bool USE_WMMA = false>
__global__ void qwen36_moe_ffn_step_kernel(
    int                            stage,
    int                            hidden,
    int                            num_experts,                  // E (256 on 35B-A3B)
    int                            moe_intermediate,             // I (512 on 35B-A3B)
    int                            shared_intermediate,          // Is (512 on 35B-A3B)
    int                            top_k,                        // k (8 on 35B-A3B)
    float                          rms_norm_eps,
    const T* __restrict__          input_hidden,                 // [hidden] post-attn
    const T* __restrict__          post_attn_norm_w,             // [hidden]
    const T* __restrict__          gate_w,                       // [E, hidden]
    const T* __restrict__          gate_up_proj_w,               // [E, 2*I, hidden]   (nullable for stage<3)
    const T* __restrict__          down_proj_w,                  // [E, hidden, I]     (nullable for stage<3)
    const T* __restrict__          shared_gate_proj_w,           // [Is, hidden]       (nullable for stage<2)
    const T* __restrict__          shared_up_proj_w,             // [Is, hidden]       (nullable for stage<2)
    const T* __restrict__          shared_down_proj_w,           // [hidden, Is]       (nullable for stage<2)
    const T* __restrict__          shared_expert_gate_w,         // [1, hidden]        (nullable for stage<2)
    // PR 4b5 INT4 sidecars. `int4_group_size == 0` ⇒ all weights are BF16
    // and every sidecar pointer must be null. When non-zero, each sidecar
    // pair (`*_scale`, `*_zero`) is consulted independently per tensor: a
    // null `*_scale` for a given tensor keeps it on the BF16 path, a
    // non-null pair switches that tensor to INT4. The kernel reinterprets
    // the matching weight pointer as `const uint8_t*` packed bytes (low
    // nibble = even col) and dequants per-element via `int4_dequant_scalar`.
    int                                  int4_group_size,
    const hip_bfloat16* __restrict__     gate_up_proj_scale,
    const hip_bfloat16* __restrict__     gate_up_proj_zero,
    const hip_bfloat16* __restrict__     down_proj_scale,
    const hip_bfloat16* __restrict__     down_proj_zero,
    const hip_bfloat16* __restrict__     shared_gate_proj_scale,
    const hip_bfloat16* __restrict__     shared_gate_proj_zero,
    const hip_bfloat16* __restrict__     shared_up_proj_scale,
    const hip_bfloat16* __restrict__     shared_up_proj_zero,
    const hip_bfloat16* __restrict__     shared_down_proj_scale,
    const hip_bfloat16* __restrict__     shared_down_proj_zero,
    T* __restrict__                output,                       // BF16, sized for the largest staged intermediate
    int* __restrict__              output_idx,                   // [k] topk indices (stage 1+)
    float* __restrict__            workspace,
    unsigned int* __restrict__     counters,
    unsigned int* __restrict__     barrier_counter,
    unsigned int* __restrict__     barrier_flag) {
    // gate_up_proj_w, down_proj_w are used at stage>=3.
    // shared_*_w are used at stage>=2.
    // INT4 sidecars (`*_scale`/`*_zero`) are read only for tensors whose
    // `*_scale` pointer is non-null. The router (`gate_w`) and the scalar
    // shared-expert gate (`shared_expert_gate_w`) are always BF16 — the
    // INT4 bake excludes them, matching `crates/qwen36_moe/src/weights.rs`.

    const int num_blocks = static_cast<int>(gridDim.x);
    const int tid        = threadIdx.x;
    const int block_size = static_cast<int>(blockDim.x);

    extern __shared__ float lds[];
    float* h_norm_lds     = lds;             // [hidden]
    float* shared_scratch = lds + hidden;    // [block_size]

    const int OFF_H_NORM        = 0;
    const int OFF_ROUTER_LOGITS = hidden;
    const int OFF_ROUTER_PROBS  = hidden + num_experts;
    const int OFF_TOPK_VAL      = hidden + 2 * num_experts;
    const int OFF_TOPK_IDX      = hidden + 2 * num_experts + top_k;
    const int OFF_SG_SCALAR     = hidden + 2 * num_experts + 2 * top_k;
    const int OFF_SGP           = OFF_SG_SCALAR + 1;
    const int OFF_SUP           = OFF_SGP + shared_intermediate;
    const int OFF_SHARED_MID    = OFF_SUP + shared_intermediate;
    const int OFF_SHARED_OUT    = OFF_SHARED_MID + shared_intermediate;
    const int OFF_EXPERT_GU     = OFF_SHARED_OUT + hidden;
    const int OFF_EXPERT_MID    = OFF_EXPERT_GU + top_k * 2 * moe_intermediate;
    const int OFF_EXPERT_STACK  = OFF_EXPERT_MID + top_k * moe_intermediate;
    const int OFF_MOE_OUT       = OFF_EXPERT_STACK + top_k * hidden;

    // -- Phase A: load input + post_attn RMS norm --------------------------
    // Same idiom as the attn kernel's Phase A. Every block computes h_norm
    // into its own LDS (cheap reduction; cross-block sync would cost more).
    for (int col = tid; col < hidden; col += block_size) {
        h_norm_lds[col] = static_cast<float>(input_hidden[col]);
    }
    __syncthreads();

    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        partial_sq += h_norm_lds[col] * h_norm_lds[col];
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float inv_rms_input;
    if (tid == 0) {
        inv_rms_input = rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // BF16-round each h_norm element so the matmul reads what PyTorch reads.
    // HF `Qwen3_5MoeRMSNorm` `(1.0 + weight)` unit offset for
    // `post_attention_layernorm`. Block 0 also stages the F32 of-BF16 view
    // into workspace[OFF_H_NORM] for any later stages that prefer global
    // memory.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = static_cast<float>(post_attn_norm_w[col]);
        const float v = bf16_round_rne_f32(h_norm_lds[col] * inv_rms_input * (1.0f + w));
        h_norm_lds[col] = v;
        if (blockIdx.x == 0) {
            workspace[OFF_H_NORM + col] = v;
        }
    }
    __syncthreads();

    // -- Phase B: router_logits = gate_w @ h_norm  (work-stealing matvec) --
    // 256 output rows on 35B-A3B; with hidden=2048 that's a small mat-vec.
    // Each block claims rows from `counters[0]` and reduces across its
    // threads. Same primitive as the attn kernel's q_proj phase.
    for (;;) {
        __shared__ unsigned int my_row_s;
        if (tid == 0) {
            my_row_s = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const int my_row = static_cast<int>(my_row_s);
        if (my_row >= num_experts) break;

        const T* w_row = gate_w + static_cast<size_t>(my_row) * hidden;
        float partial = 0.0f;
        for (int col = tid; col < hidden; col += block_size) {
            partial += static_cast<float>(w_row[col]) * h_norm_lds[col];
        }
        shared_scratch[tid] = partial;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            // Match the oracle's `(h_norm.f32 @ gate_w.f32.T).to(dtype)`:
            // F32 internals, BF16-rounded on store.
            workspace[OFF_ROUTER_LOGITS + my_row] = bf16_round_rne_f32(shared_scratch[0]);
        }
        __syncthreads();
    }

    // Cross-block barrier: all router logits finalised before block 0 reduces.
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

    // -- Phase C: softmax + topk + renorm (block 0 only) -------------------
    // 256 logits softmax fits in a single block trivially. We:
    //   1. Find max for numerical stability.
    //   2. exp(logit - max), sum.
    //   3. Divide → router_probs[E] (F32, BF16-rounded for parity).
    //   4. Top-k via k iterations of "find argmax, mark used".
    //   5. Sum the top-k probs, divide each by the sum → renormed weights.
    if (blockIdx.x == 0) {
        const int E = num_experts;
        const int K = top_k;

        // Step 1: max reduction. Threads read logits and produce a per-thread
        // max; tree-reduce in shared_scratch.
        float local_max = -INFINITY;
        for (int i = tid; i < E; i += block_size) {
            const float v = workspace[OFF_ROUTER_LOGITS + i];
            if (v > local_max) local_max = v;
        }
        shared_scratch[tid] = local_max;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const float a = shared_scratch[tid];
                const float b = shared_scratch[tid + s];
                shared_scratch[tid] = (a > b) ? a : b;
            }
            __syncthreads();
        }
        __shared__ float row_max;
        if (tid == 0) row_max = shared_scratch[0];
        __syncthreads();

        // Step 2: exp(logit - max), partial sum.
        float local_sum = 0.0f;
        for (int i = tid; i < E; i += block_size) {
            const float e = expf(workspace[OFF_ROUTER_LOGITS + i] - row_max);
            workspace[OFF_ROUTER_PROBS + i] = e;
            local_sum += e;
        }
        shared_scratch[tid] = local_sum;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
            __syncthreads();
        }
        __shared__ float row_sum;
        if (tid == 0) row_sum = shared_scratch[0];
        __syncthreads();

        // Step 3: divide. Store BF16-rounded F32 (matches oracle's `.to(dtype)`).
        const float inv_row_sum = 1.0f / row_sum;
        for (int i = tid; i < E; i += block_size) {
            const float p = bf16_round_rne_f32(workspace[OFF_ROUTER_PROBS + i] * inv_row_sum);
            workspace[OFF_ROUTER_PROBS + i] = p;
        }
        __syncthreads();

        // Step 4: top-k via k iterations. Each iter does a 256-wide argmax
        // reduction on `router_probs`, then masks the winner. With K=8 this
        // is 8 cycles → fine for a megakernel; a heap would beat it for
        // larger k but k=8 is fixed in the model card.
        //
        // Tie-breaking: torch.topk returns the lowest index on ties; we
        // match by carrying both (value, index) through the tree-reduce
        // and preferring the lower index when values are equal.
        //
        // Static __shared__ buffers for the (val, idx) reduction. Sized to
        // the fixed block_size=256; if block_size ever grows, increase
        // these in lockstep. Allocated once at kernel entry to avoid the
        // re-declare-inside-loop pitfall some compilers handle oddly.
        __shared__ int   idx_buf[256];
        __shared__ float val_buf[256];
        for (int kk = 0; kk < K; kk++) {
            // Per-thread argmax over the strided slice.
            int   local_idx = -1;
            float local_val = -INFINITY;
            for (int i = tid; i < E; i += block_size) {
                const float v = workspace[OFF_ROUTER_PROBS + i];
                if (v > local_val ||
                    (v == local_val && local_idx >= 0 && i < local_idx)) {
                    local_val = v;
                    local_idx = i;
                }
            }
            idx_buf[tid] = local_idx;
            val_buf[tid] = local_val;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const float a  = val_buf[tid];
                    const float b  = val_buf[tid + s];
                    const int   ia = idx_buf[tid];
                    const int   ib = idx_buf[tid + s];
                    bool take_b = false;
                    if (b > a) take_b = true;
                    else if (b == a && ib >= 0 &&
                             (ia < 0 || ib < ia)) take_b = true;
                    if (take_b) {
                        val_buf[tid] = b;
                        idx_buf[tid] = ib;
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int   winner_idx = idx_buf[0];
                const float winner_val = val_buf[0];
                workspace[OFF_TOPK_IDX + kk] = __int_as_float(winner_idx);
                workspace[OFF_TOPK_VAL + kk] = winner_val;
                // Mask the winner so the next iter picks the next-largest.
                workspace[OFF_ROUTER_PROBS + winner_idx] = -INFINITY;
            }
            __syncthreads();
        }

        // Step 5: renormalise the top-k probs to sum to 1.
        // Sequential on tid==0 — k is tiny (8); a parallel reduction would
        // be code-bulk for no measurable win.
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) sum += workspace[OFF_TOPK_VAL + i];
            const float inv = 1.0f / sum;
            for (int i = 0; i < K; i++) {
                const float w = bf16_round_rne_f32(workspace[OFF_TOPK_VAL + i] * inv);
                workspace[OFF_TOPK_VAL + i] = w;
                if (stage == 1) {
                    output[i]      = static_cast<T>(w);
                    output_idx[i]  = __float_as_int(workspace[OFF_TOPK_IDX + i]);
                }
            }
        }
        __syncthreads();
    }

    if (stage < 2) {
        return;
    }

    // -- Phase D: shared-expert input projections (work-stealing matvec) ---
    // Three matrices read out of one work-stealing loop. Row index space:
    //   [0,             Is)            → sgp[r]    via shared_gate_proj_w
    //   [Is,            2*Is)          → sup[r-Is] via shared_up_proj_w
    //   [2*Is]                          → sg_scalar = sigmoid(shared_expert_gate_w @ h_norm)
    // sgp and sup are stored as raw F32 (no BF16 round) so the silu*mul in
    // Phase E matches the oracle's all-F32 path. sg_scalar likewise stays
    // F32 — the only BF16 round in this stage is on the final shared_out.
    //
    // Block 0 finished topk; sync everyone before Phase D and reset the
    // work-stealing counter (Phase B left it at >=E from the gate matmul).
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    const int Is_ = shared_intermediate;
    const int total_rows_d = 2 * Is_ + 1;
    for (;;) {
        __shared__ unsigned int my_row_d_s;
        if (tid == 0) {
            my_row_d_s = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const int my_row = static_cast<int>(my_row_d_s);
        if (my_row >= total_rows_d) break;

        const T* w_row = nullptr;                          // BF16 path
        const void* i4_slab = nullptr;                     // INT4 path: full [out, hidden/2] u8
        const hip_bfloat16* i4_scale = nullptr;
        const hip_bfloat16* i4_zero  = nullptr;
        int      i4_row = 0;                               // row index within the slab
        int      write_offset;
        bool     is_gate_row = false;
        if (my_row < Is_) {
            if (shared_gate_proj_scale != nullptr) {
                i4_slab  = reinterpret_cast<const void*>(shared_gate_proj_w);
                i4_scale = shared_gate_proj_scale;
                i4_zero  = shared_gate_proj_zero;
                i4_row   = my_row;
            } else {
                w_row = shared_gate_proj_w + static_cast<size_t>(my_row) * hidden;
            }
            write_offset = OFF_SGP + my_row;
        } else if (my_row < 2 * Is_) {
            const int local = my_row - Is_;
            if (shared_up_proj_scale != nullptr) {
                i4_slab  = reinterpret_cast<const void*>(shared_up_proj_w);
                i4_scale = shared_up_proj_scale;
                i4_zero  = shared_up_proj_zero;
                i4_row   = local;
            } else {
                w_row = shared_up_proj_w + static_cast<size_t>(local) * hidden;
            }
            write_offset = OFF_SUP + local;
        } else {
            // Single-row "[1, hidden]" weight; dot it with h_norm and apply
            // sigmoid() before storing into SG_SCALAR. Always BF16 — the
            // INT4 bake excludes `shared_expert_gate`.
            w_row        = shared_expert_gate_w;
            write_offset = OFF_SG_SCALAR;
            is_gate_row  = true;
        }

        float partial = 0.0f;
        if (i4_scale != nullptr) {
            partial = int4_dq8_matvec_partial(
                static_cast<const uint8_t*>(i4_slab),
                i4_scale, i4_zero, h_norm_lds,
                i4_row, hidden, int4_group_size,
                tid, block_size);
        } else {
            for (int col = tid; col < hidden; col += block_size) {
                partial += static_cast<float>(w_row[col]) * h_norm_lds[col];
            }
        }
        shared_scratch[tid] = partial;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float val = shared_scratch[0];
            if (is_gate_row) {
                // sigmoid in F32 (matches oracle's `torch.sigmoid(sg_logit)`).
                val = 1.0f / (1.0f + expf(-val));
            }
            workspace[write_offset] = val;
        }
        __syncthreads();
    }

    // -- Phase E: smid = silu(sgp) * sup  (block-0 only, F32 throughout) ---
    // Block 0 owns the write so we don't have multiple blocks racing on
    // `workspace[OFF_SHARED_MID..]`. The grid_barrier below makes the
    // writes visible to every block before Phase F starts.
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    if (blockIdx.x == 0) {
        for (int i = tid; i < Is_; i += block_size) {
            const float gp     = workspace[OFF_SGP + i];
            const float up     = workspace[OFF_SUP + i];
            const float sigmoid_gp = 1.0f / (1.0f + expf(-gp));
            const float silu_gp    = gp * sigmoid_gp;
            // No BF16 round — oracle keeps this F32 until the final cast.
            workspace[OFF_SHARED_MID + i] = silu_gp * up;
        }
        __syncthreads();
    }

    // -- Phase F: shared_out = sg_scalar * (smid @ down_proj.T) ------------
    // Work-stealing matvec across `hidden` rows. The sg_scalar multiply is
    // folded into the per-row write; BF16-round happens once at the end
    // (matches `(sg_scalar * sdown).to(dtype)` in the oracle).
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    {
        // Read sg_scalar once per block; stays in registers.
        const float sg_scalar = workspace[OFF_SG_SCALAR];
        for (;;) {
            __shared__ unsigned int my_row_f_s;
            if (tid == 0) {
                my_row_f_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_f_s);
            if (my_row >= hidden) break;

            float partial = 0.0f;
            if (shared_down_proj_scale != nullptr) {
                // INT4 path: row `my_row` of `[hidden, Is]` packed as
                // `[hidden, Is/2]` u8.
                partial = int4_dq8_matvec_partial(
                    reinterpret_cast<const uint8_t*>(shared_down_proj_w),
                    static_cast<const hip_bfloat16*>(shared_down_proj_scale),
                    static_cast<const hip_bfloat16*>(shared_down_proj_zero),
                    workspace + OFF_SHARED_MID,
                    my_row, Is_, int4_group_size,
                    tid, block_size);
            } else {
                const T* w_row =
                    shared_down_proj_w + static_cast<size_t>(my_row) * Is_;
                for (int col = tid; col < Is_; col += block_size) {
                    partial += static_cast<float>(w_row[col])
                               * workspace[OFF_SHARED_MID + col];
                }
            }
            shared_scratch[tid] = partial;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const float val = bf16_round_rne_f32(sg_scalar * shared_scratch[0]);
                workspace[OFF_SHARED_OUT + my_row] = val;
                if (stage == 2) {
                    output[my_row] = static_cast<T>(val);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 3) {
        return;
    }

    // -- Phase G/H/I: per-expert FFN — concurrent across all top_k experts -
    // Previous revision iterated `j = 0 .. k_iters` sequentially with three
    // grid barriers per iteration (~24 total at K_top=8). This dispatch
    // assigns every block to one of `K_top` expert "groups" via cyclic
    // partitioning and runs all groups in parallel for each of G, H, I —
    // collapsing to three grid barriers total.
    //
    // Block partitioning (cyclic; distributes any leftover blocks evenly):
    //   group_id = blockIdx.x % top_k
    //   sub_id   = blockIdx.x / top_k
    // For 35B-A3B on gfx1100 (num_blocks=96, top_k=8) every group gets
    // exactly 12 blocks. On other geometries the residual `num_blocks %
    // top_k` blocks fall into the early groups; those groups simply have
    // one extra block contending the same counter slot, which is fine.
    //
    // Workspace layout (set up at the OFF_* constants above):
    //   EXPERT_GU [k * 2*I]   — group g writes [g*2*I .. (g+1)*2*I)
    //   EXPERT_MID [k * I]    — group g writes [g*I   .. (g+1)*I)
    //   EXPERT_STACK [k * hidden] — group g writes [g*hidden .. (g+1)*hidden)
    //
    // Counter slots (in `counters[]`, all zeroed by `reset_sync_buf` on the
    // host before each launch — see the sync_buf docstring in
    // `crates/kernel-ffi/src/qwen36_moe.rs`):
    //   counters[g]            — Phase G work-stealing for group g (0..2*I)
    //   counters[top_k + g]    — Phase I work-stealing for group g (0..hidden)
    //   counters[0]            — re-used by Phase J / K (reset before each)
    //
    // Phase shape:
    //   Phase G (parallel across active groups, work-stealing on counters[g])
    //   → grid_barrier
    //   Phase H (one block per active group: sub_id==0, distributes I
    //            elements via tid-stride; other blocks idle)
    //   → grid_barrier
    //   Phase I (parallel, work-stealing on counters[top_k+g])
    //   → if (stage < 4) return            ← stage 3 exits here
    //   → grid_barrier_reset_counter(&counters[0])  ← gate Phase J
    //
    // Stage 3 dispatches only group 0 by setting `active_groups = 1`. All
    // blocks still hit the unconditional barriers (the grid barrier needs
    // every block to participate or it hangs); only the work bodies are
    // gated on `group_active`. Stage 3's parity-output write fires when
    // `group_id == 0` (mirrors the old `j == 0` condition).
    //
    // Within each group g (`e = topk_idx[g]`):
    //   gu  = gate_up_proj_w[e] @ h_norm                            [2*I]
    //   mid = silu(gu[:I]) * gu[I:]                                 [I]
    //   eo  = down_proj_w[e].T @ mid                                [hidden]
    // EXPERT_STACK keeps `eo` raw F32 so stage 4 doesn't lose precision
    // through a BF16 round-trip in the weighted sum.

    const int I_ = moe_intermediate;
    const int two_I = 2 * I_;
    const int K_ = top_k;
    const int active_groups = (stage == 3) ? 1 : K_;
    const int group_id = blockIdx.x % K_;
    const bool group_active = (group_id < active_groups);

    // Phase F left counters[0] at `hidden` (atomicAdds during work-stealing).
    // Phase G of group 0 uses that same slot, so reset it here before any
    // group reads its counter. counters[1..K_] and counters[K_..2*K_] were
    // zeroed by `reset_sync_buf` before launch and stay zero through all of
    // attn/Phase A..F (which only touch counters[0]).
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    // Each block reads its own group's expert index. All blocks within a
    // group compute the same `e`; different groups compute different `e`.
    __shared__ int s_expert_idx;
    if (tid == 0) {
        s_expert_idx = __float_as_int(workspace[OFF_TOPK_IDX + group_id]);
    }
    __syncthreads();
    const int e = s_expert_idx;

    // Phase G: matvec gate_up_proj_w[e] @ h_norm → EXPERT_GU[g*2*I..].
    // BF16 layout: gate_up_proj_w[E, 2*I, hidden] — slab `e` base is
    // `e * 2*I * hidden`; row `r` adds `r * hidden`.
    // INT4 layout (when `gate_up_proj_scale != nullptr`):
    //   weights : [E, 2*I, hidden/2] u8   — slab `e` base in bytes is
    //                                        `e * 2*I * (hidden/2)`.
    //   scale   : [E, 2*I/gsz, hidden/gsz] BF16 — slab `e` base in
    //                                        BF16 elems is
    //                                        `e * (2*I/gsz) * (hidden/gsz)`.
    if (group_active) {
        const bool gu_int4 = (gate_up_proj_scale != nullptr);
        const T* gu_slab_bf16 =
            gate_up_proj_w + static_cast<size_t>(e) * two_I * hidden;
        const uint8_t*      gu_slab_packed = nullptr;
        const hip_bfloat16* gu_slab_scale  = nullptr;
        const hip_bfloat16* gu_slab_zero   = nullptr;
        if (gu_int4) {
            gu_slab_packed = reinterpret_cast<const uint8_t*>(gate_up_proj_w)
                           + static_cast<size_t>(e) * two_I
                             * (hidden / 2);
            const int gsr = two_I  / int4_group_size;
            const int gsc = hidden / int4_group_size;
            gu_slab_scale = gate_up_proj_scale
                          + static_cast<size_t>(e) * gsr * gsc;
            gu_slab_zero  = gate_up_proj_zero
                          + static_cast<size_t>(e) * gsr * gsc;
        }
        const int gu_off = OFF_EXPERT_GU + group_id * two_I;

        // WMMA path: each block claims 128 rows per atomicAdd; 8 waves do
        // 16-row WMMA tiles in parallel. Same scale/zero/packed slab base
        // as the scalar path; only the inner loop changes. Reduction across
        // K is done by the WMMA F32 accumulator — no LDS reduction needed,
        // and per-lane writes (one per output row) replace the per-block
        // single-row write. INT4 weights only; BF16 weights fall through
        // to the scalar path below.
        bool wmma_handled_g = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
        if constexpr (USE_WMMA) {
            if (gu_int4) {
                const int gsc_gu = hidden / int4_group_size;
                const int wave_id = tid >> 5;
                const int lane = tid & 31;
                const int lane_row = lane & 15;
                const int lane_half = lane >> 4;
                for (;;) {
                    __shared__ unsigned int row_base_s;
                    if (tid == 0) {
                        row_base_s = atomicAdd(&counters[group_id], 128u);
                    }
                    __syncthreads();
                    const int row_base_block = static_cast<int>(row_base_s);
                    if (row_base_block >= two_I) break;

                    const int rhs_row_idx =
                        row_base_block + wave_id * 16 + lane_row;
                    const bool rhs_in_range = rhs_row_idx < two_I;
                    const uint8_t* slab_row = rhs_in_range
                        ? gu_slab_packed +
                          static_cast<size_t>(rhs_row_idx) * (hidden / 2)
                        : nullptr;

                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row, gu_slab_scale, gu_slab_zero,
                        rhs_row_idx, rhs_in_range,
                        h_norm_lds, hidden,
                        gsc_gu, int4_group_size, lane_row);

                    // Lanes with lane_half==0 hold C[0, lane_row] in acc[0]
                    // — the M=0 row of the WMMA C tile, one element per
                    // output row.
                    if (lane_half == 0 && rhs_in_range) {
                        workspace[gu_off + rhs_row_idx] = acc[0];
                    }
                    __syncthreads();
                }
                wmma_handled_g = true;
            }
        }
#endif
        if (!wmma_handled_g) {
            // Scalar work-stealing path (covers both BF16 and INT4 via the
            // `gu_int4` branch). Unchanged from the pre-WMMA implementation;
            // also serves as the parity-stable reference when WMMA is
            // disabled by env or unavailable on the device.
            for (;;) {
                __shared__ unsigned int my_row_g_s;
                if (tid == 0) {
                    my_row_g_s = atomicAdd(&counters[group_id], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_g_s);
                if (my_row >= two_I) break;

                float partial = 0.0f;
                if (gu_int4) {
                    partial = int4_dq8_matvec_partial(
                        gu_slab_packed, gu_slab_scale, gu_slab_zero,
                        h_norm_lds,
                        my_row, hidden, int4_group_size,
                        tid, block_size);
                } else {
                    const T* w_row =
                        gu_slab_bf16 + static_cast<size_t>(my_row) * hidden;
                    for (int col = tid; col < hidden; col += block_size) {
                        partial += static_cast<float>(w_row[col])
                                   * h_norm_lds[col];
                    }
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    workspace[gu_off + my_row] = shared_scratch[0];
                }
                __syncthreads();
            }
        }
    }

    // Barrier 1: publish EXPERT_GU writes (all groups) before Phase H reads.
    // Every block must reach the barrier even if its group is inactive.
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

    // Phase H: mid = silu(gu[:I]) * gu[I:], one block per active group.
    // sub_id 0 of each active group does its own group's I elements; with
    // block_size=256 and I<=1024 a single block easily covers the slab.
    // Other blocks idle (they hit barrier 2 below).
    {
        const int sub_id = blockIdx.x / K_;
        if (sub_id == 0 && group_active) {
            const int gu_off  = OFF_EXPERT_GU + group_id * two_I;
            const int mid_off = OFF_EXPERT_MID + group_id * I_;
            for (int i = tid; i < I_; i += block_size) {
                const float gp = workspace[gu_off + i];
                const float up = workspace[gu_off + I_ + i];
                const float sigmoid_gp = 1.0f / (1.0f + expf(-gp));
                const float silu_gp    = gp * sigmoid_gp;
                workspace[mid_off + i] = silu_gp * up;
            }
            __syncthreads();
        }
    }

    // Barrier 2: publish EXPERT_MID writes (all groups) before Phase I reads.
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

    // Phase I: matvec down_proj_w[e].T @ mid → EXPERT_STACK[g*hidden..].
    // BF16 layout: down_proj_w[E, hidden, I]; slab `e` base is
    // `e * hidden * I`, row `r` adds `r * I`. Contracts over I → one
    // F32 per output row.
    // INT4 layout (when `down_proj_scale != nullptr`):
    //   weights : [E, hidden, I/2] u8   — slab `e` base in bytes is
    //                                      `e * hidden * (I/2)`.
    //   scale   : [E, hidden/gsz, I/gsz] BF16 — slab `e` base in BF16
    //                                      elems is
    //                                      `e * (hidden/gsz) * (I/gsz)`.
    // Stage-3 parity output (BF16 cast of EXPERT_STACK[0]) is emitted by
    // group 0 only — same per-row write the previous `j==0` branch did.
    if (group_active) {
        const bool dp_int4 = (down_proj_scale != nullptr);
        const T* dp_slab_bf16 =
            down_proj_w + static_cast<size_t>(e) * hidden * I_;
        const uint8_t*      dp_slab_packed = nullptr;
        const hip_bfloat16* dp_slab_scale  = nullptr;
        const hip_bfloat16* dp_slab_zero   = nullptr;
        if (dp_int4) {
            dp_slab_packed = reinterpret_cast<const uint8_t*>(down_proj_w)
                           + static_cast<size_t>(e) * hidden
                             * (I_ / 2);
            const int gsr = hidden / int4_group_size;
            const int gsc = I_     / int4_group_size;
            dp_slab_scale = down_proj_scale
                          + static_cast<size_t>(e) * gsr * gsc;
            dp_slab_zero  = down_proj_zero
                          + static_cast<size_t>(e) * gsr * gsc;
        }
        const int slot_off = OFF_EXPERT_STACK + group_id * hidden;
        const int mid_off  = OFF_EXPERT_MID   + group_id * I_;

        // WMMA path: same shape as Phase G's. The reduction dim here is
        // `I_` (per-expert intermediate, e.g. 512 on 35B-A3B) and the
        // output dim is `hidden` (e.g. 2048). The activation lives in
        // `workspace[mid_off..]` rather than `h_norm_lds`. Stage-3's
        // BF16-cast parity write for routed group 0 is preserved per-lane.
        bool wmma_handled_i = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
        if constexpr (USE_WMMA) {
            if (dp_int4) {
                const int gsc_dp = I_ / int4_group_size;
                const int wave_id = tid >> 5;
                const int lane = tid & 31;
                const int lane_row = lane & 15;
                const int lane_half = lane >> 4;
                const float* mid_lds_f32 = workspace + mid_off;
                for (;;) {
                    __shared__ unsigned int row_base_s;
                    if (tid == 0) {
                        row_base_s = atomicAdd(&counters[K_ + group_id], 128u);
                    }
                    __syncthreads();
                    const int row_base_block = static_cast<int>(row_base_s);
                    if (row_base_block >= hidden) break;

                    const int rhs_row_idx =
                        row_base_block + wave_id * 16 + lane_row;
                    const bool rhs_in_range = rhs_row_idx < hidden;
                    const uint8_t* slab_row = rhs_in_range
                        ? dp_slab_packed +
                          static_cast<size_t>(rhs_row_idx) * (I_ / 2)
                        : nullptr;

                    qwen36_float8 acc = wmma_int4_matvec_partial_16rows(
                        slab_row, dp_slab_scale, dp_slab_zero,
                        rhs_row_idx, rhs_in_range,
                        mid_lds_f32, I_,
                        gsc_dp, int4_group_size, lane_row);

                    if (lane_half == 0 && rhs_in_range) {
                        const float val = acc[0];
                        workspace[slot_off + rhs_row_idx] = val;
                        if (stage == 3 && group_id == 0) {
                            output[rhs_row_idx] =
                                static_cast<T>(bf16_round_rne_f32(val));
                        }
                    }
                    __syncthreads();
                }
                wmma_handled_i = true;
            }
        }
#endif
        if (!wmma_handled_i) {
            for (;;) {
                __shared__ unsigned int my_row_i_s;
                if (tid == 0) {
                    my_row_i_s = atomicAdd(&counters[K_ + group_id], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_i_s);
                if (my_row >= hidden) break;

                float partial = 0.0f;
                if (dp_int4) {
                    partial = int4_dq8_matvec_partial(
                        dp_slab_packed, dp_slab_scale, dp_slab_zero,
                        workspace + mid_off,
                        my_row, I_, int4_group_size,
                        tid, block_size);
                } else {
                    const T* w_row =
                        dp_slab_bf16 + static_cast<size_t>(my_row) * I_;
                    for (int col = tid; col < I_; col += block_size) {
                        partial += static_cast<float>(w_row[col])
                                   * workspace[mid_off + col];
                    }
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    const float val = shared_scratch[0];
                    workspace[slot_off + my_row] = val;
                    if (stage == 3 && group_id == 0) {
                        // Match oracle's `expert_stack.to(dtype)` exposure.
                        output[my_row] = static_cast<T>(bf16_round_rne_f32(val));
                    }
                }
                __syncthreads();
            }
        }
    }

    if (stage < 4) {
        return;
    }

    // -- Phase J: moe_out = sum_j (topk_w[j] * expert_stack[j])  -----------
    // Single weighted reduction across `k` experts. Each output row `r` is
    // independent → trivial work-stealing across hidden rows. The k loop
    // inside the per-row work is short (k=8 on 35B-A3B); reading topk_w[j]
    // and expert_stack[j*hidden + r] from workspace is cheap.
    //
    // Numerics: F32 accumulation, BF16-round once at the store. Matches
    // `(topk_w_renorm.unsqueeze(-1) * expert_stack).sum(0).to(dtype)` in
    // the oracle.
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    {
        for (;;) {
            __shared__ unsigned int my_row_j_s;
            if (tid == 0) {
                my_row_j_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_j_s);
            if (my_row >= hidden) break;

            // Sequential per-row sum on a single thread (k=8). The row's
            // 8 reads are coalesced across threads only if we parallelise
            // by k — not worth the LDS shuffle for a tiny per-row sum.
            if (tid == 0) {
                float acc = 0.0f;
                for (int j = 0; j < K_; j++) {
                    const float w = workspace[OFF_TOPK_VAL + j];
                    const float v = workspace[OFF_EXPERT_STACK + j * hidden + my_row];
                    acc += w * v;
                }
                const float val = bf16_round_rne_f32(acc);
                workspace[OFF_MOE_OUT + my_row] = val;
                if (stage == 4) {
                    output[my_row] = static_cast<T>(val);
                }
            }
            __syncthreads();
        }
    }

    if (stage < 5) {
        return;
    }

    // -- Phase K: residual add — output_hidden = input + moe + shared  -----
    // The trivial closing slice. Each output row is independent; F32 sum
    // with one BF16 round at the store, matching the oracle's
    // `(input_hidden.f32 + moe_out.f32 + shared_out.f32).to(dtype)`.
    //
    // moe_out is already in workspace at OFF_MOE_OUT (BF16-rounded F32);
    // shared_out is at OFF_SHARED_OUT (BF16-rounded F32). We add the
    // input_hidden (BF16 → F32 promotion) on top.
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    for (;;) {
        __shared__ unsigned int my_row_k_s;
        if (tid == 0) {
            my_row_k_s = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const int my_row = static_cast<int>(my_row_k_s);
        if (my_row >= hidden) break;

        if (tid == 0) {
            const float in_f   = static_cast<float>(input_hidden[my_row]);
            const float moe_f  = workspace[OFF_MOE_OUT + my_row];
            const float shr_f  = workspace[OFF_SHARED_OUT + my_row];
            const float val    = bf16_round_rne_f32(in_f + moe_f + shr_f);
            output[my_row] = static_cast<T>(val);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// PR 4b5 step 2: INT4 dequant smoke kernel.
//
// Single-thread sweep over a small `[out_rows, in_cols]` slab that calls
// both `int4_dequant_8` (fused 8-wide path) and `int4_dequant_scalar`
// (single-element path) into separate output buffers. The host test
// compares each output byte-for-byte against a Rust-side reference
// implementing the same `bf16(q*s - z*s)` math. This catches porting
// bugs in the helpers in isolation, before they're folded into the FFN
// matmuls in step 3+.
//
// Pre-conditions (the bridge validates them):
//   - in_cols % 8 == 0
//   - in_cols % gsz == 0, gsz % 2 == 0 (single-byte index by `col / 2`)
//   - out_rows % gsz == 0 (single-row scale tile per scale_row)
__global__ void int4_dequant_smoke_kernel(
    const uint8_t*      packed,        // [out_rows, in_cols / 2]
    const hip_bfloat16* scale,         // [out_rows / gsz, in_cols / gsz]
    const hip_bfloat16* zero,          // [out_rows / gsz, in_cols / gsz]
    int                 out_rows,
    int                 in_cols,
    int                 gsz,
    float*              dq_8_out,      // [out_rows * in_cols] from int4_dequant_8
    float*              dq_scalar_out  // [out_rows * in_cols] from int4_dequant_scalar
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int byte_cols = in_cols / 2;
    const int scale_cols = in_cols / gsz;
    for (int row = 0; row < out_rows; ++row) {
        // Path A: 8-wide fused dequant.
        for (int col = 0; col < in_cols; col += 8) {
            uint32_t pk = 0;
            for (int b = 0; b < 4; ++b) {
                pk |= static_cast<uint32_t>(
                          packed[row * byte_cols + col / 2 + b]) << (8 * b);
            }
            float d[8];
            int4_dequant_8(pk, scale, zero,
                           row / gsz, col, scale_cols, gsz, d);
            for (int i = 0; i < 8; ++i) {
                dq_8_out[row * in_cols + col + i] = d[i];
            }
        }
        // Path B: scalar dequant (one element at a time).
        for (int col = 0; col < in_cols; ++col) {
            dq_scalar_out[row * in_cols + col] = int4_dequant_scalar(
                packed, scale, zero, row, col, in_cols, gsz);
        }
    }
}

// ---------------------------------------------------------------------------
// Final RMSNorm + lm_head GEMV in a single kernel — replaces the host-side
// path in `qwen36_moe_decode::host_final_norm_lm_head_f32`.
//
// On 35B-A3B the host F32 GEMV at [vocab=248320, hidden=2048] dominated the
// per-token wall-clock (233 ms of 360 ms in the local greedy benchmark; see
// PR #68's stage-timings output). Lifting it to the GPU eliminates the F32
// fan-out of the lm_head matrix (previously 970 MiB → 1.94 GiB at startup +
// per-token streaming through the host LLC) and the per-token CPU work.
//
// Math (matches `host_final_norm_lm_head_f32` byte-for-byte through the
// BF16 rounding boundary):
//
//   inv_rms     = rsqrt(mean(x_i^2) + eps)               # F32 reduction
//   x_normed[i] = bf16(x_i * inv_rms * (1.0 + final_norm_w[i]))   # `(1+w)`
//   logits[v]   = bf16(sum_i(lm_head_w[v, i] * x_normed[i]))
//
// LDS layout (matches the attn/linear/ffn step kernels):
//   lds[0,                 block_size):  shared_scratch (block reduction)
//   lds[block_size,        block_size + hidden):  x_norm_lds (F32 view of
//                                                  BF16-rounded normed hidden)
//   Total = (block_size + hidden) * 4 bytes = (256 + 2048) * 4 = 9 KiB on
//   35B-A3B. Same budget as the attn kernel; ≥2 WGs/CU on gfx1100.
//
// Block strategy: each block computes the RMSNorm into its own LDS
// (redundant across blocks — cheap, ~few µs for hidden=2048 — but lets the
// matvec read x_norm from LDS instead of global memory). Then a
// work-stealing loop assigns vocab rows to blocks via an atomic counter.
// Each block fully reduces one row before grabbing the next.
//
// `counter` is a `[1] u32` device buffer; the host must memset it to 0
// before each launch.
template <typename T>
__global__ void qwen36_moe_lm_head_kernel(
    const T*               __restrict__ final_hidden,    // [hidden] BF16
    const T*               __restrict__ final_norm_w,    // [hidden] BF16
    const T*               __restrict__ lm_head_w,       // [vocab, hidden] BF16
    T*                     __restrict__ logits,          // [vocab] BF16
    unsigned int*          __restrict__ counter,         // [1] u32
    int                                  hidden,
    int                                  vocab,
    float                                rms_norm_eps) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    extern __shared__ unsigned char lds_raw[];
    float* shared_scratch = reinterpret_cast<float*>(lds_raw);
    float* x_norm_lds     = shared_scratch + block_size;

    // -- Phase A: per-block RMSNorm of final_hidden into LDS ---------------
    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        const float v = load_as_float<T>(final_hidden, col);
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

    // Apply `(1.0 + w)` HF Qwen3_5MoeRMSNorm unit offset, BF16-round each
    // x_normed element so the matmul reads what PyTorch reads.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(final_norm_w, col);
        x_norm_lds[col] = bf16_round_rne_f32(x_norm_lds[col] * inv_rms * (1.0f + w));
    }
    __syncthreads();

    // -- Phase B: work-stealing matvec across vocab rows -------------------
    for (;;) {
        __shared__ unsigned int my_row_s;
        if (tid == 0) {
            my_row_s = atomicAdd(counter, 1u);
        }
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
            logits[my_row] = static_cast<T>(bf16_round_rne_f32(shared_scratch[0]));
        }
        __syncthreads();
    }
}

// =============================================================================
// RDNA3 WMMA BF16 lm_head GEMV (wave32, 16x16x16 f32-accumulate),
// one-wavefront-per-16-vocab-tile.
//
// Companion to `qwen36_moe_lm_head_kernel` above; same math, same numerics
// boundary, but routes the matvec through `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`
// for ~3-5x speedup over the scalar work-stealing path on hidden=2048,
// vocab=248k. Adapts the small-m WMMA pattern from
// `supersonic_qwen35_matmul_int4_dequant_wmma_small_m_kernel` (m=1 query,
// no INT4 dequant — lm_head weights are pre-dequantized BF16 by the engine
// at startup).
//
//   final_hidden  [hidden]         BF16
//   final_norm_w  [hidden]         BF16 (Qwen3_5MoeRMSNorm `(1.0 + w)`)
//   lm_head_w     [vocab, hidden]  BF16
//   logits        [vocab]          BF16
//
// Grid : (ceil(vocab / 16), 1, 1)
// Block: 32 threads = one wave32
//
// Each block computes 16 vocab logits. Per-block phases:
//   A: RMSNorm of final_hidden into LDS x_norm (BF16-rounded, F32→BF16 once).
//   B: K-loop, each iter loads 16 BF16 of x_norm into the WMMA A operand
//      (only lane_row==0 carries real data — m=1 means rows 1..15 of A are
//      zero-padded; the WMMA still does 16x16x16 muls but only row 0 of C
//      is read out, the rest is ignored).
//   C: lanes 0..15 (lane_half==0) write `acc[0] = C[0, lane_row]` to logits.
//
// Bandwidth: weights read once per element (~970 MiB BF16 for vocab=248k).
// At 960 GB/s on 7900 XTX, theoretical floor ~1 ms; observed end-to-end
// drops the lm_head stage from ~18.8 ms (scalar) to ~3-5 ms.
// =============================================================================

template <typename T>
__global__ __launch_bounds__(32, 4)
void qwen36_moe_lm_head_wmma_kernel(
    const T*               __restrict__ final_hidden,    // [hidden] BF16
    const T*               __restrict__ final_norm_w,    // [hidden] BF16
    const T*               __restrict__ lm_head_w,       // [vocab, hidden] BF16
    T*                     __restrict__ logits,          // [vocab] BF16
    int                                  hidden,
    int                                  vocab,
    float                                rms_norm_eps) {
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    const int lane = threadIdx.x;          // 0..31
    const int lane_row = lane & 15;        // 0..15 (replicates across lane halves)
    const int lane_half = lane >> 4;       // 0 for lanes 0-15, 1 for lanes 16-31
    const int tile_col = blockIdx.x * 16;  // first vocab row of this tile

    // LDS layout:
    //   shared_scratch [32]      F32  — RMSNorm wave reduction
    //   x_norm_lds_u16 [hidden]  u16  — BF16-rounded RMSNorm output
    extern __shared__ unsigned char lds_raw[];
    float*    shared_scratch = reinterpret_cast<float*>(lds_raw);
    uint16_t* x_norm_lds_u16 = reinterpret_cast<uint16_t*>(shared_scratch + 32);

    // -- Phase A: per-wave RMSNorm of final_hidden -------------------------
    // Each lane processes hidden/32 elements, accumulates F32 squares,
    // then we tree-reduce across the 32 lanes via LDS.
    float partial_sq = 0.0f;
    for (int col = lane; col < hidden; col += 32) {
        const float v = load_as_float<T>(final_hidden, col);
        partial_sq += v * v;
    }
    shared_scratch[lane] = partial_sq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) shared_scratch[lane] += shared_scratch[lane + s];
        __syncthreads();
    }
    const float inv_rms =
        rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);

    // Apply `(1.0 + w)` HF unit offset, BF16-round, store as BF16 bit pattern
    // (top 16 bits of the F32) so the WMMA A loads can read it as `short`.
    for (int col = lane; col < hidden; col += 32) {
        const float vh = load_as_float<T>(final_hidden, col);
        const float vw = load_as_float<T>(final_norm_w, col);
        const float val = bf16_round_rne_f32(vh * inv_rms * (1.0f + vw));
        uint32_t bits;
        __builtin_memcpy(&bits, &val, 4);
        x_norm_lds_u16[col] = static_cast<uint16_t>(bits >> 16);
    }
    __syncthreads();

    // -- Phase B: WMMA over (1 query) × hidden contraction × 16 vocab -----
    const int rhs_row_idx = tile_col + lane_row;
    const bool rhs_in_range = rhs_row_idx < vocab;

    const uint16_t* lm_head_w_u16 = reinterpret_cast<const uint16_t*>(lm_head_w);
    const size_t rhs_row_base = static_cast<size_t>(rhs_row_idx) * hidden;

    qwen36_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = hidden & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        // A operand: each lane_row maps to one of the 16 M-rows of A. With
        // m=1, only lane_row==0 carries real activation; lanes with
        // lane_row>0 zero-fill. The WMMA still fires 4096 muls but only
        // row 0 of C contributes useful output (it's a 1x16 GEMV result).
        // Bandwidth-bound on the B operand, so the wasted compute is free.
        qwen36_short16 a;
        if (lane_row == 0) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(x_norm_lds_u16[kk + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        // B operand: 16 vocab rows × 16 K elements. Each lane reads its
        // own vocab row's K-slab; lanes 0..15 and 16..31 duplicate
        // lane_row, so both halves load the same row (the WMMA hardware
        // handles the duplicate as one operand).
        qwen36_short16 b;
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(lm_head_w_u16[rhs_row_base + kk + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }

    // K tail (defensive — hidden=2048 is a multiple of 16 in production).
    if (k_main != hidden) {
        qwen36_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        qwen36_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (lane_row == 0) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < hidden) a[i] = static_cast<short>(x_norm_lds_u16[kc]);
            }
        }
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < hidden) {
                    b[i] = static_cast<short>(lm_head_w_u16[rhs_row_base + kc]);
                }
            }
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }

    // -- Phase C: write the M=0 row of C ----------------------------------
    // RDNA3 wave32 WMMA acc layout: acc[i] -> C[2*i + lane_half, lane_row].
    // Row 0 lives at acc[0] of lanes with lane_half==0 (lanes 0..15);
    // lane_row enumerates the 16 output columns of this tile.
    if (lane_half == 0) {
        const int out_col = tile_col + lane_row;
        if (out_col < vocab) {
            const float val = bf16_round_rne_f32(acc[0]);
            logits[out_col] = static_cast<T>(val);
        }
    }
#else
    (void)final_hidden; (void)final_norm_w; (void)lm_head_w;
    (void)logits; (void)hidden; (void)vocab; (void)rms_norm_eps;
#endif
}

// =============================================================================
// Qwen3.6-MoE multi-token-prediction (MTP) pre-fusion kernel — Phase 6.2c.1.
//
// Reproduces the vLLM `Qwen3NextMultiTokenPredictor.forward` "pre-fusion"
// stage byte-for-byte through the BF16 rounding boundary. This is the part
// of the MTP draft step that runs BEFORE the (single-layer) MoE block:
//
//   e_norm = rmsnorm(embed_tokens[next_token_id], pre_fc_norm_embedding, eps)
//   h_norm = rmsnorm(h_base, pre_fc_norm_hidden, eps)
//   fused  = mtp.fc @ cat([e_norm, h_norm], dim=-1)            # cat: e first
//
// Numerics match `oracle/qwen36_moe_mtp_oracle.py`: RMSNorm reduction in F32,
// `(1 + gain)` HF Qwen3_5MoeRMSNorm unit offset, BF16-round at every store.
// `mtp.fc.weight` is `[hidden, 2*hidden]` in HF row-major; first half of each
// row coefficients e_norm, second half h_norm (vLLM cat order).
//
// Inputs (all BF16, `ordinal`-resident):
//   e_in                     [hidden]            — host-extracted embedding row
//   h_base                   [hidden]            — base model's last hidden state
//   pre_fc_norm_embedding_w  [hidden]
//   pre_fc_norm_hidden_w     [hidden]
//   fc_w                     [hidden, 2*hidden]
//
// Outputs (all BF16, written by the kernel):
//   e_norm_out               [hidden]
//   h_norm_out               [hidden]
//   fused_out                [hidden]
//
// Block strategy: single block, block_size=256. Memory budget at hidden=2048:
//   shared_scratch [block_size]  F32 → 1 KiB
//   e_norm_lds     [hidden]      F32 → 8 KiB (BF16-rounded post-RMSNorm)
//   h_norm_lds     [hidden]      F32 → 8 KiB (BF16-rounded post-RMSNorm)
// Total: 17 KiB / 64 KiB CU cap. Comfortably under.
//
// Phase G (the matmul) is the cost driver: hidden output rows × 2*hidden
// reduction = ~8.4M FMAs. Striped across 256 threads → ~32K FMAs/thread.
// At hidden=2048 the kernel completes in ~30-50 µs; the speculative loop
// runs it once per draft step, so absolute cost is small. Multi-block
// work-stealing on the matmul phase is a future optimization (would need
// grid_barrier between Phases F and G).
//
// Used by `mtp_pre_fusion_launch` in `crates/kernel-ffi/src/qwen36_moe.rs`.
// =============================================================================
template <typename T>
__global__ void qwen36_moe_mtp_pre_fusion_kernel(
    int                    hidden,
    float                  rms_norm_eps,
    const T*  __restrict__ e_in,                      // [hidden]
    const T*  __restrict__ h_base,                    // [hidden]
    const T*  __restrict__ pre_fc_norm_embedding_w,   // [hidden]
    const T*  __restrict__ pre_fc_norm_hidden_w,      // [hidden]
    const T*  __restrict__ fc_w,                      // [hidden, 2*hidden]
    T*        __restrict__ e_norm_out,                // [hidden]
    T*        __restrict__ h_norm_out,                // [hidden]
    T*        __restrict__ fused_out                  // [hidden]
) {
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ unsigned char lds_raw[];
    float* shared_scratch = reinterpret_cast<float*>(lds_raw);
    float* e_norm_lds     = shared_scratch + block_size;
    float* h_norm_lds     = e_norm_lds + hidden;

    // -- Phase A: load e_in, accumulate partial squared sum ----------------
    float partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        const float v = load_as_float<T>(e_in, col);
        e_norm_lds[col] = v;
        partial_sq += v * v;
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float e_inv_rms;
    if (tid == 0) {
        e_inv_rms =
            rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // -- Phase B: apply `(1 + w)` RMSNorm to e, BF16-round, store ---------
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(pre_fc_norm_embedding_w, col);
        const float val = bf16_round_rne_f32(e_norm_lds[col] * e_inv_rms * (1.0f + w));
        e_norm_lds[col] = val;
        e_norm_out[col] = static_cast<T>(val);
    }
    __syncthreads();

    // -- Phase C: load h_base, accumulate partial squared sum --------------
    partial_sq = 0.0f;
    for (int col = tid; col < hidden; col += block_size) {
        const float v = load_as_float<T>(h_base, col);
        h_norm_lds[col] = v;
        partial_sq += v * v;
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
        __syncthreads();
    }
    __shared__ float h_inv_rms;
    if (tid == 0) {
        h_inv_rms =
            rsqrtf(shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);
    }
    __syncthreads();

    // -- Phase D: apply `(1 + w)` RMSNorm to h, BF16-round, store ---------
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(pre_fc_norm_hidden_w, col);
        const float val = bf16_round_rne_f32(h_norm_lds[col] * h_inv_rms * (1.0f + w));
        h_norm_lds[col] = val;
        h_norm_out[col] = static_cast<T>(val);
    }
    __syncthreads();

    // -- Phase E: matmul fc_w @ cat([e_norm, h_norm]) → fused -------------
    // fc_w[row, k]    for k in [0, hidden)        → e_norm[k] coefficients
    // fc_w[row, k]    for k in [hidden, 2*hidden) → h_norm[k - hidden] coefficients
    for (int row = tid; row < hidden; row += block_size) {
        const T* w_row = fc_w + static_cast<size_t>(row) * 2 * hidden;
        float acc = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            acc += load_as_float<T>(w_row, k) * e_norm_lds[k];
        }
        for (int k = 0; k < hidden; ++k) {
            acc += load_as_float<T>(w_row, hidden + k) * h_norm_lds[k];
        }
        fused_out[row] = static_cast<T>(bf16_round_rne_f32(acc));
    }
}

} // namespace qwen36_moe
