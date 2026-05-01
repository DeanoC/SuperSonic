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

template <typename T>
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
    const T* __restrict__          input_hidden,  // [hidden]
    const T* __restrict__          input_norm_w,  // [hidden]
    const T* __restrict__          q_proj_w,      // [2*H*d, hidden]
    const T* __restrict__          k_proj_w,      // [Hkv*d, hidden]   (nullable for stage<2)
    const T* __restrict__          v_proj_w,      // [Hkv*d, hidden]   (nullable for stage<2)
    const T* __restrict__          q_norm_w,      // [d]
    const T* __restrict__          k_norm_w,      // [d]               (nullable for stage<2)
    const T* __restrict__          o_proj_w,      // [hidden, H*d]     (nullable for stage<5)
    T* __restrict__                output,        // BF16, sized for the largest staged intermediate
    float* __restrict__             workspace,    // F32 scratch, see header comment
    unsigned int* __restrict__     counters,
    unsigned int* __restrict__     barrier_counter,
    unsigned int* __restrict__     barrier_flag) {
    (void)k_proj_w;
    (void)v_proj_w;
    (void)k_norm_w;
    (void)o_proj_w;
    (void)rotary_dim;
    (void)rope_theta;
    (void)position;

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
    // (its `out.to(in_dtype)` step in rms_norm).
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(input_norm_w, col);
        x_norm_lds[col] = bf16_round_rne_f32(x_norm_lds[col] * inv_rms_input * w);
    }
    __syncthreads();

    // -- Phase B: q_raw = q_proj_w @ x_norm  (work-stealing matvec) --------
    // Output rows of q_raw are written BF16-rounded into workspace[0..2*H*d]
    // so that the per-head RMS norm in Phase C reads the same F32-of-BF16
    // values PyTorch's BF16 q_raw represents.
    const int q_out_dim = 2 * num_heads * head_dim;

    for (;;) {
        __shared__ unsigned int my_row_s;
        if (tid == 0) {
            my_row_s = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const int my_row = static_cast<int>(my_row_s);
        if (my_row >= q_out_dim) break;

        const T* w_row = q_proj_w + static_cast<size_t>(my_row) * hidden;
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
            workspace[my_row] = bf16_round_rne_f32(shared_scratch[0]);
        }
        __syncthreads();
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

    // -- Phase C: per-head RMS norm of q_lanes = workspace[Q_RAW : Q_RAW+H*d]
    // Each block grabs a head index, reduces across head_dim, writes BF16
    // q_normed to workspace[Q_NORMED..]. If stage == 1 we also publish
    // q_normed to the host-visible `output` buffer; later stages keep
    // q_normed in workspace as RoPE input.
    {
        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int h = static_cast<int>(head_s);
            if (h >= H) break;

            const int base = h * d;
            float partial = 0.0f;
            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_Q_RAW + base + i];
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
                const float v = workspace[OFF_Q_RAW + base + i];
                const float wv = load_as_float<T>(q_norm_w, i);
                const float normed = bf16_round_rne_f32(v * head_inv_rms * wv);
                // Round-trip through BF16 representation: store BF16 in
                // workspace as F32 so the next-stage RoPE reader sees the
                // exact dtype-cast values PyTorch's `.to(bf16)` produces.
                workspace[OFF_Q_NORMED + base + i] = normed;
                if (stage == 1) {
                    output[base + i] = static_cast<T>(normed);
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
    // Combine k_proj_w and v_proj_w into a single 2*Hkv*d-row work pool so
    // every block stays busy regardless of which projection it claims. Both
    // projections share the same x_norm input, so this is purely a row
    // dispatch fold — same pattern PyTorch uses internally for fused QKV.
    const int kv_total_rows = 2 * Hkv * d;
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
        const T* w_row = (is_v ? v_proj_w : k_proj_w)
            + static_cast<size_t>(proj_row) * hidden;
        const int dst_off = (is_v ? OFF_V_RAW : OFF_K_RAW) + proj_row;

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
            workspace[dst_off] = bf16_round_rne_f32(shared_scratch[0]);
        }
        __syncthreads();
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
                const float normed = bf16_round_rne_f32(v * head_inv_rms * wv);
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

    // -- Phase G: GQA self-attention, kv_len = 1 ----------------------------
    //
    // The PR-4b2 oracle runs without a KV cache, so the only K/V token is
    // the just-emitted one. With a single-token K, softmax over the score
    // vector reduces to a single element ≡ 1.0 regardless of the score.
    // The output therefore collapses to:
    //
    //   attn[h, :] = bf16(1.0) * v_full[h, :]
    //              = v_raw[h // rep, :]      (already BF16-rounded F32)
    //
    // We still compute the QK score (skipped for the output) so the path
    // exercises q_rot / k_rot loads — that's the slice of compute the
    // future kv_len > 1 extension will reuse, with the softmax becoming
    // non-trivial. For now the score is intentionally unused.
    //
    // Each block grabs one Q head, broadcasts the matching KV head's V row
    // into workspace[ATTN + h*d : ...] and (when stage == 4) into output.
    {
        const int rep   = H / Hkv;
        const float scale = rsqrtf(static_cast<float>(d));

        for (;;) {
            __shared__ unsigned int head_s;
            if (tid == 0) {
                head_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int hq = static_cast<int>(head_s);
            if (hq >= H) break;

            const int h_kv = hq / rep;

            // Score (q · k) * scale — computed for path-coverage only at
            // kv_len=1. Single-block reduction over head_dim=256.
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

            // softmax([s]) = [1.0]; bf16(1.0 * v) = v. v_raw is already
            // BF16-rounded F32 from Phase D so no extra rounding needed.
            for (int i = tid; i < d; i += block_size) {
                const float v = workspace[OFF_V_RAW + h_kv * d + i];
                workspace[OFF_ATTN + hq * d + i] = v;
                if (stage == 4) {
                    output[hq * d + i] = static_cast<T>(v);
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
    // out_gate is the upper half of q_raw stored at workspace[Q_RAW + H*d:].
    // Each block grabs one Q head and produces gated_attn[h, :] in workspace.
    // Oracle math (qwen36_moe_oracle.py line 215):
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
                const float out_gate = workspace[OFF_Q_RAW + H * d + h * d + i];
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

        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) {
                my_row_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= hidden) break;

            const T* w_row = o_proj_w + static_cast<size_t>(my_row) * qd;
            float partial = 0.0f;
            for (int j = tid; j < qd; j += block_size) {
                partial += load_as_float<T>(w_row, j) * workspace[OFF_GATED + j];
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

template <typename T>
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

    for (int col = tid; col < hidden; col += block_size) {
        const float w = static_cast<float>(input_norm_w[col]);
        x_norm_lds[col] = bf16_round_rne_f32(x_norm_lds[col] * inv_rms_input * w);
    }
    __syncthreads();

    // -- Phase B: four in-projections in one fused work pool ---------------
    //
    // qkv_raw, z_raw, a_raw, b_raw all read the same x_norm and have
    // independent output rows, so we union them into one row-stealing pool
    // of (qkv_dim + V*v_dim + 2*V) rows. Each block claims the next row,
    // figures out which projection it belongs to, computes the dot
    // product, and writes BF16-rounded F32 to the workspace at the right
    // offset.
    //
    // Row-id range layout (in the order matching the workspace offsets):
    //   [0,                       qkv_dim):                qkv (in_proj_qkv)
    //   [qkv_dim,                 qkv_dim + V*v_dim):      z   (in_proj_z)
    //   [qkv_dim + V*v_dim,       qkv_dim + V*v_dim + V):  a   (in_proj_a)
    //   [qkv_dim + V*v_dim + V,   qkv_dim + V*v_dim + 2V): b   (in_proj_b)
    const int total_rows = qkv_dim + val_dim + 2 * V;

    for (;;) {
        __shared__ unsigned int my_row_s;
        if (tid == 0) {
            my_row_s = atomicAdd(&counters[0], 1u);
        }
        __syncthreads();
        const int my_row = static_cast<int>(my_row_s);
        if (my_row >= total_rows) break;

        // Pick projection + per-projection row-index + workspace dst.
        const T* w_row;
        int dst_off;
        if (my_row < qkv_dim) {
            w_row   = in_proj_qkv_w + static_cast<size_t>(my_row) * hidden;
            dst_off = OFF_QKV_RAW + my_row;
        } else if (my_row < qkv_dim + val_dim) {
            const int r = my_row - qkv_dim;
            w_row   = in_proj_z_w + static_cast<size_t>(r) * hidden;
            dst_off = OFF_Z_RAW + r;
        } else if (my_row < qkv_dim + val_dim + V) {
            const int r = my_row - (qkv_dim + val_dim);
            w_row   = in_proj_a_w + static_cast<size_t>(r) * hidden;
            dst_off = OFF_A_RAW + r;
        } else {
            const int r = my_row - (qkv_dim + val_dim + V);
            w_row   = in_proj_b_w + static_cast<size_t>(r) * hidden;
            dst_off = OFF_B_RAW + r;
        }

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

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

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

    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);

    const int OFF_BETA    = OFF_K_REP + V * k_dim;
    const int OFF_G       = OFF_BETA + V;
    const int OFF_REC_OUT = OFF_G + V;

    // -- Phase F: beta + g per V-head ---------------------------------------
    // beta[h] = sigmoid(b_raw[h])                                  (F32)
    // g[h]    = -softplus(a_raw[h] + dt_bias[h]) * exp(A_log[h])   (F32)
    //
    // V is small (32 on 35B-A3B) so we do this on block 0 only — every
    // other block parks at the next grid barrier.
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
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

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

        for (;;) {
            __shared__ unsigned int my_row_s;
            if (tid == 0) {
                my_row_s = atomicAdd(&counters[0], 1u);
            }
            __syncthreads();
            const int my_row = static_cast<int>(my_row_s);
            if (my_row >= hidden) break;

            const T* w_row = out_proj_w + static_cast<size_t>(my_row) * qd;
            float partial = 0.0f;
            for (int j = tid; j < qd; j += block_size) {
                partial += static_cast<float>(w_row[j]) * workspace[OFF_REC_OUT + j];
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
//     EXPERT_GU     = ... + hidden                                   [2*I] gate_up_proj output for one expert (F32)
//     EXPERT_MID    = ... + 2*I                                      [I]   silu(gu[:I]) * gu[I:]  (F32)
//     EXPERT_STACK  = ... + I                                        [k * hidden] per-expert outs (F32, no BF16 round)
//     MOE_OUT       = ... + k*hidden                                 [hidden] BF16-rounded F32
//     OUTPUT_HIDDEN = ... + hidden                                   [hidden]
//   For 35B-A3B (hidden=2048, E=256, k=8, Is=512, I=512) stage-5 footprint
//   is ~28,177 F32 = ~110 KiB. Well within the persistent-decode budget.
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

template <typename T>
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
    const int OFF_EXPERT_MID    = OFF_EXPERT_GU + 2 * moe_intermediate;
    const int OFF_EXPERT_STACK  = OFF_EXPERT_MID + moe_intermediate;
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
    // Block 0 is also responsible for staging the F32 of-BF16 view into
    // workspace[OFF_H_NORM] for any later stages that prefer global memory.
    for (int col = tid; col < hidden; col += block_size) {
        const float w = static_cast<float>(post_attn_norm_w[col]);
        const float v = bf16_round_rne_f32(h_norm_lds[col] * inv_rms_input * w);
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
            // INT4 scalar dequant per element. Stride pattern matches the
            // BF16 path (thread t reads cols t, t+block_size, …) so the
            // reduction tree is unchanged.
            for (int col = tid; col < hidden; col += block_size) {
                partial += int4_dequant_scalar(
                               i4_slab, i4_scale, i4_zero,
                               i4_row, col, hidden, int4_group_size)
                           * h_norm_lds[col];
            }
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
                const void* i4_slab =
                    reinterpret_cast<const void*>(shared_down_proj_w);
                for (int col = tid; col < Is_; col += block_size) {
                    partial += int4_dequant_scalar(
                                   i4_slab,
                                   shared_down_proj_scale,
                                   shared_down_proj_zero,
                                   my_row, col, Is_, int4_group_size)
                               * workspace[OFF_SHARED_MID + col];
                }
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

    // -- Phase G/H/I: per-expert FFN — k-iteration loop  -------------------
    // Stage 3 dispatches just topk_idx[0]; stage 4+ sweeps all k experts.
    // The per-iteration body is the same; the difference is what we do
    // after the loop:
    //   stage 3 → expose EXPERT_STACK[0] BF16-cast as the parity output
    //   stage 4 → sum topk_w[j] * EXPERT_STACK[j] across j, BF16-round once
    //
    // Within each iteration:
    //   gu  = gate_up_proj_w[e] @ h_norm                            [2*I]
    //   mid = silu(gu[:I]) * gu[I:]                                 [I]
    //   eo  = down_proj_w[e].T @ mid                                [hidden]
    // EXPERT_STACK keeps `eo` raw F32 so stage 4 doesn't lose precision
    // through a BF16 round-trip in the weighted sum.

    const int I_ = moe_intermediate;
    const int two_I = 2 * I_;
    const int k_iters = (stage == 3) ? 1 : top_k;

    for (int j = 0; j < k_iters; j++) {
        // Per-iteration: every block reads the chosen expert index for
        // this slot from workspace. Visible across blocks because the
        // last grid barrier (Phase F's reset, plus any from the
        // previous iteration's Phase I) ordered the writes.
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);

        __shared__ int s_expert_idx;
        if (tid == 0) {
            s_expert_idx = __float_as_int(workspace[OFF_TOPK_IDX + j]);
        }
        __syncthreads();
        const int e = s_expert_idx;

        // Phase G: matvec gate_up_proj_w[e] @ h_norm → EXPERT_GU[2*I].
        // BF16 layout: gate_up_proj_w[E, 2*I, hidden] — slab `e` base is
        // `e * 2*I * hidden`; row `r` adds `r * hidden`.
        // INT4 layout (when `gate_up_proj_scale != nullptr`):
        //   weights : [E, 2*I, hidden/2] u8   — slab `e` base in bytes is
        //                                        `e * 2*I * (hidden/2)`.
        //   scale   : [E, 2*I/gsz, hidden/gsz] BF16 — slab `e` base in
        //                                        BF16 elems is
        //                                        `e * (2*I/gsz) * (hidden/gsz)`.
        // Same dequant + reduction discipline as the shared-expert INT4
        // path (step 3); only the per-expert slab base offsets differ.
        {
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
            for (;;) {
                __shared__ unsigned int my_row_g_s;
                if (tid == 0) {
                    my_row_g_s = atomicAdd(&counters[0], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_g_s);
                if (my_row >= two_I) break;

                float partial = 0.0f;
                if (gu_int4) {
                    for (int col = tid; col < hidden; col += block_size) {
                        partial += int4_dequant_scalar(
                                       gu_slab_packed,
                                       gu_slab_scale,
                                       gu_slab_zero,
                                       my_row, col, hidden, int4_group_size)
                                   * h_norm_lds[col];
                    }
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
                    workspace[OFF_EXPERT_GU + my_row] = shared_scratch[0];
                }
                __syncthreads();
            }
        }

        // Phase H: smid = silu(gu[:I]) * gu[I:] (block-0 only, F32 throughout).
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);
        if (blockIdx.x == 0) {
            for (int i = tid; i < I_; i += block_size) {
                const float gp = workspace[OFF_EXPERT_GU + i];
                const float up = workspace[OFF_EXPERT_GU + I_ + i];
                const float sigmoid_gp = 1.0f / (1.0f + expf(-gp));
                const float silu_gp    = gp * sigmoid_gp;
                workspace[OFF_EXPERT_MID + i] = silu_gp * up;
            }
            __syncthreads();
        }

        // Phase I: matvec down_proj_w[e].T @ mid → EXPERT_STACK[j*hidden..].
        // BF16 layout: down_proj_w[E, hidden, I]; slab `e` base is
        // `e * hidden * I`, row `r` adds `r * I`. Contracts over I → one
        // F32 per output row.
        // INT4 layout (when `down_proj_scale != nullptr`):
        //   weights : [E, hidden, I/2] u8   — slab `e` base in bytes is
        //                                      `e * hidden * (I/2)`.
        //   scale   : [E, hidden/gsz, I/gsz] BF16 — slab `e` base in BF16
        //                                      elems is
        //                                      `e * (hidden/gsz) * (I/gsz)`.
        grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                   &counters[0]);
        {
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
            const int slot_off = OFF_EXPERT_STACK + j * hidden;
            for (;;) {
                __shared__ unsigned int my_row_i_s;
                if (tid == 0) {
                    my_row_i_s = atomicAdd(&counters[0], 1u);
                }
                __syncthreads();
                const int my_row = static_cast<int>(my_row_i_s);
                if (my_row >= hidden) break;

                float partial = 0.0f;
                if (dp_int4) {
                    for (int col = tid; col < I_; col += block_size) {
                        partial += int4_dequant_scalar(
                                       dp_slab_packed,
                                       dp_slab_scale,
                                       dp_slab_zero,
                                       my_row, col, I_, int4_group_size)
                                   * workspace[OFF_EXPERT_MID + col];
                    }
                } else {
                    const T* w_row =
                        dp_slab_bf16 + static_cast<size_t>(my_row) * I_;
                    for (int col = tid; col < I_; col += block_size) {
                        partial += static_cast<float>(w_row[col])
                                   * workspace[OFF_EXPERT_MID + col];
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
                    if (stage == 3 && j == 0) {
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
        const int K_ = top_k;
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

} // namespace qwen36_moe
