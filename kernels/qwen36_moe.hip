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
    }

    // Stages 2..5 hook in here in follow-up commits.
}

} // namespace qwen36_moe
