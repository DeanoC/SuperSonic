// Linear-attention phase body for the Qwen3.6-MoE persistent megakernel.
//
// Phase 3c refactor: the entire body of `qwen36_moe_linear_step_kernel`
// (Phases A-I — RMS-norm → in_proj_qkv/z/a/b → conv1d+silu → L2-norm Q/K
// → GQA fan-out + Q scale → beta+g → delta-rule recurrent update →
// per-V-head RMS norm + z-gate → out_proj+residual) is moved out of
// `kernels/qwen36_moe.hip` into this header as a `__device__ inline`
// function. Pure refactor; no behavior change. The original kernel
// becomes a thin wrapper for backward compatibility with the per-block
// parity tests at top_k=2 and top_k=8 (BF16 + INT4, stages 1-5).
//
// Why a separate header: the persistent megakernel (Phase 3e) calls this
// device function once per linear-attention layer in the descriptor
// walk. Keeping the body here avoids cyclic includes between the
// step-kernel translation unit and the persistent kernel.
//
// Workspace layout (F32 elements) — same as documented at the top of
// `qwen36_moe_linear_step_kernel`:
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
// LDS layout: lds[0, hidden) = x_norm_lds; lds[hidden, hidden+block_size)
// = shared_scratch (block reduction). Conv state and recurrent state
// live in dedicated state buffers (state is up to 2 MiB per layer at
// F32) — those are NOT in `workspace`.

#pragma once

#include "qwen36_moe_persistent/helpers.cuh"

namespace qwen36_moe {

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase B (the four
// in-projections — INT4 on qkv/z, BF16 on a/b) and Phase I (out_proj).
// When true:
//   - Phase B splits the unified work pool into four sub-loops with grid
//     barriers between them; qkv/z run WMMA tiles when their scales are
//     present, a/b stay scalar.
//   - Phase I runs the same WMMA tile pattern as the FFN's Phase I.
// The bridge picks the instantiation at launch time based on
// `device_supports_wmma_bf16(ordinal)` plus dim divisibility checks.
template <typename T, bool USE_WMMA = false>
__device__ inline void qwen36_moe_linear_step_device(
    int                            stage,
    int                            hidden,
    int                            num_k_heads,         // K
    int                            num_v_heads,         // V
    int                            head_k_dim,          // k_dim
    int                            head_v_dim,          // v_dim
    int                            conv_kernel_dim,
    float                          rms_norm_eps,
    const T* __restrict__          input_hidden,
    const T* __restrict__          input_norm_w,
    const T* __restrict__          in_proj_qkv_w,
    const T* __restrict__          in_proj_z_w,
    const T* __restrict__          in_proj_a_w,
    const T* __restrict__          in_proj_b_w,
    const T* __restrict__          conv1d_w,
    const T* __restrict__          conv1d_bias,
    const T* __restrict__          dt_bias,
    const T* __restrict__          a_log,
    const T* __restrict__          norm_w,
    const T* __restrict__          out_proj_w,
    T* __restrict__                conv_state,
    float* __restrict__            recurrent_state,
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

}  // namespace qwen36_moe
