// Full-attention phase body for the Qwen3.6-MoE persistent megakernel.
//
// Phase 3b refactor: the entire body of `qwen36_moe_attn_step_kernel`
// (Phases A through I — RMS-norm → q_proj → q-norm → k/v_proj → k-norm
// → RoPE → GQA self-attention → out-gate → o_proj+residual) is moved
// out of `kernels/qwen36_moe.hip` into this header as a `__device__
// inline` function. Pure refactor; no behavior change. The original
// kernel becomes a thin wrapper for backward compatibility with the
// per-block parity tests at top_k=2 and top_k=8 (BF16 + INT4, stages
// 1-5).
//
// Why a separate header: the persistent megakernel (Phase 3e) calls
// this device function once per full-attention layer in the descriptor
// walk. Keeping the body here avoids cyclic includes between the
// step-kernel translation unit and the persistent kernel.
//
// Workspace layout (F32 elements) — same as documented at the top of
// `qwen36_moe_attn_step_kernel`:
//   Q_RAW    = 0                                [2*H*d]
//   K_RAW    = 2*H*d                            [Hkv*d]
//   V_RAW    = 2*H*d + Hkv*d                    [Hkv*d]
//   Q_NORMED = 2*H*d + 2*Hkv*d                  [H*d]
//   K_NORMED = 2*H*d + 2*Hkv*d + H*d            [Hkv*d]
//   Q_ROT    = 2*H*d + 2*Hkv*d + H*d + Hkv*d    [H*d]
//   K_ROT    = 3*H*d + 2*Hkv*d + H*d + Hkv*d    [Hkv*d]
//   ATTN     = 4*H*d + 4*Hkv*d                  [H*d]
//   GATED    = 5*H*d + 4*Hkv*d                  [H*d]
//   O_OUT    = 6*H*d + 4*Hkv*d                  [hidden]
//   SCORES   = 6*H*d + 4*Hkv*d + hidden         [H * kv_max_t]
//
// LDS layout: lds[0, hidden) = x_norm_lds; lds[hidden, hidden+block_size)
// = shared_scratch (block reduction). Total = (hidden + block_size) * 4
// bytes = 9 KiB at 35B-A3B (hidden=2048, block_size=256).

#pragma once

#include "qwen36_moe_persistent/helpers.cuh"

namespace qwen36_moe {

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase B (q_proj GEMV),
// Phase D (fused k_proj/v_proj GEMV — split into two WMMA sub-pools),
// and Phase I (o_proj GEMV + residual). Bridge picks the instantiation at
// launch time based on `device_supports_wmma_bf16(ordinal)` + dim
// divisibility.
template <typename T, bool USE_WMMA = false>
__device__ inline void qwen36_moe_attn_step_device(
    int                            stage,
    int                            hidden,
    int                            num_heads,
    int                            num_kv_heads,
    int                            head_dim,
    int                            rotary_dim,
    float                          rope_theta,
    float                          rms_norm_eps,
    int                            position,
    // Optional override for the KV cache slot index. Negative ⇒ "same as
    // `position`" (the base-model decode case where RoPE position == cache
    // slot). Set to a non-negative value to decouple the two — used by
    // self-speculative MTP, where RoPE rotates at absolute position
    // `base_seq_len + k` but the per-MTP-session cache writes at slot `k`
    // (the cache is fresh per draft session). When `cache_pos >= 0`,
    // Phase G writes K/V at `cache_pos` and attends over
    // `kv_len = cache_pos + 1`; RoPE still uses `position`.
    int                            cache_pos,
    const T* __restrict__          input_hidden,
    const T* __restrict__          input_norm_w,
    const T* __restrict__          q_proj_w,
    const T* __restrict__          k_proj_w,
    const T* __restrict__          v_proj_w,
    const T* __restrict__          q_norm_w,
    const T* __restrict__          k_norm_w,
    const T* __restrict__          o_proj_w,
    // PR 4b6 INT4 sidecars. `int4_group_size == 0` ⇒ all weights are BF16
    // and every sidecar pointer must be null. When non-zero, each sidecar
    // pair is consulted independently per tensor: a null `*_scale` keeps
    // that tensor on the BF16 path; a non-null pair switches to INT4.
    int                                  int4_group_size,
    const hip_bfloat16* __restrict__     q_proj_scale,
    const hip_bfloat16* __restrict__     q_proj_zero,
    const hip_bfloat16* __restrict__     k_proj_scale,
    const hip_bfloat16* __restrict__     k_proj_zero,
    const hip_bfloat16* __restrict__     v_proj_scale,
    const hip_bfloat16* __restrict__     v_proj_zero,
    const hip_bfloat16* __restrict__     o_proj_scale,
    const hip_bfloat16* __restrict__     o_proj_zero,
    T* __restrict__                output,
    float* __restrict__             workspace,
    // PR 4d KV cache extension. When non-null, Phase G writes the current
    // step's K/V at slot `eff_cache_pos` and attends over kv_len past
    // tokens. When null, falls back to the kv_len=1 self-attention path.
    T* __restrict__                kv_cache_k,
    T* __restrict__                kv_cache_v,
    int                            kv_max_t,
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
    const bool fp8_mode = int4_group_size < 0;
    const int quant_group_size = fp8_mode ? -int4_group_size : int4_group_size;
    const bool q_quant = (q_proj_scale != nullptr);
    const bool q_int4 = q_quant && !fp8_mode;

    bool wmma_handled_qproj = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    if constexpr (USE_WMMA) {
        if (q_int4) {
            // WMMA tile path — 128 rows per atomicAdd, 8 waves × 16-row
            // tiles. Same `wmma_int4_matvec_partial_16rows` helper as the
            // FFN/linear-attn WMMA paths; output is BF16-rounded to match
            // the scalar path's `workspace[my_row] = bf16_round_rne_f32(...)`.
            const int gsc_q = hidden / quant_group_size;
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
                    gsc_q, quant_group_size, lane_row);
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
                    my_row, hidden, quant_group_size,
                    tid, block_size);
            } else if (q_quant) {
                partial = fp8_matvec_partial(
                    reinterpret_cast<const void*>(q_proj_w),
                    static_cast<const hip_bfloat16*>(q_proj_scale),
                    x_norm_lds,
                    my_row, hidden, quant_group_size,
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
        const int gsc_kv = hidden / quant_group_size;
        const int wave_id = tid >> 5;
        const int lane_in_wave = tid & 31;
        const int lane_row = lane_in_wave & 15;
        const int lane_half = lane_in_wave >> 4;

        // ------- Sub-pool 1: k_proj -------
        if (k_proj_scale != nullptr && !fp8_mode) {
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
                    gsc_kv, quant_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_K_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16/FP8 fallback for k_proj.
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= Hkv * d) break;
                float partial = 0.0f;
                if (k_proj_scale != nullptr) {
                    partial = fp8_matvec_partial(
                        reinterpret_cast<const void*>(k_proj_w),
                        static_cast<const hip_bfloat16*>(k_proj_scale),
                        x_norm_lds,
                        my_row, hidden, quant_group_size,
                        tid, block_size);
                } else {
                    const T* w_row =
                        k_proj_w + static_cast<size_t>(my_row) * hidden;
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
        if (v_proj_scale != nullptr && !fp8_mode) {
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
                    gsc_kv, quant_group_size, lane_row);
                if (lane_half == 0 && in_range) {
                    workspace[OFF_V_RAW + rhs_row] =
                        bf16_round_rne_f32(acc[0]);
                }
                __syncthreads();
            }
        } else {
            // BF16/FP8 fallback for v_proj.
            for (;;) {
                __shared__ unsigned int my_row_s;
                if (tid == 0) my_row_s = atomicAdd(&counters[0], 1u);
                __syncthreads();
                const int my_row = static_cast<int>(my_row_s);
                if (my_row >= Hkv * d) break;
                float partial = 0.0f;
                if (v_proj_scale != nullptr) {
                    partial = fp8_matvec_partial(
                        reinterpret_cast<const void*>(v_proj_w),
                        static_cast<const hip_bfloat16*>(v_proj_scale),
                        x_norm_lds,
                        my_row, hidden, quant_group_size,
                        tid, block_size);
                } else {
                    const T* w_row =
                        v_proj_w + static_cast<size_t>(my_row) * hidden;
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
            if (i4_scale != nullptr && !fp8_mode) {
                partial = int4_dq8_matvec_partial(
                    reinterpret_cast<const uint8_t*>(w_slab),
                    i4_scale, i4_zero, x_norm_lds,
                    proj_row, hidden, quant_group_size,
                    tid, block_size);
            } else if (i4_scale != nullptr) {
                partial = fp8_matvec_partial(
                    reinterpret_cast<const void*>(w_slab),
                    i4_scale, x_norm_lds,
                    proj_row, hidden, quant_group_size,
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
        const bool o_quant = (o_proj_scale != nullptr);
        const bool o_int4 = o_quant && !fp8_mode;

        bool wmma_handled_oproj = false;
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
        if constexpr (USE_WMMA) {
            if (o_int4) {
                const int gsc_o = qd / quant_group_size;
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
                        gsc_o, quant_group_size, lane_row);
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
                        my_row, qd, quant_group_size,
                        tid, block_size);
                } else if (o_quant) {
                    partial = fp8_matvec_partial(
                        reinterpret_cast<const void*>(o_proj_w),
                        static_cast<const hip_bfloat16*>(o_proj_scale),
                        workspace + OFF_GATED,
                        my_row, qd, quant_group_size,
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

}  // namespace qwen36_moe
