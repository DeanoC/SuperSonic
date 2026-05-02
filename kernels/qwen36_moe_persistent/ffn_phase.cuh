// MoE FFN phase body for the Qwen3.6-MoE persistent megakernel.
//
// Phase 3d refactor: the entire body of `qwen36_moe_ffn_step_kernel`
// (Phases A-K — post-attn RMS-norm → router gate → softmax+topK+renorm
// → shared-expert in-projections → silu*mul → shared down_proj → all
// top_k routed experts in parallel (gate_up_proj, silu*mul, down_proj)
// → topK-weighted sum → final residual add) is moved out of
// `kernels/qwen36_moe.hip` into this header as a `__device__ inline`
// function. Pure refactor; no behavior change. The original kernel
// becomes a 24-line forwarding wrapper for the per-block parity tests.
//
// Why a separate header: the persistent megakernel (Phase 3e) calls
// this device function once per layer in the descriptor walk. Keeping
// the body here avoids cyclic includes between the step-kernel
// translation unit and the persistent kernel.
//
// Workspace layout (F32 elements) — same as documented inline at the
// top of `qwen36_moe_ffn_step_kernel`:
//   H_NORM        = 0                                                [hidden]
//   ROUTER_LOGITS = hidden                                           [E]
//   ROUTER_PROBS  = hidden + E                                       [E]
//   TOPK_VAL      = hidden + 2E                                      [k]
//   TOPK_IDX      = hidden + 2E + k                                  [k]
//   SG_SCALAR     = hidden + 2E + 2k                                 [1]
//   SGP           = SG_SCALAR + 1                                    [Is]
//   SUP           = SGP + Is                                         [Is]
//   SHARED_MID    = SUP + Is                                         [Is]
//   SHARED_OUT    = SHARED_MID + Is                                  [hidden]
//   EXPERT_GU     = SHARED_OUT + hidden                              [k * 2I]
//   EXPERT_MID    = EXPERT_GU + k*2I                                 [k * I]
//   EXPERT_STACK  = EXPERT_MID + k*I                                 [k * hidden]
//   MOE_OUT       = EXPERT_STACK + k*hidden                          [hidden]
//
// Counter slots (zeroed by `reset_sync_buf` on the host before each
// launch — see `crates/kernel-ffi/src/qwen36_moe.rs`):
//   counters[g]            — Phase G work-stealing for group g (0..2I)
//   counters[top_k + g]    — Phase I work-stealing for group g (0..hidden)
//   counters[0]            — re-used by Phase J / K (reset before each)
//
// LDS layout: lds[0, hidden) = h_norm_lds; lds[hidden, hidden+block_size)
// = shared_scratch (block reduction). 9 KiB at hidden=2048,
// block_size=256 — comfortably under gfx1100's 64 KiB LDS/WG cap.

#pragma once

#include "qwen36_moe_persistent/helpers.cuh"

namespace qwen36_moe {

// `USE_WMMA` (defaults to false for backward-compat with the per-block
// parity tests) gates the RDNA3 WMMA INT4 path on Phase G (gate_up_proj
// per-expert GEMV) and Phase I (down_proj per-expert GEMV). When true:
//   - Phase G/I split each block's work into 8 waves × 16-row WMMA
//     tiles (128 rows per atomicAdd) running
//     `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` against
//     INT4-dequant-on-the-fly weight tiles. Stays bit-exact with the
//     scalar `int4_dq8_matvec_partial` path up to F32 accumulation order.
//   - Phase D/F (shared expert) stays on the scalar path in both
//     variants — it's ~1/8 of the per-token matmul cost.
// The bridge picks the instantiation at launch time based on
// `device_supports_wmma_bf16(ordinal)` plus dim divisibility checks.
template <typename T, bool USE_WMMA = false>
__device__ inline void qwen36_moe_ffn_step_device(
    int                            stage,
    int                            hidden,
    int                            num_experts,                  // E (256 on 35B-A3B)
    int                            moe_intermediate,             // I (512 on 35B-A3B)
    int                            shared_intermediate,          // Is (512 on 35B-A3B)
    int                            top_k,                        // k (8 on 35B-A3B)
    float                          rms_norm_eps,
    const T* __restrict__          input_hidden,
    const T* __restrict__          post_attn_norm_w,
    const T* __restrict__          gate_w,
    const T* __restrict__          gate_up_proj_w,
    const T* __restrict__          down_proj_w,
    const T* __restrict__          shared_gate_proj_w,
    const T* __restrict__          shared_up_proj_w,
    const T* __restrict__          shared_down_proj_w,
    const T* __restrict__          shared_expert_gate_w,
    // PR 4b5 INT4 sidecars. `int4_group_size == 0` ⇒ all weights are BF16
    // and every sidecar pointer must be null.
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
    T* __restrict__                output,
    int* __restrict__              output_idx,
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

        // Step 1: max reduction.
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
        __shared__ int   idx_buf[256];
        __shared__ float val_buf[256];
        for (int kk = 0; kk < K; kk++) {
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
                workspace[OFF_ROUTER_PROBS + winner_idx] = -INFINITY;
            }
            __syncthreads();
        }

        // Step 5: renormalise the top-k probs to sum to 1.
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

        const T* w_row = nullptr;
        const void* i4_slab = nullptr;
        const hip_bfloat16* i4_scale = nullptr;
        const hip_bfloat16* i4_zero  = nullptr;
        int      i4_row = 0;
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
                val = 1.0f / (1.0f + expf(-val));
            }
            workspace[write_offset] = val;
        }
        __syncthreads();
    }

    // -- Phase E: smid = silu(sgp) * sup  (block-0 only, F32 throughout) ---
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    if (blockIdx.x == 0) {
        for (int i = tid; i < Is_; i += block_size) {
            const float gp     = workspace[OFF_SGP + i];
            const float up     = workspace[OFF_SUP + i];
            const float sigmoid_gp = 1.0f / (1.0f + expf(-gp));
            const float silu_gp    = gp * sigmoid_gp;
            workspace[OFF_SHARED_MID + i] = silu_gp * up;
        }
        __syncthreads();
    }

    // -- Phase F: shared_out = sg_scalar * (smid @ down_proj.T) ------------
    grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                               &counters[0]);
    {
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
    // Stage 3 dispatches only group 0 by setting `active_groups = 1`. All
    // blocks still hit the unconditional barriers (the grid barrier needs
    // every block to participate or it hangs); only the work bodies are
    // gated on `group_active`. Stage 3's parity-output write fires when
    // `group_id == 0` (mirrors the old `j == 0` condition).

    const int I_ = moe_intermediate;
    const int two_I = 2 * I_;
    const int K_ = top_k;
    const int active_groups = (stage == 3) ? 1 : K_;
    const int group_id = blockIdx.x % K_;
    const bool group_active = (group_id < active_groups);

    // Phase F left counters[0] at `hidden` (atomicAdds during work-stealing).
    // Phase G of group 0 uses that same slot, so reset it here before any
    // group reads its counter.
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

        // WMMA path: 128 rows per atomicAdd; 8 waves do 16-row WMMA tiles
        // in parallel. INT4 weights only; BF16 weights fall through to
        // the scalar path below.
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
    grid_barrier(barrier_counter, barrier_flag, num_blocks);

    // Phase H: mid = silu(gu[:I]) * gu[I:], one block per active group.
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
    // Single weighted reduction across `k` experts. F32 accumulation,
    // BF16-round once at the store. Matches
    // `(topk_w_renorm.unsqueeze(-1) * expert_stack).sum(0).to(dtype)`.
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

            // Sequential per-row sum on a single thread (k=8).
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
    // F32 sum with one BF16 round at the store, matching the oracle's
    // `(input_hidden.f32 + moe_out.f32 + shared_out.f32).to(dtype)`.
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

}  // namespace qwen36_moe
