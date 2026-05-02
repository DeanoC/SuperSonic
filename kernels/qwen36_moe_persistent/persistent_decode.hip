// Persistent decode megakernel for Qwen3.6-MoE — Phase 3e.
//
// One cooperative HIP launch processes the entire decode step (all
// `num_layers` layers × {attn or linear-attn, FFN}) instead of launching
// 81 separate step kernels. The pattern mirrors
// `kernels/full_attention_4b.hip::supersonic_qwen35_persistent_decode_kernel`
// (~1400 LoC, the proven Qwen3.5-4B reference).
//
// The compute lives in three `__device__ inline` headers (Phase 3a-3d):
//   - `qwen36_moe_persistent/full_attn_phase.cuh`    (PR #120)
//   - `qwen36_moe_persistent/linear_attn_phase.cuh`  (PR #121)
//   - `qwen36_moe_persistent/ffn_phase.cuh`          (PR #124)
// This file is just the descriptor walk + dispatch + barrier orchestration.
//
// Win: at 35B-A3B / gfx1100 the chained path runs 81 launches/token (40
// attn + 40 ffn + 1 lm_head) at ~30 µs HIP overhead each = ~2.4 ms/token,
// or ~9% of the 27 ms/token chain time. The megakernel collapses this to 1
// launch — recovered independent of MTP accept rate.
//
// Workspace + LDS sizing rationale
// --------------------------------
// All three phases use `extern __shared__ float lds[]` of size
// `(hidden + block_size) * 4` bytes (~9 KiB at hidden=2048, block_size=256).
// They never run concurrently within a layer (grid_barrier between them),
// so a single LDS region of that size is correct for the kernel launch.
//
// Workspace (F32 global scratch) is sized at the host as
// `max(attn_workspace_floats, ffn_workspace_floats)` — same as the
// chained driver. The phases never overlap, so they can share the buffer.
//
// Counter slots: phases use counters[0..16]. attn/linear only touch
// counters[0]; ffn uses counters[g] for Phase G work-stealing and
// counters[top_k+g] for Phase I. Between phases the megakernel zeros
// counters[0..16] under a grid_barrier so each phase starts clean. With
// top_k=8 (the production layout) all 16 slots are used.
//
// Residual ping-pong
// ------------------
// Two `[hidden]` BF16 buffers (`hidden_ping`, `hidden_pong`). The host
// uploads the initial hidden into `hidden_ping`. Inside each layer
// iteration the kernel:
//   1. attn writes to `out_buf` (post-attn residual)
//   2. swap so the next phase reads it
//   3. ffn writes to `out_buf` (post-ffn residual = layer output)
//   4. swap back — `in_buf` is now the layer output, ready for the next
//      layer's attn input.
// Two swaps per iteration cancel, so after every layer iter `in_buf` ==
// `hidden_ping`. The final hidden therefore always lands in `hidden_ping`
// regardless of `num_layers` parity.

#pragma once

#include "qwen36_moe_persistent/helpers.cuh"
#include "qwen36_moe_persistent/full_attn_phase.cuh"
#include "qwen36_moe_persistent/linear_attn_phase.cuh"
#include "qwen36_moe_persistent/ffn_phase.cuh"

namespace qwen36_moe {

// Reset counters[0..16] to zero between phases. Two grid_barriers needed:
//
//   1. First barrier ensures EVERY block has exited the previous phase
//      before any block writes counters[]. Without this, a block that
//      finished its rows early (and returned from the phase device fn)
//      could start clearing counters[0] while another block is still
//      running the previous phase's `atomicAdd(&counters[0], 1u)` work-
//      stealing tail — a real race that bit at the layer-1→layer-2
//      transition during bring-up.
//
//   2. Block 0's threads 0..15 each clear one counter slot.
//
//   3. Second barrier publishes the resets to every block before the next
//      phase's first `atomicAdd` reads them.
//
// Cost: 2 grid_barriers per phase boundary (~few µs each on gfx1100). At
// 40 layers × 2 phase boundaries/layer × 2 grid_barriers each = 160 extra
// barriers per token, ~few hundred µs total — negligible vs the 80-launch
// overhead saved.
__device__ inline void reset_counters_16(
    unsigned int* __restrict__ counters,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag,
    int                        num_blocks
) {
    // Sync: every block has exited the previous phase.
    grid_barrier(barrier_counter, barrier_flag, num_blocks);
    if (blockIdx.x == 0 && threadIdx.x < 16) {
        counters[threadIdx.x] = 0u;
    }
    // Publish: counter resets visible before the next phase reads them.
    grid_barrier(barrier_counter, barrier_flag, num_blocks);
}

// Persistent decode kernel template. Instantiated by the bridge for
// (T=hip_bfloat16, USE_WMMA={true,false}) at launch time based on
// `device_supports_wmma_bf16(ordinal)` + dim divisibility.
//
// Each layer runs:
//   1. attn or linear-attn (selected by descriptor's `is_full_attention`)
//      writes post-attn residual into `out_buf`.
//   2. reset_counters_16
//   3. ffn writes post-ffn residual into the swapped-back `out_buf` (= the
//      original `in_buf`, since we swap once between phases).
//   4. reset_counters_16, swap so `out_buf` becomes next-layer input.
template <typename T, bool USE_WMMA>
__global__ void qwen36_moe_persistent_decode_kernel(
    int                                  num_layers,
    const DecodeLayerDesc* __restrict__  layers,
    const Int4ScaleDesc*   __restrict__  int4_scales,           // nullable
    int                                  hidden,
    int                                  num_heads,
    int                                  num_kv_heads,
    int                                  head_dim,
    int                                  rotary_dim,
    int                                  num_k_heads,
    int                                  num_v_heads,
    int                                  head_k_dim,
    int                                  head_v_dim,
    int                                  conv_kernel_dim,
    int                                  num_experts,
    int                                  moe_intermediate,
    int                                  shared_intermediate,
    int                                  top_k,
    float                                rope_theta,
    float                                rms_norm_eps,
    int                                  position,
    T*    __restrict__                   hidden_ping,            // [hidden] BF16
    T*    __restrict__                   hidden_pong,            // [hidden] BF16
    float* __restrict__                  workspace,              // shared scratch
    int*   __restrict__                  ffn_topk_idx_scratch,   // [top_k]
    unsigned int* __restrict__           counters,
    unsigned int* __restrict__           barrier_counter,
    unsigned int* __restrict__           barrier_flag) {
    const int num_blocks = static_cast<int>(gridDim.x);

    // Residual ping-pong tracker. `in_buf` is what the next phase reads;
    // `out_buf` is where it writes. Each phase swap is local to this fn.
    T* in_buf  = hidden_ping;
    T* out_buf = hidden_pong;

    for (int li = 0; li < num_layers; ++li) {
        const DecodeLayerDesc& d = layers[li];
        const Int4ScaleDesc* i4 =
            (int4_scales != nullptr) ? &int4_scales[li] : nullptr;
        const int i4_gs = (i4 != nullptr) ? i4->group_size : 0;

        // ---- attn or linear-attn ----
        if (d.is_full_attention != 0) {
            qwen36_moe_attn_step_device<T, USE_WMMA>(
                /*stage*/ 5,
                hidden, num_heads, num_kv_heads, head_dim,
                rotary_dim, rope_theta, rms_norm_eps,
                position, /*cache_pos*/ -1,
                in_buf,
                static_cast<const T*>(d.input_norm_w),
                static_cast<const T*>(d.q_proj_w),
                static_cast<const T*>(d.k_proj_w),
                static_cast<const T*>(d.v_proj_w),
                static_cast<const T*>(d.q_norm_w),
                static_cast<const T*>(d.k_norm_w),
                static_cast<const T*>(d.o_proj_w),
                i4_gs,
                i4 ? static_cast<const hip_bfloat16*>(i4->q_proj_scale) : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->q_proj_zero)  : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->k_proj_scale) : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->k_proj_zero)  : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->v_proj_scale) : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->v_proj_zero)  : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->o_proj_scale) : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->o_proj_zero)  : nullptr,
                out_buf,
                workspace,
                static_cast<T*>(d.kv_cache_k),
                static_cast<T*>(d.kv_cache_v),
                d.kv_max_t,
                counters, barrier_counter, barrier_flag);
        } else {
            qwen36_moe_linear_step_device<T, USE_WMMA>(
                /*stage*/ 5,
                hidden, num_k_heads, num_v_heads,
                head_k_dim, head_v_dim, conv_kernel_dim, rms_norm_eps,
                in_buf,
                static_cast<const T*>(d.input_norm_w),
                static_cast<const T*>(d.linear_in_proj_qkv_w),
                static_cast<const T*>(d.linear_in_proj_z_w),
                static_cast<const T*>(d.linear_in_proj_a_w),
                static_cast<const T*>(d.linear_in_proj_b_w),
                static_cast<const T*>(d.linear_conv1d_w),
                /*conv1d_bias*/ static_cast<const T*>(nullptr),
                static_cast<const T*>(d.linear_dt_bias),
                static_cast<const T*>(d.linear_a_log_exp),
                static_cast<const T*>(d.linear_norm_w),
                static_cast<const T*>(d.linear_out_proj_w),
                static_cast<T*>(d.linear_conv_state),
                static_cast<float*>(d.linear_recurrent_state),
                i4_gs,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_in_proj_qkv_scale) : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_in_proj_qkv_zero)  : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_in_proj_z_scale)   : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_in_proj_z_zero)    : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_out_proj_scale)    : nullptr,
                i4 ? static_cast<const hip_bfloat16*>(i4->linear_out_proj_zero)     : nullptr,
                out_buf,
                workspace,
                counters, barrier_counter, barrier_flag);
        }

        // Reset counters[0..16] before FFN. Attn/linear only used
        // counters[0]; FFN needs counters[0..2*top_k] all clean.
        reset_counters_16(counters, barrier_counter, barrier_flag, num_blocks);

        // Swap — `out_buf` (just-written post-attn residual) becomes the
        // next phase's input, the old in_buf becomes the next-phase output
        // slot.
        T* tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;

        // ---- FFN ----
        qwen36_moe_ffn_step_device<T, USE_WMMA>(
            /*stage*/ 5,
            hidden, num_experts,
            moe_intermediate, shared_intermediate, top_k,
            rms_norm_eps,
            in_buf,
            static_cast<const T*>(d.post_attn_norm_w),
            static_cast<const T*>(d.router_w),
            static_cast<const T*>(d.experts_gate_up_w),
            static_cast<const T*>(d.experts_down_w),
            static_cast<const T*>(d.shared_expert_gate_proj_w),
            static_cast<const T*>(d.shared_expert_up_proj_w),
            static_cast<const T*>(d.shared_expert_down_proj_w),
            static_cast<const T*>(d.shared_expert_gate_w),
            i4_gs,
            i4 ? static_cast<const hip_bfloat16*>(i4->experts_gate_up_scale) : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->experts_gate_up_zero)  : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->experts_down_scale)    : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->experts_down_zero)     : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_gate_proj_scale) : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_gate_proj_zero)  : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_up_proj_scale)   : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_up_proj_zero)    : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_down_proj_scale) : nullptr,
            i4 ? static_cast<const hip_bfloat16*>(i4->shared_expert_down_proj_zero)  : nullptr,
            out_buf,
            ffn_topk_idx_scratch,
            workspace,
            counters, barrier_counter, barrier_flag);

        // Reset counters before the next layer's attn (which only touches
        // counters[0], so resetting all 16 is over-conservative but still
        // cheap — gives the FFN the same clean-slate guarantee for layers
        // at the cost of one barrier).
        reset_counters_16(counters, barrier_counter, barrier_flag, num_blocks);

        // Swap back — `out_buf` holds post-ffn residual = next layer's input.
        tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    // Each iteration does TWO swaps that cancel, so after every layer
    // `in_buf == hidden_ping`. The host downloads `hidden_ping` as the
    // final hidden regardless of `num_layers` parity.
}

}  // namespace qwen36_moe
