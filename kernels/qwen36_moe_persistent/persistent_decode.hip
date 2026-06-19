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
#include "qwen36_moe_persistent/lm_head_phase.cuh"

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
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            for (int i = 0; i < 16; ++i) {
                __atomic_store_n(counters + i, 0u, __ATOMIC_RELAXED);
            }
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_flag, phase + 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

enum PersistentDecodeMode {
    PERSISTENT_MODE_FULL = 0,
    PERSISTENT_MODE_ROUTER_ONLY = 1,
    PERSISTENT_MODE_FFN_ONLY = 2,
    PERSISTENT_MODE_ATTN_ONLY = 3,
    PERSISTENT_MODE_FFN_STAGE1 = 4,
    PERSISTENT_MODE_FFN_STAGE2 = 5,
    PERSISTENT_MODE_FFN_STAGE3 = 6,
    PERSISTENT_MODE_FFN_STAGE4 = 7,
    PERSISTENT_MODE_FFN_STAGE5 = 8,
    PERSISTENT_MODE_LINEAR_STAGE1 = 9,
    PERSISTENT_MODE_LINEAR_STAGE2 = 10,
    PERSISTENT_MODE_LINEAR_STAGE3 = 11,
    PERSISTENT_MODE_LINEAR_STAGE4 = 12,
    PERSISTENT_MODE_LINEAR_STAGE5 = 13,
};

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
    int                                  start_layer,
    int                                  end_layer_exclusive,
    int                                  mode,
    const DecodeLayerDesc* __restrict__  layers,
    const Int4ScaleDesc*   __restrict__  int4_scales,           // nullable
    const KVCacheFp8Desc*  __restrict__  kv_fp8_descs,          // nullable
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
    int                                  vocab,                  // 0 if no lm_head fold
    float                                rope_theta,
    float                                rms_norm_eps,
    int                                  position,
    // RoPE rotation uses `position`; the KV-cache slot uses
    // `cache_pos` when ≥ 0, otherwise inherits `position` (the dense
    // base-decode case). SpecPrefill sparse-prefill and MTP both rely
    // on the decoupled-slot path so kept tokens land in compact KV
    // slots while still rotating at their original prompt positions.
    int                                  cache_pos,
    const T* __restrict__                embed_w,                // [vocab, hidden] BF16, nullable
    int                                  token_id,               // >=0 => load embed_w[token_id]
    const unsigned int* __restrict__     token_ids,              // [prefill_len], nullable
    int                                  prefill_len,            // >1 => dense prompt prefix loop
    T*    __restrict__                   hidden_ping,            // [hidden] BF16
    T*    __restrict__                   hidden_pong,            // [hidden] BF16
    float* __restrict__                  workspace,              // shared scratch
    int*   __restrict__                  ffn_topk_idx_scratch,   // [top_k]
    // Phase 3f: optional folded final RMSnorm + lm_head GEMV. When all
    // three are non-null and vocab > 0, the megakernel runs the lm_head
    // phase after the last layer's FFN and writes logits to
    // `logits_out`. Pass nullptr to skip — prefill steps don't need
    // logits, the engine sets logits_out=null on those.
    const T* __restrict__                final_norm_w,           // [hidden] BF16, nullable
    const T* __restrict__                lm_head_w,              // [vocab, hidden] BF16, nullable
    T*       __restrict__                logits_out,             // [vocab] BF16, nullable
    unsigned int* __restrict__           top1_out,               // [1] U32, nullable
    unsigned int* __restrict__           counters,
    unsigned int* __restrict__           barrier_counter,
    unsigned int* __restrict__           barrier_flag) {
    const int num_blocks = static_cast<int>(gridDim.x);

    // Residual ping-pong tracker. `in_buf` is what the next phase reads;
    // `out_buf` is where it writes. Each phase swap is local to this fn.
    T* in_buf  = hidden_ping;
    T* out_buf = hidden_pong;

    const int run_tokens = (token_ids != nullptr && prefill_len > 0) ? prefill_len : 1;
    for (int token_step = 0; token_step < run_tokens; ++token_step) {
    const int step_token_id = (token_ids != nullptr)
        ? static_cast<int>(token_ids[token_step])
        : token_id;
    const int step_position = position + token_step;
    const int step_cache_pos = (cache_pos >= 0) ? (cache_pos + token_step) : cache_pos;

    if (embed_w != nullptr && step_token_id >= 0) {
        const T* row = embed_w + static_cast<size_t>(step_token_id) * hidden;
        for (int col = blockIdx.x * blockDim.x + threadIdx.x;
             col < hidden;
             col += blockDim.x * gridDim.x) {
            hidden_ping[col] = row[col];
        }
        grid_barrier(barrier_counter, barrier_flag, num_blocks);
    }

    if (mode == PERSISTENT_MODE_FFN_ONLY ||
        (mode >= PERSISTENT_MODE_FFN_STAGE1 &&
         mode <= PERSISTENT_MODE_FFN_STAGE5)) {
        const DecodeLayerDesc& d = layers[start_layer];
        const Int4ScaleDesc* i4 =
            (int4_scales != nullptr) ? &int4_scales[start_layer] : nullptr;
        const int i4_gs = (i4 != nullptr) ? i4->group_size : 0;
        const int ffn_stage = (mode == PERSISTENT_MODE_FFN_ONLY)
            ? 5
            : (mode - PERSISTENT_MODE_FFN_STAGE1 + 1);

        qwen36_moe_ffn_step_device<T, USE_WMMA>(
            ffn_stage,
            hidden, num_experts,
            moe_intermediate, shared_intermediate, top_k,
            rms_norm_eps,
            hidden_pong,
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
            hidden_ping,
            ffn_topk_idx_scratch,
            workspace,
            counters, barrier_counter, barrier_flag);
        return;
    }

    if (mode >= PERSISTENT_MODE_LINEAR_STAGE1 &&
        mode <= PERSISTENT_MODE_LINEAR_STAGE5) {
        const DecodeLayerDesc& d = layers[start_layer];
        const Int4ScaleDesc* i4 =
            (int4_scales != nullptr) ? &int4_scales[start_layer] : nullptr;
        const int i4_gs = (i4 != nullptr) ? i4->group_size : 0;
        const int linear_stage = mode - PERSISTENT_MODE_LINEAR_STAGE1 + 1;

        qwen36_moe_linear_step_device<T, USE_WMMA>(
            linear_stage,
            hidden, num_k_heads, num_v_heads,
            head_k_dim, head_v_dim, conv_kernel_dim, rms_norm_eps,
            hidden_ping,
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
            hidden_pong,
            workspace,
            counters, barrier_counter, barrier_flag,
            /*suppress_stage_publish*/ true,
            /*mutate_state*/ linear_stage == 5);
        return;
    }

    for (int li = start_layer; li < end_layer_exclusive; ++li) {
        const DecodeLayerDesc& d = layers[li];
        const Int4ScaleDesc* i4 =
            (int4_scales != nullptr) ? &int4_scales[li] : nullptr;
        const int i4_gs = (i4 != nullptr) ? i4->group_size : 0;
        const KVCacheFp8Desc* kv_fp8 =
            (kv_fp8_descs != nullptr) ? &kv_fp8_descs[li] : nullptr;

        // ---- attn or linear-attn ----
        if (d.is_full_attention != 0) {
            qwen36_moe_attn_step_device<T, USE_WMMA>(
                /*stage*/ 5,
                hidden, num_heads, num_kv_heads, head_dim,
                rotary_dim, rope_theta, rms_norm_eps,
                step_position, step_cache_pos,
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
                /* kv_scale_k */ (kv_fp8 != nullptr)
                    ? static_cast<float*>(kv_fp8->kv_scale_k) : nullptr,
                /* kv_scale_v */ (kv_fp8 != nullptr)
                    ? static_cast<float*>(kv_fp8->kv_scale_v) : nullptr,
                /* kv_shadow_k */ d.kv_shadow_k,
                /* kv_shadow_v */ d.kv_shadow_v,
                /* kv_shadow_start */ d.kv_shadow_start,
                /* kv_shadow_window */ d.kv_shadow_window,
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

        if (mode == PERSISTENT_MODE_ATTN_ONLY) {
            // Profiling-only split: leave the post-attention residual in
            // hidden_pong so a following FfnOnly launch can complete the
            // same layer without rerunning attention.
            return;
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
        const int ffn_stage = (mode == PERSISTENT_MODE_ROUTER_ONLY) ? 1 : 5;
        qwen36_moe_ffn_step_device<T, USE_WMMA>(
            ffn_stage,
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

        if (mode == PERSISTENT_MODE_ROUTER_ONLY) {
            // The post-attention hidden stays in hidden_pong. The host can
            // read ffn_topk_idx_scratch, remap sparse expert pages, then
            // resume this layer with PERSISTENT_MODE_FFN_ONLY.
            return;
        }

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

    // Phase 3f: folded final RMSnorm + lm_head GEMV. Skipped when any
    // of the three buffers is null (the engine sets them all null on
    // prefill steps to short-circuit the lm_head work).
    if (final_norm_w != nullptr && lm_head_w != nullptr &&
        (logits_out != nullptr || top1_out != nullptr) && vocab > 0) {
        // counters[0..16] are dirty from the last layer's ffn phase;
        // reset_counters_16 syncs across blocks AND publishes the layer
        // loop's writes to in_buf (= hidden_ping after the per-layer
        // double-swap) before the lm_head reads from it as activation.
        reset_counters_16(counters, barrier_counter, barrier_flag, num_blocks);

        qwen36_moe_lm_head_device<T, USE_WMMA>(
            hidden, vocab, rms_norm_eps,
            in_buf, final_norm_w, lm_head_w, logits_out,
            top1_out, workspace,
            counters, barrier_counter, barrier_flag, num_blocks);
    }
}

}  // namespace qwen36_moe
