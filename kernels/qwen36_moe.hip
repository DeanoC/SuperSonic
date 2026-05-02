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

// Phase 3a: shared __device__ primitives (bf16 rounding, INT4 dequant,
// WMMA tile, grid_barrier, load_as_float). Lives in
// `qwen36_moe_persistent/helpers.cuh` so phase-extraction headers
// (full_attn_phase.cuh, linear_attn_phase.cuh, ffn_phase.cuh) can include
// the same primitives without circular dependencies on this file's step
// kernels. The header opens namespace qwen36_moe { ... } itself.
#include "qwen36_moe_persistent/helpers.cuh"

// Phase 3b: full-attention phase body lives in this header as a
// `__device__ inline` function (`qwen36_moe_attn_step_device`). The
// existing `qwen36_moe_attn_step_kernel` __global__ stays as a thin
// wrapper for the per-block parity tests; the persistent megakernel
// (Phase 3e) calls the device fn directly.
#include "qwen36_moe_persistent/full_attn_phase.cuh"

// Phase 3c: linear-attention phase body — same wrapper pattern as 3b.
// `qwen36_moe_linear_step_device` lives in
// `qwen36_moe_persistent/linear_attn_phase.cuh`.
#include "qwen36_moe_persistent/linear_attn_phase.cuh"

// Phase 3d: MoE FFN phase body — same wrapper pattern as 3b/3c.
// `qwen36_moe_ffn_step_device` lives in
// `qwen36_moe_persistent/ffn_phase.cuh`.
#include "qwen36_moe_persistent/ffn_phase.cuh"

namespace qwen36_moe {

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

// -- Int4 scale/zero parallel struct (mirror of Rust Qwen36MoeInt4ScaleDesc) -
//
// Phase 3e: read alongside DecodeLayerDesc by the persistent megakernel.
// Each `*_scale` / `*_zero` pair is consulted independently per tensor — a
// null `*_scale` keeps that tensor on the BF16 path. `group_size` applies
// uniformly across all INT4 tensors in the descriptor; 0 disables INT4
// entirely (matches the BF16 bake).
struct Int4ScaleDesc {
    const void* q_proj_scale;
    const void* q_proj_zero;
    const void* k_proj_scale;
    const void* k_proj_zero;
    const void* v_proj_scale;
    const void* v_proj_zero;
    const void* o_proj_scale;
    const void* o_proj_zero;

    const void* linear_in_proj_qkv_scale;
    const void* linear_in_proj_qkv_zero;
    const void* linear_in_proj_z_scale;
    const void* linear_in_proj_z_zero;
    const void* linear_out_proj_scale;
    const void* linear_out_proj_zero;

    const void* experts_gate_up_scale;
    const void* experts_gate_up_zero;
    const void* experts_down_scale;
    const void* experts_down_zero;

    const void* shared_expert_gate_proj_scale;
    const void* shared_expert_gate_proj_zero;
    const void* shared_expert_up_proj_scale;
    const void* shared_expert_up_proj_zero;
    const void* shared_expert_down_proj_scale;
    const void* shared_expert_down_proj_zero;

    int group_size;
};

// `grid_barrier` and `grid_barrier_reset_counter` live in
// `qwen36_moe_persistent/helpers.cuh` (Phase 3a).

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
    // Phase 3b: body extracted to `qwen36_moe_persistent/full_attn_phase.cuh`.
    // This wrapper preserves the per-block parity-test entry point.
    qwen36_moe_attn_step_device<T, USE_WMMA>(
        stage, hidden, num_heads, num_kv_heads, head_dim,
        rotary_dim, rope_theta, rms_norm_eps,
        position, cache_pos,
        input_hidden, input_norm_w,
        q_proj_w, k_proj_w, v_proj_w,
        q_norm_w, k_norm_w, o_proj_w,
        int4_group_size,
        q_proj_scale, q_proj_zero,
        k_proj_scale, k_proj_zero,
        v_proj_scale, v_proj_zero,
        o_proj_scale, o_proj_zero,
        output, workspace,
        kv_cache_k, kv_cache_v, kv_max_t,
        counters, barrier_counter, barrier_flag);
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
    // Phase 3c: body extracted to `qwen36_moe_persistent/linear_attn_phase.cuh`.
    qwen36_moe_linear_step_device<T, USE_WMMA>(
        stage, hidden, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, conv_kernel_dim, rms_norm_eps,
        input_hidden, input_norm_w,
        in_proj_qkv_w, in_proj_z_w, in_proj_a_w, in_proj_b_w,
        conv1d_w, conv1d_bias, dt_bias, a_log,
        norm_w, out_proj_w,
        conv_state, recurrent_state,
        int4_group_size,
        in_proj_qkv_scale, in_proj_qkv_zero,
        in_proj_z_scale,   in_proj_z_zero,
        out_proj_scale,    out_proj_zero,
        output, workspace,
        counters, barrier_counter, barrier_flag);
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
    // Phase 3d: body extracted to `qwen36_moe_persistent/ffn_phase.cuh`.
    qwen36_moe_ffn_step_device<T, USE_WMMA>(
        stage, hidden, num_experts,
        moe_intermediate, shared_intermediate, top_k,
        rms_norm_eps,
        input_hidden, post_attn_norm_w, gate_w,
        gate_up_proj_w, down_proj_w,
        shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w,
        shared_expert_gate_w,
        int4_group_size,
        gate_up_proj_scale, gate_up_proj_zero,
        down_proj_scale, down_proj_zero,
        shared_gate_proj_scale, shared_gate_proj_zero,
        shared_up_proj_scale, shared_up_proj_zero,
        shared_down_proj_scale, shared_down_proj_zero,
        output, output_idx, workspace,
        counters, barrier_counter, barrier_flag);
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
    // Optional output: BF16-rounded post-RMSNorm hidden state. Used by
    // the Qwen3.6-MoE self-speculative MTP path (Phase 6.2c.3) to capture
    // `h_post` for the recurrent feed into the next draft step. Pass
    // null to skip — the base-model decode path doesn't need it. Only
    // block 0 writes (RMSNorm is computed redundantly per block, so all
    // blocks' values agree bit-for-bit; gating on block 0 saves the 95
    // other blocks' worth of redundant global writes).
    T*                     __restrict__ x_normed_out,    // [hidden] BF16, nullable
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
    const bool export_x_normed = (x_normed_out != nullptr) && (blockIdx.x == 0);
    for (int col = tid; col < hidden; col += block_size) {
        const float w = load_as_float<T>(final_norm_w, col);
        const float val = bf16_round_rne_f32(x_norm_lds[col] * inv_rms * (1.0f + w));
        x_norm_lds[col] = val;
        if (export_x_normed) {
            x_normed_out[col] = static_cast<T>(val);
        }
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
    // Optional output: BF16-rounded post-RMSNorm hidden state. See the
    // scalar-kernel comment above; only block 0 (the first vocab tile)
    // emits, the rest skip to avoid 15k redundant global writes.
    T*                     __restrict__ x_normed_out,    // [hidden] BF16, nullable
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
    const bool export_x_normed = (x_normed_out != nullptr) && (blockIdx.x == 0);
    uint16_t* x_normed_out_u16 =
        reinterpret_cast<uint16_t*>(x_normed_out);
    for (int col = lane; col < hidden; col += 32) {
        const float vh = load_as_float<T>(final_hidden, col);
        const float vw = load_as_float<T>(final_norm_w, col);
        const float val = bf16_round_rne_f32(vh * inv_rms * (1.0f + vw));
        uint32_t bits;
        __builtin_memcpy(&bits, &val, 4);
        const uint16_t hi = static_cast<uint16_t>(bits >> 16);
        x_norm_lds_u16[col] = hi;
        if (export_x_normed) {
            x_normed_out_u16[col] = hi;
        }
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
    (void)logits; (void)x_normed_out;
    (void)hidden; (void)vocab; (void)rms_norm_eps;
#endif
}

// =============================================================================
// Phase 6.4a: BATCHED RDNA3 WMMA BF16 lm_head GEMV (wave32, 16x16x16 f32-acc),
// one-wavefront-per-16-vocab-tile, M = K input rows in a single WMMA op.
//
// Companion to `qwen36_moe_lm_head_wmma_kernel` above; same numerics
// boundary, but processes `m` ≤ 16 input rows in one WMMA tile instead
// of the single-token M=1 case. Used by the speculative decode driver
// (Phase 6.4b/c will wire it in) to amortize the 970 MiB lm_head BF16
// weight read across K verify-step queries — at K=3 the batched call
// is ~1.5× the cost of single-M (vs 3× sequential), saving ~2 ms /
// spec step on the local 7900 XTX.
//
// Layout:
//   final_hidden  [m, hidden]      BF16 (K input rows)
//   final_norm_w  [hidden]         BF16 (shared across all m rows)
//   lm_head_w     [vocab, hidden]  BF16
//   logits        [m, vocab]       BF16 (K logit rows)
//   x_normed_out  [m, hidden]      BF16, nullable (K post-RMSNorm rows)
//
// Grid : (ceil(vocab / 16), 1, 1)
// Block: 32 threads = one wave32
//
// Per-block phases:
//   A: K-batched RMSNorm — iterate m rows, RMSNorm each into LDS at
//      offset `r * hidden` (BF16-rounded, F32→BF16 once).
//   B: K-loop, one WMMA tile per K-slab. A operand is 16 rows × 16
//      cols where the first `m` rows carry real activations and rows
//      m..15 zero-fill (the WMMA fires the same number of muls; only
//      the first m output rows are read out). B operand same as
//      single-M kernel — 16 vocab rows × 16 K elements.
//   C: Each lane_half writes 8 acc entries to output rows
//      `2*i + lane_half`; rows ≥ m are skipped.
//
// Constraints:
//   - m ∈ [1, 16]. Anything larger needs row-tiling on the grid Y axis
//     (mirrors `supersonic_qwen35_matmul_int4_dequant_wmma_small_m_kernel`
//     in `kernels/full_attention_4b.hip`); not needed for the K=3
//     speculative case.
//   - `x_normed_out` (when non-null) is sized for `m * hidden` BF16
//     elements. Only block 0 (the first vocab tile) writes to it,
//     same single-write contract as the single-M kernel.
// =============================================================================

template <typename T>
__global__ __launch_bounds__(32, 4)
void qwen36_moe_lm_head_batched_wmma_kernel(
    int                                  m,                // 1..=16
    const T*               __restrict__ final_hidden,    // [m, hidden] BF16
    const T*               __restrict__ final_norm_w,    // [hidden]    BF16
    const T*               __restrict__ lm_head_w,       // [vocab, hidden] BF16
    T*                     __restrict__ logits,          // [m, vocab]  BF16
    T*                     __restrict__ x_normed_out,    // [m, hidden] BF16, nullable
    int                                  hidden,
    int                                  vocab,
    float                                rms_norm_eps) {
#ifdef SUPERSONIC_QWEN36_HAS_WMMA_BF16
    const int lane = threadIdx.x;          // 0..31
    const int lane_row = lane & 15;        // 0..15
    const int lane_half = lane >> 4;       // 0 / 1
    const int tile_col = blockIdx.x * 16;

    // LDS layout:
    //   shared_scratch [32]            F32  — RMSNorm wave reduction (reused per row)
    //   x_norm_lds_u16 [m * hidden]    u16  — K BF16-rounded RMSNorm outputs
    extern __shared__ unsigned char lds_raw[];
    float*    shared_scratch = reinterpret_cast<float*>(lds_raw);
    uint16_t* x_norm_lds_u16 = reinterpret_cast<uint16_t*>(shared_scratch + 32);

    const bool export_x_normed = (x_normed_out != nullptr) && (blockIdx.x == 0);
    uint16_t* x_normed_out_u16 = reinterpret_cast<uint16_t*>(x_normed_out);

    // -- Phase A: per-row RMSNorm of final_hidden, K rows ------------------
    for (int row = 0; row < m; ++row) {
        const T* row_in = final_hidden + static_cast<size_t>(row) * hidden;
        uint16_t* row_lds = x_norm_lds_u16 + static_cast<size_t>(row) * hidden;
        uint16_t* row_export =
            export_x_normed
                ? (x_normed_out_u16 + static_cast<size_t>(row) * hidden)
                : nullptr;

        float partial_sq = 0.0f;
        for (int col = lane; col < hidden; col += 32) {
            const float v = load_as_float<T>(row_in, col);
            partial_sq += v * v;
        }
        shared_scratch[lane] = partial_sq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (lane < s) shared_scratch[lane] += shared_scratch[lane + s];
            __syncthreads();
        }
        const float inv_rms = rsqrtf(
            shared_scratch[0] / static_cast<float>(hidden) + rms_norm_eps);

        // `(1.0 + w)` HF unit offset, BF16-round, store as BF16 bit pattern.
        // RMSNorm gain is shared across K rows, so the same `final_norm_w`
        // is read K times. PCIe-resident reads are cached after the first
        // pass — the redundant re-reads cost ~few µs total at hidden=2048.
        for (int col = lane; col < hidden; col += 32) {
            const float vh = load_as_float<T>(row_in, col);
            const float vw = load_as_float<T>(final_norm_w, col);
            const float val = bf16_round_rne_f32(vh * inv_rms * (1.0f + vw));
            uint32_t bits;
            __builtin_memcpy(&bits, &val, 4);
            const uint16_t hi = static_cast<uint16_t>(bits >> 16);
            row_lds[col] = hi;
            if (row_export != nullptr) {
                row_export[col] = hi;
            }
        }
        __syncthreads();
    }

    // -- Phase B: WMMA K-loop, M=m × hidden contraction × 16 vocab ----------
    const int rhs_row_idx = tile_col + lane_row;
    const bool rhs_in_range = rhs_row_idx < vocab;

    const uint16_t* lm_head_w_u16 = reinterpret_cast<const uint16_t*>(lm_head_w);
    const size_t rhs_row_base = static_cast<size_t>(rhs_row_idx) * hidden;

    qwen36_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = hidden & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        // A operand: row `lane_row` of the M×16 tile. Real for rows 0..m-1;
        // zero-padded for rows m..15. The WMMA still issues 4096 muls;
        // bandwidth-bound on the B operand, so the M-padding is free.
        qwen36_short16 a;
        if (lane_row < m) {
            const uint16_t* row_lds = x_norm_lds_u16
                + static_cast<size_t>(lane_row) * hidden;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(row_lds[kk + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

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

    // K tail (defensive — hidden=2048 is /16 in production).
    if (k_main != hidden) {
        qwen36_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        qwen36_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (lane_row < m) {
            const uint16_t* row_lds = x_norm_lds_u16
                + static_cast<size_t>(lane_row) * hidden;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < hidden) a[i] = static_cast<short>(row_lds[kc]);
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

    // -- Phase C: write all m output rows ---------------------------------
    // RDNA3 wave32 WMMA acc layout: acc[i] -> C[2*i + lane_half, lane_row].
    // Each lane_half handles 8 output rows (0,2,4,...,14 for half=0; 1,3,
    // 5,...,15 for half=1). Skip rows ≥ m (zero-padded inputs produce
    // zero outputs but writing them would corrupt logits beyond `m`).
    const int out_col = tile_col + lane_row;
    if (out_col < vocab) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = 2 * i + lane_half;
            if (out_row < m) {
                const float val = bf16_round_rne_f32(acc[i]);
                logits[static_cast<size_t>(out_row) * vocab + out_col] =
                    static_cast<T>(val);
            }
        }
    }
#else
    (void)m;
    (void)final_hidden; (void)final_norm_w; (void)lm_head_w;
    (void)logits; (void)x_normed_out;
    (void)hidden; (void)vocab; (void)rms_norm_eps;
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
