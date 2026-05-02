// Qwen3.6-MoE HIP launch bridge — PR 4 stub.
//
// Calls into the descriptor-walk stub kernel in qwen36_moe.hip. The
// `qwen36_moe_hip_stub_launch` extern is the only symbol exposed today;
// once the real megakernel lands, this file gains a
// `qwen36_moe_hip_persistent_decode` entry alongside the stub (kept for
// the smoke test).

#include "qwen36_moe.hip"

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <mutex>
#include <stdint.h>

namespace {

// Cache the per-device gfx11xx detection result. WMMA bf16 is RDNA3-only
// (gfx1100..gfx1152). Mirrors the helper in `full_attention_bridge_4b.cpp`
// — we keep an independent copy here rather than introduce a shared header
// because each model family's bridge is its own compilation unit (hipcc
// codegen on gfx11xx is sensitive to cross-contamination, see CLAUDE.md).
//
// Honors `SUPERSONIC_QWEN4B_DISABLE_WMMA` (the existing global override
// shared with the qwen35-4b/Gemma 4 prefill paths) so a single env var
// disables every WMMA route in the runtime; useful for A/B perf work.
bool device_supports_wmma_bf16(int device_ordinal) {
    static std::once_flag env_once;
    static bool env_disabled = false;
    std::call_once(env_once, [] {
        const char* env = std::getenv("SUPERSONIC_QWEN4B_DISABLE_WMMA");
        env_disabled = (env != nullptr && env[0] != '\0' && env[0] != '0');
    });
    if (env_disabled) return false;

    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        return arch && arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
               arch[3] == '1' && arch[4] == '1';
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        return probe_arch(device_ordinal);
    }
    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}


struct ScopedHipDevice {
    int  previous = -1;
    bool changed  = false;

    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) {
            hipSetDevice(target);
            changed = true;
        }
    }
    ~ScopedHipDevice() {
        if (changed && previous >= 0) hipSetDevice(previous);
    }
};

// Pin Rust↔C++ struct layout. If this fails at compile time, someone
// reordered fields on one side without the other. Update both sides
// together.
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) >= 256,
              "Qwen36MoeDecodeLayerDesc shrunk unexpectedly");
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) <= 512,
              "Qwen36MoeDecodeLayerDesc grew unexpectedly");

} // namespace

// `dtype` encoding follows the Qwen/Gemma/Phi bridges: 0 = half, 2 = bf16.
// The stub ignores dtype because it does no math; the real kernel will
// branch on it.
extern "C" int qwen36_moe_hip_stub_launch(
    int                                  dtype,
    size_t                               device_ordinal,
    size_t                               num_layers,
    const qwen36_moe::DecodeLayerDesc*   layers,
    float*                               workspace,
    unsigned int*                        counters,
    unsigned int*                        barrier_counter,
    unsigned int*                        barrier_flag) {
    (void)dtype;
    if (num_layers == 0 || num_layers > 1024) return 100;
    if (layers == nullptr || workspace == nullptr) return 101;
    if (counters == nullptr || barrier_counter == nullptr ||
        barrier_flag == nullptr) {
        return 102;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 64; // wave-sized; descriptor walk is light

    // Zero the cooperative counter before launch. The kernel uses
    // `atomicAdd` to claim layer indices.
    if (hipMemsetAsync(counters, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    hipLaunchKernelGGL(qwen36_moe::qwen36_moe_descriptor_walk_stub,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_size),
                       0, 0,
                       static_cast<int>(num_layers),
                       layers,
                       workspace,
                       counters,
                       barrier_counter,
                       barrier_flag);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

// PR 4b2 staged-attention parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" int qwen36_moe_hip_attn_step_launch(
    int           dtype,
    size_t        device_ordinal,
    int           stage,
    int           hidden,
    int           num_heads,
    int           num_kv_heads,
    int           head_dim,
    int           rotary_dim,
    float         rope_theta,
    float         rms_norm_eps,
    int           position,
    const void*   input_hidden,
    const void*   input_norm_w,
    const void*   q_proj_w,
    const void*   k_proj_w,
    const void*   v_proj_w,
    const void*   q_norm_w,
    const void*   k_norm_w,
    const void*   o_proj_w,
    int           int4_group_size,
    const void*   q_proj_scale,
    const void*   q_proj_zero,
    const void*   k_proj_scale,
    const void*   k_proj_zero,
    const void*   v_proj_scale,
    const void*   v_proj_zero,
    const void*   o_proj_scale,
    const void*   o_proj_zero,
    void*         output,
    float*        workspace,
    void*         kv_cache_k,
    void*         kv_cache_v,
    int           kv_max_t,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag) {
    if (dtype != 2) return 110;            // only bf16 supported on stage 1
    if (stage < 1 || stage > 5) return 111;
    if (hidden <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return 112;
    }
    if (input_hidden == nullptr || input_norm_w == nullptr ||
        q_proj_w == nullptr || q_norm_w == nullptr ||
        output == nullptr || workspace == nullptr ||
        counters == nullptr || barrier_counter == nullptr ||
        barrier_flag == nullptr) {
        return 113;
    }
    // INT4 sidecars: each scale must be paired with a zero. group_size==0
    // disables INT4 entirely; otherwise it must be positive.
    if (int4_group_size < 0) return 114;
    auto pair_ok = [](const void* s, const void* z) -> bool {
        return (s == nullptr) == (z == nullptr);
    };
    if (!pair_ok(q_proj_scale, q_proj_zero) ||
        !pair_ok(k_proj_scale, k_proj_zero) ||
        !pair_ok(v_proj_scale, v_proj_zero) ||
        !pair_ok(o_proj_scale, o_proj_zero)) {
        return 115;
    }
    const bool any_int4 =
        (q_proj_scale != nullptr) || (k_proj_scale != nullptr) ||
        (v_proj_scale != nullptr) || (o_proj_scale != nullptr);
    if (any_int4 && int4_group_size <= 0) return 116;
    if (!any_int4 && int4_group_size != 0) return 117;

    // KV cache: pointers must be paired (both null or both non-null), and
    // kv_max_t must be positive when enabled + ≥ position+1 to fit the
    // current write.
    const bool kv_enabled = (kv_cache_k != nullptr || kv_cache_v != nullptr);
    if ((kv_cache_k == nullptr) != (kv_cache_v == nullptr)) return 118;
    if (kv_enabled) {
        if (kv_max_t <= 0) return 119;
        if (position < 0 || position >= kv_max_t) return 120;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    // Zero the cooperative counter + barrier state before launch. The kernel
    // expects all three to start at 0; sync_buf is documented as 32 zero
    // bytes by the Rust-side wrapper but a defence-in-depth memset here
    // keeps a misuse from corrupting the launch.
    if (hipMemsetAsync(counters, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + INT4 weights on at least one matmul +
    // dim divisibility (hidden % 16 == 0 and group_size % 16 == 0). The
    // K-chunk is 16; per-lane `in_range` checks handle non-16-aligned
    // output dims (`q_out_dim = 2*H*d`, `Hkv*d`, `hidden`).
    const bool any_int4_attn =
        (q_proj_scale != nullptr) ||
        (k_proj_scale != nullptr) ||
        (v_proj_scale != nullptr) ||
        (o_proj_scale != nullptr);
    const bool wmma_dims_ok_attn =
        (hidden % 16 == 0) &&
        (int4_group_size > 0) &&
        (int4_group_size % 16 == 0);
    const bool use_wmma_attn =
        any_int4_attn &&
        wmma_dims_ok_attn &&
        device_supports_wmma_bf16(static_cast<int>(device_ordinal));

    if (use_wmma_attn) {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_attn_step_kernel<hip_bfloat16, true>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage, hidden, num_heads, num_kv_heads, head_dim, rotary_dim,
            rope_theta, rms_norm_eps, position,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(q_proj_w),
            static_cast<const hip_bfloat16*>(k_proj_w),
            static_cast<const hip_bfloat16*>(v_proj_w),
            static_cast<const hip_bfloat16*>(q_norm_w),
            static_cast<const hip_bfloat16*>(k_norm_w),
            static_cast<const hip_bfloat16*>(o_proj_w),
            int4_group_size,
            static_cast<const hip_bfloat16*>(q_proj_scale),
            static_cast<const hip_bfloat16*>(q_proj_zero),
            static_cast<const hip_bfloat16*>(k_proj_scale),
            static_cast<const hip_bfloat16*>(k_proj_zero),
            static_cast<const hip_bfloat16*>(v_proj_scale),
            static_cast<const hip_bfloat16*>(v_proj_zero),
            static_cast<const hip_bfloat16*>(o_proj_scale),
            static_cast<const hip_bfloat16*>(o_proj_zero),
            static_cast<hip_bfloat16*>(output),
            workspace,
            static_cast<hip_bfloat16*>(kv_cache_k),
            static_cast<hip_bfloat16*>(kv_cache_v),
            kv_max_t,
            counters, barrier_counter, barrier_flag);
    } else {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_attn_step_kernel<hip_bfloat16, false>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage, hidden, num_heads, num_kv_heads, head_dim, rotary_dim,
            rope_theta, rms_norm_eps, position,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(q_proj_w),
            static_cast<const hip_bfloat16*>(k_proj_w),
            static_cast<const hip_bfloat16*>(v_proj_w),
            static_cast<const hip_bfloat16*>(q_norm_w),
            static_cast<const hip_bfloat16*>(k_norm_w),
            static_cast<const hip_bfloat16*>(o_proj_w),
            int4_group_size,
            static_cast<const hip_bfloat16*>(q_proj_scale),
            static_cast<const hip_bfloat16*>(q_proj_zero),
            static_cast<const hip_bfloat16*>(k_proj_scale),
            static_cast<const hip_bfloat16*>(k_proj_zero),
            static_cast<const hip_bfloat16*>(v_proj_scale),
            static_cast<const hip_bfloat16*>(v_proj_zero),
            static_cast<const hip_bfloat16*>(o_proj_scale),
            static_cast<const hip_bfloat16*>(o_proj_zero),
            static_cast<hip_bfloat16*>(output),
            workspace,
            static_cast<hip_bfloat16*>(kv_cache_k),
            static_cast<hip_bfloat16*>(kv_cache_v),
            kv_max_t,
            counters, barrier_counter, barrier_flag);
    }

    // Async dispatch: skip the per-launch `hipDeviceSynchronize` so the host
    // can queue the next step launch without blocking. The default stream
    // serializes all kernel launches and D2D copies in this engine, and
    // `run_chained_decode`'s final D2H copy of `final_hidden_bytes`
    // implicitly drains the queue — so per-step sync is redundant. Saves
    // ~30 µs/launch × 80 launches/token = ~2.4 ms/token. Runtime kernel
    // errors (illegal memory access etc.) defer to that final D2H copy
    // instead of the immediate per-step return; launch-config errors are
    // still caught here via `hipGetLastError`.
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) return 254;
    return 0;
}

// PR 4b3 staged linear-attention parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" int qwen36_moe_hip_linear_step_launch(
    int           dtype,
    size_t        device_ordinal,
    int           stage,
    int           hidden,
    int           num_k_heads,
    int           num_v_heads,
    int           head_k_dim,
    int           head_v_dim,
    int           conv_kernel_dim,
    float         rms_norm_eps,
    const void*   input_hidden,
    const void*   input_norm_w,
    const void*   in_proj_qkv_w,
    const void*   in_proj_z_w,
    const void*   in_proj_a_w,
    const void*   in_proj_b_w,
    const void*   conv1d_w,
    const void*   conv1d_bias,
    const void*   dt_bias,
    const void*   a_log,
    const void*   norm_w,
    const void*   out_proj_w,
    void*         conv_state,
    float*        recurrent_state,
    int           int4_group_size,
    const void*   in_proj_qkv_scale,
    const void*   in_proj_qkv_zero,
    const void*   in_proj_z_scale,
    const void*   in_proj_z_zero,
    const void*   out_proj_scale,
    const void*   out_proj_zero,
    void*         output,
    float*        workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag) {
    if (dtype != 2) return 120;
    if (stage < 1 || stage > 5) return 121;
    if (hidden <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
        head_k_dim <= 0 || head_v_dim <= 0 || conv_kernel_dim <= 0) {
        return 122;
    }
    if (input_hidden == nullptr || input_norm_w == nullptr ||
        in_proj_qkv_w == nullptr || in_proj_z_w == nullptr ||
        in_proj_a_w == nullptr || in_proj_b_w == nullptr ||
        output == nullptr || workspace == nullptr ||
        counters == nullptr || barrier_counter == nullptr ||
        barrier_flag == nullptr) {
        return 123;
    }
    // INT4 sidecars: each scale must be paired with a zero. group_size==0
    // disables INT4 entirely; otherwise it must be positive.
    if (int4_group_size < 0) return 124;
    auto pair_ok = [](const void* s, const void* z) -> bool {
        return (s == nullptr) == (z == nullptr);
    };
    if (!pair_ok(in_proj_qkv_scale, in_proj_qkv_zero) ||
        !pair_ok(in_proj_z_scale, in_proj_z_zero) ||
        !pair_ok(out_proj_scale, out_proj_zero)) {
        return 125;
    }
    const bool any_int4 =
        (in_proj_qkv_scale != nullptr) ||
        (in_proj_z_scale != nullptr) ||
        (out_proj_scale != nullptr);
    if (any_int4 && int4_group_size <= 0) return 126;
    if (!any_int4 && int4_group_size != 0) return 127;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    if (hipMemsetAsync(counters, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + INT4 weights on at least one of the big
    // matmuls (qkv/z/out_proj) + dim divisibility (hidden % 16 == 0,
    // int4_group_size % 16 == 0). Sub-pools handle short rhs_row ranges
    // via per-lane `in_range` checks so non-16-aligned qkv_dim / val_dim
    // / hidden output dims still work; the only hard requirement is that
    // the K-chunk size (16) divides hidden and the quant group_size.
    // 35B-A3B (hidden=2048, group_size=128) satisfies both.
    const bool any_int4_routed_lin =
        (in_proj_qkv_scale != nullptr) ||
        (in_proj_z_scale   != nullptr) ||
        (out_proj_scale    != nullptr);
    const bool wmma_dims_ok_lin =
        (hidden % 16 == 0) &&
        (int4_group_size > 0) &&
        (int4_group_size % 16 == 0);
    const bool use_wmma_lin =
        any_int4_routed_lin &&
        wmma_dims_ok_lin &&
        device_supports_wmma_bf16(static_cast<int>(device_ordinal));

    if (use_wmma_lin) {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_linear_step_kernel<hip_bfloat16, true>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage,
            hidden, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_kernel_dim, rms_norm_eps,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(in_proj_qkv_w),
            static_cast<const hip_bfloat16*>(in_proj_z_w),
            static_cast<const hip_bfloat16*>(in_proj_a_w),
            static_cast<const hip_bfloat16*>(in_proj_b_w),
            static_cast<const hip_bfloat16*>(conv1d_w),
            static_cast<const hip_bfloat16*>(conv1d_bias),
            static_cast<const hip_bfloat16*>(dt_bias),
            static_cast<const hip_bfloat16*>(a_log),
            static_cast<const hip_bfloat16*>(norm_w),
            static_cast<const hip_bfloat16*>(out_proj_w),
            static_cast<hip_bfloat16*>(conv_state),
            recurrent_state,
            int4_group_size,
            static_cast<const hip_bfloat16*>(in_proj_qkv_scale),
            static_cast<const hip_bfloat16*>(in_proj_qkv_zero),
            static_cast<const hip_bfloat16*>(in_proj_z_scale),
            static_cast<const hip_bfloat16*>(in_proj_z_zero),
            static_cast<const hip_bfloat16*>(out_proj_scale),
            static_cast<const hip_bfloat16*>(out_proj_zero),
            static_cast<hip_bfloat16*>(output),
            workspace, counters, barrier_counter, barrier_flag);
    } else {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_linear_step_kernel<hip_bfloat16, false>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage,
            hidden, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_kernel_dim, rms_norm_eps,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(in_proj_qkv_w),
            static_cast<const hip_bfloat16*>(in_proj_z_w),
            static_cast<const hip_bfloat16*>(in_proj_a_w),
            static_cast<const hip_bfloat16*>(in_proj_b_w),
            static_cast<const hip_bfloat16*>(conv1d_w),
            static_cast<const hip_bfloat16*>(conv1d_bias),
            static_cast<const hip_bfloat16*>(dt_bias),
            static_cast<const hip_bfloat16*>(a_log),
            static_cast<const hip_bfloat16*>(norm_w),
            static_cast<const hip_bfloat16*>(out_proj_w),
            static_cast<hip_bfloat16*>(conv_state),
            recurrent_state,
            int4_group_size,
            static_cast<const hip_bfloat16*>(in_proj_qkv_scale),
            static_cast<const hip_bfloat16*>(in_proj_qkv_zero),
            static_cast<const hip_bfloat16*>(in_proj_z_scale),
            static_cast<const hip_bfloat16*>(in_proj_z_zero),
            static_cast<const hip_bfloat16*>(out_proj_scale),
            static_cast<const hip_bfloat16*>(out_proj_zero),
            static_cast<hip_bfloat16*>(output),
            workspace, counters, barrier_counter, barrier_flag);
    }

    // Async dispatch: see attn_step_launch above for the rationale (default
    // stream serializes; chain-end D2H is the implicit barrier).
    hipError_t launch_err_lin = hipGetLastError();
    if (launch_err_lin != hipSuccess) return 254;
    return 0;
}

// PR 4b4 staged MoE FFN parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" int qwen36_moe_hip_ffn_step_launch(
    int           dtype,
    size_t        device_ordinal,
    int           stage,
    int           hidden,
    int           num_experts,
    int           moe_intermediate,
    int           shared_intermediate,
    int           top_k,
    float         rms_norm_eps,
    const void*   input_hidden,
    const void*   post_attn_norm_w,
    const void*   gate_w,
    const void*   gate_up_proj_w,
    const void*   down_proj_w,
    const void*   shared_gate_proj_w,
    const void*   shared_up_proj_w,
    const void*   shared_down_proj_w,
    const void*   shared_expert_gate_w,
    int           int4_group_size,
    const void*   gate_up_proj_scale,
    const void*   gate_up_proj_zero,
    const void*   down_proj_scale,
    const void*   down_proj_zero,
    const void*   shared_gate_proj_scale,
    const void*   shared_gate_proj_zero,
    const void*   shared_up_proj_scale,
    const void*   shared_up_proj_zero,
    const void*   shared_down_proj_scale,
    const void*   shared_down_proj_zero,
    void*         output,
    int*          output_idx,
    float*        workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag) {
    if (dtype != 2) return 130;            // only bf16 supported
    if (stage < 1 || stage > 5) return 131;
    if (hidden <= 0 || num_experts <= 0 || moe_intermediate <= 0 ||
        shared_intermediate <= 0 || top_k <= 0 || top_k > num_experts) {
        return 132;
    }
    // The concurrent-experts FFN dispatch (qwen36_moe_ffn_step_kernel)
    // uses 2*top_k counter slots, and the host-side sync_buf reserves 16
    // u32 slots before barrier_counter at +64. Pushing past slot 15 (i.e.
    // top_k > 8) would clobber barrier state and likely hang. The safe
    // wrapper in `kernel-ffi/src/qwen36_moe.rs::ffn_step_launch` enforces
    // the same cap; this is the bridge-side belt-and-braces.
    if (top_k > 8) return 138;
    if (input_hidden == nullptr || post_attn_norm_w == nullptr ||
        gate_w == nullptr || output == nullptr || output_idx == nullptr ||
        workspace == nullptr || counters == nullptr ||
        barrier_counter == nullptr || barrier_flag == nullptr) {
        return 133;
    }
    // INT4 mode: group_size must divide the relevant dims and each scale
    // must be paired with a zero. group_size==0 disables INT4 entirely.
    if (int4_group_size < 0) return 134;
    auto pair_ok = [](const void* s, const void* z) -> bool {
        return (s == nullptr) == (z == nullptr);
    };
    if (!pair_ok(gate_up_proj_scale, gate_up_proj_zero) ||
        !pair_ok(down_proj_scale, down_proj_zero) ||
        !pair_ok(shared_gate_proj_scale, shared_gate_proj_zero) ||
        !pair_ok(shared_up_proj_scale, shared_up_proj_zero) ||
        !pair_ok(shared_down_proj_scale, shared_down_proj_zero)) {
        return 135;
    }
    const bool any_int4 =
        (gate_up_proj_scale != nullptr) || (down_proj_scale != nullptr) ||
        (shared_gate_proj_scale != nullptr) ||
        (shared_up_proj_scale != nullptr) ||
        (shared_down_proj_scale != nullptr);
    if (any_int4 && int4_group_size <= 0) return 136;
    if (!any_int4 && int4_group_size != 0) return 137;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    // Zero the 2*top_k work-stealing counter slots used by the concurrent
    // per-expert G/I phases. The engine's `reset_sync_buf` covers the full
    // 96-byte buffer (counters + barrier counter + flag); this paranoid
    // memset just guards single-launch callers (parity tests) that allocate
    // sync_buf via `GpuBuffer::zeros` (already zero) and would only fail if
    // someone reused a sync_buf without resetting.
    if (hipMemsetAsync(counters, 0, 2 * top_k * sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + dim divisibility:
    //   - hidden % 16 == 0 (Phase G K-chunk + Phase I output rows)
    //   - moe_intermediate % 16 == 0 (Phase G output rows ÷ 2 + Phase I K-chunk)
    //   - int4_group_size % 16 == 0  (one scale per 16-element K-chunk)
    //   - INT4 routed weights present (gate_up_proj_scale / down_proj_scale)
    // 35B-A3B (hidden=2048, I=512, group_size=128) satisfies all of these;
    // synthetic fixtures use 16-divisible dims too. The shared expert path
    // (Phase D/F) stays scalar in both variants — Phase 2 of the roadmap.
    const bool routed_int4 =
        (gate_up_proj_scale != nullptr) && (down_proj_scale != nullptr);
    const bool wmma_dims_ok =
        (hidden % 16 == 0) &&
        (moe_intermediate % 16 == 0) &&
        (int4_group_size > 0) &&
        (int4_group_size % 16 == 0);
    const bool use_wmma =
        routed_int4 &&
        wmma_dims_ok &&
        device_supports_wmma_bf16(static_cast<int>(device_ordinal));

    if (use_wmma) {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_ffn_step_kernel<hip_bfloat16, true>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage,
            hidden, num_experts, moe_intermediate, shared_intermediate, top_k,
            rms_norm_eps,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(post_attn_norm_w),
            static_cast<const hip_bfloat16*>(gate_w),
            static_cast<const hip_bfloat16*>(gate_up_proj_w),
            static_cast<const hip_bfloat16*>(down_proj_w),
            static_cast<const hip_bfloat16*>(shared_gate_proj_w),
            static_cast<const hip_bfloat16*>(shared_up_proj_w),
            static_cast<const hip_bfloat16*>(shared_down_proj_w),
            static_cast<const hip_bfloat16*>(shared_expert_gate_w),
            int4_group_size,
            static_cast<const hip_bfloat16*>(gate_up_proj_scale),
            static_cast<const hip_bfloat16*>(gate_up_proj_zero),
            static_cast<const hip_bfloat16*>(down_proj_scale),
            static_cast<const hip_bfloat16*>(down_proj_zero),
            static_cast<const hip_bfloat16*>(shared_gate_proj_scale),
            static_cast<const hip_bfloat16*>(shared_gate_proj_zero),
            static_cast<const hip_bfloat16*>(shared_up_proj_scale),
            static_cast<const hip_bfloat16*>(shared_up_proj_zero),
            static_cast<const hip_bfloat16*>(shared_down_proj_scale),
            static_cast<const hip_bfloat16*>(shared_down_proj_zero),
            static_cast<hip_bfloat16*>(output),
            output_idx,
            workspace, counters, barrier_counter, barrier_flag);
    } else {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_ffn_step_kernel<hip_bfloat16, false>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            stage,
            hidden, num_experts, moe_intermediate, shared_intermediate, top_k,
            rms_norm_eps,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(post_attn_norm_w),
            static_cast<const hip_bfloat16*>(gate_w),
            static_cast<const hip_bfloat16*>(gate_up_proj_w),
            static_cast<const hip_bfloat16*>(down_proj_w),
            static_cast<const hip_bfloat16*>(shared_gate_proj_w),
            static_cast<const hip_bfloat16*>(shared_up_proj_w),
            static_cast<const hip_bfloat16*>(shared_down_proj_w),
            static_cast<const hip_bfloat16*>(shared_expert_gate_w),
            int4_group_size,
            static_cast<const hip_bfloat16*>(gate_up_proj_scale),
            static_cast<const hip_bfloat16*>(gate_up_proj_zero),
            static_cast<const hip_bfloat16*>(down_proj_scale),
            static_cast<const hip_bfloat16*>(down_proj_zero),
            static_cast<const hip_bfloat16*>(shared_gate_proj_scale),
            static_cast<const hip_bfloat16*>(shared_gate_proj_zero),
            static_cast<const hip_bfloat16*>(shared_up_proj_scale),
            static_cast<const hip_bfloat16*>(shared_up_proj_zero),
            static_cast<const hip_bfloat16*>(shared_down_proj_scale),
            static_cast<const hip_bfloat16*>(shared_down_proj_zero),
            static_cast<hip_bfloat16*>(output),
            output_idx,
            workspace, counters, barrier_counter, barrier_flag);
    }

    // Async dispatch: see attn_step_launch above for the rationale.
    hipError_t launch_err_ffn = hipGetLastError();
    if (launch_err_ffn != hipSuccess) return 254;
    return 0;
}

// PR 4b5 step 2: INT4 dequant smoke launcher.
// Drives `qwen36_moe::int4_dequant_smoke_kernel` over a small `[out_rows,
// in_cols]` slab and writes both helpers' outputs to separate buffers.
// The Rust-side test validates byte-for-byte against a host reference.
extern "C" int qwen36_moe_hip_int4_dequant_smoke_launch(
    size_t         device_ordinal,
    const uint8_t* packed,
    const void*    scale,
    const void*    zero,
    int            out_rows,
    int            in_cols,
    int            gsz,
    float*         dq_8_out,
    float*         dq_scalar_out) {
    if (packed == nullptr || scale == nullptr || zero == nullptr ||
        dq_8_out == nullptr || dq_scalar_out == nullptr) {
        return 140;
    }
    if (out_rows <= 0 || in_cols <= 0 || gsz <= 0) return 141;
    if (in_cols % 8 != 0) return 142;
    if (in_cols % gsz != 0 || gsz % 2 != 0) return 143;
    if (out_rows % gsz != 0) return 144;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipLaunchKernelGGL(qwen36_moe::int4_dequant_smoke_kernel,
                       dim3(1), dim3(1), 0, 0,
                       packed,
                       static_cast<const hip_bfloat16*>(scale),
                       static_cast<const hip_bfloat16*>(zero),
                       out_rows, in_cols, gsz,
                       dq_8_out, dq_scalar_out);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

// PR follow-up to #68: GPU-side final RMSNorm + lm_head GEMV launcher.
// Replaces the host-side path in `qwen36_moe_decode::host_final_norm_lm_head_f32`
// which dominated per-token wall-clock at 233 ms / 360 ms total on
// 35B-A3B greedy decode.
//
// Inputs are device pointers (BF16 throughout: final_hidden, final_norm_w,
// lm_head_w; logits is BF16 output). `counter` is a `[1] u32` device buffer
// the kernel uses for work-stealing across vocab rows; this launcher
// memsets it to 0 before launch.
//
// Currently bf16-only (`dtype == 2`). Geometry assumptions:
//   - `hidden % block_size == 0` (block reduction lane scheme assumes it).
//   - `vocab > 0`; the work-stealing loop self-terminates when
//     `my_row >= vocab`.
extern "C" int qwen36_moe_hip_lm_head_launch(
    int           dtype,
    size_t        device_ordinal,
    int           hidden,
    int           vocab,
    float         rms_norm_eps,
    const void*   final_hidden,
    const void*   final_norm_w,
    const void*   lm_head_w,
    void*         logits,
    unsigned int* counter) {
    if (dtype != 2) return 130;            // only bf16 supported
    if (hidden <= 0 || vocab <= 0) return 132;
    if (final_hidden == nullptr || final_norm_w == nullptr ||
        lm_head_w == nullptr || logits == nullptr || counter == nullptr) {
        return 133;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }

    const int ordinal_int = static_cast<int>(device_ordinal);

    // WMMA path: gfx11xx only, BF16 weights, hidden divisible by 16. Drops
    // ~14 ms / token vs the scalar work-stealing path on 35B-A3B (vocab=248k).
    // Falls back to the scalar kernel on non-gfx11xx, on group-size mismatch,
    // or when SUPERSONIC_QWEN4B_DISABLE_WMMA is set.
    if (device_supports_wmma_bf16(ordinal_int) && (hidden % 16 == 0)) {
        // Grid: one wave32 per 16-vocab tile. block_size=32 (one wave).
        // LDS: 32 F32 (RMSNorm reduction) + hidden u16 (BF16 staged x_norm).
        const int wmma_block_size = 32;
        const int grid_x = (vocab + 15) / 16;
        const size_t lds_bytes_wmma =
            static_cast<size_t>(wmma_block_size) * sizeof(float) +
            static_cast<size_t>(hidden) * sizeof(uint16_t);

        hipLaunchKernelGGL(
            qwen36_moe::qwen36_moe_lm_head_wmma_kernel<hip_bfloat16>,
            dim3(static_cast<unsigned int>(grid_x)),
            dim3(static_cast<unsigned int>(wmma_block_size)),
            lds_bytes_wmma, 0,
            static_cast<const hip_bfloat16*>(final_hidden),
            static_cast<const hip_bfloat16*>(final_norm_w),
            static_cast<const hip_bfloat16*>(lm_head_w),
            static_cast<hip_bfloat16*>(logits),
            hidden,
            vocab,
            rms_norm_eps);
        // No counter needed by the WMMA path (one block per tile, no
        // atomic claim). The host-passed counter buffer is ignored here.
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err   = hipDeviceSynchronize();
        if (launch_err != hipSuccess) return 254;
        if (sync_err != hipSuccess) return 255;
        return 0;
    }

    // Scalar fallback path. Requires the work-stealing counter zeroed.
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    if (hipMemsetAsync(counter, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    // shared_scratch [block_size] + x_norm_lds [hidden], both F32.
    const size_t lds_bytes =
        static_cast<size_t>(hidden + block_size) * sizeof(float);

    hipLaunchKernelGGL(qwen36_moe::qwen36_moe_lm_head_kernel<hip_bfloat16>,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_size),
                       lds_bytes, 0,
                       static_cast<const hip_bfloat16*>(final_hidden),
                       static_cast<const hip_bfloat16*>(final_norm_w),
                       static_cast<const hip_bfloat16*>(lm_head_w),
                       static_cast<hip_bfloat16*>(logits),
                       counter,
                       hidden,
                       vocab,
                       rms_norm_eps);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}
