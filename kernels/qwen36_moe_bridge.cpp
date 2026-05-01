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
#include <stdint.h>

namespace {

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
    void*         output,
    float*        workspace,
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

    hipLaunchKernelGGL(qwen36_moe::qwen36_moe_attn_step_kernel<hip_bfloat16>,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_size),
                       lds_bytes, 0,
                       stage,
                       hidden,
                       num_heads,
                       num_kv_heads,
                       head_dim,
                       rotary_dim,
                       rope_theta,
                       rms_norm_eps,
                       position,
                       static_cast<const hip_bfloat16*>(input_hidden),
                       static_cast<const hip_bfloat16*>(input_norm_w),
                       static_cast<const hip_bfloat16*>(q_proj_w),
                       static_cast<const hip_bfloat16*>(k_proj_w),
                       static_cast<const hip_bfloat16*>(v_proj_w),
                       static_cast<const hip_bfloat16*>(q_norm_w),
                       static_cast<const hip_bfloat16*>(k_norm_w),
                       static_cast<const hip_bfloat16*>(o_proj_w),
                       static_cast<hip_bfloat16*>(output),
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

    hipLaunchKernelGGL(qwen36_moe::qwen36_moe_linear_step_kernel<hip_bfloat16>,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_size),
                       lds_bytes, 0,
                       stage,
                       hidden,
                       num_k_heads,
                       num_v_heads,
                       head_k_dim,
                       head_v_dim,
                       conv_kernel_dim,
                       rms_norm_eps,
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
                       static_cast<hip_bfloat16*>(output),
                       workspace,
                       counters,
                       barrier_counter,
                       barrier_flag);
    hipError_t launch_err_lin = hipGetLastError();
    hipError_t sync_err_lin   = hipDeviceSynchronize();
    if (launch_err_lin != hipSuccess) return 254;
    if (sync_err_lin != hipSuccess) return 255;
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
    if (input_hidden == nullptr || post_attn_norm_w == nullptr ||
        gate_w == nullptr || output == nullptr || output_idx == nullptr ||
        workspace == nullptr || counters == nullptr ||
        barrier_counter == nullptr || barrier_flag == nullptr) {
        return 133;
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

    if (hipMemsetAsync(counters, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    hipLaunchKernelGGL(qwen36_moe::qwen36_moe_ffn_step_kernel<hip_bfloat16>,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_size),
                       lds_bytes, 0,
                       stage,
                       hidden,
                       num_experts,
                       moe_intermediate,
                       shared_intermediate,
                       top_k,
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
                       static_cast<hip_bfloat16*>(output),
                       output_idx,
                       workspace,
                       counters,
                       barrier_counter,
                       barrier_flag);
    hipError_t launch_err_ffn = hipGetLastError();
    hipError_t sync_err_ffn   = hipDeviceSynchronize();
    if (launch_err_ffn != hipSuccess) return 254;
    if (sync_err_ffn != hipSuccess) return 255;
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
