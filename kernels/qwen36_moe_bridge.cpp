// Qwen3.6-MoE HIP launch bridge — PR 4 stub.
//
// Calls into the descriptor-walk stub kernel in qwen36_moe.hip. The
// `qwen36_moe_hip_stub_launch` extern is the only symbol exposed today;
// once the real megakernel lands, this file gains a
// `qwen36_moe_hip_persistent_decode` entry alongside the stub (kept for
// the smoke test).

#include "qwen36_moe.hip"

// Phase 3e: persistent decode megakernel template + launcher. Lives in its
// own file under `qwen36_moe_persistent/` per the design doc (#117).
// Includes the phase headers and references qwen36_moe::{DecodeLayerDesc,
// Int4ScaleDesc} from qwen36_moe.hip above.
#include "qwen36_moe_persistent/persistent_decode.hip"

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <limits>
#ifndef SUPERSONIC_QWEN36_CUDA_BRIDGE
#include <hip/hip_runtime.h>
#endif
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
#ifdef SUPERSONIC_QWEN36_CUDA_BRIDGE
    (void)device_ordinal;
    return false;
#else
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
#endif
}

bool ffn_step_supports_wmma_bf16(int device_ordinal) {
    const char* enabled = std::getenv("SUPERSONIC_QWEN36_ENABLE_FFN_STEP_WMMA");
    return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0' &&
           device_supports_wmma_bf16(device_ordinal);
}

bool sync_each_kernel_enabled() {
    const char* env = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

uint64_t backend_failure(int project_status, hipError_t native_status) {
    return static_cast<uint32_t>(project_status) |
           (static_cast<uint64_t>(static_cast<uint32_t>(native_status)) << 32);
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
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) == 344,
              "Qwen36MoeDecodeLayerDesc size drift — Rust side is pinned to 344 bytes; "
              "if you appended a field, update both sides in the same commit");

using Qwen36Int4WeightDesc = qwen36_moe::Qwen36MoeInt4WeightDesc;
using Qwen36Int4ScaleDesc = qwen36_moe::Int4ScaleDesc;

static_assert(sizeof(Qwen36Int4WeightDesc) == 64);
static_assert(alignof(Qwen36Int4WeightDesc) == 8);
static_assert(offsetof(Qwen36Int4WeightDesc, scale) == 0);
static_assert(offsetof(Qwen36Int4WeightDesc, zero) == 8);
static_assert(offsetof(Qwen36Int4WeightDesc, packed_row_stride_bytes) == 16);
static_assert(offsetof(Qwen36Int4WeightDesc, packed_expert_stride_bytes) == 24);
static_assert(offsetof(Qwen36Int4WeightDesc, scale_row_stride_elements) == 32);
static_assert(offsetof(Qwen36Int4WeightDesc, scale_expert_stride_elements) == 40);
static_assert(offsetof(Qwen36Int4WeightDesc, input_group_size) == 48);
static_assert(offsetof(Qwen36Int4WeightDesc, output_group_size) == 52);
static_assert(offsetof(Qwen36Int4WeightDesc, implicit_zero_code) == 56);
static_assert(offsetof(Qwen36Int4WeightDesc, encoding) == 60);

static_assert(sizeof(Qwen36Int4ScaleDesc) == 768);
static_assert(alignof(Qwen36Int4ScaleDesc) == 8);
static_assert(offsetof(Qwen36Int4ScaleDesc, q_proj) == 0);
static_assert(offsetof(Qwen36Int4ScaleDesc, k_proj) == 64);
static_assert(offsetof(Qwen36Int4ScaleDesc, v_proj) == 128);
static_assert(offsetof(Qwen36Int4ScaleDesc, o_proj) == 192);
static_assert(offsetof(Qwen36Int4ScaleDesc, linear_in_proj_qkv) == 256);
static_assert(offsetof(Qwen36Int4ScaleDesc, linear_in_proj_z) == 320);
static_assert(offsetof(Qwen36Int4ScaleDesc, linear_out_proj) == 384);
static_assert(offsetof(Qwen36Int4ScaleDesc, experts_gate_up) == 448);
static_assert(offsetof(Qwen36Int4ScaleDesc, experts_down) == 512);
static_assert(offsetof(Qwen36Int4ScaleDesc, shared_expert_gate_proj) == 576);
static_assert(offsetof(Qwen36Int4ScaleDesc, shared_expert_up_proj) == 640);
static_assert(offsetof(Qwen36Int4ScaleDesc, shared_expert_down_proj) == 704);

static_assert(sizeof(qwen36_moe::KVCacheFp8Desc) == 16,
              "Qwen36MoeKVCacheFp8Desc layout drift — must be exactly 2 pointers");

bool checked_strided_extent(
    uint64_t rows,
    uint64_t row_stride,
    uint64_t logical_row_elements,
    uint64_t* extent
) {
    const uint64_t preceding_rows = rows - 1;
    if (preceding_rows != 0 &&
        row_stride >
            (std::numeric_limits<uint64_t>::max() - logical_row_elements) /
                preceding_rows) {
        return false;
    }
    *extent = preceding_rows * row_stride + logical_row_elements;
    return true;
}

int validate_int4_descriptor_geometry(
    const Qwen36Int4WeightDesc& desc,
    int experts,
    int out_rows,
    int in_cols
) {
    if (desc.scale == nullptr || experts <= 0 || out_rows <= 0 || in_cols <= 0) {
        return 171;
    }
    if (desc.input_group_size <= 0 || desc.output_group_size <= 0 ||
        in_cols % desc.input_group_size != 0 ||
        out_rows % desc.output_group_size != 0) {
        return 172;
    }
    if (desc.encoding == 1) {
        if (desc.zero == nullptr || desc.implicit_zero_code >= 0 ||
            desc.input_group_size != 128 || desc.output_group_size != 128) {
            return 173;
        }
    } else if (desc.encoding == 2) {
        if (desc.zero != nullptr || desc.input_group_size != 32 ||
            desc.output_group_size != 1 || desc.implicit_zero_code != 8) {
            return 174;
        }
    } else {
        return 175;
    }

    const uint64_t packed_row_elements = static_cast<uint64_t>(in_cols / 2);
    const uint64_t scale_row_elements =
        static_cast<uint64_t>(in_cols / desc.input_group_size);
    if (desc.packed_row_stride_bytes < packed_row_elements ||
        desc.scale_row_stride_elements < scale_row_elements) {
        return 176;
    }
    const uint64_t scale_rows =
        static_cast<uint64_t>(out_rows / desc.output_group_size);
    uint64_t packed_expert_elements = 0;
    uint64_t scale_expert_elements = 0;
    if (!checked_strided_extent(
            static_cast<uint64_t>(out_rows),
            desc.packed_row_stride_bytes,
            packed_row_elements,
            &packed_expert_elements) ||
        !checked_strided_extent(
            scale_rows,
            desc.scale_row_stride_elements,
            scale_row_elements,
            &scale_expert_elements)) {
        return 179;
    }
    if (experts > 1 &&
        (desc.packed_expert_stride_bytes < packed_expert_elements ||
         desc.scale_expert_stride_elements < scale_expert_elements)) {
        return 177;
    }
    return 0;
}

bool is_int4_execution_desc(const Qwen36Int4WeightDesc& desc) {
    return desc.encoding == 1 || desc.encoding == 2;
}

int validate_execution_descriptor(
    const Qwen36Int4WeightDesc& desc,
    int experts,
    int out_rows,
    int in_cols
) {
    if (desc.encoding == 0) {
        return (desc.scale == nullptr && desc.zero == nullptr) ? 0 : 181;
    }
    if (is_int4_execution_desc(desc)) {
        return validate_int4_descriptor_geometry(desc, experts, out_rows, in_cols);
    }
    if (desc.encoding != 3 || desc.scale == nullptr || desc.zero != nullptr ||
        desc.input_group_size <= 0 || desc.output_group_size <= 0 ||
        in_cols % desc.input_group_size != 0 ||
        out_rows % desc.output_group_size != 0) {
        return 182;
    }
    return 0;
}

} // namespace

extern "C" uint64_t supersonic_qwen36_encode_bridge_status(
    int project_status,
    int native_status
) {
    return native_status == 0
        ? static_cast<uint32_t>(project_status)
        : backend_failure(project_status, static_cast<hipError_t>(native_status));
}

// `dtype` encoding follows the Qwen/Gemma/Phi bridges: 0 = half, 2 = bf16.
// The stub ignores dtype because it does no math; the real kernel will
// branch on it.
extern "C" uint64_t qwen36_moe_hip_stub_launch(
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
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 64; // wave-sized; descriptor walk is light

    // Zero the cooperative counter before launch. The kernel uses
    // `atomicAdd` to claim layer indices.
    hipError_t memset_err = hipMemsetAsync(counters, 0, sizeof(unsigned int));
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
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
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// PR 4b2 staged-attention parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" uint64_t qwen36_moe_hip_attn_step_launch(
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
    int           cache_pos,    // -1 ⇒ inherit from `position` (base-model
                                //      decode); ≥0 ⇒ MTP-style decoupled
                                //      KV-cache slot (see kernel comment).
    const void*   input_hidden,
    const void*   input_norm_w,
    const void*   q_proj_w,
    const void*   k_proj_w,
    const void*   v_proj_w,
    const void*   q_norm_w,
    const void*   k_norm_w,
    const void*   o_proj_w,
    const Qwen36Int4ScaleDesc* int4_desc,
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
    const Qwen36Int4ScaleDesc quant =
        int4_desc != nullptr ? *int4_desc : Qwen36Int4ScaleDesc{};
    const int q_rows = 2 * num_heads * head_dim;
    const int kv_rows = num_kv_heads * head_dim;
    const int o_cols = num_heads * head_dim;
    if (validate_execution_descriptor(quant.q_proj, 1, q_rows, hidden) != 0 ||
        validate_execution_descriptor(quant.k_proj, 1, kv_rows, hidden) != 0 ||
        validate_execution_descriptor(quant.v_proj, 1, kv_rows, hidden) != 0 ||
        validate_execution_descriptor(quant.o_proj, 1, hidden, o_cols) != 0) {
        return 115;
    }

    // KV cache: pointers must be paired (both null or both non-null), and
    // kv_max_t must be positive when enabled + the *effective* slot
    // (cache_pos ≥ 0 ? cache_pos : position) must fit.
    const bool kv_enabled = (kv_cache_k != nullptr || kv_cache_v != nullptr);
    if ((kv_cache_k == nullptr) != (kv_cache_v == nullptr)) return 118;
    if (kv_enabled) {
        if (kv_max_t <= 0) return 119;
        const int eff_slot = (cache_pos >= 0) ? cache_pos : position;
        if (eff_slot < 0 || eff_slot >= kv_max_t) return 120;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    // Zero the cooperative counter + barrier state before launch. The kernel
    // expects all three to start at 0; sync_buf is documented as 32 zero
    // bytes by the Rust-side wrapper but a defence-in-depth memset here
    // keeps a misuse from corrupting the launch.
    hipError_t memset_err = hipMemsetAsync(counters, 0, sizeof(unsigned int));
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + INT4 weights on at least one matmul +
    // dim divisibility (hidden % 16 == 0 and group_size % 16 == 0). The
    // K-chunk is 16; per-lane `in_range` checks handle non-16-aligned
    // output dims (`q_out_dim = 2*H*d`, `Hkv*d`, `hidden`).
    const bool any_int4_attn =
        is_int4_execution_desc(quant.q_proj) ||
        is_int4_execution_desc(quant.k_proj) ||
        is_int4_execution_desc(quant.v_proj) ||
        is_int4_execution_desc(quant.o_proj);
    auto wmma_group_ok = [](const Qwen36Int4WeightDesc& desc) {
        return !is_int4_execution_desc(desc) ||
            desc.input_group_size % 16 == 0;
    };
    const bool wmma_dims_ok_attn =
        (hidden % 16 == 0) && (o_cols % 16 == 0) &&
        wmma_group_ok(quant.q_proj) && wmma_group_ok(quant.k_proj) &&
        wmma_group_ok(quant.v_proj) && wmma_group_ok(quant.o_proj);
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
            rope_theta, rms_norm_eps, position, cache_pos,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(q_proj_w),
            static_cast<const hip_bfloat16*>(k_proj_w),
            static_cast<const hip_bfloat16*>(v_proj_w),
            static_cast<const hip_bfloat16*>(q_norm_w),
            static_cast<const hip_bfloat16*>(k_norm_w),
            static_cast<const hip_bfloat16*>(o_proj_w),
            quant.q_proj, quant.k_proj, quant.v_proj, quant.o_proj,
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
            rope_theta, rms_norm_eps, position, cache_pos,
            static_cast<const hip_bfloat16*>(input_hidden),
            static_cast<const hip_bfloat16*>(input_norm_w),
            static_cast<const hip_bfloat16*>(q_proj_w),
            static_cast<const hip_bfloat16*>(k_proj_w),
            static_cast<const hip_bfloat16*>(v_proj_w),
            static_cast<const hip_bfloat16*>(q_norm_w),
            static_cast<const hip_bfloat16*>(k_norm_w),
            static_cast<const hip_bfloat16*>(o_proj_w),
            quant.q_proj, quant.k_proj, quant.v_proj, quant.o_proj,
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
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    return 0;
}

// PR 4b3 staged linear-attention parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" uint64_t qwen36_moe_hip_linear_step_launch(
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
    const Qwen36Int4ScaleDesc* int4_desc,
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
    const Qwen36Int4ScaleDesc quant =
        int4_desc != nullptr ? *int4_desc : Qwen36Int4ScaleDesc{};
    const int key_dim = num_k_heads * head_k_dim;
    const int value_dim = num_v_heads * head_v_dim;
    const int qkv_dim = 2 * key_dim + value_dim;
    if (validate_execution_descriptor(
            quant.linear_in_proj_qkv, 1, qkv_dim, hidden) != 0 ||
        validate_execution_descriptor(
            quant.linear_in_proj_z, 1, value_dim, hidden) != 0 ||
        validate_execution_descriptor(
            quant.linear_out_proj, 1, hidden, value_dim) != 0) {
        return 125;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    hipError_t memset_err = hipMemsetAsync(counters, 0, sizeof(unsigned int));
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + INT4 weights on at least one of the big
    // matmuls (qkv/z/out_proj) + dim divisibility (hidden % 16 == 0,
    // int4_group_size % 16 == 0). Sub-pools handle short rhs_row ranges
    // via per-lane `in_range` checks so non-16-aligned qkv_dim / val_dim
    // / hidden output dims still work; the only hard requirement is that
    // the K-chunk size (16) divides hidden and the quant group_size.
    // 35B-A3B (hidden=2048, group_size=32 or 128) satisfies both.
    const bool any_int4_routed_lin =
        is_int4_execution_desc(quant.linear_in_proj_qkv) ||
        is_int4_execution_desc(quant.linear_in_proj_z) ||
        is_int4_execution_desc(quant.linear_out_proj);
    const bool wmma_dims_ok_lin =
        (hidden % 16 == 0) &&
        (value_dim % 16 == 0) &&
        (!is_int4_execution_desc(quant.linear_in_proj_qkv) ||
         quant.linear_in_proj_qkv.input_group_size % 16 == 0) &&
        (!is_int4_execution_desc(quant.linear_in_proj_z) ||
         quant.linear_in_proj_z.input_group_size % 16 == 0) &&
        (!is_int4_execution_desc(quant.linear_out_proj) ||
         quant.linear_out_proj.input_group_size % 16 == 0);
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
            quant.linear_in_proj_qkv,
            quant.linear_in_proj_z,
            quant.linear_out_proj,
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
            quant.linear_in_proj_qkv,
            quant.linear_in_proj_z,
            quant.linear_out_proj,
            static_cast<hip_bfloat16*>(output),
            workspace, counters, barrier_counter, barrier_flag);
    }

    // Async dispatch: see attn_step_launch above for the rationale (default
    // stream serializes; chain-end D2H is the implicit barrier).
    hipError_t launch_err_lin = hipGetLastError();
    if (launch_err_lin != hipSuccess) return backend_failure(254, launch_err_lin);
    return 0;
}

// PR 4b4 staged MoE FFN parity launcher.
// `dtype` follows the project convention: 2 = bf16. Other values are
// rejected so the matching kernel template is unambiguous.
extern "C" uint64_t qwen36_moe_hip_ffn_step_launch(
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
    const Qwen36Int4ScaleDesc* int4_desc,
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
    const Qwen36Int4ScaleDesc quant =
        int4_desc != nullptr ? *int4_desc : Qwen36Int4ScaleDesc{};
    if (validate_execution_descriptor(
            quant.experts_gate_up, num_experts,
            2 * moe_intermediate, hidden) != 0 ||
        validate_execution_descriptor(
            quant.experts_down, num_experts,
            hidden, moe_intermediate) != 0 ||
        validate_execution_descriptor(
            quant.shared_expert_gate_proj, 1,
            shared_intermediate, hidden) != 0 ||
        validate_execution_descriptor(
            quant.shared_expert_up_proj, 1,
            shared_intermediate, hidden) != 0 ||
        validate_execution_descriptor(
            quant.shared_expert_down_proj, 1,
            hidden, shared_intermediate) != 0) {
        return 135;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
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
    hipError_t memset_err =
        hipMemsetAsync(counters, 0, 2 * top_k * sizeof(unsigned int));
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
    }

    const size_t lds_bytes = static_cast<size_t>(hidden + block_size) * sizeof(float);

    // WMMA path requires gfx11xx + dim divisibility:
    //   - hidden % 16 == 0 (Phase G K-chunk + Phase I output rows)
    //   - moe_intermediate % 16 == 0 (Phase G output rows ÷ 2 + Phase I K-chunk)
    //   - int4_group_size % 16 == 0  (one scale per 16-element K-chunk)
    //   - INT4 routed weights present (gate_up_proj_scale / down_proj_scale)
    // 35B-A3B (hidden=2048, I=512, group_size=32 or 128) satisfies all of these;
    // synthetic fixtures use 16-divisible dims too. The shared expert path
    // (Phase D/F) stays scalar in both variants — Phase 2 of the roadmap.
    const bool routed_int4 =
        is_int4_execution_desc(quant.experts_gate_up) &&
        is_int4_execution_desc(quant.experts_down);
    const bool wmma_dims_ok =
        (hidden % 16 == 0) &&
        (moe_intermediate % 16 == 0) &&
        (quant.experts_gate_up.input_group_size % 16 == 0) &&
        (quant.experts_down.input_group_size % 16 == 0);
    const bool use_wmma =
        routed_int4 &&
        wmma_dims_ok &&
        ffn_step_supports_wmma_bf16(static_cast<int>(device_ordinal));

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
            quant.experts_gate_up,
            quant.experts_down,
            quant.shared_expert_gate_proj,
            quant.shared_expert_up_proj,
            quant.shared_expert_down_proj,
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
            quant.experts_gate_up,
            quant.experts_down,
            quant.shared_expert_gate_proj,
            quant.shared_expert_up_proj,
            quant.shared_expert_down_proj,
            static_cast<hip_bfloat16*>(output),
            output_idx,
            workspace, counters, barrier_counter, barrier_flag);
    }

    // Async dispatch: see attn_step_launch above for the rationale.
    hipError_t launch_err_ffn = hipGetLastError();
    if (launch_err_ffn != hipSuccess) return backend_failure(254, launch_err_ffn);
    return 0;
}

// PR 4b5 step 2: INT4 dequant smoke launcher.
// Drives `qwen36_moe::int4_dequant_smoke_kernel` over a small `[out_rows,
// in_cols]` slab and writes both helpers' outputs to separate buffers.
// The Rust-side test validates byte-for-byte against a host reference.
extern "C" uint64_t qwen36_moe_hip_int4_dequant_smoke_launch(
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
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

extern "C" uint64_t qwen36_moe_hip_int4_descriptor_dequant_smoke_launch(
    size_t                         device_ordinal,
    const uint8_t*                 packed,
    const Qwen36Int4WeightDesc*    desc,
    int                            experts,
    int                            out_rows,
    int                            in_cols,
    float*                         dq_8_out,
    float*                         dq_scalar_out) {
    if (packed == nullptr || desc == nullptr || desc->scale == nullptr ||
        dq_8_out == nullptr || dq_scalar_out == nullptr) {
        return 170;
    }
    const int descriptor_status =
        validate_int4_descriptor_geometry(*desc, experts, out_rows, in_cols);
    if (descriptor_status != 0) return descriptor_status;
    if (in_cols % 8 != 0) return 178;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    constexpr int block_size = 256;
    if (static_cast<size_t>(experts) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(out_rows)) {
        return 179;
    }
    const size_t logical_rows = static_cast<size_t>(experts) * out_rows;
    const size_t spans_per_row = static_cast<size_t>(in_cols / 8);
    if (logical_rows > std::numeric_limits<size_t>::max() / spans_per_row) {
        return 179;
    }
    const size_t span_count = logical_rows * spans_per_row;
    const size_t requested_blocks = (span_count - 1) / block_size + 1;
    const unsigned int blocks = requested_blocks > 65535
        ? 65535
        : static_cast<unsigned int>(requested_blocks);
    hipLaunchKernelGGL(qwen36_moe::int4_descriptor_dequant_smoke_kernel,
                       dim3(blocks), dim3(block_size), 0, 0,
                       packed, *desc, experts, out_rows, in_cols,
                       dq_8_out, dq_scalar_out);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

extern "C" uint64_t qwen36_moe_hip_int4_descriptor_wmma_parity_launch(
    size_t                         device_ordinal,
    const uint8_t*                 packed,
    const Qwen36Int4WeightDesc*    desc,
    const void*                    activation,
    int                            out_rows,
    int                            in_cols,
    float*                         scalar_out,
    float*                         wmma_out) {
    if (packed == nullptr || desc == nullptr || desc->scale == nullptr ||
        activation == nullptr || scalar_out == nullptr || wmma_out == nullptr) {
        return 180;
    }
    if (out_rows != 32 || in_cols != 128) return 181;
    const int descriptor_status =
        validate_int4_descriptor_geometry(*desc, 1, out_rows, in_cols);
    if (descriptor_status != 0) return descriptor_status;
    if (!device_supports_wmma_bf16(static_cast<int>(device_ordinal))) return 182;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    hipLaunchKernelGGL(qwen36_moe::int4_descriptor_wmma_parity_kernel,
                       dim3(1), dim3(32), 0, 0,
                       packed, *desc,
                       static_cast<const hip_bfloat16*>(activation),
                       out_rows, in_cols, scalar_out, wmma_out);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
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
extern "C" uint64_t qwen36_moe_hip_lm_head_launch(
    int           dtype,
    size_t        device_ordinal,
    int           hidden,
    int           vocab,
    float         rms_norm_eps,
    const void*   final_hidden,
    const void*   final_norm_w,
    const void*   lm_head_w,
    void*         logits,
    // Phase 6.2c.3 — optional capture of the BF16-rounded post-RMSNorm
    // hidden state. Used by the MTP draft loop to feed `h_post` into
    // the next step's `h_base`. Pass nullptr (the base-decode path) to
    // skip the export.
    void*         x_normed_out,
    unsigned int* counter) {
    if (dtype != 2) return 130;            // only bf16 supported
    if (hidden <= 0 || vocab <= 0) return 132;
    if (final_hidden == nullptr || final_norm_w == nullptr ||
        lm_head_w == nullptr || logits == nullptr || counter == nullptr) {
        return 133;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
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
            static_cast<hip_bfloat16*>(x_normed_out),
            hidden,
            vocab,
            rms_norm_eps);
        // No counter needed by the WMMA path (one block per tile, no
        // atomic claim). The host-passed counter buffer is ignored here.
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err =
            sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
        if (launch_err != hipSuccess) return backend_failure(254, launch_err);
        if (sync_err != hipSuccess) return backend_failure(255, sync_err);
        return 0;
    }

    // Scalar fallback path. Requires the work-stealing counter zeroed.
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    hipError_t memset_err = hipMemsetAsync(counter, 0, sizeof(unsigned int));
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
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
                       static_cast<hip_bfloat16*>(x_normed_out),
                       counter,
                       hidden,
                       vocab,
                       rms_norm_eps);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// Phase 6.4a: batched lm_head launcher (M = K input rows in one call).
//
// Wraps `qwen36_moe_lm_head_batched_wmma_kernel`. Mirrors the single-M
// WMMA path in `qwen36_moe_hip_lm_head_launch` above, except `m` is a
// runtime parameter (1..16) and the kernel processes M input rows
// in a single WMMA tile per vocab block. WMMA-only (gfx11xx + bf16);
// callers that need a fallback should call the single-M launch in a
// loop. Status codes match the single-M launcher.
extern "C" uint64_t qwen36_moe_hip_lm_head_batched_launch(
    int           dtype,
    size_t        device_ordinal,
    int           m,                     // 1..=16
    int           hidden,
    int           vocab,
    float         rms_norm_eps,
    const void*   final_hidden,          // [m, hidden] BF16
    const void*   final_norm_w,          // [hidden]    BF16
    const void*   lm_head_w,             // [vocab, hidden] BF16
    void*         logits,                // [m, vocab]  BF16
    void*         x_normed_out) {        // [m, hidden] BF16, nullable
    if (dtype != 2) return 130;            // bf16 only
    if (hidden <= 0 || vocab <= 0) return 132;
    // M must be in 1..16 (WMMA tile bound) AND fit the per-block dynamic
    // LDS budget. The API-level 16 ceiling is necessary but not sufficient
    // — at hidden=2048 the LDS staging per row is 4 KiB BF16, plus 128 B
    // reduction scratch, so M=16 would request 65,664 B and overflow the
    // 64 KiB per-block cap. Compute the hidden-dependent ceiling and
    // reject inputs that exceed it before the kernel launch crashes
    // with status 254.
    if (m < 1) return 134;
    constexpr size_t LDS_BUDGET_BYTES = 64 * 1024;
    constexpr size_t REDUCTION_BYTES = 32 * sizeof(float); // 128 B
    const size_t lds_per_row = static_cast<size_t>(hidden) * sizeof(uint16_t);
    const size_t max_m_for_lds = (LDS_BUDGET_BYTES - REDUCTION_BYTES) / lds_per_row;
    const int max_m = (max_m_for_lds < 16u) ? static_cast<int>(max_m_for_lds) : 16;
    if (m > max_m) return 134;
    if (final_hidden == nullptr || final_norm_w == nullptr ||
        lm_head_w == nullptr || logits == nullptr) {
        return 133;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    const int ordinal_int = static_cast<int>(device_ordinal);

    // Batched WMMA path requires gfx11xx + hidden % 16 == 0.
    if (!device_supports_wmma_bf16(ordinal_int) || (hidden % 16 != 0)) {
        // Phase 6.4a only ships the WMMA kernel — non-WMMA hosts should
        // call the single-M launcher K times. Returning 138 (unsupported
        // hardware/dim combination) makes the failure mode explicit.
        return 138;
    }

    constexpr int wmma_block_size = 32;
    const int grid_x = (vocab + 15) / 16;
    // LDS: 32 F32 reduction scratch + m * hidden u16 BF16-rounded inputs.
    // Bounded above by LDS_BUDGET_BYTES via the `m > max_m` check above.
    const size_t lds_bytes =
        REDUCTION_BYTES +
        static_cast<size_t>(m) * static_cast<size_t>(hidden) * sizeof(uint16_t);

    hipLaunchKernelGGL(
        qwen36_moe::qwen36_moe_lm_head_batched_wmma_kernel<hip_bfloat16>,
        dim3(static_cast<unsigned int>(grid_x)),
        dim3(static_cast<unsigned int>(wmma_block_size)),
        lds_bytes, 0,
        m,
        static_cast<const hip_bfloat16*>(final_hidden),
        static_cast<const hip_bfloat16*>(final_norm_w),
        static_cast<const hip_bfloat16*>(lm_head_w),
        static_cast<hip_bfloat16*>(logits),
        static_cast<hip_bfloat16*>(x_normed_out),
        hidden,
        vocab,
        rms_norm_eps);

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// MTP pre-fusion launcher (Phase 6.2c.1).
//
// Single-block kernel: 256 threads, ~17 KiB LDS at hidden=2048. Computes
//   e_norm = rmsnorm(e_in,   pre_fc_norm_embedding_w, eps)
//   h_norm = rmsnorm(h_base, pre_fc_norm_hidden_w,    eps)
//   fused  = mtp.fc @ cat([e_norm, h_norm], dim=-1)
// All BF16-rounded to match the Phase 6.2a Python oracle byte-for-byte
// through the rounding boundary. Status codes match the lm_head launcher.
extern "C" uint64_t qwen36_moe_hip_mtp_pre_fusion_launch(
    int           dtype,
    size_t        device_ordinal,
    int           hidden,
    float         rms_norm_eps,
    const void*   e_in,
    const void*   h_base,
    const void*   pre_fc_norm_embedding_w,
    const void*   pre_fc_norm_hidden_w,
    const void*   fc_w,
    void*         e_norm_out,
    void*         h_norm_out,
    void*         fused_out) {
    if (dtype != 2) return 130;            // only bf16 supported
    if (hidden <= 0) return 132;
    if (e_in == nullptr || h_base == nullptr ||
        pre_fc_norm_embedding_w == nullptr || pre_fc_norm_hidden_w == nullptr ||
        fc_w == nullptr ||
        e_norm_out == nullptr || h_norm_out == nullptr || fused_out == nullptr) {
        return 133;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    constexpr int block_size = 256;
    if (hidden % block_size != 0) {
        // The block-reduction loop assumes hidden % block_size == 0 (no
        // tail elements after the strided pass). 35B-A3B has hidden=2048
        // which divides cleanly; reject anything else loudly.
        return 134;
    }

    // shared_scratch [block_size] + e_norm_lds [hidden] + h_norm_lds [hidden],
    // all F32. = (256 + 2*2048) * 4 = 17,408 bytes at hidden=2048.
    const size_t lds_bytes =
        static_cast<size_t>(block_size + 2 * hidden) * sizeof(float);

    hipLaunchKernelGGL(
        qwen36_moe::qwen36_moe_mtp_pre_fusion_kernel<hip_bfloat16>,
        dim3(1), dim3(block_size), lds_bytes, 0,
        hidden, rms_norm_eps,
        static_cast<const hip_bfloat16*>(e_in),
        static_cast<const hip_bfloat16*>(h_base),
        static_cast<const hip_bfloat16*>(pre_fc_norm_embedding_w),
        static_cast<const hip_bfloat16*>(pre_fc_norm_hidden_w),
        static_cast<const hip_bfloat16*>(fc_w),
        static_cast<hip_bfloat16*>(e_norm_out),
        static_cast<hip_bfloat16*>(h_norm_out),
        static_cast<hip_bfloat16*>(fused_out));

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// =============================================================================
// Phase 3e: persistent decode megakernel launcher.
//
// Walks the layer descriptor array in a single cooperative HIP launch,
// dispatching attn or linear-attn (selected per-layer by
// `desc.is_full_attention`) followed by FFN, with `grid_barrier` between
// phases. Replaces 81 step-kernel launches/token (40 attn + 40 ffn + 1
// lm_head; lm_head still launches separately at this stage) with one.
//
// Returns the project status in the low word and the native HIP/CUDA runtime
// status in the high word. Native status is zero for validation failures.
//
// Caller responsibility:
//   - `hidden_ping` is uploaded with the initial hidden BF16 bytes; the
//     final hidden lands back in `hidden_ping` after even `num_layers`.
//   - `int4_scales` is null for BF16 baked models, non-null for INT4
//     baked models. Each per-layer entry's per-tensor scale/zero pointers
//     are independent — null pair keeps that tensor on the BF16 path.
//   - `workspace`, `counters`, `barrier_counter`, `barrier_flag` are all
//     pre-allocated on device. `counters` is the first 16 u32s of a
//     96-byte sync_buf (counters[0..16], barrier_counter at +64,
//     barrier_flag at +68). Caller zeros the whole sync_buf before launch
//     (this fn also zeros the entire 96 bytes defensively).

extern "C" uint64_t qwen36_moe_hip_persistent_decode_launch(
    int           dtype,
    size_t        device_ordinal,
    int           num_layers,
    int           start_layer,
    int           end_layer_exclusive,
    int           mode,
    const qwen36_moe::DecodeLayerDesc* layers,
    const qwen36_moe::Int4ScaleDesc*   int4_scales,    // nullable
    const qwen36_moe::KVCacheFp8Desc*  kv_fp8_descs,   // nullable
    int           hidden,
    int           num_heads,
    int           num_kv_heads,
    int           head_dim,
    int           rotary_dim,
    int           num_k_heads,
    int           num_v_heads,
    int           head_k_dim,
    int           head_v_dim,
    int           conv_kernel_dim,
    int           num_experts,
    int           moe_intermediate,
    int           shared_intermediate,
    int           top_k,
    int           vocab,                  // 0 ⇒ skip lm_head fold (prefill)
    float         rope_theta,
    float         rms_norm_eps,
    int           position,
    int           cache_pos,    // -1 ⇒ inherit from `position` (dense base
                                //      decode); ≥ 0 ⇒ decoupled KV slot
                                //      (SpecPrefill sparse-prefill / MTP).
    const void*   embed_w,      // [vocab, hidden] BF16, nullable
    int           token_id,     // >=0 => kernel loads embed_w[token_id]
    const unsigned int* token_ids, // [prefill_len], nullable
    int           prefill_len,
    void*         hidden_ping,
    void*         hidden_pong,
    float*        workspace,
    int*          ffn_topk_idx_scratch,
    // Phase 3f folded final RMSnorm + lm_head GEMV. All three are
    // device pointers; nullptr triple ⇒ skip the fold (prefill steps).
    // The engine sets these on gen steps; production logits then come
    // out of the megakernel directly with no separate launch.
    const void*   final_norm_w,           // [hidden] BF16, nullable
    const void*   lm_head_w,              // [vocab, hidden] BF16, nullable
    void*         logits_out,             // [vocab] BF16, nullable
    unsigned int* top1_out,               // [1] U32, nullable
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag) {
    if (dtype != 2) return 130;
    if (num_layers <= 0 || num_layers > 1024) return 131;
    // Residual ping-pong: the kernel does TWO swaps per layer (one
    // between attn/ffn, one at end-of-layer), so each iteration leaves
    // `in_buf == hidden_ping`. The final hidden therefore always lands
    // in `hidden_ping`. Both even and odd `num_layers` are valid; the
    // caller downloads `hidden_ping` for the result.
    if (layers == nullptr) return 133;
    if (hidden <= 0 || num_experts <= 0 || top_k <= 0) return 134;
    if (mode < 0 || mode > 13) return 140;
    if (start_layer < 0 || start_layer >= num_layers) return 141;
    if (mode == 0) {
        if (end_layer_exclusive <= start_layer ||
            end_layer_exclusive > num_layers) {
            return 142;
        }
    } else {
        // Router-only, attention-only, FFN-only, and FFN staged-profile
        // modes are single-layer segmented sparse-VMM/profile entry points.
        // The host may remap between launches, so folded lm_head is
        // intentionally available only to the full-step mode.
        end_layer_exclusive = start_layer + 1;
    }
    // FFN's concurrent-experts dispatch uses counters[group_id] for Phase G
    // and counters[top_k + group_id] for Phase I — i.e., 2*top_k slots.
    // sync_buf provisions exactly 16 u32 counters (also matches
    // reset_counters_16's clear width), so top_k must be ≤ 8.
    if (top_k > 8) return 135;
    if (hidden_ping == nullptr || hidden_pong == nullptr) return 136;
    if ((token_ids == nullptr && prefill_len > 1) ||
        (token_ids != nullptr && prefill_len <= 0)) return 147;
    if (token_ids == nullptr) {
        if ((embed_w == nullptr && token_id >= 0) ||
            (embed_w != nullptr && token_id < 0)) return 146;
    } else if (embed_w == nullptr) {
        return 148;
    }
    if (workspace == nullptr || ffn_topk_idx_scratch == nullptr) return 137;
    if (counters == nullptr || barrier_counter == nullptr ||
        barrier_flag == nullptr) {
        return 138;
    }
    // Phase 3f lm_head fold: either all three buffers + vocab>0
    // (engine wants logits) or all four off (prefill / fold-disabled).
    // Mixed state would silently skip lm_head while pretending to do
    // it — reject up front so the caller catches the misuse.
    const bool lm_head_on = (final_norm_w != nullptr) || (lm_head_w != nullptr) ||
                            (logits_out != nullptr) || (top1_out != nullptr) ||
                            (vocab > 0);
    const bool lm_head_complete = (final_norm_w != nullptr) && (lm_head_w != nullptr) &&
                                  ((logits_out != nullptr) || (top1_out != nullptr)) &&
                                  (vocab > 0);
    if (lm_head_on && !lm_head_complete) return 139;
    if (lm_head_on && (mode != 0 || end_layer_exclusive != num_layers)) return 143;
    if (token_ids != nullptr && lm_head_on) return 149;

    // KV-FP8 desc validation: when present, every full-attn layer must
    // carry both kv_scale_k and kv_scale_v (or neither). Linear-attn
    // layers must carry null pointers in this struct.
    if (kv_fp8_descs != nullptr) {
        for (int li = 0; li < static_cast<int>(num_layers); ++li) {
            const auto& d  = layers[li];
            const auto& kf = kv_fp8_descs[li];
            const bool full = (d.is_full_attention == 1);
            const bool both = (kf.kv_scale_k != nullptr && kf.kv_scale_v != nullptr);
            const bool none = (kf.kv_scale_k == nullptr && kf.kv_scale_v == nullptr);
            if (full && !(both || none)) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_scale_k/v must both be "
                    "set or both null (got %p / %p)\n",
                    li, kf.kv_scale_k, kf.kv_scale_v);
                return 140;
            }
            if (!full && !none) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d (linear): kv_scale_k/v "
                    "must be null (got %p / %p)\n",
                    li, kf.kv_scale_k, kf.kv_scale_v);
                return 141;
            }
            if (full && both && ((d.kv_shadow_k != nullptr) != (d.kv_shadow_v != nullptr))) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_shadow_k/v must agree "
                    "(got %p / %p)\n",
                    li, d.kv_shadow_k, d.kv_shadow_v);
                return 142;
            }
            if (full && both && d.kv_shadow_k != nullptr && d.kv_shadow_window <= 0) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_shadow_window must be > 0 "
                    "when kv_shadow_k/v are set (got %d)\n",
                    li, d.kv_shadow_window);
                return 144;
            }
            if ((!full || none || d.kv_shadow_k == nullptr) && d.kv_shadow_window != 0) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_shadow_window must be 0 "
                    "when the BF16 sidecar is disabled (got %d)\n",
                    li, d.kv_shadow_window);
                return 145;
            }
        }
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    // Zero the full 96-byte sync_buf before launch — counters[0..16] (64
    // bytes) + barrier_counter (4) + barrier_flag (4) + 24 bytes of pad.
    hipError_t memset_err = hipMemsetAsync(counters, 0, 96);
    if (memset_err != hipSuccess) {
        return backend_failure(200, memset_err);
    }

    const size_t lds_bytes =
        static_cast<size_t>(hidden + block_size) * sizeof(float);

    // The persistent kernel used to share one WMMA template bit across
    // attention, linear attention, FFN, and lm-head. Their independent
    // parity results differ: attention/linear are qualified, while FFN
    // and lm-head remain on the scalar path pending qualification.
    const bool disable_wmma =
        std::getenv("SUPERSONIC_QWEN36_DISABLE_PERSISTENT_WMMA") != nullptr;
    const bool use_attn_wmma =
        !disable_wmma &&
        device_supports_wmma_bf16(static_cast<int>(device_ordinal));

    if (use_attn_wmma) {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_persistent_decode_kernel<
                hip_bfloat16, true, false, false>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            num_layers, start_layer, end_layer_exclusive, mode,
            layers, int4_scales, kv_fp8_descs,
            hidden, num_heads, num_kv_heads, head_dim, rotary_dim,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_dim,
            num_experts, moe_intermediate, shared_intermediate, top_k,
            vocab, rope_theta, rms_norm_eps, position, cache_pos,
            static_cast<const hip_bfloat16*>(embed_w), token_id,
            token_ids, prefill_len,
            static_cast<hip_bfloat16*>(hidden_ping),
            static_cast<hip_bfloat16*>(hidden_pong),
            workspace, ffn_topk_idx_scratch,
            static_cast<const hip_bfloat16*>(final_norm_w),
            static_cast<const hip_bfloat16*>(lm_head_w),
            static_cast<hip_bfloat16*>(logits_out),
            top1_out,
            counters, barrier_counter, barrier_flag);
    } else {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_persistent_decode_kernel<
                hip_bfloat16, false, false, false>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            lds_bytes, 0,
            num_layers, start_layer, end_layer_exclusive, mode,
            layers, int4_scales, kv_fp8_descs,
            hidden, num_heads, num_kv_heads, head_dim, rotary_dim,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_dim,
            num_experts, moe_intermediate, shared_intermediate, top_k,
            vocab, rope_theta, rms_norm_eps, position, cache_pos,
            static_cast<const hip_bfloat16*>(embed_w), token_id,
            token_ids, prefill_len,
            static_cast<hip_bfloat16*>(hidden_ping),
            static_cast<hip_bfloat16*>(hidden_pong),
            workspace, ffn_topk_idx_scratch,
            static_cast<const hip_bfloat16*>(final_norm_w),
            static_cast<const hip_bfloat16*>(lm_head_w),
            static_cast<hip_bfloat16*>(logits_out),
            top1_out,
            counters, barrier_counter, barrier_flag);
    }

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// =============================================================================
// Stage A (M3): batched-Q full-attention prefill kernel launcher.
//
// Standalone attention kernel — pre-projection (Q/K/V matmul + RoPE) and
// KV cache write are caller responsibilities (see batched_prefill_kv_write
// in M5/M6). Output is the pre-o_proj attention result `[B, H, q_len, D]`
// in F32. dtype: 2 = bf16 (only path supported initially since qwen3.6-moe
// runs INT4 weights → BF16 activations).
//
// Wave64 portability: the kernel hardcodes block.x = 32 and assumes
// warpSize == 32 in stride math. Refuse the launch when the device's
// warpSize differs (return code 137; mirrors full_attention_bridge.cpp's
// launch_tiled guard from PR #219).
// =============================================================================

extern "C" uint64_t qwen36_moe_hip_batched_prefill_attn_full_launch(
    int           dtype,
    size_t        device_ordinal,
    int           batch_size,
    int           q_heads,
    int           kv_heads,
    int           q_len,
    int           kv_len,
    int           head_dim,
    float         scale,
    int           seqlen_offset,
    const void*   query,
    const void*   key,
    const void*   value,
    void*         out
) {
    if (dtype != 2) return 130;
    if (q_heads <= 0 || kv_heads <= 0) return 131;
    if (q_heads % kv_heads != 0) return 132;
    if (head_dim <= 0 || head_dim > 8 * 32) return 133;
    if (q_len <= 0 || kv_len <= 0) return 134;
    if (seqlen_offset < 0 || seqlen_offset + q_len > kv_len) return 135;
    if (batch_size <= 0) return 136;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }
    if (props.warpSize != 32) {
        // Fall through to the per-token path on wave64; this kernel
        // assumes warpSize == 32 in stride math.
        return 137;
    }

    constexpr int BM = 4;

    // BK chosen so 2 * BK * head_dim * sizeof(bf16) stays under 48 KiB.
    // qwen3.6-moe runs hd=256 → BK=32 (32 KiB). Keep the dispatch open
    // for smaller head_dim values to support hypothetical future shapes.
    int bk;
    if      (head_dim <=  64) bk = 128;
    else if (head_dim <= 128) bk = 64;
    else                      bk = 32;

    const size_t lds_bytes =
        static_cast<size_t>(2) * static_cast<size_t>(bk) *
        static_cast<size_t>(head_dim) * sizeof(hip_bfloat16);
    if (lds_bytes > 48 * 1024) return 138;

    const int num_kv_groups = q_heads / kv_heads;
    const int grid_x = (q_len + BM - 1) / BM;
    dim3 grid(static_cast<unsigned int>(grid_x),
              static_cast<unsigned int>(q_heads),
              static_cast<unsigned int>(batch_size));
    dim3 block(32u, static_cast<unsigned int>(BM), 1u);

    auto launch = [&](auto bk_val) {
        constexpr int BK_C = decltype(bk_val)::value;
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_batched_prefill_attn_full_kernel<hip_bfloat16, BM, BK_C>),
            grid, block, lds_bytes, 0,
            batch_size, q_heads, kv_heads, q_len, kv_len, head_dim,
            num_kv_groups, scale, seqlen_offset,
            static_cast<const hip_bfloat16*>(query),
            static_cast<const hip_bfloat16*>(key),
            static_cast<const hip_bfloat16*>(value),
            static_cast<float*>(out));
    };

    if      (bk == 128) launch(std::integral_constant<int, 128>{});
    else if (bk ==  64) launch(std::integral_constant<int,  64>{});
    else                launch(std::integral_constant<int,  32>{});

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// =============================================================================
// Stage B (M9): router permutation kernel launcher.
//
// Groups per-token top-K expert assignments by target expert via a
// single-block counting sort. Inputs are GPU-resident `topk_idx` (i32)
// and `topk_weight` (BF16); outputs are `expert_offsets` (i32),
// `permuted_token_idx` (i32), `permuted_kpos` (i32), and
// `permuted_weight` (BF16). All output buffers must be pre-allocated by
// the caller — the kernel writes them in place.
//
// Status codes:
//   140 invalid args (n_tokens / top_k / num_experts <= 0)
//   141 num_experts > 256                  (LDS pinned at MAX_EXPERTS=256)
//   142 top_k > 16                         (sanity bound)
//   143 n_tokens * top_k > 16384           (would exceed reasonable scratch)
//   254 launch error (with native status) 255 sync error (with native status)
// =============================================================================

extern "C" uint64_t qwen36_moe_hip_batched_prefill_router_permute_launch(
    size_t      device_ordinal,
    int         n_tokens,
    int         top_k,
    int         num_experts,
    const void* topk_idx,
    const void* topk_weight,
    void*       expert_offsets,
    void*       permuted_token_idx,
    void*       permuted_kpos,
    void*       permuted_weight
) {
    if (n_tokens <= 0 || top_k <= 0 || num_experts <= 0) return 140;
    if (num_experts > 256) return 141;
    if (top_k > 16) return 142;
    if (static_cast<int64_t>(n_tokens) * static_cast<int64_t>(top_k) > 16384) return 143;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    constexpr int BLOCK = 256;
    dim3 grid(1u, 1u, 1u);
    dim3 block(static_cast<unsigned int>(BLOCK), 1u, 1u);

    hipLaunchKernelGGL(
        (qwen36_moe::qwen36_moe_batched_prefill_router_permute_kernel<256>),
        grid, block, 0, 0,
        n_tokens, top_k, num_experts,
        static_cast<const int*>(topk_idx),
        static_cast<const hip_bfloat16*>(topk_weight),
        static_cast<int*>(expert_offsets),
        static_cast<int*>(permuted_token_idx),
        static_cast<int*>(permuted_kpos),
        static_cast<hip_bfloat16*>(permuted_weight));

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// =============================================================================
// Stage B (M10): grouped-expert INT4 GEMM launcher.
//
// One launch processes ALL `num_experts` experts via persistent-block
// work-stealing on the expert id. For each claimed expert the block walks
// the segment of permuted rows produced by the M9 router permutation
// kernel, gathering x from `x_norm[permuted_token_idx[row]]` and writing
// down(silu(gate(x)) * up(x)) into `expert_out[row * hidden]`.
//
// Block geometry:
//   blocks  = props.multiProcessorCount  (one block per CU, like the
//                                          existing 4b INT4 path)
//   threads = 256                         (matches FFN block_size)
//
// LDS budget (F32 elements, per the kernel header comment):
//   hidden + 2*I + I  →  14 KiB at hidden=2048, I=512 (8K + 4K + 2K)
//
// Status codes:
//   150 invalid args (zero/negative dims)
//   151 num_experts > 256
//   152 hidden / moe_intermediate not divisible by 16
//   153 missing or invalid INT4 descriptor
//   154 top_k * n_tokens > 16384
//   155 dtype != bf16 (only path supported initially)
//   156 LDS overflow (>48 KiB)
//   254 launch error (with native status)
//   255 sync error (with native status)
// =============================================================================

extern "C" uint64_t qwen36_moe_hip_batched_prefill_grouped_expert_launch(
    int           dtype,
    size_t        device_ordinal,
    int           n_tokens,
    int           top_k,
    int           num_experts,
    int           hidden,
    int           moe_intermediate,
    const void*   x_norm,
    const void*   expert_offsets,
    const void*   permuted_token_idx,
    const void*   experts_gate_up_w,
    const Qwen36Int4WeightDesc* experts_gate_up_desc,
    const void*   experts_down_w,
    const Qwen36Int4WeightDesc* experts_down_desc,
    void*         expert_out,
    void*         counters
) {
    if (n_tokens <= 0 || top_k <= 0 || num_experts <= 0) return 150;
    if (hidden <= 0 || moe_intermediate <= 0) return 150;
    if (num_experts > 256) return 151;
    if (experts_gate_up_desc == nullptr || experts_down_desc == nullptr) return 153;
    if (validate_int4_descriptor_geometry(
            *experts_gate_up_desc, num_experts,
            2 * moe_intermediate, hidden) != 0 ||
        validate_int4_descriptor_geometry(
            *experts_down_desc, num_experts,
            hidden, moe_intermediate) != 0) {
        return 153;
    }
    if (static_cast<int64_t>(n_tokens) * static_cast<int64_t>(top_k) > 16384) return 154;
    if (dtype != 2) return 155;

    // Reduction dims must be multiples of 16 for both the WMMA path
    // (`wmma_int4_matvec_partial_16rows` strides K by 16) and the scalar
    // 8-wide dq8 path (strides cols by 8). Descriptor validation enforces
    // the quant geometry; the compute tile itself additionally requires 16.
    if ((hidden % 16) != 0 || (moe_intermediate % 16) != 0) return 152;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    hipError_t props_err =
        hipGetDeviceProperties(&props, static_cast<int>(device_ordinal));
    if (props_err != hipSuccess) {
        return backend_failure(250, props_err);
    }

    constexpr int BLOCK = 256;
    const int num_blocks = props.multiProcessorCount > 0
        ? props.multiProcessorCount
        : 32;

    // LDS sizing — see kernel header for layout. F32 elements:
    //   x_lds [hidden] + gu_lds [2*I] + silu_mul_lds [I]
    const size_t lds_bytes =
        (static_cast<size_t>(hidden) +
         static_cast<size_t>(2 * moe_intermediate) +
         static_cast<size_t>(moe_intermediate)) * sizeof(float);
    if (lds_bytes > 48 * 1024) return 156;

    dim3 grid(static_cast<unsigned int>(num_blocks), 1u, 1u);
    dim3 block(static_cast<unsigned int>(BLOCK), 1u, 1u);

    const bool wmma =
        experts_gate_up_desc->input_group_size % 16 == 0 &&
        experts_down_desc->input_group_size % 16 == 0 &&
        device_supports_wmma_bf16(static_cast<int>(device_ordinal));

    if (wmma) {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_batched_prefill_grouped_expert_kernel<hip_bfloat16, true>),
            grid, block, lds_bytes, 0,
            n_tokens, top_k, num_experts, hidden, moe_intermediate,
            static_cast<const hip_bfloat16*>(x_norm),
            static_cast<const int*>(expert_offsets),
            static_cast<const int*>(permuted_token_idx),
            static_cast<const uint8_t*>(experts_gate_up_w),
            *experts_gate_up_desc,
            static_cast<const uint8_t*>(experts_down_w),
            *experts_down_desc,
            static_cast<hip_bfloat16*>(expert_out),
            static_cast<unsigned int*>(counters));
    } else {
        hipLaunchKernelGGL(
            (qwen36_moe::qwen36_moe_batched_prefill_grouped_expert_kernel<hip_bfloat16, false>),
            grid, block, lds_bytes, 0,
            n_tokens, top_k, num_experts, hidden, moe_intermediate,
            static_cast<const hip_bfloat16*>(x_norm),
            static_cast<const int*>(expert_offsets),
            static_cast<const int*>(permuted_token_idx),
            static_cast<const uint8_t*>(experts_gate_up_w),
            *experts_gate_up_desc,
            static_cast<const uint8_t*>(experts_down_w),
            *experts_down_desc,
            static_cast<hip_bfloat16*>(expert_out),
            static_cast<unsigned int*>(counters));
    }

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}

// =============================================================================
// Stage B (M11): unpermute + weighted combine launcher.
//
// Builds the per-token weighted sum across `top_k` permuted expert outputs.
// Caller pre-computes `permuted_inverse[N * top_k]` (the inverse of M9's
// scatter, i.e. `permuted_inverse[token * top_k + kpos] = dst`) host-side
// and uploads it before launch — keeps M9 untouched and turns the unpermute
// itself into a simple gather + dot product.
//
// Block geometry:
//   blocks  = (ceil(hidden / 256), n_tokens, 1)
//   threads = 256
//
// Status codes:
//   160 invalid args (zero/negative dims)
//   161 top_k > 16
//   162 dtype != bf16
//   163 hidden too large (no shared scratch needed; this is a paranoia cap)
//   164 reserved
//   254 launch error (with native status)
//   255 sync error (with native status)
// =============================================================================

extern "C" uint64_t qwen36_moe_hip_batched_prefill_unpermute_combine_launch(
    int           dtype,
    size_t        device_ordinal,
    int           n_tokens,
    int           top_k,
    int           hidden,
    const void*   permuted_inverse,
    const void*   permuted_weight,
    const void*   expert_out,
    void*         combined
) {
    if (n_tokens <= 0 || top_k <= 0 || hidden <= 0) return 160;
    if (top_k > 16) return 161;
    if (dtype != 2) return 162;  // BF16 only.
    // hidden bound — sanity cap. The kernel itself has no per-block LDS
    // requirement so the only effective limit is grid.x = ceil(hidden/256)
    // which is huge before it matters; we cap at 65536 to keep error paths
    // simple.
    if (hidden > 65536) return 163;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    constexpr int BLOCK = 256;
    const unsigned int grid_x =
        static_cast<unsigned int>((hidden + BLOCK - 1) / BLOCK);
    dim3 grid(grid_x, static_cast<unsigned int>(n_tokens), 1u);
    dim3 block(static_cast<unsigned int>(BLOCK), 1u, 1u);

    hipLaunchKernelGGL(
        (qwen36_moe::qwen36_moe_batched_prefill_unpermute_combine_kernel<hip_bfloat16, BLOCK>),
        grid, block, 0, 0,
        n_tokens, top_k, hidden,
        static_cast<const int*>(permuted_inverse),
        static_cast<const hip_bfloat16*>(permuted_weight),
        static_cast<const hip_bfloat16*>(expert_out),
        static_cast<hip_bfloat16*>(combined));

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err   = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return backend_failure(254, launch_err);
    if (sync_err != hipSuccess) return backend_failure(255, sync_err);
    return 0;
}
