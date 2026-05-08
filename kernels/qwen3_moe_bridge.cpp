// Qwen3-MoE HIP launch bridge.

#include "qwen3_moe.hip"

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

bool sync_each_kernel_enabled() {
    const char* env = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
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

static_assert(sizeof(qwen3_moe::DecodeLayerDesc) == 168,
              "Qwen3MoeDecodeLayerDesc size drift; update Rust and C++ together");
static_assert(sizeof(qwen3_moe::Int4ScaleDesc) == 104,
              "Qwen3MoeInt4ScaleDesc size drift; update Rust and C++ together");

} // namespace

extern "C" int qwen3_moe_hip_stub_launch(
    int                                dtype,
    size_t                             device_ordinal,
    size_t                             num_layers,
    const qwen3_moe::DecodeLayerDesc*  layers,
    float*                             workspace,
    unsigned int*                      counters) {
    if (dtype != 2) return 110; // BF16 only for the descriptor smoke path.
    if (num_layers == 0 || num_layers > 1024) return 100;
    if (layers == nullptr || workspace == nullptr || counters == nullptr) {
        return 101;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;

    if (hipMemsetAsync(counters, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }

    hipLaunchKernelGGL(qwen3_moe::qwen3_moe_descriptor_walk_stub,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(64),
                       0, 0,
                       static_cast<int>(num_layers),
                       layers,
                       workspace,
                       counters);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

extern "C" int qwen3_moe_hip_decode_layer_launch(
    int                                      dtype,
    size_t                                   device_ordinal,
    const qwen3_moe::DecodeLayerDesc*        layer,
    const qwen3_moe::Int4ScaleDesc*          int4,
    int                                      hidden,
    int                                      position,
    const void*                              input_hidden,
    void*                                    output_hidden,
    float*                                   workspace) {
    if (dtype != 2) return 110; // BF16 activations only.
    if (layer == nullptr || int4 == nullptr || input_hidden == nullptr ||
        output_hidden == nullptr || workspace == nullptr) {
        return 101;
    }
    if (hidden <= 0 || position < 0) return 102;
    if (int4->group_size <= 0) return 103;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    const int block_threads = 1024;
    const size_t shmem = block_threads * sizeof(float);
    hipLaunchKernelGGL(qwen3_moe::qwen3_moe_decode_layer_kernel,
                       dim3(1),
                       dim3(block_threads),
                       shmem, 0,
                       *layer,
                       *int4,
                       hidden,
                       position,
                       static_cast<const hip_bfloat16*>(input_hidden),
                       static_cast<hip_bfloat16*>(output_hidden),
                       workspace);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

extern "C" int qwen3_moe_hip_persistent_decode_launch(
    int                                      dtype,
    size_t                                   device_ordinal,
    int                                      num_layers,
    const qwen3_moe::DecodeLayerDesc*        layers,
    const qwen3_moe::Int4ScaleDesc*          int4_descs,
    int                                      hidden,
    int                                      position,
    const void*                              input_hidden,
    void*                                    hidden_ping,
    void*                                    hidden_pong,
    float*                                   workspace,
    unsigned int*                            sync,
    uint64_t*                                profile) {
    if (dtype != 2) return 110; // BF16 activations only.
    if (num_layers <= 0 || num_layers > 1024) return 100;
    if (layers == nullptr || int4_descs == nullptr || input_hidden == nullptr ||
        hidden_ping == nullptr || hidden_pong == nullptr || workspace == nullptr ||
        sync == nullptr) {
        return 101;
    }
    if (hidden <= 0 || position < 0) return 102;

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    if (num_blocks > qwen3_moe::Q3_MAX_GRID_BLOCKS) {
        num_blocks = qwen3_moe::Q3_MAX_GRID_BLOCKS;
    }
    if (num_blocks <= 0) return 250;

    if (hipMemsetAsync(sync, 0, 96) != hipSuccess) {
        return 200;
    }
    if (profile != nullptr) {
        const size_t profile_bytes =
            static_cast<size_t>(num_layers) * qwen3_moe::Q3_PROFILE_PHASES *
            sizeof(uint64_t);
        if (hipMemsetAsync(profile, 0, profile_bytes) != hipSuccess) {
            return 200;
        }
    }

    const int block_threads = 256;
    const size_t shmem = block_threads * sizeof(float);
    hipLaunchKernelGGL(qwen3_moe::qwen3_moe_persistent_decode_kernel,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_threads),
                       shmem, 0,
                       num_layers,
                       layers,
                       int4_descs,
                       hidden,
                       position,
                       static_cast<const hip_bfloat16*>(input_hidden),
                       static_cast<hip_bfloat16*>(hidden_ping),
                       static_cast<hip_bfloat16*>(hidden_pong),
                       workspace,
                       sync,
                       profile);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

extern "C" int qwen3_moe_hip_lm_head_launch(
    int          dtype,
    size_t       device_ordinal,
    int          hidden,
    int          vocab,
    float        rms_norm_eps,
    const void*  final_hidden,
    const void*  final_norm_w,
    const void*  lm_head_w,
    void*        logits,
    unsigned int* counter) {
    if (dtype != 2) return 110;
    if (hidden <= 0 || vocab <= 0) return 100;
    if (final_hidden == nullptr || final_norm_w == nullptr ||
        lm_head_w == nullptr || logits == nullptr || counter == nullptr) {
        return 101;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    const int block_threads = 256;
    const size_t shmem = block_threads * sizeof(float);
    if (hipMemsetAsync(counter, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }
    hipLaunchKernelGGL(qwen3_moe::qwen3_moe_lm_head_kernel,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_threads),
                       shmem, 0,
                       hidden,
                       vocab,
                       rms_norm_eps,
                       static_cast<const hip_bfloat16*>(final_hidden),
                       static_cast<const hip_bfloat16*>(final_norm_w),
                       static_cast<const hip_bfloat16*>(lm_head_w),
                       static_cast<hip_bfloat16*>(logits),
                       counter);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

extern "C" int qwen3_moe_hip_lm_head_int4_launch(
    int          dtype,
    size_t       device_ordinal,
    int          hidden,
    int          vocab,
    float        rms_norm_eps,
    const void*  final_hidden,
    const void*  final_norm_w,
    const void*  lm_head_w,
    const void*  lm_head_scale,
    const void*  lm_head_zero,
    int          group_size,
    void*        logits,
    unsigned int* counter) {
    if (dtype != 2) return 110;
    if (hidden <= 0 || vocab <= 0 || group_size <= 0) return 100;
    if (final_hidden == nullptr || final_norm_w == nullptr ||
        lm_head_w == nullptr || lm_head_scale == nullptr ||
        lm_head_zero == nullptr || logits == nullptr || counter == nullptr) {
        return 101;
    }

    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, static_cast<int>(device_ordinal)) !=
        hipSuccess) {
        return 250;
    }
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    const int block_threads = 256;
    const size_t shmem = block_threads * sizeof(float);
    if (hipMemsetAsync(counter, 0, sizeof(unsigned int)) != hipSuccess) {
        return 200;
    }
    hipLaunchKernelGGL(qwen3_moe::qwen3_moe_lm_head_int4_kernel,
                       dim3(static_cast<unsigned int>(num_blocks)),
                       dim3(block_threads),
                       shmem, 0,
                       hidden,
                       vocab,
                       rms_norm_eps,
                       static_cast<const hip_bfloat16*>(final_hidden),
                       static_cast<const hip_bfloat16*>(final_norm_w),
                       static_cast<const uint8_t*>(lm_head_w),
                       static_cast<const hip_bfloat16*>(lm_head_scale),
                       static_cast<const hip_bfloat16*>(lm_head_zero),
                       group_size,
                       static_cast<hip_bfloat16*>(logits),
                       counter);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err =
        sync_each_kernel_enabled() ? hipDeviceSynchronize() : hipSuccess;
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}
