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
