// Phi-4-mini persistent decode megakernel HIP bridge.
//
// Phi-4-mini is pure full-attention across 32 layers (no linear / hybrid
// attention), SwiGLU MLP, GQA 3:1, partial RoPE (rot_dim = 96 via LongRoPE).
// This bridge forks kernels/full_attention_bridge_4b.cpp down to the single
// `persistent_decode` entry. The runner pre-bakes LongRoPE mscale into the
// cos/sin tables and selects the short/long pair per decode step based on
// kv_len, so the kernel reads whichever pointer it is handed.
//
// INT4/FP8/KV-FP8 pointer-table slots are kept for future flag-flip wiring
// (accept nullptr at BF16 launch). Linear-attention scaffolding present in
// the 4B kernel has been stripped from the persistent kernel body and
// descriptor; peripheral `phi4_linear_*` and `phi4_delta_*` __global__
// kernels are left as dead code in phi4.hip — the bridge does not expose
// them, so they have no runtime cost.

#include "phi4.hip"

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;

    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) {
            hipSetDevice(target);
            changed = true;
        }
    }

    ~ScopedHipDevice() {
        if (changed && previous >= 0) {
            hipSetDevice(previous);
        }
    }
};

template <typename T>
int persistent_decode_phi4_device(
    int device_ordinal,
    int num_layers,
    int hidden_dim,
    int intermediate_size,
    int seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    const void* cos_table,
    const void* sin_table,
    int proj_buf_floats,
    int attn_scratch_floats,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    int batch_size,
    const void* batch_descs,
    const void* int4_scales,
    float* layer_trace,
    int layer_trace_components
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 250;

    int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    // The Phi-4 HIP megakernel was tuned around RDNA wave32. On CDNA wave64,
    // the multi-block work-stealing path diverges from the HF oracle; keep the
    // initial CDNA bring-up on a single block for correctness and revisit the
    // work distribution when optimizing performance.
    if (props.warpSize > 32) num_blocks = 1;
    if (const char* override_blocks = std::getenv("SUPERSONIC_PHI4_NUM_BLOCKS")) {
        const int requested = std::atoi(override_blocks);
        if (requested > 0) num_blocks = requested;
    }
    constexpr int block_size = 256;
    // LDS layout matches the 4B kernel: reduction scratch [block_size] + input
    // cache [max(B*hidden_dim, intermediate_size)] + FP8 LUT [256].
    const size_t input_cache = static_cast<size_t>(hidden_dim) * batch_size > static_cast<size_t>(intermediate_size)
        ? static_cast<size_t>(hidden_dim) * batch_size
        : static_cast<size_t>(intermediate_size);
    const size_t fp8_lut_size = 256;
    const size_t shared_bytes = (block_size + input_cache + fp8_lut_size) * sizeof(float);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(phi4_persistent_decode_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        num_layers,
        hidden_dim,
        intermediate_size,
        seqlen_offset,
        static_cast<const Phi4DecodeLayerDesc*>(layers),
        static_cast<T*>(hidden_io),
        workspace,
        counters,
        barrier_counter,
        barrier_flag,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        proj_buf_floats,
        attn_scratch_floats,
        static_cast<const Phi4FP8ScaleDesc*>(fp8_scales),
        static_cast<const Phi4KVCacheFp8Desc*>(kv_fp8_descs),
        batch_size,
        static_cast<const Phi4BatchSeqDesc*>(batch_descs),
        static_cast<const Phi4INT4ScaleDesc*>(int4_scales),
        layer_trace,
        layer_trace_components);
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

} // namespace

// `dtype` encoding matches the Qwen/Gemma bridges: 0 = half, 2 = hip_bfloat16.
// Only BF16 is wired at launch; FP16 returns 256 and is held for future work.
extern "C" int phi4_hip_persistent_decode(
    int dtype,
    size_t device_ordinal,
    size_t num_layers,
    size_t hidden_dim,
    size_t intermediate_size,
    size_t seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    const void* cos_table,
    const void* sin_table,
    size_t proj_buf_floats,
    size_t attn_scratch_floats,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    size_t batch_size,
    const void* batch_descs,
    const void* int4_scales,
    float* layer_trace,
    size_t layer_trace_components) {
    switch (dtype) {
    case 2:
        return persistent_decode_phi4_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(num_layers),
            static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size),
            static_cast<int>(seqlen_offset),
            layers, hidden_io, workspace, counters,
            barrier_counter, barrier_flag,
            cos_table, sin_table,
            static_cast<int>(proj_buf_floats),
            static_cast<int>(attn_scratch_floats),
            fp8_scales,
            kv_fp8_descs,
            static_cast<int>(batch_size),
            batch_descs,
            int4_scales,
            layer_trace,
            static_cast<int>(layer_trace_components));
    default:
        return 256;
    }
}

// Standalone RMSNorm launcher for the final model norm (before lm_head).
// Phi-4 norms do NOT use add_unit_offset, so instantiate the kernel with
// ADD_UNIT_OFFSET=false. One block per row, 256 threads per block — the
// kernel handles n_cols > block_size via strided column iteration.
extern "C" int phi4_hip_rms_norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* xs,
    const void* weight,
    void* out) {
    if (dtype != 2) return 256;  // BF16 only
    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    constexpr int block_size = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(phi4_rms_norm_kernel<hip_bfloat16, false>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block_size),
        0,
        0,
        static_cast<int>(n_rows),
        static_cast<int>(n_cols),
        eps,
        static_cast<const hip_bfloat16*>(xs),
        static_cast<const hip_bfloat16*>(weight),
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}
