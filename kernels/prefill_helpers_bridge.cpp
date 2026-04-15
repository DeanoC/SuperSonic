// Bridge for prefill helper kernels.
// Separate compilation unit — does not touch the decode megakernel files.

#include "prefill_helpers.hip"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) { hipSetDevice(target); changed = true; }
    }
    ~ScopedHipDevice() { if (changed && previous >= 0) hipSetDevice(previous); }
};

// ---- element_add ----

template <typename T>
int element_add_device(int device_ordinal, size_t total_elems,
                       const void* lhs, const void* rhs, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_element_add_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        total_elems,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 301;
    if (hipDeviceSynchronize() != hipSuccess) return 302;
    return 0;
}

// ---- apply_rope_prefill ----

template <typename T>
int apply_rope_prefill_device(int device_ordinal,
                              int seq_len, int num_heads, int head_dim, int half_rot,
                              const void* cos_table, const void* sin_table, void* data) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_apply_rope_prefill_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, num_heads, head_dim, half_rot,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(data));
    if (hipGetLastError() != hipSuccess) return 311;
    if (hipDeviceSynchronize() != hipSuccess) return 312;
    return 0;
}

// ---- transpose [S,H,D] -> [H,S,D] ----

template <typename T>
int transpose_shd_hsd_device(int device_ordinal,
                             int S, int H, int D,
                             const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * H * D;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_shd_hsd_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, H, D,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    if (hipGetLastError() != hipSuccess) return 321;
    if (hipDeviceSynchronize() != hipSuccess) return 322;
    return 0;
}

// ---- transpose + pad for conv ----

template <typename T>
int transpose_pad_conv_device(int device_ordinal,
                              int S, int C, int pad,
                              const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    // Zero the entire dst buffer first (to get zero-padding)
    const size_t dst_bytes = static_cast<size_t>(C) * (pad + S) * sizeof(T);
    if (hipMemset(dst, 0, dst_bytes) != hipSuccess) return 330;

    const size_t total = static_cast<size_t>(S) * C;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_pad_conv_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, pad,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    if (hipGetLastError() != hipSuccess) return 331;
    if (hipDeviceSynchronize() != hipSuccess) return 332;
    return 0;
}

// ---- extract conv state ----

template <typename T>
int extract_conv_state_device(int device_ordinal,
                              int S, int C, int kern_minus_1,
                              const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(kern_minus_1) * C;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_extract_conv_state_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, kern_minus_1,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    if (hipGetLastError() != hipSuccess) return 341;
    if (hipDeviceSynchronize() != hipSuccess) return 342;
    return 0;
}

} // namespace

// ---- extern "C" wrappers ----

extern "C" int dotcache_qwen35_hip_element_add(
    int dtype, size_t device_ordinal, size_t total_elems,
    const void* lhs, const void* rhs, void* out
) {
    switch (dtype) {
    case 0: return element_add_device<half>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    case 1: return element_add_device<float>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    case 2: return element_add_device<hip_bfloat16>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    default: return 300;
    }
}

extern "C" int dotcache_qwen35_hip_apply_rope_prefill(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_heads, size_t head_dim, size_t half_rot,
    const void* cos_table, const void* sin_table, void* data
) {
    switch (dtype) {
    case 0: return apply_rope_prefill_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    case 1: return apply_rope_prefill_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    case 2: return apply_rope_prefill_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    default: return 310;
    }
}

extern "C" int dotcache_qwen35_hip_transpose_shd_hsd(
    int dtype, size_t device_ordinal,
    size_t S, size_t H, size_t D,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return transpose_shd_hsd_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    case 1: return transpose_shd_hsd_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    case 2: return transpose_shd_hsd_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    default: return 320;
    }
}

extern "C" int dotcache_qwen35_hip_transpose_pad_conv(
    int dtype, size_t device_ordinal,
    size_t S, size_t C, size_t pad,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return transpose_pad_conv_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    case 1: return transpose_pad_conv_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    case 2: return transpose_pad_conv_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    default: return 329;
    }
}

extern "C" int dotcache_qwen35_hip_extract_conv_state(
    int dtype, size_t device_ordinal,
    size_t S, size_t C, size_t kern_minus_1,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return extract_conv_state_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    case 1: return extract_conv_state_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    case 2: return extract_conv_state_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    default: return 340;
    }
}
