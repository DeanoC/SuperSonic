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

// ---- sigmoid_mul ----

template <typename T>
int sigmoid_mul_device(int device_ordinal, size_t total_elems,
                       const void* data, const void* gate, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_sigmoid_mul_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        total_elems,
        static_cast<const T*>(data),
        static_cast<const T*>(gate),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 351;
    if (hipDeviceSynchronize() != hipSuccess) return 352;
    return 0;
}

// ---- compute_beta_g ----

template <typename T>
int compute_beta_g_device(int device_ordinal,
                          int seq_len, int nv,
                          const void* B, const void* A,
                          const void* dt_bias, const void* a_log_exp,
                          void* beta, void* g) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * nv;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_compute_beta_g_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, nv,
        static_cast<const T*>(B),
        static_cast<const T*>(A),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(beta),
        static_cast<T*>(g));
    if (hipGetLastError() != hipSuccess) return 361;
    if (hipDeviceSynchronize() != hipSuccess) return 362;
    return 0;
}

// ---- split_qgate ----

template <typename T>
int split_qgate_device(int device_ordinal,
                       int S, int num_heads, int head_dim,
                       const void* src, void* query_out, void* gate_out) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * num_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_split_qgate_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, num_heads, head_dim,
        static_cast<const T*>(src),
        static_cast<T*>(query_out),
        static_cast<T*>(gate_out));
    if (hipGetLastError() != hipSuccess) return 371;
    if (hipDeviceSynchronize() != hipSuccess) return 372;
    return 0;
}

// ---- split_qkv ----

template <typename T>
int split_qkv_device(int device_ordinal,
                     int S, int key_dim, int val_dim,
                     const void* src, void* Q, void* K, void* V) {
    ScopedHipDevice scoped(device_ordinal);
    const int qkv_dim = key_dim * 2 + val_dim;
    const size_t total = static_cast<size_t>(S) * qkv_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_split_qkv_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, key_dim, val_dim,
        static_cast<const T*>(src),
        static_cast<T*>(Q),
        static_cast<T*>(K),
        static_cast<T*>(V));
    if (hipGetLastError() != hipSuccess) return 381;
    if (hipDeviceSynchronize() != hipSuccess) return 382;
    return 0;
}

// ---- repeat_interleave heads ----

template <typename T>
int repeat_interleave_heads_device(int device_ordinal,
                                   int S, int n_heads, int head_dim, int repeats,
                                   const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const int out_heads = n_heads * repeats;
    const size_t total = static_cast<size_t>(S) * out_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_repeat_interleave_heads_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, n_heads, head_dim, repeats,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    if (hipGetLastError() != hipSuccess) return 391;
    if (hipDeviceSynchronize() != hipSuccess) return 392;
    return 0;
}

// ---- full_attention_decode_flat ----

template <typename T>
int full_attention_decode_flat_device(int device_ordinal,
                                      int batch_size,
                                      int q_heads,
                                      int kv_heads,
                                      int kv_len,
                                      int head_dim,
                                      int num_kv_groups,
                                      float scale,
                                      const void* query,
                                      const void* key,
                                      const void* value,
                                      void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 32;
    if (head_dim > block * 8) return 401;
    const int rows = batch_size * q_heads;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_full_attention_decode_flat_kernel<T>),
        dim3(rows), dim3(block), 0, 0,
        batch_size, q_heads, kv_heads, kv_len, head_dim, num_kv_groups, scale,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 402;
    if (hipDeviceSynchronize() != hipSuccess) return 403;
    return 0;
}

} // namespace

// ---- extern "C" wrappers ----

extern "C" int supersonic_qwen35_hip_element_add(
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

extern "C" int supersonic_qwen35_hip_apply_rope_prefill(
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

extern "C" int supersonic_qwen35_hip_transpose_shd_hsd(
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

extern "C" int supersonic_qwen35_hip_transpose_pad_conv(
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

extern "C" int supersonic_qwen35_hip_extract_conv_state(
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

extern "C" int supersonic_qwen35_hip_sigmoid_mul(
    int dtype, size_t device_ordinal, size_t total_elems,
    const void* data, const void* gate, void* out
) {
    switch (dtype) {
    case 0: return sigmoid_mul_device<half>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    case 1: return sigmoid_mul_device<float>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    case 2: return sigmoid_mul_device<hip_bfloat16>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    default: return 350;
    }
}

extern "C" int supersonic_qwen35_hip_compute_beta_g(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t nv,
    const void* B, const void* A,
    const void* dt_bias, const void* a_log_exp,
    void* beta, void* g
) {
    switch (dtype) {
    case 0: return compute_beta_g_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    case 1: return compute_beta_g_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    case 2: return compute_beta_g_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    default: return 360;
    }
}

extern "C" int supersonic_qwen35_hip_split_qgate(
    int dtype, size_t device_ordinal,
    size_t S, size_t num_heads, size_t head_dim,
    const void* src, void* query_out, void* gate_out
) {
    switch (dtype) {
    case 0: return split_qgate_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    case 1: return split_qgate_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    case 2: return split_qgate_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    default: return 370;
    }
}

extern "C" int supersonic_qwen35_hip_split_qkv(
    int dtype, size_t device_ordinal,
    size_t S, size_t key_dim, size_t val_dim,
    const void* src, void* Q, void* K, void* V
) {
    switch (dtype) {
    case 0: return split_qkv_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    case 1: return split_qkv_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    case 2: return split_qkv_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    default: return 380;
    }
}

extern "C" int supersonic_qwen35_hip_repeat_interleave_heads(
    int dtype, size_t device_ordinal,
    size_t S, size_t n_heads, size_t head_dim, size_t repeats,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return repeat_interleave_heads_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 1: return repeat_interleave_heads_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 2: return repeat_interleave_heads_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    default: return 390;
    }
}

extern "C" int supersonic_qwen35_hip_full_attention_decode_flat(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    switch (dtype) {
    case 0: return full_attention_decode_flat_device<half>(
                static_cast<int>(device_ordinal), static_cast<int>(batch_size),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(kv_len), static_cast<int>(head_dim),
                static_cast<int>(num_kv_groups), scale, query, key, value, out);
    case 2: return full_attention_decode_flat_device<hip_bfloat16>(
                static_cast<int>(device_ordinal), static_cast<int>(batch_size),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(kv_len), static_cast<int>(head_dim),
                static_cast<int>(num_kv_groups), scale, query, key, value, out);
    default: return 400;
    }
}
