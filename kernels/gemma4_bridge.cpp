// Bridge for Gemma 4 decode primitives. Separate compilation unit from the
// Qwen kernels — see gemma4.hip for rationale.

#include "gemma4.hip"

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

// ---- RMSNorm (Gemma variant: no (w+1) offset) ----

template <typename T>
int rms_norm_device(int device_ordinal, int n_cols, float eps,
                    const void* xs, const void* weight, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rms_norm_kernel<T>),
        dim3(1), dim3(block), 0, 0,
        n_cols, eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 401;
    if (hipDeviceSynchronize() != hipSuccess) return 402;
    return 0;
}

// ---- Work-stealing matvec ----

template <typename T>
int matvec_device(int device_ordinal, int in_dim, int out_dim,
                  const void* x, const void* W, void* out,
                  unsigned int* row_counter) {
    ScopedHipDevice scoped(device_ordinal);
    if (hipMemset(row_counter, 0, sizeof(unsigned int)) != hipSuccess) return 410;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 411;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_matvec_workstealing_kernel<T>),
        dim3(num_blocks), dim3(block), 0, 0,
        in_dim, out_dim,
        static_cast<const T*>(x),
        static_cast<const T*>(W),
        static_cast<T*>(out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 412;
    if (hipDeviceSynchronize() != hipSuccess) return 413;
    return 0;
}

// ---- GeLU-tanh gated multiply ----

template <typename T>
int gelu_tanh_gate_mul_device(int device_ordinal, size_t n,
                              const void* gate, const void* up, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_gelu_tanh_gate_mul_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        n,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 421;
    if (hipDeviceSynchronize() != hipSuccess) return 422;
    return 0;
}

// ---- RoPE decode (split-half Gemma style) ----

template <typename T>
int rope_decode_device(int device_ordinal,
                       int num_heads, int head_dim, int rotary_dim, int position,
                       const void* cos_table, const void* sin_table, void* x) {
    ScopedHipDevice scoped(device_ordinal);
    const int half = rotary_dim / 2;
    const size_t total = static_cast<size_t>(num_heads) * half;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rope_split_half_decode_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        num_heads, head_dim, rotary_dim, position,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(x));
    if (hipGetLastError() != hipSuccess) return 431;
    if (hipDeviceSynchronize() != hipSuccess) return 432;
    return 0;
}

// ---- KV append ----

template <typename T>
int kv_append_device(int device_ordinal,
                     int num_kv_heads, int head_dim, int pos, int max_T,
                     const void* k_in, const void* v_in,
                     void* k_cache, void* v_cache) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(num_kv_heads) * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_kv_append_decode_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        num_kv_heads, head_dim, pos, max_T,
        static_cast<const T*>(k_in),
        static_cast<const T*>(v_in),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache));
    if (hipGetLastError() != hipSuccess) return 441;
    if (hipDeviceSynchronize() != hipSuccess) return 442;
    return 0;
}

// ---- SWA (or full) attention for one decode token ----

template <typename T>
int swa_attn_decode_device(int device_ordinal,
                           int num_q_heads, int num_kv_heads,
                           int head_dim, int kv_len, int max_T,
                           int sliding_window, float scale,
                           const void* q, const void* k_cache, const void* v_cache,
                           void* scores_scratch, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (kv_len <= 0) return 450;

    constexpr int BLOCK = 256;
    // Phase 1: per-(q_head, t) score computation.
    {
        dim3 grid(num_q_heads, (kv_len + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_scores_kernel<T>),
            grid, block, 0, 0,
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            sliding_window, scale,
            static_cast<const T*>(q),
            static_cast<const T*>(k_cache),
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 451;
    }

    // Phase 2: per-q_head softmax.
    {
        dim3 grid(num_q_heads, 1, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            g4_attn_softmax_kernel,
            grid, block, 0, 0,
            num_q_heads, kv_len, max_T,
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 452;
    }

    // Phase 3: value aggregation per (q_head, head_dim col).
    {
        dim3 grid(num_q_heads, (head_dim + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_value_aggregate_kernel<T>),
            grid, block, 0, 0,
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            static_cast<const float*>(scores_scratch),
            static_cast<const T*>(v_cache),
            static_cast<T*>(out));
        if (hipGetLastError() != hipSuccess) return 453;
    }

    if (hipDeviceSynchronize() != hipSuccess) return 454;
    return 0;
}

}  // namespace

// -----------------------------------------------------------------------------
// extern "C" entry points — called from Rust via crate kernel-ffi.
// dtype encoding: 0 = __half (fp16), 1 = float (fp32), 2 = hip_bfloat16
// -----------------------------------------------------------------------------

extern "C" int dotcache_gemma4_hip_rms_norm(
    int dtype, size_t device_ordinal, size_t n_cols, float eps,
    const void* xs, const void* weight, void* out
) {
    switch (dtype) {
    case 0: return rms_norm_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 1: return rms_norm_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 2: return rms_norm_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    default: return 400;
    }
}

extern "C" int dotcache_gemma4_hip_matvec(
    int dtype, size_t device_ordinal, size_t in_dim, size_t out_dim,
    const void* x, const void* W, void* out, unsigned int* row_counter
) {
    switch (dtype) {
    case 0: return matvec_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 1: return matvec_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 2: return matvec_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    default: return 409;
    }
}

extern "C" int dotcache_gemma4_hip_gelu_tanh_gate_mul(
    int dtype, size_t device_ordinal, size_t n,
    const void* gate, const void* up, void* out
) {
    switch (dtype) {
    case 0: return gelu_tanh_gate_mul_device<__half>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 1: return gelu_tanh_gate_mul_device<float>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 2: return gelu_tanh_gate_mul_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    default: return 420;
    }
}

extern "C" int dotcache_gemma4_hip_rope_decode(
    int dtype, size_t device_ordinal,
    size_t num_heads, size_t head_dim, size_t rotary_dim, size_t position,
    const void* cos_table, const void* sin_table, void* x
) {
    switch (dtype) {
    case 0: return rope_decode_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    case 1: return rope_decode_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    case 2: return rope_decode_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    default: return 430;
    }
}

extern "C" int dotcache_gemma4_hip_swa_attn_decode(
    int dtype, size_t device_ordinal,
    size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t kv_len, size_t max_T,
    int sliding_window, float scale,
    const void* q, const void* k_cache, const void* v_cache,
    void* scores_scratch, void* out
) {
    switch (dtype) {
    case 0: return swa_attn_decode_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 1: return swa_attn_decode_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 2: return swa_attn_decode_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    default: return 449;
    }
}

extern "C" int dotcache_gemma4_hip_kv_append(
    int dtype, size_t device_ordinal,
    size_t num_kv_heads, size_t head_dim, size_t pos, size_t max_T,
    const void* k_in, const void* v_in, void* k_cache, void* v_cache
) {
    switch (dtype) {
    case 0: return kv_append_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    case 1: return kv_append_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    case 2: return kv_append_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    default: return 440;
    }
}
