#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstdio>

using hip_bfloat16 = __nv_bfloat16;
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
static inline cudaError_t supersonic_cuda_malloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
#define cudaMalloc(ptr, size) supersonic_cuda_malloc(reinterpret_cast<void**>(ptr), size)
#include "full_attention_cuda.cuh"

#include <cstdlib>
#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;

    explicit ScopedHipDevice(int target) {
        cudaGetDevice(&previous);
        if (previous != target) {
            cudaSetDevice(target);
            changed = true;
        }
    }

    ~ScopedHipDevice() {
        if (changed && previous >= 0) {
            cudaSetDevice(previous);
        }
    }
};

int linear_prefill_block_override() {
    const char* value = std::getenv("DOTCACHE_QWEN35_HIP_FUSED_PREFILL_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed < 32) {
        return 32;
    }
    if (parsed > 512) {
        return 512;
    }
    return static_cast<int>(parsed);
}

template <typename T>
int full_attention_prefill_device(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) {
        return 1;
    }

    const T* d_query = static_cast<const T*>(query);
    const T* d_key = static_cast<const T*>(key);
    const T* d_value = static_cast<const T*>(value);
    float* d_out = static_cast<float*>(out);
    unsigned int* d_row_counter = nullptr;

    if (cudaMalloc(&d_row_counter, sizeof(unsigned int)) != cudaSuccess) return 2;
    if (cudaMemset(d_row_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 10;

    const int grid = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    const int block = props.warpSize > 0 ? props.warpSize : 32;
    if (head_dim > block * 8) return 14;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_full_attention_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_kv_groups,
        scale,
        seqlen_offset,
        d_query,
        d_key,
        d_value,
        d_out,
        d_row_counter);
    if (cudaGetLastError() != cudaSuccess) return 11;
    if (cudaDeviceSynchronize() != cudaSuccess) return 12;

    cudaFree(d_row_counter);
    return 0;
}

template <typename T>
int linear_prefill_conv_pack_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int total_len,
    int seq_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_prefill_conv_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        total_len,
        seq_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 60;
    if (cudaDeviceSynchronize() != cudaSuccess) return 61;
    return 0;
}

template <typename T>
int linear_stateful_conv_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        seq_len,
        state_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_state),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 62;
    return 0;
}

template <typename T>
int linear_stateful_conv_value_decay_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t out_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * out_width;
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    if (kernel_size == 4 && state_len == 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_value_decay_kernel_k4s3<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_value_decay_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    }
    if (cudaGetLastError() != cudaSuccess) return 64;
    return 0;
}

template <typename T>
int linear_stateful_conv_value_decay_with_state_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t total_per_batch = static_cast<size_t>(seq_len) * out_width +
        static_cast<size_t>(conv_dim) * static_cast<size_t>(state_len);
    const size_t out_elems = static_cast<size_t>(batch_size) * total_per_batch;
    const int default_block =
        (kernel_size == 4 && state_len == 3) ? ((seq_len <= 4) ? 256 : 128) : 256;
    const int override_block = linear_prefill_block_override();
    const int block = override_block > 0 ? override_block : default_block;
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    if (kernel_size == 4 && state_len == 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_value_decay_with_state_kernel_k4s3<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    } else {
        return 66;
    }
    if (cudaGetLastError() != cudaSuccess) return 67;
    return 0;
}

template <typename T>
int linear_decode_prepare_device(
    int device_ordinal,
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int state_len,
    int kernel_size,
    int head_repeat,
    const void* mixed_qkv,
    const void* prev_conv_state,
    const void* weights,
    const void* a_beta_raw,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const unsigned int grid =
        static_cast<unsigned int>(batch_size) * static_cast<unsigned int>(num_v_heads);
    unsigned int block = 64;
    while (block < static_cast<unsigned int>(head_k_dim > head_v_dim ? head_k_dim : head_v_dim) &&
           block < 256) {
        block <<= 1;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_decode_prepare_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_len,
        kernel_size,
        head_repeat,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_conv_state),
        static_cast<const T*>(weights),
        static_cast<const T*>(a_beta_raw),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<float*>(out));
    if (cudaGetLastError() != cudaSuccess) return 69;
    if (cudaDeviceSynchronize() != cudaSuccess) return 70;
    return 0;
}

int linear_decode_apply_device(
    int device_ordinal,
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const void* packed,
    const void* initial_state,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const unsigned int grid =
        static_cast<unsigned int>(batch_size) * static_cast<unsigned int>(num_v_heads);
    unsigned int block = 64;
    while (block < static_cast<unsigned int>(head_v_dim) && block < 256) {
        block <<= 1;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_decode_apply_kernel<>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        static_cast<const float*>(packed),
        static_cast<const float*>(initial_state),
        static_cast<float*>(out));
    if (cudaGetLastError() != cudaSuccess) return 71;
    if (cudaDeviceSynchronize() != cudaSuccess) return 72;
    return 0;
}

template <typename T>
int delta_recurrent_prefill_device(
    int device_ordinal,
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_recurrent_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 67;
    if (cudaDeviceSynchronize() != cudaSuccess) return 68;
    return 0;
}

template <typename T>
int delta_chunk_single_prefill_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 76;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_single_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 77;
    if (cudaDeviceSynchronize() != cudaSuccess) return 78;
    return 0;
}

template <typename T>
int delta_chunk_step_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 80;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_step_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 81;
    if (cudaDeviceSynchronize() != cudaSuccess) return 82;
    return 0;
}

template <typename T>
int delta_chunk_scan_raw_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 83;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_scan_raw_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 84;
    if (cudaDeviceSynchronize() != cudaSuccess) return 85;
    return 0;
}

template <typename T>
int delta_state_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 88;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_state_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 89;
    if (cudaDeviceSynchronize() != cudaSuccess) return 96;
    return 0;
}

template <typename T>
int delta_chunk_fused_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 97;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_fused_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(packed_chunk),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 98;
    if (cudaDeviceSynchronize() != cudaSuccess) return 99;
    return 0;
}

template <typename T>
int delta_full_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 100;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(weighted_key_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<const T*>(q_state_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(state_decay_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 101;
    if (cudaDeviceSynchronize() != cudaSuccess) return 102;
    return 0;
}

template <typename T>
int delta_local_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 112;
    if (chunk_size <= 4) {
        constexpr int block = 256;
        const size_t total =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_delta_local_attn_scan_flat_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    } else {
        const unsigned int block = chunk_size <= 32 ? 32u : 64u;
        const size_t total_rows =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>(total_rows);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_delta_local_attn_scan_row_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    }
    if (cudaGetLastError() != cudaSuccess) return 113;
    if (cudaDeviceSynchronize() != cudaSuccess) return 114;
    return 0;
}

template <typename T>
int delta_base_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 115;
    constexpr int block = 256;
    const size_t total =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_base_attn_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 116;
    if (cudaDeviceSynchronize() != cudaSuccess) return 117;
    return 0;
}

template <typename T>
int delta_attn_solve_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    const void* base_attn_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64) return 118;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_attn_solve_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        static_cast<const T*>(base_attn_scan),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 119;
    if (cudaDeviceSynchronize() != cudaSuccess) return 120;
    return 0;
}

template <typename T>
int delta_attn_solve_from_inputs_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 121;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_attn_solve_from_inputs_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 122;
    if (cudaDeviceSynchronize() != cudaSuccess) return 123;
    return 0;
}

template <typename T>
int swiglu_mul_device(
    int device_ordinal,
    int elem_count,
    const void* gate,
    const void* up,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((elem_count + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_swiglu_mul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        elem_count,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 121;
    if (cudaDeviceSynchronize() != cudaSuccess) return 122;
    return 0;
}

template <typename T, typename IndexT>
int embedding_lookup_device(
    int device_ordinal,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total_elems = token_count * hidden_size;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_embedding_lookup_kernel<T, IndexT>),
        dim3(grid),
        dim3(block),
        0,
        0,
        token_count,
        vocab_size,
        hidden_size,
        static_cast<const T*>(embeddings),
        static_cast<const IndexT*>(indexes),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 123;
    if (cudaDeviceSynchronize() != cudaSuccess) return 124;
    return 0;
}

template <typename T>
int causal_mask_device(
    int device_ordinal,
    int batch_size,
    int tgt_len,
    int seqlen_offset,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int kv_len = tgt_len + seqlen_offset;
    const int total_elems = batch_size * tgt_len * kv_len;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_causal_mask_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        tgt_len,
        seqlen_offset,
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 125;
    if (cudaDeviceSynchronize() != cudaSuccess) return 126;
    return 0;
}

template <typename T>
int cumsum_last_dim_device(
    int device_ordinal,
    int rows,
    int cols,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_cumsum_last_dim_kernel<T>),
        dim3(static_cast<unsigned int>(rows)),
        dim3(1),
        0,
        0,
        rows,
        cols,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 127;
    if (cudaDeviceSynchronize() != cudaSuccess) return 128;
    return 0;
}

template <typename T>
int exp_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_exp_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 129;
    if (cudaDeviceSynchronize() != cudaSuccess) return 130;
    return 0;
}

template <typename T>
int recip_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_recip_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 131;
    if (cudaDeviceSynchronize() != cudaSuccess) return 132;
    return 0;
}

template <typename T>
int sigmoid_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_sigmoid_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 133;
    if (cudaDeviceSynchronize() != cudaSuccess) return 134;
    return 0;
}

template <typename T>
int log_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_log_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 155;
    if (cudaDeviceSynchronize() != cudaSuccess) return 156;
    return 0;
}

template <typename In, typename Out>
int cast_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_cast_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const In*>(xs),
        static_cast<Out*>(out));
    if (cudaGetLastError() != cudaSuccess) return 135;
    if (cudaDeviceSynchronize() != cudaSuccess) return 136;
    return 0;
}

template <typename T>
int unary_view_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (cudaMalloc(&in_strides_dev, bytes) != cudaSuccess) return 158;
        if (cudaMalloc(&out_dims_dev, bytes) != cudaSuccess) {
            cudaFree(in_strides_dev);
            return 158;
        }
        if (cudaMemcpy(in_strides_dev, in_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_dims_dev, out_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
            return 158;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_unary_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 159;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 160;
    }
    if (rank > 0) {
        cudaFree(in_strides_dev);
        cudaFree(out_dims_dev);
    }
    return 0;
}

template <typename In, typename Out>
int cast_view_device(
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (cudaMalloc(&in_strides_dev, bytes) != cudaSuccess) return 161;
        if (cudaMalloc(&out_dims_dev, bytes) != cudaSuccess) {
            cudaFree(in_strides_dev);
            return 161;
        }
        if (cudaMemcpy(in_strides_dev, in_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_dims_dev, out_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
            return 161;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_cast_view_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        total_elems,
        static_cast<const In*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<Out*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 162;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 163;
    }
    if (rank > 0) {
        cudaFree(in_strides_dev);
        cudaFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int binary_broadcast_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_strides_dev = nullptr;
    int* rhs_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (cudaMalloc(&lhs_strides_dev, bytes) != cudaSuccess) return 137;
    if (cudaMalloc(&rhs_strides_dev, bytes) != cudaSuccess) {
        cudaFree(lhs_strides_dev);
        return 137;
    }
    if (cudaMalloc(&out_dims_dev, bytes) != cudaSuccess) {
        cudaFree(lhs_strides_dev);
        cudaFree(rhs_strides_dev);
        return 137;
    }
    if (cudaMemcpy(lhs_strides_dev, lhs_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(rhs_strides_dev, rhs_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(out_dims_dev, out_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(lhs_strides_dev);
        cudaFree(rhs_strides_dev);
        cudaFree(out_dims_dev);
        return 137;
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_binary_broadcast_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        lhs_strides_dev,
        rhs_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(lhs_strides_dev);
        cudaFree(rhs_strides_dev);
        cudaFree(out_dims_dev);
        return 138;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(lhs_strides_dev);
        cudaFree(rhs_strides_dev);
        cudaFree(out_dims_dev);
        return 139;
    }
    cudaFree(lhs_strides_dev);
    cudaFree(rhs_strides_dev);
    cudaFree(out_dims_dev);
    return 0;
}

template <typename T>
int reduce_keepdim_view_device(
    int device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (cudaMalloc(&in_strides_dev, bytes) != cudaSuccess) return 167;
        if (cudaMalloc(&out_dims_dev, bytes) != cudaSuccess) {
            cudaFree(in_strides_dev);
            return 167;
        }
        if (cudaMemcpy(in_strides_dev, in_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_dims_dev, out_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
            return 167;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_out_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_reduce_keepdim_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        reduce_dim,
        reduce_len,
        total_out_elems,
        sum,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 168;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (rank > 0) {
            cudaFree(in_strides_dev);
            cudaFree(out_dims_dev);
        }
        return 169;
    }
    if (rank > 0) {
        cudaFree(in_strides_dev);
        cudaFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_dims_dev = nullptr;
    int* rhs_batch_dims_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (cudaMalloc(&lhs_batch_dims_dev, bytes) != cudaSuccess) return 141;
        if (cudaMalloc(&rhs_batch_dims_dev, bytes) != cudaSuccess) {
            cudaFree(lhs_batch_dims_dev);
            return 141;
        }
        if (cudaMalloc(&out_batch_dims_dev, bytes) != cudaSuccess) {
            cudaFree(lhs_batch_dims_dev);
            cudaFree(rhs_batch_dims_dev);
            return 141;
        }
        if (cudaMemcpy(lhs_batch_dims_dev, lhs_batch_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(rhs_batch_dims_dev, rhs_batch_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_batch_dims_dev, out_batch_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(lhs_batch_dims_dev);
            cudaFree(rhs_batch_dims_dev);
            cudaFree(out_batch_dims_dev);
            return 141;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_batched_matmul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_dims_dev,
        rhs_batch_dims_dev,
        out_batch_dims_dev,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        if (batch_rank > 0) {
            cudaFree(lhs_batch_dims_dev);
            cudaFree(rhs_batch_dims_dev);
            cudaFree(out_batch_dims_dev);
        }
        return 142;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (batch_rank > 0) {
            cudaFree(lhs_batch_dims_dev);
            cudaFree(rhs_batch_dims_dev);
            cudaFree(out_batch_dims_dev);
        }
        return 143;
    }
    if (batch_rank > 0) {
        cudaFree(lhs_batch_dims_dev);
        cudaFree(rhs_batch_dims_dev);
        cudaFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_view_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_strides_dev = nullptr;
    int* rhs_batch_strides_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (cudaMalloc(&lhs_batch_strides_dev, bytes) != cudaSuccess) return 171;
        if (cudaMalloc(&rhs_batch_strides_dev, bytes) != cudaSuccess) {
            cudaFree(lhs_batch_strides_dev);
            return 171;
        }
        if (cudaMalloc(&out_batch_dims_dev, bytes) != cudaSuccess) {
            cudaFree(lhs_batch_strides_dev);
            cudaFree(rhs_batch_strides_dev);
            return 171;
        }
        if (cudaMemcpy(lhs_batch_strides_dev, lhs_batch_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(rhs_batch_strides_dev, rhs_batch_strides, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_batch_dims_dev, out_batch_dims, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(lhs_batch_strides_dev);
            cudaFree(rhs_batch_strides_dev);
            cudaFree(out_batch_dims_dev);
            return 171;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_batched_matmul_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_strides_dev,
        rhs_batch_strides_dev,
        out_batch_dims_dev,
        lhs_row_stride,
        lhs_k_stride,
        rhs_k_stride,
        rhs_col_stride,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) {
        if (batch_rank > 0) {
            cudaFree(lhs_batch_strides_dev);
            cudaFree(rhs_batch_strides_dev);
            cudaFree(out_batch_dims_dev);
        }
        return 172;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (batch_rank > 0) {
            cudaFree(lhs_batch_strides_dev);
            cudaFree(rhs_batch_strides_dev);
            cudaFree(out_batch_dims_dev);
        }
        return 173;
    }
    if (batch_rank > 0) {
        cudaFree(lhs_batch_strides_dev);
        cudaFree(rhs_batch_strides_dev);
        cudaFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int mul_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_mul_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 145;
    if (cudaDeviceSynchronize() != cudaSuccess) return 146;
    return 0;
}

template <typename T>
int reduce_keepdim_device(
    int device_ordinal,
    int outer,
    int reduce,
    int inner,
    bool sum,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total = outer * inner;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_reduce_keepdim_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        outer,
        reduce,
        inner,
        sum ? 1 : 0,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 147;
    if (cudaDeviceSynchronize() != cudaSuccess) return 148;
    return 0;
}

template <typename T>
int add_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_add_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 149;
    if (cudaDeviceSynchronize() != cudaSuccess) return 150;
    return 0;
}

template <typename T>
int sqrt_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_sqrt_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 151;
    if (cudaDeviceSynchronize() != cudaSuccess) return 152;
    return 0;
}

template <typename T>
int delta_full_scan_pack_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 106;
    constexpr int block = 256;
    const size_t total_rows =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total_rows + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(query_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 107;
    if (cudaDeviceSynchronize() != cudaSuccess) return 108;
    return 0;
}

template <typename T>
int delta_full_scan_packed_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 109;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_packed_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 110;
    if (cudaDeviceSynchronize() != cudaSuccess) return 111;
    return 0;
}

template <typename T>
int l2norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_l2norm_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 90;
    if (cudaDeviceSynchronize() != cudaSuccess) return 91;
    return 0;
}

template <typename T>
int value_decay_device(
    int device_ordinal,
    int total_elems,
    int num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_value_decay_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        num_heads,
        static_cast<const T*>(a),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 93;
    if (cudaDeviceSynchronize() != cudaSuccess) return 94;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int rms_norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_rms_norm_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 71;
    if (cudaDeviceSynchronize() != cudaSuccess) return 72;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int fused_rms_norm_linear_device(
    int device_ordinal,
    int hidden_dim,
    int out_dim,
    float eps,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) + block * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_fused_rms_norm_linear_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(out_dim)),
        dim3(block),
        shared_bytes,
        0,
        hidden_dim,
        out_dim,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(proj_weight),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 130;
    if (cudaDeviceSynchronize() != cudaSuccess) return 131;
    return 0;
}

template <typename T>
int rms_norm_gated_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_rms_norm_gated_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(gate),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 81;
    if (cudaDeviceSynchronize() != cudaSuccess) return 82;
    return 0;
}

} // namespace

extern "C" int dotcache_qwen35_hip_full_attention_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return full_attention_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 1:
        return full_attention_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 2:
        return full_attention_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    default:
        return 64;
    }
}

extern "C" int dotcache_qwen35_hip_linear_prefill_conv_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_prefill_conv_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 1:
        return linear_prefill_conv_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 2:
        return linear_prefill_conv_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    default:
        return 62;
    }
}

extern "C" int dotcache_qwen35_hip_linear_stateful_conv(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 1:
        return linear_stateful_conv_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 2:
        return linear_stateful_conv_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    default:
        return 63;
    }
}

extern "C" int dotcache_qwen35_hip_delta_recurrent_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_recurrent_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_recurrent_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_recurrent_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 66;
    }
}

extern "C" int dotcache_qwen35_hip_linear_stateful_conv_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_stateful_conv_value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_stateful_conv_value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 67;
    }
}

extern "C" int dotcache_qwen35_hip_linear_stateful_conv_value_decay_with_state(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_value_decay_with_state_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_stateful_conv_value_decay_with_state_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_stateful_conv_value_decay_with_state_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 68;
    }
}

extern "C" int dotcache_qwen35_hip_linear_decode_prepare(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t num_v_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    size_t state_len,
    size_t kernel_size,
    size_t head_repeat,
    const void* mixed_qkv,
    const void* prev_conv_state,
    const void* weights,
    const void* a_beta_raw,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_decode_prepare_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_decode_prepare_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_decode_prepare_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 73;
    }
}

extern "C" int dotcache_qwen35_hip_linear_decode_apply(
    size_t device_ordinal,
    size_t batch_size,
    size_t num_v_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    const void* packed,
    const void* initial_state,
    void* out) {
    return linear_decode_apply_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(batch_size),
        static_cast<int>(num_v_heads),
        static_cast<int>(head_k_dim),
        static_cast<int>(head_v_dim),
        packed,
        initial_state,
        out);
}

extern "C" int dotcache_qwen35_hip_delta_chunk_single_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_single_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_single_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_single_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 79;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_step(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_step_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_step_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_step_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 86;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_scan_raw(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_scan_raw_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_scan_raw_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_scan_raw_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 87;
    }
}

extern "C" int dotcache_qwen35_hip_delta_state_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_state_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 1:
        return delta_state_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 2:
        return delta_state_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    default:
        return 103;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_fused(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_fused_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 1:
        return delta_chunk_fused_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 2:
        return delta_chunk_fused_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    default:
        return 104;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 1:
        return delta_full_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 2:
        return delta_full_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    default:
        return 105;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 1:
        return delta_full_scan_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 2:
        return delta_full_scan_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    default:
        return 112;
    }
}

extern "C" int dotcache_qwen35_hip_delta_local_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_local_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_local_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_local_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 114;
    }
}

extern "C" int dotcache_qwen35_hip_delta_base_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_base_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_base_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_base_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 117;
    }
}

extern "C" int dotcache_qwen35_hip_delta_attn_solve_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    const void* base_attn_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 1:
        return delta_attn_solve_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 2:
        return delta_attn_solve_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    default:
        return 120;
    }
}

extern "C" int dotcache_qwen35_hip_delta_attn_solve_from_inputs(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_from_inputs_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_attn_solve_from_inputs_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_attn_solve_from_inputs_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 123;
    }
}

extern "C" int dotcache_qwen35_hip_swiglu_mul(
    int dtype,
    size_t device_ordinal,
    size_t elem_count,
    const void* gate,
    const void* up,
    void* out) {
    switch (dtype) {
    case 0:
        return swiglu_mul_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 1:
        return swiglu_mul_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 2:
        return swiglu_mul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    default:
        return 122;
    }
}

extern "C" int dotcache_qwen35_hip_embedding_lookup(
    int dtype,
    int index_dtype,
    size_t device_ordinal,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out) {
    switch (dtype) {
    case 0:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<half, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<half, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<half, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 1:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<float, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<float, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<float, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 2:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<hip_bfloat16, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<hip_bfloat16, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<hip_bfloat16, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    default:
        return 124;
    }
}

template <typename T>
int output_projection_lookup_device(
    int device_ordinal,
    int rows,
    int hidden_size,
    int vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    ScopedHipDevice scoped(device_ordinal);
    const int total_elems = rows * vocab_size;
    const int block = 256;
    const int grid = (total_elems + block - 1) / block;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_output_projection_lookup_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        hidden_size,
        vocab_size,
        static_cast<const T*>(hidden),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 11;
    return 0;
}

extern "C" int dotcache_qwen35_hip_output_projection_lookup(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t hidden_size,
    size_t vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return output_projection_lookup_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 1:
        return output_projection_lookup_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 2:
        return output_projection_lookup_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    default:
        return 122;
    }
}

extern "C" int dotcache_qwen35_hip_causal_mask(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t tgt_len,
    size_t seqlen_offset,
    void* out) {
    switch (dtype) {
    case 0:
        return causal_mask_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 1:
        return causal_mask_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 2:
        return causal_mask_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    default:
        return 126;
    }
}

extern "C" int dotcache_qwen35_hip_cumsum_last_dim(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return cumsum_last_dim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 1:
        return cumsum_last_dim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 2:
        return cumsum_last_dim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    default:
        return 128;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan_packed(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_packed_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 1:
        return delta_full_scan_packed_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 2:
        return delta_full_scan_packed_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    default:
        return 113;
    }
}

extern "C" int dotcache_qwen35_hip_exp(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return exp_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return exp_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return exp_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 129;
    }
}

extern "C" int dotcache_qwen35_hip_recip(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return recip_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return recip_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return recip_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 131;
    }
}

extern "C" int dotcache_qwen35_hip_sigmoid(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sigmoid_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sigmoid_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sigmoid_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 133;
    }
}

extern "C" int dotcache_qwen35_hip_log(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return log_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return log_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return log_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 157;
    }
}

extern "C" int dotcache_qwen35_hip_unary_view(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return unary_view_device<half>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 1:
        return unary_view_device<float>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 2:
        return unary_view_device<hip_bfloat16>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    default:
        return 164;
    }
}

extern "C" int dotcache_qwen35_hip_cast_view(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_view_device<half, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<half, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_view_device<float, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<float, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_view_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    default:
        return 166;
    }
}

extern "C" int dotcache_qwen35_hip_reduce_keepdim_view(
    int dtype,
    size_t device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_view_device<half>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 1:
        return reduce_keepdim_view_device<float>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 2:
        return reduce_keepdim_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    default:
        return 170;
    }
}

extern "C" int dotcache_qwen35_hip_batched_matmul_view(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out) {
    switch (dtype) {
    case 0:
        return batched_matmul_view_device<half>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 1:
        return batched_matmul_view_device<float>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 2:
        return batched_matmul_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    default:
        return 174;
    }
}

extern "C" int dotcache_qwen35_hip_cast(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_device<half, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<half, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_device<float, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<float, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    default:
        return 135;
    }
}

extern "C" int dotcache_qwen35_hip_binary_broadcast(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return binary_broadcast_device<half>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 1:
        return binary_broadcast_device<float>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 2:
        return binary_broadcast_device<hip_bfloat16>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    default:
        return 140;
    }
}

extern "C" int dotcache_qwen35_hip_batched_matmul(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    switch (dtype) {
    case 0:
        return batched_matmul_device<half>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 1:
        return batched_matmul_device<float>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 2:
        return batched_matmul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    default:
        return 144;
    }
}

extern "C" int dotcache_qwen35_hip_mul_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return mul_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return mul_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return mul_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 147;
    }
}

extern "C" int dotcache_qwen35_hip_reduce_keepdim(
    int dtype,
    size_t device_ordinal,
    size_t outer,
    size_t reduce,
    size_t inner,
    int sum,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 1:
        return reduce_keepdim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 2:
        return reduce_keepdim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    default:
        return 149;
    }
}

extern "C" int dotcache_qwen35_hip_add_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return add_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return add_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return add_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 153;
    }
}

extern "C" int dotcache_qwen35_hip_sqrt(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sqrt_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sqrt_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sqrt_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 154;
    }
}

extern "C" int dotcache_qwen35_hip_l2norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return l2norm_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 1:
        return l2norm_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 2:
        return l2norm_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    default:
        return 92;
    }
}

extern "C" int dotcache_qwen35_hip_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    size_t num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 95;
    }
}

extern "C" int dotcache_qwen35_hip_rms_norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? rms_norm_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 1:
        return add_unit_offset
            ? rms_norm_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 2:
        return add_unit_offset
            ? rms_norm_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    default:
        return 74;
    }
}

extern "C" int dotcache_qwen35_hip_fused_rms_norm_linear(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t out_dim,
    float eps,
    int add_unit_offset,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? fused_rms_norm_linear_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 1:
        return add_unit_offset
            ? fused_rms_norm_linear_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 2:
        return add_unit_offset
            ? fused_rms_norm_linear_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    default:
        return 132;
    }
}

extern "C" int dotcache_qwen35_hip_rms_norm_gated(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return rms_norm_gated_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 1:
        return rms_norm_gated_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 2:
        return rms_norm_gated_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    default:
        return 84;
    }
}

template <typename T>
int mlp_decode_megakernel_device(
    int device_ordinal,
    int hidden_dim,
    int intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 200;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 +  // hidden + normed
        block_size * sizeof(float);                              // scratch

    // --- Phase 1: RMSNorm + gate/up projections ---
    unsigned int zero = 0;
    if (cudaMemcpy(row_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        return 201;
    if (cudaDeviceSynchronize() != cudaSuccess) return 202;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_mlp_decode_megakernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        intermediate_size,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(gate_proj_w),
        static_cast<const T*>(up_proj_w),
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 203;
    if (cudaDeviceSynchronize() != cudaSuccess) return 204;

    // --- Phase 2: SwiGLU activation ---
    {
        constexpr int swiglu_block = 256;
        const unsigned int swiglu_grid =
            static_cast<unsigned int>((intermediate_size + swiglu_block - 1) / swiglu_block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(dotcache_qwen35_mlp_swiglu_kernel<T>),
            dim3(swiglu_grid),
            dim3(swiglu_block),
            0, 0,
            intermediate_size,
            gate_up_scratch);
        if (cudaGetLastError() != cudaSuccess) return 205;
        if (cudaDeviceSynchronize() != cudaSuccess) return 206;
    }

    // --- Phase 3: down_proj matvec ---
    if (cudaMemcpy(row_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        return 207;
    if (cudaDeviceSynchronize() != cudaSuccess) return 208;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_mlp_down_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        block_size * sizeof(float),
        0,
        hidden_dim,
        intermediate_size,
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 209;
    if (cudaDeviceSynchronize() != cudaSuccess) return 210;
    return 0;
}

extern "C" int dotcache_qwen35_hip_mlp_decode_megakernel(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return mlp_decode_megakernel_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    case 2:
        return mlp_decode_megakernel_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    default:
        return 205;
    }
}

template <typename T>
int norm_multi_proj_device(
    int device_ordinal,
    int hidden_dim,
    int total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const Qwen35ProjectionDesc* proj_table,
    int num_projections,
    float* output,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 220;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 + block_size * sizeof(float);

    unsigned int zero = 0;
    if (cudaMemcpy(row_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        return 221;
    if (cudaDeviceSynchronize() != cudaSuccess) return 222;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_norm_multi_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        total_rows,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        proj_table,
        num_projections,
        output,
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 223;
    if (cudaDeviceSynchronize() != cudaSuccess) return 224;
    return 0;
}

extern "C" int dotcache_qwen35_hip_norm_multi_proj(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* proj_table,       // Qwen35ProjectionDesc* on device
    size_t num_projections,
    float* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return norm_multi_proj_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    case 2:
        return norm_multi_proj_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    default:
        return 225;
    }
}

// Standalone work-stealing matvec: out[out_dim] = W[out_dim, in_dim] × input[in_dim]
// Reuses the down_proj kernel pattern for arbitrary matvec.
template <typename T>
int standalone_matvec_device(
    int device_ordinal,
    int in_dim,
    int out_dim,
    const void* input,       // [in_dim] F32
    const void* weight,      // [out_dim, in_dim] BF16
    void* output,            // [out_dim] BF16
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 230;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    unsigned int zero = 0;
    if (cudaMemcpy(row_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        return 231;
    if (cudaDeviceSynchronize() != cudaSuccess) return 232;

    const size_t shared_bytes = block_size * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_standalone_matvec_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        out_dim,
        in_dim,
        static_cast<const T*>(weight),
        static_cast<const T*>(input),
        static_cast<T*>(output),
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 233;
    if (cudaDeviceSynchronize() != cudaSuccess) return 234;
    return 0;
}

extern "C" int dotcache_qwen35_hip_standalone_matvec(
    int dtype,
    size_t device_ordinal,
    size_t in_dim,
    size_t out_dim,
    const void* input,
    const void* weight,
    void* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return standalone_matvec_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    case 2:
        return standalone_matvec_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    default:
        return 235;
    }
}

template <typename T>
int persistent_decode_device(
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
    int rotary_dim
) {
    ScopedHipDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 250;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    // LDS layout: [block_size] reduction scratch + [intermediate_size] input vector cache
    const size_t shared_bytes = (block_size + intermediate_size) * sizeof(float);

    // Barrier counters zeroed on first allocation by Rust PersistentDecodeCache.
    // Grid barrier self-resets between uses (barrier_counter reset by last block,
    // barrier_flag monotonically increments and wraps safely).

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_persistent_decode_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        num_layers,
        hidden_dim,
        intermediate_size,
        seqlen_offset,
        static_cast<const Qwen35DecodeLayerDesc*>(layers),
        static_cast<T*>(hidden_io),
        workspace,
        counters,
        barrier_counter,
        barrier_flag,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        rotary_dim);
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess) return 254;
    if (sync_err != cudaSuccess) return 255;
    return 0;
}

extern "C" int dotcache_qwen35_hip_persistent_decode(
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
    size_t rotary_dim) {
    switch (dtype) {
    case 2:
        return persistent_decode_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(num_layers),
            static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size),
            static_cast<int>(seqlen_offset),
            layers, hidden_io, workspace, counters,
            barrier_counter, barrier_flag,
            cos_table, sin_table, static_cast<int>(rotary_dim));
    default:
        return 256;
    }
}

extern "C" int dotcache_query_gpu_info(
    int device_ordinal,
    char* arch_name_out,
    size_t arch_name_len,
    uint64_t* total_vram_out) {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_ordinal);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    snprintf(arch_name_out, arch_name_len, "sm%d%d", props.major, props.minor);
    *total_vram_out = static_cast<uint64_t>(props.totalGlobalMem);
    return 0;
}
