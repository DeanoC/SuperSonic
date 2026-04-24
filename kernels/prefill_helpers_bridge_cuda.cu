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
// Bridge for prefill helper kernels.
// Separate compilation unit — does not touch the decode megakernel files.

#include "prefill_helpers_cuda.cuh"

#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedHipDevice(int target) {
        cudaGetDevice(&previous);
        if (previous != target) { cudaSetDevice(target); changed = true; }
    }
    ~ScopedHipDevice() { if (changed && previous >= 0) cudaSetDevice(previous); }
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
    if (cudaGetLastError() != cudaSuccess) return 301;
    if (cudaDeviceSynchronize() != cudaSuccess) return 302;
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
    if (cudaGetLastError() != cudaSuccess) return 311;
    if (cudaDeviceSynchronize() != cudaSuccess) return 312;
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
    if (cudaGetLastError() != cudaSuccess) return 321;
    if (cudaDeviceSynchronize() != cudaSuccess) return 322;
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
    if (cudaMemset(dst, 0, dst_bytes) != cudaSuccess) return 330;

    const size_t total = static_cast<size_t>(S) * C;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_pad_conv_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, pad,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    if (cudaGetLastError() != cudaSuccess) return 331;
    if (cudaDeviceSynchronize() != cudaSuccess) return 332;
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
    if (cudaGetLastError() != cudaSuccess) return 341;
    if (cudaDeviceSynchronize() != cudaSuccess) return 342;
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
    if (cudaGetLastError() != cudaSuccess) return 351;
    if (cudaDeviceSynchronize() != cudaSuccess) return 352;
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
    if (cudaGetLastError() != cudaSuccess) return 361;
    if (cudaDeviceSynchronize() != cudaSuccess) return 362;
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
    if (cudaGetLastError() != cudaSuccess) return 371;
    if (cudaDeviceSynchronize() != cudaSuccess) return 372;
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
    if (cudaGetLastError() != cudaSuccess) return 381;
    if (cudaDeviceSynchronize() != cudaSuccess) return 382;
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
    if (cudaGetLastError() != cudaSuccess) return 391;
    if (cudaDeviceSynchronize() != cudaSuccess) return 392;
    return 0;
}

// ---- single-row argmax (BF16 logits) ----

__global__ void pfx_argmax_bf16_kernel(
    const hip_bfloat16* __restrict__ logits,
    size_t n,
    uint32_t* __restrict__ out_index
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const unsigned int tid = threadIdx.x;
    float best_val = -1.0e30f;
    uint32_t best_idx = 0;

    for (size_t idx = tid; idx < n; idx += blockDim.x) {
        const float val = __bfloat162float(logits[idx]);
        if (val > best_val || (val == best_val && static_cast<uint32_t>(idx) < best_idx)) {
            best_val = val;
            best_idx = static_cast<uint32_t>(idx);
        }
    }

    shared_vals[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float rhs_val = shared_vals[tid + offset];
            const uint32_t rhs_idx = shared_idx[tid + offset];
            if (rhs_val > shared_vals[tid] ||
                (rhs_val == shared_vals[tid] && rhs_idx < shared_idx[tid])) {
                shared_vals[tid] = rhs_val;
                shared_idx[tid] = rhs_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_index = shared_idx[0];
    }
}

__global__ void pfx_lm_head_argmax_blocks_bf16_kernel(
    const hip_bfloat16* __restrict__ hidden,
    const hip_bfloat16* __restrict__ weight,
    int hidden_dim,
    int vocab_size,
    float* __restrict__ block_best_vals,
    uint32_t* __restrict__ block_best_idxs
) {
    extern __shared__ float shared_hidden[];
    __shared__ float warp_best_vals[8];
    __shared__ uint32_t warp_best_idxs[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int warps_per_block = blockDim.x / 32;

    for (int col = static_cast<int>(tid); col < hidden_dim; col += static_cast<int>(blockDim.x)) {
        shared_hidden[col] = __bfloat162float(hidden[col]);
    }
    __syncthreads();

    float best_val = -1.0e30f;
    uint32_t best_idx = 0;

    for (int row = static_cast<int>(blockIdx.x * warps_per_block + warp_id);
         row < vocab_size;
         row += static_cast<int>(gridDim.x * warps_per_block)) {
        const hip_bfloat16* row_weight = weight + static_cast<size_t>(row) * hidden_dim;
        float partial = 0.0f;
        for (int col = static_cast<int>(lane); col < hidden_dim; col += 32) {
            partial += __bfloat162float(row_weight[col]) * shared_hidden[col];
        }

        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }

        const float logit = __bfloat162float(__float2bfloat16(partial));
        if (lane == 0 && (logit > best_val || (logit == best_val && static_cast<uint32_t>(row) < best_idx))) {
            best_val = logit;
            best_idx = static_cast<uint32_t>(row);
        }
    }

    if (lane == 0) {
        warp_best_vals[warp_id] = best_val;
        warp_best_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_best_val = (lane < warps_per_block) ? warp_best_vals[lane] : -1.0e30f;
        uint32_t block_best_idx = (lane < warps_per_block) ? warp_best_idxs[lane] : 0;
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float rhs_val = __shfl_down_sync(0xffffffffu, block_best_val, offset);
            const uint32_t rhs_idx = __shfl_down_sync(0xffffffffu, block_best_idx, offset);
            if (rhs_val > block_best_val || (rhs_val == block_best_val && rhs_idx < block_best_idx)) {
                block_best_val = rhs_val;
                block_best_idx = rhs_idx;
            }
        }
        if (lane == 0) {
            block_best_vals[blockIdx.x] = block_best_val;
            block_best_idxs[blockIdx.x] = block_best_idx;
        }
    }
}

__global__ void pfx_argmax_blocks_kernel(
    const float* __restrict__ block_best_vals,
    const uint32_t* __restrict__ block_best_idxs,
    size_t nblocks,
    uint32_t* __restrict__ out_index
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const unsigned int tid = threadIdx.x;
    float best_val = -1.0e30f;
    uint32_t best_idx = 0;

    for (size_t idx = tid; idx < nblocks; idx += blockDim.x) {
        const float val = block_best_vals[idx];
        const uint32_t row = block_best_idxs[idx];
        if (val > best_val || (val == best_val && row < best_idx)) {
            best_val = val;
            best_idx = row;
        }
    }

    shared_vals[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float rhs_val = shared_vals[tid + offset];
            const uint32_t rhs_idx = shared_idx[tid + offset];
            if (rhs_val > shared_vals[tid] ||
                (rhs_val == shared_vals[tid] && rhs_idx < shared_idx[tid])) {
                shared_vals[tid] = rhs_val;
                shared_idx[tid] = rhs_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_index = shared_idx[0];
    }
}

__global__ void pfx_target_nll_bf16_kernel(
    const hip_bfloat16* __restrict__ logits,
    const uint32_t* __restrict__ targets,
    size_t rows,
    size_t vocab_size,
    float* __restrict__ out_nll
) {
    __shared__ float shared_vals[256];
    __shared__ float shared_target[256];

    const unsigned int tid = threadIdx.x;
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const hip_bfloat16* row_logits = logits + row * vocab_size;
    const uint32_t target = targets[row];

    float max_val = -1.0e30f;
    float target_val = 0.0f;
    for (size_t idx = tid; idx < vocab_size; idx += blockDim.x) {
        const float val = __bfloat162float(row_logits[idx]);
        max_val = fmaxf(max_val, val);
        if (idx == static_cast<size_t>(target)) {
            target_val = val;
        }
    }

    shared_vals[tid] = max_val;
    shared_target[tid] = target_val;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_vals[tid] = fmaxf(shared_vals[tid], shared_vals[tid + offset]);
            shared_target[tid] += shared_target[tid + offset];
        }
        __syncthreads();
    }

    const float row_max = shared_vals[0];
    const float row_target = shared_target[0];
    float sum_exp = 0.0f;
    for (size_t idx = tid; idx < vocab_size; idx += blockDim.x) {
        sum_exp += expf(__bfloat162float(row_logits[idx]) - row_max);
    }

    shared_vals[tid] = sum_exp;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_vals[tid] += shared_vals[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_nll[row] = row_max + logf(shared_vals[0]) - row_target;
    }
}

int argmax_bf16_device(int device_ordinal, size_t n, const void* logits, void* out_index) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    pfx_argmax_bf16_kernel<<<1, block>>>(
        static_cast<const hip_bfloat16*>(logits),
        n,
        static_cast<uint32_t*>(out_index));
    if (cudaGetLastError() != cudaSuccess) return 401;
    if (cudaDeviceSynchronize() != cudaSuccess) return 402;
    return 0;
}

int target_nll_bf16_device(
    int device_ordinal,
    size_t rows,
    size_t vocab_size,
    const void* logits,
    const void* targets,
    void* out_nll
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    pfx_target_nll_bf16_kernel<<<static_cast<unsigned int>(rows), block>>>(
        static_cast<const hip_bfloat16*>(logits),
        static_cast<const uint32_t*>(targets),
        rows,
        vocab_size,
        static_cast<float*>(out_nll));
    if (cudaGetLastError() != cudaSuccess) return 421;
    if (cudaDeviceSynchronize() != cudaSuccess) return 422;
    return 0;
}

int lm_head_argmax_bf16_device(
    int device_ordinal,
    int hidden_dim,
    int vocab_size,
    const void* hidden,
    const void* weight,
    void* block_best_vals,
    void* block_best_idxs,
    void* out_index
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    constexpr int lm_blocks = 512;
    const size_t shared_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    pfx_lm_head_argmax_blocks_bf16_kernel<<<lm_blocks, block, shared_bytes>>>(
        static_cast<const hip_bfloat16*>(hidden),
        static_cast<const hip_bfloat16*>(weight),
        hidden_dim,
        vocab_size,
        static_cast<float*>(block_best_vals),
        static_cast<uint32_t*>(block_best_idxs));
    if (cudaGetLastError() != cudaSuccess) return 411;
    pfx_argmax_blocks_kernel<<<1, block>>>(
        static_cast<const float*>(block_best_vals),
        static_cast<const uint32_t*>(block_best_idxs),
        lm_blocks,
        static_cast<uint32_t*>(out_index));
    if (cudaGetLastError() != cudaSuccess) return 412;
    if (cudaDeviceSynchronize() != cudaSuccess) return 413;
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

extern "C" int dotcache_qwen35_hip_sigmoid_mul(
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

extern "C" int dotcache_qwen35_hip_compute_beta_g(
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

extern "C" int dotcache_qwen35_hip_split_qgate(
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

extern "C" int dotcache_qwen35_hip_split_qkv(
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

extern "C" int dotcache_qwen35_hip_repeat_interleave_heads(
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

extern "C" int dotcache_qwen35_cuda_argmax_bf16(
    size_t device_ordinal,
    size_t n,
    const void* logits,
    void* out_index
) {
    return argmax_bf16_device(static_cast<int>(device_ordinal), n, logits, out_index);
}

extern "C" int dotcache_qwen35_cuda_target_nll_bf16(
    size_t device_ordinal,
    size_t rows,
    size_t vocab_size,
    const void* logits,
    const void* targets,
    void* out_nll
) {
    return target_nll_bf16_device(
        static_cast<int>(device_ordinal),
        rows,
        vocab_size,
        logits,
        targets,
        out_nll);
}

extern "C" int dotcache_qwen35_cuda_lm_head_argmax_bf16(
    size_t device_ordinal,
    size_t hidden_dim,
    size_t vocab_size,
    const void* hidden,
    const void* weight,
    void* block_best_vals,
    void* block_best_idxs,
    void* out_index
) {
    return lm_head_argmax_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(hidden_dim),
        static_cast<int>(vocab_size),
        hidden,
        weight,
        block_best_vals,
        block_best_idxs,
        out_index);
}
