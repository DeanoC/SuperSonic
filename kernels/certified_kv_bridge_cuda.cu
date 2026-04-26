#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

struct ScopedCudaDevice {
    int previous = -1;
    bool changed = false;

    explicit ScopedCudaDevice(int target) {
        cudaGetDevice(&previous);
        if (previous != target) {
            cudaSetDevice(target);
            changed = true;
        }
    }

    ~ScopedCudaDevice() {
        if (changed && previous >= 0) {
            cudaSetDevice(previous);
        }
    }
};

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
    return static_cast<float>(value);
}

__global__ void certified_kv_copy_step_bf16_kernel(
    const __nv_bfloat16* src_key,
    const __nv_bfloat16* src_value,
    __nv_bfloat16* dst_key,
    __nv_bfloat16* dst_value,
    int kv_heads,
    int dst_stride_tokens,
    int dst_token,
    int head_dim
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kv_heads * head_dim;
    if (linear >= total) return;
    const int d = linear % head_dim;
    const int kvh = linear / head_dim;
    const size_t src_idx = (static_cast<size_t>(kvh) * head_dim) + d;
    const size_t dst_idx = (static_cast<size_t>(kvh) * dst_stride_tokens + dst_token) * head_dim + d;
    dst_key[dst_idx] = src_key[src_idx];
    dst_value[dst_idx] = src_value[src_idx];
}

__global__ void certified_kv_copy_token_range_bf16_kernel(
    const __nv_bfloat16* src_key,
    const __nv_bfloat16* src_value,
    __nv_bfloat16* dst_key,
    __nv_bfloat16* dst_value,
    int kv_heads,
    int src_stride_tokens,
    int src_start_token,
    int dst_stride_tokens,
    int dst_start_token,
    int token_count,
    int head_dim
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kv_heads * token_count * head_dim;
    if (linear >= total) return;
    const int d = linear % head_dim;
    const int tok = (linear / head_dim) % token_count;
    const int kvh = linear / (token_count * head_dim);
    const size_t src_idx =
        (static_cast<size_t>(kvh) * src_stride_tokens + src_start_token + tok) * head_dim + d;
    const size_t dst_idx =
        (static_cast<size_t>(kvh) * dst_stride_tokens + dst_start_token + tok) * head_dim + d;
    dst_key[dst_idx] = src_key[src_idx];
    dst_value[dst_idx] = src_value[src_idx];
}

__device__ __forceinline__ bool certified_kv_should_run(const uint32_t* run_flag) {
    return run_flag == nullptr || run_flag[0] != 0;
}

__device__ __forceinline__ int clamp_i32(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__device__ __forceinline__ float block_reduce_max_256(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum_256(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ void atomic_max_nonnegative_float(float* address, float value) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

__global__ void certified_kv_quantize_keys_kernel(
    const __nv_bfloat16* key,
    uint8_t* key_int8,
    float* key_scale,
    float* key_zero,
    int num_kv_heads,
    int max_t,
    int aligned_tokens,
    int head_dim,
    int block_size
) {
    const int linear = blockIdx.x;
    const int dim = linear % head_dim;
    const int block_id = (linear / head_dim) % (aligned_tokens / block_size);
    const int kvh = linear / (head_dim * (aligned_tokens / block_size));
    if (kvh >= num_kv_heads) return;

    extern __shared__ float scratch[];
    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        const float x = bf16_to_float(key[src_idx]);
        local_min = fminf(local_min, x);
        local_max = fmaxf(local_max, x);
    }
    scratch[threadIdx.x] = local_min;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fminf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float block_min = scratch[0];

    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(scratch[0] - block_min, 1.0e-8f) / 255.0f;
    const float zero = block_min + 128.0f * scale;
    if (threadIdx.x == 0) {
        const size_t scale_idx = (static_cast<size_t>(kvh) * (aligned_tokens / block_size) + block_id) * head_dim + dim;
        key_scale[scale_idx] = scale;
        key_zero[scale_idx] = zero;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        const float x = bf16_to_float(key[src_idx]);
        const int q = clamp_i32(static_cast<int>(nearbyintf((x - zero) / scale)), -128, 127);
        const size_t dst_idx = (static_cast<size_t>(kvh) * aligned_tokens + tok) * head_dim + dim;
        key_int8[dst_idx] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

__global__ void certified_kv_quantize_keys_range_kernel(
    const __nv_bfloat16* key,
    uint8_t* key_int8,
    float* key_scale,
    float* key_zero,
    int num_kv_heads,
    int max_t,
    int key_stride_tokens,
    int scale_stride_blocks,
    int start_block,
    int block_count,
    int head_dim,
    int block_size
) {
    const int linear = blockIdx.x;
    const int dim = linear % head_dim;
    const int local_block = (linear / head_dim) % block_count;
    const int kvh = linear / (head_dim * block_count);
    if (kvh >= num_kv_heads) return;
    const int block_id = start_block + local_block;

    extern __shared__ float scratch[];
    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        const float x = bf16_to_float(key[src_idx]);
        local_min = fminf(local_min, x);
        local_max = fmaxf(local_max, x);
    }
    scratch[threadIdx.x] = local_min;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fminf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float block_min = scratch[0];

    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(scratch[0] - block_min, 1.0e-8f) / 255.0f;
    const float zero = block_min + 128.0f * scale;
    if (threadIdx.x == 0) {
        const size_t scale_idx =
            (static_cast<size_t>(kvh) * scale_stride_blocks + block_id) * head_dim + dim;
        key_scale[scale_idx] = scale;
        key_zero[scale_idx] = zero;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        const float x = bf16_to_float(key[src_idx]);
        const int q = clamp_i32(static_cast<int>(nearbyintf((x - zero) / scale)), -128, 127);
        const size_t dst_idx = (static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + dim;
        key_int8[dst_idx] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

__global__ void certified_kv_quantize_values_kernel(
    const __nv_bfloat16* value,
    uint8_t* value_int4,
    __half* value_scale,
    __half* value_zero,
    float* value_error,
    float* value_norm,
    int num_kv_heads,
    int max_t,
    int aligned_tokens,
    int head_dim,
    int block_size,
    int value_group_size
) {
    const int token_linear = blockIdx.x;
    const int tok = token_linear % aligned_tokens;
    const int kvh = token_linear / aligned_tokens;
    if (kvh >= num_kv_heads) return;

    constexpr int kMaxHeadDim = 256;
    constexpr int kMaxGroups = 64;
    __shared__ float group_scale[kMaxGroups];
    __shared__ float group_zero[kMaxGroups];
    __shared__ uint8_t qvals[kMaxHeadDim];
    __shared__ float err_scratch[kMaxHeadDim];
    __shared__ float norm_scratch[kMaxHeadDim];

    const int num_groups = head_dim / value_group_size;
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        float min_v = INFINITY;
        float max_v = -INFINITY;
        const int d0 = g * value_group_size;
        for (int i = 0; i < value_group_size; ++i) {
            const int d = d0 + i;
            const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + d;
            const float x = bf16_to_float(value[src_idx]);
            min_v = fminf(min_v, x);
            max_v = fmaxf(max_v, x);
        }
        const float scale = fmaxf(max_v - min_v, 1.0e-8f) / 15.0f;
        group_scale[g] = scale;
        group_zero[g] = min_v;
        const size_t meta_idx = (static_cast<size_t>(kvh) * aligned_tokens + tok) * num_groups + g;
        value_scale[meta_idx] = __float2half(scale);
        value_zero[meta_idx] = __float2half(min_v);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int g = d / value_group_size;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + d;
        const float x = bf16_to_float(value[src_idx]);
        const int q = clamp_i32(static_cast<int>(nearbyintf((x - group_zero[g]) / group_scale[g])), 0, 15);
        const float deq = static_cast<float>(q) * group_scale[g] + group_zero[g];
        qvals[d] = static_cast<uint8_t>(q);
        const float delta = x - deq;
        err_scratch[d] = delta * delta;
        norm_scratch[d] = x * x;
    }
    __syncthreads();

    for (int p = threadIdx.x; p < head_dim / 2; p += blockDim.x) {
        const uint8_t lo = qvals[p * 2] & 0x0f;
        const uint8_t hi = qvals[p * 2 + 1] & 0x0f;
        const size_t dst_idx = (static_cast<size_t>(kvh) * aligned_tokens + tok) * (head_dim / 2) + p;
        value_int4[dst_idx] = static_cast<uint8_t>(lo | (hi << 4));
    }

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < head_dim) {
            err_scratch[threadIdx.x] += err_scratch[threadIdx.x + stride];
            norm_scratch[threadIdx.x] += norm_scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float token_l2 = sqrtf(err_scratch[0]);
        const float token_norm = sqrtf(norm_scratch[0]);
        const int block_id = tok / block_size;
        const int num_blocks = aligned_tokens / block_size;
        atomic_max_nonnegative_float(value_error + kvh * num_blocks + block_id, token_l2);
        atomic_max_nonnegative_float(value_norm + kvh * num_blocks + block_id, token_norm);
    }
}

__global__ void certified_kv_quantize_values_range_kernel(
    const __nv_bfloat16* value,
    uint8_t* value_int4,
    __half* value_scale,
    __half* value_zero,
    float* value_error,
    float* value_norm,
    int num_kv_heads,
    int max_t,
    int value_stride_tokens,
    int value_error_stride_blocks,
    int start_token,
    int token_count,
    int head_dim,
    int block_size,
    int value_group_size
) {
    const int token_linear = blockIdx.x;
    const int local_tok = token_linear % token_count;
    const int tok = start_token + local_tok;
    const int kvh = token_linear / token_count;
    if (kvh >= num_kv_heads) return;

    constexpr int kMaxHeadDim = 256;
    constexpr int kMaxGroups = 64;
    __shared__ float group_scale[kMaxGroups];
    __shared__ float group_zero[kMaxGroups];
    __shared__ uint8_t qvals[kMaxHeadDim];
    __shared__ float err_scratch[kMaxHeadDim];
    __shared__ float norm_scratch[kMaxHeadDim];

    const int num_groups = head_dim / value_group_size;
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        float min_v = INFINITY;
        float max_v = -INFINITY;
        const int d0 = g * value_group_size;
        for (int i = 0; i < value_group_size; ++i) {
            const int d = d0 + i;
            const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + d;
            const float x = bf16_to_float(value[src_idx]);
            min_v = fminf(min_v, x);
            max_v = fmaxf(max_v, x);
        }
        const float scale = fmaxf(max_v - min_v, 1.0e-8f) / 15.0f;
        group_scale[g] = scale;
        group_zero[g] = min_v;
        const size_t meta_idx =
            (static_cast<size_t>(kvh) * value_stride_tokens + tok) * num_groups + g;
        value_scale[meta_idx] = __float2half(scale);
        value_zero[meta_idx] = __float2half(min_v);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int g = d / value_group_size;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + d;
        const float x = bf16_to_float(value[src_idx]);
        const int q = clamp_i32(static_cast<int>(nearbyintf((x - group_zero[g]) / group_scale[g])), 0, 15);
        const float deq = static_cast<float>(q) * group_scale[g] + group_zero[g];
        qvals[d] = static_cast<uint8_t>(q);
        const float delta = x - deq;
        err_scratch[d] = delta * delta;
        norm_scratch[d] = x * x;
    }
    __syncthreads();

    for (int p = threadIdx.x; p < head_dim / 2; p += blockDim.x) {
        const uint8_t lo = qvals[p * 2] & 0x0f;
        const uint8_t hi = qvals[p * 2 + 1] & 0x0f;
        const size_t dst_idx =
            (static_cast<size_t>(kvh) * value_stride_tokens + tok) * (head_dim / 2) + p;
        value_int4[dst_idx] = static_cast<uint8_t>(lo | (hi << 4));
    }

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < head_dim) {
            err_scratch[threadIdx.x] += err_scratch[threadIdx.x + stride];
            norm_scratch[threadIdx.x] += norm_scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float token_l2 = sqrtf(err_scratch[0]);
        const float token_norm = sqrtf(norm_scratch[0]);
        const int block_id = tok / block_size;
        atomic_max_nonnegative_float(
            value_error + static_cast<size_t>(kvh) * value_error_stride_blocks + block_id,
            token_l2
        );
        atomic_max_nonnegative_float(
            value_norm + static_cast<size_t>(kvh) * value_error_stride_blocks + block_id,
            token_norm
        );
    }
}

__global__ void certified_kv_score_blocks_int8_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    float* block_max,
    float* block_sum,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    const int block_id = blockIdx.x % num_blocks;
    const int qh = blockIdx.x / num_blocks;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;

    constexpr int kMaxBlockSize = 256;
    __shared__ float scores[kMaxBlockSize];
    if (threadIdx.x < block_size) {
        const int tok = block_id * block_size + threadIdx.x;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const int8_t kq = static_cast<int8_t>(
                key_int8[(static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[(static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d];
            const float kz =
                key_zero[(static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d];
            acc += q * (static_cast<float>(kq) * ks + kz);
        }
        scores[threadIdx.x] = acc * q_scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int t = 0; t < block_size; ++t) {
            m = fmaxf(m, scores[t]);
        }
        float s = 0.0f;
        for (int t = 0; t < block_size; ++t) {
            s += expf(scores[t] - m);
        }
        const size_t out_idx = static_cast<size_t>(qh) * num_blocks + block_id;
        block_max[out_idx] = m;
        block_sum[out_idx] = s;
    }
}

__global__ void certified_kv_key_scale_norms_kernel(
    const float* __restrict__ key_scale,
    float* __restrict__ key_scale_norm,
    int kv_heads,
    int num_blocks,
    int key_scale_stride_blocks,
    int head_dim
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kv_heads * num_blocks;
    if (linear >= total) {
        return;
    }
    const int kvh = linear / num_blocks;
    const int block = linear - kvh * num_blocks;
    const float* scales =
        key_scale + (static_cast<size_t>(kvh) * key_scale_stride_blocks + block) * head_dim;
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        const float s = scales[d];
        sum_sq += s * s;
    }
    key_scale_norm[linear] = sqrtf(sum_sq);
}

__global__ void certified_kv_score_consistency_kernel(
    const __nv_bfloat16* __restrict__ query,
    const uint8_t* __restrict__ key_int8,
    const float* __restrict__ key_scale,
    const float* __restrict__ key_zero,
    const __nv_bfloat16* __restrict__ promoted_key,
    const uint32_t* __restrict__ promote_index,
    uint32_t* __restrict__ violation_flags,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int max_promoted_blocks,
    int head_dim,
    int gqa_group,
    float q_scale,
    float eps_guard
) {
    const int linear = blockIdx.x;
    const int qh = linear / num_blocks;
    const int block = linear - qh * num_blocks;
    if (qh >= q_heads || block >= num_blocks) {
        return;
    }
    const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + block];
    if (slot == 0xffffffffu || static_cast<int>(slot) >= max_promoted_blocks) {
        return;
    }
    const int token = threadIdx.x;
    if (token >= block_size) {
        return;
    }
    const int kvh = qh / gqa_group;
    const int token_idx = block * block_size + token;
    const __nv_bfloat16* q = query + static_cast<size_t>(qh) * head_dim;
    const uint8_t* k_i8 =
        key_int8 + (static_cast<size_t>(kvh) * key_stride_tokens + token_idx) * head_dim;
    const float* scale =
        key_scale + (static_cast<size_t>(kvh) * key_scale_stride_blocks + block) * head_dim;
    const float* zero =
        key_zero + (static_cast<size_t>(kvh) * key_scale_stride_blocks + block) * head_dim;
    const __nv_bfloat16* k_fp16 =
        promoted_key
        + ((static_cast<size_t>(qh) * max_promoted_blocks + slot) * block_size + token) * head_dim;

    float fp16_dot = 0.0f;
    float int8_dot = 0.0f;
    float weighted_scale_sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        const float qv = __bfloat162float(q[dim]);
        const float fp16_k = __bfloat162float(k_fp16[dim]);
        const float int8_k = static_cast<float>(static_cast<int8_t>(k_i8[dim])) * scale[dim] + zero[dim];
        fp16_dot += qv * fp16_k;
        int8_dot += qv * int8_k;
        weighted_scale_sum += fabsf(qv) * scale[dim];
    }
    const float delta = 0.5f * q_scale * weighted_scale_sum;
    if (fabsf(fp16_dot - int8_dot) * q_scale > delta + eps_guard) {
        violation_flags[qh] = 1u;
    }
}

__global__ void certified_kv_gather_promoted_keys_kernel(
    const __nv_bfloat16* __restrict__ tier2_key_bf16,
    const uint32_t* __restrict__ promote_index,
    __nv_bfloat16* __restrict__ promoted_key_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int max_promoted_blocks,
    int head_dim,
    int gqa_group
) {
    const int linear = blockIdx.x;
    const int qh = linear / num_blocks;
    const int block = linear - qh * num_blocks;
    if (qh >= q_heads || block >= num_blocks) {
        return;
    }
    const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + block];
    if (slot == 0xffffffffu || static_cast<int>(slot) >= max_promoted_blocks) {
        return;
    }
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) {
        return;
    }
    const int elems_per_block = block_size * head_dim;
    const size_t src_base =
        (static_cast<size_t>(kvh) * cap_tokens + static_cast<size_t>(block) * block_size) * head_dim;
    const size_t dst_base =
        (static_cast<size_t>(qh) * max_promoted_blocks + slot) * elems_per_block;
    for (int idx = threadIdx.x; idx < elems_per_block; idx += blockDim.x) {
        promoted_key_bf16[dst_base + idx] = tier2_key_bf16[src_base + idx];
    }
}

__global__ void certified_kv_key_cache_init_kernel(
    uint32_t* __restrict__ tags,
    uint32_t* __restrict__ lru,
    int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    tags[idx] = UINT32_MAX;
    lru[idx] = 0u;
}

__global__ void certified_kv_key_cache_resolve_kernel(
    const uint32_t* __restrict__ selected_blocks,
    const uint32_t* __restrict__ selected_counts,
    uint32_t* __restrict__ cache_tags,
    uint32_t* __restrict__ cache_lru,
    uint32_t* __restrict__ promote_index,
    uint32_t* __restrict__ gather_index,
    uint32_t* __restrict__ counters,
    int q_heads,
    int num_blocks,
    int max_selected_blocks,
    int cache_blocks,
    uint32_t tick_base
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;

    const size_t index_base = static_cast<size_t>(qh) * num_blocks;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        gather_index[index_base + b] = UINT32_MAX;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const uint32_t raw_count = selected_counts[qh];
    const int count = min(static_cast<int>(raw_count), max_selected_blocks);
    const size_t selected_base = static_cast<size_t>(qh) * max_selected_blocks;
    const size_t cache_base = static_cast<size_t>(qh) * cache_blocks;

    for (int i = 0; i < count; ++i) {
        const uint32_t block = selected_blocks[selected_base + i];
        if (block == UINT32_MAX || static_cast<int>(block) >= num_blocks) {
            continue;
        }

        int slot = -1;
        for (int s = 0; s < cache_blocks; ++s) {
            if (cache_tags[cache_base + s] == block) {
                slot = s;
                break;
            }
        }

        const uint32_t lru_value = tick_base + static_cast<uint32_t>(i + 1);
        if (slot >= 0) {
            cache_lru[cache_base + slot] = lru_value;
            promote_index[index_base + block] = static_cast<uint32_t>(slot);
            atomicAdd(&counters[0], 1u);
            continue;
        }

        slot = -1;
        for (int s = 0; s < cache_blocks; ++s) {
            if (cache_tags[cache_base + s] == UINT32_MAX) {
                slot = s;
                break;
            }
        }
        if (slot < 0) {
            slot = 0;
            uint32_t best_lru = cache_lru[cache_base];
            for (int s = 1; s < cache_blocks; ++s) {
                const uint32_t candidate = cache_lru[cache_base + s];
                if (candidate < best_lru) {
                    best_lru = candidate;
                    slot = s;
                }
            }
            atomicAdd(&counters[2], 1u);
        }

        cache_tags[cache_base + slot] = block;
        cache_lru[cache_base + slot] = lru_value;
        promote_index[index_base + block] = static_cast<uint32_t>(slot);
        gather_index[index_base + block] = static_cast<uint32_t>(slot);
        atomicAdd(&counters[1], 1u);
    }
}

__global__ void certified_kv_gather_promoted_keys_gqa_union_kernel(
    const __nv_bfloat16* __restrict__ tier2_key_bf16,
    const uint32_t* __restrict__ promote_index,
    __nv_bfloat16* __restrict__ promoted_key_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int max_promoted_blocks,
    int head_dim,
    int gqa_group
) {
    const int linear = blockIdx.x;
    const int kvh = linear / num_blocks;
    const int block = linear - kvh * num_blocks;
    if (kvh >= kv_heads || block >= num_blocks) {
        return;
    }
    const int elems_per_block = block_size * head_dim;
    const size_t src_base =
        (static_cast<size_t>(kvh) * cap_tokens + static_cast<size_t>(block) * block_size) * head_dim;
    const int first_qh = kvh * gqa_group;
    for (int idx = threadIdx.x; idx < elems_per_block; idx += blockDim.x) {
        bool any = false;
        for (int local_qh = 0; local_qh < gqa_group; ++local_qh) {
            const int qh = first_qh + local_qh;
            if (qh >= q_heads) continue;
            const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + block];
            if (slot != UINT32_MAX && static_cast<int>(slot) < max_promoted_blocks) {
                any = true;
                break;
            }
        }
        if (!any) {
            continue;
        }
        const __nv_bfloat16 value = tier2_key_bf16[src_base + idx];
        for (int local_qh = 0; local_qh < gqa_group; ++local_qh) {
            const int qh = first_qh + local_qh;
            if (qh >= q_heads) continue;
            const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + block];
            if (slot == UINT32_MAX || static_cast<int>(slot) >= max_promoted_blocks) continue;
            const size_t dst_base =
                (static_cast<size_t>(qh) * max_promoted_blocks + slot) * elems_per_block;
            promoted_key_bf16[dst_base + idx] = value;
        }
    }
}

__global__ void certified_kv_gather_all_promoted_keys_compact_kernel(
    const __nv_bfloat16* __restrict__ tier2_key_bf16,
    __nv_bfloat16* __restrict__ promoted_key_bf16,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int head_dim
) {
    const int linear = blockIdx.x;
    const int kvh = linear / num_blocks;
    const int block = linear - kvh * num_blocks;
    if (kvh >= kv_heads || block >= num_blocks) {
        return;
    }
    const int elems_per_block = block_size * head_dim;
    const size_t src_base =
        (static_cast<size_t>(kvh) * cap_tokens + static_cast<size_t>(block) * block_size) * head_dim;
    const size_t dst_base =
        (static_cast<size_t>(kvh) * num_blocks + block) * elems_per_block;
    for (int idx = threadIdx.x; idx < elems_per_block; idx += blockDim.x) {
        promoted_key_bf16[dst_base + idx] = tier2_key_bf16[src_base + idx];
    }
}

__global__ void certified_kv_gather_promoted_values_kernel(
    const __nv_bfloat16* __restrict__ tier2_value_bf16,
    const uint32_t* __restrict__ value_promote_index,
    __nv_bfloat16* __restrict__ promoted_value_bf16,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int max_promoted_value_blocks,
    int head_dim,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int linear = blockIdx.x;
    const int kvh = linear / num_blocks;
    const int block = linear - kvh * num_blocks;
    if (kvh >= kv_heads || block >= num_blocks) {
        return;
    }
    const uint32_t slot = value_promote_index[static_cast<size_t>(kvh) * num_blocks + block];
    if (slot == 0xffffffffu || static_cast<int>(slot) >= max_promoted_value_blocks) {
        return;
    }
    const int elems_per_block = block_size * head_dim;
    const size_t src_base =
        (static_cast<size_t>(kvh) * cap_tokens + static_cast<size_t>(block) * block_size) * head_dim;
    const size_t dst_base =
        (static_cast<size_t>(kvh) * max_promoted_value_blocks + slot) * elems_per_block;
    for (int idx = threadIdx.x; idx < elems_per_block; idx += blockDim.x) {
        promoted_value_bf16[dst_base + idx] = tier2_value_bf16[src_base + idx];
    }
}

__global__ void certified_kv_selected_fp16_log_mass_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ promoted_key,
    const uint32_t* __restrict__ promote_index,
    float* __restrict__ out_log_mass,
    int q_heads,
    int num_blocks,
    int block_size,
    int max_promoted_blocks,
    int head_dim,
    float q_scale
) {
    const int linear = blockIdx.x;
    const int qh = linear / max_promoted_blocks;
    const int slot = linear - qh * max_promoted_blocks;
    if (qh >= q_heads || slot >= max_promoted_blocks) {
        return;
    }
    extern __shared__ float scores[];
    if (threadIdx.x < block_size) {
        scores[threadIdx.x] = -INFINITY;
    }
    __syncthreads();

    uint32_t block_id = 0xffffffffu;
    for (int b = slot; b < num_blocks; b += max_promoted_blocks) {
        const uint32_t candidate = promote_index[static_cast<size_t>(qh) * num_blocks + b];
        if (candidate == static_cast<uint32_t>(slot)) {
            block_id = static_cast<uint32_t>(b);
            break;
        }
    }
    if (block_id == 0xffffffffu) {
        if (threadIdx.x == 0) {
            out_log_mass[static_cast<size_t>(qh) * max_promoted_blocks + slot] = -INFINITY;
        }
        return;
    }

    const int token = threadIdx.x;
    if (token < block_size) {
        const __nv_bfloat16* q = query + static_cast<size_t>(qh) * head_dim;
        const __nv_bfloat16* k =
            promoted_key
            + ((static_cast<size_t>(qh) * max_promoted_blocks + slot) * block_size + token) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        }
        scores[token] = dot * q_scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int t = 0; t < block_size; ++t) {
            m = fmaxf(m, scores[t]);
        }
        float s = 0.0f;
        for (int t = 0; t < block_size; ++t) {
            s += expf(scores[t] - m);
        }
        out_log_mass[static_cast<size_t>(qh) * max_promoted_blocks + slot] = m + logf(s);
    }
}

constexpr int kCertifiedSelectorMaxBlocks = 2048;

__global__ void certified_kv_selector_init_kernel(
    uint32_t* promote_index,
    uint32_t* value_promote_index,
    uint32_t* selected_blocks,
    uint32_t* selected_counts,
    uint32_t* fallback_flags,
    float* delta_blocks,
    float* e_key_by_head,
    float* delta_tail_by_head,
    float* vmax_by_head,
    float* true_tail_by_head,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int max_promoted_blocks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int promote_elems = q_heads * num_blocks;
    const int value_elems = kv_heads * num_blocks;
    const int selected_elems = q_heads * max_promoted_blocks;
    const int head_elems = q_heads;
    const int delta_elems = q_heads * num_blocks;
    const int total = promote_elems + value_elems + selected_elems + head_elems * 5 + delta_elems;
    if (idx >= total) {
        return;
    }
    int p = idx;
    if (p < promote_elems) {
        promote_index[p] = 0xffffffffu;
        return;
    }
    p -= promote_elems;
    if (p < value_elems) {
        value_promote_index[p] = 0xffffffffu;
        return;
    }
    p -= value_elems;
    if (p < selected_elems) {
        selected_blocks[p] = 0xffffffffu;
        return;
    }
    p -= selected_elems;
    if (p < head_elems) {
        selected_counts[p] = 0;
        return;
    }
    p -= head_elems;
    if (p < head_elems) {
        fallback_flags[p] = 0;
        return;
    }
    p -= head_elems;
    if (p < head_elems) {
        e_key_by_head[p] = 0.0f;
        return;
    }
    p -= head_elems;
    if (p < head_elems) {
        delta_tail_by_head[p] = 0.0f;
        return;
    }
    p -= head_elems;
    if (p < head_elems) {
        vmax_by_head[p] = 0.0f;
        return;
    }
    p -= head_elems;
    if (p < head_elems) {
        true_tail_by_head[p] = 0.0f;
        return;
    }
    p -= head_elems;
    if (p < delta_elems) {
        delta_blocks[p] = 0.0f;
    }
}

__global__ void certified_kv_select_blocks_kernel(
    const __nv_bfloat16* __restrict__ query,
    const float* __restrict__ key_scale_norm,
    const float* __restrict__ block_max,
    const float* __restrict__ block_sum,
    const float* __restrict__ value_norm,
    uint32_t* __restrict__ promote_index,
    uint32_t* __restrict__ selected_blocks,
    uint32_t* __restrict__ selected_counts,
    uint32_t* __restrict__ fallback_flags,
    float* __restrict__ delta_blocks,
    float* __restrict__ e_key_by_head,
    float* __restrict__ delta_tail_by_head,
    float* __restrict__ vmax_by_head,
    float* __restrict__ true_tail_by_head,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int key_scale_norm_stride_blocks,
    int value_norm_stride_blocks,
    int head_dim,
    int gqa_group,
    int k_min,
    int k_max,
    int max_promoted_blocks,
    float q_scale,
    float tau_cov,
    float rung1_threshold,
    float rung1_multiplier,
    float delta_guard_factor,
    float score_exploration_rate,
    int require_certified_tail_bound
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads || num_blocks > kCertifiedSelectorMaxBlocks) {
        return;
    }
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) {
        return;
    }

    __shared__ float log_mass[kCertifiedSelectorMaxBlocks];
    __shared__ float prob[kCertifiedSelectorMaxBlocks];
    __shared__ float delta[kCertifiedSelectorMaxBlocks];
    __shared__ uint8_t selected[kCertifiedSelectorMaxBlocks];
    __shared__ float reduce_scratch[256];
    __shared__ int reduce_index_scratch[256];
    __shared__ int selected_count_s;
    __shared__ int k_min_clamped_s;
    __shared__ int k_max_clamped_s;
    __shared__ int slot_cap_s;
    __shared__ int continue_select_s;
    __shared__ int best_index_s;
    __shared__ float best_prob_s;
    __shared__ float covered_s;
    __shared__ int rung_target_s;
    __shared__ float tail_mass_s;
    __shared__ float delta_tail_s;
    __shared__ float vmax_s;

    float q_norm_local = 0.0f;
    const __nv_bfloat16* q = query + static_cast<size_t>(qh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float qv = bf16_to_float(q[d]);
        q_norm_local += qv * qv;
    }
    const float q_norm = sqrtf(block_reduce_sum_256(q_norm_local, reduce_scratch));

    float local_max = -INFINITY;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * num_blocks + b;
        const float s = fmaxf(block_sum[score_idx], 1.0e-30f);
        const float lm = block_max[score_idx] + logf(s);
        log_mass[b] = lm;
        selected[b] = 0;
        local_max = fmaxf(local_max, lm);

        const float scale_norm =
            key_scale_norm[static_cast<size_t>(kvh) * key_scale_norm_stride_blocks + b];
        const float db = 0.5f * q_scale * q_norm * scale_norm;
        delta[b] = db;
        delta_blocks[static_cast<size_t>(qh) * num_blocks + b] = db;
    }
    const float max_lm = block_reduce_max_256(local_max, reduce_scratch);

    float local_sum = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        const float p = expf(log_mass[b] - max_lm);
        prob[b] = p;
        local_sum += p;
    }
    const float denom = fmaxf(block_reduce_sum_256(local_sum, reduce_scratch), 1.0e-30f);
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        prob[b] /= denom;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        selected_count_s = 0;
        covered_s = 0.0f;
        k_min_clamped_s = clamp_i32(k_min, 0, num_blocks);
        k_max_clamped_s = clamp_i32(k_max, k_min_clamped_s, num_blocks);
        slot_cap_s = clamp_i32(max_promoted_blocks, 1, num_blocks);
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            continue_select_s =
                ((covered_s < tau_cov || selected_count_s < k_min_clamped_s) &&
                 selected_count_s < k_max_clamped_s &&
                 selected_count_s < slot_cap_s)
                    ? 1
                    : 0;
        }
        __syncthreads();
        if (!continue_select_s) {
            break;
        }

        int best = -1;
        float best_prob = -1.0f;
        for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
            const float p = prob[b];
            if (!selected[b] && (p > best_prob || (p == best_prob && (best < 0 || b < best)))) {
                best_prob = p;
                best = b;
            }
        }
        reduce_scratch[threadIdx.x] = best_prob;
        reduce_index_scratch[threadIdx.x] = best;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_prob = reduce_scratch[threadIdx.x + stride];
                const int other_idx = reduce_index_scratch[threadIdx.x + stride];
                const bool take_other =
                    other_prob > reduce_scratch[threadIdx.x] ||
                    (other_prob == reduce_scratch[threadIdx.x] &&
                     other_idx >= 0 &&
                     (reduce_index_scratch[threadIdx.x] < 0 ||
                      other_idx < reduce_index_scratch[threadIdx.x]));
                if (take_other) {
                    reduce_scratch[threadIdx.x] = other_prob;
                    reduce_index_scratch[threadIdx.x] = other_idx;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            best_index_s = reduce_index_scratch[0];
            best_prob_s = reduce_scratch[0];
            if (best_index_s >= 0) {
                selected[best_index_s] = 1;
                selected_blocks[static_cast<size_t>(qh) * max_promoted_blocks + selected_count_s] =
                    static_cast<uint32_t>(best_index_s);
                promote_index[static_cast<size_t>(qh) * num_blocks + best_index_s] =
                    static_cast<uint32_t>(selected_count_s);
                covered_s += fmaxf(best_prob_s, 0.0f);
                selected_count_s += 1;
            } else {
                continue_select_s = 0;
            }
        }
        __syncthreads();
    }

    float tail_mass_local = 0.0f;
    float delta_tail_local = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        if (!selected[b]) {
            tail_mass_local += prob[b];
            delta_tail_local = fmaxf(delta_tail_local, delta[b]);
        }
    }
    const float tail_mass_initial =
        block_reduce_sum_256(tail_mass_local, reduce_scratch);
    const float delta_tail_initial =
        block_reduce_max_256(delta_tail_local, reduce_scratch);
    if (threadIdx.x == 0) {
        tail_mass_s = tail_mass_initial;
        delta_tail_s = delta_tail_initial;
    }
    __syncthreads();

    float true_tail_bound =
        expf(delta_guard_factor * delta_tail_s) * fmaxf(tail_mass_s, 0.0f);

    if (threadIdx.x == 0) {
        if (true_tail_bound > rung1_threshold && selected_count_s < slot_cap_s) {
            rung_target_s = clamp_i32(
                static_cast<int>(ceilf(static_cast<float>(selected_count_s) * rung1_multiplier)),
                selected_count_s + 1,
                slot_cap_s
            );
        } else {
            rung_target_s = selected_count_s;
        }
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            continue_select_s = (selected_count_s < rung_target_s) ? 1 : 0;
        }
        __syncthreads();
        if (!continue_select_s) {
            break;
        }

        int best = -1;
        float best_prob = -1.0f;
        for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
            const float p = prob[b];
            if (!selected[b] && (p > best_prob || (p == best_prob && (best < 0 || b < best)))) {
                best_prob = p;
                best = b;
            }
        }
        reduce_scratch[threadIdx.x] = best_prob;
        reduce_index_scratch[threadIdx.x] = best;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_prob = reduce_scratch[threadIdx.x + stride];
                const int other_idx = reduce_index_scratch[threadIdx.x + stride];
                const bool take_other =
                    other_prob > reduce_scratch[threadIdx.x] ||
                    (other_prob == reduce_scratch[threadIdx.x] &&
                     other_idx >= 0 &&
                     (reduce_index_scratch[threadIdx.x] < 0 ||
                      other_idx < reduce_index_scratch[threadIdx.x]));
                if (take_other) {
                    reduce_scratch[threadIdx.x] = other_prob;
                    reduce_index_scratch[threadIdx.x] = other_idx;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            best_index_s = reduce_index_scratch[0];
            if (best_index_s >= 0) {
                selected[best_index_s] = 1;
                selected_blocks[static_cast<size_t>(qh) * max_promoted_blocks + selected_count_s] =
                    static_cast<uint32_t>(best_index_s);
                promote_index[static_cast<size_t>(qh) * num_blocks + best_index_s] =
                    static_cast<uint32_t>(selected_count_s);
                selected_count_s += 1;
            } else {
                continue_select_s = 0;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && score_exploration_rate > 0.0f && selected_count_s < slot_cap_s) {
        const int period = max(1, static_cast<int>(ceilf(1.0f / score_exploration_rate)));
        for (int b = 0; b < num_blocks && selected_count_s < slot_cap_s; ++b) {
            if (selected[b]) {
                continue;
            }
            const uint32_t hash =
                (static_cast<uint32_t>(b) * 1103515245u) ^ static_cast<uint32_t>(qh);
            if ((hash % static_cast<uint32_t>(period)) == 0u) {
                selected[b] = 1;
                selected_blocks[static_cast<size_t>(qh) * max_promoted_blocks + selected_count_s] =
                    static_cast<uint32_t>(b);
                promote_index[static_cast<size_t>(qh) * num_blocks + b] =
                    static_cast<uint32_t>(selected_count_s);
                selected_count_s += 1;
            }
        }
    }
    __syncthreads();

    tail_mass_local = 0.0f;
    delta_tail_local = 0.0f;
    float vmax_local = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        if (!selected[b]) {
            tail_mass_local += prob[b];
            delta_tail_local = fmaxf(delta_tail_local, delta[b]);
            vmax_local = fmaxf(
                vmax_local,
                value_norm[static_cast<size_t>(kvh) * value_norm_stride_blocks + b]
            );
        }
    }
    const float tail_mass_final =
        block_reduce_sum_256(tail_mass_local, reduce_scratch);
    const float delta_tail_final =
        block_reduce_max_256(delta_tail_local, reduce_scratch);
    const float vmax_final =
        block_reduce_max_256(vmax_local, reduce_scratch);
    if (threadIdx.x == 0) {
        tail_mass_s = tail_mass_final;
        delta_tail_s = delta_tail_final;
        vmax_s = vmax_final;
        true_tail_bound = expf(delta_guard_factor * delta_tail_s) * fmaxf(tail_mass_s, 0.0f);
        const float e_key =
            2.0f * vmax_s * true_tail_bound * (expf(2.0f * delta_tail_s) - 1.0f);

        selected_counts[qh] = static_cast<uint32_t>(selected_count_s);
        delta_tail_by_head[qh] = delta_tail_s;
        vmax_by_head[qh] = vmax_s;
        true_tail_by_head[qh] = true_tail_bound;
        e_key_by_head[qh] = e_key;
        if (require_certified_tail_bound && true_tail_bound > rung1_threshold) {
            fallback_flags[qh] = 1;
        }
    }
}

__global__ void certified_kv_ranking_flags_kernel(
    const float* __restrict__ block_max,
    const float* __restrict__ block_sum,
    const float* __restrict__ delta_blocks,
    const float* __restrict__ selected_fp16_log_masses,
    const uint32_t* __restrict__ promote_index,
    uint32_t* __restrict__ fallback_flags,
    int q_heads,
    int num_blocks,
    int max_promoted_blocks
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    int best_fp16_slot = -1;
    float best_fp16 = -INFINITY;
    for (int slot = 0; slot < max_promoted_blocks; ++slot) {
        const float lm =
            selected_fp16_log_masses[static_cast<size_t>(qh) * max_promoted_blocks + slot];
        if (lm > best_fp16) {
            best_fp16 = lm;
            best_fp16_slot = slot;
        }
    }
    if (best_fp16_slot < 0 || !isfinite(best_fp16)) {
        return;
    }

    int best_int8_block = -1;
    int best_fp16_block = -1;
    float best_int8 = -INFINITY;
    for (int b = 0; b < num_blocks; ++b) {
        const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + b];
        if (slot != 0xffffffffu && static_cast<int>(slot) < max_promoted_blocks) {
            const float lm =
                block_max[static_cast<size_t>(qh) * num_blocks + b]
                + logf(fmaxf(block_sum[static_cast<size_t>(qh) * num_blocks + b], 1.0e-30f));
            if (lm > best_int8) {
                best_int8 = lm;
                best_int8_block = b;
            }
            if (static_cast<int>(slot) == best_fp16_slot) {
                best_fp16_block = b;
            }
        }
    }
    if (best_int8_block != best_fp16_block) {
        fallback_flags[qh] = 1;
        return;
    }

    for (int b = 0; b < num_blocks; ++b) {
        const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + b];
        if (slot == 0xffffffffu) {
            const float int8_log_mass =
                block_max[static_cast<size_t>(qh) * num_blocks + b]
                + logf(fmaxf(block_sum[static_cast<size_t>(qh) * num_blocks + b], 1.0e-30f));
            const float ub = int8_log_mass + delta_blocks[static_cast<size_t>(qh) * num_blocks + b];
            if (ub > best_fp16) {
                fallback_flags[qh] = 1;
                return;
            }
        }
    }
}

__device__ __forceinline__ float certified_kv_dequant_int4_value(
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    int kvh,
    int tok,
    int d,
    int value_stride_tokens,
    int head_dim,
    int value_group_size
) {
    const int packed_dim = head_dim / 2;
    const int groups = head_dim / value_group_size;
    const int group = d / value_group_size;
    const uint8_t packed =
        value_int4[(static_cast<size_t>(kvh) * value_stride_tokens + tok) * packed_dim + (d / 2)];
    const int q = (d & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const size_t meta_idx = (static_cast<size_t>(kvh) * value_stride_tokens + tok) * groups + group;

    if (value_group_size == 16 && head_dim <= blockDim.x) {
        const int lane = threadIdx.x & 31;
        const int leader_lane = lane & ~15;
        float scale = 0.0f;
        float zero = 0.0f;
        if ((d & 15) == 0) {
            scale = __half2float(value_scale[meta_idx]);
            zero = __half2float(value_zero[meta_idx]);
        }
        const unsigned int active = __activemask();
        scale = __shfl_sync(active, scale, leader_lane);
        zero = __shfl_sync(active, zero, leader_lane);
        return static_cast<float>(q) * scale + zero;
    }

    return static_cast<float>(q) * __half2float(value_scale[meta_idx]) + __half2float(value_zero[meta_idx]);
}

__device__ __forceinline__ float certified_kv_dequant_int4_value_scalar(
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    int kvh,
    int tok,
    int d,
    int value_stride_tokens,
    int head_dim,
    int value_group_size
) {
    const int packed_dim = head_dim / 2;
    const int groups = head_dim / value_group_size;
    const int group = d / value_group_size;
    const uint8_t packed =
        value_int4[(static_cast<size_t>(kvh) * value_stride_tokens + tok) * packed_dim + (d / 2)];
    const int q = (d & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const size_t meta_idx = (static_cast<size_t>(kvh) * value_stride_tokens + tok) * groups + group;
    return static_cast<float>(q) * __half2float(value_scale[meta_idx]) +
           __half2float(value_zero[meta_idx]);
}

__global__ void certified_kv_attend_int8_int4_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    float* score_scratch,
    float* output_f32,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;

    for (int tok = threadIdx.x; tok < aligned_tokens; tok += blockDim.x) {
        const int block_id = tok / block_size;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const int8_t kq = static_cast<int8_t>(
                key_int8[(static_cast<size_t>(kvh) * aligned_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[(static_cast<size_t>(kvh) * num_blocks + block_id) * head_dim + d];
            const float kz =
                key_zero[(static_cast<size_t>(kvh) * num_blocks + block_id) * head_dim + d];
            acc += q * (static_cast<float>(kq) * ks + kz);
        }
        score_scratch[static_cast<size_t>(qh) * aligned_tokens + tok] = acc * q_scale;
    }
    __syncthreads();

    __shared__ float max_score;
    __shared__ float denom;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int tok = 0; tok < aligned_tokens; ++tok) {
            m = fmaxf(m, score_scratch[static_cast<size_t>(qh) * aligned_tokens + tok]);
        }
        float s = 0.0f;
        for (int tok = 0; tok < aligned_tokens; ++tok) {
            s += expf(score_scratch[static_cast<size_t>(qh) * aligned_tokens + tok] - m);
        }
        max_score = m;
        denom = s;
    }
    __syncthreads();

    for (int tok = threadIdx.x; tok < aligned_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * aligned_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < aligned_tokens; ++tok) {
            const float w = score_scratch[static_cast<size_t>(qh) * aligned_tokens + tok];
            const float v = certified_kv_dequant_int4_value(
                value_int4,
                value_scale,
                value_zero,
                kvh,
                tok,
                d,
                aligned_tokens,
                head_dim,
                value_group_size
            );
            acc += w * v;
        }
        const size_t out_idx = static_cast<size_t>(qh) * head_dim + d;
        if (output_bf16 != nullptr) {
            output_bf16[out_idx] = __float2bfloat16(acc);
        } else {
            output_f32[out_idx] = acc;
        }
    }
}

__global__ void certified_kv_attend_int8_int4_bf16_tail_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    const __nv_bfloat16* tail_key,
    const __nv_bfloat16* tail_value,
    float* score_scratch,
    float* output_f32,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;

    for (int tok = threadIdx.x; tok < aligned_tokens; tok += blockDim.x) {
        const int block_id = tok / block_size;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const int8_t kq = static_cast<int8_t>(
                key_int8[(static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[
                    (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                ];
            const float kz =
                key_zero[
                    (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                ];
            acc += q * (static_cast<float>(kq) * ks + kz);
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
    for (int tail_tok = threadIdx.x; tail_tok < tail_len; tail_tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const float k = bf16_to_float(
                tail_key[
                    (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                     tail_key_start_tokens + tail_tok) * head_dim + d
                ]
            );
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + aligned_tokens + tail_tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);

    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] - max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w = score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
            float v;
            if (tok < aligned_tokens) {
                v = certified_kv_dequant_int4_value_scalar(
                    value_int4,
                    value_scale,
                    value_zero,
                    kvh,
                    tok,
                    d,
                    value_stride_tokens,
                    head_dim,
                    value_group_size
                );
            } else {
                const int tail_tok = tok - aligned_tokens;
                v = bf16_to_float(
                    tail_value[
                        (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                         tail_value_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += w * v;
        }
        const size_t out_idx = static_cast<size_t>(qh) * head_dim + d;
        if (output_bf16 != nullptr) {
            output_bf16[out_idx] = __float2bfloat16(acc);
        } else {
            output_f32[out_idx] = acc;
        }
    }
}

__global__ void certified_kv_attend_mixed_key_int4_bf16_tail_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    const __nv_bfloat16* promoted_key_bf16,
    const uint32_t* promote_index,
    const __nv_bfloat16* promoted_value_bf16,
    const uint32_t* value_promote_index,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    const __nv_bfloat16* tail_key,
    const __nv_bfloat16* tail_value,
    float* score_scratch,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int promoted_key_heads,
    int max_promoted_blocks,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;

    for (int tok = threadIdx.x; tok < aligned_tokens; tok += blockDim.x) {
        const int block_id = tok / block_size;
        const uint32_t promoted_slot =
            promote_index[static_cast<size_t>(qh) * num_blocks + block_id];
        const bool promote =
            promoted_slot != 0xffffffffu && static_cast<int>(promoted_slot) < max_promoted_blocks;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            float k;
            if (promote) {
                const int token_in_block = tok - block_id * block_size;
                const int promoted_head = (promoted_key_heads == kv_heads) ? kvh : qh;
                k = bf16_to_float(
                    promoted_key_bf16[
                        ((static_cast<size_t>(promoted_head) * max_promoted_blocks + promoted_slot) *
                             block_size +
                         token_in_block) *
                            head_dim +
                        d
                    ]
                );
            } else {
                const int8_t kq = static_cast<int8_t>(
                    key_int8[(static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + d]
                );
                const float ks =
                    key_scale[
                        (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                    ];
                const float kz =
                    key_zero[
                        (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                    ];
                k = static_cast<float>(kq) * ks + kz;
            }
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
    for (int tail_tok = threadIdx.x; tail_tok < tail_len; tail_tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const float k = bf16_to_float(
                tail_key[
                    (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                     tail_key_start_tokens + tail_tok) * head_dim + d
                ]
            );
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + aligned_tokens + tail_tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);

    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] - max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    if (head_dim <= 128) {
        __shared__ float value_partial[256];
        const int d = threadIdx.x % head_dim;
        const int lane = threadIdx.x / head_dim;
        const int lanes_per_dim = blockDim.x / head_dim;
        float acc = 0.0f;
        if (lane < lanes_per_dim) {
            for (int tok = lane; tok < total_tokens; tok += lanes_per_dim) {
                const float w =
                    score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
                float v;
                if (tok < aligned_tokens) {
                    const int block_id = tok / block_size;
                    const uint32_t promoted_value_slot =
                        value_promote_index[static_cast<size_t>(kvh) * num_blocks + block_id];
                    if (promoted_value_slot != 0xffffffffu &&
                        static_cast<int>(promoted_value_slot) < max_promoted_value_blocks) {
                        const int token_in_block = tok - block_id * block_size;
                        v = bf16_to_float(
                            promoted_value_bf16[
                                ((static_cast<size_t>(kvh) * max_promoted_value_blocks +
                                  promoted_value_slot) *
                                     block_size +
                                 token_in_block) *
                                    head_dim +
                                d
                            ]
                        );
                    } else {
                        v = certified_kv_dequant_int4_value(
                            value_int4,
                            value_scale,
                            value_zero,
                            kvh,
                            tok,
                            d,
                            value_stride_tokens,
                            head_dim,
                            value_group_size
                        );
                    }
                } else {
                    const int tail_tok = tok - aligned_tokens;
                    v = bf16_to_float(
                        tail_value[
                            (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                             tail_value_start_tokens + tail_tok) * head_dim + d
                        ]
                    );
                }
                acc += w * v;
            }
        }
        value_partial[threadIdx.x] = acc;
        __syncthreads();
        if (lane == 0) {
            float sum = 0.0f;
            for (int l = 0; l < lanes_per_dim; ++l) {
                sum += value_partial[l * head_dim + d];
            }
            output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(sum);
        }
    } else {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (int tok = 0; tok < total_tokens; ++tok) {
                const float w =
                    score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
                float v;
                if (tok < aligned_tokens) {
                    const int block_id = tok / block_size;
                    const uint32_t promoted_value_slot =
                        value_promote_index[static_cast<size_t>(kvh) * num_blocks + block_id];
                    if (promoted_value_slot != 0xffffffffu &&
                        static_cast<int>(promoted_value_slot) < max_promoted_value_blocks) {
                        const int token_in_block = tok - block_id * block_size;
                        v = bf16_to_float(
                            promoted_value_bf16[
                                ((static_cast<size_t>(kvh) * max_promoted_value_blocks +
                                  promoted_value_slot) *
                                     block_size +
                                 token_in_block) *
                                    head_dim +
                                d
                            ]
                        );
                    } else {
                        v = certified_kv_dequant_int4_value(
                            value_int4,
                            value_scale,
                            value_zero,
                            kvh,
                            tok,
                            d,
                            value_stride_tokens,
                            head_dim,
                            value_group_size
                        );
                    }
                } else {
                    const int tail_tok = tok - aligned_tokens;
                    v = bf16_to_float(
                        tail_value[
                            (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                             tail_value_start_tokens + tail_tok) * head_dim + d
                        ]
                    );
                }
                acc += w * v;
            }
            output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(acc);
        }
    }
}

__global__ void certified_kv_mixed_key_score_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    const __nv_bfloat16* promoted_key_bf16,
    const uint32_t* promote_index,
    const __nv_bfloat16* tail_key,
    float* score_scratch,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int promoted_key_heads,
    int max_promoted_blocks,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int total_chunks = num_blocks + (tail_len > 0 ? 1 : 0);
    const int qh = blockIdx.x / total_chunks;
    const int chunk = blockIdx.x - qh * total_chunks;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;

    const int aligned_tokens = num_blocks * block_size;
    const int chunk_tokens = chunk < num_blocks ? block_size : tail_len;
    const int token_base = chunk * block_size;
    const uint32_t promoted_slot =
        chunk < num_blocks ? promote_index[static_cast<size_t>(qh) * num_blocks + chunk] :
                             0xffffffffu;
    const bool promote =
        promoted_slot != 0xffffffffu && static_cast<int>(promoted_slot) < max_promoted_blocks;

    for (int token_in_chunk = threadIdx.x; token_in_chunk < chunk_tokens; token_in_chunk += blockDim.x) {
        const int tok = token_base + token_in_chunk;
        float acc = 0.0f;
        if (tok < aligned_tokens) {
            for (int d = 0; d < head_dim; ++d) {
                const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
                float k;
                if (promote) {
                    const int promoted_head = (promoted_key_heads == kv_heads) ? kvh : qh;
                    k = bf16_to_float(
                        promoted_key_bf16[
                            ((static_cast<size_t>(promoted_head) * max_promoted_blocks +
                              promoted_slot) *
                                 block_size +
                             token_in_chunk) *
                                head_dim +
                            d
                        ]
                    );
                } else {
                    const int8_t kq = static_cast<int8_t>(
                        key_int8[(static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + d]
                    );
                    const float ks =
                        key_scale[
                            (static_cast<size_t>(kvh) * key_scale_stride_blocks + chunk) * head_dim + d
                        ];
                    const float kz =
                        key_zero[
                            (static_cast<size_t>(kvh) * key_scale_stride_blocks + chunk) * head_dim + d
                        ];
                    k = static_cast<float>(kq) * ks + kz;
                }
                acc += q * k;
            }
        } else {
            const int tail_tok = tok - aligned_tokens;
            for (int d = 0; d < head_dim; ++d) {
                const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
                const float k = bf16_to_float(
                    tail_key[
                        (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                         tail_key_start_tokens + tail_tok) * head_dim + d
                    ]
                );
                acc += q * k;
            }
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
}

__global__ void certified_kv_mixed_value_by_dim_kernel(
    const float* score_scratch,
    const __nv_bfloat16* promoted_value_bf16,
    const uint32_t* value_promote_index,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    const __nv_bfloat16* tail_value,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int linear = blockIdx.x;
    const int qh = linear / head_dim;
    const int d = linear - qh * head_dim;
    if (qh >= q_heads || d >= head_dim) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;

    __shared__ float reduce_scratch[256];
    float local = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const float w = score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
        float v;
        if (tok < aligned_tokens) {
            const int block_id = tok / block_size;
            const uint32_t promoted_value_slot =
                value_promote_index[static_cast<size_t>(kvh) * num_blocks + block_id];
            if (promoted_value_slot != 0xffffffffu &&
                static_cast<int>(promoted_value_slot) < max_promoted_value_blocks) {
                const int token_in_block = tok - block_id * block_size;
                v = bf16_to_float(
                    promoted_value_bf16[
                        ((static_cast<size_t>(kvh) * max_promoted_value_blocks +
                          promoted_value_slot) *
                             block_size +
                         token_in_block) *
                            head_dim +
                        d
                    ]
                );
            } else {
                v = certified_kv_dequant_int4_value_scalar(
                    value_int4,
                    value_scale,
                    value_zero,
                    kvh,
                    tok,
                    d,
                    value_stride_tokens,
                    head_dim,
                    value_group_size
                );
            }
        } else {
            const int tail_tok = tok - aligned_tokens;
            v = bf16_to_float(
                tail_value[
                    (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                     tail_value_start_tokens + tail_tok) * head_dim + d
                ]
            );
        }
        local += w * v;
    }
    const float sum = block_reduce_sum_256(local, reduce_scratch);
    if (threadIdx.x == 0) {
        output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(sum);
    }
}

__global__ void certified_kv_all_promoted_score_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* promoted_key_bf16,
    const __nv_bfloat16* tail_key,
    float* score_scratch,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int total_chunks = num_blocks + (tail_len > 0 ? 1 : 0);
    const int qh = blockIdx.x / total_chunks;
    const int chunk = blockIdx.x - qh * total_chunks;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int chunk_tokens = chunk < num_blocks ? block_size : tail_len;
    const int token_base = chunk * block_size;
    for (int token_in_chunk = threadIdx.x; token_in_chunk < chunk_tokens; token_in_chunk += blockDim.x) {
        const int tok = token_base + token_in_chunk;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            float k;
            if (tok < aligned_tokens) {
                k = bf16_to_float(
                    promoted_key_bf16[
                        ((static_cast<size_t>(kvh) * num_blocks + chunk) * block_size +
                         token_in_chunk) *
                            head_dim +
                        d
                    ]
                );
            } else {
                const int tail_tok = tok - aligned_tokens;
                k = bf16_to_float(
                    tail_key[
                        (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                         tail_key_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
}

__global__ void certified_kv_softmax_stats_kernel(
    float* score_scratch,
    float* softmax_stats,
    int q_heads,
    int total_tokens,
    int score_stride_tokens,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);
    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] - max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);
    if (threadIdx.x == 0) {
        softmax_stats[static_cast<size_t>(qh) * 2] = max_score;
        softmax_stats[static_cast<size_t>(qh) * 2 + 1] = denom;
    }
}

__global__ void certified_kv_softmax_normalize_kernel(
    float* score_scratch,
    const float* softmax_stats,
    int q_heads,
    int total_tokens,
    int score_stride_tokens,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const float max_score = softmax_stats[static_cast<size_t>(qh) * 2];
    const float denom = softmax_stats[static_cast<size_t>(qh) * 2 + 1];
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
}

__global__ void certified_kv_softmax_normalize_inplace_kernel(
    float* score_scratch,
    int q_heads,
    int total_tokens,
    int score_stride_tokens,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);
    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] - max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
}

__global__ void certified_kv_block_mass_from_probs_kernel(
    const float* __restrict__ score_scratch,
    float* __restrict__ block_mass,
    int q_heads,
    int num_blocks,
    int block_size,
    int score_stride_tokens,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int linear = blockIdx.x;
    const int qh = linear / num_blocks;
    const int block = linear - qh * num_blocks;
    if (qh >= q_heads || block >= num_blocks) return;

    __shared__ float reduce_scratch[256];
    float local_sum = 0.0f;
    const int token_base = block * block_size;
    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        local_sum += score_scratch[static_cast<size_t>(qh) * score_stride_tokens + token_base + t];
    }
    const float sum = block_reduce_sum_256(local_sum, reduce_scratch);
    if (threadIdx.x == 0) {
        block_mass[static_cast<size_t>(qh) * num_blocks + block] = sum;
    }
}

__global__ void certified_kv_value_promotions_init_kernel(
    uint32_t* __restrict__ value_promote_index,
    uint32_t* __restrict__ kv_counters,
    uint32_t* __restrict__ any_promoted,
    uint32_t* __restrict__ head_promoted_flags,
    float* __restrict__ e_val_by_head,
    int q_heads,
    int kv_heads,
    int num_blocks,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = linear; i < kv_heads * num_blocks; i += gridDim.x * blockDim.x) {
        value_promote_index[i] = UINT32_MAX;
    }
    for (int i = linear; i < kv_heads; i += gridDim.x * blockDim.x) {
        kv_counters[i] = 0;
    }
    for (int i = linear; i < q_heads; i += gridDim.x * blockDim.x) {
        head_promoted_flags[i] = 0;
        e_val_by_head[i] = 0.0f;
    }
    if (linear == 0) {
        any_promoted[0] = 0;
    }
}

__global__ void certified_kv_all_promoted_indices_init_kernel(
    uint32_t* __restrict__ promote_index,
    uint32_t* __restrict__ value_promote_index,
    int q_heads,
    int kv_heads,
    int num_blocks
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int key_total = q_heads * num_blocks;
    for (int i = linear; i < key_total; i += gridDim.x * blockDim.x) {
        promote_index[i] = static_cast<uint32_t>(i % num_blocks);
    }
    const int value_total = kv_heads * num_blocks;
    for (int i = linear; i < value_total; i += gridDim.x * blockDim.x) {
        value_promote_index[i] = UINT32_MAX;
    }
}

__global__ void certified_kv_value_promotions_from_block_masses_kernel(
    const float* __restrict__ block_mass,
    const float* __restrict__ value_error,
    const uint32_t* __restrict__ ranking_fallback_head_flags,
    uint32_t* __restrict__ value_promote_index,
    uint32_t* __restrict__ kv_counters,
    uint32_t* __restrict__ any_promoted,
    uint32_t* __restrict__ head_promoted_flags,
    float* __restrict__ e_val_by_head,
    int q_heads,
    int num_blocks,
    int value_error_stride_blocks,
    int gqa_group,
    float v_tol,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = q_heads * num_blocks;

    for (int i = linear; i < total; i += gridDim.x * blockDim.x) {
        const int qh = i / num_blocks;
        const int block = i - qh * num_blocks;
        if (ranking_fallback_head_flags != nullptr && ranking_fallback_head_flags[qh] != 0) {
            continue;
        }
        const int kvh = qh / gqa_group;
        const float mass = block_mass[static_cast<size_t>(qh) * num_blocks + block];
        const float err = value_error[static_cast<size_t>(kvh) * value_error_stride_blocks + block];
        const float contribution = mass * err;
        if (contribution > v_tol) {
            const size_t flag_idx = static_cast<size_t>(kvh) * num_blocks + block;
            const uint32_t old = atomicCAS(&value_promote_index[flag_idx], UINT32_MAX, UINT32_MAX - 1u);
            if (old == UINT32_MAX) {
                const uint32_t slot = atomicAdd(&kv_counters[kvh], 1u);
                value_promote_index[flag_idx] = slot;
                atomicExch(&any_promoted[0], 1u);
            }
            head_promoted_flags[qh] = 1u;
        } else {
            atomicAdd(&e_val_by_head[qh], contribution);
        }
    }
}

__global__ void certified_kv_all_promoted_value_kernel(
    const float* score_scratch,
    const __nv_bfloat16* promoted_value_bf16,
    const uint32_t* value_promote_index,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    const __nv_bfloat16* tail_value,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    const uint32_t* run_flag
) {
    if (!certified_kv_should_run(run_flag)) return;
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim <= 128) {
        __shared__ float value_partial[256];
        const int d = threadIdx.x % head_dim;
        const int lane = threadIdx.x / head_dim;
        const int lanes_per_dim = blockDim.x / head_dim;
        float acc = 0.0f;
        if (lane < lanes_per_dim) {
            for (int tok = lane; tok < total_tokens; tok += lanes_per_dim) {
                const float w =
                    score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
                float v;
                if (tok < aligned_tokens) {
                    const int block_id = tok / block_size;
                    const uint32_t promoted_value_slot =
                        value_promote_index[static_cast<size_t>(kvh) * num_blocks + block_id];
                    if (promoted_value_slot != 0xffffffffu &&
                        static_cast<int>(promoted_value_slot) < max_promoted_value_blocks) {
                        const int token_in_block = tok - block_id * block_size;
                        v = bf16_to_float(
                            promoted_value_bf16[
                                ((static_cast<size_t>(kvh) * max_promoted_value_blocks +
                                  promoted_value_slot) *
                                     block_size +
                                 token_in_block) *
                                    head_dim +
                                d
                            ]
                        );
                    } else {
                        v = certified_kv_dequant_int4_value(
                            value_int4,
                            value_scale,
                            value_zero,
                            kvh,
                            tok,
                            d,
                            value_stride_tokens,
                            head_dim,
                            value_group_size
                        );
                    }
                } else {
                    const int tail_tok = tok - aligned_tokens;
                    v = bf16_to_float(
                        tail_value[
                            (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                             tail_value_start_tokens + tail_tok) * head_dim + d
                        ]
                    );
                }
                acc += w * v;
            }
        }
        value_partial[threadIdx.x] = acc;
        __syncthreads();
        if (lane == 0) {
            float sum = 0.0f;
            for (int l = 0; l < lanes_per_dim; ++l) {
                sum += value_partial[l * head_dim + d];
            }
            output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(sum);
        }
    }
}

__global__ void certified_kv_dense_selected_heads_kernel(
    const __nv_bfloat16* query,
    const uint32_t* fallback_heads,
    const uint32_t* fallback_kv_slots,
    const uint32_t* fallback_kv_heads,
    const __nv_bfloat16* fallback_key,
    const __nv_bfloat16* fallback_value,
    const __nv_bfloat16* tail_key,
    const __nv_bfloat16* tail_value,
    float* score_scratch,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int fallback_count,
    int fallback_kv_count,
    int prefix_tokens,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    float q_scale
) {
    const int fallback_idx = blockIdx.x;
    if (fallback_idx >= fallback_count) return;
    const int qh = static_cast<int>(fallback_heads[fallback_idx]);
    if (qh < 0 || qh >= q_heads) return;
    const int kv_slot = static_cast<int>(fallback_kv_slots[fallback_idx]);
    if (kv_slot < 0 || kv_slot >= fallback_kv_count) return;
    const int kvh = static_cast<int>(fallback_kv_heads[kv_slot]);
    const int total_tokens = prefix_tokens + tail_len;

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            float k;
            if (tok < prefix_tokens) {
                k = bf16_to_float(
                    fallback_key[(static_cast<size_t>(kv_slot) * prefix_tokens + tok) * head_dim + d]
                );
            } else {
                const int tail_tok = tok - prefix_tokens;
                k = bf16_to_float(
                    tail_key[
                        (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                         tail_key_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(fallback_idx) * score_stride_tokens + tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(fallback_idx) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);

    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(fallback_idx) * score_stride_tokens + tok] -
            max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(fallback_idx) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w =
                score_scratch[static_cast<size_t>(fallback_idx) * score_stride_tokens + tok];
            float v;
            if (tok < prefix_tokens) {
                v = bf16_to_float(
                    fallback_value[(static_cast<size_t>(kv_slot) * prefix_tokens + tok) * head_dim + d]
                );
            } else {
                const int tail_tok = tok - prefix_tokens;
                v = bf16_to_float(
                    tail_value[
                        (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                         tail_value_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += w * v;
        }
        output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(acc);
    }
}

__global__ void certified_kv_dense_flagged_heads_kernel(
    const __nv_bfloat16* query,
    const uint32_t* fallback_flags,
    const __nv_bfloat16* fallback_key,
    const __nv_bfloat16* fallback_value,
    const __nv_bfloat16* tail_key,
    const __nv_bfloat16* tail_value,
    float* score_scratch,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int prefix_tokens,
    int prefix_stride_tokens,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads || fallback_flags[qh] == 0u) return;
    const int kvh = qh / gqa_group;
    if (kvh < 0 || kvh >= kv_heads) return;
    const int total_tokens = prefix_tokens + tail_len;

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            float k;
            if (tok < prefix_tokens) {
                k = bf16_to_float(
                    fallback_key[(static_cast<size_t>(kvh) * prefix_stride_tokens + tok) * head_dim + d]
                );
            } else {
                const int tail_tok = tok - prefix_tokens;
                k = bf16_to_float(
                    tail_key[
                        (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                         tail_key_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
    __syncthreads();

    __shared__ float reduce_scratch[256];
    float local_max = -INFINITY;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_max = fmaxf(
            local_max,
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]
        );
    }
    const float max_score = block_reduce_max_256(local_max, reduce_scratch);

    float local_denom = 0.0f;
    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        local_denom += expf(
            score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] -
            max_score
        );
    }
    const float denom = block_reduce_sum_256(local_denom, reduce_scratch);

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w = score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
            float v;
            if (tok < prefix_tokens) {
                v = bf16_to_float(
                    fallback_value[(static_cast<size_t>(kvh) * prefix_stride_tokens + tok) * head_dim + d]
                );
            } else {
                const int tail_tok = tok - prefix_tokens;
                v = bf16_to_float(
                    tail_value[
                        (static_cast<size_t>(kvh) * tail_value_stride_tokens +
                         tail_value_start_tokens + tail_tok) * head_dim + d
                    ]
                );
            }
            acc += w * v;
        }
        output_bf16[static_cast<size_t>(qh) * head_dim + d] = __float2bfloat16(acc);
    }
}

__global__ void certified_kv_attend_int8_bf16_values_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const float* key_zero,
    const __nv_bfloat16* value_bf16,
    const __nv_bfloat16* tail_key,
    float* score_scratch,
    float* output_f32,
    __nv_bfloat16* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int value_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    const int qh = blockIdx.x;
    if (qh >= q_heads) return;
    const int kvh = qh / gqa_group;
    if (kvh >= kv_heads) return;
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;

    for (int tok = threadIdx.x; tok < aligned_tokens; tok += blockDim.x) {
        const int block_id = tok / block_size;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const int8_t kq = static_cast<int8_t>(
                key_int8[(static_cast<size_t>(kvh) * key_stride_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[
                    (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                ];
            const float kz =
                key_zero[
                    (static_cast<size_t>(kvh) * key_scale_stride_blocks + block_id) * head_dim + d
                ];
            acc += q * (static_cast<float>(kq) * ks + kz);
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] = acc * q_scale;
    }
    for (int tail_tok = threadIdx.x; tail_tok < tail_len; tail_tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const float k = bf16_to_float(
                tail_key[
                    (static_cast<size_t>(kvh) * tail_key_stride_tokens +
                     tail_key_start_tokens + tail_tok) * head_dim + d
                ]
            );
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * score_stride_tokens + aligned_tokens + tail_tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float max_score;
    __shared__ float denom;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int tok = 0; tok < total_tokens; ++tok) {
            m = fmaxf(m, score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok]);
        }
        float s = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            s += expf(score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok] - m);
        }
        max_score = m;
        denom = s;
    }
    __syncthreads();

    for (int tok = threadIdx.x; tok < total_tokens; tok += blockDim.x) {
        const size_t score_idx = static_cast<size_t>(qh) * score_stride_tokens + tok;
        score_scratch[score_idx] = expf(score_scratch[score_idx] - max_score) / denom;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w = score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
            const float v = bf16_to_float(
                value_bf16[(static_cast<size_t>(kvh) * value_stride_tokens + tok) * head_dim + d]
            );
            acc += w * v;
        }
        const size_t out_idx = static_cast<size_t>(qh) * head_dim + d;
        if (output_bf16 != nullptr) {
            output_bf16[out_idx] = __float2bfloat16(acc);
        } else {
            output_f32[out_idx] = acc;
        }
    }
}

} // namespace

extern "C" int dotcache_llama31_certified_kv_copy_step_bf16(
    size_t device_ordinal,
    const void* src_key_bf16,
    const void* src_value_bf16,
    void* dst_key_bf16,
    void* dst_value_bf16,
    int kv_heads,
    int dst_stride_tokens,
    int dst_token,
    int head_dim
) {
    if (src_key_bf16 == nullptr || src_value_bf16 == nullptr ||
        dst_key_bf16 == nullptr || dst_value_bf16 == nullptr) {
        return 127;
    }
    if (kv_heads <= 0 || dst_stride_tokens <= 0 || dst_token < 0 ||
        dst_token >= dst_stride_tokens || head_dim <= 0) {
        return 128;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = kv_heads * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    certified_kv_copy_step_bf16_kernel<<<blocks, threads>>>(
        static_cast<const __nv_bfloat16*>(src_key_bf16),
        static_cast<const __nv_bfloat16*>(src_value_bf16),
        static_cast<__nv_bfloat16*>(dst_key_bf16),
        static_cast<__nv_bfloat16*>(dst_value_bf16),
        kv_heads,
        dst_stride_tokens,
        dst_token,
        head_dim
    );
    if (cudaGetLastError() != cudaSuccess) return 129;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_copy_token_range_bf16(
    size_t device_ordinal,
    const void* src_key_bf16,
    const void* src_value_bf16,
    void* dst_key_bf16,
    void* dst_value_bf16,
    int kv_heads,
    int src_stride_tokens,
    int src_start_token,
    int dst_stride_tokens,
    int dst_start_token,
    int token_count,
    int head_dim
) {
    if (src_key_bf16 == nullptr || src_value_bf16 == nullptr ||
        dst_key_bf16 == nullptr || dst_value_bf16 == nullptr) {
        return 130;
    }
    if (kv_heads <= 0 || src_stride_tokens <= 0 || dst_stride_tokens <= 0 ||
        src_start_token < 0 || dst_start_token < 0 || token_count <= 0 || head_dim <= 0 ||
        src_start_token + token_count > src_stride_tokens ||
        dst_start_token + token_count > dst_stride_tokens) {
        return 131;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = kv_heads * token_count * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    certified_kv_copy_token_range_bf16_kernel<<<blocks, threads>>>(
        static_cast<const __nv_bfloat16*>(src_key_bf16),
        static_cast<const __nv_bfloat16*>(src_value_bf16),
        static_cast<__nv_bfloat16*>(dst_key_bf16),
        static_cast<__nv_bfloat16*>(dst_value_bf16),
        kv_heads,
        src_stride_tokens,
        src_start_token,
        dst_stride_tokens,
        dst_start_token,
        token_count,
        head_dim
    );
    if (cudaGetLastError() != cudaSuccess) return 132;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_quantize_bf16(
    size_t device_ordinal,
    const void* key_bf16,
    const void* value_bf16,
    void* key_int8,
    void* key_scale,
    void* key_zero,
    void* value_int4,
    void* value_scale,
    void* value_zero,
    void* value_error,
    void* value_norm,
    int num_kv_heads,
    int seq_len,
    int max_t,
    int head_dim,
    int block_size,
    int value_group_size
) {
    if (key_bf16 == nullptr || value_bf16 == nullptr || key_int8 == nullptr ||
        key_scale == nullptr || key_zero == nullptr || value_int4 == nullptr || value_scale == nullptr ||
        value_zero == nullptr || value_error == nullptr || value_norm == nullptr) {
        return 1;
    }
    if (num_kv_heads <= 0 || seq_len < 0 || max_t <= 0 || head_dim <= 0 ||
        block_size <= 0 || value_group_size <= 0) {
        return 2;
    }
    if (head_dim > 256 || head_dim % 2 != 0 || head_dim % value_group_size != 0) {
        return 3;
    }
    const int aligned_tokens = (seq_len / block_size) * block_size;
    if (aligned_tokens == 0) {
        return 0;
    }
    const int num_blocks = aligned_tokens / block_size;
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));

    if (cudaMemset(value_error, 0, static_cast<size_t>(num_kv_heads) * num_blocks * sizeof(float)) != cudaSuccess) {
        return 4;
    }
    if (cudaMemset(value_norm, 0, static_cast<size_t>(num_kv_heads) * num_blocks * sizeof(float)) != cudaSuccess) {
        return 4;
    }

    const int key_threads = 32;
    const int key_grid = num_kv_heads * num_blocks * head_dim;
    certified_kv_quantize_keys_kernel<<<key_grid, key_threads, key_threads * sizeof(float)>>>(
        static_cast<const __nv_bfloat16*>(key_bf16),
        static_cast<uint8_t*>(key_int8),
        static_cast<float*>(key_scale),
        static_cast<float*>(key_zero),
        num_kv_heads,
        max_t,
        aligned_tokens,
        head_dim,
        block_size
    );
    if (cudaGetLastError() != cudaSuccess) return 5;

    const int value_threads = 256;
    const int value_grid = num_kv_heads * aligned_tokens;
    certified_kv_quantize_values_kernel<<<value_grid, value_threads>>>(
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<uint8_t*>(value_int4),
        static_cast<__half*>(value_scale),
        static_cast<__half*>(value_zero),
        static_cast<float*>(value_error),
        static_cast<float*>(value_norm),
        num_kv_heads,
        max_t,
        aligned_tokens,
        head_dim,
        block_size,
        value_group_size
    );
    if (cudaGetLastError() != cudaSuccess) return 6;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_quantize_keys_bf16(
    size_t device_ordinal,
    const void* key_bf16,
    void* key_int8,
    void* key_scale,
    void* key_zero,
    int num_kv_heads,
    int seq_len,
    int max_t,
    int head_dim,
    int block_size
) {
    if (key_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr) {
        return 1;
    }
    if (num_kv_heads <= 0 || seq_len < 0 || max_t <= 0 || head_dim <= 0 ||
        block_size <= 0 || head_dim > 256) {
        return 2;
    }
    const int aligned_tokens = (seq_len / block_size) * block_size;
    if (aligned_tokens == 0) {
        return 0;
    }
    const int num_blocks = aligned_tokens / block_size;
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));

    const int key_threads = 32;
    const int key_grid = num_kv_heads * num_blocks * head_dim;
    certified_kv_quantize_keys_kernel<<<key_grid, key_threads, key_threads * sizeof(float)>>>(
        static_cast<const __nv_bfloat16*>(key_bf16),
        static_cast<uint8_t*>(key_int8),
        static_cast<float*>(key_scale),
        static_cast<float*>(key_zero),
        num_kv_heads,
        max_t,
        aligned_tokens,
        head_dim,
        block_size
    );
    if (cudaGetLastError() != cudaSuccess) return 3;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_quantize_keys_bf16_range(
    size_t device_ordinal,
    const void* key_bf16,
    void* key_int8,
    void* key_scale,
    void* key_zero,
    int num_kv_heads,
    int max_t,
    int key_stride_tokens,
    int scale_stride_blocks,
    int start_block,
    int block_count,
    int head_dim,
    int block_size
) {
    if (key_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr) {
        return 1;
    }
    if (num_kv_heads <= 0 || max_t <= 0 || key_stride_tokens <= 0 ||
        scale_stride_blocks <= 0 || start_block < 0 || block_count < 0 ||
        head_dim <= 0 || block_size <= 0 || head_dim > 256) {
        return 2;
    }
    if (block_count == 0) {
        return 0;
    }
    const int end_block = start_block + block_count;
    if (end_block > scale_stride_blocks || end_block * block_size > key_stride_tokens ||
        end_block * block_size > max_t) {
        return 3;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));

    const int key_threads = 32;
    const int key_grid = num_kv_heads * block_count * head_dim;
    certified_kv_quantize_keys_range_kernel<<<key_grid, key_threads, key_threads * sizeof(float)>>>(
        static_cast<const __nv_bfloat16*>(key_bf16),
        static_cast<uint8_t*>(key_int8),
        static_cast<float*>(key_scale),
        static_cast<float*>(key_zero),
        num_kv_heads,
        max_t,
        key_stride_tokens,
        scale_stride_blocks,
        start_block,
        block_count,
        head_dim,
        block_size
    );
    if (cudaGetLastError() != cudaSuccess) return 4;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_quantize_values_bf16_range(
    size_t device_ordinal,
    const void* value_bf16,
    void* value_int4,
    void* value_scale,
    void* value_zero,
    void* value_error,
    void* value_norm,
    int num_kv_heads,
    int max_t,
    int value_stride_tokens,
    int value_error_stride_blocks,
    int start_block,
    int block_count,
    int head_dim,
    int block_size,
    int value_group_size
) {
    if (value_bf16 == nullptr || value_int4 == nullptr || value_scale == nullptr ||
        value_zero == nullptr || value_error == nullptr || value_norm == nullptr) {
        return 1;
    }
    if (num_kv_heads <= 0 || max_t <= 0 || value_stride_tokens <= 0 ||
        value_error_stride_blocks <= 0 || start_block < 0 || block_count < 0 ||
        head_dim <= 0 || block_size <= 0 || value_group_size <= 0 ||
        head_dim > 256 || (head_dim % value_group_size) != 0) {
        return 2;
    }
    if (block_count == 0) {
        return 0;
    }
    const int end_block = start_block + block_count;
    if (end_block > value_error_stride_blocks || end_block * block_size > value_stride_tokens ||
        end_block * block_size > max_t) {
        return 3;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));

    const int value_threads = 256;
    const int token_count = block_count * block_size;
    const int start_token = start_block * block_size;
    const int value_grid = num_kv_heads * token_count;
    certified_kv_quantize_values_range_kernel<<<value_grid, value_threads>>>(
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<uint8_t*>(value_int4),
        static_cast<__half*>(value_scale),
        static_cast<__half*>(value_zero),
        static_cast<float*>(value_error),
        static_cast<float*>(value_norm),
        num_kv_heads,
        max_t,
        value_stride_tokens,
        value_error_stride_blocks,
        start_token,
        token_count,
        head_dim,
        block_size,
        value_group_size
    );
    if (cudaGetLastError() != cudaSuccess) return 4;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_score_blocks_int8(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    void* block_max,
    void* block_sum,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        block_max == nullptr || block_sum == nullptr) {
        return 11;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks < 0 || block_size <= 0 ||
        key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        head_dim <= 0 || gqa_group <= 0) {
        return 12;
    }
    if (num_blocks == 0) {
        return 0;
    }
    if (block_size > 256 || q_heads != kv_heads * gqa_group ||
        key_stride_tokens < num_blocks * block_size ||
        key_scale_stride_blocks < num_blocks) {
        return 13;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_score_blocks_int8_kernel<<<q_heads * num_blocks, block_size>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<float*>(block_max),
        static_cast<float*>(block_sum),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        key_stride_tokens,
        key_scale_stride_blocks,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 14;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_key_scale_norms(
    size_t device_ordinal,
    const void* key_scale,
    void* key_scale_norm,
    int kv_heads,
    int num_blocks,
    int key_scale_stride_blocks,
    int head_dim
) {
    if (key_scale == nullptr || key_scale_norm == nullptr) {
        return 76;
    }
    if (kv_heads <= 0 || num_blocks <= 0 || key_scale_stride_blocks < num_blocks ||
        head_dim <= 0) {
        return 77;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = kv_heads * num_blocks;
    const int threads = 128;
    certified_kv_key_scale_norms_kernel<<<(total + threads - 1) / threads, threads>>>(
        static_cast<const float*>(key_scale),
        static_cast<float*>(key_scale_norm),
        kv_heads,
        num_blocks,
        key_scale_stride_blocks,
        head_dim
    );
    if (cudaGetLastError() != cudaSuccess) return 78;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_score_consistency(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* promoted_key_bf16,
    const void* promote_index,
    void* violation_flags,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int max_promoted_blocks,
    int head_dim,
    int gqa_group,
    float q_scale,
    float eps_guard
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
        key_zero == nullptr || promoted_key_bf16 == nullptr || promote_index == nullptr ||
        violation_flags == nullptr) {
        return 16;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        max_promoted_blocks <= 0 || head_dim <= 0 || gqa_group <= 0) {
        return 17;
    }
    if (q_heads != kv_heads * gqa_group || block_size > 256 ||
        key_stride_tokens < num_blocks * block_size ||
        key_scale_stride_blocks < num_blocks) {
        return 18;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    if (cudaMemset(violation_flags, 0, static_cast<size_t>(q_heads) * sizeof(uint32_t)) != cudaSuccess) {
        return 19;
    }
    certified_kv_score_consistency_kernel<<<q_heads * num_blocks, 32>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const __nv_bfloat16*>(promoted_key_bf16),
        static_cast<const uint32_t*>(promote_index),
        static_cast<uint32_t*>(violation_flags),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        key_stride_tokens,
        key_scale_stride_blocks,
        max_promoted_blocks,
        head_dim,
        gqa_group,
        q_scale,
        eps_guard
    );
    if (cudaGetLastError() != cudaSuccess) return 20;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_gather_promoted_bf16(
    size_t device_ordinal,
    const void* tier2_key_bf16,
    const void* tier2_value_bf16,
    const void* promote_index,
    const void* value_promote_index,
    void* promoted_key_bf16,
    void* promoted_value_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int promoted_key_heads,
    int max_promoted_blocks,
    int max_promoted_value_blocks,
    int head_dim,
    int gqa_group
) {
    if (tier2_key_bf16 == nullptr || tier2_value_bf16 == nullptr ||
        promote_index == nullptr || value_promote_index == nullptr ||
        promoted_key_bf16 == nullptr || promoted_value_bf16 == nullptr) {
        return 22;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        cap_tokens <= 0 || max_promoted_blocks <= 0 ||
        max_promoted_value_blocks <= 0 || head_dim <= 0 || gqa_group <= 0 ||
        (promoted_key_heads != q_heads && promoted_key_heads != kv_heads) ||
        (promoted_key_heads == kv_heads && max_promoted_blocks != num_blocks)) {
        return 23;
    }
    if (q_heads != kv_heads * gqa_group || cap_tokens < num_blocks * block_size) {
        return 24;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    constexpr int threads = 256;
    if (promoted_key_heads == kv_heads && max_promoted_blocks == num_blocks) {
        certified_kv_gather_all_promoted_keys_compact_kernel<<<kv_heads * num_blocks, threads>>>(
            static_cast<const __nv_bfloat16*>(tier2_key_bf16),
            static_cast<__nv_bfloat16*>(promoted_key_bf16),
            kv_heads,
            num_blocks,
            block_size,
            cap_tokens,
            head_dim
        );
    } else {
        certified_kv_gather_promoted_keys_gqa_union_kernel<<<kv_heads * num_blocks, threads>>>(
            static_cast<const __nv_bfloat16*>(tier2_key_bf16),
            static_cast<const uint32_t*>(promote_index),
            static_cast<__nv_bfloat16*>(promoted_key_bf16),
            q_heads,
            kv_heads,
            num_blocks,
            block_size,
            cap_tokens,
            max_promoted_blocks,
            head_dim,
            gqa_group
        );
    }
    if (cudaGetLastError() != cudaSuccess) return 25;
    certified_kv_gather_promoted_values_kernel<<<kv_heads * num_blocks, threads>>>(
        static_cast<const __nv_bfloat16*>(tier2_value_bf16),
        static_cast<const uint32_t*>(value_promote_index),
        static_cast<__nv_bfloat16*>(promoted_value_bf16),
        kv_heads,
        num_blocks,
        block_size,
        cap_tokens,
        max_promoted_value_blocks,
        head_dim,
        nullptr
    );
    if (cudaGetLastError() != cudaSuccess) return 26;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_init_key_cache(
    size_t device_ordinal,
    void* cache_tags,
    void* cache_lru,
    int q_heads,
    int cache_blocks
) {
    if (cache_tags == nullptr || cache_lru == nullptr) return 120;
    if (q_heads <= 0 || cache_blocks <= 0) return 121;
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = q_heads * cache_blocks;
    const int threads = 256;
    certified_kv_key_cache_init_kernel<<<(total + threads - 1) / threads, threads>>>(
        static_cast<uint32_t*>(cache_tags),
        static_cast<uint32_t*>(cache_lru),
        total
    );
    if (cudaGetLastError() != cudaSuccess) return 122;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_resolve_key_cache(
    size_t device_ordinal,
    const void* selected_blocks,
    const void* selected_counts,
    void* cache_tags,
    void* cache_lru,
    void* promote_index,
    void* gather_index,
    void* counters,
    int q_heads,
    int num_blocks,
    int max_selected_blocks,
    int cache_blocks,
    unsigned int tick_base
) {
    if (selected_blocks == nullptr || selected_counts == nullptr || cache_tags == nullptr ||
        cache_lru == nullptr || promote_index == nullptr || gather_index == nullptr ||
        counters == nullptr) {
        return 123;
    }
    if (q_heads <= 0 || num_blocks <= 0 || max_selected_blocks <= 0 ||
        cache_blocks <= 0 || cache_blocks < max_selected_blocks) {
        return 124;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    cudaMemsetAsync(counters, 0, 3 * sizeof(uint32_t), nullptr);
    if (cudaGetLastError() != cudaSuccess) return 125;
    certified_kv_key_cache_resolve_kernel<<<q_heads, 1>>>(
        static_cast<const uint32_t*>(selected_blocks),
        static_cast<const uint32_t*>(selected_counts),
        static_cast<uint32_t*>(cache_tags),
        static_cast<uint32_t*>(cache_lru),
        static_cast<uint32_t*>(promote_index),
        static_cast<uint32_t*>(gather_index),
        static_cast<uint32_t*>(counters),
        q_heads,
        num_blocks,
        max_selected_blocks,
        cache_blocks,
        tick_base
    );
    if (cudaGetLastError() != cudaSuccess) return 126;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_gather_promoted_values_bf16(
    size_t device_ordinal,
    const void* tier2_value_bf16,
    const void* value_promote_index,
    void* promoted_value_bf16,
    int kv_heads,
    int num_blocks,
    int block_size,
    int cap_tokens,
    int max_promoted_value_blocks,
    int head_dim,
    const void* run_flag
) {
    if (tier2_value_bf16 == nullptr || value_promote_index == nullptr ||
        promoted_value_bf16 == nullptr) {
        return 27;
    }
    if (kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 || cap_tokens <= 0 ||
        max_promoted_value_blocks <= 0 || head_dim <= 0 ||
        cap_tokens < num_blocks * block_size) {
        return 28;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    constexpr int threads = 256;
    certified_kv_gather_promoted_values_kernel<<<kv_heads * num_blocks, threads>>>(
        static_cast<const __nv_bfloat16*>(tier2_value_bf16),
        static_cast<const uint32_t*>(value_promote_index),
        static_cast<__nv_bfloat16*>(promoted_value_bf16),
        kv_heads,
        num_blocks,
        block_size,
        cap_tokens,
        max_promoted_value_blocks,
        head_dim,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 29;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_selected_fp16_log_masses(
    size_t device_ordinal,
    const void* query_bf16,
    const void* promoted_key_bf16,
    const void* promote_index,
    void* out_log_masses,
    int q_heads,
    int num_blocks,
    int block_size,
    int max_promoted_blocks,
    int head_dim,
    float q_scale
) {
    if (query_bf16 == nullptr || promoted_key_bf16 == nullptr ||
        promote_index == nullptr || out_log_masses == nullptr) {
        return 28;
    }
    if (q_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        max_promoted_blocks <= 0 || head_dim <= 0) {
        return 29;
    }
    if (block_size > 256 || max_promoted_blocks > num_blocks) {
        return 30;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_selected_fp16_log_mass_kernel<<<q_heads * max_promoted_blocks, block_size, block_size * sizeof(float)>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const __nv_bfloat16*>(promoted_key_bf16),
        static_cast<const uint32_t*>(promote_index),
        static_cast<float*>(out_log_masses),
        q_heads,
        num_blocks,
        block_size,
        max_promoted_blocks,
        head_dim,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 31;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_select_blocks(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_scale_norm,
    const void* block_max,
    const void* block_sum,
    const void* value_norm,
    void* promote_index,
    void* value_promote_index,
    void* selected_blocks,
    void* selected_counts,
    void* fallback_flags,
    void* delta_blocks,
    void* e_key_by_head,
    void* delta_tail_by_head,
    void* vmax_by_head,
    void* true_tail_by_head,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int key_scale_norm_stride_blocks,
    int value_norm_stride_blocks,
    int head_dim,
    int gqa_group,
    int k_min,
    int k_max,
    int max_promoted_blocks,
    float q_scale,
    float tau_cov,
    float rung1_threshold,
    float rung1_multiplier,
    float delta_guard_factor,
    float score_exploration_rate,
    int require_certified_tail_bound
) {
    if (query_bf16 == nullptr || key_scale_norm == nullptr || block_max == nullptr ||
        block_sum == nullptr || value_norm == nullptr || promote_index == nullptr ||
        value_promote_index == nullptr || selected_blocks == nullptr ||
        selected_counts == nullptr || fallback_flags == nullptr || delta_blocks == nullptr ||
        e_key_by_head == nullptr || delta_tail_by_head == nullptr ||
        vmax_by_head == nullptr || true_tail_by_head == nullptr) {
        return 80;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 ||
        num_blocks > kCertifiedSelectorMaxBlocks || key_scale_norm_stride_blocks < num_blocks ||
        value_norm_stride_blocks < num_blocks || head_dim <= 0 || gqa_group <= 0 ||
        q_heads != kv_heads * gqa_group || max_promoted_blocks <= 0) {
        return 81;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int init_total =
        q_heads * num_blocks +
        kv_heads * num_blocks +
        q_heads * max_promoted_blocks +
        q_heads * 5 +
        q_heads * num_blocks;
    const int threads = 256;
    certified_kv_selector_init_kernel<<<(init_total + threads - 1) / threads, threads>>>(
        static_cast<uint32_t*>(promote_index),
        static_cast<uint32_t*>(value_promote_index),
        static_cast<uint32_t*>(selected_blocks),
        static_cast<uint32_t*>(selected_counts),
        static_cast<uint32_t*>(fallback_flags),
        static_cast<float*>(delta_blocks),
        static_cast<float*>(e_key_by_head),
        static_cast<float*>(delta_tail_by_head),
        static_cast<float*>(vmax_by_head),
        static_cast<float*>(true_tail_by_head),
        q_heads,
        kv_heads,
        num_blocks,
        max_promoted_blocks
    );
    if (cudaGetLastError() != cudaSuccess) return 82;
    certified_kv_select_blocks_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const float*>(key_scale_norm),
        static_cast<const float*>(block_max),
        static_cast<const float*>(block_sum),
        static_cast<const float*>(value_norm),
        static_cast<uint32_t*>(promote_index),
        static_cast<uint32_t*>(selected_blocks),
        static_cast<uint32_t*>(selected_counts),
        static_cast<uint32_t*>(fallback_flags),
        static_cast<float*>(delta_blocks),
        static_cast<float*>(e_key_by_head),
        static_cast<float*>(delta_tail_by_head),
        static_cast<float*>(vmax_by_head),
        static_cast<float*>(true_tail_by_head),
        q_heads,
        kv_heads,
        num_blocks,
        key_scale_norm_stride_blocks,
        value_norm_stride_blocks,
        head_dim,
        gqa_group,
        k_min,
        k_max,
        max_promoted_blocks,
        q_scale,
        tau_cov,
        rung1_threshold,
        rung1_multiplier,
        delta_guard_factor,
        score_exploration_rate,
        require_certified_tail_bound
    );
    if (cudaGetLastError() != cudaSuccess) return 83;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_ranking_flags(
    size_t device_ordinal,
    const void* block_max,
    const void* block_sum,
    const void* delta_blocks,
    const void* selected_fp16_log_masses,
    const void* promote_index,
    void* fallback_flags,
    int q_heads,
    int num_blocks,
    int max_promoted_blocks
) {
    if (block_max == nullptr || block_sum == nullptr || delta_blocks == nullptr ||
        selected_fp16_log_masses == nullptr || promote_index == nullptr ||
        fallback_flags == nullptr) {
        return 84;
    }
    if (q_heads <= 0 || num_blocks <= 0 || max_promoted_blocks <= 0) {
        return 85;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_ranking_flags_kernel<<<q_heads, 32>>>(
        static_cast<const float*>(block_max),
        static_cast<const float*>(block_sum),
        static_cast<const float*>(delta_blocks),
        static_cast<const float*>(selected_fp16_log_masses),
        static_cast<const uint32_t*>(promote_index),
        static_cast<uint32_t*>(fallback_flags),
        q_heads,
        num_blocks,
        max_promoted_blocks
    );
    if (cudaGetLastError() != cudaSuccess) return 86;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_int4(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        score_scratch == nullptr || output_f32 == nullptr) {
        return 21;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 22;
    }
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        block_size > 256 || q_heads != kv_heads * gqa_group) {
        return 23;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_int4_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        nullptr,
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 24;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        tail_key_bf16 == nullptr || tail_value_bf16 == nullptr ||
        score_scratch == nullptr || output_f32 == nullptr) {
        return 31;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len <= 0 || head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 32;
    }
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        block_size > 256 || q_heads != kv_heads * gqa_group) {
        return 33;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_int4_bf16_tail_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        nullptr,
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        num_blocks * block_size,
        num_blocks,
        num_blocks * block_size,
        0,
        tail_len,
        0,
        tail_len,
        num_blocks * block_size + tail_len,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 34;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        score_scratch == nullptr || output_f32 == nullptr) {
        return 61;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 62;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        value_stride_tokens <= 0 || tail_key_start_tokens < 0 || tail_value_start_tokens < 0 ||
        score_stride_tokens <= 0 || head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 63;
    }
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        block_size > 256 || q_heads != kv_heads * gqa_group ||
        key_stride_tokens < aligned_tokens || key_scale_stride_blocks < num_blocks ||
        value_stride_tokens < aligned_tokens || score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 64;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_int4_bf16_tail_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        nullptr,
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        key_stride_tokens,
        key_scale_stride_blocks,
        value_stride_tokens,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 65;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_bf16_values(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_bf16,
    const void* tail_key_bf16,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_bf16 == nullptr || score_scratch == nullptr || output_f32 == nullptr) {
        return 41;
    }
    if (tail_len > 0 && tail_key_bf16 == nullptr) {
        return 42;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || head_dim <= 0 || gqa_group <= 0) {
        return 43;
    }
    if (block_size > 256 || q_heads != kv_heads * gqa_group) {
        return 44;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_bf16_values_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        nullptr,
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        num_blocks * block_size,
        num_blocks,
        0,
        tail_len,
        num_blocks * block_size + tail_len,
        num_blocks * block_size + tail_len,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 45;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_bf16_values_strided(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_bf16,
    const void* tail_key_bf16,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int value_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_bf16 == nullptr || score_scratch == nullptr || output_f32 == nullptr) {
        return 51;
    }
    if (tail_len > 0 && tail_key_bf16 == nullptr) {
        return 52;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        tail_key_start_tokens < 0 || score_stride_tokens <= 0 || value_stride_tokens <= 0 ||
        head_dim <= 0 || gqa_group <= 0) {
        return 53;
    }
    const int total_tokens = num_blocks * block_size + tail_len;
    if (block_size > 256 || q_heads != kv_heads * gqa_group ||
        key_stride_tokens < num_blocks * block_size || key_scale_stride_blocks < num_blocks ||
        score_stride_tokens < total_tokens || value_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len)) {
        return 54;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_bf16_values_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        nullptr,
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        key_stride_tokens,
        key_scale_stride_blocks,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        score_stride_tokens,
        value_stride_tokens,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 55;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_int4_bf16_tail_strided_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        score_scratch == nullptr || output_bf16 == nullptr) {
        return 71;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 72;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        value_stride_tokens <= 0 || tail_key_start_tokens < 0 || tail_value_start_tokens < 0 ||
        score_stride_tokens <= 0 || head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 73;
    }
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        block_size > 256 || q_heads != kv_heads * gqa_group ||
        key_stride_tokens < aligned_tokens || key_scale_stride_blocks < num_blocks ||
        value_stride_tokens < aligned_tokens || score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 74;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_int4_bf16_tail_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        nullptr,
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        key_stride_tokens,
        key_scale_stride_blocks,
        value_stride_tokens,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 75;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_mixed_key_int4_bf16_tail_strided_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* promoted_key_bf16,
    const void* promote_index,
    const void* promoted_value_bf16,
    const void* value_promote_index,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int promoted_key_heads,
    int max_promoted_blocks,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale,
    const void* run_flag
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
        promoted_key_bf16 == nullptr || promote_index == nullptr ||
        promoted_value_bf16 == nullptr || value_promote_index == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        score_scratch == nullptr || output_bf16 == nullptr) {
        return 81;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 82;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        promoted_key_heads <= 0 || max_promoted_blocks <= 0 || max_promoted_value_blocks <= 0 ||
        value_stride_tokens <= 0 ||
        tail_key_start_tokens < 0 || tail_value_start_tokens < 0 ||
        score_stride_tokens <= 0 || head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 83;
    }
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        block_size > 256 || q_heads != kv_heads * gqa_group ||
        (promoted_key_heads != q_heads && promoted_key_heads != kv_heads) ||
        (promoted_key_heads == kv_heads && max_promoted_blocks != num_blocks) ||
        key_stride_tokens < aligned_tokens || key_scale_stride_blocks < num_blocks ||
        value_stride_tokens < aligned_tokens ||
        score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 84;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    if (head_dim <= 128) {
        const int total_chunks = num_blocks + (tail_len > 0 ? 1 : 0);
        certified_kv_mixed_key_score_kernel<<<q_heads * total_chunks, 256>>>(
            static_cast<const __nv_bfloat16*>(query_bf16),
            static_cast<const uint8_t*>(key_int8),
            static_cast<const float*>(key_scale),
            static_cast<const float*>(key_zero),
            static_cast<const __nv_bfloat16*>(promoted_key_bf16),
            static_cast<const uint32_t*>(promote_index),
            static_cast<const __nv_bfloat16*>(tail_key_bf16),
            static_cast<float*>(score_scratch),
            q_heads,
            kv_heads,
            num_blocks,
            block_size,
            tail_len,
            key_stride_tokens,
            key_scale_stride_blocks,
            promoted_key_heads,
            max_promoted_blocks,
            tail_key_start_tokens,
            tail_key_stride_tokens,
            score_stride_tokens,
            head_dim,
            gqa_group,
            q_scale,
            static_cast<const uint32_t*>(run_flag)
        );
        if (cudaGetLastError() != cudaSuccess) return 85;

        certified_kv_softmax_normalize_inplace_kernel<<<q_heads, 256>>>(
            static_cast<float*>(score_scratch),
            q_heads,
            total_tokens,
            score_stride_tokens,
            static_cast<const uint32_t*>(run_flag)
        );
        if (cudaGetLastError() != cudaSuccess) return 86;
        certified_kv_mixed_value_by_dim_kernel<<<q_heads * head_dim, 256>>>(
            static_cast<const float*>(score_scratch),
            static_cast<const __nv_bfloat16*>(promoted_value_bf16),
            static_cast<const uint32_t*>(value_promote_index),
            static_cast<const uint8_t*>(value_int4),
            static_cast<const __half*>(value_scale),
            static_cast<const __half*>(value_zero),
            static_cast<const __nv_bfloat16*>(tail_value_bf16),
            static_cast<__nv_bfloat16*>(output_bf16),
            q_heads,
            kv_heads,
            num_blocks,
            block_size,
            tail_len,
            max_promoted_value_blocks,
            value_stride_tokens,
            tail_value_start_tokens,
            tail_value_stride_tokens,
            score_stride_tokens,
            head_dim,
            value_group_size,
            gqa_group,
            static_cast<const uint32_t*>(run_flag)
        );
        if (cudaGetLastError() != cudaSuccess) return 87;
        return 0;
    }
    certified_kv_attend_mixed_key_int4_bf16_tail_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const __nv_bfloat16*>(promoted_key_bf16),
        static_cast<const uint32_t*>(promote_index),
        static_cast<const __nv_bfloat16*>(promoted_value_bf16),
        static_cast<const uint32_t*>(value_promote_index),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        key_stride_tokens,
        key_scale_stride_blocks,
        promoted_key_heads,
        max_promoted_blocks,
        max_promoted_value_blocks,
        value_stride_tokens,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 85;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_all_promoted_int4_bf16_tail_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* promoted_key_bf16,
    const void* promoted_value_bf16,
    const void* value_promote_index,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* softmax_stats,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group,
    float q_scale,
    const void* run_flag
) {
    if (query_bf16 == nullptr || promoted_key_bf16 == nullptr ||
        promoted_value_bf16 == nullptr || value_promote_index == nullptr ||
        value_int4 == nullptr || value_scale == nullptr || value_zero == nullptr ||
        score_scratch == nullptr || softmax_stats == nullptr || output_bf16 == nullptr) {
        return 91;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 92;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || max_promoted_value_blocks <= 0 || value_stride_tokens <= 0 ||
        tail_key_start_tokens < 0 || tail_value_start_tokens < 0 ||
        score_stride_tokens <= 0 || head_dim <= 0 || value_group_size <= 0 || gqa_group <= 0) {
        return 93;
    }
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        head_dim > 128 || block_size > 256 || q_heads != kv_heads * gqa_group ||
        value_stride_tokens < aligned_tokens || score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 94;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total_chunks = num_blocks + (tail_len > 0 ? 1 : 0);
    certified_kv_all_promoted_score_kernel<<<q_heads * total_chunks, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const __nv_bfloat16*>(promoted_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        score_stride_tokens,
        head_dim,
        gqa_group,
        q_scale,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 95;
    certified_kv_softmax_stats_kernel<<<q_heads, 256>>>(
        static_cast<float*>(score_scratch),
        static_cast<float*>(softmax_stats),
        q_heads,
        total_tokens,
        score_stride_tokens,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 96;
    certified_kv_softmax_normalize_kernel<<<q_heads, 256>>>(
        static_cast<float*>(score_scratch),
        static_cast<const float*>(softmax_stats),
        q_heads,
        total_tokens,
        score_stride_tokens,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 97;
    certified_kv_all_promoted_value_kernel<<<q_heads, 256>>>(
        static_cast<const float*>(score_scratch),
        static_cast<const __nv_bfloat16*>(promoted_value_bf16),
        static_cast<const uint32_t*>(value_promote_index),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        max_promoted_value_blocks,
        value_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        value_group_size,
        gqa_group,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 98;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_score_all_promoted_bf16_keys(
    size_t device_ordinal,
    const void* query_bf16,
    const void* promoted_key_bf16,
    const void* tail_key_bf16,
    void* score_scratch,
    void* softmax_stats,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || promoted_key_bf16 == nullptr ||
        score_scratch == nullptr || softmax_stats == nullptr) {
        return 111;
    }
    if (tail_len > 0 && tail_key_bf16 == nullptr) {
        return 112;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || tail_key_start_tokens < 0 || score_stride_tokens <= 0 ||
        head_dim <= 0 || gqa_group <= 0) {
        return 113;
    }
    const int total_tokens = num_blocks * block_size + tail_len;
    if (head_dim > 128 || block_size > 256 || q_heads != kv_heads * gqa_group ||
        score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len)) {
        return 114;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total_chunks = num_blocks + (tail_len > 0 ? 1 : 0);
    certified_kv_all_promoted_score_kernel<<<q_heads * total_chunks, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const __nv_bfloat16*>(promoted_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        score_stride_tokens,
        head_dim,
        gqa_group,
        q_scale,
        nullptr
    );
    if (cudaGetLastError() != cudaSuccess) return 115;
    certified_kv_softmax_stats_kernel<<<q_heads, 256>>>(
        static_cast<float*>(score_scratch),
        static_cast<float*>(softmax_stats),
        q_heads,
        total_tokens,
        score_stride_tokens,
        nullptr
    );
    if (cudaGetLastError() != cudaSuccess) return 116;
    certified_kv_softmax_normalize_kernel<<<q_heads, 256>>>(
        static_cast<float*>(score_scratch),
        static_cast<const float*>(softmax_stats),
        q_heads,
        total_tokens,
        score_stride_tokens,
        nullptr
    );
    if (cudaGetLastError() != cudaSuccess) return 117;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_apply_all_promoted_values_from_probs(
    size_t device_ordinal,
    const void* score_scratch,
    const void* promoted_value_bf16,
    const void* value_promote_index,
    const void* value_int4,
    const void* value_scale,
    const void* value_zero,
    const void* tail_value_bf16,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int max_promoted_value_blocks,
    int value_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int value_group_size,
    int gqa_group
) {
    if (score_scratch == nullptr || promoted_value_bf16 == nullptr ||
        value_promote_index == nullptr || value_int4 == nullptr ||
        value_scale == nullptr || value_zero == nullptr || output_bf16 == nullptr) {
        return 118;
    }
    if (tail_len > 0 && tail_value_bf16 == nullptr) {
        return 119;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || max_promoted_value_blocks <= 0 || value_stride_tokens <= 0 ||
        tail_value_start_tokens < 0 || score_stride_tokens <= 0 || head_dim <= 0 ||
        value_group_size <= 0 || gqa_group <= 0) {
        return 120;
    }
    const int aligned_tokens = num_blocks * block_size;
    const int total_tokens = aligned_tokens + tail_len;
    if (head_dim % 2 != 0 || head_dim % value_group_size != 0 ||
        head_dim > 128 || block_size > 256 || q_heads != kv_heads * gqa_group ||
        value_stride_tokens < aligned_tokens || score_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 121;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_all_promoted_value_kernel<<<q_heads, 256>>>(
        static_cast<const float*>(score_scratch),
        static_cast<const __nv_bfloat16*>(promoted_value_bf16),
        static_cast<const uint32_t*>(value_promote_index),
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        max_promoted_value_blocks,
        value_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        value_group_size,
        gqa_group,
        nullptr
    );
    if (cudaGetLastError() != cudaSuccess) return 122;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_block_masses_from_probs(
    size_t device_ordinal,
    const void* score_scratch,
    void* block_mass,
    int q_heads,
    int num_blocks,
    int block_size,
    int score_stride_tokens,
    const void* run_flag
) {
    if (score_scratch == nullptr || block_mass == nullptr) {
        return 101;
    }
    if (q_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        block_size > 256 || score_stride_tokens < num_blocks * block_size) {
        return 102;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_block_mass_from_probs_kernel<<<q_heads * num_blocks, 256>>>(
        static_cast<const float*>(score_scratch),
        static_cast<float*>(block_mass),
        q_heads,
        num_blocks,
        block_size,
        score_stride_tokens,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 103;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_value_promotions_from_block_masses(
    size_t device_ordinal,
    const void* block_mass,
    const void* value_error,
    const void* ranking_fallback_head_flags,
    void* value_promote_index,
    void* kv_counters,
    void* any_promoted,
    void* head_promoted_flags,
    void* e_val_by_head,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int value_error_stride_blocks,
    int gqa_group,
    float v_tol,
    const void* run_flag
) {
    if (block_mass == nullptr || value_error == nullptr || value_promote_index == nullptr ||
        kv_counters == nullptr || any_promoted == nullptr || head_promoted_flags == nullptr ||
        e_val_by_head == nullptr) {
        return 104;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 ||
        value_error_stride_blocks < num_blocks || gqa_group <= 0 ||
        q_heads != kv_heads * gqa_group) {
        return 105;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = q_heads * num_blocks;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    certified_kv_value_promotions_init_kernel<<<blocks, threads>>>(
        static_cast<uint32_t*>(value_promote_index),
        static_cast<uint32_t*>(kv_counters),
        static_cast<uint32_t*>(any_promoted),
        static_cast<uint32_t*>(head_promoted_flags),
        static_cast<float*>(e_val_by_head),
        q_heads,
        kv_heads,
        num_blocks,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 106;
    certified_kv_value_promotions_from_block_masses_kernel<<<blocks, threads>>>(
        static_cast<const float*>(block_mass),
        static_cast<const float*>(value_error),
        static_cast<const uint32_t*>(ranking_fallback_head_flags),
        static_cast<uint32_t*>(value_promote_index),
        static_cast<uint32_t*>(kv_counters),
        static_cast<uint32_t*>(any_promoted),
        static_cast<uint32_t*>(head_promoted_flags),
        static_cast<float*>(e_val_by_head),
        q_heads,
        num_blocks,
        value_error_stride_blocks,
        gqa_group,
        v_tol,
        static_cast<const uint32_t*>(run_flag)
    );
    if (cudaGetLastError() != cudaSuccess) return 107;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_init_all_promoted_indices(
    size_t device_ordinal,
    void* promote_index,
    void* value_promote_index,
    int q_heads,
    int kv_heads,
    int num_blocks
) {
    if (promote_index == nullptr || value_promote_index == nullptr) {
        return 108;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0) {
        return 109;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    const int total = max(q_heads * num_blocks, kv_heads * num_blocks);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    certified_kv_all_promoted_indices_init_kernel<<<blocks, threads>>>(
        static_cast<uint32_t*>(promote_index),
        static_cast<uint32_t*>(value_promote_index),
        q_heads,
        kv_heads,
        num_blocks
    );
    if (cudaGetLastError() != cudaSuccess) return 110;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_dense_selected_heads_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* fallback_heads,
    const void* fallback_kv_slots,
    const void* fallback_kv_heads,
    const void* fallback_key_bf16,
    const void* fallback_value_bf16,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_bf16,
    int q_heads,
    int fallback_count,
    int fallback_kv_count,
    int prefix_tokens,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    float q_scale
) {
    if (query_bf16 == nullptr || fallback_heads == nullptr ||
        fallback_kv_slots == nullptr || fallback_kv_heads == nullptr ||
        fallback_key_bf16 == nullptr || fallback_value_bf16 == nullptr ||
        score_scratch == nullptr || output_bf16 == nullptr) {
        return 91;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 94;
    }
    const int total_tokens = prefix_tokens + tail_len;
    if (q_heads <= 0 || fallback_count <= 0 || fallback_kv_count <= 0 || prefix_tokens < 0 ||
        tail_len < 0 || total_tokens <= 0 || score_stride_tokens < total_tokens ||
        head_dim <= 0 || tail_key_start_tokens < 0 || tail_value_start_tokens < 0 ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 92;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_dense_selected_heads_kernel<<<fallback_count, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint32_t*>(fallback_heads),
        static_cast<const uint32_t*>(fallback_kv_slots),
        static_cast<const uint32_t*>(fallback_kv_heads),
        static_cast<const __nv_bfloat16*>(fallback_key_bf16),
        static_cast<const __nv_bfloat16*>(fallback_value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        fallback_count,
        fallback_kv_count,
        prefix_tokens,
        tail_len,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 93;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_dense_flagged_heads_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* fallback_flags,
    const void* fallback_key_bf16,
    const void* fallback_value_bf16,
    const void* tail_key_bf16,
    const void* tail_value_bf16,
    void* score_scratch,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int prefix_tokens,
    int prefix_stride_tokens,
    int tail_len,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int tail_value_start_tokens,
    int tail_value_stride_tokens,
    int score_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || fallback_flags == nullptr ||
        fallback_key_bf16 == nullptr || fallback_value_bf16 == nullptr ||
        score_scratch == nullptr || output_bf16 == nullptr) {
        return 123;
    }
    if (tail_len > 0 && (tail_key_bf16 == nullptr || tail_value_bf16 == nullptr)) {
        return 124;
    }
    const int total_tokens = prefix_tokens + tail_len;
    if (q_heads <= 0 || kv_heads <= 0 || prefix_tokens < 0 ||
        prefix_stride_tokens < prefix_tokens || tail_len < 0 || total_tokens <= 0 ||
        score_stride_tokens < total_tokens || head_dim <= 0 || gqa_group <= 0 ||
        q_heads != kv_heads * gqa_group || tail_key_start_tokens < 0 ||
        tail_value_start_tokens < 0 ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len) ||
        (tail_len > 0 && tail_value_stride_tokens < tail_value_start_tokens + tail_len)) {
        return 125;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_dense_flagged_heads_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint32_t*>(fallback_flags),
        static_cast<const __nv_bfloat16*>(fallback_key_bf16),
        static_cast<const __nv_bfloat16*>(fallback_value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        prefix_tokens,
        prefix_stride_tokens,
        tail_len,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        tail_value_start_tokens,
        tail_value_stride_tokens,
        score_stride_tokens,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 126;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_bf16_values_strided_out_bf16(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    const void* key_zero,
    const void* value_bf16,
    const void* tail_key_bf16,
    void* score_scratch,
    void* output_bf16,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int key_stride_tokens,
    int key_scale_stride_blocks,
    int tail_key_start_tokens,
    int tail_key_stride_tokens,
    int score_stride_tokens,
    int value_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr || key_zero == nullptr ||
        value_bf16 == nullptr || score_scratch == nullptr || output_bf16 == nullptr) {
        return 81;
    }
    if (tail_len > 0 && tail_key_bf16 == nullptr) {
        return 82;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || key_stride_tokens <= 0 || key_scale_stride_blocks <= 0 ||
        tail_key_start_tokens < 0 || score_stride_tokens <= 0 || value_stride_tokens <= 0 ||
        head_dim <= 0 || gqa_group <= 0) {
        return 83;
    }
    const int total_tokens = num_blocks * block_size + tail_len;
    if (block_size > 256 || q_heads != kv_heads * gqa_group ||
        key_stride_tokens < num_blocks * block_size || key_scale_stride_blocks < num_blocks ||
        score_stride_tokens < total_tokens || value_stride_tokens < total_tokens ||
        (tail_len > 0 && tail_key_stride_tokens < tail_key_start_tokens + tail_len)) {
        return 84;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_bf16_values_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const float*>(key_zero),
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        nullptr,
        static_cast<__nv_bfloat16*>(output_bf16),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        key_stride_tokens,
        key_scale_stride_blocks,
        tail_key_start_tokens,
        tail_key_stride_tokens,
        score_stride_tokens,
        value_stride_tokens,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 85;
    return 0;
}
