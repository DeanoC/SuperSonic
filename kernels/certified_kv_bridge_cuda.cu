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
        const __nv_bfloat16 value = tier2_key_bf16[src_base + idx];
        for (int local_qh = 0; local_qh < gqa_group; ++local_qh) {
            const int qh = first_qh + local_qh;
            if (qh >= q_heads) {
                continue;
            }
            const uint32_t slot = promote_index[static_cast<size_t>(qh) * num_blocks + block];
            if (slot == 0xffffffffu || static_cast<int>(slot) >= max_promoted_blocks) {
                continue;
            }
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
    int head_dim
) {
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
        scale = __shfl_sync(0xffffffff, scale, leader_lane);
        zero = __shfl_sync(0xffffffff, zero, leader_lane);
        return static_cast<float>(q) * scale + zero;
    }

    return static_cast<float>(q) * __half2float(value_scale[meta_idx]) + __half2float(value_zero[meta_idx]);
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
        const uint32_t promoted_slot =
            promote_index[static_cast<size_t>(qh) * num_blocks + block_id];
        const bool promote = promoted_slot != 0xffffffffu;
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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w = score_scratch[static_cast<size_t>(qh) * score_stride_tokens + tok];
        float v;
        if (tok < aligned_tokens) {
            const int block_id = tok / block_size;
            const uint32_t promoted_value_slot =
                value_promote_index[static_cast<size_t>(kvh) * num_blocks + block_id];
            if (promoted_value_slot != 0xffffffffu) {
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 7;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
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
        head_dim
    );
    if (cudaGetLastError() != cudaSuccess) return 26;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 25;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 35;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 66;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 46;
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
    if (cudaDeviceSynchronize() != cudaSuccess) return 56;
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
    float q_scale
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
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 85;
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
