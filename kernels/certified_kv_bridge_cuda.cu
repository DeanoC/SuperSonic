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
    float local_absmax = 0.0f;
    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        local_absmax = fmaxf(local_absmax, fabsf(bf16_to_float(key[src_idx])));
    }
    scratch[threadIdx.x] = local_absmax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float scale = fmaxf(scratch[0], 1.0e-8f) / 127.0f;
    if (threadIdx.x == 0) {
        const size_t scale_idx = (static_cast<size_t>(kvh) * (aligned_tokens / block_size) + block_id) * head_dim + dim;
        key_scale[scale_idx] = scale;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < block_size; t += blockDim.x) {
        const int tok = block_id * block_size + t;
        const size_t src_idx = (static_cast<size_t>(kvh) * max_t + tok) * head_dim + dim;
        const float x = bf16_to_float(key[src_idx]);
        const int q = clamp_i32(static_cast<int>(nearbyintf(x / scale)), -127, 127);
        const size_t dst_idx = (static_cast<size_t>(kvh) * aligned_tokens + tok) * head_dim + dim;
        key_int8[dst_idx] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

__global__ void certified_kv_quantize_values_kernel(
    const __nv_bfloat16* value,
    uint8_t* value_int4,
    __half* value_scale,
    __half* value_zero,
    float* value_error,
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
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float token_l2 = sqrtf(err_scratch[0]);
        const int block_id = tok / block_size;
        const int num_blocks = aligned_tokens / block_size;
        atomic_max_nonnegative_float(value_error + kvh * num_blocks + block_id, token_l2);
    }
}

__global__ void certified_kv_score_blocks_int8_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    float* block_max,
    float* block_sum,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
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
                key_int8[(static_cast<size_t>(kvh) * (num_blocks * block_size) + tok) * head_dim + d]
            );
            const float ks =
                key_scale[(static_cast<size_t>(kvh) * num_blocks + block_id) * head_dim + d];
            acc += q * (static_cast<float>(kq) * ks);
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

__device__ __forceinline__ float certified_kv_dequant_int4_value(
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    int kvh,
    int tok,
    int d,
    int aligned_tokens,
    int head_dim,
    int value_group_size
) {
    const int packed_dim = head_dim / 2;
    const int groups = head_dim / value_group_size;
    const uint8_t packed =
        value_int4[(static_cast<size_t>(kvh) * aligned_tokens + tok) * packed_dim + (d / 2)];
    const int q = (d & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const int group = d / value_group_size;
    const size_t meta_idx = (static_cast<size_t>(kvh) * aligned_tokens + tok) * groups + group;
    return static_cast<float>(q) * __half2float(value_scale[meta_idx]) + __half2float(value_zero[meta_idx]);
}

__global__ void certified_kv_attend_int8_int4_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    float* score_scratch,
    float* output_f32,
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
            acc += q * (static_cast<float>(kq) * ks);
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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < aligned_tokens; ++tok) {
            const float w =
                expf(score_scratch[static_cast<size_t>(qh) * aligned_tokens + tok] - max_score) / denom;
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
        output_f32[static_cast<size_t>(qh) * head_dim + d] = acc;
    }
}

__global__ void certified_kv_attend_int8_int4_bf16_tail_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const uint8_t* value_int4,
    const __half* value_scale,
    const __half* value_zero,
    const __nv_bfloat16* tail_key,
    const __nv_bfloat16* tail_value,
    float* score_scratch,
    float* output_f32,
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
                key_int8[(static_cast<size_t>(kvh) * aligned_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[(static_cast<size_t>(kvh) * num_blocks + block_id) * head_dim + d];
            acc += q * (static_cast<float>(kq) * ks);
        }
        score_scratch[static_cast<size_t>(qh) * total_tokens + tok] = acc * q_scale;
    }
    for (int tail_tok = threadIdx.x; tail_tok < tail_len; tail_tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const float k = bf16_to_float(
                tail_key[(static_cast<size_t>(kvh) * tail_len + tail_tok) * head_dim + d]
            );
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * total_tokens + aligned_tokens + tail_tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float max_score;
    __shared__ float denom;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int tok = 0; tok < total_tokens; ++tok) {
            m = fmaxf(m, score_scratch[static_cast<size_t>(qh) * total_tokens + tok]);
        }
        float s = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            s += expf(score_scratch[static_cast<size_t>(qh) * total_tokens + tok] - m);
        }
        max_score = m;
        denom = s;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w =
                expf(score_scratch[static_cast<size_t>(qh) * total_tokens + tok] - max_score) / denom;
            float v;
            if (tok < aligned_tokens) {
                v = certified_kv_dequant_int4_value(
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
            } else {
                const int tail_tok = tok - aligned_tokens;
                v = bf16_to_float(
                    tail_value[(static_cast<size_t>(kvh) * tail_len + tail_tok) * head_dim + d]
                );
            }
            acc += w * v;
        }
        output_f32[static_cast<size_t>(qh) * head_dim + d] = acc;
    }
}

__global__ void certified_kv_attend_int8_bf16_values_kernel(
    const __nv_bfloat16* query,
    const uint8_t* key_int8,
    const float* key_scale,
    const __nv_bfloat16* value_bf16,
    const __nv_bfloat16* tail_key,
    float* score_scratch,
    float* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
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
                key_int8[(static_cast<size_t>(kvh) * aligned_tokens + tok) * head_dim + d]
            );
            const float ks =
                key_scale[(static_cast<size_t>(kvh) * num_blocks + block_id) * head_dim + d];
            acc += q * (static_cast<float>(kq) * ks);
        }
        score_scratch[static_cast<size_t>(qh) * total_tokens + tok] = acc * q_scale;
    }
    for (int tail_tok = threadIdx.x; tail_tok < tail_len; tail_tok += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float q = bf16_to_float(query[static_cast<size_t>(qh) * head_dim + d]);
            const float k = bf16_to_float(
                tail_key[(static_cast<size_t>(kvh) * tail_len + tail_tok) * head_dim + d]
            );
            acc += q * k;
        }
        score_scratch[static_cast<size_t>(qh) * total_tokens + aligned_tokens + tail_tok] =
            acc * q_scale;
    }
    __syncthreads();

    __shared__ float max_score;
    __shared__ float denom;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int tok = 0; tok < total_tokens; ++tok) {
            m = fmaxf(m, score_scratch[static_cast<size_t>(qh) * total_tokens + tok]);
        }
        float s = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            s += expf(score_scratch[static_cast<size_t>(qh) * total_tokens + tok] - m);
        }
        max_score = m;
        denom = s;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tok = 0; tok < total_tokens; ++tok) {
            const float w =
                expf(score_scratch[static_cast<size_t>(qh) * total_tokens + tok] - max_score) / denom;
            const float v = bf16_to_float(
                value_bf16[(static_cast<size_t>(kvh) * value_stride_tokens + tok) * head_dim + d]
            );
            acc += w * v;
        }
        output_f32[static_cast<size_t>(qh) * head_dim + d] = acc;
    }
}

} // namespace

extern "C" int dotcache_llama31_certified_kv_quantize_bf16(
    size_t device_ordinal,
    const void* key_bf16,
    const void* value_bf16,
    void* key_int8,
    void* key_scale,
    void* value_int4,
    void* value_scale,
    void* value_zero,
    void* value_error,
    int num_kv_heads,
    int seq_len,
    int max_t,
    int head_dim,
    int block_size,
    int value_group_size
) {
    if (key_bf16 == nullptr || value_bf16 == nullptr || key_int8 == nullptr ||
        key_scale == nullptr || value_int4 == nullptr || value_scale == nullptr ||
        value_zero == nullptr || value_error == nullptr) {
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

    const int key_threads = 32;
    const int key_grid = num_kv_heads * num_blocks * head_dim;
    certified_kv_quantize_keys_kernel<<<key_grid, key_threads, key_threads * sizeof(float)>>>(
        static_cast<const __nv_bfloat16*>(key_bf16),
        static_cast<uint8_t*>(key_int8),
        static_cast<float*>(key_scale),
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
    int num_kv_heads,
    int seq_len,
    int max_t,
    int head_dim,
    int block_size
) {
    if (key_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr) {
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

extern "C" int dotcache_llama31_certified_kv_score_blocks_int8(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
    void* block_max,
    void* block_sum,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
        block_max == nullptr || block_sum == nullptr) {
        return 11;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks < 0 || block_size <= 0 ||
        head_dim <= 0 || gqa_group <= 0) {
        return 12;
    }
    if (num_blocks == 0) {
        return 0;
    }
    if (block_size > 256 || q_heads != kv_heads * gqa_group) {
        return 13;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_score_blocks_int8_kernel<<<q_heads * num_blocks, block_size>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<float*>(block_max),
        static_cast<float*>(block_sum),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 14;
    if (cudaDeviceSynchronize() != cudaSuccess) return 15;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_int4(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
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
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
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
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
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
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
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
        static_cast<const uint8_t*>(value_int4),
        static_cast<const __half*>(value_scale),
        static_cast<const __half*>(value_zero),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<const __nv_bfloat16*>(tail_value_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        head_dim,
        value_group_size,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 34;
    if (cudaDeviceSynchronize() != cudaSuccess) return 35;
    return 0;
}

extern "C" int dotcache_llama31_certified_kv_attend_int8_bf16_values(
    size_t device_ordinal,
    const void* query_bf16,
    const void* key_int8,
    const void* key_scale,
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
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
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
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
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
    const void* value_bf16,
    const void* tail_key_bf16,
    void* score_scratch,
    void* output_f32,
    int q_heads,
    int kv_heads,
    int num_blocks,
    int block_size,
    int tail_len,
    int value_stride_tokens,
    int head_dim,
    int gqa_group,
    float q_scale
) {
    if (query_bf16 == nullptr || key_int8 == nullptr || key_scale == nullptr ||
        value_bf16 == nullptr || score_scratch == nullptr || output_f32 == nullptr) {
        return 51;
    }
    if (tail_len > 0 && tail_key_bf16 == nullptr) {
        return 52;
    }
    if (q_heads <= 0 || kv_heads <= 0 || num_blocks <= 0 || block_size <= 0 ||
        tail_len < 0 || value_stride_tokens <= 0 || head_dim <= 0 || gqa_group <= 0) {
        return 53;
    }
    const int total_tokens = num_blocks * block_size + tail_len;
    if (block_size > 256 || q_heads != kv_heads * gqa_group ||
        value_stride_tokens < total_tokens) {
        return 54;
    }
    ScopedCudaDevice scoped(static_cast<int>(device_ordinal));
    certified_kv_attend_int8_bf16_values_kernel<<<q_heads, 256>>>(
        static_cast<const __nv_bfloat16*>(query_bf16),
        static_cast<const uint8_t*>(key_int8),
        static_cast<const float*>(key_scale),
        static_cast<const __nv_bfloat16*>(value_bf16),
        static_cast<const __nv_bfloat16*>(tail_key_bf16),
        static_cast<float*>(score_scratch),
        static_cast<float*>(output_f32),
        q_heads,
        kv_heads,
        num_blocks,
        block_size,
        tail_len,
        value_stride_tokens,
        head_dim,
        gqa_group,
        q_scale
    );
    if (cudaGetLastError() != cudaSuccess) return 55;
    if (cudaDeviceSynchronize() != cudaSuccess) return 56;
    return 0;
}
