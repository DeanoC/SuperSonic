#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

using hip_bfloat16 = __nv_bfloat16;

#ifndef __HIP_PLATFORM_AMD__
#define __shfl(val, lane) __shfl_sync(0xffffffffu, val, lane)
#define __shfl_down(val, delta) __shfl_down_sync(0xffffffffu, val, delta)
#define __shfl_xor(val, lane_mask) __shfl_xor_sync(0xffffffffu, val, lane_mask)
#endif
#pragma once

#include <math.h>
#include <stdint.h>

// Weight descriptor for the persistent decode megakernel.
// One struct per decoder layer (24 total for Qwen3.5-0.8B).
// Immutable weight pointers are const; mutable state pointers are non-const.
struct Qwen35DecodeLayerDesc {
    int layer_type;                    // 0=linear_attention, 1=full_attention
    int intermediate_size;             // MLP intermediate dim (3584)
    // --- RMSNorm weights (both layer types) ---
    const void* input_norm_w;          // [hidden_size] BF16
    float input_norm_eps;
    const void* post_attn_norm_w;      // [hidden_size] BF16
    float post_attn_norm_eps;
    // --- MLP weights (both layer types) ---
    const void* gate_proj_w;           // [intermediate_size, hidden_size] BF16
    const void* up_proj_w;             // [intermediate_size, hidden_size] BF16
    const void* down_proj_w;           // [hidden_size, intermediate_size] BF16
    // --- Linear attention weights (layer_type==0) ---
    const void* qkv_proj_w;           // [6144, hidden_size] BF16
    int qkv_out_dim;                   // 6144
    const void* z_proj_w;             // [2048, hidden_size] BF16
    int z_out_dim;                     // 2048
    const void* b_proj_w;             // [16, hidden_size] BF16
    const void* a_proj_w;             // [16, hidden_size] BF16
    const void* conv1d_w;             // [6144, 1, 4] BF16 (depthwise)
    int conv_kernel_size;              // 4
    const void* linear_out_proj_w;    // [hidden_size, 2048] BF16
    int linear_value_dim;              // 2048
    int linear_num_v_heads;            // 16
    int linear_head_k_dim;             // 128
    int linear_head_v_dim;             // 128
    const void* dt_bias_w;             // [num_v_heads] BF16 (decay bias)
    const void* a_log_exp_w;           // [num_v_heads] BF16 (decay scaling)
    const void* linear_norm_w;         // [value_dim] BF16 (gated RMSNorm weight)
    float linear_norm_eps;              // gated RMSNorm epsilon
    void* conv_state;                  // [6144, 3] BF16 mutable
    void* recurrent_state;             // [num_v_heads, head_k_dim, head_v_dim] F32 mutable
    // --- Full attention weights (layer_type==1) ---
    const void* q_proj_w;             // [4096, hidden_size] BF16
    int q_out_dim;                     // 4096
    const void* k_proj_w;             // [512, hidden_size] BF16
    int k_out_dim;                     // 512
    const void* v_proj_w;             // [512, hidden_size] BF16
    const void* o_proj_w;             // [hidden_size, 2048] BF16
    int attn_head_dim;                 // 256
    int attn_num_heads;                // 8
    int attn_num_kv_heads;             // 2
    const void* q_norm_w;             // [head_dim] BF16
    const void* k_norm_w;             // [head_dim] BF16
    float q_norm_eps;
    float k_norm_eps;
    void* kv_cache_k;                  // [num_kv_heads, max_T, head_dim] BF16 mutable
    void* kv_cache_v;                  // [num_kv_heads, max_T, head_dim] BF16 mutable
    int kv_len;                        // current cache length before this token
    int kv_max_t;                      // allocated T dimension of KV cache
    void* kv_shadow_k;                 // optional BF16 KV sidecar for KV-FP8 bring-up / parity-sensitive reads
    void* kv_shadow_v;                 // optional BF16 KV sidecar for KV-FP8 bring-up / parity-sensitive reads
    int kv_shadow_start;               // first position covered by the sidecar, -1 when disabled
};

template <typename T>
__device__ inline float dotcache_qwen35_to_float(T value);

template <>
__device__ inline float dotcache_qwen35_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float dotcache_qwen35_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float dotcache_qwen35_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T dotcache_qwen35_from_float(float value);

template <>
__device__ inline __half dotcache_qwen35_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float dotcache_qwen35_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 dotcache_qwen35_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float dotcache_qwen35_binary_op_apply(int op, float lhs, float rhs) {
    switch (op) {
    case 0:
        return lhs + rhs;
    case 1:
        return lhs - rhs;
    case 2:
        return lhs * rhs;
    case 3:
        return lhs / rhs;
    default:
        return 0.0f;
    }
}

__device__ inline size_t dotcache_qwen35_broadcast_elem_index(
    size_t out_index,
    int rank,
    const int* out_dims,
    const int* src_dims
) {
    if (rank == 0) {
        return 0;
    }
    size_t src_index = 0;
    size_t stride = 1;
    for (int dim = rank - 1; dim >= 0; --dim) {
        const size_t coord = out_index % static_cast<size_t>(out_dims[dim]);
        out_index /= static_cast<size_t>(out_dims[dim]);
        if (src_dims[dim] != 1) {
            src_index += coord * stride;
        }
        stride *= static_cast<size_t>(src_dims[dim]);
    }
    return src_index;
}

template <typename T>
__device__ inline float dotcache_qwen35_dot_row(
    const T* lhs,
    const T* rhs,
    int size
) {
    float dot = 0.0f;
    for (int idx = 0; idx < size; ++idx) {
        dot += dotcache_qwen35_to_float(lhs[idx]) * dotcache_qwen35_to_float(rhs[idx]);
    }
    return dot;
}

template <typename T>
__device__ inline float dotcache_qwen35_dot_row_input_f32_hero(
    const T* lhs,
    const float* rhs,
    int size
) {
    float dot = 0.0f;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    for (int idx = tid * 2; idx < size; idx += bs * 2) {
        dot += dotcache_qwen35_to_float(lhs[idx]) * rhs[idx];
        if (idx + 1 < size) {
            dot += dotcache_qwen35_to_float(lhs[idx + 1]) * rhs[idx + 1];
        }
    }
    return dot;
}

template <>
__device__ inline float dotcache_qwen35_dot_row_input_f32_hero<hip_bfloat16>(
    const hip_bfloat16* lhs,
    const float* rhs,
    int size
) {
    float dot = 0.0f;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    for (int idx = tid * 2; idx + 1 < size; idx += bs * 2) {
        const __nv_bfloat162 packed =
            *reinterpret_cast<const __nv_bfloat162*>(lhs + idx);
        const float2 w = __bfloat1622float2(packed);
        dot += w.x * rhs[idx] + w.y * rhs[idx + 1];
    }
    if ((size & 1) != 0 && tid == 0) {
        dot += dotcache_qwen35_to_float(lhs[size - 1]) * rhs[size - 1];
    }
    return dot;
}

template <>
__device__ inline float dotcache_qwen35_dot_row<__half>(
    const __half* lhs,
    const __half* rhs,
    int size
) {
    float dot = 0.0f;
    int idx = 0;
    for (; idx + 1 < size; idx += 2) {
        const __half2 lhs2 = __halves2half2(lhs[idx], lhs[idx + 1]);
        const __half2 rhs2 = __halves2half2(rhs[idx], rhs[idx + 1]);
        const float2 prod = __half22float2(__hmul2(lhs2, rhs2));
        dot += prod.x + prod.y;
    }
    if (idx < size) {
        dot += __half2float(lhs[idx]) * __half2float(rhs[idx]);
    }
    return dot;
}

__device__ inline float dotcache_qwen35_exp_fast(float x) {
    return __expf(x);
}

__device__ inline float dotcache_qwen35_sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = dotcache_qwen35_exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = dotcache_qwen35_exp_fast(x);
    return e / (1.0f + e);
}

__device__ inline float dotcache_qwen35_softplus_fast(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return dotcache_qwen35_exp_fast(x);
    return log1pf(dotcache_qwen35_exp_fast(x));
}

template <typename T>
__device__ inline float dotcache_qwen35_conv4_contiguous_t_ge3(
    const T* mixed_qkv,
    size_t mixed_c_offset,
    size_t t,
    const T* weights,
    size_t weight_offset
) {
    const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
    const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
    const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
    const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
    const float x0 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 3)]);
    const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 2)]);
    const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 1)]);
    const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + t]);
    return ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
}

template <>
__device__ inline float dotcache_qwen35_conv4_contiguous_t_ge3<__half>(
    const __half* mixed_qkv,
    size_t mixed_c_offset,
    size_t t,
    const __half* weights,
    size_t weight_offset
) {
    const __half2 x01 = __halves2half2(
        mixed_qkv[mixed_c_offset + (t - 3)],
        mixed_qkv[mixed_c_offset + (t - 2)]);
    const __half2 x23 = __halves2half2(
        mixed_qkv[mixed_c_offset + (t - 1)],
        mixed_qkv[mixed_c_offset + t]);
    const __half2 w01 = __halves2half2(
        weights[weight_offset],
        weights[weight_offset + 1]);
    const __half2 w23 = __halves2half2(
        weights[weight_offset + 2],
        weights[weight_offset + 3]);
    const float2 f01 = __half22float2(__hmul2(x01, w01));
    const float2 f23 = __half22float2(__hmul2(x23, w23));
    return (f01.x + f01.y) + (f23.x + f23.y);
}

__device__ inline float dotcache_qwen35_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    return value;
}

__device__ inline float dotcache_qwen35_block_sum_256(float value, float* scratch) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;

    value = dotcache_qwen35_wave_sum(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = (lane < 8) ? scratch[lane] : 0.0f;
        value = dotcache_qwen35_wave_sum(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

template <typename T>
__global__ void dotcache_qwen35_full_attention_prefill_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const T* query,
    const T* key,
    const T* value,
    float* out,
    unsigned int* row_counter
) {
    const int lane = threadIdx.x;
    if (lane >= warpSize) {
        return;
    }

    const int total_rows = batch_size * q_heads * q_len;
    while (true) {
        unsigned int row = 0;
        if (lane == 0) {
            row = atomicAdd(row_counter, 1u);
        }
        row = __shfl(row, 0);
        if ((int)row >= total_rows) {
            return;
        }

        const int q_pos = row % q_len;
        const int q_head = (row / q_len) % q_heads;
        const int batch = row / (q_len * q_heads);
        const int kv_head = q_head / num_kv_groups;
        const int causal_limit = min(kv_len, seqlen_offset + q_pos + 1);

        const T* q_row = query + (((batch * q_heads + q_head) * q_len + q_pos) * head_dim);
        const T* k_head_ptr = key + ((batch * kv_heads + kv_head) * kv_len * head_dim);
        const T* v_head_ptr = value + ((batch * kv_heads + kv_head) * kv_len * head_dim);
        float* out_row = out + (((batch * q_heads + q_head) * q_len + q_pos) * head_dim);

        float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int local_dims[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int local_count = 0;
        for (int d = lane; d < head_dim && local_count < 8; d += warpSize) {
            local_dims[local_count++] = d;
        }

        __shared__ float shared_max;
        __shared__ float shared_denom;
        __shared__ float shared_weight;
        __shared__ float shared_inv_denom;

        if (lane == 0) {
            shared_max = -INFINITY;
        }
        __syncthreads();

        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += dotcache_qwen35_to_float(q_row[d]) * dotcache_qwen35_to_float(k_row[d]);
            }
            float score = dotcache_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                if (score > shared_max) {
                    shared_max = score;
                }
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_denom = 0.0f;
        }
        __syncthreads();

        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;
            const T* v_row = v_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += dotcache_qwen35_to_float(q_row[d]) * dotcache_qwen35_to_float(k_row[d]);
            }
            float score = dotcache_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                shared_weight = expf(score - shared_max);
                shared_denom += shared_weight;
            }
            __syncthreads();

            for (int i = 0; i < local_count; ++i) {
                local_acc[i] += shared_weight * dotcache_qwen35_to_float(v_row[local_dims[i]]);
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * shared_inv_denom;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void dotcache_qwen35_linear_prefill_conv_pack_kernel(
    int batch_size,
    int conv_dim,
    int total_len,
    int seq_len,
    int kernel_size,
    const T* mixed_qkv,
    const T* weights,
    T* out
) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim);
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim));
    const size_t rem = tid - b * static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim);
    const size_t t = rem / static_cast<size_t>(conv_dim);
    const size_t c = rem - t * static_cast<size_t>(conv_dim);

    const size_t input_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(total_len);
    const size_t input_c_offset = input_b_offset + c * static_cast<size_t>(total_len);
    const size_t weight_offset = c * static_cast<size_t>(kernel_size);

    float acc = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        acc += dotcache_qwen35_to_float(mixed_qkv[input_c_offset + t + static_cast<size_t>(tap)]) *
            dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
    out[tid] = dotcache_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_kernel(
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    const T* mixed_qkv,
    const T* prev_state,
    const T* weights,
    T* out
) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim);
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim));
    const size_t rem = tid - b * static_cast<size_t>(seq_len) * static_cast<size_t>(conv_dim);
    const size_t t = rem / static_cast<size_t>(conv_dim);
    const size_t c = rem - t * static_cast<size_t>(conv_dim);

    const size_t mixed_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(seq_len);
    const size_t mixed_c_offset = mixed_b_offset + c * static_cast<size_t>(seq_len);
    const size_t state_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(state_len);
    const size_t state_c_offset = state_b_offset + c * static_cast<size_t>(state_len);
    const size_t weight_offset = c * static_cast<size_t>(kernel_size);

    float acc = 0.0f;
    const int history = kernel_size - 1;
    for (int tap = 0; tap < kernel_size; ++tap) {
        const int src = static_cast<int>(t) + tap - history;
        float x = 0.0f;
        if (src >= 0) {
            x = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
        } else {
            const int state_idx = state_len + src;
            if (state_idx >= 0) {
                x = dotcache_qwen35_to_float(prev_state[state_c_offset + static_cast<size_t>(state_idx)]);
            }
        }
        acc += x * dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
    out[tid] = dotcache_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_kernel(
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const T* mixed_qkv,
    const T* prev_state,
    const T* weights,
    const T* a,
    const T* dt_bias,
    const T* a_log_exp,
    T* out
) {
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * out_width;
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(seq_len) * out_width);
    const size_t rem = tid - b * static_cast<size_t>(seq_len) * out_width;
    const size_t t = rem / out_width;
    const size_t c = rem - t * out_width;

    if (c < static_cast<size_t>(conv_dim)) {
        const size_t mixed_b_offset =
            b * static_cast<size_t>(conv_dim) * static_cast<size_t>(seq_len);
        const size_t mixed_c_offset = mixed_b_offset + c * static_cast<size_t>(seq_len);
        const size_t state_b_offset =
            b * static_cast<size_t>(conv_dim) * static_cast<size_t>(state_len);
        const size_t state_c_offset = state_b_offset + c * static_cast<size_t>(state_len);
        const size_t weight_offset = c * static_cast<size_t>(kernel_size);

        float acc = 0.0f;
        const int history = kernel_size - 1;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int src = static_cast<int>(t) + tap - history;
            float x = 0.0f;
            if (src >= 0) {
                x = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
            } else {
                const int state_idx = state_len + src;
                if (state_idx >= 0) {
                    x = dotcache_qwen35_to_float(
                        prev_state[state_c_offset + static_cast<size_t>(state_idx)]
                    );
                }
            }
            acc += x * dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
        }

        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = dotcache_qwen35_softplus_fast(a_val + bias);
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_kernel_k4s3(
    int batch_size,
    int conv_dim,
    int seq_len,
    int num_heads,
    const T* mixed_qkv,
    const T* prev_state,
    const T* weights,
    const T* a,
    const T* dt_bias,
    const T* a_log_exp,
    T* out
) {
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * out_width;
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(seq_len) * out_width);
    const size_t rem = tid - b * static_cast<size_t>(seq_len) * out_width;
    const size_t t = rem / out_width;
    const size_t c = rem - t * out_width;

    if (c < static_cast<size_t>(conv_dim)) {
        const size_t mixed_b_offset =
            b * static_cast<size_t>(conv_dim) * static_cast<size_t>(seq_len);
        const size_t mixed_c_offset = mixed_b_offset + c * static_cast<size_t>(seq_len);
        const size_t state_b_offset =
            b * static_cast<size_t>(conv_dim) * 3u;
        const size_t state_c_offset = state_b_offset + c * 3u;
        const size_t weight_offset = c * 4u;

        float acc;
        if (t >= 3) {
            acc = dotcache_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = dotcache_qwen35_softplus_fast(a_val + bias);
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_with_state_kernel_k4s3(
    int batch_size,
    int conv_dim,
    int seq_len,
    int num_heads,
    const T* mixed_qkv,
    const T* prev_state,
    const T* weights,
    const T* a,
    const T* dt_bias,
    const T* a_log_exp,
    T* out
) {
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t packed_per_batch = static_cast<size_t>(seq_len) * out_width;
    const size_t state_per_batch = static_cast<size_t>(conv_dim) * 3u;
    const size_t total_per_batch = packed_per_batch + state_per_batch;
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t total_elems = static_cast<size_t>(batch_size) * total_per_batch;
    if (tid >= total_elems) {
        return;
    }

    const size_t b = tid / total_per_batch;
    const size_t batch_offset = tid - b * total_per_batch;
    const size_t mixed_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(seq_len);
    const size_t state_b_offset = b * static_cast<size_t>(conv_dim) * 3u;

    if (batch_offset >= packed_per_batch) {
        const size_t state_idx = batch_offset - packed_per_batch;
        const size_t c = state_idx / 3u;
        const size_t s = state_idx - c * 3u;
        const size_t mixed_c_offset = mixed_b_offset + c * static_cast<size_t>(seq_len);
        out[tid] = mixed_qkv[mixed_c_offset + static_cast<size_t>(seq_len - 3) + s];
        return;
    }

    const size_t t = batch_offset / out_width;
    const size_t c = batch_offset - t * out_width;

    if (c < static_cast<size_t>(conv_dim)) {
        const size_t mixed_c_offset = mixed_b_offset + c * static_cast<size_t>(seq_len);
        const size_t state_c_offset = state_b_offset + c * 3u;
        const size_t weight_offset = c * 4u;

        float acc;
        if (t >= 3) {
            acc = dotcache_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = dotcache_qwen35_softplus_fast(a_val + bias);
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, int MAX_K = 256>
__global__ void dotcache_qwen35_linear_decode_prepare_kernel(
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int state_len,
    int kernel_size,
    int head_repeat,
    const T* mixed_qkv,
    const T* prev_conv_state,
    const T* weights,
    const T* a_beta_raw,
    const T* dt_bias,
    const T* a_log_exp,
    float* out
) {
    const int pair = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int total_pairs = batch_size * num_v_heads;
    if (pair >= total_pairs || head_k_dim > MAX_K) {
        return;
    }

    const int value_dim = num_v_heads * head_v_dim;
    const int key_dim = (num_v_heads / head_repeat) * head_k_dim;
    const int conv_dim = key_dim * 2 + value_dim;
    const int packed_width = 2 * head_k_dim + head_v_dim + 2;

    const int batch = pair / num_v_heads;
    const int v_head = pair - batch * num_v_heads;
    const int k_head = v_head / head_repeat;
    const int mixed_batch_base = batch * conv_dim;
    const int state_batch_base = batch * conv_dim * state_len;
    const int pair_out_base = pair * packed_width;

    auto conv_channel = [&](int channel) -> float {
        const int weight_base = channel * kernel_size;
        const int state_base = state_batch_base + channel * state_len;
        float acc = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            float x = 0.0f;
            if (tap + 1 == kernel_size) {
                x = dotcache_qwen35_to_float(mixed_qkv[mixed_batch_base + channel]);
            } else if (tap < state_len) {
                x = dotcache_qwen35_to_float(prev_conv_state[state_base + tap]);
            }
            acc += x * dotcache_qwen35_to_float(weights[weight_base + tap]);
        }
        return acc * dotcache_qwen35_sigmoid_fast(acc);
    };

    __shared__ float shared_q[MAX_K];
    __shared__ float shared_k[MAX_K];
    if (tid < head_k_dim) {
        const int q_base = k_head * head_k_dim;
        const int k_base = key_dim + k_head * head_k_dim;
        shared_q[tid] = conv_channel(q_base + tid);
        shared_k[tid] = conv_channel(k_base + tid);
    }
    __syncthreads();

    if (tid == 0) {
        float q_sq_sum = 0.0f;
        float k_sq_sum = 0.0f;
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            q_sq_sum += shared_q[k_idx] * shared_q[k_idx];
            k_sq_sum += shared_k[k_idx] * shared_k[k_idx];
        }
        const float q_inv = rsqrtf(q_sq_sum + 1e-6f);
        const float k_inv = rsqrtf(k_sq_sum + 1e-6f);
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            out[pair_out_base + k_idx] = shared_q[k_idx] * q_inv * rsqrtf(static_cast<float>(head_k_dim));
            out[pair_out_base + head_k_dim + k_idx] = shared_k[k_idx] * k_inv;
        }
        const int head_base = batch * (2 * num_v_heads) + v_head;
        const float a_raw = dotcache_qwen35_to_float(a_beta_raw[head_base]);
        const float beta_raw = dotcache_qwen35_to_float(a_beta_raw[head_base + num_v_heads]);
        out[pair_out_base + 2 * head_k_dim + head_v_dim] = dotcache_qwen35_sigmoid_fast(beta_raw);
        const float bias = dotcache_qwen35_to_float(dt_bias[v_head]);
        const float decay = dotcache_qwen35_to_float(a_log_exp[v_head]);
        const float g = -dotcache_qwen35_softplus_fast(a_raw + bias) * decay;
        out[pair_out_base + 2 * head_k_dim + head_v_dim + 1] = dotcache_qwen35_exp_fast(g);
    }
    if (tid < head_v_dim) {
        const int value_channel = key_dim * 2 + v_head * head_v_dim + tid;
        out[pair_out_base + 2 * head_k_dim + tid] = conv_channel(value_channel);
    }
}

template <int MAX_K = 256>
__global__ void dotcache_qwen35_linear_decode_apply_kernel(
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const float* packed,
    const float* initial_state,
    float* out
) {
    const int pair = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int total_pairs = batch_size * num_v_heads;
    if (pair >= total_pairs || head_k_dim > MAX_K || tid >= head_v_dim) {
        return;
    }

    const int value_dim = num_v_heads * head_v_dim;
    const int packed_width = 2 * head_k_dim + head_v_dim + 2;
    const int batch = pair / num_v_heads;
    const int v_head = pair - batch * num_v_heads;
    const int v_idx = tid;
    const int pair_base = pair * packed_width;
    const int state_head_base =
        ((batch * num_v_heads + v_head) * head_k_dim) * head_v_dim + v_idx;
    const int out_base =
        batch * (value_dim + num_v_heads * head_k_dim * head_v_dim) + value_dim +
        (v_head * head_k_dim) * head_v_dim + v_idx;

    const float* q = packed + pair_base;
    const float* k = packed + pair_base + head_k_dim;
    const float* value = packed + pair_base + 2 * head_k_dim;
    const float beta = packed[pair_base + 2 * head_k_dim + head_v_dim];
    const float g_exp = packed[pair_base + 2 * head_k_dim + head_v_dim + 1];

    float state[MAX_K];
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        state[k_idx] = initial_state[state_head_base + k_idx * head_v_dim] * g_exp;
    }

    float kv_mem = 0.0f;
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        kv_mem += state[k_idx] * k[k_idx];
    }
    const float delta = (value[v_idx] - kv_mem) * beta;

    float out_value = 0.0f;
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        state[k_idx] += k[k_idx] * delta;
        out_value += state[k_idx] * q[k_idx];
        out[out_base + k_idx * head_v_dim] = state[k_idx];
    }
    out[batch * (value_dim + num_v_heads * head_k_dim * head_v_dim) + v_head * head_v_dim + v_idx] =
        out_value;
}

template <typename T, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_recurrent_prefill_impl(
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_stride_k = seq_len * k_head_dim;
    const int token_stride_v = seq_len * v_head_dim;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < seq_len; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + seq_len * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_recurrent_prefill_kernel(
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_recurrent_prefill_impl(
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        initial_state,
        query,
        key,
        value,
        beta,
        g,
        out,
        tid
    );
}

template <typename T, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_chunk_step_impl(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* prev_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_stride_k = chunk_size * k_head_dim;
    const int token_stride_v = chunk_size * v_head_dim;
    const int token_stride_s = chunk_size;
    const int out_base = bh * (chunk_size + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < chunk_size; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_step_kernel(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* prev_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_chunk_step_impl(
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        prev_state,
        query,
        key,
        value,
        beta,
        g,
        out,
        tid
    );
}

template <typename T, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_chunk_scan_raw_impl(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int total_tokens = num_chunks * chunk_size;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_stride_k = total_tokens * k_head_dim;
    const int token_stride_v = total_tokens * v_head_dim;
    const int token_stride_s = total_tokens;
    const int out_base = bh * (total_tokens + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < total_tokens; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + total_tokens * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_scan_raw_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_chunk_scan_raw_impl(
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        initial_state,
        query,
        key,
        value,
        beta,
        g,
        out,
        tid
    );
}

template <typename T, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_state_scan_impl(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* packed_scan,
    const T* value,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int packed_width = 2 * k_head_dim + 1;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        const int idx = bh * state_stride + k_idx * v_head_dim + v_idx;
        state[k_idx] = dotcache_qwen35_to_float(initial_state[idx]);
        out[idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
        const int value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
        const float state_decay =
            dotcache_qwen35_to_float(packed_scan[packed_chunk_base + 2 * k_head_dim]);
        float update[MAX_K];
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            update[k_idx] = 0.0f;
        }

        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_chunk_base + t * packed_width;
            const int value_row = value_chunk_base + t * v_head_dim;
            float v_prime = 0.0f;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += dotcache_qwen35_to_float(packed_scan[packed_row + k_head_dim + k_idx]) *
                    state[k_idx];
            }
            const float v_new = dotcache_qwen35_to_float(value[value_row + v_idx]) - v_prime;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                update[k_idx] += dotcache_qwen35_to_float(packed_scan[packed_row + k_idx]) * v_new;
            }
        }

        const int out_chunk_base = ((bh * (num_chunks + 1)) + (chunk + 1)) * state_stride;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] = state_decay * state[k_idx] + update[k_idx];
            out[out_chunk_base + k_idx * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(state[k_idx]);
        }
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_state_scan_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* packed_scan,
    const T* value,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_state_scan_impl(
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        initial_state,
        packed_scan,
        value,
        out,
        tid
    );
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void dotcache_qwen35_delta_chunk_fused_impl(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* prev_state,
    const T* packed_chunk,
    const T* value,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K || chunk_size > MAX_CHUNK) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int packed_width = 3 * k_head_dim + 1;
    const int chunk_out_stride = (2 * chunk_size + k_head_dim) * v_head_dim;
    const int packed_base = bh * chunk_size * packed_width;
    const int value_base = bh * chunk_size * v_head_dim;
    const int out_base = bh * chunk_out_stride;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    float v_new[MAX_CHUNK];
    float attn_inter[MAX_CHUNK];
    for (int t = 0; t < chunk_size; ++t) {
        const int packed_row = packed_base + t * packed_width;
        float v_prime = 0.0f;
        float attn = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            v_prime +=
                dotcache_qwen35_to_float(packed_chunk[packed_row + k_head_dim + k_idx]) *
                state[k_idx];
            attn +=
                dotcache_qwen35_to_float(packed_chunk[packed_row + 2 * k_head_dim + k_idx]) *
                state[k_idx];
        }
        v_new[t] = dotcache_qwen35_to_float(value[value_base + t * v_head_dim + v_idx]) - v_prime;
        attn_inter[t] = attn;
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(v_new[t]);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] =
            dotcache_qwen35_from_float<T>(attn_inter[t]);
    }

    const float state_decay = dotcache_qwen35_to_float(packed_chunk[packed_base + 3 * k_head_dim]);
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float update = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_base + t * packed_width;
            update += dotcache_qwen35_to_float(packed_chunk[packed_row + k_idx]) * v_new[t];
        }
        out[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
            dotcache_qwen35_from_float<T>(state_decay * state[k_idx] + update);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_fused_kernel(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* prev_state,
    const T* packed_chunk,
    const T* value,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_chunk_fused_impl(
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        prev_state,
        packed_chunk,
        value,
        out,
        tid
    );
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void dotcache_qwen35_delta_full_scan_impl(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* weighted_key_scan,
    const T* k_cumdecay_scan,
    const T* q_state_scan,
    const T* local_attn_scan,
    const T* state_decay_scan,
    const T* value,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K || chunk_size > MAX_CHUNK) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_count = num_chunks * chunk_size;
    const int scan_base = bh * num_chunks * chunk_size * k_head_dim;
    const int local_base = bh * num_chunks * chunk_size * chunk_size;
    const int decay_base = bh * num_chunks;
    const int value_base = bh * token_count * v_head_dim;
    const int out_base = bh * (token_count + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    float v_new[MAX_CHUNK];
    float attn_inter[MAX_CHUNK];
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int chunk_scan = scan_base + chunk * chunk_size * k_head_dim;
        const int chunk_local = local_base + chunk * chunk_size * chunk_size;
        const int chunk_value = value_base + chunk * chunk_size * v_head_dim;
        for (int t = 0; t < chunk_size; ++t) {
            float v_prime = 0.0f;
            float attn = 0.0f;
            const int row = chunk_scan + t * k_head_dim;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += dotcache_qwen35_to_float(k_cumdecay_scan[row + k_idx]) * state[k_idx];
                attn += dotcache_qwen35_to_float(q_state_scan[row + k_idx]) * state[k_idx];
            }
            v_new[t] = dotcache_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) -
                v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += dotcache_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = dotcache_qwen35_to_float(state_decay_scan[decay_base + chunk]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_scan + t * k_head_dim;
                update += dotcache_qwen35_to_float(weighted_key_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_local_attn_scan_flat_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const T* query_scan,
    const T* key_scan,
    const T* exp_g_scan,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_heads * num_chunks * chunk_size * chunk_size;
    if (tid >= total) {
        return;
    }

    const int col = tid % chunk_size;
    const int row_in_chunk = (tid / chunk_size) % chunk_size;
    const int chunk = (tid / (chunk_size * chunk_size)) % num_chunks;
    const int bh = tid / (num_chunks * chunk_size * chunk_size);

    if (col > row_in_chunk) {
        out[tid] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    const float dot = dotcache_qwen35_dot_row(
        query_scan + row_base,
        key_scan + col_base,
        k_head_dim);

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = dotcache_qwen35_from_float<T>(dot * decay);
}

template <typename T>
__global__ void dotcache_qwen35_delta_local_attn_scan_row_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const T* query_scan,
    const T* key_scan,
    const T* exp_g_scan,
    T* out
) {
    __shared__ T query_row[256];

    const int row_idx = static_cast<int>(blockIdx.x);
    const int total_rows = batch_heads * num_chunks * chunk_size;
    if (row_idx >= total_rows) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    const int row_in_chunk = row_idx % chunk_size;
    const int chunk = (row_idx / chunk_size) % num_chunks;
    const int bh = row_idx / (num_chunks * chunk_size);
    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;

    for (int k_idx = lane; k_idx < k_head_dim; k_idx += blockDim.x) {
        query_row[k_idx] = query_scan[row_base + k_idx];
    }
    __syncthreads();

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const int out_base = (bh * num_chunks + chunk) * chunk_size * chunk_size;

    for (int col = lane; col < chunk_size; col += blockDim.x) {
        const int out_idx = out_base + row_in_chunk * chunk_size + col;
        if (col > row_in_chunk) {
            out[out_idx] = dotcache_qwen35_from_float<T>(0.0f);
            continue;
        }

        const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
        const float dot = dotcache_qwen35_dot_row(
            query_row,
            key_scan + col_base,
            k_head_dim);

        const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
        const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
        out[out_idx] = dotcache_qwen35_from_float<T>(dot * decay);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_base_attn_scan_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const T* k_beta_scan,
    const T* key_scan,
    const T* exp_g_scan,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch_heads * num_chunks * chunk_size * chunk_size;
    if (tid >= total) {
        return;
    }

    const int col = tid % chunk_size;
    const int row_in_chunk = (tid / chunk_size) % chunk_size;
    const int chunk = (tid / (chunk_size * chunk_size)) % num_chunks;
    const int bh = tid / (num_chunks * chunk_size * chunk_size);

    if (col >= row_in_chunk) {
        out[tid] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    float dot = 0.0f;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        dot += dotcache_qwen35_to_float(k_beta_scan[row_base + k_idx]) *
               dotcache_qwen35_to_float(key_scan[col_base + k_idx]);
    }

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = dotcache_qwen35_from_float<T>(-dot * decay);
}

template <typename T, int MAX_CHUNK = 64>
__global__ void dotcache_qwen35_delta_attn_solve_scan_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    const T* base_attn_scan,
    T* out
) {
    const int matrix = static_cast<int>(blockIdx.x);
    const int total_mats = batch_heads * num_chunks;
    if (matrix >= total_mats || threadIdx.x != 0) {
        return;
    }

    const int stride = chunk_size * chunk_size;
    const int base = matrix * stride;
    float rows[MAX_CHUNK * MAX_CHUNK];
    #pragma unroll
    for (int idx = 0; idx < MAX_CHUNK * MAX_CHUNK; ++idx) {
        rows[idx] = 0.0f;
    }

    for (int i = 1; i < chunk_size; ++i) {
        for (int j = 0; j < i; ++j) {
            const float row_val = dotcache_qwen35_to_float(base_attn_scan[base + i * chunk_size + j]);
            float correction = 0.0f;
            for (int k = 0; k < i; ++k) {
                correction += dotcache_qwen35_to_float(base_attn_scan[base + i * chunk_size + k]) *
                    rows[k * chunk_size + j];
            }
            rows[i * chunk_size + j] = row_val + correction;
        }
    }

    for (int i = 0; i < chunk_size; ++i) {
        for (int j = 0; j < chunk_size; ++j) {
            float value = rows[i * chunk_size + j];
            if (i == j) {
                value += 1.0f;
            }
            out[base + i * chunk_size + j] = dotcache_qwen35_from_float<T>(value);
        }
    }
}

template <typename T, int MAX_CHUNK = 64>
__global__ void dotcache_qwen35_delta_attn_solve_from_inputs_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const T* k_beta_scan,
    const T* key_scan,
    const T* exp_g_scan,
    T* out
) {
    const int matrix = static_cast<int>(blockIdx.x);
    const int total_mats = batch_heads * num_chunks;
    if (matrix >= total_mats || threadIdx.x != 0) {
        return;
    }

    const int chunk_base = matrix * chunk_size;
    const int qk_base = chunk_base * k_head_dim;
    const int out_base = matrix * chunk_size * chunk_size;
    float rows[MAX_CHUNK * MAX_CHUNK];
    float base_row[MAX_CHUNK];
    #pragma unroll
    for (int idx = 0; idx < MAX_CHUNK * MAX_CHUNK; ++idx) {
        rows[idx] = 0.0f;
    }

    for (int i = 1; i < chunk_size; ++i) {
        const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[chunk_base + i]);
        for (int k = 0; k < i; ++k) {
            const int row_base = qk_base + i * k_head_dim;
            const int col_base = qk_base + k * k_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < k_head_dim; ++d) {
                dot += dotcache_qwen35_to_float(k_beta_scan[row_base + d]) *
                    dotcache_qwen35_to_float(key_scan[col_base + d]);
            }
            const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[chunk_base + k]);
            const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
            base_row[k] = -dot * decay;
        }
        for (int j = 0; j < i; ++j) {
            float correction = 0.0f;
            for (int k = 0; k < i; ++k) {
                correction += base_row[k] * rows[k * chunk_size + j];
            }
            rows[i * chunk_size + j] = base_row[j] + correction;
        }
    }

    for (int i = 0; i < chunk_size; ++i) {
        for (int j = 0; j < chunk_size; ++j) {
            float value = rows[i * chunk_size + j];
            if (i == j) {
                value += 1.0f;
            }
            out[out_base + i * chunk_size + j] = dotcache_qwen35_from_float<T>(value);
        }
    }
}

template <typename T>
__global__ void dotcache_qwen35_swiglu_mul_kernel(
    int elem_count,
    const T* gate,
    const T* up,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= elem_count) {
        return;
    }
    const float gate_x = dotcache_qwen35_to_float(gate[idx]);
    const float up_x = dotcache_qwen35_to_float(up[idx]);
    const float silu = gate_x / (1.0f + expf(-gate_x));
    out[idx] = dotcache_qwen35_from_float<T>(silu * up_x);
}

template <typename T, typename IndexT>
__global__ void dotcache_qwen35_embedding_lookup_kernel(
    int token_count,
    int vocab_size,
    int hidden_size,
    const T* embeddings,
    const IndexT* indexes,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_elems = token_count * hidden_size;
    if (idx >= total_elems) {
        return;
    }
    const int token_idx = idx / hidden_size;
    const int col = idx - token_idx * hidden_size;
    const int64_t row = static_cast<int64_t>(indexes[token_idx]);
    if (row < 0 || row >= static_cast<int64_t>(vocab_size)) {
        out[idx] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }
    out[idx] = embeddings[row * hidden_size + col];
}

template <typename T>
__global__ void dotcache_qwen35_output_projection_lookup_kernel(
    int rows,
    int hidden_size,
    int vocab_size,
    const T* hidden,
    const T* weights,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_elems = rows * vocab_size;
    if (idx >= total_elems) {
        return;
    }
    const int row = idx / vocab_size;
    const int vocab = idx - row * vocab_size;
    const T* row_hidden = hidden + static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
    const T* row_weight = weights + static_cast<size_t>(vocab) * static_cast<size_t>(hidden_size);
    float acc = 0.0f;
    for (int col = 0; col < hidden_size; ++col) {
        acc += dotcache_qwen35_to_float(row_hidden[col]) * dotcache_qwen35_to_float(row_weight[col]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_causal_mask_kernel(
    int batch_size,
    int tgt_len,
    int seqlen_offset,
    T* out
) {
    const int kv_len = tgt_len + seqlen_offset;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_elems = batch_size * tgt_len * kv_len;
    if (idx >= total_elems) {
        return;
    }
    const int col = idx % kv_len;
    const int row = (idx / kv_len) % tgt_len;
    const bool allowed = col <= (seqlen_offset + row);
    out[idx] = dotcache_qwen35_from_float<T>(allowed ? 0.0f : -INFINITY);
}

template <typename T>
__global__ void dotcache_qwen35_cumsum_last_dim_kernel(
    int rows,
    int cols,
    const T* xs,
    T* out
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0) {
        return;
    }
    const int row_offset = row * cols;
    float acc = 0.0f;
    for (int col = 0; col < cols; ++col) {
        acc += dotcache_qwen35_to_float(xs[row_offset + col]);
        out[row_offset + col] = dotcache_qwen35_from_float<T>(acc);
    }
}

template <typename T>
__global__ void dotcache_qwen35_exp_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(expf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void dotcache_qwen35_recip_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(1.0f / dotcache_qwen35_to_float(xs[idx]));
}

template <typename T>
__global__ void dotcache_qwen35_sigmoid_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(
        dotcache_qwen35_sigmoid_fast(dotcache_qwen35_to_float(xs[idx]))
    );
}

template <typename T>
__global__ void dotcache_qwen35_log_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(logf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename In, typename Out>
__global__ void dotcache_qwen35_cast_kernel(
    int total_elems,
    const In* xs,
    Out* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<Out>(dotcache_qwen35_to_float(xs[idx]));
}

__device__ __forceinline__ float dotcache_qwen35_unary_op_apply(
    int op,
    float x,
    float scalar
) {
    switch (op) {
    case 0:
        return expf(x);
    case 1:
        return 1.0f / x;
    case 2:
        return dotcache_qwen35_sigmoid_fast(x);
    case 3:
        return logf(x);
    case 4:
        return sqrtf(x);
    case 5:
        return x * scalar;
    case 6:
        return x + scalar;
    default:
        return x;
    }
}

template <typename T>
__global__ void dotcache_qwen35_unary_view_kernel(
    int op,
    int rank,
    size_t total_elems,
    float scalar,
    const T* xs,
    const int* in_strides,
    const int* out_dims,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }
    size_t remaining = idx;
    size_t in_offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
        const int out_dim = out_dims[dim];
        const int coord = static_cast<int>(remaining % static_cast<size_t>(out_dim));
        remaining /= static_cast<size_t>(out_dim);
        in_offset += static_cast<size_t>(coord) * static_cast<size_t>(in_strides[dim]);
    }
    const float x = dotcache_qwen35_to_float(xs[in_offset]);
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_unary_op_apply(op, x, scalar));
}

template <typename In, typename Out>
__global__ void dotcache_qwen35_cast_view_kernel(
    int rank,
    size_t total_elems,
    const In* xs,
    const int* in_strides,
    const int* out_dims,
    Out* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }
    size_t remaining = idx;
    size_t in_offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
        const int out_dim = out_dims[dim];
        const int coord = static_cast<int>(remaining % static_cast<size_t>(out_dim));
        remaining /= static_cast<size_t>(out_dim);
        in_offset += static_cast<size_t>(coord) * static_cast<size_t>(in_strides[dim]);
    }
    out[idx] = dotcache_qwen35_from_float<Out>(dotcache_qwen35_to_float(xs[in_offset]));
}

template <typename T>
__global__ void dotcache_qwen35_binary_broadcast_kernel(
    int op,
    int rank,
    size_t total_elems,
    const T* lhs,
    const T* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }

    size_t remaining = idx;
    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
        const int out_dim = out_dims[dim];
        const int coord = static_cast<int>(remaining % static_cast<size_t>(out_dim));
        remaining /= static_cast<size_t>(out_dim);
        lhs_offset += static_cast<size_t>(coord) * static_cast<size_t>(lhs_strides[dim]);
        rhs_offset += static_cast<size_t>(coord) * static_cast<size_t>(rhs_strides[dim]);
    }

    const float lhs_f = dotcache_qwen35_to_float(lhs[lhs_offset]);
    const float rhs_f = dotcache_qwen35_to_float(rhs[rhs_offset]);
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_binary_op_apply(op, lhs_f, rhs_f));
}

template <typename T>
__global__ void dotcache_qwen35_mul_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_to_float(xs[idx]) * scalar);
}

template <typename T>
__global__ void dotcache_qwen35_add_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_to_float(xs[idx]) + scalar);
}

template <typename T>
__global__ void dotcache_qwen35_sqrt_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(sqrtf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void dotcache_qwen35_reduce_keepdim_kernel(
    int outer,
    int reduce,
    int inner,
    int sum,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = outer * inner;
    if (idx >= total) {
        return;
    }
    const int outer_idx = idx / inner;
    const int inner_idx = idx % inner;
    const int base = (outer_idx * reduce) * inner + inner_idx;
    float acc = dotcache_qwen35_to_float(xs[base]);
    for (int r = 1; r < reduce; ++r) {
        const float value = dotcache_qwen35_to_float(xs[base + r * inner]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_reduce_keepdim_view_kernel(
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const T* xs,
    const int* in_strides,
    const int* out_dims,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_out_elems) {
        return;
    }
    size_t remaining = idx;
    size_t base_offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
        const int out_dim = out_dims[dim];
        const int coord = static_cast<int>(remaining % static_cast<size_t>(out_dim));
        remaining /= static_cast<size_t>(out_dim);
        base_offset += static_cast<size_t>(coord) * static_cast<size_t>(in_strides[dim]);
    }
    float acc = dotcache_qwen35_to_float(xs[base_offset]);
    for (size_t r = 1; r < reduce_len; ++r) {
        const float value = dotcache_qwen35_to_float(
            xs[base_offset + r * static_cast<size_t>(in_strides[reduce_dim])]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_batched_matmul_kernel(
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const T* lhs,
    const T* rhs,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    if (idx >= total) {
        return;
    }

    const size_t matrix_idx = idx % (static_cast<size_t>(m) * static_cast<size_t>(n));
    const size_t batch_idx = idx / (static_cast<size_t>(m) * static_cast<size_t>(n));
    const int row = static_cast<int>(matrix_idx / static_cast<size_t>(n));
    const int col = static_cast<int>(matrix_idx % static_cast<size_t>(n));

    const size_t lhs_batch_idx = dotcache_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, lhs_batch_dims);
    const size_t rhs_batch_idx = dotcache_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, rhs_batch_dims);

    const size_t lhs_base = lhs_batch_idx * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t rhs_base = rhs_batch_idx * static_cast<size_t>(k) * static_cast<size_t>(n);
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += dotcache_qwen35_to_float(lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(kk)]) *
               dotcache_qwen35_to_float(rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_batched_matmul_view_kernel(
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
    const T* lhs,
    const T* rhs,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    if (idx >= total) {
        return;
    }

    const size_t matrix_idx = idx % (static_cast<size_t>(m) * static_cast<size_t>(n));
    const size_t batch_idx = idx / (static_cast<size_t>(m) * static_cast<size_t>(n));
    const int row = static_cast<int>(matrix_idx / static_cast<size_t>(n));
    const int col = static_cast<int>(matrix_idx % static_cast<size_t>(n));

    size_t remaining = batch_idx;
    size_t lhs_base = 0;
    size_t rhs_base = 0;
    for (int dim = batch_rank - 1; dim >= 0; --dim) {
        const int out_dim = out_batch_dims[dim];
        const int coord = static_cast<int>(remaining % static_cast<size_t>(out_dim));
        remaining /= static_cast<size_t>(out_dim);
        lhs_base += static_cast<size_t>(coord) * static_cast<size_t>(lhs_batch_strides[dim]);
        rhs_base += static_cast<size_t>(coord) * static_cast<size_t>(rhs_batch_strides[dim]);
    }

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += dotcache_qwen35_to_float(
                   lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(lhs_row_stride) +
                       static_cast<size_t>(kk) * static_cast<size_t>(lhs_k_stride)]) *
               dotcache_qwen35_to_float(
                   rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(rhs_k_stride) +
                       static_cast<size_t>(col) * static_cast<size_t>(rhs_col_stride)]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_delta_full_scan_pack_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const T* query_scan,
    const T* key_scan,
    const T* exp_g_scan,
    const T* k_cumdecay_scan,
    T* out
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_rows = batch_heads * num_chunks * chunk_size;
    if (row >= total_rows) {
        return;
    }
    const int row_stride = k_head_dim;
    const int packed_width = 3 * k_head_dim + 1;
    const int bh = row / (num_chunks * chunk_size);
    const int rem = row - bh * num_chunks * chunk_size;
    const int chunk = rem / chunk_size;
    const int t = rem - chunk * chunk_size;
    const int scan_row = row * row_stride;
    const int packed_row = row * packed_width;

    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[row]);
    const float exp_g_last = dotcache_qwen35_to_float(exp_g_scan[bh * num_chunks * chunk_size + chunk * chunk_size + (chunk_size - 1)]);
    const float chunk_decay = exp_g_t != 0.0f ? (exp_g_last / exp_g_t) : 0.0f;

    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[packed_row + k_idx] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(key_scan[scan_row + k_idx]) * chunk_decay
        );
        out[packed_row + k_head_dim + k_idx] = k_cumdecay_scan[scan_row + k_idx];
        out[packed_row + 2 * k_head_dim + k_idx] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(query_scan[scan_row + k_idx]) * exp_g_t
        );
    }
    out[packed_row + 3 * k_head_dim] = dotcache_qwen35_from_float<T>(exp_g_last);
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void dotcache_qwen35_delta_full_scan_packed_impl(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* packed_scan,
    const T* local_attn_scan,
    const T* value,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K || chunk_size > MAX_CHUNK) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_count = num_chunks * chunk_size;
    const int packed_width = 3 * k_head_dim + 1;
    const int packed_base = bh * num_chunks * chunk_size * packed_width;
    const int local_base = bh * num_chunks * chunk_size * chunk_size;
    const int value_base = bh * token_count * v_head_dim;
    const int out_base = bh * (token_count + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    float v_new[MAX_CHUNK];
    float attn_inter[MAX_CHUNK];
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int chunk_packed = packed_base + chunk * chunk_size * packed_width;
        const int chunk_local = local_base + chunk * chunk_size * chunk_size;
        const int chunk_value = value_base + chunk * chunk_size * v_head_dim;
        for (int t = 0; t < chunk_size; ++t) {
            float v_prime = 0.0f;
            float attn = 0.0f;
            const int row = chunk_packed + t * packed_width;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += dotcache_qwen35_to_float(packed_scan[row + k_head_dim + k_idx]) * state[k_idx];
                attn += dotcache_qwen35_to_float(packed_scan[row + 2 * k_head_dim + k_idx]) * state[k_idx];
            }
            v_new[t] = dotcache_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) - v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += dotcache_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = dotcache_qwen35_to_float(packed_scan[chunk_packed + 3 * k_head_dim]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_packed + t * packed_width;
                update += dotcache_qwen35_to_float(packed_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_full_scan_packed_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* packed_scan,
    const T* local_attn_scan,
    const T* value,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_full_scan_packed_impl(
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        initial_state,
        packed_scan,
        local_attn_scan,
        value,
        out,
        tid
    );
}

template <typename T>
__global__ void dotcache_qwen35_delta_full_scan_kernel(
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* initial_state,
    const T* weighted_key_scan,
    const T* k_cumdecay_scan,
    const T* q_state_scan,
    const T* local_attn_scan,
    const T* state_decay_scan,
    const T* value,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_full_scan_impl(
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        initial_state,
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
        out,
        tid
    );
}

template <typename T, int MAX_CHUNK = 64, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_chunk_single_prefill_impl(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g_raw,
    T* out,
    int tid
) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    const int bh = tid / v_head_dim;
    const int v_idx = tid - bh * v_head_dim;
    const int token_stride_k = chunk_size * k_head_dim;
    const int token_stride_v = chunk_size * v_head_dim;
    const int token_stride_s = chunk_size;
    const int out_base = bh * (chunk_size + k_head_dim) * v_head_dim;

    float prefix_g[MAX_CHUNK];
    float raw_g[MAX_CHUNK];

    float g_acc = 0.0f;
    for (int t = 0; t < chunk_size; ++t) {
        raw_g[t] = dotcache_qwen35_to_float(g_raw[bh * token_stride_s + t]);
        g_acc += raw_g[t];
        prefix_g[t] = g_acc;
    }

    for (int i = 0; i < chunk_size; ++i) {
        const int row_i_k = bh * token_stride_k + i * k_head_dim;
        const int row_i_v = bh * token_stride_v + i * v_head_dim;
        float out_i = 0.0f;
        for (int j = 0; j <= i; ++j) {
            const int row_j_k = bh * token_stride_k + j * k_head_dim;
            float dot = 0.0f;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                dot += dotcache_qwen35_to_float(query[row_i_k + k_idx]) *
                    dotcache_qwen35_to_float(key[row_j_k + k_idx]);
            }
            const float local = dot * expf(prefix_g[i] - prefix_g[j]);
            out_i += local * dotcache_qwen35_to_float(value[bh * token_stride_v + j * v_head_dim + v_idx]);
        }
        out[out_base + i * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_i);
    }

    const float g_last = raw_g[chunk_size - 1];
    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float state = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const float beta_t = dotcache_qwen35_to_float(beta[bh * token_stride_s + t]);
            state += dotcache_qwen35_to_float(key[bh * token_stride_k + t * k_head_dim + k_idx]) *
                expf(g_last - raw_g[t]) *
                dotcache_qwen35_to_float(value[bh * token_stride_v + t * v_head_dim + v_idx]);
        }
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_single_prefill_kernel(
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const T* query,
    const T* key,
    const T* value,
    const T* beta,
    const T* g_raw,
    T* out
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_chunk_single_prefill_impl(
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        query,
        key,
        value,
        beta,
        g_raw,
        out,
        tid
    );
}

template <typename T>
__global__ void dotcache_qwen35_l2norm_kernel(
    int n_rows,
    int n_cols,
    float eps,
    const T* xs,
    T* out
) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const T* row_in = xs + static_cast<size_t>(row) * static_cast<size_t>(n_cols);
    T* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(n_cols);

    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = dotcache_qwen35_to_float(row_in[col]);
        partial += x * x;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float shared_inv_norm;
    if (tid == 0) {
        shared_inv_norm = rsqrtf(shared_sum[0] + eps);
    }
    __syncthreads();

    for (int col = tid; col < n_cols; col += blockDim.x) {
        row_out[col] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(row_in[col]) * shared_inv_norm
        );
    }
}

template <typename T>
__global__ void dotcache_qwen35_value_decay_kernel(
    int total_elems,
    int num_heads,
    const T* a,
    const T* dt_bias,
    const T* a_log_exp,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }

    const int head = idx % num_heads;
    const float a_val = dotcache_qwen35_to_float(a[idx]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = dotcache_qwen35_softplus_fast(a_val + bias);
    out[idx] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, bool ADD_UNIT_OFFSET>
__global__ void dotcache_qwen35_rms_norm_kernel(
    int n_rows,
    int n_cols,
    float eps,
    const T* xs,
    const T* weight,
    T* out
) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const T* row_in = xs + static_cast<size_t>(row) * static_cast<size_t>(n_cols);
    T* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(n_cols);

    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = dotcache_qwen35_to_float(row_in[col]);
        partial += x * x;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float shared_inv_rms;
    if (tid == 0) {
        const float mean = shared_sum[0] / static_cast<float>(n_cols);
        shared_inv_rms = rsqrtf(mean + eps);
    }
    __syncthreads();

    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float weight_val = dotcache_qwen35_to_float(weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        const float x = dotcache_qwen35_to_float(row_in[col]);
        row_out[col] = dotcache_qwen35_from_float<T>(x * shared_inv_rms * weight_val);
    }
}

// Fused RMSNorm + Matrix-Vector Product kernel for single-token decode.
//
// Computes: out[i] = dot(W[i, :], rms_norm(hidden)) for each output row i.
//
// Phase 1: All threads in the block cooperatively compute the RMSNorm of the
//          hidden vector and store the F32 normalized result in shared memory.
// Phase 2: Each block computes one dot product of W[blockIdx.x, :] with the
//          shared normalized vector, producing one output element.
//
// This keeps the normalized hidden in F32 throughout (no BF16 round-trip),
// which eliminates the BF16 quantization noise between RMSNorm and projection
// that accumulates across decoder layers.
//
// Grid:  (out_dim, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)  where BLOCK_SIZE <= 256
//
// Requires shared memory: hidden_dim * sizeof(float) + BLOCK_SIZE * sizeof(float)
template <typename T, bool ADD_UNIT_OFFSET>
__global__ void dotcache_qwen35_fused_rms_norm_linear_kernel(
    int hidden_dim,
    int out_dim,
    float eps,
    const T* __restrict__ hidden,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out
) {
    const int out_row = blockIdx.x;
    if (out_row >= out_dim) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory layout:
    //   [0 .. hidden_dim-1]:          F32 normalized hidden vector
    //   [hidden_dim .. hidden_dim+block_size-1]: reduction scratch
    extern __shared__ float shared_mem[];
    float* shared_normed = shared_mem;
    float* shared_scratch = shared_mem + hidden_dim;

    // Phase 1: Compute RMSNorm — only block 0 needs to write shared_normed,
    // but ALL blocks need the same result. Since each block runs independently,
    // every block recomputes the norm (hidden_dim=1024 is cheap).

    // 1a: Accumulate sum of squares
    float partial_sq = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        const float x = dotcache_qwen35_to_float(hidden[col]);
        partial_sq += x * x;
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();

    // 1b: Tree reduction for sum of squares
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_scratch[tid] += shared_scratch[tid + stride];
        }
        __syncthreads();
    }

    // 1c: Compute inv_rms and write normalized vector to shared memory
    __shared__ float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    const float inv_rms = shared_inv_rms;
    for (int col = tid; col < hidden_dim; col += block_size) {
        const float x = dotcache_qwen35_to_float(hidden[col]);
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        shared_normed[col] = x * inv_rms * w;
    }
    __syncthreads();

    // Phase 2: Dot product of proj_weight[out_row, :] with shared_normed
    const T* w_row = proj_weight + static_cast<size_t>(out_row) * static_cast<size_t>(hidden_dim);

    float partial_dot = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        partial_dot += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
    }
    shared_scratch[tid] = partial_dot;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_scratch[tid] += shared_scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[out_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_rms_norm_gated_kernel(
    int n_rows,
    int n_cols,
    float eps,
    const T* hidden,
    const T* gate,
    const T* weight,
    T* out
) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const T* row_hidden = hidden + static_cast<size_t>(row) * static_cast<size_t>(n_cols);
    const T* row_gate = gate + static_cast<size_t>(row) * static_cast<size_t>(n_cols);
    T* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(n_cols);

    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = dotcache_qwen35_to_float(row_hidden[col]);
        partial += x * x;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float shared_inv_rms;
    if (tid == 0) {
        const float mean = shared_sum[0] / static_cast<float>(n_cols);
        shared_inv_rms = rsqrtf(mean + eps);
    }
    __syncthreads();

    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = dotcache_qwen35_to_float(row_hidden[col]);
        const float gate_x = dotcache_qwen35_to_float(row_gate[col]);
    const float gate_silu = gate_x * dotcache_qwen35_sigmoid_fast(gate_x);
        const float weight_val = dotcache_qwen35_to_float(weight[col]);
        row_out[col] =
            dotcache_qwen35_from_float<T>(x * shared_inv_rms * weight_val * gate_silu);
    }
}

// =============================================================================
// Persistent MLP decode megakernel (v2 — block-level reduction)
//
// Fuses: RMSNorm + gate_proj + up_proj + SwiGLU + down_proj for one layer.
// All CUs cooperate via atomic work-stealing for the matvec rows.
//
// v2 change: uses full-block (256-thread) reduction for dot products instead
// of wave-level (32-thread). This matches the accuracy of the block-wide
// RMSNorm reduction and avoids the accumulation errors from 32-wide reduction
// of 1024-element dot products.
//
// Grid:  (num_blocks, 1, 1) — typically 16 (one per CU)
// Block: (block_size, 1, 1) — typically 256 (8 waves × 32 lanes)
//
// LDS layout:
//   shared_hidden[hidden_dim]  : F32 — input hidden state
//   shared_normed[hidden_dim]  : F32 — RMSNorm output
//   shared_scratch[block_size] : F32 — reduction scratch
//
// Global scratch: gate_up_buf[intermediate_size * 2] F32 — holds gate and up projections
// =============================================================================

// Device-side wave-level sum reduction (Wave32 on RDNA3.5)
__device__ inline float dotcache_qwen35_wave_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__global__ void dotcache_qwen35_mlp_decode_megakernel(
    int hidden_dim,           // 1024
    int intermediate_size,    // 3584
    float norm_eps,
    const T* __restrict__ hidden_in,      // [hidden_dim] input hidden state
    const T* __restrict__ norm_weight,    // [hidden_dim] RMSNorm weight
    const T* __restrict__ gate_proj_w,    // [intermediate_size, hidden_dim]
    const T* __restrict__ up_proj_w,      // [intermediate_size, hidden_dim]
    const T* __restrict__ down_proj_w,    // [hidden_dim, intermediate_size]
    float* __restrict__ gate_up_scratch,  // [intermediate_size * 2] global scratch
    T* __restrict__ hidden_out,           // [hidden_dim] output
    unsigned int* __restrict__ row_counter // atomic work counter (reset to 0 before launch)
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int wave_size = warpSize;  // 32 on RDNA3.5
    const int lane = tid % wave_size;
    const int wave_id = tid / wave_size;
    const int waves_per_block = block_size / wave_size;

    // --- LDS allocation ---
    extern __shared__ float lds[];
    float* shared_hidden = lds;                          // [hidden_dim]
    float* shared_normed = lds + hidden_dim;             // [hidden_dim]
    float* shared_scratch = lds + hidden_dim * 2;        // [block_size]

    // =========================================================================
    // Step 1: Load hidden state into LDS and compute RMSNorm
    // =========================================================================

    // Load hidden from global → LDS as F32
    for (int col = tid; col < hidden_dim; col += block_size) {
        shared_hidden[col] = dotcache_qwen35_to_float(hidden_in[col]);
    }
    __syncthreads();

    // Compute sum of squares (parallel reduction)
    float partial_sq = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        partial_sq += shared_hidden[col] * shared_hidden[col];
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_scratch[tid] += shared_scratch[tid + stride];
        }
        __syncthreads();
    }

    // Compute inv_rms and write normed to LDS
    __shared__ float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(hidden_dim) + norm_eps);
    }
    __syncthreads();

    const float inv_rms = shared_inv_rms;
    for (int col = tid; col < hidden_dim; col += block_size) {
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + 1.0f;  // add_unit_offset
        shared_normed[col] = shared_hidden[col] * inv_rms * w;
    }
    __syncthreads();

    // =========================================================================
    // Step 2: gate_proj and up_proj matvecs — block-level work-stealing
    //
    // Each BLOCK steals one output row at a time. All 256 threads cooperate
    // on the dot product (256-wide reduction), matching RMSNorm accuracy.
    // gate results → gate_up_scratch[0..intermediate_size)
    // up results   → gate_up_scratch[intermediate_size..intermediate_size*2)
    // =========================================================================

    const int total_gate_up_rows = intermediate_size * 2;
    for (;;) {
        __shared__ unsigned int shared_row;
        if (tid == 0) {
            shared_row = atomicAdd(row_counter, 1u);
        }
        __syncthreads();
        const unsigned int my_row = shared_row;
        if (my_row >= static_cast<unsigned int>(total_gate_up_rows)) break;

        const bool is_up = my_row >= static_cast<unsigned int>(intermediate_size);
        const int proj_row = is_up ? (my_row - intermediate_size) : my_row;
        const T* w_matrix = is_up ? up_proj_w : gate_proj_w;
        const T* w_row = w_matrix + static_cast<size_t>(proj_row) * hidden_dim;

        float partial = 0.0f;
        for (int col = tid; col < hidden_dim; col += block_size) {
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
        }
        shared_scratch[tid] = partial;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_scratch[tid] += shared_scratch[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            gate_up_scratch[my_row] = shared_scratch[0];
        }
        __syncthreads();
    }

    // NOTE: SwiGLU and down_proj are launched as separate kernels from the
    // bridge to avoid inter-block synchronization issues. This kernel only
    // handles RMSNorm + gate/up projections.
}

// SwiGLU activation kernel: silu(gate) * up, element-wise
template <typename T>
__global__ void dotcache_qwen35_mlp_swiglu_kernel(
    int intermediate_size,
    float* __restrict__ gate_up_scratch    // [intermediate_size * 2]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate_size) return;
    const float gate_val = gate_up_scratch[idx];
    const float up_val = gate_up_scratch[intermediate_size + idx];
    const float silu = gate_val / (1.0f + expf(-gate_val));
    gate_up_scratch[idx] = silu * up_val;
}

// down_proj matvec kernel: [hidden_dim, intermediate_size] × SwiGLU[intermediate_size]
// Block-level work-stealing with full-block reduction.
template <typename T>
__global__ void dotcache_qwen35_mlp_down_proj_kernel(
    int hidden_dim,
    int intermediate_size,
    const T* __restrict__ down_proj_w,
    const float* __restrict__ swiglu_scratch,
    T* __restrict__ hidden_out,
    unsigned int* __restrict__ row_counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    extern __shared__ float shared_scratch[];

    for (;;) {
        __shared__ unsigned int shared_row;
        if (tid == 0) {
            shared_row = atomicAdd(row_counter, 1u);
        }
        __syncthreads();
        const unsigned int my_row = shared_row;
        if (my_row >= static_cast<unsigned int>(hidden_dim)) break;

        const T* w_row = down_proj_w + static_cast<size_t>(my_row) * intermediate_size;

        float partial = 0.0f;
        for (int col = tid; col < intermediate_size; col += block_size) {
            partial += dotcache_qwen35_to_float(w_row[col]) * swiglu_scratch[col];
        }
        shared_scratch[tid] = partial;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_scratch[tid] += shared_scratch[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            hidden_out[my_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
        }
        __syncthreads();
    }
}

// =============================================================================
// Standalone work-stealing matvec kernel for decode
// Computes: out[out_dim] = W[out_dim, in_dim] × input[in_dim]
// Input is typed T (BF16), weight is typed T (BF16), output is typed T (BF16)
// All accumulation in F32 with block-level reduction.
// =============================================================================
template <typename T>
__global__ void dotcache_qwen35_standalone_matvec_kernel(
    int out_dim,
    int in_dim,
    const T* __restrict__ weight,      // [out_dim, in_dim]
    const T* __restrict__ input,       // [in_dim]
    T* __restrict__ output,            // [out_dim]
    unsigned int* __restrict__ row_counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    extern __shared__ float shared_scratch[];

    // Load input into shared memory as F32
    __shared__ float shared_input[4096];  // max in_dim we support
    for (int col = tid; col < in_dim; col += block_size) {
        shared_input[col] = dotcache_qwen35_to_float(input[col]);
    }
    __syncthreads();

    for (;;) {
        __shared__ unsigned int shared_row;
        if (tid == 0) {
            shared_row = atomicAdd(row_counter, 1u);
        }
        __syncthreads();
        const unsigned int my_row = shared_row;
        if (my_row >= static_cast<unsigned int>(out_dim)) break;

        const T* w_row = weight + static_cast<size_t>(my_row) * in_dim;

        float partial = 0.0f;
        for (int col = tid; col < in_dim; col += block_size) {
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_input[col];
        }
        shared_scratch[tid] = partial;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_scratch[tid] += shared_scratch[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[my_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
        }
        __syncthreads();
    }
}

// =============================================================================
// General-purpose RMSNorm + multi-projection kernel for decode
//
// Computes RMSNorm on hidden[hidden_dim], then performs N separate matvec
// projections from the normed vector, writing results to a packed output buffer.
//
// Projections are described by a table of (weight_ptr, out_dim) pairs.
// Output is packed: proj0[out_dim0] || proj1[out_dim1] || ...
//
// This generalizes the MLP gate/up pattern to arbitrary projection counts
// (e.g., Q/K/V for attention, or gate/up for MLP).
// =============================================================================

struct Qwen35ProjectionDesc {
    const void* weight;    // [out_dim, hidden_dim] BF16
    int out_dim;           // number of output rows
    int output_offset;     // offset in the output buffer
};

template <typename T>
__global__ void dotcache_qwen35_norm_multi_proj_kernel(
    int hidden_dim,
    int total_rows,                           // sum of all out_dims
    float norm_eps,
    const T* __restrict__ hidden_in,          // [hidden_dim]
    const T* __restrict__ norm_weight,        // [hidden_dim]
    const Qwen35ProjectionDesc* __restrict__ proj_table,  // device ptr
    int num_projections,
    float* __restrict__ output,               // [total_rows] F32
    unsigned int* __restrict__ row_counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float lds[];
    float* shared_hidden = lds;
    float* shared_normed = lds + hidden_dim;
    float* shared_scratch = lds + hidden_dim * 2;

    // Step 1: Load hidden + RMSNorm
    for (int col = tid; col < hidden_dim; col += block_size) {
        shared_hidden[col] = dotcache_qwen35_to_float(hidden_in[col]);
    }
    __syncthreads();

    float partial_sq = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        partial_sq += shared_hidden[col] * shared_hidden[col];
    }
    shared_scratch[tid] = partial_sq;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_scratch[tid] += shared_scratch[tid + stride];
        __syncthreads();
    }
    __shared__ float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = rsqrtf(shared_scratch[0] / static_cast<float>(hidden_dim) + norm_eps);
    }
    __syncthreads();

    for (int col = tid; col < hidden_dim; col += block_size) {
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + 1.0f;
        shared_normed[col] = shared_hidden[col] * shared_inv_rms * w;
    }
    __syncthreads();

    // Step 2: Work-stealing matvec across all projection rows
    for (;;) {
        __shared__ unsigned int shared_row;
        if (tid == 0) {
            shared_row = atomicAdd(row_counter, 1u);
        }
        __syncthreads();
        const unsigned int my_row = shared_row;
        if (my_row >= static_cast<unsigned int>(total_rows)) break;

        // Find which projection this row belongs to
        int proj_idx = 0;
        int row_in_proj = static_cast<int>(my_row);
        for (int p = 0; p < num_projections; ++p) {
            if (row_in_proj < proj_table[p].out_dim) {
                proj_idx = p;
                break;
            }
            row_in_proj -= proj_table[p].out_dim;
        }

        const T* w_row = static_cast<const T*>(proj_table[proj_idx].weight)
            + static_cast<size_t>(row_in_proj) * hidden_dim;

        float partial = 0.0f;
        for (int col = tid; col < hidden_dim; col += block_size) {
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
        }
        shared_scratch[tid] = partial;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_scratch[tid] += shared_scratch[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            output[proj_table[proj_idx].output_offset + row_in_proj] = shared_scratch[0];
        }
        __syncthreads();
    }
}

// =============================================================================
// MONOLITHIC PERSISTENT DECODE MEGAKERNEL
//
// Processes all 24 decoder layers in a single kernel launch.
// Hidden state lives in global workspace (F32). All matvecs use multi-block
// work-stealing. Sequential ops done by block 0, others spin on barrier.
//
// Grid:  (num_cus, 1, 1)
// Block: (256, 1, 1)
// =============================================================================

__device__ inline void grid_barrier(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* counter_v = barrier_counter;
        volatile unsigned int* flag_v = barrier_flag;
        unsigned int phase = *flag_v;
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            *counter_v = 0u;
            __threadfence();
            *flag_v = phase + 1;
        } else {
            while (*flag_v == phase) {}
            __threadfence();
        }
    }
    __syncthreads();
}

template <typename T>
__device__ inline void block_rms_norm_global(
    float* dst, const float* src, const T* weight,
    int dim, float eps, float* scratch
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    float sq = 0.0f;
    for (int c = tid; c < dim; c += bs) sq += src[c] * src[c];
    scratch[tid] = sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
    for (int c = tid; c < dim; c += bs) {
        dst[c] = src[c] * inv * (dotcache_qwen35_to_float(weight[c]) + 1.0f);
    }
    __syncthreads();
}

template <typename T, bool HERO_MODE>
__global__ void dotcache_qwen35_persistent_decode_kernel(
    int num_layers,
    int hidden_dim,
    int intermediate_size,
    int seqlen_offset,
    const Qwen35DecodeLayerDesc* __restrict__ layers,
    T* __restrict__ hidden_io,
    float* __restrict__ workspace,
    unsigned int* __restrict__ counters,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag,
    const T* __restrict__ cos_table,   // [max_positions, rotary_dim/2] RoPE cos
    const T* __restrict__ sin_table,   // [max_positions, rotary_dim/2] RoPE sin
    int rotary_dim                      // partial rotary dimension (64 for Qwen3.5)
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    // Workspace layout (F32 unless noted):
    // Section 1: persistent state
    float* hidden_f32 = workspace;                              // [hidden_dim]
    float* normed     = hidden_f32 + hidden_dim;                // [hidden_dim]
    // Section 2: MLP scratch
    float* gate_up    = normed + hidden_dim;                    // [intermediate_size * 2]
    float* mlp_out    = gate_up + intermediate_size * 2;        // [hidden_dim]
    // Section 3: token mixer scratch
    float* token_out  = mlp_out + hidden_dim;                   // [hidden_dim]
    float* proj_buf   = token_out + hidden_dim;                 // [8224] max proj output
    float* attn_scratch = proj_buf + 8224;                      // [2048] attention/recurrent output
    // proj_buf holds projection results as F32 for all layer types
    // attn_scratch holds intermediate attention or recurrent output

    extern __shared__ float lds[];
    // LDS layout: lds[0..bs-1] = reduction scratch, lds[bs..] = input vector cache
    float* lds_input = lds + bs;  // LDS-resident copy of input vectors for matvecs

    // Load hidden BF16 → F32
    for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
        hidden_f32[c] = dotcache_qwen35_to_float(hidden_io[c]);
    }
    grid_barrier(barrier_counter, barrier_flag, nb);

    const bool qwen08_hero =
        HERO_MODE &&
        num_layers == 24 &&
        hidden_dim == 1024 &&
        intermediate_size == 3584 &&
        bs == 256;

    for (int layer = 0; layer < num_layers; ++layer) {
        const Qwen35DecodeLayerDesc& L = layers[layer];

        // === Input RMSNorm (block 0 only) ===
        if (blockIdx.x == 0) {
            block_rms_norm_global<T>(normed, hidden_f32,
                static_cast<const T*>(L.input_norm_w),
                hidden_dim, L.input_norm_eps, lds);
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // All blocks: cache normed vector in LDS for fast projection reads
        for (int c = tid; c < hidden_dim; c += bs)
            lds_input[c] = normed[c];
        __syncthreads();

        // === Token mixer: projections + core ===
        if (L.layer_type == 1) {
            // ---- FULL ATTENTION ----
            // Step A: Q/K/V projections via work-stealing
            // Q: [q_out_dim, hidden_dim], K: [k_out_dim, hidden_dim], V: same as K
            const int total_proj = L.q_out_dim + L.k_out_dim + L.k_out_dim;
            {
                if (qwen08_hero) {
                    for (int sr = blockIdx.x; sr < total_proj; sr += nb) {
                        const T* w;
                        int row;
                        if (sr < L.q_out_dim) {
                            w = static_cast<const T*>(L.q_proj_w);
                            row = sr;
                        } else if (sr < L.q_out_dim + L.k_out_dim) {
                            w = static_cast<const T*>(L.k_proj_w);
                            row = sr - L.q_out_dim;
                        } else {
                            w = static_cast<const T*>(L.v_proj_w);
                            row = sr - L.q_out_dim - L.k_out_dim;
                        }
                        const T* wr = w + static_cast<size_t>(row) * hidden_dim;

                        float p = 0.0f;
                        for (int c = tid; c < hidden_dim; c += bs)
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        const float sum = dotcache_qwen35_block_sum_256(p, lds);
                        if (tid == 0) proj_buf[sr] = sum;
                        __syncthreads();
                    }
                } else {
                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);
                    for (;;) {
                        __shared__ unsigned int sr;
                        if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                        __syncthreads();
                        if (sr >= static_cast<unsigned int>(total_proj)) break;

                        const T* w;
                        int row;
                        if (sr < static_cast<unsigned int>(L.q_out_dim)) {
                            w = static_cast<const T*>(L.q_proj_w);
                            row = sr;
                        } else if (sr < static_cast<unsigned int>(L.q_out_dim + L.k_out_dim)) {
                            w = static_cast<const T*>(L.k_proj_w);
                            row = sr - L.q_out_dim;
                        } else {
                            w = static_cast<const T*>(L.v_proj_w);
                            row = sr - L.q_out_dim - L.k_out_dim;
                        }
                        const T* wr = w + static_cast<size_t>(row) * hidden_dim;

                        float p = 0.0f;
                        for (int c = tid; c < hidden_dim; c += bs)
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        lds[tid] = p;
                        __syncthreads();
                        for (int s = bs/2; s > 0; s >>= 1) {
                            if (tid < s) lds[tid] += lds[tid+s];
                            __syncthreads();
                        }
                        if (tid == 0) proj_buf[sr] = lds[0];
                        __syncthreads();
                    }
                }
            }
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step B-G: QK-norm, RoPE, KV cache, attention, gating, o_proj
            // Block 0 handles the per-head sequential ops.
            // O_proj uses all blocks via work-stealing.
            const bool qwen08_attn_hero = qwen08_hero && nb >= 8;

            if (qwen08_attn_hero) {
                float* q_f32 = proj_buf;
                float* k_f32 = proj_buf + L.q_out_dim;
                float* v_f32 = proj_buf + L.q_out_dim + L.k_out_dim;

                const int hd = L.attn_head_dim;       // 256
                const int nh = L.attn_num_heads;      // 8
                const int nkv = L.attn_num_kv_heads;  // 2
                const int kv_groups = nh / nkv;       // 4
                const float scale = 1.0f / sqrtf(static_cast<float>(hd));
                const int rotary_dim = hd / 4;        // 64
                const int kv_len = L.kv_len + 1;
                float* saved_gate = attn_scratch;

                if (blockIdx.x < nh) {
                    const int qh = blockIdx.x;
                    float* q_head = q_f32 + qh * hd * 2;

                    float sq = 0.0f;
                    for (int d = tid; d < hd; d += bs) sq += q_head[d] * q_head[d];
                    float inv = rsqrtf(
                        dotcache_qwen35_block_sum_256(sq, lds) / static_cast<float>(hd) + 1e-6f);
                    const T* qnw = static_cast<const T*>(L.q_norm_w);
                    for (int d = tid; d < hd; d += bs) {
                        q_head[d] = q_head[d] * inv * (dotcache_qwen35_to_float(qnw[d]) + 1.0f);
                    }
                    __syncthreads();
                }

                if (blockIdx.x < nkv) {
                    const int kh_idx = blockIdx.x;
                    float* k_head = k_f32 + kh_idx * hd;
                    float sq = 0.0f;
                    for (int d = tid; d < hd; d += bs) sq += k_head[d] * k_head[d];
                    float inv = rsqrtf(
                        dotcache_qwen35_block_sum_256(sq, lds) / static_cast<float>(hd) + 1e-6f);
                    const T* knw = static_cast<const T*>(L.k_norm_w);
                    for (int d = tid; d < hd; d += bs) {
                        k_head[d] = k_head[d] * inv * (dotcache_qwen35_to_float(knw[d]) + 1.0f);
                    }
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (cos_table != nullptr && rotary_dim > 0) {
                    const int half_rot = rotary_dim / 2;
                    const size_t cos_off = static_cast<size_t>(seqlen_offset) * half_rot;

                    if (blockIdx.x < nh) {
                        const int qh = blockIdx.x;
                        float* q_head = q_f32 + qh * hd * 2;
                        if (tid < half_rot) {
                            float c = dotcache_qwen35_to_float(cos_table[cos_off + tid]);
                            float s = dotcache_qwen35_to_float(sin_table[cos_off + tid]);
                            float x0 = q_head[tid];
                            float x1 = q_head[half_rot + tid];
                            q_head[tid] = x0 * c - x1 * s;
                            q_head[half_rot + tid] = x0 * s + x1 * c;
                        }
                        __syncthreads();
                    }

                    if (blockIdx.x < nkv) {
                        const int kh_idx = blockIdx.x;
                        float* k_head = k_f32 + kh_idx * hd;
                        if (tid < half_rot) {
                            float c = dotcache_qwen35_to_float(cos_table[cos_off + tid]);
                            float s = dotcache_qwen35_to_float(sin_table[cos_off + tid]);
                            float x0 = k_head[tid];
                            float x1 = k_head[half_rot + tid];
                            k_head[tid] = x0 * c - x1 * s;
                            k_head[half_rot + tid] = x0 * s + x1 * c;
                        }
                        __syncthreads();
                    }
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (blockIdx.x < nkv) {
                    const int kh_idx = blockIdx.x;
                    T* cache_k = static_cast<T*>(L.kv_cache_k);
                    T* cache_v = static_cast<T*>(L.kv_cache_v);
                    for (int d = tid; d < hd; d += bs) {
                        const size_t offset =
                            static_cast<size_t>(kh_idx) * L.kv_max_t * hd +
                            static_cast<size_t>(seqlen_offset) * hd + d;
                        cache_k[offset] = dotcache_qwen35_from_float<T>(k_f32[kh_idx * hd + d]);
                        cache_v[offset] = dotcache_qwen35_from_float<T>(v_f32[kh_idx * hd + d]);
                    }
                    __syncthreads();
                }

                if (blockIdx.x < nh) {
                    const int qh = blockIdx.x;
                    for (int d = tid; d < hd; d += bs) {
                        saved_gate[qh * hd + d] = q_f32[qh * hd * 2 + hd + d];
                    }
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (blockIdx.x < nh) {
                    const int qh = blockIdx.x;
                    const int kvh = qh / kv_groups;
                    const float* q_head = q_f32 + qh * hd * 2;
                    const T* ck = static_cast<const T*>(L.kv_cache_k);
                    const T* cv = static_cast<const T*>(L.kv_cache_v);
                    const size_t kv_head_base = static_cast<size_t>(kvh) * L.kv_max_t * hd;

                    float my_acc = 0.0f;
                    float my_max = -1e30f;
                    float my_sum = 0.0f;
                    const float q_val = q_head[tid];

                    for (int t = 0; t < kv_len; ++t) {
                        const size_t pos_base = kv_head_base + static_cast<size_t>(t) * hd;
                        float partial = q_val * dotcache_qwen35_to_float(ck[pos_base + tid]);
                        const float score = dotcache_qwen35_block_sum_256(partial, lds) * scale;
                        const float v_val = dotcache_qwen35_to_float(cv[pos_base + tid]);
                        const float old_max = my_max;
                        my_max = fmaxf(my_max, score);
                        const float rescale = expf(old_max - my_max);
                        const float w = expf(score - my_max);
                        my_acc = my_acc * rescale + w * v_val;
                        my_sum = my_sum * rescale + w;
                    }

                    proj_buf[qh * hd + tid] =
                        (my_sum > 0.0f) ? (my_acc / my_sum) : 0.0f;
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (blockIdx.x < nh) {
                    const int qh = blockIdx.x;
                    for (int d = tid; d < hd; d += bs) {
                        float gate_val = saved_gate[qh * hd + d];
                        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
                        proj_buf[qh * hd + d] *= sigmoid_gate;
                    }
                    __syncthreads();
                }
            } else if (blockIdx.x == 0) {
                float* q_f32 = proj_buf;
                float* k_f32 = proj_buf + L.q_out_dim;
                float* v_f32 = proj_buf + L.q_out_dim + L.k_out_dim;

                const int hd = L.attn_head_dim;       // 256
                const int nh = L.attn_num_heads;       // 8
                const int nkv = L.attn_num_kv_heads;   // 2
                const int kv_groups = nh / nkv;         // 4
                const float scale = 1.0f / sqrtf(static_cast<float>(hd));
                const int rotary_dim = hd / 4;          // 64 (partial_rotary_factor=0.25)
                const int kv_len = L.kv_len + 1;       // includes this new token
                // q_f32 layout: [nh * hd * 2] = per head: [query(hd) | gate(hd)]

                // Step B: QK-norm — RMSNorm per head on head_dim
                for (int h = 0; h < nh; ++h) {
                    float* qh = q_f32 + h * hd * 2;  // query portion
                    float sq = 0.0f;
                    for (int d = tid; d < hd; d += bs) sq += qh[d] * qh[d];
                    lds[tid] = sq;
                    __syncthreads();
                    for (int s = bs/2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid+s];
                        __syncthreads();
                    }
                    float inv = rsqrtf(lds[0] / static_cast<float>(hd) + 1e-6f);
                    const T* qnw = static_cast<const T*>(L.q_norm_w);
                    for (int d = tid; d < hd; d += bs) {
                        qh[d] = qh[d] * inv * (dotcache_qwen35_to_float(qnw[d]) + 1.0f);
                    }
                    __syncthreads();
                }
                for (int h = 0; h < nkv; ++h) {
                    float* kh = k_f32 + h * hd;
                    float sq = 0.0f;
                    for (int d = tid; d < hd; d += bs) sq += kh[d] * kh[d];
                    lds[tid] = sq;
                    __syncthreads();
                    for (int s = bs/2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid+s];
                        __syncthreads();
                    }
                    float inv = rsqrtf(lds[0] / static_cast<float>(hd) + 1e-6f);
                    const T* knw = static_cast<const T*>(L.k_norm_w);
                    for (int d = tid; d < hd; d += bs) {
                        kh[d] = kh[d] * inv * (dotcache_qwen35_to_float(knw[d]) + 1.0f);
                    }
                    __syncthreads();
                }

                // Step C: RoPE (partial, first rotary_dim dims of head_dim)
                // Qwen3.5 uses rotate_half over the rotary slice.
                if (cos_table != nullptr && rotary_dim > 0) {
                    const int half_rot = rotary_dim / 2;
                    const size_t cos_off =
                        static_cast<size_t>(seqlen_offset) * half_rot;

                    // Apply to all Q heads in parallel (nh * half_rot <= 256)
                    if (tid < nh * half_rot) {
                        const int h = tid / half_rot;
                        const int i = tid % half_rot;
                        float* qh = q_f32 + h * hd * 2;
                        float c = dotcache_qwen35_to_float(cos_table[cos_off + i]);
                        float s = dotcache_qwen35_to_float(sin_table[cos_off + i]);
                        float x0 = qh[i];
                        float x1 = qh[half_rot + i];
                        qh[i]            = x0 * c - x1 * s;
                        qh[half_rot + i] = x0 * s + x1 * c;
                    }
                    __syncthreads();

                    // Apply to all K heads (nkv * half_rot threads)
                    if (tid < nkv * half_rot) {
                        const int h = tid / half_rot;
                        const int i = tid % half_rot;
                        float* kh = k_f32 + h * hd;
                        float c = dotcache_qwen35_to_float(cos_table[cos_off + i]);
                        float s = dotcache_qwen35_to_float(sin_table[cos_off + i]);
                        float x0 = kh[i];
                        float x1 = kh[half_rot + i];
                        kh[i]            = x0 * c - x1 * s;
                        kh[half_rot + i] = x0 * s + x1 * c;
                    }
                    __syncthreads();
                }

                // Step D: KV cache append — write new K/V at seqlen_offset
                {
                    T* cache_k = static_cast<T*>(L.kv_cache_k);
                    T* cache_v = static_cast<T*>(L.kv_cache_v);
                    // Cache layout: [num_kv_heads, max_T, head_dim] BF16
                    // New K/V at position seqlen_offset
                    for (int h = 0; h < nkv; ++h) {
                        for (int d = tid; d < hd; d += bs) {
                            const size_t offset =
                                static_cast<size_t>(h) * L.kv_max_t * hd +
                                static_cast<size_t>(seqlen_offset) * hd + d;
                            cache_k[offset] = dotcache_qwen35_from_float<T>(k_f32[h * hd + d]);
                            cache_v[offset] = dotcache_qwen35_from_float<T>(v_f32[h * hd + d]);
                        }
                    }
                    __syncthreads();
                }

                // Save gate values before attention overwrites proj_buf.
                // Gate is at q_f32[h*hd*2 + hd .. h*hd*2 + 2*hd-1] for each head.
                // Attention writes to proj_buf[0..nh*hd-1] which overlaps gates.
                float* saved_gate = attn_scratch;  // [2048] scratch
                for (int i = tid; i < nh * hd; i += bs) {
                    const int h = i / hd;
                    const int d = i % hd;
                    saved_gate[i] = q_f32[h * hd * 2 + hd + d];
                }
                __syncthreads();

                // Step E: Parallel attention — Flash-style online softmax
                // All 256 threads cooperate per head: parallel Q·K reduction,
                // each thread accumulates one V dimension with running max/sum.
                // hd == bs == 256, so thread tid owns dimension tid.
                float* attn_flat = proj_buf;  // reuse projection buffer [8224] > 2048

                const T* ck = static_cast<const T*>(L.kv_cache_k);
                const T* cv = static_cast<const T*>(L.kv_cache_v);

                for (int qh = 0; qh < nh; ++qh) {
                    const int kvh = qh / kv_groups;
                    const float* q_head = q_f32 + qh * hd * 2;

                    // Per-thread online softmax accumulators (one V dim each)
                    float my_acc = 0.0f;
                    float my_max = -1e30f;
                    float my_sum = 0.0f;

                    const float q_val = q_head[tid];
                    const size_t kv_head_base =
                        static_cast<size_t>(kvh) * L.kv_max_t * hd;

                    for (int t = 0; t < kv_len; ++t) {
                        const size_t pos_base = kv_head_base +
                            static_cast<size_t>(t) * hd;

                        // All 256 threads: partial Q·K dot product
                        float partial = q_val *
                            dotcache_qwen35_to_float(ck[pos_base + tid]);
                        lds[tid] = partial;
                        __syncthreads();
                        for (int s = bs / 2; s > 0; s >>= 1) {
                            if (tid < s) lds[tid] += lds[tid + s];
                            __syncthreads();
                        }
                        float score = lds[0] * scale;

                        // Load this thread's V dimension
                        float v_val =
                            dotcache_qwen35_to_float(cv[pos_base + tid]);

                        // Online softmax + V accumulation
                        float old_max = my_max;
                        my_max = fmaxf(my_max, score);
                        float rescale = expf(old_max - my_max);
                        float w = expf(score - my_max);
                        my_acc = my_acc * rescale + w * v_val;
                        my_sum = my_sum * rescale + w;
                    }

                    // Write normalized attention output
                    attn_flat[qh * hd + tid] =
                        (my_sum > 0.0f) ? (my_acc / my_sum) : 0.0f;
                    __syncthreads();
                }

                // Step F: Gate — attn_flat[nh*hd] * sigmoid(gate[nh*hd])
                // Gate was saved to saved_gate before attention overwrote proj_buf.
                // Output: gated_output[nh*hd = 2048] stored back in attn_flat
                for (int i = tid; i < nh * hd; i += bs) {
                    float gate_val = saved_gate[i];
                    float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
                    attn_flat[i] *= sigmoid_gate;
                }
                __syncthreads();

            }
            // Grid barrier: block 0 wrote attn_flat, all blocks need it for o_proj
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step G: o_proj [hidden_dim, attn_size] × attn_flat → hidden_f32 (fused residual)
            // ALL blocks work-steal rows; result added directly to hidden_f32
            {
                const int attn_size = L.attn_num_heads * L.attn_head_dim;  // 2048

                // Cache attn_flat in LDS for fast o_proj reads
                for (int c = tid; c < attn_size; c += bs)
                    lds_input[c] = proj_buf[c];
                __syncthreads();

                const T* ow = static_cast<const T*>(L.o_proj_w);

                if (qwen08_attn_hero) {
                    for (int sr = blockIdx.x; sr < hidden_dim; sr += nb) {
                        const T* wr = ow + static_cast<size_t>(sr) * attn_size;
                        float p = dotcache_qwen35_dot_row_input_f32_hero(wr, lds_input, attn_size);
                        const float sum = dotcache_qwen35_block_sum_256(p, lds);
                        if (tid == 0) hidden_f32[sr] += sum;
                        __syncthreads();
                    }
                } else {
                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);
                    for (;;) {
                        __shared__ unsigned int sr;
                        if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                        __syncthreads();
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        const T* wr = ow + static_cast<size_t>(sr) * attn_size;
                        float p = 0.0f;
                        for (int c = tid; c < attn_size; c += bs) {
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        }
                        lds[tid] = p;
                        __syncthreads();
                        for (int s = bs / 2; s > 0; s >>= 1) {
                            if (tid < s) lds[tid] += lds[tid + s];
                            __syncthreads();
                        }
                        if (tid == 0) hidden_f32[sr] += lds[0];
                        __syncthreads();
                    }
                }
            }
            grid_barrier(barrier_counter, barrier_flag, nb);

        } else {
            // ---- LINEAR ATTENTION ----
            // Step A: qkv/z/b/a projections via work-stealing
            const int total_proj = L.qkv_out_dim + L.z_out_dim + 16 + 16;
            {
                if (qwen08_hero) {
                    for (int sr = blockIdx.x; sr < total_proj; sr += nb) {
                        const T* w;
                        int row;
                        if (sr < L.qkv_out_dim) {
                            w = static_cast<const T*>(L.qkv_proj_w);
                            row = sr;
                        } else if (sr < L.qkv_out_dim + L.z_out_dim) {
                            w = static_cast<const T*>(L.z_proj_w);
                            row = sr - L.qkv_out_dim;
                        } else if (sr < L.qkv_out_dim + L.z_out_dim + 16) {
                            w = static_cast<const T*>(L.b_proj_w);
                            row = sr - L.qkv_out_dim - L.z_out_dim;
                        } else {
                            w = static_cast<const T*>(L.a_proj_w);
                            row = sr - L.qkv_out_dim - L.z_out_dim - 16;
                        }
                        const T* wr = w + static_cast<size_t>(row) * hidden_dim;

                        float p = 0.0f;
                        for (int c = tid; c < hidden_dim; c += bs)
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        const float sum = dotcache_qwen35_block_sum_256(p, lds);
                        if (tid == 0) proj_buf[sr] = sum;
                        __syncthreads();
                    }
                } else {
                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);
                    for (;;) {
                        __shared__ unsigned int sr;
                        if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                        __syncthreads();
                        if (sr >= static_cast<unsigned int>(total_proj)) break;

                        const T* w;
                        int row;
                        if (sr < static_cast<unsigned int>(L.qkv_out_dim)) {
                            w = static_cast<const T*>(L.qkv_proj_w);
                            row = sr;
                        } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim)) {
                            w = static_cast<const T*>(L.z_proj_w);
                            row = sr - L.qkv_out_dim;
                        } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim + 16)) {
                            w = static_cast<const T*>(L.b_proj_w);
                            row = sr - L.qkv_out_dim - L.z_out_dim;
                        } else {
                            w = static_cast<const T*>(L.a_proj_w);
                            row = sr - L.qkv_out_dim - L.z_out_dim - 16;
                        }
                        const T* wr = w + static_cast<size_t>(row) * hidden_dim;

                        float p = 0.0f;
                        for (int c = tid; c < hidden_dim; c += bs)
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        lds[tid] = p;
                        __syncthreads();
                        for (int s = bs/2; s > 0; s >>= 1) {
                            if (tid < s) lds[tid] += lds[tid+s];
                            __syncthreads();
                        }
                        if (tid == 0) proj_buf[sr] = lds[0];
                        __syncthreads();
                    }
                }
            }
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step B-E: conv1d, recurrent state, gated norm, out_proj
            // Block 0 handles sequential recurrent operations.
            // O_proj uses all blocks via work-stealing.
            const bool qwen08_linear_hero =
                qwen08_hero &&
                bs == 256 &&
                nb >= 16 &&
                L.linear_num_v_heads == 16 &&
                L.linear_head_k_dim == 128 &&
                L.linear_head_v_dim == 128 &&
                L.qkv_out_dim == 6144 &&
                L.z_out_dim == 2048;

            if (qwen08_linear_hero) {
                float* qkv_f32 = proj_buf;
                float* z_f32 = proj_buf + L.qkv_out_dim;
                float* b_f32 = proj_buf + L.qkv_out_dim + L.z_out_dim;
                float* a_f32 = b_f32 + 16;

                const int conv_dim = L.qkv_out_dim;
                const int kern = L.conv_kernel_size;
                const int nv = L.linear_num_v_heads;
                const int hkd = L.linear_head_k_dim;
                const int hvd = L.linear_head_v_dim;
                const int key_dim = nv * hkd;
                const int val_dim = L.linear_value_dim;
                const int q_key_offset = 0;
                const int k_key_offset = key_dim;
                const int v_val_offset = key_dim * 2;

                {
                    T* cs = static_cast<T*>(L.conv_state);
                    const T* cw = static_cast<const T*>(L.conv1d_w);
                    float* conv_out = gate_up;
                    for (int c = tid + blockIdx.x * bs; c < conv_dim; c += bs * nb) {
                        T qkv_bf16 = dotcache_qwen35_from_float<T>(qkv_f32[c]);
                        float acc = 0.0f;
                        for (int t = 0; t < kern - 1; ++t) {
                            acc += dotcache_qwen35_to_float(cs[c * (kern - 1) + t]) *
                                   dotcache_qwen35_to_float(cw[c * kern + t]);
                        }
                        acc += dotcache_qwen35_to_float(qkv_bf16) *
                               dotcache_qwen35_to_float(cw[c * kern + (kern - 1)]);
                        conv_out[c] = acc / (1.0f + expf(-acc));

                        for (int t = 0; t < kern - 2; ++t) {
                            cs[c * (kern - 1) + t] = cs[c * (kern - 1) + t + 1];
                        }
                        cs[c * (kern - 1) + (kern - 2)] = qkv_bf16;
                    }
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (blockIdx.x < nv) {
                    const int h = blockIdx.x;
                    float* conv_out = gate_up;

                    float sq = 0.0f;
                    for (int d = tid; d < hkd; d += bs) {
                        float qv = conv_out[q_key_offset + h * hkd + d];
                        sq += qv * qv;
                    }
                    lds[tid] = sq;
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    float q_inv = rsqrtf(lds[0] + 1e-6f) * rsqrtf(static_cast<float>(hkd));
                    for (int d = tid; d < hkd; d += bs) {
                        conv_out[q_key_offset + h * hkd + d] *= q_inv;
                    }
                    __syncthreads();

                    sq = 0.0f;
                    for (int d = tid; d < hkd; d += bs) {
                        float kv = conv_out[k_key_offset + h * hkd + d];
                        sq += kv * kv;
                    }
                    lds[tid] = sq;
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    float k_inv = rsqrtf(lds[0] + 1e-6f);
                    for (int d = tid; d < hkd; d += bs) {
                        conv_out[k_key_offset + h * hkd + d] *= k_inv;
                    }
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                const int linear_head_slices =
                    (nb >= nv * 4) ? 4 : ((nb >= nv * 2) ? 2 : 1);

                if (blockIdx.x < nv * linear_head_slices) {
                    const int h = blockIdx.x / linear_head_slices;
                    const int slice = blockIdx.x % linear_head_slices;
                    float* conv_out = gate_up;
                    float* sh = static_cast<float*>(L.recurrent_state) + h * hkd * hvd;
                    const T* dt_bw = static_cast<const T*>(L.dt_bias_w);
                    const T* ale_w = static_cast<const T*>(L.a_log_exp_w);
                    const int v_chunk = hvd / linear_head_slices;
                    const int v_local = tid % v_chunk;
                    const int v = slice * v_chunk + v_local;
                    const int part = tid / v_chunk;
                    const int part_count = bs / v_chunk;

                    const float beta = 1.0f / (1.0f + expf(-b_f32[h]));
                    float decay = 1.0f;
                    if (dt_bw != nullptr && ale_w != nullptr) {
                        float sp = logf(1.0f + expf(
                            a_f32[h] + dotcache_qwen35_to_float(dt_bw[h])));
                        decay = expf(-sp * dotcache_qwen35_to_float(ale_w[h]));
                    }

                    for (int k = part; k < hkd; k += part_count) {
                        sh[k * hvd + v] *= decay;
                    }

                    float kv_mem_v = 0.0f;
                    for (int k = part; k < hkd; k += part_count) {
                        kv_mem_v += sh[k * hvd + v] *
                                    conv_out[k_key_offset + h * hkd + k];
                    }
                    lds[tid] = kv_mem_v;
                    __syncthreads();
                    if (tid < v_chunk) {
                        float kv_mem_total = lds[tid];
                        for (int off = v_chunk; off < bs; off += v_chunk) {
                            kv_mem_total += lds[tid + off];
                        }
                        const float val = conv_out[v_val_offset + h * hvd + v];
                        lds[tid] = (val - kv_mem_total) * beta;
                    }
                    __syncthreads();
                    const float delta = lds[v_local];

                    for (int k = part; k < hkd; k += part_count) {
                        sh[k * hvd + v] +=
                            conv_out[k_key_offset + h * hkd + k] * delta;
                    }

                    float out_v = 0.0f;
                    for (int k = part; k < hkd; k += part_count) {
                        out_v += sh[k * hvd + v] *
                                 conv_out[q_key_offset + h * hkd + k];
                    }
                    lds[tid] = out_v;
                    __syncthreads();
                    if (tid < v_chunk) {
                        float out_total = lds[tid];
                        for (int off = v_chunk; off < bs; off += v_chunk) {
                            out_total += lds[tid + off];
                        }
                        attn_scratch[h * hvd + v] = out_total;
                    }
                    __syncthreads();
                }
                grid_barrier(barrier_counter, barrier_flag, nb);

                if (blockIdx.x < nv) {
                    const int h = blockIdx.x;
                    const float* nw = static_cast<const float*>(L.linear_norm_w);
                    const float* ho = attn_scratch + h * hvd;

                    float sq = 0.0f;
                    for (int j = tid; j < hvd; j += bs) {
                        sq += ho[j] * ho[j];
                    }
                    lds[tid] = sq;
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    const float rms_inv =
                        rsqrtf(lds[0] / static_cast<float>(hvd) + L.linear_norm_eps);

                    for (int j = tid; j < hvd; j += bs) {
                        float n = attn_scratch[h * hvd + j] * rms_inv;
                        float w = (nw != nullptr) ? nw[j] : 1.0f;
                        float zv = z_f32[h * hvd + j];
                        attn_scratch[h * hvd + j] = n * w * (zv / (1.0f + expf(-zv)));
                    }
                    __syncthreads();
                }
            } else if (blockIdx.x == 0) {
                float* qkv_f32 = proj_buf;
                float* z_f32 = proj_buf + L.qkv_out_dim;
                float* b_f32 = proj_buf + L.qkv_out_dim + L.z_out_dim;
                float* a_f32 = b_f32 + 16;

                const int conv_dim = L.qkv_out_dim;  // 6144
                const int kern = L.conv_kernel_size;  // 4
                const int nv = L.linear_num_v_heads;  // 16
                const int hkd = L.linear_head_k_dim;  // 128
                const int hvd = L.linear_head_v_dim;  // 128
                const int key_dim = nv * hkd;         // 2048 (or num_k_heads * k_head_dim)
                const int val_dim = L.linear_value_dim; // 2048

                // Step B: Conv1d stateful update
                // conv_state: [conv_dim, kern-1] BF16 (sliding window of past values)
                // New qkv values: qkv_f32[conv_dim] (current step)
                // Conv: for each channel c, output = sum(conv_state[c,t] * weight[c,t]) + qkv[c] * weight[c,kern-1]
                // Then update conv_state: shift left, append new value
                {
                    T* cs = static_cast<T*>(L.conv_state);
                    const T* cw = static_cast<const T*>(L.conv1d_w);
                    // cw layout: [conv_dim, 1, kern] — depthwise conv weight
                    // We treat it as [conv_dim, kern]

                    float* conv_out = gate_up;  // reuse MLP scratch for conv output
                    for (int c = tid; c < conv_dim; c += bs) {
                        // Depthwise conv: sum(state[t] * weight[t]) + new_value * weight[kern-1]
                        // Match standard path: convert new value to BF16 for consistency
                        T qkv_bf16 = dotcache_qwen35_from_float<T>(qkv_f32[c]);
                        float acc = 0.0f;
                        for (int t = 0; t < kern - 1; ++t) {
                            acc += dotcache_qwen35_to_float(cs[c * (kern-1) + t])
                                 * dotcache_qwen35_to_float(cw[c * kern + t]);
                        }
                        acc += dotcache_qwen35_to_float(qkv_bf16)
                             * dotcache_qwen35_to_float(cw[c * kern + (kern-1)]);
                        // SiLU activation
                        conv_out[c] = acc / (1.0f + expf(-acc));

                        // Update conv_state: shift left, append new
                        for (int t = 0; t < kern - 2; ++t) {
                            cs[c * (kern-1) + t] = cs[c * (kern-1) + t + 1];
                        }
                        cs[c * (kern-1) + (kern-2)] = qkv_bf16;
                    }
                    __syncthreads();

                    // conv_out[conv_dim] now has the post-conv, post-SiLU output
                    // Split into query, key, value
                    // Layout: [key_dim | key_dim | val_dim] within conv_dim
                    // Actually conv_dim = key_dim*2 + val_dim = 2048+2048+2048? No...
                    // conv_dim = 6144. With nv=16, hkd=128, hvd=128:
                    // key_dim = 16*128 = 2048, val_dim = 16*128 = 2048
                    // So 6144 = 2048 + 2048 + 2048? That's only if there are separate Q/K/V.
                    // Actually for linear attention: conv_dim = in_proj_qkv out = 6144
                    // which packs Q, K, V differently. Let me use the existing split:
                    // First key_dim values = query (normalized later)
                    // Next key_dim values = key
                    // Next val_dim values = value
                    // But wait, with 16 k_heads * 128 k_dim = 2048 for Q
                    //          16 k_heads * 128 k_dim = 2048 for K
                    //          16 v_heads * 128 v_dim = 2048 for V
                    // Total = 6144 ✓
                }

                // Debug: dump conv_out[0..3] and state[0..3] before normalization
                if (blockIdx.x == 0 && tid == 0 && layer == 0) {
                    float* dbg = mlp_out;  // reuse unused workspace for debug
                    dbg[0] = gate_up[0];     // conv_out Q head0 dim0
                    dbg[1] = gate_up[1];     // conv_out Q head0 dim1
                    dbg[2] = gate_up[2048];  // conv_out K head0 dim0
                    dbg[3] = gate_up[4096];  // conv_out V head0 dim0
                    dbg[4] = static_cast<float*>(L.recurrent_state)[0]; // state[0,0,0]
                    dbg[5] = static_cast<float*>(L.recurrent_state)[1]; // state[0,0,1]
                }
                __syncthreads();

                // Step C0: L2-normalize Q and K per-head (matches standard path)
                // Q_norm = Q / ||Q|| * rsqrt(k_head_dim)
                // K_norm = K / ||K||
                {
                    float* conv_out = gate_up;  // reused
                    const int nk = nv / L.linear_head_k_dim;  // num_k_heads (might differ from nv)
                    // Actually: num_k_heads = num_v_heads / head_repeat where head_repeat = nv/nk
                    // For Qwen3.5: nk=16, nv=16, head_repeat=1 so each v_head maps to one k_head
                    // Q layout: conv_out[k_head * hkd .. (k_head+1) * hkd]
                    // K layout: conv_out[key_dim + k_head * hkd .. key_dim + (k_head+1) * hkd]

                    // Normalize Q and K per head using threads 0..nv-1
                    if (tid < nv) {
                        const int h = tid;
                        // Q head
                        float* qh = conv_out + h * hkd;  // q_key_offset + h * hkd
                        float sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += qh[d] * qh[d];
                        float inv = rsqrtf(sq + 1e-6f) * rsqrtf(static_cast<float>(hkd));
                        for (int d = 0; d < hkd; ++d)
                            qh[d] *= inv;

                        // K head
                        float* kh = conv_out + key_dim + h * hkd;
                        sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += kh[d] * kh[d];
                        inv = rsqrtf(sq + 1e-6f);
                        for (int d = 0; d < hkd; ++d)
                            kh[d] *= inv;
                    }
                    __syncthreads();
                }

                // Step C: Parallel delta recurrent state update
                // state: [nv, hkd, hvd] F32 — the recurrent memory
                // 256 threads process 2 heads at a time: threads 0-127 → head h,
                // threads 128-255 → head h+1. Each thread owns one v-dimension.
                // Per head: kv_mem = state @ key, delta = (val - kv_mem) * beta,
                //           state += outer(key, delta), output = state @ query
                {
                    float* conv_out = gate_up;  // reused
                    float* state = static_cast<float*>(L.recurrent_state);

                    const T* dt_bw = static_cast<const T*>(L.dt_bias_w);
                    const T* ale_w = static_cast<const T*>(L.a_log_exp_w);

                    const int q_key_offset = 0;
                    const int k_key_offset = key_dim;
                    const int v_val_offset = key_dim * 2;
                    const int v = tid % hvd;          // v-dimension for this thread

                    // Process 2 heads per iteration (256 threads / 128 hvd)
                    for (int hp = 0; hp < nv; hp += 2) {
                        const int h = hp + (tid >= hvd ? 1 : 0);

                        if (h < nv) {
                            float* sh = state + h * hkd * hvd;
                            // Compute beta = sigmoid(b) and decay = exp(g) for this head only
                            const float beta = 1.0f / (1.0f + expf(-b_f32[h]));
                            float decay = 1.0f;
                            if (dt_bw != nullptr && ale_w != nullptr) {
                                float sp = logf(1.0f + expf(
                                    a_f32[h] + dotcache_qwen35_to_float(dt_bw[h])));
                                decay = expf(-sp * dotcache_qwen35_to_float(ale_w[h]));
                            }

                            // Apply decay: state *= exp(g)
                            for (int k = 0; k < hkd; ++k) {
                                sh[k * hvd + v] *= decay;
                            }

                            // kv_mem[v] = sum_k(state[k, v] * key[k])
                            float kv_mem_v = 0.0f;
                            for (int k = 0; k < hkd; ++k) {
                                kv_mem_v += sh[k * hvd + v]
                                          * conv_out[k_key_offset + h * hkd + k];
                            }

                            // delta = (value - kv_mem) * beta
                            const float val = conv_out[v_val_offset + h * hvd + v];
                            const float delta = (val - kv_mem_v) * beta;

                            // state[k, v] += key[k] * delta (each thread owns its v)
                            for (int k = 0; k < hkd; ++k) {
                                sh[k * hvd + v] +=
                                    conv_out[k_key_offset + h * hkd + k] * delta;
                            }

                            // output[v] = sum_k(state[k, v] * query[k])
                            // (query already L2-normed with 1/sqrt(hkd) scaling in Step C0)
                            float out_v = 0.0f;
                            for (int k = 0; k < hkd; ++k) {
                                out_v += sh[k * hvd + v]
                                       * conv_out[q_key_offset + h * hkd + k];
                            }

                            attn_scratch[h * hvd + v] = out_v;
                        }
                        __syncthreads();
                    }
                }

                // Step D: Per-head RMSNorm + weight + SiLU(z) gating
                // Single-threaded to avoid LDS reduction issues (will parallelize once correct)
                {
                    // Norm weight is F32 (not BF16) — matches the standard path's dtype
                    const float* nw = static_cast<const float*>(L.linear_norm_w);
                    // Per-head RMSNorm: weight is [hvd] shared across all heads
                    // Two-pass to avoid data race (read RMS → barrier → write normalized)
                    // Pass 1: compute per-head rms_inv, store in lds[h]
                    // With nv=16 heads, use threads 0..15 to compute each head's RMS
                    if (tid < nv) {
                        const int h = tid;
                        const float* ho = attn_scratch + h * hvd;
                        float sq = 0.0f;
                        for (int j = 0; j < hvd; ++j)
                            sq += ho[j] * ho[j];
                        lds[h] = rsqrtf(sq / static_cast<float>(hvd) + L.linear_norm_eps);
                    }
                    __syncthreads();
                    // Pass 2: normalize + weight + gate
                    for (int d = tid; d < val_dim; d += bs) {
                        const int h = d / hvd;
                        const int hd_idx = d % hvd;
                        float rms_inv = lds[h];
                        float n = attn_scratch[d] * rms_inv;
                        float w = (nw != nullptr) ? nw[hd_idx] : 1.0f;
                        float zv = z_f32[d];
                        attn_scratch[d] = n * w * (zv / (1.0f + expf(-zv)));
                    }
                    __syncthreads();
                }

            }
            // Grid barrier: block 0 wrote attn_scratch, all blocks need it for out_proj
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step E: out_proj [hidden_dim, val_dim] × attn_scratch → hidden_f32 (fused residual)
            // ALL blocks work-steal rows; result added directly to hidden_f32
            {
                const int vd = L.linear_value_dim;  // 2048

                // Cache attn_scratch in LDS for fast out_proj reads
                for (int c = tid; c < vd; c += bs)
                    lds_input[c] = attn_scratch[c];
                __syncthreads();

                const T* ow = static_cast<const T*>(L.linear_out_proj_w);

                if (qwen08_linear_hero) {
                    for (int sr = blockIdx.x; sr < hidden_dim; sr += nb) {
                        const T* wr = ow + static_cast<size_t>(sr) * vd;
                        float p = dotcache_qwen35_dot_row_input_f32_hero(wr, lds_input, vd);
                        const float sum = dotcache_qwen35_block_sum_256(p, lds);
                        if (tid == 0) hidden_f32[sr] += sum;
                        __syncthreads();
                    }
                } else {
                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);
                    for (;;) {
                        __shared__ unsigned int sr;
                        if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                        __syncthreads();
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        const T* wr = ow + static_cast<size_t>(sr) * vd;
                        float p = 0.0f;
                        for (int c = tid; c < vd; c += bs) {
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        }
                        lds[tid] = p;
                        __syncthreads();
                        for (int s = bs / 2; s > 0; s >>= 1) {
                            if (tid < s) lds[tid] += lds[tid + s];
                            __syncthreads();
                        }
                        if (tid == 0) hidden_f32[sr] += lds[0];
                        __syncthreads();
                    }
                }
            }
            grid_barrier(barrier_counter, barrier_flag, nb);
        }

        // Residual add for token mixer is fused into o_proj/out_proj above.
        // === Post-attention RMSNorm (block 0 only) ===
        if (blockIdx.x == 0) {
            block_rms_norm_global<T>(normed, hidden_f32,
                static_cast<const T*>(L.post_attn_norm_w),
                hidden_dim, L.post_attn_norm_eps, lds);
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // All blocks: cache normed vector in LDS for fast MLP projection reads
        for (int c = tid; c < hidden_dim; c += bs)
            lds_input[c] = normed[c];
        __syncthreads();

        // === Fused MLP gate+up+SwiGLU (all blocks work-steal) ===
        // Each work unit computes gate[i] AND up[i], applies SwiGLU, writes result.
        // Removes 2 barriers vs. separate gate+up → SwiGLU phases.
        {
            const T* gw = static_cast<const T*>(L.gate_proj_w);
            const T* uw = static_cast<const T*>(L.up_proj_w);
            if (qwen08_hero) {
                for (int sr = blockIdx.x; sr < L.intermediate_size; sr += nb) {
                    const T* gr = gw + static_cast<size_t>(sr) * hidden_dim;
                    float gp = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs)
                        gp += dotcache_qwen35_to_float(gr[c]) * lds_input[c];
                    float gate_val = dotcache_qwen35_block_sum_256(gp, lds);

                    const T* ur = uw + static_cast<size_t>(sr) * hidden_dim;
                    float up = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs)
                        up += dotcache_qwen35_to_float(ur[c]) * lds_input[c];
                    float up_val = dotcache_qwen35_block_sum_256(up, lds);

                    if (tid == 0) {
                        float silu = gate_val / (1.0f + expf(-gate_val));
                        gate_up[sr] = silu * up_val;
                    }
                    __syncthreads();
                }
            } else {
                if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                grid_barrier(barrier_counter, barrier_flag, nb);
                for (;;) {
                    __shared__ unsigned int sr;
                    if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                    __syncthreads();
                    if (sr >= static_cast<unsigned int>(L.intermediate_size)) break;

                    const T* gr = gw + static_cast<size_t>(sr) * hidden_dim;
                    float gp = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs)
                        gp += dotcache_qwen35_to_float(gr[c]) * lds_input[c];
                    lds[tid] = gp;
                    __syncthreads();
                    for (int s = bs/2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid+s];
                        __syncthreads();
                    }
                    float gate_val = lds[0];

                    const T* ur = uw + static_cast<size_t>(sr) * hidden_dim;
                    float up = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs)
                        up += dotcache_qwen35_to_float(ur[c]) * lds_input[c];
                    lds[tid] = up;
                    __syncthreads();
                    for (int s = bs/2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid+s];
                        __syncthreads();
                    }
                    float up_val = lds[0];

                    if (tid == 0) {
                        float silu = gate_val / (1.0f + expf(-gate_val));
                        gate_up[sr] = silu * up_val;
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // === MLP down_proj (all blocks work-steal) ===
        // Cache SwiGLU output in LDS for fast down_proj reads
        for (int c = tid; c < L.intermediate_size; c += bs)
            lds_input[c] = gate_up[c];
        __syncthreads();

        {
            const T* dw = static_cast<const T*>(L.down_proj_w);
            if (qwen08_hero) {
                for (int sr = blockIdx.x; sr < hidden_dim; sr += nb) {
                    const T* wr = dw + static_cast<size_t>(sr) * L.intermediate_size;
                    float p =
                        dotcache_qwen35_dot_row_input_f32_hero(wr, lds_input, L.intermediate_size);
                    const float sum = dotcache_qwen35_block_sum_256(p, lds);
                    if (tid == 0) hidden_f32[sr] += sum;
                    __syncthreads();
                }
            } else {
                if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                grid_barrier(barrier_counter, barrier_flag, nb);
                for (;;) {
                    __shared__ unsigned int sr;
                    if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                    __syncthreads();
                    if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                    const T* wr = dw + static_cast<size_t>(sr) * L.intermediate_size;
                    float p = 0.0f;
                    for (int c = tid; c < L.intermediate_size; c += bs)
                        p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                    lds[tid] = p;
                    __syncthreads();
                    for (int s = bs/2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid+s];
                        __syncthreads();
                    }
                    if (tid == 0) hidden_f32[sr] += lds[0];
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // Residual add for MLP is fused into down_proj above.
    }

    // Write back F32 → BF16
    for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
        hidden_io[c] = dotcache_qwen35_from_float<T>(hidden_f32[c]);
    }
}
