#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
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
    int rms_norm_add_unit_offset;      // 1 for Gemma-style RMSNorm weight+1, 0 for plain RMSNorm
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
    void* debug_linear_trace_out;      // optional F32[(kern-1)+2] export for one linear channel's Step-B state
    int debug_linear_trace_channel;    // selected linear channel for debug_linear_trace_out, -1 disables
};

template <typename T>
__device__ inline float supersonic_qwen35_to_float(T value);

template <>
__device__ inline float supersonic_qwen35_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float supersonic_qwen35_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float supersonic_qwen35_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T supersonic_qwen35_from_float(float value);

template <>
__device__ inline __half supersonic_qwen35_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float supersonic_qwen35_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 supersonic_qwen35_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float supersonic_qwen35_binary_op_apply(int op, float lhs, float rhs) {
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

__device__ inline size_t supersonic_qwen35_broadcast_elem_index(
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
__device__ inline float supersonic_qwen35_dot_row(
    const T* lhs,
    const T* rhs,
    int size
) {
    float dot = 0.0f;
    for (int idx = 0; idx < size; ++idx) {
        dot += supersonic_qwen35_to_float(lhs[idx]) * supersonic_qwen35_to_float(rhs[idx]);
    }
    return dot;
}

template <>
__device__ inline float supersonic_qwen35_dot_row<__half>(
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

__device__ inline float supersonic_qwen35_exp_fast(float x) {
    return __expf(x);
}

__device__ inline float supersonic_qwen35_sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = supersonic_qwen35_exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = supersonic_qwen35_exp_fast(x);
    return e / (1.0f + e);
}

__device__ inline float supersonic_qwen35_softplus_fast(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return supersonic_qwen35_exp_fast(x);
    return log1pf(supersonic_qwen35_exp_fast(x));
}

template <typename T>
__device__ inline float supersonic_qwen35_conv4_contiguous_t_ge3(
    const T* mixed_qkv,
    size_t mixed_c_offset,
    size_t t,
    const T* weights,
    size_t weight_offset
) {
    const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
    const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
    const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
    const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
    const float x0 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 3)]);
    const float x1 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 2)]);
    const float x2 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 1)]);
    const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + t]);
    return ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
}

template <>
__device__ inline float supersonic_qwen35_conv4_contiguous_t_ge3<__half>(
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

__device__ inline float supersonic_qwen35_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    return value;
}

template <typename T>
__global__ void supersonic_qwen35_full_attention_prefill_kernel(
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

        // Online single-pass softmax (FlashAttention recurrence).
        //
        //   m_i = running max of scores
        //   l_i = running denom (sum of exp(score - m_i))
        //   acc = running output accumulator (in registers, per lane)
        //
        // On each new k_pos we form score = Q · K[k_pos] * scale, take
        // m_new = max(m_i, score), rescale (acc, l_i) by alpha = exp(m_i - m_new),
        // add the new contribution exp(score - m_new) * V[k_pos], and update m_i.
        // Final out = acc / l_i.

        __shared__ float shared_max;
        __shared__ float shared_alpha;
        __shared__ float shared_weight;
        __shared__ float shared_denom;
        __shared__ float shared_inv_denom;

        if (lane == 0) {
            shared_max = -INFINITY;
            shared_denom = 0.0f;
        }
        __syncthreads();

        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;
            const T* v_row = v_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
            }
            float score = supersonic_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                float m_old = shared_max;
                float m_new = score > m_old ? score : m_old;
                // alpha == 0 on the very first iteration (m_old = -inf): we
                // explicitly zero the accumulators rather than evaluating
                // expf(-inf - finite) which can produce NaN on some HW.
                float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                shared_alpha = alpha;
                shared_weight = expf(score - m_new);
                shared_denom = shared_denom * alpha + shared_weight;
                shared_max = m_new;
            }
            __syncthreads();

            const float alpha = shared_alpha;
            const float weight = shared_weight;
            for (int i = 0; i < local_count; ++i) {
                local_acc[i] = local_acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[local_dims[i]]);
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
        }
        __syncthreads();
        const float inv_denom = shared_inv_denom;
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * inv_denom;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void supersonic_qwen35_full_attention_tree_prefill_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int tree_len,
    int prefix_len,
    int prefix_stride,
    int head_dim,
    int num_kv_groups,
    float scale,
    const T* query,
    const T* prefix_key,
    const T* prefix_value,
    const T* tree_key,
    const T* tree_value,
    const uint8_t* visibility,
    float* out,
    unsigned int* row_counter
) {
    const int lane = threadIdx.x;
    if (lane >= warpSize) {
        return;
    }

    const int total_rows = batch_size * q_heads * tree_len;
    while (true) {
        unsigned int row = 0;
        if (lane == 0) {
            row = atomicAdd(row_counter, 1u);
        }
        row = __shfl(row, 0);
        if ((int)row >= total_rows) {
            return;
        }

        const int q_pos = row % tree_len;
        const int q_head = (row / tree_len) % q_heads;
        const int batch = row / (tree_len * q_heads);
        const int kv_head = q_head / num_kv_groups;

        const T* q_row = query + (((batch * q_heads + q_head) * tree_len + q_pos) * head_dim);
        const T* prefix_k_head = prefix_key + ((batch * kv_heads + kv_head) * prefix_stride * head_dim);
        const T* prefix_v_head = prefix_value + ((batch * kv_heads + kv_head) * prefix_stride * head_dim);
        const T* tree_k_head = tree_key + ((batch * kv_heads + kv_head) * tree_len * head_dim);
        const T* tree_v_head = tree_value + ((batch * kv_heads + kv_head) * tree_len * head_dim);
        float* out_row = out + (((batch * q_heads + q_head) * tree_len + q_pos) * head_dim);

        float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int local_dims[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int local_count = 0;
        for (int d = lane; d < head_dim && local_count < 8; d += warpSize) {
            local_dims[local_count++] = d;
        }

        __shared__ float shared_max;
        __shared__ float shared_alpha;
        __shared__ float shared_weight;
        __shared__ float shared_denom;
        __shared__ float shared_inv_denom;

        if (lane == 0) {
            shared_max = -INFINITY;
            shared_denom = 0.0f;
        }
        __syncthreads();

        for (int k_pos = 0; k_pos < prefix_len; ++k_pos) {
            const T* k_row = prefix_k_head + k_pos * head_dim;
            const T* v_row = prefix_v_head + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
            }
            const float score = supersonic_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                const float m_old = shared_max;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                shared_alpha = alpha;
                shared_weight = expf(score - m_new);
                shared_denom = shared_denom * alpha + shared_weight;
                shared_max = m_new;
            }
            __syncthreads();

            const float alpha = shared_alpha;
            const float weight = shared_weight;
            for (int i = 0; i < local_count; ++i) {
                local_acc[i] =
                    local_acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[local_dims[i]]);
            }
            __syncthreads();
        }

        for (int k_pos = 0; k_pos < tree_len; ++k_pos) {
            if (visibility[q_pos * tree_len + k_pos] == 0) {
                continue;
            }
            const T* k_row = tree_k_head + k_pos * head_dim;
            const T* v_row = tree_v_head + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
            }
            const float score = supersonic_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                const float m_old = shared_max;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                shared_alpha = alpha;
                shared_weight = expf(score - m_new);
                shared_denom = shared_denom * alpha + shared_weight;
                shared_max = m_new;
            }
            __syncthreads();

            const float alpha = shared_alpha;
            const float weight = shared_weight;
            for (int i = 0; i < local_count; ++i) {
                local_acc[i] =
                    local_acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[local_dims[i]]);
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
        }
        __syncthreads();
        const float inv_denom = shared_inv_denom;
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * inv_denom;
        }
        __syncthreads();
    }
}

template <typename T, int BM, int BK>
__global__ void supersonic_qwen35_full_attention_tree_prefill_tiled_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int tree_len,
    int prefix_len,
    int prefix_stride,
    int head_dim,
    int num_kv_groups,
    float scale,
    const T* query,
    const T* prefix_key,
    const T* prefix_value,
    const T* tree_key,
    const T* tree_value,
    const uint8_t* visibility,
    float* out
) {
    constexpr int ACC_MAX = 8;

    const int qr_base = blockIdx.x * BM;
    const int qh = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_h = qh / num_kv_groups;
    const int warp_y = threadIdx.y;
    const int lane = threadIdx.x;
    const int qr = qr_base + warp_y;
    if (lane >= warpSize) return;
    if (batch >= batch_size || qh >= q_heads) return;

    extern __shared__ unsigned char smem_raw[];
    T* k_tile = reinterpret_cast<T*>(smem_raw);
    T* v_tile = k_tile + BK * head_dim;

    const bool active = (qr < tree_len);
    const T* q_row = active ? (query + (((batch * q_heads + qh) * tree_len + qr) * head_dim)) : nullptr;
    const T* prefix_k_head = prefix_key + ((batch * kv_heads + kv_h) * prefix_stride * head_dim);
    const T* prefix_v_head = prefix_value + ((batch * kv_heads + kv_h) * prefix_stride * head_dim);
    const T* tree_k_head = tree_key + ((batch * kv_heads + kv_h) * tree_len * head_dim);
    const T* tree_v_head = tree_value + ((batch * kv_heads + kv_h) * tree_len * head_dim);
    float* out_row = active ? (out + (((batch * q_heads + qh) * tree_len + qr) * head_dim)) : nullptr;

    float acc[ACC_MAX];
    int acc_dim[ACC_MAX];
    int acc_count = 0;
    #pragma unroll
    for (int i = 0; i < ACC_MAX; ++i) acc[i] = 0.0f;
    for (int d = lane; d < head_dim && acc_count < ACC_MAX; d += warpSize) {
        acc_dim[acc_count++] = d;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int tile_base = 0; tile_base < prefix_len; tile_base += BK) {
        const int tile_len = min(BK, prefix_len - tile_base);
        const int load_total = tile_len * head_dim;
        const int total_threads = BM * warpSize;
        const int linear_tid = warp_y * warpSize + lane;
        for (int i = linear_tid; i < load_total; i += total_threads) {
            const int row = i / head_dim;
            const int col = i - row * head_dim;
            const int k_pos = tile_base + row;
            const size_t off = static_cast<size_t>(k_pos) * head_dim + static_cast<size_t>(col);
            k_tile[i] = prefix_k_head[off];
            v_tile[i] = prefix_v_head[off];
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < tile_len; ++j) {
                const T* k_row = k_tile + j * head_dim;
                const T* v_row = v_tile + j * head_dim;

                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += warpSize) {
                    partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
                }
                const float score = __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;
                const float m_old = m_i;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                const float weight = expf(score - m_new);
                l_i = l_i * alpha + weight;
                m_i = m_new;

                for (int i = 0; i < acc_count; ++i) {
                    acc[i] = acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        for (int k_pos = 0; k_pos < tree_len; ++k_pos) {
            if (visibility[qr * tree_len + k_pos] == 0) {
                continue;
            }
            const T* k_row = tree_k_head + k_pos * head_dim;
            const T* v_row = tree_v_head + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
            }
            const float score = __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;
            const float m_old = m_i;
            const float m_new = score > m_old ? score : m_old;
            const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            const float weight = expf(score - m_new);
            l_i = l_i * alpha + weight;
            m_i = m_new;

            for (int i = 0; i < acc_count; ++i) {
                acc[i] = acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
            }
        }

        const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int i = 0; i < acc_count; ++i) {
            out_row[acc_dim[i]] = acc[i] * inv;
        }
    }
}

// Stage 3: K-tiled FlashAttention-style prefill attention.
//
// Grid:  (ceil(q_len / BM), q_heads, batch)
// Block: (warpSize, BM)   — BM warps × warpSize lanes
//
// Each warp owns one query row qr = block_qr_base + warp_y and walks the
// K dimension in tiles of BK rows. K and V tiles are loaded into LDS
// once per tile and shared across all BM warps in the block, dropping
// global K/V bandwidth by ~BM× vs the single-warp kernel.
//
// Per-warp register state:
//   m_i           (float, running max of softmax scores)
//   l_i           (float, running denom)
//   acc[ACC_MAX]  (float per lane, output-side accumulators; up to ACC_MAX
//                  elements per lane covering ceil(head_dim / warpSize) dims)
template <typename T, int BM, int BK>
__global__ void supersonic_qwen35_full_attention_prefill_tiled_kernel(
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
    float* out
) {
    constexpr int ACC_MAX = 8;   // head_dim <= 8 * warpSize asserted in launcher

    const int qr_base = blockIdx.x * BM;
    const int qh      = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_h    = qh / num_kv_groups;
    const int warp_y  = threadIdx.y;        // 0..BM-1
    const int lane    = threadIdx.x;        // 0..warpSize-1
    const int qr      = qr_base + warp_y;
    if (lane >= warpSize) return;
    if (batch >= batch_size || qh >= q_heads) return;

    extern __shared__ unsigned char smem_raw[];
    T* k_tile = reinterpret_cast<T*>(smem_raw);
    T* v_tile = k_tile + BK * head_dim;

    const bool active = (qr < q_len);
    const int causal_limit = active ? min(kv_len, seqlen_offset + qr + 1) : 0;

    const T* q_row = active ? (query + (((batch * q_heads + qh) * q_len + qr) * head_dim)) : nullptr;
    const T* k_head_ptr = key   + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    const T* v_head_ptr = value + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    float*   out_row = active ? (out + (((batch * q_heads + qh) * q_len + qr) * head_dim)) : nullptr;

    // Per-lane output accumulators (in registers). Each lane owns the dims
    // d = lane, lane + warpSize, lane + 2*warpSize, ... (up to ACC_MAX of them).
    float acc[ACC_MAX];
    int   acc_dim[ACC_MAX];
    int   acc_count = 0;
    #pragma unroll
    for (int i = 0; i < ACC_MAX; ++i) acc[i] = 0.0f;
    for (int d = lane; d < head_dim && acc_count < ACC_MAX; d += warpSize) {
        acc_dim[acc_count++] = d;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Iterate K tiles. Even inactive warps must participate in cooperative
    // loads and __syncthreads() — they just won't write output.
    const int causal_for_block = active ? causal_limit : 0;
    // For the cooperative load we need an upper bound on K positions any
    // warp in this block will need. Different warps can have different
    // causal_limit values (qr varies). Use the maximum across the block.
    __shared__ int block_max_kpos;
    if (threadIdx.x == 0 && threadIdx.y == 0) block_max_kpos = 0;
    __syncthreads();
    if (lane == 0) atomicMax(&block_max_kpos, causal_for_block);
    __syncthreads();
    const int outer_limit = block_max_kpos;

    for (int tile_base = 0; tile_base < outer_limit; tile_base += BK) {
        const int tile_len = min(BK, outer_limit - tile_base);

        // Cooperative load: BM warps × warpSize lanes pull tile_len * head_dim
        // T values into LDS. Stride by total threads = BM * warpSize.
        const int load_total = tile_len * head_dim;
        const int total_threads = BM * warpSize;
        const int linear_tid = warp_y * warpSize + lane;
        for (int i = linear_tid; i < load_total; i += total_threads) {
            const int row = i / head_dim;
            const int col = i - row * head_dim;
            const int gk_pos = tile_base + row;
            const size_t off = (size_t)gk_pos * head_dim + (size_t)col;
            k_tile[i] = k_head_ptr[off];
            v_tile[i] = v_head_ptr[off];
        }
        __syncthreads();

        // Walk this tile, applying the online softmax recurrence per query row.
        if (active) {
            const int j_max = min(tile_len, causal_limit - tile_base);
            for (int j = 0; j < j_max; ++j) {
                const T* k_row = k_tile + j * head_dim;
                const T* v_row = v_tile + j * head_dim;

                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += warpSize) {
                    partial += supersonic_qwen35_to_float(q_row[d]) * supersonic_qwen35_to_float(k_row[d]);
                }
                // wave_sum is a __shfl_down butterfly that leaves the sum
                // only in lane 0; broadcast it so every lane sees the same
                // score (each lane runs its own per-output-dim recurrence).
                float score = __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;

                const float m_old = m_i;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                const float weight = expf(score - m_new);
                l_i = l_i * alpha + weight;
                m_i = m_new;

                for (int i = 0; i < acc_count; ++i) {
                    acc[i] = acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
                }
            }
        }
        __syncthreads();   // before next tile reuses k_tile/v_tile
    }

    if (active) {
        const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int i = 0; i < acc_count; ++i) {
            out_row[acc_dim[i]] = acc[i] * inv;
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_linear_prefill_conv_pack_kernel(
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
        acc += supersonic_qwen35_to_float(mixed_qkv[input_c_offset + t + static_cast<size_t>(tap)]) *
            supersonic_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
    out[tid] = supersonic_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void supersonic_qwen35_linear_tree_conv_pack_kernel(
    int batch_size,
    int conv_dim,
    int total_len,
    int tree_len,
    int kernel_size,
    const T* mixed_qkv,
    const T* weights,
    const int* parent_ids,
    T* out
) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim);
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim));
    const size_t rem = tid - b * static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim);
    const int t = static_cast<int>(rem / static_cast<size_t>(conv_dim));
    const size_t c = rem - static_cast<size_t>(t) * static_cast<size_t>(conv_dim);

    const int state_len = kernel_size - 1;
    const size_t input_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(total_len);
    const size_t input_c_offset = input_b_offset + c * static_cast<size_t>(total_len);
    const size_t weight_offset = c * static_cast<size_t>(kernel_size);

    float acc = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        const int steps = state_len - tap;
        int source_col = state_len + t;
        if (steps > 0) {
            int node = t;
            int walked = 0;
            while (walked < steps) {
                const int parent = parent_ids[node];
                if (parent < 0) {
                    break;
                }
                node = parent;
                ++walked;
            }
            if (walked == steps) {
                source_col = state_len + node;
            } else {
                source_col = tap + walked;
            }
        }
        if (source_col >= 0 && source_col < total_len) {
            acc += supersonic_qwen35_to_float(mixed_qkv[input_c_offset + static_cast<size_t>(source_col)]) *
                supersonic_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
        }
    }

    const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
    out[tid] = supersonic_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void supersonic_qwen35_linear_tree_conv_pack_indexed_kernel(
    int batch_size,
    int conv_dim,
    int total_len,
    int tree_len,
    int kernel_size,
    int source_stride,
    const T* mixed_qkv,
    const T* weights,
    const int* source_cols,
    T* out
) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t output_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim);
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim));
    const size_t rem = tid - b * static_cast<size_t>(tree_len) * static_cast<size_t>(conv_dim);
    const int t = static_cast<int>(rem / static_cast<size_t>(conv_dim));
    const size_t c = rem - static_cast<size_t>(t) * static_cast<size_t>(conv_dim);

    const size_t input_b_offset =
        b * static_cast<size_t>(conv_dim) * static_cast<size_t>(total_len);
    const size_t input_c_offset = input_b_offset + c * static_cast<size_t>(total_len);
    const size_t weight_offset = c * static_cast<size_t>(kernel_size);
    const size_t source_row = static_cast<size_t>(t) * static_cast<size_t>(source_stride);

    float acc = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        const int source_col = source_cols[source_row + static_cast<size_t>(tap)];
        if (source_col >= 0 && source_col < total_len) {
            acc += supersonic_qwen35_to_float(mixed_qkv[input_c_offset + static_cast<size_t>(source_col)]) *
                supersonic_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
        }
    }

    const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
    out[tid] = supersonic_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void supersonic_qwen35_linear_stateful_conv_kernel(
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
            x = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
        } else {
            const int state_idx = state_len + src;
            if (state_idx >= 0) {
                x = supersonic_qwen35_to_float(prev_state[state_c_offset + static_cast<size_t>(state_idx)]);
            }
        }
        acc += x * supersonic_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
    out[tid] = supersonic_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void supersonic_qwen35_linear_stateful_conv_value_decay_kernel(
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
                x = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
            } else {
                const int state_idx = state_len + src;
                if (state_idx >= 0) {
                    x = supersonic_qwen35_to_float(
                        prev_state[state_c_offset + static_cast<size_t>(state_idx)]
                    );
                }
            }
            acc += x * supersonic_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
        }

        const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
        out[tid] = supersonic_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = supersonic_qwen35_to_float(a[a_base + head]);
    const float bias = supersonic_qwen35_to_float(dt_bias[head]);
    const float decay = supersonic_qwen35_to_float(a_log_exp[head]);
    const float softplus = supersonic_qwen35_softplus_fast(a_val + bias);
    out[tid] = supersonic_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void supersonic_qwen35_linear_stateful_conv_value_decay_kernel_k4s3(
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
            acc = supersonic_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = supersonic_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
        out[tid] = supersonic_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = supersonic_qwen35_to_float(a[a_base + head]);
    const float bias = supersonic_qwen35_to_float(dt_bias[head]);
    const float decay = supersonic_qwen35_to_float(a_log_exp[head]);
    const float softplus = supersonic_qwen35_softplus_fast(a_val + bias);
    out[tid] = supersonic_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void supersonic_qwen35_linear_stateful_conv_value_decay_with_state_kernel_k4s3(
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
            acc = supersonic_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
            const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = supersonic_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = supersonic_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = supersonic_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * supersonic_qwen35_sigmoid_fast(acc);
        out[tid] = supersonic_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = supersonic_qwen35_to_float(a[a_base + head]);
    const float bias = supersonic_qwen35_to_float(dt_bias[head]);
    const float decay = supersonic_qwen35_to_float(a_log_exp[head]);
    const float softplus = supersonic_qwen35_softplus_fast(a_val + bias);
    out[tid] = supersonic_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, int MAX_K = 256>
__global__ void supersonic_qwen35_linear_decode_prepare_kernel(
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
    const int num_k_heads = num_v_heads / head_repeat;
    const int k_head = v_head % num_k_heads;
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
                x = supersonic_qwen35_to_float(mixed_qkv[mixed_batch_base + channel]);
            } else if (tap < state_len) {
                x = supersonic_qwen35_to_float(prev_conv_state[state_base + tap]);
            }
            acc += x * supersonic_qwen35_to_float(weights[weight_base + tap]);
        }
        return acc * supersonic_qwen35_sigmoid_fast(acc);
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
        const float a_raw = supersonic_qwen35_to_float(a_beta_raw[head_base]);
        const float beta_raw = supersonic_qwen35_to_float(a_beta_raw[head_base + num_v_heads]);
        out[pair_out_base + 2 * head_k_dim + head_v_dim] = supersonic_qwen35_sigmoid_fast(beta_raw);
        const float bias = supersonic_qwen35_to_float(dt_bias[v_head]);
        const float decay = supersonic_qwen35_to_float(a_log_exp[v_head]);
        const float g = -supersonic_qwen35_softplus_fast(a_raw + bias) * decay;
        out[pair_out_base + 2 * head_k_dim + head_v_dim + 1] = supersonic_qwen35_exp_fast(g);
    }
    if (tid < head_v_dim) {
        const int value_channel = key_dim * 2 + v_head * head_v_dim + tid;
        out[pair_out_base + 2 * head_k_dim + tid] = conv_channel(value_channel);
    }
}

template <int MAX_K = 256>
__global__ void supersonic_qwen35_linear_decode_apply_kernel(
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
__device__ inline void supersonic_qwen35_delta_recurrent_prefill_impl(
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
    T* state_trace,
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
        state[k_idx] = supersonic_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < seq_len; ++t) {
        const float g_t = expf(supersonic_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * supersonic_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (supersonic_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            supersonic_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += supersonic_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        if (state_trace) {
            const int trace_base =
                ((bh * seq_len + t) * k_head_dim) * v_head_dim + v_idx;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                state_trace[trace_base + k_idx * v_head_dim] =
                    supersonic_qwen35_from_float<T>(state[k_idx]);
            }
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * supersonic_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + seq_len * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_recurrent_prefill_kernel(
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
    T* state_trace
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    supersonic_qwen35_delta_recurrent_prefill_impl(
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
        state_trace,
        tid
    );
}

__device__ inline float supersonic_qwen35_block_sum_128(float value, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = value;
    __syncthreads();
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) smem[tid] += smem[tid + 32];
    __syncthreads();
    if (tid < 16) smem[tid] += smem[tid + 16];
    __syncthreads();
    if (tid < 8) smem[tid] += smem[tid + 8];
    __syncthreads();
    if (tid < 4) smem[tid] += smem[tid + 4];
    __syncthreads();
    if (tid < 2) smem[tid] += smem[tid + 2];
    __syncthreads();
    if (tid < 1) smem[tid] += smem[tid + 1];
    __syncthreads();
    const float sum = smem[0];
    __syncthreads();
    return sum;
}

__device__ inline float supersonic_qwen35_block_sum_128_wave(float value, float* smem) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int wave = tid / warpSize;
    const int waves = (blockDim.x + warpSize - 1) / warpSize;

    value = supersonic_qwen35_wave_sum(value);
    if (lane == 0) {
        smem[wave] = value;
    }
    __syncthreads();

    float total = 0.0f;
    if (wave == 0) {
        total = lane < waves ? smem[lane] : 0.0f;
        total = supersonic_qwen35_wave_sum(total);
        if (lane == 0) {
            smem[0] = total;
        }
    }
    __syncthreads();
    return smem[0];
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_capture_f32_k128_legacy_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ out,
    float* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float state = initial_state[bh * state_stride + k_idx * V + v_idx];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * K + k_idx) * V + v_idx] = state;

        const float out_t =
            supersonic_qwen35_block_sum_128(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_capture_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ out,
    float* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float state = initial_state[bh * state_stride + k_idx * V + v_idx];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * K + k_idx) * V + v_idx] = state;

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_capture_bf16_trace_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ out,
    hip_bfloat16* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float state = initial_state[bh * state_stride + k_idx * V + v_idx];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * K + k_idx) * V + v_idx] =
            supersonic_qwen35_from_float<hip_bfloat16>(state);

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_capture_bf16_trace_transposed_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ out,
    hip_bfloat16* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float state = initial_state[bh * state_stride + k_idx * V + v_idx];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * V + v_idx) * K + k_idx] =
            supersonic_qwen35_from_float<hip_bfloat16>(state);

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = state;
}

__device__ __forceinline__ void qwen35_store_q8_0_block_scale(uint8_t* dst, half d_half) {
    union {
        half h;
        uint16_t u;
    } bits;
    bits.h = d_half;
    dst[0] = static_cast<uint8_t>(bits.u & 0xffu);
    dst[1] = static_cast<uint8_t>((bits.u >> 8) & 0xffu);
}

__device__ __forceinline__ half qwen35_load_q8_0_block_scale(const uint8_t* src) {
    union {
        uint16_t u;
        half h;
    } bits;
    bits.u = static_cast<uint16_t>(src[0]) |
        (static_cast<uint16_t>(src[1]) << 8);
    return bits.h;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_capture_q8_trace_transposed_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ out,
    uint8_t* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int q8_block = k_idx / QK8_0;
    const int q8_lane = k_idx - q8_block * QK8_0;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float state = initial_state[bh * state_stride + k_idx * V + v_idx];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;

        float amax = fabsf(state);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset));
        }
        const float d = amax / 127.0f;
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        const int q = static_cast<int>(roundf(state * id));
        const int row_base =
            (((bh * seq_len + t) * V + v_idx) * (K / QK8_0) + q8_block) * Q8_0_BLOCK_BYTES;
        if (q8_lane == 0) {
            qwen35_store_q8_0_block_scale(state_trace + row_base, __float2half(d));
        }
        state_trace[row_base + 2 + q8_lane] = static_cast<uint8_t>(static_cast<int8_t>(q));

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = state;
}

__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_delta_recurrent_prefill_capture_q8_trace_attn_bf16_transposed_f32_k128_warp32_kernel(
    int batch_heads,
    int seq_len,
    float* __restrict__ recurrent_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    hip_bfloat16* __restrict__ attn_output,
    uint8_t* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int lane = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int state_base = bh * state_stride + v_idx;
    const int query_key_base = bh * token_stride_k;
    const int value_base = bh * token_stride_v;
    const int scalar_base = bh * token_stride_s;

    float state0 = recurrent_state[state_base + (lane + 0 * QK8_0) * V];
    float state1 = recurrent_state[state_base + (lane + 1 * QK8_0) * V];
    float state2 = recurrent_state[state_base + (lane + 2 * QK8_0) * V];
    float state3 = recurrent_state[state_base + (lane + 3 * QK8_0) * V];

    for (int t = 0; t < seq_len; ++t) {
        const int key_row = query_key_base + t * K;
        const int value_row = value_base + t * V;
        const int beta_row = scalar_base + t;
        const float g_t = expf(g[beta_row]);
        const float beta_t = beta[beta_row];
        const float value_t = value[value_row + v_idx];
        const float k0 = key[key_row + lane + 0 * QK8_0];
        const float k1 = key[key_row + lane + 1 * QK8_0];
        const float k2 = key[key_row + lane + 2 * QK8_0];
        const float k3 = key[key_row + lane + 3 * QK8_0];

        state0 *= g_t;
        state1 *= g_t;
        state2 *= g_t;
        state3 *= g_t;

        const float kv_mem = __shfl(
            supersonic_qwen35_wave_sum(
                (state0 * k0 + state2 * k2) + (state1 * k1 + state3 * k3)),
            0);
        const float delta = (value_t - kv_mem) * beta_t;

        state0 += k0 * delta;
        state1 += k1 * delta;
        state2 += k2 * delta;
        state3 += k3 * delta;

        float amax0 = fabsf(state0);
        float amax1 = fabsf(state1);
        float amax2 = fabsf(state2);
        float amax3 = fabsf(state3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax0 = fmaxf(amax0, __shfl_xor(amax0, offset));
            amax1 = fmaxf(amax1, __shfl_xor(amax1, offset));
            amax2 = fmaxf(amax2, __shfl_xor(amax2, offset));
            amax3 = fmaxf(amax3, __shfl_xor(amax3, offset));
        }
        const float d0 = amax0 / 127.0f;
        const float d1 = amax1 / 127.0f;
        const float d2 = amax2 / 127.0f;
        const float d3 = amax3 / 127.0f;
        const float id0 = amax0 == 0.0f ? 0.0f : __builtin_amdgcn_rcpf(amax0) * 127.0f;
        const float id1 = amax1 == 0.0f ? 0.0f : __builtin_amdgcn_rcpf(amax1) * 127.0f;
        const float id2 = amax2 == 0.0f ? 0.0f : __builtin_amdgcn_rcpf(amax2) * 127.0f;
        const float id3 = amax3 == 0.0f ? 0.0f : __builtin_amdgcn_rcpf(amax3) * 127.0f;
        const int q0 = static_cast<int>(roundf(state0 * id0));
        const int q1 = static_cast<int>(roundf(state1 * id1));
        const int q2 = static_cast<int>(roundf(state2 * id2));
        const int q3 = static_cast<int>(roundf(state3 * id3));
        const int row_base0 =
            (((scalar_base + t) * V + v_idx) * (K / QK8_0)) * Q8_0_BLOCK_BYTES;
        const int row_base1 = row_base0 + Q8_0_BLOCK_BYTES;
        const int row_base2 = row_base1 + Q8_0_BLOCK_BYTES;
        const int row_base3 = row_base2 + Q8_0_BLOCK_BYTES;
        if (lane == 0) {
            qwen35_store_q8_0_block_scale(state_trace + row_base0, __float2half(d0));
            qwen35_store_q8_0_block_scale(state_trace + row_base1, __float2half(d1));
            qwen35_store_q8_0_block_scale(state_trace + row_base2, __float2half(d2));
            qwen35_store_q8_0_block_scale(state_trace + row_base3, __float2half(d3));
        }
        state_trace[row_base0 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q0));
        state_trace[row_base1 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q1));
        state_trace[row_base2 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q2));
        state_trace[row_base3 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q3));

        const float q0v = query[key_row + lane + 0 * QK8_0];
        const float q1v = query[key_row + lane + 1 * QK8_0];
        const float q2v = query[key_row + lane + 2 * QK8_0];
        const float q3v = query[key_row + lane + 3 * QK8_0];
        const float out = supersonic_qwen35_wave_sum(
            (state0 * q0v + state2 * q2v) + (state1 * q1v + state3 * q3v));
        if (lane == 0) {
            attn_output[(bh * seq_len + t) * V + v_idx] = hip_bfloat16(out);
        }
    }

    recurrent_state[state_base + (lane + 0 * QK8_0) * V] = state0;
    recurrent_state[state_base + (lane + 1 * QK8_0) * V] = state1;
    recurrent_state[state_base + (lane + 2 * QK8_0) * V] = state2;
    recurrent_state[state_base + (lane + 3 * QK8_0) * V] = state3;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_tree_prefill_capture_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const int* __restrict__ parent_ids,
    float* __restrict__ out,
    float* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float last_state = initial_state[bh * state_stride + k_idx * V + v_idx];
    for (int t = 0; t < seq_len; ++t) {
        const int parent = parent_ids[t];
        float state = parent < 0
            ? initial_state[bh * state_stride + k_idx * V + v_idx]
            : state_trace[((bh * seq_len + parent) * K + k_idx) * V + v_idx];

        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * K + k_idx) * V + v_idx] = state;
        last_state = state;

        const float out_t =
            supersonic_qwen35_block_sum_128(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = last_state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_tree_prefill_capture_bf16_trace_transposed_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const int* __restrict__ parent_ids,
    float* __restrict__ out,
    hip_bfloat16* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float last_state = initial_state[bh * state_stride + k_idx * V + v_idx];
    for (int t = 0; t < seq_len; ++t) {
        const int parent = parent_ids[t];
        float state = parent < 0
            ? initial_state[bh * state_stride + k_idx * V + v_idx]
            : static_cast<float>(state_trace[((bh * seq_len + parent) * V + v_idx) * K + k_idx]);

        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;
        state_trace[((bh * seq_len + t) * V + v_idx) * K + k_idx] =
            supersonic_qwen35_from_float<hip_bfloat16>(state);
        last_state = state;

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = last_state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_tree_prefill_capture_q8_trace_transposed_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const int* __restrict__ parent_ids,
    float* __restrict__ out,
    uint8_t* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int q8_block = k_idx / QK8_0;
    const int q8_lane = k_idx - q8_block * QK8_0;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + K) * V;

    float last_state = initial_state[bh * state_stride + k_idx * V + v_idx];
    for (int t = 0; t < seq_len; ++t) {
        const int parent = parent_ids[t];
        float state;
        if (parent < 0) {
            state = initial_state[bh * state_stride + k_idx * V + v_idx];
        } else {
            const int parent_base =
                (((bh * seq_len + parent) * V + v_idx) * (K / QK8_0) + q8_block) * Q8_0_BLOCK_BYTES;
            const half d_half = qwen35_load_q8_0_block_scale(state_trace + parent_base);
            const int8_t q = static_cast<int8_t>(state_trace[parent_base + 2 + q8_lane]);
            state = static_cast<float>(q) * __half2float(d_half);
        }

        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;

        float amax = fabsf(state);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset));
        }
        const float d = amax / 127.0f;
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        const int q = static_cast<int>(roundf(state * id));
        const int row_base =
            (((bh * seq_len + t) * V + v_idx) * (K / QK8_0) + q8_block) * Q8_0_BLOCK_BYTES;
        if (q8_lane == 0) {
            qwen35_store_q8_0_block_scale(state_trace + row_base, __float2half(d));
        }
        state_trace[row_base + 2 + q8_lane] = static_cast<uint8_t>(static_cast<int8_t>(q));
        last_state = state;

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            out[out_base + t * V + v_idx] = out_t;
        }
    }

    const int state_out = out_base + seq_len * V;
    out[state_out + k_idx * V + v_idx] = last_state;
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_tree_prefill_capture_q8_trace_attn_bf16_transposed_f32_k128_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const int* __restrict__ parent_ids,
    hip_bfloat16* __restrict__ attn_output,
    uint8_t* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int k_idx = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int q8_block = k_idx / QK8_0;
    const int q8_lane = k_idx - q8_block * QK8_0;

    __shared__ float reduce_smem[K];

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;

    for (int t = 0; t < seq_len; ++t) {
        const int parent = parent_ids[t];
        float state;
        if (parent < 0) {
            state = initial_state[bh * state_stride + k_idx * V + v_idx];
        } else {
            const int parent_base =
                (((bh * seq_len + parent) * V + v_idx) * (K / QK8_0) + q8_block) * Q8_0_BLOCK_BYTES;
            const half d_half = qwen35_load_q8_0_block_scale(state_trace + parent_base);
            const int8_t q = static_cast<int8_t>(state_trace[parent_base + 2 + q8_lane]);
            state = static_cast<float>(q) * __half2float(d_half);
        }

        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float k_t = key[key_row + k_idx];

        state *= g_t;
        const float kv_mem = supersonic_qwen35_block_sum_128_wave(state * k_t, reduce_smem);
        const float delta = (value[value_row + v_idx] - kv_mem) * beta[beta_row];

        state += k_t * delta;

        float amax = fabsf(state);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset));
        }
        const float d = amax / 127.0f;
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        const int q = static_cast<int>(roundf(state * id));
        const int row_base =
            (((bh * seq_len + t) * V + v_idx) * (K / QK8_0) + q8_block) * Q8_0_BLOCK_BYTES;
        if (q8_lane == 0) {
            qwen35_store_q8_0_block_scale(state_trace + row_base, __float2half(d));
        }
        state_trace[row_base + 2 + q8_lane] = static_cast<uint8_t>(static_cast<int8_t>(q));

        const float out_t =
            supersonic_qwen35_block_sum_128_wave(state * query[key_row + k_idx], reduce_smem);
        if (k_idx == 0) {
            attn_output[(bh * seq_len + t) * V + v_idx] = hip_bfloat16(out_t);
        }
    }
}

__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_delta_recurrent_tree_prefill_capture_q8_trace_attn_bf16_transposed_f32_k128_warp32_kernel(
    int batch_heads,
    int seq_len,
    const float* __restrict__ initial_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const int* __restrict__ parent_ids,
    hip_bfloat16* __restrict__ attn_output,
    uint8_t* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int lane = threadIdx.x;
    const int v_idx = blockIdx.x;
    const int bh = blockIdx.y;

    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int token_stride_s = seq_len;
    const int state_base = bh * state_stride + v_idx;

    for (int t = 0; t < seq_len; ++t) {
        const int parent = parent_ids[t];
        float state0;
        float state1;
        float state2;
        float state3;
        if (parent < 0) {
            state0 = initial_state[state_base + (lane + 0 * QK8_0) * V];
            state1 = initial_state[state_base + (lane + 1 * QK8_0) * V];
            state2 = initial_state[state_base + (lane + 2 * QK8_0) * V];
            state3 = initial_state[state_base + (lane + 3 * QK8_0) * V];
        } else {
            const int parent_row = (bh * seq_len + parent) * V + v_idx;
            const int base0 = (parent_row * (K / QK8_0) + 0) * Q8_0_BLOCK_BYTES;
            const int base1 = (parent_row * (K / QK8_0) + 1) * Q8_0_BLOCK_BYTES;
            const int base2 = (parent_row * (K / QK8_0) + 2) * Q8_0_BLOCK_BYTES;
            const int base3 = (parent_row * (K / QK8_0) + 3) * Q8_0_BLOCK_BYTES;
            const float d0 = __half2float(qwen35_load_q8_0_block_scale(state_trace + base0));
            const float d1 = __half2float(qwen35_load_q8_0_block_scale(state_trace + base1));
            const float d2 = __half2float(qwen35_load_q8_0_block_scale(state_trace + base2));
            const float d3 = __half2float(qwen35_load_q8_0_block_scale(state_trace + base3));
            state0 = static_cast<float>(static_cast<int8_t>(state_trace[base0 + 2 + lane])) * d0;
            state1 = static_cast<float>(static_cast<int8_t>(state_trace[base1 + 2 + lane])) * d1;
            state2 = static_cast<float>(static_cast<int8_t>(state_trace[base2 + 2 + lane])) * d2;
            state3 = static_cast<float>(static_cast<int8_t>(state_trace[base3 + 2 + lane])) * d3;
        }

        const int key_row = bh * token_stride_k + t * K;
        const int value_row = bh * token_stride_v + t * V;
        const int beta_row = bh * token_stride_s + t;
        const float g_t = expf(g[beta_row]);
        const float beta_t = beta[beta_row];
        const float value_t = value[value_row + v_idx];
        const float k0 = key[key_row + lane + 0 * QK8_0];
        const float k1 = key[key_row + lane + 1 * QK8_0];
        const float k2 = key[key_row + lane + 2 * QK8_0];
        const float k3 = key[key_row + lane + 3 * QK8_0];

        state0 *= g_t;
        state1 *= g_t;
        state2 *= g_t;
        state3 *= g_t;

        const float kv_mem = __shfl(
            supersonic_qwen35_wave_sum(
                (state0 * k0 + state2 * k2) + (state1 * k1 + state3 * k3)),
            0);
        const float delta = (value_t - kv_mem) * beta_t;

        state0 += k0 * delta;
        state1 += k1 * delta;
        state2 += k2 * delta;
        state3 += k3 * delta;

        float amax0 = fabsf(state0);
        float amax1 = fabsf(state1);
        float amax2 = fabsf(state2);
        float amax3 = fabsf(state3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax0 = fmaxf(amax0, __shfl_xor(amax0, offset));
            amax1 = fmaxf(amax1, __shfl_xor(amax1, offset));
            amax2 = fmaxf(amax2, __shfl_xor(amax2, offset));
            amax3 = fmaxf(amax3, __shfl_xor(amax3, offset));
        }
        const float d0 = amax0 / 127.0f;
        const float d1 = amax1 / 127.0f;
        const float d2 = amax2 / 127.0f;
        const float d3 = amax3 / 127.0f;
        const float id0 = d0 == 0.0f ? 0.0f : 1.0f / d0;
        const float id1 = d1 == 0.0f ? 0.0f : 1.0f / d1;
        const float id2 = d2 == 0.0f ? 0.0f : 1.0f / d2;
        const float id3 = d3 == 0.0f ? 0.0f : 1.0f / d3;
        const int q0 = static_cast<int>(roundf(state0 * id0));
        const int q1 = static_cast<int>(roundf(state1 * id1));
        const int q2 = static_cast<int>(roundf(state2 * id2));
        const int q3 = static_cast<int>(roundf(state3 * id3));
        const int trace_row = (bh * seq_len + t) * V + v_idx;
        const int row_base0 = (trace_row * (K / QK8_0) + 0) * Q8_0_BLOCK_BYTES;
        const int row_base1 = (trace_row * (K / QK8_0) + 1) * Q8_0_BLOCK_BYTES;
        const int row_base2 = (trace_row * (K / QK8_0) + 2) * Q8_0_BLOCK_BYTES;
        const int row_base3 = (trace_row * (K / QK8_0) + 3) * Q8_0_BLOCK_BYTES;
        if (lane == 0) {
            qwen35_store_q8_0_block_scale(state_trace + row_base0, __float2half(d0));
            qwen35_store_q8_0_block_scale(state_trace + row_base1, __float2half(d1));
            qwen35_store_q8_0_block_scale(state_trace + row_base2, __float2half(d2));
            qwen35_store_q8_0_block_scale(state_trace + row_base3, __float2half(d3));
        }
        state_trace[row_base0 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q0));
        state_trace[row_base1 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q1));
        state_trace[row_base2 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q2));
        state_trace[row_base3 + 2 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q3));

        const float q0v = query[key_row + lane + 0 * QK8_0];
        const float q1v = query[key_row + lane + 1 * QK8_0];
        const float q2v = query[key_row + lane + 2 * QK8_0];
        const float q3v = query[key_row + lane + 3 * QK8_0];
        const float out = supersonic_qwen35_wave_sum(
            (state0 * q0v + state2 * q2v) + (state1 * q1v + state3 * q3v));
        if (lane == 0) {
            attn_output[(bh * seq_len + t) * V + v_idx] = hip_bfloat16(out);
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_rollback_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int chunk_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    T* conv_state,
    const float* recurrent_trace,
    float* recurrent_state
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        conv_state[idx] = conv_input[ch * conv_input_len + commit_len + tap];
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int accepted_idx = commit_len - 1;
        recurrent_state[idx] =
            recurrent_trace[(h * chunk_len + accepted_idx) * rec_head_elems + rem];
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_rollback_bf16_trace_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int chunk_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    T* conv_state,
    const hip_bfloat16* recurrent_trace,
    float* recurrent_state
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        conv_state[idx] = conv_input[ch * conv_input_len + commit_len + tap];
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int accepted_idx = commit_len - 1;
        recurrent_state[idx] = static_cast<float>(
            recurrent_trace[(h * chunk_len + accepted_idx) * rec_head_elems + rem]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_rollback_bf16_trace_transposed_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int chunk_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    T* conv_state,
    const hip_bfloat16* recurrent_trace,
    float* recurrent_state
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        conv_state[idx] = conv_input[ch * conv_input_len + commit_len + tap];
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int k = rem / head_v_dim;
        const int v = rem - k * head_v_dim;
        const int accepted_idx = commit_len - 1;
        recurrent_state[idx] = static_cast<float>(
            recurrent_trace[((h * chunk_len + accepted_idx) * head_v_dim + v) * head_k_dim + k]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_rollback_q8_trace_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int chunk_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    T* conv_state,
    const uint8_t* recurrent_trace,
    float* recurrent_state
) {
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        conv_state[idx] = conv_input[ch * conv_input_len + commit_len + tap];
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int k = rem / head_v_dim;
        const int v = rem - k * head_v_dim;
        const int accepted_idx = commit_len - 1;
        const int q8_block = k / QK8_0;
        const int q8_lane = k - q8_block * QK8_0;
        const int row_base =
            (((h * chunk_len + accepted_idx) * head_v_dim + v) * (head_k_dim / QK8_0)
             + q8_block) * Q8_0_BLOCK_BYTES;
        const half d_half = qwen35_load_q8_0_block_scale(recurrent_trace + row_base);
        const int8_t q = static_cast<int8_t>(recurrent_trace[row_base + 2 + q8_lane]);
        recurrent_state[idx] = static_cast<float>(q) * __half2float(d_half);
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_tree_rollback_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int tree_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    const uint32_t* accepted_indices,
    T* conv_state,
    const float* recurrent_trace,
    float* recurrent_state
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        const int logical = commit_len + tap - conv_state_len;
        int src_col = commit_len + tap;
        if (logical >= 0) {
            const uint32_t node = accepted_indices[logical];
            src_col = conv_state_len + static_cast<int>(node);
        }
        if (src_col >= 0 && src_col < conv_input_len) {
            conv_state[idx] = conv_input[ch * conv_input_len + src_col];
        }
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const uint32_t accepted_idx = accepted_indices[commit_len - 1];
        if (accepted_idx < static_cast<uint32_t>(tree_len)) {
            recurrent_state[idx] =
                recurrent_trace[(h * tree_len + static_cast<int>(accepted_idx)) * rec_head_elems + rem];
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_tree_rollback_bf16_trace_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int tree_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    const uint32_t* accepted_indices,
    T* conv_state,
    const hip_bfloat16* recurrent_trace,
    float* recurrent_state
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        const int logical = commit_len + tap - conv_state_len;
        int src_col = commit_len + tap;
        if (logical >= 0) {
            const uint32_t node = accepted_indices[logical];
            src_col = conv_state_len + static_cast<int>(node);
        }
        if (src_col >= 0 && src_col < conv_input_len) {
            conv_state[idx] = conv_input[ch * conv_input_len + src_col];
        }
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int k = rem / head_v_dim;
        const int v = rem - k * head_v_dim;
        const uint32_t accepted_idx = accepted_indices[commit_len - 1];
        if (accepted_idx < static_cast<uint32_t>(tree_len)) {
            recurrent_state[idx] = static_cast<float>(
                recurrent_trace[((h * tree_len + static_cast<int>(accepted_idx)) * head_v_dim + v) * head_k_dim + k]);
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_dflash_apply_tree_rollback_q8_trace_kernel(
    int qkv_dim,
    int conv_state_len,
    int conv_input_len,
    int tree_len,
    int commit_len,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const T* conv_input,
    const uint32_t* accepted_indices,
    T* conv_state,
    const uint8_t* recurrent_trace,
    float* recurrent_state
) {
    constexpr int QK8_0 = 32;
    constexpr int Q8_0_BLOCK_BYTES = 34;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_total = qkv_dim * conv_state_len;
    const int rec_head_elems = head_k_dim * head_v_dim;
    const int rec_total = num_v_heads * rec_head_elems;

    if (idx < conv_total) {
        const int ch = idx / conv_state_len;
        const int tap = idx - ch * conv_state_len;
        const int logical = commit_len + tap - conv_state_len;
        int src_col = commit_len + tap;
        if (logical >= 0) {
            const uint32_t node = accepted_indices[logical];
            src_col = conv_state_len + static_cast<int>(node);
        }
        if (src_col >= 0 && src_col < conv_input_len) {
            conv_state[idx] = conv_input[ch * conv_input_len + src_col];
        }
    }

    if (idx < rec_total) {
        const int h = idx / rec_head_elems;
        const int rem = idx - h * rec_head_elems;
        const int k = rem / head_v_dim;
        const int v = rem - k * head_v_dim;
        const uint32_t accepted_idx = accepted_indices[commit_len - 1];
        if (accepted_idx < static_cast<uint32_t>(tree_len)) {
            const int q8_block = k / QK8_0;
            const int q8_lane = k - q8_block * QK8_0;
            const int row_base =
                (((h * tree_len + static_cast<int>(accepted_idx)) * head_v_dim + v) * (head_k_dim / QK8_0)
                 + q8_block) * Q8_0_BLOCK_BYTES;
            const half d_half = qwen35_load_q8_0_block_scale(recurrent_trace + row_base);
            const int8_t q = static_cast<int8_t>(recurrent_trace[row_base + 2 + q8_lane]);
            recurrent_state[idx] = static_cast<float>(q) * __half2float(d_half);
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_fill_conv_tail_kernel(
    int qkv_dim,
    int pad,
    int total_len,
    const T* tail,
    T* conv_input
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = qkv_dim * pad;
    if (idx >= total) return;
    const int ch = idx / pad;
    const int tap = idx - ch * pad;
    conv_input[ch * total_len + tap] = tail[idx];
}

__global__ void supersonic_qwen35_dflash_extract_recurrent_attn_kernel(
    int num_v_heads,
    int chunk_len,
    int head_k_dim,
    int head_v_dim,
    const float* __restrict__ delta_out,
    float* __restrict__ recurrent_state,
    hip_bfloat16* __restrict__ attn_output
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int out_rows = chunk_len + head_k_dim;
    const int attn_total = num_v_heads * chunk_len * head_v_dim;
    const int state_total = num_v_heads * head_k_dim * head_v_dim;
    const int head_attn_elems = chunk_len * head_v_dim;
    const int head_state_elems = head_k_dim * head_v_dim;
    const int delta_head_stride = out_rows * head_v_dim;

    if (idx < attn_total) {
        const int h = idx / head_attn_elems;
        const int rem = idx - h * head_attn_elems;
        const int t = rem / head_v_dim;
        const int v = rem - t * head_v_dim;
        const int src = h * delta_head_stride + t * head_v_dim + v;
        attn_output[idx] = hip_bfloat16(delta_out[src]);
    }

    if (idx < state_total) {
        const int h = idx / head_state_elems;
        const int rem = idx - h * head_state_elems;
        const int k = rem / head_v_dim;
        const int v = rem - k * head_v_dim;
        const int src = h * delta_head_stride + (chunk_len + k) * head_v_dim + v;
        recurrent_state[idx] = delta_out[src];
    }
}

template <typename T, int MAX_K = 256>
__device__ inline void supersonic_qwen35_delta_chunk_step_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(
            prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < chunk_size; ++t) {
        const float g_t = expf(supersonic_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * supersonic_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (supersonic_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            supersonic_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += supersonic_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * supersonic_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_chunk_step_kernel(
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
    supersonic_qwen35_delta_chunk_step_impl(
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
__device__ inline void supersonic_qwen35_delta_chunk_scan_raw_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < total_tokens; ++t) {
        const float g_t = expf(supersonic_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * supersonic_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (supersonic_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            supersonic_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += supersonic_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * supersonic_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + total_tokens * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_chunk_scan_raw_kernel(
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
    supersonic_qwen35_delta_chunk_scan_raw_impl(
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
__device__ inline void supersonic_qwen35_delta_state_scan_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(initial_state[idx]);
        out[idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
        const int value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
        const float state_decay =
            supersonic_qwen35_to_float(packed_scan[packed_chunk_base + 2 * k_head_dim]);
        float update[MAX_K];
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            update[k_idx] = 0.0f;
        }

        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_chunk_base + t * packed_width;
            const int value_row = value_chunk_base + t * v_head_dim;
            float v_prime = 0.0f;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += supersonic_qwen35_to_float(packed_scan[packed_row + k_head_dim + k_idx]) *
                    state[k_idx];
            }
            const float v_new = supersonic_qwen35_to_float(value[value_row + v_idx]) - v_prime;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                update[k_idx] += supersonic_qwen35_to_float(packed_scan[packed_row + k_idx]) * v_new;
            }
        }

        const int out_chunk_base = ((bh * (num_chunks + 1)) + (chunk + 1)) * state_stride;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] = state_decay * state[k_idx] + update[k_idx];
            out[out_chunk_base + k_idx * v_head_dim + v_idx] =
                supersonic_qwen35_from_float<T>(state[k_idx]);
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_state_scan_kernel(
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
    supersonic_qwen35_delta_state_scan_impl(
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
__device__ inline void supersonic_qwen35_delta_chunk_fused_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(
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
                supersonic_qwen35_to_float(packed_chunk[packed_row + k_head_dim + k_idx]) *
                state[k_idx];
            attn +=
                supersonic_qwen35_to_float(packed_chunk[packed_row + 2 * k_head_dim + k_idx]) *
                state[k_idx];
        }
        v_new[t] = supersonic_qwen35_to_float(value[value_base + t * v_head_dim + v_idx]) - v_prime;
        attn_inter[t] = attn;
        out[out_base + t * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(v_new[t]);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] =
            supersonic_qwen35_from_float<T>(attn_inter[t]);
    }

    const float state_decay = supersonic_qwen35_to_float(packed_chunk[packed_base + 3 * k_head_dim]);
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float update = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_base + t * packed_width;
            update += supersonic_qwen35_to_float(packed_chunk[packed_row + k_idx]) * v_new[t];
        }
        out[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
            supersonic_qwen35_from_float<T>(state_decay * state[k_idx] + update);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_chunk_fused_kernel(
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
    supersonic_qwen35_delta_chunk_fused_impl(
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
__device__ inline void supersonic_qwen35_delta_full_scan_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(
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
                v_prime += supersonic_qwen35_to_float(k_cumdecay_scan[row + k_idx]) * state[k_idx];
                attn += supersonic_qwen35_to_float(q_state_scan[row + k_idx]) * state[k_idx];
            }
            v_new[t] = supersonic_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) -
                v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += supersonic_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                supersonic_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = supersonic_qwen35_to_float(state_decay_scan[decay_base + chunk]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_scan + t * k_head_dim;
                update += supersonic_qwen35_to_float(weighted_key_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_local_attn_scan_flat_kernel(
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
        out[tid] = supersonic_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    const float dot = supersonic_qwen35_dot_row(
        query_scan + row_base,
        key_scan + col_base,
        k_head_dim);

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = supersonic_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = supersonic_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = supersonic_qwen35_from_float<T>(dot * decay);
}

template <typename T>
__global__ void supersonic_qwen35_delta_local_attn_scan_row_kernel(
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
    const float exp_g_t = supersonic_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const int out_base = (bh * num_chunks + chunk) * chunk_size * chunk_size;

    for (int col = lane; col < chunk_size; col += blockDim.x) {
        const int out_idx = out_base + row_in_chunk * chunk_size + col;
        if (col > row_in_chunk) {
            out[out_idx] = supersonic_qwen35_from_float<T>(0.0f);
            continue;
        }

        const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
        const float dot = supersonic_qwen35_dot_row(
            query_row,
            key_scan + col_base,
            k_head_dim);

        const float exp_g_s = supersonic_qwen35_to_float(exp_g_scan[exp_base + col]);
        const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
        out[out_idx] = supersonic_qwen35_from_float<T>(dot * decay);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_base_attn_scan_kernel(
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
        out[tid] = supersonic_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    float dot = 0.0f;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        dot += supersonic_qwen35_to_float(k_beta_scan[row_base + k_idx]) *
               supersonic_qwen35_to_float(key_scan[col_base + k_idx]);
    }

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = supersonic_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = supersonic_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = supersonic_qwen35_from_float<T>(-dot * decay);
}

template <typename T, int MAX_CHUNK = 64>
__global__ void supersonic_qwen35_delta_attn_solve_scan_kernel(
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
            const float row_val = supersonic_qwen35_to_float(base_attn_scan[base + i * chunk_size + j]);
            float correction = 0.0f;
            for (int k = 0; k < i; ++k) {
                correction += supersonic_qwen35_to_float(base_attn_scan[base + i * chunk_size + k]) *
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
            out[base + i * chunk_size + j] = supersonic_qwen35_from_float<T>(value);
        }
    }
}

template <typename T, int MAX_CHUNK = 64>
__global__ void supersonic_qwen35_delta_attn_solve_from_inputs_kernel(
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
        const float exp_g_t = supersonic_qwen35_to_float(exp_g_scan[chunk_base + i]);
        for (int k = 0; k < i; ++k) {
            const int row_base = qk_base + i * k_head_dim;
            const int col_base = qk_base + k * k_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < k_head_dim; ++d) {
                dot += supersonic_qwen35_to_float(k_beta_scan[row_base + d]) *
                    supersonic_qwen35_to_float(key_scan[col_base + d]);
            }
            const float exp_g_s = supersonic_qwen35_to_float(exp_g_scan[chunk_base + k]);
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
            out[out_base + i * chunk_size + j] = supersonic_qwen35_from_float<T>(value);
        }
    }
}

template <typename T>
__global__ void supersonic_qwen35_swiglu_mul_kernel(
    int elem_count,
    const T* gate,
    const T* up,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= elem_count) {
        return;
    }
    const float gate_x = supersonic_qwen35_to_float(gate[idx]);
    const float up_x = supersonic_qwen35_to_float(up[idx]);
    const float silu = gate_x / (1.0f + expf(-gate_x));
    out[idx] = supersonic_qwen35_from_float<T>(silu * up_x);
}

template <typename T>
__global__ void supersonic_qwen35_swiglu_mul_split_kernel(
    int rows,
    int cols,
    const T* gate_up,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int elem_count = rows * cols;
    if (idx >= elem_count) {
        return;
    }
    const int row = idx / cols;
    const int col = idx - row * cols;
    const T* row_base = gate_up + static_cast<size_t>(row) * cols * 2;
    const float gate_x = supersonic_qwen35_to_float(row_base[col]);
    const float up_x = supersonic_qwen35_to_float(row_base[cols + col]);
    const float silu = gate_x / (1.0f + expf(-gate_x));
    out[idx] = supersonic_qwen35_from_float<T>(silu * up_x);
}

template <typename T, typename IndexT>
__global__ void supersonic_qwen35_embedding_lookup_kernel(
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
        out[idx] = supersonic_qwen35_from_float<T>(0.0f);
        return;
    }
    out[idx] = embeddings[row * hidden_size + col];
}

template <typename T>
__global__ void supersonic_qwen35_output_projection_lookup_kernel(
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
        acc += supersonic_qwen35_to_float(row_hidden[col]) * supersonic_qwen35_to_float(row_weight[col]);
    }
    out[idx] = supersonic_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void supersonic_qwen35_causal_mask_kernel(
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
    out[idx] = supersonic_qwen35_from_float<T>(allowed ? 0.0f : -INFINITY);
}

template <typename T>
__global__ void supersonic_qwen35_cumsum_last_dim_kernel(
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
        acc += supersonic_qwen35_to_float(xs[row_offset + col]);
        out[row_offset + col] = supersonic_qwen35_from_float<T>(acc);
    }
}

template <typename T>
__global__ void supersonic_qwen35_exp_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(expf(supersonic_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void supersonic_qwen35_recip_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(1.0f / supersonic_qwen35_to_float(xs[idx]));
}

template <typename T>
__global__ void supersonic_qwen35_sigmoid_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(
        supersonic_qwen35_sigmoid_fast(supersonic_qwen35_to_float(xs[idx]))
    );
}

template <typename T>
__global__ void supersonic_qwen35_log_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(logf(supersonic_qwen35_to_float(xs[idx])));
}

template <typename In, typename Out>
__global__ void supersonic_qwen35_cast_kernel(
    int total_elems,
    const In* xs,
    Out* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<Out>(supersonic_qwen35_to_float(xs[idx]));
}

__device__ __forceinline__ float supersonic_qwen35_unary_op_apply(
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
        return supersonic_qwen35_sigmoid_fast(x);
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
__global__ void supersonic_qwen35_unary_view_kernel(
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
    const float x = supersonic_qwen35_to_float(xs[in_offset]);
    out[idx] = supersonic_qwen35_from_float<T>(supersonic_qwen35_unary_op_apply(op, x, scalar));
}

template <typename In, typename Out>
__global__ void supersonic_qwen35_cast_view_kernel(
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
    out[idx] = supersonic_qwen35_from_float<Out>(supersonic_qwen35_to_float(xs[in_offset]));
}

template <typename T>
__global__ void supersonic_qwen35_binary_broadcast_kernel(
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

    const float lhs_f = supersonic_qwen35_to_float(lhs[lhs_offset]);
    const float rhs_f = supersonic_qwen35_to_float(rhs[rhs_offset]);
    out[idx] = supersonic_qwen35_from_float<T>(supersonic_qwen35_binary_op_apply(op, lhs_f, rhs_f));
}

template <typename T>
__global__ void supersonic_qwen35_mul_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(supersonic_qwen35_to_float(xs[idx]) * scalar);
}

template <typename T>
__global__ void supersonic_qwen35_add_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(supersonic_qwen35_to_float(xs[idx]) + scalar);
}

template <typename T>
__global__ void supersonic_qwen35_sqrt_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = supersonic_qwen35_from_float<T>(sqrtf(supersonic_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void supersonic_qwen35_reduce_keepdim_kernel(
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
    float acc = supersonic_qwen35_to_float(xs[base]);
    for (int r = 1; r < reduce; ++r) {
        const float value = supersonic_qwen35_to_float(xs[base + r * inner]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = supersonic_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void supersonic_qwen35_reduce_keepdim_view_kernel(
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
    float acc = supersonic_qwen35_to_float(xs[base_offset]);
    for (size_t r = 1; r < reduce_len; ++r) {
        const float value = supersonic_qwen35_to_float(
            xs[base_offset + r * static_cast<size_t>(in_strides[reduce_dim])]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = supersonic_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void supersonic_qwen35_batched_matmul_kernel(
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

    const size_t lhs_batch_idx = supersonic_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, lhs_batch_dims);
    const size_t rhs_batch_idx = supersonic_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, rhs_batch_dims);

    const size_t lhs_base = lhs_batch_idx * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t rhs_base = rhs_batch_idx * static_cast<size_t>(k) * static_cast<size_t>(n);
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += supersonic_qwen35_to_float(lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(kk)]) *
               supersonic_qwen35_to_float(rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
    }
    out[idx] = supersonic_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void supersonic_qwen35_batched_matmul_view_kernel(
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
        acc += supersonic_qwen35_to_float(
                   lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(lhs_row_stride) +
                       static_cast<size_t>(kk) * static_cast<size_t>(lhs_k_stride)]) *
               supersonic_qwen35_to_float(
                   rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(rhs_k_stride) +
                       static_cast<size_t>(col) * static_cast<size_t>(rhs_col_stride)]);
    }
    out[idx] = supersonic_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void supersonic_qwen35_delta_full_scan_pack_kernel(
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

    const float exp_g_t = supersonic_qwen35_to_float(exp_g_scan[row]);
    const float exp_g_last = supersonic_qwen35_to_float(exp_g_scan[bh * num_chunks * chunk_size + chunk * chunk_size + (chunk_size - 1)]);
    const float chunk_decay = exp_g_t != 0.0f ? (exp_g_last / exp_g_t) : 0.0f;

    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[packed_row + k_idx] = supersonic_qwen35_from_float<T>(
            supersonic_qwen35_to_float(key_scan[scan_row + k_idx]) * chunk_decay
        );
        out[packed_row + k_head_dim + k_idx] = k_cumdecay_scan[scan_row + k_idx];
        out[packed_row + 2 * k_head_dim + k_idx] = supersonic_qwen35_from_float<T>(
            supersonic_qwen35_to_float(query_scan[scan_row + k_idx]) * exp_g_t
        );
    }
    out[packed_row + 3 * k_head_dim] = supersonic_qwen35_from_float<T>(exp_g_last);
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void supersonic_qwen35_delta_full_scan_packed_impl(
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
        state[k_idx] = supersonic_qwen35_to_float(
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
                v_prime += supersonic_qwen35_to_float(packed_scan[row + k_head_dim + k_idx]) * state[k_idx];
                attn += supersonic_qwen35_to_float(packed_scan[row + 2 * k_head_dim + k_idx]) * state[k_idx];
            }
            v_new[t] = supersonic_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) - v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += supersonic_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                supersonic_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = supersonic_qwen35_to_float(packed_scan[chunk_packed + 3 * k_head_dim]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_packed + t * packed_width;
                update += supersonic_qwen35_to_float(packed_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_full_scan_packed_kernel(
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
    supersonic_qwen35_delta_full_scan_packed_impl(
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
__global__ void supersonic_qwen35_delta_full_scan_kernel(
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
    supersonic_qwen35_delta_full_scan_impl(
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
__device__ inline void supersonic_qwen35_delta_chunk_single_prefill_impl(
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
        raw_g[t] = supersonic_qwen35_to_float(g_raw[bh * token_stride_s + t]);
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
                dot += supersonic_qwen35_to_float(query[row_i_k + k_idx]) *
                    supersonic_qwen35_to_float(key[row_j_k + k_idx]);
            }
            const float local = dot * expf(prefix_g[i] - prefix_g[j]);
            out_i += local * supersonic_qwen35_to_float(value[bh * token_stride_v + j * v_head_dim + v_idx]);
        }
        out[out_base + i * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(out_i);
    }

    const float g_last = raw_g[chunk_size - 1];
    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float state = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const float beta_t = supersonic_qwen35_to_float(beta[bh * token_stride_s + t]);
            state += supersonic_qwen35_to_float(key[bh * token_stride_k + t * k_head_dim + k_idx]) *
                expf(g_last - raw_g[t]) *
                supersonic_qwen35_to_float(value[bh * token_stride_v + t * v_head_dim + v_idx]);
        }
        out[state_out + k_idx * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(state);
    }
}

template <typename T>
__global__ void supersonic_qwen35_delta_chunk_single_prefill_kernel(
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
    supersonic_qwen35_delta_chunk_single_prefill_impl(
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
__global__ void supersonic_qwen35_l2norm_kernel(
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
        const float x = supersonic_qwen35_to_float(row_in[col]);
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
        row_out[col] = supersonic_qwen35_from_float<T>(
            supersonic_qwen35_to_float(row_in[col]) * shared_inv_norm
        );
    }
}

template <typename T>
__global__ void supersonic_qwen35_value_decay_kernel(
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
    const float a_val = supersonic_qwen35_to_float(a[idx]);
    const float bias = supersonic_qwen35_to_float(dt_bias[head]);
    const float decay = supersonic_qwen35_to_float(a_log_exp[head]);
    const float softplus = supersonic_qwen35_softplus_fast(a_val + bias);
    out[idx] = supersonic_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, bool ADD_UNIT_OFFSET>
__global__ void supersonic_qwen35_rms_norm_kernel(
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
        const float x = supersonic_qwen35_to_float(row_in[col]);
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
        const float weight_val = supersonic_qwen35_to_float(weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        const float x = supersonic_qwen35_to_float(row_in[col]);
        row_out[col] = supersonic_qwen35_from_float<T>(x * shared_inv_rms * weight_val);
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
__global__ void supersonic_qwen35_fused_rms_norm_linear_kernel(
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
        const float x = supersonic_qwen35_to_float(hidden[col]);
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
        const float x = supersonic_qwen35_to_float(hidden[col]);
        const float w = supersonic_qwen35_to_float(norm_weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        shared_normed[col] = x * inv_rms * w;
    }
    __syncthreads();

    // Phase 2: Dot product of proj_weight[out_row, :] with shared_normed
    const T* w_row = proj_weight + static_cast<size_t>(out_row) * static_cast<size_t>(hidden_dim);

    float partial_dot = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        partial_dot += supersonic_qwen35_to_float(w_row[col]) * shared_normed[col];
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
        out[out_row] = supersonic_qwen35_from_float<T>(shared_scratch[0]);
    }
}

template <typename T>
__global__ void supersonic_qwen35_rms_norm_gated_kernel(
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
        const float x = supersonic_qwen35_to_float(row_hidden[col]);
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
        const float x = supersonic_qwen35_to_float(row_hidden[col]);
        const float gate_x = supersonic_qwen35_to_float(row_gate[col]);
    const float gate_silu = gate_x * supersonic_qwen35_sigmoid_fast(gate_x);
        const float weight_val = supersonic_qwen35_to_float(weight[col]);
        row_out[col] =
            supersonic_qwen35_from_float<T>(x * shared_inv_rms * weight_val * gate_silu);
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
__device__ inline float supersonic_qwen35_wave_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__global__ void supersonic_qwen35_mlp_decode_megakernel(
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
        shared_hidden[col] = supersonic_qwen35_to_float(hidden_in[col]);
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
        const float w = supersonic_qwen35_to_float(norm_weight[col]) + 1.0f;  // add_unit_offset
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
            partial += supersonic_qwen35_to_float(w_row[col]) * shared_normed[col];
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
__global__ void supersonic_qwen35_mlp_swiglu_kernel(
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
__global__ void supersonic_qwen35_mlp_down_proj_kernel(
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
            partial += supersonic_qwen35_to_float(w_row[col]) * swiglu_scratch[col];
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
            hidden_out[my_row] = supersonic_qwen35_from_float<T>(shared_scratch[0]);
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
__global__ void supersonic_qwen35_standalone_matvec_kernel(
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
    __shared__ float shared_input[8192];  // max in_dim we support
    for (int col = tid; col < in_dim; col += block_size) {
        shared_input[col] = supersonic_qwen35_to_float(input[col]);
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
            partial += supersonic_qwen35_to_float(w_row[col]) * shared_input[col];
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
            output[my_row] = supersonic_qwen35_from_float<T>(shared_scratch[0]);
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
__global__ void supersonic_qwen35_norm_multi_proj_kernel(
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
        shared_hidden[col] = supersonic_qwen35_to_float(hidden_in[col]);
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
        const float w = supersonic_qwen35_to_float(norm_weight[col]) + 1.0f;
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
            partial += supersonic_qwen35_to_float(w_row[col]) * shared_normed[col];
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
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __threadfence();
            __atomic_store_n(barrier_flag, phase + 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
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
        dst[c] = src[c] * inv * (supersonic_qwen35_to_float(weight[c]) + 1.0f);
    }
    __syncthreads();
}
