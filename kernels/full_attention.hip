#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Weight descriptor for the persistent decode megakernel.
// One struct per decoder layer (64 total for canonical Qwen3.8-27B).
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
    constexpr int QPW = 1;

    const int qr_base = blockIdx.x * BM;
    const int qh      = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_h    = qh / num_kv_groups;
    const int warp_y  = threadIdx.y;        // 0..BM-1
    const int lane    = threadIdx.x;        // 0..warpSize-1
    const int qr      = qr_base + warp_y;
    const int qr1     = qr_base + BM + warp_y;
    if (lane >= warpSize) return;
    if (batch >= batch_size || qh >= q_heads) return;

    extern __shared__ unsigned char smem_raw[];
    T* k_tile = reinterpret_cast<T*>(smem_raw);
    T* v_tile = k_tile + BK * head_dim;

    const bool active = (qr < q_len);
    const bool active1 = (QPW == 2) && (qr1 < q_len);
    const int causal_limit = active ? min(kv_len, seqlen_offset + qr + 1) : 0;
    const int causal_limit1 = active1 ? min(kv_len, seqlen_offset + qr1 + 1) : 0;

    const T* q_row = active ? (query + (((batch * q_heads + qh) * q_len + qr) * head_dim)) : nullptr;
    const T* q_row1 = active1 ? (query + (((batch * q_heads + qh) * q_len + qr1) * head_dim)) : nullptr;
    const T* k_head_ptr = key   + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    const T* v_head_ptr = value + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    float*   out_row = active ? (out + (((batch * q_heads + qh) * q_len + qr) * head_dim)) : nullptr;
    float*   out_row1 = active1 ? (out + (((batch * q_heads + qh) * q_len + qr1) * head_dim)) : nullptr;

    // Per-lane output accumulators (in registers). Each lane owns the dims
    // d = lane, lane + warpSize, lane + 2*warpSize, ... (up to ACC_MAX of them).
    float acc[ACC_MAX];
    float acc1[ACC_MAX];
    float q_reg[ACC_MAX];
    float q_reg1[ACC_MAX];
    int   acc_dim[ACC_MAX];
    int   acc_count = 0;
    #pragma unroll
    for (int i = 0; i < ACC_MAX; ++i) {
        acc[i] = 0.0f;
        acc1[i] = 0.0f;
        q_reg[i] = 0.0f;
        q_reg1[i] = 0.0f;
        acc_dim[i] = -1;
    }
    for (int d = lane; d < head_dim && acc_count < ACC_MAX; d += warpSize) {
        acc_dim[acc_count++] = d;
    }
    if (active) {
        for (int i = 0; i < acc_count; ++i) {
            q_reg[i] = supersonic_qwen35_to_float(q_row[acc_dim[i]]);
        }
    }
    if (active1) {
        for (int i = 0; i < acc_count; ++i) {
            q_reg1[i] = supersonic_qwen35_to_float(q_row1[acc_dim[i]]);
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float m1 = -INFINITY;
    float l1 = 0.0f;

    // Iterate K tiles. Even inactive warps must participate in cooperative
    // loads and __syncthreads() — they just won't write output.
    const int causal_for_block = active
        ? (active1 ? ((causal_limit1 > causal_limit) ? causal_limit1 : causal_limit)
                   : causal_limit)
        : (active1 ? causal_limit1 : 0);
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
        if (sizeof(T) == 2 && (head_dim % 2) == 0) {
            const int vec_total = load_total >> 1;
            for (int vi = linear_tid; vi < vec_total; vi += total_threads) {
                const int elem = vi << 1;
                const int row = elem / head_dim;
                const int col = elem - row * head_dim;
                const size_t off =
                    (size_t)(tile_base + row) * (size_t)head_dim + (size_t)col;
                *reinterpret_cast<uint32_t*>(&k_tile[elem]) =
                    *reinterpret_cast<const uint32_t*>(k_head_ptr + off);
                *reinterpret_cast<uint32_t*>(&v_tile[elem]) =
                    *reinterpret_cast<const uint32_t*>(v_head_ptr + off);
            }
        } else {
            for (int i = linear_tid; i < load_total; i += total_threads) {
                const int row = i / head_dim;
                const int col = i - row * head_dim;
                const int gk_pos = tile_base + row;
                const size_t off = (size_t)gk_pos * head_dim + (size_t)col;
                k_tile[i] = k_head_ptr[off];
                v_tile[i] = v_head_ptr[off];
            }
        }
        __syncthreads();

        // Walk this tile, applying the online softmax recurrence per query row.
        if (active) {
            const int j_max = min(tile_len, causal_limit - tile_base);
            int j = 0;
            for (; j + 3 < j_max; j += 4) {
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                const T* k0 = k_tile + j * head_dim;
                const T* k1 = k0 + head_dim;
                const T* k2 = k1 + head_dim;
                const T* k3 = k2 + head_dim;
                #pragma unroll
                for (int i = 0; i < ACC_MAX; ++i) {
                    if (i < acc_count) {
                        const float qv = q_reg[i];
                        const int d = acc_dim[i];
                        p0 += qv * supersonic_qwen35_to_float(k0[d]);
                        p1 += qv * supersonic_qwen35_to_float(k1[d]);
                        p2 += qv * supersonic_qwen35_to_float(k2[d]);
                        p3 += qv * supersonic_qwen35_to_float(k3[d]);
                    }
                }
                const float s0 = __shfl(supersonic_qwen35_wave_sum(p0), 0) * scale;
                const float s1 = __shfl(supersonic_qwen35_wave_sum(p1), 0) * scale;
                const float s2 = __shfl(supersonic_qwen35_wave_sum(p2), 0) * scale;
                const float s3 = __shfl(supersonic_qwen35_wave_sum(p3), 0) * scale;
                const T* v0 = v_tile + j * head_dim;
                const T* vs[4] = {v0, v0 + head_dim, v0 + 2 * head_dim, v0 + 3 * head_dim};
                const float scores[4] = {s0, s1, s2, s3};
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float m_old = m_i;
                    const float m_new = scores[u] > m_old ? scores[u] : m_old;
                    const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                    const float weight = __expf(scores[u] - m_new);
                    l_i = l_i * alpha + weight;
                    m_i = m_new;
                    for (int i = 0; i < acc_count; ++i) {
                        acc[i] = acc[i] * alpha +
                            weight * supersonic_qwen35_to_float(vs[u][acc_dim[i]]);
                    }
                    if (active1 && (tile_base + j + u) < causal_limit1) {
                        float p1q = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < ACC_MAX; ++i) {
                            if (i < acc_count) {
                                p1q += q_reg1[i] * supersonic_qwen35_to_float(
                                    (u == 0 ? k0 : u == 1 ? k1 : u == 2 ? k2 : k3)[acc_dim[i]]);
                            }
                        }
                        const float sc1 =
                            __shfl(supersonic_qwen35_wave_sum(p1q), 0) * scale;
                        const float m_old1 = m1;
                        const float m_new1 = sc1 > m_old1 ? sc1 : m_old1;
                        const float al1 = (m_old1 == -INFINITY) ? 0.0f : __expf(m_old1 - m_new1);
                        const float w1 = __expf(sc1 - m_new1);
                        l1 = l1 * al1 + w1;
                        m1 = m_new1;
                        for (int i = 0; i < acc_count; ++i) {
                            acc1[i] = acc1[i] * al1 +
                                w1 * supersonic_qwen35_to_float(vs[u][acc_dim[i]]);
                        }
                    }
                }
            }
            for (; j < j_max; ++j) {
                const T* k_row = k_tile + j * head_dim;
                const T* v_row = v_tile + j * head_dim;

                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < ACC_MAX; ++i) {
                    if (i < acc_count) {
                        partial += q_reg[i] * supersonic_qwen35_to_float(k_row[acc_dim[i]]);
                    }
                }
                float score = __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;

                const float m_old = m_i;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                const float weight = __expf(score - m_new);
                l_i = l_i * alpha + weight;
                m_i = m_new;

                for (int i = 0; i < acc_count; ++i) {
                    acc[i] = acc[i] * alpha + weight * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
                }
                if (active1 && (tile_base + j) < causal_limit1) {
                    float p1q = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < ACC_MAX; ++i) {
                        if (i < acc_count) {
                            p1q += q_reg1[i] * supersonic_qwen35_to_float(k_row[acc_dim[i]]);
                        }
                    }
                    const float sc1 =
                        __shfl(supersonic_qwen35_wave_sum(p1q), 0) * scale;
                    const float m_old1 = m1;
                    const float m_new1 = sc1 > m_old1 ? sc1 : m_old1;
                    const float al1 = (m_old1 == -INFINITY) ? 0.0f : __expf(m_old1 - m_new1);
                    const float w1 = __expf(sc1 - m_new1);
                    l1 = l1 * al1 + w1;
                    m1 = m_new1;
                    for (int i = 0; i < acc_count; ++i) {
                        acc1[i] = acc1[i] * al1 +
                            w1 * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
                    }
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
    if (active1) {
        const float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        for (int i = 0; i < acc_count; ++i) {
            out_row1[acc_dim[i]] = acc1[i] * inv1;
        }
    }
}

// Causal softmax over scores[n_rows, kv_len], one block per row.
// GQA packs stack Q heads, so qi = row % q_len (same mask per query
// position across the heads that share a KV head).
// Applies `scale` and masks j >= seqlen_offset+qi+1.
__global__ void supersonic_qwen35_causal_softmax_rows(
    float* scores,
    int n_rows,
    int q_len,
    int kv_len,
    int seqlen_offset,
    float scale
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row < 0 || row >= n_rows || q_len <= 0) {
        return;
    }
    const int qi = row % q_len;
    int limit = seqlen_offset + qi + 1;
    if (limit > kv_len) {
        limit = kv_len;
    }
    if (limit < 0) {
        limit = 0;
    }
    float* srow = scores + static_cast<size_t>(row) * kv_len;
    const int tid = static_cast<int>(threadIdx.x);
    const int bs = static_cast<int>(blockDim.x);

    float local_max = -INFINITY;
    for (int j = tid; j < kv_len; j += bs) {
        float s = srow[j] * scale;
        if (j >= limit) {
            s = -INFINITY;
        }
        srow[j] = s;
        if (s > local_max) {
            local_max = s;
        }
    }
    __shared__ float red[256];
    red[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = fmaxf(red[tid], red[tid + s]);
        }
        __syncthreads();
    }
    const float m = red[0];
    float local_sum = 0.0f;
    for (int j = tid; j < limit; j += bs) {
        const float e = expf(srow[j] - m);
        srow[j] = e;
        local_sum += e;
    }
    for (int j = limit + tid; j < kv_len; j += bs) {
        srow[j] = 0.0f;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float inv = (red[0] > 0.0f) ? (1.0f / red[0]) : 0.0f;
    for (int j = tid; j < limit; j += bs) {
        srow[j] *= inv;
    }
}

// Same causal softmax, BF16 scores in and out. Accumulates in F32.
__global__ void supersonic_qwen35_causal_softmax_rows_bf16(
    hip_bfloat16* scores,
    int n_rows,
    int q_len,
    int kv_len,
    int seqlen_offset,
    float scale
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row < 0 || row >= n_rows || q_len <= 0) {
        return;
    }
    const int qi = row % q_len;
    int limit = seqlen_offset + qi + 1;
    if (limit > kv_len) {
        limit = kv_len;
    }
    if (limit < 0) {
        limit = 0;
    }
    hip_bfloat16* srow = scores + static_cast<size_t>(row) * kv_len;
    const int tid = static_cast<int>(threadIdx.x);
    const int bs = static_cast<int>(blockDim.x);

    float local_max = -INFINITY;
    for (int j = tid; j < kv_len; j += bs) {
        float s = static_cast<float>(srow[j]) * scale;
        if (j >= limit) {
            s = -INFINITY;
        }
        srow[j] = static_cast<hip_bfloat16>(s);
        if (s > local_max) {
            local_max = s;
        }
    }
    __shared__ float red[256];
    red[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = fmaxf(red[tid], red[tid + s]);
        }
        __syncthreads();
    }
    const float m = red[0];
    float local_sum = 0.0f;
    for (int j = tid; j < limit; j += bs) {
        const float e = expf(static_cast<float>(srow[j]) - m);
        srow[j] = static_cast<hip_bfloat16>(e);
        local_sum += e;
    }
    for (int j = limit + tid; j < kv_len; j += bs) {
        srow[j] = static_cast<hip_bfloat16>(0.0f);
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float inv = (red[0] > 0.0f) ? (1.0f / red[0]) : 0.0f;
    for (int j = tid; j < limit; j += bs) {
        srow[j] = static_cast<hip_bfloat16>(static_cast<float>(srow[j]) * inv);
    }
}

// BF16 V -> F32 for the AV GEMM.
__global__ void supersonic_qwen35_cast_bf16_to_f32(
    const hip_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = supersonic_qwen35_to_float(src[i]);
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
                ((t * batch_heads + bh) * k_head_dim) * v_head_dim + v_idx;
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

__global__ void supersonic_qwen35_delta_recurrent_extract_kernel(
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const float* __restrict__ recurrent_out,
    float* __restrict__ recurrent_state,
    hip_bfloat16* __restrict__ attn_output
) {
    const size_t state_elems =
        static_cast<size_t>(batch_heads) * k_head_dim * v_head_dim;
    const size_t attn_elems =
        static_cast<size_t>(batch_heads) * seq_len * v_head_dim;
    const size_t tid =
        blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid < state_elems) {
        const size_t state_stride = k_head_dim * v_head_dim;
        const size_t head = tid / state_stride;
        const size_t within = tid - head * state_stride;
        const size_t src =
            head * ((seq_len + k_head_dim) * v_head_dim) +
            seq_len * v_head_dim + within;
        recurrent_state[tid] = recurrent_out[src];
    } else if (tid < state_elems + attn_elems) {
        const size_t attn_index = tid - state_elems;
        const size_t attn_stride = seq_len * v_head_dim;
        const size_t head = attn_index / attn_stride;
        const size_t within = attn_index - head * attn_stride;
        const size_t src =
            head * ((seq_len + k_head_dim) * v_head_dim) + within;
        attn_output[attn_index] =
            supersonic_qwen35_from_float<hip_bfloat16>(recurrent_out[src]);
    }
}

// Specialized k=v=128 rec. Same product order as the generic warp
// kernel (scale, kv_mem, delta, state+=, out) but k-per-lane is a
// compile-time 4, so the 8070-token loop has no nk branches and no
// block-wide LDS sync. Launch: 4 warps × 32, bounds 128/2.
template <typename T>
__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_delta_recurrent_prefill_warp_k128_kernel(
    int batch_heads,
    int seq_len,
    const T* __restrict__ initial_state,
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ beta,
    const T* __restrict__ g,
    T* __restrict__ out,
    half* __restrict__ state_trace
) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int R = 4;
    constexpr int V_PER_WARP = 4;
    const int lane = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int total = batch_heads * (V / V_PER_WARP);
    if (lane >= 32 || warp_id >= total) {
        return;
    }
    const int groups_per_head = V / V_PER_WARP;
    const int bh = warp_id / groups_per_head;
    const int v0 = (warp_id - bh * groups_per_head) * V_PER_WARP;
    const int v1 = v0 + 1;
    const int v2 = v0 + 2;
    const int v3 = v0 + 3;
    const int state_stride = K * V;
    const int token_stride_k = seq_len * K;
    const int token_stride_v = seq_len * V;
    const int out_base = bh * (seq_len + K) * V;

    float state0[R];
    float state1[R];
    float state2[R];
    float state3[R];
    int k_ids[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        k_ids[r] = lane + r * 32;
        const int ks = bh * state_stride + k_ids[r] * V;
        state0[r] = supersonic_qwen35_to_float(initial_state[ks + v0]);
        state1[r] = supersonic_qwen35_to_float(initial_state[ks + v1]);
        state2[r] = supersonic_qwen35_to_float(initial_state[ks + v2]);
        state3[r] = supersonic_qwen35_to_float(initial_state[ks + v3]);
    }

    float g_cur = 0.0f;
    float ds_cur = 0.0f;
    float v0_cur = 0.0f;
    float v1_cur = 0.0f;
    float v2_cur = 0.0f;
    float v3_cur = 0.0f;
    float k_cur[R];
    float q_cur[R];
    if (seq_len > 0) {
        g_cur = __expf(supersonic_qwen35_to_float(g[bh * seq_len]));
        ds_cur = supersonic_qwen35_to_float(beta[bh * seq_len]);
        v0_cur = supersonic_qwen35_to_float(value[bh * token_stride_v + v0]);
        v1_cur = supersonic_qwen35_to_float(value[bh * token_stride_v + v1]);
        v2_cur = supersonic_qwen35_to_float(value[bh * token_stride_v + v2]);
        v3_cur = supersonic_qwen35_to_float(value[bh * token_stride_v + v3]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            k_cur[r] = supersonic_qwen35_to_float(key[bh * token_stride_k + k_ids[r]]);
            q_cur[r] = supersonic_qwen35_to_float(query[bh * token_stride_k + k_ids[r]]);
        }
    }
    for (int t = 0; t < seq_len; ++t) {
        float g_n = 0.0f;
        float ds_n = 0.0f;
        float v0_n = 0.0f;
        float v1_n = 0.0f;
        float v2_n = 0.0f;
        float v3_n = 0.0f;
        float k_n[R];
        float q_n[R];
        if (t + 1 < seq_len) {
            const int tn = t + 1;
            g_n = __expf(supersonic_qwen35_to_float(g[bh * seq_len + tn]));
            ds_n = supersonic_qwen35_to_float(beta[bh * seq_len + tn]);
            const int val_row_n = bh * token_stride_v + tn * V;
            v0_n = supersonic_qwen35_to_float(value[val_row_n + v0]);
            v1_n = supersonic_qwen35_to_float(value[val_row_n + v1]);
            v2_n = supersonic_qwen35_to_float(value[val_row_n + v2]);
            v3_n = supersonic_qwen35_to_float(value[val_row_n + v3]);
            const int key_row_n = bh * token_stride_k + tn * K;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                k_n[r] = supersonic_qwen35_to_float(key[key_row_n + k_ids[r]]);
                q_n[r] = supersonic_qwen35_to_float(query[key_row_n + k_ids[r]]);
            }
        }
        asm volatile("" ::: "memory");

        float kv0 = 0.0f;
        float kv1 = 0.0f;
        float kv2 = 0.0f;
        float kv3 = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            state0[r] *= g_cur;
            state1[r] *= g_cur;
            state2[r] *= g_cur;
            state3[r] *= g_cur;
            kv0 += state0[r] * k_cur[r];
            kv1 += state1[r] * k_cur[r];
            kv2 += state2[r] * k_cur[r];
            kv3 += state3[r] * k_cur[r];
        }
        kv0 = __shfl(supersonic_qwen35_wave_sum(kv0), 0);
        kv1 = __shfl(supersonic_qwen35_wave_sum(kv1), 0);
        kv2 = __shfl(supersonic_qwen35_wave_sum(kv2), 0);
        kv3 = __shfl(supersonic_qwen35_wave_sum(kv3), 0);
        const float d0 = (v0_cur - kv0) * ds_cur;
        const float d1 = (v1_cur - kv1) * ds_cur;
        const float d2 = (v2_cur - kv2) * ds_cur;
        const float d3 = (v3_cur - kv3) * ds_cur;

        float o0 = 0.0f;
        float o1 = 0.0f;
        float o2 = 0.0f;
        float o3 = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            state0[r] += k_cur[r] * d0;
            state1[r] += k_cur[r] * d1;
            state2[r] += k_cur[r] * d2;
            state3[r] += k_cur[r] * d3;
            o0 += state0[r] * q_cur[r];
            o1 += state1[r] * q_cur[r];
            o2 += state2[r] * q_cur[r];
            o3 += state3[r] * q_cur[r];
        }
        o0 = __shfl(supersonic_qwen35_wave_sum(o0), 0);
        o1 = __shfl(supersonic_qwen35_wave_sum(o1), 0);
        o2 = __shfl(supersonic_qwen35_wave_sum(o2), 0);
        o3 = __shfl(supersonic_qwen35_wave_sum(o3), 0);
        if (state_trace != nullptr) {
            const int trace_base = ((t * batch_heads + bh) * K) * V;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int trace_row = trace_base + k_ids[r] * V;
                state_trace[trace_row + v0] =
                    supersonic_qwen35_from_float<half>(state0[r]);
                state_trace[trace_row + v1] =
                    supersonic_qwen35_from_float<half>(state1[r]);
                state_trace[trace_row + v2] =
                    supersonic_qwen35_from_float<half>(state2[r]);
                state_trace[trace_row + v3] =
                    supersonic_qwen35_from_float<half>(state3[r]);
            }
        }
        if (lane == 0) {
            out[out_base + t * V + v0] = supersonic_qwen35_from_float<T>(o0);
            out[out_base + t * V + v1] = supersonic_qwen35_from_float<T>(o1);
            out[out_base + t * V + v2] = supersonic_qwen35_from_float<T>(o2);
            out[out_base + t * V + v3] = supersonic_qwen35_from_float<T>(o3);
        }

        g_cur = g_n;
        ds_cur = ds_n;
        v0_cur = v0_n;
        v1_cur = v1_n;
        v2_cur = v2_n;
        v3_cur = v3_n;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            k_cur[r] = k_n[r];
            q_cur[r] = q_n[r];
        }
    }

    const int state_out = out_base + seq_len * V;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int so = state_out + k_ids[r] * V;
        out[so + v0] = supersonic_qwen35_from_float<T>(state0[r]);
        out[so + v1] = supersonic_qwen35_from_float<T>(state1[r]);
        out[so + v2] = supersonic_qwen35_from_float<T>(state2[r]);
        out[so + v3] = supersonic_qwen35_from_float<T>(state3[r]);
    }
}

// Decode seq=1 k128 rec that reads unique q/k (repeat via h%nk) and
// packs beta/g in-register. Same rec product order as the prefill
// k128 kernel; same beta/g math as decode_pack_rec_inputs.
__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_decode_rec_k128_fused_kernel(
    int nv,
    int nk,
    float* __restrict__ rec_state,
    const float* __restrict__ q_unique,
    const float* __restrict__ k_unique,
    const float* __restrict__ value,
    const float* __restrict__ b,
    const float* __restrict__ a,
    const hip_bfloat16* __restrict__ dt_bias,
    const hip_bfloat16* __restrict__ a_log_exp,
    float* __restrict__ out) {
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr int R = 4;
    constexpr int V_PER_WARP = 4;
    const int lane = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int total = nv * (V / V_PER_WARP);
    if (lane >= 32 || warp_id >= total) {
        return;
    }
    const int groups_per_head = V / V_PER_WARP;
    const int bh = warp_id / groups_per_head;
    const int v0 = (warp_id - bh * groups_per_head) * V_PER_WARP;
    const int v1 = v0 + 1;
    const int v2 = v0 + 2;
    const int v3 = v0 + 3;
    const int state_stride = K * V;
    const int src_h = bh % nk;
    const int out_base = bh * V;

    int k_ids[R];
    float state0[R], state1[R], state2[R], state3[R];
    float k_cur[R], q_cur[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        k_ids[r] = lane + r * 32;
        const int ks = bh * state_stride + k_ids[r] * V;
        state0[r] = rec_state[ks + v0];
        state1[r] = rec_state[ks + v1];
        state2[r] = rec_state[ks + v2];
        state3[r] = rec_state[ks + v3];
        q_cur[r] = q_unique[src_h * K + k_ids[r]];
        k_cur[r] = k_unique[src_h * K + k_ids[r]];
    }

    const float beta = 1.0f / (1.0f + expf(-b[bh]));
    float g_cur = 0.0f;
    if (dt_bias != nullptr && a_log_exp != nullptr) {
        const float dt = supersonic_qwen35_to_float(dt_bias[bh]);
        const float ale = supersonic_qwen35_to_float(a_log_exp[bh]);
        const float sp = logf(1.0f + expf(a[bh] + dt));
        g_cur = __expf(-sp * ale);
    } else {
        g_cur = 1.0f;
    }
    const float v0_cur = value[bh * V + v0];
    const float v1_cur = value[bh * V + v1];
    const float v2_cur = value[bh * V + v2];
    const float v3_cur = value[bh * V + v3];

    float kv0 = 0.0f, kv1 = 0.0f, kv2 = 0.0f, kv3 = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        state0[r] *= g_cur;
        state1[r] *= g_cur;
        state2[r] *= g_cur;
        state3[r] *= g_cur;
        kv0 += state0[r] * k_cur[r];
        kv1 += state1[r] * k_cur[r];
        kv2 += state2[r] * k_cur[r];
        kv3 += state3[r] * k_cur[r];
    }
    kv0 = __shfl(supersonic_qwen35_wave_sum(kv0), 0);
    kv1 = __shfl(supersonic_qwen35_wave_sum(kv1), 0);
    kv2 = __shfl(supersonic_qwen35_wave_sum(kv2), 0);
    kv3 = __shfl(supersonic_qwen35_wave_sum(kv3), 0);
    const float d0 = (v0_cur - kv0) * beta;
    const float d1 = (v1_cur - kv1) * beta;
    const float d2 = (v2_cur - kv2) * beta;
    const float d3 = (v3_cur - kv3) * beta;

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        state0[r] += k_cur[r] * d0;
        state1[r] += k_cur[r] * d1;
        state2[r] += k_cur[r] * d2;
        state3[r] += k_cur[r] * d3;
        o0 += state0[r] * q_cur[r];
        o1 += state1[r] * q_cur[r];
        o2 += state2[r] * q_cur[r];
        o3 += state3[r] * q_cur[r];
    }
    o0 = __shfl(supersonic_qwen35_wave_sum(o0), 0);
    o1 = __shfl(supersonic_qwen35_wave_sum(o1), 0);
    o2 = __shfl(supersonic_qwen35_wave_sum(o2), 0);
    o3 = __shfl(supersonic_qwen35_wave_sum(o3), 0);
    if (lane == 0) {
        out[out_base + v0] = o0;
        out[out_base + v1] = o1;
        out[out_base + v2] = o2;
        out[out_base + v3] = o3;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int so = bh * state_stride + k_ids[r] * V;
        rec_state[so + v0] = state0[r];
        rec_state[so + v1] = state1[r];
        rec_state[so + v2] = state2[r];
        rec_state[so + v3] = state3[r];
    }
}

// Warp-split rec: one warp per (head, v_idx). Each lane holds
// k = lane, lane+32, ... so state is 4–8 floats instead of 128.
// The one-thread-per-v path spills `float state[128]` to scratch and
// serializes 8k timesteps at ~356 ms/layer on gfx1100.
template <typename T, int MAX_K_PER_LANE = 8>
__global__ void supersonic_qwen35_delta_recurrent_prefill_warp_kernel(
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
    const int lane = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int total = batch_heads * v_head_dim;
    const bool valid = lane < 32 && warp_id < total &&
        k_head_dim <= MAX_K_PER_LANE * 32;
    // Consecutive warps in a block share a head, so they share k/q rows.
    // Do not early-return: the token loop uses block-wide LDS + syncthreads.
    const int bh = valid ? (warp_id / v_head_dim) : 0;
    const int v_idx = valid ? (warp_id - bh * v_head_dim) : 0;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_stride_k = seq_len * k_head_dim;
    const int token_stride_v = seq_len * v_head_dim;
    const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + k_head_dim) * v_head_dim;
    const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);

    // Double-buffered k/q so token t+1 can fill LDS under token t FMAs.
    __shared__ float s_k[2][256];
    __shared__ float s_q[2][256];

    float state[MAX_K_PER_LANE];
    int k_ids[MAX_K_PER_LANE];
    int nk = 0;
    if (valid) {
        for (int k_idx = lane; k_idx < k_head_dim && nk < MAX_K_PER_LANE; k_idx += 32) {
            k_ids[nk] = k_idx;
            state[nk] = supersonic_qwen35_to_float(
                initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
            ++nk;
        }
    }

    // Block-shared k/q: 8 warps of one head used to reread the same row
    // from global. Double-buffer so token t+1 fills LDS under token t.
    float g_cur = 0.0f;
    float ds_cur = 0.0f;
    float v_cur = 0.0f;
    if (seq_len > 0) {
        if (valid) {
            g_cur = __expf(supersonic_qwen35_to_float(g[bh * token_stride_s]));
            ds_cur = supersonic_qwen35_to_float(beta[bh * token_stride_s]);
            v_cur = supersonic_qwen35_to_float(value[bh * token_stride_v + v_idx]);
        }
        if (tid < k_head_dim) {
            s_k[0][tid] = supersonic_qwen35_to_float(key[bh * token_stride_k + tid]);
            s_q[0][tid] = supersonic_qwen35_to_float(query[bh * token_stride_k + tid]);
        }
    }
    __syncthreads();
    for (int t = 0; t < seq_len; ++t) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;
        float g_n = 0.0f;
        float ds_n = 0.0f;
        float v_n = 0.0f;
        if (t + 1 < seq_len) {
            const int tn = t + 1;
            if (valid) {
                g_n = __expf(supersonic_qwen35_to_float(g[bh * token_stride_s + tn]));
                ds_n = supersonic_qwen35_to_float(beta[bh * token_stride_s + tn]);
                v_n = supersonic_qwen35_to_float(
                    value[bh * token_stride_v + tn * v_head_dim + v_idx]);
            }
            if (tid < k_head_dim) {
                const int key_row_n = bh * token_stride_k + tn * k_head_dim;
                s_k[nxt][tid] = supersonic_qwen35_to_float(key[key_row_n + tid]);
                s_q[nxt][tid] = supersonic_qwen35_to_float(query[key_row_n + tid]);
            }
        }

        float k_reg[MAX_K_PER_LANE];
        float q_reg[MAX_K_PER_LANE];
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < nk) {
                k_reg[i] = s_k[cur][k_ids[i]];
                q_reg[i] = s_q[cur][k_ids[i]];
                state[i] *= g_cur;
                kv_mem += state[i] * k_reg[i];
            }
        }
        kv_mem = __shfl(supersonic_qwen35_wave_sum(kv_mem), 0);
        const float delta = (v_cur - kv_mem) * ds_cur;

        float out_t = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < nk) {
                state[i] += k_reg[i] * delta;
                out_t += state[i] * q_reg[i];
            }
        }
        out_t = __shfl(supersonic_qwen35_wave_sum(out_t), 0);
        if (valid && lane == 0) {
            out[out_base + t * v_head_dim + v_idx] = supersonic_qwen35_from_float<T>(out_t);
        }

        __syncthreads();
        g_cur = g_n;
        ds_cur = ds_n;
        v_cur = v_n;
    }

    if (valid) {
        const int state_out = out_base + seq_len * v_head_dim;
        for (int i = 0; i < nk; ++i) {
            out[state_out + k_ids[i] * v_head_dim + v_idx] =
                supersonic_qwen35_from_float<T>(state[i]);
        }
    }
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
__global__ void supersonic_qwen35_fill_conv_tail_kernel(
    int qkv_dim,
    int pad,
    int total_len,
    const T* tail,
    T* conv_input
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = qkv_dim * pad;
    if (idx >= total) {
        return;
    }
    const int channel = idx / pad;
    const int tap = idx - channel * pad;
    conv_input[channel * total_len + tap] = tail[idx];
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
// Processes all 64 decoder layers in a single kernel launch.
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
