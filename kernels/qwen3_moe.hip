// Qwen3-MoE HIP decode kernels.
//
// This file is intentionally separate from Qwen3.6-MoE. Qwen3-30B-A3B is a
// pure full-attention MoE model with different tensor geometry and no hybrid
// linear-attention/shared-expert path. The decode path uses a Qwen3-owned
// cooperative block implementation with INT4 dequant for attention/expert
// weights, BF16 KV cache, and an optional single-launch persistent layer chain.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <math.h>
#include <stdint.h>

namespace qwen3_moe {

struct DecodeLayerDesc {
    int         layer_idx;
    const void* input_norm_w;
    float       input_norm_eps;
    const void* post_attn_norm_w;
    float       post_attn_norm_eps;

    const void* q_proj_w;
    const void* k_proj_w;
    const void* v_proj_w;
    const void* o_proj_w;
    const void* q_norm_w;
    const void* k_norm_w;
    float       rope_theta;
    int         head_dim;
    int         num_heads;
    int         num_kv_heads;
    void*       kv_cache_k;
    void*       kv_cache_v;
    int         kv_len;
    int         kv_max_t;

    const void* router_w;
    const void* experts_gate_up_w;
    const void* experts_down_w;
    int         num_experts;
    int         top_k;
    int         moe_intermediate_size;
    int         norm_topk_prob;
};

struct Int4ScaleDesc {
    const void* q_proj_scale;
    const void* q_proj_zero;
    const void* k_proj_scale;
    const void* k_proj_zero;
    const void* v_proj_scale;
    const void* v_proj_zero;
    const void* o_proj_scale;
    const void* o_proj_zero;
    const void* experts_gate_up_scale;
    const void* experts_gate_up_zero;
    const void* experts_down_scale;
    const void* experts_down_zero;
    int         group_size;
};

__global__ void qwen3_moe_descriptor_walk_stub(
    int                              num_layers,
    const DecodeLayerDesc* __restrict__ layers,
    float* __restrict__              workspace,
    unsigned int* __restrict__       counters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        workspace[0] = 0.0f; // layers seen
        workspace[1] = 0.0f; // total experts
        workspace[2] = 0.0f; // total top_k
        workspace[3] = 1.0f; // descriptor geometry OK
    }
    __syncthreads();

    while (true) {
        const unsigned int idx = atomicAdd(&counters[0], 1u);
        if (idx >= static_cast<unsigned int>(num_layers)) return;

        const DecodeLayerDesc& d = layers[idx];
        if (threadIdx.x == 0) {
            atomicAdd(&workspace[0], 1.0f);
            atomicAdd(&workspace[1], static_cast<float>(d.num_experts));
            atomicAdd(&workspace[2], static_cast<float>(d.top_k));
            if (d.layer_idx != static_cast<int>(idx) ||
                d.head_dim <= 0 ||
                d.num_heads <= 0 ||
                d.num_kv_heads <= 0 ||
                d.num_experts <= 0 ||
                d.top_k <= 0 ||
                d.moe_intermediate_size <= 0) {
                workspace[3] = 0.0f;
            }
        }
    }
}

__device__ __forceinline__ float bf16_round_rne_f32(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += bias;
    bits &= 0xFFFF0000u;
    float y;
    __builtin_memcpy(&y, &bits, 4);
    return y;
}

__device__ __forceinline__ float load_bf16(const void* p, int idx) {
    return static_cast<float>(static_cast<const hip_bfloat16*>(p)[idx]);
}

__device__ __forceinline__ void store_bf16(void* p, int idx, float v) {
    static_cast<hip_bfloat16*>(p)[idx] = static_cast<hip_bfloat16>(v);
}

__device__ float block_sum(float v, float* scratch) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    scratch[tid] = v;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    return scratch[0];
}

__device__ float int4_value(
    const uint8_t* __restrict__ packed,
    const hip_bfloat16* __restrict__ scale,
    const hip_bfloat16* __restrict__ zero,
    int row,
    int col,
    int cols,
    int group_size) {
    const int byte_cols = (cols + 1) / 2;
    const uint8_t byte = packed[static_cast<size_t>(row) * byte_cols + col / 2];
    const int nibble = (col & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    const int scale_cols = (cols + group_size - 1) / group_size;
    const int scale_idx = (row / group_size) * scale_cols + (col / group_size);
    const float s = static_cast<float>(scale[scale_idx]);
    const float z = static_cast<float>(zero[scale_idx]);
    return bf16_round_rne_f32(static_cast<float>(nibble) * s - z * s);
}

__device__ float int4_value_3d(
    const uint8_t* __restrict__ packed,
    const hip_bfloat16* __restrict__ scale,
    const hip_bfloat16* __restrict__ zero,
    int expert,
    int row,
    int col,
    int rows,
    int cols,
    int group_size) {
    const int byte_cols = (cols + 1) / 2;
    const size_t packed_base =
        (static_cast<size_t>(expert) * rows + row) * byte_cols;
    const uint8_t byte = packed[packed_base + col / 2];
    const int nibble = (col & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);

    const int scale_rows = (rows + group_size - 1) / group_size;
    const int scale_cols = (cols + group_size - 1) / group_size;
    const size_t scale_idx =
        (static_cast<size_t>(expert) * scale_rows + row / group_size) *
            scale_cols +
        col / group_size;
    const float s = static_cast<float>(scale[scale_idx]);
    const float z = static_cast<float>(zero[scale_idx]);
    return bf16_round_rne_f32(static_cast<float>(nibble) * s - z * s);
}

__device__ __forceinline__ void int4_dequant_8(
    uint32_t packed,
    const hip_bfloat16* __restrict__ scale,
    const hip_bfloat16* __restrict__ zero,
    int scale_idx,
    float out[8]) {
    const float s = static_cast<float>(scale[scale_idx]);
    const float zs = static_cast<float>(zero[scale_idx]) * s;
    out[0] = bf16_round_rne_f32(static_cast<float>((packed >>  0) & 0xF) * s - zs);
    out[1] = bf16_round_rne_f32(static_cast<float>((packed >>  4) & 0xF) * s - zs);
    out[2] = bf16_round_rne_f32(static_cast<float>((packed >>  8) & 0xF) * s - zs);
    out[3] = bf16_round_rne_f32(static_cast<float>((packed >> 12) & 0xF) * s - zs);
    out[4] = bf16_round_rne_f32(static_cast<float>((packed >> 16) & 0xF) * s - zs);
    out[5] = bf16_round_rne_f32(static_cast<float>((packed >> 20) & 0xF) * s - zs);
    out[6] = bf16_round_rne_f32(static_cast<float>((packed >> 24) & 0xF) * s - zs);
    out[7] = bf16_round_rne_f32(static_cast<float>((packed >> 28) & 0xF) * s - zs);
}

__device__ float dense_or_int4_dot(
    const void* weight,
    const void* scale,
    const void* zero,
    int row,
    const float* x,
    int cols,
    int group_size,
    float* scratch) {
    float partial = 0.0f;
    if (scale != nullptr) {
        const uint8_t* w = static_cast<const uint8_t*>(weight);
        const hip_bfloat16* s = static_cast<const hip_bfloat16*>(scale);
        const hip_bfloat16* z = static_cast<const hip_bfloat16*>(zero);
        const int byte_cols = (cols + 1) / 2;
        const int scale_cols = (cols + group_size - 1) / group_size;
        const int scale_row = row / group_size;
        int c = threadIdx.x * 8;
        for (; c + 7 < cols; c += blockDim.x * 8) {
            uint32_t packed;
            __builtin_memcpy(
                &packed,
                w + static_cast<size_t>(row) * byte_cols + c / 2,
                sizeof(packed));
            float vals[8];
            int4_dequant_8(packed, s, z, scale_row * scale_cols + c / group_size, vals);
            partial += vals[0] * x[c + 0] + vals[1] * x[c + 1] +
                       vals[2] * x[c + 2] + vals[3] * x[c + 3] +
                       vals[4] * x[c + 4] + vals[5] * x[c + 5] +
                       vals[6] * x[c + 6] + vals[7] * x[c + 7];
        }
        for (; c < cols; ++c) {
            partial += int4_value(w, s, z, row, c, cols, group_size) * x[c];
        }
    } else {
        const hip_bfloat16* w = static_cast<const hip_bfloat16*>(weight);
        const size_t base = static_cast<size_t>(row) * cols;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            partial += static_cast<float>(w[base + c]) * x[c];
        }
    }
    return block_sum(partial, scratch);
}

__device__ float expert_int4_dot(
    const void* weight,
    const void* scale,
    const void* zero,
    int expert,
    int row,
    const float* x,
    int rows,
    int cols,
    int group_size,
    float* scratch) {
    const uint8_t* w = static_cast<const uint8_t*>(weight);
    const hip_bfloat16* s = static_cast<const hip_bfloat16*>(scale);
    const hip_bfloat16* z = static_cast<const hip_bfloat16*>(zero);
    float partial = 0.0f;
    const int byte_cols = (cols + 1) / 2;
    const int scale_rows = (rows + group_size - 1) / group_size;
    const int scale_cols = (cols + group_size - 1) / group_size;
    const size_t packed_base =
        (static_cast<size_t>(expert) * rows + row) * byte_cols;
    const int scale_base =
        (expert * scale_rows + row / group_size) * scale_cols;
    int c = threadIdx.x * 8;
    for (; c + 7 < cols; c += blockDim.x * 8) {
        uint32_t packed;
        __builtin_memcpy(&packed, w + packed_base + c / 2, sizeof(packed));
        float vals[8];
        int4_dequant_8(packed, s, z, scale_base + c / group_size, vals);
        partial += vals[0] * x[c + 0] + vals[1] * x[c + 1] +
                   vals[2] * x[c + 2] + vals[3] * x[c + 3] +
                   vals[4] * x[c + 4] + vals[5] * x[c + 5] +
                   vals[6] * x[c + 6] + vals[7] * x[c + 7];
    }
    for (; c < cols; ++c) {
        partial += int4_value_3d(w, s, z, expert, row, c, rows, cols, group_size) * x[c];
    }
    return block_sum(partial, scratch);
}

__device__ void rms_norm_vec(
    const float* in,
    const void* weight,
    float eps,
    float* out,
    int n,
    float* scratch) {
    float partial = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) partial += in[i] * in[i];
    const float sum = block_sum(partial, scratch);
    const float inv = rsqrtf(sum / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float w = load_bf16(weight, i);
        out[i] = bf16_round_rne_f32(in[i] * inv * w);
    }
    __syncthreads();
}

__device__ void qwen3_moe_decode_layer_device(
    const DecodeLayerDesc layer,
    const Int4ScaleDesc int4,
    int hidden,
    int position,
    const hip_bfloat16* __restrict__ input_hidden,
    hip_bfloat16* __restrict__ output_hidden,
    float* __restrict__ workspace,
    float* __restrict__ scratch) {
    const int tid = threadIdx.x;
    const int hd = layer.head_dim;
    const int h = layer.num_heads;
    const int hkv = layer.num_kv_heads;
    const int q_dim = h * hd;
    const int kv_dim = hkv * hd;
    const int group_size = int4.group_size;
    const int top_k = layer.top_k;
    const int experts = layer.num_experts;
    const int moe_i = layer.moe_intermediate_size;

    const int OFF_X = 0;
    const int OFF_Q = OFF_X + hidden;
    const int OFF_K = OFF_Q + q_dim;
    const int OFF_V = OFF_K + kv_dim;
    const int OFF_ATTN = OFF_V + kv_dim;
    const int OFF_HN = OFF_ATTN + q_dim;
    const int OFF_ROUTER = OFF_HN + hidden;
    const int OFF_PROBS = OFF_ROUTER + experts;
    const int OFF_TOPK_VAL = OFF_PROBS + experts;
    const int OFF_TOPK_IDX = OFF_TOPK_VAL + top_k;
    const int OFF_GU = OFF_TOPK_IDX + top_k;
    const int OFF_MID = OFF_GU + 2 * moe_i;
    const int OFF_EXPERT_OUT = OFF_MID + moe_i;
    const int OFF_MOE_OUT = OFF_EXPERT_OUT + hidden;

    float* x = workspace + OFF_X;
    float* q = workspace + OFF_Q;
    float* k = workspace + OFF_K;
    float* v = workspace + OFF_V;
    float* attn = workspace + OFF_ATTN;
    float* hnorm = workspace + OFF_HN;
    float* router = workspace + OFF_ROUTER;
    float* probs = workspace + OFF_PROBS;
    float* topk_val = workspace + OFF_TOPK_VAL;
    float* topk_idx = workspace + OFF_TOPK_IDX;
    float* gu = workspace + OFF_GU;
    float* mid = workspace + OFF_MID;
    float* expert_out = workspace + OFF_EXPERT_OUT;
    float* moe_out = workspace + OFF_MOE_OUT;

    for (int i = tid; i < hidden; i += blockDim.x) {
        x[i] = static_cast<float>(input_hidden[i]);
    }
    __syncthreads();

    rms_norm_vec(x, layer.input_norm_w, layer.input_norm_eps, hnorm, hidden, scratch);

    for (int row = 0; row < q_dim; ++row) {
        const float sum = dense_or_int4_dot(
            layer.q_proj_w, int4.q_proj_scale, int4.q_proj_zero,
            row, hnorm, hidden, group_size, scratch);
        if (tid == 0) q[row] = bf16_round_rne_f32(sum);
        __syncthreads();
    }
    for (int row = 0; row < kv_dim; ++row) {
        float sum = dense_or_int4_dot(
            layer.k_proj_w, int4.k_proj_scale, int4.k_proj_zero,
            row, hnorm, hidden, group_size, scratch);
        if (tid == 0) k[row] = bf16_round_rne_f32(sum);
        __syncthreads();
        sum = dense_or_int4_dot(
            layer.v_proj_w, int4.v_proj_scale, int4.v_proj_zero,
            row, hnorm, hidden, group_size, scratch);
        if (tid == 0) v[row] = bf16_round_rne_f32(sum);
        __syncthreads();
    }

    for (int head = 0; head < h; ++head) {
        float partial = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) {
            const float val = q[head * hd + i];
            partial += val * val;
        }
        const float sum = block_sum(partial, scratch);
        const float inv = rsqrtf(sum / static_cast<float>(hd) + layer.input_norm_eps);
        for (int i = tid; i < hd; i += blockDim.x) {
            const float w = load_bf16(layer.q_norm_w, i);
            q[head * hd + i] = bf16_round_rne_f32(q[head * hd + i] * inv * w);
        }
        __syncthreads();
    }
    for (int head = 0; head < hkv; ++head) {
        float partial = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) {
            const float val = k[head * hd + i];
            partial += val * val;
        }
        const float sum = block_sum(partial, scratch);
        const float inv = rsqrtf(sum / static_cast<float>(hd) + layer.input_norm_eps);
        for (int i = tid; i < hd; i += blockDim.x) {
            const float w = load_bf16(layer.k_norm_w, i);
            k[head * hd + i] = bf16_round_rne_f32(k[head * hd + i] * inv * w);
        }
        __syncthreads();
    }

    const int half = hd / 2;
    const float theta_log = logf(layer.rope_theta);
    for (int head = 0; head < h; ++head) {
        for (int i = tid; i < half; i += blockDim.x) {
            const float a = q[head * hd + i];
            const float b = q[head * hd + half + i];
            const float freq = static_cast<float>(position) *
                expf(-(static_cast<float>(i) / static_cast<float>(half)) * theta_log);
            const float c = bf16_round_rne_f32(cosf(freq));
            const float s = bf16_round_rne_f32(sinf(freq));
            q[head * hd + i] = bf16_round_rne_f32(bf16_round_rne_f32(a * c) - bf16_round_rne_f32(b * s));
            q[head * hd + half + i] = bf16_round_rne_f32(bf16_round_rne_f32(b * c) + bf16_round_rne_f32(a * s));
        }
        __syncthreads();
    }
    for (int head = 0; head < hkv; ++head) {
        for (int i = tid; i < half; i += blockDim.x) {
            const float a = k[head * hd + i];
            const float b = k[head * hd + half + i];
            const float freq = static_cast<float>(position) *
                expf(-(static_cast<float>(i) / static_cast<float>(half)) * theta_log);
            const float c = bf16_round_rne_f32(cosf(freq));
            const float s = bf16_round_rne_f32(sinf(freq));
            k[head * hd + i] = bf16_round_rne_f32(bf16_round_rne_f32(a * c) - bf16_round_rne_f32(b * s));
            k[head * hd + half + i] = bf16_round_rne_f32(bf16_round_rne_f32(b * c) + bf16_round_rne_f32(a * s));
        }
        __syncthreads();
    }

    hip_bfloat16* cache_k = static_cast<hip_bfloat16*>(layer.kv_cache_k);
    hip_bfloat16* cache_v = static_cast<hip_bfloat16*>(layer.kv_cache_v);
    const int kv_slot = layer.kv_len;
    if (cache_k != nullptr && cache_v != nullptr && kv_slot < layer.kv_max_t) {
        const int base = kv_slot * kv_dim;
        for (int i = tid; i < kv_dim; i += blockDim.x) {
            cache_k[base + i] = static_cast<hip_bfloat16>(k[i]);
            cache_v[base + i] = static_cast<hip_bfloat16>(v[i]);
        }
    }
    __syncthreads();

    const int kv_len = (cache_k != nullptr && cache_v != nullptr) ? (kv_slot + 1) : 1;
    const float attn_scale = rsqrtf(static_cast<float>(hd));
    const int rep = h / hkv;
    for (int head = 0; head < h; ++head) {
        const int kv_head = head / rep;
        float my_acc = 0.0f;
        float my_max = -INFINITY;
        float my_sum = 0.0f;
        const float qv = (tid < hd) ? q[head * hd + tid] : 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float partial = 0.0f;
            if (tid < hd) {
                const float kval = (cache_k != nullptr)
                    ? static_cast<float>(cache_k[t * kv_dim + kv_head * hd + tid])
                    : k[kv_head * hd + tid];
                partial = qv * kval;
            }
            const float score = block_sum(partial, scratch) * attn_scale;
            float vv = 0.0f;
            if (tid < hd) {
                vv = (cache_v != nullptr)
                    ? static_cast<float>(cache_v[t * kv_dim + kv_head * hd + tid])
                    : v[kv_head * hd + tid];
            }
            const float new_max = fmaxf(my_max, score);
            const float old_scale = expf(my_max - new_max);
            const float new_weight = expf(score - new_max);
            my_acc = my_acc * old_scale + new_weight * vv;
            my_sum = my_sum * old_scale + new_weight;
            my_max = new_max;
            __syncthreads();
        }
        if (tid < hd) {
            attn[head * hd + tid] = bf16_round_rne_f32(my_sum > 0.0f ? my_acc / my_sum : 0.0f);
        }
        __syncthreads();
    }

    for (int row = 0; row < hidden; ++row) {
        const float sum = dense_or_int4_dot(
            layer.o_proj_w, int4.o_proj_scale, int4.o_proj_zero,
            row, attn, q_dim, group_size, scratch);
        if (tid == 0) x[row] = bf16_round_rne_f32(x[row] + bf16_round_rne_f32(sum));
        __syncthreads();
    }

    rms_norm_vec(x, layer.post_attn_norm_w, layer.post_attn_norm_eps, hnorm, hidden, scratch);

    for (int e = 0; e < experts; ++e) {
        const float sum = dense_or_int4_dot(
            layer.router_w, nullptr, nullptr, e, hnorm, hidden, group_size, scratch);
        if (tid == 0) router[e] = bf16_round_rne_f32(sum);
        __syncthreads();
    }
    if (tid == 0) {
        float maxv = -INFINITY;
        for (int e = 0; e < experts; ++e) maxv = fmaxf(maxv, router[e]);
        float denom = 0.0f;
        for (int e = 0; e < experts; ++e) {
            probs[e] = expf(router[e] - maxv);
            denom += probs[e];
        }
        for (int e = 0; e < experts; ++e) probs[e] = bf16_round_rne_f32(probs[e] / denom);
        for (int kpos = 0; kpos < top_k; ++kpos) {
            int best_i = 0;
            float best_v = -INFINITY;
            for (int e = 0; e < experts; ++e) {
                if (probs[e] > best_v) {
                    best_v = probs[e];
                    best_i = e;
                }
            }
            topk_idx[kpos] = static_cast<float>(best_i);
            topk_val[kpos] = best_v;
            probs[best_i] = -INFINITY;
        }
        if (layer.norm_topk_prob) {
            float sum = 0.0f;
            for (int kpos = 0; kpos < top_k; ++kpos) sum += topk_val[kpos];
            const float inv = 1.0f / sum;
            for (int kpos = 0; kpos < top_k; ++kpos) topk_val[kpos] = bf16_round_rne_f32(topk_val[kpos] * inv);
        }
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += blockDim.x) moe_out[i] = 0.0f;
    __syncthreads();

    for (int kpos = 0; kpos < top_k; ++kpos) {
        const int expert = static_cast<int>(topk_idx[kpos]);
        for (int row = 0; row < 2 * moe_i; ++row) {
            const float sum = expert_int4_dot(
                layer.experts_gate_up_w,
                int4.experts_gate_up_scale,
                int4.experts_gate_up_zero,
                expert, row, hnorm, 2 * moe_i, hidden, group_size, scratch);
            if (tid == 0) gu[row] = bf16_round_rne_f32(sum);
            __syncthreads();
        }
        for (int i = tid; i < moe_i; i += blockDim.x) {
            const float g = gu[i];
            const float u = gu[moe_i + i];
            mid[i] = bf16_round_rne_f32((g / (1.0f + expf(-g))) * u);
        }
        __syncthreads();
        for (int row = 0; row < hidden; ++row) {
            const float sum = expert_int4_dot(
                layer.experts_down_w,
                int4.experts_down_scale,
                int4.experts_down_zero,
                expert, row, mid, hidden, moe_i, group_size, scratch);
            if (tid == 0) expert_out[row] = bf16_round_rne_f32(sum);
            __syncthreads();
        }
        for (int i = tid; i < hidden; i += blockDim.x) {
            moe_out[i] += topk_val[kpos] * expert_out[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < hidden; i += blockDim.x) {
        output_hidden[i] = static_cast<hip_bfloat16>(bf16_round_rne_f32(x[i] + bf16_round_rne_f32(moe_out[i])));
    }
}

__global__ void qwen3_moe_decode_layer_kernel(
    const DecodeLayerDesc layer,
    const Int4ScaleDesc int4,
    int hidden,
    int position,
    const hip_bfloat16* __restrict__ input_hidden,
    hip_bfloat16* __restrict__ output_hidden,
    float* __restrict__ workspace) {
    extern __shared__ float lds[];
    qwen3_moe_decode_layer_device(
        layer, int4, hidden, position, input_hidden, output_hidden, workspace, lds);
}

__global__ void qwen3_moe_persistent_decode_kernel(
    int num_layers,
    const DecodeLayerDesc* __restrict__ layers,
    const Int4ScaleDesc* __restrict__ int4_descs,
    int hidden,
    int position,
    const hip_bfloat16* __restrict__ input_hidden,
    hip_bfloat16* __restrict__ hidden_ping,
    hip_bfloat16* __restrict__ hidden_pong,
    float* __restrict__ workspace) {
    extern __shared__ float lds[];
    if (blockIdx.x != 0) return;
    if (num_layers <= 0) return;

    const hip_bfloat16* in = input_hidden;
    hip_bfloat16* out = hidden_ping;
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        qwen3_moe_decode_layer_device(
            layers[layer_idx],
            int4_descs[layer_idx],
            hidden,
            position,
            in,
            out,
            workspace,
            lds);
        __syncthreads();
        in = out;
        out = (out == hidden_ping) ? hidden_pong : hidden_ping;
    }

    if (in != hidden_ping) {
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            hidden_ping[i] = in[i];
        }
    }
}

__global__ void qwen3_moe_lm_head_kernel(
    int hidden,
    int vocab,
    float rms_norm_eps,
    const hip_bfloat16* __restrict__ final_hidden,
    const hip_bfloat16* __restrict__ final_norm_w,
    const hip_bfloat16* __restrict__ lm_head_w,
    hip_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ counter) {
    extern __shared__ float lds[];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    float partial_sq = 0.0f;
    for (int i = tid; i < hidden; i += bs) {
        const float v = static_cast<float>(final_hidden[i]);
        partial_sq += v * v;
    }
    lds[tid] = partial_sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) lds[tid] += lds[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(lds[0] / static_cast<float>(hidden) + rms_norm_eps);
    __syncthreads();

    for (;;) {
        __shared__ unsigned int row_s;
        if (tid == 0) row_s = atomicAdd(counter, 1u);
        __syncthreads();
        const int row = static_cast<int>(row_s);
        if (row >= vocab) return;

        float partial = 0.0f;
        const size_t base = static_cast<size_t>(row) * hidden;
        for (int i = tid; i < hidden; i += bs) {
            const float x = bf16_round_rne_f32(
                static_cast<float>(final_hidden[i]) * inv *
                static_cast<float>(final_norm_w[i]));
            partial += static_cast<float>(lm_head_w[base + i]) * x;
        }
        lds[tid] = partial;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) logits[row] = static_cast<hip_bfloat16>(bf16_round_rne_f32(lds[0]));
        __syncthreads();
    }
}

__global__ void qwen3_moe_lm_head_int4_kernel(
    int hidden,
    int vocab,
    float rms_norm_eps,
    const hip_bfloat16* __restrict__ final_hidden,
    const hip_bfloat16* __restrict__ final_norm_w,
    const uint8_t* __restrict__ lm_head_w,
    const hip_bfloat16* __restrict__ lm_head_scale,
    const hip_bfloat16* __restrict__ lm_head_zero,
    int group_size,
    hip_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ counter) {
    extern __shared__ float lds[];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    float partial_sq = 0.0f;
    for (int i = tid; i < hidden; i += bs) {
        const float v = static_cast<float>(final_hidden[i]);
        partial_sq += v * v;
    }
    lds[tid] = partial_sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) lds[tid] += lds[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(lds[0] / static_cast<float>(hidden) + rms_norm_eps);
    __syncthreads();

    for (;;) {
        __shared__ unsigned int row_s;
        if (tid == 0) row_s = atomicAdd(counter, 1u);
        __syncthreads();
        const int row = static_cast<int>(row_s);
        if (row >= vocab) return;

        float partial = 0.0f;
        const int byte_cols = (hidden + 1) / 2;
        const int scale_cols = (hidden + group_size - 1) / group_size;
        const int scale_row = row / group_size;
        int i = tid * 8;
        for (; i + 7 < hidden; i += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(
                &packed,
                lm_head_w + static_cast<size_t>(row) * byte_cols + i / 2,
                sizeof(packed));
            float vals[8];
            int4_dequant_8(
                packed, lm_head_scale, lm_head_zero,
                scale_row * scale_cols + i / group_size, vals);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float x = bf16_round_rne_f32(
                    static_cast<float>(final_hidden[i + j]) * inv *
                    static_cast<float>(final_norm_w[i + j]));
                partial += vals[j] * x;
            }
        }
        for (; i < hidden; ++i) {
            const float x = bf16_round_rne_f32(
                static_cast<float>(final_hidden[i]) * inv *
                static_cast<float>(final_norm_w[i]));
            partial += int4_value(lm_head_w, lm_head_scale, lm_head_zero,
                                  row, i, hidden, group_size) * x;
        }
        lds[tid] = partial;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) logits[row] = static_cast<hip_bfloat16>(bf16_round_rne_f32(lds[0]));
        __syncthreads();
    }
}

} // namespace qwen3_moe
