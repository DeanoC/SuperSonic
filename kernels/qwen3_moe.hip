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

constexpr int Q3_MAX_GRID_BLOCKS = 128;
constexpr int Q3_PROFILE_PHASES = 20;

enum Q3ProfilePhase {
    Q3_PROF_INPUT_LOAD = 0,
    Q3_PROF_INPUT_NORM = 1,
    Q3_PROF_QKV = 2,
    Q3_PROF_Q_NORM_ROPE = 3,
    Q3_PROF_K_NORM_ROPE = 4,
    Q3_PROF_KV_STORE = 5,
    Q3_PROF_ATTN_SCORES = 6,
    Q3_PROF_SOFTMAX = 7,
    Q3_PROF_ATTN_VALUES = 8,
    Q3_PROF_O_PROJ = 9,
    Q3_PROF_ATTN_RESIDUAL = 10,
    Q3_PROF_POST_NORM = 11,
    Q3_PROF_ROUTER = 12,
    Q3_PROF_TOPK = 13,
    Q3_PROF_MOE_ZERO = 14,
    Q3_PROF_EXPERT_GATE_UP = 15,
    Q3_PROF_EXPERT_ACT = 16,
    Q3_PROF_EXPERT_DOWN = 17,
    Q3_PROF_EXPERT_ACCUM = 18,
    Q3_PROF_OUTPUT = 19,
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

__device__ __forceinline__ float q3_wave_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
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

__device__ inline void q3_grid_barrier(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        __threadfence();
        const unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1u) {
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_flag, phase + 1u, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

__device__ inline void q3_grid_barrier_reset_counter(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks,
    unsigned int* work_counter) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        __threadfence();
        const unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1u) {
            __atomic_store_n(work_counter, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_flag, phase + 1u, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

__device__ inline void q3_profile_next(
    uint64_t* profile,
    int layer_idx,
    int phase,
    uint64_t* phase_start) {
    if (profile != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t now = static_cast<uint64_t>(wall_clock64());
        profile[layer_idx * Q3_PROFILE_PHASES + phase] += now - *phase_start;
        *phase_start = now;
    }
}

__device__ void q3_grid_rms_norm(
    const float* in,
    const void* weight,
    float eps,
    float* out,
    int n,
    float* partials,
    float* scratch,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int nb) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    float partial = 0.0f;
    for (int i = blockIdx.x * bs + tid; i < n; i += nb * bs) {
        const float v = in[i];
        partial += v * v;
    }
    const float block = block_sum(partial, scratch);
    if (tid == 0) partials[blockIdx.x] = block;
    q3_grid_barrier(barrier_counter, barrier_flag, nb);

    if (blockIdx.x == 0) {
        float total = 0.0f;
        for (int i = tid; i < nb; i += bs) total += partials[i];
        total = block_sum(total, scratch);
        if (tid == 0) partials[0] = rsqrtf(total / static_cast<float>(n) + eps);
    }
    q3_grid_barrier(barrier_counter, barrier_flag, nb);

    const float inv = partials[0];
    for (int i = blockIdx.x * bs + tid; i < n; i += nb * bs) {
        out[i] = bf16_round_rne_f32(in[i] * inv * load_bf16(weight, i));
    }
}

__device__ void q3_block_head_rms_norm_rope(
    float* vec,
    const void* weight,
    int heads,
    int head_dim,
    float eps,
    int position,
    float rope_theta,
    float* scratch) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int tid = threadIdx.x;
    float* row = vec + head * head_dim;
    float partial = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = row[i];
        partial += v * v;
    }
    const float sum = block_sum(partial, scratch);
    const float inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        row[i] = bf16_round_rne_f32(row[i] * inv * load_bf16(weight, i));
    }
    __syncthreads();

    const int half = head_dim / 2;
    const float theta_log = logf(rope_theta);
    for (int i = tid; i < half; i += blockDim.x) {
        const float a = row[i];
        const float b = row[half + i];
        const float freq = static_cast<float>(position) *
            expf(-(static_cast<float>(i) / static_cast<float>(half)) * theta_log);
        const float c = bf16_round_rne_f32(cosf(freq));
        const float s = bf16_round_rne_f32(sinf(freq));
        row[i] = bf16_round_rne_f32(
            bf16_round_rne_f32(a * c) - bf16_round_rne_f32(b * s));
        row[half + i] = bf16_round_rne_f32(
            bf16_round_rne_f32(b * c) + bf16_round_rne_f32(a * s));
    }
}

__device__ void q3_qkv_rows(
    const DecodeLayerDesc& layer,
    const Int4ScaleDesc& int4,
    const float* hnorm,
    float* q,
    float* k,
    float* v,
    int hidden,
    int q_dim,
    int kv_dim,
    float* scratch,
    unsigned int* work_counter) {
    while (true) {
        __shared__ unsigned int row_s;
        if (threadIdx.x == 0) row_s = atomicAdd(work_counter, 1u);
        __syncthreads();
        const int row = static_cast<int>(row_s);
        if (row >= q_dim + 2 * kv_dim) return;
        if (row < q_dim) {
            const float sum = dense_or_int4_dot(
                layer.q_proj_w, int4.q_proj_scale, int4.q_proj_zero,
                row, hnorm, hidden, int4.group_size, scratch);
            if (threadIdx.x == 0) q[row] = bf16_round_rne_f32(sum);
        } else if (row < q_dim + kv_dim) {
            const int r = row - q_dim;
            const float sum = dense_or_int4_dot(
                layer.k_proj_w, int4.k_proj_scale, int4.k_proj_zero,
                r, hnorm, hidden, int4.group_size, scratch);
            if (threadIdx.x == 0) k[r] = bf16_round_rne_f32(sum);
        } else {
            const int r = row - q_dim - kv_dim;
            const float sum = dense_or_int4_dot(
                layer.v_proj_w, int4.v_proj_scale, int4.v_proj_zero,
                r, hnorm, hidden, int4.group_size, scratch);
            if (threadIdx.x == 0) v[r] = bf16_round_rne_f32(sum);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float q3_wave_dense_or_int4_dot(
    const void* weight,
    const void* scale,
    const void* zero,
    int row,
    const float* x,
    int cols,
    int group_size) {
    float partial = 0.0f;
    if (scale != nullptr) {
        const uint8_t* w = static_cast<const uint8_t*>(weight);
        const hip_bfloat16* s = static_cast<const hip_bfloat16*>(scale);
        const hip_bfloat16* z = static_cast<const hip_bfloat16*>(zero);
        const int byte_cols = (cols + 1) / 2;
        const int scale_cols = (cols + group_size - 1) / group_size;
        const int scale_row = row / group_size;
        const int lane = threadIdx.x % warpSize;
        int c = lane * 8;
        for (; c + 7 < cols; c += warpSize * 8) {
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
        const int lane = threadIdx.x % warpSize;
        for (int c = lane; c < cols; c += warpSize) {
            partial += static_cast<float>(w[base + c]) * x[c];
        }
    }
    return q3_wave_sum(partial);
}

__device__ void q3_qkv_wave_rows(
    const DecodeLayerDesc& layer,
    const Int4ScaleDesc& int4,
    const float* hnorm,
    float* q,
    float* k,
    float* v,
    int hidden,
    int q_dim,
    int kv_dim,
    unsigned int* work_counter) {
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int waves_per_block = blockDim.x / warpSize;
    const int total = q_dim + 2 * kv_dim;
    while (true) {
        __shared__ unsigned int base_s;
        if (threadIdx.x == 0) {
            base_s = atomicAdd(work_counter, static_cast<unsigned int>(waves_per_block));
        }
        __syncthreads();
        if (static_cast<int>(base_s) >= total) return;

        const int row = static_cast<int>(base_s) + wave;
        if (row < total) {
            if (row < q_dim) {
                const float sum = q3_wave_dense_or_int4_dot(
                    layer.q_proj_w, int4.q_proj_scale, int4.q_proj_zero,
                    row, hnorm, hidden, int4.group_size);
                if (lane == 0) q[row] = bf16_round_rne_f32(sum);
            } else if (row < q_dim + kv_dim) {
                const int r = row - q_dim;
                const float sum = q3_wave_dense_or_int4_dot(
                    layer.k_proj_w, int4.k_proj_scale, int4.k_proj_zero,
                    r, hnorm, hidden, int4.group_size);
                if (lane == 0) k[r] = bf16_round_rne_f32(sum);
            } else {
                const int r = row - q_dim - kv_dim;
                const float sum = q3_wave_dense_or_int4_dot(
                    layer.v_proj_w, int4.v_proj_scale, int4.v_proj_zero,
                    r, hnorm, hidden, int4.group_size);
                if (lane == 0) v[r] = bf16_round_rne_f32(sum);
            }
        }
        __syncthreads();
    }
}

__device__ void q3_dense_rows(
    const void* weight,
    const void* scale,
    const void* zero,
    const float* x,
    float* out,
    int rows,
    int cols,
    int group_size,
    float* scratch,
    unsigned int* work_counter,
    bool round_output) {
    while (true) {
        __shared__ unsigned int row_s;
        if (threadIdx.x == 0) row_s = atomicAdd(work_counter, 1u);
        __syncthreads();
        const int row = static_cast<int>(row_s);
        if (row >= rows) return;
        const float sum = dense_or_int4_dot(
            weight, scale, zero, row, x, cols, group_size, scratch);
        if (threadIdx.x == 0) {
            out[row] = round_output ? bf16_round_rne_f32(sum) : sum;
        }
        __syncthreads();
    }
}

__device__ void q3_dense_wave_rows(
    const void* weight,
    const void* scale,
    const void* zero,
    const float* x,
    float* out,
    int rows,
    int cols,
    int group_size,
    unsigned int* work_counter,
    bool round_output) {
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int waves_per_block = blockDim.x / warpSize;
    while (true) {
        __shared__ unsigned int base_s;
        if (threadIdx.x == 0) {
            base_s = atomicAdd(work_counter, static_cast<unsigned int>(waves_per_block));
        }
        __syncthreads();
        if (static_cast<int>(base_s) >= rows) return;

        const int row = static_cast<int>(base_s) + wave;
        if (row < rows) {
            const float sum = q3_wave_dense_or_int4_dot(
                weight, scale, zero, row, x, cols, group_size);
            if (lane == 0) out[row] = round_output ? bf16_round_rne_f32(sum) : sum;
        }
        __syncthreads();
    }
}

__device__ void q3_expert_batch_wave_rows(
    const void* weight,
    const void* scale,
    const void* zero,
    const float* topk_idx,
    int top_k,
    const float* x,
    bool x_per_topk,
    float* out,
    int rows,
    int cols,
    int group_size,
    unsigned int* work_counter) {
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int waves_per_block = blockDim.x / warpSize;
    const int total = top_k * rows;
    const uint8_t* w = static_cast<const uint8_t*>(weight);
    const hip_bfloat16* s = static_cast<const hip_bfloat16*>(scale);
    const hip_bfloat16* z = static_cast<const hip_bfloat16*>(zero);
    const int byte_cols = (cols + 1) / 2;
    const int scale_rows = (rows + group_size - 1) / group_size;
    const int scale_cols = (cols + group_size - 1) / group_size;

    while (true) {
        __shared__ unsigned int base_s;
        if (threadIdx.x == 0) {
            base_s = atomicAdd(work_counter, static_cast<unsigned int>(waves_per_block));
        }
        __syncthreads();
        if (static_cast<int>(base_s) >= total) return;

        const int item = static_cast<int>(base_s) + wave;
        if (item < total) {
            const int kpos = item / rows;
            const int row = item - kpos * rows;
            const int expert = static_cast<int>(topk_idx[kpos]);
            const float* x_row = x + (x_per_topk ? kpos * cols : 0);
            const size_t packed_base =
                (static_cast<size_t>(expert) * rows + row) * byte_cols;
            const int scale_base =
                (expert * scale_rows + row / group_size) * scale_cols;

            float partial = 0.0f;
            int c = lane * 8;
            for (; c + 7 < cols; c += warpSize * 8) {
                uint32_t packed;
                __builtin_memcpy(&packed, w + packed_base + c / 2, sizeof(packed));
                float vals[8];
                int4_dequant_8(packed, s, z, scale_base + c / group_size, vals);
                partial += vals[0] * x_row[c + 0] + vals[1] * x_row[c + 1] +
                           vals[2] * x_row[c + 2] + vals[3] * x_row[c + 3] +
                           vals[4] * x_row[c + 4] + vals[5] * x_row[c + 5] +
                           vals[6] * x_row[c + 6] + vals[7] * x_row[c + 7];
            }
            for (; c < cols; ++c) {
                partial +=
                    int4_value_3d(w, s, z, expert, row, c, rows, cols, group_size) *
                    x_row[c];
            }
            const float sum = q3_wave_sum(partial);
            if (lane == 0) out[kpos * rows + row] = bf16_round_rne_f32(sum);
        }
        __syncthreads();
    }
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
    float* __restrict__ workspace,
    unsigned int* __restrict__ sync,
    uint64_t* __restrict__ profile) __attribute__((launch_bounds(256, 1))) {
    extern __shared__ float lds[];
    if (num_layers <= 0) return;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;
    unsigned int* work_counter = sync + 0;
    unsigned int* barrier_counter = sync + 1;
    unsigned int* barrier_flag = sync + 2;

    const hip_bfloat16* in = input_hidden;
    hip_bfloat16* out = hidden_ping;
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        uint64_t phase_start = 0;
        if (profile != nullptr && blockIdx.x == 0 && tid == 0) {
            phase_start = static_cast<uint64_t>(wall_clock64());
        }
        const DecodeLayerDesc layer = layers[layer_idx];
        const Int4ScaleDesc int4 = int4_descs[layer_idx];
        const int hd = layer.head_dim;
        const int h = layer.num_heads;
        const int hkv = layer.num_kv_heads;
        const int q_dim = h * hd;
        const int kv_dim = hkv * hd;
        const int group_size = int4.group_size;
        const int top_k = layer.top_k;
        const int experts = layer.num_experts;
        const int moe_i = layer.moe_intermediate_size;
        const int kv_slot = layer.kv_len;
        const int kv_len = (layer.kv_cache_k != nullptr && layer.kv_cache_v != nullptr)
            ? (kv_slot + 1)
            : 1;

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
        const int OFF_MID = OFF_GU + top_k * 2 * moe_i;
        const int OFF_EXPERT_OUT = OFF_MID + top_k * moe_i;
        const int OFF_MOE_OUT = OFF_EXPERT_OUT + top_k * hidden;
        const int OFF_SCORES = OFF_MOE_OUT + hidden;
        const int OFF_PARTIALS = OFF_SCORES + h * layer.kv_max_t;

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
        float* scores = workspace + OFF_SCORES;
        float* partials = workspace + OFF_PARTIALS;

        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) {
            x[i] = static_cast<float>(in[i]);
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_INPUT_LOAD, &phase_start);

        q3_grid_rms_norm(
            x, layer.input_norm_w, layer.input_norm_eps, hnorm, hidden,
            partials, lds, barrier_counter, barrier_flag, nb);
        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        q3_profile_next(profile, layer_idx, Q3_PROF_INPUT_NORM, &phase_start);

        q3_qkv_wave_rows(layer, int4, hnorm, q, k, v, hidden, q_dim, kv_dim, work_counter);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_QKV, &phase_start);

        q3_block_head_rms_norm_rope(
            q, layer.q_norm_w, h, hd, layer.input_norm_eps,
            position, layer.rope_theta, lds);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_Q_NORM_ROPE, &phase_start);

        q3_block_head_rms_norm_rope(
            k, layer.k_norm_w, hkv, hd, layer.input_norm_eps,
            position, layer.rope_theta, lds);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_K_NORM_ROPE, &phase_start);

        hip_bfloat16* cache_k = static_cast<hip_bfloat16*>(layer.kv_cache_k);
        hip_bfloat16* cache_v = static_cast<hip_bfloat16*>(layer.kv_cache_v);
        if (cache_k != nullptr && cache_v != nullptr && kv_slot < layer.kv_max_t) {
            const int base = kv_slot * kv_dim;
            for (int i = blockIdx.x * bs + tid; i < kv_dim; i += nb * bs) {
                cache_k[base + i] = static_cast<hip_bfloat16>(k[i]);
                cache_v[base + i] = static_cast<hip_bfloat16>(v[i]);
            }
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_KV_STORE, &phase_start);

        const float attn_scale = rsqrtf(static_cast<float>(hd));
        const int rep = h / hkv;
        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        while (true) {
            __shared__ unsigned int item_s;
            if (tid == 0) item_s = atomicAdd(work_counter, 1u);
            __syncthreads();
            const int item = static_cast<int>(item_s);
            if (item >= h * kv_len) break;
            const int t = item % kv_len;
            const int head = item / kv_len;
            const int kv_head = head / rep;
            float partial = 0.0f;
            for (int d = tid; d < hd; d += bs) {
                const float kval = (cache_k != nullptr)
                    ? static_cast<float>(cache_k[t * kv_dim + kv_head * hd + d])
                    : k[kv_head * hd + d];
                partial += q[head * hd + d] * kval;
            }
            const float score = block_sum(partial, lds) * attn_scale;
            if (tid == 0) scores[head * layer.kv_max_t + t] = score;
            __syncthreads();
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_ATTN_SCORES, &phase_start);

        if (blockIdx.x < static_cast<unsigned int>(h)) {
            const int head = blockIdx.x;
            float* row = scores + head * layer.kv_max_t;
            float local_max = -INFINITY;
            for (int t = tid; t < kv_len; t += bs) local_max = fmaxf(local_max, row[t]);
            lds[tid] = local_max;
            __syncthreads();
            for (int s = bs / 2; s > 0; s >>= 1) {
                if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
                __syncthreads();
            }
            const float maxv = lds[0];
            __syncthreads();
            float local_sum = 0.0f;
            for (int t = tid; t < kv_len; t += bs) {
                const float e = expf(row[t] - maxv);
                row[t] = e;
                local_sum += e;
            }
            const float sum = block_sum(local_sum, lds);
            const float inv = 1.0f / sum;
            for (int t = tid; t < kv_len; t += bs) row[t] = bf16_round_rne_f32(row[t] * inv);
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_SOFTMAX, &phase_start);

        for (int item = blockIdx.x * bs + tid; item < q_dim; item += nb * bs) {
            const int d = item % hd;
            const int head = item / hd;
            const int kv_head = head / rep;
            const float* row = scores + head * layer.kv_max_t;
            float acc = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                const float vv = (cache_v != nullptr)
                    ? static_cast<float>(cache_v[t * kv_dim + kv_head * hd + d])
                    : v[kv_head * hd + d];
                acc += row[t] * vv;
            }
            attn[item] = bf16_round_rne_f32(acc);
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_ATTN_VALUES, &phase_start);

        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        q3_dense_wave_rows(
            layer.o_proj_w, int4.o_proj_scale, int4.o_proj_zero,
            attn, hnorm, hidden, q_dim, group_size, work_counter, true);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_O_PROJ, &phase_start);

        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) {
            x[i] = bf16_round_rne_f32(x[i] + hnorm[i]);
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_ATTN_RESIDUAL, &phase_start);

        q3_grid_rms_norm(
            x, layer.post_attn_norm_w, layer.post_attn_norm_eps, hnorm, hidden,
            partials, lds, barrier_counter, barrier_flag, nb);
        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        q3_profile_next(profile, layer_idx, Q3_PROF_POST_NORM, &phase_start);

        q3_dense_rows(
            layer.router_w, nullptr, nullptr, hnorm, router, experts, hidden,
            group_size, lds, work_counter, true);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_ROUTER, &phase_start);

        if (blockIdx.x == 0) {
            if (tid == 0) {
                float maxv = -INFINITY;
                for (int e = 0; e < experts; ++e) maxv = fmaxf(maxv, router[e]);
                float denom = 0.0f;
                for (int e = 0; e < experts; ++e) {
                    probs[e] = expf(router[e] - maxv);
                    denom += probs[e];
                }
                for (int e = 0; e < experts; ++e) {
                    probs[e] = bf16_round_rne_f32(probs[e] / denom);
                }
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
                    for (int kpos = 0; kpos < top_k; ++kpos) {
                        topk_val[kpos] = bf16_round_rne_f32(topk_val[kpos] * inv);
                    }
                }
            }
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_TOPK, &phase_start);

        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) moe_out[i] = 0.0f;
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_MOE_ZERO, &phase_start);

        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        q3_expert_batch_wave_rows(
            layer.experts_gate_up_w,
            int4.experts_gate_up_scale,
            int4.experts_gate_up_zero,
            topk_idx, top_k, hnorm, false, gu, 2 * moe_i, hidden,
            group_size, work_counter);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_EXPERT_GATE_UP, &phase_start);

        for (int item = blockIdx.x * bs + tid; item < top_k * moe_i; item += nb * bs) {
            const int kpos = item / moe_i;
            const int i = item - kpos * moe_i;
            const float* gu_k = gu + kpos * 2 * moe_i;
            const float g = gu_k[i];
            const float u = gu_k[moe_i + i];
            mid[kpos * moe_i + i] = bf16_round_rne_f32((g / (1.0f + expf(-g))) * u);
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_EXPERT_ACT, &phase_start);

        q3_grid_barrier_reset_counter(
            barrier_counter, barrier_flag, nb, work_counter);
        q3_expert_batch_wave_rows(
            layer.experts_down_w,
            int4.experts_down_scale,
            int4.experts_down_zero,
            topk_idx, top_k, mid, true, expert_out, hidden, moe_i,
            group_size, work_counter);
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_EXPERT_DOWN, &phase_start);

        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) {
            float acc = 0.0f;
            for (int kpos = 0; kpos < top_k; ++kpos) {
                acc += topk_val[kpos] * expert_out[kpos * hidden + i];
            }
            moe_out[i] = acc;
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_EXPERT_ACCUM, &phase_start);

        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) {
            out[i] = static_cast<hip_bfloat16>(
                bf16_round_rne_f32(x[i] + bf16_round_rne_f32(moe_out[i])));
        }
        q3_grid_barrier(barrier_counter, barrier_flag, nb);
        q3_profile_next(profile, layer_idx, Q3_PROF_OUTPUT, &phase_start);

        in = out;
        out = (out == hidden_ping) ? hidden_pong : hidden_ping;
    }

    if (in != hidden_ping) {
        for (int i = blockIdx.x * bs + tid; i < hidden; i += nb * bs) {
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
