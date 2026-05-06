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
// Prefill-specific helper kernels.
// Kept in a separate compilation unit to avoid touching the decode megakernel
// files (hipcc codegen sensitivity on gfx1150).


// ---- Type conversion helpers (matching the megakernel's conventions) ----

template <typename T>
__device__ inline float pfx_to_float(T value);

template <>
__device__ inline float pfx_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float pfx_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float pfx_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T pfx_from_float(float value);

template <>
__device__ inline __half pfx_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float pfx_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 pfx_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float pfx_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
    }
    return value;
}

// ---- Kernel 1: Element-wise addition ----
// out[i] = lhs[i] + rhs[i]

template <typename T>
__global__ void pfx_element_add_kernel(
    size_t total_elems,
    const T* lhs,
    const T* rhs,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    out[idx] = pfx_from_float<T>(pfx_to_float(lhs[idx]) + pfx_to_float(rhs[idx]));
}

// ---- Kernel 2: RoPE for prefill ----
// Applies RoPE in-place on tensor with layout [seq_len, num_heads, head_dim].
// Only the first rotary_dim dimensions of each head are rotated.
// cos_table/sin_table: [max_pos, half_rot] where half_rot = rotary_dim / 2.
// Qwen3.6 GGUF uses interleaved RoPE pairs over the rotary slice:
// new[2*i]     = old[2*i] * cos[i] - old[2*i + 1] * sin[i]
// new[2*i + 1] = old[2*i] * sin[i] + old[2*i + 1] * cos[i]

template <typename T>
__global__ void pfx_apply_rope_prefill_kernel(
    int seq_len,
    int num_heads,
    int head_dim,
    int half_rot,       // rotary_dim / 2
    const T* cos_table, // [max_pos, half_rot]
    const T* sin_table, // [max_pos, half_rot]
    T* data             // [seq_len, num_heads, head_dim] — modified in-place
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    if (idx >= total) return;

    const int i   = static_cast<int>(idx % half_rot);
    const int h   = static_cast<int>((idx / half_rot) % num_heads);
    const int pos = static_cast<int>(idx / (static_cast<size_t>(half_rot) * num_heads));

    const size_t cos_off = static_cast<size_t>(pos) * half_rot + i;
    const float c = pfx_to_float(cos_table[cos_off]);
    const float s = pfx_to_float(sin_table[cos_off]);

    const size_t base = static_cast<size_t>(pos) * num_heads * head_dim
                      + static_cast<size_t>(h) * head_dim;
    const int d0 = i;
    const int d1 = half_rot + i;
    const float x0 = pfx_to_float(data[base + d0]);
    const float x1 = pfx_to_float(data[base + d1]);

    data[base + d0] = pfx_from_float<T>(x0 * c - x1 * s);
    data[base + d1] = pfx_from_float<T>(x0 * s + x1 * c);
}

// Same as pfx_apply_rope_prefill_kernel, but cos/sin rows are selected by
// per-slot original prompt positions. Sparse SpecPrefill uses compact
// sequence slots while preserving absolute RoPE positions.
template <typename T>
__global__ void pfx_apply_rope_prefill_indirect_kernel(
    int seq_len,
    int num_heads,
    int head_dim,
    int half_rot,
    const T* cos_table,
    const T* sin_table,
    const int* pos_ids,
    T* data
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    if (idx >= total) return;

    const int i    = static_cast<int>(idx % half_rot);
    const int h    = static_cast<int>((idx / half_rot) % num_heads);
    const int slot = static_cast<int>(idx / (static_cast<size_t>(half_rot) * num_heads));
    const int pos  = pos_ids[slot];

    const size_t cos_off = static_cast<size_t>(pos) * half_rot + i;
    const float c = pfx_to_float(cos_table[cos_off]);
    const float s = pfx_to_float(sin_table[cos_off]);

    const size_t base = static_cast<size_t>(slot) * num_heads * head_dim
                      + static_cast<size_t>(h) * head_dim;
    const int d0 = i;
    const int d1 = half_rot + i;
    const float x0 = pfx_to_float(data[base + d0]);
    const float x1 = pfx_to_float(data[base + d1]);

    data[base + d0] = pfx_from_float<T>(x0 * c - x1 * s);
    data[base + d1] = pfx_from_float<T>(x0 * s + x1 * c);
}

// ---- Kernel 3: Transpose [S, H, D] <-> [H, S, D] ----
// src layout: [S, H, D] — element(s, h, d) at s*H*D + h*D + d
// dst layout: [H, S, D] — element(h, s, d) at h*S*D + s*D + d

template <typename T>
__global__ void pfx_transpose_shd_hsd_kernel(
    int S, int H, int D,
    const T* src,
    T* dst
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * H * D;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % D);
    const int h = static_cast<int>((idx / D) % H);
    const int s = static_cast<int>(idx / (static_cast<size_t>(D) * H));

    const size_t dst_off = static_cast<size_t>(h) * S * D + static_cast<size_t>(s) * D + d;
    dst[dst_off] = src[idx];
}

// ---- Kernel 4: Transpose [S, C] -> [C, pad + S] with left zero-padding ----
// For causal conv1d input preparation.
// src: [S, C] row-major
// dst: [C, pad + S] row-major, first 'pad' elements per channel are zero.

template <typename T>
__global__ void pfx_transpose_pad_conv_kernel(
    int S, int C, int pad,
    const T* src,
    T* dst
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * C;
    if (idx >= total) return;

    const int c = static_cast<int>(idx % C);
    const int s = static_cast<int>(idx / C);

    const size_t total_len = pad + S;
    const size_t dst_off = static_cast<size_t>(c) * total_len + pad + s;
    dst[dst_off] = src[idx];
}

// ---- Kernel 5: Extract conv state from QKV output ----
// After prefill, save the last (kern-1) values per channel for the decode conv state.
// src: [S, C] row-major (QKV matmul output, pre-conv BF16 values)
// dst: [C, kern-1] row-major
// Extracts rows [S - kern + 1, S) and transposes to [C, kern-1].

template <typename T>
__global__ void pfx_extract_conv_state_kernel(
    int S, int C, int kern_minus_1,
    const T* src,
    T* dst
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(kern_minus_1) * C;
    if (idx >= total) return;

    const int c = static_cast<int>(idx % C);
    const int t = static_cast<int>(idx / C);  // 0..kern_minus_1-1

    const int src_row = S - kern_minus_1 + t;
    const size_t src_off = static_cast<size_t>(src_row) * C + c;
    const size_t dst_off = static_cast<size_t>(c) * kern_minus_1 + t;
    dst[dst_off] = src[src_off];
}

// ---- Kernel 6: Fused sigmoid-gate multiply ----
// out[i] = data[i] * sigmoid(gate[i])

template <typename T>
__global__ void pfx_sigmoid_mul_kernel(
    size_t total_elems,
    const T* data,
    const T* gate,
    T* out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    const float g = pfx_to_float(gate[idx]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    out[idx] = pfx_from_float<T>(pfx_to_float(data[idx]) * sigmoid_g);
}

// ---- Kernel 7: Compute beta and g for delta recurrent ----
// beta[h, t] = sigmoid(B[t, h])
// g[h, t] = -softplus(A[t, h] + dt_bias[h]) * a_log_exp[h]
// Inputs: B [S, nv], A [S, nv] in BF16; dt_bias [nv], a_log_exp [nv] in BF16
// Outputs: beta [nv, S], g [nv, S] in BF16 (transposed)

template <typename T>
__global__ void pfx_compute_beta_g_kernel(
    int seq_len,
    int nv,
    const T* B,          // [seq_len, nv]
    const T* A,          // [seq_len, nv]
    const T* dt_bias,    // [nv]
    const T* a_log_exp,  // [nv]
    T* beta,             // [nv, seq_len]
    T* g                 // [nv, seq_len]
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(seq_len) * nv;
    if (idx >= total) return;

    const int t = static_cast<int>(idx / nv);
    const int h = static_cast<int>(idx % nv);

    // B[t, h] → sigmoid → beta[h, t]
    const float b_val = pfx_to_float(B[idx]);
    const float beta_val = 1.0f / (1.0f + expf(-b_val));
    beta[static_cast<size_t>(h) * seq_len + t] = pfx_from_float<T>(beta_val);

    // A[t, h] + dt_bias[h] → softplus → * a_log_exp[h] → negate → g[h, t]
    const float a_val = pfx_to_float(A[idx]);
    const float dt = pfx_to_float(dt_bias[h]);
    const float ale = pfx_to_float(a_log_exp[h]);
    const float sp = logf(1.0f + expf(a_val + dt));
    g[static_cast<size_t>(h) * seq_len + t] = pfx_from_float<T>(-sp * ale);
}

// ---- Kernel 8: Split gated Q projection ----
// src: [S, num_heads, 2*head_dim] — each head has [query(hd) | gate(hd)]
// query_out: [S, num_heads, head_dim]
// gate_out:  [S, num_heads, head_dim]

template <typename T>
__global__ void pfx_split_qgate_kernel(
    int S, int num_heads, int head_dim,
    const T* src,
    T* query_out,
    T* gate_out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * num_heads * head_dim;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % head_dim);
    const int h = static_cast<int>((idx / head_dim) % num_heads);
    const int s = static_cast<int>(idx / (static_cast<size_t>(head_dim) * num_heads));

    const size_t src_base = static_cast<size_t>(s) * num_heads * head_dim * 2
                          + static_cast<size_t>(h) * head_dim * 2;
    query_out[idx] = src[src_base + d];
    gate_out[idx] = src[src_base + head_dim + d];
}

// ---- Kernel 9: Split interleaved QKV ----
// src: [S, qkv_dim] where qkv_dim = [Q(key_dim) | K(key_dim) | V(val_dim)]
// Splits into separate contiguous Q, K, V buffers.
// Q_out: [S, key_dim], K_out: [S, key_dim], V_out: [S, val_dim]

template <typename T>
__global__ void pfx_split_qkv_kernel(
    int S, int key_dim, int val_dim,
    const T* src,
    T* Q_out,
    T* K_out,
    T* V_out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int qkv_dim = key_dim * 2 + val_dim;
    const size_t total = static_cast<size_t>(S) * qkv_dim;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / qkv_dim);
    const int c = static_cast<int>(idx % qkv_dim);
    const T val = src[idx];

    if (c < key_dim) {
        Q_out[static_cast<size_t>(s) * key_dim + c] = val;
    } else if (c < key_dim * 2) {
        K_out[static_cast<size_t>(s) * key_dim + (c - key_dim)] = val;
    } else {
        V_out[static_cast<size_t>(s) * val_dim + (c - key_dim * 2)] = val;
    }
}

// ---- Kernel 10: Repeat interleave along head dimension ----
// src: [S, n_heads, head_dim] -> dst: [S, n_heads * repeats, head_dim].
// Qwen gated-delta-net expands key/query heads with repeat_interleave:
// source heads appear in contiguous repeat blocks.

template <typename T>
__global__ void pfx_repeat_interleave_heads_kernel(
    int S, int n_heads, int head_dim, int repeats,
    const T* src,
    T* dst
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int out_heads = n_heads * repeats;
    const size_t total = static_cast<size_t>(S) * out_heads * head_dim;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % head_dim);
    const int oh = static_cast<int>((idx / head_dim) % out_heads);
    const int s = static_cast<int>(idx / (static_cast<size_t>(head_dim) * out_heads));
    const int src_h = oh / repeats;

    const size_t src_off = static_cast<size_t>(s) * n_heads * head_dim
                         + static_cast<size_t>(src_h) * head_dim + d;
    dst[idx] = src[src_off];
}

// ---- Kernel 11: per-block cosine(block_mean_K, last_K) for SpecPrefill ----
// PFlash-style importance scoring. k_cache layout:
//   [1, kv_heads, cap, head_dim], with per-head stride cap * head_dim.

template <typename T>
__global__ void pfx_pflash_cosine_score_kernel(
    const T* __restrict__ k_cache,
    float* __restrict__ scores,
    int n_pos,
    int kv_heads,
    int cap,
    int head_dim,
    int block_size,
    int n_blocks,
    int last_pos
) {
    const int lane = threadIdx.x;
    if (lane >= warpSize) return;

    const int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    const int block_start = block_idx * block_size;
    int block_end = block_start + block_size;
    if (block_end > n_pos) block_end = n_pos;
    const int block_len = block_end - block_start;
    if (block_len <= 0) {
        if (lane == 0) scores[block_idx] = 0.0f;
        return;
    }

    const int kv_dim = kv_heads * head_dim;
    float dot_acc = 0.0f;
    float nb_acc = 0.0f;
    float nl_acc = 0.0f;

    for (int d = lane; d < kv_dim; d += warpSize) {
        const int h = d / head_dim;
        const int dim_in_head = d - h * head_dim;
        const size_t base_h = static_cast<size_t>(h) * cap * head_dim
                            + static_cast<size_t>(dim_in_head);

        const float last_v = pfx_to_float(
            k_cache[base_h + static_cast<size_t>(last_pos) * head_dim]);

        float sum = 0.0f;
        for (int pos = block_start; pos < block_end; ++pos) {
            sum += pfx_to_float(k_cache[base_h + static_cast<size_t>(pos) * head_dim]);
        }
        const float mean = sum / static_cast<float>(block_len);

        dot_acc += mean * last_v;
        nb_acc += mean * mean;
        nl_acc += last_v * last_v;
    }

    const float dot = pfx_wave_sum(dot_acc);
    const float nb = pfx_wave_sum(nb_acc);
    const float nl = pfx_wave_sum(nl_acc);

    if (lane == 0) {
        float denom = sqrtf(nb) * sqrtf(nl);
        if (denom < 1e-12f) denom = 1e-12f;
        scores[block_idx] = dot / denom;
    }
}
