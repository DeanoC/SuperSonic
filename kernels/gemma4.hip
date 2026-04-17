// Gemma 4 decode primitives.
//
// Kept in a **separate compilation unit** from full_attention.hip / _4b.hip:
// hipcc's codegen on gfx1150 is fragile and merging Gemma-specific kernels
// into the Qwen files has caused hard-to-diagnose regressions in the past
// (see feedback_hipcc_codegen in the project memory). Every Gemma 4 kernel
// we need lives in this file — duplication of simple helpers across
// gemma4.hip / full_attention.hip is intentional and preferred over risking
// Qwen kernels.
//
// Symbols use the `g4_` prefix (kernels) and `dotcache_gemma4_hip_` prefix
// (extern "C" bridge entry points) to avoid collisions with the Qwen
// kernels linked into the same static library.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Type-conversion helpers (F32 ↔ T)
// -----------------------------------------------------------------------------

template <typename T>
__device__ inline float g4_to_float(T value);

template <>
__device__ inline float g4_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float g4_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float g4_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T g4_from_float(float value);

template <>
__device__ inline __half g4_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float g4_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 g4_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

// -----------------------------------------------------------------------------
// RMSNorm (Gemma 4 variant — plain `weight * normed`, NO (w + 1) offset).
// Single-row single-block launch: grid=(1,1,1), block=(256,1,1).
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_rms_norm_kernel(
    int n_cols,
    float eps,
    const T* __restrict__ xs,      // [n_cols]
    const T* __restrict__ weight,  // [n_cols] (nullptr allowed → acts like with_scale=False)
    T* __restrict__ out            // [n_cols]
) {
    const int tid = threadIdx.x;

    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = g4_to_float(xs[col]);
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

    const bool has_weight = (weight != nullptr);
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = g4_to_float(xs[col]);
        const float normed = x * shared_inv_rms;
        const float w = has_weight ? g4_to_float(weight[col]) : 1.0f;
        out[col] = g4_from_float<T>(normed * w);
    }
}

// -----------------------------------------------------------------------------
// Work-stealing matrix-vector multiply (row-major weight, single-token activation).
// Computes: out[i] = dot(W[i, :], x) for i in [0, out_dim).
// Grid: `num_blocks` blocks compete for rows via an atomic counter in `row_counter`.
// Block size: 256 threads. Each block reduces one row at a time.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_matvec_workstealing_kernel(
    int in_dim,
    int out_dim,
    const T* __restrict__ x,        // [in_dim]
    const T* __restrict__ W,        // [out_dim, in_dim] row-major
    T* __restrict__ out,            // [out_dim]
    unsigned int* __restrict__ row_counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    while (true) {
        int row;
        if (tid == 0) {
            row = static_cast<int>(atomicAdd(row_counter, 1u));
        }
        __shared__ int shared_row;
        if (tid == 0) shared_row = row;
        __syncthreads();
        row = shared_row;
        if (row >= out_dim) return;

        float partial = 0.0f;
        const T* w_row = W + static_cast<size_t>(row) * static_cast<size_t>(in_dim);
        for (int i = tid; i < in_dim; i += block_size) {
            partial += g4_to_float(w_row[i]) * g4_to_float(x[i]);
        }

        __shared__ float shared_sum[256];
        shared_sum[tid] = partial;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            out[row] = g4_from_float<T>(shared_sum[0]);
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// GeLU-tanh gated multiply.
// out[i] = gelu_tanh(gate[i]) * up[i]
// GELU-tanh formula (per Gemma 4's hidden_activation="gelu_pytorch_tanh"):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_gelu_tanh_gate_mul_kernel(
    size_t n,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = g4_to_float(gate[idx]);
    const float u = g4_to_float(up[idx]);
    constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kCoef = 0.044715f;
    const float inner = kSqrt2OverPi * (g + kCoef * g * g * g);
    const float gelu = 0.5f * g * (1.0f + tanhf(inner));
    out[idx] = g4_from_float<T>(gelu * u);
}

// -----------------------------------------------------------------------------
// Gemma-style RoPE apply for a SINGLE decode token.
// Gemma uses "split-half" RoPE (not interleaved pairs):
//   cos/sin tables have shape [max_pos, head_dim] where
//   cos[p, i] == cos[p, i + half] and sim for sin (half-duplicated).
//   apply(x) = x * cos + rotate_half(x) * sin
//     rotate_half(x) = concat(-x[half:], x[:half])
// Input layout: `x` is [num_heads, head_dim] contiguous (single token).
// If `rotary_dim < head_dim`, only the first `rotary_dim` columns are rotated;
// the rest are passed through. For Gemma 4 sliding layers rotary_dim == head_dim
// (full rotation); for full layers with partial_rotary_factor=0.25 only the
// first quarter is rotated.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_rope_split_half_decode_kernel(
    int num_heads,
    int head_dim,
    int rotary_dim,          // may be < head_dim; must be even
    int position,            // position in the sequence for cos/sin lookup
    const T* __restrict__ cos_table,  // [max_pos, rotary_dim]
    const T* __restrict__ sin_table,  // [max_pos, rotary_dim]
    T* __restrict__ x                 // [num_heads, head_dim] in-place
) {
    const int half = rotary_dim / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_heads) * half;
    if (idx >= total) return;

    const int i = static_cast<int>(idx % half);
    const int h = static_cast<int>(idx / half);

    const size_t cos_off = static_cast<size_t>(position) * rotary_dim + i;
    const float c = g4_to_float(cos_table[cos_off]);
    const float s = g4_to_float(sin_table[cos_off]);

    T* row = x + static_cast<size_t>(h) * head_dim;
    const float x0 = g4_to_float(row[i]);         // first half
    const float x1 = g4_to_float(row[i + half]);  // second half
    // rotate_half: first-half output uses (-x_second); second-half output uses x_first
    const float y0 = x0 * c - x1 * s;
    const float y1 = x1 * c + x0 * s;
    row[i] = g4_from_float<T>(y0);
    row[i + half] = g4_from_float<T>(y1);
}

// -----------------------------------------------------------------------------
// Append one token's K/V into a pre-allocated cache slot at `pos`.
// Cache layout: [num_kv_heads, max_T, head_dim] (matches HF dump layout).
// `k_in` / `v_in` layout: [num_kv_heads, head_dim] contiguous.
// One thread per element of K and V independently.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Sliding-Window Attention for a single decode token (Gemma 4).
//
// Inputs (all post-norm, post-RoPE, and with KV already appended at position
// `kv_len - 1`, so the cache holds `kv_len` valid entries):
//   Q          [num_q_heads, head_dim]
//   K_cache    [num_kv_heads, max_T, head_dim]
//   V_cache    [num_kv_heads, max_T, head_dim]
//
// Output:
//   out        [num_q_heads, head_dim]
//
// Scratch (caller-allocated, F32):
//   scores     [num_q_heads, max_T]
//
// Gemma 4 uses `scaling = 1.0` (no 1/sqrt(d_k)); the caller passes that scalar.
// If `sliding_window <= 0` the entire cache is attended (behaves as full attn).
// If `num_kv_heads < num_q_heads`, GQA is applied (one KV head services
// `num_q_heads / num_kv_heads` consecutive Q heads).
//
// The three kernels are launched in sequence from the bridge; no cross-kernel
// synchronization is needed beyond hipDeviceSynchronize inside the caller.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_attn_scores_kernel(
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,           // number of valid tokens currently in the cache
    int max_T,
    int sliding_window,   // <= 0 means no window
    float scale,          // Gemma 4 passes 1.0
    const T* __restrict__ Q,
    const T* __restrict__ K_cache,
    float* __restrict__ scores
) {
    const int q_h = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (q_h >= num_q_heads || t >= kv_len) return;

    // SWA mask: last valid token is at kv_len-1. Window includes tokens
    // within `sliding_window - 1` positions before last (and last itself).
    const int last = kv_len - 1;
    const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
    if (t < min_t) {
        scores[static_cast<size_t>(q_h) * max_T + t] = -INFINITY;
        return;
    }

    const int num_q_per_kv = num_q_heads / num_kv_heads;
    const int kv_h = q_h / num_q_per_kv;

    const T* q_row = Q + static_cast<size_t>(q_h) * head_dim;
    const T* k_row = K_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;

    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        acc += g4_to_float(q_row[d]) * g4_to_float(k_row[d]);
    }
    scores[static_cast<size_t>(q_h) * max_T + t] = acc * scale;
}

__global__ void g4_attn_softmax_kernel(
    int num_q_heads,
    int kv_len,
    int max_T,
    float* __restrict__ scores
) {
    const int q_h = blockIdx.x;
    const int tid = threadIdx.x;
    if (q_h >= num_q_heads) return;

    float* row = scores + static_cast<size_t>(q_h) * max_T;

    // Pass 1: block-wide max (stabilizer).
    __shared__ float s_red[256];
    float partial_max = -INFINITY;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        partial_max = fmaxf(partial_max, row[t]);
    }
    s_red[tid] = partial_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
        __syncthreads();
    }
    const float mx = s_red[0];
    __syncthreads();

    // Pass 2: exp + block-wide sum.
    float partial_sum = 0.0f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        const float v = expf(row[t] - mx);
        row[t] = v;
        partial_sum += v;
    }
    s_red[tid] = partial_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / s_red[0];
    __syncthreads();

    // Pass 3: normalize.
    for (int t = tid; t < kv_len; t += blockDim.x) {
        row[t] *= inv_sum;
    }
}

template <typename T>
__global__ void g4_attn_value_aggregate_kernel(
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int max_T,
    const float* __restrict__ scores,
    const T* __restrict__ V_cache,
    T* __restrict__ out
) {
    const int q_h = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (q_h >= num_q_heads || d >= head_dim) return;

    const int num_q_per_kv = num_q_heads / num_kv_heads;
    const int kv_h = q_h / num_q_per_kv;

    float acc = 0.0f;
    const float* p_row = scores + static_cast<size_t>(q_h) * max_T;
    for (int t = 0; t < kv_len; ++t) {
        const float p = p_row[t];
        const T* v_row = V_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
        acc += p * g4_to_float(v_row[d]);
    }
    out[static_cast<size_t>(q_h) * head_dim + d] = g4_from_float<T>(acc);
}

template <typename T>
__global__ void g4_kv_append_decode_kernel(
    int num_kv_heads,
    int head_dim,
    int pos,
    int max_T,
    const T* __restrict__ k_in,   // [num_kv_heads, head_dim]
    const T* __restrict__ v_in,
    T* __restrict__ k_cache,      // [num_kv_heads, max_T, head_dim]
    T* __restrict__ v_cache
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(num_kv_heads) * head_dim;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % head_dim);
    const int h = static_cast<int>(idx / head_dim);
    const size_t dst_off = (static_cast<size_t>(h) * max_T + pos) * head_dim + d;
    k_cache[dst_off] = k_in[idx];
    v_cache[dst_off] = v_in[idx];
}
