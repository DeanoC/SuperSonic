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

// =============================================================================
// Prefill / batched variants — Step 13.
//
// These process `seq_len` tokens in parallel within a single kernel launch.
// Layout assumption: "seq" is the outermost dimension so a slice for token
// `s` starts at pointer base + s * stride. Same weights apply to every token.
//
// Each batched kernel below is a minimal generalization of its single-token
// counterpart above — correctness first, fusion later. Used only by the new
// prefill path in `gemma4_e2e_validate.rs`; the decode path keeps the
// single-token primitives intact.
// =============================================================================

// Multi-row RMSNorm. grid=(n_rows,1,1), block=(256,1,1). Each block normalizes
// one row of `n_cols` scalars. The same `weight[n_cols]` vector (or nullptr
// for with_scale=False) is broadcast across every row.
//
// Subsumes both the single-row primitive and the per-head norm applied to
// [num_heads, head_dim] tensors, so we can use one kernel for every RMSNorm
// call site in the prefill path.
template <typename T>
__global__ void g4_rms_norm_rows_kernel(
    int n_rows,
    int n_cols,
    float eps,
    const T* __restrict__ xs,      // [n_rows, n_cols]
    const T* __restrict__ weight,  // [n_cols] or nullptr
    T* __restrict__ out            // [n_rows, n_cols]
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const T* xr = xs + static_cast<size_t>(row) * n_cols;
    T* outr = out + static_cast<size_t>(row) * n_cols;

    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x) {
        const float x = g4_to_float(xr[col]);
        partial += x * x;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
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
        const float x = g4_to_float(xr[col]);
        const float normed = x * shared_inv_rms;
        const float w = has_weight ? g4_to_float(weight[col]) : 1.0f;
        outr[col] = g4_from_float<T>(normed * w);
    }
}

// Batched work-stealing matvec. Computes `out[s, r] = dot(W[r, :], in[s, :])`
// for every (s, r) pair, s ∈ [0, seq_len), r ∈ [0, out_dim). Each block
// atomically claims the next (s, r) work item from `counter`; one block per
// work item runs the same dot product the single-token kernel already does.
template <typename T>
__global__ void g4_matvec_batched_kernel(
    int seq_len,
    int in_dim,
    int out_dim,
    const T* __restrict__ x,        // [seq_len, in_dim]
    const T* __restrict__ W,        // [out_dim, in_dim] row-major
    T* __restrict__ out,            // [seq_len, out_dim]
    unsigned int* __restrict__ counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const size_t total = static_cast<size_t>(seq_len) * out_dim;

    while (true) {
        int idx;
        if (tid == 0) {
            idx = static_cast<int>(atomicAdd(counter, 1u));
        }
        __shared__ int shared_idx;
        if (tid == 0) shared_idx = idx;
        __syncthreads();
        idx = shared_idx;
        if (static_cast<size_t>(idx) >= total) return;

        const int s = idx / out_dim;
        const int r = idx % out_dim;

        float partial = 0.0f;
        const T* w_row = W + static_cast<size_t>(r) * in_dim;
        const T* x_row = x + static_cast<size_t>(s) * in_dim;
        for (int i = tid; i < in_dim; i += block_size) {
            partial += g4_to_float(w_row[i]) * g4_to_float(x_row[i]);
        }

        __shared__ float shared_sum[256];
        shared_sum[tid] = partial;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            out[static_cast<size_t>(s) * out_dim + r] = g4_from_float<T>(shared_sum[0]);
        }
        __syncthreads();
    }
}

// Prefill split-half RoPE. Rotates a `[seq_len, num_heads, head_dim]` tensor
// where token `s` uses position `pos_base + s`. One thread per (s, h, i) with
// i ∈ [0, rotary_dim/2); like the single-token kernel, nope lanes (>=
// rotary_dim) are pass-through by construction (we only write indices in
// [0, rotary_dim)).
template <typename T>
__global__ void g4_rope_prefill_kernel(
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int pos_base,
    const T* __restrict__ cos_table,  // [max_pos, rotary_dim]
    const T* __restrict__ sin_table,
    T* __restrict__ x                 // [seq_len, num_heads, head_dim]
) {
    const int half = rotary_dim / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t per_seq = static_cast<size_t>(num_heads) * half;
    const size_t total = static_cast<size_t>(seq_len) * per_seq;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / per_seq);
    const size_t rem = idx - static_cast<size_t>(s) * per_seq;
    const int h = static_cast<int>(rem / half);
    const int i = static_cast<int>(rem % half);

    const int position = pos_base + s;
    const size_t cos_off = static_cast<size_t>(position) * rotary_dim + i;
    const float c = g4_to_float(cos_table[cos_off]);
    const float si = g4_to_float(sin_table[cos_off]);

    T* row = x + (static_cast<size_t>(s) * num_heads + h) * head_dim;
    const float x0 = g4_to_float(row[i]);
    const float x1 = g4_to_float(row[i + half]);
    row[i] = g4_from_float<T>(x0 * c - x1 * si);
    row[i + half] = g4_from_float<T>(x1 * c + x0 * si);
}

// Prefill KV append. Writes a `[seq_len, num_kv_heads, head_dim]` K/V tensor
// into cache slots `[pos_base, pos_base + seq_len)` of a
// `[num_kv_heads, max_T, head_dim]` cache. One thread per output element.
template <typename T>
__global__ void g4_kv_append_prefill_kernel(
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int pos_base,
    int max_T,
    const T* __restrict__ k_in,   // [seq_len, num_kv_heads, head_dim]
    const T* __restrict__ v_in,
    T* __restrict__ k_cache,      // [num_kv_heads, max_T, head_dim]
    T* __restrict__ v_cache
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t per_seq = static_cast<size_t>(num_kv_heads) * head_dim;
    const size_t total = static_cast<size_t>(seq_len) * per_seq;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / per_seq);
    const size_t rem = idx - static_cast<size_t>(s) * per_seq;
    const int h = static_cast<int>(rem / head_dim);
    const int d = static_cast<int>(rem % head_dim);

    const size_t src_off = (static_cast<size_t>(s) * num_kv_heads + h) * head_dim + d;
    const size_t dst_off = (static_cast<size_t>(h) * max_T + pos_base + s) * head_dim + d;
    k_cache[dst_off] = k_in[src_off];
    v_cache[dst_off] = v_in[src_off];
}

// -----------------------------------------------------------------------------
// Prefill attention (3 phases), generalizing the single-token SWA kernel.
// Cache already contains `kv_total = pos_base + seq_len` valid slots (filled
// by `g4_kv_append_prefill_kernel`). For each token `s` the valid attend
// range is `[min_t(s), pos_base + s]` with:
//   min_t(s) = max(0, pos_base + s - sliding_window + 1)  if sliding_window > 0
//            = 0                                           otherwise
// Scores layout: [seq_len, num_q_heads, max_T]. Entries outside the causal /
// SWA range are set to -inf so softmax ignores them.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_attn_prefill_scores_kernel(
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int pos_base,
    int max_T,
    int sliding_window,
    float scale,
    const T* __restrict__ Q,         // [seq_len, num_q_heads, head_dim]
    const T* __restrict__ K_cache,   // [num_kv_heads, max_T, head_dim]
    float* __restrict__ scores       // [seq_len, num_q_heads, max_T]
) {
    const int q_h = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    const int s = blockIdx.z;
    if (q_h >= num_q_heads || s >= seq_len) return;
    const int kv_total = pos_base + seq_len;
    if (t >= kv_total) return;

    const int last = pos_base + s;
    const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
    const size_t out_off =
        (static_cast<size_t>(s) * num_q_heads + q_h) * max_T + t;

    if (t < min_t || t > last) {
        scores[out_off] = -INFINITY;
        return;
    }

    const int num_q_per_kv = num_q_heads / num_kv_heads;
    const int kv_h = q_h / num_q_per_kv;

    const T* q_row = Q + (static_cast<size_t>(s) * num_q_heads + q_h) * head_dim;
    const T* k_row = K_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;

    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        acc += g4_to_float(q_row[d]) * g4_to_float(k_row[d]);
    }
    scores[out_off] = acc * scale;
}

// Softmax per (s, q_h). grid=(num_q_heads, seq_len, 1), block=(256,1,1).
__global__ void g4_attn_prefill_softmax_kernel(
    int seq_len,
    int num_q_heads,
    int kv_total,
    int max_T,
    float* __restrict__ scores       // [seq_len, num_q_heads, max_T]
) {
    const int q_h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    if (q_h >= num_q_heads || s >= seq_len) return;

    float* row =
        scores + (static_cast<size_t>(s) * num_q_heads + q_h) * max_T;

    // Pass 1: block-wide max.
    __shared__ float s_red[256];
    float partial_max = -INFINITY;
    for (int t = tid; t < kv_total; t += blockDim.x) {
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

    // Pass 2: exp + sum.
    float partial_sum = 0.0f;
    for (int t = tid; t < kv_total; t += blockDim.x) {
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
    for (int t = tid; t < kv_total; t += blockDim.x) {
        row[t] *= inv_sum;
    }
}

template <typename T>
__global__ void g4_attn_prefill_value_kernel(
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int pos_base,
    int max_T,
    const float* __restrict__ scores,  // [seq_len, num_q_heads, max_T]
    const T* __restrict__ V_cache,     // [num_kv_heads, max_T, head_dim]
    T* __restrict__ out                // [seq_len, num_q_heads, head_dim]
) {
    const int q_h = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int s = blockIdx.z;
    if (q_h >= num_q_heads || s >= seq_len || d >= head_dim) return;

    const int kv_total = pos_base + seq_len;
    const int num_q_per_kv = num_q_heads / num_kv_heads;
    const int kv_h = q_h / num_q_per_kv;

    float acc = 0.0f;
    const float* p_row =
        scores + (static_cast<size_t>(s) * num_q_heads + q_h) * max_T;
    for (int t = 0; t < kv_total; ++t) {
        const float p = p_row[t];
        const T* v_row = V_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
        acc += p * g4_to_float(v_row[d]);
    }
    out[(static_cast<size_t>(s) * num_q_heads + q_h) * head_dim + d] =
        g4_from_float<T>(acc);
}

// Elementwise residual add: out[i] = a[i] + b[i].
template <typename T>
__global__ void g4_add_residual_kernel(
    size_t n,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = g4_from_float<T>(g4_to_float(a[idx]) + g4_to_float(b[idx]));
}

// Elementwise scaled residual: out[i] = (a[i] + b[i]) * scalar.
// Used for the Gemma 4 PLE "(h_pre_ple + normed) * layer_scalar" step.
template <typename T>
__global__ void g4_add_scaled_residual_kernel(
    size_t n,
    float scalar,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = g4_from_float<T>((g4_to_float(a[idx]) + g4_to_float(b[idx])) * scalar);
}

// In-place scalar multiply: x[i] *= scalar. Implements the "BF16-rounded
// Python-float times tensor" pattern used by the PLE scaling steps in
// `compute_per_layer_inputs`. Caller converts the scale through BF16 rounding
// on the host (e.g. `bf16::from_f32(x).to_f32()`) before passing it.
template <typename T>
__global__ void g4_scalar_mul_inplace_kernel(
    size_t n,
    float scalar,
    T* __restrict__ x
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] = g4_from_float<T>(g4_to_float(x[idx]) * scalar);
}

// =============================================================================
// Fused attention-block kernel (Step 14) — runs one Gemma 4 decoder layer's
// attention half (input_norm → QKV → qk/v norm → RoPE → kv_append → SWA/full
// attention → o_proj → post_attn_norm → residual) in a single kernel launch.
//
// Multi-block cooperative design: uses `grid_barrier` (atomic counter/flag) to
// synchronize the phases across all blocks, and an atomic `matvec_counter` for
// work-stealing on the QKV and o_proj matmuls. Each intermediate lives in a
// caller-allocated F32 workspace so sub-ops consume/produce in a known layout
// without per-phase kernel launches.
//
// MLP + PLE remain as separate primitive launches for now; extending this
// kernel to cover them follows the same pattern and is the natural next step.
// =============================================================================

__device__ inline void g4_grid_barrier(
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

// Block-level Gemma RMS norm (plain weight * normed — no (w+1) offset).
// Operates over F32 data in/out; `weight` is BF16 (or null for with_scale=False).
// `lds_scratch` must be at least `blockDim.x` F32 slots; all threads participate.
template <typename T>
__device__ inline void g4_block_rms_norm_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const T* __restrict__ weight,
    int dim,
    float eps,
    float* __restrict__ lds_scratch
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    float sq = 0.0f;
    for (int c = tid; c < dim; c += bs) {
        const float v = src[c];
        sq += v * v;
    }
    lds_scratch[tid] = sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) lds_scratch[tid] += lds_scratch[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(lds_scratch[0] / static_cast<float>(dim) + eps);
    const bool has_weight = (weight != nullptr);
    for (int c = tid; c < dim; c += bs) {
        const float w = has_weight ? g4_to_float(weight[c]) : 1.0f;
        dst[c] = src[c] * inv * w;
    }
    __syncthreads();
}

// Per-head block RMS norm over a contiguous `[num_heads * head_dim]` F32 buffer.
// Block 0 walks each head sequentially and normalizes it with the shared
// `weight[head_dim]` tensor (or null for no-weight v_norm). All threads of the
// block cooperate on one head at a time.
template <typename T>
__device__ inline void g4_block_rms_norm_per_head_f32(
    float* __restrict__ data,
    const T* __restrict__ weight,
    int num_heads,
    int head_dim,
    float eps,
    float* __restrict__ lds_scratch
) {
    for (int h = 0; h < num_heads; ++h) {
        float* row = data + static_cast<size_t>(h) * head_dim;
        g4_block_rms_norm_f32<T>(row, row, weight, head_dim, eps, lds_scratch);
    }
}

// RoPE (split-half Gemma style) on a single-token tensor in F32 workspace,
// applied by one block. `data` layout: [num_heads, head_dim]. `rotary_dim`
// may be < head_dim (nope lanes pass through).
template <typename T>
__device__ inline void g4_block_rope_f32(
    float* __restrict__ data,
    const T* __restrict__ cos_table,   // [max_pos, rotary_dim]
    const T* __restrict__ sin_table,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int position
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int half = rotary_dim / 2;
    const int total = num_heads * half;
    for (int idx = tid; idx < total; idx += bs) {
        const int h = idx / half;
        const int i = idx - h * half;
        const size_t ct_off = static_cast<size_t>(position) * rotary_dim + i;
        const float c = g4_to_float(cos_table[ct_off]);
        const float s = g4_to_float(sin_table[ct_off]);
        float* row = data + static_cast<size_t>(h) * head_dim;
        const float x0 = row[i];
        const float x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x1 * c + x0 * s;
    }
    __syncthreads();
}

// Store one decode token's K/V into the cache at position `pos`. Reads from
// F32 staging buffers, writes BF16 to device cache. One block runs this in
// ~num_kv_heads*head_dim elements of parallel work.
template <typename T>
__device__ inline void g4_block_kv_append_f32_to_bf16(
    const float* __restrict__ k_f32,
    const float* __restrict__ v_f32,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache,
    int num_kv_heads,
    int head_dim,
    int pos,
    int max_T
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int total = num_kv_heads * head_dim;
    for (int idx = tid; idx < total; idx += bs) {
        const int d = idx % head_dim;
        const int h = idx / head_dim;
        const size_t dst_off =
            (static_cast<size_t>(h) * max_T + pos) * head_dim + d;
        k_cache[dst_off] = g4_from_float<T>(k_f32[idx]);
        v_cache[dst_off] = g4_from_float<T>(v_f32[idx]);
    }
    __syncthreads();
}

// Fused attention-block persistent kernel. Single launch per Gemma 4
// decoder-layer attention half. Orchestration:
//   0. All blocks: hidden BF16 → hidden_f32 (residual copy).
//   1. Block 0: input_norm(hidden_f32 → normed_f32).
//   2. All blocks: work-stealing QKV matmul over rows of (q_dim+2*kv_dim)
//      into proj_f32.
//   3. Block 0: q_norm, k_norm, v_norm (per-head RMSNorm in F32).
//   4. Block 0: RoPE on Q and K (using cos/sin tables for this layer type).
//   5. Block 0: store K, V into cache at slot `pos`.
//   6. All blocks: attention scores with SWA mask → scores_f32.
//   7. Per-q-head (block_idx==q_h): softmax over scores.
//   8. All blocks: value aggregation → attn_out_f32 [num_q_heads, head_dim].
//   9. All blocks: work-stealing o_proj over rows of hidden_size into
//      oproj_f32.
//  10. Block 0: post_attn_norm(oproj_f32 → normed_oproj_f32).
//  11. All blocks: residual: h_out_bf16[c] = bf16(hidden_f32[c] + normed_oproj_f32[c]).
//
// Workspace layout (F32, provided by caller via single flat buffer):
//   [0, hidden)                                : hidden_f32 (residual)
//   [hidden, 2*hidden)                         : normed_f32 (after input_norm)
//   [2h, 2h + q_dim + 2*kv_dim)                : proj_f32 (QKV output)
//   [2h + qkv, 2h + qkv + num_q_heads*max_T)   : scores_f32
//   [2h + qkv + nq*max_T, ...+ q_dim)          : attn_out_f32
//   [..., ... + hidden)                        : oproj_f32
//   [..., ... + hidden)                        : oproj_normed_f32
//
// The caller computes the total F32 count and allocates accordingly.
template <typename T>
__global__ void g4_fused_attn_block_kernel(
    int hidden_size, int num_q_heads, int num_kv_heads, int head_dim,
    int rotary_dim, int sliding_window, int position, int max_T,
    int shared_kv,              // 0 = compute+store K/V; non-zero = assume cache already populated
    float eps, float scale,
    const T* __restrict__ hidden_in,   // [hidden_size] BF16 (residual source)
    T* __restrict__ hidden_out,        // [hidden_size] BF16 (h_after_attn)
    const T* __restrict__ input_norm_w,
    const T* __restrict__ q_proj_w,
    const T* __restrict__ k_proj_w,
    const T* __restrict__ v_proj_w,
    const T* __restrict__ q_norm_w,
    const T* __restrict__ k_norm_w,
    const T* __restrict__ o_proj_w,
    const T* __restrict__ post_attn_norm_w,
    const T* __restrict__ cos_table,
    const T* __restrict__ sin_table,
    T* __restrict__ k_cache,           // [num_kv_heads, max_T, head_dim]
    T* __restrict__ v_cache,
    float* __restrict__ workspace,
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    // Workspace slots.
    float* hidden_f32     = workspace;
    float* normed_f32     = hidden_f32 + hidden_size;
    float* proj_f32       = normed_f32 + hidden_size;         // [q_dim + 2*kv_dim]
    float* scores_f32     = proj_f32 + (q_dim + 2 * kv_dim);  // [num_q_heads * max_T]
    float* attn_out_f32   = scores_f32 + num_q_heads * max_T; // [q_dim]
    float* oproj_f32      = attn_out_f32 + q_dim;             // [hidden_size]
    float* oproj_normed   = oproj_f32 + hidden_size;          // [hidden_size]

    extern __shared__ float lds[];

    // Phase 0: load hidden BF16 → F32 (residual), each block handles a stripe.
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        hidden_f32[c] = g4_to_float(hidden_in[c]);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 1: block 0 computes input_norm(hidden_f32) → normed_f32.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 2: work-stealing QKV matmul.
    // Reset counter (block 0).
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(total_qkv)) break;

        const T* w;
        int dst_off;
        if (row < static_cast<unsigned int>(q_dim)) {
            w = q_proj_w;
            dst_off = static_cast<int>(row);
        } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
            w = k_proj_w;
            dst_off = static_cast<int>(row);      // proj_f32 concatenates Q|K|V
        } else {
            w = v_proj_w;
            dst_off = static_cast<int>(row);
        }
        const int weight_row = (row < static_cast<unsigned int>(q_dim))
            ? static_cast<int>(row)
            : (row < static_cast<unsigned int>(q_dim + kv_dim)
                ? static_cast<int>(row - q_dim)
                : static_cast<int>(row - q_dim - kv_dim));
        const T* w_row = w + static_cast<size_t>(weight_row) * hidden_size;

        float p = 0.0f;
        for (int c = tid; c < hidden_size; c += bs) {
            p += g4_to_float(w_row[c]) * normed_f32[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) proj_f32[dst_off] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    float* q_f32 = proj_f32;
    float* k_f32 = proj_f32 + q_dim;
    float* v_f32 = proj_f32 + q_dim + kv_dim;

    // Phase 3: q/k/v norms + RoPE + KV append — block 0 serializes the per-head
    // normalizations and the RoPE apply, then stages the K/V into the cache.
    if (!shared_kv && blockIdx.x == 0) {
        g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
            num_q_heads, head_dim, eps, lds);
        g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
            num_kv_heads, head_dim, eps, lds);
        // v_norm has with_scale=False → pass null weight.
        g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
            num_kv_heads, head_dim, eps, lds);
        g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
            num_q_heads, head_dim, rotary_dim, position);
        g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
            num_kv_heads, head_dim, rotary_dim, position);
        g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache, v_cache,
            num_kv_heads, head_dim, position, max_T);
    } else if (shared_kv && blockIdx.x == 0) {
        // Only Q needs to be produced in this path.
        g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
            num_q_heads, head_dim, eps, lds);
        g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
            num_q_heads, head_dim, rotary_dim, position);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 4: attention scores (SWA or full) — parallelize over (q_head, t).
    const int kv_len = position + 1;
    const int last = kv_len - 1;
    const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
    const int num_q_per_kv = num_q_heads / num_kv_heads;
    {
        // Linearize (q_h, t) work items across all blocks; each block
        // processes a stripe of t values for all q heads.
        const int total_items = num_q_heads * kv_len;
        for (int item = blockIdx.x * bs + tid; item < total_items; item += nb * bs) {
            const int t = item % kv_len;
            const int q_h = item / kv_len;
            float val;
            if (t < min_t) {
                val = -INFINITY;
            } else {
                const int kv_h = q_h / num_q_per_kv;
                const float* qr = q_f32 + static_cast<size_t>(q_h) * head_dim;
                const T* kr = k_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                float acc = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    acc += qr[d] * g4_to_float(kr[d]);
                }
                val = acc * scale;
            }
            scores_f32[static_cast<size_t>(q_h) * max_T + t] = val;
        }
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 5: softmax — one block per q-head if we have enough blocks,
    // otherwise block 0 iterates over remaining heads sequentially.
    for (int q_h = blockIdx.x; q_h < num_q_heads; q_h += nb) {
        float* row = scores_f32 + static_cast<size_t>(q_h) * max_T;
        float pm = -INFINITY;
        for (int t = tid; t < kv_len; t += bs) pm = fmaxf(pm, row[t]);
        lds[tid] = pm;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
            __syncthreads();
        }
        const float mx = lds[0];
        __syncthreads();
        float ps = 0.0f;
        for (int t = tid; t < kv_len; t += bs) {
            const float e = expf(row[t] - mx);
            row[t] = e;
            ps += e;
        }
        lds[tid] = ps;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        const float inv_sum = 1.0f / lds[0];
        __syncthreads();
        for (int t = tid; t < kv_len; t += bs) row[t] *= inv_sum;
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 6: value aggregation — one thread per (q_head, d) element across blocks.
    {
        const int total_items = num_q_heads * head_dim;
        for (int item = blockIdx.x * bs + tid; item < total_items; item += nb * bs) {
            const int d = item % head_dim;
            const int q_h = item / head_dim;
            const int kv_h = q_h / num_q_per_kv;
            const float* pr = scores_f32 + static_cast<size_t>(q_h) * max_T;
            float acc = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                const T* vr = v_cache + (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                acc += pr[t] * g4_to_float(vr[d]);
            }
            attn_out_f32[static_cast<size_t>(q_h) * head_dim + d] = acc;
        }
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 7: work-stealing o_proj.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;
        const T* wr = o_proj_w + static_cast<size_t>(row) * q_dim;
        float p = 0.0f;
        for (int c = tid; c < q_dim; c += bs) {
            p += g4_to_float(wr[c]) * attn_out_f32[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) oproj_f32[row] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 8: post_attn_norm on block 0.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(oproj_normed, oproj_f32, post_attn_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 9: residual add + BF16 store.
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        hidden_out[c] = g4_from_float<T>(hidden_f32[c] + oproj_normed[c]);
    }
}

// =============================================================================
// Fused MLP + PLE kernel — the second half of a Gemma 4 decoder layer in one
// kernel launch. Pairs with `g4_fused_attn_block_kernel` so each layer can be
// executed as exactly two kernel launches (attention block + MLP/PLE block)
// regardless of its inner sub-op count.
//
// Phases (each separated by `g4_grid_barrier`):
//   0. Load h_mid BF16 → hidden_f32 (residual for the MLP + first residual
//      for the PLE branch).
//   1. Block 0: pre_ff_norm(hidden_f32) → normed_ff.
//   2. Work-stealing matvec over (gate_proj + up_proj) rows → gate_up_f32
//      [2*intermediate]. Gate rows come first, up rows follow.
//   3. Elementwise across blocks: gated_proj[i] = gelu_pytorch_tanh(gate[i])
//      * up[i] → gated_proj_f32 [intermediate].
//   4. Work-stealing matvec over down_proj rows → down_out_f32 [hidden].
//   5. Block 0: post_ff_norm(down_out_f32) → normed_down.
//   6. Elementwise residual: h_pre_ple_f32 = hidden_f32 + normed_down.
//   7. Work-stealing matvec over per_layer_input_gate_w rows → gated_ple_f32
//      [ple_hidden].
//   8. Elementwise gelu_pytorch_tanh(gated_ple) * per_layer_input →
//      gated_act_ple_f32 [ple_hidden].
//   9. Work-stealing matvec over per_layer_projection_w rows → projected_ple
//      [hidden].
//  10. Block 0: post_per_layer_input_norm(projected_ple) → normed_ple.
//  11. Elementwise: hidden_out[c] = bf16((h_pre_ple[c] + normed_ple[c]) *
//      layer_scalar).
//
// Workspace layout (F32, contiguous):
//   [hidden][hidden][2*intermediate][intermediate][hidden][hidden]
//   [hidden][ple_hidden][ple_hidden][hidden][hidden]
// = 7*hidden + 3*intermediate + 2*ple_hidden scalars.
// =============================================================================

// GELU-pytorch-tanh scalar helper (reused by multiple phases of the fused
// kernel below). Kept as a plain device inline to match Gemma 4's
// `hidden_activation = "gelu_pytorch_tanh"` exactly.
__device__ inline float g4_gelu_tanh_scalar(float g) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoef = 0.044715f;
    const float inner = kSqrt2OverPi * (g + kCoef * g * g * g);
    return 0.5f * g * (1.0f + tanhf(inner));
}

template <typename T>
__global__ void g4_fused_mlp_ple_kernel(
    int hidden_size, int intermediate_size, int ple_hidden,
    float eps, float layer_scalar,
    const T* __restrict__ hidden_in,    // [hidden_size] BF16 (h_mid)
    T* __restrict__ hidden_out,         // [hidden_size] BF16 (h_post_ple)
    const T* __restrict__ pre_ff_norm_w,
    const T* __restrict__ gate_proj_w,
    const T* __restrict__ up_proj_w,
    const T* __restrict__ down_proj_w,
    const T* __restrict__ post_ff_norm_w,
    const T* __restrict__ per_layer_input,           // [ple_hidden] BF16
    const T* __restrict__ per_layer_input_gate_w,
    const T* __restrict__ per_layer_projection_w,
    const T* __restrict__ post_per_layer_input_norm_w,
    float* __restrict__ workspace,
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    // Workspace slots.
    float* hidden_f32      = workspace;
    float* normed_ff       = hidden_f32 + hidden_size;
    float* gate_up_f32     = normed_ff + hidden_size;                   // [2*intermediate]
    float* gated_proj_f32  = gate_up_f32 + 2 * intermediate_size;       // [intermediate]
    float* down_out_f32    = gated_proj_f32 + intermediate_size;        // [hidden]
    float* normed_down_f32 = down_out_f32 + hidden_size;                // [hidden]
    float* h_pre_ple_f32   = normed_down_f32 + hidden_size;             // [hidden]
    float* gated_ple_f32   = h_pre_ple_f32 + hidden_size;               // [ple_hidden]
    float* gated_act_ple   = gated_ple_f32 + ple_hidden;                // [ple_hidden]
    float* projected_ple   = gated_act_ple + ple_hidden;                // [hidden]
    float* normed_ple_f32  = projected_ple + hidden_size;               // [hidden]

    extern __shared__ float lds[];

    // Phase 0: load h_mid BF16 → hidden_f32.
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        hidden_f32[c] = g4_to_float(hidden_in[c]);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 1: pre_ff_norm on block 0.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(normed_ff, hidden_f32, pre_ff_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 2: work-stealing gate_proj + up_proj matvec (concatenated).
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int total_mlp1 = 2 * intermediate_size;
    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(total_mlp1)) break;

        const T* w = (row < static_cast<unsigned int>(intermediate_size))
            ? gate_proj_w : up_proj_w;
        const int weight_row = (row < static_cast<unsigned int>(intermediate_size))
            ? static_cast<int>(row)
            : static_cast<int>(row - intermediate_size);
        const T* wr = w + static_cast<size_t>(weight_row) * hidden_size;

        float p = 0.0f;
        for (int c = tid; c < hidden_size; c += bs) {
            p += g4_to_float(wr[c]) * normed_ff[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) gate_up_f32[row] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 3: gated_proj[i] = gelu(gate[i]) * up[i].
    {
        const float* gate_slot = gate_up_f32;
        const float* up_slot   = gate_up_f32 + intermediate_size;
        for (int i = tid + blockIdx.x * bs; i < intermediate_size; i += bs * nb) {
            gated_proj_f32[i] = g4_gelu_tanh_scalar(gate_slot[i]) * up_slot[i];
        }
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 4: work-stealing down_proj matvec → down_out_f32.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;

        const T* wr = down_proj_w + static_cast<size_t>(row) * intermediate_size;
        float p = 0.0f;
        for (int c = tid; c < intermediate_size; c += bs) {
            p += g4_to_float(wr[c]) * gated_proj_f32[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) down_out_f32[row] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 5: post_ff_norm on block 0.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(normed_down_f32, down_out_f32, post_ff_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 6: h_pre_ple = hidden_f32 + normed_down.
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        h_pre_ple_f32[c] = hidden_f32[c] + normed_down_f32[c];
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 7: work-stealing per_layer_input_gate matvec → gated_ple [ple_hidden].
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(ple_hidden)) break;

        const T* wr = per_layer_input_gate_w + static_cast<size_t>(row) * hidden_size;
        float p = 0.0f;
        for (int c = tid; c < hidden_size; c += bs) {
            p += g4_to_float(wr[c]) * h_pre_ple_f32[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) gated_ple_f32[row] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 8: gated_act_ple[i] = gelu(gated_ple[i]) * per_layer_input[i].
    for (int i = tid + blockIdx.x * bs; i < ple_hidden; i += bs * nb) {
        const float g = g4_gelu_tanh_scalar(gated_ple_f32[i]);
        gated_act_ple[i] = g * g4_to_float(per_layer_input[i]);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 9: work-stealing per_layer_projection matvec → projected_ple [hidden].
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;

        const T* wr = per_layer_projection_w + static_cast<size_t>(row) * ple_hidden;
        float p = 0.0f;
        for (int c = tid; c < ple_hidden; c += bs) {
            p += g4_to_float(wr[c]) * gated_act_ple[c];
        }
        lds[tid] = p;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }
        if (tid == 0) projected_ple[row] = lds[0];
        __syncthreads();
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 10: post_per_layer_input_norm on block 0.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(normed_ple_f32, projected_ple,
                                  post_per_layer_input_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 11: hidden_out = (h_pre_ple + normed_ple) * layer_scalar, BF16 store.
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        const float v = (h_pre_ple_f32[c] + normed_ple_f32[c]) * layer_scalar;
        hidden_out[c] = g4_from_float<T>(v);
    }
}

// Gather one layer's per-layer-input slice from a batched PLI table. Input is
// `[seq_len, num_layers, ple_hidden]` layout (tokens outermost); output is
// `[seq_len, ple_hidden]` contiguous for the requested `layer_idx`. Used in
// the prefill PLE branch so the existing elementwise gelu_tanh_gate_mul can
// consume the correct PLI slice per layer without per-token strides.
template <typename T>
__global__ void g4_gather_layer_slice_kernel(
    int seq_len,
    int num_layers,
    int ple_hidden,
    int layer_idx,
    const T* __restrict__ src,    // [seq_len, num_layers, ple_hidden]
    T* __restrict__ out           // [seq_len, ple_hidden]
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (s >= seq_len || col >= ple_hidden) return;
    const size_t src_off =
        (static_cast<size_t>(s) * num_layers + layer_idx) * ple_hidden + col;
    out[static_cast<size_t>(s) * ple_hidden + col] = src[src_off];
}

// Embedding gather: out[s, d] = table[token_ids[s], d] * scale (scale is a
// Python-float * BF16-tensor rounding, so it's applied in FP32 after BF16
// load and rounded back on store). `token_ids` is a host-side array of u32
// token IDs, uploaded as a `[seq_len]` u32 device buffer.
template <typename T>
__global__ void g4_embed_gather_scaled_kernel(
    int seq_len,
    int hidden_size,
    int vocab_size,
    float scale,
    const unsigned int* __restrict__ token_ids,  // [seq_len]
    const T* __restrict__ table,                 // [vocab, hidden]
    T* __restrict__ out                          // [seq_len, hidden]
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (s >= seq_len || d >= hidden_size) return;

    const unsigned int tok = token_ids[s];
    // Out-of-range tokens produce zero (caller validates in practice).
    if (tok >= static_cast<unsigned int>(vocab_size)) {
        out[static_cast<size_t>(s) * hidden_size + d] = g4_from_float<T>(0.0f);
        return;
    }
    const size_t src_off = static_cast<size_t>(tok) * hidden_size + d;
    const float v = g4_to_float(table[src_off]) * scale;
    out[static_cast<size_t>(s) * hidden_size + d] = g4_from_float<T>(v);
}
