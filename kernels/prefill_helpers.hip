// Prefill-specific helper kernels.
// Kept in a separate compilation unit to avoid touching the decode megakernel
// files (hipcc codegen sensitivity on gfx1150).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <math.h>

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

__device__ inline float pfx_exp_fast(float x) {
    return __expf(x);
}

__device__ inline float pfx_sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = pfx_exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = pfx_exp_fast(x);
    return e / (1.0f + e);
}

__device__ inline float pfx_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
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

__global__ void pfx_argmax_bf16_rows_kernel(
    size_t rows,
    size_t cols,
    const hip_bfloat16* __restrict__ logits,
    uint32_t* __restrict__ out_index
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const hip_bfloat16* row_logits = logits + row * cols;

    float best_val = -1.0e30f;
    uint32_t best_idx = 0;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        const float val = static_cast<float>(row_logits[col]);
        const uint32_t idx = static_cast<uint32_t>(col);
        if (val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float rhs_val = shared_vals[tid + offset];
            const uint32_t rhs_idx = shared_idx[tid + offset];
            if (rhs_val > shared_vals[tid] ||
                (rhs_val == shared_vals[tid] && rhs_idx < shared_idx[tid])) {
                shared_vals[tid] = rhs_val;
                shared_idx[tid] = rhs_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_index[row] = shared_idx[0];
    }
}

// Greedy argmax on F32 logits, comparing the same BF16-rounded values
// argmax_bf16_rows would see after an F32→BF16 store. Used so HIP greedy
// decode can skip the 248k-logit cast.
__global__ void pfx_argmax_f32_as_bf16_rows_kernel(
    size_t rows,
    size_t cols,
    const float* __restrict__ logits,
    uint32_t* __restrict__ out_index
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const float* row_logits = logits + row * cols;

    float best_val = -1.0e30f;
    uint32_t best_idx = 0;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        const float val = static_cast<float>(hip_bfloat16(row_logits[col]));
        const uint32_t idx = static_cast<uint32_t>(col);
        if (val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float rhs_val = shared_vals[tid + offset];
            const uint32_t rhs_idx = shared_idx[tid + offset];
            if (rhs_val > shared_vals[tid] ||
                (rhs_val == shared_vals[tid] && rhs_idx < shared_idx[tid])) {
                shared_vals[tid] = rhs_val;
                shared_idx[tid] = rhs_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_index[row] = shared_idx[0];
    }
}

// ---- Kernel 2: RoPE for prefill ----
// Applies RoPE in-place on tensor with layout [seq_len, num_heads, head_dim].
// Only the first rotary_dim dimensions of each head are rotated.
// cos_table/sin_table: [max_pos, half_rot] where half_rot = rotary_dim / 2.
// Qwen3.5 uses rotate_half over the rotary slice:
// new[i]            = old[i] * cos[i] - old[i + half_rot] * sin[i]
// new[i + half_rot] = old[i] * sin[i] + old[i + half_rot] * cos[i]

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
    const float x0 = pfx_to_float(data[base + i]);
    const float x1 = pfx_to_float(data[base + half_rot + i]);

    data[base + i]            = pfx_from_float<T>(x0 * c - x1 * s);
    data[base + half_rot + i] = pfx_from_float<T>(x0 * s + x1 * c);
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

template <typename T>
__global__ void pfx_transpose_shd_hsd_pair_kernel(
    int S, int H, int D,
    const T* src_a,
    const T* src_b,
    T* dst_a,
    T* dst_b
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * H * D;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % D);
    const int h = static_cast<int>((idx / D) % H);
    const int s = static_cast<int>(idx / (static_cast<size_t>(D) * H));

    const size_t dst_off = static_cast<size_t>(h) * S * D + static_cast<size_t>(s) * D + d;
    dst_a[dst_off] = src_a[idx];
    dst_b[dst_off] = src_b[idx];
}

__global__ void pfx_transpose_shd_to_cache_bf16_kernel(
    int S, int H, int D,
    int cache_len,
    int dst_pos,
    const hip_bfloat16* __restrict__ src,
    hip_bfloat16* __restrict__ cache
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * H * D;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % D);
    const int h = static_cast<int>((idx / D) % H);
    const int s = static_cast<int>(idx / (static_cast<size_t>(D) * H));

    const size_t dst_off = (static_cast<size_t>(h) * cache_len + dst_pos + s) * D + d;
    cache[dst_off] = src[idx];
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

// ---- Kernel 5b: Prepare conv input and next conv tail in one pass ----
// src: [S, C] row-major current QKV projection
// old_tail: [C, pad] row-major previous conv tail
// conv_input: [C, pad + S] row-major with old_tail prepended then src transposed
// new_tail: [C, pad] row-major, extracted from the final pad rows of src

template <typename T>
__global__ void pfx_prepare_conv_input_tail_kernel(
    int S, int C, int pad,
    const T* src,
    const T* old_tail,
    T* conv_input,
    T* new_tail
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_len = static_cast<size_t>(pad + S);
    const size_t conv_total = static_cast<size_t>(C) * total_len;
    const size_t tail_total = static_cast<size_t>(C) * pad;
    const size_t total = conv_total > tail_total ? conv_total : tail_total;
    if (idx >= total) return;

    if (idx < conv_total) {
        const int c = static_cast<int>(idx / total_len);
        const int t = static_cast<int>(idx - static_cast<size_t>(c) * total_len);
        if (t < pad) {
            conv_input[idx] = old_tail[static_cast<size_t>(c) * pad + t];
        } else {
            conv_input[idx] = src[static_cast<size_t>(t - pad) * C + c];
        }
    }

    if (idx < tail_total) {
        const int c = static_cast<int>(idx / pad);
        const int t = static_cast<int>(idx - static_cast<size_t>(c) * pad);
        const int src_row = S - pad + t;
        new_tail[idx] = src[static_cast<size_t>(src_row) * C + c];
    }
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

// Fused full-attention epilogue staging:
//   1. read F32 attention output in [H, S, D],
//   2. round to BF16 to match the old cast kernel,
//   3. transpose to [S, H, D],
//   4. apply the BF16 attention gate in S-first layout.
__global__ void pfx_cast_transpose_gate_hsd_to_shd_bf16_kernel(
    int S,
    int H,
    int D,
    const float* __restrict__ attn_hsd,
    const hip_bfloat16* __restrict__ gate_shd,
    hip_bfloat16* __restrict__ out_shd
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * H * D;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % D);
    const int h = static_cast<int>((idx / D) % H);
    const int s = static_cast<int>(idx / (static_cast<size_t>(D) * H));
    const size_t src_off = (static_cast<size_t>(h) * S + s) * D + d;

    const hip_bfloat16 rounded = hip_bfloat16(attn_hsd[src_off]);
    const float x = static_cast<float>(rounded);
    const float g = static_cast<float>(gate_shd[idx]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    out_shd[idx] = hip_bfloat16(x * sigmoid_g);
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

__global__ void pfx_compute_beta_g_ba_bf16_kernel(
    int seq_len,
    int nv,
    const hip_bfloat16* BA,        // [seq_len, 2*nv], first B then A
    const hip_bfloat16* dt_bias,   // [nv]
    const hip_bfloat16* a_log_exp, // [nv]
    float* beta,                   // [nv, seq_len]
    float* g                       // [nv, seq_len]
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(seq_len) * nv;
    if (idx >= total) return;

    const int t = static_cast<int>(idx / nv);
    const int h = static_cast<int>(idx % nv);
    const size_t row = static_cast<size_t>(t) * static_cast<size_t>(2 * nv);

    const float b_val = pfx_to_float(BA[row + h]);
    beta[static_cast<size_t>(h) * seq_len + t] = 1.0f / (1.0f + expf(-b_val));

    const float a_val = pfx_to_float(BA[row + static_cast<size_t>(nv) + h]);
    const float dt = pfx_to_float(dt_bias[h]);
    const float ale = pfx_to_float(a_log_exp[h]);
    const float sp = logf(1.0f + expf(a_val + dt));
    g[static_cast<size_t>(h) * seq_len + t] = -sp * ale;
}

#define PFX_BA_TILE_M 16
#define PFX_BA_TILE_N 16
#define PFX_BA_TILE_K 32

__global__ void pfx_project_ba_compute_beta_g_bf16_kernel(
    int seq_len,
    int hidden_dim,
    int nv,
    const hip_bfloat16* __restrict__ hidden,   // [seq_len, hidden_dim]
    const hip_bfloat16* __restrict__ ba_weight, // [2*nv, hidden_dim]
    const hip_bfloat16* __restrict__ dt_bias,   // [nv]
    const hip_bfloat16* __restrict__ a_log_exp, // [nv]
    float* __restrict__ beta,                   // [nv, seq_len]
    float* __restrict__ g                       // [nv, seq_len]
) {
    const int tx = threadIdx.x % PFX_BA_TILE_N;
    const int ty = threadIdx.x / PFX_BA_TILE_N;

    const int tile_row = blockIdx.y * PFX_BA_TILE_M;
    const int tile_col = blockIdx.x * PFX_BA_TILE_N;
    const int row = tile_row + ty;
    const int col = tile_col + tx;
    const int out_cols = 2 * nv;

    __shared__ float s_hidden[PFX_BA_TILE_M][PFX_BA_TILE_K];
    __shared__ float s_weight[PFX_BA_TILE_N][PFX_BA_TILE_K];

    float acc = 0.0f;
    for (int kk_base = 0; kk_base < hidden_dim; kk_base += PFX_BA_TILE_K) {
        for (int i = threadIdx.x; i < PFX_BA_TILE_M * PFX_BA_TILE_K; i += blockDim.x) {
            const int lr = i / PFX_BA_TILE_K;
            const int lc = i % PFX_BA_TILE_K;
            const int gr = tile_row + lr;
            const int gc = kk_base + lc;
            if (gr < seq_len && gc < hidden_dim) {
                s_hidden[lr][lc] =
                    pfx_to_float(hidden[static_cast<size_t>(gr) * hidden_dim + gc]);
            } else {
                s_hidden[lr][lc] = 0.0f;
            }
        }

        for (int i = threadIdx.x; i < PFX_BA_TILE_N * PFX_BA_TILE_K; i += blockDim.x) {
            const int wr = i / PFX_BA_TILE_K;
            const int wc = i % PFX_BA_TILE_K;
            const int gn = tile_col + wr;
            const int gk = kk_base + wc;
            if (gn < out_cols && gk < hidden_dim) {
                s_weight[wr][wc] =
                    pfx_to_float(ba_weight[static_cast<size_t>(gn) * hidden_dim + gk]);
            } else {
                s_weight[wr][wc] = 0.0f;
            }
        }

        __syncthreads();

        if (row < seq_len && col < out_cols) {
            for (int kk = 0; kk < PFX_BA_TILE_K; ++kk) {
                acc += s_hidden[ty][kk] * s_weight[tx][kk];
            }
        }

        __syncthreads();
    }

    if (row >= seq_len || col >= out_cols) return;

    const float val = pfx_to_float(pfx_from_float<hip_bfloat16>(acc));
    if (col < nv) {
        beta[static_cast<size_t>(col) * seq_len + row] = 1.0f / (1.0f + expf(-val));
    } else {
        const int h = col - nv;
        const float dt = pfx_to_float(dt_bias[h]);
        const float ale = pfx_to_float(a_log_exp[h]);
        const float sp = logf(1.0f + expf(val + dt));
        g[static_cast<size_t>(h) * seq_len + row] = -sp * ale;
    }
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

__global__ void pfx_split_qgate_norm_bf16_kernel(
    int S,
    int num_heads,
    int head_dim,
    float eps,
    const hip_bfloat16* __restrict__ src,
    const hip_bfloat16* __restrict__ norm_w,
    hip_bfloat16* __restrict__ query_out,
    hip_bfloat16* __restrict__ gate_out
) {
    const int row = static_cast<int>(blockIdx.x);
    const int rows = S * num_heads;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int s = row / num_heads;
    const int h = row - s * num_heads;
    const size_t src_base = static_cast<size_t>(s) * num_heads * head_dim * 2
                          + static_cast<size_t>(h) * head_dim * 2;
    const size_t out_base = static_cast<size_t>(row) * head_dim;

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float x = static_cast<float>(src[src_base + d]);
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

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float x = static_cast<float>(src[src_base + d]);
        const float w = static_cast<float>(norm_w[d]);
        query_out[out_base + d] = hip_bfloat16(x * inv_rms * w);
        gate_out[out_base + d] = src[src_base + head_dim + d];
    }
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

__global__ void pfx_split_kv_bf16_kernel(
    int S,
    int kv_dim,
    const hip_bfloat16* src,
    hip_bfloat16* K_out,
    hip_bfloat16* V_out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(S) * kv_dim;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / kv_dim);
    const int c = static_cast<int>(idx % kv_dim);
    const size_t src_base = static_cast<size_t>(s) * static_cast<size_t>(2 * kv_dim);
    K_out[idx] = src[src_base + c];
    V_out[idx] = src[src_base + static_cast<size_t>(kv_dim) + c];
}

__global__ void pfx_split_qkv_bf16_to_f32_kernel(
    int S,
    int key_dim,
    int val_dim,
    const hip_bfloat16* src,
    float* Q_out,
    float* K_out,
    float* V_out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int qkv_dim = key_dim * 2 + val_dim;
    const size_t total = static_cast<size_t>(S) * qkv_dim;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / qkv_dim);
    const int c = static_cast<int>(idx % qkv_dim);
    const float val = static_cast<float>(src[idx]);

    if (c < key_dim) {
        Q_out[static_cast<size_t>(s) * key_dim + c] = val;
    } else if (c < key_dim * 2) {
        K_out[static_cast<size_t>(s) * key_dim + (c - key_dim)] = val;
    } else {
        V_out[static_cast<size_t>(s) * val_dim + (c - key_dim * 2)] = val;
    }
}

__global__ void pfx_split_norm_transpose_qkv_bf16_kernel(
    int S,
    int nk,
    int nv,
    int khd,
    int vhd,
    float q_scale,
    float eps,
    const hip_bfloat16* src,
    float* Q_out,
    float* K_out,
    float* V_out
) {
    const int key_dim = nk * khd;
    const int val_dim = nv * vhd;
    const int qkv_dim = 2 * key_dim + val_dim;
    const int q_rows = S * nk;
    const int kv_rows = S * nv;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row < q_rows) {
        const int s = row / nk;
        const int h = row - s * nk;
        const size_t src_base = static_cast<size_t>(s) * qkv_dim +
                                static_cast<size_t>(h) * khd;
        float partial = 0.0f;
        for (int d = tid; d < khd; d += blockDim.x) {
            const float x = static_cast<float>(src[src_base + d]);
            partial += x * x;
        }

        __shared__ float shared_sum[256];
        shared_sum[tid] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
            __syncthreads();
        }

        __shared__ float shared_inv_norm;
        if (tid == 0) shared_inv_norm = rsqrtf(shared_sum[0] + eps);
        __syncthreads();

        for (int d = tid; d < khd; d += blockDim.x) {
            const float val =
                (static_cast<float>(src[src_base + d]) * shared_inv_norm) * q_scale;
            for (int oh = h; oh < nv; oh += nk) {
                Q_out[(static_cast<size_t>(oh) * S + s) * khd + d] = val;
            }
        }
        return;
    }

    const int k_row = row - q_rows;
    if (k_row < q_rows) {
        const int s = k_row / nk;
        const int h = k_row - s * nk;
        const size_t src_base = static_cast<size_t>(s) * qkv_dim +
                                static_cast<size_t>(key_dim) +
                                static_cast<size_t>(h) * khd;
        float partial = 0.0f;
        for (int d = tid; d < khd; d += blockDim.x) {
            const float x = static_cast<float>(src[src_base + d]);
            partial += x * x;
        }

        __shared__ float shared_sum[256];
        shared_sum[tid] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
            __syncthreads();
        }

        __shared__ float shared_inv_norm;
        if (tid == 0) shared_inv_norm = rsqrtf(shared_sum[0] + eps);
        __syncthreads();

        for (int d = tid; d < khd; d += blockDim.x) {
            const float val = static_cast<float>(src[src_base + d]) * shared_inv_norm;
            for (int oh = h; oh < nv; oh += nk) {
                K_out[(static_cast<size_t>(oh) * S + s) * khd + d] = val;
            }
        }
        return;
    }

    const int v_row = k_row - q_rows;
    if (v_row >= kv_rows) return;
    const int s = v_row / nv;
    const int h = v_row - s * nv;
    const size_t src_base = static_cast<size_t>(s) * qkv_dim +
                            static_cast<size_t>(2 * key_dim) +
                            static_cast<size_t>(h) * vhd;
    for (int d = tid; d < vhd; d += blockDim.x) {
        V_out[(static_cast<size_t>(h) * S + s) * vhd + d] =
            static_cast<float>(src[src_base + d]);
    }
}

__global__ void pfx_rms_norm_gated_sfirst_bf16_kernel(
    int S,
    int nv,
    int vhd,
    float eps,
    const hip_bfloat16* hidden_hsd,
    const hip_bfloat16* gate_sfirst,
    const hip_bfloat16* weight,
    hip_bfloat16* out_sfirst
) {
    const int row = blockIdx.x;
    const int total_rows = S * nv;
    if (row >= total_rows) return;

    const int tid = threadIdx.x;
    const int s = row / nv;
    const int h = row - s * nv;
    const hip_bfloat16* row_hidden =
        hidden_hsd + (static_cast<size_t>(h) * S + s) * vhd;
    const hip_bfloat16* row_gate =
        gate_sfirst + static_cast<size_t>(s) * nv * vhd + static_cast<size_t>(h) * vhd;
    hip_bfloat16* row_out =
        out_sfirst + static_cast<size_t>(s) * nv * vhd + static_cast<size_t>(h) * vhd;

    float partial = 0.0f;
    for (int col = tid; col < vhd; col += blockDim.x) {
        const float x = static_cast<float>(row_hidden[col]);
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
        const float mean = shared_sum[0] / static_cast<float>(vhd);
        shared_inv_rms = rsqrtf(mean + eps);
    }
    __syncthreads();

    for (int col = tid; col < vhd; col += blockDim.x) {
        const float x = static_cast<float>(row_hidden[col]);
        const float gate_x = static_cast<float>(row_gate[col]);
        const float gate_silu = gate_x * pfx_sigmoid_fast(gate_x);
        const float weight_val = static_cast<float>(weight[col]);
        row_out[col] = hip_bfloat16(x * shared_inv_rms * weight_val * gate_silu);
    }
}

__global__ void pfx_split_qkvz_bf16_kernel(
    int S,
    int qkv_dim,
    int z_dim,
    const hip_bfloat16* src,
    hip_bfloat16* QKV_out,
    hip_bfloat16* Z_out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total_dim = qkv_dim + z_dim;
    const size_t total = static_cast<size_t>(S) * total_dim;
    if (idx >= total) return;

    const int s = static_cast<int>(idx / total_dim);
    const int c = static_cast<int>(idx % total_dim);
    const hip_bfloat16 val = src[idx];

    if (c < qkv_dim) {
        QKV_out[static_cast<size_t>(s) * qkv_dim + c] = val;
    } else {
        Z_out[static_cast<size_t>(s) * z_dim + (c - qkv_dim)] = val;
    }
}

// ---- Kernel 10: Repeat interleave along head dimension ----
// src: [S, n_heads, head_dim] → dst: [S, n_heads * repeats, head_dim]
// Each head is repeated 'repeats' times consecutively.

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
    const int src_h = oh % n_heads;

    const size_t src_off = static_cast<size_t>(s) * n_heads * head_dim
                         + static_cast<size_t>(src_h) * head_dim + d;
    dst[idx] = src[src_off];
}

// Fused repeat-interleave and transpose:
//   src [S, n_heads, head_dim]
//   dst [n_heads * repeats, S, head_dim]
// This is equivalent to repeat_interleave_heads followed by transpose_shd_hsd.
template <typename T>
__global__ void pfx_repeat_interleave_transpose_hsd_kernel(
    int S, int n_heads, int head_dim, int repeats,
    const T* src,
    T* dst
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int out_heads = n_heads * repeats;
    const size_t total = static_cast<size_t>(out_heads) * S * head_dim;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % head_dim);
    const int s = static_cast<int>((idx / head_dim) % S);
    const int oh = static_cast<int>(idx / (static_cast<size_t>(head_dim) * S));
    const int src_h = oh % n_heads;

    const size_t src_off = static_cast<size_t>(s) * n_heads * head_dim
                         + static_cast<size_t>(src_h) * head_dim + d;
    dst[idx] = src[src_off];
}

// ---- Kernel 11: Decode-only full attention for q_len=1 ----
// query: [B, QH, D], key/value: [B, KVH, T, D], out: [B, QH, D].

template <typename T>
__global__ void pfx_full_attention_decode_flat_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    const T* query,
    const T* key,
    const T* value,
    T* out
) {
    const int lane = threadIdx.x;
    if (lane >= warpSize) return;

    const int total_rows = batch_size * q_heads;
    const int row = blockIdx.x;
    if (row >= total_rows) return;

    const int q_head = row % q_heads;
    const int batch = row / q_heads;
    const int kv_head = q_head / num_kv_groups;

    const T* q_row = query + (static_cast<size_t>(batch) * q_heads + q_head) * head_dim;
    const T* k_head = key + (static_cast<size_t>(batch) * kv_heads + kv_head) * kv_len * head_dim;
    const T* v_head = value + (static_cast<size_t>(batch) * kv_heads + kv_head) * kv_len * head_dim;
    T* out_row = out + (static_cast<size_t>(batch) * q_heads + q_head) * head_dim;

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

    if (lane == 0) shared_max = -INFINITY;
    __syncthreads();

    for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
        const T* k_row = k_head + static_cast<size_t>(k_pos) * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += warpSize) {
            partial += pfx_to_float(q_row[d]) * pfx_to_float(k_row[d]);
        }
        const float score = pfx_wave_sum(partial) * scale;
        if (lane == 0 && score > shared_max) shared_max = score;
        __syncthreads();
    }

    if (lane == 0) shared_denom = 0.0f;
    __syncthreads();

    for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
        const T* k_row = k_head + static_cast<size_t>(k_pos) * head_dim;
        const T* v_row = v_head + static_cast<size_t>(k_pos) * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += warpSize) {
            partial += pfx_to_float(q_row[d]) * pfx_to_float(k_row[d]);
        }
        const float score = pfx_wave_sum(partial) * scale;

        if (lane == 0) {
            shared_weight = expf(score - shared_max);
            shared_denom += shared_weight;
        }
        __syncthreads();

        for (int i = 0; i < local_count; ++i) {
            local_acc[i] += shared_weight * pfx_to_float(v_row[local_dims[i]]);
        }
        __syncthreads();
    }

    if (lane == 0) shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
    __syncthreads();

    for (int i = 0; i < local_count; ++i) {
        out_row[local_dims[i]] = pfx_from_float<T>(local_acc[i] * shared_inv_denom);
    }
}
