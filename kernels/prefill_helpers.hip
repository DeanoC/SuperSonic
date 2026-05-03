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

// ---- Kernel 2b: RoPE for prefill with per-token position IDs ----
// Same math as `pfx_apply_rope_prefill_kernel`, but the cos/sin lookup
// position is `pos_ids[slot]` instead of the slot index itself. Used by
// SpecPrefill (arXiv 2502.02789): the target's prefill runs on a
// compacted subset of prompt tokens, but each token must rotate by its
// ORIGINAL prompt position so the attention math sees the same RoPE
// frequencies as the full-prefill baseline.
//
// When `pos_ids[slot] == slot` for all slots, this kernel produces
// byte-identical output to `pfx_apply_rope_prefill_kernel` — the
// parity test in `crates/runner/tests/specprefill_rope_indirect_parity.rs`
// pins this invariant.
template <typename T>
__global__ void pfx_apply_rope_prefill_indirect_kernel(
    int seq_len,
    int num_heads,
    int head_dim,
    int half_rot,           // rotary_dim / 2
    const T* cos_table,     // [max_pos, half_rot]
    const T* sin_table,     // [max_pos, half_rot]
    const int* pos_ids,     // [seq_len] — original prompt position per slot
    T* data                 // [seq_len, num_heads, head_dim] — modified in-place
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    if (idx >= total) return;

    const int i    = static_cast<int>(idx % half_rot);
    const int h    = static_cast<int>((idx / half_rot) % num_heads);
    const int slot = static_cast<int>(idx / (static_cast<size_t>(half_rot) * num_heads));
    const int pos  = pos_ids[slot];   // ← only difference vs the dense kernel

    const size_t cos_off = static_cast<size_t>(pos) * half_rot + i;
    const float c = pfx_to_float(cos_table[cos_off]);
    const float s = pfx_to_float(sin_table[cos_off]);

    const size_t base = static_cast<size_t>(slot) * num_heads * head_dim
                      + static_cast<size_t>(h) * head_dim;
    const float x0 = pfx_to_float(data[base + i]);
    const float x1 = pfx_to_float(data[base + half_rot + i]);

    data[base + i]            = pfx_from_float<T>(x0 * c - x1 * s);
    data[base + half_rot + i] = pfx_from_float<T>(x0 * s + x1 * c);
}

// ---- Kernel 2c: per-row softmax(Q · Kᵀ) for SpecPrefill look-ahead ----
// Computes the post-softmax attention weights for the last `lookahead_count`
// query rows of a single full-attention layer against the prompt-only K
// cache. No V, no causality. Used by SpecPrefill (arXiv 2502.02789) to
// harvest the importance signal for token selection.
//
// Grid: (q_heads * lookahead_count) blocks, each one warp (warpSize threads).
// Shared memory: kv_len floats for the per-row exponentials. Caller
// must size dynamic shared mem at launch.
//
// q:      [lookahead_count, q_heads, head_dim]  BF16, post-RoPE
// k:      [kv_heads, kv_len, head_dim]          BF16
// scores: [q_heads, lookahead_count, kv_len]    F32
template <typename T>
__global__ void pfx_lookahead_attention_scores_kernel(
    int q_heads,
    int kv_heads,
    int lookahead_count,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    const T* __restrict__ q,
    const T* __restrict__ k,
    float* __restrict__ scores
) {
    extern __shared__ float lds[]; // kv_len floats
    const int lane = threadIdx.x;
    if (lane >= warpSize) return;

    const int row_idx = blockIdx.x;                  // 0 .. q_heads * lookahead_count
    const int q_row = row_idx % lookahead_count;
    const int q_head = row_idx / lookahead_count;
    if (q_head >= q_heads) return;
    const int kv_head = q_head / num_kv_groups;

    // q layout: [lookahead_count, q_heads, head_dim]
    const T* q_ptr = q + (static_cast<size_t>(q_row) * q_heads + q_head) * head_dim;
    // k layout: [kv_heads, kv_len, head_dim]
    const T* k_head = k + static_cast<size_t>(kv_head) * kv_len * head_dim;
    // scores layout: [q_heads, lookahead_count, kv_len]
    float* out_row = scores +
        (static_cast<size_t>(q_head) * lookahead_count + q_row) * kv_len;

    // Pass 1: compute scaled dot-products into LDS, track running max.
    __shared__ float shared_max;
    if (lane == 0) shared_max = -INFINITY;
    __syncthreads();

    for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
        const T* k_row = k_head + k_pos * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += warpSize) {
            partial += pfx_to_float(q_ptr[d]) * pfx_to_float(k_row[d]);
        }
        // Warp-reduction (existing supersonic_qwen35_wave_sum lives in
        // full_attention.hip; we keep prefill_helpers.hip self-contained
        // by inlining the reduction here.)
        partial += __shfl_xor(partial, 16);
        partial += __shfl_xor(partial, 8);
        partial += __shfl_xor(partial, 4);
        partial += __shfl_xor(partial, 2);
        partial += __shfl_xor(partial, 1);
        const float score = partial * scale;
        if (lane == 0) {
            lds[k_pos] = score;
            if (score > shared_max) shared_max = score;
        }
        __syncthreads();
    }

    // Pass 2: exponentiate (centred on shared_max), accumulate denom.
    __shared__ float shared_denom;
    if (lane == 0) shared_denom = 0.0f;
    __syncthreads();
    for (int k_pos = lane; k_pos < kv_len; k_pos += warpSize) {
        const float e = expf(lds[k_pos] - shared_max);
        lds[k_pos] = e;
        atomicAdd(&shared_denom, e);
    }
    __syncthreads();

    // Pass 3: normalise and store.
    const float inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
    for (int k_pos = lane; k_pos < kv_len; k_pos += warpSize) {
        out_row[k_pos] = lds[k_pos] * inv_denom;
    }
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
    const int src_h = oh / repeats;

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
