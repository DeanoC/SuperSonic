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
// INT4 dequant helpers.
//
// Runtime format matches the Qwen35 INT4 bake (intentional — see
// feedback_hipcc_codegen about not touching the Qwen kernels directly):
//   - Weights packed 2 nibbles per byte: W[row, col]
//     lives in byte `byte_cols * row + col/2`, low nibble for even col,
//     high nibble for odd col.
//   - BF16 scale + BF16 zero per `gs × gs` tile.
//     tile coords: (row / gs, col / gs); shape [out/gs, in/gs].
//   - Dequant per element: `q * scale - zero * scale`, rounded through
//     BF16 (matches Python's bf16-stored `Q_dq` in bake_int4_gemma4.py).
//
// `g4_bf16_round_rne_f32_finite` is a NaN-check-free BF16-round that's
// safe here because inputs are `nibble × bf16 scale − bf16 zero_scale`
// with all factors finite. Same math as the Qwen `bf16_round_rne_f32_finite`
// helper — duplicated here to keep gemma4.hip a standalone TU.
// -----------------------------------------------------------------------------

__device__ __forceinline__ float g4_bf16_round_rne_f32_finite(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    float y;
    __builtin_memcpy(&y, &bits, 4);
    return y;
}

// -----------------------------------------------------------------------------
// FP8-E4M3-FN helpers for KV-cache quantization. Mirrors phi4.hip's
// `fp8_e4m3_to_float` / `float_to_fp8_e4m3` — duplicated here to keep
// gemma4.hip a standalone TU (per CLAUDE.md, hipcc on gfx11xx is fragile to
// cross-TU contamination, so we never #include another model's .hip).
// E4M3-FN encoding: 1 sign + 4 exponent + 3 mantissa, bias=7, no inf,
// NaN encoded at 0x7F / 0xFF, max representable absolute value = 448.
// -----------------------------------------------------------------------------
__device__ inline float g4_fp8_e4m3_to_float(uint8_t byte) {
    int sign = (byte >> 7) & 1;
    int exp = (byte >> 3) & 0xF;
    int mantissa = byte & 0x7;
    if (byte == 0x7F || byte == 0xFF) return 0.0f;  // NaN → 0
    float val;
    if (exp == 0) {
        val = static_cast<float>(mantissa) / 8.0f * 1.52587890625e-2f;
    } else {
        val = (1.0f + static_cast<float>(mantissa) / 8.0f)
              * exp2f(static_cast<float>(exp - 7));
    }
    return sign ? -val : val;
}

// Per-block FP8 weight dequant via an LDS lookup table. `lut[i]` precomputes
// `g4_fp8_e4m3_to_float(i)` for i in 0..256, eliminating the branchy decoder
// from the matmul hot loop. Caller fills the LUT once at kernel entry (one
// thread per byte). `block_size` is typically 128 (matches the bake script).
//
// Returns the BF16-rounded F32 weight matching what `bake_fp8_gemma4.py`
// emits — preserving the round trip `bf16 → fp8 → bf16` so dequant agrees
// with PyTorch's reference. Mirrors `phi4.hip`'s `fp8_dequant_weight_lut`.
__device__ inline float g4_fp8_dequant_weight_lut(
    const void* w_ptr, const void* scale_ptr,
    int row, int col, int cols, int block_size,
    const float* __restrict__ lut
) {
    const uint8_t* fp8 = static_cast<const uint8_t*>(w_ptr);
    const float val = lut[fp8[static_cast<size_t>(row) * cols + col]];
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    const int scale_row = row / block_size;
    const int scale_col = col / block_size;
    const int scale_cols = (cols + block_size - 1) / block_size;
    const float scale = static_cast<float>(scales[scale_row * scale_cols + scale_col]);
    return g4_bf16_round_rne_f32_finite(val * scale);
}

__device__ inline uint8_t g4_float_to_fp8_e4m3(float val) {
    uint8_t sign = 0;
    if (val < 0.0f) { sign = 0x80; val = -val; }
    if (val >= 448.0f) return sign | 0x7E;
    if (val < 1.52587890625e-2f * 0.125f) return sign;
    if (val < 0.015625f) {
        int mantissa = __float2int_rn(val / 1.52587890625e-2f * 8.0f);
        if (mantissa < 0) mantissa = 0;
        if (mantissa >= 8) return sign | 0x08;
        return sign | static_cast<uint8_t>(mantissa);
    }
    float log2_val = log2f(val);
    int exp = static_cast<int>(floorf(log2_val)) + 7;
    if (exp < 1) exp = 1;
    if (exp > 14) exp = 14;
    float pow2 = exp2f(static_cast<float>(exp - 7));
    int mantissa = __float2int_rn((val / pow2 - 1.0f) * 8.0f);
    if (mantissa < 0) mantissa = 0;
    if (mantissa >= 8) {
        mantissa = 0;
        exp += 1;
        if (exp > 14) return sign | 0x7E;
    }
    return sign | (static_cast<uint8_t>(exp) << 3) | static_cast<uint8_t>(mantissa);
}

// Dequantize 8 INT4 weights from 4 packed bytes = 1 packed uint32.
// `col` is the starting weight column (must be 2-aligned; always is since
// we step by 8 across a row whose byte stride is `in_dim/2`).
__device__ inline void g4_int4_dequant_8(
    uint32_t packed,
    const hip_bfloat16* __restrict__ scales,
    const hip_bfloat16* __restrict__ zeros,
    int scale_row, int col, int scale_cols, int gsz,
    float out[8]
) {
    const int sb = scale_row * scale_cols;
    int n0 = (packed >>  0) & 0xF;
    int n1 = (packed >>  4) & 0xF;
    int n2 = (packed >>  8) & 0xF;
    int n3 = (packed >> 12) & 0xF;
    int n4 = (packed >> 16) & 0xF;
    int n5 = (packed >> 20) & 0xF;
    int n6 = (packed >> 24) & 0xF;
    int n7 = (packed >> 28) & 0xF;

    const int g0 = col / gsz;
    const int g7 = (col + 7) / gsz;
    if (g0 == g7) {
        const float s = static_cast<float>(scales[sb + g0]);
        const float zs = static_cast<float>(zeros[sb + g0]) * s;
        out[0] = g4_bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
        out[1] = g4_bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
        out[2] = g4_bf16_round_rne_f32_finite(static_cast<float>(n2) * s - zs);
        out[3] = g4_bf16_round_rne_f32_finite(static_cast<float>(n3) * s - zs);
        out[4] = g4_bf16_round_rne_f32_finite(static_cast<float>(n4) * s - zs);
        out[5] = g4_bf16_round_rne_f32_finite(static_cast<float>(n5) * s - zs);
        out[6] = g4_bf16_round_rne_f32_finite(static_cast<float>(n6) * s - zs);
        out[7] = g4_bf16_round_rne_f32_finite(static_cast<float>(n7) * s - zs);
    } else {
        const float s0 = static_cast<float>(scales[sb + g0]);
        const float zs0 = static_cast<float>(zeros[sb + g0]) * s0;
        const float s1 = static_cast<float>(scales[sb + g7]);
        const float zs1 = static_cast<float>(zeros[sb + g7]) * s1;
        #define G4_I4DQ(idx, ni) do { \
            const int gi = (col + idx) / gsz; \
            const float si = (gi == g0) ? s0 : s1; \
            const float zsi = (gi == g0) ? zs0 : zs1; \
            out[idx] = g4_bf16_round_rne_f32_finite(static_cast<float>(ni) * si - zsi); \
        } while (0)
        G4_I4DQ(0, n0); G4_I4DQ(1, n1); G4_I4DQ(2, n2); G4_I4DQ(3, n3);
        G4_I4DQ(4, n4); G4_I4DQ(5, n5); G4_I4DQ(6, n6); G4_I4DQ(7, n7);
        #undef G4_I4DQ
    }
}

// Scalar INT4 dequant at (row, col). Used by unit tests and for tail handling;
// the hot paths use `g4_int4_dequant_8` for 8-wide vectorization.
__device__ inline float g4_int4_dequant_scalar(
    const uint8_t* __restrict__ W_packed,
    const hip_bfloat16* __restrict__ scales,
    const hip_bfloat16* __restrict__ zeros,
    int row, int col, int cols, int gsz
) {
    const int byte_cols = cols / 2;
    const uint8_t byte = W_packed[static_cast<size_t>(row) * byte_cols + col / 2];
    const int nibble = (col & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    const int scale_cols = (cols + gsz - 1) / gsz;
    const int si = (row / gsz) * scale_cols + col / gsz;
    const float s = static_cast<float>(scales[si]);
    const float zs = static_cast<float>(zeros[si]) * s;
    return g4_bf16_round_rne_f32_finite(static_cast<float>(nibble) * s - zs);
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
// INT4 work-stealing matrix-vector multiply (single-token).
// Computes `out[r] = dot(dequant(W[r, :]), x)` using the same runtime format
// as Qwen's INT4 path: packed bytes [out_dim, in_dim/2], BF16 scale + zero
// [out_dim/gsz, in_dim/gsz]. Dequant round-through-BF16 matches the Python
// bake (see oracle/bake_int4_gemma4.py). Caller zeroes `row_counter` before
// launch; device-side loop claims one output row per iteration.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void g4_matvec_int4_workstealing_kernel(
    int in_dim,
    int out_dim,
    int gsz,
    const T* __restrict__ x,                      // [in_dim]
    const uint8_t* __restrict__ W_packed,         // [out_dim, in_dim/2]
    const hip_bfloat16* __restrict__ W_scale,     // [out_dim/gsz, in_dim/gsz]
    const hip_bfloat16* __restrict__ W_zero,      // [out_dim/gsz, in_dim/gsz]
    T* __restrict__ out,                          // [out_dim]
    unsigned int* __restrict__ row_counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int scale_cols = (in_dim + gsz - 1) / gsz;
    const int byte_cols = in_dim / 2;

    while (true) {
        int row;
        if (tid == 0) row = static_cast<int>(atomicAdd(row_counter, 1u));
        __shared__ int shared_row;
        if (tid == 0) shared_row = row;
        __syncthreads();
        row = shared_row;
        if (row >= out_dim) return;

        const int scale_row = row / gsz;
        const uint8_t* w_row_bytes = W_packed + static_cast<size_t>(row) * byte_cols;

        float partial = 0.0f;
        // Each thread processes 8 consecutive columns per step.
        for (int c0 = tid * 8; c0 < in_dim; c0 += block_size * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, W_scale, W_zero,
                              scale_row, c0, scale_cols, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                partial += w[k] * g4_to_float(x[c0 + k]);
            }
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

// =============================================================================
// RDNA3 WMMA BF16 kernel for Gemma 4 prefill matvec, shared-memory tiled.
//
// `g4_matvec_batched_kernel` is work-stealing and reduces one output element
// per thread block — fine for decode (seq_len=1) where parallelism comes from
// the `out_dim` axis, but it leaves the matrix units idle on prefill where
// seq_len >> 1 and the work naturally has a matmul shape
// (seq_len × in_dim × out_dim).
//
// This kernel runs the same operation as a 2D tiled WMMA GEMM: 64x64 output
// tile, 4 wavefronts per block arranged 2x2 (each wave owns a 32x32
// sub-region, a 2x2 grid of 16x16 WMMA sub-tiles), K-step 64, A/B staged
// through LDS. The previous one-wave-per-16x16-tile version was bandwidth-
// bound (~100x global-read amplification on Gemma prefill shapes).
//
// Grid  : (ceil(out_dim/64), ceil(seq_len/64), 1)
// Block : 128 threads (4 wavefronts)
//
// WMMA is a cross-lane instruction: every lane must participate with
// well-defined A/B inputs. We keep the WMMA call uniform across the wave
// and zero-pad out-of-range elements in LDS so divergent input can't
// corrupt in-range lanes' outputs.
// =============================================================================

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__)
#define G4_HAS_WMMA_BF16 1
#endif

typedef short  g4_short16 __attribute__((ext_vector_type(16)));
typedef float  g4_float8  __attribute__((ext_vector_type(8)));

// Tile dimensions. Shared by the BF16 and INT4 WMMA prefill kernels.
constexpr int G4_TILED_WMMA_BM = 64;
constexpr int G4_TILED_WMMA_BN = 64;
constexpr int G4_TILED_WMMA_BK = 64;
// LDS row padded by 1 DWORD (2 BF16) to reduce 32-bank aliasing on the
// WMMA operand gather.
constexpr int G4_TILED_WMMA_LDS_STRIDE = G4_TILED_WMMA_BK + 2;

__global__ __launch_bounds__(128, 2)
void g4_matvec_batched_wmma_bf16_kernel(
    int seq_len,
    int in_dim,
    int out_dim,
    const hip_bfloat16* __restrict__ x,    // [seq_len, in_dim]
    const hip_bfloat16* __restrict__ W,    // [out_dim, in_dim]
    hip_bfloat16* __restrict__ out         // [seq_len, out_dim]
) {
#ifdef G4_HAS_WMMA_BF16
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;       // 0..3
    const int wave_row = wid >> 1;       // 0 or 1
    const int wave_col = wid & 1;        // 0 or 1
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;

    const int blk_row = blockIdx.y * G4_TILED_WMMA_BM;
    const int blk_col = blockIdx.x * G4_TILED_WMMA_BN;

    __shared__ uint16_t s_x[G4_TILED_WMMA_BM * G4_TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_w[G4_TILED_WMMA_BN * G4_TILED_WMMA_LDS_STRIDE];

    const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x);
    const uint16_t* w_u16 = reinterpret_cast<const uint16_t*>(W);

    g4_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    static_assert(G4_TILED_WMMA_BK % 32 == 0, "BK must be a multiple of 32 (lane count)");
    constexpr int K_COL_HALVES = G4_TILED_WMMA_BK / 32;
    constexpr int K_CHUNKS     = G4_TILED_WMMA_BK / 16;

    const int k_main = in_dim & ~(G4_TILED_WMMA_BK - 1);

    for (int kk0 = 0; kk0 < k_main; kk0 += G4_TILED_WMMA_BK) {
        // Load A (x) tile [BM, BK]: wave wid loads rows wid*16..wid*16+15,
        // lane L loads col L across those rows. Coalesced global read.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gr < seq_len) {
                    v = x_u16[static_cast<size_t>(gr) * in_dim + gk];
                }
                s_x[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // Load B (W) tile [BN, BK] — W is [out_dim, in_dim] row-major
        // which matches the "virtual transpose" layout for the output dim.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gc < out_dim) {
                    v = w_u16[static_cast<size_t>(gc) * in_dim + gk];
                }
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        __syncthreads();

        // Compute: K_CHUNKS 16-K chunks, 4 WMMAs each (2x2 sub-tiles).
        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            g4_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_x[a0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_x[a1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_w[b0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_w[b1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc00);
            acc01 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc01);
            acc10 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc10);
            acc11 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc11);
        }
        __syncthreads();
    }

    // Defensive K-tail when in_dim % BK != 0. Gemma hidden/intermediate
    // dims are multiples of 256 so this branch is unreachable in practice.
    if (k_main != in_dim) {
        const int k_rem = in_dim - k_main;
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < seq_len && col < k_rem) {
                    v = x_u16[static_cast<size_t>(gr) * in_dim + k_main + col];
                }
                s_x[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gc < out_dim && col < k_rem) {
                    v = w_u16[static_cast<size_t>(gc) * in_dim + k_main + col];
                }
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            g4_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_x[a0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_x[a1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_w[b0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_w[b1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc00);
            acc01 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc01);
            acc10 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc10);
            acc11 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc11);
        }
    }

    // Store 4 sub-tiles. RDNA3 wave32 WMMA acc layout:
    //   acc[i] -> C[2*i + lane_half, lane_row] relative to sub-tile (0,0).
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const g4_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < out_dim) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int out_row = sub_row_base + sub_row_off + 2 * i + lane_half;
                if (out_row < seq_len) {
                    out[static_cast<size_t>(out_row) * out_dim + out_col] =
                        hip_bfloat16(acc[i]);
                }
            }
        }
    };
    store_sub(acc00, 0,  0);
    store_sub(acc01, 0,  16);
    store_sub(acc10, 16, 0);
    store_sub(acc11, 16, 16);
#else
    (void)seq_len; (void)in_dim; (void)out_dim;
    (void)x; (void)W; (void)out;
#endif
}

// Batched INT4 matvec. Computes `out[s, r] = dot(dequant(W[r, :]), x[s, :])`
// for every (s, r) pair. Work-stealing block claims one (s, r) item at a time
// from `counter`. Same runtime format as the single-token INT4 kernel.
template <typename T>
__global__ void g4_matvec_batched_int4_kernel(
    int seq_len,
    int in_dim,
    int out_dim,
    int gsz,
    const T* __restrict__ x,                      // [seq_len, in_dim]
    const uint8_t* __restrict__ W_packed,         // [out_dim, in_dim/2]
    const hip_bfloat16* __restrict__ W_scale,     // [out_dim/gsz, in_dim/gsz]
    const hip_bfloat16* __restrict__ W_zero,      // [out_dim/gsz, in_dim/gsz]
    T* __restrict__ out,                          // [seq_len, out_dim]
    unsigned int* __restrict__ counter
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int scale_cols = (in_dim + gsz - 1) / gsz;
    const int byte_cols = in_dim / 2;
    const size_t total = static_cast<size_t>(seq_len) * out_dim;

    while (true) {
        int idx;
        if (tid == 0) idx = static_cast<int>(atomicAdd(counter, 1u));
        __shared__ int shared_idx;
        if (tid == 0) shared_idx = idx;
        __syncthreads();
        idx = shared_idx;
        if (static_cast<size_t>(idx) >= total) return;

        const int s = idx / out_dim;
        const int r = idx % out_dim;

        const int scale_row = r / gsz;
        const uint8_t* w_row_bytes = W_packed + static_cast<size_t>(r) * byte_cols;
        const T* x_row = x + static_cast<size_t>(s) * in_dim;

        float partial = 0.0f;
        for (int c0 = tid * 8; c0 < in_dim; c0 += block_size * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, W_scale, W_zero,
                              scale_row, c0, scale_cols, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                partial += w[k] * g4_to_float(x_row[c0 + k]);
            }
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

// =============================================================================
// RDNA3 WMMA INT4-dequant prefill matvec (wave32, 16x16x16 f32-accumulate),
// one-wavefront-per-16x16-tile. Kept as a small-M fallback since the tiled
// version below suffers proportionally larger tile-overhang waste (4x)
// when seq_len is much smaller than the 64-row tile, and INT4 already has
// ~4x less bandwidth than BF16 for tiling to amortize — same tradeoff as
// the Qwen INT4 kernels in kernels/full_attention_4b.hip.
//
//   x        [seq_len, in_dim]               BF16
//   W_packed [out_dim, in_dim/2]             packed INT4 (low nibble = even col)
//   W_scale  [out_dim/gsz, in_dim/gsz]       BF16
//   W_zero   [out_dim/gsz, in_dim/gsz]       BF16
//   out      [seq_len, out_dim]              BF16
// =============================================================================

__global__ void g4_matvec_batched_int4_wmma_bf16_small_m_kernel(
    int seq_len,
    int in_dim,
    int out_dim,
    int gsz,
    const hip_bfloat16* __restrict__ x,          // [seq_len, in_dim]
    const uint8_t* __restrict__ W_packed,        // [out_dim, in_dim/2]
    const hip_bfloat16* __restrict__ W_scale,    // [out_dim/gsz, in_dim/gsz]
    const hip_bfloat16* __restrict__ W_zero,     // [out_dim/gsz, in_dim/gsz]
    hip_bfloat16* __restrict__ out               // [seq_len, out_dim]
) {
#ifdef G4_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int tile_row = blockIdx.y * 16;         // seq_len axis
    const int tile_col = blockIdx.x * 16;         // out_dim axis

    const int x_row_idx = tile_row + lane_row;
    const int w_row_idx = tile_col + lane_row;
    const bool x_in_range = x_row_idx < seq_len;
    const bool w_in_range = w_row_idx < out_dim;

    const int byte_cols = in_dim / 2;
    const int scale_cols = (in_dim + gsz - 1) / gsz;

    const uint16_t* x_row_u16 = reinterpret_cast<const uint16_t*>(x) +
                                static_cast<size_t>(x_row_idx) * in_dim;
    const uint8_t*  w_row_u8  = W_packed +
                                static_cast<size_t>(w_row_idx) * byte_cols;
    const int scale_row_base  = (w_row_idx / gsz) * scale_cols;

    g4_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = in_dim & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        // A (activations)
        g4_short16 a;
        if (x_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = static_cast<short>(x_row_u16[kk + i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        // B (weights, INT4 → BF16)
        g4_short16 b;
        if (w_in_range) {
            const int scale_idx = scale_row_base + (kk / gsz);
            const float s = static_cast<float>(W_scale[scale_idx]);
            const float zs = static_cast<float>(W_zero[scale_idx]) * s;
            const int byte_base = kk >> 1;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint8_t packed = w_row_u8[byte_base + i];
                int n0 = packed & 0xF;
                int n1 = (packed >> 4) & 0xF;
                float v0 = g4_bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
                float v1 = g4_bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
                uint32_t b0, b1;
                __builtin_memcpy(&b0, &v0, 4);
                __builtin_memcpy(&b1, &v1, 4);
                b[2 * i]     = static_cast<short>(b0 >> 16);
                b[2 * i + 1] = static_cast<short>(b1 >> 16);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    // Tail (in_dim % 16 != 0): Gemma 4 layers use multiples of 128, so this
    // is defensive only.
    if (k_main != in_dim) {
        g4_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        g4_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (x_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < in_dim) a[i] = static_cast<short>(x_row_u16[kc]);
            }
        }
        if (w_in_range) {
            const int scale_idx = scale_row_base + (k_main / gsz);
            const float s = static_cast<float>(W_scale[scale_idx]);
            const float zs = static_cast<float>(W_zero[scale_idx]) * s;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < in_dim) {
                    uint8_t packed = w_row_u8[kc >> 1];
                    int nib = (kc & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    float v = g4_bf16_round_rne_f32_finite(static_cast<float>(nib) * s - zs);
                    uint32_t bits;
                    __builtin_memcpy(&bits, &v, 4);
                    b[i] = static_cast<short>(bits >> 16);
                }
            }
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }

    const int out_col = tile_col + lane_row;
    if (out_col < out_dim) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + 2 * i + lane_half;
            if (out_row < seq_len) {
                out[static_cast<size_t>(out_row) * out_dim + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)seq_len; (void)in_dim; (void)out_dim; (void)gsz;
    (void)x; (void)W_packed; (void)W_scale; (void)W_zero; (void)out;
#endif
}

// =============================================================================
// RDNA3 WMMA INT4-dequant prefill matvec (wave32, 16x16x16 f32-accumulate),
// shared-memory tiled.
//
// Same input layout as the small_m variant above. Uses the same 64x64 tile
// shape as g4_matvec_batched_wmma_bf16_kernel above — 4 waves per block in
// 2x2, each owns a 32x32 sub-region; K-step = G4_TILED_WMMA_BK (64). The
// B-tile load dequantizes packed INT4 into BF16 while staging into LDS, one
// (scale, zero) fetch per BK-wide K slab per row.
//
// Bridge gates this kernel on (gsz % BK == 0) so every BK-aligned K slab
// stays inside one GPTQ group; shipped Gemma 4 bakes use gsz=128 so this
// holds.
// =============================================================================

__global__ __launch_bounds__(128, 2)
void g4_matvec_batched_int4_wmma_bf16_kernel(
    int seq_len,
    int in_dim,
    int out_dim,
    int gsz,
    const hip_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W_packed,
    const hip_bfloat16* __restrict__ W_scale,
    const hip_bfloat16* __restrict__ W_zero,
    hip_bfloat16* __restrict__ out
) {
#ifdef G4_HAS_WMMA_BF16
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;
    const int wave_row = wid >> 1;
    const int wave_col = wid & 1;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;

    const int blk_row = blockIdx.y * G4_TILED_WMMA_BM;
    const int blk_col = blockIdx.x * G4_TILED_WMMA_BN;

    __shared__ uint16_t s_x[G4_TILED_WMMA_BM * G4_TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_w[G4_TILED_WMMA_BN * G4_TILED_WMMA_LDS_STRIDE];

    const uint16_t* x_u16 = reinterpret_cast<const uint16_t*>(x);

    const int byte_cols  = in_dim / 2;
    const int scale_cols = (in_dim + gsz - 1) / gsz;

    g4_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    g4_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    static_assert(G4_TILED_WMMA_BK % 32 == 0, "BK must be a multiple of 32");
    static_assert(G4_TILED_WMMA_BK % 64 == 0, "BK must be a multiple of 64 (INT4 packs 2 K/byte, 32 lanes)");
    constexpr int K_COL_HALVES  = G4_TILED_WMMA_BK / 32;   // BF16 A load passes
    constexpr int K_INT4_PASSES = G4_TILED_WMMA_BK / 64;   // INT4 B load passes
    constexpr int K_CHUNKS      = G4_TILED_WMMA_BK / 16;

    const int k_main = in_dim & ~(G4_TILED_WMMA_BK - 1);

    for (int kk0 = 0; kk0 < k_main; kk0 += G4_TILED_WMMA_BK) {
        // Load A (x) tile [BM, BK] BF16.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gr < seq_len) {
                    v = x_u16[static_cast<size_t>(gr) * in_dim + gk];
                }
                s_x[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // Load + dequantize B (W) tile [BN, BK] INT4 -> BF16.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            float s = 0.f;
            float zs = 0.f;
            if (gc < out_dim) {
                int group_id = kk0 / gsz;
                int scale_idx = (gc / gsz) * scale_cols + group_id;
                float s_val = static_cast<float>(W_scale[scale_idx]);
                float z_val = static_cast<float>(W_zero[scale_idx]);
                s = s_val;
                zs = z_val * s_val;
            }
            #pragma unroll
            for (int pass = 0; pass < K_INT4_PASSES; ++pass) {
                int lane_byte = pass * 32 + lane;
                int col_lo    = lane_byte * 2;
                int col_hi    = col_lo + 1;
                uint8_t packed = 0;
                if (gc < out_dim) {
                    packed = W_packed[static_cast<size_t>(gc) * byte_cols +
                                      (kk0 >> 1) + lane_byte];
                }
                int n0 = packed & 0xF;
                int n1 = (packed >> 4) & 0xF;
                float v0 = g4_bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
                float v1 = g4_bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
                uint32_t b0, b1;
                __builtin_memcpy(&b0, &v0, 4);
                __builtin_memcpy(&b1, &v1, 4);
                uint16_t out_lo = (gc < out_dim) ? static_cast<uint16_t>(b0 >> 16) : 0;
                uint16_t out_hi = (gc < out_dim) ? static_cast<uint16_t>(b1 >> 16) : 0;
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col_lo] = out_lo;
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col_hi] = out_hi;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            g4_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_x[a0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_x[a1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_w[b0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_w[b1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc00);
            acc01 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc01);
            acc10 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc10);
            acc11 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc11);
        }
        __syncthreads();
    }

    // Defensive K-tail. Gemma dims are multiples of 256 so unreachable.
    if (k_main != in_dim) {
        const int k_rem = in_dim - k_main;
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < seq_len && col < k_rem) {
                    v = x_u16[static_cast<size_t>(gr) * in_dim + k_main + col];
                }
                s_x[lds_row * G4_TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            float s = 0.f;
            float zs = 0.f;
            if (gc < out_dim) {
                int group_id = k_main / gsz;
                int scale_idx = (gc / gsz) * scale_cols + group_id;
                float s_val = static_cast<float>(W_scale[scale_idx]);
                float z_val = static_cast<float>(W_zero[scale_idx]);
                s = s_val;
                zs = z_val * s_val;
            }
            #pragma unroll
            for (int pass = 0; pass < K_INT4_PASSES; ++pass) {
                int lane_byte = pass * 32 + lane;
                int col_lo = lane_byte * 2;
                int col_hi = col_lo + 1;
                uint16_t out_lo = 0;
                uint16_t out_hi = 0;
                if (gc < out_dim) {
                    uint8_t packed = 0;
                    if (col_lo < k_rem || col_hi < k_rem) {
                        packed = W_packed[static_cast<size_t>(gc) * byte_cols +
                                          (k_main >> 1) + lane_byte];
                    }
                    if (col_lo < k_rem) {
                        int n0 = packed & 0xF;
                        float v0 = g4_bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
                        uint32_t b0;
                        __builtin_memcpy(&b0, &v0, 4);
                        out_lo = static_cast<uint16_t>(b0 >> 16);
                    }
                    if (col_hi < k_rem) {
                        int n1 = (packed >> 4) & 0xF;
                        float v1 = g4_bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
                        uint32_t b1;
                        __builtin_memcpy(&b1, &v1, 4);
                        out_hi = static_cast<uint16_t>(b1 >> 16);
                    }
                }
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col_lo] = out_lo;
                s_w[lds_row * G4_TILED_WMMA_LDS_STRIDE + col_hi] = out_hi;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            g4_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_x[a0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_x[a1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_w[b0_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_w[b1_row * G4_TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b0, acc00);
            acc01 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b1, acc01);
            acc10 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b0, acc10);
            acc11 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b1, acc11);
        }
    }

    // Store 4 sub-tiles.
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const g4_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < out_dim) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int out_row = sub_row_base + sub_row_off + 2 * i + lane_half;
                if (out_row < seq_len) {
                    out[static_cast<size_t>(out_row) * out_dim + out_col] =
                        hip_bfloat16(acc[i]);
                }
            }
        }
    };
    store_sub(acc00, 0,  0);
    store_sub(acc01, 0,  16);
    store_sub(acc10, 16, 0);
    store_sub(acc11, 16, 16);
#else
    (void)seq_len; (void)in_dim; (void)out_dim; (void)gsz;
    (void)x; (void)W_packed; (void)W_scale; (void)W_zero; (void)out;
#endif
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

// Wave-level shuffle reduction (log2(warpSize) __shfl_xor instead of the
// 8 __syncthreads + LDS tree). Uses runtime warpSize so the matvec loops
// (which partition columns by warpSize * N) and this reduction agree on
// wave width across both RDNA (32-lane) and CDNA (64-lane) backends.
__device__ inline float g4_wave_reduce_sum_f32(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

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

// FP8-E4M3 variant of the KV append. Quantizes one decode token's K/V into
// the cache at position `pos`, with a per-(head, position) F32 absmax scale.
// The cache buffers are reinterpreted as `uint8_t*` (1 byte per element);
// `kv_scale_k` / `kv_scale_v` hold one F32 scale per (head, position).
//
// Block-collaborative absmax: one wave-wide reduction per K head, then per V
// head. `lds[bs]` is a scratch buffer in shared memory.
//
// Caller must call this from a single block. Threads cooperate; output is
// written at `kv_cache_{k,v}[h * max_T + pos][d]` and
// `kv_scale_{k,v}[h * max_T + pos]`.
__device__ inline void g4_block_kv_append_f32_to_fp8(
    const float* __restrict__ k_f32,
    const float* __restrict__ v_f32,
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    float* __restrict__ k_scale_buf,
    float* __restrict__ v_scale_buf,
    int num_kv_heads,
    int head_dim,
    int pos,
    int max_T,
    float* __restrict__ lds
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    for (int h = 0; h < num_kv_heads; ++h) {
        // Per-head absmax over [head_dim] — bf16-rounded values match the
        // BF16-round in the existing append path, keeping the FP8 dequant
        // numerically aligned with the BF16 reference.
        //
        // Strided local max FIRST, then block reduce. Gemma 4 full-attention
        // layers have head_dim=512 with blockDim.x=256, so each thread must
        // walk all its lanes before reducing — otherwise dimensions >= bs
        // (i.e. half the head on full-attn) silently drop out of the absmax
        // and the chosen scale undershoots, causing widespread FP8
        // saturation/clamping. (Reported by Codex review on PR #48.)
        float my_k = 0.0f, my_v = 0.0f;
        for (int d = tid; d < head_dim; d += bs) {
            const float k_bf16 =
                g4_bf16_round_rne_f32_finite(k_f32[h * head_dim + d]);
            const float v_bf16 =
                g4_bf16_round_rne_f32_finite(v_f32[h * head_dim + d]);
            my_k = fmaxf(my_k, fabsf(k_bf16));
            my_v = fmaxf(my_v, fabsf(v_bf16));
        }
        lds[tid] = my_k;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
            __syncthreads();
        }
        const float k_scale = fmaxf(lds[0] / 448.0f, 1e-12f);
        const float k_inv = 1.0f / k_scale;
        if (tid == 0) {
            k_scale_buf[static_cast<size_t>(h) * max_T + pos] = k_scale;
        }
        __syncthreads();

        lds[tid] = my_v;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
            __syncthreads();
        }
        const float v_scale = fmaxf(lds[0] / 448.0f, 1e-12f);
        const float v_inv = 1.0f / v_scale;
        if (tid == 0) {
            v_scale_buf[static_cast<size_t>(h) * max_T + pos] = v_scale;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += bs) {
            const size_t off =
                (static_cast<size_t>(h) * max_T + pos) * head_dim + d;
            const float k_store =
                g4_bf16_round_rne_f32_finite(k_f32[h * head_dim + d]);
            const float v_store =
                g4_bf16_round_rne_f32_finite(v_f32[h * head_dim + d]);
            k_cache[off] = g4_float_to_fp8_e4m3(k_store * k_inv);
            v_cache[off] = g4_float_to_fp8_e4m3(v_store * v_inv);
        }
        __syncthreads();
    }
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
// Fused attention-block kernel — INT4 variant (Step 28).
//
// Structurally identical to `g4_fused_attn_block_kernel` above: same phase
// order, same workspace layout, same F32 intermediates, same grid_barrier
// synchronization. The only behavioral difference is that Phase 2 (Q/K/V
// work-stealing matmul) and Phase 7 (o_proj work-stealing matmul) route
// their inner column-reduction loops through `g4_int4_dequant_8` instead of
// the direct `bf16 -> float` load. All other phases (norms, RoPE, KV append,
// SWA/full attention, post_attn_norm, residual) are byte-for-byte copies of
// the BF16 kernel's phases — duplicated intentionally per the feedback rule
// about not touching already-validated kernels (see feedback_hipcc_codegen).
//
// Weight format:
//   q/k/v/o_proj_packed : uint8_t [out, in/2]    (2 nibbles per byte)
//   q/k/v/o_proj_scale  : bf16   [out/gsz, in/gsz]
//   q/k/v/o_proj_zero   : bf16   [out/gsz, in/gsz]
// Dequant math: `bf16_round_rne(nibble * scale - zero_scale)` via
// `g4_int4_dequant_8` — 8 weights per 4-byte packed load. Hot-path guaranteed
// when `gsz | in_dim` and columns are 8-aligned (always true for the Gemma 4
// projection shapes: hidden_size=1536/2560, intermediate=6144/10240/12288, q_dim
// multiples of 256 — all multiples of 128).
// =============================================================================

template <typename T>
__global__ void g4_fused_attn_block_int4_kernel(
    int hidden_size, int num_q_heads, int num_kv_heads, int head_dim,
    int rotary_dim, int sliding_window, int position, int max_T,
    int shared_kv,              // 0 = compute+store K/V; non-zero = inherit cache
    int gsz,                    // INT4 group size (bake fixes this at 128)
    float eps, float scale,
    const T* __restrict__ hidden_in,
    T* __restrict__ hidden_out,
    const T* __restrict__ input_norm_w,
    const uint8_t*       __restrict__ q_proj_packed,
    const hip_bfloat16*  __restrict__ q_proj_scale,
    const hip_bfloat16*  __restrict__ q_proj_zero,
    const uint8_t*       __restrict__ k_proj_packed,
    const hip_bfloat16*  __restrict__ k_proj_scale,
    const hip_bfloat16*  __restrict__ k_proj_zero,
    const uint8_t*       __restrict__ v_proj_packed,
    const hip_bfloat16*  __restrict__ v_proj_scale,
    const hip_bfloat16*  __restrict__ v_proj_zero,
    const T* __restrict__ q_norm_w,
    const T* __restrict__ k_norm_w,
    const uint8_t*       __restrict__ o_proj_packed,
    const hip_bfloat16*  __restrict__ o_proj_scale,
    const hip_bfloat16*  __restrict__ o_proj_zero,
    const T* __restrict__ post_attn_norm_w,
    const T* __restrict__ cos_table,
    const T* __restrict__ sin_table,
    T* __restrict__ k_cache,
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

    // Workspace slots (same layout as BF16 kernel).
    float* hidden_f32     = workspace;
    float* normed_f32     = hidden_f32 + hidden_size;
    float* proj_f32       = normed_f32 + hidden_size;
    float* scores_f32     = proj_f32 + (q_dim + 2 * kv_dim);
    float* attn_out_f32   = scores_f32 + num_q_heads * max_T;
    float* oproj_f32      = attn_out_f32 + q_dim;
    float* oproj_normed   = oproj_f32 + hidden_size;

    extern __shared__ float lds[];

    // Phase 0: hidden BF16 → F32 (residual copy).
    for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
        hidden_f32[c] = g4_to_float(hidden_in[c]);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 1: input_norm on block 0.
    if (blockIdx.x == 0) {
        g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                  hidden_size, eps, lds);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 2: work-stealing Q/K/V matmul with INT4 dequant.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int scale_cols_in = (hidden_size + gsz - 1) / gsz;
    const int byte_cols_in = hidden_size / 2;

    const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(total_qkv)) break;

        const uint8_t*      w_packed_base;
        const hip_bfloat16* w_scale_base;
        const hip_bfloat16* w_zero_base;
        int weight_row;
        int dst_off;
        if (row < static_cast<unsigned int>(q_dim)) {
            w_packed_base = q_proj_packed;
            w_scale_base  = q_proj_scale;
            w_zero_base   = q_proj_zero;
            weight_row    = static_cast<int>(row);
            dst_off       = static_cast<int>(row);
        } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
            w_packed_base = k_proj_packed;
            w_scale_base  = k_proj_scale;
            w_zero_base   = k_proj_zero;
            weight_row    = static_cast<int>(row - q_dim);
            dst_off       = static_cast<int>(row);
        } else {
            w_packed_base = v_proj_packed;
            w_scale_base  = v_proj_scale;
            w_zero_base   = v_proj_zero;
            weight_row    = static_cast<int>(row - q_dim - kv_dim);
            dst_off       = static_cast<int>(row);
        }
        const int scale_row = weight_row / gsz;
        const uint8_t* w_row_bytes =
            w_packed_base + static_cast<size_t>(weight_row) * byte_cols_in;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                              scale_row, c0, scale_cols_in, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * normed_f32[c0 + k];
            }
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

    // Phase 3: q/k/v norms + RoPE + KV append (block 0 serializes).
    if (!shared_kv && blockIdx.x == 0) {
        g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
            num_q_heads, head_dim, eps, lds);
        g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
            num_kv_heads, head_dim, eps, lds);
        g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
            num_kv_heads, head_dim, eps, lds);
        g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
            num_q_heads, head_dim, rotary_dim, position);
        g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
            num_kv_heads, head_dim, rotary_dim, position);
        g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache, v_cache,
            num_kv_heads, head_dim, position, max_T);
    } else if (shared_kv && blockIdx.x == 0) {
        g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
            num_q_heads, head_dim, eps, lds);
        g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
            num_q_heads, head_dim, rotary_dim, position);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    // Phase 4: attention scores (SWA or full).
    const int kv_len = position + 1;
    const int last = kv_len - 1;
    const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
    const int num_q_per_kv = num_q_heads / num_kv_heads;
    {
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

    // Phase 5: softmax (one block per q-head).
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

    // Phase 6: value aggregation.
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

    // Phase 7: work-stealing o_proj with INT4 dequant.
    //   in_dim  = q_dim
    //   out_dim = hidden_size
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int scale_cols_o = (q_dim + gsz - 1) / gsz;
    const int byte_cols_o = q_dim / 2;

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;

        const int scale_row = static_cast<int>(row) / gsz;
        const uint8_t* w_row_bytes =
            o_proj_packed + static_cast<size_t>(row) * byte_cols_o;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < q_dim; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, o_proj_scale, o_proj_zero,
                              scale_row, c0, scale_cols_o, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * attn_out_f32[c0 + k];
            }
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

    // Phase 8: post_attn_norm.
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

// =============================================================================
// Fused MLP + PLE kernel — INT4 variant (Step 30).
//
// Structural clone of `g4_fused_mlp_ple_kernel` above with the four matmul
// phases (gate+up concat, down, per_layer_input_gate, per_layer_projection)
// retargeted to INT4 work-stealing via `g4_int4_dequant_8`. Every non-matmul
// phase (norms, gelu*up, residuals, post_ple_norm, layer_scalar multiply,
// BF16 store) is byte-for-byte identical to the BF16 kernel — intentional
// duplication per `feedback_hipcc_codegen`.
//
// Weight format:
//   gate/up/down/ple_gate/ple_proj_packed : uint8_t [out, in/2]
//   gate/up/down/ple_gate/ple_proj_scale  : bf16   [out/gsz, in/gsz]
//   gate/up/down/ple_gate/ple_proj_zero   : bf16   [out/gsz, in/gsz]
//
// Dimensions per projection (Gemma 4 E2B / E4B):
//   gate_proj, up_proj           : in=hidden, out=intermediate
//   down_proj                    : in=intermediate, out=hidden
//   per_layer_input_gate         : in=hidden, out=ple_hidden
//   per_layer_projection         : in=ple_hidden, out=hidden
// All multiples of group_size=128.
//
// Workspace layout matches the BF16 kernel bit-for-bit
// (`g4::fused_mlp_ple_workspace_elems` is reused to size the scratch).
// =============================================================================

template <typename T>
__global__ void g4_fused_mlp_ple_int4_kernel(
    int hidden_size, int intermediate_size, int ple_hidden,
    int gsz,
    float eps, float layer_scalar,
    const T* __restrict__ hidden_in,
    T* __restrict__ hidden_out,
    const T* __restrict__ pre_ff_norm_w,
    const uint8_t*       __restrict__ gate_proj_packed,
    const hip_bfloat16*  __restrict__ gate_proj_scale,
    const hip_bfloat16*  __restrict__ gate_proj_zero,
    const uint8_t*       __restrict__ up_proj_packed,
    const hip_bfloat16*  __restrict__ up_proj_scale,
    const hip_bfloat16*  __restrict__ up_proj_zero,
    const uint8_t*       __restrict__ down_proj_packed,
    const hip_bfloat16*  __restrict__ down_proj_scale,
    const hip_bfloat16*  __restrict__ down_proj_zero,
    const T* __restrict__ post_ff_norm_w,
    const T* __restrict__ per_layer_input,
    const uint8_t*       __restrict__ per_layer_input_gate_packed,
    const hip_bfloat16*  __restrict__ per_layer_input_gate_scale,
    const hip_bfloat16*  __restrict__ per_layer_input_gate_zero,
    const uint8_t*       __restrict__ per_layer_projection_packed,
    const hip_bfloat16*  __restrict__ per_layer_projection_scale,
    const hip_bfloat16*  __restrict__ per_layer_projection_zero,
    const T* __restrict__ post_per_layer_input_norm_w,
    float* __restrict__ workspace,
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    // Workspace slots — identical layout to the BF16 kernel.
    float* hidden_f32      = workspace;
    float* normed_ff       = hidden_f32 + hidden_size;
    float* gate_up_f32     = normed_ff + hidden_size;
    float* gated_proj_f32  = gate_up_f32 + 2 * intermediate_size;
    float* down_out_f32    = gated_proj_f32 + intermediate_size;
    float* normed_down_f32 = down_out_f32 + hidden_size;
    float* h_pre_ple_f32   = normed_down_f32 + hidden_size;
    float* gated_ple_f32   = h_pre_ple_f32 + hidden_size;
    float* gated_act_ple   = gated_ple_f32 + ple_hidden;
    float* projected_ple   = gated_act_ple + ple_hidden;
    float* normed_ple_f32  = projected_ple + hidden_size;

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

    // Phase 2: work-stealing gate+up INT4 matmul (concatenated as [0, intermediate)
    // = gate, [intermediate, 2*intermediate) = up). in_dim = hidden_size for both.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int scale_cols_h = (hidden_size + gsz - 1) / gsz;
    const int byte_cols_h  = hidden_size / 2;

    const int total_mlp1 = 2 * intermediate_size;
    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(total_mlp1)) break;

        const uint8_t*      w_packed_base;
        const hip_bfloat16* w_scale_base;
        const hip_bfloat16* w_zero_base;
        int weight_row;
        if (row < static_cast<unsigned int>(intermediate_size)) {
            w_packed_base = gate_proj_packed;
            w_scale_base  = gate_proj_scale;
            w_zero_base   = gate_proj_zero;
            weight_row    = static_cast<int>(row);
        } else {
            w_packed_base = up_proj_packed;
            w_scale_base  = up_proj_scale;
            w_zero_base   = up_proj_zero;
            weight_row    = static_cast<int>(row - intermediate_size);
        }
        const int scale_row = weight_row / gsz;
        const uint8_t* w_row_bytes =
            w_packed_base + static_cast<size_t>(weight_row) * byte_cols_h;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                              scale_row, c0, scale_cols_h, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * normed_ff[c0 + k];
            }
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

    // Phase 4: work-stealing down_proj INT4 matmul.
    //   in_dim = intermediate_size, out_dim = hidden_size.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int scale_cols_i = (intermediate_size + gsz - 1) / gsz;
    const int byte_cols_i  = intermediate_size / 2;

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;

        const int scale_row = static_cast<int>(row) / gsz;
        const uint8_t* w_row_bytes =
            down_proj_packed + static_cast<size_t>(row) * byte_cols_i;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < intermediate_size; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed, down_proj_scale, down_proj_zero,
                              scale_row, c0, scale_cols_i, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * gated_proj_f32[c0 + k];
            }
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

    // Phase 7: work-stealing per_layer_input_gate INT4 matmul.
    //   in_dim = hidden_size, out_dim = ple_hidden.
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

        const int scale_row = static_cast<int>(row) / gsz;
        const uint8_t* w_row_bytes =
            per_layer_input_gate_packed + static_cast<size_t>(row) * byte_cols_h;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed,
                              per_layer_input_gate_scale,
                              per_layer_input_gate_zero,
                              scale_row, c0, scale_cols_h, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * h_pre_ple_f32[c0 + k];
            }
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

    // Phase 9: work-stealing per_layer_projection INT4 matmul.
    //   in_dim = ple_hidden, out_dim = hidden_size.
    if (blockIdx.x == 0 && tid == 0) {
        __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
    }
    g4_grid_barrier(barrier_counter, barrier_flag, nb);

    const int scale_cols_ple = (ple_hidden + gsz - 1) / gsz;
    const int byte_cols_ple  = ple_hidden / 2;

    while (true) {
        __shared__ unsigned int claimed;
        if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
        __syncthreads();
        const unsigned int row = claimed;
        if (row >= static_cast<unsigned int>(hidden_size)) break;

        const int scale_row = static_cast<int>(row) / gsz;
        const uint8_t* w_row_bytes =
            per_layer_projection_packed + static_cast<size_t>(row) * byte_cols_ple;

        float p = 0.0f;
        for (int c0 = tid * 8; c0 < ple_hidden; c0 += bs * 8) {
            uint32_t packed;
            __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
            float w[8];
            g4_int4_dequant_8(packed,
                              per_layer_projection_scale,
                              per_layer_projection_zero,
                              scale_row, c0, scale_cols_ple, gsz, w);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                p += w[k] * gated_act_ple[c0 + k];
            }
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

// =============================================================================
// Persistent decode megakernel — one launch runs the entire Gemma 4 forward
// pass for a single decode token across all `num_layers` decoder layers.
// Matches the layer-by-layer dispatch pattern of Qwen's
// `dotcache_qwen35_persistent_decode_kernel` but with Gemma 4's dual attention
// shapes (sliding vs. full) and the PLE branch after the MLP.
//
// Per-layer work copies the phase sequence from `g4_fused_attn_block_kernel`
// followed by `g4_fused_mlp_ple_kernel`. The existing fused kernels remain
// untouched — this kernel is a standalone __global__ body to avoid any
// codegen perturbation of the already-validated primitives.
//
// Layer-array descriptor — binary-compatible with
// `Gemma4DecodeLayerDesc` in `crates/kernel-ffi/src/gemma4.rs`. Field order
// must match exactly.
struct Gemma4DecodeLayerDesc {
    int   layer_type;            // 0 = sliding, 1 = full
    int   shared_kv;             // 0 = owns cache; 1 = aliases source layer
    int   num_q_heads;
    int   num_kv_heads;
    int   head_dim;
    int   rotary_dim;
    int   sliding_window;
    int   intermediate_size;
    int   kv_max_t;
    float layer_scalar;
    const void* input_norm_w;
    const void* q_proj_w;
    const void* k_proj_w;
    const void* v_proj_w;
    const void* q_norm_w;
    const void* k_norm_w;
    const void* o_proj_w;
    const void* post_attn_norm_w;
    const void* pre_ff_norm_w;
    const void* gate_proj_w;
    const void* up_proj_w;
    const void* down_proj_w;
    const void* post_ff_norm_w;
    const void* per_layer_input_gate_w;
    const void* per_layer_projection_w;
    const void* post_per_layer_input_norm_w;
    const void* cos_table;
    const void* sin_table;
    void* kv_cache_k;
    void* kv_cache_v;
};

// Mirror of `Gemma4Int4ScaleDesc` in `crates/kernel-ffi/src/gemma4.rs`.
// Parallel-struct to `Gemma4DecodeLayerDesc` — the main desc's `*_w` slots
// hold packed INT4 weights (reinterpreted at the kernel site) and this
// struct carries the matching BF16 scale/zero tables. Field order must
// match the Rust `#[repr(C)]` declaration exactly.
struct Gemma4Int4ScaleDesc {
    const void* q_proj_scale;
    const void* q_proj_zero;
    const void* k_proj_scale;
    const void* k_proj_zero;
    const void* v_proj_scale;
    const void* v_proj_zero;
    const void* o_proj_scale;
    const void* o_proj_zero;
    const void* gate_proj_scale;
    const void* gate_proj_zero;
    const void* up_proj_scale;
    const void* up_proj_zero;
    const void* down_proj_scale;
    const void* down_proj_zero;
    const void* per_layer_input_gate_scale;
    const void* per_layer_input_gate_zero;
    const void* per_layer_projection_scale;
    const void* per_layer_projection_zero;
    int         group_size;
};

// Mirror of `Gemma4BatchSeqDesc` in `crates/kernel-ffi/src/gemma4.rs`. One
// per layer (parallel to `Gemma4DecodeLayerDesc`), holding per-sequence
// state arrays for the batched persistent decode kernel. Field order and
// array length must match the Rust `#[repr(C)]` declaration exactly.
//
// `G4_MAX_BATCH_SIZE` mirrors `kernel_ffi::layer_desc::MAX_BATCH_SIZE` (8).
// Only the first `batch_size` entries are populated; remaining slots are
// ignored by the kernel.
#define G4_MAX_BATCH_SIZE 8
struct Gemma4BatchSeqDesc {
    int   seqlen_offset[G4_MAX_BATCH_SIZE];
    void* kv_cache_k[G4_MAX_BATCH_SIZE];
    void* kv_cache_v[G4_MAX_BATCH_SIZE];
    int   kv_max_t[G4_MAX_BATCH_SIZE];
};

// Mirror of `Gemma4KVCacheFp8Desc` in `crates/kernel-ffi/src/gemma4.rs`.
// Parallel-struct to `Gemma4DecodeLayerDesc` — when --kv-fp8 is active the
// main desc's `kv_cache_k` / `kv_cache_v` slots hold u8-packed FP8-E4M3
// bytes; this struct carries the matching per-(head, position) F32 absmax
// scales. Both fields are nullptr in BF16 KV mode (current single-batch
// kernel uses non-null as the dispatch signal).
struct Gemma4KVCacheFp8Desc {
    void* kv_scale_k;  // [num_kv_heads, max_T] F32, or nullptr for BF16 KV
    void* kv_scale_v;  // [num_kv_heads, max_T] F32, or nullptr for BF16 KV
};

// Mirror of `Gemma4FP8ScaleDesc` in `crates/kernel-ffi/src/gemma4.rs`.
// Parallel-struct to `Gemma4DecodeLayerDesc` — under --fp8-runtime the main
// desc's projection slots hold u8-packed FP8 bytes (reinterpreted at the
// matmul site) and this struct carries the matching BF16 scale_inv tiles.
struct Gemma4FP8ScaleDesc {
    const void* q_proj_scale;
    const void* k_proj_scale;  // null when shared_kv
    const void* v_proj_scale;  // null when shared_kv
    const void* o_proj_scale;
    const void* gate_proj_scale;
    const void* up_proj_scale;
    const void* down_proj_scale;
    const void* per_layer_input_gate_scale;
    const void* per_layer_projection_scale;
    int         block_size;
};

template <typename T>
__global__ void g4_persistent_decode_kernel(
    int num_layers,
    int hidden_size,
    int ple_hidden,
    int position,
    float eps,
    float scale,
    const Gemma4DecodeLayerDesc* __restrict__ layers,
    const Gemma4KVCacheFp8Desc* __restrict__ kv_fp8_descs, // null = BF16 KV
    const Gemma4FP8ScaleDesc* __restrict__ fp8_scales,     // null = BF16 weights
    T* __restrict__ hidden_io,                          // [hidden_size] BF16 in/out
    const T* __restrict__ per_layer_inputs,              // [num_layers, ple_hidden] BF16
    float* __restrict__ workspace,
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    // FP8-runtime weight dequant LUT — populated once per launch when active.
    // Body wiring in PR-B; for now `fp8_scales` is parameter-only so the
    // binary contract is stable and the engine PR can flip the flag without
    // another bridge/FFI change. Reads of `fp8_scales` are no-ops here.
    (void)fp8_scales;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    extern __shared__ float lds[];

    for (int layer = 0; layer < num_layers; ++layer) {
        const Gemma4DecodeLayerDesc L = layers[layer];
        const int num_q_heads = L.num_q_heads;
        const int num_kv_heads = L.num_kv_heads;
        const int head_dim = L.head_dim;
        const int rotary_dim = L.rotary_dim;
        const int sliding_window = L.sliding_window;
        const int shared_kv = L.shared_kv;
        const int max_T = L.kv_max_t;
        const int q_dim = num_q_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;

        // FP8 KV cache scales for THIS layer. Shared-KV layers receive scale
        // pointers aliased to the source layer's scale tensors via the engine,
        // matching how `kv_cache_k` / `kv_cache_v` are aliased. nullptr in
        // BF16 KV mode (and on layers we never wrote scales for).
        const float* kv_scale_k = (kv_fp8_descs != nullptr)
            ? static_cast<const float*>(kv_fp8_descs[layer].kv_scale_k)
            : nullptr;
        const float* kv_scale_v = (kv_fp8_descs != nullptr)
            ? static_cast<const float*>(kv_fp8_descs[layer].kv_scale_v)
            : nullptr;
        const bool use_fp8_kv = (kv_scale_k != nullptr) && (kv_scale_v != nullptr);

        const T* input_norm_w = static_cast<const T*>(L.input_norm_w);
        const T* q_proj_w = static_cast<const T*>(L.q_proj_w);
        const T* k_proj_w = static_cast<const T*>(L.k_proj_w);
        const T* v_proj_w = static_cast<const T*>(L.v_proj_w);
        const T* q_norm_w = static_cast<const T*>(L.q_norm_w);
        const T* k_norm_w = static_cast<const T*>(L.k_norm_w);
        const T* o_proj_w = static_cast<const T*>(L.o_proj_w);
        const T* post_attn_norm_w = static_cast<const T*>(L.post_attn_norm_w);
        const T* cos_table = static_cast<const T*>(L.cos_table);
        const T* sin_table = static_cast<const T*>(L.sin_table);
        // When use_fp8_kv is true, kv_cache_k/v hold u8 FP8-E4M3 bytes; we
        // reinterpret at the read/write sites to keep the BF16 path's
        // template `T*` typing untouched everywhere else.
        T* k_cache = static_cast<T*>(L.kv_cache_k);
        T* v_cache = static_cast<T*>(L.kv_cache_v);

        // =========================================================
        // Phase A — attention block. Mirrors g4_fused_attn_block_kernel.
        // Workspace layout (from offset 0):
        //   hidden_f32 [hidden]
        //   normed_f32 [hidden]
        //   proj_f32   [q_dim + 2*kv_dim]
        //   scores_f32 [num_q_heads * max_T]
        //   attn_out   [q_dim]
        //   oproj_f32  [hidden]
        //   oproj_norm [hidden]
        // =========================================================
        float* hidden_f32   = workspace;
        float* normed_f32   = hidden_f32 + hidden_size;
        float* proj_f32     = normed_f32 + hidden_size;
        float* scores_f32   = proj_f32 + (q_dim + 2 * kv_dim);
        float* attn_out_f32 = scores_f32 + num_q_heads * max_T;
        float* oproj_f32    = attn_out_f32 + q_dim;
        float* oproj_normed = oproj_f32 + hidden_size;

        // A0: load hidden_io BF16 → hidden_f32 (residual).
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            hidden_f32[c] = g4_to_float(hidden_io[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A1: input_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A2: work-stealing QKV matmul (reset counter first).
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // Wave-per-row QKV matmul: each wave independently steals a row via
        // atomicAdd + __shfl broadcast, accumulates with 4-wide BF16 loads,
        // reduces with wave shuffle. No __syncthreads per row.
        {
            const int lane = tid % warpSize;
            const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(total_qkv)) break;

                const T* w;
                int weight_row;
                if (row < static_cast<unsigned int>(q_dim)) {
                    w = q_proj_w;
                    weight_row = static_cast<int>(row);
                } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
                    w = k_proj_w;
                    weight_row = static_cast<int>(row - q_dim);
                } else {
                    w = v_proj_w;
                    weight_row = static_cast<int>(row - q_dim - kv_dim);
                }
                const T* w_row =
                    w + static_cast<size_t>(weight_row) * hidden_size;

                float p = 0.0f;
                const int vd4 = hidden_size & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(w_row[c]);
                    float w1 = g4_to_float(w_row[c+1]);
                    float w2 = g4_to_float(w_row[c+2]);
                    float w3 = g4_to_float(w_row[c+3]);
                    p += w0 * normed_f32[c]   + w1 * normed_f32[c+1]
                       + w2 * normed_f32[c+2] + w3 * normed_f32[c+3];
                }
                for (int c = vd4 + lane; c < hidden_size; c += warpSize) {
                    p += g4_to_float(w_row[c]) * normed_f32[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) proj_f32[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        float* q_f32 = proj_f32;
        float* k_f32 = proj_f32 + q_dim;
        float* v_f32 = proj_f32 + q_dim + kv_dim;

        // A3: q/k/v norms + RoPE + KV append (block 0 only).
        if (!shared_kv && blockIdx.x == 0) {
            g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                num_q_heads, head_dim, eps, lds);
            g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
                num_kv_heads, head_dim, eps, lds);
            g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
                num_kv_heads, head_dim, eps, lds);
            g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                num_q_heads, head_dim, rotary_dim, position);
            g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
                num_kv_heads, head_dim, rotary_dim, position);
            if (use_fp8_kv) {
                g4_block_kv_append_f32_to_fp8(
                    k_f32, v_f32,
                    reinterpret_cast<uint8_t*>(k_cache),
                    reinterpret_cast<uint8_t*>(v_cache),
                    const_cast<float*>(kv_scale_k),
                    const_cast<float*>(kv_scale_v),
                    num_kv_heads, head_dim, position, max_T, lds);
            } else {
                g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache, v_cache,
                    num_kv_heads, head_dim, position, max_T);
            }
        } else if (shared_kv && blockIdx.x == 0) {
            g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                num_q_heads, head_dim, eps, lds);
            g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                num_q_heads, head_dim, rotary_dim, position);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A4: attention scores.
        const int kv_len = position + 1;
        const int last = kv_len - 1;
        const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
        const int num_q_per_kv = num_q_heads / num_kv_heads;
        {
            const int total_items = num_q_heads * kv_len;
            const uint8_t* k_cache_fp8 = use_fp8_kv
                ? reinterpret_cast<const uint8_t*>(L.kv_cache_k)
                : nullptr;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int t = item % kv_len;
                const int q_h = item / kv_len;
                float val;
                if (t < min_t) {
                    val = -INFINITY;
                } else {
                    const int kv_h = q_h / num_q_per_kv;
                    const float* qr = q_f32 +
                        static_cast<size_t>(q_h) * head_dim;
                    float acc = 0.0f;
                    if (use_fp8_kv) {
                        const uint8_t* kr_fp8 = k_cache_fp8 +
                            (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                        const float k_s =
                            kv_scale_k[static_cast<size_t>(kv_h) * max_T + t];
                        for (int d = 0; d < head_dim; ++d) {
                            acc += qr[d] * (g4_fp8_e4m3_to_float(kr_fp8[d]) * k_s);
                        }
                    } else {
                        const T* kr = k_cache +
                            (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                        for (int d = 0; d < head_dim; ++d) {
                            acc += qr[d] * g4_to_float(kr[d]);
                        }
                    }
                    val = acc * scale;
                }
                scores_f32[static_cast<size_t>(q_h) * max_T + t] = val;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A5: softmax per q head.
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

        // A6: value aggregation.
        {
            const int total_items = num_q_heads * head_dim;
            const uint8_t* v_cache_fp8 = use_fp8_kv
                ? reinterpret_cast<const uint8_t*>(L.kv_cache_v)
                : nullptr;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int d = item % head_dim;
                const int q_h = item / head_dim;
                const int kv_h = q_h / num_q_per_kv;
                const float* pr = scores_f32 +
                    static_cast<size_t>(q_h) * max_T;
                float acc = 0.0f;
                if (use_fp8_kv) {
                    for (int t = 0; t < kv_len; ++t) {
                        const uint8_t* vr_fp8 = v_cache_fp8 +
                            (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                        const float v_s =
                            kv_scale_v[static_cast<size_t>(kv_h) * max_T + t];
                        acc += pr[t] * (g4_fp8_e4m3_to_float(vr_fp8[d]) * v_s);
                    }
                } else {
                    for (int t = 0; t < kv_len; ++t) {
                        const T* vr = v_cache +
                            (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                        acc += pr[t] * g4_to_float(vr[d]);
                    }
                }
                attn_out_f32[static_cast<size_t>(q_h) * head_dim + d] = acc;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A7: work-stealing o_proj.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A7 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;
                const T* wr = o_proj_w + static_cast<size_t>(row) * q_dim;
                float p = 0.0f;
                const int vd4 = q_dim & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(wr[c]);
                    float w1 = g4_to_float(wr[c+1]);
                    float w2 = g4_to_float(wr[c+2]);
                    float w3 = g4_to_float(wr[c+3]);
                    p += w0 * attn_out_f32[c]   + w1 * attn_out_f32[c+1]
                       + w2 * attn_out_f32[c+2] + w3 * attn_out_f32[c+3];
                }
                for (int c = vd4 + lane; c < q_dim; c += warpSize) {
                    p += g4_to_float(wr[c]) * attn_out_f32[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) oproj_f32[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A8: post_attn_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(oproj_normed, oproj_f32, post_attn_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A9: residual add + BF16 store → hidden_io (= h_mid).
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            hidden_io[c] = g4_from_float<T>(hidden_f32[c] + oproj_normed[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // =========================================================
        // Phase B — MLP + PLE. Mirrors g4_fused_mlp_ple_kernel.
        // Workspace (from offset 0):
        //   hidden_f32 [hidden]
        //   normed_ff  [hidden]
        //   gate_up    [2*intermediate]
        //   gated_proj [intermediate]
        //   down_out   [hidden]
        //   normed_down[hidden]
        //   h_pre_ple  [hidden]
        //   gated_ple  [ple_hidden]
        //   gated_act  [ple_hidden]
        //   proj_ple   [hidden]
        //   normed_ple [hidden]
        // =========================================================
        const int intermediate_size = L.intermediate_size;
        const float layer_scalar = L.layer_scalar;
        const T* pre_ff_norm_w = static_cast<const T*>(L.pre_ff_norm_w);
        const T* gate_proj_w = static_cast<const T*>(L.gate_proj_w);
        const T* up_proj_w = static_cast<const T*>(L.up_proj_w);
        const T* down_proj_w = static_cast<const T*>(L.down_proj_w);
        const T* post_ff_norm_w = static_cast<const T*>(L.post_ff_norm_w);
        const T* pli_gate_w =
            static_cast<const T*>(L.per_layer_input_gate_w);
        const T* pli_proj_w =
            static_cast<const T*>(L.per_layer_projection_w);
        const T* pli_post_norm_w =
            static_cast<const T*>(L.post_per_layer_input_norm_w);
        const T* per_layer_input =
            per_layer_inputs + static_cast<size_t>(layer) * ple_hidden;

        float* b_hidden_f32   = workspace;
        float* b_normed_ff    = b_hidden_f32 + hidden_size;
        float* b_gate_up      = b_normed_ff + hidden_size;
        float* b_gated_proj   = b_gate_up + 2 * intermediate_size;
        float* b_down_out     = b_gated_proj + intermediate_size;
        float* b_normed_down  = b_down_out + hidden_size;
        float* b_h_pre_ple    = b_normed_down + hidden_size;
        float* b_gated_ple    = b_h_pre_ple + hidden_size;
        float* b_gated_act    = b_gated_ple + ple_hidden;
        float* b_projected    = b_gated_act + ple_hidden;
        float* b_normed_ple   = b_projected + hidden_size;

        // B0: load h_mid BF16 → b_hidden_f32.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            b_hidden_f32[c] = g4_to_float(hidden_io[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B1: pre_ff_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_ff, b_hidden_f32, pre_ff_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2: work-stealing gate+up matmul.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2 wave-per-row.
        {
            const int lane = tid % warpSize;
            const int total_mlp1 = 2 * intermediate_size;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(total_mlp1)) break;

                const T* w = (row < static_cast<unsigned int>(intermediate_size))
                    ? gate_proj_w : up_proj_w;
                const int weight_row =
                    (row < static_cast<unsigned int>(intermediate_size))
                        ? static_cast<int>(row)
                        : static_cast<int>(row - intermediate_size);
                const T* wr = w + static_cast<size_t>(weight_row) * hidden_size;

                float p = 0.0f;
                const int vd4 = hidden_size & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(wr[c]);
                    float w1 = g4_to_float(wr[c+1]);
                    float w2 = g4_to_float(wr[c+2]);
                    float w3 = g4_to_float(wr[c+3]);
                    p += w0 * b_normed_ff[c]   + w1 * b_normed_ff[c+1]
                       + w2 * b_normed_ff[c+2] + w3 * b_normed_ff[c+3];
                }
                for (int c = vd4 + lane; c < hidden_size; c += warpSize) {
                    p += g4_to_float(wr[c]) * b_normed_ff[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_gate_up[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B3: gated_proj[i] = gelu(gate[i]) * up[i].
        {
            const float* gate_slot = b_gate_up;
            const float* up_slot = b_gate_up + intermediate_size;
            for (int i = tid + blockIdx.x * bs;
                 i < intermediate_size; i += bs * nb) {
                b_gated_proj[i] = g4_gelu_tanh_scalar(gate_slot[i]) * up_slot[i];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B4: work-stealing down_proj.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B4 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;

                const T* wr = down_proj_w +
                    static_cast<size_t>(row) * intermediate_size;
                float p = 0.0f;
                const int vd4 = intermediate_size & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(wr[c]);
                    float w1 = g4_to_float(wr[c+1]);
                    float w2 = g4_to_float(wr[c+2]);
                    float w3 = g4_to_float(wr[c+3]);
                    p += w0 * b_gated_proj[c]   + w1 * b_gated_proj[c+1]
                       + w2 * b_gated_proj[c+2] + w3 * b_gated_proj[c+3];
                }
                for (int c = vd4 + lane; c < intermediate_size; c += warpSize) {
                    p += g4_to_float(wr[c]) * b_gated_proj[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_down_out[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B5: post_ff_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_down, b_down_out, post_ff_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B6: h_pre_ple = b_hidden_f32 + b_normed_down.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            b_h_pre_ple[c] = b_hidden_f32[c] + b_normed_down[c];
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7: work-stealing per_layer_input_gate matvec.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(ple_hidden)) break;

                const T* wr = pli_gate_w +
                    static_cast<size_t>(row) * hidden_size;
                float p = 0.0f;
                const int vd4 = hidden_size & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(wr[c]);
                    float w1 = g4_to_float(wr[c+1]);
                    float w2 = g4_to_float(wr[c+2]);
                    float w3 = g4_to_float(wr[c+3]);
                    p += w0 * b_h_pre_ple[c]   + w1 * b_h_pre_ple[c+1]
                       + w2 * b_h_pre_ple[c+2] + w3 * b_h_pre_ple[c+3];
                }
                for (int c = vd4 + lane; c < hidden_size; c += warpSize) {
                    p += g4_to_float(wr[c]) * b_h_pre_ple[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_gated_ple[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B8: gated_act = gelu(gated_ple) * per_layer_input.
        for (int i = tid + blockIdx.x * bs; i < ple_hidden; i += bs * nb) {
            const float g = g4_gelu_tanh_scalar(b_gated_ple[i]);
            b_gated_act[i] = g * g4_to_float(per_layer_input[i]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B9: work-stealing per_layer_projection matvec.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B9 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;

                const T* wr = pli_proj_w +
                    static_cast<size_t>(row) * ple_hidden;
                float p = 0.0f;
                const int vd4 = ple_hidden & ~3;
                for (int c = lane * 4; c < vd4; c += warpSize * 4) {
                    float w0 = g4_to_float(wr[c]);
                    float w1 = g4_to_float(wr[c+1]);
                    float w2 = g4_to_float(wr[c+2]);
                    float w3 = g4_to_float(wr[c+3]);
                    p += w0 * b_gated_act[c]   + w1 * b_gated_act[c+1]
                       + w2 * b_gated_act[c+2] + w3 * b_gated_act[c+3];
                }
                for (int c = vd4 + lane; c < ple_hidden; c += warpSize) {
                    p += g4_to_float(wr[c]) * b_gated_act[c];
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_projected[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B10: post_per_layer_input_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_ple, b_projected,
                                      pli_post_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B11: hidden_io = bf16((h_pre_ple + normed_ple) * layer_scalar).
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            const float v = (b_h_pre_ple[c] + b_normed_ple[c]) * layer_scalar;
            hidden_io[c] = g4_from_float<T>(v);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);
    }
}

// =============================================================================
// Persistent decode megakernel — batched variant (Phase 2).
//
// Structural clone of `g4_persistent_decode_kernel` that runs `batch_size`
// decode tokens through all layers in a single kernel launch. Weight reads in
// the six matvec phases (A2 QKV, A7 o_proj, B2 gate+up, B4 down,
// B7 per_layer_input_gate, B9 per_layer_projection) are amortized across
// sequences via per-sequence scalar accumulators — each weight row is loaded
// once and dot-producted against `batch_size` activation vectors.
//
// Non-matmul phases (norms, RoPE, attention scores/softmax/value aggregation,
// KV append, gelu*up, gelu*ple, residuals, layer_scalar multiply) iterate
// over sequences serially — they're not bandwidth-bound on weights and this
// keeps the control flow close to the validated single-seq body.
//
// Per-sequence state (`seqlen_offset`, `kv_cache_k/v`, `kv_max_t`) lives in
// the `batch_descs[num_layers]` array, indexed by `[layer]`. Shared-KV layers
// must have their per-sequence `kv_cache_k/v` pointers aliased to the source
// layer's per-sequence pointers — the kernel does not replicate.
//
// Workspace layout: each sequence gets its own `ws_stride`-element slice of
// `workspace`. Within a slice the layout matches the single-seq kernel exactly
// (Phase A and Phase B share the slice from offset 0). The caller sizes
// `workspace` as `batch_size * persistent_decode_workspace_elems(...)` floats.
//
// `hidden_io` is `[batch_size, hidden_size]` contiguous; `per_layer_inputs` is
// `[batch_size, num_layers, ple_hidden]` contiguous.
// =============================================================================
template <typename T>
__global__ void g4_persistent_decode_batch_kernel(
    int num_layers,
    int hidden_size,
    int ple_hidden,
    float eps,
    float scale,
    int batch_size,
    int ws_stride,
    const Gemma4DecodeLayerDesc* __restrict__ layers,
    const Gemma4BatchSeqDesc* __restrict__ batch_descs,  // [num_layers]
    T* __restrict__ hidden_io,                           // [B, hidden_size] BF16
    const T* __restrict__ per_layer_inputs,              // [B, num_layers, ple_hidden] BF16
    float* __restrict__ workspace,                       // [B * ws_stride]
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;
    const int B = batch_size;

    extern __shared__ float lds[];

    for (int layer = 0; layer < num_layers; ++layer) {
        const Gemma4DecodeLayerDesc L = layers[layer];
        const Gemma4BatchSeqDesc   S = batch_descs[layer];
        const int num_q_heads = L.num_q_heads;
        const int num_kv_heads = L.num_kv_heads;
        const int head_dim = L.head_dim;
        const int rotary_dim = L.rotary_dim;
        const int sliding_window = L.sliding_window;
        const int shared_kv = L.shared_kv;
        const int q_dim = num_q_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;

        const T* input_norm_w = static_cast<const T*>(L.input_norm_w);
        const T* q_proj_w = static_cast<const T*>(L.q_proj_w);
        const T* k_proj_w = static_cast<const T*>(L.k_proj_w);
        const T* v_proj_w = static_cast<const T*>(L.v_proj_w);
        const T* q_norm_w = static_cast<const T*>(L.q_norm_w);
        const T* k_norm_w = static_cast<const T*>(L.k_norm_w);
        const T* o_proj_w = static_cast<const T*>(L.o_proj_w);
        const T* post_attn_norm_w = static_cast<const T*>(L.post_attn_norm_w);
        const T* cos_table = static_cast<const T*>(L.cos_table);
        const T* sin_table = static_cast<const T*>(L.sin_table);

        // =========================================================
        // Phase A — attention block (per-seq loops).
        // =========================================================

        // A0: load hidden_io BF16 → hidden_f32 for each sequence.
        for (int b = 0; b < B; ++b) {
            float* hidden_f32 = workspace + b * ws_stride;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                hidden_f32[c] = g4_to_float(hb[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A1: input_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* hidden_f32 = workspace + b * ws_stride;
                float* normed_f32 = hidden_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A2: work-stealing QKV matmul — amortize weight load across B seqs.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        {
            const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
            while (true) {
                __shared__ unsigned int claimed;
                if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
                __syncthreads();
                const unsigned int row = claimed;
                if (row >= static_cast<unsigned int>(total_qkv)) break;

                const T* w;
                int weight_row;
                if (row < static_cast<unsigned int>(q_dim)) {
                    w = q_proj_w;
                    weight_row = static_cast<int>(row);
                } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
                    w = k_proj_w;
                    weight_row = static_cast<int>(row - q_dim);
                } else {
                    w = v_proj_w;
                    weight_row = static_cast<int>(row - q_dim - kv_dim);
                }
                const T* w_row =
                    w + static_cast<size_t>(weight_row) * hidden_size;

                float pacc[G4_MAX_BATCH_SIZE];
                #pragma unroll
                for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

                for (int c = tid; c < hidden_size; c += bs) {
                    const float wv = g4_to_float(w_row[c]);
                    for (int b = 0; b < B; ++b) {
                        const float* normed_f32 = workspace + b * ws_stride + hidden_size;
                        pacc[b] += wv * normed_f32[c];
                    }
                }

                for (int b = 0; b < B; ++b) {
                    lds[tid] = pacc[b];
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
                        proj_f32[row] = lds[0];
                    }
                    __syncthreads();
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A3: q/k/v norms + RoPE + KV append (block 0, per-seq loop).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
                float* q_f32 = proj_f32;
                float* k_f32 = proj_f32 + q_dim;
                float* v_f32 = proj_f32 + q_dim + kv_dim;
                const int position_b = S.seqlen_offset[b];
                const int max_T_b = S.kv_max_t[b];
                T* k_cache_b = static_cast<T*>(S.kv_cache_k[b]);
                T* v_cache_b = static_cast<T*>(S.kv_cache_v[b]);
                if (!shared_kv) {
                    g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                        num_q_heads, head_dim, eps, lds);
                    g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
                        num_kv_heads, head_dim, eps, lds);
                    g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
                        num_kv_heads, head_dim, eps, lds);
                    g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                        num_q_heads, head_dim, rotary_dim, position_b);
                    g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
                        num_kv_heads, head_dim, rotary_dim, position_b);
                    g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache_b, v_cache_b,
                        num_kv_heads, head_dim, position_b, max_T_b);
                } else {
                    g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                        num_q_heads, head_dim, eps, lds);
                    g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                        num_q_heads, head_dim, rotary_dim, position_b);
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A4: attention scores (per-seq loop).
        const int num_q_per_kv = num_q_heads / num_kv_heads;
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            const T* k_cache_b = static_cast<const T*>(S.kv_cache_k[b]);
            const float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
            const float* q_f32 = proj_f32;
            float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                + q_dim + 2 * kv_dim;
            const int kv_len = position_b + 1;
            const int last = kv_len - 1;
            const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
            const int total_items = num_q_heads * kv_len;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int t = item % kv_len;
                const int q_h = item / kv_len;
                float val;
                if (t < min_t) {
                    val = -INFINITY;
                } else {
                    const int kv_h = q_h / num_q_per_kv;
                    const float* qr = q_f32 +
                        static_cast<size_t>(q_h) * head_dim;
                    const T* kr = k_cache_b +
                        (static_cast<size_t>(kv_h) * max_T_b + t) * head_dim;
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += qr[d] * g4_to_float(kr[d]);
                    }
                    val = acc * scale;
                }
                scores_f32[static_cast<size_t>(q_h) * max_T_b + t] = val;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A5: softmax per q head (per-seq loop).
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                + q_dim + 2 * kv_dim;
            const int kv_len = position_b + 1;
            for (int q_h = blockIdx.x; q_h < num_q_heads; q_h += nb) {
                float* row = scores_f32 + static_cast<size_t>(q_h) * max_T_b;
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
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A6: value aggregation (per-seq loop).
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            const T* v_cache_b = static_cast<const T*>(S.kv_cache_v[b]);
            const float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                      + q_dim + 2 * kv_dim;
            float* attn_out_f32 = workspace + b * ws_stride + 2 * hidden_size
                                  + q_dim + 2 * kv_dim + num_q_heads * max_T_b;
            const int kv_len = position_b + 1;
            const int total_items = num_q_heads * head_dim;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int d = item % head_dim;
                const int q_h = item / head_dim;
                const int kv_h = q_h / num_q_per_kv;
                const float* pr = scores_f32 +
                    static_cast<size_t>(q_h) * max_T_b;
                float acc = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const T* vr = v_cache_b +
                        (static_cast<size_t>(kv_h) * max_T_b + t) * head_dim;
                    acc += pr[t] * g4_to_float(vr[d]);
                }
                attn_out_f32[static_cast<size_t>(q_h) * head_dim + d] = acc;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A7: work-stealing o_proj — amortize weight load across B seqs.
        // `scores_f32` has stride `num_q_heads * max_T_b` per seq. All seqs in
        // this engine share the same allocated `max_t` (L.kv_max_t), so use it
        // as the uniform stride when computing offsets past scores_f32.
        const int attn_out_base_off = 2 * hidden_size + q_dim + 2 * kv_dim
                                    + num_q_heads * L.kv_max_t;
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

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c = tid; c < q_dim; c += bs) {
                const float wv = g4_to_float(wr[c]);
                for (int b = 0; b < B; ++b) {
                    const float* attn_out_f32 = workspace + b * ws_stride
                                                + attn_out_base_off;
                    pacc[b] += wv * attn_out_f32[c];
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* oproj_f32 = workspace + b * ws_stride
                                       + attn_out_base_off + q_dim;
                    oproj_f32[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A8: post_attn_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* oproj_f32 = workspace + b * ws_stride
                                   + attn_out_base_off + q_dim;
                float* oproj_normed = oproj_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(oproj_normed, oproj_f32, post_attn_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A9: residual add + BF16 store → hidden_io = h_mid, per-seq.
        for (int b = 0; b < B; ++b) {
            const float* hidden_f32 = workspace + b * ws_stride;
            const float* oproj_normed = workspace + b * ws_stride
                                        + attn_out_base_off + q_dim + hidden_size;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                hb[c] = g4_from_float<T>(hidden_f32[c] + oproj_normed[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // =========================================================
        // Phase B — MLP + PLE (per-seq loops).
        // =========================================================
        const int intermediate_size = L.intermediate_size;
        const float layer_scalar = L.layer_scalar;
        const T* pre_ff_norm_w = static_cast<const T*>(L.pre_ff_norm_w);
        const T* gate_proj_w = static_cast<const T*>(L.gate_proj_w);
        const T* up_proj_w = static_cast<const T*>(L.up_proj_w);
        const T* down_proj_w = static_cast<const T*>(L.down_proj_w);
        const T* post_ff_norm_w = static_cast<const T*>(L.post_ff_norm_w);
        const T* pli_gate_w =
            static_cast<const T*>(L.per_layer_input_gate_w);
        const T* pli_proj_w =
            static_cast<const T*>(L.per_layer_projection_w);
        const T* pli_post_norm_w =
            static_cast<const T*>(L.post_per_layer_input_norm_w);

        // Per-seq Phase-B workspace layout (slice base = workspace + b*ws_stride):
        //   [0]                               b_hidden_f32  [hidden]
        //   [hidden]                          b_normed_ff   [hidden]
        //   [2*hidden]                        b_gate_up     [2*intermediate]
        //   [2*hidden + 2*inter]              b_gated_proj  [intermediate]
        //   [2*hidden + 3*inter]              b_down_out    [hidden]
        //   [3*hidden + 3*inter]              b_normed_down [hidden]
        //   [4*hidden + 3*inter]              b_h_pre_ple   [hidden]
        //   [5*hidden + 3*inter]              b_gated_ple   [ple_hidden]
        //   [5*hidden + 3*inter + ple_h]      b_gated_act   [ple_hidden]
        //   [5*hidden + 3*inter + 2*ple_h]    b_projected   [hidden]
        //   [6*hidden + 3*inter + 2*ple_h]    b_normed_ple  [hidden]

        // B0: load h_mid BF16 → b_hidden_f32 (per-seq).
        for (int b = 0; b < B; ++b) {
            float* b_hidden_f32 = workspace + b * ws_stride;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                b_hidden_f32[c] = g4_to_float(hb[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B1: pre_ff_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_hidden_f32 = workspace + b * ws_stride;
                float* b_normed_ff = b_hidden_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(b_normed_ff, b_hidden_f32, pre_ff_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2: work-stealing gate+up matmul — amortized.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        {
            const int total_mlp1 = 2 * intermediate_size;
            while (true) {
                __shared__ unsigned int claimed;
                if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
                __syncthreads();
                const unsigned int row = claimed;
                if (row >= static_cast<unsigned int>(total_mlp1)) break;

                const T* w = (row < static_cast<unsigned int>(intermediate_size))
                    ? gate_proj_w : up_proj_w;
                const int weight_row =
                    (row < static_cast<unsigned int>(intermediate_size))
                        ? static_cast<int>(row)
                        : static_cast<int>(row - intermediate_size);
                const T* wr = w + static_cast<size_t>(weight_row) * hidden_size;

                float pacc[G4_MAX_BATCH_SIZE];
                #pragma unroll
                for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

                for (int c = tid; c < hidden_size; c += bs) {
                    const float wv = g4_to_float(wr[c]);
                    for (int b = 0; b < B; ++b) {
                        const float* b_normed_ff = workspace + b * ws_stride + hidden_size;
                        pacc[b] += wv * b_normed_ff[c];
                    }
                }

                for (int b = 0; b < B; ++b) {
                    lds[tid] = pacc[b];
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float* b_gate_up = workspace + b * ws_stride + 2 * hidden_size;
                        b_gate_up[row] = lds[0];
                    }
                    __syncthreads();
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B3: gated_proj[i] = gelu(gate[i]) * up[i], per-seq.
        for (int b = 0; b < B; ++b) {
            const float* b_gate_up = workspace + b * ws_stride + 2 * hidden_size;
            const float* gate_slot = b_gate_up;
            const float* up_slot = b_gate_up + intermediate_size;
            float* b_gated_proj = workspace + b * ws_stride + 2 * hidden_size
                                  + 2 * intermediate_size;
            for (int i = tid + blockIdx.x * bs;
                 i < intermediate_size; i += bs * nb) {
                b_gated_proj[i] = g4_gelu_tanh_scalar(gate_slot[i]) * up_slot[i];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B4: work-stealing down_proj — amortized.
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

            const T* wr = down_proj_w +
                static_cast<size_t>(row) * intermediate_size;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c = tid; c < intermediate_size; c += bs) {
                const float wv = g4_to_float(wr[c]);
                for (int b = 0; b < B; ++b) {
                    const float* b_gated_proj = workspace + b * ws_stride
                                                + 2 * hidden_size + 2 * intermediate_size;
                    pacc[b] += wv * b_gated_proj[c];
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_down_out = workspace + b * ws_stride
                                        + 2 * hidden_size + 3 * intermediate_size;
                    b_down_out[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B5: post_ff_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_down_out = workspace + b * ws_stride
                                    + 2 * hidden_size + 3 * intermediate_size;
                float* b_normed_down = b_down_out + hidden_size;
                g4_block_rms_norm_f32<T>(b_normed_down, b_down_out, post_ff_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B6: h_pre_ple = b_hidden_f32 + b_normed_down, per-seq.
        for (int b = 0; b < B; ++b) {
            const float* b_hidden_f32 = workspace + b * ws_stride;
            const float* b_normed_down = workspace + b * ws_stride
                                         + 3 * hidden_size + 3 * intermediate_size;
            float* b_h_pre_ple = workspace + b * ws_stride
                                 + 4 * hidden_size + 3 * intermediate_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                b_h_pre_ple[c] = b_hidden_f32[c] + b_normed_down[c];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7: work-stealing per_layer_input_gate matvec — amortized.
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

            const T* wr = pli_gate_w +
                static_cast<size_t>(row) * hidden_size;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c = tid; c < hidden_size; c += bs) {
                const float wv = g4_to_float(wr[c]);
                for (int b = 0; b < B; ++b) {
                    const float* b_h_pre_ple = workspace + b * ws_stride
                                               + 4 * hidden_size + 3 * intermediate_size;
                    pacc[b] += wv * b_h_pre_ple[c];
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_gated_ple = workspace + b * ws_stride
                                         + 5 * hidden_size + 3 * intermediate_size;
                    b_gated_ple[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B8: gated_act = gelu(gated_ple) * per_layer_input, per-seq.
        for (int b = 0; b < B; ++b) {
            const float* b_gated_ple = workspace + b * ws_stride
                                       + 5 * hidden_size + 3 * intermediate_size;
            float* b_gated_act = workspace + b * ws_stride
                                 + 5 * hidden_size + 3 * intermediate_size + ple_hidden;
            const T* per_layer_input = per_layer_inputs
                + (static_cast<size_t>(b) * num_layers + layer) * ple_hidden;
            for (int i = tid + blockIdx.x * bs; i < ple_hidden; i += bs * nb) {
                const float g = g4_gelu_tanh_scalar(b_gated_ple[i]);
                b_gated_act[i] = g * g4_to_float(per_layer_input[i]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B9: work-stealing per_layer_projection matvec — amortized.
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

            const T* wr = pli_proj_w +
                static_cast<size_t>(row) * ple_hidden;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c = tid; c < ple_hidden; c += bs) {
                const float wv = g4_to_float(wr[c]);
                for (int b = 0; b < B; ++b) {
                    const float* b_gated_act = workspace + b * ws_stride
                                               + 5 * hidden_size + 3 * intermediate_size
                                               + ple_hidden;
                    pacc[b] += wv * b_gated_act[c];
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_projected = workspace + b * ws_stride
                                         + 5 * hidden_size + 3 * intermediate_size
                                         + 2 * ple_hidden;
                    b_projected[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B10: post_per_layer_input_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_projected = workspace + b * ws_stride
                                     + 5 * hidden_size + 3 * intermediate_size
                                     + 2 * ple_hidden;
                float* b_normed_ple = workspace + b * ws_stride
                                      + 6 * hidden_size + 3 * intermediate_size
                                      + 2 * ple_hidden;
                g4_block_rms_norm_f32<T>(b_normed_ple, b_projected,
                                          pli_post_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B11: hidden_io = bf16((h_pre_ple + normed_ple) * layer_scalar), per-seq.
        for (int b = 0; b < B; ++b) {
            const float* b_h_pre_ple = workspace + b * ws_stride
                                       + 4 * hidden_size + 3 * intermediate_size;
            const float* b_normed_ple = workspace + b * ws_stride
                                        + 6 * hidden_size + 3 * intermediate_size
                                        + 2 * ple_hidden;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                const float v = (b_h_pre_ple[c] + b_normed_ple[c]) * layer_scalar;
                hb[c] = g4_from_float<T>(v);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);
    }
}

// =============================================================================
// Persistent decode megakernel — INT4 variant (Step 31).
//
// Structural clone of `g4_persistent_decode_kernel` above. Six matmul phases
// (A2 QKV, A7 o_proj, B2 gate+up, B4 down, B7 per_layer_input_gate,
// B9 per_layer_projection) retarget their inner reduction loops to the
// 8-wide INT4 dequant path via `g4_int4_dequant_8`. All other phases
// (norms, RoPE, kv_append, attention scores/softmax/value, gelu*up, gelu*ple,
// residuals, layer_scalar multiply) are byte-for-byte copies of the BF16
// persistent kernel — intentional duplication per `feedback_hipcc_codegen`.
//
// Weight access pattern: the main `Gemma4DecodeLayerDesc` array holds packed
// INT4 weights (reinterpret `const void*` → `const uint8_t*` at the kernel
// site); the parallel `Gemma4Int4ScaleDesc` array holds matching BF16 scale
// and zero tables. `group_size` lives in the scale desc (not per-layer — one
// group_size for the whole bake; bake format fixes it at 128).
//
// Workspace layout matches the BF16 kernel exactly; same
// `persistent_decode_workspace_elems` helper sizes the scratch. Shared-KV
// layers alias their source's K/V cache pointers in the main desc array so
// the megakernel sees a single coherent memory region for the source-
// to-shared dependency, identical to the BF16 path.
// =============================================================================

template <typename T>
__global__ void g4_persistent_decode_int4_kernel(
    int num_layers,
    int hidden_size,
    int ple_hidden,
    int position,
    float eps,
    float scale,
    const Gemma4DecodeLayerDesc* __restrict__ layers,
    const Gemma4Int4ScaleDesc*   __restrict__ int4_scales,
    T* __restrict__ hidden_io,
    const T* __restrict__ per_layer_inputs,
    float* __restrict__ workspace,
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;

    extern __shared__ float lds[];

    for (int layer = 0; layer < num_layers; ++layer) {
        const Gemma4DecodeLayerDesc L = layers[layer];
        const Gemma4Int4ScaleDesc   S = int4_scales[layer];
        const int num_q_heads = L.num_q_heads;
        const int num_kv_heads = L.num_kv_heads;
        const int head_dim = L.head_dim;
        const int rotary_dim = L.rotary_dim;
        const int sliding_window = L.sliding_window;
        const int shared_kv = L.shared_kv;
        const int max_T = L.kv_max_t;
        const int q_dim = num_q_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const int gsz = S.group_size;

        // BF16 tensors (norms, RoPE, KV cache, per_layer_input).
        const T* input_norm_w = static_cast<const T*>(L.input_norm_w);
        const T* q_norm_w = static_cast<const T*>(L.q_norm_w);
        const T* k_norm_w = static_cast<const T*>(L.k_norm_w);
        const T* post_attn_norm_w = static_cast<const T*>(L.post_attn_norm_w);
        const T* cos_table = static_cast<const T*>(L.cos_table);
        const T* sin_table = static_cast<const T*>(L.sin_table);
        T* k_cache = static_cast<T*>(L.kv_cache_k);
        T* v_cache = static_cast<T*>(L.kv_cache_v);

        // INT4 packed weights (reinterpret).
        const uint8_t*      q_proj_packed = static_cast<const uint8_t*>(L.q_proj_w);
        const uint8_t*      k_proj_packed = static_cast<const uint8_t*>(L.k_proj_w);
        const uint8_t*      v_proj_packed = static_cast<const uint8_t*>(L.v_proj_w);
        const uint8_t*      o_proj_packed = static_cast<const uint8_t*>(L.o_proj_w);
        const hip_bfloat16* q_proj_scale  = static_cast<const hip_bfloat16*>(S.q_proj_scale);
        const hip_bfloat16* q_proj_zero   = static_cast<const hip_bfloat16*>(S.q_proj_zero);
        const hip_bfloat16* k_proj_scale  = static_cast<const hip_bfloat16*>(S.k_proj_scale);
        const hip_bfloat16* k_proj_zero   = static_cast<const hip_bfloat16*>(S.k_proj_zero);
        const hip_bfloat16* v_proj_scale  = static_cast<const hip_bfloat16*>(S.v_proj_scale);
        const hip_bfloat16* v_proj_zero   = static_cast<const hip_bfloat16*>(S.v_proj_zero);
        const hip_bfloat16* o_proj_scale  = static_cast<const hip_bfloat16*>(S.o_proj_scale);
        const hip_bfloat16* o_proj_zero   = static_cast<const hip_bfloat16*>(S.o_proj_zero);

        // =========================================================
        // Phase A — attention block (INT4 QKV + INT4 o_proj).
        // =========================================================
        float* hidden_f32   = workspace;
        float* normed_f32   = hidden_f32 + hidden_size;
        float* proj_f32     = normed_f32 + hidden_size;
        float* scores_f32   = proj_f32 + (q_dim + 2 * kv_dim);
        float* attn_out_f32 = scores_f32 + num_q_heads * max_T;
        float* oproj_f32    = attn_out_f32 + q_dim;
        float* oproj_normed = oproj_f32 + hidden_size;

        // A0: load hidden_io BF16 → hidden_f32.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            hidden_f32[c] = g4_to_float(hidden_io[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A1: input_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A2: work-stealing Q/K/V INT4 matmul.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_in = (hidden_size + gsz - 1) / gsz;
        const int byte_cols_in  = hidden_size / 2;

        // A2 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(total_qkv)) break;

                const uint8_t*      w_packed_base;
                const hip_bfloat16* w_scale_base;
                const hip_bfloat16* w_zero_base;
                int weight_row;
                if (row < static_cast<unsigned int>(q_dim)) {
                    w_packed_base = q_proj_packed;
                    w_scale_base  = q_proj_scale;
                    w_zero_base   = q_proj_zero;
                    weight_row    = static_cast<int>(row);
                } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
                    w_packed_base = k_proj_packed;
                    w_scale_base  = k_proj_scale;
                    w_zero_base   = k_proj_zero;
                    weight_row    = static_cast<int>(row - q_dim);
                } else {
                    w_packed_base = v_proj_packed;
                    w_scale_base  = v_proj_scale;
                    w_zero_base   = v_proj_zero;
                    weight_row    = static_cast<int>(row - q_dim - kv_dim);
                }
                const int scale_row = weight_row / gsz;
                const uint8_t* w_row_bytes =
                    w_packed_base + static_cast<size_t>(weight_row) * byte_cols_in;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < hidden_size; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                                      scale_row, c0, scale_cols_in, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * normed_f32[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) proj_f32[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        float* q_f32 = proj_f32;
        float* k_f32 = proj_f32 + q_dim;
        float* v_f32 = proj_f32 + q_dim + kv_dim;

        // A3: q/k/v norms + RoPE + KV append (block 0 only).
        if (!shared_kv && blockIdx.x == 0) {
            g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                num_q_heads, head_dim, eps, lds);
            g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
                num_kv_heads, head_dim, eps, lds);
            g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
                num_kv_heads, head_dim, eps, lds);
            g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                num_q_heads, head_dim, rotary_dim, position);
            g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
                num_kv_heads, head_dim, rotary_dim, position);
            g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache, v_cache,
                num_kv_heads, head_dim, position, max_T);
        } else if (shared_kv && blockIdx.x == 0) {
            g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                num_q_heads, head_dim, eps, lds);
            g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                num_q_heads, head_dim, rotary_dim, position);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A4: attention scores.
        const int kv_len = position + 1;
        const int last = kv_len - 1;
        const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
        const int num_q_per_kv = num_q_heads / num_kv_heads;
        {
            const int total_items = num_q_heads * kv_len;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int t = item % kv_len;
                const int q_h = item / kv_len;
                float val;
                if (t < min_t) {
                    val = -INFINITY;
                } else {
                    const int kv_h = q_h / num_q_per_kv;
                    const float* qr = q_f32 +
                        static_cast<size_t>(q_h) * head_dim;
                    const T* kr = k_cache +
                        (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
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

        // A5: softmax per q head.
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

        // A6: value aggregation.
        {
            const int total_items = num_q_heads * head_dim;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int d = item % head_dim;
                const int q_h = item / head_dim;
                const int kv_h = q_h / num_q_per_kv;
                const float* pr = scores_f32 +
                    static_cast<size_t>(q_h) * max_T;
                float acc = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const T* vr = v_cache +
                        (static_cast<size_t>(kv_h) * max_T + t) * head_dim;
                    acc += pr[t] * g4_to_float(vr[d]);
                }
                attn_out_f32[static_cast<size_t>(q_h) * head_dim + d] = acc;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A7: work-stealing o_proj INT4 matmul. in_dim=q_dim, out=hidden_size.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_o = (q_dim + gsz - 1) / gsz;
        const int byte_cols_o  = q_dim / 2;

        // A7 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;

                const int scale_row = static_cast<int>(row) / gsz;
                const uint8_t* w_row_bytes =
                    o_proj_packed + static_cast<size_t>(row) * byte_cols_o;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < q_dim; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, o_proj_scale, o_proj_zero,
                                      scale_row, c0, scale_cols_o, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * attn_out_f32[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) oproj_f32[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A8: post_attn_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(oproj_normed, oproj_f32, post_attn_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A9: residual add + BF16 store → hidden_io.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            hidden_io[c] = g4_from_float<T>(hidden_f32[c] + oproj_normed[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // =========================================================
        // Phase B — MLP + PLE (INT4 gate/up, down, ple_gate, ple_proj).
        // =========================================================
        const int intermediate_size = L.intermediate_size;
        const float layer_scalar = L.layer_scalar;
        const T* pre_ff_norm_w = static_cast<const T*>(L.pre_ff_norm_w);
        const T* post_ff_norm_w = static_cast<const T*>(L.post_ff_norm_w);
        const T* pli_post_norm_w =
            static_cast<const T*>(L.post_per_layer_input_norm_w);
        const T* per_layer_input =
            per_layer_inputs + static_cast<size_t>(layer) * ple_hidden;

        const uint8_t*      gate_proj_packed = static_cast<const uint8_t*>(L.gate_proj_w);
        const uint8_t*      up_proj_packed   = static_cast<const uint8_t*>(L.up_proj_w);
        const uint8_t*      down_proj_packed = static_cast<const uint8_t*>(L.down_proj_w);
        const uint8_t*      pli_gate_packed  = static_cast<const uint8_t*>(L.per_layer_input_gate_w);
        const uint8_t*      pli_proj_packed  = static_cast<const uint8_t*>(L.per_layer_projection_w);
        const hip_bfloat16* gate_proj_scale  = static_cast<const hip_bfloat16*>(S.gate_proj_scale);
        const hip_bfloat16* gate_proj_zero   = static_cast<const hip_bfloat16*>(S.gate_proj_zero);
        const hip_bfloat16* up_proj_scale    = static_cast<const hip_bfloat16*>(S.up_proj_scale);
        const hip_bfloat16* up_proj_zero     = static_cast<const hip_bfloat16*>(S.up_proj_zero);
        const hip_bfloat16* down_proj_scale  = static_cast<const hip_bfloat16*>(S.down_proj_scale);
        const hip_bfloat16* down_proj_zero   = static_cast<const hip_bfloat16*>(S.down_proj_zero);
        const hip_bfloat16* pli_gate_scale   = static_cast<const hip_bfloat16*>(S.per_layer_input_gate_scale);
        const hip_bfloat16* pli_gate_zero    = static_cast<const hip_bfloat16*>(S.per_layer_input_gate_zero);
        const hip_bfloat16* pli_proj_scale   = static_cast<const hip_bfloat16*>(S.per_layer_projection_scale);
        const hip_bfloat16* pli_proj_zero    = static_cast<const hip_bfloat16*>(S.per_layer_projection_zero);

        float* b_hidden_f32   = workspace;
        float* b_normed_ff    = b_hidden_f32 + hidden_size;
        float* b_gate_up      = b_normed_ff + hidden_size;
        float* b_gated_proj   = b_gate_up + 2 * intermediate_size;
        float* b_down_out     = b_gated_proj + intermediate_size;
        float* b_normed_down  = b_down_out + hidden_size;
        float* b_h_pre_ple    = b_normed_down + hidden_size;
        float* b_gated_ple    = b_h_pre_ple + hidden_size;
        float* b_gated_act    = b_gated_ple + ple_hidden;
        float* b_projected    = b_gated_act + ple_hidden;
        float* b_normed_ple   = b_projected + hidden_size;

        // B0: load h_mid BF16 → b_hidden_f32.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            b_hidden_f32[c] = g4_to_float(hidden_io[c]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B1: pre_ff_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_ff, b_hidden_f32, pre_ff_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2: work-stealing gate+up INT4 matmul.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            const int total_mlp1 = 2 * intermediate_size;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(total_mlp1)) break;

                const uint8_t*      w_packed_base;
                const hip_bfloat16* w_scale_base;
                const hip_bfloat16* w_zero_base;
                int weight_row;
                if (row < static_cast<unsigned int>(intermediate_size)) {
                    w_packed_base = gate_proj_packed;
                    w_scale_base  = gate_proj_scale;
                    w_zero_base   = gate_proj_zero;
                    weight_row    = static_cast<int>(row);
                } else {
                    w_packed_base = up_proj_packed;
                    w_scale_base  = up_proj_scale;
                    w_zero_base   = up_proj_zero;
                    weight_row    = static_cast<int>(row - intermediate_size);
                }
                const int scale_row = weight_row / gsz;
                const uint8_t* w_row_bytes =
                    w_packed_base + static_cast<size_t>(weight_row) * byte_cols_in;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < hidden_size; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                                      scale_row, c0, scale_cols_in, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * b_normed_ff[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_gate_up[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B3: gated_proj[i] = gelu(gate[i]) * up[i].
        {
            const float* gate_slot = b_gate_up;
            const float* up_slot = b_gate_up + intermediate_size;
            for (int i = tid + blockIdx.x * bs;
                 i < intermediate_size; i += bs * nb) {
                b_gated_proj[i] = g4_gelu_tanh_scalar(gate_slot[i]) * up_slot[i];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B4: work-stealing down_proj INT4 matmul. in=intermediate, out=hidden.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_i = (intermediate_size + gsz - 1) / gsz;
        const int byte_cols_i  = intermediate_size / 2;

        // B4 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;

                const int scale_row = static_cast<int>(row) / gsz;
                const uint8_t* w_row_bytes =
                    down_proj_packed + static_cast<size_t>(row) * byte_cols_i;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < intermediate_size; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, down_proj_scale, down_proj_zero,
                                      scale_row, c0, scale_cols_i, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * b_gated_proj[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_down_out[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B5: post_ff_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_down, b_down_out, post_ff_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B6: h_pre_ple = b_hidden_f32 + b_normed_down.
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            b_h_pre_ple[c] = b_hidden_f32[c] + b_normed_down[c];
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7: work-stealing per_layer_input_gate INT4 matmul.
        //   in_dim = hidden_size, out = ple_hidden.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(ple_hidden)) break;

                const int scale_row = static_cast<int>(row) / gsz;
                const uint8_t* w_row_bytes =
                    pli_gate_packed + static_cast<size_t>(row) * byte_cols_in;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < hidden_size; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, pli_gate_scale, pli_gate_zero,
                                      scale_row, c0, scale_cols_in, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * b_h_pre_ple[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_gated_ple[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B8: gated_act = gelu(gated_ple) * per_layer_input.
        for (int i = tid + blockIdx.x * bs; i < ple_hidden; i += bs * nb) {
            const float g = g4_gelu_tanh_scalar(b_gated_ple[i]);
            b_gated_act[i] = g * g4_to_float(per_layer_input[i]);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B9: work-stealing per_layer_projection INT4 matmul.
        //   in_dim = ple_hidden, out = hidden_size.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_ple = (ple_hidden + gsz - 1) / gsz;
        const int byte_cols_ple  = ple_hidden / 2;

        // B9 INT4 wave-per-row.
        {
            const int lane = tid % warpSize;
            while (true) {
                unsigned int row;
                if (lane == 0) row = atomicAdd(matvec_counter, 1u);
                row = __shfl(row, 0);
                if (row >= static_cast<unsigned int>(hidden_size)) break;

                const int scale_row = static_cast<int>(row) / gsz;
                const uint8_t* w_row_bytes =
                    pli_proj_packed + static_cast<size_t>(row) * byte_cols_ple;

                float p = 0.0f;
                for (int c0 = lane * 8; c0 < ple_hidden; c0 += warpSize * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, pli_proj_scale, pli_proj_zero,
                                      scale_row, c0, scale_cols_ple, gsz, w);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        p += w[k] * b_gated_act[c0 + k];
                    }
                }
                float result = g4_wave_reduce_sum_f32(p);
                if (lane == 0) b_projected[row] = result;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B10: post_per_layer_input_norm on block 0.
        if (blockIdx.x == 0) {
            g4_block_rms_norm_f32<T>(b_normed_ple, b_projected,
                                      pli_post_norm_w,
                                      hidden_size, eps, lds);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B11: hidden_io = bf16((h_pre_ple + normed_ple) * layer_scalar).
        for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
            const float v = (b_h_pre_ple[c] + b_normed_ple[c]) * layer_scalar;
            hidden_io[c] = g4_from_float<T>(v);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);
    }
}

// =============================================================================
// Persistent decode megakernel — batched INT4 variant.
//
// Structural clone of `g4_persistent_decode_batch_kernel`. Six matmul phases
// (A2 QKV, A7 o_proj, B2 gate+up, B4 down, B7 per_layer_input_gate,
// B9 per_layer_projection) retarget their inner reduction loops to the 8-wide
// INT4 dequant path via `g4_int4_dequant_8`. Each work-stealing iteration
// holds `float pacc[G4_MAX_BATCH_SIZE]` — the dequant'd row is shared across
// all `B` sequences, so weight *and* dequant work are amortized. Non-matmul
// phases (norms, RoPE, attn scores/softmax/value, KV append, gelu gating,
// residuals, layer_scalar multiply) iterate sequences serially and are
// byte-for-byte copies of the BF16 batched kernel.
//
// Per-sequence state lives in `batch_descs[num_layers]` (same struct as the
// BF16 batched kernel). Workspace sliced per-seq with stride `ws_stride`.
// `hidden_io` is `[B, hidden_size]` contiguous; `per_layer_inputs` is
// `[B, num_layers, ple_hidden]` contiguous.
// =============================================================================
template <typename T>
__global__ void g4_persistent_decode_batch_int4_kernel(
    int num_layers,
    int hidden_size,
    int ple_hidden,
    float eps,
    float scale,
    int batch_size,
    int ws_stride,
    const Gemma4DecodeLayerDesc* __restrict__ layers,
    const Gemma4Int4ScaleDesc*   __restrict__ int4_scales,
    const Gemma4BatchSeqDesc*    __restrict__ batch_descs,  // [num_layers]
    T* __restrict__ hidden_io,                              // [B, hidden_size] BF16
    const T* __restrict__ per_layer_inputs,                  // [B, num_layers, ple_hidden] BF16
    float* __restrict__ workspace,                           // [B * ws_stride]
    unsigned int* __restrict__ matvec_counter,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;
    const int B = batch_size;

    extern __shared__ float lds[];

    for (int layer = 0; layer < num_layers; ++layer) {
        const Gemma4DecodeLayerDesc L = layers[layer];
        const Gemma4Int4ScaleDesc   SI = int4_scales[layer];
        const Gemma4BatchSeqDesc    S = batch_descs[layer];
        const int num_q_heads = L.num_q_heads;
        const int num_kv_heads = L.num_kv_heads;
        const int head_dim = L.head_dim;
        const int rotary_dim = L.rotary_dim;
        const int sliding_window = L.sliding_window;
        const int shared_kv = L.shared_kv;
        const int q_dim = num_q_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const int gsz = SI.group_size;

        const T* input_norm_w = static_cast<const T*>(L.input_norm_w);
        const T* q_norm_w = static_cast<const T*>(L.q_norm_w);
        const T* k_norm_w = static_cast<const T*>(L.k_norm_w);
        const T* post_attn_norm_w = static_cast<const T*>(L.post_attn_norm_w);
        const T* cos_table = static_cast<const T*>(L.cos_table);
        const T* sin_table = static_cast<const T*>(L.sin_table);

        const uint8_t*      q_proj_packed = static_cast<const uint8_t*>(L.q_proj_w);
        const uint8_t*      k_proj_packed = static_cast<const uint8_t*>(L.k_proj_w);
        const uint8_t*      v_proj_packed = static_cast<const uint8_t*>(L.v_proj_w);
        const uint8_t*      o_proj_packed = static_cast<const uint8_t*>(L.o_proj_w);
        const hip_bfloat16* q_proj_scale  = static_cast<const hip_bfloat16*>(SI.q_proj_scale);
        const hip_bfloat16* q_proj_zero   = static_cast<const hip_bfloat16*>(SI.q_proj_zero);
        const hip_bfloat16* k_proj_scale  = static_cast<const hip_bfloat16*>(SI.k_proj_scale);
        const hip_bfloat16* k_proj_zero   = static_cast<const hip_bfloat16*>(SI.k_proj_zero);
        const hip_bfloat16* v_proj_scale  = static_cast<const hip_bfloat16*>(SI.v_proj_scale);
        const hip_bfloat16* v_proj_zero   = static_cast<const hip_bfloat16*>(SI.v_proj_zero);
        const hip_bfloat16* o_proj_scale  = static_cast<const hip_bfloat16*>(SI.o_proj_scale);
        const hip_bfloat16* o_proj_zero   = static_cast<const hip_bfloat16*>(SI.o_proj_zero);

        // A0: load hidden_io BF16 -> hidden_f32 per seq.
        for (int b = 0; b < B; ++b) {
            float* hidden_f32 = workspace + b * ws_stride;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                hidden_f32[c] = g4_to_float(hb[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A1: input_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* hidden_f32 = workspace + b * ws_stride;
                float* normed_f32 = hidden_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(normed_f32, hidden_f32, input_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A2: work-stealing INT4 Q/K/V matmul - amortize weight+dequant across B seqs.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_in = (hidden_size + gsz - 1) / gsz;
        const int byte_cols_in  = hidden_size / 2;

        {
            const int total_qkv = shared_kv ? q_dim : (q_dim + 2 * kv_dim);
            while (true) {
                __shared__ unsigned int claimed;
                if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
                __syncthreads();
                const unsigned int row = claimed;
                if (row >= static_cast<unsigned int>(total_qkv)) break;

                const uint8_t*      w_packed_base;
                const hip_bfloat16* w_scale_base;
                const hip_bfloat16* w_zero_base;
                int weight_row;
                if (row < static_cast<unsigned int>(q_dim)) {
                    w_packed_base = q_proj_packed;
                    w_scale_base  = q_proj_scale;
                    w_zero_base   = q_proj_zero;
                    weight_row    = static_cast<int>(row);
                } else if (row < static_cast<unsigned int>(q_dim + kv_dim)) {
                    w_packed_base = k_proj_packed;
                    w_scale_base  = k_proj_scale;
                    w_zero_base   = k_proj_zero;
                    weight_row    = static_cast<int>(row - q_dim);
                } else {
                    w_packed_base = v_proj_packed;
                    w_scale_base  = v_proj_scale;
                    w_zero_base   = v_proj_zero;
                    weight_row    = static_cast<int>(row - q_dim - kv_dim);
                }
                const int scale_row = weight_row / gsz;
                const uint8_t* w_row_bytes =
                    w_packed_base + static_cast<size_t>(weight_row) * byte_cols_in;

                float pacc[G4_MAX_BATCH_SIZE];
                #pragma unroll
                for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

                for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                                      scale_row, c0, scale_cols_in, gsz, w);
                    for (int b = 0; b < B; ++b) {
                        const float* normed_f32 = workspace + b * ws_stride + hidden_size;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            pacc[b] += w[k] * normed_f32[c0 + k];
                        }
                    }
                }
                for (int b = 0; b < B; ++b) {
                    lds[tid] = pacc[b];
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
                        proj_f32[row] = lds[0];
                    }
                    __syncthreads();
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A3: q/k/v norms + RoPE + KV append (block 0, per-seq loop).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
                float* q_f32 = proj_f32;
                float* k_f32 = proj_f32 + q_dim;
                float* v_f32 = proj_f32 + q_dim + kv_dim;
                const int position_b = S.seqlen_offset[b];
                const int max_T_b = S.kv_max_t[b];
                T* k_cache_b = static_cast<T*>(S.kv_cache_k[b]);
                T* v_cache_b = static_cast<T*>(S.kv_cache_v[b]);
                if (!shared_kv) {
                    g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                        num_q_heads, head_dim, eps, lds);
                    g4_block_rms_norm_per_head_f32<T>(k_f32, k_norm_w,
                        num_kv_heads, head_dim, eps, lds);
                    g4_block_rms_norm_per_head_f32<T>(v_f32, (const T*)nullptr,
                        num_kv_heads, head_dim, eps, lds);
                    g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                        num_q_heads, head_dim, rotary_dim, position_b);
                    g4_block_rope_f32<T>(k_f32, cos_table, sin_table,
                        num_kv_heads, head_dim, rotary_dim, position_b);
                    g4_block_kv_append_f32_to_bf16<T>(k_f32, v_f32, k_cache_b, v_cache_b,
                        num_kv_heads, head_dim, position_b, max_T_b);
                } else {
                    g4_block_rms_norm_per_head_f32<T>(q_f32, q_norm_w,
                        num_q_heads, head_dim, eps, lds);
                    g4_block_rope_f32<T>(q_f32, cos_table, sin_table,
                        num_q_heads, head_dim, rotary_dim, position_b);
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A4: attention scores (per-seq loop).
        const int num_q_per_kv = num_q_heads / num_kv_heads;
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            const T* k_cache_b = static_cast<const T*>(S.kv_cache_k[b]);
            const float* proj_f32 = workspace + b * ws_stride + 2 * hidden_size;
            const float* q_f32 = proj_f32;
            float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                + q_dim + 2 * kv_dim;
            const int kv_len = position_b + 1;
            const int last = kv_len - 1;
            const int min_t = (sliding_window > 0) ? max(0, last - sliding_window + 1) : 0;
            const int total_items = num_q_heads * kv_len;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int t = item % kv_len;
                const int q_h = item / kv_len;
                float val;
                if (t < min_t) {
                    val = -INFINITY;
                } else {
                    const int kv_h = q_h / num_q_per_kv;
                    const float* qr = q_f32 +
                        static_cast<size_t>(q_h) * head_dim;
                    const T* kr = k_cache_b +
                        (static_cast<size_t>(kv_h) * max_T_b + t) * head_dim;
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += qr[d] * g4_to_float(kr[d]);
                    }
                    val = acc * scale;
                }
                scores_f32[static_cast<size_t>(q_h) * max_T_b + t] = val;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A5: softmax per q head (per-seq loop).
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                + q_dim + 2 * kv_dim;
            const int kv_len = position_b + 1;
            for (int q_h = blockIdx.x; q_h < num_q_heads; q_h += nb) {
                float* row = scores_f32 + static_cast<size_t>(q_h) * max_T_b;
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
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A6: value aggregation (per-seq loop).
        for (int b = 0; b < B; ++b) {
            const int position_b = S.seqlen_offset[b];
            const int max_T_b = S.kv_max_t[b];
            const T* v_cache_b = static_cast<const T*>(S.kv_cache_v[b]);
            const float* scores_f32 = workspace + b * ws_stride + 2 * hidden_size
                                      + q_dim + 2 * kv_dim;
            float* attn_out_f32 = workspace + b * ws_stride + 2 * hidden_size
                                  + q_dim + 2 * kv_dim + num_q_heads * max_T_b;
            const int kv_len = position_b + 1;
            const int total_items = num_q_heads * head_dim;
            for (int item = blockIdx.x * bs + tid;
                 item < total_items; item += nb * bs) {
                const int d = item % head_dim;
                const int q_h = item / head_dim;
                const int kv_h = q_h / num_q_per_kv;
                const float* pr = scores_f32 +
                    static_cast<size_t>(q_h) * max_T_b;
                float acc = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const T* vr = v_cache_b +
                        (static_cast<size_t>(kv_h) * max_T_b + t) * head_dim;
                    acc += pr[t] * g4_to_float(vr[d]);
                }
                attn_out_f32[static_cast<size_t>(q_h) * head_dim + d] = acc;
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A7: work-stealing INT4 o_proj - amortized.
        const int attn_out_base_off = 2 * hidden_size + q_dim + 2 * kv_dim
                                    + num_q_heads * L.kv_max_t;
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_o = (q_dim + gsz - 1) / gsz;
        const int byte_cols_o  = q_dim / 2;

        while (true) {
            __shared__ unsigned int claimed;
            if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
            __syncthreads();
            const unsigned int row = claimed;
            if (row >= static_cast<unsigned int>(hidden_size)) break;

            const int scale_row = static_cast<int>(row) / gsz;
            const uint8_t* w_row_bytes =
                o_proj_packed + static_cast<size_t>(row) * byte_cols_o;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c0 = tid * 8; c0 < q_dim; c0 += bs * 8) {
                uint32_t packed;
                __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                float w[8];
                g4_int4_dequant_8(packed, o_proj_scale, o_proj_zero,
                                  scale_row, c0, scale_cols_o, gsz, w);
                for (int b = 0; b < B; ++b) {
                    const float* attn_out_f32 = workspace + b * ws_stride
                                                + attn_out_base_off;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        pacc[b] += w[k] * attn_out_f32[c0 + k];
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* oproj_f32 = workspace + b * ws_stride
                                       + attn_out_base_off + q_dim;
                    oproj_f32[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A8: post_attn_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* oproj_f32 = workspace + b * ws_stride
                                   + attn_out_base_off + q_dim;
                float* oproj_normed = oproj_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(oproj_normed, oproj_f32, post_attn_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // A9: residual add + BF16 store -> hidden_io = h_mid.
        for (int b = 0; b < B; ++b) {
            const float* hidden_f32 = workspace + b * ws_stride;
            const float* oproj_normed = workspace + b * ws_stride
                                        + attn_out_base_off + q_dim + hidden_size;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                hb[c] = g4_from_float<T>(hidden_f32[c] + oproj_normed[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // ==== Phase B - MLP + PLE ====
        const int intermediate_size = L.intermediate_size;
        const float layer_scalar = L.layer_scalar;
        const T* pre_ff_norm_w = static_cast<const T*>(L.pre_ff_norm_w);
        const T* post_ff_norm_w = static_cast<const T*>(L.post_ff_norm_w);
        const T* pli_post_norm_w =
            static_cast<const T*>(L.post_per_layer_input_norm_w);

        const uint8_t*      gate_proj_packed = static_cast<const uint8_t*>(L.gate_proj_w);
        const uint8_t*      up_proj_packed   = static_cast<const uint8_t*>(L.up_proj_w);
        const uint8_t*      down_proj_packed = static_cast<const uint8_t*>(L.down_proj_w);
        const uint8_t*      pli_gate_packed  = static_cast<const uint8_t*>(L.per_layer_input_gate_w);
        const uint8_t*      pli_proj_packed  = static_cast<const uint8_t*>(L.per_layer_projection_w);
        const hip_bfloat16* gate_proj_scale  = static_cast<const hip_bfloat16*>(SI.gate_proj_scale);
        const hip_bfloat16* gate_proj_zero   = static_cast<const hip_bfloat16*>(SI.gate_proj_zero);
        const hip_bfloat16* up_proj_scale    = static_cast<const hip_bfloat16*>(SI.up_proj_scale);
        const hip_bfloat16* up_proj_zero     = static_cast<const hip_bfloat16*>(SI.up_proj_zero);
        const hip_bfloat16* down_proj_scale  = static_cast<const hip_bfloat16*>(SI.down_proj_scale);
        const hip_bfloat16* down_proj_zero   = static_cast<const hip_bfloat16*>(SI.down_proj_zero);
        const hip_bfloat16* pli_gate_scale   = static_cast<const hip_bfloat16*>(SI.per_layer_input_gate_scale);
        const hip_bfloat16* pli_gate_zero    = static_cast<const hip_bfloat16*>(SI.per_layer_input_gate_zero);
        const hip_bfloat16* pli_proj_scale   = static_cast<const hip_bfloat16*>(SI.per_layer_projection_scale);
        const hip_bfloat16* pli_proj_zero    = static_cast<const hip_bfloat16*>(SI.per_layer_projection_zero);

        // B0: load h_mid BF16 -> b_hidden_f32 per seq.
        for (int b = 0; b < B; ++b) {
            float* b_hidden_f32 = workspace + b * ws_stride;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                b_hidden_f32[c] = g4_to_float(hb[c]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B1: pre_ff_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_hidden_f32 = workspace + b * ws_stride;
                float* b_normed_ff = b_hidden_f32 + hidden_size;
                g4_block_rms_norm_f32<T>(b_normed_ff, b_hidden_f32, pre_ff_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B2: work-stealing INT4 gate+up - amortized.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        {
            const int total_mlp1 = 2 * intermediate_size;
            while (true) {
                __shared__ unsigned int claimed;
                if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
                __syncthreads();
                const unsigned int row = claimed;
                if (row >= static_cast<unsigned int>(total_mlp1)) break;

                const uint8_t*      w_packed_base;
                const hip_bfloat16* w_scale_base;
                const hip_bfloat16* w_zero_base;
                int weight_row;
                if (row < static_cast<unsigned int>(intermediate_size)) {
                    w_packed_base = gate_proj_packed;
                    w_scale_base  = gate_proj_scale;
                    w_zero_base   = gate_proj_zero;
                    weight_row    = static_cast<int>(row);
                } else {
                    w_packed_base = up_proj_packed;
                    w_scale_base  = up_proj_scale;
                    w_zero_base   = up_proj_zero;
                    weight_row    = static_cast<int>(row - intermediate_size);
                }
                const int scale_row = weight_row / gsz;
                const uint8_t* w_row_bytes =
                    w_packed_base + static_cast<size_t>(weight_row) * byte_cols_in;

                float pacc[G4_MAX_BATCH_SIZE];
                #pragma unroll
                for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

                for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                    float w[8];
                    g4_int4_dequant_8(packed, w_scale_base, w_zero_base,
                                      scale_row, c0, scale_cols_in, gsz, w);
                    for (int b = 0; b < B; ++b) {
                        const float* b_normed_ff = workspace + b * ws_stride + hidden_size;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            pacc[b] += w[k] * b_normed_ff[c0 + k];
                        }
                    }
                }
                for (int b = 0; b < B; ++b) {
                    lds[tid] = pacc[b];
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float* b_gate_up = workspace + b * ws_stride + 2 * hidden_size;
                        b_gate_up[row] = lds[0];
                    }
                    __syncthreads();
                }
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B3: gated_proj[i] = gelu(gate[i]) * up[i], per seq.
        for (int b = 0; b < B; ++b) {
            const float* b_gate_up = workspace + b * ws_stride + 2 * hidden_size;
            const float* gate_slot = b_gate_up;
            const float* up_slot = b_gate_up + intermediate_size;
            float* b_gated_proj = workspace + b * ws_stride + 2 * hidden_size
                                  + 2 * intermediate_size;
            for (int i = tid + blockIdx.x * bs;
                 i < intermediate_size; i += bs * nb) {
                b_gated_proj[i] = g4_gelu_tanh_scalar(gate_slot[i]) * up_slot[i];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B4: work-stealing INT4 down_proj - amortized.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_i = (intermediate_size + gsz - 1) / gsz;
        const int byte_cols_i  = intermediate_size / 2;

        while (true) {
            __shared__ unsigned int claimed;
            if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
            __syncthreads();
            const unsigned int row = claimed;
            if (row >= static_cast<unsigned int>(hidden_size)) break;

            const int scale_row = static_cast<int>(row) / gsz;
            const uint8_t* w_row_bytes =
                down_proj_packed + static_cast<size_t>(row) * byte_cols_i;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c0 = tid * 8; c0 < intermediate_size; c0 += bs * 8) {
                uint32_t packed;
                __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                float w[8];
                g4_int4_dequant_8(packed, down_proj_scale, down_proj_zero,
                                  scale_row, c0, scale_cols_i, gsz, w);
                for (int b = 0; b < B; ++b) {
                    const float* b_gated_proj = workspace + b * ws_stride
                                                + 2 * hidden_size + 2 * intermediate_size;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        pacc[b] += w[k] * b_gated_proj[c0 + k];
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_down_out = workspace + b * ws_stride
                                        + 2 * hidden_size + 3 * intermediate_size;
                    b_down_out[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B5: post_ff_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_down_out = workspace + b * ws_stride
                                    + 2 * hidden_size + 3 * intermediate_size;
                float* b_normed_down = b_down_out + hidden_size;
                g4_block_rms_norm_f32<T>(b_normed_down, b_down_out, post_ff_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B6: h_pre_ple = b_hidden_f32 + b_normed_down, per seq.
        for (int b = 0; b < B; ++b) {
            const float* b_hidden_f32 = workspace + b * ws_stride;
            const float* b_normed_down = workspace + b * ws_stride
                                         + 3 * hidden_size + 3 * intermediate_size;
            float* b_h_pre_ple = workspace + b * ws_stride
                                 + 4 * hidden_size + 3 * intermediate_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                b_h_pre_ple[c] = b_hidden_f32[c] + b_normed_down[c];
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B7: work-stealing INT4 per_layer_input_gate - amortized.
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

            const int scale_row = static_cast<int>(row) / gsz;
            const uint8_t* w_row_bytes =
                pli_gate_packed + static_cast<size_t>(row) * byte_cols_in;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c0 = tid * 8; c0 < hidden_size; c0 += bs * 8) {
                uint32_t packed;
                __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                float w[8];
                g4_int4_dequant_8(packed, pli_gate_scale, pli_gate_zero,
                                  scale_row, c0, scale_cols_in, gsz, w);
                for (int b = 0; b < B; ++b) {
                    const float* b_h_pre_ple = workspace + b * ws_stride
                                               + 4 * hidden_size + 3 * intermediate_size;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        pacc[b] += w[k] * b_h_pre_ple[c0 + k];
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_gated_ple = workspace + b * ws_stride
                                         + 5 * hidden_size + 3 * intermediate_size;
                    b_gated_ple[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B8: gated_act = gelu(gated_ple) * per_layer_input, per seq.
        for (int b = 0; b < B; ++b) {
            const float* b_gated_ple = workspace + b * ws_stride
                                       + 5 * hidden_size + 3 * intermediate_size;
            float* b_gated_act = workspace + b * ws_stride
                                 + 5 * hidden_size + 3 * intermediate_size + ple_hidden;
            const T* per_layer_input = per_layer_inputs
                + (static_cast<size_t>(b) * num_layers + layer) * ple_hidden;
            for (int i = tid + blockIdx.x * bs; i < ple_hidden; i += bs * nb) {
                const float g = g4_gelu_tanh_scalar(b_gated_ple[i]);
                b_gated_act[i] = g * g4_to_float(per_layer_input[i]);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B9: work-stealing INT4 per_layer_projection - amortized.
        if (blockIdx.x == 0 && tid == 0) {
            __atomic_store_n(matvec_counter, 0u, __ATOMIC_RELAXED);
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        const int scale_cols_ple = (ple_hidden + gsz - 1) / gsz;
        const int byte_cols_ple  = ple_hidden / 2;

        while (true) {
            __shared__ unsigned int claimed;
            if (tid == 0) claimed = atomicAdd(matvec_counter, 1u);
            __syncthreads();
            const unsigned int row = claimed;
            if (row >= static_cast<unsigned int>(hidden_size)) break;

            const int scale_row = static_cast<int>(row) / gsz;
            const uint8_t* w_row_bytes =
                pli_proj_packed + static_cast<size_t>(row) * byte_cols_ple;

            float pacc[G4_MAX_BATCH_SIZE];
            #pragma unroll
            for (int b = 0; b < G4_MAX_BATCH_SIZE; ++b) pacc[b] = 0.0f;

            for (int c0 = tid * 8; c0 < ple_hidden; c0 += bs * 8) {
                uint32_t packed;
                __builtin_memcpy(&packed, w_row_bytes + (c0 / 2), 4);
                float w[8];
                g4_int4_dequant_8(packed, pli_proj_scale, pli_proj_zero,
                                  scale_row, c0, scale_cols_ple, gsz, w);
                for (int b = 0; b < B; ++b) {
                    const float* b_gated_act = workspace + b * ws_stride
                                               + 5 * hidden_size + 3 * intermediate_size
                                               + ple_hidden;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        pacc[b] += w[k] * b_gated_act[c0 + k];
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                lds[tid] = pacc[b];
                __syncthreads();
                for (int s = bs / 2; s > 0; s >>= 1) {
                    if (tid < s) lds[tid] += lds[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    float* b_projected = workspace + b * ws_stride
                                         + 5 * hidden_size + 3 * intermediate_size
                                         + 2 * ple_hidden;
                    b_projected[row] = lds[0];
                }
                __syncthreads();
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B10: post_per_layer_input_norm on block 0 (loop B).
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; ++b) {
                float* b_projected = workspace + b * ws_stride
                                     + 5 * hidden_size + 3 * intermediate_size
                                     + 2 * ple_hidden;
                float* b_normed_ple = workspace + b * ws_stride
                                      + 6 * hidden_size + 3 * intermediate_size
                                      + 2 * ple_hidden;
                g4_block_rms_norm_f32<T>(b_normed_ple, b_projected,
                                          pli_post_norm_w,
                                          hidden_size, eps, lds);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);

        // B11: hidden_io = bf16((h_pre_ple + normed_ple) * layer_scalar), per seq.
        for (int b = 0; b < B; ++b) {
            const float* b_h_pre_ple = workspace + b * ws_stride
                                       + 4 * hidden_size + 3 * intermediate_size;
            const float* b_normed_ple = workspace + b * ws_stride
                                        + 6 * hidden_size + 3 * intermediate_size
                                        + 2 * ple_hidden;
            T* hb = hidden_io + static_cast<size_t>(b) * hidden_size;
            for (int c = tid + blockIdx.x * bs; c < hidden_size; c += bs * nb) {
                const float v = (b_h_pre_ple[c] + b_normed_ple[c]) * layer_scalar;
                hb[c] = g4_from_float<T>(v);
            }
        }
        g4_grid_barrier(barrier_counter, barrier_flag, nb);
    }
}
