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
#pragma once

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
    void* kv_shadow_k;                 // [1, num_kv_heads, max_T, head_dim] BF16 shadow for decode-appended tokens
    void* kv_shadow_v;                 // [1, num_kv_heads, max_T, head_dim] BF16 shadow for decode-appended tokens
    int kv_shadow_start;               // first position stored in shadow, -1 when disabled
};

// =============================================================================
// FP8 E4M3 runtime dequantization support
// =============================================================================

// FP8 scale_inv pointers for runtime dequantization.
// One per decoder layer, passed as a separate kernel parameter.
// Struct layout must match kernel_ffi::FP8ScaleDesc in Rust.
struct Qwen35FP8ScaleDesc {
    // Common MLP weights
    const void* gate_proj_scale;
    const void* up_proj_scale;
    const void* down_proj_scale;
    // Linear attention weights
    const void* qkv_proj_scale;
    const void* z_proj_scale;
    const void* b_proj_scale;
    const void* a_proj_scale;
    const void* linear_out_proj_scale;
    // Full attention weights
    const void* q_proj_scale;
    const void* k_proj_scale;
    const void* v_proj_scale;
    const void* o_proj_scale;
    // Block size for scale_inv indexing (typically 128)
    int block_size;
};

// Convert FP8 E4M3 byte to F32.
// E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7, no inf, NaN=0x7F/0xFF
__device__ inline float fp8_e4m3_to_float(uint8_t byte) {
    int sign = (byte >> 7) & 1;
    int exp = (byte >> 3) & 0xF;
    int mantissa = byte & 0x7;
    if (byte == 0x7F || byte == 0xFF) return 0.0f;  // NaN → 0
    float val;
    if (exp == 0) {
        // Subnormal: 2^(-6) * (mantissa / 8)
        val = static_cast<float>(mantissa) / 8.0f * 1.52587890625e-2f;  // 2^(-6) = 1/64
    } else {
        // Normal: 2^(exp-7) * (1 + mantissa/8)
        val = (1.0f + static_cast<float>(mantissa) / 8.0f) * exp2f(static_cast<float>(exp - 7));
    }
    return sign ? -val : val;
}

// Round an f32 value to BF16 precision (round-to-nearest-even) and widen
// back to f32. Matches PyTorch's `.to(bfloat16)` semantics.
// Used so INT4 / FP8 weight reconstruction produces the same effective
// bf16-rounded weights that a bf16-dtype PyTorch Linear sees when
// `.weight.data` is written as `bf16((q - zf) * s)`.
__device__ __forceinline__ float bf16_round_rne_f32(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    // Preserve NaN: quiet any signaling NaN in the upper-16 representation.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu) != 0u) {
        bits = (bits & 0xFFFF0000u) | 0x00400000u;
    } else {
        uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
        bits += rounding_bias;
        bits &= 0xFFFF0000u;
    }
    float y;
    __builtin_memcpy(&y, &bits, 4);
    return y;
}

// Same semantics as bf16_round_rne_f32 for finite inputs — no NaN-preservation
// branch. Use in weight-dequant hot paths where inputs are provably finite
// (INT4: nibble∈[0,15] × finite bf16 scale − finite bf16 zero-scale;
// FP8: finite e4m3 value × finite bf16 scale). The NaN branch was measurable
// on the 890M iGPU where int4_dequant_8 runs it 8× per packed u32.
__device__ __forceinline__ float bf16_round_rne_f32_finite(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    float y;
    __builtin_memcpy(&y, &bits, 4);
    return y;
}

// KV cache FP8 scale pointers for dynamic quantization.
// One per decoder layer, passed as a separate kernel parameter.
// Struct layout must match kernel_ffi::KVCacheFp8Desc in Rust.
struct KVCacheFp8Desc {
    void* kv_scale_k;   // [num_kv_heads, max_T] F32 — per-head-per-position absmax scale
    void* kv_scale_v;   // [num_kv_heads, max_T] F32
};

// Per-sequence state pointers for batched decode.
// One per layer (parallel to Qwen35DecodeLayerDesc).
// When batch_size > 1, the kernel reads per-sequence state from here
// instead of from Qwen35DecodeLayerDesc.
// Struct layout must match kernel_ffi::BatchSeqDesc in Rust.
#define MAX_BATCH_SIZE 8
struct BatchSeqDesc {
    int seqlen_offset[MAX_BATCH_SIZE];
    // Full attention per-sequence
    void* kv_cache_k[MAX_BATCH_SIZE];
    void* kv_cache_v[MAX_BATCH_SIZE];
    int kv_len[MAX_BATCH_SIZE];
    int kv_max_t[MAX_BATCH_SIZE];
    void* kv_shadow_k[MAX_BATCH_SIZE];
    void* kv_shadow_v[MAX_BATCH_SIZE];
    int kv_shadow_start[MAX_BATCH_SIZE];
    // Linear attention per-sequence
    void* conv_state[MAX_BATCH_SIZE];
    void* recurrent_state[MAX_BATCH_SIZE];
    // FP8 KV scales per-sequence
    void* kv_scale_k[MAX_BATCH_SIZE];
    void* kv_scale_v[MAX_BATCH_SIZE];
};

// Convert F32 to FP8 E4M3 byte (inverse of fp8_e4m3_to_float).
// Clamps to representable range [-448, 448], rounds to nearest.
__device__ inline uint8_t float_to_fp8_e4m3(float val) {
    uint8_t sign = 0;
    if (val < 0.0f) { sign = 0x80; val = -val; }
    // Clamp to max representable E4M3 value
    if (val >= 448.0f) return sign | 0x7E;  // max normal: exp=14, mantissa=6 → 2^7*(1+6/8)=448
    if (val < 1.52587890625e-2f * 0.125f) return sign;  // too small → ±0
    // Subnormal range: val < 2^(-6) = 0.015625
    if (val < 0.015625f) {
        int mantissa = __float2int_rn(val / 1.52587890625e-2f * 8.0f);
        if (mantissa < 0) mantissa = 0;
        if (mantissa >= 8) return sign | 0x08;  // rounds up to the smallest normal
        return sign | static_cast<uint8_t>(mantissa);
    }
    // Normal range
    float log2_val = log2f(val);
    int exp = static_cast<int>(floorf(log2_val)) + 7;
    if (exp < 1) exp = 1;
    if (exp > 14) exp = 14;  // would be NaN range — clamp
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

// Read one FP8 weight element with block-wise dequantization.
// w_ptr: void pointer to FP8 weight data [out_dim, in_dim] stored as uint8_t
// scale_ptr: void pointer to BF16 scale_inv [out_dim/block, in_dim/block]
// row, col: element coordinates in the weight matrix
// cols: number of columns (in_dim) in the weight matrix
// block_size: FP8 quantization block size (typically 128)
__device__ inline float fp8_dequant_weight(
    const void* w_ptr, const void* scale_ptr,
    int row, int col, int cols, int block_size
) {
    const uint8_t* fp8 = static_cast<const uint8_t*>(w_ptr);
    float val = fp8_e4m3_to_float(fp8[static_cast<size_t>(row) * cols + col]);
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    int scale_row = row / block_size;
    int scale_col = col / block_size;
    int scale_cols = (cols + block_size - 1) / block_size;
    float scale = static_cast<float>(scales[scale_row * scale_cols + scale_col]);
    // Truncate to BF16 precision to match PyTorch's dequant path (fp8→F32→BF16).
    // Without this, F32 accumulation of F32-dequanted weights drifts from the
    // BF16-truncated weights that PyTorch uses, causing token divergence on marginal
    // argmax decisions after 32 layers of compounding.
    return bf16_round_rne_f32_finite((val * scale));
}

// Fast FP8 dequant using LDS lookup table.
// lut: 256-entry F32 table in LDS, precomputed from fp8_e4m3_to_float.
// Replaces branchy fp8_e4m3_to_float() with a single LDS read.
__device__ inline float fp8_dequant_weight_lut(
    const void* w_ptr, const void* scale_ptr,
    int row, int col, int cols, int block_size,
    const float* __restrict__ lut
) {
    const uint8_t* fp8 = static_cast<const uint8_t*>(w_ptr);
    float val = lut[fp8[static_cast<size_t>(row) * cols + col]];
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    int scale_row = row / block_size;
    int scale_col = col / block_size;
    int scale_cols = (cols + block_size - 1) / block_size;
    float scale = static_cast<float>(scales[scale_row * scale_cols + scale_col]);
    return bf16_round_rne_f32_finite((val * scale));
}

// =============================================================================
// INT4 group-quantized weight dequantization support
// =============================================================================

// INT4 scale+zero pointers for runtime dequantization.
// One per decoder layer, passed as a separate kernel parameter.
// Struct layout must match kernel_ffi::INT4ScaleDesc in Rust.
// Weights are packed as 2×INT4 per byte (low nibble = even col, high nibble = odd col).
// Asymmetric quantization: dequant = (int4_val - zero) * scale
struct Qwen35INT4ScaleDesc {
    // Common MLP weights
    const void* gate_proj_scale;
    const void* gate_proj_zero;
    const void* up_proj_scale;
    const void* up_proj_zero;
    const void* down_proj_scale;
    const void* down_proj_zero;
    // Linear attention weights
    const void* qkv_proj_scale;
    const void* qkv_proj_zero;
    const void* z_proj_scale;
    const void* z_proj_zero;
    const void* linear_out_proj_scale;
    const void* linear_out_proj_zero;
    // Full attention weights
    const void* q_proj_scale;
    const void* q_proj_zero;
    const void* k_proj_scale;
    const void* k_proj_zero;
    const void* v_proj_scale;
    const void* v_proj_zero;
    const void* o_proj_scale;
    const void* o_proj_zero;
    // Group size for INT4 quantization (typically 128)
    int group_size;
};

// Dequantize 8 INT4 weights from 4 packed bytes.
// packed: 4 bytes = 8 nibbles, low nibble first per byte.
// scales: BF16 scale array, zeros: BF16 zero array.
// row, col: starting column (must be aligned to 2), cols: weight matrix width.
// gsz: group size, scale_row: row / gsz, scale_cols: (cols + gsz - 1) / gsz.
// out[0..7]: dequantized F32 values.
//
// Optimized: with group_size=128, 8 consecutive columns almost always share
// the same group. Fast path loads scale/zero once and precomputes zero*scale
// to replace per-element (nibble - zero) * scale with nibble * scale - zs.
// Only falls back to per-element lookup at group boundaries.
__device__ inline void int4_dequant_8(
    uint32_t packed,
    const hip_bfloat16* __restrict__ scales,
    const hip_bfloat16* __restrict__ zeros,
    int scale_row, int col, int scale_cols, int gsz,
    float out[8]
) {
    const int sb = scale_row * scale_cols;
    // Extract 8 nibbles
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
    // Round reconstructed weight values through BF16 so the kernel produces the
    // same effective weight that a PyTorch bf16-dtype Linear holds after
    // `.weight.data.copy_(bf16((q - zf) * s))` — matches the GPTQ-baked Python
    // oracle and the existing FP8 dequant precision fix.
    if (g0 == g7) {
        // Fast path: all 8 elements in same group — 1 scale + 1 zero load
        float s = static_cast<float>(scales[sb + g0]);
        float zs = static_cast<float>(zeros[sb + g0]) * s;  // precompute zero*scale
        out[0] = bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
        out[1] = bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
        out[2] = bf16_round_rne_f32_finite(static_cast<float>(n2) * s - zs);
        out[3] = bf16_round_rne_f32_finite(static_cast<float>(n3) * s - zs);
        out[4] = bf16_round_rne_f32_finite(static_cast<float>(n4) * s - zs);
        out[5] = bf16_round_rne_f32_finite(static_cast<float>(n5) * s - zs);
        out[6] = bf16_round_rne_f32_finite(static_cast<float>(n6) * s - zs);
        out[7] = bf16_round_rne_f32_finite(static_cast<float>(n7) * s - zs);
    } else {
        // Slow path: group boundary crossing — per-element lookup
        float s0 = static_cast<float>(scales[sb + g0]);
        float zs0 = static_cast<float>(zeros[sb + g0]) * s0;
        float s1 = static_cast<float>(scales[sb + g7]);
        float zs1 = static_cast<float>(zeros[sb + g7]) * s1;
        // Elements 0..boundary use g0, rest use g7
        #define I4DQ(idx, ni) do { \
            int gi = (col + idx) / gsz; \
            float si = (gi == g0) ? s0 : s1; \
            float zsi = (gi == g0) ? zs0 : zs1; \
            out[idx] = bf16_round_rne_f32_finite(static_cast<float>(ni) * si - zsi); \
        } while(0)
        I4DQ(0, n0); I4DQ(1, n1); I4DQ(2, n2); I4DQ(3, n3);
        I4DQ(4, n4); I4DQ(5, n5); I4DQ(6, n6); I4DQ(7, n7);
        #undef I4DQ
    }
}

// Dequantize a single INT4 weight at (row, col) from packed storage.
__device__ inline float int4_dequant_scalar(
    const void* w_ptr, const void* scale_ptr, const void* zero_ptr,
    int row, int col, int cols, int group_size
) {
    const uint8_t* data = static_cast<const uint8_t*>(w_ptr);
    int byte_cols = cols / 2;
    uint8_t packed_byte = data[static_cast<size_t>(row) * byte_cols + col / 2];
    int nibble = (col & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
    const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(scale_ptr);
    const hip_bfloat16* zeros = static_cast<const hip_bfloat16*>(zero_ptr);
    int si = (row / group_size) * ((cols + group_size - 1) / group_size) + col / group_size;
    float s = static_cast<float>(scales[si]);
    // Round through BF16 so reconstruction matches Python's bf16-stored Q_dq.
    return bf16_round_rne_f32_finite(
        static_cast<float>(nibble) * s - static_cast<float>(zeros[si]) * s);
}

template <typename T>
__device__ inline float dotcache_qwen35_to_float(T value);

template <>
__device__ inline float dotcache_qwen35_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float dotcache_qwen35_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float dotcache_qwen35_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T dotcache_qwen35_from_float(float value);

template <>
__device__ inline __half dotcache_qwen35_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float dotcache_qwen35_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 dotcache_qwen35_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float dotcache_qwen35_binary_op_apply(int op, float lhs, float rhs) {
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

__device__ inline size_t dotcache_qwen35_broadcast_elem_index(
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
__device__ inline float dotcache_qwen35_dot_row(
    const T* lhs,
    const T* rhs,
    int size
) {
    float dot = 0.0f;
    for (int idx = 0; idx < size; ++idx) {
        dot += dotcache_qwen35_to_float(lhs[idx]) * dotcache_qwen35_to_float(rhs[idx]);
    }
    return dot;
}

template <>
__device__ inline float dotcache_qwen35_dot_row<__half>(
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

__device__ inline float dotcache_qwen35_exp_fast(float x) {
    return __expf(x);
}

__device__ inline float dotcache_qwen35_sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = dotcache_qwen35_exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = dotcache_qwen35_exp_fast(x);
    return e / (1.0f + e);
}

__device__ inline float dotcache_qwen35_softplus_fast(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return dotcache_qwen35_exp_fast(x);
    return log1pf(dotcache_qwen35_exp_fast(x));
}

template <typename T>
__device__ inline float dotcache_qwen35_conv4_contiguous_t_ge3(
    const T* mixed_qkv,
    size_t mixed_c_offset,
    size_t t,
    const T* weights,
    size_t weight_offset
) {
    const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
    const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
    const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
    const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
    const float x0 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 3)]);
    const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 2)]);
    const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + (t - 1)]);
    const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + t]);
    return ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
}

template <>
__device__ inline float dotcache_qwen35_conv4_contiguous_t_ge3<__half>(
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

template <typename T>
__device__ inline float dotcache_qwen35_conv4_state3_decode(
    const T* mixed_qkv,
    size_t mixed_offset,
    const T* prev_state,
    size_t state_offset,
    const T* weights,
    size_t weight_offset
) {
    const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
    const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
    const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
    const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
    const float x0 = dotcache_qwen35_to_float(prev_state[state_offset]);
    const float x1 = dotcache_qwen35_to_float(prev_state[state_offset + 1]);
    const float x2 = dotcache_qwen35_to_float(prev_state[state_offset + 2]);
    const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_offset]);
    return ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
}

template <>
__device__ inline float dotcache_qwen35_conv4_state3_decode<__half>(
    const __half* mixed_qkv,
    size_t mixed_offset,
    const __half* prev_state,
    size_t state_offset,
    const __half* weights,
    size_t weight_offset
) {
    const __half2 x01 = __halves2half2(
        prev_state[state_offset],
        prev_state[state_offset + 1]);
    const __half2 x23 = __halves2half2(
        prev_state[state_offset + 2],
        mixed_qkv[mixed_offset]);
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

__device__ inline float dotcache_qwen35_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    return value;
}

template <typename T>
__global__ void dotcache_qwen35_full_attention_prefill_kernel(
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

        float running_max = -INFINITY;
        float denom = 0.0f;
        float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int local_dims[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int local_count = 0;
        for (int d = lane; d < head_dim && local_count < 8; d += warpSize) {
            local_dims[local_count++] = d;
        }

        __shared__ float shared_prev_scale;
        __shared__ float shared_curr_scale;
        __shared__ float shared_denom;
        __shared__ float shared_inv_denom;
        __shared__ float shared_new_max;

        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;
            const T* v_row = v_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += dotcache_qwen35_to_float(q_row[d]) * dotcache_qwen35_to_float(k_row[d]);
            }
            float score = dotcache_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                if (!isfinite(running_max)) {
                    running_max = score;
                    denom = 1.0f;
                    shared_prev_scale = 0.0f;
                    shared_curr_scale = 1.0f;
                    shared_new_max = running_max;
                } else {
                    const float new_max = fmaxf(running_max, score);
                    const float prev_scale = expf(running_max - new_max);
                    const float curr_scale = expf(score - new_max);
                    denom = denom * prev_scale + curr_scale;
                    running_max = new_max;
                    shared_prev_scale = prev_scale;
                    shared_curr_scale = curr_scale;
                    shared_new_max = new_max;
                }
                shared_denom = denom;
            }
            __syncthreads();

            if (shared_curr_scale == 1.0f && shared_prev_scale == 0.0f) {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = dotcache_qwen35_to_float(v_row[local_dims[i]]);
                }
            } else {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = local_acc[i] * shared_prev_scale +
                        shared_curr_scale * dotcache_qwen35_to_float(v_row[local_dims[i]]);
                }
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * shared_inv_denom;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void dotcache_qwen35_linear_prefill_conv_pack_kernel(
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
        acc += dotcache_qwen35_to_float(mixed_qkv[input_c_offset + t + static_cast<size_t>(tap)]) *
            dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
    out[tid] = dotcache_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_kernel(
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
            x = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
        } else {
            const int state_idx = state_len + src;
            if (state_idx >= 0) {
                x = dotcache_qwen35_to_float(prev_state[state_c_offset + static_cast<size_t>(state_idx)]);
            }
        }
        acc += x * dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
    }

    const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
    out[tid] = dotcache_qwen35_from_float<T>(silu);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_kernel(
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
                x = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + static_cast<size_t>(src)]);
            } else {
                const int state_idx = state_len + src;
                if (state_idx >= 0) {
                    x = dotcache_qwen35_to_float(
                        prev_state[state_c_offset + static_cast<size_t>(state_idx)]
                    );
                }
            }
            acc += x * dotcache_qwen35_to_float(weights[weight_offset + static_cast<size_t>(tap)]);
        }

        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = logf(1.0f + expf(a_val + bias));
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_kernel_k4s3(
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
            acc = dotcache_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = logf(1.0f + expf(a_val + bias));
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T>
__global__ void dotcache_qwen35_linear_stateful_conv_value_decay_with_state_kernel_k4s3(
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
            acc = dotcache_qwen35_conv4_contiguous_t_ge3(
                mixed_qkv, mixed_c_offset, t, weights, weight_offset);
        } else if (t == 2) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x1 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 2]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else if (t == 1) {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x2 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 1]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        } else {
            const float w0 = dotcache_qwen35_to_float(weights[weight_offset]);
            const float w1 = dotcache_qwen35_to_float(weights[weight_offset + 1]);
            const float w2 = dotcache_qwen35_to_float(weights[weight_offset + 2]);
            const float w3 = dotcache_qwen35_to_float(weights[weight_offset + 3]);
            const float x0 = dotcache_qwen35_to_float(prev_state[state_c_offset + 0]);
            const float x1 = dotcache_qwen35_to_float(prev_state[state_c_offset + 1]);
            const float x2 = dotcache_qwen35_to_float(prev_state[state_c_offset + 2]);
            const float x3 = dotcache_qwen35_to_float(mixed_qkv[mixed_c_offset + 0]);
            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
        }
        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        out[tid] = dotcache_qwen35_from_float<T>(silu);
        return;
    }

    const size_t head = c - static_cast<size_t>(conv_dim);
    const size_t a_base =
        b * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) +
        t * static_cast<size_t>(num_heads);
    const float a_val = dotcache_qwen35_to_float(a[a_base + head]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = logf(1.0f + expf(a_val + bias));
    out[tid] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, int MAX_K = 256>
__global__ void dotcache_qwen35_linear_decode_prepare_kernel(
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
    const int k_head = v_head / head_repeat;
    const int mixed_batch_base = batch * conv_dim;
    const int state_batch_base = batch * conv_dim * state_len;
    const int pair_out_base = pair * packed_width;

    auto conv_channel = [&](int channel) -> float {
        const int weight_base = channel * kernel_size;
        const int state_base = state_batch_base + channel * state_len;
        float acc = 0.0f;
        if (kernel_size == 4 && state_len == 3) {
            acc = dotcache_qwen35_conv4_state3_decode(
                mixed_qkv,
                static_cast<size_t>(mixed_batch_base + channel),
                prev_conv_state,
                static_cast<size_t>(state_base),
                weights,
                static_cast<size_t>(weight_base));
        } else {
            for (int tap = 0; tap < kernel_size; ++tap) {
                float x = 0.0f;
                if (tap + 1 == kernel_size) {
                    x = dotcache_qwen35_to_float(mixed_qkv[mixed_batch_base + channel]);
                } else if (tap < state_len) {
                    x = dotcache_qwen35_to_float(prev_conv_state[state_base + tap]);
                }
                acc += x * dotcache_qwen35_to_float(weights[weight_base + tap]);
            }
        }
        const float silu = acc * dotcache_qwen35_sigmoid_fast(acc);
        return dotcache_qwen35_to_float(dotcache_qwen35_from_float<T>(silu));
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
        const float a_raw = dotcache_qwen35_to_float(a_beta_raw[head_base]);
        const float beta_raw = dotcache_qwen35_to_float(a_beta_raw[head_base + num_v_heads]);
        out[pair_out_base + 2 * head_k_dim + head_v_dim] =
            1.0f / (1.0f + expf(-beta_raw));
        const float bias = dotcache_qwen35_to_float(dt_bias[v_head]);
        const float decay = dotcache_qwen35_to_float(a_log_exp[v_head]);
        const float g = -logf(1.0f + expf(a_raw + bias)) * decay;
        out[pair_out_base + 2 * head_k_dim + head_v_dim + 1] = expf(g);
    }
    if (tid < head_v_dim) {
        const int value_channel = key_dim * 2 + v_head * head_v_dim + tid;
        out[pair_out_base + 2 * head_k_dim + tid] = conv_channel(value_channel);
    }
}

template <int MAX_K = 256>
__global__ void dotcache_qwen35_linear_decode_apply_kernel(
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
__device__ inline void dotcache_qwen35_delta_recurrent_prefill_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < seq_len; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + seq_len * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_recurrent_prefill_kernel(
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
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    dotcache_qwen35_delta_recurrent_prefill_impl(
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
        tid
    );
}

template <typename T, int MAX_K = 256>
__device__ inline void dotcache_qwen35_delta_chunk_step_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
            prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < chunk_size; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_step_kernel(
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
    dotcache_qwen35_delta_chunk_step_impl(
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
__device__ inline void dotcache_qwen35_delta_chunk_scan_raw_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
            initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]
        );
    }

    for (int t = 0; t < total_tokens; ++t) {
        const float g_t = expf(dotcache_qwen35_to_float(g[bh * token_stride_s + t]));
        const int key_row = bh * token_stride_k + t * k_head_dim;
        const int value_row = bh * token_stride_v + t * v_head_dim;
        const int beta_row = bh * token_stride_s + t;

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * dotcache_qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (dotcache_qwen35_to_float(value[value_row + v_idx]) - kv_mem) *
            dotcache_qwen35_to_float(beta[beta_row]);

        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += dotcache_qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * dotcache_qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_t);
    }

    const int state_out = out_base + total_tokens * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_scan_raw_kernel(
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
    dotcache_qwen35_delta_chunk_scan_raw_impl(
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
__device__ inline void dotcache_qwen35_delta_state_scan_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(initial_state[idx]);
        out[idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
        const int value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
        const float state_decay =
            dotcache_qwen35_to_float(packed_scan[packed_chunk_base + 2 * k_head_dim]);
        float update[MAX_K];
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            update[k_idx] = 0.0f;
        }

        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_chunk_base + t * packed_width;
            const int value_row = value_chunk_base + t * v_head_dim;
            float v_prime = 0.0f;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += dotcache_qwen35_to_float(packed_scan[packed_row + k_head_dim + k_idx]) *
                    state[k_idx];
            }
            const float v_new = dotcache_qwen35_to_float(value[value_row + v_idx]) - v_prime;
            for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                update[k_idx] += dotcache_qwen35_to_float(packed_scan[packed_row + k_idx]) * v_new;
            }
        }

        const int out_chunk_base = ((bh * (num_chunks + 1)) + (chunk + 1)) * state_stride;
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] = state_decay * state[k_idx] + update[k_idx];
            out[out_chunk_base + k_idx * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(state[k_idx]);
        }
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_state_scan_kernel(
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
    dotcache_qwen35_delta_state_scan_impl(
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
__device__ inline void dotcache_qwen35_delta_chunk_fused_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
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
                dotcache_qwen35_to_float(packed_chunk[packed_row + k_head_dim + k_idx]) *
                state[k_idx];
            attn +=
                dotcache_qwen35_to_float(packed_chunk[packed_row + 2 * k_head_dim + k_idx]) *
                state[k_idx];
        }
        v_new[t] = dotcache_qwen35_to_float(value[value_base + t * v_head_dim + v_idx]) - v_prime;
        attn_inter[t] = attn;
        out[out_base + t * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(v_new[t]);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] =
            dotcache_qwen35_from_float<T>(attn_inter[t]);
    }

    const float state_decay = dotcache_qwen35_to_float(packed_chunk[packed_base + 3 * k_head_dim]);
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float update = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const int packed_row = packed_base + t * packed_width;
            update += dotcache_qwen35_to_float(packed_chunk[packed_row + k_idx]) * v_new[t];
        }
        out[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
            dotcache_qwen35_from_float<T>(state_decay * state[k_idx] + update);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_fused_kernel(
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
    dotcache_qwen35_delta_chunk_fused_impl(
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
__device__ inline void dotcache_qwen35_delta_full_scan_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
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
                v_prime += dotcache_qwen35_to_float(k_cumdecay_scan[row + k_idx]) * state[k_idx];
                attn += dotcache_qwen35_to_float(q_state_scan[row + k_idx]) * state[k_idx];
            }
            v_new[t] = dotcache_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) -
                v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += dotcache_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = dotcache_qwen35_to_float(state_decay_scan[decay_base + chunk]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_scan + t * k_head_dim;
                update += dotcache_qwen35_to_float(weighted_key_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_local_attn_scan_flat_kernel(
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
        out[tid] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    const float dot = dotcache_qwen35_dot_row(
        query_scan + row_base,
        key_scan + col_base,
        k_head_dim);

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = dotcache_qwen35_from_float<T>(dot * decay);
}

template <typename T>
__global__ void dotcache_qwen35_delta_local_attn_scan_row_kernel(
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
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const int out_base = (bh * num_chunks + chunk) * chunk_size * chunk_size;

    for (int col = lane; col < chunk_size; col += blockDim.x) {
        const int out_idx = out_base + row_in_chunk * chunk_size + col;
        if (col > row_in_chunk) {
            out[out_idx] = dotcache_qwen35_from_float<T>(0.0f);
            continue;
        }

        const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
        const float dot = dotcache_qwen35_dot_row(
            query_row,
            key_scan + col_base,
            k_head_dim);

        const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
        const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
        out[out_idx] = dotcache_qwen35_from_float<T>(dot * decay);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_base_attn_scan_kernel(
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
        out[tid] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }

    const int row_base =
        ((bh * num_chunks + chunk) * chunk_size + row_in_chunk) * k_head_dim;
    const int col_base = ((bh * num_chunks + chunk) * chunk_size + col) * k_head_dim;
    float dot = 0.0f;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        dot += dotcache_qwen35_to_float(k_beta_scan[row_base + k_idx]) *
               dotcache_qwen35_to_float(key_scan[col_base + k_idx]);
    }

    const int exp_base = (bh * num_chunks + chunk) * chunk_size;
    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[exp_base + row_in_chunk]);
    const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[exp_base + col]);
    const float decay = exp_g_s != 0.0f ? (exp_g_t / exp_g_s) : 0.0f;
    out[tid] = dotcache_qwen35_from_float<T>(-dot * decay);
}

template <typename T, int MAX_CHUNK = 64>
__global__ void dotcache_qwen35_delta_attn_solve_scan_kernel(
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
            const float row_val = dotcache_qwen35_to_float(base_attn_scan[base + i * chunk_size + j]);
            float correction = 0.0f;
            for (int k = 0; k < i; ++k) {
                correction += dotcache_qwen35_to_float(base_attn_scan[base + i * chunk_size + k]) *
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
            out[base + i * chunk_size + j] = dotcache_qwen35_from_float<T>(value);
        }
    }
}

template <typename T, int MAX_CHUNK = 64>
__global__ void dotcache_qwen35_delta_attn_solve_from_inputs_kernel(
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
        const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[chunk_base + i]);
        for (int k = 0; k < i; ++k) {
            const int row_base = qk_base + i * k_head_dim;
            const int col_base = qk_base + k * k_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < k_head_dim; ++d) {
                dot += dotcache_qwen35_to_float(k_beta_scan[row_base + d]) *
                    dotcache_qwen35_to_float(key_scan[col_base + d]);
            }
            const float exp_g_s = dotcache_qwen35_to_float(exp_g_scan[chunk_base + k]);
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
            out[out_base + i * chunk_size + j] = dotcache_qwen35_from_float<T>(value);
        }
    }
}

template <typename T>
__global__ void dotcache_qwen35_swiglu_mul_kernel(
    int elem_count,
    const T* gate,
    const T* up,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= elem_count) {
        return;
    }
    const float gate_x = dotcache_qwen35_to_float(gate[idx]);
    const float up_x = dotcache_qwen35_to_float(up[idx]);
    const float silu = gate_x / (1.0f + expf(-gate_x));
    out[idx] = dotcache_qwen35_from_float<T>(silu * up_x);
}

template <typename T, typename IndexT>
__global__ void dotcache_qwen35_embedding_lookup_kernel(
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
        out[idx] = dotcache_qwen35_from_float<T>(0.0f);
        return;
    }
    out[idx] = embeddings[row * hidden_size + col];
}

template <typename T>
__global__ void dotcache_qwen35_output_projection_lookup_kernel(
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
        acc += dotcache_qwen35_to_float(row_hidden[col]) * dotcache_qwen35_to_float(row_weight[col]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_causal_mask_kernel(
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
    out[idx] = dotcache_qwen35_from_float<T>(allowed ? 0.0f : -INFINITY);
}

template <typename T>
__global__ void dotcache_qwen35_cumsum_last_dim_kernel(
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
        acc += dotcache_qwen35_to_float(xs[row_offset + col]);
        out[row_offset + col] = dotcache_qwen35_from_float<T>(acc);
    }
}

template <typename T>
__global__ void dotcache_qwen35_exp_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(expf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void dotcache_qwen35_recip_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(1.0f / dotcache_qwen35_to_float(xs[idx]));
}

template <typename T>
__global__ void dotcache_qwen35_sigmoid_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(
        dotcache_qwen35_sigmoid_fast(dotcache_qwen35_to_float(xs[idx]))
    );
}

template <typename T>
__global__ void dotcache_qwen35_log_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(logf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename In, typename Out>
__global__ void dotcache_qwen35_cast_kernel(
    int total_elems,
    const In* xs,
    Out* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<Out>(dotcache_qwen35_to_float(xs[idx]));
}

__device__ __forceinline__ float dotcache_qwen35_unary_op_apply(
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
        return dotcache_qwen35_sigmoid_fast(x);
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
__global__ void dotcache_qwen35_unary_view_kernel(
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
    const float x = dotcache_qwen35_to_float(xs[in_offset]);
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_unary_op_apply(op, x, scalar));
}

template <typename In, typename Out>
__global__ void dotcache_qwen35_cast_view_kernel(
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
    out[idx] = dotcache_qwen35_from_float<Out>(dotcache_qwen35_to_float(xs[in_offset]));
}

template <typename T>
__global__ void dotcache_qwen35_binary_broadcast_kernel(
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

    const float lhs_f = dotcache_qwen35_to_float(lhs[lhs_offset]);
    const float rhs_f = dotcache_qwen35_to_float(rhs[rhs_offset]);
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_binary_op_apply(op, lhs_f, rhs_f));
}

template <typename T>
__global__ void dotcache_qwen35_mul_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_to_float(xs[idx]) * scalar);
}

template <typename T>
__global__ void dotcache_qwen35_add_scalar_kernel(
    int total_elems,
    float scalar,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(dotcache_qwen35_to_float(xs[idx]) + scalar);
}

template <typename T>
__global__ void dotcache_qwen35_sqrt_kernel(
    int total_elems,
    const T* xs,
    T* out
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) {
        return;
    }
    out[idx] = dotcache_qwen35_from_float<T>(sqrtf(dotcache_qwen35_to_float(xs[idx])));
}

template <typename T>
__global__ void dotcache_qwen35_reduce_keepdim_kernel(
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
    float acc = dotcache_qwen35_to_float(xs[base]);
    for (int r = 1; r < reduce; ++r) {
        const float value = dotcache_qwen35_to_float(xs[base + r * inner]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_reduce_keepdim_view_kernel(
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
    float acc = dotcache_qwen35_to_float(xs[base_offset]);
    for (size_t r = 1; r < reduce_len; ++r) {
        const float value = dotcache_qwen35_to_float(
            xs[base_offset + r * static_cast<size_t>(in_strides[reduce_dim])]);
        if (sum) {
            acc += value;
        } else {
            acc = value > acc ? value : acc;
        }
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

template <typename T>
__global__ void dotcache_qwen35_batched_matmul_kernel(
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

    const size_t lhs_batch_idx = dotcache_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, lhs_batch_dims);
    const size_t rhs_batch_idx = dotcache_qwen35_broadcast_elem_index(
        batch_idx, batch_rank, out_batch_dims, rhs_batch_dims);

    const size_t lhs_base = lhs_batch_idx * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t rhs_base = rhs_batch_idx * static_cast<size_t>(k) * static_cast<size_t>(n);
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += dotcache_qwen35_to_float(lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(kk)]) *
               dotcache_qwen35_to_float(rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

// =============================================================================
// Tiled matmul kernel for prefill (BF16 weights, rhs transposed)
//
// Computes: out[batch, m, n] = lhs[batch, m, k] × rhs[batch, n, k]^T
//
// lhs: [batch, m, k] row-major
// rhs: [batch, n, k] row-major (virtually transposed: reads rhs[col*k + kk])
// out: [batch, m, n]
//
// Same tiling as FP8 kernel: TILE_M×TILE_N output tile, TILE_K reduction steps.
// =============================================================================

#define BF16_TILE_M 16
#define BF16_TILE_N 16
#define BF16_TILE_K 32

template <typename T>
__global__ void dotcache_qwen35_matmul_rhs_transposed_tiled_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const T* __restrict__ lhs,   // [batch, m, k]
    const T* __restrict__ rhs,   // [batch, n, k] (virtually transposed)
    T* __restrict__ out          // [batch, m, n]
) {
    const int tx = threadIdx.x % BF16_TILE_N;
    const int ty = threadIdx.x / BF16_TILE_N;

    const int batch_idx = blockIdx.z;
    const int tile_row = blockIdx.y * BF16_TILE_M;
    const int tile_col = blockIdx.x * BF16_TILE_N;

    const int row = tile_row + ty;
    const int col = tile_col + tx;

    const size_t lhs_base = static_cast<size_t>(batch_idx) * m * k;
    const size_t rhs_base = static_cast<size_t>(batch_idx) * n * k;

    __shared__ float s_lhs[BF16_TILE_M][BF16_TILE_K];
    __shared__ float s_rhs[BF16_TILE_N][BF16_TILE_K];

    float acc = 0.0f;

    for (int kk_base = 0; kk_base < k; kk_base += BF16_TILE_K) {
        // Cooperative load LHS tile [TILE_M, TILE_K]
        for (int i = threadIdx.x; i < BF16_TILE_M * BF16_TILE_K; i += blockDim.x) {
            int lr = i / BF16_TILE_K;
            int lc = i % BF16_TILE_K;
            int gr = tile_row + lr;
            int gc = kk_base + lc;
            if (gr < m && gc < k)
                s_lhs[lr][lc] = dotcache_qwen35_to_float(lhs[lhs_base + static_cast<size_t>(gr) * k + gc]);
            else
                s_lhs[lr][lc] = 0.0f;
        }

        // Cooperative load RHS tile [TILE_N, TILE_K] (rhs is [n, k], virtual transpose)
        for (int i = threadIdx.x; i < BF16_TILE_N * BF16_TILE_K; i += blockDim.x) {
            int rr = i / BF16_TILE_K;
            int rc = i % BF16_TILE_K;
            int gn = tile_col + rr;
            int gk = kk_base + rc;
            if (gn < n && gk < k)
                s_rhs[rr][rc] = dotcache_qwen35_to_float(rhs[rhs_base + static_cast<size_t>(gn) * k + gk]);
            else
                s_rhs[rr][rc] = 0.0f;
        }

        __syncthreads();

        if (row < m && col < n) {
            for (int kk = 0; kk < BF16_TILE_K; ++kk)
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        size_t out_idx = static_cast<size_t>(batch_idx) * m * n +
                         static_cast<size_t>(row) * n + col;
        out[out_idx] = dotcache_qwen35_from_float<T>(acc);
    }
}

template <typename T>
__global__ void dotcache_qwen35_batched_matmul_view_kernel(
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
        acc += dotcache_qwen35_to_float(
                   lhs[lhs_base + static_cast<size_t>(row) * static_cast<size_t>(lhs_row_stride) +
                       static_cast<size_t>(kk) * static_cast<size_t>(lhs_k_stride)]) *
               dotcache_qwen35_to_float(
                   rhs[rhs_base + static_cast<size_t>(kk) * static_cast<size_t>(rhs_k_stride) +
                       static_cast<size_t>(col) * static_cast<size_t>(rhs_col_stride)]);
    }
    out[idx] = dotcache_qwen35_from_float<T>(acc);
}

// =============================================================================
// FP8 dequant matmul kernel for prefill (tiled version)
//
// Computes: out[batch, m, n] = lhs[batch, m, k] × dequant(rhs_fp8[batch, n, k])^T
//
// lhs: BF16 activations [batch, m, k] (row-major)
// rhs: FP8 E4M3 weights [batch, n, k] (row-major, virtually transposed)
// scale: BF16 scale_inv [n/block_size, k/block_size] (same for all batches)
// out: BF16 [batch, m, n]
//
// Tiled approach: each thread block computes a TILE_M×TILE_N output tile.
// Threads cooperate to load TILE_K slices into shared memory.
// =============================================================================

#define FP8_TILE_M 16
#define FP8_TILE_N 16
#define FP8_TILE_K 32

template <typename T>
__global__ void dotcache_qwen35_matmul_fp8_dequant_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const T* __restrict__ lhs,          // [batch, m, k] BF16
    const uint8_t* __restrict__ rhs,    // [batch, n, k] FP8 (virtually transposed)
    const T* __restrict__ scale,        // [n/block_size, k/block_size] BF16
    int block_size,
    T* __restrict__ out                 // [batch, m, n] BF16
) {
    // Block computes output tile [TILE_M, TILE_N]
    // Thread layout: 16×16 = 256 threads, each computes one output element
    const int tx = threadIdx.x % FP8_TILE_N;
    const int ty = threadIdx.x / FP8_TILE_N;

    const int batch_idx = blockIdx.z;
    const int tile_row = blockIdx.y * FP8_TILE_M;
    const int tile_col = blockIdx.x * FP8_TILE_N;

    const int row = tile_row + ty;
    const int col = tile_col + tx;

    const size_t lhs_base = static_cast<size_t>(batch_idx) * m * k;
    const size_t rhs_base = static_cast<size_t>(batch_idx) * n * k;

    const int scale_cols = (k + block_size - 1) / block_size;

    // Shared memory for tiles
    __shared__ float s_lhs[FP8_TILE_M][FP8_TILE_K];
    __shared__ float s_rhs[FP8_TILE_N][FP8_TILE_K];

    float acc = 0.0f;

    // Iterate over K dimension in TILE_K chunks
    for (int kk_base = 0; kk_base < k; kk_base += FP8_TILE_K) {
        // Cooperative load: each thread loads multiple elements to fill the tiles
        // LHS tile: [TILE_M, TILE_K] = 16×32 = 512 elements, 256 threads → 2 loads each
        for (int i = threadIdx.x; i < FP8_TILE_M * FP8_TILE_K; i += blockDim.x) {
            int lr = i / FP8_TILE_K;
            int lc = i % FP8_TILE_K;
            int global_r = tile_row + lr;
            int global_c = kk_base + lc;
            if (global_r < m && global_c < k) {
                s_lhs[lr][lc] = dotcache_qwen35_to_float(
                    lhs[lhs_base + static_cast<size_t>(global_r) * k + global_c]);
            } else {
                s_lhs[lr][lc] = 0.0f;
            }
        }

        // RHS tile: [TILE_N, TILE_K] = 16×32 = 512 elements, 256 threads → 2 loads each
        // rhs is [n, k], reading element [col, kk] for virtual transpose
        for (int i = threadIdx.x; i < FP8_TILE_N * FP8_TILE_K; i += blockDim.x) {
            int rr = i / FP8_TILE_K;  // which output column (= which weight row)
            int rc = i % FP8_TILE_K;  // which k position
            int global_n = tile_col + rr;
            int global_k = kk_base + rc;
            if (global_n < n && global_k < k) {
                uint8_t fp8_byte = rhs[rhs_base + static_cast<size_t>(global_n) * k + global_k];
                float fp8_val = fp8_e4m3_to_float(fp8_byte);
                int sr = global_n / block_size;
                int sc = global_k / block_size;
                float s = dotcache_qwen35_to_float(scale[sr * scale_cols + sc]);
                s_rhs[rr][rc] = bf16_round_rne_f32_finite((fp8_val * s));
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot products from shared memory
        if (row < m && col < n) {
            for (int kk = 0; kk < FP8_TILE_K; ++kk) {
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        size_t out_idx = static_cast<size_t>(batch_idx) * m * n +
                         static_cast<size_t>(row) * n + col;
        out[out_idx] = dotcache_qwen35_from_float<T>(acc);
    }
}

// INT4 dequant tiled matmul for prefill.
// out [batch, m, n] = lhs [batch, m, k] × dequant(rhs_int4 [batch, n, k/2])^T
// rhs_int4 is packed INT4 (2 weights per byte, low nibble = even k index).
// scale/zero are [n/group_size, k/group_size] BF16.
#define INT4_TILE_M 16
#define INT4_TILE_N 16
#define INT4_TILE_K 32
template <typename T>
__global__ void dotcache_qwen35_matmul_int4_dequant_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const T* __restrict__ lhs,             // [batch, m, k] BF16
    const uint8_t* __restrict__ rhs,       // [batch, n, k/2] packed INT4
    const T* __restrict__ scale,           // [n/group_size, k/group_size] BF16
    const T* __restrict__ zero,            // [n/group_size, k/group_size] BF16
    int group_size,
    T* __restrict__ out                    // [batch, m, n] BF16
) {
    const int tx = threadIdx.x % INT4_TILE_N;
    const int ty = threadIdx.x / INT4_TILE_N;

    const int batch_idx = blockIdx.z;
    const int tile_row = blockIdx.y * INT4_TILE_M;
    const int tile_col = blockIdx.x * INT4_TILE_N;

    const int row = tile_row + ty;
    const int col = tile_col + tx;

    const size_t lhs_base = static_cast<size_t>(batch_idx) * m * k;
    const int k_packed = k / 2;
    const size_t rhs_base = static_cast<size_t>(batch_idx) * n * k_packed;

    const int scale_cols = (k + group_size - 1) / group_size;

    __shared__ float s_lhs[INT4_TILE_M][INT4_TILE_K];
    __shared__ float s_rhs[INT4_TILE_N][INT4_TILE_K];

    float acc = 0.0f;

    for (int kk_base = 0; kk_base < k; kk_base += INT4_TILE_K) {
        // Cooperative load LHS tile
        for (int i = threadIdx.x; i < INT4_TILE_M * INT4_TILE_K; i += blockDim.x) {
            int lr = i / INT4_TILE_K;
            int lc = i % INT4_TILE_K;
            int global_r = tile_row + lr;
            int global_c = kk_base + lc;
            if (global_r < m && global_c < k) {
                s_lhs[lr][lc] = dotcache_qwen35_to_float(
                    lhs[lhs_base + static_cast<size_t>(global_r) * k + global_c]);
            } else {
                s_lhs[lr][lc] = 0.0f;
            }
        }

        // Cooperative load RHS tile with INT4 dequant
        for (int i = threadIdx.x; i < INT4_TILE_N * INT4_TILE_K; i += blockDim.x) {
            int rr = i / INT4_TILE_K;  // which weight row (output column)
            int rc = i % INT4_TILE_K;  // which k position
            int global_n = tile_col + rr;
            int global_k = kk_base + rc;
            if (global_n < n && global_k < k) {
                // Unpack INT4 nibble
                int byte_idx = global_k / 2;
                uint8_t packed_byte = rhs[rhs_base + static_cast<size_t>(global_n) * k_packed + byte_idx];
                int nibble = (global_k & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
                // Dequant: (nibble - zero) * scale
                int sr = global_n / group_size;
                int sc = global_k / group_size;
                int si = sr * scale_cols + sc;
                float s = dotcache_qwen35_to_float(scale[si]);
                float z = dotcache_qwen35_to_float(zero[si]);
                // Round through BF16 so prefill matches the decode megakernel's
                // dequant path and the Python GPTQ reference (`bf16(q*s - zf*s)`).
                s_rhs[rr][rc] = bf16_round_rne_f32_finite(static_cast<float>(nibble) * s - z * s);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            for (int kk = 0; kk < INT4_TILE_K; ++kk) {
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        size_t out_idx = static_cast<size_t>(batch_idx) * m * n +
                         static_cast<size_t>(row) * n + col;
        out[out_idx] = dotcache_qwen35_from_float<T>(acc);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_full_scan_pack_kernel(
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

    const float exp_g_t = dotcache_qwen35_to_float(exp_g_scan[row]);
    const float exp_g_last = dotcache_qwen35_to_float(exp_g_scan[bh * num_chunks * chunk_size + chunk * chunk_size + (chunk_size - 1)]);
    const float chunk_decay = exp_g_t != 0.0f ? (exp_g_last / exp_g_t) : 0.0f;

    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[packed_row + k_idx] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(key_scan[scan_row + k_idx]) * chunk_decay
        );
        out[packed_row + k_head_dim + k_idx] = k_cumdecay_scan[scan_row + k_idx];
        out[packed_row + 2 * k_head_dim + k_idx] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(query_scan[scan_row + k_idx]) * exp_g_t
        );
    }
    out[packed_row + 3 * k_head_dim] = dotcache_qwen35_from_float<T>(exp_g_last);
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void dotcache_qwen35_delta_full_scan_packed_impl(
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
        state[k_idx] = dotcache_qwen35_to_float(
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
                v_prime += dotcache_qwen35_to_float(packed_scan[row + k_head_dim + k_idx]) * state[k_idx];
                attn += dotcache_qwen35_to_float(packed_scan[row + 2 * k_head_dim + k_idx]) * state[k_idx];
            }
            v_new[t] = dotcache_qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) - v_prime;
            attn_inter[t] = attn;
        }

        for (int t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const int row = chunk_local + t * chunk_size;
            for (int s = 0; s < chunk_size; ++s) {
                local += dotcache_qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                dotcache_qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = dotcache_qwen35_to_float(packed_scan[chunk_packed + 3 * k_head_dim]);
        for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (int t = 0; t < chunk_size; ++t) {
                const int row = chunk_packed + t * packed_width;
                update += dotcache_qwen35_to_float(packed_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const int state_out = out_base + token_count * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_full_scan_packed_kernel(
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
    dotcache_qwen35_delta_full_scan_packed_impl(
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
__global__ void dotcache_qwen35_delta_full_scan_kernel(
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
    dotcache_qwen35_delta_full_scan_impl(
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
__device__ inline void dotcache_qwen35_delta_chunk_single_prefill_impl(
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
        raw_g[t] = dotcache_qwen35_to_float(g_raw[bh * token_stride_s + t]);
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
                dot += dotcache_qwen35_to_float(query[row_i_k + k_idx]) *
                    dotcache_qwen35_to_float(key[row_j_k + k_idx]);
            }
            const float local = dot * expf(prefix_g[i] - prefix_g[j]);
            out_i += local * dotcache_qwen35_to_float(value[bh * token_stride_v + j * v_head_dim + v_idx]);
        }
        out[out_base + i * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(out_i);
    }

    const float g_last = raw_g[chunk_size - 1];
    const int state_out = out_base + chunk_size * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float state = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            const float beta_t = dotcache_qwen35_to_float(beta[bh * token_stride_s + t]);
            state += dotcache_qwen35_to_float(key[bh * token_stride_k + t * k_head_dim + k_idx]) *
                expf(g_last - raw_g[t]) *
                dotcache_qwen35_to_float(value[bh * token_stride_v + t * v_head_dim + v_idx]);
        }
        out[state_out + k_idx * v_head_dim + v_idx] = dotcache_qwen35_from_float<T>(state);
    }
}

template <typename T>
__global__ void dotcache_qwen35_delta_chunk_single_prefill_kernel(
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
    dotcache_qwen35_delta_chunk_single_prefill_impl(
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
__global__ void dotcache_qwen35_l2norm_kernel(
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
        const float x = dotcache_qwen35_to_float(row_in[col]);
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
        row_out[col] = dotcache_qwen35_from_float<T>(
            dotcache_qwen35_to_float(row_in[col]) * shared_inv_norm
        );
    }
}

template <typename T>
__global__ void dotcache_qwen35_value_decay_kernel(
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
    const float a_val = dotcache_qwen35_to_float(a[idx]);
    const float bias = dotcache_qwen35_to_float(dt_bias[head]);
    const float decay = dotcache_qwen35_to_float(a_log_exp[head]);
    const float softplus = dotcache_qwen35_softplus_fast(a_val + bias);
    out[idx] = dotcache_qwen35_from_float<T>(-softplus * decay);
}

template <typename T, bool ADD_UNIT_OFFSET>
__global__ void dotcache_qwen35_rms_norm_kernel(
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
        const float x = dotcache_qwen35_to_float(row_in[col]);
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
        const float weight_val = dotcache_qwen35_to_float(weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        const float x = dotcache_qwen35_to_float(row_in[col]);
        row_out[col] = dotcache_qwen35_from_float<T>(x * shared_inv_rms * weight_val);
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
__global__ void dotcache_qwen35_fused_rms_norm_linear_kernel(
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
        const float x = dotcache_qwen35_to_float(hidden[col]);
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
        const float x = dotcache_qwen35_to_float(hidden[col]);
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + (ADD_UNIT_OFFSET ? 1.0f : 0.0f);
        shared_normed[col] = x * inv_rms * w;
    }
    __syncthreads();

    // Phase 2: Dot product of proj_weight[out_row, :] with shared_normed
    const T* w_row = proj_weight + static_cast<size_t>(out_row) * static_cast<size_t>(hidden_dim);

    float partial_dot = 0.0f;
    for (int col = tid; col < hidden_dim; col += block_size) {
        partial_dot += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
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
        out[out_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
    }
}

template <typename T>
__global__ void dotcache_qwen35_rms_norm_gated_kernel(
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
        const float x = dotcache_qwen35_to_float(row_hidden[col]);
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
        const float x = dotcache_qwen35_to_float(row_hidden[col]);
        const float gate_x = dotcache_qwen35_to_float(row_gate[col]);
    const float gate_silu = gate_x * dotcache_qwen35_sigmoid_fast(gate_x);
        const float weight_val = dotcache_qwen35_to_float(weight[col]);
        row_out[col] =
            dotcache_qwen35_from_float<T>(x * shared_inv_rms * weight_val * gate_silu);
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
__device__ inline float dotcache_qwen35_wave_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__global__ void dotcache_qwen35_mlp_decode_megakernel(
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
        shared_hidden[col] = dotcache_qwen35_to_float(hidden_in[col]);
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
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + 1.0f;  // add_unit_offset
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
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
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
__global__ void dotcache_qwen35_mlp_swiglu_kernel(
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
__global__ void dotcache_qwen35_mlp_down_proj_kernel(
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
            partial += dotcache_qwen35_to_float(w_row[col]) * swiglu_scratch[col];
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
            hidden_out[my_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
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
__global__ void dotcache_qwen35_standalone_matvec_kernel(
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
    __shared__ float shared_input[4096];  // max in_dim we support
    for (int col = tid; col < in_dim; col += block_size) {
        shared_input[col] = dotcache_qwen35_to_float(input[col]);
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
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_input[col];
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
            output[my_row] = dotcache_qwen35_from_float<T>(shared_scratch[0]);
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
__global__ void dotcache_qwen35_norm_multi_proj_kernel(
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
        shared_hidden[col] = dotcache_qwen35_to_float(hidden_in[col]);
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
        const float w = dotcache_qwen35_to_float(norm_weight[col]) + 1.0f;
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
            partial += dotcache_qwen35_to_float(w_row[col]) * shared_normed[col];
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
//
// Optimization: wave-per-row matvec with vectorized loads.
// Each wave (32 threads) handles one output row. 8 waves per block process
// 8 rows in parallel. Wave shuffle reduction replaces LDS tree reduction,
// eliminating syncthreads from the hot matvec loop.
// =============================================================================

// Wave-level sum reduction using DPP/shuffle (Wave32 on RDNA3.5)
__device__ inline float wave_reduce_sum_f32(float val) {
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// Packed 2×BF16 vector type for v_dot2_f32_bf16 instruction.
// On GFX11 (RDNA 3.0+), v_dot2_f32_bf16 computes:
//   acc += (a.x * b.x) + (a.y * b.y)  where a,b are packed BF16 pairs
// This does 2 multiply-adds per instruction vs 1 for scalar FMA.
struct v2bf16 { __nv_bfloat16 x; __nv_bfloat16 y; };

// Packed 2×BF16 dot product accumulating into F32.
// Compiles to a single v_dot2_f32_bf16 instruction on gfx1100+.
__device__ inline float dot2_bf16_f32(v2bf16 a, v2bf16 b, float acc) {
    return acc + __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(a.y) * __bfloat162float(b.y);
}

__device__ inline void grid_barrier(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* counter_v = barrier_counter;
        volatile unsigned int* flag_v = barrier_flag;
        unsigned int phase = *flag_v;
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            *counter_v = 0u;
            __threadfence();
            *flag_v = phase + 1;
        } else {
            while (*flag_v == phase) {}
            __threadfence();
        }
    }
    __syncthreads();
}

// Grid barrier that also resets a work-stealing counter (fuses barrier + counter reset)
__device__ inline void grid_barrier_reset_counter(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks,
    unsigned int* counter_to_reset
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* counter_v = barrier_counter;
        volatile unsigned int* flag_v = barrier_flag;
        volatile unsigned int* reset_v = counter_to_reset;
        unsigned int phase = *flag_v;
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            *reset_v = 0u;
            *counter_v = 0u;
            __threadfence();
            *flag_v = phase + 1;
        } else {
            while (*flag_v == phase) {}
            __threadfence();
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
    const int lane = tid % warpSize;
    const int wave = tid / warpSize;
    const int nwaves = bs / warpSize;
    float sq = 0.0f;
    for (int c = tid; c < dim; c += bs) sq += src[c] * src[c];
    float wsq = wave_reduce_sum_f32(sq);
    if (lane == 0) scratch[wave] = wsq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < nwaves; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    float inv = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
    for (int c = tid; c < dim; c += bs) {
        dst[c] = src[c] * inv * (dotcache_qwen35_to_float(weight[c]) + 1.0f);
    }
    __syncthreads();
}

// =============================================================================
// KV cache BF16→FP8 quantization kernel (for prefill and oracle state loading)
// =============================================================================
// Each block handles one (head, position) pair: 256 threads for head_dim=256.
// Input:  BF16 K or V tensor [num_kv_heads, seq_len, head_dim]
// Output: FP8 cache [num_kv_heads, max_T, head_dim] (U8) + scale [num_kv_heads, max_T] (F32)
template <typename T>
__global__ void quantize_kv_to_fp8_kernel(
    const T* __restrict__ src,       // [num_kv_heads, seq_len, head_dim] contiguous
    uint8_t* __restrict__ dst_fp8,   // [num_kv_heads, max_T, head_dim] strided cache
    float* __restrict__ dst_scale,   // [num_kv_heads, max_T]
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int max_T,                       // allocated T dimension of cache
    int pos_offset                   // starting position in cache
) {
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int h = block_idx / seq_len;
    const int t = block_idx % seq_len;
    if (h >= num_kv_heads || t >= seq_len) return;

    extern __shared__ float smem[];

    // Load value
    const size_t src_offset = static_cast<size_t>(h) * seq_len * head_dim +
                              static_cast<size_t>(t) * head_dim + tid;
    float val = (tid < head_dim) ? static_cast<float>(src[src_offset]) : 0.0f;

    // Absmax reduction
    smem[tid] = fabsf(val);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float scale = fmaxf(smem[0] / 448.0f, 1e-12f);
    float inv_scale = 1.0f / scale;

    // Write scale
    if (tid == 0)
        dst_scale[h * max_T + (pos_offset + t)] = scale;

    // Quantize and write FP8
    if (tid < head_dim) {
        const size_t dst_offset = static_cast<size_t>(h) * max_T * head_dim +
                                  static_cast<size_t>(pos_offset + t) * head_dim + tid;
        dst_fp8[dst_offset] = float_to_fp8_e4m3(val * inv_scale);
    }
}

template <typename T>
__global__ void dotcache_qwen35_persistent_decode_kernel(
    int num_layers,
    int hidden_dim,
    int intermediate_size,
    int seqlen_offset,
    const Qwen35DecodeLayerDesc* __restrict__ layers,
    T* __restrict__ hidden_io,
    float* __restrict__ workspace,
    unsigned int* __restrict__ counters,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_flag,
    const T* __restrict__ cos_table,   // [max_positions, rotary_dim/2] RoPE cos
    const T* __restrict__ sin_table,   // [max_positions, rotary_dim/2] RoPE sin
    int rotary_dim,                      // partial rotary dimension (64 for Qwen3.5)
    int proj_buf_floats,                 // max projection output buffer size
    int attn_scratch_floats,             // attention/recurrent scratch buffer size
    const Qwen35FP8ScaleDesc* __restrict__ fp8_scales,  // nullptr for BF16, else [num_layers]
    const KVCacheFp8Desc* __restrict__ kv_fp8,          // nullptr for BF16 KV, else [num_layers]
    int batch_size,                      // 1 for single-sequence, >1 for batched
    const BatchSeqDesc* __restrict__ batch_descs,  // nullptr for single, else [num_layers]
    const Qwen35INT4ScaleDesc* __restrict__ int4_scales  // nullptr for non-INT4, else [num_layers]
) __attribute__((launch_bounds(256, 1))) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;
    const int B = batch_size;

    // Workspace layout (F32 unless noted).
    // Each section is multiplied by batch_size. Per-batch offset: section + b * section_size.
    // Total floats per batch item:
    const int per_batch = hidden_dim + hidden_dim + intermediate_size * 2
                        + hidden_dim + hidden_dim + proj_buf_floats + attn_scratch_floats;
    float* hidden_f32   = workspace;                                     // [B * hidden_dim]
    float* normed       = hidden_f32 + B * hidden_dim;                   // [B * hidden_dim]
    float* gate_up      = normed + B * hidden_dim;                       // [B * intermediate_size * 2]
    float* mlp_out      = gate_up + B * intermediate_size * 2;           // [B * hidden_dim]
    float* token_out    = mlp_out + B * hidden_dim;                      // [B * hidden_dim]
    float* proj_buf     = token_out + B * hidden_dim;                    // [B * proj_buf_floats]
    float* attn_scratch = proj_buf + B * proj_buf_floats;                // [B * attn_scratch_floats]

    extern __shared__ float lds[];
    // LDS layout: lds[0..bs-1] = reduction scratch, lds[bs..] = input vector cache (F32)
    float* lds_input = lds + bs;

    // FP8 E4M3 → F32 lookup table in LDS (256 entries = 1KB).
    // Placed after the input cache area. Populated once, used for all FP8 dequant.
    // LDS is sized by the bridge as: block_size + max(B*hidden_dim, intermediate_size).
    // The LUT sits after the input cache region.
    const int lds_input_size = (B * hidden_dim > intermediate_size) ? B * hidden_dim : intermediate_size;
    float* fp8_lut = lds + bs + lds_input_size;

    // Populate FP8 LUT: thread i fills entry i (256 threads → 256 entries, one pass).
    // Required whenever ANY FP8 dequant runs — weight dequant (fp8_scales) OR
    // KV-cache dequant in attention (kv_fp8). Previously gated on fp8_scales
    // alone, which left the LUT uninitialized for --kv-fp8 without
    // --fp8-runtime (and for --int4 --kv-fp8), producing gibberish from step 2
    // onward as attention read garbage floats for K/V.
    if (fp8_scales != nullptr || kv_fp8 != nullptr) {
        fp8_lut[tid] = fp8_e4m3_to_float(static_cast<uint8_t>(tid));
    }
    __syncthreads();

    // Load hidden BF16 → F32 for all batch items
    for (int b = 0; b < B; b++) {
        for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
            hidden_f32[b * hidden_dim + c] = dotcache_qwen35_to_float(hidden_io[b * hidden_dim + c]);
        }
    }
    grid_barrier(barrier_counter, barrier_flag, nb);

    for (int layer = 0; layer < num_layers; ++layer) {
        const Qwen35DecodeLayerDesc& L = layers[layer];

        // Match the component path's BF16 hidden-state boundaries: each layer
        // starts from BF16-hidden activations rather than carrying full F32
        // residual sums across layers inside the persistent kernel.
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; b++) {
                for (int c = tid; c < hidden_dim; c += bs) {
                    hidden_f32[b * hidden_dim + c] =
                        bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + c]);
                }
                __syncthreads();
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // === Input RMSNorm (all blocks cooperate) ===
        // Phase 1: each block computes partial sum-of-squares, atomicAdd to global accum
        {
            const T* norm_w = static_cast<const T*>(L.input_norm_w);
            const float norm_eps = L.input_norm_eps;
            if (blockIdx.x == 0) {
                for (int b = 0; b < B; b++) {
                    const float* src = hidden_f32 + b * hidden_dim;
                    float* dst = normed + b * hidden_dim;
                    float partial_sq = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs) {
                        partial_sq += src[c] * src[c];
                    }
                    lds[tid] = partial_sq;
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    float inv_rms = rsqrtf(lds[0] / static_cast<float>(hidden_dim) + norm_eps);
                    for (int c = tid; c < hidden_dim; c += bs) {
                        dst[c] = dotcache_qwen35_from_float<T>(
                            src[c] * inv_rms * (dotcache_qwen35_to_float(norm_w[c]) + 1.0f));
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // All blocks: cache B normed vectors in LDS for fast projection reads
        for (int b = 0; b < B; b++)
            for (int c = tid; c < hidden_dim; c += bs)
                lds_input[b * hidden_dim + c] = normed[b * hidden_dim + c];
        __syncthreads();

        // === Token mixer: projections + core ===
        if (L.layer_type == 1) {
            // ---- FULL ATTENTION ----
            // Step A: Q/K/V projections via work-stealing
            // Q: [q_out_dim, hidden_dim], K: [k_out_dim, hidden_dim], V: same as K
            if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
            grid_barrier(barrier_counter, barrier_flag, nb);

            const int total_proj = L.q_out_dim + L.k_out_dim + L.k_out_dim;
            {
                const int lane_p = tid % warpSize;
                for (;;) {
                    unsigned int sr;
                    if (lane_p == 0) sr = atomicAdd(&counters[0], 1u);
                    sr = __shfl(sr, 0);
                    if (sr >= static_cast<unsigned int>(total_proj)) break;

                    const void* w_raw;
                    const void* w_scale = nullptr;
                    const void* w_i4_scale = nullptr;
                    const void* w_i4_zero = nullptr;
                    int row;
                    if (sr < static_cast<unsigned int>(L.q_out_dim)) {
                        w_raw = L.q_proj_w;
                        row = sr;
                        if (fp8_scales) w_scale = fp8_scales[layer].q_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].q_proj_scale; w_i4_zero = int4_scales[layer].q_proj_zero; }
                    } else if (sr < static_cast<unsigned int>(L.q_out_dim + L.k_out_dim)) {
                        w_raw = L.k_proj_w;
                        row = sr - L.q_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].k_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].k_proj_scale; w_i4_zero = int4_scales[layer].k_proj_zero; }
                    } else {
                        w_raw = L.v_proj_w;
                        row = sr - L.q_out_dim - L.k_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].v_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].v_proj_scale; w_i4_zero = int4_scales[layer].v_proj_zero; }
                    }

                    float p[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) p[b] = 0.0f;
                    if (int4_scales != nullptr && w_i4_scale != nullptr) {
                        // INT4 dequant: load 4 packed bytes (8 weights) at a time
                        const int gsz = int4_scales[layer].group_size;
                        const int byte_cols = hidden_dim / 2;  // packed width
                        const uint8_t* i4_row = static_cast<const uint8_t*>(w_raw) + static_cast<size_t>(row) * byte_cols;
                        const hip_bfloat16* scales_p = static_cast<const hip_bfloat16*>(w_i4_scale);
                        const hip_bfloat16* zeros_p = static_cast<const hip_bfloat16*>(w_i4_zero);
                        const int scale_row = row / gsz;
                        const int scale_cols = (hidden_dim + gsz - 1) / gsz;
                        const int vd8 = hidden_dim & ~7;
                        for (int c = lane_p * 8; c < vd8; c += warpSize * 8) {
                            uint32_t packed = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                            float w[8];
                            int4_dequant_8(packed, scales_p, zeros_p, scale_row, c, scale_cols, gsz, w);
                            for (int b = 0; b < B; b++) {
                                const float* inp = lds_input + b * hidden_dim + c;
                                p[b] += w[0]*inp[0] + w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3]
                                      + w[4]*inp[4] + w[5]*inp[5] + w[6]*inp[6] + w[7]*inp[7];
                            }
                        }
                        for (int c = vd8 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = int4_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, row, c, hidden_dim, gsz);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    } else if (fp8_scales != nullptr && w_scale != nullptr) {
                        // Vectorized FP8 dequant: load 4 bytes at a time, LUT decode, scale
                        const uint8_t* fp8_row = static_cast<const uint8_t*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const hip_bfloat16* scales = static_cast<const hip_bfloat16*>(w_scale);
                        const int bsz = fp8_scales[layer].block_size;
                        const int scale_row = row / bsz;
                        const int scale_cols = (hidden_dim + bsz - 1) / bsz;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = lane_p * 4; c < vd4; c += warpSize * 4) {
                            uint32_t packed = *reinterpret_cast<const uint32_t*>(&fp8_row[c]);
                            float w0 = fp8_lut[packed & 0xFF];
                            float w1 = fp8_lut[(packed >> 8) & 0xFF];
                            float w2 = fp8_lut[(packed >> 16) & 0xFF];
                            float w3 = fp8_lut[(packed >> 24) & 0xFF];
                            const int sb = scale_row * scale_cols;
                            w0 = bf16_round_rne_f32_finite((w0 * static_cast<float>(scales[sb + c / bsz])));
                            w1 = bf16_round_rne_f32_finite((w1 * static_cast<float>(scales[sb + (c+1) / bsz])));
                            w2 = bf16_round_rne_f32_finite((w2 * static_cast<float>(scales[sb + (c+2) / bsz])));
                            w3 = bf16_round_rne_f32_finite((w3 * static_cast<float>(scales[sb + (c+3) / bsz])));
                            for (int b = 0; b < B; b++) {
                                const float* inp = lds_input + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = fp8_dequant_weight_lut(w_raw, w_scale, row, c, hidden_dim, bsz, fp8_lut);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    } else {
                        const T* wr = static_cast<const T*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = lane_p * 4; c < vd4; c += warpSize * 4) {
                            float w0 = dotcache_qwen35_to_float(wr[c]);
                            float w1 = dotcache_qwen35_to_float(wr[c+1]);
                            float w2 = dotcache_qwen35_to_float(wr[c+2]);
                            float w3 = dotcache_qwen35_to_float(wr[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = lds_input + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = dotcache_qwen35_to_float(wr[c]);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    }
                    for (int b = 0; b < B; b++) {
                        float result = wave_reduce_sum_f32(p[b]);
                        if (lane_p == 0)
                            proj_buf[b * proj_buf_floats + sr] = bf16_round_rne_f32_finite(result);
                    }
                }
            }
            __syncthreads();
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step B-G: QK-norm, RoPE, KV cache, attention, gating, o_proj
            // Block 0 handles per-head sequential ops (per-sequence loop).
            // O_proj uses all blocks via work-stealing (per-sequence loop).
            {
                const int hd = L.attn_head_dim;       // 256
                const int nh = L.attn_num_heads;       // 8
                const int nkv = L.attn_num_kv_heads;   // 2
                const int kv_groups = nh / nkv;         // 4
                const float scale = 1.0f / sqrtf(static_cast<float>(hd));
                const int rot_dim = hd / 4;             // 64 (partial_rotary_factor=0.25)
                const int attn_size = nh * hd;          // 2048

            for (int b = 0; b < B; b++) {
                // Per-sequence state: read from batch_descs when batched, else from L/seqlen_offset
                const int seq_off_b  = batch_descs ? batch_descs[layer].seqlen_offset[b] : seqlen_offset;
                void* kv_k_b         = batch_descs ? batch_descs[layer].kv_cache_k[b]    : L.kv_cache_k;
                void* kv_v_b         = batch_descs ? batch_descs[layer].kv_cache_v[b]    : L.kv_cache_v;
                const int kv_len_b   = (batch_descs ? batch_descs[layer].kv_len[b]       : L.kv_len) + 1;
                const int kv_max_b   = batch_descs ? batch_descs[layer].kv_max_t[b]      : L.kv_max_t;
                void* kv_shadow_k_b  = batch_descs ? batch_descs[layer].kv_shadow_k[b]   : L.kv_shadow_k;
                void* kv_shadow_v_b  = batch_descs ? batch_descs[layer].kv_shadow_v[b]   : L.kv_shadow_v;
                const int kv_shadow_start_b = batch_descs ? batch_descs[layer].kv_shadow_start[b] : L.kv_shadow_start;
                void* kv_scale_k_b   = (batch_descs && kv_fp8) ? batch_descs[layer].kv_scale_k[b] : (kv_fp8 ? kv_fp8[layer].kv_scale_k : nullptr);
                void* kv_scale_v_b   = (batch_descs && kv_fp8) ? batch_descs[layer].kv_scale_v[b] : (kv_fp8 ? kv_fp8[layer].kv_scale_v : nullptr);

                float* proj_b = proj_buf + b * proj_buf_floats;

                if (blockIdx.x == 0) {
                float* q_f32 = proj_b;
                float* k_f32 = proj_b + L.q_out_dim;
                float* v_f32 = proj_b + L.q_out_dim + L.k_out_dim;

                // Step B: QK-norm — RMSNorm per head on head_dim
                // Wave-level reduce + cross-wave LDS reduce (2 syncthreads vs 8)
                {
                    const int norm_lane = tid % warpSize;
                    const int norm_wave = tid / warpSize;
                    const int norm_nwaves = bs / warpSize;

                    for (int h = 0; h < nh; ++h) {
                        float* qh = q_f32 + h * hd * 2;
                        float sq = 0.0f;
                        for (int d = tid; d < hd; d += bs) sq += qh[d] * qh[d];
                        float wsq = wave_reduce_sum_f32(sq);
                        if (norm_lane == 0) lds[norm_wave] = wsq;
                        __syncthreads();
                        if (tid == 0) {
                            float total = 0.0f;
                            for (int w = 0; w < norm_nwaves; ++w) total += lds[w];
                            lds[0] = total;
                        }
                        __syncthreads();
                        float inv = rsqrtf(lds[0] / static_cast<float>(hd) + 1e-6f);
                        const T* qnw = static_cast<const T*>(L.q_norm_w);
                        for (int d = tid; d < hd; d += bs) {
                            qh[d] = bf16_round_rne_f32_finite(
                                qh[d] * inv * (dotcache_qwen35_to_float(qnw[d]) + 1.0f));
                        }
                        __syncthreads();
                    }
                    for (int h = 0; h < nkv; ++h) {
                        float* kh = k_f32 + h * hd;
                        float sq = 0.0f;
                        for (int d = tid; d < hd; d += bs) sq += kh[d] * kh[d];
                        float wsq = wave_reduce_sum_f32(sq);
                        if (norm_lane == 0) lds[norm_wave] = wsq;
                        __syncthreads();
                        if (tid == 0) {
                            float total = 0.0f;
                            for (int w = 0; w < norm_nwaves; ++w) total += lds[w];
                            lds[0] = total;
                        }
                        __syncthreads();
                        float inv = rsqrtf(lds[0] / static_cast<float>(hd) + 1e-6f);
                        const T* knw = static_cast<const T*>(L.k_norm_w);
                        for (int d = tid; d < hd; d += bs) {
                            kh[d] = bf16_round_rne_f32_finite(
                                kh[d] * inv * (dotcache_qwen35_to_float(knw[d]) + 1.0f));
                        }
                        __syncthreads();
                    }
                }

                // Step C: RoPE (partial, first rot_dim dims of head_dim)
                if (cos_table != nullptr && rot_dim > 0) {
                    const int half_rot = rot_dim / 2;
                    const size_t cos_off =
                        static_cast<size_t>(seq_off_b) * half_rot;

                    if (tid < nh * half_rot) {
                        const int h = tid / half_rot;
                        const int i = tid % half_rot;
                        float* qh = q_f32 + h * hd * 2;
                        float c = dotcache_qwen35_to_float(cos_table[cos_off + i]);
                        float s = dotcache_qwen35_to_float(sin_table[cos_off + i]);
                        float x0 = qh[i];
                        float x1 = qh[half_rot + i];
                        qh[i] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
                        qh[half_rot + i] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
                    }
                    __syncthreads();

                    if (tid < nkv * half_rot) {
                        const int h = tid / half_rot;
                        const int i = tid % half_rot;
                        float* kh = k_f32 + h * hd;
                        float c = dotcache_qwen35_to_float(cos_table[cos_off + i]);
                        float s = dotcache_qwen35_to_float(sin_table[cos_off + i]);
                        float x0 = kh[i];
                        float x1 = kh[half_rot + i];
                        kh[i] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
                        kh[half_rot + i] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
                    }
                    __syncthreads();
                }

                // Step D: KV cache append — write new K/V at seq_off_b
                {
                    const bool use_fp8_kv = (kv_scale_k_b != nullptr);
                    if (use_fp8_kv) {
                        uint8_t* fp8_k = static_cast<uint8_t*>(kv_k_b);
                        uint8_t* fp8_v = static_cast<uint8_t*>(kv_v_b);
                        T* shadow_k = static_cast<T*>(kv_shadow_k_b);
                        T* shadow_v = static_cast<T*>(kv_shadow_v_b);
                        float* scale_k_buf = static_cast<float*>(kv_scale_k_b);
                        float* scale_v_buf = static_cast<float*>(kv_scale_v_b);

                        for (int h = 0; h < nkv; ++h) {
                            float k_bf16 =
                                (tid < hd) ? bf16_round_rne_f32_finite(k_f32[h * hd + tid]) : 0.0f;
                            float my_k = (tid < hd) ? fabsf(k_bf16) : 0.0f;
                            lds[tid] = my_k;
                            __syncthreads();
                            for (int s = bs / 2; s > 0; s >>= 1) {
                                if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
                                __syncthreads();
                            }
                            float k_scale = fmaxf(lds[0] / 448.0f, 1e-12f);
                            float k_inv = 1.0f / k_scale;
                            if (tid == 0)
                                scale_k_buf[h * kv_max_b + seq_off_b] = k_scale;

                            float v_bf16 =
                                (tid < hd) ? bf16_round_rne_f32_finite(v_f32[h * hd + tid]) : 0.0f;
                            float my_v = (tid < hd) ? fabsf(v_bf16) : 0.0f;
                            lds[tid] = my_v;
                            __syncthreads();
                            for (int s = bs / 2; s > 0; s >>= 1) {
                                if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
                                __syncthreads();
                            }
                            float v_scale = fmaxf(lds[0] / 448.0f, 1e-12f);
                            float v_inv = 1.0f / v_scale;
                            if (tid == 0)
                                scale_v_buf[h * kv_max_b + seq_off_b] = v_scale;

                            for (int d = tid; d < hd; d += bs) {
                                const size_t offset =
                                    static_cast<size_t>(h) * kv_max_b * hd +
                                    static_cast<size_t>(seq_off_b) * hd + d;
                                float k_store =
                                    bf16_round_rne_f32_finite(k_f32[h * hd + d]);
                                float v_store =
                                    bf16_round_rne_f32_finite(v_f32[h * hd + d]);
                                fp8_k[offset] = float_to_fp8_e4m3(k_store * k_inv);
                                fp8_v[offset] = float_to_fp8_e4m3(v_store * v_inv);
                                if (shadow_k != nullptr && shadow_v != nullptr) {
                                    shadow_k[offset] = dotcache_qwen35_from_float<T>(k_store);
                                    shadow_v[offset] = dotcache_qwen35_from_float<T>(v_store);
                                }
                            }
                        }
                    } else {
                        T* cache_k = static_cast<T*>(kv_k_b);
                        T* cache_v = static_cast<T*>(kv_v_b);
                        for (int h = 0; h < nkv; ++h) {
                            for (int d = tid; d < hd; d += bs) {
                                const size_t offset =
                                    static_cast<size_t>(h) * kv_max_b * hd +
                                    static_cast<size_t>(seq_off_b) * hd + d;
                                cache_k[offset] = dotcache_qwen35_from_float<T>(k_f32[h * hd + d]);
                                cache_v[offset] = dotcache_qwen35_from_float<T>(v_f32[h * hd + d]);
                            }
                        }
                    }
                    __syncthreads();
                }

                // Save gate values before attention overwrites proj_buf.
                float* saved_q = attn_scratch + b * attn_scratch_floats;
                float* saved_gate = saved_q + nh * hd;
                float* saved_pre_gate = saved_gate + nh * hd;
                float* saved_scores = saved_pre_gate + nh * hd;
                for (int i = tid; i < nh * hd; i += bs) {
                    const int h = i / hd;
                    const int d = i % hd;
                    saved_q[i] = bf16_round_rne_f32_finite(q_f32[h * hd * 2 + d]);
                }
                for (int i = tid; i < nh * hd; i += bs) {
                    const int h = i / hd;
                    const int d = i % hd;
                    saved_gate[i] = bf16_round_rne_f32_finite(
                        q_f32[h * hd * 2 + hd + d]);
                }
                __syncthreads();

                // Step E: Parallel attention — Flash-style online softmax
                float* attn_flat = proj_b;  // reuse projection buffer

                const bool use_fp8_kv_attn = (kv_scale_k_b != nullptr);

                // Attention uses 256 threads for head_dim=256.
                // Wave-level reduce (32-wide) then cross-wave reduce via LDS (8 waves).
                const int attn_lane = tid % warpSize;
                const int attn_wave = tid / warpSize;
                const int attn_nwaves = bs / warpSize;  // 8

                if (use_fp8_kv_attn) {
                    const uint8_t* fp8_ck = static_cast<const uint8_t*>(kv_k_b);
                    const uint8_t* fp8_cv = static_cast<const uint8_t*>(kv_v_b);
                    const float* scale_k_buf = static_cast<const float*>(kv_scale_k_b);
                    const float* scale_v_buf = static_cast<const float*>(kv_scale_v_b);
                    for (int qh = 0; qh < nh; ++qh) {
                        const int kvh = qh / kv_groups;
                        const float* q_head = q_f32 + qh * hd * 2;
                        float my_acc = 0.0f;
                        float my_max = -1e30f;
                        float my_sum = 0.0f;
                        const float q_val = q_head[tid];
                        const size_t kv_head_base =
                            static_cast<size_t>(kvh) * kv_max_b * hd;

                        for (int t = 0; t < kv_len_b; ++t) {
                            const size_t pos_base = kv_head_base + static_cast<size_t>(t) * hd;
                            float k_val = 0.0f;
                            if (kv_shadow_k_b != nullptr && kv_shadow_v_b != nullptr &&
                                kv_shadow_start_b >= 0 && t >= kv_shadow_start_b) {
                                const T* shadow_k = static_cast<const T*>(kv_shadow_k_b);
                                k_val = dotcache_qwen35_to_float(shadow_k[pos_base + tid]);
                            } else if (t == seq_off_b) {
                                k_val = bf16_round_rne_f32_finite(k_f32[kvh * hd + tid]);
                            } else {
                                k_val = bf16_round_rne_f32_finite(
                                    fp8_lut[fp8_ck[pos_base + tid]] *
                                    scale_k_buf[kvh * kv_max_b + t]);
                            }
                            float partial = q_val * k_val;
                            float wave_sum = wave_reduce_sum_f32(partial);
                            if (attn_lane == 0) lds[attn_wave] = wave_sum;
                            __syncthreads();

                            float score_val = 0.0f;
                            if (tid == 0) {
                                float total = 0.0f;
                                for (int w = 0; w < attn_nwaves; ++w) total += lds[w];
                                lds[0] = total;
                                saved_scores[qh * kv_max_b + t] = total * scale;
                            }
                            __syncthreads();
                            score_val = lds[0] * scale;

                            float v_val = 0.0f;
                            if (kv_shadow_k_b != nullptr && kv_shadow_v_b != nullptr &&
                                kv_shadow_start_b >= 0 && t >= kv_shadow_start_b) {
                                const T* shadow_v = static_cast<const T*>(kv_shadow_v_b);
                                v_val = dotcache_qwen35_to_float(shadow_v[pos_base + tid]);
                            } else if (t == seq_off_b) {
                                v_val = bf16_round_rne_f32_finite(v_f32[kvh * hd + tid]);
                            } else {
                                v_val = bf16_round_rne_f32_finite(
                                    fp8_lut[fp8_cv[pos_base + tid]] *
                                    scale_v_buf[kvh * kv_max_b + t]);
                            }

                            float old_max = my_max;
                            my_max = fmaxf(my_max, score_val);
                            float rescale = expf(old_max - my_max);
                            float w = expf(score_val - my_max);
                            my_acc = my_acc * rescale + w * v_val;
                            my_sum = my_sum * rescale + w;
                        }

                        attn_flat[qh * hd + tid] =
                            bf16_round_rne_f32_finite(
                                (my_sum > 0.0f) ? (my_acc / my_sum) : 0.0f);
                        __syncthreads();
                    }
                } else {
                    const T* ck = static_cast<const T*>(kv_k_b);
                    const T* cv = static_cast<const T*>(kv_v_b);

                    for (int qh = 0; qh < nh; ++qh) {
                        const int kvh = qh / kv_groups;
                        const float* q_head = q_f32 + qh * hd * 2;

                        float my_acc = 0.0f;
                        float my_max = -1e30f;
                        float my_sum = 0.0f;

                        const float q_val = q_head[tid];
                        const size_t kv_head_base =
                            static_cast<size_t>(kvh) * kv_max_b * hd;

                        for (int t = 0; t < kv_len_b; ++t) {
                            const size_t pos_base = kv_head_base +
                                static_cast<size_t>(t) * hd;

                            float partial = q_val *
                                dotcache_qwen35_to_float(ck[pos_base + tid]);
                            // Wave-level reduce then cross-wave via LDS
                            float wave_sum = wave_reduce_sum_f32(partial);
                            if (attn_lane == 0) lds[attn_wave] = wave_sum;
                            __syncthreads();
                            float score_val = 0.0f;
                            if (tid == 0) {
                                float total = 0.0f;
                                for (int w = 0; w < attn_nwaves; ++w) total += lds[w];
                                lds[0] = total;
                                saved_scores[qh * kv_max_b + t] = total * scale;
                            }
                            __syncthreads();
                            score_val = lds[0] * scale;

                            float v_val =
                                dotcache_qwen35_to_float(cv[pos_base + tid]);

                            float old_max = my_max;
                            my_max = fmaxf(my_max, score_val);
                            float rescale = expf(old_max - my_max);
                            float w = expf(score_val - my_max);
                            my_acc = my_acc * rescale + w * v_val;
                            my_sum = my_sum * rescale + w;
                        }

                        attn_flat[qh * hd + tid] =
                            bf16_round_rne_f32_finite(
                                (my_sum > 0.0f) ? (my_acc / my_sum) : 0.0f);
                        __syncthreads();
                    }
                }

                for (int i = tid; i < nh * hd; i += bs) {
                    saved_pre_gate[i] = attn_flat[i];
                }
                __syncthreads();

                // Step F: Gate
                for (int i = tid; i < nh * hd; i += bs) {
                    float gate_val = saved_gate[i];
                    float sigmoid_gate = dotcache_qwen35_sigmoid_fast(gate_val);
                    attn_flat[i] = bf16_round_rne_f32_finite(attn_flat[i] * sigmoid_gate);
                }
                __syncthreads();

                } // end if (blockIdx.x == 0)
                // Grid barrier: block 0 wrote attn_flat, all blocks need it for o_proj
                grid_barrier(barrier_counter, barrier_flag, nb);

                // Step G: o_proj [hidden_dim, attn_size] × attn_flat → hidden_f32 (fused residual)
                // Per-sequence: cache batch b's attn output in LDS, work-steal hidden_dim rows
                {
                    for (int c = tid; c < attn_size; c += bs)
                        lds_input[c] = proj_b[c];  // attn_flat = proj_b after attention
                    __syncthreads();

                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);

                    const int lane_o = tid % warpSize;
                    for (;;) {
                        unsigned int sr;
                        if (lane_o == 0) sr = atomicAdd(&counters[0], 1u);
                        sr = __shfl(sr, 0);
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        if (int4_scales != nullptr && int4_scales[layer].o_proj_scale != nullptr) {
                            const int gsz = int4_scales[layer].group_size;
                            if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < attn_size; ++c) {
                                        p += int4_dequant_scalar(
                                            L.o_proj_w,
                                            int4_scales[layer].o_proj_scale,
                                            int4_scales[layer].o_proj_zero,
                                            sr,
                                            c,
                                            attn_size,
                                            gsz) * lds_input[c];
                                    }
                                }
                            } else {
                                const int byte_cols = attn_size / 2;
                                const uint8_t* i4_row = static_cast<const uint8_t*>(L.o_proj_w) + static_cast<size_t>(sr) * byte_cols;
                                const hip_bfloat16* sc = static_cast<const hip_bfloat16*>(int4_scales[layer].o_proj_scale);
                                const hip_bfloat16* zr = static_cast<const hip_bfloat16*>(int4_scales[layer].o_proj_zero);
                                const int sr_g = sr / gsz;
                                const int sc_cols = (attn_size + gsz - 1) / gsz;
                                const int as8 = attn_size & ~7;
                                for (int c = lane_o * 8; c < as8; c += warpSize * 8) {
                                    uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                                    float w[8];
                                    int4_dequant_8(pk, sc, zr, sr_g, c, sc_cols, gsz, w);
                                    p += w[0]*lds_input[c] + w[1]*lds_input[c+1] + w[2]*lds_input[c+2] + w[3]*lds_input[c+3]
                                       + w[4]*lds_input[c+4] + w[5]*lds_input[c+5] + w[6]*lds_input[c+6] + w[7]*lds_input[c+7];
                                }
                                for (int c = as8 + lane_o; c < attn_size; c += warpSize)
                                    p += int4_dequant_scalar(L.o_proj_w, int4_scales[layer].o_proj_scale, int4_scales[layer].o_proj_zero, sr, c, attn_size, gsz) * lds_input[c];
                            }
                        } else if (fp8_scales != nullptr && fp8_scales[layer].o_proj_scale != nullptr) {
                            if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < attn_size; ++c) {
                                        p += fp8_dequant_weight_lut(
                                            L.o_proj_w,
                                            fp8_scales[layer].o_proj_scale,
                                            sr,
                                            c,
                                            attn_size,
                                            fp8_scales[layer].block_size,
                                            fp8_lut) * lds_input[c];
                                    }
                                }
                            } else {
                                for (int c = lane_o; c < attn_size; c += warpSize)
                                    p += fp8_dequant_weight_lut(L.o_proj_w, fp8_scales[layer].o_proj_scale, sr, c, attn_size, fp8_scales[layer].block_size, fp8_lut) * lds_input[c];
                            }
                        } else {
                            const T* wr = static_cast<const T*>(L.o_proj_w) + static_cast<size_t>(sr) * attn_size;
                            if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < attn_size; ++c) {
                                        p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                                    }
                                }
                            } else {
                                const int as4 = attn_size & ~3;
                                for (int c = lane_o * 4; c < as4; c += warpSize * 4) {
                                    p += dotcache_qwen35_to_float(wr[c])   * lds_input[c]
                                       + dotcache_qwen35_to_float(wr[c+1]) * lds_input[c+1]
                                       + dotcache_qwen35_to_float(wr[c+2]) * lds_input[c+2]
                                       + dotcache_qwen35_to_float(wr[c+3]) * lds_input[c+3];
                                }
                                for (int c = as4 + lane_o; c < attn_size; c += warpSize)
                                    p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                            }
                        }
                        float result = wave_reduce_sum_f32(p);
                        if (lane_o == 0) {
                            float add = bf16_round_rne_f32_finite(result);
                            hidden_f32[b * hidden_dim + sr] =
                                bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + sr] + add);
                        }
                    }
                }
                __syncthreads();
                grid_barrier(barrier_counter, barrier_flag, nb);
            } // end for (b) batch loop
            } // end full attention scope

        } else {
            // ---- LINEAR ATTENTION ----
            // Step A: qkv/z/b/a projections via work-stealing
            if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
            grid_barrier(barrier_counter, barrier_flag, nb);

            // b_proj and a_proj have linear_num_v_heads rows (not num_k_heads)
            const int nv_heads = L.linear_num_v_heads;
            const int total_proj = L.qkv_out_dim + L.z_out_dim + nv_heads + nv_heads;
            {
                for (;;) {
                    unsigned int sr;
                    if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                    __shared__ unsigned int shared_sr;
                    if (tid == 0) shared_sr = sr;
                    __syncthreads();
                    sr = shared_sr;
                    if (sr >= static_cast<unsigned int>(total_proj)) break;

                    const void* w_raw;
                    const void* w_scale = nullptr;
                    const void* w_i4_scale = nullptr;
                    const void* w_i4_zero = nullptr;
                    int row;
                    if (sr < static_cast<unsigned int>(L.qkv_out_dim)) {
                        w_raw = L.qkv_proj_w;
                        row = sr;
                        if (fp8_scales) w_scale = fp8_scales[layer].qkv_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].qkv_proj_scale; w_i4_zero = int4_scales[layer].qkv_proj_zero; }
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim)) {
                        w_raw = L.z_proj_w;
                        row = sr - L.qkv_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].z_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].z_proj_scale; w_i4_zero = int4_scales[layer].z_proj_zero; }
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim + nv_heads)) {
                        w_raw = L.b_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].b_proj_scale;
                        // b_proj: keep BF16 (only 16 rows, no INT4)
                    } else {
                        w_raw = L.a_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim - nv_heads;
                        if (fp8_scales) w_scale = fp8_scales[layer].a_proj_scale;
                        // a_proj: keep BF16 (only 16 rows, no INT4)
                    }

                    float p[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) p[b] = 0.0f;
                    if (int4_scales != nullptr && w_i4_scale != nullptr) {
                        // INT4 dequant path for qkv/z projections
                        const int gsz = int4_scales[layer].group_size;
                        const int byte_cols = hidden_dim / 2;
                        const uint8_t* i4_row = static_cast<const uint8_t*>(w_raw) + static_cast<size_t>(row) * byte_cols;
                        const hip_bfloat16* sc = static_cast<const hip_bfloat16*>(w_i4_scale);
                        const hip_bfloat16* zr = static_cast<const hip_bfloat16*>(w_i4_zero);
                        const int sr_g = row / gsz;
                        const int sc_cols = (hidden_dim + gsz - 1) / gsz;
                        const int vd8 = hidden_dim & ~7;
                        for (int c = tid * 8; c < vd8; c += bs * 8) {
                            uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                            float w[8];
                            int4_dequant_8(pk, sc, zr, sr_g, c, sc_cols, gsz, w);
                            for (int b = 0; b < B; b++) {
                                const float* inp = lds_input + b * hidden_dim + c;
                                p[b] += w[0]*inp[0] + w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3]
                                      + w[4]*inp[4] + w[5]*inp[5] + w[6]*inp[6] + w[7]*inp[7];
                            }
                        }
                        for (int c = vd8 + tid; c < hidden_dim; c += bs) {
                            float w = int4_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, row, c, hidden_dim, gsz);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    } else if (fp8_scales != nullptr && w_scale != nullptr) {
                        for (int c = tid; c < hidden_dim; c += bs) {
                            float w = fp8_dequant_weight_lut(w_raw, w_scale, row, c, hidden_dim, fp8_scales[layer].block_size, fp8_lut);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    } else {
                        const T* wr = static_cast<const T*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = tid * 4; c < vd4; c += bs * 4) {
                            float w0 = dotcache_qwen35_to_float(wr[c]);
                            float w1 = dotcache_qwen35_to_float(wr[c+1]);
                            float w2 = dotcache_qwen35_to_float(wr[c+2]);
                            float w3 = dotcache_qwen35_to_float(wr[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = lds_input + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + tid; c < hidden_dim; c += bs) {
                            float w = dotcache_qwen35_to_float(wr[c]);
                            for (int b = 0; b < B; b++)
                                p[b] += w * lds_input[b * hidden_dim + c];
                        }
                    }
                    for (int b = 0; b < B; b++) {
                        lds[tid] = p[b];
                        __syncthreads();
                        for (int stride = bs / 2; stride > 0; stride >>= 1) {
                            if (tid < stride) lds[tid] += lds[tid + stride];
                            __syncthreads();
                        }
                        if (tid == 0)
                            proj_buf[b * proj_buf_floats + sr] = bf16_round_rne_f32_finite(lds[0]);
                        __syncthreads();
                    }
                }
            }
            __syncthreads();
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step B-E: conv1d, recurrent state, gated norm, out_proj
            // Block 0 handles sequential recurrent operations (per-sequence loop).
            // O_proj uses all blocks via work-stealing (per-sequence loop).
            for (int b = 0; b < B; b++) {
            // Per-sequence state
            void* conv_b = batch_descs ? batch_descs[layer].conv_state[b] : L.conv_state;
            void* rec_b  = batch_descs ? batch_descs[layer].recurrent_state[b] : L.recurrent_state;
            float* proj_b = proj_buf + b * proj_buf_floats;

            if (blockIdx.x == 0) {
                float* qkv_f32 = proj_b;
                float* z_f32 = proj_b + L.qkv_out_dim;
                float* b_f32 = proj_b + L.qkv_out_dim + L.z_out_dim;
                float* a_f32 = b_f32 + nv_heads;
                // Batch-strided scratch aliases
                float* gate_up_b = gate_up + b * intermediate_size * 2;
                float* attn_scratch_b = attn_scratch + b * attn_scratch_floats;
                float* mlp_out_b = mlp_out + b * hidden_dim;

                const int conv_dim = L.qkv_out_dim;
                const int kern = L.conv_kernel_size;
                const int nv = L.linear_num_v_heads;
                const int nk = (L.qkv_out_dim - nv * L.linear_head_v_dim) / (2 * L.linear_head_k_dim);
                const int hkd = L.linear_head_k_dim;
                const int hvd = L.linear_head_v_dim;
                const int key_dim = nk * hkd;         // num_k_heads * k_head_dim
                const int val_dim = L.linear_value_dim; // 2048

                // Step B: Conv1d stateful update
                // conv_state: [conv_dim, kern-1] BF16 (sliding window of past values)
                // New qkv values: qkv_f32[conv_dim] (current step)
                // Conv: for each channel c, output = sum(conv_state[c,t] * weight[c,t]) + qkv[c] * weight[c,kern-1]
                // Then update conv_state: shift left, append new value
                {
                    T* cs = static_cast<T*>(conv_b);
                    const T* cw = static_cast<const T*>(L.conv1d_w);
                    // cw layout: [conv_dim, 1, kern] — depthwise conv weight
                    // We treat it as [conv_dim, kern]

                    float* conv_out = gate_up_b;  // reuse MLP scratch for conv output
                    for (int c = tid; c < conv_dim; c += bs) {
                        // Depthwise conv: sum(state[t] * weight[t]) + new_value * weight[kern-1]
                        // Match standard path: convert new value to BF16 for consistency
                        T qkv_bf16 = dotcache_qwen35_from_float<T>(qkv_f32[c]);
                        float acc = 0.0f;
                        if (kern == 4) {
                            const int state_base = c * (kern - 1);
                            const int weight_base = c * kern;
                            const float x0 = dotcache_qwen35_to_float(cs[state_base + 0]);
                            const float x1 = dotcache_qwen35_to_float(cs[state_base + 1]);
                            const float x2 = dotcache_qwen35_to_float(cs[state_base + 2]);
                            const float x3 = dotcache_qwen35_to_float(qkv_bf16);
                            const float w0 = dotcache_qwen35_to_float(cw[weight_base + 0]);
                            const float w1 = dotcache_qwen35_to_float(cw[weight_base + 1]);
                            const float w2 = dotcache_qwen35_to_float(cw[weight_base + 2]);
                            const float w3 = dotcache_qwen35_to_float(cw[weight_base + 3]);
                            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
                        } else {
                            for (int t = 0; t < kern - 1; ++t) {
                                acc += dotcache_qwen35_to_float(cs[c * (kern-1) + t])
                                     * dotcache_qwen35_to_float(cw[c * kern + t]);
                            }
                            acc += dotcache_qwen35_to_float(qkv_bf16)
                                 * dotcache_qwen35_to_float(cw[c * kern + (kern-1)]);
                        }
                        // SiLU activation
                        conv_out[c] = bf16_round_rne_f32_finite(
                            acc * dotcache_qwen35_sigmoid_fast(acc));

                        // Update conv_state: shift left, append new
                        for (int t = 0; t < kern - 2; ++t) {
                            cs[c * (kern-1) + t] = cs[c * (kern-1) + t + 1];
                        }
                        cs[c * (kern-1) + (kern-2)] = qkv_bf16;
                    }
                    __syncthreads();

                    // conv_out[conv_dim] now has the post-conv, post-SiLU output
                    // Split into query, key, value
                    // Layout: [key_dim | key_dim | val_dim] within conv_dim
                    // Actually conv_dim = key_dim*2 + val_dim = 2048+2048+2048? No...
                    // conv_dim = 6144. With nv=16, hkd=128, hvd=128:
                    // key_dim = 16*128 = 2048, val_dim = 16*128 = 2048
                    // So 6144 = 2048 + 2048 + 2048? That's only if there are separate Q/K/V.
                    // Actually for linear attention: conv_dim = in_proj_qkv out = 6144
                    // which packs Q, K, V differently. Let me use the existing split:
                    // First key_dim values = query (normalized later)
                    // Next key_dim values = key
                    // Next val_dim values = value
                    // But wait, with 16 k_heads * 128 k_dim = 2048 for Q
                    //          16 k_heads * 128 k_dim = 2048 for K
                    //          16 v_heads * 128 v_dim = 2048 for V
                    // Total = 6144 ✓
                }

                // Step C0: L2-normalize Q and K per-head (matches standard path)
                // Q_norm = Q / ||Q|| * rsqrt(k_head_dim)
                // K_norm = K / ||K||
                {
                    float* conv_out = gate_up_b;  // reused
                    // nk already derived above from qkv_out_dim
                    // Q layout: conv_out[k_head * hkd .. (k_head+1) * hkd]
                    // K layout: conv_out[key_dim + k_head * hkd .. key_dim + (k_head+1) * hkd]

                    // Normalize Q and K per head using threads 0..nk-1
                    if (tid < nk) {
                        const int h = tid;
                        // Q head
                        float* qh = conv_out + h * hkd;  // q_key_offset + h * hkd
                        float sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += qh[d] * qh[d];
                        float inv = rsqrtf(sq + 1e-6f) * rsqrtf(static_cast<float>(hkd));
                        for (int d = 0; d < hkd; ++d)
                            qh[d] *= inv;

                        // K head
                        float* kh = conv_out + key_dim + h * hkd;
                        sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += kh[d] * kh[d];
                        inv = rsqrtf(sq + 1e-6f);
                        for (int d = 0; d < hkd; ++d)
                            kh[d] *= inv;
                    }
                    __syncthreads();
                }

                // Step C: Parallel delta recurrent state update
                // state: [nv, hkd, hvd] F32 — the recurrent memory
                // 256 threads process 2 heads at a time: threads 0-127 → head h,
                // threads 128-255 → head h+1. Each thread owns one v-dimension.
                // Per head: kv_mem = state @ key, delta = (val - kv_mem) * beta,
                //           state += outer(key, delta), output = state @ query
                {
                    float* conv_out = gate_up_b;  // reused
                    float* state = static_cast<float*>(rec_b);

                    const T* dt_bw = static_cast<const T*>(L.dt_bias_w);
                    const T* ale_w = static_cast<const T*>(L.a_log_exp_w);

                    const int q_key_offset = 0;
                    const int k_key_offset = key_dim;
                    const int v_val_offset = key_dim * 2;
                    const int v = tid % hvd;          // v-dimension for this thread

                    // Process 2 heads per iteration (256 threads / 128 hvd)
                    for (int hp = 0; hp < nv; hp += 2) {
                        const int h = hp + (tid >= hvd ? 1 : 0);

                        if (h < nv) {
                            float* sh = state + h * hkd * hvd;
                            // b/a/dt_bias/a_log are indexed by value head (nv elements each)
                            const float beta = 1.0f / (1.0f + expf(-b_f32[h]));
                            float decay = 1.0f;
                            if (dt_bw != nullptr && ale_w != nullptr) {
                                const float sp = logf(1.0f + expf(
                                    a_f32[h] + dotcache_qwen35_to_float(dt_bw[h])));
                                decay = expf(-sp * dotcache_qwen35_to_float(ale_w[h]));
                            }

                            // Apply decay: state *= exp(g)
                            for (int k = 0; k < hkd; ++k) {
                                sh[k * hvd + v] *= decay;
                            }

                            // Map value head to key head for Q/K indexing
                            // nk key heads shared among nv value heads
                            const int h_k = h * nk / nv;

                            // kv_mem[v] = sum_k(state[k, v] * key[k])
                            float kv_mem_v = 0.0f;
                            for (int k = 0; k < hkd; ++k) {
                                kv_mem_v += sh[k * hvd + v]
                                          * conv_out[k_key_offset + h_k * hkd + k];
                            }

                            // delta = (value - kv_mem) * beta
                            const float val = conv_out[v_val_offset + h * hvd + v];
                            const float delta = (val - kv_mem_v) * beta;

                            // state[k, v] += key[k] * delta (each thread owns its v)
                            for (int k = 0; k < hkd; ++k) {
                                sh[k * hvd + v] +=
                                    conv_out[k_key_offset + h_k * hkd + k] * delta;
                            }

                            // output[v] = sum_k(state[k, v] * query[k])
                            float out_v = 0.0f;
                            for (int k = 0; k < hkd; ++k) {
                                out_v += sh[k * hvd + v]
                                       * conv_out[q_key_offset + h_k * hkd + k];
                            }

                            attn_scratch_b[h * hvd + v] = bf16_round_rne_f32_finite(out_v);
                        }
                        __syncthreads();
                    }
                }

                __syncthreads();

                // Step D: Per-head RMSNorm + weight + SiLU(z) gating
                {
                    const float* nw = static_cast<const float*>(L.linear_norm_w);
                    if (tid < nv) {
                        const int h = tid;
                        const float* ho = attn_scratch_b + h * hvd;
                        float sq = 0.0f;
                        for (int j = 0; j < hvd; ++j)
                            sq += ho[j] * ho[j];
                        lds[h] = rsqrtf(sq / static_cast<float>(hvd) + L.linear_norm_eps);
                    }
                    __syncthreads();
                    for (int d = tid; d < val_dim; d += bs) {
                        const int h = d / hvd;
                        const int hd_idx = d % hvd;
                        float rms_inv = lds[h];
                        float n = attn_scratch_b[d] * rms_inv;
                        float w = (nw != nullptr) ? bf16_round_rne_f32_finite(nw[hd_idx]) : 1.0f;
                        float zv = z_f32[d];
                        attn_scratch_b[d] = bf16_round_rne_f32_finite(
                            n * w * (zv * dotcache_qwen35_sigmoid_fast(zv)));
                    }
                    __syncthreads();
                }

            } // end if (blockIdx.x == 0) for linear attention core
            // Grid barrier: block 0 wrote attn_scratch_b, all blocks need it for out_proj
            grid_barrier(barrier_counter, barrier_flag, nb);

            // Step E: out_proj [hidden_dim, val_dim] × attn_scratch → hidden_f32 (fused residual)
            // Per-sequence: cache batch b's output in LDS, work-steal hidden_dim rows
            {
                const int vd = L.linear_value_dim;  // 2048

                for (int c = tid; c < vd; c += bs)
                    lds_input[c] = attn_scratch[b * attn_scratch_floats + c];
                __syncthreads();

                if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                grid_barrier(barrier_counter, barrier_flag, nb);

                for (;;) {
                    unsigned int sr;
                    if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                    __shared__ unsigned int shared_sr;
                    if (tid == 0) shared_sr = sr;
                    __syncthreads();
                    sr = shared_sr;
                    if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                    float p = 0.0f;
                    if (int4_scales != nullptr && int4_scales[layer].linear_out_proj_scale != nullptr) {
                        const int gsz = int4_scales[layer].group_size;
                        const int byte_cols = vd / 2;
                        const uint8_t* i4_row = static_cast<const uint8_t*>(L.linear_out_proj_w) + static_cast<size_t>(sr) * byte_cols;
                        const hip_bfloat16* sc = static_cast<const hip_bfloat16*>(int4_scales[layer].linear_out_proj_scale);
                        const hip_bfloat16* zr = static_cast<const hip_bfloat16*>(int4_scales[layer].linear_out_proj_zero);
                        const int sr_g = sr / gsz;
                        const int sc_cols = (vd + gsz - 1) / gsz;
                        const int vd8 = vd & ~7;
                        for (int c = tid * 8; c < vd8; c += bs * 8) {
                            uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                            float w[8];
                            int4_dequant_8(pk, sc, zr, sr_g, c, sc_cols, gsz, w);
                            p += w[0]*lds_input[c] + w[1]*lds_input[c+1] + w[2]*lds_input[c+2] + w[3]*lds_input[c+3]
                               + w[4]*lds_input[c+4] + w[5]*lds_input[c+5] + w[6]*lds_input[c+6] + w[7]*lds_input[c+7];
                        }
                        for (int c = vd8 + tid; c < vd; c += bs)
                            p += int4_dequant_scalar(L.linear_out_proj_w, int4_scales[layer].linear_out_proj_scale, int4_scales[layer].linear_out_proj_zero, sr, c, vd, gsz) * lds_input[c];
                    } else if (fp8_scales != nullptr && fp8_scales[layer].linear_out_proj_scale != nullptr) {
                        for (int c = tid; c < vd; c += bs)
                            p += fp8_dequant_weight_lut(L.linear_out_proj_w, fp8_scales[layer].linear_out_proj_scale, sr, c, vd, fp8_scales[layer].block_size, fp8_lut) * lds_input[c];
                    } else {
                        const T* wr = static_cast<const T*>(L.linear_out_proj_w) + static_cast<size_t>(sr) * vd;
                        if (kv_fp8 != nullptr) {
                            if (tid == 0) {
                                for (int c = 0; c < vd; ++c) {
                                    p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                                }
                            }
                        } else {
                            for (int c = tid; c < vd; c += bs)
                                p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                        }
                    }
                    lds[tid] = p;
                    __syncthreads();
                    for (int stride = bs / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) lds[tid] += lds[tid + stride];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float add = bf16_round_rne_f32_finite(lds[0]);
                        hidden_f32[b * hidden_dim + sr] =
                            bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + sr] + add);
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
            grid_barrier(barrier_counter, barrier_flag, nb);
            } // end for (b) batch loop for linear attention

        }

        if (blockIdx.x == 0) {
            for (int b = 0; b < B; b++) {
                for (int c = tid; c < hidden_dim; c += bs) {
                    const float checkpoint =
                        bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + c]);
                    token_out[b * hidden_dim + c] = checkpoint;
                    hidden_io[b * hidden_dim + c] = dotcache_qwen35_from_float<T>(checkpoint);
                    hidden_f32[b * hidden_dim + c] =
                        dotcache_qwen35_to_float(hidden_io[b * hidden_dim + c]);
                }
                __syncthreads();
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // Residual add for token mixer is fused into o_proj/out_proj above.
        // === Post-attention RMSNorm (all blocks cooperate) ===
        {
            const T* norm_w = static_cast<const T*>(L.post_attn_norm_w);
            const float norm_eps = L.post_attn_norm_eps;
            if (blockIdx.x == 0) {
                for (int b = 0; b < B; b++) {
                    const float* src = hidden_f32 + b * hidden_dim;
                    float* dst = normed + b * hidden_dim;
                    float partial_sq = 0.0f;
                    for (int c = tid; c < hidden_dim; c += bs) {
                        partial_sq += src[c] * src[c];
                    }
                    lds[tid] = partial_sq;
                    __syncthreads();
                    for (int s = bs / 2; s > 0; s >>= 1) {
                        if (tid < s) lds[tid] += lds[tid + s];
                        __syncthreads();
                    }
                    float inv_rms = rsqrtf(lds[0] / static_cast<float>(hidden_dim) + norm_eps);
                    for (int c = tid; c < hidden_dim; c += bs) {
                        dst[c] = dotcache_qwen35_from_float<T>(
                            src[c] * inv_rms * (dotcache_qwen35_to_float(norm_w[c]) + 1.0f));
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // All blocks: cache B normed vectors in LDS for fast projection reads
        for (int b = 0; b < B; b++)
            for (int c = tid; c < hidden_dim; c += bs)
                lds_input[b * hidden_dim + c] = normed[b * hidden_dim + c];
        __syncthreads();

        // === Fused MLP gate+up+SwiGLU (all blocks work-steal, BATCHED) ===
        // Use full-block reductions here for parity with the component path.
        if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
        grid_barrier(barrier_counter, barrier_flag, nb);

        {
            const bool gate_fp8 = (fp8_scales != nullptr && fp8_scales[layer].gate_proj_scale != nullptr);
            const bool up_fp8 = (fp8_scales != nullptr && fp8_scales[layer].up_proj_scale != nullptr);
            const bool gate_int4 = (int4_scales != nullptr && int4_scales[layer].gate_proj_scale != nullptr);
            const bool up_int4 = (int4_scales != nullptr && int4_scales[layer].up_proj_scale != nullptr);
            __shared__ unsigned int shared_row_mlp;

            for (;;) {
                if (tid == 0) shared_row_mlp = atomicAdd(&counters[0], 1u);
                __syncthreads();
                unsigned int my_row = shared_row_mlp;
                if (my_row >= static_cast<unsigned int>(L.intermediate_size)) break;

                float gp[MAX_BATCH_SIZE];
                float up[MAX_BATCH_SIZE];
                for (int b = 0; b < B; b++) { gp[b] = 0.0f; up[b] = 0.0f; }

                if (gate_int4 || up_int4) {
                    const int gsz = int4_scales[layer].group_size;
                    const int byte_cols = hidden_dim / 2;
                    const uint8_t* g_i4 = static_cast<const uint8_t*>(L.gate_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                    const uint8_t* u_i4 = static_cast<const uint8_t*>(L.up_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                    const hip_bfloat16* g_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_scale);
                    const hip_bfloat16* g_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_zero);
                    const hip_bfloat16* u_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_scale);
                    const hip_bfloat16* u_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_zero);
                    const int g_sr = my_row / gsz;
                    const int sc_cols = (hidden_dim + gsz - 1) / gsz;
                    const int vd8 = hidden_dim & ~7;
                    for (int c = tid * 8; c < vd8; c += bs * 8) {
                        uint32_t gpk = *reinterpret_cast<const uint32_t*>(&g_i4[c / 2]);
                        uint32_t upk = *reinterpret_cast<const uint32_t*>(&u_i4[c / 2]);
                        float gw[8], uw[8];
                        int4_dequant_8(gpk, g_sc, g_zr, g_sr, c, sc_cols, gsz, gw);
                        int4_dequant_8(upk, u_sc, u_zr, g_sr, c, sc_cols, gsz, uw);
                        for (int b = 0; b < B; b++) {
                            const float* inp = lds_input + b * hidden_dim + c;
                            gp[b] += gw[0]*inp[0] + gw[1]*inp[1] + gw[2]*inp[2] + gw[3]*inp[3]
                                   + gw[4]*inp[4] + gw[5]*inp[5] + gw[6]*inp[6] + gw[7]*inp[7];
                            up[b] += uw[0]*inp[0] + uw[1]*inp[1] + uw[2]*inp[2] + uw[3]*inp[3]
                                   + uw[4]*inp[4] + uw[5]*inp[5] + uw[6]*inp[6] + uw[7]*inp[7];
                        }
                    }
                    for (int c = vd8 + tid; c < hidden_dim; c += bs) {
                        float gw = int4_dequant_scalar(L.gate_proj_w, int4_scales[layer].gate_proj_scale, int4_scales[layer].gate_proj_zero, my_row, c, hidden_dim, gsz);
                        float uw = int4_dequant_scalar(L.up_proj_w, int4_scales[layer].up_proj_scale, int4_scales[layer].up_proj_zero, my_row, c, hidden_dim, gsz);
                        for (int b = 0; b < B; b++) {
                            float inp = lds_input[b * hidden_dim + c];
                            gp[b] += gw * inp;
                            up[b] += uw * inp;
                        }
                    }
                } else if (gate_fp8 || up_fp8) {
                    const uint8_t* g_fp8 = static_cast<const uint8_t*>(L.gate_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                    const uint8_t* u_fp8 = static_cast<const uint8_t*>(L.up_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                    const hip_bfloat16* g_scales = static_cast<const hip_bfloat16*>(fp8_scales[layer].gate_proj_scale);
                    const hip_bfloat16* u_scales = static_cast<const hip_bfloat16*>(fp8_scales[layer].up_proj_scale);
                    const int bsz = fp8_scales[layer].block_size;
                    const int g_scale_row = my_row / bsz;
                    const int scale_cols = (hidden_dim + bsz - 1) / bsz;
                    const int vd4 = hidden_dim & ~3;
                    for (int c = tid * 4; c < vd4; c += bs * 4) {
                        uint32_t gp4 = *reinterpret_cast<const uint32_t*>(&g_fp8[c]);
                        uint32_t up4 = *reinterpret_cast<const uint32_t*>(&u_fp8[c]);
                        const int sb = g_scale_row * scale_cols;
                        float gs0 = static_cast<float>(g_scales[sb + c / bsz]);
                        float gs1 = static_cast<float>(g_scales[sb + (c+1) / bsz]);
                        float gs2 = static_cast<float>(g_scales[sb + (c+2) / bsz]);
                        float gs3 = static_cast<float>(g_scales[sb + (c+3) / bsz]);
                        float us0 = static_cast<float>(u_scales[sb + c / bsz]);
                        float us1 = static_cast<float>(u_scales[sb + (c+1) / bsz]);
                        float us2 = static_cast<float>(u_scales[sb + (c+2) / bsz]);
                        float us3 = static_cast<float>(u_scales[sb + (c+3) / bsz]);
                        float gw0 = bf16_round_rne_f32_finite((fp8_lut[gp4 & 0xFF] * gs0));
                        float gw1 = bf16_round_rne_f32_finite((fp8_lut[(gp4>>8) & 0xFF] * gs1));
                        float gw2 = bf16_round_rne_f32_finite((fp8_lut[(gp4>>16) & 0xFF] * gs2));
                        float gw3 = bf16_round_rne_f32_finite((fp8_lut[(gp4>>24)] * gs3));
                        float uw0 = bf16_round_rne_f32_finite((fp8_lut[up4 & 0xFF] * us0));
                        float uw1 = bf16_round_rne_f32_finite((fp8_lut[(up4>>8) & 0xFF] * us1));
                        float uw2 = bf16_round_rne_f32_finite((fp8_lut[(up4>>16) & 0xFF] * us2));
                        float uw3 = bf16_round_rne_f32_finite((fp8_lut[(up4>>24)] * us3));
                        for (int b = 0; b < B; b++) {
                            const float* inp = lds_input + b * hidden_dim + c;
                            gp[b] += gw0*inp[0] + gw1*inp[1] + gw2*inp[2] + gw3*inp[3];
                            up[b] += uw0*inp[0] + uw1*inp[1] + uw2*inp[2] + uw3*inp[3];
                        }
                    }
                    for (int c = vd4 + tid; c < hidden_dim; c += bs) {
                        float gw = fp8_dequant_weight_lut(L.gate_proj_w, fp8_scales[layer].gate_proj_scale, my_row, c, hidden_dim, bsz, fp8_lut);
                        float uw = fp8_dequant_weight_lut(L.up_proj_w, fp8_scales[layer].up_proj_scale, my_row, c, hidden_dim, bsz, fp8_lut);
                        for (int b = 0; b < B; b++) {
                            float inp = lds_input[b * hidden_dim + c];
                            gp[b] += gw * inp;
                            up[b] += uw * inp;
                        }
                    }
                } else {
                    const T* gr = static_cast<const T*>(L.gate_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                    const T* ur = static_cast<const T*>(L.up_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                    const int vec_dim = hidden_dim & ~3;
                    for (int c = tid * 4; c < vec_dim; c += bs * 4) {
                        float gw0 = dotcache_qwen35_to_float(gr[c]);
                        float gw1 = dotcache_qwen35_to_float(gr[c+1]);
                        float gw2 = dotcache_qwen35_to_float(gr[c+2]);
                        float gw3 = dotcache_qwen35_to_float(gr[c+3]);
                        float uw0 = dotcache_qwen35_to_float(ur[c]);
                        float uw1 = dotcache_qwen35_to_float(ur[c+1]);
                        float uw2 = dotcache_qwen35_to_float(ur[c+2]);
                        float uw3 = dotcache_qwen35_to_float(ur[c+3]);
                        for (int b = 0; b < B; b++) {
                            const float* inp = lds_input + b * hidden_dim + c;
                            gp[b] += gw0 * inp[0] + gw1 * inp[1] + gw2 * inp[2] + gw3 * inp[3];
                            up[b] += uw0 * inp[0] + uw1 * inp[1] + uw2 * inp[2] + uw3 * inp[3];
                        }
                    }
                    for (int c = vec_dim + tid; c < hidden_dim; c += bs) {
                        float gw = dotcache_qwen35_to_float(gr[c]);
                        float uw = dotcache_qwen35_to_float(ur[c]);
                        for (int b = 0; b < B; b++) {
                            float inp = lds_input[b * hidden_dim + c];
                            gp[b] += gw * inp;
                            up[b] += uw * inp;
                        }
                    }
                }

                for (int b = 0; b < B; b++) {
                    lds[tid] = gp[b];
                    __syncthreads();
                    for (int stride = bs / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) lds[tid] += lds[tid + stride];
                        __syncthreads();
                    }
                    float g = bf16_round_rne_f32_finite(lds[0]);
                    lds[tid] = up[b];
                    __syncthreads();
                    for (int stride = bs / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) lds[tid] += lds[tid + stride];
                        __syncthreads();
                    }
                    float u = bf16_round_rne_f32_finite(lds[0]);
                    if (tid == 0) {
                        float silu = g / (1.0f + expf(-g));
                        gate_up[b * intermediate_size * 2 + my_row] =
                            bf16_round_rne_f32_finite(silu * u);
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);

        // === MLP down_proj (all blocks work-steal, per-sequence) ===
        // Process one batch item at a time: cache SwiGLU output in LDS, work-steal
        for (int b = 0; b < B; b++) {
            for (int c = tid; c < L.intermediate_size; c += bs)
                lds_input[c] = gate_up[b * intermediate_size * 2 + c];
            __syncthreads();

            if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
            grid_barrier(barrier_counter, barrier_flag, nb);

            {
                const bool down_fp8 = (fp8_scales != nullptr && fp8_scales[layer].down_proj_scale != nullptr);
                const bool down_int4 = (int4_scales != nullptr && int4_scales[layer].down_proj_scale != nullptr);
                __shared__ unsigned int shared_row_down;
                for (;;) {
                    if (tid == 0) shared_row_down = atomicAdd(&counters[0], 1u);
                    __syncthreads();
                    unsigned int my_row = shared_row_down;
                    if (my_row >= static_cast<unsigned int>(hidden_dim)) break;

                    float p = 0.0f;
                    if (down_int4) {
                        const int gsz = int4_scales[layer].group_size;
                        const int byte_cols = L.intermediate_size / 2;
                        const uint8_t* i4_row = static_cast<const uint8_t*>(L.down_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                        const hip_bfloat16* d_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_scale);
                        const hip_bfloat16* d_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_zero);
                        const int sr_g = my_row / gsz;
                        const int sc_cols = (L.intermediate_size + gsz - 1) / gsz;
                        const int is8 = L.intermediate_size & ~7;
                        for (int c = tid * 8; c < is8; c += bs * 8) {
                            uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                            float w[8];
                            int4_dequant_8(pk, d_sc, d_zr, sr_g, c, sc_cols, gsz, w);
                            p += w[0]*lds_input[c] + w[1]*lds_input[c+1] + w[2]*lds_input[c+2] + w[3]*lds_input[c+3]
                               + w[4]*lds_input[c+4] + w[5]*lds_input[c+5] + w[6]*lds_input[c+6] + w[7]*lds_input[c+7];
                        }
                        for (int c = is8 + tid; c < L.intermediate_size; c += bs)
                            p += int4_dequant_scalar(L.down_proj_w, int4_scales[layer].down_proj_scale, int4_scales[layer].down_proj_zero, my_row, c, L.intermediate_size, gsz) * lds_input[c];
                    } else if (down_fp8) {
                        const uint8_t* d_fp8 = static_cast<const uint8_t*>(L.down_proj_w) + static_cast<size_t>(my_row) * L.intermediate_size;
                        const hip_bfloat16* d_scales = static_cast<const hip_bfloat16*>(fp8_scales[layer].down_proj_scale);
                        const int bsz = fp8_scales[layer].block_size;
                        const int d_scale_row = my_row / bsz;
                        const int d_scale_cols = (L.intermediate_size + bsz - 1) / bsz;
                        const int is4 = L.intermediate_size & ~3;
                        for (int c = tid * 4; c < is4; c += bs * 4) {
                            uint32_t pk = *reinterpret_cast<const uint32_t*>(&d_fp8[c]);
                            const int sb = d_scale_row * d_scale_cols;
                            float w0 = bf16_round_rne_f32_finite((fp8_lut[pk & 0xFF] * static_cast<float>(d_scales[sb + c / bsz])));
                            float w1 = bf16_round_rne_f32_finite((fp8_lut[(pk>>8) & 0xFF] * static_cast<float>(d_scales[sb + (c+1) / bsz])));
                            float w2 = bf16_round_rne_f32_finite((fp8_lut[(pk>>16) & 0xFF] * static_cast<float>(d_scales[sb + (c+2) / bsz])));
                            float w3 = bf16_round_rne_f32_finite((fp8_lut[(pk>>24)] * static_cast<float>(d_scales[sb + (c+3) / bsz])));
                            p += w0*lds_input[c] + w1*lds_input[c+1] + w2*lds_input[c+2] + w3*lds_input[c+3];
                        }
                        for (int c = is4 + tid; c < L.intermediate_size; c += bs)
                            p += fp8_dequant_weight_lut(L.down_proj_w, fp8_scales[layer].down_proj_scale, my_row, c, L.intermediate_size, bsz, fp8_lut) * lds_input[c];
                    } else {
                        const T* wr = static_cast<const T*>(L.down_proj_w) + static_cast<size_t>(my_row) * L.intermediate_size;
                        for (int c = tid; c < L.intermediate_size; c += bs)
                            p += dotcache_qwen35_to_float(wr[c]) * lds_input[c];
                    }
                    lds[tid] = p;
                    __syncthreads();
                    for (int stride = bs / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) lds[tid] += lds[tid + stride];
                        __syncthreads();
                    }
                    if (tid == 0) {
                        float add = bf16_round_rne_f32_finite(lds[0]);
                        mlp_out[b * hidden_dim + my_row] = add;
                        hidden_f32[b * hidden_dim + my_row] =
                            bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + my_row] + add);
                    }
                    __syncthreads();
                }
            }
            // Need syncthreads before grid barrier since waves may finish at different times
            __syncthreads();
            grid_barrier(barrier_counter, barrier_flag, nb);
        } // end for (b) down_proj batch loop

        // Residual add for MLP is fused into down_proj above.
    }

    // Write back F32 → BF16 for all batch items
    for (int b = 0; b < B; b++) {
        for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
            hidden_io[b * hidden_dim + c] = dotcache_qwen35_from_float<T>(hidden_f32[b * hidden_dim + c]);
        }
    }
}
