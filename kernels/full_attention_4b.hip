#pragma once

// Qwen3.8 is the sole public model. The qwen35-prefixed HIP identifiers in
// this translation unit are retained only where the bridge/archive ABI or
// hipcc's existing kernel symbol linkage requires the historical spelling;
// they are not compatibility aliases or additional model implementations.

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "gqh-stride.h"

// Host writes this before a split-kernel graph replay so seqlen is not
// baked into captured node params.
__device__ int qwen35_split_seqlen;
// Legacy symbol; steal skip is now per-layer via io_flags bit 4.
__device__ int qwen35_gemv_out_mask;

// Per-section timing slot layout mirrors the host decoder's layout so the Rust
// reader `decode_persistent_4b_timing_slots` is backend-agnostic). Each layer
// owns QWEN35_4B_TIMING_COUNT u64 slots; per-batch-item sections (CORE/OUT)
// reserve 8 slots each so batch_size up to 8 is covered.
enum Qwen35Persistent4BTimingSlot {
    QWEN35_4B_TIMING_FULL_ATTN = 0,
    QWEN35_4B_TIMING_FULL_ATTN_PROJ = 1,
    QWEN35_4B_TIMING_FULL_ATTN_CORE_BASE = 2,
    QWEN35_4B_TIMING_FULL_ATTN_OUT_BASE = 10,
    QWEN35_4B_TIMING_LINEAR_PROJ = 18,
    QWEN35_4B_TIMING_LINEAR_CORE_BASE = 19,
    QWEN35_4B_TIMING_LINEAR_OUT_BASE = 27,
    QWEN35_4B_TIMING_LINEAR_CORE_CONV_BASE = 35,
    QWEN35_4B_TIMING_LINEAR_CORE_RECURRENT_BASE = 37,
    QWEN35_4B_TIMING_LINEAR_CORE_POST_BASE = 39,
    QWEN35_4B_TIMING_MLP_GATE_UP = 41,
    QWEN35_4B_TIMING_MLP_DOWN = 42,
    QWEN35_4B_TIMING_COUNT = 43,
};

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

__device__ __forceinline__ uint16_t f32_to_bf16_bits_rne_finite(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits_trunc(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits16) {
    uint32_t bits = static_cast<uint32_t>(bits16) << 16;
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
    if (val >= 448.0f) return sign | 0x7E;  // max normal: stored-exp=15, mantissa=6 → 2^8*(1+6/8)=448
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
    const void* gate_proj_awq_inv_scale;
    const void* up_proj_scale;
    const void* up_proj_zero;
    const void* up_proj_awq_inv_scale;
    const void* down_proj_scale;
    const void* down_proj_zero;
    const void* down_proj_awq_inv_scale;
    // Linear attention weights
    const void* qkv_proj_scale;
    const void* qkv_proj_zero;
    const void* qkv_proj_awq_inv_scale;
    const void* z_proj_scale;
    const void* z_proj_zero;
    const void* z_proj_awq_inv_scale;
    const void* linear_out_proj_scale;
    const void* linear_out_proj_zero;
    const void* linear_out_proj_awq_inv_scale;
    // Full attention weights
    const void* q_proj_scale;
    const void* q_proj_zero;
    const void* q_proj_awq_inv_scale;
    const void* k_proj_scale;
    const void* k_proj_zero;
    const void* k_proj_awq_inv_scale;
    const void* v_proj_scale;
    const void* v_proj_zero;
    const void* v_proj_awq_inv_scale;
    const void* o_proj_scale;
    const void* o_proj_zero;
    const void* o_proj_awq_inv_scale;
    // Group size for INT4 quantization (typically 128)
    int group_size;
    int gate_proj_type;
    int up_proj_type;
    int down_proj_type;
    int qkv_proj_type;
    int z_proj_type;
    int linear_out_proj_type;
    int q_proj_type;
    int k_proj_type;
    int v_proj_type;
    int o_proj_type;
    int a_proj_type;
    int b_proj_type;
};

struct Qwen35GqhHdr {
    float tensor_scale;
    int grid_code;
};

static constexpr int QWEN35_LOWBIT_NATIVE_INT4 = 4;
static constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
static constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
static constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
static constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
static constexpr int QWEN35_LOWBIT_GQH3 = 108;
static constexpr int QWEN35_LOWBIT_GQH2_H = 109;
static constexpr int QWEN35_LOWBIT_GQH2_C = 110;
static constexpr int QWEN35_LOWBIT_GQH4 = 111;

static __constant__ uint32_t QWEN35_GQH_E4M3[32][8] = GQH_E4M3_LUT_INIT;
static __constant__ uint32_t QWEN35_GQH_RATIO_Q[16][1] = GQH_RATIO_Q_INIT;
static __constant__ uint32_t QWEN35_GQH3_GRID[12][8] = GQH3_GRID_INIT;
static __constant__ uint32_t QWEN35_GQH2H_GRID[12][4] = GQH2H_GRID_INIT;
static __constant__ uint32_t QWEN35_GQH4_GRID[12][16] = GQH4_GRID_INIT;

__device__ inline bool qwen35_lowbit_native_int4(int qtype) {
    return qtype == QWEN35_LOWBIT_NATIVE_INT4;
}

__device__ inline bool qwen35_lowbit_ggml_k(int qtype) {
    return qtype == QWEN35_LOWBIT_GGML_Q8_0 ||
           qtype == QWEN35_LOWBIT_GGML_Q4_K ||
           qtype == QWEN35_LOWBIT_GGML_Q5_K ||
           qtype == QWEN35_LOWBIT_GGML_Q6_K;
}

__device__ inline bool qwen35_lowbit_gqh(int qtype) {
    return qtype == QWEN35_LOWBIT_GQH3 ||
           qtype == QWEN35_LOWBIT_GQH2_H ||
           qtype == QWEN35_LOWBIT_GQH2_C ||
           qtype == QWEN35_LOWBIT_GQH4;
}

__device__ inline bool qwen35_lowbit_supported(int qtype) {
    return qwen35_lowbit_native_int4(qtype) || qwen35_lowbit_ggml_k(qtype) ||
           qwen35_lowbit_gqh(qtype);
}

#ifndef GQH_ALLOW_FMA
#pragma clang fp contract(off)
#endif
__device__ inline float qwen35_gqh_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

__device__ inline float qwen35_gqh_subblock_scale(
    const uint8_t* __restrict__ b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real =
        qwen35_gqh_bits(QWEN35_GQH_E4M3[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * qwen35_gqh_bits(QWEN35_GQH_RATIO_Q[ratio][0]);
}

__device__ inline int qwen35_gqh_row_bytes(int qtype, int cols) {
    const int nsb = cols / GQH_SUPERBLOCK;
    if (qtype == QWEN35_LOWBIT_GQH3) return gqh_plane_row_bytes(nsb, 1);
    if (qtype == QWEN35_LOWBIT_GQH2_H) return gqh_plane_row_bytes(nsb, 0);
    if (qtype == QWEN35_LOWBIT_GQH2_C) return nsb * GQH2C_SB_BYTES;
    if (qtype == QWEN35_LOWBIT_GQH4) return nsb * GQH4_SB_BYTES;
    return 0;
}

__device__ inline float qwen35_gqh_scale_planes(
    uint8_t d, const uint8_t* __restrict__ ratio8, int sub, float tensor_scale) {
    const float d_real =
        qwen35_gqh_bits(QWEN35_GQH_E4M3[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = ratio8[sub >> 1];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * qwen35_gqh_bits(QWEN35_GQH_RATIO_Q[ratio][0]);
}

__device__ inline float qwen35_gqh_dequant_scalar(
    const void* w_ptr, int qtype, int row, int col, int cols,
    float tensor_scale, int grid_code) {
    const int nsb = cols / GQH_SUPERBLOCK;
    const int is3 = qtype == QWEN35_LOWBIT_GQH3;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* rowbase = static_cast<const uint8_t*>(w_ptr) +
        static_cast<size_t>(row) * static_cast<size_t>(stride);
    const int sb = col / GQH_SUPERBLOCK;
    const int j = col % GQH_SUPERBLOCK;
    const uint8_t d = rowbase[sb];
    const uint8_t* rat = rowbase + off_r + (sb << 3);
    const uint8_t* lo_p = rowbase + off_lo + (sb << 6);
    const float s_b = qwen35_gqh_scale_planes(d, rat, j >> 4, tensor_scale);
    if (is3) {
        const uint8_t* hi_p = rowbase + off_hi + (sb << 5);
        const int lo = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
        const int hi = (hi_p[j >> 3] >> (j & 7)) & 0x01;
        return qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][lo | (hi << 2)]) * s_b;
    }
    if (qtype == QWEN35_LOWBIT_GQH2_H) {
        const int code = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
        return qwen35_gqh_bits(QWEN35_GQH2H_GRID[grid_code][code]) * s_b;
    }
    if (qtype == QWEN35_LOWBIT_GQH4) {
        const uint8_t* b = static_cast<const uint8_t*>(w_ptr) +
            static_cast<size_t>(row) * static_cast<size_t>(nsb * GQH4_SB_BYTES) +
            static_cast<size_t>(sb) * GQH4_SB_BYTES;
        const float s4 = qwen35_gqh_subblock_scale(b, j >> 4, tensor_scale);
        const uint8_t cb = b[9 + (j >> 1)];
        const int code = (j & 1) ? (cb >> 4) : (cb & 0x0f);
        return qwen35_gqh_bits(QWEN35_GQH4_GRID[grid_code][code]) * s4;
    }
    return 0.0f;
}

__device__ inline void qwen35_gqh_dequant_8(
    const void* w_ptr, int qtype, int row, int col, int cols,
    float tensor_scale, int grid_code, float out[8]) {
    const int nsb = cols / GQH_SUPERBLOCK;
    if (qtype == QWEN35_LOWBIT_GQH4) {
        const uint8_t* b = static_cast<const uint8_t*>(w_ptr) +
            static_cast<size_t>(row) * static_cast<size_t>(nsb * GQH4_SB_BYTES) +
            static_cast<size_t>(col / GQH_SUPERBLOCK) * GQH4_SB_BYTES;
        const int j0 = col % GQH_SUPERBLOCK;
        const float s4 = qwen35_gqh_subblock_scale(b, j0 >> 4, tensor_scale);
        for (int i = 0; i < 8; ++i) {
            const int j = j0 + i;
            const uint8_t cb = b[9 + (j >> 1)];
            const int code = (j & 1) ? (cb >> 4) : (cb & 0x0f);
            out[i] = qwen35_gqh_bits(QWEN35_GQH4_GRID[grid_code][code]) * s4;
        }
        return;
    }
    const int is3 = qtype == QWEN35_LOWBIT_GQH3;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* rowbase = static_cast<const uint8_t*>(w_ptr) +
        static_cast<size_t>(row) * static_cast<size_t>(stride);
    const int sb = col / GQH_SUPERBLOCK;
    const int j0 = col % GQH_SUPERBLOCK;
    const uint8_t d = rowbase[sb];
    const uint8_t* rat = rowbase + off_r + (sb << 3);
    const uint8_t* lo_p = rowbase + off_lo + (sb << 6);
    const uint8_t* hi_p = is3 ? rowbase + off_hi + (sb << 5) : nullptr;
    const float s_b = qwen35_gqh_scale_planes(d, rat, j0 >> 4, tensor_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int j = j0 + i;
        if (is3) {
            const int lo = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
            const int hi = (hi_p[j >> 3] >> (j & 7)) & 0x01;
            out[i] = qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][lo | (hi << 2)]) * s_b;
        } else if (qtype == QWEN35_LOWBIT_GQH2_H) {
            const int code = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
            out[i] = qwen35_gqh_bits(QWEN35_GQH2H_GRID[grid_code][code]) * s_b;
        } else {
            out[i] = 0.0f;
        }
    }
}

__device__ inline void qwen35_gqh_hdr(
    const void* scale_ptr, float* tensor_scale, int* grid_code) {
    if (scale_ptr == nullptr) {
        *tensor_scale = 0.0f;
        *grid_code = 0;
        return;
    }
    const Qwen35GqhHdr* h = static_cast<const Qwen35GqhHdr*>(scale_ptr);
    *tensor_scale = h->tensor_scale;
    *grid_code = h->grid_code;
}

// Same register-grid + packed-code walk as kernels/gqh.hip. Each lane owns
// 8 consecutive weights in the 256-wide superblock (lane*8).
__device__ inline float qwen35_gqh_with_sign(float mag, int positive) {
    uint32_t bits;
    memcpy(&bits, &mag, 4);
    bits ^= (uint32_t)(positive - 1) & 0x80000000u;
    memcpy(&mag, &bits, 4);
    return mag;
}

__device__ inline float qwen35_gqh_level(int code, float4 mag, bool is3) {
    if (is3) {
        const int hi = (code >> 2) & 1;
        const int k = code & 3;
        const int idx = hi ? k : (3 - k);
        const float e0 = (idx & 1) ? mag.y : mag.x;
        const float e1 = (idx & 1) ? mag.w : mag.z;
        const float m = (idx & 2) ? e1 : e0;
        return qwen35_gqh_with_sign(m, hi);
    }
    const int hi = (code >> 1) & 1;
    const float m = ((code & 1) != hi) ? mag.x : 1.0f;
    return qwen35_gqh_with_sign(m, hi);
}

__device__ inline float4 qwen35_gqh_mag(int qtype, int grid_code) {
    float4 mag;
    mag.x = mag.y = mag.z = mag.w = 0.0f;
    if (qtype == QWEN35_LOWBIT_GQH3) {
        mag.x = qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][4]);
        mag.y = qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][5]);
        mag.z = qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][6]);
        mag.w = qwen35_gqh_bits(QWEN35_GQH3_GRID[grid_code][7]);
    } else if (qtype == QWEN35_LOWBIT_GQH2_H) {
        mag.x = qwen35_gqh_bits(QWEN35_GQH2H_GRID[grid_code][2]);
    }
    return mag;
}

// x[0] is column sb0 * 256. nsb is the number of superblocks to walk.
// noinline keeps the giant persistent kernel's VGPR live ranges smaller.
__device__ __noinline__ float qwen35_gqh_lane_dot_range(
    const void* w_ptr, int qtype, int row, int cols,
    float tensor_scale, int grid_code,
    const float* __restrict__ x, int sb0, int nsb, int lane) {
    const bool is3 = qtype == QWEN35_LOWBIT_GQH3;
    const int row_nsb = cols / GQH_SUPERBLOCK;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(row_nsb, is3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* rowbase = static_cast<const uint8_t*>(w_ptr) +
        static_cast<size_t>(row) * static_cast<size_t>(stride);
    const int j0 = lane * 8;
    const int sub = j0 >> 4;
    const float4 mag = qwen35_gqh_mag(qtype, grid_code);
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const int s = sb0 + sb;
        const uint8_t d = rowbase[s];
        const float d_real =
            qwen35_gqh_bits(QWEN35_GQH_E4M3[d >> 3][d & 7]) * tensor_scale;
        const uint8_t rb = rowbase[off_r + (s << 3) + (sub >> 1)];
        const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
        const float s_b = d_real * qwen35_gqh_bits(QWEN35_GQH_RATIO_Q[ratio][0]);
        uint16_t lo2;
        memcpy(&lo2, rowbase + off_lo + (s << 6) + lane * 2, sizeof(lo2));
        const uint8_t hi1 = is3 ? rowbase[off_hi + (s << 5) + lane] : 0;
        const float4 x0 = *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 = *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int lo = (lo2 >> (2 * t)) & 0x03;
            const int code = is3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
            acc += (qwen35_gqh_level(code, mag, is3) * s_b) * xs[t];
        }
    }
    return acc;
}

// Two output rows, one activation pass. Cuts x traffic in half on fat K.
__device__ __noinline__ void qwen35_gqh_lane_dot_pair(
    const void* w_ptr, int qtype, int row0, int row1, int cols,
    float tensor_scale, int grid_code,
    const float* __restrict__ x, int nsb, int lane,
    float* acc0, float* acc1) {
    const bool is3 = qtype == QWEN35_LOWBIT_GQH3;
    const int row_nsb = cols / GQH_SUPERBLOCK;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(row_nsb, is3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* base0 = static_cast<const uint8_t*>(w_ptr) +
        static_cast<size_t>(row0) * static_cast<size_t>(stride);
    const uint8_t* base1 = (row1 >= 0)
        ? static_cast<const uint8_t*>(w_ptr) +
            static_cast<size_t>(row1) * static_cast<size_t>(stride)
        : nullptr;
    const int j0 = lane * 8;
    const int sub = j0 >> 4;
    const float4 mag = qwen35_gqh_mag(qtype, grid_code);
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const float4 x0 = *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0);
        const float4 x1v = *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1v.x, x1v.y, x1v.z, x1v.w};
        const uint8_t d0 = base0[sb];
        const float d_real0 =
            qwen35_gqh_bits(QWEN35_GQH_E4M3[d0 >> 3][d0 & 7]) * tensor_scale;
        const uint8_t rb0 = base0[off_r + (sb << 3) + (sub >> 1)];
        const float s0 = d_real0 * qwen35_gqh_bits(
            QWEN35_GQH_RATIO_Q[(sub & 1) ? (rb0 >> 4) : (rb0 & 0x0f)][0]);
        uint16_t lo0;
        memcpy(&lo0, base0 + off_lo + (sb << 6) + lane * 2, sizeof(lo0));
        const uint8_t hi0 = is3 ? base0[off_hi + (sb << 5) + lane] : 0;
        float s1 = 0.0f;
        uint16_t lo1 = 0;
        uint8_t hi1 = 0;
        if (base1 != nullptr) {
            const uint8_t d1 = base1[sb];
            const float d_real1 =
                qwen35_gqh_bits(QWEN35_GQH_E4M3[d1 >> 3][d1 & 7]) * tensor_scale;
            const uint8_t rb1 = base1[off_r + (sb << 3) + (sub >> 1)];
            s1 = d_real1 * qwen35_gqh_bits(
                QWEN35_GQH_RATIO_Q[(sub & 1) ? (rb1 >> 4) : (rb1 & 0x0f)][0]);
            memcpy(&lo1, base1 + off_lo + (sb << 6) + lane * 2, sizeof(lo1));
            hi1 = is3 ? base1[off_hi + (sb << 5) + lane] : 0;
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int c0 = is3
                ? ((lo0 >> (2 * t)) & 0x03) | (((hi0 >> t) & 1) << 2)
                : ((lo0 >> (2 * t)) & 0x03);
            a0 += (qwen35_gqh_level(c0, mag, is3) * s0) * xs[t];
            if (base1 != nullptr) {
                const int c1 = is3
                    ? ((lo1 >> (2 * t)) & 0x03) | (((hi1 >> t) & 1) << 2)
                    : ((lo1 >> (2 * t)) & 0x03);
                a1 += (qwen35_gqh_level(c1, mag, is3) * s1) * xs[t];
            }
        }
    }
    *acc0 += a0;
    *acc1 += a1;
}

__device__ inline void qwen35_gqh_lane_dots(
    const void* w_ptr, int qtype, int row, int cols, const void* hdr,
    const float* __restrict__ x_base, int x_stride, int B,
    float* acc, int lane) {
    float tensor_scale;
    int grid_code;
    qwen35_gqh_hdr(hdr, &tensor_scale, &grid_code);
    const int nsb = cols / GQH_SUPERBLOCK;
    if (B == 1) {
        acc[0] += qwen35_gqh_lane_dot_range(
            w_ptr, qtype, row, cols, tensor_scale, grid_code,
            x_base, 0, nsb, lane);
        return;
    }
    for (int bi = 0; bi < B; ++bi) {
        acc[bi] += qwen35_gqh_lane_dot_range(
            w_ptr, qtype, row, cols, tensor_scale, grid_code,
            x_base + bi * x_stride, 0, nsb, lane);
    }
}

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

__device__ inline int ggml_k_row_bytes(int qtype, int cols) {
    if (qtype == QWEN35_LOWBIT_GGML_Q8_0) return (cols / 32) * 34;
    const int blocks = cols / 256;
    if (qtype == QWEN35_LOWBIT_GGML_Q4_K) return blocks * 144;
    if (qtype == QWEN35_LOWBIT_GGML_Q5_K) return blocks * 176;
    if (qtype == QWEN35_LOWBIT_GGML_Q6_K) return blocks * 210;
    return cols / 2;
}

__device__ inline float ggml_f16_to_f32_unaligned(const uint8_t* p) {
    uint16_t bits = static_cast<uint16_t>(p[0]) |
                    (static_cast<uint16_t>(p[1]) << 8);
    __half h = *reinterpret_cast<__half*>(&bits);
    return __half2float(h);
}

__device__ inline void ggml_q4_k_scale_min(const uint8_t* q, int j, int& scale, int& minv) {
    if (j < 4) {
        scale = static_cast<int>(q[j] & 63);
        minv = static_cast<int>(q[j + 4] & 63);
    } else {
        scale = static_cast<int>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        minv = static_cast<int>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__device__ inline float ggml_k_dequant_scalar_from_row(
    const uint8_t* row_data, int qtype, int col, int cols
) {
    if (qtype == QWEN35_LOWBIT_GGML_Q8_0) {
        const int block = col / 32;
        const int inb = col - block * 32;
        const uint8_t* b = row_data + block * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
        return d * static_cast<float>(qs[inb]);
    }

    const int block = col / 256;
    const int inb = col - block * 256;
    const uint8_t* b = row_data;

    if (qtype == QWEN35_LOWBIT_GGML_Q4_K) {
        b += block * 144;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qs = b + 16;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const uint8_t qbyte = qs[g * 32 + (inb % 32)];
        const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        return d * static_cast<float>(scale) * static_cast<float>(q) -
               dmin * static_cast<float>(minv);
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q5_K) {
        b += block * 176;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const int idx = inb % 32;
        const uint8_t qbyte = ql[g * 32 + idx];
        const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        const int hi = (qh[idx] & (sub != 0 ? (2 << (2 * g)) : (1 << (2 * g)))) ? 16 : 0;
        return d * static_cast<float>(scale) * static_cast<float>(lo + hi) -
               dmin * static_cast<float>(minv);
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q6_K) {
        b += block * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = ggml_f16_to_f32_unaligned(b + 208);
        const int half_idx = inb / 128;
        const int pos = inb - half_idx * 128;
        const int idx = pos % 32;
        int q = 0;
        int scale_idx = half_idx * 8 + idx / 16;
        const uint8_t qh_byte = qh[half_idx * 32 + idx];
        if (pos < 32) {
            q = static_cast<int>(ql[half_idx * 64 + idx] & 0x0F) |
                static_cast<int>(((qh_byte >> 0) & 3) << 4);
        } else if (pos < 64) {
            q = static_cast<int>(ql[half_idx * 64 + 32 + idx] & 0x0F) |
                static_cast<int>(((qh_byte >> 2) & 3) << 4);
            scale_idx += 2;
        } else if (pos < 96) {
            q = static_cast<int>((ql[half_idx * 64 + idx] >> 4) & 0x0F) |
                static_cast<int>(((qh_byte >> 4) & 3) << 4);
            scale_idx += 4;
        } else {
            q = static_cast<int>((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0F) |
                static_cast<int>(((qh_byte >> 6) & 3) << 4);
            scale_idx += 6;
        }
        return d * static_cast<float>(sc[scale_idx]) * static_cast<float>(q - 32);
    }

    return 0.0f;
}

__device__ inline float ggml_k_dequant_scalar(
    const void* w_ptr, int qtype, int row, int col, int cols
) {
    const uint8_t* data = static_cast<const uint8_t*>(w_ptr);
    const uint8_t* row_data = data + static_cast<size_t>(row) * ggml_k_row_bytes(qtype, cols);
    return ggml_k_dequant_scalar_from_row(row_data, qtype, col, cols);
}

__device__ inline void ggml_k_dequant_pair_bf16_bits_from_row(
    const uint8_t* row_data,
    int qtype,
    int col0,
    int cols,
    uint16_t& out0,
    uint16_t& out1
) {
    const int col1 = col0 + 1;
    const int block = col0 / 256;
    const int inb = col0 - block * 256;

    if ((col0 & 1) != 0 || col1 >= cols || (col1 / 256) != block) {
        out0 = f32_to_bf16_bits_rne_finite(
            ggml_k_dequant_scalar_from_row(row_data, qtype, col0, cols));
        out1 = (col1 < cols)
            ? f32_to_bf16_bits_rne_finite(
                  ggml_k_dequant_scalar_from_row(row_data, qtype, col1, cols))
            : 0;
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q8_0) {
        const uint8_t* b = row_data + (col0 / 32) * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
        const int idx = col0 & 31;
        out0 = f32_to_bf16_bits_rne_finite(d * static_cast<float>(qs[idx]));
        out1 = f32_to_bf16_bits_rne_finite(d * static_cast<float>(qs[idx + 1]));
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q4_K) {
        const uint8_t* b = row_data + block * 144;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qs = b + 16;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        const uint8_t qbyte0 = qs[g * 32 + idx];
        const uint8_t qbyte1 = qs[g * 32 + idx + 1];
        const int q0 = sub != 0 ? ((qbyte0 >> 4) & 0x0F) : (qbyte0 & 0x0F);
        const int q1 = sub != 0 ? ((qbyte1 >> 4) & 0x0F) : (qbyte1 & 0x0F);
        out0 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(q0) - dminv);
        out1 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(q1) - dminv);
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q5_K) {
        const uint8_t* b = row_data + block * 176;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const uint8_t qbyte0 = ql[g * 32 + idx];
        const uint8_t qbyte1 = ql[g * 32 + idx + 1];
        const int lo0 = sub != 0 ? ((qbyte0 >> 4) & 0x0F) : (qbyte0 & 0x0F);
        const int lo1 = sub != 0 ? ((qbyte1 >> 4) & 0x0F) : (qbyte1 & 0x0F);
        const int hi_mask = sub != 0 ? (2 << (2 * g)) : (1 << (2 * g));
        const int hi0 = (qh[idx] & hi_mask) ? 16 : 0;
        const int hi1 = (qh[idx + 1] & hi_mask) ? 16 : 0;
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        out0 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(lo0 + hi0) - dminv);
        out1 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(lo1 + hi1) - dminv);
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q6_K) {
        const uint8_t* b = row_data + block * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = ggml_f16_to_f32_unaligned(b + 208);
        const int half_idx = inb / 128;
        const int pos = inb - half_idx * 128;
        const int idx = pos % 32;
        const uint8_t qh0 = qh[half_idx * 32 + idx];
        const uint8_t qh1 = qh[half_idx * 32 + idx + 1];
        int q0 = 0;
        int q1 = 0;
        int scale_idx = half_idx * 8 + idx / 16;
        if (pos < 32) {
            q0 = static_cast<int>(ql[half_idx * 64 + idx] & 0x0F) |
                 static_cast<int>(((qh0 >> 0) & 3) << 4);
            q1 = static_cast<int>(ql[half_idx * 64 + idx + 1] & 0x0F) |
                 static_cast<int>(((qh1 >> 0) & 3) << 4);
        } else if (pos < 64) {
            q0 = static_cast<int>(ql[half_idx * 64 + 32 + idx] & 0x0F) |
                 static_cast<int>(((qh0 >> 2) & 3) << 4);
            q1 = static_cast<int>(ql[half_idx * 64 + 32 + idx + 1] & 0x0F) |
                 static_cast<int>(((qh1 >> 2) & 3) << 4);
            scale_idx += 2;
        } else if (pos < 96) {
            q0 = static_cast<int>((ql[half_idx * 64 + idx] >> 4) & 0x0F) |
                 static_cast<int>(((qh0 >> 4) & 3) << 4);
            q1 = static_cast<int>((ql[half_idx * 64 + idx + 1] >> 4) & 0x0F) |
                 static_cast<int>(((qh1 >> 4) & 3) << 4);
            scale_idx += 4;
        } else {
            q0 = static_cast<int>((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0F) |
                 static_cast<int>(((qh0 >> 6) & 3) << 4);
            q1 = static_cast<int>((ql[half_idx * 64 + 32 + idx + 1] >> 4) & 0x0F) |
                 static_cast<int>(((qh1 >> 6) & 3) << 4);
            scale_idx += 6;
        }
        const float ds = d * static_cast<float>(sc[scale_idx]);
        out0 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(q0 - 32));
        out1 = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(q1 - 32));
        return;
    }

    out0 = 0;
    out1 = 0;
}

__device__ inline float lowbit_dequant_scalar(
    const void* w_ptr, const void* scale_ptr, const void* zero_ptr,
    int qtype, int row, int col, int cols, int group_size
) {
    if (qwen35_lowbit_native_int4(qtype)) {
        return int4_dequant_scalar(w_ptr, scale_ptr, zero_ptr, row, col, cols, group_size);
    }
    if (qwen35_lowbit_gqh(qtype)) {
        float tensor_scale;
        int grid_code;
        qwen35_gqh_hdr(scale_ptr, &tensor_scale, &grid_code);
        return qwen35_gqh_dequant_scalar(
            w_ptr, qtype, row, col, cols, tensor_scale, grid_code);
    }
    return ggml_k_dequant_scalar(w_ptr, qtype, row, col, cols);
}

__device__ inline void lowbit_dequant_8(
    const void* w_ptr, const void* scale_ptr, const void* zero_ptr,
    int qtype, int row, int col, int cols, int group_size,
    float out[8]
) {
    if (qwen35_lowbit_native_int4(qtype)) {
        const int byte_cols = cols / 2;
        const uint8_t* i4_row = static_cast<const uint8_t*>(w_ptr) +
            static_cast<size_t>(row) * byte_cols;
        const int scale_row = row / group_size;
        const int scale_cols = (cols + group_size - 1) / group_size;
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(&i4_row[col / 2]);
        int4_dequant_8(
            packed,
            static_cast<const hip_bfloat16*>(scale_ptr),
            static_cast<const hip_bfloat16*>(zero_ptr),
            scale_row,
            col,
            scale_cols,
            group_size,
            out);
    } else if (qwen35_lowbit_gqh(qtype)) {
        float tensor_scale;
        int grid_code;
        qwen35_gqh_hdr(scale_ptr, &tensor_scale, &grid_code);
        qwen35_gqh_dequant_8(
            w_ptr, qtype, row, col, cols, tensor_scale, grid_code, out);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = ggml_k_dequant_scalar(w_ptr, qtype, row, col + i, cols);
        }
    }
}

__device__ inline float awq_inv_scale_at(const void* awq_inv_scale, int col) {
    if (awq_inv_scale == nullptr) return 1.0f;
    const hip_bfloat16* inv = static_cast<const hip_bfloat16*>(awq_inv_scale);
    return static_cast<float>(inv[col]);
}

__device__ inline void apply_awq_inv_scale_8(float w[8], const void* awq_inv_scale, int col) {
    if (awq_inv_scale == nullptr) return;
    const hip_bfloat16* inv = static_cast<const hip_bfloat16*>(awq_inv_scale);
    w[0] *= static_cast<float>(inv[col + 0]);
    w[1] *= static_cast<float>(inv[col + 1]);
    w[2] *= static_cast<float>(inv[col + 2]);
    w[3] *= static_cast<float>(inv[col + 3]);
    w[4] *= static_cast<float>(inv[col + 4]);
    w[5] *= static_cast<float>(inv[col + 5]);
    w[6] *= static_cast<float>(inv[col + 6]);
    w[7] *= static_cast<float>(inv[col + 7]);
}

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

template <typename T>
__device__ inline float supersonic_qwen35_conv4_state3_decode(
    const T* mixed_qkv,
    size_t mixed_offset,
    const T* prev_state,
    size_t state_offset,
    const T* weights,
    size_t weight_offset
) {
    const float w0 = supersonic_qwen35_to_float(weights[weight_offset]);
    const float w1 = supersonic_qwen35_to_float(weights[weight_offset + 1]);
    const float w2 = supersonic_qwen35_to_float(weights[weight_offset + 2]);
    const float w3 = supersonic_qwen35_to_float(weights[weight_offset + 3]);
    const float x0 = supersonic_qwen35_to_float(prev_state[state_offset]);
    const float x1 = supersonic_qwen35_to_float(prev_state[state_offset + 1]);
    const float x2 = supersonic_qwen35_to_float(prev_state[state_offset + 2]);
    const float x3 = supersonic_qwen35_to_float(mixed_qkv[mixed_offset]);
    return ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
}

template <>
__device__ inline float supersonic_qwen35_conv4_state3_decode<__half>(
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

        float running_max = -INFINITY;
        float denom = 0.0f;
        float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int local_dims[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int local_count = 0;
        for (int d = lane; d < head_dim && local_count < 8; d += warpSize) {
            local_dims[local_count++] = d;
        }

        // Warp-only online softmax. The previous shared+syncthreads path
        // issued two block barriers per key (8070 keys × 24 heads).
        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;
            const T* v_row = v_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += supersonic_qwen35_to_float(q_row[d]) *
                    supersonic_qwen35_to_float(k_row[d]);
            }
            const float score = supersonic_qwen35_wave_sum(partial) * scale;

            float prev_scale;
            float curr_scale;
            if (!isfinite(running_max)) {
                running_max = score;
                denom = 1.0f;
                prev_scale = 0.0f;
                curr_scale = 1.0f;
            } else {
                const float new_max = fmaxf(running_max, score);
                prev_scale = expf(running_max - new_max);
                curr_scale = expf(score - new_max);
                denom = denom * prev_scale + curr_scale;
                running_max = new_max;
            }

            if (curr_scale == 1.0f && prev_scale == 0.0f) {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = supersonic_qwen35_to_float(v_row[local_dims[i]]);
                }
            } else {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = local_acc[i] * prev_scale +
                        curr_scale * supersonic_qwen35_to_float(v_row[local_dims[i]]);
                }
            }
        }

        const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * inv_denom;
        }
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

        float kv_mem = 0.0f;
        float out_t = 0.0f;
        const float delta_scale = supersonic_qwen35_to_float(beta[beta_row]);
        const float v_t = supersonic_qwen35_to_float(value[value_row + v_idx]);
        int k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const float k0 = supersonic_qwen35_to_float(key[key_row + k_idx]);
            const float k1 = supersonic_qwen35_to_float(key[key_row + k_idx + 1]);
            const float k2 = supersonic_qwen35_to_float(key[key_row + k_idx + 2]);
            const float k3 = supersonic_qwen35_to_float(key[key_row + k_idx + 3]);
            float s0 = state[k_idx] * g_t;
            float s1 = state[k_idx + 1] * g_t;
            float s2 = state[k_idx + 2] * g_t;
            float s3 = state[k_idx + 3] * g_t;
            kv_mem += s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
            state[k_idx] = s0;
            state[k_idx + 1] = s1;
            state[k_idx + 2] = s2;
            state[k_idx + 3] = s3;
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
            kv_mem += state[k_idx] * supersonic_qwen35_to_float(key[key_row + k_idx]);
        }
        const float delta = (v_t - kv_mem) * delta_scale;
        k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const float k0 = supersonic_qwen35_to_float(key[key_row + k_idx]);
            const float k1 = supersonic_qwen35_to_float(key[key_row + k_idx + 1]);
            const float k2 = supersonic_qwen35_to_float(key[key_row + k_idx + 2]);
            const float k3 = supersonic_qwen35_to_float(key[key_row + k_idx + 3]);
            float s0 = state[k_idx] + k0 * delta;
            float s1 = state[k_idx + 1] + k1 * delta;
            float s2 = state[k_idx + 2] + k2 * delta;
            float s3 = state[k_idx + 3] + k3 * delta;
            state[k_idx] = s0;
            state[k_idx + 1] = s1;
            state[k_idx + 2] = s2;
            state[k_idx + 3] = s3;
            out_t += s0 * supersonic_qwen35_to_float(query[key_row + k_idx]) +
                s1 * supersonic_qwen35_to_float(query[key_row + k_idx + 1]) +
                s2 * supersonic_qwen35_to_float(query[key_row + k_idx + 2]) +
                s3 * supersonic_qwen35_to_float(query[key_row + k_idx + 3]);
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += supersonic_qwen35_to_float(key[key_row + k_idx]) * delta;
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
    T* out
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
        tid
    );
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
__global__ void supersonic_qwen35_matmul_rhs_transposed_tiled_kernel(
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
                s_lhs[lr][lc] = supersonic_qwen35_to_float(lhs[lhs_base + static_cast<size_t>(gr) * k + gc]);
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
                s_rhs[rr][rc] = supersonic_qwen35_to_float(rhs[rhs_base + static_cast<size_t>(gn) * k + gk]);
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
        out[out_idx] = supersonic_qwen35_from_float<T>(acc);
    }
}

// =============================================================================
// RDNA WMMA BF16 prefill matmul (wave32, 16x16x16 f32-accumulate), tiled.
//
// Computes: out[batch, m, n] = lhs[batch, m, k] × rhs[batch, n, k]^T
//
// Grid  : (ceil(n/64), ceil(m/64), batch_elems)
// Block : 128 threads = 4 wavefronts arranged 2x2
//
// Each block computes a 64x64 output tile. The four wavefronts cooperate
// to stage A (lhs) and B (rhs) source tiles through LDS, then each wave
// runs WMMAs over its 32x32 sub-region (a 2x2 grid of 16x16 sub-tiles).
// This amortizes the global loads of A and B by 4x each vs the previous
// one-wave-per-16x16-tile kernel, which was suffering ~100x read
// amplification for typical Qwen prefill shapes.
// =============================================================================

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__)
#define SUPERSONIC_HAS_WMMA_BF16 1
#define SUPERSONIC_HAS_WMMA_I8 1
#define SUPERSONIC_WMMA_GFX12 0
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define SUPERSONIC_HAS_WMMA_BF16 1
#define SUPERSONIC_HAS_WMMA_I8 1
#define SUPERSONIC_WMMA_GFX12 1
#endif

typedef short  supersonic_short16 __attribute__((ext_vector_type(16)));
typedef short  supersonic_short8  __attribute__((ext_vector_type(8)));
typedef float  supersonic_float8  __attribute__((ext_vector_type(8)));
typedef int    supersonic_i32x2  __attribute__((ext_vector_type(2)));
typedef int    supersonic_i32x4  __attribute__((ext_vector_type(4)));
typedef int    supersonic_i32x8  __attribute__((ext_vector_type(8)));

#ifdef SUPERSONIC_HAS_WMMA_BF16
__device__ __forceinline__ float supersonic_shfl_float8_at(
    supersonic_float8 values,
    int elem,
    int src_lane
) {
    float out = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float v = __shfl(values[i], src_lane, 32);
        if (i == elem) out = v;
    }
    return out;
}

__device__ __forceinline__ supersonic_float8 supersonic_wmma_bf16(
    supersonic_short16 a,
    supersonic_short16 b,
    supersonic_float8 acc
) {
#if SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x & 31;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;

    supersonic_short8 a12;
    supersonic_short8 b12;
    supersonic_float8 c12;

    // gfx12 A/B fragments interleave the two 8-row halves as 0..3,8..11
    // for lanes 0..15 and 4..7,12..15 for lanes 16..31.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = lane_group * 8 + i;
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        a12[i] = a[ab_row];
        b12[i] = b[ab_row];

        const int old_src_lane = ((row & 1) ? 16 : 0) + lane_col;
        c12[i] = supersonic_shfl_float8_at(acc, row >> 1, old_src_lane);
    }

    supersonic_float8 d12 =
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a12, b12, c12);
    supersonic_float8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int old_row = 2 * i + lane_group;
        const int new_src_lane = ((old_row >= 8) ? 16 : 0) + lane_col;
        out[i] = supersonic_shfl_float8_at(d12, old_row & 7, new_src_lane);
    }
    return out;
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
#endif
}

#if SUPERSONIC_WMMA_GFX12
__device__ __forceinline__ supersonic_float8 supersonic_wmma_bf16_gfx12_acc(
    supersonic_short16 a,
    supersonic_short16 b,
    supersonic_float8 acc12
) {
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 4;

    supersonic_short8 a12;
    supersonic_short8 b12;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        a12[i] = a[ab_row];
        b12[i] = b[ab_row];
    }

    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a12, b12, acc12);
}

__device__ __forceinline__ supersonic_float8 supersonic_wmma_bf16_gfx12_native_acc(
    supersonic_short8 a12,
    supersonic_short8 b12,
    supersonic_float8 acc12
) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a12, b12, acc12);
}
#endif
#endif

#ifdef SUPERSONIC_HAS_WMMA_I8
__device__ __forceinline__ int supersonic_shfl_i32x8_at(
    supersonic_i32x8 values,
    int elem,
    int src_lane
) {
    int out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int v = __shfl(values[i], src_lane, 32);
        if (i == elem) out = v;
    }
    return out;
}

__device__ __forceinline__ supersonic_i32x8 supersonic_wmma_i8_iu8(
    supersonic_i32x4 a,
    supersonic_i32x4 b,
    supersonic_i32x8 acc
) {
#if SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x & 31;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;

    supersonic_i32x2 a12;
    supersonic_i32x2 b12;
    supersonic_i32x8 c12;

    // Packed i8 uses the same A/B permutation at 4-element word granularity.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int packed_idx = lane_group + 2 * i;
        a12[i] = a[packed_idx];
        b12[i] = b[packed_idx];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = lane_group * 8 + i;
        const int old_src_lane = ((row & 1) ? 16 : 0) + lane_col;
        c12[i] = supersonic_shfl_i32x8_at(acc, row >> 1, old_src_lane);
    }

    supersonic_i32x8 d12 =
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a12, true, b12, c12, false);
    supersonic_i32x8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int old_row = 2 * i + lane_group;
        const int new_src_lane = ((old_row >= 8) ? 16 : 0) + lane_col;
        out[i] = supersonic_shfl_i32x8_at(d12, old_row & 7, new_src_lane);
    }
    return out;
#else
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, false);
#endif
}
#endif

__device__ inline uint16_t f32_to_f16_bits_rne(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

__device__ inline void ggml_k_dequant_16_bf16_bits_from_row(
    const uint8_t* row_data,
    int qtype,
    int col0,
    int cols,
    uint16_t out[16]
) {
    const int block = col0 / 256;
    const int inb = col0 - block * 256;

    if ((col0 & 15) != 0 || col0 + 15 >= cols || ((col0 + 15) / 256) != block) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            ggml_k_dequant_pair_bf16_bits_from_row(
                row_data, qtype, col0 + 2 * i, cols, out[2 * i], out[2 * i + 1]);
        }
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q8_0) {
        const uint8_t* b = row_data + (col0 / 32) * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
        const int idx = col0 & 31;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = f32_to_bf16_bits_rne_finite(d * static_cast<float>(qs[idx + i]));
        }
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q4_K) {
        const uint8_t* b = row_data + block * 144;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qs = b + 16;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[g * 32 + idx + i];
            const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
            out[i] = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(q) - dminv);
        }
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q5_K) {
        const uint8_t* b = row_data + block * 176;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const int hi_mask = sub != 0 ? (2 << (2 * g)) : (1 << (2 * g));
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = ql[g * 32 + idx + i];
            const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
            const int hi = (qh[idx + i] & hi_mask) ? 16 : 0;
            out[i] = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(lo + hi) - dminv);
        }
        return;
    }

    if (qtype == QWEN35_LOWBIT_GGML_Q6_K) {
        const uint8_t* b = row_data + block * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = ggml_f16_to_f32_unaligned(b + 208);
        const int half_idx = inb / 128;
        const int pos = inb - half_idx * 128;
        const int idx = pos % 32;
        int scale_idx = half_idx * 8 + idx / 16;
        const uint8_t* ql_base = ql + half_idx * 64 + idx;
        int qh_shift = 0;
        bool high_nibble = false;
        if (pos < 32) {
            qh_shift = 0;
        } else if (pos < 64) {
            ql_base += 32;
            qh_shift = 2;
            scale_idx += 2;
        } else if (pos < 96) {
            qh_shift = 4;
            scale_idx += 4;
            high_nibble = true;
        } else {
            ql_base += 32;
            qh_shift = 6;
            scale_idx += 6;
            high_nibble = true;
        }
        const uint8_t* qh_base = qh + half_idx * 32 + idx;
        const float ds = d * static_cast<float>(sc[scale_idx]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = high_nibble ? ((ql_base[i] >> 4) & 0x0F) : (ql_base[i] & 0x0F);
            const int hi = static_cast<int>(((qh_base[i] >> qh_shift) & 3) << 4);
            out[i] = f32_to_bf16_bits_rne_finite(ds * static_cast<float>(lo + hi - 32));
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = 0;
}

template <int QTYPE>
__device__ inline int ggml_k_row_bytes_fixed(int cols) {
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) return (cols / 32) * 34;
    const int blocks = cols / 256;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) return blocks * 144;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) return blocks * 176;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) return blocks * 210;
    return cols / 2;
}

template <bool TRUNC_DEQUANT>
__device__ inline uint16_t ggml_small_m_bf16_bits(float x) {
    if constexpr (TRUNC_DEQUANT) {
        return f32_to_bf16_bits_trunc(x);
    }
    return f32_to_bf16_bits_rne_finite(x);
}

template <int QTYPE>
__device__ inline int ggml_k_block_bytes_fixed() {
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) return 8 * 34;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) return 144;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) return 176;
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) return 210;
    return 0;
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__device__ inline void ggml_k_dequant_16_bf16_bits_from_block_fixed(
    const uint8_t* block_data,
    int inb,
    uint16_t out[16]
) {
    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) {
        const int subblock = inb / 32;
        const int idx = inb - subblock * 32;
        const uint8_t* b = block_data + subblock * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                d * static_cast<float>(qs[idx + i]));
        }
        return;
    }

    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) {
        const float d = ggml_f16_to_f32_unaligned(block_data);
        const float dmin = ggml_f16_to_f32_unaligned(block_data + 2);
        const uint8_t* sc = block_data + 4;
        const uint8_t* qs = block_data + 16;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[g * 32 + idx + i];
            const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
            out[i] = ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                ds * static_cast<float>(q) - dminv);
        }
        return;
    }

    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) {
        const float d = ggml_f16_to_f32_unaligned(block_data);
        const float dmin = ggml_f16_to_f32_unaligned(block_data + 2);
        const uint8_t* sc = block_data + 4;
        const uint8_t* qh = block_data + 16;
        const uint8_t* ql = block_data + 48;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int idx = inb % 32;
        const int j = 2 * g + sub;
        int scale = 0;
        int minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const int hi_mask = sub != 0 ? (2 << (2 * g)) : (1 << (2 * g));
        const float ds = d * static_cast<float>(scale);
        const float dminv = dmin * static_cast<float>(minv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = ql[g * 32 + idx + i];
            const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
            const int hi = (qh[idx + i] & hi_mask) ? 16 : 0;
            out[i] = ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                ds * static_cast<float>(lo + hi) - dminv);
        }
        return;
    }

    if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) {
        const uint8_t* ql = block_data;
        const uint8_t* qh = block_data + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(block_data + 192);
        const float d = ggml_f16_to_f32_unaligned(block_data + 208);
        const int half_idx = inb / 128;
        const int pos = inb - half_idx * 128;
        const int idx = pos % 32;
        int scale_idx = half_idx * 8 + idx / 16;
        const uint8_t* ql_base = ql + half_idx * 64 + idx;
        int qh_shift = 0;
        bool high_nibble = false;
        if (pos < 32) {
            qh_shift = 0;
        } else if (pos < 64) {
            ql_base += 32;
            qh_shift = 2;
            scale_idx += 2;
        } else if (pos < 96) {
            qh_shift = 4;
            scale_idx += 4;
            high_nibble = true;
        } else {
            ql_base += 32;
            qh_shift = 6;
            scale_idx += 6;
            high_nibble = true;
        }
        const uint8_t* qh_base = qh + half_idx * 32 + idx;
        const float ds = d * static_cast<float>(sc[scale_idx]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = high_nibble ? ((ql_base[i] >> 4) & 0x0F) : (ql_base[i] & 0x0F);
            const int hi = static_cast<int>(((qh_base[i] >> qh_shift) & 3) << 4);
            out[i] = ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                ds * static_cast<float>(lo + hi - 32));
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = 0;
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__device__ inline void ggml_k_dequant_16_bf16_bits_from_row_fixed(
    const uint8_t* row_data,
    int col0,
    int cols,
    uint16_t out[16]
) {
    const int block = col0 / 256;
    const int inb = col0 - block * 256;

    if ((col0 & 15) != 0 || col0 + 15 >= cols || ((col0 + 15) / 256) != block) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int kc = col0 + i;
            out[i] = kc < cols
                ? ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                      ggml_k_dequant_scalar_from_row(row_data, QTYPE, kc, cols))
                : 0;
        }
        return;
    }

    const uint8_t* b = row_data + block * ggml_k_block_bytes_fixed<QTYPE>();
    ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(b, inb, out);
}

#if SUPERSONIC_WMMA_GFX12
template <bool TRUNC_DEQUANT = true>
__device__ inline supersonic_short8 ggml_q8_0_dequant_bf16_gfx12_fragment(
    const uint8_t* block_data,
    int inb,
    int lane_group
) {
    const int subblock = inb / 32;
    const int idx = inb - subblock * 32;
    const uint8_t* b = block_data + subblock * 34;
    const float d = ggml_f16_to_f32_unaligned(b);
    const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);

    supersonic_short8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        out[i] = static_cast<short>(
            ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                d * static_cast<float>(qs[idx + ab_row])));
    }
    return out;
}

template <bool TRUNC_DEQUANT = true>
__device__ inline supersonic_short8 ggml_q4_k_dequant_bf16_gfx12_fragment(
    const uint8_t* block_data,
    int inb,
    int lane_group
) {
    const float d = ggml_f16_to_f32_unaligned(block_data);
    const float dmin = ggml_f16_to_f32_unaligned(block_data + 2);
    const uint8_t* sc = block_data + 4;
    const uint8_t* qs = block_data + 16;
    const int g = inb / 64;
    const int sub = (inb % 64) / 32;
    const int idx = inb % 32;
    const int j = 2 * g + sub;
    int scale = 0;
    int minv = 0;
    ggml_q4_k_scale_min(sc, j, scale, minv);
    const float ds = d * static_cast<float>(scale);
    const float dminv = dmin * static_cast<float>(minv);

    supersonic_short8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        const uint8_t qbyte = qs[g * 32 + idx + ab_row];
        const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        out[i] = static_cast<short>(
            ggml_small_m_bf16_bits<TRUNC_DEQUANT>(ds * static_cast<float>(q) - dminv));
    }
    return out;
}

template <bool TRUNC_DEQUANT = true>
__device__ inline supersonic_short8 ggml_q5_k_dequant_bf16_gfx12_fragment(
    const uint8_t* block_data,
    int inb,
    int lane_group
) {
    const float d = ggml_f16_to_f32_unaligned(block_data);
    const float dmin = ggml_f16_to_f32_unaligned(block_data + 2);
    const uint8_t* sc = block_data + 4;
    const uint8_t* qh = block_data + 16;
    const uint8_t* ql = block_data + 48;
    const int g = inb / 64;
    const int sub = (inb % 64) / 32;
    const int idx = inb % 32;
    const int j = 2 * g + sub;
    int scale = 0;
    int minv = 0;
    ggml_q4_k_scale_min(sc, j, scale, minv);
    const int hi_mask = sub != 0 ? (2 << (2 * g)) : (1 << (2 * g));
    const float ds = d * static_cast<float>(scale);
    const float dminv = dmin * static_cast<float>(minv);

    supersonic_short8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        const uint8_t qbyte = ql[g * 32 + idx + ab_row];
        const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        const int hi = (qh[idx + ab_row] & hi_mask) ? 16 : 0;
        out[i] = static_cast<short>(
            ggml_small_m_bf16_bits<TRUNC_DEQUANT>(ds * static_cast<float>(lo + hi) - dminv));
    }
    return out;
}

template <bool TRUNC_DEQUANT = true>
__device__ inline supersonic_short8 ggml_q6_k_dequant_bf16_gfx12_fragment(
    const uint8_t* block_data,
    int inb,
    int lane_group
) {
    const uint8_t* ql = block_data;
    const uint8_t* qh = block_data + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(block_data + 192);
    const float d = ggml_f16_to_f32_unaligned(block_data + 208);
    const int half_idx = inb / 128;
    const int pos = inb - half_idx * 128;
    const int idx = pos % 32;
    int scale_idx = half_idx * 8 + idx / 16;
    const uint8_t* ql_base = ql + half_idx * 64 + idx;
    int qh_shift = 0;
    bool high_nibble = false;
    if (pos < 32) {
        qh_shift = 0;
    } else if (pos < 64) {
        ql_base += 32;
        qh_shift = 2;
        scale_idx += 2;
    } else if (pos < 96) {
        qh_shift = 4;
        scale_idx += 4;
        high_nibble = true;
    } else {
        ql_base += 32;
        qh_shift = 6;
        scale_idx += 6;
        high_nibble = true;
    }
    const uint8_t* qh_base = qh + half_idx * 32 + idx;
    const float ds = d * static_cast<float>(sc[scale_idx]);

    supersonic_short8 out;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
        const int lo = high_nibble ? ((ql_base[ab_row] >> 4) & 0x0F) : (ql_base[ab_row] & 0x0F);
        const int hi = static_cast<int>(((qh_base[ab_row] >> qh_shift) & 3) << 4);
        out[i] = static_cast<short>(
            ggml_small_m_bf16_bits<TRUNC_DEQUANT>(ds * static_cast<float>(lo + hi - 32)));
    }
    return out;
}
#endif

// Tile dimensions.
constexpr int TILED_WMMA_BM = 64;   // output tile rows (m)
constexpr int TILED_WMMA_BN = 64;   // output tile cols (n)
constexpr int TILED_WMMA_BK = 64;   // K-step per outer iteration
// LDS row stride padded by 1 DWORD (2 BF16) to break the 32-bank-aligned
// access pattern; reduces WMMA operand gather bank conflicts.
constexpr int TILED_WMMA_LDS_STRIDE = TILED_WMMA_BK + 2;  // BF16 units

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_matmul_rhs_transposed_tiled_wmma_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,   // [batch, m, k]
    const hip_bfloat16* __restrict__ rhs,   // [batch, n, k] (virtually transposed)
    hip_bfloat16* __restrict__ out          // [batch, m, n]
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;       // 0..3
    const int wave_row = wid >> 1;       // 0 or 1
    const int wave_col = wid & 1;        // 0 or 1
    const int lane_row = lane & 15;      // 0..15 (duplicated across lane_half)
    const int lane_half = lane >> 4;     // 0 for lanes 0-15, 1 for lanes 16-31

    const int batch   = blockIdx.z;
    const int blk_row = blockIdx.y * TILED_WMMA_BM;
    const int blk_col = blockIdx.x * TILED_WMMA_BN;

    // LDS staging tiles. Stored as uint16_t; BF16 bit-patterns.
    __shared__ uint16_t s_lhs[TILED_WMMA_BM * TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_rhs[TILED_WMMA_BN * TILED_WMMA_LDS_STRIDE];

    const uint16_t* lhs_u16 = reinterpret_cast<const uint16_t*>(lhs);
    const uint16_t* rhs_u16 = reinterpret_cast<const uint16_t*>(rhs);

    const size_t lhs_batch_base = static_cast<size_t>(batch) * m * k;
    const size_t rhs_batch_base = static_cast<size_t>(batch) * n * k;

    // 2x2 sub-tile accumulators for this wave's 32x32 output region.
    supersonic_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    static_assert(TILED_WMMA_BK % 32 == 0, "BK must be a multiple of 32 (lane count)");
    constexpr int K_COL_HALVES = TILED_WMMA_BK / 32;  // 1 per 32 lanes
    constexpr int K_CHUNKS = TILED_WMMA_BK / 16;      // WMMA chunks of 16

    const int k_main = k & ~(TILED_WMMA_BK - 1);

    for (int kk0 = 0; kk0 < k_main; kk0 += TILED_WMMA_BK) {
        // --- Load A tile [BM=64, BK] into s_lhs ---
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gr < m) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + gk];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // --- Load B tile [BN=64, BK] into s_rhs ---
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gc < n) {
                    v = rhs_u16[rhs_batch_base + static_cast<size_t>(gc) * k + gk];
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        __syncthreads();

        // --- Compute: K_CHUNKS WMMA chunks of 16, 4 WMMAs each chunk ---
        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
        __syncthreads();
    }

    // K tail when k % BK != 0. Qwen hidden/intermediate dims are multiples of
    // 128 so this branch is unreachable in practice; keep it defensive.
    if (k_main != k) {
        const int k_rem = k - k_main;
        // Load partial K slab; zero-pad cols >= k_rem. Still full BK cols in LDS
        // so the compute loop below is uniform.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < m && col < k_rem) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + k_main + col];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
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
                if (gc < n && col < k_rem) {
                    v = rhs_u16[rhs_batch_base + static_cast<size_t>(gc) * k + k_main + col];
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
    }

    // Store 4 sub-tiles. RDNA3 wave32 WMMA acc layout:
    //   acc[i] for i in 0..7 → C[2*i + lane_half, lane_row]
    // relative to the sub-tile's (0,0) corner.
    const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const supersonic_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int out_row = sub_row_base + sub_row_off + 2 * i + lane_half;
                if (out_row < m) {
                    out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
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
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs; (void)out;
#endif
}

// RDNA3 WMMA BF16 prefill matmul for small M, one wavefront per 16x16 tile.
//
// The 64x64 tiled BF16 kernel is efficient for long prefill, but for
// draft blocks (M=16) it computes mostly out-of-range row tiles. This path
// keeps the same K-order and BF16 WMMA accumulate/output semantics while
// matching the small-M tile geometry used by the INT4-dequant path below.
__global__ void supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,   // [batch, m, k]
    const hip_bfloat16* __restrict__ rhs,   // [batch, n, k]
    hip_bfloat16* __restrict__ out          // [batch, m, n]
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;

    const int lhs_row_idx = tile_row + lane_row;
    const int rhs_row_idx = tile_col + lane_row;
    const bool lhs_in_range = lhs_row_idx < m;
    const bool rhs_in_range = rhs_row_idx < n;

    const size_t lhs_row_base = static_cast<size_t>(batch) * m * k +
                                static_cast<size_t>(lhs_row_idx) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * k +
                                static_cast<size_t>(rhs_row_idx) * k;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint16_t* rhs_row_u16 = reinterpret_cast<const uint16_t*>(rhs) + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short16 a;
        supersonic_short16 b;
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(lhs_row_u16[kk + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(rhs_row_u16[kk + i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    if (k_main != k) {
        supersonic_short16 a = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        supersonic_short16 b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < k) a[i] = static_cast<short>(lhs_row_u16[kc]);
            }
        }
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < k) b[i] = static_cast<short>(rhs_row_u16[kc]);
            }
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    const int out_col = tile_col + lane_row;
    if (out_col < n) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + 2 * i + lane_half;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs; (void)out;
#endif
}

template <bool HOT_EXACT = false>
__global__ void supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_gfx12_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const hip_bfloat16* __restrict__ rhs,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;

    const int lhs_row_idx = tile_row + lane_row;
    const int rhs_row_idx = tile_col + lane_row;
    const bool lhs_in_range = HOT_EXACT || lhs_row_idx < m;
    const bool rhs_in_range = HOT_EXACT || rhs_row_idx < n;

    const size_t lhs_row_base = static_cast<size_t>(batch) * m * k +
                                static_cast<size_t>(lhs_row_idx) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * k +
                                static_cast<size_t>(rhs_row_idx) * k;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint16_t* rhs_row_u16 = reinterpret_cast<const uint16_t*>(rhs) + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short8 a;
        supersonic_short8 b;
        if constexpr (HOT_EXACT) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }
        } else if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = 0;
        }
        if constexpr (HOT_EXACT) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                b[i] = static_cast<short>(rhs_row_u16[kk + ab_row]);
            }
        } else if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                b[i] = static_cast<short>(rhs_row_u16[kk + ab_row]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) b[i] = 0;
        }
        acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
    }

    if constexpr (!HOT_EXACT) {
        if (k_main != k) {
            supersonic_short8 a = {0, 0, 0, 0, 0, 0, 0, 0};
            supersonic_short8 b = {0, 0, 0, 0, 0, 0, 0, 0};
            if (lhs_in_range) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    int kc = k_main + ab_row;
                    if (kc < k) a[i] = static_cast<short>(lhs_row_u16[kc]);
                }
            }
            if (rhs_in_range) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    int kc = k_main + ab_row;
                    if (kc < k) b[i] = static_cast<short>(rhs_row_u16[kc]);
                }
            }
            acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
        }
    }

    const int out_col = tile_col + lane_row;
    if constexpr (HOT_EXACT) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + lane_group * 8 + i;
            out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                hip_bfloat16(acc[i]);
        }
    } else if (out_col < n) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + lane_group * 8 + i;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs; (void)out;
#endif
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
__global__ void supersonic_qwen35_matmul_fp8_dequant_kernel(
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
                s_lhs[lr][lc] = supersonic_qwen35_to_float(
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
                float s = supersonic_qwen35_to_float(scale[sr * scale_cols + sc]);
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
        out[out_idx] = supersonic_qwen35_from_float<T>(acc);
    }
}

// =============================================================================
// RDNA3 WMMA FP8-dequant prefill matmul (wave32, 16x16x16 f32-accumulate),
// shared-memory tiled. Uses the same tile shape as the BF16 WMMA kernel
// above (64x64 output, 4 waves in 2x2, BK=64 staged through LDS).
//
// B-tile load reads FP8 E4M3 bytes, multiplies by the per-block scale, and
// rounds to BF16 before storing into LDS. The bridge gates this path on
// `block_size % 64 == 0` so every BK-aligned K slab stays inside one scale
// block (shipped FP8 bakes use block_size=128). Within one outer iter,
// every row and every K-value in this block's 64x64 tile share one scalar
// scale — BN=64 <= block_size=128 and blk_col is a multiple of 64, so the
// 64 rows also lie inside a single scale_row.
//
// Bit-exactness note: the existing scalar FP8 kernel also rounds each
// dequantized value to BF16 before accumulating, so the precision path is
// equivalent (BF16 × BF16 → F32 WMMA accumulate here, vs F32 × BF16-rounded-F32
// → F32 scalar accumulate there).
// =============================================================================

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_matmul_fp8_dequant_wmma_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,   // [batch, m, k]
    const uint8_t* __restrict__ rhs_fp8,    // [batch, n, k] FP8 E4M3
    const hip_bfloat16* __restrict__ scale, // [n/block, k/block] BF16
    int block_size,
    hip_bfloat16* __restrict__ out          // [batch, m, n]
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;
    const int wave_row = wid >> 1;
    const int wave_col = wid & 1;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;

    const int batch   = blockIdx.z;
    const int blk_row = blockIdx.y * TILED_WMMA_BM;
    const int blk_col = blockIdx.x * TILED_WMMA_BN;

    __shared__ uint16_t s_lhs[TILED_WMMA_BM * TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_rhs[TILED_WMMA_BN * TILED_WMMA_LDS_STRIDE];

    const uint16_t* lhs_u16 = reinterpret_cast<const uint16_t*>(lhs);

    const int scale_cols = (k + block_size - 1) / block_size;
    const size_t lhs_batch_base = static_cast<size_t>(batch) * m * k;
    const size_t rhs_batch_base = static_cast<size_t>(batch) * n * k;

    supersonic_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    static_assert(TILED_WMMA_BK % 32 == 0, "BK must be a multiple of 32");
    constexpr int K_COL_HALVES = TILED_WMMA_BK / 32;
    constexpr int K_CHUNKS     = TILED_WMMA_BK / 16;

    const int k_main = k & ~(TILED_WMMA_BK - 1);

    // Per-tile scale (constant across all rows and all K in this BK slab
    // when block_size >= max(BN, BK) and block_size % both).
    // Row index for the scale table is just blk_col / block_size since BN <= block_size.
    const int scale_row = blk_col / block_size;

    for (int kk0 = 0; kk0 < k_main; kk0 += TILED_WMMA_BK) {
        // Fetch the single scale for this tile's K slab. All 64 rows share
        // the same scale_row; all BK K-values share the same scale_col.
        float tile_scale = 0.f;
        if (blk_col < n) {
            int scale_col = kk0 / block_size;
            tile_scale = static_cast<float>(scale[scale_row * scale_cols + scale_col]);
        }

        // Load A (lhs) tile BF16.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gr < m) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + gk];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // Load + dequantize B (rhs_fp8) tile (FP8 E4M3 -> BF16 via tile_scale).
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t stored = 0;
                if (gc < n) {
                    uint8_t fp8_byte = rhs_fp8[rhs_batch_base + static_cast<size_t>(gc) * k + gk];
                    float fp8_val = fp8_e4m3_to_float(fp8_byte);
                    float v = bf16_round_rne_f32_finite(fp8_val * tile_scale);
                    uint32_t bits;
                    __builtin_memcpy(&bits, &v, 4);
                    stored = static_cast<uint16_t>(bits >> 16);
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = stored;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
        __syncthreads();
    }

    // Defensive K-tail (Qwen k is multiple of 128, so unreachable with BK=64).
    if (k_main != k) {
        const int k_rem = k - k_main;
        float tile_scale = 0.f;
        if (blk_col < n) {
            int scale_col = k_main / block_size;
            tile_scale = static_cast<float>(scale[scale_row * scale_cols + scale_col]);
        }
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < m && col < k_rem) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + k_main + col];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t stored = 0;
                if (gc < n && col < k_rem) {
                    uint8_t fp8_byte = rhs_fp8[rhs_batch_base + static_cast<size_t>(gc) * k + k_main + col];
                    float fp8_val = fp8_e4m3_to_float(fp8_byte);
                    float v = bf16_round_rne_f32_finite(fp8_val * tile_scale);
                    uint32_t bits;
                    __builtin_memcpy(&bits, &v, 4);
                    stored = static_cast<uint16_t>(bits >> 16);
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = stored;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
    }

    // Store 4 sub-tiles.
    const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const supersonic_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int out_row = sub_row_base + sub_row_off + 2 * i + lane_half;
                if (out_row < m) {
                    out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
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
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs_fp8; (void)scale; (void)block_size; (void)out;
#endif
}

// INT4 dequant tiled matmul for prefill.
// out [batch, m, n] = lhs [batch, m, k] × dequant(rhs_int4 [batch, n, k/2])^T
// rhs_int4 is packed INT4 (2 weights per byte, low nibble = even k index).
// scale/zero are [n/group_size, k/group_size] BF16.
#define INT4_TILE_M 16
#define INT4_TILE_N 16
#define INT4_TILE_K 32
template <typename T>
__global__ void supersonic_qwen35_matmul_int4_dequant_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const T* __restrict__ lhs,             // [batch, m, k] BF16
    const uint8_t* __restrict__ rhs,       // [batch, n, k/2] packed INT4
    const T* __restrict__ scale,           // [n/group_size, k/group_size] BF16
    const T* __restrict__ zero,            // [n/group_size, k/group_size] BF16
    const T* __restrict__ awq_inv_scale,   // optional [k] BF16, nullptr for GPTQ/HQQ
    int group_size,
    int quant_type,
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
    const int row_bytes = qwen35_lowbit_native_int4(quant_type)
        ? k / 2
        : ggml_k_row_bytes(quant_type, k);
    const size_t rhs_base = static_cast<size_t>(batch_idx) * n * row_bytes;

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
                s_lhs[lr][lc] = supersonic_qwen35_to_float(
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
                float w = 0.0f;
                if (qwen35_lowbit_native_int4(quant_type)) {
                    // Unpack INT4 nibble
                    int byte_idx = global_k / 2;
                    uint8_t packed_byte = rhs[
                        rhs_base + static_cast<size_t>(global_n) * row_bytes + byte_idx];
                    int nibble = (global_k & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
                    // Dequant: (nibble - zero) * scale
                    int sr = global_n / group_size;
                    int sc = global_k / group_size;
                    int si = sr * scale_cols + sc;
                    float s = supersonic_qwen35_to_float(scale[si]);
                    float z = supersonic_qwen35_to_float(zero[si]);
                    // Round through BF16 so prefill matches the decode megakernel's
                    // dequant path and the Python GPTQ reference (`bf16(q*s - zf*s)`).
                    w = bf16_round_rne_f32_finite(static_cast<float>(nibble) * s - z * s);
                } else {
                    w = ggml_k_dequant_scalar(rhs + rhs_base, quant_type, global_n, global_k, k);
                }
                if (awq_inv_scale != nullptr) {
                    w *= supersonic_qwen35_to_float(awq_inv_scale[global_k]);
                }
                s_rhs[rr][rc] = w;
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
        out[out_idx] = supersonic_qwen35_from_float<T>(acc);
    }
}

// =============================================================================
// RDNA3 WMMA INT4-dequant prefill matmul (wave32, 16x16x16 f32-accumulate),
// one-wavefront-per-16x16-tile. Kept as a small-M fallback since the tiled
// version below suffers proportionally larger tile-overhang waste (4x) when
// M is much smaller than the 64-row tile, and INT4 has little bandwidth to
// save by tiling (~4x less than BF16 packed).
//
//   lhs  [batch, m, k]          BF16
//   rhs  [batch, n, k/2]        packed INT4 (low nibble = even k index)
//   scl  [n/group, k/group]     BF16 scale
//   zro  [n/group, k/group]     BF16 zero
//   out  [batch, m, n]          BF16
// =============================================================================

__global__ void supersonic_qwen35_matmul_int4_dequant_wmma_small_m_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    const hip_bfloat16* __restrict__ scale,
    const hip_bfloat16* __restrict__ zero,
    const hip_bfloat16* __restrict__ awq_inv_scale,
    int group_size,
    int quant_type,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;

    const int lhs_row_idx = tile_row + lane_row;
    const int rhs_row_idx = tile_col + lane_row;
    const bool lhs_in_range = lhs_row_idx < m;
    const bool rhs_in_range = rhs_row_idx < n;

    const bool native_int4 = qwen35_lowbit_native_int4(quant_type);
    const int row_bytes = native_int4 ? (k / 2) : ggml_k_row_bytes(quant_type, k);
    const int scale_cols = (k + group_size - 1) / group_size;

    const size_t lhs_row_base = static_cast<size_t>(batch) * m * k +
                                static_cast<size_t>(lhs_row_idx) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t*  rhs_row_u8  = rhs_int4 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short16 a;
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = static_cast<short>(lhs_row_u16[kk + i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        supersonic_short16 b;
        if (rhs_in_range) {
            if (native_int4) {
                const int group_id = kk / group_size;
                const int scale_idx = rhs_row_idx / group_size * scale_cols + group_id;
                const float s = static_cast<float>(scale[scale_idx]);
                const float z = static_cast<float>(zero[scale_idx]);
                const float zs = z * s;
                const int byte_base = kk >> 1;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint8_t packed = rhs_row_u8[byte_base + i];
                    int n0 = packed & 0xF;
                    int n1 = (packed >> 4) & 0xF;
                    uint16_t bits0 = f32_to_bf16_bits_rne_finite(static_cast<float>(n0) * s - zs);
                    uint16_t bits1 = f32_to_bf16_bits_rne_finite(static_cast<float>(n1) * s - zs);
                    if (awq_inv_scale != nullptr) {
                        bits0 = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits0) * static_cast<float>(awq_inv_scale[kk + 2 * i]));
                        bits1 = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits1) * static_cast<float>(awq_inv_scale[kk + 2 * i + 1]));
                    }
                    b[2 * i]     = static_cast<short>(bits0);
                    b[2 * i + 1] = static_cast<short>(bits1);
                }
            } else {
                uint16_t bits[16];
                ggml_k_dequant_16_bf16_bits_from_row(rhs_row_u8, quant_type, kk, k, bits);
                if (awq_inv_scale != nullptr) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        bits[i] = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits[i]) * static_cast<float>(awq_inv_scale[kk + i]));
                    }
                }
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    b[i] = static_cast<short>(bits[i]);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = supersonic_wmma_bf16(a, b, acc);
    }
    if (k_main != k) {
        supersonic_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        supersonic_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < k) a[i] = static_cast<short>(lhs_row_u16[kc]);
            }
        }
        if (rhs_in_range) {
            if (native_int4) {
                const int group_id0 = k_main / group_size;
                const int scale_idx = rhs_row_idx / group_size * scale_cols + group_id0;
                const float s = static_cast<float>(scale[scale_idx]);
                const float z = static_cast<float>(zero[scale_idx]);
                const float zs = z * s;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int kc = k_main + i;
                    if (kc < k) {
                        uint8_t packed = rhs_row_u8[kc >> 1];
                        int nib = (kc & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                        uint16_t bits = f32_to_bf16_bits_rne_finite(static_cast<float>(nib) * s - zs);
                        if (awq_inv_scale != nullptr) {
                            bits = f32_to_bf16_bits_trunc(
                                bf16_bits_to_f32(bits) * static_cast<float>(awq_inv_scale[kc]));
                        }
                        b[i] = static_cast<short>(bits);
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int kc = k_main + i;
                    if (kc < k) {
                        uint16_t bits = f32_to_bf16_bits_rne_finite(
                            ggml_k_dequant_scalar_from_row(rhs_row_u8, quant_type, kc, k));
                        if (awq_inv_scale != nullptr) {
                            bits = f32_to_bf16_bits_trunc(
                                bf16_bits_to_f32(bits) * static_cast<float>(awq_inv_scale[kc]));
                        }
                        b[i] = static_cast<short>(bits);
                    }
                }
            }
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    const int out_col = tile_col + lane_row;
    if (out_col < n) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + 2 * i + lane_half;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)scale; (void)zero; (void)group_size;
    (void)quant_type; (void)out;
#endif
}

__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_quantize_mmq_q8_1_kernel(
    size_t batch_elems,
    int m,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    int quant_type,
    uint8_t* __restrict__ out
) {
    constexpr int Q8_BLOCK = 128;
    constexpr int Q8_BLOCK_BYTES = 144;
    const int lane = threadIdx.x;
    const int group = lane >> 3;
    const int group_lane = lane & 7;
    const int k_block = blockIdx.x;
    const int row = blockIdx.y;
    const int batch = blockIdx.z;

    if (row >= m || static_cast<size_t>(batch) >= batch_elems) {
        return;
    }

    const int k0 = k_block * Q8_BLOCK + lane * 4;
    const size_t lhs_base = (static_cast<size_t>(batch) * m + row) * static_cast<size_t>(k);

    float x0 = 0.f;
    float x1 = 0.f;
    float x2 = 0.f;
    float x3 = 0.f;
    if (k0 + 3 < k) {
        x0 = static_cast<float>(lhs[lhs_base + k0 + 0]);
        x1 = static_cast<float>(lhs[lhs_base + k0 + 1]);
        x2 = static_cast<float>(lhs[lhs_base + k0 + 2]);
        x3 = static_cast<float>(lhs[lhs_base + k0 + 3]);
    }

    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    float sum = x0 + x1 + x2 + x3;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset));
        sum += __shfl_xor(sum, offset);
    }

    const float d = amax > 0.f ? amax / 127.f : 0.f;
    const float d_inv = amax > 0.f ? 127.f / amax : 0.f;
    const int q0 = max(-127, min(127, static_cast<int>(nearbyintf(x0 * d_inv))));
    const int q1 = max(-127, min(127, static_cast<int>(nearbyintf(x1 * d_inv))));
    const int q2 = max(-127, min(127, static_cast<int>(nearbyintf(x2 * d_inv))));
    const int q3 = max(-127, min(127, static_cast<int>(nearbyintf(x3 * d_inv))));

    const int blocks_per_row = (k + Q8_BLOCK - 1) / Q8_BLOCK;
    // Match ggml/llama MMQ's activation-side tensor order:
    // [batch][k_block][row][block_q8_1_mmq]. The block contents stay as the
    // 144-byte q8_1 MMQ struct, but rows for the same K block are adjacent.
    const size_t out_block =
        (static_cast<size_t>(batch) * blocks_per_row * m +
         static_cast<size_t>(k_block) * m + row) *
        Q8_BLOCK_BYTES;
    int8_t* qs = reinterpret_cast<int8_t*>(out + out_block + 16);
    qs[lane * 4 + 0] = static_cast<int8_t>(q0);
    qs[lane * 4 + 1] = static_cast<int8_t>(q1);
    qs[lane * 4 + 2] = static_cast<int8_t>(q2);
    qs[lane * 4 + 3] = static_cast<int8_t>(q3);

    if (group_lane == 0) {
        const bool d4_layout = quant_type == QWEN35_LOWBIT_GGML_Q6_K;
        if (d4_layout) {
            float* scales = reinterpret_cast<float*>(out + out_block);
            scales[group] = d;
        } else {
            uint16_t* ds = reinterpret_cast<uint16_t*>(out + out_block);
            ds[group * 2 + 0] = f32_to_f16_bits_rne(d);
            ds[group * 2 + 1] = f32_to_f16_bits_rne(sum);
        }
    }
}

__device__ __forceinline__ int supersonic_get_int_b2(const void* ptr, int i32) {
    const uint16_t* x16 = reinterpret_cast<const uint16_t*>(ptr);
    int x32 = static_cast<int>(x16[2 * i32 + 0]);
    x32 |= static_cast<int>(x16[2 * i32 + 1]) << 16;
    return x32;
}

__device__ __forceinline__ int supersonic_sub_i8x4(int packed, int sub) {
    uint32_t x = static_cast<uint32_t>(packed);
    uint32_t out = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int v = static_cast<int>((x >> (8 * i)) & 0xffu) - sub;
        out |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(v))) << (8 * i);
    }
    return static_cast<int>(out);
}

template <bool HOT_EXACT_N = false>
__device__ __forceinline__ void supersonic_mmq_load_q6_k_tile_128(
    const uint8_t* __restrict__ rhs_q6,
    int* __restrict__ tile_x,
    int batch,
    int tile_n,
    int n,
    int k,
    int q6_block
) {
    constexpr int MMQ_Y = 128;
    constexpr int MMQ_TILE_NE_K = 32;
    constexpr int QI6_K = 32;
    constexpr int QR6_K = 2;
    constexpr int MMQ_ITER_K = 256;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K =
        2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;
    constexpr int Q6_BLOCK_BYTES = 210;
    constexpr int NWARPS = 8;
    constexpr int WARP_SIZE = 32;

    int* x_qs = tile_x;
    float* x_df = reinterpret_cast<float*>(x_qs + 2 * MMQ_TILE_NE_K);
    int* x_sc = reinterpret_cast<int*>(x_df + MMQ_TILE_NE_K / QI6_K);
    const int row_bytes = (k / 256) * Q6_BLOCK_BYTES;
    const size_t batch_base =
        static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = WARP_SIZE / threads_per_row;
    const int txi = threadIdx.x % threads_per_row;
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
        const int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        const int row = tile_n + i;

        int q0 = 0;
        int q1 = 0;
        if constexpr (HOT_EXACT_N) {
            const uint8_t* b = rhs_q6 + batch_base +
                               static_cast<size_t>(row) * row_bytes +
                               static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
            const int ql = supersonic_get_int_b2(b, txi);
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;
            const int qh = supersonic_get_int_b2(
                b + 128, (QI6_K / 4) * (txi / (QI6_K / 2)) + txi % (QI6_K / 4));
            const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
            const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;
            q0 = supersonic_sub_i8x4(ql0 | qh0, 32);
            q1 = supersonic_sub_i8x4(ql1 | qh1, 32);
        } else if (row < n) {
            const uint8_t* b = rhs_q6 + batch_base +
                               static_cast<size_t>(row) * row_bytes +
                               static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
            const int ql = supersonic_get_int_b2(b, txi);
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;
            const int qh = supersonic_get_int_b2(
                b + 128, (QI6_K / 4) * (txi / (QI6_K / 2)) + txi % (QI6_K / 4));
            const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
            const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;
            q0 = supersonic_sub_i8x4(ql0 | qh0, 32);
            q1 = supersonic_sub_i8x4(ql1 | qh1, 32);
        }
        const int kq0 = 2 * txi - txi % (QI6_K / 2) + 0;
        const int kq1 = 2 * txi - txi % (QI6_K / 2) + QI6_K / 2;
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq0] = q0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq1] = q1;
    }

    {
        const int i = threadIdx.y * WARP_SIZE + threadIdx.x;
        if (i < MMQ_Y) {
            const int row = tile_n + i;
            float d = 0.0f;
            if constexpr (HOT_EXACT_N) {
                const uint8_t* b = rhs_q6 + batch_base + static_cast<size_t>(row) * row_bytes +
                                   static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
                d = ggml_f16_to_f32_unaligned(b + 208);
            } else if (row < n) {
                const uint8_t* b = rhs_q6 + batch_base + static_cast<size_t>(row) * row_bytes +
                                   static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
                d = ggml_f16_to_f32_unaligned(b + 208);
            }
            x_df[i * MMQ_MMA_TILE_X_K_Q6_K] = d;
        }
    }

    constexpr int rows_per_warp = WARP_SIZE / 4;
    for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp) {
        const int i = (i0 + threadIdx.y * rows_per_warp +
                       threadIdx.x / (MMQ_TILE_NE_K / 8)) % MMQ_Y;
        const int row = tile_n + i;
        const int scale_word = threadIdx.x % (MMQ_TILE_NE_K / 8);
        int scales = 0;
        if constexpr (HOT_EXACT_N) {
            const uint8_t* b = rhs_q6 + batch_base + static_cast<size_t>(row) * row_bytes +
                               static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
            scales = supersonic_get_int_b2(b + 192, scale_word);
        } else if (row < n) {
            const uint8_t* b = rhs_q6 + batch_base + static_cast<size_t>(row) * row_bytes +
                               static_cast<size_t>(q6_block) * Q6_BLOCK_BYTES;
            scales = supersonic_get_int_b2(b + 192, scale_word);
        }
        x_sc[i * MMQ_MMA_TILE_X_K_Q6_K + scale_word] = scales;
    }
}

template <bool HOT_EXACT_M16 = false>
__device__ __forceinline__ void supersonic_mmq_load_q8_1_tile_16(
    const uint8_t* __restrict__ q8,
    int* __restrict__ tile_y,
    int batch,
    int tile_m,
    int m,
    int k,
    int q8_block
) {
    constexpr int MMQ_X = 16;
    constexpr int MMQ_TILE_Y_K = 36;
    constexpr int Q8_BLOCK_BYTES = 144;
    const int blocks_per_row = k / 128;

    if constexpr (HOT_EXACT_M16) {
        const int tid = threadIdx.y * 32 + threadIdx.x;
        const size_t batch_q8_base =
            static_cast<size_t>(batch) * blocks_per_row * m +
            static_cast<size_t>(q8_block) * m;

        int l = tid;
        int row = l / MMQ_TILE_Y_K;
        int word = l - row * MMQ_TILE_Y_K;
        size_t block = (batch_q8_base + static_cast<size_t>(tile_m + row)) * Q8_BLOCK_BYTES;
        tile_y[l] = reinterpret_cast<const int*>(q8 + block)[word];

        l = tid + 8 * 32;
        row = l / MMQ_TILE_Y_K;
        word = l - row * MMQ_TILE_Y_K;
        block = (batch_q8_base + static_cast<size_t>(tile_m + row)) * Q8_BLOCK_BYTES;
        tile_y[l] = reinterpret_cast<const int*>(q8 + block)[word];

        if (tid < MMQ_X * MMQ_TILE_Y_K - 2 * 8 * 32) {
            l = tid + 2 * 8 * 32;
            row = l / MMQ_TILE_Y_K;
            word = l - row * MMQ_TILE_Y_K;
            block = (batch_q8_base + static_cast<size_t>(tile_m + row)) * Q8_BLOCK_BYTES;
            tile_y[l] = reinterpret_cast<const int*>(q8 + block)[word];
        }
        return;
    }

    for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += 8 * 32) {
        const int l = l0 + threadIdx.y * 32 + threadIdx.x;
        if (l >= MMQ_X * MMQ_TILE_Y_K) {
            continue;
        }
        const int row = l / MMQ_TILE_Y_K;
        const int word = l - row * MMQ_TILE_Y_K;
        const int global_row = tile_m + row;
        int v = 0;
        if (global_row < m) {
            const size_t block =
                (static_cast<size_t>(batch) * blocks_per_row * m +
                 static_cast<size_t>(q8_block) * m + global_row) *
                Q8_BLOCK_BYTES;
            v = reinterpret_cast<const int*>(q8 + block)[word];
        }
        tile_y[l] = v;
    }
}

__device__ __forceinline__ void supersonic_mmq_i8_wmma_16x4(
    int c[8],
    const int a[4],
    const int b[4]
) {
#ifdef SUPERSONIC_HAS_WMMA_I8
    supersonic_i32x8* acc = reinterpret_cast<supersonic_i32x8*>(c);
    const supersonic_i32x4* avec = reinterpret_cast<const supersonic_i32x4*>(a);
    const supersonic_i32x4* bvec = reinterpret_cast<const supersonic_i32x4*>(b);
    acc[0] = supersonic_wmma_i8_iu8(avec[0], bvec[0], acc[0]);
#else
    (void)c; (void)a; (void)b;
#endif
}

#if SUPERSONIC_WMMA_GFX12
__device__ __forceinline__ supersonic_i32x8 supersonic_mmq_i8_wmma_16x4_gfx12_native(
    const int a[4],
    const int b[4]
) {
#ifdef SUPERSONIC_HAS_WMMA_I8
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 4;

    supersonic_i32x2 a12;
    supersonic_i32x2 b12;
    supersonic_i32x8 c12 = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int packed_idx = lane_group + 2 * i;
        a12[i] = a[packed_idx];
        b12[i] = b[packed_idx];
    }

    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true, a12, true, b12, c12, false);
#else
    (void)a; (void)b;
    return {0, 0, 0, 0, 0, 0, 0, 0};
#endif
}
#endif

__device__ __forceinline__ void supersonic_mmq_vec_dot_q6_k_q8_1_wmma(
    const int* __restrict__ tile_x,
    const int* __restrict__ tile_y,
    float sum[8],
    int k00
) {
    constexpr int MMQ_TILE_NE_K = 32;
    constexpr int MMQ_TILE_Y_K = 36;
    constexpr int QI6_K = 32;
    constexpr int QI8_1 = 8;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K =
        2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;

    const int* x_qs = tile_x;
    const float* x_df = reinterpret_cast<const float*>(x_qs + 2 * MMQ_TILE_NE_K);
    const int* x_sc = reinterpret_cast<const int*>(x_df + MMQ_TILE_NE_K / QI6_K);
    const int* y_qs = tile_y + 4;
    const float* y_df = reinterpret_cast<const float*>(tile_y);
    const int lane_row = threadIdx.x & 15;
    const int lane_half = threadIdx.x >> 4;
    const int i0 = threadIdx.y * 16;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;
        int a[4];
        int b[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            a[l] = x_qs[(i0 + lane_row) * MMQ_MMA_TILE_X_K_Q6_K + k0 + l];
            b[l] = y_qs[lane_row * MMQ_TILE_Y_K + k01 + l];
        }

        int c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        supersonic_mmq_i8_wmma_16x4(c, a, b);

        const int j = lane_row;
        const float dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = i0 + 2 * l + lane_half;
            const int8_t* sc =
                reinterpret_cast<const int8_t*>(x_sc + i * MMQ_MMA_TILE_X_K_Q6_K + k00 / 16);
            const float dA = x_df[i * MMQ_MMA_TILE_X_K_Q6_K];
            sum[l] += static_cast<float>(c[l]) * static_cast<float>(sc[k01 / 4]) * dA * dB;
        }
    }
}

#if SUPERSONIC_WMMA_GFX12
__device__ __forceinline__ void supersonic_mmq_vec_dot_q6_k_q8_1_wmma_gfx12_native(
    const int* __restrict__ tile_x,
    const int* __restrict__ tile_y,
    float sum[8],
    int k00
) {
    constexpr int MMQ_TILE_NE_K = 32;
    constexpr int MMQ_TILE_Y_K = 36;
    constexpr int QI6_K = 32;
    constexpr int QI8_1 = 8;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K =
        2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;

    const int* x_qs = tile_x;
    const float* x_df = reinterpret_cast<const float*>(x_qs + 2 * MMQ_TILE_NE_K);
    const int* x_sc = reinterpret_cast<const int*>(x_df + MMQ_TILE_NE_K / QI6_K);
    const int* y_qs = tile_y + 4;
    const float* y_df = reinterpret_cast<const float*>(tile_y);
    const int lane = threadIdx.x & 31;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int i0 = threadIdx.y * 16;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;
        int a[4];
        int b[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            a[l] = x_qs[(i0 + lane_row) * MMQ_MMA_TILE_X_K_Q6_K + k0 + l];
            b[l] = y_qs[lane_row * MMQ_TILE_Y_K + k01 + l];
        }

        const supersonic_i32x8 c = supersonic_mmq_i8_wmma_16x4_gfx12_native(a, b);

        const int j = lane_row;
        const float dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = i0 + lane_group * 8 + l;
            const int8_t* sc =
                reinterpret_cast<const int8_t*>(x_sc + i * MMQ_MMA_TILE_X_K_Q6_K + k00 / 16);
            const float dA = x_df[i * MMQ_MMA_TILE_X_K_Q6_K];
            sum[l] += static_cast<float>(c[l]) * static_cast<float>(sc[k01 / 4]) * dA * dB;
        }
    }
}

#endif

template <bool HOT_EXACT = false, bool HAS_RESIDUAL = false>
__global__ __launch_bounds__(256, 2)
void supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const uint8_t* __restrict__ q8,
    const uint8_t* __restrict__ rhs_q6,
    const hip_bfloat16* __restrict__ residual,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_I8
    constexpr int MMQ_X = 16;
    constexpr int MMQ_Y = 128;
    constexpr int MMQ_TILE_Y_K = 36;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K = 76;
    constexpr int PADDED_TILE_Y_INTS = 768;

    const int tile_n = blockIdx.x * MMQ_Y;
    const int tile_m = blockIdx.y * MMQ_X;
    const int batch = blockIdx.z;
    if (static_cast<size_t>(batch) >= batch_elems) {
        return;
    }

    extern __shared__ int supersonic_mmq_smem[];
    int* tile_y = supersonic_mmq_smem;
    int* tile_x = supersonic_mmq_smem + PADDED_TILE_Y_INTS;
    float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int q6_blocks = k / 256;
    for (int block = 0; block < q6_blocks; ++block) {
        supersonic_mmq_load_q6_k_tile_128<HOT_EXACT>(
            rhs_q6, tile_x, batch, tile_n, n, k, block);
        supersonic_mmq_load_q8_1_tile_16<HOT_EXACT>(
            q8, tile_y, batch, tile_m, m, k, 2 * block);
        __syncthreads();
        supersonic_mmq_vec_dot_q6_k_q8_1_wmma(tile_x, tile_y, sum, 0);
        __syncthreads();

        supersonic_mmq_load_q8_1_tile_16<HOT_EXACT>(
            q8, tile_y, batch, tile_m, m, k, 2 * block + 1);
        __syncthreads();
        supersonic_mmq_vec_dot_q6_k_q8_1_wmma(tile_x, tile_y, sum, 32);
        __syncthreads();
    }

    const int out_row = tile_m + (threadIdx.x & 15);
    const int lane_half = threadIdx.x >> 4;
    const size_t out_base = static_cast<size_t>(batch) * m * static_cast<size_t>(n);
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int out_col = tile_n + threadIdx.y * 16 + 2 * l + lane_half;
        if constexpr (HOT_EXACT) {
            const size_t out_idx = out_base + static_cast<size_t>(out_row) * n + out_col;
            const hip_bfloat16 projected(sum[l]);
            if constexpr (HAS_RESIDUAL) {
                out[out_idx] = hip_bfloat16(static_cast<float>(residual[out_idx]) +
                                           static_cast<float>(projected));
            } else {
                out[out_idx] = projected;
            }
        } else if (out_row < m && out_col < n) {
            const size_t out_idx = out_base + static_cast<size_t>(out_row) * n + out_col;
            const hip_bfloat16 projected(sum[l]);
            if constexpr (HAS_RESIDUAL) {
                out[out_idx] = hip_bfloat16(static_cast<float>(residual[out_idx]) +
                                           static_cast<float>(projected));
            } else {
                out[out_idx] = projected;
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)q8; (void)rhs_q6; (void)residual; (void)out;
#endif
}

template <bool HOT_EXACT = false, bool HAS_RESIDUAL = false>
__global__ __launch_bounds__(256, 2)
void supersonic_qwen35_matmul_mmq_q8_1_q6_k_gfx12_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const uint8_t* __restrict__ q8,
    const uint8_t* __restrict__ rhs_q6,
    const hip_bfloat16* __restrict__ residual,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_I8) && SUPERSONIC_WMMA_GFX12
    constexpr int MMQ_X = 16;
    constexpr int MMQ_Y = 128;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K = 76;
    constexpr int PADDED_TILE_Y_INTS = 768;

    const int tile_n = blockIdx.x * MMQ_Y;
    const int tile_m = blockIdx.y * MMQ_X;
    const int batch = blockIdx.z;
    if (static_cast<size_t>(batch) >= batch_elems) {
        return;
    }

    extern __shared__ int supersonic_mmq_smem[];
    int* tile_y = supersonic_mmq_smem;
    int* tile_x = supersonic_mmq_smem + PADDED_TILE_Y_INTS;
    float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int q6_blocks = k / 256;
    for (int block = 0; block < q6_blocks; ++block) {
        supersonic_mmq_load_q6_k_tile_128<HOT_EXACT>(
            rhs_q6, tile_x, batch, tile_n, n, k, block);
        supersonic_mmq_load_q8_1_tile_16<HOT_EXACT>(
            q8, tile_y, batch, tile_m, m, k, 2 * block);
        __syncthreads();
        supersonic_mmq_vec_dot_q6_k_q8_1_wmma_gfx12_native(tile_x, tile_y, sum, 0);
        __syncthreads();

        supersonic_mmq_load_q8_1_tile_16<HOT_EXACT>(
            q8, tile_y, batch, tile_m, m, k, 2 * block + 1);
        __syncthreads();
        supersonic_mmq_vec_dot_q6_k_q8_1_wmma_gfx12_native(tile_x, tile_y, sum, 32);
        __syncthreads();
    }

    const int lane = threadIdx.x & 31;
    const int out_row = tile_m + (lane & 15);
    const int lane_group = lane >> 4;
    const size_t out_base = static_cast<size_t>(batch) * m * static_cast<size_t>(n);
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int out_col = tile_n + threadIdx.y * 16 + lane_group * 8 + l;
        if constexpr (HOT_EXACT) {
            const size_t out_idx = out_base + static_cast<size_t>(out_row) * n + out_col;
            const hip_bfloat16 projected(sum[l]);
            if constexpr (HAS_RESIDUAL) {
                out[out_idx] = hip_bfloat16(static_cast<float>(residual[out_idx]) +
                                           static_cast<float>(projected));
            } else {
                out[out_idx] = projected;
            }
        } else if (out_row < m && out_col < n) {
            const size_t out_idx = out_base + static_cast<size_t>(out_row) * n + out_col;
            const hip_bfloat16 projected(sum[l]);
            if constexpr (HAS_RESIDUAL) {
                out[out_idx] = hip_bfloat16(static_cast<float>(residual[out_idx]) +
                                           static_cast<float>(projected));
            } else {
                out[out_idx] = projected;
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)q8; (void)rhs_q6; (void)residual; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    const hip_bfloat16* __restrict__ awq_inv_scale,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;

    const int lhs_row_idx = tile_row + lane_row;
    const int rhs_row_idx = tile_col + lane_row;
    const bool lhs_in_range = lhs_row_idx < m;
    const bool rhs_in_range = rhs_row_idx < n;
    const int row_bytes = ggml_k_row_bytes_fixed<QTYPE>(k);

    const size_t lhs_row_base = static_cast<size_t>(batch) * m * k +
                                static_cast<size_t>(lhs_row_idx) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short16 a;
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = static_cast<short>(lhs_row_u16[kk + i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        supersonic_short16 b;
        if (rhs_in_range) {
            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_row_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_row_u8, kk, k, bits);
            if (awq_inv_scale != nullptr) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    bits[i] = f32_to_bf16_bits_trunc(
                        bf16_bits_to_f32(bits[i]) * static_cast<float>(awq_inv_scale[kk + i]));
                }
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = supersonic_wmma_bf16(a, b, acc);
    }

    if (k_main != k) {
        supersonic_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        supersonic_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int kc = k_main + i;
                if (kc < k) a[i] = static_cast<short>(lhs_row_u16[kc]);
            }
        }
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int kc = k_main + i;
                if (kc < k) {
                    uint16_t bits = ggml_small_m_bf16_bits<TRUNC_DEQUANT>(
                        ggml_k_dequant_scalar_from_row(rhs_row_u8, QTYPE, kc, k));
                    if (awq_inv_scale != nullptr) {
                        bits = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits) * static_cast<float>(awq_inv_scale[kc]));
                    }
                    b[i] = static_cast<short>(bits);
                }
            }
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    const int out_col = tile_col + lane_row;
    if (out_col < n) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = tile_row + 2 * i + lane_half;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)awq_inv_scale; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 8 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;
    const bool lhs_in_range = lane_row < 8;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short16 a;
            if (lhs_in_range) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    a[i] = static_cast<short>(lhs_row_u16[kk + i]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) a[i] = 0;
            }

            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_block, sub * 16, bits);
            supersonic_short16 b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }

            acc = supersonic_wmma_bf16(a, b, acc);
        }
    }

    const int out_col = tile_col + lane_row;
    const size_t out_batch_base = static_cast<size_t>(batch) * 8 * static_cast<size_t>(n);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int out_row = 2 * i + lane_half;
        out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
            hip_bfloat16(acc[i]);
    }
#else
    (void)batch_elems; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 4)
void supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 8 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;
    const bool lhs_in_range = lane_row < 8;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short8 a;
            if (lhs_in_range) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) a[i] = 0;
            }

            supersonic_short8 b;
            if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) {
                b = ggml_q8_0_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) {
                b = ggml_q4_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) {
                b = ggml_q5_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) {
                b = ggml_q6_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else {
                uint16_t bits[16];
                ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                    rhs_block, sub * 16, bits);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    b[i] = static_cast<short>(bits[ab_row]);
                }
            }

            acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
        }
    }

    if (lane_group == 0) {
        const int out_col = tile_col + lane_row;
        const size_t out_batch_base = static_cast<size_t>(batch) * 8 * static_cast<size_t>(n);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[out_batch_base + static_cast<size_t>(i) * n + out_col] =
                hip_bfloat16(acc[i]);
        }
    }
#else
    (void)batch_elems; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false, bool ADD_RESIDUAL = false>
__global__ __launch_bounds__(32, 4)
void supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    const hip_bfloat16* __restrict__ residual,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short16 a;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(lhs_row_u16[kk + i]);
            }

            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_block, sub * 16, bits);
            supersonic_short16 b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }

            acc = supersonic_wmma_bf16(a, b, acc);
        }
    }

    const int out_col = tile_col + lane_row;
    const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = 2 * i + lane_half;
        const size_t out_idx = out_batch_base + static_cast<size_t>(out_row) * n + out_col;
        hip_bfloat16 projected(acc[i]);
        if constexpr (ADD_RESIDUAL) {
            const float summed = static_cast<float>(residual[out_idx]) + static_cast<float>(projected);
            out[out_idx] = hip_bfloat16(summed);
        } else {
            out[out_idx] = projected;
        }
    }
#else
    (void)batch_elems; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)residual; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false, bool ADD_RESIDUAL = false>
__global__ __launch_bounds__(32, 4)
void supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    const hip_bfloat16* __restrict__ residual,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short8 a;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }

            supersonic_short8 b;
            if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) {
                b = ggml_q8_0_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) {
                b = ggml_q4_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) {
                b = ggml_q5_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) {
                b = ggml_q6_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else {
                uint16_t bits[16];
                ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                    rhs_block, sub * 16, bits);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    b[i] = static_cast<short>(bits[ab_row]);
                }
            }

            acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
        }
    }

    const int out_col = tile_col + lane_row;
    const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = lane_group * 8 + i;
        const size_t out_idx = out_batch_base + static_cast<size_t>(out_row) * n + out_col;
        hip_bfloat16 projected(acc[i]);
        if constexpr (ADD_RESIDUAL) {
            const float summed = static_cast<float>(residual[out_idx]) + static_cast<float>(projected);
            out[out_idx] = hip_bfloat16(summed);
        } else {
            out[out_idx] = projected;
        }
    }
#else
    (void)batch_elems; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)residual; (void)out;
#endif
}

template <bool TRUNC_DEQUANT = true>
__global__ __launch_bounds__(32, 4)
void supersonic_qwen35_matmul_q6_k_m16_tile_argmax_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_q6,
    float* __restrict__ block_best_vals,
    uint32_t* __restrict__ block_best_indices
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    __shared__ float tile_vals[16][16];

    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    if (static_cast<size_t>(batch) >= batch_elems) {
        return;
    }

    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;
    const int block_bytes = ggml_k_block_bytes_fixed<QWEN35_LOWBIT_GGML_Q6_K>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_q6 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short16 a;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(lhs_row_u16[kk + i]);
            }

            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QWEN35_LOWBIT_GGML_Q6_K, TRUNC_DEQUANT>(
                rhs_block, sub * 16, bits);
            supersonic_short16 b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }

            acc = supersonic_wmma_bf16(a, b, acc);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = 2 * i + lane_half;
        const hip_bfloat16 rounded(acc[i]);
        tile_vals[out_row][lane_row] = static_cast<float>(rounded);
    }
    __syncthreads();

    if (lane < 16) {
        const int row = lane;
        float best_val = -1.0e30f;
        uint32_t best_idx = 0;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            const float val = tile_vals[row][c];
            const uint32_t idx = static_cast<uint32_t>(tile_col + c);
            if (val > best_val || (val == best_val && idx < best_idx)) {
                best_val = val;
                best_idx = idx;
            }
        }
        const size_t tiles = gridDim.x;
        const size_t out = (static_cast<size_t>(batch) * 16 + row) * tiles + blockIdx.x;
        block_best_vals[out] = best_val;
        block_best_indices[out] = best_idx;
    }
#else
    (void)batch_elems; (void)n; (void)k; (void)lhs; (void)rhs_q6;
    (void)block_best_vals; (void)block_best_indices;
#endif
}

__device__ __forceinline__ float q6_scalar_rn_mul(float a, float b) {
    return a * b;
}

__device__ __forceinline__ float q6_scalar_rn_fma(float a, float b, float c) {
    return __builtin_fmaf(a, b, c);
}

__global__ __launch_bounds__(128)
void supersonic_qwen38_q6_k_scalar_head_f32_kernel(
    const uint16_t* lhs_bf16,
    const uint8_t* rhs_q6,
    float* out_f32,
    int row_start,
    int row_count
) {
    constexpr int Q6_BLOCK_BYTES = 210;
    constexpr int Q6_BLOCK_VALUES = 256;
    constexpr int BLOCKS_PER_ROW = 20;
    constexpr int ROW_BYTES = BLOCKS_PER_ROW * Q6_BLOCK_BYTES;
    constexpr int ROWS_PER_CTA = 4;

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int local_row = static_cast<int>(blockIdx.x) * ROWS_PER_CTA + wave;
    if (local_row >= row_count) {
        return;
    }

    const int global_row = row_start + local_row;
    const uint8_t* row =
        rhs_q6 + static_cast<size_t>(global_row) * static_cast<size_t>(ROW_BYTES);

    {
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        float acc = 0.0f;
        for (int block_index = 0; block_index < BLOCKS_PER_ROW; ++block_index) {
            const uint8_t* block = row + block_index * Q6_BLOCK_BYTES;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
            const float d = ggml_f16_to_f32_unaligned(block + 208);

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int half_idx = t >> 2;
                const int group = t & 3;
                const int scale_idx = half_idx * 8 + (lane >> 4) + group * 2;
                const int ql_offset = half_idx * 64 + lane + ((group & 1) ? 32 : 0);
                const int qh_offset = half_idx * 32 + lane;
                const int qh_shift = group * 2;
                const uint8_t packed = ql[ql_offset];
                const int lo = group >= 2 ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                const int hi = static_cast<int>(((qh[qh_offset] >> qh_shift) & 3) << 4);
                const int q = lo + hi - 32;
                const float weight = q6_scalar_rn_mul(
                    q6_scalar_rn_mul(d, static_cast<float>(scales[scale_idx])),
                    static_cast<float>(q));
                const int coordinate = block_index * Q6_BLOCK_VALUES + lane + 32 * t;
                const float activation = bf16_bits_to_f32(lhs_bf16[coordinate]);
                acc = q6_scalar_rn_fma(weight, activation, acc);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset, 32);
        }
        if (lane == 0) {
            out_f32[global_row] = acc;
        }
    }
}

template <bool TRUNC_DEQUANT = true>
__global__ __launch_bounds__(32, 4)
void supersonic_qwen35_matmul_q6_k_m16_tile_argmax_gfx12_kernel(
    size_t batch_elems,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_q6,
    float* __restrict__ block_best_vals,
    uint32_t* __restrict__ block_best_indices
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    __shared__ float tile_vals[16][16];

    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    if (static_cast<size_t>(batch) >= batch_elems) {
        return;
    }

    const int tile_col = blockIdx.x * 16;
    const int rhs_row_idx = tile_col + lane_row;
    const int block_bytes = ggml_k_block_bytes_fixed<QWEN35_LOWBIT_GGML_Q6_K>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_q6 + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short8 a;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }

            const supersonic_short8 b =
                ggml_q6_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);

            acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = lane_group * 8 + i;
        const hip_bfloat16 rounded(acc[i]);
        tile_vals[out_row][lane_row] = static_cast<float>(rounded);
    }
    __syncthreads();

    if (lane < 16) {
        const int row = lane;
        float best_val = -1.0e30f;
        uint32_t best_idx = 0;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            const float val = tile_vals[row][c];
            const uint32_t idx = static_cast<uint32_t>(tile_col + c);
            if (val > best_val || (val == best_val && idx < best_idx)) {
                best_val = val;
                best_idx = idx;
            }
        }
        const size_t tiles = gridDim.x;
        const size_t out =
            (static_cast<size_t>(batch) * tiles + blockIdx.x) * 16 + row;
        block_best_vals[out] = best_val;
        block_best_indices[out] = best_idx;
    }
#else
    (void)batch_elems; (void)n; (void)k; (void)lhs; (void)rhs_q6;
    (void)block_best_vals; (void)block_best_indices;
#endif
}

__global__ void supersonic_qwen35_reduce_m16_tile_argmax_kernel(
    size_t batch_elems,
    int tiles,
    const float* __restrict__ block_best_vals,
    const uint32_t* __restrict__ block_best_indices,
    uint32_t* __restrict__ out_indices
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= 16 || static_cast<size_t>(batch) >= batch_elems) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const size_t base = (static_cast<size_t>(batch) * 16 + row) * static_cast<size_t>(tiles);

    float best_val = -1.0e30f;
    uint32_t best_idx = 0;
    for (int tile = static_cast<int>(tid); tile < tiles; tile += blockDim.x) {
        const float val = block_best_vals[base + tile];
        const uint32_t idx = block_best_indices[base + tile];
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
        out_indices[static_cast<size_t>(batch) * 16 + row] = shared_idx[0];
    }
}

__global__ void supersonic_qwen35_reduce_m16_tile_argmax_tile_major_kernel(
    size_t batch_elems,
    int tiles,
    const float* __restrict__ block_best_vals,
    const uint32_t* __restrict__ block_best_indices,
    uint32_t* __restrict__ out_indices
) {
    __shared__ float shared_vals[256];
    __shared__ uint32_t shared_idx[256];

    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= 16 || static_cast<size_t>(batch) >= batch_elems) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(tiles) + row;

    float best_val = -1.0e30f;
    uint32_t best_idx = 0;
    for (int tile = static_cast<int>(tid); tile < tiles; tile += blockDim.x) {
        const size_t offset = base + static_cast<size_t>(tile) * 16;
        const float val = block_best_vals[offset];
        const uint32_t idx = block_best_indices[offset];
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
        out_indices[static_cast<size_t>(batch) * 16 + row] = shared_idx[0];
    }
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel(
    size_t batch_elems,
    int n_each,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_first,
    const uint8_t* __restrict__ rhs_second,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int out_col_idx = tile_col + lane_row;
    const int n_out = n_each * 2;
    const bool second = out_col_idx >= n_each;
    const int rhs_row_idx = second ? out_col_idx - n_each : out_col_idx;
    const uint8_t* rhs_base_ptr = second ? rhs_second : rhs_first;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n_each * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_base_ptr + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short16 a;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(lhs_row_u16[kk + i]);
            }

            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_block, sub * 16, bits);
            supersonic_short16 b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }

            acc = supersonic_wmma_bf16(a, b, acc);
        }
    }

    if (out_col_idx < n_out) {
        const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = 2 * i + lane_half;
            out[out_batch_base + static_cast<size_t>(out_row) * n_out + out_col_idx] =
                hip_bfloat16(acc[i]);
        }
    }
#else
    (void)batch_elems; (void)n_each; (void)k;
    (void)lhs; (void)rhs_first; (void)rhs_second; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel(
    size_t batch_elems,
    int n_each,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_first,
    const uint8_t* __restrict__ rhs_second,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int out_col_idx = tile_col + lane_row;
    const int n_out = n_each * 2;
    const bool second = out_col_idx >= n_each;
    const int rhs_row_idx = second ? out_col_idx - n_each : out_col_idx;
    const uint8_t* rhs_base_ptr = second ? rhs_second : rhs_first;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n_each * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_base_ptr + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_block = rhs_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short8 a;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }

            supersonic_short8 b;
            if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) {
                b = ggml_q4_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) {
                b = ggml_q5_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_block, sub * 16, lane_group);
            } else {
                uint16_t bits[16];
                ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                    rhs_block, sub * 16, bits);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    b[i] = static_cast<short>(bits[ab_row]);
                }
            }

            acc = supersonic_wmma_bf16_gfx12_native_acc(a, b, acc);
        }
    }

    if (out_col_idx < n_out) {
        const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = lane_group * 8 + i;
            out[out_batch_base + static_cast<size_t>(out_row) * n_out + out_col_idx] =
                hip_bfloat16(acc[i]);
        }
    }
#else
    (void)batch_elems; (void)n_each; (void)k;
    (void)lhs; (void)rhs_first; (void)rhs_second; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel(
    size_t batch_elems,
    int n_each,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_gate,
    const uint8_t* __restrict__ rhs_up,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int out_col_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n_each * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(out_col_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_gate_row_u8 = rhs_gate + rhs_row_base;
    const uint8_t* rhs_up_row_u8 = rhs_up + rhs_row_base;

    supersonic_float8 gate_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    supersonic_float8 up_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_gate_block = rhs_gate_row_u8 + block * block_bytes;
        const uint8_t* rhs_up_block = rhs_up_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short16 a;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = static_cast<short>(lhs_row_u16[kk + i]);
            }

            uint16_t gate_bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_gate_block, sub * 16, gate_bits);
            supersonic_short16 gate_b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                gate_b[i] = static_cast<short>(gate_bits[i]);
            }
            gate_acc = supersonic_wmma_bf16(a, gate_b, gate_acc);

            uint16_t up_bits[16];
            ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                rhs_up_block, sub * 16, up_bits);
            supersonic_short16 up_b;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                up_b[i] = static_cast<short>(up_bits[i]);
            }
            up_acc = supersonic_wmma_bf16(a, up_b, up_acc);
        }
    }

    if (out_col_idx < n_each) {
        const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n_each);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = 2 * i + lane_half;
            const hip_bfloat16 gate_bf16(gate_acc[i]);
            const hip_bfloat16 up_bf16(up_acc[i]);
            const float gate_x = static_cast<float>(gate_bf16);
            const float up_x = static_cast<float>(up_bf16);
            const float silu = gate_x / (1.0f + expf(-gate_x));
            out[out_batch_base + static_cast<size_t>(out_row) * n_each + out_col_idx] =
                hip_bfloat16(silu * up_x);
        }
    }
#else
    (void)batch_elems; (void)n_each; (void)k;
    (void)lhs; (void)rhs_gate; (void)rhs_up; (void)out;
#endif
}

template <int QTYPE, bool TRUNC_DEQUANT = false>
__global__ __launch_bounds__(32, 8)
void supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel(
    size_t batch_elems,
    int n_each,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_gate,
    const uint8_t* __restrict__ rhs_up,
    hip_bfloat16* __restrict__ out
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_col = blockIdx.x * 16;
    const int out_col_idx = tile_col + lane_row;

    const int block_bytes = ggml_k_block_bytes_fixed<QTYPE>();
    const int blocks = k / 256;
    const int row_bytes = blocks * block_bytes;

    const size_t lhs_row_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(k) +
                                static_cast<size_t>(lane_row) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n_each * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(out_col_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_gate_row_u8 = rhs_gate + rhs_row_base;
    const uint8_t* rhs_up_row_u8 = rhs_up + rhs_row_base;

    supersonic_float8 gate_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    supersonic_float8 up_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int block = 0; block < blocks; ++block) {
        const uint8_t* rhs_gate_block = rhs_gate_row_u8 + block * block_bytes;
        const uint8_t* rhs_up_block = rhs_up_row_u8 + block * block_bytes;
        const int kk_block = block * 256;
        #pragma unroll
        for (int sub = 0; sub < 16; ++sub) {
            const int kk = kk_block + sub * 16;
            supersonic_short8 a;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a[i] = static_cast<short>(lhs_row_u16[kk + ab_row]);
            }

            supersonic_short8 gate_b;
            supersonic_short8 up_b;
            if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q8_0) {
                gate_b = ggml_q8_0_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_gate_block, sub * 16, lane_group);
                up_b = ggml_q8_0_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_up_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q4_K) {
                gate_b = ggml_q4_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_gate_block, sub * 16, lane_group);
                up_b = ggml_q4_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_up_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q5_K) {
                gate_b = ggml_q5_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_gate_block, sub * 16, lane_group);
                up_b = ggml_q5_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_up_block, sub * 16, lane_group);
            } else if constexpr (QTYPE == QWEN35_LOWBIT_GGML_Q6_K) {
                gate_b = ggml_q6_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_gate_block, sub * 16, lane_group);
                up_b = ggml_q6_k_dequant_bf16_gfx12_fragment<TRUNC_DEQUANT>(
                    rhs_up_block, sub * 16, lane_group);
            } else {
                uint16_t gate_bits[16];
                uint16_t up_bits[16];
                ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                    rhs_gate_block, sub * 16, gate_bits);
                ggml_k_dequant_16_bf16_bits_from_block_fixed<QTYPE, TRUNC_DEQUANT>(
                    rhs_up_block, sub * 16, up_bits);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int ab_row = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                    gate_b[i] = static_cast<short>(gate_bits[ab_row]);
                    up_b[i] = static_cast<short>(up_bits[ab_row]);
                }
            }

            gate_acc = supersonic_wmma_bf16_gfx12_native_acc(a, gate_b, gate_acc);
            up_acc = supersonic_wmma_bf16_gfx12_native_acc(a, up_b, up_acc);
        }
    }

    if (out_col_idx < n_each) {
        const size_t out_batch_base = static_cast<size_t>(batch) * 16 * static_cast<size_t>(n_each);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = lane_group * 8 + i;
            const hip_bfloat16 gate_bf16(gate_acc[i]);
            const hip_bfloat16 up_bf16(up_acc[i]);
            const float gate_x = static_cast<float>(gate_bf16);
            const float up_x = static_cast<float>(up_bf16);
            const float silu = gate_x / (1.0f + expf(-gate_x));
            out[out_batch_base + static_cast<size_t>(out_row) * n_each + out_col_idx] =
                hip_bfloat16(silu * up_x);
        }
    }
#else
    (void)batch_elems; (void)n_each; (void)k;
    (void)lhs; (void)rhs_gate; (void)rhs_up; (void)out;
#endif
}

__global__ void supersonic_qwen35_matmul_ggml_pair_wmma_small_m_kernel(
    size_t batch_elems,
    int m,
    int n_each,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_first,
    const uint8_t* __restrict__ rhs_second,
    int quant_type,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int lane = threadIdx.x;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;
    const int n_out = n_each * 2;

    const int lhs_row_idx = tile_row + lane_row;
    const int out_col_idx = tile_col + lane_row;
    const bool lhs_in_range = lhs_row_idx < m;
    const bool rhs_in_range = out_col_idx < n_out;

    const int rhs_row_idx = out_col_idx < n_each ? out_col_idx : out_col_idx - n_each;
    const uint8_t* rhs_base_ptr = out_col_idx < n_each ? rhs_first : rhs_second;
    const int row_bytes = ggml_k_row_bytes(quant_type, k);

    const size_t lhs_row_base = static_cast<size_t>(batch) * m * k +
                                static_cast<size_t>(lhs_row_idx) * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n_each * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_row_u16 = reinterpret_cast<const uint16_t*>(lhs) + lhs_row_base;
    const uint8_t* rhs_row_u8 = rhs_base_ptr + rhs_row_base;

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short16 a;
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = static_cast<short>(lhs_row_u16[kk + i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a[i] = 0;
        }

        supersonic_short16 b;
        if (rhs_in_range) {
            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_row(rhs_row_u8, quant_type, kk, k, bits);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b[i] = static_cast<short>(bits[i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = supersonic_wmma_bf16(a, b, acc);
    }

    if (k_main != k) {
        supersonic_short16 a = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        supersonic_short16 b = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        if (lhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < k) a[i] = static_cast<short>(lhs_row_u16[kc]);
            }
        }
        if (rhs_in_range) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int kc = k_main + i;
                if (kc < k) {
                    b[i] = static_cast<short>(f32_to_bf16_bits_rne_finite(
                        ggml_k_dequant_scalar_from_row(rhs_row_u8, quant_type, kc, k)));
                }
            }
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    if (out_col_idx < n_out) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n_out;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int out_row = tile_row + 2 * i + lane_half;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n_out + out_col_idx] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n_each; (void)k;
    (void)lhs; (void)rhs_first; (void)rhs_second; (void)quant_type; (void)out;
#endif
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_matmul_int4_dequant_wmma_small_m_n64_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,
    const uint8_t* __restrict__ rhs_int4,
    const hip_bfloat16* __restrict__ scale,
    const hip_bfloat16* __restrict__ zero,
    const hip_bfloat16* __restrict__ awq_inv_scale,
    int group_size,
    int quant_type,
    hip_bfloat16* __restrict__ out
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 64;
    const int rhs_row_idx = tile_col + wid * 16 + lane_row;

    const bool native_int4 = qwen35_lowbit_native_int4(quant_type);
    const int row_bytes = native_int4 ? (k / 2) : ggml_k_row_bytes(quant_type, k);
    const int scale_cols = (k + group_size - 1) / group_size;

    const size_t lhs_batch_base = static_cast<size_t>(batch) * m * k;
    const size_t rhs_row_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes) +
                                static_cast<size_t>(rhs_row_idx) * row_bytes;

    const uint16_t* lhs_u16 = reinterpret_cast<const uint16_t*>(lhs);
    const uint8_t* rhs_row_u8 = rhs_int4 + rhs_row_base;

    __shared__ uint16_t s_lhs[16 * 16];
    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        for (int load_idx = tid; load_idx < 16 * 16; load_idx += blockDim.x) {
            const int r = load_idx >> 4;
            const int c = load_idx & 15;
            const int gr = tile_row + r;
            uint16_t v = 0;
            if (gr < m) {
                v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + kk + c];
            }
            s_lhs[load_idx] = v;
        }
        __syncthreads();

        supersonic_short16 a;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<short>(s_lhs[lane_row * 16 + i]);
        }

        supersonic_short16 b;
        if (rhs_row_idx < n) {
            if (native_int4) {
                const int group_id = kk / group_size;
                const int scale_idx = rhs_row_idx / group_size * scale_cols + group_id;
                const float s = static_cast<float>(scale[scale_idx]);
                const float z = static_cast<float>(zero[scale_idx]);
                const float zs = z * s;
                const int byte_base = kk >> 1;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint8_t packed = rhs_row_u8[byte_base + i];
                    int n0 = packed & 0xF;
                    int n1 = (packed >> 4) & 0xF;
                    uint16_t bits0 = f32_to_bf16_bits_rne_finite(static_cast<float>(n0) * s - zs);
                    uint16_t bits1 = f32_to_bf16_bits_rne_finite(static_cast<float>(n1) * s - zs);
                    if (awq_inv_scale != nullptr) {
                        bits0 = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits0) * static_cast<float>(awq_inv_scale[kk + 2 * i]));
                        bits1 = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits1) * static_cast<float>(awq_inv_scale[kk + 2 * i + 1]));
                    }
                    b[2 * i] = static_cast<short>(bits0);
                    b[2 * i + 1] = static_cast<short>(bits1);
                }
            } else {
                uint16_t bits[16];
                ggml_k_dequant_16_bf16_bits_from_row(rhs_row_u8, quant_type, kk, k, bits);
                if (awq_inv_scale != nullptr) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        bits[i] = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(bits[i]) * static_cast<float>(awq_inv_scale[kk + i]));
                    }
                }
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    b[i] = static_cast<short>(bits[i]);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b[i] = 0;
        }

        acc = supersonic_wmma_bf16(a, b, acc);
        __syncthreads();
    }

    if (k_main != k) {
        for (int load_idx = tid; load_idx < 16 * 16; load_idx += blockDim.x) {
            const int r = load_idx >> 4;
            const int c = load_idx & 15;
            const int gr = tile_row + r;
            const int gk = k_main + c;
            uint16_t v = 0;
            if (gr < m && gk < k) {
                v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + gk];
            }
            s_lhs[load_idx] = v;
        }
        __syncthreads();

        supersonic_short16 a;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<short>(s_lhs[lane_row * 16 + i]);
        }

        supersonic_short16 b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (rhs_row_idx < n) {
            if (native_int4) {
                const int group_id0 = k_main / group_size;
                const int scale_idx = rhs_row_idx / group_size * scale_cols + group_id0;
                const float s = static_cast<float>(scale[scale_idx]);
                const float z = static_cast<float>(zero[scale_idx]);
                const float zs = z * s;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kc = k_main + i;
                    if (kc < k) {
                        const uint8_t packed = rhs_row_u8[kc >> 1];
                        const int nib = (kc & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                        uint16_t bits = f32_to_bf16_bits_rne_finite(static_cast<float>(nib) * s - zs);
                        if (awq_inv_scale != nullptr) {
                            bits = f32_to_bf16_bits_trunc(
                                bf16_bits_to_f32(bits) * static_cast<float>(awq_inv_scale[kc]));
                        }
                        b[i] = static_cast<short>(bits);
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kc = k_main + i;
                    if (kc < k) {
                        uint16_t bits = f32_to_bf16_bits_rne_finite(
                            ggml_k_dequant_scalar_from_row(rhs_row_u8, quant_type, kc, k));
                        if (awq_inv_scale != nullptr) {
                            bits = f32_to_bf16_bits_trunc(
                                bf16_bits_to_f32(bits) * static_cast<float>(awq_inv_scale[kc]));
                        }
                        b[i] = static_cast<short>(bits);
                    }
                }
            }
        }
        acc = supersonic_wmma_bf16(a, b, acc);
    }

    const int out_col = rhs_row_idx;
    if (out_col < n) {
        const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = tile_row + 2 * i + lane_half;
            if (out_row < m) {
                out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
                    hip_bfloat16(acc[i]);
            }
        }
    }
#else
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)scale; (void)zero; (void)awq_inv_scale;
    (void)group_size; (void)quant_type; (void)out;
#endif
}

// =============================================================================
// RDNA3 WMMA INT4-dequant prefill matmul (wave32, 16x16x16 f32-accumulate),
// shared-memory tiled.
//
// Input layout matches supersonic_qwen35_matmul_int4_dequant_kernel:
//   lhs  [batch, m, k]          BF16
//   rhs  [batch, n, k/2]        packed INT4 (low nibble = even k index)
//   scl  [n/group, k/group]     BF16 scale
//   zro  [n/group, k/group]     BF16 zero
//   out  [batch, m, n]          BF16
//
// Same 64x64 output tile + 2x2 wave grid + LDS staging as the BF16 tiled
// kernel above. The only differences are the B-tile load (which dequantizes
// packed INT4 into BF16 on the way into LDS) and the per-row scale/zero
// fetch.
//
// Bit-exactness requirement: the dequant must match the Python GPTQ bake
// and the per-lane naive kernel, so the round-trip is:
//   val = bf16_round_rne_f32_finite(nibble * scale - zero * scale)
// stored as BF16 (top 16 bits of the float).
//
// Group constraint: the bridge only dispatches to this kernel when
// group_size % TILED_WMMA_BK == 0 (so every BK-wide K-slab lies inside one
// quantization group and needs exactly one scale/zero fetch). Scalar
// fallback covers other group sizes.
// =============================================================================

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_matmul_int4_dequant_wmma_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,    // [batch, m, k]
    const uint8_t* __restrict__ rhs_int4,    // [batch, n, row_bytes]
    const hip_bfloat16* __restrict__ scale,  // [n/group, k/group]
    const hip_bfloat16* __restrict__ zero,   // [n/group, k/group]
    const hip_bfloat16* __restrict__ awq_inv_scale, // optional [k]
    int group_size,
    int quant_type,
    hip_bfloat16* __restrict__ out           // [batch, m, n]
) {
#ifdef SUPERSONIC_HAS_WMMA_BF16
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;       // 0..3
    const int wave_row = wid >> 1;       // 0 or 1
    const int wave_col = wid & 1;        // 0 or 1
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;

    const int batch   = blockIdx.z;
    const int blk_row = blockIdx.y * TILED_WMMA_BM;
    const int blk_col = blockIdx.x * TILED_WMMA_BN;

    __shared__ uint16_t s_lhs[TILED_WMMA_BM * TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_rhs[TILED_WMMA_BN * TILED_WMMA_LDS_STRIDE];

    const uint16_t* lhs_u16 = reinterpret_cast<const uint16_t*>(lhs);

    const bool native_int4 = qwen35_lowbit_native_int4(quant_type);
    const int row_bytes = native_int4 ? (k / 2) : ggml_k_row_bytes(quant_type, k);
    const int scale_cols = (k + group_size - 1) / group_size;

    const size_t lhs_batch_base = static_cast<size_t>(batch) * m * k;
    const size_t rhs_batch_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes);

    supersonic_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    static_assert(TILED_WMMA_BK % 32 == 0, "BK must be a multiple of 32 (BF16 load lanes)");
    static_assert(TILED_WMMA_BK % 64 == 0, "BK must be a multiple of 64 (INT4 packs 2 K/byte, 32 lanes)");
    constexpr int K_COL_HALVES = TILED_WMMA_BK / 32;     // BF16 A load passes
    constexpr int K_INT4_PASSES = TILED_WMMA_BK / 64;    // INT4 B load passes
    constexpr int K_CHUNKS     = TILED_WMMA_BK / 16;

    const int k_main = k & ~(TILED_WMMA_BK - 1);

    for (int kk0 = 0; kk0 < k_main; kk0 += TILED_WMMA_BK) {
        // --- Load A tile [BM, BK] BF16 into s_lhs ---
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                int gk = kk0 + col;
                uint16_t v = 0;
                if (gr < m) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + gk];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // --- Load + dequantize B tile (INT4 -> BF16) into s_rhs ---
        // Each lane reads one packed byte covering 2 K-values (low + high
        // nibble). 32 lanes cover 64 K-values per pass; K_INT4_PASSES passes
        // cover the full BK. The bridge guarantees group_size is a multiple
        // of BK so all BK K-values in one outer iter share one scale/zero
        // pair per row.
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            float s = 0.f;
            float zs = 0.f;
            if (native_int4 && gc < n) {
                int group_id = kk0 / group_size;
                int scale_idx = gc / group_size * scale_cols + group_id;
                float s_val = static_cast<float>(scale[scale_idx]);
                float z_val = static_cast<float>(zero[scale_idx]);
                s = s_val;
                zs = z_val * s_val;
            }
            #pragma unroll
            for (int pass = 0; pass < K_INT4_PASSES; ++pass) {
                int lane_byte = pass * 32 + lane;
                int col_lo    = lane_byte * 2;      // K column (low nibble)
                int col_hi    = col_lo + 1;         // K column (high nibble)
                uint16_t out_lo = 0;
                uint16_t out_hi = 0;
                if (gc < n) {
                    if (native_int4) {
                        const uint8_t packed = rhs_int4[
                            rhs_batch_base + static_cast<size_t>(gc) * row_bytes +
                            (kk0 >> 1) + lane_byte];
                        const int n0 = packed & 0xF;
                        const int n1 = (packed >> 4) & 0xF;
                        out_lo = f32_to_bf16_bits_rne_finite(static_cast<float>(n0) * s - zs);
                        out_hi = f32_to_bf16_bits_rne_finite(static_cast<float>(n1) * s - zs);
                    } else {
                        const uint8_t* rhs_row = rhs_int4 + rhs_batch_base +
                            static_cast<size_t>(gc) * row_bytes;
                        ggml_k_dequant_pair_bf16_bits_from_row(
                            rhs_row, quant_type, kk0 + col_lo, k, out_lo, out_hi);
                    }
                    if (awq_inv_scale != nullptr) {
                        out_lo = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(out_lo) * static_cast<float>(awq_inv_scale[kk0 + col_lo]));
                        out_hi = f32_to_bf16_bits_trunc(
                            bf16_bits_to_f32(out_hi) * static_cast<float>(awq_inv_scale[kk0 + col_hi]));
                    }
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col_lo] = out_lo;
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col_hi] = out_hi;
            }
        }
        __syncthreads();

        // --- Compute: K_CHUNKS chunks of 16, 4 WMMAs each ---
        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
        __syncthreads();
    }

    // Defensive K-tail path. Qwen K is always a multiple of 128 and BK=64,
    // so k_main == k in practice and this branch is unreachable. Left in
    // case of future bakes with unusual shapes.
    if (k_main != k) {
        const int k_rem = k - k_main;
        // A tail
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < m && col < k_rem) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + k_main + col];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }
        // B tail (dequantized)
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int lds_row = wid * 16 + iter;
            int gc = blk_col + lds_row;
            float s = 0.f;
            float zs = 0.f;
            if (native_int4 && gc < n) {
                int group_id = k_main / group_size;
                int scale_idx = gc / group_size * scale_cols + group_id;
                float s_val = static_cast<float>(scale[scale_idx]);
                float z_val = static_cast<float>(zero[scale_idx]);
                s = s_val;
                zs = z_val * s_val;
            }
            #pragma unroll
            for (int pass = 0; pass < K_INT4_PASSES; ++pass) {
                int lane_byte = pass * 32 + lane;
                int col_lo    = lane_byte * 2;
                int col_hi    = col_lo + 1;
                uint16_t out_lo = 0;
                uint16_t out_hi = 0;
                if (gc < n) {
                    if (native_int4) {
                        uint8_t packed = 0;
                        if (col_lo < k_rem || col_hi < k_rem) {
                            packed = rhs_int4[rhs_batch_base +
                                              static_cast<size_t>(gc) * row_bytes +
                                              (k_main >> 1) + lane_byte];
                        }
                        if (col_lo < k_rem) {
                            int n0 = packed & 0xF;
                            float v0 = bf16_round_rne_f32_finite(static_cast<float>(n0) * s - zs);
                            if (awq_inv_scale != nullptr) {
                                v0 *= static_cast<float>(awq_inv_scale[k_main + col_lo]);
                            }
                            uint32_t b0;
                            __builtin_memcpy(&b0, &v0, 4);
                            out_lo = static_cast<uint16_t>(b0 >> 16);
                        }
                        if (col_hi < k_rem) {
                            int n1 = (packed >> 4) & 0xF;
                            float v1 = bf16_round_rne_f32_finite(static_cast<float>(n1) * s - zs);
                            if (awq_inv_scale != nullptr) {
                                v1 *= static_cast<float>(awq_inv_scale[k_main + col_hi]);
                            }
                            uint32_t b1;
                            __builtin_memcpy(&b1, &v1, 4);
                            out_hi = static_cast<uint16_t>(b1 >> 16);
                        }
                    } else {
                        if (col_lo < k_rem) {
                            const uint8_t* rhs_row = rhs_int4 + rhs_batch_base +
                                static_cast<size_t>(gc) * row_bytes;
                            out_lo = f32_to_bf16_bits_rne_finite(
                                ggml_k_dequant_scalar_from_row(rhs_row, quant_type, k_main + col_lo, k));
                        }
                        if (col_hi < k_rem) {
                            const uint8_t* rhs_row = rhs_int4 + rhs_batch_base +
                                static_cast<size_t>(gc) * row_bytes;
                            out_hi = f32_to_bf16_bits_rne_finite(
                                ggml_k_dequant_scalar_from_row(rhs_row, quant_type, k_main + col_hi, k));
                        }
                        if (awq_inv_scale != nullptr) {
                            if (col_lo < k_rem) {
                                out_lo = f32_to_bf16_bits_trunc(
                                    bf16_bits_to_f32(out_lo) *
                                    static_cast<float>(awq_inv_scale[k_main + col_lo]));
                            }
                            if (col_hi < k_rem) {
                                out_hi = f32_to_bf16_bits_trunc(
                                    bf16_bits_to_f32(out_hi) *
                                    static_cast<float>(awq_inv_scale[k_main + col_hi]));
                            }
                        }
                    }
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col_lo] = out_lo;
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col_hi] = out_hi;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short16 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + 0  + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + 0  + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + i]);
            }

            acc00 = supersonic_wmma_bf16(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16(a1, b1, acc11);
        }
    }

    // Store 4 sub-tiles, same layout as the BF16 tiled kernel.
    const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const supersonic_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int out_row = sub_row_base + sub_row_off + 2 * i + lane_half;
                if (out_row < m) {
                    out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
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
    (void)batch_elems; (void)m; (void)n; (void)k;
    (void)lhs; (void)rhs_int4; (void)scale; (void)zero; (void)awq_inv_scale; (void)group_size; (void)out;
#endif
}

__global__ __launch_bounds__(128, 2)
void supersonic_qwen35_matmul_ggml_q8_0_wmma_gfx12_large_m_kernel(
    size_t batch_elems,
    int m,
    int n,
    int k,
    const hip_bfloat16* __restrict__ lhs,    // [batch, m, k]
    const uint8_t* __restrict__ rhs_q8,      // [batch, n, row_bytes]
    hip_bfloat16* __restrict__ out           // [batch, m, n]
) {
#if defined(SUPERSONIC_HAS_WMMA_BF16) && SUPERSONIC_WMMA_GFX12
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int wid      = tid >> 5;
    const int wave_row = wid >> 1;
    const int wave_col = wid & 1;
    const int lane_row = lane & 15;
    const int lane_group = lane >> 4;

    const int batch   = blockIdx.z;
    const int blk_row = blockIdx.y * TILED_WMMA_BM;
    const int blk_col = blockIdx.x * TILED_WMMA_BN;

    __shared__ uint16_t s_lhs[TILED_WMMA_BM * TILED_WMMA_LDS_STRIDE];
    __shared__ uint16_t s_rhs[TILED_WMMA_BN * TILED_WMMA_LDS_STRIDE];

    const uint16_t* lhs_u16 = reinterpret_cast<const uint16_t*>(lhs);
    const int row_bytes = (k / 32) * 34;
    const size_t lhs_batch_base = static_cast<size_t>(batch) * m * k;
    const size_t rhs_batch_base = static_cast<size_t>(batch) * n * static_cast<size_t>(row_bytes);

    supersonic_float8 acc00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc01 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc10 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    supersonic_float8 acc11 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    constexpr int K_COL_HALVES = TILED_WMMA_BK / 32;
    constexpr int K_Q8_PASSES = TILED_WMMA_BK / 32;
    constexpr int K_CHUNKS = TILED_WMMA_BK / 16;

    for (int kk0 = 0; kk0 < k; kk0 += TILED_WMMA_BK) {
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const int lds_row = wid * 16 + iter;
            const int gr = blk_row + lds_row;
            #pragma unroll
            for (int ch = 0; ch < K_COL_HALVES; ++ch) {
                const int col = ch * 32 + lane;
                uint16_t v = 0;
                if (gr < m) {
                    v = lhs_u16[lhs_batch_base + static_cast<size_t>(gr) * k + kk0 + col];
                }
                s_lhs[lds_row * TILED_WMMA_LDS_STRIDE + col] = v;
            }
        }

        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const int lds_row = wid * 16 + iter;
            const int gc = blk_col + lds_row;
            #pragma unroll
            for (int pass = 0; pass < K_Q8_PASSES; ++pass) {
                const int col0 = pass * 32;
                uint16_t qbits = 0;
                if (gc < n) {
                    const uint8_t* block = rhs_q8 + rhs_batch_base +
                        static_cast<size_t>(gc) * row_bytes + ((kk0 + col0) / 32) * 34;
                    const float d = ggml_f16_to_f32_unaligned(block);
                    const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
                    qbits = f32_to_bf16_bits_rne_finite(d * static_cast<float>(q));
                }
                s_rhs[lds_row * TILED_WMMA_LDS_STRIDE + col0 + lane] = qbits;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kchunk = 0; kchunk < K_CHUNKS; ++kchunk) {
            const int k_off = kchunk * 16;
            supersonic_short8 a0, a1, b0, b1;
            const int a0_row = wave_row * 32 + lane_row;
            const int a1_row = wave_row * 32 + 16 + lane_row;
            const int b0_row = wave_col * 32 + lane_row;
            const int b1_row = wave_col * 32 + 16 + lane_row;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ab_col = lane_group * 4 + (i & 3) + ((i >> 2) * 8);
                a0[i] = static_cast<short>(s_lhs[a0_row * TILED_WMMA_LDS_STRIDE + k_off + ab_col]);
                a1[i] = static_cast<short>(s_lhs[a1_row * TILED_WMMA_LDS_STRIDE + k_off + ab_col]);
                b0[i] = static_cast<short>(s_rhs[b0_row * TILED_WMMA_LDS_STRIDE + k_off + ab_col]);
                b1[i] = static_cast<short>(s_rhs[b1_row * TILED_WMMA_LDS_STRIDE + k_off + ab_col]);
            }

            acc00 = supersonic_wmma_bf16_gfx12_native_acc(a0, b0, acc00);
            acc01 = supersonic_wmma_bf16_gfx12_native_acc(a0, b1, acc01);
            acc10 = supersonic_wmma_bf16_gfx12_native_acc(a1, b0, acc10);
            acc11 = supersonic_wmma_bf16_gfx12_native_acc(a1, b1, acc11);
        }
        __syncthreads();
    }

    const size_t out_batch_base = static_cast<size_t>(batch) * m * n;
    const int sub_row_base = blk_row + wave_row * 32;
    const int sub_col_base = blk_col + wave_col * 32;

    auto store_sub = [&](const supersonic_float8& acc, int sub_row_off, int sub_col_off) {
        const int out_col = sub_col_base + sub_col_off + lane_row;
        if (out_col < n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_row = sub_row_base + sub_row_off + lane_group * 8 + i;
                if (out_row < m) {
                    out[out_batch_base + static_cast<size_t>(out_row) * n + out_col] =
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
    (void)batch_elems; (void)m; (void)n; (void)k; (void)lhs; (void)rhs_q8; (void)out;
#endif
}

template <typename T>
__global__ void supersonic_qwen35_int4_sparse_outlier_add_kernel(
    int rows,
    int n,
    int k,
    int sub_cols,
    const T* __restrict__ lhs,
    const uint32_t* __restrict__ outlier_cols,
    const T* __restrict__ outlier_delta,
    T* __restrict__ out
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(n);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx / static_cast<size_t>(n));
    const int out_col = static_cast<int>(idx % static_cast<size_t>(n));
    float acc = 0.0f;
    for (int j = 0; j < sub_cols; ++j) {
        const int k_col = static_cast<int>(outlier_cols[j]);
        const float x = supersonic_qwen35_to_float(lhs[row * k + k_col]);
        const float delta = supersonic_qwen35_to_float(outlier_delta[out_col * sub_cols + j]);
        acc += x * delta;
    }

    const float base = supersonic_qwen35_to_float(out[idx]);
    out[idx] = supersonic_qwen35_from_float<T>(base + acc);
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

    __shared__ float shared_inv_rms;
    __shared__ float shared_warp_sum[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float warp_sum = supersonic_qwen35_wave_sum(partial);
    if (lane == 0) {
        shared_warp_sum[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < (blockDim.x + 31) / 32 ? shared_warp_sum[lane] : 0.0f;
        block_sum = supersonic_qwen35_wave_sum(block_sum);
        if (lane == 0) {
            const float mean = block_sum / static_cast<float>(n_cols);
            shared_inv_rms = rsqrtf(mean + eps);
        }
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

    __shared__ float shared_inv_rms;
    __shared__ float shared_warp_sum[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float warp_sum = supersonic_qwen35_wave_sum(partial);
    if (lane == 0) {
        shared_warp_sum[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < (blockDim.x + 31) / 32 ? shared_warp_sum[lane] : 0.0f;
        block_sum = supersonic_qwen35_wave_sum(block_sum);
        if (lane == 0) {
            const float mean = block_sum / static_cast<float>(n_cols);
            shared_inv_rms = rsqrtf(mean + eps);
        }
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

__global__ void supersonic_qwen35_swiglu_f32_kernel(
    float* __restrict__ gate, const float* __restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = bf16_round_rne_f32_finite(gate[i]);
    const float u = bf16_round_rne_f32_finite(up[i]);
    const float silu = g / (1.0f + expf(-g));
    gate[i] = bf16_round_rne_f32_finite(silu * u);
}

// Dedicated decode RMS. Matches split-1 / split-4 numerics in
// supersonic_qwen35_persistent_decode_kernel (256-thread tree reduction,
// BF16 round-trip store into dest). flags: bit0 unpack hidden_io→hidden_f32,
// bit1 BF16-round hidden_f32 before RMS (split-1 always, split-4 never).
// 20×256 covers hidden=5120 with one element per thread. Two kernels on
// the same stream keep all 20 CUs busy (the 1-block RMS left 95 CUs idle).
__global__ __launch_bounds__(256, 4) void supersonic_qwen35_decode_rms_partial_kernel(
    float* __restrict__ hidden_f32,
    const hip_bfloat16* __restrict__ hidden_io,
    float* __restrict__ partials,
    int hidden_dim,
    float /*eps*/,
    int flags
) {
    extern __shared__ float lds[];
    const int tid = threadIdx.x;
    const int row = static_cast<int>(blockIdx.y);
    hidden_f32 += static_cast<int64_t>(row) * hidden_dim;
    if (hidden_io != nullptr) {
        hidden_io += static_cast<int64_t>(row) * hidden_dim;
    }
    const int c = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    float x = 0.0f;
    if (c < hidden_dim) {
        if ((flags & 1) != 0 && hidden_io != nullptr) {
            x = static_cast<float>(hidden_io[c]);
        } else {
            x = hidden_f32[c];
        }
        if ((flags & 2) != 0) {
            x = bf16_round_rne_f32_finite(x);
        }
        hidden_f32[c] = x;
    }
    float partial_sq = x * x;
    lds[tid] = partial_sq;
    __syncthreads();
    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            lds[tid] += lds[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[row * static_cast<int>(gridDim.x) + blockIdx.x] = lds[0];
    }
}

__global__ __launch_bounds__(256, 4) void supersonic_qwen35_decode_rms_scale_kernel(
    const float* __restrict__ hidden_f32,
    float* __restrict__ normed,
    const hip_bfloat16* __restrict__ norm_w,
    const float* __restrict__ partials,
    int hidden_dim,
    int npartials,
    float eps,
    float unit_offset
) {
    const int row = static_cast<int>(blockIdx.y);
    hidden_f32 += static_cast<int64_t>(row) * hidden_dim;
    normed += static_cast<int64_t>(row) * hidden_dim;
    const float* row_partials = partials + static_cast<int64_t>(row) * npartials;
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < npartials; ++i) {
            tot += row_partials[i];
        }
        s_inv = rsqrtf(tot / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();
    const int c = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (c < hidden_dim) {
        const float y =
            hidden_f32[c] * s_inv * (static_cast<float>(norm_w[c]) + unit_offset);
        normed[c] = static_cast<float>(hip_bfloat16(y));
    }
}

// Same two-phase math as partial+scale, one cooperative launch (grid.sync).
// Cuts 128 scale launches/token on decode.
#if defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#endif
__global__ __launch_bounds__(256, 4) void supersonic_qwen35_decode_rms_coop_kernel(
    float* __restrict__ hidden_f32,
    float* __restrict__ normed,
    const hip_bfloat16* __restrict__ hidden_io,
    const hip_bfloat16* __restrict__ norm_w,
    float* __restrict__ partials,
    int hidden_dim,
    int npartials,
    float eps,
    float unit_offset,
    int flags
) {
    extern __shared__ float lds[];
    const int tid = threadIdx.x;
    const int c = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    float x = 0.0f;
    if (c < hidden_dim) {
        if ((flags & 1) != 0 && hidden_io != nullptr) {
            x = static_cast<float>(hidden_io[c]);
        } else {
            x = hidden_f32[c];
        }
        if ((flags & 2) != 0) {
            x = bf16_round_rne_f32_finite(x);
        }
        hidden_f32[c] = x;
    }
    float partial_sq = x * x;
    lds[tid] = partial_sq;
    __syncthreads();
    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            lds[tid] += lds[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = lds[0];
    }
#if defined(__HIPCC__)
    cooperative_groups::this_grid().sync();
#else
    __syncthreads();
#endif
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < npartials; ++i) {
            tot += partials[i];
        }
        s_inv = rsqrtf(tot / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();
    if (c < hidden_dim) {
        const float y =
            hidden_f32[c] * s_inv * (static_cast<float>(norm_w[c]) + unit_offset);
        normed[c] = static_cast<float>(hip_bfloat16(y));
    }
}

__global__ __launch_bounds__(256, 8) void supersonic_qwen35_decode_rms_kernel(
    float* __restrict__ hidden_f32,
    float* __restrict__ normed,
    const hip_bfloat16* __restrict__ hidden_io,
    const hip_bfloat16* __restrict__ norm_w,
    int hidden_dim,
    int batch_size,
    float eps,
    float unit_offset,
    int flags
) {
    extern __shared__ float lds[];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int B = batch_size > 0 ? batch_size : 1;
    const int b0 = (gridDim.x > 1) ? static_cast<int>(blockIdx.x) : 0;
    const int b1 = (gridDim.x > 1) ? (b0 + 1) : B;
    for (int b = b0; b < b1; ++b) {
        float* src = hidden_f32 + b * hidden_dim;
        float* dst = normed + b * hidden_dim;
        if ((flags & 1) != 0 && hidden_io != nullptr) {
            const hip_bfloat16* io = hidden_io + b * hidden_dim;
            for (int c = tid; c < hidden_dim; c += bs) {
                src[c] = static_cast<float>(io[c]);
            }
            __syncthreads();
        }
        if ((flags & 2) != 0) {
            for (int c = tid; c < hidden_dim; c += bs) {
                src[c] = bf16_round_rne_f32_finite(src[c]);
            }
            __syncthreads();
        }
        float partial_sq = 0.0f;
        for (int c = tid; c < hidden_dim; c += bs) {
            partial_sq += src[c] * src[c];
        }
        lds[tid] = partial_sq;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) {
                lds[tid] += lds[tid + s];
            }
            __syncthreads();
        }
        const float inv_rms =
            rsqrtf(lds[0] / static_cast<float>(hidden_dim) + eps);
        for (int c = tid; c < hidden_dim; c += bs) {
            const float y =
                src[c] * inv_rms * (static_cast<float>(norm_w[c]) + unit_offset);
            dst[c] = static_cast<float>(hip_bfloat16(y));
        }
        __syncthreads();
    }
}

__global__ void supersonic_qwen35_bf16_round_f32_kernel(float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = bf16_round_rne_f32_finite(y[i]);
    }
}

__global__ void supersonic_qwen35_add_round_f32_kernel(
    float* __restrict__ dst, const float* __restrict__ add, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = bf16_round_rne_f32_finite(
            dst[i] + bf16_round_rne_f32_finite(add[i]));
    }
}

__global__ void supersonic_qwen35_pack_f32_to_bf16_kernel(
    const float* __restrict__ src, hip_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = hip_bfloat16(src[i]);
    }
}

__global__ void supersonic_qwen35_unpack_bf16_to_f32_kernel(
    const hip_bfloat16* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Interleaved decode Q [h][q|gate] → packed Q [h][q] BF16.
__global__ void supersonic_qwen35_pack_full_q_interleaved_kernel(
    const float* __restrict__ q_int,
    hip_bfloat16* __restrict__ q_out,
    int nh,
    int hd) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = nh * hd;
    if (i >= n) {
        return;
    }
    const int h = i / hd;
    const int d = i - h * hd;
    q_out[i] = hip_bfloat16(q_int[h * hd * 2 + d]);
}

// Decode attention over [kv_heads, kv_stride, head_dim].
// Pass 1: grid (q_heads, n_splits), block 32. Each warp owns every
// n_splits-th key and writes (m, l, acc[8]) into partials.
// Pass 2: grid q_heads, block 32. Merges the split stats.
template <typename T>
__global__ void supersonic_qwen35_full_attention_decode_split_kernel(
    int kv_len,
    int kv_stride,
    int head_dim,
    int num_kv_groups,
    int n_splits,
    float scale,
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_acc
) {
    if (kv_len < 0) {
        kv_len = qwen35_split_seqlen + (-kv_len);
    }
    const int lane = threadIdx.x;
    const int qh = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int kvh = qh / num_kv_groups;
    const T* q_row = query + static_cast<size_t>(qh) * head_dim;
    const T* k_head = key +
        static_cast<size_t>(kvh) * static_cast<size_t>(kv_stride) * head_dim;
    const T* v_head = value +
        static_cast<size_t>(kvh) * static_cast<size_t>(kv_stride) * head_dim;

    float q_reg[8];
    float acc[8];
    int acc_dim[8];
    int acc_count = 0;
    const bool pack8 = (head_dim == 256);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        q_reg[i] = 0.0f;
        acc[i] = 0.0f;
        acc_dim[i] = -1;
    }
    if (pack8) {
        const int d0 = lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc_dim[i] = d0 + i;
            q_reg[i] = supersonic_qwen35_to_float(q_row[d0 + i]);
        }
        acc_count = 8;
    } else {
        for (int d = lane; d < head_dim && acc_count < 8; d += 32) {
            acc_dim[acc_count] = d;
            q_reg[acc_count] = supersonic_qwen35_to_float(q_row[d]);
            ++acc_count;
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    for (int t = split; t < kv_len; t += n_splits) {
        const T* k_row = k_head + static_cast<size_t>(t) * head_dim;
        const T* v_row = v_head + static_cast<size_t>(t) * head_dim;
        float partial = 0.0f;
        if (pack8) {
            const int d0 = lane * 8;
            const uint4 ku = *reinterpret_cast<const uint4*>(k_row + d0);
            const uint32_t kbits[4] = {ku.x, ku.y, ku.z, ku.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w = kbits[i];
                hip_bfloat16 lo, hi;
                memcpy(&lo, &w, 2);
                const uint16_t hi16 = static_cast<uint16_t>(w >> 16);
                memcpy(&hi, &hi16, 2);
                partial += q_reg[2 * i] * supersonic_qwen35_to_float(lo);
                partial += q_reg[2 * i + 1] * supersonic_qwen35_to_float(hi);
            }
        } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < acc_count) {
                partial += q_reg[i] * supersonic_qwen35_to_float(k_row[acc_dim[i]]);
            }
        }
        }
        const float score =
            __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;
        const float m_old = m_i;
        const float m_new = score > m_old ? score : m_old;
        const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
        const float weight = expf(score - m_new);
        l_i = l_i * alpha + weight;
        m_i = m_new;
        if (pack8) {
            const int d0 = lane * 8;
            const uint4 vu = *reinterpret_cast<const uint4*>(v_row + d0);
            const uint32_t vbits[4] = {vu.x, vu.y, vu.z, vu.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w = vbits[i];
                hip_bfloat16 lo, hi;
                memcpy(&lo, &w, 2);
                const uint16_t hi16 = static_cast<uint16_t>(w >> 16);
                memcpy(&hi, &hi16, 2);
                acc[2 * i] = acc[2 * i] * alpha +
                    weight * supersonic_qwen35_to_float(lo);
                acc[2 * i + 1] = acc[2 * i + 1] * alpha +
                    weight * supersonic_qwen35_to_float(hi);
            }
        } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < acc_count) {
                acc[i] = acc[i] * alpha +
                    weight * supersonic_qwen35_to_float(v_row[acc_dim[i]]);
            }
        }
        }
    }

    const int slot = qh * n_splits + split;
    if (lane == 0) {
        part_m[slot] = m_i;
        part_l[slot] = l_i;
    }
    float* acc_out = part_acc + (static_cast<size_t>(slot) * 32 + lane) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc_out[i] = acc[i];
    }
}

// GQA decode: one warp owns kQ consecutive Q heads that share a KV head,
// so K/V load once. Same per-head online softmax as the 1-head kernel.
template <typename T, int kQ>
__global__ void supersonic_qwen35_full_attention_decode_split_gqa_kernel(
    int kv_len,
    int kv_stride,
    int head_dim,
    int num_kv_groups,
    int n_splits,
    float scale,
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_acc
) {
    if (kv_len < 0) {
        kv_len = qwen35_split_seqlen + (-kv_len);
    }
    const int lane = threadIdx.x;
    const int split = static_cast<int>(blockIdx.y);
    const int qh0 = static_cast<int>(blockIdx.x) * kQ;
    const int kvh = qh0 / num_kv_groups;
    const T* k_head = key +
        static_cast<size_t>(kvh) * static_cast<size_t>(kv_stride) * head_dim;
    const T* v_head = value +
        static_cast<size_t>(kvh) * static_cast<size_t>(kv_stride) * head_dim;

    const int d0 = lane * 8;
    float q_reg[kQ][8];
    float acc[kQ][8];
#pragma unroll
    for (int h = 0; h < kQ; ++h) {
        const T* q_row = query + static_cast<size_t>(qh0 + h) * head_dim;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q_reg[h][i] = supersonic_qwen35_to_float(q_row[d0 + i]);
            acc[h][i] = 0.0f;
        }
    }

    float m_i[kQ];
    float l_i[kQ];
#pragma unroll
    for (int h = 0; h < kQ; ++h) {
        m_i[h] = -INFINITY;
        l_i[h] = 0.0f;
    }

    for (int t = split; t < kv_len; t += n_splits) {
        const T* k_row = k_head + static_cast<size_t>(t) * head_dim;
        const T* v_row = v_head + static_cast<size_t>(t) * head_dim;
        const uint4 ku = *reinterpret_cast<const uint4*>(k_row + d0);
        const uint32_t kbits[4] = {ku.x, ku.y, ku.z, ku.w};
        float k_f[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = kbits[i];
            hip_bfloat16 lo, hi;
            memcpy(&lo, &w, 2);
            const uint16_t hi16 = static_cast<uint16_t>(w >> 16);
            memcpy(&hi, &hi16, 2);
            k_f[2 * i] = supersonic_qwen35_to_float(lo);
            k_f[2 * i + 1] = supersonic_qwen35_to_float(hi);
        }
        const uint4 vu = *reinterpret_cast<const uint4*>(v_row + d0);
        const uint32_t vbits[4] = {vu.x, vu.y, vu.z, vu.w};
        float v_f[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = vbits[i];
            hip_bfloat16 lo, hi;
            memcpy(&lo, &w, 2);
            const uint16_t hi16 = static_cast<uint16_t>(w >> 16);
            memcpy(&hi, &hi16, 2);
            v_f[2 * i] = supersonic_qwen35_to_float(lo);
            v_f[2 * i + 1] = supersonic_qwen35_to_float(hi);
        }
#pragma unroll
        for (int h = 0; h < kQ; ++h) {
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                partial += q_reg[h][i] * k_f[i];
            }
            const float score =
                __shfl(supersonic_qwen35_wave_sum(partial), 0) * scale;
            const float m_old = m_i[h];
            const float m_new = score > m_old ? score : m_old;
            const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            const float weight = expf(score - m_new);
            l_i[h] = l_i[h] * alpha + weight;
            m_i[h] = m_new;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[h][i] = acc[h][i] * alpha + weight * v_f[i];
            }
        }
    }

#pragma unroll
    for (int h = 0; h < kQ; ++h) {
        const int slot = (qh0 + h) * n_splits + split;
        if (lane == 0) {
            part_m[slot] = m_i[h];
            part_l[slot] = l_i[h];
        }
        float* acc_out = part_acc + (static_cast<size_t>(slot) * 32 + lane) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc_out[i] = acc[h][i];
        }
    }
}

__global__ void supersonic_qwen35_full_attention_decode_merge_kernel(
    int n_splits,
    int head_dim,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_acc,
    float* __restrict__ out
) {
    const int lane = threadIdx.x;
    const int qh = static_cast<int>(blockIdx.x);
    int acc_count = 0;
    int acc_dim[8];
    if (head_dim == 256) {
        const int d0 = lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc_dim[i] = d0 + i;
        }
        acc_count = 8;
    } else {
        for (int d = lane; d < head_dim && acc_count < 8; d += 32) {
            acc_dim[acc_count++] = d;
        }
    }
    float m_all = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        m_all = fmaxf(m_all, part_m[qh * n_splits + s]);
    }
    float l_all = 0.0f;
    float acc_all[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < n_splits; ++s) {
        const int slot = qh * n_splits + s;
        const float m_s = part_m[slot];
        const float alpha = (m_s == -INFINITY) ? 0.0f : expf(m_s - m_all);
        l_all += part_l[slot] * alpha;
        const float* acc_s = part_acc + (static_cast<size_t>(slot) * 32 + lane) * 8;
        for (int i = 0; i < acc_count; ++i) {
            acc_all[i] += acc_s[i] * alpha;
        }
    }
    const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
    float* out_row = out + static_cast<size_t>(qh) * head_dim;
    for (int i = 0; i < acc_count; ++i) {
        out_row[acc_dim[i]] = acc_all[i] * inv;
    }
}

__global__ void supersonic_qwen35_full_attn_gate_f32_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float x = bf16_round_rne_f32_finite(attn[i]);
    const float g = gate[i];
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    out[i] = bf16_round_rne_f32_finite(x * sigmoid_g);
}

// Pack unique Q/K (nk heads) to nv with GQA repeat, plus beta/g.
// Q/K unique are already L2-normalized (Q also * 1/sqrt(hkd)).
__global__ void supersonic_qwen35_decode_pack_rec_inputs_kernel(
    int nv,
    int nk,
    int khd,
    const float* __restrict__ q_unique,
    const float* __restrict__ k_unique,
    const float* __restrict__ b,
    const float* __restrict__ a,
    const hip_bfloat16* __restrict__ dt_bias,
    const hip_bfloat16* __restrict__ a_log_exp,
    float* __restrict__ q_rep,
    float* __restrict__ k_rep,
    float* __restrict__ beta,
    float* __restrict__ g) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int qk_n = nv * khd;
    if (idx < qk_n) {
        const int h = idx / khd;
        const int d = idx - h * khd;
        const int src = (h % nk) * khd + d;
        q_rep[idx] = q_unique[src];
        k_rep[idx] = k_unique[src];
    }
    if (idx < nv) {
        beta[idx] = 1.0f / (1.0f + expf(-b[idx]));
        if (dt_bias != nullptr && a_log_exp != nullptr) {
            const float dt = supersonic_qwen35_to_float(dt_bias[idx]);
            const float ale = supersonic_qwen35_to_float(a_log_exp[idx]);
            const float sp = logf(1.0f + expf(a[idx] + dt));
            g[idx] = -sp * ale;
        } else {
            g[idx] = 0.0f;
        }
    }
}

// One Q8_0 row × x, same product order as ggml_k_matvec_kernel<8>.
// Caller must be a full wave; only lane 0's returned value is the reduction.
__device__ inline float ggml_q8_0_warp_dot(
    const uint8_t* __restrict__ w,
    int row,
    const float* __restrict__ x,
    int in_dim) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const uint8_t* row_data =
        w + static_cast<size_t>(row) * static_cast<size_t>((in_dim / 32) * 34);
    float p = 0.0f;
    const int nblk = in_dim / 32;
    for (int blk = 0; blk < nblk; ++blk) {
        const uint8_t* b = row_data + static_cast<size_t>(blk) * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        const int8_t q = reinterpret_cast<const int8_t*>(b + 2)[lane];
        const float term = (d * static_cast<float>(q)) * x[blk * 32 + lane];
        // Sequential 0..31 sum matches int4-dequant TILE_K inner loop
        // (the earlier xor reduction was 1 ULP off for Q8_0).
        float tile = 0.0f;
        for (int i = 0; i < 32; ++i) {
            tile += __shfl(term, i);
        }
        if (lane == 0) {
            p += tile;
        }
    }
    return p;
}

#if defined(SUPERSONIC_HAS_WMMA_BF16)
// One 16-column tile of the ggml-K small-m WMMA GEMV (m=1).
// x is BF16-aligned F32. 32-thread wave. Writes y[col0 .. col0+15].
template <int kQtype>
__device__ inline void ggml_k_wmma_m1_tile16(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    int k,
    int col0,
    int n,
    float* __restrict__ y) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_row = lane & 15;
    const int lane_half = lane >> 4;
    const bool lhs_in_range = lane_row < 1;
    const int rhs_row_idx = col0 + lane_row;
    const bool rhs_in_range = rhs_row_idx < n;
    const int row_bytes = ggml_k_row_bytes_fixed<kQtype>(k);
    const uint8_t* rhs_row =
        w + static_cast<size_t>(rhs_row_idx) * static_cast<size_t>(row_bytes);

    supersonic_float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int k_main = k & ~15;
    for (int kk = 0; kk < k_main; kk += 16) {
        supersonic_short16 a;
        if (lhs_in_range) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint32_t bits;
                __builtin_memcpy(&bits, &x[kk + i], 4);
                a[i] = static_cast<short>(bits >> 16);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                a[i] = 0;
            }
        }
        supersonic_short16 bvec;
        if (rhs_in_range) {
            uint16_t bits[16];
            ggml_k_dequant_16_bf16_bits_from_row_fixed<kQtype, false>(
                rhs_row, kk, k, bits);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                bvec[i] = static_cast<short>(bits[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                bvec[i] = 0;
            }
        }
        acc = supersonic_wmma_bf16(a, bvec, acc);
    }
    const int out_col = col0 + lane_row;
    if (out_col < n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = 2 * i + lane_half;
            if (out_row < 1) {
                y[out_col] = static_cast<float>(hip_bfloat16(acc[i]));
            }
        }
    }
}
__device__ inline void ggml_q8_0_wmma_m1_tile16(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    int k,
    int col0,
    int n,
    float* __restrict__ y) {
    ggml_k_wmma_m1_tile16<QWEN35_LOWBIT_GGML_Q8_0>(w, x, k, col0, n, y);
}
#endif

// Host-visible even when SUPERSONIC_HAS_WMMA_BF16 is device-only.
template <int kQtype>
__global__ void supersonic_qwen35_ggml_k_wmma_gemv_f32_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int k,
    int n) {
#if defined(SUPERSONIC_HAS_WMMA_BF16)
    if (threadIdx.x >= 32) {
        return;
    }
    const int col0 = static_cast<int>(blockIdx.x) * 16;
    ggml_k_wmma_m1_tile16<kQtype>(w, x, k, col0, n, y);
#else
    (void)w;
    (void)x;
    (void)y;
    (void)k;
    (void)n;
#endif
}

__global__ void supersonic_qwen35_q8_0_wmma_gemv_f32_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int k,
    int n,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride) {
    if (threadIdx.x >= 32) {
        return;
    }
#if defined(SUPERSONIC_HAS_WMMA_BF16)
    const int tok = static_cast<int>(blockIdx.y);
    if (ncols > 0 && tok >= ncols) {
        return;
    }
    const int64_t xs = x_col_stride > 0 ? x_col_stride : k;
    const int64_t ys = y_col_stride > 0 ? y_col_stride : n;
    const float* xc = x + static_cast<int64_t>(tok) * xs;
    float* yc = y + static_cast<int64_t>(tok) * ys;
    const int col0 = static_cast<int>(blockIdx.x) * 16;
    ggml_k_wmma_m1_tile16<QWEN35_LOWBIT_GGML_Q8_0>(w, xc, k, col0, n, yc);
#else
    (void)w;
    (void)x;
    (void)y;
    (void)k;
    (void)n;
    (void)ncols;
    (void)x_col_stride;
    (void)y_col_stride;
#endif
}

// Same wave-per-row walk as steal o/out: lane owns columns lane, lane+32, ...
// then xor-reduce. Per-block scales are hoisted; the product order matches
// ggml_k_dequant_scalar_from_row so GQH-mixed ggml-K rows stay bit-exact.
template <int kQtype>
__device__ inline float ggml_k_matvec_row_dot(
    const uint8_t* __restrict__ row_data,
    const float* __restrict__ x,
    int in_dim,
    int lane) {
    float p = 0.0f;
    if constexpr (kQtype == QWEN35_LOWBIT_GGML_Q8_0) {
        const int nblk = in_dim / 32;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t* b = row_data + static_cast<size_t>(blk) * 34;
            const float d = ggml_f16_to_f32_unaligned(b);
            const int8_t q = reinterpret_cast<const int8_t*>(b + 2)[lane];
            p += (d * static_cast<float>(q)) * x[blk * 32 + lane];
        }
    } else if constexpr (kQtype == QWEN35_LOWBIT_GGML_Q4_K) {
        const int nblk = in_dim / 256;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t* b = row_data + static_cast<size_t>(blk) * 144;
            const float d = ggml_f16_to_f32_unaligned(b);
            const float dmin = ggml_f16_to_f32_unaligned(b + 2);
            const uint8_t* sc = b + 4;
            const uint8_t* qs = b + 16;
            const float* xb = x + blk * 256;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int inb = lane + t * 32;
                const int g = inb / 64;
                const int sub = (inb % 64) / 32;
                const int j = 2 * g + sub;
                int scale = 0;
                int minv = 0;
                ggml_q4_k_scale_min(sc, j, scale, minv);
                const uint8_t qbyte = qs[g * 32 + (inb % 32)];
                const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
                p += (d * static_cast<float>(scale) * static_cast<float>(q) -
                      dmin * static_cast<float>(minv)) *
                    xb[inb];
            }
        }
    } else if constexpr (kQtype == QWEN35_LOWBIT_GGML_Q5_K) {
        const int nblk = in_dim / 256;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t* b = row_data + static_cast<size_t>(blk) * 176;
            const float d = ggml_f16_to_f32_unaligned(b);
            const float dmin = ggml_f16_to_f32_unaligned(b + 2);
            const uint8_t* sc = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = b + 48;
            const float* xb = x + blk * 256;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int inb = lane + t * 32;
                const int g = inb / 64;
                const int sub = (inb % 64) / 32;
                const int j = 2 * g + sub;
                int scale = 0;
                int minv = 0;
                ggml_q4_k_scale_min(sc, j, scale, minv);
                const int idx = inb % 32;
                const uint8_t qbyte = ql[g * 32 + idx];
                const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
                const int hi =
                    (qh[idx] & (sub != 0 ? (2 << (2 * g)) : (1 << (2 * g)))) ? 16
                                                                            : 0;
                p += (d * static_cast<float>(scale) * static_cast<float>(lo + hi) -
                      dmin * static_cast<float>(minv)) *
                    xb[inb];
            }
        }
    } else if constexpr (kQtype == QWEN35_LOWBIT_GGML_Q6_K) {
        const int nblk = in_dim / 256;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t* b = row_data + static_cast<size_t>(blk) * 210;
            const uint8_t* ql = b;
            const uint8_t* qh = b + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
            const float d = ggml_f16_to_f32_unaligned(b + 208);
            const float* xb = x + blk * 256;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int inb = lane + t * 32;
                const int half_idx = inb / 128;
                const int pos = inb - half_idx * 128;
                const int idx = pos % 32;
                int q = 0;
                int scale_idx = half_idx * 8 + idx / 16;
                const uint8_t qh_byte = qh[half_idx * 32 + idx];
                if (pos < 32) {
                    q = static_cast<int>(ql[half_idx * 64 + idx] & 0x0F) |
                        static_cast<int>(((qh_byte >> 0) & 3) << 4);
                } else if (pos < 64) {
                    q = static_cast<int>(ql[half_idx * 64 + 32 + idx] & 0x0F) |
                        static_cast<int>(((qh_byte >> 2) & 3) << 4);
                    scale_idx += 2;
                } else if (pos < 96) {
                    q = static_cast<int>((ql[half_idx * 64 + idx] >> 4) & 0x0F) |
                        static_cast<int>(((qh_byte >> 4) & 3) << 4);
                    scale_idx += 4;
                } else {
                    q = static_cast<int>((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0F) |
                        static_cast<int>(((qh_byte >> 6) & 3) << 4);
                    scale_idx += 6;
                }
                p += (d * static_cast<float>(sc[scale_idx]) *
                      static_cast<float>(q - 32)) *
                    xb[inb];
            }
        }
    } else {
        for (int c = lane; c < in_dim; c += 32) {
            p += ggml_k_dequant_scalar_from_row(row_data, kQtype, c, in_dim) * x[c];
        }
    }
    return p;
}

// kRows=2: one warp does two consecutive rows and shares x. Product order
// per row is unchanged. Launch only when out % (warps*2) == 0.
// kCols=1 is decode GEMV. kCols>1 is skinny GEMM: one weight load, ncol
// activation columns (B=1 launch path is unchanged).
template <int kQtype, bool kAcc = false, int kRows = 1, int kCols = 1>
__global__ void supersonic_qwen35_ggml_k_matvec_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in_dim,
    int out_dim,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride) {
    constexpr int kWarp = 32;
    constexpr int kWarps = 4;
    const int row0 = (static_cast<int>(blockIdx.x) * kWarps +
        (static_cast<int>(threadIdx.x) / kWarp)) * kRows;
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    if (row0 >= out_dim) {
        return;
    }
    const int ncol = (kCols == 1) ? 1 : (ncols < kCols ? ncols : kCols);
    const int64_t xs = x_col_stride > 0 ? x_col_stride : in_dim;
    const int64_t ys = y_col_stride > 0 ? y_col_stride : out_dim;
    const int row_bytes = ggml_k_row_bytes(kQtype, in_dim);
    const uint8_t* row0_data =
        w + static_cast<size_t>(row0) * static_cast<size_t>(row_bytes);
    float p0[kCols];
    float p1[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        p0[c] = 0.0f;
        p1[c] = 0.0f;
    }
    if constexpr (kCols == 1) {
        p0[0] = ggml_k_matvec_row_dot<kQtype>(row0_data, x, in_dim, lane);
        if constexpr (kRows == 2) {
            p1[0] = ggml_k_matvec_row_dot<kQtype>(
                row0_data + row_bytes, x, in_dim, lane);
        }
    } else if constexpr (kQtype == QWEN35_LOWBIT_GGML_Q8_0) {
        // Decode each Q8 block once, then fma every column. Same product
        // as ggml_k_matvec_row_dot: p += (d * q) * x.
        const int nblk = in_dim / 32;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t* b0 =
                row0_data + static_cast<size_t>(blk) * 34;
            const float d0 = ggml_f16_to_f32_unaligned(b0);
            const int8_t q0 = reinterpret_cast<const int8_t*>(b0 + 2)[lane];
            const float w0 = d0 * static_cast<float>(q0);
            float w1 = 0.0f;
            if constexpr (kRows == 2) {
                const uint8_t* b1 = row0_data + row_bytes +
                    static_cast<size_t>(blk) * 34;
                const float d1 = ggml_f16_to_f32_unaligned(b1);
                const int8_t q1 =
                    reinterpret_cast<const int8_t*>(b1 + 2)[lane];
                w1 = d1 * static_cast<float>(q1);
            }
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (c >= ncol) {
                    break;
                }
                const float xv =
                    x[static_cast<int64_t>(c) * xs + blk * 32 + lane];
                p0[c] += w0 * xv;
                if constexpr (kRows == 2) {
                    p1[c] += w1 * xv;
                }
            }
        }
    } else {
        // Same per-column row_dot as kCols=1 so B>1 verify stays
        // greedy-identical. Weight tiles are re-read; occupancy still
        // beats WMMA m=1 on nv=48.
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            const float* xc = x + static_cast<int64_t>(c) * xs;
            p0[c] = ggml_k_matvec_row_dot<kQtype>(row0_data, xc, in_dim, lane);
            if constexpr (kRows == 2) {
                p1[c] = ggml_k_matvec_row_dot<kQtype>(
                    row0_data + row_bytes, xc, in_dim, lane);
            }
        }
    }
    for (int off = kWarp / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            p0[c] += __shfl_xor(p0[c], off);
            if constexpr (kRows == 2) {
                p1[c] += __shfl_xor(p1[c], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            float* yc = y + static_cast<int64_t>(c) * ys;
            if constexpr (kAcc) {
                yc[row0] = bf16_round_rne_f32_finite(
                    yc[row0] + bf16_round_rne_f32_finite(p0[c]));
            } else {
                yc[row0] = p0[c];
            }
            if constexpr (kRows == 2) {
                if constexpr (kAcc) {
                    yc[row0 + 1] = bf16_round_rne_f32_finite(
                        yc[row0 + 1] + bf16_round_rne_f32_finite(p1[c]));
                } else {
                    yc[row0 + 1] = p1[c];
                }
            }
        }
    }
}

__global__ void supersonic_qwen35_bf16w_f32_gemv_kernel(
    const hip_bfloat16* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in_dim,
    int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) {
        return;
    }
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    float acc = 0.0f;
    const hip_bfloat16* wr = w + static_cast<size_t>(row) * in_dim;
    for (int c = tid; c < in_dim; c += bs) {
        acc += supersonic_qwen35_to_float(wr[c]) * x[c];
    }
    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = bf16_round_rne_f32_finite(sm[0]);
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
//
// Optimization: wave-per-row matvec with vectorized loads.
// Each wave (32 threads) handles one output row. 8 waves per block process
// 8 rows in parallel. Wave shuffle reduction replaces LDS tree reduction,
// eliminating syncthreads from the hot matvec loop.
// =============================================================================

// Wave-level sum reduction using DPP/shuffle (Wave32 on RDNA3.5)
__device__ inline float wave_reduce_sum_f32(float val) {
    // Use runtime warpSize so this agrees with matvec loops that partition
    // columns by `warpSize * N`. On gfx1150 (wave32) the loop unrolls to
    // the same 5 __shfl_xor ops; on CDNA (wave64) it covers all 64 lanes.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Bit-match `supersonic_qwen35_rms_norm_kernel`: per-wave shfl_down, then
// wave 0 shfl_down of the wave totals. A serial tid==0 add of 8 wave
// sums is a different FP association and can miss a BF16 ULP on QK-norm.
__device__ inline float block_rms_inv_plain(float partial, int dim, float eps, float* scratch) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float warp_sum = supersonic_qwen35_wave_sum(partial);
    if (lane == 0) {
        scratch[warp] = warp_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < (blockDim.x + 31) / 32 ? scratch[lane] : 0.0f;
        block_sum = supersonic_qwen35_wave_sum(block_sum);
        if (lane == 0) {
            const float mean = block_sum / static_cast<float>(dim);
            scratch[0] = rsqrtf(mean + eps);
        }
    }
    __syncthreads();
    return scratch[0];
}

// Packed 2×BF16 vector type for v_dot2_f32_bf16 instruction.
// On GFX11 (RDNA 3.0+), v_dot2_f32_bf16 computes:
//   acc += (a.x * b.x) + (a.y * b.y)  where a,b are packed BF16 pairs
// This does 2 multiply-adds per instruction vs 1 for scalar FMA.
typedef short v2bf16 __attribute__((ext_vector_type(2)));

// Packed 2×BF16 dot product accumulating into F32.
// Compiles to a single v_dot2_f32_bf16 instruction on gfx1100+.
__device__ inline float dot2_bf16_f32(v2bf16 a, v2bf16 b, float acc) {
    return __builtin_amdgcn_fdot2_f32_bf16(a, b, acc, false);
}

// Sense-reversing grid barrier. Every launched block must be resident;
// over-subscription deadlocks because waiters occupy the CU. Do not reset
// `barrier_flag` across decode steps — it is a monotonic phase.
__device__ inline void grid_barrier(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        __threadfence();
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_flag, phase + 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(barrier_flag, __ATOMIC_ACQUIRE) == phase) {}
        }
    }
    __syncthreads();
}

// Per-k-head arrive/wait so Q/K conv_state is not overwritten until every
// V-head that shares that k-head has copied the old taps. Split-2 launches
// one layer per kernel, so the owner resets the slot before the kernel ends.
__device__ inline void gqh_kq_arrive(unsigned int* slot, unsigned seq_key) {
    while (true) {
        const unsigned cur = __atomic_load_n(slot, __ATOMIC_RELAXED);
        const unsigned cur_key = cur >> 16;
        const unsigned cur_cnt = cur & 0xffffu;
        unsigned neu;
        if (cur_key == seq_key) {
            neu = (seq_key << 16) | (cur_cnt + 1u);
        } else if (cur_key < seq_key) {
            neu = (seq_key << 16) | 1u;
        } else {
            return;
        }
        if (atomicCAS(slot, cur, neu) == cur) {
            return;
        }
    }
}

__device__ inline void gqh_kq_wait(
    unsigned int* slot, unsigned seq_key, unsigned n_share) {
    const unsigned need = (seq_key << 16) | n_share;
    while (__atomic_load_n(slot, __ATOMIC_ACQUIRE) != need) {
    }
}

// Same tap order as linear_prefill_conv_pack (sequential acc += x*w).
__device__ inline float gqh_dwconv4_silu(
    float x0, float x1, float x2, float x3,
    float w0, float w1, float w2, float w3) {
    float acc = 0.0f;
    acc += x0 * w0;
    acc += x1 * w1;
    acc += x2 * w2;
    acc += x3 * w3;
    return bf16_round_rne_f32_finite(
        acc * supersonic_qwen35_sigmoid_fast(acc));
}

// Pairwise tree of n=power-of-two terms. Matches split_norm / gated
// RMS (256-wide tree with the upper half zero). Overwrites x[0..n).
__device__ inline float gqh_tree_sum(float* x, int n, int tid) {
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            x[tid] += x[tid + stride];
        }
        __syncthreads();
    }
    return x[0];
}

// Copy attn+state out of standalone delta_recurrent_prefill (seq_len=1)
// and apply the same gated RMS as the inlined mid path.
__global__ void supersonic_qwen35_decode_extract_rec_gated_kernel(
    int nv,
    int khd,
    int vhd,
    float eps,
    const float* __restrict__ rec_out,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    float* __restrict__ rec_state,
    float* __restrict__ attn_out,
    int compact) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= nv) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int out_base = compact ? (h * vhd) : (h * (1 + khd) * vhd);
    const int state_n = khd * vhd;
    if (rec_state != nullptr && !compact) {
        for (int i = tid; i < state_n; i += static_cast<int>(blockDim.x)) {
            rec_state[h * state_n + i] = rec_out[out_base + vhd + i];
        }
    }
    __shared__ float xs[256];
    const float xv = (tid < vhd)
        ? bf16_round_rne_f32_finite(rec_out[out_base + tid])
        : 0.0f;
    if (tid < vhd) {
        xs[tid] = xv * xv;
    } else {
        xs[tid] = 0.0f;
    }
    __syncthreads();
    const float g_sum = gqh_tree_sum(xs, vhd, tid);
    if (tid == 0) {
        xs[0] = rsqrtf(g_sum / static_cast<float>(vhd) + eps);
    }
    __syncthreads();
    if (tid < vhd) {
        const float inv = xs[0];
        const float w = (norm_w != nullptr)
            ? bf16_round_rne_f32_finite(norm_w[tid])
            : 1.0f;
        const float zv = z[h * vhd + tid];
        const float gate_silu = zv * supersonic_qwen35_sigmoid_fast(zv);
        attn_out[h * vhd + tid] =
            bf16_round_rne_f32_finite(xv * inv * w * gate_silu);
    }
}

// Per-channel dwconv4+SiLU+state shift. Each qkv channel is independent, so
// this replaces the split-2 kq handshake + 48-block grid_barrier.
// Lightweight QK-RMS + partial RoPE + BF16 KV append + gate save.
// Replaces split-2 of the 4B megakernel when host_full attention runs.
__global__ void supersonic_qwen35_decode_full_prep_kernel(
    float* __restrict__ proj,
    float* __restrict__ attn_scratch,
    hip_bfloat16* __restrict__ cache_k,
    hip_bfloat16* __restrict__ cache_v,
    const hip_bfloat16* __restrict__ q_norm,
    const hip_bfloat16* __restrict__ k_norm,
    const hip_bfloat16* __restrict__ cos_table,
    const hip_bfloat16* __restrict__ sin_table,
    int nh,
    int nkv,
    int hd,
    int q_out,
    int k_out,
    int rot_dim,
    int kv_max_t,
    int seq0,
    int proj_stride,
    int attn_stride,
    float unit_offset) {
    const int h = static_cast<int>(blockIdx.x);
    const int b = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int seq_off = qwen35_split_seqlen + b;
    (void)seq0;
    float* proj_b = proj + static_cast<int64_t>(b) * proj_stride;
    float* q_f32 = proj_b;
    float* k_f32 = proj_b + q_out;
    float* v_f32 = k_f32 + k_out;
    extern __shared__ float lds[];

    float* qh = q_f32 + h * hd * 2;
    const float qv = (tid < hd) ? qh[tid] : 0.0f;
    float qinv = block_rms_inv_plain(qv * qv, hd, 1e-6f, lds);
    if (tid < hd) {
        const float w = static_cast<float>(q_norm[tid]) + unit_offset;
        qh[tid] = bf16_round_rne_f32_finite(qv * qinv * w);
    }
    __syncthreads();
    float* kh = k_f32 + h * hd;
    const float kv = (h < nkv && tid < hd) ? kh[tid] : 0.0f;
    float kinv = block_rms_inv_plain(kv * kv, hd, 1e-6f, lds);
    if (h < nkv && tid < hd) {
        const float w = static_cast<float>(k_norm[tid]) + unit_offset;
        kh[tid] = bf16_round_rne_f32_finite(kv * kinv * w);
    }
    __syncthreads();
    if (cos_table != nullptr && rot_dim > 0) {
        const int half_rot = rot_dim / 2;
        const size_t cos_off = static_cast<size_t>(seq_off) * half_rot;
        if (h < nh && tid < half_rot) {
            float* qh = q_f32 + h * hd * 2;
            const float c = static_cast<float>(cos_table[cos_off + tid]);
            const float s = static_cast<float>(sin_table[cos_off + tid]);
            const float x0 = qh[tid];
            const float x1 = qh[half_rot + tid];
            qh[tid] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
            qh[half_rot + tid] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
        }
        if (h < nkv && tid < half_rot) {
            float* kh = k_f32 + h * hd;
            const float c = static_cast<float>(cos_table[cos_off + tid]);
            const float s = static_cast<float>(sin_table[cos_off + tid]);
            const float x0 = kh[tid];
            const float x1 = kh[half_rot + tid];
            kh[tid] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
            kh[half_rot + tid] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
        }
    }
    __syncthreads();
    if (h < nkv && tid < hd && cache_k != nullptr && cache_v != nullptr) {
        const size_t offset = static_cast<size_t>(h) * kv_max_t * hd +
            static_cast<size_t>(seq_off) * hd + tid;
        cache_k[offset] = hip_bfloat16(k_f32[h * hd + tid]);
        cache_v[offset] = hip_bfloat16(v_f32[h * hd + tid]);
    }
    if (h < nh && tid < hd) {
        float* saved_gate =
            attn_scratch + static_cast<int64_t>(b) * attn_stride + nh * hd;
        saved_gate[h * hd + tid] = bf16_round_rne_f32_finite(
            q_f32[h * hd * 2 + hd + tid]);
    }
}

__global__ void supersonic_qwen35_decode_rec_conv_kernel(
    int conv_dim,
    const float* __restrict__ qkv_f32,
    hip_bfloat16* __restrict__ conv_state,
    const hip_bfloat16* __restrict__ conv_w,
    float* __restrict__ conv_out,
    int ncols,
    int64_t qkv_col_stride,
    int64_t out_col_stride,
    hip_bfloat16* __restrict__ conv_snaps,
    int64_t snap_col_stride) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= conv_dim) {
        return;
    }
    hip_bfloat16* cs = conv_state + static_cast<size_t>(c) * 3;
    const hip_bfloat16* cw = conv_w + static_cast<size_t>(c) * 4;
    const int n = ncols > 0 ? ncols : 1;
    const int64_t q_stride = qkv_col_stride > 0 ? qkv_col_stride : conv_dim;
    const int64_t o_stride = out_col_stride > 0 ? out_col_stride : conv_dim;
    for (int col = 0; col < n; ++col) {
        const hip_bfloat16 qnew =
            hip_bfloat16(qkv_f32[static_cast<int64_t>(col) * q_stride + c]);
        conv_out[static_cast<int64_t>(col) * o_stride + c] = gqh_dwconv4_silu(
            supersonic_qwen35_to_float(cs[0]),
            supersonic_qwen35_to_float(cs[1]),
            supersonic_qwen35_to_float(cs[2]),
            supersonic_qwen35_to_float(qnew),
            supersonic_qwen35_to_float(cw[0]),
            supersonic_qwen35_to_float(cw[1]),
            supersonic_qwen35_to_float(cw[2]),
            supersonic_qwen35_to_float(cw[3]));
        cs[0] = cs[1];
        cs[1] = cs[2];
        cs[2] = qnew;
        if (conv_snaps != nullptr && snap_col_stride > 0) {
            hip_bfloat16* snap = conv_snaps +
                static_cast<int64_t>(col) * snap_col_stride +
                static_cast<int64_t>(c) * 3;
            snap[0] = cs[0];
            snap[1] = cs[1];
            snap[2] = cs[2];
        }
    }
}

// Q then K RMS. blockIdx.x < nk is Q, else K. Tree sum matches split-2.
__global__ void supersonic_qwen35_decode_rec_qk_rms_kernel(
    int nk,
    int hkd,
    int key_dim,
    int conv_dim,
    float* __restrict__ conv_out,
    int64_t col_stride) {
    const int is_k = static_cast<int>(blockIdx.x) >= nk;
    const int h = is_k ? static_cast<int>(blockIdx.x) - nk
                       : static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    float* row = conv_out + static_cast<int64_t>(col) * col_stride;
    const int src_off = (is_k ? key_dim : 0) + h * hkd;
    const int dst_off = conv_dim + (is_k ? key_dim : 0) + h * hkd;
    __shared__ float xs[256];
    const float qv = (tid < hkd) ? row[src_off + tid] : 0.0f;
    xs[tid] = (tid < hkd) ? qv * qv : 0.0f;
    __syncthreads();
    const float q_sum = gqh_tree_sum(xs, hkd, tid);
    const float inv = rsqrtf(q_sum + 1e-6f);
    if (tid < hkd) {
        float y = qv * inv;
        if (!is_k) {
            y *= 1.0f / sqrtf(static_cast<float>(hkd));
        }
        row[dst_off + tid] = y;
    }
}

__device__ inline void grid_barrier_reset_counter(
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    int num_blocks,
    unsigned int* counter_to_reset
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int phase = __atomic_load_n(barrier_flag, __ATOMIC_RELAXED);
        __threadfence();
        unsigned int old = atomicAdd(barrier_counter, 1u);
        if (old == static_cast<unsigned int>(num_blocks) - 1) {
            __atomic_store_n(counter_to_reset, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(barrier_counter, 0u, __ATOMIC_RELAXED);
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
        dst[c] = src[c] * inv * (supersonic_qwen35_to_float(weight[c]) + 1.0f);
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

__device__ __forceinline__ void qwen35_4b_hip_record_timing(
    unsigned long long* timing_slots,
    int slot,
    unsigned long long start_clock
) {
    if (timing_slots != nullptr && threadIdx.x == 0) {
        atomicMax(&timing_slots[slot], clock64() - start_clock);
    }
}

// Isolated so split-2 host-rec (flag 128) does not allocate st[256] in
// the parent kernel. That array was 1024 B of scratch and crushed occupancy.
template <typename T>
__device__ __noinline__ void qwen35_split2_rec_st256(
    int tid,
    int h,
    int hvd,
    int hkd,
    int h_k,
    int v,
    int key_dim,
    int conv_dim,
    int q_key_offset,
    int k_key_offset,
    int v_val_offset,
    bool cache_kq,
    bool head_local,
    const float* b_f32_,
    const float* a_f32_,
    const T* dt_bw,
    const T* ale_w,
    float* sh,
    float* conv_out,
    float* k_s,
    const float* q_s,
    float v_conv,
    float* attn_scratch_b_) {
    if (tid >= hvd) {
        return;
    }
    const float beta = 1.0f / (1.0f + expf(-b_f32_[h]));
    float decay = 1.0f;
    if (dt_bw != nullptr && ale_w != nullptr) {
        const float sp = logf(1.0f + expf(
            a_f32_[h] + supersonic_qwen35_to_float(dt_bw[h])));
        const float g = -sp * supersonic_qwen35_to_float(ale_w[h]);
        decay = expf(g);
    }
    float st[256];
    const int nk_rec = hkd < 256 ? hkd : 256;
    for (int k = 0; k < nk_rec; ++k) {
        st[k] = sh[k * hvd + v];
    }
    for (int k = 0; k < nk_rec; ++k) {
        st[k] *= decay;
    }
    float kv_mem_v = 0.0f;
    for (int k = 0; k < nk_rec; ++k) {
        const float kk = cache_kq
            ? k_s[k]
            : conv_out[k_key_offset + h_k * hkd + k];
        kv_mem_v += st[k] * kk;
    }
    const float val = head_local
        ? v_conv
        : conv_out[v_val_offset + h * hvd + v];
    const float delta = (val - kv_mem_v) * beta;
    for (int k = 0; k < nk_rec; ++k) {
        const float kk = cache_kq
            ? k_s[k]
            : conv_out[k_key_offset + h_k * hkd + k];
        st[k] += kk * delta;
    }
    float out_v = 0.0f;
    for (int k = 0; k < nk_rec; ++k) {
        const float qq = cache_kq
            ? q_s[k]
            : conv_out[q_key_offset + h_k * hkd + k];
        out_v += st[k] * qq;
    }
    for (int k = 0; k < nk_rec; ++k) {
        sh[k * hvd + v] = st[k];
    }
    const float out_r = bf16_round_rne_f32_finite(out_v);
    if (head_local) {
        k_s[v] = out_r;
        conv_out[conv_dim + key_dim * 2 + h * hvd + v] = out_r;
    } else {
        attn_scratch_b_[h * hvd + v] = out_r;
    }
}

// kSplit: 0 = full kernel (legacy). 1 = RMS + in-projections. 2 = attn/linear
// core. 3 = o/out residual. 4 = post-RMS + gate. 6 = up + SwiGLU. 5 = MLP
// down + residual store. Separate instantiations drop unused VGPR live ranges
// so a 3-block-per-WGP grid can actually reside.
// io_flags: bit0 = load hidden_io→workspace, bit1 = store workspace→hidden_io.
// kSplit==0 always loads and stores.
template <typename T, int kSplit = 0>
__global__ __launch_bounds__(256, 2) void supersonic_qwen35_persistent_decode_kernel(
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
    unsigned long long* __restrict__ timing_slots,  // nullptr disables; else [num_layers * 37]
    const T* __restrict__ cos_table,   // [max_positions, rotary_dim/2] RoPE cos
    const T* __restrict__ sin_table,   // [max_positions, rotary_dim/2] RoPE sin
    int rotary_dim,                      // partial rotary dimension (64 for canonical Qwen3.8-27B)
    int proj_buf_floats,                 // max projection output buffer size
    int attn_scratch_floats,             // attention/recurrent scratch buffer size
    int enable_attention_trace,          // 1 to emit full-attn trace scratch, 0 for hot path
    const Qwen35FP8ScaleDesc* __restrict__ fp8_scales,  // nullptr for BF16, else [num_layers]
    const KVCacheFp8Desc* __restrict__ kv_fp8,          // nullptr for BF16 KV, else [num_layers]
    int batch_size,                      // 1 for single-sequence, >1 for batched
    const BatchSeqDesc* __restrict__ batch_descs,  // nullptr for single, else [num_layers]
    const Qwen35INT4ScaleDesc* __restrict__ int4_scales,  // nullptr for non-INT4, else [num_layers]
    int io_flags
) {
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int nb = gridDim.x;
    const int B = batch_size;
    const bool emit_attention_trace = enable_attention_trace != 0;

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
    // Activations stream from global workspace. LDS is reduction scratch
    // plus an optional 256-float FP8 LUT.
    float* fp8_lut = lds + bs;

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
    if constexpr (kSplit == 0 || kSplit == 1) {
        if (kSplit == 0 || (io_flags & 1) != 0) {
            for (int b = 0; b < B; b++) {
                for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
                    hidden_f32[b * hidden_dim + c] =
                        supersonic_qwen35_to_float(hidden_io[b * hidden_dim + c]);
                }
            }
            grid_barrier(barrier_counter, barrier_flag, nb);
        }
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        const Qwen35DecodeLayerDesc& L = layers[layer];
        unsigned long long* layer_timing_slots = timing_slots != nullptr
            ? timing_slots + static_cast<size_t>(layer) * QWEN35_4B_TIMING_COUNT
            : nullptr;

        if constexpr (kSplit == 0 || kSplit == 1) {
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
                        const float norm_unit_offset = L.rms_norm_add_unit_offset ? 1.0f : 0.0f;
                        dst[c] = supersonic_qwen35_from_float<T>(
                            src[c] * inv_rms * (supersonic_qwen35_to_float(norm_w[c]) + norm_unit_offset));
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);
        } // kSplit IN: RMS

        // === Token mixer: projections + core ===
        if (L.layer_type == 1) {
            // ---- FULL ATTENTION ----
            const unsigned long long full_attn_clock = clock64();
            if constexpr (kSplit == 0 || kSplit == 1) {
            // Step A: Q/K/V projections via work-stealing
            // Q: [q_out_dim, hidden_dim], K: [k_out_dim, hidden_dim], V: same as K
            const bool skip_in_steal = (kSplit == 1) && ((io_flags & 32) != 0);
            const unsigned long long full_attn_proj_clock = clock64();
            if (!skip_in_steal) {
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
                    if constexpr (kSplit == 1) {
                        if (sr < static_cast<unsigned int>(L.q_out_dim)) {
                            continue;
                        }
                        if (sr < static_cast<unsigned int>(L.q_out_dim + L.k_out_dim)) {
                            if ((io_flags & 8) != 0) {
                                continue;
                            }
                        } else if ((io_flags & 16) != 0) {
                            continue;
                        }
                    }

                    const void* w_raw;
                    const void* w_scale = nullptr;
                    const void* w_i4_scale = nullptr;
                    const void* w_i4_zero = nullptr;
                    const void* w_i4_awq_inv_scale = nullptr;
                    int w_qtype = 0;
                    int row;
                    if (sr < static_cast<unsigned int>(L.q_out_dim)) {
                        w_raw = L.q_proj_w;
                        row = sr;
                        if (fp8_scales) w_scale = fp8_scales[layer].q_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].q_proj_scale; w_i4_zero = int4_scales[layer].q_proj_zero; w_i4_awq_inv_scale = int4_scales[layer].q_proj_awq_inv_scale; }
                        if (int4_scales) w_qtype = int4_scales[layer].q_proj_type;
                    } else if (sr < static_cast<unsigned int>(L.q_out_dim + L.k_out_dim)) {
                        w_raw = L.k_proj_w;
                        row = sr - L.q_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].k_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].k_proj_scale; w_i4_zero = int4_scales[layer].k_proj_zero; w_i4_awq_inv_scale = int4_scales[layer].k_proj_awq_inv_scale; }
                        if (int4_scales) w_qtype = int4_scales[layer].k_proj_type;
                    } else {
                        w_raw = L.v_proj_w;
                        row = sr - L.q_out_dim - L.k_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].v_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].v_proj_scale; w_i4_zero = int4_scales[layer].v_proj_zero; w_i4_awq_inv_scale = int4_scales[layer].v_proj_awq_inv_scale; }
                        if (int4_scales) w_qtype = int4_scales[layer].v_proj_type;
                    }

                    float p[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) p[b] = 0.0f;
                    if (qwen35_lowbit_supported(w_qtype)) {
                        const int gsz = int4_scales[layer].group_size;
                        if (qwen35_lowbit_gqh(w_qtype)) {
                            qwen35_gqh_lane_dots(
                                w_raw, w_qtype, row, hidden_dim, w_i4_scale,
                                normed, hidden_dim, B, p, lane_p);
                        } else if (qwen35_lowbit_native_int4(w_qtype)) {
                            // INT4 dequant: load 4 packed bytes (8 weights) at a time
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
                                apply_awq_inv_scale_8(w, w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++) {
                                    const float* inp = normed + b * hidden_dim + c;
                                    p[b] += w[0]*inp[0] + w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3]
                                          + w[4]*inp[4] + w[5]*inp[5] + w[6]*inp[6] + w[7]*inp[7];
                                }
                            }
                            for (int c = vd8 + lane_p; c < hidden_dim; c += warpSize) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz) *
                                          awq_inv_scale_at(w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
                        } else {
                            for (int c = lane_p; c < hidden_dim; c += warpSize) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
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
                                const float* inp = normed + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = fp8_dequant_weight_lut(w_raw, w_scale, row, c, hidden_dim, bsz, fp8_lut);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
                        }
                    } else {
                        const T* wr = static_cast<const T*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = lane_p * 4; c < vd4; c += warpSize * 4) {
                            float w0 = supersonic_qwen35_to_float(wr[c]);
                            float w1 = supersonic_qwen35_to_float(wr[c+1]);
                            float w2 = supersonic_qwen35_to_float(wr[c+2]);
                            float w3 = supersonic_qwen35_to_float(wr[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = normed + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = supersonic_qwen35_to_float(wr[c]);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
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
            qwen35_4b_hip_record_timing(
                layer_timing_slots,
                QWEN35_4B_TIMING_FULL_ATTN_PROJ,
                full_attn_proj_clock);
            grid_barrier(barrier_counter, barrier_flag, nb);
            } // !skip_in_steal
            } // kSplit IN: full QKV projs

            if constexpr (kSplit == 0 || kSplit == 2 || kSplit == 3) {
            // Step B-G: QK-norm, RoPE, KV cache, attention, gating, o_proj
            // Each block owns Q-heads h = bid, bid+nb, ... (full-attn mid
            // launches nh blocks). O_proj is a separate split / GEMV.
            {
                const int hd = L.attn_head_dim;       // 256
                const int nh = L.attn_num_heads;       // 8
                const int nkv = L.attn_num_kv_heads;   // 2
                const int kv_groups = nh / nkv;         // 4
                const float scale = 1.0f / sqrtf(static_cast<float>(hd));
                const int rot_dim = hd / 4;             // 64 (partial_rotary_factor=0.25)
                const int attn_size = nh * hd;          // 2048

            for (int b = 0; b < B; b++) {
                float* proj_b = proj_buf + b * proj_buf_floats;
                if constexpr (kSplit == 0 || kSplit == 2) {
                const unsigned long long full_attn_core_clock = clock64();
                // Per-sequence state: read from batch_descs when batched, else from L/seqlen_offset
                const int seq_off_use = (kSplit == 0) ? seqlen_offset : qwen35_split_seqlen;
                const int seq_off_b  = batch_descs ? batch_descs[layer].seqlen_offset[b] : seq_off_use;
                void* kv_k_b         = batch_descs ? batch_descs[layer].kv_cache_k[b]    : L.kv_cache_k;
                void* kv_v_b         = batch_descs ? batch_descs[layer].kv_cache_v[b]    : L.kv_cache_v;
                const int kv_len_b   = (batch_descs ? batch_descs[layer].kv_len[b]       : L.kv_len) + 1;
                const int kv_max_b   = batch_descs ? batch_descs[layer].kv_max_t[b]      : L.kv_max_t;
                void* kv_shadow_k_b  = batch_descs ? batch_descs[layer].kv_shadow_k[b]   : L.kv_shadow_k;
                void* kv_shadow_v_b  = batch_descs ? batch_descs[layer].kv_shadow_v[b]   : L.kv_shadow_v;
                const int kv_shadow_start_b = batch_descs ? batch_descs[layer].kv_shadow_start[b] : L.kv_shadow_start;
                void* kv_scale_k_b   = (batch_descs && kv_fp8) ? batch_descs[layer].kv_scale_k[b] : (kv_fp8 ? kv_fp8[layer].kv_scale_k : nullptr);
                void* kv_scale_v_b   = (batch_descs && kv_fp8) ? batch_descs[layer].kv_scale_v[b] : (kv_fp8 ? kv_fp8[layer].kv_scale_v : nullptr);

                float* q_f32 = proj_b;
                float* k_f32 = proj_b + L.q_out_dim;
                float* v_f32 = proj_b + L.q_out_dim + L.k_out_dim;

                // Step B: QK-norm — RMSNorm per head on head_dim.
                // Each block owns Q-heads h = bid, bid+nb, ... so a 24-head
                // layer can run 24-wide instead of one serial block.
                // Reduction matches `rms_norm_rows_plain` / rms_norm_kernel
                // (wave shfl_down + wave-0 shfl_down), not a serial 8-add.
                {
                    for (int h = blockIdx.x; h < nh; h += nb) {
                        float* qh = q_f32 + h * hd * 2;
                        float sq = 0.0f;
                        for (int d = tid; d < hd; d += bs) sq += qh[d] * qh[d];
                        float inv = block_rms_inv_plain(sq, hd, 1e-6f, lds);
                        const T* qnw = static_cast<const T*>(L.q_norm_w);
                        const float qk_unit = L.rms_norm_add_unit_offset ? 1.0f : 0.0f;
                        for (int d = tid; d < hd; d += bs) {
                            const float weight_val =
                                supersonic_qwen35_to_float(qnw[d]) + qk_unit;
                            qh[d] = bf16_round_rne_f32_finite(qh[d] * inv * weight_val);
                        }
                        __syncthreads();
                    }
                    for (int h = blockIdx.x; h < nkv; h += nb) {
                        float* kh = k_f32 + h * hd;
                        float sq = 0.0f;
                        for (int d = tid; d < hd; d += bs) sq += kh[d] * kh[d];
                        float inv = block_rms_inv_plain(sq, hd, 1e-6f, lds);
                        const T* knw = static_cast<const T*>(L.k_norm_w);
                        const float qk_unit = L.rms_norm_add_unit_offset ? 1.0f : 0.0f;
                        for (int d = tid; d < hd; d += bs) {
                            const float weight_val =
                                supersonic_qwen35_to_float(knw[d]) + qk_unit;
                            kh[d] = bf16_round_rne_f32_finite(kh[d] * inv * weight_val);
                        }
                        __syncthreads();
                    }
                }

                // Step C: RoPE (partial, first rot_dim dims of head_dim)
                if (cos_table != nullptr && rot_dim > 0) {
                    const int half_rot = rot_dim / 2;
                    const size_t cos_off =
                        static_cast<size_t>(seq_off_b) * half_rot;

                    for (int h = blockIdx.x; h < nh; h += nb) {
                        float* qh = q_f32 + h * hd * 2;
                        for (int i = tid; i < half_rot; i += bs) {
                            float c = supersonic_qwen35_to_float(cos_table[cos_off + i]);
                            float s = supersonic_qwen35_to_float(sin_table[cos_off + i]);
                            float x0 = qh[i];
                            float x1 = qh[half_rot + i];
                            qh[i] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
                            qh[half_rot + i] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
                        }
                    }
                    __syncthreads();

                    for (int h = blockIdx.x; h < nkv; h += nb) {
                        float* kh = k_f32 + h * hd;
                        for (int i = tid; i < half_rot; i += bs) {
                            float c = supersonic_qwen35_to_float(cos_table[cos_off + i]);
                            float s = supersonic_qwen35_to_float(sin_table[cos_off + i]);
                            float x0 = kh[i];
                            float x1 = kh[half_rot + i];
                            kh[i] = bf16_round_rne_f32_finite(x0 * c - x1 * s);
                            kh[half_rot + i] = bf16_round_rne_f32_finite(x0 * s + x1 * c);
                        }
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

                        for (int h = blockIdx.x; h < nkv; h += nb) {
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
                                    shadow_k[offset] = supersonic_qwen35_from_float<T>(k_store);
                                    shadow_v[offset] = supersonic_qwen35_from_float<T>(v_store);
                                }
                            }
                        }
                    } else {
                        T* cache_k = static_cast<T*>(kv_k_b);
                        T* cache_v = static_cast<T*>(kv_v_b);
                        for (int h = blockIdx.x; h < nkv; h += nb) {
                            for (int d = tid; d < hd; d += bs) {
                                const size_t offset =
                                    static_cast<size_t>(h) * kv_max_b * hd +
                                    static_cast<size_t>(seq_off_b) * hd + d;
                                cache_k[offset] = supersonic_qwen35_from_float<T>(k_f32[h * hd + d]);
                                cache_v[offset] = supersonic_qwen35_from_float<T>(v_f32[h * hd + d]);
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
                if (emit_attention_trace) {
                    for (int h = blockIdx.x; h < nh; h += nb) {
                        for (int d = tid; d < hd; d += bs) {
                            saved_q[h * hd + d] = bf16_round_rne_f32_finite(
                                q_f32[h * hd * 2 + d]);
                        }
                    }
                }
                for (int h = blockIdx.x; h < nh; h += nb) {
                    for (int d = tid; d < hd; d += bs) {
                        saved_gate[h * hd + d] = bf16_round_rne_f32_finite(
                            q_f32[h * hd * 2 + hd + d]);
                    }
                }
                __syncthreads();

                const bool host_full = (kSplit == 2) && ((io_flags & 256) != 0);
                if (host_full) {
                    grid_barrier(barrier_counter, barrier_flag, nb);
                    qwen35_4b_hip_record_timing(
                        layer_timing_slots,
                        QWEN35_4B_TIMING_FULL_ATTN_CORE_BASE + b,
                        full_attn_core_clock);
                    continue;
                }

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
                    for (int qh = blockIdx.x; qh < nh; qh += nb) {
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
                                k_val = supersonic_qwen35_to_float(shadow_k[pos_base + tid]);
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
                            float total = 0.0f;
                            for (int w = 0; w < attn_nwaves; ++w) total += lds[w];
                            const float score_val = total * scale;
                            if (emit_attention_trace && tid == 0) {
                                saved_scores[qh * kv_max_b + t] = score_val;
                            }

                            float v_val = 0.0f;
                            if (kv_shadow_k_b != nullptr && kv_shadow_v_b != nullptr &&
                                kv_shadow_start_b >= 0 && t >= kv_shadow_start_b) {
                                const T* shadow_v = static_cast<const T*>(kv_shadow_v_b);
                                v_val = supersonic_qwen35_to_float(shadow_v[pos_base + tid]);
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
                    // Flash-decode: 8 warps split the KV axis, then merge
                    // online-softmax stats. The previous path used one warp
                    // of a 256-thread block and scanned every key serially.
                    const T* ck = static_cast<const T*>(kv_k_b);
                    const T* cv = static_cast<const T*>(kv_v_b);
                    const int lane = tid % warpSize;
                    const int warp = tid / warpSize;
                    const int nwarps = bs / warpSize;

                    for (int qh = blockIdx.x; qh < nh; qh += nb) {
                        const int kvh = qh / kv_groups;
                        const float* q_head = q_f32 + qh * hd * 2;
                        const size_t kv_head_base =
                            static_cast<size_t>(kvh) * kv_max_b * hd;

                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        int acc_dim[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                        int acc_count = 0;
                        for (int d = lane; d < hd && acc_count < 8;
                             d += warpSize) {
                            acc_dim[acc_count++] = d;
                            q_reg[acc_count - 1] = q_head[d];
                        }
                        float m_i = -INFINITY;
                        float l_i = 0.0f;
                        for (int t = warp; t < kv_len_b; t += nwarps) {
                            const size_t pos =
                                kv_head_base + static_cast<size_t>(t) * hd;
                            float partial = 0.0f;
                            for (int i = 0; i < acc_count; ++i) {
                                partial += q_reg[i] *
                                    supersonic_qwen35_to_float(ck[pos + acc_dim[i]]);
                            }
                            const float score = __shfl(
                                supersonic_qwen35_wave_sum(partial), 0) *
                                scale;
                            if (emit_attention_trace && lane == 0) {
                                saved_scores[qh * kv_max_b + t] = score;
                            }
                            const float m_old = m_i;
                            const float m_new =
                                score > m_old ? score : m_old;
                            const float alpha = (m_old == -INFINITY)
                                ? 0.0f
                                : __expf(m_old - m_new);
                            const float weight = __expf(score - m_new);
                            l_i = l_i * alpha + weight;
                            m_i = m_new;
                            for (int i = 0; i < acc_count; ++i) {
                                const float v = supersonic_qwen35_to_float(
                                    cv[pos + acc_dim[i]]);
                                acc[i] = acc[i] * alpha + weight * v;
                            }
                        }
                        // Merge per-warp (m, l, acc) with the online-softmax
                        // rescale identity. LDS layout: m[8] | l[8] | acc[8][8].
                        if (lane == 0) {
                            lds[warp] = m_i;
                            lds[nwarps + warp] = l_i;
                        }
                        for (int i = 0; i < acc_count; ++i) {
                            lds[2 * nwarps + warp * 8 + i] = acc[i];
                        }
                        __syncthreads();
                        float m_all = -INFINITY;
                        for (int w = 0; w < nwarps; ++w) {
                            m_all = fmaxf(m_all, lds[w]);
                        }
                        float l_all = 0.0f;
                        float acc_all[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (int w = 0; w < nwarps; ++w) {
                            const float m_w = lds[w];
                            const float alpha = (m_w == -INFINITY)
                                ? 0.0f
                                : __expf(m_w - m_all);
                            l_all += lds[nwarps + w] * alpha;
                            for (int i = 0; i < acc_count; ++i) {
                                acc_all[i] +=
                                    lds[2 * nwarps + w * 8 + i] * alpha;
                            }
                        }
                        const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
                        for (int i = 0; i < acc_count; ++i) {
                            attn_flat[qh * hd + acc_dim[i]] =
                                bf16_round_rne_f32_finite(acc_all[i] * inv);
                        }
                        __syncthreads();
                    }
                }

                if (emit_attention_trace) {
                    for (int h = blockIdx.x; h < nh; h += nb) {
                        for (int d = tid; d < hd; d += bs) {
                            saved_pre_gate[h * hd + d] = attn_flat[h * hd + d];
                        }
                    }
                }
                __syncthreads();

                // Step F: Gate
                for (int h = blockIdx.x; h < nh; h += nb) {
                    for (int d = tid; d < hd; d += bs) {
                        const int i = h * hd + d;
                        float gate_val = saved_gate[i];
                        // Exact sigmoid matches component
                        // cast_transpose_gate (not sigmoid_fast).
                        float sigmoid_gate =
                            1.0f / (1.0f + expf(-gate_val));
                        attn_flat[i] = bf16_round_rne_f32_finite(
                            attn_flat[i] * sigmoid_gate);
                    }
                }
                __syncthreads();

                // Grid barrier: every block wrote its Q-heads into attn_flat.
                grid_barrier(barrier_counter, barrier_flag, nb);
                qwen35_4b_hip_record_timing(
                    layer_timing_slots,
                    QWEN35_4B_TIMING_FULL_ATTN_CORE_BASE + b,
                    full_attn_core_clock);
                } // kSplit MID: full attn core

                if constexpr (kSplit == 0 || kSplit == 3) {
                const unsigned long long full_attn_out_clock = clock64();
                // Step G: o_proj [hidden_dim, attn_size] × attn_flat → hidden_f32 (fused residual)
                if constexpr (kSplit == 3) {
                    if ((io_flags & 4) != 0) {
                        qwen35_4b_hip_record_timing(
                            layer_timing_slots,
                            QWEN35_4B_TIMING_FULL_ATTN_OUT_BASE + b,
                            full_attn_out_clock);
                        continue;
                    }
                }
                {
                    const float* x_o = proj_b;

                    if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                    grid_barrier(barrier_counter, barrier_flag, nb);

                    const int lane_o = tid % warpSize;
                    for (;;) {
                        unsigned int sr;
                        if (lane_o == 0) sr = atomicAdd(&counters[0], 1u);
                        sr = __shfl(sr, 0);
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        const int o_qtype =
                            int4_scales != nullptr ? int4_scales[layer].o_proj_type : 0;
                        if (qwen35_lowbit_supported(o_qtype)) {
                            const int gsz = int4_scales[layer].group_size;
                            const void* o_awq_inv = int4_scales[layer].o_proj_awq_inv_scale;
                            if (qwen35_lowbit_gqh(o_qtype) && kv_fp8 == nullptr) {
                                qwen35_gqh_lane_dots(
                                    L.o_proj_w, o_qtype, sr, attn_size,
                                    int4_scales[layer].o_proj_scale,
                                    x_o, attn_size, 1, &p, lane_o);
                            } else if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < attn_size; ++c) {
                                        p += lowbit_dequant_scalar(
                                            L.o_proj_w,
                                            int4_scales[layer].o_proj_scale,
                                            int4_scales[layer].o_proj_zero,
                                            o_qtype,
                                            sr,
                                            c,
                                            attn_size,
                                            gsz) * awq_inv_scale_at(o_awq_inv, c) * x_o[c];
                                    }
                                }
                            } else if (qwen35_lowbit_native_int4(o_qtype)) {
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
                                    apply_awq_inv_scale_8(w, o_awq_inv, c);
                                    p += w[0]*x_o[c] + w[1]*x_o[c+1] + w[2]*x_o[c+2] + w[3]*x_o[c+3]
                                       + w[4]*x_o[c+4] + w[5]*x_o[c+5] + w[6]*x_o[c+6] + w[7]*x_o[c+7];
                                }
                                for (int c = as8 + lane_o; c < attn_size; c += warpSize)
                                    p += lowbit_dequant_scalar(L.o_proj_w, int4_scales[layer].o_proj_scale, int4_scales[layer].o_proj_zero, o_qtype, sr, c, attn_size, gsz) *
                                         awq_inv_scale_at(o_awq_inv, c) * x_o[c];
                            } else {
                                for (int c = lane_o; c < attn_size; c += warpSize)
                                    p += lowbit_dequant_scalar(L.o_proj_w, int4_scales[layer].o_proj_scale, int4_scales[layer].o_proj_zero, o_qtype, sr, c, attn_size, gsz) *
                                         x_o[c];
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
                                            fp8_lut) * x_o[c];
                                    }
                                }
                            } else {
                                for (int c = lane_o; c < attn_size; c += warpSize)
                                    p += fp8_dequant_weight_lut(L.o_proj_w, fp8_scales[layer].o_proj_scale, sr, c, attn_size, fp8_scales[layer].block_size, fp8_lut) * x_o[c];
                            }
                        } else {
                            const T* wr = static_cast<const T*>(L.o_proj_w) + static_cast<size_t>(sr) * attn_size;
                            if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < attn_size; ++c) {
                                        p += supersonic_qwen35_to_float(wr[c]) * x_o[c];
                                    }
                                }
                            } else {
                                const int as4 = attn_size & ~3;
                                for (int c = lane_o * 4; c < as4; c += warpSize * 4) {
                                    p += supersonic_qwen35_to_float(wr[c])   * x_o[c]
                                       + supersonic_qwen35_to_float(wr[c+1]) * x_o[c+1]
                                       + supersonic_qwen35_to_float(wr[c+2]) * x_o[c+2]
                                       + supersonic_qwen35_to_float(wr[c+3]) * x_o[c+3];
                                }
                                for (int c = as4 + lane_o; c < attn_size; c += warpSize)
                                    p += supersonic_qwen35_to_float(wr[c]) * x_o[c];
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
                qwen35_4b_hip_record_timing(
                    layer_timing_slots,
                    QWEN35_4B_TIMING_FULL_ATTN_OUT_BASE + b,
                    full_attn_out_clock);
                grid_barrier(barrier_counter, barrier_flag, nb);
                } // kSplit OUT: o_proj
            } // end for (b) batch loop
            } // end full attention scope
            if constexpr (kSplit == 0) {
            qwen35_4b_hip_record_timing(
                layer_timing_slots, QWEN35_4B_TIMING_FULL_ATTN, full_attn_clock);
            }
            } // kSplit MID||OUT full attn

        } else {
            // ---- LINEAR ATTENTION ----
            const int nv_heads = L.linear_num_v_heads;
            if constexpr (kSplit == 0 || kSplit == 1) {
            // Step A: qkv/z/b/a projections via work-stealing
            const bool skip_in_steal = (kSplit == 1) && ((io_flags & 32) != 0);
            const unsigned long long linear_proj_clock = clock64();
            if (!skip_in_steal) {
            if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
            grid_barrier(barrier_counter, barrier_flag, nb);

            // b_proj and a_proj have linear_num_v_heads rows (not num_k_heads)
            const int total_proj = L.qkv_out_dim + L.z_out_dim + nv_heads + nv_heads;
            // INT4 on small models keeps block-level reduction for bit-exact
            // numerics: 0.8B INT4 argmax was knife-edge sensitive to the
            // wave-per-row reassociation (PR #20). 2B/4B/9B INT4 are
            // token-exact under wave-per-row so they fall through to the
            // fast path below alongside BF16/FP8.
            if (int4_scales != nullptr && hidden_dim <= 1024) {
                for (;;) {
                    unsigned int sr;
                    if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                    __shared__ unsigned int shared_sr;
                    if (tid == 0) shared_sr = sr;
                    __syncthreads();
                    sr = shared_sr;
                    if (sr >= static_cast<unsigned int>(total_proj)) break;

                    const void* w_raw;
                    const void* w_scale = nullptr;      // FP8 scale (may coexist)
                    const void* w_i4_scale = nullptr;
                    const void* w_i4_zero = nullptr;
                    const void* w_i4_awq_inv_scale = nullptr;
                    int w_qtype = 0;
                    int row;
                    if (sr < static_cast<unsigned int>(L.qkv_out_dim)) {
                        w_raw = L.qkv_proj_w;
                        row = sr;
                        w_i4_scale = int4_scales[layer].qkv_proj_scale;
                        w_i4_zero = int4_scales[layer].qkv_proj_zero;
                        w_i4_awq_inv_scale = int4_scales[layer].qkv_proj_awq_inv_scale;
                        w_qtype = int4_scales[layer].qkv_proj_type;
                        if (fp8_scales) w_scale = fp8_scales[layer].qkv_proj_scale;
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim)) {
                        w_raw = L.z_proj_w;
                        row = sr - L.qkv_out_dim;
                        w_i4_scale = int4_scales[layer].z_proj_scale;
                        w_i4_zero = int4_scales[layer].z_proj_zero;
                        w_i4_awq_inv_scale = int4_scales[layer].z_proj_awq_inv_scale;
                        w_qtype = int4_scales[layer].z_proj_type;
                        if (fp8_scales) w_scale = fp8_scales[layer].z_proj_scale;
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim + nv_heads)) {
                        w_raw = L.b_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim;
                        w_qtype = int4_scales[layer].b_proj_type;
                        if (fp8_scales) w_scale = fp8_scales[layer].b_proj_scale;
                    } else {
                        w_raw = L.a_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim - nv_heads;
                        w_qtype = int4_scales[layer].a_proj_type;
                        if (fp8_scales) w_scale = fp8_scales[layer].a_proj_scale;
                    }

                    float p[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) p[b] = 0.0f;
                    if (qwen35_lowbit_supported(w_qtype)) {
                        const int gsz = int4_scales[layer].group_size;
                        if (qwen35_lowbit_native_int4(w_qtype)) {
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
                                apply_awq_inv_scale_8(w, w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++) {
                                    const float* inp = normed + b * hidden_dim + c;
                                    p[b] += w[0]*inp[0] + w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3]
                                          + w[4]*inp[4] + w[5]*inp[5] + w[6]*inp[6] + w[7]*inp[7];
                                }
                            }
                            for (int c = vd8 + tid; c < hidden_dim; c += bs) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz) *
                                          awq_inv_scale_at(w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
                        } else {
                            for (int c = tid; c < hidden_dim; c += bs) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
                        }
                    } else if (w_scale != nullptr) {
                        for (int c = tid; c < hidden_dim; c += bs) {
                            float w = fp8_dequant_weight_lut(w_raw, w_scale, row, c, hidden_dim, fp8_scales[layer].block_size, fp8_lut);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
                        }
                    } else {
                        const T* wr = static_cast<const T*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = tid * 4; c < vd4; c += bs * 4) {
                            float w0 = supersonic_qwen35_to_float(wr[c]);
                            float w1 = supersonic_qwen35_to_float(wr[c+1]);
                            float w2 = supersonic_qwen35_to_float(wr[c+2]);
                            float w3 = supersonic_qwen35_to_float(wr[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = normed + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + tid; c < hidden_dim; c += bs) {
                            float w = supersonic_qwen35_to_float(wr[c]);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
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
            } else {
                const int lane_p = tid % warpSize;
                for (;;) {
                    unsigned int sr;
                    if (lane_p == 0) sr = atomicAdd(&counters[0], 1u);
                    sr = __shfl(sr, 0);
                    if (sr >= static_cast<unsigned int>(total_proj)) break;
                    if constexpr (kSplit == 1) {
                        if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim)) {
                            continue;
                        }
                        if (sr < static_cast<unsigned int>(
                                L.qkv_out_dim + L.z_out_dim + nv_heads)) {
                            if ((io_flags & 8) != 0) {
                                continue;
                            }
                        } else if ((io_flags & 16) != 0) {
                            continue;
                        }
                    }

                    const void* w_raw;
                    const void* w_scale = nullptr;
                    const void* w_i4_scale = nullptr;
                    const void* w_i4_zero = nullptr;
                    const void* w_i4_awq_inv_scale = nullptr;
                    int w_qtype = 0;
                    int row;
                    if (sr < static_cast<unsigned int>(L.qkv_out_dim)) {
                        w_raw = L.qkv_proj_w;
                        row = sr;
                        if (fp8_scales) w_scale = fp8_scales[layer].qkv_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].qkv_proj_scale; w_i4_zero = int4_scales[layer].qkv_proj_zero; w_i4_awq_inv_scale = int4_scales[layer].qkv_proj_awq_inv_scale; }
                        if (int4_scales) w_qtype = int4_scales[layer].qkv_proj_type;
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim)) {
                        w_raw = L.z_proj_w;
                        row = sr - L.qkv_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].z_proj_scale;
                        if (int4_scales) { w_i4_scale = int4_scales[layer].z_proj_scale; w_i4_zero = int4_scales[layer].z_proj_zero; w_i4_awq_inv_scale = int4_scales[layer].z_proj_awq_inv_scale; }
                        if (int4_scales) w_qtype = int4_scales[layer].z_proj_type;
                    } else if (sr < static_cast<unsigned int>(L.qkv_out_dim + L.z_out_dim + nv_heads)) {
                        w_raw = L.b_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim;
                        if (fp8_scales) w_scale = fp8_scales[layer].b_proj_scale;
                        if (int4_scales) w_qtype = int4_scales[layer].b_proj_type;
                    } else {
                        w_raw = L.a_proj_w;
                        row = sr - L.qkv_out_dim - L.z_out_dim - nv_heads;
                        if (fp8_scales) w_scale = fp8_scales[layer].a_proj_scale;
                        if (int4_scales) w_qtype = int4_scales[layer].a_proj_type;
                    }

                    float p[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) p[b] = 0.0f;
                    if (qwen35_lowbit_supported(w_qtype)) {
                        const int gsz = int4_scales[layer].group_size;
                        if (qwen35_lowbit_gqh(w_qtype)) {
                            qwen35_gqh_lane_dots(
                                w_raw, w_qtype, row, hidden_dim, w_i4_scale,
                                normed, hidden_dim, B, p, lane_p);
                        } else if (qwen35_lowbit_native_int4(w_qtype)) {
                            const int byte_cols = hidden_dim / 2;
                            const uint8_t* i4_row = static_cast<const uint8_t*>(w_raw) + static_cast<size_t>(row) * byte_cols;
                            const hip_bfloat16* sc = static_cast<const hip_bfloat16*>(w_i4_scale);
                            const hip_bfloat16* zr = static_cast<const hip_bfloat16*>(w_i4_zero);
                            const int sr_g = row / gsz;
                            const int sc_cols = (hidden_dim + gsz - 1) / gsz;
                            const int vd8 = hidden_dim & ~7;
                            for (int c = lane_p * 8; c < vd8; c += warpSize * 8) {
                                uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                                float w[8];
                                int4_dequant_8(pk, sc, zr, sr_g, c, sc_cols, gsz, w);
                                apply_awq_inv_scale_8(w, w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++) {
                                    const float* inp = normed + b * hidden_dim + c;
                                    p[b] += w[0]*inp[0] + w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3]
                                          + w[4]*inp[4] + w[5]*inp[5] + w[6]*inp[6] + w[7]*inp[7];
                                }
                            }
                            for (int c = vd8 + lane_p; c < hidden_dim; c += warpSize) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz) *
                                          awq_inv_scale_at(w_i4_awq_inv_scale, c);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
                        } else {
                            for (int c = lane_p; c < hidden_dim; c += warpSize) {
                                float w = lowbit_dequant_scalar(w_raw, w_i4_scale, w_i4_zero, w_qtype, row, c, hidden_dim, gsz);
                                for (int b = 0; b < B; b++)
                                    p[b] += w * normed[b * hidden_dim + c];
                            }
                        }
                    } else if (fp8_scales != nullptr && w_scale != nullptr) {
                        for (int c = lane_p; c < hidden_dim; c += warpSize) {
                            float w = fp8_dequant_weight_lut(w_raw, w_scale, row, c, hidden_dim, fp8_scales[layer].block_size, fp8_lut);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
                        }
                    } else {
                        const T* wr = static_cast<const T*>(w_raw) + static_cast<size_t>(row) * hidden_dim;
                        const int vd4 = hidden_dim & ~3;
                        for (int c = lane_p * 4; c < vd4; c += warpSize * 4) {
                            float w0 = supersonic_qwen35_to_float(wr[c]);
                            float w1 = supersonic_qwen35_to_float(wr[c+1]);
                            float w2 = supersonic_qwen35_to_float(wr[c+2]);
                            float w3 = supersonic_qwen35_to_float(wr[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = normed + b * hidden_dim + c;
                                p[b] += w0 * inp[0] + w1 * inp[1] + w2 * inp[2] + w3 * inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float w = supersonic_qwen35_to_float(wr[c]);
                            for (int b = 0; b < B; b++)
                                p[b] += w * normed[b * hidden_dim + c];
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
            qwen35_4b_hip_record_timing(
                layer_timing_slots, QWEN35_4B_TIMING_LINEAR_PROJ, linear_proj_clock);
            grid_barrier(barrier_counter, barrier_flag, nb);
            } // !skip_in_steal
            } // kSplit IN: linear projs

            // Step B-E: conv1d, recurrent state, gated norm, out_proj
            // Block 0 handles sequential recurrent operations (per-sequence loop).
            // O_proj uses all blocks via work-stealing (per-sequence loop).
            for (int b = 0; b < B; b++) {
            if constexpr (kSplit == 0 || kSplit == 2) {
            const unsigned long long linear_core_clock = clock64();
            // Per-sequence state
            void* conv_b = batch_descs ? batch_descs[layer].conv_state[b] : L.conv_state;
            void* rec_b  = batch_descs ? batch_descs[layer].recurrent_state[b] : L.recurrent_state;
            float* proj_b = proj_buf + b * proj_buf_floats;

            // Tiny Q8_0 a/b (nv rows × hidden) used to be two dedicated GEMV
            // launches (~30 µs of dispatch each). Compute them here with the
            // mid grid: block h owns head h. Steal/GEMV skip is io_flags 8/16.
            if constexpr (kSplit == 2) {
                if (int4_scales != nullptr &&
                    ((io_flags & 8) != 0 || (io_flags & 16) != 0)) {
                    float* b_dst = proj_b + L.qkv_out_dim + L.z_out_dim;
                    float* a_dst = b_dst + nv_heads;
                    const float* x_ab = normed + b * hidden_dim;
                    const int b_qt = int4_scales[layer].b_proj_type;
                    const int a_qt = int4_scales[layer].a_proj_type;
#if defined(SUPERSONIC_HAS_WMMA_BF16)
                    // Same 16-wide Q8_0 WMMA tiles as the small-m matmul.
                    if (tid < 32 &&
                        static_cast<int>(blockIdx.x) * 16 < nv_heads) {
                        const int col0 = static_cast<int>(blockIdx.x) * 16;
                        if ((io_flags & 8) != 0 &&
                            b_qt == QWEN35_LOWBIT_GGML_Q8_0 &&
                            L.b_proj_w != nullptr) {
                            ggml_q8_0_wmma_m1_tile16(
                                static_cast<const uint8_t*>(L.b_proj_w),
                                x_ab,
                                hidden_dim,
                                col0,
                                nv_heads,
                                b_dst);
                        }
                        if ((io_flags & 16) != 0 &&
                            a_qt == QWEN35_LOWBIT_GGML_Q8_0 &&
                            L.a_proj_w != nullptr) {
                            ggml_q8_0_wmma_m1_tile16(
                                static_cast<const uint8_t*>(L.a_proj_w),
                                x_ab,
                                hidden_dim,
                                col0,
                                nv_heads,
                                a_dst);
                        }
                    }
                    grid_barrier(barrier_counter, barrier_flag, nb);
#else
                    for (int h = blockIdx.x; h < nv_heads; h += nb) {
                        if ((io_flags & 8) != 0 &&
                            b_qt == QWEN35_LOWBIT_GGML_Q8_0 &&
                            L.b_proj_w != nullptr && tid < 32) {
                            const float p = ggml_q8_0_warp_dot(
                                static_cast<const uint8_t*>(L.b_proj_w),
                                h,
                                x_ab,
                                hidden_dim);
                            if (tid == 0) {
                                b_dst[h] = bf16_round_rne_f32_finite(p);
                            }
                        }
                        if ((io_flags & 16) != 0 &&
                            a_qt == QWEN35_LOWBIT_GGML_Q8_0 &&
                            L.a_proj_w != nullptr && tid >= 32 && tid < 64) {
                            const float p = ggml_q8_0_warp_dot(
                                static_cast<const uint8_t*>(L.a_proj_w),
                                h,
                                x_ab,
                                hidden_dim);
                            if (tid == 32) {
                                a_dst[h] = bf16_round_rne_f32_finite(p);
                            }
                        }
                    }
#endif
                }
            }

            const int conv_dim = L.qkv_out_dim;
            const int kern = L.conv_kernel_size;
            const int nv = L.linear_num_v_heads;
            const int nk = (L.qkv_out_dim - nv * L.linear_head_v_dim) /
                (2 * L.linear_head_k_dim);
            const int hkd = L.linear_head_k_dim;
            const int hvd = L.linear_head_v_dim;
            const int key_dim = nk * hkd;
            const bool head_local =
                (kSplit == 2) && (kern == 4) && (hkd * 2 <= bs) && (hkd == hvd);

            if constexpr (kSplit != 2) {
            {
                float* qkv_f32 = proj_b;
                float* gate_up_b = gate_up + b * intermediate_size * 2;

                // Step B: depthwise conv. Channels are independent; stride
                // across the mid grid instead of serializing on block 0.
                {
                    T* cs = static_cast<T*>(conv_b);
                    const T* cw = static_cast<const T*>(L.conv1d_w);
                    float* conv_out = gate_up_b;
                    for (int c = tid + blockIdx.x * bs; c < conv_dim; c += bs * nb) {
                        // Depthwise conv: sum(state[t] * weight[t]) + new_value * weight[kern-1]
                        // Match standard path: convert new value to BF16 for consistency
                        T qkv_bf16 = supersonic_qwen35_from_float<T>(qkv_f32[c]);
                        float acc = 0.0f;
                        if (kern == 4) {
                            const int state_base = c * (kern - 1);
                            const int weight_base = c * kern;
                            const float x0 = supersonic_qwen35_to_float(cs[state_base + 0]);
                            const float x1 = supersonic_qwen35_to_float(cs[state_base + 1]);
                            const float x2 = supersonic_qwen35_to_float(cs[state_base + 2]);
                            const float x3 = supersonic_qwen35_to_float(qkv_bf16);
                            const float w0 = supersonic_qwen35_to_float(cw[weight_base + 0]);
                            const float w1 = supersonic_qwen35_to_float(cw[weight_base + 1]);
                            const float w2 = supersonic_qwen35_to_float(cw[weight_base + 2]);
                            const float w3 = supersonic_qwen35_to_float(cw[weight_base + 3]);
                            acc = ((x0 * w0 + x1 * w1) + (x2 * w2 + x3 * w3));
                        } else {
                            for (int t = 0; t < kern - 1; ++t) {
                                acc += supersonic_qwen35_to_float(cs[c * (kern-1) + t])
                                     * supersonic_qwen35_to_float(cw[c * kern + t]);
                            }
                            acc += supersonic_qwen35_to_float(qkv_bf16)
                                 * supersonic_qwen35_to_float(cw[c * kern + (kern-1)]);
                        }
                        // SiLU activation
                        conv_out[c] = bf16_round_rne_f32_finite(
                            acc * supersonic_qwen35_sigmoid_fast(acc));

                        // Update conv_state: shift left, append new
                        for (int t = 0; t < kern - 2; ++t) {
                            cs[c * (kern-1) + t] = cs[c * (kern-1) + t + 1];
                        }
                        cs[c * (kern-1) + (kern-2)] = qkv_bf16;

                        if (L.debug_linear_trace_out != nullptr &&
                            c == L.debug_linear_trace_channel) {
                            float* debug = static_cast<float*>(L.debug_linear_trace_out);
                            const int state_base = c * (kern - 1);
                            for (int t = 0; t < kern - 1; ++t) {
                                debug[t] = supersonic_qwen35_to_float(cs[state_base + t]);
                            }
                            debug[kern - 1] = supersonic_qwen35_to_float(qkv_bf16);
                            debug[kern] = conv_out[c];
                        }
                    }
                }

            }
            // All blocks wrote their conv channels. Publish before recurrent
            // reads Q/K/V. L2 is folded into the per-block K/Q LDS fill so
            // we do not need a second 48-block barrier after a 16-head L2.
            grid_barrier(barrier_counter, barrier_flag, nb);
            } else if (!head_local) {
                float* qkv_f32 = proj_b;
                float* conv_out = gate_up + b * intermediate_size * 2;
                T* cs = static_cast<T*>(conv_b);
                const T* cw = static_cast<const T*>(L.conv1d_w);
                for (int c = tid + blockIdx.x * bs; c < conv_dim; c += bs * nb) {
                    T qkv_bf16 = supersonic_qwen35_from_float<T>(qkv_f32[c]);
                    float acc = 0.0f;
                    for (int t = 0; t < kern - 1; ++t) {
                        acc += supersonic_qwen35_to_float(cs[c * (kern - 1) + t])
                            * supersonic_qwen35_to_float(cw[c * kern + t]);
                    }
                    acc += supersonic_qwen35_to_float(qkv_bf16)
                        * supersonic_qwen35_to_float(cw[c * kern + (kern - 1)]);
                    conv_out[c] = bf16_round_rne_f32_finite(
                        acc * supersonic_qwen35_sigmoid_fast(acc));
                    for (int t = 0; t < kern - 2; ++t) {
                        cs[c * (kern - 1) + t] = cs[c * (kern - 1) + t + 1];
                    }
                    cs[c * (kern - 1) + (kern - 2)] = qkv_bf16;
                }
                grid_barrier(barrier_counter, barrier_flag, nb);
            }

            // Step C: Parallel delta recurrent state update — multi-block.
            // state: [nv, hkd, hvd] F32 — the recurrent memory.
            // Each thread owns one (head, v) pair. Block bid processes heads
            // h = bid, bid+nb, bid+2*nb, ... — a stride loop so 4B's nv=32
            // still fits in nb=16 blocks (each block handles 2 heads).
            // State slices are disjoint across heads so no cross-block races.
            // All threads enter so the per-head K/Q LDS cache can syncthreads.
            {
                float* proj_b_ = proj_buf + b * proj_buf_floats;
                float* b_f32_ = proj_b_ + L.qkv_out_dim + L.z_out_dim;
                float* a_f32_ = b_f32_ + nv_heads;
                float* conv_out = gate_up + b * intermediate_size * 2;
                float* state = static_cast<float*>(rec_b);
                float* attn_scratch_b_ = attn_scratch + b * attn_scratch_floats;

                const int q_key_offset = 0;
                const int k_key_offset = key_dim;
                const int v_val_offset = key_dim * 2;

                const T* dt_bw = static_cast<const T*>(L.dt_bias_w);
                const T* ale_w = static_cast<const T*>(L.a_log_exp_w);
                T* cs = static_cast<T*>(conv_b);
                const T* cw = static_cast<const T*>(L.conv1d_w);
                const float* qkv_f32 = proj_b;
                unsigned int* kq_flag = reinterpret_cast<unsigned int*>(mlp_out);
                const int seq_off = (kSplit == 2) ? qwen35_split_seqlen : seqlen_offset;

                const int v = tid;
                const bool cache_kq = (hkd * 2 <= bs);
                float* k_s = lds;
                float* q_s = lds + hkd;
                for (int h = blockIdx.x; h < nv; h += nb) {
                    float* sh = state + h * hkd * hvd;
                    const int h_k = h % nk;
                    float v_conv = 0.0f;
                    if (head_local) {
                        // Block h computes V[h] and Q/K for h%nk from conv
                        // state. Arrive after the load so conv overlaps the
                        // 3-way join; owner waits after conv. Same kern=4
                        // product order. Sequential L2 keeps the inv exact.
                        const int q_base = q_key_offset + h_k * hkd;
                        const int k_base = k_key_offset + h_k * hkd;
                        const int v_base = v_val_offset + h * hvd;
                        const bool in_d = (tid < hkd);
                        T qx0, qx1, qx2, qnew;
                        T kx0, kx1, kx2, knew;
                        T vx0, vx1, vx2, vnew;
                        if (in_d) {
                            const int qs = q_base + tid;
                            const int ks = k_base + tid;
                            const int vs = v_base + tid;
                            qx0 = cs[qs * 3 + 0];
                            qx1 = cs[qs * 3 + 1];
                            qx2 = cs[qs * 3 + 2];
                            qnew = supersonic_qwen35_from_float<T>(qkv_f32[qs]);
                            kx0 = cs[ks * 3 + 0];
                            kx1 = cs[ks * 3 + 1];
                            kx2 = cs[ks * 3 + 2];
                            knew = supersonic_qwen35_from_float<T>(qkv_f32[ks]);
                            vx0 = cs[vs * 3 + 0];
                            vx1 = cs[vs * 3 + 1];
                            vx2 = cs[vs * 3 + 2];
                            vnew = supersonic_qwen35_from_float<T>(qkv_f32[vs]);
                        }
                        if (tid == 0) {
                            unsigned n_share = 0;
                            for (int hh = h_k; hh < nv; hh += nk) {
                                ++n_share;
                            }
                            gqh_kq_arrive(
                                &kq_flag[h_k],
                                static_cast<unsigned>(seq_off) + 1u);
                        }
                        if (in_d) {
                            const int qw = (q_base + tid) * 4;
                            const int kw = (k_base + tid) * 4;
                            const int vw = (v_base + tid) * 4;
                            q_s[tid] = gqh_dwconv4_silu(
                                supersonic_qwen35_to_float(qx0),
                                supersonic_qwen35_to_float(qx1),
                                supersonic_qwen35_to_float(qx2),
                                supersonic_qwen35_to_float(qnew),
                                supersonic_qwen35_to_float(cw[qw + 0]),
                                supersonic_qwen35_to_float(cw[qw + 1]),
                                supersonic_qwen35_to_float(cw[qw + 2]),
                                supersonic_qwen35_to_float(cw[qw + 3]));
                            k_s[tid] = gqh_dwconv4_silu(
                                supersonic_qwen35_to_float(kx0),
                                supersonic_qwen35_to_float(kx1),
                                supersonic_qwen35_to_float(kx2),
                                supersonic_qwen35_to_float(knew),
                                supersonic_qwen35_to_float(cw[kw + 0]),
                                supersonic_qwen35_to_float(cw[kw + 1]),
                                supersonic_qwen35_to_float(cw[kw + 2]),
                                supersonic_qwen35_to_float(cw[kw + 3]));
                            v_conv = gqh_dwconv4_silu(
                                supersonic_qwen35_to_float(vx0),
                                supersonic_qwen35_to_float(vx1),
                                supersonic_qwen35_to_float(vx2),
                                supersonic_qwen35_to_float(vnew),
                                supersonic_qwen35_to_float(cw[vw + 0]),
                                supersonic_qwen35_to_float(cw[vw + 1]),
                                supersonic_qwen35_to_float(cw[vw + 2]),
                                supersonic_qwen35_to_float(cw[vw + 3]));
                        }
                        if (tid == 0 && h < nk) {
                            unsigned n_share = 0;
                            for (int hh = h_k; hh < nv; hh += nk) {
                                ++n_share;
                            }
                            gqh_kq_wait(
                                &kq_flag[h_k],
                                static_cast<unsigned>(seq_off) + 1u,
                                n_share);
                        }
                        __syncthreads();
                        if (h < nk && in_d) {
                            cs[(q_base + tid) * 3 + 0] = qx1;
                            cs[(q_base + tid) * 3 + 1] = qx2;
                            cs[(q_base + tid) * 3 + 2] = qnew;
                            cs[(k_base + tid) * 3 + 0] = kx1;
                            cs[(k_base + tid) * 3 + 1] = kx2;
                            cs[(k_base + tid) * 3 + 2] = knew;
                        }
                        if (in_d) {
                            cs[(v_base + tid) * 3 + 0] = vx1;
                            cs[(v_base + tid) * 3 + 1] = vx2;
                            cs[(v_base + tid) * 3 + 2] = vnew;
                        }
                        if (h < nk && tid == 0) {
                            atomicExch(&kq_flag[h_k], 0u);
                        }
                        if (in_d) {
                            if (h < nk) {
                                conv_out[q_base + tid] = q_s[tid];
                                conv_out[k_base + tid] = k_s[tid];
                            }
                            conv_out[v_base + tid] = v_conv;
                        }
                        __syncthreads();
                        // split_norm_transpose_qkv: inv = rsqrt(sumsq+eps),
                        // Q *= inv * (1/sqrt(hkd)), K *= inv. Pairwise tree
                        // of 128 terms matches the 256-wide reduction.
                        {
                            const float qv = (tid < hkd) ? q_s[tid] : 0.0f;
                            const float kv = (tid < hkd) ? k_s[tid] : 0.0f;
                            if (tid < hkd) {
                                k_s[tid] = qv * qv;
                            }
                            __syncthreads();
                            const float q_sum = gqh_tree_sum(k_s, hkd, tid);
                            const float q_inv = rsqrtf(q_sum + 1e-6f);
                            const float q_scale =
                                1.0f / sqrtf(static_cast<float>(hkd));
                            if (tid < hkd) {
                                q_s[tid] = (qv * q_inv) * q_scale;
                                k_s[tid] = kv * kv;
                            }
                            __syncthreads();
                            const float k_sum = gqh_tree_sum(k_s, hkd, tid);
                            const float k_inv = rsqrtf(k_sum + 1e-6f);
                            if (tid < hkd) {
                                k_s[tid] = kv * k_inv;
                            }
                            if (in_d && h < nk) {
                                conv_out[conv_dim + h_k * hkd + tid] = q_s[tid];
                                conv_out[conv_dim + key_dim + h_k * hkd + tid] =
                                    k_s[tid];
                            }
                        }
                    } else if (cache_kq && tid == 0) {
                        // Sequential per-head sum (same order as the old
                        // tid==h loop) so the inv stays bit-exact.
                        const float* qh = conv_out + q_key_offset + h_k * hkd;
                        float sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += qh[d] * qh[d];
                        float inv = rsqrtf(sq + 1e-6f);
                        const float q_scale =
                            1.0f / sqrtf(static_cast<float>(hkd));
                        for (int d = 0; d < hkd; ++d)
                            q_s[d] = (qh[d] * inv) * q_scale;

                        const float* kh = conv_out + k_key_offset + h_k * hkd;
                        sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += kh[d] * kh[d];
                        inv = rsqrtf(sq + 1e-6f);
                        for (int d = 0; d < hkd; ++d)
                            k_s[d] = kh[d] * inv;
                    } else if (!cache_kq && tid == 0) {
                        float* qh = conv_out + q_key_offset + h_k * hkd;
                        float sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += qh[d] * qh[d];
                        float inv = rsqrtf(sq + 1e-6f);
                        const float q_scale =
                            1.0f / sqrtf(static_cast<float>(hkd));
                        for (int d = 0; d < hkd; ++d)
                            qh[d] = (qh[d] * inv) * q_scale;
                        float* kh = conv_out + k_key_offset + h_k * hkd;
                        sq = 0.0f;
                        for (int d = 0; d < hkd; ++d)
                            sq += kh[d] * kh[d];
                        inv = rsqrtf(sq + 1e-6f);
                        for (int d = 0; d < hkd; ++d)
                            kh[d] *= inv;
                    }
                    __syncthreads();

                    const bool host_rec = (kSplit == 2) && ((io_flags & 128) != 0);
                    if (host_rec) {
                        continue;
                    }

                    // kSplit==2 always host-launches rec (flag 128). Keep
                    // st[256] out of this instantiation so mid has no scratch.
                    if constexpr (kSplit != 2) {
                    qwen35_split2_rec_st256<T>(
                        tid,
                        h,
                        hvd,
                        hkd,
                        h_k,
                        v,
                        key_dim,
                        conv_dim,
                        q_key_offset,
                        k_key_offset,
                        v_val_offset,
                        cache_kq,
                        head_local,
                        b_f32_,
                        a_f32_,
                        dt_bw,
                        ale_w,
                        sh,
                        conv_out,
                        k_s,
                        q_s,
                        v_conv,
                        attn_scratch_b_);
                    }
                    __syncthreads();
                    if (head_local) {
                        // rms_norm_gated_sfirst: inv = rsqrt(mean(x^2)+eps),
                        // out = bf16(x * inv * w * silu(z)). Tree sum matches
                        // the 256-wide reduction (upper half zero).
                        const float xv = (tid < hvd) ? k_s[tid] : 0.0f;
                        if (tid < hvd) {
                            k_s[tid] = xv * xv;
                        }
                        __syncthreads();
                        const float g_sum = gqh_tree_sum(k_s, hvd, tid);
                        if (tid == 0) {
                            q_s[0] = rsqrtf(
                                g_sum / static_cast<float>(hvd) +
                                L.linear_norm_eps);
                        }
                        __syncthreads();
                        if (tid < hvd) {
                            const float* nw =
                                static_cast<const float*>(L.linear_norm_w);
                            const float* z_f32_ = proj_b + L.qkv_out_dim;
                            const float inv = q_s[0];
                            const float w = (nw != nullptr)
                                ? bf16_round_rne_f32_finite(nw[tid])
                                : 1.0f;
                            const float zv = z_f32_[h * hvd + tid];
                            const float gate_silu =
                                zv * supersonic_qwen35_sigmoid_fast(zv);
                            attn_scratch_b_[h * hvd + tid] =
                                bf16_round_rne_f32_finite(
                                    xv * inv * w * gate_silu);
                        }
                    }
                }
            }
            // Gated RMS is already per-block on head h. attn_scratch[h]
            // was written by this block. Publish those stores before the
            // split-2 kernel exits — the out GEMV is a later launch and
            // gfx1100 does not always make unfenced block stores visible
            // across that boundary (Hello 8192 today/? split).
            __threadfence();

            const bool host_rec_done = (kSplit == 2) && ((io_flags & 128) != 0);
            if (!head_local && !host_rec_done) {
            {
                // Step D: Per-head RMSNorm + weight + SiLU(z) gating.
                // Sequential per-head sum stays on tid 0 (same order as
                // the old tid==h loop). Each block applies its own head.
                const int nv = L.linear_num_v_heads;
                const int hvd = L.linear_head_v_dim;
                const float* nw = static_cast<const float*>(L.linear_norm_w);
                float* z_f32_ = proj_b + L.qkv_out_dim;
                float* attn_scratch_bd = attn_scratch + b * attn_scratch_floats;
                for (int h = blockIdx.x; h < nv; h += nb) {
                    if (tid == 0) {
                        const float* ho = attn_scratch_bd + h * hvd;
                        float sq = 0.0f;
                        for (int j = 0; j < hvd; ++j)
                            sq += ho[j] * ho[j];
                        lds[0] = rsqrtf(
                            sq / static_cast<float>(hvd) + L.linear_norm_eps);
                    }
                    __syncthreads();
                    const float rms_inv = lds[0];
                    for (int j = tid; j < hvd; j += bs) {
                        const int d = h * hvd + j;
                        float n = attn_scratch_bd[d] * rms_inv;
                        float w = (nw != nullptr)
                            ? bf16_round_rne_f32_finite(nw[j])
                            : 1.0f;
                        float zv = z_f32_[d];
                        attn_scratch_bd[d] = bf16_round_rne_f32_finite(
                            n * w * (zv * supersonic_qwen35_sigmoid_fast(zv)));
                    }
                    __syncthreads();
                }
            }
            }
            // All mid blocks must finish attn_scratch before any block
            // exits. kSplit==2 used to skip this and trust the kernel
            // boundary; that dropped Hello token 3242 ("today") on gfx1100.
            grid_barrier(barrier_counter, barrier_flag, nb);
            qwen35_4b_hip_record_timing(
                layer_timing_slots,
                QWEN35_4B_TIMING_LINEAR_CORE_BASE + b,
                linear_core_clock);
            } // kSplit MID: linear core

            if constexpr (kSplit == 0 || kSplit == 3) {
            const unsigned long long linear_out_clock = clock64();
            // Step E: out_proj [hidden_dim, val_dim] × attn_scratch → hidden_f32 (fused residual)
            if constexpr (kSplit == 3) {
                if ((io_flags & 4) != 0) {
                    qwen35_4b_hip_record_timing(
                        layer_timing_slots,
                        QWEN35_4B_TIMING_LINEAR_OUT_BASE + b,
                        linear_out_clock);
                    continue;
                }
            }
            {
                const int vd = L.linear_value_dim;  // 2048
                const float* x_lin = attn_scratch + b * attn_scratch_floats;

                if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
                grid_barrier(barrier_counter, barrier_flag, nb);

                // INT4 on small models keeps block-level reduction (see
                // PR #20 for the 0.8B argmax sensitivity). Larger hidden_dim
                // (2B/4B/9B) falls through to wave-per-row.
                const int out_qtype =
                    int4_scales != nullptr ? int4_scales[layer].linear_out_proj_type : 0;
                const bool out_lowbit = qwen35_lowbit_supported(out_qtype);
                if (out_lowbit && hidden_dim <= 1024) {
                    for (;;) {
                        unsigned int sr;
                        if (tid == 0) sr = atomicAdd(&counters[0], 1u);
                        __shared__ unsigned int shared_sr;
                        if (tid == 0) shared_sr = sr;
                        __syncthreads();
                        sr = shared_sr;
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        const int gsz = int4_scales[layer].group_size;
                        const void* out_awq_inv = int4_scales[layer].linear_out_proj_awq_inv_scale;
                        if (qwen35_lowbit_native_int4(out_qtype)) {
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
                                apply_awq_inv_scale_8(w, out_awq_inv, c);
                                p += w[0]*x_lin[c] + w[1]*x_lin[c+1] + w[2]*x_lin[c+2] + w[3]*x_lin[c+3]
                                   + w[4]*x_lin[c+4] + w[5]*x_lin[c+5] + w[6]*x_lin[c+6] + w[7]*x_lin[c+7];
                            }
                            for (int c = vd8 + tid; c < vd; c += bs)
                                p += lowbit_dequant_scalar(L.linear_out_proj_w, int4_scales[layer].linear_out_proj_scale, int4_scales[layer].linear_out_proj_zero, out_qtype, sr, c, vd, gsz) *
                                     awq_inv_scale_at(out_awq_inv, c) * x_lin[c];
                        } else {
                            for (int c = tid; c < vd; c += bs)
                                p += lowbit_dequant_scalar(L.linear_out_proj_w, int4_scales[layer].linear_out_proj_scale, int4_scales[layer].linear_out_proj_zero, out_qtype, sr, c, vd, gsz) *
                                     x_lin[c];
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
                } else {
                    const int lane_o = tid % warpSize;
                    for (;;) {
                        unsigned int sr;
                        if (lane_o == 0) sr = atomicAdd(&counters[0], 1u);
                        sr = __shfl(sr, 0);
                        if (sr >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        if (out_lowbit) {
                            const int gsz = int4_scales[layer].group_size;
                            const void* out_awq_inv = int4_scales[layer].linear_out_proj_awq_inv_scale;
                            if (qwen35_lowbit_gqh(out_qtype)) {
                                qwen35_gqh_lane_dots(
                                    L.linear_out_proj_w, out_qtype, sr, vd,
                                    int4_scales[layer].linear_out_proj_scale,
                                    x_lin, vd, 1, &p, lane_o);
                            } else if (qwen35_lowbit_native_int4(out_qtype)) {
                                const int byte_cols = vd / 2;
                                const uint8_t* i4_row = static_cast<const uint8_t*>(L.linear_out_proj_w) + static_cast<size_t>(sr) * byte_cols;
                                const hip_bfloat16* sc = static_cast<const hip_bfloat16*>(int4_scales[layer].linear_out_proj_scale);
                                const hip_bfloat16* zr = static_cast<const hip_bfloat16*>(int4_scales[layer].linear_out_proj_zero);
                                const int sr_g = sr / gsz;
                                const int sc_cols = (vd + gsz - 1) / gsz;
                                const int vd8 = vd & ~7;
                                for (int c = lane_o * 8; c < vd8; c += warpSize * 8) {
                                    uint32_t pk = *reinterpret_cast<const uint32_t*>(&i4_row[c / 2]);
                                    float w[8];
                                    int4_dequant_8(pk, sc, zr, sr_g, c, sc_cols, gsz, w);
                                    apply_awq_inv_scale_8(w, out_awq_inv, c);
                                    p += w[0]*x_lin[c] + w[1]*x_lin[c+1] + w[2]*x_lin[c+2] + w[3]*x_lin[c+3]
                                       + w[4]*x_lin[c+4] + w[5]*x_lin[c+5] + w[6]*x_lin[c+6] + w[7]*x_lin[c+7];
                                }
                                for (int c = vd8 + lane_o; c < vd; c += warpSize)
                                    p += lowbit_dequant_scalar(L.linear_out_proj_w, int4_scales[layer].linear_out_proj_scale, int4_scales[layer].linear_out_proj_zero, out_qtype, sr, c, vd, gsz) *
                                         awq_inv_scale_at(out_awq_inv, c) * x_lin[c];
                            } else {
                                for (int c = lane_o; c < vd; c += warpSize)
                                    p += lowbit_dequant_scalar(L.linear_out_proj_w, int4_scales[layer].linear_out_proj_scale, int4_scales[layer].linear_out_proj_zero, out_qtype, sr, c, vd, gsz) *
                                         x_lin[c];
                            }
                        } else if (fp8_scales != nullptr && fp8_scales[layer].linear_out_proj_scale != nullptr) {
                            for (int c = lane_o; c < vd; c += warpSize)
                                p += fp8_dequant_weight_lut(L.linear_out_proj_w, fp8_scales[layer].linear_out_proj_scale, sr, c, vd, fp8_scales[layer].block_size, fp8_lut) * x_lin[c];
                        } else {
                            const T* wr = static_cast<const T*>(L.linear_out_proj_w) + static_cast<size_t>(sr) * vd;
                            if (kv_fp8 != nullptr) {
                                if (lane_o == 0) {
                                    for (int c = 0; c < vd; ++c) {
                                        p += supersonic_qwen35_to_float(wr[c]) * x_lin[c];
                                    }
                                }
                            } else {
                                const int vd4 = vd & ~3;
                                for (int c = lane_o * 4; c < vd4; c += warpSize * 4) {
                                    float w0 = supersonic_qwen35_to_float(wr[c]);
                                    float w1 = supersonic_qwen35_to_float(wr[c+1]);
                                    float w2 = supersonic_qwen35_to_float(wr[c+2]);
                                    float w3 = supersonic_qwen35_to_float(wr[c+3]);
                                    p += w0 * x_lin[c]   + w1 * x_lin[c+1]
                                       + w2 * x_lin[c+2] + w3 * x_lin[c+3];
                                }
                                for (int c = vd4 + lane_o; c < vd; c += warpSize)
                                    p += supersonic_qwen35_to_float(wr[c]) * x_lin[c];
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
            }
            __syncthreads();
            qwen35_4b_hip_record_timing(
                layer_timing_slots,
                QWEN35_4B_TIMING_LINEAR_OUT_BASE + b,
                linear_out_clock);
            grid_barrier(barrier_counter, barrier_flag, nb);
            } // kSplit OUT: linear out
            } // end for (b) batch loop for linear attention

        }

        if constexpr (kSplit == 0 || kSplit == 3) {
        if (blockIdx.x == 0) {
            for (int b = 0; b < B; b++) {
                for (int c = tid; c < hidden_dim; c += bs) {
                    const float checkpoint =
                        bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + c]);
                    token_out[b * hidden_dim + c] = checkpoint;
                    hidden_io[b * hidden_dim + c] = supersonic_qwen35_from_float<T>(checkpoint);
                    hidden_f32[b * hidden_dim + c] =
                        supersonic_qwen35_to_float(hidden_io[b * hidden_dim + c]);
                }
                __syncthreads();
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);
        } // kSplit 3: mixer residual checkpoint

        if constexpr (kSplit == 0 || kSplit == 4) {
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
                        const float norm_unit_offset = L.rms_norm_add_unit_offset ? 1.0f : 0.0f;
                        dst[c] = supersonic_qwen35_from_float<T>(
                            src[c] * inv_rms * (supersonic_qwen35_to_float(norm_w[c]) + norm_unit_offset));
                    }
                    __syncthreads();
                }
            }
        }
        grid_barrier(barrier_counter, barrier_flag, nb);
        if constexpr (kSplit == 0) {

        // === Fused MLP gate+up+SwiGLU (all blocks work-steal, BATCHED) ===
        // BF16/FP8 use wave-per-row with shuffle reduction; INT4 keeps
        // block-level reduction for bit-exact numerics (same reassociation
        // sensitivity pattern as linear_proj/out, see PR #20).
        const unsigned long long mlp_gate_up_clock = clock64();
        if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
        grid_barrier(barrier_counter, barrier_flag, nb);

        {
            const bool gate_fp8 = (fp8_scales != nullptr && fp8_scales[layer].gate_proj_scale != nullptr);
            const bool up_fp8 = (fp8_scales != nullptr && fp8_scales[layer].up_proj_scale != nullptr);
            const int gate_qtype =
                int4_scales != nullptr ? int4_scales[layer].gate_proj_type : 0;
            const int up_qtype =
                int4_scales != nullptr ? int4_scales[layer].up_proj_type : 0;
            const bool mlp_lowbit =
                qwen35_lowbit_supported(gate_qtype) || qwen35_lowbit_supported(up_qtype);

            if (mlp_lowbit && hidden_dim <= 1024) {
                __shared__ unsigned int shared_row_mlp;
                for (;;) {
                    if (tid == 0) shared_row_mlp = atomicAdd(&counters[0], 1u);
                    __syncthreads();
                    unsigned int my_row = shared_row_mlp;
                    if (my_row >= static_cast<unsigned int>(L.intermediate_size)) break;

                    float gp[MAX_BATCH_SIZE];
                    float up[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) { gp[b] = 0.0f; up[b] = 0.0f; }

                    const int gsz = int4_scales[layer].group_size;
                    const int byte_cols = hidden_dim / 2;
                    const uint8_t* g_i4 = static_cast<const uint8_t*>(L.gate_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                    const uint8_t* u_i4 = static_cast<const uint8_t*>(L.up_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                    const hip_bfloat16* g_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_scale);
                    const hip_bfloat16* g_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_zero);
                    const hip_bfloat16* u_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_scale);
                    const hip_bfloat16* u_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_zero);
                    const void* g_awq_inv = int4_scales[layer].gate_proj_awq_inv_scale;
                    const void* u_awq_inv = int4_scales[layer].up_proj_awq_inv_scale;
                    const int g_sr = my_row / gsz;
                    const int sc_cols = (hidden_dim + gsz - 1) / gsz;
                    const int vd8 = hidden_dim & ~7;
                    for (int c = tid * 8; c < vd8; c += bs * 8) {
                        float gw[8], uw[8];
                        lowbit_dequant_8(L.gate_proj_w, g_sc, g_zr, gate_qtype, my_row, c, hidden_dim, gsz, gw);
                        lowbit_dequant_8(L.up_proj_w, u_sc, u_zr, up_qtype, my_row, c, hidden_dim, gsz, uw);
                        apply_awq_inv_scale_8(gw, g_awq_inv, c);
                        apply_awq_inv_scale_8(uw, u_awq_inv, c);
                        for (int b = 0; b < B; b++) {
                            const float* inp = normed + b * hidden_dim + c;
                            gp[b] += gw[0]*inp[0] + gw[1]*inp[1] + gw[2]*inp[2] + gw[3]*inp[3]
                                   + gw[4]*inp[4] + gw[5]*inp[5] + gw[6]*inp[6] + gw[7]*inp[7];
                            up[b] += uw[0]*inp[0] + uw[1]*inp[1] + uw[2]*inp[2] + uw[3]*inp[3]
                                   + uw[4]*inp[4] + uw[5]*inp[5] + uw[6]*inp[6] + uw[7]*inp[7];
                        }
                    }
                    for (int c = vd8 + tid; c < hidden_dim; c += bs) {
                        float gw = lowbit_dequant_scalar(L.gate_proj_w, int4_scales[layer].gate_proj_scale, int4_scales[layer].gate_proj_zero, gate_qtype, my_row, c, hidden_dim, gsz) *
                                   awq_inv_scale_at(g_awq_inv, c);
                        float uw = lowbit_dequant_scalar(L.up_proj_w, int4_scales[layer].up_proj_scale, int4_scales[layer].up_proj_zero, up_qtype, my_row, c, hidden_dim, gsz) *
                                   awq_inv_scale_at(u_awq_inv, c);
                        for (int b = 0; b < B; b++) {
                            float inp = normed[b * hidden_dim + c];
                            gp[b] += gw * inp;
                            up[b] += uw * inp;
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
            } else {
                const int lane_p = tid % warpSize;
                for (;;) {
                    unsigned int my_row;
                    if (lane_p == 0) my_row = atomicAdd(&counters[0], 1u);
                    my_row = __shfl(my_row, 0);
                    if (my_row >= static_cast<unsigned int>(L.intermediate_size)) break;

                    float gp[MAX_BATCH_SIZE];
                    float up[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) { gp[b] = 0.0f; up[b] = 0.0f; }

                    if (mlp_lowbit && qwen35_lowbit_gqh(gate_qtype) && qwen35_lowbit_gqh(up_qtype)) {
                        qwen35_gqh_lane_dots(
                            L.gate_proj_w, gate_qtype, my_row, hidden_dim,
                            int4_scales[layer].gate_proj_scale,
                            normed, hidden_dim, B, gp, lane_p);
                        if constexpr (kSplit == 0) {
                            qwen35_gqh_lane_dots(
                                L.up_proj_w, up_qtype, my_row, hidden_dim,
                                int4_scales[layer].up_proj_scale,
                                normed, hidden_dim, B, up, lane_p);
                        }
                    } else if constexpr (kSplit == 0) {
                    if (mlp_lowbit) {
                        const int gsz = int4_scales[layer].group_size;
                        const int byte_cols = hidden_dim / 2;
                        const uint8_t* g_i4 = static_cast<const uint8_t*>(L.gate_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                        const uint8_t* u_i4 = static_cast<const uint8_t*>(L.up_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                        const hip_bfloat16* g_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_scale);
                        const hip_bfloat16* g_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].gate_proj_zero);
                        const hip_bfloat16* u_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_scale);
                        const hip_bfloat16* u_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].up_proj_zero);
                        const void* g_awq_inv = int4_scales[layer].gate_proj_awq_inv_scale;
                        const void* u_awq_inv = int4_scales[layer].up_proj_awq_inv_scale;
                        const int g_sr = my_row / gsz;
                        const int sc_cols = (hidden_dim + gsz - 1) / gsz;
                            const int vd8 = hidden_dim & ~7;
                            for (int c = lane_p * 8; c < vd8; c += warpSize * 8) {
                                float gw[8], uw[8];
                                lowbit_dequant_8(L.gate_proj_w, g_sc, g_zr, gate_qtype, my_row, c, hidden_dim, gsz, gw);
                                lowbit_dequant_8(L.up_proj_w, u_sc, u_zr, up_qtype, my_row, c, hidden_dim, gsz, uw);
                                apply_awq_inv_scale_8(gw, g_awq_inv, c);
                                apply_awq_inv_scale_8(uw, u_awq_inv, c);
                            for (int b = 0; b < B; b++) {
                                const float* inp = normed + b * hidden_dim + c;
                                gp[b] += gw[0]*inp[0] + gw[1]*inp[1] + gw[2]*inp[2] + gw[3]*inp[3]
                                       + gw[4]*inp[4] + gw[5]*inp[5] + gw[6]*inp[6] + gw[7]*inp[7];
                                up[b] += uw[0]*inp[0] + uw[1]*inp[1] + uw[2]*inp[2] + uw[3]*inp[3]
                                       + uw[4]*inp[4] + uw[5]*inp[5] + uw[6]*inp[6] + uw[7]*inp[7];
                            }
                        }
                        for (int c = vd8 + lane_p; c < hidden_dim; c += warpSize) {
                            float gw = lowbit_dequant_scalar(L.gate_proj_w, int4_scales[layer].gate_proj_scale, int4_scales[layer].gate_proj_zero, gate_qtype, my_row, c, hidden_dim, gsz) *
                                       awq_inv_scale_at(g_awq_inv, c);
                            float uw = lowbit_dequant_scalar(L.up_proj_w, int4_scales[layer].up_proj_scale, int4_scales[layer].up_proj_zero, up_qtype, my_row, c, hidden_dim, gsz) *
                                       awq_inv_scale_at(u_awq_inv, c);
                            for (int b = 0; b < B; b++) {
                                float inp = normed[b * hidden_dim + c];
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
                        for (int c = lane_p * 4; c < vd4; c += warpSize * 4) {
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
                                const float* inp = normed + b * hidden_dim + c;
                                gp[b] += gw0*inp[0] + gw1*inp[1] + gw2*inp[2] + gw3*inp[3];
                                up[b] += uw0*inp[0] + uw1*inp[1] + uw2*inp[2] + uw3*inp[3];
                            }
                        }
                        for (int c = vd4 + lane_p; c < hidden_dim; c += warpSize) {
                            float gw = fp8_dequant_weight_lut(L.gate_proj_w, fp8_scales[layer].gate_proj_scale, my_row, c, hidden_dim, bsz, fp8_lut);
                            float uw = fp8_dequant_weight_lut(L.up_proj_w, fp8_scales[layer].up_proj_scale, my_row, c, hidden_dim, bsz, fp8_lut);
                            for (int b = 0; b < B; b++) {
                                float inp = normed[b * hidden_dim + c];
                                gp[b] += gw * inp;
                                up[b] += uw * inp;
                            }
                        }
                    } else {
                        const T* gr = static_cast<const T*>(L.gate_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                        const T* ur = static_cast<const T*>(L.up_proj_w) + static_cast<size_t>(my_row) * hidden_dim;
                        const int vec_dim = hidden_dim & ~3;
                        for (int c = lane_p * 4; c < vec_dim; c += warpSize * 4) {
                            float gw0 = supersonic_qwen35_to_float(gr[c]);
                            float gw1 = supersonic_qwen35_to_float(gr[c+1]);
                            float gw2 = supersonic_qwen35_to_float(gr[c+2]);
                            float gw3 = supersonic_qwen35_to_float(gr[c+3]);
                            float uw0 = supersonic_qwen35_to_float(ur[c]);
                            float uw1 = supersonic_qwen35_to_float(ur[c+1]);
                            float uw2 = supersonic_qwen35_to_float(ur[c+2]);
                            float uw3 = supersonic_qwen35_to_float(ur[c+3]);
                            for (int b = 0; b < B; b++) {
                                const float* inp = normed + b * hidden_dim + c;
                                gp[b] += gw0 * inp[0] + gw1 * inp[1] + gw2 * inp[2] + gw3 * inp[3];
                                up[b] += uw0 * inp[0] + uw1 * inp[1] + uw2 * inp[2] + uw3 * inp[3];
                            }
                        }
                        for (int c = vec_dim + lane_p; c < hidden_dim; c += warpSize) {
                            float gw = supersonic_qwen35_to_float(gr[c]);
                            float uw = supersonic_qwen35_to_float(ur[c]);
                            for (int b = 0; b < B; b++) {
                                float inp = normed[b * hidden_dim + c];
                                gp[b] += gw * inp;
                                up[b] += uw * inp;
                            }
                        }
                    }
                    } // kSplit 0: non-GQH fused dequant

                    for (int b = 0; b < B; b++) {
                        float g = wave_reduce_sum_f32(gp[b]);
                        if constexpr (kSplit == 4) {
                            const bool gqh_gate_only = mlp_lowbit
                                && qwen35_lowbit_gqh(gate_qtype)
                                && qwen35_lowbit_gqh(up_qtype);
                            if (gqh_gate_only) {
                                if (lane_p == 0) {
                                    gate_up[b * intermediate_size * 2 + my_row] =
                                        bf16_round_rne_f32_finite(g);
                                }
                            } else {
                                float u = wave_reduce_sum_f32(up[b]);
                                if (lane_p == 0) {
                                    float gr = bf16_round_rne_f32_finite(g);
                                    float ur = bf16_round_rne_f32_finite(u);
                                    float silu = gr / (1.0f + expf(-gr));
                                    gate_up[b * intermediate_size * 2 + my_row] =
                                        bf16_round_rne_f32_finite(silu * ur);
                                }
                            }
                        } else {
                            float u = wave_reduce_sum_f32(up[b]);
                            if (lane_p == 0) {
                                float gr = bf16_round_rne_f32_finite(g);
                                float ur = bf16_round_rne_f32_finite(u);
                                float silu = gr / (1.0f + expf(-gr));
                                gate_up[b * intermediate_size * 2 + my_row] =
                                    bf16_round_rne_f32_finite(silu * ur);
                            }
                        }
                    }
                }
            }
        }
        // Waves finish independent rows at different times under wave-per-row;
        // sync before thread-0 records the timing so it reflects the slowest wave.
        __syncthreads();
        qwen35_4b_hip_record_timing(
            layer_timing_slots, QWEN35_4B_TIMING_MLP_GATE_UP, mlp_gate_up_clock);
        grid_barrier(barrier_counter, barrier_flag, nb);
        } // kSplit 0 fused gate+up
        } // kSplit 4: post-RMS (gate/up are dedicated GQH GEMVs on the split path)

        if constexpr (kSplit == 6) {
        // GQH up + SwiGLU. Gate activations were stored raw by kSplit 4.
        const unsigned long long mlp_up_clock = clock64();
        if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
        grid_barrier(barrier_counter, barrier_flag, nb);
        {
            const int up_qtype =
                int4_scales != nullptr ? int4_scales[layer].up_proj_type : 0;
            if (qwen35_lowbit_gqh(up_qtype)) {
                const int lane_p = tid % warpSize;
                for (;;) {
                    unsigned int my_row;
                    if (lane_p == 0) my_row = atomicAdd(&counters[0], 1u);
                    my_row = __shfl(my_row, 0);
                    if (my_row >= static_cast<unsigned int>(L.intermediate_size)) break;
                    float up[MAX_BATCH_SIZE];
                    for (int b = 0; b < B; b++) up[b] = 0.0f;
                    qwen35_gqh_lane_dots(
                        L.up_proj_w, up_qtype, my_row, hidden_dim,
                        int4_scales[layer].up_proj_scale,
                        normed, hidden_dim, B, up, lane_p);
                    for (int b = 0; b < B; b++) {
                        float u = wave_reduce_sum_f32(up[b]);
                        if (lane_p == 0) {
                            float gr = gate_up[b * intermediate_size * 2 + my_row];
                            float ur = bf16_round_rne_f32_finite(u);
                            float silu = gr / (1.0f + expf(-gr));
                            gate_up[b * intermediate_size * 2 + my_row] =
                                bf16_round_rne_f32_finite(silu * ur);
                        }
                    }
                }
            }
        }
        __syncthreads();
        qwen35_4b_hip_record_timing(
            layer_timing_slots, QWEN35_4B_TIMING_MLP_GATE_UP, mlp_up_clock);
        grid_barrier(barrier_counter, barrier_flag, nb);
        } // kSplit 6: up + SwiGLU

        if constexpr (kSplit == 5) {
        // Dedicated GEMV may have already fused the residual into hidden.
        const unsigned long long mlp_down_clock = clock64();
        if ((io_flags & 64) == 0) {
        for (int b = 0; b < B; b++) {
            const float* add = mlp_out + b * hidden_dim;
            float* hid = hidden_f32 + b * hidden_dim;
            for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
                hid[c] = bf16_round_rne_f32_finite(
                    hid[c] + bf16_round_rne_f32_finite(add[c]));
            }
        }
        __syncthreads();
        grid_barrier(barrier_counter, barrier_flag, nb);
        }
        qwen35_4b_hip_record_timing(
            layer_timing_slots, QWEN35_4B_TIMING_MLP_DOWN, mlp_down_clock);
        } else if constexpr (kSplit == 0) {
        // === MLP down_proj (all blocks work-steal, per-sequence) ===
        // Process one batch item at a time: cache SwiGLU output in LDS, work-steal
        const unsigned long long mlp_down_clock = clock64();
        for (int b = 0; b < B; b++) {
            const float* swiglu_b = gate_up + b * intermediate_size * 2;

            if (blockIdx.x == 0 && tid == 0) { counters[0] = 0; __threadfence(); }
            grid_barrier(barrier_counter, barrier_flag, nb);

            {
                const bool down_fp8 = (fp8_scales != nullptr && fp8_scales[layer].down_proj_scale != nullptr);
                const int down_qtype =
                    int4_scales != nullptr ? int4_scales[layer].down_proj_type : 0;
                const bool down_lowbit = qwen35_lowbit_supported(down_qtype);
                if (down_lowbit && hidden_dim <= 1024) {
                    __shared__ unsigned int shared_row_down;
                    for (;;) {
                        if (tid == 0) shared_row_down = atomicAdd(&counters[0], 1u);
                        __syncthreads();
                        unsigned int my_row = shared_row_down;
                        if (my_row >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        const int gsz = int4_scales[layer].group_size;
                        const void* d_awq_inv = int4_scales[layer].down_proj_awq_inv_scale;
                        const int byte_cols = L.intermediate_size / 2;
                        const uint8_t* i4_row = static_cast<const uint8_t*>(L.down_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                        const hip_bfloat16* d_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_scale);
                        const hip_bfloat16* d_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_zero);
                        const int sr_g = my_row / gsz;
                        const int sc_cols = (L.intermediate_size + gsz - 1) / gsz;
                        const int is8 = L.intermediate_size & ~7;
                        for (int c = tid * 8; c < is8; c += bs * 8) {
                            float w[8];
                            lowbit_dequant_8(L.down_proj_w, d_sc, d_zr, down_qtype, my_row, c, L.intermediate_size, gsz, w);
                            apply_awq_inv_scale_8(w, d_awq_inv, c);
                            p += w[0]*swiglu_b[c] + w[1]*swiglu_b[c+1] + w[2]*swiglu_b[c+2] + w[3]*swiglu_b[c+3]
                               + w[4]*swiglu_b[c+4] + w[5]*swiglu_b[c+5] + w[6]*swiglu_b[c+6] + w[7]*swiglu_b[c+7];
                        }
                        for (int c = is8 + tid; c < L.intermediate_size; c += bs)
                            p += lowbit_dequant_scalar(L.down_proj_w, int4_scales[layer].down_proj_scale, int4_scales[layer].down_proj_zero, down_qtype, my_row, c, L.intermediate_size, gsz) *
                                 awq_inv_scale_at(d_awq_inv, c) * swiglu_b[c];

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
                } else {
                    const int lane_d = tid % warpSize;
                    for (;;) {
                        unsigned int my_row;
                        const unsigned int steal =
                            (down_lowbit && qwen35_lowbit_gqh(down_qtype)) ? 2u : 1u;
                        if (lane_d == 0) my_row = atomicAdd(&counters[0], steal);
                        my_row = __shfl(my_row, 0);
                        if (my_row >= static_cast<unsigned int>(hidden_dim)) break;

                        float p = 0.0f;
                        if (down_lowbit && qwen35_lowbit_gqh(down_qtype)) {
                            float p1 = 0.0f;
                            float ts;
                            int gc;
                            qwen35_gqh_hdr(
                                int4_scales[layer].down_proj_scale, &ts, &gc);
                            const int row1 =
                                (my_row + 1 < static_cast<unsigned int>(hidden_dim))
                                    ? static_cast<int>(my_row + 1)
                                    : -1;
                            qwen35_gqh_lane_dot_pair(
                                L.down_proj_w, down_qtype, static_cast<int>(my_row),
                                row1, L.intermediate_size, ts, gc, swiglu_b,
                                L.intermediate_size / GQH_SUPERBLOCK, lane_d, &p, &p1);
                            p = wave_reduce_sum_f32(p);
                            if (lane_d == 0) {
                                float add = bf16_round_rne_f32_finite(p);
                                mlp_out[b * hidden_dim + my_row] = add;
                                hidden_f32[b * hidden_dim + my_row] =
                                    bf16_round_rne_f32_finite(
                                        hidden_f32[b * hidden_dim + my_row] + add);
                            }
                            if (row1 >= 0) {
                                p1 = wave_reduce_sum_f32(p1);
                                if (lane_d == 0) {
                                    float add = bf16_round_rne_f32_finite(p1);
                                    mlp_out[b * hidden_dim + row1] = add;
                                    hidden_f32[b * hidden_dim + row1] =
                                        bf16_round_rne_f32_finite(
                                            hidden_f32[b * hidden_dim + row1] + add);
                                }
                            }
                            continue;
                        } else if (down_lowbit) {
                            const int gsz = int4_scales[layer].group_size;
                            const void* d_awq_inv = int4_scales[layer].down_proj_awq_inv_scale;
                            const int byte_cols = L.intermediate_size / 2;
                            const uint8_t* i4_row = static_cast<const uint8_t*>(L.down_proj_w) + static_cast<size_t>(my_row) * byte_cols;
                            const hip_bfloat16* d_sc = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_scale);
                            const hip_bfloat16* d_zr = static_cast<const hip_bfloat16*>(int4_scales[layer].down_proj_zero);
                            const int sr_g = my_row / gsz;
                            const int sc_cols = (L.intermediate_size + gsz - 1) / gsz;
                            const int is8 = L.intermediate_size & ~7;
                            for (int c = lane_d * 8; c < is8; c += warpSize * 8) {
                                float w[8];
                                lowbit_dequant_8(L.down_proj_w, d_sc, d_zr, down_qtype, my_row, c, L.intermediate_size, gsz, w);
                                apply_awq_inv_scale_8(w, d_awq_inv, c);
                                p += w[0]*swiglu_b[c] + w[1]*swiglu_b[c+1] + w[2]*swiglu_b[c+2] + w[3]*swiglu_b[c+3]
                                   + w[4]*swiglu_b[c+4] + w[5]*swiglu_b[c+5] + w[6]*swiglu_b[c+6] + w[7]*swiglu_b[c+7];
                            }
                            for (int c = is8 + lane_d; c < L.intermediate_size; c += warpSize)
                                p += lowbit_dequant_scalar(L.down_proj_w, int4_scales[layer].down_proj_scale, int4_scales[layer].down_proj_zero, down_qtype, my_row, c, L.intermediate_size, gsz) *
                                     awq_inv_scale_at(d_awq_inv, c) * swiglu_b[c];
                        } else if (down_fp8) {
                            const uint8_t* d_fp8 = static_cast<const uint8_t*>(L.down_proj_w) + static_cast<size_t>(my_row) * L.intermediate_size;
                            const hip_bfloat16* d_scales = static_cast<const hip_bfloat16*>(fp8_scales[layer].down_proj_scale);
                            const int bsz = fp8_scales[layer].block_size;
                            const int d_scale_row = my_row / bsz;
                            const int d_scale_cols = (L.intermediate_size + bsz - 1) / bsz;
                            const int is4 = L.intermediate_size & ~3;
                            for (int c = lane_d * 4; c < is4; c += warpSize * 4) {
                                uint32_t pk = *reinterpret_cast<const uint32_t*>(&d_fp8[c]);
                                const int sb = d_scale_row * d_scale_cols;
                                float w0 = bf16_round_rne_f32_finite((fp8_lut[pk & 0xFF] * static_cast<float>(d_scales[sb + c / bsz])));
                                float w1 = bf16_round_rne_f32_finite((fp8_lut[(pk>>8) & 0xFF] * static_cast<float>(d_scales[sb + (c+1) / bsz])));
                                float w2 = bf16_round_rne_f32_finite((fp8_lut[(pk>>16) & 0xFF] * static_cast<float>(d_scales[sb + (c+2) / bsz])));
                                float w3 = bf16_round_rne_f32_finite((fp8_lut[(pk>>24)] * static_cast<float>(d_scales[sb + (c+3) / bsz])));
                                p += w0*swiglu_b[c] + w1*swiglu_b[c+1] + w2*swiglu_b[c+2] + w3*swiglu_b[c+3];
                            }
                            for (int c = is4 + lane_d; c < L.intermediate_size; c += warpSize)
                                p += fp8_dequant_weight_lut(L.down_proj_w, fp8_scales[layer].down_proj_scale, my_row, c, L.intermediate_size, bsz, fp8_lut) * swiglu_b[c];
                        } else {
                            const T* wr = static_cast<const T*>(L.down_proj_w) + static_cast<size_t>(my_row) * L.intermediate_size;
                            const int is4 = L.intermediate_size & ~3;
                            for (int c = lane_d * 4; c < is4; c += warpSize * 4) {
                                float w0 = supersonic_qwen35_to_float(wr[c]);
                                float w1 = supersonic_qwen35_to_float(wr[c+1]);
                                float w2 = supersonic_qwen35_to_float(wr[c+2]);
                                float w3 = supersonic_qwen35_to_float(wr[c+3]);
                                p += w0*swiglu_b[c]   + w1*swiglu_b[c+1]
                                   + w2*swiglu_b[c+2] + w3*swiglu_b[c+3];
                            }
                            for (int c = is4 + lane_d; c < L.intermediate_size; c += warpSize)
                                p += supersonic_qwen35_to_float(wr[c]) * swiglu_b[c];
                        }
                        float result = wave_reduce_sum_f32(p);
                        if (lane_d == 0) {
                            float add = bf16_round_rne_f32_finite(result);
                            mlp_out[b * hidden_dim + my_row] = add;
                            hidden_f32[b * hidden_dim + my_row] =
                                bf16_round_rne_f32_finite(hidden_f32[b * hidden_dim + my_row] + add);
                        }
                    }
                }
            }
            // Need syncthreads before grid barrier since waves may finish at different times
            __syncthreads();
            grid_barrier(barrier_counter, barrier_flag, nb);
        } // end for (b) down_proj batch loop
        qwen35_4b_hip_record_timing(
            layer_timing_slots, QWEN35_4B_TIMING_MLP_DOWN, mlp_down_clock);

        // Residual add for MLP is fused into down_proj above.

        } // kSplit 0 fused down
    }

    // Write back F32 → BF16 for all batch items
    if constexpr (kSplit == 0 || kSplit == 5) {
        if (kSplit == 0 || (io_flags & 2) != 0) {
            for (int b = 0; b < B; b++) {
                for (int c = tid + blockIdx.x * bs; c < hidden_dim; c += bs * nb) {
                    hidden_io[b * hidden_dim + c] =
                        supersonic_qwen35_from_float<T>(hidden_f32[b * hidden_dim + c]);
                }
            }
        }
    }
}
