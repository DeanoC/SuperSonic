#ifndef PREFILL_COMMON_METAL
#define PREFILL_COMMON_METAL
#include <metal_stdlib>
using namespace metal;

// dtype: 0=F16, 1=F32, 2=BF16 (matches ScalarType::kernel_dtype_code)
inline float load_elem(device const uchar* base, uint index, int dtype) {
    if (dtype == 2) {
        return float(as_type<bfloat>(reinterpret_cast<device const bfloat*>(base)[index]));
    }
    if (dtype == 0) {
        return float(as_type<half>(reinterpret_cast<device const half*>(base)[index]));
    }
    return reinterpret_cast<device const float*>(base)[index];
}

inline void store_elem(device uchar* base, uint index, int dtype, float value) {
    if (dtype == 2) {
        reinterpret_cast<device bfloat*>(base)[index] = bfloat(value);
        return;
    }
    if (dtype == 0) {
        reinterpret_cast<device half*>(base)[index] = half(value);
        return;
    }
    reinterpret_cast<device float*>(base)[index] = value;
}

inline float exp_fast(float x) { return fast::exp(x); }

inline float sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp_fast(x);
    return e / (1.0f + e);
}

inline float bf16_round_rne(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu) != 0u) {
        bits = (bits & 0xFFFF0000u) | 0x00400000u;
    } else {
        uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
        bits += rounding_bias;
        bits &= 0xFFFF0000u;
    }
    return as_type<float>(bits);
}

inline float bf16_round_rne_finite(float x) {
    uint bits = as_type<uint>(x);
    uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    return as_type<float>(bits);
}

inline float wave_sum(float value) { return simd_sum(value); }

constant constexpr uint kTileM = 16;
constant constexpr uint kTileN = 16;
constant constexpr uint kTileK = 32;

constant constexpr int NATIVE_INT4 = 4;
constant constexpr int GGML_Q8_0 = 8;
constant constexpr int GGML_Q4_K = 12;
constant constexpr int GGML_Q5_K = 13;
constant constexpr int GGML_Q6_K = 14;
constant constexpr int GQH3 = 108;
constant constexpr int GQH2_H = 109;
constant constexpr int GQH2_C = 110;
constant constexpr int GQH4 = 111;

constant constexpr int GQH_SUPERBLOCK = 256;
constant constexpr int GQH4_SB_BYTES = 137;

inline bool qwen35_lowbit_native_int4(int qtype) { return qtype == NATIVE_INT4; }

inline float fp8_e4m3_to_float(uint8_t byte) {
    int sign = (byte >> 7) & 1;
    int exp = (byte >> 3) & 0xF;
    int mantissa = byte & 0x7;
    if (byte == 0x7F || byte == 0xFF) return 0.0f;
    float val;
    if (exp == 0) {
        val = float(mantissa) / 8.0f * 1.52587890625e-2f;
    } else {
        val = (1.0f + float(mantissa) / 8.0f) * exp2(float(exp - 7));
    }
    return sign ? -val : val;
}

inline float dequant_weight(device const uchar* w_ptr, device const uchar* scale_ptr,
                            int row, int col, int cols, int block_size, int scale_dtype) {
    device const uint8_t* fp8 = reinterpret_cast<device const uint8_t*>(w_ptr);
    float val = fp8_e4m3_to_float(fp8[uint(row) * uint(cols) + uint(col)]);
    int scale_row = row / block_size;
    int scale_col = col / block_size;
    int scale_cols = (cols + block_size - 1) / block_size;
    uint scale_index = uint(scale_row * scale_cols + scale_col);
    float scale = load_elem(scale_ptr, scale_index, scale_dtype);
    return bf16_round_rne_finite(val * scale);
}


constant uint kGqhE4M3[32][8] = {{ 0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u, 0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u },  
    { 0x3c800000u, 0x3c900000u, 0x3ca00000u, 0x3cb00000u, 0x3cc00000u, 0x3cd00000u, 0x3ce00000u, 0x3cf00000u },  
    { 0x3d000000u, 0x3d100000u, 0x3d200000u, 0x3d300000u, 0x3d400000u, 0x3d500000u, 0x3d600000u, 0x3d700000u },  
    { 0x3d800000u, 0x3d900000u, 0x3da00000u, 0x3db00000u, 0x3dc00000u, 0x3dd00000u, 0x3de00000u, 0x3df00000u },  
    { 0x3e000000u, 0x3e100000u, 0x3e200000u, 0x3e300000u, 0x3e400000u, 0x3e500000u, 0x3e600000u, 0x3e700000u },  
    { 0x3e800000u, 0x3e900000u, 0x3ea00000u, 0x3eb00000u, 0x3ec00000u, 0x3ed00000u, 0x3ee00000u, 0x3ef00000u },  
    { 0x3f000000u, 0x3f100000u, 0x3f200000u, 0x3f300000u, 0x3f400000u, 0x3f500000u, 0x3f600000u, 0x3f700000u },  
    { 0x3f800000u, 0x3f900000u, 0x3fa00000u, 0x3fb00000u, 0x3fc00000u, 0x3fd00000u, 0x3fe00000u, 0x3ff00000u },  
    { 0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u },  
    { 0x40800000u, 0x40900000u, 0x40a00000u, 0x40b00000u, 0x40c00000u, 0x40d00000u, 0x40e00000u, 0x40f00000u },  
    { 0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u },  
    { 0x41800000u, 0x41900000u, 0x41a00000u, 0x41b00000u, 0x41c00000u, 0x41d00000u, 0x41e00000u, 0x41f00000u },  
    { 0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u },  
    { 0x42800000u, 0x42900000u, 0x42a00000u, 0x42b00000u, 0x42c00000u, 0x42d00000u, 0x42e00000u, 0x42f00000u },  
    { 0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u },  
    { 0x43800000u, 0x43900000u, 0x43a00000u, 0x43b00000u, 0x43c00000u, 0x43d00000u, 0x43e00000u, 0x7ff00000u },  
    { 0x80000000u, 0xbb000000u, 0xbb800000u, 0xbbc00000u, 0xbc000000u, 0xbc200000u, 0xbc400000u, 0xbc600000u },  
    { 0xbc800000u, 0xbc900000u, 0xbca00000u, 0xbcb00000u, 0xbcc00000u, 0xbcd00000u, 0xbce00000u, 0xbcf00000u },  
    { 0xbd000000u, 0xbd100000u, 0xbd200000u, 0xbd300000u, 0xbd400000u, 0xbd500000u, 0xbd600000u, 0xbd700000u },  
    { 0xbd800000u, 0xbd900000u, 0xbda00000u, 0xbdb00000u, 0xbdc00000u, 0xbdd00000u, 0xbde00000u, 0xbdf00000u },  
    { 0xbe000000u, 0xbe100000u, 0xbe200000u, 0xbe300000u, 0xbe400000u, 0xbe500000u, 0xbe600000u, 0xbe700000u },  
    { 0xbe800000u, 0xbe900000u, 0xbea00000u, 0xbeb00000u, 0xbec00000u, 0xbed00000u, 0xbee00000u, 0xbef00000u },  
    { 0xbf000000u, 0xbf100000u, 0xbf200000u, 0xbf300000u, 0xbf400000u, 0xbf500000u, 0xbf600000u, 0xbf700000u },  
    { 0xbf800000u, 0xbf900000u, 0xbfa00000u, 0xbfb00000u, 0xbfc00000u, 0xbfd00000u, 0xbfe00000u, 0xbff00000u },  
    { 0xc0000000u, 0xc0100000u, 0xc0200000u, 0xc0300000u, 0xc0400000u, 0xc0500000u, 0xc0600000u, 0xc0700000u },  
    { 0xc0800000u, 0xc0900000u, 0xc0a00000u, 0xc0b00000u, 0xc0c00000u, 0xc0d00000u, 0xc0e00000u, 0xc0f00000u },  
    { 0xc1000000u, 0xc1100000u, 0xc1200000u, 0xc1300000u, 0xc1400000u, 0xc1500000u, 0xc1600000u, 0xc1700000u },  
    { 0xc1800000u, 0xc1900000u, 0xc1a00000u, 0xc1b00000u, 0xc1c00000u, 0xc1d00000u, 0xc1e00000u, 0xc1f00000u },  
    { 0xc2000000u, 0xc2100000u, 0xc2200000u, 0xc2300000u, 0xc2400000u, 0xc2500000u, 0xc2600000u, 0xc2700000u },  
    { 0xc2800000u, 0xc2900000u, 0xc2a00000u, 0xc2b00000u, 0xc2c00000u, 0xc2d00000u, 0xc2e00000u, 0xc2f00000u },  
    { 0xc3000000u, 0xc3100000u, 0xc3200000u, 0xc3300000u, 0xc3400000u, 0xc3500000u, 0xc3600000u, 0xc3700000u },  
    { 0xc3800000u, 0xc3900000u, 0xc3a00000u, 0xc3b00000u, 0xc3c00000u, 0xc3d00000u, 0xc3e00000u, 0xfff00000u },};
constant uint kGqhRatioQ[16][1] = {{ 0x00000000u },  
    { 0x3d888889u },  
    { 0x3e088889u },  
    { 0x3e4ccccdu },  
    { 0x3e888889u },  
    { 0x3eaaaaabu },  
    { 0x3ecccccdu },  
    { 0x3eeeeeefu },  
    { 0x3f088889u },  
    { 0x3f19999au },  
    { 0x3f2aaaabu },  
    { 0x3f3bbbbcu },  
    { 0x3f4ccccdu },  
    { 0x3f5ddddeu },  
    { 0x3f6eeeefu },  
    { 0x3f800000u },};
constant uint kGqh3Grid[12][8] = {{ 0xbf800000u, 0xbf400000u, 0xbf000000u, 0xbe800000u, 0x3e800000u, 0x3f000000u, 0x3f400000u, 0x3f800000u },  
    { 0xbf800000u, 0xbf363725u, 0xbee1aff8u, 0xbe46f6cau, 0x3e46f6cau, 0x3ee1aff8u, 0x3f363725u, 0x3f800000u },  
    { 0xbf800000u, 0xbf2cedf1u, 0xbec6f6c9u, 0xbe1aa2acu, 0x3e1aa2acu, 0x3ec6f6c9u, 0x3f2cedf1u, 0x3f800000u },  
    { 0xbf800000u, 0xbf241de1u, 0xbeaf67aau, 0xbdf05dc8u, 0x3df05dc8u, 0x3eaf67aau, 0x3f241de1u, 0x3f800000u },  
    { 0xbf800000u, 0xbf1bc0cbu, 0xbe9aa2acu, 0xbdbad03eu, 0x3dbad03eu, 0x3e9aa2acu, 0x3f1bc0cbu, 0x3f800000u },  
    { 0xbf800000u, 0xbf13d0d1u, 0xbe885344u, 0xbd913128u, 0x3d913128u, 0x3e885344u, 0x3f13d0d1u, 0x3f800000u },  
    { 0xbf800000u, 0xbf0c4866u, 0xbe705dc8u, 0xbd61aff9u, 0x3d61aff9u, 0x3e705dc8u, 0x3f0c4866u, 0x3f800000u },  
    { 0xbf800000u, 0xbf052240u, 0xbe53e7a4u, 0xbd2f67aau, 0x3d2f67aau, 0x3e53e7a4u, 0x3f052240u, 0x3f800000u },  
    { 0xbf800000u, 0xbefcb2bdu, 0xbe3ad03eu, 0xbd085343u, 0x3d085343u, 0x3e3ad03eu, 0x3efcb2bdu, 0x3f800000u },  
    { 0xbf800000u, 0xbeefd202u, 0xbe24b16eu, 0xbcd3e7a1u, 0x3cd3e7a1u, 0x3e24b16eu, 0x3eefd202u, 0x3f800000u },  
    { 0xbf800000u, 0xbee39948u, 0xbe113128u, 0xbca4b170u, 0x3ca4b170u, 0x3e113128u, 0x3ee39948u, 0x3f800000u },  
    { 0xbf800000u, 0xbed80000u, 0xbe000000u, 0xbc800000u, 0x3c800000u, 0x3e000000u, 0x3ed80000u, 0x3f800000u },};
constant uint kGqh2hGrid[12][4] = {{ 0xbf800000u, 0xbd4ccccdu, 0x3d4ccccdu, 0x3f800000u },  
    { 0xbf800000u, 0xbd805a78u, 0x3d805a78u, 0x3f800000u },  
    { 0xbf800000u, 0xbda0e27bu, 0x3da0e27bu, 0x3f800000u },  
    { 0xbf800000u, 0xbdc9a93du, 0x3dc9a93du, 0x3f800000u },  
    { 0xbf800000u, 0xbdfcc5b6u, 0x3dfcc5b6u, 0x3f800000u },  
    { 0xbf800000u, 0xbe1e6b3bu, 0x3e1e6b3bu, 0x3f800000u },  
    { 0xbf800000u, 0xbe4691ffu, 0x3e4691ffu, 0x3f800000u },  
    { 0xbf800000u, 0xbe78e5edu, 0x3e78e5edu, 0x3f800000u },  
    { 0xbf800000u, 0xbe9bfda7u, 0x3e9bfda7u, 0x3f800000u },  
    { 0xbf800000u, 0xbec386e1u, 0x3ec386e1u, 0x3f800000u },  
    { 0xbf800000u, 0xbef51557u, 0x3ef51557u, 0x3f800000u },  
    { 0xbf800000u, 0xbf19999au, 0x3f19999au, 0x3f800000u },};
constant uint kGqh4Grid[12][16] = {{ 0xbf800000u, 0xbf600000u, 0xbf400000u, 0xbf200000u, 0xbf000000u, 0xbec00000u, 0xbe800000u, 0xbe000000u, 0x3e000000u, 0x3e800000u, 0x3ec00000u, 0x3f000000u, 0x3f200000u, 0x3f400000u, 0x3f600000u, 0x3f800000u },  
    { 0xbf800000u, 0xbf5aa08bu, 0xbf363725u, 0xbf12e524u, 0xbee1aff8u, 0xbea0a3b7u, 0xbe46f6cau, 0xbdaf67abu, 0x3daf67abu, 0x3e46f6cau, 0x3ea0a3b7u, 0x3ee1aff8u, 0x3f12e524u, 0x3f363725u, 0x3f5aa08bu, 0x3f800000u },  
    { 0xbf800000u, 0xbf556213u, 0xbf2cedf1u, 0xbf06dd10u, 0xbec6f6c9u, 0xbe8666bcu, 0xbe1aa2acu, 0xbd705dc6u, 0x3d705dc6u, 0x3e1aa2acu, 0x3e8666bcu, 0x3ec6f6c9u, 0x3f06dd10u, 0x3f2cedf1u, 0x3f556213u, 0x3f800000u },  
    { 0xbf800000u, 0xbf5043cfu, 0xbf241de1u, 0xbef7a287u, 0xbeaf67aau, 0xbe60e5c4u, 0xbdf05dc8u, 0xbd24b170u, 0x3d24b170u, 0x3df05dc8u, 0x3e60e5c4u, 0x3eaf67aau, 0x3ef7a287u, 0x3f241de1u, 0x3f5043cfu, 0x3f800000u },  
    { 0xbf800000u, 0xbf4b44f9u, 0xbf1bc0cbu, 0xbee35a27u, 0xbe9aa2acu, 0xbe3c29e6u, 0xbdbad03eu, 0xbce1aff6u, 0x3ce1aff6u, 0x3dbad03eu, 0x3e3c29e6u, 0x3e9aa2acu, 0x3ee35a27u, 0x3f1bc0cbu, 0x3f4b44f9u, 0x3f800000u },  
    { 0xbf800000u, 0xbf4664cfu, 0xbf13d0d1u, 0xbed0bb10u, 0xbe885344u, 0xbe1d6e07u, 0xbd913128u, 0xbc9aa2adu, 0x3c9aa2adu, 0x3d913128u, 0x3e1d6e07u, 0x3e885344u, 0x3ed0bb10u, 0x3f13d0d1u, 0x3f4664cfu, 0x3f800000u },  
    { 0xbf800000u, 0xbf41a296u, 0xbf0c4866u, 0xbebfa26eu, 0xbe705dc8u, 0xbe03b743u, 0xbd61aff9u, 0xbc53e7a6u, 0x3c53e7a6u, 0x3d61aff9u, 0x3e03b743u, 0x3e705dc8u, 0x3ebfa26eu, 0x3f0c4866u, 0x3f41a296u, 0x3f800000u },  
    { 0xbf800000u, 0xbf3cfd94u, 0xbf052240u, 0xbeaff042u, 0xbe53e7a4u, 0xbddc6764u, 0xbd2f67aau, 0xbc113128u, 0x3c113128u, 0x3d2f67aau, 0x3ddc6764u, 0x3e53e7a4u, 0x3eaff042u, 0x3f052240u, 0x3f3cfd94u, 0x3f800000u },  
    { 0xbf800000u, 0xbf387518u, 0xbefcb2bdu, 0xbea18733u, 0xbe3ad03eu, 0xbdb8676cu, 0xbd085343u, 0xbbc6f6c8u, 0x3bc6f6c8u, 0x3d085343u, 0x3db8676cu, 0x3e3ad03eu, 0x3ea18733u, 0x3efcb2bdu, 0x3f387518u, 0x3f800000u },  
    { 0xbf800000u, 0xbf340872u, 0xbeefd202u, 0xbe944c4du, 0xbe24b16eu, 0xbd9a48c0u, 0xbcd3e7a1u, 0xbb885342u, 0x3b885342u, 0x3cd3e7a1u, 0x3d9a48c0u, 0x3e24b16eu, 0x3e944c4du, 0x3eefd202u, 0x3f340872u, 0x3f800000u },  
    { 0xbf800000u, 0xbf2fb6f7u, 0xbee39948u, 0xbe8826cfu, 0xbe113128u, 0xbd811586u, 0xbca4b170u, 0xbb3ad040u, 0x3b3ad040u, 0x3ca4b170u, 0x3d811586u, 0x3e113128u, 0x3e8826cfu, 0x3ee39948u, 0x3f2fb6f7u, 0x3f800000u },  
    { 0xbf800000u, 0xbf2b8000u, 0xbed80000u, 0xbe7a0000u, 0xbe000000u, 0xbd580000u, 0xbc800000u, 0xbb000000u, 0x3b000000u, 0x3c800000u, 0x3d580000u, 0x3e000000u, 0x3e7a0000u, 0x3ed80000u, 0x3f2b8000u, 0x3f800000u },};

inline float gqh_bits(uint u) { return as_type<float>(u); }

inline void gqh_plane_offsets(int nsb, int is3, thread int& off_r, thread int& off_lo, thread int& off_hi, thread int& stride) {
    int o = (nsb + 7) & ~7;
    off_r = o;
    o += nsb << 3;
    o = (o + 63) & ~63;
    off_lo = o;
    o += nsb << 6;
    off_hi = o;
    if (is3) o += nsb << 5;
    stride = (o + 63) & ~63;
}

inline float gqh_scale_planes(uint8_t d, device const uint8_t* ratio8, int sub, float tensor_scale) {
    const float d_real = gqh_bits(kGqhE4M3[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = ratio8[sub >> 1];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(kGqhRatioQ[ratio][0]);
}

inline float gqh_subblock_scale(device const uint8_t* b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_bits(kGqhE4M3[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(kGqhRatioQ[ratio][0]);
}

inline float qwen35_gqh_dequant_scalar(device const uchar* w_ptr, int qtype, int row, int col, int cols,
                                       float tensor_scale, int grid_code) {
    const int nsb = cols / GQH_SUPERBLOCK;
    const int is3 = qtype == GQH3;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, off_r, off_lo, off_hi, stride);
    device const uint8_t* rowbase = reinterpret_cast<device const uint8_t*>(w_ptr) + uint(row) * uint(stride);
    const int sb = col / GQH_SUPERBLOCK;
    const int j = col % GQH_SUPERBLOCK;
    const uint8_t d = rowbase[sb];
    device const uint8_t* rat = rowbase + off_r + (sb << 3);
    device const uint8_t* lo_p = rowbase + off_lo + (sb << 6);
    const float s_b = gqh_scale_planes(d, rat, j >> 4, tensor_scale);
    if (is3) {
        device const uint8_t* hi_p = rowbase + off_hi + (sb << 5);
        const int lo = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
        const int hi = (hi_p[j >> 3] >> (j & 7)) & 0x01;
        return gqh_bits(kGqh3Grid[grid_code][lo | (hi << 2)]) * s_b;
    }
    if (qtype == GQH2_H) {
        const int code = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
        return gqh_bits(kGqh2hGrid[grid_code][code]) * s_b;
    }
    if (qtype == GQH4) {
        device const uint8_t* b = reinterpret_cast<device const uint8_t*>(w_ptr) +
            uint(row) * uint(nsb * GQH4_SB_BYTES) + uint(sb) * GQH4_SB_BYTES;
        const float s4 = gqh_subblock_scale(b, j >> 4, tensor_scale);
        const uint8_t cb = b[9 + (j >> 1)];
        const int code = (j & 1) ? (cb >> 4) : (cb & 0x0f);
        return gqh_bits(kGqh4Grid[grid_code][code]) * s4;
    }
    return 0.0f;
}

inline int ggml_k_row_bytes(int qtype, int cols) {
    if (qtype == GGML_Q8_0) return (cols / 32) * 34;
    const int blocks = cols / 256;
    if (qtype == GGML_Q4_K) return blocks * 144;
    if (qtype == GGML_Q5_K) return blocks * 176;
    if (qtype == GGML_Q6_K) return blocks * 210;
    return cols / 2;
}

inline float ggml_f16_to_f32_unaligned(device const uint8_t* p) {
    uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return float(as_type<half>(bits));
}

inline void ggml_q4_k_scale_min(device const uint8_t* q, int j, thread int& scale, thread int& minv) {
    if (j < 4) {
        scale = int(q[j] & 63);
        minv = int(q[j + 4] & 63);
    } else {
        scale = int((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        minv = int((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

inline float ggml_k_dequant_scalar_from_row(device const uint8_t* row_data, int qtype, int col, int cols) {
    if (qtype == GGML_Q8_0) {
        const int block = col / 32;
        const int inb = col - block * 32;
        device const uint8_t* b = row_data + block * 34;
        const float d = ggml_f16_to_f32_unaligned(b);
        device const int8_t* qs = reinterpret_cast<device const int8_t*>(b + 2);
        return d * float(qs[inb]);
    }
    const int block = col / 256;
    const int inb = col - block * 256;
    device const uint8_t* b = row_data;
    if (qtype == GGML_Q4_K) {
        b += block * 144;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        device const uint8_t* sc = b + 4;
        device const uint8_t* qs = b + 16;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int j = 2 * g + sub;
        int scale = 0, minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const uint8_t qbyte = qs[g * 32 + (inb % 32)];
        const int q = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        return d * float(scale) * float(q) - dmin * float(minv);
    }
    if (qtype == GGML_Q5_K) {
        b += block * 176;
        const float d = ggml_f16_to_f32_unaligned(b);
        const float dmin = ggml_f16_to_f32_unaligned(b + 2);
        device const uint8_t* sc = b + 4;
        device const uint8_t* qh = b + 16;
        device const uint8_t* ql = b + 48;
        const int g = inb / 64;
        const int sub = (inb % 64) / 32;
        const int j = 2 * g + sub;
        int scale = 0, minv = 0;
        ggml_q4_k_scale_min(sc, j, scale, minv);
        const int idx = inb % 32;
        const uint8_t qbyte = ql[g * 32 + idx];
        const int lo = sub != 0 ? ((qbyte >> 4) & 0x0F) : (qbyte & 0x0F);
        const int hi = (qh[idx] & (sub != 0 ? (2 << (2 * g)) : (1 << (2 * g)))) ? 16 : 0;
        return d * float(scale) * float(lo + hi) - dmin * float(minv);
    }
    if (qtype == GGML_Q6_K) {
        b += block * 210;
        device const uint8_t* ql = b;
        device const uint8_t* qh = b + 128;
        device const int8_t* sc = reinterpret_cast<device const int8_t*>(b + 192);
        const float d = ggml_f16_to_f32_unaligned(b + 208);
        const int half_idx = inb / 128;
        const int pos = inb - half_idx * 128;
        const int idx = pos % 32;
        int q = 0;
        int scale_idx = half_idx * 8 + idx / 16;
        const uint8_t qh_byte = qh[half_idx * 32 + idx];
        if (pos < 32) {
            q = int(ql[half_idx * 64 + idx] & 0x0F) | int(((qh_byte >> 0) & 3) << 4);
        } else if (pos < 64) {
            q = int(ql[half_idx * 64 + 32 + idx] & 0x0F) | int(((qh_byte >> 2) & 3) << 4);
            scale_idx += 2;
        } else if (pos < 96) {
            q = int((ql[half_idx * 64 + idx] >> 4) & 0x0F) | int(((qh_byte >> 4) & 3) << 4);
            scale_idx += 4;
        } else {
            q = int((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0F) | int(((qh_byte >> 6) & 3) << 4);
            scale_idx += 6;
        }
        return d * float(sc[scale_idx]) * float(q - 32);
    }
    return 0.0f;
}

inline float ggml_k_dequant_scalar(device const uchar* w_ptr, int qtype, int row, int col, int cols) {
    device const uint8_t* data = reinterpret_cast<device const uint8_t*>(w_ptr);
    device const uint8_t* row_data = data + uint(row) * uint(ggml_k_row_bytes(qtype, cols));
    return ggml_k_dequant_scalar_from_row(row_data, qtype, col, cols);
}

inline float int4_dequant_scalar(device const uchar* w_ptr, device const uchar* scale_ptr, device const uchar* zero_ptr,
                                 int row, int col, int cols, int group_size, int scale_dtype) {
    device const uint8_t* data = reinterpret_cast<device const uint8_t*>(w_ptr);
    int byte_cols = cols / 2;
    uint8_t packed_byte = data[uint(row) * uint(byte_cols) + col / 2];
    int nibble = (col & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
    int si = (row / group_size) * ((cols + group_size - 1) / group_size) + col / group_size;
    float s = load_elem(scale_ptr, uint(si), scale_dtype);
    float z = load_elem(zero_ptr, uint(si), scale_dtype);
    return bf16_round_rne_finite(float(nibble) * s - z * s);
}

inline float lowbit_dequant_scalar(device const uchar* rhs_base, int quant_type, int global_n, int global_k, int k,
                                   device const uchar* scale_ptr, device const uchar* zero_ptr, int group_size,
                                   int scale_dtype, float tensor_scale, int grid_code) {
    if (qwen35_lowbit_native_int4(quant_type)) {
        return int4_dequant_scalar(rhs_base, scale_ptr, zero_ptr, global_n, global_k, k, group_size, scale_dtype);
    }
    if (quant_type == GQH3 || quant_type == GQH2_H || quant_type == GQH4) {
        return qwen35_gqh_dequant_scalar(rhs_base, quant_type, global_n, global_k, k, tensor_scale, grid_code);
    }
    return ggml_k_dequant_scalar(rhs_base, quant_type, global_n, global_k, k);
}

inline float dequant_matmul_weight(
    device const uchar* rhs,
    uint rhs_base,
    int quant_type,
    int global_n,
    int global_k,
    int k,
    int group_size,
    device const uchar* scale,
    device const uchar* zero,
    device const uchar* awq_inv_scale,
    int scale_dtype,
    float tensor_scale,
    int grid_code) {
    float w = 0.0f;
    if (qwen35_lowbit_native_int4(quant_type)) {
        const int row_bytes = k / 2;
        const int byte_idx = global_k / 2;
        const uint8_t packed_byte =
            reinterpret_cast<device const uint8_t*>(rhs)[rhs_base + uint(global_n) * uint(row_bytes) + uint(byte_idx)];
        const int nibble = (global_k & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
        const int scale_cols = (k + group_size - 1) / group_size;
        const int si = (global_n / group_size) * scale_cols + (global_k / group_size);
        const float s = load_elem(scale, uint(si), scale_dtype);
        const float z = load_elem(zero, uint(si), scale_dtype);
        w = bf16_round_rne_finite(float(nibble) * s - z * s);
    } else {
        w = lowbit_dequant_scalar(
            rhs + rhs_base, quant_type, global_n, global_k, k, scale, zero, group_size, scale_dtype, tensor_scale,
            grid_code);
    }
    if (awq_inv_scale != nullptr) {
        w *= load_elem(awq_inv_scale, uint(global_k), scale_dtype);
    }
    return w;
}

#endif // PREFILL_COMMON_METAL
