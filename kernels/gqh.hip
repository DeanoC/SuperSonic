// GQH (Geo-Quant Hierarchical) HIP decode and fused batch-1 matvec.
//
// Bit-exact against geoquant/formats/gqh.py and crates/model-store/src/gqh.rs.
// Superblock scale is d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15). Do not reassociate those products.
// Ported from geo-lucebox ggml-cuda/gqh.cu (wave32, header as kernel args).

#include "gqh-tables.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

static __constant__ uint32_t GQH_E4M3_D[32][8] = GQH_E4M3_LUT_INIT;
static __constant__ uint32_t GQH_RATIO_Q_D[16][1] = GQH_RATIO_Q_INIT;
static __constant__ uint32_t GQH2C_CB_D[256][8] = GQH2C_CODEBOOK_INIT;
static __constant__ uint8_t GQH2C_SIGN_D[128] = GQH2C_SIGN_MASK_INIT;

struct gqh_grid8 {
    float v[8];
};
struct gqh_grid4 {
    float v[4];
};

static __device__ __forceinline__ float gqh_bits(uint32_t u) {
    return __int_as_float((int)u);
}

#pragma clang fp contract(off)
static __device__ __forceinline__ float gqh_subblock_scale(
    const uint8_t* __restrict__ b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

__global__ void gqh3_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid8 grid,
    float* __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;
    const uint8_t* __restrict__ b = wire + sb * GQH3_SB_BYTES;

    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);
    const int lo = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
    const int hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;
    dst[sb * GQH_SUPERBLOCK + j] = grid.v[lo | (hi << 2)] * s_b;
}

__global__ void gqh2h_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid4 grid,
    float* __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;
    const uint8_t* __restrict__ b = wire + sb * GQH2H_SB_BYTES;

    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);
    const int code = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
    dst[sb * GQH_SUPERBLOCK + j] = grid.v[code] * s_b;
}

__global__ void gqh2c_decode_kernel(
    const uint8_t* __restrict__ wire, float* __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;
    const uint8_t* __restrict__ b = wire + sb * GQH2C_SB_BYTES;

    const int blk = j / GQH2C_BLOCK;
    const int grp = (j % GQH2C_BLOCK) / GQH2C_GROUP;
    const int e = j % GQH2C_GROUP;

    __half dh;
    memcpy(&dh, b, sizeof(dh));
    const uint8_t* p = b + 2 + blk * 8;
    uint32_t u;
    memcpy(&u, p + 4, sizeof(u));

    const float s_blk = __half2float(dh) * gqh_bits(GQH_RATIO_Q_D[(u >> 28) & 0x0f][0]);
    const uint8_t mask = GQH2C_SIGN_D[(u >> (7 * grp)) & 0x7f];
    const float raw = gqh_bits(GQH2C_CB_D[p[grp]][e]);
    const float mag = ((mask >> e) & 1) ? -raw : raw;
    dst[sb * GQH_SUPERBLOCK + j] = mag * s_blk;
}

#define GQH_WARP 32
#define GQH_MATVEC_WARPS 4
#define GQH_PER_LANE (GQH_SUPERBLOCK / GQH_WARP)

static __device__ __forceinline__ float gqh_warp_shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(v, off, GQH_WARP);
#else
    return __shfl_down_sync(0xffffffffu, v, off, GQH_WARP);
#endif
}

// Grids are symmetric about zero. Selecting magnitude+sign from registers
// avoids a divergent 8-way constant-bank load of the full table.
// gqh3:   grid = [-m3,-m2,-m1,-m0, m0,m1,m2,m3]
// gqh2_h: grid = [-1, -a, +a, +1]
template <bool IS_GQH3>
static __device__ __forceinline__ float gqh_level(int code, const float4& m) {
    if (IS_GQH3) {
        const int hi = (code >> 2) & 1;
        const int k = code & 3;
        const int idx = hi ? k : (3 - k);
        const float e0 = (idx & 1) ? m.y : m.x;
        const float e1 = (idx & 1) ? m.w : m.z;
        const float mag = (idx & 2) ? e1 : e0;
        return hi ? mag : -mag;
    }
    const int hi = (code >> 1) & 1;
    const float mag = ((code & 1) != hi) ? m.x : 1.0f;
    return hi ? mag : -mag;
}

template <bool IS_GQH3>
__global__ void gqh_matvec_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, float4 mag, int64_t x_col_stride,
    int64_t y_col_stride) {
    const int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;

    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    if (row >= out) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    const uint8_t* __restrict__ rowbase = data + (int64_t)row * nsb * sb_bytes;
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;

    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* __restrict__ b = rowbase + (int64_t)sb * sb_bytes;

        const uint8_t d = b[0];
        const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
        const uint8_t rb = b[1 + (sub >> 1)];
        const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];

        const float4 x0 = *reinterpret_cast<const float4*>(xc + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 = *reinterpret_cast<const float4*>(xc + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};

        uint16_t lo2;
        memcpy(&lo2, b + 9 + lane * 2, sizeof(lo2));
        const uint8_t hi1 = IS_GQH3 ? b[73 + lane] : 0;

#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            const int lo = (lo2 >> (2 * t)) & 0x03;
            const int code = IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
            acc += (gqh_level<IS_GQH3>(code, mag) * s_b) * xs[t];
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    if (lane == 0) {
        y[(int64_t)col * y_col_stride + row] = acc;
    }
}

__global__ void gqh2c_matvec_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, int64_t x_col_stride, int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;
    if (row >= out) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    const uint8_t* __restrict__ rowbase = data + (int64_t)row * nsb * GQH2C_SB_BYTES;
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;

    const int blk = lane / GQH2C_GROUPS_PER_BLOCK;
    const int grp = lane % GQH2C_GROUPS_PER_BLOCK;
    const int j0 = blk * GQH2C_BLOCK + grp * GQH2C_GROUP;

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* __restrict__ b = rowbase + (int64_t)sb * GQH2C_SB_BYTES;
        __half dh;
        memcpy(&dh, b, sizeof(dh));
        const uint8_t* __restrict__ p = b + 2 + blk * 8;
        uint32_t u;
        memcpy(&u, p + 4, sizeof(u));

        const float s_blk = __half2float(dh) * gqh_bits(GQH_RATIO_Q_D[(u >> 28) & 0x0f][0]);
        const uint8_t mask = GQH2C_SIGN_D[(u >> (7 * grp)) & 0x7f];
        const uint32_t* cb = GQH2C_CB_D[p[grp]];
        const float* __restrict__ xs = xc + sb * GQH_SUPERBLOCK + j0;

#pragma unroll
        for (int e = 0; e < GQH2C_GROUP; ++e) {
            const float raw = gqh_bits(cb[e]);
            const float m = ((mask >> e) & 1) ? -raw : raw;
            acc += (m * s_blk) * xs[e];
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    if (lane == 0) {
        y[(int64_t)col * y_col_stride + row] = acc;
    }
}
