// GQH (Geo-Quant Hierarchical) HIP decode and fused batch-1 matvec.
//
// Bit-exact against geoquant/formats/gqh.py and crates/model-store/src/gqh.rs.
// Superblock scale is d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15). Do not reassociate those products.
// Ported from geo-lucebox ggml-cuda/gqh.cu (wave32, header as kernel args).

#include "gqh-stride.h"

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

static __device__ __forceinline__ float gqh_bf16_round(float x) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    bits &= 0xFFFF0000u;
    float y;
    memcpy(&y, &bits, 4);
    return y;
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

#pragma clang fp contract(off)
static __device__ __forceinline__ float gqh_scale_from_planes(
    uint8_t d, const uint8_t* __restrict__ ratio8, int sub, float tensor_scale) {
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = ratio8[sub >> 1];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

// Planes are 8/64/32-byte aligned. Typed loads so the compiler emits
// ushort/dword/dwordx2 instead of byte gathers + memcpy.
static __device__ __forceinline__ uint16_t gqh_ld_u16(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}
static __device__ __forceinline__ uint32_t gqh_ld_u32(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
static __device__ __forceinline__ uint64_t gqh_ld_u64(const uint8_t* p) {
    return *reinterpret_cast<const uint64_t*>(p);
}

static __device__ __forceinline__ uint8_t gqh_ratio_byte(uint64_t r64, int sub) {
    return static_cast<uint8_t>(r64 >> (8 * (sub >> 1)));
}

static __device__ __forceinline__ uint8_t gqh_hi_byte(const uint8_t* hi_sb, int lane) {
    const uint32_t w = gqh_ld_u32(hi_sb + (lane & ~3));
    return static_cast<uint8_t>(w >> (8 * (lane & 3)));
}

__global__ void gqh3_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid8 grid,
    float* __restrict__ dst, int nsb) {
    const int row = (int)blockIdx.x / nsb;
    const int sb = (int)blockIdx.x % nsb;
    const int j = threadIdx.x;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, 1, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* __restrict__ rowbase = wire + (size_t)row * (size_t)stride;
    const uint8_t d = rowbase[sb];
    const uint8_t* __restrict__ rat = rowbase + off_r + (sb << 3);
    const uint8_t* __restrict__ lo_p = rowbase + off_lo + (sb << 6);
    const uint8_t* __restrict__ hi_p = rowbase + off_hi + (sb << 5);
    const float s_b = gqh_scale_from_planes(d, rat, j >> 4, tensor_scale);
    const int lo = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
    const int hi = (hi_p[j >> 3] >> (j & 7)) & 0x01;
    dst[(int64_t)blockIdx.x * GQH_SUPERBLOCK + j] = grid.v[lo | (hi << 2)] * s_b;
}

__global__ void gqh2h_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid4 grid,
    float* __restrict__ dst, int nsb) {
    const int row = (int)blockIdx.x / nsb;
    const int sb = (int)blockIdx.x % nsb;
    const int j = threadIdx.x;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, 0, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* __restrict__ rowbase = wire + (size_t)row * (size_t)stride;
    const uint8_t d = rowbase[sb];
    const uint8_t* __restrict__ rat = rowbase + off_r + (sb << 3);
    const uint8_t* __restrict__ lo_p = rowbase + off_lo + (sb << 6);
    const float s_b = gqh_scale_from_planes(d, rat, j >> 4, tensor_scale);
    const int code = (lo_p[j >> 2] >> (2 * (j & 3))) & 0x03;
    dst[(int64_t)blockIdx.x * GQH_SUPERBLOCK + j] = grid.v[code] * s_b;
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
#define GQH_MATVEC_WARPS 8
#define GQH_PER_LANE (GQH_SUPERBLOCK / GQH_WARP)
// 20 KiB x-in-LDS limits occupancy to ~3 blocks/WGP. x is 5120 floats and
// is reused across every row, so L2 should hold it. 0 disables staging.
#define GQH_LDS_X_MAX 0

extern __shared__ char gqh_dyn_smem[];

static __device__ __forceinline__ float gqh_warp_shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(v, off, GQH_WARP);
#else
    return __shfl_down_sync(0xffffffffu, v, off, GQH_WARP);
#endif
}

// Grids are symmetric about zero.
// gqh3:   grid = [-m3,-m2,-m1,-m0, m0,m1,m2,m3]
// gqh2_h: grid = [-1, -a, +a, +1]
// Sign is an IEEE xor so the inner loop is not a cndmask tree. Magnitude
// is a 4-entry VGPR table (kernel-arg float4 lives in SGPRs; indexing
// that with a VGPR rebuilds ~3 cndmasks per weight).
static __device__ __forceinline__ float gqh_with_sign(float mag, int positive) {
    uint32_t bits;
    memcpy(&bits, &mag, 4);
    bits ^= (uint32_t)(positive - 1) & 0x80000000u;
    memcpy(&mag, &bits, 4);
    return mag;
}

static __device__ __forceinline__ float gqh3_level_tbl(int code, const float* tbl) {
    const int hi = (code >> 2) & 1;
    const int k = code & 3;
    // hi=1 → k; hi=0 → k^3. Same as `hi ? k : (3-k)` for k in 0..3.
    const int idx = k ^ ((hi - 1) & 3);
    return gqh_with_sign(tbl[idx], hi);
}

static __device__ __forceinline__ float gqh3_level(
    int code, float m0, float m1, float m2, float m3) {
    const float tbl[4] = {m0, m1, m2, m3};
    return gqh3_level_tbl(code, tbl);
}

static __device__ __forceinline__ float gqh2h_level(int code, float a) {
    const int hi = (code >> 1) & 1;
    const float mag = ((code & 1) != hi) ? a : 1.0f;
    return gqh_with_sign(mag, hi);
}

template <bool IS_GQH3>
static __device__ __forceinline__ float gqh_level(int code, const float4& m) {
    if (IS_GQH3) {
        return gqh3_level(code, m.x, m.y, m.z, m.w);
    }
    return gqh2h_level(code, m.x);
}

// One superblock's per-lane payload. Fat-K (nsb=68) issues the next
// pack while the current FMA chain runs. Product order is unchanged:
// d_real = e4m3(d)*tensor_scale, s_b = d_real*(ratio/15), then
// acc += (lvl * s_b) * x, superblocks in order.
struct gqh_sb_pack {
    uint8_t d;
    uint8_t hi1;
    uint16_t lo2;
    uint64_t r64;
    float4 x0;
    float4 x1;
};

template <bool IS_GQH3>
static __device__ __forceinline__ gqh_sb_pack gqh_load_sb(
    const uint8_t* __restrict__ d_row,
    const uint8_t* __restrict__ ratio_row,
    const uint8_t* __restrict__ lo_row,
    const uint8_t* __restrict__ hi_row,
    const float* __restrict__ xsrc,
    int sb,
    int lane,
    int j0) {
    gqh_sb_pack p;
    p.d = d_row[sb];
    p.r64 = gqh_ld_u64(ratio_row + (sb << 3));
    p.lo2 = gqh_ld_u16(lo_row + (sb << 6) + lane * 2);
    p.hi1 = IS_GQH3 ? gqh_hi_byte(hi_row + (sb << 5), lane) : 0;
    p.x0 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0);
    p.x1 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0 + 4);
    return p;
}

static __device__ __forceinline__ void gqh_pipe_barrier() {
    // Stop clang sinking the next-SB loads into the FMA body.
    asm volatile("" ::: "memory");
}

#pragma clang fp contract(off)
template <bool IS_GQH3>
static __device__ __forceinline__ float gqh_sb_fma(
    float acc,
    const gqh_sb_pack& p,
    float tensor_scale,
    const float* __restrict__ s_ratio,
    const float* __restrict__ s_mag,
    float m0,
    int sub) {
    const uint8_t rb = gqh_ratio_byte(p.r64, sub);
    const float d_real = gqh_bits(GQH_E4M3_D[p.d >> 3][p.d & 7]) * tensor_scale;
    const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
    const float xs[GQH_PER_LANE] = {
        p.x0.x, p.x0.y, p.x0.z, p.x0.w, p.x1.x, p.x1.y, p.x1.z, p.x1.w};
#pragma unroll
    for (int t = 0; t < GQH_PER_LANE; ++t) {
        const int lo = (p.lo2 >> (2 * t)) & 0x03;
        const int code = IS_GQH3 ? (lo | (((p.hi1 >> t) & 1) << 2)) : lo;
        const float lvl = IS_GQH3 ? gqh3_level_tbl(code, s_mag)
                                  : gqh2h_level(code, m0);
        acc += (lvl * s_b) * xs[t];
    }
    return acc;
}

// kPipe=1: depth-1 software pipeline for fat-K (down, nsb=68).
// kPipe=0: original loop; hidden-width singles stay on this codegen.
template <bool IS_GQH3, bool kAcc = false, int kPipe = 0>
__global__ void gqh_matvec_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, float4 mag, int64_t x_col_stride,
    int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;

    __shared__ float s_ratio[16];
    __shared__ float s_mag[4];
    float* s_x = reinterpret_cast<float*>(gqh_dyn_smem);
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    if constexpr (IS_GQH3) {
        if (threadIdx.x < 4) {
            s_mag[threadIdx.x] = (&mag.x)[threadIdx.x];
        }
    }
    const bool use_lds = (in <= GQH_LDS_X_MAX);
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;
    if (use_lds) {
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            s_x[i] = xc[i];
        }
    }
    __syncthreads();

    const bool valid = row < out;
    if (!valid) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    const uint8_t* __restrict__ rowbase = data + (int64_t)row * stride;
    const uint8_t* __restrict__ d_row = rowbase;
    const uint8_t* __restrict__ ratio_row = rowbase + off_r;
    const uint8_t* __restrict__ lo_row = rowbase + off_lo;
    const uint8_t* __restrict__ hi_row = rowbase + off_hi;
    const float* __restrict__ xsrc = use_lds ? s_x : xc;

    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;
    const float m0 = mag.x, m1 = mag.y, m2 = mag.z, m3 = mag.w;

    float acc = 0.0f;
    if constexpr (kPipe != 0) {
        // Issue SB[i+1] under SB[i] FMAs. Same SB order and products.
        gqh_sb_pack cur = gqh_load_sb<IS_GQH3>(
            d_row, ratio_row, lo_row, hi_row, xsrc, 0, lane, j0);
        for (int sb = 0; sb < nsb - 1; ++sb) {
            gqh_sb_pack nxt = gqh_load_sb<IS_GQH3>(
                d_row, ratio_row, lo_row, hi_row, xsrc, sb + 1, lane, j0);
            gqh_pipe_barrier();
            acc = gqh_sb_fma<IS_GQH3>(
                acc, cur, tensor_scale, s_ratio, s_mag, m0, sub);
            cur = nxt;
        }
        acc = gqh_sb_fma<IS_GQH3>(
            acc, cur, tensor_scale, s_ratio, s_mag, m0, sub);
    } else {
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t d = d_row[sb];
            const uint64_t r64 = gqh_ld_u64(ratio_row + (sb << 3));
            const uint8_t rb = gqh_ratio_byte(r64, sub);
            const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
            const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];

            const float4 x0 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0);
            const float4 x1 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0 + 4);
            const float xs[GQH_PER_LANE] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};

            const uint16_t lo2 = gqh_ld_u16(lo_row + (sb << 6) + lane * 2);
            const uint8_t hi1 = IS_GQH3 ? gqh_hi_byte(hi_row + (sb << 5), lane) : 0;

#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int lo = (lo2 >> (2 * t)) & 0x03;
                const int code = IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
                const float lvl = IS_GQH3 ? gqh3_level_tbl(code, s_mag)
                                          : gqh2h_level(code, m0);
                acc += (lvl * s_b) * xs[t];
            }
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    if (valid && lane == 0) {
        const int64_t yi = (int64_t)col * y_col_stride + row;
        if constexpr (kAcc) {
            y[yi] = gqh_bf16_round(y[yi] + gqh_bf16_round(acc));
        } else {
            y[yi] = acc;
        }
    }
}

// One launch, one x load, two independent row dots (gate + up). Same
// product order as gqh_matvec_kernel; the two accs never mix.
template <bool A_IS_GQH3, bool B_IS_GQH3>
__global__ void gqh_matvec_pair_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y_a,
    float* __restrict__ y_b,
    int in,
    int out,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;

    __shared__ float s_ratio[16];
    __shared__ float s_mag_a[4];
    __shared__ float s_mag_b[4];
    float* s_x = reinterpret_cast<float*>(gqh_dyn_smem);
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    if (threadIdx.x < 4) {
        if constexpr (A_IS_GQH3) {
            s_mag_a[threadIdx.x] = (&mag_a.x)[threadIdx.x];
        }
        if constexpr (B_IS_GQH3) {
            s_mag_b[threadIdx.x] = (&mag_b.x)[threadIdx.x];
        }
    }
    const bool use_lds = (in <= GQH_LDS_X_MAX);
    if (use_lds) {
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            s_x[i] = x[i];
        }
    }
    __syncthreads();
    if (row >= out) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    int off_ra, off_loa, off_hia, stride_a;
    int off_rb, off_lob, off_hib, stride_b;
    gqh_plane_offsets(nsb, A_IS_GQH3 ? 1 : 0, &off_ra, &off_loa, &off_hia, &stride_a);
    gqh_plane_offsets(nsb, B_IS_GQH3 ? 1 : 0, &off_rb, &off_lob, &off_hib, &stride_b);
    const uint8_t* __restrict__ row_a = data_a + (int64_t)row * stride_a;
    const uint8_t* __restrict__ row_b = data_b + (int64_t)row * stride_b;
    const float* __restrict__ xsrc = use_lds ? s_x : x;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;
    const float a0 = mag_a.x, a1 = mag_a.y, a2 = mag_a.z, a3 = mag_a.w;
    const float b0 = mag_b.x, b1 = mag_b.y, b2 = mag_b.z, b3 = mag_b.w;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const float4 x0 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};

        {
            const uint8_t d = row_a[sb];
            const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * scale_a;
            const uint64_t r64 = gqh_ld_u64(row_a + off_ra + (sb << 3));
            const uint8_t rb = gqh_ratio_byte(r64, sub);
            const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
            const uint16_t lo2 = gqh_ld_u16(row_a + off_loa + (sb << 6) + lane * 2);
            const uint8_t hi1 = A_IS_GQH3 ? gqh_hi_byte(row_a + off_hia + (sb << 5), lane) : 0;
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int lo = (lo2 >> (2 * t)) & 0x03;
                const int code = A_IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
                const float lvl = A_IS_GQH3 ? gqh3_level_tbl(code, s_mag_a)
                                            : gqh2h_level(code, a0);
                acc_a += (lvl * s_b) * xs[t];
            }
        }
        {
            const uint8_t d = row_b[sb];
            const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * scale_b;
            const uint64_t r64 = gqh_ld_u64(row_b + off_rb + (sb << 3));
            const uint8_t rb = gqh_ratio_byte(r64, sub);
            const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
            const uint16_t lo2 = gqh_ld_u16(row_b + off_lob + (sb << 6) + lane * 2);
            const uint8_t hi1 = B_IS_GQH3 ? gqh_hi_byte(row_b + off_hib + (sb << 5), lane) : 0;
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int lo = (lo2 >> (2 * t)) & 0x03;
                const int code = B_IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
                const float lvl = B_IS_GQH3 ? gqh3_level_tbl(code, s_mag_b)
                                            : gqh2h_level(code, b0);
                acc_b += (lvl * s_b) * xs[t];
            }
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc_a += gqh_warp_shfl_down(acc_a, off);
        acc_b += gqh_warp_shfl_down(acc_b, off);
    }
    if (lane == 0) {
        y_a[row] = acc_a;
        y_b[row] = acc_b;
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
