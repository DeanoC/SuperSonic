// GQH (Geo-Quant Hierarchical) HIP decode and fused batch-1 matvec.
//
// Bit-exact against geoquant/formats/gqh.py and crates/model-store/src/gqh.rs.
// Superblock scale is d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15). Do not reassociate those products.
// Ported from geo-lucebox ggml-cuda/gqh.cu (wave32, header as kernel args).

#include "gqh-stride.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

template <typename T>
__device__ __forceinline__ T gqh_store_val(float v);

template <>
__device__ __forceinline__ float gqh_store_val<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ hip_bfloat16 gqh_store_val<hip_bfloat16>(float v) {
    return static_cast<hip_bfloat16>(v);
}

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

#ifndef GQH_ALLOW_FMA
#pragma clang fp contract(off)
#endif
static __device__ __forceinline__ float gqh_subblock_scale(
    const uint8_t* __restrict__ b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

#ifndef GQH_ALLOW_FMA
#pragma clang fp contract(off)
#endif
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

// Streaming weight loads so x stays in L2. 1-byte builtin is always aligned.
static __device__ __forceinline__ uint8_t gqh_ld_nt_u8(const uint8_t* p) {
    return __builtin_nontemporal_load(p);
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

template <typename T>
__global__ void gqh3_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid8 grid,
    T* __restrict__ dst, int nsb) {
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
    dst[(int64_t)blockIdx.x * GQH_SUPERBLOCK + j] =
        gqh_store_val<T>(grid.v[lo | (hi << 2)] * s_b);
}

template <typename T>
__global__ void gqh2h_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid4 grid,
    T* __restrict__ dst, int nsb) {
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
    dst[(int64_t)blockIdx.x * GQH_SUPERBLOCK + j] =
        gqh_store_val<T>(grid.v[code] * s_b);
}

template <typename T>
__global__ void gqh2c_decode_kernel(
    const uint8_t* __restrict__ wire, T* __restrict__ dst) {
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
    dst[sb * GQH_SUPERBLOCK + j] = gqh_store_val<T>(mag * s_blk);
}

#define GQH_WARP 32
#define GQH_MATVEC_WARPS 16
#define GQH_PER_LANE (GQH_SUPERBLOCK / GQH_WARP)
// Staging x in LDS costs 20 KiB and a fill per block. A full-grid lm_head
// is 15520 blocks × 20 KiB ≈ 310 MiB of LDS traffic for a 20 KiB vector
// that L2 already holds. Stage when few blocks share x: either a short
// output, or a persist grid that walks rows (grid*warps < out).
#define GQH_LDS_X_MAX 8192
#define GQH_LDS_OUT_MAX 4096

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
    // llama.cpp gqh.cu: cndmask tree, not a VGPR/LDS table. Indexing a
    // float4 with a VGPR rebuilds ~3 cndmasks per weight; this is the
    // same tree the CUDA kernel uses so occupancy stays comparable.
    if (IS_GQH3) {
        const int hi = (code >> 2) & 1;
        const int k = code & 3;
        const int idx = hi ? k : (3 - k);
        const float e0 = (idx & 1) ? m.y : m.x;
        const float e1 = (idx & 1) ? m.w : m.z;
        const float mag = (idx & 2) ? e1 : e0;
        return hi ? mag : -mag;
    }
    return gqh2h_level(code, m.x);
}

// llama.cpp gqh.cu does acc += (lvl * s_b) * x with FMA on the add.
// Scale stays two muls: e4m3(d)*tensor_scale, then *(ratio/15).
static __device__ __forceinline__ float gqh_acc_fma(
    float acc, float lvl, float s_b, float x) {
    return fmaf(lvl * s_b, x, acc);
}

// Tight AoS lane payload. Scale products stay two muls.
struct gqh_tight_lane {
    float s_b;
    uint16_t lo2;
    uint8_t hi1;
};

static __device__ __forceinline__ const uint8_t* gqh_tight_rowptr(
    const uint8_t* data, int row, int stride, int sb_bytes, int ileave) {
    if (ileave) {
        return data + (int64_t)(row & ~1) * stride + (row & 1) * sb_bytes;
    }
    return data + (int64_t)row * stride;
}

template <bool IS_GQH3>
static __device__ __forceinline__ gqh_tight_lane gqh_load_tight_lane(
    const uint8_t* __restrict__ row,
    int sb,
    int sb_pitch,
    int lane,
    int sub,
    float tensor_scale,
    const float* __restrict__ s_ratio) {
    const uint8_t* __restrict__ b = row + sb * sb_pitch;
    const uint8_t d = gqh_ld_nt_u8(b);
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = gqh_ld_nt_u8(b + 1 + (sub >> 1));
    gqh_tight_lane p;
    p.s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
    memcpy(&p.lo2, b + 9 + lane * 2, sizeof(p.lo2));
    p.hi1 = IS_GQH3 ? gqh_ld_nt_u8(b + 73 + lane) : 0;
    return p;
}

template <bool IS_GQH3>
static __device__ __forceinline__ gqh_tight_lane gqh_load_padded_lane(
    const uint8_t* __restrict__ row,
    int sb,
    int sb_pitch,
    int lane,
    int sub,
    float tensor_scale,
    const float* __restrict__ s_ratio) {
    const uint8_t* __restrict__ b = row + sb * sb_pitch;
    const uint8_t d = b[0];
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    gqh_tight_lane p;
    p.s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
    p.lo2 = *reinterpret_cast<const uint16_t*>(b + GQH_PAD_LO + lane * 2);
    p.hi1 = IS_GQH3 ? b[GQH_PAD_HI + lane] : 0;
    return p;
}

template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_acc_tight_lane(
    float& acc, const gqh_tight_lane& p, const float4& mag, const float xs[8]) {
#pragma unroll
    for (int t = 0; t < GQH_PER_LANE; ++t) {
        const int lo = (p.lo2 >> (2 * t)) & 0x03;
        const int code = IS_GQH3 ? (lo | (((p.hi1 >> t) & 1) << 2)) : lo;
        acc = gqh_acc_fma(acc, gqh_level<IS_GQH3>(code, mag), p.s_b, xs[t]);
    }
}

// One output row, same product order as gqh_matvec_tight_kernel kRows=1.
template <bool IS_GQH3>
static __device__ __forceinline__ float gqh_tight_row_dot(
    const uint8_t* __restrict__ data,
    const float* __restrict__ x,
    int row,
    int nsb,
    int stride,
    int sb_bytes,
    int ileave,
    int lane,
    int sub,
    int j0,
    float tensor_scale,
    const float4& mag,
    const float* __restrict__ s_ratio) {
    const uint8_t* __restrict__ rowbase =
        gqh_tight_rowptr(data, row, stride, sb_bytes, ileave);
    const int sb_pitch = ileave ? sb_bytes * 2 : sb_bytes;
    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const float4 x0 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {
            x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const gqh_tight_lane p = gqh_load_tight_lane<IS_GQH3>(
            rowbase, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        gqh_acc_tight_lane<IS_GQH3>(acc, p, mag, xs);
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    return acc;
}

// One launch streams two independent GEMVs (shared x). Grid covers
// out_a+out_b rows so both weight tensors are in flight together.
// Per-row product order matches two separate tight singles launches.
template <bool A_IS_GQH3, bool B_IS_GQH3>
__global__ void gqh_matvec_ab_tight_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y_a,
    float* __restrict__ y_b,
    int in,
    int out_a,
    int out_b,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b,
    int ileave) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int warp = threadIdx.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + warp;
    const int total = out_a + out_b;

    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    if (row >= total) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;
    if (row < out_a) {
        int off_r, off_lo, off_hi, stride;
        gqh_plane_offsets(nsb, A_IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
        (void)off_r;
        (void)off_lo;
        (void)off_hi;
        constexpr int sb = A_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
        const float acc = gqh_tight_row_dot<A_IS_GQH3>(
            data_a, x, row, nsb, stride, sb, ileave, lane, sub, j0, scale_a,
            mag_a, s_ratio);
        if (lane == 0) {
            y_a[row] = acc;
        }
    } else {
        const int rowb = row - out_a;
        int off_r, off_lo, off_hi, stride;
        gqh_plane_offsets(nsb, B_IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
        (void)off_r;
        (void)off_lo;
        (void)off_hi;
        constexpr int sb = B_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
        const float acc = gqh_tight_row_dot<B_IS_GQH3>(
            data_b, x, rowb, nsb, stride, sb, ileave, lane, sub, j0, scale_b,
            mag_b, s_ratio);
        if (lane == 0) {
            y_b[rowb] = acc;
        }
    }
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

#ifndef GQH_ALLOW_FMA
#pragma clang fp contract(off)
#endif
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
template <bool IS_GQH3, bool kAcc = false, int kPipe = 0, int kRows = 1>
__global__ void gqh_matvec_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, float4 mag, int64_t x_col_stride,
    int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;
    const int warp = threadIdx.x / GQH_WARP;
    const int row_base = (kRows == 2)
        ? ((blockIdx.x * warps_per_block + warp) * 2)
        : (blockIdx.x * warps_per_block + warp);
    const int row_stride = (kRows == 2)
        ? (gridDim.x * warps_per_block * 2)
        : (gridDim.x * warps_per_block);

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
    const bool use_lds = (in <= GQH_LDS_X_MAX && out <= GQH_LDS_OUT_MAX);
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;
    if (use_lds) {
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            s_x[i] = xc[i];
        }
    }
    __syncthreads();

    if (row_base >= out) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    const float* __restrict__ xsrc = use_lds ? s_x : xc;

    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;
    const float m0 = mag.x, m1 = mag.y, m2 = mag.z, m3 = mag.w;
    (void)m1;
    (void)m2;
    (void)m3;

    for (int row0 = row_base; row0 < out; row0 += row_stride) {
        const bool valid1 = (kRows == 2) && (row0 + 1 < out);
        const uint8_t* __restrict__ rowbase0 = data + (int64_t)row0 * stride;
        const uint8_t* __restrict__ d0 = rowbase0;
        const uint8_t* __restrict__ ratio0 = rowbase0 + off_r;
        const uint8_t* __restrict__ lo0 = rowbase0 + off_lo;
        const uint8_t* __restrict__ hi0 = rowbase0 + off_hi;
        const uint8_t* __restrict__ rowbase1 =
            valid1 ? data + (int64_t)(row0 + 1) * stride : rowbase0;
        const uint8_t* __restrict__ d1 = rowbase1;
        const uint8_t* __restrict__ ratio1 = rowbase1 + off_r;
        const uint8_t* __restrict__ lo1 = rowbase1 + off_lo;
        const uint8_t* __restrict__ hi1p = rowbase1 + off_hi;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        if constexpr (kPipe != 0) {
            gqh_sb_pack cur = gqh_load_sb<IS_GQH3>(
                d0, ratio0, lo0, hi0, xsrc, 0, lane, j0);
            for (int sb = 0; sb < nsb - 1; ++sb) {
                gqh_sb_pack nxt = gqh_load_sb<IS_GQH3>(
                    d0, ratio0, lo0, hi0, xsrc, sb + 1, lane, j0);
                gqh_pipe_barrier();
                acc0 = gqh_sb_fma<IS_GQH3>(
                    acc0, cur, tensor_scale, s_ratio, s_mag, m0, sub);
                cur = nxt;
            }
            acc0 = gqh_sb_fma<IS_GQH3>(
                acc0, cur, tensor_scale, s_ratio, s_mag, m0, sub);
        } else {
            for (int sb = 0; sb < nsb; ++sb) {
                const float4 x0 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0);
                const float4 x1 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0 + 4);
                const float xs[GQH_PER_LANE] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};

                {
                    const uint8_t d = d0[sb];
                    const uint64_t r64 = gqh_ld_u64(ratio0 + (sb << 3));
                    const uint8_t rb = gqh_ratio_byte(r64, sub);
                    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
                    const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
                    const uint16_t lo2 = gqh_ld_u16(lo0 + (sb << 6) + lane * 2);
                    const uint8_t hi1 = IS_GQH3 ? gqh_hi_byte(hi0 + (sb << 5), lane) : 0;
#pragma unroll
                    for (int t = 0; t < GQH_PER_LANE; ++t) {
                        const int lo = (lo2 >> (2 * t)) & 0x03;
                        const int code = IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
                        const float lvl = IS_GQH3 ? gqh3_level_tbl(code, s_mag)
                                                  : gqh2h_level(code, m0);
                        acc0 += (lvl * s_b) * xs[t];
                    }
                }
                if constexpr (kRows == 2) {
                    if (valid1) {
                        const uint8_t d = d1[sb];
                        const uint64_t r64 = gqh_ld_u64(ratio1 + (sb << 3));
                        const uint8_t rb = gqh_ratio_byte(r64, sub);
                        const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
                        const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];
                        const uint16_t lo2 = gqh_ld_u16(lo1 + (sb << 6) + lane * 2);
                        const uint8_t hi1 = IS_GQH3 ? gqh_hi_byte(hi1p + (sb << 5), lane) : 0;
#pragma unroll
                        for (int t = 0; t < GQH_PER_LANE; ++t) {
                            const int lo = (lo2 >> (2 * t)) & 0x03;
                            const int code = IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
                            const float lvl = IS_GQH3 ? gqh3_level_tbl(code, s_mag)
                                                      : gqh2h_level(code, m0);
                            acc1 += (lvl * s_b) * xs[t];
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
            acc0 += gqh_warp_shfl_down(acc0, off);
            if constexpr (kRows == 2) {
                acc1 += gqh_warp_shfl_down(acc1, off);
            }
        }
        if (lane == 0) {
            const int64_t yi0 = (int64_t)col * y_col_stride + row0;
            if constexpr (kAcc) {
                y[yi0] = gqh_bf16_round(y[yi0] + gqh_bf16_round(acc0));
            } else {
                y[yi0] = acc0;
            }
            if constexpr (kRows == 2) {
                if (valid1) {
                    const int64_t yi1 = (int64_t)col * y_col_stride + row0 + 1;
                    if constexpr (kAcc) {
                        y[yi1] = gqh_bf16_round(y[yi1] + gqh_bf16_round(acc1));
                    } else {
                        y[yi1] = acc1;
                    }
                }
            }
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
    int out_b,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b,
    int fuse_swiglu) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row_base = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int row_stride = gridDim.x * warps_per_block;
    const int out_b_eff = out_b > 0 ? out_b : out;
    const int lane = threadIdx.x % GQH_WARP;

    __shared__ float s_ratio[16];
    float* s_x = reinterpret_cast<float*>(gqh_dyn_smem);
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    const bool use_lds = (in <= GQH_LDS_X_MAX && out <= GQH_LDS_OUT_MAX);
    if (use_lds) {
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            s_x[i] = x[i];
        }
    }
    __syncthreads();
    if (row_base >= out && row_base >= out_b_eff) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    int off_ra, off_loa, off_hia, stride_a;
    int off_rb, off_lob, off_hib, stride_b;
    gqh_plane_offsets(nsb, A_IS_GQH3 ? 1 : 0, &off_ra, &off_loa, &off_hia, &stride_a);
    gqh_plane_offsets(nsb, B_IS_GQH3 ? 1 : 0, &off_rb, &off_lob, &off_hib, &stride_b);
    const float* __restrict__ xsrc = use_lds ? s_x : x;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    const int row_lim = out > out_b_eff ? out : out_b_eff;
    for (int row = row_base; row < row_lim; row += row_stride) {
    const bool do_a = row < out;
    const bool do_b = row < out_b_eff;
    const uint8_t* __restrict__ row_a =
        do_a ? data_a + (int64_t)row * stride_a : data_a;
    const uint8_t* __restrict__ row_b =
        do_b ? data_b + (int64_t)row * stride_b : data_b;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const float4 x0 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 = *reinterpret_cast<const float4*>(xsrc + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};

        if (do_a) {
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
                acc_a = gqh_acc_fma(
                    acc_a, gqh_level<A_IS_GQH3>(code, mag_a), s_b, xs[t]);
            }
        }
        if (do_b) {
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
                acc_b = gqh_acc_fma(
                    acc_b, gqh_level<B_IS_GQH3>(code, mag_b), s_b, xs[t]);
            }
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc_a += gqh_warp_shfl_down(acc_a, off);
        acc_b += gqh_warp_shfl_down(acc_b, off);
    }
    if (lane == 0) {
        if (fuse_swiglu) {
            const float g = gqh_bf16_round(acc_a);
            const float u = gqh_bf16_round(acc_b);
            const float silu = g / (1.0f + expf(-g));
            y_a[row] = gqh_bf16_round(silu * u);
        } else {
            if (do_a) {
                y_a[row] = acc_a;
            }
            if (do_b) {
                y_b[row] = acc_b;
            }
        }
    }
    }
}

// Scatter one planar row into llama-style tight superblocks at the
// start of the same row. LDS holds the original planes so the write
// cannot clobber unread bytes. Product bytes are a permutation only.
template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_pack_tight_sb(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ smem,
    int sb,
    int off_r,
    int off_lo,
    int off_hi) {
    dst[0] = smem[sb];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst[1 + k] = smem[off_r + (sb << 3) + k];
    }
#pragma unroll
    for (int k = 0; k < 64; ++k) {
        dst[9 + k] = smem[off_lo + (sb << 6) + k];
    }
    if constexpr (IS_GQH3) {
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            dst[73 + k] = smem[off_hi + (sb << 5) + k];
        }
    }
}

template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_pack_padded_sb(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ smem,
    int sb,
    int off_r,
    int off_lo,
    int off_hi) {
    dst[0] = smem[sb];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst[1 + k] = smem[off_r + (sb << 3) + k];
    }
#pragma unroll
    for (int k = 0; k < 64; ++k) {
        dst[GQH_PAD_LO + k] = smem[off_lo + (sb << 6) + k];
    }
    if constexpr (IS_GQH3) {
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            dst[GQH_PAD_HI + k] = smem[off_hi + (sb << 5) + k];
        }
    }
}

template <bool IS_GQH3>
__global__ void gqh_planar_to_padded_kernel(
    const uint8_t* __restrict__ src_base,
    uint8_t* __restrict__ dst_base,
    int nsb,
    int rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    uint8_t* smem = reinterpret_cast<uint8_t*>(gqh_dyn_smem);
    const uint8_t* src = src_base + (int64_t)row * stride;
    for (int i = static_cast<int>(threadIdx.x); i < stride;
         i += static_cast<int>(blockDim.x)) {
        smem[i] = src[i];
    }
    __syncthreads();
    constexpr int pad_sb = IS_GQH3 ? GQH3_PAD_SB_BYTES : GQH2H_PAD_SB_BYTES;
    uint8_t* dst = dst_base + (int64_t)row * nsb * pad_sb;
    for (int sb = static_cast<int>(threadIdx.x); sb < nsb;
         sb += static_cast<int>(blockDim.x)) {
        gqh_pack_padded_sb<IS_GQH3>(dst + sb * pad_sb, smem, sb, off_r, off_lo, off_hi);
    }
}

template <bool IS_GQH3>
__global__ void gqh_planar_to_tight_kernel(uint8_t* data, int nsb, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    uint8_t* smem = reinterpret_cast<uint8_t*>(gqh_dyn_smem);
    uint8_t* src = data + (int64_t)row * stride;
    for (int i = static_cast<int>(threadIdx.x); i < stride; i += static_cast<int>(blockDim.x)) {
        smem[i] = src[i];
    }
    __syncthreads();
    constexpr int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    for (int sb = static_cast<int>(threadIdx.x); sb < nsb; sb += static_cast<int>(blockDim.x)) {
        gqh_pack_tight_sb<IS_GQH3>(src + sb * sb_bytes, smem, sb, off_r, off_lo, off_hi);
    }
}

// Two consecutive planar rows → interleaved tight:
//   [sb0_r0][sb0_r1][sb1_r0][sb1_r1]...
// Dual-row GEMV then streams 2*sb_bytes of adjacent weights per superblock.
template <bool IS_GQH3>
__global__ void gqh_planar_to_tight_ileave2_kernel(
    uint8_t* data, int nsb, int rows) {
    const int row0 = static_cast<int>(blockIdx.x) * 2;
    if (row0 + 1 >= rows) {
        return;
    }
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    uint8_t* smem = reinterpret_cast<uint8_t*>(gqh_dyn_smem);
    uint8_t* src0 = data + (int64_t)row0 * stride;
    uint8_t* src1 = src0 + stride;
    for (int i = static_cast<int>(threadIdx.x); i < stride;
         i += static_cast<int>(blockDim.x)) {
        smem[i] = src0[i];
        smem[stride + i] = src1[i];
    }
    __syncthreads();
    constexpr int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    for (int sb = static_cast<int>(threadIdx.x); sb < nsb;
         sb += static_cast<int>(blockDim.x)) {
        uint8_t* d0 = src0 + sb * 2 * sb_bytes;
        gqh_pack_tight_sb<IS_GQH3>(d0, smem, sb, off_r, off_lo, off_hi);
        gqh_pack_tight_sb<IS_GQH3>(
            d0 + sb_bytes, smem + stride, sb, off_r, off_lo, off_hi);
    }
}

// Tight (AoS superblock) matvec. Same product order as the planar
// kernel; each superblock is 105/73 contiguous bytes like llama.cpp.
// kRows=2: one warp does two consecutive rows and shares the x load.
// Launch only when out is a multiple of warps*2 so both rows are valid.
template <bool IS_GQH3, bool kAcc, int kRows = 1, bool kPad = false, int kNsb = 0>
__global__ void gqh_matvec_tight_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, float4 mag, int64_t x_col_stride,
    int64_t y_col_stride, int ileave, int row_off) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;
    const int warp = threadIdx.x / GQH_WARP;
    const int row0 = (blockIdx.x * warps_per_block + warp) * kRows + row_off;

    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    if (row0 >= row_off + out) {
        return;
    }

    const int nsb = kNsb > 0 ? kNsb : (in / GQH_SUPERBLOCK);
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    (void)off_r;
    (void)off_lo;
    (void)off_hi;
    constexpr int sb_bytes = kPad
        ? (IS_GQH3 ? GQH3_PAD_SB_BYTES : GQH2H_PAD_SB_BYTES)
        : (IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES);
    const int row_pitch = kPad ? (nsb * sb_bytes) : stride;
    (void)ileave;
    const int sb_pitch = sb_bytes;

    const uint8_t* __restrict__ rowbase0 =
        data + (int64_t)row0 * row_pitch;
    const uint8_t* __restrict__ rowbase1 = (kRows == 2)
        ? (data + (int64_t)(row0 + 1) * row_pitch)
        : rowbase0;
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    // llama.cpp gqh_matvec_kernel: simple SB loop, 4 waves/block.
    // The old software-pipe path was never taken at 4 waves but still
    // compiled into this kernel and inflated VGPR on gfx1100.
    for (int sb = 0; sb < nsb; ++sb) {
        const float4 x0 =
            *reinterpret_cast<const float4*>(xc + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 =
            *reinterpret_cast<const float4*>(xc + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {
            x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        gqh_tight_lane p0;
        if constexpr (kPad) {
            p0 = gqh_load_padded_lane<IS_GQH3>(
                rowbase0, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        } else {
            p0 = gqh_load_tight_lane<IS_GQH3>(
                rowbase0, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        }
        gqh_acc_tight_lane<IS_GQH3>(acc0, p0, mag, xs);
        if constexpr (kRows == 2) {
            gqh_tight_lane p1;
            if constexpr (kPad) {
                p1 = gqh_load_padded_lane<IS_GQH3>(
                    rowbase1, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
            } else {
                p1 = gqh_load_tight_lane<IS_GQH3>(
                    rowbase1, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
            }
            gqh_acc_tight_lane<IS_GQH3>(acc1, p1, mag, xs);
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc0 += gqh_warp_shfl_down(acc0, off);
        if constexpr (kRows == 2) {
            acc1 += gqh_warp_shfl_down(acc1, off);
        }
    }
    if (lane == 0) {
        const int64_t yi0 = (int64_t)col * y_col_stride + row0;
        if constexpr (kAcc) {
            y[yi0] = gqh_bf16_round(y[yi0] + gqh_bf16_round(acc0));
        } else {
            y[yi0] = acc0;
        }
        if constexpr (kRows == 2) {
            const int64_t yi1 = yi0 + 1;
            if constexpr (kAcc) {
                y[yi1] = gqh_bf16_round(y[yi1] + gqh_bf16_round(acc1));
            } else {
                y[yi1] = acc1;
            }
        }
    }
}

// kRows=2: one warp does two consecutive output rows and shares x.
// Same product order as kRows=1. Launch only when out % (warps*2) == 0.
// 2-wave blocks: pair holds 4 accs (2 tensors x kRows) so a 4-wave
// block is VGPR-heavy on gfx1100. Mix's 2-wave launch is the same trade.
template <bool A_IS_GQH3, bool B_IS_GQH3, int kRows = 1>
__global__ void gqh_matvec_pair_tight_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y_a,
    float* __restrict__ y_b,
    int in,
    int out,
    int out_b,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b,
    int fuse_swiglu,
    int ileave) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int out_b_eff = out_b > 0 ? out_b : out;
    const int lane = threadIdx.x % GQH_WARP;
    const int warp = threadIdx.x / GQH_WARP;
    const int row0 = (blockIdx.x * warps_per_block + warp) * kRows;

    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    (void)out_b_eff;
    if (row0 >= out) {
        return;
    }

    const int nsb = in / GQH_SUPERBLOCK;
    int off_ra, off_loa, off_hia, stride_a;
    int off_rb, off_lob, off_hib, stride_b;
    gqh_plane_offsets(nsb, A_IS_GQH3 ? 1 : 0, &off_ra, &off_loa, &off_hia, &stride_a);
    gqh_plane_offsets(nsb, B_IS_GQH3 ? 1 : 0, &off_rb, &off_lob, &off_hib, &stride_b);
    (void)off_ra;
    (void)off_loa;
    (void)off_hia;
    (void)off_rb;
    (void)off_lob;
    (void)off_hib;
    constexpr int sb_a = A_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    constexpr int sb_b = B_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    const int pitch_a = ileave ? sb_a * 2 : sb_a;
    const int pitch_b = ileave ? sb_b * 2 : sb_b;
    const uint8_t* __restrict__ row_a0 =
        gqh_tight_rowptr(data_a, row0, stride_a, sb_a, ileave);
    const uint8_t* __restrict__ row_b0 =
        gqh_tight_rowptr(data_b, row0, stride_b, sb_b, ileave);
    const uint8_t* __restrict__ row_a1 = (kRows == 2)
        ? (ileave ? row_a0 + sb_a
                  : data_a + (int64_t)(row0 + 1) * stride_a)
        : row_a0;
    const uint8_t* __restrict__ row_b1 = (kRows == 2)
        ? (ileave ? row_b0 + sb_b
                  : data_b + (int64_t)(row0 + 1) * stride_b)
        : row_b0;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc_a0 = 0.0f;
    float acc_b0 = 0.0f;
    float acc_a1 = 0.0f;
    float acc_b1 = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* __restrict__ ba = row_a0 + (int64_t)sb * pitch_a;
        const uint8_t* __restrict__ bb = row_b0 + (int64_t)sb * pitch_b;
        const uint8_t da = ba[0];
        const uint8_t db = bb[0];
        const float d_real_a =
            gqh_bits(GQH_E4M3_D[da >> 3][da & 7]) * scale_a;
        const float d_real_b =
            gqh_bits(GQH_E4M3_D[db >> 3][db & 7]) * scale_b;
        const uint8_t rba = ba[1 + (sub >> 1)];
        const uint8_t rbb = bb[1 + (sub >> 1)];
        const float s_ba =
            d_real_a * s_ratio[(sub & 1) ? (rba >> 4) : (rba & 0x0f)];
        const float s_bb =
            d_real_b * s_ratio[(sub & 1) ? (rbb >> 4) : (rbb & 0x0f)];
        const float4 x0 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {
            x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint16_t lo2a, lo2b;
        memcpy(&lo2a, ba + 9 + lane * 2, sizeof(lo2a));
        memcpy(&lo2b, bb + 9 + lane * 2, sizeof(lo2b));
        const uint8_t hi1a = A_IS_GQH3 ? ba[73 + lane] : 0;
        const uint8_t hi1b = B_IS_GQH3 ? bb[73 + lane] : 0;
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            const int loa = (lo2a >> (2 * t)) & 0x03;
            const int lob = (lo2b >> (2 * t)) & 0x03;
            const int ca =
                A_IS_GQH3 ? (loa | (((hi1a >> t) & 1) << 2)) : loa;
            const int cb =
                B_IS_GQH3 ? (lob | (((hi1b >> t) & 1) << 2)) : lob;
            acc_a0 = gqh_acc_fma(
                acc_a0, gqh_level<A_IS_GQH3>(ca, mag_a), s_ba, xs[t]);
            acc_b0 = gqh_acc_fma(
                acc_b0, gqh_level<B_IS_GQH3>(cb, mag_b), s_bb, xs[t]);
        }
        if constexpr (kRows == 2) {
            const gqh_tight_lane pa1 = gqh_load_tight_lane<A_IS_GQH3>(
                row_a1, sb, pitch_a, lane, sub, scale_a, s_ratio);
            const gqh_tight_lane pb1 = gqh_load_tight_lane<B_IS_GQH3>(
                row_b1, sb, pitch_b, lane, sub, scale_b, s_ratio);
            gqh_acc_tight_lane<A_IS_GQH3>(acc_a1, pa1, mag_a, xs);
            gqh_acc_tight_lane<B_IS_GQH3>(acc_b1, pb1, mag_b, xs);
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc_a0 += gqh_warp_shfl_down(acc_a0, off);
        acc_b0 += gqh_warp_shfl_down(acc_b0, off);
        if constexpr (kRows == 2) {
            acc_a1 += gqh_warp_shfl_down(acc_a1, off);
            acc_b1 += gqh_warp_shfl_down(acc_b1, off);
        }
    }
    if (lane == 0) {
        if (fuse_swiglu) {
            const float g0 = gqh_bf16_round(acc_a0);
            const float u0 = gqh_bf16_round(acc_b0);
            y_a[row0] = gqh_bf16_round((g0 / (1.0f + expf(-g0))) * u0);
            if constexpr (kRows == 2) {
                const float g1 = gqh_bf16_round(acc_a1);
                const float u1 = gqh_bf16_round(acc_b1);
                y_a[row0 + 1] = gqh_bf16_round((g1 / (1.0f + expf(-g1))) * u1);
            }
        } else {
            y_a[row0] = acc_a0;
            y_b[row0] = acc_b0;
            if constexpr (kRows == 2) {
                y_a[row0 + 1] = acc_a1;
                y_b[row0 + 1] = acc_b1;
            }
        }
    }
}

// SwiGLU pair: no ileave, no dual-row, fuse always on. Same product order
// as gqh_matvec_pair_tight_kernel<1> with fuse_swiglu=1. Fewer runtime
// branches so gfx1100 occupancy can match llama's 4-wave singles.
template <bool A_IS_GQH3, bool B_IS_GQH3, int kNsb = 0>
__global__ void gqh_matvec_pair_swiglu_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in,
    int out,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int warp = threadIdx.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + warp;

    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    if (row >= out) {
        return;
    }

    const int nsb = kNsb > 0 ? kNsb : (in / GQH_SUPERBLOCK);
    int off_ra, off_loa, off_hia, stride_a;
    int off_rb, off_lob, off_hib, stride_b;
    gqh_plane_offsets(nsb, A_IS_GQH3 ? 1 : 0, &off_ra, &off_loa, &off_hia, &stride_a);
    gqh_plane_offsets(nsb, B_IS_GQH3 ? 1 : 0, &off_rb, &off_lob, &off_hib, &stride_b);
    (void)off_ra;
    (void)off_loa;
    (void)off_hia;
    (void)off_rb;
    (void)off_lob;
    (void)off_hib;
    constexpr int sb_a = A_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    constexpr int sb_b = B_IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    const uint8_t* __restrict__ row_a = data_a + (int64_t)row * stride_a;
    const uint8_t* __restrict__ row_b = data_b + (int64_t)row * stride_b;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
#pragma unroll 2
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* __restrict__ ba = row_a + (int64_t)sb * sb_a;
        const uint8_t* __restrict__ bb = row_b + (int64_t)sb * sb_b;
        const uint8_t da = gqh_ld_nt_u8(ba);
        const uint8_t db = gqh_ld_nt_u8(bb);
        const float d_real_a =
            gqh_bits(GQH_E4M3_D[da >> 3][da & 7]) * scale_a;
        const float d_real_b =
            gqh_bits(GQH_E4M3_D[db >> 3][db & 7]) * scale_b;
        const uint8_t rba = gqh_ld_nt_u8(ba + 1 + (sub >> 1));
        const uint8_t rbb = gqh_ld_nt_u8(bb + 1 + (sub >> 1));
        const float s_ba =
            d_real_a * s_ratio[(sub & 1) ? (rba >> 4) : (rba & 0x0f)];
        const float s_bb =
            d_real_b * s_ratio[(sub & 1) ? (rbb >> 4) : (rbb & 0x0f)];
        const float4 x0 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0);
        const float4 x1 =
            *reinterpret_cast<const float4*>(x + sb * GQH_SUPERBLOCK + j0 + 4);
        const float xs[GQH_PER_LANE] = {
            x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint16_t lo2a, lo2b;
        memcpy(&lo2a, ba + 9 + lane * 2, sizeof(lo2a));
        memcpy(&lo2b, bb + 9 + lane * 2, sizeof(lo2b));
        const uint8_t hi1a = A_IS_GQH3 ? gqh_ld_nt_u8(ba + 73 + lane) : 0;
        const uint8_t hi1b = B_IS_GQH3 ? gqh_ld_nt_u8(bb + 73 + lane) : 0;
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            const int loa = (lo2a >> (2 * t)) & 0x03;
            const int lob = (lo2b >> (2 * t)) & 0x03;
            const int ca =
                A_IS_GQH3 ? (loa | (((hi1a >> t) & 1) << 2)) : loa;
            const int cb =
                B_IS_GQH3 ? (lob | (((hi1b >> t) & 1) << 2)) : lob;
            acc_a = gqh_acc_fma(
                acc_a, gqh_level<A_IS_GQH3>(ca, mag_a), s_ba, xs[t]);
            acc_b = gqh_acc_fma(
                acc_b, gqh_level<B_IS_GQH3>(cb, mag_b), s_bb, xs[t]);
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc_a += gqh_warp_shfl_down(acc_a, off);
        acc_b += gqh_warp_shfl_down(acc_b, off);
    }
    if (lane == 0) {
        const float g = gqh_bf16_round(acc_a);
        const float u = gqh_bf16_round(acc_b);
        y[row] = gqh_bf16_round((g / (1.0f + expf(-g))) * u);
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

// Dense ROCmFPX mix (qtype 105/106). One warp per row, lanes stride 32-weight
// blocks. Decode matches lucebox rocmfp{2,3}_mix: mode 0 uses fixed levels,
// mode 1 uses the 2-book sidecar LUT. Product is s * w * x folded into acc.
#define MIX_QK 32
#define MIX_FP3_QS 12
#define MIX_FP3_BYTES 14
#define MIX_FP2_QS 8
#define MIX_FP2_BYTES 10

static __device__ __forceinline__ float mix_ue4m3(uint8_t e) {
    const int exp = e >> 3;
    const int mant = e & 7;
    const float normal = ldexpf((float)(8 + mant), exp - 11);
    const float sub = (float)mant * 0.0009765625f;
    const float r = (exp == 0) ? sub : normal;
    return (e > 0x7e) ? 0.0f : r;
}

static __device__ __forceinline__ uint32_t mix_fp3_code(const uint8_t* qs, int i) {
    const int bit = i * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = qs[byte];
    if (byte + 1 < MIX_FP3_QS) {
        v |= (uint32_t)qs[byte + 1] << 8;
    }
    if (byte + 2 < MIX_FP3_QS) {
        v |= (uint32_t)qs[byte + 2] << 16;
    }
    return (v >> shift) & 7u;
}

static __device__ __forceinline__ float mix_fp3_fixed(uint32_t code) {
    const uint32_t m = code & 3u;
    const int mag = (m == 3u) ? 4 : (int)m;
    return (code & 4u) ? -(float)mag : (float)mag;
}

static __device__ __forceinline__ uint32_t mix_fp2_code(const uint8_t* qs, int i) {
    return (uint32_t)(qs[i >> 2] >> (2 * (i & 3))) & 3u;
}

static __device__ __forceinline__ float mix_fp2_fixed(uint32_t code) {
    return (float)((int)code - 1);
}

template <bool IS_FP3, bool kAcc>
__global__ void gqh_mix_matvec_kernel(
    const uint8_t* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in,
    int out,
    int mode,
    float4 lut0,
    float4 lut1,
    float4 lut2,
    float4 lut3,
    int64_t x_col_stride,
    int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;
    const int col = blockIdx.y;
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        const float src[16] = {
            lut0.x, lut0.y, lut0.z, lut0.w, lut1.x, lut1.y, lut1.z, lut1.w,
            lut2.x, lut2.y, lut2.z, lut2.w, lut3.x, lut3.y, lut3.z, lut3.w};
        s_lut[threadIdx.x] = src[threadIdx.x];
    }
    __syncthreads();
    if (row >= out) {
        return;
    }
    const int blk_bytes = IS_FP3 ? MIX_FP3_BYTES : MIX_FP2_BYTES;
    const int nsb = in / MIX_QK;
    const uint8_t* __restrict__ rowbase = data + (int64_t)row * nsb * blk_bytes;
    const float* __restrict__ xc = x + (int64_t)col * x_col_stride;
    const int klev = IS_FP3 ? 8 : 4;
    float acc = 0.0f;
    auto accum_blk = [&](int b) {
        const uint8_t* __restrict__ blk = rowbase + (int64_t)b * blk_bytes;
        const uint8_t m0 = blk[IS_FP3 ? MIX_FP3_QS : MIX_FP2_QS];
        const uint8_t m1 = blk[(IS_FP3 ? MIX_FP3_QS : MIX_FP2_QS) + 1];
        const float* xs = xc + b * MIX_QK;
        if (mode == 0) {
            const float s0 = mix_ue4m3(m0);
            const float s1 = mix_ue4m3(m1);
#pragma unroll
            for (int j = 0; j < MIX_QK; ++j) {
                const float s = (j < MIX_QK / 2) ? s0 : s1;
                const float w = IS_FP3 ? mix_fp3_fixed(mix_fp3_code(blk, j))
                                       : mix_fp2_fixed(mix_fp2_code(blk, j));
                acc += (s * w) * xs[j];
            }
        } else {
            const float s0 = mix_ue4m3(m0 & 0x7f);
            const float s1 = mix_ue4m3(m1 & 0x7f);
            const float* bk0 = s_lut + ((m0 >> 7) * klev);
            const float* bk1 = s_lut + ((m1 >> 7) * klev);
#pragma unroll
            for (int j = 0; j < MIX_QK; ++j) {
                const float s = (j < MIX_QK / 2) ? s0 : s1;
                const float* bk = (j < MIX_QK / 2) ? bk0 : bk1;
                const int code = IS_FP3 ? (int)mix_fp3_code(blk, j)
                                        : (int)mix_fp2_code(blk, j);
                acc += (s * bk[code]) * xs[j];
            }
        }
    };
    constexpr int kUnroll = 8;
    int blk = lane;
    for (; blk + (kUnroll - 1) * GQH_WARP < nsb; blk += kUnroll * GQH_WARP) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            accum_blk(blk + u * GQH_WARP);
        }
    }
    for (; blk < nsb; blk += GQH_WARP) {
        accum_blk(blk);
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    if (lane == 0) {
        const int64_t yi = (int64_t)col * y_col_stride + row;
        if constexpr (kAcc) {
            y[yi] = gqh_bf16_round(y[yi] + gqh_bf16_round(acc));
        } else {
            y[yi] = acc;
        }
    }
}
