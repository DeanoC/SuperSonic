// GQH (Geo-Quant Hierarchical) HIP decode and fused batch-1 matvec.
//
// Bit-exact against geoquant/formats/gqh.py and crates/model-store/src/gqh.rs.
// Superblock scale is d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15). Do not reassociate those products.
// Ported from geo-lucebox's low-bit GQH kernel (wave32, header as kernel args).

#include "gqh-stride.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

#include <type_traits>

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

// Flat float view of the e4m3 decode table.
//
// `GQH_E4M3_D[d >> 3][d & 7]` addresses element `(d >> 3) * 8 + (d & 7)`, which
// is `d` -- the 2-D shape is presentation only. Spelling it 2-D costs real work
// in the superblock loop on gfx1201, though: LLVM rebuilds the address as
// `base + ((d * 4) & 0x3e0) + ((d & 7) * 4)`, i.e. 4 VALU of index math plus
// *two* 64-bit `v_add_co_u32`/`v_add_co_ci_u32` pairs (each needing an
// `s_wait_alu depctr_va_sdst(0)`), and it re-materialises `base` from the GOT
// every loop body (`s_getpc_b64` / `s_add_co_u32` x2 / `s_load_b64` / a full
// `s_wait_kmcnt 0x0` scalar-memory stall).
//
// Flat `tbl[d]` is one `v_lshlrev_b32` feeding the `global_load_b32 vdst, voff,
// s[base]` saddr form: no 64-bit add, no GOT, no kmcnt wait. Same array
// element, same raw bits (the table stores float32 bit patterns), so this is
// bit-exact by construction -- not a numeric change of any kind.
//
// Do NOT reach for an LDS table here: a 256-entry `d_real` table in LDS cuts
// more instructions but measured *slower* end to end (see the handoff's
// rejected-experiment section). `s_b` gates the whole superblock's FMA chain,
// and the global load prefetches into a clause many superblocks ahead whereas
// an LDS load contends with the 8 `s_grid` reads per superblock for dscnt.
static __device__ __forceinline__ float gqh_e4m3_dreal(uint8_t d) {
    return reinterpret_cast<const float*>(&GQH_E4M3_D[0][0])[d];
}
static __constant__ uint32_t GQH_RATIO_Q_D[16][1] = GQH_RATIO_Q_INIT;
static __constant__ uint32_t GQH2C_CB_D[256][8] = GQH2C_CODEBOOK_INIT;
static __constant__ uint8_t GQH2C_SIGN_D[128] = GQH2C_SIGN_MASK_INIT;

struct gqh_grid16 {
    float v[16];
};
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
    const float d_real = gqh_e4m3_dreal(d) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

#ifndef GQH_ALLOW_FMA
#pragma clang fp contract(off)
#endif
static __device__ __forceinline__ float gqh_scale_from_planes(
    uint8_t d, const uint8_t* __restrict__ ratio8, int sub, float tensor_scale) {
    const float d_real = gqh_e4m3_dreal(d) * tensor_scale;
    const uint8_t rb = ratio8[sub >> 1];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

// Planes are 8/64/32-byte aligned. Typed loads so the compiler emits
// ushort/dword/dwordx2 instead of byte gathers + memcpy.
static __device__ __forceinline__ uint16_t gqh_ld_u16(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

// Superblock-stream S_PREFETCH_DATA on the GQH3 rung only.
//
// READ THE MECHANISM NOTE BEFORE TUNING THIS. Only about half the win is
// prefetching, and on the pair arms essentially none of it is.
//
// The control is a *backward* target -- prefetch a superblock this wave has
// already consumed (`sb - 8`), so the instruction, the gate and the loop shape
// are unchanged but there is provably zero speculative lookahead. iteration 13, drv32,
// order-balanced B,C,C,B, 3 passes, distance 8:
//     arm                  ahead    backward control   prefetching buys
//     3,acc,68  o=5120     +0.69%   +0.13%             ~80% of it
//     3,_,20    o=10240    +1.23%   +0.60%             ~50%
//     3,acc,24  o=5120     +0.89%   +0.62%             ~30%
//     pss<2h,3> o=17408    +1.64%   +1.41%             ~15%
//     ps<3,3>   o=17408    +1.35%   +1.79%             none -- control wins
// So: on the tight singles roughly half the gain is genuine latency hiding and
// half is codegen; on pair_swiglu it is codegen outright. (An earlier control
// that targeted the *current* superblock was too weak to show this: 4 lines is
// 512 B, which at sb_pitch 105 spans sb..sb+4 while the unrolled body only
// consumes sb..sb+3, so it still prefetched about a body ahead. Use the
// backward target.)
//
// The codegen half is this: forming the SGPR address materialises the row base
// on the scalar unit and the loop's load clauses consolidate -- `s_clause` runs
// 8 -> 6 on tight<3,acc,68> -- the same correlate iteration 8 credited for its
// +1.75%. Per 4-superblock body on that arm: v_readfirstlane 4 -> 6,
// s_add_nc_u64 2 -> 1, global_load_u8 unchanged at 8, s_wait_kmcnt unchanged at
// 4. That last number is why the singles win here and lost in iteration 9: a
// prefetch returns no data, so unlike iteration 9's scalarised header chain it
// adds no kmcnt wait to the FMA chain.
//
// Consequences, and they are the reason this is written down at length:
//  * Distance and lines are only *partly* a prefetch-depth knob, so tuning them
//    on a latency model will mislead. Distances 4/8/12/16 and line counts
//    2/3/4/6 all measure positive on every GQH3 arm, and on nsb=20 arms
//    distance 16 fires at most one prefetch per row yet still reads +1.9/+2.8%
//    on the pair arms -- that part cannot be latency hiding.
//  * The neighbourhood is a knife edge, exactly as this file's iteration 4-6
//    record says. The current-sb control at distance 8 reads -13.4% on
//    `3,_,20 o=10240` and -33.9% on `o=6144`. Re-measure all seven decode arms
//    after any change to this block -- and after any hipcc / ROCm bump, since
//    half the effect is codegen. Landed on HIP 7.14.60850 / AMD clang 23.0.0git
//    (ROCm llvm-project 46fcb339). Keep the GQH2H arms in the table: they are
//    instruction-identical here and are the controls.
//  * On pair_swiglu this *reverts* iteration 9's scalarised `d`: s_load_u8
//    8 -> 0, global_load_u8 8 -> 16. It is still +1.35..1.79% there, but that
//    means the two changes interact and iteration 9's numbers no longer
//    describe the shipped pair body.
//
// Gated to GQH3: the GQH2H arms lose -2.2% to -6.8% at every distance and line
// count tried, so the split is the rung and not over-prefetch. 2H carries
// 45-55% exposed compute against GQH3's ~24% (iteration 5's add-work probe) in
// a 40% smaller body, so the same three instructions perturb it much harder --
// the same asymmetry gqh_ratio_nibble records in the other direction.
//
// Bit-exact by construction: a prefetch moves no data into any register.
//
// S_PREFETCH_DATA is a gfx12 instruction and this bridge builds a fat
// gfx1100 + gfx1201 binary, so the body has to be arch-guarded the same way
// full_attention_4b.hip guards its WMMA paths. On gfx1100 the whole helper
// compiles away and every kernel there is byte-identical to iteration 12.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define GQH_HAS_PREFETCH 1
#else
#define GQH_HAS_PREFETCH 0
#endif
#define GQH_PF_SB    8
#define GQH_PF_LINES 4
template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_prefetch_sb(
    const uint8_t* __restrict__ p, int sb, int nsb, int sb_pitch) {
#if GQH_HAS_PREFETCH
    if constexpr (IS_GQH3) {
        // Keep the whole prefetched span inside the row: the last row's end is
        // the end of the tensor allocation, and a 128*GQH_PF_LINES-byte span
        // starting at the last superblock would run past it.
        const int span = (128 * GQH_PF_LINES + sb_pitch - 1) / sb_pitch;
        if ((sb & 3) == 0 && sb + GQH_PF_SB + span <= nsb) {
            __builtin_amdgcn_s_prefetch_data(
                p + (int64_t)(sb + GQH_PF_SB) * sb_pitch, GQH_PF_LINES);
        }
    } else {
        (void)p; (void)sb; (void)nsb; (void)sb_pitch;
    }
#else
    (void)p; (void)sb; (void)nsb; (void)sb_pitch;
#endif
}

// Superblock header byte (d, ratio). A plain CACHED load, on purpose.
//
// This used to be `__builtin_nontemporal_load` -- "streaming weight loads so x
// stays in L2". That reasoning does not survive the tight layout: d, ratio and
// the code plane are all inside the *same* 105/73-byte superblock, so they
// share cache lines, and the code plane is read as 32 overlapping 4-byte
// windows at stride 3 -- every 64 B line is requested by ~21 lanes of one
// instruction. Tagging the header bytes NT evicts lines the code plane is about
// to hit, so the two policies fight over the same line.
//
// gfx1201 microbench (warps=4, kRows=1, A/A columns, two runs agreeing to
// 0.3%), dropping NT from the header bytes:
//   pair_swiglu<3,3,20>   o=17408 +8.8%   tight<3,_,20>    o=17408 +3.0%
//   pair_swiglu<2h,2h,20> o=17408 +7.2%   tight<2h,_,20>   o=17408 +2.8%
//   tight<2h,_,20>        o=12288 +2.9%   tight<2h,acc,68> o=5120  +3.3%
//   tight<2h,_,20>       lm_head  +2.6%   tight<3,acc,68>  o=5120  +1.9%
//   tight<3,_,20>         o=10240 +1.3%   tight<3,acc,24>  o=5120  +0.9%
// Dropping NT from only d, or only ratio, wins about half as much each -- a
// dose-response, which is why the shared-line story is the one to believe.
// Lone exception: tight<3,_,20> o=6144 reads -13.6% at warps=4 yet +0.1% at
// warps=8, while o=5120/7168/8192 are flat -- a launch-geometry resonance in
// the serialised microbench, not a cache effect (that arm is already known to
// react -15..-45% to *any* change).
//
// Do NOT put NT back without re-running that sweep. Note this is bit-exact by
// construction -- a cache policy hint cannot change the value loaded -- and the
// ISA is otherwise byte-identical: same 2776 lines, same instruction
// histogram, same registers, same s_clause runs, only the th:TH_LOAD_NT bits
// gone. So it cannot perturb the knife-edge schedule either.
static __device__ __forceinline__ uint8_t gqh_ld_sb_u8(const uint8_t* p) {
    return *p;
}
static __device__ __forceinline__ uint32_t gqh_ld_u32(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
static __device__ __forceinline__ uint64_t gqh_ld_u64(const uint8_t* p) {
    return *reinterpret_cast<const uint64_t*>(p);
}

// Wave-uniform superblock scale: d, and therefore d_real, is the same in every
// lane of the wave. `row = blockIdx.x * warps_per_block + warp` with
// `warp = threadIdx.x / 32` is constant across a wave32 (which this file
// already assumes -- the reduction is `__shfl_down(v, off, GQH_WARP)`), and the
// only guard between the row computation and the loop
// (`if (row >= out) return;`) is uniform too, so exec is full here. Every
// caller of gqh_load_tight_lane / of the pair bodies derives its row base that
// way: gqh_matvec_tight_kernel (both kRows), gqh_tight_row_dot (via
// gqh_matvec_ab_tight_kernel, whose `row < out_a` branch is uniform),
// gqh_matvec_pair_tight_kernel and gqh_matvec_pair_swiglu_kernel.
//
// Saying so with readfirstlane hands the whole d_real chain to the scalar unit:
// the e4m3 table lookup becomes `s_load_b32` instead of
// `v_lshlrev_b32` + `global_load_b32`, and `* tensor_scale` becomes
// `s_mul_f32` (gfx1201 has SALU float). Per 4-superblock body on
// tight<3,_,20>: `global_load_b32` 9 -> 5, `v_lshlrev_b32` 10 -> 5,
// `s_load_b32` 2 -> 6, `s_mul_f32` 0 -> 4, `v_readfirstlane_b32` 0 -> 4.
// The three load clauses the loop depends on (`s_clause 0x3` / `0x5` / `0x3`)
// survive unchanged; on pair_swiglu the clauses *consolidate* (10 runs -> 3 on
// <3,3>, 13 -> 9 on <2h,3>) and dual-issue rises (`v_dual_fmac_f32` 15 -> 26 on
// <2h,3>). VGPRs 78 -> 80 / 73 -> 77 on the singles, unchanged at 95 on the
// pairs; allocation granularity means occupancy does not move.
//
// gfx1201 microbench, shipping geometry, A/A columns, two runs agreeing to
// 0.01 ms/tok (GB/s, base vs scalar-d):
//   3,acc,68  o=5120  +2.5..2.7%   ps<3,3>   o=17408 +2.5..2.6%
//   pss<2h,3> o=17408 +3.3..3.6%   3,_,20    o=10240 +3.2..3.3%
//   3,_,20    o=6144  +8.4..8.8%   3,acc,24  o=5120  +2.1..2.3%
//   2h,_,20   o=12288 +3.2..3.6%   2h,_,20   lm_head +1.7%
//   2h,acc,68 o=5120  -3.5..-4.2%  <- the only loss, 0.45 ms/tok, +0.02 ms
// Weighted over the nine decode arms: -0.55 ms/tok, +2.05% of hot_ms.
//
// Bit-exact by construction (same table element, same product order) and
// measured: 16/16 arm x rung combinations compare 0 raw-uint32 mismatches
// against the shipping kernels over random weights *and* random x, with a
// nonzero-output guard so the check cannot pass vacuously.
//
// If d ever stops being wave-uniform -- a kRows variant that gives lanes
// different rows, or a wave64 build -- this silently broadcasts lane 0's scale
// to the whole wave. Re-run the bitwise gate before changing the row mapping.
static __device__ __forceinline__ float gqh_dreal_uniform(uint8_t d, float ts) {
    const int du = __builtin_amdgcn_readfirstlane((int)d);
    return gqh_e4m3_dreal((uint8_t)du) * ts;
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

// Tight 137 B superblock: [0] E4M3 d, [1:9] ratios, [9:137] uint4 codes.
// 16-level grid in LDS -- same reason as llama.cpp: divergent code indexes.
template <typename T>
__global__ void gqh4_decode_kernel(
    const uint8_t* __restrict__ wire, float tensor_scale, gqh_grid16 grid,
    T* __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;
    const uint8_t* __restrict__ b = wire + sb * GQH4_SB_BYTES;
    __shared__ float s_grid[16];
    if (j < 16) {
        s_grid[j] = grid.v[j];
    }
    __syncthreads();
    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);
    const uint8_t cb = b[9 + (j >> 1)];
    const int code = (j & 1) ? (cb >> 4) : (cb & 0x0f);
    dst[sb * GQH_SUPERBLOCK + j] = gqh_store_val<T>(s_grid[code] * s_b);
}

#define GQH_WARP 32
#define GQH_MATVEC_WARPS 16
#define GQH_PER_LANE (GQH_SUPERBLOCK / GQH_WARP)

// Wave-uniform warp index -- **pair kernels only**, and that restriction is
// measured, not stylistic.
//
// `threadIdx.x` is lane-linear inside a wave32, so `threadIdx.x / 32` is
// identical in every lane; this file already rests on that (the reduction is
// `__shfl_down(v, off, GQH_WARP)`). LLVM's divergence analysis cannot see it,
// so `row` -- and with it `rowbase` -- is treated as divergent and every
// superblock-header load off it is pinned to the vector path. Iteration 8
// scalarised the *e4m3 table* lookup by readfirstlane-ing the loaded value;
// this scalarises the `d` byte load itself, plus the 64-bit row address math.
//
// Per 4-superblock body of gqh_matvec_pair_swiglu_kernel<3,3,20>:
// `global_load_u8` 16 -> 8, `s_load_u8` 0 -> 8, `v_readfirstlane_b32` 8 -> 1,
// `v_add_co_u32` 18 -> 12, `v_add_co_ci_u32_e64` 19 -> 13, total lines
// 719 -> 688, VGPRs 95 -> 93. The scalar loads cost one extra `s_wait_kmcnt`
// in the loop, and that is exactly why this is gated: the pair body has two
// independent tensors' worth of VMEM in flight per superblock to hide it
// behind, and the singles have nothing.
//
// gfx1201 microbench, shipping geometry, A/A columns, two runs:
//   ps<3,3>   o=17408 **+2.9 / +3.4%**   pss<2h,3> o=17408 **+2.2 / +2.0%**
//   3,acc,68  o=5120      -15.0 / -14.9%   3,_,20   o=10240   -10.2 / -10.2%
//   3,acc,24  o=5120       -7.6 /  -7.9%   2h,_,20  o=12288    -9.3 /  -9.3%
//   2h,acc,68 o=5120       -9.2 /  -8.5%   2h,_,20 lm_head     -9.3 /  -9.2%
// Applied to every kernel it is **-6.5% of hot_ms**; applied to the two pair
// arms alone (6.75 of 26.54 ms/tok) it is +0.64..0.68%. Do NOT hand it to
// gqh_matvec_tight_kernel -- the `s_wait_kmcnt` stall the GQH_E4M3_D comment
// above warns about is real and it dominates there.
//
// Bit-exact by construction: readfirstlane of an already-wave-uniform value is
// the identity, and no product changes. Exec is full at the call site (it
// precedes the `threadIdx.x < 16` LDS fill), and the intrinsic is convergent
// so it cannot be sunk into a narrower mask. If a future kRows variant hands
// lanes different rows, this and gqh_dreal_uniform both break -- re-run the
// bitwise gate before changing the row mapping.
static __device__ __forceinline__ int gqh_warp_id() {
    return __builtin_amdgcn_readfirstlane((int)(threadIdx.x / GQH_WARP));
}
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

// Reference tree. Kept for the planar / gqh2c paths and, in the tight
// GEMVs, only to *fill* the LDS level table (gqh_fill_level_lds) once per
// block -- indexing a float4 with a VGPR costs ~5 cndmask + 3 cmp per
// weight, which measured 15.6 VALU/weight on gfx1201.
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
    return gqh2h_level(code, m.x);
}

// 8-entry level table in LDS, filled from gqh_level itself so the entries are
// bit-identical to the tree by construction (GQH3 grid is
// [-m3,-m2,-m1,-m0, m0,m1,m2,m3]; GQH2H only uses codes 0..3 but all 8 are
// filled so an out-of-range index can never read garbage). The tight GEMVs
// then spend one ds_load_b32 per weight instead of the select tree -- the
// same trade gqh4_matvec_tight_kernel already makes with its s_grid[16].
struct gqh_lvl_grid {
    float grid[8];
};

// GQH2H only: `pair2` is the same LDS-table trade taken one level further.
//
// The 2H code plane is already lane-contiguous with 8 two-bit codes at bit 2t,
// so *two* consecutive codes are one 4-bit field at bit 4j -- and a 16-entry
// table of level *pairs* turns one bitfield extract + one address shift + one
// ds_load_b32 per weight into one extract + one shift + one ds_load_b64 per
// pair. Measured on the shipping `2h,_,1,_,20,1` arm, per 4-superblock body:
// VALU 114 -> 88 (-23%), LDS 36 -> 20 (-44%), VGPR 77 -> 69, and VMEM 20 /
// SMEM 4 / the six s_clause runs (0x3 0x5 0x3 0x1 0x3 0x3) all *unchanged* --
// the memory side is untouched, byte for byte.
//
// That is what makes this different from every instruction-count change on
// iterations 4-6: it strictly *reduces* both issue ports instead of trading
// between them (FINDING 3's "neither port binds" is about trades), and it
// *removes* LDS ops from the FMA chain rather than adding one to it, which is
// what killed every FINDING 5 variant. Size of the prize, calibrated on the
// same rung: iteration 5 cut 2H VALU 3.6% and measured +0.8..1.8%. 2H arms are
// 3.79 ms/tok of decode and carry 45-55% exposed compute (iteration 5's
// add-work probe), the highest in the file.
//
// 16 slots x 8 B = 128 B at an 8-byte stride, so the slots cover all 32 LDS
// banks exactly once and the b64 read cannot bank-conflict whatever the codes
// are. GQH3 deliberately keeps the plain 8-entry `grid`: its 3-bit codes make a
// pair a 6-bit field, i.e. a 64-slot / 512 B table whose slots alias 4-to-1
// onto bank pairs, which is a data-dependent conflict risk on the rung that has
// lost on every schedule perturbation tried (see gqh_ratio_nibble).
//
// pair2[c] holds exactly (level(c & 3), level((c >> 2) & 3)) ==
// (level(code 2j), level(code 2j+1)) with c the 4-bit field, filled from
// gqh_level itself -- so every entry is bit-identical to the select tree by
// construction, and the inner loop keeps the same products in the same order.
// All 16 slots are reachable from a 4-bit field and all 16 are filled, so an
// out-of-range index cannot read garbage.
struct alignas(8) gqh_lvl_pair_tbl : gqh_lvl_grid {
    float pair2[32];
};

// GQH3 has no pair table, so the kernels that never take the paired path keep
// a plain 32-byte `gqh_lvl_grid` and their LDS layout -- and hence their ISA --
// byte-identical to before this change.
template <bool IS_GQH3>
using gqh_lvl_tbl =
    std::conditional_t<IS_GQH3, gqh_lvl_grid, gqh_lvl_pair_tbl>;

template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_fill_level_lds(
    gqh_lvl_grid* __restrict__ tbl, const float4& m, int tid) {
    if (tid < 8) {
        tbl->grid[tid] = gqh_level<IS_GQH3>(tid, m);
    }
}

static __device__ __forceinline__ void gqh_fill_pair_lds(
    gqh_lvl_pair_tbl* __restrict__ tbl, const float4& m, int tid) {
    if (tid < 16) {
        tbl->pair2[2 * tid + 0] = gqh_level<false>(tid & 3, m);
        tbl->pair2[2 * tid + 1] = gqh_level<false>((tid >> 2) & 3, m);
    }
}

// Both levels of one code pair (codes 2j, 2j+1) in one ds_load_b64. GQH2H only.
static __device__ __forceinline__ float2 gqh_lvl_pair(
    const float* __restrict__ pair2, uint32_t cw, int j) {
    return *reinterpret_cast<const float2*>(
        pair2 + 2 * (int)((cw >> (4 * j)) & 15u));
}

// llama.cpp gqh.cu does acc += (lvl * s_b) * x with FMA on the add.
// Scale stays two muls: e4m3(d)*tensor_scale, then *(ratio/15).
static __device__ __forceinline__ float gqh_acc_fma(
    float acc, float lvl, float s_b, float x) {
    return fmaf(lvl * s_b, x, acc);
}

// GQH3 tight code plane: 3-bit stride, lane-contiguous.
//
// The two-plane layout (64 B lo at +9, 32 B hi at +73) costs 4 VALU per weight
// just to rebuild `code` -- shift lo2, mask, shift hi1, mask, OR -- because the
// two bits and the third bit live in different registers at different strides.
// Packing each lane's 8 codes into 24 contiguous bits costs the *same* 96 B per
// superblock (32 lanes x 3 B), so DRAM traffic is unchanged, and the inner loop
// becomes one shift + one mask-or per weight.
//
// Lane L owns bytes [GQH_CODES_OFF + 3L, +3). The GEMV loads the dword one
// byte early (at GQH_CODES_OFF - 1 + 3L) so lane 31's window ends exactly at
// the superblock boundary -- 8 + 3*31 + 4 == 105 == GQH3_SB_BYTES, no
// over-read on any lane -- which puts code t at bit (8 + 3t).
//
// Bit budget has zero slack: 8 + 3*7 + 3 == 32. An off-by-one in the packer
// silently corrupts code 7 of every lane instead of failing loudly, so the
// pack and the extract are both derived from GQH3_CODE_BIT0/GQH3_CODE_STRIDE.
#define GQH_CODES_OFF     9
// Bytes the codeword load starts before the code plane. The extract's bit
// origin is derived from it, so the load offset and the shift cannot drift.
#define GQH_CODE_PREROLL  1
#define GQH3_CODE_BIT0   (8 * GQH_CODE_PREROLL)
#define GQH3_CODE_STRIDE  3
static_assert(
    GQH3_CODE_BIT0 + GQH3_CODE_STRIDE * (GQH_PER_LANE - 1) + 3 <= 32,
    "GQH3 lane codeword must fit in 32 bits");
static_assert(
    GQH_CODES_OFF - GQH_CODE_PREROLL + GQH3_CODE_STRIDE * (GQH_WARP - 1) + 4 <=
        GQH3_SB_BYTES,
    "GQH3 codeword load must not read past the superblock");
static_assert(
    GQH_CODES_OFF + GQH3_CODE_STRIDE * GQH_WARP <= GQH3_SB_BYTES,
    "GQH3 code plane must fit in the superblock");
static_assert(
    GQH_PAD_LO - GQH_CODE_PREROLL + GQH3_CODE_STRIDE * (GQH_WARP - 1) + 4 <=
        GQH3_PAD_SB_BYTES,
    "GQH3 padded codeword load must not read past the padded superblock");

// Tight AoS lane payload. Scale products stay two muls.
// `cw` is the lane's packed code word: GQH3 keeps 8 3-bit codes at bit
// (GQH3_CODE_BIT0 + 3t); GQH2H keeps its 8 2-bit codes at bit 2t (that plane is
// already lane-contiguous, so it stays a plain 16-bit load).
struct gqh_tight_lane {
    float s_b;
    uint32_t cw;
};

// One code out of the packed word. Both shifts are compile-time constants, so
// this is a single bitfield extract.
template <bool IS_GQH3>
static __device__ __forceinline__ int gqh_code_at(uint32_t cw, int t) {
    if constexpr (IS_GQH3) {
        return (int)((cw >> (GQH3_CODE_BIT0 + GQH3_CODE_STRIDE * t)) & 7u);
    }
    return (int)((cw >> (2 * t)) & 3u);
}

// Codeword load. GQH3 is one align-1 dword; GQH2H keeps its u16 lo plane.
template <bool IS_GQH3>
static __device__ __forceinline__ uint32_t gqh_load_codeword(
    const uint8_t* __restrict__ b, int lane, int codes_off) {
    if constexpr (IS_GQH3) {
        uint32_t w;
        memcpy(
            &w, b + codes_off - GQH_CODE_PREROLL + lane * GQH3_CODE_STRIDE,
            sizeof(w));
        return w;
    }
    uint16_t lo2;
    memcpy(&lo2, b + codes_off + lane * 2, sizeof(lo2));
    return lo2;
}

static __device__ __forceinline__ const uint8_t* gqh_tight_rowptr(
    const uint8_t* data, int row, int stride, int sb_bytes, int ileave) {
    if (ileave) {
        return data + (int64_t)(row & ~1) * stride + (row & 1) * sb_bytes;
    }
    return data + (int64_t)row * stride;
}

// Ratio nibble for this lane's sub-block. `sub` is loop-invariant, so on the
// GQH2H arms the select can be one variable shift + mask, dropping the
// per-superblock v_cndmask: 137 -> 132 VALU lines per 4-superblock body, worth
// +0.8% (o=17408), +1.1% (lm_head), +1.8% (acc,68) and -0.1% (o=12288) on the
// gfx1201 microbench -- ~+0.8% weighted over the 2H arms.
//
// GQH3 keeps the cndmask spelling on purpose. The variable-shift form there is
// *also* strictly better ISA (152 -> 138 lines, 12 -> 18 VOPD pairs, same LDS,
// same 24 global loads, no spills) and measured **-11 to -44%** on every GQH3
// arm. GQH3 sits on a knife-edge schedule: adding work (-20%), removing work
// (-25%), reordering it (sched_group_barrier, -10..-47%), changing the unroll
// (-5..-58%) and throttling occupancy (-2%) are all losses. Do not "clean this
// up" into one spelling without re-measuring both rungs.
static __device__ __forceinline__ int gqh_ratio_nibble(uint8_t rb, int sub, int is3) {
    if (is3) {
        return (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    }
    return (rb >> ((sub & 1) << 2)) & 0x0f;
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
    const uint8_t d = gqh_ld_sb_u8(b);
    const float d_real = gqh_dreal_uniform(d, tensor_scale);
    const uint8_t rb = gqh_ld_sb_u8(b + 1 + (sub >> 1));
    gqh_tight_lane p;
    p.s_b = d_real * s_ratio[gqh_ratio_nibble(rb, sub, IS_GQH3 ? 1 : 0)];
    p.cw = gqh_load_codeword<IS_GQH3>(b, lane, GQH_CODES_OFF);
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
    const float d_real = gqh_e4m3_dreal(d) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    gqh_tight_lane p;
    p.s_b = d_real * s_ratio[gqh_ratio_nibble(rb, sub, IS_GQH3 ? 1 : 0)];
    p.cw = gqh_load_codeword<IS_GQH3>(b, lane, GQH_PAD_LO);
    return p;
}

template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_acc_tight_lane(
    float& acc, const gqh_tight_lane& p, const gqh_lvl_tbl<IS_GQH3>& tbl,
    const float xs[8]) {
    if constexpr (IS_GQH3) {
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            acc = gqh_acc_fma(
                acc, tbl.grid[gqh_code_at<true>(p.cw, t)], p.s_b, xs[t]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < GQH_PER_LANE / 2; ++j) {
            const float2 lv = gqh_lvl_pair(tbl.pair2, p.cw, j);
            acc = gqh_acc_fma(acc, lv.x, p.s_b, xs[2 * j]);
            acc = gqh_acc_fma(acc, lv.y, p.s_b, xs[2 * j + 1]);
        }
    }
}

// Skinny B>1: dequant once (w[t] = lvl * s_b), then fmaf per column. Same
// product as gqh_acc_fma so verify stays bit-identical to kCols=1. llama.cpp
// gqh_matvec_kernel does this reuse; we previously re-dequantized per column.
template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_lane_weights(
    float w[GQH_PER_LANE], const gqh_tight_lane& p,
    const gqh_lvl_tbl<IS_GQH3>& tbl) {
    if constexpr (IS_GQH3) {
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            w[t] = tbl.grid[gqh_code_at<true>(p.cw, t)] * p.s_b;
        }
    } else {
#pragma unroll
        for (int j = 0; j < GQH_PER_LANE / 2; ++j) {
            const float2 lv = gqh_lvl_pair(tbl.pair2, p.cw, j);
            w[2 * j + 0] = lv.x * p.s_b;
            w[2 * j + 1] = lv.y * p.s_b;
        }
    }
}

static __device__ __forceinline__ void gqh_fma_weights(
    float& acc, const float w[GQH_PER_LANE], const float xs[GQH_PER_LANE]) {
#pragma unroll
    for (int t = 0; t < GQH_PER_LANE; ++t) {
        acc = fmaf(w[t], xs[t], acc);
    }
}

static __device__ __forceinline__ void gqh_load_lane_xs(
    float xs[GQH_PER_LANE],
    const float* __restrict__ x,
    int64_t col,
    int64_t x_col_stride,
    int sb,
    int j0) {
    const float* __restrict__ xc =
        x + col * x_col_stride + sb * GQH_SUPERBLOCK + j0;
    const float4 x0 = *reinterpret_cast<const float4*>(xc);
    const float4 x1 = *reinterpret_cast<const float4*>(xc + 4);
    xs[0] = x0.x;
    xs[1] = x0.y;
    xs[2] = x0.z;
    xs[3] = x0.w;
    xs[4] = x1.x;
    xs[5] = x1.y;
    xs[6] = x1.z;
    xs[7] = x1.w;
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
    const gqh_lvl_tbl<IS_GQH3>& tbl,
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
        gqh_acc_tight_lane<IS_GQH3>(acc, p, tbl, xs);
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
    __shared__ gqh_lvl_tbl<A_IS_GQH3> s_lvl_a;
    __shared__ gqh_lvl_tbl<B_IS_GQH3> s_lvl_b;
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        gqh_fill_level_lds<A_IS_GQH3>(&s_lvl_a, mag_a, threadIdx.x);
        gqh_fill_level_lds<B_IS_GQH3>(&s_lvl_b, mag_b, threadIdx.x);
        if constexpr (!A_IS_GQH3) {
            gqh_fill_pair_lds(&s_lvl_a, mag_a, threadIdx.x);
        }
        if constexpr (!B_IS_GQH3) {
            gqh_fill_pair_lds(&s_lvl_b, mag_b, threadIdx.x);
        }
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
            s_lvl_a, s_ratio);
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
            s_lvl_b, s_ratio);
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
    const float d_real = gqh_e4m3_dreal(p.d) * tensor_scale;
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
                    const float d_real = gqh_e4m3_dreal(d) * tensor_scale;
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
                        const float d_real = gqh_e4m3_dreal(d) * tensor_scale;
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
            const float d_real = gqh_e4m3_dreal(d) * scale_a;
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
            const float d_real = gqh_e4m3_dreal(d) * scale_b;
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

// Scatter one superblock's planar code planes into the lane-contiguous code
// plane the GEMV reads back with gqh_load_codeword / gqh_code_at. GQH3 folds
// the 2-bit lo plane and the 1-bit hi plane into 8 3-bit codes per lane, 24
// bits at byte codes_off + 3*lane -- same 96 B the two planes already cost.
// GQH2H's 2-bit plane is already lane-contiguous, so it is copied verbatim.
// Both packers go through here so tight and padded cannot drift apart.
template <bool IS_GQH3>
static __device__ __forceinline__ void gqh_pack_code_plane(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ smem,
    int sb,
    int off_lo,
    int off_hi,
    int codes_off) {
    if constexpr (IS_GQH3) {
        const uint8_t* __restrict__ lo = smem + off_lo + (sb << 6);
        const uint8_t* __restrict__ hi = smem + off_hi + (sb << 5);
        for (int lane = 0; lane < GQH_WARP; ++lane) {
            const uint32_t lo2 = (uint32_t)lo[lane * 2] |
                ((uint32_t)lo[lane * 2 + 1] << 8);
            const uint32_t hi1 = hi[lane];
            uint32_t w = 0;
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const uint32_t code =
                    ((lo2 >> (2 * t)) & 3u) | (((hi1 >> t) & 1u) << 2);
                w |= code << (GQH3_CODE_STRIDE * t);
            }
#pragma unroll
            for (int k = 0; k < GQH3_CODE_STRIDE; ++k) {
                dst[codes_off + lane * GQH3_CODE_STRIDE + k] =
                    (uint8_t)(w >> (8 * k));
            }
        }
    } else {
        (void)off_hi;
#pragma unroll
        for (int k = 0; k < 64; ++k) {
            dst[codes_off + k] = smem[off_lo + (sb << 6) + k];
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
    gqh_pack_code_plane<IS_GQH3>(dst, smem, sb, off_lo, off_hi, GQH_CODES_OFF);
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
    gqh_pack_code_plane<IS_GQH3>(dst, smem, sb, off_lo, off_hi, GQH_PAD_LO);
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

template <bool IS_GQH3>
__global__ void gqh_tight_to_planar_kernel(uint8_t* data, int nsb, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    int off_r, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &stride);
    constexpr int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    uint8_t* smem = reinterpret_cast<uint8_t*>(gqh_dyn_smem);
    uint8_t* row_data = data + (int64_t)row * stride;
    const int tight_bytes = nsb * sb_bytes;
    for (int i = static_cast<int>(threadIdx.x); i < tight_bytes;
         i += static_cast<int>(blockDim.x)) {
        smem[i] = row_data[i];
    }
    __syncthreads();
    for (int sb = static_cast<int>(threadIdx.x); sb < nsb;
         sb += static_cast<int>(blockDim.x)) {
        const uint8_t* src = smem + sb * sb_bytes;
        row_data[sb] = src[0];
        for (int k = 0; k < 8; ++k) {
            row_data[off_r + (sb << 3) + k] = src[1 + k];
        }
        if constexpr (IS_GQH3) {
            for (int lane = 0; lane < GQH_WARP; ++lane) {
                const uint8_t* codes = src + GQH_CODES_OFF +
                    lane * GQH3_CODE_STRIDE;
                const uint32_t word = static_cast<uint32_t>(codes[0]) |
                    (static_cast<uint32_t>(codes[1]) << 8) |
                    (static_cast<uint32_t>(codes[2]) << 16);
                uint16_t lo = 0;
                uint8_t hi = 0;
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    const uint32_t code = (word >> (GQH3_CODE_STRIDE * t)) & 7u;
                    lo |= static_cast<uint16_t>(code & 3u) << (2 * t);
                    hi |= static_cast<uint8_t>((code >> 2) & 1u) << t;
                }
                row_data[off_lo + (sb << 6) + lane * 2] =
                    static_cast<uint8_t>(lo);
                row_data[off_lo + (sb << 6) + lane * 2 + 1] =
                    static_cast<uint8_t>(lo >> 8);
                row_data[off_hi + (sb << 5) + lane] = hi;
            }
        } else {
            for (int k = 0; k < 64; ++k) {
                row_data[off_lo + (sb << 6) + k] = src[GQH_CODES_OFF + k];
            }
        }
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
template <bool IS_GQH3, bool kAcc, int kRows = 1, bool kPad = false, int kNsb = 0,
          int kCols = 1>
__global__ void gqh_matvec_tight_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, float4 mag, int64_t x_col_stride,
    int64_t y_col_stride, int ileave, int row_off, int ncols) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int col0 = (kCols == 1) ? (int)blockIdx.y : 0;
    const int ncol = (kCols == 1) ? 1 : (ncols < kCols ? ncols : kCols);
    const int warp = threadIdx.x / GQH_WARP;
    const int row0 = (blockIdx.x * warps_per_block + warp) * kRows + row_off;

    __shared__ float s_ratio[16];
    __shared__ gqh_lvl_tbl<IS_GQH3> s_lvl;
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        gqh_fill_level_lds<IS_GQH3>(&s_lvl, mag, threadIdx.x);
        if constexpr (!IS_GQH3) {
            gqh_fill_pair_lds(&s_lvl, mag, threadIdx.x);
        }
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
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc0[kCols];
    float acc1[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        acc0[c] = 0.0f;
        if constexpr (kRows == 2) {
            acc1[c] = 0.0f;
        }
    }
    auto load_p0 = [&](int sb) {
        if constexpr (kPad) {
            return gqh_load_padded_lane<IS_GQH3>(
                rowbase0, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        } else {
            return gqh_load_tight_lane<IS_GQH3>(
                rowbase0, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        }
    };
    auto load_p1 = [&](int sb) {
        if constexpr (kPad) {
            return gqh_load_padded_lane<IS_GQH3>(
                rowbase1, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        } else {
            return gqh_load_tight_lane<IS_GQH3>(
                rowbase1, sb, sb_pitch, lane, sub, tensor_scale, s_ratio);
        }
    };
    if constexpr (kCols == 1) {
#pragma unroll 4
        for (int sb = 0; sb < nsb; ++sb) {
            gqh_prefetch_sb<IS_GQH3>(rowbase0, sb, nsb, sb_pitch);
            const gqh_tight_lane p0 = load_p0(sb);
            float xs[GQH_PER_LANE];
            gqh_load_lane_xs(xs, x, col0, x_col_stride, sb, j0);
            gqh_acc_tight_lane<IS_GQH3>(acc0[0], p0, s_lvl, xs);
            if constexpr (kRows == 2) {
                const gqh_tight_lane p1 = load_p1(sb);
                gqh_acc_tight_lane<IS_GQH3>(acc1[0], p1, s_lvl, xs);
            }
        }
    } else {
        for (int sb = 0; sb < nsb; ++sb) {
            const gqh_tight_lane p0 = load_p0(sb);
            float w0[GQH_PER_LANE];
            gqh_lane_weights<IS_GQH3>(w0, p0, s_lvl);
            float w1[GQH_PER_LANE];
            if constexpr (kRows == 2) {
                const gqh_tight_lane p1 = load_p1(sb);
                gqh_lane_weights<IS_GQH3>(w1, p1, s_lvl);
            }
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (c >= ncol) {
                    break;
                }
                float xs[GQH_PER_LANE];
                gqh_load_lane_xs(xs, x, col0 + c, x_col_stride, sb, j0);
                gqh_fma_weights(acc0[c], w0, xs);
                if constexpr (kRows == 2) {
                    gqh_fma_weights(acc1[c], w1, xs);
                }
            }
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            acc0[c] += gqh_warp_shfl_down(acc0[c], off);
            if constexpr (kRows == 2) {
                acc1[c] += gqh_warp_shfl_down(acc1[c], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            const int64_t yi0 = (int64_t)(col0 + c) * y_col_stride + row0;
            if constexpr (kAcc) {
                y[yi0] = gqh_bf16_round(y[yi0] + gqh_bf16_round(acc0[c]));
            } else {
                y[yi0] = acc0[c];
            }
            if constexpr (kRows == 2) {
                const int64_t yi1 = yi0 + 1;
                if constexpr (kAcc) {
                    y[yi1] = gqh_bf16_round(y[yi1] + gqh_bf16_round(acc1[c]));
                } else {
                    y[yi1] = acc1[c];
                }
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

    // Only the kRows=2 arms reach gqh_acc_tight_lane, so kRows=1 keeps the
    // narrow table and stays byte-identical.
    using tbl_a_t =
        std::conditional_t<kRows == 2, gqh_lvl_tbl<A_IS_GQH3>, gqh_lvl_grid>;
    using tbl_b_t =
        std::conditional_t<kRows == 2, gqh_lvl_tbl<B_IS_GQH3>, gqh_lvl_grid>;
    __shared__ float s_ratio[16];
    __shared__ tbl_a_t s_lvl_a;
    __shared__ tbl_b_t s_lvl_b;
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        gqh_fill_level_lds<A_IS_GQH3>(&s_lvl_a, mag_a, threadIdx.x);
        gqh_fill_level_lds<B_IS_GQH3>(&s_lvl_b, mag_b, threadIdx.x);
        if constexpr (kRows == 2 && !A_IS_GQH3) {
            gqh_fill_pair_lds(&s_lvl_a, mag_a, threadIdx.x);
        }
        if constexpr (kRows == 2 && !B_IS_GQH3) {
            gqh_fill_pair_lds(&s_lvl_b, mag_b, threadIdx.x);
        }
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
        const float d_real_a = gqh_dreal_uniform(da, scale_a);
        const float d_real_b = gqh_dreal_uniform(db, scale_b);
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
        const uint32_t cwa =
            gqh_load_codeword<A_IS_GQH3>(ba, lane, GQH_CODES_OFF);
        const uint32_t cwb =
            gqh_load_codeword<B_IS_GQH3>(bb, lane, GQH_CODES_OFF);
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            const int ca = gqh_code_at<A_IS_GQH3>(cwa, t);
            const int cb = gqh_code_at<B_IS_GQH3>(cwb, t);
            acc_a0 = gqh_acc_fma(acc_a0, s_lvl_a.grid[ca], s_ba, xs[t]);
            acc_b0 = gqh_acc_fma(acc_b0, s_lvl_b.grid[cb], s_bb, xs[t]);
        }
        if constexpr (kRows == 2) {
            const gqh_tight_lane pa1 = gqh_load_tight_lane<A_IS_GQH3>(
                row_a1, sb, pitch_a, lane, sub, scale_a, s_ratio);
            const gqh_tight_lane pb1 = gqh_load_tight_lane<B_IS_GQH3>(
                row_b1, sb, pitch_b, lane, sub, scale_b, s_ratio);
            gqh_acc_tight_lane<A_IS_GQH3>(acc_a1, pa1, s_lvl_a, xs);
            gqh_acc_tight_lane<B_IS_GQH3>(acc_b1, pb1, s_lvl_b, xs);
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

// SwiGLU pair: no ileave, no dual-row. Same product order as
// gqh_matvec_pair_tight_kernel<1>. Fewer runtime branches so gfx1100
// occupancy can match llama's 4-wave singles.
//
// kFuse=false keeps the two accumulators apart -- y = A.x, y_b = B.x, both
// fp32, no bf16 rounding -- which makes it **bit-exact with the two
// gqh_matvec_tight_kernel singles it replaces** (identical per-row product
// order, identical epilogue). That is the arm the mixed-rung MLP layers use:
// see the gate/up fusion block in gqh_bridge.cpp. Only the epilogue differs,
// so the inner loop keeps the codegen the fused arm was tuned for -- measured
// at in=5120 out=17408 nsb=20 warps=4 on gfx1201, against the two singles:
//   <3,2h> 118.1 vs 131.9 us (+11.7%)   <2h,3>  112.7 vs 135.4 (+20.0%)
//   <3,3>  120.7 vs 138.6 us (+14.9%)   <2h,2h> 111.4 vs 124.4 (+11.6%)
// (gqh_matvec_pair_tight_kernel, even given a compile-time kNsb=20 and an
// unroll-2, only reaches 122.8 us on <3,2h> -- its runtime ileave pitch
// blocks the strength reduction -- so this kernel is the one to reuse.)
template <bool A_IS_GQH3, bool B_IS_GQH3, int kNsb = 0, int kCols = 1,
    bool kFuse = true>
__global__ void gqh_matvec_pair_swiglu_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ y_b,
    int in,
    int out,
    float scale_a,
    float scale_b,
    float4 mag_a,
    float4 mag_b,
    int64_t x_col_stride,
    int64_t y_col_stride,
    int ncols) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int warp = gqh_warp_id();
    const int col0 = (kCols == 1) ? (int)blockIdx.y : 0;
    const int ncol = (kCols == 1) ? 1 : (ncols < kCols ? ncols : kCols);
    const int row = blockIdx.x * warps_per_block + warp;

    // Narrow table on purpose: this kernel never takes the paired path, so its
    // LDS layout -- and its whole ISA -- is byte-identical to before the
    // gqh_lvl_pair_tbl change. The pss<2h,3> / ps<3,3> arms are 6.75 ms/tok of
    // decode and this file has measured them to react to any perturbation.
    __shared__ float s_ratio[16];
    __shared__ gqh_lvl_grid s_lvl_a;
    __shared__ gqh_lvl_grid s_lvl_b;
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        gqh_fill_level_lds<A_IS_GQH3>(&s_lvl_a, mag_a, threadIdx.x);
        gqh_fill_level_lds<B_IS_GQH3>(&s_lvl_b, mag_b, threadIdx.x);
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

    float acc_a[kCols];
    float acc_b[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        acc_a[c] = 0.0f;
        acc_b[c] = 0.0f;
    }
    auto pair_sb = [&](int sb) {
        gqh_prefetch_sb<A_IS_GQH3>(row_a, sb, nsb, sb_a);
        gqh_prefetch_sb<B_IS_GQH3>(row_b, sb, nsb, sb_b);
        const uint8_t* __restrict__ ba = row_a + (int64_t)sb * sb_a;
        const uint8_t* __restrict__ bb = row_b + (int64_t)sb * sb_b;
        const uint8_t da = gqh_ld_sb_u8(ba);
        const uint8_t db = gqh_ld_sb_u8(bb);
        const float d_real_a = gqh_dreal_uniform(da, scale_a);
        const float d_real_b = gqh_dreal_uniform(db, scale_b);
        const uint8_t rba = gqh_ld_sb_u8(ba + 1 + (sub >> 1));
        const uint8_t rbb = gqh_ld_sb_u8(bb + 1 + (sub >> 1));
        const float s_ba =
            d_real_a * s_ratio[(sub & 1) ? (rba >> 4) : (rba & 0x0f)];
        const float s_bb =
            d_real_b * s_ratio[(sub & 1) ? (rbb >> 4) : (rbb & 0x0f)];
        const uint32_t cwa =
            gqh_load_codeword<A_IS_GQH3>(ba, lane, GQH_CODES_OFF);
        const uint32_t cwb =
            gqh_load_codeword<B_IS_GQH3>(bb, lane, GQH_CODES_OFF);
        if constexpr (kCols == 1) {
            float xs[GQH_PER_LANE];
            gqh_load_lane_xs(xs, x, col0, x_col_stride, sb, j0);
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int ca = gqh_code_at<A_IS_GQH3>(cwa, t);
                const int cb = gqh_code_at<B_IS_GQH3>(cwb, t);
                acc_a[0] = gqh_acc_fma(acc_a[0], s_lvl_a.grid[ca], s_ba, xs[t]);
                acc_b[0] = gqh_acc_fma(acc_b[0], s_lvl_b.grid[cb], s_bb, xs[t]);
            }
        } else {
            float wa[GQH_PER_LANE];
            float wb[GQH_PER_LANE];
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int ca = gqh_code_at<A_IS_GQH3>(cwa, t);
                const int cb = gqh_code_at<B_IS_GQH3>(cwb, t);
                wa[t] = s_lvl_a.grid[ca] * s_ba;
                wb[t] = s_lvl_b.grid[cb] * s_bb;
            }
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (c >= ncol) {
                    break;
                }
                float xs[GQH_PER_LANE];
                gqh_load_lane_xs(xs, x, col0 + c, x_col_stride, sb, j0);
                gqh_fma_weights(acc_a[c], wa, xs);
                gqh_fma_weights(acc_b[c], wb, xs);
            }
        }
    };
    if constexpr (kCols == 1) {
#pragma unroll 4
        for (int sb = 0; sb < nsb; ++sb) {
            pair_sb(sb);
        }
    } else {
        for (int sb = 0; sb < nsb; ++sb) {
            pair_sb(sb);
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            acc_a[c] += gqh_warp_shfl_down(acc_a[c], off);
            acc_b[c] += gqh_warp_shfl_down(acc_b[c], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            const int64_t yi = (int64_t)(col0 + c) * y_col_stride + row;
            if constexpr (kFuse) {
                const float g = gqh_bf16_round(acc_a[c]);
                const float u = gqh_bf16_round(acc_b[c]);
                y[yi] = gqh_bf16_round((g / (1.0f + expf(-g))) * u);
            } else {
                y[yi] = acc_a[c];
                y_b[yi] = acc_b[c];
            }
        }
    }
}

// Compact llama-style GQH4 GEMV. Pitch is nsb*137, not the planar stride.
// kCols=1 is decode GEMV (grid.y is the token). kCols>1 is skinny GEMM:
// one weight load, kCols activation columns (blockIdx.y unused).
template <bool kAcc, int kRows = 1, int kNsb = 0, int kCols = 1>
__global__ void gqh4_matvec_tight_kernel(
    const uint8_t* __restrict__ data, const float* __restrict__ x, float* __restrict__ y,
    int in, int out, float tensor_scale, gqh_grid16 grid, int64_t x_col_stride,
    int64_t y_col_stride, int row_off, int ncols) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int col0 = (kCols == 1) ? (int)blockIdx.y : 0;
    const int ncol = (kCols == 1) ? 1 : (ncols < kCols ? ncols : kCols);
    const int warp = threadIdx.x / GQH_WARP;
    const int row0 = (blockIdx.x * warps_per_block + warp) * kRows + row_off;

    __shared__ float s_ratio[16];
    __shared__ float s_grid[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        s_grid[threadIdx.x] = grid.v[threadIdx.x];
    }
    __syncthreads();
    if (row0 >= row_off + out) {
        return;
    }

    const int nsb = kNsb > 0 ? kNsb : (in / GQH_SUPERBLOCK);
    constexpr int sb_bytes = GQH4_SB_BYTES;
    const int row_pitch = nsb * sb_bytes;
    const uint8_t* __restrict__ rowbase0 = data + (int64_t)row0 * row_pitch;
    const uint8_t* __restrict__ rowbase1 = (kRows == 2)
        ? (data + (int64_t)(row0 + 1) * row_pitch)
        : rowbase0;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc0[kCols];
    float acc1[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        acc0[c] = 0.0f;
        if constexpr (kRows == 2) {
            acc1[c] = 0.0f;
        }
    }
    // kNsb=68 (down): no SB unroll. Unroll 4 on skinny B>1 spilled;
    // llama.cpp gqh_matvec_kernel uses a plain loop. Full unroll blows VGPRs.
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* __restrict__ b0 = rowbase0 + (int64_t)sb * sb_bytes;
        const uint8_t d0 = gqh_ld_sb_u8(b0);
        const float d_real0 = gqh_e4m3_dreal(d0) * tensor_scale;
        const uint8_t rb0 = gqh_ld_sb_u8(b0 + 1 + (sub >> 1));
        const float s_b0 = d_real0 * s_ratio[(sub & 1) ? (rb0 >> 4) : (rb0 & 0x0f)];
        uint32_t codes0;
        memcpy(&codes0, b0 + 9 + lane * 4, sizeof(codes0));
        uint32_t codes1 = 0;
        float s_b1 = 0.0f;
        if constexpr (kRows == 2) {
            const uint8_t* __restrict__ b1 = rowbase1 + (int64_t)sb * sb_bytes;
            const uint8_t d1 = gqh_ld_sb_u8(b1);
            const float d_real1 =
                gqh_e4m3_dreal(d1) * tensor_scale;
            const uint8_t rb1 = gqh_ld_sb_u8(b1 + 1 + (sub >> 1));
            s_b1 = d_real1 * s_ratio[(sub & 1) ? (rb1 >> 4) : (rb1 & 0x0f)];
            memcpy(&codes1, b1 + 9 + lane * 4, sizeof(codes1));
        }
        if constexpr (kCols == 1) {
            float xs[GQH_PER_LANE];
            gqh_load_lane_xs(xs, x, col0, x_col_stride, sb, j0);
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                acc0[0] = gqh_acc_fma(
                    acc0[0], s_grid[(codes0 >> (4 * t)) & 0x0f], s_b0, xs[t]);
            }
            if constexpr (kRows == 2) {
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    acc1[0] = gqh_acc_fma(
                        acc1[0],
                        s_grid[(codes1 >> (4 * t)) & 0x0f],
                        s_b1,
                        xs[t]);
                }
            }
        } else {
            float w0[GQH_PER_LANE];
            float w1[GQH_PER_LANE];
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                w0[t] = s_grid[(codes0 >> (4 * t)) & 0x0f] * s_b0;
                if constexpr (kRows == 2) {
                    w1[t] = s_grid[(codes1 >> (4 * t)) & 0x0f] * s_b1;
                }
            }
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (c >= ncol) {
                    break;
                }
                float xs[GQH_PER_LANE];
                gqh_load_lane_xs(xs, x, col0 + c, x_col_stride, sb, j0);
                gqh_fma_weights(acc0[c], w0, xs);
                if constexpr (kRows == 2) {
                    gqh_fma_weights(acc1[c], w1, xs);
                }
            }
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            acc0[c] += gqh_warp_shfl_down(acc0[c], off);
            if constexpr (kRows == 2) {
                acc1[c] += gqh_warp_shfl_down(acc1[c], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            const int64_t yi0 = (int64_t)(col0 + c) * y_col_stride + row0;
            if constexpr (kAcc) {
                y[yi0] = gqh_bf16_round(y[yi0] + gqh_bf16_round(acc0[c]));
            } else {
                y[yi0] = acc0[c];
            }
            if constexpr (kRows == 2) {
                const int64_t yi1 = yi0 + 1;
                if constexpr (kAcc) {
                    y[yi1] = gqh_bf16_round(y[yi1] + gqh_bf16_round(acc1[c]));
                } else {
                    y[yi1] = acc1[c];
                }
            }
        }
    }
}


// kCols=1: grid.y is the token (batched GEMVs). kCols>1: one weight
// load, kCols activation columns (blockIdx.y unused).
template <int kNsb = 0, int kRows = 1, int kCols = 1>
__global__ void gqh4_matvec_pair_swiglu_kernel(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    const float* __restrict__ x,
    float* __restrict__ y,
    int in,
    int out,
    float scale_a,
    float scale_b,
    gqh_grid16 grid_a,
    gqh_grid16 grid_b,
    int64_t x_col_stride,
    int64_t y_col_stride,
    int ncols) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int lane = threadIdx.x % GQH_WARP;
    const int warp = threadIdx.x / GQH_WARP;
    const int col0 = (kCols == 1) ? (int)blockIdx.y : 0;
    const int ncol = (kCols == 1) ? 1 : (ncols < kCols ? ncols : kCols);
    const int row = (blockIdx.x * warps_per_block + warp) * kRows;

    __shared__ float s_ratio[16];
    __shared__ float s_ga[16];
    __shared__ float s_gb[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        s_ga[threadIdx.x] = grid_a.v[threadIdx.x];
        s_gb[threadIdx.x] = grid_b.v[threadIdx.x];
    }
    __syncthreads();
    if (row >= out) {
        return;
    }

    const int nsb = kNsb > 0 ? kNsb : (in / GQH_SUPERBLOCK);
    constexpr int sb = GQH4_SB_BYTES;
    const int pitch = nsb * sb;
    const uint8_t* __restrict__ row_a0 = data_a + (int64_t)row * pitch;
    const uint8_t* __restrict__ row_b0 = data_b + (int64_t)row * pitch;
    const uint8_t* __restrict__ row_a1 =
        (kRows == 2) ? row_a0 + pitch : row_a0;
    const uint8_t* __restrict__ row_b1 =
        (kRows == 2) ? row_b0 + pitch : row_b0;
    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;

    float acc_a0[kCols];
    float acc_b0[kCols];
    float acc_a1[kCols];
    float acc_b1[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        acc_a0[c] = 0.0f;
        acc_b0[c] = 0.0f;
        if constexpr (kRows == 2) {
            acc_a1[c] = 0.0f;
            acc_b1[c] = 0.0f;
        }
    }
    auto load_sb = [&](const uint8_t* ra, int sbi, float scale, uint32_t& codes,
                       float& s_b) {
        const uint8_t* __restrict__ bw = ra + (int64_t)sbi * sb;
        const uint8_t d = gqh_ld_sb_u8(bw);
        const float d_real = gqh_e4m3_dreal(d) * scale;
        const uint8_t rbv = gqh_ld_sb_u8(bw + 1 + (sub >> 1));
        s_b = d_real * s_ratio[(sub & 1) ? (rbv >> 4) : (rbv & 0x0f)];
        memcpy(&codes, bw + 9 + lane * 4, sizeof(codes));
    };
    const int nloop = kNsb > 0 ? kNsb : nsb;
    for (int sbi = 0; sbi < nloop; ++sbi) {
        uint32_t ca0 = 0, cb0 = 0, ca1 = 0, cb1 = 0;
        float s_a0 = 0.0f, s_b0 = 0.0f, s_a1 = 0.0f, s_b1 = 0.0f;
        load_sb(row_a0, sbi, scale_a, ca0, s_a0);
        load_sb(row_b0, sbi, scale_b, cb0, s_b0);
        if constexpr (kRows == 2) {
            load_sb(row_a1, sbi, scale_a, ca1, s_a1);
            load_sb(row_b1, sbi, scale_b, cb1, s_b1);
        }
        if constexpr (kCols == 1) {
            float xs[GQH_PER_LANE];
            gqh_load_lane_xs(xs, x, col0, x_col_stride, sbi, j0);
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                acc_a0[0] = gqh_acc_fma(
                    acc_a0[0], s_ga[(ca0 >> (4 * t)) & 0x0f], s_a0, xs[t]);
                acc_b0[0] = gqh_acc_fma(
                    acc_b0[0], s_gb[(cb0 >> (4 * t)) & 0x0f], s_b0, xs[t]);
            }
            if constexpr (kRows == 2) {
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    acc_a1[0] = gqh_acc_fma(
                        acc_a1[0], s_ga[(ca1 >> (4 * t)) & 0x0f], s_a1, xs[t]);
                    acc_b1[0] = gqh_acc_fma(
                        acc_b1[0], s_gb[(cb1 >> (4 * t)) & 0x0f], s_b1, xs[t]);
                }
            }
        } else {
            float wa0[GQH_PER_LANE];
            float wb0[GQH_PER_LANE];
            float wa1[GQH_PER_LANE];
            float wb1[GQH_PER_LANE];
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                wa0[t] = s_ga[(ca0 >> (4 * t)) & 0x0f] * s_a0;
                wb0[t] = s_gb[(cb0 >> (4 * t)) & 0x0f] * s_b0;
                if constexpr (kRows == 2) {
                    wa1[t] = s_ga[(ca1 >> (4 * t)) & 0x0f] * s_a1;
                    wb1[t] = s_gb[(cb1 >> (4 * t)) & 0x0f] * s_b1;
                }
            }
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (c >= ncol) {
                    break;
                }
                float xs[GQH_PER_LANE];
                gqh_load_lane_xs(xs, x, col0 + c, x_col_stride, sbi, j0);
                gqh_fma_weights(acc_a0[c], wa0, xs);
                gqh_fma_weights(acc_b0[c], wb0, xs);
                if constexpr (kRows == 2) {
                    gqh_fma_weights(acc_a1[c], wa1, xs);
                    gqh_fma_weights(acc_b1[c], wb1, xs);
                }
            }
        }
    }
#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            acc_a0[c] += gqh_warp_shfl_down(acc_a0[c], off);
            acc_b0[c] += gqh_warp_shfl_down(acc_b0[c], off);
            if constexpr (kRows == 2) {
                acc_a1[c] += gqh_warp_shfl_down(acc_a1[c], off);
                acc_b1[c] += gqh_warp_shfl_down(acc_b1[c], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= ncol) {
                break;
            }
            const int64_t yi = (int64_t)(col0 + c) * y_col_stride + row;
            const float g = gqh_bf16_round(acc_a0[c]);
            const float u = gqh_bf16_round(acc_b0[c]);
            y[yi] = gqh_bf16_round((g / (1.0f + expf(-g))) * u);
            if constexpr (kRows == 2) {
                const float g1 = gqh_bf16_round(acc_a1[c]);
                const float u1 = gqh_bf16_round(acc_b1[c]);
                y[yi + 1] = gqh_bf16_round((g1 / (1.0f + expf(-g1))) * u1);
            }
        }
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
