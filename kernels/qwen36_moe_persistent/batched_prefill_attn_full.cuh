// Batched-Q full attention prefill kernel for Qwen 3.6 MoE.
//
// Direct algorithmic port of the Qwen 3.5 tiled prefill attention kernel
// (kernels/full_attention.hip:378, see PR #219) into the qwen36_moe namespace.
// Same K-tile FlashAttention-style design: BM warps in a block cooperatively
// load BK rows of K/V into LDS once per outer tile, then each warp runs an
// online-softmax recurrence (m_i, l_i, acc[]) over its own query row.
//
// **Scope of this kernel:** attention math only. Pre-projection (Q/K/V matmul +
// RoPE) and KV cache write happen outside (separate kernel calls; see
// batched_prefill_kv_write.cuh once it exists in M5/M6). Output is the
// pre-o_proj attention result `[B, H, q_len, D]` in F32.
//
// **Causal contract:** caller passes `seqlen_offset = past_len` (number of
// cache positions already valid before this chunk's queries). Query at
// chunk position `qr` attends to cache positions `[0, past_len + qr]`
// (inclusive). `kv_len` is the total cache length the kernel may read
// (typically `past_len + q_len`, possibly more if the cache has been
// pre-extended).
//
// **Wave64 portability:** the kernel hardcodes block.x = 32 and assumes
// `warpSize == 32` in stride math. The launcher must refuse the launch
// when `props.warpSize != 32` (return code 137 in the bridge — same
// pattern as full_attention_bridge.cpp's `launch_tiled` guard).
//
// **LDS budget:** `2 * BK * head_dim * sizeof(T)`. At Qwen 3.6 MoE's
// hd=256 with BK=32: 32 KiB BF16 — under the 48 KiB safety cap on
// gfx1100 (matches the Qwen 3.5 hd=256 case in PR #219).

#pragma once

#include "../qwen36_moe_cuda_prelude.cuh"

namespace qwen36_moe {

// __shfl_down butterfly reduction; result lives only on lane 0. Callers
// MUST broadcast via `__shfl(result, 0)` before reading on other lanes
// (see PR #219 wave_sum bug: bf16_round_rne_f32 of partial-sum noise on
// lanes 1..31 produced max_rel=7.31 in the parity sweep).
__device__ __forceinline__ float wave_sum_f32(float v) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down(v, offset);
    }
    return v;
}

// Activation type narrow: qwen3.6-moe runs INT4 weights → BF16 acts only.
// Adding __half/float specializations later would require also pulling
// hip/hip_fp16.h into qwen36_moe_cuda_prelude.cuh, which currently keeps
// the translation unit tight. Caller passes BF16; the bridge rejects
// dtype != 2.
template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

// Grid:  (ceil(q_len / BM), q_heads, batch)
// Block: (warpSize, BM)   — BM warps × warpSize lanes
//
// Each warp owns one query row qr = block_qr_base + warp_y and walks the
// K dimension in tiles of BK rows. K and V tiles are loaded into LDS
// once per tile and shared across all BM warps in the block, dropping
// global K/V bandwidth by ~BM× vs single-warp / per-token kernels.
template <typename T, int BM, int BK>
__global__ void qwen36_moe_batched_prefill_attn_full_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    float* __restrict__ out
) {
    constexpr int ACC_MAX = 8;   // head_dim <= 8 * warpSize asserted in launcher

    const int qr_base = blockIdx.x * BM;
    const int qh      = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_h    = qh / num_kv_groups;
    const int warp_y  = threadIdx.y;        // 0..BM-1
    const int lane    = threadIdx.x;        // 0..warpSize-1
    const int qr      = qr_base + warp_y;
    if (lane >= warpSize) return;
    if (batch >= batch_size || qh >= q_heads) return;

    extern __shared__ unsigned char qwen36_smem_raw[];
    T* k_tile = reinterpret_cast<T*>(qwen36_smem_raw);
    T* v_tile = k_tile + BK * head_dim;

    const bool active = (qr < q_len);
    const int causal_limit = active ? min(kv_len, seqlen_offset + qr + 1) : 0;

    const T* q_row = active
        ? (query + (((batch * q_heads + qh) * q_len + qr) * head_dim))
        : nullptr;
    const T* k_head_ptr = key   + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    const T* v_head_ptr = value + ((batch * kv_heads + kv_h) * kv_len * head_dim);
    float*   out_row = active
        ? (out + (((batch * q_heads + qh) * q_len + qr) * head_dim))
        : nullptr;

    // Per-lane output accumulators (in registers). Each lane owns the dims
    // d = lane, lane + warpSize, lane + 2*warpSize, ... (up to ACC_MAX of them).
    float acc[ACC_MAX];
    int   acc_dim[ACC_MAX];
    int   acc_count = 0;
    #pragma unroll
    for (int i = 0; i < ACC_MAX; ++i) acc[i] = 0.0f;
    for (int d = lane; d < head_dim && acc_count < ACC_MAX; d += warpSize) {
        acc_dim[acc_count++] = d;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Iterate K tiles. Even inactive warps must participate in cooperative
    // loads and __syncthreads() — they just won't write output.
    const int causal_for_block = active ? causal_limit : 0;
    // For the cooperative load we need an upper bound on K positions any
    // warp in this block will need. Different warps can have different
    // causal_limit values (qr varies). Use the maximum across the block.
    __shared__ int block_max_kpos;
    if (threadIdx.x == 0 && threadIdx.y == 0) block_max_kpos = 0;
    __syncthreads();
    if (lane == 0) atomicMax(&block_max_kpos, causal_for_block);
    __syncthreads();
    const int outer_limit = block_max_kpos;

    for (int tile_base = 0; tile_base < outer_limit; tile_base += BK) {
        const int tile_len = min(BK, outer_limit - tile_base);

        // Cooperative load: BM warps × warpSize lanes pull tile_len * head_dim
        // T values into LDS. Stride by total threads = BM * warpSize.
        const int load_total = tile_len * head_dim;
        const int total_threads = BM * warpSize;
        const int linear_tid = warp_y * warpSize + lane;
        for (int i = linear_tid; i < load_total; i += total_threads) {
            const int row = i / head_dim;
            const int col = i - row * head_dim;
            const int gk_pos = tile_base + row;
            const size_t off = (size_t)gk_pos * head_dim + (size_t)col;
            k_tile[i] = k_head_ptr[off];
            v_tile[i] = v_head_ptr[off];
        }
        __syncthreads();

        // Walk this tile, applying the online softmax recurrence per query row.
        if (active) {
            const int j_max = min(tile_len, causal_limit - tile_base);
            for (int j = 0; j < j_max; ++j) {
                const T* k_row = k_tile + j * head_dim;
                const T* v_row = v_tile + j * head_dim;

                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += warpSize) {
                    partial += to_float<T>(q_row[d]) * to_float<T>(k_row[d]);
                }
                // wave_sum_f32 leaves the result only on lane 0; broadcast so
                // every lane sees the same score (each lane runs its own
                // per-output-dim recurrence).
                float score = __shfl(wave_sum_f32(partial), 0) * scale;

                const float m_old = m_i;
                const float m_new = score > m_old ? score : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                const float weight = expf(score - m_new);
                l_i = l_i * alpha + weight;
                m_i = m_new;

                for (int i = 0; i < acc_count; ++i) {
                    acc[i] = acc[i] * alpha + weight * to_float<T>(v_row[acc_dim[i]]);
                }
            }
        }
        __syncthreads();   // before next tile reuses k_tile/v_tile
    }

    if (active) {
        const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int i = 0; i < acc_count; ++i) {
            out_row[acc_dim[i]] = acc[i] * inv;
        }
    }
}

}  // namespace qwen36_moe
