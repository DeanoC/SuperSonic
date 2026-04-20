// Bidirectional attention helper for the DFlash draft model.
//
// This is the one and only new HIP kernel for DFlash M2. Everything else in
// the draft's forward pass reuses existing Qwen3.5 primitives (rms_norm with
// add_unit_offset=0, matmul_rhs_transposed_4b, apply_rope_prefill,
// swiglu_mul, element_add). Keeping this file self-contained (no #include of
// other .hip sources) follows the per-family codegen-fork rule in memory
// feedback_hipcc_codegen.md.
//
// Shape contract (SHD layout, contiguous):
//   Q:  [q_len, num_q_heads, head_dim]       BF16
//   K:  [seq_len, num_kv_heads, head_dim]    BF16   seq_len = past + ctx + q_len
//   V:  [seq_len, num_kv_heads, head_dim]    BF16
//   Out:[q_len, num_q_heads, head_dim]       BF16
//
// Attention is fully bidirectional (is_causal = False in dflash.py). GQA is
// handled by `kv_head = q_head / (num_q_heads / num_kv_heads)`. The scale is
// `1 / sqrt(head_dim)` and must be passed in so the caller can bake any
// softmax-scaling quirks if they ever show up in the checkpoint (none today).
//
// This kernel is correctness-first, not perf-tuned. At M2 smoke the shapes
// are tiny (q_len=16, seq_len≤17, nH=32), so launch overhead dominates and a
// persistent/fused design would be premature. M4 profiling will tell us
// whether to megakernel-ify the whole draft.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace {

// LDS budget per block:
//   scores[seq_len]           — fp32 attention scores before softmax
//   q_vec[head_dim]           — fp32 cache of one Q row (head_dim lanes)
//   reduce_scratch[blockDim]  — fp32 reduction workspace (max/sum)
// For gfx1150 at max seq_len=32K and head_dim=128 this would overflow LDS
// (128 KiB budget). M2 smoke has seq_len ≤ 17, so we're firmly inside. The
// kernel takes seq_len as a runtime parameter and returns error status 91
// if the combined LDS request exceeds a safe 48 KiB cap.

__device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __hip_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Grid: (q_len, num_q_heads)
// Block: (head_dim)   — head_dim must be a power of 2 and ≤ 1024
__global__ void dflash_draft_bidir_attn_kernel(
    const __hip_bfloat16* __restrict__ q,      // [q_len, nH, hd]
    const __hip_bfloat16* __restrict__ k,      // [seq_len, nKV, hd]
    const __hip_bfloat16* __restrict__ v,      // [seq_len, nKV, hd]
    __hip_bfloat16* __restrict__ out,          // [q_len, nH, hd]
    int q_len,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    const int qi = blockIdx.x;   // query position
    const int h  = blockIdx.y;   // query head
    const int kv_h = h / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;  // == head_dim

    extern __shared__ float smem[];
    float* q_vec   = smem;                   // [head_dim]
    float* scores  = smem + head_dim;        // [seq_len]
    float* reduce  = smem + head_dim + seq_len;  // [block_threads]

    // ---- Load Q[qi, h, :] into LDS as fp32 ----
    {
        const size_t q_row_off = (size_t)(qi * num_q_heads + h) * head_dim;
        if (tid < head_dim) {
            q_vec[tid] = bf16_to_f32(q[q_row_off + tid]);
        }
    }
    __syncthreads();

    // ---- Phase 1: scores[j] = dot(Q, K[j, kv_h, :]) * scale ----
    for (int j = tid; j < seq_len; j += block_threads) {
        const size_t k_row_off = (size_t)(j * num_kv_heads + kv_h) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += q_vec[d] * bf16_to_f32(k[k_row_off + d]);
        }
        scores[j] = s * scale;
    }
    __syncthreads();

    // ---- Phase 2: online softmax (max then exp then normalize) ----
    // Block-wide max across seq_len. seq_len is small (≤ a few K), so a serial
    // pass by thread 0 is fine for M2. M4 can switch to a block-stride reduce.
    if (tid == 0) {
        float m = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            m = fmaxf(m, scores[j]);
        }
        float s = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = expf(scores[j] - m);
            s += scores[j];
        }
        const float inv_s = s > 0.0f ? 1.0f / s : 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] *= inv_s;
        }
    }
    __syncthreads();

    // ---- Phase 3: out[qi, h, tid] = sum_j scores[j] * V[j, kv_h, tid] ----
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            const size_t v_off = (size_t)(j * num_kv_heads + kv_h) * head_dim + tid;
            acc += scores[j] * bf16_to_f32(v[v_off]);
        }
        const size_t o_off = (size_t)(qi * num_q_heads + h) * head_dim + tid;
        out[o_off] = f32_to_bf16(acc);
    }
    // `reduce` is currently unused — placeholder for M4 block-stride softmax.
    (void)reduce;
}

struct ScopedDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) {
            hipSetDevice(target);
            changed = true;
        }
    }
    ~ScopedDevice() {
        if (changed && previous >= 0) hipSetDevice(previous);
    }
};

} // namespace

extern "C" int dflash_draft_hip_bidir_attention(
    int device_ordinal,
    int q_len,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    const void* q,
    const void* k,
    const void* v,
    void* out
) {
    if (q_len <= 0 || seq_len <= 0 || head_dim <= 0 || num_q_heads <= 0 || num_kv_heads <= 0) return 91;
    if (num_q_heads % num_kv_heads != 0) return 92;
    if (head_dim > 1024) return 93;
    const size_t lds_floats = (size_t)head_dim + (size_t)seq_len + (size_t)head_dim;
    const size_t lds_bytes  = lds_floats * sizeof(float);
    if (lds_bytes > 48 * 1024) return 94;

    ScopedDevice scoped(device_ordinal);

    dim3 grid(q_len, num_q_heads, 1);
    dim3 block(head_dim, 1, 1);

    hipLaunchKernelGGL(
        dflash_draft_bidir_attn_kernel,
        grid, block, lds_bytes, 0,
        reinterpret_cast<const __hip_bfloat16*>(q),
        reinterpret_cast<const __hip_bfloat16*>(k),
        reinterpret_cast<const __hip_bfloat16*>(v),
        reinterpret_cast<__hip_bfloat16*>(out),
        q_len, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return (int)err;
    return 0;
}
