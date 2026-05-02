// Bridge for Gemma 4 decode primitives. Separate compilation unit from the
// Qwen kernels — see gemma4.hip for rationale.

#include "gemma4_cuda.cuh"

#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>
#include <stdint.h>

namespace {

struct ScopedCudaDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedCudaDevice(int target) {
        cudaGetDevice(&previous);
        if (previous != target) { cudaSetDevice(target); changed = true; }
    }
    ~ScopedCudaDevice() { if (changed && previous >= 0) cudaSetDevice(previous); }
};

// ---- RMSNorm (Gemma variant: no (w+1) offset) ----

template <typename T>
int rms_norm_device(int device_ordinal, int n_cols, float eps,
                    const void* xs, const void* weight, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    constexpr int block = 256;
    g4_rms_norm_kernel<T><<<dim3(1), dim3(block), 0, 0>>>(
        n_cols, eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 401;
    if (cudaDeviceSynchronize() != cudaSuccess) return 402;
    return 0;
}

// ---- Work-stealing matvec ----

template <typename T>
int matvec_device(int device_ordinal, int in_dim, int out_dim,
                  const void* x, const void* W, void* out,
                  unsigned int* row_counter) {
    ScopedCudaDevice scoped(device_ordinal);
    if (cudaMemset(row_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 410;

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 411;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    g4_matvec_workstealing_kernel<T><<<dim3(num_blocks), dim3(block), 0, 0>>>(
        in_dim, out_dim,
        static_cast<const T*>(x),
        static_cast<const T*>(W),
        static_cast<T*>(out),
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 412;
    if (cudaDeviceSynchronize() != cudaSuccess) return 413;
    return 0;
}

// ---- GeLU-tanh gated multiply ----

template <typename T>
int gelu_tanh_gate_mul_device(int device_ordinal, size_t n,
                              const void* gate, const void* up, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    g4_gelu_tanh_gate_mul_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        n,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 421;
    if (cudaDeviceSynchronize() != cudaSuccess) return 422;
    return 0;
}

// ---- RoPE decode (split-half Gemma style) ----

template <typename T>
int rope_decode_device(int device_ordinal,
                       int num_heads, int head_dim, int rotary_dim, int position,
                       const void* cos_table, const void* sin_table, void* x) {
    ScopedCudaDevice scoped(device_ordinal);
    const int half = rotary_dim / 2;
    const size_t total = static_cast<size_t>(num_heads) * half;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    g4_rope_split_half_decode_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        num_heads, head_dim, rotary_dim, position,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(x));
    if (cudaGetLastError() != cudaSuccess) return 431;
    if (cudaDeviceSynchronize() != cudaSuccess) return 432;
    return 0;
}

// ---- KV append ----

template <typename T>
int kv_append_device(int device_ordinal,
                     int num_kv_heads, int head_dim, int pos, int max_T,
                     const void* k_in, const void* v_in,
                     void* k_cache, void* v_cache) {
    ScopedCudaDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(num_kv_heads) * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    g4_kv_append_decode_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        num_kv_heads, head_dim, pos, max_T,
        static_cast<const T*>(k_in),
        static_cast<const T*>(v_in),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache));
    if (cudaGetLastError() != cudaSuccess) return 441;
    if (cudaDeviceSynchronize() != cudaSuccess) return 442;
    return 0;
}

// ---- SWA (or full) attention for one decode token ----

template <typename T>
int swa_attn_decode_device(int device_ordinal,
                           int num_q_heads, int num_kv_heads,
                           int head_dim, int kv_len, int max_T,
                           int sliding_window, float scale,
                           const void* q, const void* k_cache, const void* v_cache,
                           void* scores_scratch, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (kv_len <= 0) return 450;

    constexpr int BLOCK = 256;
    // Phase 1: per-(q_head, t) score computation.
    {
        dim3 grid(num_q_heads, (kv_len + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        g4_attn_scores_kernel<T><<<grid, block, 0, 0>>>(
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            sliding_window, scale,
            static_cast<const T*>(q),
            static_cast<const T*>(k_cache),
            static_cast<float*>(scores_scratch));
        if (cudaGetLastError() != cudaSuccess) return 451;
    }

    // Phase 2: per-q_head softmax.
    {
        dim3 grid(num_q_heads, 1, 1);
        dim3 block(BLOCK, 1, 1);
        g4_attn_softmax_kernel<<<grid, block, 0, 0>>>(
            num_q_heads, kv_len, max_T,
            static_cast<float*>(scores_scratch));
        if (cudaGetLastError() != cudaSuccess) return 452;
    }

    // Phase 3: value aggregation per (q_head, head_dim col).
    {
        dim3 grid(num_q_heads, (head_dim + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        g4_attn_value_aggregate_kernel<T><<<grid, block, 0, 0>>>(
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            static_cast<const float*>(scores_scratch),
            static_cast<const T*>(v_cache),
            static_cast<T*>(out));
        if (cudaGetLastError() != cudaSuccess) return 453;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) return 454;
    return 0;
}

// =============================================================================
// Prefill / batched device wrappers (Step 13).
// Mirror the single-token device wrappers above but accept a `seq_len`
// (or `n_rows`) dimension so the kernel launch covers all prompt tokens at
// once. Same dtype dispatch pattern (__half / float / __nv_bfloat16).
// =============================================================================

template <typename T>
int rms_norm_rows_device(int device_ordinal, int n_rows, int n_cols, float eps,
                         const void* xs, const void* weight, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (n_rows <= 0) return 0;
    constexpr int block = 256;
    g4_rms_norm_rows_kernel<T><<<dim3(n_rows), dim3(block), 0, 0>>>(
        n_rows, n_cols, eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 461;
    if (cudaDeviceSynchronize() != cudaSuccess) return 462;
    return 0;
}

// Does this device support RDNA3 WMMA intrinsics? See
// kernels/full_attention_bridge_4b.cpp for the analogous helper on the Qwen
// side — the helper is duplicated here (not shared) to keep Gemma 4 in its
// own compilation unit (gemma4.hip codegen has historically been sensitive
// to cross-contamination with the Qwen bridge). `std::call_once` protects
// against data races from concurrent `supersonic-serve` request threads;
// `SUPERSONIC_GEMMA4_DISABLE_WMMA=1` forces the scalar path for A/B compare.
static bool g4_device_supports_wmma_bf16(int device_ordinal) {
    (void)device_ordinal;
    return false;
}

static int matvec_batched_wmma_bf16_device(int device_ordinal, int seq_len, int in_dim, int out_dim, const void* x, const void* W, void* out) {
    (void)device_ordinal; (void)seq_len; (void)in_dim; (void)out_dim; (void)x; (void)W; (void)out;
    return 474;
}

template <typename T>
int matvec_batched_device(int device_ordinal,
                          int seq_len, int in_dim, int out_dim,
                          const void* x, const void* W, void* out,
                          unsigned int* counter) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    if (cudaMemset(counter, 0, sizeof(unsigned int)) != cudaSuccess) return 470;

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 471;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    g4_matvec_batched_kernel<T><<<dim3(num_blocks), dim3(block), 0, 0>>>(
        seq_len, in_dim, out_dim,
        static_cast<const T*>(x),
        static_cast<const T*>(W),
        static_cast<T*>(out),
        counter);
    if (cudaGetLastError() != cudaSuccess) return 472;
    if (cudaDeviceSynchronize() != cudaSuccess) return 473;
    return 0;
}

// ---- INT4 matvec (single-token, work-stealing on output rows) ----

template <typename T>
int matvec_int4_device(int device_ordinal, int in_dim, int out_dim, int gsz,
                       const void* x, const void* W_packed,
                       const void* W_scale, const void* W_zero,
                       void* out, unsigned int* row_counter) {
    ScopedCudaDevice scoped(device_ordinal);
    if (cudaMemset(row_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 480;

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 481;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    g4_matvec_int4_workstealing_kernel<T><<<dim3(num_blocks), dim3(block), 0, 0>>>(
        in_dim, out_dim, gsz,
        static_cast<const T*>(x),
        static_cast<const uint8_t*>(W_packed),
        static_cast<const __nv_bfloat16*>(W_scale),
        static_cast<const __nv_bfloat16*>(W_zero),
        static_cast<T*>(out),
        row_counter);
    if (cudaGetLastError() != cudaSuccess) return 482;
    if (cudaDeviceSynchronize() != cudaSuccess) return 483;
    return 0;
}

// ---- INT4 batched matvec ----

static int matvec_batched_int4_wmma_bf16_device(int device_ordinal, int seq_len, int in_dim, int out_dim, int gsz, const void* x, const void* W_packed, const void* W_scale, const void* W_zero, void* out) {
    (void)device_ordinal; (void)seq_len; (void)in_dim; (void)out_dim; (void)gsz; (void)x; (void)W_packed; (void)W_scale; (void)W_zero; (void)out;
    return 494;
}

template <typename T>
int matvec_batched_int4_device(int device_ordinal,
                               int seq_len, int in_dim, int out_dim, int gsz,
                               const void* x, const void* W_packed,
                               const void* W_scale, const void* W_zero,
                               void* out, unsigned int* counter) {
    (void)device_ordinal; (void)seq_len; (void)in_dim; (void)out_dim; (void)gsz;
    (void)x; (void)W_packed; (void)W_scale; (void)W_zero; (void)out; (void)counter;
    return 492;
}

template <typename T>
int rope_prefill_device(int device_ordinal,
                        int seq_len, int num_heads, int head_dim,
                        int rotary_dim, int pos_base,
                        const void* cos_table, const void* sin_table, void* x) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const int half = rotary_dim / 2;
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half;
    if (total == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    g4_rope_prefill_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        seq_len, num_heads, head_dim, rotary_dim, pos_base,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(x));
    if (cudaGetLastError() != cudaSuccess) return 481;
    if (cudaDeviceSynchronize() != cudaSuccess) return 482;
    return 0;
}

template <typename T>
int kv_append_prefill_device(int device_ordinal,
                             int seq_len, int num_kv_heads, int head_dim,
                             int pos_base, int max_T,
                             const void* k_in, const void* v_in,
                             void* k_cache, void* v_cache) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const size_t total = static_cast<size_t>(seq_len) * num_kv_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    g4_kv_append_prefill_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        seq_len, num_kv_heads, head_dim, pos_base, max_T,
        static_cast<const T*>(k_in),
        static_cast<const T*>(v_in),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache));
    if (cudaGetLastError() != cudaSuccess) return 491;
    if (cudaDeviceSynchronize() != cudaSuccess) return 492;
    return 0;
}

template <typename T>
int attn_prefill_device(int device_ordinal,
                        int seq_len, int num_q_heads, int num_kv_heads,
                        int head_dim, int pos_base, int max_T,
                        int sliding_window, float scale,
                        const void* q, const void* k_cache, const void* v_cache,
                        void* scores_scratch, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const int kv_total = pos_base + seq_len;
    if (kv_total <= 0) return 500;

    constexpr int BLOCK = 256;
    {
        dim3 grid(num_q_heads, (kv_total + BLOCK - 1) / BLOCK, seq_len);
        dim3 block(BLOCK, 1, 1);
        g4_attn_prefill_scores_kernel<T><<<grid, block, 0, 0>>>(
            seq_len, num_q_heads, num_kv_heads, head_dim,
            pos_base, max_T, sliding_window, scale,
            static_cast<const T*>(q),
            static_cast<const T*>(k_cache),
            static_cast<float*>(scores_scratch));
        if (cudaGetLastError() != cudaSuccess) return 501;
    }
    {
        dim3 grid(num_q_heads, seq_len, 1);
        dim3 block(BLOCK, 1, 1);
        g4_attn_prefill_softmax_kernel<<<grid, block, 0, 0>>>(
            seq_len, num_q_heads, kv_total, max_T,
            static_cast<float*>(scores_scratch));
        if (cudaGetLastError() != cudaSuccess) return 502;
    }
    {
        dim3 grid(num_q_heads, (head_dim + BLOCK - 1) / BLOCK, seq_len);
        dim3 block(BLOCK, 1, 1);
        g4_attn_prefill_value_kernel<T><<<grid, block, 0, 0>>>(
            seq_len, num_q_heads, num_kv_heads, head_dim,
            pos_base, max_T,
            static_cast<const float*>(scores_scratch),
            static_cast<const T*>(v_cache),
            static_cast<T*>(out));
        if (cudaGetLastError() != cudaSuccess) return 503;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) return 504;
    return 0;
}

template <typename T>
int add_residual_device(int device_ordinal, size_t n,
                        const void* a, const void* b, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    g4_add_residual_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        n,
        static_cast<const T*>(a),
        static_cast<const T*>(b),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 511;
    if (cudaDeviceSynchronize() != cudaSuccess) return 512;
    return 0;
}

template <typename T>
int add_scaled_residual_device(int device_ordinal, size_t n, float scalar,
                               const void* a, const void* b, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    g4_add_scaled_residual_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        n, scalar,
        static_cast<const T*>(a),
        static_cast<const T*>(b),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 521;
    if (cudaDeviceSynchronize() != cudaSuccess) return 522;
    return 0;
}

template <typename T>
int scalar_mul_inplace_device(int device_ordinal, size_t n, float scalar, void* x) {
    ScopedCudaDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    g4_scalar_mul_inplace_kernel<T><<<dim3(grid), dim3(block), 0, 0>>>(
        n, scalar, static_cast<T*>(x));
    if (cudaGetLastError() != cudaSuccess) return 541;
    if (cudaDeviceSynchronize() != cudaSuccess) return 542;
    return 0;
}

template <typename T>
int fused_attn_block_device(int device_ordinal,
                            int hidden_size, int num_q_heads, int num_kv_heads,
                            int head_dim, int rotary_dim, int sliding_window,
                            int position, int max_T, int shared_kv,
                            float eps, float scale,
                            const void* hidden_in, void* hidden_out,
                            const void* input_norm_w,
                            const void* q_proj_w, const void* k_proj_w,
                            const void* v_proj_w,
                            const void* q_norm_w, const void* k_norm_w,
                            const void* o_proj_w, const void* post_attn_norm_w,
                            const void* cos_table, const void* sin_table,
                            void* k_cache, void* v_cache,
                            void* workspace,
                            unsigned int* matvec_counter,
                            unsigned int* barrier_counter,
                            unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 601;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;

    // LDS: bs slots for reduction scratch (shared with softmax/matmul). Single
    // scratch pool — the kernel reuses it across phases.
    const size_t lds_bytes = BLOCK * sizeof(float);

    // Zero the barrier counter / flag before launch. We leave matvec_counter
    // to be reset by the kernel itself (it's cleared twice during execution).
    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 602;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 603;

    g4_fused_attn_block_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        hidden_size, num_q_heads, num_kv_heads, head_dim,
        rotary_dim, sliding_window, position, max_T, shared_kv,
        eps, scale,
        static_cast<const T*>(hidden_in),
        static_cast<T*>(hidden_out),
        static_cast<const T*>(input_norm_w),
        static_cast<const T*>(q_proj_w),
        static_cast<const T*>(k_proj_w),
        static_cast<const T*>(v_proj_w),
        static_cast<const T*>(q_norm_w),
        static_cast<const T*>(k_norm_w),
        static_cast<const T*>(o_proj_w),
        static_cast<const T*>(post_attn_norm_w),
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 604;
    if (cudaDeviceSynchronize() != cudaSuccess) return 605;
    return 0;
}

template <typename T>
int fused_attn_block_int4_device(int device_ordinal,
                                 int hidden_size, int num_q_heads, int num_kv_heads,
                                 int head_dim, int rotary_dim, int sliding_window,
                                 int position, int max_T, int shared_kv, int gsz,
                                 float eps, float scale,
                                 const void* hidden_in, void* hidden_out,
                                 const void* input_norm_w,
                                 const void* q_proj_packed,
                                 const void* q_proj_scale,
                                 const void* q_proj_zero,
                                 const void* k_proj_packed,
                                 const void* k_proj_scale,
                                 const void* k_proj_zero,
                                 const void* v_proj_packed,
                                 const void* v_proj_scale,
                                 const void* v_proj_zero,
                                 const void* q_norm_w, const void* k_norm_w,
                                 const void* o_proj_packed,
                                 const void* o_proj_scale,
                                 const void* o_proj_zero,
                                 const void* post_attn_norm_w,
                                 const void* cos_table, const void* sin_table,
                                 void* k_cache, void* v_cache,
                                 void* workspace,
                                 unsigned int* matvec_counter,
                                 unsigned int* barrier_counter,
                                 unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 621;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 622;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 623;

    g4_fused_attn_block_int4_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        hidden_size, num_q_heads, num_kv_heads, head_dim,
        rotary_dim, sliding_window, position, max_T, shared_kv, gsz,
        eps, scale,
        static_cast<const T*>(hidden_in),
        static_cast<T*>(hidden_out),
        static_cast<const T*>(input_norm_w),
        static_cast<const uint8_t*>(q_proj_packed),
        static_cast<const __nv_bfloat16*>(q_proj_scale),
        static_cast<const __nv_bfloat16*>(q_proj_zero),
        static_cast<const uint8_t*>(k_proj_packed),
        static_cast<const __nv_bfloat16*>(k_proj_scale),
        static_cast<const __nv_bfloat16*>(k_proj_zero),
        static_cast<const uint8_t*>(v_proj_packed),
        static_cast<const __nv_bfloat16*>(v_proj_scale),
        static_cast<const __nv_bfloat16*>(v_proj_zero),
        static_cast<const T*>(q_norm_w),
        static_cast<const T*>(k_norm_w),
        static_cast<const uint8_t*>(o_proj_packed),
        static_cast<const __nv_bfloat16*>(o_proj_scale),
        static_cast<const __nv_bfloat16*>(o_proj_zero),
        static_cast<const T*>(post_attn_norm_w),
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 624;
    if (cudaDeviceSynchronize() != cudaSuccess) return 625;
    return 0;
}

template <typename T>
int fused_mlp_ple_device(int device_ordinal,
                         int hidden_size, int intermediate_size, int ple_hidden,
                         float eps, float layer_scalar,
                         const void* hidden_in, void* hidden_out,
                         const void* pre_ff_norm_w,
                         const void* gate_proj_w, const void* up_proj_w,
                         const void* down_proj_w, const void* post_ff_norm_w,
                         const void* per_layer_input,
                         const void* per_layer_input_gate_w,
                         const void* per_layer_projection_w,
                         const void* post_per_layer_input_norm_w,
                         void* workspace,
                         unsigned int* matvec_counter,
                         unsigned int* barrier_counter,
                         unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 611;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 612;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 613;

    g4_fused_mlp_ple_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        hidden_size, intermediate_size, ple_hidden,
        eps, layer_scalar,
        static_cast<const T*>(hidden_in),
        static_cast<T*>(hidden_out),
        static_cast<const T*>(pre_ff_norm_w),
        static_cast<const T*>(gate_proj_w),
        static_cast<const T*>(up_proj_w),
        static_cast<const T*>(down_proj_w),
        static_cast<const T*>(post_ff_norm_w),
        static_cast<const T*>(per_layer_input),
        static_cast<const T*>(per_layer_input_gate_w),
        static_cast<const T*>(per_layer_projection_w),
        static_cast<const T*>(post_per_layer_input_norm_w),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 614;
    if (cudaDeviceSynchronize() != cudaSuccess) return 615;
    return 0;
}

template <typename T>
int fused_mlp_ple_int4_device(int device_ordinal,
                              int hidden_size, int intermediate_size, int ple_hidden,
                              int gsz,
                              float eps, float layer_scalar,
                              const void* hidden_in, void* hidden_out,
                              const void* pre_ff_norm_w,
                              const void* gate_proj_packed,
                              const void* gate_proj_scale,
                              const void* gate_proj_zero,
                              const void* up_proj_packed,
                              const void* up_proj_scale,
                              const void* up_proj_zero,
                              const void* down_proj_packed,
                              const void* down_proj_scale,
                              const void* down_proj_zero,
                              const void* post_ff_norm_w,
                              const void* per_layer_input,
                              const void* per_layer_input_gate_packed,
                              const void* per_layer_input_gate_scale,
                              const void* per_layer_input_gate_zero,
                              const void* per_layer_projection_packed,
                              const void* per_layer_projection_scale,
                              const void* per_layer_projection_zero,
                              const void* post_per_layer_input_norm_w,
                              void* workspace,
                              unsigned int* matvec_counter,
                              unsigned int* barrier_counter,
                              unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 631;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 632;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 633;

    g4_fused_mlp_ple_int4_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        hidden_size, intermediate_size, ple_hidden, gsz,
        eps, layer_scalar,
        static_cast<const T*>(hidden_in),
        static_cast<T*>(hidden_out),
        static_cast<const T*>(pre_ff_norm_w),
        static_cast<const uint8_t*>(gate_proj_packed),
        static_cast<const __nv_bfloat16*>(gate_proj_scale),
        static_cast<const __nv_bfloat16*>(gate_proj_zero),
        static_cast<const uint8_t*>(up_proj_packed),
        static_cast<const __nv_bfloat16*>(up_proj_scale),
        static_cast<const __nv_bfloat16*>(up_proj_zero),
        static_cast<const uint8_t*>(down_proj_packed),
        static_cast<const __nv_bfloat16*>(down_proj_scale),
        static_cast<const __nv_bfloat16*>(down_proj_zero),
        static_cast<const T*>(post_ff_norm_w),
        static_cast<const T*>(per_layer_input),
        static_cast<const uint8_t*>(per_layer_input_gate_packed),
        static_cast<const __nv_bfloat16*>(per_layer_input_gate_scale),
        static_cast<const __nv_bfloat16*>(per_layer_input_gate_zero),
        static_cast<const uint8_t*>(per_layer_projection_packed),
        static_cast<const __nv_bfloat16*>(per_layer_projection_scale),
        static_cast<const __nv_bfloat16*>(per_layer_projection_zero),
        static_cast<const T*>(post_per_layer_input_norm_w),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 634;
    if (cudaDeviceSynchronize() != cudaSuccess) return 635;
    return 0;
}

template <typename T>
int persistent_decode_device(int device_ordinal,
                             int num_layers, int hidden_size, int ple_hidden,
                             int position, float eps, float scale,
                             const void* layers,
                             const void* kv_fp8_descs,
                             const void* fp8_scales,
                             void* hidden_io, const void* per_layer_inputs,
                             void* workspace,
                             unsigned int* matvec_counter,
                             unsigned int* barrier_counter,
                             unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 701;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    // Allocate `BLOCK` floats for block-wide reductions plus `256` floats for
    // the FP8-runtime LUT (`fp8_lut[256]` populated at kernel entry only when
    // `fp8_scales` is non-null). The extra LDS is wasted in BF16 mode (1 KiB)
    // but stays a constant across modes — keeping the launch parameters
    // mode-agnostic avoids cross-contamination between BF16 and FP8 codegen.
    const size_t lds_bytes = (BLOCK + 256) * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 702;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 703;

    g4_persistent_decode_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        num_layers, hidden_size, ple_hidden, position, eps, scale,
        static_cast<const Gemma4DecodeLayerDesc*>(layers),
        static_cast<const Gemma4KVCacheFp8Desc*>(kv_fp8_descs),
        static_cast<const Gemma4FP8ScaleDesc*>(fp8_scales),
        static_cast<T*>(hidden_io),
        static_cast<const T*>(per_layer_inputs),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 704;
    if (cudaDeviceSynchronize() != cudaSuccess) return 705;
    return 0;
}

template <typename T>
int persistent_decode_batch_device(int device_ordinal,
                                   int num_layers, int hidden_size, int ple_hidden,
                                   float eps, float scale,
                                   int batch_size, int ws_stride,
                                   const void* layers,
                                   const void* batch_descs,
                                   void* hidden_io,
                                   const void* per_layer_inputs,
                                   void* workspace,
                                   unsigned int* matvec_counter,
                                   unsigned int* barrier_counter,
                                   unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 721;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 722;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 723;

    g4_persistent_decode_batch_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        num_layers, hidden_size, ple_hidden, eps, scale,
        batch_size, ws_stride,
        static_cast<const Gemma4DecodeLayerDesc*>(layers),
        static_cast<const Gemma4BatchSeqDesc*>(batch_descs),
        static_cast<T*>(hidden_io),
        static_cast<const T*>(per_layer_inputs),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 724;
    if (cudaDeviceSynchronize() != cudaSuccess) return 725;
    return 0;
}

template <typename T>
int persistent_decode_batch_int4_device(int device_ordinal,
                                        int num_layers, int hidden_size, int ple_hidden,
                                        float eps, float scale,
                                        int batch_size, int ws_stride,
                                        const void* layers,
                                        const void* int4_scales,
                                        const void* batch_descs,
                                        void* hidden_io,
                                        const void* per_layer_inputs,
                                        void* workspace,
                                        unsigned int* matvec_counter,
                                        unsigned int* barrier_counter,
                                        unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 731;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 732;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 733;

    g4_persistent_decode_batch_int4_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        num_layers, hidden_size, ple_hidden, eps, scale,
        batch_size, ws_stride,
        static_cast<const Gemma4DecodeLayerDesc*>(layers),
        static_cast<const Gemma4Int4ScaleDesc*>(int4_scales),
        static_cast<const Gemma4BatchSeqDesc*>(batch_descs),
        static_cast<T*>(hidden_io),
        static_cast<const T*>(per_layer_inputs),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 734;
    if (cudaDeviceSynchronize() != cudaSuccess) return 735;
    return 0;
}

template <typename T>
int persistent_decode_int4_device(int device_ordinal,
                                  int num_layers, int hidden_size, int ple_hidden,
                                  int position, float eps, float scale,
                                  const void* layers,
                                  const void* int4_scales,
                                  void* hidden_io,
                                  const void* per_layer_inputs,
                                  void* workspace,
                                  unsigned int* matvec_counter,
                                  unsigned int* barrier_counter,
                                  unsigned int* barrier_flag) {
    ScopedCudaDevice scoped(device_ordinal);

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) != cudaSuccess) return 711;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (cudaMemset(barrier_counter, 0, sizeof(unsigned int)) != cudaSuccess) return 712;
    if (cudaMemset(barrier_flag, 0, sizeof(unsigned int)) != cudaSuccess) return 713;

    g4_persistent_decode_int4_kernel<T><<<dim3(num_blocks), dim3(BLOCK), lds_bytes, 0>>>(
        num_layers, hidden_size, ple_hidden, position, eps, scale,
        static_cast<const Gemma4DecodeLayerDesc*>(layers),
        static_cast<const Gemma4Int4ScaleDesc*>(int4_scales),
        static_cast<T*>(hidden_io),
        static_cast<const T*>(per_layer_inputs),
        static_cast<float*>(workspace),
        matvec_counter, barrier_counter, barrier_flag);
    if (cudaGetLastError() != cudaSuccess) return 714;
    if (cudaDeviceSynchronize() != cudaSuccess) return 715;
    return 0;
}

template <typename T>
int gather_layer_slice_device(int device_ordinal,
                              int seq_len, int num_layers, int ple_hidden,
                              int layer_idx, const void* src, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0 || ple_hidden <= 0) return 0;
    constexpr int BLOCK = 256;
    dim3 grid((ple_hidden + BLOCK - 1) / BLOCK, seq_len, 1);
    dim3 block(BLOCK, 1, 1);
    g4_gather_layer_slice_kernel<T><<<grid, block, 0, 0>>>(
        seq_len, num_layers, ple_hidden, layer_idx,
        static_cast<const T*>(src),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 551;
    if (cudaDeviceSynchronize() != cudaSuccess) return 552;
    return 0;
}

template <typename T>
int embed_gather_scaled_device(int device_ordinal, int seq_len, int hidden_size,
                               int vocab_size, float scale,
                               const unsigned int* token_ids,
                               const void* table, void* out) {
    ScopedCudaDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    constexpr int BLOCK = 256;
    dim3 grid((hidden_size + BLOCK - 1) / BLOCK, seq_len, 1);
    dim3 block(BLOCK, 1, 1);
    g4_embed_gather_scaled_kernel<T><<<grid, block, 0, 0>>>(
        seq_len, hidden_size, vocab_size, scale,
        token_ids,
        static_cast<const T*>(table),
        static_cast<T*>(out));
    if (cudaGetLastError() != cudaSuccess) return 531;
    if (cudaDeviceSynchronize() != cudaSuccess) return 532;
    return 0;
}

}  // namespace

// -----------------------------------------------------------------------------
// extern "C" entry points — called from Rust via crate kernel-ffi.
// dtype encoding: 0 = __half (fp16), 1 = float (fp32), 2 = __nv_bfloat16
// -----------------------------------------------------------------------------

extern "C" int supersonic_gemma4_cuda_rms_norm(
    int dtype, size_t device_ordinal, size_t n_cols, float eps,
    const void* xs, const void* weight, void* out
) {
    switch (dtype) {
    case 0: return rms_norm_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 1: return rms_norm_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 2: return rms_norm_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    default: return 400;
    }
}

extern "C" int supersonic_gemma4_cuda_matvec(
    int dtype, size_t device_ordinal, size_t in_dim, size_t out_dim,
    const void* x, const void* W, void* out, unsigned int* row_counter
) {
    switch (dtype) {
    case 0: return matvec_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 1: return matvec_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 2: return matvec_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    default: return 409;
    }
}

extern "C" int supersonic_gemma4_cuda_gelu_tanh_gate_mul(
    int dtype, size_t device_ordinal, size_t n,
    const void* gate, const void* up, void* out
) {
    switch (dtype) {
    case 0: return gelu_tanh_gate_mul_device<__half>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 1: return gelu_tanh_gate_mul_device<float>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 2: return gelu_tanh_gate_mul_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    default: return 420;
    }
}

extern "C" int supersonic_gemma4_cuda_rope_decode(
    int dtype, size_t device_ordinal,
    size_t num_heads, size_t head_dim, size_t rotary_dim, size_t position,
    const void* cos_table, const void* sin_table, void* x
) {
    switch (dtype) {
    case 0: return rope_decode_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    case 1: return rope_decode_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    case 2: return rope_decode_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    default: return 430;
    }
}

extern "C" int supersonic_gemma4_cuda_swa_attn_decode(
    int dtype, size_t device_ordinal,
    size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t kv_len, size_t max_T,
    int sliding_window, float scale,
    const void* q, const void* k_cache, const void* v_cache,
    void* scores_scratch, void* out
) {
    switch (dtype) {
    case 0: return swa_attn_decode_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 1: return swa_attn_decode_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 2: return swa_attn_decode_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    default: return 449;
    }
}

extern "C" int supersonic_gemma4_cuda_kv_append(
    int dtype, size_t device_ordinal,
    size_t num_kv_heads, size_t head_dim, size_t pos, size_t max_T,
    const void* k_in, const void* v_in, void* k_cache, void* v_cache
) {
    switch (dtype) {
    case 0: return kv_append_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    case 1: return kv_append_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    case 2: return kv_append_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    default: return 440;
    }
}

// ------------------------ Prefill / batched entry points ---------------------

extern "C" int supersonic_gemma4_cuda_rms_norm_rows(
    int dtype, size_t device_ordinal, size_t n_rows, size_t n_cols, float eps,
    const void* xs, const void* weight, void* out
) {
    switch (dtype) {
    case 0: return rms_norm_rows_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(n_rows), static_cast<int>(n_cols), eps,
                xs, weight, out);
    case 1: return rms_norm_rows_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(n_rows), static_cast<int>(n_cols), eps,
                xs, weight, out);
    case 2: return rms_norm_rows_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(n_rows), static_cast<int>(n_cols), eps,
                xs, weight, out);
    default: return 460;
    }
}

extern "C" int supersonic_gemma4_cuda_matvec_batched(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t in_dim, size_t out_dim,
    const void* x, const void* W, void* out, unsigned int* counter
) {
    switch (dtype) {
    case 0: return matvec_batched_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(in_dim),
                static_cast<int>(out_dim), x, W, out, counter);
    case 1: return matvec_batched_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(in_dim),
                static_cast<int>(out_dim), x, W, out, counter);
    case 2: {
        const int ordinal = static_cast<int>(device_ordinal);
        const int s = static_cast<int>(seq_len);
        const int id = static_cast<int>(in_dim);
        const int od = static_cast<int>(out_dim);
        // WMMA pays off only once we have enough rows to fill a 16-row tile —
        // below that the work-stealing scalar kernel uses more of the wave.
        if (s >= 16 && g4_device_supports_wmma_bf16(ordinal)) {
            return matvec_batched_wmma_bf16_device(ordinal, s, id, od, x, W, out);
        }
        return matvec_batched_device<__nv_bfloat16>(ordinal, s, id, od, x, W, out, counter);
    }
    default: return 469;
    }
}

extern "C" int supersonic_gemma4_cuda_rope_prefill(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_heads, size_t head_dim,
    size_t rotary_dim, size_t pos_base,
    const void* cos_table, const void* sin_table, void* x
) {
    switch (dtype) {
    case 0: return rope_prefill_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(rotary_dim),
                static_cast<int>(pos_base), cos_table, sin_table, x);
    case 1: return rope_prefill_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(rotary_dim),
                static_cast<int>(pos_base), cos_table, sin_table, x);
    case 2: return rope_prefill_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(rotary_dim),
                static_cast<int>(pos_base), cos_table, sin_table, x);
    default: return 480;
    }
}

extern "C" int supersonic_gemma4_cuda_kv_append_prefill(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_kv_heads, size_t head_dim,
    size_t pos_base, size_t max_T,
    const void* k_in, const void* v_in, void* k_cache, void* v_cache
) {
    switch (dtype) {
    case 0: return kv_append_prefill_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(pos_base),
                static_cast<int>(max_T), k_in, v_in, k_cache, v_cache);
    case 1: return kv_append_prefill_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(pos_base),
                static_cast<int>(max_T), k_in, v_in, k_cache, v_cache);
    case 2: return kv_append_prefill_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(pos_base),
                static_cast<int>(max_T), k_in, v_in, k_cache, v_cache);
    default: return 490;
    }
}

extern "C" int supersonic_gemma4_cuda_attn_prefill(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t pos_base, size_t max_T,
    int sliding_window, float scale,
    const void* q, const void* k_cache, const void* v_cache,
    void* scores_scratch, void* out
) {
    switch (dtype) {
    case 0: return attn_prefill_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos_base), static_cast<int>(max_T),
                sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 1: return attn_prefill_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos_base), static_cast<int>(max_T),
                sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    case 2: return attn_prefill_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos_base), static_cast<int>(max_T),
                sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    default: return 499;
    }
}

extern "C" int supersonic_gemma4_cuda_add_residual(
    int dtype, size_t device_ordinal, size_t n,
    const void* a, const void* b, void* out
) {
    switch (dtype) {
    case 0: return add_residual_device<__half>(static_cast<int>(device_ordinal),
                n, a, b, out);
    case 1: return add_residual_device<float>(static_cast<int>(device_ordinal),
                n, a, b, out);
    case 2: return add_residual_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                n, a, b, out);
    default: return 510;
    }
}

extern "C" int supersonic_gemma4_cuda_add_scaled_residual(
    int dtype, size_t device_ordinal, size_t n, float scalar,
    const void* a, const void* b, void* out
) {
    switch (dtype) {
    case 0: return add_scaled_residual_device<__half>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    case 1: return add_scaled_residual_device<float>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    case 2: return add_scaled_residual_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    default: return 520;
    }
}

extern "C" int supersonic_gemma4_cuda_scalar_mul_inplace(
    int dtype, size_t device_ordinal, size_t n, float scalar, void* x
) {
    switch (dtype) {
    case 0: return scalar_mul_inplace_device<__half>(static_cast<int>(device_ordinal),
                n, scalar, x);
    case 1: return scalar_mul_inplace_device<float>(static_cast<int>(device_ordinal),
                n, scalar, x);
    case 2: return scalar_mul_inplace_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                n, scalar, x);
    default: return 540;
    }
}

extern "C" int supersonic_gemma4_cuda_fused_attn_block(
    int dtype, size_t device_ordinal,
    size_t hidden_size, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t rotary_dim, size_t position, size_t max_T,
    int sliding_window, int shared_kv, float eps, float scale,
    const void* hidden_in, void* hidden_out,
    const void* input_norm_w,
    const void* q_proj_w, const void* k_proj_w, const void* v_proj_w,
    const void* q_norm_w, const void* k_norm_w,
    const void* o_proj_w, const void* post_attn_norm_w,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_FUSED_ARGS                                                      \
        static_cast<int>(device_ordinal),                                      \
        static_cast<int>(hidden_size), static_cast<int>(num_q_heads),          \
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim),            \
        static_cast<int>(rotary_dim), sliding_window,                          \
        static_cast<int>(position), static_cast<int>(max_T), shared_kv,        \
        eps, scale,                                                             \
        hidden_in, hidden_out, input_norm_w,                                   \
        q_proj_w, k_proj_w, v_proj_w, q_norm_w, k_norm_w,                      \
        o_proj_w, post_attn_norm_w, cos_table, sin_table,                      \
        k_cache, v_cache, workspace,                                           \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return fused_attn_block_device<__half>(G4_FUSED_ARGS);
    case 1: return fused_attn_block_device<float>(G4_FUSED_ARGS);
    case 2: return fused_attn_block_device<__nv_bfloat16>(G4_FUSED_ARGS);
    default: return 600;
    }
    #undef G4_FUSED_ARGS
}

extern "C" int supersonic_gemma4_cuda_fused_attn_block_int4(
    int dtype, size_t device_ordinal,
    size_t hidden_size, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t rotary_dim, size_t position, size_t max_T,
    int sliding_window, int shared_kv, int group_size,
    float eps, float scale,
    const void* hidden_in, void* hidden_out,
    const void* input_norm_w,
    const void* q_proj_packed, const void* q_proj_scale, const void* q_proj_zero,
    const void* k_proj_packed, const void* k_proj_scale, const void* k_proj_zero,
    const void* v_proj_packed, const void* v_proj_scale, const void* v_proj_zero,
    const void* q_norm_w, const void* k_norm_w,
    const void* o_proj_packed, const void* o_proj_scale, const void* o_proj_zero,
    const void* post_attn_norm_w,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_FUSED_INT4_ARGS                                                   \
        static_cast<int>(device_ordinal),                                        \
        static_cast<int>(hidden_size), static_cast<int>(num_q_heads),            \
        static_cast<int>(num_kv_heads), static_cast<int>(head_dim),              \
        static_cast<int>(rotary_dim), sliding_window,                            \
        static_cast<int>(position), static_cast<int>(max_T), shared_kv,          \
        group_size, eps, scale,                                                  \
        hidden_in, hidden_out, input_norm_w,                                     \
        q_proj_packed, q_proj_scale, q_proj_zero,                                \
        k_proj_packed, k_proj_scale, k_proj_zero,                                \
        v_proj_packed, v_proj_scale, v_proj_zero,                                \
        q_norm_w, k_norm_w,                                                      \
        o_proj_packed, o_proj_scale, o_proj_zero, post_attn_norm_w,              \
        cos_table, sin_table, k_cache, v_cache, workspace,                       \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return fused_attn_block_int4_device<__half>(G4_FUSED_INT4_ARGS);
    case 1: return fused_attn_block_int4_device<float>(G4_FUSED_INT4_ARGS);
    case 2: return fused_attn_block_int4_device<__nv_bfloat16>(G4_FUSED_INT4_ARGS);
    default: return 620;
    }
    #undef G4_FUSED_INT4_ARGS
}

extern "C" int supersonic_gemma4_cuda_fused_mlp_ple(
    int dtype, size_t device_ordinal,
    size_t hidden_size, size_t intermediate_size, size_t ple_hidden,
    float eps, float layer_scalar,
    const void* hidden_in, void* hidden_out,
    const void* pre_ff_norm_w,
    const void* gate_proj_w, const void* up_proj_w,
    const void* down_proj_w, const void* post_ff_norm_w,
    const void* per_layer_input,
    const void* per_layer_input_gate_w,
    const void* per_layer_projection_w,
    const void* post_per_layer_input_norm_w,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_MLP_PLE_ARGS                                                    \
        static_cast<int>(device_ordinal),                                      \
        static_cast<int>(hidden_size), static_cast<int>(intermediate_size),    \
        static_cast<int>(ple_hidden), eps, layer_scalar,                       \
        hidden_in, hidden_out, pre_ff_norm_w, gate_proj_w, up_proj_w,          \
        down_proj_w, post_ff_norm_w, per_layer_input,                          \
        per_layer_input_gate_w, per_layer_projection_w,                        \
        post_per_layer_input_norm_w, workspace,                                \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return fused_mlp_ple_device<__half>(G4_MLP_PLE_ARGS);
    case 1: return fused_mlp_ple_device<float>(G4_MLP_PLE_ARGS);
    case 2: return fused_mlp_ple_device<__nv_bfloat16>(G4_MLP_PLE_ARGS);
    default: return 610;
    }
    #undef G4_MLP_PLE_ARGS
}

extern "C" int supersonic_gemma4_cuda_fused_mlp_ple_int4(
    int dtype, size_t device_ordinal,
    size_t hidden_size, size_t intermediate_size, size_t ple_hidden,
    int group_size,
    float eps, float layer_scalar,
    const void* hidden_in, void* hidden_out,
    const void* pre_ff_norm_w,
    const void* gate_proj_packed, const void* gate_proj_scale, const void* gate_proj_zero,
    const void* up_proj_packed, const void* up_proj_scale, const void* up_proj_zero,
    const void* down_proj_packed, const void* down_proj_scale, const void* down_proj_zero,
    const void* post_ff_norm_w,
    const void* per_layer_input,
    const void* per_layer_input_gate_packed,
    const void* per_layer_input_gate_scale,
    const void* per_layer_input_gate_zero,
    const void* per_layer_projection_packed,
    const void* per_layer_projection_scale,
    const void* per_layer_projection_zero,
    const void* post_per_layer_input_norm_w,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_MLP_PLE_INT4_ARGS                                                 \
        static_cast<int>(device_ordinal),                                        \
        static_cast<int>(hidden_size), static_cast<int>(intermediate_size),      \
        static_cast<int>(ple_hidden), group_size, eps, layer_scalar,             \
        hidden_in, hidden_out, pre_ff_norm_w,                                    \
        gate_proj_packed, gate_proj_scale, gate_proj_zero,                       \
        up_proj_packed, up_proj_scale, up_proj_zero,                             \
        down_proj_packed, down_proj_scale, down_proj_zero,                       \
        post_ff_norm_w, per_layer_input,                                         \
        per_layer_input_gate_packed, per_layer_input_gate_scale, per_layer_input_gate_zero, \
        per_layer_projection_packed, per_layer_projection_scale, per_layer_projection_zero, \
        post_per_layer_input_norm_w,                                             \
        workspace, matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return fused_mlp_ple_int4_device<__half>(G4_MLP_PLE_INT4_ARGS);
    case 1: return fused_mlp_ple_int4_device<float>(G4_MLP_PLE_INT4_ARGS);
    case 2: return fused_mlp_ple_int4_device<__nv_bfloat16>(G4_MLP_PLE_INT4_ARGS);
    default: return 630;
    }
    #undef G4_MLP_PLE_INT4_ARGS
}

extern "C" int supersonic_gemma4_cuda_gather_layer_slice(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_layers, size_t ple_hidden, size_t layer_idx,
    const void* src, void* out
) {
    switch (dtype) {
    case 0: return gather_layer_slice_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_layers),
                static_cast<int>(ple_hidden), static_cast<int>(layer_idx),
                src, out);
    case 1: return gather_layer_slice_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_layers),
                static_cast<int>(ple_hidden), static_cast<int>(layer_idx),
                src, out);
    case 2: return gather_layer_slice_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_layers),
                static_cast<int>(ple_hidden), static_cast<int>(layer_idx),
                src, out);
    default: return 550;
    }
}

extern "C" int supersonic_gemma4_cuda_embed_gather_scaled(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t hidden_size, size_t vocab_size, float scale,
    const unsigned int* token_ids, const void* table, void* out
) {
    switch (dtype) {
    case 0: return embed_gather_scaled_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(hidden_size),
                static_cast<int>(vocab_size), scale, token_ids, table, out);
    case 1: return embed_gather_scaled_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(hidden_size),
                static_cast<int>(vocab_size), scale, token_ids, table, out);
    case 2: return embed_gather_scaled_device<__nv_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(hidden_size),
                static_cast<int>(vocab_size), scale, token_ids, table, out);
    default: return 530;
    }
}

extern "C" int supersonic_gemma4_cuda_persistent_decode(
    int dtype, size_t device_ordinal,
    size_t num_layers, size_t hidden_size, size_t ple_hidden,
    size_t position, float eps, float scale,
    const void* layers,
    const void* kv_fp8_descs,
    const void* fp8_scales,
    void* hidden_io, const void* per_layer_inputs,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_PERSIST_ARGS                                                    \
        static_cast<int>(device_ordinal),                                      \
        static_cast<int>(num_layers), static_cast<int>(hidden_size),           \
        static_cast<int>(ple_hidden), static_cast<int>(position),              \
        eps, scale, layers, kv_fp8_descs, fp8_scales, hidden_io,               \
        per_layer_inputs, workspace,                                           \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return persistent_decode_device<__half>(G4_PERSIST_ARGS);
    case 1: return persistent_decode_device<float>(G4_PERSIST_ARGS);
    case 2: return persistent_decode_device<__nv_bfloat16>(G4_PERSIST_ARGS);
    default: return 700;
    }
    #undef G4_PERSIST_ARGS
}

extern "C" int supersonic_gemma4_cuda_persistent_decode_batch(
    int dtype, size_t device_ordinal,
    size_t num_layers, size_t hidden_size, size_t ple_hidden,
    float eps, float scale,
    size_t batch_size, size_t ws_stride,
    const void* layers,
    const void* batch_descs,
    void* hidden_io, const void* per_layer_inputs,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_PERSIST_BATCH_ARGS                                                \
        static_cast<int>(device_ordinal),                                        \
        static_cast<int>(num_layers), static_cast<int>(hidden_size),             \
        static_cast<int>(ple_hidden), eps, scale,                                \
        static_cast<int>(batch_size), static_cast<int>(ws_stride),               \
        layers, batch_descs, hidden_io, per_layer_inputs, workspace,             \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return persistent_decode_batch_device<__half>(G4_PERSIST_BATCH_ARGS);
    case 1: return persistent_decode_batch_device<float>(G4_PERSIST_BATCH_ARGS);
    case 2: return persistent_decode_batch_device<__nv_bfloat16>(G4_PERSIST_BATCH_ARGS);
    default: return 720;
    }
    #undef G4_PERSIST_BATCH_ARGS
}

extern "C" int supersonic_gemma4_cuda_persistent_decode_batch_int4(
    int dtype, size_t device_ordinal,
    size_t num_layers, size_t hidden_size, size_t ple_hidden,
    float eps, float scale,
    size_t batch_size, size_t ws_stride,
    const void* layers,
    const void* int4_scales,
    const void* batch_descs,
    void* hidden_io, const void* per_layer_inputs,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_PERSIST_BATCH_INT4_ARGS                                           \
        static_cast<int>(device_ordinal),                                        \
        static_cast<int>(num_layers), static_cast<int>(hidden_size),             \
        static_cast<int>(ple_hidden), eps, scale,                                \
        static_cast<int>(batch_size), static_cast<int>(ws_stride),               \
        layers, int4_scales, batch_descs, hidden_io, per_layer_inputs,           \
        workspace, matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return persistent_decode_batch_int4_device<__half>(G4_PERSIST_BATCH_INT4_ARGS);
    case 1: return persistent_decode_batch_int4_device<float>(G4_PERSIST_BATCH_INT4_ARGS);
    case 2: return persistent_decode_batch_int4_device<__nv_bfloat16>(G4_PERSIST_BATCH_INT4_ARGS);
    default: return 730;
    }
    #undef G4_PERSIST_BATCH_INT4_ARGS
}

extern "C" int supersonic_gemma4_cuda_persistent_decode_int4(
    int dtype, size_t device_ordinal,
    size_t num_layers, size_t hidden_size, size_t ple_hidden,
    size_t position, float eps, float scale,
    const void* layers,
    const void* int4_scales,
    void* hidden_io, const void* per_layer_inputs,
    void* workspace,
    unsigned int* matvec_counter,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag
) {
    #define G4_PERSIST_INT4_ARGS                                                 \
        static_cast<int>(device_ordinal),                                        \
        static_cast<int>(num_layers), static_cast<int>(hidden_size),             \
        static_cast<int>(ple_hidden), static_cast<int>(position),                \
        eps, scale, layers, int4_scales, hidden_io, per_layer_inputs, workspace, \
        matvec_counter, barrier_counter, barrier_flag
    switch (dtype) {
    case 0: return persistent_decode_int4_device<__half>(G4_PERSIST_INT4_ARGS);
    case 1: return persistent_decode_int4_device<float>(G4_PERSIST_INT4_ARGS);
    case 2: return persistent_decode_int4_device<__nv_bfloat16>(G4_PERSIST_INT4_ARGS);
    default: return 710;
    }
    #undef G4_PERSIST_INT4_ARGS
}

// INT4 matvec (single-token). Activation dtype is controlled by `dtype`;
// weight format is always (packed u8, bf16 scale, bf16 zero).
extern "C" int supersonic_gemma4_cuda_matvec_int4(
    int dtype, size_t device_ordinal,
    size_t in_dim, size_t out_dim, size_t group_size,
    const void* x,
    const void* W_packed, const void* W_scale, const void* W_zero,
    void* out, unsigned int* row_counter
) {
    switch (dtype) {
    case 0: return matvec_int4_device<__half>(
                static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim),
                static_cast<int>(group_size),
                x, W_packed, W_scale, W_zero, out, row_counter);
    case 1: return matvec_int4_device<float>(
                static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim),
                static_cast<int>(group_size),
                x, W_packed, W_scale, W_zero, out, row_counter);
    case 2: return matvec_int4_device<__nv_bfloat16>(
                static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim),
                static_cast<int>(group_size),
                x, W_packed, W_scale, W_zero, out, row_counter);
    default: return 489;
    }
}

extern "C" int supersonic_gemma4_cuda_matvec_batched_int4(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t in_dim, size_t out_dim, size_t group_size,
    const void* x,
    const void* W_packed, const void* W_scale, const void* W_zero,
    void* out, unsigned int* counter
) {
    switch (dtype) {
    case 0: return matvec_batched_int4_device<__half>(
                static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(in_dim),
                static_cast<int>(out_dim), static_cast<int>(group_size),
                x, W_packed, W_scale, W_zero, out, counter);
    case 1: return matvec_batched_int4_device<float>(
                static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(in_dim),
                static_cast<int>(out_dim), static_cast<int>(group_size),
                x, W_packed, W_scale, W_zero, out, counter);
    case 2: {
        const int ordinal = static_cast<int>(device_ordinal);
        const int s = static_cast<int>(seq_len);
        const int id = static_cast<int>(in_dim);
        const int od = static_cast<int>(out_dim);
        const int gs = static_cast<int>(group_size);
        // WMMA wants a 16-row tile minimum, and dequantizes each 16-K chunk
        // with one scale/zero fetch — only valid if gsz is a multiple of 16
        // (128 in the shipped Gemma 4 bake; weights.rs reads from metadata
        // so a custom bake could land something else, same guard as Qwen).
        if (s >= 16 && gs % 16 == 0 && g4_device_supports_wmma_bf16(ordinal)) {
            return matvec_batched_int4_wmma_bf16_device(
                ordinal, s, id, od, gs, x, W_packed, W_scale, W_zero, out);
        }
        return matvec_batched_int4_device<__nv_bfloat16>(
            ordinal, s, id, od, gs, x, W_packed, W_scale, W_zero, out, counter);
    }
    default: return 499;
    }
}
