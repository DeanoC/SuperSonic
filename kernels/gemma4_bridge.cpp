// Bridge for Gemma 4 decode primitives. Separate compilation unit from the
// Qwen kernels — see gemma4.hip for rationale.

#include "gemma4.hip"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) { hipSetDevice(target); changed = true; }
    }
    ~ScopedHipDevice() { if (changed && previous >= 0) hipSetDevice(previous); }
};

// ---- RMSNorm (Gemma variant: no (w+1) offset) ----

template <typename T>
int rms_norm_device(int device_ordinal, int n_cols, float eps,
                    const void* xs, const void* weight, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rms_norm_kernel<T>),
        dim3(1), dim3(block), 0, 0,
        n_cols, eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 401;
    if (hipDeviceSynchronize() != hipSuccess) return 402;
    return 0;
}

// ---- Work-stealing matvec ----

template <typename T>
int matvec_device(int device_ordinal, int in_dim, int out_dim,
                  const void* x, const void* W, void* out,
                  unsigned int* row_counter) {
    ScopedHipDevice scoped(device_ordinal);
    if (hipMemset(row_counter, 0, sizeof(unsigned int)) != hipSuccess) return 410;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 411;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_matvec_workstealing_kernel<T>),
        dim3(num_blocks), dim3(block), 0, 0,
        in_dim, out_dim,
        static_cast<const T*>(x),
        static_cast<const T*>(W),
        static_cast<T*>(out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 412;
    if (hipDeviceSynchronize() != hipSuccess) return 413;
    return 0;
}

// ---- GeLU-tanh gated multiply ----

template <typename T>
int gelu_tanh_gate_mul_device(int device_ordinal, size_t n,
                              const void* gate, const void* up, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_gelu_tanh_gate_mul_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        n,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 421;
    if (hipDeviceSynchronize() != hipSuccess) return 422;
    return 0;
}

// ---- RoPE decode (split-half Gemma style) ----

template <typename T>
int rope_decode_device(int device_ordinal,
                       int num_heads, int head_dim, int rotary_dim, int position,
                       const void* cos_table, const void* sin_table, void* x) {
    ScopedHipDevice scoped(device_ordinal);
    const int half = rotary_dim / 2;
    const size_t total = static_cast<size_t>(num_heads) * half;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rope_split_half_decode_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        num_heads, head_dim, rotary_dim, position,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(x));
    if (hipGetLastError() != hipSuccess) return 431;
    if (hipDeviceSynchronize() != hipSuccess) return 432;
    return 0;
}

// ---- KV append ----

template <typename T>
int kv_append_device(int device_ordinal,
                     int num_kv_heads, int head_dim, int pos, int max_T,
                     const void* k_in, const void* v_in,
                     void* k_cache, void* v_cache) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(num_kv_heads) * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_kv_append_decode_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        num_kv_heads, head_dim, pos, max_T,
        static_cast<const T*>(k_in),
        static_cast<const T*>(v_in),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache));
    if (hipGetLastError() != hipSuccess) return 441;
    if (hipDeviceSynchronize() != hipSuccess) return 442;
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
    ScopedHipDevice scoped(device_ordinal);
    if (kv_len <= 0) return 450;

    constexpr int BLOCK = 256;
    // Phase 1: per-(q_head, t) score computation.
    {
        dim3 grid(num_q_heads, (kv_len + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_scores_kernel<T>),
            grid, block, 0, 0,
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            sliding_window, scale,
            static_cast<const T*>(q),
            static_cast<const T*>(k_cache),
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 451;
    }

    // Phase 2: per-q_head softmax.
    {
        dim3 grid(num_q_heads, 1, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            g4_attn_softmax_kernel,
            grid, block, 0, 0,
            num_q_heads, kv_len, max_T,
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 452;
    }

    // Phase 3: value aggregation per (q_head, head_dim col).
    {
        dim3 grid(num_q_heads, (head_dim + BLOCK - 1) / BLOCK, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_value_aggregate_kernel<T>),
            grid, block, 0, 0,
            num_q_heads, num_kv_heads, head_dim, kv_len, max_T,
            static_cast<const float*>(scores_scratch),
            static_cast<const T*>(v_cache),
            static_cast<T*>(out));
        if (hipGetLastError() != hipSuccess) return 453;
    }

    if (hipDeviceSynchronize() != hipSuccess) return 454;
    return 0;
}

// =============================================================================
// Prefill / batched device wrappers (Step 13).
// Mirror the single-token device wrappers above but accept a `seq_len`
// (or `n_rows`) dimension so the kernel launch covers all prompt tokens at
// once. Same dtype dispatch pattern (__half / float / hip_bfloat16).
// =============================================================================

template <typename T>
int rms_norm_rows_device(int device_ordinal, int n_rows, int n_cols, float eps,
                         const void* xs, const void* weight, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (n_rows <= 0) return 0;
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rms_norm_rows_kernel<T>),
        dim3(n_rows), dim3(block), 0, 0,
        n_rows, n_cols, eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 461;
    if (hipDeviceSynchronize() != hipSuccess) return 462;
    return 0;
}

template <typename T>
int matvec_batched_device(int device_ordinal,
                          int seq_len, int in_dim, int out_dim,
                          const void* x, const void* W, void* out,
                          unsigned int* counter) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    if (hipMemset(counter, 0, sizeof(unsigned int)) != hipSuccess) return 470;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 471;
    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int block = 256;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_matvec_batched_kernel<T>),
        dim3(num_blocks), dim3(block), 0, 0,
        seq_len, in_dim, out_dim,
        static_cast<const T*>(x),
        static_cast<const T*>(W),
        static_cast<T*>(out),
        counter);
    if (hipGetLastError() != hipSuccess) return 472;
    if (hipDeviceSynchronize() != hipSuccess) return 473;
    return 0;
}

template <typename T>
int rope_prefill_device(int device_ordinal,
                        int seq_len, int num_heads, int head_dim,
                        int rotary_dim, int pos_base,
                        const void* cos_table, const void* sin_table, void* x) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const int half = rotary_dim / 2;
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half;
    if (total == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_rope_prefill_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, num_heads, head_dim, rotary_dim, pos_base,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(x));
    if (hipGetLastError() != hipSuccess) return 481;
    if (hipDeviceSynchronize() != hipSuccess) return 482;
    return 0;
}

template <typename T>
int kv_append_prefill_device(int device_ordinal,
                             int seq_len, int num_kv_heads, int head_dim,
                             int pos_base, int max_T,
                             const void* k_in, const void* v_in,
                             void* k_cache, void* v_cache) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const size_t total = static_cast<size_t>(seq_len) * num_kv_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_kv_append_prefill_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, num_kv_heads, head_dim, pos_base, max_T,
        static_cast<const T*>(k_in),
        static_cast<const T*>(v_in),
        static_cast<T*>(k_cache),
        static_cast<T*>(v_cache));
    if (hipGetLastError() != hipSuccess) return 491;
    if (hipDeviceSynchronize() != hipSuccess) return 492;
    return 0;
}

template <typename T>
int attn_prefill_device(int device_ordinal,
                        int seq_len, int num_q_heads, int num_kv_heads,
                        int head_dim, int pos_base, int max_T,
                        int sliding_window, float scale,
                        const void* q, const void* k_cache, const void* v_cache,
                        void* scores_scratch, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    const int kv_total = pos_base + seq_len;
    if (kv_total <= 0) return 500;

    constexpr int BLOCK = 256;
    {
        dim3 grid(num_q_heads, (kv_total + BLOCK - 1) / BLOCK, seq_len);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_prefill_scores_kernel<T>),
            grid, block, 0, 0,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            pos_base, max_T, sliding_window, scale,
            static_cast<const T*>(q),
            static_cast<const T*>(k_cache),
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 501;
    }
    {
        dim3 grid(num_q_heads, seq_len, 1);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            g4_attn_prefill_softmax_kernel,
            grid, block, 0, 0,
            seq_len, num_q_heads, kv_total, max_T,
            static_cast<float*>(scores_scratch));
        if (hipGetLastError() != hipSuccess) return 502;
    }
    {
        dim3 grid(num_q_heads, (head_dim + BLOCK - 1) / BLOCK, seq_len);
        dim3 block(BLOCK, 1, 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(g4_attn_prefill_value_kernel<T>),
            grid, block, 0, 0,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            pos_base, max_T,
            static_cast<const float*>(scores_scratch),
            static_cast<const T*>(v_cache),
            static_cast<T*>(out));
        if (hipGetLastError() != hipSuccess) return 503;
    }

    if (hipDeviceSynchronize() != hipSuccess) return 504;
    return 0;
}

template <typename T>
int add_residual_device(int device_ordinal, size_t n,
                        const void* a, const void* b, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_add_residual_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        n,
        static_cast<const T*>(a),
        static_cast<const T*>(b),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 511;
    if (hipDeviceSynchronize() != hipSuccess) return 512;
    return 0;
}

template <typename T>
int add_scaled_residual_device(int device_ordinal, size_t n, float scalar,
                               const void* a, const void* b, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_add_scaled_residual_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        n, scalar,
        static_cast<const T*>(a),
        static_cast<const T*>(b),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 521;
    if (hipDeviceSynchronize() != hipSuccess) return 522;
    return 0;
}

template <typename T>
int scalar_mul_inplace_device(int device_ordinal, size_t n, float scalar, void* x) {
    ScopedHipDevice scoped(device_ordinal);
    if (n == 0) return 0;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_scalar_mul_inplace_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        n, scalar, static_cast<T*>(x));
    if (hipGetLastError() != hipSuccess) return 541;
    if (hipDeviceSynchronize() != hipSuccess) return 542;
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
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 601;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;

    // LDS: bs slots for reduction scratch (shared with softmax/matmul). Single
    // scratch pool — the kernel reuses it across phases.
    const size_t lds_bytes = BLOCK * sizeof(float);

    // Zero the barrier counter / flag before launch. We leave matvec_counter
    // to be reset by the kernel itself (it's cleared twice during execution).
    if (hipMemset(barrier_counter, 0, sizeof(unsigned int)) != hipSuccess) return 602;
    if (hipMemset(barrier_flag, 0, sizeof(unsigned int)) != hipSuccess) return 603;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_fused_attn_block_kernel<T>),
        dim3(num_blocks), dim3(BLOCK), lds_bytes, 0,
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
    if (hipGetLastError() != hipSuccess) return 604;
    if (hipDeviceSynchronize() != hipSuccess) return 605;
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
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 611;
    const int num_blocks =
        props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    constexpr int BLOCK = 256;
    const size_t lds_bytes = BLOCK * sizeof(float);

    if (hipMemset(barrier_counter, 0, sizeof(unsigned int)) != hipSuccess) return 612;
    if (hipMemset(barrier_flag, 0, sizeof(unsigned int)) != hipSuccess) return 613;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_fused_mlp_ple_kernel<T>),
        dim3(num_blocks), dim3(BLOCK), lds_bytes, 0,
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
    if (hipGetLastError() != hipSuccess) return 614;
    if (hipDeviceSynchronize() != hipSuccess) return 615;
    return 0;
}

template <typename T>
int gather_layer_slice_device(int device_ordinal,
                              int seq_len, int num_layers, int ple_hidden,
                              int layer_idx, const void* src, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0 || ple_hidden <= 0) return 0;
    constexpr int BLOCK = 256;
    dim3 grid((ple_hidden + BLOCK - 1) / BLOCK, seq_len, 1);
    dim3 block(BLOCK, 1, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_gather_layer_slice_kernel<T>),
        grid, block, 0, 0,
        seq_len, num_layers, ple_hidden, layer_idx,
        static_cast<const T*>(src),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 551;
    if (hipDeviceSynchronize() != hipSuccess) return 552;
    return 0;
}

template <typename T>
int embed_gather_scaled_device(int device_ordinal, int seq_len, int hidden_size,
                               int vocab_size, float scale,
                               const unsigned int* token_ids,
                               const void* table, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    if (seq_len <= 0) return 0;
    constexpr int BLOCK = 256;
    dim3 grid((hidden_size + BLOCK - 1) / BLOCK, seq_len, 1);
    dim3 block(BLOCK, 1, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(g4_embed_gather_scaled_kernel<T>),
        grid, block, 0, 0,
        seq_len, hidden_size, vocab_size, scale,
        token_ids,
        static_cast<const T*>(table),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 531;
    if (hipDeviceSynchronize() != hipSuccess) return 532;
    return 0;
}

}  // namespace

// -----------------------------------------------------------------------------
// extern "C" entry points — called from Rust via crate kernel-ffi.
// dtype encoding: 0 = __half (fp16), 1 = float (fp32), 2 = hip_bfloat16
// -----------------------------------------------------------------------------

extern "C" int dotcache_gemma4_hip_rms_norm(
    int dtype, size_t device_ordinal, size_t n_cols, float eps,
    const void* xs, const void* weight, void* out
) {
    switch (dtype) {
    case 0: return rms_norm_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 1: return rms_norm_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    case 2: return rms_norm_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(n_cols), eps, xs, weight, out);
    default: return 400;
    }
}

extern "C" int dotcache_gemma4_hip_matvec(
    int dtype, size_t device_ordinal, size_t in_dim, size_t out_dim,
    const void* x, const void* W, void* out, unsigned int* row_counter
) {
    switch (dtype) {
    case 0: return matvec_device<__half>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 1: return matvec_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    case 2: return matvec_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(in_dim), static_cast<int>(out_dim), x, W, out, row_counter);
    default: return 409;
    }
}

extern "C" int dotcache_gemma4_hip_gelu_tanh_gate_mul(
    int dtype, size_t device_ordinal, size_t n,
    const void* gate, const void* up, void* out
) {
    switch (dtype) {
    case 0: return gelu_tanh_gate_mul_device<__half>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 1: return gelu_tanh_gate_mul_device<float>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    case 2: return gelu_tanh_gate_mul_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                n, gate, up, out);
    default: return 420;
    }
}

extern "C" int dotcache_gemma4_hip_rope_decode(
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
    case 2: return rope_decode_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_heads), static_cast<int>(head_dim),
                static_cast<int>(rotary_dim), static_cast<int>(position),
                cos_table, sin_table, x);
    default: return 430;
    }
}

extern "C" int dotcache_gemma4_hip_swa_attn_decode(
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
    case 2: return swa_attn_decode_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(kv_len),
                static_cast<int>(max_T), sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    default: return 449;
    }
}

extern "C" int dotcache_gemma4_hip_kv_append(
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
    case 2: return kv_append_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos), static_cast<int>(max_T),
                k_in, v_in, k_cache, v_cache);
    default: return 440;
    }
}

// ------------------------ Prefill / batched entry points ---------------------

extern "C" int dotcache_gemma4_hip_rms_norm_rows(
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
    case 2: return rms_norm_rows_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(n_rows), static_cast<int>(n_cols), eps,
                xs, weight, out);
    default: return 460;
    }
}

extern "C" int dotcache_gemma4_hip_matvec_batched(
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
    case 2: return matvec_batched_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(in_dim),
                static_cast<int>(out_dim), x, W, out, counter);
    default: return 469;
    }
}

extern "C" int dotcache_gemma4_hip_rope_prefill(
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
    case 2: return rope_prefill_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(rotary_dim),
                static_cast<int>(pos_base), cos_table, sin_table, x);
    default: return 480;
    }
}

extern "C" int dotcache_gemma4_hip_kv_append_prefill(
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
    case 2: return kv_append_prefill_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim), static_cast<int>(pos_base),
                static_cast<int>(max_T), k_in, v_in, k_cache, v_cache);
    default: return 490;
    }
}

extern "C" int dotcache_gemma4_hip_attn_prefill(
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
    case 2: return attn_prefill_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                static_cast<int>(pos_base), static_cast<int>(max_T),
                sliding_window, scale,
                q, k_cache, v_cache, scores_scratch, out);
    default: return 499;
    }
}

extern "C" int dotcache_gemma4_hip_add_residual(
    int dtype, size_t device_ordinal, size_t n,
    const void* a, const void* b, void* out
) {
    switch (dtype) {
    case 0: return add_residual_device<__half>(static_cast<int>(device_ordinal),
                n, a, b, out);
    case 1: return add_residual_device<float>(static_cast<int>(device_ordinal),
                n, a, b, out);
    case 2: return add_residual_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                n, a, b, out);
    default: return 510;
    }
}

extern "C" int dotcache_gemma4_hip_add_scaled_residual(
    int dtype, size_t device_ordinal, size_t n, float scalar,
    const void* a, const void* b, void* out
) {
    switch (dtype) {
    case 0: return add_scaled_residual_device<__half>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    case 1: return add_scaled_residual_device<float>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    case 2: return add_scaled_residual_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                n, scalar, a, b, out);
    default: return 520;
    }
}

extern "C" int dotcache_gemma4_hip_scalar_mul_inplace(
    int dtype, size_t device_ordinal, size_t n, float scalar, void* x
) {
    switch (dtype) {
    case 0: return scalar_mul_inplace_device<__half>(static_cast<int>(device_ordinal),
                n, scalar, x);
    case 1: return scalar_mul_inplace_device<float>(static_cast<int>(device_ordinal),
                n, scalar, x);
    case 2: return scalar_mul_inplace_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                n, scalar, x);
    default: return 540;
    }
}

extern "C" int dotcache_gemma4_hip_fused_attn_block(
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
    case 2: return fused_attn_block_device<hip_bfloat16>(G4_FUSED_ARGS);
    default: return 600;
    }
    #undef G4_FUSED_ARGS
}

extern "C" int dotcache_gemma4_hip_fused_mlp_ple(
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
    case 2: return fused_mlp_ple_device<hip_bfloat16>(G4_MLP_PLE_ARGS);
    default: return 610;
    }
    #undef G4_MLP_PLE_ARGS
}

extern "C" int dotcache_gemma4_hip_gather_layer_slice(
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
    case 2: return gather_layer_slice_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_layers),
                static_cast<int>(ple_hidden), static_cast<int>(layer_idx),
                src, out);
    default: return 550;
    }
}

extern "C" int dotcache_gemma4_hip_embed_gather_scaled(
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
    case 2: return embed_gather_scaled_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(hidden_size),
                static_cast<int>(vocab_size), scale, token_ids, table, out);
    default: return 530;
    }
}
