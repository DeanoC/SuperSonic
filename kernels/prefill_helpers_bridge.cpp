// Bridge for prefill helper kernels.
// Separate compilation unit — does not touch the decode megakernel files.

#include "prefill_helpers.hip"

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

hipError_t maybe_sync() {
    const char* value = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    const bool enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
    return enabled ? hipDeviceSynchronize() : hipSuccess;
}

int backend_failure(int project_status, hipError_t native_status) {
    return static_cast<int>(
        0x80000000u
        | ((static_cast<uint32_t>(project_status) & 0x7fffu) << 16)
        | (static_cast<uint32_t>(native_status) & 0xffffu));
}

int launch_result(int launch_project_status, int sync_project_status) {
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        return backend_failure(launch_project_status, launch_status);
    }
    const hipError_t sync_status = maybe_sync();
    if (sync_status != hipSuccess) {
        return backend_failure(sync_project_status, sync_status);
    }
    return 0;
}

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) { hipSetDevice(target); changed = true; }
    }
    ~ScopedHipDevice() { if (changed && previous >= 0) hipSetDevice(previous); }
};

// ---- element_add ----

template <typename T>
int element_add_device(int device_ordinal, size_t total_elems,
                       const void* lhs, const void* rhs, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_element_add_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        total_elems,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    return launch_result(301, 302);
}

int argmax_bf16_rows_device(
    int device_ordinal,
    size_t rows,
    size_t cols,
    const void* logits,
    void* out_index
) {
    ScopedHipDevice scoped(device_ordinal);
    if (rows == 0 || cols == 0) return 331;
    constexpr int block = 256;
    hipLaunchKernelGGL(
        pfx_argmax_bf16_rows_kernel,
        dim3(static_cast<unsigned int>(rows)),
        dim3(block),
        0,
        0,
        rows,
        cols,
        static_cast<const hip_bfloat16*>(logits),
        static_cast<uint32_t*>(out_index));
    return launch_result(332, 333);
}

// ---- apply_rope_prefill ----

template <typename T>
int apply_rope_prefill_device(int device_ordinal,
                              int seq_len, int num_heads, int head_dim, int half_rot,
                              const void* cos_table, const void* sin_table, void* data) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_apply_rope_prefill_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, num_heads, head_dim, half_rot,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        static_cast<T*>(data));
    return launch_result(311, 312);
}

// ---- apply_rope_prefill_indirect (SpecPrefill — arXiv 2502.02789) ----

template <typename T>
int apply_rope_prefill_indirect_device(int device_ordinal,
                                       int seq_len, int num_heads, int head_dim, int half_rot,
                                       const void* cos_table, const void* sin_table,
                                       const int* pos_ids, void* data) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * num_heads * half_rot;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_apply_rope_prefill_indirect_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, num_heads, head_dim, half_rot,
        static_cast<const T*>(cos_table),
        static_cast<const T*>(sin_table),
        pos_ids,
        static_cast<T*>(data));
    return launch_result(313, 314);
}

// ---- lookahead_attention_scores (SpecPrefill — arXiv 2502.02789) ----

template <typename T>
int lookahead_attention_scores_device(int device_ordinal,
                                      int q_heads, int kv_heads,
                                      int lookahead_count, int kv_len, int head_dim,
                                      float scale,
                                      const void* q, const void* k, void* scores) {
    if (q_heads <= 0 || kv_heads <= 0 || lookahead_count <= 0 || kv_len <= 0 || head_dim <= 0) {
        return 318; // invalid shape
    }
    if (q_heads % kv_heads != 0) {
        return 319; // q_heads must be a multiple of kv_heads
    }
    ScopedHipDevice scoped(device_ordinal);
    // Query the device's wavefront size so we launch exactly one full
    // wave: 32 threads on wave32 (gfx1100, RDNA3), 64 on wave64
    // (gfx9xx, RDNA1/2). This avoids divergent __syncthreads()
    // participation that would happen with a fixed block size when
    // the kernel uses `if (lane >= warpSize) return;` followed by
    // block-wide barriers.
    hipDeviceProp_t prop;
    const hipError_t properties_status = hipGetDeviceProperties(&prop, device_ordinal);
    if (properties_status != hipSuccess) {
        return backend_failure(323, properties_status);
    }
    const int wave = prop.warpSize;
    if (wave != 32 && wave != 64) {
        return 324; // unexpected wavefront size
    }
    const int num_kv_groups = q_heads / kv_heads;
    const dim3 grid(static_cast<unsigned int>(q_heads * lookahead_count));
    const dim3 block(static_cast<unsigned int>(wave));
    const size_t shared_bytes = static_cast<size_t>(kv_len) * sizeof(float);
    // Per-block dynamic shared memory cap. gfx1100 has 64 KiB LDS but
    // some is used by other allocations; 32 KiB is a conservative bound
    // that gives ~8000 tokens of headroom (32768 / 4 = 8192). Long
    // prompts above this limit need a tiled / online-softmax kernel
    // that doesn't store the per-row exponentials in LDS — out of
    // scope for Phase C. Bail loudly so the caller gets a clear error
    // instead of a runtime kernel-launch failure.
    constexpr size_t kMaxSharedBytes = 32 * 1024;
    if (shared_bytes > kMaxSharedBytes) {
        return 325; // kv_len exceeds shared-mem budget
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_lookahead_attention_scores_kernel<T>),
        grid, block, shared_bytes, 0,
        q_heads, kv_heads, lookahead_count, kv_len, head_dim, num_kv_groups,
        scale,
        static_cast<const T*>(q),
        static_cast<const T*>(k),
        static_cast<float*>(scores));
    return launch_result(316, 317);
}

// ---- pflash_cosine_score (SpecPrefill — Phase D PFlash-style scoring) ----

template <typename T>
int pflash_cosine_score_device(int device_ordinal,
                               int n_pos, int kv_heads, int cap, int head_dim,
                               int block_size, int n_blocks, int last_pos,
                               const void* k_cache, void* scores) {
    if (n_pos <= 0 || kv_heads <= 0 || cap <= 0 || head_dim <= 0
        || block_size <= 0 || n_blocks <= 0) {
        return 326; // invalid shape
    }
    if (last_pos < 0 || last_pos >= n_pos || cap < n_pos) {
        return 327; // out-of-range positions
    }
    ScopedHipDevice scoped(device_ordinal);
    // Query the device's wavefront size at runtime — same pattern as
    // lookahead_attention_scores_device. wave32 on gfx1100 (RDNA3),
    // wave64 on gfx9xx/RDNA1/RDNA2.
    hipDeviceProp_t prop;
    const hipError_t properties_status = hipGetDeviceProperties(&prop, device_ordinal);
    if (properties_status != hipSuccess) {
        return backend_failure(328, properties_status);
    }
    const int wave = prop.warpSize;
    if (wave != 32 && wave != 64) {
        return 333; // unexpected wavefront size
    }
    const dim3 grid(static_cast<unsigned int>(n_blocks));
    const dim3 block(static_cast<unsigned int>(wave));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_pflash_cosine_score_kernel<T>),
        grid, block, 0, 0,
        static_cast<const T*>(k_cache),
        static_cast<float*>(scores),
        n_pos, kv_heads, cap, head_dim, block_size, n_blocks, last_pos);
    return launch_result(334, 335);
}

// ---- transpose [S,H,D] -> [H,S,D] ----

template <typename T>
int transpose_shd_hsd_device(int device_ordinal,
                             int S, int H, int D,
                             const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * H * D;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_shd_hsd_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, H, D,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    return launch_result(321, 322);
}

template <typename T>
int transpose_shd_hsd_pair_device(int device_ordinal,
                                  int S, int H, int D,
                                  const void* src_a, const void* src_b,
                                  void* dst_a, void* dst_b) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * H * D;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_shd_hsd_pair_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, H, D,
        static_cast<const T*>(src_a),
        static_cast<const T*>(src_b),
        static_cast<T*>(dst_a),
        static_cast<T*>(dst_b));
    return launch_result(323, 324);
}

int transpose_shd_to_cache_bf16_device(int device_ordinal,
                                       int S, int H, int D,
                                       int cache_len,
                                       int dst_pos,
                                       const void* src,
                                       void* cache) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * H * D;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_transpose_shd_to_cache_bf16_kernel,
        dim3(grid), dim3(block), 0, 0,
        S, H, D, cache_len, dst_pos,
        static_cast<const hip_bfloat16*>(src),
        static_cast<hip_bfloat16*>(cache));
    return launch_result(323, 324);
}

// ---- transpose + pad for conv ----

template <typename T>
int transpose_pad_conv_device(int device_ordinal,
                              int S, int C, int pad,
                              const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    // Zero the entire dst buffer first (to get zero-padding)
    const size_t dst_bytes = static_cast<size_t>(C) * (pad + S) * sizeof(T);
    const hipError_t memset_status = hipMemset(dst, 0, dst_bytes);
    if (memset_status != hipSuccess) return backend_failure(330, memset_status);

    const size_t total = static_cast<size_t>(S) * C;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_transpose_pad_conv_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, pad,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    return launch_result(331, 332);
}

// ---- extract conv state ----

template <typename T>
int extract_conv_state_device(int device_ordinal,
                              int S, int C, int kern_minus_1,
                              const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(kern_minus_1) * C;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_extract_conv_state_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, kern_minus_1,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    return launch_result(341, 342);
}

// ---- prepare conv input and next tail ----

template <typename T>
int prepare_conv_input_tail_device(int device_ordinal,
                                   int S, int C, int pad,
                                   const void* src,
                                   const void* old_tail,
                                   void* conv_input,
                                   void* new_tail) {
    ScopedHipDevice scoped(device_ordinal);
    if (S < pad) return 343;
    const size_t conv_total = static_cast<size_t>(C) * static_cast<size_t>(pad + S);
    const size_t tail_total = static_cast<size_t>(C) * static_cast<size_t>(pad);
    const size_t total = conv_total > tail_total ? conv_total : tail_total;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_prepare_conv_input_tail_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, C, pad,
        static_cast<const T*>(src),
        static_cast<const T*>(old_tail),
        static_cast<T*>(conv_input),
        static_cast<T*>(new_tail));
    return launch_result(344, 345);
}

// ---- sigmoid_mul ----

template <typename T>
int sigmoid_mul_device(int device_ordinal, size_t total_elems,
                       const void* data, const void* gate, void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_sigmoid_mul_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        total_elems,
        static_cast<const T*>(data),
        static_cast<const T*>(gate),
        static_cast<T*>(out));
    return launch_result(351, 352);
}

int cast_transpose_gate_bf16_device(int device_ordinal,
                                    int S, int H, int D,
                                    const void* attn_hsd,
                                    const void* gate_shd,
                                    void* out_shd) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * H * D;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_cast_transpose_gate_hsd_to_shd_bf16_kernel,
        dim3(grid), dim3(block), 0, 0,
        S, H, D,
        static_cast<const float*>(attn_hsd),
        static_cast<const hip_bfloat16*>(gate_shd),
        static_cast<hip_bfloat16*>(out_shd));
    return launch_result(353, 354);
}

// ---- compute_beta_g ----

template <typename T>
int compute_beta_g_device(int device_ordinal,
                          int seq_len, int nv,
                          const void* B, const void* A,
                          const void* dt_bias, const void* a_log_exp,
                          void* beta, void* g) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * nv;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_compute_beta_g_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        seq_len, nv,
        static_cast<const T*>(B),
        static_cast<const T*>(A),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(beta),
        static_cast<T*>(g));
    return launch_result(361, 362);
}

int compute_beta_g_ba_bf16_device(int device_ordinal,
                                  int seq_len, int nv,
                                  const void* BA,
                                  const void* dt_bias,
                                  const void* a_log_exp,
                                  void* beta, void* g) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(seq_len) * nv;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_compute_beta_g_ba_bf16_kernel,
        dim3(grid), dim3(block), 0, 0,
        seq_len, nv,
        static_cast<const hip_bfloat16*>(BA),
        static_cast<const hip_bfloat16*>(dt_bias),
        static_cast<const hip_bfloat16*>(a_log_exp),
        static_cast<float*>(beta),
        static_cast<float*>(g));
    return launch_result(363, 364);
}

int project_ba_compute_beta_g_bf16_device(int device_ordinal,
                                          int seq_len, int hidden_dim, int nv,
                                          const void* hidden,
                                          const void* ba_weight,
                                          const void* dt_bias,
                                          const void* a_log_exp,
                                          void* beta, void* g) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = PFX_BA_TILE_M * PFX_BA_TILE_N;
    const unsigned int grid_x =
        static_cast<unsigned int>((2 * nv + PFX_BA_TILE_N - 1) / PFX_BA_TILE_N);
    const unsigned int grid_y =
        static_cast<unsigned int>((seq_len + PFX_BA_TILE_M - 1) / PFX_BA_TILE_M);
    hipLaunchKernelGGL(
        pfx_project_ba_compute_beta_g_bf16_kernel,
        dim3(grid_x, grid_y), dim3(block), 0, 0,
        seq_len, hidden_dim, nv,
        static_cast<const hip_bfloat16*>(hidden),
        static_cast<const hip_bfloat16*>(ba_weight),
        static_cast<const hip_bfloat16*>(dt_bias),
        static_cast<const hip_bfloat16*>(a_log_exp),
        static_cast<float*>(beta),
        static_cast<float*>(g));
    return launch_result(365, 366);
}

// ---- split_qgate ----

template <typename T>
int split_qgate_device(int device_ordinal,
                       int S, int num_heads, int head_dim,
                       const void* src, void* query_out, void* gate_out) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * num_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_split_qgate_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, num_heads, head_dim,
        static_cast<const T*>(src),
        static_cast<T*>(query_out),
        static_cast<T*>(gate_out));
    return launch_result(371, 372);
}

int split_qgate_norm_bf16_device(
    int device_ordinal,
    int S,
    int num_heads,
    int head_dim,
    float eps,
    const void* src,
    const void* norm_w,
    void* query_out,
    void* gate_out
) {
    ScopedHipDevice scoped(device_ordinal);
    const int rows = S * num_heads;
    if (rows <= 0 || head_dim <= 0) return 373;
    constexpr int block = 256;
    hipLaunchKernelGGL(
        pfx_split_qgate_norm_bf16_kernel,
        dim3(static_cast<unsigned int>(rows)),
        dim3(block),
        0,
        0,
        S,
        num_heads,
        head_dim,
        eps,
        static_cast<const hip_bfloat16*>(src),
        static_cast<const hip_bfloat16*>(norm_w),
        static_cast<hip_bfloat16*>(query_out),
        static_cast<hip_bfloat16*>(gate_out));
    return launch_result(374, 375);
}

// ---- split_qkv ----

template <typename T>
int split_qkv_device(int device_ordinal,
                     int S, int key_dim, int val_dim,
                     const void* src, void* Q, void* K, void* V) {
    ScopedHipDevice scoped(device_ordinal);
    const int qkv_dim = key_dim * 2 + val_dim;
    const size_t total = static_cast<size_t>(S) * qkv_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_split_qkv_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, key_dim, val_dim,
        static_cast<const T*>(src),
        static_cast<T*>(Q),
        static_cast<T*>(K),
        static_cast<T*>(V));
    return launch_result(381, 382);
}

int split_qkv_bf16_to_f32_device(int device_ordinal,
                                 int S, int key_dim, int val_dim,
                                 const void* src, void* Q, void* K, void* V) {
    ScopedHipDevice scoped(device_ordinal);
    const int qkv_dim = key_dim * 2 + val_dim;
    const size_t total = static_cast<size_t>(S) * qkv_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_split_qkv_bf16_to_f32_kernel,
        dim3(grid), dim3(block), 0, 0,
        S, key_dim, val_dim,
        static_cast<const hip_bfloat16*>(src),
        static_cast<float*>(Q),
        static_cast<float*>(K),
        static_cast<float*>(V));
    return launch_result(383, 384);
}

int split_kv_bf16_device(int device_ordinal,
                         int S,
                         int kv_dim,
                         const void* src,
                         void* K,
                         void* V) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total = static_cast<size_t>(S) * static_cast<size_t>(kv_dim);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_split_kv_bf16_kernel,
        dim3(grid), dim3(block), 0, 0,
        S,
        kv_dim,
        static_cast<const hip_bfloat16*>(src),
        static_cast<hip_bfloat16*>(K),
        static_cast<hip_bfloat16*>(V));
    return launch_result(385, 386);
}

int split_norm_transpose_qkv_bf16_device(int device_ordinal,
                                         int S,
                                         int nk,
                                         int nv,
                                         int khd,
                                         int vhd,
                                         float q_scale,
                                         float eps,
                                         const void* src,
                                         void* Q,
                                         void* K,
                                         void* V) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total_rows =
        2 * static_cast<size_t>(S) * static_cast<size_t>(nk) +
        static_cast<size_t>(S) * static_cast<size_t>(nv);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        pfx_split_norm_transpose_qkv_bf16_kernel,
        dim3(static_cast<unsigned int>(total_rows)), dim3(block), 0, 0,
        S, nk, nv, khd, vhd, q_scale, eps,
        static_cast<const hip_bfloat16*>(src),
        static_cast<float*>(Q),
        static_cast<float*>(K),
        static_cast<float*>(V));
    return launch_result(387, 388);
}

int rms_norm_gated_sfirst_bf16_device(int device_ordinal,
                                      int S,
                                      int nv,
                                      int vhd,
                                      float eps,
                                      const void* hidden_hsd,
                                      const void* gate_sfirst,
                                      const void* weight,
                                      void* out_sfirst) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t total_rows = static_cast<size_t>(S) * static_cast<size_t>(nv);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        pfx_rms_norm_gated_sfirst_bf16_kernel,
        dim3(static_cast<unsigned int>(total_rows)), dim3(block), 0, 0,
        S, nv, vhd, eps,
        static_cast<const hip_bfloat16*>(hidden_hsd),
        static_cast<const hip_bfloat16*>(gate_sfirst),
        static_cast<const hip_bfloat16*>(weight),
        static_cast<hip_bfloat16*>(out_sfirst));
    return launch_result(389, 390);
}

int split_qkvz_bf16_device(int device_ordinal,
                           int S, int qkv_dim, int z_dim,
                           const void* src, void* QKV, void* Z) {
    ScopedHipDevice scoped(device_ordinal);
    const int total_dim = qkv_dim + z_dim;
    const size_t total = static_cast<size_t>(S) * total_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        pfx_split_qkvz_bf16_kernel,
        dim3(grid), dim3(block), 0, 0,
        S, qkv_dim, z_dim,
        static_cast<const hip_bfloat16*>(src),
        static_cast<hip_bfloat16*>(QKV),
        static_cast<hip_bfloat16*>(Z));
    return launch_result(385, 386);
}

// ---- repeat_interleave heads ----

template <typename T>
int repeat_interleave_heads_device(int device_ordinal,
                                   int S, int n_heads, int head_dim, int repeats,
                                   const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const int out_heads = n_heads * repeats;
    const size_t total = static_cast<size_t>(S) * out_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_repeat_interleave_heads_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, n_heads, head_dim, repeats,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    return launch_result(391, 392);
}

template <typename T>
int repeat_interleave_transpose_hsd_device(int device_ordinal,
                                           int S, int n_heads, int head_dim, int repeats,
                                           const void* src, void* dst) {
    ScopedHipDevice scoped(device_ordinal);
    const int out_heads = n_heads * repeats;
    const size_t total = static_cast<size_t>(S) * out_heads * head_dim;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_repeat_interleave_transpose_hsd_kernel<T>),
        dim3(grid), dim3(block), 0, 0,
        S, n_heads, head_dim, repeats,
        static_cast<const T*>(src),
        static_cast<T*>(dst));
    return launch_result(393, 394);
}

// ---- full_attention_decode_flat ----

template <typename T>
int full_attention_decode_flat_device(int device_ordinal,
                                      int batch_size,
                                      int q_heads,
                                      int kv_heads,
                                      int kv_len,
                                      int head_dim,
                                      int num_kv_groups,
                                      float scale,
                                      const void* query,
                                      const void* key,
                                      const void* value,
                                      void* out) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 32;
    if (head_dim > block * 8) return 401;
    const int rows = batch_size * q_heads;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_full_attention_decode_flat_kernel<T>),
        dim3(rows), dim3(block), 0, 0,
        batch_size, q_heads, kv_heads, kv_len, head_dim, num_kv_groups, scale,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    return launch_result(402, 403);
}

} // namespace

// ---- extern "C" wrappers ----

extern "C" int supersonic_prefill_encode_bridge_status(
    int project_status,
    int native_status
) {
    return native_status == 0
        ? project_status
        : backend_failure(project_status, static_cast<hipError_t>(native_status));
}

extern "C" int supersonic_qwen35_hip_element_add(
    int dtype, size_t device_ordinal, size_t total_elems,
    const void* lhs, const void* rhs, void* out
) {
    switch (dtype) {
    case 0: return element_add_device<half>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    case 1: return element_add_device<float>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    case 2: return element_add_device<hip_bfloat16>(static_cast<int>(device_ordinal), total_elems, lhs, rhs, out);
    default: return 300;
    }
}

extern "C" int supersonic_qwen35_hip_argmax_bf16_rows(
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* logits,
    void* out_index
) {
    return argmax_bf16_rows_device(
        static_cast<int>(device_ordinal),
        rows,
        cols,
        logits,
        out_index);
}

extern "C" int supersonic_qwen35_hip_apply_rope_prefill(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_heads, size_t head_dim, size_t half_rot,
    const void* cos_table, const void* sin_table, void* data
) {
    switch (dtype) {
    case 0: return apply_rope_prefill_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    case 1: return apply_rope_prefill_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    case 2: return apply_rope_prefill_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, data);
    default: return 310;
    }
}

extern "C" int supersonic_qwen35_hip_apply_rope_prefill_indirect(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t num_heads, size_t head_dim, size_t half_rot,
    const void* cos_table, const void* sin_table,
    const int* pos_ids, void* data
) {
    switch (dtype) {
    case 0: return apply_rope_prefill_indirect_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, pos_ids, data);
    case 1: return apply_rope_prefill_indirect_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, pos_ids, data);
    case 2: return apply_rope_prefill_indirect_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(num_heads),
                static_cast<int>(head_dim), static_cast<int>(half_rot),
                cos_table, sin_table, pos_ids, data);
    default: return 315;
    }
}

extern "C" int supersonic_qwen35_hip_lookahead_attention_scores(
    int dtype, size_t device_ordinal,
    size_t q_heads, size_t kv_heads,
    size_t lookahead_count, size_t kv_len, size_t head_dim,
    float scale,
    const void* q, const void* k, void* scores
) {
    switch (dtype) {
    case 0: return lookahead_attention_scores_device<half>(
                static_cast<int>(device_ordinal),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(lookahead_count), static_cast<int>(kv_len),
                static_cast<int>(head_dim), scale, q, k, scores);
    case 2: return lookahead_attention_scores_device<hip_bfloat16>(
                static_cast<int>(device_ordinal),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(lookahead_count), static_cast<int>(kv_len),
                static_cast<int>(head_dim), scale, q, k, scores);
    default: return 310;
    }
}

extern "C" int supersonic_qwen35_hip_pflash_cosine_score(
    int dtype, size_t device_ordinal,
    size_t n_pos, size_t kv_heads, size_t cap, size_t head_dim,
    size_t block_size, size_t n_blocks, size_t last_pos,
    const void* k_cache, void* scores
) {
    switch (dtype) {
    case 0: return pflash_cosine_score_device<half>(
                static_cast<int>(device_ordinal),
                static_cast<int>(n_pos), static_cast<int>(kv_heads),
                static_cast<int>(cap), static_cast<int>(head_dim),
                static_cast<int>(block_size), static_cast<int>(n_blocks),
                static_cast<int>(last_pos), k_cache, scores);
    case 2: return pflash_cosine_score_device<hip_bfloat16>(
                static_cast<int>(device_ordinal),
                static_cast<int>(n_pos), static_cast<int>(kv_heads),
                static_cast<int>(cap), static_cast<int>(head_dim),
                static_cast<int>(block_size), static_cast<int>(n_blocks),
                static_cast<int>(last_pos), k_cache, scores);
    default: return 336; // unsupported dtype
    }
}

extern "C" int supersonic_qwen35_hip_transpose_shd_hsd(
    int dtype, size_t device_ordinal,
    size_t S, size_t H, size_t D,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return transpose_shd_hsd_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    case 1: return transpose_shd_hsd_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    case 2: return transpose_shd_hsd_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst);
    default: return 320;
    }
}

extern "C" int supersonic_qwen35_hip_transpose_shd_hsd_pair(
    int dtype, size_t device_ordinal,
    size_t S, size_t H, size_t D,
    const void* src_a, const void* src_b,
    void* dst_a, void* dst_b
) {
    switch (dtype) {
    case 0: return transpose_shd_hsd_pair_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D),
                src_a, src_b, dst_a, dst_b);
    case 1: return transpose_shd_hsd_pair_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D),
                src_a, src_b, dst_a, dst_b);
    case 2: return transpose_shd_hsd_pair_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(H), static_cast<int>(D),
                src_a, src_b, dst_a, dst_b);
    default: return 325;
    }
}

extern "C" int supersonic_qwen35_hip_transpose_shd_to_cache_bf16(
    size_t device_ordinal,
    size_t S, size_t H, size_t D,
    size_t cache_len,
    size_t dst_pos,
    const void* src,
    void* cache
) {
    return transpose_shd_to_cache_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(H),
        static_cast<int>(D),
        static_cast<int>(cache_len),
        static_cast<int>(dst_pos),
        src,
        cache);
}

extern "C" int supersonic_qwen35_hip_transpose_pad_conv(
    int dtype, size_t device_ordinal,
    size_t S, size_t C, size_t pad,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return transpose_pad_conv_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    case 1: return transpose_pad_conv_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    case 2: return transpose_pad_conv_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst);
    default: return 329;
    }
}

extern "C" int supersonic_qwen35_hip_extract_conv_state(
    int dtype, size_t device_ordinal,
    size_t S, size_t C, size_t kern_minus_1,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return extract_conv_state_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    case 1: return extract_conv_state_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    case 2: return extract_conv_state_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst);
    default: return 340;
    }
}

extern "C" int supersonic_qwen35_hip_prepare_conv_input_tail(
    int dtype, size_t device_ordinal,
    size_t S, size_t C, size_t pad,
    const void* src, const void* old_tail,
    void* conv_input, void* new_tail
) {
    switch (dtype) {
    case 0: return prepare_conv_input_tail_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad),
                src, old_tail, conv_input, new_tail);
    case 1: return prepare_conv_input_tail_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad),
                src, old_tail, conv_input, new_tail);
    case 2: return prepare_conv_input_tail_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad),
                src, old_tail, conv_input, new_tail);
    default: return 346;
    }
}

extern "C" int supersonic_qwen35_hip_sigmoid_mul(
    int dtype, size_t device_ordinal, size_t total_elems,
    const void* data, const void* gate, void* out
) {
    switch (dtype) {
    case 0: return sigmoid_mul_device<half>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    case 1: return sigmoid_mul_device<float>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    case 2: return sigmoid_mul_device<hip_bfloat16>(static_cast<int>(device_ordinal), total_elems, data, gate, out);
    default: return 350;
    }
}

extern "C" int supersonic_qwen35_hip_cast_transpose_gate_bf16(
    size_t device_ordinal,
    size_t S,
    size_t H,
    size_t D,
    const void* attn_hsd,
    const void* gate_shd,
    void* out_shd
) {
    return cast_transpose_gate_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(H),
        static_cast<int>(D),
        attn_hsd,
        gate_shd,
        out_shd);
}

extern "C" int supersonic_qwen35_hip_compute_beta_g(
    int dtype, size_t device_ordinal,
    size_t seq_len, size_t nv,
    const void* B, const void* A,
    const void* dt_bias, const void* a_log_exp,
    void* beta, void* g
) {
    switch (dtype) {
    case 0: return compute_beta_g_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    case 1: return compute_beta_g_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    case 2: return compute_beta_g_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g);
    default: return 360;
    }
}

extern "C" int supersonic_qwen35_hip_compute_beta_g_ba_bf16(
    size_t device_ordinal,
    size_t seq_len, size_t nv,
    const void* BA,
    const void* dt_bias,
    const void* a_log_exp,
    void* beta, void* g
) {
    return compute_beta_g_ba_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(seq_len),
        static_cast<int>(nv),
        BA, dt_bias, a_log_exp, beta, g);
}

extern "C" int supersonic_qwen35_hip_project_ba_compute_beta_g_bf16(
    size_t device_ordinal,
    size_t seq_len,
    size_t hidden_dim,
    size_t nv,
    const void* hidden,
    const void* ba_weight,
    const void* dt_bias,
    const void* a_log_exp,
    void* beta,
    void* g
) {
    return project_ba_compute_beta_g_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(seq_len),
        static_cast<int>(hidden_dim),
        static_cast<int>(nv),
        hidden,
        ba_weight,
        dt_bias,
        a_log_exp,
        beta,
        g);
}

extern "C" int supersonic_qwen35_hip_split_qgate(
    int dtype, size_t device_ordinal,
    size_t S, size_t num_heads, size_t head_dim,
    const void* src, void* query_out, void* gate_out
) {
    switch (dtype) {
    case 0: return split_qgate_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    case 1: return split_qgate_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    case 2: return split_qgate_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out);
    default: return 370;
    }
}

extern "C" int supersonic_qwen35_hip_split_qgate_norm_bf16(
    size_t device_ordinal,
    size_t S,
    size_t num_heads,
    size_t head_dim,
    float eps,
    const void* src,
    const void* norm_w,
    void* query_out,
    void* gate_out
) {
    return split_qgate_norm_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim),
        eps,
        src,
        norm_w,
        query_out,
        gate_out);
}

extern "C" int supersonic_qwen35_hip_split_qkv(
    int dtype, size_t device_ordinal,
    size_t S, size_t key_dim, size_t val_dim,
    const void* src, void* Q, void* K, void* V
) {
    switch (dtype) {
    case 0: return split_qkv_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    case 1: return split_qkv_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    case 2: return split_qkv_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V);
    default: return 380;
    }
}

extern "C" int supersonic_qwen35_hip_split_qkv_bf16_to_f32(
    size_t device_ordinal,
    size_t S, size_t key_dim, size_t val_dim,
    const void* src, void* Q, void* K, void* V
) {
    return split_qkv_bf16_to_f32_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(key_dim),
        static_cast<int>(val_dim),
        src,
        Q,
        K,
        V);
}

extern "C" int supersonic_qwen35_hip_split_kv_bf16(
    size_t device_ordinal,
    size_t S,
    size_t kv_dim,
    const void* src,
    void* K,
    void* V
) {
    return split_kv_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(kv_dim),
        src,
        K,
        V);
}

extern "C" int supersonic_qwen35_hip_split_norm_transpose_qkv_bf16(
    size_t device_ordinal,
    size_t S,
    size_t nk,
    size_t nv,
    size_t khd,
    size_t vhd,
    float q_scale,
    float eps,
    const void* src,
    void* Q,
    void* K,
    void* V
) {
    return split_norm_transpose_qkv_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(nk),
        static_cast<int>(nv),
        static_cast<int>(khd),
        static_cast<int>(vhd),
        q_scale,
        eps,
        src,
        Q,
        K,
        V);
}

extern "C" int supersonic_qwen35_hip_rms_norm_gated_sfirst_bf16(
    size_t device_ordinal,
    size_t S,
    size_t nv,
    size_t vhd,
    float eps,
    const void* hidden_hsd,
    const void* gate_sfirst,
    const void* weight,
    void* out_sfirst
) {
    return rms_norm_gated_sfirst_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(nv),
        static_cast<int>(vhd),
        eps,
        hidden_hsd,
        gate_sfirst,
        weight,
        out_sfirst);
}

extern "C" int supersonic_qwen35_hip_split_qkvz_bf16(
    size_t device_ordinal,
    size_t S, size_t qkv_dim, size_t z_dim,
    const void* src, void* QKV, void* Z
) {
    return split_qkvz_bf16_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(S),
        static_cast<int>(qkv_dim),
        static_cast<int>(z_dim),
        src,
        QKV,
        Z);
}

extern "C" int supersonic_qwen35_hip_repeat_interleave_heads(
    int dtype, size_t device_ordinal,
    size_t S, size_t n_heads, size_t head_dim, size_t repeats,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return repeat_interleave_heads_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 1: return repeat_interleave_heads_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 2: return repeat_interleave_heads_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    default: return 390;
    }
}

extern "C" int supersonic_qwen35_hip_repeat_interleave_transpose_hsd(
    int dtype, size_t device_ordinal,
    size_t S, size_t n_heads, size_t head_dim, size_t repeats,
    const void* src, void* dst
) {
    switch (dtype) {
    case 0: return repeat_interleave_transpose_hsd_device<half>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 1: return repeat_interleave_transpose_hsd_device<float>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    case 2: return repeat_interleave_transpose_hsd_device<hip_bfloat16>(static_cast<int>(device_ordinal),
                static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim),
                static_cast<int>(repeats), src, dst);
    default: return 395;
    }
}

extern "C" int supersonic_qwen35_hip_full_attention_decode_flat(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    switch (dtype) {
    case 0: return full_attention_decode_flat_device<half>(
                static_cast<int>(device_ordinal), static_cast<int>(batch_size),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(kv_len), static_cast<int>(head_dim),
                static_cast<int>(num_kv_groups), scale, query, key, value, out);
    case 2: return full_attention_decode_flat_device<hip_bfloat16>(
                static_cast<int>(device_ordinal), static_cast<int>(batch_size),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(kv_len), static_cast<int>(head_dim),
                static_cast<int>(num_kv_groups), scale, query, key, value, out);
    default: return 400;
    }
}
