// Bridge for the Qwen3.8 prefill helper kernels.
// Separate compilation unit — does not touch the decode megakernel files.
// The qwen35-prefixed extern symbols below are historical ABI spellings only.

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

int argmax_f32_as_bf16_rows_device(
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
        pfx_argmax_f32_as_bf16_rows_kernel,
        dim3(static_cast<unsigned int>(rows)),
        dim3(block),
        0,
        0,
        rows,
        cols,
        static_cast<const float*>(logits),
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

extern "C" int supersonic_qwen35_hip_argmax_f32_as_bf16_rows(
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* logits,
    void* out_index
) {
    return argmax_f32_as_bf16_rows_device(
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


// ---- dflash2 dynamic depthwise conv (draft block) ----

int dflash_dyn_conv_device(
    int device_ordinal, int hidden, int nq, int K, int gs, int s,
    const void* x, const void* base, const void* dyn, void* out,
    int out_dtype
) {
    ScopedHipDevice scoped(device_ordinal);
    if (hidden <= 0 || nq <= 0 || K <= 0 || gs <= 0 || (hidden % gs) != 0) return 350;
    constexpr int block = 256;
    const unsigned int grid_x =
        (static_cast<unsigned int>(hidden) + block - 1) / block;
    if (out_dtype == 1) {
        // F32 output: x is F32, dyn is F32, out is F32.
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(pfx_dflash_dyn_conv_f32_kernel),
            dim3(grid_x, static_cast<unsigned int>(nq)), dim3(block), 0, 0,
            hidden, nq, K, gs, s,
            static_cast<const float*>(x),
            static_cast<const float*>(base),
            static_cast<const float*>(dyn),
            static_cast<float*>(out));
    } else {
        // BF16 output (default): x is BF16, dyn is BF16, out is BF16.
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(pfx_dflash_dyn_conv_kernel),
            dim3(grid_x, static_cast<unsigned int>(nq)), dim3(block), 0, 0,
            hidden, nq, K, gs, s,
            static_cast<const hip_bfloat16*>(x),
            static_cast<const float*>(base),
            static_cast<const hip_bfloat16*>(dyn),
            static_cast<hip_bfloat16*>(out));
    }
    return launch_result(351, 352);
}

extern "C" int supersonic_dflash_dyn_conv(
    int device_ordinal, int hidden, int nq, int K, int gs, int s,
    const void* x, const void* base, const void* dyn, void* out,
    int out_dtype
) {
    return dflash_dyn_conv_device(device_ordinal, hidden, nq, K, gs, s, x, base, dyn, out, out_dtype);
}

// ---- dflash2 target hidden-state strided scatter ----

int dflash_scatter_cols_device(
    int device_ordinal,
    const void* src, void* dst,
    int n_rows, int n_cols, int col_offset, int dst_stride)
{
    ScopedHipDevice scoped(device_ordinal);
    if (n_rows <= 0 || n_cols <= 0 || dst_stride < n_cols) return 360;
    constexpr int block = 256;
    const unsigned int total = static_cast<unsigned int>(n_rows) *
                               static_cast<unsigned int>(n_cols);
    const unsigned int grid_x = (total + block - 1) / block;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_dflash_scatter_cols_kernel),
        dim3(grid_x), dim3(block), 0, 0,
        static_cast<const hip_bfloat16*>(src),
        static_cast<hip_bfloat16*>(dst),
        n_rows, n_cols, col_offset, dst_stride);
    return launch_result(361, 362);
}

extern "C" int supersonic_dflash_scatter_cols(
    int device_ordinal,
    const void* src, void* dst,
    int n_rows, int n_cols, int col_offset, int dst_stride)
{
    return dflash_scatter_cols_device(device_ordinal, src, dst,
                                      n_rows, n_cols, col_offset, dst_stride);
}

// Batched conv-tail assembly for the chunk_len < pad case.
extern "C" int supersonic_pfx_assemble_conv_tail_short(
    int dtype,
    int device_ordinal,
    int qkv_dim,
    int pad,
    int chunk_len,
    int chunk_start,
    const void* old_tail,
    const void* qkv,
    void* new_tail)
{
    ScopedHipDevice scoped(device_ordinal);
    if (qkv_dim <= 0 || pad <= 0 || chunk_len <= 0 || chunk_len >= pad) return 370;
    const int keep_old = pad - chunk_len;
    const int total = qkv_dim * pad;
    const int block = 256;
    const int grid_x = (total + block - 1) / block;
    if (dtype == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(pfx_assemble_conv_tail_short_kernel<float>),
            dim3(grid_x), dim3(block), 0, 0,
            qkv_dim, pad, chunk_len, keep_old, chunk_start,
            static_cast<const float*>(old_tail),
            static_cast<const float*>(qkv),
            static_cast<float*>(new_tail));
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(pfx_assemble_conv_tail_short_kernel<hip_bfloat16>),
            dim3(grid_x), dim3(block), 0, 0,
            qkv_dim, pad, chunk_len, keep_old, chunk_start,
            static_cast<const hip_bfloat16*>(old_tail),
            static_cast<const hip_bfloat16*>(qkv),
            static_cast<hip_bfloat16*>(new_tail));
    }
    return launch_result(371, 372);
}
