#include "full_attention.hip"

// This bridge is the retained Qwen3.8 implementation. Its qwen35-prefixed
// kernel/C symbols are historical bridge ABI spellings; changing them would
// invalidate the linked HIP archive, so no alternate qwen35 model path exists.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdint.h>
#include <type_traits>

extern "C" void supersonic_gqh_hip_lock();
extern "C" void supersonic_gqh_hip_unlock();
extern "C" [[noreturn]] void supersonic_gpu_integrity_fail_stop(
    const char* operation,
    int status,
    int device_ordinal);

namespace {

// The legacy prefill GEMM path retains process-global hipBLAS/scratch state.
// Use the same recursive bridge lock as the GQH/4B bridges so a concurrent
// request cannot overwrite the shared scores buffer or mutate a handle while
// another request is using it.
struct DecodeBridgeLockGuard {
    DecodeBridgeLockGuard() { supersonic_gqh_hip_lock(); }
    ~DecodeBridgeLockGuard() { supersonic_gqh_hip_unlock(); }
};

int prefill_backend_failure(int project_status, hipError_t native_status) {
    return static_cast<int>(
        0x80000000u
        | ((static_cast<uint32_t>(project_status) & 0x7fffu) << 16)
        | (static_cast<uint32_t>(native_status) & 0xffffu));
}

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    hipError_t status = hipSuccess;

    explicit ScopedHipDevice(int target) {
        status = hipGetDevice(&previous);
        if (status != hipSuccess) {
            return;
        }
        if (previous != target) {
            status = hipSetDevice(target);
            if (status == hipSuccess) {
                changed = true;
            }
        }
    }

    hipError_t restore() {
        if (!changed || previous < 0) {
            return hipSuccess;
        }
        const hipError_t err = hipSetDevice(previous);
        if (err == hipSuccess) {
            changed = false;
        } else {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention device restore", static_cast<int>(err), previous);
        }
        return hipSuccess;
    }

    ~ScopedHipDevice() {
        const hipError_t err = restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention device restore", static_cast<int>(err), previous);
        }
    }

    bool ok() const { return status == hipSuccess; }
};

int linear_prefill_block_override() {
    const char* value = std::getenv("DOTCACHE_QWEN38_HIP_FUSED_PREFILL_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed < 32) {
        return 32;
    }
    if (parsed > 512) {
        return 512;
    }
    return static_cast<int>(parsed);
}

hipError_t maybe_sync() {
    const char* value = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    const bool enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
    return enabled ? hipDeviceSynchronize() : hipSuccess;
}

hipblasHandle_t attn_hipblas(int device_ordinal) {
    static hipblasHandle_t handles[16] = {};
    static bool ready[16] = {};
    if (device_ordinal < 0 || device_ordinal >= 16) {
        return nullptr;
    }
    if (!ready[device_ordinal]) {
        if (hipblasCreate(&handles[device_ordinal]) != HIPBLAS_STATUS_SUCCESS) {
            handles[device_ordinal] = nullptr;
        }
        ready[device_ordinal] = true;
    }
    return handles[device_ordinal];
}

struct AttnScratchBf16 {
    hip_bfloat16* ptr = nullptr;
    size_t cap = 0;
    int device_ordinal = -1;
};

hip_bfloat16* attn_scratch_bf16(
    int device_ordinal, size_t n, AttnScratchBf16* scratch) {
    if (n <= scratch->cap && scratch->ptr != nullptr &&
        scratch->device_ordinal == device_ordinal) {
        return scratch->ptr;
    }
    if (scratch->ptr != nullptr) {
        if (scratch->device_ordinal < 0) {
            return nullptr;
        }
        const int old_device = scratch->device_ordinal;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention scratch owner switch",
                static_cast<int>(old_owner.status),
                scratch->device_ordinal);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention scratch synchronize",
                static_cast<int>(err),
                scratch->device_ordinal);
        }
        err = hipFree(scratch->ptr);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention scratch free",
                static_cast<int>(err),
                scratch->device_ordinal);
        }
        *scratch = AttnScratchBf16{};
        err = old_owner.restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "legacy attention scratch owner restore",
                static_cast<int>(err),
                old_device);
        }
    }
    ScopedHipDevice target(device_ordinal);
    if (!target.ok()) {
        return nullptr;
    }
    const hipError_t err = hipMalloc(&scratch->ptr, n * sizeof(hip_bfloat16));
    if (err != hipSuccess) {
        scratch->ptr = nullptr;
        (void)target.restore();
        return nullptr;
    }
    scratch->cap = n;
    scratch->device_ordinal = device_ordinal;
    const hipError_t restore_err = target.restore();
    if (restore_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "legacy attention scratch target restore",
            static_cast<int>(restore_err),
            device_ordinal);
    }
    return scratch->ptr;
}

// scores = Q[pack*q_len,hd] @ K[kv_len,hd]^T then causal softmax, then
// out = scores @ V[kv_len,hd]. Packs the Q heads that share a KV head.
// Scores stay BF16 (half the F32 footprint, tensor-core AV). Softmax
// uses qi = row % q_len. Falls back to a smaller pack if the score
// matrix will not allocate.
int launch_gemm_attn_bf16(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const hip_bfloat16* query,
    const hip_bfloat16* key,
    const hip_bfloat16* value,
    float* out
) {
    hipblasHandle_t blas = attn_hipblas(device_ordinal);
    if (blas == nullptr || q_len <= 0 || kv_len <= 0 || q_heads <= 0 || kv_heads <= 0) {
        return 1;
    }
    static AttnScratchBf16 scores;
    const int groups = num_kv_groups > 0 ? num_kv_groups : 1;
    const size_t per_head = static_cast<size_t>(q_len) * static_cast<size_t>(kv_len);
    int pack = groups;
    if (pack < 1) {
        pack = 1;
    }
    if (pack > q_heads) {
        pack = q_heads;
    }
    while (pack > 1 &&
           attn_scratch_bf16(
               device_ordinal,
               per_head * static_cast<size_t>(pack),
               &scores) ==
               nullptr) {
        pack = pack > 2 ? pack / 2 : 1;
    }
    if (attn_scratch_bf16(
            device_ordinal,
            per_head * static_cast<size_t>(pack),
            &scores) == nullptr) {
        return 2;
    }
    static bool dumped_pack = false;
    if (!dumped_pack) {
        dumped_pack = true;
        std::fprintf(
            stderr,
            "[attn-gemm] gqa pack=%d/%d scores_bf16=%.1fMiB q_len=%d kv_len=%d\n",
            pack,
            groups,
            static_cast<double>(per_head * static_cast<size_t>(pack) * 2) / (1024.0 * 1024.0),
            q_len,
            kv_len);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int kvh = 0; kvh < kv_heads; ++kvh) {
            const int h_begin = kvh * groups;
            int h_end = h_begin + groups;
            if (h_end > q_heads) {
                h_end = q_heads;
            }
            if (h_begin >= h_end) {
                continue;
            }
            const hip_bfloat16* k =
                key + (static_cast<size_t>(b * kv_heads + kvh) * kv_len * head_dim);
            const hip_bfloat16* v =
                value + (static_cast<size_t>(b * kv_heads + kvh) * kv_len * head_dim);
            for (int h0 = h_begin; h0 < h_end; h0 += pack) {
                int ph = pack;
                if (h0 + ph > h_end) {
                    ph = h_end - h0;
                }
                const int m = ph * q_len;
                const hip_bfloat16* q =
                    query + (static_cast<size_t>(b * q_heads + h0) * q_len * head_dim);
                float* y =
                    out + (static_cast<size_t>(b * q_heads + h0) * q_len * head_dim);

                // scores[m, kv_len] = Q[m, hd] @ K[kv_len, hd]^T  (BF16 out)
                const hipblasStatus_t st_qk = hipblasGemmEx(
                    blas,
                    HIPBLAS_OP_T,
                    HIPBLAS_OP_N,
                    kv_len,
                    m,
                    head_dim,
                    &alpha,
                    k,
                    HIP_R_16BF,
                    head_dim,
                    q,
                    HIP_R_16BF,
                    head_dim,
                    &beta,
                    scores.ptr,
                    HIP_R_16BF,
                    kv_len,
                    HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT);
                if (st_qk != HIPBLAS_STATUS_SUCCESS) {
                    return 4;
                }
                hipLaunchKernelGGL(
                    supersonic_qwen35_causal_softmax_rows_bf16,
                    dim3(static_cast<unsigned int>(m)),
                    dim3(256),
                    0,
                    0,
                    scores.ptr,
                    m,
                    q_len,
                    kv_len,
                    seqlen_offset,
                    scale);
                if (hipGetLastError() != hipSuccess) {
                    return 5;
                }
                // out[m, hd] = scores[m, kv_len] @ V[kv_len, hd]
                const hipblasStatus_t st_av = hipblasGemmEx(
                    blas,
                    HIPBLAS_OP_N,
                    HIPBLAS_OP_N,
                    head_dim,
                    m,
                    kv_len,
                    &alpha,
                    v,
                    HIP_R_16BF,
                    head_dim,
                    scores.ptr,
                    HIP_R_16BF,
                    kv_len,
                    &beta,
                    y,
                    HIP_R_32F,
                    head_dim,
                    HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT);
                if (st_av != HIPBLAS_STATUS_SUCCESS) {
                    return 7;
                }
            }
        }
    }
return maybe_sync() == hipSuccess ? 0 : 8;
}
template <typename T>
int full_attention_prefill_device(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return prefill_backend_failure(136, scoped.status);
    }

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) {
        return 1;
    }

    const T* d_query = static_cast<const T*>(query);
    const T* d_key = static_cast<const T*>(key);
    const T* d_value = static_cast<const T*>(value);
    float* d_out = static_cast<float*>(out);
    unsigned int* d_row_counter = nullptr;

    if (hipMalloc(&d_row_counter, sizeof(unsigned int)) != hipSuccess) return 2;
    if (hipMemset(d_row_counter, 0, sizeof(unsigned int)) != hipSuccess) return 10;

    // On RDNA3/gfx11xx `multiProcessorCount` reports WGPs (half the CU
    // count), leaving half the device idle when we grid the prefill
    // attention against it directly. Oversubscribe 2x on those arches,
    // matching the decode default. The atomic row-counter inside the
    // kernel already balances work across blocks, so extra blocks are
    // harmless on arches where `multiProcessorCount` already reports CUs.
    int grid = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    {
        const char* arch = props.gcnArchName;
        const bool is_rdna3_wgp_arch =
            arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
            arch[3] == '1' && arch[4] == '1';
        if (is_rdna3_wgp_arch) grid *= 2;
    }
    const int block = props.warpSize > 0 ? props.warpSize : 32;
    if (head_dim > block * 8) return 14;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_full_attention_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_kv_groups,
        scale,
        seqlen_offset,
        d_query,
        d_key,
        d_value,
        d_out,
        d_row_counter);
    if (hipGetLastError() != hipSuccess) return 11;
    if (maybe_sync() != hipSuccess) return 12;

    hipFree(d_row_counter);
    return 0;
}

template <typename T, int BM, int BK>
static int launch_tiled(
    int batch_size, int q_heads, int kv_heads,
    int q_len, int kv_len, int head_dim, int num_kv_groups,
    float scale, int seqlen_offset,
    const void* query, const void* key, const void* value, void* out
) {
    const int grid_x = (q_len + BM - 1) / BM;
    dim3 grid(grid_x, q_heads, batch_size);
    dim3 block(32, BM, 1);
    const size_t lds_bytes = (size_t)2 * BK * head_dim * sizeof(T);
    if (lds_bytes > 64 * 1024) return 133;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_full_attention_prefill_tiled_kernel<T, BM, BK>),
        grid, block, lds_bytes, 0,
        batch_size, q_heads, kv_heads, q_len, kv_len, head_dim,
        num_kv_groups, scale, seqlen_offset,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<float*>(out));
    if (hipGetLastError() != hipSuccess) return 134;
    if (maybe_sync() != hipSuccess) return 135;
    return 0;
}

template <typename T>
int full_attention_prefill_tiled_device(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    if (head_dim > 8 * 32) return 132;       // ACC_MAX=8 × warpSize=32 cap
    if (q_len <= 0) return 0;

    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return prefill_backend_failure(137, scoped.status);
    }

    // The kernel hardcodes block.x = 32 and the per-lane acc_dim/load
    // strides assume warpSize == 32. On wave64 (CDNA gfx9xx) the kernel
    // would launch with only 32 of 64 lanes per warp doing useful work and
    // the strided loops would skip half of head_dim. Refuse the launch so
    // the dispatcher falls through to the legacy single-warp kernel, which
    // does adapt to props.warpSize.
    static int cached_warp = 0;
    if (cached_warp == 0) {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 136;
        cached_warp = props.warpSize > 0 ? props.warpSize : -1;
    }
    if (cached_warp != 32) return 137;

    // BM=4 left K/V tiles reused by only 4 queries. Longer prefills share
    // each tile across 8 or 16 query rows (LDS size does not depend on BM).
    const bool long_seq = q_len >= 1024;

    if (head_dim <= 64) {
        if (long_seq) {
            return launch_tiled<T, 16, 128>(batch_size, q_heads, kv_heads,
                q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
                query, key, value, out);
        }
        return launch_tiled<T, 8, 128>(batch_size, q_heads, kv_heads,
            q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
            query, key, value, out);
    } else if (head_dim <= 128) {
        if (long_seq) {
            return launch_tiled<T, 16, 64>(batch_size, q_heads, kv_heads,
                q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
                query, key, value, out);
        }
        return launch_tiled<T, 8, 64>(batch_size, q_heads, kv_heads,
            q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
            query, key, value, out);
    } else if (long_seq) {
        if constexpr (std::is_same<T, hip_bfloat16>::value) {
            const int rc = launch_gemm_attn_bf16(
                device_ordinal, batch_size, q_heads, kv_heads, q_len, kv_len, head_dim,
                num_kv_groups, scale, seqlen_offset,
                static_cast<const hip_bfloat16*>(query),
                static_cast<const hip_bfloat16*>(key),
                static_cast<const hip_bfloat16*>(value),
                static_cast<float*>(out));
            if (rc == 0) {
                return 0;
            }
        }
        return launch_tiled<T, 32, 32>(batch_size, q_heads, kv_heads,
            q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
            query, key, value, out);
    } else {
        return launch_tiled<T, 8, 32>(batch_size, q_heads, kv_heads,
            q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset,
            query, key, value, out);
    }
}

template <typename T>
int linear_prefill_conv_pack_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int total_len,
    int seq_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_linear_prefill_conv_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        total_len,
        seq_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 60;
    if (maybe_sync() != hipSuccess) return 61;
    return 0;
}

template <typename T>
int delta_recurrent_prefill_device(
    int device_ordinal,
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out,
    void* state_trace
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
    static const bool disable_warp = [] {
        const char* e = std::getenv("SUPERSONIC_REC_WARP");
        return e != nullptr && e[0] == '0';
    }();
    // Capture/trace needs the original one-thread-per-v walk so the
    // exported state dump stays element-order identical.
    if (state_trace == nullptr && !disable_warp && seq_len > 1 &&
        k_head_dim == 128 && v_head_dim == 128) {
        constexpr int warps_per_block = 4;
        const size_t total_warps =
            static_cast<size_t>(batch_heads) *
            static_cast<size_t>(v_head_dim / 4);
        const unsigned int grid = static_cast<unsigned int>(
            (total_warps + static_cast<size_t>(warps_per_block) - 1) /
            static_cast<size_t>(warps_per_block));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_warp_k128_kernel<T>),
            dim3(grid > 0 ? grid : 1u),
            dim3(32, warps_per_block),
            0,
            0,
            batch_heads,
            seq_len,
            static_cast<const T*>(initial_state),
            static_cast<const T*>(query),
            static_cast<const T*>(key),
            static_cast<const T*>(value),
            static_cast<const T*>(beta),
            static_cast<const T*>(g),
            static_cast<T*>(out));
        if (hipGetLastError() != hipSuccess) return 67;
        if (maybe_sync() != hipSuccess) return 68;
        return 0;
    }
    if (state_trace == nullptr && !disable_warp && seq_len > 1) {
        constexpr int warps_per_block = 8;
        const size_t total_warps =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
        const unsigned int grid = static_cast<unsigned int>(
            (total_warps + static_cast<size_t>(warps_per_block) - 1) /
            static_cast<size_t>(warps_per_block));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_warp_kernel<T>),
            dim3(grid > 0 ? grid : 1u),
            dim3(32, warps_per_block),
            0,
            0,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            static_cast<const T*>(initial_state),
            static_cast<const T*>(query),
            static_cast<const T*>(key),
            static_cast<const T*>(value),
            static_cast<const T*>(beta),
            static_cast<const T*>(g),
            static_cast<T*>(out));
        if (hipGetLastError() != hipSuccess) return 67;
        if (maybe_sync() != hipSuccess) return 68;
        return 0;
    }
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out),
        static_cast<T*>(state_trace));
    if (hipGetLastError() != hipSuccess) return 67;
    if (maybe_sync() != hipSuccess) return 68;
    return 0;
}

template <typename T>
int delta_recurrent_prefill_device_stream(
    int device_ordinal,
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out,
    hipStream_t stream
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
    static const bool disable_warp = [] {
        const char* e = std::getenv("SUPERSONIC_REC_WARP");
        return e != nullptr && e[0] == '0';
    }();
    if (!disable_warp && seq_len >= 1 &&
        k_head_dim == 128 && v_head_dim == 128) {
        constexpr int warps_per_block = 4;
        const size_t total_warps =
            static_cast<size_t>(batch_heads) *
            static_cast<size_t>(v_head_dim / 4);
        const unsigned int grid = static_cast<unsigned int>(
            (total_warps + static_cast<size_t>(warps_per_block) - 1) /
            static_cast<size_t>(warps_per_block));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_warp_k128_kernel<T>),
            dim3(grid > 0 ? grid : 1u),
            dim3(32, warps_per_block),
            0,
            stream,
            batch_heads,
            seq_len,
            static_cast<const T*>(initial_state),
            static_cast<const T*>(query),
            static_cast<const T*>(key),
            static_cast<const T*>(value),
            static_cast<const T*>(beta),
            static_cast<const T*>(g),
            static_cast<T*>(out));
        return hipGetLastError() == hipSuccess ? 0 : 67;
    }
    if (!disable_warp && seq_len >= 1) {
        constexpr int warps_per_block = 8;
        const size_t total_warps =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
        const unsigned int grid = static_cast<unsigned int>(
            (total_warps + static_cast<size_t>(warps_per_block) - 1) /
            static_cast<size_t>(warps_per_block));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_warp_kernel<T>),
            dim3(grid > 0 ? grid : 1u),
            dim3(32, warps_per_block),
            0,
            stream,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            static_cast<const T*>(initial_state),
            static_cast<const T*>(query),
            static_cast<const T*>(key),
            static_cast<const T*>(value),
            static_cast<const T*>(beta),
            static_cast<const T*>(g),
            static_cast<T*>(out));
        return hipGetLastError() == hipSuccess ? 0 : 67;
    }
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>(
        (total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        stream,
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out),
        static_cast<T*>(nullptr));
    return hipGetLastError() == hipSuccess ? 0 : 67;
}



template <typename T>
int fill_conv_tail_device(
    int device_ordinal,
    int qkv_dim,
    int pad,
    int total_len,
    const void* tail,
    void* conv_input
) {
    ScopedHipDevice scoped(device_ordinal);
    const int total = qkv_dim * pad;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_fill_conv_tail_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        qkv_dim,
        pad,
        total_len,
        static_cast<const T*>(tail),
        static_cast<T*>(conv_input));
    if (hipGetLastError() != hipSuccess) return 76;
    if (maybe_sync() != hipSuccess) return 77;
    return 0;
}


template <typename T>
int delta_chunk_single_prefill_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 76;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_single_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 77;
    if (maybe_sync() != hipSuccess) return 78;
    return 0;
}

template <typename T>
int delta_chunk_step_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 80;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_step_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (maybe_sync() != hipSuccess) return 82;
    return 0;
}

template <typename T>
int delta_chunk_scan_raw_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 83;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_scan_raw_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 84;
    if (maybe_sync() != hipSuccess) return 85;
    return 0;
}

template <typename T>
int delta_state_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 88;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_state_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 89;
    if (maybe_sync() != hipSuccess) return 96;
    return 0;
}

template <typename T>
int delta_chunk_fused_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 97;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_fused_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(packed_chunk),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 98;
    if (maybe_sync() != hipSuccess) return 99;
    return 0;
}

template <typename T>
int delta_full_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 100;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(weighted_key_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<const T*>(q_state_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(state_decay_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 101;
    if (maybe_sync() != hipSuccess) return 102;
    return 0;
}

template <typename T>
int delta_local_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 112;
    if (chunk_size <= 4) {
        constexpr int block = 256;
        const size_t total =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_local_attn_scan_flat_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    } else {
        const unsigned int block = chunk_size <= 32 ? 32u : 64u;
        const size_t total_rows =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>(total_rows);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_local_attn_scan_row_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    }
    if (hipGetLastError() != hipSuccess) return 113;
    if (maybe_sync() != hipSuccess) return 114;
    return 0;
}

template <typename T>
int delta_base_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 115;
    constexpr int block = 256;
    const size_t total =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_base_attn_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 116;
    if (maybe_sync() != hipSuccess) return 117;
    return 0;
}

template <typename T>
int delta_attn_solve_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    const void* base_attn_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64) return 118;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_attn_solve_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        static_cast<const T*>(base_attn_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 119;
    if (maybe_sync() != hipSuccess) return 120;
    return 0;
}

template <typename T>
int delta_attn_solve_from_inputs_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 121;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_attn_solve_from_inputs_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 122;
    if (maybe_sync() != hipSuccess) return 123;
    return 0;
}

template <typename T>
int swiglu_mul_device(
    int device_ordinal,
    int elem_count,
    const void* gate,
    const void* up,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((elem_count + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_swiglu_mul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        elem_count,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        return prefill_backend_failure(121, launch_status);
    }
    const hipError_t sync_status = maybe_sync();
    if (sync_status != hipSuccess) {
        return prefill_backend_failure(122, sync_status);
    }
    return 0;
}

template <typename T>
int swiglu_mul_split_device(
    int device_ordinal,
    int rows,
    int cols,
    const void* gate_up,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const int elem_count = rows * cols;
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((elem_count + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_swiglu_mul_split_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        cols,
        static_cast<const T*>(gate_up),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 123;
    if (maybe_sync() != hipSuccess) return 124;
    return 0;
}

template <typename T, typename IndexT>
int embedding_lookup_device(
    int device_ordinal,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total_elems = token_count * hidden_size;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_embedding_lookup_kernel<T, IndexT>),
        dim3(grid),
        dim3(block),
        0,
        0,
        token_count,
        vocab_size,
        hidden_size,
        static_cast<const T*>(embeddings),
        static_cast<const IndexT*>(indexes),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 123;
    if (maybe_sync() != hipSuccess) return 124;
    return 0;
}

template <typename T>
int causal_mask_device(
    int device_ordinal,
    int batch_size,
    int tgt_len,
    int seqlen_offset,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int kv_len = tgt_len + seqlen_offset;
    const int total_elems = batch_size * tgt_len * kv_len;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_causal_mask_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        tgt_len,
        seqlen_offset,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 125;
    if (maybe_sync() != hipSuccess) return 126;
    return 0;
}

template <typename T>
int cumsum_last_dim_device(
    int device_ordinal,
    int rows,
    int cols,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cumsum_last_dim_kernel<T>),
        dim3(static_cast<unsigned int>(rows)),
        dim3(1),
        0,
        0,
        rows,
        cols,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 127;
    if (maybe_sync() != hipSuccess) return 128;
    return 0;
}

template <typename T>
int exp_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_exp_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 129;
    if (maybe_sync() != hipSuccess) return 130;
    return 0;
}

template <typename T>
int recip_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_recip_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 131;
    if (maybe_sync() != hipSuccess) return 132;
    return 0;
}

template <typename T>
int sigmoid_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_sigmoid_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 133;
    if (maybe_sync() != hipSuccess) return 134;
    return 0;
}

template <typename T>
int log_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_log_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 155;
    if (maybe_sync() != hipSuccess) return 156;
    return 0;
}

template <typename In, typename Out>
int cast_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cast_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const In*>(xs),
        static_cast<Out*>(out));
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        return prefill_backend_failure(135, launch_status);
    }
    if (maybe_sync() != hipSuccess) {
        return prefill_backend_failure(136, hipGetLastError());
    }
    return 0;
}

template <typename T>
int unary_view_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 158;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 158;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 158;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_unary_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 159;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 160;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename In, typename Out>
int cast_view_device(
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 161;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 161;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 161;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cast_view_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        total_elems,
        static_cast<const In*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<Out*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 162;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 163;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int binary_broadcast_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_strides_dev = nullptr;
    int* rhs_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (hipMalloc(&lhs_strides_dev, bytes) != hipSuccess) return 137;
    if (hipMalloc(&rhs_strides_dev, bytes) != hipSuccess) {
        hipFree(lhs_strides_dev);
        return 137;
    }
    if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        return 137;
    }
    if (hipMemcpy(lhs_strides_dev, lhs_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(rhs_strides_dev, rhs_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 137;
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_binary_broadcast_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        lhs_strides_dev,
        rhs_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 138;
    }
    if (maybe_sync() != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 139;
    }
    hipFree(lhs_strides_dev);
    hipFree(rhs_strides_dev);
    hipFree(out_dims_dev);
    return 0;
}

template <typename T>
int reduce_keepdim_view_device(
    int device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 167;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 167;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 167;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_out_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_reduce_keepdim_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        reduce_dim,
        reduce_len,
        total_out_elems,
        sum,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 168;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 169;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_dims_dev = nullptr;
    int* rhs_batch_dims_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (hipMalloc(&lhs_batch_dims_dev, bytes) != hipSuccess) return 141;
        if (hipMalloc(&rhs_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            return 141;
        }
        if (hipMalloc(&out_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            return 141;
        }
        if (hipMemcpy(lhs_batch_dims_dev, lhs_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(rhs_batch_dims_dev, rhs_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_batch_dims_dev, out_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
            return 141;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_batched_matmul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_dims_dev,
        rhs_batch_dims_dev,
        out_batch_dims_dev,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
        }
        return 142;
    }
    if (maybe_sync() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
        }
        return 143;
    }
    if (batch_rank > 0) {
        hipFree(lhs_batch_dims_dev);
        hipFree(rhs_batch_dims_dev);
        hipFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_view_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_strides_dev = nullptr;
    int* rhs_batch_strides_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (hipMalloc(&lhs_batch_strides_dev, bytes) != hipSuccess) return 171;
        if (hipMalloc(&rhs_batch_strides_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            return 171;
        }
        if (hipMalloc(&out_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            return 171;
        }
        if (hipMemcpy(lhs_batch_strides_dev, lhs_batch_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(rhs_batch_strides_dev, rhs_batch_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_batch_dims_dev, out_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
            return 171;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_batched_matmul_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_strides_dev,
        rhs_batch_strides_dev,
        out_batch_dims_dev,
        lhs_row_stride,
        lhs_k_stride,
        rhs_k_stride,
        rhs_col_stride,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
        }
        return 172;
    }
    if (maybe_sync() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
        }
        return 173;
    }
    if (batch_rank > 0) {
        hipFree(lhs_batch_strides_dev);
        hipFree(rhs_batch_strides_dev);
        hipFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int mul_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mul_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 145;
    if (maybe_sync() != hipSuccess) return 146;
    return 0;
}

template <typename T>
int reduce_keepdim_device(
    int device_ordinal,
    int outer,
    int reduce,
    int inner,
    bool sum,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total = outer * inner;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_reduce_keepdim_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        outer,
        reduce,
        inner,
        sum ? 1 : 0,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 147;
    if (maybe_sync() != hipSuccess) return 148;
    return 0;
}

template <typename T>
int add_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_add_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 149;
    if (maybe_sync() != hipSuccess) return 150;
    return 0;
}

template <typename T>
int sqrt_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_sqrt_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 151;
    if (maybe_sync() != hipSuccess) return 152;
    return 0;
}

template <typename T>
int delta_full_scan_pack_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 106;
    constexpr int block = 256;
    const size_t total_rows =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total_rows + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(query_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 107;
    if (maybe_sync() != hipSuccess) return 108;
    return 0;
}

template <typename T>
int delta_full_scan_packed_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 109;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_packed_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 110;
    if (maybe_sync() != hipSuccess) return 111;
    return 0;
}

template <typename T>
int l2norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_l2norm_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 90;
    if (maybe_sync() != hipSuccess) return 91;
    return 0;
}

template <typename T>
int value_decay_device(
    int device_ordinal,
    int total_elems,
    int num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_value_decay_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        num_heads,
        static_cast<const T*>(a),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 93;
    if (maybe_sync() != hipSuccess) return 94;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int rms_norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_rms_norm_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        return prefill_backend_failure(71, launch_status);
    }
    const hipError_t sync_status = maybe_sync();
    if (sync_status != hipSuccess) {
        return prefill_backend_failure(72, sync_status);
    }
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int fused_rms_norm_linear_device(
    int device_ordinal,
    int hidden_dim,
    int out_dim,
    float eps,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) + block * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_fused_rms_norm_linear_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(out_dim)),
        dim3(block),
        shared_bytes,
        0,
        hidden_dim,
        out_dim,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(proj_weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 130;
    if (maybe_sync() != hipSuccess) return 131;
    return 0;
}

template <typename T>
int rms_norm_gated_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_rms_norm_gated_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(gate),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (maybe_sync() != hipSuccess) return 82;
    return 0;
}

} // namespace

extern "C" int supersonic_qwen35_hip_full_attention_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out) {
    DecodeBridgeLockGuard guard;

    // Default: use the K-tiled FlashAttention-style kernel for BF16 prefill.
    // SUPERSONIC_PREFILL_ATTN_TILED=0 forces the legacy single-warp
    // online-softmax kernel (kept as a bisect/escape hatch).
    static const bool disable_tiled = []{
        const char* e = std::getenv("SUPERSONIC_PREFILL_ATTN_TILED");
        return e != nullptr && e[0] == '0';
    }();

    if (!disable_tiled && dtype == 2) {  // BF16 only for the tiled path; non-BF16 falls through
        int rc = full_attention_prefill_tiled_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query, key, value, out);
        if (rc == 0) return 0;
        // On any tiled error (LDS overflow, kernel launch fail, dispatch shape unsupported),
        // fall through to the existing non-tiled path. This matches the spirit of an A/B
        // gate: a misconfiguration shouldn't break inference.
    }

    switch (dtype) {
    case 0:
        return full_attention_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 1:
        return full_attention_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 2:
        return full_attention_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    default:
        return 64;
    }
}


extern "C" int supersonic_qwen35_hip_linear_prefill_conv_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_prefill_conv_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 1:
        return linear_prefill_conv_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 2:
        return linear_prefill_conv_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    default:
        return 62;
    }
}

extern "C" int supersonic_qwen35_hip_delta_recurrent_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_recurrent_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out,
            nullptr);
    case 1:
        return delta_recurrent_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out,
            nullptr);
    case 2:
        return delta_recurrent_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out,
            nullptr);
    default:
        return 66;
    }
}

extern "C" int supersonic_qwen35_hip_delta_recurrent_prefill_on_stream(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out,
    void* stream) {
    if (dtype != 1) {
        return 66;
    }
    return delta_recurrent_prefill_device_stream<float>(
        static_cast<int>(device_ordinal),
        static_cast<int>(batch_heads),
        static_cast<int>(seq_len),
        static_cast<int>(k_head_dim),
        static_cast<int>(v_head_dim),
        initial_state,
        query,
        key,
        value,
        beta,
        g,
        out,
        static_cast<hipStream_t>(stream));
}

extern "C" int supersonic_qwen35_hip_decode_rec_k128_fused(
    int device_ordinal,
    int nv,
    int nk,
    float* rec_state,
    const float* q_unique,
    const float* k_unique,
    const float* value,
    const float* b,
    const float* a,
    const hip_bfloat16* dt_bias,
    const hip_bfloat16* a_log_exp,
    float* out,
    void* stream) {
    if (nv <= 0 || nk <= 0 || rec_state == nullptr || q_unique == nullptr ||
        k_unique == nullptr || value == nullptr || b == nullptr || a == nullptr ||
        out == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    constexpr int warps_per_block = 4;
    const size_t total_warps =
        static_cast<size_t>(nv) * static_cast<size_t>(128 / 4);
    const unsigned int grid = static_cast<unsigned int>(
        (total_warps + static_cast<size_t>(warps_per_block) - 1) /
        static_cast<size_t>(warps_per_block));
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_rec_k128_fused_kernel,
        dim3(grid > 0 ? grid : 1u),
        dim3(32, warps_per_block),
        0,
        static_cast<hipStream_t>(stream),
        nv,
        nk,
        rec_state,
        q_unique,
        k_unique,
        value,
        b,
        a,
        dt_bias,
        a_log_exp,
        out);
    return hipGetLastError() == hipSuccess ? 0 : 67;
}


extern "C" int supersonic_qwen35_hip_fill_conv_tail(
    int dtype,
    size_t device_ordinal,
    size_t qkv_dim,
    size_t pad,
    size_t total_len,
    const void* tail,
    void* conv_input) {
    switch (dtype) {
    case 0:
        return fill_conv_tail_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(qkv_dim),
            static_cast<int>(pad),
            static_cast<int>(total_len),
            tail,
            conv_input);
    case 1:
        return fill_conv_tail_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(qkv_dim),
            static_cast<int>(pad),
            static_cast<int>(total_len),
            tail,
            conv_input);
    case 2:
        return fill_conv_tail_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(qkv_dim),
            static_cast<int>(pad),
            static_cast<int>(total_len),
            tail,
            conv_input);
    default:
        return 66;
    }
}


extern "C" int supersonic_qwen35_hip_delta_chunk_single_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_single_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_single_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_single_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 79;
    }
}

extern "C" int supersonic_qwen35_hip_delta_chunk_step(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_step_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_step_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_step_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 86;
    }
}

extern "C" int supersonic_qwen35_hip_delta_chunk_scan_raw(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_scan_raw_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_scan_raw_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_scan_raw_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 87;
    }
}

extern "C" int supersonic_qwen35_hip_delta_state_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_state_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 1:
        return delta_state_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 2:
        return delta_state_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    default:
        return 103;
    }
}

extern "C" int supersonic_qwen35_hip_delta_chunk_fused(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_fused_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 1:
        return delta_chunk_fused_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 2:
        return delta_chunk_fused_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    default:
        return 104;
    }
}

extern "C" int supersonic_qwen35_hip_delta_full_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 1:
        return delta_full_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 2:
        return delta_full_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    default:
        return 105;
    }
}

extern "C" int supersonic_qwen35_hip_delta_full_scan_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 1:
        return delta_full_scan_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 2:
        return delta_full_scan_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    default:
        return 112;
    }
}

extern "C" int supersonic_qwen35_hip_delta_local_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_local_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_local_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_local_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 114;
    }
}

extern "C" int supersonic_qwen35_hip_delta_base_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_base_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_base_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_base_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 117;
    }
}

extern "C" int supersonic_qwen35_hip_delta_attn_solve_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    const void* base_attn_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 1:
        return delta_attn_solve_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 2:
        return delta_attn_solve_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    default:
        return 120;
    }
}

extern "C" int supersonic_qwen35_hip_delta_attn_solve_from_inputs(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_from_inputs_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_attn_solve_from_inputs_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_attn_solve_from_inputs_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 123;
    }
}

extern "C" int supersonic_qwen35_hip_swiglu_mul(
    int dtype,
    size_t device_ordinal,
    size_t elem_count,
    const void* gate,
    const void* up,
    void* out) {
    switch (dtype) {
    case 0:
        return swiglu_mul_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 1:
        return swiglu_mul_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 2:
        return swiglu_mul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    default:
        return 122;
    }
}

extern "C" int supersonic_qwen35_hip_swiglu_mul_split(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* gate_up,
    void* out) {
    switch (dtype) {
    case 0:
        return swiglu_mul_split_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            gate_up,
            out);
    case 1:
        return swiglu_mul_split_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            gate_up,
            out);
    case 2:
        return swiglu_mul_split_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            gate_up,
            out);
    default:
        return 124;
    }
}

extern "C" int supersonic_qwen35_hip_embedding_lookup(
    int dtype,
    int index_dtype,
    size_t device_ordinal,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out) {
    switch (dtype) {
    case 0:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<half, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<half, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<half, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 1:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<float, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<float, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<float, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 2:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<hip_bfloat16, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<hip_bfloat16, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<hip_bfloat16, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    default:
        return 124;
    }
}

template <typename T>
int output_projection_lookup_device(
    int device_ordinal,
    int rows,
    int hidden_size,
    int vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    ScopedHipDevice scoped(device_ordinal);
    const int total_elems = rows * vocab_size;
    const int block = 256;
    const int grid = (total_elems + block - 1) / block;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_output_projection_lookup_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        hidden_size,
        vocab_size,
        static_cast<const T*>(hidden),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 11;
    return 0;
}

extern "C" int supersonic_qwen35_hip_output_projection_lookup(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t hidden_size,
    size_t vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return output_projection_lookup_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 1:
        return output_projection_lookup_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 2:
        return output_projection_lookup_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    default:
        return 122;
    }
}

extern "C" int supersonic_qwen35_hip_causal_mask(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t tgt_len,
    size_t seqlen_offset,
    void* out) {
    switch (dtype) {
    case 0:
        return causal_mask_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 1:
        return causal_mask_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 2:
        return causal_mask_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    default:
        return 126;
    }
}

extern "C" int supersonic_qwen35_hip_cumsum_last_dim(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return cumsum_last_dim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 1:
        return cumsum_last_dim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 2:
        return cumsum_last_dim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    default:
        return 128;
    }
}

extern "C" int supersonic_qwen35_hip_delta_full_scan_packed(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_packed_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 1:
        return delta_full_scan_packed_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 2:
        return delta_full_scan_packed_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    default:
        return 113;
    }
}

extern "C" int supersonic_qwen35_hip_exp(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return exp_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return exp_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return exp_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 129;
    }
}

extern "C" int supersonic_qwen35_hip_recip(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return recip_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return recip_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return recip_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 131;
    }
}

extern "C" int supersonic_qwen35_hip_sigmoid(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sigmoid_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sigmoid_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sigmoid_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 133;
    }
}

extern "C" int supersonic_qwen35_hip_log(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return log_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return log_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return log_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 157;
    }
}

extern "C" int supersonic_qwen35_hip_unary_view(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return unary_view_device<half>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 1:
        return unary_view_device<float>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 2:
        return unary_view_device<hip_bfloat16>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    default:
        return 164;
    }
}

extern "C" int supersonic_qwen35_hip_cast_view(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_view_device<half, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<half, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_view_device<float, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<float, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_view_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    default:
        return 166;
    }
}

extern "C" int supersonic_qwen35_hip_reduce_keepdim_view(
    int dtype,
    size_t device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_view_device<half>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 1:
        return reduce_keepdim_view_device<float>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 2:
        return reduce_keepdim_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    default:
        return 170;
    }
}

extern "C" int supersonic_qwen35_hip_batched_matmul_view(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out) {
    switch (dtype) {
    case 0:
        return batched_matmul_view_device<half>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 1:
        return batched_matmul_view_device<float>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 2:
        return batched_matmul_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    default:
        return 174;
    }
}

extern "C" int supersonic_qwen35_hip_cast(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_device<half, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<half, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_device<float, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<float, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    default:
        return 135;
    }
}

extern "C" int supersonic_qwen35_hip_binary_broadcast(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return binary_broadcast_device<half>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 1:
        return binary_broadcast_device<float>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 2:
        return binary_broadcast_device<hip_bfloat16>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    default:
        return 140;
    }
}

extern "C" int supersonic_qwen35_hip_batched_matmul(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    switch (dtype) {
    case 0:
        return batched_matmul_device<half>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 1:
        return batched_matmul_device<float>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 2:
        return batched_matmul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    default:
        return 144;
    }
}

extern "C" int supersonic_qwen35_hip_mul_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return mul_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return mul_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return mul_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 147;
    }
}

extern "C" int supersonic_qwen35_hip_reduce_keepdim(
    int dtype,
    size_t device_ordinal,
    size_t outer,
    size_t reduce,
    size_t inner,
    int sum,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 1:
        return reduce_keepdim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 2:
        return reduce_keepdim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    default:
        return 149;
    }
}

extern "C" int supersonic_qwen35_hip_add_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return add_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return add_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return add_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 153;
    }
}

extern "C" int supersonic_qwen35_hip_sqrt(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sqrt_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sqrt_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sqrt_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 154;
    }
}

extern "C" int supersonic_qwen35_hip_l2norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return l2norm_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 1:
        return l2norm_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 2:
        return l2norm_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    default:
        return 92;
    }
}

extern "C" int supersonic_qwen35_hip_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    size_t num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 95;
    }
}

extern "C" int supersonic_qwen35_hip_rms_norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? rms_norm_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 1:
        return add_unit_offset
            ? rms_norm_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 2:
        return add_unit_offset
            ? rms_norm_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    default:
        return 74;
    }
}

extern "C" int supersonic_qwen35_hip_fused_rms_norm_linear(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t out_dim,
    float eps,
    int add_unit_offset,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? fused_rms_norm_linear_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 1:
        return add_unit_offset
            ? fused_rms_norm_linear_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 2:
        return add_unit_offset
            ? fused_rms_norm_linear_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    default:
        return 132;
    }
}

extern "C" int supersonic_qwen35_hip_rms_norm_gated(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return rms_norm_gated_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 1:
        return rms_norm_gated_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 2:
        return rms_norm_gated_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    default:
        return 84;
    }
}

template <typename T>
int mlp_decode_megakernel_device(
    int device_ordinal,
    int hidden_dim,
    int intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 200;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 +  // hidden + normed
        block_size * sizeof(float);                              // scratch

    // --- Phase 1: RMSNorm + gate/up projections ---
    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 201;
    if (maybe_sync() != hipSuccess) return 202;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mlp_decode_megakernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        intermediate_size,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(gate_proj_w),
        static_cast<const T*>(up_proj_w),
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 203;
    if (maybe_sync() != hipSuccess) return 204;

    // --- Phase 2: SwiGLU activation ---
    {
        constexpr int swiglu_block = 256;
        const unsigned int swiglu_grid =
            static_cast<unsigned int>((intermediate_size + swiglu_block - 1) / swiglu_block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_mlp_swiglu_kernel<T>),
            dim3(swiglu_grid),
            dim3(swiglu_block),
            0, 0,
            intermediate_size,
            gate_up_scratch);
        if (hipGetLastError() != hipSuccess) return 205;
        if (maybe_sync() != hipSuccess) return 206;
    }

    // --- Phase 3: down_proj matvec ---
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 207;
    if (maybe_sync() != hipSuccess) return 208;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mlp_down_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        block_size * sizeof(float),
        0,
        hidden_dim,
        intermediate_size,
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 209;
    if (maybe_sync() != hipSuccess) return 210;
    return 0;
}

extern "C" int supersonic_qwen35_hip_mlp_decode_megakernel(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return mlp_decode_megakernel_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    case 2:
        return mlp_decode_megakernel_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    default:
        return 205;
    }
}

template <typename T>
int norm_multi_proj_device(
    int device_ordinal,
    int hidden_dim,
    int total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const Qwen35ProjectionDesc* proj_table,
    int num_projections,
    float* output,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 220;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 + block_size * sizeof(float);

    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 221;
    if (maybe_sync() != hipSuccess) return 222;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_norm_multi_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        total_rows,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        proj_table,
        num_projections,
        output,
        row_counter);
    if (hipGetLastError() != hipSuccess) return 223;
    if (maybe_sync() != hipSuccess) return 224;
    return 0;
}

extern "C" int supersonic_qwen35_hip_norm_multi_proj(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* proj_table,       // Qwen35ProjectionDesc* on device
    size_t num_projections,
    float* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return norm_multi_proj_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    case 2:
        return norm_multi_proj_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    default:
        return 225;
    }
}

// Standalone work-stealing matvec: out[out_dim] = W[out_dim, in_dim] × input[in_dim]
// Reuses the down_proj kernel pattern for arbitrary matvec.
template <typename T>
int standalone_matvec_device(
    int device_ordinal,
    int in_dim,
    int out_dim,
    const void* input,       // [in_dim] F32
    const void* weight,      // [out_dim, in_dim] BF16
    void* output,            // [out_dim] BF16
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 230;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 231;
    if (maybe_sync() != hipSuccess) return 232;

    const size_t shared_bytes = block_size * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_standalone_matvec_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        out_dim,
        in_dim,
        static_cast<const T*>(weight),
        static_cast<const T*>(input),
        static_cast<T*>(output),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 233;
    if (maybe_sync() != hipSuccess) return 234;
    return 0;
}

extern "C" int supersonic_qwen35_hip_standalone_matvec(
    int dtype,
    size_t device_ordinal,
    size_t in_dim,
    size_t out_dim,
    const void* input,
    const void* weight,
    void* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return standalone_matvec_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    case 2:
        return standalone_matvec_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    default:
        return 235;
    }
}

// The HIP 0.8B-native persistent decode kernel was deleted on 2026-04-20 —
// it had no INT4/FP8 path and ran ~2.8x slower than routing 0.8B through
// `full_attention_4b.hip`. The retained HIP Qwen3.8 path sets
// `use_4b_kernel = true` in the registry, so `persistent_decode` (non-4B)
// is never dispatched to HIP at runtime. The symbol below is kept as a
// linker-visible stub: the Rust FFI wrapper `kernel_ffi::persistent_decode`
// is shared with the generic non-4B path, so the HIP
// build needs to resolve the reference even though it's unreachable.
extern "C" int supersonic_qwen35_hip_persistent_decode(
    int /*dtype*/,
    size_t /*device_ordinal*/,
    size_t /*num_layers*/,
    size_t /*hidden_dim*/,
    size_t /*intermediate_size*/,
    size_t /*seqlen_offset*/,
    const void* /*layers*/,
    void* /*hidden_io*/,
    float* /*workspace*/,
    unsigned int* /*counters*/,
    unsigned int* /*barrier_counter*/,
    unsigned int* /*barrier_flag*/,
    const void* /*cos_table*/,
    const void* /*sin_table*/,
    size_t /*rotary_dim*/) {
    // Non-zero: clearly distinct from the live kernel's launch/sync error
    // codes (254/255) and the dtype-unsupported code (256). If something
    // ever does dispatch to this on HIP, the caller gets a distinct error
    // it can grep for.
    return 260;
}

extern "C" int supersonic_query_gpu_info(
    int device_ordinal,
    char* arch_name_out,
    size_t arch_name_len,
    uint64_t* total_vram_out) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_ordinal);
    if (err != hipSuccess) {
        return static_cast<int>(err);
    }
    snprintf(arch_name_out, arch_name_len, "%s", props.gcnArchName);
    *total_vram_out = static_cast<uint64_t>(props.totalGlobalMem);
    return 0;
}

extern "C" int supersonic_hip_device_clock_khz(
    int device_ordinal,
    uint32_t* clock_rate_khz_out) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, device_ordinal);
    if (err != hipSuccess) {
        return static_cast<int>(err);
    }
    *clock_rate_khz_out = static_cast<uint32_t>(props.clockRate);
    return 0;
}
