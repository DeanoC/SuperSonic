#include "full_attention_4b.hip"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <mutex>
#include <stdint.h>

extern "C" int supersonic_prefill_encode_bridge_status(
    int project_status,
    int native_status);

extern "C" int supersonic_qwen35_4b_bf16_matmul_bridge_status(
    int project_status,
    int native_status) {
    return supersonic_prefill_encode_bridge_status(project_status, native_status);
}

namespace {

// Per-model launch preset, set once at startup by the Rust registry via
// `supersonic_qwen35_4b_hip_set_launch_preset`. Read by the persistent-decode
// bridge when the user hasn't supplied `SUPERSONIC_QWEN4B_BLOCKS`. Zero
// means "no preset, use the hardcoded gfx11xx default".
int g_preset_blocks = 0;
int g_preset_coop = 0;

inline void qwen4b_get_launch_preset(int& blocks, int& coop) {
    blocks = g_preset_blocks;
    coop = g_preset_coop;
}

} // anonymous namespace

extern "C" void supersonic_qwen35_4b_hip_set_launch_preset(int blocks, int coop) {
    g_preset_blocks = blocks;
    g_preset_coop = coop;
}

namespace {

int prefill_backend_failure(int project_status, hipError_t native_status) {
    return static_cast<int>(
        0x80000000u
        | ((static_cast<uint32_t>(project_status) & 0x7fffu) << 16)
        | (static_cast<uint32_t>(native_status) & 0xffffu));
}

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;

    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) {
            hipSetDevice(target);
            changed = true;
        }
    }

    ~ScopedHipDevice() {
        if (changed && previous >= 0) {
            hipSetDevice(previous);
        }
    }
};

int linear_prefill_block_override() {
    const char* value = std::getenv("DOTCACHE_QWEN35_HIP_FUSED_PREFILL_BLOCK");
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

    // RDNA3 `multiProcessorCount` reports WGPs, not CUs. Oversubscribe 2x
    // on gfx11xx so the prefill attention kernel fills every CU; the
    // kernel's atomic row-counter already handles extra blocks gracefully.
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
    if (hipDeviceSynchronize() != hipSuccess) return 12;

    hipFree(d_row_counter);
    return 0;
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
int linear_stateful_conv_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_linear_stateful_conv_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        seq_len,
        state_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_state),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 62;
    return 0;
}

template <typename T>
int linear_stateful_conv_value_decay_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t out_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * out_width;
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    if (kernel_size == 4 && state_len == 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_linear_stateful_conv_value_decay_kernel_k4s3<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_linear_stateful_conv_value_decay_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            state_len,
            kernel_size,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    }
    if (hipGetLastError() != hipSuccess) return 64;
    return 0;
}

template <typename T>
int linear_stateful_conv_value_decay_with_state_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t total_per_batch = static_cast<size_t>(seq_len) * out_width +
        static_cast<size_t>(conv_dim) * static_cast<size_t>(state_len);
    const size_t out_elems = static_cast<size_t>(batch_size) * total_per_batch;
    const int default_block =
        (kernel_size == 4 && state_len == 3) ? ((seq_len <= 4) ? 256 : 128) : 256;
    const int override_block = linear_prefill_block_override();
    const int block = override_block > 0 ? override_block : default_block;
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    if (kernel_size == 4 && state_len == 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_linear_stateful_conv_value_decay_with_state_kernel_k4s3<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            seq_len,
            num_heads,
            static_cast<const T*>(mixed_qkv),
            static_cast<const T*>(prev_state),
            static_cast<const T*>(weights),
            static_cast<const T*>(a),
            static_cast<const T*>(dt_bias),
            static_cast<const T*>(a_log_exp),
            static_cast<T*>(out));
    } else {
        return 66;
    }
    if (hipGetLastError() != hipSuccess) return 67;
    return 0;
}

template <typename T>
int linear_decode_prepare_device(
    int device_ordinal,
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int state_len,
    int kernel_size,
    int head_repeat,
    const void* mixed_qkv,
    const void* prev_conv_state,
    const void* weights,
    const void* a_beta_raw,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const unsigned int grid =
        static_cast<unsigned int>(batch_size) * static_cast<unsigned int>(num_v_heads);
    unsigned int block = 64;
    while (block < static_cast<unsigned int>(head_k_dim > head_v_dim ? head_k_dim : head_v_dim) &&
           block < 256) {
        block <<= 1;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_linear_decode_prepare_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_len,
        kernel_size,
        head_repeat,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_conv_state),
        static_cast<const T*>(weights),
        static_cast<const T*>(a_beta_raw),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<float*>(out));
    if (hipGetLastError() != hipSuccess) return 69;
    if (maybe_sync() != hipSuccess) return 70;
    return 0;
}

int linear_decode_apply_device(
    int device_ordinal,
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const void* packed,
    const void* initial_state,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    const unsigned int grid =
        static_cast<unsigned int>(batch_size) * static_cast<unsigned int>(num_v_heads);
    unsigned int block = 64;
    while (block < static_cast<unsigned int>(head_v_dim) && block < 256) {
        block <<= 1;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_linear_decode_apply_kernel<>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        static_cast<const float*>(packed),
        static_cast<const float*>(initial_state),
        static_cast<float*>(out));
    if (hipGetLastError() != hipSuccess) return 71;
    if (maybe_sync() != hipSuccess) return 72;
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
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
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
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 67;
    if (maybe_sync() != hipSuccess) return 68;
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
    if (hipDeviceSynchronize() != hipSuccess) return 78;
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
    if (hipDeviceSynchronize() != hipSuccess) return 82;
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
    if (hipDeviceSynchronize() != hipSuccess) return 85;
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
    if (hipDeviceSynchronize() != hipSuccess) return 96;
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
    if (hipDeviceSynchronize() != hipSuccess) return 99;
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
    if (hipDeviceSynchronize() != hipSuccess) return 102;
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
    if (hipDeviceSynchronize() != hipSuccess) return 114;
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
    if (hipDeviceSynchronize() != hipSuccess) return 117;
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
    if (hipDeviceSynchronize() != hipSuccess) return 120;
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
    if (hipDeviceSynchronize() != hipSuccess) return 123;
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
    if (hipGetLastError() != hipSuccess) return 121;
    if (maybe_sync() != hipSuccess) return 122;
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
    if (hipDeviceSynchronize() != hipSuccess) return 126;
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
    if (hipDeviceSynchronize() != hipSuccess) return 128;
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
    if (hipDeviceSynchronize() != hipSuccess) return 130;
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
    if (hipDeviceSynchronize() != hipSuccess) return 132;
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
    if (hipDeviceSynchronize() != hipSuccess) return 134;
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
    if (hipDeviceSynchronize() != hipSuccess) return 156;
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
    if (hipGetLastError() != hipSuccess) return 135;
    if (hipDeviceSynchronize() != hipSuccess) return 136;
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) {
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
    if (hipDeviceSynchronize() != hipSuccess) return 148;
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
    if (hipDeviceSynchronize() != hipSuccess) return 150;
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
    if (hipDeviceSynchronize() != hipSuccess) return 152;
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
    if (hipDeviceSynchronize() != hipSuccess) return 108;
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
    if (hipDeviceSynchronize() != hipSuccess) return 111;
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
    if (hipGetLastError() != hipSuccess) return 71;
    if (maybe_sync() != hipSuccess) return 72;
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

extern "C" int supersonic_qwen35_4b_hip_full_attention_prefill(
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

extern "C" int supersonic_qwen35_4b_hip_linear_prefill_conv_pack(
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

extern "C" int supersonic_qwen35_4b_hip_linear_stateful_conv(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 1:
        return linear_stateful_conv_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 2:
        return linear_stateful_conv_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    default:
        return 63;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_recurrent_prefill(
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
            out);
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
            out);
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
            out);
    default:
        return 66;
    }
}

extern "C" int supersonic_qwen35_4b_hip_linear_stateful_conv_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_stateful_conv_value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_stateful_conv_value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 67;
    }
}

extern "C" int supersonic_qwen35_4b_hip_linear_stateful_conv_value_decay_with_state(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_value_decay_with_state_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_stateful_conv_value_decay_with_state_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_stateful_conv_value_decay_with_state_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 68;
    }
}

extern "C" int supersonic_qwen35_4b_hip_linear_decode_prepare(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t num_v_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    size_t state_len,
    size_t kernel_size,
    size_t head_repeat,
    const void* mixed_qkv,
    const void* prev_conv_state,
    const void* weights,
    const void* a_beta_raw,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_decode_prepare_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_decode_prepare_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_decode_prepare_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(num_v_heads),
            static_cast<int>(head_k_dim),
            static_cast<int>(head_v_dim),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(head_repeat),
            mixed_qkv,
            prev_conv_state,
            weights,
            a_beta_raw,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 73;
    }
}

extern "C" int supersonic_qwen35_4b_hip_linear_decode_apply(
    size_t device_ordinal,
    size_t batch_size,
    size_t num_v_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    const void* packed,
    const void* initial_state,
    void* out) {
    return linear_decode_apply_device(
        static_cast<int>(device_ordinal),
        static_cast<int>(batch_size),
        static_cast<int>(num_v_heads),
        static_cast<int>(head_k_dim),
        static_cast<int>(head_v_dim),
        packed,
        initial_state,
        out);
}

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_single_prefill(
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

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_step(
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

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_scan_raw(
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

extern "C" int supersonic_qwen35_4b_hip_delta_state_scan(
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

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_fused(
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

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan(
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

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan_pack(
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

extern "C" int supersonic_qwen35_4b_hip_delta_local_attn_scan(
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

extern "C" int supersonic_qwen35_4b_hip_delta_base_attn_scan(
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

extern "C" int supersonic_qwen35_4b_hip_delta_attn_solve_scan(
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

extern "C" int supersonic_qwen35_4b_hip_delta_attn_solve_from_inputs(
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

extern "C" int supersonic_qwen35_4b_hip_swiglu_mul(
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

extern "C" int supersonic_qwen35_4b_hip_embedding_lookup(
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

extern "C" int supersonic_qwen35_4b_hip_output_projection_lookup(
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

extern "C" int supersonic_qwen35_4b_hip_causal_mask(
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

extern "C" int supersonic_qwen35_4b_hip_cumsum_last_dim(
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

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan_packed(
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

extern "C" int supersonic_qwen35_4b_hip_exp(
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

extern "C" int supersonic_qwen35_4b_hip_recip(
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

extern "C" int supersonic_qwen35_4b_hip_sigmoid(
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

extern "C" int supersonic_qwen35_4b_hip_log(
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

extern "C" int supersonic_qwen35_4b_hip_unary_view(
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

extern "C" int supersonic_qwen35_4b_hip_cast_view(
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

extern "C" int supersonic_qwen35_4b_hip_reduce_keepdim_view(
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

extern "C" int supersonic_qwen35_4b_hip_batched_matmul_view(
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

// Tiled BF16 matmul for prefill: out = lhs × rhs^T (rhs stored [n, k])
template <typename T>
int matmul_rhs_transposed_tiled_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;  // 256
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_tiled_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(270, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(271, sync_err);
    return 0;
}

// Cached per-device arch flag: does this device support the validated RDNA WMMA
// path? gfx11xx (RDNA3 / RDNA3.5) supports the original wave32 BF16/i8 layout.
// gfx12xx (RDNA4) uses the new `_gfx12` operand layout, adapted in
// full_attention_4b.hip.
//
// Env var SUPERSONIC_QWEN4B_DISABLE_WMMA=1 forces the scalar path for debugging
// / perf comparison.
//
// `supersonic-serve` can call this concurrently from multiple request threads,
// so initialization goes through `std::call_once` — plain non-atomic writes
// would be a data race.
static bool device_supports_wmma_bf16(int device_ordinal) {
    static std::once_flag env_once;
    static bool env_disabled = false;
    std::call_once(env_once, [] {
        const char* env = std::getenv("SUPERSONIC_QWEN4B_DISABLE_WMMA");
        env_disabled = (env != nullptr && env[0] != '\0' && env[0] != '0');
    });
    if (env_disabled) return false;

    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        if (!arch || arch[0] != 'g' || arch[1] != 'f' || arch[2] != 'x' ||
            arch[3] != '1') {
            return false;
        }
        if (arch[4] == '1') return true;
        return arch[4] == '2';
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        // Uncached lookup for unusual ordinals — happens at most once per call
        // for a device outside the cached range.
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

static bool device_supports_wmma_i8(int device_ordinal) {
    // gfx12 i8 WMMA drops the gfx11 operand replication and uses a narrower
    // packed A/B operand. The device code adapts the existing four-word packed
    // fragments to the gfx12 two-word form.
    if (!device_supports_wmma_bf16(device_ordinal)) return false;

    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        return arch && arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
               arch[3] == '1' && (arch[4] == '1' || arch[4] == '2');
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

bool device_is_gfx12(int device_ordinal) {
    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        return arch && arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
               arch[3] == '1' && arch[4] == '2';
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

static int matmul_rhs_transposed_tiled_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    // Must match the TILED_WMMA_B{M,N} constants in full_attention_4b.hip.
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 128;  // 4 wavefronts per block, arranged 2x2
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_rhs_transposed_tiled_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const hip_bfloat16*>(rhs),
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            280, static_cast<int>(launch_err));
    }
    if (sync_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            281, static_cast<int>(sync_err));
    }
    return 0;
}

static int matmul_rhs_transposed_wmma_small_m_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    if (device_is_gfx12(device_ordinal)) {
        const bool hot_exact =
            m == TILE_M && (n % TILE_N) == 0 && (k % 16) == 0 &&
            std::getenv("SUPERSONIC_DFLASH_DISABLE_BF16_SMALL_M_HOT_EXACT") == nullptr;
        if (hot_exact) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_gfx12_kernel<true>),
                dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const hip_bfloat16*>(rhs),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_gfx12_kernel<false>),
                dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const hip_bfloat16*>(rhs),
                static_cast<hip_bfloat16*>(out));
        }
    } else {
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const hip_bfloat16*>(rhs),
            static_cast<hip_bfloat16*>(out));
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            282, static_cast<int>(launch_err));
    }
    if (sync_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            283, static_cast<int>(sync_err));
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out) {
    switch (dtype) {
    case 2:
        if (device_supports_wmma_bf16(static_cast<int>(device_ordinal))) {
            const bool disable_small_m =
                std::getenv("SUPERSONIC_DFLASH_DISABLE_BF16_SMALL_M") != nullptr;
            if (!disable_small_m && m < 32) {
                return matmul_rhs_transposed_wmma_small_m_bf16_device(
                    static_cast<int>(device_ordinal), batch_elems, m, n, k,
                    lhs, rhs, out);
            }
            return matmul_rhs_transposed_tiled_wmma_bf16_device(
                static_cast<int>(device_ordinal), batch_elems, m, n, k,
                lhs, rhs, out);
        }
        return matmul_rhs_transposed_tiled_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs, out);
    default:
        return 272;
    }
}

// FP8 dequant matmul for prefill: out = lhs (BF16) × dequant(rhs_fp8)^T
// Uses tiled kernel with 3D grid: (n_tiles, m_tiles, batch)
template <typename T>
int matmul_fp8_dequant_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;  // 256
    // Shared memory: s_lhs[16][32] + s_rhs[16][32] = 2 * 16 * 32 * 4 = 4096 bytes
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_fp8_dequant_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const uint8_t*>(rhs_fp8),
        static_cast<const T*>(scale),
        block_size,
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 260;
    if (sync_err != hipSuccess) return 261;
    return 0;
}

// WMMA-tiled FP8 dequant matmul for BF16 activations. Same 64x64 block tile
// shape as the BF16 tiled path; reads FP8 bytes from global, dequantizes
// into LDS as BF16, then runs WMMAs from LDS. Only activated when
// block_size is a multiple of TILED_WMMA_BK=64 so every BK-aligned K slab
// (and the 64-row N slab) lies inside a single FP8 scale block. The
// shipped lovedheart Qwen FP8 bakes use block_size=128.
static int matmul_fp8_dequant_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    constexpr int threads = 128;  // 4 wavefronts 2x2
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_fp8_dequant_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_fp8),
        static_cast<const hip_bfloat16*>(scale),
        block_size,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 262;
    if (sync_err != hipSuccess) return 263;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_fp8_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out) {
    switch (dtype) {
    case 2: {
        // WMMA fast path when block_size is a multiple of the tile's K slab
        // (= 64), and m is large enough that the 64-row tile doesn't waste
        // most of its compute on overhang. Otherwise fall back to the scalar
        // FP32-accumulate tiled kernel (which is what shipped before WMMA).
        constexpr int TILED_BK = 64;
        constexpr int TILED_M_THRESHOLD = 32;
        const int ordinal = static_cast<int>(device_ordinal);
        if (m >= TILED_M_THRESHOLD && block_size % TILED_BK == 0 &&
            device_supports_wmma_bf16(ordinal)) {
            return matmul_fp8_dequant_wmma_bf16_device(
                ordinal, batch_elems, m, n, k,
                lhs, rhs_fp8, scale, block_size, out);
        }
        return matmul_fp8_dequant_device<hip_bfloat16>(
            ordinal, batch_elems, m, n, k,
            lhs, rhs_fp8, scale, block_size, out);
    }
    default:
        return 262;
    }
}

// INT4 dequant matmul bridge.
template <typename T>
int matmul_int4_dequant_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_int4_dequant_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const uint8_t*>(rhs_int4),
        static_cast<const T*>(scale),
        static_cast<const T*>(zero),
        static_cast<const T*>(awq_inv_scale),
        group_size,
        quant_type,
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return prefill_backend_failure(270, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(271, sync_err);
    return 0;
}

static int matmul_int4_dequant_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_NATIVE_INT4 = 4;
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool native_int4 = quant_type == QWEN35_LOWBIT_NATIVE_INT4;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    const bool disable_ggml_small_m =
        ggml_k && std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_SMALL_M") != nullptr;

    // Tiled WMMA is a clear win when M is large enough to use most of the
    // 64-row block tile (long-ctx prefill). For small M (short prompts,
    // decode-like verify blocks) the 4x row overhang dominates, so dispatch
    // to the one-wave-per-16x16-tile kernel in that regime, including raw
    // GGML K-block vocab projections.
    constexpr int TILED_M_THRESHOLD = 32;
    const bool raw_ggml_small_m =
        ggml_k && !disable_ggml_small_m;
    if ((native_int4 || raw_ggml_small_m) && m < TILED_M_THRESHOLD) {
        const bool enable_ggml_small_m_n64 =
            ggml_k && std::getenv("SUPERSONIC_DFLASH_ENABLE_GGML_SMALL_M_N64") != nullptr;
        if (raw_ggml_small_m && enable_ggml_small_m_n64) {
            constexpr int TILE_M = 16;
            constexpr int TILE_N = 64;
            const int grid_x = (n + TILE_N - 1) / TILE_N;
            const int grid_y = (m + TILE_M - 1) / TILE_M;
            const int grid_z = static_cast<int>(batch_elems);
            const int threads = 128;
            hipLaunchKernelGGL(
                supersonic_qwen35_matmul_int4_dequant_wmma_small_m_n64_kernel,
                dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(scale),
                static_cast<const hip_bfloat16*>(zero),
                static_cast<const hip_bfloat16*>(awq_inv_scale),
                group_size,
                quant_type,
                static_cast<hip_bfloat16*>(out));
            hipError_t launch_err = hipGetLastError();
            hipError_t sync_err = maybe_sync();
            if (launch_err != hipSuccess) return prefill_backend_failure(292, launch_err);
            if (sync_err != hipSuccess) return prefill_backend_failure(293, sync_err);
            return 0;
        }

        const bool disable_ggml_small_m_qtype =
            raw_ggml_small_m &&
            std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_SMALL_M_QTYPE") != nullptr;
        if (raw_ggml_small_m && !disable_ggml_small_m_qtype) {
            const bool trunc_dequant =
                quant_type != QWEN35_LOWBIT_GGML_Q8_0 &&
                std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_TRUNC_DEQUANT") == nullptr;
            constexpr int TILE_M = 16;
            constexpr int TILE_N = 16;
            const int grid_x = (n + TILE_N - 1) / TILE_N;
            const int grid_y = (m + TILE_M - 1) / TILE_M;
            const int grid_z = static_cast<int>(batch_elems);
            const int threads = 32;
            const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
            const bool enable_m8_block =
                m == 8 && (n % TILE_N) == 0 && (k % 256) == 0 && awq_inv_scale == nullptr &&
                std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_M8_BLOCK") == nullptr;
            const bool enable_m16_block =
                m == 16 && (n % TILE_N) == 0 && (k % 256) == 0 && awq_inv_scale == nullptr &&
                std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_M16_BLOCK") == nullptr;
            if (enable_m16_block) {
                if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                }
                hipError_t launch_err = hipGetLastError();
                hipError_t sync_err = maybe_sync();
                if (launch_err != hipSuccess) return prefill_backend_failure(298, launch_err);
                if (sync_err != hipSuccess) return prefill_backend_failure(299, sync_err);
                return 0;
            }
            if (enable_m8_block) {
                const bool use_m8_gfx12 =
                    use_gfx12_acc &&
                    std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_M8_GFX12_NATIVE") == nullptr;
                if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                }
                hipError_t launch_err = hipGetLastError();
                hipError_t sync_err = maybe_sync();
                if (launch_err != hipSuccess) return prefill_backend_failure(296, launch_err);
                if (sync_err != hipSuccess) return prefill_backend_failure(297, sync_err);
                return 0;
            }
            if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            }
            hipError_t launch_err = hipGetLastError();
            hipError_t sync_err = maybe_sync();
            if (launch_err != hipSuccess) return prefill_backend_failure(294, launch_err);
            if (sync_err != hipSuccess) return prefill_backend_failure(295, sync_err);
            return 0;
        }

        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16;
        const int grid_x = (n + TILE_N - 1) / TILE_N;
        const int grid_y = (m + TILE_M - 1) / TILE_M;
        const int grid_z = static_cast<int>(batch_elems);
        const int threads = 32;
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_int4_dequant_wmma_small_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const uint8_t*>(rhs_int4),
            static_cast<const hip_bfloat16*>(scale),
            static_cast<const hip_bfloat16*>(zero),
            static_cast<const hip_bfloat16*>(awq_inv_scale),
            group_size,
            quant_type,
            static_cast<hip_bfloat16*>(out));
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return prefill_backend_failure(290, launch_err);
        if (sync_err != hipSuccess) return prefill_backend_failure(291, sync_err);
        return 0;
    }

    // Large-M: tiled 64x64 block tile, 4 waves in 2x2.
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 128;
    const bool use_gfx12_large_q8 =
        device_is_gfx12(device_ordinal) &&
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 &&
        awq_inv_scale == nullptr &&
        (n % TILE_N) == 0 &&
        (k % 64) == 0 &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_Q8_LARGE_M_GFX12_NATIVE") == nullptr;
    if (use_gfx12_large_q8) {
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_ggml_q8_0_wmma_gfx12_large_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const uint8_t*>(rhs_int4),
            static_cast<hip_bfloat16*>(out));
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return prefill_backend_failure(300, launch_err);
        if (sync_err != hipSuccess) return prefill_backend_failure(301, sync_err);
        return 0;
    }
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_int4_dequant_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_int4),
        static_cast<const hip_bfloat16*>(scale),
        static_cast<const hip_bfloat16*>(zero),
        static_cast<const hip_bfloat16*>(awq_inv_scale),
        group_size,
        quant_type,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(290, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(291, sync_err);
    return 0;
}

static int matmul_int4_dequant_residual_add_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* awq_inv_scale,
    int quant_type,
    const void* residual,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 312;
    if (!device_supports_wmma_bf16(device_ordinal)) return 313;
    if (lhs == nullptr || rhs_int4 == nullptr || residual == nullptr || out == nullptr) return 314;
    if (m != 16 || n <= 0 || k <= 0 || (n % 16) != 0 || (k % 256) != 0) return 315;
    if (awq_inv_scale != nullptr) return 316;

    const bool trunc_dequant =
        quant_type != QWEN35_LOWBIT_GGML_Q8_0 &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_TRUNC_DEQUANT") == nullptr;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);

    if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
        if (trunc_dequant) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
        if (trunc_dequant) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        }
    }

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 317;
    if (sync_err != hipSuccess) return 318;
    return 0;
}

static int matmul_ggml_pair_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_first,
    const void* rhs_second,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 294;
    if (!device_supports_wmma_bf16(device_ordinal)) return 295;

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const bool enable_m16_qtype =
        m == TILE_M && (n_each % TILE_N) == 0 && (k % 256) == 0 &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_PAIR_M16_QTYPE") == nullptr;
    if (enable_m16_qtype) {
        const int n_out = n_each * 2;
        const int grid_x = (n_out + TILE_N - 1) / TILE_N;
        const int grid_z = static_cast<int>(batch_elems);
        const int threads = 32;
        const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
        const bool trunc_dequant =
            quant_type != QWEN35_LOWBIT_GGML_Q8_0 &&
            std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_TRUNC_DEQUANT") == nullptr;
        if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_first),
                static_cast<const uint8_t*>(rhs_second),
                static_cast<hip_bfloat16*>(out));
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
            if (trunc_dequant) {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            } else {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            }
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
            if (trunc_dequant) {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            } else {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            }
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
            if (trunc_dequant) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_first),
                    static_cast<const uint8_t*>(rhs_second),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_first),
                    static_cast<const uint8_t*>(rhs_second),
                    static_cast<hip_bfloat16*>(out));
            }
        }
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return 320;
        if (sync_err != hipSuccess) return 321;
        return 0;
    }

    const int n_out = n_each * 2;
    const int grid_x = (n_out + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_ggml_pair_wmma_small_m_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n_each, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_first),
        static_cast<const uint8_t*>(rhs_second),
        quant_type,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 296;
    if (sync_err != hipSuccess) return 297;
    return 0;
}

static int matmul_ggml_pair_swiglu_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_gate,
    const void* rhs_up,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 322;
    if (!device_supports_wmma_bf16(device_ordinal)) return 323;

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    if (m != TILE_M || (n_each % TILE_N) != 0 || (k % 256) != 0) return 324;

    const int grid_x = (n_each + TILE_N - 1) / TILE_N;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    const bool trunc_dequant =
        quant_type != QWEN35_LOWBIT_GGML_Q8_0 &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_TRUNC_DEQUANT") == nullptr;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
    if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_gate),
                static_cast<const uint8_t*>(rhs_up),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_gate),
                static_cast<const uint8_t*>(rhs_up),
                static_cast<hip_bfloat16*>(out));
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 325;
    if (sync_err != hipSuccess) return 326;
    return 0;
}

static int quantize_mmq_q8_1_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int k,
    const void* lhs,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 300;
    if (m <= 0 || k <= 0 || (k % 128) != 0) return 301;

    const int blocks_per_row = (k + 127) / 128;
    hipLaunchKernelGGL(
        supersonic_qwen35_quantize_mmq_q8_1_kernel,
        dim3(blocks_per_row, m, static_cast<unsigned int>(batch_elems)),
        dim3(32),
        0,
        0,
        batch_elems,
        m,
        k,
        static_cast<const hip_bfloat16*>(lhs),
        quant_type,
        static_cast<uint8_t*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 302;
    if (sync_err != hipSuccess) return 303;
    return 0;
}

static int matmul_mmq_q8_1_q6_k_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    const void* residual,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (!device_supports_wmma_i8(device_ordinal)) return 309;
    if (m <= 0 || n <= 0 || k <= 0 || (k % 256) != 0) return 305;
    constexpr int MMQ_X = 16;
    constexpr int MMQ_Y = 128;
    constexpr int PADDED_TILE_Y_INTS = 768;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K = 76;
    const int grid_x = (n + MMQ_Y - 1) / MMQ_Y;
    const int grid_y = (m + MMQ_X - 1) / MMQ_X;
    const int grid_z = static_cast<int>(batch_elems);
    const size_t shared_bytes =
        static_cast<size_t>(PADDED_TILE_Y_INTS + MMQ_Y * MMQ_MMA_TILE_X_K_Q6_K) *
        sizeof(int);
    const bool hot_exact =
        m == MMQ_X && (n % MMQ_Y) == 0 &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_Q6_K_MMQ_HOT_EXACT") == nullptr;
    const bool use_gfx12_native =
        hot_exact &&
        device_is_gfx12(device_ordinal) &&
        std::getenv("SUPERSONIC_DFLASH_DISABLE_Q6_K_MMQ_GFX12_NATIVE") == nullptr;
    const bool has_residual = residual != nullptr;

    if (use_gfx12_native && has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_gfx12_kernel<true, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (use_gfx12_native) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_gfx12_kernel<true, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (hot_exact && has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<true, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (hot_exact) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<true, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<false, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<false, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 306;
    if (sync_err != hipSuccess) return 307;
    return 0;
}

static int matmul_q6_k_m16_argmax_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs_q6,
    void* block_best_vals,
    void* block_best_indices,
    void* out_indices
) {
    ScopedHipDevice scoped(device_ordinal);
    if (!device_supports_wmma_bf16(device_ordinal)) return 340;
    if (m != 16 || n <= 0 || k <= 0 || (n % 16) != 0 || (k % 256) != 0) return 341;
    if (lhs == nullptr || rhs_q6 == nullptr || block_best_vals == nullptr ||
        block_best_indices == nullptr || out_indices == nullptr) return 342;

    const int tiles = n / 16;
    const int grid_z = static_cast<int>(batch_elems);
    const bool trunc_dequant =
        std::getenv("SUPERSONIC_DFLASH_DISABLE_GGML_TRUNC_DEQUANT") == nullptr;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
    if (trunc_dequant) {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_gfx12_kernel<true>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_kernel<true>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        }
    } else {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_gfx12_kernel<false>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_kernel<false>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        }
    }
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) return 343;

    if (use_gfx12_acc) {
        hipLaunchKernelGGL(
            supersonic_qwen35_reduce_m16_tile_argmax_tile_major_kernel,
            dim3(16, grid_z),
            dim3(256),
            0,
            0,
            batch_elems,
            tiles,
            static_cast<const float*>(block_best_vals),
            static_cast<const uint32_t*>(block_best_indices),
            static_cast<uint32_t*>(out_indices));
    } else {
        hipLaunchKernelGGL(
            supersonic_qwen35_reduce_m16_tile_argmax_kernel,
            dim3(16, grid_z),
            dim3(256),
            0,
            0,
            batch_elems,
            tiles,
            static_cast<const float*>(block_best_vals),
            static_cast<const uint32_t*>(block_best_indices),
            static_cast<uint32_t*>(out_indices));
    }
    launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 344;
    if (sync_err != hipSuccess) return 345;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_device_supports_wmma_i8(
    size_t device_ordinal,
    int* out_supported) {
    if (out_supported == nullptr) return 310;
    *out_supported = device_supports_wmma_i8(static_cast<int>(device_ordinal)) ? 1 : 0;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out) {
    constexpr int QWEN35_LOWBIT_NATIVE_INT4 = 4;
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool native_int4 = quant_type == QWEN35_LOWBIT_NATIVE_INT4;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!native_int4 && !ggml_k) {
        return 273;
    }

    switch (dtype) {
    case 2: {
        // The tiled WMMA kernel fetches one (scale, zero) pair per BK-wide
        // K slab per row, so it's only correct when every BK-aligned slab
        // stays inside a single quantization group. That requires
        // group_size to be a multiple of TILED_WMMA_BK (= 64). The shipped
        // GPTQ bakes use group_size=128, so this path activates in
        // practice; other group sizes fall back to the scalar kernel.
        constexpr int TILED_BK = 64;
        if (device_supports_wmma_bf16(static_cast<int>(device_ordinal)) &&
            ((native_int4 && group_size % TILED_BK == 0) || ggml_k)) {
            return matmul_int4_dequant_wmma_bf16_device(
                static_cast<int>(device_ordinal), batch_elems, m, n, k,
                lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out);
        }
        return matmul_int4_dequant_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out);
    }
    default:
        return 272;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    const void* residual,
    void* out) {
    (void)scale;
    (void)zero;
    (void)group_size;
    switch (dtype) {
    case 2:
        return matmul_int4_dequant_residual_add_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_int4, awq_inv_scale, quant_type, residual, out);
    default:
        return 319;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_first,
    const void* rhs_second,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_ggml_pair_wmma_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n_each, k,
            lhs, rhs_first, rhs_second, quant_type, out);
    default:
        return 298;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_gate,
    const void* rhs_up,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_ggml_pair_swiglu_wmma_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n_each, k,
            lhs, rhs_gate, rhs_up, quant_type, out);
    default:
        return 327;
    }
}

extern "C" int supersonic_qwen35_4b_hip_quantize_mmq_q8_1(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int k,
    const void* lhs,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 2:
        return quantize_mmq_q8_1_device(
            static_cast<int>(device_ordinal), batch_elems, m, k, lhs, quant_type, out);
    default:
        return 304;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_mmq_q8_1_q6_k_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k, q8, rhs_q6, nullptr, out);
    default:
        return 308;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k_residual_add(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    const void* residual,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_mmq_q8_1_q6_k_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k, q8, rhs_q6, residual, out);
    default:
        return 330;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_q6_k_m16_argmax(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs_q6,
    void* block_best_vals,
    void* block_best_indices,
    void* out_indices) {
    switch (dtype) {
    case 2:
        return matmul_q6_k_m16_argmax_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_q6, block_best_vals, block_best_indices, out_indices);
    default:
        return 346;
    }
}

template <typename T>
int int4_sparse_outlier_add_device(
    int device_ordinal,
    int rows,
    int n,
    int k,
    int sub_cols,
    const void* lhs,
    const void* outlier_cols,
    const void* outlier_delta,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(n);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_int4_sparse_outlier_add_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        n,
        k,
        sub_cols,
        static_cast<const T*>(lhs),
        static_cast<const uint32_t*>(outlier_cols),
        static_cast<const T*>(outlier_delta),
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 294;
    if (sync_err != hipSuccess) return 295;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_int4_sparse_outlier_add(
    int dtype,
    size_t device_ordinal,
    int rows,
    int n,
    int k,
    int sub_cols,
    const void* lhs,
    const void* outlier_cols,
    const void* outlier_delta,
    void* out
) {
    switch (dtype) {
    case 2:
        return int4_sparse_outlier_add_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), rows, n, k, sub_cols,
            lhs, outlier_cols, outlier_delta, out);
    default:
        return 296;
    }
}

extern "C" int supersonic_qwen35_4b_hip_cast(
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

extern "C" int supersonic_qwen35_4b_hip_binary_broadcast(
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

extern "C" int supersonic_qwen35_4b_hip_batched_matmul(
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

extern "C" int supersonic_qwen35_4b_hip_mul_scalar(
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

extern "C" int supersonic_qwen35_4b_hip_reduce_keepdim(
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

extern "C" int supersonic_qwen35_4b_hip_add_scalar(
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

extern "C" int supersonic_qwen35_4b_hip_sqrt(
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

extern "C" int supersonic_qwen35_4b_hip_l2norm(
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

extern "C" int supersonic_qwen35_4b_hip_value_decay(
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

extern "C" int supersonic_qwen35_4b_hip_rms_norm(
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

extern "C" int supersonic_qwen35_4b_hip_fused_rms_norm_linear(
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

extern "C" int supersonic_qwen35_4b_hip_rms_norm_gated(
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
    if (hipDeviceSynchronize() != hipSuccess) return 202;

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
    if (hipDeviceSynchronize() != hipSuccess) return 204;

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
        if (hipDeviceSynchronize() != hipSuccess) return 206;
    }

    // --- Phase 3: down_proj matvec ---
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 207;
    if (hipDeviceSynchronize() != hipSuccess) return 208;

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
    if (hipDeviceSynchronize() != hipSuccess) return 210;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_mlp_decode_megakernel(
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
    if (hipDeviceSynchronize() != hipSuccess) return 222;

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
    if (hipDeviceSynchronize() != hipSuccess) return 224;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_norm_multi_proj(
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
    if (hipDeviceSynchronize() != hipSuccess) return 232;

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
    if (hipDeviceSynchronize() != hipSuccess) return 234;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_standalone_matvec(
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

template <typename T>
int persistent_decode_device(
    int device_ordinal,
    int num_layers,
    int hidden_dim,
    int intermediate_size,
    int seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    unsigned long long* timing_slots,
    const void* cos_table,
    const void* sin_table,
    int rotary_dim,
    int proj_buf_floats,
    int attn_scratch_floats,
    int enable_attention_trace,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    int batch_size,
    const void* batch_descs,
    const void* int4_scales,
    void* tap_workspace,
    const int* tap_layers,
    int num_taps
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 250;

    // rocprofv3 on gfx1150 revealed `multiProcessorCount` reports 8 (WGP
    // count on RDNA3, not CU count) while the device has 16 CUs — the
    // original 1x grid left half the GPU idle with only 1 block/CU, and
    // register pressure (VGPR=128) prevents a second resident block.
    // Oversubscribing 2x on RDNA3 (= one block per CU) is a proven safe
    // win: ~1.57x faster on qwen3.5-0.8b BF16, no hangs across tested
    // Qwen variants.
    //
    // HIP docs note `multiProcessorCount` reports CUs on CDNA (MI-series)
    // and WGPs on RDNA. On arches where it already reports CU count, 2x
    // would over-subscribe and can deadlock via `grid_barrier` when only
    // one block/CU is resident. Restrict the default multiplier to the
    // arches we've actually validated (gfx11xx RDNA3/3.5 at time of
    // writing); other arches get the conservative 1x default.
    //
    // Higher multipliers (3x+) hang silently on models with more
    // transformer layers (qwen3.5-2b at 4x produces no output) —
    // suspected grid_barrier scaling issue. Env var
    // Grid-size priority (first match wins):
    //   1. SUPERSONIC_QWEN4B_BLOCKS env var (explicit user override).
    //   2. Per-model preset set via `supersonic_qwen35_4b_hip_set_launch_preset`
    //      from the Rust registry (e.g. 0.8B gets 32 + cooperative).
    //   3. 2x multiProcessorCount default on RDNA3/gfx11xx (empirically
    //      safe at non-cooperative launch on every tested Qwen variant).
    //
    // Cooperative launch is enabled when SUPERSONIC_QWEN4B_COOP is set OR
    // when the active preset opts in. The homebrew `grid_barrier` assumes
    // every block is co-resident; cooperative launch enforces that and
    // fails cleanly on over-subscription instead of deadlocking.
    //
    // Why cooperative is opt-in rather than always-on: `hipOccupancyMax-
    // ActiveBlocksPerMultiprocessor` is strictly conservative — on 4B it
    // reports 1 block/MP while non-cooperative launch empirically handles
    // 2. Cooperative-by-default would regress 4B throughput.
    int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    int preset_blocks = 0, preset_coop = 0;
    qwen4b_get_launch_preset(preset_blocks, preset_coop);
    bool preset_coop_hint = false;
    bool user_grid = false;
    if (const char* bs_env = std::getenv("SUPERSONIC_QWEN4B_BLOCKS")) {
        int override_val = std::atoi(bs_env);
        if (override_val > 0) {
            num_blocks = override_val;
            user_grid = true;
        }
    } else if (preset_blocks > 0) {
        num_blocks = preset_blocks;
        preset_coop_hint = preset_coop != 0;
    }
    const bool coop_requested =
        std::getenv("SUPERSONIC_QWEN4B_COOP") != nullptr || preset_coop_hint;
    constexpr int block_size = 256;
    const size_t fp8_lut_size =
        (fp8_scales != nullptr || kv_fp8_descs != nullptr) ? 256u : 0u;
    const size_t shared_bytes = (block_size + fp8_lut_size) * sizeof(float);

    int coop_supported = 0;
    int max_blocks_per_mp = 0;
    const void* kernel_fp = reinterpret_cast<const void*>(
        &supersonic_qwen35_persistent_decode_kernel<T>);
    int api_occ = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &api_occ, kernel_fp, block_size, shared_bytes);
    hipFuncAttributes attr{};
    (void)hipFuncGetAttributes(&attr, kernel_fp);
    // gfx1100 WGP: 4 SIMDs × 1536 VGPR, wave32. Occupancy API is optimistic
    // once scratch/spills show up; size the default grid from VGPR+LDS so
    // the ticket barrier never launches more blocks than can be resident.
    int vgpr = attr.numRegs > 0 ? attr.numRegs : 256;
    int waves_simd = 1536 / vgpr;
    if (attr.localSizeBytes > 16 && waves_simd > 1) {
        waves_simd -= 1;
    }
    if (waves_simd < 1) waves_simd = 1;
    if (waves_simd > 16) waves_simd = 16;
    int from_vgpr = (waves_simd * 4) / (block_size / 32);
    if (from_vgpr < 1) from_vgpr = 1;
    int from_lds = static_cast<int>(65536 / (shared_bytes > 0 ? shared_bytes : 1));
    if (from_lds < 1) from_lds = 1;
    int safe_per_mp = from_vgpr < from_lds ? from_vgpr : from_lds;
    if (!user_grid && preset_blocks <= 0) {
        const char* arch = props.gcnArchName;
        const bool is_rdna3_wgp_arch =
            arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
            arch[3] == '1' && arch[4] == '1';
        if (is_rdna3_wgp_arch) {
            // Ticket barrier is fine at 144 on a skinny kernel, but this
            // 253-VGPR megakernel still deadlocks at 3x (only 2 blocks
            // actually stay resident). Keep the proven 2x default.
            int mult = safe_per_mp;
            if (mult > 2) mult = 2;
            if (mult < 1) mult = 1;
            num_blocks = props.multiProcessorCount * mult;
        }
    }
    {
        static bool dumped_occ = false;
        if (!dumped_occ) {
            dumped_occ = true;
            std::fprintf(
                stderr,
                "[hip-occ] persistent_decode hidden=%d B=%d grid=%d CUs=%d "
                "api_blocks/CU=%d safe_blocks/CU=%d vgpr=%d scratch=%zu "
                "static_lds=%zu dyn_lds=%zu max_threads=%d\n",
                hidden_dim,
                batch_size,
                num_blocks,
                props.multiProcessorCount,
                api_occ,
                safe_per_mp,
                attr.numRegs,
                attr.localSizeBytes,
                attr.sharedSizeBytes,
                shared_bytes,
                attr.maxThreadsPerBlock);
        }
    }
    if (coop_requested) {
        (void)hipDeviceGetAttribute(
            &coop_supported, hipDeviceAttributeCooperativeLaunch, device_ordinal);
        if (coop_supported) {
            if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_blocks_per_mp, kernel_fp, block_size, shared_bytes) !=
                hipSuccess) {
                max_blocks_per_mp = 0;
            }
            if (max_blocks_per_mp > 0) {
                int coop_max_grid = props.multiProcessorCount * max_blocks_per_mp;
                if (num_blocks > coop_max_grid) num_blocks = coop_max_grid;
            }
        }
    }

    // If the caller asked for cooperative launch but the device or runtime
    // can't actually provide it, refuse rather than fall back to the
    // non-cooperative path. A `SUPERSONIC_QWEN4B_BLOCKS=128` with `COOP=1`
    // expects the cooperative cap to keep it safe; silently running the
    // non-coop launcher with 128 blocks is exactly the grid_barrier
    // oversubscription hang the opt-in was designed to prevent.
    if (coop_requested && (!coop_supported || max_blocks_per_mp <= 0)) {
        return 261;
    }

    int io_flags = 3;
    const bool use_gqh_split = int4_scales != nullptr
        && hidden_dim >= 5120
        && tap_workspace == nullptr
        && std::getenv("SUPERSONIC_QWEN35_GQH_NOSPLIT") == nullptr
        && !coop_requested;
    hipError_t launch_err = hipSuccess;
    if (use_gqh_split) {
        const Qwen35DecodeLayerDesc* layers_base =
            static_cast<const Qwen35DecodeLayerDesc*>(layers);
        const Qwen35INT4ScaleDesc* int4_base =
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales);
        auto phase_grid = [&](int vgpr, size_t scratch, int api_occ, int cap) -> int {
            int waves_s = 1536 / (vgpr > 0 ? vgpr : 256);
            if (scratch > 16 && waves_s > 1) waves_s -= 1;
            if (waves_s < 1) waves_s = 1;
            if (waves_s > 16) waves_s = 16;
            int from_vgpr = (waves_s * 4) / (block_size / 32);
            if (from_vgpr < 1) from_vgpr = 1;
            int g = from_vgpr;
            if (api_occ > 0 && api_occ < g) g = api_occ;
            if (g > cap) g = cap;
            if (g < 1) g = 1;
            return props.multiProcessorCount * g;
        };
        auto launch_split = [&](int split, int layer, int flags, int grid, hipStream_t stream)
            -> hipError_t {
            const Qwen35DecodeLayerDesc* layer_ptr = layers_base + layer;
            const Qwen35INT4ScaleDesc* int4_ptr =
                int4_base != nullptr ? int4_base + layer : nullptr;
            if (split == 1) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 1>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    static_cast<const BatchSeqDesc*>(batch_descs),
                    int4_ptr,
                    static_cast<T*>(tap_workspace),
                    tap_layers,
                    0,
                    flags);
            } else if (split == 2) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 2>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    static_cast<const BatchSeqDesc*>(batch_descs),
                    int4_ptr,
                    static_cast<T*>(tap_workspace),
                    tap_layers,
                    0,
                    flags);
            } else if (split == 3) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 3>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    static_cast<const BatchSeqDesc*>(batch_descs),
                    int4_ptr,
                    static_cast<T*>(tap_workspace),
                    tap_layers,
                    0,
                    flags);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 4>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    static_cast<const BatchSeqDesc*>(batch_descs),
                    int4_ptr,
                    static_cast<T*>(tap_workspace),
                    tap_layers,
                    0,
                    flags);
            }
            return hipGetLastError();
        };
        int occ1 = 0, occ2 = 0, occ3 = 0, occ4 = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ1,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 1>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ2,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 2>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ3,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 3>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ4,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 4>,
            block_size,
            shared_bytes);
        hipFuncAttributes a1{}, a2{}, a3{}, a4{};
        (void)hipFuncGetAttributes(
            &a1, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 1>);
        (void)hipFuncGetAttributes(
            &a2, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 2>);
        (void)hipFuncGetAttributes(
            &a3, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 3>);
        (void)hipFuncGetAttributes(
            &a4, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 4>);
        static bool dumped_split = false;
        if (!dumped_split) {
            dumped_split = true;
            std::fprintf(
                stderr,
                "[hip-occ] gqh-split in vgpr=%d occ=%d scratch=%zu  "
                "mid vgpr=%d occ=%d scratch=%zu  out vgpr=%d occ=%d scratch=%zu  "
                "mlp vgpr=%d occ=%d scratch=%zu\n",
                a1.numRegs,
                occ1,
                a1.localSizeBytes,
                a2.numRegs,
                occ2,
                a2.localSizeBytes,
                a3.numRegs,
                occ3,
                a3.localSizeBytes,
                a4.numRegs,
                occ4,
                a4.localSizeBytes);
        }
        const int grid_in = phase_grid(a1.numRegs, a1.localSizeBytes, occ1, 4);
        const int grid_mid = phase_grid(a2.numRegs, a2.localSizeBytes, occ2, 6);
        const int grid_out = phase_grid(a3.numRegs, a3.localSizeBytes, occ3, 3);
        const int grid_mlp = phase_grid(a4.numRegs, a4.localSizeBytes, occ4, 4);
        static bool dumped_grid = false;
        if (!dumped_grid) {
            dumped_grid = true;
            std::fprintf(
                stderr,
                "[hip-occ] gqh-split grids in=%d mid=%d out=%d mlp=%d\n",
                grid_in,
                grid_mid,
                grid_out,
                grid_mlp);
        }
        const bool use_graph = std::getenv("SUPERSONIC_QWEN35_GQH_NOGRAPH") == nullptr;
        struct SplitGraphCache {
            hipGraphExec_t exec = nullptr;
            hipGraph_t graph = nullptr;
            hipStream_t stream = nullptr;
            int num_layers = -1;
            int grid_in = 0, grid_mid = 0, grid_out = 0, grid_mlp = 0;
            const void* layers = nullptr;
            void* hidden_io = nullptr;
            float* workspace = nullptr;
            unsigned int* counters = nullptr;
            unsigned int* barrier_counter = nullptr;
            unsigned int* barrier_flag = nullptr;
            const void* int4 = nullptr;
            int batch_size = 0;
        };
        static SplitGraphCache cache;
        auto record_layers = [&](hipStream_t stream) -> hipError_t {
            hipError_t err = hipSuccess;
            for (int layer = 0; layer < num_layers; ++layer) {
                int in_flags = (layer == 0) ? 1 : 0;
                int mlp_flags = (layer == num_layers - 1) ? 2 : 0;
                err = launch_split(1, layer, in_flags, grid_in, stream);
                if (err != hipSuccess) return err;
                err = launch_split(2, layer, 0, grid_mid, stream);
                if (err != hipSuccess) return err;
                err = launch_split(3, layer, 0, grid_out, stream);
                if (err != hipSuccess) return err;
                err = launch_split(4, layer, mlp_flags, grid_mlp, stream);
                if (err != hipSuccess) return err;
            }
            return hipSuccess;
        };
        const int host_seqlen = seqlen_offset;
        launch_err = hipMemcpyToSymbol(
            HIP_SYMBOL(qwen35_split_seqlen), &host_seqlen, sizeof(host_seqlen));
        if (launch_err != hipSuccess) {
            // fall through to eager launches
        } else if (use_graph) {
            const bool reuse = cache.exec != nullptr
                && cache.num_layers == num_layers
                && cache.grid_in == grid_in
                && cache.grid_mid == grid_mid
                && cache.grid_out == grid_out
                && cache.grid_mlp == grid_mlp
                && cache.layers == layers
                && cache.hidden_io == hidden_io
                && cache.workspace == workspace
                && cache.counters == counters
                && cache.barrier_counter == barrier_counter
                && cache.barrier_flag == barrier_flag
                && cache.int4 == int4_scales
                && cache.batch_size == batch_size;
            if (!reuse) {
                if (cache.exec) {
                    (void)hipGraphExecDestroy(cache.exec);
                    cache.exec = nullptr;
                }
                if (cache.graph) {
                    (void)hipGraphDestroy(cache.graph);
                    cache.graph = nullptr;
                }
                if (!cache.stream) {
                    launch_err = hipStreamCreate(&cache.stream);
                }
                if (launch_err == hipSuccess) {
                    launch_err = hipStreamBeginCapture(
                        cache.stream, hipStreamCaptureModeGlobal);
                }
                if (launch_err == hipSuccess) {
                    launch_err = record_layers(cache.stream);
                }
                if (launch_err == hipSuccess) {
                    launch_err = hipStreamEndCapture(cache.stream, &cache.graph);
                }
                if (launch_err == hipSuccess) {
                    launch_err = hipGraphInstantiate(
                        &cache.exec, cache.graph, nullptr, nullptr, 0);
                }
                if (launch_err == hipSuccess) {
                    cache.num_layers = num_layers;
                    cache.grid_in = grid_in;
                    cache.grid_mid = grid_mid;
                    cache.grid_out = grid_out;
                    cache.grid_mlp = grid_mlp;
                    cache.layers = layers;
                    cache.hidden_io = hidden_io;
                    cache.workspace = workspace;
                    cache.counters = counters;
                    cache.barrier_counter = barrier_counter;
                    cache.barrier_flag = barrier_flag;
                    cache.int4 = int4_scales;
                    cache.batch_size = batch_size;
                    static bool dumped_graph = false;
                    if (!dumped_graph) {
                        dumped_graph = true;
                        std::fprintf(stderr, "[hip-occ] gqh-split HIP graph captured\n");
                    }
                } else {
                    if (cache.exec) {
                        (void)hipGraphExecDestroy(cache.exec);
                        cache.exec = nullptr;
                    }
                    if (cache.graph) {
                        (void)hipGraphDestroy(cache.graph);
                        cache.graph = nullptr;
                    }
                }
            }
            if (launch_err == hipSuccess && cache.exec != nullptr) {
                launch_err = hipGraphLaunch(cache.exec, cache.stream);
                if (launch_err == hipSuccess) {
                    launch_err = hipStreamSynchronize(cache.stream);
                }
            } else if (launch_err != hipSuccess) {
                // Eager fallback if capture failed.
                launch_err = record_layers(0);
            }
        } else {
            launch_err = record_layers(0);
        }
    } else if (coop_requested && coop_supported && max_blocks_per_mp > 0) {
        // Args for cooperative launch: void** where each entry points to
        // local storage holding one argument value. Locals must stay alive
        // through the launch — they're destroyed at function exit, and
        // we call hipDeviceSynchronize before returning.
        const Qwen35DecodeLayerDesc* layers_typed =
            static_cast<const Qwen35DecodeLayerDesc*>(layers);
        T* hidden_io_typed = static_cast<T*>(hidden_io);
        const T* cos_typed = static_cast<const T*>(cos_table);
        const T* sin_typed = static_cast<const T*>(sin_table);
        const Qwen35FP8ScaleDesc* fp8_typed =
            static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales);
        const KVCacheFp8Desc* kv_fp8_typed =
            static_cast<const KVCacheFp8Desc*>(kv_fp8_descs);
        const BatchSeqDesc* batch_descs_typed =
            static_cast<const BatchSeqDesc*>(batch_descs);
        const Qwen35INT4ScaleDesc* int4_typed =
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales);
        T* tap_ws_typed = static_cast<T*>(tap_workspace);

        void* args[] = {
            &num_layers, &hidden_dim, &intermediate_size, &seqlen_offset,
            &layers_typed, &hidden_io_typed, &workspace, &counters,
            &barrier_counter, &barrier_flag,
            &timing_slots,
            &cos_typed, &sin_typed, &rotary_dim,
            &proj_buf_floats, &attn_scratch_floats, &enable_attention_trace,
            &fp8_typed, &kv_fp8_typed, &batch_size,
            &batch_descs_typed, &int4_typed,
            &tap_ws_typed, &tap_layers, &num_taps, &io_flags,
        };

        launch_err = hipLaunchCooperativeKernel(
            kernel_fp,
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            args,
            static_cast<uint32_t>(shared_bytes),
            0);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            shared_bytes,
            0,
            num_layers,
            hidden_dim,
            intermediate_size,
            seqlen_offset,
            static_cast<const Qwen35DecodeLayerDesc*>(layers),
            static_cast<T*>(hidden_io),
            workspace,
            counters,
            barrier_counter,
            barrier_flag,
            timing_slots,
            static_cast<const T*>(cos_table),
            static_cast<const T*>(sin_table),
            rotary_dim,
            proj_buf_floats,
            attn_scratch_floats,
            enable_attention_trace,
            static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
            static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
            batch_size,
            static_cast<const BatchSeqDesc*>(batch_descs),
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales),
            static_cast<T*>(tap_workspace),
            tap_layers,
            num_taps,
            io_flags);
        launch_err = hipGetLastError();
    }

    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

// tap_workspace/tap_layers/num_taps: DFlash hidden-state taps (M1 plumbing).
//   tap_workspace = nullptr, tap_layers = nullptr, num_taps = 0 disables; the kernel body
//   must short-circuit when tap_workspace is nullptr so existing decode callers see no
//   change in observable behavior or runtime (gfx1150 codegen-sensitivity guard).
extern "C" int supersonic_qwen35_4b_hip_persistent_decode(
    int dtype,
    size_t device_ordinal,
    size_t num_layers,
    size_t hidden_dim,
    size_t intermediate_size,
    size_t seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    unsigned long long* timing_slots,
    const void* cos_table,
    const void* sin_table,
    size_t rotary_dim,
    size_t proj_buf_floats,
    size_t attn_scratch_floats,
    int enable_attention_trace,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    size_t batch_size,
    const void* batch_descs,
    const void* int4_scales,
    void* tap_workspace,
    const int* tap_layers,
    size_t num_taps) {
    switch (dtype) {
    case 2:
        return persistent_decode_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(num_layers),
            static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size),
            static_cast<int>(seqlen_offset),
            layers, hidden_io, workspace, counters,
            barrier_counter, barrier_flag, timing_slots,
            cos_table, sin_table, static_cast<int>(rotary_dim),
            static_cast<int>(proj_buf_floats),
            static_cast<int>(attn_scratch_floats),
            enable_attention_trace,
            fp8_scales,
            kv_fp8_descs,
            static_cast<int>(batch_size),
            batch_descs,
            int4_scales,
            tap_workspace,
            tap_layers,
            static_cast<int>(num_taps));
    default:
        return 256;
    }
}

// BF16→FP8 KV cache quantization bridge
extern "C" int supersonic_qwen35_4b_hip_quantize_kv_to_fp8(
    int dtype,
    size_t device_ordinal,
    const void* src,
    void* dst_fp8,
    float* dst_scale,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int max_T,
    int pos_offset) {
    ScopedHipDevice scoped(static_cast<int>(device_ordinal));

    const int num_blocks = num_kv_heads * seq_len;
    constexpr int block_size = 256;
    const size_t shared_bytes = block_size * sizeof(float);

    switch (dtype) {
    case 2:
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(quantize_kv_to_fp8_kernel<hip_bfloat16>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            shared_bytes,
            0,
            static_cast<const hip_bfloat16*>(src),
            static_cast<uint8_t*>(dst_fp8),
            dst_scale,
            num_kv_heads, seq_len, head_dim, max_T, pos_offset);
        break;
    default:
        return 256;
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = hipDeviceSynchronize();
    if (launch_err != hipSuccess) return 254;
    if (sync_err != hipSuccess) return 255;
    return 0;
}

// supersonic_query_gpu_info is in the 0.8B bridge, not duplicated here
