#ifndef SUPERSONIC_METAL_NATIVE_FFI_H
#define SUPERSONIC_METAL_NATIVE_FFI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void supersonic_metal_profile_record(
    const char* op,
    const char* path,
    double elapsed_ms
);

void supersonic_metal_profile_record_explicit(
    int enabled,
    const char* op,
    const char* path,
    double elapsed_ms
);

int supersonic_metal_qwen36_batched_ffn_grouped_expert_direct(
    size_t n_tokens,
    size_t top_k,
    size_t hidden,
    size_t moe_intermediate,
    size_t group_size,
    const void* x_norm_ptr,
    const void* topk_idx_ptr,
    const void* topk_weight_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* expert_mid_ptr,
    void* combined_ptr,
    int wait_for_completion
);

int supersonic_metal_qwen36_batched_ffn_grouped_expert_direct_with_options(
    size_t n_tokens,
    size_t top_k,
    size_t hidden,
    size_t moe_intermediate,
    size_t group_size,
    const void* x_norm_ptr,
    const void* topk_idx_ptr,
    const void* topk_weight_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* expert_mid_ptr,
    void* combined_ptr,
    int profile_enabled,
    int phase_profile_enabled,
    int wait_for_completion
);

#ifdef __cplusplus
}
#endif

#endif
