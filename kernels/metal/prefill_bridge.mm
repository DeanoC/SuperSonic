#include "metal_dispatch.hpp"

#include <cstddef>
#include <cstdint>

namespace {

int validate_device_ordinal(std::size_t device_ordinal) {
    return device_ordinal == 0 ? 0 : 1;
}

}  // namespace

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
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    if (index_dtype != 1) {
        return 123;
    }
    if (!supersonic::metal::init_prefill_library()) {
        return 2;
    }
    if (!supersonic::metal::embedding_lookup_u32(
            dtype,
            static_cast<int>(token_count),
            static_cast<int>(vocab_size),
            static_cast<int>(hidden_size),
            embeddings,
            indexes,
            out)) {
        return 3;
    }
    return 0;
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
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    if (!supersonic::metal::init_prefill_library()) {
        return 2;
    }
    if (!supersonic::metal::rms_norm(
            dtype,
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            add_unit_offset,
            xs,
            weight,
            out)) {
        return 71;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs,
    void* out) {
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    if (!supersonic::metal::init_prefill_library()) {
        return 2;
    }
    if (!supersonic::metal::matmul_rhs_transposed_tiled(
            dtype,
            static_cast<std::uint32_t>(batch_elems),
            m,
            n,
            k,
            lhs,
            rhs,
            out)) {
        return 270;
    }
    return 0;
}
