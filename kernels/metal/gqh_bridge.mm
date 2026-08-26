#include "metal_dispatch.hpp"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kGqhSuperblock = 256;
constexpr int kGqhGridCodes = 16;

int validate_device_ordinal(int device_ordinal) {
    return device_ordinal == 0 ? 0 : 1;
}

bool preflight(int device_ordinal) {
    if (validate_device_ordinal(device_ordinal) != 0) {
        return false;
    }
    return supersonic::metal::init_prefill_library();
}

int rung_to_quant_type(int rung) {
    switch (rung) {
        case 0:
            return 108;  // GQH3
        case 1:
            return 109;  // GQH2_H
        case 2:
            return 110;  // GQH2_C
        case 3:
            return 111;  // GQH4
        default:
            return -1;
    }
}

int validate_shape(int rung, int rows, int cols, int grid_code) {
    const int quant_type = rung_to_quant_type(rung);
    if (quant_type < 0) {
        return 401;
    }
    if (quant_type == 110) {
        return 401;
    }
    if (rows <= 0 || cols <= 0 || (cols % kGqhSuperblock) != 0) {
        return 402;
    }
    if (grid_code < 0 || grid_code >= kGqhGridCodes) {
        return 403;
    }
    return 0;
}

int hip_dst_to_metal_dtype(int dst_is_bf16) {
    // HIP: 0=f32, 1=bf16. Metal ScalarType kernel codes: 1=f32, 2=bf16.
    return dst_is_bf16 != 0 ? 2 : 1;
}

}  // namespace

extern "C" const void* supersonic_metal_dummy_buffer();

extern "C" int supersonic_gqh_hip_decode(
    int device_ordinal,
    int rung,
    const void* wire,
    float tensor_scale,
    int grid_code,
    void* dst,
    int rows,
    int cols,
    int dst_is_bf16,
    void* stream) {
    (void)stream;
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    const int shape_status = validate_shape(rung, rows, cols, grid_code);
    if (shape_status != 0) {
        return shape_status;
    }
    if (wire == nullptr || dst == nullptr) {
        return 405;
    }
    const int quant_type = rung_to_quant_type(rung);
    if (!preflight(device_ordinal)) {
        return 2;
    }
    if (!supersonic::metal::gqh_decode(
            quant_type,
            rows,
            cols,
            tensor_scale,
            grid_code,
            hip_dst_to_metal_dtype(dst_is_bf16),
            wire,
            dst)) {
        return 297;
    }
    return 0;
}

extern "C" int supersonic_gqh_hip_dequant_gemm_bf16(
    int device_ordinal,
    int rung,
    const void* wire,
    float tensor_scale,
    int grid_code,
    const void* lhs,
    void* out,
    int m,
    int n,
    int k) {
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    if (wire == nullptr || lhs == nullptr || out == nullptr || m <= 0 || n <= 0 || k <= 0) {
        return 1;
    }
    if ((k % kGqhSuperblock) != 0) {
        return 402;
    }
    const int quant_type = rung_to_quant_type(rung);
    if (quant_type < 0 || quant_type == 110) {
        return 401;
    }
    if (grid_code < 0 || grid_code >= kGqhGridCodes) {
        return 403;
    }
    if (!preflight(device_ordinal)) {
        return 2;
    }
    const void* dummy = supersonic_metal_dummy_buffer();
    if (dummy == nullptr) {
        return 295;
    }
    constexpr int kBf16 = 2;
    if (!supersonic::metal::matmul_int4_dequant(
            kBf16,
            1,
            m,
            n,
            k,
            lhs,
            wire,
            dummy,
            dummy,
            nullptr,
            1,
            quant_type,
            tensor_scale,
            grid_code,
            out)) {
        return 298;
    }
    return 0;
}

extern "C" int supersonic_gqh_hip_matvec(
    int device_ordinal,
    int rung,
    const void* wire,
    const void* x,
    void* y,
    int in_dim,
    int out_dim,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride,
    float tensor_scale,
    int grid_code) {
    if (validate_device_ordinal(device_ordinal) != 0) {
        return 1;
    }
    if (ncols <= 0 || wire == nullptr || x == nullptr || y == nullptr) {
        return 405;
    }
    if (in_dim <= 0 || out_dim <= 0) {
        return 405;
    }
    if (x_col_stride < in_dim || y_col_stride < out_dim) {
        return 406;
    }
    if (x_col_stride != in_dim || y_col_stride != out_dim) {
        return 406;
    }
    const int quant_type = rung_to_quant_type(rung);
    if (quant_type < 0) {
        return 401;
    }
    if (quant_type == 110) {
        return 401;
    }
    if (!preflight(device_ordinal)) {
        return 2;
    }
    const void* dummy = supersonic_metal_dummy_buffer();
    if (dummy == nullptr) {
        return 295;
    }
    constexpr int kF32 = 1;
    if (!supersonic::metal::matmul_int4_dequant(
            kF32,
            1,
            ncols,
            out_dim,
            in_dim,
            x,
            wire,
            dummy,
            dummy,
            nullptr,
            1,
            quant_type,
            tensor_scale,
            grid_code,
            y)) {
        return 296;
    }
    return 0;
}
