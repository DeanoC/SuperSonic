// HIP launch wrappers for GQH decode and fused matvec.

#include "gqh.hip"

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

hipError_t maybe_sync() {
    const char* value = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    const bool enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
    return enabled ? hipDeviceSynchronize() : hipSuccess;
}

int backend_failure(int project_status, hipError_t native_status) {
    return static_cast<int>(
        0x80000000u | ((static_cast<uint32_t>(project_status) & 0x7fffu) << 16) |
        (static_cast<uint32_t>(native_status) & 0xffffu));
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

enum GqhRung : int {
    GQH_RUNG_GQH3 = 0,
    GQH_RUNG_GQH2_H = 1,
    GQH_RUNG_GQH2_C = 2,
};

int packed_row_bytes(int rung) {
    switch (rung) {
        case GQH_RUNG_GQH3:
            return GQH3_SB_BYTES;
        case GQH_RUNG_GQH2_H:
            return GQH2H_SB_BYTES;
        case GQH_RUNG_GQH2_C:
            return GQH2C_SB_BYTES;
        default:
            return -1;
    }
}

int validate_shape(int rung, int rows, int cols, int grid_code) {
    if (packed_row_bytes(rung) < 0) {
        return 401;
    }
    if (rows <= 0 || cols <= 0 || (cols % GQH_SUPERBLOCK) != 0) {
        return 402;
    }
    if (rung != GQH_RUNG_GQH2_C && (grid_code < 0 || grid_code >= GQH_GRID_CODES)) {
        return 403;
    }
    return 0;
}

gqh_grid8 load_gqh3_grid(int grid_code) {
    gqh_grid8 grid{};
    memcpy(grid.v, GQH3_GRID[grid_code], sizeof(grid.v));
    return grid;
}

gqh_grid4 load_gqh2h_grid(int grid_code) {
    gqh_grid4 grid{};
    memcpy(grid.v, GQH2H_GRID[grid_code], sizeof(grid.v));
    return grid;
}

float4 load_gqh3_mag(int grid_code) {
    float4 mag{};
    memcpy(&mag.x, &GQH3_GRID[grid_code][4], 4 * sizeof(float));
    return mag;
}

float4 load_gqh2h_mag(int grid_code) {
    float4 mag{};
    memcpy(&mag.x, &GQH2H_GRID[grid_code][2], sizeof(float));
    return mag;
}

}  // namespace

extern "C" int supersonic_gqh_hip_decode(
    int device_ordinal,
    int rung,
    const void* wire,
    float tensor_scale,
    int grid_code,
    void* dst,
    int rows,
    int cols) {
    const int shape_status = validate_shape(rung, rows, cols, grid_code);
    if (shape_status != 0) {
        return shape_status;
    }
    if (wire == nullptr || dst == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    const int64_t nsb_total = static_cast<int64_t>(rows) * (cols / GQH_SUPERBLOCK);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* out = static_cast<float*>(dst);
    switch (rung) {
        case GQH_RUNG_GQH3:
            hipLaunchKernelGGL(
                gqh3_decode_kernel,
                dim3(static_cast<unsigned int>(nsb_total)),
                dim3(GQH_SUPERBLOCK),
                0,
                0,
                packed,
                tensor_scale,
                load_gqh3_grid(grid_code),
                out);
            break;
        case GQH_RUNG_GQH2_H:
            hipLaunchKernelGGL(
                gqh2h_decode_kernel,
                dim3(static_cast<unsigned int>(nsb_total)),
                dim3(GQH_SUPERBLOCK),
                0,
                0,
                packed,
                tensor_scale,
                load_gqh2h_grid(grid_code),
                out);
            break;
        case GQH_RUNG_GQH2_C:
            hipLaunchKernelGGL(
                gqh2c_decode_kernel,
                dim3(static_cast<unsigned int>(nsb_total)),
                dim3(GQH_SUPERBLOCK),
                0,
                0,
                packed,
                out);
            break;
        default:
            return 401;
    }
    return launch_result(411, 412);
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
    const int shape_status = validate_shape(rung, out_dim, in_dim, grid_code);
    if (shape_status != 0) {
        return shape_status;
    }
    if (ncols <= 0 || wire == nullptr || x == nullptr || y == nullptr) {
        return 405;
    }
    if (x_col_stride < in_dim || y_col_stride < out_dim) {
        return 406;
    }
    // float4 activation loads fault on AMD if the base is not 16-byte aligned.
    if ((reinterpret_cast<uintptr_t>(x) % sizeof(float4)) != 0 ||
        ((x_col_stride * static_cast<int64_t>(sizeof(float))) %
         static_cast<int64_t>(sizeof(float4))) != 0) {
        return 404;
    }
    ScopedHipDevice scoped(device_ordinal);
    const dim3 blocks(
        static_cast<unsigned int>((out_dim + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    switch (rung) {
        case GQH_RUNG_GQH3:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gqh_matvec_kernel<true>),
                blocks,
                threads,
                0,
                0,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                load_gqh3_mag(grid_code),
                x_col_stride,
                y_col_stride);
            break;
        case GQH_RUNG_GQH2_H:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gqh_matvec_kernel<false>),
                blocks,
                threads,
                0,
                0,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                load_gqh2h_mag(grid_code),
                x_col_stride,
                y_col_stride);
            break;
        case GQH_RUNG_GQH2_C:
            hipLaunchKernelGGL(
                gqh2c_matvec_kernel,
                blocks,
                threads,
                0,
                0,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                x_col_stride,
                y_col_stride);
            break;
        default:
            return 401;
    }
    return launch_result(421, 422);
}

extern "C" int supersonic_gqh_hip_matvec_stream(
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
    int grid_code,
    void* stream) {
    const int shape_status = validate_shape(rung, out_dim, in_dim, grid_code);
    if (shape_status != 0) {
        return shape_status;
    }
    if (ncols <= 0 || wire == nullptr || x == nullptr || y == nullptr) {
        return 405;
    }
    if (x_col_stride < in_dim || y_col_stride < out_dim) {
        return 406;
    }
    if ((reinterpret_cast<uintptr_t>(x) % sizeof(float4)) != 0 ||
        ((x_col_stride * static_cast<int64_t>(sizeof(float))) %
         static_cast<int64_t>(sizeof(float4))) != 0) {
        return 404;
    }
    ScopedHipDevice scoped(device_ordinal);
    const dim3 blocks(
        static_cast<unsigned int>((out_dim + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    switch (rung) {
        case GQH_RUNG_GQH3:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gqh_matvec_kernel<true>),
                blocks,
                threads,
                0,
                hs,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                load_gqh3_mag(grid_code),
                x_col_stride,
                y_col_stride);
            break;
        case GQH_RUNG_GQH2_H:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gqh_matvec_kernel<false>),
                blocks,
                threads,
                0,
                hs,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                load_gqh2h_mag(grid_code),
                x_col_stride,
                y_col_stride);
            break;
        case GQH_RUNG_GQH2_C:
            hipLaunchKernelGGL(
                gqh2c_matvec_kernel,
                blocks,
                threads,
                0,
                hs,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                x_col_stride,
                y_col_stride);
            break;
        default:
            return 401;
    }
    return launch_result(421, 422);
}
