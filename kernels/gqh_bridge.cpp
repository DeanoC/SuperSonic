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

size_t gqh_x_lds_bytes(int in_dim) {
    if (in_dim <= 0 || in_dim > GQH_LDS_X_MAX) {
        return 0;
    }
    return static_cast<size_t>(in_dim) * sizeof(float);
}

// Fat-K down is nsb=68. Hidden-width singles stay on the original
// loop so their VGPR/occupancy codegen does not change.
template <bool kAcc>
void launch_gqh12_matvec(
    int rung,
    dim3 blocks,
    dim3 threads,
    size_t lds,
    hipStream_t stream,
    const uint8_t* packed,
    const float* xv,
    float* yv,
    int in_dim,
    int out_dim,
    float tensor_scale,
    int grid_code,
    int64_t x_col_stride,
    int64_t y_col_stride) {
    const bool fat = (in_dim / GQH_SUPERBLOCK) >= 64;
    if (rung == GQH_RUNG_GQH3) {
        const float4 mag = load_gqh3_mag(grid_code);
        if (fat) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh_matvec_kernel<true, kAcc, 1>)),
                blocks,
                threads,
                lds,
                stream,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                mag,
                x_col_stride,
                y_col_stride);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh_matvec_kernel<true, kAcc, 0>)),
                blocks,
                threads,
                lds,
                stream,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                mag,
                x_col_stride,
                y_col_stride);
        }
        return;
    }
    const float4 mag = load_gqh2h_mag(grid_code);
    if (fat) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_kernel<false, kAcc, 1>)),
            blocks,
            threads,
            lds,
            stream,
            packed,
            xv,
            yv,
            in_dim,
            out_dim,
            tensor_scale,
            mag,
            x_col_stride,
            y_col_stride);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_kernel<false, kAcc, 0>)),
            blocks,
            threads,
            lds,
            stream,
            packed,
            xv,
            yv,
            in_dim,
            out_dim,
            tensor_scale,
            mag,
            x_col_stride,
            y_col_stride);
    }
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
    const int nsb = cols / GQH_SUPERBLOCK;
    const int64_t nsb_total = static_cast<int64_t>(rows) * nsb;
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
                out,
                nsb);
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
                out,
                nsb);
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
    const size_t lds = gqh_x_lds_bytes(in_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
            launch_gqh12_matvec<false>(
                rung,
                blocks,
                threads,
                lds,
                0,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                grid_code,
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
    const size_t lds = gqh_x_lds_bytes(in_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
            launch_gqh12_matvec<false>(
                rung,
                blocks,
                threads,
                lds,
                hs,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                grid_code,
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

extern "C" int supersonic_gqh_hip_matvec_stream_acc(
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
    const size_t lds = gqh_x_lds_bytes(in_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
            launch_gqh12_matvec<true>(
                rung,
                blocks,
                threads,
                lds,
                hs,
                packed,
                xv,
                yv,
                in_dim,
                out_dim,
                tensor_scale,
                grid_code,
                x_col_stride,
                y_col_stride);
            break;
        default:
            return 401;
    }
    return launch_result(421, 422);
}

extern "C" int supersonic_gqh_hip_matvec_stream_pair(
    int device_ordinal,
    int rung_a,
    int rung_b,
    const void* wire_a,
    const void* wire_b,
    const void* x,
    void* y_a,
    void* y_b,
    int in_dim,
    int out_dim,
    float scale_a,
    float scale_b,
    int grid_a,
    int grid_b,
    void* stream) {
    if (rung_a != GQH_RUNG_GQH3 && rung_a != GQH_RUNG_GQH2_H) {
        return 401;
    }
    if (rung_b != GQH_RUNG_GQH3 && rung_b != GQH_RUNG_GQH2_H) {
        return 401;
    }
    const int shape_a = validate_shape(rung_a, out_dim, in_dim, grid_a);
    const int shape_b = validate_shape(rung_b, out_dim, in_dim, grid_b);
    if (shape_a != 0) {
        return shape_a;
    }
    if (shape_b != 0) {
        return shape_b;
    }
    if (wire_a == nullptr || wire_b == nullptr || x == nullptr || y_a == nullptr ||
        y_b == nullptr) {
        return 405;
    }
    if ((reinterpret_cast<uintptr_t>(x) % sizeof(float4)) != 0) {
        return 404;
    }
    ScopedHipDevice scoped(device_ordinal);
    const dim3 blocks(
        static_cast<unsigned int>((out_dim + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS),
        1,
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const size_t lds = gqh_x_lds_bytes(in_dim);
    auto* pa = static_cast<const uint8_t*>(wire_a);
    auto* pb = static_cast<const uint8_t*>(wire_b);
    auto* xv = static_cast<const float*>(x);
    auto* ya = static_cast<float*>(y_a);
    auto* yb = static_cast<float*>(y_b);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    const float4 mag_a = (rung_a == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_a)
                                                   : load_gqh2h_mag(grid_a);
    const float4 mag_b = (rung_b == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_b)
                                                   : load_gqh2h_mag(grid_b);
    if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_pair_kernel<true, true>)),
            blocks,
            threads,
            lds,
            hs,
            pa,
            pb,
            xv,
            ya,
            yb,
            in_dim,
            out_dim,
            scale_a,
            scale_b,
            mag_a,
            mag_b);
    } else if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH2_H) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_pair_kernel<true, false>)),
            blocks,
            threads,
            lds,
            hs,
            pa,
            pb,
            xv,
            ya,
            yb,
            in_dim,
            out_dim,
            scale_a,
            scale_b,
            mag_a,
            mag_b);
    } else if (rung_a == GQH_RUNG_GQH2_H && rung_b == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_pair_kernel<false, true>)),
            blocks,
            threads,
            lds,
            hs,
            pa,
            pb,
            xv,
            ya,
            yb,
            in_dim,
            out_dim,
            scale_a,
            scale_b,
            mag_a,
            mag_b);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_pair_kernel<false, false>)),
            blocks,
            threads,
            lds,
            hs,
            pa,
            pb,
            xv,
            ya,
            yb,
            in_dim,
            out_dim,
            scale_a,
            scale_b,
            mag_a,
            mag_b);
    }
    return launch_result(421, 422);
}

extern "C" int supersonic_gqh_hip_mix_matvec_stream(
    int device_ordinal,
    int qtype,
    const void* wire,
    const void* x,
    void* y,
    int in_dim,
    int out_dim,
    int ncols,
    int acc,
    int mode,
    const float* lut,
    void* stream) {
    if (wire == nullptr || x == nullptr || y == nullptr || lut == nullptr) {
        return 405;
    }
    if (in_dim <= 0 || in_dim % 32 != 0 || out_dim <= 0 || ncols <= 0) {
        return 406;
    }
    if (qtype != 105 && qtype != 106) {
        return 401;
    }
    ScopedHipDevice scoped(device_ordinal);
    const dim3 blocks(
        static_cast<unsigned int>((out_dim + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    float4 lut0{}, lut1{}, lut2{}, lut3{};
    memcpy(&lut0, lut + 0, sizeof(float4));
    memcpy(&lut1, lut + 4, sizeof(float4));
    memcpy(&lut2, lut + 8, sizeof(float4));
    memcpy(&lut3, lut + 12, sizeof(float4));
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    const bool is_fp3 = qtype == 105;
    if (is_fp3 && acc) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_mix_matvec_kernel<true, true>)),
            blocks, threads, 0, hs, packed, xv, yv, in_dim, out_dim, mode,
            lut0, lut1, lut2, lut3, (int64_t)in_dim, (int64_t)out_dim);
    } else if (is_fp3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_mix_matvec_kernel<true, false>)),
            blocks, threads, 0, hs, packed, xv, yv, in_dim, out_dim, mode,
            lut0, lut1, lut2, lut3, (int64_t)in_dim, (int64_t)out_dim);
    } else if (acc) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_mix_matvec_kernel<false, true>)),
            blocks, threads, 0, hs, packed, xv, yv, in_dim, out_dim, mode,
            lut0, lut1, lut2, lut3, (int64_t)in_dim, (int64_t)out_dim);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_mix_matvec_kernel<false, false>)),
            blocks, threads, 0, hs, packed, xv, yv, in_dim, out_dim, mode,
            lut0, lut1, lut2, lut3, (int64_t)in_dim, (int64_t)out_dim);
    }
    return launch_result(421, 422);
}
