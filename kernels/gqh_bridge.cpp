// HIP launch wrappers for GQH decode and fused matvec.

#include "gqh.hip"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
bool g_gqh_allow_tight = false;
std::unordered_set<const void*> g_gqh_tight;
std::unordered_set<const void*> g_gqh_ileave;
std::unordered_map<const void*, uint8_t*> g_gqh_padded;
std::vector<uint8_t*> g_gqh_padded_allocs;
int g_gqh_row_off = 0;
}  // namespace

extern "C" void supersonic_gqh_hip_set_row_off(int off) {
    g_gqh_row_off = off < 0 ? 0 : off;
}

extern "C" void supersonic_gqh_hip_enable_tight_decode() {
    g_gqh_allow_tight = true;
}

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
    GQH_RUNG_GQH4 = 3,
};

int gqh4_tight_warps() {
    return 4;
}

int packed_row_bytes(int rung) {
    switch (rung) {
        case GQH_RUNG_GQH3:
            return GQH3_SB_BYTES;
        case GQH_RUNG_GQH2_H:
            return GQH2H_SB_BYTES;
        case GQH_RUNG_GQH2_C:
            return GQH2C_SB_BYTES;
        case GQH_RUNG_GQH4:
            return GQH4_SB_BYTES;
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

gqh_grid16 load_gqh4_grid(int grid_code) {
    gqh_grid16 grid{};
    memcpy(grid.v, GQH4_GRID[grid_code], sizeof(grid.v));
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

int gqh_row_blocks(int in_dim, int out_dim) {
    (void)in_dim;
    const int full = (out_dim + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS;
    return full < 1 ? 1 : full;
}

size_t gqh_x_lds_bytes(int in_dim, int out_dim) {
    if (in_dim <= 0 || in_dim > GQH_LDS_X_MAX || out_dim > GQH_LDS_OUT_MAX) {
        return 0;
    }
    return static_cast<size_t>(in_dim) * sizeof(float);
}

bool gqh_is_tight(const void* wire) {
    return wire != nullptr && g_gqh_tight.find(wire) != g_gqh_tight.end();
}

bool gqh_is_ileave(const void* wire) {
    return wire != nullptr && g_gqh_ileave.find(wire) != g_gqh_ileave.end();
}

const uint8_t* gqh_decode_wire(const uint8_t* packed) {
    auto it = g_gqh_padded.find(packed);
    return it != g_gqh_padded.end() ? it->second : packed;
}

bool gqh_is_padded(const void* wire) {
    return wire != nullptr && g_gqh_padded.find(wire) != g_gqh_padded.end();
}

bool gqh_want_ileave(int out_dim, int stride) {
    return out_dim >= 4096 && (out_dim % 2) == 0 && stride > 0 &&
        (2 * static_cast<size_t>(stride)) <= static_cast<size_t>(64 * 1024);
}

void gqh_ensure_tight(
    uint8_t* wire, int in_dim, int out_dim, int rung, hipStream_t stream) {
    if (!g_gqh_allow_tight || wire == nullptr) {
        return;
    }
    if (rung != GQH_RUNG_GQH3 && rung != GQH_RUNG_GQH2_H) {
        return;
    }
    if (in_dim <= 0 || out_dim <= 0 || (in_dim % GQH_SUPERBLOCK) != 0) {
        return;
    }
    if (gqh_is_tight(wire)) {
        return;
    }
    const int nsb = in_dim / GQH_SUPERBLOCK;
    const int is3 = rung == GQH_RUNG_GQH3 ? 1 : 0;
    const int stride = gqh_plane_row_bytes(nsb, is3);
    if (stride <= 0 || stride > 48 * 1024) {
        return;
    }
    const dim3 threads(256, 1, 1);
    const dim3 blocks(static_cast<unsigned int>(out_dim));
    if (rung == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_planar_to_tight_kernel<true>)),
            blocks,
            threads,
            static_cast<size_t>(stride),
            stream,
            wire,
            nsb,
            out_dim);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_planar_to_tight_kernel<false>)),
            blocks,
            threads,
            static_cast<size_t>(stride),
            stream,
            wire,
            nsb,
            out_dim);
    }
    g_gqh_tight.insert(wire);
}

void gqh_ensure_padded(
    uint8_t* wire, int in_dim, int out_dim, int rung, hipStream_t stream) {
    if (!g_gqh_allow_tight || wire == nullptr) {
        return;
    }
    if (rung != GQH_RUNG_GQH3 && rung != GQH_RUNG_GQH2_H) {
        return;
    }
    if (in_dim <= 0 || out_dim <= 0 || (in_dim % GQH_SUPERBLOCK) != 0) {
        return;
    }
    if (g_gqh_padded.find(wire) != g_gqh_padded.end()) {
        return;
    }
    const int nsb = in_dim / GQH_SUPERBLOCK;
    const int is3 = rung == GQH_RUNG_GQH3 ? 1 : 0;
    const int stride = gqh_plane_row_bytes(nsb, is3);
    if (stride <= 0 || stride > 48 * 1024) {
        return;
    }
    const int pad_sb = is3 ? GQH3_PAD_SB_BYTES : GQH2H_PAD_SB_BYTES;
    const size_t bytes =
        static_cast<size_t>(out_dim) * static_cast<size_t>(nsb) *
        static_cast<size_t>(pad_sb);
    uint8_t* dst = nullptr;
    if (bytes == 0 || hipMalloc(&dst, bytes) != hipSuccess || dst == nullptr) {
        gqh_ensure_tight(wire, in_dim, out_dim, rung, stream);
        return;
    }
    g_gqh_padded_allocs.push_back(dst);
    const dim3 threads(256, 1, 1);
    const dim3 blocks(static_cast<unsigned int>(out_dim));
    if (rung == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_planar_to_padded_kernel<true>)),
            blocks,
            threads,
            static_cast<size_t>(stride),
            stream,
            wire,
            dst,
            nsb,
            out_dim);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_planar_to_padded_kernel<false>)),
            blocks,
            threads,
            static_cast<size_t>(stride),
            stream,
            wire,
            dst,
            nsb,
            out_dim);
    }
    g_gqh_padded[wire] = dst;
    g_gqh_tight.insert(wire);
    static bool dumped_pad = false;
    if (!dumped_pad) {
        dumped_pad = true;
        std::fprintf(
            stderr,
            "[gqh-gemv] padded AoS MLP layout pad_sb=%d lo@%d hi@%d\n",
            pad_sb,
            GQH_PAD_LO,
            GQH_PAD_HI);
    }
}

// Fat-K down is nsb=68. Hidden-width singles stay on the original
// loop so their VGPR/occupancy codegen does not change.
template <bool kAcc, int kRows, int kNsb>
void launch_gqh4_tight(
    dim3 blocks,
    dim3 threads,
    hipStream_t stream,
    const uint8_t* packed,
    const float* xv,
    float* yv,
    int in_dim,
    int out_dim,
    float tensor_scale,
    gqh_grid16 grid4,
    int64_t x_col_stride,
    int64_t y_col_stride,
    int row_off) {
    const int ncols = (int)blocks.y;
    dim3 g = blocks;
    if (ncols > 1 && ncols <= 8) {
        static bool dumped_skinny = false;
        if (!dumped_skinny) {
            dumped_skinny = true;
            std::fprintf(
                stderr,
                "[gqh-gemv] skinny M=%d kCols=%d in=%d out=%d acc=%d rows=%d\n",
                ncols,
                ncols <= 3 ? 3 : (ncols <= 4 ? 4 : 8),
                in_dim,
                out_dim,
                kAcc ? 1 : 0,
                kRows);
        }
        g.y = 1;
        if (ncols <= 3) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_tight_kernel<kAcc, kRows, kNsb, 3>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, row_off, ncols);
        } else if (ncols <= 4) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_tight_kernel<kAcc, kRows, kNsb, 4>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, row_off, ncols);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_tight_kernel<kAcc, kRows, kNsb, 8>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, row_off, ncols);
        }
        return;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME((gqh4_matvec_tight_kernel<kAcc, kRows, kNsb, 1>)),
        g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
        tensor_scale, grid4, x_col_stride, y_col_stride, row_off, 1);
}

template <bool IS_GQH3, bool kAcc, int kRows, bool kPad, int kNsb>
void launch_gqh12_tight(
    dim3 blocks,
    dim3 threads,
    hipStream_t stream,
    const uint8_t* packed,
    const float* xv,
    float* yv,
    int in_dim,
    int out_dim,
    float tensor_scale,
    float4 mag,
    int64_t x_col_stride,
    int64_t y_col_stride,
    int ileave,
    int row_off) {
    const int ncols = (int)blocks.y;
    dim3 g = blocks;
    if (ncols > 1 && ncols <= 8) {
        static bool dumped_skinny12 = false;
        if (!dumped_skinny12) {
            dumped_skinny12 = true;
            const int kc = ncols <= 3 ? 3 : (ncols <= 4 ? 4 : 8);
            hipFuncAttributes attr{};
            if (kc == 3) {
                (void)hipFuncGetAttributes(
                    &attr,
                    reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                        (gqh_matvec_tight_kernel<
                            IS_GQH3, kAcc, kRows, kPad, kNsb, 3>))));
            }
            std::fprintf(
                stderr,
                "[gqh-gemv] skinny12 M=%d kCols=%d gqh3=%d in=%d out=%d acc=%d "
                "vgpr=%d scratch=%zu warps=%d\n",
                ncols,
                kc,
                IS_GQH3 ? 1 : 0,
                in_dim,
                out_dim,
                kAcc ? 1 : 0,
                attr.numRegs,
                attr.localSizeBytes,
                (int)(threads.x / 32));
        }
        g.y = 1;
        if (ncols <= 3) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    (gqh_matvec_tight_kernel<IS_GQH3, kAcc, kRows, kPad, kNsb, 3>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, mag, x_col_stride, y_col_stride, ileave, row_off,
                ncols);
        } else if (ncols <= 4) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    (gqh_matvec_tight_kernel<IS_GQH3, kAcc, kRows, kPad, kNsb, 4>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, mag, x_col_stride, y_col_stride, ileave, row_off,
                ncols);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    (gqh_matvec_tight_kernel<IS_GQH3, kAcc, kRows, kPad, kNsb, 8>)),
                g, threads, 0, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, mag, x_col_stride, y_col_stride, ileave, row_off,
                ncols);
        }
        return;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            (gqh_matvec_tight_kernel<IS_GQH3, kAcc, kRows, kPad, kNsb, 1>)),
        g, threads, 0, stream, packed, xv, yv, in_dim, out_dim, tensor_scale,
        mag, x_col_stride, y_col_stride, ileave, row_off, 1);
}

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
    if (rung == GQH_RUNG_GQH4) {
        const int nsb = in_dim / GQH_SUPERBLOCK;
        // gfx12: 8 waves/block. 4-wave GQH4 pair+down left decode at 61 ms/tok.
        const int kTightWarps = gqh4_tight_warps();
        const bool dual =
            out_dim > 5120 && (out_dim % (kTightWarps * 2)) == 0;
        dim3 tblocks = blocks;
        tblocks.x = static_cast<unsigned int>(
            (out_dim + (dual ? kTightWarps * 2 : kTightWarps) - 1) /
            (dual ? kTightWarps * 2 : kTightWarps));
        if (tblocks.x == 0) {
            tblocks.x = 1;
        }
        const dim3 tthreads(GQH_WARP * kTightWarps, 1, 1);
        const gqh_grid16 grid4 = load_gqh4_grid(grid_code);
        if (nsb == 68) {
            // Fat-K down: 17408/256. Dual-row at out=5120 shares x.
            // LDS tiling of x was 66 ms/tok vs 61 eager.
            const bool down_dual = (out_dim % (kTightWarps * 2)) == 0;
            dim3 dblocks = tblocks;
            dblocks.x = static_cast<unsigned int>(
                (out_dim + (down_dual ? kTightWarps * 2 : kTightWarps) - 1) /
                (down_dual ? kTightWarps * 2 : kTightWarps));
            if (dblocks.x == 0) {
                dblocks.x = 1;
            }
            static bool dumped_down = false;
            if (!dumped_down) {
                dumped_down = true;
                std::fprintf(
                    stderr,
                    "[gqh-gemv] gqh4 down nsb=68 warps=%d dual=%d in=%d out=%d\n",
                    kTightWarps,
                    down_dual ? 1 : 0,
                    in_dim,
                    out_dim);
            }
            if (down_dual) {
                launch_gqh4_tight<kAcc, 2, 68>(
                    dblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
            } else {
                launch_gqh4_tight<kAcc, 1, 68>(
                    dblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
            }
        } else if (dual && nsb == 20) {
            launch_gqh4_tight<kAcc, 2, 20>(
                tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
        } else if (dual) {
            launch_gqh4_tight<kAcc, 2, 0>(
                tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
        } else if (nsb == 20) {
            launch_gqh4_tight<kAcc, 1, 20>(
                tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
        } else {
            launch_gqh4_tight<kAcc, 1, 0>(
                tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                tensor_scale, grid4, x_col_stride, y_col_stride, g_gqh_row_off);
        }
        return;
    }
    gqh_ensure_tight(const_cast<uint8_t*>(packed), in_dim, out_dim, rung, stream);
    if (gqh_is_tight(packed)) {
        const bool pad = gqh_is_padded(packed);
        packed = gqh_decode_wire(packed);
        const int nsb = in_dim / GQH_SUPERBLOCK;
        const int ncols_in = (int)blocks.y;
        // Fat-K / residual / lm_head / inproj: 4-wave. Skinny B>1 down
        // (nsb=68, out=5120) uses 8-wave blocks so fewer CTAs share x.
        const int kTightWarps =
            (ncols_in > 1 && nsb == 68 && out_dim == 5120) ? 8 : 4;
        const int ileave = gqh_is_ileave(packed) ? 1 : 0;
        const bool dual =
            out_dim > 5120 && (out_dim % (kTightWarps * 2)) == 0;
        static bool dumped_tight = false;
        if (!dumped_tight) {
            dumped_tight = true;
            std::fprintf(
                stderr,
                "[gqh-gemv] tight singles warps=%d dual=%d in=%d out=%d\n",
                kTightWarps,
                dual ? 1 : 0,
                in_dim,
                out_dim);
        }
        dim3 tblocks = blocks;
        tblocks.x = static_cast<unsigned int>(
            (out_dim + (dual ? kTightWarps * 2 : kTightWarps) - 1) /
            (dual ? kTightWarps * 2 : kTightWarps));
        if (tblocks.x == 0) {
            tblocks.x = 1;
        }
        const dim3 tthreads(GQH_WARP * kTightWarps, 1, 1);
        auto launch12 = [&](auto /*tag*/, bool is3, int rows, bool is_pad, int nsb_t,
                            float4 mag) {
            if (is3 && rows == 2 && is_pad) {
                launch_gqh12_tight<true, kAcc, 2, true, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && rows == 1 && is_pad) {
                launch_gqh12_tight<true, kAcc, 1, true, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (!is3 && rows == 2 && is_pad) {
                launch_gqh12_tight<false, kAcc, 2, true, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (!is3 && rows == 1 && is_pad) {
                launch_gqh12_tight<false, kAcc, 1, true, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && rows == 2 && nsb_t == 68) {
                launch_gqh12_tight<true, kAcc, 2, false, 68>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && rows == 2 && nsb_t == 20) {
                launch_gqh12_tight<true, kAcc, 2, false, 20>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && rows == 2) {
                launch_gqh12_tight<true, kAcc, 2, false, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && nsb_t == 68) {
                launch_gqh12_tight<true, kAcc, 1, false, 68>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3) {
                launch_gqh12_tight<true, kAcc, 1, false, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (rows == 2 && nsb_t == 68) {
                launch_gqh12_tight<false, kAcc, 2, false, 68>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (rows == 2 && nsb_t == 20) {
                launch_gqh12_tight<false, kAcc, 2, false, 20>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (rows == 2) {
                launch_gqh12_tight<false, kAcc, 2, false, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (nsb_t == 68) {
                launch_gqh12_tight<false, kAcc, 1, false, 68>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else {
                launch_gqh12_tight<false, kAcc, 1, false, 0>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            }
        };
        if (pad) {
            if (rung == GQH_RUNG_GQH3) {
                launch12(0, true, dual ? 2 : 1, true, 0, load_gqh3_mag(grid_code));
            } else {
                launch12(0, false, dual ? 2 : 1, true, 0, load_gqh2h_mag(grid_code));
            }
            return;
        }
        if (rung == GQH_RUNG_GQH3) {
            launch12(
                0, true, dual ? 2 : 1, false, nsb, load_gqh3_mag(grid_code));
        } else {
            launch12(
                0, false, dual ? 2 : 1, false, nsb, load_gqh2h_mag(grid_code));
        }
        return;
    }
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
    int cols,
    int dst_is_bf16,
    void* stream) {
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
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(nsb_total));
    const dim3 block(GQH_SUPERBLOCK);
    if (dst_is_bf16) {
        auto* out = static_cast<hip_bfloat16*>(dst);
        switch (rung) {
            case GQH_RUNG_GQH3:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh3_decode_kernel<hip_bfloat16>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh3_grid(grid_code),
                    out,
                    nsb);
                break;
            case GQH_RUNG_GQH2_H:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh2h_decode_kernel<hip_bfloat16>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh2h_grid(grid_code),
                    out,
                    nsb);
                break;
            case GQH_RUNG_GQH2_C:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh2c_decode_kernel<hip_bfloat16>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    out);
                break;
            case GQH_RUNG_GQH4:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh4_decode_kernel<hip_bfloat16>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh4_grid(grid_code),
                    out);
                break;
            default:
                return 401;
        }
    } else {
        auto* out = static_cast<float*>(dst);
        switch (rung) {
            case GQH_RUNG_GQH3:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh3_decode_kernel<float>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh3_grid(grid_code),
                    out,
                    nsb);
                break;
            case GQH_RUNG_GQH2_H:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh2h_decode_kernel<float>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh2h_grid(grid_code),
                    out,
                    nsb);
                break;
            case GQH_RUNG_GQH2_C:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh2c_decode_kernel<float>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    out);
                break;
            case GQH_RUNG_GQH4:
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gqh4_decode_kernel<float>),
                    grid,
                    block,
                    0,
                    st,
                    packed,
                    tensor_scale,
                    load_gqh4_grid(grid_code),
                    out);
                break;
            default:
                return 401;
        }
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
        static_cast<unsigned int>(gqh_row_blocks(in_dim, out_dim)),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const size_t lds = gqh_x_lds_bytes(in_dim, out_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
        case GQH_RUNG_GQH4:
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
        static_cast<unsigned int>(gqh_row_blocks(in_dim, out_dim)),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const size_t lds = gqh_x_lds_bytes(in_dim, out_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
        case GQH_RUNG_GQH4:
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
        static_cast<unsigned int>(gqh_row_blocks(in_dim, out_dim)),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const size_t lds = gqh_x_lds_bytes(in_dim, out_dim);
    auto* packed = static_cast<const uint8_t*>(wire);
    auto* xv = static_cast<const float*>(x);
    auto* yv = static_cast<float*>(y);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    switch (rung) {
        case GQH_RUNG_GQH3:
        case GQH_RUNG_GQH2_H:
        case GQH_RUNG_GQH4:
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
    int out_b,
    float scale_a,
    float scale_b,
    int grid_a,
    int grid_b,
    int fuse_swiglu,
    void* stream,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride) {
    const bool a_ok = rung_a == GQH_RUNG_GQH3 || rung_a == GQH_RUNG_GQH2_H ||
        rung_a == GQH_RUNG_GQH4;
    const bool b_ok = rung_b == GQH_RUNG_GQH3 || rung_b == GQH_RUNG_GQH2_H ||
        rung_b == GQH_RUNG_GQH4;
    if (!a_ok || !b_ok) {
        return 401;
    }
    const int out_b_eff = out_b > 0 ? out_b : out_dim;
    const int row_max = out_dim > out_b_eff ? out_dim : out_b_eff;
    const int shape_a = validate_shape(rung_a, out_dim, in_dim, grid_a);
    const int shape_b = validate_shape(rung_b, out_b_eff, in_dim, grid_b);
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
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    ScopedHipDevice scoped(device_ordinal);
    const dim3 blocks(
        static_cast<unsigned int>(gqh_row_blocks(in_dim, row_max)),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const size_t lds = gqh_x_lds_bytes(in_dim, out_dim);
    auto* pa = static_cast<const uint8_t*>(wire_a);
    auto* pb = static_cast<const uint8_t*>(wire_b);
    auto* xv = static_cast<const float*>(x);
    auto* ya = static_cast<float*>(y_a);
    auto* yb = static_cast<float*>(y_b);
    hipStream_t hs = static_cast<hipStream_t>(stream);
    if (rung_a == GQH_RUNG_GQH4 || rung_b == GQH_RUNG_GQH4) {
        if (rung_a != GQH_RUNG_GQH4 || rung_b != GQH_RUNG_GQH4 || !fuse_swiglu ||
            out_dim != out_b_eff) {
            return 401;
        }
        // 4-wave 1-row + nsb=20 unroll-4 was the best pair (3421 ms).
        // 2-wave dual-row was 3429; 4-wave dual-row was 3429.
        const int kSwigluWarps = 4;
        dim3 sblocks(
            static_cast<unsigned int>(
                (out_dim + kSwigluWarps - 1) / kSwigluWarps),
            static_cast<unsigned int>(ncols),
            1);
        if (sblocks.x == 0) {
            sblocks.x = 1;
        }
        const dim3 sthreads(GQH_WARP * kSwigluWarps, 1, 1);
        const gqh_grid16 ga = load_gqh4_grid(grid_a);
        const gqh_grid16 gb = load_gqh4_grid(grid_b);
        dim3 g = sblocks;
        const bool skinny = ncols > 1 && ncols <= 8;
        if (skinny) {
            static bool dumped_pair_skinny = false;
            if (!dumped_pair_skinny) {
                dumped_pair_skinny = true;
                std::fprintf(
                    stderr,
                    "[gqh-gemv] pair skinny M=%d kCols=%d in=%d out=%d\n",
                    ncols,
                    ncols <= 3 ? 3 : (ncols <= 4 ? 4 : 8),
                    in_dim,
                    out_dim);
            }
            g.y = 1;
        }
        if (skinny && in_dim == 5120 && ncols <= 3) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<20, 1, 3>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (skinny && in_dim == 5120 && ncols <= 4) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<20, 1, 4>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (skinny && in_dim == 5120) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<20, 1, 8>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (skinny && ncols <= 3) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<0, 1, 3>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (skinny && ncols <= 4) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<0, 1, 4>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (skinny) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<0, 1, 8>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else if (in_dim == 5120) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<20, 1, 1>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gqh4_matvec_pair_swiglu_kernel<0, 1, 1>)),
                g, sthreads, 0, hs, pa, pb, xv, ya, in_dim, out_dim,
                scale_a, scale_b, ga, gb, x_col_stride, y_col_stride, ncols);
        }
        return launch_result(421, 422);
    }
    gqh_ensure_tight(const_cast<uint8_t*>(pa), in_dim, out_dim, rung_a, hs);
    gqh_ensure_tight(const_cast<uint8_t*>(pb), in_dim, out_b_eff, rung_b, hs);
    const bool tight = gqh_is_tight(pa) && gqh_is_tight(pb);
    const bool pad = gqh_is_padded(pa) && gqh_is_padded(pb);
    if (tight) {
        pa = gqh_decode_wire(pa);
        pb = gqh_decode_wire(pb);
    }
    const float4 mag_a = (rung_a == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_a)
                                                   : load_gqh2h_mag(grid_a);
    const float4 mag_b = (rung_b == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_b)
                                                   : load_gqh2h_mag(grid_b);
    if (tight) {
        // llama.cpp: 4 waves/block, simple SB loop, no LDS.
        constexpr int kTightWarps = 4;
        const int ileave =
            (gqh_is_ileave(pa) && gqh_is_ileave(pb)) ? 1 : 0;
        const bool dual = false;
        dim3 tblocks = blocks;
        tblocks.x = static_cast<unsigned int>(
            (row_max + (dual ? kTightWarps * 2 : kTightWarps) - 1) /
            (dual ? kTightWarps * 2 : kTightWarps));
        if (tblocks.x == 0) {
            tblocks.x = 1;
        }
        const dim3 tthreads(GQH_WARP * kTightWarps, 1, 1);
        const size_t tlds = 0;
        if (fuse_swiglu && !dual && ileave == 0 && out_dim == out_b_eff) {
            constexpr int kSwigluWarps = 4;
            const bool pair_skinny = ncols > 1 && ncols <= 4;
            dim3 sblocks(
                static_cast<unsigned int>(
                    (out_dim + kSwigluWarps - 1) / kSwigluWarps),
                pair_skinny ? 1u : static_cast<unsigned int>(ncols),
                1);
            if (sblocks.x == 0) {
                sblocks.x = 1;
            }
            const dim3 sthreads(GQH_WARP * kSwigluWarps, 1, 1);
            static bool dumped_pair_occ = false;
            if (!dumped_pair_occ) {
                dumped_pair_occ = true;
                hipFuncAttributes a1{};
                hipFuncAttributes a3{};
                (void)hipFuncGetAttributes(
                    &a1,
                    reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                        (gqh_matvec_pair_swiglu_kernel<true, true, 20, 1>))));
                (void)hipFuncGetAttributes(
                    &a3,
                    reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                        (gqh_matvec_pair_swiglu_kernel<true, true, 20, 3>))));
                std::fprintf(
                    stderr,
                    "[gqh-gemv] pair_swiglu nsb20 warps=%d kCols1 vgpr=%d "
                    "kCols3 vgpr=%d scratch3=%zu lds=%zu\n",
                    kSwigluWarps,
                    a1.numRegs,
                    a3.numRegs,
                    a3.localSizeBytes,
                    a3.sharedSizeBytes);
            }
            const bool nsb20 = in_dim == 5120;
            const int kC = pair_skinny ? (ncols <= 3 ? 3 : 4) : 1;
            auto launch_ps = [&](bool a3, bool b3, int nsb_t, int cols) {
                if (a3 && b3 && nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                }
            };
            if (pair_skinny) {
                static bool dumped_ps = false;
                if (!dumped_ps) {
                    dumped_ps = true;
                    std::fprintf(
                        stderr,
                        "[gqh-gemv] pair skinny12 M=%d kCols=%d in=%d out=%d\n",
                        ncols,
                        kC,
                        in_dim,
                        out_dim);
                }
            }
            launch_ps(
                rung_a == GQH_RUNG_GQH3,
                rung_b == GQH_RUNG_GQH3,
                nsb20 ? 20 : 0,
                kC);
            return launch_result(421, 422);
        }
        if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH3) {
            if (dual) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<true, true, 2>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<true, true, 1>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            }
        } else if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH2_H) {
            if (dual) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<true, false, 2>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<true, false, 1>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            }
        } else if (rung_a == GQH_RUNG_GQH2_H && rung_b == GQH_RUNG_GQH3) {
            if (dual) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<false, true, 2>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<false, true, 1>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            }
        } else {
            if (dual) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<false, false, 2>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME((gqh_matvec_pair_tight_kernel<false, false, 1>)),
                    tblocks, tthreads, tlds, hs, pa, pb, xv, ya, yb, in_dim,
                    out_dim, out_b_eff, scale_a, scale_b, mag_a, mag_b, fuse_swiglu, ileave);
            }
        }
        return launch_result(421, 422);
    }
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
            out_b_eff,
            scale_a,
            scale_b,
            mag_a,
            mag_b,
            fuse_swiglu);
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
            out_b_eff,
            scale_a,
            scale_b,
            mag_a,
            mag_b,
            fuse_swiglu);
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
            out_b_eff,
            scale_a,
            scale_b,
            mag_a,
            mag_b,
            fuse_swiglu);
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
            out_b_eff,
            scale_a,
            scale_b,
            mag_a,
            mag_b,
            fuse_swiglu);
    }
    return launch_result(421, 422);
}

extern "C" int supersonic_gqh_hip_matvec_stream_ab(
    int device_ordinal,
    int rung_a,
    int rung_b,
    const void* wire_a,
    const void* wire_b,
    const void* x,
    void* y_a,
    void* y_b,
    int in_dim,
    int out_a,
    int out_b,
    float scale_a,
    float scale_b,
    int grid_a,
    int grid_b,
    void* stream) {
    const bool a_ok = rung_a == GQH_RUNG_GQH3 || rung_a == GQH_RUNG_GQH2_H;
    const bool b_ok = rung_b == GQH_RUNG_GQH3 || rung_b == GQH_RUNG_GQH2_H;
    if (!a_ok || !b_ok) {
        return 401;
    }
    const int shape_a = validate_shape(rung_a, out_a, in_dim, grid_a);
    const int shape_b = validate_shape(rung_b, out_b, in_dim, grid_b);
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
    hipStream_t hs = static_cast<hipStream_t>(stream);
    auto* pa = static_cast<const uint8_t*>(wire_a);
    auto* pb = static_cast<const uint8_t*>(wire_b);
    gqh_ensure_tight(const_cast<uint8_t*>(pa), in_dim, out_a, rung_a, hs);
    gqh_ensure_tight(const_cast<uint8_t*>(pb), in_dim, out_b, rung_b, hs);
    if (!gqh_is_tight(pa) || !gqh_is_tight(pb)) {
        return 401;
    }
    pa = gqh_decode_wire(pa);
    pb = gqh_decode_wire(pb);
    constexpr int kWarps = 1;
    const int total = out_a + out_b;
    dim3 tblocks(static_cast<unsigned int>((total + kWarps - 1) / kWarps), 1, 1);
    if (tblocks.x == 0) {
        tblocks.x = 1;
    }
    const dim3 tthreads(GQH_WARP * kWarps, 1, 1);
    const int ileave =
        (gqh_is_ileave(pa) && gqh_is_ileave(pb)) ? 1 : 0;
    const float4 mag_a = (rung_a == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_a)
                                                   : load_gqh2h_mag(grid_a);
    const float4 mag_b = (rung_b == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_b)
                                                   : load_gqh2h_mag(grid_b);
    auto* xv = static_cast<const float*>(x);
    auto* ya = static_cast<float*>(y_a);
    auto* yb = static_cast<float*>(y_b);
    if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_ab_tight_kernel<true, true>)),
            tblocks, tthreads, 0, hs, pa, pb, xv, ya, yb, in_dim, out_a, out_b,
            scale_a, scale_b, mag_a, mag_b, ileave);
    } else if (rung_a == GQH_RUNG_GQH3 && rung_b == GQH_RUNG_GQH2_H) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_ab_tight_kernel<true, false>)),
            tblocks, tthreads, 0, hs, pa, pb, xv, ya, yb, in_dim, out_a, out_b,
            scale_a, scale_b, mag_a, mag_b, ileave);
    } else if (rung_a == GQH_RUNG_GQH2_H && rung_b == GQH_RUNG_GQH3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_ab_tight_kernel<false, true>)),
            tblocks, tthreads, 0, hs, pa, pb, xv, ya, yb, in_dim, out_a, out_b,
            scale_a, scale_b, mag_a, mag_b, ileave);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gqh_matvec_ab_tight_kernel<false, false>)),
            tblocks, tthreads, 0, hs, pa, pb, xv, ya, yb, in_dim, out_a, out_b,
            scale_a, scale_b, mag_a, mag_b, ileave);
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
    // llama.cpp mix matvec: 2 warps/block. 16 warps (GQH_MATVEC_WARPS)
    // oversubscribes gfx1100 and was measured slower there.
    constexpr int kMixWarps = 2;
    const dim3 blocks(
        static_cast<unsigned int>((out_dim + kMixWarps - 1) / kMixWarps),
        static_cast<unsigned int>(ncols),
        1);
    const dim3 threads(GQH_WARP * kMixWarps, 1, 1);
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

extern "C" int supersonic_gqh_hip_ensure_tight(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim) {
    if (wire == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    gqh_ensure_tight(static_cast<uint8_t*>(wire), in_dim, out_dim, rung, 0);
    return 0;
}

extern "C" int supersonic_gqh_hip_ensure_padded(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim) {
    if (wire == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    gqh_ensure_padded(static_cast<uint8_t*>(wire), in_dim, out_dim, rung, 0);
    return 0;
}
