// HIP launch wrappers for GQH decode and fused matvec.

#include "gqh.hip"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <mutex>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace {
bool g_gqh_allow_tight = false;
std::recursive_mutex g_gqh_bridge_mutex;

struct GqhWireKey {
    int device_ordinal;
    const void* wire;

    bool operator==(const GqhWireKey& other) const {
        return device_ordinal == other.device_ordinal && wire == other.wire;
    }
};

struct GqhWireKeyHash {
    size_t operator()(const GqhWireKey& key) const {
        const size_t ptr_hash = std::hash<const void*>{}(key.wire);
        return ptr_hash ^ (std::hash<int>{}(key.device_ordinal) +
                           (ptr_hash << 6) + (ptr_hash >> 2));
    }
};

std::unordered_set<GqhWireKey, GqhWireKeyHash> g_gqh_tight;
std::unordered_set<GqhWireKey, GqhWireKeyHash> g_gqh_ileave;
std::unordered_map<GqhWireKey, uint8_t*, GqhWireKeyHash> g_gqh_padded;
int g_gqh_row_off = 0;

// Defined with the gate/up fusion block below. Issues any launch that block is
// holding back; every bridge entry point calls it before doing anything else.
int gqh_gemv_flush();

#ifdef SUPERSONIC_FAILURE_INJECTION
int g_test_unregister_sync_failure = 0;
int g_test_post_enqueue_failure = 0;
#endif
}  // namespace

extern "C" void supersonic_gqh_hip_lock() {
    g_gqh_bridge_mutex.lock();
}

extern "C" void supersonic_gqh_hip_unlock() {
    g_gqh_bridge_mutex.unlock();
}

extern "C" [[noreturn]] void supersonic_gpu_integrity_fail_stop(
    const char* operation,
    int status,
    int device_ordinal) {
    // This handler runs precisely when process-global GPU state may be
    // inconsistent. Do not acquire a lock or call a stdio/allocator path
    // here: either can deadlock or allocate while unwinding the failed path.
    char message[256];
    size_t length = 0;
    auto append = [&](const char* text) {
        if (text == nullptr) {
            text = "unknown";
        }
        while (*text != '\0' && length + 1 < sizeof(message)) {
            message[length++] = *text++;
        }
    };
    auto append_int = [&](int value) {
        long long signed_value = value;
        if (signed_value < 0) {
            append("-");
            signed_value = -(signed_value + 1);
            signed_value += 1;
        }
        char digits[32];
        size_t count = 0;
        do {
            digits[count++] = static_cast<char>('0' + (signed_value % 10));
            signed_value /= 10;
        } while (signed_value != 0 && count < sizeof(digits));
        while (count != 0 && length + 1 < sizeof(message)) {
            message[length++] = digits[--count];
        }
    };
    append("[gpu-integrity] fatal operation=");
    append(operation);
    append(" status=");
    append_int(status);
    append(" ordinal=");
    append_int(device_ordinal);
    append("\n");
    (void)::write(2, message, length);
    (void)::raise(SIGABRT);
    (void)::kill(::getpid(), SIGABRT);
    ::_exit(128 + SIGABRT);
}

namespace {

struct GqhBridgeLockGuard {
    GqhBridgeLockGuard() {
        g_gqh_bridge_mutex.lock();
    }

    ~GqhBridgeLockGuard() {
        g_gqh_bridge_mutex.unlock();
    }
};

#ifdef SUPERSONIC_FAILURE_INJECTION
extern "C" void supersonic_gqh_test_track_wire(int device_ordinal, const void* wire) {
    GqhBridgeLockGuard guard;
    if (wire != nullptr) {
        try {
            g_gqh_tight.insert({device_ordinal, wire});
        } catch (...) {
            // The hook is used before any GPU work in the death-test child;
            // do not let a host allocation exception cross the C ABI.
            return;
        }
    }
}

extern "C" void supersonic_gqh_test_inject_unregister_sync_failure(int status) {
    GqhBridgeLockGuard guard;
    g_test_unregister_sync_failure = status;
}

extern "C" void supersonic_gqh_test_trigger_post_enqueue_failure(int status);
#endif

}  // namespace

extern "C" void supersonic_gqh_hip_set_row_off(int off) {
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
    g_gqh_row_off = off < 0 ? 0 : off;
}

extern "C" void supersonic_gqh_hip_enable_tight_decode() {
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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

int launch_result(int device_ordinal, const char* operation) {
#ifdef SUPERSONIC_FAILURE_INJECTION
    if (g_test_post_enqueue_failure != 0) {
        const int injected_status = g_test_post_enqueue_failure;
        g_test_post_enqueue_failure = 0;
        supersonic_gpu_integrity_fail_stop(operation, injected_status, device_ordinal);
    }
#endif
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            operation, static_cast<int>(launch_status), device_ordinal);
    }
    const hipError_t sync_status = maybe_sync();
    if (sync_status != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            operation, static_cast<int>(sync_status), device_ordinal);
    }
    return 0;
}

#ifdef SUPERSONIC_FAILURE_INJECTION
extern "C" void supersonic_gqh_test_trigger_post_enqueue_failure(int status) {
    g_test_post_enqueue_failure = status;
    (void)launch_result(0, "GQH test post-enqueue");
}
#endif

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
                "gqh device restore", static_cast<int>(err), previous);
        }
        return hipSuccess;
    }

    ~ScopedHipDevice() {
        const hipError_t err = restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "gqh device restore", static_cast<int>(err), previous);
        }
    }
    bool ok() const { return status == hipSuccess; }
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

bool gqh_is_tight(int device_ordinal, const void* wire) {
    return wire != nullptr &&
        g_gqh_tight.find({device_ordinal, wire}) != g_gqh_tight.end();
}

bool gqh_is_ileave(int device_ordinal, const void* wire) {
    return wire != nullptr &&
        g_gqh_ileave.find({device_ordinal, wire}) != g_gqh_ileave.end();
}

const uint8_t* gqh_decode_wire(int device_ordinal, const uint8_t* packed) {
    auto it = g_gqh_padded.find({device_ordinal, packed});
    return it != g_gqh_padded.end() ? it->second : packed;
}

bool gqh_is_padded(int device_ordinal, const void* wire) {
    return wire != nullptr &&
        g_gqh_padded.find({device_ordinal, wire}) != g_gqh_padded.end();
}

bool gqh_want_ileave(int out_dim, int stride) {
    return out_dim >= 4096 && (out_dim % 2) == 0 && stride > 0 &&
        (2 * static_cast<size_t>(stride)) <= static_cast<size_t>(64 * 1024);
}

void gqh_ensure_tight(
    int device_ordinal,
    uint8_t* wire,
    int in_dim,
    int out_dim,
    int rung,
    hipStream_t stream) {
    if (!g_gqh_allow_tight || wire == nullptr) {
        return;
    }
    if (rung != GQH_RUNG_GQH3 && rung != GQH_RUNG_GQH2_H) {
        return;
    }
    if (in_dim <= 0 || out_dim <= 0 || (in_dim % GQH_SUPERBLOCK) != 0) {
        return;
    }
    if (gqh_is_tight(device_ordinal, wire)) {
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
    (void)launch_result(device_ordinal, "GQH tight conversion");
    try {
        g_gqh_tight.insert({device_ordinal, wire});
    } catch (...) {
        supersonic_gpu_integrity_fail_stop(
            "GQH tight metadata publish", -1, device_ordinal);
    }
}

void gqh_ensure_padded(
    int device_ordinal,
    uint8_t* wire,
    int in_dim,
    int out_dim,
    int rung,
    hipStream_t stream) {
    if (!g_gqh_allow_tight || wire == nullptr) {
        return;
    }
    if (rung != GQH_RUNG_GQH3 && rung != GQH_RUNG_GQH2_H) {
        return;
    }
    if (in_dim <= 0 || out_dim <= 0 || (in_dim % GQH_SUPERBLOCK) != 0) {
        return;
    }
    if (g_gqh_padded.find({device_ordinal, wire}) != g_gqh_padded.end()) {
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
        gqh_ensure_tight(device_ordinal, wire, in_dim, out_dim, rung, stream);
        return;
    }
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
    (void)launch_result(device_ordinal, "GQH padded conversion");
    try {
        g_gqh_padded[{device_ordinal, wire}] = dst;
        g_gqh_tight.insert({device_ordinal, wire});
    } catch (...) {
        supersonic_gpu_integrity_fail_stop(
            "GQH padded metadata publish", -1, device_ordinal);
    }
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
    int device_ordinal,
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
    gqh_ensure_tight(
        device_ordinal, const_cast<uint8_t*>(packed), in_dim, out_dim, rung, stream);
    if (gqh_is_tight(device_ordinal, packed)) {
        const bool pad = gqh_is_padded(device_ordinal, packed);
        packed = gqh_decode_wire(device_ordinal, packed);
        const int nsb = in_dim / GQH_SUPERBLOCK;
        const int ncols_in = (int)blocks.y;
        // Fat-K / residual / inproj: 4-wave. Skinny B>1 down
        // (nsb=68, out=5120) uses 8-wave blocks so fewer CTAs share x.
        // The GQH2H lm_head (nsb=20, out=248320 -> 62080 4-wave blocks) also
        // prefers 8-wave: 404 -> 413 GB/s (+2.2%, reproduced in two separate
        // sweeps). It is the only nsb=20 shape that does -- out=17408 is -1.1%
        // and out=12288 is -0.7% at 8 waves -- so the arm is gated on an
        // out_dim no decode projection reaches.
        const int kBaseWarps =
            (ncols_in > 1 && nsb == 68 && out_dim == 5120) ||
                (rung == GQH_RUNG_GQH2_H && nsb == 20 && out_dim >= 65536)
            ? 8
            : 4;
        const int ileave = gqh_is_ileave(device_ordinal, packed) ? 1 : 0;
        // kRows=2 shares one x float4 pair between two weight rows. That pays
        // on the fat-K arms (nsb=68: 467 vs 430 GB/s for kRows=1) but loses on
        // nsb=20, where halving the grid costs more than the x reuse buys.
        // Measured kRows=1 vs kRows=2, nsb=20, gfx1201 (isolated, warm clocks,
        // DRAM-only): out=6144 +5.7%, 10240 +6.9%, 12288 +8.0%, 17408 +7.5%,
        // 248320 +3.5%. Needs the rows==1 && nsb_t==20 dispatch arms below --
        // without them kRows=1 lands on the runtime-nsb instantiation and the
        // -18% there swamps the win (that is why the iteration-1 attempt at
        // this regressed).
        const bool dual = out_dim > 5120 && nsb != 20 &&
            (out_dim % (kBaseWarps * 2)) == 0;
        // 2-wave blocks on the two GQH3 decode singles that measure faster
        // with them (iteration 14, gfx1201, A/A-controlled, 5 passes, base
        // column of an identical-TU driver so this is one kernel at two launch
        // geometries):
        //     tight<3,acc,68>  o=5120   482.9 -> 491.4 GB/s  +1.76%
        //     tight<3,_,20>    o=10240  472.2 -> 474.4       +0.47%
        // This is NOT an occupancy change: hipOccupancyMaxActiveBlocksPer-
        // MultiprocessorS reports 64 waves/WGP = 16 waves/SIMD at 1, 2, 4 and
        // 8 warps/block alike (96 B LDS), so wave count, per-wave bytes and
        // per-wave quantum are all identical -- only block retire/refill
        // granularity changes. That is what distinguishes it from Killed 2's
        // kRows=2, which halved the wave count and doubled the quantum.
        // Bit-exact by construction: each output row is still computed by one
        // wave running the identical superblock loop; only the wave -> block
        // packing moves.
        // Gated to the two shapes that measured positive, and to GQH3:
        //     tight<3,_,20>    o=6144   418.6 -> 412.5  -1.46%  (stays 4)
        //     tight<3,acc,24>  o=5120   414.9 -> 415.0  +0.02%  (stays 4)
        //     tight<2h,_,20>   o=12288  375.7 -> 356.4  -5.14%  (stays 4)
        //     tight<2h,acc,68> o=5120   414.0 -> 407.5  -1.57%  (stays 4)
        //     tight<2h,_,20>  lm_head   466.3 (w8) -> 458.7     (stays 8)
        // The rung split is the same asymmetry the prefetch block records: 2H
        // carries far more exposed compute per byte and reacts badly to any
        // schedule perturbation. Requires !dual so the dual arm's grid math
        // and its (out_dim % (kBaseWarps*2)) test are provably untouched.
        const bool narrow2 = !dual && ncols_in == 1 &&
            rung == GQH_RUNG_GQH3 &&
            ((nsb == 68 && out_dim == 5120) ||
             (nsb == 20 && out_dim == 10240));
        const int kTightWarps = narrow2 ? 2 : kBaseWarps;
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
        // Compile-time kNsb arms for every nsb the decode path launches:
        // 20 (in=5120), 24 (in=6144), 68 (in=17408). Falling through to the
        // runtime-nsb kNsb=0 instantiation loses the unrolled/strength-reduced
        // superblock loop -- measured 338 -> 414 GB/s (+22.5%) at nsb=24,
        // out=5120 on gfx1201. The rows==1 && nsb_t==20 arms are what a
        // kRows=1 dispatch for in=5120 needs; without them it silently lands
        // on kNsb=0 and loses more than the row change can win.
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
            } else if (is3 && rows == 2 && nsb_t == 24) {
                launch_gqh12_tight<true, kAcc, 2, false, 24>(
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
            } else if (is3 && nsb_t == 20) {
                launch_gqh12_tight<true, kAcc, 1, false, 20>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (is3 && nsb_t == 24) {
                launch_gqh12_tight<true, kAcc, 1, false, 24>(
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
            } else if (rows == 2 && nsb_t == 24) {
                launch_gqh12_tight<false, kAcc, 2, false, 24>(
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
            } else if (nsb_t == 20) {
                launch_gqh12_tight<false, kAcc, 1, false, 20>(
                    tblocks, tthreads, stream, packed, xv, yv, in_dim, out_dim,
                    tensor_scale, mag, x_col_stride, y_col_stride, ileave,
                    g_gqh_row_off);
            } else if (nsb_t == 24) {
                launch_gqh12_tight<false, kAcc, 1, false, 24>(
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
    const int shape_status = validate_shape(rung, rows, cols, grid_code);
    if (shape_status != 0) {
        return shape_status;
    }
    if (wire == nullptr || dst == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return backend_failure(411, scoped.status);
    }
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
    return launch_result(device_ordinal, "GQH decode launch");
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
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
                device_ordinal,
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
    return launch_result(device_ordinal, "GQH matvec launch");
}

namespace {
// The un-fused single-tensor GEMV launch. This used to be the body of
// supersonic_gqh_hip_matvec_stream; the extern "C" entry point below wraps it
// with the mixed-rung gate/up pairing.
int gqh_gemv_launch_single(
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
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
                device_ordinal,
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
    return launch_result(device_ordinal, "GQH GEMV launch");
}

// --- mixed-rung MLP gate/up fusion ---------------------------------------
//
// full_attention_bridge_4b.cpp takes the fused gate/up path only when
// `gate_h.rung == up_h.rung`. On Qwen3.8-27B GQH ~28.5 of 64 layers per token
// have gate=GQH3 and up=GQH2H, so they fall into the else branch: two separate
// matvec_stream launches plus a host swiglu_f32. The fused kernel and its
// <3,2h> / <2h,3> dispatch arms already exist and are dead code today. That
// call site is not in this file, so recover the fusion here -- hold the first
// launch of a pair that has already been *observed*, and when its partner
// arrives issue gqh_matvec_pair_swiglu_kernel<..., kFuse=false> instead of the
// two singles.
//
// Bit-exact: kFuse=false writes both fp32 accumulators (y = A.x, y_b = B.x)
// with the same per-row product order as the singles, so the host's own
// swiglu_f32 still runs on identical inputs. gfx1201, in=5120 out=17408
// nsb=20 warps=4: 118.1 us fused vs 131.9 us for the two singles (+11.7%).
//
// Safety, in order of how much it matters:
//  * A call is held only if its wire has already been seen being immediately
//    followed by a pairable partner. The first occurrence of any pair runs
//    unfused, so the pattern is learned, never assumed.
//  * "Immediately followed" is enforced by clearing g_gemv_prev in every other
//    bridge entry point -- all of which also flush a held launch first, so the
//    stream order a caller sees is never reordered.
//  * A call is held only if it would have passed every validation the single
//    path applies, so holding cannot swallow an error status.
//  * Held state is per-process and single-threaded, like g_gqh_tight above.
struct GqhGemvArgs {
    int device_ordinal;
    int rung;
    const void* wire;
    const void* x;
    void* y;
    int in_dim;
    int out_dim;
    int ncols;
    int64_t x_col_stride;
    int64_t y_col_stride;
    float tensor_scale;
    int grid_code;
    void* stream;
    int row_off;
};

GqhGemvArgs g_gemv_prev{};
bool g_gemv_prev_valid = false;
GqhGemvArgs g_gemv_held{};
bool g_gemv_held_valid = false;
std::unordered_set<GqhWireKey, GqhWireKeyHash> g_gemv_fusable;

int gqh_gemv_launch_args(const GqhGemvArgs& a) {
    return gqh_gemv_launch_single(
        a.device_ordinal, a.rung, a.wire, a.x, a.y, a.in_dim, a.out_dim,
        a.ncols, a.x_col_stride, a.y_col_stride, a.tensor_scale, a.grid_code,
        a.stream);
}

// Everything the fused kernel needs from one half, plus everything the single
// path validates -- so a held call can never turn into a late error.
bool gqh_gemv_fusable(const GqhGemvArgs& a) {
    if (!g_gqh_allow_tight || a.ncols != 1 || a.row_off != 0) {
        return false;
    }
    if (a.rung != GQH_RUNG_GQH3 && a.rung != GQH_RUNG_GQH2_H) {
        return false;
    }
    if (validate_shape(a.rung, a.out_dim, a.in_dim, a.grid_code) != 0) {
        return false;
    }
    if (a.wire == nullptr || a.x == nullptr || a.y == nullptr) {
        return false;
    }
    // nsb=20 only. gqh_matvec_pair_swiglu_kernel has compile-time kNsb arms
    // for 20 and nothing else; falling through to the runtime-nsb kNsb=0
    // instantiation costs ~18% (iteration 4), which would swamp the fusion.
    if (a.in_dim != 20 * GQH_SUPERBLOCK) {
        return false;
    }
    if (a.x_col_stride < a.in_dim || a.y_col_stride < a.out_dim) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(a.x) % sizeof(float4)) != 0 ||
        ((a.x_col_stride * static_cast<int64_t>(sizeof(float))) %
         static_cast<int64_t>(sizeof(float4))) != 0) {
        return false;
    }
    return true;
}

// One kernel, one grid, one x: the two halves must agree on everything except
// which weight tensor they read and where they write.
bool gqh_gemv_pairable(const GqhGemvArgs& a, const GqhGemvArgs& b) {
    return gqh_gemv_fusable(a) && gqh_gemv_fusable(b) &&
        a.device_ordinal == b.device_ordinal && a.stream == b.stream &&
        a.x == b.x && a.in_dim == b.in_dim && a.out_dim == b.out_dim &&
        a.x_col_stride == b.x_col_stride &&
        a.y_col_stride == b.y_col_stride && a.wire != b.wire && a.y != b.y;
}

// 0 on launch, 1 when the fused arm is unavailable and the caller must fall
// back to the two singles, or an encoded HIP failure after the pair launch.
int gqh_gemv_launch_pair(const GqhGemvArgs& a, const GqhGemvArgs& b) {
    ScopedHipDevice scoped(a.device_ordinal);
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
    hipStream_t hs = static_cast<hipStream_t>(a.stream);
    auto* pa = static_cast<const uint8_t*>(a.wire);
    auto* pb = static_cast<const uint8_t*>(b.wire);
    gqh_ensure_tight(
        a.device_ordinal, const_cast<uint8_t*>(pa), a.in_dim, a.out_dim, a.rung, hs);
    gqh_ensure_tight(
        b.device_ordinal, const_cast<uint8_t*>(pb), b.in_dim, b.out_dim, b.rung, hs);
    // The pair kernel is tight-only, has no padded-AoS or interleaved arm, and
    // !padded means gqh_decode_wire() is the identity here.
    if (!gqh_is_tight(a.device_ordinal, pa) ||
        !gqh_is_tight(b.device_ordinal, pb) ||
        gqh_is_padded(a.device_ordinal, pa) ||
        gqh_is_padded(b.device_ordinal, pb) ||
        gqh_is_ileave(a.device_ordinal, pa) ||
        gqh_is_ileave(b.device_ordinal, pb)) {
        return 1;
    }
    // Mirror the single path's wave count so a fused pair never launches at a
    // different occupancy than the two launches it replaces (GQH2H nsb=20 with
    // out_dim >= 65536 -- the lm_head shape -- prefers 8).
    // 2-wave blocks when either half is GQH3 (iteration 14; see the narrow2
    // note on the tight singles for why this is not an occupancy change and
    // why it is bit-exact). Measured on the shipped decode shape, nsb=20
    // out=17408: pss<2h,3> 552.8 -> 558.5 GB/s (+1.03%). in_dim is pinned to
    // nsb=20 by gqh_gemv_fusable() and this site is kCols=1 only, so the gate
    // matches the measured arm exactly. A 2H/2H pair keeps its old count.
    const bool narrow2 =
        a.in_dim == 5120 &&
        (a.rung == GQH_RUNG_GQH3 || b.rung == GQH_RUNG_GQH3);
    const int kSwigluWarps =
        (a.rung == GQH_RUNG_GQH2_H && b.rung == GQH_RUNG_GQH2_H &&
         a.out_dim >= 65536)
        ? 8
        : (narrow2 ? 2 : 4);
    dim3 sblocks(
        static_cast<unsigned int>(
            (a.out_dim + kSwigluWarps - 1) / kSwigluWarps),
        1,
        1);
    if (sblocks.x == 0) {
        sblocks.x = 1;
    }
    const dim3 sthreads(GQH_WARP * kSwigluWarps, 1, 1);
    const bool a3 = a.rung == GQH_RUNG_GQH3;
    const bool b3 = b.rung == GQH_RUNG_GQH3;
    const float4 mag_a =
        a3 ? load_gqh3_mag(a.grid_code) : load_gqh2h_mag(a.grid_code);
    const float4 mag_b =
        b3 ? load_gqh3_mag(b.grid_code) : load_gqh2h_mag(b.grid_code);
    auto* xv = static_cast<const float*>(a.x);
    auto* ya = static_cast<float*>(a.y);
    auto* yb = static_cast<float*>(b.y);
    // One line per distinct (a3, b3, out_dim) signature: the learned set is
    // not just gate/up (k/v in the layer_type==1 branch pair too), so a single
    // run has to say which arms actually fused.
    static std::unordered_set<int64_t> dumped_split;
    const int64_t sig = ((int64_t)a.out_dim << 2) | (a3 ? 2 : 0) | (b3 ? 1 : 0);
    bool first_split_signature = false;
    try {
        first_split_signature = dumped_split.insert(sig).second;
    } catch (...) {
        supersonic_gpu_integrity_fail_stop(
            "GQH pair diagnostics allocation", -1, a.device_ordinal);
    }
    if (first_split_signature) {
        std::fprintf(
            stderr,
            "[gqh-gemv] split-pair fused a=%s b=%s in=%d out=%d warps=%d\n",
            a3 ? "3" : "2h",
            b3 ? "3" : "2h",
            a.in_dim,
            a.out_dim,
            kSwigluWarps);
    }
    // in_dim is pinned to nsb=20 by gqh_gemv_fusable(), so these four arms are
    // the whole dispatch.
#define GQH_LAUNCH_SPLIT_PAIR(A, B)                                           \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(                                                      \
            (gqh_matvec_pair_swiglu_kernel<A, B, 20, 1, false>)),              \
        sblocks, sthreads, 0, hs, pa, pb, xv, ya, yb, a.in_dim, a.out_dim,     \
        a.tensor_scale, b.tensor_scale, mag_a, mag_b, a.x_col_stride,          \
        a.y_col_stride, 1)
    if (a3 && b3) {
        GQH_LAUNCH_SPLIT_PAIR(true, true);
    } else if (a3) {
        GQH_LAUNCH_SPLIT_PAIR(true, false);
    } else if (b3) {
        GQH_LAUNCH_SPLIT_PAIR(false, true);
    } else {
        GQH_LAUNCH_SPLIT_PAIR(false, false);
    }
#undef GQH_LAUNCH_SPLIT_PAIR
    // Keep launch/error inspection on the owning device. The scope must not
    // end before hipGetLastError()/maybe_sync: those calls inspect the launch
    // state of the current device.
    return launch_result(a.device_ordinal, "GQH pair GEMV launch");
}

int gqh_gemv_flush() {
    g_gemv_prev_valid = false;
    if (!g_gemv_held_valid) {
        return 0;
    }
    const GqhGemvArgs held = g_gemv_held;
    g_gemv_held_valid = false;
    const int status = gqh_gemv_launch_args(held);
    if (status != 0) {
        supersonic_gpu_integrity_fail_stop(
            "GQH held GEMV flush", status, held.device_ordinal);
    }
    return 0;
}
}  // namespace

extern "C" int supersonic_gqh_hip_unregister_wires(
    int device_ordinal, const void* const* wires, size_t count) {
    GqhBridgeLockGuard guard;
    if (wires == nullptr || count == 0) {
        return 0;
    }

    std::vector<GqhWireKey> keys;
    std::unordered_set<GqhWireKey, GqhWireKeyHash> unique;
    try {
        keys.reserve(count);
        unique.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (wires[i] == nullptr) {
                continue;
            }
            const GqhWireKey key{device_ordinal, wires[i]};
            if (unique.insert(key).second) {
                keys.push_back(key);
            }
        }
    } catch (...) {
        // No bridge state has been touched yet, so an allocator failure can
        // safely be reported to the Rust caller without exposing an exception
        // across the extern-C ABI.
        return static_cast<int>(hipErrorOutOfMemory);
    }
    if (keys.empty()) {
        return 0;
    }

    try {
    bool needs_sync = false;
    for (const GqhWireKey& key : keys) {
        needs_sync = needs_sync || g_gqh_tight.find(key) != g_gqh_tight.end() ||
            g_gqh_ileave.find(key) != g_gqh_ileave.end() ||
            g_gqh_padded.find(key) != g_gqh_padded.end() ||
            (g_gemv_held_valid && g_gemv_held.device_ordinal == device_ordinal &&
             g_gemv_held.wire == key.wire) ||
            (g_gemv_prev_valid && g_gemv_prev.device_ordinal == device_ordinal &&
             g_gemv_prev.wire == key.wire);
    }

    auto remove_metadata = [&]() -> hipError_t {
        for (const GqhWireKey& key : keys) {
            const auto padded = g_gqh_padded.find(key);
            if (padded != g_gqh_padded.end()) {
                // Remove one allocation at a time, retaining the entry when
                // A free failure leaves the entry intact until the fatal
                // integrity handler terminates the process; never erase
                // bookkeeping before HIP confirms the free.
                const hipError_t err = hipFree(padded->second);
                if (err != hipSuccess) {
                    return err;
                }
                g_gqh_padded.erase(padded);
            }
            if (g_gemv_held_valid && g_gemv_held.device_ordinal == device_ordinal &&
                g_gemv_held.wire == key.wire) {
                g_gemv_held_valid = false;
            }
            if (g_gemv_prev_valid && g_gemv_prev.device_ordinal == device_ordinal &&
                g_gemv_prev.wire == key.wire) {
                g_gemv_prev_valid = false;
            }
            g_gemv_fusable.erase(key);
            g_gqh_tight.erase(key);
            g_gqh_ileave.erase(key);
        }
        return hipSuccess;
    };

    // Metadata and padded allocations can be read by an already-launched
    // kernel. Synchronize the owning device before removing or freeing them.
    // Keep the owner scope alive through hipFree; freeing a device pointer on
    // the caller's incidental current device is not ownership-safe. Do not
    // touch HIP for an untracked/fake pointer: registration cleanup is also
    // used by CPU-only unwind tests.
    if (needs_sync) {
        ScopedHipDevice scoped(device_ordinal);
        if (!scoped.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "gqh unregister owner switch", static_cast<int>(scoped.status), device_ordinal);
        }
#ifdef SUPERSONIC_FAILURE_INJECTION
        if (g_test_unregister_sync_failure != 0) {
            const int status = g_test_unregister_sync_failure;
            g_test_unregister_sync_failure = 0;
            supersonic_gpu_integrity_fail_stop(
                "gqh unregister synchronize", status, device_ordinal);
        }
#endif
        bool held_matches = false;
        for (const GqhWireKey& key : keys) {
            held_matches = held_matches ||
                (g_gemv_held_valid && g_gemv_held.device_ordinal == device_ordinal &&
                 g_gemv_held.wire == key.wire);
        }
        if (held_matches) {
            const int flush_status = gqh_gemv_flush();
            if (flush_status != 0) {
                supersonic_gpu_integrity_fail_stop(
                    "gqh unregister held GEMV flush", flush_status, device_ordinal);
            }
        }
        for (const GqhWireKey& key : keys) {
            if (g_gemv_prev_valid && g_gemv_prev.device_ordinal == device_ordinal &&
                g_gemv_prev.wire == key.wire) {
                g_gemv_prev_valid = false;
            }
        }
        const hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "gqh unregister synchronize", static_cast<int>(sync_err), device_ordinal);
        }
        const hipError_t remove_err = remove_metadata();
        if (remove_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "gqh unregister padded free", static_cast<int>(remove_err), device_ordinal);
        }
        const hipError_t restore_err = scoped.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "gqh unregister owner restore", static_cast<int>(restore_err), device_ordinal);
        }
        return 0;
    } else {
        const hipError_t remove_err = remove_metadata();
        if (remove_err != hipSuccess) {
            return static_cast<int>(remove_err);
        }
    }
    } catch (...) {
        // Once the key set is built, any exception means bridge bookkeeping or
        // retained GPU state may already have been touched. Continuing would
        // risk freeing an allocation still referenced by the bridge.
        supersonic_gpu_integrity_fail_stop(
            "GQH unregister bookkeeping exception", -1, device_ordinal);
    }
    return 0;
}

extern "C" int supersonic_gqh_hip_unregister_wire(int device_ordinal, const void* wire) {
    const void* wires[1] = {wire};
    return supersonic_gqh_hip_unregister_wires(device_ordinal, wires, 1);
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
    GqhBridgeLockGuard guard;
    const GqhGemvArgs cur{device_ordinal, rung,          wire,
                          x,              y,             in_dim,
                          out_dim,        ncols,         x_col_stride,
                          y_col_stride,   tensor_scale,  grid_code,
                          stream,         g_gqh_row_off};
    if (g_gemv_held_valid) {
        const GqhGemvArgs held = g_gemv_held;
        g_gemv_held_valid = false;
        g_gemv_prev_valid = false;
        if (gqh_gemv_pairable(held, cur)) {
            const int pair_status = gqh_gemv_launch_pair(held, cur);
            if (pair_status == 0) {
                return 0;
            }
            if (pair_status != 1) {
                return pair_status;
            }
            const int st = gqh_gemv_launch_args(held);
            return st != 0 ? st : gqh_gemv_launch_args(cur);
        }
        const int st = gqh_gemv_launch_args(held);
        if (st != 0) {
            return st;
        }
    }
    if (g_gemv_prev_valid && gqh_gemv_pairable(g_gemv_prev, cur)) {
        try {
            g_gemv_fusable.insert({g_gemv_prev.device_ordinal, g_gemv_prev.wire});
        } catch (...) {
            supersonic_gpu_integrity_fail_stop(
                "GQH fusable metadata allocation", -1, device_ordinal);
        }
    }
    if (gqh_gemv_fusable(cur) &&
        g_gemv_fusable.find({cur.device_ordinal, cur.wire}) != g_gemv_fusable.end()) {
        g_gemv_held = cur;
        g_gemv_held_valid = true;
        g_gemv_prev_valid = false;
        return 0;
    }
    const int st = gqh_gemv_launch_args(cur);
    g_gemv_prev = cur;
    g_gemv_prev_valid = st == 0;
    return st;
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
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
                device_ordinal,
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
    return launch_result(device_ordinal, "GQH accumulated matvec launch");
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
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
        return launch_result(device_ordinal, "GQH GQH4 pair launch");
    }
    gqh_ensure_tight(
        device_ordinal, const_cast<uint8_t*>(pa), in_dim, out_dim, rung_a, hs);
    gqh_ensure_tight(
        device_ordinal, const_cast<uint8_t*>(pb), in_dim, out_b_eff, rung_b, hs);
    const bool tight = gqh_is_tight(device_ordinal, pa) &&
        gqh_is_tight(device_ordinal, pb);
    const bool pad = gqh_is_padded(device_ordinal, pa) &&
        gqh_is_padded(device_ordinal, pb);
    if (tight) {
        pa = gqh_decode_wire(device_ordinal, pa);
        pb = gqh_decode_wire(device_ordinal, pb);
    }
    const float4 mag_a = (rung_a == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_a)
                                                   : load_gqh2h_mag(grid_a);
    const float4 mag_b = (rung_b == GQH_RUNG_GQH3) ? load_gqh3_mag(grid_b)
                                                   : load_gqh2h_mag(grid_b);
    if (tight) {
        // llama.cpp: 4 waves/block, simple SB loop, no LDS.
        constexpr int kTightWarps = 4;
        const int ileave =
            (gqh_is_ileave(device_ordinal, pa) &&
             gqh_is_ileave(device_ordinal, pb)) ? 1 : 0;
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
            const bool pair_skinny = ncols > 1 && ncols <= 4;
            // 2-wave blocks for the kCols=1 (decode) GQH3 pair; see the
            // narrow2 note on the tight singles. Measured nsb=20 out=17408:
            // ps<3,3> 565.8 -> 571.9 GB/s (+1.08%). Restricted to ncols == 1
            // and in_dim == 5120 so the B>1 / skinny arms keep the geometry
            // their own sweeps chose.
            const bool pair_narrow2 = !pair_skinny && ncols == 1 &&
                in_dim == 5120 &&
                (rung_a == GQH_RUNG_GQH3 || rung_b == GQH_RUNG_GQH3);
            const int kSwigluWarps = pair_narrow2 ? 2 : 4;
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
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && b3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, true, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (a3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<true, false, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (b3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, true, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20 && cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20 && cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (nsb_t == 20) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 20, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (cols == 3) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 3>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else if (cols == 4) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 4>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
                        out_dim, scale_a, scale_b, mag_a, mag_b, x_col_stride,
                        y_col_stride, ncols);
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(
                            (gqh_matvec_pair_swiglu_kernel<false, false, 0, 1>)),
                        sblocks, sthreads, tlds, hs, pa, pb, xv, ya, (float*)nullptr, in_dim,
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
            return launch_result(device_ordinal, "GQH skinny pair launch");
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
        return launch_result(device_ordinal, "GQH tight pair launch");
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
    return launch_result(device_ordinal, "GQH pair launch");
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
    hipStream_t hs = static_cast<hipStream_t>(stream);
    auto* pa = static_cast<const uint8_t*>(wire_a);
    auto* pb = static_cast<const uint8_t*>(wire_b);
    gqh_ensure_tight(
        device_ordinal, const_cast<uint8_t*>(pa), in_dim, out_a, rung_a, hs);
    gqh_ensure_tight(
        device_ordinal, const_cast<uint8_t*>(pb), in_dim, out_b, rung_b, hs);
    if (!gqh_is_tight(device_ordinal, pa) || !gqh_is_tight(device_ordinal, pb)) {
        return 401;
    }
    pa = gqh_decode_wire(device_ordinal, pa);
    pb = gqh_decode_wire(device_ordinal, pb);
    constexpr int kWarps = 1;
    const int total = out_a + out_b;
    dim3 tblocks(static_cast<unsigned int>((total + kWarps - 1) / kWarps), 1, 1);
    if (tblocks.x == 0) {
        tblocks.x = 1;
    }
    const dim3 tthreads(GQH_WARP * kWarps, 1, 1);
    const int ileave =
        (gqh_is_ileave(device_ordinal, pa) &&
         gqh_is_ileave(device_ordinal, pb)) ? 1 : 0;
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
    return launch_result(device_ordinal, "GQH AB launch");
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
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
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
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
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
    return launch_result(device_ordinal, "GQH mix launch");
}

extern "C" int supersonic_gqh_hip_ensure_tight(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim) {
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
    if (wire == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
    gqh_ensure_tight(
        device_ordinal, static_cast<uint8_t*>(wire), in_dim, out_dim, rung, 0);
    return 0;
}

extern "C" int supersonic_gqh_hip_ensure_padded(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim) {
    GqhBridgeLockGuard guard;
    (void)gqh_gemv_flush();
    if (wire == nullptr) {
        return 405;
    }
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return backend_failure(421, scoped.status);
    }
    gqh_ensure_padded(
        device_ordinal, static_cast<uint8_t*>(wire), in_dim, out_dim, rung, 0);
    return 0;
}
