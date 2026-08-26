#include "prefill_common.metal"

constant constexpr uint kInt4TileM = 16;
constant constexpr uint kInt4TileN = 16;
constant constexpr uint kInt4TileK = 32;

struct Int4MatmulParams {
    uint batch_elems;
    int m;
    int n;
    int k;
    int group_size;
    int quant_type;
    int lhs_dtype;
    int out_dtype;
    int has_awq_inv_scale;
    float tensor_scale;
    int grid_code;
};

kernel void supersonic_metal_matmul_int4_dequant(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device const uchar* zero [[buffer(3)]],
    device const uchar* awq_inv_scale [[buffer(4)]],
    device uchar* out [[buffer(5)]],
    constant Int4MatmulParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) {
        return;
    }
    const uint tx = tid % kInt4TileN;
    const uint ty = tid / kInt4TileN;
    const uint tile_row = tgid.y * kInt4TileM;
    const uint tile_col = tgid.x * kInt4TileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const int row_bytes = qwen35_lowbit_native_int4(params.quant_type)
        ? params.k / 2
        : ggml_k_row_bytes(params.quant_type, params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n) * size_t(row_bytes);
    threadgroup float s_lhs[kInt4TileM][kInt4TileK];
    threadgroup float s_rhs[kInt4TileN][kInt4TileK];
    float acc = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kInt4TileK)) {
        for (uint i = tid; i < kInt4TileM * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint lr = i / kInt4TileK;
            const uint lc = i % kInt4TileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.lhs_dtype)
                : 0.0f;
        }
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n) && gk < uint(params.k)) {
                s_rhs[rr][rc] = dequant_matmul_weight(
                    rhs,
                    uint(rhs_base),
                    params.quant_type,
                    int(gn),
                    int(gk),
                    params.k,
                    params.group_size,
                    scale,
                    zero,
                    params.has_awq_inv_scale != 0 ? awq_inv_scale : nullptr,
                    2,
                    params.tensor_scale,
                    params.grid_code);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n)) {
        const uint out_index = uint(
            size_t(batch_idx) * size_t(params.m) * size_t(params.n) + size_t(row) * size_t(params.n) +
            size_t(col));
        store_elem(out, out_index, params.out_dtype, acc);
    }
}

kernel void supersonic_metal_matmul_int4_dequant_residual_add(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device const uchar* zero [[buffer(3)]],
    device const uchar* awq_inv_scale [[buffer(4)]],
    device const uchar* residual [[buffer(5)]],
    device uchar* out [[buffer(6)]],
    constant Int4MatmulParams& params [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) {
        return;
    }
    const uint tx = tid % kInt4TileN;
    const uint ty = tid / kInt4TileN;
    const uint tile_row = tgid.y * kInt4TileM;
    const uint tile_col = tgid.x * kInt4TileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const int row_bytes = qwen35_lowbit_native_int4(params.quant_type)
        ? params.k / 2
        : ggml_k_row_bytes(params.quant_type, params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n) * size_t(row_bytes);
    threadgroup float s_lhs[kInt4TileM][kInt4TileK];
    threadgroup float s_rhs[kInt4TileN][kInt4TileK];
    float acc = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kInt4TileK)) {
        for (uint i = tid; i < kInt4TileM * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint lr = i / kInt4TileK;
            const uint lc = i % kInt4TileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.lhs_dtype)
                : 0.0f;
        }
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n) && gk < uint(params.k)) {
                s_rhs[rr][rc] = dequant_matmul_weight(
                    rhs,
                    uint(rhs_base),
                    params.quant_type,
                    int(gn),
                    int(gk),
                    params.k,
                    params.group_size,
                    scale,
                    zero,
                    params.has_awq_inv_scale != 0 ? awq_inv_scale : nullptr,
                    2,
                    params.tensor_scale,
                    params.grid_code);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n)) {
        const uint out_index = uint(
            size_t(batch_idx) * size_t(params.m) * size_t(params.n) + size_t(row) * size_t(params.n) +
            size_t(col));
        const float residual_val = load_elem(residual, out_index, params.out_dtype);
        store_elem(out, out_index, params.out_dtype, acc + residual_val);
    }
}

struct Fp8MatmulParams {
    uint batch_elems;
    int m;
    int n;
    int k;
    int block_size;
    int lhs_dtype;
    int out_dtype;
};

kernel void supersonic_metal_matmul_fp8_dequant(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs_fp8 [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant Fp8MatmulParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) {
        return;
    }
    const uint tx = tid % kInt4TileN;
    const uint ty = tid / kInt4TileN;
    const uint tile_row = tgid.y * kInt4TileM;
    const uint tile_col = tgid.x * kInt4TileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n) * size_t(params.k);
    threadgroup float s_lhs[kInt4TileM][kInt4TileK];
    threadgroup float s_rhs[kInt4TileN][kInt4TileK];
    float acc = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kInt4TileK)) {
        for (uint i = tid; i < kInt4TileM * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint lr = i / kInt4TileK;
            const uint lc = i % kInt4TileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.lhs_dtype)
                : 0.0f;
        }
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n) && gk < uint(params.k)) {
                s_rhs[rr][rc] = dequant_weight(
                    rhs_fp8 + uint(rhs_base), scale, int(gn), int(gk), params.k, params.block_size, 2);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n)) {
        const uint out_index = uint(
            size_t(batch_idx) * size_t(params.m) * size_t(params.n) + size_t(row) * size_t(params.n) +
            size_t(col));
        store_elem(out, out_index, params.out_dtype, acc);
    }
}

struct GgmlPairParams {
    uint batch_elems;
    int m;
    int n_each;
    int k;
    int quant_type;
    int lhs_dtype;
    int out_dtype;
};

kernel void supersonic_metal_matmul_ggml_pair_dequant(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs_first [[buffer(1)]],
    device const uchar* rhs_second [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant GgmlPairParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) {
        return;
    }
    const uint tx = tid % kInt4TileN;
    const uint ty = tid / kInt4TileN;
    const uint tile_row = tgid.y * kInt4TileM;
    const uint tile_col = tgid.x * kInt4TileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const int row_bytes = ggml_k_row_bytes(params.quant_type, params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n_each) * size_t(row_bytes);
    threadgroup float s_lhs[kInt4TileM][kInt4TileK];
    threadgroup float s_rhs[kInt4TileN][kInt4TileK];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kInt4TileK)) {
        for (uint i = tid; i < kInt4TileM * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint lr = i / kInt4TileK;
            const uint lc = i % kInt4TileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.lhs_dtype)
                : 0.0f;
        }
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n_each) && gk < uint(params.k)) {
                s_rhs[rr][rc] = ggml_k_dequant_scalar(rhs_first + rhs_base, params.quant_type, int(gn), int(gk), params.k);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n_each)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc0 += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n_each) && gk < uint(params.k)) {
                s_rhs[rr][rc] = ggml_k_dequant_scalar(rhs_second + rhs_base, params.quant_type, int(gn), int(gk), params.k);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n_each)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc1 += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n_each)) {
        const uint out_index0 = uint(
            size_t(batch_idx) * size_t(params.m) * size_t(params.n_each * 2) + size_t(row) * size_t(params.n_each * 2) +
            size_t(col));
        const uint out_index1 = out_index0 + uint(params.n_each);
        store_elem(out, out_index0, params.out_dtype, acc0);
        store_elem(out, out_index1, params.out_dtype, acc1);
    }
}

kernel void supersonic_metal_matmul_ggml_pair_swiglu(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs_gate [[buffer(1)]],
    device const uchar* rhs_up [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant GgmlPairParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) {
        return;
    }
    const uint tx = tid % kInt4TileN;
    const uint ty = tid / kInt4TileN;
    const uint tile_row = tgid.y * kInt4TileM;
    const uint tile_col = tgid.x * kInt4TileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const int row_bytes = ggml_k_row_bytes(params.quant_type, params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n_each) * size_t(row_bytes);
    threadgroup float s_lhs[kInt4TileM][kInt4TileK];
    threadgroup float s_rhs[kInt4TileN][kInt4TileK];
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kInt4TileK)) {
        for (uint i = tid; i < kInt4TileM * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint lr = i / kInt4TileK;
            const uint lc = i % kInt4TileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.lhs_dtype)
                : 0.0f;
        }
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n_each) && gk < uint(params.k)) {
                s_rhs[rr][rc] = ggml_k_dequant_scalar(rhs_gate + rhs_base, params.quant_type, int(gn), int(gk), params.k);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n_each)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc_gate += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kInt4TileN * kInt4TileK; i += kInt4TileM * kInt4TileN) {
            const uint rr = i / kInt4TileK;
            const uint rc = i % kInt4TileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            if (gn < uint(params.n_each) && gk < uint(params.k)) {
                s_rhs[rr][rc] = ggml_k_dequant_scalar(rhs_up + rhs_base, params.quant_type, int(gn), int(gk), params.k);
            } else {
                s_rhs[rr][rc] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n_each)) {
            for (uint kk = 0u; kk < kInt4TileK; ++kk) {
                acc_up += s_lhs[ty][kk] * s_rhs[tx][kk];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n_each)) {
        const uint out_index = uint(
            size_t(batch_idx) * size_t(params.m) * size_t(params.n_each) + size_t(row) * size_t(params.n_each) +
            size_t(col));
        const float silu = acc_gate * sigmoid_fast(acc_gate);
        store_elem(out, out_index, params.out_dtype, silu * acc_up);
    }
}

struct GqhDecodeParams {
    int rows;
    int cols;
    int quant_type;
    float tensor_scale;
    int grid_code;
    int dst_dtype;
};

kernel void supersonic_metal_gqh_decode(
    device const uchar* wire [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant GqhDecodeParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = uint(params.rows) * uint(params.cols);
    if (gid >= total) {
        return;
    }
    const int row = int(gid / uint(params.cols));
    const int col = int(gid % uint(params.cols));
    const float value = qwen35_gqh_dequant_scalar(
        wire,
        params.quant_type,
        row,
        col,
        params.cols,
        params.tensor_scale,
        params.grid_code);
    store_elem(dst, gid, params.dst_dtype, value);
}
