#include "prefill_common.metal"

struct EmbeddingParams {
    int token_count;
    int vocab_size;
    int hidden_size;
    int dtype;
};

kernel void supersonic_metal_embedding_lookup_u32(
    device const uchar* embeddings [[buffer(0)]],
    device const uint* indexes [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int total_elems = params.token_count * params.hidden_size;
    if (int(gid) >= total_elems) return;
    const int token_idx = int(gid) / params.hidden_size;
    const int col = int(gid) - token_idx * params.hidden_size;
    const int64_t row = int64_t(indexes[token_idx]);
    if (row < 0 || row >= int64_t(params.vocab_size)) {
        store_elem(out, gid, params.dtype, 0.0f);
        return;
    }
    const uint src_index = uint(row) * uint(params.hidden_size) + uint(col);
    store_elem(out, gid, params.dtype, load_elem(embeddings, src_index, params.dtype));
}

struct RmsNormParams {
    int n_rows;
    int n_cols;
    float eps;
    int add_unit_offset;
    int dtype;
};

kernel void supersonic_metal_rms_norm(
    device const uchar* xs [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]) {
    if (int(row) >= params.n_rows) return;
    const uint base = row * uint(params.n_cols);
    float partial = 0.0f;
    for (uint col = tid; col < uint(params.n_cols); col += threads_per_tg) {
        const float x = load_elem(xs, base + col, params.dtype);
        partial += x * x;
    }
    const float simd_partial = simd_sum(partial);
    threadgroup float shared_warp_sum[8];
    const uint simd_group = tid / 32u;
    if ((tid & 31u) == 0u) shared_warp_sum[simd_group] = simd_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float shared_inv_rms = 0.0f;
    if (tid == 0) {
        float total = 0.0f;
        const uint warp_count = (threads_per_tg + 31u) / 32u;
        for (uint warp = 0; warp < warp_count; ++warp) total += shared_warp_sum[warp];
        shared_inv_rms = rsqrt(total / float(params.n_cols) + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shared_inv_rms;
    for (uint col = tid; col < uint(params.n_cols); col += threads_per_tg) {
        const float weight_val = load_elem(weight, col, params.dtype) + (params.add_unit_offset != 0 ? 1.0f : 0.0f);
        const float x = load_elem(xs, base + col, params.dtype);
        store_elem(out, base + col, params.dtype, x * inv_rms * weight_val);
    }
}

struct MatmulParams {
    uint batch_elems;
    int m;
    int n;
    int k;
    int dtype;
};

kernel void supersonic_metal_matmul_rhs_transposed_tiled(
    device const uchar* lhs [[buffer(0)]],
    device const uchar* rhs [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint batch_idx = tgid.z;
    if (batch_idx >= params.batch_elems) return;
    const uint tx = tid % kTileN;
    const uint ty = tid / kTileN;
    const uint tile_row = tgid.y * kTileM;
    const uint tile_col = tgid.x * kTileN;
    const uint row = tile_row + ty;
    const uint col = tile_col + tx;
    const size_t lhs_base = size_t(batch_idx) * size_t(params.m) * size_t(params.k);
    const size_t rhs_base = size_t(batch_idx) * size_t(params.n) * size_t(params.k);
    threadgroup float s_lhs[kTileM][kTileK];
    threadgroup float s_rhs[kTileN][kTileK];
    float acc = 0.0f;
    for (int kk_base = 0; kk_base < params.k; kk_base += int(kTileK)) {
        for (uint i = tid; i < kTileM * kTileK; i += kTileM * kTileN) {
            const uint lr = i / kTileK;
            const uint lc = i % kTileK;
            const uint gr = tile_row + lr;
            const uint gc = uint(kk_base) + lc;
            s_lhs[lr][lc] = (gr < uint(params.m) && gc < uint(params.k))
                ? load_elem(lhs, uint(lhs_base) + gr * uint(params.k) + gc, params.dtype) : 0.0f;
        }
        for (uint i = tid; i < kTileN * kTileK; i += kTileM * kTileN) {
            const uint rr = i / kTileK;
            const uint rc = i % kTileK;
            const uint gn = tile_col + rr;
            const uint gk = uint(kk_base) + rc;
            s_rhs[rr][rc] = (gn < uint(params.n) && gk < uint(params.k))
                ? load_elem(rhs, uint(rhs_base) + gn * uint(params.k) + gk, params.dtype) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < uint(params.m) && col < uint(params.n)) {
            for (uint kk = 0; kk < kTileK; ++kk) acc += s_lhs[ty][kk] * s_rhs[tx][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(params.m) && col < uint(params.n)) {
        const uint out_index = uint(size_t(batch_idx) * size_t(params.m) * size_t(params.n) + size_t(row) * size_t(params.n) + size_t(col));
        store_elem(out, out_index, params.dtype, acc);
    }
}

struct ElemParams { uint total_elems; int dtype; };
kernel void supersonic_metal_element_add(
    device const uchar* lhs [[buffer(0)]], device const uchar* rhs [[buffer(1)]], device uchar* out [[buffer(2)]],
    constant ElemParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total_elems) return;
    store_elem(out, gid, p.dtype, load_elem(lhs, gid, p.dtype) + load_elem(rhs, gid, p.dtype));
}

struct ArgmaxParams { uint rows; uint cols; };
kernel void supersonic_metal_argmax_bf16_rows(
    device const bfloat* logits [[buffer(0)]], device uint* out_index [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    threadgroup float shared_vals[256]; threadgroup uint shared_idx[256];
    float best_val = -INFINITY; uint best_idx = 0;
    for (uint col = tid; col < p.cols; col += 256) {
        const float val = float(logits[row * p.cols + col]); const uint idx = col;
        if (val > best_val || (val == best_val && idx < best_idx)) { best_val = val; best_idx = idx; }
    }
    shared_vals[tid] = best_val; shared_idx[tid] = best_idx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shared_vals[tid + offset] > shared_vals[tid] || (shared_vals[tid + offset] == shared_vals[tid] && shared_idx[tid + offset] < shared_idx[tid])) {
                shared_vals[tid] = shared_vals[tid + offset]; shared_idx[tid] = shared_idx[tid + offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_index[row] = shared_idx[0];
}

kernel void supersonic_metal_argmax_f32_as_bf16_rows(
    device const float* logits [[buffer(0)]], device uint* out_index [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    threadgroup float shared_vals[256]; threadgroup uint shared_idx[256];
    float best_val = -INFINITY; uint best_idx = 0;
    for (uint col = tid; col < p.cols; col += 256) {
        const float val = float(bfloat(logits[row * p.cols + col])); const uint idx = col;
        if (val > best_val || (val == best_val && idx < best_idx)) { best_val = val; best_idx = idx; }
    }
    shared_vals[tid] = best_val; shared_idx[tid] = best_idx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shared_vals[tid + offset] > shared_vals[tid] || (shared_vals[tid + offset] == shared_vals[tid] && shared_idx[tid + offset] < shared_idx[tid])) {
                shared_vals[tid] = shared_vals[tid + offset]; shared_idx[tid] = shared_idx[tid + offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_index[row] = shared_idx[0];
}

struct RopeParams { int seq_len; int num_heads; int head_dim; int half_rot; int dtype; };
kernel void supersonic_metal_apply_rope_prefill(
    device const uchar* cos_table [[buffer(0)]], device const uchar* sin_table [[buffer(1)]], device uchar* data [[buffer(2)]],
    constant RopeParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.seq_len) * uint(p.num_heads) * uint(p.half_rot);
    if (gid >= total) return;
    const int i = int(gid % uint(p.half_rot));
    const int h = int((gid / uint(p.half_rot)) % uint(p.num_heads));
    const int pos = int(gid / (uint(p.half_rot) * uint(p.num_heads)));
    const uint cos_off = uint(pos) * uint(p.half_rot) + uint(i);
    const float c = load_elem(cos_table, cos_off, p.dtype);
    const float s = load_elem(sin_table, cos_off, p.dtype);
    const uint base = uint(pos) * uint(p.num_heads) * uint(p.head_dim) + uint(h) * uint(p.head_dim);
    const float x0 = load_elem(data, base + uint(i), p.dtype);
    const float x1 = load_elem(data, base + uint(p.half_rot) + uint(i), p.dtype);
    store_elem(data, base + uint(i), p.dtype, x0 * c - x1 * s);
    store_elem(data, base + uint(p.half_rot) + uint(i), p.dtype, x0 * s + x1 * c);
}

struct ShdParams { int S; int H; int D; int dtype; };
kernel void supersonic_metal_transpose_shd_hsd(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant ShdParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.H) * uint(p.D); if (gid >= total) return;
    const int d = int(gid % uint(p.D)); const int h = int((gid / uint(p.D)) % uint(p.H)); const int s = int(gid / (uint(p.D) * uint(p.H)));
    const uint dst_off = uint(h) * uint(p.S) * uint(p.D) + uint(s) * uint(p.D) + uint(d);
    store_elem(dst, dst_off, p.dtype, load_elem(src, gid, p.dtype));
}

kernel void supersonic_metal_transpose_shd_hsd_pair(
    device const uchar* src_a [[buffer(0)]], device const uchar* src_b [[buffer(1)]],
    device uchar* dst_a [[buffer(2)]], device uchar* dst_b [[buffer(3)]], constant ShdParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.H) * uint(p.D); if (gid >= total) return;
    const int d = int(gid % uint(p.D)); const int h = int((gid / uint(p.D)) % uint(p.H)); const int s = int(gid / (uint(p.D) * uint(p.H)));
    const uint dst_off = uint(h) * uint(p.S) * uint(p.D) + uint(s) * uint(p.D) + uint(d);
    store_elem(dst_a, dst_off, p.dtype, load_elem(src_a, gid, p.dtype));
    store_elem(dst_b, dst_off, p.dtype, load_elem(src_b, gid, p.dtype));
}

struct CacheParams { int S; int H; int D; int cache_len; int dst_pos; };
kernel void supersonic_metal_transpose_shd_to_cache_bf16(
    device const bfloat* src [[buffer(0)]], device bfloat* cache [[buffer(1)]], constant CacheParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.H) * uint(p.D); if (gid >= total) return;
    const int d = int(gid % uint(p.D)); const int h = int((gid / uint(p.D)) % uint(p.H)); const int s = int(gid / (uint(p.D) * uint(p.H)));
    const uint dst_off = (uint(h) * uint(p.cache_len) + uint(p.dst_pos) + uint(s)) * uint(p.D) + uint(d);
    cache[dst_off] = src[gid];
}

struct ConvPadParams { int S; int C; int pad; int dtype; };
kernel void supersonic_metal_transpose_pad_conv(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant ConvPadParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.C); if (gid >= total) return;
    const int c = int(gid % uint(p.C)); const int s = int(gid / uint(p.C));
    const uint dst_off = uint(c) * uint(p.pad + p.S) + uint(p.pad) + uint(s);
    store_elem(dst, dst_off, p.dtype, load_elem(src, gid, p.dtype));
}

struct ExtractConvParams { int S; int C; int kern_minus_1; int dtype; };
kernel void supersonic_metal_extract_conv_state(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant ExtractConvParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.kern_minus_1) * uint(p.C); if (gid >= total) return;
    const int c = int(gid % uint(p.C)); const int t = int(gid / uint(p.C));
    const int src_row = p.S - p.kern_minus_1 + t;
    const uint src_off = uint(src_row) * uint(p.C) + uint(c);
    const uint dst_off = uint(c) * uint(p.kern_minus_1) + uint(t);
    store_elem(dst, dst_off, p.dtype, load_elem(src, src_off, p.dtype));
}

struct PrepConvParams { int S; int C; int pad; int dtype; };
kernel void supersonic_metal_prepare_conv_input_tail(
    device const uchar* src [[buffer(0)]], device const uchar* old_tail [[buffer(1)]],
    device uchar* conv_input [[buffer(2)]], device uchar* new_tail [[buffer(3)]],
    constant PrepConvParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint total_len = uint(p.pad + p.S);
    const uint conv_total = uint(p.C) * total_len;
    const uint tail_total = uint(p.C) * uint(p.pad);
    const uint total = conv_total > tail_total ? conv_total : tail_total;
    if (gid >= total) return;
    if (gid < conv_total) {
        const int c = int(gid / total_len); const int t = int(gid - uint(c) * total_len);
        if (t < p.pad) store_elem(conv_input, gid, p.dtype, load_elem(old_tail, uint(c) * uint(p.pad) + uint(t), p.dtype));
        else store_elem(conv_input, gid, p.dtype, load_elem(src, uint(t - p.pad) * uint(p.C) + uint(c), p.dtype));
    }
    if (gid < tail_total) {
        const int c = int(gid / uint(p.pad)); const int t = int(gid - uint(c) * uint(p.pad));
        const int src_row = p.S - p.pad + t;
        store_elem(new_tail, gid, p.dtype, load_elem(src, uint(src_row) * uint(p.C) + uint(c), p.dtype));
    }
}

kernel void supersonic_metal_sigmoid_mul(
    device const uchar* data [[buffer(0)]], device const uchar* gate [[buffer(1)]], device uchar* out [[buffer(2)]],
    constant ElemParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total_elems) return;
    const float g = load_elem(gate, gid, p.dtype);
    store_elem(out, gid, p.dtype, load_elem(data, gid, p.dtype) * sigmoid_fast(g));
}

struct CastGateParams { int S; int H; int D; };
kernel void supersonic_metal_cast_transpose_gate_hsd_to_shd_bf16(
    device const float* attn_hsd [[buffer(0)]], device const bfloat* gate_shd [[buffer(1)]], device bfloat* out_shd [[buffer(2)]],
    constant CastGateParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.H) * uint(p.D); if (gid >= total) return;
    const int d = int(gid % uint(p.D)); const int h = int((gid / uint(p.D)) % uint(p.H)); const int s = int(gid / (uint(p.D) * uint(p.H)));
    const uint src_off = (uint(h) * uint(p.S) + uint(s)) * uint(p.D) + uint(d);
    const float x = float(bfloat(attn_hsd[src_off]));
    const float g = float(gate_shd[gid]);
    out_shd[gid] = bfloat(x * sigmoid_fast(g));
}

struct BetaGParams { int seq_len; int nv; int dtype; };
kernel void supersonic_metal_compute_beta_g(
    device const uchar* B [[buffer(0)]], device const uchar* A [[buffer(1)]],
    device const uchar* dt_bias [[buffer(2)]], device const uchar* a_log_exp [[buffer(3)]],
    device uchar* beta [[buffer(4)]], device uchar* g [[buffer(5)]], constant BetaGParams& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.seq_len) * uint(p.nv); if (gid >= total) return;
    const int t = int(gid / uint(p.nv)); const int h = int(gid % uint(p.nv));
    const float b_val = load_elem(B, gid, p.dtype);
    store_elem(beta, uint(h) * uint(p.seq_len) + uint(t), p.dtype, sigmoid_fast(b_val));
    const float a_val = load_elem(A, gid, p.dtype);
    const float sp = log(1.0f + exp(a_val + load_elem(dt_bias, uint(h), p.dtype)));
    store_elem(g, uint(h) * uint(p.seq_len) + uint(t), p.dtype, -sp * load_elem(a_log_exp, uint(h), p.dtype));
}

kernel void supersonic_metal_compute_beta_g_ba_bf16(
    device const bfloat* BA [[buffer(0)]], device const bfloat* dt_bias [[buffer(1)]], device const bfloat* a_log_exp [[buffer(2)]],
    device float* beta [[buffer(3)]], device float* g [[buffer(4)]], constant BetaGParams& p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.seq_len) * uint(p.nv); if (gid >= total) return;
    const int t = int(gid / uint(p.nv)); const int h = int(gid % uint(p.nv));
    const uint row = uint(t) * uint(2 * p.nv);
    beta[uint(h) * uint(p.seq_len) + uint(t)] = 1.0f / (1.0f + exp(-float(BA[row + uint(h)])));
    const float sp = log(1.0f + exp(float(BA[row + uint(p.nv) + uint(h)]) + float(dt_bias[h])));
    g[uint(h) * uint(p.seq_len) + uint(t)] = -sp * float(a_log_exp[h]);
}

struct ProjectBaParams { int seq_len; int hidden_dim; int nv; };
kernel void supersonic_metal_project_ba_compute_beta_g_bf16(
    device const bfloat* hidden [[buffer(0)]], device const bfloat* ba_weight [[buffer(1)]],
    device const bfloat* dt_bias [[buffer(2)]], device const bfloat* a_log_exp [[buffer(3)]],
    device float* beta [[buffer(4)]], device float* g [[buffer(5)]], constant ProjectBaParams& p [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint tx = tid % kTileN; const uint ty = tid / kTileN;
    const int tile_row = int(tgid.y) * int(kTileM); const int tile_col = int(tgid.x) * int(kTileN);
    const int row = tile_row + int(ty); const int col = tile_col + int(tx); const int out_cols = 2 * p.nv;
    threadgroup float s_hidden[kTileM][kTileK]; threadgroup float s_weight[kTileN][kTileK]; float acc = 0.0f;
    for (int kk_base = 0; kk_base < p.hidden_dim; kk_base += int(kTileK)) {
        for (uint i = tid; i < kTileM * kTileK; i += kTileM * kTileN) {
            const uint lr = i / kTileK; const uint lc = i % kTileK;
            const int gr = tile_row + int(lr); const int gc = kk_base + int(lc);
            s_hidden[lr][lc] = (gr < p.seq_len && gc < p.hidden_dim) ? float(hidden[uint(gr) * uint(p.hidden_dim) + uint(gc)]) : 0.0f;
        }
        for (uint i = tid; i < kTileN * kTileK; i += kTileM * kTileN) {
            const uint wr = i / kTileK; const uint wc = i % kTileK;
            const int gn = tile_col + int(wr); const int gk = kk_base + int(wc);
            s_weight[wr][wc] = (gn < out_cols && gk < p.hidden_dim) ? float(ba_weight[uint(gn) * uint(p.hidden_dim) + uint(gk)]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < p.seq_len && col < out_cols) for (uint kk = 0; kk < kTileK; ++kk) acc += s_hidden[ty][kk] * s_weight[tx][kk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row >= p.seq_len || col >= out_cols) return;
    const float val = float(bfloat(acc));
    if (col < p.nv) beta[uint(col) * uint(p.seq_len) + uint(row)] = 1.0f / (1.0f + exp(-val));
    else {
        const int h = col - p.nv;
        const float sp = log(1.0f + exp(val + float(dt_bias[h])));
        g[uint(h) * uint(p.seq_len) + uint(row)] = -sp * float(a_log_exp[h]);
    }
}

struct SplitQgateParams { int S; int num_heads; int head_dim; int dtype; };
kernel void supersonic_metal_split_qgate(
    device const uchar* src [[buffer(0)]], device uchar* query_out [[buffer(1)]], device uchar* gate_out [[buffer(2)]],
    constant SplitQgateParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.num_heads) * uint(p.head_dim); if (gid >= total) return;
    const int d = int(gid % uint(p.head_dim)); const int h = int((gid / uint(p.head_dim)) % uint(p.num_heads)); const int s = int(gid / (uint(p.head_dim) * uint(p.num_heads)));
    const uint src_base = uint(s) * uint(p.num_heads) * uint(p.head_dim) * 2u + uint(h) * uint(p.head_dim) * 2u;
    store_elem(query_out, gid, p.dtype, load_elem(src, src_base + uint(d), p.dtype));
    store_elem(gate_out, gid, p.dtype, load_elem(src, src_base + uint(p.head_dim) + uint(d), p.dtype));
}

struct SplitQgateNormParams { int S; int num_heads; int head_dim; float eps; };
kernel void supersonic_metal_split_qgate_norm_bf16(
    device const bfloat* src [[buffer(0)]], device const bfloat* norm_w [[buffer(1)]],
    device bfloat* query_out [[buffer(2)]], device bfloat* gate_out [[buffer(3)]],
    constant SplitQgateNormParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int rows = p.S * p.num_heads; if (int(row) >= rows) return;
    const int s = int(row) / p.num_heads; const int h = int(row) - s * p.num_heads;
    const uint src_base = uint(s) * uint(p.num_heads) * uint(p.head_dim) * 2u + uint(h) * uint(p.head_dim) * 2u;
    const uint out_base = row * uint(p.head_dim);
    float partial = 0.0f; for (int d = int(tid); d < p.head_dim; d += 256) partial += float(src[src_base + uint(d)]) * float(src[src_base + uint(d)]);
    threadgroup float shared_sum[256]; shared_sum[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 128; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_sum[tid] += shared_sum[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    threadgroup float inv_rms; if (tid == 0) inv_rms = rsqrt(shared_sum[0] / float(p.head_dim) + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = int(tid); d < p.head_dim; d += 256) {
        query_out[out_base + uint(d)] = bfloat(float(src[src_base + uint(d)]) * inv_rms * float(norm_w[d]));
        gate_out[out_base + uint(d)] = src[src_base + uint(p.head_dim) + uint(d)];
    }
}

struct SplitQkvParams { int S; int key_dim; int val_dim; int dtype; };
kernel void supersonic_metal_split_qkv(
    device const uchar* src [[buffer(0)]], device uchar* Q_out [[buffer(1)]], device uchar* K_out [[buffer(2)]], device uchar* V_out [[buffer(3)]],
    constant SplitQkvParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const int qkv_dim = p.key_dim * 2 + p.val_dim; const uint total = uint(p.S) * uint(qkv_dim); if (gid >= total) return;
    const int s = int(gid / uint(qkv_dim)); const int c = int(gid % uint(qkv_dim)); const float val = load_elem(src, gid, p.dtype);
    if (c < p.key_dim) store_elem(Q_out, uint(s) * uint(p.key_dim) + uint(c), p.dtype, val);
    else if (c < p.key_dim * 2) store_elem(K_out, uint(s) * uint(p.key_dim) + uint(c - p.key_dim), p.dtype, val);
    else store_elem(V_out, uint(s) * uint(p.val_dim) + uint(c - p.key_dim * 2), p.dtype, val);
}

struct SplitKvParams { int S; int kv_dim; };
kernel void supersonic_metal_split_kv_bf16(
    device const bfloat* src [[buffer(0)]], device bfloat* K_out [[buffer(1)]], device bfloat* V_out [[buffer(2)]],
    constant SplitKvParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint total = uint(p.S) * uint(p.kv_dim); if (gid >= total) return;
    const int s = int(gid / uint(p.kv_dim)); const int c = int(gid % uint(p.kv_dim));
    const uint src_base = uint(s) * uint(2 * p.kv_dim);
    K_out[gid] = src[src_base + uint(c)]; V_out[gid] = src[src_base + uint(p.kv_dim) + uint(c)];
}

kernel void supersonic_metal_split_qkv_bf16_to_f32(
    device const bfloat* src [[buffer(0)]], device float* Q_out [[buffer(1)]], device float* K_out [[buffer(2)]], device float* V_out [[buffer(3)]],
    constant SplitQkvParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const int qkv_dim = p.key_dim * 2 + p.val_dim; const uint total = uint(p.S) * uint(qkv_dim); if (gid >= total) return;
    const int s = int(gid / uint(qkv_dim)); const int c = int(gid % uint(qkv_dim)); const float val = float(src[gid]);
    if (c < p.key_dim) Q_out[uint(s) * uint(p.key_dim) + uint(c)] = val;
    else if (c < p.key_dim * 2) K_out[uint(s) * uint(p.key_dim) + uint(c - p.key_dim)] = val;
    else V_out[uint(s) * uint(p.val_dim) + uint(c - p.key_dim * 2)] = val;
}

struct SplitNormQkvParams { int S; int nk; int nv; int khd; int vhd; float q_scale; float eps; };
kernel void supersonic_metal_split_norm_transpose_qkv_bf16(
    device const bfloat* src [[buffer(0)]], device float* Q_out [[buffer(1)]], device float* K_out [[buffer(2)]], device float* V_out [[buffer(3)]],
    constant SplitNormQkvParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int key_dim = p.nk * p.khd; const int val_dim = p.nv * p.vhd; const int qkv_dim = 2 * key_dim + val_dim;
    const int q_rows = p.S * p.nk; const int kv_rows = p.S * p.nv; const int r = int(row);
    if (r < q_rows) {
        const int s = r / p.nk; const int h = r - s * p.nk; const uint src_base = uint(s) * uint(qkv_dim) + uint(h) * uint(p.khd);
        float partial = 0.0f; for (int d = int(tid); d < p.khd; d += 256) partial += float(src[src_base + uint(d)]) * float(src[src_base + uint(d)]);
        threadgroup float shared_sum[256]; shared_sum[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = 128; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_sum[tid] += shared_sum[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        threadgroup float inv_norm; if (tid == 0) inv_norm = rsqrt(shared_sum[0] + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int d = int(tid); d < p.khd; d += 256) {
            const float val = float(src[src_base + uint(d)]) * inv_norm * p.q_scale;
            for (int oh = h; oh < p.nv; oh += p.nk) Q_out[(uint(oh) * uint(p.S) + uint(s)) * uint(p.khd) + uint(d)] = val;
        }
        return;
    }
    const int k_row = r - q_rows;
    if (k_row < q_rows) {
        const int s = k_row / p.nk; const int h = k_row - s * p.nk;
        const uint src_base = uint(s) * uint(qkv_dim) + uint(key_dim) + uint(h) * uint(p.khd);
        float partial = 0.0f; for (int d = int(tid); d < p.khd; d += 256) partial += float(src[src_base + uint(d)]) * float(src[src_base + uint(d)]);
        threadgroup float shared_sum[256]; shared_sum[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = 128; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_sum[tid] += shared_sum[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        threadgroup float inv_norm; if (tid == 0) inv_norm = rsqrt(shared_sum[0] + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int d = int(tid); d < p.khd; d += 256) {
            const float val = float(src[src_base + uint(d)]) * inv_norm;
            for (int oh = h; oh < p.nv; oh += p.nk) K_out[(uint(oh) * uint(p.S) + uint(s)) * uint(p.khd) + uint(d)] = val;
        }
        return;
    }
    const int v_row = k_row - q_rows; if (v_row >= kv_rows) return;
    const int s = v_row / p.nv; const int h = v_row - s * p.nv;
    const uint src_base = uint(s) * uint(qkv_dim) + uint(2 * key_dim) + uint(h) * uint(p.vhd);
    for (int d = int(tid); d < p.vhd; d += 256) V_out[(uint(h) * uint(p.S) + uint(s)) * uint(p.vhd) + uint(d)] = float(src[src_base + uint(d)]);
}

struct RmsGatedSfirstParams { int S; int nv; int vhd; float eps; };
kernel void supersonic_metal_rms_norm_gated_sfirst_bf16(
    device const bfloat* hidden_hsd [[buffer(0)]], device const bfloat* gate_sfirst [[buffer(1)]], device const bfloat* weight [[buffer(2)]], device bfloat* out_sfirst [[buffer(3)]],
    constant RmsGatedSfirstParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int total_rows = p.S * p.nv; if (int(row) >= total_rows) return;
    const int s = int(row) / p.nv; const int h = int(row) - s * p.nv;
    device const bfloat* row_hidden = hidden_hsd + (uint(h) * uint(p.S) + uint(s)) * uint(p.vhd);
    device const bfloat* row_gate = gate_sfirst + uint(s) * uint(p.nv) * uint(p.vhd) + uint(h) * uint(p.vhd);
    device bfloat* row_out = out_sfirst + uint(s) * uint(p.nv) * uint(p.vhd) + uint(h) * uint(p.vhd);
    float partial = 0.0f; for (int col = int(tid); col < p.vhd; col += 256) partial += float(row_hidden[col]) * float(row_hidden[col]);
    threadgroup float shared_sum[256]; shared_sum[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 128; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_sum[tid] += shared_sum[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    threadgroup float inv_rms; if (tid == 0) inv_rms = rsqrt(shared_sum[0] / float(p.vhd) + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int col = int(tid); col < p.vhd; col += 256) {
        const float gate_x = float(row_gate[col]);
        row_out[col] = bfloat(float(row_hidden[col]) * inv_rms * float(weight[col]) * gate_x * sigmoid_fast(gate_x));
    }
}

struct SplitQkvzParams { int S; int qkv_dim; int z_dim; };
kernel void supersonic_metal_split_qkvz_bf16(
    device const bfloat* src [[buffer(0)]], device bfloat* QKV_out [[buffer(1)]], device bfloat* Z_out [[buffer(2)]],
    constant SplitQkvzParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const int total_dim = p.qkv_dim + p.z_dim; const uint total = uint(p.S) * uint(total_dim); if (gid >= total) return;
    const int s = int(gid / uint(total_dim)); const int c = int(gid % uint(total_dim)); const bfloat val = src[gid];
    if (c < p.qkv_dim) QKV_out[uint(s) * uint(p.qkv_dim) + uint(c)] = val;
    else Z_out[uint(s) * uint(p.z_dim) + uint(c - p.qkv_dim)] = val;
}

struct RepeatParams { int S; int n_heads; int head_dim; int repeats; int dtype; };
kernel void supersonic_metal_repeat_interleave_heads(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant RepeatParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const int out_heads = p.n_heads * p.repeats; const uint total = uint(p.S) * uint(out_heads) * uint(p.head_dim); if (gid >= total) return;
    const int d = int(gid % uint(p.head_dim)); const int oh = int((gid / uint(p.head_dim)) % uint(out_heads)); const int s = int(gid / (uint(p.head_dim) * uint(out_heads)));
    const int src_h = oh % p.n_heads;
    const uint src_off = uint(s) * uint(p.n_heads) * uint(p.head_dim) + uint(src_h) * uint(p.head_dim) + uint(d);
    store_elem(dst, gid, p.dtype, load_elem(src, src_off, p.dtype));
}

kernel void supersonic_metal_repeat_interleave_transpose_hsd(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant RepeatParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const int out_heads = p.n_heads * p.repeats; const uint total = uint(out_heads) * uint(p.S) * uint(p.head_dim); if (gid >= total) return;
    const int d = int(gid % uint(p.head_dim)); const int s = int((gid / uint(p.head_dim)) % uint(p.S)); const int oh = int(gid / (uint(p.head_dim) * uint(p.S)));
    const int src_h = oh % p.n_heads;
    const uint src_off = uint(s) * uint(p.n_heads) * uint(p.head_dim) + uint(src_h) * uint(p.head_dim) + uint(d);
    store_elem(dst, gid, p.dtype, load_elem(src, src_off, p.dtype));
}
