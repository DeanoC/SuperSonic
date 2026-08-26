#include "prefill_common.metal"

struct CastParams { int total_elems; int in_dtype; int out_dtype; };
kernel void supersonic_metal_cast(
    device const uchar* xs [[buffer(0)]], device uchar* out [[buffer(1)]], constant CastParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.total_elems) return;
    store_elem(out, gid, p.out_dtype, load_elem(xs, gid, p.in_dtype));
}

struct SwigluParams { int elem_count; int dtype; };
kernel void supersonic_metal_swiglu_mul(
    device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]], device uchar* out [[buffer(2)]],
    constant SwigluParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.elem_count) return;
    const float gate_x = load_elem(gate, gid, p.dtype);
    const float up_x = load_elem(up, gid, p.dtype);
    store_elem(out, gid, p.dtype, (gate_x / (1.0f + exp(-gate_x))) * up_x);
}

struct SwigluSplitParams { int rows; int cols; int dtype; };
kernel void supersonic_metal_swiglu_mul_split(
    device const uchar* gate_up [[buffer(0)]], device uchar* out [[buffer(1)]], constant SwigluSplitParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const int elem_count = p.rows * p.cols; if (int(gid) >= elem_count) return;
    const int row = int(gid) / p.cols; const int col = int(gid) - row * p.cols;
    const uint row_base = uint(row) * uint(p.cols) * 2u;
    const float gate_x = load_elem(gate_up, row_base + uint(col), p.dtype);
    const float up_x = load_elem(gate_up, row_base + uint(p.cols) + uint(col), p.dtype);
    store_elem(out, gid, p.dtype, (gate_x / (1.0f + exp(-gate_x))) * up_x);
}

struct L2NormParams { int n_rows; int n_cols; float eps; int dtype; };
kernel void supersonic_metal_l2norm(
    device const uchar* xs [[buffer(0)]], device uchar* out [[buffer(1)]], constant L2NormParams& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= p.n_rows) return;
    const uint base = uint(row) * uint(p.n_cols);
    float partial = 0.0f;
    for (int col = int(tid); col < p.n_cols; col += 256) partial += load_elem(xs, base + uint(col), p.dtype) * load_elem(xs, base + uint(col), p.dtype);
    threadgroup float shared_sum[256]; shared_sum[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 128; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_sum[tid] += shared_sum[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    threadgroup float inv_norm; if (tid == 0) inv_norm = rsqrt(shared_sum[0] + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int col = int(tid); col < p.n_cols; col += 256) store_elem(out, base + uint(col), p.dtype, load_elem(xs, base + uint(col), p.dtype) * inv_norm);
}

struct MulScalarParams { int total_elems; float scalar; int dtype; };
kernel void supersonic_metal_mul_scalar(
    device const uchar* xs [[buffer(0)]], device uchar* out [[buffer(1)]], constant MulScalarParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= p.total_elems) return;
    store_elem(out, gid, p.dtype, load_elem(xs, gid, p.dtype) * p.scalar);
}

struct RmsNormGatedParams { int n_rows; int n_cols; float eps; int dtype; };
kernel void supersonic_metal_rms_norm_gated(
    device const uchar* hidden [[buffer(0)]], device const uchar* gate [[buffer(1)]], device const uchar* weight [[buffer(2)]], device uchar* out [[buffer(3)]],
    constant RmsNormGatedParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= p.n_rows) return;
    const uint base = uint(row) * uint(p.n_cols);
    float partial = 0.0f;
    for (int col = int(tid); col < p.n_cols; col += 256) partial += load_elem(hidden, base + uint(col), p.dtype) * load_elem(hidden, base + uint(col), p.dtype);
    const float warp_sum = wave_sum(partial);
    threadgroup float shared_warp_sum[8]; threadgroup float shared_inv_rms;
    const uint lane = tid & 31u; const uint warp = tid >> 5;
    if (lane == 0u) shared_warp_sum[warp] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        float block_sum = lane < ((256u + 31u) / 32u) ? shared_warp_sum[lane] : 0.0f;
        block_sum = wave_sum(block_sum);
        if (lane == 0u) shared_inv_rms = rsqrt(block_sum / float(p.n_cols) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int col = int(tid); col < p.n_cols; col += 256) {
        const float gate_x = load_elem(gate, base + uint(col), p.dtype);
        const float gate_silu = gate_x * sigmoid_fast(gate_x);
        store_elem(out, base + uint(col), p.dtype, load_elem(hidden, base + uint(col), p.dtype) * shared_inv_rms * load_elem(weight, uint(col), p.dtype) * gate_silu);
    }
}

struct FillConvTailParams { int qkv_dim; int pad; int total_len; int dtype; };
kernel void supersonic_metal_fill_conv_tail(
    device const uchar* tail [[buffer(0)]], device uchar* conv_input [[buffer(1)]], constant FillConvTailParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const int total = p.qkv_dim * p.pad; if (int(gid) >= total) return;
    const int channel = int(gid) / p.pad; const int tap = int(gid) - channel * p.pad;
    store_elem(conv_input, uint(channel) * uint(p.total_len) + uint(tap), p.dtype, load_elem(tail, gid, p.dtype));
}

struct LinearConvPackParams { int batch_size; int conv_dim; int total_len; int seq_len; int kernel_size; int dtype; };
kernel void supersonic_metal_linear_prefill_conv_pack(
    device const uchar* mixed_qkv [[buffer(0)]], device const uchar* weights [[buffer(1)]], device uchar* out [[buffer(2)]],
    constant LinearConvPackParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint output_elems = uint(p.batch_size) * uint(p.seq_len) * uint(p.conv_dim);
    if (gid >= output_elems) return;
    const uint b = gid / (uint(p.seq_len) * uint(p.conv_dim));
    const uint rem = gid - b * uint(p.seq_len) * uint(p.conv_dim);
    const uint t = rem / uint(p.conv_dim); const uint c = rem - t * uint(p.conv_dim);
    const uint input_c_offset = b * uint(p.conv_dim) * uint(p.total_len) + c * uint(p.total_len);
    const uint weight_offset = c * uint(p.kernel_size);
    float acc = 0.0f;
    for (int tap = 0; tap < p.kernel_size; ++tap) {
        acc += load_elem(mixed_qkv, input_c_offset + t + uint(tap), p.dtype) * load_elem(weights, weight_offset + uint(tap), p.dtype);
    }
    store_elem(out, gid, p.dtype, acc * sigmoid_fast(acc));
}

constant constexpr int kDeltaMaxK = 256;
inline void delta_recurrent_prefill_impl(
    int batch_heads, int seq_len, int k_head_dim, int v_head_dim,
    device const uchar* initial_state, device const uchar* query, device const uchar* key, device const uchar* value,
    device const uchar* beta, device const uchar* g, device uchar* out, int tid, int dtype) {
    const int total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > kDeltaMaxK) return;
    const int bh = tid / v_head_dim; const int v_idx = tid - bh * v_head_dim;
    const int state_stride = k_head_dim * v_head_dim;
    const int token_stride_k = seq_len * k_head_dim; const int token_stride_v = seq_len * v_head_dim; const int token_stride_s = seq_len;
    const int out_base = bh * (seq_len + k_head_dim) * v_head_dim;
    float state[kDeltaMaxK];
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) state[k_idx] = load_elem(initial_state, uint(bh * state_stride + k_idx * v_head_dim + v_idx), dtype);
    for (int t = 0; t < seq_len; ++t) {
        const float g_t = exp(load_elem(g, uint(bh * token_stride_s + t), dtype));
        const uint key_row = uint(bh * token_stride_k + t * k_head_dim);
        const uint value_row = uint(bh * token_stride_v + t * v_head_dim);
        const uint beta_row = uint(bh * token_stride_s + t);
        float kv_mem = 0.0f; float out_t = 0.0f;
        const float delta_scale = load_elem(beta, beta_row, dtype);
        const float v_t = load_elem(value, value_row + uint(v_idx), dtype);
        int k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const float k0 = load_elem(key, key_row + uint(k_idx), dtype);
            const float k1 = load_elem(key, key_row + uint(k_idx + 1), dtype);
            const float k2 = load_elem(key, key_row + uint(k_idx + 2), dtype);
            const float k3 = load_elem(key, key_row + uint(k_idx + 3), dtype);
            float s0 = state[k_idx] * g_t; float s1 = state[k_idx + 1] * g_t; float s2 = state[k_idx + 2] * g_t; float s3 = state[k_idx + 3] * g_t;
            kv_mem += s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
            state[k_idx] = s0; state[k_idx + 1] = s1; state[k_idx + 2] = s2; state[k_idx + 3] = s3;
        }
        for (; k_idx < k_head_dim; ++k_idx) { state[k_idx] *= g_t; kv_mem += state[k_idx] * load_elem(key, key_row + uint(k_idx), dtype); }
        const float delta = (v_t - kv_mem) * delta_scale;
        k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const float k0 = load_elem(key, key_row + uint(k_idx), dtype);
            const float k1 = load_elem(key, key_row + uint(k_idx + 1), dtype);
            const float k2 = load_elem(key, key_row + uint(k_idx + 2), dtype);
            const float k3 = load_elem(key, key_row + uint(k_idx + 3), dtype);
            float s0 = state[k_idx] + k0 * delta; float s1 = state[k_idx + 1] + k1 * delta; float s2 = state[k_idx + 2] + k2 * delta; float s3 = state[k_idx + 3] + k3 * delta;
            state[k_idx] = s0; state[k_idx + 1] = s1; state[k_idx + 2] = s2; state[k_idx + 3] = s3;
            out_t += s0 * load_elem(query, key_row + uint(k_idx), dtype) + s1 * load_elem(query, key_row + uint(k_idx + 1), dtype)
                + s2 * load_elem(query, key_row + uint(k_idx + 2), dtype) + s3 * load_elem(query, key_row + uint(k_idx + 3), dtype);
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += load_elem(key, key_row + uint(k_idx), dtype) * delta;
            out_t += state[k_idx] * load_elem(query, key_row + uint(k_idx), dtype);
        }
        store_elem(out, uint(out_base + t * v_head_dim + v_idx), dtype, out_t);
    }
    const int state_out = out_base + seq_len * v_head_dim;
    for (int k_idx = 0; k_idx < k_head_dim; ++k_idx) store_elem(out, uint(state_out + k_idx * v_head_dim + v_idx), dtype, state[k_idx]);
}

struct DeltaRecurrentParams { int batch_heads; int seq_len; int k_head_dim; int v_head_dim; int dtype; };
kernel void supersonic_metal_delta_recurrent_prefill(
    device const uchar* initial_state [[buffer(0)]], device const uchar* query [[buffer(1)]], device const uchar* key [[buffer(2)]],
    device const uchar* value [[buffer(3)]], device const uchar* beta [[buffer(4)]], device const uchar* g [[buffer(5)]], device uchar* out [[buffer(6)]],
    constant DeltaRecurrentParams& p [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
    delta_recurrent_prefill_impl(p.batch_heads, p.seq_len, p.k_head_dim, p.v_head_dim, initial_state, query, key, value, beta, g, out, int(tid), p.dtype);
}

struct FullAttnParams {
    int batch_size; int q_heads; int kv_heads; int q_len; int kv_len; int head_dim; int num_kv_groups;
    float scale; int seqlen_offset; int dtype;
};
kernel void supersonic_metal_full_attention_prefill(
    device const uchar* query [[buffer(0)]], device const uchar* key [[buffer(1)]], device const uchar* value [[buffer(2)]],
    device float* out [[buffer(3)]], device atomic_uint* row_counter [[buffer(4)]], constant FullAttnParams& p [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_group_idx [[simdgroup_index_in_threadgroup]]) {
    if (simd_lane >= 32u) return;
    const int total_rows = p.batch_size * p.q_heads * p.q_len;
    while (true) {
        uint row = 0;
        if (simd_lane == 0u) row = atomic_fetch_add_explicit(row_counter, 1u, memory_order_relaxed);
        row = simd_broadcast(row, 0u);
        if (int(row) >= total_rows) return;
        const int q_pos = int(row) % p.q_len;
        const int q_head = (int(row) / p.q_len) % p.q_heads;
        const int batch = int(row) / (p.q_len * p.q_heads);
        const int kv_head = q_head / p.num_kv_groups;
        const int causal_limit = min(p.kv_len, p.seqlen_offset + q_pos + 1);
        const uint q_base = uint(((batch * p.q_heads + q_head) * p.q_len + q_pos) * p.head_dim);
        const uint k_head_base = uint((batch * p.kv_heads + kv_head) * p.kv_len * p.head_dim);
        const uint v_head_base = uint((batch * p.kv_heads + kv_head) * p.kv_len * p.head_dim);
        const uint out_base = uint(((batch * p.q_heads + q_head) * p.q_len + q_pos) * p.head_dim);
        float running_max = -INFINITY; float denom = 0.0f;
        float local_acc[8] = {0,0,0,0,0,0,0,0}; int local_dims[8] = {-1,-1,-1,-1,-1,-1,-1,-1}; int local_count = 0;
        for (int d = int(simd_lane); d < p.head_dim && local_count < 8; d += 32) local_dims[local_count++] = d;
        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const uint k_row = k_head_base + uint(k_pos * p.head_dim);
            const uint v_row = v_head_base + uint(k_pos * p.head_dim);
            float partial = 0.0f;
            for (int d = int(simd_lane); d < p.head_dim; d += 32) partial += load_elem(query, q_base + uint(d), p.dtype) * load_elem(key, k_row + uint(d), p.dtype);
            const float score = wave_sum(partial) * p.scale;
            float prev_scale; float curr_scale;
            if (!isfinite(running_max)) { running_max = score; denom = 1.0f; prev_scale = 0.0f; curr_scale = 1.0f; }
            else {
                const float new_max = max(running_max, score);
                prev_scale = exp(running_max - new_max); curr_scale = exp(score - new_max);
                denom = denom * prev_scale + curr_scale; running_max = new_max;
            }
            if (curr_scale == 1.0f && prev_scale == 0.0f) {
                for (int i = 0; i < local_count; ++i) local_acc[i] = load_elem(value, v_row + uint(local_dims[i]), p.dtype);
            } else {
                for (int i = 0; i < local_count; ++i) local_acc[i] = local_acc[i] * prev_scale + curr_scale * load_elem(value, v_row + uint(local_dims[i]), p.dtype);
            }
        }
        const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int i = 0; i < local_count; ++i) out[out_base + uint(local_dims[i])] = local_acc[i] * inv_denom;
    }
}

struct BatchedMatmulParams { uint batch_elems; int m; int n; int k; int dtype; };
kernel void supersonic_metal_batched_matmul(
    device const uchar* lhs [[buffer(0)]], device const uchar* rhs [[buffer(1)]], device uchar* out [[buffer(2)]],
    constant BatchedMatmulParams& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint total = p.batch_elems * uint(p.m) * uint(p.n);
    if (gid >= total) return;
    const uint matrix_idx = gid % (uint(p.m) * uint(p.n));
    const uint batch_idx = gid / (uint(p.m) * uint(p.n));
    const int row = int(matrix_idx / uint(p.n)); const int col = int(matrix_idx % uint(p.n));
    const uint lhs_base = batch_idx * uint(p.m) * uint(p.k);
    const uint rhs_base = batch_idx * uint(p.k) * uint(p.n);
    float acc = 0.0f;
    for (int kk = 0; kk < p.k; ++kk) acc += load_elem(lhs, lhs_base + uint(row) * uint(p.k) + uint(kk), p.dtype) * load_elem(rhs, rhs_base + uint(kk) * uint(p.n) + uint(col), p.dtype);
    store_elem(out, gid, p.dtype, acc);
}

struct FusedRmsLinearParams { int hidden_dim; int out_dim; float eps; int add_unit_offset; int dtype; };
kernel void supersonic_metal_fused_rms_norm_linear(
    device const uchar* hidden [[buffer(0)]], device const uchar* norm_weight [[buffer(1)]], device const uchar* proj_weight [[buffer(2)]], device uchar* out [[buffer(3)]],
    constant FusedRmsLinearParams& p [[buffer(4)]], uint out_row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint block_size [[threads_per_threadgroup]]) {
    if (int(out_row) >= p.out_dim) return;
    threadgroup float shared_normed[2048]; threadgroup float shared_scratch[256];
    float partial_sq = 0.0f;
    for (int col = int(tid); col < p.hidden_dim; col += int(block_size)) partial_sq += load_elem(hidden, uint(col), p.dtype) * load_elem(hidden, uint(col), p.dtype);
    shared_scratch[tid] = partial_sq; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = int(block_size) / 2; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_scratch[tid] += shared_scratch[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    threadgroup float inv_rms; if (tid == 0) inv_rms = rsqrt(shared_scratch[0] / float(p.hidden_dim) + p.eps); threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int col = int(tid); col < p.hidden_dim; col += int(block_size)) {
        const float w = load_elem(norm_weight, uint(col), p.dtype) + (p.add_unit_offset != 0 ? 1.0f : 0.0f);
        shared_normed[col] = load_elem(hidden, uint(col), p.dtype) * inv_rms * w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint w_row_base = uint(out_row) * uint(p.hidden_dim);
    float partial_dot = 0.0f;
    for (int col = int(tid); col < p.hidden_dim; col += int(block_size)) partial_dot += load_elem(proj_weight, w_row_base + uint(col), p.dtype) * shared_normed[col];
    shared_scratch[tid] = partial_dot; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = int(block_size) / 2; stride > 0; stride >>= 1) { if (int(tid) < stride) shared_scratch[tid] += shared_scratch[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (tid == 0) store_elem(out, uint(out_row), p.dtype, shared_scratch[0]);
}
