namespace {

extern "C" const void* supersonic_metal_dummy_buffer();

const void* registered_dummy_buffer() {
    static const void* dummy = nullptr;
    if (dummy == nullptr) {
        dummy = supersonic_metal_dummy_buffer();
    }
    return dummy;
}

const void* optional_device_buffer(const void* ptr) {
    return ptr != nullptr ? ptr : registered_dummy_buffer();
}

bool bind_all(id<MTLComputeCommandEncoder> encoder, bool* configured, std::initializer_list<const void*> ptrs) {
    NSUInteger index = 0;
    for (const void* ptr : ptrs) {
        if (!bind_host_buffer(encoder, index++, ptr)) {
            *configured = false;
            return false;
        }
    }
    return true;
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, int total_elems, NSUInteger threads = 256) {
    const NSUInteger grid = static_cast<NSUInteger>((total_elems + int(threads) - 1) / int(threads));
    [encoder dispatchThreads:MTLSizeMake(grid * threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void dispatch_rows(id<MTLComputeCommandEncoder> encoder, int rows, NSUInteger threads = 256) {
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void dispatch_tiled_matmul(id<MTLComputeCommandEncoder> encoder, int m, int n, std::uint32_t batch) {
    const int grid_x = (n + 15) / 16;
    const int grid_y = (m + 15) / 16;
    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

}  // namespace

bool cast(int input_dtype, int output_dtype, int total_elems, const void* xs, void* out) {
    struct P { int total_elems; int in_dtype; int out_dtype; } p{total_elems, input_dtype, output_dtype};
    return run_command_encoder("supersonic_metal_cast", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {xs, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, total_elems);
    });
}

bool element_add(int dtype, int total_elems, const void* lhs, const void* rhs, void* out) {
    struct P { int total_elems; int dtype; } p{total_elems, dtype};
    return run_command_encoder("supersonic_metal_element_add", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, total_elems);
    });
}

bool argmax_bf16_rows(int rows, int cols, const void* logits, void* out_index) {
    struct P { unsigned rows; unsigned cols; } p{static_cast<unsigned>(rows), static_cast<unsigned>(cols)};
    return run_command_encoder("supersonic_metal_argmax_bf16_rows", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {logits, out_index})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_rows(encoder, rows);
    });
}

bool argmax_f32_as_bf16_rows(int rows, int cols, const void* logits, void* out_index) {
    struct P { unsigned rows; unsigned cols; } p{static_cast<unsigned>(rows), static_cast<unsigned>(cols)};
    return run_command_encoder("supersonic_metal_argmax_f32_as_bf16_rows", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {logits, out_index})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_rows(encoder, rows);
    });
}

bool apply_rope_prefill(int dtype, int seq_len, int num_heads, int head_dim, int half_rot,
    const void* cos_table, const void* sin_table, void* data) {
    struct P { int seq_len; int num_heads; int head_dim; int half_rot; int dtype; }
        p{seq_len, num_heads, head_dim, half_rot, dtype};
    return run_command_encoder("supersonic_metal_apply_rope_prefill", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {cos_table, sin_table, data})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, seq_len * num_heads * half_rot);
    });
}

bool transpose_shd_hsd(int dtype, int S, int H, int D, const void* src, void* dst) {
    struct P { int S; int H; int D; int dtype; } p{S, H, D, dtype};
    return run_command_encoder("supersonic_metal_transpose_shd_hsd", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, S * H * D);
    });
}

bool transpose_shd_hsd_pair(int dtype, int S, int H, int D,
    const void* src_a, const void* src_b, void* dst_a, void* dst_b) {
    struct P { int S; int H; int D; int dtype; } p{S, H, D, dtype};
    return run_command_encoder("supersonic_metal_transpose_shd_hsd_pair", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src_a, src_b, dst_a, dst_b})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(encoder, S * H * D);
    });
}

bool transpose_shd_to_cache_bf16(int S, int H, int D, int cache_len, int dst_pos,
    const void* src, void* cache) {
    struct P { int S; int H; int D; int cache_len; int dst_pos; } p{S, H, D, cache_len, dst_pos};
    return run_command_encoder("supersonic_metal_transpose_shd_to_cache_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, cache})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, S * H * D);
    });
}

bool transpose_pad_conv(int dtype, int S, int C, int pad, const void* src, void* dst) {
    struct P { int S; int C; int pad; int dtype; } p{S, C, pad, dtype};
    return run_command_encoder("supersonic_metal_transpose_pad_conv", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, S * C);
    });
}

bool extract_conv_state(int dtype, int S, int C, int kern_minus_1, const void* src, void* dst) {
    struct P { int S; int C; int kern_minus_1; int dtype; } p{S, C, kern_minus_1, dtype};
    return run_command_encoder("supersonic_metal_extract_conv_state", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, kern_minus_1 * C);
    });
}

bool prepare_conv_input_tail(int dtype, int S, int C, int pad,
    const void* src, const void* old_tail, void* conv_input, void* new_tail) {
    struct P { int S; int C; int pad; int dtype; } p{S, C, pad, dtype};
    const int conv_total = C * (pad + S);
    const int tail_total = C * pad;
    return run_command_encoder("supersonic_metal_prepare_conv_input_tail", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, old_tail, conv_input, new_tail})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(encoder, conv_total > tail_total ? conv_total : tail_total);
    });
}

bool sigmoid_mul(int dtype, int total_elems, const void* data, const void* gate, void* out) {
    struct P { int total_elems; int dtype; } p{total_elems, dtype};
    return run_command_encoder("supersonic_metal_sigmoid_mul", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {data, gate, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, total_elems);
    });
}

bool cast_transpose_gate_hsd_to_shd_bf16(int S, int H, int D,
    const void* attn_hsd, const void* gate_shd, void* out_shd) {
    struct P { int S; int H; int D; } p{S, H, D};
    return run_command_encoder("supersonic_metal_cast_transpose_gate_hsd_to_shd_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {attn_hsd, gate_shd, out_shd})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, S * H * D);
    });
}

bool compute_beta_g(int dtype, int seq_len, int nv,
    const void* B, const void* A, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    struct P { int seq_len; int nv; int dtype; } p{seq_len, nv, dtype};
    return run_command_encoder("supersonic_metal_compute_beta_g", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {B, A, dt_bias, a_log_exp, beta, g})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:6];
        dispatch_1d(encoder, seq_len * nv);
    });
}

bool compute_beta_g_ba_bf16(int seq_len, int nv,
    const void* BA, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    struct P { int seq_len; int nv; int dtype; } p{seq_len, nv, 2};
    return run_command_encoder("supersonic_metal_compute_beta_g_ba_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {BA, dt_bias, a_log_exp, beta, g})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:5];
        dispatch_1d(encoder, seq_len * nv);
    });
}

bool project_ba_compute_beta_g_bf16(int seq_len, int hidden_dim, int nv,
    const void* hidden, const void* ba_weight, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    struct P { int seq_len; int hidden_dim; int nv; } p{seq_len, hidden_dim, nv};
    const int out_cols = 2 * nv;
    return run_command_encoder("supersonic_metal_project_ba_compute_beta_g_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {hidden, ba_weight, dt_bias, a_log_exp, beta, g})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:6];
        const int grid_x = (out_cols + 15) / 16;
        const int grid_y = (seq_len + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}

bool split_qgate(int dtype, int S, int num_heads, int head_dim,
    const void* src, void* query_out, void* gate_out) {
    struct P { int S; int num_heads; int head_dim; int dtype; } p{S, num_heads, head_dim, dtype};
    return run_command_encoder("supersonic_metal_split_qgate", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, query_out, gate_out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, S * num_heads * head_dim);
    });
}

bool split_qgate_norm_bf16(int S, int num_heads, int head_dim, float eps,
    const void* src, const void* norm_w, void* query_out, void* gate_out) {
    struct P { int S; int num_heads; int head_dim; float eps; } p{S, num_heads, head_dim, eps};
    return run_command_encoder("supersonic_metal_split_qgate_norm_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, norm_w, query_out, gate_out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_rows(encoder, S * num_heads);
    });
}

bool split_qkv(int dtype, int S, int key_dim, int val_dim,
    const void* src, void* Q, void* K, void* V) {
    struct P { int S; int key_dim; int val_dim; int dtype; } p{S, key_dim, val_dim, dtype};
    return run_command_encoder("supersonic_metal_split_qkv", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, Q, K, V})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(encoder, S * (key_dim * 2 + val_dim));
    });
}

bool split_qkv_bf16_to_f32(int S, int key_dim, int val_dim,
    const void* src, void* Q, void* K, void* V) {
    struct P { int S; int key_dim; int val_dim; int dtype; } p{S, key_dim, val_dim, 2};
    return run_command_encoder("supersonic_metal_split_qkv_bf16_to_f32", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, Q, K, V})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(encoder, S * (key_dim * 2 + val_dim));
    });
}

bool split_kv_bf16(int S, int kv_dim, const void* src, void* K, void* V) {
    struct P { int S; int kv_dim; } p{S, kv_dim};
    return run_command_encoder("supersonic_metal_split_kv_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, K, V})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, S * kv_dim);
    });
}

bool split_norm_transpose_qkv_bf16(int S, int nk, int nv, int khd, int vhd, float q_scale, float eps,
    const void* src, void* Q, void* K, void* V) {
    struct P { int S; int nk; int nv; int khd; int vhd; float q_scale; float eps; } p{S, nk, nv, khd, vhd, q_scale, eps};
    const int rows = S * nk + S * nk + S * nv;
    return run_command_encoder("supersonic_metal_split_norm_transpose_qkv_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, Q, K, V})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_rows(encoder, rows);
    });
}

bool rms_norm_gated_sfirst_bf16(int S, int nv, int vhd, float eps,
    const void* hidden_hsd, const void* gate_sfirst, const void* weight, void* out_sfirst) {
    struct P { int S; int nv; int vhd; float eps; } p{S, nv, vhd, eps};
    return run_command_encoder("supersonic_metal_rms_norm_gated_sfirst_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {hidden_hsd, gate_sfirst, weight, out_sfirst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_rows(encoder, S * nv);
    });
}

bool split_qkvz_bf16(int S, int qkv_dim, int z_dim, const void* src, void* QKV, void* Z) {
    struct P { int S; int qkv_dim; int z_dim; } p{S, qkv_dim, z_dim};
    return run_command_encoder("supersonic_metal_split_qkvz_bf16", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, QKV, Z})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, S * (qkv_dim + z_dim));
    });
}

bool repeat_interleave_heads(int dtype, int S, int n_heads, int head_dim, int repeats,
    const void* src, void* dst) {
    struct P { int S; int n_heads; int head_dim; int repeats; int dtype; } p{S, n_heads, head_dim, repeats, dtype};
    return run_command_encoder("supersonic_metal_repeat_interleave_heads", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, S * n_heads * head_dim * repeats);
    });
}

bool repeat_interleave_transpose_hsd(int dtype, int S, int n_heads, int head_dim, int repeats,
    const void* src, void* dst) {
    struct P { int S; int n_heads; int head_dim; int repeats; int dtype; } p{S, n_heads, head_dim, repeats, dtype};
    return run_command_encoder("supersonic_metal_repeat_interleave_transpose_hsd", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {src, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, S * n_heads * head_dim * repeats);
    });
}

bool swiglu_mul(int dtype, int total_elems, const void* gate, const void* up, void* out) {
    struct P { int elem_count; int dtype; } p{total_elems, dtype};
    return run_command_encoder("supersonic_metal_swiglu_mul", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {gate, up, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, total_elems);
    });
}

bool swiglu_mul_split(int dtype, int rows, int cols, const void* gate_up, void* out) {
    struct P { int rows; int cols; int dtype; } p{rows, cols, dtype};
    return run_command_encoder("supersonic_metal_swiglu_mul_split", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {gate_up, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, rows * cols);
    });
}

bool l2norm(int dtype, int n_rows, int n_cols, float eps, const void* xs, void* out) {
    struct P { int n_rows; int n_cols; float eps; int dtype; } p{n_rows, n_cols, eps, dtype};
    return run_command_encoder("supersonic_metal_l2norm", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {xs, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_rows(encoder, n_rows);
    });
}

bool mul_scalar(int dtype, int total_elems, float scalar, const void* xs, void* out) {
    struct P { int total_elems; float scalar; int dtype; } p{total_elems, scalar, dtype};
    return run_command_encoder("supersonic_metal_mul_scalar", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {xs, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, total_elems);
    });
}

bool rms_norm_gated(int dtype, int n_rows, int n_cols, float eps,
    const void* hidden, const void* gate, const void* weight, void* out) {
    struct P { int n_rows; int n_cols; float eps; int dtype; } p{n_rows, n_cols, eps, dtype};
    return run_command_encoder("supersonic_metal_rms_norm_gated", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {hidden, gate, weight, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_rows(encoder, n_rows);
    });
}

bool fill_conv_tail(int dtype, int qkv_dim, int pad, int total_len, const void* tail, void* conv_input) {
    struct P { int qkv_dim; int pad; int total_len; int dtype; } p{qkv_dim, pad, total_len, dtype};
    return run_command_encoder("supersonic_metal_fill_conv_tail", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {tail, conv_input})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, qkv_dim * pad);
    });
}

bool linear_prefill_conv_pack(int dtype, int batch_size, int conv_dim, int total_len, int seq_len,
    int kernel_size, const void* mixed_qkv, const void* weights, void* out) {
    struct P { int batch_size; int conv_dim; int total_len; int seq_len; int kernel_size; int dtype; }
        p{batch_size, conv_dim, total_len, seq_len, kernel_size, dtype};
    return run_command_encoder("supersonic_metal_linear_prefill_conv_pack", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {mixed_qkv, weights, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, batch_size * seq_len * conv_dim);
    });
}

bool delta_recurrent_prefill(int dtype, int batch_heads, int seq_len, int k_head_dim, int v_head_dim,
    const void* initial_state, const void* query, const void* key, const void* value,
    const void* beta, const void* g, void* out) {
    struct P { int batch_heads; int seq_len; int k_head_dim; int v_head_dim; int dtype; }
        p{batch_heads, seq_len, k_head_dim, v_head_dim, dtype};
    return run_command_encoder("supersonic_metal_delta_recurrent_prefill", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {initial_state, query, key, value, beta, g, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:7];
        dispatch_1d(encoder, batch_heads * v_head_dim, 256);
    });
}

bool full_attention_prefill(int dtype, int batch_size, int q_heads, int kv_heads, int q_len,
    int kv_len, int head_dim, int num_kv_groups, float scale, int seqlen_offset,
    const void* query, const void* key, const void* value, void* out) {
    struct P {
        int batch_size; int q_heads; int kv_heads; int q_len; int kv_len; int head_dim;
        int num_kv_groups; float scale; int seqlen_offset; int dtype;
    } p{batch_size, q_heads, kv_heads, q_len, kv_len, head_dim, num_kv_groups, scale, seqlen_offset, dtype};
    id<MTLBuffer> counter = full_attention_row_counter();
    if (counter == nil) {
        return false;
    }
    *static_cast<uint32_t*>(counter.contents) = 0u;
    return run_command_encoder("supersonic_metal_full_attention_prefill", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_host_buffer(encoder, 0, query) || !bind_host_buffer(encoder, 1, key) ||
            !bind_host_buffer(encoder, 2, value) || !bind_host_buffer(encoder, 3, out)) {
            *configured = false;
            return;
        }
        [encoder setBuffer:counter offset:0 atIndex:4];
        [encoder setBytes:&p length:sizeof(p) atIndex:5];
        const int total_rows = batch_size * q_heads * q_len;
        const int groups = (total_rows + 31) / 32;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    });
}

bool batched_matmul(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs, void* out) {
    struct P { unsigned batch_elems; int m; int n; int k; int dtype; }
        p{batch_elems, m, n, k, dtype};
    return run_command_encoder("supersonic_metal_batched_matmul", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(encoder, static_cast<int>(batch_elems) * m * n);
    });
}

bool fused_rms_norm_linear(int dtype, int hidden_dim, int out_dim, float eps, int add_unit_offset,
    const void* hidden, const void* norm_weight, const void* proj_weight, void* out) {
    struct P { int hidden_dim; int out_dim; float eps; int add_unit_offset; int dtype; }
        p{hidden_dim, out_dim, eps, add_unit_offset, dtype};
    return run_command_encoder("supersonic_metal_fused_rms_norm_linear", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {hidden, norm_weight, proj_weight, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_rows(encoder, out_dim);
    });
}

bool matmul_int4_dequant(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_int4, const void* scale, const void* zero,
    const void* awq_inv_scale, int group_size, int quant_type, float tensor_scale, int grid_code,
    void* out) {
    struct P {
        unsigned batch_elems; int m; int n; int k; int group_size; int quant_type;
        int lhs_dtype; int out_dtype; int has_awq_inv_scale; float tensor_scale; int grid_code;
    } p{batch_elems, m, n, k, group_size, quant_type, dtype, dtype,
        awq_inv_scale != nullptr ? 1 : 0, tensor_scale, grid_code};
    const void* awq = optional_device_buffer(awq_inv_scale);
    return run_command_encoder("supersonic_metal_matmul_int4_dequant", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs_int4, scale, zero, awq, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:6];
        dispatch_tiled_matmul(encoder, m, n, batch_elems);
    });
}

bool matmul_int4_dequant_residual_add(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_int4, const void* scale, const void* zero,
    const void* awq_inv_scale, int group_size, int quant_type, float tensor_scale, int grid_code,
    const void* residual, void* out) {
    struct P {
        unsigned batch_elems; int m; int n; int k; int group_size; int quant_type;
        int lhs_dtype; int out_dtype; int has_awq_inv_scale; float tensor_scale; int grid_code;
    } p{batch_elems, m, n, k, group_size, quant_type, dtype, dtype,
        awq_inv_scale != nullptr ? 1 : 0, tensor_scale, grid_code};
    const void* awq = optional_device_buffer(awq_inv_scale);
    return run_command_encoder("supersonic_metal_matmul_int4_dequant_residual_add", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs_int4, scale, zero, awq, residual, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:7];
        dispatch_tiled_matmul(encoder, m, n, batch_elems);
    });
}

bool matmul_fp8_dequant(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_fp8, const void* scale, int block_size, void* out) {
    struct P { unsigned batch_elems; int m; int n; int k; int block_size; int lhs_dtype; int out_dtype; }
        p{batch_elems, m, n, k, block_size, dtype, dtype};
    return run_command_encoder("supersonic_metal_matmul_fp8_dequant", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs_fp8, scale, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_tiled_matmul(encoder, m, n, batch_elems);
    });
}

bool matmul_ggml_pair_dequant(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_first, const void* rhs_second, int quant_type, void* out) {
    struct P { unsigned batch_elems; int m; int n_each; int k; int quant_type; int lhs_dtype; int out_dtype; }
        p{batch_elems, m, n_each, k, quant_type, dtype, dtype};
    return run_command_encoder("supersonic_metal_matmul_ggml_pair_dequant", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs_first, rhs_second, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_tiled_matmul(encoder, m, n_each, batch_elems);
    });
}

bool matmul_ggml_pair_swiglu(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_gate, const void* rhs_up, int quant_type, void* out) {
    struct P { unsigned batch_elems; int m; int n_each; int k; int quant_type; int lhs_dtype; int out_dtype; }
        p{batch_elems, m, n_each, k, quant_type, dtype, dtype};
    return run_command_encoder("supersonic_metal_matmul_ggml_pair_swiglu", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {lhs, rhs_gate, rhs_up, out})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_tiled_matmul(encoder, m, n_each, batch_elems);
    });
}

bool gqh_decode(int quant_type, int rows, int cols, float tensor_scale, int grid_code, int dst_dtype,
    const void* wire, void* dst) {
    struct P { int rows; int cols; int quant_type; float tensor_scale; int grid_code; int dst_dtype; }
        p{rows, cols, quant_type, tensor_scale, grid_code, dst_dtype};
    return run_command_encoder("supersonic_metal_gqh_decode", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_all(encoder, configured, {wire, dst})) return;
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(encoder, rows * cols);
    });
}
