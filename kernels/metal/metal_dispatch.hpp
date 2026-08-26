#pragma once

#include <cstddef>
#include <cstdint>

namespace supersonic::metal {

bool init_prefill_library();

bool embedding_lookup_u32(
    int dtype,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out);

bool rms_norm(
    int dtype,
    int n_rows,
    int n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out);

bool matmul_rhs_transposed_tiled(
    int dtype,
    std::uint32_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs,
    void* out);


bool cast(int input_dtype, int output_dtype, int total_elems, const void* xs, void* out);
bool element_add(int dtype, int total_elems, const void* lhs, const void* rhs, void* out);
bool argmax_bf16_rows(int rows, int cols, const void* logits, void* out_index);
bool argmax_f32_as_bf16_rows(int rows, int cols, const void* logits, void* out_index);
bool apply_rope_prefill(int dtype, int seq_len, int num_heads, int head_dim, int half_rot,
    const void* cos_table, const void* sin_table, void* data);
bool transpose_shd_hsd(int dtype, int S, int H, int D, const void* src, void* dst);
bool transpose_shd_hsd_pair(int dtype, int S, int H, int D,
    const void* src_a, const void* src_b, void* dst_a, void* dst_b);
bool transpose_shd_to_cache_bf16(int S, int H, int D, int cache_len, int dst_pos,
    const void* src, void* cache);
bool transpose_pad_conv(int dtype, int S, int C, int pad, const void* src, void* dst);
bool extract_conv_state(int dtype, int S, int C, int kern_minus_1, const void* src, void* dst);
bool prepare_conv_input_tail(int dtype, int S, int C, int pad,
    const void* src, const void* old_tail, void* conv_input, void* new_tail);
bool sigmoid_mul(int dtype, int total_elems, const void* data, const void* gate, void* out);
bool cast_transpose_gate_hsd_to_shd_bf16(int S, int H, int D,
    const void* attn_hsd, const void* gate_shd, void* out_shd);
bool compute_beta_g(int dtype, int seq_len, int nv,
    const void* B, const void* A, const void* dt_bias, const void* a_log_exp, void* beta, void* g);
bool compute_beta_g_ba_bf16(int seq_len, int nv,
    const void* BA, const void* dt_bias, const void* a_log_exp, void* beta, void* g);
bool project_ba_compute_beta_g_bf16(int seq_len, int hidden_dim, int nv,
    const void* hidden, const void* ba_weight, const void* dt_bias, const void* a_log_exp, void* beta, void* g);
bool split_qgate(int dtype, int S, int num_heads, int head_dim,
    const void* src, void* query_out, void* gate_out);
bool split_qgate_norm_bf16(int S, int num_heads, int head_dim, float eps,
    const void* src, const void* norm_w, void* query_out, void* gate_out);
bool split_qkv(int dtype, int S, int key_dim, int val_dim,
    const void* src, void* Q, void* K, void* V);
bool split_qkv_bf16_to_f32(int S, int key_dim, int val_dim,
    const void* src, void* Q, void* K, void* V);
bool split_kv_bf16(int S, int kv_dim, const void* src, void* K, void* V);
bool split_norm_transpose_qkv_bf16(int S, int nk, int nv, int khd, int vhd, float q_scale, float eps,
    const void* src, void* Q, void* K, void* V);
bool rms_norm_gated_sfirst_bf16(int S, int nv, int vhd, float eps,
    const void* hidden_hsd, const void* gate_sfirst, const void* weight, void* out_sfirst);
bool split_qkvz_bf16(int S, int qkv_dim, int z_dim, const void* src, void* QKV, void* Z);
bool repeat_interleave_heads(int dtype, int S, int n_heads, int head_dim, int repeats,
    const void* src, void* dst);
bool repeat_interleave_transpose_hsd(int dtype, int S, int n_heads, int head_dim, int repeats,
    const void* src, void* dst);
bool swiglu_mul(int dtype, int total_elems, const void* gate, const void* up, void* out);
bool swiglu_mul_split(int dtype, int rows, int cols, const void* gate_up, void* out);
bool l2norm(int dtype, int n_rows, int n_cols, float eps, const void* xs, void* out);
bool mul_scalar(int dtype, int total_elems, float scalar, const void* xs, void* out);
bool rms_norm_gated(int dtype, int n_rows, int n_cols, float eps,
    const void* hidden, const void* gate, const void* weight, void* out);
bool fill_conv_tail(int dtype, int qkv_dim, int pad, int total_len, const void* tail, void* conv_input);
bool linear_prefill_conv_pack(int dtype, int batch_size, int conv_dim, int total_len, int seq_len,
    int kernel_size, const void* mixed_qkv, const void* weights, void* out);
bool delta_recurrent_prefill(int dtype, int batch_heads, int seq_len, int k_head_dim, int v_head_dim,
    const void* initial_state, const void* query, const void* key, const void* value,
    const void* beta, const void* g, void* out);
bool full_attention_prefill(int dtype, int batch_size, int q_heads, int kv_heads, int q_len,
    int kv_len, int head_dim, int num_kv_groups, float scale, int seqlen_offset,
    const void* query, const void* key, const void* value, void* out);
bool batched_matmul(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs, void* out);
bool fused_rms_norm_linear(int dtype, int hidden_dim, int out_dim, float eps, int add_unit_offset,
    const void* hidden, const void* norm_weight, const void* proj_weight, void* out);
bool matmul_int4_dequant(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_int4, const void* scale, const void* zero,
    const void* awq_inv_scale, int group_size, int quant_type, float tensor_scale, int grid_code,
    void* out);
bool matmul_int4_dequant_residual_add(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_int4, const void* scale, const void* zero,
    const void* awq_inv_scale, int group_size, int quant_type, float tensor_scale, int grid_code,
    const void* residual, void* out);
bool matmul_fp8_dequant(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_fp8, const void* scale, int block_size, void* out);
bool matmul_ggml_pair_dequant(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_first, const void* rhs_second, int quant_type, void* out);
bool matmul_ggml_pair_swiglu(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_gate, const void* rhs_up, int quant_type, void* out);
bool gqh_decode(int quant_type, int rows, int cols, float tensor_scale, int grid_code, int dst_dtype,
    const void* wire, void* dst);

}  // namespace supersonic::metal
