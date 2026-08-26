#!/usr/bin/env python3
"""Generate Metal prefill dispatch helpers and HIP ABI bridge wrappers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPATCH_HPP = ROOT / "kernels/metal/metal_dispatch.hpp"
DISPATCH_MM = ROOT / "kernels/metal/metal_dispatch.mm"
DISPATCH_INC = ROOT / "kernels/metal/metal_dispatch_ops.inc"
BRIDGE_MM = ROOT / "kernels/metal/prefill_bridge_rest.mm"
STUBS = ROOT / "kernels/metal/hip_symbol_stubs.cc"
MANIFEST = ROOT / "crates/kernel-ffi/kernel-groups.toml"

DISPATCH_DECLS = """
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
    const void* awq_inv_scale, int group_size, int quant_type, void* out);
bool matmul_int4_dequant_residual_add(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_int4, const void* scale, const void* zero,
    const void* awq_inv_scale, int group_size, int quant_type, const void* residual, void* out);
bool matmul_fp8_dequant(int dtype, std::uint32_t batch_elems, int m, int n, int k,
    const void* lhs, const void* rhs_fp8, const void* scale, int block_size, void* out);
bool matmul_ggml_pair_dequant(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_first, const void* rhs_second, int quant_type, void* out);
bool matmul_ggml_pair_swiglu(int dtype, std::uint32_t batch_elems, int m, int n_each, int k,
    const void* lhs, const void* rhs_gate, const void* rhs_up, int quant_type, void* out);
"""

BRIDGES: list[tuple[str, str, int, str, str]] = [
    ("supersonic_prefill_encode_bridge_status", "", 0, "", ""),
    ("supersonic_qwen35_4b_bf16_matmul_bridge_status", "", 0, "", ""),
    ("supersonic_qwen35_4b_hip_device_supports_wmma_i8", "", 0,
     "size_t device_ordinal, int* out_supported", ""),
    ("supersonic_qwen35_hip_cast", "cast", 280,
     "int input_dtype, int output_dtype, size_t device_ordinal, size_t total_elems, const void* xs, void* out",
     "supersonic::metal::cast(input_dtype, output_dtype, static_cast<int>(total_elems), xs, out)"),
    ("supersonic_qwen35_hip_element_add", "element_add", 300,
     "int dtype, size_t device_ordinal, size_t total_elems, const void* lhs, const void* rhs, void* out",
     "supersonic::metal::element_add(dtype, static_cast<int>(total_elems), lhs, rhs, out)"),
    ("supersonic_qwen35_hip_argmax_bf16_rows", "argmax_bf16_rows", 400,
     "size_t device_ordinal, size_t rows, size_t cols, const void* logits, void* out_index",
     "supersonic::metal::argmax_bf16_rows(static_cast<int>(rows), static_cast<int>(cols), logits, out_index)"),
    ("supersonic_qwen35_hip_argmax_f32_as_bf16_rows", "argmax_f32_as_bf16_rows", 401,
     "size_t device_ordinal, size_t rows, size_t cols, const void* logits, void* out_index",
     "supersonic::metal::argmax_f32_as_bf16_rows(static_cast<int>(rows), static_cast<int>(cols), logits, out_index)"),
    ("supersonic_qwen35_hip_apply_rope_prefill", "apply_rope_prefill", 310,
     "int dtype, size_t device_ordinal, size_t seq_len, size_t num_heads, size_t head_dim, size_t half_rot, const void* cos_table, const void* sin_table, void* data",
     "supersonic::metal::apply_rope_prefill(dtype, static_cast<int>(seq_len), static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<int>(half_rot), cos_table, sin_table, data)"),
    ("supersonic_qwen35_hip_transpose_shd_hsd", "transpose_shd_hsd", 320,
     "int dtype, size_t device_ordinal, size_t S, size_t H, size_t D, const void* src, void* dst",
     "supersonic::metal::transpose_shd_hsd(dtype, static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst)"),
    ("supersonic_qwen35_hip_transpose_shd_hsd_pair", "transpose_shd_hsd_pair", 325,
     "int dtype, size_t device_ordinal, size_t S, size_t H, size_t D, const void* src_a, const void* src_b, void* dst_a, void* dst_b",
     "supersonic::metal::transpose_shd_hsd_pair(dtype, static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src_a, src_b, dst_a, dst_b)"),
    ("supersonic_qwen35_hip_transpose_shd_to_cache_bf16", "transpose_shd_to_cache_bf16", 321,
     "size_t device_ordinal, size_t S, size_t H, size_t D, size_t cache_len, size_t dst_pos, const void* src, void* cache",
     "supersonic::metal::transpose_shd_to_cache_bf16(static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), static_cast<int>(cache_len), static_cast<int>(dst_pos), src, cache)"),
    ("supersonic_qwen35_hip_transpose_pad_conv", "transpose_pad_conv", 329,
     "int dtype, size_t device_ordinal, size_t S, size_t C, size_t pad, const void* src, void* dst",
     "supersonic::metal::transpose_pad_conv(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst)"),
    ("supersonic_qwen35_hip_extract_conv_state", "extract_conv_state", 340,
     "int dtype, size_t device_ordinal, size_t S, size_t C, size_t kern_minus_1, const void* src, void* dst",
     "supersonic::metal::extract_conv_state(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst)"),
    ("supersonic_qwen35_hip_prepare_conv_input_tail", "prepare_conv_input_tail", 346,
     "int dtype, size_t device_ordinal, size_t S, size_t C, size_t pad, const void* src, const void* old_tail, void* conv_input, void* new_tail",
     "supersonic::metal::prepare_conv_input_tail(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, old_tail, conv_input, new_tail)"),
    ("supersonic_qwen35_hip_sigmoid_mul", "sigmoid_mul", 350,
     "int dtype, size_t device_ordinal, size_t total_elems, const void* data, const void* gate, void* out",
     "supersonic::metal::sigmoid_mul(dtype, static_cast<int>(total_elems), data, gate, out)"),
    ("supersonic_qwen35_hip_cast_transpose_gate_bf16", "cast_transpose_gate_hsd_to_shd_bf16", 351,
     "size_t device_ordinal, size_t S, size_t H, size_t D, const void* attn_hsd, const void* gate_shd, void* out_shd",
     "supersonic::metal::cast_transpose_gate_hsd_to_shd_bf16(static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), attn_hsd, gate_shd, out_shd)"),
    ("supersonic_qwen35_hip_compute_beta_g", "compute_beta_g", 360,
     "int dtype, size_t device_ordinal, size_t seq_len, size_t nv, const void* B, const void* A, const void* dt_bias, const void* a_log_exp, void* beta, void* g",
     "supersonic::metal::compute_beta_g(dtype, static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g)"),
    ("supersonic_qwen35_hip_compute_beta_g_ba_bf16", "compute_beta_g_ba_bf16", 361,
     "size_t device_ordinal, size_t seq_len, size_t nv, const void* BA, const void* dt_bias, const void* a_log_exp, void* beta, void* g",
     "supersonic::metal::compute_beta_g_ba_bf16(static_cast<int>(seq_len), static_cast<int>(nv), BA, dt_bias, a_log_exp, beta, g)"),
    ("supersonic_qwen35_hip_project_ba_compute_beta_g_bf16", "project_ba_compute_beta_g_bf16", 362,
     "size_t device_ordinal, size_t seq_len, size_t hidden_dim, size_t nv, const void* hidden, const void* ba_weight, const void* dt_bias, const void* a_log_exp, void* beta, void* g",
     "supersonic::metal::project_ba_compute_beta_g_bf16(static_cast<int>(seq_len), static_cast<int>(hidden_dim), static_cast<int>(nv), hidden, ba_weight, dt_bias, a_log_exp, beta, g)"),
    ("supersonic_qwen35_hip_split_qgate", "split_qgate", 370,
     "int dtype, size_t device_ordinal, size_t S, size_t num_heads, size_t head_dim, const void* src, void* query_out, void* gate_out",
     "supersonic::metal::split_qgate(dtype, static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out)"),
    ("supersonic_qwen35_hip_split_qgate_norm_bf16", "split_qgate_norm_bf16", 371,
     "size_t device_ordinal, size_t S, size_t num_heads, size_t head_dim, float eps, const void* src, const void* norm_w, void* query_out, void* gate_out",
     "supersonic::metal::split_qgate_norm_bf16(static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), eps, src, norm_w, query_out, gate_out)"),
    ("supersonic_qwen35_hip_split_qkv", "split_qkv", 380,
     "int dtype, size_t device_ordinal, size_t S, size_t key_dim, size_t val_dim, const void* src, void* Q, void* K, void* V",
     "supersonic::metal::split_qkv(dtype, static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V)"),
    ("supersonic_qwen35_hip_split_qkv_bf16_to_f32", "split_qkv_bf16_to_f32", 381,
     "size_t device_ordinal, size_t S, size_t key_dim, size_t val_dim, const void* src, void* Q, void* K, void* V",
     "supersonic::metal::split_qkv_bf16_to_f32(static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V)"),
    ("supersonic_qwen35_hip_split_kv_bf16", "split_kv_bf16", 382,
     "size_t device_ordinal, size_t S, size_t kv_dim, const void* src, void* K, void* V",
     "supersonic::metal::split_kv_bf16(static_cast<int>(S), static_cast<int>(kv_dim), src, K, V)"),
    ("supersonic_qwen35_hip_split_norm_transpose_qkv_bf16", "split_norm_transpose_qkv_bf16", 383,
     "size_t device_ordinal, size_t S, size_t nk, size_t nv, size_t khd, size_t vhd, float q_scale, float eps, const void* src, void* Q, void* K, void* V",
     "supersonic::metal::split_norm_transpose_qkv_bf16(static_cast<int>(S), static_cast<int>(nk), static_cast<int>(nv), static_cast<int>(khd), static_cast<int>(vhd), q_scale, eps, src, Q, K, V)"),
    ("supersonic_qwen35_hip_rms_norm_gated_sfirst_bf16", "rms_norm_gated_sfirst_bf16", 384,
     "size_t device_ordinal, size_t S, size_t nv, size_t vhd, float eps, const void* hidden_hsd, const void* gate_sfirst, const void* weight, void* out_sfirst",
     "supersonic::metal::rms_norm_gated_sfirst_bf16(static_cast<int>(S), static_cast<int>(nv), static_cast<int>(vhd), eps, hidden_hsd, gate_sfirst, weight, out_sfirst)"),
    ("supersonic_qwen35_hip_split_qkvz_bf16", "split_qkvz_bf16", 385,
     "size_t device_ordinal, size_t S, size_t qkv_dim, size_t z_dim, const void* src, void* QKV, void* Z",
     "supersonic::metal::split_qkvz_bf16(static_cast<int>(S), static_cast<int>(qkv_dim), static_cast<int>(z_dim), src, QKV, Z)"),
    ("supersonic_qwen35_hip_repeat_interleave_heads", "repeat_interleave_heads", 390,
     "int dtype, size_t device_ordinal, size_t S, size_t n_heads, size_t head_dim, size_t repeats, const void* src, void* dst",
     "supersonic::metal::repeat_interleave_heads(dtype, static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim), static_cast<int>(repeats), src, dst)"),
    ("supersonic_qwen35_hip_repeat_interleave_transpose_hsd", "repeat_interleave_transpose_hsd", 395,
     "int dtype, size_t device_ordinal, size_t S, size_t n_heads, size_t head_dim, size_t repeats, const void* src, void* dst",
     "supersonic::metal::repeat_interleave_transpose_hsd(dtype, static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim), static_cast<int>(repeats), src, dst)"),
    ("supersonic_qwen35_hip_swiglu_mul", "swiglu_mul", 410,
     "int dtype, size_t device_ordinal, size_t total_elems, const void* gate, const void* up, void* out",
     "supersonic::metal::swiglu_mul(dtype, static_cast<int>(total_elems), gate, up, out)"),
    ("supersonic_qwen35_hip_swiglu_mul_split", "swiglu_mul_split", 411,
     "int dtype, size_t device_ordinal, size_t rows, size_t cols, const void* gate_up, void* out",
     "supersonic::metal::swiglu_mul_split(dtype, static_cast<int>(rows), static_cast<int>(cols), gate_up, out)"),
    ("supersonic_qwen35_hip_l2norm", "l2norm", 92,
     "int dtype, size_t device_ordinal, size_t n_rows, size_t n_cols, float eps, const void* xs, void* out",
     "supersonic::metal::l2norm(dtype, static_cast<int>(n_rows), static_cast<int>(n_cols), eps, xs, out)"),
    ("supersonic_qwen35_hip_mul_scalar", "mul_scalar", 147,
     "int dtype, size_t device_ordinal, size_t total_elems, float scalar, const void* xs, void* out",
     "supersonic::metal::mul_scalar(dtype, static_cast<int>(total_elems), scalar, xs, out)"),
    ("supersonic_qwen35_hip_rms_norm_gated", "rms_norm_gated", 420,
     "int dtype, size_t device_ordinal, size_t n_rows, size_t n_cols, float eps, const void* hidden, const void* gate, const void* weight, void* out",
     "supersonic::metal::rms_norm_gated(dtype, static_cast<int>(n_rows), static_cast<int>(n_cols), eps, hidden, gate, weight, out)"),
    ("supersonic_qwen35_hip_fill_conv_tail", "fill_conv_tail", 430,
     "int dtype, size_t device_ordinal, size_t qkv_dim, size_t pad, size_t total_len, const void* tail, void* conv_input",
     "supersonic::metal::fill_conv_tail(dtype, static_cast<int>(qkv_dim), static_cast<int>(pad), static_cast<int>(total_len), tail, conv_input)"),
    ("supersonic_qwen35_hip_linear_prefill_conv_pack", "linear_prefill_conv_pack", 62,
     "int dtype, size_t device_ordinal, size_t batch_size, size_t conv_dim, size_t total_len, size_t seq_len, size_t kernel_size, const void* mixed_qkv, const void* weights, void* out",
     "supersonic::metal::linear_prefill_conv_pack(dtype, static_cast<int>(batch_size), static_cast<int>(conv_dim), static_cast<int>(total_len), static_cast<int>(seq_len), static_cast<int>(kernel_size), mixed_qkv, weights, out)"),
    ("supersonic_qwen35_hip_delta_recurrent_prefill", "delta_recurrent_prefill", 63,
     "int dtype, size_t device_ordinal, size_t batch_heads, size_t seq_len, size_t k_head_dim, size_t v_head_dim, const void* initial_state, const void* query, const void* key, const void* value, const void* beta, const void* g, void* out",
     "supersonic::metal::delta_recurrent_prefill(dtype, static_cast<int>(batch_heads), static_cast<int>(seq_len), static_cast<int>(k_head_dim), static_cast<int>(v_head_dim), initial_state, query, key, value, beta, g, out)"),
    ("supersonic_qwen35_hip_full_attention_prefill", "full_attention_prefill", 64,
     "int dtype, size_t device_ordinal, size_t batch_size, size_t q_heads, size_t kv_heads, size_t q_len, size_t kv_len, size_t head_dim, size_t num_kv_groups, float scale, size_t seqlen_offset, const void* query, const void* key, const void* value, void* out",
     "supersonic::metal::full_attention_prefill(dtype, static_cast<int>(batch_size), static_cast<int>(q_heads), static_cast<int>(kv_heads), static_cast<int>(q_len), static_cast<int>(kv_len), static_cast<int>(head_dim), static_cast<int>(num_kv_groups), scale, static_cast<int>(seqlen_offset), query, key, value, out)"),
    ("supersonic_qwen35_hip_batched_matmul", "batched_matmul", 144,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs, void* out",
     "supersonic::metal::batched_matmul(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs, out)"),
    ("supersonic_qwen35_hip_fused_rms_norm_linear", "fused_rms_norm_linear", 145,
     "int dtype, size_t device_ordinal, size_t hidden_dim, size_t out_dim, float eps, int add_unit_offset, const void* hidden, const void* norm_weight, const void* proj_weight, void* out",
     "supersonic::metal::fused_rms_norm_linear(dtype, static_cast<int>(hidden_dim), static_cast<int>(out_dim), eps, add_unit_offset, hidden, norm_weight, proj_weight, out)"),
    ("supersonic_qwen35_4b_hip_matmul_int4_dequant", "matmul_int4_dequant", 290,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_int4, const void* scale, const void* zero, const void* awq_inv_scale, int group_size, int quant_type, void* out",
     "supersonic::metal::matmul_int4_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out)"),
    ("supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add", "matmul_int4_dequant_residual_add", 291,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_int4, const void* scale, const void* zero, const void* awq_inv_scale, int group_size, int quant_type, const void* residual, void* out",
     "supersonic::metal::matmul_int4_dequant_residual_add(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, residual, out)"),
    ("supersonic_qwen35_4b_hip_matmul_fp8_dequant", "matmul_fp8_dequant", 292,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_fp8, const void* scale, int block_size, void* out",
     "supersonic::metal::matmul_fp8_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_fp8, scale, block_size, out)"),
    ("supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant", "matmul_ggml_pair_dequant", 293,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n_each, int k, const void* lhs, const void* rhs_first, const void* rhs_second, int quant_type, void* out",
     "supersonic::metal::matmul_ggml_pair_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n_each, k, lhs, rhs_first, rhs_second, quant_type, out)"),
    ("supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu", "matmul_ggml_pair_swiglu", 294,
     "int dtype, size_t device_ordinal, size_t batch_elems, int m, int n_each, int k, const void* lhs, const void* rhs_gate, const void* rhs_up, int quant_type, void* out",
     "supersonic::metal::matmul_ggml_pair_swiglu(dtype, static_cast<std::uint32_t>(batch_elems), m, n_each, k, lhs, rhs_gate, rhs_up, quant_type, out)"),
]


def write_dispatch_inc() -> None:
    DISPATCH_INC.write_text(
        (ROOT / "kernels/metal/metal_dispatch_ops_body.txt").read_text()
        if (ROOT / "kernels/metal/metal_dispatch_ops_body.txt").exists()
        else ""
    )


def patch_dispatch_hpp() -> None:
    text = DISPATCH_HPP.read_text()
    marker = "}  // namespace supersonic::metal"
    if "bool cast(" in text:
        return
    text = text.replace(marker, DISPATCH_DECLS + "\n" + marker)
    DISPATCH_HPP.write_text(text)


def patch_dispatch_mm() -> None:
    text = DISPATCH_MM.read_text()
    if '#include "metal_dispatch_ops.inc"' in text:
        return
    text = text.replace(
        "}  // namespace supersonic::metal\n",
        '#include "metal_dispatch_ops.inc"\n\n}  // namespace supersonic::metal\n',
    )
    # Add row counter helper in anonymous namespace before supersonic::metal
    if "full_attention_row_counter" not in text:
        insert = """
id<MTLBuffer> full_attention_row_counter() {
    static id<MTLBuffer> counter = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        counter = [metal_device() newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        if (counter != nil) {
            *static_cast<uint32_t*>(counter.contents) = 0u;
        }
    });
    return counter;
}

"""
        text = text.replace("}  // namespace\n\nnamespace supersonic::metal {", "}  // namespace\n" + insert + "\nnamespace supersonic::metal {")
    DISPATCH_MM.write_text(text)


def write_bridge_mm() -> None:
    lines = [
        '#include "metal_dispatch.hpp"',
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace {",
        "int validate_device_ordinal(std::size_t device_ordinal) { return device_ordinal == 0 ? 0 : 1; }",
        "bool preflight(std::size_t device_ordinal) {",
        "    if (validate_device_ordinal(device_ordinal) != 0) return false;",
        "    return supersonic::metal::init_prefill_library();",
        "}",
        "}  // namespace",
        "",
        'extern "C" int supersonic_prefill_encode_bridge_status(int project_status, int native_status) {',
        "    return native_status == 0 ? project_status : project_status * 1000 + native_status;",
        "}",
        "",
        'extern "C" int supersonic_qwen35_4b_bf16_matmul_bridge_status(int project_status, int native_status) {',
        "    return native_status == 0 ? project_status : project_status * 1000 + native_status;",
        "}",
        "",
        'extern "C" int supersonic_qwen35_4b_hip_device_supports_wmma_i8(size_t device_ordinal, int* out_supported) {',
        "    if (validate_device_ordinal(device_ordinal) != 0) return 1;",
        "    if (out_supported == nullptr) return 2;",
        "    *out_supported = 0;",
        "    return 0;",
        "}",
        "",
    ]
    for hip, _metal, err, sig, call in BRIDGES:
        if not call:
            continue
        lines += [
            f'extern "C" int {hip}({sig}) {{',
            "    if (!preflight(device_ordinal)) {",
            "        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;",
            "    }",
            f"    if (!{call}) {{",
            f"        return {err};",
            "    }",
            "    return 0;",
            "}",
            "",
        ]
    BRIDGE_MM.write_text("\n".join(lines))


def patch_stubs() -> None:
    text = STUBS.read_text()
    for hip, metal, _err, _sig, _call in BRIDGES:
        if not metal:
            continue
        text = text.replace(f"SUPERSONIC_STUB({hip})\n", "")
    STUBS.write_text(text)


def patch_manifest() -> None:
    text = MANIFEST.read_text()
    if "prefill_bridge_rest.mm" not in text:
        text = text.replace(
            '"kernels/metal/prefill_bridge.mm",\n',
            '"kernels/metal/prefill_bridge.mm",\n  "kernels/metal/prefill_bridge_rest.mm",\n',
        )
        MANIFEST.write_text(text)


def main() -> None:
    patch_dispatch_hpp()
    patch_dispatch_mm()
    write_bridge_mm()
    patch_stubs()
    patch_manifest()
    print("Patched headers/bridge/stubs. Ensure metal_dispatch_ops.inc exists.")


if __name__ == "__main__":
    main()
