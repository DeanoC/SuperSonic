#include "metal_dispatch.hpp"

#include <cstddef>
#include <cstdint>

namespace {
int validate_device_ordinal(std::size_t device_ordinal) { return device_ordinal == 0 ? 0 : 1; }
bool preflight(std::size_t device_ordinal) {
    if (validate_device_ordinal(device_ordinal) != 0) return false;
    return supersonic::metal::init_prefill_library();
}
}  // namespace

extern "C" int supersonic_prefill_encode_bridge_status(int project_status, int native_status) {
    return native_status == 0 ? project_status : project_status * 1000 + native_status;
}

extern "C" int supersonic_qwen35_4b_bf16_matmul_bridge_status(int project_status, int native_status) {
    return native_status == 0 ? project_status : project_status * 1000 + native_status;
}

extern "C" int supersonic_qwen35_4b_hip_device_supports_wmma_i8(size_t device_ordinal, int* out_supported) {
    if (validate_device_ordinal(device_ordinal) != 0) return 1;
    if (out_supported == nullptr) return 2;
    *out_supported = 0;
    return 0;
}

extern "C" int supersonic_qwen35_hip_cast(int input_dtype, int output_dtype, size_t device_ordinal, size_t total_elems, const void* xs, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::cast(input_dtype, output_dtype, static_cast<int>(total_elems), xs, out)) {
        return 280;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_element_add(int dtype, size_t device_ordinal, size_t total_elems, const void* lhs, const void* rhs, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::element_add(dtype, static_cast<int>(total_elems), lhs, rhs, out)) {
        return 300;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_argmax_bf16_rows(size_t device_ordinal, size_t rows, size_t cols, const void* logits, void* out_index) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::argmax_bf16_rows(static_cast<int>(rows), static_cast<int>(cols), logits, out_index)) {
        return 400;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_argmax_f32_as_bf16_rows(size_t device_ordinal, size_t rows, size_t cols, const void* logits, void* out_index) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::argmax_f32_as_bf16_rows(static_cast<int>(rows), static_cast<int>(cols), logits, out_index)) {
        return 401;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_apply_rope_prefill(int dtype, size_t device_ordinal, size_t seq_len, size_t num_heads, size_t head_dim, size_t half_rot, const void* cos_table, const void* sin_table, void* data) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::apply_rope_prefill(dtype, static_cast<int>(seq_len), static_cast<int>(num_heads), static_cast<int>(head_dim), static_cast<int>(half_rot), cos_table, sin_table, data)) {
        return 310;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_transpose_shd_hsd(int dtype, size_t device_ordinal, size_t S, size_t H, size_t D, const void* src, void* dst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::transpose_shd_hsd(dtype, static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src, dst)) {
        return 320;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_transpose_shd_hsd_pair(int dtype, size_t device_ordinal, size_t S, size_t H, size_t D, const void* src_a, const void* src_b, void* dst_a, void* dst_b) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::transpose_shd_hsd_pair(dtype, static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), src_a, src_b, dst_a, dst_b)) {
        return 325;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_transpose_shd_to_cache_bf16(size_t device_ordinal, size_t S, size_t H, size_t D, size_t cache_len, size_t dst_pos, const void* src, void* cache) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::transpose_shd_to_cache_bf16(static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), static_cast<int>(cache_len), static_cast<int>(dst_pos), src, cache)) {
        return 321;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_transpose_pad_conv(int dtype, size_t device_ordinal, size_t S, size_t C, size_t pad, const void* src, void* dst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::transpose_pad_conv(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, dst)) {
        return 329;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_extract_conv_state(int dtype, size_t device_ordinal, size_t S, size_t C, size_t kern_minus_1, const void* src, void* dst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::extract_conv_state(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(kern_minus_1), src, dst)) {
        return 340;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_prepare_conv_input_tail(int dtype, size_t device_ordinal, size_t S, size_t C, size_t pad, const void* src, const void* old_tail, void* conv_input, void* new_tail) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::prepare_conv_input_tail(dtype, static_cast<int>(S), static_cast<int>(C), static_cast<int>(pad), src, old_tail, conv_input, new_tail)) {
        return 346;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_sigmoid_mul(int dtype, size_t device_ordinal, size_t total_elems, const void* data, const void* gate, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::sigmoid_mul(dtype, static_cast<int>(total_elems), data, gate, out)) {
        return 350;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_cast_transpose_gate_bf16(size_t device_ordinal, size_t S, size_t H, size_t D, const void* attn_hsd, const void* gate_shd, void* out_shd) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::cast_transpose_gate_hsd_to_shd_bf16(static_cast<int>(S), static_cast<int>(H), static_cast<int>(D), attn_hsd, gate_shd, out_shd)) {
        return 351;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_compute_beta_g(int dtype, size_t device_ordinal, size_t seq_len, size_t nv, const void* B, const void* A, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::compute_beta_g(dtype, static_cast<int>(seq_len), static_cast<int>(nv), B, A, dt_bias, a_log_exp, beta, g)) {
        return 360;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_compute_beta_g_ba_bf16(size_t device_ordinal, size_t seq_len, size_t nv, const void* BA, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::compute_beta_g_ba_bf16(static_cast<int>(seq_len), static_cast<int>(nv), BA, dt_bias, a_log_exp, beta, g)) {
        return 361;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_project_ba_compute_beta_g_bf16(size_t device_ordinal, size_t seq_len, size_t hidden_dim, size_t nv, const void* hidden, const void* ba_weight, const void* dt_bias, const void* a_log_exp, void* beta, void* g) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::project_ba_compute_beta_g_bf16(static_cast<int>(seq_len), static_cast<int>(hidden_dim), static_cast<int>(nv), hidden, ba_weight, dt_bias, a_log_exp, beta, g)) {
        return 362;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_qgate(int dtype, size_t device_ordinal, size_t S, size_t num_heads, size_t head_dim, const void* src, void* query_out, void* gate_out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_qgate(dtype, static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), src, query_out, gate_out)) {
        return 370;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_qgate_norm_bf16(size_t device_ordinal, size_t S, size_t num_heads, size_t head_dim, float eps, const void* src, const void* norm_w, void* query_out, void* gate_out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_qgate_norm_bf16(static_cast<int>(S), static_cast<int>(num_heads), static_cast<int>(head_dim), eps, src, norm_w, query_out, gate_out)) {
        return 371;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_qkv(int dtype, size_t device_ordinal, size_t S, size_t key_dim, size_t val_dim, const void* src, void* Q, void* K, void* V) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_qkv(dtype, static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V)) {
        return 380;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_qkv_bf16_to_f32(size_t device_ordinal, size_t S, size_t key_dim, size_t val_dim, const void* src, void* Q, void* K, void* V) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_qkv_bf16_to_f32(static_cast<int>(S), static_cast<int>(key_dim), static_cast<int>(val_dim), src, Q, K, V)) {
        return 381;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_kv_bf16(size_t device_ordinal, size_t S, size_t kv_dim, const void* src, void* K, void* V) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_kv_bf16(static_cast<int>(S), static_cast<int>(kv_dim), src, K, V)) {
        return 382;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_norm_transpose_qkv_bf16(size_t device_ordinal, size_t S, size_t nk, size_t nv, size_t khd, size_t vhd, float q_scale, float eps, const void* src, void* Q, void* K, void* V) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_norm_transpose_qkv_bf16(static_cast<int>(S), static_cast<int>(nk), static_cast<int>(nv), static_cast<int>(khd), static_cast<int>(vhd), q_scale, eps, src, Q, K, V)) {
        return 383;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_rms_norm_gated_sfirst_bf16(size_t device_ordinal, size_t S, size_t nv, size_t vhd, float eps, const void* hidden_hsd, const void* gate_sfirst, const void* weight, void* out_sfirst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::rms_norm_gated_sfirst_bf16(static_cast<int>(S), static_cast<int>(nv), static_cast<int>(vhd), eps, hidden_hsd, gate_sfirst, weight, out_sfirst)) {
        return 384;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_split_qkvz_bf16(size_t device_ordinal, size_t S, size_t qkv_dim, size_t z_dim, const void* src, void* QKV, void* Z) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::split_qkvz_bf16(static_cast<int>(S), static_cast<int>(qkv_dim), static_cast<int>(z_dim), src, QKV, Z)) {
        return 385;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_repeat_interleave_heads(int dtype, size_t device_ordinal, size_t S, size_t n_heads, size_t head_dim, size_t repeats, const void* src, void* dst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::repeat_interleave_heads(dtype, static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim), static_cast<int>(repeats), src, dst)) {
        return 390;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_repeat_interleave_transpose_hsd(int dtype, size_t device_ordinal, size_t S, size_t n_heads, size_t head_dim, size_t repeats, const void* src, void* dst) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::repeat_interleave_transpose_hsd(dtype, static_cast<int>(S), static_cast<int>(n_heads), static_cast<int>(head_dim), static_cast<int>(repeats), src, dst)) {
        return 395;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_swiglu_mul(int dtype, size_t device_ordinal, size_t total_elems, const void* gate, const void* up, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::swiglu_mul(dtype, static_cast<int>(total_elems), gate, up, out)) {
        return 410;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_swiglu_mul_split(int dtype, size_t device_ordinal, size_t rows, size_t cols, const void* gate_up, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::swiglu_mul_split(dtype, static_cast<int>(rows), static_cast<int>(cols), gate_up, out)) {
        return 411;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_l2norm(int dtype, size_t device_ordinal, size_t n_rows, size_t n_cols, float eps, const void* xs, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::l2norm(dtype, static_cast<int>(n_rows), static_cast<int>(n_cols), eps, xs, out)) {
        return 92;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_mul_scalar(int dtype, size_t device_ordinal, size_t total_elems, float scalar, const void* xs, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::mul_scalar(dtype, static_cast<int>(total_elems), scalar, xs, out)) {
        return 147;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_rms_norm_gated(int dtype, size_t device_ordinal, size_t n_rows, size_t n_cols, float eps, const void* hidden, const void* gate, const void* weight, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::rms_norm_gated(dtype, static_cast<int>(n_rows), static_cast<int>(n_cols), eps, hidden, gate, weight, out)) {
        return 420;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_fill_conv_tail(int dtype, size_t device_ordinal, size_t qkv_dim, size_t pad, size_t total_len, const void* tail, void* conv_input) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::fill_conv_tail(dtype, static_cast<int>(qkv_dim), static_cast<int>(pad), static_cast<int>(total_len), tail, conv_input)) {
        return 430;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_linear_prefill_conv_pack(int dtype, size_t device_ordinal, size_t batch_size, size_t conv_dim, size_t total_len, size_t seq_len, size_t kernel_size, const void* mixed_qkv, const void* weights, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::linear_prefill_conv_pack(dtype, static_cast<int>(batch_size), static_cast<int>(conv_dim), static_cast<int>(total_len), static_cast<int>(seq_len), static_cast<int>(kernel_size), mixed_qkv, weights, out)) {
        return 62;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_delta_recurrent_prefill(int dtype, size_t device_ordinal, size_t batch_heads, size_t seq_len, size_t k_head_dim, size_t v_head_dim, const void* initial_state, const void* query, const void* key, const void* value, const void* beta, const void* g, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::delta_recurrent_prefill(dtype, static_cast<int>(batch_heads), static_cast<int>(seq_len), static_cast<int>(k_head_dim), static_cast<int>(v_head_dim), initial_state, query, key, value, beta, g, out)) {
        return 63;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_full_attention_prefill(int dtype, size_t device_ordinal, size_t batch_size, size_t q_heads, size_t kv_heads, size_t q_len, size_t kv_len, size_t head_dim, size_t num_kv_groups, float scale, size_t seqlen_offset, const void* query, const void* key, const void* value, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::full_attention_prefill(dtype, static_cast<int>(batch_size), static_cast<int>(q_heads), static_cast<int>(kv_heads), static_cast<int>(q_len), static_cast<int>(kv_len), static_cast<int>(head_dim), static_cast<int>(num_kv_groups), scale, static_cast<int>(seqlen_offset), query, key, value, out)) {
        return 64;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_batched_matmul(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::batched_matmul(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs, out)) {
        return 144;
    }
    return 0;
}

extern "C" int supersonic_qwen35_hip_fused_rms_norm_linear(int dtype, size_t device_ordinal, size_t hidden_dim, size_t out_dim, float eps, int add_unit_offset, const void* hidden, const void* norm_weight, const void* proj_weight, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::fused_rms_norm_linear(dtype, static_cast<int>(hidden_dim), static_cast<int>(out_dim), eps, add_unit_offset, hidden, norm_weight, proj_weight, out)) {
        return 145;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_int4, const void* scale, const void* zero, const void* awq_inv_scale, int group_size, int quant_type, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::matmul_int4_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, 0.0f, 0, out)) {
        return 290;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_int4, const void* scale, const void* zero, const void* awq_inv_scale, int group_size, int quant_type, const void* residual, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::matmul_int4_dequant_residual_add(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, 0.0f, 0, residual, out)) {
        return 291;
    }
    return 0;
}

extern "C" const void* supersonic_metal_dummy_buffer();

extern "C" int supersonic_metal_matmul_gqh_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* wire,
    int quant_type,
    float tensor_scale,
    int grid_code,
    void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    const void* dummy = supersonic_metal_dummy_buffer();
    if (dummy == nullptr) {
        return 295;
    }
    if (!supersonic::metal::matmul_int4_dequant(
            dtype,
            static_cast<std::uint32_t>(batch_elems),
            m,
            n,
            k,
            lhs,
            wire,
            dummy,
            dummy,
            nullptr,
            1,
            quant_type,
            tensor_scale,
            grid_code,
            out)) {
        return 295;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_fp8_dequant(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n, int k, const void* lhs, const void* rhs_fp8, const void* scale, int block_size, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::matmul_fp8_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n, k, lhs, rhs_fp8, scale, block_size, out)) {
        return 292;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n_each, int k, const void* lhs, const void* rhs_first, const void* rhs_second, int quant_type, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::matmul_ggml_pair_dequant(dtype, static_cast<std::uint32_t>(batch_elems), m, n_each, k, lhs, rhs_first, rhs_second, quant_type, out)) {
        return 293;
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu(int dtype, size_t device_ordinal, size_t batch_elems, int m, int n_each, int k, const void* lhs, const void* rhs_gate, const void* rhs_up, int quant_type, void* out) {
    if (!preflight(device_ordinal)) {
        return validate_device_ordinal(device_ordinal) != 0 ? 1 : 2;
    }
    if (!supersonic::metal::matmul_ggml_pair_swiglu(dtype, static_cast<std::uint32_t>(batch_elems), m, n_each, k, lhs, rhs_gate, rhs_up, quant_type, out)) {
        return 294;
    }
    return 0;
}
