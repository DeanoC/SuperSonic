use std::ffi::{c_char, c_int, c_void, CString};

use gpu_hal::{GpuBuffer, GpuError, ScalarType};

pub(crate) fn disabled_by_env() -> bool {
    std::env::var_os("SUPERSONIC_METAL_FORCE_HOST_NATIVE").is_some()
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
unsafe extern "C" {
    fn supersonic_metal_batch_begin() -> c_int;
    fn supersonic_metal_batch_flush() -> c_int;
    fn supersonic_metal_batch_set_label(label: *const c_char) -> c_int;
    fn supersonic_metal_batch_end() -> c_int;
    fn supersonic_metal_copy_d2d(
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
        bytes: usize,
    ) -> c_int;
    fn supersonic_metal_embedding_lookup_bf16(
        token_count: usize,
        vocab_size: usize,
        hidden_size: usize,
        embeddings_ptr: *const c_void,
        indexes_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_matmul_rhs_transposed_bf16(
        batch_elems: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_matmul_rhs_transposed_residual_bf16(
        batch_elems: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        residual_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_matmul_rhs_transposed_int4_bf16(
        batch_elems: usize,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
        lhs_ptr: *const c_void,
        rhs_int4_ptr: *const c_void,
        scale_ptr: *const c_void,
        zero_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_matmul_rhs_transposed_f32(
        batch_elems: usize,
        m: usize,
        n: usize,
        k: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_projections_bf16(
        hidden_dim: usize,
        qkv_dim: usize,
        val_dim: usize,
        num_value_heads: usize,
        input_ptr: *const c_void,
        qkv_weight_ptr: *const c_void,
        z_weight_ptr: *const c_void,
        a_weight_ptr: *const c_void,
        b_weight_ptr: *const c_void,
        qkv_out_ptr: *mut c_void,
        z_out_ptr: *mut c_void,
        a_out_ptr: *mut c_void,
        b_out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_prep_decode_apply_bf16_f32(
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_pack_ptr: *const c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        initial_state_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_decode_apply_inplace_bf16(
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_pack_ptr: *const c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        state_ptr: *mut c_void,
        attn_out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_mlp_gate_up_bf16(
        hidden_dim: usize,
        intermediate_dim: usize,
        input_ptr: *const c_void,
        gate_weight_ptr: *const c_void,
        up_weight_ptr: *const c_void,
        gate_out_ptr: *mut c_void,
        up_out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_mlp_gate_up_swiglu_bf16(
        hidden_dim: usize,
        intermediate_dim: usize,
        input_ptr: *const c_void,
        gate_weight_ptr: *const c_void,
        up_weight_ptr: *const c_void,
        mlp_out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_full_attention_gate_bf16(
        total_elems: usize,
        attn_f32_ptr: *const c_void,
        gate_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_mlp_down_residual_bf16(
        hidden_dim: usize,
        intermediate_dim: usize,
        gate_ptr: *const c_void,
        up_ptr: *const c_void,
        down_weight_ptr: *const c_void,
        residual_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_out_residual_f32_bf16(
        hidden_dim: usize,
        num_rows: usize,
        row_dim: usize,
        eps: f32,
        attn_ptr: *const c_void,
        gate_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_proj_ptr: *const c_void,
        residual_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_out_residual_bf16_bf16(
        hidden_dim: usize,
        num_rows: usize,
        row_dim: usize,
        eps: f32,
        attn_ptr: *const c_void,
        gate_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_proj_ptr: *const c_void,
        residual_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_full_projections_bf16(
        hidden_dim: usize,
        q_proj_dim: usize,
        kv_dim: usize,
        input_ptr: *const c_void,
        q_weight_ptr: *const c_void,
        k_weight_ptr: *const c_void,
        v_weight_ptr: *const c_void,
        q_out_ptr: *mut c_void,
        k_out_ptr: *mut c_void,
        v_out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_lm_head_argmax_bf16(
        in_dim: usize,
        vocab_size: usize,
        hidden_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_index_ptr: *mut c_void,
        partial_values_ptr: *mut c_void,
        partial_indices_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_argmax_bf16(
        n: usize,
        logits_ptr: *const c_void,
        out_index_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_full_attention_prefill_bf16_f32(
        q_heads: usize,
        kv_heads: usize,
        q_len: usize,
        kv_len: usize,
        kv_stride: usize,
        head_dim: usize,
        scale: f32,
        seqlen_offset: usize,
        query_ptr: *const c_void,
        key_ptr: *const c_void,
        value_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_full_attention_decode_bf16_f32(
        q_heads: usize,
        kv_heads: usize,
        kv_len: usize,
        kv_stride: usize,
        head_dim: usize,
        scale: f32,
        query_ptr: *const c_void,
        key_ptr: *const c_void,
        value_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_rows_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        add_unit_offset: bool,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_rope_rows_bf16(
        n_rows: usize,
        n_cols: usize,
        rotary_dim: usize,
        eps: f32,
        pos_offset: usize,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        cos_ptr: *const c_void,
        sin_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_rows_f32_weight_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        add_unit_offset: bool,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_rows_f32_weight_f32(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        add_unit_offset: bool,
        input_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_gated_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        hidden_ptr: *const c_void,
        gate_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_gated_f32_weight_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        hidden_ptr: *const c_void,
        gate_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_rms_norm_gated_f32_weight_f32(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        hidden_ptr: *const c_void,
        gate_ptr: *const c_void,
        weight_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_l2norm_f32(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_l2norm_bf16(
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_linear_prefill_conv_pack_bf16(
        conv_dim: usize,
        total_len: usize,
        seq_len: usize,
        kernel_size: usize,
        mixed_ptr: *const c_void,
        weights_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_element_add_bf16(
        total_elems: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_element_add_f32(
        total_elems: usize,
        lhs_ptr: *const c_void,
        rhs_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_sigmoid_mul_bf16(
        total_elems: usize,
        data_ptr: *const c_void,
        gate_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_sigmoid_mul_f32(
        total_elems: usize,
        data_ptr: *const c_void,
        gate_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_swiglu_mul_bf16(
        total_elems: usize,
        gate_ptr: *const c_void,
        up_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_swiglu_mul_f32(
        total_elems: usize,
        gate_ptr: *const c_void,
        up_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_bf16_to_bf16(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_f32_to_f32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_u32_to_u32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_bf16_to_f32(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_cast_f32_to_bf16(
        total_elems: usize,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_mul_scalar_bf16(
        total_elems: usize,
        scalar: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_mul_scalar_f32(
        total_elems: usize,
        scalar: f32,
        input_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_shd_hsd_bf16(
        s: usize,
        h: usize,
        d: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_shd_hsd_f32(
        s: usize,
        h: usize,
        d: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_apply_rope_prefill_bf16(
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_offset: usize,
        cos_ptr: *const c_void,
        sin_ptr: *const c_void,
        data_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_apply_rope_prefill_f32(
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_offset: usize,
        cos_ptr: *const c_void,
        sin_ptr: *const c_void,
        data_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_pad_conv_bf16(
        s: usize,
        c: usize,
        pad: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_transpose_pad_conv_f32(
        s: usize,
        c: usize,
        pad: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_extract_conv_state_bf16(
        s: usize,
        c: usize,
        kern_minus_1: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_extract_conv_state_f32(
        s: usize,
        c: usize,
        kern_minus_1: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qkv_bf16(
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src_ptr: *const c_void,
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qkv_f32(
        s: usize,
        key_dim: usize,
        val_dim: usize,
        src_ptr: *const c_void,
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qgate_bf16(
        s: usize,
        num_heads: usize,
        head_dim: usize,
        src_ptr: *const c_void,
        query_ptr: *mut c_void,
        gate_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_split_qgate_f32(
        s: usize,
        num_heads: usize,
        head_dim: usize,
        src_ptr: *const c_void,
        query_ptr: *mut c_void,
        gate_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_repeat_interleave_heads_bf16(
        s: usize,
        n_heads: usize,
        head_dim: usize,
        repeats: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_repeat_interleave_heads_f32(
        s: usize,
        n_heads: usize,
        head_dim: usize,
        repeats: usize,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_compute_beta_g_f32(
        seq_len: usize,
        nv: usize,
        b_ptr: *const c_void,
        a_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        beta_ptr: *mut c_void,
        g_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_delta_recurrent_prefill_f32(
        batch_heads: usize,
        seq_len: usize,
        k_head_dim: usize,
        v_head_dim: usize,
        initial_state_ptr: *const c_void,
        query_ptr: *const c_void,
        key_ptr: *const c_void,
        value_ptr: *const c_void,
        beta_ptr: *const c_void,
        g_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_linear_decode_apply_parts_f32(
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        q_scaled_ptr: *const c_void,
        k_normed_ptr: *const c_void,
        v_linear_ptr: *const c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        initial_state_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_qwen_linear_prep_bf16_f32(
        key_dim: usize,
        val_dim: usize,
        num_key_heads: usize,
        key_head_dim: usize,
        conv_pack_ptr: *const c_void,
        q_bf16_ptr: *mut c_void,
        k_bf16_ptr: *mut c_void,
        v_bf16_ptr: *mut c_void,
        q_f32_ptr: *mut c_void,
        k_f32_ptr: *mut c_void,
        v_f32_ptr: *mut c_void,
        q_normed_ptr: *mut c_void,
        q_scaled_ptr: *mut c_void,
        k_normed_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_conv_state_update_bf16(
        channels: usize,
        state_len: usize,
        qkv_ptr: *const c_void,
        state_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_linear_conv_value_decay_bf16(
        conv_dim: usize,
        state_len: usize,
        kernel_size: usize,
        num_heads: usize,
        mixed_qkv_ptr: *const c_void,
        prev_state_ptr: *const c_void,
        weights_ptr: *const c_void,
        a_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
    fn supersonic_metal_linear_conv_value_decay_update_bf16(
        conv_dim: usize,
        state_len: usize,
        kernel_size: usize,
        num_heads: usize,
        mixed_qkv_ptr: *const c_void,
        state_ptr: *mut c_void,
        weights_ptr: *const c_void,
        a_ptr: *const c_void,
        dt_bias_ptr: *const c_void,
        a_log_exp_ptr: *const c_void,
        out_ptr: *mut c_void,
    ) -> c_int;
}

pub(crate) struct MetalBatchGuard {
    active: bool,
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
impl MetalBatchGuard {
    pub(crate) fn begin() -> Result<Self, GpuError> {
        if std::env::var_os("SUPERSONIC_METAL_DISABLE_BATCH").is_some() {
            return Ok(Self { active: false });
        }
        let status = unsafe { supersonic_metal_batch_begin() };
        if status != 0 {
            return Err(GpuError::Metal(format!(
                "metal native batch begin failed with status {status}"
            )));
        }
        Ok(Self { active: true })
    }

    pub(crate) fn finish(mut self) -> Result<(), GpuError> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let status = unsafe { supersonic_metal_batch_end() };
        if status != 0 {
            return Err(GpuError::Metal(format!(
                "metal native batch end failed with status {status}"
            )));
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
impl Drop for MetalBatchGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = unsafe { supersonic_metal_batch_end() };
            self.active = false;
        }
    }
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn flush_batch() -> Result<(), GpuError> {
    let status = unsafe { supersonic_metal_batch_flush() };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native batch flush failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn set_batch_label(label: &str) -> Result<(), GpuError> {
    let label = CString::new(label)
        .map_err(|_| GpuError::Metal("metal native batch label contains NUL byte".to_string()))?;
    let status = unsafe { supersonic_metal_batch_set_label(label.as_ptr()) };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native batch set label failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn copy_d2d(src: *const c_void, dst: *mut c_void, bytes: usize) -> Result<(), GpuError> {
    if src.is_null() || dst.is_null() || bytes == 0 {
        return Err(GpuError::InvalidArg(
            "metal native copy_d2d requires non-null pointers and non-zero bytes".into(),
        ));
    }
    let status = unsafe { supersonic_metal_copy_d2d(src, dst, bytes) };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native copy_d2d failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn lm_head_argmax_bf16(
    hidden: &GpuBuffer,
    weight: &GpuBuffer,
    out_index: &mut GpuBuffer,
    in_dim: usize,
    vocab_size: usize,
) -> Result<(), GpuError> {
    lm_head_argmax_bf16_with_partials(hidden, weight, out_index, None, None, in_dim, vocab_size)
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn lm_head_argmax_bf16_with_partials(
    hidden: &GpuBuffer,
    weight: &GpuBuffer,
    out_index: &mut GpuBuffer,
    partial_values: Option<&mut GpuBuffer>,
    partial_indices: Option<&mut GpuBuffer>,
    in_dim: usize,
    vocab_size: usize,
) -> Result<(), GpuError> {
    if hidden.dtype() != ScalarType::BF16 || weight.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "metal native lm_head_argmax_bf16 expects BF16 hidden/weight, got {:?}/{:?}",
            hidden.dtype(),
            weight.dtype()
        )));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() != 1 {
        return Err(GpuError::InvalidArg(
            "metal native lm_head_argmax_bf16 requires a U32[1] output buffer".into(),
        ));
    }
    if in_dim > u32::MAX as usize || vocab_size > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native lm_head_argmax_bf16 dimensions exceed u32: in_dim={in_dim}, vocab_size={vocab_size}"
        )));
    }
    let partial_count = vocab_size.div_ceil(256);
    if let Some(buf) = partial_values.as_ref() {
        if buf.dtype() != ScalarType::F32 || buf.elem_count() < partial_count {
            return Err(GpuError::InvalidArg(format!(
                "metal native lm_head_argmax_bf16 partial_values must be F32[{partial_count}], got {:?}[{}]",
                buf.dtype(),
                buf.elem_count()
            )));
        }
    }
    if let Some(buf) = partial_indices.as_ref() {
        if buf.dtype() != ScalarType::U32 || buf.elem_count() < partial_count {
            return Err(GpuError::InvalidArg(format!(
                "metal native lm_head_argmax_bf16 partial_indices must be U32[{partial_count}], got {:?}[{}]",
                buf.dtype(),
                buf.elem_count()
            )));
        }
    }
    let partial_values_ptr = partial_values
        .map(|buf| buf.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());
    let partial_indices_ptr = partial_indices
        .map(|buf| buf.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status = unsafe {
        supersonic_metal_lm_head_argmax_bf16(
            in_dim,
            vocab_size,
            hidden.as_ptr(),
            weight.as_ptr(),
            out_index.as_mut_ptr(),
            partial_values_ptr,
            partial_indices_ptr,
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native lm_head_argmax_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn argmax_bf16(
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    if logits.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "metal native argmax_bf16 expects BF16 logits, got {:?}",
            logits.dtype()
        )));
    }
    if out_index.dtype() != ScalarType::U32 || out_index.elem_count() != 1 {
        return Err(GpuError::InvalidArg(
            "metal native argmax_bf16 requires a U32[1] output buffer".into(),
        ));
    }
    if n > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native argmax_bf16 n exceeds u32: n={n}"
        )));
    }
    let status =
        unsafe { supersonic_metal_argmax_bf16(n, logits.as_ptr(), out_index.as_mut_ptr()) };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native argmax_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn embedding_lookup_bf16(
    token_count: usize,
    vocab_size: usize,
    hidden_size: usize,
    embeddings: &GpuBuffer,
    indexes: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if token_count > u32::MAX as usize
        || vocab_size > u32::MAX as usize
        || hidden_size > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native embedding_lookup_bf16 dimensions exceed u32: token_count={token_count}, vocab_size={vocab_size}, hidden_size={hidden_size}"
        )));
    }
    if embeddings.dtype() != ScalarType::BF16
        || indexes.dtype() != ScalarType::U32
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native embedding_lookup_bf16 expects BF16/U32/BF16 buffers, got {:?}/{:?}/{:?}",
            embeddings.dtype(),
            indexes.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_embedding_lookup_bf16(
            token_count,
            vocab_size,
            hidden_size,
            embeddings.as_ptr(),
            indexes.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native embedding_lookup_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn matmul_rhs_transposed_bf16(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16
        || rhs.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_matmul_rhs_transposed_bf16(
            batch_elems,
            m,
            n,
            k,
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native matmul_rhs_transposed_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn matmul_rhs_transposed_residual_bf16(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16
        || rhs.dtype() != ScalarType::BF16
        || residual.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_residual_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            residual.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_matmul_rhs_transposed_residual_bf16(
            batch_elems,
            m,
            n,
            k,
            lhs.as_ptr(),
            rhs.as_ptr(),
            residual.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native matmul_rhs_transposed_residual_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_rhs_transposed_int4_bf16(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    lhs: &GpuBuffer,
    rhs_int4: &GpuBuffer,
    scale: &GpuBuffer,
    zero: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16
        || scale.dtype() != ScalarType::BF16
        || zero.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_int4_bf16 expects BF16 lhs/scale/zero/out, got {:?}/{:?}/{:?}/{:?}",
            lhs.dtype(),
            scale.dtype(),
            zero.dtype(),
            out.dtype(),
        )));
    }
    if rhs_int4.dtype() != ScalarType::U8 {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_int4_bf16 expects U8 rhs_int4, got {:?}",
            rhs_int4.dtype(),
        )));
    }
    let status = unsafe {
        supersonic_metal_matmul_rhs_transposed_int4_bf16(
            batch_elems,
            m,
            n,
            k,
            group_size,
            lhs.as_ptr(),
            rhs_int4.as_ptr(),
            scale.as_ptr(),
            zero.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native matmul_rhs_transposed_int4_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn matmul_rhs_transposed_f32(
    batch_elems: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::F32
        || rhs.dtype() != ScalarType::F32
        || out.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native matmul_rhs_transposed_f32 expects F32 buffers, got {:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_matmul_rhs_transposed_f32(
            batch_elems,
            m,
            n,
            k,
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native matmul_rhs_transposed_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_projections_bf16(
    hidden_dim: usize,
    qkv_dim: usize,
    val_dim: usize,
    num_value_heads: usize,
    input: &GpuBuffer,
    qkv_weight: &GpuBuffer,
    z_weight: &GpuBuffer,
    a_weight: &GpuBuffer,
    b_weight: &GpuBuffer,
    qkv_out: &mut GpuBuffer,
    z_out: &mut GpuBuffer,
    a_out: &mut GpuBuffer,
    b_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        input.dtype(),
        qkv_weight.dtype(),
        z_weight.dtype(),
        a_weight.dtype(),
        b_weight.dtype(),
        qkv_out.dtype(),
        z_out.dtype(),
        a_out.dtype(),
        b_out.dtype(),
    ];
    if dtypes.iter().any(|dtype| *dtype != ScalarType::BF16) {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_projections_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || qkv_dim == 0 || val_dim == 0 || num_value_heads == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_projections_bf16 invalid shape: hidden_dim={hidden_dim} qkv_dim={qkv_dim} val_dim={val_dim} num_value_heads={num_value_heads}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_projections_bf16(
            hidden_dim,
            qkv_dim,
            val_dim,
            num_value_heads,
            input.as_ptr(),
            qkv_weight.as_ptr(),
            z_weight.as_ptr(),
            a_weight.as_ptr(),
            b_weight.as_ptr(),
            qkv_out.as_mut_ptr(),
            z_out.as_mut_ptr(),
            a_out.as_mut_ptr(),
            b_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_projections_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_gate_up_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    input: &GpuBuffer,
    gate_weight: &GpuBuffer,
    up_weight: &GpuBuffer,
    gate_out: &mut GpuBuffer,
    up_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        input.dtype(),
        gate_weight.dtype(),
        up_weight.dtype(),
        gate_out.dtype(),
        up_out.dtype(),
    ];
    if dtypes.iter().any(|dtype| *dtype != ScalarType::BF16) {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_gate_up_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || intermediate_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_gate_up_bf16 invalid shape: hidden_dim={hidden_dim} intermediate_dim={intermediate_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_mlp_gate_up_bf16(
            hidden_dim,
            intermediate_dim,
            input.as_ptr(),
            gate_weight.as_ptr(),
            up_weight.as_ptr(),
            gate_out.as_mut_ptr(),
            up_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_mlp_gate_up_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_gate_up_swiglu_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    input: &GpuBuffer,
    gate_weight: &GpuBuffer,
    up_weight: &GpuBuffer,
    mlp_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        input.dtype(),
        gate_weight.dtype(),
        up_weight.dtype(),
        mlp_out.dtype(),
    ];
    if dtypes.iter().any(|dtype| *dtype != ScalarType::BF16) {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_gate_up_swiglu_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || intermediate_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_gate_up_swiglu_bf16 invalid shape: hidden_dim={hidden_dim} intermediate_dim={intermediate_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_mlp_gate_up_swiglu_bf16(
            hidden_dim,
            intermediate_dim,
            input.as_ptr(),
            gate_weight.as_ptr(),
            up_weight.as_ptr(),
            mlp_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_mlp_gate_up_swiglu_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_down_residual_bf16(
    hidden_dim: usize,
    intermediate_dim: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    down_weight: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        gate.dtype(),
        up.dtype(),
        down_weight.dtype(),
        residual.dtype(),
        out.dtype(),
    ];
    if dtypes.iter().any(|dtype| *dtype != ScalarType::BF16) {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_down_residual_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || intermediate_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_mlp_down_residual_bf16 invalid shape: hidden_dim={hidden_dim} intermediate_dim={intermediate_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_mlp_down_residual_bf16(
            hidden_dim,
            intermediate_dim,
            gate.as_ptr(),
            up.as_ptr(),
            down_weight.as_ptr(),
            residual.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_mlp_down_residual_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_out_residual_f32_bf16(
    hidden_dim: usize,
    num_rows: usize,
    row_dim: usize,
    eps: f32,
    attn: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out_proj: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        attn.dtype(),
        gate.dtype(),
        weight.dtype(),
        out_proj.dtype(),
        residual.dtype(),
        out.dtype(),
    ];
    if dtypes
        != [
            ScalarType::F32,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
        ]
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_out_residual_f32_bf16 expects F32/BF16/BF16/BF16/BF16/BF16, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || num_rows == 0 || row_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_out_residual_f32_bf16 invalid shape: hidden_dim={hidden_dim} num_rows={num_rows} row_dim={row_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_out_residual_f32_bf16(
            hidden_dim,
            num_rows,
            row_dim,
            eps,
            attn.as_ptr(),
            gate.as_ptr(),
            weight.as_ptr(),
            out_proj.as_ptr(),
            residual.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_out_residual_f32_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_out_residual_bf16_bf16(
    hidden_dim: usize,
    num_rows: usize,
    row_dim: usize,
    eps: f32,
    attn: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out_proj: &GpuBuffer,
    residual: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        attn.dtype(),
        gate.dtype(),
        weight.dtype(),
        out_proj.dtype(),
        residual.dtype(),
        out.dtype(),
    ];
    if dtypes
        != [
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
            ScalarType::BF16,
        ]
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_out_residual_bf16_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || num_rows == 0 || row_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_out_residual_bf16_bf16 invalid shape: hidden_dim={hidden_dim} num_rows={num_rows} row_dim={row_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_out_residual_bf16_bf16(
            hidden_dim,
            num_rows,
            row_dim,
            eps,
            attn.as_ptr(),
            gate.as_ptr(),
            weight.as_ptr(),
            out_proj.as_ptr(),
            residual.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_out_residual_bf16_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_full_projections_bf16(
    hidden_dim: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    input: &GpuBuffer,
    q_weight: &GpuBuffer,
    k_weight: &GpuBuffer,
    v_weight: &GpuBuffer,
    q_out: &mut GpuBuffer,
    k_out: &mut GpuBuffer,
    v_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dtypes = [
        input.dtype(),
        q_weight.dtype(),
        k_weight.dtype(),
        v_weight.dtype(),
        q_out.dtype(),
        k_out.dtype(),
        v_out.dtype(),
    ];
    if dtypes.iter().any(|dtype| *dtype != ScalarType::BF16) {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_full_projections_bf16 expects BF16 buffers, got {dtypes:?}"
        )));
    }
    if hidden_dim == 0 || q_proj_dim == 0 || kv_dim == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_full_projections_bf16 invalid shape: hidden_dim={hidden_dim} q_proj_dim={q_proj_dim} kv_dim={kv_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_full_projections_bf16(
            hidden_dim,
            q_proj_dim,
            kv_dim,
            input.as_ptr(),
            q_weight.as_ptr(),
            k_weight.as_ptr(),
            v_weight.as_ptr(),
            q_out.as_mut_ptr(),
            k_out.as_mut_ptr(),
            v_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_full_projections_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn full_attention_prefill_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    full_attention_prefill_strided_bf16_f32(
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        kv_len,
        head_dim,
        scale,
        seqlen_offset,
        query,
        key,
        value,
        out,
    )
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_attention_prefill_strided_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    kv_stride: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query.dtype() != ScalarType::BF16
        || key.dtype() != ScalarType::BF16
        || value.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_prefill expects BF16 query/key/value, got {:?}/{:?}/{:?}",
            query.dtype(),
            key.dtype(),
            value.dtype()
        )));
    }
    if out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_prefill expects F32 output, got {:?}",
            out.dtype()
        )));
    }
    if kv_stride < kv_len {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_prefill requires kv_stride >= kv_len, got {kv_stride} < {kv_len}"
        )));
    }
    let status = unsafe {
        supersonic_metal_full_attention_prefill_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            kv_stride,
            head_dim,
            scale,
            seqlen_offset,
            query.as_ptr(),
            key.as_ptr(),
            value.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native full_attention_prefill_bf16_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_attention_decode_bf16_f32(
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    kv_stride: usize,
    head_dim: usize,
    scale: f32,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if query.dtype() != ScalarType::BF16
        || key.dtype() != ScalarType::BF16
        || value.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_decode expects BF16 query/key/value, got {:?}/{:?}/{:?}",
            query.dtype(),
            key.dtype(),
            value.dtype()
        )));
    }
    if out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_decode expects F32 output, got {:?}",
            out.dtype()
        )));
    }
    if kv_stride < kv_len {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_decode requires kv_stride >= kv_len, got {kv_stride} < {kv_len}"
        )));
    }
    if head_dim > 256 {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_decode supports head_dim <= 256, got {head_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_full_attention_decode_bf16_f32(
            q_heads,
            kv_heads,
            kv_len,
            kv_stride,
            head_dim,
            scale,
            query.as_ptr(),
            key.as_ptr(),
            value.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native full_attention_decode_bf16_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn rms_norm_rows_bf16(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    add_unit_offset: bool,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if input.dtype() != ScalarType::BF16
        || weight.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_rows_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_rms_norm_rows_bf16(
            n_rows,
            n_cols,
            eps,
            add_unit_offset,
            input.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_rows_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_rope_rows_bf16(
    n_rows: usize,
    n_cols: usize,
    rotary_dim: usize,
    eps: f32,
    pos_offset: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    cos: &GpuBuffer,
    sin: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if input.dtype() != ScalarType::BF16
        || weight.dtype() != ScalarType::BF16
        || cos.dtype() != ScalarType::BF16
        || sin.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_rope_rows_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            cos.dtype(),
            sin.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_rms_norm_rope_rows_bf16(
            n_rows,
            n_cols,
            rotary_dim,
            eps,
            pos_offset,
            input.as_ptr(),
            weight.as_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_rope_rows_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn rms_norm_rows_f32(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    add_unit_offset: bool,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if input.dtype() != ScalarType::F32 || out.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_rows_f32 expects F32 input/out, got {:?}/{:?}",
            input.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match weight.dtype() {
            ScalarType::BF16 => supersonic_metal_rms_norm_rows_f32_weight_bf16(
                n_rows,
                n_cols,
                eps,
                add_unit_offset,
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_rms_norm_rows_f32_weight_f32(
                n_rows,
                n_cols,
                eps,
                add_unit_offset,
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native rms_norm_rows_f32 expects BF16 or F32 weight, got {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_rows_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn rms_norm_gated_bf16(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    hidden: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if hidden.dtype() != ScalarType::BF16
        || gate.dtype() != ScalarType::BF16
        || weight.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_gated_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}/{:?}",
            hidden.dtype(),
            gate.dtype(),
            weight.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_rms_norm_gated_bf16(
            n_rows,
            n_cols,
            eps,
            hidden.as_ptr(),
            gate.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_gated_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn rms_norm_gated_f32(
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    hidden: &GpuBuffer,
    gate: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if hidden.dtype() != ScalarType::F32
        || gate.dtype() != ScalarType::F32
        || out.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native rms_norm_gated_f32 expects F32 hidden/gate/out, got {:?}/{:?}/{:?}",
            hidden.dtype(),
            gate.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match weight.dtype() {
            ScalarType::BF16 => supersonic_metal_rms_norm_gated_f32_weight_bf16(
                n_rows,
                n_cols,
                eps,
                hidden.as_ptr(),
                gate.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_rms_norm_gated_f32_weight_f32(
                n_rows,
                n_cols,
                eps,
                hidden.as_ptr(),
                gate.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native rms_norm_gated_f32 expects BF16 or F32 weight, got {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native rms_norm_gated_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn linear_prefill_conv_pack_bf16(
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
    mixed_qkv: &GpuBuffer,
    weights: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if mixed_qkv.dtype() != ScalarType::BF16
        || weights.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_prefill_conv_pack_bf16 expects BF16 buffers, got {:?}/{:?}/{:?}",
            mixed_qkv.dtype(),
            weights.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_linear_prefill_conv_pack_bf16(
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
            mixed_qkv.as_ptr(),
            weights.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native linear_prefill_conv_pack_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn l2norm(
    dtype: ScalarType,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = n_rows.checked_mul(n_cols).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native l2norm shape overflows: n_rows={n_rows} n_cols={n_cols}"
        ))
    })?;
    if total > u32::MAX as usize || n_rows > u32::MAX as usize || n_cols > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native l2norm supports u32-sized shapes, got n_rows={n_rows} n_cols={n_cols}"
        )));
    }
    if input.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native l2norm expects dtype {dtype:?}, got {:?}->{:?}",
            input.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::F32 => {
                supersonic_metal_l2norm_f32(n_rows, n_cols, eps, input.as_ptr(), out.as_mut_ptr())
            }
            ScalarType::BF16 => {
                supersonic_metal_l2norm_bf16(n_rows, n_cols, eps, input.as_ptr(), out.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native l2norm does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native l2norm failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn element_add(
    dtype: ScalarType,
    total_elems: usize,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native element_add supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if lhs.dtype() != dtype || rhs.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native element_add expects matching dtype {dtype:?}, got {:?}/{:?}/{:?}",
            lhs.dtype(),
            rhs.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_element_add_bf16(
                total_elems,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_element_add_f32(
                total_elems,
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native element_add does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native element_add failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn sigmoid_mul(
    dtype: ScalarType,
    total_elems: usize,
    data: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native sigmoid_mul supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if data.dtype() != dtype || gate.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native sigmoid_mul expects matching dtype {dtype:?}, got {:?}/{:?}/{:?}",
            data.dtype(),
            gate.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_sigmoid_mul_bf16(
                total_elems,
                data.as_ptr(),
                gate.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_sigmoid_mul_f32(
                total_elems,
                data.as_ptr(),
                gate.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native sigmoid_mul does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native sigmoid_mul failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn full_attention_gate_bf16(
    total_elems: usize,
    attn_f32: &GpuBuffer,
    gate: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_gate_bf16 supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if attn_f32.dtype() != ScalarType::F32
        || gate.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native full_attention_gate_bf16 expects F32/BF16/BF16 buffers, got {:?}/{:?}/{:?}",
            attn_f32.dtype(),
            gate.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        supersonic_metal_full_attention_gate_bf16(
            total_elems,
            attn_f32.as_ptr(),
            gate.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native full_attention_gate_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn swiglu_mul(
    dtype: ScalarType,
    total_elems: usize,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native swiglu_mul supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if gate.dtype() != dtype || up.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native swiglu_mul expects matching dtype {dtype:?}, got {:?}/{:?}/{:?}",
            gate.dtype(),
            up.dtype(),
            out.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_swiglu_mul_bf16(
                total_elems,
                gate.as_ptr(),
                up.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_swiglu_mul_f32(
                total_elems,
                gate.as_ptr(),
                up.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native swiglu_mul does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native swiglu_mul failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn cast(
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    total_elems: usize,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native cast supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if input.dtype() != input_dtype || out.dtype() != output_dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native cast expects buffer dtypes {input_dtype:?}->{output_dtype:?}, got {:?}->{:?}",
            input.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        match (input_dtype, output_dtype) {
            (ScalarType::BF16, ScalarType::BF16) => {
                supersonic_metal_cast_bf16_to_bf16(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::F32, ScalarType::F32) => {
                supersonic_metal_cast_f32_to_f32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::U32, ScalarType::U32) => {
                supersonic_metal_cast_u32_to_u32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::BF16, ScalarType::F32) => {
                supersonic_metal_cast_bf16_to_f32(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            (ScalarType::F32, ScalarType::BF16) => {
                supersonic_metal_cast_f32_to_bf16(total_elems, input.as_ptr(), out.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native cast does not support {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native cast failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn mul_scalar(
    dtype: ScalarType,
    total_elems: usize,
    scalar: f32,
    input: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if total_elems > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native mul_scalar supports at most {} elements, got {total_elems}",
            u32::MAX
        )));
    }
    if input.dtype() != dtype || out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native mul_scalar expects dtype {dtype:?}, got {:?}->{:?}",
            input.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_mul_scalar_bf16(
                total_elems,
                scalar,
                input.as_ptr(),
                out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_mul_scalar_f32(
                total_elems,
                scalar,
                input.as_ptr(),
                out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native mul_scalar does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native mul_scalar failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn transpose_shd_hsd(
    dtype: ScalarType,
    s: usize,
    h: usize,
    d: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = s
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native transpose_shd_hsd shape overflows: {s}x{h}x{d}"
            ))
        })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || h > u32::MAX as usize
        || d > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native transpose_shd_hsd supports u32-sized shapes, got {s}x{h}x{d}"
        )));
    }
    if src.dtype() != dtype || dst.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native transpose_shd_hsd expects dtype {dtype:?}, got {:?}->{:?}",
            src.dtype(),
            dst.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => {
                supersonic_metal_transpose_shd_hsd_bf16(s, h, d, src.as_ptr(), dst.as_mut_ptr())
            }
            ScalarType::F32 => {
                supersonic_metal_transpose_shd_hsd_f32(s, h, d, src.as_ptr(), dst.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native transpose_shd_hsd does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native transpose_shd_hsd failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn apply_rope_prefill(
    dtype: ScalarType,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    pos_offset: usize,
    data: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if data.dtype() != dtype
        || cos_table.dtype() != ScalarType::BF16
        || sin_table.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native apply_rope_prefill expects data={dtype:?} and BF16 tables, got data={:?} cos={:?} sin={:?}",
            data.dtype(),
            cos_table.dtype(),
            sin_table.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_apply_rope_prefill_bf16(
                seq_len,
                num_heads,
                head_dim,
                rotary_dim,
                pos_offset,
                cos_table.as_ptr(),
                sin_table.as_ptr(),
                data.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_apply_rope_prefill_f32(
                seq_len,
                num_heads,
                head_dim,
                rotary_dim,
                pos_offset,
                cos_table.as_ptr(),
                sin_table.as_ptr(),
                data.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native apply_rope_prefill does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native apply_rope_prefill failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn transpose_pad_conv(
    dtype: ScalarType,
    s: usize,
    c: usize,
    pad: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if src.dtype() != dtype || dst.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native transpose_pad_conv expects dtype {dtype:?}, got {:?}->{:?}",
            src.dtype(),
            dst.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => {
                supersonic_metal_transpose_pad_conv_bf16(s, c, pad, src.as_ptr(), dst.as_mut_ptr())
            }
            ScalarType::F32 => {
                supersonic_metal_transpose_pad_conv_f32(s, c, pad, src.as_ptr(), dst.as_mut_ptr())
            }
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native transpose_pad_conv does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native transpose_pad_conv failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn extract_conv_state(
    dtype: ScalarType,
    s: usize,
    c: usize,
    kern_minus_1: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if src.dtype() != dtype || dst.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native extract_conv_state expects dtype {dtype:?}, got {:?}->{:?}",
            src.dtype(),
            dst.dtype()
        )));
    }
    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_extract_conv_state_bf16(
                s,
                c,
                kern_minus_1,
                src.as_ptr(),
                dst.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_extract_conv_state_f32(
                s,
                c,
                kern_minus_1,
                src.as_ptr(),
                dst.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native extract_conv_state does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native extract_conv_state failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn split_qkv(
    dtype: ScalarType,
    s: usize,
    key_dim: usize,
    val_dim: usize,
    src: &GpuBuffer,
    q: &mut GpuBuffer,
    k: &mut GpuBuffer,
    v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let src_stride = key_dim
        .checked_mul(2)
        .and_then(|v| v.checked_add(val_dim))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native split_qkv shape overflows: s={s} key_dim={key_dim} val_dim={val_dim}"
            ))
        })?;
    let total = s.checked_mul(src_stride).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native split_qkv total overflows: s={s} stride={src_stride}"
        ))
    })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || key_dim > u32::MAX as usize
        || val_dim > u32::MAX as usize
        || src_stride > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qkv supports u32-sized shapes, got s={s} key_dim={key_dim} val_dim={val_dim}"
        )));
    }
    if src.dtype() != dtype || q.dtype() != dtype || k.dtype() != dtype || v.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qkv expects dtype {dtype:?}, got src={:?} q={:?} k={:?} v={:?}",
            src.dtype(),
            q.dtype(),
            k.dtype(),
            v.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_split_qkv_bf16(
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_split_qkv_f32(
                s,
                key_dim,
                val_dim,
                src.as_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native split_qkv does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native split_qkv failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn split_qgate(
    dtype: ScalarType,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    src: &GpuBuffer,
    query_out: &mut GpuBuffer,
    gate_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = s
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native split_qgate shape overflows: s={s} heads={num_heads} head_dim={head_dim}"
            ))
        })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || num_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qgate supports u32-sized shapes, got s={s} heads={num_heads} head_dim={head_dim}"
        )));
    }
    if src.dtype() != dtype || query_out.dtype() != dtype || gate_out.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native split_qgate expects dtype {dtype:?}, got src={:?} query={:?} gate={:?}",
            src.dtype(),
            query_out.dtype(),
            gate_out.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_split_qgate_bf16(
                s,
                num_heads,
                head_dim,
                src.as_ptr(),
                query_out.as_mut_ptr(),
                gate_out.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_split_qgate_f32(
                s,
                num_heads,
                head_dim,
                src.as_ptr(),
                query_out.as_mut_ptr(),
                gate_out.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native split_qgate does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native split_qgate failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn repeat_interleave_heads(
    dtype: ScalarType,
    s: usize,
    n_heads: usize,
    head_dim: usize,
    repeats: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let dst_heads = n_heads.checked_mul(repeats).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native repeat_interleave_heads overflows: n_heads={n_heads} repeats={repeats}"
        ))
    })?;
    let total = s
        .checked_mul(dst_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "metal native repeat_interleave_heads shape overflows: s={s} dst_heads={dst_heads} head_dim={head_dim}"
            ))
        })?;
    if total > u32::MAX as usize
        || s > u32::MAX as usize
        || n_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || repeats > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native repeat_interleave_heads supports u32-sized shapes, got s={s} n_heads={n_heads} head_dim={head_dim} repeats={repeats}"
        )));
    }
    if src.dtype() != dtype || dst.dtype() != dtype {
        return Err(GpuError::InvalidArg(format!(
            "metal native repeat_interleave_heads expects dtype {dtype:?}, got src={:?} dst={:?}",
            src.dtype(),
            dst.dtype()
        )));
    }

    let status = unsafe {
        match dtype {
            ScalarType::BF16 => supersonic_metal_repeat_interleave_heads_bf16(
                s,
                n_heads,
                head_dim,
                repeats,
                src.as_ptr(),
                dst.as_mut_ptr(),
            ),
            ScalarType::F32 => supersonic_metal_repeat_interleave_heads_f32(
                s,
                n_heads,
                head_dim,
                repeats,
                src.as_ptr(),
                dst.as_mut_ptr(),
            ),
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "metal native repeat_interleave_heads does not support dtype {other:?}"
                )));
            }
        }
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native repeat_interleave_heads failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn compute_beta_g_f32(
    seq_len: usize,
    nv: usize,
    b: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    beta: &mut GpuBuffer,
    g: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total = seq_len.checked_mul(nv).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native compute_beta_g_f32 shape overflows: seq_len={seq_len} nv={nv}"
        ))
    })?;
    if total > u32::MAX as usize || seq_len > u32::MAX as usize || nv > u32::MAX as usize {
        return Err(GpuError::InvalidArg(format!(
            "metal native compute_beta_g_f32 supports u32-sized shapes, got seq_len={seq_len} nv={nv}"
        )));
    }
    if b.dtype() != ScalarType::F32
        || a.dtype() != ScalarType::F32
        || dt_bias.dtype() != ScalarType::F32
        || a_log_exp.dtype() != ScalarType::F32
        || beta.dtype() != ScalarType::F32
        || g.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native compute_beta_g_f32 expects F32 buffers, got b={:?} a={:?} dt_bias={:?} a_log_exp={:?} beta={:?} g={:?}",
            b.dtype(),
            a.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            beta.dtype(),
            g.dtype()
        )));
    }

    let status = unsafe {
        supersonic_metal_compute_beta_g_f32(
            seq_len,
            nv,
            b.as_ptr(),
            a.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            beta.as_mut_ptr(),
            g.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native compute_beta_g_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn delta_recurrent_prefill_f32(
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: &GpuBuffer,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    beta: &GpuBuffer,
    g: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let total_threads = batch_heads.checked_mul(v_head_dim).ok_or_else(|| {
        GpuError::InvalidArg(format!(
            "metal native delta_recurrent_prefill_f32 shape overflows: batch_heads={batch_heads} v_head_dim={v_head_dim}"
        ))
    })?;
    if total_threads > u32::MAX as usize
        || batch_heads > u32::MAX as usize
        || seq_len > u32::MAX as usize
        || k_head_dim > u32::MAX as usize
        || v_head_dim > u32::MAX as usize
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native delta_recurrent_prefill_f32 supports u32-sized shapes, got batch_heads={batch_heads} seq_len={seq_len} k_head_dim={k_head_dim} v_head_dim={v_head_dim}"
        )));
    }
    if initial_state.dtype() != ScalarType::F32
        || query.dtype() != ScalarType::F32
        || key.dtype() != ScalarType::F32
        || value.dtype() != ScalarType::F32
        || beta.dtype() != ScalarType::F32
        || g.dtype() != ScalarType::F32
        || out.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native delta_recurrent_prefill_f32 expects F32 buffers, got initial_state={:?} query={:?} key={:?} value={:?} beta={:?} g={:?} out={:?}",
            initial_state.dtype(),
            query.dtype(),
            key.dtype(),
            value.dtype(),
            beta.dtype(),
            g.dtype(),
            out.dtype()
        )));
    }

    let status = unsafe {
        supersonic_metal_delta_recurrent_prefill_f32(
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            initial_state.as_ptr(),
            query.as_ptr(),
            key.as_ptr(),
            value.as_ptr(),
            beta.as_ptr(),
            g.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native delta_recurrent_prefill_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn linear_decode_apply_parts_f32(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    q_scaled: &GpuBuffer,
    k_normed: &GpuBuffer,
    v_linear: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if q_scaled.dtype() != ScalarType::F32
        || k_normed.dtype() != ScalarType::F32
        || v_linear.dtype() != ScalarType::F32
        || initial_state.dtype() != ScalarType::F32
        || out.dtype() != ScalarType::F32
        || a.dtype() != ScalarType::BF16
        || b.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_decode_apply_parts_f32 expects F32 q/k/v/state/out and BF16 a/b/dt/a_log, got q={:?} k={:?} v={:?} a={:?} b={:?} dt={:?} a_log={:?} state={:?} out={:?}",
            q_scaled.dtype(),
            k_normed.dtype(),
            v_linear.dtype(),
            a.dtype(),
            b.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            initial_state.dtype(),
            out.dtype()
        )));
    }
    if num_v_heads == 0 || num_k_heads == 0 || num_v_heads % num_k_heads != 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_decode_apply_parts_f32 invalid heads: num_v_heads={num_v_heads} num_k_heads={num_k_heads}"
        )));
    }
    let status = unsafe {
        supersonic_metal_linear_decode_apply_parts_f32(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            q_scaled.as_ptr(),
            k_normed.as_ptr(),
            v_linear.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            initial_state.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native linear_decode_apply_parts_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_prep_bf16_f32(
    key_dim: usize,
    val_dim: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    conv_pack: &GpuBuffer,
    q_bf16: &mut GpuBuffer,
    k_bf16: &mut GpuBuffer,
    v_bf16: &mut GpuBuffer,
    q_f32: &mut GpuBuffer,
    k_f32: &mut GpuBuffer,
    v_f32: &mut GpuBuffer,
    q_normed: &mut GpuBuffer,
    q_scaled: &mut GpuBuffer,
    k_normed: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_pack.dtype() != ScalarType::BF16
        || q_bf16.dtype() != ScalarType::BF16
        || k_bf16.dtype() != ScalarType::BF16
        || v_bf16.dtype() != ScalarType::BF16
        || q_f32.dtype() != ScalarType::F32
        || k_f32.dtype() != ScalarType::F32
        || v_f32.dtype() != ScalarType::F32
        || q_normed.dtype() != ScalarType::F32
        || q_scaled.dtype() != ScalarType::F32
        || k_normed.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_prep_bf16_f32 expects BF16 conv/q/k/v and F32 q/k/v/norms, got conv={:?} q={:?} k={:?} v={:?} qf={:?} kf={:?} vf={:?} qn={:?} qs={:?} kn={:?}",
            conv_pack.dtype(),
            q_bf16.dtype(),
            k_bf16.dtype(),
            v_bf16.dtype(),
            q_f32.dtype(),
            k_f32.dtype(),
            v_f32.dtype(),
            q_normed.dtype(),
            q_scaled.dtype(),
            k_normed.dtype()
        )));
    }
    if key_dim == 0 || val_dim == 0 || num_key_heads == 0 || key_dim != num_key_heads * key_head_dim
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_prep_bf16_f32 invalid shape: key_dim={key_dim} val_dim={val_dim} num_key_heads={num_key_heads} key_head_dim={key_head_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_prep_bf16_f32(
            key_dim,
            val_dim,
            num_key_heads,
            key_head_dim,
            conv_pack.as_ptr(),
            q_bf16.as_mut_ptr(),
            k_bf16.as_mut_ptr(),
            v_bf16.as_mut_ptr(),
            q_f32.as_mut_ptr(),
            k_f32.as_mut_ptr(),
            v_f32.as_mut_ptr(),
            q_normed.as_mut_ptr(),
            q_scaled.as_mut_ptr(),
            k_normed.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_prep_bf16_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_prep_decode_apply_bf16_f32(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_pack: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    initial_state: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_pack.dtype() != ScalarType::BF16
        || a.dtype() != ScalarType::BF16
        || b.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
        || initial_state.dtype() != ScalarType::F32
        || out.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_prep_decode_apply_bf16_f32 expects BF16 conv/a/b/dt/a_log and F32 state/out, got conv={:?} a={:?} b={:?} dt={:?} a_log={:?} state={:?} out={:?}",
            conv_pack.dtype(),
            a.dtype(),
            b.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            initial_state.dtype(),
            out.dtype(),
        )));
    }
    if num_v_heads == 0
        || num_k_heads == 0
        || head_k_dim == 0
        || head_v_dim == 0
        || num_v_heads % num_k_heads != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_prep_decode_apply_bf16_f32 invalid shape: num_v_heads={num_v_heads} num_k_heads={num_k_heads} head_k_dim={head_k_dim} head_v_dim={head_v_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_prep_decode_apply_bf16_f32(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_pack.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            initial_state.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_prep_decode_apply_bf16_f32 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_decode_apply_inplace_bf16(
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_pack: &GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    state: &mut GpuBuffer,
    attn_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if conv_pack.dtype() != ScalarType::BF16
        || a.dtype() != ScalarType::BF16
        || b.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
        || state.dtype() != ScalarType::F32
        || attn_out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_decode_apply_inplace_bf16 expects BF16 conv/a/b/dt/a_log/attn and F32 state, got conv={:?} a={:?} b={:?} dt={:?} a_log={:?} state={:?} attn={:?}",
            conv_pack.dtype(),
            a.dtype(),
            b.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            state.dtype(),
            attn_out.dtype()
        )));
    }
    if num_v_heads == 0
        || num_k_heads == 0
        || head_k_dim == 0
        || head_v_dim == 0
        || num_v_heads % num_k_heads != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native qwen_linear_decode_apply_inplace_bf16 invalid shape: num_v_heads={num_v_heads} num_k_heads={num_k_heads} head_k_dim={head_k_dim} head_v_dim={head_v_dim}"
        )));
    }
    let status = unsafe {
        supersonic_metal_qwen_linear_decode_apply_inplace_bf16(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_pack.as_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            state.as_mut_ptr(),
            attn_out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native qwen_linear_decode_apply_inplace_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn conv_state_update_bf16(
    channels: usize,
    state_len: usize,
    qkv: &GpuBuffer,
    state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if qkv.dtype() != ScalarType::BF16 || state.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "metal native conv_state_update_bf16 expects BF16 qkv/state, got qkv={:?} state={:?}",
            qkv.dtype(),
            state.dtype()
        )));
    }
    if channels == 0 || state_len == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native conv_state_update_bf16 invalid shape: channels={channels} state_len={state_len}"
        )));
    }
    let status = unsafe {
        supersonic_metal_conv_state_update_bf16(
            channels,
            state_len,
            qkv.as_ptr(),
            state.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native conv_state_update_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
pub(crate) fn linear_conv_value_decay_bf16(
    conv_dim: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
    mixed_qkv: &GpuBuffer,
    prev_state: &GpuBuffer,
    weights: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if mixed_qkv.dtype() != ScalarType::BF16
        || prev_state.dtype() != ScalarType::BF16
        || weights.dtype() != ScalarType::BF16
        || a.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_conv_value_decay_bf16 expects BF16 buffers, got mixed={:?} state={:?} weights={:?} a={:?} dt={:?} a_log={:?} out={:?}",
            mixed_qkv.dtype(),
            prev_state.dtype(),
            weights.dtype(),
            a.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            out.dtype()
        )));
    }
    if conv_dim == 0 || state_len == 0 || kernel_size != state_len + 1 || num_heads == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_conv_value_decay_bf16 invalid shape: conv_dim={conv_dim} state_len={state_len} kernel_size={kernel_size} num_heads={num_heads}"
        )));
    }
    let status = unsafe {
        supersonic_metal_linear_conv_value_decay_bf16(
            conv_dim,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv.as_ptr(),
            prev_state.as_ptr(),
            weights.as_ptr(),
            a.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native linear_conv_value_decay_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", supersonic_backend_metal))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_conv_value_decay_update_bf16(
    conv_dim: usize,
    state_len: usize,
    kernel_size: usize,
    num_heads: usize,
    mixed_qkv: &GpuBuffer,
    state: &mut GpuBuffer,
    weights: &GpuBuffer,
    a: &GpuBuffer,
    dt_bias: &GpuBuffer,
    a_log_exp: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if mixed_qkv.dtype() != ScalarType::BF16
        || state.dtype() != ScalarType::BF16
        || weights.dtype() != ScalarType::BF16
        || a.dtype() != ScalarType::BF16
        || dt_bias.dtype() != ScalarType::BF16
        || a_log_exp.dtype() != ScalarType::BF16
        || out.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_conv_value_decay_update_bf16 expects BF16 buffers, got mixed={:?} state={:?} weights={:?} a={:?} dt={:?} a_log={:?} out={:?}",
            mixed_qkv.dtype(),
            state.dtype(),
            weights.dtype(),
            a.dtype(),
            dt_bias.dtype(),
            a_log_exp.dtype(),
            out.dtype()
        )));
    }
    if conv_dim == 0 || state_len == 0 || kernel_size != state_len + 1 || num_heads == 0 {
        return Err(GpuError::InvalidArg(format!(
            "metal native linear_conv_value_decay_update_bf16 invalid shape: conv_dim={conv_dim} state_len={state_len} kernel_size={kernel_size} num_heads={num_heads}"
        )));
    }
    let status = unsafe {
        supersonic_metal_linear_conv_value_decay_update_bf16(
            conv_dim,
            state_len,
            kernel_size,
            num_heads,
            mixed_qkv.as_ptr(),
            state.as_mut_ptr(),
            weights.as_ptr(),
            a.as_ptr(),
            dt_bias.as_ptr(),
            a_log_exp.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::Metal(format!(
            "metal native linear_conv_value_decay_update_bf16 failed with status {status}"
        )));
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
impl MetalBatchGuard {
    pub(crate) fn begin() -> Result<Self, GpuError> {
        Ok(Self { active: false })
    }

    pub(crate) fn finish(self) -> Result<(), GpuError> {
        Ok(())
    }
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn flush_batch() -> Result<(), GpuError> {
    Ok(())
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn set_batch_label(_label: &str) -> Result<(), GpuError> {
    Ok(())
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn copy_d2d(
    _src: *const c_void,
    _dst: *mut c_void,
    _bytes: usize,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native copy_d2d is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn linear_decode_apply_parts_f32(
    _num_v_heads: usize,
    _num_k_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
    _q_scaled: &GpuBuffer,
    _k_normed: &GpuBuffer,
    _v_linear: &GpuBuffer,
    _a: &GpuBuffer,
    _b: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _initial_state: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native linear_decode_apply_parts_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_prep_bf16_f32(
    _key_dim: usize,
    _val_dim: usize,
    _num_key_heads: usize,
    _key_head_dim: usize,
    _conv_pack: &GpuBuffer,
    _q_bf16: &mut GpuBuffer,
    _k_bf16: &mut GpuBuffer,
    _v_bf16: &mut GpuBuffer,
    _q_f32: &mut GpuBuffer,
    _k_f32: &mut GpuBuffer,
    _v_f32: &mut GpuBuffer,
    _q_normed: &mut GpuBuffer,
    _q_scaled: &mut GpuBuffer,
    _k_normed: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_prep_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_prep_decode_apply_bf16_f32(
    _num_v_heads: usize,
    _num_k_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
    _conv_pack: &GpuBuffer,
    _a: &GpuBuffer,
    _b: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _initial_state: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_prep_decode_apply_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_decode_apply_inplace_bf16(
    _num_v_heads: usize,
    _num_k_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
    _conv_pack: &GpuBuffer,
    _a: &GpuBuffer,
    _b: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _state: &mut GpuBuffer,
    _attn_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_decode_apply_inplace_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn conv_state_update_bf16(
    _channels: usize,
    _state_len: usize,
    _qkv: &GpuBuffer,
    _state: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native conv_state_update_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn linear_conv_value_decay_bf16(
    _conv_dim: usize,
    _state_len: usize,
    _kernel_size: usize,
    _num_heads: usize,
    _mixed_qkv: &GpuBuffer,
    _prev_state: &GpuBuffer,
    _weights: &GpuBuffer,
    _a: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native linear_conv_value_decay_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_conv_value_decay_update_bf16(
    _conv_dim: usize,
    _state_len: usize,
    _kernel_size: usize,
    _num_heads: usize,
    _mixed_qkv: &GpuBuffer,
    _state: &mut GpuBuffer,
    _weights: &GpuBuffer,
    _a: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native linear_conv_value_decay_update_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn embedding_lookup_bf16(
    _token_count: usize,
    _vocab_size: usize,
    _hidden_size: usize,
    _embeddings: &GpuBuffer,
    _indexes: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native embedding_lookup_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn matmul_rhs_transposed_bf16(
    _batch_elems: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native matmul_rhs_transposed_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn matmul_rhs_transposed_residual_bf16(
    _batch_elems: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _residual: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native matmul_rhs_transposed_residual_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_rhs_transposed_int4_bf16(
    _batch_elems: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _group_size: usize,
    _lhs: &GpuBuffer,
    _rhs_int4: &GpuBuffer,
    _scale: &GpuBuffer,
    _zero: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native matmul_rhs_transposed_int4_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn matmul_rhs_transposed_f32(
    _batch_elems: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native matmul_rhs_transposed_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_projections_bf16(
    _hidden_dim: usize,
    _qkv_dim: usize,
    _val_dim: usize,
    _num_value_heads: usize,
    _input: &GpuBuffer,
    _qkv_weight: &GpuBuffer,
    _z_weight: &GpuBuffer,
    _a_weight: &GpuBuffer,
    _b_weight: &GpuBuffer,
    _qkv_out: &mut GpuBuffer,
    _z_out: &mut GpuBuffer,
    _a_out: &mut GpuBuffer,
    _b_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_projections_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_gate_up_bf16(
    _hidden_dim: usize,
    _intermediate_dim: usize,
    _input: &GpuBuffer,
    _gate_weight: &GpuBuffer,
    _up_weight: &GpuBuffer,
    _gate_out: &mut GpuBuffer,
    _up_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_mlp_gate_up_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_gate_up_swiglu_bf16(
    _hidden_dim: usize,
    _intermediate_dim: usize,
    _input: &GpuBuffer,
    _gate_weight: &GpuBuffer,
    _up_weight: &GpuBuffer,
    _mlp_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_mlp_gate_up_swiglu_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_mlp_down_residual_bf16(
    _hidden_dim: usize,
    _intermediate_dim: usize,
    _gate: &GpuBuffer,
    _up: &GpuBuffer,
    _down_weight: &GpuBuffer,
    _residual: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_mlp_down_residual_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_out_residual_f32_bf16(
    _hidden_dim: usize,
    _num_rows: usize,
    _row_dim: usize,
    _eps: f32,
    _attn: &GpuBuffer,
    _gate: &GpuBuffer,
    _weight: &GpuBuffer,
    _out_proj: &GpuBuffer,
    _residual: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_out_residual_f32_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_linear_out_residual_bf16_bf16(
    _hidden_dim: usize,
    _num_rows: usize,
    _row_dim: usize,
    _eps: f32,
    _attn: &GpuBuffer,
    _gate: &GpuBuffer,
    _weight: &GpuBuffer,
    _out_proj: &GpuBuffer,
    _residual: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_linear_out_residual_bf16_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_full_projections_bf16(
    _hidden_dim: usize,
    _q_proj_dim: usize,
    _kv_dim: usize,
    _input: &GpuBuffer,
    _q_weight: &GpuBuffer,
    _k_weight: &GpuBuffer,
    _v_weight: &GpuBuffer,
    _q_out: &mut GpuBuffer,
    _k_out: &mut GpuBuffer,
    _v_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native qwen_full_projections_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn lm_head_argmax_bf16(
    _hidden: &GpuBuffer,
    _weight: &GpuBuffer,
    _out_index: &mut GpuBuffer,
    _in_dim: usize,
    _vocab_size: usize,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native lm_head_argmax_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn argmax_bf16(
    _logits: &GpuBuffer,
    _out_index: &mut GpuBuffer,
    _n: usize,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native argmax_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn full_attention_prefill_bf16_f32(
    _q_heads: usize,
    _kv_heads: usize,
    _q_len: usize,
    _kv_len: usize,
    _head_dim: usize,
    _scale: f32,
    _seqlen_offset: usize,
    _query: &GpuBuffer,
    _key: &GpuBuffer,
    _value: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native full_attention_prefill_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_attention_prefill_strided_bf16_f32(
    _q_heads: usize,
    _kv_heads: usize,
    _q_len: usize,
    _kv_len: usize,
    _kv_stride: usize,
    _head_dim: usize,
    _scale: f32,
    _seqlen_offset: usize,
    _query: &GpuBuffer,
    _key: &GpuBuffer,
    _value: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native full_attention_prefill_strided_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_attention_decode_bf16_f32(
    _q_heads: usize,
    _kv_heads: usize,
    _kv_len: usize,
    _kv_stride: usize,
    _head_dim: usize,
    _scale: f32,
    _query: &GpuBuffer,
    _key: &GpuBuffer,
    _value: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native full_attention_decode_bf16_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn rms_norm_rows_bf16(
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _add_unit_offset: bool,
    _input: &GpuBuffer,
    _weight: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_rows_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_rope_rows_bf16(
    _n_rows: usize,
    _n_cols: usize,
    _rotary_dim: usize,
    _eps: f32,
    _pos_offset: usize,
    _input: &GpuBuffer,
    _weight: &GpuBuffer,
    _cos: &GpuBuffer,
    _sin: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_rope_rows_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn rms_norm_rows_f32(
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _add_unit_offset: bool,
    _input: &GpuBuffer,
    _weight: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_rows_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn rms_norm_gated_bf16(
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _hidden: &GpuBuffer,
    _gate: &GpuBuffer,
    _weight: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_gated_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn rms_norm_gated_f32(
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _hidden: &GpuBuffer,
    _gate: &GpuBuffer,
    _weight: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native rms_norm_gated_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn linear_prefill_conv_pack_bf16(
    _conv_dim: usize,
    _total_len: usize,
    _seq_len: usize,
    _kernel_size: usize,
    _mixed_qkv: &GpuBuffer,
    _weights: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native linear_prefill_conv_pack_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn l2norm(
    _dtype: ScalarType,
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _input: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native l2norm is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn element_add(
    _dtype: ScalarType,
    _total_elems: usize,
    _lhs: &GpuBuffer,
    _rhs: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native element_add is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn sigmoid_mul(
    _dtype: ScalarType,
    _total_elems: usize,
    _data: &GpuBuffer,
    _gate: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native sigmoid_mul is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn full_attention_gate_bf16(
    _total_elems: usize,
    _attn_f32: &GpuBuffer,
    _gate: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native full_attention_gate_bf16 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn swiglu_mul(
    _dtype: ScalarType,
    _total_elems: usize,
    _gate: &GpuBuffer,
    _up: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native swiglu_mul is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn cast(
    _input_dtype: ScalarType,
    _output_dtype: ScalarType,
    _total_elems: usize,
    _input: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal("metal native cast is not compiled".into()))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn mul_scalar(
    _dtype: ScalarType,
    _total_elems: usize,
    _scalar: f32,
    _input: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native mul_scalar is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn transpose_shd_hsd(
    _dtype: ScalarType,
    _s: usize,
    _h: usize,
    _d: usize,
    _src: &GpuBuffer,
    _dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native transpose_shd_hsd is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn apply_rope_prefill(
    _dtype: ScalarType,
    _seq_len: usize,
    _num_heads: usize,
    _head_dim: usize,
    _rotary_dim: usize,
    _cos_table: &GpuBuffer,
    _sin_table: &GpuBuffer,
    _pos_offset: usize,
    _data: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native apply_rope_prefill is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn transpose_pad_conv(
    _dtype: ScalarType,
    _s: usize,
    _c: usize,
    _pad: usize,
    _src: &GpuBuffer,
    _dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native transpose_pad_conv is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn extract_conv_state(
    _dtype: ScalarType,
    _s: usize,
    _c: usize,
    _kern_minus_1: usize,
    _src: &GpuBuffer,
    _dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native extract_conv_state is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn split_qkv(
    _dtype: ScalarType,
    _s: usize,
    _key_dim: usize,
    _val_dim: usize,
    _src: &GpuBuffer,
    _q: &mut GpuBuffer,
    _k: &mut GpuBuffer,
    _v: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native split_qkv is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn split_qgate(
    _dtype: ScalarType,
    _s: usize,
    _num_heads: usize,
    _head_dim: usize,
    _src: &GpuBuffer,
    _query_out: &mut GpuBuffer,
    _gate_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native split_qgate is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn repeat_interleave_heads(
    _dtype: ScalarType,
    _s: usize,
    _n_heads: usize,
    _head_dim: usize,
    _repeats: usize,
    _src: &GpuBuffer,
    _dst: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native repeat_interleave_heads is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn compute_beta_g_f32(
    _seq_len: usize,
    _nv: usize,
    _b: &GpuBuffer,
    _a: &GpuBuffer,
    _dt_bias: &GpuBuffer,
    _a_log_exp: &GpuBuffer,
    _beta: &mut GpuBuffer,
    _g: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native compute_beta_g_f32 is not compiled".into(),
    ))
}

#[cfg(not(all(target_os = "macos", supersonic_backend_metal)))]
pub(crate) fn delta_recurrent_prefill_f32(
    _batch_heads: usize,
    _seq_len: usize,
    _k_head_dim: usize,
    _v_head_dim: usize,
    _initial_state: &GpuBuffer,
    _query: &GpuBuffer,
    _key: &GpuBuffer,
    _value: &GpuBuffer,
    _beta: &GpuBuffer,
    _g: &GpuBuffer,
    _out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    Err(GpuError::Metal(
        "metal native delta_recurrent_prefill_f32 is not compiled".into(),
    ))
}

#[cfg(all(test, target_os = "macos", supersonic_backend_metal))]
mod tests {
    use super::*;
    use gpu_hal::{set_backend, Backend};
    use half::bf16;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16 buffer");
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    fn read_f32(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download f32 buffer");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_u32(buffer: &GpuBuffer) -> Vec<u32> {
        let bytes = buffer.to_host_bytes().expect("download u32 buffer");
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn metal_native_embedding_lookup_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let embeddings_vals = [
            1.0f32, 1.5, 2.0, 10.0, 10.5, 11.0, -2.0, -2.5, -3.0, 4.0, 4.5, 5.0,
        ];
        let embeddings = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4, 3],
            &bf16_bytes(&embeddings_vals),
        )
        .expect("upload embeddings");
        let indexes =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[3], &u32_bytes(&[2, 0, 3]))
                .expect("upload indexes");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3, 3]).expect("allocate native out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3, 3]).expect("allocate ref out");

        crate::metal_host::embedding_lookup(3, 4, 3, &embeddings, &indexes, &mut out_ref)
            .expect("host embedding lookup");
        embedding_lookup_bf16(3, 4, 3, &embeddings, &indexes, &mut out_native)
            .expect("native embedding lookup");

        assert_eq!(read_bf16(&out_native), read_bf16(&out_ref));
    }

    #[test]
    fn metal_native_matmul_rhs_transposed_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 3],
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .expect("upload lhs");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 0.0, 1.0, 0.5, -1.0, 2.0]),
        )
        .expect("upload rhs");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2, 2]).expect("allocate out");

        matmul_rhs_transposed_bf16(1, 2, 2, 3, &lhs, &rhs, &mut out).expect("run native matmul");

        let actual = read_bf16(&out);
        let expected = [4.0f32, 4.5, 10.0, 9.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_matmul_rhs_transposed_residual_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 3],
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .expect("upload lhs");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 0.0, 1.0, 0.5, -1.0, 2.0]),
        )
        .expect("upload rhs");
        let residual = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[0.25, -0.5, 1.0, 2.0]),
        )
        .expect("upload residual");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2, 2]).expect("allocate out");

        matmul_rhs_transposed_residual_bf16(1, 2, 2, 3, &lhs, &rhs, &residual, &mut out)
            .expect("run native matmul residual");

        let actual = read_bf16(&out);
        let expected = [4.25f32, 4.0, 11.0, 11.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_matmul_rhs_transposed_int4_matches_reference() {
        // GPTQ INT4: m=1, n=4, k=4, group_size=2.
        // Scale grid is [n/gs, k/gs] = [2, 2]: cols 0-1 use scale row 0;
        // cols 2-3 use scale row 1. Each row has two K-direction tiles
        // (k 0-1 share one (s, z), k 2-3 share another).
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let lhs_vals: [f32; 4] = [1.0, 0.5, -1.0, 2.0];
        let lhs =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[1, 1, 4], &bf16_bytes(&lhs_vals))
                .expect("upload lhs");

        // rhs [batch=1, n=4, k/2=2]: low nibble = even k, high nibble = odd k.
        let nibbles: [[u8; 4]; 4] = [
            [1, 2, 3, 4], // col 0
            [5, 6, 7, 8], // col 1
            [9, 10, 11, 12], // col 2
            [13, 14, 15, 0], // col 3
        ];
        let mut rhs_bytes = Vec::with_capacity(4 * 2);
        for col_nibbles in &nibbles {
            rhs_bytes.push(col_nibbles[0] | (col_nibbles[1] << 4));
            rhs_bytes.push(col_nibbles[2] | (col_nibbles[3] << 4));
        }
        let rhs_int4 =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[1, 4, 2], &rhs_bytes)
                .expect("upload rhs_int4");

        // scale/zero shape [scale_rows=2, scale_cols=2].
        // Index: sc_idx = (col / gs) * scale_cols + (kk / gs).
        let scale_vals: [f32; 4] = [0.5, 0.25, 0.125, 1.0];
        let zero_vals: [f32; 4] = [2.0, 1.0, 4.0, 0.5];
        let scale = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 2],
            &bf16_bytes(&scale_vals),
        )
        .expect("upload scale");
        let zero = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 2],
            &bf16_bytes(&zero_vals),
        )
        .expect("upload zero");

        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, 4]).expect("allocate out");

        matmul_rhs_transposed_int4_bf16(1, 1, 4, 4, 2, &lhs, &rhs_int4, &scale, &zero, &mut out)
            .expect("run native matmul int4");

        let bf16_round = |x: f32| -> f32 { bf16::from_f32(x).to_f32() };
        let scale_cols_n = 2usize;
        let group_size = 2usize;
        let mut expected = [0.0f32; 4];
        for col in 0..4usize {
            let scale_row = col / group_size;
            let mut acc = 0.0f32;
            for kk in 0..4usize {
                let sc_col = kk / group_size;
                let si = scale_row * scale_cols_n + sc_col;
                let s = scale_vals[si];
                let z = zero_vals[si];
                let w = bf16_round(nibbles[col][kk] as f32 * s - z * s);
                acc += lhs_vals[kk] * w;
            }
            expected[col] = acc;
        }

        let actual = read_bf16(&out);
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.05,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_qwen_linear_projections_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 3],
            &bf16_bytes(&[1.0, 2.0, -1.0]),
        )
        .expect("upload input");
        let qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 0.0, 1.0, 0.5, -1.0, 2.0]),
        )
        .expect("upload qkv weight");
        let z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[0.0, 1.0, 1.0, -1.0, 0.5, 0.0]),
        )
        .expect("upload z weight");
        let a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 3],
            &bf16_bytes(&[2.0, 1.0, -1.0]),
        )
        .expect("upload a weight");
        let b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 3],
            &bf16_bytes(&[-1.0, 0.0, 0.5]),
        )
        .expect("upload b weight");
        let mut qkv = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2]).expect("qkv out");
        let mut z = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2]).expect("z out");
        let mut a = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1]).expect("a out");
        let mut b = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1]).expect("b out");

        qwen_linear_projections_bf16(
            3, 2, 2, 1, &input, &qkv_w, &z_w, &a_w, &b_w, &mut qkv, &mut z, &mut a, &mut b,
        )
        .expect("run fused qwen projections");

        let cases = [
            ("qkv", read_bf16(&qkv), vec![0.0, -3.5]),
            ("z", read_bf16(&z), vec![1.0, 0.0]),
            ("a", read_bf16(&a), vec![5.0]),
            ("b", read_bf16(&b), vec![-1.5]),
        ];
        for (name, actual, expected) in cases {
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 0.02,
                    "{name}[{idx}]: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_qwen_full_projections_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_dim = 3usize;
        let q_proj_dim = 4usize;
        let kv_dim = 2usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            &bf16_bytes(&[1.0, 2.0, -1.0]),
        )
        .expect("upload input");
        let q_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[q_proj_dim, hidden_dim],
            &bf16_bytes(&[
                1.0, 0.0, 1.0, 0.5, -1.0, 2.0, -2.0, 0.5, 1.0, 0.25, 0.75, -0.5,
            ]),
        )
        .expect("upload q weight");
        let k_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[kv_dim, hidden_dim],
            &bf16_bytes(&[0.0, 1.0, 1.0, -1.0, 0.5, 0.0]),
        )
        .expect("upload k weight");
        let v_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[kv_dim, hidden_dim],
            &bf16_bytes(&[2.0, 1.0, -1.0, -1.0, 0.0, 0.5]),
        )
        .expect("upload v weight");
        let mut q_out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_proj_dim]).expect("q out");
        let mut k_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim]).expect("k out");
        let mut v_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim]).expect("v out");

        qwen_full_projections_bf16(
            hidden_dim, q_proj_dim, kv_dim, &input, &q_w, &k_w, &v_w, &mut q_out, &mut k_out,
            &mut v_out,
        )
        .expect("run fused qwen full projections");

        let cases = [
            ("q", read_bf16(&q_out), vec![0.0, -3.5, -2.0, 2.25]),
            ("k", read_bf16(&k_out), vec![1.0, 0.0]),
            ("v", read_bf16(&v_out), vec![5.0, -1.5]),
        ];
        for (name, actual, expected) in cases {
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 0.02,
                    "{name}[{idx}]: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_qwen_mlp_fusions_match_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_dim = 3usize;
        let intermediate = 4usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            &bf16_bytes(&[1.0, 2.0, -1.0]),
        )
        .expect("upload input");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[intermediate, hidden_dim],
            &bf16_bytes(&[
                1.0, 0.0, 1.0, 0.5, -1.0, 2.0, -2.0, 0.5, 1.0, 0.25, 0.75, -0.5,
            ]),
        )
        .expect("upload gate weight");
        let up_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[intermediate, hidden_dim],
            &bf16_bytes(&[
                0.0, 1.0, 0.5, 1.0, 1.0, -1.0, 0.25, -0.5, 2.0, -1.0, 0.0, 0.5,
            ]),
        )
        .expect("upload up weight");
        let down_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_dim, intermediate],
            &bf16_bytes(&[
                0.5, 1.0, -0.25, 0.75, -1.0, 0.25, 0.5, 1.5, 0.0, -0.5, 1.0, 0.25,
            ]),
        )
        .expect("upload down weight");
        let residual = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            &bf16_bytes(&[0.25, -0.5, 1.0]),
        )
        .expect("upload residual");
        let mut gate_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("gate ref");
        let mut up_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("up ref");
        let mut mlp_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("mlp ref");
        let mut down_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim]).expect("down ref");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim]).expect("out ref");
        let mut gate =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("gate");
        let mut up = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("up");
        let mut mlp = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate]).expect("mlp");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim]).expect("out");

        crate::metal_host::matmul_rhs_transposed(
            ScalarType::BF16,
            1,
            1,
            intermediate,
            hidden_dim,
            &input,
            &gate_w,
            &mut gate_ref,
        )
        .expect("host gate");
        crate::metal_host::matmul_rhs_transposed(
            ScalarType::BF16,
            1,
            1,
            intermediate,
            hidden_dim,
            &input,
            &up_w,
            &mut up_ref,
        )
        .expect("host up");
        crate::metal_host::swiglu_mul(
            ScalarType::BF16,
            intermediate,
            &gate_ref,
            &up_ref,
            &mut mlp_ref,
        )
        .expect("host swiglu");
        crate::metal_host::matmul_rhs_transposed(
            ScalarType::BF16,
            1,
            1,
            hidden_dim,
            intermediate,
            &mlp_ref,
            &down_w,
            &mut down_ref,
        )
        .expect("host down");
        crate::metal_host::element_add(
            ScalarType::BF16,
            hidden_dim,
            &residual,
            &down_ref,
            &mut out_ref,
        )
        .expect("host residual");

        qwen_mlp_gate_up_bf16(
            hidden_dim,
            intermediate,
            &input,
            &gate_w,
            &up_w,
            &mut gate,
            &mut up,
        )
        .expect("native gate/up");
        qwen_mlp_gate_up_swiglu_bf16(hidden_dim, intermediate, &input, &gate_w, &up_w, &mut mlp)
            .expect("native gate/up/swiglu");
        qwen_mlp_down_residual_bf16(
            hidden_dim,
            intermediate,
            &gate,
            &up,
            &down_w,
            &residual,
            &mut out,
        )
        .expect("native down/residual");

        for (label, actual, expected) in [
            ("gate", read_bf16(&gate), read_bf16(&gate_ref)),
            ("up", read_bf16(&up), read_bf16(&up_ref)),
            ("mlp", read_bf16(&mlp), read_bf16(&mlp_ref)),
            ("out", read_bf16(&out), read_bf16(&out_ref)),
        ] {
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 0.02,
                    "{label} idx {idx}: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_qwen_linear_prep_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let conv_pack = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 11],
            &bf16_bytes(&[3.0, 4.0, 0.0, 5.0, 1.0, 2.0, 2.0, 1.0, -1.0, 0.5, 7.0]),
        )
        .expect("upload conv_pack");
        let mut q_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 4]).expect("q bf16");
        let mut k_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 4]).expect("k bf16");
        let mut v_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 3]).expect("v bf16");
        let mut q_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("q f32");
        let mut k_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 4]).expect("k f32");
        let mut v_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 3]).expect("v f32");
        let mut q_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("q normed");
        let mut q_scaled = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("q scaled");
        let mut k_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("k normed");

        qwen_linear_prep_bf16_f32(
            4,
            3,
            2,
            2,
            &conv_pack,
            &mut q_bf16,
            &mut k_bf16,
            &mut v_bf16,
            &mut q_f32,
            &mut k_f32,
            &mut v_f32,
            &mut q_normed,
            &mut q_scaled,
            &mut k_normed,
        )
        .expect("run fused qwen linear prep");

        assert_eq!(read_bf16(&q_bf16), vec![3.0, 4.0, 0.0, 5.0]);
        assert_eq!(read_bf16(&k_bf16), vec![1.0, 2.0, 2.0, 1.0]);
        assert_eq!(read_bf16(&v_bf16), vec![-1.0, 0.5, 7.0]);
        assert_eq!(read_f32(&q_f32), vec![3.0, 4.0, 0.0, 5.0]);
        assert_eq!(read_f32(&k_f32), vec![1.0, 2.0, 2.0, 1.0]);
        assert_eq!(read_f32(&v_f32), vec![-1.0, 0.5, 7.0]);

        let inv_sqrt_2 = 2.0f32.sqrt().recip();
        let cases = [
            ("q_normed", read_f32(&q_normed), vec![0.6, 0.8, 0.0, 1.0]),
            (
                "q_scaled",
                read_f32(&q_scaled),
                vec![0.6 * inv_sqrt_2, 0.8 * inv_sqrt_2, 0.0, inv_sqrt_2],
            ),
            (
                "k_normed",
                read_f32(&k_normed),
                vec![
                    1.0 / 5.0f32.sqrt(),
                    2.0 / 5.0f32.sqrt(),
                    2.0 / 5.0f32.sqrt(),
                    1.0 / 5.0f32.sqrt(),
                ],
            ),
        ];
        for (name, actual, expected) in cases {
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 1e-4,
                    "{name}[{idx}]: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_linear_conv_value_decay_update_matches_two_step_path() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let conv_dim = 4usize;
        let state_len = 2usize;
        let kernel_size = 3usize;
        let num_heads = 2usize;
        let mixed_qkv = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[conv_dim],
            &bf16_bytes(&[0.5, -1.0, 2.0, 3.0]),
        )
        .expect("upload mixed_qkv");
        let state_vals = [0.25f32, -0.5, 1.0, 1.5, -2.0, 0.75, 0.0, 4.0];
        let mut state_ref = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[conv_dim, state_len],
            &bf16_bytes(&state_vals),
        )
        .expect("upload ref state");
        let mut state_fused = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[conv_dim, state_len],
            &bf16_bytes(&state_vals),
        )
        .expect("upload fused state");
        let weights = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[conv_dim, kernel_size],
            &bf16_bytes(&[
                0.5, -0.25, 1.0, -1.0, 0.75, 0.25, 0.125, 0.5, -0.5, 1.5, -0.75, 0.25,
            ]),
        )
        .expect("upload conv weights");
        let a = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_heads],
            &bf16_bytes(&[0.25, -1.25]),
        )
        .expect("upload a");
        let dt_bias = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_heads],
            &bf16_bytes(&[0.5, -0.25]),
        )
        .expect("upload dt_bias");
        let a_log_exp = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_heads],
            &bf16_bytes(&[0.75, 1.5]),
        )
        .expect("upload a_log_exp");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[conv_dim + num_heads]).expect("out ref");
        let mut out_fused = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[conv_dim + num_heads])
            .expect("out fused");

        linear_conv_value_decay_bf16(
            conv_dim,
            state_len,
            kernel_size,
            num_heads,
            &mixed_qkv,
            &state_ref,
            &weights,
            &a,
            &dt_bias,
            &a_log_exp,
            &mut out_ref,
        )
        .expect("run unfused conv/value decay");
        conv_state_update_bf16(conv_dim, state_len, &mixed_qkv, &mut state_ref)
            .expect("run state update");
        linear_conv_value_decay_update_bf16(
            conv_dim,
            state_len,
            kernel_size,
            num_heads,
            &mixed_qkv,
            &mut state_fused,
            &weights,
            &a,
            &dt_bias,
            &a_log_exp,
            &mut out_fused,
        )
        .expect("run fused conv/value decay update");

        assert_eq!(read_bf16(&out_fused), read_bf16(&out_ref));
        assert_eq!(read_bf16(&state_fused), read_bf16(&state_ref));
    }

    #[test]
    fn metal_native_qwen_linear_decode_apply_inplace_matches_out_buffer_path() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let num_v_heads = 2usize;
        let num_k_heads = 1usize;
        let head_k_dim = 2usize;
        let head_v_dim = 2usize;
        let value_dim = num_v_heads * head_v_dim;
        let state_dim = num_v_heads * head_k_dim * head_v_dim;
        let conv_pack = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * num_k_heads * head_k_dim + value_dim],
            &bf16_bytes(&[0.5, 1.0, -0.25, 0.75, 1.5, -1.0, 0.25, 2.0]),
        )
        .expect("upload conv_pack");
        let a = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_v_heads],
            &bf16_bytes(&[0.25, -0.5]),
        )
        .expect("upload a");
        let b = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_v_heads],
            &bf16_bytes(&[1.0, -1.5]),
        )
        .expect("upload b");
        let dt_bias = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_v_heads],
            &bf16_bytes(&[0.125, -0.25]),
        )
        .expect("upload dt_bias");
        let a_log_exp = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[num_v_heads],
            &bf16_bytes(&[0.75, 1.25]),
        )
        .expect("upload a_log_exp");
        let state_vals = [0.5f32, -0.25, 1.0, 0.75, -1.5, 0.25, 0.0, 2.0];
        let initial_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[state_dim],
            &f32_bytes(&state_vals),
        )
        .expect("upload initial state");
        let mut inplace_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[state_dim],
            &f32_bytes(&state_vals),
        )
        .expect("upload inplace state");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[value_dim + state_dim]).expect("out ref");
        let mut attn_inplace =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[value_dim]).expect("attn inplace");

        qwen_linear_prep_decode_apply_bf16_f32(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            &conv_pack,
            &a,
            &b,
            &dt_bias,
            &a_log_exp,
            &initial_state,
            &mut out_ref,
        )
        .expect("run out-buffer decode apply");
        qwen_linear_decode_apply_inplace_bf16(
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            &conv_pack,
            &a,
            &b,
            &dt_bias,
            &a_log_exp,
            &mut inplace_state,
            &mut attn_inplace,
        )
        .expect("run in-place decode apply");

        let out_ref_f32 = read_f32(&out_ref);
        let expected_attn: Vec<f32> = out_ref_f32[..value_dim]
            .iter()
            .map(|value| bf16::from_f32(*value).to_f32())
            .collect();
        for (idx, (actual, expected)) in read_bf16(&attn_inplace)
            .iter()
            .zip(expected_attn.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 0.0,
                "attn {idx}: expected {expected}, got {actual}"
            );
        }
        for (idx, (actual, expected)) in read_f32(&inplace_state)
            .iter()
            .zip(out_ref_f32[value_dim..].iter())
            .enumerate()
        {
            let delta = (actual - expected).abs();
            assert!(
                delta <= 1e-5,
                "state {idx}: expected {expected}, got {actual}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_matmul_rhs_transposed_f32_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 3],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .expect("upload lhs");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 3],
            &f32_bytes(&[1.0, 0.0, 1.0, 0.5, -1.0, 2.0]),
        )
        .expect("upload rhs");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 2]).expect("allocate out");

        matmul_rhs_transposed_f32(1, 2, 2, 3, &lhs, &rhs, &mut out).expect("run native F32 matmul");

        let actual = read_f32(&out);
        let expected = [4.0f32, 4.5, 10.0, 9.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_lm_head_argmax_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_vals = [1.0f32, -2.0, 0.5];
        let weight_vals = [
            0.0f32, 0.0, 1.0, // 0.5
            1.0, 1.0, 1.0, // -0.5
            -1.0, -2.0, 0.0, // 3.0
            0.25, 0.0, 0.25, // 0.375
        ];
        let hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 3],
            &bf16_bytes(&hidden_vals),
        )
        .expect("upload hidden");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4, 3],
            &bf16_bytes(&weight_vals),
        )
        .expect("upload weight");
        let mut out_index =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("allocate out_index");

        lm_head_argmax_bf16(&hidden, &weight, &mut out_index, 3, 4)
            .expect("run native lm-head argmax");

        assert_eq!(read_u32(&out_index), vec![2]);
    }

    #[test]
    fn metal_native_full_attention_prefill_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .expect("upload query");
        let key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .expect("upload key");
        let value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[10.0, 1.0, 1.0, 20.0]),
        )
        .expect("upload value");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 2]).expect("allocate out");

        full_attention_prefill_bf16_f32(1, 1, 2, 2, 2, 1.0, 0, &query, &key, &value, &mut out)
            .expect("run native full attention");

        let actual = read_f32(&out);
        let prob0 = 1.0f32 / (1.0 + 1.0f32.exp());
        let prob1 = 1.0 - prob0;
        let expected = [
            10.0f32,
            1.0,
            prob0 * 10.0 + prob1 * 1.0,
            prob0 * 1.0 + prob1 * 20.0,
        ];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-4,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_full_attention_strided_matches_contiguous() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 1, 2],
            &bf16_bytes(&[0.5, 1.0]),
        )
        .expect("upload query");
        let key_contig = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .expect("upload contiguous key");
        let value_contig = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 2, 2],
            &bf16_bytes(&[10.0, 1.0, 1.0, 20.0]),
        )
        .expect("upload contiguous value");
        let key_strided = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 4, 2],
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0, 99.0, 99.0, 77.0, 77.0]),
        )
        .expect("upload strided key");
        let value_strided = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, 4, 2],
            &bf16_bytes(&[10.0, 1.0, 1.0, 20.0, 99.0, 99.0, 77.0, 77.0]),
        )
        .expect("upload strided value");
        let mut out_contig =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 2]).expect("allocate contig out");
        let mut out_strided =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, 2]).expect("allocate strided out");

        full_attention_prefill_bf16_f32(
            1,
            1,
            1,
            2,
            2,
            1.0,
            1,
            &query,
            &key_contig,
            &value_contig,
            &mut out_contig,
        )
        .expect("run contiguous attention");
        full_attention_prefill_strided_bf16_f32(
            1,
            1,
            1,
            2,
            4,
            2,
            1.0,
            1,
            &query,
            &key_strided,
            &value_strided,
            &mut out_strided,
        )
        .expect("run strided attention");

        for (idx, (a, e)) in read_f32(&out_strided)
            .iter()
            .zip(read_f32(&out_contig).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_rms_norm_rows_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[1.0, 2.0, 2.0, 2.0, 0.0, 2.0]),
        )
        .expect("upload input");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[0.0, 0.5, -0.5]),
        )
        .expect("upload weight");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("allocate out");

        rms_norm_rows_bf16(2, 3, 1e-5, true, &input, &weight, &mut out)
            .expect("run native rms norm");

        let actual = read_bf16(&out);
        let row0_inv = 1.0f32 / ((3.0f32 + 1e-5).sqrt());
        let row1_inv = 1.0f32 / (((8.0f32 / 3.0f32) + 1e-5).sqrt());
        let expected = [
            1.0 * row0_inv * 1.0,
            2.0 * row0_inv * 1.5,
            2.0 * row0_inv * 0.5,
            2.0 * row1_inv * 1.0,
            0.0,
            2.0 * row1_inv * 0.5,
        ];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_rms_norm_rows_f32_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 3],
            &f32_bytes(&[1.0, 2.0, 2.0, 2.0, 0.0, 2.0]),
        )
        .expect("upload input");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[0.0, 0.5, -0.5]),
        )
        .expect("upload weight");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate native out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate ref out");

        crate::metal_host::rms_norm_rows(
            ScalarType::F32,
            2,
            3,
            1e-5,
            true,
            &input,
            &weight,
            &mut out_ref,
        )
        .expect("host F32 rms norm");
        rms_norm_rows_f32(2, 3, 1e-5, true, &input, &weight, &mut out_native)
            .expect("native F32 rms norm");

        for (idx, (a, e)) in read_f32(&out_native)
            .iter()
            .zip(read_f32(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_rms_norm_gated_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_vals = [1.0f32, 2.0, 2.0, 2.0, 0.0, 2.0];
        let gate_vals = [0.0f32, 1.0, -1.0, 3.0, -2.0, 0.5];
        let hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&hidden_vals),
        )
        .expect("upload hidden");
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[2, 3], &bf16_bytes(&gate_vals))
                .expect("upload gate");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[1.0, 0.5, -0.5]),
        )
        .expect("upload weight");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("allocate native out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("allocate ref out");

        crate::metal_host::rms_norm_gated(
            ScalarType::BF16,
            2,
            3,
            1e-5,
            &hidden,
            &gate,
            &weight,
            &mut out_ref,
        )
        .expect("host gated rms norm");
        rms_norm_gated_bf16(2, 3, 1e-5, &hidden, &gate, &weight, &mut out_native)
            .expect("native gated rms norm");

        for (idx, (a, e)) in read_bf16(&out_native)
            .iter()
            .zip(read_bf16(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_rms_norm_gated_f32_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_vals = [1.0f32, 2.0, 2.0, 2.0, 0.0, 2.0];
        let gate_vals = [0.0f32, 1.0, -1.0, 3.0, -2.0, 0.5];
        let hidden =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[2, 3], &f32_bytes(&hidden_vals))
                .expect("upload hidden");
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[2, 3], &f32_bytes(&gate_vals))
                .expect("upload gate");
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[1.0, 0.5, -0.5]),
        )
        .expect("upload weight");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate native out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate ref out");

        crate::metal_host::rms_norm_gated(
            ScalarType::F32,
            2,
            3,
            1e-5,
            &hidden,
            &gate,
            &weight,
            &mut out_ref,
        )
        .expect("host gated F32 rms norm");
        rms_norm_gated_f32(2, 3, 1e-5, &hidden, &gate, &weight, &mut out_native)
            .expect("native gated F32 rms norm");

        for (idx, (a, e)) in read_f32(&out_native)
            .iter()
            .zip(read_f32(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_linear_prefill_conv_pack_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let cases = [
            (
                2usize,
                4usize,
                2usize,
                3usize,
                vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, -2.0],
                vec![1.0, 0.5, -1.0, -1.0, 0.5, 1.0],
            ),
            (
                96usize,
                11usize,
                8usize,
                4usize,
                (0..(96 * 11))
                    .map(|idx| {
                        let centered = (idx % 23) as f32 - 11.0;
                        centered / 7.0
                    })
                    .collect::<Vec<_>>(),
                (0..(96 * 4))
                    .map(|idx| {
                        let centered = (idx % 17) as f32 - 8.0;
                        centered / 5.0
                    })
                    .collect::<Vec<_>>(),
            ),
        ];

        for (case_idx, (conv_dim, total_len, seq_len, kernel_size, mixed_vals, weight_vals)) in
            cases.into_iter().enumerate()
        {
            let mixed = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[conv_dim, total_len],
                &bf16_bytes(&mixed_vals),
            )
            .expect("upload mixed");
            let weights = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[conv_dim, kernel_size],
                &bf16_bytes(&weight_vals),
            )
            .expect("upload weights");
            let mut expected = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, conv_dim])
                .expect("allocate expected");
            let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, conv_dim])
                .expect("allocate out");

            crate::metal_host::linear_prefill_conv_pack(
                ScalarType::BF16,
                1,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                &mixed,
                &weights,
                &mut expected,
            )
            .expect("run host conv");

            linear_prefill_conv_pack_bf16(
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                &mixed,
                &weights,
                &mut out,
            )
            .expect("run native conv");

            let expected = read_bf16(&expected);
            let actual = read_bf16(&out);
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let delta = (a - e).abs();
                assert!(
                    delta <= 0.02,
                    "case {case_idx} idx {idx}: expected {e}, got {a}, delta {delta}"
                );
            }
        }
    }

    #[test]
    fn metal_native_element_add_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[1.0, -2.0, 0.5, 10.0]),
        )
        .expect("upload lhs bf16");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[0.25, 3.0, -0.25, -1.5]),
        )
        .expect("upload rhs bf16");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate bf16 out");
        element_add(ScalarType::BF16, 4, &lhs, &rhs, &mut out).expect("run bf16 add");
        let actual = read_bf16(&out);
        let expected = [1.25f32, 1.0, 0.25, 8.5];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let lhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.25, -4.0, 8.0]),
        )
        .expect("upload lhs f32");
        let rhs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[-0.25, 2.0, 0.5]),
        )
        .expect("upload rhs f32");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        element_add(ScalarType::F32, 3, &lhs, &rhs, &mut out).expect("run f32 add");
        assert_eq!(read_f32(&out), vec![1.0, -2.0, 8.5]);
    }

    #[test]
    fn metal_native_sigmoid_mul_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let data_vals = [1.0f32, -2.0, 0.5, 10.0];
        let gate_vals = [0.0f32, 1.0, -1.0, 3.0];
        let data =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[4], &bf16_bytes(&data_vals))
                .expect("upload bf16 data");
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[4], &bf16_bytes(&gate_vals))
                .expect("upload bf16 gate");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate native bf16 out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate ref bf16 out");
        crate::metal_host::sigmoid_mul(ScalarType::BF16, 4, &data, &gate, &mut out_ref)
            .expect("host bf16 sigmoid_mul");
        sigmoid_mul(ScalarType::BF16, 4, &data, &gate, &mut out_native)
            .expect("native bf16 sigmoid_mul");
        for (idx, (a, e)) in read_bf16(&out_native)
            .iter()
            .zip(read_bf16(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let data =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[4], &f32_bytes(&data_vals))
                .expect("upload f32 data");
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[4], &f32_bytes(&gate_vals))
                .expect("upload f32 gate");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[4]).expect("allocate native f32 out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[4]).expect("allocate ref f32 out");
        crate::metal_host::sigmoid_mul(ScalarType::F32, 4, &data, &gate, &mut out_ref)
            .expect("host f32 sigmoid_mul");
        sigmoid_mul(ScalarType::F32, 4, &data, &gate, &mut out_native)
            .expect("native f32 sigmoid_mul");
        for (idx, (a, e)) in read_f32(&out_native)
            .iter()
            .zip(read_f32(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-6,
                "f32 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_full_attention_gate_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let attn_vals = [1.125f32, -2.75, 0.03125, 8.5, -0.5];
        let gate_vals = [0.0f32, 1.0, -1.0, 3.0, -3.0];
        let attn =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[5], &f32_bytes(&attn_vals))
                .expect("upload f32 attn");
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[5], &bf16_bytes(&gate_vals))
                .expect("upload bf16 gate");
        let mut attn_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[5]).expect("allocate cast out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[5]).expect("allocate ref out");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[5]).expect("allocate native out");

        crate::metal_host::cast(ScalarType::F32, ScalarType::BF16, 5, &attn, &mut attn_bf16)
            .expect("host cast attention");
        crate::metal_host::sigmoid_mul(ScalarType::BF16, 5, &attn_bf16, &gate, &mut out_ref)
            .expect("host gate attention");
        full_attention_gate_bf16(5, &attn, &gate, &mut out_native)
            .expect("native full attention gate");

        for (idx, (a, e)) in read_bf16(&out_native)
            .iter()
            .zip(read_bf16(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_swiglu_mul_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let gate_vals = [0.0f32, 1.0, -1.0, 3.0];
        let up_vals = [1.0f32, -2.0, 0.5, 10.0];
        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[4], &bf16_bytes(&gate_vals))
                .expect("upload bf16 gate");
        let up = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[4], &bf16_bytes(&up_vals))
            .expect("upload bf16 up");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate native bf16 out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate ref bf16 out");
        crate::metal_host::swiglu_mul(ScalarType::BF16, 4, &gate, &up, &mut out_ref)
            .expect("host bf16 swiglu_mul");
        swiglu_mul(ScalarType::BF16, 4, &gate, &up, &mut out_native)
            .expect("native bf16 swiglu_mul");
        for (idx, (a, e)) in read_bf16(&out_native)
            .iter()
            .zip(read_bf16(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let gate =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[4], &f32_bytes(&gate_vals))
                .expect("upload f32 gate");
        let up = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[4], &f32_bytes(&up_vals))
            .expect("upload f32 up");
        let mut out_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[4]).expect("allocate native f32 out");
        let mut out_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[4]).expect("allocate ref f32 out");
        crate::metal_host::swiglu_mul(ScalarType::F32, 4, &gate, &up, &mut out_ref)
            .expect("host f32 swiglu_mul");
        swiglu_mul(ScalarType::F32, 4, &gate, &up, &mut out_native).expect("native f32 swiglu_mul");
        for (idx, (a, e)) in read_f32(&out_native)
            .iter()
            .zip(read_f32(&out_ref).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 5e-6,
                "f32 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_cast_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[3],
            &bf16_bytes(&[1.0, -2.5, 0.25]),
        )
        .expect("upload bf16 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        cast(ScalarType::BF16, ScalarType::F32, 3, &input, &mut out).expect("run bf16->f32 cast");
        assert_eq!(read_f32(&out), vec![1.0, -2.5, 0.25]);

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.0, -2.25, 3.5]),
        )
        .expect("upload f32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3]).expect("allocate bf16 out");
        cast(ScalarType::F32, ScalarType::BF16, 3, &input, &mut out).expect("run f32->bf16 cast");
        let actual = read_bf16(&out);
        for (idx, (a, e)) in actual.iter().zip([1.0f32, -2.25, 3.5].iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "f32->bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U32,
            &[3],
            &u32_bytes(&[7, 42, u32::MAX - 1]),
        )
        .expect("upload u32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::U32, &[3]).expect("allocate u32 out");
        cast(ScalarType::U32, ScalarType::U32, 3, &input, &mut out).expect("run u32 copy cast");
        assert_eq!(read_u32(&out), vec![7, 42, u32::MAX - 1]);
    }

    #[test]
    fn metal_native_mul_scalar_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[4],
            &bf16_bytes(&[1.0, -2.0, 0.5, 8.0]),
        )
        .expect("upload bf16 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("allocate bf16 out");
        mul_scalar(ScalarType::BF16, 4, -0.5, &input, &mut out).expect("run bf16 mul_scalar");
        let actual = read_bf16(&out);
        let expected = [-0.5f32, 1.0, -0.25, -4.0];
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.02,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3],
            &f32_bytes(&[1.25, -4.0, 8.0]),
        )
        .expect("upload f32 input");
        let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, &[3]).expect("allocate f32 out");
        mul_scalar(ScalarType::F32, 3, 2.0, &input, &mut out).expect("run f32 mul_scalar");
        assert_eq!(read_f32(&out), vec![2.5, -8.0, 16.0]);
    }

    #[test]
    fn metal_native_transpose_shd_hsd_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input_vals = (0..12).map(|v| v as f32).collect::<Vec<_>>();
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3, 2],
            &bf16_bytes(&input_vals),
        )
        .expect("upload bf16 input");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[3, 2, 2]).expect("allocate bf16 out");
        transpose_shd_hsd(ScalarType::BF16, 2, 3, 2, &input, &mut out).expect("run bf16 transpose");
        assert_eq!(
            read_bf16(&out),
            vec![0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]
        );

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 2, 2],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .expect("upload f32 input");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2, 2]).expect("allocate f32 out");
        transpose_shd_hsd(ScalarType::F32, 2, 2, 2, &input, &mut out).expect("run f32 transpose");
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn metal_native_apply_rope_prefill_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input_vals = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        let cos_vals = [1.0f32, 1.0, 0.5, 0.25, -0.5, 0.75];
        let sin_vals = [0.0f32, 0.0, 0.5, -0.25, 0.25, 0.5];
        let mut native = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 1, 4],
            &f32_bytes(&input_vals),
        )
        .expect("upload native data");
        let mut reference = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 1, 4],
            &f32_bytes(&input_vals),
        )
        .expect("upload reference data");
        let cos =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[3, 2], &bf16_bytes(&cos_vals))
                .expect("upload cos");
        let sin =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[3, 2], &bf16_bytes(&sin_vals))
                .expect("upload sin");

        crate::metal_host::apply_rope_prefill(
            ScalarType::F32,
            2,
            1,
            4,
            4,
            &cos,
            &sin,
            1,
            &mut reference,
        )
        .expect("host rope");
        apply_rope_prefill(ScalarType::F32, 2, 1, 4, 4, &cos, &sin, 1, &mut native)
            .expect("native rope");

        for (idx, (a, e)) in read_f32(&native)
            .iter()
            .zip(read_f32(&reference).iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_conv_layout_helpers_match_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[3, 2],
            &f32_bytes(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]),
        )
        .expect("upload input");

        let mut pad_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 5]).expect("allocate pad native");
        let mut pad_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 5]).expect("allocate pad ref");
        crate::metal_host::transpose_pad_conv(ScalarType::F32, 3, 2, 2, &input, &mut pad_ref)
            .expect("host transpose_pad_conv");
        transpose_pad_conv(ScalarType::F32, 3, 2, 2, &input, &mut pad_native)
            .expect("native transpose_pad_conv");
        assert_eq!(read_f32(&pad_native), read_f32(&pad_ref));

        let mut state_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 4]).expect("allocate state native");
        let mut state_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 4]).expect("allocate state ref");
        crate::metal_host::extract_conv_state(ScalarType::F32, 3, 2, 4, &input, &mut state_ref)
            .expect("host extract_conv_state");
        extract_conv_state(ScalarType::F32, 3, 2, 4, &input, &mut state_native)
            .expect("native extract_conv_state");
        assert_eq!(read_f32(&state_native), read_f32(&state_ref));
    }

    #[test]
    fn metal_native_split_qkv_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 5],
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        .expect("upload bf16 input");
        let mut q = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("allocate bf16 q");
        let mut k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("allocate bf16 k");
        let mut v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 1]).expect("allocate bf16 v");
        split_qkv(ScalarType::BF16, 2, 2, 1, &input, &mut q, &mut k, &mut v)
            .expect("run bf16 split_qkv");
        assert_eq!(read_bf16(&q), vec![1.0, 2.0, 10.0, 20.0]);
        assert_eq!(read_bf16(&k), vec![3.0, 4.0, 30.0, 40.0]);
        assert_eq!(read_bf16(&v), vec![5.0, 50.0]);

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 7],
            &f32_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            ]),
        )
        .expect("upload f32 input");
        let mut q = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("allocate f32 q");
        let mut k = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("allocate f32 k");
        let mut v = GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("allocate f32 v");
        split_qkv(ScalarType::F32, 2, 2, 3, &input, &mut q, &mut k, &mut v)
            .expect("run f32 split_qkv");
        assert_eq!(read_f32(&q), vec![1.0, 2.0, 11.0, 12.0]);
        assert_eq!(read_f32(&k), vec![3.0, 4.0, 13.0, 14.0]);
        assert_eq!(read_f32(&v), vec![5.0, 6.0, 7.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn metal_native_split_qgate_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 2, 4],
            &bf16_bytes(&[
                1.0, 2.0, 101.0, 102.0, 3.0, 4.0, 103.0, 104.0, 10.0, 20.0, 110.0, 120.0, 30.0,
                40.0, 130.0, 140.0,
            ]),
        )
        .expect("upload bf16 input");
        let mut query =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2, 2]).expect("allocate bf16 query");
        let mut gate =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2, 2]).expect("allocate bf16 gate");
        split_qgate(ScalarType::BF16, 2, 2, 2, &input, &mut query, &mut gate)
            .expect("run bf16 split_qgate");
        assert_eq!(
            read_bf16(&query),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(
            read_bf16(&gate),
            vec![101.0, 102.0, 103.0, 104.0, 110.0, 120.0, 130.0, 140.0]
        );

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[1, 2, 4],
            &f32_bytes(&[1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0]),
        )
        .expect("upload f32 input");
        let mut query =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 2]).expect("allocate f32 query");
        let mut gate =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 2, 2]).expect("allocate f32 gate");
        split_qgate(ScalarType::F32, 1, 2, 2, &input, &mut query, &mut gate)
            .expect("run f32 split_qgate");
        assert_eq!(read_f32(&query), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read_f32(&gate), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn metal_native_repeat_interleave_heads_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 2, 2],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
        )
        .expect("upload f32 input");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 6, 2]).expect("allocate f32 out");
        repeat_interleave_heads(ScalarType::F32, 2, 2, 2, 3, &input, &mut out)
            .expect("run f32 repeat_interleave_heads");
        assert_eq!(
            read_f32(&out),
            vec![
                1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 10.0, 20.0, 10.0, 20.0,
                10.0, 20.0, 30.0, 40.0, 30.0, 40.0, 30.0, 40.0
            ]
        );
    }

    #[test]
    fn metal_native_compute_beta_g_f32_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let seq_len = 3usize;
        let nv = 2usize;
        let b_vals = vec![0.0f32, 1.0, -1.0, 2.0, 3.0, -2.0];
        let a_vals = vec![0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0];
        let dt_bias_vals = vec![0.1f32, -0.2];
        let a_log_exp_vals = vec![1.5f32, 0.75];

        let b = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[seq_len, nv],
            &f32_bytes(&b_vals),
        )
        .expect("upload b");
        let a = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[seq_len, nv],
            &f32_bytes(&a_vals),
        )
        .expect("upload a");
        let dt_bias =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[nv], &f32_bytes(&dt_bias_vals))
                .expect("upload dt_bias");
        let a_log_exp = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[nv],
            &f32_bytes(&a_log_exp_vals),
        )
        .expect("upload a_log_exp");
        let mut beta_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, seq_len]).expect("alloc beta native");
        let mut g_native =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, seq_len]).expect("alloc g native");
        let mut beta_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, seq_len]).expect("alloc beta ref");
        let mut g_ref =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[nv, seq_len]).expect("alloc g ref");

        crate::metal_host::compute_beta_g(
            ScalarType::F32,
            seq_len,
            nv,
            &b,
            &a,
            &dt_bias,
            &a_log_exp,
            &mut beta_ref,
            &mut g_ref,
        )
        .expect("host compute_beta_g");
        compute_beta_g_f32(
            seq_len,
            nv,
            &b,
            &a,
            &dt_bias,
            &a_log_exp,
            &mut beta_native,
            &mut g_native,
        )
        .expect("native compute_beta_g");

        let beta_ref_vals = read_f32(&beta_ref);
        let beta_native_vals = read_f32(&beta_native);
        let g_ref_vals = read_f32(&g_ref);
        let g_native_vals = read_f32(&g_native);
        for (idx, (a, e)) in beta_native_vals
            .iter()
            .zip(beta_ref_vals.iter())
            .enumerate()
        {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-6,
                "beta idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
        for (idx, (a, e)) in g_native_vals.iter().zip(g_ref_vals.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-6,
                "g idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_l2norm_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let input_f32 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[2, 3],
            &f32_bytes(&[3.0f32, 4.0, 0.0, 1.0, 2.0, 2.0]),
        )
        .expect("upload f32 input");
        let mut out_f32 =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("alloc f32 out");
        let mut ref_f32 =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 3]).expect("alloc f32 ref");
        crate::metal_host::l2norm(ScalarType::F32, 2, 3, 1e-6, &input_f32, &mut ref_f32)
            .expect("host f32 l2norm");
        l2norm(ScalarType::F32, 2, 3, 1e-6, &input_f32, &mut out_f32).expect("native f32 l2norm");
        let actual_f32 = read_f32(&out_f32);
        let expect_f32 = read_f32(&ref_f32);
        for (idx, (a, e)) in actual_f32.iter().zip(expect_f32.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-6,
                "f32 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }

        let input_bf16 = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2, 3],
            &bf16_bytes(&[3.0f32, 4.0, 0.0, 1.0, 2.0, 2.0]),
        )
        .expect("upload bf16 input");
        let mut out_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("alloc bf16 out");
        let mut ref_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 3]).expect("alloc bf16 ref");
        crate::metal_host::l2norm(ScalarType::BF16, 2, 3, 1e-6, &input_bf16, &mut ref_bf16)
            .expect("host bf16 l2norm");
        l2norm(ScalarType::BF16, 2, 3, 1e-6, &input_bf16, &mut out_bf16)
            .expect("native bf16 l2norm");
        let actual_bf16 = read_bf16(&out_bf16);
        let expect_bf16 = read_bf16(&ref_bf16);
        for (idx, (a, e)) in actual_bf16.iter().zip(expect_bf16.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 0.01,
                "bf16 idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }

    #[test]
    fn metal_native_delta_recurrent_prefill_f32_matches_reference() {
        set_backend(Backend::Metal);
        let ordinal = 0usize;

        let batch_heads = 2usize;
        let seq_len = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_vals = vec![
            0.1f32, 0.2, 0.3, 0.4, // head0 [k=2,v=2]
            -0.2, 0.5, 0.7, -0.1, // head1
        ];
        let query_vals = vec![
            0.5f32, -0.3, 0.1, 0.2, -0.4, 0.8, // head0 [t=3,k=2]
            0.2, 0.6, -0.7, 0.4, 0.3, -0.5, // head1
        ];
        let key_vals = vec![
            -0.1f32, 0.9, 0.3, -0.2, 0.4, 0.5, // head0
            0.2, -0.6, 0.8, 0.1, -0.3, 0.7, // head1
        ];
        let value_vals = vec![
            0.3f32, -0.4, 0.9, 0.2, -0.5, 0.7, // head0 [t=3,v=2]
            0.6, -0.1, -0.2, 0.4, 0.8, -0.9, // head1
        ];
        let beta_vals = vec![0.2f32, 0.5, 0.7, 0.4, 0.6, 0.3];
        let g_vals = vec![-0.3f32, 0.1, -0.2, 0.0, -0.4, 0.2];

        let initial_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, k_head_dim, v_head_dim],
            &f32_bytes(&initial_state_vals),
        )
        .expect("upload initial_state");
        let query = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, seq_len, k_head_dim],
            &f32_bytes(&query_vals),
        )
        .expect("upload query");
        let key = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, seq_len, k_head_dim],
            &f32_bytes(&key_vals),
        )
        .expect("upload key");
        let value = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, seq_len, v_head_dim],
            &f32_bytes(&value_vals),
        )
        .expect("upload value");
        let beta = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, seq_len],
            &f32_bytes(&beta_vals),
        )
        .expect("upload beta");
        let g = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[batch_heads, seq_len],
            &f32_bytes(&g_vals),
        )
        .expect("upload g");

        let out_rows = seq_len + k_head_dim;
        let mut out_native = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[batch_heads, out_rows, v_head_dim],
        )
        .expect("alloc native out");
        let mut out_ref = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[batch_heads, out_rows, v_head_dim],
        )
        .expect("alloc ref out");

        crate::metal_host::delta_recurrent_prefill(
            ScalarType::F32,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            &mut out_ref,
        )
        .expect("host delta recurrent");
        delta_recurrent_prefill_f32(
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            &mut out_native,
        )
        .expect("native delta recurrent");

        let ref_vals = read_f32(&out_ref);
        let native_vals = read_f32(&out_native);
        for (idx, (a, e)) in native_vals.iter().zip(ref_vals.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= 1e-5,
                "idx {idx}: expected {e}, got {a}, delta {delta}"
            );
        }
    }
}
