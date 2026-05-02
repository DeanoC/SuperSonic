//! FFI surface for the Gemma 4 decode primitives (`kernels/gemma4.hip`).
//!
//! Kept separate from the Qwen FFI (`ffi.rs`) because Gemma 4 has a different
//! layer shape (four RMSNorms per block, dual RoPE tables, optional PLE) and
//! its kernels are in their own compilation unit for hipcc codegen stability.
//!
//! Every wrapper here is intentionally a single-kernel launch — the goal for
//! the first correctness milestone is layer-by-layer validation against the
//! PyTorch oracle, not a fused persistent megakernel. Fusion comes later.

use std::ffi::{c_int, c_uint, c_void};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

#[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
unsafe extern "C" {
    #[cfg_attr(supersonic_backend_cuda, link_name = "supersonic_gemma4_cuda_rms_norm")]
    fn supersonic_gemma4_hip_rms_norm(
        dtype: c_int,
        device_ordinal: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(supersonic_backend_cuda, link_name = "supersonic_gemma4_cuda_matvec")]
    fn supersonic_gemma4_hip_matvec(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_gelu_tanh_gate_mul"
    )]
    fn supersonic_gemma4_hip_gelu_tanh_gate_mul(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_rope_decode"
    )]
    fn supersonic_gemma4_hip_rope_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        x: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_swa_attn_decode"
    )]
    fn supersonic_gemma4_hip_swa_attn_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_t: usize,
        sliding_window: c_int,
        scale: f32,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        scores_scratch: *mut c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_kv_append"
    )]
    fn supersonic_gemma4_hip_kv_append(
        dtype: c_int,
        device_ordinal: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos: usize,
        max_t: usize,
        k_in: *const c_void,
        v_in: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_rms_norm_rows"
    )]
    fn supersonic_gemma4_hip_rms_norm_rows(
        dtype: c_int,
        device_ordinal: usize,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
        xs: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_matvec_batched"
    )]
    fn supersonic_gemma4_hip_matvec_batched(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_matvec_int4"
    )]
    fn supersonic_gemma4_hip_matvec_int4(
        dtype: c_int,
        device_ordinal: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        x: *const c_void,
        w_packed: *const c_void,
        w_scale: *const c_void,
        w_zero: *const c_void,
        out: *mut c_void,
        row_counter: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_matvec_batched_int4"
    )]
    fn supersonic_gemma4_hip_matvec_batched_int4(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        x: *const c_void,
        w_packed: *const c_void,
        w_scale: *const c_void,
        w_zero: *const c_void,
        out: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_rope_prefill"
    )]
    fn supersonic_gemma4_hip_rope_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_base: usize,
        cos_table: *const c_void,
        sin_table: *const c_void,
        x: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_kv_append_prefill"
    )]
    fn supersonic_gemma4_hip_kv_append_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_base: usize,
        max_t: usize,
        k_in: *const c_void,
        v_in: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_attn_prefill"
    )]
    fn supersonic_gemma4_hip_attn_prefill(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_base: usize,
        max_t: usize,
        sliding_window: c_int,
        scale: f32,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        scores_scratch: *mut c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_add_residual"
    )]
    fn supersonic_gemma4_hip_add_residual(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_add_scaled_residual"
    )]
    fn supersonic_gemma4_hip_add_scaled_residual(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        scalar: f32,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_scalar_mul_inplace"
    )]
    fn supersonic_gemma4_hip_scalar_mul_inplace(
        dtype: c_int,
        device_ordinal: usize,
        n: usize,
        scalar: f32,
        x: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_fused_attn_block"
    )]
    fn supersonic_gemma4_hip_fused_attn_block(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        max_t: usize,
        sliding_window: c_int,
        shared_kv: c_int,
        eps: f32,
        scale: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        input_norm_w: *const c_void,
        q_proj_w: *const c_void,
        k_proj_w: *const c_void,
        v_proj_w: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_w: *const c_void,
        post_attn_norm_w: *const c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_fused_attn_block_int4"
    )]
    fn supersonic_gemma4_hip_fused_attn_block_int4(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        max_t: usize,
        sliding_window: c_int,
        shared_kv: c_int,
        group_size: c_int,
        eps: f32,
        scale: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        input_norm_w: *const c_void,
        q_proj_packed: *const c_void,
        q_proj_scale: *const c_void,
        q_proj_zero: *const c_void,
        k_proj_packed: *const c_void,
        k_proj_scale: *const c_void,
        k_proj_zero: *const c_void,
        v_proj_packed: *const c_void,
        v_proj_scale: *const c_void,
        v_proj_zero: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_packed: *const c_void,
        o_proj_scale: *const c_void,
        o_proj_zero: *const c_void,
        post_attn_norm_w: *const c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_fused_mlp_ple_int4"
    )]
    fn supersonic_gemma4_hip_fused_mlp_ple_int4(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        intermediate_size: usize,
        ple_hidden: usize,
        group_size: c_int,
        eps: f32,
        layer_scalar: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        pre_ff_norm_w: *const c_void,
        gate_proj_packed: *const c_void,
        gate_proj_scale: *const c_void,
        gate_proj_zero: *const c_void,
        up_proj_packed: *const c_void,
        up_proj_scale: *const c_void,
        up_proj_zero: *const c_void,
        down_proj_packed: *const c_void,
        down_proj_scale: *const c_void,
        down_proj_zero: *const c_void,
        post_ff_norm_w: *const c_void,
        per_layer_input: *const c_void,
        per_layer_input_gate_packed: *const c_void,
        per_layer_input_gate_scale: *const c_void,
        per_layer_input_gate_zero: *const c_void,
        per_layer_projection_packed: *const c_void,
        per_layer_projection_scale: *const c_void,
        per_layer_projection_zero: *const c_void,
        post_per_layer_input_norm_w: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_fused_mlp_ple"
    )]
    fn supersonic_gemma4_hip_fused_mlp_ple(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        intermediate_size: usize,
        ple_hidden: usize,
        eps: f32,
        layer_scalar: f32,
        hidden_in: *const c_void,
        hidden_out: *mut c_void,
        pre_ff_norm_w: *const c_void,
        gate_proj_w: *const c_void,
        up_proj_w: *const c_void,
        down_proj_w: *const c_void,
        post_ff_norm_w: *const c_void,
        per_layer_input: *const c_void,
        per_layer_input_gate_w: *const c_void,
        per_layer_projection_w: *const c_void,
        post_per_layer_input_norm_w: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_gather_layer_slice"
    )]
    fn supersonic_gemma4_hip_gather_layer_slice(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        num_layers: usize,
        ple_hidden: usize,
        layer_idx: usize,
        src: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_embed_gather_scaled"
    )]
    fn supersonic_gemma4_hip_embed_gather_scaled(
        dtype: c_int,
        device_ordinal: usize,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        scale: f32,
        token_ids: *const c_uint,
        table: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_persistent_decode_int4"
    )]
    fn supersonic_gemma4_hip_persistent_decode_int4(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        position: usize,
        eps: f32,
        scale: f32,
        layers: *const c_void,
        int4_scales: *const c_void,
        hidden_io: *mut c_void,
        per_layer_inputs: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_persistent_decode"
    )]
    fn supersonic_gemma4_hip_persistent_decode(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        position: usize,
        eps: f32,
        scale: f32,
        layers: *const c_void,
        kv_fp8_descs: *const c_void,
        fp8_scales: *const c_void,
        hidden_io: *mut c_void,
        per_layer_inputs: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    #[link_name = "supersonic_gemma4_cuda_persistent_decode_fused_input"]
    fn supersonic_gemma4_cuda_persistent_decode_fused_input(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        vocab_size: usize,
        token_id: c_uint,
        position: usize,
        eps: f32,
        scale: f32,
        embed_scale: f32,
        proj_scale: f32,
        ple_scale: f32,
        combine_scale: f32,
        layers: *const c_void,
        kv_fp8_descs: *const c_void,
        fp8_scales: *const c_void,
        embed_tokens: *const c_void,
        embed_tokens_per_layer: *const c_void,
        per_layer_model_projection_w: *const c_void,
        per_layer_projection_norm_w: *const c_void,
        hidden_io: *mut c_void,
        pli_proj: *mut c_void,
        pli_normed: *mut c_void,
        ple_raw: *mut c_void,
        per_layer_inputs: *mut c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    #[link_name = "supersonic_gemma4_cuda_persistent_decode_fused_input_argmax"]
    fn supersonic_gemma4_cuda_persistent_decode_fused_input_argmax(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        vocab_size: usize,
        token_id: c_uint,
        position: usize,
        eps: f32,
        scale: f32,
        embed_scale: f32,
        proj_scale: f32,
        ple_scale: f32,
        combine_scale: f32,
        layers: *const c_void,
        embed_tokens: *const c_void,
        embed_tokens_per_layer: *const c_void,
        per_layer_model_projection_w: *const c_void,
        per_layer_projection_norm_w: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        hidden_io: *mut c_void,
        pli_proj: *mut c_void,
        pli_normed: *mut c_void,
        ple_raw: *mut c_void,
        per_layer_inputs: *mut c_void,
        workspace: *mut c_void,
        out_token: *mut c_uint,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg(supersonic_backend_cuda)]
    #[link_name = "supersonic_gemma4_cuda_final_norm_lm_head_argmax"]
    fn supersonic_gemma4_cuda_final_norm_lm_head_argmax(
        dtype: c_int,
        device_ordinal: usize,
        hidden_size: usize,
        vocab_size: usize,
        eps: f32,
        hidden_io: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        workspace: *mut c_void,
        out_token: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_persistent_decode_batch"
    )]
    fn supersonic_gemma4_hip_persistent_decode_batch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        eps: f32,
        scale: f32,
        batch_size: usize,
        ws_stride: usize,
        layers: *const c_void,
        batch_descs: *const c_void,
        hidden_io: *mut c_void,
        per_layer_inputs: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    #[cfg_attr(
        supersonic_backend_cuda,
        link_name = "supersonic_gemma4_cuda_persistent_decode_batch_int4"
    )]
    fn supersonic_gemma4_hip_persistent_decode_batch_int4(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        hidden_size: usize,
        ple_hidden: usize,
        eps: f32,
        scale: f32,
        batch_size: usize,
        ws_stride: usize,
        layers: *const c_void,
        int4_scales: *const c_void,
        batch_descs: *const c_void,
        hidden_io: *mut c_void,
        per_layer_inputs: *const c_void,
        workspace: *mut c_void,
        matvec_counter: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;
}

#[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
macro_rules! gemma4_stub {
    ($(fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> c_int;)+) => {
        $(
            #[no_mangle]
            unsafe extern "C" fn $name($($arg: $ty),*) -> c_int {
                let _ = ($($arg),*);
                1
            }
        )+
    };
}

#[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
gemma4_stub! {
    fn supersonic_gemma4_hip_rms_norm(dtype: c_int, device_ordinal: usize, n_cols: usize, eps: f32, xs: *const c_void, weight: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_matvec(dtype: c_int, device_ordinal: usize, in_dim: usize, out_dim: usize, x: *const c_void, w: *const c_void, out: *mut c_void, row_counter: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_gelu_tanh_gate_mul(dtype: c_int, device_ordinal: usize, n: usize, gate: *const c_void, up: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_rope_decode(dtype: c_int, device_ordinal: usize, num_heads: usize, head_dim: usize, rotary_dim: usize, position: usize, cos_table: *const c_void, sin_table: *const c_void, x: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_swa_attn_decode(dtype: c_int, device_ordinal: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize, kv_len: usize, max_t: usize, sliding_window: c_int, scale: f32, q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, scores_scratch: *mut c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_kv_append(dtype: c_int, device_ordinal: usize, num_kv_heads: usize, head_dim: usize, pos: usize, max_t: usize, k_in: *const c_void, v_in: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_rms_norm_rows(dtype: c_int, device_ordinal: usize, n_rows: usize, n_cols: usize, eps: f32, xs: *const c_void, weight: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_matvec_batched(dtype: c_int, device_ordinal: usize, seq_len: usize, in_dim: usize, out_dim: usize, x: *const c_void, w: *const c_void, out: *mut c_void, counter: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_matvec_int4(dtype: c_int, device_ordinal: usize, in_dim: usize, out_dim: usize, group_size: usize, x: *const c_void, w_packed: *const c_void, w_scale: *const c_void, w_zero: *const c_void, out: *mut c_void, row_counter: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_matvec_batched_int4(dtype: c_int, device_ordinal: usize, seq_len: usize, in_dim: usize, out_dim: usize, group_size: usize, x: *const c_void, w_packed: *const c_void, w_scale: *const c_void, w_zero: *const c_void, out: *mut c_void, counter: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_rope_prefill(dtype: c_int, device_ordinal: usize, seq_len: usize, num_heads: usize, head_dim: usize, rotary_dim: usize, pos_base: usize, cos_table: *const c_void, sin_table: *const c_void, x: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_kv_append_prefill(dtype: c_int, device_ordinal: usize, seq_len: usize, num_kv_heads: usize, head_dim: usize, pos_base: usize, max_t: usize, k_in: *const c_void, v_in: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_attn_prefill(dtype: c_int, device_ordinal: usize, seq_len: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize, pos_base: usize, max_t: usize, sliding_window: c_int, scale: f32, q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, scores_scratch: *mut c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_add_residual(dtype: c_int, device_ordinal: usize, n: usize, a: *const c_void, b: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_add_scaled_residual(dtype: c_int, device_ordinal: usize, n: usize, scalar: f32, a: *const c_void, b: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_scalar_mul_inplace(dtype: c_int, device_ordinal: usize, n: usize, scalar: f32, x: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_fused_attn_block(dtype: c_int, device_ordinal: usize, hidden_size: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize, rotary_dim: usize, position: usize, max_t: usize, sliding_window: c_int, shared_kv: c_int, eps: f32, scale: f32, hidden_in: *const c_void, hidden_out: *mut c_void, input_norm_w: *const c_void, q_proj_w: *const c_void, k_proj_w: *const c_void, v_proj_w: *const c_void, q_norm_w: *const c_void, k_norm_w: *const c_void, o_proj_w: *const c_void, post_attn_norm_w: *const c_void, cos_table: *const c_void, sin_table: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_fused_attn_block_int4(dtype: c_int, device_ordinal: usize, hidden_size: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize, rotary_dim: usize, position: usize, max_t: usize, sliding_window: c_int, shared_kv: c_int, group_size: c_int, eps: f32, scale: f32, hidden_in: *const c_void, hidden_out: *mut c_void, input_norm_w: *const c_void, q_proj_packed: *const c_void, q_proj_scale: *const c_void, q_proj_zero: *const c_void, k_proj_packed: *const c_void, k_proj_scale: *const c_void, k_proj_zero: *const c_void, v_proj_packed: *const c_void, v_proj_scale: *const c_void, v_proj_zero: *const c_void, q_norm_w: *const c_void, k_norm_w: *const c_void, o_proj_packed: *const c_void, o_proj_scale: *const c_void, o_proj_zero: *const c_void, post_attn_norm_w: *const c_void, cos_table: *const c_void, sin_table: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_fused_mlp_ple_int4(dtype: c_int, device_ordinal: usize, hidden_size: usize, intermediate_size: usize, ple_hidden: usize, group_size: c_int, eps: f32, layer_scalar: f32, hidden_in: *const c_void, hidden_out: *mut c_void, pre_ff_norm_w: *const c_void, gate_proj_packed: *const c_void, gate_proj_scale: *const c_void, gate_proj_zero: *const c_void, up_proj_packed: *const c_void, up_proj_scale: *const c_void, up_proj_zero: *const c_void, down_proj_packed: *const c_void, down_proj_scale: *const c_void, down_proj_zero: *const c_void, post_ff_norm_w: *const c_void, per_layer_input: *const c_void, per_layer_input_gate_packed: *const c_void, per_layer_input_gate_scale: *const c_void, per_layer_input_gate_zero: *const c_void, per_layer_projection_packed: *const c_void, per_layer_projection_scale: *const c_void, per_layer_projection_zero: *const c_void, post_per_layer_input_norm_w: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_fused_mlp_ple(dtype: c_int, device_ordinal: usize, hidden_size: usize, intermediate_size: usize, ple_hidden: usize, eps: f32, layer_scalar: f32, hidden_in: *const c_void, hidden_out: *mut c_void, pre_ff_norm_w: *const c_void, gate_proj_w: *const c_void, up_proj_w: *const c_void, down_proj_w: *const c_void, post_ff_norm_w: *const c_void, per_layer_input: *const c_void, per_layer_input_gate_w: *const c_void, per_layer_projection_w: *const c_void, post_per_layer_input_norm_w: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_gather_layer_slice(dtype: c_int, device_ordinal: usize, seq_len: usize, num_layers: usize, ple_hidden: usize, layer_idx: usize, src: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_embed_gather_scaled(dtype: c_int, device_ordinal: usize, seq_len: usize, hidden_size: usize, vocab_size: usize, scale: f32, token_ids: *const c_uint, table: *const c_void, out: *mut c_void) -> c_int;
    fn supersonic_gemma4_hip_persistent_decode_int4(dtype: c_int, device_ordinal: usize, num_layers: usize, hidden_size: usize, ple_hidden: usize, position: usize, eps: f32, scale: f32, layers: *const c_void, int4_scales: *const c_void, hidden_io: *mut c_void, per_layer_inputs: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_persistent_decode(dtype: c_int, device_ordinal: usize, num_layers: usize, hidden_size: usize, ple_hidden: usize, position: usize, eps: f32, scale: f32, layers: *const c_void, kv_fp8_descs: *const c_void, fp8_scales: *const c_void, hidden_io: *mut c_void, per_layer_inputs: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_persistent_decode_batch(dtype: c_int, device_ordinal: usize, num_layers: usize, hidden_size: usize, ple_hidden: usize, eps: f32, scale: f32, batch_size: usize, ws_stride: usize, layers: *const c_void, batch_descs: *const c_void, hidden_io: *mut c_void, per_layer_inputs: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
    fn supersonic_gemma4_hip_persistent_decode_batch_int4(dtype: c_int, device_ordinal: usize, num_layers: usize, hidden_size: usize, ple_hidden: usize, eps: f32, scale: f32, batch_size: usize, ws_stride: usize, layers: *const c_void, int4_scales: *const c_void, batch_descs: *const c_void, hidden_io: *mut c_void, per_layer_inputs: *const c_void, workspace: *mut c_void, matvec_counter: *mut c_uint, barrier_counter: *mut c_uint, barrier_flag: *mut c_uint) -> c_int;
}

/// Gemma-variant RMSNorm — plain `weight * (x / sqrt(mean(x^2) + eps))` with
/// no `(w + 1)` offset (unlike Qwen's). Pass a null `weight` pointer via
/// `weight.is_none()` for `with_scale=False` (used by Gemma 4's `v_norm`).
pub fn rms_norm(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        supersonic_gemma4_hip_rms_norm(
            dtype.kernel_dtype_code(),
            ordinal,
            n_cols,
            eps,
            input.as_ptr(),
            weight_ptr,
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 rms_norm failed with status {status}"),
        ));
    }
    Ok(())
}

/// Apply the Gemma RMSNorm to each of `num_rows` rows of a packed
/// `[num_rows, n_cols]` tensor independently, using the same `weight` vector
/// (or `None` for with_scale=False) for every row.
///
/// Implemented as a serial launch loop — one kernel launch per row, each
/// handling `n_cols` scalars. The underlying kernel is a single-block launch
/// that normalizes exactly one row, so this helper advances the input/output
/// pointers by `n_cols * sizeof(T)` bytes per iteration. Fine for the Gemma
/// 4 layer-0 validator (num_heads ≤ 8, n_cols=256), correctness before fusion.
pub fn rms_norm_per_row(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    num_rows: usize,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let row_bytes = n_cols * dtype.size_in_bytes();
    let in_base = input.as_ptr() as *const u8;
    let out_base = output.as_mut_ptr() as *mut u8;
    for row in 0..num_rows {
        let offset = row * row_bytes;
        let in_ptr = unsafe { in_base.add(offset) as *const c_void };
        let out_ptr = unsafe { out_base.add(offset) as *mut c_void };
        let status = unsafe {
            supersonic_gemma4_hip_rms_norm(
                dtype.kernel_dtype_code(),
                ordinal,
                n_cols,
                eps,
                in_ptr,
                weight_ptr,
                out_ptr,
            )
        };
        if status != 0 {
            return Err(GpuError::backend(
                Backend::Hip,
                format!("gemma4 rms_norm_per_row failed at row {row} with status {status}"),
            ));
        }
    }
    Ok(())
}

/// Single-token matvec: `out = W @ x` where `W` is `[out_dim, in_dim]` row-major.
/// `counter_buf` must hold ≥4 mutable bytes — it's reset to 0 inside the call
/// and drives the work-stealing row assignment.
pub fn matvec(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_matvec(
            dtype.kernel_dtype_code(),
            ordinal,
            in_dim,
            out_dim,
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
            counter_buf.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 matvec failed with status {status}"),
        ));
    }
    Ok(())
}

/// INT4 matvec: `out = dequant(W_packed, W_scale, W_zero) @ input`. The weight
/// format matches the Gemma 4 GPTQ bake (packed u8 [out_dim, in_dim/2], bf16
/// scale/zero [out_dim/group_size, in_dim/group_size]). `input` and `output`
/// use `dtype` (typically BF16). Work-stealing across output rows is driven by
/// `counter_buf` — the kernel zeroes it before the launch.
pub fn matvec_int4(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    w_packed: &GpuBuffer,
    w_scale: &GpuBuffer,
    w_zero: &GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_matvec_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            in_dim,
            out_dim,
            group_size,
            input.as_ptr(),
            w_packed.as_ptr(),
            w_scale.as_ptr(),
            w_zero.as_ptr(),
            output.as_mut_ptr(),
            counter_buf.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 matvec_int4 failed with status {status}"),
        ));
    }
    Ok(())
}

/// Batched INT4 matvec: `out[s, r] = dequant(W[r, :]) · input[s, :]` for every
/// (s, r) pair, with work-stealing over (s, r) items via `counter_buf`.
pub fn matvec_batched_int4(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    w_packed: &GpuBuffer,
    w_scale: &GpuBuffer,
    w_zero: &GpuBuffer,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_matvec_batched_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            in_dim,
            out_dim,
            group_size,
            input.as_ptr(),
            w_packed.as_ptr(),
            w_scale.as_ptr(),
            w_zero.as_ptr(),
            output.as_mut_ptr(),
            counter_buf.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 matvec_batched_int4 failed with status {status}"),
        ));
    }
    Ok(())
}

/// Elementwise `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]`. Matches Gemma 4's
/// `hidden_activation = "gelu_pytorch_tanh"` exactly.
pub fn gelu_tanh_gate_mul(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_gelu_tanh_gate_mul(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            gate.as_ptr(),
            up.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 gelu_tanh_gate_mul failed with status {status}"),
        ));
    }
    Ok(())
}

/// Apply Gemma-style (split-half) RoPE to a single decode token's Q or K tensor
/// of shape `[num_heads, head_dim]`, using position `pos` from the cos/sin table.
/// `rotary_dim` may be less than `head_dim` for partial rotation (Gemma 4 full-
/// attention layers use `partial_rotary_factor=0.25`).
pub fn rope_decode(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_rope_decode(
            dtype.kernel_dtype_code(),
            ordinal,
            num_heads,
            head_dim,
            rotary_dim,
            position,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 rope_decode failed with status {status}"),
        ));
    }
    Ok(())
}

/// Run sliding-window attention for one decode token.
///
/// The caller is responsible for having already appended the current token's
/// K and V into the caches (see [`kv_append`]) so that `kv_len` entries are
/// valid. `scores_scratch` must hold at least `num_q_heads * max_t * 4` bytes
/// of F32 storage; it is written and read inside the kernel but its contents
/// are not meaningful to the caller afterwards.
///
/// Pass `sliding_window <= 0` to attend over the whole cache (behaves as full
/// attention). Gemma 4 uses `scale = 1.0` (no 1/sqrt(d_k)).
#[allow(clippy::too_many_arguments)]
pub fn swa_attn_decode(
    ordinal: usize,
    dtype: ScalarType,
    q: &GpuBuffer,
    k_cache: &GpuBuffer,
    v_cache: &GpuBuffer,
    scores_scratch: &mut GpuBuffer,
    out: &mut GpuBuffer,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    max_t: usize,
    sliding_window: i32,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_swa_attn_decode(
            dtype.kernel_dtype_code(),
            ordinal,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            max_t,
            sliding_window as c_int,
            scale,
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            scores_scratch.as_mut_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 swa_attn_decode failed with status {status}"),
        ));
    }
    Ok(())
}

/// Append a single decode token's K and V into pre-allocated caches of layout
/// `[num_kv_heads, max_T, head_dim]`. `pos` is the absolute position in the
/// cache (= token index in the sequence so far). Does not bounds-check pos
/// against `max_T`; caller must ensure capacity.
pub fn kv_append(
    ordinal: usize,
    dtype: ScalarType,
    k_in: &GpuBuffer,
    v_in: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_t: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_kv_append(
            dtype.kernel_dtype_code(),
            ordinal,
            num_kv_heads,
            head_dim,
            pos,
            max_t,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 kv_append failed with status {status}"),
        ));
    }
    Ok(())
}

// =============================================================================
// Prefill / batched primitives (Step 13).
//
// Each wrapper below mirrors its single-token counterpart above but takes a
// `seq_len` (or `n_rows`) parameter so the kernel launch processes the whole
// batch in one shot. Only `gemma4_e2e_validate`'s Phase A consumes them today;
// the decode path in `gemma4_decode_validate` continues to use the
// single-token primitives unchanged.
// =============================================================================

/// Multi-row Gemma RMSNorm. Normalizes each row of a `[n_rows, n_cols]` tensor
/// independently using the same `weight[n_cols]` (or `None` for
/// `with_scale=False`). Dispatches to a single kernel launch (grid=n_rows).
pub fn rms_norm_rows(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    eps: f32,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), GpuError> {
    let weight_ptr = weight.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        supersonic_gemma4_hip_rms_norm_rows(
            dtype.kernel_dtype_code(),
            ordinal,
            n_rows,
            n_cols,
            eps,
            input.as_ptr(),
            weight_ptr,
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 rms_norm_rows failed with status {status}"),
        ));
    }
    Ok(())
}

/// Batched matvec `out[s, r] = dot(W[r, :], in[s, :])` computed for all (s, r).
/// Single kernel launch with work-stealing over `seq_len * out_dim` items.
pub fn matvec_batched(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_matvec_batched(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            in_dim,
            out_dim,
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
            counter_buf.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 matvec_batched failed with status {status}"),
        ));
    }
    Ok(())
}

/// Apply Gemma split-half RoPE to every token in a `[seq_len, num_heads,
/// head_dim]` tensor, with token `s` using position `pos_base + s` into the
/// shared cos/sin tables.
#[allow(clippy::too_many_arguments)]
pub fn rope_prefill(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos_base: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_rope_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_heads,
            head_dim,
            rotary_dim,
            pos_base,
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 rope_prefill failed with status {status}"),
        ));
    }
    Ok(())
}

/// Write a `[seq_len, num_kv_heads, head_dim]` K/V tensor into cache slots
/// `[pos_base, pos_base+seq_len)` of a `[num_kv_heads, max_t, head_dim]` cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_prefill(
    ordinal: usize,
    dtype: ScalarType,
    k_in: &GpuBuffer,
    v_in: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_base: usize,
    max_t: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_kv_append_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_kv_heads,
            head_dim,
            pos_base,
            max_t,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 kv_append_prefill failed with status {status}"),
        ));
    }
    Ok(())
}

/// Prefill-style SWA/full attention over `seq_len` query tokens. The cache
/// must already contain `pos_base + seq_len` valid entries (fill via
/// `kv_append_prefill`). `scores_scratch` needs at least
/// `seq_len * num_q_heads * max_t * 4` bytes of F32 storage.
///
/// Output layout: `[seq_len, num_q_heads, head_dim]`.
/// Pass `sliding_window <= 0` for full attention.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    ordinal: usize,
    dtype: ScalarType,
    q: &GpuBuffer,
    k_cache: &GpuBuffer,
    v_cache: &GpuBuffer,
    scores_scratch: &mut GpuBuffer,
    out: &mut GpuBuffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_base: usize,
    max_t: usize,
    sliding_window: i32,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_attn_prefill(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_base,
            max_t,
            sliding_window as c_int,
            scale,
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            scores_scratch.as_mut_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 attn_prefill failed with status {status}"),
        ));
    }
    Ok(())
}

/// Elementwise residual add `out[i] = a[i] + b[i]` over `n` scalars.
pub fn add_residual(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_add_residual(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 add_residual failed with status {status}"),
        ));
    }
    Ok(())
}

/// Elementwise `out[i] = (a[i] + b[i]) * scalar` for the Gemma 4 PLE residual
/// "(h_pre_ple + normed) * layer_scalar" step.
pub fn add_scaled_residual(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    a: &GpuBuffer,
    b: &GpuBuffer,
    scalar: f32,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_add_scaled_residual(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            scalar,
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 add_scaled_residual failed with status {status}"),
        ));
    }
    Ok(())
}

/// In-place scalar multiply `x[i] *= scalar`. Use after a matvec when the
/// output needs a constant BF16-rounded multiplier applied (mirrors HF's
/// Python-float times BF16-tensor rounding — the caller should pass the
/// scalar pre-rounded to BF16 on the host).
pub fn scalar_mul_inplace(
    ordinal: usize,
    dtype: ScalarType,
    x: &mut GpuBuffer,
    scalar: f32,
    n: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_scalar_mul_inplace(
            dtype.kernel_dtype_code(),
            ordinal,
            n,
            scalar,
            x.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 scalar_mul_inplace failed with status {status}"),
        ));
    }
    Ok(())
}

/// Required F32 workspace (elements) for `fused_attn_block`. Layout matches
/// the kernel: hidden + normed + proj (q+2kv) + scores (nq*max_t) + attn_out
/// (q_dim) + oproj + oproj_normed.
pub fn fused_attn_block_workspace_elems(
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_t: usize,
) -> usize {
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    2 * hidden_size + q_dim + 2 * kv_dim + num_q_heads * max_t + q_dim + 2 * hidden_size
}

/// Run one Gemma 4 decoder layer's attention half (input_norm → QKV → qk/v
/// norm → RoPE → kv_append → SWA/full attention → o_proj → post_attn_norm
/// → residual) in a single kernel launch. `shared_kv` skips K/V generation
/// and assumes the cache at slot `position` already holds the inherited K/V.
///
/// `workspace` must be at least `fused_attn_block_workspace_elems(...) * 4`
/// bytes (F32). The three counter buffers are each ≥4 bytes of U32 storage;
/// the kernel clears the barrier pair before launch and reuses
/// `matvec_counter` internally between the QKV and o_proj phases.
#[allow(clippy::too_many_arguments)]
pub fn fused_attn_block(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    input_norm_w: &GpuBuffer,
    q_proj_w: &GpuBuffer,
    k_proj_w: Option<&GpuBuffer>,
    v_proj_w: Option<&GpuBuffer>,
    q_norm_w: &GpuBuffer,
    k_norm_w: Option<&GpuBuffer>,
    o_proj_w: &GpuBuffer,
    post_attn_norm_w: &GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    sliding_window: i32,
    position: usize,
    max_t: usize,
    shared_kv: bool,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    let null = std::ptr::null();
    let k_proj_ptr = k_proj_w.map(|b| b.as_ptr()).unwrap_or(null);
    let v_proj_ptr = v_proj_w.map(|b| b.as_ptr()).unwrap_or(null);
    let k_norm_ptr = k_norm_w.map(|b| b.as_ptr()).unwrap_or(null);
    let status = unsafe {
        supersonic_gemma4_hip_fused_attn_block(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            position,
            max_t,
            sliding_window as c_int,
            if shared_kv { 1 } else { 0 },
            eps,
            scale,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            input_norm_w.as_ptr(),
            q_proj_w.as_ptr(),
            k_proj_ptr,
            v_proj_ptr,
            q_norm_w.as_ptr(),
            k_norm_ptr,
            o_proj_w.as_ptr(),
            post_attn_norm_w.as_ptr(),
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 fused_attn_block failed with status {status}"),
        ));
    }
    Ok(())
}

/// INT4 version of [`fused_attn_block`]. Same orchestration as the BF16 path
/// (input_norm → QKV → qk/v norm → RoPE → kv_append → SWA/full attention →
/// o_proj → post_attn_norm → residual) but the four projections (Q, K, V, O)
/// are INT4-dequantized on the fly from `(packed u8, BF16 scale, BF16 zero)`
/// triples at `group_size=128` (the only size the Gemma 4 GPTQ bake emits).
///
/// Shape invariants:
///   - `q_proj_packed`  shape `[num_q_heads * head_dim, hidden_size / 2]`
///   - `k_proj_packed` / `v_proj_packed` shape `[num_kv_heads * head_dim, hidden_size / 2]`
///   - `o_proj_packed` shape `[hidden_size, (num_q_heads * head_dim) / 2]`
///   - Every scale/zero tensor shape `[out_dim / group_size, in_dim / group_size]`
///
/// `k_proj_*`, `v_proj_*`, and `k_norm_w` must be `Some` unless `shared_kv` is
/// true — shared-KV layers skip K/V computation and inherit from the source
/// layer's cache, same as the BF16 path.
///
/// Workspace sizing, counter/barrier semantics, and the `shared_kv` toggle are
/// identical to [`fused_attn_block`]. This wrapper exists solely to route the
/// Q/K/V/O matmuls through the INT4 work-stealing inner loop.
#[allow(clippy::too_many_arguments)]
pub fn fused_attn_block_int4(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    input_norm_w: &GpuBuffer,
    q_proj_packed: &GpuBuffer,
    q_proj_scale: &GpuBuffer,
    q_proj_zero: &GpuBuffer,
    k_proj_packed: Option<&GpuBuffer>,
    k_proj_scale: Option<&GpuBuffer>,
    k_proj_zero: Option<&GpuBuffer>,
    v_proj_packed: Option<&GpuBuffer>,
    v_proj_scale: Option<&GpuBuffer>,
    v_proj_zero: Option<&GpuBuffer>,
    q_norm_w: &GpuBuffer,
    k_norm_w: Option<&GpuBuffer>,
    o_proj_packed: &GpuBuffer,
    o_proj_scale: &GpuBuffer,
    o_proj_zero: &GpuBuffer,
    post_attn_norm_w: &GpuBuffer,
    cos_table: &GpuBuffer,
    sin_table: &GpuBuffer,
    k_cache: &mut GpuBuffer,
    v_cache: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    sliding_window: i32,
    position: usize,
    max_t: usize,
    shared_kv: bool,
    group_size: usize,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    let null = std::ptr::null();
    let k_packed_ptr = k_proj_packed.map(|b| b.as_ptr()).unwrap_or(null);
    let k_scale_ptr = k_proj_scale.map(|b| b.as_ptr()).unwrap_or(null);
    let k_zero_ptr = k_proj_zero.map(|b| b.as_ptr()).unwrap_or(null);
    let v_packed_ptr = v_proj_packed.map(|b| b.as_ptr()).unwrap_or(null);
    let v_scale_ptr = v_proj_scale.map(|b| b.as_ptr()).unwrap_or(null);
    let v_zero_ptr = v_proj_zero.map(|b| b.as_ptr()).unwrap_or(null);
    let k_norm_ptr = k_norm_w.map(|b| b.as_ptr()).unwrap_or(null);
    let status = unsafe {
        supersonic_gemma4_hip_fused_attn_block_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            position,
            max_t,
            sliding_window as c_int,
            if shared_kv { 1 } else { 0 },
            group_size as c_int,
            eps,
            scale,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            input_norm_w.as_ptr(),
            q_proj_packed.as_ptr(),
            q_proj_scale.as_ptr(),
            q_proj_zero.as_ptr(),
            k_packed_ptr,
            k_scale_ptr,
            k_zero_ptr,
            v_packed_ptr,
            v_scale_ptr,
            v_zero_ptr,
            q_norm_w.as_ptr(),
            k_norm_ptr,
            o_proj_packed.as_ptr(),
            o_proj_scale.as_ptr(),
            o_proj_zero.as_ptr(),
            post_attn_norm_w.as_ptr(),
            cos_table.as_ptr(),
            sin_table.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 fused_attn_block_int4 failed with status {status}"),
        ));
    }
    Ok(())
}

/// INT4 version of [`fused_mlp_ple`]. Runs the second half of a Gemma 4
/// decoder layer (pre_ff_norm → gate/up INT4 → gelu*up → down INT4 →
/// post_ff_norm → residual → per_layer_input_gate INT4 → gelu*ple →
/// per_layer_projection INT4 → post_per_layer_input_norm → (+)*layer_scalar)
/// in a single kernel launch. All 5 projections (`gate`, `up`, `down`,
/// `per_layer_input_gate`, `per_layer_projection`) are INT4-dequantized via
/// the same `(packed u8, BF16 scale, BF16 zero)` / group_size=128 format the
/// attention-block INT4 kernel uses.
///
/// Workspace sizing identical to [`fused_mlp_ple`] — call
/// [`fused_mlp_ple_workspace_elems`] with the layer's `intermediate_size`
/// and `ple_hidden`. Counter/barrier semantics match the BF16 kernel.
#[allow(clippy::too_many_arguments)]
pub fn fused_mlp_ple_int4(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    pre_ff_norm_w: &GpuBuffer,
    gate_proj_packed: &GpuBuffer,
    gate_proj_scale: &GpuBuffer,
    gate_proj_zero: &GpuBuffer,
    up_proj_packed: &GpuBuffer,
    up_proj_scale: &GpuBuffer,
    up_proj_zero: &GpuBuffer,
    down_proj_packed: &GpuBuffer,
    down_proj_scale: &GpuBuffer,
    down_proj_zero: &GpuBuffer,
    post_ff_norm_w: &GpuBuffer,
    per_layer_input: &GpuBuffer,
    per_layer_input_gate_packed: &GpuBuffer,
    per_layer_input_gate_scale: &GpuBuffer,
    per_layer_input_gate_zero: &GpuBuffer,
    per_layer_projection_packed: &GpuBuffer,
    per_layer_projection_scale: &GpuBuffer,
    per_layer_projection_zero: &GpuBuffer,
    post_per_layer_input_norm_w: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    intermediate_size: usize,
    ple_hidden: usize,
    group_size: usize,
    eps: f32,
    layer_scalar: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_fused_mlp_ple_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            intermediate_size,
            ple_hidden,
            group_size as c_int,
            eps,
            layer_scalar,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            pre_ff_norm_w.as_ptr(),
            gate_proj_packed.as_ptr(),
            gate_proj_scale.as_ptr(),
            gate_proj_zero.as_ptr(),
            up_proj_packed.as_ptr(),
            up_proj_scale.as_ptr(),
            up_proj_zero.as_ptr(),
            down_proj_packed.as_ptr(),
            down_proj_scale.as_ptr(),
            down_proj_zero.as_ptr(),
            post_ff_norm_w.as_ptr(),
            per_layer_input.as_ptr(),
            per_layer_input_gate_packed.as_ptr(),
            per_layer_input_gate_scale.as_ptr(),
            per_layer_input_gate_zero.as_ptr(),
            per_layer_projection_packed.as_ptr(),
            per_layer_projection_scale.as_ptr(),
            per_layer_projection_zero.as_ptr(),
            post_per_layer_input_norm_w.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 fused_mlp_ple_int4 failed with status {status}"),
        ));
    }
    Ok(())
}

/// Required F32 workspace (elements) for `fused_mlp_ple`. Matches the
/// kernel's layout: 7 * hidden + 3 * intermediate + 2 * ple_hidden.
pub fn fused_mlp_ple_workspace_elems(
    hidden_size: usize,
    intermediate_size: usize,
    ple_hidden: usize,
) -> usize {
    7 * hidden_size + 3 * intermediate_size + 2 * ple_hidden
}

/// Run one Gemma 4 decoder layer's MLP + PLE half (pre_ff_norm → gate/up
/// proj → gelu*up → down_proj → post_ff_norm → residual → per_layer_input_gate
/// → gelu*per_layer_input → per_layer_projection → post_per_layer_input_norm
/// → (+)*layer_scalar) in a single kernel launch. `hidden_in` is `h_mid`
/// (output of `fused_attn_block`); `hidden_out` is the new `h_running`.
#[allow(clippy::too_many_arguments)]
pub fn fused_mlp_ple(
    ordinal: usize,
    dtype: ScalarType,
    hidden_in: &GpuBuffer,
    hidden_out: &mut GpuBuffer,
    pre_ff_norm_w: &GpuBuffer,
    gate_proj_w: &GpuBuffer,
    up_proj_w: &GpuBuffer,
    down_proj_w: &GpuBuffer,
    post_ff_norm_w: &GpuBuffer,
    per_layer_input: &GpuBuffer,
    per_layer_input_gate_w: &GpuBuffer,
    per_layer_projection_w: &GpuBuffer,
    post_per_layer_input_norm_w: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    intermediate_size: usize,
    ple_hidden: usize,
    eps: f32,
    layer_scalar: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_fused_mlp_ple(
            dtype.kernel_dtype_code(),
            ordinal,
            hidden_size,
            intermediate_size,
            ple_hidden,
            eps,
            layer_scalar,
            hidden_in.as_ptr(),
            hidden_out.as_mut_ptr(),
            pre_ff_norm_w.as_ptr(),
            gate_proj_w.as_ptr(),
            up_proj_w.as_ptr(),
            down_proj_w.as_ptr(),
            post_ff_norm_w.as_ptr(),
            per_layer_input.as_ptr(),
            per_layer_input_gate_w.as_ptr(),
            per_layer_projection_w.as_ptr(),
            post_per_layer_input_norm_w.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 fused_mlp_ple failed with status {status}"),
        ));
    }
    Ok(())
}

/// Extract one layer's `[seq_len, ple_hidden]` slice from a batched
/// `[seq_len, num_layers, ple_hidden]` PLI table. The output buffer is
/// contiguous so the existing elementwise primitives can consume it.
#[allow(clippy::too_many_arguments)]
pub fn gather_layer_slice(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    src: &GpuBuffer,
    seq_len: usize,
    num_layers: usize,
    ple_hidden: usize,
    layer_idx: usize,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_gather_layer_slice(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            num_layers,
            ple_hidden,
            layer_idx,
            src.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 gather_layer_slice failed with status {status}"),
        ));
    }
    Ok(())
}

/// Gather rows from an embedding table and scale by a host-side multiplier.
/// `token_ids` is a device-resident `[seq_len]` u32 buffer; `table` is a
/// `[vocab_size, hidden_size]` row-major embedding. The multiplier is applied
/// in FP32 after the BF16 load (mirroring HF's `embed * sqrt(hidden_size)`).
#[allow(clippy::too_many_arguments)]
pub fn embed_gather_scaled(
    ordinal: usize,
    dtype: ScalarType,
    output: &mut GpuBuffer,
    token_ids: &GpuBuffer,
    table: &GpuBuffer,
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_embed_gather_scaled(
            dtype.kernel_dtype_code(),
            ordinal,
            seq_len,
            hidden_size,
            vocab_size,
            scale,
            token_ids.as_ptr() as *const c_uint,
            table.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 embed_gather_scaled failed with status {status}"),
        ));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Gemma4DecodeLayerDesc — one decoder layer's weight pointers + per-layer
// state. Consumed by the persistent decode megakernel
// (`g4_persistent_decode_kernel`), which takes a contiguous `[num_layers]`
// array and loops over layers inside a single kernel launch. `#[repr(C)]`
// with explicit field order so the HIP-side struct in `kernels/gemma4.hip`
// stays binary-compatible.
//
// Shared-KV layers set `shared_kv = 1` and alias `kv_cache_k` / `kv_cache_v`
// to their source layer's pointers — no intra-kernel replication needed.
// -----------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4DecodeLayerDesc {
    /// 0 = sliding_attention, 1 = full_attention. Controls sliding-window
    /// masking only; kernel plumbing is identical for both kinds.
    pub layer_type: c_int,
    /// 1 when this layer reuses a source layer's KV cache (k/v_proj + k_norm
    /// skipped, cache pointers aliased). 0 for layers that own their cache.
    pub shared_kv: c_int,
    pub num_q_heads: c_int,
    pub num_kv_heads: c_int,
    /// Attention `head_dim` (256 for SWA, 512 for full on E2B).
    pub head_dim: c_int,
    /// Columns of Q/K that receive rotary. `head_dim` for sliding layers;
    /// also `head_dim` for full layers (the "nope" tail is handled by
    /// zero-filling the inv_freq tail at table-build time).
    pub rotary_dim: c_int,
    /// Sliding-window size in tokens, or 0 for full attention.
    pub sliding_window: c_int,
    /// MLP intermediate size (6144 for dense layers 0–14; 12288 for the
    /// double-wide layers 15–34 on E2B).
    pub intermediate_size: c_int,
    /// Allocated `T` dimension of the KV cache.
    pub kv_max_t: c_int,
    /// Per-layer output scale (`layer_scalar[N]`, applied in PLE phase 11).
    pub layer_scalar: f32,

    // --- Attention weights ---
    pub input_norm_w: *const c_void,
    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void, // null when `shared_kv`
    pub v_proj_w: *const c_void, // null when `shared_kv`
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void, // null when `shared_kv`
    pub o_proj_w: *const c_void,
    pub post_attn_norm_w: *const c_void,

    // --- MLP weights ---
    pub pre_ff_norm_w: *const c_void,
    pub gate_proj_w: *const c_void,
    pub up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,
    pub post_ff_norm_w: *const c_void,

    // --- PLE weights ---
    pub per_layer_input_gate_w: *const c_void,
    pub per_layer_projection_w: *const c_void,
    pub post_per_layer_input_norm_w: *const c_void,

    // --- RoPE tables for this layer's kind (sliding vs. proportional) ---
    pub cos_table: *const c_void,
    pub sin_table: *const c_void,

    // --- KV cache (shared-KV layers alias source layer's pointers) ---
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
}

unsafe impl Send for Gemma4DecodeLayerDesc {}
unsafe impl Sync for Gemma4DecodeLayerDesc {}

impl Default for Gemma4DecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// INT4 scale/zero tensors for one Gemma 4 decoder layer.
///
/// Parallel-struct to [`Gemma4DecodeLayerDesc`] — the main desc's projection
/// weight slots (`q/k/v/o_proj_w`, `gate/up/down_proj_w`,
/// `per_layer_input_gate_w`, `per_layer_projection_w`) hold packed-u8 INT4
/// weights (reinterpreted at the kernel site) and this struct carries the
/// matching BF16 scale/zero tables. Mirrors Qwen's
/// [`INT4ScaleDesc`](crate::layer_desc::INT4ScaleDesc) pattern.
///
/// Step 29 (attention-block INT4) consumes the Q/K/V/O fields; Step 30
/// (MLP+PLE INT4) adds the gate/up/down + per_layer_input_gate +
/// per_layer_projection fields. Scaffolding for Step 31 INT4 persistent
/// megakernel which will pass this struct as a parallel array alongside
/// `Gemma4DecodeLayerDesc`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4Int4ScaleDesc {
    // --- Attention projections (Step 29) ---
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
    // --- MLP projections (Step 30) ---
    pub gate_proj_scale: *const c_void,
    pub gate_proj_zero: *const c_void,
    pub up_proj_scale: *const c_void,
    pub up_proj_zero: *const c_void,
    pub down_proj_scale: *const c_void,
    pub down_proj_zero: *const c_void,
    // --- PLE projections (Step 30) ---
    pub per_layer_input_gate_scale: *const c_void,
    pub per_layer_input_gate_zero: *const c_void,
    pub per_layer_projection_scale: *const c_void,
    pub per_layer_projection_zero: *const c_void,
    /// Quantization group size (bake format fixes this at 128).
    pub group_size: c_int,
}

unsafe impl Send for Gemma4Int4ScaleDesc {}
unsafe impl Sync for Gemma4Int4ScaleDesc {}

impl Default for Gemma4Int4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-layer FP8 KV-cache scale-buffer pointers for Gemma 4. Parallel-struct
/// to [`Gemma4DecodeLayerDesc`] — when `--kv-fp8` is active the main desc's
/// `kv_cache_k`/`kv_cache_v` slots hold u8-packed FP8-E4M3 bytes, and this
/// struct carries the matching per-(head, position) F32 absmax scales.
///
/// Mirrors the Phi-4 pattern (`Phi4KVCacheFp8Desc`). Shared-KV layers (Gemma
/// 4 aliases earlier layers' caches via pointer equality) must alias scale
/// buffers too — both fields point at the source layer's scale tensors so
/// dequant reads stay self-consistent.
///
/// Both fields are null for layers that didn't allocate KV-FP8 (i.e. the
/// kernel runs in BF16 KV mode); the kernel uses non-null as the dispatch
/// signal.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4KVCacheFp8Desc {
    /// `[num_kv_heads, max_T]` F32 absmax scales for K cache. Null = BF16 K.
    pub kv_scale_k: *mut c_void,
    /// `[num_kv_heads, max_T]` F32 absmax scales for V cache. Null = BF16 V.
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for Gemma4KVCacheFp8Desc {}
unsafe impl Sync for Gemma4KVCacheFp8Desc {}

impl Default for Gemma4KVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-layer FP8-E4M3-FN weight scale-inv tensors for Gemma 4. Parallel-struct
/// to [`Gemma4DecodeLayerDesc`] — when `--fp8-runtime` is active the main
/// desc's projection slots (`q/k/v/o_proj_w`, `gate/up/down_proj_w`,
/// `per_layer_input_gate_w`, `per_layer_projection_w`) hold u8-packed FP8
/// bytes (reinterpreted at the kernel site) and this struct carries the
/// matching BF16 per-block (typically 128×128) scale_inv tables.
///
/// Mirrors the Phi-4 layout (`Phi4FP8ScaleDesc`). Shared-KV layers do not
/// need k/v scale entries (the `*_proj_w` slots themselves are null on those
/// layers), but the kernel reads from this struct unconditionally — entries
/// for skipped projections may be null.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4FP8ScaleDesc {
    // --- Attention projections ---
    pub q_proj_scale: *const c_void,
    pub k_proj_scale: *const c_void, // null when shared_kv
    pub v_proj_scale: *const c_void, // null when shared_kv
    pub o_proj_scale: *const c_void,
    // --- MLP projections ---
    pub gate_proj_scale: *const c_void,
    pub up_proj_scale: *const c_void,
    pub down_proj_scale: *const c_void,
    // --- PLE projections ---
    pub per_layer_input_gate_scale: *const c_void,
    pub per_layer_projection_scale: *const c_void,
    /// Per-block scale tile dimension (typically 128). Same value across
    /// all projections in a bake.
    pub block_size: c_int,
}

unsafe impl Send for Gemma4FP8ScaleDesc {}
unsafe impl Sync for Gemma4FP8ScaleDesc {}

impl Default for Gemma4FP8ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-sequence state pointers for batched Gemma 4 decode.
///
/// One `Gemma4BatchSeqDesc` per layer (parallel array to
/// [`Gemma4DecodeLayerDesc`]), holding per-sequence mutable state for up to
/// [`crate::layer_desc::MAX_BATCH_SIZE`] sequences. Mirrors Qwen's
/// [`crate::layer_desc::BatchSeqDesc`] but trimmed to Gemma 4's needs:
/// no linear-attention (`conv_state` / `recurrent_state`), no FP8 KV
/// (`kv_scale_*`), no BF16 shadow caches.
///
/// When `batch_size == 1` the kernel reads per-sequence state from the layer
/// descriptor's own `kv_cache_k` / `kv_cache_v` and the scalar `position`
/// argument — `batch_descs` is `nullptr`. When `batch_size > 1`, the kernel
/// reads per-sequence pointers and offsets from this struct instead.
///
/// Shared-KV layers must alias the source layer's per-sequence cache pointers
/// (i.e. `batch_descs[shared_layer].kv_cache_k[b] == batch_descs[source_layer].kv_cache_k[b]`)
/// — replication across sequences is the engine's responsibility, not the
/// kernel's.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Gemma4BatchSeqDesc {
    /// Per-sequence position in the sequence (RoPE table lookup + KV write slot).
    pub seqlen_offset: [c_int; crate::layer_desc::MAX_BATCH_SIZE],
    /// Per-sequence K cache pointer (`[num_kv_heads, kv_max_t, head_dim]` BF16).
    pub kv_cache_k: [*mut c_void; crate::layer_desc::MAX_BATCH_SIZE],
    /// Per-sequence V cache pointer (`[num_kv_heads, kv_max_t, head_dim]` BF16).
    pub kv_cache_v: [*mut c_void; crate::layer_desc::MAX_BATCH_SIZE],
    /// Per-sequence allocated `T` dimension of the KV cache.
    pub kv_max_t: [c_int; crate::layer_desc::MAX_BATCH_SIZE],
}

unsafe impl Send for Gemma4BatchSeqDesc {}
unsafe impl Sync for Gemma4BatchSeqDesc {}

impl Default for Gemma4BatchSeqDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Required F32 workspace (elements) for the persistent decode megakernel.
/// Sized to the max of (fused_attn_block, fused_mlp_ple) needs across all
/// layers — phase A and phase B run sequentially within each layer and share
/// the buffer from offset 0.
pub fn persistent_decode_workspace_elems(
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim_max: usize,
    max_t: usize,
    intermediate_size_max: usize,
    ple_hidden: usize,
) -> usize {
    let attn = fused_attn_block_workspace_elems(
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim_max,
        max_t,
    );
    let mlp = fused_mlp_ple_workspace_elems(hidden_size, intermediate_size_max, ple_hidden);
    attn.max(mlp)
}

/// Run a full Gemma 4 forward pass for one decode token in a single kernel
/// launch. The `layers` buffer holds a contiguous `[num_layers]` array of
/// `Gemma4DecodeLayerDesc` (uploaded by the caller). `hidden_io` is the
/// token's BF16 hidden state on entry and exit (`[hidden_size]`).
///
/// Shared-KV layers must have their `kv_cache_k` / `kv_cache_v` pointers
/// aliased to the source layer's cache buffers — no replication is performed
/// inside the kernel.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    kv_fp8_descs: Option<&GpuBuffer>,
    fp8_scales: Option<&GpuBuffer>,
    hidden_io: &mut GpuBuffer,
    per_layer_inputs: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    position: usize,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    let kv_fp8_ptr = kv_fp8_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let fp8_scales_ptr = fp8_scales.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe {
        supersonic_gemma4_hip_persistent_decode(
            dtype.kernel_dtype_code(),
            ordinal,
            num_layers,
            hidden_size,
            ple_hidden,
            position,
            eps,
            scale,
            layers.as_ptr(),
            kv_fp8_ptr,
            fp8_scales_ptr,
            hidden_io.as_mut_ptr(),
            per_layer_inputs.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 persistent_decode failed with status {status}"),
        ));
    }
    Ok(())
}

/// CUDA-only fused input-staging + persistent decode path for Gemma 4 BF16
/// single-token decode. The caller keeps the staging buffers alive so the same
/// memory can be reused by the fallback multi-launch path.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_fused_input(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    kv_fp8_descs: Option<&GpuBuffer>,
    fp8_scales: Option<&GpuBuffer>,
    embed_tokens: &GpuBuffer,
    embed_tokens_per_layer: &GpuBuffer,
    per_layer_model_projection_w: &GpuBuffer,
    per_layer_projection_norm_w: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    pli_proj: &mut GpuBuffer,
    pli_normed: &mut GpuBuffer,
    ple_raw: &mut GpuBuffer,
    per_layer_inputs: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    vocab_size: usize,
    token_id: u32,
    position: usize,
    eps: f32,
    scale: f32,
    embed_scale: f32,
    proj_scale: f32,
    ple_scale: f32,
    combine_scale: f32,
) -> Result<(), GpuError> {
    if hidden_io.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "gemma4 persistent_decode_fused_input is CUDA-only".into(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        let kv_fp8_ptr = kv_fp8_descs.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        let fp8_scales_ptr = fp8_scales.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        let status = unsafe {
            supersonic_gemma4_cuda_persistent_decode_fused_input(
                dtype.kernel_dtype_code(),
                ordinal,
                num_layers,
                hidden_size,
                ple_hidden,
                vocab_size,
                token_id as c_uint,
                position,
                eps,
                scale,
                embed_scale,
                proj_scale,
                ple_scale,
                combine_scale,
                layers.as_ptr(),
                kv_fp8_ptr,
                fp8_scales_ptr,
                embed_tokens.as_ptr(),
                embed_tokens_per_layer.as_ptr(),
                per_layer_model_projection_w.as_ptr(),
                per_layer_projection_norm_w.as_ptr(),
                hidden_io.as_mut_ptr(),
                pli_proj.as_mut_ptr(),
                pli_normed.as_mut_ptr(),
                ple_raw.as_mut_ptr(),
                per_layer_inputs.as_mut_ptr(),
                workspace.as_mut_ptr(),
                matvec_counter.as_mut_ptr() as *mut c_uint,
                barrier_counter.as_mut_ptr() as *mut c_uint,
                barrier_flag.as_mut_ptr() as *mut c_uint,
            )
        };
        if status != 0 {
            return Err(GpuError::backend(
                Backend::Cuda,
                format!("gemma4 persistent_decode_fused_input failed with status {status}"),
            ));
        }
        Ok(())
    }

    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            dtype,
            layers,
            kv_fp8_descs,
            fp8_scales,
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection_w,
            per_layer_projection_norm_w,
            hidden_io,
            pli_proj,
            pli_normed,
            ple_raw,
            per_layer_inputs,
            workspace,
            matvec_counter,
            barrier_counter,
            barrier_flag,
            num_layers,
            hidden_size,
            ple_hidden,
            vocab_size,
            token_id,
            position,
            eps,
            scale,
            embed_scale,
            proj_scale,
            ple_scale,
            combine_scale,
        );
        Err(GpuError::InvalidArg(
            "gemma4 persistent_decode_fused_input requires a CUDA build".into(),
        ))
    }
}

/// CUDA-only one-launch greedy path for Gemma 4 BF16 batch-1 decode:
/// input staging + persistent decode + final RMSNorm + LM-head argmax.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_fused_input_argmax(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    embed_tokens: &GpuBuffer,
    embed_tokens_per_layer: &GpuBuffer,
    per_layer_model_projection_w: &GpuBuffer,
    per_layer_projection_norm_w: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    pli_proj: &mut GpuBuffer,
    pli_normed: &mut GpuBuffer,
    ple_raw: &mut GpuBuffer,
    per_layer_inputs: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    out_token: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    vocab_size: usize,
    token_id: u32,
    position: usize,
    eps: f32,
    scale: f32,
    embed_scale: f32,
    proj_scale: f32,
    ple_scale: f32,
    combine_scale: f32,
) -> Result<(), GpuError> {
    if hidden_io.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "gemma4 persistent_decode_fused_input_argmax is CUDA-only".into(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        let status = unsafe {
            supersonic_gemma4_cuda_persistent_decode_fused_input_argmax(
                dtype.kernel_dtype_code(),
                ordinal,
                num_layers,
                hidden_size,
                ple_hidden,
                vocab_size,
                token_id as c_uint,
                position,
                eps,
                scale,
                embed_scale,
                proj_scale,
                ple_scale,
                combine_scale,
                layers.as_ptr(),
                embed_tokens.as_ptr(),
                embed_tokens_per_layer.as_ptr(),
                per_layer_model_projection_w.as_ptr(),
                per_layer_projection_norm_w.as_ptr(),
                final_norm_w.as_ptr(),
                lm_head_w.as_ptr(),
                hidden_io.as_mut_ptr(),
                pli_proj.as_mut_ptr(),
                pli_normed.as_mut_ptr(),
                ple_raw.as_mut_ptr(),
                per_layer_inputs.as_mut_ptr(),
                workspace.as_mut_ptr(),
                out_token.as_mut_ptr() as *mut c_uint,
                matvec_counter.as_mut_ptr() as *mut c_uint,
                barrier_counter.as_mut_ptr() as *mut c_uint,
                barrier_flag.as_mut_ptr() as *mut c_uint,
            )
        };
        if status != 0 {
            return Err(GpuError::backend(
                Backend::Cuda,
                format!("gemma4 persistent_decode_fused_input_argmax failed with status {status}"),
            ));
        }
        Ok(())
    }

    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            dtype,
            layers,
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection_w,
            per_layer_projection_norm_w,
            final_norm_w,
            lm_head_w,
            hidden_io,
            pli_proj,
            pli_normed,
            ple_raw,
            per_layer_inputs,
            workspace,
            out_token,
            matvec_counter,
            barrier_counter,
            barrier_flag,
            num_layers,
            hidden_size,
            ple_hidden,
            vocab_size,
            token_id,
            position,
            eps,
            scale,
            embed_scale,
            proj_scale,
            ple_scale,
            combine_scale,
        );
        Err(GpuError::InvalidArg(
            "gemma4 persistent_decode_fused_input_argmax requires a CUDA build".into(),
        ))
    }
}

/// CUDA-only final RMSNorm + LM-head greedy argmax. This deliberately returns
/// only the selected token, so validation paths that need full logits should
/// keep using `rms_norm` + `matvec`.
#[allow(clippy::too_many_arguments)]
pub fn final_norm_lm_head_argmax(
    ordinal: usize,
    dtype: ScalarType,
    hidden_io: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    workspace: &mut GpuBuffer,
    out_token: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    hidden_size: usize,
    vocab_size: usize,
    eps: f32,
) -> Result<(), GpuError> {
    if hidden_io.backend() != Backend::Cuda {
        return Err(GpuError::InvalidArg(
            "gemma4 final_norm_lm_head_argmax is CUDA-only".into(),
        ));
    }

    #[cfg(supersonic_backend_cuda)]
    {
        let status = unsafe {
            supersonic_gemma4_cuda_final_norm_lm_head_argmax(
                dtype.kernel_dtype_code(),
                ordinal,
                hidden_size,
                vocab_size,
                eps,
                hidden_io.as_ptr(),
                final_norm_w.as_ptr(),
                lm_head_w.as_ptr(),
                workspace.as_mut_ptr(),
                out_token.as_mut_ptr() as *mut c_uint,
                barrier_counter.as_mut_ptr() as *mut c_uint,
                barrier_flag.as_mut_ptr() as *mut c_uint,
            )
        };
        if status != 0 {
            return Err(GpuError::backend(
                Backend::Cuda,
                format!("gemma4 final_norm_lm_head_argmax failed with status {status}"),
            ));
        }
        Ok(())
    }

    #[cfg(not(supersonic_backend_cuda))]
    {
        let _ = (
            ordinal,
            dtype,
            hidden_io,
            final_norm_w,
            lm_head_w,
            workspace,
            out_token,
            barrier_counter,
            barrier_flag,
            hidden_size,
            vocab_size,
            eps,
        );
        Err(GpuError::InvalidArg(
            "gemma4 final_norm_lm_head_argmax requires a CUDA build".into(),
        ))
    }
}

/// Batched variant of [`persistent_decode`] — runs `batch_size` parallel
/// decode tokens through all layers in a single kernel launch. Weight reads
/// in the six matmul phases (Q/K/V, o_proj, gate+up, down, per_layer_input_gate,
/// per_layer_projection) are amortized across sequences.
///
/// Buffer shapes:
/// - `layers`: `[num_layers]` of [`Gemma4DecodeLayerDesc`] (same as single-seq).
/// - `batch_descs`: `[num_layers]` of [`Gemma4BatchSeqDesc`], one per layer.
///   `batch_descs[l].kv_cache_k[b]`, `.kv_cache_v[b]`, `.seqlen_offset[b]`,
///   `.kv_max_t[b]` encode sequence `b`'s state for layer `l`. Shared-KV layers
///   must alias the source layer's per-sequence KV pointers in the same `b`.
/// - `hidden_io`: `[batch_size, hidden_size]` BF16 contiguous (in/out per seq).
/// - `per_layer_inputs`: `[batch_size, num_layers, ple_hidden]` BF16 contiguous.
/// - `workspace`: `batch_size * persistent_decode_workspace_elems(...)` F32.
///
/// `ws_stride` must equal the value returned by
/// [`persistent_decode_workspace_elems`] for this engine's configuration —
/// it's the per-sequence slice size into `workspace`.
///
/// All sequences in the batch must share the same allocated `max_t`
/// (`batch_descs[l].kv_max_t[b] == layers[l].kv_max_t` for every `b`); the
/// kernel uses `layers[l].kv_max_t` as the uniform scores-stride across seqs.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_batch(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    batch_descs: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    per_layer_inputs: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    batch_size: usize,
    ws_stride: usize,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    if batch_size == 0 || batch_size > crate::layer_desc::MAX_BATCH_SIZE {
        return Err(GpuError::backend(
            Backend::Hip,
            format!(
                "gemma4 persistent_decode_batch: batch_size {batch_size} out of range [1, {}]",
                crate::layer_desc::MAX_BATCH_SIZE
            ),
        ));
    }
    let status = unsafe {
        supersonic_gemma4_hip_persistent_decode_batch(
            dtype.kernel_dtype_code(),
            ordinal,
            num_layers,
            hidden_size,
            ple_hidden,
            eps,
            scale,
            batch_size,
            ws_stride,
            layers.as_ptr(),
            batch_descs.as_ptr(),
            hidden_io.as_mut_ptr(),
            per_layer_inputs.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 persistent_decode_batch failed with status {status}"),
        ));
    }
    Ok(())
}

/// INT4 version of [`persistent_decode`]. Runs a full Gemma 4 forward pass
/// for one decode token in a single kernel launch with all Q/K/V/O/gate/up/
/// down/per_layer_input_gate/per_layer_projection matmuls INT4-dequantized
/// inline via the same `(packed u8, BF16 scale, BF16 zero)` format the
/// Step-29/30 fused kernels use. Same workspace sizing as the BF16 variant
/// ([`persistent_decode_workspace_elems`]).
///
/// The `layers` buffer holds a contiguous `[num_layers]` array of
/// [`Gemma4DecodeLayerDesc`] with its projection weight slots pointing at
/// packed-u8 INT4 tensors (reinterpreted at the kernel site). The
/// `int4_scales` buffer holds the matching `[num_layers]` array of
/// [`Gemma4Int4ScaleDesc`] entries. Shared-KV layers must have their
/// `kv_cache_k` / `kv_cache_v` pointers aliased to the source layer's cache
/// buffers — no replication is performed inside the kernel.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_int4(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    int4_scales: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    per_layer_inputs: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    position: usize,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    let status = unsafe {
        supersonic_gemma4_hip_persistent_decode_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            num_layers,
            hidden_size,
            ple_hidden,
            position,
            eps,
            scale,
            layers.as_ptr(),
            int4_scales.as_ptr(),
            hidden_io.as_mut_ptr(),
            per_layer_inputs.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 persistent_decode_int4 failed with status {status}"),
        ));
    }
    Ok(())
}

/// Batched INT4 variant of [`persistent_decode_int4`] — runs `batch_size`
/// parallel decode tokens through all layers in a single kernel launch with
/// every matmul INT4-dequantized inline. Same buffer conventions as
/// [`persistent_decode_batch`] plus the `int4_scales` parallel array.
///
/// `ws_stride` must equal [`persistent_decode_workspace_elems`] for this
/// engine's configuration. `workspace` must be sized `batch_size * ws_stride`
/// F32 elements.
#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_batch_int4(
    ordinal: usize,
    dtype: ScalarType,
    layers: &GpuBuffer,
    int4_scales: &GpuBuffer,
    batch_descs: &GpuBuffer,
    hidden_io: &mut GpuBuffer,
    per_layer_inputs: &GpuBuffer,
    workspace: &mut GpuBuffer,
    matvec_counter: &mut GpuBuffer,
    barrier_counter: &mut GpuBuffer,
    barrier_flag: &mut GpuBuffer,
    num_layers: usize,
    hidden_size: usize,
    ple_hidden: usize,
    batch_size: usize,
    ws_stride: usize,
    eps: f32,
    scale: f32,
) -> Result<(), GpuError> {
    if batch_size == 0 || batch_size > crate::layer_desc::MAX_BATCH_SIZE {
        return Err(GpuError::backend(
            Backend::Hip,
            format!(
                "gemma4 persistent_decode_batch_int4: batch_size {batch_size} out of range [1, {}]",
                crate::layer_desc::MAX_BATCH_SIZE
            ),
        ));
    }
    let status = unsafe {
        supersonic_gemma4_hip_persistent_decode_batch_int4(
            dtype.kernel_dtype_code(),
            ordinal,
            num_layers,
            hidden_size,
            ple_hidden,
            eps,
            scale,
            batch_size,
            ws_stride,
            layers.as_ptr(),
            int4_scales.as_ptr(),
            batch_descs.as_ptr(),
            hidden_io.as_mut_ptr(),
            per_layer_inputs.as_ptr(),
            workspace.as_mut_ptr(),
            matvec_counter.as_mut_ptr() as *mut c_uint,
            barrier_counter.as_mut_ptr() as *mut c_uint,
            barrier_flag.as_mut_ptr() as *mut c_uint,
        )
    };
    if status != 0 {
        return Err(GpuError::backend(
            Backend::Hip,
            format!("gemma4 persistent_decode_batch_int4 failed with status {status}"),
        ));
    }
    Ok(())
}
