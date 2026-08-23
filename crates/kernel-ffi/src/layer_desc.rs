use std::ffi::{c_int, c_void};

/// Maximum batch size supported by the batched decode kernel.
/// Struct arrays in BatchSeqDesc are fixed to this size.
pub const MAX_BATCH_SIZE: usize = 8;

/// Rust mirror of the historical `Qwen35DecodeLayerDesc` bridge layout.
/// The C++ field spelling is an external ABI detail; the product is Qwen3.8.
/// Describes one decoder layer for the persistent decode megakernel.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DecodeLayerDesc {
    // --- Common fields ---
    pub layer_type: c_int,        // 0 = linear_attention, 1 = full_attention
    pub intermediate_size: c_int, // MLP intermediate dim
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,
    pub rms_norm_add_unit_offset: c_int,
    pub gate_proj_w: *const c_void,
    pub up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,

    // --- Linear attention (layer_type == 0) ---
    pub qkv_proj_w: *const c_void,
    pub qkv_out_dim: c_int,
    pub z_proj_w: *const c_void,
    pub z_out_dim: c_int,
    pub b_proj_w: *const c_void,
    pub a_proj_w: *const c_void,
    pub conv1d_w: *const c_void,
    pub conv_kernel_size: c_int,
    pub linear_out_proj_w: *const c_void,
    pub linear_value_dim: c_int,
    pub linear_num_v_heads: c_int,
    pub linear_head_k_dim: c_int,
    pub linear_head_v_dim: c_int,
    pub dt_bias_w: *const c_void,
    pub a_log_exp_w: *const c_void,
    pub linear_norm_w: *const c_void,
    pub linear_norm_eps: f32,
    pub conv_state: *mut c_void,
    pub recurrent_state: *mut c_void,

    // --- Full attention (layer_type == 1) ---
    pub q_proj_w: *const c_void,
    pub q_out_dim: c_int,
    pub k_proj_w: *const c_void,
    pub k_out_dim: c_int,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub attn_head_dim: c_int,
    pub attn_num_heads: c_int,
    pub attn_num_kv_heads: c_int,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub q_norm_eps: f32,
    pub k_norm_eps: f32,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,
    // Optional BF16 sidecar for KV-FP8 bring-up / parity-sensitive reads.
    pub kv_shadow_k: *mut c_void,
    pub kv_shadow_v: *mut c_void,
    pub kv_shadow_start: c_int,
    // Optional linear Step-B debug export. When non-null on a linear layer,
    // the kernel writes one selected channel's post-update conv-state taps,
    // followed by qkv and conv_out scalars, into this F32 buffer.
    pub debug_linear_trace_out: *mut c_void,
    pub debug_linear_trace_channel: c_int,
}

unsafe impl Send for DecodeLayerDesc {}
unsafe impl Sync for DecodeLayerDesc {}

impl Default for DecodeLayerDesc {
    fn default() -> Self {
        // Safety: all-zeros is valid for this #[repr(C)] struct (null pointers, zero ints/floats).
        unsafe { std::mem::zeroed() }
    }
}

/// FP8 scale_inv pointers for runtime dequantization.
/// Parallel struct to DecodeLayerDesc — one per layer.
/// Passed as a separate kernel argument to avoid modifying DecodeLayerDesc layout
/// (which triggers hipcc codegen bugs on gfx1150).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FP8ScaleDesc {
    // Common (all layer types) — MLP projections
    pub gate_proj_scale: *const c_void,
    pub up_proj_scale: *const c_void,
    pub down_proj_scale: *const c_void,
    // Linear attention projections
    pub qkv_proj_scale: *const c_void,
    pub z_proj_scale: *const c_void,
    pub b_proj_scale: *const c_void,
    pub a_proj_scale: *const c_void,
    pub linear_out_proj_scale: *const c_void,
    // Full attention projections
    pub q_proj_scale: *const c_void,
    pub k_proj_scale: *const c_void,
    pub v_proj_scale: *const c_void,
    pub o_proj_scale: *const c_void,
    // Block size for scale_inv indexing (typically 128)
    pub block_size: c_int,
}

unsafe impl Send for FP8ScaleDesc {}
unsafe impl Sync for FP8ScaleDesc {}

impl Default for FP8ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-layer KV cache FP8 scale pointers for dynamic quantization.
/// Parallel struct to DecodeLayerDesc — one per layer, passed as a separate
/// kernel argument (same pattern as FP8ScaleDesc).
/// Only meaningful for full-attention layers; linear-attention layers use null pointers.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KVCacheFp8Desc {
    /// Per-head-per-position absmax scale for K cache: [num_kv_heads, max_T] F32
    pub kv_scale_k: *mut c_void,
    /// Per-head-per-position absmax scale for V cache: [num_kv_heads, max_T] F32
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for KVCacheFp8Desc {}
unsafe impl Sync for KVCacheFp8Desc {}

impl Default for KVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Low-bit weight descriptors for runtime dequantization.
/// Parallel struct to DecodeLayerDesc — one per layer.
/// Passed as a separate kernel argument (same pattern as FP8ScaleDesc).
/// Type code 4 means native packed INT4 with scale/zero sidecars. Type codes
/// 8/12/13/14 mean verbatim GGML Q8_0/Q4_K/Q5_K/Q6_K blocks. Type codes
/// 108/109/110/111 are GQH3/GQH2_H/GQH2_C/GQH4; `*_proj_scale` then points at an
/// 8-byte `{f32 tensor_scale, i32 grid_code}` sidecar. Type codes 20/21/22
/// are reserved for HIGGS4, QuIP# E8, and QTIP trellis runtime profiles.
///
/// The historical `INT4ScaleDesc` name is retained for the external kernel
/// ABI. For GQH rows the `*_proj_scale` pointers carry GQH tensor-header
/// sidecars, not GPTQ metadata; the C++ `Qwen35INT4ScaleDesc` mirror and
/// `qwen38::desc_builder::build_int4_scale_descs` depend on this exact field
/// order and tail layout. The C++ type spelling is retained for ABI only.
/// Do not reorder, remove, or repurpose these fields
/// without changing both FFI mirrors together and updating the layout test.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct INT4ScaleDesc {
    // Common MLP weights
    pub gate_proj_scale: *const c_void,
    pub gate_proj_zero: *const c_void,
    pub gate_proj_awq_inv_scale: *const c_void,
    pub up_proj_scale: *const c_void,
    pub up_proj_zero: *const c_void,
    pub up_proj_awq_inv_scale: *const c_void,
    pub down_proj_scale: *const c_void,
    pub down_proj_zero: *const c_void,
    pub down_proj_awq_inv_scale: *const c_void,
    // Linear attention weights
    pub qkv_proj_scale: *const c_void,
    pub qkv_proj_zero: *const c_void,
    pub qkv_proj_awq_inv_scale: *const c_void,
    pub z_proj_scale: *const c_void,
    pub z_proj_zero: *const c_void,
    pub z_proj_awq_inv_scale: *const c_void,
    pub linear_out_proj_scale: *const c_void,
    pub linear_out_proj_zero: *const c_void,
    pub linear_out_proj_awq_inv_scale: *const c_void,
    // Full attention weights
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub q_proj_awq_inv_scale: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub k_proj_awq_inv_scale: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub v_proj_awq_inv_scale: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
    pub o_proj_awq_inv_scale: *const c_void,
    // Group size for INT4 quantization (typically 128)
    pub group_size: c_int,
    pub gate_proj_type: c_int,
    pub up_proj_type: c_int,
    pub down_proj_type: c_int,
    pub qkv_proj_type: c_int,
    pub z_proj_type: c_int,
    pub linear_out_proj_type: c_int,
    pub q_proj_type: c_int,
    pub k_proj_type: c_int,
    pub v_proj_type: c_int,
    pub o_proj_type: c_int,
    pub a_proj_type: c_int,
    pub b_proj_type: c_int,
}

unsafe impl Send for INT4ScaleDesc {}
unsafe impl Sync for INT4ScaleDesc {}

impl Default for INT4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-sequence state pointers for batched decode.
/// One BatchSeqDesc per layer (parallel to DecodeLayerDesc), containing
/// per-sequence mutable state for up to MAX_BATCH_SIZE sequences.
/// When batch_size == 1, this struct is not used (nullptr passed to kernel).
/// When batch_size > 1, the kernel reads per-sequence state from here
/// instead of from DecodeLayerDesc.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BatchSeqDesc {
    /// Per-sequence position in the sequence (for RoPE table lookup).
    pub seqlen_offset: [c_int; MAX_BATCH_SIZE],
    // --- Full attention per-sequence state ---
    pub kv_cache_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_cache_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_len: [c_int; MAX_BATCH_SIZE],
    pub kv_max_t: [c_int; MAX_BATCH_SIZE],
    // Optional BF16 sidecar for KV-FP8 bring-up / parity-sensitive reads.
    pub kv_shadow_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_shadow_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_shadow_start: [c_int; MAX_BATCH_SIZE],
    // --- Linear attention per-sequence state ---
    pub conv_state: [*mut c_void; MAX_BATCH_SIZE],
    pub recurrent_state: [*mut c_void; MAX_BATCH_SIZE],
    // --- FP8 KV cache per-sequence scale buffers ---
    pub kv_scale_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_scale_v: [*mut c_void; MAX_BATCH_SIZE],
}

unsafe impl Send for BatchSeqDesc {}
unsafe impl Sync for BatchSeqDesc {}

impl Default for BatchSeqDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::INT4ScaleDesc;
    use std::ffi::{c_int, c_void};
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn gqh_sidecar_descriptor_matches_cpp_wire_layout() {
        // Thirty pointer slots (three per projection) are followed by
        // group_size and twelve low-bit type codes in the HIP mirror.
        let pointer_bytes = 30 * size_of::<*const c_void>();
        let scalar_bytes = 13 * size_of::<c_int>();
        let alignment = align_of::<*const c_void>();
        let raw_size = pointer_bytes + scalar_bytes;
        let expected_size = (raw_size + alignment - 1) & !(alignment - 1);

        assert_eq!(offset_of!(INT4ScaleDesc, group_size), pointer_bytes);
        assert_eq!(
            offset_of!(INT4ScaleDesc, b_proj_type),
            pointer_bytes + 12 * size_of::<c_int>()
        );
        assert_eq!(size_of::<INT4ScaleDesc>(), expected_size);
        assert_eq!(align_of::<INT4ScaleDesc>(), alignment);
    }
}
