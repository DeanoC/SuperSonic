/// Per-layer descriptor for the Qwen3.6-MoE megakernel. Field order and
/// natural x86_64 alignment must match the C++ struct in
/// `kernels/qwen36_moe.hip` exactly. The repr-C layout is fixed at PR 4
/// time and grows by appending new fields, never reordering existing
/// ones — see the matching `static_assert(sizeof(...))` on the C++ side.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeDecodeLayerDesc {
    /// Layer index in `[0, num_hidden_layers)`. Used by the kernel to pick
    /// the cos/sin RoPE entry and to sanity-check the descriptor pointer.
    pub layer_idx: c_int,
    /// 0 = linear-attention layer, 1 = full-attention layer.
    pub is_full_attention: c_int,

    // --- RMS norms --------------------------------------------------------
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,

    // --- Full-attention slots (read iff is_full_attention == 1) -----------
    /// q_proj output dim. With `attn_output_gate=true` (Qwen3-Next) this is
    /// `2 * num_heads * head_dim`; the kernel splits the upper half off as
    /// the sigmoid output gate. With `attn_output_gate=false` it's just
    /// `num_heads * head_dim`. The sign is captured by `attn_output_gate`.
    pub q_proj_w: *const c_void,
    pub q_proj_out_dim: c_int,
    /// 0 = no output gate (q_proj_out_dim == num_heads*head_dim),
    /// 1 = attn_output_gate fused (q_proj_out_dim == 2*num_heads*head_dim).
    pub attn_output_gate: c_int,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub attn_head_dim: c_int,
    pub attn_num_heads: c_int,
    pub attn_num_kv_heads: c_int,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,

    // --- Linear-attention slots (read iff is_full_attention == 0) ---------
    pub linear_in_proj_qkv_w: *const c_void,
    pub linear_in_proj_z_w: *const c_void,
    pub linear_in_proj_b_w: *const c_void,
    pub linear_in_proj_a_w: *const c_void,
    pub linear_out_proj_w: *const c_void,
    pub linear_conv1d_w: *const c_void,
    pub linear_dt_bias: *const c_void,
    pub linear_a_log_exp: *const c_void,
    pub linear_norm_w: *const c_void,
    pub linear_qkv_dim: c_int,
    pub linear_v_dim: c_int,
    pub linear_v_heads: c_int,
    pub linear_conv_kernel_dim: c_int,
    /// Linear-attention conv state pointer, shape `[batch, qkv_dim,
    /// kernel-1]`. NULL on first decode step (kernel will zero on read).
    pub linear_conv_state: *mut c_void,
    /// Linear-attention recurrent state, shape `[batch, V_heads, V_dim,
    /// K_dim]`. NULL on first decode step.
    pub linear_recurrent_state: *mut c_void,

    // --- MoE block (always read, regardless of attention type) ------------
    /// Router weight `[num_experts, hidden]`, BF16. Always BF16 (excluded
    /// from INT4 quant by `is_int4_target`).
    pub router_w: *const c_void,
    /// Fused expert gate+up `[num_experts, 2*moe_intermediate_size, hidden]`.
    /// At INT4 launch the pointer reinterprets as packed `u8` (2 nibbles
    /// per byte), with sidecar scale/zero in `Qwen36MoeInt4ScaleDesc`.
    pub experts_gate_up_w: *const c_void,
    /// Fused expert down `[num_experts, hidden, moe_intermediate_size]`.
    pub experts_down_w: *const c_void,
    /// Shared expert (always-on). `gate_proj` and `up_proj` are
    /// `[shared_int, hidden]`; `down_proj` is `[hidden, shared_int]`.
    pub shared_expert_gate_proj_w: *const c_void,
    pub shared_expert_up_proj_w: *const c_void,
    pub shared_expert_down_proj_w: *const c_void,
    /// Scalar shared-expert gate `[1, hidden]`, BF16. Applied as
    /// `sigmoid(gate · x) * shared_expert(x)`.
    pub shared_expert_gate_w: *const c_void,
    /// Number of routed experts present in this layer. Must match
    /// `desc.num_experts` across layers (sanity-checked by the host).
    pub num_experts: c_int,
    /// Top-k for routing.
    pub top_k: c_int,
    pub moe_intermediate_size: c_int,
    pub shared_expert_intermediate_size: c_int,
    /// 1 if router applies `softmax(top_k_logits)` renormalization
    /// (`norm_topk_prob=true` in config). 0 otherwise.
    pub norm_topk_prob: c_int,

    // --- KV-FP8 sidecar (read iff is_full_attention == 1 AND
    // matching kv_fp8_descs[layer].kv_scale_k != null) ---------------
    /// BF16 sidecar buffer `[num_kv_heads, kv_shadow_window, head_dim]`.
    /// Null when the sidecar is disabled. The kernel reads from the sidecar
    /// (instead of dequantising FP8) for positions covered by the rolling
    /// sidecar window.
    pub kv_shadow_k: *mut c_void,
    /// BF16 sidecar buffer `[num_kv_heads, kv_shadow_window, head_dim]`.
    /// Paired with [`Self::kv_shadow_k`]; null under the same conditions.
    pub kv_shadow_v: *mut c_void,
    /// Earliest absolute KV position the sidecar may cover. `-1` when the
    /// sidecar is disabled. Runtime coverage is
    /// `max(kv_shadow_start, position + 1 - kv_shadow_window)..=position`.
    pub kv_shadow_start: c_int,
    /// Number of recent KV positions physically stored in the BF16 sidecar.
    /// Zero when the sidecar is disabled. The kernel uses modulo indexing so
    /// the descriptor can remain fixed across decode steps.
    pub kv_shadow_window: c_int,
}

unsafe impl Send for Qwen36MoeDecodeLayerDesc {}
unsafe impl Sync for Qwen36MoeDecodeLayerDesc {}

impl Default for Qwen36MoeDecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Storage geometry for one quantized projection.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeInt4WeightDesc {
    pub scale: *const c_void,
    pub zero: *const c_void,
    pub packed_row_stride_bytes: u64,
    pub packed_expert_stride_bytes: u64,
    pub scale_row_stride_elements: u64,
    pub scale_expert_stride_elements: u64,
    pub input_group_size: c_int,
    pub output_group_size: c_int,
    pub implicit_zero_code: c_int,
    pub encoding: c_int,
}

unsafe impl Send for Qwen36MoeInt4WeightDesc {}
unsafe impl Sync for Qwen36MoeInt4WeightDesc {}

impl Default for Qwen36MoeInt4WeightDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Parallel-struct to [`Qwen36MoeDecodeLayerDesc`] carrying one explicit
/// quantized-storage descriptor per projection. Projection order is ABI-fixed.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen36MoeInt4ScaleDesc {
    pub q_proj: Qwen36MoeInt4WeightDesc,
    pub k_proj: Qwen36MoeInt4WeightDesc,
    pub v_proj: Qwen36MoeInt4WeightDesc,
    pub o_proj: Qwen36MoeInt4WeightDesc,

    pub linear_in_proj_qkv: Qwen36MoeInt4WeightDesc,
    pub linear_in_proj_z: Qwen36MoeInt4WeightDesc,
    pub linear_out_proj: Qwen36MoeInt4WeightDesc,

    pub experts_gate_up: Qwen36MoeInt4WeightDesc,
    pub experts_down: Qwen36MoeInt4WeightDesc,

    pub shared_expert_gate_proj: Qwen36MoeInt4WeightDesc,
    pub shared_expert_up_proj: Qwen36MoeInt4WeightDesc,
    pub shared_expert_down_proj: Qwen36MoeInt4WeightDesc,
}

unsafe impl Send for Qwen36MoeInt4ScaleDesc {}
unsafe impl Sync for Qwen36MoeInt4ScaleDesc {}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<Qwen36MoeInt4WeightDesc>() == 64);
    assert!(align_of::<Qwen36MoeInt4WeightDesc>() == 8);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, scale) == 0);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, zero) == 8);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, packed_row_stride_bytes) == 16);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, packed_expert_stride_bytes) == 24);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, scale_row_stride_elements) == 32);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, scale_expert_stride_elements) == 40);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, input_group_size) == 48);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, output_group_size) == 52);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, implicit_zero_code) == 56);
    assert!(offset_of!(Qwen36MoeInt4WeightDesc, encoding) == 60);

    assert!(size_of::<Qwen36MoeInt4ScaleDesc>() == 768);
    assert!(align_of::<Qwen36MoeInt4ScaleDesc>() == 8);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, q_proj) == 0);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, k_proj) == 64);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, v_proj) == 128);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, o_proj) == 192);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_in_proj_qkv) == 256);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_in_proj_z) == 320);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_out_proj) == 384);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, experts_gate_up) == 448);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, experts_down) == 512);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_gate_proj) == 576);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_up_proj) == 640);
    assert!(offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_down_proj) == 704);
};

/// Per-layer KV cache FP8 scale pointers for Qwen3.6-MoE.
///
/// Parallel struct to [`Qwen36MoeDecodeLayerDesc`] — one entry per layer,
/// passed as a separate kernel argument (same pattern as
/// [`Qwen36MoeInt4ScaleDesc`]). Linear-attention layers leave both
/// pointers null. When KV-FP8 is off, the entire
/// `*const Qwen36MoeKVCacheFp8Desc` array argument is null.
///
/// Mirrors the qwen35 `KVCacheFp8Desc` shape: F32 absmax scale per
/// (kv_head, position).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeKVCacheFp8Desc {
    /// `[num_kv_heads, max_T]` F32. Null for linear-attn layers.
    pub kv_scale_k: *mut c_void,
    /// `[num_kv_heads, max_T]` F32. Null for linear-attn layers.
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for Qwen36MoeKVCacheFp8Desc {}
unsafe impl Sync for Qwen36MoeKVCacheFp8Desc {}

impl Default for Qwen36MoeKVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-sequence batched-decode state, parallel to the layer descriptor
/// array. Only the first `batch_size` slots are read.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeBatchSeqDesc {
    pub seqlen_offset: [c_int; MAX_BATCH_SIZE],
    pub kv_cache_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_cache_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_len: [c_int; MAX_BATCH_SIZE],
    pub kv_max_t: [c_int; MAX_BATCH_SIZE],
    pub linear_conv_state: [*mut c_void; MAX_BATCH_SIZE],
    pub linear_recurrent_state: [*mut c_void; MAX_BATCH_SIZE],
}

unsafe impl Send for Qwen36MoeBatchSeqDesc {}
unsafe impl Sync for Qwen36MoeBatchSeqDesc {}

impl Default for Qwen36MoeBatchSeqDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(test)]
mod qwen36_int4_weight_desc_layout_tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn qwen36_int4_weight_desc_matches_cpp_layout() {
        assert_eq!(size_of::<Qwen36MoeInt4WeightDesc>(), 64);
        assert_eq!(align_of::<Qwen36MoeInt4WeightDesc>(), 8);
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, scale), 0);
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, zero), 8);
        assert_eq!(
            offset_of!(Qwen36MoeInt4WeightDesc, packed_row_stride_bytes),
            16
        );
        assert_eq!(
            offset_of!(Qwen36MoeInt4WeightDesc, packed_expert_stride_bytes),
            24
        );
        assert_eq!(
            offset_of!(Qwen36MoeInt4WeightDesc, scale_row_stride_elements),
            32
        );
        assert_eq!(
            offset_of!(Qwen36MoeInt4WeightDesc, scale_expert_stride_elements),
            40
        );
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, input_group_size), 48);
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, output_group_size), 52);
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, implicit_zero_code), 56);
        assert_eq!(offset_of!(Qwen36MoeInt4WeightDesc, encoding), 60);

        assert_eq!(size_of::<Qwen36MoeInt4ScaleDesc>(), 768);
        assert_eq!(align_of::<Qwen36MoeInt4ScaleDesc>(), 8);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, q_proj), 0);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, k_proj), 64);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, v_proj), 128);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, o_proj), 192);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_in_proj_qkv), 256);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_in_proj_z), 320);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, linear_out_proj), 384);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, experts_gate_up), 448);
        assert_eq!(offset_of!(Qwen36MoeInt4ScaleDesc, experts_down), 512);
        assert_eq!(
            offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_gate_proj),
            576
        );
        assert_eq!(
            offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_up_proj),
            640
        );
        assert_eq!(
            offset_of!(Qwen36MoeInt4ScaleDesc, shared_expert_down_proj),
            704
        );
    }
}
