//! Shared Qwen3.6-MoE decode geometry, buffers, and routing types.
#![allow(dead_code)]

use std::ffi::c_void;

use gpu_hal::{GpuBuffer, ScalarType};
#[allow(unused_imports)]
pub use supersonic_runtime::qwen36_moe::types::{
    is_full_attn_layer, PositionPair, HYBRID_FULL_ATTN_STRIDE,
};

/// Geometry the chained decoder needs at every layer + the lm_head.
/// Mirrors the synthetic + production cases.
#[derive(Debug, Clone, Copy)]
pub struct MultiLayerGeom {
    pub hidden: i32,
    pub vocab: i32,
    pub num_layers: i32,
    pub rms_norm_eps: f32,

    // Full-attention (read iff a layer's `attn` is `Full`).
    pub num_attention_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub rope_theta: f32,

    // Linear-attention (read iff a layer's `attn` is `Linear`).
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,

    // MoE FFN (every layer).
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
}

/// Per-layer attention weight buffers. The two variants are mutually
/// exclusive: a layer is full xor linear. Selection happens at populate time
/// via [`is_full_attn_layer`]. When [`AttnLayerBuffers::Full::int4`] /
/// [`AttnLayerBuffers::Linear::int4`] is `Some`, the matching weight buffer
/// holds INT4 packed nibbles (`u8`, `[out, in/2]`) instead of BF16; the
/// sidecar carries `(scale, zero)` BF16 tiles `[out/gs, in/gs]` and the
/// active group_size.
pub enum AttnLayerBuffers {
    Full {
        input_norm_w: GpuBuffer,
        q_proj_w: GpuBuffer,
        k_proj_w: GpuBuffer,
        v_proj_w: GpuBuffer,
        q_norm_w: GpuBuffer,
        k_norm_w: GpuBuffer,
        o_proj_w: GpuBuffer,
        int4: Option<FullAttnInt4Sidecars>,
        /// KV cache for this full-attention layer. When `Some`, the kernel
        /// writes the current step's K/V at slot `position` and attends over
        /// `kv_len = position + 1` past tokens. When `None` (parity tests,
        /// single-token decode), back-compat kv_len=1.
        kv_cache: Option<FullAttnKvCache>,
    },
    Linear {
        input_norm_w: GpuBuffer,
        in_proj_qkv_w: GpuBuffer,
        in_proj_z_w: GpuBuffer,
        in_proj_a_w: GpuBuffer,
        in_proj_b_w: GpuBuffer,
        conv1d_w: GpuBuffer,
        conv1d_bias: Option<GpuBuffer>,
        dt_bias: GpuBuffer,
        a_log: GpuBuffer,
        norm_w: GpuBuffer,
        out_proj_w: GpuBuffer,
        // Conv state ([qkv_dim, kernel-1] BF16) and recurrent state
        // ([V*K*Vd] F32), both mutated in place by the kernel.
        conv_state: GpuBuffer,
        recurrent_state: GpuBuffer,
        int4: Option<LinearAttnInt4Sidecars>,
    },
}

/// KV cache for a full-attention layer. `[kv_max_t, num_kv_heads * head_dim]`
/// BF16 each (or U8 FP8 bytes when `kv_fp8` is on), mutated by the kernel.
///
/// Exactly one of `k` / `virtual_kv_cache_k` is `Some` (same for V).
/// `virtual_kv_cache_k` is `Some` when Qwen3.6 KV VMM was selected.
pub struct FullAttnKvCache {
    /// `Some` when dense KV backing was selected. `None` when
    /// `virtual_kv_cache_k` is `Some` (VMM-backed). Exactly one of `k` /
    /// `virtual_kv_cache_k` is `Some`. Same for V.
    pub k: Option<GpuBuffer>,
    pub v: Option<GpuBuffer>,
    pub kv_max_t: i32,
    /// Some only when KV-FP8 is on.
    pub kv_scale_k: Option<GpuBuffer>,
    pub kv_scale_v: Option<GpuBuffer>,
    /// Some only when KV-FP8 is on AND the sidecar is enabled.
    pub kv_shadow_k: Option<GpuBuffer>,
    pub kv_shadow_v: Option<GpuBuffer>,
    /// First absolute KV position covered by the sidecar (`-1` when
    /// the sidecar is disabled).
    pub kv_shadow_start: i32,
    /// Rolling sidecar capacity in positions (`0` when disabled).
    pub kv_shadow_window: i32,
    /// `Some` when Qwen3.6 KV VMM was selected at allocation time AND the
    /// backend supports VMM. Mutually exclusive with `k` (exactly one is
    /// `Some`).
    pub virtual_kv_cache_k: Option<gpu_hal::VirtualBuffer>,
    pub virtual_kv_cache_v: Option<gpu_hal::VirtualBuffer>,
    /// The VMM reservation's logical max_T (matches `kv_max_t` above).
    pub virtual_kv_max_t: Option<usize>,
}

impl FullAttnKvCache {
    /// Device pointer to the K cache, regardless of VMM vs dense backing.
    pub fn k_device_ptr(&mut self) -> *mut std::ffi::c_void {
        if let Some(vk) = self.virtual_kv_cache_k.as_mut() {
            vk.as_mut_ptr()
        } else if let Some(b) = self.k.as_mut() {
            b.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        }
    }

    /// Device pointer to the V cache, regardless of VMM vs dense backing.
    pub fn v_device_ptr(&mut self) -> *mut std::ffi::c_void {
        if let Some(vv) = self.virtual_kv_cache_v.as_mut() {
            vv.as_mut_ptr()
        } else if let Some(b) = self.v.as_mut() {
            b.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        }
    }
}

/// INT4 sidecars for a full-attention layer.
pub struct FullAttnInt4Sidecars {
    pub group_size: i32,
    pub q_proj_type: i32,
    pub q_proj_scale: GpuBuffer,
    pub q_proj_zero: GpuBuffer,
    pub k_proj_type: i32,
    pub k_proj_scale: GpuBuffer,
    pub k_proj_zero: GpuBuffer,
    pub v_proj_type: i32,
    pub v_proj_scale: GpuBuffer,
    pub v_proj_zero: GpuBuffer,
    pub o_proj_type: i32,
    pub o_proj_scale: GpuBuffer,
    pub o_proj_zero: GpuBuffer,
}

/// INT4 sidecars for a linear-attention layer.
pub struct LinearAttnInt4Sidecars {
    pub group_size: i32,
    pub in_proj_qkv_type: i32,
    pub in_proj_qkv_scale: GpuBuffer,
    pub in_proj_qkv_zero: GpuBuffer,
    pub in_proj_z_type: i32,
    pub in_proj_z_scale: GpuBuffer,
    pub in_proj_z_zero: GpuBuffer,
    pub out_proj_type: i32,
    pub out_proj_scale: GpuBuffer,
    pub out_proj_zero: GpuBuffer,
}

/// INT4 sidecars for an MoE FFN block. The router (`gate_w`) and scalar
/// `shared_expert_gate` stay BF16.
pub struct FfnInt4Sidecars {
    pub group_size: i32,
    pub gate_up_proj_type: i32,
    pub gate_up_proj_scale: GpuBuffer,
    pub gate_up_proj_zero: GpuBuffer,
    pub down_proj_type: i32,
    pub down_proj_scale: GpuBuffer,
    pub down_proj_zero: GpuBuffer,
    pub shared_gate_proj_type: i32,
    pub shared_gate_proj_scale: GpuBuffer,
    pub shared_gate_proj_zero: GpuBuffer,
    pub shared_up_proj_type: i32,
    pub shared_up_proj_scale: GpuBuffer,
    pub shared_up_proj_zero: GpuBuffer,
    pub shared_down_proj_type: i32,
    pub shared_down_proj_scale: GpuBuffer,
    pub shared_down_proj_zero: GpuBuffer,
}

/// GPU-resident weight pointer used by decode descriptors.
#[allow(dead_code)]
pub enum ResidentWeight {
    Dense(GpuBuffer),
    Virtual {
        allocation_id: usize,
        ptr: *const c_void,
        dtype: ScalarType,
        shape: Vec<usize>,
        len_bytes: usize,
    },
}

unsafe impl Send for ResidentWeight {}
unsafe impl Sync for ResidentWeight {}

#[allow(dead_code)]
impl ResidentWeight {
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            Self::Dense(buf) => buf.as_ptr(),
            Self::Virtual { ptr, .. } => *ptr,
        }
    }

    pub fn len_bytes(&self) -> usize {
        match self {
            Self::Dense(buf) => buf.len_bytes(),
            Self::Virtual { len_bytes, .. } => *len_bytes,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(buf) => buf.shape(),
            Self::Virtual { shape, .. } => shape.as_slice(),
        }
    }

    pub fn dtype(&self) -> ScalarType {
        match self {
            Self::Dense(buf) => buf.dtype(),
            Self::Virtual { dtype, .. } => *dtype,
        }
    }

    pub fn allocation_id(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::Virtual { allocation_id, .. } => Some(*allocation_id),
        }
    }
}

impl From<GpuBuffer> for ResidentWeight {
    fn from(value: GpuBuffer) -> Self {
        Self::Dense(value)
    }
}

/// Per-layer MoE FFN weight buffers. Always present (every layer has an
/// FFN block). When `int4` is `Some`, every `*_proj_w` field carries
/// packed nibbles instead of BF16 weights.
pub struct FfnLayerBuffers {
    pub post_attn_norm_w: GpuBuffer,
    pub gate_w: GpuBuffer,
    pub gate_up_proj_w: ResidentWeight,
    pub down_proj_w: ResidentWeight,
    pub shared_gate_proj_w: GpuBuffer,
    pub shared_up_proj_w: GpuBuffer,
    pub shared_down_proj_w: GpuBuffer,
    pub shared_expert_gate_w: GpuBuffer,
    pub int4: Option<FfnInt4Sidecars>,
}

/// One layer's worth of GPU-resident weight + state buffers.
pub struct LayerBuffers {
    pub attn: AttnLayerBuffers,
    pub ffn: FfnLayerBuffers,
}

/// Multi-token-prediction (MTP) head weights for Qwen3.6-MoE
/// self-speculative decode. The MTP block is structurally one full-attention
/// MoE layer plus fusion RMSNorms/linear, sharing `embed_tokens` and
/// `lm_head` with the base model.
pub struct MtpLayerBuffers {
    /// Fusion linears + norms (top-level `mtp.*` tensors).
    pub pre_fc_norm_hidden_w: GpuBuffer, // [hidden]
    pub pre_fc_norm_embedding_w: GpuBuffer, // [hidden]
    pub fc_w: GpuBuffer,                    // [hidden, 2*hidden]
    pub norm_w: GpuBuffer,                  // [hidden]

    /// Single-layer full-attn block (`mtp.layers.0.*`).
    pub input_norm_w: GpuBuffer, // [hidden]
    pub post_attn_norm_w: GpuBuffer, // [hidden]
    pub q_proj_w: GpuBuffer,         // [2*H*d, hidden]
    pub k_proj_w: GpuBuffer,         // [Hkv*d, hidden]
    pub v_proj_w: GpuBuffer,         // [Hkv*d, hidden]
    pub o_proj_w: GpuBuffer,         // [hidden, H*d]
    pub q_norm_w: GpuBuffer,         // [head_dim]
    pub k_norm_w: GpuBuffer,         // [head_dim]

    /// MoE FFN sub-block (`mtp.layers.0.mlp.*`).
    pub gate_w: GpuBuffer, // [num_experts, hidden]
    pub gate_up_proj_w: GpuBuffer,       // [num_experts, 2*I, hidden]
    pub down_proj_w: GpuBuffer,          // [num_experts, hidden, I]
    pub shared_gate_proj_w: GpuBuffer,   // [Is, hidden]
    pub shared_up_proj_w: GpuBuffer,     // [Is, hidden]
    pub shared_down_proj_w: GpuBuffer,   // [hidden, Is]
    pub shared_expert_gate_w: GpuBuffer, // [1, hidden]

    /// Per-step KV cache for the MTP layer's self-attention.
    pub kv_cache: Option<FullAttnKvCache>,
}

impl LayerBuffers {
    pub fn is_full_attn(&self) -> bool {
        matches!(self.attn, AttnLayerBuffers::Full { .. })
    }
}

/// Captured intermediates from a chained decode pass. The per-layer hiddens
/// are useful for granular parity diagnostics; `final_hidden_bytes` is what
/// final RMSnorm + lm_head consume.
pub struct DecodeOutputs {
    pub path_label: &'static str,
    pub final_hidden_bytes: Vec<u8>,
    pub per_layer_attn_out: Vec<Vec<u8>>,
    pub per_layer_ffn_out: Vec<Vec<u8>>,
    pub kernel_full_attn_us: u64,
    pub kernel_linear_attn_us: u64,
    pub kernel_ffn_us: u64,
    pub sparse_lookahead_prefetch_us: u64,
    pub sparse_router_launch_us: u64,
    pub sparse_route_d2h_us: u64,
    pub sparse_demand_prefetch_us: u64,
    pub sparse_ffn_launch_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPrefetchPhase {
    Lookahead,
    Demand,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertRoute {
    pub rank: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_pattern_marks_every_fourth_layer_full() {
        for li in 0..40 {
            let expect_full = (li + 1) % 4 == 0;
            assert_eq!(is_full_attn_layer(li), expect_full, "layer {li}");
        }
    }
}
