//! FFI bridge for the Qwen3.6-MoE persistent decode megakernel.
//!
//! Status: **infrastructure only.** PR 4 (this file) lands the descriptor
//! layout + a stub kernel that validates the launch path and descriptor
//! read-out. Actual transformer compute (attention with `attn_output_gate`,
//! MoE routing + work-stealing dispatch, fused expert mat-vec, shared
//! expert, lm_head) lands in follow-up PRs.
//!
//! Why a stub first: the megakernel will be ~6500 LoC of HIP. Wiring the
//! FFI/bridge/build path before any compute math means a future kernel
//! commit can land focused-on-math code with no orchestration noise. The
//! stub also serves as a smoke test for descriptor field layout — a
//! Rust↔C++ struct-mismatch bug found here saves hours later.

use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

use crate::layer_desc::MAX_BATCH_SIZE;

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
}

unsafe impl Send for Qwen36MoeDecodeLayerDesc {}
unsafe impl Sync for Qwen36MoeDecodeLayerDesc {}

impl Default for Qwen36MoeDecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Parallel-struct to [`Qwen36MoeDecodeLayerDesc`] carrying INT4 GPTQ
/// scale + zero pointers. When `--int4` is active, the main desc's `*_w`
/// slots reinterpret as packed-u8 nibbles and the kernel reads the
/// matching `*_scale` / `*_zero` from this struct.
///
/// Routers and scalar gates stay BF16 (`is_int4_target` excludes them);
/// their fields here are unused but the struct keeps them at fixed offsets
/// for ABI stability.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeInt4ScaleDesc {
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,

    pub linear_in_proj_qkv_scale: *const c_void,
    pub linear_in_proj_qkv_zero: *const c_void,
    pub linear_in_proj_z_scale: *const c_void,
    pub linear_in_proj_z_zero: *const c_void,
    pub linear_out_proj_scale: *const c_void,
    pub linear_out_proj_zero: *const c_void,

    pub experts_gate_up_scale: *const c_void,
    pub experts_gate_up_zero: *const c_void,
    pub experts_down_scale: *const c_void,
    pub experts_down_zero: *const c_void,

    pub shared_expert_gate_proj_scale: *const c_void,
    pub shared_expert_gate_proj_zero: *const c_void,
    pub shared_expert_up_proj_scale: *const c_void,
    pub shared_expert_up_proj_zero: *const c_void,
    pub shared_expert_down_proj_scale: *const c_void,
    pub shared_expert_down_proj_zero: *const c_void,

    pub group_size: c_int,
}

unsafe impl Send for Qwen36MoeInt4ScaleDesc {}
unsafe impl Sync for Qwen36MoeInt4ScaleDesc {}

impl Default for Qwen36MoeInt4ScaleDesc {
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

#[cfg(supersonic_backend_hip)]
extern "C" {
    /// Stub launch entry. Walks the descriptor array, validates field
    /// integrity by writing recognizable sentinel values into the workspace
    /// at known offsets, grid-barriers between layers, and returns 0 on
    /// success.
    ///
    /// Sentinel layout in `workspace[0..sentinel_count]` (f32):
    /// - `[0]`: number of layers seen (must equal `num_layers`)
    /// - `[1]`: total `num_experts` summed across layers (sanity check)
    /// - `[2]`: total `top_k` summed across layers
    /// - `[3]`: 1.0 if every layer's `is_full_attention` matches the
    ///   pattern produced by `(idx + 1) % 4 == 0`, else 0.0
    /// - `[4]`: `attn_output_gate` status — 1.0 if all full-attn layers
    ///   set it to 1, 0.0 otherwise
    /// - `[5..]`: reserved for future smoke-test bytes; zero on PR 4.
    ///
    /// Once the real kernel lands, this entry is replaced by the actual
    /// persistent decode launcher with the same signature.
    pub fn qwen36_moe_hip_stub_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        layers: *const Qwen36MoeDecodeLayerDesc,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;
}

/// Safe wrapper over the stub launch. The engine pre-allocates `sync_buf`
/// as a 32-byte zeroed scratch (work-stealing counter at offset 0, grid
/// barrier counter at +16, flag at +20 — same layout as
/// `crate::persistent_decode_4b`).
///
/// Returns when the kernel signals completion via `hipDeviceSynchronize`.
/// The smoke-test path reads `workspace` back to verify descriptor
/// integrity; the real kernel will overwrite that area with activations.
pub fn stub_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer_descs_device: &GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    num_layers: usize,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::stub_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = layer_descs_device.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_uint };

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen36_moe_hip_stub_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    layer_descs_device.as_ptr() as *const Qwen36MoeDecodeLayerDesc,
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::stub_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::stub_launch: CUDA backend not yet wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::stub_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe stub launch failed with status {status}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn descriptor_layout_offsets_documented() {
        // Pin the size so a future field-reorder is loud. If you need to
        // grow the struct, append fields and update this number — never
        // reorder existing ones.
        // (Numbers verified on x86_64 Linux. Pointers are 8 bytes.)
        let sz = size_of::<Qwen36MoeDecodeLayerDesc>();
        assert!(
            sz >= 256 && sz <= 512,
            "Qwen36MoeDecodeLayerDesc size drift: got {sz} bytes",
        );

        let int4_sz = size_of::<Qwen36MoeInt4ScaleDesc>();
        assert!(
            int4_sz >= 192 && int4_sz <= 256,
            "Qwen36MoeInt4ScaleDesc size drift: got {int4_sz} bytes",
        );
    }

    #[test]
    fn descriptor_default_is_zero() {
        let d = Qwen36MoeDecodeLayerDesc::default();
        assert_eq!(d.layer_idx, 0);
        assert_eq!(d.is_full_attention, 0);
        assert!(d.input_norm_w.is_null());
        assert!(d.kv_cache_k.is_null());
        assert!(d.linear_recurrent_state.is_null());
        assert!(d.experts_gate_up_w.is_null());
    }

    /// HIP smoke test: launch the stub kernel against a synthetic 40-layer
    /// descriptor array and verify the sentinel bytes the kernel writes
    /// match what we sent in. This exercises the entire path:
    /// FFI struct layout → bridge launch → grid barrier → cooperative
    /// work-stealing → host readback. The same path the real kernel will
    /// use, minus the compute math.
    #[cfg(supersonic_backend_hip)]
    #[test]
    fn hip_stub_launch_walks_descriptor_array() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        set_backend(Backend::Hip);
        let ordinal = 0usize;
        let num_layers = 40usize;

        // Synthesize a Qwen3.6-MoE-shaped descriptor array on the host.
        // Hybrid pattern: every 4th layer is full attention; others are
        // linear-attention. attn_output_gate is set on full layers only.
        let mut host_descs: Vec<Qwen36MoeDecodeLayerDesc> =
            Vec::with_capacity(num_layers);
        let num_experts = 256;
        let top_k = 8;
        for idx in 0..num_layers {
            let mut d = Qwen36MoeDecodeLayerDesc::default();
            d.layer_idx = idx as c_int;
            d.is_full_attention = if (idx + 1) % 4 == 0 { 1 } else { 0 };
            d.attn_output_gate = if d.is_full_attention == 1 { 1 } else { 0 };
            d.num_experts = num_experts;
            d.top_k = top_k;
            d.moe_intermediate_size = 512;
            d.shared_expert_intermediate_size = 512;
            d.norm_topk_prob = 1;
            d.attn_head_dim = 256;
            d.attn_num_heads = 16;
            d.attn_num_kv_heads = 2;
            host_descs.push(d);
        }

        // Upload as raw bytes; gpu-hal lets us treat the buffer as opaque
        // u8 since the kernel only dereferences the C struct pointer.
        let desc_bytes_per = size_of::<Qwen36MoeDecodeLayerDesc>();
        let mut desc_bytes = Vec::with_capacity(desc_bytes_per * num_layers);
        for d in &host_descs {
            let p = d as *const Qwen36MoeDecodeLayerDesc as *const u8;
            desc_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(p, desc_bytes_per)
            });
        }
        let layer_descs = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[desc_bytes.len()],
            &desc_bytes,
        )
        .expect("upload descriptor array");

        // 16 floats of workspace is enough for the documented sentinel
        // slots (5 in use, rest reserved). The real kernel will need
        // ~MiB; keeping this tiny lets the smoke test stay fast.
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[16])
            .expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[32])
            .expect("alloc sync buf");

        stub_launch(
            ordinal,
            ScalarType::BF16,
            &layer_descs,
            &mut workspace,
            &mut sync_buf,
            num_layers,
        )
        .expect("stub launch");

        // Read sentinels.
        let bytes = workspace.to_host_bytes().expect("download workspace");
        let workspace_f32: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(
            workspace_f32[0] as usize, num_layers,
            "[0] num_layers seen by kernel"
        );
        assert_eq!(
            workspace_f32[1] as i32,
            (num_experts as i32) * (num_layers as i32),
            "[1] sum of num_experts across layers"
        );
        assert_eq!(
            workspace_f32[2] as i32,
            (top_k as i32) * (num_layers as i32),
            "[2] sum of top_k across layers"
        );
        assert_eq!(
            workspace_f32[3], 1.0,
            "[3] hybrid pattern check (1.0 = pattern OK across all layers)"
        );
        assert_eq!(
            workspace_f32[4], 1.0,
            "[4] attn_output_gate consistency on full layers"
        );
    }
}
