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

    /// PR 4b2 staged single-layer attention parity launcher. Runs the
    /// full-attention path through `stage` (1..=5) and writes the matching
    /// intermediate to `output`:
    ///
    /// | stage | output buffer contents (BF16)                      |
    /// |-------|----------------------------------------------------|
    /// |   1   | `q_normed[H*d]`                                    |
    /// |   2   | `k_normed[Hkv*d]`         (`q_normed` recomputed)  |
    /// |   3   | `q_rot[H*d] || k_rot[Hkv*d]` (planned)             |
    /// |   4   | `attn[H*d]`                                        |
    /// |   5   | `output_hidden[hidden]`                            |
    ///
    /// At PR 4b2 step 1 only `stage == 1` is wired; the kernel returns the
    /// q-path intermediate and ignores the k_*/v_*/o_proj/RoPE/position
    /// arguments. They're declared up front so the FFI ABI doesn't change
    /// between staged commits.
    ///
    /// `workspace` must be at least `2 * num_heads * head_dim` F32 entries
    /// (used to hold the BF16-rounded F32 view of `q_raw` between phases).
    /// `output` must be at least `num_heads * head_dim` BF16 entries on
    /// stage 1 — sized for the largest staged intermediate, BF16.
    /// `sync_buf` (counters/barrier_counter/barrier_flag) must be 32 zero
    /// bytes — see [`stub_launch`] for the layout convention.
    pub fn qwen36_moe_hip_attn_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        rope_theta: f32,
        rms_norm_eps: f32,
        position: c_int,
        input_hidden: *const c_void,
        input_norm_w: *const c_void,
        q_proj_w: *const c_void,
        k_proj_w: *const c_void,
        v_proj_w: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_w: *const c_void,
        output: *mut c_void,
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

/// Geometry + position state for the staged-attention parity launcher.
/// These are constants of the layer being tested; bundling them into a
/// struct keeps the safe wrapper below from sprouting eight scalar args.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeAttnStepParams {
    pub stage: i32,
    pub hidden: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub position: i32,
}

/// Weight pointers for the staged-attention parity launcher. Pointers
/// unused by the requested `stage` may be null; the kernel won't dereference
/// them. See [`qwen36_moe_hip_attn_step_launch`] for the per-stage matrix.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeAttnStepWeights {
    pub input_hidden: *const c_void,
    pub input_norm_w: *const c_void,
    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub o_proj_w: *const c_void,
}

/// Safe wrapper for the PR 4b2 staged-attention parity launcher.
///
/// `output` must be a BF16 buffer with at least `num_heads * head_dim`
/// elements (the size of the largest staged intermediate). `workspace` must
/// be an F32 buffer with at least `2 * num_heads * head_dim` elements.
/// `sync_buf` must be a 32-byte zero buffer (counter @ +0, barrier counter
/// @ +16, barrier flag @ +20).
pub fn attn_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeAttnStepParams,
    weights: &Qwen36MoeAttnStepWeights,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    // All five stages are wired through PR 4b2 step 5.

    let backend = output.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_uint };

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen36_moe_hip_attn_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_heads as c_int,
                    params.num_kv_heads as c_int,
                    params.head_dim as c_int,
                    params.rotary_dim as c_int,
                    params.rope_theta,
                    params.rms_norm_eps,
                    params.position as c_int,
                    weights.input_hidden,
                    weights.input_norm_w,
                    weights.q_proj_w,
                    weights.k_proj_w,
                    weights.v_proj_w,
                    weights.q_norm_w,
                    weights.k_norm_w,
                    weights.o_proj_w,
                    output.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::attn_step_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::attn_step_launch: CUDA backend not yet wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::attn_step_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe attn_step launch failed with status {status}"),
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

    // ---- PR 4b2 step 1: q-path parity vs the PyTorch oracle --------------
    //
    // The test reads a JSON produced by `oracle/qwen36_moe_oracle.py`
    // (synthetic or checkpoint mode), uploads the four input tensors needed
    // for stage 1 (input_hidden, input_norm_w, q_proj_w, q_norm_w), runs
    // the staged kernel, downloads the BF16 q_normed output, and compares
    // against `intermediates.q_normed` from the oracle.
    //
    // To run: produce a JSON, then point the test at it via env var:
    //
    //   python oracle/qwen36_moe_oracle.py --mode synthetic \
    //       --hidden 2048 --num-attention-heads 16 --num-kv-heads 2 \
    //       --head-dim 256 --out /tmp/qwen36_syn.json
    //   SUPERSONIC_QWEN36_ORACLE_JSON=/tmp/qwen36_syn.json \
    //       cargo test --release -p kernel-ffi qwen36_moe_attn_step_1
    //
    // Without the env var the test prints a clear skip message and exits.
    // We don't fail-on-missing because the FFI test must remain runnable
    // on hosts without Python/PyTorch (CI without GPU, header-only checks).

    /// Decode a base64 string to bytes. Inline so the test stays
    /// dependency-free aside from serde_json. RFC 4648 alphabet, no padding
    /// tolerance shortcuts (we know the oracle always emits valid BF16
    /// payloads ≡ even-length byte streams ≡ 4n base64 chars after padding).
    #[cfg(supersonic_backend_hip)]
    fn b64_decode(input: &str) -> Vec<u8> {
        const TABLE: &[u8; 256] = &{
            let mut t = [255u8; 256];
            let mut i = 0;
            let charset = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            while i < charset.len() {
                t[charset[i] as usize] = i as u8;
                i += 1;
            }
            t
        };
        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut buf = 0u32;
        let mut bits = 0;
        for &b in input.as_bytes() {
            if b == b'=' || b == b'\n' || b == b'\r' || b == b' ' {
                continue;
            }
            let v = TABLE[b as usize];
            assert!(v != 255, "qwen36_moe parity: invalid base64 byte {b:#x}");
            buf = (buf << 6) | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xFF) as u8);
            }
        }
        out
    }

    /// Convert a stream of BF16 little-endian bytes to F32. The oracle
    /// stores BF16 as raw int16 → bytes, matching the kernel's ABI.
    #[cfg(supersonic_backend_hip)]
    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        assert!(bytes.len() % 2 == 0, "qwen36_moe parity: BF16 bytes must be even");
        bytes
            .chunks_exact(2)
            .map(|c| {
                // BF16 = top 16 bits of an F32. Reconstruct by zero-extending.
                let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                f32::from_bits(bits << 16)
            })
            .collect()
    }

    /// Geometry pulled from the oracle JSON's `config` block — pinned to
    /// what every parity test in this file consumes.
    #[cfg(supersonic_backend_hip)]
    struct OracleGeom {
        hidden: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_oracle_json() -> Option<(serde_json::Value, OracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read oracle json {json_path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&raw).expect("oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "PR 4b2 parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = OracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_heads: cfg["num_attention_heads"].as_i64().unwrap() as i32,
            num_kv_heads: cfg["num_kv_heads"].as_i64().unwrap() as i32,
            head_dim: cfg["head_dim"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    /// Compare a kernel-produced BF16 buffer against the matching oracle
    /// intermediate, emit a one-line summary, and assert tolerances. BF16
    /// stores at every boundary mean most elements are bit-exact; the rare
    /// 1-ULP misses come from F32 accumulation-order drift in the matmul.
    #[cfg(supersonic_backend_hip)]
    fn assert_parity(
        label: &str,
        got_bytes: &[u8],
        want_bytes: &[u8],
        max_abs_tol: f32,
        cos_sim_floor: f64,
    ) {
        assert_eq!(
            got_bytes.len(), want_bytes.len(),
            "{label}: byte length mismatch"
        );
        let got = bf16_bytes_to_f32(got_bytes);
        let want = bf16_bytes_to_f32(want_bytes);
        let n = got.len();
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut dot = 0.0f64;
        let mut got_sq = 0.0f64;
        let mut want_sq = 0.0f64;
        let mut exact = 0usize;
        for i in 0..n {
            let d = (got[i] - want[i]).abs();
            if d == 0.0 { exact += 1; }
            max_abs_diff = max_abs_diff.max(d);
            sum_abs_diff += d;
            dot += got[i] as f64 * want[i] as f64;
            got_sq += (got[i] as f64).powi(2);
            want_sq += (want[i] as f64).powi(2);
        }
        let cos_sim = dot / (got_sq.sqrt() * want_sq.sqrt() + 1e-30);
        let mean_abs_diff = sum_abs_diff / n as f32;
        eprintln!(
            "[parity {label}] n={n} exact={exact} max_abs={max_abs_diff:.5e} \
             mean_abs={mean_abs_diff:.5e} cos_sim={cos_sim:.7}"
        );
        assert!(
            max_abs_diff <= max_abs_tol,
            "{label}: max_abs={max_abs_diff} exceeds tolerance {max_abs_tol}"
        );
        assert!(
            cos_sim >= cos_sim_floor,
            "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}"
        );
    }

    /// Returns workspace size sufficient for the largest staged
    /// intermediate. Stage 5 is the final stage: 6*H*d + 4*Hkv*d + hidden F32.
    #[cfg(supersonic_backend_hip)]
    fn parity_workspace_floats(
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        hidden: i32,
    ) -> usize {
        let h = num_heads as usize;
        let hkv = num_kv_heads as usize;
        let d = head_dim as usize;
        let hd = hidden as usize;
        6 * h * d + 4 * hkv * d + hd
    }

    /// Output buffer size sufficient for the largest staged intermediate.
    /// Stage 3 publishes q_rot || k_rot, so the buffer must hold (H + Hkv)*d
    /// BF16 elements; stages 1, 2, 4, 5 fit in a strict subset of that.
    #[cfg(supersonic_backend_hip)]
    fn parity_output_elems(num_heads: i32, num_kv_heads: i32, head_dim: i32) -> usize {
        let h = num_heads as usize;
        let hkv = num_kv_heads as usize;
        let d = head_dim as usize;
        h * d + hkv * d
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_1_q_normed_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_oracle.py --mode synthetic --out /tmp/syn.json` \
                 and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let q_normed_expected_bytes = b64_decode(inters["q_normed"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(input_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(q_proj_w_bytes.len(), 2 * h_us * d_us * hidden_us * 2);
        assert_eq!(q_norm_w_bytes.len(), d_us * 2);
        assert_eq!(q_normed_expected_bytes.len(), h_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[2 * h_us * d_us, hidden_us], &q_proj_w_bytes,
        ).expect("upload q_proj_w");
        let q_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes,
        ).expect("upload q_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16,
            &[parity_output_elems(geom.num_heads, geom.num_kv_heads, geom.head_dim)],
        ).expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32,
            &[parity_workspace_floats(geom.num_heads, geom.num_kv_heads, geom.head_dim, geom.hidden)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 1,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: 0,
            rope_theta: 0.0,
            rms_norm_eps: geom.rms_norm_eps,
            position: 0,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 1");

        // Stage 1 publishes q_normed into the first H*d BF16 elements of
        // the (now stage-3-sized) output buffer. Slice down to just those
        // bytes for parity. BF16 ULP at magnitude 1 is ~7.8e-3; allow 4×
        // that for matmul accumulation order drift over the 2048 reduction.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..h_us * d_us * 2];
        assert_parity("step1 q_normed", got_bytes, &q_normed_expected_bytes, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_2_k_normed_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still runs the stage-1 prerequisite (q_normed lives in
        // workspace for later RoPE), so the kernel needs the q-side weights
        // even though we only verify k_normed here.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let k_normed_expected_bytes = b64_decode(inters["k_normed"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(k_proj_w_bytes.len(), hkv_us * d_us * hidden_us * 2);
        assert_eq!(v_proj_w_bytes.len(), hkv_us * d_us * hidden_us * 2);
        assert_eq!(k_norm_w_bytes.len(), d_us * 2);
        assert_eq!(k_normed_expected_bytes.len(), hkv_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[2 * h_us * d_us, hidden_us], &q_proj_w_bytes,
        ).expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &k_proj_w_bytes,
        ).expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &v_proj_w_bytes,
        ).expect("upload v_proj_w");
        let q_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes,
        ).expect("upload q_norm_w");
        let k_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes,
        ).expect("upload k_norm_w");

        // Output is sized for the largest staged intermediate (H*d). Stage 2
        // writes Hkv*d BF16 elements at the start of the buffer.
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16,
            &[parity_output_elems(geom.num_heads, geom.num_kv_heads, geom.head_dim)],
        ).expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32,
            &[parity_workspace_floats(geom.num_heads, geom.num_kv_heads, geom.head_dim, geom.hidden)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: 0,
            rope_theta: 0.0,
            rms_norm_eps: geom.rms_norm_eps,
            position: 0,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 2");

        // Stage 2 publishes k_normed into the first Hkv*d BF16 elements of
        // the output buffer. Slice down to just those bytes for parity.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hkv_us * d_us * 2];
        assert_parity("step2 k_normed", got_bytes, &k_normed_expected_bytes, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_3_qk_rot_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 Generate at a non-zero position so RoPE is non-identity \
                 (e.g. `python oracle/qwen36_moe_oracle.py --mode synthetic \
                 --position 7 --out /tmp/qwen36_syn_pos7.json`)."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        let position = json["position"].as_i64().unwrap_or(0) as i32;
        if position == 0 {
            // RoPE at position 0 is the identity rotation — the parity
            // test would pass even with a no-op kernel, which defeats the
            // purpose. Refuse and tell the caller how to fix it.
            panic!(
                "step3 RoPE parity requires position > 0, got 0. \
                 Re-run the oracle with `--position 7` (or any non-zero value)."
            );
        }

        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let q_rot_expected_bytes = b64_decode(inters["q_rot"].as_str().unwrap());
        let k_rot_expected_bytes = b64_decode(inters["k_rot"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(q_rot_expected_bytes.len(), h_us * d_us * 2);
        assert_eq!(k_rot_expected_bytes.len(), hkv_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[2 * h_us * d_us, hidden_us], &q_proj_w_bytes,
        ).expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &k_proj_w_bytes,
        ).expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &v_proj_w_bytes,
        ).expect("upload v_proj_w");
        let q_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes,
        ).expect("upload q_norm_w");
        let k_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes,
        ).expect("upload k_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16,
            &[parity_output_elems(geom.num_heads, geom.num_kv_heads, geom.head_dim)],
        ).expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32,
            &[parity_workspace_floats(geom.num_heads, geom.num_kv_heads, geom.head_dim, geom.hidden)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 3,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 3");

        // Output layout: [q_rot (H*d) | k_rot (Hkv*d)] in BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let q_end = h_us * d_us * 2;
        let k_end = q_end + hkv_us * d_us * 2;
        assert_parity("step3 q_rot", &got_bytes_full[..q_end],
                      &q_rot_expected_bytes, 0.04, 0.9999);
        assert_parity("step3 k_rot", &got_bytes_full[q_end..k_end],
                      &k_rot_expected_bytes, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_4_attn_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        // Stage 4 still walks the stage-3 RoPE prerequisite, so we need
        // the same RoPE config + non-trivial position discipline as step 3.
        let position = json["position"].as_i64().unwrap_or(0) as i32;
        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let attn_expected_bytes = b64_decode(inters["attn"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(attn_expected_bytes.len(), h_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[2 * h_us * d_us, hidden_us], &q_proj_w_bytes,
        ).expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &k_proj_w_bytes,
        ).expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &v_proj_w_bytes,
        ).expect("upload v_proj_w");
        let q_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes,
        ).expect("upload q_norm_w");
        let k_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes,
        ).expect("upload k_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16,
            &[parity_output_elems(geom.num_heads, geom.num_kv_heads, geom.head_dim)],
        ).expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32,
            &[parity_workspace_floats(geom.num_heads, geom.num_kv_heads, geom.head_dim, geom.hidden)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 4,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 4");

        // Stage 4 publishes attn into output[0..H*d) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..h_us * d_us * 2];
        // kv_len=1 makes softmax trivially 1.0, so attn = v_full and the
        // parity should be effectively bit-exact (both kernel and oracle
        // skip any precision-losing accumulation here).
        assert_parity("step4 attn", got_bytes, &attn_expected_bytes, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        let position = json["position"].as_i64().unwrap_or(0) as i32;
        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let o_proj_w_bytes = b64_decode(weights["o_proj_w"].as_str().unwrap());
        let output_hidden_expected_bytes =
            b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(o_proj_w_bytes.len(), hidden_us * h_us * d_us * 2);
        assert_eq!(output_hidden_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[2 * h_us * d_us, hidden_us], &q_proj_w_bytes,
        ).expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &k_proj_w_bytes,
        ).expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hkv_us * d_us, hidden_us], &v_proj_w_bytes,
        ).expect("upload v_proj_w");
        let q_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes,
        ).expect("upload q_norm_w");
        let k_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes,
        ).expect("upload k_norm_w");
        let o_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, h_us * d_us], &o_proj_w_bytes,
        ).expect("upload o_proj_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16,
            &[parity_output_elems(geom.num_heads, geom.num_kv_heads, geom.head_dim)],
        ).expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32,
            &[parity_workspace_floats(geom.num_heads, geom.num_kv_heads, geom.head_dim, geom.hidden)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: o_proj_w.as_ptr(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 5");

        // Stage 5 publishes output_hidden into output[0..hidden) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        // o_proj reduces over H*d=4096 lanes; allow more headroom on
        // max_abs_diff than the smaller stage-1 reduction (2048 lanes).
        // Cosine similarity is the more meaningful metric for the residual.
        assert_parity(
            "step5 output_hidden",
            got_bytes,
            &output_hidden_expected_bytes,
            0.05,
            0.9999,
        );
    }
}
