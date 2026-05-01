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

    /// PR 4b3 staged single-layer linear-attention parity launcher. Same
    /// staged-build-up discipline as `qwen36_moe_hip_attn_step_launch`,
    /// but for the 3-of-4 hybrid layers that aren't full-attention.
    /// `stage` selects how far to run; the matching staged intermediate
    /// is published to `output` (BF16):
    ///
    /// | stage | output buffer contents (BF16)           |
    /// |-------|------------------------------------------|
    /// |   1   | `qkv_raw[qkv_dim]`                       |
    /// |   2   | `silu_out[qkv_dim]`         (planned)    |
    /// |   3   | `q_scaled || k_rep || v_heads` (planned) |
    /// |   4   | `recurrent_out[V*v_dim]`    (planned)    |
    /// |   5   | `output_hidden[hidden]`     (planned)    |
    ///
    /// PR 4b3 step 2 wires only `stage == 1`; the kernel ignores the conv
    /// / dt / norm / out_proj / state pointers and the matching arguments
    /// can be null. They're declared up front so subsequent staged commits
    /// don't perturb the FFI ABI.
    ///
    /// `workspace` must be at least `qkv_dim + V*v_dim + 2*V` F32 entries
    /// for stage 1 (later stages bump that up via the safe wrapper).
    /// `output` must be at least `qkv_dim` BF16 entries on stage 1 (sized
    /// for the largest staged intermediate by the safe wrapper). `sync_buf`
    /// (counters/barrier_counter/barrier_flag) must be 32 zero bytes.
    pub fn qwen36_moe_hip_linear_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        conv_kernel_dim: c_int,
        rms_norm_eps: f32,
        input_hidden: *const c_void,
        input_norm_w: *const c_void,
        in_proj_qkv_w: *const c_void,
        in_proj_z_w: *const c_void,
        in_proj_a_w: *const c_void,
        in_proj_b_w: *const c_void,
        conv1d_w: *const c_void,
        conv1d_bias: *const c_void,
        dt_bias: *const c_void,
        a_log: *const c_void,
        norm_w: *const c_void,
        out_proj_w: *const c_void,
        conv_state: *mut c_void,
        recurrent_state: *mut f32,
        output: *mut c_void,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// PR 4b4 staged single-block MoE FFN parity launcher. Same staged-build-up
    /// discipline as `qwen36_moe_hip_attn_step_launch` and
    /// `qwen36_moe_hip_linear_step_launch`, but for the post-attention half
    /// of one Qwen3.6-MoE layer. `stage` selects how far to run; the matching
    /// staged intermediate is published to `output` (BF16) and `output_idx`
    /// (i32, top-k indices for stages 1+):
    ///
    /// | stage | output buffer contents (BF16)                    |
    /// |-------|--------------------------------------------------|
    /// |   1   | `topk_weights[k]`           (idx via `output_idx`) |
    /// |   2   | `shared_out[hidden]`                             |
    /// |   3   | `expert_0_out[hidden]`      (top-1 dispatch)     |
    /// |   4   | `moe_out[hidden]`                                |
    /// |   5   | `output_hidden[hidden]`     (final residual)     |
    ///
    /// PR 4b4 step 1 wires only `stage == 1`; the kernel ignores the
    /// gate_up_proj / down_proj / shared_expert_* pointers and the matching
    /// arguments can be null. They're declared up front so subsequent staged
    /// commits don't perturb the FFI ABI.
    ///
    /// `workspace` must be at least `hidden + 2*num_experts + 2*top_k` F32
    /// entries for stage 1 (later stages bump that up). `output` must be at
    /// least `top_k` BF16 entries on stage 1 and `output_idx` must be at
    /// least `top_k` i32 entries. `sync_buf` (counters/barrier_counter/
    /// barrier_flag) must be 32 zero bytes.
    ///
    /// PR 4b5 step 2: INT4 dequant smoke launcher.
    ///
    /// Drives a tiny single-thread kernel that runs both `int4_dequant_8`
    /// and `int4_dequant_scalar` over a `[out_rows, in_cols]` slab, writing
    /// each helper's outputs into a separate buffer. The Rust-side test
    /// validates byte-for-byte against a host reference computing the same
    /// `bf16(q*s - z*s)` reconstruction. Catches porting bugs in the
    /// helpers in isolation, before they're folded into the real FFN
    /// matmuls in step 3+.
    ///
    /// `packed`: u8, shape `[out_rows, in_cols / 2]`, even col → low nibble.
    /// `scale` / `zero`: BF16, shape `[out_rows / gsz, in_cols / gsz]`.
    /// `dq_8_out`, `dq_scalar_out`: F32 device buffers, each
    /// `out_rows * in_cols` long.
    ///
    /// Pre-conditions (the bridge validates them):
    /// - `in_cols % 8 == 0`
    /// - `in_cols % gsz == 0` and `gsz % 2 == 0`
    /// - `out_rows % gsz == 0`
    pub fn qwen36_moe_hip_int4_dequant_smoke_launch(
        device_ordinal: usize,
        packed: *const u8,
        scale: *const c_void,
        zero: *const c_void,
        out_rows: c_int,
        in_cols: c_int,
        gsz: c_int,
        dq_8_out: *mut f32,
        dq_scalar_out: *mut f32,
    ) -> c_int;

    pub fn qwen36_moe_hip_ffn_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_experts: c_int,
        moe_intermediate: c_int,
        shared_intermediate: c_int,
        top_k: c_int,
        rms_norm_eps: f32,
        input_hidden: *const c_void,
        post_attn_norm_w: *const c_void,
        gate_w: *const c_void,
        gate_up_proj_w: *const c_void,
        down_proj_w: *const c_void,
        shared_gate_proj_w: *const c_void,
        shared_up_proj_w: *const c_void,
        shared_down_proj_w: *const c_void,
        shared_expert_gate_w: *const c_void,
        int4_group_size: c_int,
        gate_up_proj_scale: *const c_void,
        gate_up_proj_zero: *const c_void,
        down_proj_scale: *const c_void,
        down_proj_zero: *const c_void,
        shared_gate_proj_scale: *const c_void,
        shared_gate_proj_zero: *const c_void,
        shared_up_proj_scale: *const c_void,
        shared_up_proj_zero: *const c_void,
        shared_down_proj_scale: *const c_void,
        shared_down_proj_zero: *const c_void,
        output: *mut c_void,
        output_idx: *mut c_int,
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

/// Geometry for the staged linear-attention parity launcher. Mirrors
/// `Qwen36MoeAttnStepParams`. Bundling these into a struct keeps the safe
/// wrapper from sprouting a long scalar arglist.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeLinearStepParams {
    pub stage: i32,
    pub hidden: i32,
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,
    pub rms_norm_eps: f32,
}

/// Weight + state pointers for the staged linear-attention parity launcher.
/// Pointers unused by the requested `stage` may be null; the kernel won't
/// dereference them. See [`qwen36_moe_hip_linear_step_launch`] for the
/// per-stage matrix.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeLinearStepWeights {
    pub input_hidden: *const c_void,
    pub input_norm_w: *const c_void,
    pub in_proj_qkv_w: *const c_void,
    pub in_proj_z_w: *const c_void,
    pub in_proj_a_w: *const c_void,
    pub in_proj_b_w: *const c_void,
    pub conv1d_w: *const c_void,
    pub conv1d_bias: *const c_void,
    pub dt_bias: *const c_void,
    pub a_log: *const c_void,
    pub norm_w: *const c_void,
    pub out_proj_w: *const c_void,
    pub conv_state: *mut c_void,
    pub recurrent_state: *mut f32,
}

/// Safe wrapper for the PR 4b3 staged linear-attention parity launcher.
/// Same workspace / sync_buf layout as
/// [`attn_step_launch`]: 32-byte zero scratch, F32 workspace sized for the
/// stage's footprint, BF16 output sized for the largest staged intermediate.
pub fn linear_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeLinearStepParams,
    weights: &Qwen36MoeLinearStepWeights,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    // All five stages are wired through PR 4b3 step 6.

    let backend = output.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_uint };

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen36_moe_hip_linear_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_k_heads as c_int,
                    params.num_v_heads as c_int,
                    params.head_k_dim as c_int,
                    params.head_v_dim as c_int,
                    params.conv_kernel_dim as c_int,
                    params.rms_norm_eps,
                    weights.input_hidden,
                    weights.input_norm_w,
                    weights.in_proj_qkv_w,
                    weights.in_proj_z_w,
                    weights.in_proj_a_w,
                    weights.in_proj_b_w,
                    weights.conv1d_w,
                    weights.conv1d_bias,
                    weights.dt_bias,
                    weights.a_log,
                    weights.norm_w,
                    weights.out_proj_w,
                    weights.conv_state,
                    weights.recurrent_state,
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
                    "qwen36_moe::linear_step_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::linear_step_launch: CUDA backend not yet wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::linear_step_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe linear_step launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Geometry for the staged MoE FFN parity launcher. Mirrors
/// `Qwen36MoeAttnStepParams` / `Qwen36MoeLinearStepParams`. Bundling these
/// keeps the safe wrapper's signature short.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepParams {
    pub stage: i32,
    pub hidden: i32,
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
    pub rms_norm_eps: f32,
}

/// Weight pointers for the staged MoE FFN parity launcher. Pointers unused
/// by the requested `stage` may be null; the kernel won't dereference them.
/// See [`qwen36_moe_hip_ffn_step_launch`] for the per-stage matrix.
///
/// Each `*_proj_w` pointer carries either a BF16 weight slab (when the
/// matching field in [`Qwen36MoeFfnStepInt4`] is null) or an INT4 packed-u8
/// slab (when the matching `*_scale`/`*_zero` pair is set).
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepWeights {
    pub input_hidden: *const c_void,
    pub post_attn_norm_w: *const c_void,
    pub gate_w: *const c_void,
    pub gate_up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,
    pub shared_gate_proj_w: *const c_void,
    pub shared_up_proj_w: *const c_void,
    pub shared_down_proj_w: *const c_void,
    pub shared_expert_gate_w: *const c_void,
}

/// Optional INT4 sidecar pointers + group size for the FFN parity launcher
/// (PR 4b5). `group_size == 0` ⇒ INT4 disabled and every sidecar pointer
/// must be null. When non-zero, each tensor is independently switchable:
/// a non-null `*_scale`/`*_zero` pair routes that tensor's matvec through
/// `int4_dequant_scalar`; a null pair keeps it on the BF16 path. Scales
/// and zeros are BF16 with the bake's group layout — `[..., out/gs, in/gs]`.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepInt4 {
    pub group_size: i32,
    pub gate_up_proj_scale: *const c_void,
    pub gate_up_proj_zero: *const c_void,
    pub down_proj_scale: *const c_void,
    pub down_proj_zero: *const c_void,
    pub shared_gate_proj_scale: *const c_void,
    pub shared_gate_proj_zero: *const c_void,
    pub shared_up_proj_scale: *const c_void,
    pub shared_up_proj_zero: *const c_void,
    pub shared_down_proj_scale: *const c_void,
    pub shared_down_proj_zero: *const c_void,
}

impl Qwen36MoeFfnStepInt4 {
    /// All-null sidecars + group_size=0. Use this when the BF16 path is
    /// what you want — the kernel falls through to the existing matvecs
    /// for every tensor.
    pub const fn disabled() -> Self {
        Self {
            group_size: 0,
            gate_up_proj_scale: std::ptr::null(),
            gate_up_proj_zero: std::ptr::null(),
            down_proj_scale: std::ptr::null(),
            down_proj_zero: std::ptr::null(),
            shared_gate_proj_scale: std::ptr::null(),
            shared_gate_proj_zero: std::ptr::null(),
            shared_up_proj_scale: std::ptr::null(),
            shared_up_proj_zero: std::ptr::null(),
            shared_down_proj_scale: std::ptr::null(),
            shared_down_proj_zero: std::ptr::null(),
        }
    }
}

/// Safe wrapper for the PR 4b4 staged MoE FFN parity launcher.
///
/// `output` must be a BF16 buffer with at least `max(top_k, hidden)` elements
/// (the size of the largest staged intermediate). `output_idx` must be an
/// i32 buffer with at least `top_k` elements. `workspace` must be an F32
/// buffer sized for the requested stage's footprint (see the layout comment
/// in `kernels/qwen36_moe.hip`). `sync_buf` must be a 32-byte zero buffer
/// (counter @ +0, barrier counter @ +16, barrier flag @ +20).
pub fn ffn_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
    output: &mut GpuBuffer,
    output_idx: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    if params.top_k > params.num_experts {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: top_k ({}) > num_experts ({})",
            params.top_k, params.num_experts,
        )));
    }
    // Only stage 1 is wired through PR 4b4 step 2; stage 2..=5 will land in
    // follow-up commits to this PR.

    let backend = output.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(16) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(20) as *mut c_uint };

    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen36_moe_hip_ffn_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_experts as c_int,
                    params.moe_intermediate as c_int,
                    params.shared_intermediate as c_int,
                    params.top_k as c_int,
                    params.rms_norm_eps,
                    weights.input_hidden,
                    weights.post_attn_norm_w,
                    weights.gate_w,
                    weights.gate_up_proj_w,
                    weights.down_proj_w,
                    weights.shared_gate_proj_w,
                    weights.shared_up_proj_w,
                    weights.shared_down_proj_w,
                    weights.shared_expert_gate_w,
                    int4.group_size as c_int,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    int4.shared_gate_proj_scale,
                    int4.shared_gate_proj_zero,
                    int4.shared_up_proj_scale,
                    int4.shared_up_proj_zero,
                    int4.shared_down_proj_scale,
                    int4.shared_down_proj_zero,
                    output.as_mut_ptr(),
                    output_idx.as_mut_ptr() as *mut c_int,
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::ffn_step_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::ffn_step_launch: CUDA backend not yet wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::ffn_step_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe ffn_step launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// PR 4b5 step 2 safe wrapper for the INT4 dequant smoke launcher.
///
/// Drives the smoke kernel that exercises both `int4_dequant_8` and
/// `int4_dequant_scalar` over the supplied `[out_rows, in_cols]` slab.
/// `packed_buf` must be a u8 buffer with `out_rows * in_cols / 2` bytes.
/// `scale_buf` and `zero_buf` must be BF16 buffers with `(out_rows / gsz)
/// * (in_cols / gsz)` elements each. `dq_8_out` and `dq_scalar_out` must
/// each be F32 buffers with at least `out_rows * in_cols` elements.
#[allow(clippy::too_many_arguments)]
pub fn int4_dequant_smoke_launch(
    ordinal: usize,
    packed_buf: &GpuBuffer,
    scale_buf: &GpuBuffer,
    zero_buf: &GpuBuffer,
    out_rows: i32,
    in_cols: i32,
    gsz: i32,
    dq_8_out: &mut GpuBuffer,
    dq_scalar_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out_rows <= 0 || in_cols <= 0 || gsz <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: positive dims required, \
             got out_rows={out_rows} in_cols={in_cols} gsz={gsz}"
        )));
    }
    if in_cols % 8 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: in_cols ({in_cols}) must \
             be divisible by 8 (the helpers' fast-path stride)"
        )));
    }
    if in_cols % gsz != 0 || gsz % 2 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: in_cols ({in_cols}) must \
             be divisible by gsz ({gsz}) and gsz must be even"
        )));
    }
    if out_rows % gsz != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: out_rows ({out_rows}) must \
             be divisible by gsz ({gsz})"
        )));
    }

    let backend = packed_buf.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen36_moe_hip_int4_dequant_smoke_launch(
                    ordinal,
                    packed_buf.as_ptr() as *const u8,
                    scale_buf.as_ptr(),
                    zero_buf.as_ptr(),
                    out_rows as c_int,
                    in_cols as c_int,
                    gsz as c_int,
                    dq_8_out.as_mut_ptr() as *mut f32,
                    dq_scalar_out.as_mut_ptr() as *mut f32,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::int4_dequant_smoke_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::int4_dequant_smoke_launch: CUDA backend not yet wired".into(),
            ));
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::int4_dequant_smoke_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe int4_dequant_smoke_launch failed with status {status}"),
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

    /// Convert a stream of F32 little-endian bytes to F32 values. Used
    /// for parity-checking the recurrent state buffer (stage 4+), which
    /// production keeps in F32 across decode steps.
    #[cfg(supersonic_backend_hip)]
    fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        assert!(bytes.len() % 4 == 0, "qwen36_moe parity: F32 bytes must be multiple of 4");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Same shape as `assert_parity` but for F32 buffers — used for the
    /// recurrent state which never casts to BF16. Tolerances are tighter
    /// (no BF16 rounding noise to absorb).
    #[cfg(supersonic_backend_hip)]
    fn assert_parity_f32(
        label: &str,
        got_bytes: &[u8],
        want_bytes: &[u8],
        max_abs_tol: f32,
        cos_sim_floor: f64,
    ) {
        assert_eq!(got_bytes.len(), want_bytes.len(),
                   "{label}: byte length mismatch");
        let got = f32_bytes_to_f32(got_bytes);
        let want = f32_bytes_to_f32(want_bytes);
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
        assert!(max_abs_diff <= max_abs_tol,
                "{label}: max_abs={max_abs_diff} exceeds tolerance {max_abs_tol}");
        assert!(cos_sim >= cos_sim_floor,
                "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}");
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

    // ---- PR 4b3 step 2: linear-attn stage 1 parity vs the oracle ---------
    //
    // Same env-var pattern as the full-attn parity tests, but pointed at a
    // JSON produced by `oracle/qwen36_moe_linear_oracle.py`:
    //
    //   python oracle/qwen36_moe_linear_oracle.py --mode synthetic \
    //       --state fresh --out /tmp/qwen36_lin_fresh.json
    //   SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON=/tmp/qwen36_lin_fresh.json \
    //       cargo test --release -p kernel-ffi qwen36_moe_linear_step
    //
    // Skipped with a clear message when the env var is unset so the FFI
    // test stays runnable on Python-less hosts.

    #[cfg(supersonic_backend_hip)]
    struct LinearOracleGeom {
        hidden: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_kernel_dim: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_linear_oracle_json() -> Option<(serde_json::Value, LinearOracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read linear oracle json {json_path}: {e}"));
        let json: serde_json::Value =
            serde_json::from_str(&raw).expect("linear oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "linear-attn parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = LinearOracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_k_heads: cfg["num_k_heads"].as_i64().unwrap() as i32,
            num_v_heads: cfg["num_v_heads"].as_i64().unwrap() as i32,
            head_k_dim: cfg["head_k_dim"].as_i64().unwrap() as i32,
            head_v_dim: cfg["head_v_dim"].as_i64().unwrap() as i32,
            conv_kernel_dim: cfg["conv_kernel_dim"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_1_qkv_raw_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_linear_oracle.py --mode synthetic \
                 --out /tmp/qwen36_lin.json` and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let qkv_raw_expected_bytes = b64_decode(inters["qkv_raw"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(input_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(in_proj_qkv_w_bytes.len(), qkv_dim * hidden_us * 2);
        assert_eq!(in_proj_z_w_bytes.len(), val_dim * hidden_us * 2);
        assert_eq!(in_proj_a_w_bytes.len(), v_us * hidden_us * 2);
        assert_eq!(in_proj_b_w_bytes.len(), v_us * hidden_us * 2);
        assert_eq!(qkv_raw_expected_bytes.len(), qkv_dim * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, hidden_us], &in_proj_qkv_w_bytes,
        ).expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[val_dim, hidden_us], &in_proj_z_w_bytes,
        ).expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_a_w_bytes,
        ).expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_b_w_bytes,
        ).expect("upload in_proj_b_w");

        // Output sized for the largest staged intermediate (qkv_dim BF16
        // is the biggest until later stages bump this).
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[qkv_dim],
        ).expect("alloc output");
        // Workspace sized for stage 1 (qkv_dim + V*v_dim + 2*V F32). Later
        // stages will need more; keep this tight to fail loudly if a stage
        // overruns.
        let workspace_floats = qkv_dim + val_dim + 2 * v_us;
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[workspace_floats],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 1,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: std::ptr::null(),
            conv1d_bias: std::ptr::null(),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: std::ptr::null_mut(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 1");

        // Stage 1 publishes qkv_raw as the full output buffer.
        let got_bytes = output.to_host_bytes().expect("download output");
        // 2048-wide F32 reduction; same envelope as PR 4b2 step 1's q_proj.
        assert_parity(
            "linear step1 qkv_raw",
            &got_bytes,
            &qkv_raw_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_2_silu_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still walks the stage-1 prerequisite (qkv_raw is the
        // conv1d input), so we need every weight stage 1 needed plus
        // conv1d_w, the conv state, and (optionally) conv1d_bias.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights.get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let silu_out_expected_bytes = b64_decode(inters["silu_out"].as_str().unwrap());
        let conv_state_after_expected_bytes =
            b64_decode(inters["conv_state_after"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;

        assert_eq!(conv1d_w_bytes.len(), qkv_dim * 1 * kernel * 2);
        assert_eq!(conv_state_before_bytes.len(), qkv_dim * kstate * 2);
        assert_eq!(silu_out_expected_bytes.len(), qkv_dim * 2);
        assert_eq!(conv_state_after_expected_bytes.len(), qkv_dim * kstate * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, hidden_us], &in_proj_qkv_w_bytes,
        ).expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[val_dim, hidden_us], &in_proj_z_w_bytes,
        ).expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_a_w_bytes,
        ).expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_b_w_bytes,
        ).expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, 1, kernel], &conv1d_w_bytes,
        ).expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, kstate], &conv_state_before_bytes,
        ).expect("upload conv_state");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[qkv_dim],
        ).expect("alloc output");
        let workspace_floats = qkv_dim + val_dim + 2 * v_us;
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[workspace_floats],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null()),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 2");

        // Stage 2 publishes silu_out as the full output buffer.
        let got_bytes = output.to_host_bytes().expect("download output");
        assert_parity(
            "linear step2 silu_out",
            &got_bytes,
            &silu_out_expected_bytes,
            0.04,
            0.9999,
        );
        // The kernel also updates conv_state in place; verify it matches
        // the oracle's conv_state_after so the next decode step has the
        // right starting state.
        let conv_state_got = conv_state.to_host_bytes().expect("download conv_state");
        assert_parity(
            "linear step2 conv_state_after",
            &conv_state_got,
            &conv_state_after_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_3_qkv_post_norm_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 3 needs everything stage 2 needed.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights.get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let q_rep_expected = b64_decode(inters["q_rep"].as_str().unwrap());
        let k_rep_expected = b64_decode(inters["k_rep"].as_str().unwrap());
        let v_heads_expected = b64_decode(inters["v_heads"].as_str().unwrap());
        let q_scaled_expected = b64_decode(inters["q_scaled"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;

        assert_eq!(q_rep_expected.len(), v_kdim * 2);
        assert_eq!(k_rep_expected.len(), v_kdim * 2);
        assert_eq!(v_heads_expected.len(), v_vdim * 2);
        assert_eq!(q_scaled_expected.len(), v_kdim * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, hidden_us], &in_proj_qkv_w_bytes,
        ).expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[val_dim, hidden_us], &in_proj_z_w_bytes,
        ).expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_a_w_bytes,
        ).expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_b_w_bytes,
        ).expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, 1, kernel], &conv1d_w_bytes,
        ).expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, kstate], &conv_state_before_bytes,
        ).expect("upload conv_state");

        // Output sized for stage 3's largest publish (q_scaled || k_rep || v_heads).
        let stage3_publish_elems = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[stage3_publish_elems],
        ).expect("alloc output");
        // Workspace for stage 3: stage-2 footprint plus Q_NORMED, K_NORMED,
        // Q_REP, K_REP slots.
        let workspace_floats = qkv_dim + val_dim + 2 * v_us + 2 * (k_us * kd_us) + 2 * v_kdim;
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[workspace_floats],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 3,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null()),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 3");

        // Output layout: [q_scaled (V*k_dim) | k_rep (V*k_dim) | v_heads (V*v_dim)] BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let q_end = v_kdim * 2;
        let k_end = q_end + v_kdim * 2;
        let v_end = k_end + v_vdim * 2;
        assert_parity("linear step3 q_scaled", &got_bytes_full[..q_end],
                      &q_scaled_expected, 0.04, 0.9999);
        assert_parity("linear step3 k_rep", &got_bytes_full[q_end..k_end],
                      &k_rep_expected, 0.04, 0.9999);
        assert_parity("linear step3 v_heads", &got_bytes_full[k_end..v_end],
                      &v_heads_expected, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_4_recurrent_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 4 needs everything stage 3 needed plus dt_bias, A_log, and
        // the prior recurrent state.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights.get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let dt_bias_bytes = b64_decode(weights["dt_bias"].as_str().unwrap());
        let a_log_bytes = b64_decode(weights["a_log"].as_str().unwrap());
        // recurrent_state_before is encoded as F32 (production layout).
        let recurrent_state_before_bytes =
            b64_decode(weights["recurrent_state_before"].as_str().unwrap());
        let recurrent_out_expected = b64_decode(inters["recurrent_out"].as_str().unwrap());
        let state_after_expected = b64_decode(inters["state_after"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;
        let state_elems = v_us * kd_us * vd_us;

        assert_eq!(dt_bias_bytes.len(), v_us * 2);
        assert_eq!(a_log_bytes.len(), v_us * 2);
        // recurrent_state encoded F32 (4 bytes/elem).
        assert_eq!(recurrent_state_before_bytes.len(), state_elems * 4);
        assert_eq!(recurrent_out_expected.len(), v_vdim * 2);
        assert_eq!(state_after_expected.len(), state_elems * 4);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, hidden_us], &in_proj_qkv_w_bytes,
        ).expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[val_dim, hidden_us], &in_proj_z_w_bytes,
        ).expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_a_w_bytes,
        ).expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_b_w_bytes,
        ).expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, 1, kernel], &conv1d_w_bytes,
        ).expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let dt_bias = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us], &dt_bias_bytes,
        ).expect("upload dt_bias");
        let a_log = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us], &a_log_bytes,
        ).expect("upload a_log");
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, kstate], &conv_state_before_bytes,
        ).expect("upload conv_state");
        let mut recurrent_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::F32, &[state_elems], &recurrent_state_before_bytes,
        ).expect("upload recurrent_state");

        // Stage 4 publishes recurrent_out [V*v_dim] BF16. The buffer is
        // sized for the largest staged intermediate (still stage 3's
        // q_scaled||k_rep||v_heads = 2*V*k_dim + V*v_dim).
        let stage_publish_max = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[stage_publish_max],
        ).expect("alloc output");
        // Workspace for stage 4 = previous + BETA + G + REC_OUT.
        let workspace_floats =
            qkv_dim + val_dim + 2 * v_us
            + 2 * (k_us * kd_us)
            + 2 * v_kdim
            + v_us + v_us
            + v_vdim;
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[workspace_floats],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 4,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null()),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 4");

        // Stage 4 publishes recurrent_out [V*v_dim] BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let rec_out_bytes = &got_bytes_full[..v_vdim * 2];
        // Recurrent state mixes BF16-rounded inputs (k_rep, q_scaled,
        // v_heads) into F32-precision math; the per-V*v_dim reduction is
        // 128-wide so allow the same envelope as the qkv_proj reduction.
        assert_parity(
            "linear step4 recurrent_out",
            rec_out_bytes,
            &recurrent_out_expected,
            0.04,
            0.9999,
        );

        // Also verify state_after — the F32 recurrent state has been
        // mutated in place by the kernel and must match the oracle's
        // post-update state for the next decode step to work.
        let state_after_got = recurrent_state.to_host_bytes().expect("download state");
        // F32 throughout (no BF16 rounding); be tighter on max_abs.
        // Per-element rounding error from F32 arithmetic + cast-from-BF16
        // operands is at most a few ULPs of the magnitude.
        assert_parity_f32(
            "linear step4 state_after",
            &state_after_got,
            &state_after_expected,
            5e-3,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 5 needs every weight from earlier stages plus norm_w and out_proj_w.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights.get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let dt_bias_bytes = b64_decode(weights["dt_bias"].as_str().unwrap());
        let a_log_bytes = b64_decode(weights["a_log"].as_str().unwrap());
        let recurrent_state_before_bytes =
            b64_decode(weights["recurrent_state_before"].as_str().unwrap());
        let norm_w_bytes = b64_decode(weights["norm_w"].as_str().unwrap());
        let out_proj_w_bytes = b64_decode(weights["out_proj_w"].as_str().unwrap());
        let output_hidden_expected = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;
        let state_elems = v_us * kd_us * vd_us;

        assert_eq!(norm_w_bytes.len(), vd_us * 2);
        assert_eq!(out_proj_w_bytes.len(), hidden_us * v_vdim * 2);
        assert_eq!(output_hidden_expected.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_norm_w_bytes,
        ).expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, hidden_us], &in_proj_qkv_w_bytes,
        ).expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[val_dim, hidden_us], &in_proj_z_w_bytes,
        ).expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_a_w_bytes,
        ).expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us, hidden_us], &in_proj_b_w_bytes,
        ).expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, 1, kernel], &conv1d_w_bytes,
        ).expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let dt_bias = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us], &dt_bias_bytes,
        ).expect("upload dt_bias");
        let a_log = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[v_us], &a_log_bytes,
        ).expect("upload a_log");
        let norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[vd_us], &norm_w_bytes,
        ).expect("upload norm_w");
        let out_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, v_vdim], &out_proj_w_bytes,
        ).expect("upload out_proj_w");
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[qkv_dim, kstate], &conv_state_before_bytes,
        ).expect("upload conv_state");
        let mut recurrent_state = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::F32, &[state_elems], &recurrent_state_before_bytes,
        ).expect("upload recurrent_state");

        let stage_publish_max = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[stage_publish_max],
        ).expect("alloc output");
        let workspace_floats =
            qkv_dim + val_dim + 2 * v_us
            + 2 * (k_us * kd_us)
            + 2 * v_kdim
            + v_us + v_us
            + v_vdim;
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[workspace_floats],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null()),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: norm_w.as_ptr(),
            out_proj_w: out_proj_w.as_ptr(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 5");

        // Stage 5 publishes output_hidden into output[0..hidden) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        // out_proj reduces over V*v_dim=4096 lanes; same envelope as
        // PR 4b2 stage 5.
        assert_parity(
            "linear step5 output_hidden",
            got_bytes,
            &output_hidden_expected,
            0.05,
            0.9999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b4 — staged MoE FFN parity tests against the Python oracle
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    struct FfnOracleGeom {
        hidden: i32,
        num_experts: i32,
        moe_intermediate: i32,
        shared_intermediate: i32,
        top_k: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_ffn_oracle_json() -> Option<(serde_json::Value, FfnOracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_FFN_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read ffn oracle json {json_path}: {e}"));
        let json: serde_json::Value =
            serde_json::from_str(&raw).expect("ffn oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "MoE FFN parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = FfnOracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_experts: cfg["num_experts"].as_i64().unwrap() as i32,
            moe_intermediate: cfg["moe_intermediate"].as_i64().unwrap() as i32,
            shared_intermediate: cfg["shared_intermediate"].as_i64().unwrap() as i32,
            top_k: cfg["top_k"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    /// Decode a base64 i32 buffer (oracle uses int32 for `topk_idx`).
    #[cfg(supersonic_backend_hip)]
    fn i32_bytes_to_vec(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| {
                let mut a = [0u8; 4];
                a.copy_from_slice(c);
                i32::from_le_bytes(a)
            })
            .collect()
    }

    /// Workspace floats sufficient for the largest staged FFN intermediate
    /// (stage 5). Mirrors the kernel's per-stage offset layout in
    /// `kernels/qwen36_moe.hip` (search `OFF_H_NORM`):
    ///   H_NORM        [hidden]
    ///   ROUTER_LOGITS [E]
    ///   ROUTER_PROBS  [E]
    ///   TOPK_VAL      [k]
    ///   TOPK_IDX      [k]
    ///   SG_SCALAR     [1]
    ///   SGP           [Is]
    ///   SUP           [Is]
    ///   SHARED_MID    [Is]
    ///   SHARED_OUT    [hidden]
    ///   EXPERT_GU     [2*I]
    ///   EXPERT_MID    [I]
    ///   EXPERT_STACK  [k*hidden]
    ///   MOE_OUT       [hidden]
    ///
    /// PR 4b4 step 4 (the original stage-3+ test wiring) silently undersized
    /// this by 385 floats on the synthetic config — it omitted SG_SCALAR,
    /// SGP, SUP, EXPERT_GU and EXPERT_MID. The HIP allocator's slack hid the
    /// OOB on the BF16 path. The new INT4-sidecar kernel parameters PR 4b5
    /// step 3 adds shift register/stack layout enough that the same OOB
    /// starts overwriting live cooperative-launch state and stage>=4 hangs.
    /// The right fix is to size the buffer for what the kernel actually uses.
    #[cfg(supersonic_backend_hip)]
    fn ffn_parity_workspace_floats(geom: &FfnOracleGeom) -> usize {
        let hidden = geom.hidden as usize;
        let e = geom.num_experts as usize;
        let k = geom.top_k as usize;
        let is_dim = geom.shared_intermediate as usize;
        let i_dim = geom.moe_intermediate as usize;
        // 3*hidden = H_NORM + SHARED_OUT + MOE_OUT
        // 2*e      = ROUTER_LOGITS + ROUTER_PROBS
        // 2*k      = TOPK_VAL + TOPK_IDX
        // 1        = SG_SCALAR
        // 3*is_dim = SGP + SUP + SHARED_MID
        // 3*i_dim  = EXPERT_GU(2*I) + EXPERT_MID(I)
        // k*hidden = EXPERT_STACK
        3 * hidden + 2 * e + 2 * k + 1 + 3 * is_dim + 3 * i_dim + k * hidden
    }

    /// Output BF16 elements sufficient for the largest staged FFN
    /// intermediate. Stages 2..=5 publish a `[hidden]` buffer; stage 1
    /// publishes `[k]`. Sized for `hidden`.
    #[cfg(supersonic_backend_hip)]
    fn ffn_parity_output_elems(geom: &FfnOracleGeom) -> usize {
        geom.hidden as usize
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_1_topk_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_ffn_oracle.py --mode synthetic \
                 --out /tmp/qwen36_ffn.json` and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let topk_idx_expected = i32_bytes_to_vec(
            &b64_decode(inters["topk_idx"].as_str().unwrap())
        );
        let topk_weights_expected_bytes =
            b64_decode(inters["topk_weights"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(post_attn_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(topk_idx_expected.len(), k_us);
        assert_eq!(topk_weights_expected_bytes.len(), k_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");

        // Output sized for the largest staged intermediate (hidden BF16).
        // Stage 1 publishes only `topk_weights[k]` into `output[0..k]`,
        // and `topk_idx[k]` into the separate `output_idx` buffer.
        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        // No I32 variant in `ScalarType`; U32 has the same 4-byte storage
        // and the kernel reinterprets via the FFI signature's `*mut c_int`.
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 1,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: std::ptr::null(),
            shared_up_proj_w: std::ptr::null(),
            shared_down_proj_w: std::ptr::null(),
            shared_expert_gate_w: std::ptr::null(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 1");

        // Verify topk_idx (int32) — must match oracle exactly. Routing
        // decisions are categorical; any disagreement is a real bug, no
        // tolerance.
        let got_idx_bytes = output_idx.to_host_bytes().expect("download output_idx");
        let got_idx = i32_bytes_to_vec(&got_idx_bytes);
        assert_eq!(
            got_idx, topk_idx_expected,
            "ffn step1 topk_idx mismatch: got {got_idx:?}, want {topk_idx_expected:?}"
        );

        // Verify topk_weights (BF16) — these come from softmax + renorm,
        // so we expect bit-exactness for most elements with rare 1-ULP
        // drift from F32 accumulation order.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..k_us * 2];
        assert_parity(
            "ffn step1 topk_weights",
            got_bytes,
            &topk_weights_expected_bytes,
            0.01,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_2_shared_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still runs the stage-1 prerequisites (rmsnorm + router
        // gate), so the gate weight is still required. The new tensors
        // are the four shared-expert weights; the per-expert
        // gate_up_proj / down_proj stay null until stage 3.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes =
            b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes =
            b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes =
            b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let shared_out_expected_bytes =
            b64_decode(inters["shared_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(post_attn_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(shared_gate_proj_w_bytes.len(), is_us * hidden_us * 2);
        assert_eq!(shared_up_proj_w_bytes.len(), is_us * hidden_us * 2);
        assert_eq!(shared_down_proj_w_bytes.len(), hidden_us * is_us * 2);
        assert_eq!(shared_expert_gate_w_bytes.len(), 1 * hidden_us * 2);
        assert_eq!(shared_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_gate_proj_w_bytes,
        ).expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_up_proj_w_bytes,
        ).expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, is_us], &shared_down_proj_w_bytes,
        ).expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[1, hidden_us], &shared_expert_gate_w_bytes,
        ).expect("upload shared_expert_gate_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 2");

        // Stage 2 publishes shared_out into output[0..hidden] BF16. Tolerance
        // is looser than stage 1 because the shared expert path stacks four
        // matmuls (gate, up, down, plus the 1-row sigmoid gate); each one
        // accumulates F32 reduction-order drift. Cos_sim ≥ 0.999 on a deep
        // chain like this is the same envelope linear-attn stage 5 uses.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step2 shared_out",
            got_bytes,
            &shared_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_3_expert0_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 3 still walks stage-2 prereqs (the kernel runs the shared
        // expert path even when only the per-expert output is wanted) plus
        // the new fused expert weight slabs.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes =
            b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes =
            b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes =
            b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes =
            b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        // The oracle's `expert_stack` is [K, hidden] BF16 — the first
        // `hidden` elements are the FFN output of `topk_idx[0]`, which is
        // the same expert stage 3 dispatches. Slice off that first chunk.
        let expert_stack_full = b64_decode(inters["expert_stack"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(gate_up_proj_w_bytes.len(), e_us * 2 * i_us * hidden_us * 2);
        assert_eq!(down_proj_w_bytes.len(), e_us * hidden_us * i_us * 2);
        assert_eq!(expert_stack_full.len(), k_us * hidden_us * 2);
        let expert0_expected_bytes = &expert_stack_full[..hidden_us * 2];

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, 2 * i_us, hidden_us], &gate_up_proj_w_bytes,
        ).expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us, i_us], &down_proj_w_bytes,
        ).expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_gate_proj_w_bytes,
        ).expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_up_proj_w_bytes,
        ).expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, is_us], &shared_down_proj_w_bytes,
        ).expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[1, hidden_us], &shared_expert_gate_w_bytes,
        ).expect("upload shared_expert_gate_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 3,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 3");

        // Stage 3 publishes expert_stack[0] (BF16-rounded F32) into
        // output[0..hidden]. Same envelope as stage 2's shared expert —
        // four matmuls deep, F32 throughout.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step3 expert0_out",
            got_bytes,
            expert0_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_4_moe_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 4 walks all of stage-3's path (k iterations of the per-expert
        // FFN) and adds the final topk-weighted sum. Same weight set; the
        // expected output is `moe_out` instead of `expert_stack[0]`.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes =
            b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes =
            b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes =
            b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes =
            b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let moe_out_expected_bytes = b64_decode(inters["moe_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(moe_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, 2 * i_us, hidden_us], &gate_up_proj_w_bytes,
        ).expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us, i_us], &down_proj_w_bytes,
        ).expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_gate_proj_w_bytes,
        ).expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_up_proj_w_bytes,
        ).expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, is_us], &shared_down_proj_w_bytes,
        ).expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[1, hidden_us], &shared_expert_gate_w_bytes,
        ).expect("upload shared_expert_gate_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 4,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 4");

        // Stage 4 publishes moe_out (= sum_j topk_w[j] * expert_stack[j],
        // BF16-rounded once) into output[0..hidden]. Tolerance same as
        // stage 3; the only added work is a k=8-wide reduction with
        // BF16-cast renormed weights, which is well-conditioned.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step4 moe_out",
            got_bytes,
            &moe_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 5 is the trivial closer — same weights as stage 4 plus the
        // residual add against `input_hidden`.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes =
            b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes =
            b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes =
            b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes =
            b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let output_hidden_expected_bytes =
            b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(output_hidden_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, 2 * i_us, hidden_us], &gate_up_proj_w_bytes,
        ).expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us, i_us], &down_proj_w_bytes,
        ).expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_gate_proj_w_bytes,
        ).expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[is_us, hidden_us], &shared_up_proj_w_bytes,
        ).expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us, is_us], &shared_down_proj_w_bytes,
        ).expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[1, hidden_us], &shared_expert_gate_w_bytes,
        ).expect("upload shared_expert_gate_w");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 5");

        // Stage 5 publishes output_hidden = input_hidden + moe_out +
        // shared_out (all F32, BF16-round once at the end). The residual
        // addition can't make accuracy worse than stage 4's moe_out, so
        // the same envelope applies.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step5 output_hidden",
            got_bytes,
            &output_hidden_expected_bytes,
            0.05,
            0.999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b5 step 3: shared-expert INT4 parity vs the INT4 oracle.
    //
    // Driven by the same env var as the BF16 FFN tests; skipped silently
    // when the JSON's schema is not `qwen36-moe-oracle-ffn-int4-v1`. The
    // INT4 oracle's `weights` block carries the BF16-reconstruction of
    // each quantized tensor, so the BF16 reference computed against those
    // weights is the exact intermediate the kernel must reproduce when
    // it dequantizes (packed, scale, zero) on the fly.
    //
    // Step 3 wires only the shared-expert tensors (gate_proj, up_proj,
    // down_proj). Fused-expert weights stay BF16 in this test —
    // `weights.gate_up_proj_w` and `weights.down_proj_w` are uploaded as
    // BF16 and `Qwen36MoeFfnStepInt4`'s fused-expert sidecars stay null.
    // Steps 4/5 will switch those over.
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    fn ffn_oracle_is_int4(json: &serde_json::Value) -> bool {
        json["schema"].as_str() == Some("qwen36-moe-oracle-ffn-int4-v1")
    }

    /// Pulls (packed, scale, zero) bytes for one INT4-quantized FFN tensor
    /// from the oracle JSON. Returns owned `Vec<u8>` buffers in their
    /// native byte representations (u8 for packed, BF16 LE for scale/zero).
    #[cfg(supersonic_backend_hip)]
    fn decode_int4_sidecar(
        json: &serde_json::Value, name: &str,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let blk = &json["int4_weights"][name];
        let packed = b64_decode(blk["packed"].as_str()
            .unwrap_or_else(|| panic!("missing int4_weights[{name}].packed")));
        let scale = b64_decode(blk["scale"].as_str()
            .unwrap_or_else(|| panic!("missing int4_weights[{name}].scale")));
        let zero = b64_decode(blk["zero"].as_str()
            .unwrap_or_else(|| panic!("missing int4_weights[{name}].zero")));
        (packed, scale, zero)
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_2_shared_out_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 Generate an INT4 fixture with \
                 `python oracle/qwen36_moe_ffn_oracle.py --mode synthetic \
                 --int4 --out /tmp/qwen36_ffn_int4.json` and re-run."
            );
            return;
        };
        if !ffn_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Generate one with \
                 the `--int4` flag if you want to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Shared-expert weights become INT4 packed bytes; everything else
        // stays as BF16 from `weights`. The BF16 reconstruction is *not*
        // used at the kernel call site for INT4 tensors (the kernel reads
        // packed bytes), but the oracle ran its reference computation
        // against the same reconstruction so the intermediate target lines
        // up byte-for-byte with the kernel's output.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        let (sgp_packed, sgp_scale, sgp_zero) =
            decode_int4_sidecar(&json, "shared_gate_proj_w");
        let (sup_packed, sup_scale, sup_zero) =
            decode_int4_sidecar(&json, "shared_up_proj_w");
        let (sdp_packed, sdp_scale, sdp_zero) =
            decode_int4_sidecar(&json, "shared_down_proj_w");

        let shared_out_expected_bytes =
            b64_decode(inters["shared_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;
        let gsz_us = group_size as usize;

        // Sanity: shapes match the bake convention.
        assert_eq!(sgp_packed.len(), is_us * (hidden_us / 2),
            "shared_gate_proj packed bytes mismatch");
        assert_eq!(sgp_scale.len(), (is_us / gsz_us) * (hidden_us / gsz_us) * 2,
            "shared_gate_proj scale bytes mismatch");
        assert_eq!(sup_packed.len(), is_us * (hidden_us / 2));
        assert_eq!(sdp_packed.len(), hidden_us * (is_us / 2));
        assert_eq!(shared_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &input_hidden_bytes,
        ).expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[hidden_us], &post_attn_norm_w_bytes,
        ).expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[e_us, hidden_us], &gate_w_bytes,
        ).expect("upload gate_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[1, hidden_us], &shared_expert_gate_w_bytes,
        ).expect("upload shared_expert_gate_w");

        // Packed INT4 buffers — uploaded as u8.
        let sgp_packed_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::U8, &[sgp_packed.len()], &sgp_packed,
        ).expect("upload sgp packed");
        let sup_packed_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::U8, &[sup_packed.len()], &sup_packed,
        ).expect("upload sup packed");
        let sdp_packed_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::U8, &[sdp_packed.len()], &sdp_packed,
        ).expect("upload sdp packed");
        // BF16 scale/zero sidecars.
        let sgp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sgp_scale.len() / 2], &sgp_scale,
        ).expect("upload sgp scale");
        let sgp_zero_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sgp_zero.len() / 2], &sgp_zero,
        ).expect("upload sgp zero");
        let sup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sup_scale.len() / 2], &sup_scale,
        ).expect("upload sup scale");
        let sup_zero_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sup_zero.len() / 2], &sup_zero,
        ).expect("upload sup zero");
        let sdp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sdp_scale.len() / 2], &sdp_scale,
        ).expect("upload sdp scale");
        let sdp_zero_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[sdp_zero.len() / 2], &sdp_zero,
        ).expect("upload sdp zero");

        let mut output = GpuBuffer::zeros(
            ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)],
        ).expect("alloc output");
        let mut output_idx = GpuBuffer::zeros(
            ordinal, ScalarType::U32, &[k_us],
        ).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal, ScalarType::F32, &[ffn_parity_workspace_floats(&geom)],
        ).expect("alloc workspace");
        let mut sync_buf = GpuBuffer::zeros(
            ordinal, ScalarType::U8, &[32],
        ).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        // Shared-expert weight pointers point at the *packed* u8 buffers.
        // Fused-expert pointers stay null (stage<3 doesn't read them).
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: sgp_packed_buf.as_ptr(),
            shared_up_proj_w: sup_packed_buf.as_ptr(),
            shared_down_proj_w: sdp_packed_buf.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size,
            gate_up_proj_scale: std::ptr::null(),
            gate_up_proj_zero: std::ptr::null(),
            down_proj_scale: std::ptr::null(),
            down_proj_zero: std::ptr::null(),
            shared_gate_proj_scale: sgp_scale_buf.as_ptr(),
            shared_gate_proj_zero: sgp_zero_buf.as_ptr(),
            shared_up_proj_scale: sup_scale_buf.as_ptr(),
            shared_up_proj_zero: sup_zero_buf.as_ptr(),
            shared_down_proj_scale: sdp_scale_buf.as_ptr(),
            shared_down_proj_zero: sdp_zero_buf.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 2 (int4)");

        // Same envelope as the BF16 stage 2 test: cos_sim ≥ 0.999, max
        // |delta| ≤ 0.05. The reconstruction is bit-identical to what the
        // oracle ran reference against, so any residual disagreement is
        // F32 reduction-order drift through the matvec — same as BF16.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step2 int4 shared_out",
            got_bytes,
            &shared_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b5 step 2: INT4 dequant smoke test.
    //
    // Builds a small `[out_rows, in_cols]` weight slab in F32 on host,
    // runs min/max group-quant to (packed u8, BF16 scale, BF16 zero) with
    // exactly the same math as `oracle/qwen36_moe_ffn_oracle.py`, uploads
    // the sidecars, calls the smoke launcher, and verifies the GPU
    // outputs of `int4_dequant_8` and `int4_dequant_scalar` both equal a
    // host-computed reference byte-for-byte (these are F32 values rounded
    // through BF16, so exact equality is the right bar).
    //
    // Two configs run in succession to cover both helper paths:
    //   gsz=8, in_cols=16  → every 8-col span lies in one group     (fast)
    //   gsz=4, in_cols=16  → every 8-col span crosses a boundary    (slow)
    // -------------------------------------------------------------------

    /// BF16 round-to-nearest-even of an F32 value, returning the 16-bit
    /// big-end-of-F32 representation. Same math as the kernel's
    /// `bf16_round_rne_f32`.
    #[cfg(supersonic_backend_hip)]
    fn bf16_round_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
        let r = bits.wrapping_add(rounding_bias);
        (r >> 16) as u16
    }

    /// Reverse: F32 from a BF16 bit pattern.
    #[cfg(supersonic_backend_hip)]
    fn f32_from_bf16(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// Min/max INT4 group-quant on a 2D `[out, in]` F32 slab — Rust mirror
    /// of `minmax_int4_packed_and_recon` in the FFN oracle. Returns
    /// `(packed [out, in/2] u8, scale [out/gs, in/gs] u16-as-BF16,
    /// zero [out/gs, in/gs] u16-as-BF16)`.
    #[cfg(supersonic_backend_hip)]
    fn host_minmax_int4(
        w: &[f32], out_rows: usize, in_cols: usize, gsz: usize,
    ) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
        assert_eq!(w.len(), out_rows * in_cols);
        assert_eq!(out_rows % gsz, 0);
        assert_eq!(in_cols % gsz, 0);
        assert_eq!(in_cols % 2, 0);
        let sr = out_rows / gsz;
        let sc = in_cols / gsz;

        let mut packed = vec![0u8; out_rows * (in_cols / 2)];
        let mut scale = vec![0u16; sr * sc];
        let mut zero = vec![0u16; sr * sc];

        for gr in 0..sr {
            for gc in 0..sc {
                let mut tmin = f32::INFINITY;
                let mut tmax = f32::NEG_INFINITY;
                for r in 0..gsz {
                    for c in 0..gsz {
                        let v = w[(gr * gsz + r) * in_cols + gc * gsz + c];
                        tmin = tmin.min(v);
                        tmax = tmax.max(v);
                    }
                }
                let rng = tmax - tmin;
                let s_f = if rng > 0.0 { rng / 15.0 } else { 1.0 };
                let z_f = if rng > 0.0 { -tmin / s_f } else { 0.0 };
                let s_bits = bf16_round_bits(s_f);
                let z_bits = bf16_round_bits(z_f);
                scale[gr * sc + gc] = s_bits;
                zero[gr * sc + gc] = z_bits;
                let s = f32_from_bf16(s_bits);
                let z = f32_from_bf16(z_bits);
                for r in 0..gsz {
                    for c in 0..gsz {
                        let row = gr * gsz + r;
                        let col = gc * gsz + c;
                        let v = w[row * in_cols + col];
                        let q = (v / s + z).round().clamp(0.0, 15.0) as u8;
                        // Pack: even col → low nibble, odd col → high nibble.
                        let byte_idx = row * (in_cols / 2) + col / 2;
                        if col & 1 == 0 {
                            packed[byte_idx] = (packed[byte_idx] & 0xF0) | (q & 0x0F);
                        } else {
                            packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                        }
                    }
                }
            }
        }
        (packed, scale, zero)
    }

    /// Reference reconstruction: `bf16(q*s - z*s)` per element. Returns
    /// F32 values whose lower 16 bits are zero (i.e. exactly BF16-precision).
    #[cfg(supersonic_backend_hip)]
    fn host_dequant_recon(
        packed: &[u8], scale: &[u16], zero: &[u16],
        out_rows: usize, in_cols: usize, gsz: usize,
    ) -> Vec<f32> {
        let sc = in_cols / gsz;
        let mut out = vec![0.0f32; out_rows * in_cols];
        for row in 0..out_rows {
            for col in 0..in_cols {
                let gi = (row / gsz) * sc + col / gsz;
                let s = f32_from_bf16(scale[gi]);
                let z = f32_from_bf16(zero[gi]);
                let byte = packed[row * (in_cols / 2) + col / 2];
                let n = if col & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let v = (n as f32) * s - z * s;
                out[row * in_cols + col] = f32::from_bits(
                    (bf16_round_bits(v) as u32) << 16,
                );
            }
        }
        out
    }

    /// Encode a slice of BF16 16-bit values to LE bytes for `from_host_bytes`.
    #[cfg(supersonic_backend_hip)]
    fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(bits.len() * 2);
        for b in bits {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        bytes
    }

    /// Drives one (out_rows, in_cols, gsz) configuration through the smoke
    /// kernel and asserts both helper outputs match the host reference
    /// exactly.
    #[cfg(supersonic_backend_hip)]
    fn run_int4_dequant_smoke(out_rows: usize, in_cols: usize, gsz: usize, label: &str) {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
        set_backend(Backend::Hip);
        let ordinal = 0usize;

        // Deterministic synthetic weights: a 32-bit LCG seeded by config so
        // each smoke variant uses different but reproducible values.
        let n = out_rows * in_cols;
        let mut rng_state: u32 = 0xC0FFEE
            ^ ((out_rows as u32) << 16)
            ^ ((in_cols as u32) << 8)
            ^ (gsz as u32);
        let mut w = vec![0.0f32; n];
        for v in w.iter_mut() {
            // LCG (Numerical Recipes constants) → uniform [-1, 1).
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (rng_state >> 8) as f32 / ((1u32 << 24) as f32); // [0,1)
            *v = u * 2.0 - 1.0;
        }

        let (packed, scale_bits, zero_bits) = host_minmax_int4(&w, out_rows, in_cols, gsz);
        let recon_ref = host_dequant_recon(&packed, &scale_bits, &zero_bits,
                                            out_rows, in_cols, gsz);

        let packed_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::U8, &[packed.len()], &packed,
        ).expect("upload packed");
        let scale_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[scale_bits.len()],
            &bf16_bits_to_bytes(&scale_bits),
        ).expect("upload scale");
        let zero_buf = GpuBuffer::from_host_bytes(
            ordinal, ScalarType::BF16, &[zero_bits.len()],
            &bf16_bits_to_bytes(&zero_bits),
        ).expect("upload zero");

        let mut dq_8 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[n])
            .expect("alloc dq_8");
        let mut dq_scalar = GpuBuffer::zeros(ordinal, ScalarType::F32, &[n])
            .expect("alloc dq_scalar");

        int4_dequant_smoke_launch(
            ordinal,
            &packed_buf, &scale_buf, &zero_buf,
            out_rows as i32, in_cols as i32, gsz as i32,
            &mut dq_8, &mut dq_scalar,
        ).expect("smoke launch");

        let dq_8_bytes = dq_8.to_host_bytes().expect("download dq_8");
        let dq_scalar_bytes = dq_scalar.to_host_bytes().expect("download dq_scalar");
        let dq_8_v: Vec<f32> = dq_8_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let dq_scalar_v: Vec<f32> = dq_scalar_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        for i in 0..n {
            assert_eq!(
                dq_8_v[i].to_bits(), recon_ref[i].to_bits(),
                "[{label}] int4_dequant_8 mismatch at i={i}: got {} ({:#010x}), want {} ({:#010x})",
                dq_8_v[i], dq_8_v[i].to_bits(),
                recon_ref[i], recon_ref[i].to_bits(),
            );
            assert_eq!(
                dq_scalar_v[i].to_bits(), recon_ref[i].to_bits(),
                "[{label}] int4_dequant_scalar mismatch at i={i}: got {} ({:#010x}), want {} ({:#010x})",
                dq_scalar_v[i], dq_scalar_v[i].to_bits(),
                recon_ref[i], recon_ref[i].to_bits(),
            );
        }
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_int4_dequant_smoke_fast_path() {
        // gsz=8, in_cols=16 → every 8-col span lies in one group, so
        // `int4_dequant_8` exercises its `g0 == g7` fast path on every span.
        run_int4_dequant_smoke(8, 16, 8, "fast (gsz=8)");
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_int4_dequant_smoke_slow_path() {
        // gsz=4, in_cols=16 → 8-col spans starting at col=0 cross from
        // group 0 to group 1 (boundary at col=4); spans starting at col=8
        // cross 2→3. Exercises the per-element `g0 != g7` slow path.
        run_int4_dequant_smoke(8, 16, 4, "slow (gsz=4)");
    }
}
