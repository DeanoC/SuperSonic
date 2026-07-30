/// Geometry constants for the persistent decode megakernel — packed into
/// one struct to keep [`persistent_decode_launch`]'s arg list tractable.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoePersistentGeom {
    pub hidden: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

/// Phase 3e safe wrapper for the persistent decode megakernel. Replaces
/// the chained 80 step-kernel launches/token with one cooperative HIP
/// launch.
///
/// Caller responsibilities:
/// - `layers_device` is a device-resident array of
///   [`Qwen36MoeDecodeLayerDesc`] (`num_layers` entries). Even
///   `num_layers` only — the bridge enforces this.
/// - `int4_scales_device` is null for BF16 bakes, or a device-resident
///   array of [`Qwen36MoeInt4ScaleDesc`] (parallel to `layers_device`).
/// - `hidden_ping` is uploaded with the BF16 initial hidden bytes; the
///   final hidden lands back in `hidden_ping` after `num_layers`.
/// - `workspace` (F32) sized for `max(attn_workspace_floats(geom),
///   ffn_workspace_floats(geom))`.
/// - `ffn_topk_idx_scratch` is a small `[top_k]` i32 buffer (used only
///   internally by the FFN phase at stage 1; we run stage 5 so it's
///   inert, but must be valid).
/// - `sync_buf` is at least 96 zeroed bytes (counters[0..16] + barrier
///   counter/flag); the bridge defensively re-zeros it on entry.
/// Phase 3f folded final RMSnorm + lm_head GEMV. Pass `Some(...)` on
/// generation steps to write logits directly from the megakernel and
/// skip the separate `lm_head_launch` call; pass `None` on prefill
/// steps where the caller doesn't need logits. Bundled rather than
/// scattered as ~4 args so the call site stays readable.
pub struct Qwen36MoePersistentLmHeadFold<'a> {
    /// `[hidden]` BF16. Same final_norm tensor the standalone
    /// `lm_head_launch` consumes.
    pub final_norm_w: &'a GpuBuffer,
    /// `[vocab, hidden]` BF16. The bake may store INT4 on disk, but the
    /// folded kernel reads the pre-dequantized BF16 upload.
    pub lm_head_w: &'a GpuBuffer,
    /// Optional `[vocab]` BF16 output buffer. Kernel writes one logit per row
    /// when full logits are needed for sampling or diagnostics.
    pub logits_out: Option<&'a mut GpuBuffer>,
    /// Optional `[1]` U32 output buffer for greedy decode. When present, the
    /// persistent kernel reduces the lm_head argmax internally.
    pub top1_out: Option<&'a mut GpuBuffer>,
    /// Vocab size. Must be `> 0` and match `lm_head_w.shape()[0]`.
    pub vocab: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoePersistentMode {
    Full,
    RouterOnly,
    FfnOnly,
    AttnOnly,
    FfnStage(i32),
    LinearStage(i32),
}

impl Qwen36MoePersistentMode {
    #[allow(dead_code)]
    fn as_ffi(self) -> c_int {
        match self {
            Self::Full => 0,
            Self::RouterOnly => 1,
            Self::FfnOnly => 2,
            Self::AttnOnly => 3,
            Self::FfnStage(stage) => 3 + stage,
            Self::LinearStage(stage) => 8 + stage,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_launch(
    ordinal: usize,
    dtype: ScalarType,
    geom: Qwen36MoePersistentGeom,
    position: i32,
    cache_pos: i32,
    layers_device: &GpuBuffer,
    int4_scales_device: Option<&GpuBuffer>,
    kv_fp8_descs_device: Option<&GpuBuffer>,
    num_layers: usize,
    hidden_ping: &mut GpuBuffer,
    hidden_pong: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    ffn_topk_idx_scratch: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    lm_head_fold: Option<Qwen36MoePersistentLmHeadFold<'_>>,
) -> Result<(), GpuError> {
    persistent_decode_launch_range(
        ordinal,
        dtype,
        geom,
        0,
        num_layers,
        Qwen36MoePersistentMode::Full,
        position,
        cache_pos,
        layers_device,
        int4_scales_device,
        kv_fp8_descs_device,
        num_layers,
        hidden_ping,
        hidden_pong,
        workspace,
        ffn_topk_idx_scratch,
        sync_buf,
        None,
        -1,
        None,
        1,
        lm_head_fold,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_launch_range(
    ordinal: usize,
    dtype: ScalarType,
    geom: Qwen36MoePersistentGeom,
    start_layer: usize,
    end_layer_exclusive: usize,
    mode: Qwen36MoePersistentMode,
    position: i32,
    // `-1` (`Qwen36MoeAttnStepParams::CACHE_POS_INHERIT`) ⇒ inherit
    // from `position` (dense base decode). `≥ 0` ⇒ decoupled KV slot
    // for SpecPrefill sparse-prefill or MTP draft layers.
    cache_pos: i32,
    layers_device: &GpuBuffer,
    int4_scales_device: Option<&GpuBuffer>,
    kv_fp8_descs_device: Option<&GpuBuffer>,
    num_layers: usize,
    hidden_ping: &mut GpuBuffer,
    hidden_pong: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    ffn_topk_idx_scratch: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    embed_w: Option<&GpuBuffer>,
    token_id: i32,
    token_ids: Option<&GpuBuffer>,
    prefill_len: i32,
    lm_head_fold: Option<Qwen36MoePersistentLmHeadFold<'_>>,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::persistent_decode_launch: only BF16 wired, got {dtype:?}"
        )));
    }
    if let Qwen36MoePersistentMode::FfnStage(stage) = mode {
        if !(1..=5).contains(&stage) {
            return Err(GpuError::InvalidArg(format!(
                "qwen36_moe::persistent_decode_launch: FFN stage mode must be in 1..=5, got {stage}"
            )));
        }
    }
    if let Qwen36MoePersistentMode::LinearStage(stage) = mode {
        if !(1..=5).contains(&stage) {
            return Err(GpuError::InvalidArg(format!(
                "qwen36_moe::persistent_decode_launch: linear stage mode must be in 1..=5, got {stage}"
            )));
        }
    }
    let backend = layers_device.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };
    let int4_ptr: *const Qwen36MoeInt4ScaleDesc = int4_scales_device
        .map(|b| b.as_ptr() as *const Qwen36MoeInt4ScaleDesc)
        .unwrap_or(std::ptr::null());
    let kv_fp8_ptr: *const Qwen36MoeKVCacheFp8Desc = kv_fp8_descs_device
        .map(|b| b.as_ptr() as *const Qwen36MoeKVCacheFp8Desc)
        .unwrap_or(std::ptr::null());
    let embed_w_ptr = embed_w.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let token_ids_ptr = token_ids
        .map(|b| b.as_ptr() as *const c_uint)
        .unwrap_or(std::ptr::null());

    // Fold pointers default to null; the kernel skips the lm_head phase
    // when any of the three is null.
    let (vocab, final_norm_w_ptr, lm_head_w_ptr, logits_out_ptr, top1_out_ptr) = match lm_head_fold
    {
        Some(mut f) => {
            let logits_out_ptr = f
                .logits_out
                .as_deref_mut()
                .map(|buf| buf.as_mut_ptr())
                .unwrap_or(std::ptr::null_mut());
            let top1_out_ptr = f
                .top1_out
                .as_deref_mut()
                .map(|buf| buf.as_mut_ptr() as *mut c_uint)
                .unwrap_or(std::ptr::null_mut());
            (
                f.vocab,
                f.final_norm_w.as_ptr(),
                f.lm_head_w.as_ptr(),
                logits_out_ptr,
                top1_out_ptr,
            )
        }
        None => (
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
    };

    let op = match mode {
        Qwen36MoePersistentMode::Full => "qwen36.persistent_decode",
        Qwen36MoePersistentMode::RouterOnly => "qwen36.persistent_router_only",
        Qwen36MoePersistentMode::FfnOnly => "qwen36.persistent_ffn_only",
        Qwen36MoePersistentMode::AttnOnly => "qwen36.persistent_attn_only",
        Qwen36MoePersistentMode::FfnStage(1) => "qwen36.persistent_ffn_stage1",
        Qwen36MoePersistentMode::FfnStage(2) => "qwen36.persistent_ffn_stage2",
        Qwen36MoePersistentMode::FfnStage(3) => "qwen36.persistent_ffn_stage3",
        Qwen36MoePersistentMode::FfnStage(4) => "qwen36.persistent_ffn_stage4",
        Qwen36MoePersistentMode::FfnStage(5) => "qwen36.persistent_ffn_stage5",
        Qwen36MoePersistentMode::FfnStage(_) => "qwen36.persistent_ffn_stage_invalid",
        Qwen36MoePersistentMode::LinearStage(1) => "qwen36.persistent_linear_stage1",
        Qwen36MoePersistentMode::LinearStage(2) => "qwen36.persistent_linear_stage2",
        Qwen36MoePersistentMode::LinearStage(3) => "qwen36.persistent_linear_stage3",
        Qwen36MoePersistentMode::LinearStage(4) => "qwen36.persistent_linear_stage4",
        Qwen36MoePersistentMode::LinearStage(5) => "qwen36.persistent_linear_stage5",
        Qwen36MoePersistentMode::LinearStage(_) => "qwen36.persistent_linear_stage_invalid",
    };
    crate::prefill_ffi::ffi_profile_time_result(op, ordinal, || {
        let status: c_int = match backend {
            Backend::Hip | Backend::Cuda => {
                #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
                unsafe {
                    qwen36_moe_hip_persistent_decode_launch(
                        dtype.kernel_dtype_code(),
                        ordinal,
                        num_layers as c_int,
                        start_layer as c_int,
                        end_layer_exclusive as c_int,
                        mode.as_ffi(),
                        layers_device.as_ptr() as *const Qwen36MoeDecodeLayerDesc,
                        int4_ptr,
                        kv_fp8_ptr,
                        geom.hidden,
                        geom.num_heads,
                        geom.num_kv_heads,
                        geom.head_dim,
                        geom.rotary_dim,
                        geom.num_k_heads,
                        geom.num_v_heads,
                        geom.head_k_dim,
                        geom.head_v_dim,
                        geom.conv_kernel_dim,
                        geom.num_experts,
                        geom.moe_intermediate,
                        geom.shared_intermediate,
                        geom.top_k,
                        vocab,
                        geom.rope_theta,
                        geom.rms_norm_eps,
                        position,
                        cache_pos,
                        embed_w_ptr,
                        token_id as c_int,
                        token_ids_ptr,
                        prefill_len as c_int,
                        hidden_ping.as_mut_ptr(),
                        hidden_pong.as_mut_ptr(),
                        workspace.as_mut_ptr() as *mut f32,
                        ffn_topk_idx_scratch.as_mut_ptr() as *mut c_int,
                        final_norm_w_ptr,
                        lm_head_w_ptr,
                        logits_out_ptr,
                        top1_out_ptr,
                        counters,
                        barrier_counter,
                        barrier_flag,
                    )
                }
                #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
                {
                    return Err(GpuError::InvalidArg(
                        "qwen36_moe::persistent_decode_launch: GPU backend not compiled".into(),
                    ));
                }
            }
            Backend::Metal => {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::persistent_decode_launch: Metal backend not yet wired".into(),
                ));
            }
        };
        if status != 0 {
            return Err(qwen36_backend_error(
                backend,
                "qwen36_moe persistent decode launch",
                status,
            ));
        }
        Ok(())
    })
}
