/// Attribution-only MPP pilot used by the Apple M5 Metal bench harness.
///
/// This measures repeated exact `64x32x64` MPP tensor tiles as an equivalent
/// square GEMM throughput number. It does not consume Qwen3.6 model weights
/// and must not be interpreted as a decode-path replacement.
pub fn metal_mpp_tile_gemm_f16_tflops(size: u32, iterations: u32) -> Result<f64, GpuError> {
    crate::metal_native::mpp_tile_gemm_f16_tflops(size, iterations)
}

#[derive(Debug, Clone, Copy)]
pub struct MetalMpsExpertF16Probe {
    pub gate_up_ms: f64,
    pub down_ms: f64,
    pub gate_up_tflops: f64,
    pub down_tflops: f64,
}

/// Attribution-only MPS probe for Qwen3.6 active-expert GEMV shapes.
///
/// This is a resident-FP16 vendor-library upper-bound probe. It does not use the
/// GPTQ INT4 expert buffers directly and does not change the decode path.
pub fn metal_mps_expert_f16_probe(
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
    iterations: u32,
) -> Result<MetalMpsExpertF16Probe, GpuError> {
    let probe = crate::metal_native::qwen36_mps_expert_f16_probe(
        hidden,
        moe_intermediate,
        top_k,
        iterations,
    )?;
    Ok(MetalMpsExpertF16Probe {
        gate_up_ms: probe.gate_up_ms,
        down_ms: probe.down_ms,
        gate_up_tflops: probe.gate_up_tflops,
        down_tflops: probe.down_tflops,
    })
}

#[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
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
    ) -> Qwen36BridgeStatus;

    /// Phase 3e: persistent decode megakernel launcher. One cooperative
    /// HIP launch processes all `num_layers` of {attn or linear-attn, FFN}
    /// — replaces 80 step launches/token with 1 (the lm_head still
    /// launches separately at this stage).
    ///
    /// See `kernels/qwen36_moe_persistent/persistent_decode.hip` for the
    /// kernel and `kernels/qwen36_moe_bridge.cpp::qwen36_moe_hip_persistent_decode_launch`
    /// for the launcher.
    ///
    /// Caller responsibilities:
    /// - `hidden_ping` is uploaded with the initial hidden BF16 bytes; the
    ///   final hidden lands back in `hidden_ping` after even `num_layers`
    ///   (the bridge rejects odd `num_layers`).
    /// - `int4_scales` is null for BF16 baked models, non-null for INT4
    ///   bakes (one entry per layer, parallel to `layers`).
    /// - `workspace` is at least
    ///   `max(attn_workspace_floats(geom), ffn_workspace_floats(geom))` F32
    ///   entries — same as the chained driver.
    /// - `ffn_topk_idx_scratch` is a small `[top_k]` i32 buffer (the FFN
    ///   phase only writes it at stage 1, but the parameter must be valid).
    /// - sync_buf layout: counters[0..16] u32 + barrier_counter at +64 +
    ///   barrier_flag at +68 (96 bytes total, zeroed by the bridge).
    pub fn qwen36_moe_hip_persistent_decode_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: c_int,
        start_layer: c_int,
        end_layer_exclusive: c_int,
        mode: c_int,
        layers: *const Qwen36MoeDecodeLayerDesc,
        int4_scales: *const Qwen36MoeInt4ScaleDesc,
        // Null when KV-FP8 is off globally. Otherwise an array of
        // `num_layers` entries parallel to `layers`. Full-attention
        // layers populate `kv_scale_k` / `kv_scale_v` together (both
        // set, or both null to disable KV-FP8 for that layer specifically
        // — a valid mixed-mode configuration). Linear-attention layers
        // must leave both null. The bridge validates these invariants
        // and rejects malformed descriptors before kernel launch.
        kv_fp8_descs: *const Qwen36MoeKVCacheFp8Desc,
        hidden: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        conv_kernel_dim: c_int,
        num_experts: c_int,
        moe_intermediate: c_int,
        shared_intermediate: c_int,
        top_k: c_int,
        vocab: c_int,
        rope_theta: f32,
        rms_norm_eps: f32,
        position: c_int,
        // -1 ⇒ inherit from `position` (dense base-decode case);
        // ≥ 0 ⇒ decoupled KV slot for SpecPrefill sparse-prefill or
        // MTP draft layers.
        cache_pos: c_int,
        embed_w: *const c_void,
        token_id: c_int,
        token_ids: *const c_uint,
        prefill_len: c_int,
        hidden_ping: *mut c_void,
        hidden_pong: *mut c_void,
        workspace: *mut f32,
        ffn_topk_idx_scratch: *mut c_int,
        // Phase 3f folded final RMSnorm + lm_head GEMV. Pass nullptr
        // triple + vocab=0 to skip (prefill steps); otherwise the
        // megakernel writes logits to `logits_out` and the host can
        // skip the separate `lm_head_launch` call.
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits_out: *mut c_void,
        top1_out: *mut c_uint,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> Qwen36BridgeStatus;

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
    /// `sync_buf` (counters/barrier_counter/barrier_flag) must be 96 zero
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
        cache_pos: c_int,
        input_hidden: *const c_void,
        input_norm_w: *const c_void,
        q_proj_w: *const c_void,
        k_proj_w: *const c_void,
        v_proj_w: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_w: *const c_void,
        int4_group_size: c_int,
        q_proj_scale: *const c_void,
        q_proj_zero: *const c_void,
        k_proj_scale: *const c_void,
        k_proj_zero: *const c_void,
        v_proj_scale: *const c_void,
        v_proj_zero: *const c_void,
        o_proj_scale: *const c_void,
        o_proj_zero: *const c_void,
        output: *mut c_void,
        workspace: *mut f32,
        kv_cache_k: *mut c_void,
        kv_cache_v: *mut c_void,
        kv_max_t: c_int,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> Qwen36BridgeStatus;

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
    /// (counters/barrier_counter/barrier_flag) must be 96 zero bytes.
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
        int4_group_size: c_int,
        in_proj_qkv_scale: *const c_void,
        in_proj_qkv_zero: *const c_void,
        in_proj_z_scale: *const c_void,
        in_proj_z_zero: *const c_void,
        out_proj_scale: *const c_void,
        out_proj_zero: *const c_void,
        output: *mut c_void,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> Qwen36BridgeStatus;

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
    /// barrier_flag) must be 96 zero bytes.
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
    ) -> Qwen36BridgeStatus;

    /// Task 8 descriptor-driven scalar/8-wide dequant test surface. This is
    /// deliberately separate from production chained and persistent decode.
    pub fn qwen36_moe_hip_int4_descriptor_dequant_smoke_launch(
        device_ordinal: usize,
        packed: *const u8,
        desc: *const Qwen36MoeInt4WeightDesc,
        experts: c_int,
        out_rows: c_int,
        in_cols: c_int,
        dq_8_out: *mut f32,
        dq_scalar_out: *mut f32,
    ) -> Qwen36BridgeStatus;

    /// Task 8 gfx11-only descriptor-driven scalar/WMMA parity surface.
    pub fn qwen36_moe_hip_int4_descriptor_wmma_parity_launch(
        device_ordinal: usize,
        packed: *const u8,
        desc: *const Qwen36MoeInt4WeightDesc,
        activation: *const c_void,
        out_rows: c_int,
        in_cols: c_int,
        scalar_out: *mut f32,
        wmma_out: *mut f32,
    ) -> Qwen36BridgeStatus;

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
    ) -> Qwen36BridgeStatus;

    /// Final RMSNorm + lm_head GEMV in a single kernel — replaces the
    /// host-side `host_final_norm_lm_head_f32` for qwen3.6-MoE.
    ///
    /// All buffers are device pointers, all BF16 (`dtype = 2`):
    ///   - `final_hidden`: [hidden] BF16, the output of `run_chained_decode`.
    ///   - `final_norm_w`: [hidden] BF16 — `model.norm.weight`. Applies the
    ///      HF `Qwen3_5MoeRMSNorm` `(1 + w)` unit offset.
    ///   - `lm_head_w`: [vocab, hidden] BF16, dequantized once at startup.
    ///   - `logits`: [vocab] BF16, output.
    ///   - `counter`: [1] u32. Used as a work-stealing atomic across vocab
    ///     rows; the launcher memsets it to 0 before each call so the
    ///     caller doesn't need to.
    ///
    /// Returns 0 on success; non-zero on validation / launch failure (see
    /// `qwen36_moe_hip_lm_head_launch` in `kernels/qwen36_moe_bridge.cpp`
    /// for the error code matrix).
    pub fn qwen36_moe_hip_lm_head_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        // Optional BF16 [hidden] export of the post-RMSNorm hidden
        // state. Phase 6.2c.3 plumbing for the MTP draft loop's
        // recurrent feed; null = base-decode behavior unchanged.
        x_normed_out: *mut c_void,
        counter: *mut c_uint,
    ) -> Qwen36BridgeStatus;

    /// FFI bridge for the batched lm_head WMMA kernel (Phase 6.4a). Wraps
    /// `qwen36_moe_lm_head_batched_wmma_kernel`. `m` is the runtime batch
    /// size (1..16); for `m == 1` the single-M path
    /// (`qwen36_moe_hip_lm_head_launch`) is faster — use the batched
    /// launcher when `m >= 2` to amortize the lm_head BF16 weight read.
    /// WMMA-only (gfx11xx); returns status 138 on unsupported hardware
    /// or `hidden % 16 != 0` so the caller can fall back to a per-row
    /// loop over the single-M launcher.
    pub fn qwen36_moe_hip_lm_head_batched_launch(
        dtype: c_int,
        device_ordinal: usize,
        m: c_int,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        x_normed_out: *mut c_void,
    ) -> Qwen36BridgeStatus;

    /// FFI bridge for the MTP pre-fusion kernel (Phase 6.2c.1). Single-block
    /// launch: BF16 RMSNorms over `e_in` and `h_base` followed by a
    /// `mtp.fc @ cat([e_norm, h_norm])` matvec into `fused_out`. All buffers
    /// must be device-resident on `device_ordinal` and BF16. See
    /// `qwen36_moe_hip_mtp_pre_fusion_launch` in `kernels/qwen36_moe_bridge.cpp`
    /// for the error code matrix.
    pub fn qwen36_moe_hip_mtp_pre_fusion_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        rms_norm_eps: f32,
        e_in: *const c_void,
        h_base: *const c_void,
        pre_fc_norm_embedding_w: *const c_void,
        pre_fc_norm_hidden_w: *const c_void,
        fc_w: *const c_void,
        e_norm_out: *mut c_void,
        h_norm_out: *mut c_void,
        fused_out: *mut c_void,
    ) -> Qwen36BridgeStatus;

    /// Stage A (M3) batched-Q full-attention prefill kernel. Standalone
    /// attention math: Q/K/V are pre-projected and pre-RoPE'd by the
    /// caller, the K/V cache is pre-written. Output is pre-o_proj
    /// `[batch, q_heads, q_len, head_dim]` in F32.
    ///
    /// Shapes (BF16 unless noted):
    /// - `query`: `[batch, q_heads, q_len, head_dim]`
    /// - `key`:   `[batch, kv_heads, kv_len, head_dim]`
    /// - `value`: `[batch, kv_heads, kv_len, head_dim]`
    /// - `out` (F32): `[batch, q_heads, q_len, head_dim]`
    ///
    /// `seqlen_offset = past_len`; query at chunk position `qr` attends to
    /// cache positions `[0, past_len + qr]` (causal, inclusive). `kv_len`
    /// is the total cache length the kernel may read (typically
    /// `past_len + q_len`).
    ///
    /// Status codes (non-zero = failure):
    ///   130 dtype != bf16    131 invalid heads        132 q_heads % kv_heads
    ///   133 head_dim out of range  134 q_len/kv_len   135 seqlen_offset / overflow
    ///   136 batch_size       137 wave64 (unsupported) 138 LDS overflow
    ///   254 launch error     255 sync error
    pub fn qwen36_moe_hip_batched_prefill_attn_full_launch(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: c_int,
        q_heads: c_int,
        kv_heads: c_int,
        q_len: c_int,
        kv_len: c_int,
        head_dim: c_int,
        scale: f32,
        seqlen_offset: c_int,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        out: *mut c_void,
    ) -> Qwen36BridgeStatus;

    /// Stage B (M9) router permutation kernel. Groups per-token top-K expert
    /// assignments by target expert (counting-sort, single block).
    ///
    /// Inputs (GPU buffers):
    /// - `topk_idx`     : `[n_tokens, top_k]` i32 — per-token expert ids in
    ///                     `[0, num_experts)`.
    /// - `topk_weight`  : `[n_tokens, top_k]` BF16 — routing weights.
    ///
    /// Outputs (caller-allocated GPU buffers):
    /// - `expert_offsets`     : `[num_experts + 1]` i32 — prefix sum.
    /// - `permuted_token_idx` : `[n_tokens * top_k]` i32 — sorted token ids.
    /// - `permuted_kpos`      : `[n_tokens * top_k]` i32 — top-K slot ids.
    /// - `permuted_weight`    : `[n_tokens * top_k]` BF16 — routing weights.
    ///
    /// Within an expert's segment the order is unstable (atomicAdd cursor);
    /// callers comparing against a CPU reference must compare per-segment as
    /// a multiset.
    ///
    /// Status codes (non-zero = failure):
    ///   140 invalid args (n_tokens/top_k/num_experts <= 0)
    ///   141 num_experts > 256       142 top_k > 16
    ///   143 n_tokens * top_k > 16384
    ///   254 launch error            255 sync error
    pub fn qwen36_moe_hip_batched_prefill_router_permute_launch(
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        num_experts: c_int,
        topk_idx: *const c_void,
        topk_weight: *const c_void,
        expert_offsets: *mut c_void,
        permuted_token_idx: *mut c_void,
        permuted_kpos: *mut c_void,
        permuted_weight: *mut c_void,
    ) -> Qwen36BridgeStatus;

    /// Stage B (M10) grouped-expert INT4 GEMM kernel. One launch processes
    /// ALL `num_experts` experts via persistent-block work-stealing on the
    /// expert id; for each expert it walks the segment of permuted rows
    /// produced by the M9 router permutation kernel and runs gate_up +
    /// silu*mul + down INT4 matmuls per row.
    ///
    /// Inputs (GPU buffers):
    /// - `x_norm`              : `[n_tokens, hidden]` BF16 — post-input-RMSnorm
    ///                            hidden states; gathered by `permuted_token_idx`.
    /// - `expert_offsets`      : `[num_experts + 1]` i32 — M9 prefix sum.
    /// - `permuted_token_idx`  : `[n_tokens * top_k]` i32 — M9 sort output.
    /// - `experts_gate_up_w/s/z` : `[E, 2*I, hidden/2]` u8 + `[E, 2*I/gs, hidden/gs]` BF16.
    /// - `experts_down_w/s/z`    : `[E, hidden, I/2]` u8 + `[E, hidden/gs, I/gs]` BF16.
    ///
    /// Caller-owned buffers:
    /// - `expert_out` : `[n_tokens * top_k, hidden]` BF16 — per-permuted-row
    ///                   expert output; M11 unpermutes + combines.
    /// - `counters`   : `[1]` u32 — work-stealing claim counter; CALLER MUST
    ///                   ZERO BEFORE LAUNCH.
    ///
    /// Status codes (non-zero = failure):
    ///   150 invalid args (zero/negative dims)
    ///   151 num_experts > 256
    ///   152 hidden / moe_intermediate not divisible by group_size (or 16)
    ///   153 group_size != 128
    ///   154 top_k * n_tokens > 16384
    ///   155 dtype != bf16
    ///   156 LDS overflow
    ///   254 launch error                255 sync error
    pub fn qwen36_moe_hip_batched_prefill_grouped_expert_launch(
        dtype: c_int,
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        num_experts: c_int,
        hidden: c_int,
        moe_intermediate: c_int,
        group_size: c_int,
        x_norm: *const c_void,
        expert_offsets: *const c_void,
        permuted_token_idx: *const c_void,
        experts_gate_up_w: *const c_void,
        experts_gate_up_scale: *const c_void,
        experts_gate_up_zero: *const c_void,
        experts_down_w: *const c_void,
        experts_down_scale: *const c_void,
        experts_down_zero: *const c_void,
        expert_out: *mut c_void,
        counters: *mut c_void,
    ) -> Qwen36BridgeStatus;

    /// Stage B (M11) unpermute + weighted combine kernel. Inverts the M9
    /// router permutation (host-built `permuted_inverse` table) and computes
    /// the per-token weighted sum of `top_k` expert outputs.
    ///
    /// Inputs (GPU buffers):
    /// - `permuted_inverse` : `[n_tokens * top_k]` i32 — host-built inverse
    ///                         of M9's scatter, so
    ///                         `permuted_inverse[token * top_k + kpos] = dst`
    ///                         where `dst` is the M9/M10 row index for that
    ///                         (token, kpos) pair.
    /// - `permuted_weight`  : `[n_tokens * top_k]` BF16 — M9 output.
    /// - `expert_out`       : `[n_tokens * top_k, hidden]` BF16 — M10 output.
    ///
    /// Output (caller-allocated GPU buffer):
    /// - `combined`         : `[n_tokens, hidden]` BF16 — weighted sum
    ///                         of expert outputs per token.
    ///
    /// Status codes (non-zero = failure):
    ///   160 invalid args (zero/negative dims)
    ///   161 top_k > 16
    ///   162 dtype != bf16
    ///   163 hidden too large (>65536)
    ///   254 launch error            255 sync error
    pub fn qwen36_moe_hip_batched_prefill_unpermute_combine_launch(
        dtype: c_int,
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        hidden: c_int,
        permuted_inverse: *const c_void,
        permuted_weight: *const c_void,
        expert_out: *const c_void,
        combined: *mut c_void,
    ) -> Qwen36BridgeStatus;
}

fn validate_descriptor_int4_common(
    operation: &str,
    packed: &GpuBuffer,
    desc: &Qwen36MoeInt4WeightDesc,
    experts: i32,
    out_rows: i32,
    in_cols: i32,
) -> Result<usize, GpuError> {
    if packed.backend() != Backend::Hip {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: HIP buffer required"
        )));
    }
    if packed.dtype() != ScalarType::U8 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: packed weights must be U8"
        )));
    }
    if experts <= 0 || out_rows <= 0 || in_cols <= 0 || in_cols % 8 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: positive dimensions and in_cols divisible by 8 required"
        )));
    }
    if desc.encoding == 3 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: encoding 3 is FP8, not INT4"
        )));
    }
    if !matches!(desc.encoding, 1 | 2) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: unsupported INT4 encoding {}",
            desc.encoding
        )));
    }
    if desc.scale.is_null()
        || desc.input_group_size <= 0
        || desc.output_group_size <= 0
        || in_cols % desc.input_group_size != 0
        || out_rows % desc.output_group_size != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: malformed descriptor group geometry"
        )));
    }
    if desc.encoding == 1
        && (desc.zero.is_null()
            || desc.implicit_zero_code >= 0
            || desc.input_group_size != 128
            || desc.output_group_size != 128)
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: tile-v1 encoding 1 requires explicit zero values"
        )));
    }
    if desc.encoding == 2
        && (!desc.zero.is_null()
            || desc.input_group_size != 32
            || desc.output_group_size != 1
            || desc.implicit_zero_code != 8)
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: row-group encoding 2 requires G32, output group 1, and implicit zero 8"
        )));
    }
    let logical_row_bytes = (in_cols / 2) as u64;
    let logical_scale_row = (in_cols / desc.input_group_size) as u64;
    if desc.packed_row_stride_bytes < logical_row_bytes
        || desc.scale_row_stride_elements < logical_scale_row
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: descriptor row stride is shorter than its logical row"
        )));
    }
    let packed_per_expert = (out_rows as u64 - 1)
        .checked_mul(desc.packed_row_stride_bytes)
        .and_then(|offset| offset.checked_add(logical_row_bytes))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!("qwen36_moe::{operation}: packed extent overflows"))
        })?;
    if experts > 1 && desc.packed_expert_stride_bytes < packed_per_expert {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: packed expert stride is too short"
        )));
    }
    let scale_rows = (out_rows / desc.output_group_size) as u64;
    let scale_per_expert = (scale_rows - 1)
        .checked_mul(desc.scale_row_stride_elements)
        .and_then(|offset| offset.checked_add(logical_scale_row))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!("qwen36_moe::{operation}: scale extent overflows"))
        })?;
    if experts > 1 && desc.scale_expert_stride_elements < scale_per_expert {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: scale expert stride is too short"
        )));
    }
    let packed_extent = (experts as u64 - 1)
        .checked_mul(desc.packed_expert_stride_bytes)
        .and_then(|offset| offset.checked_add(packed_per_expert))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!("qwen36_moe::{operation}: packed extent overflows"))
        })?;
    if packed_extent > packed.len_bytes() as u64 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::{operation}: packed buffer is shorter than descriptor extent"
        )));
    }
    (experts as usize)
        .checked_mul(out_rows as usize)
        .and_then(|count| count.checked_mul(in_cols as usize))
        .ok_or_else(|| {
            GpuError::InvalidArg(format!("qwen36_moe::{operation}: output extent overflows"))
        })
}

#[allow(clippy::too_many_arguments)]
pub fn int4_descriptor_dequant_smoke_launch(
    ordinal: usize,
    packed: &GpuBuffer,
    desc: &Qwen36MoeInt4WeightDesc,
    experts: i32,
    out_rows: i32,
    in_cols: i32,
    dq_8_out: &mut GpuBuffer,
    dq_scalar_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let output_count = validate_descriptor_int4_common(
        "int4_descriptor_dequant_smoke_launch",
        packed,
        desc,
        experts,
        out_rows,
        in_cols,
    )?;
    for output in [&*dq_8_out, &*dq_scalar_out] {
        if output.backend() != Backend::Hip
            || output.dtype() != ScalarType::F32
            || output.elem_count() < output_count
        {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::int4_descriptor_dequant_smoke_launch: F32 HIP outputs are too short"
                    .into(),
            ));
        }
    }
    #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
    let status = unsafe {
        qwen36_moe_hip_int4_descriptor_dequant_smoke_launch(
            ordinal,
            packed.as_ptr() as *const u8,
            desc,
            experts,
            out_rows,
            in_cols,
            dq_8_out.as_mut_ptr() as *mut f32,
            dq_scalar_out.as_mut_ptr() as *mut f32,
        )
    };
    #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
    return Err(GpuError::InvalidArg(
        "qwen36_moe::int4_descriptor_dequant_smoke_launch: HIP backend not compiled".into(),
    ));
    #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
    qwen36_bridge_result(
        Backend::Hip,
        "qwen36_moe int4 descriptor dequant smoke launch",
        status,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn int4_descriptor_wmma_parity_launch(
    ordinal: usize,
    packed: &GpuBuffer,
    desc: &Qwen36MoeInt4WeightDesc,
    activation: &GpuBuffer,
    out_rows: i32,
    in_cols: i32,
    scalar_out: &mut GpuBuffer,
    wmma_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let output_count = validate_descriptor_int4_common(
        "int4_descriptor_wmma_parity_launch",
        packed,
        desc,
        1,
        out_rows,
        in_cols,
    )?;
    if out_rows != 32 || in_cols != 128 {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::int4_descriptor_wmma_parity_launch: fixture must be [32, 128]".into(),
        ));
    }
    if activation.backend() != Backend::Hip
        || activation.dtype() != ScalarType::BF16
        || activation.elem_count() < in_cols as usize
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::int4_descriptor_wmma_parity_launch: BF16 HIP activation is too short"
                .into(),
        ));
    }
    for output in [&*scalar_out, &*wmma_out] {
        if output.backend() != Backend::Hip
            || output.dtype() != ScalarType::F32
            || output.elem_count() < output_count / in_cols as usize
        {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::int4_descriptor_wmma_parity_launch: F32 HIP outputs are too short"
                    .into(),
            ));
        }
    }
    #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
    let status = unsafe {
        qwen36_moe_hip_int4_descriptor_wmma_parity_launch(
            ordinal,
            packed.as_ptr() as *const u8,
            desc,
            activation.as_ptr(),
            out_rows,
            in_cols,
            scalar_out.as_mut_ptr() as *mut f32,
            wmma_out.as_mut_ptr() as *mut f32,
        )
    };
    #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
    return Err(GpuError::InvalidArg(
        "qwen36_moe::int4_descriptor_wmma_parity_launch: HIP backend not compiled".into(),
    ));
    #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
    qwen36_bridge_result(
        Backend::Hip,
        "qwen36_moe int4 descriptor WMMA parity launch",
        status,
    )
}

/// Safe wrapper over the stub launch. The engine pre-allocates `sync_buf`
/// as a 96-byte zeroed scratch — 16 u32 work-stealing counter slots at
/// +0..+63 (only counters[0] used here; the FFN concurrent-experts dispatch
/// uses 2*K_top of them), grid barrier counter at +64, flag at +68. The
/// 32-byte form used by `crate::persistent_decode_4b` is the older single-
/// counter layout — qwen36_moe shares one widened sync_buf across all four
/// step launchers (stub/attn/linear/ffn) so any can run with any.
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
    // Layout: 16 u32 work-stealing counter slots at +0..+63 (the FFN
    // concurrent-experts dispatch uses 2*K_top of these; attn/linear/stub
    // only touch counters[0]). Barrier counter+flag follow at +64/+68.
    // Sync_buf must be at least 96 bytes zeroed before launch.
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };

    let status: Qwen36BridgeStatus = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
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
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::stub_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::stub_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    qwen36_bridge_result(backend, "qwen36_moe stub launch", status)?;
    Ok(())
}
