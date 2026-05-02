//! Production wiring for the Qwen3.6-MoE persistent decode megakernel.
//!
//! The megakernel and its bit-exact-vs-chained parity test live in
//! `kernels/qwen36_moe_persistent/persistent_decode.hip` and
//! `crates/runner/tests/qwen36_moe_multilayer_parity.rs::multilayer_persistent_decode_matches_chained`
//! (PR #126). This module is the engine-side glue: it builds the layer
//! descriptor array, allocates the persistent-launch scratch buffers
//! once before the decode loop, and per-step calls `persistent_decode_launch`.
//!
//! ## What it replaces
//!
//! [`PersistentScratch::run`] is a drop-in replacement for
//! [`crate::qwen36_moe_decode::run_chained_decode_fast`]: same signature
//! shape (`initial_hidden`, `position`), same return type
//! ([`crate::qwen36_moe_decode::DecodeOutputs`]). The chained path runs 80
//! step launches/token (40 attn + 40 ffn); the persistent path runs 1
//! cooperative launch.
//!
//! ## What's lost in the timing surface
//!
//! `DecodeOutputs.kernel_full_attn_us` / `kernel_linear_attn_us` /
//! `kernel_ffn_us` can't be split apart inside one launch. The persistent
//! path lumps the wall-clock into `kernel_full_attn_us` and reports
//! `kernel_linear_attn_us = kernel_ffn_us = 0` so existing
//! `--emit-stage-timings` infra keeps working. Per-stage attribution
//! requires re-running through the chained path (still available — engine
//! gates on `--persistent-decode`).

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_d2h, copy_h2d, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    persistent_decode_launch, Qwen36MoeDecodeLayerDesc, Qwen36MoeInt4ScaleDesc,
    Qwen36MoePersistentGeom,
};
use std::ffi::c_void;
use std::os::raw::c_int;

use crate::qwen36_moe_decode::{
    ffn_workspace_floats, full_attn_workspace_floats, linear_attn_workspace_floats,
    AttnLayerBuffers, DecodeOutputs, LayerBuffers, MultiLayerGeom,
};

/// Pre-allocated scratch + cached descriptor arrays for the persistent
/// decode megakernel. Built once before the decode loop; reused for every
/// step. The layer descriptors hold *device pointers* into the live
/// `LayerBuffers` GpuBuffers — those pointers stay valid because the
/// engine never re-allocates per-layer weights or state during decode.
///
/// Lifetime: bound to the engine's `layers: Vec<LayerBuffers>` (via the
/// pointers cached in `layer_descs_dev`). If the engine ever re-allocates
/// any layer's weight or state buffer, the scratch must be rebuilt.
pub struct PersistentScratch {
    pub geom: Qwen36MoePersistentGeom,
    pub num_layers: usize,
    /// `[num_layers]` descriptors uploaded as opaque U8 bytes.
    pub layer_descs_dev: GpuBuffer,
    /// `[num_layers]` INT4 sidecar descriptors. `None` for BF16 bakes.
    pub int4_scales_dev: Option<GpuBuffer>,
    /// Two `[hidden]` BF16 buffers — kernel ping-pongs residuals through
    /// them. Per-step, host uploads the fresh `initial_hidden` into
    /// `hidden_ping`; the kernel returns the final hidden in
    /// `hidden_ping` (two swaps per layer cancel — see the kernel
    /// docstring for the math).
    pub hidden_ping: GpuBuffer,
    pub hidden_pong: GpuBuffer,
    /// F32 shared scratch sized for `max(full_attn, linear_attn, ffn)`
    /// workspace footprints.
    pub workspace: GpuBuffer,
    /// `[top_k]` i32 — only consumed by the FFN phase at stage 1; passed
    /// to satisfy the kernel signature (its body never derefs at stage 5).
    pub ffn_topk_idx_scratch: GpuBuffer,
    /// 96-byte sync_buf: counters[0..16] + barrier_counter (+64) +
    /// barrier_flag (+68). Bridge zeros it on every launch.
    pub sync_buf: GpuBuffer,
}

impl PersistentScratch {
    /// Build the descriptor array + allocate scratch. Mutably borrows
    /// `layers` only for descriptor construction (mutable state pointers
    /// are cached into the descs); subsequent [`Self::run`] calls don't
    /// need `&mut layers`.
    pub fn new(
        ordinal: usize,
        geom: &MultiLayerGeom,
        layers: &mut [LayerBuffers],
    ) -> Result<Self> {
        let num_layers = layers.len();
        let descs = build_layer_descs(layers);
        let layer_descs_dev =
            upload_descs(ordinal, &descs).context("upload layer descriptor array")?;
        let int4_scales_dev = match build_int4_descs(layers) {
            Some(int4) => Some(upload_descs(ordinal, &int4).context("upload int4 scale descs")?),
            None => None,
        };

        let hidden = geom.hidden as usize;
        let hidden_ping = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .context("alloc persistent hidden_ping")?;
        let hidden_pong = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .context("alloc persistent hidden_pong")?;

        // Kv-cache adds OFF_SCORES = [num_attention_heads * kv_max_t] F32
        // when any full-attn layer carries a cache. Mirror the chained
        // driver's `attn_extra` calc.
        let max_kv_t = layers
            .iter()
            .filter_map(|l| match &l.attn {
                AttnLayerBuffers::Full {
                    kv_cache: Some(c), ..
                } => Some(c.kv_max_t as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let attn_extra = if max_kv_t > 0 {
            geom.num_attention_heads as usize * max_kv_t
        } else {
            0
        };
        let ws_floats = full_attn_workspace_floats(geom)
            .max(linear_attn_workspace_floats(geom))
            .max(ffn_workspace_floats(geom))
            + attn_extra;
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ws_floats])
            .context("alloc persistent workspace")?;
        let ffn_topk_idx_scratch =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
                .context("alloc ffn_topk_idx_scratch")?;
        let sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])
            .context("alloc persistent sync_buf")?;

        let pgeom = Qwen36MoePersistentGeom {
            hidden: geom.hidden,
            num_heads: geom.num_attention_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: geom.rotary_dim,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rope_theta: geom.rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
        };

        Ok(Self {
            geom: pgeom,
            num_layers,
            layer_descs_dev,
            int4_scales_dev,
            hidden_ping,
            hidden_pong,
            workspace,
            ffn_topk_idx_scratch,
            sync_buf,
        })
    }

    /// One decode step. H2D the freshly-embedded `initial_hidden` into
    /// `hidden_ping`, run the megakernel, D2H the final hidden back.
    /// Mutates the linear-attn state in place (via the pointers cached in
    /// `layer_descs_dev`) — same semantics as
    /// `run_chained_decode_fast`.
    pub fn run(
        &mut self,
        ordinal: usize,
        initial_hidden_bytes: &[u8],
        position: i32,
    ) -> Result<DecodeOutputs> {
        let hidden_bytes = self.geom.hidden as usize * 2;
        if initial_hidden_bytes.len() != hidden_bytes {
            return Err(anyhow!(
                "initial_hidden_bytes len {} != expected {} (hidden*2 BF16 bytes)",
                initial_hidden_bytes.len(),
                hidden_bytes,
            ));
        }
        copy_h2d(
            ordinal,
            self.hidden_ping.as_mut_ptr(),
            initial_hidden_bytes.as_ptr() as *const _,
            hidden_bytes,
        )
        .context("h2d initial_hidden -> hidden_ping")?;

        let t_launch = std::time::Instant::now();
        persistent_decode_launch(
            ordinal,
            ScalarType::BF16,
            self.geom,
            position,
            &self.layer_descs_dev,
            self.int4_scales_dev.as_ref(),
            self.num_layers,
            &mut self.hidden_ping,
            &mut self.hidden_pong,
            &mut self.workspace,
            &mut self.ffn_topk_idx_scratch,
            &mut self.sync_buf,
        )
        .map_err(|e: GpuError| anyhow!(e))
        .context("persistent_decode_launch")?;
        let elapsed_us = t_launch.elapsed().as_micros() as u64;

        // D2H the final hidden — same as run_chained_decode_fast does at
        // the end of its chain. The bridge syncs (hipDeviceSynchronize)
        // before returning, so this measurement covers real GPU compute
        // time, not host queue time.
        let mut final_hidden_bytes = vec![0u8; hidden_bytes];
        copy_d2h(
            ordinal,
            final_hidden_bytes.as_mut_ptr() as *mut _,
            self.hidden_ping.as_ptr(),
            hidden_bytes,
        )
        .context("d2h hidden_ping -> final_hidden_bytes")?;

        // Stage attribution isn't recoverable inside one launch — we
        // lump the whole wall-clock into `kernel_full_attn_us` so
        // `--emit-stage-timings` still surfaces *something*. Per-phase
        // breakdowns require running through the chained path.
        Ok(DecodeOutputs {
            final_hidden_bytes,
            per_layer_attn_out: Vec::new(),
            per_layer_ffn_out: Vec::new(),
            kernel_full_attn_us: elapsed_us,
            kernel_linear_attn_us: 0,
            kernel_ffn_us: 0,
        })
    }
}

/// Build the `Qwen36MoeDecodeLayerDesc[num_layers]` array from the live
/// `LayerBuffers` slice. All weight pointers come from the GpuBuffers'
/// device addresses.
pub fn build_layer_descs(layers: &mut [LayerBuffers]) -> Vec<Qwen36MoeDecodeLayerDesc> {
    let mut descs = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter_mut().enumerate() {
        let mut d = Qwen36MoeDecodeLayerDesc::default();
        d.layer_idx = li as c_int;
        d.is_full_attention = if l.is_full_attn() { 1 } else { 0 };
        match &mut l.attn {
            AttnLayerBuffers::Full {
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
                kv_cache,
                ..
            } => {
                d.input_norm_w = input_norm_w.as_ptr() as *const c_void;
                d.q_proj_w = q_proj_w.as_ptr() as *const c_void;
                d.k_proj_w = k_proj_w.as_ptr() as *const c_void;
                d.v_proj_w = v_proj_w.as_ptr() as *const c_void;
                d.q_norm_w = q_norm_w.as_ptr() as *const c_void;
                d.k_norm_w = k_norm_w.as_ptr() as *const c_void;
                d.o_proj_w = o_proj_w.as_ptr() as *const c_void;
                if let Some(c) = kv_cache.as_mut() {
                    d.kv_cache_k = c.k.as_mut_ptr();
                    d.kv_cache_v = c.v.as_mut_ptr();
                    d.kv_max_t = c.kv_max_t;
                }
            }
            AttnLayerBuffers::Linear {
                input_norm_w,
                in_proj_qkv_w,
                in_proj_z_w,
                in_proj_a_w,
                in_proj_b_w,
                conv1d_w,
                dt_bias,
                a_log,
                norm_w,
                out_proj_w,
                conv_state,
                recurrent_state,
                ..
            } => {
                d.input_norm_w = input_norm_w.as_ptr() as *const c_void;
                d.linear_in_proj_qkv_w = in_proj_qkv_w.as_ptr() as *const c_void;
                d.linear_in_proj_z_w = in_proj_z_w.as_ptr() as *const c_void;
                d.linear_in_proj_a_w = in_proj_a_w.as_ptr() as *const c_void;
                d.linear_in_proj_b_w = in_proj_b_w.as_ptr() as *const c_void;
                d.linear_conv1d_w = conv1d_w.as_ptr() as *const c_void;
                d.linear_dt_bias = dt_bias.as_ptr() as *const c_void;
                d.linear_a_log_exp = a_log.as_ptr() as *const c_void;
                d.linear_norm_w = norm_w.as_ptr() as *const c_void;
                d.linear_out_proj_w = out_proj_w.as_ptr() as *const c_void;
                d.linear_conv_state = conv_state.as_mut_ptr();
                d.linear_recurrent_state = recurrent_state.as_mut_ptr();
            }
        }
        d.post_attn_norm_w = l.ffn.post_attn_norm_w.as_ptr() as *const c_void;
        d.router_w = l.ffn.gate_w.as_ptr() as *const c_void;
        d.experts_gate_up_w = l.ffn.gate_up_proj_w.as_ptr() as *const c_void;
        d.experts_down_w = l.ffn.down_proj_w.as_ptr() as *const c_void;
        d.shared_expert_gate_proj_w = l.ffn.shared_gate_proj_w.as_ptr() as *const c_void;
        d.shared_expert_up_proj_w = l.ffn.shared_up_proj_w.as_ptr() as *const c_void;
        d.shared_expert_down_proj_w = l.ffn.shared_down_proj_w.as_ptr() as *const c_void;
        d.shared_expert_gate_w = l.ffn.shared_expert_gate_w.as_ptr() as *const c_void;
        descs.push(d);
    }
    descs
}

/// Build the parallel `Qwen36MoeInt4ScaleDesc[num_layers]`. Returns
/// `None` when no layer carries INT4 sidecars (BF16 bake).
pub fn build_int4_descs(layers: &[LayerBuffers]) -> Option<Vec<Qwen36MoeInt4ScaleDesc>> {
    let any_int4 = layers.iter().any(|l| {
        let attn_q = match &l.attn {
            AttnLayerBuffers::Full { int4, .. } => int4.is_some(),
            AttnLayerBuffers::Linear { int4, .. } => int4.is_some(),
        };
        attn_q || l.ffn.int4.is_some()
    });
    if !any_int4 {
        return None;
    }
    let mut int4 = Vec::with_capacity(layers.len());
    for l in layers.iter() {
        let mut d = Qwen36MoeInt4ScaleDesc::default();
        match &l.attn {
            AttnLayerBuffers::Full { int4: Some(s), .. } => {
                d.q_proj_scale = s.q_proj_scale.as_ptr() as *const c_void;
                d.q_proj_zero = s.q_proj_zero.as_ptr() as *const c_void;
                d.k_proj_scale = s.k_proj_scale.as_ptr() as *const c_void;
                d.k_proj_zero = s.k_proj_zero.as_ptr() as *const c_void;
                d.v_proj_scale = s.v_proj_scale.as_ptr() as *const c_void;
                d.v_proj_zero = s.v_proj_zero.as_ptr() as *const c_void;
                d.o_proj_scale = s.o_proj_scale.as_ptr() as *const c_void;
                d.o_proj_zero = s.o_proj_zero.as_ptr() as *const c_void;
                d.group_size = s.group_size;
            }
            AttnLayerBuffers::Linear { int4: Some(s), .. } => {
                d.linear_in_proj_qkv_scale = s.in_proj_qkv_scale.as_ptr() as *const c_void;
                d.linear_in_proj_qkv_zero = s.in_proj_qkv_zero.as_ptr() as *const c_void;
                d.linear_in_proj_z_scale = s.in_proj_z_scale.as_ptr() as *const c_void;
                d.linear_in_proj_z_zero = s.in_proj_z_zero.as_ptr() as *const c_void;
                d.linear_out_proj_scale = s.out_proj_scale.as_ptr() as *const c_void;
                d.linear_out_proj_zero = s.out_proj_zero.as_ptr() as *const c_void;
                d.group_size = s.group_size;
            }
            _ => {}
        }
        if let Some(s) = &l.ffn.int4 {
            d.experts_gate_up_scale = s.gate_up_proj_scale.as_ptr() as *const c_void;
            d.experts_gate_up_zero = s.gate_up_proj_zero.as_ptr() as *const c_void;
            d.experts_down_scale = s.down_proj_scale.as_ptr() as *const c_void;
            d.experts_down_zero = s.down_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_gate_proj_scale = s.shared_gate_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_gate_proj_zero = s.shared_gate_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_up_proj_scale = s.shared_up_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_up_proj_zero = s.shared_up_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_down_proj_scale = s.shared_down_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_down_proj_zero = s.shared_down_proj_zero.as_ptr() as *const c_void;
            d.group_size = s.group_size;
        }
        int4.push(d);
    }
    Some(int4)
}

/// Upload a `[T]` slice to a GPU buffer as opaque U8 bytes — the kernel
/// reads through a `*const Qwen36Moe*Desc` pointer cast.
pub fn upload_descs<T: Sized>(ordinal: usize, descs: &[T]) -> Result<GpuBuffer, GpuError> {
    let per = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(per * descs.len());
    for d in descs {
        let p = d as *const T as *const u8;
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(p, per) });
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[bytes.len()], &bytes)
}
