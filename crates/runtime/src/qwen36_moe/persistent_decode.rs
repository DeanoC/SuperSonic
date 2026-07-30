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
//! [`crate::qwen36_moe::decode::run_chained_decode_fast`]: same signature
//! shape (`initial_hidden`, `position`), same return type
//! ([`crate::qwen36_moe::types::DecodeOutputs`]). The chained path runs 80
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
use gpu_hal::{copy_d2h, copy_h2d, sync, Backend, GpuBuffer, GpuError, ScalarType};
use kernel_ffi::qwen36_moe::{
    persistent_decode_launch, persistent_decode_launch_range, Qwen36MoeAttnStepParams,
    Qwen36MoeDecodeLayerDesc, Qwen36MoeInt4ScaleDesc, Qwen36MoeKVCacheFp8Desc,
    Qwen36MoePersistentGeom, Qwen36MoePersistentLmHeadFold, Qwen36MoePersistentMode,
};

use std::ffi::c_void;
use std::os::raw::c_int;

/// Compatibility alias for callers that use the persistent module's
/// historical cache-position sentinel. Prefer
/// `Qwen36MoeAttnStepParams::CACHE_POS_INHERIT` in new code.
#[allow(dead_code)]
pub const CACHE_POS_INHERIT: i32 = Qwen36MoeAttnStepParams::CACHE_POS_INHERIT;

/// Phase 3f folded final RMSnorm + lm_head GEMV. Pass to
/// [`PersistentScratch::run`] on generation steps to write logits
/// directly from the megakernel; pass `None` on prefill steps.
pub struct LmHeadFold<'a> {
    pub final_norm_w: &'a GpuBuffer,
    pub lm_head_w: &'a GpuBuffer,
    pub logits_out: Option<&'a mut GpuBuffer>,
    pub top1_out: Option<&'a mut GpuBuffer>,
    pub vocab: i32,
}

use crate::qwen36_moe::decode::{
    ffn_workspace_floats, full_attn_score_workspace_floats, full_attn_workspace_floats,
    linear_attn_workspace_floats, reset_sync_buf, Qwen36ExecutionOptions,
};
use crate::qwen36_moe::layer_loader::{classify_layer_weight_encoding, Qwen36LayerWeightEncoding};
use crate::qwen36_moe::layers::{
    validate_persistent_embedding_request, validate_persistent_position_plan,
    PersistentEmbeddingMetadata, PersistentKvCapacity,
};
use crate::qwen36_moe::lm_head::bf16_bytes_to_f32;
use crate::qwen36_moe::types::{
    AttnLayerBuffers, DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom,
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
    geom: Qwen36MoePersistentGeom,
    num_layers: usize,
    layer_is_full_attention: Vec<bool>,
    full_attn_kv_capacities: Vec<PersistentKvCapacity>,
    /// `[num_layers]` descriptors uploaded as opaque U8 bytes.
    pub(crate) layer_descs_dev: GpuBuffer,
    /// `[num_layers]` INT4 sidecar descriptors. `None` for BF16 bakes.
    int4_scales_dev: Option<GpuBuffer>,
    /// Optional GPU upload of `Vec<Qwen36MoeKVCacheFp8Desc>` (one entry
    /// per layer). `Some` when KV-FP8 is active.
    kv_fp8_descs_dev: Option<GpuBuffer>,
    /// Two `[hidden]` BF16 buffers — kernel ping-pongs residuals through
    /// them. Per-step, host uploads the fresh `initial_hidden` into
    /// `hidden_ping`; the kernel returns the final hidden in
    /// `hidden_ping` (two swaps per layer cancel — see the kernel
    /// docstring for the math).
    pub(crate) hidden_ping: GpuBuffer,
    hidden_pong: GpuBuffer,
    /// F32 shared scratch sized for `max(full_attn, linear_attn, ffn)`
    /// workspace footprints.
    pub(crate) workspace: GpuBuffer,
    /// `[top_k]` i32. The full megakernel keeps this as internal FFN
    /// scratch; segmented sparse-VMM decode downloads it after router-only
    /// launches to know which expert pages to remap.
    ffn_topk_idx_scratch: GpuBuffer,
    /// 96-byte sync_buf: counters[0..16] + barrier_counter (+64) +
    /// barrier_flag (+68). Bridge zeros it on every launch.
    sync_buf: GpuBuffer,
}

fn persistent_supports_encoding(encoding: Qwen36LayerWeightEncoding) -> bool {
    matches!(
        encoding,
        Qwen36LayerWeightEncoding::NativeInt4 | Qwen36LayerWeightEncoding::Fp8
    )
}

fn validate_lm_head_fold(
    ordinal: usize,
    hidden: usize,
    backend: Backend,
    fold: &LmHeadFold<'_>,
) -> Result<()> {
    if fold.vocab <= 0 {
        return Err(anyhow!(
            "Qwen3.6 persistent lm-head fold requires positive vocab, got {}",
            fold.vocab
        ));
    }
    if fold.logits_out.is_some() == fold.top1_out.is_some() {
        return Err(anyhow!(
            "Qwen3.6 persistent lm-head fold requires exactly one logits or top1 output"
        ));
    }
    let hidden_bytes = hidden
        .checked_mul(2)
        .ok_or_else(|| anyhow!("Qwen3.6 persistent lm-head hidden size overflow"))?;
    let vocab = fold.vocab as usize;
    let lm_head_bytes = vocab
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow!("Qwen3.6 persistent lm-head weight size overflow"))?;
    let validate =
        |label: &str, buffer: &GpuBuffer, dtype: ScalarType, required_bytes: usize| -> Result<()> {
            if buffer.backend() != backend {
                return Err(anyhow!(
                    "{label} backend mismatch: got {:?}, expected {backend:?}",
                    buffer.backend()
                ));
            }
            if buffer.device_ordinal() != ordinal {
                return Err(anyhow!(
                    "{label} device ordinal mismatch: got {}, expected {ordinal}",
                    buffer.device_ordinal()
                ));
            }
            if buffer.dtype() != dtype {
                return Err(anyhow!(
                    "{label} dtype mismatch: got {:?}, expected {dtype:?}",
                    buffer.dtype()
                ));
            }
            if buffer.len_bytes() < required_bytes {
                return Err(anyhow!(
                    "{label} buffer too small: got {} bytes, need at least {required_bytes}",
                    buffer.len_bytes()
                ));
            }
            Ok(())
        };
    validate(
        "persistent final_norm",
        fold.final_norm_w,
        ScalarType::BF16,
        hidden_bytes,
    )?;
    validate(
        "persistent lm_head",
        fold.lm_head_w,
        ScalarType::BF16,
        lm_head_bytes,
    )?;
    if let Some(logits) = fold.logits_out.as_deref() {
        validate(
            "persistent logits_out",
            logits,
            ScalarType::BF16,
            vocab
                .checked_mul(2)
                .ok_or_else(|| anyhow!("Qwen3.6 persistent logits size overflow"))?,
        )?;
    }
    if let Some(top1) = fold.top1_out.as_deref() {
        validate(
            "persistent top1_out",
            top1,
            ScalarType::U32,
            std::mem::size_of::<u32>(),
        )?;
    }
    Ok(())
}

impl PersistentScratch {
    fn validate_device(&self, ordinal: usize) -> Result<()> {
        let active_backend = gpu_hal::current_backend();
        if self.hidden_ping.backend() != active_backend {
            return Err(anyhow!(
                "Qwen3.6 persistent scratch backend mismatch: active backend is \
                 {active_backend:?}, scratch backend is {:?}",
                self.hidden_ping.backend()
            ));
        }
        if self.hidden_ping.device_ordinal() != ordinal {
            return Err(anyhow!(
                "Qwen3.6 persistent scratch device ordinal mismatch: got {ordinal}, expected {}",
                self.hidden_ping.device_ordinal()
            ));
        }
        Ok(())
    }

    /// Build the descriptor array + allocate scratch. Mutably borrows
    /// `layers` only for descriptor construction (mutable state pointers
    /// are cached into the descs); subsequent [`Self::run`] calls don't
    /// need `&mut layers`.
    pub(crate) fn new(
        ordinal: usize,
        geom: &MultiLayerGeom,
        layers: &mut [LayerBuffers],
    ) -> Result<Self> {
        if geom.hidden <= 0 || geom.num_layers < 0 {
            return Err(anyhow!(
                "invalid Qwen3.6 persistent geometry: hidden={} layers={}",
                geom.hidden,
                geom.num_layers
            ));
        }
        let encoding = classify_layer_weight_encoding(layers)
            .context("classify Qwen3.6 persistent layer weight encoding")?;
        if !persistent_supports_encoding(encoding) {
            return Err(anyhow!(
                "Qwen3.6 persistent decode does not support {encoding:?} layer weights; \
                 use the chained decode path"
            ));
        }
        let num_layers = layers.len();
        let layer_is_full_attention = layers.iter().map(LayerBuffers::is_full_attn).collect();
        let full_attn_kv_capacities = layers
            .iter()
            .enumerate()
            .filter_map(|(layer_idx, layer)| match &layer.attn {
                AttnLayerBuffers::Full {
                    kv_cache: Some(cache),
                    ..
                } => Some(PersistentKvCapacity {
                    layer_idx,
                    capacity: cache.kv_max_t,
                }),
                _ => None,
            })
            .collect();
        let descs = build_layer_descs(layers);
        let layer_descs_dev =
            upload_descs(ordinal, &descs).context("upload layer descriptor array")?;
        let int4_scales_dev = match build_int4_descs(layers) {
            Some(int4) => Some(upload_descs(ordinal, &int4).context("upload int4 scale descs")?),
            None => None,
        };
        let kv_fp8_descs_dev = match build_kv_fp8_descs(layers) {
            Some(descs) => {
                Some(upload_descs(ordinal, &descs).context("upload kv_fp8 scale descs")?)
            }
            None => None,
        };

        let hidden = geom.hidden as usize;
        let hidden_ping = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .context("alloc persistent hidden_ping")?;
        let hidden_pong = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .context("alloc persistent hidden_pong")?;

        // KV-cache adds per-head score/partial workspace when any
        // full-attn layer carries a cache. Mirror the chained driver's
        // score-stride calc so the HIP tiled-attention path can use
        // tile-local online-softmax partials.
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
        let full_attn_ws =
            full_attn_workspace_floats(geom) + full_attn_score_workspace_floats(geom, max_kv_t);
        let ws_floats = full_attn_ws
            .max(linear_attn_workspace_floats(geom))
            .max(ffn_workspace_floats(geom));
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
            layer_is_full_attention,
            full_attn_kv_capacities,
            layer_descs_dev,
            int4_scales_dev,
            kv_fp8_descs_dev,
            hidden_ping,
            hidden_pong,
            workspace,
            ffn_topk_idx_scratch,
            sync_buf,
        })
    }

    /// One decode step. H2D the freshly-embedded `initial_hidden` into
    /// `hidden_ping`, run the megakernel, and optionally D2H the final
    /// hidden back.
    /// Mutates the linear-attn state in place (via the pointers cached
    /// in `layer_descs_dev`) — same semantics as
    /// `run_chained_decode_fast`.
    ///
    /// `lm_head_fold`: when `Some`, runs the folded final RMSnorm +
    /// lm_head GEMV phase (Phase 3f) at the tail of the megakernel,
    /// writing logits to `fold.logits_out`. The host can then D2H
    /// logits directly without a separate `lm_head_launch`. Pass
    /// `None` on prefill steps.
    pub fn run(
        &mut self,
        ordinal: usize,
        initial_hidden_bytes: &[u8],
        position: i32,
        // `-1` inherits from `position` (dense base decode). `>= 0`
        // decouples the KV slot from RoPE position; SpecPrefill sparse
        // prefill and MTP draft layers use that shape.
        cache_pos: i32,
        lm_head_fold: Option<LmHeadFold<'_>>,
        download_final_hidden: bool,
    ) -> Result<DecodeOutputs> {
        self.validate_device(ordinal)?;
        validate_persistent_position_plan(position, cache_pos, 1, &self.full_attn_kv_capacities)?;
        let hidden_bytes = self.geom.hidden as usize * 2;
        if initial_hidden_bytes.len() != hidden_bytes {
            return Err(anyhow!(
                "initial_hidden_bytes len {} != expected {} (hidden*2 BF16 bytes)",
                initial_hidden_bytes.len(),
                hidden_bytes,
            ));
        }
        if let Some(fold) = lm_head_fold.as_ref() {
            validate_lm_head_fold(
                ordinal,
                self.geom.hidden as usize,
                self.hidden_ping.backend(),
                fold,
            )?;
        }
        copy_h2d(
            ordinal,
            self.hidden_ping.as_mut_ptr(),
            initial_hidden_bytes.as_ptr() as *const _,
            hidden_bytes,
        )
        .context("h2d initial_hidden -> hidden_ping")?;

        let ffi_fold = lm_head_fold.map(|f| Qwen36MoePersistentLmHeadFold {
            final_norm_w: f.final_norm_w,
            lm_head_w: f.lm_head_w,
            logits_out: f.logits_out,
            top1_out: f.top1_out,
            vocab: f.vocab,
        });

        let t_launch = std::time::Instant::now();
        persistent_decode_launch(
            ordinal,
            ScalarType::BF16,
            self.geom,
            position,
            cache_pos,
            &self.layer_descs_dev,
            self.int4_scales_dev.as_ref(),
            self.kv_fp8_descs_dev.as_ref(),
            self.num_layers,
            &mut self.hidden_ping,
            &mut self.hidden_pong,
            &mut self.workspace,
            &mut self.ffn_topk_idx_scratch,
            &mut self.sync_buf,
            ffi_fold,
        )
        .map_err(|e: GpuError| anyhow!(e))
        .context("persistent_decode_launch")?;
        let elapsed_us = t_launch.elapsed().as_micros() as u64;

        // D2H the final hidden only when a downstream host consumer needs it.
        // Folded generation can consume logits/token from the same stream and
        // let that later transfer be the synchronization point.
        let final_hidden_bytes = if download_final_hidden {
            let mut bytes = vec![0u8; hidden_bytes];
            copy_d2h(
                ordinal,
                bytes.as_mut_ptr() as *mut _,
                self.hidden_ping.as_ptr(),
                hidden_bytes,
            )
            .context("d2h hidden_ping -> final_hidden_bytes")?;
            bytes
        } else {
            Vec::new()
        };

        // Stage attribution isn't recoverable inside one launch — we
        // lump the whole wall-clock into `kernel_full_attn_us` so
        // `--emit-stage-timings` still surfaces *something*. Per-phase
        // breakdowns require running through the chained path.
        Ok(DecodeOutputs {
            path_label: "persistent",
            final_hidden_bytes,
            per_layer_attn_out: Vec::new(),
            per_layer_ffn_out: Vec::new(),
            kernel_full_attn_us: elapsed_us,
            kernel_linear_attn_us: 0,
            kernel_ffn_us: 0,
            sparse_lookahead_prefetch_us: 0,
            sparse_router_launch_us: 0,
            sparse_route_d2h_us: 0,
            sparse_demand_prefetch_us: 0,
            sparse_ffn_launch_us: 0,
        })
    }

    /// Dense prefill fast path. The kernel loads `embed_tokens[token_id]`
    /// directly into `hidden_ping`, runs the persistent chain, and leaves the
    /// final hidden on device. Prompt-prefill callers do not consume
    /// `final_hidden_bytes`, so skipping the host embed lookup, H2D upload,
    /// and D2H download lets the default stream carry the state dependency.
    pub fn run_from_device_embedding_no_download(
        &mut self,
        ordinal: usize,
        embed_w: &GpuBuffer,
        token_id: u32,
        position: i32,
        cache_pos: i32,
    ) -> Result<std::time::Duration> {
        self.validate_device(ordinal)?;
        validate_persistent_position_plan(position, cache_pos, 1, &self.full_attn_kv_capacities)?;
        validate_persistent_embedding_request(
            self.hidden_ping.backend(),
            ordinal,
            self.geom.hidden as usize,
            PersistentEmbeddingMetadata {
                backend: embed_w.backend(),
                ordinal: embed_w.device_ordinal(),
                dtype: embed_w.dtype(),
                shape: embed_w.shape(),
                len_bytes: embed_w.len_bytes(),
            },
            &[token_id],
        )?;
        let t_launch = std::time::Instant::now();
        persistent_decode_launch_range(
            ordinal,
            ScalarType::BF16,
            self.geom,
            0,
            self.num_layers,
            Qwen36MoePersistentMode::Full,
            position,
            cache_pos,
            &self.layer_descs_dev,
            self.int4_scales_dev.as_ref(),
            self.kv_fp8_descs_dev.as_ref(),
            self.num_layers,
            &mut self.hidden_ping,
            &mut self.hidden_pong,
            &mut self.workspace,
            &mut self.ffn_topk_idx_scratch,
            &mut self.sync_buf,
            Some(embed_w),
            token_id as i32,
            None,
            1,
            None,
        )
        .map_err(|e: GpuError| anyhow!(e))
        .context("persistent_decode_launch from device embedding")?;
        Ok(t_launch.elapsed())
    }

    pub fn run_dense_prefill_tokens_from_device_embedding(
        &mut self,
        ordinal: usize,
        embed_w: &GpuBuffer,
        token_ids: &[u32],
        start_position: i32,
        start_cache_pos: i32,
    ) -> Result<std::time::Duration> {
        self.validate_device(ordinal)?;
        validate_persistent_position_plan(
            start_position,
            start_cache_pos,
            token_ids.len(),
            &self.full_attn_kv_capacities,
        )?;
        validate_persistent_embedding_request(
            self.hidden_ping.backend(),
            ordinal,
            self.geom.hidden as usize,
            PersistentEmbeddingMetadata {
                backend: embed_w.backend(),
                ordinal: embed_w.device_ordinal(),
                dtype: embed_w.dtype(),
                shape: embed_w.shape(),
                len_bytes: embed_w.len_bytes(),
            },
            token_ids,
        )?;
        if token_ids.is_empty() {
            return Ok(std::time::Duration::ZERO);
        }
        let token_bytes = unsafe {
            std::slice::from_raw_parts(
                token_ids.as_ptr() as *const u8,
                token_ids.len() * std::mem::size_of::<u32>(),
            )
        };
        let token_ids_dev =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[token_ids.len()], token_bytes)
                .context("upload dense prefill token ids")?;

        let t_launch = std::time::Instant::now();
        persistent_decode_launch_range(
            ordinal,
            ScalarType::BF16,
            self.geom,
            0,
            self.num_layers,
            Qwen36MoePersistentMode::Full,
            start_position,
            start_cache_pos,
            &self.layer_descs_dev,
            self.int4_scales_dev.as_ref(),
            self.kv_fp8_descs_dev.as_ref(),
            self.num_layers,
            &mut self.hidden_ping,
            &mut self.hidden_pong,
            &mut self.workspace,
            &mut self.ffn_topk_idx_scratch,
            &mut self.sync_buf,
            Some(embed_w),
            -1,
            Some(&token_ids_dev),
            token_ids.len() as i32,
            None,
        )
        .map_err(|e: GpuError| anyhow!(e))
        .context("persistent dense prefill token loop")?;
        sync(ordinal).context("sync persistent dense prefill token loop")?;
        Ok(t_launch.elapsed())
    }

    /// Sparse MoE residency variant. Each layer is split into:
    ///
    /// 1. attention/linear-attention + router top-k
    /// 2. host VMM remap for the routed experts
    /// 3. FFN completion for that layer
    ///
    /// This keeps the persistent phase bodies in use while preserving the
    /// host-side remap point required by HIP VMM. The folded lm_head phase is
    /// intentionally not run here; callers keep using the standalone
    /// `lm_head_launch` path after downloading `final_hidden_bytes`.
    pub fn run_sparse_with_expert_prefetch<F>(
        &mut self,
        ordinal: usize,
        initial_hidden_bytes: &[u8],
        position: i32,
        // See [`Self::run`] for cache_pos semantics. The two
        // segmented launches (RouterOnly + FfnOnly) share one
        // cache_pos — RoPE rotates at `position` regardless of which
        // mode is active, and the KV slot is only written by the
        // attn pre-router phase (slot = effective cache_pos).
        cache_pos: i32,
        mut prefetch: F,
    ) -> Result<DecodeOutputs>
    where
        F: FnMut(ExpertPrefetchPhase, usize, &[ExpertRoute]) -> Result<()>,
    {
        self.validate_device(ordinal)?;
        validate_persistent_position_plan(position, cache_pos, 1, &self.full_attn_kv_capacities)?;
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

        let mut router_launch_us = 0u64;
        let mut route_d2h_us = 0u64;
        let mut lookahead_us = 0u64;
        let mut demand_us = 0u64;
        let mut ffn_launch_us = 0u64;
        let mut router_us = 0u64;
        let mut ffn_us = 0u64;
        for layer_idx in 0..self.num_layers {
            let t_lookahead = std::time::Instant::now();
            prefetch(ExpertPrefetchPhase::Lookahead, layer_idx, &[]).with_context(|| {
                format!("lookahead prefetch routed experts (layer {layer_idx})")
            })?;
            let lookahead_elapsed = t_lookahead.elapsed().as_micros() as u64;
            lookahead_us = lookahead_us.saturating_add(lookahead_elapsed);
            ffn_us = ffn_us.saturating_add(lookahead_elapsed);

            let t_router_launch = std::time::Instant::now();
            persistent_decode_launch_range(
                ordinal,
                ScalarType::BF16,
                self.geom,
                layer_idx,
                layer_idx + 1,
                Qwen36MoePersistentMode::RouterOnly,
                position,
                cache_pos,
                &self.layer_descs_dev,
                self.int4_scales_dev.as_ref(),
                self.kv_fp8_descs_dev.as_ref(),
                self.num_layers,
                &mut self.hidden_ping,
                &mut self.hidden_pong,
                &mut self.workspace,
                &mut self.ffn_topk_idx_scratch,
                &mut self.sync_buf,
                None,
                -1,
                None,
                1,
                None,
            )
            .map_err(|e: GpuError| anyhow!(e))
            .with_context(|| format!("persistent router-only launch (layer {layer_idx})"))?;
            let router_elapsed = t_router_launch.elapsed().as_micros() as u64;
            router_launch_us = router_launch_us.saturating_add(router_elapsed);
            router_us = router_us.saturating_add(router_elapsed);

            let t_route_d2h = std::time::Instant::now();
            let routes = self
                .download_topk_routes(ordinal)
                .with_context(|| format!("download FFN top-k routes (layer {layer_idx})"))?;
            let route_d2h_elapsed = t_route_d2h.elapsed().as_micros() as u64;
            route_d2h_us = route_d2h_us.saturating_add(route_d2h_elapsed);
            router_us = router_us.saturating_add(route_d2h_elapsed);

            let t_demand = std::time::Instant::now();
            prefetch(ExpertPrefetchPhase::Demand, layer_idx, &routes)
                .with_context(|| format!("prefetch routed experts (layer {layer_idx})"))?;
            let demand_elapsed = t_demand.elapsed().as_micros() as u64;
            demand_us = demand_us.saturating_add(demand_elapsed);
            ffn_us = ffn_us.saturating_add(demand_elapsed);

            let t_ffn = std::time::Instant::now();
            persistent_decode_launch_range(
                ordinal,
                ScalarType::BF16,
                self.geom,
                layer_idx,
                layer_idx + 1,
                Qwen36MoePersistentMode::FfnOnly,
                position,
                cache_pos,
                &self.layer_descs_dev,
                self.int4_scales_dev.as_ref(),
                self.kv_fp8_descs_dev.as_ref(),
                self.num_layers,
                &mut self.hidden_ping,
                &mut self.hidden_pong,
                &mut self.workspace,
                &mut self.ffn_topk_idx_scratch,
                &mut self.sync_buf,
                None,
                -1,
                None,
                1,
                None,
            )
            .map_err(|e: GpuError| anyhow!(e))
            .with_context(|| format!("persistent ffn-only launch (layer {layer_idx})"))?;
            let ffn_elapsed = t_ffn.elapsed().as_micros() as u64;
            ffn_launch_us = ffn_launch_us.saturating_add(ffn_elapsed);
            ffn_us = ffn_us.saturating_add(ffn_elapsed);
        }

        let mut final_hidden_bytes = vec![0u8; hidden_bytes];
        copy_d2h(
            ordinal,
            final_hidden_bytes.as_mut_ptr() as *mut _,
            self.hidden_ping.as_ptr(),
            hidden_bytes,
        )
        .context("d2h hidden_ping -> final_hidden_bytes")?;

        Ok(DecodeOutputs {
            path_label: "persistent",
            final_hidden_bytes,
            per_layer_attn_out: Vec::new(),
            per_layer_ffn_out: Vec::new(),
            kernel_full_attn_us: router_us,
            kernel_linear_attn_us: 0,
            kernel_ffn_us: ffn_us,
            sparse_lookahead_prefetch_us: lookahead_us,
            sparse_router_launch_us: router_launch_us,
            sparse_route_d2h_us: route_d2h_us,
            sparse_demand_prefetch_us: demand_us,
            sparse_ffn_launch_us: ffn_launch_us,
        })
    }

    /// Profiling-only segmented decode. Splits every layer into attention-only
    /// and FFN-only persistent entry points, but skips VMM remap and route
    /// downloads. This is intentionally slower than [`Self::run`]; use it only
    /// under `SUPERSONIC_QWEN36_SEGMENTED_PROFILE=1` so rocprof can attribute
    /// the one-launch megakernel's layer halves.
    pub fn run_segmented_profile(
        &mut self,
        ordinal: usize,
        initial_hidden_bytes: &[u8],
        position: i32,
        cache_pos: i32,
        execution: &Qwen36ExecutionOptions,
    ) -> Result<DecodeOutputs> {
        self.validate_device(ordinal)?;
        validate_persistent_position_plan(position, cache_pos, 1, &self.full_attn_kv_capacities)?;
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

        let mut attn_us = 0u64;
        let mut ffn_us = 0u64;
        let ffn_stage_profile = execution.diagnostics.ffn_stage_profile;
        let linear_stage_profile = execution.diagnostics.linear_stage_profile;
        for layer_idx in 0..self.num_layers {
            if linear_stage_profile && !self.layer_is_full_attention[layer_idx] {
                for stage in 1..=5 {
                    reset_sync_buf(ordinal, &mut self.sync_buf).with_context(|| {
                        format!("reset sync_buf (segmented linear stage {stage} layer {layer_idx})")
                    })?;
                    let t_attn = std::time::Instant::now();
                    persistent_decode_launch_range(
                        ordinal,
                        ScalarType::BF16,
                        self.geom,
                        layer_idx,
                        layer_idx + 1,
                        Qwen36MoePersistentMode::LinearStage(stage),
                        position,
                        cache_pos,
                        &self.layer_descs_dev,
                        self.int4_scales_dev.as_ref(),
                        self.kv_fp8_descs_dev.as_ref(),
                        self.num_layers,
                        &mut self.hidden_ping,
                        &mut self.hidden_pong,
                        &mut self.workspace,
                        &mut self.ffn_topk_idx_scratch,
                        &mut self.sync_buf,
                        None,
                        -1,
                        None,
                        1,
                        None,
                    )
                    .map_err(|e: GpuError| anyhow!(e))
                    .with_context(|| {
                        format!(
                            "persistent segmented linear stage {stage} launch (layer {layer_idx})"
                        )
                    })?;
                    attn_us = attn_us.saturating_add(t_attn.elapsed().as_micros() as u64);
                }
            } else {
                reset_sync_buf(ordinal, &mut self.sync_buf).with_context(|| {
                    format!("reset sync_buf (segmented attention layer {layer_idx})")
                })?;
                let t_attn = std::time::Instant::now();
                persistent_decode_launch_range(
                    ordinal,
                    ScalarType::BF16,
                    self.geom,
                    layer_idx,
                    layer_idx + 1,
                    Qwen36MoePersistentMode::AttnOnly,
                    position,
                    cache_pos,
                    &self.layer_descs_dev,
                    self.int4_scales_dev.as_ref(),
                    self.kv_fp8_descs_dev.as_ref(),
                    self.num_layers,
                    &mut self.hidden_ping,
                    &mut self.hidden_pong,
                    &mut self.workspace,
                    &mut self.ffn_topk_idx_scratch,
                    &mut self.sync_buf,
                    None,
                    -1,
                    None,
                    1,
                    None,
                )
                .map_err(|e: GpuError| anyhow!(e))
                .with_context(|| {
                    format!("persistent segmented attention launch (layer {layer_idx})")
                })?;
                attn_us = attn_us.saturating_add(t_attn.elapsed().as_micros() as u64);
            }

            if ffn_stage_profile {
                // Profiling-only cumulative stages. Each launch recomputes
                // from post-attention hidden_pong; stage 5 is the final one
                // and writes the real layer output to hidden_ping.
                for stage in 1..=5 {
                    reset_sync_buf(ordinal, &mut self.sync_buf).with_context(|| {
                        format!("reset sync_buf (segmented ffn stage {stage} layer {layer_idx})")
                    })?;
                    let t_ffn = std::time::Instant::now();
                    persistent_decode_launch_range(
                        ordinal,
                        ScalarType::BF16,
                        self.geom,
                        layer_idx,
                        layer_idx + 1,
                        Qwen36MoePersistentMode::FfnStage(stage),
                        position,
                        cache_pos,
                        &self.layer_descs_dev,
                        self.int4_scales_dev.as_ref(),
                        self.kv_fp8_descs_dev.as_ref(),
                        self.num_layers,
                        &mut self.hidden_ping,
                        &mut self.hidden_pong,
                        &mut self.workspace,
                        &mut self.ffn_topk_idx_scratch,
                        &mut self.sync_buf,
                        None,
                        -1,
                        None,
                        1,
                        None,
                    )
                    .map_err(|e: GpuError| anyhow!(e))
                    .with_context(|| {
                        format!("persistent segmented ffn stage {stage} launch (layer {layer_idx})")
                    })?;
                    ffn_us = ffn_us.saturating_add(t_ffn.elapsed().as_micros() as u64);
                }
            } else {
                reset_sync_buf(ordinal, &mut self.sync_buf)
                    .with_context(|| format!("reset sync_buf (segmented ffn layer {layer_idx})"))?;
                let t_ffn = std::time::Instant::now();
                persistent_decode_launch_range(
                    ordinal,
                    ScalarType::BF16,
                    self.geom,
                    layer_idx,
                    layer_idx + 1,
                    Qwen36MoePersistentMode::FfnOnly,
                    position,
                    cache_pos,
                    &self.layer_descs_dev,
                    self.int4_scales_dev.as_ref(),
                    self.kv_fp8_descs_dev.as_ref(),
                    self.num_layers,
                    &mut self.hidden_ping,
                    &mut self.hidden_pong,
                    &mut self.workspace,
                    &mut self.ffn_topk_idx_scratch,
                    &mut self.sync_buf,
                    None,
                    -1,
                    None,
                    1,
                    None,
                )
                .map_err(|e: GpuError| anyhow!(e))
                .with_context(|| format!("persistent segmented ffn launch (layer {layer_idx})"))?;
                ffn_us = ffn_us.saturating_add(t_ffn.elapsed().as_micros() as u64);
            }
        }

        let mut final_hidden_bytes = vec![0u8; hidden_bytes];
        copy_d2h(
            ordinal,
            final_hidden_bytes.as_mut_ptr() as *mut _,
            self.hidden_ping.as_ptr(),
            hidden_bytes,
        )
        .context("d2h hidden_ping -> final_hidden_bytes")?;

        Ok(DecodeOutputs {
            path_label: if ffn_stage_profile {
                "persistent-segmented-ffn-stage-profile"
            } else if linear_stage_profile {
                "persistent-segmented-linear-stage-profile"
            } else {
                "persistent-segmented-profile"
            },
            final_hidden_bytes,
            per_layer_attn_out: Vec::new(),
            per_layer_ffn_out: Vec::new(),
            kernel_full_attn_us: attn_us,
            kernel_linear_attn_us: 0,
            kernel_ffn_us: ffn_us,
            sparse_lookahead_prefetch_us: 0,
            sparse_router_launch_us: attn_us,
            sparse_route_d2h_us: 0,
            sparse_demand_prefetch_us: 0,
            sparse_ffn_launch_us: ffn_us,
        })
    }

    fn download_topk_indices(&self, ordinal: usize) -> Result<Vec<usize>> {
        let top_k = self.geom.top_k as usize;
        let mut host = vec![0u32; top_k];
        copy_d2h(
            ordinal,
            host.as_mut_ptr() as *mut _,
            self.ffn_topk_idx_scratch.as_ptr(),
            top_k * std::mem::size_of::<u32>(),
        )
        .context("d2h ffn_topk_idx_scratch")?;
        Ok(host.into_iter().map(|idx| idx as usize).collect())
    }

    fn download_topk_routes(&self, ordinal: usize) -> Result<Vec<ExpertRoute>> {
        let top_k = self.geom.top_k as usize;
        let idx = self.download_topk_indices(ordinal)?;
        let mut weight_bytes = vec![0u8; top_k * std::mem::size_of::<u16>()];
        copy_d2h(
            ordinal,
            weight_bytes.as_mut_ptr() as *mut _,
            self.hidden_ping.as_ptr(),
            weight_bytes.len(),
        )
        .context("d2h ffn top-k route weights")?;
        let weights = bf16_bytes_to_f32(&weight_bytes);
        Ok(idx
            .into_iter()
            .zip(weights)
            .enumerate()
            .map(|(rank, (expert_idx, weight))| ExpertRoute {
                rank,
                expert_idx,
                weight,
            })
            .collect())
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
                    d.kv_cache_k = c.k_device_ptr();
                    d.kv_cache_v = c.v_device_ptr();
                    d.kv_max_t = c.kv_max_t;
                    d.kv_shadow_k = c
                        .kv_shadow_k
                        .as_mut()
                        .map(|b| b.as_mut_ptr())
                        .unwrap_or(std::ptr::null_mut());
                    d.kv_shadow_v = c
                        .kv_shadow_v
                        .as_mut()
                        .map(|b| b.as_mut_ptr())
                        .unwrap_or(std::ptr::null_mut());
                    d.kv_shadow_window = if c.kv_shadow_k.is_some() {
                        c.kv_shadow_window
                    } else {
                        0
                    };
                    // The descriptor is uploaded once. For rolling windows
                    // the kernel derives the active start each step from
                    // `position + 1 - kv_shadow_window`, with this field as
                    // a lower bound. A zero start preserves the previous
                    // full-sidecar behavior when window == kv_max_t.
                    d.kv_shadow_start = if c.kv_shadow_k.is_some() { 0 } else { -1 };
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

/// Build the parallel `Qwen36MoeKVCacheFp8Desc[num_layers]`. Returns
/// `None` when no layer carries FP8 KV scales (BF16 or INT4 bake
/// without KV-FP8). Linear-attn layers emit a zeroed descriptor
/// (null scale pointers); the kernel checks `is_full_attention != 0`
/// before dereferencing them.
pub fn build_kv_fp8_descs(layers: &mut [LayerBuffers]) -> Option<Vec<Qwen36MoeKVCacheFp8Desc>> {
    let any_fp8 = layers.iter().any(|l| match &l.attn {
        AttnLayerBuffers::Full {
            kv_cache: Some(c), ..
        } => c.kv_scale_k.is_some(),
        _ => false,
    });
    if !any_fp8 {
        return None;
    }
    let mut v = Vec::with_capacity(layers.len());
    for layer in layers.iter_mut() {
        let mut d = Qwen36MoeKVCacheFp8Desc::default();
        if let AttnLayerBuffers::Full {
            kv_cache: Some(c), ..
        } = &mut layer.attn
        {
            if let Some(sk) = c.kv_scale_k.as_mut() {
                d.kv_scale_k = sk.as_mut_ptr();
            }
            if let Some(sv) = c.kv_scale_v.as_mut() {
                d.kv_scale_v = sv.as_mut_ptr();
            }
        }
        v.push(d);
    }
    Some(v)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen36_moe::types::{FfnLayerBuffers, ResidentWeight};
    use gpu_hal::{Backend, VirtualAllocationRole, VirtualArena, VirtualBacking};

    #[test]
    fn persistent_mode_requires_supported_descriptor_encoding() {
        assert!(persistent_supports_encoding(
            Qwen36LayerWeightEncoding::NativeInt4
        ));
        assert!(persistent_supports_encoding(Qwen36LayerWeightEncoding::Fp8));
        assert!(!persistent_supports_encoding(
            Qwen36LayerWeightEncoding::Bf16
        ));
        assert!(!persistent_supports_encoding(
            Qwen36LayerWeightEncoding::GgmlKBlock
        ));
    }

    #[test]
    fn persistent_fold_rejects_malformed_buffers_before_launch() {
        let ordinal = 0;
        let final_norm =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc final norm");
        let lm_head =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2, 2]).expect("alloc wrong lm head");
        let mut logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc logits");
        let fold = LmHeadFold {
            final_norm_w: &final_norm,
            lm_head_w: &lm_head,
            logits_out: Some(&mut logits),
            top1_out: None,
            vocab: 2,
        };

        let err = validate_lm_head_fold(ordinal, 2, final_norm.backend(), &fold)
            .expect_err("wrong folded lm-head dtype must fail");

        assert!(err
            .to_string()
            .contains("persistent lm_head dtype mismatch"));
    }

    fn stub_bf16(ordinal: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1])?)
    }

    fn stub_u8(ordinal: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?)
    }

    #[test]
    fn layer_desc_uses_virtual_expert_weight_pointers() {
        let _backend_lock = crate::qwen36_moe::layer_loader::GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !gpu_hal::is_backend_compiled(Backend::Hip) {
            eprintln!("skip: HIP backend not compiled");
            return;
        }
        gpu_hal::set_backend(Backend::Hip);
        let ordinal = 0usize;
        if !gpu_hal::vmm_is_supported(Backend::Hip, ordinal) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let mut arena = VirtualArena::new(ordinal, VirtualBacking::CpuBackup);
        let gate_up_id = arena
            .reserve(
                "test.gate_up_proj",
                VirtualAllocationRole::MoeExpert,
                ScalarType::U8,
                &[4096],
            )
            .expect("reserve virtual gate_up");
        let down_id = arena
            .reserve(
                "test.down_proj",
                VirtualAllocationRole::MoeExpert,
                ScalarType::U8,
                &[4096],
            )
            .expect("reserve virtual down");
        let gate_up_buf = arena.allocation(gate_up_id).unwrap().buffer();
        let down_buf = arena.allocation(down_id).unwrap().buffer();
        let gate_up_ptr = gate_up_buf.as_ptr();
        let down_ptr = down_buf.as_ptr();

        let mut layers = vec![LayerBuffers {
            attn: AttnLayerBuffers::Linear {
                input_norm_w: stub_bf16(ordinal).unwrap(),
                in_proj_qkv_w: stub_u8(ordinal).unwrap(),
                in_proj_z_w: stub_u8(ordinal).unwrap(),
                in_proj_a_w: stub_bf16(ordinal).unwrap(),
                in_proj_b_w: stub_bf16(ordinal).unwrap(),
                conv1d_w: stub_bf16(ordinal).unwrap(),
                conv1d_bias: None,
                dt_bias: stub_bf16(ordinal).unwrap(),
                a_log: stub_bf16(ordinal).unwrap(),
                norm_w: stub_bf16(ordinal).unwrap(),
                out_proj_w: stub_u8(ordinal).unwrap(),
                conv_state: stub_bf16(ordinal).unwrap(),
                recurrent_state: GpuBuffer::zeros(ordinal, ScalarType::F32, &[1]).unwrap(),
                int4: None,
            },
            ffn: FfnLayerBuffers {
                post_attn_norm_w: stub_bf16(ordinal).unwrap(),
                gate_w: stub_bf16(ordinal).unwrap(),
                gate_up_proj_w: ResidentWeight::Virtual {
                    allocation_id: gate_up_id,
                    ptr: gate_up_ptr,
                    dtype: gate_up_buf.dtype(),
                    shape: gate_up_buf.shape().to_vec(),
                    len_bytes: gate_up_buf.len_bytes(),
                },
                down_proj_w: ResidentWeight::Virtual {
                    allocation_id: down_id,
                    ptr: down_ptr,
                    dtype: down_buf.dtype(),
                    shape: down_buf.shape().to_vec(),
                    len_bytes: down_buf.len_bytes(),
                },
                shared_gate_proj_w: stub_u8(ordinal).unwrap(),
                shared_up_proj_w: stub_u8(ordinal).unwrap(),
                shared_down_proj_w: stub_u8(ordinal).unwrap(),
                shared_expert_gate_w: stub_bf16(ordinal).unwrap(),
                int4: None,
            },
        }];

        let descs = build_layer_descs(&mut layers);
        assert_eq!(descs[0].experts_gate_up_w, gate_up_ptr);
        assert_eq!(descs[0].experts_down_w, down_ptr);
    }
}
