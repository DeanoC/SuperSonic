//! Host-side orchestrator for Qwen 3.6 MoE batched-Q prefill (Stage A).
//!
//! M6.2: chunks the prompt into N=PREFILL_CHUNK_SIZE tokens and per-layer
//! drives a batched primitive sequence (RMSnorm + INT4 projections +
//! per-head norms + RoPE + KV write + M3 batched attention + sigmoid_mul
//! gate + INT4 O-proj + residual) for full-attention layers. Linear
//! attention layers and the MoE FFN keep the per-token chained launchers
//! (their batched paths land in M9-M11 / a follow-up).
//!
//! When `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=1` is set the engine
//! routes prefill through `run_batched_prefill_stub` instead of running
//! the prefill iterations of its main per-step loop. This module owns
//! the *prefill range only* — the FIRST generation step (where
//! `step + 1 == effective_prompt_len` and logits are computed) is left
//! to the engine's main loop.
//!
//! Plan: docs/superpowers/plans/2026-05-05-qwen36-moe-batched-prefill-phase1.md
//!
//! ## Risks accepted at M6.2
//!
//! 1. **KV cache layout transpose:** the per-token persistent kernel
//!    stores K/V in `[t, Hkv, hd]`, while M3 expects `[B, Hkv, kv_len, hd]`.
//!    Each full-attn layer per chunk transposes the relevant prefix
//!    `[past_len + N, Hkv, hd]` -> `[Hkv, past_len + N, hd]` via
//!    `prefill_ffi::transpose_shd_hsd`. This is wasteful but keeps the
//!    decode megakernel happy (it reads the cache in the original
//!    layout for any subsequent decode steps).
//!
//! 2. **BF16 rounding boundaries differ** between the per-token kernel
//!    (which rounds at every intermediate) and the prefill primitives
//!    (which round at fewer points). Parity bar (cossim ≥ 0.9999) is
//!    expected to hold but is the most likely fail point.
//!
//! 3. **BF16 KV only:** if any full-attn layer carries an FP8 KV scale
//!    sidecar, this orchestrator refuses to run and falls back to the
//!    per-token path token by token. (FP8 KV write in batched-form is
//!    a follow-up.)

use std::ffi::c_void;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use gpu_hal::{copy_h2d, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi;
use kernel_ffi::qwen36_moe::{
    attn_step_launch, batched_prefill_attn_full_launch, ffn_step_launch, linear_step_launch,
    Qwen36MoeAttnStepInt4, Qwen36MoeAttnStepParams, Qwen36MoeAttnStepWeights,
    Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights, Qwen36MoeLinearStepInt4,
    Qwen36MoeLinearStepParams, Qwen36MoeLinearStepWeights,
};
use model_store::BakedStore;

use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::engine::current_position;
use crate::qwen36_moe_cli::host::lookup_embed_row;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_decode::{
    ffn_output_elems, ffn_workspace_floats, full_attn_output_elems, full_attn_workspace_floats,
    linear_attn_workspace_floats, reset_sync_buf,
};
use crate::qwen36_moe_persistent_decode::PersistentScratch;
use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_telemetry::MoeRouteRuntime;
use crate::qwen36_moe_types::{
    AttnLayerBuffers, FullAttnInt4Sidecars, FullAttnKvCache, LayerBuffers, MultiLayerGeom,
};

/// Number of prompt tokens per outer chunk. M6.2: this is the actual
/// `q_len` per M3 launch.
pub(crate) const PREFILL_CHUNK_SIZE: usize = 64;

/// Native INT4 quant_type code (matches qwen35::weights::LOWBIT_NATIVE_INT4).
const QUANT_TYPE_NATIVE_INT4: i32 = 4;

/// Aggregated timings for the batched-prefill orchestrator pass.
pub(crate) struct BatchedPrefillTimings {
    pub embed_total: Duration,
    pub chain_total: Duration,
    pub chunks: usize,
    pub tokens: usize,
}

/// CPU-built RoPE cos/sin tables uploaded once per orchestrator invocation.
struct RotaryTables {
    cos: GpuBuffer,
    sin: GpuBuffer,
}

impl RotaryTables {
    fn build(ordinal: usize, max_pos: usize, rotary_dim: usize, theta: f32) -> Result<Self> {
        let half = rotary_dim / 2;
        let mut cos_data: Vec<u8> = Vec::with_capacity(max_pos * half * 2);
        let mut sin_data: Vec<u8> = Vec::with_capacity(max_pos * half * 2);
        let theta = theta as f64;
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                let c = half::bf16::from_f64(angle.cos());
                let s = half::bf16::from_f64(angle.sin());
                cos_data.extend_from_slice(&c.to_le_bytes());
                sin_data.extend_from_slice(&s.to_le_bytes());
            }
        }
        let cos = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[max_pos, half],
            &cos_data,
        )
        .context("alloc rope cos")?;
        let sin = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[max_pos, half],
            &sin_data,
        )
        .context("alloc rope sin")?;
        Ok(Self { cos, sin })
    }
}

/// Scratch buffers reused across all 32 full-attn layers in a chunk.
/// Allocated once per chunk and reset implicitly by overwrite.
struct FullAttnBatchScratch {
    /// `[N, hidden]` BF16. Input + RMSnorm output.
    x_norm: GpuBuffer,
    /// `[N, 2*H*hd]` BF16. Concatenated q+gate per-head interleaved.
    qg_raw: GpuBuffer,
    /// `[N, H, hd]` BF16. q values extracted from qg_raw.
    q: GpuBuffer,
    /// `[N, H, hd]` BF16. gate values extracted from qg_raw.
    gate: GpuBuffer,
    /// `[N, Hkv*hd]` BF16. k_proj output (also q_norm input view as `[N*Hkv, hd]`).
    k: GpuBuffer,
    /// `[N, Hkv*hd]` BF16. v_proj output.
    v: GpuBuffer,
    /// `[H, N, hd]` BF16. q transposed for M3 input.
    q_thsd: GpuBuffer,
    /// `[Hkv, kv_max_t, hd]` BF16. K cache transposed for M3 input.
    /// (Allocated but currently unused — the live path mallocs a per-call
    /// `[hkv, kv_len, hd]` since `kv_len` varies per chunk; this slot is
    /// kept for the fixed-size path the next sub-task may switch to.)
    #[allow(dead_code)]
    k_thsd: GpuBuffer,
    /// `[Hkv, kv_max_t, hd]` BF16. V cache transposed for M3 input.
    #[allow(dead_code)]
    v_thsd: GpuBuffer,
    /// `[H, N, hd]` F32. M3 output.
    attn_out_f32: GpuBuffer,
    /// `[N, H, hd]` F32. M3 output transposed back to per-token layout.
    attn_out_nhd_f32: GpuBuffer,
    /// `[N, H*hd]` BF16. attn output cast back to BF16.
    attn_out_bf16: GpuBuffer,
    /// `[N, H, hd]` BF16. sigmoid_mul(gate) * attn_out.
    gated: GpuBuffer,
    /// `[N, hidden]` BF16. O-projection result.
    o: GpuBuffer,
    /// `[N, 2, H, hd]` view of qg_raw as `[N, H, 2, hd]` requires a stride
    /// kernel. We instead allocate a contiguous `[N, H, hd]` for q and gate
    /// each, populated by host loop + d2d strided copies. To keep the
    /// inner loop simple we use a single staging buffer here.
    _phantom: std::marker::PhantomData<()>,
}

impl FullAttnBatchScratch {
    fn alloc(ordinal: usize, geom: &MultiLayerGeom, n: usize, kv_max_t: usize) -> Result<Self> {
        let hidden = geom.hidden as usize;
        let h = geom.num_attention_heads as usize;
        let hkv = geom.num_kv_heads as usize;
        let hd = geom.head_dim as usize;
        let q_dim = 2 * h * hd;
        let kv_dim = hkv * hd;

        let x_norm = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc x_norm")?;
        let qg_raw = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, q_dim])
            .context("alloc qg_raw")?;
        let q = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd]).context("alloc q")?;
        let gate =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd]).context("alloc gate")?;
        let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, kv_dim]).context("alloc k")?;
        let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, kv_dim]).context("alloc v")?;
        let q_thsd = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[h, n, hd])
            .context("alloc q_thsd")?;
        let k_thsd = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hkv, kv_max_t.max(1), hd])
            .context("alloc k_thsd")?;
        let v_thsd = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hkv, kv_max_t.max(1), hd])
            .context("alloc v_thsd")?;
        let attn_out_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[h, n, hd])
            .context("alloc attn_out_f32")?;
        let attn_out_nhd_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[n, h, hd])
            .context("alloc attn_out_nhd_f32")?;
        let attn_out_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd])
            .context("alloc attn_out_bf16")?;
        let gated = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd])
            .context("alloc gated")?;
        let o =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden]).context("alloc o_proj")?;

        Ok(Self {
            x_norm,
            qg_raw,
            q,
            gate,
            k,
            v,
            q_thsd,
            k_thsd,
            v_thsd,
            attn_out_f32,
            attn_out_nhd_f32,
            attn_out_bf16,
            gated,
            o,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Drive the prefill range `[0, effective_prompt_len - 1)` through the
/// chunked batched-attn orchestrator. After return,
/// `loop_state.position == effective_prompt_len - 1` and
/// `loop_state.current_token` is the LAST prompt token (the gen-fold step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_batched_prefill_stub(
    ordinal: usize,
    geom: &MultiLayerGeom,
    store: &BakedStore,
    weight_prefix: &str,
    layers: &mut [LayerBuffers],
    persistent_scratch: Option<&mut PersistentScratch>,
    moe_expert_residency: Option<&mut MoeExpertResidencyManager>,
    moe_runtime: &mut MoeRuntimeConfig,
    moe_routes: &mut MoeRouteRuntime,
    loop_state: &mut Qwen36DecodeLoopState,
    prompt_ids: &[u32],
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    emit_stage_timings: bool,
) -> Result<BatchedPrefillTimings> {
    let prefill_count = effective_prompt_len.saturating_sub(1);
    let mut timings = BatchedPrefillTimings {
        embed_total: Duration::ZERO,
        chain_total: Duration::ZERO,
        chunks: 0,
        tokens: 0,
    };
    if prefill_count == 0 {
        return Ok(timings);
    }

    // Refuse the batched path on any non-dense or FP8-KV configuration —
    // those carry unique invariants not yet supported here. Fall back to
    // the per-token path for the entire prefill in that case.
    let supports_batched = supports_batched_path(layers, keep_mask);
    if !supports_batched {
        return run_pertoken_chunked(
            ordinal,
            geom,
            store,
            weight_prefix,
            layers,
            persistent_scratch,
            moe_expert_residency,
            moe_runtime,
            moe_routes,
            loop_state,
            prompt_ids,
            keep_mask,
            kept_positions,
            effective_prompt_len,
            emit_stage_timings,
        );
    }

    // Build RoPE tables once. Size by the largest plausible position
    // (`effective_prompt_len + max_new_tokens` worth of slots; here we
    // bound by the cache's `kv_max_t` to be safe).
    let max_kv_t = layers
        .iter()
        .filter_map(|l| match &l.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(c), ..
            } => Some(c.kv_max_t as usize),
            _ => None,
        })
        .max()
        .unwrap_or(1);
    let max_pos = max_kv_t.max(effective_prompt_len);
    let rotary = RotaryTables::build(
        ordinal,
        max_pos,
        geom.rotary_dim as usize,
        geom.rope_theta,
    )
    .context("build rotary tables")?;

    let chunk_n = PREFILL_CHUNK_SIZE.min(prefill_count);
    let mut scratch = FullAttnBatchScratch::alloc(ordinal, geom, chunk_n, max_kv_t)
        .context("alloc full-attn batched scratch")?;

    let full_prompt_len = prompt_ids.len();

    // Persistent kernel & MoE residency are mutated per-token via the
    // FFN/linear-attn fallbacks; pull them through the orchestrator.
    let mut persistent_scratch = persistent_scratch;
    let mut moe_expert_residency = moe_expert_residency;

    let mut step = 0usize;
    while step < prefill_count {
        let chunk_end = (step + PREFILL_CHUNK_SIZE).min(prefill_count);
        let n = chunk_end - step;
        timings.chunks += 1;

        let t_chunk = Instant::now();
        process_chunk_batched(
            ordinal,
            geom,
            store,
            weight_prefix,
            layers,
            persistent_scratch.as_deref_mut(),
            moe_expert_residency.as_deref_mut(),
            moe_runtime,
            moe_routes,
            loop_state,
            prompt_ids,
            keep_mask,
            kept_positions,
            effective_prompt_len,
            full_prompt_len,
            emit_stage_timings,
            step,
            n,
            &rotary,
            &mut scratch,
            &mut timings,
        )?;
        let _ = t_chunk;

        step = chunk_end;
    }

    Ok(timings)
}

/// Whether the layer stack supports the batched attn path.
/// Currently disqualifies FP8 KV (kv_scale_k Some) and SpecPrefill mode
/// (keep_mask Some). Both are deferrable to follow-ups.
fn supports_batched_path(layers: &[LayerBuffers], keep_mask: Option<&Vec<bool>>) -> bool {
    if keep_mask.is_some() {
        return false;
    }
    for l in layers {
        if let AttnLayerBuffers::Full {
            kv_cache: Some(c), ..
        } = &l.attn
        {
            if c.kv_scale_k.is_some() || c.kv_scale_v.is_some() {
                return false;
            }
        }
    }
    true
}

/// Per-chunk driver. Stages N tokens onto GPU as `[N, hidden]` BF16, then
/// loops layers. Full-attn → batched primitive sequence with M3.
/// Linear-attn + FFN → per-token fallback over the chunk.
///
/// The mode (real-batched vs per-token-stub) is gated by
/// `SUPERSONIC_QWEN36_MOE_BATCHED_ATTN`. Default (env unset / 0) keeps
/// the per-token chain step inside the chunk loop — preserves the M6.1
/// parity baseline while we develop the batched path. Set the env var
/// to 1 to engage the real batched primitive sequence.
#[allow(clippy::too_many_arguments)]
fn process_chunk_batched(
    ordinal: usize,
    geom: &MultiLayerGeom,
    store: &BakedStore,
    weight_prefix: &str,
    layers: &mut [LayerBuffers],
    mut persistent_scratch: Option<&mut PersistentScratch>,
    mut moe_expert_residency: Option<&mut MoeExpertResidencyManager>,
    moe_runtime: &mut MoeRuntimeConfig,
    moe_routes: &mut MoeRouteRuntime,
    loop_state: &mut Qwen36DecodeLoopState,
    prompt_ids: &[u32],
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    full_prompt_len: usize,
    emit_stage_timings: bool,
    chunk_start: usize,
    n: usize,
    rotary: &RotaryTables,
    scratch: &mut FullAttnBatchScratch,
    timings: &mut BatchedPrefillTimings,
) -> Result<()> {
    let use_batched_attn = std::env::var("SUPERSONIC_QWEN36_MOE_BATCHED_ATTN")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    if !use_batched_attn {
        // Default: per-token chain step inside the chunk. Mirrors the
        // engine main loop's prefill iterations exactly, so M1 parity
        // continues to pass while the M3 batched-attn path is being
        // brought up under SUPERSONIC_QWEN36_MOE_BATCHED_ATTN=1.
        let _ = (rotary, scratch);
        for inner in 0..n {
            let step = chunk_start + inner;
            let position = current_position(
                step,
                loop_state.position,
                keep_mask,
                kept_positions,
                effective_prompt_len,
                full_prompt_len,
            );
            let t0 = Instant::now();
            let initial_hidden = lookup_embed_row(
                store,
                weight_prefix,
                loop_state.current_token as usize,
                geom.hidden as usize,
            )
            .with_context(|| {
                format!(
                    "embed lookup token {} (batched prefill step {step})",
                    loop_state.current_token
                )
            })?;
            let t_embed_step = t0.elapsed();
            let scratch_arg: Option<&mut PersistentScratch> = persistent_scratch.as_deref_mut();
            let residency_arg: Option<&mut MoeExpertResidencyManager> =
                moe_expert_residency.as_deref_mut();
            let t1 = Instant::now();
            let _chain_step = run_chain_step(Qwen36ChainStep {
                ordinal,
                geom,
                store,
                layers,
                persistent_scratch: scratch_arg,
                moe_expert_residency: residency_arg,
                moe_runtime,
                moe_routes,
                initial_hidden: &initial_hidden,
                position,
                step,
                is_gen_step: false,
                emit_stage_timings,
                fold: None,
            })?;
            let t_chain_step = t1.elapsed();
            loop_state.position += 1;
            timings.embed_total += t_embed_step;
            timings.chain_total += t_chain_step;
            timings.tokens += 1;
            loop_state.current_token = prompt_ids[kept_positions[step + 1]];
        }
        return Ok(());
    }

    // Batched-attn enabled path: stage chunk on GPU and drive per-layer.
    let _ = (persistent_scratch, moe_expert_residency, moe_runtime, moe_routes);
    let _ = (emit_stage_timings, full_prompt_len, keep_mask);

    // 1. Stage the N chunk tokens onto the GPU.
    let hidden = geom.hidden as usize;
    let mut chunk_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
        .context("alloc chunk_hidden")?;
    let t0 = Instant::now();
    stage_chunk_tokens_on_gpu(
        ordinal,
        store,
        weight_prefix,
        geom,
        chunk_start,
        n,
        prompt_ids,
        kept_positions,
        loop_state,
        &mut chunk_hidden,
    )?;
    let t_embed = t0.elapsed();
    timings.embed_total += t_embed;

    // Per-token fallback workspaces (linear-attn + FFN paths still go
    // token-by-token at this stage; they will be batched in M9-M11).
    let attn_ws_floats = full_attn_workspace_floats(geom)
        .max(linear_attn_workspace_floats(geom))
        + geom.num_attention_heads as usize * pertoken_attn_extra_kv(layers);
    let attn_out_elems = full_attn_output_elems(geom);
    let mut attn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[attn_out_elems])
        .context("alloc attn_output")?;
    let mut attn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[attn_ws_floats])
        .context("alloc attn_workspace")?;
    let mut ffn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_output_elems(geom)])
        .context("alloc ffn_output")?;
    let mut ffn_output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
        .context("alloc ffn_output_idx")?;
    let mut ffn_workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_workspace_floats(geom)])
        .context("alloc ffn_workspace")?;
    let mut sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])
        .context("alloc sync_buf")?;

    let past_len_at_chunk_start = loop_state.position as usize;

    let t_chain = Instant::now();
    for layer_idx in 0..layers.len() {
        let is_full = layers[layer_idx].is_full_attn();
        if is_full {
            // Take exclusive borrow of layer for the batched path.
            let layer = &mut layers[layer_idx];
            let AttnLayerBuffers::Full {
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
                int4,
                kv_cache,
            } = &mut layer.attn
            else {
                unreachable!();
            };
            let int4 = int4
                .as_ref()
                .ok_or_else(|| anyhow!("layer {layer_idx} missing INT4 sidecars"))?;
            let kv_cache = kv_cache
                .as_mut()
                .ok_or_else(|| anyhow!("layer {layer_idx} missing KV cache"))?;
            process_full_attn_layer_batched(
                ordinal,
                geom,
                n,
                past_len_at_chunk_start,
                &mut chunk_hidden,
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
                int4,
                kv_cache,
                rotary,
                scratch,
            )
            .with_context(|| format!("batched full-attn layer {layer_idx}"))?;
        } else {
            // Linear-attn: per-token fallback.
            let layer = &mut layers[layer_idx];
            process_linear_attn_layer_pertoken(
                ordinal,
                geom,
                n,
                &mut chunk_hidden,
                &mut layer.attn,
                &mut attn_output,
                &mut attn_workspace,
                &mut sync_buf,
            )
            .with_context(|| format!("pertoken linear-attn layer {layer_idx}"))?;
        }
        // FFN: per-token fallback for every layer (Stage A scope).
        let layer = &layers[layer_idx];
        process_ffn_pertoken(
            ordinal,
            geom,
            n,
            &mut chunk_hidden,
            &layer.ffn,
            &mut ffn_output,
            &mut ffn_output_idx,
            &mut ffn_workspace,
            &mut sync_buf,
        )
        .with_context(|| format!("pertoken ffn layer {layer_idx}"))?;
    }
    let t_chain_elapsed = t_chain.elapsed();
    timings.chain_total += t_chain_elapsed;

    // Advance loop_state: position by N, and current_token to the
    // post-chunk first prompt token (the engine's main loop resumes at
    // step = chunk_end and needs the next kept token to fold).
    loop_state.position += n as i32;
    timings.tokens += n;
    let post_chunk_step = chunk_start + n;
    loop_state.current_token = prompt_ids[kept_positions[post_chunk_step]];

    Ok(())
}

/// Stage N consecutive chunk tokens contiguously on GPU as `[N, hidden]`
/// BF16. Reads `loop_state.current_token` for the first row, then
/// `kept_positions[chunk_start + 1 ..]` for subsequent rows. Does NOT
/// mutate loop_state — caller advances after the chunk is processed.
#[allow(clippy::too_many_arguments)]
fn stage_chunk_tokens_on_gpu(
    ordinal: usize,
    store: &BakedStore,
    weight_prefix: &str,
    geom: &MultiLayerGeom,
    chunk_start: usize,
    n: usize,
    prompt_ids: &[u32],
    kept_positions: &[usize],
    loop_state: &Qwen36DecodeLoopState,
    out: &mut GpuBuffer,
) -> Result<()> {
    let hidden = geom.hidden as usize;
    let row_bytes = hidden * 2;
    let mut staging = vec![0u8; n * row_bytes];
    for inner in 0..n {
        let token = if inner == 0 {
            loop_state.current_token as usize
        } else {
            let step = chunk_start + inner;
            prompt_ids[kept_positions[step]] as usize
        };
        let row = lookup_embed_row(store, weight_prefix, token, hidden)?;
        let dst_off = inner * row_bytes;
        staging[dst_off..dst_off + row_bytes].copy_from_slice(&row);
    }
    copy_h2d(
        ordinal,
        out.as_mut_ptr(),
        staging.as_ptr() as *const c_void,
        staging.len(),
    )
    .context("copy_h2d chunk hidden")?;
    Ok(())
}

/// Per-token attn workspace's `attn_extra` term: extra F32 floats the
/// per-token full-attn kernel needs when KV cache is enabled (one row of
/// scores per Q head per cache slot).
fn pertoken_attn_extra_kv(layers: &[LayerBuffers]) -> usize {
    layers
        .iter()
        .filter_map(|l| match &l.attn {
            AttnLayerBuffers::Full {
                kv_cache: Some(c), ..
            } => Some(c.kv_max_t as usize),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Per-token chunked fallback. Used when `supports_batched_path` returns
/// false (e.g. FP8 KV, SpecPrefill keep_mask). Identical semantics to the
/// engine main loop's prefill iterations.
#[allow(clippy::too_many_arguments)]
fn run_pertoken_chunked(
    ordinal: usize,
    geom: &MultiLayerGeom,
    store: &BakedStore,
    weight_prefix: &str,
    layers: &mut [LayerBuffers],
    persistent_scratch: Option<&mut PersistentScratch>,
    moe_expert_residency: Option<&mut MoeExpertResidencyManager>,
    moe_runtime: &mut MoeRuntimeConfig,
    moe_routes: &mut MoeRouteRuntime,
    loop_state: &mut Qwen36DecodeLoopState,
    prompt_ids: &[u32],
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    emit_stage_timings: bool,
) -> Result<BatchedPrefillTimings> {
    let prefill_count = effective_prompt_len.saturating_sub(1);
    let mut timings = BatchedPrefillTimings {
        embed_total: Duration::ZERO,
        chain_total: Duration::ZERO,
        chunks: 0,
        tokens: 0,
    };
    if prefill_count == 0 {
        return Ok(timings);
    }
    let mut persistent_scratch = persistent_scratch;
    let mut moe_expert_residency = moe_expert_residency;
    let full_prompt_len = prompt_ids.len();

    let mut step = 0usize;
    while step < prefill_count {
        let chunk_end = (step + PREFILL_CHUNK_SIZE).min(prefill_count);
        timings.chunks += 1;

        while step < chunk_end {
            let position = current_position(
                step,
                loop_state.position,
                keep_mask,
                kept_positions,
                effective_prompt_len,
                full_prompt_len,
            );
            let t0 = Instant::now();
            let initial_hidden = lookup_embed_row(
                store,
                weight_prefix,
                loop_state.current_token as usize,
                geom.hidden as usize,
            )
            .with_context(|| {
                format!(
                    "embed lookup token {} (batched prefill step {step})",
                    loop_state.current_token
                )
            })?;
            let t_embed_step = t0.elapsed();
            let scratch_arg = persistent_scratch.as_deref_mut();
            let residency_arg = moe_expert_residency.as_deref_mut();
            let t1 = Instant::now();
            let _chain = run_chain_step(Qwen36ChainStep {
                ordinal,
                geom,
                store,
                layers,
                persistent_scratch: scratch_arg,
                moe_expert_residency: residency_arg,
                moe_runtime,
                moe_routes,
                initial_hidden: &initial_hidden,
                position,
                step,
                is_gen_step: false,
                emit_stage_timings,
                fold: None,
            })?;
            let t_chain_step = t1.elapsed();
            loop_state.position += 1;
            timings.embed_total += t_embed_step;
            timings.chain_total += t_chain_step;
            timings.tokens += 1;
            loop_state.current_token = prompt_ids[kept_positions[step + 1]];
            step += 1;
        }
    }
    Ok(timings)
}

// ---------------------------------------------------------------------------
// Batched primitive sequence helpers.
// ---------------------------------------------------------------------------

/// Run the 11-step batched full-attn primitive sequence for one layer.
///
/// On entry: `chunk_hidden[N, hidden]` is the layer's input residual.
/// On exit: `chunk_hidden` has been updated in-place (residual sum).
/// `kv_cache` slots `[past_len, past_len + N)` get the new K/V written.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_full_attn_layer_batched(
    ordinal: usize,
    geom: &MultiLayerGeom,
    n: usize,
    past_len: usize,
    chunk_hidden: &mut GpuBuffer,
    input_norm_w: &GpuBuffer,
    q_proj_w: &GpuBuffer,
    k_proj_w: &GpuBuffer,
    v_proj_w: &GpuBuffer,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    o_proj_w: &GpuBuffer,
    int4: &FullAttnInt4Sidecars,
    kv_cache: &mut FullAttnKvCache,
    rotary: &RotaryTables,
    scratch: &mut FullAttnBatchScratch,
) -> Result<()> {
    let hidden = geom.hidden as usize;
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let hd = geom.head_dim as usize;
    let rotary_dim = geom.rotary_dim as usize;
    let kv_dim = hkv * hd;
    let q_dim = 2 * h * hd;
    let kv_max_t = kv_cache.kv_max_t as usize;
    let group_size = int4.group_size as usize;
    let qtype = QUANT_TYPE_NATIVE_INT4;
    let scale = 1.0f32 / (hd as f32).sqrt();

    // 1. RMSnorm rows (with add_unit_offset = HF (1+w) form).
    prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        n,
        hidden,
        geom.rms_norm_eps,
        chunk_hidden,
        input_norm_w,
        &mut scratch.x_norm,
    )
    .map_err(|e| anyhow!("rms_norm_rows input: {e}"))?;

    // 2. Q/K/V projections (INT4).
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        q_dim,
        hidden,
        &scratch.x_norm,
        q_proj_w,
        &int4.q_proj_scale,
        &int4.q_proj_zero,
        group_size,
        qtype,
        &mut scratch.qg_raw,
    )
    .map_err(|e| anyhow!("matmul q_proj: {e}"))?;
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        kv_dim,
        hidden,
        &scratch.x_norm,
        k_proj_w,
        &int4.k_proj_scale,
        &int4.k_proj_zero,
        group_size,
        qtype,
        &mut scratch.k,
    )
    .map_err(|e| anyhow!("matmul k_proj: {e}"))?;
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        kv_dim,
        hidden,
        &scratch.x_norm,
        v_proj_w,
        &int4.v_proj_scale,
        &int4.v_proj_zero,
        group_size,
        qtype,
        &mut scratch.v,
    )
    .map_err(|e| anyhow!("matmul v_proj: {e}"))?;

    // 3. Split q+gate. qg_raw layout per row: [h0_q[hd], h0_gate[hd],
    //    h1_q[hd], h1_gate[hd], ...] — interleaved per-head halves.
    //    Extract into contiguous q[N,H,hd] and gate[N,H,hd] via per-head
    //    d2d copies. (No batched primitive for this strided copy yet.)
    let row_bytes = hd * 2;
    for nn in 0..n {
        for hh in 0..h {
            let qg_row_off = (nn * q_dim + hh * 2 * hd) * 2;
            let q_off = (nn * h * hd + hh * hd) * 2;
            let gate_off = q_off; // same layout in `gate`
            // q half
            let src_q = scratch.qg_raw.offset_ptr(qg_row_off);
            let dst_q = unsafe { (scratch.q.as_mut_ptr() as *mut u8).add(q_off) as *mut c_void };
            gpu_hal::copy_d2d(ordinal, dst_q, src_q, row_bytes)
                .context("split q")?;
            // gate half
            let src_g = scratch.qg_raw.offset_ptr(qg_row_off + hd * 2);
            let dst_g =
                unsafe { (scratch.gate.as_mut_ptr() as *mut u8).add(gate_off) as *mut c_void };
            gpu_hal::copy_d2d(ordinal, dst_g, src_g, row_bytes).context("split gate")?;
        }
    }

    // 4. Per-head q_norm + k_norm (RMSnorm on rows of head_dim).
    let mut q_after = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n * h, hd])
        .context("alloc q_after_norm")?;
    prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        n * h,
        hd,
        geom.rms_norm_eps,
        &scratch.q,
        q_norm_w,
        &mut q_after,
    )
    .map_err(|e| anyhow!("rms_norm_rows q: {e}"))?;
    let mut k_after = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n * hkv, hd])
        .context("alloc k_after_norm")?;
    prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        n * hkv,
        hd,
        geom.rms_norm_eps,
        &scratch.k,
        k_norm_w,
        &mut k_after,
    )
    .map_err(|e| anyhow!("rms_norm_rows k: {e}"))?;

    // 5. RoPE on q and k. positions [past_len, past_len + n).
    prefill_ffi::apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        n,
        h,
        hd,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        past_len,
        &mut q_after,
    )
    .map_err(|e| anyhow!("rope q: {e}"))?;
    prefill_ffi::apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        n,
        hkv,
        hd,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        past_len,
        &mut k_after,
    )
    .map_err(|e| anyhow!("rope k: {e}"))?;

    // 6. KV cache write — append k_after, v to slots [past_len, past_len+n).
    //    Cache layout: [t, Hkv, hd] BF16. Source: k_after [n, Hkv, hd]
    //    contiguous. Dest: byte offset past_len * Hkv * hd * 2.
    let cache_k_ptr = kv_cache.k_device_ptr();
    let cache_v_ptr = kv_cache.v_device_ptr();
    let row_kv_bytes = kv_dim * 2;
    let cache_byte_off = past_len * row_kv_bytes;
    let copy_bytes = n * row_kv_bytes;
    unsafe {
        let dst_k = (cache_k_ptr as *mut u8).add(cache_byte_off) as *mut c_void;
        let dst_v = (cache_v_ptr as *mut u8).add(cache_byte_off) as *mut c_void;
        gpu_hal::copy_d2d(ordinal, dst_k, k_after.as_ptr(), copy_bytes)
            .context("kv cache K write")?;
        gpu_hal::copy_d2d(ordinal, dst_v, scratch.v.as_ptr(), copy_bytes)
            .context("kv cache V write")?;
    }

    // 7. Transpose q [n, h, hd] -> [h, n, hd] for M3 input.
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        n,
        h,
        hd,
        &q_after,
        &mut scratch.q_thsd,
    )
    .map_err(|e| anyhow!("transpose q s,h,d -> h,s,d: {e}"))?;

    //    Transpose KV cache prefix [past_len + n, hkv, hd] -> [hkv, past_len + n, hd].
    let kv_len = past_len + n;
    // We need to wrap the cache as a GpuBuffer view to call transpose. Do
    // a direct D2D into a re-shaped temporary. The transpose primitive
    // takes a `&GpuBuffer` so we materialize a temp view by allocating
    // and copying — wasteful but correct. (If profiling shows this as a
    // hotspot, add a raw-pointer transpose variant.)
    let mut kv_prefix_k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_len, hkv, hd])
        .context("alloc kv_prefix_k")?;
    let mut kv_prefix_v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_len, hkv, hd])
        .context("alloc kv_prefix_v")?;
    let kv_bytes = kv_len * row_kv_bytes;
    gpu_hal::copy_d2d(ordinal, kv_prefix_k.as_mut_ptr(), cache_k_ptr, kv_bytes)
        .context("kv prefix copy K")?;
    gpu_hal::copy_d2d(ordinal, kv_prefix_v.as_mut_ptr(), cache_v_ptr, kv_bytes)
        .context("kv prefix copy V")?;
    let mut k_thsd_prefix = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hkv, kv_len, hd])
        .context("alloc k_thsd_prefix")?;
    let mut v_thsd_prefix = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hkv, kv_len, hd])
        .context("alloc v_thsd_prefix")?;
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        kv_len,
        hkv,
        hd,
        &kv_prefix_k,
        &mut k_thsd_prefix,
    )
    .map_err(|e| anyhow!("transpose K: {e}"))?;
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        kv_len,
        hkv,
        hd,
        &kv_prefix_v,
        &mut v_thsd_prefix,
    )
    .map_err(|e| anyhow!("transpose V: {e}"))?;
    let _ = kv_max_t; // unused after transpose

    // 8. M3 batched-Q full-attention.
    batched_prefill_attn_full_launch(
        ordinal,
        1,        // batch
        h,
        hkv,
        n,        // q_len
        kv_len,
        hd,
        scale,
        past_len, // seqlen_offset
        &scratch.q_thsd,
        &k_thsd_prefix,
        &v_thsd_prefix,
        &mut scratch.attn_out_f32,
    )
    .map_err(|e| anyhow!("batched_prefill_attn_full_launch: {e}"))?;

    // 9. Transpose attn_out F32 [h, n, hd] -> [n, h, hd]. Do it via
    //    transpose_shd_hsd with s=h, h=n: it transposes [s, h, d] to
    //    [h, s, d], so passing s=h, h=n yields the desired [n, h, hd].
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::F32,
        h, // s
        n, // h
        hd,
        &scratch.attn_out_f32,
        &mut scratch.attn_out_nhd_f32,
    )
    .map_err(|e| anyhow!("transpose attn_out h,n,d -> n,h,d: {e}"))?;

    // 10. Cast F32 -> BF16.
    prefill_ffi::cast(
        ordinal,
        ScalarType::F32,
        ScalarType::BF16,
        n * h * hd,
        &scratch.attn_out_nhd_f32,
        &mut scratch.attn_out_bf16,
    )
    .map_err(|e| anyhow!("cast attn f32->bf16: {e}"))?;

    // 11. Apply attn_output_gate: gated = sigmoid(gate) * attn_out_bf16.
    //     `pfx_sigmoid_mul`: out = data * sigmoid(gate). Pass data=attn,
    //     gate=gate. Output BF16.
    prefill_ffi::sigmoid_mul(
        ordinal,
        ScalarType::BF16,
        n * h * hd,
        &scratch.attn_out_bf16,
        &scratch.gate,
        &mut scratch.gated,
    )
    .map_err(|e| anyhow!("sigmoid_mul: {e}"))?;

    // 12. O-projection (INT4).
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        hidden,
        h * hd,
        &scratch.gated,
        o_proj_w,
        &int4.o_proj_scale,
        &int4.o_proj_zero,
        group_size,
        qtype,
        &mut scratch.o,
    )
    .map_err(|e| anyhow!("matmul o_proj: {e}"))?;

    // 13. Residual add: chunk_hidden += o.
    prefill_ffi::element_add_inplace(
        ordinal,
        ScalarType::BF16,
        n * hidden,
        chunk_hidden,
        &scratch.o,
    )
    .map_err(|e| anyhow!("residual add: {e}"))?;

    Ok(())
}

/// Per-token linear-attn fallback for one layer over `n` tokens of the
/// chunk. Reads `chunk_hidden[t*hidden..]`, writes the residual sum
/// back to the same slot.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_linear_attn_layer_pertoken(
    ordinal: usize,
    geom: &MultiLayerGeom,
    n: usize,
    chunk_hidden: &mut GpuBuffer,
    layer: &mut AttnLayerBuffers,
    attn_output: &mut GpuBuffer,
    attn_workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<()> {
    let hidden = geom.hidden as usize;
    let row_bytes = hidden * 2;
    let AttnLayerBuffers::Linear {
        input_norm_w,
        in_proj_qkv_w,
        in_proj_z_w,
        in_proj_a_w,
        in_proj_b_w,
        conv1d_w,
        conv1d_bias,
        dt_bias,
        a_log,
        norm_w,
        out_proj_w,
        conv_state,
        recurrent_state,
        int4,
    } = layer
    else {
        return Err(anyhow!("expected Linear layer"));
    };
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
    let int4_ptrs = match int4 {
        Some(s) => {
            let fp8 = s.group_size < 0;
            Qwen36MoeLinearStepInt4 {
                group_size: s.group_size,
                in_proj_qkv_scale: s.in_proj_qkv_scale.as_ptr(),
                in_proj_qkv_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.in_proj_qkv_zero.as_ptr()
                },
                in_proj_z_scale: s.in_proj_z_scale.as_ptr(),
                in_proj_z_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.in_proj_z_zero.as_ptr()
                },
                out_proj_scale: s.out_proj_scale.as_ptr(),
                out_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.out_proj_zero.as_ptr()
                },
            }
        }
        None => Qwen36MoeLinearStepInt4::disabled(),
    };
    for t in 0..n {
        reset_sync_buf(ordinal, sync_buf).context("reset sync_buf (linear-attn pertoken)")?;
        let token_byte_off = t * row_bytes;
        let input_ptr = chunk_hidden.offset_ptr(token_byte_off);
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_ptr,
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
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
            &weights,
            &int4_ptrs,
            attn_output,
            attn_workspace,
            sync_buf,
        )
        .with_context(|| format!("linear_step_launch (pertoken t={t})"))?;
        // Copy attn_output[..hidden] back into chunk_hidden[t]
        let dst = unsafe {
            (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void
        };
        gpu_hal::copy_d2d(ordinal, dst, attn_output.as_ptr(), row_bytes)
            .context("d2d linear-attn output -> chunk row")?;
    }
    Ok(())
}

/// Per-token full-attn fallback for one layer over `n` tokens of the
/// chunk. Used when the batched path is disabled (e.g. parity bisect).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_full_attn_layer_pertoken(
    ordinal: usize,
    geom: &MultiLayerGeom,
    n: usize,
    past_len_at_chunk_start: usize,
    chunk_hidden: &mut GpuBuffer,
    layer: &mut AttnLayerBuffers,
    attn_output: &mut GpuBuffer,
    attn_workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<()> {
    let hidden = geom.hidden as usize;
    let row_bytes = hidden * 2;
    let AttnLayerBuffers::Full {
        input_norm_w,
        q_proj_w,
        k_proj_w,
        v_proj_w,
        q_norm_w,
        k_norm_w,
        o_proj_w,
        int4,
        kv_cache,
    } = layer
    else {
        return Err(anyhow!("expected Full layer"));
    };
    let (kv_k_ptr, kv_v_ptr, kv_max_t) = match kv_cache {
        Some(c) => (c.k_device_ptr(), c.v_device_ptr(), c.kv_max_t),
        None => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
    };
    let int4_ptrs = match int4 {
        Some(s) => {
            let fp8 = s.group_size < 0;
            Qwen36MoeAttnStepInt4 {
                group_size: s.group_size,
                q_proj_scale: s.q_proj_scale.as_ptr(),
                q_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.q_proj_zero.as_ptr()
                },
                k_proj_scale: s.k_proj_scale.as_ptr(),
                k_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.k_proj_zero.as_ptr()
                },
                v_proj_scale: s.v_proj_scale.as_ptr(),
                v_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.v_proj_zero.as_ptr()
                },
                o_proj_scale: s.o_proj_scale.as_ptr(),
                o_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.o_proj_zero.as_ptr()
                },
            }
        }
        None => Qwen36MoeAttnStepInt4::disabled(),
    };
    for t in 0..n {
        reset_sync_buf(ordinal, sync_buf).context("reset sync_buf (full-attn pertoken)")?;
        let token_byte_off = t * row_bytes;
        let input_ptr = chunk_hidden.offset_ptr(token_byte_off);
        let position = (past_len_at_chunk_start + t) as i32;
        let params = Qwen36MoeAttnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_heads: geom.num_attention_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: geom.rotary_dim,
            rope_theta: geom.rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weights = Qwen36MoeAttnStepWeights {
            input_hidden: input_ptr,
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: o_proj_w.as_ptr(),
            kv_cache_k: kv_k_ptr,
            kv_cache_v: kv_v_ptr,
            kv_max_t,
        };
        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &int4_ptrs,
            attn_output,
            attn_workspace,
            sync_buf,
        )
        .with_context(|| format!("attn_step_launch (pertoken t={t})"))?;
        let dst = unsafe {
            (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void
        };
        gpu_hal::copy_d2d(ordinal, dst, attn_output.as_ptr(), row_bytes)
            .context("d2d full-attn output -> chunk row")?;
    }
    Ok(())
}

/// Per-token MoE FFN fallback for one layer over `n` tokens.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_ffn_pertoken(
    ordinal: usize,
    geom: &MultiLayerGeom,
    n: usize,
    chunk_hidden: &mut GpuBuffer,
    ffn: &crate::qwen36_moe_types::FfnLayerBuffers,
    ffn_output: &mut GpuBuffer,
    ffn_output_idx: &mut GpuBuffer,
    ffn_workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<()> {
    let hidden = geom.hidden as usize;
    let row_bytes = hidden * 2;
    let params = Qwen36MoeFfnStepParams {
        stage: 5,
        hidden: geom.hidden,
        num_experts: geom.num_experts,
        moe_intermediate: geom.moe_intermediate,
        shared_intermediate: geom.shared_intermediate,
        top_k: geom.top_k,
        rms_norm_eps: geom.rms_norm_eps,
    };
    let int4_ptrs = match &ffn.int4 {
        Some(s) => {
            let fp8 = s.group_size < 0;
            Qwen36MoeFfnStepInt4 {
                group_size: s.group_size,
                gate_up_proj_scale: s.gate_up_proj_scale.as_ptr(),
                gate_up_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.gate_up_proj_zero.as_ptr()
                },
                down_proj_scale: s.down_proj_scale.as_ptr(),
                down_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.down_proj_zero.as_ptr()
                },
                shared_gate_proj_scale: s.shared_gate_proj_scale.as_ptr(),
                shared_gate_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.shared_gate_proj_zero.as_ptr()
                },
                shared_up_proj_scale: s.shared_up_proj_scale.as_ptr(),
                shared_up_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.shared_up_proj_zero.as_ptr()
                },
                shared_down_proj_scale: s.shared_down_proj_scale.as_ptr(),
                shared_down_proj_zero: if fp8 {
                    std::ptr::null()
                } else {
                    s.shared_down_proj_zero.as_ptr()
                },
            }
        }
        None => Qwen36MoeFfnStepInt4::disabled(),
    };
    for t in 0..n {
        reset_sync_buf(ordinal, sync_buf).context("reset sync_buf (ffn pertoken)")?;
        let token_byte_off = t * row_bytes;
        let input_ptr = chunk_hidden.offset_ptr(token_byte_off);
        let weights = Qwen36MoeFfnStepWeights {
            input_hidden: input_ptr,
            post_attn_norm_w: ffn.post_attn_norm_w.as_ptr(),
            gate_w: ffn.gate_w.as_ptr(),
            gate_up_proj_w: ffn.gate_up_proj_w.as_ptr(),
            down_proj_w: ffn.down_proj_w.as_ptr(),
            shared_gate_proj_w: ffn.shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: ffn.shared_up_proj_w.as_ptr(),
            shared_down_proj_w: ffn.shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: ffn.shared_expert_gate_w.as_ptr(),
        };
        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &int4_ptrs,
            ffn_output,
            ffn_output_idx,
            ffn_workspace,
            sync_buf,
        )
        .with_context(|| format!("ffn_step_launch (pertoken t={t})"))?;
        let dst = unsafe {
            (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void
        };
        gpu_hal::copy_d2d(ordinal, dst, ffn_output.as_ptr(), row_bytes)
            .context("d2d ffn output -> chunk row")?;
    }
    Ok(())
}

