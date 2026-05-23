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
use gpu_hal::{copy_d2h, copy_h2d, Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi;
use kernel_ffi::qwen36_moe::{
    attn_step_launch, batched_prefill_attn_full_launch, batched_prefill_grouped_expert_launch_raw,
    batched_prefill_router_permute_launch, batched_prefill_unpermute_combine_launch,
    ffn_step_launch, linear_step_launch, Qwen36MoeAttnStepInt4, Qwen36MoeAttnStepParams,
    Qwen36MoeAttnStepWeights, Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams,
    Qwen36MoeFfnStepWeights, Qwen36MoeLinearStepInt4, Qwen36MoeLinearStepParams,
    Qwen36MoeLinearStepWeights,
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

/// Chunk-size policy.
///
/// **WMMA-100% threshold**: the M10 grouped MoE GEMM's per-expert WMMA tile
/// width is 16 rows. With `top_k=8` and `num_experts=256` the avg
/// tokens-per-expert hits that threshold at chunk_size = `16 * 256 / 8 = 512`.
/// Larger chunks don't gain WMMA utilization, only marginal K-tile-share in
/// attention (diminishing returns past ~256 because GEMM bandwidth dominates).
///
/// **Policy** (`pick_chunk_size`):
/// 1. Use the WMMA-100% size (`512`) for the bulk of long prompts.
/// 2. For the trailing partial chunk where `remaining < 512`, use exactly
///    `remaining` in one call — splitting into multiple smaller chunks just
///    adds launch overhead with no perf benefit.
/// 3. For very-short prompts (prompt_len ≤ 512), use one chunk = `prompt_len`.
/// 4. `LARGE = 1024` is reserved as the scratch-allocation ceiling so the
///    kernels can grow in the future without re-tuning the buffer size; the
///    runtime dispatch never picks larger than `WMMA_FULL` today.
pub(crate) const PREFILL_CHUNK_SIZE_WMMA_FULL: usize = 512;
pub(crate) const PREFILL_CHUNK_SIZE_MAX: usize = 1024;

/// Pick the chunk size for this iteration based on the remaining prefill
/// tokens. See policy comment on `PREFILL_CHUNK_SIZE_WMMA_FULL` above.
fn pick_chunk_size(remaining: usize) -> usize {
    if remaining >= PREFILL_CHUNK_SIZE_WMMA_FULL {
        PREFILL_CHUNK_SIZE_WMMA_FULL
    } else {
        // Trailing partial OR short-prompt single chunk: process whatever's
        // left in one call. Sub-WMMA-threshold but a single launch wins
        // over multiple sub-256 chunks (each of which would also miss the
        // WMMA threshold).
        remaining
    }
}

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
        let cos =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half], &cos_data)
                .context("alloc rope cos")?;
        let sin =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half], &sin_data)
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

        let x_norm =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden]).context("alloc x_norm")?;
        let qg_raw =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, q_dim]).context("alloc qg_raw")?;
        let q = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd]).context("alloc q")?;
        let gate =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd]).context("alloc gate")?;
        let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, kv_dim]).context("alloc k")?;
        let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, kv_dim]).context("alloc v")?;
        let q_thsd =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[h, n, hd]).context("alloc q_thsd")?;
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
        let gated =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, h, hd]).context("alloc gated")?;
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

    // Refuse the batched path on FP8-KV / FP8-runtime / sparse-residency
    // configurations. SpecPrefill sparse prompts are supported by staging
    // only kept tokens and applying RoPE through an indirect position-id
    // buffer, so they can use the same compact KV timeline as dense chunks.
    let supports_batched = supports_batched_path(layers, keep_mask, moe_expert_residency.is_some());
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

    // Build RoPE tables once. Dense chunks only need compact KV slots;
    // sparse SpecPrefill chunks need the original prompt positions too.
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
    let full_prompt_len = prompt_ids.len();
    let max_pos = max_kv_t.max(full_prompt_len).max(effective_prompt_len);
    let rotary = RotaryTables::build(ordinal, max_pos, geom.rotary_dim as usize, geom.rope_theta)
        .context("build rotary tables")?;

    // Allocate scratch sized for the LARGEST chunk we might use; per-chunk
    // code uses only the `n` prefix needed. At hidden=2048 hd=256 H=16 and
    // MAX=1024 the heaviest buffer is q_thsd [H, N, hd] = 8 MB — fine for
    // the 24 GiB target.
    let scratch_n = PREFILL_CHUNK_SIZE_MAX.min(prefill_count.max(1));
    let mut scratch = FullAttnBatchScratch::alloc(ordinal, geom, scratch_n, max_kv_t)
        .context("alloc full-attn batched scratch (max chunk)")?;

    // Persistent kernel & MoE residency are mutated per-token via the
    // FFN/linear-attn fallbacks; pull them through the orchestrator.
    let mut persistent_scratch = persistent_scratch;
    let mut moe_expert_residency = moe_expert_residency;

    let mut step = 0usize;
    while step < prefill_count {
        // Runtime dispatch among compile-time chunk-size options. picked is
        // one of {64, 256, 1024} or `remaining` for the trailing partial.
        // Each kernel accepts `n_tokens` as a runtime arg so we just pass
        // the actual chunk size; the dispatch pre-shapes scratch usage.
        let remaining = prefill_count - step;
        let n = pick_chunk_size(remaining).min(remaining);
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

        step += n;
    }

    Ok(timings)
}

/// Whether the layer stack supports the batched attn + grouped FFN path.
///
/// Refuses (falls through to per-token):
/// - **FP8 KV cache** (`kv_scale_k.is_some()` on any layer) — the batched
///   KV cache write path is BF16-only; FP8 sidecar quantize-on-write isn't
///   wired here.
/// - **FP8-runtime weights** (`int4.group_size < 0` on any layer) — the
///   negative-group-size sentinel signals FP8 weight sidecars (see
///   `kernels/qwen36_moe_persistent/ffn_phase.cuh:109`); the new grouped
///   FFN GEMM only handles INT4 with positive group_size.
/// - **Sparse-VMM MoE residency** (caller-provided `moe_expert_residency`
///   is `Some`) — sparse mode only RESERVES expert weight tensors upfront;
///   per-step `handle_moe_expert_prefetch` must run via `run_chain_step`
///   before FFN compute to page in any non-resident routed experts. The
///   batched FFN bypasses that hook, so it would read non-resident memory.
fn supports_batched_path(
    layers: &[LayerBuffers],
    keep_mask: Option<&Vec<bool>>,
    moe_expert_residency_active: bool,
) -> bool {
    let _ = keep_mask;
    if gpu_hal::current_backend() == Backend::Metal
        && std::env::var_os("SUPERSONIC_QWEN36_MOE_METAL_BATCHED_PREFILL_PROTOTYPE").is_none()
    {
        return false;
    }
    if moe_expert_residency_active {
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
        if let Some(int4) = &l.ffn.int4 {
            // Negative group_size = FP8 weight sidecar mode (see
            // kernels/qwen36_moe_persistent/ffn_phase.cuh:109).
            if int4.group_size < 0 {
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
    // M13: batched M3 attention is the DEFAULT inside the orchestrator.
    // Set SUPERSONIC_QWEN36_MOE_BATCHED_ATTN=0 to fall back to the
    // per-token chain step inside the chunk (kept as a bisect/escape
    // hatch — the orchestrator still runs but each chunk's tokens go
    // through the existing per-token persistent megakernel one at a
    // time, so the chunking adds no perf benefit, just structure).
    let batched_attn_disabled = std::env::var("SUPERSONIC_QWEN36_MOE_BATCHED_ATTN")
        .map(|v| v == "0")
        .unwrap_or(false);

    if batched_attn_disabled {
        // Per-token chain step inside the chunk. Mirrors the engine main
        // loop's prefill iterations exactly, so M1 parity continues to
        // pass while bisecting against the M3 batched-attn path.
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
    let _ = (
        persistent_scratch,
        moe_expert_residency,
        moe_runtime,
        moe_routes,
    );
    let _ = (emit_stage_timings, full_prompt_len);

    // 1. Stage the N chunk tokens onto the GPU.
    let hidden = geom.hidden as usize;
    let mut chunk_hidden =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden]).context("alloc chunk_hidden")?;
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
    let pos_ids = if keep_mask.is_some() {
        let mut pos_ids_host = Vec::with_capacity(n);
        for slot in 0..n {
            pos_ids_host.push(kept_positions[chunk_start + slot] as u32);
        }
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(pos_ids_host.as_ptr() as *const u8, pos_ids_host.len() * 4)
        };
        Some(
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[n], pos_bytes)
                .context("upload sparse prefill pos_ids")?,
        )
    } else {
        None
    };

    // M13: grouped MoE FFN is the DEFAULT once we're on the batched-attn
    // path. Set SUPERSONIC_QWEN36_MOE_GROUPED_FFN=0 to fall back to per-token
    // FFN inside the chunk while keeping the batched attention.
    let grouped_ffn_disabled = std::env::var("SUPERSONIC_QWEN36_MOE_GROUPED_FFN")
        .map(|v| v == "0")
        .unwrap_or(false);
    let use_grouped_ffn = !grouped_ffn_disabled;

    // Per-token fallback workspaces. Always allocated — used either by
    // linear-attn (always) or by per-token FFN (when grouped FFN is off).
    let attn_ws_floats = full_attn_workspace_floats(geom).max(linear_attn_workspace_floats(geom))
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
    let mut ffn_workspace =
        GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_workspace_floats(geom)])
            .context("alloc ffn_workspace")?;
    let mut sync_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).context("alloc sync_buf")?;

    // M11 batched-FFN scratch — allocated once per chunk, reused per layer.
    let mut grouped_scratch = if use_grouped_ffn {
        Some(GroupedFfnScratch::alloc(ordinal, geom, n)?)
    } else {
        None
    };

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
                pos_ids.as_ref(),
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
        // FFN: M11 batched grouped path (default) or per-token fallback.
        let layer = &layers[layer_idx];
        if let Some(scratch) = grouped_scratch.as_mut() {
            process_ffn_batched_grouped(
                ordinal,
                geom,
                n,
                &mut chunk_hidden,
                &layer.ffn,
                scratch,
                &mut sync_buf,
            )
            .with_context(|| format!("batched grouped ffn layer {layer_idx}"))?;
        } else {
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
        let remaining = prefill_count - step;
        let chunk_end = step + pick_chunk_size(remaining).min(remaining);
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
    pos_ids: Option<&GpuBuffer>,
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
        None,
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
        None,
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
        None,
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
            gpu_hal::copy_d2d(ordinal, dst_q, src_q, row_bytes).context("split q")?;
            // gate half
            let src_g = scratch.qg_raw.offset_ptr(qg_row_off + hd * 2);
            let dst_g =
                unsafe { (scratch.gate.as_mut_ptr() as *mut u8).add(gate_off) as *mut c_void };
            gpu_hal::copy_d2d(ordinal, dst_g, src_g, row_bytes).context("split gate")?;
        }
    }

    // 4. Per-head q_norm + k_norm (RMSnorm on rows of head_dim).
    let mut q_after =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n * h, hd]).context("alloc q_after_norm")?;
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

    // 5. RoPE on q and k. Dense uses compact positions
    // [past_len, past_len + n). Sparse SpecPrefill uses the original
    // prompt positions for rotation while still writing compact cache slots.
    if let Some(pos_ids) = pos_ids {
        prefill_ffi::apply_rope_prefill_indirect(
            ordinal,
            ScalarType::BF16,
            n,
            h,
            hd,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_ids,
            &mut q_after,
        )
        .map_err(|e| anyhow!("rope q indirect: {e}"))?;
        prefill_ffi::apply_rope_prefill_indirect(
            ordinal,
            ScalarType::BF16,
            n,
            hkv,
            hd,
            rotary_dim,
            &rotary.cos,
            &rotary.sin,
            pos_ids,
            &mut k_after,
        )
        .map_err(|e| anyhow!("rope k indirect: {e}"))?;
    } else {
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
    }

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
        1, // batch
        h,
        hkv,
        n, // q_len
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
        None,
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
        let dst =
            unsafe { (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void };
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
        let dst =
            unsafe { (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void };
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
        let dst =
            unsafe { (chunk_hidden.as_mut_ptr() as *mut u8).add(token_byte_off) as *mut c_void };
        gpu_hal::copy_d2d(ordinal, dst, ffn_output.as_ptr(), row_bytes)
            .context("d2d ffn output -> chunk row")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// M11 batched grouped MoE FFN.
// ---------------------------------------------------------------------------

/// Scratch buffers reused across all 40 MoE FFN layers in a chunk.
///
/// All sized at `chunk_n` tokens (or `chunk_n * top_k` for the permutation
/// arrays / expert outputs). Allocated once per chunk by `process_chunk_batched`
/// and overwritten per layer, so allocation cost is amortised.
struct GroupedFfnScratch {
    n: usize,
    top_k: usize,
    num_experts: usize,
    /// `[N, hidden]` BF16 — post-attention RMSnorm output.
    h_norm: GpuBuffer,
    /// `[N, num_experts]` BF16 — router logits. Caller treats this as
    /// "BF16-rounded F32" to match the per-token kernel's `bf16_round_rne_f32`
    /// store at the end of Phase B (router matvec). Softmax then runs on
    /// the BF16-widened-to-F32 values to match Phase C.
    router_logits_bf16: GpuBuffer,
    /// `[N, top_k]` i32 (U32 storage) — top-K expert indices.
    topk_idx: GpuBuffer,
    /// `[N, top_k]` BF16 — top-K renormalised weights.
    topk_weight: GpuBuffer,
    /// `[num_experts + 1]` i32 (U32 storage) — M9 prefix sum.
    expert_offsets: GpuBuffer,
    /// `[N * top_k]` i32 — M9 token index per permuted entry.
    permuted_token_idx: GpuBuffer,
    /// `[N * top_k]` i32 — M9 kpos per permuted entry.
    permuted_kpos: GpuBuffer,
    /// `[N * top_k]` BF16 — M9 weight per permuted entry.
    permuted_weight: GpuBuffer,
    /// `[N * top_k]` i32 — host-built M9 inverse for the unpermute kernel.
    permuted_inverse: GpuBuffer,
    /// `[N * top_k, hidden]` BF16 — M10 expert outputs.
    expert_out: GpuBuffer,
    /// `[N * top_k, moe_intermediate]` F32 — Metal direct routed expert
    /// intermediate between gate/up and down/combine.
    expert_mid: GpuBuffer,
    /// `[N, hidden]` BF16 — unpermute+combine output (sum of routed experts).
    combined: GpuBuffer,
    /// `[1]` u32 — M10 work-stealing counter (must be re-zeroed per launch).
    expert_counters: GpuBuffer,
    // Shared expert workspace.
    /// `[N, shared_intermediate]` BF16 — shared gate projection.
    shared_gate: GpuBuffer,
    /// `[N, shared_intermediate]` BF16 — shared up projection.
    shared_up: GpuBuffer,
    /// `[N, shared_intermediate]` BF16 — silu(gate) * up.
    shared_silu_mul: GpuBuffer,
    /// `[N, hidden]` BF16 — shared expert down projection (pre-gate scalar).
    shared_down: GpuBuffer,
    /// `[N, 1]` BF16 — shared expert scalar gate (sigmoid applied later).
    shared_gate_scalar: GpuBuffer,
    /// `[N, hidden]` BF16 — shared expert output after sigmoid gating.
    shared_out: GpuBuffer,
    /// `[N, hidden]` BF16 — fallback temp for shared expert scalar gating.
    shared_out_final: GpuBuffer,
}

impl GroupedFfnScratch {
    fn alloc(ordinal: usize, geom: &MultiLayerGeom, n: usize) -> Result<Self> {
        let hidden = geom.hidden as usize;
        let num_experts = geom.num_experts as usize;
        let top_k = geom.top_k as usize;
        let shared_intermediate = geom.shared_intermediate as usize;
        let nk = n * top_k;

        let h_norm = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc grouped_ffn h_norm")?;
        let router_logits_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, num_experts])
            .context("alloc grouped_ffn router_logits_bf16")?;
        let topk_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[n, top_k])
            .context("alloc grouped_ffn topk_idx")?;
        let topk_weight = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, top_k])
            .context("alloc grouped_ffn topk_weight")?;
        let expert_offsets = GpuBuffer::zeros(ordinal, ScalarType::U32, &[num_experts + 1])
            .context("alloc grouped_ffn expert_offsets")?;
        let permuted_token_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[nk])
            .context("alloc grouped_ffn permuted_token_idx")?;
        let permuted_kpos = GpuBuffer::zeros(ordinal, ScalarType::U32, &[nk])
            .context("alloc grouped_ffn permuted_kpos")?;
        let permuted_weight = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nk])
            .context("alloc grouped_ffn permuted_weight")?;
        let permuted_inverse = GpuBuffer::zeros(ordinal, ScalarType::U32, &[nk])
            .context("alloc grouped_ffn permuted_inverse")?;
        let expert_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nk, hidden])
            .context("alloc grouped_ffn expert_out")?;
        let expert_mid = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[nk, geom.moe_intermediate as usize],
        )
        .context("alloc grouped_ffn expert_mid")?;
        let combined = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc grouped_ffn combined")?;
        let expert_counters = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .context("alloc grouped_ffn expert_counters")?;
        let shared_gate = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, shared_intermediate])
            .context("alloc grouped_ffn shared_gate")?;
        let shared_up = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, shared_intermediate])
            .context("alloc grouped_ffn shared_up")?;
        let shared_silu_mul =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, shared_intermediate])
                .context("alloc grouped_ffn shared_silu_mul")?;
        let shared_down = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc grouped_ffn shared_down")?;
        let shared_gate_scalar = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, 1])
            .context("alloc grouped_ffn shared_gate_scalar")?;
        let shared_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc grouped_ffn shared_out")?;
        let shared_out_final = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n, hidden])
            .context("alloc grouped_ffn shared_out_final")?;

        Ok(Self {
            n,
            top_k,
            num_experts,
            h_norm,
            router_logits_bf16,
            topk_idx,
            topk_weight,
            expert_offsets,
            permuted_token_idx,
            permuted_kpos,
            permuted_weight,
            permuted_inverse,
            expert_out,
            expert_mid,
            combined,
            expert_counters,
            shared_gate,
            shared_up,
            shared_silu_mul,
            shared_down,
            shared_gate_scalar,
            shared_out,
            shared_out_final,
        })
    }
}

/// M11 batched grouped FFN — replaces the per-token MoE FFN loop with a
/// batched primitive sequence:
///
/// 1. RMSnorm rows (post-attention, HF (1+w) form) → `h_norm[N, hidden]`.
/// 2. Router matvec (BF16 weight) → `router_logits[N, num_experts]` F32.
/// 3. Host softmax + top-K + renorm → `topk_idx[N, top_k]` i32 +
///    `topk_weight[N, top_k]` BF16. (TODO future optimisation: GPU kernel.)
/// 4. M9 router permute → `expert_offsets`, `permuted_token_idx`,
///    `permuted_kpos`, `permuted_weight`. Inverse table built host-side.
/// 5. M10 grouped expert GEMM → `expert_out[N * top_k, hidden]`.
/// 6. M11 unpermute + weighted combine → `combined[N, hidden]`.
/// 7. Shared expert (per-token branch every token):
///       shared_gate = INT4_matmul(h_norm, shared_gate_proj_w)
///       shared_up   = INT4_matmul(h_norm, shared_up_proj_w)
///       shared_silu_mul = silu(shared_gate) * shared_up   (via swiglu_mul)
///       shared_down = INT4_matmul(shared_silu_mul, shared_down_proj_w)
///       shared_gate_scalar = BF16_matmul(h_norm, shared_expert_gate_w)
///       shared_out  = sigmoid(shared_gate_scalar) * shared_down  (via sigmoid_mul)
/// 8. Residual: `chunk_hidden += combined + shared_out`.
///
/// Mirrors the per-token kernel's math; combine is permutation-invariant
/// inside an expert's segment (M9's atomicAdd is unstable but the weighted
/// sum doesn't care about order).
#[allow(clippy::too_many_arguments)]
fn process_ffn_batched_grouped(
    ordinal: usize,
    geom: &MultiLayerGeom,
    n: usize,
    chunk_hidden: &mut GpuBuffer,
    ffn: &crate::qwen36_moe_types::FfnLayerBuffers,
    scratch: &mut GroupedFfnScratch,
    _sync_buf: &mut GpuBuffer,
) -> Result<()> {
    debug_assert_eq!(
        scratch.n, n,
        "GroupedFfnScratch sized for chunk_n != current n"
    );
    let hidden = geom.hidden as usize;
    let num_experts = geom.num_experts as usize;
    let top_k = geom.top_k as usize;
    let moe_intermediate = geom.moe_intermediate as usize;
    let shared_intermediate = geom.shared_intermediate as usize;

    let int4 = ffn.int4.as_ref().ok_or_else(|| {
        anyhow!("grouped FFN requires INT4 sidecars; BF16-only path not supported")
    })?;
    let group_size = int4.group_size;
    if group_size != 128 {
        return Err(anyhow!(
            "grouped FFN expects INT4 group_size=128, got {group_size}"
        ));
    }
    let group_size = group_size as usize;
    // INT4 quant_type code (matches LOWBIT_NATIVE_INT4 used elsewhere).
    let qtype = QUANT_TYPE_NATIVE_INT4;

    // 1. Post-attention RMSnorm (HF (1+w) form).
    prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        n,
        hidden,
        geom.rms_norm_eps,
        chunk_hidden,
        &ffn.post_attn_norm_w,
        &mut scratch.h_norm,
    )
    .map_err(|e| anyhow!("rms_norm_rows post_attn: {e}"))?;

    // 2. Router matvec (BF16). gate_w is `[num_experts, hidden]` BF16, which
    //    matches the matmul_rhs_transposed contract:
    //    out[m, n] = lhs[m, k] @ rhs[n, k]^T → out[N, num_experts] BF16.
    //    Per-token kernel stores logits as `bf16_round_rne_f32(F32 dot)`
    //    in workspace (F32 storage); we keep them as BF16 here, then widen
    //    for the softmax — bit-exact equivalent.
    prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        n,
        num_experts,
        hidden,
        &scratch.h_norm,
        &ffn.gate_w,
        &mut scratch.router_logits_bf16,
    )
    .map_err(|e| anyhow!("router matmul: {e}"))?;

    // 3. Host softmax + top-K + renorm. D2H router_logits (BF16), widen to
    //    F32 per-element, run softmax + top-K + renorm matching the per-token
    //    kernel's Phase C. H2D topk_idx + topk_weight. At chunk_n=64,
    //    num_experts=256 this is 64*256*2 = 32 KiB D2H + 64*8*4 + 64*8*2 =
    //    ~2 KiB H2D per layer per chunk — negligible vs the matmul cost.
    //    (TODO: GPU softmax/top-K fusion as a future M12+ perf opportunity.)
    let mut logits_bytes = vec![0u8; n * num_experts * 2];
    copy_d2h(
        ordinal,
        logits_bytes.as_mut_ptr() as *mut c_void,
        scratch.router_logits_bf16.as_ptr(),
        logits_bytes.len(),
    )
    .context("d2h router_logits_bf16")?;
    let mut topk_idx_host = vec![0i32; n * top_k];
    let mut topk_weight_host = vec![0u16; n * top_k];
    for token in 0..n {
        let row_bf16 = unsafe {
            std::slice::from_raw_parts(
                (logits_bytes.as_ptr() as *const u16).add(token * num_experts),
                num_experts,
            )
        };
        // Widen BF16 → F32 to match the per-token kernel's Phase C, which
        // reads workspace[OFF_ROUTER_LOGITS] as F32 (the value was stored
        // via `bf16_round_rne_f32` in Phase B).
        let row: Vec<f32> = row_bf16.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        // Match per-token kernel (`ffn_phase.cuh` Phase C): softmax with
        // BF16-rounded probs, then top-K with low-index tie-breaking, then
        // renormalise top-K to sum to 1 (BF16-rounded again).
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - row_max).exp()).collect();
        let row_sum: f32 = exps.iter().copied().sum();
        let inv_sum = 1.0f32 / row_sum;
        // BF16-round each prob to match the per-token kernel.
        let mut probs: Vec<f32> = exps
            .iter()
            .map(|&e| bf16_round_rne_f32(e * inv_sum))
            .collect();

        // Top-K with low-index tie-break (mark with -inf as we go).
        for k in 0..top_k {
            let mut best_idx = -1i32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in probs.iter().enumerate() {
                if v > best_val || (v == best_val && best_idx >= 0 && (i as i32) < best_idx) {
                    best_val = v;
                    best_idx = i as i32;
                }
            }
            topk_idx_host[token * top_k + k] = best_idx;
            topk_weight_host[token * top_k + k] = f32_to_bf16_bits(best_val);
            // Mask the winner so the next iteration picks the next highest.
            if best_idx >= 0 {
                probs[best_idx as usize] = f32::NEG_INFINITY;
            }
        }
        // Renormalise the top-K weights.
        let sum_k: f32 = (0..top_k)
            .map(|k| bf16_bits_to_f32(topk_weight_host[token * top_k + k]))
            .sum();
        let inv_k = 1.0f32 / sum_k;
        for k in 0..top_k {
            let w = bf16_bits_to_f32(topk_weight_host[token * top_k + k]);
            topk_weight_host[token * top_k + k] = f32_to_bf16_bits(bf16_round_rne_f32(w * inv_k));
        }
        let active_experts: Vec<usize> = (0..top_k)
            .map(|k| topk_idx_host[token * top_k + k].max(0) as usize)
            .collect();
        kernel_ffi::qwen36_moe::qwen36_route_profile_record_active_experts(&active_experts);
    }
    copy_h2d(
        ordinal,
        scratch.topk_idx.as_mut_ptr(),
        topk_idx_host.as_ptr() as *const c_void,
        topk_idx_host.len() * 4,
    )
    .context("h2d topk_idx")?;
    copy_h2d(
        ordinal,
        scratch.topk_weight.as_mut_ptr(),
        topk_weight_host.as_ptr() as *const c_void,
        topk_weight_host.len() * 2,
    )
    .context("h2d topk_weight")?;

    if scratch.h_norm.backend() == Backend::Metal {
        // Metal v1 prototype: keep the same host router/top-k contract but
        // bypass the HIP M9/M10/M11 permutation kernels with direct batched
        // routed-expert compute. This is correctness-first and opt-in through
        // the outer Metal batched-prefill prototype gate.
        unsafe {
            kernel_ffi::qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw(
                ordinal,
                n,
                top_k,
                hidden,
                moe_intermediate,
                group_size,
                &scratch.h_norm,
                &scratch.topk_idx,
                &scratch.topk_weight,
                ffn.gate_up_proj_w.as_ptr(),
                int4.gate_up_proj_scale.as_ptr(),
                int4.gate_up_proj_zero.as_ptr(),
                ffn.down_proj_w.as_ptr(),
                int4.down_proj_scale.as_ptr(),
                int4.down_proj_zero.as_ptr(),
                &mut scratch.expert_mid,
                &mut scratch.combined,
            )
        }
        .map_err(|e| anyhow!("Metal direct grouped expert: {e}"))?;
    } else {
        // 4a. M9 router permute.
        batched_prefill_router_permute_launch(
            ordinal,
            n,
            top_k,
            num_experts,
            &scratch.topk_idx,
            &scratch.topk_weight,
            &mut scratch.expert_offsets,
            &mut scratch.permuted_token_idx,
            &mut scratch.permuted_kpos,
            &mut scratch.permuted_weight,
        )
        .map_err(|e| anyhow!("M9 router permute: {e}"))?;

        // 4b. Build host-side inverse: for each (token, kpos) pair, the dst slot
        //     in the permuted_*[] arrays. M9's scatter is unstable (atomicAdd
        //     cursor) so we have to read back permuted_token_idx + permuted_kpos
        //     to know where each entry landed. O(N * top_k) — trivial cost.
        let nk = n * top_k;
        let mut perm_tok = vec![0i32; nk];
        let mut perm_kpos = vec![0i32; nk];
        copy_d2h(
            ordinal,
            perm_tok.as_mut_ptr() as *mut c_void,
            scratch.permuted_token_idx.as_ptr(),
            nk * 4,
        )
        .context("d2h permuted_token_idx")?;
        copy_d2h(
            ordinal,
            perm_kpos.as_mut_ptr() as *mut c_void,
            scratch.permuted_kpos.as_ptr(),
            nk * 4,
        )
        .context("d2h permuted_kpos")?;
        let mut inverse = vec![-1i32; nk];
        for dst in 0..nk {
            let token = perm_tok[dst];
            let kpos = perm_kpos[dst];
            if token < 0 || kpos < 0 || token >= n as i32 || kpos >= top_k as i32 {
                return Err(anyhow!(
                    "M9 permutation out-of-range entry at dst={dst}: token={token} kpos={kpos}"
                ));
            }
            let logical = token as usize * top_k + kpos as usize;
            if inverse[logical] != -1 {
                return Err(anyhow!(
                    "M9 permutation collision at logical entry token={token} kpos={kpos}"
                ));
            }
            inverse[logical] = dst as i32;
        }
        for (logical, &v) in inverse.iter().enumerate() {
            if v < 0 {
                return Err(anyhow!("M9 permutation missing entry at logical={logical}"));
            }
        }
        copy_h2d(
            ordinal,
            scratch.permuted_inverse.as_mut_ptr(),
            inverse.as_ptr() as *const c_void,
            nk * 4,
        )
        .context("h2d permuted_inverse")?;

        // 5. M10 grouped expert GEMM. Counter MUST be zeroed before launch.
        gpu_hal::memset_zeros(ordinal, scratch.expert_counters.as_mut_ptr(), 4)
            .context("zero expert_counters")?;
        // ResidentWeight as_ptr() may point to either a Dense GpuBuffer or a
        // Virtual VMM allocation; the raw-pointer launcher accepts both.
        let gate_up_w_ptr = ffn.gate_up_proj_w.as_ptr();
        let down_w_ptr = ffn.down_proj_w.as_ptr();
        unsafe {
            batched_prefill_grouped_expert_launch_raw(
                ordinal,
                n,
                top_k,
                num_experts,
                hidden,
                moe_intermediate,
                group_size,
                &scratch.h_norm,
                &scratch.expert_offsets,
                &scratch.permuted_token_idx,
                gate_up_w_ptr,
                int4.gate_up_proj_scale.as_ptr(),
                int4.gate_up_proj_zero.as_ptr(),
                down_w_ptr,
                int4.down_proj_scale.as_ptr(),
                int4.down_proj_zero.as_ptr(),
                &mut scratch.expert_out,
                &mut scratch.expert_counters,
            )
        }
        .map_err(|e| anyhow!("M10 grouped expert: {e}"))?;

        // 6. M11 unpermute + weighted combine.
        batched_prefill_unpermute_combine_launch(
            ordinal,
            n,
            top_k,
            hidden,
            &scratch.permuted_inverse,
            &scratch.permuted_weight,
            &scratch.expert_out,
            &mut scratch.combined,
        )
        .map_err(|e| anyhow!("M11 unpermute combine: {e}"))?;
    }

    // 7. Shared expert (batched primitives).
    //
    //    7a. shared_gate = INT4_matmul(h_norm, shared_gate_proj_w)
    //         shape [shared_intermediate, hidden] INT4 → out [N, shared_intermediate]
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        shared_intermediate,
        hidden,
        &scratch.h_norm,
        &ffn.shared_gate_proj_w,
        &int4.shared_gate_proj_scale,
        &int4.shared_gate_proj_zero,
        None,
        group_size,
        qtype,
        &mut scratch.shared_gate,
    )
    .map_err(|e| anyhow!("matmul shared_gate_proj: {e}"))?;
    //    7b. shared_up = INT4_matmul(h_norm, shared_up_proj_w)
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        shared_intermediate,
        hidden,
        &scratch.h_norm,
        &ffn.shared_up_proj_w,
        &int4.shared_up_proj_scale,
        &int4.shared_up_proj_zero,
        None,
        group_size,
        qtype,
        &mut scratch.shared_up,
    )
    .map_err(|e| anyhow!("matmul shared_up_proj: {e}"))?;
    //    7c. shared_silu_mul = silu(shared_gate) * shared_up
    prefill_ffi::swiglu_mul(
        ordinal,
        ScalarType::BF16,
        n * shared_intermediate,
        &scratch.shared_gate,
        &scratch.shared_up,
        &mut scratch.shared_silu_mul,
    )
    .map_err(|e| anyhow!("swiglu_mul shared: {e}"))?;
    //    7d. shared_down = INT4_matmul(shared_silu_mul, shared_down_proj_w)
    //         shape [hidden, shared_intermediate] INT4 → out [N, hidden]
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        n,
        hidden,
        shared_intermediate,
        &scratch.shared_silu_mul,
        &ffn.shared_down_proj_w,
        &int4.shared_down_proj_scale,
        &int4.shared_down_proj_zero,
        None,
        group_size,
        qtype,
        &mut scratch.shared_down,
    )
    .map_err(|e| anyhow!("matmul shared_down_proj: {e}"))?;
    //    7e. shared_gate_scalar = BF16_matmul(h_norm, shared_expert_gate_w)
    //         shape [1, hidden] BF16 → out [N, 1]
    prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        n,
        1,
        hidden,
        &scratch.h_norm,
        &ffn.shared_expert_gate_w,
        &mut scratch.shared_gate_scalar,
    )
    .map_err(|e| anyhow!("matmul shared_expert_gate: {e}"))?;
    //    7f. shared_out = sigmoid(shared_gate_scalar) * shared_down.
    //         Metal uses a row-scalar kernel so the `[N, 1]` gate never
    //         round-trips through host memory as an expanded `[N, hidden]`
    //         buffer. HIP/CUDA keep the older explicit expansion path for now.
    if scratch.shared_down.backend() == Backend::Metal {
        prefill_ffi::sigmoid_mul_row_scalar_bf16(
            ordinal,
            n,
            hidden,
            &scratch.shared_down,
            &scratch.shared_gate_scalar,
            &mut scratch.shared_out,
        )
        .map_err(|e| anyhow!("sigmoid_mul_row_scalar shared: {e}"))?;
    } else {
        expand_scalar_gate_bf16(
            ordinal,
            n,
            hidden,
            &scratch.shared_gate_scalar,
            &mut scratch.shared_out, // reuse shared_out as the temp expanded gate
        )?;
        prefill_ffi::sigmoid_mul(
            ordinal,
            ScalarType::BF16,
            n * hidden,
            &scratch.shared_down,
            &scratch.shared_out,
            &mut scratch.shared_out_final,
        )
        .map_err(|e| anyhow!("sigmoid_mul shared: {e}"))?;
        gpu_hal::copy_d2d(
            ordinal,
            scratch.shared_out.as_mut_ptr(),
            scratch.shared_out_final.as_ptr(),
            n * hidden * 2,
        )
        .context("d2d shared_out_final -> shared_out")?;
    }

    // 8. Residual add: chunk_hidden += combined; chunk_hidden += shared_out.
    prefill_ffi::element_add_inplace(
        ordinal,
        ScalarType::BF16,
        n * hidden,
        chunk_hidden,
        &scratch.combined,
    )
    .map_err(|e| anyhow!("residual add (combined): {e}"))?;
    prefill_ffi::element_add_inplace(
        ordinal,
        ScalarType::BF16,
        n * hidden,
        chunk_hidden,
        &scratch.shared_out,
    )
    .map_err(|e| anyhow!("residual add (shared_out): {e}"))?;

    // Touch unused fields to keep the compiler happy on cfg(feature) gates.
    let _ = (geom, scratch.num_experts, scratch.top_k);
    Ok(())
}

/// Host-side BF16 round-to-nearest-even, matching `bf16_round_rne_f32` in
/// `helpers.cuh`. Used by the host softmax+top-K to keep parity with the
/// per-token FFN's BF16 round points.
#[inline]
fn bf16_round_rne_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1u32);
    let rounded = bits.wrapping_add(rounding_bias) & 0xFFFF_0000u32;
    f32::from_bits(rounded)
}

/// Host-side BF16 ↔ F32 helpers for the host softmax pass. We avoid a
/// `half::bf16` dependency in this hot loop — direct bit twiddling matches
/// `bf16_round_rne_f32` semantics.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFFu32 + ((bits >> 16) & 1u32));
    (rounded >> 16) as u16
}

#[inline]
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Expand a `[N, 1]` BF16 scalar-per-token into a `[N, hidden]` BF16 buffer
/// by replicating each scalar across the hidden axis. Done host-side
/// because gpu-hal lacks a broadcast primitive and the data volume is small
/// (n * hidden * 2 bytes per layer per chunk ≈ 256 KiB at chunk_n=64,
/// hidden=2048). A future GPU memset-broadcast would be cheaper for large
/// chunks; for M11's parity gate, host expansion is fine.
fn expand_scalar_gate_bf16(
    ordinal: usize,
    n: usize,
    hidden: usize,
    scalars: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<()> {
    let mut host_scalars = vec![0u8; n * 2];
    copy_d2h(
        ordinal,
        host_scalars.as_mut_ptr() as *mut c_void,
        scalars.as_ptr(),
        host_scalars.len(),
    )
    .context("d2h scalar gate")?;
    let mut host_expanded = vec![0u8; n * hidden * 2];
    for token in 0..n {
        let s0 = host_scalars[token * 2];
        let s1 = host_scalars[token * 2 + 1];
        let row_off = token * hidden * 2;
        for col in 0..hidden {
            host_expanded[row_off + col * 2] = s0;
            host_expanded[row_off + col * 2 + 1] = s1;
        }
    }
    copy_h2d(
        ordinal,
        out.as_mut_ptr(),
        host_expanded.as_ptr() as *const c_void,
        host_expanded.len(),
    )
    .context("h2d expanded gate")?;
    Ok(())
}
