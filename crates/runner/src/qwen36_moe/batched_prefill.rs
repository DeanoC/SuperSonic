//! Host-side orchestrator for Qwen 3.6 MoE batched-Q prefill (Stage A).
//!
//! M6.1 SKELETON: runs the existing per-token persistent megakernel
//! `run_chain_step` once per token within each chunk. Functionally
//! identical to the engine's per-token loop — exists so M6.3+ can
//! incrementally replace the inner per-token call with truly batched
//! kernel invocations without disturbing the chunking + state-management
//! infrastructure.
//!
//! When `SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=1` is set the engine
//! routes prefill through `run_batched_prefill_stub` instead of running
//! the prefill iterations of its main per-step loop. This module owns
//! the *prefill range only* — the FIRST generation step (where
//! `step + 1 == effective_prompt_len` and logits are computed) is left
//! to the engine's main loop.
//!
//! Plan: docs/superpowers/plans/2026-05-05-qwen36-moe-batched-prefill-phase1.md

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use model_store::BakedStore;

use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::engine::current_position;
use crate::qwen36_moe_cli::host::lookup_embed_row;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_persistent_decode::PersistentScratch;
use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_telemetry::MoeRouteRuntime;
use crate::qwen36_moe_types::{LayerBuffers, MultiLayerGeom};

/// Number of prompt tokens per outer chunk. The skeleton still calls
/// the per-token persistent megakernel inside the inner loop, so this
/// is purely a demarcation constant — M6.3+ replaces the inner per-token
/// chain calls with truly batched kernels and at that point this becomes
/// the actual `q_len` per launch.
pub(crate) const PREFILL_CHUNK_SIZE: usize = 64;

/// Aggregated timings for the batched-prefill orchestrator pass.
pub(crate) struct BatchedPrefillTimings {
    pub embed_total: Duration,
    pub chain_total: Duration,
    pub chunks: usize,
    pub tokens: usize,
}

/// Drive the prefill range `[0, effective_prompt_len - 1)` through the
/// per-token persistent megakernel, mirroring the engine's main-loop
/// body exactly. Mutates `loop_state.position` and
/// `loop_state.current_token` so the engine's main loop can resume at
/// `step = effective_prompt_len - 1` (the FIRST generation step) without
/// double-processing or skipping anything.
///
/// This is a no-op refactor in spirit: when invoked, the chain-step
/// calls and per-step state mutations match the engine's existing
/// `for step in 0..total_steps` body for the prefill portion. The
/// outer chunking exists so M6.3+ can swap `run_chain_step` per token
/// for batched kernel launches without disturbing this scaffolding.
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
    // Range invariant. Prefill steps in the engine's main loop are
    // step ∈ [0, effective_prompt_len - 1); at step ==
    // effective_prompt_len - 1 logits are computed (gen-fold step) and
    // that step is owned by the engine's main loop. So this orchestrator
    // processes exactly `effective_prompt_len - 1` tokens.
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

    // The chain step needs `&mut` access to many engine-owned bits.
    // Re-borrow as `Option<&mut _>` per inner iteration so we can pass
    // them anew to each `Qwen36ChainStep` without moving them out of
    // the outer mutable borrows.
    let mut persistent_scratch = persistent_scratch;
    let mut moe_expert_residency = moe_expert_residency;

    let full_prompt_len = prompt_ids.len();

    let mut step = 0usize;
    while step < prefill_count {
        let chunk_end = (step + PREFILL_CHUNK_SIZE).min(prefill_count);
        timings.chunks += 1;

        while step < chunk_end {
            // Per-step (rope, cache) pair — identical to engine main loop.
            let position = current_position(
                step,
                loop_state.position,
                keep_mask,
                kept_positions,
                effective_prompt_len,
                full_prompt_len,
            );

            // Embed lookup for the current token — host-side BF16 row slice.
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

            // Re-borrow the optional `&mut` references per call so we
            // don't move them out of the orchestrator's outer borrow.
            let scratch_arg: Option<&mut PersistentScratch> =
                persistent_scratch.as_deref_mut();
            let residency_arg: Option<&mut MoeExpertResidencyManager> =
                moe_expert_residency.as_deref_mut();

            // Run the per-token chain. Prefill steps never compute
            // logits, so `fold = None`. `is_gen_step = false` matches
            // the engine's main loop for prefill iterations
            // (`is_gen_step = step + 1 >= effective_prompt_len`,
            // and the orchestrator's range is exactly the indices
            // where that condition is false).
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

            // Per-step state advance — mirrors the engine main loop.
            loop_state.position += 1;
            timings.embed_total += t_embed_step;
            timings.chain_total += t_chain_step;
            timings.tokens += 1;

            // Update `current_token` to the NEXT token to process. The
            // step we just finished was at index `step`; the next
            // iteration (whether the next inner step here, the next
            // chunk, or the engine's first gen step at index
            // effective_prompt_len - 1) needs the kept-positions[step+1]
            // token. Because `prefill_count == effective_prompt_len - 1`
            // and `step` reaches at most `prefill_count - 1`, `step + 1`
            // is at most `effective_prompt_len - 1`, which is a valid
            // index into `kept_positions`.
            loop_state.current_token = prompt_ids[kept_positions[step + 1]];
            step += 1;
        }
    }

    Ok(timings)
}
