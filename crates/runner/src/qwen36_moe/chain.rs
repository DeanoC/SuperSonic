use anyhow::{Context, Result};
use model_store::BakedStore;

use crate::qwen36_moe_cli::prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_decode::{
    run_chained_decode_fast, run_chained_decode_fast_with_cache_pos,
    run_chained_decode_fast_with_expert_prefetch,
    run_chained_decode_fast_with_expert_prefetch_and_cache_pos,
};
use crate::qwen36_moe_persistent_decode::{LmHeadFold, PersistentScratch};
use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_telemetry::{MoeRouteRuntime, MoeSparseTelemetrySnapshot};
use crate::qwen36_moe_types::{
    DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom, PositionPair,
};

pub(crate) struct Qwen36ChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) moe_expert_residency: Option<&'a mut MoeExpertResidencyManager>,
    pub(crate) moe_runtime: &'a mut MoeRuntimeConfig,
    pub(crate) moe_routes: &'a mut MoeRouteRuntime,
    pub(crate) initial_hidden: &'a [u8],
    /// `(rope, cache)` for this step. Dense decode uses
    /// `PositionPair::dense(loop_state.position)`; SpecPrefill uses
    /// `PositionPair::split(rope_pos, loop_state.position)`. The
    /// chained-fallback branch only uses the cache_pos sibling fns
    /// when `!position.is_dense()`.
    pub(crate) position: PositionPair,
    pub(crate) step: usize,
    pub(crate) is_gen_step: bool,
    pub(crate) emit_stage_timings: bool,
    pub(crate) fold: Option<LmHeadFold<'a>>,
}

pub(crate) struct Qwen36ChainStepOutput {
    pub(crate) outputs: DecodeOutputs,
    pub(crate) lm_head_folded: bool,
}

pub(crate) fn run_chain_step(mut args: Qwen36ChainStep<'_>) -> Result<Qwen36ChainStepOutput> {
    let moe_telemetry_before = args
        .moe_expert_residency
        .as_deref()
        .map(MoeSparseTelemetrySnapshot::capture);
    let track_moe_routes = args
        .moe_routes
        .should_track_routes(args.moe_runtime.prefetch_mode);
    let mut next_moe_topk_by_layer = args.moe_routes.next_topk_buffer(track_moe_routes);

    let rope = args.position.rope;
    let cache = args.position.cache;

    let mut lm_head_folded = false;
    let outputs = if let Some(scratch) = args.persistent_scratch.as_deref_mut() {
        // Persistent kernel takes (rope, cache) directly. The
        // megakernel's full-attn phase consumes cache_pos via
        // `eff_cache_pos = (cache_pos >= 0) ? cache_pos : position`,
        // so passing `cache` works for both dense and SpecPrefill
        // cases (when rope == cache the kernel produces bit-identical
        // output to the pre-PR-#211 hard-coded -1 path).
        if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
            drop(args.fold);
            let mut prefetch = |phase: ExpertPrefetchPhase,
                                layer_idx: usize,
                                routes: &[ExpertRoute]|
             -> Result<()> {
                handle_moe_expert_prefetch(
                    manager,
                    args.store,
                    args.moe_runtime.prefetch_mode,
                    args.moe_runtime.prefetch_ranks,
                    args.moe_runtime.prefetch_evict_min_probability,
                    args.moe_runtime.protect_demand_routes,
                    args.moe_runtime.hot_protect_min_hits,
                    args.moe_runtime.fixed_hot_min_hits,
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
                    args.moe_routes.hot_expert_counts.as_deref_mut(),
                    phase,
                    layer_idx,
                    routes,
                )
            };
            scratch
                .run_sparse_with_expert_prefetch(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    &mut prefetch,
                )
                .with_context(|| {
                    format!(
                        "segmented persistent sparse decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        } else {
            lm_head_folded = args.fold.is_some();
            scratch
                .run(args.ordinal, args.initial_hidden, rope, cache, args.fold)
                .with_context(|| {
                    format!(
                        "persistent decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        }
    } else {
        // Chained fallback for when the persistent megakernel isn't
        // available (engine started without --persistent-decode, or
        // a dispatch path that the persistent kernel doesn't yet
        // support). Chained has parallel cache_pos siblings; we
        // pick them only when the pair actually diverges.
        drop(args.fold);
        if !args.position.is_dense() {
            if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        args.store,
                        args.moe_runtime.prefetch_mode,
                        args.moe_runtime.prefetch_ranks,
                        args.moe_runtime.prefetch_evict_min_probability,
                        args.moe_runtime.protect_demand_routes,
                        args.moe_runtime.hot_protect_min_hits,
                        args.moe_runtime.fixed_hot_min_hits,
                        &args.moe_routes.previous_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        args.moe_routes.route_telemetry.as_mut(),
                        args.moe_routes.transition_predictors.as_deref_mut(),
                        args.moe_routes.hot_expert_counts.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                run_chained_decode_fast_with_expert_prefetch_and_cache_pos(
                    args.ordinal,
                    args.geom,
                    args.layers,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.emit_stage_timings,
                    &mut prefetch,
                )
            } else {
                run_chained_decode_fast_with_cache_pos(
                    args.ordinal,
                    args.geom,
                    args.layers,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.emit_stage_timings,
                )
            }
            .with_context(|| {
                format!(
                    "chained sparse-prefill decode (step {}, rope {}, cache {})",
                    args.step, rope, cache
                )
            })?
        } else if let Some(manager) = args.moe_expert_residency.as_deref_mut() {
            let mut prefetch = |phase: ExpertPrefetchPhase,
                                layer_idx: usize,
                                routes: &[ExpertRoute]|
             -> Result<()> {
                handle_moe_expert_prefetch(
                    manager,
                    args.store,
                    args.moe_runtime.prefetch_mode,
                    args.moe_runtime.prefetch_ranks,
                    args.moe_runtime.prefetch_evict_min_probability,
                    args.moe_runtime.protect_demand_routes,
                    args.moe_runtime.hot_protect_min_hits,
                    args.moe_runtime.fixed_hot_min_hits,
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
                    args.moe_routes.hot_expert_counts.as_deref_mut(),
                    phase,
                    layer_idx,
                    routes,
                )
            };
            run_chained_decode_fast_with_expert_prefetch(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.emit_stage_timings,
                &mut prefetch,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?
        } else {
            run_chained_decode_fast(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.emit_stage_timings,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?
        }
    };

    args.moe_routes
        .advance(track_moe_routes, next_moe_topk_by_layer);
    if let (Some(telemetry), Some(before), Some(manager)) = (
        args.moe_runtime.sparse_telemetry.as_mut(),
        moe_telemetry_before,
        args.moe_expert_residency.as_deref(),
    ) {
        let after = MoeSparseTelemetrySnapshot::capture(manager);
        telemetry.record_step(args.step, rope, args.is_gen_step, before, after);
    }

    Ok(Qwen36ChainStepOutput {
        outputs,
        lm_head_folded,
    })
}
