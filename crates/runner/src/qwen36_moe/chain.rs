use anyhow::{Context, Result};
use model_store::BakedStore;

use crate::qwen36_moe_cli::prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_decode::{
    run_chained_decode_fast, run_chained_decode_fast_with_expert_prefetch,
};
use crate::qwen36_moe_persistent_decode::{LmHeadFold, PersistentScratch};
use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_telemetry::{MoeRouteRuntime, MoeSparseTelemetrySnapshot};
use crate::qwen36_moe_types::{
    DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom,
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
    pub(crate) position: i32,
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

    let mut lm_head_folded = false;
    let outputs = if let Some(scratch) = args.persistent_scratch.as_deref_mut() {
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
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
                    phase,
                    layer_idx,
                    routes,
                )
            };
            scratch
                .run_sparse_with_expert_prefetch(
                    args.ordinal,
                    args.initial_hidden,
                    args.position,
                    &mut prefetch,
                )
                .with_context(|| {
                    format!(
                        "segmented persistent sparse decode (step {}, position {})",
                        args.step, args.position
                    )
                })?
        } else {
            lm_head_folded = args.fold.is_some();
            scratch
                .run(args.ordinal, args.initial_hidden, args.position, args.fold)
                .with_context(|| {
                    format!(
                        "persistent decode (step {}, position {})",
                        args.step, args.position
                    )
                })?
        }
    } else {
        drop(args.fold);
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
                    &args.moe_routes.previous_topk_by_layer,
                    &mut next_moe_topk_by_layer,
                    track_moe_routes,
                    args.moe_routes.route_telemetry.as_mut(),
                    args.moe_routes.transition_predictors.as_deref_mut(),
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
                args.position,
                args.emit_stage_timings,
                &mut prefetch,
            )
        } else {
            run_chained_decode_fast(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                args.position,
                args.emit_stage_timings,
            )
        }
        .with_context(|| {
            format!(
                "chained decode (step {}, position {})",
                args.step, args.position
            )
        })?
    };

    args.moe_routes
        .advance(track_moe_routes, next_moe_topk_by_layer);
    if let (Some(telemetry), Some(before), Some(manager)) = (
        args.moe_runtime.sparse_telemetry.as_mut(),
        moe_telemetry_before,
        args.moe_expert_residency.as_deref(),
    ) {
        let after = MoeSparseTelemetrySnapshot::capture(manager);
        telemetry.record_step(args.step, args.position, args.is_gen_step, before, after);
    }

    Ok(Qwen36ChainStepOutput {
        outputs,
        lm_head_folded,
    })
}
