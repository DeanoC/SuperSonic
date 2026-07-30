use anyhow::{Context, Result};
use model_store::BakedStore;

use crate::qwen36_moe_cli::prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe_cli::vmm_config::MoeRuntimeConfig;
use crate::qwen36_moe_persistent_decode::LmHeadFold;
use crate::qwen36_moe_telemetry::{MoeRouteRuntime, MoeSparseTelemetrySnapshot};
use crate::qwen36_moe_types::{
    DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, MultiLayerGeom, PositionPair,
};
use supersonic_runtime::qwen36_moe::decode::Qwen36ExecutionOptions;
use supersonic_runtime::qwen36_moe::layers::LoadedQwen36Layers;

pub(crate) struct Qwen36ChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) loaded_layers: &'a mut LoadedQwen36Layers,
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
    pub(crate) download_final_hidden: bool,
    pub(crate) execution: &'a Qwen36ExecutionOptions,
}

pub(crate) struct Qwen36ChainStepOutput {
    pub(crate) outputs: DecodeOutputs,
    pub(crate) lm_head_folded: bool,
    pub(crate) lm_head_folded_top1: bool,
}

pub(crate) fn run_chain_step(args: Qwen36ChainStep<'_>) -> Result<Qwen36ChainStepOutput> {
    let moe_telemetry_before = args
        .loaded_layers
        .sparse_expert_residency()
        .map(MoeSparseTelemetrySnapshot::capture);
    let track_moe_routes = args
        .moe_routes
        .should_track_routes(args.moe_runtime.prefetch_mode);
    let mut next_moe_topk_by_layer = args.moe_routes.next_topk_buffer(track_moe_routes);

    let rope = args.position.rope;
    let cache = args.position.cache;
    let runtime_output = if std::env::var_os("SUPERSONIC_QWEN36_SEGMENTED_PROFILE").is_some()
        && args.loaded_layers.persistent_enabled()
        && !args.loaded_layers.has_sparse_expert_residency()
    {
        drop(args.fold);
        let outputs = args
            .loaded_layers
            .run_segmented_profile(
                args.ordinal,
                args.initial_hidden,
                rope,
                cache,
                args.execution,
            )
            .with_context(|| {
                format!(
                    "persistent segmented profile decode (step {}, rope {}, cache {})",
                    args.step, rope, cache
                )
            })?;
        supersonic_runtime::qwen36_moe::chain::Qwen36ChainStepOutput {
            outputs,
            lm_head_folded: false,
            lm_head_folded_top1: false,
        }
    } else {
        let mut prefetch =
            |manager: &mut crate::qwen36_moe_residency::MoeExpertResidencyManager,
             phase: ExpertPrefetchPhase,
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
        let expert_prefetch = args.loaded_layers.has_sparse_expert_residency().then_some(
            &mut prefetch
                as &mut supersonic_runtime::qwen36_moe::chain::ChainExpertPrefetchCallback<'_>,
        );
        supersonic_runtime::qwen36_moe::chain::run_chain_step(
            supersonic_runtime::qwen36_moe::chain::Qwen36ChainStep {
                ordinal: args.ordinal,
                geom: args.geom,
                loaded_layers: args.loaded_layers,
                initial_hidden: args.initial_hidden,
                position: args.position,
                step: args.step,
                accurate_stage_timings: args.emit_stage_timings,
                fold: args.fold,
                download_final_hidden: args.download_final_hidden,
                expert_prefetch,
                execution: args.execution,
            },
        )?
    };

    args.moe_routes
        .advance(track_moe_routes, next_moe_topk_by_layer);
    if let (Some(telemetry), Some(before), Some(manager)) = (
        args.moe_runtime.sparse_telemetry.as_mut(),
        moe_telemetry_before,
        args.loaded_layers.sparse_expert_residency(),
    ) {
        let after = MoeSparseTelemetrySnapshot::capture(manager);
        telemetry.record_step(args.step, rope, args.is_gen_step, before, after);
    }

    Ok(Qwen36ChainStepOutput {
        outputs: runtime_output.outputs,
        lm_head_folded: runtime_output.lm_head_folded,
        lm_head_folded_top1: runtime_output.lm_head_folded_top1,
    })
}
