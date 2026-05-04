use anyhow::Result;
use std::borrow::Cow;

use model_store::BakedStore;

use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_residency_types::{MoeExpertKey, MoeExpertProjection};
use crate::qwen36_moe_telemetry::{
    MoeIslandPrefetchMode, MoeRouteTelemetry, MoeTransitionPredictor,
};
use crate::qwen36_moe_types::{ExpertPrefetchPhase, ExpertRoute};

pub(crate) fn handle_moe_expert_prefetch(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    previous_topk_by_layer: &[Vec<usize>],
    next_topk_by_layer: &mut [Vec<usize>],
    track_routes: bool,
    route_telemetry: Option<&mut MoeRouteTelemetry>,
    transition_predictors: Option<&mut [MoeTransitionPredictor]>,
    phase: ExpertPrefetchPhase,
    layer_idx: usize,
    routes: &[ExpertRoute],
) -> Result<()> {
    match phase {
        ExpertPrefetchPhase::Lookahead if mode.uses_previous_token_routes() => {
            prefetch_previous_token_routes(
                manager,
                store,
                mode,
                prefetch_ranks,
                previous_topk_by_layer,
                transition_predictors,
                layer_idx,
            )?;
        }
        ExpertPrefetchPhase::Lookahead => {}
        ExpertPrefetchPhase::Demand => {
            ensure_demand_routes(
                manager,
                store,
                previous_topk_by_layer,
                next_topk_by_layer,
                track_routes,
                route_telemetry,
                transition_predictors,
                layer_idx,
                routes,
            )?;
        }
    }
    Ok(())
}

fn prefetch_previous_token_routes(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    previous_topk_by_layer: &[Vec<usize>],
    transition_predictors: Option<&mut [MoeTransitionPredictor]>,
    layer_idx: usize,
) -> Result<()> {
    let previous_routes = previous_topk_by_layer
        .get(layer_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let transition_candidates;
    let candidate_experts: Cow<'_, [usize]> = if mode.transition_weighted() {
        transition_candidates = transition_predictors
            .as_ref()
            .and_then(|predictors| predictors.get(layer_idx))
            .map(|predictor| predictor.candidates(previous_routes, prefetch_ranks))
            .unwrap_or_default();
        Cow::Owned(transition_candidates)
    } else {
        Cow::Borrowed(&previous_routes[..previous_routes.len().min(prefetch_ranks)])
    };

    for &expert_idx in candidate_experts.iter() {
        let gate_up = expert_key(layer_idx, expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, expert_idx, MoeExpertProjection::Down);
        if mode.resident_only() && !(manager.is_resident(gate_up) && manager.is_resident(down)) {
            continue;
        }
        manager.prefetch_resident(store, gate_up)?;
        manager.prefetch_resident(store, down)?;
    }
    Ok(())
}

fn ensure_demand_routes(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    previous_topk_by_layer: &[Vec<usize>],
    next_topk_by_layer: &mut [Vec<usize>],
    track_routes: bool,
    route_telemetry: Option<&mut MoeRouteTelemetry>,
    transition_predictors: Option<&mut [MoeTransitionPredictor]>,
    layer_idx: usize,
    routes: &[ExpertRoute],
) -> Result<()> {
    let previous_routes = previous_topk_by_layer
        .get(layer_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if let Some(route_telemetry) = route_telemetry {
        route_telemetry.record(manager, layer_idx, routes, previous_routes);
    }
    if let Some(predictors) = transition_predictors {
        if let Some(predictor) = predictors.get_mut(layer_idx) {
            predictor.update(routes, previous_routes);
        }
    }
    for route in routes {
        let expert_idx = route.expert_idx;
        let gate_up = expert_key(layer_idx, expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, expert_idx, MoeExpertProjection::Down);
        manager.ensure_resident(store, gate_up)?;
        manager.ensure_resident(store, down)?;
        if previous_routes
            .iter()
            .any(|&previous_expert| previous_expert == expert_idx)
        {
            manager.protect_resident(gate_up)?;
            manager.protect_resident(down)?;
        }
    }
    if track_routes {
        if let Some(slot) = next_topk_by_layer.get_mut(layer_idx) {
            slot.clear();
            slot.extend(routes.iter().map(|route| route.expert_idx));
        }
    }
    Ok(())
}

fn expert_key(
    layer_idx: usize,
    expert_idx: usize,
    projection: MoeExpertProjection,
) -> MoeExpertKey {
    MoeExpertKey {
        layer_idx,
        expert_idx,
        projection,
    }
}
