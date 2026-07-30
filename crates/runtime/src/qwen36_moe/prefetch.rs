use anyhow::Result;
use std::collections::HashMap;

use model_store::BakedStore;

use crate::qwen36_moe::residency::{MoeExpertKey, MoeExpertProjection, MoeExpertResidencyManager};
use crate::qwen36_moe::route_telemetry::{MoeRouteTelemetry, MoeTransitionPredictor};
use crate::qwen36_moe::types::{ExpertPrefetchPhase, ExpertRoute};
use crate::qwen36_moe_config::MoeIslandPrefetchMode;

pub fn handle_moe_expert_prefetch(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    prefetch_evict_min_probability: f64,
    protect_demand_routes: bool,
    hot_protect_min_hits: Option<u32>,
    fixed_hot_min_hits: Option<u32>,
    previous_topk_by_layer: &[Vec<usize>],
    next_topk_by_layer: &mut [Vec<usize>],
    track_routes: bool,
    route_telemetry: Option<&mut MoeRouteTelemetry>,
    transition_predictors: Option<&mut [MoeTransitionPredictor]>,
    hot_expert_counts: Option<&mut [HashMap<usize, u32>]>,
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
                prefetch_evict_min_probability,
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
                protect_demand_routes,
                hot_protect_min_hits,
                fixed_hot_min_hits,
                hot_expert_counts,
                layer_idx,
                routes,
            )?;
        }
    }
    Ok(())
}

fn record_moe_route_telemetry(
    telemetry: &mut MoeRouteTelemetry,
    manager: &MoeExpertResidencyManager,
    layer_idx: usize,
    routes: &[ExpertRoute],
    previous_routes: &[usize],
) {
    for route in routes {
        if route.rank >= telemetry.observations_by_rank.len() {
            continue;
        }
        telemetry.record_route_observation(route, previous_routes);
        let gate_up = expert_key(layer_idx, route.expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, route.expert_idx, MoeExpertProjection::Down);
        if manager.is_resident(gate_up) && manager.is_resident(down) {
            telemetry.record_resident_before(route.rank);
        }
    }
}

fn prefetch_previous_token_routes(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    prefetch_evict_min_probability: f64,
    previous_topk_by_layer: &[Vec<usize>],
    transition_predictors: Option<&mut [MoeTransitionPredictor]>,
    layer_idx: usize,
) -> Result<()> {
    let previous_routes = previous_topk_by_layer
        .get(layer_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let candidate_experts: Vec<(usize, bool)> = if mode.transition_weighted() {
        transition_predictors
            .as_ref()
            .and_then(|predictors| predictors.get(layer_idx))
            .map(|predictor| {
                predictor
                    .scored_candidates(previous_routes, prefetch_ranks)
                    .into_iter()
                    .map(|candidate| {
                        (
                            candidate.expert_idx,
                            candidate.reuse_probability() >= prefetch_evict_min_probability,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        previous_routes[..previous_routes.len().min(prefetch_ranks)]
            .iter()
            .copied()
            .map(|expert_idx| (expert_idx, true))
            .collect()
    };

    for (expert_idx, allow_evict) in candidate_experts {
        let gate_up = expert_key(layer_idx, expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, expert_idx, MoeExpertProjection::Down);
        if mode.resident_only() && !(manager.is_resident(gate_up) && manager.is_resident(down)) {
            continue;
        }
        manager.prefetch_resident_with_evict(store, gate_up, allow_evict)?;
        manager.prefetch_resident_with_evict(store, down, allow_evict)?;
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
    protect_demand_routes: bool,
    hot_protect_min_hits: Option<u32>,
    fixed_hot_min_hits: Option<u32>,
    hot_expert_counts: Option<&mut [HashMap<usize, u32>]>,
    layer_idx: usize,
    routes: &[ExpertRoute],
) -> Result<()> {
    let previous_routes = previous_topk_by_layer
        .get(layer_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if let Some(route_telemetry) = route_telemetry {
        record_moe_route_telemetry(route_telemetry, manager, layer_idx, routes, previous_routes);
    }
    if let Some(predictors) = transition_predictors {
        if let Some(predictor) = predictors.get_mut(layer_idx) {
            predictor.update(routes, previous_routes);
        }
    }
    let mut route_hit_counts = vec![0u32; routes.len()];
    let hot_min_hits = match (hot_protect_min_hits, fixed_hot_min_hits) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    if let (Some(min_hits), Some(counts_by_layer)) = (hot_min_hits, hot_expert_counts) {
        if let Some(counts) = counts_by_layer.get_mut(layer_idx) {
            for (idx, route) in routes.iter().enumerate() {
                let count = counts
                    .entry(route.expert_idx)
                    .and_modify(|count| *count = count.saturating_add(1))
                    .or_insert(1);
                if *count >= min_hits {
                    route_hit_counts[idx] = *count;
                }
            }
        }
    }
    for (route_idx, route) in routes.iter().enumerate() {
        let expert_idx = route.expert_idx;
        let gate_up = expert_key(layer_idx, expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, expert_idx, MoeExpertProjection::Down);
        manager.ensure_resident(store, gate_up)?;
        manager.ensure_resident(store, down)?;
        if fixed_hot_min_hits
            .map(|min_hits| route_hit_counts.get(route_idx).copied().unwrap_or(0) >= min_hits)
            .unwrap_or(false)
        {
            manager.mark_fixed_hot_resident(gate_up)?;
            manager.mark_fixed_hot_resident(down)?;
        }
        if protect_demand_routes
            || (hot_protect_min_hits.is_some()
                && hot_protect_min_hits
                    .map(|min_hits| {
                        route_hit_counts.get(route_idx).copied().unwrap_or(0) >= min_hits
                    })
                    .unwrap_or(false))
            || previous_routes
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
