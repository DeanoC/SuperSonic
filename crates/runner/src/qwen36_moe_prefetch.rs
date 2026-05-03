use anyhow::Result;
use model_store::BakedStore;

use crate::qwen36_moe_decode::{ExpertPrefetchPhase, ExpertRoute};
use crate::qwen36_moe_residency::{MoeExpertKey, MoeExpertProjection, MoeExpertResidencyManager};
use crate::qwen36_moe_telemetry::{MoeIslandPrefetchMode, MoeRouteTelemetry};

pub(crate) fn handle_moe_expert_prefetch(
    manager: &mut MoeExpertResidencyManager,
    store: &BakedStore,
    mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    previous_topk_by_layer: &[Vec<usize>],
    next_topk_by_layer: &mut [Vec<usize>],
    track_routes: bool,
    route_telemetry: Option<&mut MoeRouteTelemetry>,
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
    layer_idx: usize,
) -> Result<()> {
    for &expert_idx in previous_topk_by_layer
        .get(layer_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .take(prefetch_ranks)
    {
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
    for route in routes {
        let expert_idx = route.expert_idx;
        let gate_up = expert_key(layer_idx, expert_idx, MoeExpertProjection::GateUp);
        let down = expert_key(layer_idx, expert_idx, MoeExpertProjection::Down);
        manager.ensure_resident(store, gate_up)?;
        manager.ensure_resident(store, down)?;
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
