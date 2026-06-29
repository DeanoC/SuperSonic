#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_residency_types::{MoeExpertKey, MoeExpertProjection};
use crate::qwen36_moe_types::ExpertRoute;

const MIB: f64 = (1024 * 1024) as f64;

pub(crate) use supersonic_runtime::qwen36_moe::route_telemetry::{
    MoeRouteTelemetry, MoeTransitionPredictor,
};
pub(crate) use supersonic_runtime::qwen36_moe_config::MoeIslandPrefetchMode;

pub(crate) fn print_and_write_moe_residency_summary(
    manager: &MoeExpertResidencyManager,
    virtual_kv_stats: VirtualKvStats,
    generated_ids: &[u32],
    route_telemetry: Option<&MoeRouteTelemetry>,
    telemetry: Option<&MoeSparseTelemetry>,
) -> Result<()> {
    let residency = manager.stats();
    let arena = manager.arena().stats();
    let total_resident_bytes = arena.resident_bytes + virtual_kv_stats.resident_bytes;
    let total_reserved_bytes = arena.reserved_bytes + virtual_kv_stats.reserved_bytes;
    if let Some(telemetry) = telemetry {
        println!(
            "  [vmm] MoE island residency: resident_slices={} peak_slices={} \
             resident_pages={} peak_pages={} page_backed_slices={} \
             hits={} misses={} page_hits={} page_misses={} evicted_slices={} evicted_pages={} \
             prefetch_requests={} prefetch_hits={} prefetch_misses={} \
             prefetch_page_hits={} prefetch_page_misses={} \
             prefetch_skipped={} prefetch_skipped_pages={} \
             prefetch_evicted_pages={} \
             fixed_hot_pages={} fixed_hot_page_budget={} fixed_hot_requests={} \
             fixed_hot_hits={} fixed_hot_misses={} fixed_hot_skipped={} \
             fixed_hot_evicted_pages={} \
             uploaded={:.2}MiB unmapped={:.2}MiB \
             resident={:.2}MiB peak_resident={:.2}MiB reserved={:.2}MiB \
             kv_resident={:.2}MiB total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
            residency.resident_slices,
            telemetry.peak_resident_slices,
            residency.resident_pages,
            telemetry.peak_resident_pages,
            residency.page_backed_slices,
            residency.hits,
            residency.misses,
            residency.page_hits,
            residency.page_misses,
            residency.evicted_slices,
            residency.evicted_pages,
            residency.prefetch_requests,
            residency.prefetch_hits,
            residency.prefetch_misses,
            residency.prefetch_page_hits,
            residency.prefetch_page_misses,
            residency.prefetch_skipped,
            residency.prefetch_skipped_pages,
            residency.prefetch_evicted_pages,
            residency.fixed_hot_pages,
            residency.fixed_hot_page_budget,
            residency.fixed_hot_requests,
            residency.fixed_hot_hits,
            residency.fixed_hot_misses,
            residency.fixed_hot_skipped,
            residency.fixed_hot_evicted_pages,
            residency.uploaded_bytes as f64 / MIB,
            residency.unmapped_bytes as f64 / MIB,
            arena.resident_bytes as f64 / MIB,
            telemetry.peak_resident_bytes as f64 / MIB,
            arena.reserved_bytes as f64 / MIB,
            virtual_kv_stats.resident_bytes as f64 / MIB,
            total_resident_bytes as f64 / MIB,
            total_reserved_bytes as f64 / MIB,
        );
        telemetry.write_json(manager, virtual_kv_stats, generated_ids, route_telemetry)?;
    } else {
        println!(
            "  [vmm] MoE island residency: resident_slices={} resident_pages={} \
             page_backed_slices={} hits={} misses={} page_hits={} page_misses={} \
             evicted_slices={} evicted_pages={} prefetch_requests={} \
             prefetch_hits={} prefetch_misses={} prefetch_page_hits={} \
             prefetch_page_misses={} prefetch_skipped={} prefetch_skipped_pages={} \
             prefetch_evicted_pages={} \
             fixed_hot_pages={} fixed_hot_page_budget={} fixed_hot_requests={} \
             fixed_hot_hits={} fixed_hot_misses={} fixed_hot_skipped={} \
             fixed_hot_evicted_pages={} \
             uploaded={:.2}MiB unmapped={:.2}MiB \
             resident={:.2}MiB reserved={:.2}MiB kv_resident={:.2}MiB \
             total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
            residency.resident_slices,
            residency.resident_pages,
            residency.page_backed_slices,
            residency.hits,
            residency.misses,
            residency.page_hits,
            residency.page_misses,
            residency.evicted_slices,
            residency.evicted_pages,
            residency.prefetch_requests,
            residency.prefetch_hits,
            residency.prefetch_misses,
            residency.prefetch_page_hits,
            residency.prefetch_page_misses,
            residency.prefetch_skipped,
            residency.prefetch_skipped_pages,
            residency.prefetch_evicted_pages,
            residency.fixed_hot_pages,
            residency.fixed_hot_page_budget,
            residency.fixed_hot_requests,
            residency.fixed_hot_hits,
            residency.fixed_hot_misses,
            residency.fixed_hot_skipped,
            residency.fixed_hot_evicted_pages,
            residency.uploaded_bytes as f64 / MIB,
            residency.unmapped_bytes as f64 / MIB,
            arena.resident_bytes as f64 / MIB,
            arena.reserved_bytes as f64 / MIB,
            virtual_kv_stats.resident_bytes as f64 / MIB,
            total_resident_bytes as f64 / MIB,
            total_reserved_bytes as f64 / MIB,
        );
    }
    Ok(())
}

pub(crate) fn record_moe_route_telemetry(
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
        let gate_up = MoeExpertKey {
            layer_idx,
            expert_idx: route.expert_idx,
            projection: MoeExpertProjection::GateUp,
        };
        let down = MoeExpertKey {
            layer_idx,
            expert_idx: route.expert_idx,
            projection: MoeExpertProjection::Down,
        };
        if manager.is_resident(gate_up) && manager.is_resident(down) {
            telemetry.record_resident_before(route.rank);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MoeIslandPrefetchMode, MoeRouteTelemetry, MoeTransitionPredictor};
    use crate::qwen36_moe_types::ExpertRoute;

    #[test]
    fn moe_prefetch_mode_env_accepts_disabled_and_previous_token_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(None).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("disabled")).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("prev-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_resident_only_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token-resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous_token_resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("resident-previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert!(MoeIslandPrefetchMode::from_env_value(Some("resident")).is_err());
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_transition_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition-weighted")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(MoeIslandPrefetchMode::Transition.as_str(), "transition");
        assert!(MoeIslandPrefetchMode::Transition.uses_previous_token_routes());
        assert!(MoeIslandPrefetchMode::Transition.transition_weighted());
    }

    #[test]
    fn moe_transition_predictor_waits_for_warmup_and_scores_repeats() {
        let mut predictor = MoeTransitionPredictor::new(3, 2);
        let previous_routes = [10, 20, 30];
        let routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 20,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 99,
                weight: 0.25,
            },
        ];

        predictor.update(&routes, &previous_routes);
        assert!(predictor.candidates(&previous_routes, 2).is_empty());

        predictor.update(&routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20]);
        let scored = predictor.scored_candidates(&previous_routes, 2);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].expert_idx, 20);
        assert_eq!(scored[0].observations, 2);
        assert_eq!(scored[0].repeats, 2);
        assert_eq!(scored[0].reuse_probability(), 1.0);

        let later_routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 10,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 20,
                weight: 0.25,
            },
        ];
        predictor.update(&later_routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20, 10]);
        let scored = predictor.scored_candidates(&previous_routes, 2);
        assert_eq!(scored[0].expert_idx, 20);
        assert!(scored[0].reuse_probability() > scored[1].reuse_probability());
    }

    #[test]
    fn moe_route_telemetry_records_previous_rank_transition_matrix() {
        let mut telemetry = MoeRouteTelemetry::new(3);
        let previous_routes = [7, 11, 13];
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 0,
                expert_idx: 11,
                weight: 0.5,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 1,
                expert_idx: 7,
                weight: 0.25,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 2,
                expert_idx: 99,
                weight: 0.125,
            },
            &previous_routes,
        );

        assert_eq!(telemetry.observations_by_rank, vec![1, 1, 1]);
        assert_eq!(telemetry.repeated_previous_by_rank, vec![1, 1, 0]);
        assert_eq!(
            telemetry.repeated_previous_rank_by_current_rank,
            vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]]
        );
        assert_eq!(
            telemetry
                .to_json()
                .get("repeated_previous_rank_by_current_rank")
                .unwrap(),
            &serde_json::json!([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        );
        let json = telemetry.to_json();
        assert_eq!(
            json.get("repeated_previous_probability_by_current_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("same_rank_repeat_probability_by_rank").unwrap(),
            &serde_json::json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            json.get("repeated_current_by_previous_rank").unwrap(),
            &serde_json::json!([1, 1, 0])
        );
        assert_eq!(
            json.get("repeated_current_probability_by_previous_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("best_previous_rank_by_current_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_current_rank_by_previous_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_transition").unwrap(),
            &serde_json::json!({
                "current_rank": 0,
                "previous_rank": 1,
                "count": 1,
                "probability_by_current_rank": 1.0,
            })
        );
    }
}

pub(crate) struct MoeRouteRuntime {
    pub(crate) previous_topk_by_layer: Vec<Vec<usize>>,
    pub(crate) route_telemetry: Option<MoeRouteTelemetry>,
    pub(crate) transition_predictors: Option<Vec<MoeTransitionPredictor>>,
    pub(crate) hot_expert_counts: Option<Vec<HashMap<usize, u32>>>,
}

impl MoeRouteRuntime {
    pub(crate) fn new(
        num_layers: usize,
        top_k: usize,
        sparse_moe_requested: bool,
        prefetch_mode: MoeIslandPrefetchMode,
        transition_min_observations: u32,
        hot_protect_min_hits: Option<u32>,
        fixed_hot_min_hits: Option<u32>,
    ) -> Self {
        let route_telemetry = sparse_moe_requested.then(|| MoeRouteTelemetry::new(top_k));
        let transition_predictors = prefetch_mode.transition_weighted().then(|| {
            vec![MoeTransitionPredictor::new(top_k, transition_min_observations); num_layers]
        });
        let hot_expert_counts = (hot_protect_min_hits.is_some() || fixed_hot_min_hits.is_some())
            .then(|| vec![HashMap::<usize, u32>::new(); num_layers]);
        Self {
            previous_topk_by_layer: vec![Vec::new(); num_layers],
            route_telemetry,
            transition_predictors,
            hot_expert_counts,
        }
    }

    pub(crate) fn should_track_routes(&self, prefetch_mode: MoeIslandPrefetchMode) -> bool {
        prefetch_mode.uses_previous_token_routes() || self.route_telemetry.is_some()
    }

    pub(crate) fn next_topk_buffer(&self, track_routes: bool) -> Vec<Vec<usize>> {
        if track_routes {
            self.previous_topk_by_layer.clone()
        } else {
            Vec::new()
        }
    }

    pub(crate) fn advance(&mut self, track_routes: bool, next_topk_by_layer: Vec<Vec<usize>>) {
        if track_routes {
            self.previous_topk_by_layer = next_topk_by_layer;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VirtualKvStats {
    pub(crate) layers: usize,
    pub(crate) logical_bytes: usize,
    pub(crate) reserved_bytes: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) logical_resident_bytes: usize,
    pub(crate) mappings: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeSparseTelemetrySnapshot {
    stats: crate::qwen36_moe_residency_types::MoeExpertResidencyStats,
    arena: gpu_hal::VirtualArenaStats,
}

impl MoeSparseTelemetrySnapshot {
    pub(crate) fn capture(manager: &MoeExpertResidencyManager) -> Self {
        Self {
            stats: manager.stats(),
            arena: manager.arena().stats(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct MoeSparseTelemetry {
    pub(crate) dump_path: Option<PathBuf>,
    decode_path: &'static str,
    prefetch_mode: MoeIslandPrefetchMode,
    prefetch_ranks: usize,
    steps: Vec<serde_json::Value>,
    pub(crate) peak_resident_slices: usize,
    pub(crate) peak_resident_pages: usize,
    peak_page_backed_slices: usize,
    pub(crate) peak_resident_bytes: usize,
    peak_logical_resident_bytes: usize,
}

impl MoeSparseTelemetry {
    pub(crate) fn from_env(
        active: bool,
        persistent_decode: bool,
        prefetch_mode: MoeIslandPrefetchMode,
        prefetch_ranks: usize,
    ) -> Result<Option<Self>> {
        if !active {
            return Ok(None);
        }
        let dump_path = std::env::var_os("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON").map(PathBuf::from);
        Ok(Some(Self {
            dump_path,
            decode_path: if persistent_decode {
                "segmented_persistent"
            } else {
                "chained"
            },
            prefetch_mode,
            prefetch_ranks,
            steps: Vec::new(),
            peak_resident_slices: 0,
            peak_resident_pages: 0,
            peak_page_backed_slices: 0,
            peak_resident_bytes: 0,
            peak_logical_resident_bytes: 0,
        }))
    }

    pub(crate) fn record_step(
        &mut self,
        step: usize,
        position: i32,
        is_generation_step: bool,
        before: MoeSparseTelemetrySnapshot,
        after: MoeSparseTelemetrySnapshot,
    ) {
        self.peak_resident_slices = self.peak_resident_slices.max(after.stats.resident_slices);
        self.peak_resident_pages = self.peak_resident_pages.max(after.stats.resident_pages);
        self.peak_page_backed_slices = self
            .peak_page_backed_slices
            .max(after.stats.page_backed_slices);
        self.peak_resident_bytes = self.peak_resident_bytes.max(after.arena.resident_bytes);
        self.peak_logical_resident_bytes = self
            .peak_logical_resident_bytes
            .max(after.arena.logical_resident_bytes);

        if self.dump_path.is_none() {
            return;
        }

        self.steps.push(serde_json::json!({
            "step": step,
            "position": position,
            "kind": if is_generation_step { "generate" } else { "prefill" },
            "delta": {
                "hits": after.stats.hits.saturating_sub(before.stats.hits),
                "misses": after.stats.misses.saturating_sub(before.stats.misses),
                "page_hits": after.stats.page_hits.saturating_sub(before.stats.page_hits),
                "page_misses": after.stats.page_misses.saturating_sub(before.stats.page_misses),
                "evicted_slices": after.stats.evicted_slices.saturating_sub(before.stats.evicted_slices),
                "evicted_pages": after.stats.evicted_pages.saturating_sub(before.stats.evicted_pages),
                "uploaded_bytes": after.stats.uploaded_bytes.saturating_sub(before.stats.uploaded_bytes),
                "unmapped_bytes": after.stats.unmapped_bytes.saturating_sub(before.stats.unmapped_bytes),
                "prefetch_requests": after.stats.prefetch_requests.saturating_sub(before.stats.prefetch_requests),
                "prefetch_hits": after.stats.prefetch_hits.saturating_sub(before.stats.prefetch_hits),
                "prefetch_misses": after.stats.prefetch_misses.saturating_sub(before.stats.prefetch_misses),
                "prefetch_page_hits": after.stats.prefetch_page_hits.saturating_sub(before.stats.prefetch_page_hits),
                "prefetch_page_misses": after.stats.prefetch_page_misses.saturating_sub(before.stats.prefetch_page_misses),
                "prefetch_skipped": after.stats.prefetch_skipped.saturating_sub(before.stats.prefetch_skipped),
                "prefetch_skipped_pages": after.stats.prefetch_skipped_pages.saturating_sub(before.stats.prefetch_skipped_pages),
                "prefetch_evicted_pages": after.stats.prefetch_evicted_pages.saturating_sub(before.stats.prefetch_evicted_pages),
                "prefetch_uploaded_bytes": after.stats.prefetch_uploaded_bytes.saturating_sub(before.stats.prefetch_uploaded_bytes),
                "protect_requests": after.stats.protect_requests.saturating_sub(before.stats.protect_requests),
                "protect_hits": after.stats.protect_hits.saturating_sub(before.stats.protect_hits),
                "protect_misses": after.stats.protect_misses.saturating_sub(before.stats.protect_misses),
                "protect_demotions": after.stats.protect_demotions.saturating_sub(before.stats.protect_demotions),
                "protected_evicted_pages": after.stats.protected_evicted_pages.saturating_sub(before.stats.protected_evicted_pages),
                "fixed_hot_requests": after.stats.fixed_hot_requests.saturating_sub(before.stats.fixed_hot_requests),
                "fixed_hot_hits": after.stats.fixed_hot_hits.saturating_sub(before.stats.fixed_hot_hits),
                "fixed_hot_misses": after.stats.fixed_hot_misses.saturating_sub(before.stats.fixed_hot_misses),
                "fixed_hot_skipped": after.stats.fixed_hot_skipped.saturating_sub(before.stats.fixed_hot_skipped),
                "fixed_hot_evicted_pages": after.stats.fixed_hot_evicted_pages.saturating_sub(before.stats.fixed_hot_evicted_pages),
                "async_scheduled_pages": after.stats.async_scheduled_pages.saturating_sub(before.stats.async_scheduled_pages),
                "async_completed_pages": after.stats.async_completed_pages.saturating_sub(before.stats.async_completed_pages),
                "async_waited_pages": after.stats.async_waited_pages.saturating_sub(before.stats.async_waited_pages),
                "async_skipped_no_slot": after.stats.async_skipped_no_slot.saturating_sub(before.stats.async_skipped_no_slot),
                "async_skipped_no_capacity": after.stats.async_skipped_no_capacity.saturating_sub(before.stats.async_skipped_no_capacity),
                "async_uploaded_bytes": after.stats.async_uploaded_bytes.saturating_sub(before.stats.async_uploaded_bytes),
            },
            "resident": {
                "slices": after.stats.resident_slices,
                "pages": after.stats.resident_pages,
                "protected_pages": after.stats.protected_pages,
                "protected_page_budget": after.stats.protected_page_budget,
                "fixed_hot_pages": after.stats.fixed_hot_pages,
                "fixed_hot_page_budget": after.stats.fixed_hot_page_budget,
                "page_backed_slices": after.stats.page_backed_slices,
                "logical_bytes": after.arena.logical_resident_bytes,
                "physical_bytes": after.arena.resident_bytes,
                "mappings": after.arena.mapping_count,
            },
            "cumulative": {
                "hits": after.stats.hits,
                "misses": after.stats.misses,
                "page_hits": after.stats.page_hits,
                "page_misses": after.stats.page_misses,
                "evicted_slices": after.stats.evicted_slices,
                "evicted_pages": after.stats.evicted_pages,
                "uploaded_bytes": after.stats.uploaded_bytes,
                "unmapped_bytes": after.stats.unmapped_bytes,
                "prefetch_requests": after.stats.prefetch_requests,
                "prefetch_hits": after.stats.prefetch_hits,
                "prefetch_misses": after.stats.prefetch_misses,
                "prefetch_page_hits": after.stats.prefetch_page_hits,
                "prefetch_page_misses": after.stats.prefetch_page_misses,
                "prefetch_skipped": after.stats.prefetch_skipped,
                "prefetch_skipped_pages": after.stats.prefetch_skipped_pages,
                "prefetch_evicted_pages": after.stats.prefetch_evicted_pages,
                "prefetch_uploaded_bytes": after.stats.prefetch_uploaded_bytes,
                "protected_pages": after.stats.protected_pages,
                "protected_page_budget": after.stats.protected_page_budget,
                "protect_requests": after.stats.protect_requests,
                "protect_hits": after.stats.protect_hits,
                "protect_misses": after.stats.protect_misses,
                "protect_demotions": after.stats.protect_demotions,
                "protected_evicted_pages": after.stats.protected_evicted_pages,
                "fixed_hot_pages": after.stats.fixed_hot_pages,
                "fixed_hot_page_budget": after.stats.fixed_hot_page_budget,
                "fixed_hot_requests": after.stats.fixed_hot_requests,
                "fixed_hot_hits": after.stats.fixed_hot_hits,
                "fixed_hot_misses": after.stats.fixed_hot_misses,
                "fixed_hot_skipped": after.stats.fixed_hot_skipped,
                "fixed_hot_evicted_pages": after.stats.fixed_hot_evicted_pages,
                "async_scheduled_pages": after.stats.async_scheduled_pages,
                "async_completed_pages": after.stats.async_completed_pages,
                "async_waited_pages": after.stats.async_waited_pages,
                "async_skipped_no_slot": after.stats.async_skipped_no_slot,
                "async_skipped_no_capacity": after.stats.async_skipped_no_capacity,
                "async_uploaded_bytes": after.stats.async_uploaded_bytes,
                "async_pending_pages_peak": after.stats.async_pending_pages_peak,
            }
        }));
    }

    pub(crate) fn write_json(
        &self,
        manager: &MoeExpertResidencyManager,
        virtual_kv_stats: VirtualKvStats,
        generated_ids: &[u32],
        route_telemetry: Option<&MoeRouteTelemetry>,
    ) -> Result<()> {
        let Some(path) = self.dump_path.as_ref() else {
            return Ok(());
        };
        let final_snapshot = MoeSparseTelemetrySnapshot::capture(manager);
        let total_vmm_logical_bytes =
            final_snapshot.arena.logical_bytes + virtual_kv_stats.logical_bytes;
        let total_vmm_logical_resident_bytes =
            final_snapshot.arena.logical_resident_bytes + virtual_kv_stats.logical_resident_bytes;
        let total_vmm_resident_bytes =
            final_snapshot.arena.resident_bytes + virtual_kv_stats.resident_bytes;
        let total_vmm_reserved_bytes =
            final_snapshot.arena.reserved_bytes + virtual_kv_stats.reserved_bytes;
        let total_vmm_mappings = final_snapshot.arena.mapping_count + virtual_kv_stats.mappings;
        let payload = serde_json::json!({
            "schema": "supersonic-qwen36-moe-sparse-vmm-telemetry-v1",
            "summary": {
                "decode_path": self.decode_path,
                "prefetch_mode": self.prefetch_mode.as_str(),
                "prefetch_ranks": self.prefetch_ranks,
                "registered_tensors": final_snapshot.stats.registered_tensors,
                "max_resident_pages": manager.max_resident_pages(),
                "max_protected_pages": manager.max_protected_pages(),
                "max_fixed_hot_pages": manager.max_fixed_hot_pages(),
                "final_resident_slices": final_snapshot.stats.resident_slices,
                "final_resident_pages": final_snapshot.stats.resident_pages,
                "final_protected_pages": final_snapshot.stats.protected_pages,
                "final_fixed_hot_pages": final_snapshot.stats.fixed_hot_pages,
                "final_page_backed_slices": final_snapshot.stats.page_backed_slices,
                "peak_resident_slices": self.peak_resident_slices,
                "peak_resident_pages": self.peak_resident_pages,
                "peak_page_backed_slices": self.peak_page_backed_slices,
                "peak_resident_bytes": self.peak_resident_bytes,
                "peak_logical_resident_bytes": self.peak_logical_resident_bytes,
                "reserved_bytes": final_snapshot.arena.reserved_bytes,
                "logical_bytes": final_snapshot.arena.logical_bytes,
                "moe_logical_bytes": final_snapshot.arena.logical_bytes,
                "moe_logical_resident_bytes": final_snapshot.arena.logical_resident_bytes,
                "moe_resident_bytes": final_snapshot.arena.resident_bytes,
                "moe_reserved_bytes": final_snapshot.arena.reserved_bytes,
                "moe_mappings": final_snapshot.arena.mapping_count,
                "kv_layers": virtual_kv_stats.layers,
                "kv_mappings": virtual_kv_stats.mappings,
                "kv_logical_bytes": virtual_kv_stats.logical_bytes,
                "kv_logical_resident_bytes": virtual_kv_stats.logical_resident_bytes,
                "kv_resident_bytes": virtual_kv_stats.resident_bytes,
                "kv_reserved_bytes": virtual_kv_stats.reserved_bytes,
                "total_vmm_logical_bytes": total_vmm_logical_bytes,
                "total_vmm_logical_resident_bytes": total_vmm_logical_resident_bytes,
                "total_vmm_resident_bytes": total_vmm_resident_bytes,
                "total_vmm_reserved_bytes": total_vmm_reserved_bytes,
                "total_vmm_mappings": total_vmm_mappings,
                "hits": final_snapshot.stats.hits,
                "misses": final_snapshot.stats.misses,
                "page_hits": final_snapshot.stats.page_hits,
                "page_misses": final_snapshot.stats.page_misses,
                "evicted_slices": final_snapshot.stats.evicted_slices,
                "evicted_pages": final_snapshot.stats.evicted_pages,
                "uploaded_bytes": final_snapshot.stats.uploaded_bytes,
                "unmapped_bytes": final_snapshot.stats.unmapped_bytes,
                "prefetch_requests": final_snapshot.stats.prefetch_requests,
                "prefetch_hits": final_snapshot.stats.prefetch_hits,
                "prefetch_misses": final_snapshot.stats.prefetch_misses,
                "prefetch_page_hits": final_snapshot.stats.prefetch_page_hits,
                "prefetch_page_misses": final_snapshot.stats.prefetch_page_misses,
                "prefetch_skipped": final_snapshot.stats.prefetch_skipped,
                "prefetch_skipped_pages": final_snapshot.stats.prefetch_skipped_pages,
                "prefetch_evicted_pages": final_snapshot.stats.prefetch_evicted_pages,
                "prefetch_uploaded_bytes": final_snapshot.stats.prefetch_uploaded_bytes,
                "protected_pages": final_snapshot.stats.protected_pages,
                "protected_page_budget": final_snapshot.stats.protected_page_budget,
                "protect_requests": final_snapshot.stats.protect_requests,
                "protect_hits": final_snapshot.stats.protect_hits,
                "protect_misses": final_snapshot.stats.protect_misses,
                "protect_demotions": final_snapshot.stats.protect_demotions,
                "protected_evicted_pages": final_snapshot.stats.protected_evicted_pages,
                "fixed_hot_pages": final_snapshot.stats.fixed_hot_pages,
                "fixed_hot_page_budget": final_snapshot.stats.fixed_hot_page_budget,
                "fixed_hot_requests": final_snapshot.stats.fixed_hot_requests,
                "fixed_hot_hits": final_snapshot.stats.fixed_hot_hits,
                "fixed_hot_misses": final_snapshot.stats.fixed_hot_misses,
                "fixed_hot_skipped": final_snapshot.stats.fixed_hot_skipped,
                "fixed_hot_evicted_pages": final_snapshot.stats.fixed_hot_evicted_pages,
                "async_scheduled_pages": final_snapshot.stats.async_scheduled_pages,
                "async_completed_pages": final_snapshot.stats.async_completed_pages,
                "async_waited_pages": final_snapshot.stats.async_waited_pages,
                "async_skipped_no_slot": final_snapshot.stats.async_skipped_no_slot,
                "async_skipped_no_capacity": final_snapshot.stats.async_skipped_no_capacity,
                "async_uploaded_bytes": final_snapshot.stats.async_uploaded_bytes,
                "async_pending_pages_peak": final_snapshot.stats.async_pending_pages_peak,
                "route_summary": route_telemetry.map(MoeRouteTelemetry::to_json),
            },
            "generated_ids": generated_ids,
            "steps": self.steps,
        });
        let bytes = serde_json::to_vec_pretty(&payload)?;
        std::fs::write(path, bytes)
            .with_context(|| format!("write MoE sparse VMM telemetry to {}", path.display()))
    }
}
