#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::qwen36_moe_decode::ExpertRoute;
use crate::qwen36_moe_residency::{MoeExpertKey, MoeExpertProjection, MoeExpertResidencyManager};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoeIslandPrefetchMode {
    Disabled,
    PreviousToken,
    PreviousTokenResidentOnly,
    Transition,
}

impl MoeIslandPrefetchMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PreviousToken => "previous-token",
            Self::PreviousTokenResidentOnly => "previous-token-resident",
            Self::Transition => "transition",
        }
    }

    pub(crate) fn uses_previous_token_routes(self) -> bool {
        matches!(
            self,
            Self::PreviousToken | Self::PreviousTokenResidentOnly | Self::Transition
        )
    }

    pub(crate) fn resident_only(self) -> bool {
        matches!(self, Self::PreviousTokenResidentOnly)
    }

    pub(crate) fn transition_weighted(self) -> bool {
        matches!(self, Self::Transition)
    }

    pub(crate) fn from_env() -> Result<Self> {
        Self::from_env_value(
            std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH")
                .ok()
                .as_deref(),
        )
    }

    pub(crate) fn from_env_value(raw: Option<&str>) -> Result<Self> {
        match raw {
            None | Some("0") | Some("off") | Some("disabled") => Ok(Self::Disabled),
            Some("previous-token") | Some("previous_token") | Some("prev-token") => {
                Ok(Self::PreviousToken)
            }
            Some("previous-token-resident")
            | Some("previous_token_resident")
            | Some("prev-token-resident")
            | Some("resident-previous-token") => Ok(Self::PreviousTokenResidentOnly),
            Some("transition") | Some("transition-weighted") | Some("transition_weighted") => {
                Ok(Self::Transition)
            }
            Some(other) => anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_PREFETCH must be unset, 0, off, disabled, \
                 previous-token, previous_token, prev-token, previous-token-resident, \
                 previous_token_resident, prev-token-resident, resident-previous-token, \
                 transition, transition-weighted, or transition_weighted; \
                 got {other:?}"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MoeTransitionPredictor {
    top_k: usize,
    min_observations: u32,
    observations_by_previous_rank: Vec<u32>,
    repeated_current_by_previous_rank: Vec<u32>,
}

impl MoeTransitionPredictor {
    pub(crate) fn new(top_k: usize, min_observations: u32) -> Self {
        Self {
            top_k,
            min_observations,
            observations_by_previous_rank: vec![0; top_k],
            repeated_current_by_previous_rank: vec![0; top_k],
        }
    }

    pub(crate) fn update(&mut self, routes: &[ExpertRoute], previous_routes: &[usize]) {
        for (previous_rank, &expert_idx) in previous_routes.iter().take(self.top_k).enumerate() {
            self.observations_by_previous_rank[previous_rank] =
                self.observations_by_previous_rank[previous_rank].saturating_add(1);
            if routes.iter().any(|route| route.expert_idx == expert_idx) {
                self.repeated_current_by_previous_rank[previous_rank] =
                    self.repeated_current_by_previous_rank[previous_rank].saturating_add(1);
            }
        }
    }

    pub(crate) fn candidates(&self, previous_routes: &[usize], limit: usize) -> Vec<usize> {
        let mut scored = Vec::new();
        for (previous_rank, &expert_idx) in previous_routes.iter().take(self.top_k).enumerate() {
            let observations = self.observations_by_previous_rank[previous_rank];
            if observations < self.min_observations {
                continue;
            }
            let repeats = self.repeated_current_by_previous_rank[previous_rank];
            if repeats == 0 {
                continue;
            }
            scored.push((repeats, observations, previous_rank, expert_idx));
        }
        scored.sort_by(|a, b| {
            let lhs = (a.0 as u64) * (b.1 as u64);
            let rhs = (b.0 as u64) * (a.1 as u64);
            rhs.cmp(&lhs).then_with(|| a.2.cmp(&b.2))
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(_, _, _, expert_idx)| expert_idx)
            .collect()
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
    stats: crate::qwen36_moe_residency::MoeExpertResidencyStats,
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

#[derive(Debug, Clone)]
pub(crate) struct MoeRouteTelemetry {
    pub(crate) observations_by_rank: Vec<u64>,
    pub(crate) resident_before_by_rank: Vec<u64>,
    pub(crate) repeated_previous_by_rank: Vec<u64>,
    pub(crate) repeated_previous_rank_by_current_rank: Vec<Vec<u64>>,
    pub(crate) weight_sum_by_rank: Vec<f64>,
}

impl MoeRouteTelemetry {
    pub(crate) fn new(top_k: usize) -> Self {
        Self {
            observations_by_rank: vec![0; top_k],
            resident_before_by_rank: vec![0; top_k],
            repeated_previous_by_rank: vec![0; top_k],
            repeated_previous_rank_by_current_rank: vec![vec![0; top_k]; top_k],
            weight_sum_by_rank: vec![0.0; top_k],
        }
    }

    pub(crate) fn record_route_observation(
        &mut self,
        route: &ExpertRoute,
        previous_routes: &[usize],
    ) {
        if route.rank >= self.observations_by_rank.len() {
            return;
        }
        self.observations_by_rank[route.rank] += 1;
        self.weight_sum_by_rank[route.rank] += route.weight as f64;
        if let Some(previous_rank) = previous_routes
            .iter()
            .position(|&expert_idx| expert_idx == route.expert_idx)
        {
            self.repeated_previous_by_rank[route.rank] += 1;
            if let Some(row) = self
                .repeated_previous_rank_by_current_rank
                .get_mut(route.rank)
            {
                if let Some(cell) = row.get_mut(previous_rank) {
                    *cell += 1;
                }
            }
        }
    }

    pub(crate) fn record(
        &mut self,
        manager: &MoeExpertResidencyManager,
        layer_idx: usize,
        routes: &[ExpertRoute],
        previous_routes: &[usize],
    ) {
        for route in routes {
            if route.rank >= self.observations_by_rank.len() {
                continue;
            }
            self.record_route_observation(route, previous_routes);
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
                self.resident_before_by_rank[route.rank] += 1;
            }
        }
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        fn probability(count: u64, observations: u64) -> f64 {
            if observations == 0 {
                0.0
            } else {
                count as f64 / observations as f64
            }
        }

        let avg_weight_by_rank: Vec<f64> = self
            .weight_sum_by_rank
            .iter()
            .zip(&self.observations_by_rank)
            .map(|(sum, count)| {
                if *count == 0 {
                    0.0
                } else {
                    sum / *count as f64
                }
            })
            .collect();
        let repeated_previous_probability_by_current_rank: Vec<f64> = self
            .repeated_previous_by_rank
            .iter()
            .zip(&self.observations_by_rank)
            .map(|(count, observations)| probability(*count, *observations))
            .collect();
        let same_rank_repeat_probability_by_rank: Vec<f64> = self
            .repeated_previous_rank_by_current_rank
            .iter()
            .enumerate()
            .map(|(rank, row)| {
                probability(
                    row.get(rank).copied().unwrap_or(0),
                    self.observations_by_rank.get(rank).copied().unwrap_or(0),
                )
            })
            .collect();
        let top_k = self.observations_by_rank.len();
        let mut repeated_current_by_previous_rank = vec![0u64; top_k];
        let mut best_previous_rank_by_current_rank = vec![None; top_k];
        let mut best_current_rank_by_previous_rank = vec![None; top_k];
        let mut best_transition: Option<(usize, usize, u64)> = None;
        for (current_rank, row) in self
            .repeated_previous_rank_by_current_rank
            .iter()
            .enumerate()
        {
            let mut best_previous: Option<(usize, u64)> = None;
            for (previous_rank, &count) in row.iter().enumerate() {
                if previous_rank < repeated_current_by_previous_rank.len() {
                    repeated_current_by_previous_rank[previous_rank] += count;
                }
                if count > 0
                    && best_previous
                        .map(|(_, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_previous = Some((previous_rank, count));
                }
                if count > 0
                    && best_transition
                        .map(|(_, _, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_transition = Some((current_rank, previous_rank, count));
                }
            }
            best_previous_rank_by_current_rank[current_rank] =
                best_previous.map(|(previous_rank, _)| previous_rank);
        }
        for previous_rank in 0..top_k {
            let mut best_current: Option<(usize, u64)> = None;
            for (current_rank, row) in self
                .repeated_previous_rank_by_current_rank
                .iter()
                .enumerate()
            {
                let count = row.get(previous_rank).copied().unwrap_or(0);
                if count > 0
                    && best_current
                        .map(|(_, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_current = Some((current_rank, count));
                }
            }
            best_current_rank_by_previous_rank[previous_rank] =
                best_current.map(|(current_rank, _)| current_rank);
        }
        let repeated_current_probability_by_previous_rank: Vec<f64> =
            repeated_current_by_previous_rank
                .iter()
                .zip(&self.observations_by_rank)
                .map(|(count, observations)| probability(*count, *observations))
                .collect();
        let best_transition_json = best_transition.map(|(current_rank, previous_rank, count)| {
            serde_json::json!({
                "current_rank": current_rank,
                "previous_rank": previous_rank,
                "count": count,
                "probability_by_current_rank": probability(
                    count,
                    self.observations_by_rank
                        .get(current_rank)
                        .copied()
                        .unwrap_or(0),
                ),
            })
        });
        serde_json::json!({
            "observations_by_rank": &self.observations_by_rank,
            "resident_before_by_rank": &self.resident_before_by_rank,
            "repeated_previous_by_rank": &self.repeated_previous_by_rank,
            "repeated_previous_probability_by_current_rank": repeated_previous_probability_by_current_rank,
            "repeated_previous_rank_by_current_rank": &self.repeated_previous_rank_by_current_rank,
            "same_rank_repeat_probability_by_rank": same_rank_repeat_probability_by_rank,
            "repeated_current_by_previous_rank": repeated_current_by_previous_rank,
            "repeated_current_probability_by_previous_rank": repeated_current_probability_by_previous_rank,
            "best_previous_rank_by_current_rank": best_previous_rank_by_current_rank,
            "best_current_rank_by_previous_rank": best_current_rank_by_previous_rank,
            "best_transition": best_transition_json,
            "avg_weight_by_rank": avg_weight_by_rank,
        })
    }
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
                "prefetch_uploaded_bytes": after.stats.prefetch_uploaded_bytes.saturating_sub(before.stats.prefetch_uploaded_bytes),
                "protect_requests": after.stats.protect_requests.saturating_sub(before.stats.protect_requests),
                "protect_hits": after.stats.protect_hits.saturating_sub(before.stats.protect_hits),
                "protect_misses": after.stats.protect_misses.saturating_sub(before.stats.protect_misses),
                "protect_demotions": after.stats.protect_demotions.saturating_sub(before.stats.protect_demotions),
                "protected_evicted_pages": after.stats.protected_evicted_pages.saturating_sub(before.stats.protected_evicted_pages),
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
                "prefetch_uploaded_bytes": after.stats.prefetch_uploaded_bytes,
                "protected_pages": after.stats.protected_pages,
                "protected_page_budget": after.stats.protected_page_budget,
                "protect_requests": after.stats.protect_requests,
                "protect_hits": after.stats.protect_hits,
                "protect_misses": after.stats.protect_misses,
                "protect_demotions": after.stats.protect_demotions,
                "protected_evicted_pages": after.stats.protected_evicted_pages,
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
                "final_resident_slices": final_snapshot.stats.resident_slices,
                "final_resident_pages": final_snapshot.stats.resident_pages,
                "final_protected_pages": final_snapshot.stats.protected_pages,
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
                "prefetch_uploaded_bytes": final_snapshot.stats.prefetch_uploaded_bytes,
                "protected_pages": final_snapshot.stats.protected_pages,
                "protected_page_budget": final_snapshot.stats.protected_page_budget,
                "protect_requests": final_snapshot.stats.protect_requests,
                "protect_hits": final_snapshot.stats.protect_hits,
                "protect_misses": final_snapshot.stats.protect_misses,
                "protect_demotions": final_snapshot.stats.protect_demotions,
                "protected_evicted_pages": final_snapshot.stats.protected_evicted_pages,
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
