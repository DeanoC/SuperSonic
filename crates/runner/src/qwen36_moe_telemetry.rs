#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::qwen36_moe_decode::ExpertRoute;
use crate::qwen36_moe_residency::{MoeExpertKey, MoeExpertProjection, MoeExpertResidencyManager};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoeIslandPrefetchMode {
    Disabled,
    PreviousToken,
}

impl MoeIslandPrefetchMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PreviousToken => "previous-token",
        }
    }

    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH")
            .ok()
            .as_deref()
        {
            None | Some("0") | Some("off") | Some("disabled") => Ok(Self::Disabled),
            Some("previous-token") | Some("previous_token") | Some("prev-token") => {
                Ok(Self::PreviousToken)
            }
            Some(other) => anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_PREFETCH must be unset, 0, off, disabled, \
                 previous-token, previous_token, or prev-token; got {other:?}"
            ),
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
    observations_by_rank: Vec<u64>,
    resident_before_by_rank: Vec<u64>,
    repeated_previous_by_rank: Vec<u64>,
    weight_sum_by_rank: Vec<f64>,
}

impl MoeRouteTelemetry {
    pub(crate) fn new(top_k: usize) -> Self {
        Self {
            observations_by_rank: vec![0; top_k],
            resident_before_by_rank: vec![0; top_k],
            repeated_previous_by_rank: vec![0; top_k],
            weight_sum_by_rank: vec![0.0; top_k],
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
            self.observations_by_rank[route.rank] += 1;
            self.weight_sum_by_rank[route.rank] += route.weight as f64;
            if previous_routes.contains(&route.expert_idx) {
                self.repeated_previous_by_rank[route.rank] += 1;
            }
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

    fn to_json(&self) -> serde_json::Value {
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
        serde_json::json!({
            "observations_by_rank": &self.observations_by_rank,
            "resident_before_by_rank": &self.resident_before_by_rank,
            "repeated_previous_by_rank": &self.repeated_previous_by_rank,
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
            },
            "resident": {
                "slices": after.stats.resident_slices,
                "pages": after.stats.resident_pages,
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
                "final_resident_slices": final_snapshot.stats.resident_slices,
                "final_resident_pages": final_snapshot.stats.resident_pages,
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
