/// RAII scope that enables Metal/HAL profiling when SUPERSONIC_METAL_PROFILE
/// is set in the environment, and dumps a per-op breakdown to stderr when
/// the scope drops. Used to investigate Metal v2 decode hot paths without
/// adding a permanent profiling cost.
pub(crate) struct MetalProfileScope {
    metal_active: bool,
    qwen_active: bool,
}

impl MetalProfileScope {
    pub(crate) fn new() -> Self {
        let metal_active = std::env::var_os("SUPERSONIC_METAL_PROFILE").is_some();
        let qwen_active = metal_active
            || std::env::var_os("SUPERSONIC_QWEN36_ROUTE_PROFILE").is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_EXPERT_RESIDENCY_PROFILE").is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_PACK_CACHE_PROFILE").is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_FEASIBILITY").is_some();
        if metal_active {
            kernel_ffi::prefill_ffi::metal_profile_set_enabled(true);
            gpu_hal::hal_profile_set_enabled(true);
            kernel_ffi::prefill_ffi::metal_profile_reset();
            gpu_hal::hal_profile_reset();
        }
        if qwen_active {
            kernel_ffi::qwen36_moe::qwen36_route_profile_reset();
            kernel_ffi::qwen36_moe::qwen36_expert_residency_profile_reset();
            kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_reset();
        }
        Self {
            metal_active,
            qwen_active,
        }
    }
}

impl Drop for MetalProfileScope {
    fn drop(&mut self) {
        if !self.metal_active && !self.qwen_active {
            return;
        }
        if self.metal_active {
            let metal = kernel_ffi::prefill_ffi::metal_profile_snapshot();
            eprintln!();
            eprintln!("=== Metal native/host op profile ===");
            eprintln!(
                "calls={} total_ms={:.3} (native={:.3} ms / host={:.3} ms)",
                metal.total_calls, metal.total_ms, metal.native_ms, metal.host_ms
            );
            eprintln!(
                "[metal-profile] calls={} total_ms={:.3} native_ms={:.3} host_ms={:.3}",
                metal.total_calls, metal.total_ms, metal.native_ms, metal.host_ms
            );
            eprintln!(
                "{:<48} {:>10} {:>10} {:>12} {:>12}",
                "op (path)", "calls", "mean_ms", "total_ms", "max_ms"
            );
            for entry in metal.entries.iter().take(40) {
                let mean_ms = if entry.calls > 0 {
                    entry.total_ms / entry.calls as f64
                } else {
                    0.0
                };
                eprintln!(
                    "{:<48} {:>10} {:>10.4} {:>12.3} {:>12.3}",
                    format!("{} ({})", entry.op, entry.path),
                    entry.calls,
                    mean_ms,
                    entry.total_ms,
                    entry.max_ms
                );
                eprintln!(
                    "[metal-profile-op] op={} path={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                    entry.op,
                    entry.path,
                    entry.calls,
                    mean_ms,
                    entry.total_ms,
                    entry.max_ms,
                );
            }
        }
        if self.qwen_active {
            let route = kernel_ffi::qwen36_moe::qwen36_route_profile_snapshot();
            if route.calls > 0 {
                eprintln!();
                eprintln!("=== Qwen3.6 route locality profile ===");
                eprintln!(
                "calls={} layers={} assignments={} unique_layer_experts={} adjacent_hit_rate={:.2}% dropped_calls={}",
                route.calls,
                route.layers,
                route.assignments,
                route.unique_layer_experts,
                route.adjacent_hit_rate() * 100.0,
                route.dropped_calls,
            );
                eprintln!(
                "[qwen36-route-profile] calls={} layers={} assignments={} unique_layer_experts={} adjacent_hits={} adjacent_total={} adjacent_hit_rate={:.6} dropped_calls={}",
                route.calls,
                route.layers,
                route.assignments,
                route.unique_layer_experts,
                route.adjacent_hits,
                route.adjacent_total,
                route.adjacent_hit_rate(),
                route.dropped_calls,
            );
                for sim in &route.cache_sims {
                    eprintln!(
                    "[qwen36-route-cache-sim] scope=per_layer_lru capacity={} hits={} misses={} hit_rate={:.6}",
                    sim.capacity,
                    sim.hits,
                    sim.misses,
                    sim.hit_rate(),
                );
                }
                for sim in &route.topn_sims {
                    eprintln!(
                    "[qwen36-route-topn] scope=per_layer_oracle_topn capacity={} covered={} total={} coverage={:.6}",
                    sim.capacity,
                    sim.covered,
                    sim.total,
                    sim.coverage(),
                );
                }
            }
            let expert_residency =
                kernel_ffi::qwen36_moe::qwen36_expert_residency_profile_snapshot();
            if expert_residency.calls > 0 || expert_residency.entries > 0 {
                eprintln!();
                eprintln!("=== Qwen3.6 expert residency profile ===");
                eprintln!(
                "calls={} entries={} exact_hits={} route_refills={} allocations={} copied_bytes={} exact_hit_rate={:.2}% slot_hit_rate={:.2}% slot_hits={} slot_misses={} evictions={} avg_groups={:.2} max_groups={} avg_copy_bytes={:.0}",
                expert_residency.calls,
                expert_residency.entries,
                expert_residency.exact_hits,
                expert_residency.route_refills,
                expert_residency.allocations,
                expert_residency.copied_bytes,
                expert_residency.exact_hit_rate() * 100.0,
                expert_residency.slot_hit_rate() * 100.0,
                expert_residency.slot_hits,
                expert_residency.slot_misses,
                expert_residency.evictions,
                expert_residency.avg_active_groups(),
                expert_residency.max_active_groups,
                expert_residency.avg_copied_bytes(),
            );
                eprintln!(
                "[qwen36-expert-residency] calls={} entries={} exact_hits={} route_refills={} allocations={} copied_bytes={} exact_hit_rate={:.6} slot_hits={} slot_misses={} slot_hit_rate={:.6} evictions={} avg_active_groups={:.6} max_active_groups={} avg_copy_bytes={:.3}",
                expert_residency.calls,
                expert_residency.entries,
                expert_residency.exact_hits,
                expert_residency.route_refills,
                expert_residency.allocations,
                expert_residency.copied_bytes,
                expert_residency.exact_hit_rate(),
                expert_residency.slot_hits,
                expert_residency.slot_misses,
                expert_residency.slot_hit_rate(),
                expert_residency.evictions,
                expert_residency.avg_active_groups(),
                expert_residency.max_active_groups,
                expert_residency.avg_copied_bytes(),
            );
                eprintln!(
                "[qwen36-pack-cache] calls={} entries={} exact_hits={} route_refills={} allocations={} copied_bytes={} exact_hit_rate={:.6} slot_hits={} slot_misses={} slot_hit_rate={:.6} evictions={} avg_active_groups={:.6} max_active_groups={} avg_copy_bytes={:.3}",
                expert_residency.calls,
                expert_residency.entries,
                expert_residency.exact_hits,
                expert_residency.route_refills,
                expert_residency.allocations,
                expert_residency.copied_bytes,
                expert_residency.exact_hit_rate(),
                expert_residency.slot_hits,
                expert_residency.slot_misses,
                expert_residency.slot_hit_rate(),
                expert_residency.evictions,
                expert_residency.avg_active_groups(),
                expert_residency.max_active_groups,
                expert_residency.avg_copied_bytes(),
            );
                for policy in &expert_residency.policies {
                    eprintln!(
                    "[qwen36-expert-residency-policy] resident_format={} scope={} miss_policy={} capacity={} calls={} exact_hits={} route_refills={} allocations={} copied_bytes={} exact_hit_rate={:.6} slot_hits={} slot_misses={} slot_hit_rate={:.6} evictions={} avg_active_groups={:.6} max_active_groups={} avg_copy_bytes={:.3}",
                    policy.resident_format,
                    policy.scope,
                    policy.miss_policy,
                    policy.capacity,
                    policy.calls,
                    policy.exact_hits,
                    policy.route_refills,
                    policy.allocations,
                    policy.copied_bytes,
                    policy.exact_hit_rate(),
                    policy.slot_hits,
                    policy.slot_misses,
                    policy.slot_hit_rate(),
                    policy.evictions,
                    policy.avg_active_groups(),
                    policy.max_active_groups,
                    policy.avg_copied_bytes(),
                );
                }
            }
            let batched_prefill =
                kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_snapshot();
            if batched_prefill.profiled_tokens > 0 || batched_prefill.dropped_calls > 0 {
                eprintln!();
                eprintln!("=== Qwen3.6 batched-prefill feasibility profile ===");
                eprintln!(
                    "tokens={} chunks={} chunk_size={} assignments={} expert_segments={} avg_unique_experts={:.2} avg_rows_per_segment={:.2} wmma16_assignment_coverage={:.2}% dropped_calls={}",
                    batched_prefill.profiled_tokens,
                    batched_prefill.chunks,
                    batched_prefill.chunk_size,
                    batched_prefill.assignments,
                    batched_prefill.expert_segments,
                    batched_prefill.avg_unique_experts_per_layer_chunk(),
                    batched_prefill.avg_rows_per_segment(),
                    batched_prefill.wmma16_assignment_coverage() * 100.0,
                    batched_prefill.dropped_calls,
                );
                eprintln!(
                    "[qwen36-batched-prefill-feasibility] calls={} dropped_calls={} layers={} top_k={} num_experts={} chunk_size={} prefill_tokens={} profiled_tokens={} chunks={} assignments={} permutation_entries={} expert_segments={} avg_unique_experts_per_layer_chunk={:.6} avg_rows_per_segment={:.6} max_rows_per_segment={} max_unique_experts_per_layer_chunk={} wmma16_segments={} wmma16_covered_assignments={} wmma16_assignment_coverage={:.6} metadata_only=1",
                    batched_prefill.calls,
                    batched_prefill.dropped_calls,
                    batched_prefill.layers,
                    batched_prefill.top_k,
                    batched_prefill.num_experts,
                    batched_prefill.chunk_size,
                    batched_prefill.prefill_tokens,
                    batched_prefill.profiled_tokens,
                    batched_prefill.chunks,
                    batched_prefill.assignments,
                    batched_prefill.permutation_entries,
                    batched_prefill.expert_segments,
                    batched_prefill.avg_unique_experts_per_layer_chunk(),
                    batched_prefill.avg_rows_per_segment(),
                    batched_prefill.max_rows_per_segment,
                    batched_prefill.max_unique_experts_per_layer_chunk,
                    batched_prefill.wmma16_segments,
                    batched_prefill.wmma16_covered_assignments,
                    batched_prefill.wmma16_assignment_coverage(),
                );
            }
        }
        if self.metal_active {
            let hal = gpu_hal::hal_profile_snapshot();
            eprintln!();
            eprintln!("=== HAL op profile (gpu_hal level) ===");
            eprintln!(
            "calls={} total_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            hal.total_calls,
            hal.total_ms,
            hal.alloc_calls,
            hal.alloc_bytes,
            hal.h2d_bytes,
            hal.d2h_bytes,
            hal.d2d_bytes,
            hal.memset_bytes,
            hal.sync_calls,
        );
            eprintln!(
            "[hal-profile] calls={} total_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            hal.total_calls,
            hal.total_ms,
            hal.alloc_calls,
            hal.alloc_bytes,
            hal.h2d_bytes,
            hal.d2h_bytes,
            hal.d2d_bytes,
            hal.memset_bytes,
            hal.sync_calls,
        );
            eprintln!(
                "{:<32} {:>10} {:>10} {:>12} {:>12} {:>14}",
                "op", "calls", "mean_ms", "total_ms", "max_ms", "total_bytes"
            );
            for entry in hal.entries.iter().take(20) {
                let mean_ms = if entry.calls > 0 {
                    entry.total_ms / entry.calls as f64
                } else {
                    0.0
                };
                eprintln!(
                    "{:<32} {:>10} {:>10.4} {:>12.3} {:>12.3} {:>14}",
                    entry.op, entry.calls, mean_ms, entry.total_ms, entry.max_ms, entry.total_bytes
                );
                eprintln!(
                "[hal-profile-op] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} total_bytes={}",
                entry.op,
                entry.calls,
                mean_ms,
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
            );
            }
            kernel_ffi::prefill_ffi::metal_profile_set_enabled(false);
            gpu_hal::hal_profile_set_enabled(false);
        }
    }
}

pub(crate) struct PrefillProfileScope<'a> {
    active: bool,
    json_path: Option<&'a std::path::Path>,
    family: &'a str,
    model: &'a str,
    backend: &'a str,
    prompt_tokens: usize,
    start: std::time::Instant,
}

impl<'a> PrefillProfileScope<'a> {
    pub(crate) fn new(
        enabled: bool,
        json_path: Option<&'a std::path::Path>,
        family: &'a str,
        model: &'a str,
        backend: &'a str,
        prompt_tokens: usize,
    ) -> Self {
        let active = enabled || json_path.is_some();
        if active {
            gpu_hal::hal_profile_set_enabled(true);
            kernel_ffi::prefill_ffi::ffi_profile_set_enabled(true);
            gpu_hal::hal_profile_reset();
            kernel_ffi::prefill_ffi::ffi_profile_reset();
        }
        Self {
            active,
            json_path,
            family,
            model,
            backend,
            prompt_tokens,
            start: std::time::Instant::now(),
        }
    }

    pub(crate) fn finish(mut self) -> anyhow::Result<()> {
        if !self.active {
            return Ok(());
        }
        let wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let hal = gpu_hal::hal_profile_snapshot();
        let ffi = kernel_ffi::prefill_ffi::ffi_profile_snapshot();
        eprintln!(
            "[prefill-profile] family={} model={} backend={} prompt_tokens={} wall_ms={:.3} ffi_calls={} ffi_ms={:.3} hal_calls={} hal_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            self.family,
            self.model,
            self.backend,
            self.prompt_tokens,
            wall_ms,
            ffi.total_calls,
            ffi.total_ms,
            hal.total_calls,
            hal.total_ms,
            hal.alloc_calls,
            hal.alloc_bytes,
            hal.h2d_bytes,
            hal.d2h_bytes,
            hal.d2d_bytes,
            hal.memset_bytes,
            hal.sync_calls,
        );
        for entry in ffi.entries.iter().take(30) {
            eprintln!(
                "[prefill-profile] ffi_op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
            );
        }
        for entry in hal.entries.iter().take(20) {
            eprintln!(
                "[prefill-profile] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} bytes={}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
            );
        }
        disable_prefill_profiles();
        self.active = false;
        if let Some(path) = self.json_path {
            let ffi_entries: Vec<_> = ffi
                .entries
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "op": entry.op,
                        "calls": entry.calls,
                        "mean_ms": entry.mean_ms(),
                        "total_ms": entry.total_ms,
                        "max_ms": entry.max_ms,
                    })
                })
                .collect();
            let hal_entries: Vec<_> = hal
                .entries
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "op": entry.op,
                        "calls": entry.calls,
                        "mean_ms": entry.mean_ms(),
                        "total_ms": entry.total_ms,
                        "max_ms": entry.max_ms,
                        "total_bytes": entry.total_bytes,
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "family": self.family,
                "model": self.model,
                "backend": self.backend,
                "prompt_tokens": self.prompt_tokens,
                "wall_ms": wall_ms,
                "ffi": {
                    "total_calls": ffi.total_calls,
                    "total_ms": ffi.total_ms,
                    "entries": ffi_entries,
                },
                "hal": {
                    "total_calls": hal.total_calls,
                    "total_ms": hal.total_ms,
                    "alloc_calls": hal.alloc_calls,
                    "alloc_bytes": hal.alloc_bytes,
                    "free_calls": hal.free_calls,
                    "h2d_bytes": hal.h2d_bytes,
                    "d2h_bytes": hal.d2h_bytes,
                    "d2d_bytes": hal.d2d_bytes,
                    "memset_bytes": hal.memset_bytes,
                    "sync_calls": hal.sync_calls,
                    "entries": hal_entries,
                },
            });
            std::fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
        }
        Ok(())
    }
}

impl Drop for PrefillProfileScope<'_> {
    fn drop(&mut self) {
        if self.active {
            disable_prefill_profiles();
        }
    }
}

fn disable_prefill_profiles() {
    kernel_ffi::prefill_ffi::ffi_profile_set_enabled(false);
    gpu_hal::hal_profile_set_enabled(false);
}
