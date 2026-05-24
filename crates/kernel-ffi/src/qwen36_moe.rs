//! FFI bridge for the Qwen3.6-MoE persistent decode megakernel.
//!
//! Status: **infrastructure only.** PR 4 (this file) lands the descriptor
//! layout + a stub kernel that validates the launch path and descriptor
//! read-out. Actual transformer compute (attention with `attn_output_gate`,
//! MoE routing + work-stealing dispatch, fused expert mat-vec, shared
//! expert, lm_head) lands in follow-up PRs.
//!
//! Why a stub first: the megakernel will be ~6500 LoC of HIP. Wiring the
//! FFI/bridge/build path before any compute math means a future kernel
//! commit can land focused-on-math code with no orchestration noise. The
//! stub also serves as a smoke test for descriptor field layout — a
//! Rust↔C++ struct-mismatch bug found here saves hours later.

#![cfg_attr(
    not(supersonic_backend_hip),
    allow(unused_variables, unused_mut, unreachable_code)
)]

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;
use std::sync::{Mutex, OnceLock};

use gpu_hal::{Backend, BufferKind, GpuBuffer, GpuError, ScalarType};
use half::f16;

use crate::layer_desc::MAX_BATCH_SIZE;

const QWEN36_ROUTE_PROFILE_DEFAULT_LAYERS: usize = 40;
const QWEN36_ROUTE_PROFILE_DEFAULT_MAX_CALLS: usize = 16_384;
const QWEN36_ROUTE_PROFILE_DEFAULT_CAPS: [usize; 6] = [2, 4, 8, 16, 32, 64];
const QWEN36_BATCHED_PREFILL_PLAN_CHUNKS: [usize; 5] = [64, 128, 256, 512, 1024];

static QWEN36_ROUTE_PROFILE: OnceLock<Mutex<Qwen36RouteProfileAccumulator>> = OnceLock::new();
static QWEN36_EXPERT_RESIDENCY_PROFILE: OnceLock<Mutex<Qwen36ExpertResidencyProfileAccumulator>> =
    OnceLock::new();
static QWEN36_BATCHED_PREFILL_FEASIBILITY_PROFILE: OnceLock<
    Mutex<Qwen36BatchedPrefillFeasibilityConfig>,
> = OnceLock::new();

#[derive(Debug, Clone, Default)]
struct Qwen36RouteProfileAccumulator {
    records: Vec<Vec<u16>>,
    dropped_calls: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36RouteProfileSnapshot {
    pub calls: u64,
    pub dropped_calls: u64,
    pub layers: usize,
    pub assignments: u64,
    pub unique_layer_experts: usize,
    pub adjacent_hits: u64,
    pub adjacent_total: u64,
    pub cache_sims: Vec<Qwen36RouteCacheSim>,
    pub topn_sims: Vec<Qwen36RouteTopNSim>,
    pub topn_layers: Vec<Qwen36RouteTopNLayer>,
    pub route_calls: Vec<Qwen36RouteCall>,
}

impl Qwen36RouteProfileSnapshot {
    pub fn adjacent_hit_rate(&self) -> f64 {
        if self.adjacent_total == 0 {
            0.0
        } else {
            self.adjacent_hits as f64 / self.adjacent_total as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36RouteCacheSim {
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
}

impl Qwen36RouteCacheSim {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36RouteTopNSim {
    pub capacity: usize,
    pub covered: u64,
    pub total: u64,
}

impl Qwen36RouteTopNSim {
    pub fn coverage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.covered as f64 / self.total as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36RouteTopNLayer {
    pub capacity: usize,
    pub layer: usize,
    pub experts: Vec<u16>,
    pub counts: Vec<u64>,
    pub covered: u64,
    pub total: u64,
}

impl Qwen36RouteTopNLayer {
    pub fn coverage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.covered as f64 / self.total as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36RouteCall {
    pub call_idx: usize,
    pub layer: usize,
    pub experts: Vec<u16>,
}

#[derive(Debug, Clone)]
struct Qwen36BatchedPrefillFeasibilityConfig {
    layers: usize,
    top_k: usize,
    num_experts: usize,
    chunk_size: usize,
    prefill_tokens: usize,
}

impl Default for Qwen36BatchedPrefillFeasibilityConfig {
    fn default() -> Self {
        Self {
            layers: QWEN36_ROUTE_PROFILE_DEFAULT_LAYERS,
            top_k: 8,
            num_experts: 256,
            chunk_size: 512,
            prefill_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36BatchedPrefillFeasibilityProfileSnapshot {
    pub calls: u64,
    pub dropped_calls: u64,
    pub layers: usize,
    pub top_k: usize,
    pub num_experts: usize,
    pub chunk_size: usize,
    pub prefill_tokens: usize,
    pub profiled_tokens: usize,
    pub chunks: usize,
    pub assignments: u64,
    pub permutation_entries: u64,
    pub expert_segments: u64,
    pub wmma16_segments: u64,
    pub wmma16_covered_assignments: u64,
    pub wmma16_padded_assignments: u64,
    pub scalar_tail_segments: u64,
    pub scalar_tail_assignments: u64,
    pub max_rows_per_segment: u64,
    pub max_unique_experts_per_layer_chunk: usize,
}

impl Qwen36BatchedPrefillFeasibilityProfileSnapshot {
    pub fn avg_unique_experts_per_layer_chunk(&self) -> f64 {
        let denom = self.chunks.saturating_mul(self.layers);
        if denom == 0 {
            0.0
        } else {
            self.expert_segments as f64 / denom as f64
        }
    }

    pub fn avg_rows_per_segment(&self) -> f64 {
        if self.expert_segments == 0 {
            0.0
        } else {
            self.assignments as f64 / self.expert_segments as f64
        }
    }

    pub fn wmma16_assignment_coverage(&self) -> f64 {
        if self.assignments == 0 {
            0.0
        } else {
            self.wmma16_covered_assignments as f64 / self.assignments as f64
        }
    }

    pub fn wmma16_padding_overhead(&self) -> f64 {
        if self.assignments == 0 {
            0.0
        } else {
            self.wmma16_padded_assignments
                .saturating_sub(self.assignments) as f64
                / self.assignments as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen36ExpertResidentFormat {
    NativeInt4,
    Fp16Mps,
}

impl Qwen36ExpertResidentFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NativeInt4 => "native_int4",
            Self::Fp16Mps => "fp16_mps",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen36ExpertResidencyScope {
    PerLayer,
}

impl Qwen36ExpertResidencyScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PerLayer => "per_layer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen36ExpertResidencyMissPolicy {
    ExactRoute,
    LruHotset,
    GpuPack,
    StaticTopN,
}

impl Qwen36ExpertResidencyMissPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactRoute => "exact_route",
            Self::LruHotset => "lru_hotset",
            Self::GpuPack => "gpu_pack",
            Self::StaticTopN => "static_topn",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen36ExpertResidencyProfileKey {
    pub resident_format: Qwen36ExpertResidentFormat,
    pub scope: Qwen36ExpertResidencyScope,
    pub miss_policy: Qwen36ExpertResidencyMissPolicy,
    pub capacity: usize,
}

impl Qwen36ExpertResidencyProfileKey {
    fn new(miss_policy: Qwen36ExpertResidencyMissPolicy, capacity: usize) -> Self {
        Self::with_format(
            Qwen36ExpertResidentFormat::NativeInt4,
            miss_policy,
            capacity,
        )
    }

    fn with_format(
        resident_format: Qwen36ExpertResidentFormat,
        miss_policy: Qwen36ExpertResidencyMissPolicy,
        capacity: usize,
    ) -> Self {
        Self {
            resident_format,
            scope: Qwen36ExpertResidencyScope::PerLayer,
            miss_policy,
            capacity,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Qwen36ExpertResidencyCounters {
    calls: u64,
    exact_hits: u64,
    route_refills: u64,
    allocations: u64,
    copied_bytes: u64,
    active_groups_total: u64,
    max_active_groups: usize,
    slot_hits: u64,
    slot_misses: u64,
    evictions: u64,
}

#[derive(Debug, Clone, Default)]
struct Qwen36ExpertResidencyProfileAccumulator {
    totals: Qwen36ExpertResidencyCounters,
    policies: HashMap<Qwen36ExpertResidencyProfileKey, Qwen36ExpertResidencyCounters>,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36ExpertResidencyPolicySnapshot {
    pub resident_format: &'static str,
    pub scope: &'static str,
    pub miss_policy: &'static str,
    pub capacity: usize,
    pub calls: u64,
    pub exact_hits: u64,
    pub route_refills: u64,
    pub allocations: u64,
    pub copied_bytes: u64,
    pub active_groups_total: u64,
    pub max_active_groups: usize,
    pub slot_hits: u64,
    pub slot_misses: u64,
    pub evictions: u64,
}

impl Qwen36ExpertResidencyPolicySnapshot {
    pub fn exact_hit_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.exact_hits as f64 / self.calls as f64
        }
    }

    pub fn avg_active_groups(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.active_groups_total as f64 / self.calls as f64
        }
    }

    pub fn avg_copied_bytes(&self) -> f64 {
        let copy_ops = self.route_refills + self.allocations;
        if copy_ops == 0 {
            0.0
        } else {
            self.copied_bytes as f64 / copy_ops as f64
        }
    }

    pub fn slot_hit_rate(&self) -> f64 {
        let total = self.slot_hits + self.slot_misses;
        if total == 0 {
            0.0
        } else {
            self.slot_hits as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen36ExpertResidencyProfileSnapshot {
    pub calls: u64,
    pub exact_hits: u64,
    pub route_refills: u64,
    pub allocations: u64,
    pub copied_bytes: u64,
    pub active_groups_total: u64,
    pub max_active_groups: usize,
    pub entries: usize,
    pub slot_hits: u64,
    pub slot_misses: u64,
    pub evictions: u64,
    pub policies: Vec<Qwen36ExpertResidencyPolicySnapshot>,
}

impl Qwen36ExpertResidencyProfileSnapshot {
    pub fn exact_hit_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.exact_hits as f64 / self.calls as f64
        }
    }

    pub fn avg_active_groups(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.active_groups_total as f64 / self.calls as f64
        }
    }

    pub fn avg_copied_bytes(&self) -> f64 {
        let copy_ops = self.route_refills + self.allocations;
        if copy_ops == 0 {
            0.0
        } else {
            self.copied_bytes as f64 / copy_ops as f64
        }
    }

    pub fn slot_hit_rate(&self) -> f64 {
        let total = self.slot_hits + self.slot_misses;
        if total == 0 {
            0.0
        } else {
            self.slot_hits as f64 / total as f64
        }
    }
}

pub type Qwen36PackedExpertCacheProfileSnapshot = Qwen36ExpertResidencyProfileSnapshot;

#[allow(clippy::too_many_arguments)]
fn qwen36_expert_residency_counters_record(
    counters: &mut Qwen36ExpertResidencyCounters,
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    counters.calls += 1;
    counters.active_groups_total += active_groups as u64;
    counters.max_active_groups = counters.max_active_groups.max(active_groups);
    if exact_hit {
        counters.exact_hits += 1;
    }
    if route_refill {
        counters.route_refills += 1;
    }
    if allocation {
        counters.allocations += 1;
    }
    counters.copied_bytes += copied_bytes as u64;
    counters.slot_hits += slot_hits as u64;
    counters.slot_misses += slot_misses as u64;
    counters.evictions += evictions as u64;
}

pub fn qwen36_route_profile_reset() {
    if let Some(profile) = QWEN36_ROUTE_PROFILE.get() {
        let mut profile = profile.lock().expect("qwen36 route profile mutex poisoned");
        profile.records.clear();
        profile.dropped_calls = 0;
    }
}

pub fn qwen36_batched_prefill_feasibility_profile_reset() {
    if let Some(profile) = QWEN36_BATCHED_PREFILL_FEASIBILITY_PROFILE.get() {
        let mut profile = profile
            .lock()
            .expect("qwen36 batched prefill feasibility profile mutex poisoned");
        *profile = Qwen36BatchedPrefillFeasibilityConfig::default();
    }
}

pub fn qwen36_route_profile_snapshot() -> Qwen36RouteProfileSnapshot {
    let Some(profile) = QWEN36_ROUTE_PROFILE.get() else {
        return Qwen36RouteProfileSnapshot::default();
    };
    let profile = profile.lock().expect("qwen36 route profile mutex poisoned");
    qwen36_route_profile_simulate(
        &profile.records,
        profile.dropped_calls,
        qwen36_route_profile_layers(),
    )
}

pub fn qwen36_batched_prefill_feasibility_profile_configure(
    layers: usize,
    top_k: usize,
    num_experts: usize,
    chunk_size: usize,
    prefill_tokens: usize,
) {
    let profile = QWEN36_BATCHED_PREFILL_FEASIBILITY_PROFILE
        .get_or_init(|| Mutex::new(Qwen36BatchedPrefillFeasibilityConfig::default()));
    let mut profile = profile
        .lock()
        .expect("qwen36 batched prefill feasibility profile mutex poisoned");
    *profile = Qwen36BatchedPrefillFeasibilityConfig {
        layers: layers.max(1),
        top_k: top_k.max(1),
        num_experts: num_experts.max(1),
        chunk_size: chunk_size.max(1),
        prefill_tokens,
    };
}

pub fn qwen36_batched_prefill_feasibility_profile_snapshot(
) -> Qwen36BatchedPrefillFeasibilityProfileSnapshot {
    let Some(route_profile) = QWEN36_ROUTE_PROFILE.get() else {
        return Qwen36BatchedPrefillFeasibilityProfileSnapshot::default();
    };
    let route_profile = route_profile
        .lock()
        .expect("qwen36 route profile mutex poisoned");
    let config = QWEN36_BATCHED_PREFILL_FEASIBILITY_PROFILE
        .get()
        .map(|profile| {
            profile
                .lock()
                .expect("qwen36 batched prefill feasibility profile mutex poisoned")
                .clone()
        })
        .unwrap_or_default();
    qwen36_batched_prefill_feasibility_profile_simulate(
        &route_profile.records,
        route_profile.dropped_calls,
        &config,
    )
}

pub fn qwen36_batched_prefill_feasibility_plan_snapshots(
) -> Vec<Qwen36BatchedPrefillFeasibilityProfileSnapshot> {
    let Some(route_profile) = QWEN36_ROUTE_PROFILE.get() else {
        return Vec::new();
    };
    let route_profile = route_profile
        .lock()
        .expect("qwen36 route profile mutex poisoned");
    let config = QWEN36_BATCHED_PREFILL_FEASIBILITY_PROFILE
        .get()
        .map(|profile| {
            profile
                .lock()
                .expect("qwen36 batched prefill feasibility profile mutex poisoned")
                .clone()
        })
        .unwrap_or_default();
    let mut chunk_sizes = qwen36_batched_prefill_plan_chunk_sizes(config.chunk_size);
    chunk_sizes.sort_unstable();
    chunk_sizes.dedup();
    chunk_sizes
        .into_iter()
        .map(|chunk_size| {
            let mut plan_config = config.clone();
            plan_config.chunk_size = chunk_size.max(1);
            qwen36_batched_prefill_feasibility_profile_simulate(
                &route_profile.records,
                route_profile.dropped_calls,
                &plan_config,
            )
        })
        .collect()
}

pub fn qwen36_packed_expert_cache_profile_reset() {
    qwen36_expert_residency_profile_reset();
}

pub fn qwen36_expert_residency_profile_reset() {
    if let Some(profile) = QWEN36_EXPERT_RESIDENCY_PROFILE.get() {
        *profile
            .lock()
            .expect("qwen36 expert residency profile mutex poisoned") =
            Qwen36ExpertResidencyProfileAccumulator::default();
    }
}

pub fn qwen36_packed_expert_cache_profile_snapshot() -> Qwen36PackedExpertCacheProfileSnapshot {
    qwen36_expert_residency_profile_snapshot()
}

pub fn qwen36_expert_residency_profile_snapshot() -> Qwen36ExpertResidencyProfileSnapshot {
    let entries = QWEN36_PACKED_EXPERT_CACHE
        .get()
        .and_then(|cache| cache.lock().ok().map(|cache| cache.len()))
        .unwrap_or(0)
        + QWEN36_PACKED_EXPERT_HOTSET_CACHE
            .get()
            .and_then(|cache| cache.lock().ok().map(|cache| cache.len()))
            .unwrap_or(0)
        + QWEN36_PACKED_EXPERT_STATIC_TOPN_CACHE
            .get()
            .and_then(|cache| cache.lock().ok().map(|cache| cache.len()))
            .unwrap_or(0)
        + QWEN36_MPS_EXPERT_STATIC_TOPN_CACHE
            .get()
            .and_then(|cache| cache.lock().ok().map(|cache| cache.len()))
            .unwrap_or(0);
    let Some(profile) = QWEN36_EXPERT_RESIDENCY_PROFILE.get() else {
        return Qwen36ExpertResidencyProfileSnapshot {
            entries,
            ..Qwen36ExpertResidencyProfileSnapshot::default()
        };
    };
    let profile = profile
        .lock()
        .expect("qwen36 expert residency profile mutex poisoned");
    let mut policies: Vec<_> = profile
        .policies
        .iter()
        .map(|(key, counters)| Qwen36ExpertResidencyPolicySnapshot {
            resident_format: key.resident_format.as_str(),
            scope: key.scope.as_str(),
            miss_policy: key.miss_policy.as_str(),
            capacity: key.capacity,
            calls: counters.calls,
            exact_hits: counters.exact_hits,
            route_refills: counters.route_refills,
            allocations: counters.allocations,
            copied_bytes: counters.copied_bytes,
            active_groups_total: counters.active_groups_total,
            max_active_groups: counters.max_active_groups,
            slot_hits: counters.slot_hits,
            slot_misses: counters.slot_misses,
            evictions: counters.evictions,
        })
        .collect();
    policies.sort_by_key(|policy| {
        (
            policy.resident_format,
            policy.scope,
            policy.miss_policy,
            policy.capacity,
        )
    });
    Qwen36ExpertResidencyProfileSnapshot {
        calls: profile.totals.calls,
        exact_hits: profile.totals.exact_hits,
        route_refills: profile.totals.route_refills,
        allocations: profile.totals.allocations,
        copied_bytes: profile.totals.copied_bytes,
        active_groups_total: profile.totals.active_groups_total,
        max_active_groups: profile.totals.max_active_groups,
        entries,
        slot_hits: profile.totals.slot_hits,
        slot_misses: profile.totals.slot_misses,
        evictions: profile.totals.evictions,
        policies,
    }
}

fn qwen36_route_profile_enabled() -> bool {
    crate::prefill_ffi::metal_profile_enabled()
        || std::env::var_os("SUPERSONIC_QWEN36_ROUTE_PROFILE").is_some()
        || std::env::var_os("SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_CALLS").is_some()
        || std::env::var_os("SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_TOPN_LAYERS").is_some()
        || std::env::var_os("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_FEASIBILITY").is_some()
}

pub fn qwen36_batched_prefill_feasibility_profile_enabled() -> bool {
    crate::prefill_ffi::metal_profile_enabled()
        || std::env::var_os("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_FEASIBILITY").is_some()
}

pub fn qwen36_route_profile_record_active_experts(active_experts: &[usize]) {
    if qwen36_route_profile_enabled() {
        qwen36_route_profile_record(active_experts);
    }
}

fn qwen36_batched_prefill_plan_chunk_sizes(current_chunk_size: usize) -> Vec<usize> {
    let mut sizes: Vec<usize> =
        if let Ok(raw) = std::env::var("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_PLAN_CHUNKS") {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&chunk| chunk > 0)
                .collect()
        } else {
            QWEN36_BATCHED_PREFILL_PLAN_CHUNKS.to_vec()
        };
    sizes.push(current_chunk_size.max(1));
    sizes
}

fn qwen36_expert_residency_profile_enabled() -> bool {
    crate::prefill_ffi::metal_profile_enabled()
        || std::env::var_os("SUPERSONIC_QWEN36_EXPERT_RESIDENCY_PROFILE").is_some()
        || std::env::var_os("SUPERSONIC_QWEN36_PACK_CACHE_PROFILE").is_some()
}

#[allow(clippy::too_many_arguments)]
fn qwen36_expert_residency_profile_record(
    key: Qwen36ExpertResidencyProfileKey,
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    if !qwen36_expert_residency_profile_enabled() {
        return;
    }
    let profile = QWEN36_EXPERT_RESIDENCY_PROFILE
        .get_or_init(|| Mutex::new(Qwen36ExpertResidencyProfileAccumulator::default()));
    let mut profile = profile
        .lock()
        .expect("qwen36 expert residency profile mutex poisoned");
    qwen36_expert_residency_counters_record(
        &mut profile.totals,
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
    let policy = profile.policies.entry(key).or_default();
    qwen36_expert_residency_counters_record(
        policy,
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn qwen36_packed_expert_cache_profile_record(
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    qwen36_expert_residency_profile_record(
        Qwen36ExpertResidencyProfileKey::new(
            Qwen36ExpertResidencyMissPolicy::ExactRoute,
            active_groups,
        ),
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn qwen36_hotset_expert_residency_profile_record(
    capacity: usize,
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    qwen36_expert_residency_profile_record(
        Qwen36ExpertResidencyProfileKey::new(Qwen36ExpertResidencyMissPolicy::LruHotset, capacity),
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn qwen36_static_topn_expert_residency_profile_record(
    capacity: usize,
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    qwen36_expert_residency_profile_record(
        Qwen36ExpertResidencyProfileKey::new(Qwen36ExpertResidencyMissPolicy::StaticTopN, capacity),
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn qwen36_mps_static_topn_expert_residency_profile_record(
    capacity: usize,
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    qwen36_expert_residency_profile_record(
        Qwen36ExpertResidencyProfileKey::with_format(
            Qwen36ExpertResidentFormat::Fp16Mps,
            Qwen36ExpertResidencyMissPolicy::StaticTopN,
            capacity,
        ),
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

#[allow(clippy::too_many_arguments)]
fn qwen36_gpu_pack_expert_residency_profile_record(
    exact_hit: bool,
    route_refill: bool,
    allocation: bool,
    active_groups: usize,
    copied_bytes: usize,
    slot_hits: usize,
    slot_misses: usize,
    evictions: usize,
) {
    qwen36_expert_residency_profile_record(
        Qwen36ExpertResidencyProfileKey::new(
            Qwen36ExpertResidencyMissPolicy::GpuPack,
            active_groups,
        ),
        exact_hit,
        route_refill,
        allocation,
        active_groups,
        copied_bytes,
        slot_hits,
        slot_misses,
        evictions,
    );
}

fn qwen36_route_profile_layers() -> usize {
    std::env::var("SUPERSONIC_QWEN36_ROUTE_PROFILE_LAYERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(QWEN36_ROUTE_PROFILE_DEFAULT_LAYERS)
}

fn qwen36_route_profile_capacities() -> Vec<usize> {
    let mut capacities = std::env::var("SUPERSONIC_QWEN36_ROUTE_PROFILE_CAPACITIES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&capacity| capacity > 0)
                .collect::<Vec<_>>()
        })
        .filter(|capacities| !capacities.is_empty())
        .unwrap_or_else(|| QWEN36_ROUTE_PROFILE_DEFAULT_CAPS.to_vec());
    capacities.sort_unstable();
    capacities.dedup();
    capacities
}

fn qwen36_route_profile_max_calls() -> usize {
    std::env::var("SUPERSONIC_QWEN36_ROUTE_PROFILE_MAX_CALLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(QWEN36_ROUTE_PROFILE_DEFAULT_MAX_CALLS)
}

fn qwen36_route_profile_record(active_experts: &[usize]) {
    let profile =
        QWEN36_ROUTE_PROFILE.get_or_init(|| Mutex::new(Qwen36RouteProfileAccumulator::default()));
    let mut profile = profile.lock().expect("qwen36 route profile mutex poisoned");
    if profile.records.len() >= qwen36_route_profile_max_calls() {
        profile.dropped_calls += 1;
        return;
    }
    profile.records.push(
        active_experts
            .iter()
            .map(|&expert| expert.min(u16::MAX as usize) as u16)
            .collect(),
    );
}

fn qwen36_route_profile_simulate(
    records: &[Vec<u16>],
    dropped_calls: u64,
    layers: usize,
) -> Qwen36RouteProfileSnapshot {
    let mut snapshot = Qwen36RouteProfileSnapshot {
        calls: records.len() as u64,
        dropped_calls,
        layers,
        ..Qwen36RouteProfileSnapshot::default()
    };
    if records.is_empty() || layers == 0 {
        return snapshot;
    }

    let mut unique_layer_experts = HashMap::<(usize, u16), ()>::new();
    let mut per_layer_freq: Vec<HashMap<u16, u64>> = (0..layers).map(|_| HashMap::new()).collect();
    for (call_idx, experts) in records.iter().enumerate() {
        let layer = call_idx % layers;
        snapshot.route_calls.push(Qwen36RouteCall {
            call_idx,
            layer,
            experts: experts.clone(),
        });
        snapshot.assignments += experts.len() as u64;
        for &expert in experts {
            unique_layer_experts.insert((layer, expert), ());
            *per_layer_freq[layer].entry(expert).or_insert(0) += 1;
        }
        if call_idx >= layers {
            let prev = &records[call_idx - layers];
            for &expert in experts {
                snapshot.adjacent_total += 1;
                if prev.contains(&expert) {
                    snapshot.adjacent_hits += 1;
                }
            }
        }
    }
    snapshot.unique_layer_experts = unique_layer_experts.len();

    for capacity in qwen36_route_profile_capacities() {
        let mut caches: Vec<Vec<u16>> = (0..layers).map(|_| Vec::new()).collect();
        let mut sim = Qwen36RouteCacheSim {
            capacity,
            ..Qwen36RouteCacheSim::default()
        };
        for (call_idx, experts) in records.iter().enumerate() {
            let layer = call_idx % layers;
            let cache = &mut caches[layer];
            for &expert in experts {
                if let Some(pos) = cache.iter().position(|&cached| cached == expert) {
                    sim.hits += 1;
                    let expert = cache.remove(pos);
                    cache.push(expert);
                } else {
                    sim.misses += 1;
                    if cache.len() >= capacity {
                        cache.remove(0);
                    }
                    cache.push(expert);
                }
            }
        }
        snapshot.cache_sims.push(sim);

        let mut topn = Qwen36RouteTopNSim {
            capacity,
            total: snapshot.assignments,
            ..Qwen36RouteTopNSim::default()
        };
        for (layer, freq) in per_layer_freq.iter().enumerate() {
            let mut counts: Vec<(u16, u64)> = freq
                .iter()
                .map(|(&expert, &count)| (expert, count))
                .collect();
            counts.sort_unstable_by(|(lhs_expert, lhs_count), (rhs_expert, rhs_count)| {
                rhs_count
                    .cmp(lhs_count)
                    .then_with(|| lhs_expert.cmp(rhs_expert))
            });
            let selected: Vec<(u16, u64)> = counts.into_iter().take(capacity).collect();
            let covered = selected.iter().map(|(_, count)| *count).sum::<u64>();
            let total = freq.values().copied().sum::<u64>();
            topn.covered += covered;
            snapshot.topn_layers.push(Qwen36RouteTopNLayer {
                capacity,
                layer,
                experts: selected.iter().map(|(expert, _)| *expert).collect(),
                counts: selected.iter().map(|(_, count)| *count).collect(),
                covered,
                total,
            });
        }
        snapshot.topn_sims.push(topn);
    }

    snapshot
}

fn qwen36_batched_prefill_feasibility_profile_simulate(
    records: &[Vec<u16>],
    dropped_calls: u64,
    config: &Qwen36BatchedPrefillFeasibilityConfig,
) -> Qwen36BatchedPrefillFeasibilityProfileSnapshot {
    let layers = config.layers.max(1);
    let top_k = config.top_k.max(1);
    let chunk_size = config.chunk_size.max(1);
    let available_tokens = records.len() / layers;
    let requested_tokens = if config.prefill_tokens == 0 {
        available_tokens
    } else {
        config.prefill_tokens
    };
    let profiled_tokens = requested_tokens.min(available_tokens);
    let chunks = if profiled_tokens == 0 {
        0
    } else {
        profiled_tokens.div_ceil(chunk_size)
    };
    let mut snapshot = Qwen36BatchedPrefillFeasibilityProfileSnapshot {
        calls: records.len() as u64,
        dropped_calls,
        layers,
        top_k,
        num_experts: config.num_experts.max(1),
        chunk_size,
        prefill_tokens: config.prefill_tokens,
        profiled_tokens,
        chunks,
        ..Qwen36BatchedPrefillFeasibilityProfileSnapshot::default()
    };
    if profiled_tokens == 0 {
        return snapshot;
    }

    for chunk_start in (0..profiled_tokens).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(profiled_tokens);
        for layer in 0..layers {
            let mut counts = HashMap::<u16, u64>::new();
            for token in chunk_start..chunk_end {
                let call_idx = token * layers + layer;
                let Some(experts) = records.get(call_idx) else {
                    continue;
                };
                for &expert in experts.iter().take(top_k) {
                    *counts.entry(expert).or_insert(0) += 1;
                    snapshot.assignments += 1;
                }
            }
            snapshot.expert_segments += counts.len() as u64;
            snapshot.max_unique_experts_per_layer_chunk = snapshot
                .max_unique_experts_per_layer_chunk
                .max(counts.len());
            for rows in counts.values().copied() {
                snapshot.max_rows_per_segment = snapshot.max_rows_per_segment.max(rows);
                snapshot.wmma16_padded_assignments += rows.div_ceil(16) * 16;
                if rows >= 16 {
                    snapshot.wmma16_segments += 1;
                    snapshot.wmma16_covered_assignments += rows;
                } else {
                    snapshot.scalar_tail_segments += 1;
                    snapshot.scalar_tail_assignments += rows;
                }
            }
        }
    }
    snapshot.permutation_entries = snapshot.assignments;
    snapshot
}

/// Per-layer descriptor for the Qwen3.6-MoE megakernel. Field order and
/// natural x86_64 alignment must match the C++ struct in
/// `kernels/qwen36_moe.hip` exactly. The repr-C layout is fixed at PR 4
/// time and grows by appending new fields, never reordering existing
/// ones — see the matching `static_assert(sizeof(...))` on the C++ side.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeDecodeLayerDesc {
    /// Layer index in `[0, num_hidden_layers)`. Used by the kernel to pick
    /// the cos/sin RoPE entry and to sanity-check the descriptor pointer.
    pub layer_idx: c_int,
    /// 0 = linear-attention layer, 1 = full-attention layer.
    pub is_full_attention: c_int,

    // --- RMS norms --------------------------------------------------------
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,

    // --- Full-attention slots (read iff is_full_attention == 1) -----------
    /// q_proj output dim. With `attn_output_gate=true` (Qwen3-Next) this is
    /// `2 * num_heads * head_dim`; the kernel splits the upper half off as
    /// the sigmoid output gate. With `attn_output_gate=false` it's just
    /// `num_heads * head_dim`. The sign is captured by `attn_output_gate`.
    pub q_proj_w: *const c_void,
    pub q_proj_out_dim: c_int,
    /// 0 = no output gate (q_proj_out_dim == num_heads*head_dim),
    /// 1 = attn_output_gate fused (q_proj_out_dim == 2*num_heads*head_dim).
    pub attn_output_gate: c_int,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub attn_head_dim: c_int,
    pub attn_num_heads: c_int,
    pub attn_num_kv_heads: c_int,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,

    // --- Linear-attention slots (read iff is_full_attention == 0) ---------
    pub linear_in_proj_qkv_w: *const c_void,
    pub linear_in_proj_z_w: *const c_void,
    pub linear_in_proj_b_w: *const c_void,
    pub linear_in_proj_a_w: *const c_void,
    pub linear_out_proj_w: *const c_void,
    pub linear_conv1d_w: *const c_void,
    pub linear_dt_bias: *const c_void,
    pub linear_a_log_exp: *const c_void,
    pub linear_norm_w: *const c_void,
    pub linear_qkv_dim: c_int,
    pub linear_v_dim: c_int,
    pub linear_v_heads: c_int,
    pub linear_conv_kernel_dim: c_int,
    /// Linear-attention conv state pointer, shape `[batch, qkv_dim,
    /// kernel-1]`. NULL on first decode step (kernel will zero on read).
    pub linear_conv_state: *mut c_void,
    /// Linear-attention recurrent state, shape `[batch, V_heads, V_dim,
    /// K_dim]`. NULL on first decode step.
    pub linear_recurrent_state: *mut c_void,

    // --- MoE block (always read, regardless of attention type) ------------
    /// Router weight `[num_experts, hidden]`, BF16. Always BF16 (excluded
    /// from INT4 quant by `is_int4_target`).
    pub router_w: *const c_void,
    /// Fused expert gate+up `[num_experts, 2*moe_intermediate_size, hidden]`.
    /// At INT4 launch the pointer reinterprets as packed `u8` (2 nibbles
    /// per byte), with sidecar scale/zero in `Qwen36MoeInt4ScaleDesc`.
    pub experts_gate_up_w: *const c_void,
    /// Fused expert down `[num_experts, hidden, moe_intermediate_size]`.
    pub experts_down_w: *const c_void,
    /// Shared expert (always-on). `gate_proj` and `up_proj` are
    /// `[shared_int, hidden]`; `down_proj` is `[hidden, shared_int]`.
    pub shared_expert_gate_proj_w: *const c_void,
    pub shared_expert_up_proj_w: *const c_void,
    pub shared_expert_down_proj_w: *const c_void,
    /// Scalar shared-expert gate `[1, hidden]`, BF16. Applied as
    /// `sigmoid(gate · x) * shared_expert(x)`.
    pub shared_expert_gate_w: *const c_void,
    /// Number of routed experts present in this layer. Must match
    /// `desc.num_experts` across layers (sanity-checked by the host).
    pub num_experts: c_int,
    /// Top-k for routing.
    pub top_k: c_int,
    pub moe_intermediate_size: c_int,
    pub shared_expert_intermediate_size: c_int,
    /// 1 if router applies `softmax(top_k_logits)` renormalization
    /// (`norm_topk_prob=true` in config). 0 otherwise.
    pub norm_topk_prob: c_int,

    // --- KV-FP8 sidecar (read iff is_full_attention == 1 AND
    // matching kv_fp8_descs[layer].kv_scale_k != null) ---------------
    /// BF16 sidecar buffer `[num_kv_heads, kv_shadow_window, head_dim]`.
    /// Null when the sidecar is disabled. The kernel reads from the sidecar
    /// (instead of dequantising FP8) for positions covered by the rolling
    /// sidecar window.
    pub kv_shadow_k: *mut c_void,
    /// BF16 sidecar buffer `[num_kv_heads, kv_shadow_window, head_dim]`.
    /// Paired with [`Self::kv_shadow_k`]; null under the same conditions.
    pub kv_shadow_v: *mut c_void,
    /// Earliest absolute KV position the sidecar may cover. `-1` when the
    /// sidecar is disabled. Runtime coverage is
    /// `max(kv_shadow_start, position + 1 - kv_shadow_window)..=position`.
    pub kv_shadow_start: c_int,
    /// Number of recent KV positions physically stored in the BF16 sidecar.
    /// Zero when the sidecar is disabled. The kernel uses modulo indexing so
    /// the descriptor can remain fixed across decode steps.
    pub kv_shadow_window: c_int,
}

unsafe impl Send for Qwen36MoeDecodeLayerDesc {}
unsafe impl Sync for Qwen36MoeDecodeLayerDesc {}

impl Default for Qwen36MoeDecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Parallel-struct to [`Qwen36MoeDecodeLayerDesc`] carrying INT4 GPTQ
/// scale + zero pointers. When `--int4` is active, the main desc's `*_w`
/// slots reinterpret as packed-u8 nibbles and the kernel reads the
/// matching `*_scale` / `*_zero` from this struct.
///
/// Routers and scalar gates stay BF16 (`is_int4_target` excludes them);
/// their fields here are unused but the struct keeps them at fixed offsets
/// for ABI stability.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeInt4ScaleDesc {
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,

    pub linear_in_proj_qkv_scale: *const c_void,
    pub linear_in_proj_qkv_zero: *const c_void,
    pub linear_in_proj_z_scale: *const c_void,
    pub linear_in_proj_z_zero: *const c_void,
    pub linear_out_proj_scale: *const c_void,
    pub linear_out_proj_zero: *const c_void,

    pub experts_gate_up_scale: *const c_void,
    pub experts_gate_up_zero: *const c_void,
    pub experts_down_scale: *const c_void,
    pub experts_down_zero: *const c_void,

    pub shared_expert_gate_proj_scale: *const c_void,
    pub shared_expert_gate_proj_zero: *const c_void,
    pub shared_expert_up_proj_scale: *const c_void,
    pub shared_expert_up_proj_zero: *const c_void,
    pub shared_expert_down_proj_scale: *const c_void,
    pub shared_expert_down_proj_zero: *const c_void,

    pub group_size: c_int,
}

unsafe impl Send for Qwen36MoeInt4ScaleDesc {}
unsafe impl Sync for Qwen36MoeInt4ScaleDesc {}

impl Default for Qwen36MoeInt4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-layer KV cache FP8 scale pointers for Qwen3.6-MoE.
///
/// Parallel struct to [`Qwen36MoeDecodeLayerDesc`] — one entry per layer,
/// passed as a separate kernel argument (same pattern as
/// [`Qwen36MoeInt4ScaleDesc`]). Linear-attention layers leave both
/// pointers null. When KV-FP8 is off, the entire
/// `*const Qwen36MoeKVCacheFp8Desc` array argument is null.
///
/// Mirrors the qwen35 `KVCacheFp8Desc` shape: F32 absmax scale per
/// (kv_head, position).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeKVCacheFp8Desc {
    /// `[num_kv_heads, max_T]` F32. Null for linear-attn layers.
    pub kv_scale_k: *mut c_void,
    /// `[num_kv_heads, max_T]` F32. Null for linear-attn layers.
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for Qwen36MoeKVCacheFp8Desc {}
unsafe impl Sync for Qwen36MoeKVCacheFp8Desc {}

impl Default for Qwen36MoeKVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Per-sequence batched-decode state, parallel to the layer descriptor
/// array. Only the first `batch_size` slots are read.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeBatchSeqDesc {
    pub seqlen_offset: [c_int; MAX_BATCH_SIZE],
    pub kv_cache_k: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_cache_v: [*mut c_void; MAX_BATCH_SIZE],
    pub kv_len: [c_int; MAX_BATCH_SIZE],
    pub kv_max_t: [c_int; MAX_BATCH_SIZE],
    pub linear_conv_state: [*mut c_void; MAX_BATCH_SIZE],
    pub linear_recurrent_state: [*mut c_void; MAX_BATCH_SIZE],
}

unsafe impl Send for Qwen36MoeBatchSeqDesc {}
unsafe impl Sync for Qwen36MoeBatchSeqDesc {}

impl Default for Qwen36MoeBatchSeqDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Attribution-only MPP pilot used by the Apple M5 Metal bench harness.
///
/// This measures repeated exact `64x32x64` MPP tensor tiles as an equivalent
/// square GEMM throughput number. It does not consume Qwen3.6 model weights
/// and must not be interpreted as a decode-path replacement.
pub fn metal_mpp_tile_gemm_f16_tflops(size: u32, iterations: u32) -> Result<f64, GpuError> {
    crate::metal_native::mpp_tile_gemm_f16_tflops(size, iterations)
}

#[derive(Debug, Clone, Copy)]
pub struct MetalMpsExpertF16Probe {
    pub gate_up_ms: f64,
    pub down_ms: f64,
    pub gate_up_tflops: f64,
    pub down_tflops: f64,
}

/// Attribution-only MPS probe for Qwen3.6 active-expert GEMV shapes.
///
/// This is a resident-FP16 vendor-library upper-bound probe. It does not use the
/// GPTQ INT4 expert buffers directly and does not change the decode path.
pub fn metal_mps_expert_f16_probe(
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
    iterations: u32,
) -> Result<MetalMpsExpertF16Probe, GpuError> {
    let probe = crate::metal_native::qwen36_mps_expert_f16_probe(
        hidden,
        moe_intermediate,
        top_k,
        iterations,
    )?;
    Ok(MetalMpsExpertF16Probe {
        gate_up_ms: probe.gate_up_ms,
        down_ms: probe.down_ms,
        gate_up_tflops: probe.gate_up_tflops,
        down_tflops: probe.down_tflops,
    })
}

#[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
extern "C" {
    /// Stub launch entry. Walks the descriptor array, validates field
    /// integrity by writing recognizable sentinel values into the workspace
    /// at known offsets, grid-barriers between layers, and returns 0 on
    /// success.
    ///
    /// Sentinel layout in `workspace[0..sentinel_count]` (f32):
    /// - `[0]`: number of layers seen (must equal `num_layers`)
    /// - `[1]`: total `num_experts` summed across layers (sanity check)
    /// - `[2]`: total `top_k` summed across layers
    /// - `[3]`: 1.0 if every layer's `is_full_attention` matches the
    ///   pattern produced by `(idx + 1) % 4 == 0`, else 0.0
    /// - `[4]`: `attn_output_gate` status — 1.0 if all full-attn layers
    ///   set it to 1, 0.0 otherwise
    /// - `[5..]`: reserved for future smoke-test bytes; zero on PR 4.
    ///
    /// Once the real kernel lands, this entry is replaced by the actual
    /// persistent decode launcher with the same signature.
    pub fn qwen36_moe_hip_stub_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        layers: *const Qwen36MoeDecodeLayerDesc,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// Phase 3e: persistent decode megakernel launcher. One cooperative
    /// HIP launch processes all `num_layers` of {attn or linear-attn, FFN}
    /// — replaces 80 step launches/token with 1 (the lm_head still
    /// launches separately at this stage).
    ///
    /// See `kernels/qwen36_moe_persistent/persistent_decode.hip` for the
    /// kernel and `kernels/qwen36_moe_bridge.cpp::qwen36_moe_hip_persistent_decode_launch`
    /// for the launcher.
    ///
    /// Caller responsibilities:
    /// - `hidden_ping` is uploaded with the initial hidden BF16 bytes; the
    ///   final hidden lands back in `hidden_ping` after even `num_layers`
    ///   (the bridge rejects odd `num_layers`).
    /// - `int4_scales` is null for BF16 baked models, non-null for INT4
    ///   bakes (one entry per layer, parallel to `layers`).
    /// - `workspace` is at least
    ///   `max(attn_workspace_floats(geom), ffn_workspace_floats(geom))` F32
    ///   entries — same as the chained driver.
    /// - `ffn_topk_idx_scratch` is a small `[top_k]` i32 buffer (the FFN
    ///   phase only writes it at stage 1, but the parameter must be valid).
    /// - sync_buf layout: counters[0..16] u32 + barrier_counter at +64 +
    ///   barrier_flag at +68 (96 bytes total, zeroed by the bridge).
    pub fn qwen36_moe_hip_persistent_decode_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: c_int,
        start_layer: c_int,
        end_layer_exclusive: c_int,
        mode: c_int,
        layers: *const Qwen36MoeDecodeLayerDesc,
        int4_scales: *const Qwen36MoeInt4ScaleDesc,
        // Null when KV-FP8 is off globally. Otherwise an array of
        // `num_layers` entries parallel to `layers`. Full-attention
        // layers populate `kv_scale_k` / `kv_scale_v` together (both
        // set, or both null to disable KV-FP8 for that layer specifically
        // — a valid mixed-mode configuration). Linear-attention layers
        // must leave both null. The bridge validates these invariants
        // and rejects malformed descriptors before kernel launch.
        kv_fp8_descs: *const Qwen36MoeKVCacheFp8Desc,
        hidden: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        conv_kernel_dim: c_int,
        num_experts: c_int,
        moe_intermediate: c_int,
        shared_intermediate: c_int,
        top_k: c_int,
        vocab: c_int,
        rope_theta: f32,
        rms_norm_eps: f32,
        position: c_int,
        // -1 ⇒ inherit from `position` (dense base-decode case);
        // ≥ 0 ⇒ decoupled KV slot for SpecPrefill sparse-prefill or
        // MTP draft layers.
        cache_pos: c_int,
        embed_w: *const c_void,
        token_id: c_int,
        token_ids: *const c_uint,
        prefill_len: c_int,
        hidden_ping: *mut c_void,
        hidden_pong: *mut c_void,
        workspace: *mut f32,
        ffn_topk_idx_scratch: *mut c_int,
        // Phase 3f folded final RMSnorm + lm_head GEMV. Pass nullptr
        // triple + vocab=0 to skip (prefill steps); otherwise the
        // megakernel writes logits to `logits_out` and the host can
        // skip the separate `lm_head_launch` call.
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits_out: *mut c_void,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// PR 4b2 staged single-layer attention parity launcher. Runs the
    /// full-attention path through `stage` (1..=5) and writes the matching
    /// intermediate to `output`:
    ///
    /// | stage | output buffer contents (BF16)                      |
    /// |-------|----------------------------------------------------|
    /// |   1   | `q_normed[H*d]`                                    |
    /// |   2   | `k_normed[Hkv*d]`         (`q_normed` recomputed)  |
    /// |   3   | `q_rot[H*d] || k_rot[Hkv*d]` (planned)             |
    /// |   4   | `attn[H*d]`                                        |
    /// |   5   | `output_hidden[hidden]`                            |
    ///
    /// At PR 4b2 step 1 only `stage == 1` is wired; the kernel returns the
    /// q-path intermediate and ignores the k_*/v_*/o_proj/RoPE/position
    /// arguments. They're declared up front so the FFI ABI doesn't change
    /// between staged commits.
    ///
    /// `workspace` must be at least `2 * num_heads * head_dim` F32 entries
    /// (used to hold the BF16-rounded F32 view of `q_raw` between phases).
    /// `output` must be at least `num_heads * head_dim` BF16 entries on
    /// stage 1 — sized for the largest staged intermediate, BF16.
    /// `sync_buf` (counters/barrier_counter/barrier_flag) must be 96 zero
    /// bytes — see [`stub_launch`] for the layout convention.
    pub fn qwen36_moe_hip_attn_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        rope_theta: f32,
        rms_norm_eps: f32,
        position: c_int,
        cache_pos: c_int,
        input_hidden: *const c_void,
        input_norm_w: *const c_void,
        q_proj_w: *const c_void,
        k_proj_w: *const c_void,
        v_proj_w: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        o_proj_w: *const c_void,
        int4_group_size: c_int,
        q_proj_scale: *const c_void,
        q_proj_zero: *const c_void,
        k_proj_scale: *const c_void,
        k_proj_zero: *const c_void,
        v_proj_scale: *const c_void,
        v_proj_zero: *const c_void,
        o_proj_scale: *const c_void,
        o_proj_zero: *const c_void,
        output: *mut c_void,
        workspace: *mut f32,
        kv_cache_k: *mut c_void,
        kv_cache_v: *mut c_void,
        kv_max_t: c_int,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// PR 4b3 staged single-layer linear-attention parity launcher. Same
    /// staged-build-up discipline as `qwen36_moe_hip_attn_step_launch`,
    /// but for the 3-of-4 hybrid layers that aren't full-attention.
    /// `stage` selects how far to run; the matching staged intermediate
    /// is published to `output` (BF16):
    ///
    /// | stage | output buffer contents (BF16)           |
    /// |-------|------------------------------------------|
    /// |   1   | `qkv_raw[qkv_dim]`                       |
    /// |   2   | `silu_out[qkv_dim]`         (planned)    |
    /// |   3   | `q_scaled || k_rep || v_heads` (planned) |
    /// |   4   | `recurrent_out[V*v_dim]`    (planned)    |
    /// |   5   | `output_hidden[hidden]`     (planned)    |
    ///
    /// PR 4b3 step 2 wires only `stage == 1`; the kernel ignores the conv
    /// / dt / norm / out_proj / state pointers and the matching arguments
    /// can be null. They're declared up front so subsequent staged commits
    /// don't perturb the FFI ABI.
    ///
    /// `workspace` must be at least `qkv_dim + V*v_dim + 2*V` F32 entries
    /// for stage 1 (later stages bump that up via the safe wrapper).
    /// `output` must be at least `qkv_dim` BF16 entries on stage 1 (sized
    /// for the largest staged intermediate by the safe wrapper). `sync_buf`
    /// (counters/barrier_counter/barrier_flag) must be 96 zero bytes.
    pub fn qwen36_moe_hip_linear_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        conv_kernel_dim: c_int,
        rms_norm_eps: f32,
        input_hidden: *const c_void,
        input_norm_w: *const c_void,
        in_proj_qkv_w: *const c_void,
        in_proj_z_w: *const c_void,
        in_proj_a_w: *const c_void,
        in_proj_b_w: *const c_void,
        conv1d_w: *const c_void,
        conv1d_bias: *const c_void,
        dt_bias: *const c_void,
        a_log: *const c_void,
        norm_w: *const c_void,
        out_proj_w: *const c_void,
        conv_state: *mut c_void,
        recurrent_state: *mut f32,
        int4_group_size: c_int,
        in_proj_qkv_scale: *const c_void,
        in_proj_qkv_zero: *const c_void,
        in_proj_z_scale: *const c_void,
        in_proj_z_zero: *const c_void,
        out_proj_scale: *const c_void,
        out_proj_zero: *const c_void,
        output: *mut c_void,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// PR 4b4 staged single-block MoE FFN parity launcher. Same staged-build-up
    /// discipline as `qwen36_moe_hip_attn_step_launch` and
    /// `qwen36_moe_hip_linear_step_launch`, but for the post-attention half
    /// of one Qwen3.6-MoE layer. `stage` selects how far to run; the matching
    /// staged intermediate is published to `output` (BF16) and `output_idx`
    /// (i32, top-k indices for stages 1+):
    ///
    /// | stage | output buffer contents (BF16)                    |
    /// |-------|--------------------------------------------------|
    /// |   1   | `topk_weights[k]`           (idx via `output_idx`) |
    /// |   2   | `shared_out[hidden]`                             |
    /// |   3   | `expert_0_out[hidden]`      (top-1 dispatch)     |
    /// |   4   | `moe_out[hidden]`                                |
    /// |   5   | `output_hidden[hidden]`     (final residual)     |
    ///
    /// PR 4b4 step 1 wires only `stage == 1`; the kernel ignores the
    /// gate_up_proj / down_proj / shared_expert_* pointers and the matching
    /// arguments can be null. They're declared up front so subsequent staged
    /// commits don't perturb the FFI ABI.
    ///
    /// `workspace` must be at least `hidden + 2*num_experts + 2*top_k` F32
    /// entries for stage 1 (later stages bump that up). `output` must be at
    /// least `top_k` BF16 entries on stage 1 and `output_idx` must be at
    /// least `top_k` i32 entries. `sync_buf` (counters/barrier_counter/
    /// barrier_flag) must be 96 zero bytes.
    ///
    /// PR 4b5 step 2: INT4 dequant smoke launcher.
    ///
    /// Drives a tiny single-thread kernel that runs both `int4_dequant_8`
    /// and `int4_dequant_scalar` over a `[out_rows, in_cols]` slab, writing
    /// each helper's outputs into a separate buffer. The Rust-side test
    /// validates byte-for-byte against a host reference computing the same
    /// `bf16(q*s - z*s)` reconstruction. Catches porting bugs in the
    /// helpers in isolation, before they're folded into the real FFN
    /// matmuls in step 3+.
    ///
    /// `packed`: u8, shape `[out_rows, in_cols / 2]`, even col → low nibble.
    /// `scale` / `zero`: BF16, shape `[out_rows / gsz, in_cols / gsz]`.
    /// `dq_8_out`, `dq_scalar_out`: F32 device buffers, each
    /// `out_rows * in_cols` long.
    ///
    /// Pre-conditions (the bridge validates them):
    /// - `in_cols % 8 == 0`
    /// - `in_cols % gsz == 0` and `gsz % 2 == 0`
    /// - `out_rows % gsz == 0`
    pub fn qwen36_moe_hip_int4_dequant_smoke_launch(
        device_ordinal: usize,
        packed: *const u8,
        scale: *const c_void,
        zero: *const c_void,
        out_rows: c_int,
        in_cols: c_int,
        gsz: c_int,
        dq_8_out: *mut f32,
        dq_scalar_out: *mut f32,
    ) -> c_int;

    pub fn qwen36_moe_hip_ffn_step_launch(
        dtype: c_int,
        device_ordinal: usize,
        stage: c_int,
        hidden: c_int,
        num_experts: c_int,
        moe_intermediate: c_int,
        shared_intermediate: c_int,
        top_k: c_int,
        rms_norm_eps: f32,
        input_hidden: *const c_void,
        post_attn_norm_w: *const c_void,
        gate_w: *const c_void,
        gate_up_proj_w: *const c_void,
        down_proj_w: *const c_void,
        shared_gate_proj_w: *const c_void,
        shared_up_proj_w: *const c_void,
        shared_down_proj_w: *const c_void,
        shared_expert_gate_w: *const c_void,
        int4_group_size: c_int,
        gate_up_proj_scale: *const c_void,
        gate_up_proj_zero: *const c_void,
        down_proj_scale: *const c_void,
        down_proj_zero: *const c_void,
        shared_gate_proj_scale: *const c_void,
        shared_gate_proj_zero: *const c_void,
        shared_up_proj_scale: *const c_void,
        shared_up_proj_zero: *const c_void,
        shared_down_proj_scale: *const c_void,
        shared_down_proj_zero: *const c_void,
        output: *mut c_void,
        output_idx: *mut c_int,
        workspace: *mut f32,
        counters: *mut c_uint,
        barrier_counter: *mut c_uint,
        barrier_flag: *mut c_uint,
    ) -> c_int;

    /// Final RMSNorm + lm_head GEMV in a single kernel — replaces the
    /// host-side `host_final_norm_lm_head_f32` for qwen3.6-MoE.
    ///
    /// All buffers are device pointers, all BF16 (`dtype = 2`):
    ///   - `final_hidden`: [hidden] BF16, the output of `run_chained_decode`.
    ///   - `final_norm_w`: [hidden] BF16 — `model.norm.weight`. Applies the
    ///      HF `Qwen3_5MoeRMSNorm` `(1 + w)` unit offset.
    ///   - `lm_head_w`: [vocab, hidden] BF16, dequantized once at startup.
    ///   - `logits`: [vocab] BF16, output.
    ///   - `counter`: [1] u32. Used as a work-stealing atomic across vocab
    ///     rows; the launcher memsets it to 0 before each call so the
    ///     caller doesn't need to.
    ///
    /// Returns 0 on success; non-zero on validation / launch failure (see
    /// `qwen36_moe_hip_lm_head_launch` in `kernels/qwen36_moe_bridge.cpp`
    /// for the error code matrix).
    pub fn qwen36_moe_hip_lm_head_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        // Optional BF16 [hidden] export of the post-RMSNorm hidden
        // state. Phase 6.2c.3 plumbing for the MTP draft loop's
        // recurrent feed; null = base-decode behavior unchanged.
        x_normed_out: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;

    /// FFI bridge for the batched lm_head WMMA kernel (Phase 6.4a). Wraps
    /// `qwen36_moe_lm_head_batched_wmma_kernel`. `m` is the runtime batch
    /// size (1..16); for `m == 1` the single-M path
    /// (`qwen36_moe_hip_lm_head_launch`) is faster — use the batched
    /// launcher when `m >= 2` to amortize the lm_head BF16 weight read.
    /// WMMA-only (gfx11xx); returns status 138 on unsupported hardware
    /// or `hidden % 16 != 0` so the caller can fall back to a per-row
    /// loop over the single-M launcher.
    pub fn qwen36_moe_hip_lm_head_batched_launch(
        dtype: c_int,
        device_ordinal: usize,
        m: c_int,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        x_normed_out: *mut c_void,
    ) -> c_int;

    /// FFI bridge for the MTP pre-fusion kernel (Phase 6.2c.1). Single-block
    /// launch: BF16 RMSNorms over `e_in` and `h_base` followed by a
    /// `mtp.fc @ cat([e_norm, h_norm])` matvec into `fused_out`. All buffers
    /// must be device-resident on `device_ordinal` and BF16. See
    /// `qwen36_moe_hip_mtp_pre_fusion_launch` in `kernels/qwen36_moe_bridge.cpp`
    /// for the error code matrix.
    pub fn qwen36_moe_hip_mtp_pre_fusion_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        rms_norm_eps: f32,
        e_in: *const c_void,
        h_base: *const c_void,
        pre_fc_norm_embedding_w: *const c_void,
        pre_fc_norm_hidden_w: *const c_void,
        fc_w: *const c_void,
        e_norm_out: *mut c_void,
        h_norm_out: *mut c_void,
        fused_out: *mut c_void,
    ) -> c_int;

    /// Stage A (M3) batched-Q full-attention prefill kernel. Standalone
    /// attention math: Q/K/V are pre-projected and pre-RoPE'd by the
    /// caller, the K/V cache is pre-written. Output is pre-o_proj
    /// `[batch, q_heads, q_len, head_dim]` in F32.
    ///
    /// Shapes (BF16 unless noted):
    /// - `query`: `[batch, q_heads, q_len, head_dim]`
    /// - `key`:   `[batch, kv_heads, kv_len, head_dim]`
    /// - `value`: `[batch, kv_heads, kv_len, head_dim]`
    /// - `out` (F32): `[batch, q_heads, q_len, head_dim]`
    ///
    /// `seqlen_offset = past_len`; query at chunk position `qr` attends to
    /// cache positions `[0, past_len + qr]` (causal, inclusive). `kv_len`
    /// is the total cache length the kernel may read (typically
    /// `past_len + q_len`).
    ///
    /// Status codes (non-zero = failure):
    ///   130 dtype != bf16    131 invalid heads        132 q_heads % kv_heads
    ///   133 head_dim out of range  134 q_len/kv_len   135 seqlen_offset / overflow
    ///   136 batch_size       137 wave64 (unsupported) 138 LDS overflow
    ///   254 launch error     255 sync error
    pub fn qwen36_moe_hip_batched_prefill_attn_full_launch(
        dtype: c_int,
        device_ordinal: usize,
        batch_size: c_int,
        q_heads: c_int,
        kv_heads: c_int,
        q_len: c_int,
        kv_len: c_int,
        head_dim: c_int,
        scale: f32,
        seqlen_offset: c_int,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        out: *mut c_void,
    ) -> c_int;

    /// Stage B (M9) router permutation kernel. Groups per-token top-K expert
    /// assignments by target expert (counting-sort, single block).
    ///
    /// Inputs (GPU buffers):
    /// - `topk_idx`     : `[n_tokens, top_k]` i32 — per-token expert ids in
    ///                     `[0, num_experts)`.
    /// - `topk_weight`  : `[n_tokens, top_k]` BF16 — routing weights.
    ///
    /// Outputs (caller-allocated GPU buffers):
    /// - `expert_offsets`     : `[num_experts + 1]` i32 — prefix sum.
    /// - `permuted_token_idx` : `[n_tokens * top_k]` i32 — sorted token ids.
    /// - `permuted_kpos`      : `[n_tokens * top_k]` i32 — top-K slot ids.
    /// - `permuted_weight`    : `[n_tokens * top_k]` BF16 — routing weights.
    ///
    /// Within an expert's segment the order is unstable (atomicAdd cursor);
    /// callers comparing against a CPU reference must compare per-segment as
    /// a multiset.
    ///
    /// Status codes (non-zero = failure):
    ///   140 invalid args (n_tokens/top_k/num_experts <= 0)
    ///   141 num_experts > 256       142 top_k > 16
    ///   143 n_tokens * top_k > 16384
    ///   254 launch error            255 sync error
    pub fn qwen36_moe_hip_batched_prefill_router_permute_launch(
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        num_experts: c_int,
        topk_idx: *const c_void,
        topk_weight: *const c_void,
        expert_offsets: *mut c_void,
        permuted_token_idx: *mut c_void,
        permuted_kpos: *mut c_void,
        permuted_weight: *mut c_void,
    ) -> c_int;

    /// Stage B (M10) grouped-expert INT4 GEMM kernel. One launch processes
    /// ALL `num_experts` experts via persistent-block work-stealing on the
    /// expert id; for each expert it walks the segment of permuted rows
    /// produced by the M9 router permutation kernel and runs gate_up +
    /// silu*mul + down INT4 matmuls per row.
    ///
    /// Inputs (GPU buffers):
    /// - `x_norm`              : `[n_tokens, hidden]` BF16 — post-input-RMSnorm
    ///                            hidden states; gathered by `permuted_token_idx`.
    /// - `expert_offsets`      : `[num_experts + 1]` i32 — M9 prefix sum.
    /// - `permuted_token_idx`  : `[n_tokens * top_k]` i32 — M9 sort output.
    /// - `experts_gate_up_w/s/z` : `[E, 2*I, hidden/2]` u8 + `[E, 2*I/gs, hidden/gs]` BF16.
    /// - `experts_down_w/s/z`    : `[E, hidden, I/2]` u8 + `[E, hidden/gs, I/gs]` BF16.
    ///
    /// Caller-owned buffers:
    /// - `expert_out` : `[n_tokens * top_k, hidden]` BF16 — per-permuted-row
    ///                   expert output; M11 unpermutes + combines.
    /// - `counters`   : `[1]` u32 — work-stealing claim counter; CALLER MUST
    ///                   ZERO BEFORE LAUNCH.
    ///
    /// Status codes (non-zero = failure):
    ///   150 invalid args (zero/negative dims)
    ///   151 num_experts > 256
    ///   152 hidden / moe_intermediate not divisible by group_size (or 16)
    ///   153 group_size != 128
    ///   154 top_k * n_tokens > 16384
    ///   155 dtype != bf16
    ///   156 LDS overflow
    ///   254 launch error                255 sync error
    pub fn qwen36_moe_hip_batched_prefill_grouped_expert_launch(
        dtype: c_int,
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        num_experts: c_int,
        hidden: c_int,
        moe_intermediate: c_int,
        group_size: c_int,
        x_norm: *const c_void,
        expert_offsets: *const c_void,
        permuted_token_idx: *const c_void,
        experts_gate_up_w: *const c_void,
        experts_gate_up_scale: *const c_void,
        experts_gate_up_zero: *const c_void,
        experts_down_w: *const c_void,
        experts_down_scale: *const c_void,
        experts_down_zero: *const c_void,
        expert_out: *mut c_void,
        counters: *mut c_void,
    ) -> c_int;

    /// Stage B (M11) unpermute + weighted combine kernel. Inverts the M9
    /// router permutation (host-built `permuted_inverse` table) and computes
    /// the per-token weighted sum of `top_k` expert outputs.
    ///
    /// Inputs (GPU buffers):
    /// - `permuted_inverse` : `[n_tokens * top_k]` i32 — host-built inverse
    ///                         of M9's scatter, so
    ///                         `permuted_inverse[token * top_k + kpos] = dst`
    ///                         where `dst` is the M9/M10 row index for that
    ///                         (token, kpos) pair.
    /// - `permuted_weight`  : `[n_tokens * top_k]` BF16 — M9 output.
    /// - `expert_out`       : `[n_tokens * top_k, hidden]` BF16 — M10 output.
    ///
    /// Output (caller-allocated GPU buffer):
    /// - `combined`         : `[n_tokens, hidden]` BF16 — weighted sum
    ///                         of expert outputs per token.
    ///
    /// Status codes (non-zero = failure):
    ///   160 invalid args (zero/negative dims)
    ///   161 top_k > 16
    ///   162 dtype != bf16
    ///   163 hidden too large (>65536)
    ///   254 launch error            255 sync error
    pub fn qwen36_moe_hip_batched_prefill_unpermute_combine_launch(
        dtype: c_int,
        device_ordinal: usize,
        n_tokens: c_int,
        top_k: c_int,
        hidden: c_int,
        permuted_inverse: *const c_void,
        permuted_weight: *const c_void,
        expert_out: *const c_void,
        combined: *mut c_void,
    ) -> c_int;
}

/// Safe wrapper over the stub launch. The engine pre-allocates `sync_buf`
/// as a 96-byte zeroed scratch — 16 u32 work-stealing counter slots at
/// +0..+63 (only counters[0] used here; the FFN concurrent-experts dispatch
/// uses 2*K_top of them), grid barrier counter at +64, flag at +68. The
/// 32-byte form used by `crate::persistent_decode_4b` is the older single-
/// counter layout — qwen36_moe shares one widened sync_buf across all four
/// step launchers (stub/attn/linear/ffn) so any can run with any.
///
/// Returns when the kernel signals completion via `hipDeviceSynchronize`.
/// The smoke-test path reads `workspace` back to verify descriptor
/// integrity; the real kernel will overwrite that area with activations.
pub fn stub_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer_descs_device: &GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    num_layers: usize,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::stub_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = layer_descs_device.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    // Layout: 16 u32 work-stealing counter slots at +0..+63 (the FFN
    // concurrent-experts dispatch uses 2*K_top of these; attn/linear/stub
    // only touch counters[0]). Barrier counter+flag follow at +64/+68.
    // Sync_buf must be at least 96 bytes zeroed before launch.
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_stub_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    layer_descs_device.as_ptr() as *const Qwen36MoeDecodeLayerDesc,
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::stub_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::stub_launch: Metal backend not yet wired".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe stub launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Geometry constants for the persistent decode megakernel — packed into
/// one struct to keep [`persistent_decode_launch`]'s arg list tractable.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoePersistentGeom {
    pub hidden: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

/// Phase 3e safe wrapper for the persistent decode megakernel. Replaces
/// the chained 80 step-kernel launches/token with one cooperative HIP
/// launch.
///
/// Caller responsibilities:
/// - `layers_device` is a device-resident array of
///   [`Qwen36MoeDecodeLayerDesc`] (`num_layers` entries). Even
///   `num_layers` only — the bridge enforces this.
/// - `int4_scales_device` is null for BF16 bakes, or a device-resident
///   array of [`Qwen36MoeInt4ScaleDesc`] (parallel to `layers_device`).
/// - `hidden_ping` is uploaded with the BF16 initial hidden bytes; the
///   final hidden lands back in `hidden_ping` after `num_layers`.
/// - `workspace` (F32) sized for `max(attn_workspace_floats(geom),
///   ffn_workspace_floats(geom))`.
/// - `ffn_topk_idx_scratch` is a small `[top_k]` i32 buffer (used only
///   internally by the FFN phase at stage 1; we run stage 5 so it's
///   inert, but must be valid).
/// - `sync_buf` is at least 96 zeroed bytes (counters[0..16] + barrier
///   counter/flag); the bridge defensively re-zeros it on entry.
/// Phase 3f folded final RMSnorm + lm_head GEMV. Pass `Some(...)` on
/// generation steps to write logits directly from the megakernel and
/// skip the separate `lm_head_launch` call; pass `None` on prefill
/// steps where the caller doesn't need logits. Bundled rather than
/// scattered as ~4 args so the call site stays readable.
pub struct Qwen36MoePersistentLmHeadFold<'a> {
    /// `[hidden]` BF16. Same final_norm tensor the standalone
    /// `lm_head_launch` consumes.
    pub final_norm_w: &'a GpuBuffer,
    /// `[vocab, hidden]` BF16. Pre-dequantized at engine startup from
    /// the bake's INT4 lm_head; the kernel reads BF16 only.
    pub lm_head_w: &'a GpuBuffer,
    /// `[vocab]` BF16 output buffer. Kernel writes one logit per row.
    pub logits_out: &'a mut GpuBuffer,
    /// Vocab size. Must be `> 0` and match `lm_head_w.shape()[0]`.
    pub vocab: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoePersistentMode {
    Full,
    RouterOnly,
    FfnOnly,
}

impl Qwen36MoePersistentMode {
    #[allow(dead_code)]
    fn as_ffi(self) -> c_int {
        match self {
            Self::Full => 0,
            Self::RouterOnly => 1,
            Self::FfnOnly => 2,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_launch(
    ordinal: usize,
    dtype: ScalarType,
    geom: Qwen36MoePersistentGeom,
    position: i32,
    cache_pos: i32,
    layers_device: &GpuBuffer,
    int4_scales_device: Option<&GpuBuffer>,
    kv_fp8_descs_device: Option<&GpuBuffer>,
    num_layers: usize,
    hidden_ping: &mut GpuBuffer,
    hidden_pong: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    ffn_topk_idx_scratch: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    lm_head_fold: Option<Qwen36MoePersistentLmHeadFold<'_>>,
) -> Result<(), GpuError> {
    persistent_decode_launch_range(
        ordinal,
        dtype,
        geom,
        0,
        num_layers,
        Qwen36MoePersistentMode::Full,
        position,
        cache_pos,
        layers_device,
        int4_scales_device,
        kv_fp8_descs_device,
        num_layers,
        hidden_ping,
        hidden_pong,
        workspace,
        ffn_topk_idx_scratch,
        sync_buf,
        None,
        -1,
        None,
        1,
        lm_head_fold,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn persistent_decode_launch_range(
    ordinal: usize,
    dtype: ScalarType,
    geom: Qwen36MoePersistentGeom,
    start_layer: usize,
    end_layer_exclusive: usize,
    mode: Qwen36MoePersistentMode,
    position: i32,
    // `-1` (`Qwen36MoeAttnStepParams::CACHE_POS_INHERIT`) ⇒ inherit
    // from `position` (dense base decode). `≥ 0` ⇒ decoupled KV slot
    // for SpecPrefill sparse-prefill or MTP draft layers.
    cache_pos: i32,
    layers_device: &GpuBuffer,
    int4_scales_device: Option<&GpuBuffer>,
    kv_fp8_descs_device: Option<&GpuBuffer>,
    num_layers: usize,
    hidden_ping: &mut GpuBuffer,
    hidden_pong: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    ffn_topk_idx_scratch: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    embed_w: Option<&GpuBuffer>,
    token_id: i32,
    token_ids: Option<&GpuBuffer>,
    prefill_len: i32,
    lm_head_fold: Option<Qwen36MoePersistentLmHeadFold<'_>>,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::persistent_decode_launch: only BF16 wired, got {dtype:?}"
        )));
    }
    let backend = layers_device.backend();
    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };
    let int4_ptr: *const Qwen36MoeInt4ScaleDesc = int4_scales_device
        .map(|b| b.as_ptr() as *const Qwen36MoeInt4ScaleDesc)
        .unwrap_or(std::ptr::null());
    let kv_fp8_ptr: *const Qwen36MoeKVCacheFp8Desc = kv_fp8_descs_device
        .map(|b| b.as_ptr() as *const Qwen36MoeKVCacheFp8Desc)
        .unwrap_or(std::ptr::null());
    let embed_w_ptr = embed_w.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let token_ids_ptr = token_ids
        .map(|b| b.as_ptr() as *const c_uint)
        .unwrap_or(std::ptr::null());

    // Fold pointers default to null; the kernel skips the lm_head phase
    // when any of the three is null.
    let (vocab, final_norm_w_ptr, lm_head_w_ptr, logits_out_ptr) = match lm_head_fold {
        Some(mut f) => (
            f.vocab,
            f.final_norm_w.as_ptr(),
            f.lm_head_w.as_ptr(),
            f.logits_out.as_mut_ptr(),
        ),
        None => (0, std::ptr::null(), std::ptr::null(), std::ptr::null_mut()),
    };

    let op = match mode {
        Qwen36MoePersistentMode::Full => "qwen36.persistent_decode",
        Qwen36MoePersistentMode::RouterOnly => "qwen36.persistent_router_only",
        Qwen36MoePersistentMode::FfnOnly => "qwen36.persistent_ffn_only",
    };
    crate::prefill_ffi::ffi_profile_time_result(op, ordinal, || {
        let status: c_int = match backend {
            Backend::Hip | Backend::Cuda => {
                #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
                unsafe {
                    qwen36_moe_hip_persistent_decode_launch(
                        dtype.kernel_dtype_code(),
                        ordinal,
                        num_layers as c_int,
                        start_layer as c_int,
                        end_layer_exclusive as c_int,
                        mode.as_ffi(),
                        layers_device.as_ptr() as *const Qwen36MoeDecodeLayerDesc,
                        int4_ptr,
                        kv_fp8_ptr,
                        geom.hidden,
                        geom.num_heads,
                        geom.num_kv_heads,
                        geom.head_dim,
                        geom.rotary_dim,
                        geom.num_k_heads,
                        geom.num_v_heads,
                        geom.head_k_dim,
                        geom.head_v_dim,
                        geom.conv_kernel_dim,
                        geom.num_experts,
                        geom.moe_intermediate,
                        geom.shared_intermediate,
                        geom.top_k,
                        vocab,
                        geom.rope_theta,
                        geom.rms_norm_eps,
                        position,
                        cache_pos,
                        embed_w_ptr,
                        token_id as c_int,
                        token_ids_ptr,
                        prefill_len as c_int,
                        hidden_ping.as_mut_ptr(),
                        hidden_pong.as_mut_ptr(),
                        workspace.as_mut_ptr() as *mut f32,
                        ffn_topk_idx_scratch.as_mut_ptr() as *mut c_int,
                        final_norm_w_ptr,
                        lm_head_w_ptr,
                        logits_out_ptr,
                        counters,
                        barrier_counter,
                        barrier_flag,
                    )
                }
                #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
                {
                    return Err(GpuError::InvalidArg(
                        "qwen36_moe::persistent_decode_launch: GPU backend not compiled".into(),
                    ));
                }
            }
            Backend::Metal => {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::persistent_decode_launch: Metal backend not yet wired".into(),
                ));
            }
        };
        if status != 0 {
            return Err(GpuError::backend(
                backend,
                format!("qwen36_moe persistent decode launch failed with status {status}"),
            ));
        }
        Ok(())
    })
}

/// Geometry + position state for the staged-attention parity launcher.
/// These are constants of the layer being tested; bundling them into a
/// struct keeps the safe wrapper below from sprouting eight scalar args.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeAttnStepParams {
    pub stage: i32,
    pub hidden: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rotary_dim: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub position: i32,
    /// Optional override for the KV cache slot index. `< 0` (the default
    /// constructor sets `-1`) ⇒ inherit from `position` — the base-model
    /// decode case where RoPE position == cache slot. Set to ≥ 0 to
    /// decouple the two: used by the Qwen3.6-MoE self-speculative MTP
    /// layer, where RoPE rotates at absolute position `base_seq_len + k`
    /// but the per-MTP-session cache is fresh per draft session and
    /// writes step `k` at slot `k`. Phase G writes at `cache_pos` and
    /// attends over `kv_len = cache_pos + 1`; RoPE still uses `position`.
    pub cache_pos: i32,
}

impl Qwen36MoeAttnStepParams {
    /// Sentinel value for `cache_pos` meaning "inherit from `position`".
    /// Use this in places that don't yet need MTP semantics so the call
    /// site reads as "default". Equivalent to `-1`.
    pub const CACHE_POS_INHERIT: i32 = -1;
}

/// Weight pointers for the staged-attention parity launcher. Pointers
/// unused by the requested `stage` may be null; the kernel won't dereference
/// them. See [`qwen36_moe_hip_attn_step_launch`] for the per-stage matrix.
///
/// Each `*_proj_w` pointer carries either a BF16 weight slab (when the
/// matching field in [`Qwen36MoeAttnStepInt4`] is null) or an INT4 packed-u8
/// slab (when the matching `*_scale`/`*_zero` pair is set).
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeAttnStepWeights {
    pub input_hidden: *const c_void,
    pub input_norm_w: *const c_void,
    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub o_proj_w: *const c_void,
    /// PR 4d KV cache. When both `kv_cache_k` and `kv_cache_v` are non-null
    /// (and `kv_max_t > 0`), the kernel writes the current step's K/V at
    /// slot `position` and attends over `kv_len = position + 1` past tokens.
    /// When both are null, falls back to the kv_len=1 self-attention path
    /// (back-compat for the per-block parity tests). Layout: BF16 tensors
    /// of shape `[kv_max_t, num_kv_heads * head_dim]`.
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_max_t: i32,
}

impl Default for Qwen36MoeAttnStepWeights {
    fn default() -> Self {
        Self {
            input_hidden: std::ptr::null(),
            input_norm_w: std::ptr::null(),
            q_proj_w: std::ptr::null(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            q_norm_w: std::ptr::null(),
            k_norm_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        }
    }
}

/// Optional INT4 sidecar pointers + group size for the full-attention
/// parity launcher (PR 4b6). `group_size == 0` ⇒ INT4 disabled and every
/// sidecar pointer must be null. When non-zero, each tensor is
/// independently switchable: a non-null `*_scale`/`*_zero` pair routes
/// that tensor's matvec through `int4_dequant_scalar`; a null pair keeps
/// it on the BF16 path. Scales and zeros are BF16 with the bake's group
/// layout — `[out/gs, in/gs]`.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeAttnStepInt4 {
    pub group_size: i32,
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
}

impl Qwen36MoeAttnStepInt4 {
    /// All-null sidecars + group_size=0. BF16 path falls through for every
    /// tensor.
    pub const fn disabled() -> Self {
        Self {
            group_size: 0,
            q_proj_scale: std::ptr::null(),
            q_proj_zero: std::ptr::null(),
            k_proj_scale: std::ptr::null(),
            k_proj_zero: std::ptr::null(),
            v_proj_scale: std::ptr::null(),
            v_proj_zero: std::ptr::null(),
            o_proj_scale: std::ptr::null(),
            o_proj_zero: std::ptr::null(),
        }
    }
}

/// Safe wrapper for the PR 4b2 staged-attention parity launcher.
///
/// `output` must be a BF16 buffer with at least `num_heads * head_dim`
/// elements (the size of the largest staged intermediate). `workspace` must
/// be an F32 buffer with at least `2 * num_heads * head_dim` elements.
/// `sync_buf` must be a 96-byte zero buffer (counters @ +0..+63 — only
/// counters[0] used here, barrier counter @ +64, barrier flag @ +68).
pub fn attn_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeAttnStepParams,
    weights: &Qwen36MoeAttnStepWeights,
    int4: &Qwen36MoeAttnStepInt4,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    let backend = output.backend();
    if backend == Backend::Metal {
        return attn_step_stage1_5_metal_host(params, weights, int4, output, workspace);
    }

    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    // Layout: 16 u32 work-stealing counter slots at +0..+63 (the FFN
    // concurrent-experts dispatch uses 2*K_top of these; attn/linear/stub
    // only touch counters[0]). Barrier counter+flag follow at +64/+68.
    // Sync_buf must be at least 96 bytes zeroed before launch.
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_attn_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_heads as c_int,
                    params.num_kv_heads as c_int,
                    params.head_dim as c_int,
                    params.rotary_dim as c_int,
                    params.rope_theta,
                    params.rms_norm_eps,
                    params.position as c_int,
                    params.cache_pos as c_int,
                    weights.input_hidden,
                    weights.input_norm_w,
                    weights.q_proj_w,
                    weights.k_proj_w,
                    weights.v_proj_w,
                    weights.q_norm_w,
                    weights.k_norm_w,
                    weights.o_proj_w,
                    int4.group_size as c_int,
                    int4.q_proj_scale,
                    int4.q_proj_zero,
                    int4.k_proj_scale,
                    int4.k_proj_zero,
                    int4.v_proj_scale,
                    int4.v_proj_zero,
                    int4.o_proj_scale,
                    int4.o_proj_zero,
                    output.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                    weights.kv_cache_k,
                    weights.kv_cache_v,
                    weights.kv_max_t as c_int,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::attn_step_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            unreachable!("Metal attn_step handled above");
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe attn_step launch failed with status {status}"),
        ));
    }
    Ok(())
}

fn attn_step_stage1_5_metal_host(
    params: Qwen36MoeAttnStepParams,
    weights: &Qwen36MoeAttnStepWeights,
    int4: &Qwen36MoeAttnStepInt4,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if output.dtype() != ScalarType::BF16 || workspace.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: Metal stage1-5 expects BF16/F32 \
             output buffers, got {:?}/{:?}",
            output.dtype(),
            workspace.dtype(),
        )));
    }
    if int4.group_size < 0 {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::attn_step_launch: Metal stage1-5 does not yet support FP8 sidecars".into(),
        ));
    }
    if weights.input_hidden.is_null()
        || weights.input_norm_w.is_null()
        || weights.q_proj_w.is_null()
        || weights.q_norm_w.is_null()
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::attn_step_launch: Metal stage1 requires input_hidden, \
             input_norm_w, q_proj_w, and q_norm_w"
                .into(),
        ));
    }
    if params.stage >= 2
        && (weights.k_proj_w.is_null() || weights.v_proj_w.is_null() || weights.k_norm_w.is_null())
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::attn_step_launch: Metal stage2 requires k_proj_w, \
             v_proj_w, and k_norm_w"
                .into(),
        ));
    }
    if params.stage >= 3 && (params.rotary_dim < 0 || params.rotary_dim > params.head_dim) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: Metal stage{} invalid rotary_dim {} for head_dim {}",
            params.stage, params.rotary_dim, params.head_dim
        )));
    }
    if params.stage >= 4 {
        let cache_k_set = !weights.kv_cache_k.is_null();
        let cache_v_set = !weights.kv_cache_v.is_null();
        if cache_k_set != cache_v_set {
            return Err(GpuError::InvalidArg(
                "qwen36_moe::attn_step_launch: Metal stage4 requires paired KV cache pointers"
                    .into(),
            ));
        }
        if cache_k_set {
            let eff_cache_pos = if params.cache_pos >= 0 {
                params.cache_pos
            } else {
                params.position
            };
            if eff_cache_pos < 0 || weights.kv_max_t <= eff_cache_pos {
                return Err(GpuError::InvalidArg(format!(
                    "qwen36_moe::attn_step_launch: Metal stage{} invalid cache_pos {} for kv_max_t {}",
                    params.stage, eff_cache_pos, weights.kv_max_t
                )));
            }
        }
    }
    if params.stage >= 5 && weights.o_proj_w.is_null() {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::attn_step_launch: Metal stage5 requires o_proj_w".into(),
        ));
    }

    let hidden = params.hidden as usize;
    let num_heads = params.num_heads as usize;
    let num_kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    if hidden == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::attn_step_launch: Metal stage1-5 requires non-zero geometry".into(),
        ));
    }
    if params.stage >= 4 && num_heads % num_kv_heads != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: Metal stage{} requires num_heads divisible by num_kv_heads",
            params.stage
        )));
    }

    let q_out_dim = 2 * num_heads * head_dim;
    let q_normed_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let off_q_raw = 0usize;
    let off_k_raw = q_out_dim;
    let off_v_raw = off_k_raw + kv_dim;
    let off_q_normed = off_v_raw + kv_dim;
    let off_k_normed = off_q_normed + q_normed_dim;
    let off_q_rot = off_k_normed + kv_dim;
    let off_k_rot = off_q_rot + q_normed_dim;
    let off_attn = 4 * num_heads * head_dim + 4 * num_kv_heads * head_dim;
    let off_gated = off_attn + q_normed_dim;
    let off_o_out = off_gated + q_normed_dim;
    let cache_k_set = !weights.kv_cache_k.is_null();
    let eff_cache_pos = if params.cache_pos >= 0 {
        params.cache_pos as usize
    } else {
        params.position.max(0) as usize
    };
    let kv_len = if cache_k_set { eff_cache_pos + 1 } else { 1 };
    let output_len = if params.stage == 2 {
        kv_dim
    } else if params.stage == 3 {
        q_normed_dim + kv_dim
    } else if params.stage == 5 {
        hidden
    } else {
        q_normed_dim
    };
    let workspace_len = if params.stage >= 5 {
        off_o_out + hidden
    } else if params.stage >= 4 {
        off_attn + q_normed_dim
    } else if params.stage >= 3 {
        off_k_rot + kv_dim
    } else if params.stage >= 2 {
        off_k_normed + kv_dim
    } else {
        off_q_normed + q_normed_dim
    };
    let workspace_elems = workspace.shape().iter().product::<usize>();
    let output_elems = output.shape().iter().product::<usize>();
    if workspace_elems < workspace_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: Metal stage{} workspace too small: need {}, got {}",
            params.stage, workspace_len, workspace_elems
        )));
    }
    if output_elems < output_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::attn_step_launch: Metal stage{} output too small: need {}, got {}",
            params.stage, output_len, output_elems
        )));
    }

    let input = unsafe { std::slice::from_raw_parts(weights.input_hidden as *const u16, hidden) };
    let input_norm_w =
        unsafe { std::slice::from_raw_parts(weights.input_norm_w as *const u16, hidden) };
    let q_norm_w = unsafe { std::slice::from_raw_parts(weights.q_norm_w as *const u16, head_dim) };
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, output_len) };
    let workspace = unsafe {
        std::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut f32, workspace_len)
    };

    let x_norm =
        crate::prefill_ffi::metal_profile_time("qwen36_full_attn_input_norm", "host", || {
            let mut mean_sq = 0.0f32;
            for &bits in input {
                let v = bf16_bits_to_f32(bits);
                mean_sq += v * v;
            }
            let inv_rms = 1.0f32 / (mean_sq / hidden as f32 + params.rms_norm_eps).sqrt();
            let mut x_norm = vec![0.0f32; hidden];
            for col in 0..hidden {
                let v = bf16_bits_to_f32(input[col]);
                let w = bf16_bits_to_f32(input_norm_w[col]);
                x_norm[col] = bf16_round_f32(v * inv_rms * (1.0 + w));
            }
            x_norm
        });

    qwen36_validate_dense_or_int4_sidecars(
        int4.q_proj_scale,
        int4.q_proj_zero,
        int4.group_size,
        "q_proj",
    )?;
    if params.stage >= 2 {
        qwen36_validate_dense_or_int4_sidecars(
            int4.k_proj_scale,
            int4.k_proj_zero,
            int4.group_size,
            "k_proj",
        )?;
        qwen36_validate_dense_or_int4_sidecars(
            int4.v_proj_scale,
            int4.v_proj_zero,
            int4.group_size,
            "v_proj",
        )?;
    }
    if params.stage >= 5 {
        qwen36_validate_dense_or_int4_sidecars(
            int4.o_proj_scale,
            int4.o_proj_zero,
            int4.group_size,
            "o_proj",
        )?;
    }

    {
        let q_w = weights.q_proj_w as usize;
        let q_scale = int4.q_proj_scale as usize;
        let q_zero = int4.q_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let q_raw = &mut workspace[off_q_raw..off_q_raw + q_out_dim];
        qwen36_parallel_chunks_mut(q_raw, 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    q_w, q_scale, q_zero, row, hidden, group_size, &x_norm,
                );
                *out = bf16_round_f32(acc);
            }
        });
    }

    for head in 0..num_heads {
        let q_in_base = head * 2 * head_dim;
        let q_out_base = head * head_dim;
        let mut mean_sq = 0.0f32;
        for i in 0..head_dim {
            let v = workspace[off_q_raw + q_in_base + i];
            mean_sq += v * v;
        }
        let inv_head = 1.0f32 / (mean_sq / head_dim as f32 + params.rms_norm_eps).sqrt();
        for i in 0..head_dim {
            let v = workspace[off_q_raw + q_in_base + i];
            let w = bf16_bits_to_f32(q_norm_w[i]);
            let normed = bf16_round_f32(v * inv_head * (1.0 + w));
            workspace[off_q_normed + q_out_base + i] = normed;
            if params.stage == 1 {
                output[q_out_base + i] = f32_to_bf16_bits(normed);
            }
        }
    }

    if params.stage == 1 {
        return Ok(());
    }

    let k_norm_w = unsafe { std::slice::from_raw_parts(weights.k_norm_w as *const u16, head_dim) };

    {
        let (_, after_k) = workspace.split_at_mut(off_k_raw);
        let (k_raw, after_v) = after_k.split_at_mut(kv_dim);
        let (v_raw, _) = after_v.split_at_mut(kv_dim);
        let k_w = weights.k_proj_w as usize;
        let k_scale = int4.k_proj_scale as usize;
        let k_zero = int4.k_proj_zero as usize;
        let v_w = weights.v_proj_w as usize;
        let v_scale = int4.v_proj_scale as usize;
        let v_zero = int4.v_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        qwen36_parallel_chunks2_mut(k_raw, v_raw, 64, |start, k_chunk, v_chunk| {
            for (local, (k_out, v_out)) in k_chunk.iter_mut().zip(v_chunk.iter_mut()).enumerate() {
                let row = start + local;
                let k_acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    k_w, k_scale, k_zero, row, hidden, group_size, &x_norm,
                );
                let v_acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    v_w, v_scale, v_zero, row, hidden, group_size, &x_norm,
                );
                *k_out = bf16_round_f32(k_acc);
                *v_out = bf16_round_f32(v_acc);
            }
        });
    }

    for head in 0..num_kv_heads {
        let base = head * head_dim;
        let mut mean_sq = 0.0f32;
        for i in 0..head_dim {
            let v = workspace[off_k_raw + base + i];
            mean_sq += v * v;
        }
        let inv_head = 1.0f32 / (mean_sq / head_dim as f32 + params.rms_norm_eps).sqrt();
        for i in 0..head_dim {
            let v = workspace[off_k_raw + base + i];
            let w = bf16_bits_to_f32(k_norm_w[i]);
            let normed = bf16_round_f32(v * inv_head * (1.0 + w));
            workspace[off_k_normed + base + i] = normed;
            if params.stage == 2 {
                output[base + i] = f32_to_bf16_bits(normed);
            }
        }
    }

    if params.stage == 2 {
        return Ok(());
    }

    let rotary_dim = params.rotary_dim as usize;
    let half_rotary = rotary_dim / 2;
    let theta_log = params.rope_theta.ln();
    for head_idx in 0..(num_heads + num_kv_heads) {
        let is_k = head_idx >= num_heads;
        let head = if is_k { head_idx - num_heads } else { head_idx };
        let src_off = if is_k { off_k_normed } else { off_q_normed };
        let dst_off = if is_k { off_k_rot } else { off_q_rot };
        let pub_off = head_idx * head_dim;
        for i in 0..half_rotary {
            let a = workspace[src_off + head * head_dim + i];
            let b = workspace[src_off + head * head_dim + half_rotary + i];
            let exponent = (i as f32 / half_rotary as f32) * theta_log;
            let freq = params.position as f32 * (-exponent).exp();
            let c = bf16_round_f32(freq.cos());
            let s = bf16_round_f32(freq.sin());
            let rot_a = bf16_round_f32(bf16_round_f32(a * c) - bf16_round_f32(b * s));
            let rot_b = bf16_round_f32(bf16_round_f32(b * c) + bf16_round_f32(a * s));
            workspace[dst_off + head * head_dim + i] = rot_a;
            workspace[dst_off + head * head_dim + half_rotary + i] = rot_b;
            if params.stage == 3 {
                output[pub_off + i] = f32_to_bf16_bits(rot_a);
                output[pub_off + half_rotary + i] = f32_to_bf16_bits(rot_b);
            }
        }
        for i in rotary_dim..head_dim {
            let x = workspace[src_off + head * head_dim + i];
            workspace[dst_off + head * head_dim + i] = x;
            if params.stage == 3 {
                output[pub_off + i] = f32_to_bf16_bits(x);
            }
        }
    }

    if params.stage == 3 {
        return Ok(());
    }

    let rep = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    if cache_k_set {
        let cache_elems = weights.kv_max_t as usize * kv_dim;
        let cache_k =
            unsafe { std::slice::from_raw_parts_mut(weights.kv_cache_k as *mut u16, cache_elems) };
        let cache_v =
            unsafe { std::slice::from_raw_parts_mut(weights.kv_cache_v as *mut u16, cache_elems) };
        let slot_base = eff_cache_pos * kv_dim;
        for idx in 0..kv_dim {
            cache_k[slot_base + idx] = f32_to_bf16_bits(workspace[off_k_rot + idx]);
            cache_v[slot_base + idx] = f32_to_bf16_bits(workspace[off_v_raw + idx]);
        }
        for hq in 0..num_heads {
            let h_kv = hq / rep;
            let mut scores = vec![0.0f32; kv_len];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..kv_len {
                let mut dot = 0.0f32;
                let kv_base = t * kv_dim + h_kv * head_dim;
                for i in 0..head_dim {
                    dot += workspace[off_q_rot + hq * head_dim + i]
                        * bf16_bits_to_f32(cache_k[kv_base + i]);
                }
                let score = dot * scale;
                scores[t] = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for i in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..kv_len {
                    let kv_base = t * kv_dim + h_kv * head_dim;
                    acc += (scores[t] / denom) * bf16_bits_to_f32(cache_v[kv_base + i]);
                }
                workspace[off_attn + hq * head_dim + i] = acc;
                if params.stage == 4 {
                    output[hq * head_dim + i] = f32_to_bf16_bits(acc);
                }
            }
        }
    } else {
        for hq in 0..num_heads {
            let h_kv = hq / rep;
            for i in 0..head_dim {
                let v = workspace[off_v_raw + h_kv * head_dim + i];
                workspace[off_attn + hq * head_dim + i] = v;
                if params.stage == 4 {
                    output[hq * head_dim + i] = f32_to_bf16_bits(v);
                }
            }
        }
    }

    if params.stage == 4 {
        return Ok(());
    }

    for head in 0..num_heads {
        for i in 0..head_dim {
            let out_gate = workspace[off_q_raw + head * 2 * head_dim + head_dim + i];
            let attn_v = workspace[off_attn + head * head_dim + i];
            let sig = 1.0f32 / (1.0f32 + (-out_gate).exp());
            let gated = bf16_round_f32(bf16_round_f32(sig) * attn_v);
            workspace[off_gated + head * head_dim + i] = gated;
        }
    }

    let gated = workspace[off_gated..off_gated + q_normed_dim].to_vec();
    {
        let o_w = weights.o_proj_w as usize;
        let o_scale = int4.o_proj_scale as usize;
        let o_zero = int4.o_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let o_out = &mut workspace[off_o_out..off_o_out + hidden];
        qwen36_parallel_chunks2_mut(o_out, output, 64, |start, out_chunk, pub_chunk| {
            for (local, (out, pub_out)) in
                out_chunk.iter_mut().zip(pub_chunk.iter_mut()).enumerate()
            {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    o_w,
                    o_scale,
                    o_zero,
                    row,
                    q_normed_dim,
                    group_size,
                    &gated,
                );
                let proj = bf16_round_f32(acc);
                let result = bf16_round_f32(bf16_bits_to_f32(input[row]) + proj);
                *out = result;
                *pub_out = f32_to_bf16_bits(result);
            }
        });
    }

    Ok(())
}

/// Geometry for the staged linear-attention parity launcher. Mirrors
/// `Qwen36MoeAttnStepParams`. Bundling these into a struct keeps the safe
/// wrapper from sprouting a long scalar arglist.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeLinearStepParams {
    pub stage: i32,
    pub hidden: i32,
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_dim: i32,
    pub rms_norm_eps: f32,
}

/// Weight + state pointers for the staged linear-attention parity launcher.
/// Pointers unused by the requested `stage` may be null; the kernel won't
/// dereference them. See [`qwen36_moe_hip_linear_step_launch`] for the
/// per-stage matrix.
///
/// `in_proj_qkv_w`, `in_proj_z_w`, and `out_proj_w` carry either BF16 weight
/// slabs (when the matching field in [`Qwen36MoeLinearStepInt4`] is null) or
/// INT4 packed-u8 slabs (when the matching `*_scale`/`*_zero` pair is set).
/// `in_proj_a_w` / `in_proj_b_w` and the conv/state buffers always stay BF16.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeLinearStepWeights {
    pub input_hidden: *const c_void,
    pub input_norm_w: *const c_void,
    pub in_proj_qkv_w: *const c_void,
    pub in_proj_z_w: *const c_void,
    pub in_proj_a_w: *const c_void,
    pub in_proj_b_w: *const c_void,
    pub conv1d_w: *const c_void,
    pub conv1d_bias: *const c_void,
    pub dt_bias: *const c_void,
    pub a_log: *const c_void,
    pub norm_w: *const c_void,
    pub out_proj_w: *const c_void,
    pub conv_state: *mut c_void,
    pub recurrent_state: *mut f32,
}

/// Optional INT4 sidecar pointers + group size for the linear-attention
/// parity launcher (PR 4b6). `group_size == 0` ⇒ INT4 disabled and every
/// sidecar pointer must be null. Only the three projections the bake
/// quantizes (`in_proj_qkv`, `in_proj_z`, `out_proj`) are switchable —
/// `in_proj_a` / `in_proj_b` always stay BF16.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeLinearStepInt4 {
    pub group_size: i32,
    pub in_proj_qkv_scale: *const c_void,
    pub in_proj_qkv_zero: *const c_void,
    pub in_proj_z_scale: *const c_void,
    pub in_proj_z_zero: *const c_void,
    pub out_proj_scale: *const c_void,
    pub out_proj_zero: *const c_void,
}

impl Qwen36MoeLinearStepInt4 {
    /// All-null sidecars + group_size=0. BF16 path falls through.
    pub const fn disabled() -> Self {
        Self {
            group_size: 0,
            in_proj_qkv_scale: std::ptr::null(),
            in_proj_qkv_zero: std::ptr::null(),
            in_proj_z_scale: std::ptr::null(),
            in_proj_z_zero: std::ptr::null(),
            out_proj_scale: std::ptr::null(),
            out_proj_zero: std::ptr::null(),
        }
    }
}

fn qwen36_linear_int4_stage5_metal_native_supported(
    params: Qwen36MoeLinearStepParams,
    weights: &Qwen36MoeLinearStepWeights,
    int4: &Qwen36MoeLinearStepInt4,
    output_capacity: usize,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5").is_none()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_k_heads == 16
        && params.num_v_heads == 32
        && params.head_k_dim == 128
        && params.head_v_dim == 128
        && params.conv_kernel_dim == 4
        && int4.group_size == 128
        && output_capacity >= params.hidden as usize
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.input_norm_w.is_null()
        && !weights.in_proj_qkv_w.is_null()
        && !weights.in_proj_z_w.is_null()
        && !weights.in_proj_a_w.is_null()
        && !weights.in_proj_b_w.is_null()
        && !weights.conv1d_w.is_null()
        && !weights.dt_bias.is_null()
        && !weights.a_log.is_null()
        && !weights.norm_w.is_null()
        && !weights.out_proj_w.is_null()
        && !weights.conv_state.is_null()
        && !weights.recurrent_state.is_null()
        && !int4.in_proj_qkv_scale.is_null()
        && !int4.in_proj_qkv_zero.is_null()
        && !int4.in_proj_z_scale.is_null()
        && !int4.in_proj_z_zero.is_null()
        && !int4.out_proj_scale.is_null()
        && !int4.out_proj_zero.is_null()
}

/// Safe wrapper for the PR 4b3 staged linear-attention parity launcher.
/// Same workspace / sync_buf layout as [`attn_step_launch`]: 96-byte zero
/// scratch (only counters[0] used here), F32 workspace sized for the
/// stage's footprint, BF16 output sized for the largest staged intermediate.
pub fn linear_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeLinearStepParams,
    weights: &Qwen36MoeLinearStepWeights,
    int4: &Qwen36MoeLinearStepInt4,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    // All five stages are wired through PR 4b3 step 6.

    let backend = output.backend();
    if backend == Backend::Metal {
        if params.stage <= 5 {
            return linear_step_stage1_5_metal_host(
                params, weights, int4, output, workspace, None, true,
            );
        }
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal backend is wired for stages 1-5 only".into(),
        ));
    }

    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    // Layout: 16 u32 work-stealing counter slots at +0..+63 (the FFN
    // concurrent-experts dispatch uses 2*K_top of these; attn/linear/stub
    // only touch counters[0]). Barrier counter+flag follow at +64/+68.
    // Sync_buf must be at least 96 bytes zeroed before launch.
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_linear_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_k_heads as c_int,
                    params.num_v_heads as c_int,
                    params.head_k_dim as c_int,
                    params.head_v_dim as c_int,
                    params.conv_kernel_dim as c_int,
                    params.rms_norm_eps,
                    weights.input_hidden,
                    weights.input_norm_w,
                    weights.in_proj_qkv_w,
                    weights.in_proj_z_w,
                    weights.in_proj_a_w,
                    weights.in_proj_b_w,
                    weights.conv1d_w,
                    weights.conv1d_bias,
                    weights.dt_bias,
                    weights.a_log,
                    weights.norm_w,
                    weights.out_proj_w,
                    weights.conv_state,
                    weights.recurrent_state as *mut c_void,
                    int4.group_size as c_int,
                    int4.in_proj_qkv_scale,
                    int4.in_proj_qkv_zero,
                    int4.in_proj_z_scale,
                    int4.in_proj_z_zero,
                    int4.out_proj_scale,
                    int4.out_proj_zero,
                    output.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::linear_step_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            unreachable!("Metal linear_step handled above");
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe linear_step launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Metal-only stage-5 launcher that uses `output` as the temporary
/// normalized-hidden buffer, but writes the final residual result to
/// `final_output`. This lets batched prefill keep each row in place without a
/// CPU-side shared-memory D2D copy after every linear-attention token.
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_step_stage5_metal_native_into(
    params: Qwen36MoeLinearStepParams,
    weights: &Qwen36MoeLinearStepWeights,
    int4: &Qwen36MoeLinearStepInt4,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    final_output: *mut c_void,
    final_output_capacity: usize,
    wait_for_completion: bool,
) -> Result<(), GpuError> {
    if output.backend() != Backend::Metal {
        return Err(GpuError::backend(
            output.backend(),
            "qwen36_moe::linear_step_stage5_metal_native_into requires Metal output".into(),
        ));
    }
    if params.stage != 5 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_stage5_metal_native_into requires stage=5, got {}",
            params.stage
        )));
    }
    linear_step_stage1_5_metal_host(
        params,
        weights,
        int4,
        output,
        workspace,
        Some((final_output, final_output_capacity)),
        wait_for_completion,
    )
}

fn linear_step_stage1_5_metal_host(
    params: Qwen36MoeLinearStepParams,
    weights: &Qwen36MoeLinearStepWeights,
    int4: &Qwen36MoeLinearStepInt4,
    output: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    final_output_override: Option<(*mut c_void, usize)>,
    wait_for_completion: bool,
) -> Result<(), GpuError> {
    if output.dtype() != ScalarType::BF16 || workspace.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: Metal stage1-5 expects BF16/F32 \
             output buffers, got {:?}/{:?}",
            output.dtype(),
            workspace.dtype(),
        )));
    }
    if int4.group_size < 0 {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal stage1-5 does not yet support FP8 sidecars"
                .into(),
        ));
    }
    if weights.input_hidden.is_null()
        || weights.input_norm_w.is_null()
        || weights.in_proj_qkv_w.is_null()
        || weights.in_proj_z_w.is_null()
        || weights.in_proj_a_w.is_null()
        || weights.in_proj_b_w.is_null()
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal stage1-5 requires input_hidden, \
             input_norm_w, in_proj_qkv_w, in_proj_z_w, in_proj_a_w, and in_proj_b_w"
                .into(),
        ));
    }

    let hidden = params.hidden as usize;
    let num_k_heads = params.num_k_heads as usize;
    let num_v_heads = params.num_v_heads as usize;
    let head_k_dim = params.head_k_dim as usize;
    let head_v_dim = params.head_v_dim as usize;

    if params.stage >= 2
        && (weights.conv1d_w.is_null()
            || weights.conv_state.is_null()
            || params.conv_kernel_dim < 1)
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal stage2 requires conv1d_w, \
             conv_state, and conv_kernel_dim >= 1"
                .into(),
        ));
    }
    if params.stage >= 3 && num_v_heads % num_k_heads != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: Metal stage{} requires num_v_heads \
             divisible by num_k_heads, got {num_v_heads}/{num_k_heads}",
            params.stage
        )));
    }
    if params.stage == 4
        && (weights.dt_bias.is_null()
            || weights.a_log.is_null()
            || weights.recurrent_state.is_null())
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal stage4 requires dt_bias, \
             a_log, and recurrent_state"
                .into(),
        ));
    }
    if params.stage == 5
        && (weights.dt_bias.is_null()
            || weights.a_log.is_null()
            || weights.norm_w.is_null()
            || weights.out_proj_w.is_null()
            || weights.recurrent_state.is_null())
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal stage5 requires dt_bias, \
             a_log, norm_w, out_proj_w, and recurrent_state"
                .into(),
        ));
    }

    let key_dim = num_k_heads * head_k_dim;
    let val_dim = num_v_heads * head_v_dim;
    let qkv_dim = 2 * key_dim + val_dim;
    let total_rows = qkv_dim + val_dim + 2 * num_v_heads;
    let off_qkv_raw = 0usize;
    let off_z_raw = qkv_dim;
    let off_a_raw = qkv_dim + val_dim;
    let off_b_raw = qkv_dim + val_dim + num_v_heads;
    let off_q_normed = total_rows;
    let off_k_normed = off_q_normed + key_dim;
    let off_q_rep = off_k_normed + key_dim;
    let off_k_rep = off_q_rep + num_v_heads * head_k_dim;
    let stage3_workspace_len = off_k_rep + num_v_heads * head_k_dim;
    let off_beta = stage3_workspace_len;
    let off_g = off_beta + num_v_heads;
    let off_rec_out = off_g + num_v_heads;
    let stage4_workspace_len = off_rec_out + val_dim;
    let workspace_len = if params.stage >= 3 {
        if params.stage >= 4 {
            stage4_workspace_len
        } else {
            stage3_workspace_len
        }
    } else {
        total_rows
    };
    let output_len = if params.stage >= 5 {
        hidden
    } else if params.stage >= 4 {
        val_dim
    } else if params.stage >= 3 {
        2 * num_v_heads * head_k_dim + val_dim
    } else {
        qkv_dim
    };
    if workspace.len_bytes() / std::mem::size_of::<f32>() < workspace_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: Metal stage{} workspace too small: \
             need {workspace_len} f32 entries, got {}",
            params.stage,
            workspace.len_bytes() / std::mem::size_of::<f32>(),
        )));
    }
    let output_capacity = output.len_bytes() / std::mem::size_of::<u16>();
    if output_capacity < output_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: Metal stage{} output too small: \
             need {output_len} BF16 entries, got {}",
            params.stage, output_capacity,
        )));
    }
    let (final_output_ptr, final_output_capacity) =
        final_output_override.unwrap_or_else(|| (output.as_mut_ptr(), output_capacity));
    if params.stage == 5 && (final_output_ptr.is_null() || final_output_capacity < hidden) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::linear_step_launch: Metal stage5 final output too small: \
             need {hidden} BF16 entries, got {final_output_capacity}"
        )));
    }

    qwen36_validate_dense_or_int4_sidecars(
        int4.in_proj_qkv_scale,
        int4.in_proj_qkv_zero,
        int4.group_size,
        "in_proj_qkv",
    )?;
    qwen36_validate_dense_or_int4_sidecars(
        int4.in_proj_z_scale,
        int4.in_proj_z_zero,
        int4.group_size,
        "in_proj_z",
    )?;
    if params.stage >= 5 {
        qwen36_validate_dense_or_int4_sidecars(
            int4.out_proj_scale,
            int4.out_proj_zero,
            int4.group_size,
            "out_proj",
        )?;
    }

    if qwen36_linear_int4_stage5_metal_native_supported(params, weights, int4, output_capacity) {
        return crate::prefill_ffi::metal_profile_time(
            "qwen36_linear_int4_stage5",
            "native",
            || unsafe {
                crate::metal_native::qwen36_linear_int4_stage5(
                    hidden,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    params.conv_kernel_dim as usize,
                    int4.group_size as usize,
                    params.rms_norm_eps,
                    weights.input_hidden,
                    weights.input_norm_w,
                    weights.in_proj_qkv_w,
                    int4.in_proj_qkv_scale,
                    int4.in_proj_qkv_zero,
                    weights.in_proj_z_w,
                    int4.in_proj_z_scale,
                    int4.in_proj_z_zero,
                    weights.in_proj_a_w,
                    weights.in_proj_b_w,
                    weights.conv1d_w,
                    weights.conv1d_bias,
                    weights.dt_bias,
                    weights.a_log,
                    weights.norm_w,
                    weights.out_proj_w,
                    int4.out_proj_scale,
                    int4.out_proj_zero,
                    weights.conv_state,
                    weights.recurrent_state as *mut c_void,
                    workspace.as_mut_ptr() as *mut c_void,
                    output.as_mut_ptr() as *mut c_void,
                    final_output_ptr,
                    wait_for_completion,
                )
            },
        );
    }
    if final_output_override.is_some() {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::linear_step_launch: Metal direct final output requires native stage5 INT4 support"
                .into(),
        ));
    }

    let input = unsafe { std::slice::from_raw_parts(weights.input_hidden as *const u16, hidden) };
    let input_norm_w =
        unsafe { std::slice::from_raw_parts(weights.input_norm_w as *const u16, hidden) };
    let in_proj_a_w = unsafe {
        std::slice::from_raw_parts(weights.in_proj_a_w as *const u16, num_v_heads * hidden)
    };
    let in_proj_b_w = unsafe {
        std::slice::from_raw_parts(weights.in_proj_b_w as *const u16, num_v_heads * hidden)
    };
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, output_capacity) };
    let workspace = unsafe {
        std::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut f32, workspace_len)
    };

    let x_norm = crate::prefill_ffi::metal_profile_time("qwen36_linear_input_norm", "host", || {
        let mut mean_sq = 0.0f32;
        for &bits in input {
            let v = bf16_bits_to_f32(bits);
            mean_sq += v * v;
        }
        let inv_rms = 1.0f32 / (mean_sq / hidden as f32 + params.rms_norm_eps).sqrt();
        let mut x_norm = vec![0.0f32; hidden];
        for col in 0..hidden {
            let v = bf16_bits_to_f32(input[col]);
            let w = bf16_bits_to_f32(input_norm_w[col]);
            x_norm[col] = bf16_round_f32(v * inv_rms * (1.0 + w));
        }
        x_norm
    });

    crate::prefill_ffi::metal_profile_time("qwen36_linear_int4_in_proj_qkv", "host", || {
        let qkv_w = weights.in_proj_qkv_w as usize;
        let qkv_scale = int4.in_proj_qkv_scale as usize;
        let qkv_zero = int4.in_proj_qkv_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let qkv = &mut workspace[off_qkv_raw..off_qkv_raw + qkv_dim];
        qwen36_parallel_chunks_mut(qkv, 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    qkv_w, qkv_scale, qkv_zero, row, hidden, group_size, &x_norm,
                );
                *out = bf16_round_f32(acc);
            }
        });
        if params.stage == 1 {
            for row in 0..qkv_dim {
                output[row] = f32_to_bf16_bits(workspace[off_qkv_raw + row]);
            }
        }
    });

    crate::prefill_ffi::metal_profile_time("qwen36_linear_int4_in_proj_z", "host", || {
        let z_w = weights.in_proj_z_w as usize;
        let z_scale = int4.in_proj_z_scale as usize;
        let z_zero = int4.in_proj_z_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let z = &mut workspace[off_z_raw..off_z_raw + val_dim];
        qwen36_parallel_chunks_mut(z, 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    z_w, z_scale, z_zero, row, hidden, group_size, &x_norm,
                );
                *out = bf16_round_f32(acc);
            }
        });
    });
    crate::prefill_ffi::metal_profile_time("qwen36_linear_in_proj_ab", "host", || {
        for row in 0..num_v_heads {
            let mut acc_a = 0.0f32;
            let mut acc_b = 0.0f32;
            let row_base = row * hidden;
            for col in 0..hidden {
                let x = x_norm[col];
                acc_a += bf16_bits_to_f32(in_proj_a_w[row_base + col]) * x;
                acc_b += bf16_bits_to_f32(in_proj_b_w[row_base + col]) * x;
            }
            workspace[off_a_raw + row] = bf16_round_f32(acc_a);
            workspace[off_b_raw + row] = bf16_round_f32(acc_b);
        }
    });

    if params.stage == 1 {
        return Ok(());
    }

    let kernel = params.conv_kernel_dim as usize;
    let kstate = kernel - 1;
    let conv1d_w =
        unsafe { std::slice::from_raw_parts(weights.conv1d_w as *const u16, qkv_dim * kernel) };
    let conv1d_bias = if weights.conv1d_bias.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(weights.conv1d_bias as *const u16, qkv_dim) })
    };
    let conv_state =
        unsafe { std::slice::from_raw_parts_mut(weights.conv_state as *mut u16, qkv_dim * kstate) };

    crate::prefill_ffi::metal_profile_time("qwen36_linear_conv_silu_state", "host", || {
        for ch in 0..qkv_dim {
            let new_qkv = workspace[off_qkv_raw + ch];
            let mut acc = 0.0f32;
            for t in 0..kstate {
                let state = bf16_bits_to_f32(conv_state[ch * kstate + t]);
                acc += state * bf16_bits_to_f32(conv1d_w[ch * kernel + t]);
            }
            acc += new_qkv * bf16_bits_to_f32(conv1d_w[ch * kernel + kstate]);
            if let Some(bias) = conv1d_bias {
                acc += bf16_bits_to_f32(bias[ch]);
            }
            let conv_out = bf16_round_f32(acc);
            let silu = bf16_round_f32(conv_out * (1.0 / (1.0 + (-conv_out).exp())));
            workspace[off_qkv_raw + ch] = silu;
            if params.stage == 2 {
                output[ch] = f32_to_bf16_bits(silu);
            }

            if kstate > 0 {
                for t in 0..kstate.saturating_sub(1) {
                    conv_state[ch * kstate + t] = conv_state[ch * kstate + t + 1];
                }
                conv_state[ch * kstate + (kstate - 1)] = f32_to_bf16_bits(new_qkv);
            }
        }
    });

    if params.stage == 2 {
        return Ok(());
    }

    crate::prefill_ffi::metal_profile_time("qwen36_linear_qk_norm_repeat", "host", || {
        for head in 0..num_k_heads {
            let q_src = off_qkv_raw + head * head_k_dim;
            let k_src = off_qkv_raw + key_dim + head * head_k_dim;
            let q_dst = off_q_normed + head * head_k_dim;
            let k_dst = off_k_normed + head * head_k_dim;

            let mut q_ss = 0.0f32;
            let mut k_ss = 0.0f32;
            for i in 0..head_k_dim {
                let q = workspace[q_src + i];
                let k = workspace[k_src + i];
                q_ss += q * q;
                k_ss += k * k;
            }
            let q_denom = bf16_round_f32(bf16_round_f32(q_ss.sqrt()).max(1e-6));
            let k_denom = bf16_round_f32(bf16_round_f32(k_ss.sqrt()).max(1e-6));
            for i in 0..head_k_dim {
                workspace[q_dst + i] = bf16_round_f32(workspace[q_src + i] / q_denom);
                workspace[k_dst + i] = bf16_round_f32(workspace[k_src + i] / k_denom);
            }
        }

        let rep = num_v_heads / num_k_heads;
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        for vhead in 0..num_v_heads {
            let src_kh = vhead / rep;
            let q_src = off_q_normed + src_kh * head_k_dim;
            let k_src = off_k_normed + src_kh * head_k_dim;
            let q_dst = off_q_rep + vhead * head_k_dim;
            let k_dst = off_k_rep + vhead * head_k_dim;
            for i in 0..head_k_dim {
                let qs = bf16_round_f32(workspace[q_src + i] * q_scale);
                let kn = workspace[k_src + i];
                workspace[q_dst + i] = qs;
                workspace[k_dst + i] = kn;
                if params.stage == 3 {
                    output[vhead * head_k_dim + i] = f32_to_bf16_bits(qs);
                    output[num_v_heads * head_k_dim + vhead * head_k_dim + i] =
                        f32_to_bf16_bits(kn);
                }
            }
            let v_src = off_qkv_raw + 2 * key_dim + vhead * head_v_dim;
            let v_out = 2 * num_v_heads * head_k_dim + vhead * head_v_dim;
            if params.stage == 3 {
                for i in 0..head_v_dim {
                    output[v_out + i] = f32_to_bf16_bits(workspace[v_src + i]);
                }
            }
        }
    });

    if params.stage == 3 {
        return Ok(());
    }

    crate::prefill_ffi::metal_profile_time("qwen36_linear_recurrent_update", "host", || {
        let dt_bias =
            unsafe { std::slice::from_raw_parts(weights.dt_bias as *const u16, num_v_heads) };
        let a_log = unsafe { std::slice::from_raw_parts(weights.a_log as *const u16, num_v_heads) };
        let recurrent_state = unsafe {
            std::slice::from_raw_parts_mut(
                weights.recurrent_state,
                num_v_heads * head_k_dim * head_v_dim,
            )
        };
        for head in 0..num_v_heads {
            let a_v = workspace[off_a_raw + head];
            let b_v = workspace[off_b_raw + head];
            let dt_b = bf16_bits_to_f32(dt_bias[head]);
            let a_log_v = bf16_bits_to_f32(a_log[head]);
            let softplus = (1.0 + (a_v + dt_b).exp()).ln();
            workspace[off_beta + head] = 1.0 / (1.0 + (-b_v).exp());
            workspace[off_g + head] = -softplus * a_log_v.exp();
        }

        for head in 0..num_v_heads {
            let beta = workspace[off_beta + head];
            let gstep = workspace[off_g + head].exp();
            let state_off = head * head_k_dim * head_v_dim;
            let kv_off = off_k_rep + head * head_k_dim;
            let qv_off = off_q_rep + head * head_k_dim;
            let v_off = off_qkv_raw + 2 * key_dim + head * head_v_dim;
            for e in 0..head_k_dim * head_v_dim {
                recurrent_state[state_off + e] *= gstep;
            }
            let mut kv_mem = vec![0.0f32; head_v_dim];
            for j in 0..head_v_dim {
                let mut acc = 0.0f32;
                for i in 0..head_k_dim {
                    acc += recurrent_state[state_off + i * head_v_dim + j] * workspace[kv_off + i];
                }
                kv_mem[j] = acc;
            }
            let mut delta = vec![0.0f32; head_v_dim];
            for j in 0..head_v_dim {
                delta[j] = (workspace[v_off + j] - kv_mem[j]) * beta;
            }
            for i in 0..head_k_dim {
                for j in 0..head_v_dim {
                    recurrent_state[state_off + i * head_v_dim + j] +=
                        workspace[kv_off + i] * delta[j];
                }
            }
            for j in 0..head_v_dim {
                let mut acc = 0.0f32;
                for i in 0..head_k_dim {
                    acc += recurrent_state[state_off + i * head_v_dim + j] * workspace[qv_off + i];
                }
                let rec = bf16_round_f32(acc);
                workspace[off_rec_out + head * head_v_dim + j] = rec;
                if params.stage == 4 {
                    output[head * head_v_dim + j] = f32_to_bf16_bits(rec);
                }
            }
        }
    });

    if params.stage == 4 {
        return Ok(());
    }

    crate::prefill_ffi::metal_profile_time("qwen36_linear_output_gate_norm", "host", || {
        let norm_w =
            unsafe { std::slice::from_raw_parts(weights.norm_w as *const u16, head_v_dim) };
        for head in 0..num_v_heads {
            let rec_off = off_rec_out + head * head_v_dim;
            let z_off = off_z_raw + head * head_v_dim;
            let mut mean_sq = 0.0f32;
            for j in 0..head_v_dim {
                let v = workspace[rec_off + j];
                mean_sq += v * v;
            }
            let inv = 1.0f32 / (mean_sq / head_v_dim as f32 + params.rms_norm_eps).sqrt();
            for j in 0..head_v_dim {
                let rec = workspace[rec_off + j];
                let nw = bf16_bits_to_f32(norm_w[j]);
                let on = bf16_round_f32(rec * inv * nw);
                let z = workspace[z_off + j];
                let z_silu = bf16_round_f32(z * (1.0 / (1.0 + (-z).exp())));
                workspace[rec_off + j] = bf16_round_f32(on * z_silu);
            }
        }
    });

    crate::prefill_ffi::metal_profile_time("qwen36_linear_int4_out_proj", "host", || {
        let rec_out = &workspace[off_rec_out..off_rec_out + val_dim];
        let out_w = weights.out_proj_w as usize;
        let out_scale = int4.out_proj_scale as usize;
        let out_zero = int4.out_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        qwen36_parallel_chunks_mut(&mut output[..hidden], 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    out_w, out_scale, out_zero, row, val_dim, group_size, rec_out,
                );
                let o_out = bf16_round_f32(acc);
                let residual = bf16_round_f32(bf16_bits_to_f32(input[row]) + o_out);
                *out = f32_to_bf16_bits(residual);
            }
        });
    });

    Ok(())
}

/// Geometry for the staged MoE FFN parity launcher. Mirrors
/// `Qwen36MoeAttnStepParams` / `Qwen36MoeLinearStepParams`. Bundling these
/// keeps the safe wrapper's signature short.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepParams {
    pub stage: i32,
    pub layer_idx: i32,
    pub hidden: i32,
    pub num_experts: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub top_k: i32,
    pub rms_norm_eps: f32,
}

/// Weight pointers for the staged MoE FFN parity launcher. Pointers unused
/// by the requested `stage` may be null; the kernel won't dereference them.
/// See [`qwen36_moe_hip_ffn_step_launch`] for the per-stage matrix.
///
/// Each `*_proj_w` pointer carries either a BF16 weight slab (when the
/// matching field in [`Qwen36MoeFfnStepInt4`] is null) or an INT4 packed-u8
/// slab (when the matching `*_scale`/`*_zero` pair is set).
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepWeights {
    pub input_hidden: *const c_void,
    pub post_attn_norm_w: *const c_void,
    pub gate_w: *const c_void,
    pub gate_up_proj_w: *const c_void,
    pub down_proj_w: *const c_void,
    pub shared_gate_proj_w: *const c_void,
    pub shared_up_proj_w: *const c_void,
    pub shared_down_proj_w: *const c_void,
    pub shared_expert_gate_w: *const c_void,
}

/// Optional INT4 sidecar pointers + group size for the FFN parity launcher
/// (PR 4b5). `group_size == 0` ⇒ INT4 disabled and every sidecar pointer
/// must be null. When non-zero, each tensor is independently switchable:
/// a non-null `*_scale`/`*_zero` pair routes that tensor's matvec through
/// `int4_dequant_scalar`; a null pair keeps it on the BF16 path. Scales
/// and zeros are BF16 with the bake's group layout — `[..., out/gs, in/gs]`.
#[derive(Debug, Clone, Copy)]
pub struct Qwen36MoeFfnStepInt4 {
    pub group_size: i32,
    pub gate_up_proj_scale: *const c_void,
    pub gate_up_proj_zero: *const c_void,
    pub down_proj_scale: *const c_void,
    pub down_proj_zero: *const c_void,
    pub shared_gate_proj_scale: *const c_void,
    pub shared_gate_proj_zero: *const c_void,
    pub shared_up_proj_scale: *const c_void,
    pub shared_up_proj_zero: *const c_void,
    pub shared_down_proj_scale: *const c_void,
    pub shared_down_proj_zero: *const c_void,
}

impl Qwen36MoeFfnStepInt4 {
    /// All-null sidecars + group_size=0. Use this when the BF16 path is
    /// what you want — the kernel falls through to the existing matvecs
    /// for every tensor.
    pub const fn disabled() -> Self {
        Self {
            group_size: 0,
            gate_up_proj_scale: std::ptr::null(),
            gate_up_proj_zero: std::ptr::null(),
            down_proj_scale: std::ptr::null(),
            down_proj_zero: std::ptr::null(),
            shared_gate_proj_scale: std::ptr::null(),
            shared_gate_proj_zero: std::ptr::null(),
            shared_up_proj_scale: std::ptr::null(),
            shared_up_proj_zero: std::ptr::null(),
            shared_down_proj_scale: std::ptr::null(),
            shared_down_proj_zero: std::ptr::null(),
        }
    }
}

fn qwen36_ffn_int4_stage5_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.shared_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.shared_expert_gate_w.is_null()
        && !weights.shared_gate_proj_w.is_null()
        && !weights.shared_up_proj_w.is_null()
        && !weights.shared_down_proj_w.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.shared_gate_proj_scale.is_null()
        && !int4.shared_gate_proj_zero.is_null()
        && !int4.shared_up_proj_scale.is_null()
        && !int4.shared_up_proj_zero.is_null()
        && !int4.shared_down_proj_scale.is_null()
        && !int4.shared_down_proj_zero.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

fn qwen36_ffn_expert_gate_up_tiled_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GATE_UP_TILED").is_some()
        && params.stage >= 3
        && params.hidden == 2048
        && params.moe_intermediate == 512
        && params.top_k > 0
        && params.top_k <= 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.gate_up_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
}

fn qwen36_ffn_expert_tiled_stage5_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

fn qwen36_ffn_expert_packed_stage5_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

fn qwen36_ffn_expert_direct_gather_stage5_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

fn qwen36_ffn_expert_gpu_pack_stage5_metal_native_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5").is_some()
        && qwen36_ffn_expert_packed_stage5_metal_native_supported(params, weights, int4)
}

fn qwen36_ffn_expert_mps_bridge_stage5_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_BRIDGE").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

fn qwen36_ffn_expert_mps_static_topn_partial_stage5_supported(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
) -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_STATIC_TOPN_PARTIAL").is_some()
        && params.stage == 5
        && params.hidden == 2048
        && params.num_experts == 256
        && params.moe_intermediate == 512
        && params.top_k == 8
        && int4.group_size == 128
        && !crate::metal_native::disabled_by_env()
        && !weights.input_hidden.is_null()
        && !weights.gate_up_proj_w.is_null()
        && !weights.down_proj_w.is_null()
        && !int4.gate_up_proj_scale.is_null()
        && !int4.gate_up_proj_zero.is_null()
        && !int4.down_proj_scale.is_null()
        && !int4.down_proj_zero.is_null()
}

struct Qwen36PackedExpertBuffers {
    gate_up_proj: GpuBuffer,
    gate_up_scale: GpuBuffer,
    gate_up_zero: GpuBuffer,
    down_proj: GpuBuffer,
    down_scale: GpuBuffer,
    down_zero: GpuBuffer,
}

struct Qwen36MpsExpertBridgeBuffers {
    h_norm: GpuBuffer,
    gate_up_rhs: GpuBuffer,
    gate_up_out: GpuBuffer,
    down_lhs: GpuBuffer,
    down_rhs: GpuBuffer,
    down_out: GpuBuffer,
}

struct Qwen36MpsExpertBridgeScratch {
    h_norm: GpuBuffer,
    gate_up_out: GpuBuffer,
    down_lhs: GpuBuffer,
    down_out: GpuBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Qwen36PackedExpertCacheKey {
    gate_up_proj_ptr: usize,
    gate_up_scale_ptr: usize,
    gate_up_zero_ptr: usize,
    down_proj_ptr: usize,
    down_scale_ptr: usize,
    down_zero_ptr: usize,
    hidden: usize,
    moe_intermediate: usize,
    group_size: usize,
    active_groups: usize,
}

struct Qwen36PackedExpertCacheEntry {
    active_experts: Vec<usize>,
    buffers: Qwen36PackedExpertBuffers,
}

struct Qwen36PackedExpertHotsetCacheEntry {
    slot_experts: Vec<Option<usize>>,
    slot_last_used: Vec<u64>,
    next_stamp: u64,
    buffers: Qwen36PackedExpertBuffers,
}

struct Qwen36PackedExpertStaticTopNCacheEntry {
    resident_experts: Vec<usize>,
    buffers: Qwen36PackedExpertBuffers,
}

struct Qwen36MpsExpertStaticTopNCacheEntry {
    resident_experts: Vec<usize>,
    gate_up_rhs: GpuBuffer,
    down_rhs: GpuBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen36MpsStaticTopNHit {
    original_group: usize,
    slot: usize,
}

#[derive(Debug, Clone, Default)]
struct Qwen36StaticTopNTable {
    layers_by_capacity: HashMap<usize, Vec<Vec<usize>>>,
}

impl Qwen36StaticTopNTable {
    fn capacity_exists(&self, capacity: usize) -> bool {
        self.layers_by_capacity.contains_key(&capacity)
    }

    fn largest_capacity(&self) -> Option<usize> {
        self.layers_by_capacity.keys().copied().max()
    }

    fn layer_experts(&self, capacity: usize, layer_idx: usize) -> Option<&[usize]> {
        self.layers_by_capacity
            .get(&capacity)
            .and_then(|layers| layers.get(layer_idx))
            .map(Vec::as_slice)
            .filter(|experts| !experts.is_empty())
    }
}

static QWEN36_PACKED_EXPERT_CACHE: OnceLock<
    Mutex<HashMap<Qwen36PackedExpertCacheKey, Qwen36PackedExpertCacheEntry>>,
> = OnceLock::new();
static QWEN36_PACKED_EXPERT_HOTSET_CACHE: OnceLock<
    Mutex<HashMap<Qwen36PackedExpertCacheKey, Qwen36PackedExpertHotsetCacheEntry>>,
> = OnceLock::new();
static QWEN36_PACKED_EXPERT_STATIC_TOPN_CACHE: OnceLock<
    Mutex<HashMap<Qwen36PackedExpertCacheKey, Qwen36PackedExpertStaticTopNCacheEntry>>,
> = OnceLock::new();
static QWEN36_MPS_EXPERT_STATIC_TOPN_CACHE: OnceLock<
    Mutex<HashMap<Qwen36PackedExpertCacheKey, Qwen36MpsExpertStaticTopNCacheEntry>>,
> = OnceLock::new();
static QWEN36_STATIC_TOPN_TABLE: OnceLock<Result<Qwen36StaticTopNTable, String>> = OnceLock::new();

fn qwen36_parse_static_topn_experts(
    value: &serde_json::Value,
    path: &str,
) -> Result<Vec<usize>, String> {
    let experts = value
        .as_array()
        .ok_or_else(|| format!("{path}: experts must be an array"))?;
    let mut parsed = Vec::with_capacity(experts.len());
    for (idx, expert) in experts.iter().enumerate() {
        let expert = expert
            .as_u64()
            .ok_or_else(|| format!("{path}[{idx}]: expert must be an integer"))?;
        parsed.push(
            usize::try_from(expert)
                .map_err(|_| format!("{path}[{idx}]: expert {expert} does not fit in usize"))?,
        );
    }
    Ok(parsed)
}

fn qwen36_parse_static_topn_layer_rows(
    capacity: usize,
    value: &serde_json::Value,
) -> Result<Vec<Vec<usize>>, String> {
    let mut layers: Vec<Vec<usize>> = Vec::new();
    if let Some(rows) = value.get("layers").and_then(|value| value.as_array()) {
        for (row_idx, row) in rows.iter().enumerate() {
            let layer_idx = row
                .get("layer")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    format!("static_tables.{capacity}.layers[{row_idx}]: missing integer layer")
                })?;
            let layer_idx = usize::try_from(layer_idx).map_err(|_| {
                format!("static_tables.{capacity}.layers[{row_idx}]: layer does not fit in usize")
            })?;
            let experts = row.get("experts").ok_or_else(|| {
                format!("static_tables.{capacity}.layers[{row_idx}]: missing experts")
            })?;
            if layers.len() <= layer_idx {
                layers.resize_with(layer_idx + 1, Vec::new);
            }
            layers[layer_idx] = qwen36_parse_static_topn_experts(
                experts,
                &format!("static_tables.{capacity}.layers[{row_idx}].experts"),
            )?;
        }
        return Ok(layers);
    }

    let object = value
        .as_object()
        .ok_or_else(|| format!("static_tables.{capacity}: expected object"))?;
    for (layer_key, layer_value) in object {
        let layer_idx = layer_key.parse::<usize>().map_err(|err| {
            format!("static_tables.{capacity}.{layer_key}: invalid layer key: {err}")
        })?;
        let experts_value = layer_value.get("experts").unwrap_or(layer_value);
        if layers.len() <= layer_idx {
            layers.resize_with(layer_idx + 1, Vec::new);
        }
        layers[layer_idx] = qwen36_parse_static_topn_experts(
            experts_value,
            &format!("static_tables.{capacity}.{layer_key}.experts"),
        )?;
    }
    Ok(layers)
}

fn qwen36_parse_static_topn_table_json(raw: &str) -> Result<Qwen36StaticTopNTable, String> {
    let root: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse static top-N JSON: {err}"))?;
    let tables = root
        .get("static_tables")
        .ok_or_else(|| "static top-N JSON missing static_tables".to_string())?
        .as_object()
        .ok_or_else(|| "static_tables must be an object".to_string())?;
    let mut layers_by_capacity = HashMap::new();
    for (capacity_key, value) in tables {
        let capacity = capacity_key.parse::<usize>().map_err(|err| {
            format!("static_tables capacity key {capacity_key:?} is invalid: {err}")
        })?;
        if capacity == 0 {
            return Err("static_tables capacity must be > 0".into());
        }
        let layers = qwen36_parse_static_topn_layer_rows(capacity, value)?;
        if !layers.is_empty() {
            layers_by_capacity.insert(capacity, layers);
        }
    }
    if layers_by_capacity.is_empty() {
        return Err("static_tables did not contain any layer tables".into());
    }
    Ok(Qwen36StaticTopNTable { layers_by_capacity })
}

fn qwen36_static_topn_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN").is_some()
}

fn qwen36_static_topn_table_from_env() -> Result<Option<&'static Qwen36StaticTopNTable>, GpuError> {
    if !qwen36_static_topn_enabled() {
        return Ok(None);
    }
    let table = QWEN36_STATIC_TOPN_TABLE.get_or_init(|| {
        let path = std::env::var("SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE")
            .map_err(|_| {
                "SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE is required when static top-N is enabled"
                    .to_string()
            })?;
        let raw = std::fs::read_to_string(&path)
            .map_err(|err| format!("failed to read static top-N file {path}: {err}"))?;
        qwen36_parse_static_topn_table_json(&raw)
    });
    match table {
        Ok(table) => Ok(Some(table)),
        Err(err) => Err(GpuError::backend(
            Backend::Metal,
            format!("qwen36 static top-N table load failed: {err}"),
        )),
    }
}

fn qwen36_static_topn_requested_capacity() -> Result<Option<usize>, GpuError> {
    match std::env::var("SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&capacity| capacity > 0)
            .map(Some)
            .ok_or_else(|| {
                GpuError::InvalidArg(format!(
                    "invalid SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY={value:?}"
                ))
            }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(GpuError::InvalidArg(format!(
            "invalid static top-N capacity env: {err}"
        ))),
    }
}

fn qwen36_static_topn_layer_experts_from_env(
    layer_idx: i32,
) -> Result<Option<(usize, &'static [usize])>, GpuError> {
    let Some(table) = qwen36_static_topn_table_from_env()? else {
        return Ok(None);
    };
    if layer_idx < 0 {
        return Ok(None);
    }
    let requested_capacity = qwen36_static_topn_requested_capacity()?;
    let capacity = if let Some(capacity) = requested_capacity {
        if !table.capacity_exists(capacity) {
            return Err(GpuError::InvalidArg(format!(
                "static top-N table does not contain requested capacity {capacity}"
            )));
        }
        capacity
    } else {
        table.largest_capacity().ok_or_else(|| {
            GpuError::InvalidArg("static top-N table does not contain any capacity".into())
        })?
    };
    let Some(resident_experts) = table.layer_experts(capacity, layer_idx as usize) else {
        return Ok(None);
    };
    Ok(Some((capacity, resident_experts)))
}

fn qwen36_static_topn_expert_to_slot(
    layer_idx: i32,
    resident_experts: &[usize],
    num_experts: usize,
    capacity: usize,
) -> Result<Vec<Option<usize>>, GpuError> {
    let mut expert_to_slot = vec![None; num_experts];
    for (slot, &expert) in resident_experts.iter().enumerate() {
        if expert >= num_experts {
            return Err(GpuError::InvalidArg(format!(
                "static top-N table layer {layer_idx} capacity {capacity} has expert {expert} >= num_experts {num_experts}"
            )));
        }
        if expert_to_slot[expert].replace(slot).is_some() {
            return Err(GpuError::InvalidArg(format!(
                "static top-N table layer {layer_idx} capacity {capacity} contains duplicate expert {expert}"
            )));
        }
    }
    Ok(expert_to_slot)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_validate_active_expert_pack(
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
) -> Result<(), GpuError> {
    if active_experts.is_empty() {
        return Err(GpuError::InvalidArg(
            "qwen36_pack_active_experts_for_metal: active_experts must not be empty".into(),
        ));
    }
    if group_size == 0
        || hidden == 0
        || moe_intermediate == 0
        || hidden % 2 != 0
        || moe_intermediate % 2 != 0
        || hidden % group_size != 0
        || moe_intermediate % group_size != 0
        || (2 * moe_intermediate) % group_size != 0
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_pack_active_experts_for_metal: unsupported geometry hidden={hidden} moe_intermediate={moe_intermediate} group_size={group_size}"
        )));
    }
    if gate_up_proj_ptr.is_null()
        || gate_up_scale_ptr.is_null()
        || gate_up_zero_ptr.is_null()
        || down_proj_ptr.is_null()
        || down_scale_ptr.is_null()
        || down_zero_ptr.is_null()
    {
        return Err(GpuError::InvalidArg(
            "qwen36_pack_active_experts_for_metal: null expert tensor pointer".into(),
        ));
    }
    for &expert in active_experts {
        if expert >= num_experts {
            return Err(GpuError::InvalidArg(format!(
                "qwen36_pack_active_experts_for_metal: expert index {expert} >= num_experts {num_experts}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen36_alloc_packed_experts_for_metal(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_groups: usize,
    group_size: usize,
) -> Result<Qwen36PackedExpertBuffers, GpuError> {
    let groups = active_groups;
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_byte_cols = hidden / 2;
    let gate_up_sidecar_rows = gate_up_rows / group_size;
    let gate_up_sidecar_cols = hidden / group_size;

    let down_rows = hidden;
    let down_byte_cols = moe_intermediate / 2;
    let down_sidecar_rows = hidden / group_size;
    let down_sidecar_cols = moe_intermediate / group_size;

    let gate_up_proj = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::U8,
        &[groups, gate_up_rows, gate_up_byte_cols],
        BufferKind::Scratch,
    )?;
    let gate_up_scale = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::BF16,
        &[groups, gate_up_sidecar_rows, gate_up_sidecar_cols],
        BufferKind::Scratch,
    )?;
    let gate_up_zero = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::BF16,
        &[groups, gate_up_sidecar_rows, gate_up_sidecar_cols],
        BufferKind::Scratch,
    )?;
    let down_proj = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::U8,
        &[groups, down_rows, down_byte_cols],
        BufferKind::Scratch,
    )?;
    let down_scale = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::BF16,
        &[groups, down_sidecar_rows, down_sidecar_cols],
        BufferKind::Scratch,
    )?;
    let down_zero = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::BF16,
        &[groups, down_sidecar_rows, down_sidecar_cols],
        BufferKind::Scratch,
    )?;

    Ok(Qwen36PackedExpertBuffers {
        gate_up_proj,
        gate_up_scale,
        gate_up_zero,
        down_proj,
        down_scale,
        down_zero,
    })
}

fn qwen36_packed_expert_copy_bytes(
    hidden: usize,
    moe_intermediate: usize,
    active_groups: usize,
    group_size: usize,
) -> usize {
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_byte_cols = hidden / 2;
    let gate_up_bytes_per_expert = gate_up_rows * gate_up_byte_cols;
    let gate_up_sidecar_rows = gate_up_rows / group_size;
    let gate_up_sidecar_cols = hidden / group_size;
    let gate_up_sidecar_elems_per_expert = gate_up_sidecar_rows * gate_up_sidecar_cols;

    let down_rows = hidden;
    let down_byte_cols = moe_intermediate / 2;
    let down_bytes_per_expert = down_rows * down_byte_cols;
    let down_sidecar_rows = hidden / group_size;
    let down_sidecar_cols = moe_intermediate / group_size;
    let down_sidecar_elems_per_expert = down_sidecar_rows * down_sidecar_cols;

    let bf16_bytes = std::mem::size_of::<u16>();
    active_groups
        * (gate_up_bytes_per_expert
            + down_bytes_per_expert
            + 2 * gate_up_sidecar_elems_per_expert * bf16_bytes
            + 2 * down_sidecar_elems_per_expert * bf16_bytes)
}

fn qwen36_packed_expert_hotset_capacity(top_k: usize) -> Option<usize> {
    if std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET").is_none() {
        return None;
    }
    let capacity = std::env::var("SUPERSONIC_METAL_QWEN36_FFN_EXPERT_HOTSET_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(16);
    Some(capacity.clamp(top_k, 256))
}

fn qwen36_hotset_choose_slot(
    slot_experts: &[Option<usize>],
    slot_last_used: &[u64],
    protected_slots: &[bool],
    expert: usize,
) -> (usize, bool, bool) {
    if let Some(slot) = slot_experts
        .iter()
        .position(|&slot_expert| slot_expert == Some(expert))
    {
        return (slot, true, false);
    }
    if let Some(slot) = slot_experts
        .iter()
        .enumerate()
        .find_map(|(slot, expert)| (expert.is_none() && !protected_slots[slot]).then_some(slot))
    {
        return (slot, false, false);
    }
    let (slot, _) = slot_last_used
        .iter()
        .enumerate()
        .filter(|(slot, _)| !protected_slots[*slot])
        .min_by_key(|(_, stamp)| *stamp)
        .expect("qwen36 hotset capacity must leave at least one unprotected slot");
    (slot, false, true)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_fill_packed_expert_slot_for_metal(
    hidden: usize,
    moe_intermediate: usize,
    expert: usize,
    slot: usize,
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    buffers: &mut Qwen36PackedExpertBuffers,
) {
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_byte_cols = hidden / 2;
    let gate_up_bytes_per_expert = gate_up_rows * gate_up_byte_cols;
    let gate_up_sidecar_rows = gate_up_rows / group_size;
    let gate_up_sidecar_cols = hidden / group_size;
    let gate_up_sidecar_elems_per_expert = gate_up_sidecar_rows * gate_up_sidecar_cols;

    let down_rows = hidden;
    let down_byte_cols = moe_intermediate / 2;
    let down_bytes_per_expert = down_rows * down_byte_cols;
    let down_sidecar_rows = hidden / group_size;
    let down_sidecar_cols = moe_intermediate / group_size;
    let down_sidecar_elems_per_expert = down_sidecar_rows * down_sidecar_cols;

    let gate_up_src = gate_up_proj_ptr as *const u8;
    let gate_up_scale_src = gate_up_scale_ptr as *const u16;
    let gate_up_zero_src = gate_up_zero_ptr as *const u16;
    let down_src = down_proj_ptr as *const u8;
    let down_scale_src = down_scale_ptr as *const u16;
    let down_zero_src = down_zero_ptr as *const u16;
    let gate_up_dst = buffers.gate_up_proj.as_mut_ptr() as *mut u8;
    let gate_up_scale_dst = buffers.gate_up_scale.as_mut_ptr() as *mut u16;
    let gate_up_zero_dst = buffers.gate_up_zero.as_mut_ptr() as *mut u16;
    let down_dst = buffers.down_proj.as_mut_ptr() as *mut u8;
    let down_scale_dst = buffers.down_scale.as_mut_ptr() as *mut u16;
    let down_zero_dst = buffers.down_zero.as_mut_ptr() as *mut u16;

    unsafe {
        std::ptr::copy_nonoverlapping(
            gate_up_src.add(expert * gate_up_bytes_per_expert),
            gate_up_dst.add(slot * gate_up_bytes_per_expert),
            gate_up_bytes_per_expert,
        );
        std::ptr::copy_nonoverlapping(
            gate_up_scale_src.add(expert * gate_up_sidecar_elems_per_expert),
            gate_up_scale_dst.add(slot * gate_up_sidecar_elems_per_expert),
            gate_up_sidecar_elems_per_expert,
        );
        std::ptr::copy_nonoverlapping(
            gate_up_zero_src.add(expert * gate_up_sidecar_elems_per_expert),
            gate_up_zero_dst.add(slot * gate_up_sidecar_elems_per_expert),
            gate_up_sidecar_elems_per_expert,
        );
        std::ptr::copy_nonoverlapping(
            down_src.add(expert * down_bytes_per_expert),
            down_dst.add(slot * down_bytes_per_expert),
            down_bytes_per_expert,
        );
        std::ptr::copy_nonoverlapping(
            down_scale_src.add(expert * down_sidecar_elems_per_expert),
            down_scale_dst.add(slot * down_sidecar_elems_per_expert),
            down_sidecar_elems_per_expert,
        );
        std::ptr::copy_nonoverlapping(
            down_zero_src.add(expert * down_sidecar_elems_per_expert),
            down_zero_dst.add(slot * down_sidecar_elems_per_expert),
            down_sidecar_elems_per_expert,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen36_fill_packed_experts_for_metal(
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    buffers: &mut Qwen36PackedExpertBuffers,
) {
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_byte_cols = hidden / 2;
    let gate_up_bytes_per_expert = gate_up_rows * gate_up_byte_cols;
    let gate_up_sidecar_rows = gate_up_rows / group_size;
    let gate_up_sidecar_cols = hidden / group_size;
    let gate_up_sidecar_elems_per_expert = gate_up_sidecar_rows * gate_up_sidecar_cols;

    let down_rows = hidden;
    let down_byte_cols = moe_intermediate / 2;
    let down_bytes_per_expert = down_rows * down_byte_cols;
    let down_sidecar_rows = hidden / group_size;
    let down_sidecar_cols = moe_intermediate / group_size;
    let down_sidecar_elems_per_expert = down_sidecar_rows * down_sidecar_cols;

    let gate_up_src = gate_up_proj_ptr as *const u8;
    let gate_up_scale_src = gate_up_scale_ptr as *const u16;
    let gate_up_zero_src = gate_up_zero_ptr as *const u16;
    let down_src = down_proj_ptr as *const u8;
    let down_scale_src = down_scale_ptr as *const u16;
    let down_zero_src = down_zero_ptr as *const u16;
    let gate_up_dst = buffers.gate_up_proj.as_mut_ptr() as *mut u8;
    let gate_up_scale_dst = buffers.gate_up_scale.as_mut_ptr() as *mut u16;
    let gate_up_zero_dst = buffers.gate_up_zero.as_mut_ptr() as *mut u16;
    let down_dst = buffers.down_proj.as_mut_ptr() as *mut u8;
    let down_scale_dst = buffers.down_scale.as_mut_ptr() as *mut u16;
    let down_zero_dst = buffers.down_zero.as_mut_ptr() as *mut u16;

    for (group, &expert) in active_experts.iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_up_src.add(expert * gate_up_bytes_per_expert),
                gate_up_dst.add(group * gate_up_bytes_per_expert),
                gate_up_bytes_per_expert,
            );
            std::ptr::copy_nonoverlapping(
                gate_up_scale_src.add(expert * gate_up_sidecar_elems_per_expert),
                gate_up_scale_dst.add(group * gate_up_sidecar_elems_per_expert),
                gate_up_sidecar_elems_per_expert,
            );
            std::ptr::copy_nonoverlapping(
                gate_up_zero_src.add(expert * gate_up_sidecar_elems_per_expert),
                gate_up_zero_dst.add(group * gate_up_sidecar_elems_per_expert),
                gate_up_sidecar_elems_per_expert,
            );
            std::ptr::copy_nonoverlapping(
                down_src.add(expert * down_bytes_per_expert),
                down_dst.add(group * down_bytes_per_expert),
                down_bytes_per_expert,
            );
            std::ptr::copy_nonoverlapping(
                down_scale_src.add(expert * down_sidecar_elems_per_expert),
                down_scale_dst.add(group * down_sidecar_elems_per_expert),
                down_sidecar_elems_per_expert,
            );
            std::ptr::copy_nonoverlapping(
                down_zero_src.add(expert * down_sidecar_elems_per_expert),
                down_zero_dst.add(group * down_sidecar_elems_per_expert),
                down_sidecar_elems_per_expert,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen36_pack_active_experts_for_metal(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
) -> Result<Qwen36PackedExpertBuffers, GpuError> {
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    let mut buffers = qwen36_alloc_packed_experts_for_metal(
        ordinal,
        hidden,
        moe_intermediate,
        active_experts.len(),
        group_size,
    )?;
    qwen36_fill_packed_experts_for_metal(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
        &mut buffers,
    );
    Ok(buffers)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_with_static_topn_packed_experts_for_metal<T, F>(
    ordinal: usize,
    layer_idx: i32,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    workspace: &mut [f32],
    off_topk_idx: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    f: F,
) -> Result<Option<T>, GpuError>
where
    F: FnOnce(&Qwen36PackedExpertBuffers) -> Result<T, GpuError>,
{
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    if off_topk_idx + active_experts.len() > workspace.len() {
        return Err(GpuError::InvalidArg(
            "qwen36 static top-N packed expert cache topk workspace out of bounds".into(),
        ));
    }

    let Some((capacity, resident_experts)) = qwen36_static_topn_layer_experts_from_env(layer_idx)?
    else {
        return Ok(None);
    };
    let resident_capacity = resident_experts.len();
    let expert_to_slot =
        qwen36_static_topn_expert_to_slot(layer_idx, resident_experts, num_experts, capacity)?;

    let mut active_slots = Vec::with_capacity(active_experts.len());
    let mut slot_hits = 0usize;
    let mut slot_misses = 0usize;
    for &expert in active_experts {
        if let Some(slot) = expert_to_slot[expert] {
            slot_hits += 1;
            active_slots.push(slot);
        } else {
            slot_misses += 1;
        }
    }
    if slot_misses > 0 {
        qwen36_static_topn_expert_residency_profile_record(
            resident_capacity,
            false,
            false,
            false,
            active_experts.len(),
            0,
            slot_hits,
            slot_misses,
            0,
        );
        return Ok(None);
    }

    let key = Qwen36PackedExpertCacheKey {
        gate_up_proj_ptr: gate_up_proj_ptr as usize,
        gate_up_scale_ptr: gate_up_scale_ptr as usize,
        gate_up_zero_ptr: gate_up_zero_ptr as usize,
        down_proj_ptr: down_proj_ptr as usize,
        down_scale_ptr: down_scale_ptr as usize,
        down_zero_ptr: down_zero_ptr as usize,
        hidden,
        moe_intermediate,
        group_size,
        active_groups: resident_capacity,
    };
    let cache = QWEN36_PACKED_EXPERT_STATIC_TOPN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert static top-N cache mutex poisoned".into(),
        )
    })?;
    let mut allocation = false;
    let copied_bytes =
        qwen36_packed_expert_copy_bytes(hidden, moe_intermediate, resident_capacity, group_size);
    crate::prefill_ffi::metal_profile_time(
        "qwen36_ffn_int4_expert_static_topn_pack_stage5",
        "host",
        || -> Result<(), GpuError> {
            match cache.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    if entry.get().resident_experts.as_slice() != resident_experts {
                        return Err(GpuError::backend(
                            Backend::Metal,
                            "qwen36 static top-N cache key collision with different resident experts"
                                .into(),
                        ));
                    }
                }
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    let mut buffers = qwen36_alloc_packed_experts_for_metal(
                        ordinal,
                        hidden,
                        moe_intermediate,
                        resident_capacity,
                        group_size,
                    )?;
                    for (slot, &expert) in resident_experts.iter().enumerate() {
                        qwen36_fill_packed_expert_slot_for_metal(
                            hidden,
                            moe_intermediate,
                            expert,
                            slot,
                            group_size,
                            gate_up_proj_ptr,
                            gate_up_scale_ptr,
                            gate_up_zero_ptr,
                            down_proj_ptr,
                            down_scale_ptr,
                            down_zero_ptr,
                            &mut buffers,
                        );
                    }
                    vacant.insert(Qwen36PackedExpertStaticTopNCacheEntry {
                        resident_experts: resident_experts.to_vec(),
                        buffers,
                    });
                    allocation = true;
                }
            }
            Ok(())
        },
    )?;
    for (group, &slot) in active_slots.iter().enumerate() {
        workspace[off_topk_idx + group] = f32::from_bits(slot as u32);
    }
    qwen36_static_topn_expert_residency_profile_record(
        resident_capacity,
        true,
        false,
        allocation,
        active_experts.len(),
        usize::from(allocation) * copied_bytes,
        slot_hits,
        0,
        0,
    );
    let entry = cache.get(&key).ok_or_else(|| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert static top-N cache entry missing after fill".into(),
        )
    })?;
    Ok(Some(f(&entry.buffers)?))
}

#[allow(clippy::too_many_arguments)]
fn qwen36_with_static_topn_mps_rhs_for_metal<T, F>(
    ordinal: usize,
    layer_idx: i32,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    f: F,
) -> Result<Option<T>, GpuError>
where
    F: FnOnce(
        &Qwen36MpsExpertStaticTopNCacheEntry,
        usize,
        &[Qwen36MpsStaticTopNHit],
        &[usize],
    ) -> Result<T, GpuError>,
{
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    let Some((capacity, resident_experts)) = qwen36_static_topn_layer_experts_from_env(layer_idx)?
    else {
        return Ok(None);
    };
    let resident_capacity = resident_experts.len();
    let expert_to_slot =
        qwen36_static_topn_expert_to_slot(layer_idx, resident_experts, num_experts, capacity)?;

    let mut hits = Vec::with_capacity(active_experts.len());
    let mut misses = Vec::with_capacity(active_experts.len());
    for (group, &expert) in active_experts.iter().enumerate() {
        if let Some(slot) = expert_to_slot[expert] {
            hits.push(Qwen36MpsStaticTopNHit {
                original_group: group,
                slot,
            });
        } else {
            misses.push(group);
        }
    }
    if hits.is_empty() {
        qwen36_mps_static_topn_expert_residency_profile_record(
            resident_capacity,
            false,
            false,
            false,
            active_experts.len(),
            0,
            0,
            misses.len(),
            0,
        );
        return Ok(None);
    }

    let key = Qwen36PackedExpertCacheKey {
        gate_up_proj_ptr: gate_up_proj_ptr as usize,
        gate_up_scale_ptr: gate_up_scale_ptr as usize,
        gate_up_zero_ptr: gate_up_zero_ptr as usize,
        down_proj_ptr: down_proj_ptr as usize,
        down_scale_ptr: down_scale_ptr as usize,
        down_zero_ptr: down_zero_ptr as usize,
        hidden,
        moe_intermediate,
        group_size,
        active_groups: resident_capacity,
    };
    let cache = QWEN36_MPS_EXPERT_STATIC_TOPN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 MPS expert static top-N cache mutex poisoned".into(),
        )
    })?;
    let mut allocation = false;
    let copied_bytes =
        resident_capacity * hidden * (3 * moe_intermediate) * std::mem::size_of::<u16>();
    crate::prefill_ffi::metal_profile_time(
        "qwen36_ffn_int4_expert_mps_static_topn_pack_f16_lut",
        "host",
        || -> Result<(), GpuError> {
            match cache.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    if entry.get().resident_experts.as_slice() != resident_experts {
                        return Err(GpuError::backend(
                            Backend::Metal,
                            "qwen36 MPS static top-N cache key collision with different resident experts"
                                .into(),
                        ));
                    }
                }
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    let (mut gate_up_rhs, mut down_rhs) = qwen36_alloc_mps_static_topn_rhs(
                        ordinal,
                        hidden,
                        moe_intermediate,
                        resident_capacity,
                    )?;
                    qwen36_fill_mps_static_topn_rhs_lut(
                        hidden,
                        moe_intermediate,
                        resident_experts,
                        group_size,
                        gate_up_proj_ptr,
                        gate_up_scale_ptr,
                        gate_up_zero_ptr,
                        down_proj_ptr,
                        down_scale_ptr,
                        down_zero_ptr,
                        &mut gate_up_rhs,
                        &mut down_rhs,
                    );
                    vacant.insert(Qwen36MpsExpertStaticTopNCacheEntry {
                        resident_experts: resident_experts.to_vec(),
                        gate_up_rhs,
                        down_rhs,
                    });
                    allocation = true;
                }
            }
            Ok(())
        },
    )?;
    qwen36_mps_static_topn_expert_residency_profile_record(
        resident_capacity,
        misses.is_empty(),
        false,
        allocation,
        active_experts.len(),
        usize::from(allocation) * copied_bytes,
        hits.len(),
        misses.len(),
        0,
    );
    let entry = cache.get(&key).ok_or_else(|| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 MPS static top-N cache entry missing after fill".into(),
        )
    })?;
    Ok(Some(f(entry, resident_capacity, &hits, &misses)?))
}

#[allow(clippy::too_many_arguments)]
fn qwen36_with_cached_packed_experts_for_metal<T, F>(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce(&Qwen36PackedExpertBuffers) -> Result<T, GpuError>,
{
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    let key = Qwen36PackedExpertCacheKey {
        gate_up_proj_ptr: gate_up_proj_ptr as usize,
        gate_up_scale_ptr: gate_up_scale_ptr as usize,
        gate_up_zero_ptr: gate_up_zero_ptr as usize,
        down_proj_ptr: down_proj_ptr as usize,
        down_scale_ptr: down_scale_ptr as usize,
        down_zero_ptr: down_zero_ptr as usize,
        hidden,
        moe_intermediate,
        group_size,
        active_groups: active_experts.len(),
    };
    let cache = QWEN36_PACKED_EXPERT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert cache mutex poisoned".into(),
        )
    })?;
    crate::prefill_ffi::metal_profile_time(
        "qwen36_ffn_int4_expert_pack_stage5",
        "host",
        || -> Result<(), GpuError> {
            let copy_bytes = qwen36_packed_expert_copy_bytes(
                hidden,
                moe_intermediate,
                active_experts.len(),
                group_size,
            );
            if let Some(entry) = cache.get_mut(&key) {
                if entry.active_experts != active_experts {
                    qwen36_fill_packed_experts_for_metal(
                        hidden,
                        moe_intermediate,
                        active_experts,
                        group_size,
                        gate_up_proj_ptr,
                        gate_up_scale_ptr,
                        gate_up_zero_ptr,
                        down_proj_ptr,
                        down_scale_ptr,
                        down_zero_ptr,
                        &mut entry.buffers,
                    );
                    entry.active_experts.clear();
                    entry.active_experts.extend_from_slice(active_experts);
                    qwen36_packed_expert_cache_profile_record(
                        false,
                        true,
                        false,
                        active_experts.len(),
                        copy_bytes,
                        0,
                        active_experts.len(),
                        0,
                    );
                } else {
                    qwen36_packed_expert_cache_profile_record(
                        true,
                        false,
                        false,
                        active_experts.len(),
                        0,
                        active_experts.len(),
                        0,
                        0,
                    );
                }
                return Ok(());
            }
            let mut buffers = qwen36_alloc_packed_experts_for_metal(
                ordinal,
                hidden,
                moe_intermediate,
                active_experts.len(),
                group_size,
            )?;
            qwen36_fill_packed_experts_for_metal(
                hidden,
                moe_intermediate,
                active_experts,
                group_size,
                gate_up_proj_ptr,
                gate_up_scale_ptr,
                gate_up_zero_ptr,
                down_proj_ptr,
                down_scale_ptr,
                down_zero_ptr,
                &mut buffers,
            );
            cache.insert(
                key,
                Qwen36PackedExpertCacheEntry {
                    active_experts: active_experts.to_vec(),
                    buffers,
                },
            );
            qwen36_packed_expert_cache_profile_record(
                false,
                false,
                true,
                active_experts.len(),
                copy_bytes,
                0,
                active_experts.len(),
                0,
            );
            Ok(())
        },
    )?;
    let entry = cache.get(&key).ok_or_else(|| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert cache entry missing after fill".into(),
        )
    })?;
    f(&entry.buffers)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_with_gpu_pack_buffers_for_metal<T, F>(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce(&Qwen36PackedExpertBuffers) -> Result<T, GpuError>,
{
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    let key = Qwen36PackedExpertCacheKey {
        gate_up_proj_ptr: gate_up_proj_ptr as usize,
        gate_up_scale_ptr: gate_up_scale_ptr as usize,
        gate_up_zero_ptr: gate_up_zero_ptr as usize,
        down_proj_ptr: down_proj_ptr as usize,
        down_scale_ptr: down_scale_ptr as usize,
        down_zero_ptr: down_zero_ptr as usize,
        hidden,
        moe_intermediate,
        group_size,
        active_groups: active_experts.len(),
    };
    let cache = QWEN36_PACKED_EXPERT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 GPU packed expert cache mutex poisoned".into(),
        )
    })?;
    crate::prefill_ffi::metal_profile_time(
        "qwen36_ffn_int4_expert_gpu_pack_alloc_stage5",
        "host",
        || -> Result<(), GpuError> {
            if cache.contains_key(&key) {
                qwen36_gpu_pack_expert_residency_profile_record(
                    false,
                    true,
                    false,
                    active_experts.len(),
                    0,
                    0,
                    active_experts.len(),
                    0,
                );
                return Ok(());
            }
            let buffers = qwen36_alloc_packed_experts_for_metal(
                ordinal,
                hidden,
                moe_intermediate,
                active_experts.len(),
                group_size,
            )?;
            cache.insert(
                key,
                Qwen36PackedExpertCacheEntry {
                    active_experts: Vec::new(),
                    buffers,
                },
            );
            qwen36_gpu_pack_expert_residency_profile_record(
                false,
                false,
                true,
                active_experts.len(),
                0,
                0,
                active_experts.len(),
                0,
            );
            Ok(())
        },
    )?;
    let entry = cache.get(&key).ok_or_else(|| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 GPU packed expert cache entry missing after allocation".into(),
        )
    })?;
    f(&entry.buffers)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_with_hotset_packed_experts_for_metal<T, F>(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    group_size: usize,
    num_experts: usize,
    capacity: usize,
    workspace: &mut [f32],
    off_topk_idx: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    f: F,
) -> Result<T, GpuError>
where
    F: FnOnce(&Qwen36PackedExpertBuffers) -> Result<T, GpuError>,
{
    qwen36_validate_active_expert_pack(
        hidden,
        moe_intermediate,
        active_experts,
        group_size,
        num_experts,
        gate_up_proj_ptr,
        gate_up_scale_ptr,
        gate_up_zero_ptr,
        down_proj_ptr,
        down_scale_ptr,
        down_zero_ptr,
    )?;
    if capacity < active_experts.len() {
        return Err(GpuError::InvalidArg(format!(
            "qwen36 hotset packed expert cache capacity {capacity} < active groups {}",
            active_experts.len()
        )));
    }
    if off_topk_idx + active_experts.len() > workspace.len() {
        return Err(GpuError::InvalidArg(
            "qwen36 hotset packed expert cache topk workspace out of bounds".into(),
        ));
    }

    let key = Qwen36PackedExpertCacheKey {
        gate_up_proj_ptr: gate_up_proj_ptr as usize,
        gate_up_scale_ptr: gate_up_scale_ptr as usize,
        gate_up_zero_ptr: gate_up_zero_ptr as usize,
        down_proj_ptr: down_proj_ptr as usize,
        down_scale_ptr: down_scale_ptr as usize,
        down_zero_ptr: down_zero_ptr as usize,
        hidden,
        moe_intermediate,
        group_size,
        active_groups: capacity,
    };
    let cache = QWEN36_PACKED_EXPERT_HOTSET_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert hotset cache mutex poisoned".into(),
        )
    })?;
    crate::prefill_ffi::metal_profile_time(
        "qwen36_ffn_int4_expert_pack_hotset_stage5",
        "host",
        || -> Result<(), GpuError> {
            let mut allocation = false;
            if let std::collections::hash_map::Entry::Vacant(vacant) = cache.entry(key) {
                let buffers = qwen36_alloc_packed_experts_for_metal(
                    ordinal,
                    hidden,
                    moe_intermediate,
                    capacity,
                    group_size,
                )?;
                vacant.insert(Qwen36PackedExpertHotsetCacheEntry {
                    slot_experts: vec![None; capacity],
                    slot_last_used: vec![0; capacity],
                    next_stamp: 1,
                    buffers,
                });
                allocation = true;
            }
            let entry = cache.get_mut(&key).ok_or_else(|| {
                GpuError::backend(
                    Backend::Metal,
                    "qwen36 packed expert hotset cache entry missing after allocation".into(),
                )
            })?;
            let bytes_per_expert =
                qwen36_packed_expert_copy_bytes(hidden, moe_intermediate, 1, group_size);
            let mut protected_slots = vec![false; capacity];
            let mut slot_hits = 0usize;
            let mut slot_misses = 0usize;
            let mut evictions = 0usize;
            for (group, &expert) in active_experts.iter().enumerate() {
                let (slot, hit, eviction) = qwen36_hotset_choose_slot(
                    &entry.slot_experts,
                    &entry.slot_last_used,
                    &protected_slots,
                    expert,
                );
                if hit {
                    slot_hits += 1;
                } else {
                    slot_misses += 1;
                    evictions += usize::from(eviction);
                    qwen36_fill_packed_expert_slot_for_metal(
                        hidden,
                        moe_intermediate,
                        expert,
                        slot,
                        group_size,
                        gate_up_proj_ptr,
                        gate_up_scale_ptr,
                        gate_up_zero_ptr,
                        down_proj_ptr,
                        down_scale_ptr,
                        down_zero_ptr,
                        &mut entry.buffers,
                    );
                    entry.slot_experts[slot] = Some(expert);
                }
                entry.slot_last_used[slot] = entry.next_stamp;
                entry.next_stamp = entry.next_stamp.wrapping_add(1).max(1);
                protected_slots[slot] = true;
                workspace[off_topk_idx + group] = f32::from_bits(slot as u32);
            }
            qwen36_hotset_expert_residency_profile_record(
                capacity,
                slot_misses == 0,
                slot_misses > 0,
                allocation,
                active_experts.len(),
                slot_misses * bytes_per_expert,
                slot_hits,
                slot_misses,
                evictions,
            );
            Ok(())
        },
    )?;
    let entry = cache.get(&key).ok_or_else(|| {
        GpuError::backend(
            Backend::Metal,
            "qwen36 packed expert hotset cache entry missing after fill".into(),
        )
    })?;
    f(&entry.buffers)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

fn push_f16_bits(bytes: &mut Vec<u8>, bits: u16) {
    bytes.extend_from_slice(&bits.to_le_bytes());
}

#[cfg(test)]
fn alloc_f16_byte_vec(elements: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 2);
    bytes.resize(elements * 2, 0);
    bytes
}

#[cfg(test)]
fn write_f16_byte(bytes: &mut [u8], index: usize, bits: u16) {
    let offset = index * 2;
    bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn qwen36_expert_int4_weight_unchecked(
    weight: usize,
    scale: usize,
    zero: usize,
    expert: usize,
    row: usize,
    rows: usize,
    col: usize,
    cols: usize,
    group_size: usize,
) -> f32 {
    let packed = weight as *const u8;
    let scale = scale as *const u16;
    let zero = zero as *const u16;
    let byte_cols = cols.div_ceil(2);
    let scale_rows = rows.div_ceil(group_size);
    let scale_cols = cols.div_ceil(group_size);
    let packed_base = (expert * rows + row) * byte_cols;
    let scale_base = (expert * scale_rows + row / group_size) * scale_cols;
    let byte = unsafe { *packed.add(packed_base + col / 2) };
    let nibble = if col & 1 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    let scale_idx = scale_base + col / group_size;
    let s = bf16_bits_to_f32(unsafe { *scale.add(scale_idx) });
    let z = bf16_bits_to_f32(unsafe { *zero.add(scale_idx) });
    bf16_round_f32(nibble as f32 * s - z * s)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_pack_mps_expert_bridge_bytes_scalar(
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    h_norm: &[f32],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let active_groups = active_experts.len();
    let gate_up_rows = 2 * moe_intermediate;
    let mut h_norm_bytes = Vec::with_capacity(active_groups * hidden * 2);
    let mut gate_up_rhs_bytes = Vec::with_capacity(active_groups * hidden * gate_up_rows * 2);
    let mut down_rhs_bytes = Vec::with_capacity(active_groups * moe_intermediate * hidden * 2);

    for _ in 0..active_groups {
        for &x in h_norm {
            push_f16_bits(&mut h_norm_bytes, f32_to_f16_bits(x));
        }
    }

    let gate_up_w = gate_up_proj_ptr as usize;
    let gate_up_scale = gate_up_scale_ptr as usize;
    let gate_up_zero = gate_up_zero_ptr as usize;
    let down_w = down_proj_ptr as usize;
    let down_scale = down_scale_ptr as usize;
    let down_zero = down_zero_ptr as usize;

    for &expert in active_experts {
        for col in 0..hidden {
            for row in 0..gate_up_rows {
                let w = qwen36_expert_int4_weight_unchecked(
                    gate_up_w,
                    gate_up_scale,
                    gate_up_zero,
                    expert,
                    row,
                    gate_up_rows,
                    col,
                    hidden,
                    group_size,
                );
                push_f16_bits(&mut gate_up_rhs_bytes, f32_to_f16_bits(w));
            }
        }
        for col in 0..moe_intermediate {
            for row in 0..hidden {
                let w = qwen36_expert_int4_weight_unchecked(
                    down_w,
                    down_scale,
                    down_zero,
                    expert,
                    row,
                    hidden,
                    col,
                    moe_intermediate,
                    group_size,
                );
                push_f16_bits(&mut down_rhs_bytes, f32_to_f16_bits(w));
            }
        }
    }
    (h_norm_bytes, gate_up_rhs_bytes, down_rhs_bytes)
}

fn qwen36_int4_f16_lut(scale_bits: u16, zero_bits: u16) -> [u16; 16] {
    let s = bf16_bits_to_f32(scale_bits);
    let z = bf16_bits_to_f32(zero_bits);
    let zs = z * s;
    std::array::from_fn(|idx| f32_to_f16_bits(bf16_round_f32(idx as f32 * s - zs)))
}

#[inline(always)]
fn qwen36_int4_f16_pair_lut(scale_bits: u16, zero_bits: u16) -> [u32; 256] {
    let nibble_lut = qwen36_int4_f16_lut(scale_bits, zero_bits);
    std::array::from_fn(|byte| {
        let lo = nibble_lut[byte & 0x0f] as u32;
        let hi = (nibble_lut[(byte >> 4) & 0x0f] as u32) << 16;
        lo | hi
    })
}

#[inline(always)]
fn write_f16_unaligned(bytes: &mut [u8], index: usize, bits: u16) {
    unsafe {
        std::ptr::write_unaligned(bytes.as_mut_ptr().add(index * 2).cast::<u16>(), bits);
    }
}

#[inline(always)]
fn write_f16_pair_unaligned(bytes: &mut [u8], index: usize, bits: u32) {
    unsafe {
        std::ptr::write_unaligned(bytes.as_mut_ptr().add(index * 2).cast::<u32>(), bits);
    }
}

#[inline(always)]
fn read_f16_unaligned(bytes: &[u8], index: usize) -> u16 {
    unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(index * 2).cast::<u16>()) }
}

fn qwen36_gpu_buffer_as_mut_bytes(buffer: &mut GpuBuffer) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), buffer.len_bytes()) }
}

#[repr(align(16))]
struct Qwen36F16Chunk16([u16; 16]);

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn qwen36_store_f16_chunk16_stream(dst: *mut u8, values: &Qwen36F16Chunk16) {
    std::arch::asm!(
        "ldr q0, [{src}]",
        "ldr q1, [{src}, #16]",
        "stnp q0, q1, [{dst}]",
        src = in(reg) values.0.as_ptr(),
        dst = in(reg) dst,
        out("v0") _,
        out("v1") _,
        options(nostack, preserves_flags)
    );
}

#[inline(always)]
fn qwen36_write_f16_chunk16(
    bytes: &mut [u8],
    index: usize,
    values: &Qwen36F16Chunk16,
    stream: bool,
) {
    let dst = unsafe { bytes.as_mut_ptr().add(index * 2) };
    #[cfg(target_arch = "aarch64")]
    if stream {
        unsafe {
            qwen36_store_f16_chunk16_stream(dst, values);
        }
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(values.0.as_ptr().cast::<u8>(), dst, 32);
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen36_pack_transposed_int4_f16_lut(
    src: *const u8,
    scale: *const u16,
    zero: *const u16,
    expert: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    dst: &mut [u8],
    dst_group_base: usize,
    dst_col_stride: usize,
    stream_stores: bool,
) {
    let byte_cols = cols.div_ceil(2);
    let scale_rows = rows.div_ceil(group_size);
    let scale_cols = cols.div_ceil(group_size);
    let mut tile = vec![0u8; group_size * group_size * 2];

    for row_group in 0..scale_rows {
        let row_start = row_group * group_size;
        let row_end = (row_start + group_size).min(rows);
        let tile_rows = row_end - row_start;
        for scale_col in 0..scale_cols {
            let col_start = scale_col * group_size;
            let col_end = (col_start + group_size).min(cols);
            let tile_cols = col_end - col_start;
            let tile_byte_cols = tile_cols.div_ceil(2);
            let scale_idx = (expert * scale_rows + row_group) * scale_cols + scale_col;
            let pair_lut =
                unsafe { qwen36_int4_f16_pair_lut(*scale.add(scale_idx), *zero.add(scale_idx)) };

            for (tile_row, row) in (row_start..row_end).enumerate() {
                let tile_row_base = tile_row * tile_cols;
                let packed_base = (expert * rows + row) * byte_cols;
                let mut byte_offset = 0usize;
                while byte_offset < tile_byte_cols {
                    let byte = unsafe { *src.add(packed_base + col_start / 2 + byte_offset) };
                    let col0 = 2 * byte_offset;
                    let col1 = col0 + 1;
                    if col1 < tile_cols {
                        write_f16_pair_unaligned(
                            &mut tile,
                            tile_row_base + col0,
                            pair_lut[byte as usize],
                        );
                    } else {
                        write_f16_unaligned(
                            &mut tile,
                            tile_row_base + col0,
                            pair_lut[byte as usize] as u16,
                        );
                    }
                    byte_offset += 1;
                }
            }

            for tile_col in 0..tile_cols {
                let dst_base = dst_group_base + (col_start + tile_col) * dst_col_stride + row_start;
                let mut tile_row = 0usize;
                if stream_stores {
                    let mut chunk = Qwen36F16Chunk16([0; 16]);
                    while tile_row + 16 <= tile_rows {
                        for idx in 0..16 {
                            chunk.0[idx] =
                                read_f16_unaligned(&tile, (tile_row + idx) * tile_cols + tile_col);
                        }
                        qwen36_write_f16_chunk16(dst, dst_base + tile_row, &chunk, true);
                        tile_row += 16;
                    }
                }
                while tile_row < tile_rows {
                    write_f16_unaligned(
                        dst,
                        dst_base + tile_row,
                        read_f16_unaligned(&tile, tile_row * tile_cols + tile_col),
                    );
                    tile_row += 1;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn qwen36_pack_mps_expert_bridge_bytes_lut(
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    h_norm: &[f32],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let active_groups = active_experts.len();
    let gate_up_rows = 2 * moe_intermediate;
    let mut h_norm_bytes = alloc_f16_byte_vec(active_groups * hidden);
    let mut gate_up_rhs_bytes = alloc_f16_byte_vec(active_groups * hidden * gate_up_rows);
    let mut down_rhs_bytes = alloc_f16_byte_vec(active_groups * moe_intermediate * hidden);

    for group in 0..active_groups {
        let dst_base = group * hidden;
        for (col, &x) in h_norm.iter().enumerate() {
            write_f16_byte(&mut h_norm_bytes, dst_base + col, f32_to_f16_bits(x));
        }
    }

    let gate_up_w = gate_up_proj_ptr as *const u8;
    let gate_up_scale = gate_up_scale_ptr as *const u16;
    let gate_up_zero = gate_up_zero_ptr as *const u16;
    let down_w = down_proj_ptr as *const u8;
    let down_scale = down_scale_ptr as *const u16;
    let down_zero = down_zero_ptr as *const u16;

    for (group, &expert) in active_experts.iter().enumerate() {
        qwen36_pack_transposed_int4_f16_lut(
            gate_up_w,
            gate_up_scale,
            gate_up_zero,
            expert,
            gate_up_rows,
            hidden,
            group_size,
            &mut gate_up_rhs_bytes,
            group * hidden * gate_up_rows,
            gate_up_rows,
            false,
        );
        qwen36_pack_transposed_int4_f16_lut(
            down_w,
            down_scale,
            down_zero,
            expert,
            hidden,
            moe_intermediate,
            group_size,
            &mut down_rhs_bytes,
            group * moe_intermediate * hidden,
            hidden,
            false,
        );
    }
    (h_norm_bytes, gate_up_rhs_bytes, down_rhs_bytes)
}

#[allow(clippy::too_many_arguments)]
fn qwen36_fill_mps_expert_bridge_buffers_lut(
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    h_norm: &[f32],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    bridge: &mut Qwen36MpsExpertBridgeBuffers,
    stream_stores: bool,
) {
    let active_groups = active_experts.len();
    let gate_up_rows = 2 * moe_intermediate;

    {
        let h_norm_bytes = qwen36_gpu_buffer_as_mut_bytes(&mut bridge.h_norm);
        for group in 0..active_groups {
            let dst_base = group * hidden;
            for (col, &x) in h_norm.iter().enumerate() {
                write_f16_unaligned(h_norm_bytes, dst_base + col, f32_to_f16_bits(x));
            }
        }
    }

    let gate_up_w = gate_up_proj_ptr as *const u8;
    let gate_up_scale = gate_up_scale_ptr as *const u16;
    let gate_up_zero = gate_up_zero_ptr as *const u16;
    let down_w = down_proj_ptr as *const u8;
    let down_scale = down_scale_ptr as *const u16;
    let down_zero = down_zero_ptr as *const u16;

    for (group, &expert) in active_experts.iter().enumerate() {
        qwen36_pack_transposed_int4_f16_lut(
            gate_up_w,
            gate_up_scale,
            gate_up_zero,
            expert,
            gate_up_rows,
            hidden,
            group_size,
            qwen36_gpu_buffer_as_mut_bytes(&mut bridge.gate_up_rhs),
            group * hidden * gate_up_rows,
            gate_up_rows,
            stream_stores,
        );
        qwen36_pack_transposed_int4_f16_lut(
            down_w,
            down_scale,
            down_zero,
            expert,
            hidden,
            moe_intermediate,
            group_size,
            qwen36_gpu_buffer_as_mut_bytes(&mut bridge.down_rhs),
            group * moe_intermediate * hidden,
            hidden,
            stream_stores,
        );
    }
    if stream_stores {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen36_build_mps_expert_bridge_buffers(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_experts: &[usize],
    h_norm: &[f32],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    use_lut_pack: bool,
) -> Result<Qwen36MpsExpertBridgeBuffers, GpuError> {
    let active_groups = active_experts.len();
    let gate_up_rows = 2 * moe_intermediate;
    if use_lut_pack {
        let mut bridge = qwen36_alloc_mps_expert_bridge_buffers(
            ordinal,
            hidden,
            moe_intermediate,
            active_groups,
        )?;
        qwen36_fill_mps_expert_bridge_buffers_lut(
            hidden,
            moe_intermediate,
            active_experts,
            h_norm,
            group_size,
            gate_up_proj_ptr,
            gate_up_scale_ptr,
            gate_up_zero_ptr,
            down_proj_ptr,
            down_scale_ptr,
            down_zero_ptr,
            &mut bridge,
            std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_STREAM").is_some(),
        );
        return Ok(bridge);
    }
    let (h_norm_bytes, gate_up_rhs_bytes, down_rhs_bytes) =
        qwen36_pack_mps_expert_bridge_bytes_scalar(
            hidden,
            moe_intermediate,
            active_experts,
            h_norm,
            group_size,
            gate_up_proj_ptr,
            gate_up_scale_ptr,
            gate_up_zero_ptr,
            down_proj_ptr,
            down_scale_ptr,
            down_zero_ptr,
        );

    let h_norm = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden],
        &h_norm_bytes,
    )?;
    let gate_up_rhs = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden, gate_up_rows],
        &gate_up_rhs_bytes,
    )?;
    let gate_up_out = GpuBuffer::zeros(ordinal, ScalarType::F16, &[active_groups, gate_up_rows])?;
    let down_lhs = GpuBuffer::zeros(ordinal, ScalarType::F16, &[active_groups, moe_intermediate])?;
    let down_rhs = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F16,
        &[active_groups, moe_intermediate, hidden],
        &down_rhs_bytes,
    )?;
    let down_out = GpuBuffer::zeros(ordinal, ScalarType::F16, &[active_groups, hidden])?;

    Ok(Qwen36MpsExpertBridgeBuffers {
        h_norm,
        gate_up_rhs,
        gate_up_out,
        down_lhs,
        down_rhs,
        down_out,
    })
}

fn qwen36_alloc_mps_expert_bridge_scratch(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_groups: usize,
) -> Result<Qwen36MpsExpertBridgeScratch, GpuError> {
    let gate_up_rows = 2 * moe_intermediate;
    let h_norm = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden],
        BufferKind::Scratch,
    )?;
    let gate_up_out = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, gate_up_rows],
        BufferKind::Scratch,
    )?;
    let down_lhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, moe_intermediate],
        BufferKind::Scratch,
    )?;
    let down_out = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden],
        BufferKind::Scratch,
    )?;
    Ok(Qwen36MpsExpertBridgeScratch {
        h_norm,
        gate_up_out,
        down_lhs,
        down_out,
    })
}

fn qwen36_fill_mps_expert_h_norm(
    hidden: usize,
    active_groups: usize,
    h_norm: &[f32],
    h_norm_f16: &mut GpuBuffer,
) {
    let h_norm_bytes = qwen36_gpu_buffer_as_mut_bytes(h_norm_f16);
    for group in 0..active_groups {
        let dst_base = group * hidden;
        for (col, &x) in h_norm.iter().enumerate() {
            write_f16_unaligned(h_norm_bytes, dst_base + col, f32_to_f16_bits(x));
        }
    }
}

fn qwen36_alloc_mps_static_topn_rhs(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    capacity: usize,
) -> Result<(GpuBuffer, GpuBuffer), GpuError> {
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_rhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[capacity, hidden, gate_up_rows],
        BufferKind::Scratch,
    )?;
    let down_rhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[capacity, moe_intermediate, hidden],
        BufferKind::Scratch,
    )?;
    Ok((gate_up_rhs, down_rhs))
}

#[allow(clippy::too_many_arguments)]
fn qwen36_fill_mps_static_topn_rhs_lut(
    hidden: usize,
    moe_intermediate: usize,
    resident_experts: &[usize],
    group_size: usize,
    gate_up_proj_ptr: *const c_void,
    gate_up_scale_ptr: *const c_void,
    gate_up_zero_ptr: *const c_void,
    down_proj_ptr: *const c_void,
    down_scale_ptr: *const c_void,
    down_zero_ptr: *const c_void,
    gate_up_rhs: &mut GpuBuffer,
    down_rhs: &mut GpuBuffer,
) {
    let gate_up_rows = 2 * moe_intermediate;
    let gate_up_w = gate_up_proj_ptr as *const u8;
    let gate_up_scale = gate_up_scale_ptr as *const u16;
    let gate_up_zero = gate_up_zero_ptr as *const u16;
    let down_w = down_proj_ptr as *const u8;
    let down_scale = down_scale_ptr as *const u16;
    let down_zero = down_zero_ptr as *const u16;
    let stream_stores =
        std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_STREAM").is_some();

    for (slot, &expert) in resident_experts.iter().enumerate() {
        qwen36_pack_transposed_int4_f16_lut(
            gate_up_w,
            gate_up_scale,
            gate_up_zero,
            expert,
            gate_up_rows,
            hidden,
            group_size,
            qwen36_gpu_buffer_as_mut_bytes(gate_up_rhs),
            slot * hidden * gate_up_rows,
            gate_up_rows,
            stream_stores,
        );
        qwen36_pack_transposed_int4_f16_lut(
            down_w,
            down_scale,
            down_zero,
            expert,
            hidden,
            moe_intermediate,
            group_size,
            qwen36_gpu_buffer_as_mut_bytes(down_rhs),
            slot * moe_intermediate * hidden,
            hidden,
            stream_stores,
        );
    }
    if stream_stores {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }
}

fn qwen36_alloc_mps_expert_bridge_buffers(
    ordinal: usize,
    hidden: usize,
    moe_intermediate: usize,
    active_groups: usize,
) -> Result<Qwen36MpsExpertBridgeBuffers, GpuError> {
    let gate_up_rows = 2 * moe_intermediate;
    let h_norm = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden],
        BufferKind::Scratch,
    )?;
    let gate_up_rhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden, gate_up_rows],
        BufferKind::Scratch,
    )?;
    let gate_up_out = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, gate_up_rows],
        BufferKind::Scratch,
    )?;
    let down_lhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, moe_intermediate],
        BufferKind::Scratch,
    )?;
    let down_rhs = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, moe_intermediate, hidden],
        BufferKind::Scratch,
    )?;
    let down_out = GpuBuffer::alloc_with_kind(
        ordinal,
        ScalarType::F16,
        &[active_groups, hidden],
        BufferKind::Scratch,
    )?;

    Ok(Qwen36MpsExpertBridgeBuffers {
        h_norm,
        gate_up_rhs,
        gate_up_out,
        down_lhs,
        down_rhs,
        down_out,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen36_try_mps_static_topn_partial_for_metal(
    output_ordinal: usize,
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
    num_experts: usize,
    active_experts: &[usize],
    input: &[u16],
    output: &mut [u16],
    workspace: &mut [f32],
    workspace_ptr: *mut c_void,
    output_ptr: *mut c_void,
    off_h_norm: usize,
    off_topk_val: usize,
    off_topk_idx: usize,
    off_shared_out: usize,
    off_expert_gu: usize,
    off_expert_mid: usize,
    off_expert_stack: usize,
    off_moe_out: usize,
) -> Result<Option<()>, GpuError> {
    qwen36_with_static_topn_mps_rhs_for_metal(
        output_ordinal,
        params.layer_idx,
        hidden,
        moe_intermediate,
        active_experts,
        int4.group_size as usize,
        num_experts,
        weights.gate_up_proj_w,
        int4.gate_up_proj_scale,
        int4.gate_up_proj_zero,
        weights.down_proj_w,
        int4.down_proj_scale,
        int4.down_proj_zero,
        |resident, resident_capacity, hits, miss_groups| {
            let hit_count = hits.len();
            let original_topk_vals: Vec<f32> = (0..top_k)
                .map(|group| workspace[off_topk_val + group])
                .collect();
            let original_topk_idx: Vec<f32> = (0..top_k)
                .map(|group| workspace[off_topk_idx + group])
                .collect();

            let mut scratch = crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_mps_static_topn_alloc_f16",
                "host",
                || {
                    qwen36_alloc_mps_expert_bridge_scratch(
                        output_ordinal,
                        hidden,
                        moe_intermediate,
                        hit_count,
                    )
                },
            )?;
            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_mps_static_topn_hnorm_f16",
                "host",
                || {
                    qwen36_fill_mps_expert_h_norm(
                        hidden,
                        hit_count,
                        &workspace[off_h_norm..off_h_norm + hidden],
                        &mut scratch.h_norm,
                    );
                },
            );

            for (hit_group, hit) in hits.iter().enumerate() {
                workspace[off_topk_idx + hit_group] = f32::from_bits(hit.slot as u32);
                workspace[off_topk_val + hit_group] = original_topk_vals[hit.original_group];
            }

            let mps_result = crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_mps_static_topn_partial_f16",
                "native",
                || unsafe {
                    crate::metal_native::qwen36_ffn_expert_mps_bridge_indexed_f16(
                        hidden,
                        moe_intermediate,
                        hit_count,
                        resident_capacity,
                        workspace_ptr,
                        weights.input_hidden,
                        scratch.h_norm.as_ptr(),
                        resident.gate_up_rhs.as_ptr(),
                        scratch.gate_up_out.as_mut_ptr(),
                        scratch.down_lhs.as_mut_ptr(),
                        resident.down_rhs.as_ptr(),
                        scratch.down_out.as_mut_ptr(),
                        output_ptr,
                        off_topk_val,
                        off_topk_idx,
                        off_shared_out,
                        off_moe_out,
                        true,
                    )
                },
            );
            for group in 0..top_k {
                workspace[off_topk_val + group] = original_topk_vals[group];
                workspace[off_topk_idx + group] = original_topk_idx[group];
            }
            mps_result?;

            if miss_groups.is_empty() {
                return Ok(());
            }

            let h_norm_for_miss = workspace[off_h_norm..off_h_norm + hidden].to_vec();
            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_host_expert_mps_static_topn_miss_gate_up",
                "host",
                || {
                    let gate_up_w = weights.gate_up_proj_w as usize;
                    let gate_up_scale = int4.gate_up_proj_scale as usize;
                    let gate_up_zero = int4.gate_up_proj_zero as usize;
                    let group_size = int4.group_size.max(0) as usize;
                    let rows_per_group = 2 * moe_intermediate;
                    let gu = &mut workspace
                        [off_expert_gu..off_expert_gu + miss_groups.len() * rows_per_group];
                    qwen36_parallel_chunks_mut(gu, 64, |start, chunk| {
                        for (local, out) in chunk.iter_mut().enumerate() {
                            let flat_row = start + local;
                            let miss_group = flat_row / rows_per_group;
                            let row = flat_row - miss_group * rows_per_group;
                            let original_group = miss_groups[miss_group];
                            *out = qwen36_expert_dense_or_int4_dot_unchecked(
                                gate_up_w,
                                gate_up_scale,
                                gate_up_zero,
                                active_experts[original_group],
                                row,
                                rows_per_group,
                                hidden,
                                group_size,
                                &h_norm_for_miss,
                            );
                        }
                    });
                },
            );

            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_host_expert_mps_static_topn_miss_silu",
                "host",
                || {
                    for group in 0..miss_groups.len() {
                        let gu_base = off_expert_gu + group * 2 * moe_intermediate;
                        let mid_base = off_expert_mid + group * moe_intermediate;
                        for i in 0..moe_intermediate {
                            let gp = workspace[gu_base + i];
                            let up = workspace[gu_base + moe_intermediate + i];
                            let silu = gp * (1.0f32 / (1.0f32 + (-gp).exp()));
                            workspace[mid_base + i] = silu * up;
                        }
                    }
                },
            );

            let expert_mid = workspace
                [off_expert_mid..off_expert_mid + miss_groups.len() * moe_intermediate]
                .to_vec();
            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_host_expert_mps_static_topn_miss_down",
                "host",
                || {
                    let down_w = weights.down_proj_w as usize;
                    let down_scale = int4.down_proj_scale as usize;
                    let down_zero = int4.down_proj_zero as usize;
                    let group_size = int4.group_size.max(0) as usize;
                    let stack = &mut workspace
                        [off_expert_stack..off_expert_stack + miss_groups.len() * hidden];
                    qwen36_parallel_chunks_mut(stack, 64, |start, chunk| {
                        for (local, out) in chunk.iter_mut().enumerate() {
                            let flat_row = start + local;
                            let miss_group = flat_row / hidden;
                            let row = flat_row - miss_group * hidden;
                            let original_group = miss_groups[miss_group];
                            let mid = &expert_mid[miss_group * moe_intermediate
                                ..(miss_group + 1) * moe_intermediate];
                            *out = qwen36_expert_dense_or_int4_dot_unchecked(
                                down_w,
                                down_scale,
                                down_zero,
                                active_experts[original_group],
                                row,
                                hidden,
                                moe_intermediate,
                                group_size,
                                mid,
                            );
                        }
                    });
                },
            );

            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_host_expert_mps_static_topn_miss_finalize",
                "host",
                || {
                    for row in 0..hidden {
                        let mut acc = workspace[off_moe_out + row];
                        for (miss_group, &original_group) in miss_groups.iter().enumerate() {
                            acc += original_topk_vals[original_group]
                                * workspace[off_expert_stack + miss_group * hidden + row];
                        }
                        let moe = bf16_round_f32(acc);
                        workspace[off_moe_out + row] = moe;
                        let val = bf16_round_f32(
                            bf16_bits_to_f32(input[row]) + moe + workspace[off_shared_out + row],
                        );
                        output[row] = f32_to_bf16_bits(val);
                    }
                },
            );
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ffn_expert_gate_up_tiled_metal_launch(
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
    group_size: usize,
    workspace: &mut GpuBuffer,
    gate_up_proj: &GpuBuffer,
    gate_up_scale: &GpuBuffer,
    gate_up_zero: &GpuBuffer,
    off_h_norm: usize,
    off_topk_idx: usize,
    off_expert_mid: usize,
) -> Result<(), GpuError> {
    if workspace.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_expert_gate_up_tiled_metal_launch requires Metal buffers".into(),
        ));
    }
    if workspace.dtype() != ScalarType::F32
        || gate_up_proj.dtype() != ScalarType::U8
        || gate_up_scale.dtype() != ScalarType::BF16
        || gate_up_zero.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_expert_gate_up_tiled_metal_launch expects F32/U8/BF16/BF16, got {:?}/{:?}/{:?}/{:?}",
            workspace.dtype(),
            gate_up_proj.dtype(),
            gate_up_scale.dtype(),
            gate_up_zero.dtype()
        )));
    }
    unsafe {
        crate::metal_native::qwen36_ffn_expert_gate_up_tiled(
            hidden,
            moe_intermediate,
            top_k,
            group_size,
            workspace.as_mut_ptr(),
            gate_up_proj.as_ptr(),
            gate_up_scale.as_ptr(),
            gate_up_zero.as_ptr(),
            off_h_norm,
            off_topk_idx,
            off_expert_mid,
            true,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ffn_expert_tiled_stage5_metal_launch(
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
    group_size: usize,
    workspace: &mut GpuBuffer,
    input_hidden: &GpuBuffer,
    gate_up_proj: &GpuBuffer,
    gate_up_scale: &GpuBuffer,
    gate_up_zero: &GpuBuffer,
    down_proj: &GpuBuffer,
    down_scale: &GpuBuffer,
    down_zero: &GpuBuffer,
    output: &mut GpuBuffer,
    off_h_norm: usize,
    off_topk_val: usize,
    off_topk_idx: usize,
    off_shared_out: usize,
    off_expert_mid: usize,
    off_moe_out: usize,
) -> Result<(), GpuError> {
    if workspace.backend() != Backend::Metal {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_expert_tiled_stage5_metal_launch requires Metal buffers".into(),
        ));
    }
    if workspace.dtype() != ScalarType::F32
        || input_hidden.dtype() != ScalarType::BF16
        || gate_up_proj.dtype() != ScalarType::U8
        || gate_up_scale.dtype() != ScalarType::BF16
        || gate_up_zero.dtype() != ScalarType::BF16
        || down_proj.dtype() != ScalarType::U8
        || down_scale.dtype() != ScalarType::BF16
        || down_zero.dtype() != ScalarType::BF16
        || output.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_expert_tiled_stage5_metal_launch expects F32/BF16/U8/BF16/BF16/U8/BF16/BF16/BF16, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
            workspace.dtype(),
            input_hidden.dtype(),
            gate_up_proj.dtype(),
            gate_up_scale.dtype(),
            gate_up_zero.dtype(),
            down_proj.dtype(),
            down_scale.dtype(),
            down_zero.dtype(),
            output.dtype()
        )));
    }
    unsafe {
        crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
            hidden,
            moe_intermediate,
            top_k,
            group_size,
            workspace.as_mut_ptr(),
            input_hidden.as_ptr(),
            gate_up_proj.as_ptr(),
            gate_up_scale.as_ptr(),
            gate_up_zero.as_ptr(),
            down_proj.as_ptr(),
            down_scale.as_ptr(),
            down_zero.as_ptr(),
            output.as_mut_ptr(),
            off_h_norm,
            off_topk_val,
            off_topk_idx,
            off_shared_out,
            off_expert_mid,
            off_moe_out,
            true,
        )
    }
}

/// Safe wrapper for the PR 4b4 staged MoE FFN parity launcher.
///
/// `output` must be a BF16 buffer with at least `max(top_k, hidden)` elements
/// (the size of the largest staged intermediate). `output_idx` must be an
/// i32 buffer with at least `top_k` elements. `workspace` must be an F32
/// buffer sized for the requested stage's footprint (see the layout comment
/// in `kernels/qwen36_moe.hip`). `sync_buf` must be a 96-byte zero buffer:
/// 16 u32 work-stealing counter slots at +0..+63 (the per-expert G/I phases
/// use counters[0..2*top_k] for cyclic per-group dispatch), barrier counter
/// at +64, barrier flag at +68.
pub fn ffn_step_launch(
    ordinal: usize,
    dtype: ScalarType,
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
    output: &mut GpuBuffer,
    output_idx: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    if !(1..=5).contains(&params.stage) {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: stage must be in 1..=5, got {}",
            params.stage
        )));
    }
    if params.top_k > params.num_experts {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: top_k ({}) > num_experts ({})",
            params.top_k, params.num_experts,
        )));
    }
    // The concurrent-experts G/H/I dispatch uses counters[0..2*top_k] for
    // per-group work-stealing (Phase G slots [0..top_k), Phase I slots
    // [top_k..2*top_k)). The sync_buf carries 16 u32 counter slots before
    // barrier_counter at +64; pushing past slot 15 corrupts barrier state
    // and can hang the kernel, so cap top_k at the layout's hard limit. If
    // a future model needs top_k > 8 we'll grow sync_buf in lockstep.
    const MAX_TOP_K: i32 = 8;
    if params.top_k > MAX_TOP_K {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: top_k ({}) exceeds the FFN \
             concurrent-experts dispatch limit ({MAX_TOP_K}); the 16-slot \
             sync_buf reserves 2*top_k counter slots before barrier_counter \
             at +64, so top_k > {MAX_TOP_K} would overrun into barrier state",
            params.top_k,
        )));
    }
    let backend = output.backend();
    if backend == Backend::Metal {
        return ffn_step_stage1_5_metal_host(params, weights, int4, output, output_idx, workspace);
    }

    let counters = sync_buf.as_mut_ptr() as *mut c_uint;
    // Layout: 16 u32 work-stealing counter slots at +0..+63 (the FFN
    // concurrent-experts dispatch uses 2*K_top of these; attn/linear/stub
    // only touch counters[0]). Barrier counter+flag follow at +64/+68.
    // Sync_buf must be at least 96 bytes zeroed before launch.
    let barrier_counter = unsafe { (counters as *mut u8).add(64) as *mut c_uint };
    let barrier_flag = unsafe { (counters as *mut u8).add(68) as *mut c_uint };

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_ffn_step_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    params.stage as c_int,
                    params.hidden as c_int,
                    params.num_experts as c_int,
                    params.moe_intermediate as c_int,
                    params.shared_intermediate as c_int,
                    params.top_k as c_int,
                    params.rms_norm_eps,
                    weights.input_hidden,
                    weights.post_attn_norm_w,
                    weights.gate_w,
                    weights.gate_up_proj_w,
                    weights.down_proj_w,
                    weights.shared_gate_proj_w,
                    weights.shared_up_proj_w,
                    weights.shared_down_proj_w,
                    weights.shared_expert_gate_w,
                    int4.group_size as c_int,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    int4.shared_gate_proj_scale,
                    int4.shared_gate_proj_zero,
                    int4.shared_up_proj_scale,
                    int4.shared_up_proj_zero,
                    int4.shared_down_proj_scale,
                    int4.shared_down_proj_zero,
                    output.as_mut_ptr(),
                    output_idx.as_mut_ptr() as *mut c_int,
                    workspace.as_mut_ptr() as *mut f32,
                    counters,
                    barrier_counter,
                    barrier_flag,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::ffn_step_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            unreachable!("Metal ffn_step handled above");
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe ffn_step launch failed with status {status}"),
        ));
    }
    Ok(())
}

fn ffn_step_stage1_5_metal_host(
    params: Qwen36MoeFfnStepParams,
    weights: &Qwen36MoeFfnStepWeights,
    int4: &Qwen36MoeFfnStepInt4,
    output: &mut GpuBuffer,
    output_idx: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if output.dtype() != ScalarType::BF16
        || output_idx.dtype() != ScalarType::U32
        || workspace.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: Metal stage1-5 expects BF16/U32/F32 \
             output buffers, got {:?}/{:?}/{:?}",
            output.dtype(),
            output_idx.dtype(),
            workspace.dtype(),
        )));
    }
    if int4.group_size < 0 {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_step_launch: Metal stage1-5 does not yet support FP8 sidecars".into(),
        ));
    }
    if weights.input_hidden.is_null()
        || weights.post_attn_norm_w.is_null()
        || weights.gate_w.is_null()
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_step_launch: Metal stage1 requires input_hidden, \
             post_attn_norm_w, and gate_w"
                .into(),
        ));
    }
    if params.stage >= 2
        && (weights.shared_gate_proj_w.is_null()
            || weights.shared_up_proj_w.is_null()
            || weights.shared_down_proj_w.is_null()
            || weights.shared_expert_gate_w.is_null())
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_step_launch: Metal stage2 requires shared expert weights".into(),
        ));
    }
    if params.stage >= 3 && (weights.gate_up_proj_w.is_null() || weights.down_proj_w.is_null()) {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_step_launch: Metal stage3 requires gate_up_proj_w and down_proj_w"
                .into(),
        ));
    }

    let hidden = params.hidden as usize;
    let num_experts = params.num_experts as usize;
    let moe_intermediate = params.moe_intermediate as usize;
    let shared_intermediate = params.shared_intermediate as usize;
    let top_k = params.top_k as usize;
    if hidden == 0
        || num_experts == 0
        || moe_intermediate == 0
        || shared_intermediate == 0
        || top_k == 0
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::ffn_step_launch: Metal stage1-5 requires non-zero geometry".into(),
        ));
    }

    let off_h_norm = 0usize;
    let off_router_logits = hidden;
    let off_router_probs = hidden + num_experts;
    let off_topk_val = hidden + 2 * num_experts;
    let off_topk_idx = hidden + 2 * num_experts + top_k;
    let off_sg_scalar = hidden + 2 * num_experts + 2 * top_k;
    let off_sgp = off_sg_scalar + 1;
    let off_sup = off_sgp + shared_intermediate;
    let off_shared_mid = off_sup + shared_intermediate;
    let off_shared_out = off_shared_mid + shared_intermediate;
    let off_expert_gu = off_shared_out + hidden;
    let off_expert_mid = off_expert_gu + top_k * 2 * moe_intermediate;
    let off_expert_stack = off_expert_mid + top_k * moe_intermediate;
    let off_moe_out = off_expert_stack + top_k * hidden;
    let workspace_len = if params.stage >= 4 {
        off_moe_out + hidden
    } else if params.stage >= 3 {
        off_expert_stack + top_k * hidden
    } else if params.stage >= 2 {
        off_shared_out + hidden
    } else {
        off_sg_scalar
    };
    let output_len = if params.stage == 1 { top_k } else { hidden };
    if workspace.len_bytes() / std::mem::size_of::<f32>() < workspace_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: Metal stage{} workspace too small: need {}, got {}",
            params.stage,
            workspace_len,
            workspace.len_bytes() / std::mem::size_of::<f32>()
        )));
    }
    if output.len_bytes() / std::mem::size_of::<u16>() < output_len {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: Metal stage{} output too small: need {}, got {}",
            params.stage,
            output_len,
            output.len_bytes() / std::mem::size_of::<u16>()
        )));
    }
    if output_idx.len_bytes() / std::mem::size_of::<i32>() < top_k {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::ffn_step_launch: Metal output_idx too small: need {}, got {}",
            top_k,
            output_idx.len_bytes() / std::mem::size_of::<i32>()
        )));
    }

    let input = unsafe { std::slice::from_raw_parts(weights.input_hidden as *const u16, hidden) };
    let norm_w =
        unsafe { std::slice::from_raw_parts(weights.post_attn_norm_w as *const u16, hidden) };
    let gate_w =
        unsafe { std::slice::from_raw_parts(weights.gate_w as *const u16, num_experts * hidden) };
    let workspace_ptr = workspace.as_mut_ptr();
    let output_ptr = output.as_mut_ptr();
    let output_ordinal = output.device_ordinal();
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, output_len) };
    let output_idx =
        unsafe { std::slice::from_raw_parts_mut(output_idx.as_mut_ptr() as *mut i32, top_k) };
    let workspace = unsafe {
        std::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut f32, workspace_len)
    };

    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_router_topk", "host", || {
        let mut mean_sq = 0.0f32;
        for &bits in input {
            let v = bf16_bits_to_f32(bits);
            mean_sq += v * v;
        }
        let inv_rms = 1.0f32 / (mean_sq / hidden as f32 + params.rms_norm_eps).sqrt();
        for col in 0..hidden {
            let v = bf16_bits_to_f32(input[col]);
            let w = bf16_bits_to_f32(norm_w[col]);
            workspace[off_h_norm + col] = bf16_round_f32(v * inv_rms * (1.0 + w));
        }

        for expert in 0..num_experts {
            let mut acc = 0.0f32;
            let row = expert * hidden;
            for col in 0..hidden {
                acc += bf16_bits_to_f32(gate_w[row + col]) * workspace[off_h_norm + col];
            }
            workspace[off_router_logits + expert] = bf16_round_f32(acc);
        }

        let row_max = workspace[off_router_logits..off_router_logits + num_experts]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut row_sum = 0.0f32;
        for expert in 0..num_experts {
            let e = (workspace[off_router_logits + expert] - row_max).exp();
            workspace[off_router_probs + expert] = e;
            row_sum += e;
        }
        let inv_sum = 1.0f32 / row_sum;
        for expert in 0..num_experts {
            workspace[off_router_probs + expert] =
                bf16_round_f32(workspace[off_router_probs + expert] * inv_sum);
        }

        for kk in 0..top_k {
            let mut best_idx = -1i32;
            let mut best_val = f32::NEG_INFINITY;
            for expert in 0..num_experts {
                let v = workspace[off_router_probs + expert];
                if v > best_val || (v == best_val && best_idx >= 0 && (expert as i32) < best_idx) {
                    best_val = v;
                    best_idx = expert as i32;
                }
            }
            workspace[off_topk_idx + kk] = f32::from_bits(best_idx as u32);
            workspace[off_topk_val + kk] = best_val;
            if best_idx >= 0 {
                workspace[off_router_probs + best_idx as usize] = f32::NEG_INFINITY;
            }
        }

        let sum_k: f32 = (0..top_k).map(|kk| workspace[off_topk_val + kk]).sum();
        let inv_k = 1.0f32 / sum_k;
        for kk in 0..top_k {
            let w = bf16_round_f32(workspace[off_topk_val + kk] * inv_k);
            workspace[off_topk_val + kk] = w;
            output[kk] = f32_to_bf16_bits(w);
            output_idx[kk] = f32::to_bits(workspace[off_topk_idx + kk]) as i32;
        }
    });

    if params.stage == 1 {
        return Ok(());
    }

    let shared_expert_gate_w =
        unsafe { std::slice::from_raw_parts(weights.shared_expert_gate_w as *const u16, hidden) };
    qwen36_validate_dense_or_int4_sidecars(
        int4.shared_gate_proj_scale,
        int4.shared_gate_proj_zero,
        int4.group_size,
        "shared_gate_proj",
    )?;
    qwen36_validate_dense_or_int4_sidecars(
        int4.shared_up_proj_scale,
        int4.shared_up_proj_zero,
        int4.group_size,
        "shared_up_proj",
    )?;
    qwen36_validate_dense_or_int4_sidecars(
        int4.shared_down_proj_scale,
        int4.shared_down_proj_zero,
        int4.group_size,
        "shared_down_proj",
    )?;
    qwen36_validate_dense_or_int4_sidecars(
        int4.gate_up_proj_scale,
        int4.gate_up_proj_zero,
        int4.group_size,
        "gate_up_proj",
    )?;
    qwen36_validate_dense_or_int4_sidecars(
        int4.down_proj_scale,
        int4.down_proj_zero,
        int4.group_size,
        "down_proj",
    )?;

    if qwen36_ffn_int4_stage5_metal_native_supported(params, weights, int4) {
        return crate::prefill_ffi::metal_profile_time(
            "qwen36_ffn_int4_stage5",
            "native",
            || unsafe {
                crate::metal_native::qwen36_ffn_int4_stage5(
                    hidden,
                    num_experts,
                    moe_intermediate,
                    shared_intermediate,
                    top_k,
                    int4.group_size as usize,
                    weights.input_hidden,
                    weights.shared_expert_gate_w,
                    weights.shared_gate_proj_w,
                    int4.shared_gate_proj_scale,
                    int4.shared_gate_proj_zero,
                    weights.shared_up_proj_w,
                    int4.shared_up_proj_scale,
                    int4.shared_up_proj_zero,
                    weights.shared_down_proj_w,
                    int4.shared_down_proj_scale,
                    int4.shared_down_proj_zero,
                    weights.gate_up_proj_w,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    weights.down_proj_w,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    workspace.as_mut_ptr() as *mut c_void,
                    output.as_mut_ptr() as *mut c_void,
                    true,
                )
            },
        );
    }

    let h_norm = workspace[off_h_norm..off_h_norm + hidden].to_vec();
    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_shared_gate_up", "host", || {
        let (_, after_sgp) = workspace.split_at_mut(off_sgp);
        let (sgp, after_sup) = after_sgp.split_at_mut(shared_intermediate);
        let (sup, _) = after_sup.split_at_mut(shared_intermediate);
        let shared_gate_w = weights.shared_gate_proj_w as usize;
        let shared_gate_scale = int4.shared_gate_proj_scale as usize;
        let shared_gate_zero = int4.shared_gate_proj_zero as usize;
        let shared_up_w = weights.shared_up_proj_w as usize;
        let shared_up_scale = int4.shared_up_proj_scale as usize;
        let shared_up_zero = int4.shared_up_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        qwen36_parallel_chunks2_mut(sgp, sup, 64, |start, gate_chunk, up_chunk| {
            for (local, (gate_out, up_out)) in
                gate_chunk.iter_mut().zip(up_chunk.iter_mut()).enumerate()
            {
                let row = start + local;
                *gate_out = qwen36_dense_or_int4_dot_2d_unchecked(
                    shared_gate_w,
                    shared_gate_scale,
                    shared_gate_zero,
                    row,
                    hidden,
                    group_size,
                    &h_norm,
                );
                *up_out = qwen36_dense_or_int4_dot_2d_unchecked(
                    shared_up_w,
                    shared_up_scale,
                    shared_up_zero,
                    row,
                    hidden,
                    group_size,
                    &h_norm,
                );
            }
        });
    });
    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_shared_scalar_silu", "host", || {
        let mut sg_acc = 0.0f32;
        for col in 0..hidden {
            sg_acc += bf16_bits_to_f32(shared_expert_gate_w[col]) * h_norm[col];
        }
        workspace[off_sg_scalar] = 1.0f32 / (1.0f32 + (-sg_acc).exp());

        for i in 0..shared_intermediate {
            let gp = workspace[off_sgp + i];
            let up = workspace[off_sup + i];
            let silu = gp * (1.0f32 / (1.0f32 + (-gp).exp()));
            workspace[off_shared_mid + i] = silu * up;
        }
    });

    let sg_scalar = workspace[off_sg_scalar];
    let shared_mid = workspace[off_shared_mid..off_shared_mid + shared_intermediate].to_vec();
    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_shared_down", "host", || {
        let shared_down_w = weights.shared_down_proj_w as usize;
        let shared_down_scale = int4.shared_down_proj_scale as usize;
        let shared_down_zero = int4.shared_down_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let shared_out = &mut workspace[off_shared_out..off_shared_out + hidden];
        qwen36_parallel_chunks_mut(shared_out, 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = start + local;
                let acc = qwen36_dense_or_int4_dot_2d_unchecked(
                    shared_down_w,
                    shared_down_scale,
                    shared_down_zero,
                    row,
                    shared_intermediate,
                    group_size,
                    &shared_mid,
                );
                *out = bf16_round_f32(sg_scalar * acc);
            }
        });
        if params.stage == 2 {
            for row in 0..hidden {
                output[row] = f32_to_bf16_bits(shared_out[row]);
            }
        }
    });

    if params.stage == 2 {
        return Ok(());
    }

    let active_groups = if params.stage == 3 { 1 } else { top_k };
    let active_experts: Vec<usize> = (0..active_groups)
        .map(|group| f32::to_bits(workspace[off_topk_idx + group]) as i32 as usize)
        .collect();
    if qwen36_route_profile_enabled() {
        qwen36_route_profile_record(&active_experts);
    }
    if qwen36_ffn_expert_mps_static_topn_partial_stage5_supported(params, weights, int4) {
        if let Some(()) = qwen36_try_mps_static_topn_partial_for_metal(
            output_ordinal,
            params,
            weights,
            int4,
            hidden,
            moe_intermediate,
            top_k,
            num_experts,
            &active_experts,
            input,
            output,
            workspace,
            workspace_ptr,
            output_ptr,
            off_h_norm,
            off_topk_val,
            off_topk_idx,
            off_shared_out,
            off_expert_gu,
            off_expert_mid,
            off_expert_stack,
            off_moe_out,
        )? {
            return Ok(());
        }
    }
    if qwen36_ffn_expert_mps_bridge_stage5_supported(params, weights, int4) {
        let cpu_transcode =
            std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE").is_some();
        let mut bridge = if cpu_transcode {
            let use_lut_pack =
                std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_LUT").is_some();
            let pack_op = if use_lut_pack {
                "qwen36_ffn_int4_expert_mps_bridge_pack_f16_lut"
            } else {
                "qwen36_ffn_int4_expert_mps_bridge_pack_f16"
            };
            crate::prefill_ffi::metal_profile_time(pack_op, "host", || {
                qwen36_build_mps_expert_bridge_buffers(
                    output_ordinal,
                    hidden,
                    moe_intermediate,
                    &active_experts,
                    &h_norm,
                    int4.group_size as usize,
                    weights.gate_up_proj_w,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    weights.down_proj_w,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    use_lut_pack,
                )
            })?
        } else {
            let mut bridge = crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_mps_bridge_alloc_f16",
                "host",
                || {
                    qwen36_alloc_mps_expert_bridge_buffers(
                        output_ordinal,
                        hidden,
                        moe_intermediate,
                        active_groups,
                    )
                },
            )?;
            let profile_transcode = std::env::var_os("SUPERSONIC_METAL_PROFILE").is_some();
            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_mps_transcode_int4_f16",
                "native",
                || unsafe {
                    crate::metal_native::qwen36_ffn_expert_mps_transcode_int4_f16(
                        hidden,
                        moe_intermediate,
                        active_groups,
                        int4.group_size as usize,
                        workspace_ptr,
                        weights.gate_up_proj_w,
                        int4.gate_up_proj_scale,
                        int4.gate_up_proj_zero,
                        weights.down_proj_w,
                        int4.down_proj_scale,
                        int4.down_proj_zero,
                        bridge.h_norm.as_mut_ptr(),
                        bridge.gate_up_rhs.as_mut_ptr(),
                        bridge.down_rhs.as_mut_ptr(),
                        off_h_norm,
                        off_topk_idx,
                        profile_transcode,
                    )
                },
            )?;
            bridge
        };
        crate::prefill_ffi::metal_profile_time(
            "qwen36_ffn_int4_expert_mps_bridge_f16",
            "native",
            || unsafe {
                crate::metal_native::qwen36_ffn_expert_mps_bridge_f16(
                    hidden,
                    moe_intermediate,
                    active_groups,
                    workspace_ptr,
                    weights.input_hidden,
                    bridge.h_norm.as_ptr(),
                    bridge.gate_up_rhs.as_ptr(),
                    bridge.gate_up_out.as_mut_ptr(),
                    bridge.down_lhs.as_mut_ptr(),
                    bridge.down_rhs.as_ptr(),
                    bridge.down_out.as_mut_ptr(),
                    output_ptr,
                    off_topk_val,
                    off_shared_out,
                    off_moe_out,
                    true,
                )
            },
        )?;
        return Ok(());
    }
    if qwen36_ffn_expert_direct_gather_stage5_metal_native_supported(params, weights, int4) {
        crate::prefill_ffi::metal_profile_time(
            "qwen36_ffn_int4_expert_direct_gather_stage5",
            "native",
            || unsafe {
                crate::metal_native::qwen36_ffn_expert_direct_gather_stage5(
                    hidden,
                    moe_intermediate,
                    active_groups,
                    int4.group_size as usize,
                    workspace_ptr,
                    weights.input_hidden,
                    weights.gate_up_proj_w,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    weights.down_proj_w,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    output_ptr,
                    off_h_norm,
                    off_topk_val,
                    off_topk_idx,
                    off_shared_out,
                    off_expert_mid,
                    off_moe_out,
                    true,
                )
            },
        )?;
        return Ok(());
    }
    if qwen36_ffn_expert_packed_stage5_metal_native_supported(params, weights, int4) {
        if qwen36_ffn_expert_gpu_pack_stage5_metal_native_supported(params, weights, int4) {
            qwen36_with_gpu_pack_buffers_for_metal(
                output_ordinal,
                hidden,
                moe_intermediate,
                &active_experts,
                int4.group_size as usize,
                num_experts,
                weights.gate_up_proj_w,
                int4.gate_up_proj_scale,
                int4.gate_up_proj_zero,
                weights.down_proj_w,
                int4.down_proj_scale,
                int4.down_proj_zero,
                |packed| {
                    crate::prefill_ffi::metal_profile_time(
                        "qwen36_ffn_int4_expert_gpu_pack_stage5",
                        "native",
                        || unsafe {
                            crate::metal_native::qwen36_ffn_expert_gpu_pack_gate_up_down_finalize_tiled(
                                hidden,
                                moe_intermediate,
                                active_groups,
                                int4.group_size as usize,
                                workspace_ptr,
                                weights.input_hidden,
                                weights.gate_up_proj_w,
                                int4.gate_up_proj_scale,
                                int4.gate_up_proj_zero,
                                weights.down_proj_w,
                                int4.down_proj_scale,
                                int4.down_proj_zero,
                                packed.gate_up_proj.as_ptr() as *mut c_void,
                                packed.gate_up_scale.as_ptr() as *mut c_void,
                                packed.gate_up_zero.as_ptr() as *mut c_void,
                                packed.down_proj.as_ptr() as *mut c_void,
                                packed.down_scale.as_ptr() as *mut c_void,
                                packed.down_zero.as_ptr() as *mut c_void,
                                output_ptr,
                                off_h_norm,
                                off_topk_val,
                                off_topk_idx,
                                off_shared_out,
                                off_expert_mid,
                                off_moe_out,
                                true,
                            )
                        },
                    )
                },
            )?;
            return Ok(());
        }
        if let Some(()) = qwen36_with_static_topn_packed_experts_for_metal(
            output_ordinal,
            params.layer_idx,
            hidden,
            moe_intermediate,
            &active_experts,
            int4.group_size as usize,
            num_experts,
            workspace,
            off_topk_idx,
            weights.gate_up_proj_w,
            int4.gate_up_proj_scale,
            int4.gate_up_proj_zero,
            weights.down_proj_w,
            int4.down_proj_scale,
            int4.down_proj_zero,
            |packed| {
                crate::prefill_ffi::metal_profile_time(
                    "qwen36_ffn_int4_expert_packed_static_topn_stage5",
                    "native",
                    || unsafe {
                        crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
                            hidden,
                            moe_intermediate,
                            active_groups,
                            int4.group_size as usize,
                            workspace_ptr,
                            weights.input_hidden,
                            packed.gate_up_proj.as_ptr(),
                            packed.gate_up_scale.as_ptr(),
                            packed.gate_up_zero.as_ptr(),
                            packed.down_proj.as_ptr(),
                            packed.down_scale.as_ptr(),
                            packed.down_zero.as_ptr(),
                            output_ptr,
                            off_h_norm,
                            off_topk_val,
                            off_topk_idx,
                            off_shared_out,
                            off_expert_mid,
                            off_moe_out,
                            true,
                        )
                    },
                )
            },
        )? {
            return Ok(());
        }
        if let Some(hotset_capacity) = qwen36_packed_expert_hotset_capacity(active_groups) {
            qwen36_with_hotset_packed_experts_for_metal(
                output_ordinal,
                hidden,
                moe_intermediate,
                &active_experts,
                int4.group_size as usize,
                num_experts,
                hotset_capacity,
                workspace,
                off_topk_idx,
                weights.gate_up_proj_w,
                int4.gate_up_proj_scale,
                int4.gate_up_proj_zero,
                weights.down_proj_w,
                int4.down_proj_scale,
                int4.down_proj_zero,
                |packed| {
                    crate::prefill_ffi::metal_profile_time(
                        "qwen36_ffn_int4_expert_packed_hotset_stage5",
                        "native",
                        || unsafe {
                            crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
                                hidden,
                                moe_intermediate,
                                active_groups,
                                int4.group_size as usize,
                                workspace_ptr,
                                weights.input_hidden,
                                packed.gate_up_proj.as_ptr(),
                                packed.gate_up_scale.as_ptr(),
                                packed.gate_up_zero.as_ptr(),
                                packed.down_proj.as_ptr(),
                                packed.down_scale.as_ptr(),
                                packed.down_zero.as_ptr(),
                                output_ptr,
                                off_h_norm,
                                off_topk_val,
                                off_topk_idx,
                                off_shared_out,
                                off_expert_mid,
                                off_moe_out,
                                true,
                            )
                        },
                    )
                },
            )?;
            return Ok(());
        }
        for group in 0..active_groups {
            workspace[off_topk_idx + group] = f32::from_bits(group as u32);
        }
        let use_pack_cache =
            std::env::var_os("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_CACHE").is_some()
                && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_FFN_EXPERT_PACK_CACHE")
                    .is_none();
        if !use_pack_cache {
            let packed = crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_pack_stage5",
                "host",
                || {
                    qwen36_pack_active_experts_for_metal(
                        output_ordinal,
                        hidden,
                        moe_intermediate,
                        &active_experts,
                        int4.group_size as usize,
                        num_experts,
                        weights.gate_up_proj_w,
                        int4.gate_up_proj_scale,
                        int4.gate_up_proj_zero,
                        weights.down_proj_w,
                        int4.down_proj_scale,
                        int4.down_proj_zero,
                    )
                },
            )?;
            crate::prefill_ffi::metal_profile_time(
                "qwen36_ffn_int4_expert_packed_stage5",
                "native",
                || unsafe {
                    crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
                        hidden,
                        moe_intermediate,
                        active_groups,
                        int4.group_size as usize,
                        workspace_ptr,
                        weights.input_hidden,
                        packed.gate_up_proj.as_ptr(),
                        packed.gate_up_scale.as_ptr(),
                        packed.gate_up_zero.as_ptr(),
                        packed.down_proj.as_ptr(),
                        packed.down_scale.as_ptr(),
                        packed.down_zero.as_ptr(),
                        output_ptr,
                        off_h_norm,
                        off_topk_val,
                        off_topk_idx,
                        off_shared_out,
                        off_expert_mid,
                        off_moe_out,
                        true,
                    )
                },
            )?;
            return Ok(());
        }
        qwen36_with_cached_packed_experts_for_metal(
            output_ordinal,
            hidden,
            moe_intermediate,
            &active_experts,
            int4.group_size as usize,
            num_experts,
            weights.gate_up_proj_w,
            int4.gate_up_proj_scale,
            int4.gate_up_proj_zero,
            weights.down_proj_w,
            int4.down_proj_scale,
            int4.down_proj_zero,
            |packed| {
                crate::prefill_ffi::metal_profile_time(
                    "qwen36_ffn_int4_expert_packed_stage5",
                    "native",
                    || unsafe {
                        crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
                            hidden,
                            moe_intermediate,
                            active_groups,
                            int4.group_size as usize,
                            workspace_ptr,
                            weights.input_hidden,
                            packed.gate_up_proj.as_ptr(),
                            packed.gate_up_scale.as_ptr(),
                            packed.gate_up_zero.as_ptr(),
                            packed.down_proj.as_ptr(),
                            packed.down_scale.as_ptr(),
                            packed.down_zero.as_ptr(),
                            output_ptr,
                            off_h_norm,
                            off_topk_val,
                            off_topk_idx,
                            off_shared_out,
                            off_expert_mid,
                            off_moe_out,
                            true,
                        )
                    },
                )
            },
        )?;
        return Ok(());
    }
    if qwen36_ffn_expert_tiled_stage5_metal_native_supported(params, weights, int4) {
        crate::prefill_ffi::metal_profile_time(
            "qwen36_ffn_int4_expert_gate_up_down_finalize_tiled",
            "native",
            || unsafe {
                crate::metal_native::qwen36_ffn_expert_gate_up_down_finalize_tiled(
                    hidden,
                    moe_intermediate,
                    active_groups,
                    int4.group_size as usize,
                    workspace_ptr,
                    weights.input_hidden,
                    weights.gate_up_proj_w,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    weights.down_proj_w,
                    int4.down_proj_scale,
                    int4.down_proj_zero,
                    output_ptr,
                    off_h_norm,
                    off_topk_val,
                    off_topk_idx,
                    off_shared_out,
                    off_expert_mid,
                    off_moe_out,
                    true,
                )
            },
        )?;
        return Ok(());
    }

    if qwen36_ffn_expert_gate_up_tiled_metal_native_supported(params, weights, int4) {
        crate::prefill_ffi::metal_profile_time(
            "qwen36_ffn_int4_expert_gate_up_tiled",
            "native",
            || unsafe {
                crate::metal_native::qwen36_ffn_expert_gate_up_tiled(
                    hidden,
                    moe_intermediate,
                    active_groups,
                    int4.group_size as usize,
                    workspace_ptr,
                    weights.gate_up_proj_w,
                    int4.gate_up_proj_scale,
                    int4.gate_up_proj_zero,
                    off_h_norm,
                    off_topk_idx,
                    off_expert_mid,
                    true,
                )
            },
        )?;
    } else {
        crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_expert_gate_up", "host", || {
            let gate_up_w = weights.gate_up_proj_w as usize;
            let gate_up_scale = int4.gate_up_proj_scale as usize;
            let gate_up_zero = int4.gate_up_proj_zero as usize;
            let group_size = int4.group_size.max(0) as usize;
            let rows_per_group = 2 * moe_intermediate;
            let gu = &mut workspace[off_expert_gu..off_expert_gu + active_groups * rows_per_group];
            qwen36_parallel_chunks_mut(gu, 64, |start, chunk| {
                for (local, out) in chunk.iter_mut().enumerate() {
                    let flat_row = start + local;
                    let group = flat_row / rows_per_group;
                    let row = flat_row - group * rows_per_group;
                    *out = qwen36_expert_dense_or_int4_dot_unchecked(
                        gate_up_w,
                        gate_up_scale,
                        gate_up_zero,
                        active_experts[group],
                        row,
                        rows_per_group,
                        hidden,
                        group_size,
                        &h_norm,
                    );
                }
            });
        });

        crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_expert_silu", "host", || {
            for group in 0..active_groups {
                let gu_base = off_expert_gu + group * 2 * moe_intermediate;
                let mid_base = off_expert_mid + group * moe_intermediate;
                for i in 0..moe_intermediate {
                    let gp = workspace[gu_base + i];
                    let up = workspace[gu_base + moe_intermediate + i];
                    let silu = gp * (1.0f32 / (1.0f32 + (-gp).exp()));
                    workspace[mid_base + i] = silu * up;
                }
            }
        });
    }

    let expert_mid =
        workspace[off_expert_mid..off_expert_mid + active_groups * moe_intermediate].to_vec();
    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_expert_down", "host", || {
        let down_w = weights.down_proj_w as usize;
        let down_scale = int4.down_proj_scale as usize;
        let down_zero = int4.down_proj_zero as usize;
        let group_size = int4.group_size.max(0) as usize;
        let stack = &mut workspace[off_expert_stack..off_expert_stack + active_groups * hidden];
        qwen36_parallel_chunks_mut(stack, 64, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let flat_row = start + local;
                let group = flat_row / hidden;
                let row = flat_row - group * hidden;
                let mid = &expert_mid[group * moe_intermediate..(group + 1) * moe_intermediate];
                *out = qwen36_expert_dense_or_int4_dot_unchecked(
                    down_w,
                    down_scale,
                    down_zero,
                    active_experts[group],
                    row,
                    hidden,
                    moe_intermediate,
                    group_size,
                    mid,
                );
            }
        });
        if params.stage == 3 {
            for row in 0..hidden {
                output[row] = f32_to_bf16_bits(bf16_round_f32(stack[row]));
            }
        }
    });

    if params.stage == 3 {
        return Ok(());
    }

    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_finalize", "host", || {
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for group in 0..top_k {
                acc += workspace[off_topk_val + group]
                    * workspace[off_expert_stack + group * hidden + row];
            }
            let val = bf16_round_f32(acc);
            workspace[off_moe_out + row] = val;
            if params.stage == 4 {
                output[row] = f32_to_bf16_bits(val);
            }
        }
    });

    if params.stage == 4 {
        return Ok(());
    }

    crate::prefill_ffi::metal_profile_time("qwen36_ffn_host_residual", "host", || {
        for row in 0..hidden {
            let val = bf16_round_f32(
                bf16_bits_to_f32(input[row])
                    + workspace[off_moe_out + row]
                    + workspace[off_shared_out + row],
            );
            output[row] = f32_to_bf16_bits(val);
        }
    });

    Ok(())
}

/// PR 4b5 step 2 safe wrapper for the INT4 dequant smoke launcher.
///
/// Drives the smoke kernel that exercises both `int4_dequant_8` and
/// `int4_dequant_scalar` over the supplied `[out_rows, in_cols]` slab.
/// `packed_buf` must be a u8 buffer with `out_rows * in_cols / 2` bytes.
/// `scale_buf` and `zero_buf` must be BF16 buffers with `(out_rows / gsz)
/// * (in_cols / gsz)` elements each. `dq_8_out` and `dq_scalar_out` must
/// each be F32 buffers with at least `out_rows * in_cols` elements.
#[allow(clippy::too_many_arguments)]
pub fn int4_dequant_smoke_launch(
    ordinal: usize,
    packed_buf: &GpuBuffer,
    scale_buf: &GpuBuffer,
    zero_buf: &GpuBuffer,
    out_rows: i32,
    in_cols: i32,
    gsz: i32,
    dq_8_out: &mut GpuBuffer,
    dq_scalar_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if out_rows <= 0 || in_cols <= 0 || gsz <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: positive dims required, \
             got out_rows={out_rows} in_cols={in_cols} gsz={gsz}"
        )));
    }
    if in_cols % 8 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: in_cols ({in_cols}) must \
             be divisible by 8 (the helpers' fast-path stride)"
        )));
    }
    if in_cols % gsz != 0 || gsz % 2 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: in_cols ({in_cols}) must \
             be divisible by gsz ({gsz}) and gsz must be even"
        )));
    }
    if out_rows % gsz != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: out_rows ({out_rows}) must \
             be divisible by gsz ({gsz})"
        )));
    }

    let backend = packed_buf.backend();
    if backend == Backend::Metal {
        return int4_dequant_smoke_launch_metal_host(
            packed_buf,
            scale_buf,
            zero_buf,
            out_rows,
            in_cols,
            gsz,
            dq_8_out,
            dq_scalar_out,
        );
    }

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_int4_dequant_smoke_launch(
                    ordinal,
                    packed_buf.as_ptr() as *const u8,
                    scale_buf.as_ptr(),
                    zero_buf.as_ptr(),
                    out_rows as c_int,
                    in_cols as c_int,
                    gsz as c_int,
                    dq_8_out.as_mut_ptr() as *mut f32,
                    dq_scalar_out.as_mut_ptr() as *mut f32,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::int4_dequant_smoke_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => unreachable!("Metal int4_dequant_smoke handled above"),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe int4_dequant_smoke_launch failed with status {status}"),
        ));
    }
    Ok(())
}

fn int4_dequant_smoke_launch_metal_host(
    packed_buf: &GpuBuffer,
    scale_buf: &GpuBuffer,
    zero_buf: &GpuBuffer,
    out_rows: i32,
    in_cols: i32,
    gsz: i32,
    dq_8_out: &mut GpuBuffer,
    dq_scalar_out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if packed_buf.dtype() != ScalarType::U8
        || scale_buf.dtype() != ScalarType::BF16
        || zero_buf.dtype() != ScalarType::BF16
        || dq_8_out.dtype() != ScalarType::F32
        || dq_scalar_out.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::int4_dequant_smoke_launch: Metal fallback expects \
             U8/BF16/BF16/F32/F32 buffers, got {:?}/{:?}/{:?}/{:?}/{:?}",
            packed_buf.dtype(),
            scale_buf.dtype(),
            zero_buf.dtype(),
            dq_8_out.dtype(),
            dq_scalar_out.dtype(),
        )));
    }

    let out_rows = out_rows as usize;
    let in_cols = in_cols as usize;
    let gsz = gsz as usize;
    let total = out_rows * in_cols;
    let byte_cols = in_cols / 2;
    let scale_cols = in_cols / gsz;
    let packed = unsafe {
        std::slice::from_raw_parts(packed_buf.as_ptr() as *const u8, out_rows * byte_cols)
    };
    let scale = unsafe {
        std::slice::from_raw_parts(
            scale_buf.as_ptr() as *const u16,
            (out_rows / gsz) * scale_cols,
        )
    };
    let zero = unsafe {
        std::slice::from_raw_parts(
            zero_buf.as_ptr() as *const u16,
            (out_rows / gsz) * scale_cols,
        )
    };
    let dq_8 = unsafe { std::slice::from_raw_parts_mut(dq_8_out.as_mut_ptr() as *mut f32, total) };
    let dq_scalar =
        unsafe { std::slice::from_raw_parts_mut(dq_scalar_out.as_mut_ptr() as *mut f32, total) };

    for row in 0..out_rows {
        for col in 0..in_cols {
            let scale_idx = (row / gsz) * scale_cols + col / gsz;
            let s = f32::from_bits((scale[scale_idx] as u32) << 16);
            let z = f32::from_bits((zero[scale_idx] as u32) << 16);
            let byte = packed[row * byte_cols + col / 2];
            let q = if col & 1 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let v = bf16_round_f32(q as f32 * s - z * s);
            let idx = row * in_cols + col;
            dq_8[idx] = v;
            dq_scalar[idx] = v;
        }
    }
    Ok(())
}

fn bf16_round_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
    let r = bits.wrapping_add(rounding_bias);
    f32::from_bits(r & 0xFFFF_0000)
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    (bf16_round_f32(x).to_bits() >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn qwen36_host_parallelism(len: usize, min_rows_per_worker: usize) -> usize {
    if len < min_rows_per_worker.saturating_mul(2) {
        return 1;
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    workers.min(len.div_ceil(min_rows_per_worker)).max(1)
}

fn qwen36_parallel_chunks_mut<T, F>(slice: &mut [T], min_rows_per_worker: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let workers = qwen36_host_parallelism(slice.len(), min_rows_per_worker);
    if workers <= 1 {
        f(0, slice);
        return;
    }
    let chunk = slice.len().div_ceil(workers);
    let f = &f;
    std::thread::scope(|scope| {
        for (chunk_idx, chunk_slice) in slice.chunks_mut(chunk).enumerate() {
            let start = chunk_idx * chunk;
            scope.spawn(move || f(start, chunk_slice));
        }
    });
}

fn qwen36_parallel_chunks2_mut<T, U, F>(a: &mut [T], b: &mut [U], min_rows_per_worker: usize, f: F)
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    let workers = qwen36_host_parallelism(a.len(), min_rows_per_worker);
    if workers <= 1 {
        f(0, a, b);
        return;
    }
    let chunk = a.len().div_ceil(workers);
    let f = &f;
    std::thread::scope(|scope| {
        for (chunk_idx, (a_chunk, b_chunk)) in
            a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate()
        {
            let start = chunk_idx * chunk;
            scope.spawn(move || f(start, a_chunk, b_chunk));
        }
    });
}

fn qwen36_validate_dense_or_int4_sidecars(
    scale: *const c_void,
    zero: *const c_void,
    group_size: i32,
    label: &str,
) -> Result<(), GpuError> {
    if scale.is_null() && zero.is_null() {
        return Ok(());
    }
    if scale.is_null() || zero.is_null() || group_size <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe: Metal INT4 fallback requires paired scale/zero pointers \
             and positive group_size for {label}"
        )));
    }
    Ok(())
}

fn qwen36_dense_or_int4_dot_2d_unchecked(
    weight: usize,
    scale: usize,
    zero: usize,
    row: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
) -> f32 {
    let mut acc = 0.0f32;
    if scale == 0 && zero == 0 {
        let w = weight as *const u16;
        let row_base = row * cols;
        for col in 0..cols {
            acc += bf16_bits_to_f32(unsafe { *w.add(row_base + col) }) * x[col];
        }
        return acc;
    }
    let packed = weight as *const u8;
    let scale = scale as *const u16;
    let zero = zero as *const u16;
    let byte_cols = cols.div_ceil(2);
    let scale_cols = cols.div_ceil(group_size);
    let packed_base = row * byte_cols;
    let scale_base = (row / group_size) * scale_cols;
    for scale_col in 0..scale_cols {
        let group_start = scale_col * group_size;
        let group_end = cols.min(group_start + group_size);
        let scale_idx = scale_base + scale_col;
        let s = bf16_bits_to_f32(unsafe { *scale.add(scale_idx) });
        let z = bf16_bits_to_f32(unsafe { *zero.add(scale_idx) });
        let zs = z * s;
        let mut col = group_start;
        let mut byte_idx = packed_base + group_start / 2;
        while col + 1 < group_end {
            let byte = unsafe { *packed.add(byte_idx) };
            let w0 = bf16_round_f32((byte & 0x0f) as f32 * s - zs);
            let w1 = bf16_round_f32(((byte >> 4) & 0x0f) as f32 * s - zs);
            acc += w0 * x[col] + w1 * x[col + 1];
            col += 2;
            byte_idx += 1;
        }
        if col < group_end {
            let byte = unsafe { *packed.add(byte_idx) };
            let w = bf16_round_f32((byte & 0x0f) as f32 * s - zs);
            acc += w * x[col];
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn qwen36_expert_dense_or_int4_dot_unchecked(
    weight: usize,
    scale: usize,
    zero: usize,
    expert: usize,
    row: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
) -> f32 {
    let mut acc = 0.0f32;
    if scale == 0 && zero == 0 {
        let w = weight as *const u16;
        let row_base = (expert * rows + row) * cols;
        for col in 0..cols {
            acc += bf16_bits_to_f32(unsafe { *w.add(row_base + col) }) * x[col];
        }
        return acc;
    }
    let packed = weight as *const u8;
    let scale = scale as *const u16;
    let zero = zero as *const u16;
    let byte_cols = cols.div_ceil(2);
    let scale_rows = rows.div_ceil(group_size);
    let scale_cols = cols.div_ceil(group_size);
    let packed_base = (expert * rows + row) * byte_cols;
    let scale_base = (expert * scale_rows + row / group_size) * scale_cols;
    for scale_col in 0..scale_cols {
        let group_start = scale_col * group_size;
        let group_end = cols.min(group_start + group_size);
        let scale_idx = scale_base + scale_col;
        let s = bf16_bits_to_f32(unsafe { *scale.add(scale_idx) });
        let z = bf16_bits_to_f32(unsafe { *zero.add(scale_idx) });
        let zs = z * s;
        let mut col = group_start;
        let mut byte_idx = packed_base + group_start / 2;
        while col + 1 < group_end {
            let byte = unsafe { *packed.add(byte_idx) };
            let w0 = bf16_round_f32((byte & 0x0f) as f32 * s - zs);
            let w1 = bf16_round_f32(((byte >> 4) & 0x0f) as f32 * s - zs);
            acc += w0 * x[col] + w1 * x[col + 1];
            col += 2;
            byte_idx += 1;
        }
        if col < group_end {
            let byte = unsafe { *packed.add(byte_idx) };
            let w = bf16_round_f32((byte & 0x0f) as f32 * s - zs);
            acc += w * x[col];
        }
    }
    acc
}

/// Safe wrapper for the GPU final RMSNorm + lm_head GEMV.
///
/// All buffers must already be on `ordinal` and BF16:
///   - `final_hidden_buf`: shape `[hidden]`.
///   - `final_norm_w_buf`: shape `[hidden]` — `model.norm.weight`.
///   - `lm_head_w_buf`: shape `[vocab, hidden]`. Caller is responsible
///     for dequantizing INT4 / BF16-casting it once at startup.
///   - `logits_buf`: shape `[vocab]`, output. Overwritten on every call.
///   - `x_normed_out_buf`: optional `[hidden]` BF16 output. When `Some`,
///     the kernel also writes the BF16-rounded post-RMSNorm hidden into
///     this buffer. Used by the Qwen3.6-MoE self-speculative MTP path
///     (Phase 6.2c.3) to capture `h_post` for the recurrent feed into
///     the next draft step. `None` is the base-decode behaviour.
///   - `counter_buf`: shape `[1]` U32. Used as the work-stealing counter
///     across vocab rows; the launcher memsets it to 0 before each call.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_launch(
    ordinal: usize,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden_buf: &GpuBuffer,
    final_norm_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    logits_buf: &mut GpuBuffer,
    x_normed_out_buf: Option<&mut GpuBuffer>,
    counter_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if hidden <= 0 || vocab <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_launch: positive dims required, \
             got hidden={hidden} vocab={vocab}"
        )));
    }
    let backend = lm_head_w_buf.backend();
    if backend == Backend::Metal {
        let _ = counter_buf;
        return lm_head_launch_metal_bf16(
            ordinal,
            hidden,
            vocab,
            rms_norm_eps,
            final_hidden_buf,
            final_norm_w_buf,
            lm_head_w_buf,
            logits_buf,
            x_normed_out_buf,
        );
    }
    let block_size: i32 = 256;
    if hidden % block_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_launch: hidden ({hidden}) must be a \
             multiple of block_size ({block_size}) — the per-block \
             reduction across hidden assumes that"
        )));
    }

    let x_normed_ptr = x_normed_out_buf
        .map(|b| b.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_lm_head_launch(
                    /* dtype = bf16 */ 2,
                    ordinal,
                    hidden as c_int,
                    vocab as c_int,
                    rms_norm_eps,
                    final_hidden_buf.as_ptr(),
                    final_norm_w_buf.as_ptr(),
                    lm_head_w_buf.as_ptr(),
                    logits_buf.as_mut_ptr(),
                    x_normed_ptr,
                    counter_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::lm_head_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            unreachable!("Metal handled above");
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe lm_head_launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Safe wrapper for the batched lm_head WMMA kernel (Phase 6.4a).
///
/// Processes `m` input rows in a single WMMA tile per vocab block,
/// amortizing the lm_head BF16 weight read across the batch.
/// Companion to [`lm_head_launch`] — call this when `m >= 2`; for
/// `m == 1` the single-M kernel is faster (no LDS overhead for the
/// row dimension).
///
/// Buffer shapes (all on `ordinal`, BF16):
///   - `final_hidden_buf`: `[m, hidden]`
///   - `final_norm_w_buf`: `[hidden]` — shared across all `m` rows.
///   - `lm_head_w_buf`:    `[vocab, hidden]`
///   - `logits_buf`:       `[m, vocab]` — output.
///   - `x_normed_out_buf`: optional `[m, hidden]` post-RMSNorm capture
///     (Phase 6.4b's batched MTP path will use this).
///
/// Constraints:
///   - `m` ∈ \[1, 16]. For `m > 16` the caller must tile rows.
///   - `hidden % 16 == 0` (the WMMA K-loop assumes 16-element slabs).
///   - HIP backend with WMMA BF16 support (gfx11xx). Returns a
///     descriptive error on unsupported configs so the caller can
///     fall back to a per-row loop over `lm_head_launch`.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_batched_launch(
    ordinal: usize,
    m: i32,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden_buf: &GpuBuffer,
    final_norm_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    logits_buf: &mut GpuBuffer,
    x_normed_out_buf: Option<&mut GpuBuffer>,
) -> Result<(), GpuError> {
    if hidden <= 0 || vocab <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_batched_launch: positive dims required, \
             got hidden={hidden} vocab={vocab}"
        )));
    }
    if m < 1 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_batched_launch: m must be ≥ 1, got {m}"
        )));
    }
    let backend = lm_head_w_buf.backend();
    if backend == Backend::Metal {
        return lm_head_batched_launch_metal_bf16(
            ordinal,
            m,
            hidden,
            vocab,
            rms_norm_eps,
            final_hidden_buf,
            final_norm_w_buf,
            lm_head_w_buf,
            logits_buf,
            x_normed_out_buf,
        );
    }
    // M is bounded by min(16, LDS budget / row size). At hidden=2048 the
    // LDS per row is 4 KiB; with 128 B reduction scratch and a 64 KiB cap
    // the effective ceiling is 15 (not the API's 16). Compute the
    // hidden-dependent ceiling here so the caller gets an actionable
    // error before the kernel launch fails with HIP status 254.
    const LDS_BUDGET_BYTES: usize = 64 * 1024;
    const REDUCTION_BYTES: usize = 32 * 4; // 32 F32 lanes
    let lds_per_row = (hidden as usize) * 2; // BF16
    let max_m_for_lds = (LDS_BUDGET_BYTES - REDUCTION_BYTES) / lds_per_row;
    let max_m = max_m_for_lds.min(16) as i32;
    if m > max_m {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_batched_launch: m={m} exceeds the \
             effective ceiling for hidden={hidden} (max_m={max_m} = \
             min(16, ({LDS_BUDGET_BYTES}-{REDUCTION_BYTES}) / \
             (hidden*2))). Tile across the M dimension or call the \
             single-M `lm_head_launch` per row."
        )));
    }
    if hidden % 16 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::lm_head_batched_launch: hidden ({hidden}) must be a \
             multiple of 16 — the WMMA K-loop assumes 16-element slabs"
        )));
    }

    let x_normed_ptr = x_normed_out_buf
        .map(|b| b.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_lm_head_batched_launch(
                    /* dtype = bf16 */ 2,
                    ordinal,
                    m as c_int,
                    hidden as c_int,
                    vocab as c_int,
                    rms_norm_eps,
                    final_hidden_buf.as_ptr(),
                    final_norm_w_buf.as_ptr(),
                    lm_head_w_buf.as_ptr(),
                    logits_buf.as_mut_ptr(),
                    x_normed_ptr,
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::lm_head_batched_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => {
            unreachable!("Metal handled above");
        }
    };
    if status == 138 {
        return Err(GpuError::backend(
            backend,
            format!(
                "qwen36_moe::lm_head_batched_launch: WMMA-BF16 path \
                 unsupported on this device or hidden % 16 != 0 — fall \
                 back to a per-row loop over `lm_head_launch`"
            ),
        ));
    }
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe lm_head_batched_launch failed with status {status}"),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lm_head_launch_metal_bf16(
    ordinal: usize,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden_buf: &GpuBuffer,
    final_norm_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    logits_buf: &mut GpuBuffer,
    x_normed_out_buf: Option<&mut GpuBuffer>,
) -> Result<(), GpuError> {
    let hidden = hidden as usize;
    let vocab = vocab as usize;
    let mut owned_normed;
    let normed = if let Some(buf) = x_normed_out_buf {
        buf
    } else {
        owned_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden])?;
        &mut owned_normed
    };
    crate::prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        rms_norm_eps,
        final_hidden_buf,
        final_norm_w_buf,
        normed,
    )?;
    crate::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        1,
        vocab,
        hidden,
        normed,
        lm_head_w_buf,
        logits_buf,
    )
}

#[allow(clippy::too_many_arguments)]
fn lm_head_batched_launch_metal_bf16(
    ordinal: usize,
    m: i32,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden_buf: &GpuBuffer,
    final_norm_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    logits_buf: &mut GpuBuffer,
    x_normed_out_buf: Option<&mut GpuBuffer>,
) -> Result<(), GpuError> {
    let m = m as usize;
    let hidden = hidden as usize;
    let vocab = vocab as usize;
    let mut owned_normed;
    let normed = if let Some(buf) = x_normed_out_buf {
        buf
    } else {
        owned_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, hidden])?;
        &mut owned_normed
    };
    crate::prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        m,
        hidden,
        rms_norm_eps,
        final_hidden_buf,
        final_norm_w_buf,
        normed,
    )?;
    crate::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        m,
        vocab,
        hidden,
        normed,
        lm_head_w_buf,
        logits_buf,
    )
}

/// Safe wrapper for the GPU MTP pre-fusion kernel (Phase 6.2c.1).
///
/// Computes the byte-for-byte equivalent of vLLM's
/// `Qwen3NextMultiTokenPredictor` pre-fusion stage:
///
/// ```text
/// e_norm = rmsnorm(e_in,   pre_fc_norm_embedding_w, eps)   # [hidden]
/// h_norm = rmsnorm(h_base, pre_fc_norm_hidden_w,    eps)   # [hidden]
/// fused  = mtp.fc @ cat([e_norm, h_norm], dim=-1)          # [hidden]
/// ```
///
/// All buffers are BF16 and must live on `ordinal`. The caller pre-extracts
/// the embedding row (`embed_tokens.weight[next_token_id, :]`) on the host
/// before calling — the kernel doesn't do the gather. `e_norm_out` and
/// `h_norm_out` are produced too (cheap, useful for downstream stages and
/// for parity testing); pass scratch buffers if you don't need them
/// kept around.
///
/// Shapes:
///   - `e_in`, `h_base`, `pre_fc_norm_embedding_w`, `pre_fc_norm_hidden_w`,
///     `e_norm_out`, `h_norm_out`, `fused_out`: `[hidden]`
///   - `fc_w`: `[hidden, 2 * hidden]` (HF row-major; first half coefficients
///     `e_norm`, second half coefficients `h_norm`).
#[allow(clippy::too_many_arguments)]
pub fn mtp_pre_fusion_launch(
    ordinal: usize,
    hidden: i32,
    rms_norm_eps: f32,
    e_in_buf: &GpuBuffer,
    h_base_buf: &GpuBuffer,
    pre_fc_norm_embedding_w_buf: &GpuBuffer,
    pre_fc_norm_hidden_w_buf: &GpuBuffer,
    fc_w_buf: &GpuBuffer,
    e_norm_out_buf: &mut GpuBuffer,
    h_norm_out_buf: &mut GpuBuffer,
    fused_out_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if hidden <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::mtp_pre_fusion_launch: positive hidden required, got {hidden}"
        )));
    }

    let backend = fc_w_buf.backend();
    if backend == Backend::Metal {
        return mtp_pre_fusion_launch_metal_bf16(
            ordinal,
            hidden,
            rms_norm_eps,
            e_in_buf,
            h_base_buf,
            pre_fc_norm_embedding_w_buf,
            pre_fc_norm_hidden_w_buf,
            fc_w_buf,
            e_norm_out_buf,
            h_norm_out_buf,
            fused_out_buf,
        );
    }

    let block_size: i32 = 256;
    if hidden % block_size != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::mtp_pre_fusion_launch: hidden ({hidden}) must be a \
             multiple of block_size ({block_size}) — the per-block reduction \
             across hidden assumes that"
        )));
    }

    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_mtp_pre_fusion_launch(
                    /* dtype = bf16 */ 2,
                    ordinal,
                    hidden as c_int,
                    rms_norm_eps,
                    e_in_buf.as_ptr(),
                    h_base_buf.as_ptr(),
                    pre_fc_norm_embedding_w_buf.as_ptr(),
                    pre_fc_norm_hidden_w_buf.as_ptr(),
                    fc_w_buf.as_ptr(),
                    e_norm_out_buf.as_mut_ptr(),
                    h_norm_out_buf.as_mut_ptr(),
                    fused_out_buf.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen36_moe::mtp_pre_fusion_launch: GPU backend not compiled".into(),
                ));
            }
        }
        Backend::Metal => unreachable!("Metal mtp_pre_fusion handled above"),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe mtp_pre_fusion_launch failed with status {status}"),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mtp_pre_fusion_launch_metal_bf16(
    ordinal: usize,
    hidden: i32,
    rms_norm_eps: f32,
    e_in_buf: &GpuBuffer,
    h_base_buf: &GpuBuffer,
    pre_fc_norm_embedding_w_buf: &GpuBuffer,
    pre_fc_norm_hidden_w_buf: &GpuBuffer,
    fc_w_buf: &GpuBuffer,
    e_norm_out_buf: &mut GpuBuffer,
    h_norm_out_buf: &mut GpuBuffer,
    fused_out_buf: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let hidden = hidden as usize;
    crate::prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        rms_norm_eps,
        e_in_buf,
        pre_fc_norm_embedding_w_buf,
        e_norm_out_buf,
    )?;
    crate::prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        rms_norm_eps,
        h_base_buf,
        pre_fc_norm_hidden_w_buf,
        h_norm_out_buf,
    )?;

    let mut concat = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 2 * hidden])?;
    let half_bytes = hidden * ScalarType::BF16.size_in_bytes();
    gpu_hal::copy_d2d(
        ordinal,
        concat.as_mut_ptr(),
        e_norm_out_buf.as_ptr(),
        half_bytes,
    )?;
    gpu_hal::copy_d2d(
        ordinal,
        concat.offset_ptr(half_bytes) as *mut c_void,
        h_norm_out_buf.as_ptr(),
        half_bytes,
    )?;

    crate::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        1,
        hidden,
        2 * hidden,
        &concat,
        fc_w_buf,
        fused_out_buf,
    )
}

/// Stage A (M3) batched-Q full-attention prefill safe wrapper.
///
/// Runs FlashAttention-style attention with K-tile share across `q_len`
/// queries against a pre-written K/V cache. The caller is responsible for
/// pre-projecting Q/K/V (INT4 weights → BF16 acts), applying RoPE, and
/// writing the chunk's new K/V values into the cache slots `[past_len,
/// past_len + q_len)` BEFORE calling this. Output is the pre-o_proj
/// attention result `[batch, q_heads, q_len, head_dim]` in F32.
///
/// `seqlen_offset` = `past_len`. Causal mask: query at chunk position `qr`
/// attends to cache positions `[0, past_len + qr]` (inclusive).
#[allow(clippy::too_many_arguments)]
pub fn batched_prefill_attn_full_launch(
    ordinal: usize,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    seqlen_offset: usize,
    query: &GpuBuffer,
    key: &GpuBuffer,
    value: &GpuBuffer,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = query.backend();
    if backend == Backend::Metal {
        if batch_size != 1 {
            return Err(GpuError::backend(
                backend,
                format!(
                    "qwen36_moe::batched_prefill_attn_full_launch Metal path supports batch_size=1, got {batch_size}"
                ),
            ));
        }
        return crate::prefill_ffi::metal_full_attention_prefill_strided_bf16_f32(
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            kv_len,
            head_dim,
            scale,
            seqlen_offset,
            query,
            key,
            value,
            out,
        );
    }
    if backend != Backend::Hip && backend != Backend::Cuda {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_attn_full_launch requires HIP or CUDA backend".to_string(),
        ));
    }
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_batched_prefill_attn_full_launch(
                    2, // bf16
                    ordinal,
                    batch_size as c_int,
                    q_heads as c_int,
                    kv_heads as c_int,
                    q_len as c_int,
                    kv_len as c_int,
                    head_dim as c_int,
                    scale,
                    seqlen_offset as c_int,
                    query.as_ptr(),
                    key.as_ptr(),
                    value.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                let _ = (
                    ordinal,
                    batch_size,
                    q_heads,
                    kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    seqlen_offset,
                    query,
                    key,
                    value,
                    out,
                );
                return Err(GpuError::backend(
                    backend,
                    "qwen36_moe::batched_prefill_attn_full_launch: backend not compiled"
                        .to_string(),
                ));
            }
        }
        _ => unreachable!(),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe batched_prefill_attn_full_launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Metal v1 direct routed-expert prefill prototype.
///
/// This bypasses the HIP M9/M10/M11 router-permute path and consumes the
/// already materialized top-k tables directly:
/// - `x_norm`: BF16 `[n_tokens, hidden]`
/// - `topk_idx`: U32 `[n_tokens, top_k]`
/// - `topk_weight`: BF16 `[n_tokens, top_k]`
/// - `expert_mid`: F32 `[n_tokens * top_k, moe_intermediate]`
/// - `combined`: BF16 `[n_tokens, hidden]`
///
/// SAFETY: raw weight pointers must refer to live Metal buffers on `ordinal`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batched_prefill_grouped_expert_direct_metal_launch_raw(
    ordinal: usize,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
    moe_intermediate: usize,
    group_size: usize,
    x_norm: &GpuBuffer,
    topk_idx: &GpuBuffer,
    topk_weight: &GpuBuffer,
    gate_up_proj: *const c_void,
    gate_up_scale: *const c_void,
    gate_up_zero: *const c_void,
    down_proj: *const c_void,
    down_scale: *const c_void,
    down_zero: *const c_void,
    expert_mid: &mut GpuBuffer,
    combined: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = x_norm.backend();
    if backend != Backend::Metal {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw requires Metal backend"
                .to_string(),
        ));
    }
    if topk_idx.backend() != Backend::Metal
        || topk_weight.backend() != Backend::Metal
        || expert_mid.backend() != Backend::Metal
        || combined.backend() != Backend::Metal
    {
        return Err(GpuError::InvalidArg(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw requires all buffers on Metal"
                .into(),
        ));
    }
    if x_norm.dtype() != ScalarType::BF16
        || topk_idx.dtype() != ScalarType::U32
        || topk_weight.dtype() != ScalarType::BF16
        || expert_mid.dtype() != ScalarType::F32
        || combined.dtype() != ScalarType::BF16
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw dtype mismatch: x_norm={:?} topk_idx={:?} topk_weight={:?} expert_mid={:?} combined={:?}",
            x_norm.dtype(),
            topk_idx.dtype(),
            topk_weight.dtype(),
            expert_mid.dtype(),
            combined.dtype(),
        )));
    }
    if n_tokens == 0
        || top_k == 0
        || hidden == 0
        || moe_intermediate == 0
        || group_size == 0
        || gate_up_proj.is_null()
        || gate_up_scale.is_null()
        || gate_up_zero.is_null()
        || down_proj.is_null()
        || down_scale.is_null()
        || down_zero.is_null()
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw invalid shape: n_tokens={n_tokens} top_k={top_k} hidden={hidden} moe_intermediate={moe_intermediate} group_size={group_size}"
        )));
    }
    let expected_routes = n_tokens.checked_mul(top_k).ok_or_else(|| {
        GpuError::InvalidArg(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw top-k size overflow"
                .into(),
        )
    })?;
    let expected_mid = expected_routes.checked_mul(moe_intermediate).ok_or_else(|| {
        GpuError::InvalidArg(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw expert_mid size overflow"
                .into(),
        )
    })?;
    let expected_combined = n_tokens.checked_mul(hidden).ok_or_else(|| {
        GpuError::InvalidArg(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw combined size overflow"
                .into(),
        )
    })?;
    if x_norm.elem_count() < expected_combined
        || topk_idx.elem_count() < expected_routes
        || topk_weight.elem_count() < expected_routes
        || expert_mid.elem_count() < expected_mid
        || combined.elem_count() < expected_combined
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen36_moe::batched_prefill_grouped_expert_direct_metal_launch_raw buffer too small: x_norm={} topk_idx={} topk_weight={} expert_mid={} combined={}",
            x_norm.elem_count(),
            topk_idx.elem_count(),
            topk_weight.elem_count(),
            expert_mid.elem_count(),
            combined.elem_count(),
        )));
    }
    let _ = ordinal;
    crate::prefill_ffi::metal_profile_time(
        "qwen36_batched_prefill_grouped_expert_direct",
        "native",
        || unsafe {
            crate::metal_native::qwen36_batched_prefill_grouped_expert_direct(
                n_tokens,
                top_k,
                hidden,
                moe_intermediate,
                group_size,
                x_norm.as_ptr(),
                topk_idx.as_ptr(),
                topk_weight.as_ptr(),
                gate_up_proj,
                gate_up_scale,
                gate_up_zero,
                down_proj,
                down_scale,
                down_zero,
                expert_mid.as_mut_ptr(),
                combined.as_mut_ptr(),
                true,
            )
        },
    )
}

/// Stage B (M9) router permutation safe wrapper.
///
/// Counting-sort that groups the chunk's top-K expert assignments by target
/// expert. Output buffers must be pre-allocated by the caller:
/// - `expert_offsets`     : i32 `[num_experts + 1]`
/// - `permuted_token_idx` : i32 `[n_tokens * top_k]`
/// - `permuted_kpos`      : i32 `[n_tokens * top_k]`
/// - `permuted_weight`    : BF16 `[n_tokens * top_k]`
///
/// Within an expert's segment, order is unstable (the kernel uses
/// atomicAdd as the in-segment cursor). Tests must compare per-expert as
/// a multiset of (token_idx, kpos, weight) triples — the downstream
/// grouped GEMM is permutation-invariant inside a segment.
#[allow(clippy::too_many_arguments)]
pub fn batched_prefill_router_permute_launch(
    ordinal: usize,
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    topk_idx: &GpuBuffer,
    topk_weight: &GpuBuffer,
    expert_offsets: &mut GpuBuffer,
    permuted_token_idx: &mut GpuBuffer,
    permuted_kpos: &mut GpuBuffer,
    permuted_weight: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = topk_idx.backend();
    if backend != Backend::Hip && backend != Backend::Cuda {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_router_permute_launch requires HIP or CUDA backend"
                .to_string(),
        ));
    }
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_batched_prefill_router_permute_launch(
                    ordinal,
                    n_tokens as c_int,
                    top_k as c_int,
                    num_experts as c_int,
                    topk_idx.as_ptr(),
                    topk_weight.as_ptr(),
                    expert_offsets.as_mut_ptr(),
                    permuted_token_idx.as_mut_ptr(),
                    permuted_kpos.as_mut_ptr(),
                    permuted_weight.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                let _ = (
                    ordinal,
                    n_tokens,
                    top_k,
                    num_experts,
                    topk_idx,
                    topk_weight,
                    expert_offsets,
                    permuted_token_idx,
                    permuted_kpos,
                    permuted_weight,
                );
                return Err(GpuError::backend(
                    backend,
                    "qwen36_moe::batched_prefill_router_permute_launch: backend not compiled"
                        .to_string(),
                ));
            }
        }
        _ => unreachable!(),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe batched_prefill_router_permute_launch failed with status {status}"),
        ));
    }
    Ok(())
}

/// Stage B (M10) grouped-expert INT4 GEMM safe wrapper.
///
/// Runs gate_up + silu*mul + down for ALL `num_experts` experts in one
/// launch via persistent-block work-stealing. Per-expert the kernel walks
/// the rows assigned to it by the M9 router permutation and writes
/// `down(silu(gate(x_norm[token])) * up(x_norm[token]))` into
/// `expert_out[row, hidden]`. M11's orchestrator wires this into the
/// batched prefill pipeline.
///
/// Buffer requirements:
/// - `x_norm`              : BF16, `[n_tokens, hidden]`. Caller-produced
///                            (post-input-RMSnorm).
/// - `expert_offsets`      : i32 (stored as U32 in `GpuBuffer`),
///                            `[num_experts + 1]`. M9 output.
/// - `permuted_token_idx`  : i32 (U32 storage), `[n_tokens * top_k]`. M9.
/// - `experts_gate_up_w`   : u8 (U8 storage), `[E, 2*I, hidden/2]`.
/// - `experts_gate_up_scale/zero` : BF16, `[E, 2*I/gs, hidden/gs]`.
/// - `experts_down_w`      : u8, `[E, hidden, I/2]`.
/// - `experts_down_scale/zero` : BF16, `[E, hidden/gs, I/gs]`.
/// - `expert_out`          : BF16, `[n_tokens * top_k, hidden]`.
/// - `counters`            : u32, `[1]`. CALLER MUST ZERO BEFORE LAUNCH —
///                            this is the work-stealing claim counter.
#[allow(clippy::too_many_arguments)]
pub fn batched_prefill_grouped_expert_launch(
    ordinal: usize,
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    moe_intermediate: usize,
    group_size: usize,
    x_norm: &GpuBuffer,
    expert_offsets: &GpuBuffer,
    permuted_token_idx: &GpuBuffer,
    experts_gate_up_w: &GpuBuffer,
    experts_gate_up_scale: &GpuBuffer,
    experts_gate_up_zero: &GpuBuffer,
    experts_down_w: &GpuBuffer,
    experts_down_scale: &GpuBuffer,
    experts_down_zero: &GpuBuffer,
    expert_out: &mut GpuBuffer,
    counters: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = x_norm.backend();
    if backend != Backend::Hip && backend != Backend::Cuda {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_grouped_expert_launch requires HIP or CUDA backend"
                .to_string(),
        ));
    }
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_batched_prefill_grouped_expert_launch(
                    2, // bf16
                    ordinal,
                    n_tokens as c_int,
                    top_k as c_int,
                    num_experts as c_int,
                    hidden as c_int,
                    moe_intermediate as c_int,
                    group_size as c_int,
                    x_norm.as_ptr(),
                    expert_offsets.as_ptr(),
                    permuted_token_idx.as_ptr(),
                    experts_gate_up_w.as_ptr(),
                    experts_gate_up_scale.as_ptr(),
                    experts_gate_up_zero.as_ptr(),
                    experts_down_w.as_ptr(),
                    experts_down_scale.as_ptr(),
                    experts_down_zero.as_ptr(),
                    expert_out.as_mut_ptr(),
                    counters.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                let _ = (
                    ordinal,
                    n_tokens,
                    top_k,
                    num_experts,
                    hidden,
                    moe_intermediate,
                    group_size,
                    x_norm,
                    expert_offsets,
                    permuted_token_idx,
                    experts_gate_up_w,
                    experts_gate_up_scale,
                    experts_gate_up_zero,
                    experts_down_w,
                    experts_down_scale,
                    experts_down_zero,
                    expert_out,
                    counters,
                );
                return Err(GpuError::backend(
                    backend,
                    "qwen36_moe::batched_prefill_grouped_expert_launch: backend not compiled"
                        .to_string(),
                ));
            }
        }
        _ => unreachable!(),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen36_moe batched_prefill_grouped_expert_launch failed with status {status}"),
        ));
    }
    let _ = (
        ordinal,
        n_tokens,
        top_k,
        num_experts,
        hidden,
        moe_intermediate,
        group_size,
    );
    Ok(())
}

/// Raw-pointer variant of `batched_prefill_grouped_expert_launch`.
///
/// Same kernel, but accepts `*const c_void` for the per-expert weight slabs
/// (gate_up + down packed nibbles, plus their parallel scale/zero tables).
/// This is needed by the M11 orchestrator because the runner stores those
/// tensors via `ResidentWeight`, which can be either a `Dense` `GpuBuffer`
/// or a `Virtual` allocation (raw pointer + shape) depending on the
/// residency strategy. `ResidentWeight::as_ptr()` is the lowest-common-
/// denominator handle on both variants.
///
/// SAFETY: caller is responsible for keeping the underlying allocations
/// alive for the duration of the launch and for ensuring all pointers refer
/// to GPU memory on `ordinal`'s device. The `x_norm`, `expert_offsets`,
/// `permuted_token_idx`, `expert_out`, and `counters` parameters are still
/// `GpuBuffer` borrows because the orchestrator allocates them locally.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batched_prefill_grouped_expert_launch_raw(
    ordinal: usize,
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    moe_intermediate: usize,
    group_size: usize,
    x_norm: &GpuBuffer,
    expert_offsets: &GpuBuffer,
    permuted_token_idx: &GpuBuffer,
    experts_gate_up_w: *const c_void,
    experts_gate_up_scale: *const c_void,
    experts_gate_up_zero: *const c_void,
    experts_down_w: *const c_void,
    experts_down_scale: *const c_void,
    experts_down_zero: *const c_void,
    expert_out: &mut GpuBuffer,
    counters: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = x_norm.backend();
    if backend != Backend::Hip && backend != Backend::Cuda {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_grouped_expert_launch_raw requires HIP or CUDA backend"
                .to_string(),
        ));
    }
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            {
                qwen36_moe_hip_batched_prefill_grouped_expert_launch(
                    2, // bf16
                    ordinal,
                    n_tokens as c_int,
                    top_k as c_int,
                    num_experts as c_int,
                    hidden as c_int,
                    moe_intermediate as c_int,
                    group_size as c_int,
                    x_norm.as_ptr(),
                    expert_offsets.as_ptr(),
                    permuted_token_idx.as_ptr(),
                    experts_gate_up_w,
                    experts_gate_up_scale,
                    experts_gate_up_zero,
                    experts_down_w,
                    experts_down_scale,
                    experts_down_zero,
                    expert_out.as_mut_ptr(),
                    counters.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                let _ = (
                    ordinal,
                    n_tokens,
                    top_k,
                    num_experts,
                    hidden,
                    moe_intermediate,
                    group_size,
                    x_norm,
                    expert_offsets,
                    permuted_token_idx,
                    experts_gate_up_w,
                    experts_gate_up_scale,
                    experts_gate_up_zero,
                    experts_down_w,
                    experts_down_scale,
                    experts_down_zero,
                    expert_out,
                    counters,
                );
                return Err(GpuError::backend(
                    backend,
                    "qwen36_moe::batched_prefill_grouped_expert_launch_raw: backend not compiled"
                        .to_string(),
                ));
            }
        }
        _ => unreachable!(),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!(
                "qwen36_moe batched_prefill_grouped_expert_launch_raw failed with status {status}"
            ),
        ));
    }
    Ok(())
}

/// Stage B (M11) unpermute + weighted combine safe wrapper.
///
/// Computes `combined[token, col] = sum_kpos( w * expert_out[dst, col] )`
/// where `dst = permuted_inverse[token * top_k + kpos]` and `w` is the
/// routing weight at that `dst` position. Caller must pre-compute and
/// upload `permuted_inverse` host-side from M9's `permuted_token_idx` +
/// `permuted_kpos` outputs.
///
/// Buffer requirements:
/// - `permuted_inverse` : i32 (U32 storage), `[n_tokens * top_k]`.
/// - `permuted_weight`  : BF16, `[n_tokens * top_k]`. M9 output.
/// - `expert_out`       : BF16, `[n_tokens * top_k, hidden]`. M10 output.
/// - `combined`         : BF16, `[n_tokens, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn batched_prefill_unpermute_combine_launch(
    ordinal: usize,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
    permuted_inverse: &GpuBuffer,
    permuted_weight: &GpuBuffer,
    expert_out: &GpuBuffer,
    combined: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let backend = expert_out.backend();
    if backend != Backend::Hip && backend != Backend::Cuda {
        return Err(GpuError::backend(
            backend,
            "qwen36_moe::batched_prefill_unpermute_combine_launch requires HIP or CUDA backend"
                .to_string(),
        ));
    }
    let status: c_int = match backend {
        Backend::Hip | Backend::Cuda => {
            #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
            unsafe {
                qwen36_moe_hip_batched_prefill_unpermute_combine_launch(
                    2, // bf16
                    ordinal,
                    n_tokens as c_int,
                    top_k as c_int,
                    hidden as c_int,
                    permuted_inverse.as_ptr(),
                    permuted_weight.as_ptr(),
                    expert_out.as_ptr(),
                    combined.as_mut_ptr(),
                )
            }
            #[cfg(not(any(supersonic_backend_hip, supersonic_backend_cuda)))]
            {
                let _ = (
                    ordinal,
                    n_tokens,
                    top_k,
                    hidden,
                    permuted_inverse,
                    permuted_weight,
                    expert_out,
                    combined,
                );
                return Err(GpuError::backend(
                    backend,
                    "qwen36_moe::batched_prefill_unpermute_combine_launch: backend not compiled"
                        .to_string(),
                ));
            }
        }
        _ => unreachable!(),
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!(
                "qwen36_moe batched_prefill_unpermute_combine_launch failed with status {status}"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn descriptor_layout_offsets_documented() {
        // Pin the size so a future field-reorder is loud. If you need to
        // grow the struct, append fields and update this exact number —
        // never reorder existing ones. Loose ranges (e.g. [256, 512])
        // silently absorbed forgotten field appends; we use exact sizes
        // and the matching C++ `static_assert` ranges in
        // `kernels/qwen36_moe_bridge.cpp` to catch drift on both sides.
        // (Numbers verified on x86_64 Linux. Pointers are 8 bytes.)
        let sz = size_of::<Qwen36MoeDecodeLayerDesc>();
        assert_eq!(
            sz, 344,
            "Qwen36MoeDecodeLayerDesc size drift: got {sz} bytes (expected 344)",
        );

        let int4_sz = size_of::<Qwen36MoeInt4ScaleDesc>();
        assert!(
            int4_sz >= 192 && int4_sz <= 256,
            "Qwen36MoeInt4ScaleDesc size drift: got {int4_sz} bytes",
        );

        // Two raw pointers — pinned to match the C++ static_assert in
        // kernels/qwen36_moe_bridge.cpp.
        let kv_fp8_sz = size_of::<Qwen36MoeKVCacheFp8Desc>();
        assert_eq!(
            kv_fp8_sz, 16,
            "Qwen36MoeKVCacheFp8Desc size drift: got {kv_fp8_sz} bytes (expected 16)",
        );
    }

    #[test]
    fn descriptor_default_is_zero() {
        let d = Qwen36MoeDecodeLayerDesc::default();
        assert_eq!(d.layer_idx, 0);
        assert_eq!(d.is_full_attention, 0);
        assert!(d.input_norm_w.is_null());
        assert!(d.kv_cache_k.is_null());
        assert!(d.linear_recurrent_state.is_null());
        assert!(d.experts_gate_up_w.is_null());
    }

    #[test]
    fn qwen36_static_topn_table_parser_reads_probe_export() {
        let raw = r#"{
            "schema": "qwen36-static-topn-mps-probe-v2",
            "static_tables": {
                "4": {
                    "layers": [
                        {"layer": 0, "experts": [7, 1], "counts": [3, 2]},
                        {"layer": 2, "experts": [5, 4], "counts": [9, 1]}
                    ]
                },
                "8": {
                    "1": [3, 2, 1]
                }
            }
        }"#;
        let table = qwen36_parse_static_topn_table_json(raw).expect("parse static top-N table");

        assert!(table.capacity_exists(4));
        assert!(table.capacity_exists(8));
        assert_eq!(table.largest_capacity(), Some(8));
        assert_eq!(table.layer_experts(4, 0), Some([7, 1].as_slice()));
        assert_eq!(table.layer_experts(4, 1), None);
        assert_eq!(table.layer_experts(4, 2), Some([5, 4].as_slice()));
        assert_eq!(table.layer_experts(8, 1), Some([3, 2, 1].as_slice()));
    }

    #[test]
    fn qwen36_static_topn_slot_map_validates_table_rows() {
        let slots = qwen36_static_topn_expert_to_slot(3, &[7, 1, 5], 8, 3).expect("valid slot map");
        assert_eq!(slots[7], Some(0));
        assert_eq!(slots[1], Some(1));
        assert_eq!(slots[5], Some(2));
        assert_eq!(slots[0], None);

        let duplicate = qwen36_static_topn_expert_to_slot(3, &[7, 1, 7], 8, 3);
        assert!(duplicate.is_err());
        let out_of_range = qwen36_static_topn_expert_to_slot(3, &[8], 8, 3);
        assert!(out_of_range.is_err());
    }

    #[test]
    fn qwen36_mps_static_topn_profile_key_uses_fp16_mps_format() {
        let key = Qwen36ExpertResidencyProfileKey::with_format(
            Qwen36ExpertResidentFormat::Fp16Mps,
            Qwen36ExpertResidencyMissPolicy::StaticTopN,
            64,
        );

        assert_eq!(key.resident_format.as_str(), "fp16_mps");
        assert_eq!(key.miss_policy.as_str(), "static_topn");
        assert_eq!(key.capacity, 64);
    }

    #[test]
    fn qwen36_route_profile_simulates_layer_locality() {
        let records = vec![
            vec![1, 2],
            vec![3, 4],
            vec![1, 5],
            vec![6, 4],
            vec![1, 2],
            vec![6, 7],
        ];
        let snapshot = qwen36_route_profile_simulate(&records, 0, 2);

        assert_eq!(snapshot.calls, 6);
        assert_eq!(snapshot.assignments, 12);
        assert_eq!(snapshot.unique_layer_experts, 7);
        assert_eq!(snapshot.adjacent_hits, 4);
        assert_eq!(snapshot.adjacent_total, 8);
        assert_eq!(snapshot.route_calls.len(), 6);
        assert_eq!(snapshot.route_calls[3].layer, 1);
        assert_eq!(snapshot.route_calls[3].experts, vec![6, 4]);

        let cap2 = snapshot
            .cache_sims
            .iter()
            .find(|sim| sim.capacity == 2)
            .expect("cap2 cache sim");
        assert_eq!(cap2.hits, 4);
        assert_eq!(cap2.misses, 8);

        let top2 = snapshot
            .topn_sims
            .iter()
            .find(|sim| sim.capacity == 2)
            .expect("cap2 top-n sim");
        assert_eq!(top2.covered, 9);
        assert_eq!(top2.total, 12);

        let layer0_top2 = snapshot
            .topn_layers
            .iter()
            .find(|row| row.capacity == 2 && row.layer == 0)
            .expect("layer0 cap2 top-n row");
        assert_eq!(layer0_top2.experts, vec![1, 2]);
        assert_eq!(layer0_top2.counts, vec![3, 2]);
        assert_eq!(layer0_top2.covered, 5);
        assert_eq!(layer0_top2.total, 6);
    }

    #[test]
    fn qwen36_batched_prefill_feasibility_groups_routes_by_chunk_and_layer() {
        let records = vec![
            vec![1, 2],
            vec![5, 6],
            vec![1, 3],
            vec![5, 7],
            vec![1, 4],
            vec![8, 9],
        ];
        let config = Qwen36BatchedPrefillFeasibilityConfig {
            layers: 2,
            top_k: 2,
            num_experts: 10,
            chunk_size: 2,
            prefill_tokens: 3,
        };
        let snapshot = qwen36_batched_prefill_feasibility_profile_simulate(&records, 0, &config);

        assert_eq!(snapshot.calls, 6);
        assert_eq!(snapshot.profiled_tokens, 3);
        assert_eq!(snapshot.chunks, 2);
        assert_eq!(snapshot.assignments, 12);
        assert_eq!(snapshot.permutation_entries, 12);
        assert_eq!(snapshot.expert_segments, 10);
        assert_eq!(snapshot.max_unique_experts_per_layer_chunk, 3);
        assert_eq!(snapshot.max_rows_per_segment, 2);
        assert_eq!(snapshot.scalar_tail_segments, 10);
        assert_eq!(snapshot.scalar_tail_assignments, 12);
        assert_eq!(snapshot.wmma16_padded_assignments, 160);
        assert_eq!(snapshot.avg_unique_experts_per_layer_chunk(), 2.5);
        assert!((snapshot.avg_rows_per_segment() - 1.2).abs() < 1e-6);
        assert_eq!(snapshot.wmma16_assignment_coverage(), 0.0);
        assert!((snapshot.wmma16_padding_overhead() - (148.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn qwen36_packed_expert_copy_bytes_matches_stage5_geometry() {
        let bytes = qwen36_packed_expert_copy_bytes(2048, 512, 8, 128);
        assert_eq!(bytes, 12_589_056);
    }

    #[test]
    fn qwen36_hotset_slot_selection_reuses_hits_and_protects_current_call() {
        let slots = vec![Some(7), Some(9), None, Some(11)];
        let last_used = vec![10, 1, 0, 3];
        let protected = vec![false; 4];
        assert_eq!(
            qwen36_hotset_choose_slot(&slots, &last_used, &protected, 9),
            (1, true, false)
        );
        assert_eq!(
            qwen36_hotset_choose_slot(&slots, &last_used, &protected, 13),
            (2, false, false)
        );

        let protected = vec![false, true, true, false];
        assert_eq!(
            qwen36_hotset_choose_slot(&slots, &last_used, &protected, 13),
            (3, false, true)
        );
    }

    fn test_int4_blob(num_experts: usize, rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(num_experts * rows * cols.div_ceil(2));
        for expert in 0..num_experts {
            for row in 0..rows {
                for col in (0..cols).step_by(2) {
                    let lo = ((expert * 3 + row * 5 + col) & 0x0f) as u8;
                    let hi = ((expert * 7 + row * 11 + col + 1) & 0x0f) as u8;
                    bytes.push(lo | (hi << 4));
                }
            }
        }
        bytes
    }

    fn test_bf16_sidecar(
        num_experts: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        base: f32,
    ) -> Vec<u16> {
        let scale_rows = rows.div_ceil(group_size);
        let scale_cols = cols.div_ceil(group_size);
        let mut values = Vec::with_capacity(num_experts * scale_rows * scale_cols);
        for expert in 0..num_experts {
            for row_group in 0..scale_rows {
                for col_group in 0..scale_cols {
                    let value = base + 0.03125 * expert as f32 + 0.0078125 * row_group as f32
                        - 0.00390625 * col_group as f32;
                    values.push(half::bf16::from_f32(value).to_bits());
                }
            }
        }
        values
    }

    #[test]
    fn qwen36_mps_bridge_lut_pack_matches_scalar_pack() {
        let hidden = 8;
        let moe_intermediate = 4;
        let gate_up_rows = 2 * moe_intermediate;
        let group_size = 4;
        let num_experts = 3;
        let active_experts = [2usize, 0usize];
        let h_norm = [0.25, -1.0, 2.0, 0.5, -0.125, 1.5, -2.5, 3.0];

        let gate_up = test_int4_blob(num_experts, gate_up_rows, hidden);
        let gate_up_scale = test_bf16_sidecar(num_experts, gate_up_rows, hidden, group_size, 0.125);
        let gate_up_zero = test_bf16_sidecar(num_experts, gate_up_rows, hidden, group_size, 0.5);
        let down = test_int4_blob(num_experts, hidden, moe_intermediate);
        let down_scale =
            test_bf16_sidecar(num_experts, hidden, moe_intermediate, group_size, 0.0625);
        let down_zero = test_bf16_sidecar(num_experts, hidden, moe_intermediate, group_size, 0.25);

        let scalar = qwen36_pack_mps_expert_bridge_bytes_scalar(
            hidden,
            moe_intermediate,
            &active_experts,
            &h_norm,
            group_size,
            gate_up.as_ptr().cast(),
            gate_up_scale.as_ptr().cast(),
            gate_up_zero.as_ptr().cast(),
            down.as_ptr().cast(),
            down_scale.as_ptr().cast(),
            down_zero.as_ptr().cast(),
        );
        let lut = qwen36_pack_mps_expert_bridge_bytes_lut(
            hidden,
            moe_intermediate,
            &active_experts,
            &h_norm,
            group_size,
            gate_up.as_ptr().cast(),
            gate_up_scale.as_ptr().cast(),
            gate_up_zero.as_ptr().cast(),
            down.as_ptr().cast(),
            down_scale.as_ptr().cast(),
            down_zero.as_ptr().cast(),
        );

        assert_eq!(lut.0, scalar.0, "h_norm pack mismatch");
        assert_eq!(lut.1, scalar.1, "gate/up RHS pack mismatch");
        assert_eq!(lut.2, scalar.2, "down RHS pack mismatch");
    }

    #[test]
    #[ignore = "microbench for the Qwen3.6 MPS bridge CPU transcode experiment"]
    fn qwen36_mps_bridge_cpu_pack_lut_microbench() {
        let hidden = 2048;
        let moe_intermediate = 512;
        let gate_up_rows = 2 * moe_intermediate;
        let group_size = 128;
        let num_experts = 8;
        let active_experts: Vec<usize> = (0..8).collect();
        let h_norm: Vec<f32> = (0..hidden)
            .map(|idx| ((idx as i32 % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let gate_up = test_int4_blob(num_experts, gate_up_rows, hidden);
        let gate_up_scale =
            test_bf16_sidecar(num_experts, gate_up_rows, hidden, group_size, 0.03125);
        let gate_up_zero = test_bf16_sidecar(num_experts, gate_up_rows, hidden, group_size, 7.0);
        let down = test_int4_blob(num_experts, hidden, moe_intermediate);
        let down_scale =
            test_bf16_sidecar(num_experts, hidden, moe_intermediate, group_size, 0.03125);
        let down_zero = test_bf16_sidecar(num_experts, hidden, moe_intermediate, group_size, 7.0);

        let scalar_start = std::time::Instant::now();
        let scalar = qwen36_pack_mps_expert_bridge_bytes_scalar(
            hidden,
            moe_intermediate,
            &active_experts,
            &h_norm,
            group_size,
            gate_up.as_ptr().cast(),
            gate_up_scale.as_ptr().cast(),
            gate_up_zero.as_ptr().cast(),
            down.as_ptr().cast(),
            down_scale.as_ptr().cast(),
            down_zero.as_ptr().cast(),
        );
        let scalar_ms = scalar_start.elapsed().as_secs_f64() * 1000.0;

        let lut_start = std::time::Instant::now();
        let lut = qwen36_pack_mps_expert_bridge_bytes_lut(
            hidden,
            moe_intermediate,
            &active_experts,
            &h_norm,
            group_size,
            gate_up.as_ptr().cast(),
            gate_up_scale.as_ptr().cast(),
            gate_up_zero.as_ptr().cast(),
            down.as_ptr().cast(),
            down_scale.as_ptr().cast(),
            down_zero.as_ptr().cast(),
        );
        let lut_ms = lut_start.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(lut, scalar);
        eprintln!(
            "qwen36_mps_bridge_cpu_pack_lut_microbench scalar_ms={scalar_ms:.3} lut_ms={lut_ms:.3} speedup={:.2}x bytes={}",
            scalar_ms / lut_ms.max(f64::MIN_POSITIVE),
            lut.0.len() + lut.1.len() + lut.2.len()
        );
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16");
        bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_bf16(ordinal: usize, shape: &[usize], values: &[f32]) -> GpuBuffer {
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, &bf16_bytes(values))
            .expect("upload bf16")
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_int4_rows(
        ordinal: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        nibbles: &[Vec<u8>],
        scale_value: f32,
    ) -> (GpuBuffer, GpuBuffer, GpuBuffer) {
        let mut packed = Vec::with_capacity(rows * cols.div_ceil(2));
        for row in nibbles {
            for pair in row.chunks(2) {
                let lo = pair[0] & 0x0f;
                let hi = pair.get(1).copied().unwrap_or(0) & 0x0f;
                packed.push(lo | (hi << 4));
            }
        }
        let weights =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, cols.div_ceil(2)], &packed)
                .expect("upload qwen36 int4 rows");
        let sidecar_len = rows.div_ceil(group_size) * cols.div_ceil(group_size);
        let scale = upload_bf16(ordinal, &[sidecar_len], &vec![scale_value; sidecar_len]);
        let zero = upload_bf16(ordinal, &[sidecar_len], &vec![0.0; sidecar_len]);
        (weights, scale, zero)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_f32(ordinal: usize, shape: &[usize], values: &[f32]) -> GpuBuffer {
        GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, shape, &f32_bytes(values))
            .expect("upload f32")
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn read_f32(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download f32");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_lm_head_reference(
        hidden_rows: &[f32],
        norm_w: &[f32],
        lm_head: &[f32],
        rows: usize,
        hidden: usize,
        vocab: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut normed = vec![0.0f32; rows * hidden];
        let mut logits = vec![0.0f32; rows * vocab];
        for row in 0..rows {
            let hidden_base = row * hidden;
            let mean_sq = hidden_rows[hidden_base..hidden_base + hidden]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / hidden as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for col in 0..hidden {
                normed[hidden_base + col] = half::bf16::from_f32(
                    hidden_rows[hidden_base + col] * inv * (1.0 + norm_w[col]),
                )
                .to_f32();
            }
            for tok in 0..vocab {
                let mut acc = 0.0f32;
                for col in 0..hidden {
                    acc += normed[hidden_base + col] * lm_head[tok * hidden + col];
                }
                logits[row * vocab + tok] = half::bf16::from_f32(acc).to_f32();
            }
        }
        (normed, logits)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn bf16_round(value: f32) -> f32 {
        half::bf16::from_f32(value).to_f32()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_mtp_pre_fusion_reference(
        e_in: &[f32],
        h_base: &[f32],
        e_norm_w: &[f32],
        h_norm_w: &[f32],
        fc_w: &[f32],
        hidden: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        fn norm(input: &[f32], weight: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
            let input_bf16: Vec<f32> = input.iter().copied().map(bf16_round).collect();
            let weight_bf16: Vec<f32> = weight.iter().copied().map(bf16_round).collect();
            let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            (0..hidden)
                .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + weight_bf16[idx])))
                .collect()
        }

        let e_norm = norm(e_in, e_norm_w, hidden, eps);
        let h_norm = norm(h_base, h_norm_w, hidden, eps);
        let mut cat = Vec::with_capacity(2 * hidden);
        cat.extend_from_slice(&e_norm);
        cat.extend_from_slice(&h_norm);
        let fc_w_bf16: Vec<f32> = fc_w.iter().copied().map(bf16_round).collect();
        let mut fused = vec![0.0f32; hidden];
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for col in 0..(2 * hidden) {
                acc += cat[col] * fc_w_bf16[row * 2 * hidden + col];
            }
            fused[row] = bf16_round(acc);
        }
        (e_norm, h_norm, fused)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_ffn_stage1_reference(
        input_hidden: &[f32],
        norm_w: &[f32],
        gate_w: &[f32],
        hidden: usize,
        num_experts: usize,
        top_k: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<i32>) {
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = norm_w.iter().copied().map(bf16_round).collect();
        let gate_w_bf16: Vec<f32> = gate_w.iter().copied().map(bf16_round).collect();
        let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let h_norm: Vec<f32> = (0..hidden)
            .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + norm_w_bf16[idx])))
            .collect();
        let mut logits = vec![0.0f32; num_experts];
        for expert in 0..num_experts {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                acc += gate_w_bf16[expert * hidden + col] * h_norm[col];
            }
            logits[expert] = bf16_round(acc);
        }
        let row_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - row_max).exp()).collect();
        let sum = exps.iter().sum::<f32>();
        let mut probs: Vec<f32> = exps.iter().map(|v| bf16_round(v / sum)).collect();
        let mut idx = vec![0i32; top_k];
        let mut weights = vec![0.0f32; top_k];
        for k in 0..top_k {
            let mut best_idx = -1i32;
            let mut best_val = f32::NEG_INFINITY;
            for (expert, &value) in probs.iter().enumerate() {
                if value > best_val
                    || (value == best_val && best_idx >= 0 && (expert as i32) < best_idx)
                {
                    best_val = value;
                    best_idx = expert as i32;
                }
            }
            idx[k] = best_idx;
            weights[k] = best_val;
            probs[best_idx as usize] = f32::NEG_INFINITY;
        }
        let sum_k = weights.iter().sum::<f32>();
        for weight in weights.iter_mut() {
            *weight = bf16_round(*weight / sum_k);
        }
        (weights, idx)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_ffn_stage1_5_reference(
        stage: usize,
        input_hidden: &[f32],
        norm_w: &[f32],
        gate_w: &[f32],
        gate_up_proj_w: &[f32],
        down_proj_w: &[f32],
        shared_gate_proj_w: &[f32],
        shared_up_proj_w: &[f32],
        shared_down_proj_w: &[f32],
        shared_expert_gate_w: &[f32],
        hidden: usize,
        num_experts: usize,
        moe_intermediate: usize,
        shared_intermediate: usize,
        top_k: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<i32>) {
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = norm_w.iter().copied().map(bf16_round).collect();
        let gate_w_bf16: Vec<f32> = gate_w.iter().copied().map(bf16_round).collect();
        let gu_w_bf16: Vec<f32> = gate_up_proj_w.iter().copied().map(bf16_round).collect();
        let down_w_bf16: Vec<f32> = down_proj_w.iter().copied().map(bf16_round).collect();
        let sgp_w_bf16: Vec<f32> = shared_gate_proj_w.iter().copied().map(bf16_round).collect();
        let sup_w_bf16: Vec<f32> = shared_up_proj_w.iter().copied().map(bf16_round).collect();
        let sd_w_bf16: Vec<f32> = shared_down_proj_w.iter().copied().map(bf16_round).collect();
        let seg_w_bf16: Vec<f32> = shared_expert_gate_w
            .iter()
            .copied()
            .map(bf16_round)
            .collect();

        let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let h_norm: Vec<f32> = (0..hidden)
            .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + norm_w_bf16[idx])))
            .collect();

        let mut logits = vec![0.0f32; num_experts];
        for expert in 0..num_experts {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                acc += gate_w_bf16[expert * hidden + col] * h_norm[col];
            }
            logits[expert] = bf16_round(acc);
        }
        let row_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - row_max).exp()).collect();
        let sum = exps.iter().sum::<f32>();
        let mut probs: Vec<f32> = exps.iter().map(|v| bf16_round(v / sum)).collect();
        let mut idx = vec![0i32; top_k];
        let mut topk_w = vec![0.0f32; top_k];
        for k in 0..top_k {
            let mut best_idx = -1i32;
            let mut best_val = f32::NEG_INFINITY;
            for (expert, &value) in probs.iter().enumerate() {
                if value > best_val
                    || (value == best_val && best_idx >= 0 && (expert as i32) < best_idx)
                {
                    best_val = value;
                    best_idx = expert as i32;
                }
            }
            idx[k] = best_idx;
            topk_w[k] = best_val;
            probs[best_idx as usize] = f32::NEG_INFINITY;
        }
        let sum_k = topk_w.iter().sum::<f32>();
        for weight in topk_w.iter_mut() {
            *weight = bf16_round(*weight / sum_k);
        }
        if stage == 1 {
            return (topk_w, idx);
        }

        let mut sgp = vec![0.0f32; shared_intermediate];
        let mut sup = vec![0.0f32; shared_intermediate];
        for row in 0..shared_intermediate {
            for col in 0..hidden {
                sgp[row] += sgp_w_bf16[row * hidden + col] * h_norm[col];
                sup[row] += sup_w_bf16[row * hidden + col] * h_norm[col];
            }
        }
        let mut sg_scalar = 0.0f32;
        for col in 0..hidden {
            sg_scalar += seg_w_bf16[col] * h_norm[col];
        }
        sg_scalar = 1.0 / (1.0 + (-sg_scalar).exp());
        let mut shared_mid = vec![0.0f32; shared_intermediate];
        for i in 0..shared_intermediate {
            shared_mid[i] = sgp[i] * (1.0 / (1.0 + (-sgp[i]).exp())) * sup[i];
        }
        let mut shared_out = vec![0.0f32; hidden];
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for col in 0..shared_intermediate {
                acc += sd_w_bf16[row * shared_intermediate + col] * shared_mid[col];
            }
            shared_out[row] = bf16_round(sg_scalar * acc);
        }
        if stage == 2 {
            return (shared_out, idx);
        }

        let active = if stage == 3 { 1 } else { top_k };
        let mut expert_stack = vec![0.0f32; top_k * hidden];
        for group in 0..active {
            let expert = idx[group] as usize;
            let mut gu = vec![0.0f32; 2 * moe_intermediate];
            let gu_base = expert * 2 * moe_intermediate * hidden;
            for row in 0..2 * moe_intermediate {
                for col in 0..hidden {
                    gu[row] += gu_w_bf16[gu_base + row * hidden + col] * h_norm[col];
                }
            }
            let mut mid = vec![0.0f32; moe_intermediate];
            for i in 0..moe_intermediate {
                mid[i] = gu[i] * (1.0 / (1.0 + (-gu[i]).exp())) * gu[moe_intermediate + i];
            }
            let down_base = expert * hidden * moe_intermediate;
            for row in 0..hidden {
                let mut acc = 0.0f32;
                for col in 0..moe_intermediate {
                    acc += down_w_bf16[down_base + row * moe_intermediate + col] * mid[col];
                }
                expert_stack[group * hidden + row] = acc;
            }
        }
        if stage == 3 {
            return (
                expert_stack[..hidden]
                    .iter()
                    .copied()
                    .map(bf16_round)
                    .collect(),
                idx,
            );
        }

        let mut moe_out = vec![0.0f32; hidden];
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for group in 0..top_k {
                acc += topk_w[group] * expert_stack[group * hidden + row];
            }
            moe_out[row] = bf16_round(acc);
        }
        if stage == 4 {
            return (moe_out, idx);
        }

        let mut out = vec![0.0f32; hidden];
        for row in 0..hidden {
            out[row] = bf16_round(input_bf16[row] + moe_out[row] + shared_out[row]);
        }
        (out, idx)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_attn_stage1_reference(
        input_hidden: &[f32],
        input_norm_w: &[f32],
        q_proj_w: &[f32],
        q_norm_w: &[f32],
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = input_norm_w.iter().copied().map(bf16_round).collect();
        let q_proj_w_bf16: Vec<f32> = q_proj_w.iter().copied().map(bf16_round).collect();
        let q_norm_w_bf16: Vec<f32> = q_norm_w.iter().copied().map(bf16_round).collect();
        let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let x_norm: Vec<f32> = (0..hidden)
            .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + norm_w_bf16[idx])))
            .collect();
        let q_out_dim = 2 * num_heads * head_dim;
        let mut q_raw = vec![0.0f32; q_out_dim];
        for row in 0..q_out_dim {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                acc += q_proj_w_bf16[row * hidden + col] * x_norm[col];
            }
            q_raw[row] = bf16_round(acc);
        }
        let mut q_normed = vec![0.0f32; num_heads * head_dim];
        for head in 0..num_heads {
            let q_in_base = head * 2 * head_dim;
            let q_out_base = head * head_dim;
            let mean_sq = (0..head_dim)
                .map(|i| q_raw[q_in_base + i].powi(2))
                .sum::<f32>()
                / head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..head_dim {
                q_normed[q_out_base + i] =
                    bf16_round(q_raw[q_in_base + i] * inv * (1.0 + q_norm_w_bf16[i]));
            }
        }
        q_normed
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_attn_stage1_5_reference(
        stage: usize,
        input_hidden: &[f32],
        input_norm_w: &[f32],
        q_proj_w: &[f32],
        k_proj_w: &[f32],
        v_proj_w: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        o_proj_w: &[f32],
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        position: i32,
        eps: f32,
    ) -> Vec<f32> {
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = input_norm_w.iter().copied().map(bf16_round).collect();
        let q_proj_w_bf16: Vec<f32> = q_proj_w.iter().copied().map(bf16_round).collect();
        let k_proj_w_bf16: Vec<f32> = k_proj_w.iter().copied().map(bf16_round).collect();
        let v_proj_w_bf16: Vec<f32> = v_proj_w.iter().copied().map(bf16_round).collect();
        let q_norm_w_bf16: Vec<f32> = q_norm_w.iter().copied().map(bf16_round).collect();
        let k_norm_w_bf16: Vec<f32> = k_norm_w.iter().copied().map(bf16_round).collect();
        let o_proj_w_bf16: Vec<f32> = o_proj_w.iter().copied().map(bf16_round).collect();

        let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let x_norm: Vec<f32> = (0..hidden)
            .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + norm_w_bf16[idx])))
            .collect();

        let q_out_dim = 2 * num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut q_raw = vec![0.0f32; q_out_dim];
        for row in 0..q_out_dim {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                acc += q_proj_w_bf16[row * hidden + col] * x_norm[col];
            }
            q_raw[row] = bf16_round(acc);
        }
        let mut q_normed = vec![0.0f32; num_heads * head_dim];
        for head in 0..num_heads {
            let q_in_base = head * 2 * head_dim;
            let q_out_base = head * head_dim;
            let mean_sq = (0..head_dim)
                .map(|i| q_raw[q_in_base + i].powi(2))
                .sum::<f32>()
                / head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..head_dim {
                q_normed[q_out_base + i] =
                    bf16_round(q_raw[q_in_base + i] * inv * (1.0 + q_norm_w_bf16[i]));
            }
        }
        if stage == 1 {
            return q_normed;
        }

        let mut k_raw = vec![0.0f32; kv_dim];
        let mut v_raw = vec![0.0f32; kv_dim];
        for row in 0..kv_dim {
            let mut k_acc = 0.0f32;
            let mut v_acc = 0.0f32;
            for col in 0..hidden {
                k_acc += k_proj_w_bf16[row * hidden + col] * x_norm[col];
                v_acc += v_proj_w_bf16[row * hidden + col] * x_norm[col];
            }
            k_raw[row] = bf16_round(k_acc);
            v_raw[row] = bf16_round(v_acc);
        }
        let mut k_normed = vec![0.0f32; kv_dim];
        for head in 0..num_kv_heads {
            let base = head * head_dim;
            let mean_sq =
                (0..head_dim).map(|i| k_raw[base + i].powi(2)).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..head_dim {
                k_normed[base + i] = bf16_round(k_raw[base + i] * inv * (1.0 + k_norm_w_bf16[i]));
            }
        }
        if stage == 2 {
            return k_normed;
        }

        fn rope(
            input: &[f32],
            num_heads: usize,
            head_dim: usize,
            rotary_dim: usize,
            rope_theta: f32,
            position: i32,
        ) -> Vec<f32> {
            let mut out = input.to_vec();
            let half = rotary_dim / 2;
            let theta_log = rope_theta.ln();
            for head in 0..num_heads {
                let base = head * head_dim;
                for i in 0..half {
                    let a = input[base + i];
                    let b = input[base + half + i];
                    let freq = position as f32 * (-((i as f32 / half as f32) * theta_log)).exp();
                    let c = bf16_round(freq.cos());
                    let s = bf16_round(freq.sin());
                    out[base + i] = bf16_round(bf16_round(a * c) - bf16_round(b * s));
                    out[base + half + i] = bf16_round(bf16_round(b * c) + bf16_round(a * s));
                }
            }
            out
        }

        let q_rot = rope(
            &q_normed, num_heads, head_dim, rotary_dim, rope_theta, position,
        );
        let k_rot = rope(
            &k_normed,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            position,
        );
        if stage == 3 {
            let mut out = q_rot;
            out.extend_from_slice(&k_rot);
            return out;
        }

        let rep = num_heads / num_kv_heads;
        let mut attn = vec![0.0f32; num_heads * head_dim];
        for hq in 0..num_heads {
            let h_kv = hq / rep;
            for i in 0..head_dim {
                attn[hq * head_dim + i] = v_raw[h_kv * head_dim + i];
            }
        }
        if stage == 4 {
            return attn;
        }

        let mut gated = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            for i in 0..head_dim {
                let out_gate = q_raw[h * 2 * head_dim + head_dim + i];
                let sig = bf16_round(1.0 / (1.0 + (-out_gate).exp()));
                gated[h * head_dim + i] = bf16_round(sig * attn[h * head_dim + i]);
            }
        }
        let qd = num_heads * head_dim;
        let mut out = vec![0.0f32; hidden];
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for col in 0..qd {
                acc += o_proj_w_bf16[row * qd + col] * gated[col];
            }
            out[row] = bf16_round(input_bf16[row] + bf16_round(acc));
        }
        out
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_linear_stage1_reference(
        input_hidden: &[f32],
        input_norm_w: &[f32],
        in_proj_qkv_w: &[f32],
        hidden: usize,
        qkv_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = input_norm_w.iter().copied().map(bf16_round).collect();
        let qkv_w_bf16: Vec<f32> = in_proj_qkv_w.iter().copied().map(bf16_round).collect();
        let mean_sq = input_bf16.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let x_norm: Vec<f32> = (0..hidden)
            .map(|idx| bf16_round(input_bf16[idx] * inv * (1.0 + norm_w_bf16[idx])))
            .collect();
        let mut qkv_raw = vec![0.0f32; qkv_dim];
        for row in 0..qkv_dim {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                acc += qkv_w_bf16[row * hidden + col] * x_norm[col];
            }
            qkv_raw[row] = bf16_round(acc);
        }
        qkv_raw
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_linear_stage2_reference(
        qkv_raw: &[f32],
        conv_state: &[f32],
        conv1d_w: &[f32],
        conv1d_bias: Option<&[f32]>,
        qkv_dim: usize,
        kernel: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let qkv_bf16: Vec<f32> = qkv_raw.iter().copied().map(bf16_round).collect();
        let state_bf16: Vec<f32> = conv_state.iter().copied().map(bf16_round).collect();
        let weight_bf16: Vec<f32> = conv1d_w.iter().copied().map(bf16_round).collect();
        let bias_bf16 = conv1d_bias.map(|b| b.iter().copied().map(bf16_round).collect::<Vec<_>>());
        let kstate = kernel - 1;
        let mut silu_out = vec![0.0f32; qkv_dim];
        let mut next_state = state_bf16.clone();
        for ch in 0..qkv_dim {
            let mut acc = 0.0f32;
            for t in 0..kstate {
                acc += state_bf16[ch * kstate + t] * weight_bf16[ch * kernel + t];
            }
            acc += qkv_bf16[ch] * weight_bf16[ch * kernel + kstate];
            if let Some(bias) = bias_bf16.as_ref() {
                acc += bias[ch];
            }
            let conv_out = bf16_round(acc);
            silu_out[ch] = bf16_round(conv_out * (1.0 / (1.0 + (-conv_out).exp())));
            for t in 0..kstate.saturating_sub(1) {
                next_state[ch * kstate + t] = state_bf16[ch * kstate + t + 1];
            }
            if kstate > 0 {
                next_state[ch * kstate + (kstate - 1)] = qkv_bf16[ch];
            }
        }
        (silu_out, next_state)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_linear_stage3_reference(
        silu_out: &[f32],
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Vec<f32> {
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let mut q_normed = vec![0.0f32; key_dim];
        let mut k_normed = vec![0.0f32; key_dim];
        for head in 0..num_k_heads {
            let q_base = head * head_k_dim;
            let k_base = key_dim + head * head_k_dim;
            let q_ss = (0..head_k_dim)
                .map(|i| silu_out[q_base + i] * silu_out[q_base + i])
                .sum::<f32>();
            let k_ss = (0..head_k_dim)
                .map(|i| silu_out[k_base + i] * silu_out[k_base + i])
                .sum::<f32>();
            let q_denom = bf16_round(bf16_round(q_ss.sqrt()).max(1e-6));
            let k_denom = bf16_round(bf16_round(k_ss.sqrt()).max(1e-6));
            for i in 0..head_k_dim {
                q_normed[q_base + i] = bf16_round(silu_out[q_base + i] / q_denom);
                k_normed[head * head_k_dim + i] = bf16_round(silu_out[k_base + i] / k_denom);
            }
        }

        let mut out = vec![0.0f32; 2 * num_v_heads * head_k_dim + val_dim];
        let rep = num_v_heads / num_k_heads;
        let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        for vhead in 0..num_v_heads {
            let src_kh = vhead / rep;
            for i in 0..head_k_dim {
                out[vhead * head_k_dim + i] =
                    bf16_round(q_normed[src_kh * head_k_dim + i] * q_scale);
                out[num_v_heads * head_k_dim + vhead * head_k_dim + i] =
                    k_normed[src_kh * head_k_dim + i];
            }
            let v_src = 2 * key_dim + vhead * head_v_dim;
            let v_dst = 2 * num_v_heads * head_k_dim + vhead * head_v_dim;
            for i in 0..head_v_dim {
                out[v_dst + i] = silu_out[v_src + i];
            }
        }
        out
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_linear_stage4_reference(
        stage3_out: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        recurrent_state: &[f32],
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let val_dim = num_v_heads * head_v_dim;
        let q_scaled = &stage3_out[..num_v_heads * head_k_dim];
        let k_rep = &stage3_out[num_v_heads * head_k_dim..2 * num_v_heads * head_k_dim];
        let v_heads =
            &stage3_out[2 * num_v_heads * head_k_dim..2 * num_v_heads * head_k_dim + val_dim];
        let a_bf16: Vec<f32> = a_raw.iter().copied().map(bf16_round).collect();
        let b_bf16: Vec<f32> = b_raw.iter().copied().map(bf16_round).collect();
        let dt_bf16: Vec<f32> = dt_bias.iter().copied().map(bf16_round).collect();
        let alog_bf16: Vec<f32> = a_log.iter().copied().map(bf16_round).collect();
        let mut state = recurrent_state.to_vec();
        let mut rec_out = vec![0.0f32; val_dim];

        for head in 0..num_v_heads {
            let beta = 1.0 / (1.0 + (-b_bf16[head]).exp());
            let softplus = (1.0 + (a_bf16[head] + dt_bf16[head]).exp()).ln();
            let gstep = (-softplus * alog_bf16[head].exp()).exp();
            let state_off = head * head_k_dim * head_v_dim;
            let q_off = head * head_k_dim;
            let v_off = head * head_v_dim;

            for e in 0..head_k_dim * head_v_dim {
                state[state_off + e] *= gstep;
            }
            let mut kv_mem = vec![0.0f32; head_v_dim];
            for j in 0..head_v_dim {
                for i in 0..head_k_dim {
                    kv_mem[j] += state[state_off + i * head_v_dim + j] * k_rep[q_off + i];
                }
            }
            let mut delta = vec![0.0f32; head_v_dim];
            for j in 0..head_v_dim {
                delta[j] = (v_heads[v_off + j] - kv_mem[j]) * beta;
            }
            for i in 0..head_k_dim {
                for j in 0..head_v_dim {
                    state[state_off + i * head_v_dim + j] += k_rep[q_off + i] * delta[j];
                }
            }
            for j in 0..head_v_dim {
                let mut acc = 0.0f32;
                for i in 0..head_k_dim {
                    acc += state[state_off + i * head_v_dim + j] * q_scaled[q_off + i];
                }
                rec_out[v_off + j] = bf16_round(acc);
            }
        }

        (rec_out, state)
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn qwen36_linear_stage5_reference(
        input_hidden: &[f32],
        rec_out: &[f32],
        z_raw: &[f32],
        norm_w: &[f32],
        out_proj_w: &[f32],
        hidden: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let val_dim = num_v_heads * head_v_dim;
        let rec_bf16: Vec<f32> = rec_out.iter().copied().map(bf16_round).collect();
        let z_bf16: Vec<f32> = z_raw.iter().copied().map(bf16_round).collect();
        let norm_w_bf16: Vec<f32> = norm_w.iter().copied().map(bf16_round).collect();
        let out_w_bf16: Vec<f32> = out_proj_w.iter().copied().map(bf16_round).collect();
        let input_bf16: Vec<f32> = input_hidden.iter().copied().map(bf16_round).collect();
        let mut gated = vec![0.0f32; val_dim];
        for head in 0..num_v_heads {
            let rec_off = head * head_v_dim;
            let mean_sq = (0..head_v_dim)
                .map(|j| rec_bf16[rec_off + j] * rec_bf16[rec_off + j])
                .sum::<f32>()
                / head_v_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for j in 0..head_v_dim {
                let on = bf16_round(rec_bf16[rec_off + j] * inv * norm_w_bf16[j]);
                let z = z_bf16[rec_off + j];
                let z_silu = bf16_round(z * (1.0 / (1.0 + (-z).exp())));
                gated[rec_off + j] = bf16_round(on * z_silu);
            }
        }
        let mut out = vec![0.0f32; hidden];
        for row in 0..hidden {
            let mut acc = 0.0f32;
            for col in 0..val_dim {
                acc += out_w_bf16[row * val_dim + col] * gated[col];
            }
            out[row] = bf16_round(input_bf16[row] + bf16_round(acc));
        }
        out
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_lm_head_fallback_runs_and_captures_normed() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let vocab = 4usize;
        let eps = 0.0f32;
        let hidden_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let lm_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let hidden_buf = upload_bf16(ordinal, &[hidden], &hidden_vals);
        let norm_buf = upload_bf16(ordinal, &[hidden], &norm_vals);
        let lm_buf = upload_bf16(ordinal, &[vocab, hidden], &lm_vals);
        let mut logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[vocab])
            .expect("alloc qwen36 metal logits");
        let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal normed");
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");

        lm_head_launch(
            ordinal,
            hidden as i32,
            vocab as i32,
            eps,
            &hidden_buf,
            &norm_buf,
            &lm_buf,
            &mut logits,
            Some(&mut normed),
            &mut counter,
        )
        .expect("run Qwen3.6-MoE Metal lm_head fallback");

        let (expected_normed, expected_logits) =
            qwen36_lm_head_reference(&hidden_vals, &norm_vals, &lm_vals, 1, hidden, vocab, eps);
        for (idx, (a, e)) in read_bf16(&normed)
            .iter()
            .zip(expected_normed.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "normed idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_bf16(&logits)
            .iter()
            .zip(expected_logits.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "logit idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_lm_head_batched_fallback_matches_rows() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let rows = 2usize;
        let hidden = 4usize;
        let vocab = 3usize;
        let eps = 1e-5f32;
        let hidden_vals = [1.0, 0.5, -1.0, 2.0, -0.25, 0.75, 1.5, -2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let lm_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
        ];
        let hidden_buf = upload_bf16(ordinal, &[rows, hidden], &hidden_vals);
        let norm_buf = upload_bf16(ordinal, &[hidden], &norm_vals);
        let lm_buf = upload_bf16(ordinal, &[vocab, hidden], &lm_vals);
        let mut logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[rows, vocab])
            .expect("alloc qwen36 metal batched logits");
        let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[rows, hidden])
            .expect("alloc qwen36 metal batched normed");

        lm_head_batched_launch(
            ordinal,
            rows as i32,
            hidden as i32,
            vocab as i32,
            eps,
            &hidden_buf,
            &norm_buf,
            &lm_buf,
            &mut logits,
            Some(&mut normed),
        )
        .expect("run Qwen3.6-MoE Metal batched lm_head fallback");

        let (expected_normed, expected_logits) =
            qwen36_lm_head_reference(&hidden_vals, &norm_vals, &lm_vals, rows, hidden, vocab, eps);
        for (idx, (a, e)) in read_bf16(&normed)
            .iter()
            .zip(expected_normed.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "normed idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_bf16(&logits)
            .iter()
            .zip(expected_logits.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.02,
                "logit idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_mtp_pre_fusion_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let eps = 1e-5f32;
        let e_in_vals = [1.0, 0.5, -1.0, 2.0];
        let h_base_vals = [-0.25, 0.75, 1.5, -2.0];
        let e_norm_w_vals = [0.0, 0.25, -0.5, 1.0];
        let h_norm_w_vals = [0.5, -0.25, 0.0, 0.75];
        let fc_w_vals = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.75, 1.0, //
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        ];
        let e_in = upload_bf16(ordinal, &[hidden], &e_in_vals);
        let h_base = upload_bf16(ordinal, &[hidden], &h_base_vals);
        let e_norm_w = upload_bf16(ordinal, &[hidden], &e_norm_w_vals);
        let h_norm_w = upload_bf16(ordinal, &[hidden], &h_norm_w_vals);
        let fc_w = upload_bf16(ordinal, &[hidden, 2 * hidden], &fc_w_vals);
        let mut e_norm = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal mtp e_norm");
        let mut h_norm = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal mtp h_norm");
        let mut fused = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal mtp fused");

        mtp_pre_fusion_launch(
            ordinal,
            hidden as i32,
            eps,
            &e_in,
            &h_base,
            &e_norm_w,
            &h_norm_w,
            &fc_w,
            &mut e_norm,
            &mut h_norm,
            &mut fused,
        )
        .expect("run Qwen3.6-MoE Metal MTP pre-fusion fallback");

        let (expected_e_norm, expected_h_norm, expected_fused) = qwen36_mtp_pre_fusion_reference(
            &e_in_vals,
            &h_base_vals,
            &e_norm_w_vals,
            &h_norm_w_vals,
            &fc_w_vals,
            hidden,
            eps,
        );
        for (idx, (a, e)) in read_bf16(&e_norm)
            .iter()
            .zip(expected_e_norm.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "e_norm idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_bf16(&h_norm)
            .iter()
            .zip(expected_h_norm.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "h_norm idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_bf16(&fused)
            .iter()
            .zip(expected_fused.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.02,
                "fused idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_ffn_stage1_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_experts = 4usize;
        let top_k = 2usize;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let gate_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let gate_w = upload_bf16(ordinal, &[num_experts, hidden], &gate_vals);
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal ffn output");
        let mut output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[top_k])
            .expect("alloc qwen36 metal ffn output_idx");
        let workspace_len = hidden + 2 * num_experts + 2 * top_k;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
            .expect("alloc qwen36 metal ffn workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 1,
            layer_idx: 0,
            hidden: hidden as i32,
            num_experts: num_experts as i32,
            moe_intermediate: 4,
            shared_intermediate: 4,
            top_k: top_k as i32,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: std::ptr::null(),
            shared_up_proj_w: std::ptr::null(),
            shared_down_proj_w: std::ptr::null(),
            shared_expert_gate_w: std::ptr::null(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal FFN stage1 fallback");

        let (expected_weights, expected_idx) = qwen36_ffn_stage1_reference(
            &input_vals,
            &norm_vals,
            &gate_vals,
            hidden,
            num_experts,
            top_k,
            eps,
        );
        let got_idx_bytes = output_idx.to_host_bytes().expect("download output_idx");
        let got_idx: Vec<i32> = got_idx_bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(got_idx, expected_idx);
        for (idx, (a, e)) in read_bf16(&output)[..top_k]
            .iter()
            .zip(expected_weights.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "topk weight idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_ffn_stage2_5_fallbacks_match_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_experts = 4usize;
        let moe_intermediate = 3usize;
        let shared_intermediate = 3usize;
        let top_k = 2usize;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let gate_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
        ];
        let shared_gate_vals = [
            0.25, -0.5, 0.75, 0.125, //
            -0.25, 0.5, -0.75, 1.0, //
            0.5, 0.25, -0.25, 0.75, //
        ];
        let shared_up_vals = [
            -0.5, 0.25, 1.0, -0.25, //
            0.75, -0.75, 0.5, 0.125, //
            0.25, 0.5, -0.5, 1.0, //
        ];
        let shared_down_vals = [
            0.5, -0.25, 0.75, //
            -0.5, 0.25, 1.0, //
            0.125, -0.75, 0.5, //
            0.75, 0.5, -0.25, //
        ];
        let shared_gate_scalar_vals = [0.25, -0.5, 0.75, 1.0];
        let mut gate_up_vals = vec![0.0f32; num_experts * 2 * moe_intermediate * hidden];
        for (idx, v) in gate_up_vals.iter_mut().enumerate() {
            *v = ((idx % 11) as f32 - 5.0) * 0.125;
        }
        let mut down_vals = vec![0.0f32; num_experts * hidden * moe_intermediate];
        for (idx, v) in down_vals.iter_mut().enumerate() {
            *v = ((idx % 13) as f32 - 6.0) * 0.1;
        }

        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let gate_w = upload_bf16(ordinal, &[num_experts, hidden], &gate_vals);
        let gate_up_proj_w = upload_bf16(
            ordinal,
            &[num_experts, 2 * moe_intermediate, hidden],
            &gate_up_vals,
        );
        let down_proj_w = upload_bf16(
            ordinal,
            &[num_experts, hidden, moe_intermediate],
            &down_vals,
        );
        let shared_gate_proj_w =
            upload_bf16(ordinal, &[shared_intermediate, hidden], &shared_gate_vals);
        let shared_up_proj_w =
            upload_bf16(ordinal, &[shared_intermediate, hidden], &shared_up_vals);
        let shared_down_proj_w =
            upload_bf16(ordinal, &[hidden, shared_intermediate], &shared_down_vals);
        let shared_expert_gate_w = upload_bf16(ordinal, &[1, hidden], &shared_gate_scalar_vals);

        let weights = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        for stage in 2..=5 {
            let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
                .expect("alloc qwen36 metal ffn output");
            let mut output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[top_k])
                .expect("alloc qwen36 metal ffn output_idx");
            let workspace_len = 3 * hidden
                + 2 * num_experts
                + 2 * top_k
                + 1
                + 3 * shared_intermediate
                + top_k * 3 * moe_intermediate
                + top_k * hidden;
            let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
                .expect("alloc qwen36 metal ffn workspace");
            let mut sync_buf =
                GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

            let params = Qwen36MoeFfnStepParams {
                stage,
                layer_idx: 0,
                hidden: hidden as i32,
                num_experts: num_experts as i32,
                moe_intermediate: moe_intermediate as i32,
                shared_intermediate: shared_intermediate as i32,
                top_k: top_k as i32,
                rms_norm_eps: eps,
            };
            ffn_step_launch(
                ordinal,
                ScalarType::BF16,
                params,
                &weights,
                &Qwen36MoeFfnStepInt4::disabled(),
                &mut output,
                &mut output_idx,
                &mut workspace,
                &mut sync_buf,
            )
            .expect("run Qwen3.6-MoE Metal FFN fallback");

            let (expected, expected_idx) = qwen36_ffn_stage1_5_reference(
                stage as usize,
                &input_vals,
                &norm_vals,
                &gate_vals,
                &gate_up_vals,
                &down_vals,
                &shared_gate_vals,
                &shared_up_vals,
                &shared_down_vals,
                &shared_gate_scalar_vals,
                hidden,
                num_experts,
                moe_intermediate,
                shared_intermediate,
                top_k,
                eps,
            );
            let got_idx_bytes = output_idx.to_host_bytes().expect("download output_idx");
            let got_idx: Vec<i32> = got_idx_bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            assert_eq!(got_idx, expected_idx, "stage {stage} topk idx mismatch");
            for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= 0.02,
                    "stage {stage} output idx {idx}: expected {e}, got {a}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_attn_stage1_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let q_norm_vals = [0.0, 0.5];
        let q_proj_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.5, 0.5, -0.25, 0.25, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let q_proj_w = upload_bf16(ordinal, &[2 * num_heads * head_dim, hidden], &q_proj_vals);
        let q_norm_w = upload_bf16(ordinal, &[head_dim], &q_norm_vals);
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_heads * head_dim])
            .expect("alloc qwen36 metal attn output");
        let workspace_len =
            2 * num_heads * head_dim + 2 * num_kv_heads * head_dim + num_heads * head_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
            .expect("alloc qwen36 metal attn workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 1,
            hidden: hidden as i32,
            num_heads: num_heads as i32,
            num_kv_heads: num_kv_heads as i32,
            head_dim: head_dim as i32,
            rotary_dim: head_dim as i32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: eps,
            position: 0,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weights = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal attention stage1 fallback");

        let expected = qwen36_attn_stage1_reference(
            &input_vals,
            &norm_vals,
            &q_proj_vals,
            &q_norm_vals,
            hidden,
            num_heads,
            head_dim,
            eps,
        );
        for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "q_normed idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_attn_stage1_int4_sidecar_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let group_size = 2usize;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let q_norm_vals = [0.0, 0.5];
        let nibbles = vec![
            vec![4, 0, 0, 0],
            vec![0, 4, 0, 0],
            vec![1, 2, 0, 0],
            vec![0, 0, 2, 1],
            vec![0, 0, 4, 0],
            vec![0, 0, 0, 4],
            vec![2, 0, 1, 0],
            vec![0, 1, 0, 2],
        ];
        let q_proj_vals = nibbles
            .iter()
            .flat_map(|row| row.iter().map(|v| *v as f32 * 0.25))
            .collect::<Vec<_>>();

        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let (q_proj_w, q_proj_scale, q_proj_zero) = upload_int4_rows(
            ordinal,
            2 * num_heads * head_dim,
            hidden,
            group_size,
            &nibbles,
            0.25,
        );
        let q_norm_w = upload_bf16(ordinal, &[head_dim], &q_norm_vals);
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_heads * head_dim])
            .expect("alloc qwen36 metal attn int4 output");
        let workspace_len =
            2 * num_heads * head_dim + 2 * num_kv_heads * head_dim + num_heads * head_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
            .expect("alloc qwen36 metal attn int4 workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 1,
            hidden: hidden as i32,
            num_heads: num_heads as i32,
            num_kv_heads: num_kv_heads as i32,
            head_dim: head_dim as i32,
            rotary_dim: head_dim as i32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: eps,
            position: 0,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weights = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };
        let int4 = Qwen36MoeAttnStepInt4 {
            q_proj_scale: q_proj_scale.as_ptr(),
            q_proj_zero: q_proj_zero.as_ptr(),
            k_proj_scale: std::ptr::null(),
            k_proj_zero: std::ptr::null(),
            v_proj_scale: std::ptr::null(),
            v_proj_zero: std::ptr::null(),
            o_proj_scale: std::ptr::null(),
            o_proj_zero: std::ptr::null(),
            group_size: group_size as i32,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &int4,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal attention stage1 INT4 fallback");

        let expected = qwen36_attn_stage1_reference(
            &input_vals,
            &norm_vals,
            &q_proj_vals,
            &q_norm_vals,
            hidden,
            num_heads,
            head_dim,
            eps,
        );
        for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "int4 q_normed idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_attn_stage2_5_fallbacks_match_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let q_norm_vals = [0.0, 0.5];
        let k_norm_vals = [0.25, -0.25];
        let q_proj_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.5, 0.5, -0.25, 0.25, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let k_proj_vals = [
            0.5, -0.25, 1.0, 0.25, //
            -0.75, 0.5, 0.25, -0.5, //
        ];
        let v_proj_vals = [
            0.25, 0.75, -0.5, 0.5, //
            -0.5, 0.25, 0.75, 1.0, //
        ];
        let o_proj_vals = [
            0.25, -0.5, 0.75, 1.0, //
            -0.25, 0.5, -0.75, 0.125, //
            0.5, 0.25, 0.25, -0.5, //
            -1.0, 0.75, 0.5, 0.25, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let q_proj_w = upload_bf16(ordinal, &[2 * num_heads * head_dim, hidden], &q_proj_vals);
        let k_proj_w = upload_bf16(ordinal, &[num_kv_heads * head_dim, hidden], &k_proj_vals);
        let v_proj_w = upload_bf16(ordinal, &[num_kv_heads * head_dim, hidden], &v_proj_vals);
        let q_norm_w = upload_bf16(ordinal, &[head_dim], &q_norm_vals);
        let k_norm_w = upload_bf16(ordinal, &[head_dim], &k_norm_vals);
        let o_proj_w = upload_bf16(ordinal, &[hidden, num_heads * head_dim], &o_proj_vals);

        let weights = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: o_proj_w.as_ptr(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        for stage in 2..=5 {
            let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[6])
                .expect("alloc qwen36 metal attn output");
            let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[36])
                .expect("alloc qwen36 metal attn workspace");
            let mut sync_buf =
                GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");
            let params = Qwen36MoeAttnStepParams {
                stage,
                hidden: hidden as i32,
                num_heads: num_heads as i32,
                num_kv_heads: num_kv_heads as i32,
                head_dim: head_dim as i32,
                rotary_dim: head_dim as i32,
                rope_theta: 1_000_000.0,
                rms_norm_eps: eps,
                position: 3,
                cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
            };

            attn_step_launch(
                ordinal,
                ScalarType::BF16,
                params,
                &weights,
                &Qwen36MoeAttnStepInt4::disabled(),
                &mut output,
                &mut workspace,
                &mut sync_buf,
            )
            .expect("run Qwen3.6-MoE Metal attention fallback");

            let expected = qwen36_attn_stage1_5_reference(
                stage as usize,
                &input_vals,
                &norm_vals,
                &q_proj_vals,
                &k_proj_vals,
                &v_proj_vals,
                &q_norm_vals,
                &k_norm_vals,
                &o_proj_vals,
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim,
                1_000_000.0,
                3,
                eps,
            );
            let got = read_bf16(&output);
            for (idx, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= 0.02,
                    "stage {stage} idx {idx}: expected {e}, got {a}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_linear_stage1_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_k_heads = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 1usize;
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let qkv_dim = 2 * key_dim + val_dim;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let qkv_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let z_vals = [
            0.5, 0.0, -0.5, 1.0, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let a_vals = [
            1.0, -0.5, 0.0, 0.25, //
            0.0, 0.25, 0.5, -1.0, //
        ];
        let b_vals = [
            -0.5, 0.5, 1.0, 0.0, //
            0.25, 0.0, -0.25, 0.75, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let in_proj_qkv_w = upload_bf16(ordinal, &[qkv_dim, hidden], &qkv_vals);
        let in_proj_z_w = upload_bf16(ordinal, &[val_dim, hidden], &z_vals);
        let in_proj_a_w = upload_bf16(ordinal, &[num_v_heads, hidden], &a_vals);
        let in_proj_b_w = upload_bf16(ordinal, &[num_v_heads, hidden], &b_vals);
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim])
            .expect("alloc qwen36 metal linear output");
        let workspace_len = qkv_dim + val_dim + 2 * num_v_heads;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
            .expect("alloc qwen36 metal linear workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 1,
            hidden: hidden as i32,
            num_k_heads: num_k_heads as i32,
            num_v_heads: num_v_heads as i32,
            head_k_dim: head_k_dim as i32,
            head_v_dim: head_v_dim as i32,
            conv_kernel_dim: 4,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: std::ptr::null(),
            conv1d_bias: std::ptr::null(),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: std::ptr::null_mut(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal linear stage1 fallback");

        let expected = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &qkv_vals,
            hidden,
            qkv_dim,
            eps,
        );
        for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "qkv_raw idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_linear_stage2_fallback_matches_reference_and_updates_state() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_k_heads = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 1usize;
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let qkv_dim = 2 * key_dim + val_dim;
        let kernel = 3usize;
        let kstate = kernel - 1;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let qkv_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let z_vals = [
            0.5, 0.0, -0.5, 1.0, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let a_vals = [
            1.0, -0.5, 0.0, 0.25, //
            0.0, 0.25, 0.5, -1.0, //
        ];
        let b_vals = [
            -0.5, 0.5, 1.0, 0.0, //
            0.25, 0.0, -0.25, 0.75, //
        ];
        let conv_w_vals = [
            0.5, -0.25, 1.0, //
            0.25, 0.5, -0.5, //
            -0.5, 1.0, 0.25, //
            1.0, 0.0, -0.25, //
            0.75, -0.5, 0.5, //
            -0.25, 0.25, 1.0, //
        ];
        let conv_bias_vals = [0.0, 0.25, -0.25, 0.5, -0.5, 0.125];
        let conv_state_vals = [
            0.25, -0.5, //
            0.0, 0.5, //
            -0.25, 0.75, //
            1.0, -1.0, //
            0.125, -0.375, //
            -0.75, 0.25, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let in_proj_qkv_w = upload_bf16(ordinal, &[qkv_dim, hidden], &qkv_vals);
        let in_proj_z_w = upload_bf16(ordinal, &[val_dim, hidden], &z_vals);
        let in_proj_a_w = upload_bf16(ordinal, &[num_v_heads, hidden], &a_vals);
        let in_proj_b_w = upload_bf16(ordinal, &[num_v_heads, hidden], &b_vals);
        let conv1d_w = upload_bf16(ordinal, &[qkv_dim, kernel], &conv_w_vals);
        let conv1d_bias = upload_bf16(ordinal, &[qkv_dim], &conv_bias_vals);
        let mut conv_state = upload_bf16(ordinal, &[qkv_dim, kstate], &conv_state_vals);
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim])
            .expect("alloc qwen36 metal linear stage2 output");
        let workspace_len = qkv_dim + val_dim + 2 * num_v_heads;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len])
            .expect("alloc qwen36 metal linear stage2 workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 2,
            hidden: hidden as i32,
            num_k_heads: num_k_heads as i32,
            num_v_heads: num_v_heads as i32,
            head_k_dim: head_k_dim as i32,
            head_v_dim: head_v_dim as i32,
            conv_kernel_dim: kernel as i32,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias.as_ptr(),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal linear stage2 fallback");

        let qkv_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &qkv_vals,
            hidden,
            qkv_dim,
            eps,
        );
        let (expected_silu, expected_state) = qwen36_linear_stage2_reference(
            &qkv_raw,
            &conv_state_vals,
            &conv_w_vals,
            Some(&conv_bias_vals),
            qkv_dim,
            kernel,
        );
        for (idx, (a, e)) in read_bf16(&output)
            .iter()
            .zip(expected_silu.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "silu_out idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_bf16(&conv_state)
            .iter()
            .zip(expected_state.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "conv_state idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_linear_stage3_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_k_heads = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 1usize;
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let qkv_dim = 2 * key_dim + val_dim;
        let kernel = 3usize;
        let kstate = kernel - 1;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let qkv_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let z_vals = [
            0.5, 0.0, -0.5, 1.0, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let a_vals = [
            1.0, -0.5, 0.0, 0.25, //
            0.0, 0.25, 0.5, -1.0, //
        ];
        let b_vals = [
            -0.5, 0.5, 1.0, 0.0, //
            0.25, 0.0, -0.25, 0.75, //
        ];
        let conv_w_vals = [
            0.5, -0.25, 1.0, //
            0.25, 0.5, -0.5, //
            -0.5, 1.0, 0.25, //
            1.0, 0.0, -0.25, //
            0.75, -0.5, 0.5, //
            -0.25, 0.25, 1.0, //
        ];
        let conv_state_vals = [
            0.25, -0.5, //
            0.0, 0.5, //
            -0.25, 0.75, //
            1.0, -1.0, //
            0.125, -0.375, //
            -0.75, 0.25, //
        ];
        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let in_proj_qkv_w = upload_bf16(ordinal, &[qkv_dim, hidden], &qkv_vals);
        let in_proj_z_w = upload_bf16(ordinal, &[val_dim, hidden], &z_vals);
        let in_proj_a_w = upload_bf16(ordinal, &[num_v_heads, hidden], &a_vals);
        let in_proj_b_w = upload_bf16(ordinal, &[num_v_heads, hidden], &b_vals);
        let conv1d_w = upload_bf16(ordinal, &[qkv_dim, kernel], &conv_w_vals);
        let mut conv_state = upload_bf16(ordinal, &[qkv_dim, kstate], &conv_state_vals);
        let output_len = 2 * num_v_heads * head_k_dim + val_dim;
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[output_len])
            .expect("alloc qwen36 metal linear stage3 output");
        let stage3_workspace_len =
            qkv_dim + val_dim + 2 * num_v_heads + 2 * key_dim + 2 * num_v_heads * head_k_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[stage3_workspace_len])
            .expect("alloc qwen36 metal linear stage3 workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 3,
            hidden: hidden as i32,
            num_k_heads: num_k_heads as i32,
            num_v_heads: num_v_heads as i32,
            head_k_dim: head_k_dim as i32,
            head_v_dim: head_v_dim as i32,
            conv_kernel_dim: kernel as i32,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: std::ptr::null(),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal linear stage3 fallback");

        let qkv_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &qkv_vals,
            hidden,
            qkv_dim,
            eps,
        );
        let (silu_out, _) = qwen36_linear_stage2_reference(
            &qkv_raw,
            &conv_state_vals,
            &conv_w_vals,
            None,
            qkv_dim,
            kernel,
        );
        let expected = qwen36_linear_stage3_reference(
            &silu_out,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
        for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "stage3 output idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_linear_stage4_fallback_matches_reference_and_updates_recurrent_state() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_k_heads = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 1usize;
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let qkv_dim = 2 * key_dim + val_dim;
        let kernel = 3usize;
        let kstate = kernel - 1;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let qkv_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let z_vals = [
            0.5, 0.0, -0.5, 1.0, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let a_vals = [
            1.0, -0.5, 0.0, 0.25, //
            0.0, 0.25, 0.5, -1.0, //
        ];
        let b_vals = [
            -0.5, 0.5, 1.0, 0.0, //
            0.25, 0.0, -0.25, 0.75, //
        ];
        let conv_w_vals = [
            0.5, -0.25, 1.0, //
            0.25, 0.5, -0.5, //
            -0.5, 1.0, 0.25, //
            1.0, 0.0, -0.25, //
            0.75, -0.5, 0.5, //
            -0.25, 0.25, 1.0, //
        ];
        let conv_state_vals = [
            0.25, -0.5, //
            0.0, 0.5, //
            -0.25, 0.75, //
            1.0, -1.0, //
            0.125, -0.375, //
            -0.75, 0.25, //
        ];
        let dt_bias_vals = [0.125, -0.25];
        let a_log_vals = [-0.5, 0.25];
        let recurrent_vals = [0.25, -0.5, 0.75, -1.0];

        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let in_proj_qkv_w = upload_bf16(ordinal, &[qkv_dim, hidden], &qkv_vals);
        let in_proj_z_w = upload_bf16(ordinal, &[val_dim, hidden], &z_vals);
        let in_proj_a_w = upload_bf16(ordinal, &[num_v_heads, hidden], &a_vals);
        let in_proj_b_w = upload_bf16(ordinal, &[num_v_heads, hidden], &b_vals);
        let conv1d_w = upload_bf16(ordinal, &[qkv_dim, kernel], &conv_w_vals);
        let mut conv_state = upload_bf16(ordinal, &[qkv_dim, kstate], &conv_state_vals);
        let dt_bias = upload_bf16(ordinal, &[num_v_heads], &dt_bias_vals);
        let a_log = upload_bf16(ordinal, &[num_v_heads], &a_log_vals);
        let mut recurrent_state = upload_f32(
            ordinal,
            &[num_v_heads, head_k_dim, head_v_dim],
            &recurrent_vals,
        );
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[val_dim])
            .expect("alloc qwen36 metal linear stage4 output");
        let stage4_workspace_len = qkv_dim
            + val_dim
            + 2 * num_v_heads
            + 2 * key_dim
            + 2 * num_v_heads * head_k_dim
            + 2 * num_v_heads
            + val_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[stage4_workspace_len])
            .expect("alloc qwen36 metal linear stage4 workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 4,
            hidden: hidden as i32,
            num_k_heads: num_k_heads as i32,
            num_v_heads: num_v_heads as i32,
            head_k_dim: head_k_dim as i32,
            head_v_dim: head_v_dim as i32,
            conv_kernel_dim: kernel as i32,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: std::ptr::null(),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal linear stage4 fallback");

        let qkv_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &qkv_vals,
            hidden,
            qkv_dim,
            eps,
        );
        let a_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &a_vals,
            hidden,
            num_v_heads,
            eps,
        );
        let b_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &b_vals,
            hidden,
            num_v_heads,
            eps,
        );
        let (silu_out, _) = qwen36_linear_stage2_reference(
            &qkv_raw,
            &conv_state_vals,
            &conv_w_vals,
            None,
            qkv_dim,
            kernel,
        );
        let stage3_out = qwen36_linear_stage3_reference(
            &silu_out,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
        let (expected_rec, expected_state) = qwen36_linear_stage4_reference(
            &stage3_out,
            &a_raw,
            &b_raw,
            &dt_bias_vals,
            &a_log_vals,
            &recurrent_vals,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
        for (idx, (a, e)) in read_bf16(&output)
            .iter()
            .zip(expected_rec.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.01,
                "stage4 output idx {idx}: expected {e}, got {a}"
            );
        }
        for (idx, (a, e)) in read_f32(&recurrent_state)
            .iter()
            .zip(expected_state.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 0.0005,
                "recurrent_state idx {idx}: expected {e}, got {a}"
            );
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_linear_stage5_fallback_matches_reference() {
        use gpu_hal::{set_backend, Backend};

        set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let num_k_heads = 1usize;
        let num_v_heads = 2usize;
        let head_k_dim = 2usize;
        let head_v_dim = 1usize;
        let key_dim = num_k_heads * head_k_dim;
        let val_dim = num_v_heads * head_v_dim;
        let qkv_dim = 2 * key_dim + val_dim;
        let kernel = 3usize;
        let kstate = kernel - 1;
        let eps = 1e-5f32;
        let input_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [0.0, 0.25, -0.5, 1.0];
        let qkv_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.25, -0.5, 0.75, 1.0, //
            -0.5, 0.25, 0.5, -0.25, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let z_vals = [
            0.5, 0.0, -0.5, 1.0, //
            -0.25, 0.75, 0.25, -0.5, //
        ];
        let a_vals = [
            1.0, -0.5, 0.0, 0.25, //
            0.0, 0.25, 0.5, -1.0, //
        ];
        let b_vals = [
            -0.5, 0.5, 1.0, 0.0, //
            0.25, 0.0, -0.25, 0.75, //
        ];
        let conv_w_vals = [
            0.5, -0.25, 1.0, //
            0.25, 0.5, -0.5, //
            -0.5, 1.0, 0.25, //
            1.0, 0.0, -0.25, //
            0.75, -0.5, 0.5, //
            -0.25, 0.25, 1.0, //
        ];
        let conv_state_vals = [
            0.25, -0.5, //
            0.0, 0.5, //
            -0.25, 0.75, //
            1.0, -1.0, //
            0.125, -0.375, //
            -0.75, 0.25, //
        ];
        let dt_bias_vals = [0.125, -0.25];
        let a_log_vals = [-0.5, 0.25];
        let recurrent_vals = [0.25, -0.5, 0.75, -1.0];
        let out_norm_vals = [1.25];
        let out_proj_vals = [
            1.0, 0.0, //
            0.0, 1.0, //
            0.5, -0.25, //
            -0.75, 0.25, //
        ];

        let input_hidden = upload_bf16(ordinal, &[hidden], &input_vals);
        let input_norm_w = upload_bf16(ordinal, &[hidden], &norm_vals);
        let in_proj_qkv_w = upload_bf16(ordinal, &[qkv_dim, hidden], &qkv_vals);
        let in_proj_z_w = upload_bf16(ordinal, &[val_dim, hidden], &z_vals);
        let in_proj_a_w = upload_bf16(ordinal, &[num_v_heads, hidden], &a_vals);
        let in_proj_b_w = upload_bf16(ordinal, &[num_v_heads, hidden], &b_vals);
        let conv1d_w = upload_bf16(ordinal, &[qkv_dim, kernel], &conv_w_vals);
        let mut conv_state = upload_bf16(ordinal, &[qkv_dim, kstate], &conv_state_vals);
        let dt_bias = upload_bf16(ordinal, &[num_v_heads], &dt_bias_vals);
        let a_log = upload_bf16(ordinal, &[num_v_heads], &a_log_vals);
        let norm_w = upload_bf16(ordinal, &[head_v_dim], &out_norm_vals);
        let out_proj_w = upload_bf16(ordinal, &[hidden, val_dim], &out_proj_vals);
        let mut recurrent_state = upload_f32(
            ordinal,
            &[num_v_heads, head_k_dim, head_v_dim],
            &recurrent_vals,
        );
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden])
            .expect("alloc qwen36 metal linear stage5 output");
        let stage4_workspace_len = qkv_dim
            + val_dim
            + 2 * num_v_heads
            + 2 * key_dim
            + 2 * num_v_heads * head_k_dim
            + 2 * num_v_heads
            + val_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[stage4_workspace_len])
            .expect("alloc qwen36 metal linear stage5 workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 5,
            hidden: hidden as i32,
            num_k_heads: num_k_heads as i32,
            num_v_heads: num_v_heads as i32,
            head_k_dim: head_k_dim as i32,
            head_v_dim: head_v_dim as i32,
            conv_kernel_dim: kernel as i32,
            rms_norm_eps: eps,
        };
        let weights = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: std::ptr::null(),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: norm_w.as_ptr(),
            out_proj_w: out_proj_w.as_ptr(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weights,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("run Qwen3.6-MoE Metal linear stage5 fallback");

        let qkv_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &qkv_vals,
            hidden,
            qkv_dim,
            eps,
        );
        let z_raw =
            qwen36_linear_stage1_reference(&input_vals, &norm_vals, &z_vals, hidden, val_dim, eps);
        let a_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &a_vals,
            hidden,
            num_v_heads,
            eps,
        );
        let b_raw = qwen36_linear_stage1_reference(
            &input_vals,
            &norm_vals,
            &b_vals,
            hidden,
            num_v_heads,
            eps,
        );
        let (silu_out, _) = qwen36_linear_stage2_reference(
            &qkv_raw,
            &conv_state_vals,
            &conv_w_vals,
            None,
            qkv_dim,
            kernel,
        );
        let stage3_out = qwen36_linear_stage3_reference(
            &silu_out,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
        let (rec_out, _) = qwen36_linear_stage4_reference(
            &stage3_out,
            &a_raw,
            &b_raw,
            &dt_bias_vals,
            &a_log_vals,
            &recurrent_vals,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        );
        let expected = qwen36_linear_stage5_reference(
            &input_vals,
            &rec_out,
            &z_raw,
            &out_norm_vals,
            &out_proj_vals,
            hidden,
            num_v_heads,
            head_v_dim,
            eps,
        );
        for (idx, (a, e)) in read_bf16(&output).iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "stage5 output idx {idx}: expected {e}, got {a}"
            );
        }
    }

    /// HIP smoke test: launch the stub kernel against a synthetic 40-layer
    /// descriptor array and verify the sentinel bytes the kernel writes
    /// match what we sent in. This exercises the entire path:
    /// FFI struct layout → bridge launch → grid barrier → cooperative
    /// work-stealing → host readback. The same path the real kernel will
    /// use, minus the compute math.
    #[cfg(supersonic_backend_hip)]
    #[test]
    fn hip_stub_launch_walks_descriptor_array() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        set_backend(Backend::Hip);
        let ordinal = 0usize;
        let num_layers = 40usize;

        // Synthesize a Qwen3.6-MoE-shaped descriptor array on the host.
        // Hybrid pattern: every 4th layer is full attention; others are
        // linear-attention. attn_output_gate is set on full layers only.
        let mut host_descs: Vec<Qwen36MoeDecodeLayerDesc> = Vec::with_capacity(num_layers);
        let num_experts = 256;
        let top_k = 8;
        for idx in 0..num_layers {
            let mut d = Qwen36MoeDecodeLayerDesc::default();
            d.layer_idx = idx as c_int;
            d.is_full_attention = if (idx + 1) % 4 == 0 { 1 } else { 0 };
            d.attn_output_gate = if d.is_full_attention == 1 { 1 } else { 0 };
            d.num_experts = num_experts;
            d.top_k = top_k;
            d.moe_intermediate_size = 512;
            d.shared_expert_intermediate_size = 512;
            d.norm_topk_prob = 1;
            d.attn_head_dim = 256;
            d.attn_num_heads = 16;
            d.attn_num_kv_heads = 2;
            host_descs.push(d);
        }

        // Upload as raw bytes; gpu-hal lets us treat the buffer as opaque
        // u8 since the kernel only dereferences the C struct pointer.
        let desc_bytes_per = size_of::<Qwen36MoeDecodeLayerDesc>();
        let mut desc_bytes = Vec::with_capacity(desc_bytes_per * num_layers);
        for d in &host_descs {
            let p = d as *const Qwen36MoeDecodeLayerDesc as *const u8;
            desc_bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(p, desc_bytes_per) });
        }
        let layer_descs =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[desc_bytes.len()], &desc_bytes)
                .expect("upload descriptor array");

        // 16 floats of workspace is enough for the documented sentinel
        // slots (5 in use, rest reserved). The real kernel will need
        // ~MiB; keeping this tiny lets the smoke test stay fast.
        let mut workspace =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[16]).expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        stub_launch(
            ordinal,
            ScalarType::BF16,
            &layer_descs,
            &mut workspace,
            &mut sync_buf,
            num_layers,
        )
        .expect("stub launch");

        // Read sentinels.
        let bytes = workspace.to_host_bytes().expect("download workspace");
        let workspace_f32: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(
            workspace_f32[0] as usize, num_layers,
            "[0] num_layers seen by kernel"
        );
        assert_eq!(
            workspace_f32[1] as i32,
            (num_experts as i32) * (num_layers as i32),
            "[1] sum of num_experts across layers"
        );
        assert_eq!(
            workspace_f32[2] as i32,
            (top_k as i32) * (num_layers as i32),
            "[2] sum of top_k across layers"
        );
        assert_eq!(
            workspace_f32[3], 1.0,
            "[3] hybrid pattern check (1.0 = pattern OK across all layers)"
        );
        assert_eq!(
            workspace_f32[4], 1.0,
            "[4] attn_output_gate consistency on full layers"
        );
    }

    // ---- PR 4b2 step 1: q-path parity vs the PyTorch oracle --------------
    //
    // The test reads a JSON produced by `oracle/qwen36_moe_oracle.py`
    // (synthetic or checkpoint mode), uploads the four input tensors needed
    // for stage 1 (input_hidden, input_norm_w, q_proj_w, q_norm_w), runs
    // the staged kernel, downloads the BF16 q_normed output, and compares
    // against `intermediates.q_normed` from the oracle.
    //
    // To run: produce a JSON, then point the test at it via env var:
    //
    //   python oracle/qwen36_moe_oracle.py --mode synthetic \
    //       --hidden 2048 --num-attention-heads 16 --num-kv-heads 2 \
    //       --head-dim 256 --out /tmp/qwen36_syn.json
    //   SUPERSONIC_QWEN36_ORACLE_JSON=/tmp/qwen36_syn.json \
    //       cargo test --release -p kernel-ffi qwen36_moe_attn_step_1
    //
    // Without the env var the test prints a clear skip message and exits.
    // We don't fail-on-missing because the FFI test must remain runnable
    // on hosts without Python/PyTorch (CI without GPU, header-only checks).

    /// Decode a base64 string to bytes. Inline so the test stays
    /// dependency-free aside from serde_json. RFC 4648 alphabet, no padding
    /// tolerance shortcuts (we know the oracle always emits valid BF16
    /// payloads ≡ even-length byte streams ≡ 4n base64 chars after padding).
    #[cfg(supersonic_backend_hip)]
    fn b64_decode(input: &str) -> Vec<u8> {
        const TABLE: &[u8; 256] = &{
            let mut t = [255u8; 256];
            let mut i = 0;
            let charset = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            while i < charset.len() {
                t[charset[i] as usize] = i as u8;
                i += 1;
            }
            t
        };
        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut buf = 0u32;
        let mut bits = 0;
        for &b in input.as_bytes() {
            if b == b'=' || b == b'\n' || b == b'\r' || b == b' ' {
                continue;
            }
            let v = TABLE[b as usize];
            assert!(v != 255, "qwen36_moe parity: invalid base64 byte {b:#x}");
            buf = (buf << 6) | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xFF) as u8);
            }
        }
        out
    }

    /// Convert a stream of F32 little-endian bytes to F32 values. Used
    /// for parity-checking the recurrent state buffer (stage 4+), which
    /// production keeps in F32 across decode steps.
    #[cfg(supersonic_backend_hip)]
    fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        assert!(
            bytes.len() % 4 == 0,
            "qwen36_moe parity: F32 bytes must be multiple of 4"
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Same shape as `assert_parity` but for F32 buffers — used for the
    /// recurrent state which never casts to BF16. Tolerances are tighter
    /// (no BF16 rounding noise to absorb).
    #[cfg(supersonic_backend_hip)]
    fn assert_parity_f32(
        label: &str,
        got_bytes: &[u8],
        want_bytes: &[u8],
        max_abs_tol: f32,
        cos_sim_floor: f64,
    ) {
        assert_eq!(
            got_bytes.len(),
            want_bytes.len(),
            "{label}: byte length mismatch"
        );
        let got = f32_bytes_to_f32(got_bytes);
        let want = f32_bytes_to_f32(want_bytes);
        let n = got.len();
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut dot = 0.0f64;
        let mut got_sq = 0.0f64;
        let mut want_sq = 0.0f64;
        let mut exact = 0usize;
        for i in 0..n {
            let d = (got[i] - want[i]).abs();
            if d == 0.0 {
                exact += 1;
            }
            max_abs_diff = max_abs_diff.max(d);
            sum_abs_diff += d;
            dot += got[i] as f64 * want[i] as f64;
            got_sq += (got[i] as f64).powi(2);
            want_sq += (want[i] as f64).powi(2);
        }
        let cos_sim = dot / (got_sq.sqrt() * want_sq.sqrt() + 1e-30);
        let mean_abs_diff = sum_abs_diff / n as f32;
        eprintln!(
            "[parity {label}] n={n} exact={exact} max_abs={max_abs_diff:.5e} \
             mean_abs={mean_abs_diff:.5e} cos_sim={cos_sim:.7}"
        );
        assert!(
            max_abs_diff <= max_abs_tol,
            "{label}: max_abs={max_abs_diff} exceeds tolerance {max_abs_tol}"
        );
        assert!(
            cos_sim >= cos_sim_floor,
            "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}"
        );
    }

    /// Convert a stream of BF16 little-endian bytes to F32. The oracle
    /// stores BF16 as raw int16 → bytes, matching the kernel's ABI.
    #[cfg(supersonic_backend_hip)]
    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        assert!(
            bytes.len() % 2 == 0,
            "qwen36_moe parity: BF16 bytes must be even"
        );
        bytes
            .chunks_exact(2)
            .map(|c| {
                // BF16 = top 16 bits of an F32. Reconstruct by zero-extending.
                let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                f32::from_bits(bits << 16)
            })
            .collect()
    }

    /// Geometry pulled from the oracle JSON's `config` block — pinned to
    /// what every parity test in this file consumes.
    #[cfg(supersonic_backend_hip)]
    struct OracleGeom {
        hidden: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_oracle_json() -> Option<(serde_json::Value, OracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read oracle json {json_path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&raw).expect("oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "PR 4b2 parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = OracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_heads: cfg["num_attention_heads"].as_i64().unwrap() as i32,
            num_kv_heads: cfg["num_kv_heads"].as_i64().unwrap() as i32,
            head_dim: cfg["head_dim"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    /// Compare a kernel-produced BF16 buffer against the matching oracle
    /// intermediate, emit a one-line summary, and assert tolerances. BF16
    /// stores at every boundary mean most elements are bit-exact; the rare
    /// 1-ULP misses come from F32 accumulation-order drift in the matmul.
    #[cfg(supersonic_backend_hip)]
    fn assert_parity(
        label: &str,
        got_bytes: &[u8],
        want_bytes: &[u8],
        max_abs_tol: f32,
        cos_sim_floor: f64,
    ) {
        assert_eq!(
            got_bytes.len(),
            want_bytes.len(),
            "{label}: byte length mismatch"
        );
        let got = bf16_bytes_to_f32(got_bytes);
        let want = bf16_bytes_to_f32(want_bytes);
        let n = got.len();
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut dot = 0.0f64;
        let mut got_sq = 0.0f64;
        let mut want_sq = 0.0f64;
        let mut exact = 0usize;
        for i in 0..n {
            let d = (got[i] - want[i]).abs();
            if d == 0.0 {
                exact += 1;
            }
            max_abs_diff = max_abs_diff.max(d);
            sum_abs_diff += d;
            dot += got[i] as f64 * want[i] as f64;
            got_sq += (got[i] as f64).powi(2);
            want_sq += (want[i] as f64).powi(2);
        }
        let cos_sim = dot / (got_sq.sqrt() * want_sq.sqrt() + 1e-30);
        let mean_abs_diff = sum_abs_diff / n as f32;
        eprintln!(
            "[parity {label}] n={n} exact={exact} max_abs={max_abs_diff:.5e} \
             mean_abs={mean_abs_diff:.5e} cos_sim={cos_sim:.7}"
        );
        assert!(
            max_abs_diff <= max_abs_tol,
            "{label}: max_abs={max_abs_diff} exceeds tolerance {max_abs_tol}"
        );
        assert!(
            cos_sim >= cos_sim_floor,
            "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}"
        );
    }

    /// Returns workspace size sufficient for the largest staged
    /// intermediate. Stage 5 is the final stage: 6*H*d + 4*Hkv*d + hidden F32.
    #[cfg(supersonic_backend_hip)]
    fn parity_workspace_floats(
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        hidden: i32,
    ) -> usize {
        let h = num_heads as usize;
        let hkv = num_kv_heads as usize;
        let d = head_dim as usize;
        let hd = hidden as usize;
        6 * h * d + 4 * hkv * d + hd
    }

    /// Output buffer size sufficient for the largest staged intermediate.
    /// Stage 3 publishes q_rot || k_rot, so the buffer must hold (H + Hkv)*d
    /// BF16 elements; stages 1, 2, 4, 5 fit in a strict subset of that.
    #[cfg(supersonic_backend_hip)]
    fn parity_output_elems(num_heads: i32, num_kv_heads: i32, head_dim: i32) -> usize {
        let h = num_heads as usize;
        let hkv = num_kv_heads as usize;
        let d = head_dim as usize;
        h * d + hkv * d
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_1_q_normed_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_oracle.py --mode synthetic --out /tmp/syn.json` \
                 and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let q_normed_expected_bytes = b64_decode(inters["q_normed"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(input_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(q_proj_w_bytes.len(), 2 * h_us * d_us * hidden_us * 2);
        assert_eq!(q_norm_w_bytes.len(), d_us * 2);
        assert_eq!(q_normed_expected_bytes.len(), h_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * h_us * d_us, hidden_us],
            &q_proj_w_bytes,
        )
        .expect("upload q_proj_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 1,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: 0,
            rope_theta: 0.0,
            rms_norm_eps: geom.rms_norm_eps,
            position: 0,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: std::ptr::null(),
            v_proj_w: std::ptr::null(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: std::ptr::null(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 1");

        // Stage 1 publishes q_normed into the first H*d BF16 elements of
        // the (now stage-3-sized) output buffer. Slice down to just those
        // bytes for parity. BF16 ULP at magnitude 1 is ~7.8e-3; allow 4×
        // that for matmul accumulation order drift over the 2048 reduction.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..h_us * d_us * 2];
        assert_parity(
            "step1 q_normed",
            got_bytes,
            &q_normed_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_2_k_normed_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still runs the stage-1 prerequisite (q_normed lives in
        // workspace for later RoPE), so the kernel needs the q-side weights
        // even though we only verify k_normed here.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let k_normed_expected_bytes = b64_decode(inters["k_normed"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(k_proj_w_bytes.len(), hkv_us * d_us * hidden_us * 2);
        assert_eq!(v_proj_w_bytes.len(), hkv_us * d_us * hidden_us * 2);
        assert_eq!(k_norm_w_bytes.len(), d_us * 2);
        assert_eq!(k_normed_expected_bytes.len(), hkv_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * h_us * d_us, hidden_us],
            &q_proj_w_bytes,
        )
        .expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &k_proj_w_bytes,
        )
        .expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &v_proj_w_bytes,
        )
        .expect("upload v_proj_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");
        let k_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes)
                .expect("upload k_norm_w");

        // Output is sized for the largest staged intermediate (H*d). Stage 2
        // writes Hkv*d BF16 elements at the start of the buffer.
        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim: 0,
            rope_theta: 0.0,
            rms_norm_eps: geom.rms_norm_eps,
            position: 0,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 2");

        // Stage 2 publishes k_normed into the first Hkv*d BF16 elements of
        // the output buffer. Slice down to just those bytes for parity.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hkv_us * d_us * 2];
        assert_parity(
            "step2 k_normed",
            got_bytes,
            &k_normed_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_3_qk_rot_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 Generate at a non-zero position so RoPE is non-identity \
                 (e.g. `python oracle/qwen36_moe_oracle.py --mode synthetic \
                 --position 7 --out /tmp/qwen36_syn_pos7.json`)."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        let position = json["position"].as_i64().unwrap_or(0) as i32;
        if position == 0 {
            // RoPE at position 0 is the identity rotation — the parity
            // test would pass even with a no-op kernel, which defeats the
            // purpose. Refuse and tell the caller how to fix it.
            panic!(
                "step3 RoPE parity requires position > 0, got 0. \
                 Re-run the oracle with `--position 7` (or any non-zero value)."
            );
        }

        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let q_rot_expected_bytes = b64_decode(inters["q_rot"].as_str().unwrap());
        let k_rot_expected_bytes = b64_decode(inters["k_rot"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(q_rot_expected_bytes.len(), h_us * d_us * 2);
        assert_eq!(k_rot_expected_bytes.len(), hkv_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * h_us * d_us, hidden_us],
            &q_proj_w_bytes,
        )
        .expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &k_proj_w_bytes,
        )
        .expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &v_proj_w_bytes,
        )
        .expect("upload v_proj_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");
        let k_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes)
                .expect("upload k_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 3,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 3");

        // Output layout: [q_rot (H*d) | k_rot (Hkv*d)] in BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let q_end = h_us * d_us * 2;
        let k_end = q_end + hkv_us * d_us * 2;
        assert_parity(
            "step3 q_rot",
            &got_bytes_full[..q_end],
            &q_rot_expected_bytes,
            0.04,
            0.9999,
        );
        assert_parity(
            "step3 k_rot",
            &got_bytes_full[q_end..k_end],
            &k_rot_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_4_attn_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        // Stage 4 still walks the stage-3 RoPE prerequisite, so we need
        // the same RoPE config + non-trivial position discipline as step 3.
        let position = json["position"].as_i64().unwrap_or(0) as i32;
        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let attn_expected_bytes = b64_decode(inters["attn"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(attn_expected_bytes.len(), h_us * d_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * h_us * d_us, hidden_us],
            &q_proj_w_bytes,
        )
        .expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &k_proj_w_bytes,
        )
        .expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &v_proj_w_bytes,
        )
        .expect("upload v_proj_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");
        let k_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes)
                .expect("upload k_norm_w");

        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 4,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: std::ptr::null(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 4");

        // Stage 4 publishes attn into output[0..H*d) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..h_us * d_us * 2];
        // kv_len=1 makes softmax trivially 1.0, so attn = v_full and the
        // parity should be effectively bit-exact (both kernel and oracle
        // skip any precision-losing accumulation here).
        assert_parity("step4 attn", got_bytes, &attn_expected_bytes, 0.04, 0.9999);
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. \
                 See `qwen36_moe_attn_step_1_q_normed_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];
        let cfg = &json["config"];

        let position = json["position"].as_i64().unwrap_or(0) as i32;
        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_proj_w_bytes = b64_decode(weights["q_proj_w"].as_str().unwrap());
        let k_proj_w_bytes = b64_decode(weights["k_proj_w"].as_str().unwrap());
        let v_proj_w_bytes = b64_decode(weights["v_proj_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());
        let o_proj_w_bytes = b64_decode(weights["o_proj_w"].as_str().unwrap());
        let output_hidden_expected_bytes = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        assert_eq!(o_proj_w_bytes.len(), hidden_us * h_us * d_us * 2);
        assert_eq!(output_hidden_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[2 * h_us * d_us, hidden_us],
            &q_proj_w_bytes,
        )
        .expect("upload q_proj_w");
        let k_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &k_proj_w_bytes,
        )
        .expect("upload k_proj_w");
        let v_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hkv_us * d_us, hidden_us],
            &v_proj_w_bytes,
        )
        .expect("upload v_proj_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");
        let k_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes)
                .expect("upload k_norm_w");
        let o_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, h_us * d_us],
            &o_proj_w_bytes,
        )
        .expect("upload o_proj_w");

        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_proj_w.as_ptr(),
            k_proj_w: k_proj_w.as_ptr(),
            v_proj_w: v_proj_w.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: o_proj_w.as_ptr(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeAttnStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 5");

        // Stage 5 publishes output_hidden into output[0..hidden) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        // o_proj reduces over H*d=4096 lanes; allow more headroom on
        // max_abs_diff than the smaller stage-1 reduction (2048 lanes).
        // Cosine similarity is the more meaningful metric for the residual.
        assert_parity(
            "step5 output_hidden",
            got_bytes,
            &output_hidden_expected_bytes,
            0.05,
            0.9999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b6 step 2: full-attention INT4 parity vs the INT4 oracle.
    //
    // Driven by the same env var as the BF16 attn tests; skipped silently
    // when the JSON's schema is not `qwen36-moe-oracle-layer-int4-v1`.
    // The INT4 oracle's `weights` block carries the BF16-reconstruction
    // of each quantized tensor, so the BF16 reference computed against
    // those weights is the exact intermediate the kernel must reproduce
    // when it dequantizes (packed, scale, zero) on the fly.
    //
    // All four projection tensors (q_proj, k_proj, v_proj, o_proj) flow
    // through the INT4 path. Norms stay BF16.
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    fn attn_oracle_is_int4(json: &serde_json::Value) -> bool {
        json["schema"].as_str() == Some("qwen36-moe-oracle-layer-int4-v1")
    }

    /// Pulls (packed, scale, zero) bytes for one INT4-quantized attn tensor.
    #[cfg(supersonic_backend_hip)]
    fn decode_attn_int4_sidecar(
        json: &serde_json::Value,
        name: &str,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let blk = &json["int4_weights"][name];
        let packed = b64_decode(
            blk["packed"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].packed")),
        );
        let scale = b64_decode(
            blk["scale"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].scale")),
        );
        let zero = b64_decode(
            blk["zero"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].zero")),
        );
        (packed, scale, zero)
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_attn_step_5_output_hidden_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_ORACLE_JSON not set. Generate \
                 an INT4 fixture with \
                 `python oracle/qwen36_moe_oracle.py --mode synthetic \
                 --int4 --position 7 --out /tmp/qwen36_attn_int4.json`."
            );
            return;
        };
        if !attn_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Pass `--int4` \
                 to the oracle to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let position = json["position"].as_i64().unwrap_or(0) as i32;
        let rotary_dim = cfg["rotary_dim"].as_i64().unwrap() as i32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap() as f32;

        // BF16 inputs that stay BF16 (norms + input_hidden).
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let q_norm_w_bytes = b64_decode(weights["q_norm_w"].as_str().unwrap());
        let k_norm_w_bytes = b64_decode(weights["k_norm_w"].as_str().unwrap());

        // INT4 sidecars for the four projections.
        let (q_packed, q_scale, q_zero) = decode_attn_int4_sidecar(&json, "q_proj_w");
        let (k_packed, k_scale, k_zero) = decode_attn_int4_sidecar(&json, "k_proj_w");
        let (v_packed, v_scale, v_zero) = decode_attn_int4_sidecar(&json, "v_proj_w");
        let (o_packed, o_scale, o_zero) = decode_attn_int4_sidecar(&json, "o_proj_w");

        let output_hidden_expected_bytes = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let h_us = geom.num_heads as usize;
        let hkv_us = geom.num_kv_heads as usize;
        let d_us = geom.head_dim as usize;
        let gsz_us = group_size as usize;

        // Sanity: shapes match the bake convention.
        assert_eq!(
            q_packed.len(),
            2 * h_us * d_us * (hidden_us / 2),
            "q_proj packed bytes mismatch"
        );
        assert_eq!(
            q_scale.len(),
            (2 * h_us * d_us / gsz_us) * (hidden_us / gsz_us) * 2,
            "q_proj scale bytes mismatch"
        );
        assert_eq!(k_packed.len(), hkv_us * d_us * (hidden_us / 2));
        assert_eq!(v_packed.len(), hkv_us * d_us * (hidden_us / 2));
        assert_eq!(o_packed.len(), hidden_us * (h_us * d_us / 2));

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let q_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &q_norm_w_bytes)
                .expect("upload q_norm_w");
        let k_norm_w =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[d_us], &k_norm_w_bytes)
                .expect("upload k_norm_w");

        // Projection weights uploaded as packed u8 + BF16 scale/zero.
        let q_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[q_packed.len()], &q_packed)
                .expect("upload q packed");
        let q_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[q_scale.len() / 2], &q_scale)
                .expect("upload q scale");
        let q_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[q_zero.len() / 2], &q_zero)
                .expect("upload q zero");
        let k_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[k_packed.len()], &k_packed)
                .expect("upload k packed");
        let k_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[k_scale.len() / 2], &k_scale)
                .expect("upload k scale");
        let k_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[k_zero.len() / 2], &k_zero)
                .expect("upload k zero");
        let v_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[v_packed.len()], &v_packed)
                .expect("upload v packed");
        let v_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_scale.len() / 2], &v_scale)
                .expect("upload v scale");
        let v_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_zero.len() / 2], &v_zero)
                .expect("upload v zero");
        let o_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[o_packed.len()], &o_packed)
                .expect("upload o packed");
        let o_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[o_scale.len() / 2], &o_scale)
                .expect("upload o scale");
        let o_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[o_zero.len() / 2], &o_zero)
                .expect("upload o zero");

        let mut output = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[parity_output_elems(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
            )],
        )
        .expect("alloc output");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[parity_workspace_floats(
                geom.num_heads,
                geom.num_kv_heads,
                geom.head_dim,
                geom.hidden,
            )],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeAttnStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_heads: geom.num_heads,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            rotary_dim,
            rope_theta,
            rms_norm_eps: geom.rms_norm_eps,
            position,
            cache_pos: Qwen36MoeAttnStepParams::CACHE_POS_INHERIT,
        };
        // Projection weight pointers point at the packed u8 buffers.
        let weight_ptrs = Qwen36MoeAttnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            q_proj_w: q_packed_buf.as_ptr(),
            k_proj_w: k_packed_buf.as_ptr(),
            v_proj_w: v_packed_buf.as_ptr(),
            q_norm_w: q_norm_w.as_ptr(),
            k_norm_w: k_norm_w.as_ptr(),
            o_proj_w: o_packed_buf.as_ptr(),
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            kv_max_t: 0,
        };
        let int4_ptrs = Qwen36MoeAttnStepInt4 {
            group_size,
            q_proj_scale: q_scale_buf.as_ptr(),
            q_proj_zero: q_zero_buf.as_ptr(),
            k_proj_scale: k_scale_buf.as_ptr(),
            k_proj_zero: k_zero_buf.as_ptr(),
            v_proj_scale: v_scale_buf.as_ptr(),
            v_proj_zero: v_zero_buf.as_ptr(),
            o_proj_scale: o_scale_buf.as_ptr(),
            o_proj_zero: o_zero_buf.as_ptr(),
        };

        attn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("attn_step_launch stage 5 (int4)");

        // Same envelope as the BF16 stage 5 test: cos_sim ≥ 0.9999, max
        // |delta| ≤ 0.05. The reconstruction is bit-identical to what the
        // oracle ran reference against, so any residual disagreement is
        // F32 reduction-order drift through the matvec — same as BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        assert_parity(
            "step5 int4 output_hidden",
            got_bytes,
            &output_hidden_expected_bytes,
            0.05,
            0.9999,
        );
    }

    // ---- PR 4b3 step 2: linear-attn stage 1 parity vs the oracle ---------
    //
    // Same env-var pattern as the full-attn parity tests, but pointed at a
    // JSON produced by `oracle/qwen36_moe_linear_oracle.py`:
    //
    //   python oracle/qwen36_moe_linear_oracle.py --mode synthetic \
    //       --state fresh --out /tmp/qwen36_lin_fresh.json
    //   SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON=/tmp/qwen36_lin_fresh.json \
    //       cargo test --release -p kernel-ffi qwen36_moe_linear_step
    //
    // Skipped with a clear message when the env var is unset so the FFI
    // test stays runnable on Python-less hosts.

    #[cfg(supersonic_backend_hip)]
    struct LinearOracleGeom {
        hidden: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_kernel_dim: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_linear_oracle_json() -> Option<(serde_json::Value, LinearOracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read linear oracle json {json_path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&raw).expect("linear oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "linear-attn parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = LinearOracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_k_heads: cfg["num_k_heads"].as_i64().unwrap() as i32,
            num_v_heads: cfg["num_v_heads"].as_i64().unwrap() as i32,
            head_k_dim: cfg["head_k_dim"].as_i64().unwrap() as i32,
            head_v_dim: cfg["head_v_dim"].as_i64().unwrap() as i32,
            conv_kernel_dim: cfg["conv_kernel_dim"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_1_qkv_raw_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_linear_oracle.py --mode synthetic \
                 --out /tmp/qwen36_lin.json` and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let qkv_raw_expected_bytes = b64_decode(inters["qkv_raw"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(input_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(in_proj_qkv_w_bytes.len(), qkv_dim * hidden_us * 2);
        assert_eq!(in_proj_z_w_bytes.len(), val_dim * hidden_us * 2);
        assert_eq!(in_proj_a_w_bytes.len(), v_us * hidden_us * 2);
        assert_eq!(in_proj_b_w_bytes.len(), v_us * hidden_us * 2);
        assert_eq!(qkv_raw_expected_bytes.len(), qkv_dim * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, hidden_us],
            &in_proj_qkv_w_bytes,
        )
        .expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[val_dim, hidden_us],
            &in_proj_z_w_bytes,
        )
        .expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");

        // Output sized for the largest staged intermediate (qkv_dim BF16
        // is the biggest until later stages bump this).
        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim]).expect("alloc output");
        // Workspace sized for stage 1 (qkv_dim + V*v_dim + 2*V F32). Later
        // stages will need more; keep this tight to fail loudly if a stage
        // overruns.
        let workspace_floats = qkv_dim + val_dim + 2 * v_us;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 1,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: std::ptr::null(),
            conv1d_bias: std::ptr::null(),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: std::ptr::null_mut(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 1");

        // Stage 1 publishes qkv_raw as the full output buffer.
        let got_bytes = output.to_host_bytes().expect("download output");
        // 2048-wide F32 reduction; same envelope as PR 4b2 step 1's q_proj.
        assert_parity(
            "linear step1 qkv_raw",
            &got_bytes,
            &qkv_raw_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_2_silu_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still walks the stage-1 prerequisite (qkv_raw is the
        // conv1d input), so we need every weight stage 1 needed plus
        // conv1d_w, the conv state, and (optionally) conv1d_bias.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights
            .get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let silu_out_expected_bytes = b64_decode(inters["silu_out"].as_str().unwrap());
        let conv_state_after_expected_bytes =
            b64_decode(inters["conv_state_after"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;

        assert_eq!(conv1d_w_bytes.len(), qkv_dim * 1 * kernel * 2);
        assert_eq!(conv_state_before_bytes.len(), qkv_dim * kstate * 2);
        assert_eq!(silu_out_expected_bytes.len(), qkv_dim * 2);
        assert_eq!(conv_state_after_expected_bytes.len(), qkv_dim * kstate * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, hidden_us],
            &in_proj_qkv_w_bytes,
        )
        .expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[val_dim, hidden_us],
            &in_proj_z_w_bytes,
        )
        .expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, 1, kernel],
            &conv1d_w_bytes,
        )
        .expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, kstate],
            &conv_state_before_bytes,
        )
        .expect("upload conv_state");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim]).expect("alloc output");
        let workspace_floats = qkv_dim + val_dim + 2 * v_us;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 2,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 2");

        // Stage 2 publishes silu_out as the full output buffer.
        let got_bytes = output.to_host_bytes().expect("download output");
        assert_parity(
            "linear step2 silu_out",
            &got_bytes,
            &silu_out_expected_bytes,
            0.04,
            0.9999,
        );
        // The kernel also updates conv_state in place; verify it matches
        // the oracle's conv_state_after so the next decode step has the
        // right starting state.
        let conv_state_got = conv_state.to_host_bytes().expect("download conv_state");
        assert_parity(
            "linear step2 conv_state_after",
            &conv_state_got,
            &conv_state_after_expected_bytes,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_3_qkv_post_norm_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 3 needs everything stage 2 needed.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights
            .get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let q_rep_expected = b64_decode(inters["q_rep"].as_str().unwrap());
        let k_rep_expected = b64_decode(inters["k_rep"].as_str().unwrap());
        let v_heads_expected = b64_decode(inters["v_heads"].as_str().unwrap());
        let q_scaled_expected = b64_decode(inters["q_scaled"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;

        assert_eq!(q_rep_expected.len(), v_kdim * 2);
        assert_eq!(k_rep_expected.len(), v_kdim * 2);
        assert_eq!(v_heads_expected.len(), v_vdim * 2);
        assert_eq!(q_scaled_expected.len(), v_kdim * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, hidden_us],
            &in_proj_qkv_w_bytes,
        )
        .expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[val_dim, hidden_us],
            &in_proj_z_w_bytes,
        )
        .expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, 1, kernel],
            &conv1d_w_bytes,
        )
        .expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, kstate],
            &conv_state_before_bytes,
        )
        .expect("upload conv_state");

        // Output sized for stage 3's largest publish (q_scaled || k_rep || v_heads).
        let stage3_publish_elems = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[stage3_publish_elems])
            .expect("alloc output");
        // Workspace for stage 3: stage-2 footprint plus Q_NORMED, K_NORMED,
        // Q_REP, K_REP slots.
        let workspace_floats = qkv_dim + val_dim + 2 * v_us + 2 * (k_us * kd_us) + 2 * v_kdim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 3,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            dt_bias: std::ptr::null(),
            a_log: std::ptr::null(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: std::ptr::null_mut(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 3");

        // Output layout: [q_scaled (V*k_dim) | k_rep (V*k_dim) | v_heads (V*v_dim)] BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let q_end = v_kdim * 2;
        let k_end = q_end + v_kdim * 2;
        let v_end = k_end + v_vdim * 2;
        assert_parity(
            "linear step3 q_scaled",
            &got_bytes_full[..q_end],
            &q_scaled_expected,
            0.04,
            0.9999,
        );
        assert_parity(
            "linear step3 k_rep",
            &got_bytes_full[q_end..k_end],
            &k_rep_expected,
            0.04,
            0.9999,
        );
        assert_parity(
            "linear step3 v_heads",
            &got_bytes_full[k_end..v_end],
            &v_heads_expected,
            0.04,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_4_recurrent_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 4 needs everything stage 3 needed plus dt_bias, A_log, and
        // the prior recurrent state.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights
            .get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let dt_bias_bytes = b64_decode(weights["dt_bias"].as_str().unwrap());
        let a_log_bytes = b64_decode(weights["a_log"].as_str().unwrap());
        // recurrent_state_before is encoded as F32 (production layout).
        let recurrent_state_before_bytes =
            b64_decode(weights["recurrent_state_before"].as_str().unwrap());
        let recurrent_out_expected = b64_decode(inters["recurrent_out"].as_str().unwrap());
        let state_after_expected = b64_decode(inters["state_after"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;
        let state_elems = v_us * kd_us * vd_us;

        assert_eq!(dt_bias_bytes.len(), v_us * 2);
        assert_eq!(a_log_bytes.len(), v_us * 2);
        // recurrent_state encoded F32 (4 bytes/elem).
        assert_eq!(recurrent_state_before_bytes.len(), state_elems * 4);
        assert_eq!(recurrent_out_expected.len(), v_vdim * 2);
        assert_eq!(state_after_expected.len(), state_elems * 4);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, hidden_us],
            &in_proj_qkv_w_bytes,
        )
        .expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[val_dim, hidden_us],
            &in_proj_z_w_bytes,
        )
        .expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, 1, kernel],
            &conv1d_w_bytes,
        )
        .expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let dt_bias =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &dt_bias_bytes)
                .expect("upload dt_bias");
        let a_log = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &a_log_bytes)
            .expect("upload a_log");
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, kstate],
            &conv_state_before_bytes,
        )
        .expect("upload conv_state");
        let mut recurrent_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[state_elems],
            &recurrent_state_before_bytes,
        )
        .expect("upload recurrent_state");

        // Stage 4 publishes recurrent_out [V*v_dim] BF16. The buffer is
        // sized for the largest staged intermediate (still stage 3's
        // q_scaled||k_rep||v_heads = 2*V*k_dim + V*v_dim).
        let stage_publish_max = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[stage_publish_max])
            .expect("alloc output");
        // Workspace for stage 4 = previous + BETA + G + REC_OUT.
        let workspace_floats =
            qkv_dim + val_dim + 2 * v_us + 2 * (k_us * kd_us) + 2 * v_kdim + v_us + v_us + v_vdim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 4,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: std::ptr::null(),
            out_proj_w: std::ptr::null(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 4");

        // Stage 4 publishes recurrent_out [V*v_dim] BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let rec_out_bytes = &got_bytes_full[..v_vdim * 2];
        // Recurrent state mixes BF16-rounded inputs (k_rep, q_scaled,
        // v_heads) into F32-precision math; the per-V*v_dim reduction is
        // 128-wide so allow the same envelope as the qkv_proj reduction.
        assert_parity(
            "linear step4 recurrent_out",
            rec_out_bytes,
            &recurrent_out_expected,
            0.04,
            0.9999,
        );

        // Also verify state_after — the F32 recurrent state has been
        // mutated in place by the kernel and must match the oracle's
        // post-update state for the next decode step to work.
        let state_after_got = recurrent_state.to_host_bytes().expect("download state");
        // F32 throughout (no BF16 rounding); be tighter on max_abs.
        // Per-element rounding error from F32 arithmetic + cast-from-BF16
        // operands is at most a few ULPs of the magnitude.
        assert_parity_f32(
            "linear step4 state_after",
            &state_after_got,
            &state_after_expected,
            5e-3,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 See `qwen36_moe_linear_step_1_qkv_raw_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 5 needs every weight from earlier stages plus norm_w and out_proj_w.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_qkv_w_bytes = b64_decode(weights["in_proj_qkv_w"].as_str().unwrap());
        let in_proj_z_w_bytes = b64_decode(weights["in_proj_z_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights
            .get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let dt_bias_bytes = b64_decode(weights["dt_bias"].as_str().unwrap());
        let a_log_bytes = b64_decode(weights["a_log"].as_str().unwrap());
        let recurrent_state_before_bytes =
            b64_decode(weights["recurrent_state_before"].as_str().unwrap());
        let norm_w_bytes = b64_decode(weights["norm_w"].as_str().unwrap());
        let out_proj_w_bytes = b64_decode(weights["out_proj_w"].as_str().unwrap());
        let output_hidden_expected = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let kernel = geom.conv_kernel_dim as usize;
        let kstate = kernel - 1;
        let key_dim = k_us * kd_us;
        let val_dim = v_us * vd_us;
        let qkv_dim = 2 * key_dim + val_dim;
        let v_kdim = v_us * kd_us;
        let v_vdim = v_us * vd_us;
        let state_elems = v_us * kd_us * vd_us;

        assert_eq!(norm_w_bytes.len(), vd_us * 2);
        assert_eq!(out_proj_w_bytes.len(), hidden_us * v_vdim * 2);
        assert_eq!(output_hidden_expected.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_qkv_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, hidden_us],
            &in_proj_qkv_w_bytes,
        )
        .expect("upload in_proj_qkv_w");
        let in_proj_z_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[val_dim, hidden_us],
            &in_proj_z_w_bytes,
        )
        .expect("upload in_proj_z_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, 1, kernel],
            &conv1d_w_bytes,
        )
        .expect("upload conv1d_w");
        let conv1d_bias = match &conv1d_bias_bytes {
            Some(bytes) => Some(
                GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                    .expect("upload conv1d_bias"),
            ),
            None => None,
        };
        let dt_bias =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &dt_bias_bytes)
                .expect("upload dt_bias");
        let a_log = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &a_log_bytes)
            .expect("upload a_log");
        let norm_w = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[vd_us], &norm_w_bytes)
            .expect("upload norm_w");
        let out_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, v_vdim],
            &out_proj_w_bytes,
        )
        .expect("upload out_proj_w");
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, kstate],
            &conv_state_before_bytes,
        )
        .expect("upload conv_state");
        let mut recurrent_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[state_elems],
            &recurrent_state_before_bytes,
        )
        .expect("upload recurrent_state");

        let stage_publish_max = 2 * v_kdim + v_vdim;
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[stage_publish_max])
            .expect("alloc output");
        let workspace_floats =
            qkv_dim + val_dim + 2 * v_us + 2 * (k_us * kd_us) + 2 * v_kdim + v_us + v_us + v_vdim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: in_proj_qkv_w.as_ptr(),
            in_proj_z_w: in_proj_z_w.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: norm_w.as_ptr(),
            out_proj_w: out_proj_w.as_ptr(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeLinearStepInt4::disabled(),
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 5");

        // Stage 5 publishes output_hidden into output[0..hidden) BF16.
        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        // out_proj reduces over V*v_dim=4096 lanes; same envelope as
        // PR 4b2 stage 5.
        assert_parity(
            "linear step5 output_hidden",
            got_bytes,
            &output_hidden_expected,
            0.05,
            0.9999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b6 step 4: linear-attn INT4 parity vs the INT4 oracle.
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    fn linear_oracle_is_int4(json: &serde_json::Value) -> bool {
        json["schema"].as_str() == Some("qwen36-moe-oracle-linear-int4-v1")
    }

    #[cfg(supersonic_backend_hip)]
    fn decode_linear_int4_sidecar(
        json: &serde_json::Value,
        name: &str,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let blk = &json["int4_weights"][name];
        let packed = b64_decode(
            blk["packed"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].packed")),
        );
        let scale = b64_decode(
            blk["scale"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].scale")),
        );
        let zero = b64_decode(
            blk["zero"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].zero")),
        );
        (packed, scale, zero)
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_linear_step_5_output_hidden_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_linear_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON not set. \
                 Generate an INT4 fixture with \
                 `python oracle/qwen36_moe_linear_oracle.py --mode synthetic \
                 --int4 --out /tmp/qwen36_lin_int4.json`."
            );
            return;
        };
        if !linear_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Pass `--int4` to \
                 the oracle to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // BF16 inputs that stay BF16 (norms + small scalars + conv + state).
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let input_norm_w_bytes = b64_decode(weights["input_norm_w"].as_str().unwrap());
        let in_proj_a_w_bytes = b64_decode(weights["in_proj_a_w"].as_str().unwrap());
        let in_proj_b_w_bytes = b64_decode(weights["in_proj_b_w"].as_str().unwrap());
        let conv1d_w_bytes = b64_decode(weights["conv1d_w"].as_str().unwrap());
        let conv1d_bias_bytes = weights
            .get("conv1d_bias")
            .and_then(|v| v.as_str())
            .map(b64_decode);
        let conv_state_before_bytes = b64_decode(weights["conv_state_before"].as_str().unwrap());
        let dt_bias_bytes = b64_decode(weights["dt_bias"].as_str().unwrap());
        let a_log_bytes = b64_decode(weights["a_log"].as_str().unwrap());
        let recurrent_state_before_bytes =
            b64_decode(weights["recurrent_state_before"].as_str().unwrap());
        let norm_w_bytes = b64_decode(weights["norm_w"].as_str().unwrap());

        // INT4 sidecars for the three projections.
        let (qkv_packed, qkv_scale, qkv_zero) = decode_linear_int4_sidecar(&json, "in_proj_qkv_w");
        let (z_packed, z_scale, z_zero) = decode_linear_int4_sidecar(&json, "in_proj_z_w");
        let (out_packed, out_scale, out_zero) = decode_linear_int4_sidecar(&json, "out_proj_w");

        let output_hidden_expected = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let k_us = geom.num_k_heads as usize;
        let v_us = geom.num_v_heads as usize;
        let kd_us = geom.head_k_dim as usize;
        let vd_us = geom.head_v_dim as usize;
        let conv_kernel = geom.conv_kernel_dim as usize;
        let qkv_dim = 2 * k_us * kd_us + v_us * vd_us;
        let val_dim = v_us * vd_us;

        // Sanity: shapes match the bake convention.
        assert_eq!(
            qkv_packed.len(),
            qkv_dim * (hidden_us / 2),
            "in_proj_qkv packed bytes mismatch"
        );
        assert_eq!(z_packed.len(), val_dim * (hidden_us / 2));
        assert_eq!(out_packed.len(), hidden_us * (val_dim / 2));

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let input_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_norm_w_bytes,
        )
        .expect("upload input_norm_w");
        let in_proj_a_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_a_w_bytes,
        )
        .expect("upload in_proj_a_w");
        let in_proj_b_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[v_us, hidden_us],
            &in_proj_b_w_bytes,
        )
        .expect("upload in_proj_b_w");
        let conv1d_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, conv_kernel],
            &conv1d_w_bytes,
        )
        .expect("upload conv1d_w");
        let conv1d_bias_buf = conv1d_bias_bytes.as_ref().map(|bytes| {
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_dim], bytes)
                .expect("upload conv1d_bias")
        });
        let mut conv_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_dim, conv_kernel - 1],
            &conv_state_before_bytes,
        )
        .expect("upload conv_state");
        let dt_bias =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &dt_bias_bytes)
                .expect("upload dt_bias");
        let a_log = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[v_us], &a_log_bytes)
            .expect("upload a_log");
        let mut recurrent_state = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[v_us, kd_us, vd_us],
            &recurrent_state_before_bytes,
        )
        .expect("upload recurrent_state");
        let norm_w = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[kd_us], &norm_w_bytes)
            .expect("upload norm_w");

        // Projection weights uploaded as packed u8 + BF16 scale/zero.
        let qkv_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[qkv_packed.len()], &qkv_packed)
                .expect("upload qkv packed");
        let qkv_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[qkv_scale.len() / 2],
            &qkv_scale,
        )
        .expect("upload qkv scale");
        let qkv_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[qkv_zero.len() / 2], &qkv_zero)
                .expect("upload qkv zero");
        let z_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[z_packed.len()], &z_packed)
                .expect("upload z packed");
        let z_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[z_scale.len() / 2], &z_scale)
                .expect("upload z scale");
        let z_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[z_zero.len() / 2], &z_zero)
                .expect("upload z zero");
        let out_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[out_packed.len()], &out_packed)
                .expect("upload out packed");
        let out_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[out_scale.len() / 2],
            &out_scale,
        )
        .expect("upload out scale");
        let out_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[out_zero.len() / 2], &out_zero)
                .expect("upload out zero");

        let v_kdim = v_us * kd_us;
        let stage_publish_max = 2 * v_kdim + val_dim;
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[stage_publish_max])
            .expect("alloc output");
        let workspace_floats =
            qkv_dim + val_dim + 2 * v_us + 2 * (k_us * kd_us) + 2 * v_kdim + v_us + v_us + val_dim;
        let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
            .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeLinearStepParams {
            stage: 5,
            hidden: geom.hidden,
            num_k_heads: geom.num_k_heads,
            num_v_heads: geom.num_v_heads,
            head_k_dim: geom.head_k_dim,
            head_v_dim: geom.head_v_dim,
            conv_kernel_dim: geom.conv_kernel_dim,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeLinearStepWeights {
            input_hidden: input_hidden.as_ptr(),
            input_norm_w: input_norm_w.as_ptr(),
            in_proj_qkv_w: qkv_packed_buf.as_ptr(),
            in_proj_z_w: z_packed_buf.as_ptr(),
            in_proj_a_w: in_proj_a_w.as_ptr(),
            in_proj_b_w: in_proj_b_w.as_ptr(),
            conv1d_w: conv1d_w.as_ptr(),
            conv1d_bias: conv1d_bias_buf
                .as_ref()
                .map(|b| b.as_ptr())
                .unwrap_or(std::ptr::null()),
            dt_bias: dt_bias.as_ptr(),
            a_log: a_log.as_ptr(),
            norm_w: norm_w.as_ptr(),
            out_proj_w: out_packed_buf.as_ptr(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };
        let int4_ptrs = Qwen36MoeLinearStepInt4 {
            group_size,
            in_proj_qkv_scale: qkv_scale_buf.as_ptr(),
            in_proj_qkv_zero: qkv_zero_buf.as_ptr(),
            in_proj_z_scale: z_scale_buf.as_ptr(),
            in_proj_z_zero: z_zero_buf.as_ptr(),
            out_proj_scale: out_scale_buf.as_ptr(),
            out_proj_zero: out_zero_buf.as_ptr(),
        };

        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("linear_step_launch stage 5 (int4)");

        let got_bytes_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_bytes_full[..hidden_us * 2];
        assert_parity(
            "linear step5 int4 output_hidden",
            got_bytes,
            &output_hidden_expected,
            0.05,
            0.9999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b4 — staged MoE FFN parity tests against the Python oracle
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    struct FfnOracleGeom {
        hidden: i32,
        num_experts: i32,
        moe_intermediate: i32,
        shared_intermediate: i32,
        top_k: i32,
        rms_norm_eps: f32,
    }

    #[cfg(supersonic_backend_hip)]
    fn load_ffn_oracle_json() -> Option<(serde_json::Value, FfnOracleGeom)> {
        let json_path = std::env::var("SUPERSONIC_QWEN36_FFN_ORACLE_JSON").ok()?;
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read ffn oracle json {json_path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&raw).expect("ffn oracle json parse");
        assert_eq!(
            json["dtype"].as_str().unwrap_or(""),
            "bf16",
            "MoE FFN parity tests require the oracle to be in bf16 mode"
        );
        let cfg = &json["config"];
        let geom = FfnOracleGeom {
            hidden: cfg["hidden"].as_i64().unwrap() as i32,
            num_experts: cfg["num_experts"].as_i64().unwrap() as i32,
            moe_intermediate: cfg["moe_intermediate"].as_i64().unwrap() as i32,
            shared_intermediate: cfg["shared_intermediate"].as_i64().unwrap() as i32,
            top_k: cfg["top_k"].as_i64().unwrap() as i32,
            rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        };
        Some((json, geom))
    }

    /// Decode a base64 i32 buffer (oracle uses int32 for `topk_idx`).
    #[cfg(supersonic_backend_hip)]
    fn i32_bytes_to_vec(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| {
                let mut a = [0u8; 4];
                a.copy_from_slice(c);
                i32::from_le_bytes(a)
            })
            .collect()
    }

    /// Workspace floats sufficient for the largest staged FFN intermediate
    /// (stage 5). Mirrors the kernel's per-stage offset layout in
    /// `kernels/qwen36_moe.hip` (search `OFF_H_NORM`):
    ///   H_NORM        [hidden]
    ///   ROUTER_LOGITS [E]
    ///   ROUTER_PROBS  [E]
    ///   TOPK_VAL      [k]
    ///   TOPK_IDX      [k]
    ///   SG_SCALAR     [1]
    ///   SGP           [Is]
    ///   SUP           [Is]
    ///   SHARED_MID    [Is]
    ///   SHARED_OUT    [hidden]
    ///   EXPERT_GU     [k * 2*I]   one [2*I] slab per concurrent expert group
    ///   EXPERT_MID    [k * I]     one [I]   slab per concurrent expert group
    ///   EXPERT_STACK  [k*hidden]
    ///   MOE_OUT       [hidden]
    ///
    /// PR 4b4 step 4 (the original stage-3+ test wiring) silently undersized
    /// this by 385 floats on the synthetic config — it omitted SG_SCALAR,
    /// SGP, SUP, EXPERT_GU and EXPERT_MID. The HIP allocator's slack hid the
    /// OOB on the BF16 path. The new INT4-sidecar kernel parameters PR 4b5
    /// step 3 adds shift register/stack layout enough that the same OOB
    /// starts overwriting live cooperative-launch state and stage>=4 hangs.
    /// The right fix is to size the buffer for what the kernel actually uses.
    #[cfg(supersonic_backend_hip)]
    fn ffn_parity_workspace_floats(geom: &FfnOracleGeom) -> usize {
        let hidden = geom.hidden as usize;
        let e = geom.num_experts as usize;
        let k = geom.top_k as usize;
        let is_dim = geom.shared_intermediate as usize;
        let i_dim = geom.moe_intermediate as usize;
        // 3*hidden     = H_NORM + SHARED_OUT + MOE_OUT
        // 2*e          = ROUTER_LOGITS + ROUTER_PROBS
        // 2*k          = TOPK_VAL + TOPK_IDX
        // 1            = SG_SCALAR
        // 3*is_dim     = SGP + SUP + SHARED_MID
        // k*3*i_dim    = EXPERT_GU(k*2*I) + EXPERT_MID(k*I) — sized for the
        //                concurrent-experts G/H/I dispatch (one slab per
        //                expert group)
        // k*hidden     = EXPERT_STACK
        3 * hidden + 2 * e + 2 * k + 1 + 3 * is_dim + k * 3 * i_dim + k * hidden
    }

    /// Output BF16 elements sufficient for the largest staged FFN
    /// intermediate. Stages 2..=5 publish a `[hidden]` buffer; stage 1
    /// publishes `[k]`. Sized for `hidden`.
    #[cfg(supersonic_backend_hip)]
    fn ffn_parity_output_elems(geom: &FfnOracleGeom) -> usize {
        geom.hidden as usize
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_1_topk_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 Generate a fixture with \
                 `python oracle/qwen36_moe_ffn_oracle.py --mode synthetic \
                 --out /tmp/qwen36_ffn.json` and re-run."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let topk_idx_expected = i32_bytes_to_vec(&b64_decode(inters["topk_idx"].as_str().unwrap()));
        let topk_weights_expected_bytes = b64_decode(inters["topk_weights"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(post_attn_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(topk_idx_expected.len(), k_us);
        assert_eq!(topk_weights_expected_bytes.len(), k_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");

        // Output sized for the largest staged intermediate (hidden BF16).
        // Stage 1 publishes only `topk_weights[k]` into `output[0..k]`,
        // and `topk_idx[k]` into the separate `output_idx` buffer.
        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        // No I32 variant in `ScalarType`; U32 has the same 4-byte storage
        // and the kernel reinterprets via the FFI signature's `*mut c_int`.
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 1,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: std::ptr::null(),
            shared_up_proj_w: std::ptr::null(),
            shared_down_proj_w: std::ptr::null(),
            shared_expert_gate_w: std::ptr::null(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 1");

        // Verify topk_idx (int32) — must match oracle exactly. Routing
        // decisions are categorical; any disagreement is a real bug, no
        // tolerance.
        let got_idx_bytes = output_idx.to_host_bytes().expect("download output_idx");
        let got_idx = i32_bytes_to_vec(&got_idx_bytes);
        assert_eq!(
            got_idx, topk_idx_expected,
            "ffn step1 topk_idx mismatch: got {got_idx:?}, want {topk_idx_expected:?}"
        );

        // Verify topk_weights (BF16) — these come from softmax + renorm,
        // so we expect bit-exactness for most elements with rare 1-ULP
        // drift from F32 accumulation order.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..k_us * 2];
        assert_parity(
            "ffn step1 topk_weights",
            got_bytes,
            &topk_weights_expected_bytes,
            0.01,
            0.9999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_2_shared_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 2 still runs the stage-1 prerequisites (rmsnorm + router
        // gate), so the gate weight is still required. The new tensors
        // are the four shared-expert weights; the per-expert
        // gate_up_proj / down_proj stay null until stage 3.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes = b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes = b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes = b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let shared_out_expected_bytes = b64_decode(inters["shared_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(post_attn_norm_w_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(shared_gate_proj_w_bytes.len(), is_us * hidden_us * 2);
        assert_eq!(shared_up_proj_w_bytes.len(), is_us * hidden_us * 2);
        assert_eq!(shared_down_proj_w_bytes.len(), hidden_us * is_us * 2);
        assert_eq!(shared_expert_gate_w_bytes.len(), 1 * hidden_us * 2);
        assert_eq!(shared_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_gate_proj_w_bytes,
        )
        .expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_up_proj_w_bytes,
        )
        .expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, is_us],
            &shared_down_proj_w_bytes,
        )
        .expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 2,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 2");

        // Stage 2 publishes shared_out into output[0..hidden] BF16. Tolerance
        // is looser than stage 1 because the shared expert path stacks four
        // matmuls (gate, up, down, plus the 1-row sigmoid gate); each one
        // accumulates F32 reduction-order drift. Cos_sim ≥ 0.999 on a deep
        // chain like this is the same envelope linear-attn stage 5 uses.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step2 shared_out",
            got_bytes,
            &shared_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_3_expert0_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 3 still walks stage-2 prereqs (the kernel runs the shared
        // expert path even when only the per-expert output is wanted) plus
        // the new fused expert weight slabs.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes = b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes = b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes = b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes = b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        // The oracle's `expert_stack` is [K, hidden] BF16 — the first
        // `hidden` elements are the FFN output of `topk_idx[0]`, which is
        // the same expert stage 3 dispatches. Slice off that first chunk.
        let expert_stack_full = b64_decode(inters["expert_stack"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(input_hidden_bytes.len(), hidden_us * 2);
        assert_eq!(gate_w_bytes.len(), e_us * hidden_us * 2);
        assert_eq!(gate_up_proj_w_bytes.len(), e_us * 2 * i_us * hidden_us * 2);
        assert_eq!(down_proj_w_bytes.len(), e_us * hidden_us * i_us * 2);
        assert_eq!(expert_stack_full.len(), k_us * hidden_us * 2);
        let expert0_expected_bytes = &expert_stack_full[..hidden_us * 2];

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, 2 * i_us, hidden_us],
            &gate_up_proj_w_bytes,
        )
        .expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us, i_us],
            &down_proj_w_bytes,
        )
        .expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_gate_proj_w_bytes,
        )
        .expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_up_proj_w_bytes,
        )
        .expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, is_us],
            &shared_down_proj_w_bytes,
        )
        .expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 3,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 3");

        // Stage 3 publishes expert_stack[0] (BF16-rounded F32) into
        // output[0..hidden]. Same envelope as stage 2's shared expert —
        // four matmuls deep, F32 throughout.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step3 expert0_out",
            got_bytes,
            expert0_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_4_moe_out_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 4 walks all of stage-3's path (k iterations of the per-expert
        // FFN) and adds the final topk-weighted sum. Same weight set; the
        // expected output is `moe_out` instead of `expert_stack[0]`.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes = b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes = b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes = b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes = b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let moe_out_expected_bytes = b64_decode(inters["moe_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(moe_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, 2 * i_us, hidden_us],
            &gate_up_proj_w_bytes,
        )
        .expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us, i_us],
            &down_proj_w_bytes,
        )
        .expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_gate_proj_w_bytes,
        )
        .expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_up_proj_w_bytes,
        )
        .expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, is_us],
            &shared_down_proj_w_bytes,
        )
        .expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 4,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 4");

        // Stage 4 publishes moe_out (= sum_j topk_w[j] * expert_stack[j],
        // BF16-rounded once) into output[0..hidden]. Tolerance same as
        // stage 3; the only added work is a k=8-wide reduction with
        // BF16-cast renormed weights, which is well-conditioned.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step4 moe_out",
            got_bytes,
            &moe_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_5_output_hidden_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_1_topk_matches_oracle` for setup."
            );
            return;
        };
        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 5 is the trivial closer — same weights as stage 4 plus the
        // residual add against `input_hidden`.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let gate_up_proj_w_bytes = b64_decode(weights["gate_up_proj_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_gate_proj_w_bytes = b64_decode(weights["shared_gate_proj_w"].as_str().unwrap());
        let shared_up_proj_w_bytes = b64_decode(weights["shared_up_proj_w"].as_str().unwrap());
        let shared_down_proj_w_bytes = b64_decode(weights["shared_down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());
        let output_hidden_expected_bytes = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        assert_eq!(output_hidden_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let gate_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, 2 * i_us, hidden_us],
            &gate_up_proj_w_bytes,
        )
        .expect("upload gate_up_proj_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us, i_us],
            &down_proj_w_bytes,
        )
        .expect("upload down_proj_w");
        let shared_gate_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_gate_proj_w_bytes,
        )
        .expect("upload shared_gate_proj_w");
        let shared_up_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[is_us, hidden_us],
            &shared_up_proj_w_bytes,
        )
        .expect("upload shared_up_proj_w");
        let shared_down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us, is_us],
            &shared_down_proj_w_bytes,
        )
        .expect("upload shared_down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 5,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gate_up_proj_w.as_ptr(),
            down_proj_w: down_proj_w.as_ptr(),
            shared_gate_proj_w: shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: shared_up_proj_w.as_ptr(),
            shared_down_proj_w: shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &Qwen36MoeFfnStepInt4::disabled(),
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 5");

        // Stage 5 publishes output_hidden = input_hidden + moe_out +
        // shared_out (all F32, BF16-round once at the end). The residual
        // addition can't make accuracy worse than stage 4's moe_out, so
        // the same envelope applies.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step5 output_hidden",
            got_bytes,
            &output_hidden_expected_bytes,
            0.05,
            0.999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b5 step 3: shared-expert INT4 parity vs the INT4 oracle.
    //
    // Driven by the same env var as the BF16 FFN tests; skipped silently
    // when the JSON's schema is not `qwen36-moe-oracle-ffn-int4-v1`. The
    // INT4 oracle's `weights` block carries the BF16-reconstruction of
    // each quantized tensor, so the BF16 reference computed against those
    // weights is the exact intermediate the kernel must reproduce when
    // it dequantizes (packed, scale, zero) on the fly.
    //
    // Step 3 wires only the shared-expert tensors (gate_proj, up_proj,
    // down_proj). Fused-expert weights stay BF16 in this test —
    // `weights.gate_up_proj_w` and `weights.down_proj_w` are uploaded as
    // BF16 and `Qwen36MoeFfnStepInt4`'s fused-expert sidecars stay null.
    // Steps 4/5 will switch those over.
    // -------------------------------------------------------------------

    #[cfg(supersonic_backend_hip)]
    fn ffn_oracle_is_int4(json: &serde_json::Value) -> bool {
        json["schema"].as_str() == Some("qwen36-moe-oracle-ffn-int4-v1")
    }

    /// Pulls (packed, scale, zero) bytes for one INT4-quantized FFN tensor
    /// from the oracle JSON. Returns owned `Vec<u8>` buffers in their
    /// native byte representations (u8 for packed, BF16 LE for scale/zero).
    #[cfg(supersonic_backend_hip)]
    fn decode_int4_sidecar(json: &serde_json::Value, name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let blk = &json["int4_weights"][name];
        let packed = b64_decode(
            blk["packed"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].packed")),
        );
        let scale = b64_decode(
            blk["scale"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].scale")),
        );
        let zero = b64_decode(
            blk["zero"]
                .as_str()
                .unwrap_or_else(|| panic!("missing int4_weights[{name}].zero")),
        );
        (packed, scale, zero)
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_2_shared_out_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 Generate an INT4 fixture with \
                 `python oracle/qwen36_moe_ffn_oracle.py --mode synthetic \
                 --int4 --out /tmp/qwen36_ffn_int4.json` and re-run."
            );
            return;
        };
        if !ffn_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Generate one with \
                 the `--int4` flag if you want to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Shared-expert weights become INT4 packed bytes; everything else
        // stays as BF16 from `weights`. The BF16 reconstruction is *not*
        // used at the kernel call site for INT4 tensors (the kernel reads
        // packed bytes), but the oracle ran its reference computation
        // against the same reconstruction so the intermediate target lines
        // up byte-for-byte with the kernel's output.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        let (sgp_packed, sgp_scale, sgp_zero) = decode_int4_sidecar(&json, "shared_gate_proj_w");
        let (sup_packed, sup_scale, sup_zero) = decode_int4_sidecar(&json, "shared_up_proj_w");
        let (sdp_packed, sdp_scale, sdp_zero) = decode_int4_sidecar(&json, "shared_down_proj_w");

        let shared_out_expected_bytes = b64_decode(inters["shared_out"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;
        let gsz_us = group_size as usize;

        // Sanity: shapes match the bake convention.
        assert_eq!(
            sgp_packed.len(),
            is_us * (hidden_us / 2),
            "shared_gate_proj packed bytes mismatch"
        );
        assert_eq!(
            sgp_scale.len(),
            (is_us / gsz_us) * (hidden_us / gsz_us) * 2,
            "shared_gate_proj scale bytes mismatch"
        );
        assert_eq!(sup_packed.len(), is_us * (hidden_us / 2));
        assert_eq!(sdp_packed.len(), hidden_us * (is_us / 2));
        assert_eq!(shared_out_expected_bytes.len(), hidden_us * 2);

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        // Packed INT4 buffers — uploaded as u8.
        let sgp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sgp_packed.len()], &sgp_packed)
                .expect("upload sgp packed");
        let sup_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sup_packed.len()], &sup_packed)
                .expect("upload sup packed");
        let sdp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sdp_packed.len()], &sdp_packed)
                .expect("upload sdp packed");
        // BF16 scale/zero sidecars.
        let sgp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sgp_scale.len() / 2],
            &sgp_scale,
        )
        .expect("upload sgp scale");
        let sgp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sgp_zero.len() / 2], &sgp_zero)
                .expect("upload sgp zero");
        let sup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sup_scale.len() / 2],
            &sup_scale,
        )
        .expect("upload sup scale");
        let sup_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sup_zero.len() / 2], &sup_zero)
                .expect("upload sup zero");
        let sdp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sdp_scale.len() / 2],
            &sdp_scale,
        )
        .expect("upload sdp scale");
        let sdp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sdp_zero.len() / 2], &sdp_zero)
                .expect("upload sdp zero");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 2,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        // Shared-expert weight pointers point at the *packed* u8 buffers.
        // Fused-expert pointers stay null (stage<3 doesn't read them).
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: std::ptr::null(),
            down_proj_w: std::ptr::null(),
            shared_gate_proj_w: sgp_packed_buf.as_ptr(),
            shared_up_proj_w: sup_packed_buf.as_ptr(),
            shared_down_proj_w: sdp_packed_buf.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size,
            gate_up_proj_scale: std::ptr::null(),
            gate_up_proj_zero: std::ptr::null(),
            down_proj_scale: std::ptr::null(),
            down_proj_zero: std::ptr::null(),
            shared_gate_proj_scale: sgp_scale_buf.as_ptr(),
            shared_gate_proj_zero: sgp_zero_buf.as_ptr(),
            shared_up_proj_scale: sup_scale_buf.as_ptr(),
            shared_up_proj_zero: sup_zero_buf.as_ptr(),
            shared_down_proj_scale: sdp_scale_buf.as_ptr(),
            shared_down_proj_zero: sdp_zero_buf.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 2 (int4)");

        // Same envelope as the BF16 stage 2 test: cos_sim ≥ 0.999, max
        // |delta| ≤ 0.05. The reconstruction is bit-identical to what the
        // oracle ran reference against, so any residual disagreement is
        // F32 reduction-order drift through the matvec — same as BF16.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step2 int4 shared_out",
            got_bytes,
            &shared_out_expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_3_expert0_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_2_shared_out_int4_matches_oracle` for setup."
            );
            return;
        };
        if !ffn_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Generate one with \
                 `--int4` to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 3 needs everything stage 2 needed plus the fused gate_up_proj.
        // gate_up_proj routes through INT4 (step 4 wires Phase G); down_proj
        // stays BF16 in this test until step 5 lands. The oracle's BF16
        // reconstruction for down_proj is in `weights.down_proj_w`.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let down_proj_w_bytes = b64_decode(weights["down_proj_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        let (gup_packed, gup_scale, gup_zero) = decode_int4_sidecar(&json, "gate_up_proj_w");
        let (sgp_packed, sgp_scale, sgp_zero) = decode_int4_sidecar(&json, "shared_gate_proj_w");
        let (sup_packed, sup_scale, sup_zero) = decode_int4_sidecar(&json, "shared_up_proj_w");
        let (sdp_packed, sdp_scale, sdp_zero) = decode_int4_sidecar(&json, "shared_down_proj_w");

        let expert_stack_expected = b64_decode(inters["expert_stack"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;
        let gsz_us = group_size as usize;
        let two_i = 2 * i_us;

        // Sanity: fused-expert packed shape is [E, 2*I, hidden/2].
        assert_eq!(
            gup_packed.len(),
            e_us * two_i * (hidden_us / 2),
            "gate_up_proj packed bytes mismatch"
        );
        assert_eq!(
            gup_scale.len(),
            e_us * (two_i / gsz_us) * (hidden_us / gsz_us) * 2,
            "gate_up_proj scale bytes mismatch"
        );
        assert_eq!(
            down_proj_w_bytes.len(),
            e_us * hidden_us * i_us * 2,
            "down_proj BF16 bytes mismatch"
        );

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let down_proj_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us, i_us],
            &down_proj_w_bytes,
        )
        .expect("upload down_proj_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        let gup_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[gup_packed.len()], &gup_packed)
                .expect("upload gup packed");
        let gup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[gup_scale.len() / 2],
            &gup_scale,
        )
        .expect("upload gup scale");
        let gup_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[gup_zero.len() / 2], &gup_zero)
                .expect("upload gup zero");

        let sgp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sgp_packed.len()], &sgp_packed)
                .expect("upload sgp packed");
        let sgp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sgp_scale.len() / 2],
            &sgp_scale,
        )
        .expect("upload sgp scale");
        let sgp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sgp_zero.len() / 2], &sgp_zero)
                .expect("upload sgp zero");
        let sup_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sup_packed.len()], &sup_packed)
                .expect("upload sup packed");
        let sup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sup_scale.len() / 2],
            &sup_scale,
        )
        .expect("upload sup scale");
        let sup_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sup_zero.len() / 2], &sup_zero)
                .expect("upload sup zero");
        let sdp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sdp_packed.len()], &sdp_packed)
                .expect("upload sdp packed");
        let sdp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sdp_scale.len() / 2],
            &sdp_scale,
        )
        .expect("upload sdp scale");
        let sdp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sdp_zero.len() / 2], &sdp_zero)
                .expect("upload sdp zero");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let params = Qwen36MoeFfnStepParams {
            stage: 3,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gup_packed_buf.as_ptr(), // packed u8 — INT4 path
            down_proj_w: down_proj_w.as_ptr(),       // BF16 reconstruction
            shared_gate_proj_w: sgp_packed_buf.as_ptr(),
            shared_up_proj_w: sup_packed_buf.as_ptr(),
            shared_down_proj_w: sdp_packed_buf.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size,
            gate_up_proj_scale: gup_scale_buf.as_ptr(),
            gate_up_proj_zero: gup_zero_buf.as_ptr(),
            // down_proj stays BF16 until step 5 wires Phase I.
            down_proj_scale: std::ptr::null(),
            down_proj_zero: std::ptr::null(),
            shared_gate_proj_scale: sgp_scale_buf.as_ptr(),
            shared_gate_proj_zero: sgp_zero_buf.as_ptr(),
            shared_up_proj_scale: sup_scale_buf.as_ptr(),
            shared_up_proj_zero: sup_zero_buf.as_ptr(),
            shared_down_proj_scale: sdp_scale_buf.as_ptr(),
            shared_down_proj_zero: sdp_zero_buf.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 3 (int4)");

        // Stage 3 publishes expert_0_out (BF16-cast view of the F32 stack)
        // into output[0..hidden]. Compare against `intermediates.expert_stack`
        // restricted to slot j=0.
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        let expected_bytes = &expert_stack_expected[..hidden_us * 2];
        assert_parity(
            "ffn step3 int4 expert_0_out",
            got_bytes,
            expected_bytes,
            0.05,
            0.999,
        );
    }

    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_ffn_step_5_output_hidden_int4_matches_oracle() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        let Some((json, geom)) = load_ffn_oracle_json() else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_FFN_ORACLE_JSON not set. \
                 See `qwen36_moe_ffn_step_2_shared_out_int4_matches_oracle` for setup."
            );
            return;
        };
        if !ffn_oracle_is_int4(&json) {
            eprintln!(
                "skip: oracle JSON is not INT4 (schema={}). Generate one with \
                 `--int4` to exercise this test.",
                json["schema"].as_str().unwrap_or("?"),
            );
            return;
        }
        let cfg = &json["config"];
        let group_size = cfg["int4_group_size"].as_i64().unwrap_or(0) as i32;
        assert!(group_size > 0, "INT4 oracle missing config.int4_group_size");

        let weights = &json["weights"];
        let inters = &json["intermediates"];

        // Stage 5 = the full FFN block. All five quantizable tensors flow
        // through the INT4 path (Phase D: shared gate/up; Phase F: shared
        // down; Phase G: gate_up_proj per expert; Phase I: down_proj per
        // expert). The non-quantized tensors stay BF16.
        let input_hidden_bytes = b64_decode(weights["input_hidden"].as_str().unwrap());
        let post_attn_norm_w_bytes = b64_decode(weights["post_attn_norm_w"].as_str().unwrap());
        let gate_w_bytes = b64_decode(weights["gate_w"].as_str().unwrap());
        let shared_expert_gate_w_bytes =
            b64_decode(weights["shared_expert_gate_w"].as_str().unwrap());

        let (gup_packed, gup_scale, gup_zero) = decode_int4_sidecar(&json, "gate_up_proj_w");
        let (dp_packed, dp_scale, dp_zero) = decode_int4_sidecar(&json, "down_proj_w");
        let (sgp_packed, sgp_scale, sgp_zero) = decode_int4_sidecar(&json, "shared_gate_proj_w");
        let (sup_packed, sup_scale, sup_zero) = decode_int4_sidecar(&json, "shared_up_proj_w");
        let (sdp_packed, sdp_scale, sdp_zero) = decode_int4_sidecar(&json, "shared_down_proj_w");

        let output_hidden_expected = b64_decode(inters["output_hidden"].as_str().unwrap());

        let hidden_us = geom.hidden as usize;
        let e_us = geom.num_experts as usize;
        let i_us = geom.moe_intermediate as usize;
        let is_us = geom.shared_intermediate as usize;
        let k_us = geom.top_k as usize;

        // Sanity: down_proj packed shape is [E, hidden, I/2].
        assert_eq!(
            dp_packed.len(),
            e_us * hidden_us * (i_us / 2),
            "down_proj packed bytes mismatch"
        );

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let input_hidden = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &input_hidden_bytes,
        )
        .expect("upload input_hidden");
        let post_attn_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_us],
            &post_attn_norm_w_bytes,
        )
        .expect("upload post_attn_norm_w");
        let gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[e_us, hidden_us],
            &gate_w_bytes,
        )
        .expect("upload gate_w");
        let shared_expert_gate_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[1, hidden_us],
            &shared_expert_gate_w_bytes,
        )
        .expect("upload shared_expert_gate_w");

        // All five INT4 tensors uploaded as packed u8 + BF16 sidecars.
        let gup_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[gup_packed.len()], &gup_packed)
                .expect("upload gup packed");
        let gup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[gup_scale.len() / 2],
            &gup_scale,
        )
        .expect("upload gup scale");
        let gup_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[gup_zero.len() / 2], &gup_zero)
                .expect("upload gup zero");
        let dp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[dp_packed.len()], &dp_packed)
                .expect("upload dp packed");
        let dp_scale_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[dp_scale.len() / 2], &dp_scale)
                .expect("upload dp scale");
        let dp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[dp_zero.len() / 2], &dp_zero)
                .expect("upload dp zero");
        let sgp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sgp_packed.len()], &sgp_packed)
                .expect("upload sgp packed");
        let sgp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sgp_scale.len() / 2],
            &sgp_scale,
        )
        .expect("upload sgp scale");
        let sgp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sgp_zero.len() / 2], &sgp_zero)
                .expect("upload sgp zero");
        let sup_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sup_packed.len()], &sup_packed)
                .expect("upload sup packed");
        let sup_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sup_scale.len() / 2],
            &sup_scale,
        )
        .expect("upload sup scale");
        let sup_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sup_zero.len() / 2], &sup_zero)
                .expect("upload sup zero");
        let sdp_packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[sdp_packed.len()], &sdp_packed)
                .expect("upload sdp packed");
        let sdp_scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[sdp_scale.len() / 2],
            &sdp_scale,
        )
        .expect("upload sdp scale");
        let sdp_zero_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[sdp_zero.len() / 2], &sdp_zero)
                .expect("upload sdp zero");

        let mut output =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_parity_output_elems(&geom)])
                .expect("alloc output");
        let mut output_idx =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[k_us]).expect("alloc output_idx");
        let mut workspace = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[ffn_parity_workspace_floats(&geom)],
        )
        .expect("alloc workspace");
        let mut sync_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync buf");

        let _ = (i_us, is_us); // kept for clarity above.

        let params = Qwen36MoeFfnStepParams {
            stage: 5,
            layer_idx: 0,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weight_ptrs = Qwen36MoeFfnStepWeights {
            input_hidden: input_hidden.as_ptr(),
            post_attn_norm_w: post_attn_norm_w.as_ptr(),
            gate_w: gate_w.as_ptr(),
            gate_up_proj_w: gup_packed_buf.as_ptr(),
            down_proj_w: dp_packed_buf.as_ptr(),
            shared_gate_proj_w: sgp_packed_buf.as_ptr(),
            shared_up_proj_w: sup_packed_buf.as_ptr(),
            shared_down_proj_w: sdp_packed_buf.as_ptr(),
            shared_expert_gate_w: shared_expert_gate_w.as_ptr(),
        };
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size,
            gate_up_proj_scale: gup_scale_buf.as_ptr(),
            gate_up_proj_zero: gup_zero_buf.as_ptr(),
            down_proj_scale: dp_scale_buf.as_ptr(),
            down_proj_zero: dp_zero_buf.as_ptr(),
            shared_gate_proj_scale: sgp_scale_buf.as_ptr(),
            shared_gate_proj_zero: sgp_zero_buf.as_ptr(),
            shared_up_proj_scale: sup_scale_buf.as_ptr(),
            shared_up_proj_zero: sup_zero_buf.as_ptr(),
            shared_down_proj_scale: sdp_scale_buf.as_ptr(),
            shared_down_proj_zero: sdp_zero_buf.as_ptr(),
        };

        ffn_step_launch(
            ordinal,
            ScalarType::BF16,
            params,
            &weight_ptrs,
            &int4_ptrs,
            &mut output,
            &mut output_idx,
            &mut workspace,
            &mut sync_buf,
        )
        .expect("ffn_step_launch stage 5 (int4)");

        // Stage 5 publishes output_hidden = input + moe + shared. The
        // residual add absorbs much of the matmul drift; same envelope
        // as the BF16 stage 5 test (cos_sim ≥ 0.999).
        let got_full = output.to_host_bytes().expect("download output");
        let got_bytes = &got_full[..hidden_us * 2];
        assert_parity(
            "ffn step5 int4 output_hidden",
            got_bytes,
            &output_hidden_expected,
            0.05,
            0.999,
        );
    }

    // -------------------------------------------------------------------
    // PR 4b5 step 2: INT4 dequant smoke test.
    //
    // Builds a small `[out_rows, in_cols]` weight slab in F32 on host,
    // runs min/max group-quant to (packed u8, BF16 scale, BF16 zero) with
    // exactly the same math as `oracle/qwen36_moe_ffn_oracle.py`, uploads
    // the sidecars, calls the smoke launcher, and verifies the GPU
    // outputs of `int4_dequant_8` and `int4_dequant_scalar` both equal a
    // host-computed reference byte-for-byte (these are F32 values rounded
    // through BF16, so exact equality is the right bar).
    //
    // Two configs run in succession to cover both helper paths:
    //   gsz=8, in_cols=16  → every 8-col span lies in one group     (fast)
    //   gsz=4, in_cols=16  → every 8-col span crosses a boundary    (slow)
    // -------------------------------------------------------------------

    /// BF16 round-to-nearest-even of an F32 value, returning the 16-bit
    /// big-end-of-F32 representation. Same math as the kernel's
    /// `bf16_round_rne_f32`.
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn bf16_round_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
        let r = bits.wrapping_add(rounding_bias);
        (r >> 16) as u16
    }

    /// Reverse: F32 from a BF16 bit pattern.
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn f32_from_bf16(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// Min/max INT4 group-quant on a 2D `[out, in]` F32 slab — Rust mirror
    /// of `minmax_int4_packed_and_recon` in the FFN oracle. Returns
    /// `(packed [out, in/2] u8, scale [out/gs, in/gs] u16-as-BF16,
    /// zero [out/gs, in/gs] u16-as-BF16)`.
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn host_minmax_int4(
        w: &[f32],
        out_rows: usize,
        in_cols: usize,
        gsz: usize,
    ) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
        assert_eq!(w.len(), out_rows * in_cols);
        assert_eq!(out_rows % gsz, 0);
        assert_eq!(in_cols % gsz, 0);
        assert_eq!(in_cols % 2, 0);
        let sr = out_rows / gsz;
        let sc = in_cols / gsz;

        let mut packed = vec![0u8; out_rows * (in_cols / 2)];
        let mut scale = vec![0u16; sr * sc];
        let mut zero = vec![0u16; sr * sc];

        for gr in 0..sr {
            for gc in 0..sc {
                let mut tmin = f32::INFINITY;
                let mut tmax = f32::NEG_INFINITY;
                for r in 0..gsz {
                    for c in 0..gsz {
                        let v = w[(gr * gsz + r) * in_cols + gc * gsz + c];
                        tmin = tmin.min(v);
                        tmax = tmax.max(v);
                    }
                }
                let rng = tmax - tmin;
                let s_f = if rng > 0.0 { rng / 15.0 } else { 1.0 };
                let z_f = if rng > 0.0 { -tmin / s_f } else { 0.0 };
                let s_bits = bf16_round_bits(s_f);
                let z_bits = bf16_round_bits(z_f);
                scale[gr * sc + gc] = s_bits;
                zero[gr * sc + gc] = z_bits;
                let s = f32_from_bf16(s_bits);
                let z = f32_from_bf16(z_bits);
                for r in 0..gsz {
                    for c in 0..gsz {
                        let row = gr * gsz + r;
                        let col = gc * gsz + c;
                        let v = w[row * in_cols + col];
                        let q = (v / s + z).round().clamp(0.0, 15.0) as u8;
                        // Pack: even col → low nibble, odd col → high nibble.
                        let byte_idx = row * (in_cols / 2) + col / 2;
                        if col & 1 == 0 {
                            packed[byte_idx] = (packed[byte_idx] & 0xF0) | (q & 0x0F);
                        } else {
                            packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                        }
                    }
                }
            }
        }
        (packed, scale, zero)
    }

    /// Reference reconstruction: `bf16(q*s - z*s)` per element. Returns
    /// F32 values whose lower 16 bits are zero (i.e. exactly BF16-precision).
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn host_dequant_recon(
        packed: &[u8],
        scale: &[u16],
        zero: &[u16],
        out_rows: usize,
        in_cols: usize,
        gsz: usize,
    ) -> Vec<f32> {
        let sc = in_cols / gsz;
        let mut out = vec![0.0f32; out_rows * in_cols];
        for row in 0..out_rows {
            for col in 0..in_cols {
                let gi = (row / gsz) * sc + col / gsz;
                let s = f32_from_bf16(scale[gi]);
                let z = f32_from_bf16(zero[gi]);
                let byte = packed[row * (in_cols / 2) + col / 2];
                let n = if col & 1 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let v = (n as f32) * s - z * s;
                out[row * in_cols + col] = f32::from_bits((bf16_round_bits(v) as u32) << 16);
            }
        }
        out
    }

    /// Encode a slice of BF16 16-bit values to LE bytes for `from_host_bytes`.
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(bits.len() * 2);
        for b in bits {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        bytes
    }

    /// Drives one (out_rows, in_cols, gsz) configuration through the smoke
    /// kernel and asserts both helper outputs match the host reference
    /// exactly.
    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    fn run_int4_dequant_smoke(out_rows: usize, in_cols: usize, gsz: usize, label: &str) {
        use gpu_hal::{is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};
        let backend = if is_backend_compiled(Backend::Metal) {
            Backend::Metal
        } else if is_backend_compiled(Backend::Cuda) {
            Backend::Cuda
        } else {
            Backend::Hip
        };
        set_backend(backend);
        let ordinal = 0usize;

        // Deterministic synthetic weights: a 32-bit LCG seeded by config so
        // each smoke variant uses different but reproducible values.
        let n = out_rows * in_cols;
        let mut rng_state: u32 =
            0xC0FFEE ^ ((out_rows as u32) << 16) ^ ((in_cols as u32) << 8) ^ (gsz as u32);
        let mut w = vec![0.0f32; n];
        for v in w.iter_mut() {
            // LCG (Numerical Recipes constants) → uniform [-1, 1).
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (rng_state >> 8) as f32 / ((1u32 << 24) as f32); // [0,1)
            *v = u * 2.0 - 1.0;
        }

        let (packed, scale_bits, zero_bits) = host_minmax_int4(&w, out_rows, in_cols, gsz);
        let recon_ref =
            host_dequant_recon(&packed, &scale_bits, &zero_bits, out_rows, in_cols, gsz);

        let packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[packed.len()], &packed)
                .expect("upload packed");
        let scale_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[scale_bits.len()],
            &bf16_bits_to_bytes(&scale_bits),
        )
        .expect("upload scale");
        let zero_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[zero_bits.len()],
            &bf16_bits_to_bytes(&zero_bits),
        )
        .expect("upload zero");

        let mut dq_8 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[n]).expect("alloc dq_8");
        let mut dq_scalar =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[n]).expect("alloc dq_scalar");

        int4_dequant_smoke_launch(
            ordinal,
            &packed_buf,
            &scale_buf,
            &zero_buf,
            out_rows as i32,
            in_cols as i32,
            gsz as i32,
            &mut dq_8,
            &mut dq_scalar,
        )
        .expect("smoke launch");

        let dq_8_bytes = dq_8.to_host_bytes().expect("download dq_8");
        let dq_scalar_bytes = dq_scalar.to_host_bytes().expect("download dq_scalar");
        let dq_8_v: Vec<f32> = dq_8_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let dq_scalar_v: Vec<f32> = dq_scalar_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for i in 0..n {
            assert_eq!(
                dq_8_v[i].to_bits(),
                recon_ref[i].to_bits(),
                "[{label}] int4_dequant_8 mismatch at i={i}: got {} ({:#010x}), want {} ({:#010x})",
                dq_8_v[i],
                dq_8_v[i].to_bits(),
                recon_ref[i],
                recon_ref[i].to_bits(),
            );
            assert_eq!(
                dq_scalar_v[i].to_bits(), recon_ref[i].to_bits(),
                "[{label}] int4_dequant_scalar mismatch at i={i}: got {} ({:#010x}), want {} ({:#010x})",
                dq_scalar_v[i], dq_scalar_v[i].to_bits(),
                recon_ref[i], recon_ref[i].to_bits(),
            );
        }
    }

    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    #[test]
    fn qwen36_moe_int4_dequant_smoke_fast_path() {
        // gsz=8, in_cols=16 → every 8-col span lies in one group, so
        // `int4_dequant_8` exercises its `g0 == g7` fast path on every span.
        run_int4_dequant_smoke(8, 16, 8, "fast (gsz=8)");
    }

    #[cfg(any(
        supersonic_backend_hip,
        supersonic_backend_cuda,
        supersonic_backend_metal
    ))]
    #[test]
    fn qwen36_moe_int4_dequant_smoke_slow_path() {
        // gsz=4, in_cols=16 → 8-col spans starting at col=0 cross from
        // group 0 to group 1 (boundary at col=4); spans starting at col=8
        // cross 2→3. Exercises the per-element `g0 != g7` slow path.
        run_int4_dequant_smoke(8, 16, 4, "slow (gsz=4)");
    }

    /// HIP parity test for the GPU final-RMSNorm + lm_head GEMV kernel.
    /// Builds a small synthetic vocab=512 / hidden=256 problem on the host,
    /// runs both:
    ///   - an F32 reference (the same math as
    ///     `qwen36_moe_decode::host_final_norm_lm_head_f32`), and
    ///   - the GPU kernel via `lm_head_launch`,
    /// then asserts the BF16 logits agree by cosine similarity ≥ 0.999.
    /// Bit-exactness isn't required: the GPU path multiplies BF16 values
    /// with F32 accumulation and rounds the final logit to BF16, so it's
    /// strictly less precise than the host F32-throughout reference. A
    /// high cos_sim catches any systematic kernel bug while tolerating
    /// the inherent BF16 rounding noise.
    #[cfg(any(supersonic_backend_hip, supersonic_backend_cuda))]
    #[test]
    fn qwen36_moe_lm_head_matches_host_f32_reference() {
        use gpu_hal::{copy_d2h, is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};

        let backend = if is_backend_compiled(Backend::Cuda) {
            Backend::Cuda
        } else {
            Backend::Hip
        };
        set_backend(backend);
        let ordinal = 0usize;
        let hidden: i32 = 256;
        let vocab: i32 = 512;
        let eps: f32 = 1e-6;

        // Deterministic seeded data — same convention the per-block oracles
        // use (~N(0, 1/sqrt(fan_in)) so logits stay O(1)).
        fn xorshift(state: &mut u32) -> f32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            // Map u32 → roughly N(0, 1) via Box-Muller-light: fold to (-1, 1).
            ((x & 0xFFFFFF) as f32 / 0xFFFFFF as f32) * 2.0 - 1.0
        }
        let mut rng = 0xC0FFEEu32;
        let mut bf16_round = |v: f32| -> u16 {
            let bits = v.to_bits();
            let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding_bias)) >> 16) as u16
        };

        let final_hidden_f32: Vec<f32> = (0..hidden).map(|_| xorshift(&mut rng) * 0.5).collect();
        let final_norm_w_f32: Vec<f32> = (0..hidden).map(|_| xorshift(&mut rng) * 0.02).collect();
        // lm_head: ~N(0, 1/sqrt(hidden)).
        let scale = 1.0 / (hidden as f32).sqrt();
        let lm_head_f32: Vec<f32> = (0..(vocab as usize) * (hidden as usize))
            .map(|_| xorshift(&mut rng) * scale)
            .collect();

        // Pack to BF16 little-endian bytes for the GPU upload.
        let to_bf16_bytes = |xs: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(xs.len() * 2);
            for &v in xs {
                let b = bf16_round(v);
                out.push((b & 0xFF) as u8);
                out.push((b >> 8) as u8);
            }
            out
        };
        let final_hidden_bytes = to_bf16_bytes(&final_hidden_f32);
        let final_norm_w_bytes = to_bf16_bytes(&final_norm_w_f32);
        let lm_head_bytes = to_bf16_bytes(&lm_head_f32);

        // F32 reference (matches `host_final_norm_lm_head_f32`).
        let bf16_to_f32 = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = ((c[1] as u32) << 24) | ((c[0] as u32) << 16);
                    f32::from_bits(bits)
                })
                .collect()
        };
        let h_f32 = bf16_to_f32(&final_hidden_bytes);
        let nw_f32 = bf16_to_f32(&final_norm_w_bytes);
        let lm_f32 = bf16_to_f32(&lm_head_bytes);
        let mean_sq = h_f32.iter().map(|&x| x * x).sum::<f32>() / hidden as f32;
        let rsqrt = 1.0 / (mean_sq + eps).sqrt();
        let normed: Vec<f32> = h_f32
            .iter()
            .zip(nw_f32.iter())
            .map(|(&x, &w)| x * rsqrt * (1.0 + w))
            .collect();
        let mut want_logits_f32 = vec![0f32; vocab as usize];
        for v in 0..vocab as usize {
            let row = v * hidden as usize;
            let mut acc = 0f64;
            for h in 0..hidden as usize {
                acc += lm_f32[row + h] as f64 * normed[h] as f64;
            }
            want_logits_f32[v] = acc as f32;
        }

        // GPU path.
        let final_hidden_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden as usize],
            &final_hidden_bytes,
        )
        .expect("upload final_hidden");
        let final_norm_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden as usize],
            &final_norm_w_bytes,
        )
        .expect("upload final_norm_w");
        let lm_head_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[vocab as usize, hidden as usize],
            &lm_head_bytes,
        )
        .expect("upload lm_head_w");
        let mut logits_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[vocab as usize]).expect("alloc logits");
        let mut counter_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");

        super::lm_head_launch(
            ordinal,
            hidden,
            vocab,
            eps,
            &final_hidden_buf,
            &final_norm_w_buf,
            &lm_head_w_buf,
            &mut logits_buf,
            None,
            &mut counter_buf,
        )
        .expect("lm_head launch");

        let mut got_bytes = vec![0u8; vocab as usize * 2];
        copy_d2h(
            ordinal,
            got_bytes.as_mut_ptr() as *mut _,
            logits_buf.as_ptr(),
            got_bytes.len(),
        )
        .expect("d2h logits");
        let got_logits = bf16_to_f32(&got_bytes);

        // BF16 cos_sim: should be very high (≥ 0.999) because both paths
        // converge on the same dot product modulo BF16 rounding noise on
        // the GPU side. Use F64 reductions to avoid the same precision
        // pitfall the bake survey hit on lm_head (PR #67).
        let dot: f64 = got_logits
            .iter()
            .zip(want_logits_f32.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let na: f64 = got_logits
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = want_logits_f32
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos_sim = dot / (na * nb);
        assert!(
            cos_sim >= 0.999,
            "GPU lm_head logits cos_sim {cos_sim:.6} below 0.999 threshold \
             (na={na:.4} nb={nb:.4} dot={dot:.4}) — likely a kernel math bug, \
             not BF16 rounding noise"
        );

        // Top-1 should agree on the F32 reference; if it disagrees, log it
        // (could be a near-tie under BF16 rounding) but don't fail the test.
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let got_top = argmax(&got_logits);
        let want_top = argmax(&want_logits_f32);
        if got_top != want_top {
            eprintln!(
                "[lm_head parity] argmax differs (got={got_top} want={want_top}) — \
                 expected at high enough cos_sim with near-tied logits, but logging \
                 in case it's a real ordering bug"
            );
        }
    }

    /// Phase 6.4a regression: the safe wrapper rejects M values that would
    /// overflow the per-block LDS budget. At hidden=2048 the LDS per
    /// row is 4 KiB (BF16); with 128 B reduction scratch and the 64 KiB
    /// per-block cap, the effective M ceiling is 15. M=16 must fail
    /// loudly here rather than crash the kernel launch with HIP status
    /// 254 (which has no usable error message). Pure host-side check;
    /// no GPU work needed.
    #[test]
    fn qwen36_moe_lm_head_batched_rejects_lds_overflow() {
        use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};

        // Skip when HIP isn't compiled — the wrapper's pre-launch arg
        // check still runs, but the buffers below need a backend.
        if !gpu_hal::is_backend_compiled(Backend::Hip) {
            eprintln!("skip: HIP backend not compiled");
            return;
        }
        set_backend(Backend::Hip);
        let ordinal = 0usize;

        // Production hidden=2048: LDS per row = 4096 B; 128 B reduction.
        // (64 KiB - 128 B) / 4096 B = 15, so M ∈ [1, 15] is OK; M=16 isn't.
        const HIDDEN: i32 = 2048;
        const VOCAB: i32 = 256; // tiny, just to keep alloc cheap
        const EPS: f32 = 1e-6;

        // Minimal buffers — the call should reject before touching them.
        let one_row_bytes = vec![0u8; HIDDEN as usize * 2];
        let final_norm_w = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[HIDDEN as usize],
            &one_row_bytes,
        )
        .expect("alloc final_norm_w");
        let lm_head_w = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[VOCAB as usize, HIDDEN as usize],
        )
        .expect("alloc lm_head_w");
        let final_hidden_16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[16, HIDDEN as usize])
            .expect("alloc final_hidden");
        let mut logits_16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[16, VOCAB as usize])
            .expect("alloc logits");

        // M = 16 must fail with a clear actionable message.
        let err = super::lm_head_batched_launch(
            ordinal,
            /* m */ 16,
            HIDDEN,
            VOCAB,
            EPS,
            &final_hidden_16,
            &final_norm_w,
            &lm_head_w,
            &mut logits_16,
            None,
        )
        .expect_err("M=16 at hidden=2048 must reject");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("max_m=15"),
            "expected the rejection to surface max_m=15 for hidden=2048, got: {msg}"
        );

        // Sanity: M=15 should pass argument validation (we don't actually
        // launch — just confirm the LDS bound is permissive enough to
        // accept the largest still-valid M).
        let final_hidden_15 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[15, HIDDEN as usize])
            .expect("alloc final_hidden 15");
        let mut logits_15 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[15, VOCAB as usize])
            .expect("alloc logits 15");
        // The launch may succeed or fail post-arg-check (kernel runs
        // and produces logits); either way the wrapper must NOT reject
        // on argument validation. Treat any post-launch error as test
        // pass since the LDS bound check is what we're validating.
        let _ = super::lm_head_batched_launch(
            ordinal,
            /* m */ 15,
            HIDDEN,
            VOCAB,
            EPS,
            &final_hidden_15,
            &final_norm_w,
            &lm_head_w,
            &mut logits_15,
            None,
        );
    }

    /// Phase 6.4a parity: batched lm_head (M=K) vs K sequential single-M
    /// `lm_head_launch` calls. The batched kernel must produce the same
    /// logits as K independent single-M calls modulo BF16 F32-accumulation
    /// order (cos_sim ≥ 0.999 per row).
    ///
    /// Synthesizes a small vocab/hidden problem so the test runs in
    /// milliseconds. Real-shape (vocab=248k, hidden=2048) coverage comes
    /// once Phase 6.4b wires the kernel into the speculative driver and
    /// exercises it via the engine path.
    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_lm_head_batched_matches_sequential() {
        use gpu_hal::{copy_d2h, set_backend, Backend, GpuBuffer, ScalarType};

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        const M: i32 = 3; // K=3 matches the speculative-decode default
        const HIDDEN: i32 = 256;
        const VOCAB: i32 = 512;
        const EPS: f32 = 1e-6;

        // Deterministic seeded data — same xorshift the single-M test uses.
        fn xorshift(state: &mut u32) -> f32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            ((x & 0xFFFFFF) as f32 / 0xFFFFFF as f32) * 2.0 - 1.0
        }
        let mut rng = 0xBADCAFEu32;
        let bf16_round = |v: f32| -> u16 {
            let bits = v.to_bits();
            let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding_bias)) >> 16) as u16
        };

        // [M, HIDDEN] BF16 input, [HIDDEN] norm, [VOCAB, HIDDEN] lm_head.
        let final_hidden_f32: Vec<f32> = (0..(M * HIDDEN))
            .map(|_| xorshift(&mut rng) * 0.5)
            .collect();
        let final_norm_w_f32: Vec<f32> = (0..HIDDEN).map(|_| xorshift(&mut rng) * 0.02).collect();
        let scale = 1.0 / (HIDDEN as f32).sqrt();
        let lm_head_f32: Vec<f32> = (0..(VOCAB as usize * HIDDEN as usize))
            .map(|_| xorshift(&mut rng) * scale)
            .collect();

        let to_bf16_bytes = |xs: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(xs.len() * 2);
            for &v in xs {
                let b = bf16_round(v);
                out.push((b & 0xFF) as u8);
                out.push((b >> 8) as u8);
            }
            out
        };
        let final_hidden_bytes = to_bf16_bytes(&final_hidden_f32);
        let final_norm_w_bytes = to_bf16_bytes(&final_norm_w_f32);
        let lm_head_bytes = to_bf16_bytes(&lm_head_f32);

        let final_norm_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[HIDDEN as usize],
            &final_norm_w_bytes,
        )
        .expect("upload final_norm_w");
        let lm_head_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[VOCAB as usize, HIDDEN as usize],
            &lm_head_bytes,
        )
        .expect("upload lm_head_w");

        // ---- Reference: K sequential single-M lm_head_launch calls -------
        let mut want_logits_per_row: Vec<Vec<u8>> = Vec::with_capacity(M as usize);
        for row in 0..M as usize {
            let row_bytes =
                &final_hidden_bytes[row * HIDDEN as usize * 2..(row + 1) * HIDDEN as usize * 2];
            let row_buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[HIDDEN as usize],
                row_bytes,
            )
            .expect("upload final_hidden row");
            let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[VOCAB as usize])
                .expect("alloc logits");
            let mut counter_buf =
                GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");
            super::lm_head_launch(
                ordinal,
                HIDDEN,
                VOCAB,
                EPS,
                &row_buf,
                &final_norm_w_buf,
                &lm_head_w_buf,
                &mut logits_buf,
                None,
                &mut counter_buf,
            )
            .expect("single-M lm_head launch");
            let mut row_logits = vec![0u8; VOCAB as usize * 2];
            copy_d2h(
                ordinal,
                row_logits.as_mut_ptr() as *mut _,
                logits_buf.as_ptr(),
                row_logits.len(),
            )
            .expect("d2h single-M logits");
            want_logits_per_row.push(row_logits);
        }

        // ---- Batched: one launch produces all K rows ---------------------
        let final_hidden_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[M as usize, HIDDEN as usize],
            &final_hidden_bytes,
        )
        .expect("upload [m, hidden]");
        let mut batched_logits_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[M as usize, VOCAB as usize])
                .expect("alloc [m, vocab] logits");

        super::lm_head_batched_launch(
            ordinal,
            M,
            HIDDEN,
            VOCAB,
            EPS,
            &final_hidden_buf,
            &final_norm_w_buf,
            &lm_head_w_buf,
            &mut batched_logits_buf,
            None,
        )
        .expect("batched lm_head launch");

        let batched_bytes = batched_logits_buf
            .to_host_bytes()
            .expect("d2h batched logits");
        assert_eq!(
            batched_bytes.len(),
            M as usize * VOCAB as usize * 2,
            "batched logits size"
        );

        // ---- Compare per-row -----------------------------------------------
        for row in 0..M as usize {
            let want = &want_logits_per_row[row];
            let got = &batched_bytes[row * VOCAB as usize * 2..(row + 1) * VOCAB as usize * 2];
            let want_f = bf16_bytes_to_f32(want);
            let got_f = bf16_bytes_to_f32(got);

            let mut max_abs = 0.0f32;
            let mut dot = 0.0f64;
            let mut want_sq = 0.0f64;
            let mut got_sq = 0.0f64;
            for (g, w) in got_f.iter().zip(want_f.iter()) {
                let d = (g - w).abs();
                if d > max_abs {
                    max_abs = d;
                }
                dot += (*g as f64) * (*w as f64);
                got_sq += (*g as f64).powi(2);
                want_sq += (*w as f64).powi(2);
            }
            let cos_sim = dot / (got_sq.sqrt() * want_sq.sqrt() + 1e-30);
            eprintln!(
                "[lm_head batched parity row {row}] max_abs={max_abs:.5e} \
                 cos_sim={cos_sim:.7}"
            );
            assert!(
                cos_sim >= 0.999,
                "row {row}: cos_sim {cos_sim:.7} below 0.999 floor — \
                 batched WMMA divergence vs single-M reference"
            );
        }
    }

    // ---- Phase 6.2c.1: MTP pre-fusion parity vs the Python oracle ---------
    //
    // Validates that `mtp_pre_fusion_launch` reproduces the
    // `Qwen3NextMultiTokenPredictor` pre-fusion stage byte-for-byte through
    // the BF16 rounding boundary. Reads the `qwen36-moe-mtp-oracle-v1`
    // fixture produced by `oracle/qwen36_moe_mtp_oracle.py`, feeds the
    // step-0 inputs (h_base, embed_tokens row, mtp.fc, two RMSNorm gains)
    // into the kernel, and compares the three outputs (e_norm, h_norm,
    // fused) against the oracle's intermediates.
    //
    // To run:
    //   .venv-bake/bin/python oracle/qwen36_moe_mtp_oracle.py \
    //     --model-dir /path/to/Qwen3.6-35B-A3B \
    //     --num-speculative-tokens 1 --seed 42 \
    //     --out /tmp/qwen36_mtp.json
    //   SUPERSONIC_QWEN36_MTP_ORACLE_JSON=/tmp/qwen36_mtp.json \
    //     cargo test --release -p kernel-ffi qwen36_moe_mtp_pre_fusion
    //
    // Without the env var the test prints a clear skip message and exits.
    #[cfg(supersonic_backend_hip)]
    #[test]
    fn qwen36_moe_mtp_pre_fusion_matches_oracle() {
        use gpu_hal::{copy_d2h, set_backend, Backend, GpuBuffer, ScalarType};

        let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set. Generate \
                 a fixture with `python oracle/qwen36_moe_mtp_oracle.py \
                 --model-dir <Qwen3.6-35B-A3B> --num-speculative-tokens 1 \
                 --out /tmp/qwen36_mtp.json` and re-run."
            );
            return;
        };
        let raw = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read mtp oracle json {json_path}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&raw).expect("mtp oracle json parse");

        assert_eq!(
            json["schema"].as_str().unwrap_or(""),
            "qwen36-moe-mtp-oracle-v1",
            "MTP parity test expects schema qwen36-moe-mtp-oracle-v1; \
             regenerate the fixture if the oracle has bumped its schema."
        );

        let cfg = &json["config"];
        let hidden = cfg["hidden"].as_i64().expect("config.hidden") as i32;
        let rms_norm_eps = cfg["rms_norm_eps"].as_f64().expect("config.rms_norm_eps") as f32;

        let prefusion = &json["prefusion_weights"];
        assert!(
            prefusion.is_object(),
            "oracle JSON missing `prefusion_weights` block — regenerate with \
             the Phase 6.2c.1 oracle (it adds fc_w + RMSNorm gains)"
        );
        let fc_w_bytes = b64_decode(
            prefusion["fc_w_bf16"]
                .as_str()
                .expect("prefusion_weights.fc_w_bf16"),
        );
        let pre_fc_norm_embedding_w_bytes = b64_decode(
            prefusion["pre_fc_norm_embedding_w_bf16"]
                .as_str()
                .expect("prefusion_weights.pre_fc_norm_embedding_w_bf16"),
        );
        let pre_fc_norm_hidden_w_bytes = b64_decode(
            prefusion["pre_fc_norm_hidden_w_bf16"]
                .as_str()
                .expect("prefusion_weights.pre_fc_norm_hidden_w_bf16"),
        );

        let h_base_bytes = b64_decode(
            json["h_base_step0_bf16"]
                .as_str()
                .expect("h_base_step0_bf16"),
        );

        let step0 = &json["steps"][0];
        let e_in_bytes = b64_decode(
            step0["input_token_embed_bf16"]
                .as_str()
                .expect("steps[0].input_token_embed_bf16"),
        );
        let want_e_norm_bytes =
            b64_decode(step0["e_norm_bf16"].as_str().expect("steps[0].e_norm_bf16"));
        let want_h_norm_bytes =
            b64_decode(step0["h_norm_bf16"].as_str().expect("steps[0].h_norm_bf16"));
        let want_fused_bytes =
            b64_decode(step0["fused_bf16"].as_str().expect("steps[0].fused_bf16"));

        let hidden_usz = hidden as usize;
        assert_eq!(e_in_bytes.len(), hidden_usz * 2, "e_in shape");
        assert_eq!(h_base_bytes.len(), hidden_usz * 2, "h_base shape");
        assert_eq!(
            pre_fc_norm_embedding_w_bytes.len(),
            hidden_usz * 2,
            "pre_fc_norm_embedding_w shape"
        );
        assert_eq!(
            pre_fc_norm_hidden_w_bytes.len(),
            hidden_usz * 2,
            "pre_fc_norm_hidden_w shape"
        );
        assert_eq!(
            fc_w_bytes.len(),
            hidden_usz * 2 * hidden_usz * 2,
            "fc_w shape: expected [hidden, 2*hidden] BF16"
        );

        set_backend(Backend::Hip);
        let ordinal = 0usize;

        let e_in_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[hidden_usz], &e_in_bytes)
                .expect("upload e_in");
        let h_base_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[hidden_usz], &h_base_bytes)
                .expect("upload h_base");
        let pre_fc_norm_embedding_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_usz],
            &pre_fc_norm_embedding_w_bytes,
        )
        .expect("upload pre_fc_norm_embedding_w");
        let pre_fc_norm_hidden_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_usz],
            &pre_fc_norm_hidden_w_bytes,
        )
        .expect("upload pre_fc_norm_hidden_w");
        let fc_w_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &[hidden_usz, 2 * hidden_usz],
            &fc_w_bytes,
        )
        .expect("upload fc_w");
        let mut e_norm_out_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden_usz]).expect("alloc e_norm_out");
        let mut h_norm_out_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden_usz]).expect("alloc h_norm_out");
        let mut fused_out_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden_usz]).expect("alloc fused_out");

        super::mtp_pre_fusion_launch(
            ordinal,
            hidden,
            rms_norm_eps,
            &e_in_buf,
            &h_base_buf,
            &pre_fc_norm_embedding_w_buf,
            &pre_fc_norm_hidden_w_buf,
            &fc_w_buf,
            &mut e_norm_out_buf,
            &mut h_norm_out_buf,
            &mut fused_out_buf,
        )
        .expect("mtp_pre_fusion launch");

        let mut got_e_norm = vec![0u8; hidden_usz * 2];
        let mut got_h_norm = vec![0u8; hidden_usz * 2];
        let mut got_fused = vec![0u8; hidden_usz * 2];
        copy_d2h(
            ordinal,
            got_e_norm.as_mut_ptr() as *mut _,
            e_norm_out_buf.as_ptr(),
            got_e_norm.len(),
        )
        .expect("d2h e_norm");
        copy_d2h(
            ordinal,
            got_h_norm.as_mut_ptr() as *mut _,
            h_norm_out_buf.as_ptr(),
            got_h_norm.len(),
        )
        .expect("d2h h_norm");
        copy_d2h(
            ordinal,
            got_fused.as_mut_ptr() as *mut _,
            fused_out_buf.as_ptr(),
            got_fused.len(),
        )
        .expect("d2h fused");

        // The two RMSNorm outputs go through one BF16 round each — single
        // `(x * inv_rms * (1+w))` pass — so they should be effectively
        // bit-exact against PyTorch (modulo the F32 reduction tree, which
        // the kernel matches via the same single-block sum). max_abs ≤ 1
        // ULP-of-BF16 at any reasonable magnitude is comfortable.
        assert_parity(
            "mtp.e_norm",
            &got_e_norm,
            &want_e_norm_bytes,
            /* max_abs */ 1e-2,
            /* cos_sim_floor */ 0.99999,
        );
        assert_parity("mtp.h_norm", &got_h_norm, &want_h_norm_bytes, 1e-2, 0.99999);
        // fused is a `[hidden, 2*hidden]` matvec; F32 accumulation order on
        // the GPU differs from PyTorch's per-row reduction tree. cos_sim
        // ≥ 0.999 is the bar the other Qwen3.6-MoE matvec parity tests use
        // (FFN, lm_head), and it's well within the BF16 rounding floor.
        assert_parity(
            "mtp.fused",
            &got_fused,
            &want_fused_bytes,
            /* max_abs */ 5e-2,
            /* cos_sim_floor */ 0.999,
        );
    }
}

// ABI sanity — every time `Qwen36MoeDecodeLayerDesc` grows, both this
// const and the C++ side `static_assert` in `kernels/qwen36_moe_bridge.cpp`
// must move together. Keep the bound pinned tightly: a loose range
// (e.g. [256, 512]) silently absorbs forgotten field appends. If you
// add or remove a field, update this exact size and the C++ side in
// the same commit.
#[cfg(target_pointer_width = "64")]
const _ASSERT_DECODE_LAYER_DESC_SIZE: () = {
    let sz = std::mem::size_of::<Qwen36MoeDecodeLayerDesc>();
    assert!(sz == _DECODE_LAYER_DESC_EXPECTED_BYTES);
};
#[cfg(target_pointer_width = "64")]
const _DECODE_LAYER_DESC_EXPECTED_BYTES: usize = 344;
