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
static QWEN36_FFN_ROUTER_PARITY_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static QWEN36_FFN_SHARED_PARITY_TAP_CALLS: AtomicUsize = AtomicUsize::new(0);

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
