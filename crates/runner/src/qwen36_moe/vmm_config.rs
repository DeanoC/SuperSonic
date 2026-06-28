use std::ops::{Deref, DerefMut};

use anyhow::Result;
use gpu_hal::Backend;
use supersonic_runtime::qwen36_moe_config::{
    qwen36_kv_vmm_mode_from_env_value, should_use_qwen36_kv_vmm as should_use_qwen36_kv_vmm_mode,
    Qwen36MoeRuntimeConfig, Qwen36MoeRuntimeConfigInputs,
};

use crate::qwen36_moe_telemetry::MoeSparseTelemetry;

pub(crate) use supersonic_runtime::qwen36_moe_config::{
    should_try_moe_expert_vmm, MoeExpertVmmMode,
};

pub(crate) struct MoeRuntimeConfig {
    policy: Qwen36MoeRuntimeConfig,
    pub(crate) sparse_telemetry: Option<MoeSparseTelemetry>,
}

impl Deref for MoeRuntimeConfig {
    type Target = Qwen36MoeRuntimeConfig;

    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}

impl DerefMut for MoeRuntimeConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.policy
    }
}

pub(crate) fn prepare_moe_runtime_config(
    speculative_decode: bool,
    persistent_decode: bool,
    backend: Backend,
    top_k: usize,
) -> Result<MoeRuntimeConfig> {
    let vmm_mode = std::env::var("SUPERSONIC_VMM_MOE_ISLANDS").ok();
    let island_cap_experts = std::env::var("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS").ok();
    let protected_experts = std::env::var("SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS").ok();
    let fixed_hot_experts = std::env::var("SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS").ok();
    let prefetch_mode = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH").ok();
    let prefetch_ranks = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS").ok();
    let transition_min_observations =
        std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS").ok();
    let async_prefetch = std::env::var("SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH").ok();
    let async_staging_pages = std::env::var("SUPERSONIC_MOE_ISLAND_ASYNC_STAGING_PAGES").ok();
    let prefetch_evict = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT").ok();
    let prefetch_evict_min_probability =
        std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB").ok();
    let protect_demand_routes = std::env::var("SUPERSONIC_MOE_ISLAND_PROTECT_DEMAND").ok();
    let hot_protect_min_hits = std::env::var("SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS").ok();
    let fixed_hot_min_hits = std::env::var("SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS").ok();

    let inputs = Qwen36MoeRuntimeConfigInputs {
        vmm_mode: vmm_mode.as_deref(),
        island_cap_experts: island_cap_experts.as_deref(),
        protected_experts: protected_experts.as_deref(),
        fixed_hot_experts: fixed_hot_experts.as_deref(),
        prefetch_mode: prefetch_mode.as_deref(),
        prefetch_ranks: prefetch_ranks.as_deref(),
        transition_min_observations: transition_min_observations.as_deref(),
        async_prefetch: async_prefetch.as_deref(),
        async_staging_pages: async_staging_pages.as_deref(),
        prefetch_evict: prefetch_evict.as_deref(),
        prefetch_evict_min_probability: prefetch_evict_min_probability.as_deref(),
        protect_demand_routes: protect_demand_routes.as_deref(),
        hot_protect_min_hits: hot_protect_min_hits.as_deref(),
        fixed_hot_min_hits: fixed_hot_min_hits.as_deref(),
    };
    let policy = Qwen36MoeRuntimeConfig::from_inputs(&inputs, speculative_decode, backend, top_k)?;

    let sparse_telemetry = MoeSparseTelemetry::from_env(
        policy.sparse_requested,
        persistent_decode,
        policy.prefetch_mode,
        policy.prefetch_ranks,
    )?;
    if let Some(path) = sparse_telemetry
        .as_ref()
        .and_then(|telemetry| telemetry.dump_path.as_ref())
    {
        println!(
            "  [vmm] sparse MoE residency telemetry will be written to {}",
            path.display()
        );
    }

    Ok(MoeRuntimeConfig {
        policy,
        sparse_telemetry,
    })
}

pub(crate) fn should_use_qwen36_kv_vmm(backend: Backend, ordinal: usize) -> Result<bool> {
    let raw = std::env::var("SUPERSONIC_VMM_KV").ok();
    let mode = qwen36_kv_vmm_mode_from_env_value(raw.as_deref(), backend)?;
    should_use_qwen36_kv_vmm_mode(mode, backend, ordinal)
}
