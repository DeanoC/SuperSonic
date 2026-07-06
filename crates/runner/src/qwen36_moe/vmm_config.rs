use std::ops::{Deref, DerefMut};

use anyhow::Result;
use gpu_hal::Backend;
use model_store::VirtualArenaTransferBackend;
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
    pub(crate) virtual_transfer_backend: VirtualArenaTransferBackend,
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
    flm_virtual_transfer_backend_cli: Option<&str>,
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
    let virtual_transfer_backend_env =
        std::env::var("SUPERSONIC_FLM_VIRTUAL_TRANSFER_BACKEND").ok();

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

    let virtual_transfer_backend = virtual_arena_transfer_backend_from_cli_or_env_value(
        flm_virtual_transfer_backend_cli,
        virtual_transfer_backend_env.as_deref(),
    )?;
    validate_virtual_transfer_backend_supported(backend, virtual_transfer_backend)?;

    Ok(MoeRuntimeConfig {
        policy,
        sparse_telemetry,
        virtual_transfer_backend,
    })
}

pub(crate) fn should_use_qwen36_kv_vmm(backend: Backend, ordinal: usize) -> Result<bool> {
    let raw = std::env::var("SUPERSONIC_VMM_KV").ok();
    let mode = qwen36_kv_vmm_mode_from_env_value(raw.as_deref(), backend)?;
    should_use_qwen36_kv_vmm_mode(mode, backend, ordinal)
}

pub(crate) fn effective_moe_expert_vmm_mode_for_transfer_backend(
    mode: MoeExpertVmmMode,
    island_cap_experts: Option<usize>,
    backend: VirtualArenaTransferBackend,
) -> Result<MoeExpertVmmMode> {
    if backend == VirtualArenaTransferBackend::PageableH2d || island_cap_experts.is_some() {
        return Ok(mode);
    }
    match mode {
        MoeExpertVmmMode::Auto => Ok(MoeExpertVmmMode::Force),
        MoeExpertVmmMode::Force => Ok(MoeExpertVmmMode::Force),
        MoeExpertVmmMode::Disabled => anyhow::bail!(
            "FLM virtual transfer backend gpu-direct-storage requires MoE expert VMM; \
             unset SUPERSONIC_VMM_MOE_ISLANDS=0"
        ),
    }
}

pub(crate) fn validate_virtual_transfer_backend_supported(
    backend: Backend,
    transfer_backend: VirtualArenaTransferBackend,
) -> Result<()> {
    if transfer_backend == VirtualArenaTransferBackend::PageableH2d
        || gpu_hal::storage_to_device_is_supported(backend)
    {
        return Ok(());
    }
    let reason = match backend {
        Backend::Hip => {
            "hipFile support is not compiled; ROCm >= 7.2 with hipfile.h and libhipfile is required"
        }
        Backend::Cuda => "CUDA storage-to-device transfer is not implemented yet",
        Backend::Metal => "Metal storage-to-device transfer is not implemented",
    };
    anyhow::bail!(
        "FLM virtual transfer backend gpu-direct-storage is not available for {backend}: {reason}"
    )
}

pub(crate) fn virtual_arena_transfer_backend_from_env_value(
    value: Option<&str>,
) -> Result<VirtualArenaTransferBackend> {
    virtual_arena_transfer_backend_from_value(value, "SUPERSONIC_FLM_VIRTUAL_TRANSFER_BACKEND")
}

pub(crate) fn virtual_arena_transfer_backend_from_cli_or_env_value(
    cli_value: Option<&str>,
    env_value: Option<&str>,
) -> Result<VirtualArenaTransferBackend> {
    match cli_value {
        Some(value) => {
            virtual_arena_transfer_backend_from_value(Some(value), "--flm-virtual-transfer-backend")
        }
        None => virtual_arena_transfer_backend_from_env_value(env_value),
    }
}

fn virtual_arena_transfer_backend_from_value(
    value: Option<&str>,
    source_label: &str,
) -> Result<VirtualArenaTransferBackend> {
    let Some(value) = value else {
        return Ok(VirtualArenaTransferBackend::PageableH2d);
    };
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Ok(VirtualArenaTransferBackend::PageableH2d);
    }
    match value.as_str() {
        "pageable" | "pageable-h2d" | "h2d" => Ok(VirtualArenaTransferBackend::PageableH2d),
        "gpu-direct-storage" | "gpu_direct_storage" | "gds" | "hipfile" => {
            Ok(VirtualArenaTransferBackend::GpuDirectStorage)
        }
        _ => anyhow::bail!(
            "{source_label} must be one of pageable-h2d, gpu-direct-storage, gds, or hipfile \
             (got {value:?})"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_store::VirtualArenaTransferBackend;

    #[test]
    fn flm_virtual_transfer_backend_env_defaults_to_pageable_h2d() {
        assert_eq!(
            virtual_arena_transfer_backend_from_env_value(None).unwrap(),
            VirtualArenaTransferBackend::PageableH2d
        );
        assert_eq!(
            virtual_arena_transfer_backend_from_env_value(Some("")).unwrap(),
            VirtualArenaTransferBackend::PageableH2d
        );
        assert_eq!(
            virtual_arena_transfer_backend_from_env_value(Some("pageable-h2d")).unwrap(),
            VirtualArenaTransferBackend::PageableH2d
        );
    }

    #[test]
    fn flm_virtual_transfer_backend_env_accepts_direct_storage_aliases() {
        for value in ["gpu-direct-storage", "gds", "hipfile"] {
            assert_eq!(
                virtual_arena_transfer_backend_from_env_value(Some(value)).unwrap(),
                VirtualArenaTransferBackend::GpuDirectStorage
            );
        }
    }

    #[test]
    fn flm_virtual_transfer_backend_env_rejects_unknown_values() {
        let err = virtual_arena_transfer_backend_from_env_value(Some("maybe")).unwrap_err();
        assert!(err
            .to_string()
            .contains("SUPERSONIC_FLM_VIRTUAL_TRANSFER_BACKEND"));
    }

    #[test]
    fn flm_virtual_transfer_backend_cli_value_overrides_env_value() {
        assert_eq!(
            virtual_arena_transfer_backend_from_cli_or_env_value(
                Some("pageable-h2d"),
                Some("hipfile"),
            )
            .unwrap(),
            VirtualArenaTransferBackend::PageableH2d
        );
        assert_eq!(
            virtual_arena_transfer_backend_from_cli_or_env_value(Some("hipfile"), Some("h2d"))
                .unwrap(),
            VirtualArenaTransferBackend::GpuDirectStorage
        );
    }

    #[test]
    fn flm_virtual_transfer_backend_env_remains_fallback_when_cli_omitted() {
        assert_eq!(
            virtual_arena_transfer_backend_from_cli_or_env_value(None, Some("hipfile")).unwrap(),
            VirtualArenaTransferBackend::GpuDirectStorage
        );
    }

    #[test]
    fn flm_virtual_transfer_backend_reports_cli_flag_for_bad_cli_value() {
        let err =
            virtual_arena_transfer_backend_from_cli_or_env_value(Some("maybe"), Some("hipfile"))
                .unwrap_err();

        assert!(err.to_string().contains("--flm-virtual-transfer-backend"));
    }

    #[test]
    fn flm_direct_transfer_backend_requires_storage_to_device_support() {
        assert!(validate_virtual_transfer_backend_supported(
            Backend::Hip,
            VirtualArenaTransferBackend::PageableH2d,
        )
        .is_ok());

        if !gpu_hal::storage_to_device_is_supported(Backend::Hip) {
            let err = validate_virtual_transfer_backend_supported(
                Backend::Hip,
                VirtualArenaTransferBackend::GpuDirectStorage,
            )
            .expect_err("HIP direct transfer should be rejected when hipFile is unavailable");

            assert!(
                err.to_string().contains("hipFile support is not compiled"),
                "{err}"
            );
        }
    }

    #[test]
    fn flm_direct_transfer_backend_forces_eager_virtual_expert_route() {
        assert_eq!(
            effective_moe_expert_vmm_mode_for_transfer_backend(
                MoeExpertVmmMode::Auto,
                None,
                VirtualArenaTransferBackend::GpuDirectStorage,
            )
            .unwrap(),
            MoeExpertVmmMode::Force
        );
        assert_eq!(
            effective_moe_expert_vmm_mode_for_transfer_backend(
                MoeExpertVmmMode::Auto,
                None,
                VirtualArenaTransferBackend::PageableH2d,
            )
            .unwrap(),
            MoeExpertVmmMode::Auto
        );
        assert!(effective_moe_expert_vmm_mode_for_transfer_backend(
            MoeExpertVmmMode::Disabled,
            None,
            VirtualArenaTransferBackend::GpuDirectStorage,
        )
        .is_err());
        assert_eq!(
            effective_moe_expert_vmm_mode_for_transfer_backend(
                MoeExpertVmmMode::Auto,
                Some(8),
                VirtualArenaTransferBackend::GpuDirectStorage,
            )
            .unwrap(),
            MoeExpertVmmMode::Auto
        );
    }
}
