use anyhow::{anyhow, Context, Result};
use gpu_hal::Backend;

use crate::qwen36_moe_telemetry::{MoeIslandPrefetchMode, MoeSparseTelemetry};

pub(crate) const DEFAULT_SPARSE_MOE_PREFETCH_RANKS: usize = 4;
pub(crate) const DEFAULT_SPARSE_MOE_TRANSITION_MIN_OBSERVATIONS: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoeExpertVmmMode {
    Auto,
    Disabled,
    Force,
}

impl MoeExpertVmmMode {
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("SUPERSONIC_VMM_MOE_ISLANDS").ok().as_deref() {
            None => Ok(Self::Auto),
            Some("0") => Ok(Self::Disabled),
            Some("1") => Ok(Self::Force),
            Some(other) => Err(anyhow!(
                "SUPERSONIC_VMM_MOE_ISLANDS must be unset, 0, or 1; got {other:?}"
            )),
        }
    }
}

pub(crate) struct MoeRuntimeConfig {
    pub(crate) vmm_mode: MoeExpertVmmMode,
    pub(crate) island_cap_experts: Option<usize>,
    pub(crate) protected_experts: Option<usize>,
    pub(crate) sparse_requested: bool,
    pub(crate) prefetch_mode: MoeIslandPrefetchMode,
    pub(crate) prefetch_ranks: usize,
    pub(crate) transition_min_observations: u32,
    pub(crate) async_prefetch: bool,
    pub(crate) async_staging_pages: usize,
    pub(crate) sparse_telemetry: Option<MoeSparseTelemetry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen36KvVmmMode {
    Auto,
    Disabled,
    Force,
}

pub(crate) fn moe_island_cap_experts_from_env() -> Result<Option<usize>> {
    let Some(raw) = std::env::var("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS").ok() else {
        return Ok(None);
    };
    let cap = raw.parse::<usize>().with_context(|| {
        format!("parse SUPERSONIC_MOE_ISLAND_CAP_EXPERTS={raw:?} as positive integer")
    })?;
    if cap == 0 {
        anyhow::bail!("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS must be > 0");
    }
    Ok(Some(cap))
}

pub(crate) fn moe_island_protected_experts_from_env_value(
    raw: Option<&str>,
) -> Result<Option<usize>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let value = raw.parse::<usize>().with_context(|| {
        format!("parse SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS={raw:?} as non-negative integer")
    })?;
    Ok((value > 0).then_some(value))
}

pub(crate) fn moe_island_protected_experts_from_env() -> Result<Option<usize>> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS").ok();
    moe_island_protected_experts_from_env_value(raw.as_deref())
}

pub(crate) fn moe_island_prefetch_ranks_from_env_value(
    raw: Option<&str>,
    mode: MoeIslandPrefetchMode,
    top_k: usize,
) -> Result<usize> {
    match mode {
        MoeIslandPrefetchMode::Disabled => {
            if raw.is_some() {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS requires \
                     SUPERSONIC_MOE_ISLAND_PREFETCH=previous-token, \
                     previous-token-resident, or transition"
                );
            }
            Ok(0)
        }
        MoeIslandPrefetchMode::PreviousToken
        | MoeIslandPrefetchMode::PreviousTokenResidentOnly
        | MoeIslandPrefetchMode::Transition => match raw {
            None | Some("all") => Ok(top_k),
            Some(value) => {
                let ranks = value.parse::<usize>().with_context(|| {
                    format!(
                        "parse SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS={value:?} as positive integer"
                    )
                })?;
                if ranks == 0 || ranks > top_k {
                    anyhow::bail!(
                        "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS must be in 1..={top_k}; got {ranks}"
                    );
                }
                Ok(ranks)
            }
        },
    }
}

pub(crate) fn moe_island_prefetch_mode_from_env_for_sparse(
    sparse_moe_requested: bool,
) -> Result<MoeIslandPrefetchMode> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH").ok();
    match raw.as_deref() {
        None if sparse_moe_requested => Ok(MoeIslandPrefetchMode::Transition),
        _ => MoeIslandPrefetchMode::from_env_value(raw.as_deref()),
    }
}

pub(crate) fn moe_island_prefetch_ranks_from_env_for_sparse(
    mode: MoeIslandPrefetchMode,
    top_k: usize,
    sparse_moe_requested: bool,
) -> Result<usize> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS").ok();
    if raw.is_none() && sparse_moe_requested && mode == MoeIslandPrefetchMode::Transition {
        return Ok(DEFAULT_SPARSE_MOE_PREFETCH_RANKS.min(top_k));
    }
    moe_island_prefetch_ranks_from_env_value(raw.as_deref(), mode, top_k)
}

pub(crate) fn moe_island_prefetch_transition_min_observations_from_env_value(
    raw: Option<&str>,
    mode: MoeIslandPrefetchMode,
) -> Result<u32> {
    if !mode.transition_weighted() {
        if raw.is_some() {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS requires \
                 SUPERSONIC_MOE_ISLAND_PREFETCH=transition"
            );
        }
        return Ok(0);
    }

    let Some(value) = raw else {
        return Ok(32);
    };
    value.parse::<u32>().with_context(|| {
        format!("parse SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS={value:?} as integer")
    })
}

pub(crate) fn moe_island_prefetch_transition_min_observations_for_sparse(
    mode: MoeIslandPrefetchMode,
    sparse_moe_requested: bool,
) -> Result<u32> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS").ok();
    if raw.is_none() && sparse_moe_requested && mode == MoeIslandPrefetchMode::Transition {
        return Ok(DEFAULT_SPARSE_MOE_TRANSITION_MIN_OBSERVATIONS);
    }
    moe_island_prefetch_transition_min_observations_from_env_value(raw.as_deref(), mode)
}

pub(crate) fn moe_island_async_prefetch_from_env_value(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") | Some("off") | Some("disabled") | Some("false") => Ok(false),
        Some("1") | Some("on") | Some("enabled") | Some("true") => Ok(true),
        Some(other) => Err(anyhow!(
            "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH must be unset, 0, 1, off, on, disabled, enabled, false, or true; got {other:?}"
        )),
    }
}

pub(crate) fn moe_island_async_prefetch_from_env() -> Result<bool> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH").ok();
    moe_island_async_prefetch_from_env_value(raw.as_deref())
}

pub(crate) fn moe_island_async_staging_pages_from_env_value(raw: Option<&str>) -> Result<usize> {
    let Some(raw) = raw else {
        return Ok(4);
    };
    let pages = raw.parse::<usize>().with_context(|| {
        format!("parse SUPERSONIC_MOE_ISLAND_ASYNC_STAGING_PAGES={raw:?} as positive integer")
    })?;
    if pages == 0 {
        anyhow::bail!("SUPERSONIC_MOE_ISLAND_ASYNC_STAGING_PAGES must be > 0");
    }
    Ok(pages)
}

pub(crate) fn moe_island_async_staging_pages_from_env() -> Result<usize> {
    let raw = std::env::var("SUPERSONIC_MOE_ISLAND_ASYNC_STAGING_PAGES").ok();
    moe_island_async_staging_pages_from_env_value(raw.as_deref())
}

pub(crate) fn prepare_moe_runtime_config(
    speculative_decode: bool,
    persistent_decode: bool,
    backend: Backend,
    top_k: usize,
) -> Result<MoeRuntimeConfig> {
    let vmm_mode = MoeExpertVmmMode::from_env()?;
    let island_cap_experts = moe_island_cap_experts_from_env()?;
    let protected_experts = moe_island_protected_experts_from_env()?;
    if island_cap_experts.is_some() && speculative_decode {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS sparse residency is not wired through speculative decode yet"
        );
    }
    if island_cap_experts.is_some() && vmm_mode == MoeExpertVmmMode::Disabled {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS requires VMM expert slabs; unset SUPERSONIC_VMM_MOE_ISLANDS=0"
        );
    }

    let sparse_requested = island_cap_experts.is_some();
    let prefetch_mode = moe_island_prefetch_mode_from_env_for_sparse(sparse_requested)?;
    let prefetch_ranks =
        moe_island_prefetch_ranks_from_env_for_sparse(prefetch_mode, top_k, sparse_requested)?;
    let transition_min_observations = moe_island_prefetch_transition_min_observations_for_sparse(
        prefetch_mode,
        sparse_requested,
    )?;
    let async_prefetch = moe_island_async_prefetch_from_env()?;
    let async_staging_pages = moe_island_async_staging_pages_from_env()?;
    if prefetch_mode != MoeIslandPrefetchMode::Disabled && !sparse_requested {
        anyhow::bail!("SUPERSONIC_MOE_ISLAND_PREFETCH requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS");
    }
    if async_prefetch {
        if !sparse_requested {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
            );
        }
        if backend != Backend::Hip {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH=1 is HIP-only in v1; backend={backend}"
            );
        }
        if prefetch_mode == MoeIslandPrefetchMode::Disabled {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH=1 requires SUPERSONIC_MOE_ISLAND_PREFETCH=previous-token, previous-token-resident, or transition"
            );
        }
    }
    if protected_experts.is_some() && !sparse_requested {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
        );
    }

    let sparse_telemetry = MoeSparseTelemetry::from_env(
        sparse_requested,
        persistent_decode,
        prefetch_mode,
        prefetch_ranks,
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
        vmm_mode,
        island_cap_experts,
        protected_experts,
        sparse_requested,
        prefetch_mode,
        prefetch_ranks,
        transition_min_observations,
        async_prefetch,
        async_staging_pages,
        sparse_telemetry,
    })
}

pub(crate) fn qwen36_kv_vmm_mode_from_env_value(
    raw: Option<&str>,
    backend: Backend,
) -> Result<Qwen36KvVmmMode> {
    match raw {
        None if backend == Backend::Hip => Ok(Qwen36KvVmmMode::Auto),
        None => Ok(Qwen36KvVmmMode::Disabled),
        Some("0") => Ok(Qwen36KvVmmMode::Disabled),
        Some("1") => Ok(Qwen36KvVmmMode::Force),
        Some(other) => Err(anyhow!(
            "SUPERSONIC_VMM_KV must be unset, 0, or 1 for Qwen3.6-MoE; got {other:?}"
        )),
    }
}

pub(crate) fn should_use_qwen36_kv_vmm(backend: Backend, ordinal: usize) -> Result<bool> {
    let mode = qwen36_kv_vmm_mode_from_env_value(
        std::env::var("SUPERSONIC_VMM_KV").ok().as_deref(),
        backend,
    )?;
    let requested = match mode {
        Qwen36KvVmmMode::Disabled => return Ok(false),
        Qwen36KvVmmMode::Auto | Qwen36KvVmmMode::Force => true,
    };
    if !gpu_hal::vmm_is_supported(backend, ordinal) {
        if mode == Qwen36KvVmmMode::Force {
            eprintln!(
                "[vmm] SUPERSONIC_VMM_KV=1 requested for Qwen3.6-MoE but backend={backend} \
                 device {ordinal} does not support VMM; using dense KV buffers"
            );
        } else {
            eprintln!(
                "[vmm] Qwen3.6-MoE HIP KV VMM auto-enable skipped because backend={backend} \
                 device {ordinal} does not support VMM; using dense KV buffers"
            );
        }
        return Ok(false);
    }
    Ok(requested)
}

pub(crate) fn should_try_moe_expert_vmm(
    mode: MoeExpertVmmMode,
    backend: Backend,
    has_int4_weights: bool,
    weight_mode_label: &str,
    ordinal: usize,
) -> Result<bool> {
    if mode == MoeExpertVmmMode::Disabled {
        return Ok(false);
    }
    if !has_int4_weights {
        if mode == MoeExpertVmmMode::Force {
            anyhow::bail!(
                "SUPERSONIC_VMM_MOE_ISLANDS=1 requires Qwen3.6-MoE INT4 weights; got {weight_mode_label}"
            );
        }
        return Ok(false);
    }
    let supported = gpu_hal::vmm_is_supported(backend, ordinal);
    if !supported {
        if mode == MoeExpertVmmMode::Force {
            anyhow::bail!(
                "SUPERSONIC_VMM_MOE_ISLANDS=1 requested but backend={backend} VMM is unsupported on device {ordinal}"
            );
        }
        return Ok(false);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{
        moe_island_prefetch_ranks_from_env_value,
        moe_island_prefetch_transition_min_observations_from_env_value,
        qwen36_kv_vmm_mode_from_env_value, Qwen36KvVmmMode,
    };
    use crate::qwen36_moe_telemetry::MoeIslandPrefetchMode;
    use gpu_hal::Backend;

    #[test]
    fn qwen36_kv_vmm_defaults_to_auto_on_hip_only() {
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, Backend::Hip).unwrap(),
            Qwen36KvVmmMode::Auto
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, Backend::Cuda).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, Backend::Metal).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
    }

    #[test]
    fn qwen36_kv_vmm_env_override_is_explicit() {
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(Some("0"), Backend::Hip).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(Some("1"), Backend::Cuda).unwrap(),
            Qwen36KvVmmMode::Force
        );
        assert!(qwen36_kv_vmm_mode_from_env_value(Some("yes"), Backend::Hip).is_err());
    }

    #[test]
    fn moe_prefetch_ranks_default_to_all_previous_token_routes() {
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(
                None,
                MoeIslandPrefetchMode::PreviousToken,
                8,
            )
            .unwrap(),
            8
        );
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(
                Some("all"),
                MoeIslandPrefetchMode::PreviousTokenResidentOnly,
                8,
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn moe_prefetch_ranks_accept_rank_limited_previous_token_routes() {
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(
                Some("1"),
                MoeIslandPrefetchMode::PreviousToken,
                8,
            )
            .unwrap(),
            1
        );
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(
                Some("4"),
                MoeIslandPrefetchMode::PreviousTokenResidentOnly,
                8,
            )
            .unwrap(),
            4
        );
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(
                Some("2"),
                MoeIslandPrefetchMode::Transition,
                8,
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn moe_prefetch_ranks_reject_disabled_or_out_of_range_values() {
        assert_eq!(
            moe_island_prefetch_ranks_from_env_value(None, MoeIslandPrefetchMode::Disabled, 8)
                .unwrap(),
            0
        );
        assert!(moe_island_prefetch_ranks_from_env_value(
            Some("1"),
            MoeIslandPrefetchMode::Disabled,
            8,
        )
        .is_err());
        assert!(moe_island_prefetch_ranks_from_env_value(
            Some("0"),
            MoeIslandPrefetchMode::PreviousToken,
            8,
        )
        .is_err());
        assert!(moe_island_prefetch_ranks_from_env_value(
            Some("9"),
            MoeIslandPrefetchMode::Transition,
            8,
        )
        .is_err());
    }

    #[test]
    fn moe_transition_min_observations_defaults_only_for_transition_mode() {
        assert_eq!(
            moe_island_prefetch_transition_min_observations_from_env_value(
                None,
                MoeIslandPrefetchMode::Transition,
            )
            .unwrap(),
            32
        );
        assert_eq!(
            moe_island_prefetch_transition_min_observations_from_env_value(
                Some("4"),
                MoeIslandPrefetchMode::Transition,
            )
            .unwrap(),
            4
        );
        assert_eq!(
            moe_island_prefetch_transition_min_observations_from_env_value(
                None,
                MoeIslandPrefetchMode::PreviousToken,
            )
            .unwrap(),
            0
        );
        assert!(
            moe_island_prefetch_transition_min_observations_from_env_value(
                Some("4"),
                MoeIslandPrefetchMode::PreviousToken,
            )
            .is_err()
        );
    }
}
