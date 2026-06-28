use anyhow::{anyhow, Context, Result};
use gpu_hal::Backend;

pub const DEFAULT_MOE_PREFETCH_EVICT_MIN_PROBABILITY: f64 = 0.90;
pub const DEFAULT_SPARSE_MOE_PREFETCH_RANKS: usize = 4;
pub const DEFAULT_SPARSE_MOE_TRANSITION_MIN_OBSERVATIONS: u32 = 1;
pub const DEFAULT_MOE_ASYNC_STAGING_PAGES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpertVmmMode {
    Auto,
    Disabled,
    Force,
}

impl MoeExpertVmmMode {
    pub fn from_env_value(raw: Option<&str>) -> Result<Self> {
        parse_moe_expert_vmm_mode("SUPERSONIC_VMM_MOE_ISLANDS", raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeIslandPrefetchMode {
    Disabled,
    PreviousToken,
    PreviousTokenResidentOnly,
    Transition,
}

impl MoeIslandPrefetchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PreviousToken => "previous-token",
            Self::PreviousTokenResidentOnly => "previous-token-resident",
            Self::Transition => "transition",
        }
    }

    pub fn uses_previous_token_routes(self) -> bool {
        matches!(
            self,
            Self::PreviousToken | Self::PreviousTokenResidentOnly | Self::Transition
        )
    }

    pub fn resident_only(self) -> bool {
        matches!(self, Self::PreviousTokenResidentOnly)
    }

    pub fn transition_weighted(self) -> bool {
        matches!(self, Self::Transition)
    }

    pub fn from_env_value(raw: Option<&str>) -> Result<Self> {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36MoeRuntimeConfig {
    pub vmm_mode: MoeExpertVmmMode,
    pub island_cap_experts: Option<usize>,
    pub protected_experts: Option<usize>,
    pub fixed_hot_experts: Option<usize>,
    pub sparse_requested: bool,
    pub prefetch_mode: MoeIslandPrefetchMode,
    pub prefetch_ranks: usize,
    pub transition_min_observations: u32,
    pub async_prefetch: bool,
    pub async_staging_pages: usize,
    pub prefetch_evict: bool,
    pub prefetch_evict_min_probability: f64,
    pub protect_demand_routes: bool,
    pub hot_protect_min_hits: Option<u32>,
    pub fixed_hot_min_hits: Option<u32>,
}

impl Default for Qwen36MoeRuntimeConfig {
    fn default() -> Self {
        Self {
            vmm_mode: MoeExpertVmmMode::Auto,
            island_cap_experts: None,
            protected_experts: None,
            fixed_hot_experts: None,
            sparse_requested: false,
            prefetch_mode: MoeIslandPrefetchMode::Disabled,
            prefetch_ranks: 0,
            transition_min_observations: 0,
            async_prefetch: false,
            async_staging_pages: DEFAULT_MOE_ASYNC_STAGING_PAGES,
            prefetch_evict: false,
            prefetch_evict_min_probability: DEFAULT_MOE_PREFETCH_EVICT_MIN_PROBABILITY,
            protect_demand_routes: false,
            hot_protect_min_hits: None,
            fixed_hot_min_hits: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen36MoeRuntimeConfigInputs<'a> {
    pub vmm_mode: Option<&'a str>,
    pub island_cap_experts: Option<&'a str>,
    pub protected_experts: Option<&'a str>,
    pub fixed_hot_experts: Option<&'a str>,
    pub prefetch_mode: Option<&'a str>,
    pub prefetch_ranks: Option<&'a str>,
    pub transition_min_observations: Option<&'a str>,
    pub async_prefetch: Option<&'a str>,
    pub async_staging_pages: Option<&'a str>,
    pub prefetch_evict: Option<&'a str>,
    pub prefetch_evict_min_probability: Option<&'a str>,
    pub protect_demand_routes: Option<&'a str>,
    pub hot_protect_min_hits: Option<&'a str>,
    pub fixed_hot_min_hits: Option<&'a str>,
}

impl Qwen36MoeRuntimeConfig {
    pub fn from_inputs(
        inputs: &Qwen36MoeRuntimeConfigInputs<'_>,
        speculative_decode: bool,
        backend: Backend,
        top_k: usize,
    ) -> Result<Self> {
        let mut vmm_mode =
            parse_moe_expert_vmm_mode("SUPERSONIC_VMM_MOE_ISLANDS", inputs.vmm_mode)?;
        if backend == Backend::Cuda {
            if vmm_mode == MoeExpertVmmMode::Force {
                anyhow::bail!(
                    "SUPERSONIC_VMM_MOE_ISLANDS=1 is not supported for Qwen3.6-MoE CUDA v1; \
                     unset it or use SUPERSONIC_VMM_MOE_ISLANDS=0"
                );
            }
            vmm_mode = MoeExpertVmmMode::Disabled;
        }

        let island_cap_experts = parse_optional_positive_usize(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS",
            inputs.island_cap_experts,
        )?;
        let protected_experts = parse_optional_nonzero_usize(
            "SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS",
            inputs.protected_experts,
        )?;
        let fixed_hot_experts = parse_optional_nonzero_usize(
            "SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS",
            inputs.fixed_hot_experts,
        )?;
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
        let prefetch_mode = match inputs.prefetch_mode {
            None if sparse_requested => MoeIslandPrefetchMode::Transition,
            raw => MoeIslandPrefetchMode::from_env_value(raw)?,
        };
        let prefetch_ranks = if inputs.prefetch_ranks.is_none()
            && sparse_requested
            && prefetch_mode == MoeIslandPrefetchMode::Transition
        {
            DEFAULT_SPARSE_MOE_PREFETCH_RANKS.min(top_k)
        } else {
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
                inputs.prefetch_ranks,
                prefetch_mode,
                top_k,
            )?
        };
        let transition_min_observations = if inputs.transition_min_observations.is_none()
            && sparse_requested
            && prefetch_mode == MoeIslandPrefetchMode::Transition
        {
            DEFAULT_SPARSE_MOE_TRANSITION_MIN_OBSERVATIONS
        } else {
            parse_moe_transition_min_observations(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS",
                inputs.transition_min_observations,
                prefetch_mode,
            )?
        };
        let async_prefetch = parse_bool_flag(
            "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH",
            inputs.async_prefetch,
        )?;
        let async_staging_pages = parse_positive_usize_with_default(
            "SUPERSONIC_MOE_ISLAND_ASYNC_STAGING_PAGES",
            inputs.async_staging_pages,
            DEFAULT_MOE_ASYNC_STAGING_PAGES,
        )?;
        let prefetch_evict = parse_bool_flag(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT",
            inputs.prefetch_evict,
        )?;
        let prefetch_evict_min_probability = parse_unit_interval_with_default(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
            inputs.prefetch_evict_min_probability,
            DEFAULT_MOE_PREFETCH_EVICT_MIN_PROBABILITY,
        )?;
        let protect_demand_routes = parse_bool_flag(
            "SUPERSONIC_MOE_ISLAND_PROTECT_DEMAND",
            inputs.protect_demand_routes,
        )?;
        let hot_protect_min_hits = parse_optional_positive_u32_or_disabled(
            "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
            inputs.hot_protect_min_hits,
        )?;
        let fixed_hot_min_hits = parse_optional_positive_u32_or_disabled(
            "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
            inputs.fixed_hot_min_hits,
        )?;

        if prefetch_mode != MoeIslandPrefetchMode::Disabled && !sparse_requested {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_PREFETCH requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
            );
        }
        if prefetch_evict {
            if !sparse_requested {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
                );
            }
            if prefetch_mode == MoeIslandPrefetchMode::Disabled {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT=1 requires SUPERSONIC_MOE_ISLAND_PREFETCH=previous-token, previous-token-resident, or transition"
                );
            }
        }
        if !prefetch_evict && inputs.prefetch_evict_min_probability.is_some() {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB requires SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT=1"
            );
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
        if fixed_hot_experts.is_some() && !sparse_requested {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
            );
        }
        if protect_demand_routes {
            if !sparse_requested {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_PROTECT_DEMAND requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
                );
            }
            if protected_experts.is_none() {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_PROTECT_DEMAND=1 requires SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS"
                );
            }
        }
        if hot_protect_min_hits.is_some() {
            if !sparse_requested {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
                );
            }
            if protected_experts.is_none() {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS requires SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS"
                );
            }
        }
        if fixed_hot_min_hits.is_some() {
            if !sparse_requested {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"
                );
            }
            if fixed_hot_experts.is_none() {
                anyhow::bail!(
                    "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS requires SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS"
                );
            }
        }
        if fixed_hot_experts.is_some() && fixed_hot_min_hits.is_none() {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS requires SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS"
            );
        }

        Ok(Self {
            vmm_mode,
            island_cap_experts,
            protected_experts,
            fixed_hot_experts,
            sparse_requested,
            prefetch_mode,
            prefetch_ranks,
            transition_min_observations,
            async_prefetch,
            async_staging_pages,
            prefetch_evict,
            prefetch_evict_min_probability,
            protect_demand_routes,
            hot_protect_min_hits,
            fixed_hot_min_hits,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36KvVmmMode {
    Auto,
    Disabled,
    Force,
}

pub fn qwen36_kv_vmm_mode_from_env_value(
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

pub fn should_use_qwen36_kv_vmm(
    mode: Qwen36KvVmmMode,
    backend: Backend,
    ordinal: usize,
) -> Result<bool> {
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

pub fn should_try_moe_expert_vmm(
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

pub fn parse_moe_expert_vmm_mode(name: &str, raw: Option<&str>) -> Result<MoeExpertVmmMode> {
    match raw {
        None => Ok(MoeExpertVmmMode::Auto),
        Some("0") => Ok(MoeExpertVmmMode::Disabled),
        Some("1") => Ok(MoeExpertVmmMode::Force),
        Some(other) => Err(anyhow!("{name} must be unset, 0, or 1; got {other:?}")),
    }
}

pub fn parse_optional_positive_usize(name: &str, raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("parse {name}={raw:?} as positive integer"))?;
    if value == 0 {
        anyhow::bail!("{name} must be > 0");
    }
    Ok(Some(value))
}

pub fn parse_optional_nonzero_usize(name: &str, raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("parse {name}={raw:?} as non-negative integer"))?;
    Ok((value > 0).then_some(value))
}

pub fn parse_moe_prefetch_ranks(
    name: &str,
    raw: Option<&str>,
    mode: MoeIslandPrefetchMode,
    top_k: usize,
) -> Result<usize> {
    match mode {
        MoeIslandPrefetchMode::Disabled => {
            if raw.is_some() {
                anyhow::bail!(
                    "{name} requires SUPERSONIC_MOE_ISLAND_PREFETCH=previous-token, \
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
                let ranks = value
                    .parse::<usize>()
                    .with_context(|| format!("parse {name}={value:?} as positive integer"))?;
                if ranks == 0 || ranks > top_k {
                    anyhow::bail!("{name} must be in 1..={top_k}; got {ranks}");
                }
                Ok(ranks)
            }
        },
    }
}

pub fn parse_moe_transition_min_observations(
    name: &str,
    raw: Option<&str>,
    mode: MoeIslandPrefetchMode,
) -> Result<u32> {
    if !mode.transition_weighted() {
        if raw.is_some() {
            anyhow::bail!("{name} requires SUPERSONIC_MOE_ISLAND_PREFETCH=transition");
        }
        return Ok(0);
    }

    let Some(value) = raw else {
        return Ok(32);
    };
    value
        .parse::<u32>()
        .with_context(|| format!("parse {name}={value:?} as integer"))
}

pub fn parse_bool_flag(name: &str, raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") | Some("off") | Some("disabled") | Some("false") => Ok(false),
        Some("1") | Some("on") | Some("enabled") | Some("true") => Ok(true),
        Some(other) => Err(anyhow!(
            "{name} must be unset, 0, 1, off, on, disabled, enabled, false, or true; got {other:?}"
        )),
    }
}

pub fn parse_positive_usize_with_default(
    name: &str,
    raw: Option<&str>,
    default: usize,
) -> Result<usize> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("parse {name}={raw:?} as positive integer"))?;
    if value == 0 {
        anyhow::bail!("{name} must be > 0");
    }
    Ok(value)
}

pub fn parse_unit_interval_with_default(
    name: &str,
    raw: Option<&str>,
    default: f64,
) -> Result<f64> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    let probability = raw
        .parse::<f64>()
        .with_context(|| format!("parse {name}={raw:?} as probability"))?;
    if !(0.0..=1.0).contains(&probability) {
        anyhow::bail!("{name} must be in 0.0..=1.0; got {probability}");
    }
    Ok(probability)
}

pub fn parse_optional_positive_u32_or_disabled(
    name: &str,
    raw: Option<&str>,
) -> Result<Option<u32>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    match raw {
        "0" | "off" | "disabled" | "false" => return Ok(None),
        _ => {}
    }
    let value = raw
        .parse::<u32>()
        .with_context(|| format!("parse {name}={raw:?} as positive integer"))?;
    if value == 0 {
        anyhow::bail!("{name} must be > 0");
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_positive_usize_accepts_absent_and_positive() {
        assert_eq!(parse_optional_positive_usize("X", None).unwrap(), None);
        assert_eq!(
            parse_optional_positive_usize("X", Some("7")).unwrap(),
            Some(7)
        );
    }

    #[test]
    fn optional_positive_usize_rejects_zero_and_bad_values() {
        assert!(parse_optional_positive_usize("X", Some("0")).is_err());
        assert!(parse_optional_positive_usize("X", Some("abc")).is_err());
    }

    #[test]
    fn prefetch_mode_accepts_telemetry_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("prev-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous_token_resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition-weighted")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert!(MoeIslandPrefetchMode::from_env_value(Some("resident")).is_err());
    }

    #[test]
    fn boolean_parser_accepts_existing_aliases() {
        assert!(!parse_bool_flag("X", None).unwrap());
        assert!(!parse_bool_flag("X", Some("off")).unwrap());
        assert!(parse_bool_flag("X", Some("1")).unwrap());
        assert!(parse_bool_flag("X", Some("enabled")).unwrap());
        assert!(parse_bool_flag("X", Some("yes")).is_err());
    }

    #[test]
    fn unit_interval_parser_rejects_out_of_range_values() {
        assert_eq!(
            parse_unit_interval_with_default("X", None, 0.9).unwrap(),
            0.9
        );
        assert_eq!(
            parse_unit_interval_with_default("X", Some("1.0"), 0.9).unwrap(),
            1.0
        );
        assert!(parse_unit_interval_with_default("X", Some("1.1"), 0.9).is_err());
    }

    #[test]
    fn qwen36_kv_vmm_defaults_to_auto_on_hip_only() {
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, gpu_hal::Backend::Hip).unwrap(),
            Qwen36KvVmmMode::Auto
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, gpu_hal::Backend::Cuda).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(None, gpu_hal::Backend::Metal).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
    }

    #[test]
    fn qwen36_kv_vmm_env_override_is_explicit() {
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(Some("0"), gpu_hal::Backend::Hip).unwrap(),
            Qwen36KvVmmMode::Disabled
        );
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(Some("1"), gpu_hal::Backend::Cuda).unwrap(),
            Qwen36KvVmmMode::Force
        );
        assert!(qwen36_kv_vmm_mode_from_env_value(Some("yes"), gpu_hal::Backend::Hip).is_err());
    }

    #[test]
    fn moe_prefetch_ranks_default_to_all_previous_token_routes() {
        assert_eq!(
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
                None,
                MoeIslandPrefetchMode::PreviousToken,
                8,
            )
            .unwrap(),
            8
        );
        assert_eq!(
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
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
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
                Some("1"),
                MoeIslandPrefetchMode::PreviousToken,
                8,
            )
            .unwrap(),
            1
        );
        assert_eq!(
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
                Some("4"),
                MoeIslandPrefetchMode::PreviousTokenResidentOnly,
                8,
            )
            .unwrap(),
            4
        );
        assert_eq!(
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
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
            parse_moe_prefetch_ranks(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
                None,
                MoeIslandPrefetchMode::Disabled,
                8,
            )
            .unwrap(),
            0
        );
        assert!(parse_moe_prefetch_ranks(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
            Some("1"),
            MoeIslandPrefetchMode::Disabled,
            8,
        )
        .is_err());
        assert!(parse_moe_prefetch_ranks(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
            Some("0"),
            MoeIslandPrefetchMode::PreviousToken,
            8,
        )
        .is_err());
        assert!(parse_moe_prefetch_ranks(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
            Some("9"),
            MoeIslandPrefetchMode::Transition,
            8,
        )
        .is_err());
    }

    #[test]
    fn moe_transition_min_observations_defaults_only_for_transition_mode() {
        assert_eq!(
            parse_moe_transition_min_observations(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS",
                None,
                MoeIslandPrefetchMode::Transition,
            )
            .unwrap(),
            32
        );
        assert_eq!(
            parse_moe_transition_min_observations(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS",
                Some("4"),
                MoeIslandPrefetchMode::Transition,
            )
            .unwrap(),
            4
        );
        assert_eq!(
            parse_moe_transition_min_observations(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS",
                None,
                MoeIslandPrefetchMode::PreviousToken,
            )
            .unwrap(),
            0
        );
        assert!(parse_moe_transition_min_observations(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_TRANSITION_MIN_OBS",
            Some("4"),
            MoeIslandPrefetchMode::PreviousToken,
        )
        .is_err());
    }

    #[test]
    fn moe_prefetch_evict_min_probability_accepts_unit_interval() {
        assert_eq!(
            parse_unit_interval_with_default(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
                None,
                0.90
            )
            .unwrap(),
            0.90
        );
        assert_eq!(
            parse_unit_interval_with_default(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
                Some("0"),
                0.90,
            )
            .unwrap(),
            0.0
        );
        assert_eq!(
            parse_unit_interval_with_default(
                "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
                Some("1.0"),
                0.90,
            )
            .unwrap(),
            1.0
        );
        assert!(parse_unit_interval_with_default(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
            Some("-0.1"),
            0.90,
        )
        .is_err());
        assert!(parse_unit_interval_with_default(
            "SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB",
            Some("1.1"),
            0.90,
        )
        .is_err());
    }

    #[test]
    fn moe_hot_protect_min_hits_accepts_positive_integer_or_disabled() {
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
                None,
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
                Some("0"),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
                Some("off"),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
                Some("32"),
            )
            .unwrap(),
            Some(32)
        );
        assert!(parse_optional_positive_u32_or_disabled(
            "SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS",
            Some("yes"),
        )
        .is_err());
    }

    #[test]
    fn moe_fixed_hot_experts_accepts_non_negative_integer() {
        assert_eq!(
            parse_optional_nonzero_usize("SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS", None).unwrap(),
            None
        );
        assert_eq!(
            parse_optional_nonzero_usize("SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS", Some("0"))
                .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_nonzero_usize("SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS", Some("64"))
                .unwrap(),
            Some(64)
        );
        assert!(parse_optional_nonzero_usize(
            "SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS",
            Some("yes")
        )
        .is_err());
    }

    #[test]
    fn moe_fixed_hot_min_hits_accepts_positive_integer_or_disabled() {
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
                None,
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
                Some("0"),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
                Some("off"),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_optional_positive_u32_or_disabled(
                "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
                Some("32"),
            )
            .unwrap(),
            Some(32)
        );
        assert!(parse_optional_positive_u32_or_disabled(
            "SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS",
            Some("yes"),
        )
        .is_err());
    }

    #[test]
    fn runtime_policy_sparse_defaults_live_in_runtime() {
        let inputs = Qwen36MoeRuntimeConfigInputs {
            island_cap_experts: Some("16"),
            ..Default::default()
        };

        let policy =
            Qwen36MoeRuntimeConfig::from_inputs(&inputs, false, gpu_hal::Backend::Hip, 8).unwrap();

        assert_eq!(policy.vmm_mode, MoeExpertVmmMode::Auto);
        assert_eq!(policy.island_cap_experts, Some(16));
        assert!(policy.sparse_requested);
        assert_eq!(policy.prefetch_mode, MoeIslandPrefetchMode::Transition);
        assert_eq!(policy.prefetch_ranks, 4);
        assert_eq!(policy.transition_min_observations, 1);
        assert!(!policy.async_prefetch);
        assert_eq!(policy.async_staging_pages, 4);
        assert!(!policy.prefetch_evict);
        assert_eq!(policy.prefetch_evict_min_probability, 0.90);
    }

    #[test]
    fn runtime_policy_rejects_cross_field_sparse_misconfigurations() {
        let prefetch_without_sparse = Qwen36MoeRuntimeConfigInputs {
            prefetch_mode: Some("previous-token"),
            ..Default::default()
        };
        assert!(Qwen36MoeRuntimeConfig::from_inputs(
            &prefetch_without_sparse,
            false,
            gpu_hal::Backend::Hip,
            8,
        )
        .is_err());

        let fixed_without_min_hits = Qwen36MoeRuntimeConfigInputs {
            island_cap_experts: Some("16"),
            fixed_hot_experts: Some("4"),
            ..Default::default()
        };
        assert!(Qwen36MoeRuntimeConfig::from_inputs(
            &fixed_without_min_hits,
            false,
            gpu_hal::Backend::Hip,
            8,
        )
        .is_err());
    }

    #[test]
    fn runtime_policy_forces_cuda_moe_vmm_disabled() {
        let inputs = Qwen36MoeRuntimeConfigInputs::default();

        let policy =
            Qwen36MoeRuntimeConfig::from_inputs(&inputs, false, gpu_hal::Backend::Cuda, 8).unwrap();
        assert_eq!(policy.vmm_mode, MoeExpertVmmMode::Disabled);

        let forced = Qwen36MoeRuntimeConfigInputs {
            vmm_mode: Some("1"),
            ..Default::default()
        };
        assert!(
            Qwen36MoeRuntimeConfig::from_inputs(&forced, false, gpu_hal::Backend::Cuda, 8).is_err()
        );
    }

    #[test]
    fn qwen36_kv_vmm_mode_is_runtime_owned() {
        assert_eq!(
            qwen36_kv_vmm_mode_from_env_value(Some("1"), gpu_hal::Backend::Metal).unwrap(),
            Qwen36KvVmmMode::Force
        );
    }
}
