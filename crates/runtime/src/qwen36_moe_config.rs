use anyhow::{anyhow, Context, Result};

pub const DEFAULT_MOE_PREFETCH_EVICT_MIN_PROBABILITY: f64 = 0.90;

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
    pub prefetch_mode: MoeIslandPrefetchMode,
    pub prefetch_ranks: usize,
    pub async_prefetch: bool,
    pub async_staging_pages: usize,
    pub prefetch_evict: bool,
    pub prefetch_evict_min_probability: f64,
}

impl Default for Qwen36MoeRuntimeConfig {
    fn default() -> Self {
        Self {
            vmm_mode: MoeExpertVmmMode::Auto,
            island_cap_experts: None,
            protected_experts: None,
            fixed_hot_experts: None,
            prefetch_mode: MoeIslandPrefetchMode::Disabled,
            prefetch_ranks: 4,
            async_prefetch: false,
            async_staging_pages: 2,
            prefetch_evict: false,
            prefetch_evict_min_probability: DEFAULT_MOE_PREFETCH_EVICT_MIN_PROBABILITY,
        }
    }
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
}
