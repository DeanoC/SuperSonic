use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::Cli;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum CertifiedKvPreset {
    Legacy,
    PaperV2,
}

impl CertifiedKvPreset {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::PaperV2 => "paper-v2",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CertifiedKvConfig {
    pub preset: CertifiedKvPreset,
    pub block_size: usize,
    pub value_group_size: usize,
    pub bf16_values: bool,
    pub tau_cov: f32,
    pub k_min: usize,
    pub k_max: Option<usize>,
    pub v_tol: f32,
    pub value_cache_blocks: usize,
    pub ranking_r: usize,
    pub rung1_threshold: f32,
    pub rung1_multiplier: f32,
    pub key_cache_blocks: usize,
    pub delta_guard_factor: f32,
    pub score_exploration_rate: f32,
    pub require_certified_tail_bound: bool,
    pub eps_guard: f32,
    pub telemetry_path: Option<PathBuf>,
}

impl CertifiedKvConfig {
    pub(crate) fn from_cli(cli: &Cli) -> Result<Self> {
        let mut tau_cov = cli.certified_kv_tau_cov;
        let mut k_min = cli.certified_kv_k_min;
        let mut k_max = (cli.certified_kv_k_max != 0).then_some(cli.certified_kv_k_max);
        let mut v_tol = cli.certified_kv_v_tol;
        let mut rung1_threshold = cli.certified_kv_rung1_threshold;
        let mut rung1_multiplier = cli.certified_kv_rung1_multiplier;

        if cli.certified_kv_preset == CertifiedKvPreset::PaperV2 {
            tau_cov = 0.995;
            k_min = 2;
            k_max = None;
            v_tol = 0.5;
            rung1_threshold = 0.02;
            rung1_multiplier = 2.0;
        }

        let cfg = Self {
            preset: cli.certified_kv_preset,
            block_size: cli.certified_kv_block_size,
            value_group_size: cli.certified_kv_value_group_size,
            bf16_values: cli.certified_kv_bf16_values,
            tau_cov,
            k_min,
            k_max,
            v_tol,
            value_cache_blocks: cli.certified_kv_value_cache_blocks,
            ranking_r: cli.certified_kv_ranking_r,
            rung1_threshold,
            rung1_multiplier,
            key_cache_blocks: cli.certified_kv_key_cache_blocks,
            delta_guard_factor: cli.certified_kv_delta_guard_factor,
            score_exploration_rate: cli.certified_kv_score_exploration_rate,
            require_certified_tail_bound: !cli.certified_kv_allow_uncertified_tail,
            eps_guard: cli.certified_kv_eps_guard,
            telemetry_path: cli.certified_kv_telemetry.clone(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.block_size == 0 {
            bail!("--certified-kv-block-size must be > 0");
        }
        if self.value_group_size == 0 {
            bail!("--certified-kv-value-group-size must be > 0");
        }
        if self.value_group_size % 2 != 0 {
            bail!("--certified-kv-value-group-size must be even for INT4 packing");
        }
        if !(0.0..=1.0).contains(&self.tau_cov) || self.tau_cov == 0.0 {
            bail!("--certified-kv-tau-cov must be in (0, 1]");
        }
        if self.k_min == 0 {
            bail!("--certified-kv-k-min must be > 0");
        }
        if let Some(k_max) = self.k_max {
            if k_max < self.k_min {
                bail!("--certified-kv-k-max must be >= --certified-kv-k-min, or 0 for unclamped");
            }
        }
        if self.v_tol < 0.0 {
            bail!("--certified-kv-v-tol must be >= 0");
        }
        if self.ranking_r == 0 {
            bail!("--certified-kv-ranking-r must be > 0");
        }
        if self.rung1_threshold < 0.0 {
            bail!("--certified-kv-rung1-threshold must be >= 0");
        }
        if self.rung1_multiplier < 1.0 {
            bail!("--certified-kv-rung1-multiplier must be >= 1");
        }
        if let Some(k_max) = self.k_max {
            let expanded_key_budget =
                k_max.saturating_mul(self.rung1_multiplier.ceil().max(1.0) as usize);
            if self.key_cache_blocks < expanded_key_budget {
                bail!(
                    "--certified-kv-key-cache-blocks must be >= ceil(--certified-kv-rung1-multiplier) * --certified-kv-k-max"
                );
            }
        } else if self.key_cache_blocks == 0 {
            bail!("--certified-kv-key-cache-blocks must be > 0 when --certified-kv-k-max=0");
        }
        if self.delta_guard_factor < 0.0 {
            bail!("--certified-kv-delta-guard-factor must be >= 0");
        }
        if !(0.0..=1.0).contains(&self.score_exploration_rate) {
            bail!("--certified-kv-score-exploration-rate must be in [0, 1]");
        }
        if self.eps_guard < 0.0 {
            bail!("--certified-kv-eps-guard must be >= 0");
        }
        Ok(())
    }

    pub(crate) fn k_max_mode(&self) -> &'static str {
        if self.k_max.is_some() {
            "clamped"
        } else {
            "unclamped"
        }
    }

    pub(crate) fn k_max_effective(&self, num_blocks: usize) -> usize {
        let floor = self.k_min.min(num_blocks.max(1));
        self.k_max.unwrap_or(num_blocks).min(num_blocks).max(floor)
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "preset={} block={} value_group={} value_mode={} tau_cov={:.6} k_min={} k_max={} v_tol={:.6} key_cache_blocks={} value_cache_blocks={} ranking_r={} rung1_threshold={:.6} rung1_multiplier={:.3} delta_guard_factor={:.3} score_exploration_rate={:.3} require_certified_tail_bound={} eps_guard={:.6} telemetry={}",
            self.preset.as_str(),
            self.block_size,
            self.value_group_size,
            if self.bf16_values { "bf16" } else { "int4" },
            self.tau_cov,
            self.k_min,
            self.k_max
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unclamped".to_string()),
            self.v_tol,
            self.key_cache_blocks,
            self.value_cache_blocks,
            self.ranking_r,
            self.rung1_threshold,
            self.rung1_multiplier,
            self.delta_guard_factor,
            self.score_exploration_rate,
            self.require_certified_tail_bound,
            self.eps_guard,
            self.telemetry_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none".to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> CertifiedKvConfig {
        CertifiedKvConfig {
            preset: CertifiedKvPreset::Legacy,
            block_size: 16,
            value_group_size: 16,
            bf16_values: false,
            tau_cov: 0.995,
            k_min: 2,
            k_max: Some(128),
            v_tol: 0.05,
            value_cache_blocks: 128,
            ranking_r: 1,
            rung1_threshold: 0.005,
            rung1_multiplier: 2.0,
            key_cache_blocks: 256,
            delta_guard_factor: 3.0,
            score_exploration_rate: 0.01,
            require_certified_tail_bound: true,
            eps_guard: 0.0001,
            telemetry_path: None,
        }
    }

    #[test]
    fn legacy_defaults_remain_current_values() {
        let cfg = base_cfg();
        cfg.validate().unwrap();
        assert_eq!(cfg.preset, CertifiedKvPreset::Legacy);
        assert_eq!(cfg.tau_cov, 0.995);
        assert_eq!(cfg.k_min, 2);
        assert_eq!(cfg.k_max, Some(128));
        assert_eq!(cfg.v_tol, 0.05);
        assert_eq!(cfg.rung1_threshold, 0.005);
        assert_eq!(cfg.rung1_multiplier, 2.0);
    }

    #[test]
    fn unclamped_k_max_uses_num_blocks_as_effective_clamp() {
        let mut cfg = base_cfg();
        cfg.k_max = None;
        cfg.key_cache_blocks = 16;
        cfg.validate().unwrap();
        assert_eq!(cfg.k_max_mode(), "unclamped");
        assert_eq!(cfg.k_max_effective(13), 13);
        assert_eq!(cfg.k_max_effective(1), 1);
    }

    #[test]
    fn clamped_k_max_keeps_existing_key_cache_validation() {
        let mut cfg = base_cfg();
        cfg.key_cache_blocks = 127;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("--certified-kv-key-cache-blocks"));
    }
}
