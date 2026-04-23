use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::Cli;

#[derive(Debug, Clone)]
pub(crate) struct CertifiedKvConfig {
    pub block_size: usize,
    pub value_group_size: usize,
    pub bf16_values: bool,
    pub tau_cov: f32,
    pub k_min: usize,
    pub k_max: usize,
    pub v_tol: f32,
    pub ranking_r: usize,
    pub rung1_threshold: f32,
    pub rung1_multiplier: f32,
    pub eps_guard: f32,
    pub telemetry_path: Option<PathBuf>,
}

impl CertifiedKvConfig {
    pub(crate) fn from_cli(cli: &Cli) -> Result<Self> {
        let cfg = Self {
            block_size: cli.certified_kv_block_size,
            value_group_size: cli.certified_kv_value_group_size,
            bf16_values: cli.certified_kv_bf16_values,
            tau_cov: cli.certified_kv_tau_cov,
            k_min: cli.certified_kv_k_min,
            k_max: cli.certified_kv_k_max,
            v_tol: cli.certified_kv_v_tol,
            ranking_r: cli.certified_kv_ranking_r,
            rung1_threshold: cli.certified_kv_rung1_threshold,
            rung1_multiplier: cli.certified_kv_rung1_multiplier,
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
        if self.k_max < self.k_min {
            bail!("--certified-kv-k-max must be >= --certified-kv-k-min");
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
        if self.eps_guard < 0.0 {
            bail!("--certified-kv-eps-guard must be >= 0");
        }
        Ok(())
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "block={} value_group={} value_mode={} tau_cov={:.6} k_min={} k_max={} v_tol={:.6} ranking_r={} rung1_threshold={:.6} rung1_multiplier={:.3} eps_guard={:.6} telemetry={}",
            self.block_size,
            self.value_group_size,
            if self.bf16_values { "bf16" } else { "int4" },
            self.tau_cov,
            self.k_min,
            self.k_max,
            self.v_tol,
            self.ranking_r,
            self.rung1_threshold,
            self.rung1_multiplier,
            self.eps_guard,
            self.telemetry_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none".to_string())
        )
    }
}
