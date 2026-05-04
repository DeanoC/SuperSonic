use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::ValueEnum;
use gpu_hal::Backend;
use qwen35::config::TextConfig;
use serde::{Deserialize, Serialize};
use supersonic_core::backend::BackendChoice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    Auto,
    Cuda,
    Hip,
    Metal,
}

impl Default for BackendArg {
    fn default() -> Self {
        Self::Metal
    }
}

impl From<BackendArg> for BackendChoice {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => Self::Auto,
            BackendArg::Cuda => Self::Explicit(Backend::Cuda),
            BackendArg::Hip => Self::Explicit(Backend::Hip),
            BackendArg::Metal => Self::Explicit(Backend::Metal),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BughuntMode {
    Gate,
    DecodeGate,
    Localize,
    Dump,
    Bench,
}

impl BughuntMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::DecodeGate => "decode_gate",
            Self::Localize => "localize",
            Self::Dump => "dump",
            Self::Bench => "bench",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BughuntLayerKind {
    Linear,
    Full,
    Mlp,
}

impl BughuntLayerKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Full => "full",
            Self::Mlp => "mlp",
        }
    }

    pub(crate) fn from_model_layer(config: &TextConfig, layer: usize) -> Self {
        if config.is_full_attention(layer) {
            Self::Full
        } else {
            Self::Linear
        }
    }
}

#[derive(Debug, Clone)]
pub struct BughuntArgs {
    pub mode: BughuntMode,
    pub model_dir: PathBuf,
    pub backend: BackendArg,
    pub ordinal: usize,
    pub oracle_device: String,
    pub prompt_manifest: PathBuf,
    pub prompt: Option<String>,
    pub report_json: Option<PathBuf>,
    pub position: Option<usize>,
    pub layer: Option<usize>,
    pub layer_kind: Option<BughuntLayerKind>,
    pub bench_iterations: usize,
    pub bench_warmup: usize,
    pub bench_decode_tokens: usize,
    pub bench_profile_ops: bool,
}

pub(crate) fn validate_args(args: &BughuntArgs) -> Result<()> {
    if args.layer.is_some() && args.layer_kind.is_none() {
        bail!("--layer-kind is required when --layer is provided");
    }
    if args.layer.is_none() && args.layer_kind.is_some() {
        bail!("--layer-kind requires --layer");
    }
    if matches!(args.mode, BughuntMode::Dump) && args.prompt.is_none() {
        bail!("--prompt is required in dump mode");
    }
    if matches!(args.mode, BughuntMode::Bench) && args.bench_iterations == 0 {
        bail!("--iters must be greater than zero in bench mode");
    }
    if matches!(args.mode, BughuntMode::DecodeGate) && args.bench_decode_tokens == 0 {
        bail!("--decode-tokens must be greater than zero in decode-gate mode");
    }
    Ok(())
}
