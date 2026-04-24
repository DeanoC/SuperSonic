use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use qwen35::config::{self, TextConfig};
use qwen35::rotary::RotaryTables;
use qwen35::state::ModelState;
use qwen35::weights::Qwen35Weights;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decode_engine::{decode_f32_le, DecodeEngine};
use crate::oracle;
use crate::prefill_engine;
use crate::registry::{self, FamilyParams, GpuArch, ModelVariant};
use crate::validate;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BughuntMode {
    Gate,
    Localize,
    Dump,
    Bench,
}

impl BughuntMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "gate",
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
    fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Full => "full",
            Self::Mlp => "mlp",
        }
    }

    fn from_model_layer(config: &TextConfig, layer: usize) -> Self {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptThresholds {
    pub prefill_logit_max_abs: f32,
    pub layer_hidden_max_abs: f32,
    pub restart_tail_logit_max_abs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptManifestEntry {
    pub name: String,
    pub prompt_ids: Vec<u32>,
    pub positions: Vec<usize>,
    pub thresholds: PromptThresholds,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptManifest {
    pub prompts: Vec<PromptManifestEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunMetadata {
    pub mode: String,
    pub model: String,
    pub backend: String,
    pub device: usize,
    pub arch: String,
    pub model_dir: String,
    pub oracle_device: String,
    pub commit_ish: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopDeltaDim {
    pub index: usize,
    pub native: f32,
    pub oracle: f32,
    pub delta: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageMetricReport {
    pub stage: String,
    pub native_field: String,
    pub oracle_field: String,
    pub len: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub mse: f32,
    pub max_index: usize,
    pub native_at_max: f32,
    pub oracle_at_max: f32,
    pub top_dims: Vec<TopDeltaDim>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TracedMetricsReport {
    pub layer: usize,
    pub layer_kind: String,
    pub position: usize,
    pub max_stage_delta: f32,
    pub stages: Vec<StageMetricReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDeltaReport {
    pub layer: usize,
    pub kind: String,
    pub max_abs_delta: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PositionSweepReport {
    pub position: usize,
    pub worst_layer: usize,
    pub worst_layer_kind: String,
    pub worst_layer_delta: f32,
    pub first_exceeding_layer: Option<usize>,
    pub layers: Vec<LayerDeltaReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseTimingReport {
    pub phase: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptGateReport {
    pub name: String,
    pub notes: Option<String>,
    pub pass: bool,
    pub thresholds: PromptThresholds,
    pub prefill_logit_reference: String,
    pub prefill_logit_max_abs: f32,
    pub prefill_logit_mean_abs: f32,
    pub prefill_logit_mse: f32,
    pub raw_oracle_prefill_logit_max_abs: f32,
    pub gpu_reference_logit_max_abs: f32,
    pub native_vs_gpu_reference_logit_max_abs: f32,
    pub worst_checked_position: usize,
    pub worst_layer: usize,
    pub worst_layer_kind: String,
    pub worst_layer_delta: f32,
    pub checked_positions: Vec<PositionSweepReport>,
    pub timings: Vec<PhaseTimingReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestartSweepReport {
    pub source_layer: usize,
    pub start_layer: usize,
    pub failing: bool,
    pub tail_logit_max_abs: f32,
    pub tail_logit_mean_abs: f32,
    pub selected_position: usize,
    pub selected_position_worst_layer: usize,
    pub selected_position_worst_layer_delta: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestartPositionScanReport {
    pub position: usize,
    pub worst_layer: usize,
    pub worst_layer_kind: String,
    pub worst_layer_delta: f32,
    pub final_hidden_logit_max_abs: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalizationSummary {
    pub prompt_name: String,
    pub initial_suspicious_position: usize,
    pub initial_suspicious_layer: usize,
    pub initial_suspicious_layer_kind: String,
    pub per_layer_hidden_sweep: Vec<PositionSweepReport>,
    pub restart_layer_sweep: Vec<RestartSweepReport>,
    pub first_suspicious_restart_layer: Option<usize>,
    pub restart_position_scan: Vec<RestartPositionScanReport>,
    pub worst_sampled_position: Option<usize>,
    pub chosen_traced_layer: Option<usize>,
    pub chosen_traced_layer_kind: Option<String>,
    pub traced_metrics: Option<TracedMetricsReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DumpSummary {
    pub prompt_name: String,
    pub position: usize,
    pub layer: usize,
    pub layer_kind: String,
    pub prompt_pass: bool,
    pub traced_metrics: TracedMetricsReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct GateRunSection {
    pub pass: bool,
    pub prompt_results: Vec<PromptGateReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalizeRunSection {
    pub pass: bool,
    pub gate_prompt: PromptGateReport,
    pub localization: LocalizationSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct DumpRunSection {
    pub pass: bool,
    pub gate_prompt: PromptGateReport,
    pub dump: DumpSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchPromptReport {
    pub name: String,
    pub notes: Option<String>,
    pub prompt_len: usize,
    pub warmup_iterations: usize,
    pub iterations: usize,
    pub decode_tokens: usize,
    pub native_prefill_ms: Vec<f64>,
    pub min_native_prefill_ms: f64,
    pub max_native_prefill_ms: f64,
    pub mean_native_prefill_ms: f64,
    pub greedy_prefill_ms: Vec<f64>,
    pub min_greedy_prefill_ms: f64,
    pub max_greedy_prefill_ms: f64,
    pub mean_greedy_prefill_ms: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub replay_decode_ms: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_replay_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_replay_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_replay_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_replay_decode_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_decode_ms: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_component_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_component_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_component_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_component_decode_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_profile: Option<MetalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub greedy_prefill_profile: Option<MetalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_decode_profile: Option<MetalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_decode_profile: Option<MetalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_hal_profile: Option<HalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub greedy_prefill_hal_profile: Option<HalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_decode_hal_profile: Option<HalProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_decode_hal_profile: Option<HalProfileReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetalProfileReport {
    pub total_calls: u64,
    pub native_calls: u64,
    pub host_calls: u64,
    pub total_ms: f64,
    pub native_ms: f64,
    pub host_ms: f64,
    pub entries: Vec<MetalProfileOpReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetalProfileOpReport {
    pub op: String,
    pub path: String,
    pub calls: u64,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HalProfileReport {
    pub total_calls: u64,
    pub total_ms: f64,
    pub alloc_calls: u64,
    pub alloc_bytes: u64,
    pub free_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub d2d_bytes: u64,
    pub memset_bytes: u64,
    pub sync_calls: u64,
    pub entries: Vec<HalProfileOpReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HalProfileOpReport {
    pub op: String,
    pub calls: u64,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchRunSection {
    pub pass: bool,
    pub prompt_results: Vec<BenchPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BughuntReport {
    pub mode: String,
    pub metadata: RunMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<GateRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub localize: Option<LocalizeRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dump: Option<DumpRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench: Option<BenchRunSection>,
}

impl BughuntReport {
    pub fn exit_code(&self) -> i32 {
        match self.mode.as_str() {
            "gate" => {
                if self
                    .gate
                    .as_ref()
                    .map(|section| section.pass)
                    .unwrap_or(false)
                {
                    0
                } else {
                    1
                }
            }
            "localize" => {
                if self
                    .localize
                    .as_ref()
                    .map(|section| section.pass)
                    .unwrap_or(false)
                {
                    0
                } else {
                    1
                }
            }
            "dump" => {
                if self
                    .dump
                    .as_ref()
                    .map(|section| section.pass)
                    .unwrap_or(false)
                {
                    0
                } else {
                    1
                }
            }
            "bench" => {
                if self
                    .bench
                    .as_ref()
                    .map(|section| section.pass)
                    .unwrap_or(false)
                {
                    0
                } else {
                    1
                }
            }
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
struct PromptGateAnalysis {
    report: PromptGateReport,
}

struct QwenBughuntRuntime {
    backend: Backend,
    ordinal: usize,
    arch_name: String,
    model_dir: PathBuf,
    oracle_device: String,
    model_variant: ModelVariant,
    weights: Qwen35Weights,
    rotary: RotaryTables,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
    weight_prefix: String,
    oracle_script: PathBuf,
    qwen35_trace_script: PathBuf,
    commit_ish: Option<String>,
}

pub fn run(args: BughuntArgs) -> Result<BughuntReport> {
    validate_args(&args)?;
    let manifest = load_prompt_manifest(&args.prompt_manifest)?;
    let runtime = QwenBughuntRuntime::new(
        &args.model_dir,
        args.backend,
        args.ordinal,
        &args.oracle_device,
    )?;
    let metadata = runtime.metadata(args.mode);
    let report = match args.mode {
        BughuntMode::Gate => {
            let reports = run_gate_mode(&runtime, &manifest, args.prompt.as_deref())?;
            BughuntReport {
                mode: args.mode.as_str().to_string(),
                metadata,
                gate: Some(reports),
                localize: None,
                dump: None,
                bench: None,
            }
        }
        BughuntMode::Localize => {
            let section = run_localize_mode(&runtime, &manifest, args.prompt.as_deref())?;
            BughuntReport {
                mode: args.mode.as_str().to_string(),
                metadata,
                gate: None,
                localize: Some(section),
                dump: None,
                bench: None,
            }
        }
        BughuntMode::Dump => {
            let section = run_dump_mode(
                &runtime,
                &manifest,
                args.prompt.as_deref(),
                args.position,
                args.layer,
                args.layer_kind,
            )?;
            BughuntReport {
                mode: args.mode.as_str().to_string(),
                metadata,
                gate: None,
                localize: None,
                dump: Some(section),
                bench: None,
            }
        }
        BughuntMode::Bench => {
            let section = run_bench_mode(
                &runtime,
                &manifest,
                args.prompt.as_deref(),
                args.bench_iterations,
                args.bench_warmup,
                args.bench_decode_tokens,
                args.bench_profile_ops,
            )?;
            BughuntReport {
                mode: args.mode.as_str().to_string(),
                metadata,
                gate: None,
                localize: None,
                dump: None,
                bench: Some(section),
            }
        }
    };

    print_report_summary(&report);
    if let Some(path) = args.report_json.as_ref() {
        write_report_json(path, &report)?;
        println!("report_json={}", path.display());
    }
    Ok(report)
}

fn validate_args(args: &BughuntArgs) -> Result<()> {
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
    Ok(())
}

impl QwenBughuntRuntime {
    fn new(
        model_dir: &Path,
        backend_choice: BackendArg,
        ordinal: usize,
        oracle_device_spec: &str,
    ) -> Result<Self> {
        let backend = resolve_backend(backend_choice, ordinal)?;
        gpu_hal::set_backend(backend);

        let (arch_name, _, _) = query_backend_device(backend, ordinal)?;
        let model_variant = ModelVariant::Qwen3_5_0_8B;
        let gpu_arch = GpuArch::from_backend_name(&backend, &arch_name);
        let entry = registry::lookup(&model_variant, &backend, &gpu_arch).ok_or_else(|| {
            let supported = registry::supported_archs_for(&model_variant, &backend);
            anyhow::anyhow!(
                "No registry entry for model={} backend={} arch={}. Supported archs: [{}]",
                model_variant,
                backend,
                gpu_arch,
                supported.join(", ")
            )
        })?;
        let params = match entry.params {
            FamilyParams::Qwen35(params) => params,
            _ => bail!("bughunt harness only supports Qwen3.5"),
        };

        let loaded = config::load_config(model_dir)
            .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
        let text_config = loaded.text_config;
        let weights = load_qwen35_weights(model_dir, &text_config, ordinal, params.weight_prefix)?;
        let rotary = RotaryTables::build(&text_config, ordinal)
            .map_err(|e| anyhow::anyhow!("rotary: {e}"))?;

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .context("runner crate missing repo root")?
            .to_path_buf();

        Ok(Self {
            backend,
            ordinal,
            arch_name,
            model_dir: model_dir.to_path_buf(),
            oracle_device: resolve_oracle_device(oracle_device_spec, backend, ordinal),
            model_variant,
            weights,
            rotary,
            kv_chunk_size: params.kv_chunk_size,
            prefill_chunk_size: 0,
            use_4b_kernel: params.use_4b_kernel,
            proj_buf_floats: params.proj_buf_floats,
            attn_scratch_floats: params.attn_scratch_floats,
            weight_prefix: params.weight_prefix.to_string(),
            oracle_script: repo_root.join("oracle/run_oracle.py"),
            qwen35_trace_script: repo_root.join("oracle/qwen35_oracle.py"),
            commit_ish: git_commit_ish(&repo_root),
        })
    }

    fn metadata(&self, mode: BughuntMode) -> RunMetadata {
        RunMetadata {
            mode: mode.as_str().to_string(),
            model: self.model_variant.to_string(),
            backend: self.backend.to_string(),
            device: self.ordinal,
            arch: self.arch_name.clone(),
            model_dir: self.model_dir.display().to_string(),
            oracle_device: self.oracle_device.clone(),
            commit_ish: self.commit_ish.clone(),
        }
    }

    fn new_component_decode_engine(&self, context_tokens: usize) -> Result<DecodeEngine> {
        let attn_scratch_floats = qwen35::scratch::required_attn_scratch_floats(
            self.weights.config.num_attention_heads,
            self.weights.config.head_dim,
            context_tokens,
            self.kv_chunk_size,
        )
        .max(self.attn_scratch_floats);
        let weights = load_qwen35_weights(
            &self.model_dir,
            &self.weights.config,
            self.ordinal,
            &self.weight_prefix,
        )?;
        DecodeEngine::new(
            weights,
            self.ordinal,
            self.proj_buf_floats,
            attn_scratch_floats,
            self.kv_chunk_size,
            self.use_4b_kernel,
            self.prefill_chunk_size,
            false,
            1,
        )
    }
}

fn resolve_backend(choice: BackendArg, ordinal: usize) -> Result<Backend> {
    match choice {
        BackendArg::Cuda => require_backend(Backend::Cuda),
        BackendArg::Hip => require_backend(Backend::Hip),
        BackendArg::Metal => require_backend(Backend::Metal),
        BackendArg::Auto => {
            if gpu_hal::is_backend_compiled(Backend::Cuda)
                && gpu_hal::query_device_info(Backend::Cuda, ordinal).is_ok()
            {
                return Ok(Backend::Cuda);
            }
            if gpu_hal::is_backend_compiled(Backend::Hip)
                && kernel_ffi::query_gpu_info(ordinal).is_ok()
            {
                return Ok(Backend::Hip);
            }
            if gpu_hal::is_backend_compiled(Backend::Metal)
                && gpu_hal::query_device_info(Backend::Metal, ordinal).is_ok()
            {
                return Ok(Backend::Metal);
            }
            bail!(
                "No usable GPU backend available for device {}. Compiled backends: [{}]",
                ordinal,
                gpu_hal::compiled_backends()
                    .into_iter()
                    .map(|backend| backend.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}

fn require_backend(backend: Backend) -> Result<Backend> {
    if !gpu_hal::is_backend_compiled(backend) {
        bail!(
            "Requested backend {} is not compiled into this build. Compiled backends: [{}]",
            backend,
            gpu_hal::compiled_backends()
                .into_iter()
                .map(|candidate| candidate.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(backend)
}

fn resolve_oracle_device(spec: &str, backend: Backend, ordinal: usize) -> String {
    match spec.trim().to_ascii_lowercase().as_str() {
        "auto" => match backend {
            Backend::Cuda => format!("cuda:{ordinal}"),
            Backend::Hip => "cpu".to_string(),
            Backend::Metal => "cpu".to_string(),
        },
        other => other.to_string(),
    }
}

fn query_backend_device(backend: Backend, ordinal: usize) -> Result<(String, u64, u32)> {
    match backend {
        Backend::Hip => {
            let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            Ok((arch_name, total_vram, 32))
        }
        Backend::Cuda | Backend::Metal => {
            let info = gpu_hal::query_device_info(backend, ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            Ok((info.arch_name, info.total_vram_bytes, info.warp_size))
        }
    }
}

fn git_commit_ish(repo_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn load_qwen35_weights(
    model_dir: &Path,
    text_config: &TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    let bake_dir = model_store::fetch::BakeVariant::Bf16.bake_dir(model_dir);
    if model_store::version_ok(&bake_dir) {
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow::anyhow!("open baked store: {e}"))?;
        Qwen35Weights::load_baked(&store, text_config, ordinal, weight_prefix)
            .map_err(|e| anyhow::anyhow!("load baked weights: {e}"))
    } else {
        Qwen35Weights::load(model_dir, text_config, ordinal, weight_prefix)
            .map_err(|e| anyhow::anyhow!("load weights: {e}"))
    }
}

fn load_prompt_manifest(path: &Path) -> Result<PromptManifest> {
    let manifest_text = fs::read_to_string(path)
        .with_context(|| format!("read prompt manifest {}", path.display()))?;
    parse_prompt_manifest_str(&manifest_text)
}

fn parse_prompt_manifest_str(manifest_text: &str) -> Result<PromptManifest> {
    let manifest: PromptManifest =
        serde_json::from_str(manifest_text).context("parse prompt manifest JSON")?;
    validate_prompt_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_prompt_manifest(manifest: &PromptManifest) -> Result<()> {
    if manifest.prompts.is_empty() {
        bail!("prompt manifest must contain at least one prompt");
    }
    let mut names = HashSet::new();
    for prompt in &manifest.prompts {
        if prompt.name.trim().is_empty() {
            bail!("prompt manifest contains an entry with an empty name");
        }
        if !names.insert(prompt.name.clone()) {
            bail!(
                "prompt manifest contains duplicate prompt name '{}'",
                prompt.name
            );
        }
        if prompt.prompt_ids.is_empty() {
            bail!(
                "prompt '{}' must contain at least one prompt token id",
                prompt.name
            );
        }
        if prompt.positions.is_empty() {
            bail!(
                "prompt '{}' must contain at least one checked position",
                prompt.name
            );
        }
        let mut seen_positions = HashSet::new();
        for &position in &prompt.positions {
            if position >= prompt.prompt_ids.len() {
                bail!(
                    "prompt '{}' position {} is out of range for {} prompt tokens",
                    prompt.name,
                    position,
                    prompt.prompt_ids.len()
                );
            }
            if !seen_positions.insert(position) {
                bail!(
                    "prompt '{}' contains duplicate checked position {}",
                    prompt.name,
                    position
                );
            }
        }
        validate_positive_threshold(
            "prefill_logit_max_abs",
            prompt.thresholds.prefill_logit_max_abs,
            &prompt.name,
        )?;
        validate_positive_threshold(
            "layer_hidden_max_abs",
            prompt.thresholds.layer_hidden_max_abs,
            &prompt.name,
        )?;
        validate_positive_threshold(
            "restart_tail_logit_max_abs",
            prompt.thresholds.restart_tail_logit_max_abs,
            &prompt.name,
        )?;
    }
    Ok(())
}

fn validate_positive_threshold(label: &str, value: f32, prompt_name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!(
            "prompt '{}' threshold '{}' must be a finite positive number",
            prompt_name,
            label
        );
    }
    Ok(())
}

fn run_gate_mode(
    runtime: &QwenBughuntRuntime,
    manifest: &PromptManifest,
    selected_prompt: Option<&str>,
) -> Result<GateRunSection> {
    let prompts = select_prompts(manifest, selected_prompt)?;
    let mut prompt_results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        eprintln!("[bughunt] gate prompt={} start", prompt.name);
        let analysis = analyze_gate_prompt(runtime, prompt)?;
        eprintln!(
            "[bughunt] gate prompt={} done pass={} prefill_max_abs={:.4} worst_layer_delta={:.4}",
            prompt.name,
            analysis.report.pass,
            analysis.report.prefill_logit_max_abs,
            analysis.report.worst_layer_delta,
        );
        prompt_results.push(analysis.report);
    }
    let pass = prompt_results.iter().all(|prompt| prompt.pass);
    Ok(GateRunSection {
        pass,
        prompt_results,
    })
}

fn run_bench_mode(
    runtime: &QwenBughuntRuntime,
    manifest: &PromptManifest,
    selected_prompt: Option<&str>,
    iterations: usize,
    warmup_iterations: usize,
    decode_tokens: usize,
    profile_ops: bool,
) -> Result<BenchRunSection> {
    let prompts = select_prompts(manifest, selected_prompt)?;
    let mut prompt_results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        eprintln!(
            "[bughunt] bench prompt={} warmup={} iters={} decode_tokens={} profile_ops={} start",
            prompt.name, warmup_iterations, iterations, decode_tokens, profile_ops
        );
        let mut greedy_prefill_state = ModelState::new(&runtime.weights.config, runtime.ordinal)
            .map_err(|e| anyhow::anyhow!("bench greedy prefill state init: {e}"))?;
        for warmup in 0..warmup_iterations {
            let _ = run_native_prefill(runtime, &prompt.prompt_ids)
                .with_context(|| format!("bench warmup {warmup} prompt {}", prompt.name))?;
            gpu_hal::sync(runtime.ordinal)
                .with_context(|| format!("bench warmup sync prompt {}", prompt.name))?;
            let _ = run_native_prefill_greedy_token_with_state(
                runtime,
                &mut greedy_prefill_state,
                &prompt.prompt_ids,
            )
            .with_context(|| format!("bench greedy warmup {warmup} prompt {}", prompt.name))?;
            gpu_hal::sync(runtime.ordinal)
                .with_context(|| format!("bench greedy warmup sync prompt {}", prompt.name))?;
            if decode_tokens > 0 {
                let _ = run_replay_decode_once(runtime, &prompt.prompt_ids, decode_tokens)
                    .with_context(|| {
                        format!("bench replay decode warmup {warmup} prompt {}", prompt.name)
                    })?;
            }
        }

        let mut native_prefill_ms = Vec::with_capacity(iterations);
        for iter in 0..iterations {
            let start = Instant::now();
            let _ = run_native_prefill(runtime, &prompt.prompt_ids)
                .with_context(|| format!("bench iter {iter} prompt {}", prompt.name))?;
            gpu_hal::sync(runtime.ordinal)
                .with_context(|| format!("bench iter sync prompt {}", prompt.name))?;
            native_prefill_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let (min_native_prefill_ms, max_native_prefill_ms, mean_native_prefill_ms) =
            bench_stats(&native_prefill_ms);
        let mut greedy_prefill_ms = Vec::with_capacity(iterations);
        for iter in 0..iterations {
            let start = Instant::now();
            let _ = run_native_prefill_greedy_token_with_state(
                runtime,
                &mut greedy_prefill_state,
                &prompt.prompt_ids,
            )
            .with_context(|| format!("bench greedy iter {iter} prompt {}", prompt.name))?;
            gpu_hal::sync(runtime.ordinal)
                .with_context(|| format!("bench greedy iter sync prompt {}", prompt.name))?;
            greedy_prefill_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let (min_greedy_prefill_ms, max_greedy_prefill_ms, mean_greedy_prefill_ms) =
            bench_stats(&greedy_prefill_ms);
        let prefill_profiles = if profile_ops {
            collect_profiles(|| {
                let _ = run_native_prefill(runtime, &prompt.prompt_ids)
                    .with_context(|| format!("bench profile prefill prompt {}", prompt.name))?;
                gpu_hal::sync(runtime.ordinal)
                    .with_context(|| format!("bench profile prefill sync prompt {}", prompt.name))
            })?
        } else {
            ProfileReports::default()
        };
        let greedy_prefill_profiles = if profile_ops {
            collect_profiles(|| {
                let _ = run_native_prefill_greedy_token_with_state(
                    runtime,
                    &mut greedy_prefill_state,
                    &prompt.prompt_ids,
                )
                .with_context(|| format!("bench profile greedy prefill prompt {}", prompt.name))?;
                gpu_hal::sync(runtime.ordinal).with_context(|| {
                    format!("bench profile greedy prefill sync prompt {}", prompt.name)
                })
            })?
        } else {
            ProfileReports::default()
        };

        let mut component_engine = if decode_tokens > 0 && runtime.backend == Backend::Metal {
            Some(
                runtime
                    .new_component_decode_engine(prompt.prompt_ids.len() + decode_tokens)
                    .with_context(|| {
                        format!("bench component decode engine prompt {}", prompt.name)
                    })?,
            )
        } else {
            None
        };
        let mut replay_decode_ms = Vec::with_capacity(iterations);
        if decode_tokens > 0 {
            for iter in 0..iterations {
                let elapsed_ms = run_replay_decode_once(runtime, &prompt.prompt_ids, decode_tokens)
                    .with_context(|| {
                        format!("bench replay decode iter {iter} prompt {}", prompt.name)
                    })?;
                replay_decode_ms.push(elapsed_ms);
            }
        }
        let (min_replay_decode_ms, max_replay_decode_ms, mean_replay_decode_ms) =
            optional_bench_stats(&replay_decode_ms);
        let mean_replay_decode_ms_per_token = mean_replay_decode_ms
            .filter(|_| decode_tokens > 0)
            .map(|value| value / decode_tokens as f64);
        let replay_decode_profiles = if profile_ops && decode_tokens > 0 {
            collect_replay_decode_profile(runtime, &prompt.prompt_ids, decode_tokens)?
        } else {
            ProfileReports::default()
        };
        let mut component_decode_ms = Vec::with_capacity(iterations);
        if let Some(engine) = component_engine.as_mut() {
            for warmup in 0..warmup_iterations {
                let _ = run_component_decode_once(engine, &prompt.prompt_ids, decode_tokens)
                    .with_context(|| {
                        format!(
                            "bench component decode warmup {warmup} prompt {}",
                            prompt.name
                        )
                    })?;
            }
            for iter in 0..iterations {
                let elapsed_ms =
                    run_component_decode_once(engine, &prompt.prompt_ids, decode_tokens)
                        .with_context(|| {
                            format!("bench component decode iter {iter} prompt {}", prompt.name)
                        })?;
                component_decode_ms.push(elapsed_ms);
            }
        }
        let (min_component_decode_ms, max_component_decode_ms, mean_component_decode_ms) =
            optional_bench_stats(&component_decode_ms);
        let mean_component_decode_ms_per_token = mean_component_decode_ms
            .filter(|_| decode_tokens > 0)
            .map(|value| value / decode_tokens as f64);
        let component_decode_profiles = if profile_ops && decode_tokens > 0 {
            if let Some(engine) = component_engine.as_mut() {
                collect_component_decode_profile(
                    runtime.ordinal,
                    engine,
                    &prompt.prompt_ids,
                    decode_tokens,
                )?
            } else {
                ProfileReports::default()
            }
        } else {
            ProfileReports::default()
        };
        eprintln!(
            "[bughunt] bench prompt={} done mean_native_prefill_ms={:.1} min={:.1} max={:.1} mean_greedy_prefill_ms={:.1} mean_replay_decode_ms_per_token={} mean_component_decode_ms_per_token={}",
            prompt.name,
            mean_native_prefill_ms,
            min_native_prefill_ms,
            max_native_prefill_ms,
            mean_greedy_prefill_ms,
            mean_replay_decode_ms_per_token
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string()),
            mean_component_decode_ms_per_token
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string())
        );
        prompt_results.push(BenchPromptReport {
            name: prompt.name.clone(),
            notes: prompt.notes.clone(),
            prompt_len: prompt.prompt_ids.len(),
            warmup_iterations,
            iterations,
            decode_tokens,
            native_prefill_ms,
            min_native_prefill_ms,
            max_native_prefill_ms,
            mean_native_prefill_ms,
            greedy_prefill_ms,
            min_greedy_prefill_ms,
            max_greedy_prefill_ms,
            mean_greedy_prefill_ms,
            replay_decode_ms,
            min_replay_decode_ms,
            max_replay_decode_ms,
            mean_replay_decode_ms,
            mean_replay_decode_ms_per_token,
            component_decode_ms: (!component_decode_ms.is_empty()).then_some(component_decode_ms),
            min_component_decode_ms,
            max_component_decode_ms,
            mean_component_decode_ms,
            mean_component_decode_ms_per_token,
            prefill_profile: prefill_profiles.metal,
            greedy_prefill_profile: greedy_prefill_profiles.metal,
            replay_decode_profile: replay_decode_profiles.metal,
            component_decode_profile: component_decode_profiles.metal,
            prefill_hal_profile: prefill_profiles.hal,
            greedy_prefill_hal_profile: greedy_prefill_profiles.hal,
            replay_decode_hal_profile: replay_decode_profiles.hal,
            component_decode_hal_profile: component_decode_profiles.hal,
        });
    }
    Ok(BenchRunSection {
        pass: true,
        prompt_results,
    })
}

fn bench_stats(values: &[f64]) -> (f64, f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (min, max, mean)
}

fn optional_bench_stats(values: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None, None);
    }
    let (min, max, mean) = bench_stats(values);
    (Some(min), Some(max), Some(mean))
}

#[derive(Debug, Default)]
struct ProfileReports {
    metal: Option<MetalProfileReport>,
    hal: Option<HalProfileReport>,
}

fn collect_profiles<F>(f: F) -> Result<ProfileReports>
where
    F: FnOnce() -> Result<()>,
{
    let _guard = ProfileGuard::new();
    kernel_ffi::prefill_ffi::metal_profile_reset();
    gpu_hal::hal_profile_reset();
    f()?;
    Ok(ProfileReports {
        metal: Some(metal_profile_report(
            kernel_ffi::prefill_ffi::metal_profile_snapshot(),
        )),
        hal: Some(hal_profile_report(gpu_hal::hal_profile_snapshot())),
    })
}

fn collect_replay_decode_profile(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<ProfileReports> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("bench profile replay state init: {e}"))?;
    let mut history = prompt_ids.to_vec();
    let mut token = run_native_prefill_greedy_token_with_state(runtime, &mut state, &history)
        .context("bench profile replay decode initial prefill")?;
    gpu_hal::sync(runtime.ordinal).context("bench profile replay decode initial sync")?;

    let _guard = ProfileGuard::new();
    kernel_ffi::prefill_ffi::metal_profile_reset();
    gpu_hal::hal_profile_reset();
    for step in 0..decode_tokens {
        history.push(token);
        token = run_native_prefill_greedy_token_with_state(runtime, &mut state, &history)
            .with_context(|| format!("bench profile replay decode step {step}"))?;
        gpu_hal::sync(runtime.ordinal)
            .with_context(|| format!("bench profile replay decode sync step {step}"))?;
    }
    Ok(ProfileReports {
        metal: Some(metal_profile_report(
            kernel_ffi::prefill_ffi::metal_profile_snapshot(),
        )),
        hal: Some(hal_profile_report(gpu_hal::hal_profile_snapshot())),
    })
}

fn collect_component_decode_profile(
    ordinal: usize,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<ProfileReports> {
    let token = engine
        .rebuild_prefill_state_greedy_token(prompt_ids)
        .context("bench profile component decode initial prefill")?;
    gpu_hal::sync(ordinal).context("bench profile component decode initial sync")?;

    let _guard = ProfileGuard::new();
    kernel_ffi::prefill_ffi::metal_profile_reset();
    gpu_hal::hal_profile_reset();
    run_component_decode_steps(engine, token, prompt_ids.len(), decode_tokens)
        .context("bench profile component decode steps")?;
    Ok(ProfileReports {
        metal: Some(metal_profile_report(
            kernel_ffi::prefill_ffi::metal_profile_snapshot(),
        )),
        hal: Some(hal_profile_report(gpu_hal::hal_profile_snapshot())),
    })
}

fn metal_profile_report(
    snapshot: kernel_ffi::prefill_ffi::MetalProfileSnapshot,
) -> MetalProfileReport {
    MetalProfileReport {
        total_calls: snapshot.total_calls,
        native_calls: snapshot.native_calls,
        host_calls: snapshot.host_calls,
        total_ms: snapshot.total_ms,
        native_ms: snapshot.native_ms,
        host_ms: snapshot.host_ms,
        entries: snapshot
            .entries
            .into_iter()
            .map(|entry| MetalProfileOpReport {
                mean_ms: entry.mean_ms(),
                op: entry.op,
                path: entry.path,
                calls: entry.calls,
                total_ms: entry.total_ms,
                max_ms: entry.max_ms,
            })
            .collect(),
    }
}

fn hal_profile_report(snapshot: gpu_hal::HalProfileSnapshot) -> HalProfileReport {
    HalProfileReport {
        total_calls: snapshot.total_calls,
        total_ms: snapshot.total_ms,
        alloc_calls: snapshot.alloc_calls,
        alloc_bytes: snapshot.alloc_bytes,
        free_calls: snapshot.free_calls,
        h2d_bytes: snapshot.h2d_bytes,
        d2h_bytes: snapshot.d2h_bytes,
        d2d_bytes: snapshot.d2d_bytes,
        memset_bytes: snapshot.memset_bytes,
        sync_calls: snapshot.sync_calls,
        entries: snapshot
            .entries
            .into_iter()
            .map(|entry| HalProfileOpReport {
                mean_ms: entry.mean_ms(),
                op: entry.op,
                calls: entry.calls,
                total_ms: entry.total_ms,
                max_ms: entry.max_ms,
                total_bytes: entry.total_bytes,
            })
            .collect(),
    }
}

struct ProfileGuard;

impl ProfileGuard {
    fn new() -> Self {
        kernel_ffi::prefill_ffi::metal_profile_set_enabled(true);
        gpu_hal::hal_profile_set_enabled(true);
        Self
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        kernel_ffi::prefill_ffi::metal_profile_set_enabled(false);
        gpu_hal::hal_profile_set_enabled(false);
    }
}

fn run_replay_decode_once(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<f64> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("bench replay state init: {e}"))?;
    let mut history = prompt_ids.to_vec();
    let mut token = run_native_prefill_greedy_token_with_state(runtime, &mut state, &history)
        .context("bench replay decode initial prefill")?;
    gpu_hal::sync(runtime.ordinal).context("bench replay decode initial sync")?;

    let start = Instant::now();
    for step in 0..decode_tokens {
        history.push(token);
        token = run_native_prefill_greedy_token_with_state(runtime, &mut state, &history)
            .with_context(|| format!("bench replay decode step {step}"))?;
        gpu_hal::sync(runtime.ordinal)
            .with_context(|| format!("bench replay decode sync step {step}"))?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn run_component_decode_once(
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<f64> {
    let token = engine
        .rebuild_prefill_state_greedy_token(prompt_ids)
        .context("bench component decode initial prefill")?;
    run_component_decode_steps(engine, token, prompt_ids.len(), decode_tokens)
}

fn run_component_decode_steps(
    engine: &mut DecodeEngine,
    mut token: u32,
    initial_seqlen: usize,
    decode_tokens: usize,
) -> Result<f64> {
    let start = Instant::now();
    for step in 0..decode_tokens {
        let (next, _) = engine
            .decode_step_metal_component_greedy(token, initial_seqlen + step)
            .with_context(|| format!("bench component decode step {step}"))?;
        token = next;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn run_localize_mode(
    runtime: &QwenBughuntRuntime,
    manifest: &PromptManifest,
    selected_prompt: Option<&str>,
) -> Result<LocalizeRunSection> {
    let prompts = select_prompts(manifest, selected_prompt)?;
    let mut first_failure: Option<(&PromptManifestEntry, PromptGateAnalysis)> = None;
    let mut last_success: Option<PromptGateAnalysis> = None;
    for prompt in prompts {
        let analysis = analyze_gate_prompt(runtime, prompt)?;
        if !analysis.report.pass {
            first_failure = Some((prompt, analysis));
            break;
        }
        last_success = Some(analysis);
    }

    let (prompt, gate_analysis) = match first_failure {
        Some(found) => found,
        None => {
            let gate_prompt = last_success
                .map(|analysis| analysis.report)
                .ok_or_else(|| anyhow::anyhow!("no prompts available for localization"))?;
            return Ok(LocalizeRunSection {
                pass: true,
                gate_prompt,
                localization: LocalizationSummary {
                    prompt_name: manifest.prompts[0].name.clone(),
                    initial_suspicious_position: 0,
                    initial_suspicious_layer: 0,
                    initial_suspicious_layer_kind: "linear".to_string(),
                    per_layer_hidden_sweep: Vec::new(),
                    restart_layer_sweep: Vec::new(),
                    first_suspicious_restart_layer: None,
                    restart_position_scan: Vec::new(),
                    worst_sampled_position: None,
                    chosen_traced_layer: None,
                    chosen_traced_layer_kind: None,
                    traced_metrics: None,
                },
            });
        }
    };

    eprintln!(
        "[bughunt] localize prompt={} gate_fail prefill_max_abs={:.4} worst_position={} worst_layer={}({})",
        prompt.name,
        gate_analysis.report.prefill_logit_max_abs,
        gate_analysis.report.worst_checked_position,
        gate_analysis.report.worst_layer,
        gate_analysis.report.worst_layer_kind,
    );
    let per_layer_hidden_sweep = gate_analysis.report.checked_positions.clone();
    let suspicious_position = choose_worst_position(&per_layer_hidden_sweep)
        .map(|position| position.position)
        .unwrap_or(0);
    eprintln!(
        "[bughunt] localize prompt={} restart_trace position={}",
        prompt.name, suspicious_position
    );
    let trace = run_trace_oracle(
        runtime,
        &prompt.prompt_ids,
        Some(suspicious_position),
        None,
        None,
    )?;
    let suspicious_layer = per_layer_hidden_sweep
        .iter()
        .find(|position| position.position == suspicious_position)
        .and_then(|position| {
            position.first_exceeding_layer.or_else(|| {
                position
                    .layers
                    .iter()
                    .max_by(|lhs, rhs| lhs.max_abs_delta.total_cmp(&rhs.max_abs_delta))
                    .map(|layer| layer.layer)
            })
        })
        .unwrap_or(0);
    let suspicious_kind =
        BughuntLayerKind::from_model_layer(&runtime.weights.config, suspicious_layer);

    eprintln!(
        "[bughunt] localize prompt={} restart_sweep position={} layer={}({})",
        prompt.name,
        suspicious_position,
        suspicious_layer,
        suspicious_kind.as_str(),
    );
    let restart_layer_sweep =
        run_restart_layer_sweep(runtime, prompt, &trace, suspicious_position)?;
    let first_suspicious_restart_layer = choose_deepest_failing_restart_layer(&restart_layer_sweep);
    eprintln!(
        "[bughunt] localize prompt={} restart_sweep_done first_restart_layer={}",
        prompt.name,
        first_suspicious_restart_layer
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
    );
    let restart_position_scan = if let Some(start_layer) = first_suspicious_restart_layer {
        eprintln!(
            "[bughunt] localize prompt={} restart_position_scan start_layer={}",
            prompt.name, start_layer
        );
        run_restart_position_scan(runtime, prompt, &trace, start_layer)?
    } else {
        Vec::new()
    };
    let worst_sampled_position =
        choose_worst_restart_position(&restart_position_scan).map(|report| report.position);

    let chosen_traced_layer = first_suspicious_restart_layer.or(Some(suspicious_layer));
    let chosen_traced_layer_kind = chosen_traced_layer
        .map(|layer| BughuntLayerKind::from_model_layer(&runtime.weights.config, layer));
    let traced_metrics = match (
        chosen_traced_layer,
        chosen_traced_layer_kind,
        worst_sampled_position,
    ) {
        (Some(layer), Some(kind), Some(position)) => {
            eprintln!(
                "[bughunt] localize prompt={} dump_trace layer={}({}) position={}",
                prompt.name,
                layer,
                kind.as_str(),
                position,
            );
            Some(build_traced_metrics(
                runtime,
                &prompt.prompt_ids,
                position,
                layer,
                kind,
            )?)
        }
        _ => None,
    };

    Ok(LocalizeRunSection {
        pass: false,
        gate_prompt: gate_analysis.report,
        localization: LocalizationSummary {
            prompt_name: prompt.name.clone(),
            initial_suspicious_position: suspicious_position,
            initial_suspicious_layer: suspicious_layer,
            initial_suspicious_layer_kind: suspicious_kind.as_str().to_string(),
            per_layer_hidden_sweep,
            restart_layer_sweep,
            first_suspicious_restart_layer,
            restart_position_scan,
            worst_sampled_position,
            chosen_traced_layer,
            chosen_traced_layer_kind: chosen_traced_layer_kind
                .map(|kind| kind.as_str().to_string()),
            traced_metrics,
        },
    })
}

fn run_dump_mode(
    runtime: &QwenBughuntRuntime,
    manifest: &PromptManifest,
    prompt_name: Option<&str>,
    requested_position: Option<usize>,
    requested_layer: Option<usize>,
    requested_layer_kind: Option<BughuntLayerKind>,
) -> Result<DumpRunSection> {
    let prompt_name = prompt_name.context("dump mode requires --prompt")?;
    let prompt = manifest
        .prompts
        .iter()
        .find(|entry| entry.name == prompt_name)
        .ok_or_else(|| anyhow::anyhow!("prompt '{}' not found in manifest", prompt_name))?;
    let gate_analysis = analyze_gate_prompt(runtime, prompt)?;
    eprintln!(
        "[bughunt] dump prompt={} gate_pass={} worst_position={} worst_layer={}({})",
        prompt.name,
        gate_analysis.report.pass,
        gate_analysis.report.worst_checked_position,
        gate_analysis.report.worst_layer,
        gate_analysis.report.worst_layer_kind,
    );
    let trace = run_trace_oracle(runtime, &prompt.prompt_ids, requested_position, None, None)?;
    let position = requested_position.unwrap_or(gate_analysis.report.worst_checked_position);
    if position >= prompt.prompt_ids.len() {
        bail!(
            "dump position {} is out of range for prompt '{}' with {} tokens",
            position,
            prompt.name,
            prompt.prompt_ids.len()
        );
    }
    let sweep = analyze_position_against_trace(runtime, prompt, position, &trace)?;
    let default_layer = sweep
        .first_exceeding_layer
        .or_else(|| {
            sweep
                .layers
                .iter()
                .max_by(|lhs, rhs| lhs.max_abs_delta.total_cmp(&rhs.max_abs_delta))
                .map(|layer| layer.layer)
        })
        .unwrap_or(0);
    let layer = requested_layer.unwrap_or(default_layer);
    if layer >= runtime.weights.config.num_hidden_layers {
        bail!(
            "dump layer {} is out of range for {} layers",
            layer,
            runtime.weights.config.num_hidden_layers
        );
    }
    let layer_kind = requested_layer_kind
        .unwrap_or_else(|| BughuntLayerKind::from_model_layer(&runtime.weights.config, layer));
    eprintln!(
        "[bughunt] dump prompt={} position={} layer={}({})",
        prompt.name,
        position,
        layer,
        layer_kind.as_str(),
    );
    let traced_metrics =
        build_traced_metrics(runtime, &prompt.prompt_ids, position, layer, layer_kind)?;
    let prompt_pass = gate_analysis.report.pass;

    Ok(DumpRunSection {
        pass: gate_analysis.report.pass,
        gate_prompt: gate_analysis.report,
        dump: DumpSummary {
            prompt_name: prompt.name.clone(),
            position,
            layer,
            layer_kind: layer_kind.as_str().to_string(),
            prompt_pass,
            traced_metrics,
        },
    })
}

fn select_prompts<'a>(
    manifest: &'a PromptManifest,
    selected_prompt: Option<&str>,
) -> Result<Vec<&'a PromptManifestEntry>> {
    if let Some(name) = selected_prompt {
        let prompt = manifest
            .prompts
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| anyhow::anyhow!("prompt '{}' not found in manifest", name))?;
        Ok(vec![prompt])
    } else {
        Ok(manifest.prompts.iter().collect())
    }
}

fn analyze_gate_prompt(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
) -> Result<PromptGateAnalysis> {
    let prompt_start = Instant::now();
    let mut timings = Vec::new();

    eprintln!("[bughunt] gate prompt={} oracle_compact", prompt.name);
    let phase_start = Instant::now();
    let oracle_output = oracle::run_oracle(
        &runtime.oracle_script,
        runtime.model_variant.hf_model_id(),
        &prompt.prompt_ids,
        0,
        "bf16",
        &runtime.oracle_device,
        false,
        false,
        None,
        None,
    )?;
    timings.push(phase_timing("oracle_compact", phase_start));

    eprintln!("[bughunt] gate prompt={} oracle_trace", prompt.name);
    let phase_start = Instant::now();
    let trace = run_trace_oracle(runtime, &prompt.prompt_ids, None, None, None)?;
    timings.push(phase_timing("oracle_trace", phase_start));

    eprintln!("[bughunt] gate prompt={} native_prefill", prompt.name);
    let phase_start = Instant::now();
    let native_prefill = run_native_prefill(runtime, &prompt.prompt_ids)?;
    timings.push(phase_timing("native_prefill", phase_start));

    eprintln!("[bughunt] gate prompt={} gpu_reference", prompt.name);
    let phase_start = Instant::now();
    let gpu_reference_logits = prefill_engine::gpu_reference_replay_step(
        &runtime.weights,
        &runtime.rotary,
        &prompt.prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        runtime.use_4b_kernel,
    )?;
    timings.push(phase_timing("gpu_reference", phase_start));

    eprintln!(
        "[bughunt] gate prompt={} position_sweep count={}",
        prompt.name,
        prompt.positions.len()
    );
    let phase_start = Instant::now();
    let checked_positions = sweep_prompt_positions(runtime, prompt, &trace)?;
    timings.push(phase_timing("position_sweep", phase_start));
    let worst_position = choose_worst_position(&checked_positions)
        .ok_or_else(|| anyhow::anyhow!("prompt '{}' produced no checked positions", prompt.name))?;

    let oracle_final_hidden = trace
        .decoder_layer_outputs
        .last()
        .and_then(|value| flatten_token_bsd(value, None))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "prompt '{}' oracle trace missing final hidden for last prompt position",
                prompt.name
            )
        })?;
    let oracle_aligned_prefill_logits =
        compute_qwen_logits_from_hidden_row(runtime, &oracle_final_hidden)?;

    let prefill_logit_max_abs =
        validate::max_abs_delta(&native_prefill.logits, &oracle_aligned_prefill_logits);
    let prefill_logit_mean_abs =
        mean_abs_delta(&native_prefill.logits, &oracle_aligned_prefill_logits);
    let prefill_logit_mse =
        mean_square_delta(&native_prefill.logits, &oracle_aligned_prefill_logits);
    let raw_oracle_prefill_logit_max_abs =
        validate::max_abs_delta(&native_prefill.logits, &oracle_output.prefill_logits);
    let gpu_reference_logit_max_abs =
        validate::max_abs_delta(&gpu_reference_logits, &oracle_aligned_prefill_logits);
    let native_vs_gpu_reference_logit_max_abs =
        validate::max_abs_delta(&native_prefill.logits, &gpu_reference_logits);

    let pass = gate_pass(
        prefill_logit_max_abs,
        worst_position.worst_layer_delta,
        &prompt.thresholds,
    );
    timings.push(phase_timing("total", prompt_start));

    Ok(PromptGateAnalysis {
        report: PromptGateReport {
            name: prompt.name.clone(),
            notes: prompt.notes.clone(),
            pass,
            thresholds: prompt.thresholds.clone(),
            prefill_logit_reference: "oracle_final_hidden_recomputed".to_string(),
            prefill_logit_max_abs,
            prefill_logit_mean_abs,
            prefill_logit_mse,
            raw_oracle_prefill_logit_max_abs,
            gpu_reference_logit_max_abs,
            native_vs_gpu_reference_logit_max_abs,
            worst_checked_position: worst_position.position,
            worst_layer: worst_position.worst_layer,
            worst_layer_kind: worst_position.worst_layer_kind.clone(),
            worst_layer_delta: worst_position.worst_layer_delta,
            checked_positions,
            timings,
        },
    })
}

fn phase_timing(phase: &str, start: Instant) -> PhaseTimingReport {
    PhaseTimingReport {
        phase: phase.to_string(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

fn gate_pass(
    prefill_logit_max_abs: f32,
    worst_layer_delta: f32,
    thresholds: &PromptThresholds,
) -> bool {
    prefill_logit_max_abs <= thresholds.prefill_logit_max_abs
        && worst_layer_delta <= thresholds.layer_hidden_max_abs
}

fn choose_worst_position(reports: &[PositionSweepReport]) -> Option<&PositionSweepReport> {
    reports.iter().max_by(|lhs, rhs| {
        lhs.worst_layer_delta
            .total_cmp(&rhs.worst_layer_delta)
            .then_with(|| rhs.position.cmp(&lhs.position))
    })
}

fn sweep_prompt_positions(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
    trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<PositionSweepReport>> {
    let mut reports = Vec::with_capacity(prompt.positions.len());
    for &position in &prompt.positions {
        reports.push(analyze_position_against_trace(
            runtime, prompt, position, trace,
        )?);
    }
    Ok(reports)
}

fn analyze_position_against_trace(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
    position: usize,
    trace: &oracle::Qwen35TraceOutput,
) -> Result<PositionSweepReport> {
    let native =
        run_native_prefill_with_trace(runtime, &prompt.prompt_ids, Some(position), None, None)?;
    let native_layer_trace = native
        .layer_hidden_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native prefill trace missing layer_hidden_trace"))?;
    let mut layers = Vec::with_capacity(native_layer_trace.len());
    let mut worst_layer = 0usize;
    let mut worst_layer_kind = "linear".to_string();
    let mut worst_layer_delta = -1.0f32;
    let mut first_exceeding_layer = None;

    for (layer, native_layer_bytes) in native_layer_trace.iter().enumerate() {
        let oracle_layer = trace
            .decoder_layer_outputs
            .get(layer)
            .and_then(|value| flatten_token_bsd(value, Some(position)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing oracle decoder_layer_outputs[{}] for prompt position {}",
                    layer,
                    position
                )
            })?;
        let native_layer = decode_bf16_le(native_layer_bytes);
        let delta = validate::max_abs_delta(&native_layer, &oracle_layer);
        let kind = BughuntLayerKind::from_model_layer(&runtime.weights.config, layer)
            .as_str()
            .to_string();
        if first_exceeding_layer.is_none() && delta > prompt.thresholds.layer_hidden_max_abs {
            first_exceeding_layer = Some(layer);
        }
        if delta > worst_layer_delta {
            worst_layer = layer;
            worst_layer_kind = kind.clone();
            worst_layer_delta = delta;
        }
        layers.push(LayerDeltaReport {
            layer,
            kind,
            max_abs_delta: delta,
        });
    }

    Ok(PositionSweepReport {
        position,
        worst_layer,
        worst_layer_kind,
        worst_layer_delta: worst_layer_delta.max(0.0),
        first_exceeding_layer,
        layers,
    })
}

fn run_restart_layer_sweep(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
    trace: &oracle::Qwen35TraceOutput,
    selected_position: usize,
) -> Result<Vec<RestartSweepReport>> {
    let oracle_output = oracle::run_oracle(
        &runtime.oracle_script,
        runtime.model_variant.hf_model_id(),
        &prompt.prompt_ids,
        0,
        "bf16",
        &runtime.oracle_device,
        false,
        false,
        None,
        None,
    )?;
    let num_layers = runtime.weights.config.num_hidden_layers;
    let mut reports = Vec::with_capacity(num_layers.saturating_sub(1));

    for start_layer in 1..num_layers {
        eprintln!(
            "[bughunt] restart_sweep prompt={} start_layer={}/{} position={}",
            prompt.name,
            start_layer,
            num_layers - 1,
            selected_position,
        );
        let source_layer = start_layer - 1;
        let source_hidden = trace
            .decoder_layer_outputs
            .get(source_layer)
            .and_then(flatten_bsh)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing oracle decoder_layer_outputs[{}] for restart sweep",
                    source_layer
                )
            })?;
        let hidden_bf16 = encode_bf16_le(&source_hidden);
        let replay = run_tail_replay_with_trace(
            runtime,
            &hidden_bf16,
            start_layer,
            Some(selected_position),
            None,
            None,
        )?;
        let tail_logit_max_abs =
            validate::max_abs_delta(&replay.logits, &oracle_output.prefill_logits);
        let tail_logit_mean_abs = mean_abs_delta(&replay.logits, &oracle_output.prefill_logits);
        let selected_position_worst = compare_tail_position_against_trace(
            runtime,
            &replay,
            trace,
            start_layer,
            selected_position,
        )?;
        let failing = tail_logit_max_abs > prompt.thresholds.restart_tail_logit_max_abs;
        reports.push(RestartSweepReport {
            source_layer,
            start_layer,
            failing,
            tail_logit_max_abs,
            tail_logit_mean_abs,
            selected_position,
            selected_position_worst_layer: selected_position_worst.worst_layer,
            selected_position_worst_layer_delta: selected_position_worst.worst_layer_delta,
        });
    }

    Ok(reports)
}

fn choose_deepest_failing_restart_layer(reports: &[RestartSweepReport]) -> Option<usize> {
    reports
        .iter()
        .filter(|report| report.failing)
        .max_by_key(|report| report.start_layer)
        .map(|report| report.start_layer)
}

fn compare_tail_position_against_trace(
    runtime: &QwenBughuntRuntime,
    replay: &prefill_engine::PrefillResult,
    trace: &oracle::Qwen35TraceOutput,
    start_layer: usize,
    position: usize,
) -> Result<PositionSweepReport> {
    let native_layer_trace = replay
        .layer_hidden_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("tail replay missing layer_hidden_trace"))?;
    let mut layers = Vec::with_capacity(native_layer_trace.len());
    let mut worst_layer = start_layer;
    let mut worst_layer_kind = "linear".to_string();
    let mut worst_layer_delta = -1.0f32;

    for (offset, native_layer_bytes) in native_layer_trace.iter().enumerate() {
        let layer = start_layer + offset;
        let oracle_layer = trace
            .decoder_layer_outputs
            .get(layer)
            .and_then(|value| flatten_token_bsd(value, Some(position)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing oracle decoder_layer_outputs[{}] for restart position {}",
                    layer,
                    position
                )
            })?;
        let native_layer = decode_bf16_le(native_layer_bytes);
        let delta = validate::max_abs_delta(&native_layer, &oracle_layer);
        let kind = BughuntLayerKind::from_model_layer(&runtime.weights.config, layer)
            .as_str()
            .to_string();
        if delta > worst_layer_delta {
            worst_layer = layer;
            worst_layer_kind = kind.clone();
            worst_layer_delta = delta;
        }
        layers.push(LayerDeltaReport {
            layer,
            kind,
            max_abs_delta: delta,
        });
    }

    Ok(PositionSweepReport {
        position,
        worst_layer,
        worst_layer_kind,
        worst_layer_delta: worst_layer_delta.max(0.0),
        first_exceeding_layer: None,
        layers,
    })
}

fn run_restart_position_scan(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
    trace: &oracle::Qwen35TraceOutput,
    start_layer: usize,
) -> Result<Vec<RestartPositionScanReport>> {
    let source_layer = start_layer
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("restart position scan requires start_layer >= 1"))?;
    let source_hidden = trace
        .decoder_layer_outputs
        .get(source_layer)
        .and_then(flatten_bsh)
        .ok_or_else(|| anyhow::anyhow!("missing oracle decoder_layer_outputs[{}]", source_layer))?;
    let hidden_bf16 = encode_bf16_le(&source_hidden);
    let mut reports = Vec::with_capacity(prompt.positions.len());

    for &position in &prompt.positions {
        eprintln!(
            "[bughunt] restart_position_scan prompt={} start_layer={} position={}",
            prompt.name, start_layer, position
        );
        let replay = run_tail_replay_with_trace(
            runtime,
            &hidden_bf16,
            start_layer,
            Some(position),
            None,
            None,
        )?;
        let compared =
            compare_tail_position_against_trace(runtime, &replay, trace, start_layer, position)?;
        let native_layer_trace = replay
            .layer_hidden_trace
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restart position scan missing layer_hidden_trace"))?;
        let native_final_hidden = native_layer_trace
            .last()
            .map(|bytes| decode_bf16_le(bytes))
            .ok_or_else(|| anyhow::anyhow!("restart position scan missing final hidden trace"))?;
        let oracle_final_hidden = trace
            .decoder_layer_outputs
            .last()
            .and_then(|value| flatten_token_bsd(value, Some(position)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "restart position scan missing oracle final hidden for position {}",
                    position
                )
            })?;
        let native_hidden_logits =
            compute_qwen_logits_from_hidden_row(runtime, &native_final_hidden)?;
        let oracle_hidden_logits =
            compute_qwen_logits_from_hidden_row(runtime, &oracle_final_hidden)?;
        reports.push(RestartPositionScanReport {
            position,
            worst_layer: compared.worst_layer,
            worst_layer_kind: compared.worst_layer_kind,
            worst_layer_delta: compared.worst_layer_delta,
            final_hidden_logit_max_abs: validate::max_abs_delta(
                &native_hidden_logits,
                &oracle_hidden_logits,
            ),
        });
    }

    Ok(reports)
}

fn choose_worst_restart_position(
    reports: &[RestartPositionScanReport],
) -> Option<&RestartPositionScanReport> {
    reports.iter().max_by(|lhs, rhs| {
        lhs.worst_layer_delta
            .total_cmp(&rhs.worst_layer_delta)
            .then_with(|| {
                lhs.final_hidden_logit_max_abs
                    .total_cmp(&rhs.final_hidden_logit_max_abs)
            })
    })
}

fn build_traced_metrics(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    position: usize,
    layer: usize,
    layer_kind: BughuntLayerKind,
) -> Result<TracedMetricsReport> {
    eprintln!(
        "[bughunt] traced_metrics layer={}({}) position={} native_trace",
        layer,
        layer_kind.as_str(),
        position,
    );
    let native = run_native_prefill_with_trace(
        runtime,
        prompt_ids,
        Some(position),
        Some(layer),
        Some(layer_kind),
    )?;
    eprintln!(
        "[bughunt] traced_metrics layer={}({}) position={} oracle_trace",
        layer,
        layer_kind.as_str(),
        position,
    );
    let trace = run_trace_oracle(
        runtime,
        prompt_ids,
        Some(position),
        Some(layer),
        Some(layer_kind),
    )?;

    let stages = match layer_kind {
        BughuntLayerKind::Linear => {
            build_linear_stage_metrics(runtime, layer, position, &native, &trace)?
        }
        BughuntLayerKind::Full => {
            build_full_stage_metrics(runtime, layer, position, &native, &trace)?
        }
        BughuntLayerKind::Mlp => {
            build_mlp_stage_metrics(runtime, layer, position, &native, &trace)?
        }
    };
    let max_stage_delta = stages
        .iter()
        .map(|stage| stage.max_abs_delta)
        .fold(0.0f32, f32::max);

    Ok(TracedMetricsReport {
        layer,
        layer_kind: layer_kind.as_str().to_string(),
        position,
        max_stage_delta,
        stages,
    })
}

fn build_linear_stage_metrics(
    runtime: &QwenBughuntRuntime,
    layer: usize,
    position: usize,
    native: &prefill_engine::PrefillResult,
    trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<StageMetricReport>> {
    let native = native
        .linear_debug_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native linear debug trace missing"))?;

    let oracle_normed = require_trace_vec(
        trace
            .trace_linear_input_layernorm_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_input_layernorm_output",
    )?;
    let oracle_qkv = require_trace_vec(
        trace
            .trace_linear_qkv_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_qkv_output",
    )?;
    let oracle_z = require_trace_vec(
        trace
            .trace_linear_z_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_z_output",
    )?;
    let oracle_post_conv = require_trace_vec(
        trace
            .trace_linear_post_conv_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_post_conv_output",
    )?;
    let oracle_recurrent = require_trace_vec(
        trace
            .trace_linear_direct_recurrent_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_direct_recurrent_output",
    )?;
    let oracle_gated = require_trace_vec(
        trace
            .trace_linear_norm_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_norm_output",
    )?;
    let oracle_proj_out = require_trace_vec(
        trace
            .trace_linear_token_mixer_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_linear_token_mixer_output",
    )?;

    let key_dim =
        runtime.weights.config.linear_num_key_heads * runtime.weights.config.linear_key_head_dim;
    let val_dim = runtime.weights.config.linear_num_value_heads
        * runtime.weights.config.linear_value_head_dim;
    let qkv_dim = key_dim * 2 + val_dim;
    let oracle_conv_window = trace
        .trace_linear_qkv_output
        .as_ref()
        .and_then(|value| {
            extract_causal_conv_window_bsd(
                value,
                position,
                qkv_dim,
                runtime.weights.config.linear_conv_kernel_dim,
            )
        })
        .ok_or_else(|| anyhow::anyhow!("trace_linear_qkv_output missing causal conv window"))?;

    let native_qkv = decode_bf16_le(&native.qkv);
    let (oracle_q, oracle_k, oracle_v) = split_linear_qkv(&oracle_qkv, key_dim, val_dim)?;
    let (native_q, native_k, native_v) = split_linear_qkv(&native_qkv, key_dim, val_dim)?;
    let linear_weights = runtime.weights.layers[layer]
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing linear weights for layer {}", layer))?;
    let host_qkv_from_native_normed = host_projection_bf16_rounded(
        &decode_bf16_le(&native.normed),
        &linear_weights.qkv_proj_w,
        qkv_dim,
    )?;
    let host_qkv_from_oracle_normed =
        host_projection_bf16_rounded(&oracle_normed, &linear_weights.qkv_proj_w, qkv_dim)?;
    let (_, _, host_v_from_native_normed) =
        split_linear_qkv(&host_qkv_from_native_normed, key_dim, val_dim)?;
    let (_, _, host_v_from_oracle_normed) =
        split_linear_qkv(&host_qkv_from_oracle_normed, key_dim, val_dim)?;
    let native_z = decode_bf16_le(&native.z);
    let host_z_from_native_normed = host_projection_bf16_rounded(
        &decode_bf16_le(&native.normed),
        &linear_weights.z_proj_w,
        val_dim,
    )?;
    let host_z_from_oracle_normed =
        host_projection_bf16_rounded(&oracle_normed, &linear_weights.z_proj_w, val_dim)?;

    let native_recurrent = decode_f32_le(&native.rec_apply);
    let recurrent_len = native_recurrent.len().min(oracle_recurrent.len());
    let native_attn = decode_bf16_le(&native.attn);
    let native_gated = decode_bf16_le(&native.gated);
    let host_gated_from_native_inputs = host_linear_gated_bf16_rounded(
        &native_attn,
        &native_z,
        &linear_weights.norm_w,
        runtime.weights.config.linear_num_value_heads,
        runtime.weights.config.linear_value_head_dim,
        runtime.weights.config.rms_norm_eps as f32,
    )?;
    let host_gated_from_oracle_inputs = host_linear_gated_bf16_rounded(
        &oracle_recurrent,
        &oracle_z,
        &linear_weights.norm_w,
        runtime.weights.config.linear_num_value_heads,
        runtime.weights.config.linear_value_head_dim,
        runtime.weights.config.rms_norm_eps as f32,
    )?;
    let host_gated_from_native_attn_oracle_z = host_linear_gated_bf16_rounded(
        &native_attn,
        &oracle_z,
        &linear_weights.norm_w,
        runtime.weights.config.linear_num_value_heads,
        runtime.weights.config.linear_value_head_dim,
        runtime.weights.config.rms_norm_eps as f32,
    )?;
    let host_gated_from_oracle_attn_native_z = host_linear_gated_bf16_rounded(
        &oracle_recurrent,
        &native_z,
        &linear_weights.norm_w,
        runtime.weights.config.linear_num_value_heads,
        runtime.weights.config.linear_value_head_dim,
        runtime.weights.config.rms_norm_eps as f32,
    )?;

    Ok(vec![
        compare_stage(
            "input_norm",
            "normed",
            "trace_linear_input_layernorm_output",
            decode_bf16_le(&native.normed),
            oracle_normed,
        )?,
        compare_stage(
            "qkv",
            "qkv",
            "trace_linear_qkv_output",
            native_qkv.clone(),
            oracle_qkv.clone(),
        )?,
        compare_stage(
            "qkv_q",
            "qkv[:key_dim]",
            "trace_linear_qkv_output[:key_dim]",
            native_q.clone(),
            oracle_q.clone(),
        )?,
        compare_stage(
            "qkv_k",
            "qkv[key_dim:2*key_dim]",
            "trace_linear_qkv_output[key_dim:2*key_dim]",
            native_k.clone(),
            oracle_k.clone(),
        )?,
        compare_stage(
            "qkv_v",
            "qkv[2*key_dim:]",
            "trace_linear_qkv_output[2*key_dim:]",
            native_v.clone(),
            oracle_v.clone(),
        )?,
        compare_stage(
            "qkv_host_from_native_normed_consistency",
            "qkv",
            "host_bf16_round(native_normed @ qkv_proj_w)",
            native_qkv.clone(),
            host_qkv_from_native_normed.clone(),
        )?,
        compare_stage(
            "qkv_v_host_from_native_normed_consistency",
            "qkv[2*key_dim:]",
            "host_bf16_round(native_normed @ qkv_proj_w)[2*key_dim:]",
            native_v,
            host_v_from_native_normed.clone(),
        )?,
        compare_stage(
            "qkv_host_from_oracle_normed_consistency",
            "trace_linear_qkv_output",
            "host_bf16_round(oracle_normed @ qkv_proj_w)",
            oracle_qkv.clone(),
            host_qkv_from_oracle_normed.clone(),
        )?,
        compare_stage(
            "qkv_v_host_from_oracle_normed_consistency",
            "trace_linear_qkv_output[2*key_dim:]",
            "host_bf16_round(oracle_normed @ qkv_proj_w)[2*key_dim:]",
            oracle_v.clone(),
            host_v_from_oracle_normed,
        )?,
        compare_stage(
            "z",
            "z",
            "trace_linear_z_output",
            native_z.clone(),
            oracle_z.clone(),
        )?,
        compare_stage(
            "z_host_from_native_normed_consistency",
            "z",
            "host_bf16_round(native_normed @ z_proj_w)",
            native_z,
            host_z_from_native_normed,
        )?,
        compare_stage(
            "z_host_from_oracle_normed_consistency",
            "trace_linear_z_output",
            "host_bf16_round(oracle_normed @ z_proj_w)",
            oracle_z.clone(),
            host_z_from_oracle_normed,
        )?,
        compare_stage(
            "conv_window",
            "conv_window",
            "trace_linear_qkv_output(causal_window)",
            decode_bf16_le(&native.conv_window),
            oracle_conv_window,
        )?,
        compare_stage(
            "post_conv",
            "post_conv",
            "trace_linear_post_conv_output",
            decode_bf16_le(&native.post_conv),
            oracle_post_conv,
        )?,
        compare_stage(
            "recurrent",
            "rec_apply",
            "trace_linear_direct_recurrent_output",
            native_recurrent[..recurrent_len].to_vec(),
            oracle_recurrent[..recurrent_len].to_vec(),
        )?,
        compare_stage(
            "attn",
            "attn",
            "trace_linear_direct_recurrent_output",
            native_attn.clone(),
            oracle_recurrent.clone(),
        )?,
        compare_stage(
            "gated",
            "gated",
            "trace_linear_norm_output",
            native_gated.clone(),
            oracle_gated.clone(),
        )?,
        compare_stage(
            "gated_host_from_native_inputs_consistency",
            "gated",
            "host_bf16_round(rms_norm_gated(native_attn, native_z))",
            native_gated.clone(),
            host_gated_from_native_inputs,
        )?,
        compare_stage(
            "gated_host_from_oracle_inputs_consistency",
            "trace_linear_norm_output",
            "host_bf16_round(rms_norm_gated(oracle_attn, oracle_z))",
            oracle_gated.clone(),
            host_gated_from_oracle_inputs,
        )?,
        compare_stage(
            "gated_native_attn_oracle_z_delta",
            "host_bf16_round(rms_norm_gated(native_attn, oracle_z))",
            "trace_linear_norm_output",
            host_gated_from_native_attn_oracle_z,
            oracle_gated.clone(),
        )?,
        compare_stage(
            "gated_oracle_attn_native_z_delta",
            "host_bf16_round(rms_norm_gated(oracle_attn, native_z))",
            "trace_linear_norm_output",
            host_gated_from_oracle_attn_native_z,
            oracle_gated,
        )?,
        compare_stage(
            "proj_out",
            "proj_out",
            "trace_linear_token_mixer_output",
            decode_bf16_le(&native.proj_out),
            oracle_proj_out,
        )?,
    ])
}

fn build_full_stage_metrics(
    runtime: &QwenBughuntRuntime,
    layer: usize,
    position: usize,
    native: &prefill_engine::PrefillResult,
    trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<StageMetricReport>> {
    let native = native
        .layer3_full_attn_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native full-attention debug trace missing"))?;
    let config = &runtime.weights.config;
    let hidden_dim = config.hidden_size;
    let num_q_heads = config.num_attention_heads;
    let head_dim = config.head_dim;
    let q_dim = num_q_heads * head_dim;

    let oracle_input_hidden = if layer == 0 {
        bail!("full-attention trace requires layer >= 1");
    } else {
        trace
            .decoder_layer_outputs
            .get(layer - 1)
            .and_then(|value| flatten_token_bsd(value, Some(position)))
            .ok_or_else(|| {
                anyhow::anyhow!("missing oracle input hidden for full layer {}", layer)
            })?
    };
    let oracle_input_norm = compute_qwen_rms_norm_from_hidden_row(
        &oracle_input_hidden,
        &runtime.weights.layers[layer].input_norm_w,
        runtime.weights.config.rms_norm_eps as f32,
    )?;
    let oracle_q_and_gate = require_trace_vec(
        trace
            .trace_full_q_and_gate_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_q_and_gate_output",
    )?;
    if oracle_q_and_gate.len() != q_dim * 2 {
        bail!(
            "trace_full_q_and_gate_output len {} did not match expected {}",
            oracle_q_and_gate.len(),
            q_dim * 2
        );
    }
    let (oracle_q_proj, oracle_gate_from_q_and_gate) =
        split_qgate_heads(&oracle_q_and_gate, num_q_heads, head_dim)?;
    let oracle_gate = require_trace_vec(
        trace
            .trace_full_gate_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_gate_output",
    )?;
    let oracle_k_proj = require_trace_vec(
        trace
            .trace_full_k_proj_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_k_proj_output",
    )?;
    let oracle_v_proj = require_trace_vec(
        trace
            .trace_full_v_proj_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_v_proj_output",
    )?;
    let oracle_q_prepared = require_trace_vec(
        trace
            .trace_full_prepared_query_output
            .as_ref()
            .and_then(|value| flatten_token_bhsd(value, Some(position))),
        "trace_full_prepared_query_output",
    )?;
    let oracle_k_prepared = require_trace_vec(
        trace
            .trace_full_prepared_key_output
            .as_ref()
            .and_then(|value| flatten_token_bhsd(value, Some(position))),
        "trace_full_prepared_key_output",
    )?;
    let oracle_v_prepared = require_trace_vec(
        trace
            .trace_full_prepared_value_output
            .as_ref()
            .and_then(|value| flatten_token_bhsd(value, Some(position))),
        "trace_full_prepared_value_output",
    )?;
    let oracle_q_rotated = require_trace_vec(
        trace
            .trace_full_rotated_query_output
            .as_ref()
            .and_then(|value| flatten_token_bhsd(value, Some(position))),
        "trace_full_rotated_query_output",
    )?;
    let oracle_k_rotated = require_trace_vec(
        trace
            .trace_full_rotated_key_output
            .as_ref()
            .and_then(|value| flatten_token_bhsd(value, Some(position))),
        "trace_full_rotated_key_output",
    )?;
    let oracle_attn_pregate = require_trace_vec(
        trace
            .trace_full_raw_attention_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_raw_attention_output",
    )?;
    let oracle_attn_output = require_trace_vec(
        trace
            .trace_full_attention_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_full_attention_output",
    )?;
    let native_q_and_gate = decode_bf16_le(&native.q_and_gate);
    let native_q_proj = decode_bf16_le(&native.q_proj);
    let native_gate_proj = decode_bf16_le(&native.gate_proj);
    let (native_q_slice, native_gate_slice) =
        split_qgate_heads(&native_q_and_gate, num_q_heads, head_dim)?;

    if oracle_input_norm.len() != hidden_dim {
        bail!(
            "oracle input norm len {} did not match hidden size {}",
            oracle_input_norm.len(),
            hidden_dim
        );
    }

    Ok(vec![
        compare_stage(
            "input_norm",
            "input_norm",
            "decoder_layer_outputs[layer-1] -> rms_norm(input_layernorm)",
            decode_bf16_le(&native.input_norm),
            oracle_input_norm,
        )?,
        compare_stage(
            "q_and_gate",
            "q_and_gate",
            "trace_full_q_and_gate_output",
            native_q_and_gate.clone(),
            oracle_q_and_gate,
        )?,
        compare_stage(
            "q_proj",
            "q_proj",
            "split(trace_full_q_and_gate_output).q",
            native_q_proj.clone(),
            oracle_q_proj,
        )?,
        compare_stage(
            "gate_proj",
            "gate_proj",
            "trace_full_gate_output",
            native_gate_proj.clone(),
            oracle_gate,
        )?,
        compare_stage(
            "oracle_gate_consistency",
            "trace_full_gate_output",
            "split(trace_full_q_and_gate_output).gate",
            native_gate_proj.clone(),
            oracle_gate_from_q_and_gate,
        )?,
        compare_stage(
            "native_q_slice_consistency",
            "q_proj",
            "split(native.q_and_gate).q",
            native_q_proj,
            native_q_slice,
        )?,
        compare_stage(
            "native_gate_slice_consistency",
            "gate_proj",
            "split(native.q_and_gate).gate",
            native_gate_proj,
            native_gate_slice,
        )?,
        compare_stage(
            "k_proj",
            "k_proj",
            "trace_full_k_proj_output",
            decode_bf16_le(&native.k_proj),
            oracle_k_proj,
        )?,
        compare_stage(
            "v_proj",
            "v_proj",
            "trace_full_v_proj_output",
            decode_bf16_le(&native.v_proj),
            oracle_v_proj,
        )?,
        compare_stage(
            "q_prepared",
            "q_prepared",
            "trace_full_prepared_query_output",
            decode_bf16_le(&native.q_prepared),
            oracle_q_prepared,
        )?,
        compare_stage(
            "k_prepared",
            "k_prepared",
            "trace_full_prepared_key_output",
            decode_bf16_le(&native.k_prepared),
            oracle_k_prepared,
        )?,
        compare_stage(
            "q_rotated",
            "q_rotated",
            "trace_full_rotated_query_output",
            decode_bf16_le(&native.q_rotated),
            oracle_q_rotated,
        )?,
        compare_stage(
            "k_rotated",
            "k_rotated",
            "trace_full_rotated_key_output",
            decode_bf16_le(&native.k_rotated),
            oracle_k_rotated,
        )?,
        compare_stage(
            "v_prepared",
            "v_prepared",
            "trace_full_prepared_value_output",
            decode_bf16_le(&native.v_prepared),
            oracle_v_prepared,
        )?,
        compare_stage(
            "attn_pregate",
            "attn_raw",
            "trace_full_raw_attention_output",
            decode_bf16_le(&native.attn_raw),
            oracle_attn_pregate,
        )?,
        compare_stage(
            "attn_output",
            "attn_output",
            "trace_full_attention_output",
            decode_bf16_le(&native.attn_output),
            oracle_attn_output,
        )?,
    ])
}

fn build_mlp_stage_metrics(
    runtime: &QwenBughuntRuntime,
    layer: usize,
    position: usize,
    native: &prefill_engine::PrefillResult,
    trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<StageMetricReport>> {
    let native = native
        .mlp_debug_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native mlp debug trace missing"))?;
    let oracle_pre_mlp_hidden = require_trace_vec(
        trace
            .trace_mlp_post_attention_layernorm_input
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_post_attention_layernorm_input",
    )?;
    let oracle_post_norm = require_trace_vec(
        trace
            .trace_mlp_post_attention_layernorm_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_post_attention_layernorm_output",
    )?;
    let oracle_gate_proj = require_trace_vec(
        trace
            .trace_mlp_gate_proj_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_gate_proj_output",
    )?;
    let oracle_up_proj = require_trace_vec(
        trace
            .trace_mlp_up_proj_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_up_proj_output",
    )?;
    let oracle_swiglu = require_trace_vec(
        trace
            .trace_mlp_activated_hidden
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_activated_hidden",
    )?;
    let oracle_down_proj = require_trace_vec(
        trace
            .trace_mlp_down_proj_output
            .as_ref()
            .and_then(|value| flatten_token_bsd(value, Some(position))),
        "trace_mlp_down_proj_output",
    )?;
    let layer_weights = &runtime.weights.layers[layer];
    let hidden_dim = runtime.weights.config.hidden_size;
    let intermediate_dim = runtime.weights.config.intermediate_size;

    let native_pre_mlp_hidden = decode_bf16_le(&native.pre_mlp_hidden);
    let native_post_norm = decode_bf16_le(&native.post_norm);
    let native_gate_proj = decode_bf16_le(&native.gate_proj);
    let native_up_proj = decode_bf16_le(&native.up_proj);
    let native_swiglu = decode_bf16_le(&native.swiglu);
    let native_down_proj = decode_bf16_le(&native.down_proj);
    let host_post_norm_from_native_hidden = bf16_round_vec(&compute_qwen_rms_norm_from_hidden_row(
        &native_pre_mlp_hidden,
        &layer_weights.post_attn_norm_w,
        runtime.weights.config.rms_norm_eps as f32,
    )?);
    let host_post_norm_from_oracle_hidden = bf16_round_vec(&compute_qwen_rms_norm_from_hidden_row(
        &oracle_pre_mlp_hidden,
        &layer_weights.post_attn_norm_w,
        runtime.weights.config.rms_norm_eps as f32,
    )?);

    let host_gate_from_native_post_norm = host_projection_bf16_rounded(
        &native_post_norm,
        &layer_weights.gate_proj_w,
        intermediate_dim,
    )?;
    let host_gate_from_oracle_post_norm = host_projection_bf16_rounded(
        &oracle_post_norm,
        &layer_weights.gate_proj_w,
        intermediate_dim,
    )?;
    let host_up_from_native_post_norm = host_projection_bf16_rounded(
        &native_post_norm,
        &layer_weights.up_proj_w,
        intermediate_dim,
    )?;
    let host_up_from_oracle_post_norm = host_projection_bf16_rounded(
        &oracle_post_norm,
        &layer_weights.up_proj_w,
        intermediate_dim,
    )?;
    let host_down_from_native_swiglu =
        host_projection_bf16_rounded(&native_swiglu, &layer_weights.down_proj_w, hidden_dim)?;
    let host_down_from_oracle_swiglu =
        host_projection_bf16_rounded(&oracle_swiglu, &layer_weights.down_proj_w, hidden_dim)?;

    Ok(vec![
        compare_stage(
            "pre_mlp_hidden",
            "pre_mlp_hidden",
            "trace_mlp_post_attention_layernorm_input",
            native_pre_mlp_hidden,
            oracle_pre_mlp_hidden,
        )?,
        compare_stage(
            "post_norm",
            "post_norm",
            "trace_mlp_post_attention_layernorm_output",
            native_post_norm.clone(),
            oracle_post_norm.clone(),
        )?,
        compare_stage(
            "post_norm_host_from_native_hidden_consistency",
            "post_norm",
            "host_bf16_round(rms_norm(native_pre_mlp_hidden))",
            native_post_norm.clone(),
            host_post_norm_from_native_hidden,
        )?,
        compare_stage(
            "post_norm_host_from_oracle_hidden_consistency",
            "trace_mlp_post_attention_layernorm_output",
            "host_bf16_round(rms_norm(oracle_pre_mlp_hidden))",
            oracle_post_norm.clone(),
            host_post_norm_from_oracle_hidden,
        )?,
        compare_stage(
            "gate_proj",
            "gate_proj",
            "trace_mlp_gate_proj_output",
            native_gate_proj.clone(),
            oracle_gate_proj.clone(),
        )?,
        compare_stage(
            "gate_proj_host_from_native_post_norm_consistency",
            "gate_proj",
            "host_bf16_round(native_post_norm @ gate_proj_w)",
            native_gate_proj,
            host_gate_from_native_post_norm,
        )?,
        compare_stage(
            "gate_proj_host_from_oracle_post_norm_consistency",
            "trace_mlp_gate_proj_output",
            "host_bf16_round(oracle_post_norm @ gate_proj_w)",
            oracle_gate_proj.clone(),
            host_gate_from_oracle_post_norm,
        )?,
        compare_stage(
            "up_proj",
            "up_proj",
            "trace_mlp_up_proj_output",
            native_up_proj.clone(),
            oracle_up_proj.clone(),
        )?,
        compare_stage(
            "up_proj_host_from_native_post_norm_consistency",
            "up_proj",
            "host_bf16_round(native_post_norm @ up_proj_w)",
            native_up_proj,
            host_up_from_native_post_norm,
        )?,
        compare_stage(
            "up_proj_host_from_oracle_post_norm_consistency",
            "trace_mlp_up_proj_output",
            "host_bf16_round(oracle_post_norm @ up_proj_w)",
            oracle_up_proj.clone(),
            host_up_from_oracle_post_norm,
        )?,
        compare_stage(
            "swiglu",
            "swiglu",
            "trace_mlp_activated_hidden",
            native_swiglu.clone(),
            oracle_swiglu.clone(),
        )?,
        compare_stage(
            "down_proj",
            "down_proj",
            "trace_mlp_down_proj_output",
            native_down_proj.clone(),
            oracle_down_proj.clone(),
        )?,
        compare_stage(
            "down_proj_host_from_native_swiglu_consistency",
            "down_proj",
            "host_bf16_round(native_swiglu @ down_proj_w)",
            native_down_proj,
            host_down_from_native_swiglu,
        )?,
        compare_stage(
            "down_proj_host_from_oracle_swiglu_consistency",
            "trace_mlp_down_proj_output",
            "host_bf16_round(oracle_swiglu @ down_proj_w)",
            oracle_down_proj,
            host_down_from_oracle_swiglu,
        )?,
    ])
}

fn require_trace_vec(value: Option<Vec<f32>>, label: &str) -> Result<Vec<f32>> {
    value.ok_or_else(|| anyhow::anyhow!("missing {}", label))
}

fn compare_stage(
    stage: &str,
    native_field: &str,
    oracle_field: &str,
    native: Vec<f32>,
    oracle: Vec<f32>,
) -> Result<StageMetricReport> {
    if native.len() != oracle.len() {
        bail!(
            "stage '{}' length mismatch: native={} oracle={}",
            stage,
            native.len(),
            oracle.len()
        );
    }
    let (max_index, native_at_max, oracle_at_max, max_abs_delta) =
        max_abs_delta_details(&native, &oracle);
    Ok(StageMetricReport {
        stage: stage.to_string(),
        native_field: native_field.to_string(),
        oracle_field: oracle_field.to_string(),
        len: native.len(),
        max_abs_delta,
        mean_abs_delta: mean_abs_delta(&native, &oracle),
        mse: mean_square_delta(&native, &oracle),
        max_index,
        native_at_max,
        oracle_at_max,
        top_dims: top_abs_delta_dims(&native, &oracle, 6),
    })
}

fn split_qgate_heads(
    values: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected = num_heads * head_dim * 2;
    if values.len() != expected {
        bail!(
            "q_and_gate len {} did not match expected {} for {} heads x {} head_dim",
            values.len(),
            expected,
            num_heads,
            head_dim
        );
    }

    let mut q = Vec::with_capacity(num_heads * head_dim);
    let mut gate = Vec::with_capacity(num_heads * head_dim);
    for head in 0..num_heads {
        let base = head * head_dim * 2;
        q.extend_from_slice(&values[base..base + head_dim]);
        gate.extend_from_slice(&values[base + head_dim..base + head_dim * 2]);
    }

    Ok((q, gate))
}

fn split_linear_qkv(
    values: &[f32],
    key_dim: usize,
    val_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let expected = key_dim * 2 + val_dim;
    if values.len() != expected {
        bail!(
            "linear qkv len {} did not match expected {} (key_dim={} val_dim={})",
            values.len(),
            expected,
            key_dim,
            val_dim
        );
    }

    Ok((
        values[..key_dim].to_vec(),
        values[key_dim..key_dim * 2].to_vec(),
        values[key_dim * 2..].to_vec(),
    ))
}

fn host_projection_bf16_rounded(
    input_row: &[f32],
    weight_buf: &GpuBuffer,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let in_dim = input_row.len();
    let weights = read_buffer_all_f32(weight_buf)?;
    if weights.len() != out_dim * in_dim {
        bail!(
            "projection weight len {} did not match expected {} x {}",
            weights.len(),
            out_dim,
            in_dim
        );
    }

    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let weight_row = &weights[row * in_dim..(row + 1) * in_dim];
        let mut acc = 0.0f32;
        for (lhs, rhs) in input_row.iter().zip(weight_row.iter()) {
            acc += lhs * rhs;
        }
        out[row] = half::bf16::from_f32(acc).to_f32();
    }
    Ok(out)
}

fn host_linear_gated_bf16_rounded(
    attn: &[f32],
    z: &[f32],
    norm_weight_buf: &GpuBuffer,
    num_value_heads: usize,
    value_head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let expected = num_value_heads * value_head_dim;
    if attn.len() != expected || z.len() != expected {
        bail!(
            "linear gated input length mismatch: attn={} z={} expected={}",
            attn.len(),
            z.len(),
            expected
        );
    }
    let norm_weight = read_buffer_all_f32(norm_weight_buf)?;
    if norm_weight.len() != value_head_dim {
        bail!(
            "linear gated norm weight length {} did not match value head dim {}",
            norm_weight.len(),
            value_head_dim
        );
    }

    let mut out = vec![0.0f32; expected];
    for head in 0..num_value_heads {
        let base = head * value_head_dim;
        let row = &attn[base..base + value_head_dim];
        let inv_rms = 1.0f32 / (mean_square(row) + eps).sqrt();
        for col in 0..value_head_dim {
            let idx = base + col;
            let value = row[col] * inv_rms * norm_weight[col] * qwen_silu(z[idx]);
            out[idx] = half::bf16::from_f32(value).to_f32();
        }
    }
    Ok(out)
}

fn qwen_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn bf16_round_vec(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| half::bf16::from_f32(*value).to_f32())
        .collect()
}

fn run_native_prefill(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
) -> Result<prefill_engine::PrefillResult> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("native prefill model state init: {e}"))?;
    prefill_engine::prefill(
        &runtime.weights,
        &mut state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        false,
        None,
        None,
        None,
    )
}

fn run_native_prefill_greedy_token_with_state(
    runtime: &QwenBughuntRuntime,
    state: &mut ModelState,
    prompt_ids: &[u32],
) -> Result<u32> {
    state.reset_for_prefill_reuse();
    prefill_engine::prefill_greedy_token(
        &runtime.weights,
        state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
    )
}

fn run_native_prefill_with_trace(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<prefill_engine::PrefillResult> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("native traced prefill model state init: {e}"))?;
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    prefill_engine::prefill_with_trace_position(
        &runtime.weights,
        &mut state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        true,
        debug_linear_layer,
        debug_full_layer,
        debug_mlp_layer,
        trace_position,
    )
}

fn run_tail_replay_with_trace(
    runtime: &QwenBughuntRuntime,
    hidden_bf16: &[u8],
    start_layer: usize,
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<prefill_engine::PrefillResult> {
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("tail replay model state init: {e}"))?;
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    prefill_engine::prefill_tail_from_hidden_with_trace_position(
        &runtime.weights,
        &mut state,
        &runtime.rotary,
        hidden_bf16,
        start_layer,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        true,
        debug_linear_layer,
        debug_full_layer,
        debug_mlp_layer,
        trace_position,
    )
}

fn run_trace_oracle(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<oracle::Qwen35TraceOutput> {
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    oracle::run_qwen35_trace_oracle(
        &runtime.qwen35_trace_script,
        runtime.model_variant.hf_model_id(),
        prompt_ids,
        0,
        "bf16",
        &runtime.oracle_device,
        debug_linear_layer,
        debug_full_layer,
        debug_mlp_layer,
        trace_position,
    )
}

fn debug_layer_flags(
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    match (debug_layer, debug_kind) {
        (Some(layer), Some(BughuntLayerKind::Linear)) => (Some(layer), None, None),
        (Some(layer), Some(BughuntLayerKind::Full)) => (None, Some(layer), None),
        (Some(layer), Some(BughuntLayerKind::Mlp)) => (None, None, Some(layer)),
        _ => (None, None, None),
    }
}

fn print_report_summary(report: &BughuntReport) {
    match report.mode.as_str() {
        "gate" => {
            if let Some(gate) = report.gate.as_ref() {
                println!(
                    "mode=gate backend={} prompts={} pass={}",
                    report.metadata.backend,
                    gate.prompt_results.len(),
                    gate.pass
                );
                for prompt in &gate.prompt_results {
                    let native_ms = prompt
                        .timings
                        .iter()
                        .find(|timing| timing.phase == "native_prefill")
                        .map(|timing| timing.elapsed_ms)
                        .unwrap_or(0.0);
                    let total_ms = prompt
                        .timings
                        .iter()
                        .find(|timing| timing.phase == "total")
                        .map(|timing| timing.elapsed_ms)
                        .unwrap_or(0.0);
                    println!(
                        "{} prompt={} prefill_max_abs={:.4} gpu_ref_max_abs={:.4} worst_position={} worst_layer={}({}) worst_layer_delta={:.4} native_prefill_ms={:.1} total_ms={:.1}",
                        if prompt.pass { "PASS" } else { "FAIL" },
                        prompt.name,
                        prompt.prefill_logit_max_abs,
                        prompt.gpu_reference_logit_max_abs,
                        prompt.worst_checked_position,
                        prompt.worst_layer,
                        prompt.worst_layer_kind,
                        prompt.worst_layer_delta,
                        native_ms,
                        total_ms,
                    );
                }
            }
        }
        "localize" => {
            if let Some(localize) = report.localize.as_ref() {
                println!(
                    "mode=localize prompt={} pass={} worst_position={} suspicious_layer={} restart_layer={} sampled_position={}",
                    localize.gate_prompt.name,
                    localize.pass,
                    localize.localization.initial_suspicious_position,
                    localize.localization.initial_suspicious_layer,
                    localize
                        .localization
                        .first_suspicious_restart_layer
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    localize
                        .localization
                        .worst_sampled_position
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                );
                if let Some(position) = localize.localization.worst_sampled_position {
                    println!("worst_sampled_position={}", position);
                }
                if let Some(layer) = localize.localization.chosen_traced_layer {
                    println!(
                        "traced_layer={}({}) max_stage_delta={}",
                        layer,
                        localize
                            .localization
                            .chosen_traced_layer_kind
                            .as_deref()
                            .unwrap_or("n/a"),
                        localize
                            .localization
                            .traced_metrics
                            .as_ref()
                            .map(|metrics| format!("{:.4}", metrics.max_stage_delta))
                            .unwrap_or_else(|| "n/a".to_string())
                    );
                }
            }
        }
        "dump" => {
            if let Some(dump) = report.dump.as_ref() {
                println!(
                    "mode=dump prompt={} pass={} position={} layer={}({}) max_stage_delta={:.4}",
                    dump.gate_prompt.name,
                    dump.pass,
                    dump.dump.position,
                    dump.dump.layer,
                    dump.dump.layer_kind,
                    dump.dump.traced_metrics.max_stage_delta,
                );
                for stage in &dump.dump.traced_metrics.stages {
                    println!(
                        "stage={} max_abs_delta={:.4} mean_abs_delta={:.4e} mse={:.4e}",
                        stage.stage, stage.max_abs_delta, stage.mean_abs_delta, stage.mse,
                    );
                }
            }
        }
        "bench" => {
            if let Some(bench) = report.bench.as_ref() {
                println!(
                    "mode=bench backend={} prompts={} pass={}",
                    report.metadata.backend,
                    bench.prompt_results.len(),
                    bench.pass
                );
                for prompt in &bench.prompt_results {
                    println!(
                        "BENCH prompt={} tokens={} iters={} warmup={} native_prefill_ms_mean={:.1} min={:.1} max={:.1} greedy_prefill_ms_mean={:.1} min={:.1} max={:.1} decode_tokens={} replay_decode_ms_per_token_mean={} component_decode_ms_per_token_mean={}",
                        prompt.name,
                        prompt.prompt_len,
                        prompt.iterations,
                        prompt.warmup_iterations,
                        prompt.mean_native_prefill_ms,
                        prompt.min_native_prefill_ms,
                        prompt.max_native_prefill_ms,
                        prompt.mean_greedy_prefill_ms,
                        prompt.min_greedy_prefill_ms,
                        prompt.max_greedy_prefill_ms,
                        prompt.decode_tokens,
                        prompt
                            .mean_replay_decode_ms_per_token
                            .map(|value| format!("{value:.1}"))
                            .unwrap_or_else(|| "n/a".to_string()),
                        prompt
                            .mean_component_decode_ms_per_token
                            .map(|value| format!("{value:.1}"))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    if let Some(profile) = prompt.prefill_profile.as_ref() {
                        print_profile_summary(&prompt.name, "prefill", profile);
                    }
                    if let Some(profile) = prompt.greedy_prefill_profile.as_ref() {
                        print_profile_summary(&prompt.name, "greedy_prefill", profile);
                    }
                    if let Some(profile) = prompt.replay_decode_profile.as_ref() {
                        print_profile_summary(&prompt.name, "replay_decode", profile);
                    }
                    if let Some(profile) = prompt.component_decode_profile.as_ref() {
                        print_profile_summary(&prompt.name, "component_decode", profile);
                    }
                    if let Some(profile) = prompt.prefill_hal_profile.as_ref() {
                        print_hal_profile_summary(&prompt.name, "prefill", profile);
                    }
                    if let Some(profile) = prompt.greedy_prefill_hal_profile.as_ref() {
                        print_hal_profile_summary(&prompt.name, "greedy_prefill", profile);
                    }
                    if let Some(profile) = prompt.replay_decode_hal_profile.as_ref() {
                        print_hal_profile_summary(&prompt.name, "replay_decode", profile);
                    }
                    if let Some(profile) = prompt.component_decode_hal_profile.as_ref() {
                        print_hal_profile_summary(&prompt.name, "component_decode", profile);
                    }
                }
            }
        }
        _ => {}
    }
}

fn print_profile_summary(prompt_name: &str, phase: &str, profile: &MetalProfileReport) {
    println!(
        "PROFILE prompt={} phase={} total_calls={} native_calls={} host_calls={} total_ms={:.1} native_ms={:.1} host_ms={:.1}",
        prompt_name,
        phase,
        profile.total_calls,
        profile.native_calls,
        profile.host_calls,
        profile.total_ms,
        profile.native_ms,
        profile.host_ms,
    );
    for entry in profile.entries.iter().take(8) {
        println!(
            "PROFILE_OP prompt={} phase={} op={} path={} calls={} total_ms={:.1} mean_ms={:.3} max_ms={:.3}",
            prompt_name,
            phase,
            entry.op,
            entry.path,
            entry.calls,
            entry.total_ms,
            entry.mean_ms,
            entry.max_ms,
        );
    }
}

fn print_hal_profile_summary(prompt_name: &str, phase: &str, profile: &HalProfileReport) {
    println!(
        "HAL_PROFILE prompt={} phase={} total_calls={} total_ms={:.1} alloc_calls={} alloc_mb={:.1} free_calls={} h2d_mb={:.1} d2h_mb={:.1} d2d_mb={:.1} memset_mb={:.1} sync_calls={}",
        prompt_name,
        phase,
        profile.total_calls,
        profile.total_ms,
        profile.alloc_calls,
        bytes_to_mb(profile.alloc_bytes),
        profile.free_calls,
        bytes_to_mb(profile.h2d_bytes),
        bytes_to_mb(profile.d2h_bytes),
        bytes_to_mb(profile.d2d_bytes),
        bytes_to_mb(profile.memset_bytes),
        profile.sync_calls,
    );
    for entry in profile.entries.iter().take(8) {
        println!(
            "HAL_OP prompt={} phase={} op={} calls={} total_ms={:.1} mean_ms={:.3} max_ms={:.3} total_mb={:.1}",
            prompt_name,
            phase,
            entry.op,
            entry.calls,
            entry.total_ms,
            entry.mean_ms,
            entry.max_ms,
            bytes_to_mb(entry.total_bytes),
        );
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn write_report_json(path: &Path, report: &BughuntReport) -> Result<()> {
    let text = serde_json::to_string_pretty(report).context("serialize bughunt report JSON")?;
    fs::write(path, text).with_context(|| format!("write bughunt report {}", path.display()))
}

fn mean_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f32>()
        / len as f32
}

fn mean_square_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(lhs, rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mean_square(values: &[f32]) -> f32 {
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    sum_sq / values.len() as f32
}

fn max_abs_delta_details(lhs: &[f32], rhs: &[f32]) -> (usize, f32, f32, f32) {
    let mut best = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (index, (lhs, rhs)) in lhs.iter().copied().zip(rhs.iter().copied()).enumerate() {
        let delta = (lhs - rhs).abs();
        if delta > best.3 {
            best = (index, lhs, rhs, delta);
        }
    }
    best
}

fn top_abs_delta_dims(lhs: &[f32], rhs: &[f32], top_k: usize) -> Vec<TopDeltaDim> {
    let mut dims = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .enumerate()
        .map(|(index, (native, oracle))| TopDeltaDim {
            index,
            native,
            oracle,
            delta: (native - oracle).abs(),
        })
        .collect::<Vec<_>>();
    dims.sort_by(|lhs, rhs| {
        rhs.delta
            .total_cmp(&lhs.delta)
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    dims.truncate(top_k.min(dims.len()));
    dims
}

fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn encode_bf16_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

fn flatten_json_numbers(value: &Value, out: &mut Vec<f32>) {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_json_numbers(value, out);
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                out.push(value as f32);
            }
        }
        _ => {}
    }
}

fn flatten_bsh(value: &Value) -> Option<Vec<f32>> {
    value.as_array()?;
    let mut out = Vec::new();
    flatten_json_numbers(value, &mut out);
    Some(out)
}

fn flatten_json_vector(value: &Value) -> Option<Vec<f32>> {
    let array = value.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for elem in array {
        out.push(elem.as_f64()? as f32);
    }
    Some(out)
}

fn flatten_token_bsd(value: &Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = match position {
        Some(position) => batch.get(position)?,
        None => batch.last()?,
    };
    let mut out = Vec::new();
    flatten_json_numbers(token, &mut out);
    Some(out)
}

fn flatten_token_bhsd(value: &Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let mut out = Vec::new();
    for head in batch {
        let tokens = head.as_array()?;
        let token = match position {
            Some(position) => tokens.get(position)?,
            None => tokens.last()?,
        };
        flatten_json_numbers(token, &mut out);
    }
    Some(out)
}

fn extract_causal_conv_window_bsd(
    value: &Value,
    position: usize,
    dim: usize,
    kernel_size: usize,
) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    if position >= batch.len() {
        return None;
    }
    let pad = kernel_size.saturating_sub(1);
    let mut out = vec![0.0f32; dim * kernel_size];
    for tap in 0..kernel_size {
        let src_pos = position as isize - pad as isize + tap as isize;
        if src_pos < 0 {
            continue;
        }
        let row = flatten_json_vector(batch.get(src_pos as usize)?)?;
        if row.len() != dim {
            return None;
        }
        for channel in 0..dim {
            out[channel * kernel_size + tap] = row[channel];
        }
    }
    Some(out)
}

fn read_buffer_all_f32(buf: &GpuBuffer) -> Result<Vec<f32>> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("buffer D2H: {e}"))?;
    match buf.dtype() {
        ScalarType::BF16 => Ok(decode_bf16_le(&bytes)),
        ScalarType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        other => bail!("unsupported buffer dtype for debug read: {other:?}"),
    }
}

fn compute_qwen_rms_norm_from_hidden_row(
    hidden_row: &[f32],
    weight_buf: &GpuBuffer,
    eps: f32,
) -> Result<Vec<f32>> {
    let weights = read_buffer_all_f32(weight_buf)?;
    if weights.len() != hidden_row.len() {
        bail!(
            "norm weight length {} did not match hidden size {}",
            weights.len(),
            hidden_row.len()
        );
    }
    let inv_rms = 1.0f32 / (mean_square(hidden_row) + eps).sqrt();
    Ok(hidden_row
        .iter()
        .zip(weights.iter())
        .map(|(hidden, weight)| hidden * inv_rms * (weight + 1.0))
        .collect())
}

fn compute_qwen_logits_from_hidden_row(
    runtime: &QwenBughuntRuntime,
    hidden_row: &[f32],
) -> Result<Vec<f32>> {
    let hidden_dim = runtime.weights.config.hidden_size;
    if hidden_row.len() != hidden_dim {
        bail!(
            "hidden row length {} did not match hidden size {}",
            hidden_row.len(),
            hidden_dim
        );
    }
    let hidden_bf16 = encode_bf16_le(hidden_row);
    let hidden_gpu = GpuBuffer::from_host_bytes(
        runtime.ordinal,
        ScalarType::BF16,
        &[1, hidden_dim],
        &hidden_bf16,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row upload: {e}"))?;
    kernel_ffi::qwen_rms_norm_standalone_matvec_host_f32(
        runtime.ordinal,
        ScalarType::BF16,
        &hidden_gpu,
        &runtime.weights.norm_weight,
        runtime.weights.config.rms_norm_eps as f32,
        &runtime.weights.lm_head,
        hidden_dim,
        runtime.weights.config.vocab_size,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row logits: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn manifest_validation_rejects_empty_prompt_ids() {
        let bad = json!({
            "prompts": [{
                "name": "bad",
                "prompt_ids": [],
                "positions": [0],
                "thresholds": {
                    "prefill_logit_max_abs": 0.1,
                    "layer_hidden_max_abs": 0.1,
                    "restart_tail_logit_max_abs": 0.1
                }
            }]
        });
        let err = parse_prompt_manifest_str(&bad.to_string()).unwrap_err();
        assert!(err.to_string().contains("at least one prompt token id"));
    }

    #[test]
    fn manifest_validation_requires_thresholds() {
        let bad = json!({
            "prompts": [{
                "name": "bad",
                "prompt_ids": [1, 2],
                "positions": [0]
            }]
        });
        assert!(parse_prompt_manifest_str(&bad.to_string()).is_err());
    }

    #[test]
    fn gate_pass_uses_manifest_thresholds() {
        let thresholds = PromptThresholds {
            prefill_logit_max_abs: 0.2,
            layer_hidden_max_abs: 0.1,
            restart_tail_logit_max_abs: 0.3,
        };
        assert!(gate_pass(0.19, 0.09, &thresholds));
        assert!(!gate_pass(0.21, 0.09, &thresholds));
        assert!(!gate_pass(0.19, 0.11, &thresholds));
    }

    #[test]
    fn report_serialization_includes_localization_fields() {
        let report = BughuntReport {
            mode: "localize".to_string(),
            metadata: RunMetadata {
                mode: "localize".to_string(),
                model: "qwen3.5-0.8b".to_string(),
                backend: "metal".to_string(),
                device: 0,
                arch: "apple-m4".to_string(),
                model_dir: "/tmp/model".to_string(),
                oracle_device: "cpu".to_string(),
                commit_ish: Some("abc123".to_string()),
            },
            gate: None,
            localize: Some(LocalizeRunSection {
                pass: false,
                gate_prompt: PromptGateReport {
                    name: "code_prompt".to_string(),
                    notes: Some("code".to_string()),
                    pass: false,
                    thresholds: PromptThresholds {
                        prefill_logit_max_abs: 0.1,
                        layer_hidden_max_abs: 0.1,
                        restart_tail_logit_max_abs: 0.1,
                    },
                    prefill_logit_reference: "oracle_final_hidden_recomputed".to_string(),
                    prefill_logit_max_abs: 0.2,
                    prefill_logit_mean_abs: 0.02,
                    prefill_logit_mse: 0.01,
                    raw_oracle_prefill_logit_max_abs: 0.25,
                    gpu_reference_logit_max_abs: 0.19,
                    native_vs_gpu_reference_logit_max_abs: 0.03,
                    worst_checked_position: 15,
                    worst_layer: 18,
                    worst_layer_kind: "linear".to_string(),
                    worst_layer_delta: 0.12,
                    checked_positions: Vec::new(),
                    timings: vec![PhaseTimingReport {
                        phase: "native_prefill".to_string(),
                        elapsed_ms: 12.5,
                    }],
                },
                localization: LocalizationSummary {
                    prompt_name: "code_prompt".to_string(),
                    initial_suspicious_position: 15,
                    initial_suspicious_layer: 18,
                    initial_suspicious_layer_kind: "linear".to_string(),
                    per_layer_hidden_sweep: Vec::new(),
                    restart_layer_sweep: Vec::new(),
                    first_suspicious_restart_layer: Some(18),
                    restart_position_scan: Vec::new(),
                    worst_sampled_position: Some(15),
                    chosen_traced_layer: Some(18),
                    chosen_traced_layer_kind: Some("linear".to_string()),
                    traced_metrics: None,
                },
            }),
            dump: None,
            bench: None,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["mode"], "localize");
        assert_eq!(
            value["localize"]["localization"]["first_suspicious_restart_layer"],
            18
        );
    }

    #[test]
    fn bench_report_serialization_includes_decode_and_profile_fields() {
        let report = BughuntReport {
            mode: "bench".to_string(),
            metadata: RunMetadata {
                mode: "bench".to_string(),
                model: "qwen3.5-0.8b".to_string(),
                backend: "metal".to_string(),
                device: 0,
                arch: "apple-m4".to_string(),
                model_dir: "/tmp/model".to_string(),
                oracle_device: "cpu".to_string(),
                commit_ish: Some("abc123".to_string()),
            },
            gate: None,
            localize: None,
            dump: None,
            bench: Some(BenchRunSection {
                pass: true,
                prompt_results: vec![BenchPromptReport {
                    name: "hello_world".to_string(),
                    notes: Some("smoke".to_string()),
                    prompt_len: 2,
                    warmup_iterations: 0,
                    iterations: 1,
                    decode_tokens: 1,
                    native_prefill_ms: vec![3.0],
                    min_native_prefill_ms: 3.0,
                    max_native_prefill_ms: 3.0,
                    mean_native_prefill_ms: 3.0,
                    greedy_prefill_ms: vec![1.0],
                    min_greedy_prefill_ms: 1.0,
                    max_greedy_prefill_ms: 1.0,
                    mean_greedy_prefill_ms: 1.0,
                    replay_decode_ms: vec![4.0],
                    min_replay_decode_ms: Some(4.0),
                    max_replay_decode_ms: Some(4.0),
                    mean_replay_decode_ms: Some(4.0),
                    mean_replay_decode_ms_per_token: Some(4.0),
                    component_decode_ms: Some(vec![2.0]),
                    min_component_decode_ms: Some(2.0),
                    max_component_decode_ms: Some(2.0),
                    mean_component_decode_ms: Some(2.0),
                    mean_component_decode_ms_per_token: Some(2.0),
                    prefill_profile: Some(MetalProfileReport {
                        total_calls: 2,
                        native_calls: 1,
                        host_calls: 1,
                        total_ms: 1.5,
                        native_ms: 1.0,
                        host_ms: 0.5,
                        entries: vec![MetalProfileOpReport {
                            op: "matmul_rhs_transposed".to_string(),
                            path: "native".to_string(),
                            calls: 1,
                            total_ms: 1.0,
                            mean_ms: 1.0,
                            max_ms: 1.0,
                        }],
                    }),
                    greedy_prefill_profile: None,
                    replay_decode_profile: None,
                    component_decode_profile: None,
                    prefill_hal_profile: Some(HalProfileReport {
                        total_calls: 3,
                        total_ms: 2.0,
                        alloc_calls: 1,
                        alloc_bytes: 1024,
                        free_calls: 1,
                        h2d_bytes: 0,
                        d2h_bytes: 2048,
                        d2d_bytes: 0,
                        memset_bytes: 1024,
                        sync_calls: 1,
                        entries: vec![HalProfileOpReport {
                            op: "alloc".to_string(),
                            calls: 1,
                            total_ms: 1.0,
                            mean_ms: 1.0,
                            max_ms: 1.0,
                            total_bytes: 1024,
                        }],
                    }),
                    greedy_prefill_hal_profile: None,
                    replay_decode_hal_profile: None,
                    component_decode_hal_profile: None,
                }],
            }),
        };
        let value = serde_json::to_value(&report).unwrap();
        let prompt = &value["bench"]["prompt_results"][0];
        assert_eq!(prompt["decode_tokens"], 1);
        assert_eq!(prompt["mean_greedy_prefill_ms"], 1.0);
        assert_eq!(prompt["mean_replay_decode_ms_per_token"], 4.0);
        assert_eq!(prompt["mean_component_decode_ms_per_token"], 2.0);
        assert_eq!(prompt["prefill_profile"]["native_calls"], 1);
        assert_eq!(
            prompt["prefill_profile"]["entries"][0]["op"],
            "matmul_rhs_transposed"
        );
        assert_eq!(prompt["prefill_hal_profile"]["alloc_calls"], 1);
        assert_eq!(prompt["prefill_hal_profile"]["d2h_bytes"], 2048);
    }

    #[test]
    fn metal_profile_report_preserves_dispatch_summary() {
        let report = metal_profile_report(kernel_ffi::prefill_ffi::MetalProfileSnapshot {
            total_calls: 3,
            native_calls: 2,
            host_calls: 1,
            total_ms: 4.0,
            native_ms: 3.0,
            host_ms: 1.0,
            entries: vec![kernel_ffi::prefill_ffi::MetalProfileEntry {
                op: "cast".to_string(),
                path: "native".to_string(),
                calls: 2,
                total_ms: 3.0,
                max_ms: 2.0,
            }],
        });
        assert_eq!(report.total_calls, 3);
        assert_eq!(report.native_calls, 2);
        assert_eq!(report.host_calls, 1);
        assert_eq!(report.entries[0].mean_ms, 1.5);
    }

    #[test]
    fn hal_profile_report_preserves_memory_summary() {
        let report = hal_profile_report(gpu_hal::HalProfileSnapshot {
            total_calls: 2,
            total_ms: 5.0,
            alloc_calls: 1,
            alloc_bytes: 4096,
            free_calls: 1,
            h2d_bytes: 128,
            d2h_bytes: 256,
            d2d_bytes: 512,
            memset_bytes: 1024,
            sync_calls: 1,
            entries: vec![gpu_hal::HalProfileEntry {
                op: "alloc".to_string(),
                calls: 1,
                total_ms: 4.0,
                max_ms: 4.0,
                total_bytes: 4096,
            }],
        });
        assert_eq!(report.alloc_calls, 1);
        assert_eq!(report.alloc_bytes, 4096);
        assert_eq!(report.entries[0].mean_ms, 4.0);
    }

    #[test]
    fn deepest_failing_restart_layer_prefers_latest_failing_boundary() {
        let reports = vec![
            RestartSweepReport {
                source_layer: 0,
                start_layer: 1,
                failing: true,
                tail_logit_max_abs: 0.2,
                tail_logit_mean_abs: 0.02,
                selected_position: 0,
                selected_position_worst_layer: 1,
                selected_position_worst_layer_delta: 0.1,
            },
            RestartSweepReport {
                source_layer: 17,
                start_layer: 18,
                failing: true,
                tail_logit_max_abs: 0.3,
                tail_logit_mean_abs: 0.03,
                selected_position: 15,
                selected_position_worst_layer: 18,
                selected_position_worst_layer_delta: 0.2,
            },
            RestartSweepReport {
                source_layer: 18,
                start_layer: 19,
                failing: false,
                tail_logit_max_abs: 0.01,
                tail_logit_mean_abs: 0.001,
                selected_position: 15,
                selected_position_worst_layer: 19,
                selected_position_worst_layer_delta: 0.02,
            },
        ];
        assert_eq!(choose_deepest_failing_restart_layer(&reports), Some(18));
    }
}
