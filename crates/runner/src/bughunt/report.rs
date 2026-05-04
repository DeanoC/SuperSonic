use serde::{Deserialize, Serialize};

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
pub struct DecodeStepGateReport {
    pub step: usize,
    pub oracle_token: u32,
    pub replay_token: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_token: Option<u32>,
    pub replay_logit_reference: String,
    pub replay_logit_max_abs: f32,
    pub replay_logit_mean_abs: f32,
    pub token_match_replay: bool,
    pub token_match_component: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeGatePromptReport {
    pub name: String,
    pub notes: Option<String>,
    pub pass: bool,
    pub prompt_len: usize,
    pub decode_tokens: usize,
    pub oracle_tokens: Vec<u32>,
    pub replay_tokens: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_tokens: Option<Vec<u32>>,
    pub max_replay_logit_abs: f32,
    pub mean_replay_logit_abs: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_mismatch_step: Option<usize>,
    pub steps: Vec<DecodeStepGateReport>,
    pub timings: Vec<PhaseTimingReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeGateRunSection {
    pub pass: bool,
    pub prompt_results: Vec<DecodeGatePromptReport>,
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
    pub decode_gate: Option<DecodeGateRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub localize: Option<LocalizeRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dump: Option<DumpRunSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench: Option<BenchRunSection>,
}

impl BughuntReport {
    pub fn exit_code(&self) -> i32 {
        let pass = match self.mode.as_str() {
            "gate" => self
                .gate
                .as_ref()
                .map(|section| section.pass)
                .unwrap_or(false),
            "decode_gate" => self
                .decode_gate
                .as_ref()
                .map(|section| section.pass)
                .unwrap_or(false),
            "localize" => self
                .localize
                .as_ref()
                .map(|section| section.pass)
                .unwrap_or(false),
            "dump" => self
                .dump
                .as_ref()
                .map(|section| section.pass)
                .unwrap_or(false),
            "bench" => self
                .bench
                .as_ref()
                .map(|section| section.pass)
                .unwrap_or(false),
            _ => false,
        };
        if pass {
            0
        } else {
            1
        }
    }
}
