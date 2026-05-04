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

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(mode: &str) -> RunMetadata {
        RunMetadata {
            mode: mode.to_string(),
            model: "qwen3.5-0.8b".to_string(),
            backend: "metal".to_string(),
            device: 0,
            arch: "apple-m4".to_string(),
            model_dir: "/tmp/model".to_string(),
            oracle_device: "cpu".to_string(),
            commit_ish: Some("abc123".to_string()),
        }
    }

    #[test]
    fn report_serialization_includes_localization_fields() {
        let report = BughuntReport {
            mode: "localize".to_string(),
            metadata: metadata("localize"),
            gate: None,
            decode_gate: None,
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
            metadata: metadata("bench"),
            gate: None,
            decode_gate: None,
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
    fn decode_gate_report_serialization_includes_token_comparisons() {
        let report = BughuntReport {
            mode: "decode_gate".to_string(),
            metadata: metadata("decode_gate"),
            gate: None,
            decode_gate: Some(DecodeGateRunSection {
                pass: false,
                prompt_results: vec![DecodeGatePromptReport {
                    name: "code_prompt".to_string(),
                    notes: Some("code".to_string()),
                    pass: false,
                    prompt_len: 31,
                    decode_tokens: 2,
                    oracle_tokens: vec![271, 16],
                    replay_tokens: vec![271, 17],
                    component_tokens: Some(vec![271, 16]),
                    max_replay_logit_abs: 0.2,
                    mean_replay_logit_abs: 0.05,
                    first_mismatch_step: Some(1),
                    steps: vec![DecodeStepGateReport {
                        step: 1,
                        oracle_token: 16,
                        replay_token: 17,
                        component_token: Some(16),
                        replay_logit_reference: "oracle_decode_logits".to_string(),
                        replay_logit_max_abs: 0.2,
                        replay_logit_mean_abs: 0.05,
                        token_match_replay: false,
                        token_match_component: true,
                    }],
                    timings: vec![PhaseTimingReport {
                        phase: "oracle".to_string(),
                        elapsed_ms: 10.0,
                    }],
                }],
            }),
            localize: None,
            dump: None,
            bench: None,
        };
        let value = serde_json::to_value(&report).unwrap();
        let prompt = &value["decode_gate"]["prompt_results"][0];
        assert_eq!(value["mode"], "decode_gate");
        assert_eq!(prompt["first_mismatch_step"], 1);
        assert_eq!(prompt["oracle_tokens"][1], 16);
        assert_eq!(prompt["replay_tokens"][1], 17);
        assert_eq!(prompt["component_tokens"][1], 16);
    }
}
