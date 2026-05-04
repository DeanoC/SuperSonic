use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use super::profile::{print_hal_profile_summary, print_profile_summary};
use super::report::BughuntReport;

pub(crate) fn print_report_summary(report: &BughuntReport) {
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
        "decode_gate" => {
            if let Some(decode_gate) = report.decode_gate.as_ref() {
                println!(
                    "mode=decode_gate backend={} prompts={} pass={}",
                    report.metadata.backend,
                    decode_gate.prompt_results.len(),
                    decode_gate.pass
                );
                for prompt in &decode_gate.prompt_results {
                    println!(
                        "{} prompt={} tokens={} first_mismatch={} max_replay_logit_abs={:.4} oracle_tokens={:?} replay_tokens={:?} component_tokens={}",
                        if prompt.pass { "PASS" } else { "FAIL" },
                        prompt.name,
                        prompt.decode_tokens,
                        prompt
                            .first_mismatch_step
                            .map(|step| step.to_string())
                            .unwrap_or_else(|| "n/a".to_string()),
                        prompt.max_replay_logit_abs,
                        prompt.oracle_tokens,
                        prompt.replay_tokens,
                        prompt
                            .component_tokens
                            .as_ref()
                            .map(|tokens| format!("{tokens:?}"))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    if let Some(step) = prompt
                        .steps
                        .iter()
                        .find(|step| !step.token_match_replay || !step.token_match_component)
                    {
                        println!(
                            "first_bad_step={} oracle={} replay={} component={} replay_logit_max_abs={:.4}",
                            step.step,
                            step.oracle_token,
                            step.replay_token,
                            step.component_token
                                .map(|token| token.to_string())
                                .unwrap_or_else(|| "n/a".to_string()),
                            step.replay_logit_max_abs,
                        );
                    }
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

pub(crate) fn write_report_json(path: &Path, report: &BughuntReport) -> Result<()> {
    let text = serde_json::to_string_pretty(report).context("serialize bughunt report JSON")?;
    fs::write(path, text).with_context(|| format!("write bughunt report {}", path.display()))
}
