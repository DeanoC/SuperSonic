use std::time::Instant;

use anyhow::{bail, Context, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use qwen35::state::ModelState;

use crate::decode_engine::{decode_f32_le, DecodeEngine};
use crate::oracle;
use crate::prefill_engine;
use crate::validate;

mod args;
mod manifest;
mod output;
mod profile;
mod report;
mod runtime;
mod util;
pub use args::*;
use args::validate_args;
use manifest::load_prompt_manifest;
use output::{print_report_summary, write_report_json};
use profile::{
    collect_profiles, reset_profiles, snapshot_profiles, ProfileGuard, ProfileReports,
};
pub use report::*;
use runtime::QwenBughuntRuntime;
use util::{
    decode_bf16_le, encode_bf16_le, extract_causal_conv_window_bsd, flatten_bsh,
    flatten_token_bsd, max_abs_delta_details, mean_abs_delta, mean_square, mean_square_delta,
    read_buffer_all_f32, top_abs_delta_dims,
};

#[derive(Debug, Clone)]
struct PromptGateAnalysis {
    report: PromptGateReport,
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
                decode_gate: None,
                localize: None,
                dump: None,
                bench: None,
            }
        }
        BughuntMode::DecodeGate => {
            let section = run_decode_gate_mode(
                &runtime,
                &manifest,
                args.prompt.as_deref(),
                args.bench_decode_tokens,
            )?;
            BughuntReport {
                mode: args.mode.as_str().to_string(),
                metadata,
                gate: None,
                decode_gate: Some(section),
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
                decode_gate: None,
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
                decode_gate: None,
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
                decode_gate: None,
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

fn run_decode_gate_mode(
    runtime: &QwenBughuntRuntime,
    manifest: &PromptManifest,
    selected_prompt: Option<&str>,
    decode_tokens: usize,
) -> Result<DecodeGateRunSection> {
    let prompts = select_prompts(manifest, selected_prompt)?;
    let mut prompt_results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        eprintln!(
            "[bughunt] decode_gate prompt={} decode_tokens={} start",
            prompt.name, decode_tokens
        );
        let report = analyze_decode_gate_prompt(runtime, prompt, decode_tokens)?;
        eprintln!(
            "[bughunt] decode_gate prompt={} done pass={} first_mismatch={} max_replay_logit_abs={:.4}",
            prompt.name,
            report.pass,
            report
                .first_mismatch_step
                .map(|step| step.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            report.max_replay_logit_abs
        );
        prompt_results.push(report);
    }
    let pass = prompt_results.iter().all(|report| report.pass);
    Ok(DecodeGateRunSection {
        pass,
        prompt_results,
    })
}

fn analyze_decode_gate_prompt(
    runtime: &QwenBughuntRuntime,
    prompt: &PromptManifestEntry,
    decode_tokens: usize,
) -> Result<DecodeGatePromptReport> {
    let prompt_start = Instant::now();
    let mut timings = Vec::new();

    eprintln!("[bughunt] decode_gate prompt={} oracle", prompt.name);
    let phase_start = Instant::now();
    let oracle_output = oracle::run_oracle(
        &runtime.oracle_script,
        runtime.model_variant.hf_model_id(),
        &prompt.prompt_ids,
        decode_tokens,
        "bf16",
        &runtime.oracle_device,
        false,
        false,
        None,
        None,
    )?;
    timings.push(phase_timing("oracle", phase_start));

    if oracle_output.generated_token_ids.len() < decode_tokens {
        bail!(
            "oracle returned {} generated tokens, expected {}",
            oracle_output.generated_token_ids.len(),
            decode_tokens
        );
    }
    if decode_tokens > 1 && oracle_output.decode_logits.len() + 1 < decode_tokens {
        bail!(
            "oracle returned {} decode logit rows, expected at least {}",
            oracle_output.decode_logits.len(),
            decode_tokens - 1
        );
    }

    eprintln!("[bughunt] decode_gate prompt={} replay_forced", prompt.name);
    let phase_start = Instant::now();
    let mut replay_tokens = Vec::with_capacity(decode_tokens);
    let mut steps = Vec::with_capacity(decode_tokens);
    for step in 0..decode_tokens {
        let mut forced_history = prompt.prompt_ids.clone();
        forced_history.extend_from_slice(&oracle_output.generated_token_ids[..step]);
        let native_prefill = run_native_prefill(runtime, &forced_history)
            .with_context(|| format!("decode_gate replay forced step {step}"))?;
        gpu_hal::sync(runtime.ordinal)
            .with_context(|| format!("decode_gate replay forced sync step {step}"))?;
        let replay_token = DecodeEngine::greedy_sample(&native_prefill.logits);
        replay_tokens.push(replay_token);
        let (reference, reference_name) = if step == 0 {
            (&oracle_output.prefill_logits, "oracle_prefill_logits")
        } else {
            (
                oracle_output.decode_logits.get(step - 1).ok_or_else(|| {
                    anyhow::anyhow!("missing oracle decode logits for step {step}")
                })?,
                "oracle_decode_logits",
            )
        };
        let replay_logit_max_abs = validate::max_abs_delta(&native_prefill.logits, reference);
        let replay_logit_mean_abs = mean_abs_delta(&native_prefill.logits, reference);
        let oracle_token = oracle_output.generated_token_ids[step];
        steps.push(DecodeStepGateReport {
            step,
            oracle_token,
            replay_token,
            component_token: None,
            replay_logit_reference: reference_name.to_string(),
            replay_logit_max_abs,
            replay_logit_mean_abs,
            token_match_replay: replay_token == oracle_token,
            token_match_component: runtime.backend != Backend::Metal,
        });
    }
    timings.push(phase_timing("replay_forced", phase_start));

    let component_tokens = if runtime.backend == Backend::Metal {
        eprintln!("[bughunt] decode_gate prompt={} component", prompt.name);
        let phase_start = Instant::now();
        let tokens =
            run_component_decode_token_sequence(runtime, &prompt.prompt_ids, decode_tokens)
                .with_context(|| format!("decode_gate component prompt {}", prompt.name))?;
        for (step, token) in tokens.iter().copied().enumerate() {
            if let Some(step_report) = steps.get_mut(step) {
                step_report.component_token = Some(token);
                step_report.token_match_component =
                    token == oracle_output.generated_token_ids[step];
            }
        }
        timings.push(phase_timing("component", phase_start));
        Some(tokens)
    } else {
        None
    };

    let max_replay_logit_abs = steps
        .iter()
        .map(|step| step.replay_logit_max_abs)
        .fold(0.0_f32, f32::max);
    let mean_replay_logit_abs = if steps.is_empty() {
        0.0
    } else {
        steps
            .iter()
            .map(|step| step.replay_logit_mean_abs)
            .sum::<f32>()
            / steps.len() as f32
    };
    let first_mismatch_step = steps
        .iter()
        .find(|step| !step.token_match_replay || !step.token_match_component)
        .map(|step| step.step);
    let pass = first_mismatch_step.is_none();
    timings.push(phase_timing("total", prompt_start));

    Ok(DecodeGatePromptReport {
        name: prompt.name.clone(),
        notes: prompt.notes.clone(),
        pass,
        prompt_len: prompt.prompt_ids.len(),
        decode_tokens,
        oracle_tokens: oracle_output.generated_token_ids[..decode_tokens].to_vec(),
        replay_tokens,
        component_tokens,
        max_replay_logit_abs,
        mean_replay_logit_abs,
        first_mismatch_step,
        steps,
        timings,
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
    reset_profiles();
    for step in 0..decode_tokens {
        history.push(token);
        token = run_native_prefill_greedy_token_with_state(runtime, &mut state, &history)
            .with_context(|| format!("bench profile replay decode step {step}"))?;
        gpu_hal::sync(runtime.ordinal)
            .with_context(|| format!("bench profile replay decode sync step {step}"))?;
    }
    snapshot_profiles()
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
    reset_profiles();
    run_component_decode_steps(engine, token, prompt_ids.len(), decode_tokens)
        .context("bench profile component decode steps")?;
    snapshot_profiles()
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

fn run_component_decode_token_sequence(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<Vec<u32>> {
    let mut engine = runtime
        .new_component_decode_engine(prompt_ids.len() + decode_tokens)
        .context("decode gate component decode engine")?;
    let mut token = engine
        .rebuild_prefill_state_greedy_token(prompt_ids)
        .context("decode gate component initial prefill")?;
    gpu_hal::sync(runtime.ordinal).context("decode gate component initial sync")?;

    let mut tokens = Vec::with_capacity(decode_tokens);
    tokens.push(token);
    for step in 1..decode_tokens {
        let (next, _) = engine
            .decode_step_metal_component_greedy(token, prompt_ids.len() + step - 1)
            .with_context(|| format!("decode gate component step {step}"))?;
        token = next;
        tokens.push(token);
    }
    Ok(tokens)
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
    _runtime: &QwenBughuntRuntime,
    layer: usize,
    _position: usize,
    _native: &prefill_engine::PrefillResult,
    _trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<StageMetricReport>> {
    bail!("full-attention stage metrics for layer {layer} require removed native debug taps")
}

fn build_mlp_stage_metrics(
    _runtime: &QwenBughuntRuntime,
    layer: usize,
    _position: usize,
    _native: &prefill_engine::PrefillResult,
    _trace: &oracle::Qwen35TraceOutput,
) -> Result<Vec<StageMetricReport>> {
    bail!("MLP stage metrics for layer {layer} require removed native debug taps")
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
    )
}

fn run_native_prefill_greedy_token_with_state(
    runtime: &QwenBughuntRuntime,
    state: &mut ModelState,
    prompt_ids: &[u32],
) -> Result<u32> {
    state.reset_for_prefill_reuse();
    let result = prefill_engine::prefill(
        &runtime.weights,
        state,
        &runtime.rotary,
        prompt_ids,
        runtime.ordinal,
        runtime.kv_chunk_size,
        runtime.prefill_chunk_size,
        false,
        runtime.use_4b_kernel,
        false,
        None,
    )?;
    Ok(DecodeEngine::greedy_sample(&result.logits))
}

fn run_native_prefill_with_trace(
    runtime: &QwenBughuntRuntime,
    prompt_ids: &[u32],
    trace_position: Option<usize>,
    debug_layer: Option<usize>,
    debug_kind: Option<BughuntLayerKind>,
) -> Result<prefill_engine::PrefillResult> {
    let _ = trace_position;
    let mut state = ModelState::new(&runtime.weights.config, runtime.ordinal)
        .map_err(|e| anyhow::anyhow!("native traced prefill model state init: {e}"))?;
    let (debug_linear_layer, debug_full_layer, debug_mlp_layer) =
        debug_layer_flags(debug_layer, debug_kind);
    if debug_full_layer.is_some() || debug_mlp_layer.is_some() {
        bail!("native full-attention/MLP debug traces are not available in the current prefill API");
    }
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
        true,
        debug_linear_layer,
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
    let _ = (
        runtime,
        hidden_bf16,
        start_layer,
        trace_position,
        debug_layer,
        debug_kind,
    );
    bail!("tail replay tracing is not available in the current prefill API")
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
