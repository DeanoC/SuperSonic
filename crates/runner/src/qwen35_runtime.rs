use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use crate::bakes::ensure_hf_metadata_present;
use crate::decode_engine::DecodeEngine;
use crate::qwen35_alt_runtime::run_qwen35_alt_runtime_if_requested;
use crate::qwen35_decode_loop::{run_qwen35_decode_loop, Qwen35DecodeLoop};
use crate::qwen35_decode_modes::{report_qwen35_decode_modes, resolve_qwen35_decode_modes};
use crate::qwen35_decode_report::{
    emit_qwen35_decode_report, emit_qwen35_last_logits_if_requested, Qwen35DecodeReport,
};
use crate::qwen35_engine_setup::{
    install_qwen35_launch_preset, load_qwen35_engine, Qwen35EngineSetup,
};
use crate::qwen35_prefill::{run_qwen35_prefill, HostLmHeadRescorer, Qwen35Prefill};
use crate::qwen35_startup::{
    load_qwen35_startup, validate_qwen35_startup, Qwen35Policy, Qwen35Startup,
};
use crate::qwen35_validation::{
    resolve_qwen35_oracle_context, run_qwen35_oracle_validation,
    trace_qwen35_oracle_prefill_layer_if_requested,
};
use crate::qwen35_virtual_kv::report_qwen35_virtual_kv_after_prefill;
use crate::qwen35_vram::check_qwen35_vram;
use crate::registry::{Backend, FamilyParams, GpuArch, ModelVariant, RegistryEntry};
use crate::Cli;

pub(crate) fn run_qwen35(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    backend: Backend,
    gpu_arch: GpuArch,
    ordinal: usize,
    total_vram: u64,
    q4km_like: bool,
) -> Result<()> {
    if run_qwen35_alt_runtime_if_requested(cli, model_variant, entry, ordinal, total_vram)? {
        return Ok(());
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Qwen3Moe(_) => unreachable!("qwen3-moe handled above"),
        FamilyParams::Qwen36Moe(_) => unreachable!("qwen3.6-moe handled above"),
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
        FamilyParams::Phi4(_) => unreachable!("phi4 handled above"),
        FamilyParams::Llama31(_) => unreachable!("llama3.1 handled above"),
    };
    let host_lm_head_rescorer = HostLmHeadRescorer::from_model_dir(&cli.model_dir)?;

    install_qwen35_launch_preset(entry);

    let Qwen35Policy {
        trace_kv_cache_enabled,
    } = validate_qwen35_startup(
        &cli,
        &model_variant,
        params,
        backend,
        &entry.arch,
        q4km_like,
    )?;

    // If --model-dir is pristine (no config.json), fetch a bake first so the
    // downloader can populate HF metadata before we try to read it.
    let bootstrap_downloaded = ensure_hf_metadata_present(&cli, &model_variant)?;

    let Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
        flm_source,
    } = load_qwen35_startup(&cli)?;
    check_qwen35_vram(
        &cli,
        &text_config,
        &entry.vram,
        context_tokens,
        params.kv_chunk_size,
        total_vram,
    )?;

    let gpu_validate_enabled = cli.gpu_validate && cli.batch_size == 1;
    if cli.gpu_validate && cli.batch_size > 1 {
        eprintln!("[gpu-validate] GPU oracle disabled for batch_size > 1");
    }
    let Qwen35EngineSetup {
        mut engine,
        use_4b_kernel,
        cuda_08b_hero_enabled,
        allow_host_lm_head_rescore,
    } = load_qwen35_engine(
        &cli,
        &model_variant,
        &text_config,
        flm_source.as_ref(),
        params,
        backend,
        gpu_arch,
        ordinal,
        bootstrap_downloaded,
        q4km_like,
        context_tokens,
    )?;

    // Teacher-forced scoring: do not proceed to normal prefill+decode.
    if cli.teacher_forced {
        return run_qwen35_teacher_forced(cli, model_variant, &mut engine, &prompt_ids);
    }

    let oracle_context = resolve_qwen35_oracle_context(cli, backend, ordinal, model_variant);
    let Qwen35Prefill {
        logits: prefill_logits,
        native_trace: native_prefill_trace,
        next_token,
    } = run_qwen35_prefill(
        &cli,
        &mut engine,
        &prompt_ids,
        &oracle_context.model_id,
        &oracle_context.device,
        oracle_context.fp8_oracle_dir.as_deref(),
        host_lm_head_rescorer.as_ref(),
        allow_host_lm_head_rescore,
    )?;

    emit_qwen35_last_logits_if_requested(cli, &prefill_logits);

    report_qwen35_virtual_kv_after_prefill(&mut engine)?;

    // Optionally run oracle for validation
    let oracle_output = run_qwen35_oracle_validation(
        &cli,
        &engine,
        &text_config,
        &prompt_ids,
        &oracle_context.model_id,
        &oracle_context.device,
        oracle_context.fp8_oracle_dir.as_deref(),
        &prefill_logits,
        native_prefill_trace.as_ref(),
        next_token,
    )?;

    trace_qwen35_oracle_prefill_layer_if_requested(
        cli,
        &mut engine,
        &prompt_ids,
        &oracle_context,
        oracle_output.as_ref(),
    )?;

    // Replicate prefill state to batch items if batch_size > 1
    if cli.batch_size > 1 {
        eprintln!(
            "[batch] replicating prefill state to {} sequences",
            cli.batch_size
        );
        engine.replicate_state_to_batch()?;
    }

    if gpu_validate_enabled {
        eprintln!(
            "[gpu-validate] replaying full token history through GPU prefill for reference..."
        );
    }
    let decode_modes = resolve_qwen35_decode_modes(
        &cli,
        backend,
        &model_variant,
        use_4b_kernel,
        gpu_validate_enabled,
        oracle_output.is_some(),
        cuda_08b_hero_enabled,
        !engine.weights().gqh_headers.is_empty(),
    );
    report_qwen35_decode_modes(
        &cli,
        &decode_modes,
        use_4b_kernel,
        cuda_08b_hero_enabled,
        !engine.weights().gqh_headers.is_empty(),
    );

    let mut eos_ids = text_config.eos_token_ids();
    if cli.chat {
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
    }
    let prefill_normed = native_prefill_trace
        .as_ref()
        .and_then(|(norm, ..)| norm.clone());
    let decode_output = run_qwen35_decode_loop(Qwen35DecodeLoop {
        cli,
        engine: &mut engine,
        decode_modes: &decode_modes,
        prompt_ids: &prompt_ids,
        eos_ids: &eos_ids,
        next_token,
        oracle_output: oracle_output.as_ref(),
        host_lm_head_rescorer: host_lm_head_rescorer.as_ref(),
        allow_host_lm_head_rescore,
        trace_kv_cache_enabled,
        gpu_validate_enabled,
        cuda_08b_hero_enabled,
        ordinal,
        kv_chunk_size: params.kv_chunk_size,
        use_4b_kernel,
        prefill_normed,
    })?;

    emit_qwen35_decode_report(Qwen35DecodeReport {
        tokenizer: &tokenizer,
        prompt_ids: &prompt_ids,
        generated_ids: &decode_output.generated_ids,
        emit_generated_json: cli.emit_generated_json,
        decode_ms: decode_output.decode_ms,
        max_delta: decode_output.max_delta,
        gpu_max_delta: decode_output.gpu_max_delta,
        batch_size: cli.batch_size,
        emit_stage_timings: cli.emit_stage_timings,
        native_decode_timings: &decode_output.native_decode_timings,
        native_decode_timing_steps: decode_output.native_decode_timing_steps,
    })?;

    Ok(())
}

/// Compute the negative log-likelihood of `target_token` under the distribution
/// implied by `logits` (numerically stable log-softmax).
fn qwen35_target_nll(logits: &[f32], target_token: u32) -> Result<f64> {
    let target_idx = target_token as usize;
    let target_logit = logits.get(target_idx).ok_or_else(|| {
        anyhow!(
            "target token {target_token} outside logits len {}",
            logits.len()
        )
    })?;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp = logits
        .iter()
        .map(|&x| ((x as f64) - max_logit).exp())
        .sum::<f64>();
    Ok(max_logit + sum_exp.ln() - *target_logit as f64)
}

/// Score `prompt_ids` with teacher forcing: run GPU prefill over the full
/// sequence, then decode step-by-step while feeding the *true* next token
/// rather than the model's argmax.  Emits a human-readable `[teacher_forced]`
/// line to stderr and a machine-parseable `[teacher_forced_json]` line to
/// stdout with the same fields as the Llama 3.1 reference scorer.
pub(crate) fn run_qwen35_teacher_forced(
    cli: &Cli,
    model_variant: &ModelVariant,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
) -> Result<()> {
    if prompt_ids.len() < 2 {
        bail!("--teacher-forced requires at least 2 prompt tokens");
    }
    if cli.validate || cli.gpu_validate {
        bail!("--teacher-forced does not currently support --validate or --gpu-validate");
    }
    if cli.trace_prefill_layers
        || cli.trace_oracle_prefill_layer.is_some()
        || cli.certified_kv_trace_layer.is_some()
        || cli.certified_kv_trace_all
        || cli.certified_kv_shadow_validate
    {
        bail!("--teacher-forced does not support trace/debug validation flags yet");
    }

    let score_start = Instant::now();
    let prefill_start = Instant::now();

    // Use GPU prefill scoring (full-prompt prefill with per-token NLL) unless
    // the caller explicitly requested the legacy decode-step-by-step path,
    // OR the backend is non-CUDA — `cuda_target_nll_bf16` is a CUDA-only kernel
    // (no HIP/Metal port) so HIP/Metal fall back to the decode-step scorer.
    let cuda_only_scorer_available = engine.backend() == gpu_hal::Backend::Cuda;
    let use_gpu_prefill_scoring = !cli.teacher_forced_decode_step && cuda_only_scorer_available;

    let mut total_nll = 0.0f64;
    let mut scored_tokens = 0usize;
    let decode_loop_start;
    let mut logits;

    if use_gpu_prefill_scoring {
        // Prefill the full prompt and score every position except position 0
        // (there is no "previous token" to predict position 0 from).
        let targets = &prompt_ids[1..];
        let prefill = engine.prefill_native_with_target_nll(prompt_ids, 0, targets)?;
        let scored = prefill
            .target_nll
            .as_ref()
            .ok_or_else(|| anyhow!("scored prefill did not return target NLL"))?;
        total_nll += scored.total_nll;
        scored_tokens += scored.scored_tokens;
        logits = prefill.logits;
        decode_loop_start = prompt_ids.len();
    } else {
        // Legacy path: prefill on a single token, then decode step by step
        // scoring each position from its logits.
        logits = engine.prefill_native(&prompt_ids[..1])?;
        decode_loop_start = 1;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut decode_steps = 0usize;

    // decode_loop_start is either prompt_ids.len() (gpu prefill path, nothing
    // left to decode) or 1 (legacy path, must step through remaining tokens).
    for target_idx in decode_loop_start..prompt_ids.len() {
        total_nll += qwen35_target_nll(&logits, prompt_ids[target_idx])?;
        scored_tokens += 1;
        if target_idx + 1 < prompt_ids.len() {
            let input_token = prompt_ids[target_idx];
            let pos = target_idx;
            logits = engine.decode_step(input_token, pos)?;
            decode_steps += 1;
        }
    }

    if scored_tokens == 0 {
        bail!("--teacher-forced scored zero tokens");
    }

    let total_ms = score_start.elapsed().as_secs_f64() * 1000.0;
    let avg_nll = total_nll / scored_tokens as f64;
    let perplexity = avg_nll.exp();
    let bits_per_token = avg_nll / std::f64::consts::LN_2;
    let ms_per_token = total_ms / scored_tokens as f64;
    eprintln!(
        "[teacher_forced] tokens={} scored_tokens={} decode_steps={} nll={:.6} avg_nll={:.6} ppl={:.6} bpt={:.6} prefill_ms={:.1} total_ms={:.1} ms_per_token={:.2}",
        prompt_ids.len(),
        scored_tokens,
        decode_steps,
        total_nll,
        avg_nll,
        perplexity,
        bits_per_token,
        prefill_ms,
        total_ms,
        ms_per_token,
    );
    println!(
        "[teacher_forced_json] {}",
        serde_json::to_string(&serde_json::json!({
            "backend": backend_label(engine.backend()),
            "model": model_variant.to_string(),
            "mode": "dense",
            "teacher_forced_scoring": if use_gpu_prefill_scoring { "gpu_prefill_target_nll" } else { "decode_step_logits" },
            "prompt_tokens": prompt_ids.len(),
            "scored_tokens": scored_tokens,
            "skipped_boundary_tokens": 0,
            "dense_prefix_len": 0,
            "decode_steps": decode_steps,
            "certified_decode_steps": 0,
            "total_nll": total_nll,
            "avg_nll": avg_nll,
            "perplexity": perplexity,
            "bits_per_token": bits_per_token,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "ms_per_token": ms_per_token,
        }))?
    );
    Ok(())
}

/// Map a `gpu_hal::Backend` to the lowercase label embedded in the
/// `[teacher_forced_json]` payload. Downstream tooling keys per-backend
/// aggregations off this string, so it MUST match the actual runtime backend
/// rather than the (Llama-original) hardcoded "hip" literal that the early
/// drafts of run_qwen35_teacher_forced inherited.
fn backend_label(backend: gpu_hal::Backend) -> &'static str {
    match backend {
        gpu_hal::Backend::Hip => "hip",
        gpu_hal::Backend::Cuda => "cuda",
        gpu_hal::Backend::Metal => "metal",
    }
}
