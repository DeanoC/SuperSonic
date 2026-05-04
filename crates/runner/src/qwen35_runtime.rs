use anyhow::Result;
use std::time::Instant;

use crate::bakes::ensure_hf_metadata_present;
use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::prefill_engine;
use crate::qwen35_component_decode::{run_qwen35_component_single_decode, Qwen35ComponentDecode};
use crate::qwen35_decode_modes::{report_qwen35_decode_modes, resolve_qwen35_decode_modes};
use crate::qwen35_decode_report::{emit_qwen35_decode_report, Qwen35DecodeReport};
use crate::qwen35_decode_traces::{
    qwen35_persistent_decode_trace_enabled, run_qwen35_persistent_decode_traces,
    Qwen35PersistentDecodeTrace,
};
use crate::qwen35_decode_util::{token_history, token_history_with_next};
use crate::qwen35_engine_setup::{load_qwen35_engine, Qwen35EngineSetup};
use crate::qwen35_kv_trace::trace_kv_cache;
use crate::qwen35_prefill::{
    run_qwen35_prefill, sample_qwen_logits_with_rescore, HostLmHeadRescorer, Qwen35Prefill,
};
use crate::qwen35_startup::{
    load_qwen35_startup, validate_qwen35_startup, Qwen35Policy, Qwen35Startup,
};
use crate::qwen35_validation::{
    qwen35_oracle_script_path, resolve_qwen_oracle_model_id, run_qwen35_oracle_validation,
    trace_qwen35_oracle_prefill_layer,
};
use crate::qwen35_virtual_kv::report_qwen35_virtual_kv_after_prefill;
use crate::qwen35_vram::check_qwen35_vram;
use crate::registry::{self, Backend, FamilyParams, GpuArch, ModelVariant, RegistryEntry};
use crate::{
    qwen35_dflash_engine, resolve_oracle_device, should_fetch_exact_bake, specprefill_engine,
    try_download_bake, validate, Cli,
};

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
    if cli.dflash {
        // DFlash needs the target's HF metadata (config.json + tokenizer.json)
        // and the INT4 bake. Reuse the same download hooks as the regular
        // Qwen35 path so the dflash dispatch is self-contained on a fresh
        // machine: ensure_hf_metadata_present fetches HF metadata from the
        // bake tarball if config.json is missing, then we verify or download
        // the INT4 bake itself.
        ensure_hf_metadata_present(&cli, &model_variant)?;
        if !cli.no_bake {
            let variant = model_store::fetch::BakeVariant::Int4Gptq;
            let bake_dir = variant.bake_dir(&cli.model_dir);
            let _lock = model_store::BakeLock::acquire(&cli.model_dir)
                .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
            if should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir)) {
                let canonical_model = model_variant.to_string();
                match try_download_bake(&cli, variant, &canonical_model, &bake_dir) {
                    Ok(true) => {
                        eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                    }
                    Ok(false) => {
                        anyhow::bail!(
                            "no INT4 bake at {} and --no-download set.\n\
                             Run:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                    Err(e) => {
                        anyhow::bail!(
                            "could not obtain INT4 bake for --dflash: {e}\n\n\
                             INT4 baking requires a GPTQ calibration pass in Python. \
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            cli.model_dir.display(),
                        );
                    }
                }
            }
        }
        return qwen35_dflash_engine::run_qwen35_dflash(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }
    // --dflash-* guard already ran before the family dispatch above.

    // --specprefill-* dispatch. Validation already ran in
    // validate_specprefill_flags; the presence of --specprefill-draft-dir
    // is the gate that switches to the SpecPrefill orchestrator.
    if cli.specprefill_draft_dir.is_some() {
        return specprefill_engine::run_specprefill(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Qwen36Moe(_) => unreachable!("qwen3.6-moe handled above"),
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
        FamilyParams::Phi4(_) => unreachable!("phi4 handled above"),
        FamilyParams::Llama31(_) => unreachable!("llama3.1 handled above"),
    };
    let host_lm_head_rescorer = HostLmHeadRescorer::from_model_dir(&cli.model_dir)?;

    // Install the per-(arch, model) HIP launch preset (grid size +
    // cooperative flag) if one is registered. User env vars still override
    // inside the bridge. Always called — `(0, false)` clears any stale
    // preset from a prior run, so switching models doesn't inherit the
    // previous one's grid. No-op on CUDA builds.
    {
        let preset = registry::qwen35_4b_launch_preset(&entry.arch, &entry.model);
        let (blocks, coop) = preset.unwrap_or((0, false));
        kernel_ffi::set_qwen35_4b_launch_preset(blocks, coop);
        if let Some((blocks, coop)) = preset {
            eprintln!("[preset] qwen35_4b launch: blocks={blocks} cooperative={coop}");
        }
    }

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
        params,
        backend,
        gpu_arch,
        ordinal,
        bootstrap_downloaded,
        q4km_like,
        context_tokens,
    )?;

    // When using FP8 runtime weights, tell the oracle to use the same FP8 weights
    // (dequanted to BF16) so we compare apples-to-apples.
    let fp8_oracle_dir = if cli.fp8_runtime {
        Some(cli.model_dir.clone())
    } else {
        None
    };
    let oracle_device = resolve_oracle_device(&cli.oracle_device, backend, ordinal);

    // Run prefill (native GPU or oracle)
    let qwen_oracle_model_id =
        resolve_qwen_oracle_model_id(cli.model_id.as_deref(), &cli.model_dir, &model_variant);
    let Qwen35Prefill {
        logits: prefill_logits,
        native_trace: native_prefill_trace,
        mut next_token,
    } = run_qwen35_prefill(
        &cli,
        &mut engine,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        host_lm_head_rescorer.as_ref(),
        allow_host_lm_head_rescore,
    )?;

    if cli.dump_last_logits {
        use std::io::Write as _;
        print!("\nLAST_LOGITS: ");
        for (i, x) in prefill_logits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    report_qwen35_virtual_kv_after_prefill(&mut engine)?;

    // Optionally run oracle for validation
    let oracle_output = run_qwen35_oracle_validation(
        &cli,
        &engine,
        &text_config,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        &prefill_logits,
        native_prefill_trace.as_ref(),
        next_token,
    )?;

    if let (Some(trace_layer), Some(output)) =
        (cli.trace_oracle_prefill_layer, oracle_output.as_ref())
    {
        let oracle_script = qwen35_oracle_script_path();
        trace_qwen35_oracle_prefill_layer(
            &mut engine,
            trace_layer,
            &prompt_ids,
            &oracle_script,
            &qwen_oracle_model_id,
            &cli.oracle_dtype,
            &oracle_device,
            fp8_oracle_dir.as_deref(),
            output,
        )?;
    }

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
    );
    report_qwen35_decode_modes(&cli, &decode_modes, use_4b_kernel, cuda_08b_hero_enabled);

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;
    let mut gpu_max_delta = 0.0f32;
    let mut native_decode_timings = DecodeStageTimings::default();
    let mut native_decode_timing_steps = 0usize;
    let eos_ids = text_config.eos_token_ids();

    // For batched decode, track per-sequence tokens
    let mut batch_next_tokens: Vec<u32> = vec![next_token; cli.batch_size];

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        // Stop on EOS token (sequence 0 drives the output)
        if eos_ids.contains(&next_token) {
            break;
        }

        let seqlen_offset = seqlen_start + step;

        if cli.batch_size > 1 {
            if qwen35_persistent_decode_trace_enabled(cli) {
                let trace_token_ids =
                    token_history_with_next(&prompt_ids, &generated_ids, next_token);
                let trace_tokens = vec![next_token; cli.batch_size];
                run_qwen35_persistent_decode_traces(
                    &mut engine,
                    Qwen35PersistentDecodeTrace {
                        cli,
                        trace_token_ids: trace_token_ids.as_slice(),
                        trace_tokens: trace_tokens.as_slice(),
                        seqlen_offset,
                        ordinal,
                        kv_chunk_size: params.kv_chunk_size,
                        use_4b_kernel,
                        batch_mode: true,
                    },
                )?;
            }
            let (batch_logits, batch_timings) = if decode_modes.replay_kv_fp8_enabled {
                let token_ids = token_history_with_next(&prompt_ids, &generated_ids, next_token);
                let logits = engine.rebuild_prefill_state(&token_ids, true)?;
                (vec![logits; cli.batch_size], None)
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_batch_with_timings(&batch_next_tokens, seqlen_offset)?;
                (logits, Some(timings))
            } else {
                // Batched decode
                (
                    engine.decode_step_batch(&batch_next_tokens, seqlen_offset)?,
                    None,
                )
            };
            if let Some(timings) = batch_timings {
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
            }

            // Use sequence 0's logits for output and validation
            let logits = &batch_logits[0];

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(logits, oracle_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token} batch_size={}",
                        cli.batch_size
                    );
                }
            }

            // Sample next tokens for all sequences
            let sampling_start = Instant::now();
            for (bi, seq_logits) in batch_logits.iter().enumerate() {
                batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
            }
            if batch_timings.is_some() {
                native_decode_timings.host_sampling_ms +=
                    sampling_start.elapsed().as_secs_f64() * 1000.0;
            }

            generated_ids.push(next_token);
            if trace_kv_cache_enabled {
                let cache_token_ids = token_history(&prompt_ids, &generated_ids);
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
            next_token = batch_next_tokens[0];
        } else {
            // Single-sequence decode (original path)
            let mut maybe_fast_token = None;
            let mut can_rescore_with_normed = false;
            let logits = if decode_modes.cuda_fast_greedy_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if decode_modes.metal_fast_greedy_enabled {
                let token = engine.decode_step_metal_fast_greedy(next_token, seqlen_offset)?;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if cuda_08b_hero_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_08b_hero(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if decode_modes.replay_decode_enabled {
                let token_ids = token_history_with_next(&prompt_ids, &generated_ids, next_token);
                prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?
            } else if decode_modes.replay_kv_fp8_enabled {
                let token_ids = token_history_with_next(&prompt_ids, &generated_ids, next_token);
                engine.rebuild_prefill_state(&token_ids, false)?
            } else if decode_modes.component_single_decode_enabled {
                run_qwen35_component_single_decode(
                    &mut engine,
                    Qwen35ComponentDecode {
                        cli,
                        prompt_ids: &prompt_ids,
                        generated_ids: &generated_ids,
                        next_token,
                        seqlen_offset,
                        ordinal,
                        kv_chunk_size: params.kv_chunk_size,
                        use_4b_kernel,
                    },
                )?
            } else if decode_modes.kernel_single_decode_enabled {
                if qwen35_persistent_decode_trace_enabled(cli) {
                    let trace_token_ids =
                        token_history_with_next(&prompt_ids, &generated_ids, next_token);
                    run_qwen35_persistent_decode_traces(
                        &mut engine,
                        Qwen35PersistentDecodeTrace {
                            cli,
                            trace_token_ids: trace_token_ids.as_slice(),
                            trace_tokens: &[next_token],
                            seqlen_offset,
                            ordinal,
                            kv_chunk_size: params.kv_chunk_size,
                            use_4b_kernel,
                            batch_mode: false,
                        },
                    )?;
                }
                if cli.emit_stage_timings {
                    let (logits, timings) = engine
                        .decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    can_rescore_with_normed = true;
                    logits
                } else {
                    can_rescore_with_normed = true;
                    engine
                        .decode_step_batch(&[next_token], seqlen_offset)?
                        .remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                can_rescore_with_normed = true;
                logits
            } else {
                can_rescore_with_normed = true;
                engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token = if let Some(token) = maybe_fast_token {
                token
            } else {
                let normed = if can_rescore_with_normed && allow_host_lm_head_rescore {
                    Some(engine.last_normed_host_f32()?)
                } else {
                    None
                };
                sample_qwen_logits_with_rescore(
                    &logits,
                    normed.as_deref(),
                    host_lm_head_rescorer
                        .as_ref()
                        .filter(|_| allow_host_lm_head_rescore),
                )?
            };

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(&logits, oracle_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
                    );
                }
            }

            if gpu_validate_enabled {
                let gpu_token_ids =
                    token_history_with_next(&prompt_ids, &generated_ids, next_token);
                let gpu_logits = prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &gpu_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                )?;
                let delta = validate::max_abs_delta(&logits, &gpu_logits);
                let gpu_token = DecodeEngine::greedy_sample(&gpu_logits);
                let token_match = if gpu_token == native_token {
                    ""
                } else {
                    " MISMATCH"
                };
                if delta > gpu_max_delta {
                    gpu_max_delta = delta;
                }
                eprintln!(
                    "[gpu-validate] step={step} seq_off={seqlen_offset} delta={delta:.4} native_token={native_token} gpu_token={gpu_token}{token_match}"
                );
            }

            generated_ids.push(next_token);
            next_token = native_token;

            if trace_kv_cache_enabled {
                let cache_token_ids = token_history(&prompt_ids, &generated_ids);
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
        }
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    emit_qwen35_decode_report(Qwen35DecodeReport {
        tokenizer: &tokenizer,
        prompt_ids: &prompt_ids,
        generated_ids: &generated_ids,
        emit_generated_json: cli.emit_generated_json,
        decode_ms,
        max_delta,
        gpu_max_delta,
        batch_size: cli.batch_size,
        emit_stage_timings: cli.emit_stage_timings,
        native_decode_timings: &native_decode_timings,
        native_decode_timing_steps,
    })?;

    Ok(())
}
