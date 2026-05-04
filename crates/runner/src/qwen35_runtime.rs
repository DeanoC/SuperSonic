use anyhow::Result;

use crate::bakes::ensure_hf_metadata_present;
use crate::qwen35_alt_runtime::run_qwen35_alt_runtime_if_requested;
use crate::qwen35_decode_loop::{run_qwen35_decode_loop, Qwen35DecodeLoop};
use crate::qwen35_decode_modes::{report_qwen35_decode_modes, resolve_qwen35_decode_modes};
use crate::qwen35_decode_report::{emit_qwen35_decode_report, Qwen35DecodeReport};
use crate::qwen35_engine_setup::{load_qwen35_engine, Qwen35EngineSetup};
use crate::qwen35_prefill::{run_qwen35_prefill, HostLmHeadRescorer, Qwen35Prefill};
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
use crate::{resolve_oracle_device, Cli};

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
        next_token,
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

    let eos_ids = text_config.eos_token_ids();
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
