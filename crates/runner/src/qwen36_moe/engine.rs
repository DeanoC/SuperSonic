//! Qwen3.6-MoE runtime engine.
//!
//! Owns the CLI-facing Qwen3.6-MoE flow: bake selection, dry-run/budget
//! reporting, prompt setup, layer loading, session allocation, prefill,
//! generation, optional speculative extension, and final telemetry. The
//! GPU launch details live in the lower-level chain, persistent-decode,
//! generation, and spec-verify modules.

use std::path::Path;

use anyhow::{Context, Result};
use gpu_hal::{set_backend, Backend};
use model_store::BakedStore;

use crate::qwen36_moe_cli::bake::{ensure_qwen36_bake, select_decode_bake};
use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::dry_run::{print_report, run_qwen36_moe_dry_run, DryRunReport};
use crate::qwen36_moe_cli::generation::{run_generation_step, Qwen36GenerationStep};
use crate::qwen36_moe_cli::geom::build_multi_layer_geom;
use crate::qwen36_moe_cli::host::lookup_embed_row;
use crate::qwen36_moe_cli::output::{
    print_decode_stream_start, print_generation_summary, print_last_logits_if_requested,
    print_sampling_summary,
};
use crate::qwen36_moe_cli::policy::{
    resolve_context_size, validate_cuda_v1_flags, validate_decode_backend,
    validate_persistent_kv_fp8_flags,
};
use crate::qwen36_moe_cli::prompt::{
    prepare_prompt, print_prompt_summary, validate_speculative_sampling,
};
use crate::qwen36_moe_cli::session::{prepare_decode_session, Qwen36DecodeSession};
use crate::qwen36_moe_cli::spec_verify::{run_speculative_extension, Qwen36SpeculativeExtension};
use crate::qwen36_moe_cli::timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_cli::vmm::{
    load_decode_layers_with_vmm_strategy, print_virtual_kv_stats_if_active,
    virtual_kv_stats_for_layers,
};
use crate::qwen36_moe_cli::vmm_config::{prepare_moe_runtime_config, should_use_qwen36_kv_vmm};
use crate::qwen36_moe_logits::XorshiftRng;
use crate::qwen36_moe_telemetry::{print_and_write_moe_residency_summary, MoeRouteRuntime};
use crate::registry::RegistryEntry;

pub fn run(cli: &crate::Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    run_inner(cli, entry, total_vram, None)
}

/// SpecPrefill sparse-prefill variant. `keep_mask[i] == true` means the
/// drafter selected prompt token `i` to be included in the target's
/// prefill; pruned positions are skipped entirely. The mask must be the
/// same length as the tokenized prompt (validated downstream); the
/// drafter side guarantees the last prompt token is kept (its logits
/// produce the first generation token). Inside the prefill loop, kept
/// tokens use their original prompt position for RoPE rotation but land
/// in compact KV-cache slots via `Qwen36MoeAttnStepParams::cache_pos`,
/// the same kernel-side split MTP already uses for draft-step rotation.
pub fn run_with_sparse_prefill(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    total_vram: u64,
    keep_mask: Vec<bool>,
) -> Result<()> {
    run_inner(cli, entry, total_vram, Some(keep_mask))
}

fn run_inner(
    cli: &crate::Cli,
    entry: &RegistryEntry,
    total_vram: u64,
    keep_mask: Option<Vec<bool>>,
) -> Result<()> {
    ensure_qwen36_bake(cli, entry)?;

    let (context_size, context_size_source) = resolve_context_size(cli);
    validate_persistent_kv_fp8_flags(cli)?;
    validate_cuda_v1_flags(cli, entry)?;

    let report = run_qwen36_moe_dry_run(
        &cli.model_dir,
        entry,
        total_vram,
        context_size,
        context_size_source,
        cli.batch_size.max(1),
        cli.kv_fp8,
        cli.no_bake,
        cli.device,
    )?;
    print_report(&report);
    if cli.dry_run {
        return Ok(());
    }

    validate_decode_backend(entry)?;

    println!();
    println!("=== Decode (Qwen3.6-MoE) ===");
    let sampling = SamplingParams {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        seed: cli.sampling_seed,
    };
    decode_text(
        &cli.model_dir,
        &report,
        &cli.prompt,
        cli.max_new_tokens.max(1),
        sampling,
        cli.emit_stage_timings,
        cli.speculative_decode,
        cli.fp8_runtime,
        cli.batched_spec_verify,
        entry.backend,
        cli.device,
        // Phase 3e.4: persistent decode is now the default. The legacy
        // `--persistent-decode` flag is a hidden no-op (kept for harness
        // back-compat); `--no-persistent-decode` is the documented
        // opt-out for A/B comparison or bisecting megakernel regressions.
        !cli.no_persistent_decode,
        cli.kv_fp8,
        cli.dump_last_logits,
        keep_mask,
    )?;
    Ok(())
}

/// Tokenize the prompt and run the multi-token decode loop end-to-end:
/// prefill the prompt one token at a time, then generate `max_new`
/// tokens via the configured sampling policy. Streams decoded text to stdout
/// as each token arrives.
///
/// State persistence across decode steps:
///  - Linear-attn `conv_state` + `recurrent_state` mutated in place by
///    the kernel.
///  - Full-attn KV cache: per-layer `[kv_max_t, Hkv*d]` buffers; the kernel
///    writes the current step's K/V at slot `position` and attends over
///    `kv_len = position + 1` past tokens. `kv_max_t` is sized for
///    `prompt_len + max_new` here.
///  - Persistent decode is the default path. The host-orchestrated chained
///    path remains available behind `--no-persistent-decode` for parity and
///    regression isolation.
///  - When self-speculative decode is enabled, each generation iteration can
///    append extra accepted MTP drafts after the regular base-model sample.
fn decode_text(
    model_dir: &Path,
    report: &DryRunReport,
    prompt: &str,
    max_new: usize,
    sampling: SamplingParams,
    emit_stage_timings: bool,
    speculative_decode: bool,
    fp8_runtime: bool,
    batched_spec_verify: bool,
    backend: Backend,
    ordinal: usize,
    persistent_decode: bool,
    kv_fp8: bool,
    dump_last_logits: bool,
    keep_mask: Option<Vec<bool>>,
) -> Result<()> {
    validate_speculative_sampling(speculative_decode, sampling)?;

    if keep_mask.is_some() && speculative_decode {
        anyhow::bail!(
            "SpecPrefill (sparse prefill via --specprefill-draft-dir) cannot be \
             combined with --speculative-decode (MTP self-speculative). MTP uses \
             the persistent decode kernel, which takes only `position` (no \
             `cache_pos` decoupling), so it would write at wrong KV slots when \
             the prompt has been pruned. R1 follow-up: plumb cache_pos through \
             MTP too."
        );
    }

    let decode_wall_start = std::time::Instant::now();
    let weight_prefix = report.kernel_params.weight_prefix;

    let prompt_setup_start = std::time::Instant::now();
    let prompt_setup = prepare_prompt(model_dir, &report.config.text_config, prompt)?;
    let prompt_setup_elapsed = prompt_setup_start.elapsed();
    let tokenizer = prompt_setup.tokenizer;
    let prompt_ids = prompt_setup.prompt_ids;
    let eos_id = prompt_setup.eos_id;
    print_prompt_summary(prompt, &prompt_ids);

    let bake_open_start = std::time::Instant::now();
    let bake = select_decode_bake(model_dir, fp8_runtime)?;
    if backend == Backend::Cuda && !bake.weight_mode.is_int4() {
        anyhow::bail!(
            "Qwen3.6-35B-A3B CUDA v1 requires an INT4/q4km bake; selected {} from {}",
            bake.weight_mode.display_name(),
            bake.bake_dir.display(),
        );
    }
    println!(
        "  loading from bake: {} ({})",
        bake.bake_dir.display(),
        bake.weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake.bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake.bake_dir.display()))?;
    let bake_open_elapsed = bake_open_start.elapsed();

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(backend);

    // KV cache size: needs to fit prompt_len + max_new past tokens. Sized
    // generously here since per-layer KV is small (10 full-attn layers ×
    // [kv_max_t, Hkv*d=512] BF16 = 10 KiB per token of context).
    let kv_max_t = prompt_ids.len() + max_new;

    println!(
        "  loading {} layers ({} INT4 sidecar sets, KV cache cap = {} tokens)…",
        geom.num_layers,
        if bake.weight_mode.is_int4() {
            geom.num_layers
        } else {
            0
        },
        kv_max_t,
    );

    let layer_load_start = std::time::Instant::now();
    let mut moe_runtime = prepare_moe_runtime_config(
        speculative_decode,
        persistent_decode,
        backend,
        geom.top_k as usize,
    )?;
    let kv_vmm = should_use_qwen36_kv_vmm(backend, ordinal)?;
    let loaded_layers = load_decode_layers_with_vmm_strategy(
        &store,
        ordinal,
        backend,
        &geom,
        &report.config.text_config,
        weight_prefix,
        bake.weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        moe_runtime.vmm_mode,
        moe_runtime.island_cap_experts,
        moe_runtime.protected_experts,
        moe_runtime.prefetch_mode,
        moe_runtime.prefetch_ranks,
        moe_runtime.transition_min_observations,
        moe_runtime.async_prefetch,
        moe_runtime.async_staging_pages,
        persistent_decode,
    )?;
    let layer_load_elapsed = layer_load_start.elapsed();
    let mut layers = loaded_layers.layers;
    let _moe_expert_arena = loaded_layers.moe_expert_arena;
    let mut _moe_expert_residency = loaded_layers.moe_expert_residency;
    let virtual_kv_stats = virtual_kv_stats_for_layers(&layers);
    print_virtual_kv_stats_if_active(virtual_kv_stats, kv_fp8, backend, ordinal);
    let session_start = std::time::Instant::now();
    let session = prepare_decode_session(
        &store,
        ordinal,
        &geom,
        &report.config.text_config,
        weight_prefix,
        kv_max_t,
        speculative_decode,
        batched_spec_verify,
        persistent_decode,
        &mut layers,
    )?;
    let session_elapsed = session_start.elapsed();
    let Qwen36DecodeSession {
        final_norm_w_buf,
        lm_head_w_buf,
        mut logits_buf,
        mut counter_buf,
        mut final_hidden_buf,
        mut mtp_buffers,
        mut mtp_forward_scratch,
        mut mtp_chain_scratch,
        embed_w_buf,
        mut linear_attn_snapshot,
        mut persistent_scratch,
    } = session;

    print_decode_stream_start(tokenizer.as_ref(), &prompt_ids, max_new);

    // Sparse-prefill setup. `kept_positions[i]` holds the original prompt
    // position of the i-th kept token; the prefill loop iterates over
    // these positions instead of every prompt token. In the dense case
    // (keep_mask=None) it's just `0..prompt_ids.len()` and the loop is
    // bit-equal to before. The drafter side (run_specprefill_qwen36_moe)
    // guarantees `keep_mask.last() == true` and `keep_mask.len() ==
    // prompt_ids.len()`; we re-validate as a defence against future
    // mis-wiring.
    let kept_positions: Vec<usize> = match &keep_mask {
        Some(mask) => {
            if mask.len() != prompt_ids.len() {
                anyhow::bail!(
                    "sparse-prefill: keep_mask.len()={} != prompt_ids.len()={}",
                    mask.len(),
                    prompt_ids.len(),
                );
            }
            let kept: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| k.then_some(i))
                .collect();
            if kept.is_empty() {
                anyhow::bail!("sparse-prefill: keep_mask kept zero positions");
            }
            if *kept.last().unwrap() != prompt_ids.len() - 1 {
                anyhow::bail!(
                    "sparse-prefill: last prompt position must be kept (got last kept={})",
                    kept.last().unwrap()
                );
            }
            kept
        }
        None => (0..prompt_ids.len()).collect(),
    };
    let effective_prompt_len = kept_positions.len();
    if keep_mask.is_some() {
        eprintln!(
            "[specprefill] sparse prefill: {}/{} prompt tokens kept",
            effective_prompt_len,
            prompt_ids.len(),
        );
    }

    // `Qwen36DecodeLoopState::new` assumes dense (every position kept).
    // For sparse, override the initial token to be the first *kept*
    // prompt token and shrink `total_steps` to `effective_prompt_len +
    // max_new - 1`. `position` (the loop's compact KV-slot counter) and
    // `current_token` advance per chain-step iteration.
    let mut loop_state = Qwen36DecodeLoopState::new(&prompt_ids, max_new);
    if keep_mask.is_some() {
        loop_state.current_token = prompt_ids[kept_positions[0]];
        loop_state.total_steps = effective_prompt_len + max_new - 1;
    }
    let mut rng = XorshiftRng::new(sampling.seed);
    print_sampling_summary(sampling);

    // Per-stage wall-clock accumulators. Aggregated across generation steps
    // only (prefill steps run the chain but skip the lm_head/sample stages,
    // so timing prefill mixed with gen would distort the per-token average).
    // `chain_ms` includes the GPU work + the D2H copy of `final_hidden_bytes`
    // — `run_chained_decode` syncs before returning, so the wall-clock here
    // is a real GPU+sync measurement. CPU-side stages (embed lookup, lm_head
    // GEMV, sampling, detokenize) are pure host work.
    let mut stage_timings = Qwen36StageTimingTotals::default();
    let mut prefill_steps = 0usize;
    let mut prefill_embed_elapsed = std::time::Duration::ZERO;
    let mut prefill_chain_elapsed = std::time::Duration::ZERO;
    let mut generation_wall_start = None;
    let mut moe_routes = MoeRouteRuntime::new(
        geom.num_layers as usize,
        geom.top_k as usize,
        moe_runtime.sparse_requested,
        moe_runtime.prefetch_mode,
        moe_runtime.transition_min_observations,
    );

    for step in 0..loop_state.total_steps {
        // When speculative decode is on, each iteration can commit
        // multiple tokens (up to K+1), so the standard `total_steps =
        // prompt_len + max_new - 1` count over-shoots. Break here once
        // we've already committed `max_new` tokens — otherwise the
        // next regular chain call would request a cache slot beyond
        // `kv_max_t = prompt_len + max_new` (status 120). Plain decode
        // stays bit-identical because it always emits exactly one
        // token per iteration.
        if loop_state.reached_max_new() {
            break;
        }
        let is_gen_step = step + 1 >= effective_prompt_len;
        if is_gen_step && generation_wall_start.is_none() {
            generation_wall_start = Some(std::time::Instant::now());
        }
        // Per-step (rope_pos, cache_pos) for the chain step. In dense
        // mode they equal `loop_state.position`. In sparse mode rope_pos
        // is the original prompt position of the kept token (during
        // prefill) or `prompt_ids.len() + gen_offset` (during gen);
        // cache_pos is always `loop_state.position` (the compact slot
        // index that advances by 1 per processed step).
        let (rope_pos, chain_cache_pos): (i32, Option<i32>) = match &keep_mask {
            None => (loop_state.position, None),
            Some(_) => {
                let rp = if step < effective_prompt_len {
                    kept_positions[step] as i32
                } else {
                    let gen_off = step - effective_prompt_len;
                    (prompt_ids.len() + gen_off) as i32
                };
                (rp, Some(loop_state.position))
            }
        };
        // Embed lookup for the current token.
        let t0 = std::time::Instant::now();
        let initial_hidden = lookup_embed_row(
            &store,
            weight_prefix,
            loop_state.current_token as usize,
            geom.hidden as usize,
        )
        .with_context(|| {
            format!(
                "embed lookup token {} (step {step})",
                loop_state.current_token
            )
        })?;
        let t_embed_step = t0.elapsed();

        // Run the chain. Linear-attn state mutates in `layers` in place.
        // `run_chained_decode_fast` skips the per-layer D2H sync chain
        // (~80 GPU syncs/token on 35B-A3B) — `decode_text` only consumes
        // `final_hidden_bytes`. The multilayer parity test still calls
        // the legacy `run_chained_decode` which captures per-layer.
        let t1 = std::time::Instant::now();
        // When `--emit-stage-timings` is set, sync after each step launch
        // so the per-stage `kernel_*_us` accumulators in `outputs` reflect
        // GPU compute time. Without it, PR #80's async dispatch path
        // would record host queue time instead — fast but useless for
        // stage-level perf attribution. The total `chain_ms` measured by
        // the wall-clock around this call stays correct either way
        // because `run_chained_decode_fast` ends with a D2H copy that
        // implicitly drains the queue.
        // Phase 3f: on generation steps when the persistent path is
        // active, fold final RMSnorm + lm_head GEMV into the
        // megakernel — saves the separate `lm_head_launch` (one launch
        // + ~30 µs) and the H2D round-trip that staged final_hidden
        // into final_hidden_buf. The host then D2Hs `logits_buf`
        // directly. On prefill steps logits aren't needed; on the
        // chained path the explicit lm_head_launch path stays.
        let fold = if is_gen_step {
            Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                final_norm_w: &final_norm_w_buf,
                lm_head_w: &lm_head_w_buf,
                logits_out: &mut logits_buf,
                vocab: geom.vocab,
            })
        } else {
            None
        };
        let chain_step = run_chain_step(Qwen36ChainStep {
            ordinal,
            geom: &geom,
            store: &store,
            layers: &mut layers,
            persistent_scratch: persistent_scratch.as_mut(),
            moe_expert_residency: _moe_expert_residency.as_mut(),
            moe_runtime: &mut moe_runtime,
            moe_routes: &mut moe_routes,
            initial_hidden: &initial_hidden,
            position: rope_pos,
            cache_pos: chain_cache_pos,
            step,
            is_gen_step,
            emit_stage_timings,
            fold,
        })?;
        let outputs = chain_step.outputs;
        let lm_head_folded = chain_step.lm_head_folded;
        let t_chain_step = t1.elapsed();
        loop_state.position += 1;

        // KV-FP8 sidecar descriptors stay fixed across decode. The
        // persistent kernel computes the rolling covered range from
        // `position` and `kv_shadow_window`, so no descriptor re-upload is
        // needed when old sidecar slots roll over.

        // Prefill steps: feed the next prompt token without computing logits.
        // For sparse-prefill, the next "prompt token" is the next *kept*
        // prompt token (`kept_positions[step + 1]` indexes into the
        // original prompt).
        if step + 1 < effective_prompt_len {
            prefill_steps += 1;
            prefill_embed_elapsed += t_embed_step;
            prefill_chain_elapsed += t_chain_step;
            loop_state.current_token = prompt_ids[kept_positions[step + 1]];
            continue;
        }

        let next_token = run_generation_step(Qwen36GenerationStep {
            ordinal,
            geom: &geom,
            step,
            lm_head_folded,
            dump_last_logits,
            tokenizer: tokenizer.as_ref(),
            sampling,
            t_embed_step,
            t_chain_step,
            outputs: &outputs,
            final_norm_w_buf: &final_norm_w_buf,
            lm_head_w_buf: &lm_head_w_buf,
            final_hidden_buf: &mut final_hidden_buf,
            logits_buf: &mut logits_buf,
            counter_buf: &mut counter_buf,
            loop_state: &mut loop_state,
            rng: &mut rng,
            stage_timings: &mut stage_timings,
        })?;

        if Some(next_token) == eos_id {
            break;
        }
        loop_state.current_token = next_token;

        if let (Some(mtp), Some(fwd_scratch), Some(chain_scratch), Some(embed_w)) = (
            mtp_buffers.as_mut(),
            mtp_forward_scratch.as_mut(),
            mtp_chain_scratch.as_mut(),
            embed_w_buf.as_ref(),
        ) {
            if loop_state.reached_max_new() {
                break;
            }

            let h_base = outputs.final_hidden_bytes.clone();
            // Runs either batched or sequential speculative verify depending on
            // whether session setup allocated a linear-attn snapshot.
            let result = run_speculative_extension(Qwen36SpeculativeExtension {
                ordinal,
                geom: &geom,
                store: &store,
                weight_prefix,
                layers: &mut layers,
                persistent_scratch: persistent_scratch.as_mut(),
                mtp,
                forward_scratch: fwd_scratch,
                chain_scratch,
                embed_w,
                final_norm_w: &final_norm_w_buf,
                lm_head_w: &lm_head_w_buf,
                final_hidden: &mut final_hidden_buf,
                logits: &mut logits_buf,
                counter: &mut counter_buf,
                linear_attn_snapshot: linear_attn_snapshot.as_mut(),
                loop_state: &loop_state,
                h_base_in: &h_base,
                first_token: next_token,
                stage_timings: &mut stage_timings,
                emit_stage_timings,
            })?;

            if loop_state.append_speculative_emissions(&result, tokenizer.as_ref(), eos_id) {
                break;
            }
        }
    }

    print_last_logits_if_requested(dump_last_logits, &loop_state.last_logits_bytes);
    print_generation_summary(&loop_state.generated_ids, prompt_ids.len(), eos_id);
    if let Some(manager) = _moe_expert_residency.as_ref() {
        print_and_write_moe_residency_summary(
            manager,
            virtual_kv_stats,
            &loop_state.generated_ids,
            moe_routes.route_telemetry.as_ref(),
            moe_runtime.sparse_telemetry.as_ref(),
        )?;
    }
    stage_timings.print_if_requested(emit_stage_timings);
    if emit_stage_timings {
        let to_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let prefill_total_ms = to_ms(prefill_embed_elapsed + prefill_chain_elapsed);
        let generation_wall_ms = generation_wall_start
            .as_ref()
            .map(|start| to_ms(start.elapsed()))
            .unwrap_or(0.0);
        eprintln!(
            "[qwen36-moe lifecycle-timings] prompt_setup_ms={:.3} \
             bake_open_ms={:.3} layer_load_ms={:.3} session_ms={:.3} \
             prefill_steps={} prefill_embed_ms={:.3} prefill_chain_ms={:.3} \
             prefill_total_ms={:.3} generation_wall_ms={:.3} total_wall_ms={:.3}",
            to_ms(prompt_setup_elapsed),
            to_ms(bake_open_elapsed),
            to_ms(layer_load_elapsed),
            to_ms(session_elapsed),
            prefill_steps,
            to_ms(prefill_embed_elapsed),
            to_ms(prefill_chain_elapsed),
            prefill_total_ms,
            generation_wall_ms,
            to_ms(decode_wall_start.elapsed()),
        );
    }

    Ok(())
}
