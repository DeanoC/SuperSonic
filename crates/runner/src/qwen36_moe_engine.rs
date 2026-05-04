//! Qwen3.6-MoE runtime engine.
//!
//! PR 3 stage: dry-run only. Loads `config.json`, enumerates the safetensors
//! checkpoint, computes the analytic weight + state footprint, and reports
//! it against the registry's VRAM budget. No GPU allocation, no kernel —
//! that lands in PR 4 (CUDA) and PR 6 (HIP).
//!
//! The reason for the enumerate-only dry-run is the BF16 35B-A3B checkpoint
//! is ~65 GiB and won't fit a 24 GiB GPU. Until the INT4/q4km bake exists,
//! the only meaningful runtime check is "did the safetensors index match
//! what we expect from the config" plus a budget comparison.

use std::path::Path;

use anyhow::{Context, Result};
use gpu_hal::{set_backend, Backend};
use model_store::BakedStore;

use crate::qwen36_moe_bake::{ensure_qwen36_bake, select_decode_bake};
use crate::qwen36_moe_chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_decode::XorshiftRng;
use crate::qwen36_moe_dry_run::{print_report, run_qwen36_moe_dry_run, DryRunReport};
use crate::qwen36_moe_generation::{run_generation_step, Qwen36GenerationStep};
use crate::qwen36_moe_geom::build_multi_layer_geom;
use crate::qwen36_moe_host::lookup_embed_row;
use crate::qwen36_moe_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_output::{
    print_decode_stream_start, print_generation_summary, print_last_logits_if_requested,
    print_sampling_summary,
};
use crate::qwen36_moe_policy::{
    resolve_context_size, validate_decode_backend, validate_persistent_kv_fp8_flags,
};
use crate::qwen36_moe_prompt::{
    prepare_prompt, print_prompt_summary, validate_speculative_sampling,
};
use crate::qwen36_moe_session::{prepare_decode_session, Qwen36DecodeSession};
use crate::qwen36_moe_spec_verify::{
    restore_and_replay_accepted_prefix, run_batched_spec_verify_inputs,
    run_sequential_spec_verify_input, Qwen36BatchedSpecVerifyInputs,
    Qwen36SequentialSpecVerifyInput, Qwen36SpecReplayAccepted,
};
use crate::qwen36_moe_speculative::{
    run_speculative_decode_step, run_speculative_decode_step_batched,
};
use crate::qwen36_moe_state::refresh_linear_attn_state;
use crate::qwen36_moe_telemetry::{print_and_write_moe_residency_summary, MoeRouteRuntime};
use crate::qwen36_moe_timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_vmm::{
    load_decode_layers_with_vmm_strategy, prepare_moe_runtime_config,
    print_virtual_kv_stats_if_active, should_use_qwen36_kv_vmm, virtual_kv_stats_for_layers,
};
use crate::registry::RegistryEntry;

const QWEN36_NUM_SPECULATIVE_TOKENS: usize = 3;

pub fn run(cli: &crate::Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    ensure_qwen36_bake(cli, entry)?;

    let (context_size, context_size_source) = resolve_context_size(cli);
    validate_persistent_kv_fp8_flags(cli)?;

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

    // Real decode path (PR 4c step 2). Uses the host-orchestrated chained
    // launches in `crate::qwen36_moe_decode::run_chained_decode` against
    // per-layer weight buffers loaded from the baked package. INT4 GPTQ
    // is the realistic path on 24 GiB VRAM; the BF16 fallback is wired
    // for completeness but won't fit the 65 GiB 35B model. The multi-layer
    // parity test in `crates/runner/tests/qwen36_moe_multilayer_parity.rs`
    // gates the decode core against the Python multi-layer oracle for both
    // BF16 (cos_sim 0.9999) and INT4 (cos_sim 0.9999) modes.
    //
    // Caveats for PR 4c step 2:
    //  - One token, fresh state. Conv + recurrent state start zeroed; the
    //    full-attn KV cache isn't allocated (single-block kernels run with
    //    `kv_len=1`). Multi-token generation needs prefill + state
    //    persistence which land later.
    //  - lm_head INT4 dequant runs host-side (~1 GiB BF16 buffer); the
    //    lm_head GEMV likewise. Lifting both to the GPU is PR 4d.
    //  - Tokenizer not wired — the produced token is printed as a raw vocab
    //    id so the "doesn't bail" criterion is verifiable end-to-end.
    println!();
    println!("=== Decode (PR 4c step 2: host-orchestrated chained launches) ===");
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
    )?;
    Ok(())
}

/// Tokenize the prompt and run the multi-token decode loop end-to-end:
/// prefill the prompt one token at a time, then generate `max_new`
/// tokens via greedy argmax against the (cached) host-side lm_head
/// GEMV. Streams decoded text to stdout as each token arrives.
///
/// State persistence across decode steps:
///  - Linear-attn `conv_state` + `recurrent_state` mutated in place by
///    the kernel.
///  - Full-attn KV cache (PR 4d): per-layer `[kv_max_t, Hkv*d]` BF16
///    buffers; the kernel writes the current step's K/V at slot
///    `position` and attends over `kv_len = position + 1` past tokens.
///    `kv_max_t` sized for `prompt_len + max_new` here.
///
/// Greedy decoding only — sampling (temperature/top-p) and GPU-side
/// lm_head GEMV (currently host F32) are next perf/quality steps.
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
) -> Result<()> {
    validate_speculative_sampling(speculative_decode, sampling)?;

    let weight_prefix = report.kernel_params.weight_prefix;

    let prompt_setup = prepare_prompt(model_dir, &report.config.text_config, prompt)?;
    let tokenizer = prompt_setup.tokenizer;
    let prompt_ids = prompt_setup.prompt_ids;
    let eos_id = prompt_setup.eos_id;
    print_prompt_summary(prompt, &prompt_ids);

    let bake = select_decode_bake(model_dir, fp8_runtime)?;
    println!(
        "  loading from bake: {} ({})",
        bake.bake_dir.display(),
        bake.weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake.bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake.bake_dir.display()))?;

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
    let mut layers = loaded_layers.layers;
    let _moe_expert_arena = loaded_layers.moe_expert_arena;
    let mut _moe_expert_residency = loaded_layers.moe_expert_residency;
    let virtual_kv_stats = virtual_kv_stats_for_layers(&layers);
    print_virtual_kv_stats_if_active(virtual_kv_stats, kv_fp8, backend, ordinal);
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

    let mut loop_state = Qwen36DecodeLoopState::new(&prompt_ids, max_new);
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
        let is_gen_step = step + 1 >= prompt_ids.len();
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
            position: loop_state.position,
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
        if step + 1 < prompt_ids.len() {
            loop_state.current_token = prompt_ids[step + 1];
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

        // Phase 6.3d: speculative extension. After the regular sample,
        // try to commit additional tokens via MTP draft chain +
        // sequential base verification. The closure wraps one base
        // decode step (embed → chain → lm_head → host argmax). Honors
        // `max_new` and EOS by truncating emitted tokens; the
        // outer-loop counter advances normally because each iteration
        // still runs at least one base step.
        //
        // Sequential verify gives no amortized speedup vs plain greedy
        // (each accepted draft costs one base step to produce the next
        // prediction). Phase 6.4's batched verification is what lifts
        // tok/s. This wiring is the correctness foundation.
        if let (Some(mtp), Some(fwd_scratch), Some(chain_scratch), Some(embed_w)) = (
            mtp_buffers.as_mut(),
            mtp_forward_scratch.as_mut(),
            mtp_chain_scratch.as_mut(),
            embed_w_buf.as_ref(),
        ) {
            if loop_state.reached_max_new() {
                break;
            }
            // Cap K to the remaining max_new headroom so the verify
            // loop never writes cache slots beyond what we'll
            // actually commit to `generated_ids`. Spec emits up to
            // K+1 tokens (K accepted + 1 corrected/bonus), so the
            // available draft count is `headroom - 1`. If headroom <=
            // 1 we can still emit 1 token via the K=0 fallback; if
            // headroom == 0 we already broke out above.
            let dynamic_k = loop_state.speculative_draft_count(QWEN36_NUM_SPECULATIVE_TOKENS);
            let h_base = outputs.final_hidden_bytes.clone();
            // P2: thread spec-verify timings into the engine-level
            // accumulators so `--emit-stage-timings` reports honest
            // per-token costs under speculative decode. Without these
            // captures, every base step inside the verify loop would
            // be invisible to `gen_steps` / `t_chain` / `t_lm_head`,
            // making the speculative path look ~K+1× faster than it
            // really is on stage-timings dashboards.
            //
            // Phase 6.4c.2: route through the BATCHED driver when
            // `--batched-spec-verify` is set. The batched closure runs
            // K+1 chains sequentially (state mutates through them),
            // accumulates K+1 final_hidden bytes, runs ONE batched
            // lm_head over [K+1, hidden], does K+1 host argmaxes. On
            // partial-accept the engine restores linear-attn state
            // from the pre-spec snapshot then replays the accepted
            // prefix sequentially to advance state correctly.
            let result = if let Some(snapshot) = linear_attn_snapshot.as_mut() {
                refresh_linear_attn_state(ordinal, &layers, snapshot)
                    .context("refresh linear-attn snapshot before batched verify")?;

                let r = run_speculative_decode_step_batched(
                    ordinal,
                    &geom,
                    mtp,
                    fwd_scratch,
                    chain_scratch,
                    embed_w,
                    &lm_head_w_buf,
                    &h_base,
                    next_token,
                    loop_state.position,
                    dynamic_k,
                    |inputs| -> anyhow::Result<Vec<(u32, Vec<u8>)>> {
                        run_batched_spec_verify_inputs(Qwen36BatchedSpecVerifyInputs {
                            ordinal,
                            geom: &geom,
                            store: &store,
                            weight_prefix,
                            layers: &mut layers,
                            persistent_scratch: persistent_scratch.as_mut(),
                            final_norm_w: &final_norm_w_buf,
                            lm_head_w: &lm_head_w_buf,
                            stage_timings: &mut stage_timings,
                            inputs,
                            emit_stage_timings,
                        })
                    },
                )
                .context("batched speculative decode step")?;

                // State mgmt: on partial-accept, restore + replay the
                // accepted prefix to advance linear-attn state correctly.
                // Full-accept (n_accepted == dynamic_k) needs no fixup —
                // K+1 chains advanced state through K+1 inputs which is
                // exactly what we committed (drafts + bonus's input was
                // drafts[K-1], one more chain's worth advances naturally
                // when next iter feeds the bonus token).
                if r.n_accepted < dynamic_k {
                    // Replay (j+1) chains: first_token at the current
                    // position, then the j accepted drafts at the
                    // following positions.
                    let replay = loop_state.speculative_replay_inputs(next_token, &r);
                    restore_and_replay_accepted_prefix(Qwen36SpecReplayAccepted {
                        ordinal,
                        geom: &geom,
                        store: &store,
                        weight_prefix,
                        layers: &mut layers,
                        snapshot,
                        persistent_scratch: persistent_scratch.as_mut(),
                        stage_timings: &mut stage_timings,
                        replay_inputs: &replay,
                        emit_stage_timings,
                    })?;
                }
                r
            } else {
                run_speculative_decode_step(
                    ordinal,
                    &geom,
                    mtp,
                    fwd_scratch,
                    chain_scratch,
                    embed_w,
                    &lm_head_w_buf,
                    &h_base,
                    next_token,
                    loop_state.position,
                    dynamic_k,
                    |position, input| -> anyhow::Result<(u32, Vec<u8>)> {
                        run_sequential_spec_verify_input(Qwen36SequentialSpecVerifyInput {
                            ordinal,
                            geom: &geom,
                            store: &store,
                            weight_prefix,
                            layers: &mut layers,
                            persistent_scratch: persistent_scratch.as_mut(),
                            final_norm_w: &final_norm_w_buf,
                            lm_head_w: &lm_head_w_buf,
                            final_hidden: &mut final_hidden_buf,
                            logits: &mut logits_buf,
                            counter: &mut counter_buf,
                            stage_timings: &mut stage_timings,
                            position,
                            input,
                            emit_stage_timings,
                        })
                    },
                )
                .context("speculative decode step")?
            };

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

    Ok(())
}
