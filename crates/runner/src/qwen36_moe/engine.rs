//! Qwen3.6-MoE runtime engine.
//!
//! Owns the CLI-facing Qwen3.6-MoE flow: bake selection, dry-run/budget
//! reporting, prompt setup, layer loading, session allocation, prefill,
//! generation, optional speculative extension, and final telemetry. The
//! GPU launch details live in the lower-level chain, persistent-decode,
//! generation, and spec-verify modules.

use std::{path::Path, ptr};

use anyhow::{Context, Result};
use gpu_hal::{set_backend, Backend};
use kernel_ffi::qwen36_moe::{
    Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights,
};
use model_store::manifest::QuantProfile;
use model_store::BakedStore;

use crate::profiling::{PrefillProfileScope, Qwen36DecodeProfileScope};
use crate::qwen36_moe_cli::bake::{ensure_qwen36_bake, select_decode_bake};
use crate::qwen36_moe_cli::chain::{run_chain_step, Qwen36ChainStep};
use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::dry_run::{print_report, run_qwen36_moe_dry_run, DryRunReport};
use crate::qwen36_moe_cli::generation::{run_generation_step, Qwen36GenerationStep};
use crate::qwen36_moe_cli::geom::build_multi_layer_geom;
use crate::qwen36_moe_cli::host::{lookup_embed_row, lookup_embed_row_timed};
use crate::qwen36_moe_cli::output::{
    print_decode_stream_start, print_generation_summary, print_last_logits_if_requested,
    print_sampling_summary,
};
use crate::qwen36_moe_cli::policy::{
    max_speculative_tokens_for_backend, metal_mtp_experiment_enabled, resolve_context_size,
    validate_cuda_v1_flags, validate_decode_backend, validate_metal_v1_flags,
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
use crate::qwen36_moe_speculative::SpeculativeStepResult;
use crate::qwen36_moe_telemetry::{print_and_write_moe_residency_summary, MoeRouteRuntime};
use crate::qwen36_moe_types::{LayerBuffers, PositionPair};
use crate::registry::RegistryEntry;

fn prewarm_qwen36_mps_static_topn_if_requested(
    ordinal: usize,
    backend: Backend,
    geom: &crate::qwen36_moe_types::MultiLayerGeom,
    layers: &mut [LayerBuffers],
) -> Result<std::time::Duration> {
    if backend != Backend::Metal
        || std::env::var_os("SUPERSONIC_METAL_PREWARM_QWEN36_FFN_MPS_STATIC_TOPN").is_none()
    {
        return Ok(std::time::Duration::ZERO);
    }

    let started = std::time::Instant::now();
    let mut attempted_layers = 0usize;
    let mut warmed_layers = 0usize;
    let mut allocations = 0usize;
    let mut copied_bytes = 0usize;
    let mut resident_capacity = 0usize;

    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        let ffn = &mut layer.ffn;
        let Some(int4) = &ffn.int4 else {
            continue;
        };
        attempted_layers += 1;
        let params = Qwen36MoeFfnStepParams {
            stage: 5,
            layer_idx: layer_idx as i32,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        };
        let weights = Qwen36MoeFfnStepWeights {
            input_hidden: ptr::null(),
            post_attn_norm_w: ffn.post_attn_norm_w.as_ptr(),
            gate_w: ffn.gate_w.as_ptr(),
            gate_up_proj_w: ffn.gate_up_proj_w.as_ptr(),
            down_proj_w: ffn.down_proj_w.as_ptr(),
            shared_gate_proj_w: ffn.shared_gate_proj_w.as_ptr(),
            shared_up_proj_w: ffn.shared_up_proj_w.as_ptr(),
            shared_down_proj_w: ffn.shared_down_proj_w.as_ptr(),
            shared_expert_gate_w: ffn.shared_expert_gate_w.as_ptr(),
        };
        let fp8 = int4.group_size < 0;
        let int4_ptrs = Qwen36MoeFfnStepInt4 {
            group_size: int4.group_size,
            gate_up_proj_type: int4.gate_up_proj_type,
            gate_up_proj_scale: int4.gate_up_proj_scale.as_ptr(),
            gate_up_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.gate_up_proj_zero.as_ptr()
            },
            down_proj_type: int4.down_proj_type,
            down_proj_scale: int4.down_proj_scale.as_ptr(),
            down_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.down_proj_zero.as_ptr()
            },
            shared_gate_proj_type: int4.shared_gate_proj_type,
            shared_gate_proj_scale: int4.shared_gate_proj_scale.as_ptr(),
            shared_gate_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_gate_proj_zero.as_ptr()
            },
            shared_up_proj_type: int4.shared_up_proj_type,
            shared_up_proj_scale: int4.shared_up_proj_scale.as_ptr(),
            shared_up_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_up_proj_zero.as_ptr()
            },
            shared_down_proj_type: int4.shared_down_proj_type,
            shared_down_proj_scale: int4.shared_down_proj_scale.as_ptr(),
            shared_down_proj_zero: if fp8 {
                ptr::null()
            } else {
                int4.shared_down_proj_zero.as_ptr()
            },
        };
        if let Some(stats) = kernel_ffi::qwen36_moe::qwen36_prewarm_mps_static_topn_rhs_for_metal(
            ordinal, params, &weights, &int4_ptrs,
        )
        .with_context(|| format!("prewarm Qwen3.6 MPS static top-N RHS layer {layer_idx}"))?
        {
            warmed_layers += 1;
            allocations += usize::from(stats.allocated);
            copied_bytes += stats.copied_bytes;
            resident_capacity = resident_capacity.max(stats.resident_capacity);
        }
    }

    let elapsed = started.elapsed();
    eprintln!(
        "[qwen36-moe ffn-prewarm] mode=mps-static-topn status=ok attempted_layers={} warmed_layers={} allocations={} resident_capacity={} copied_bytes={} elapsed_ms={:.3}",
        attempted_layers,
        warmed_layers,
        allocations,
        resident_capacity,
        copied_bytes,
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(elapsed)
}

/// Compute the `(rope, cache)` PositionPair for one step of the
/// decode loop. In dense mode the rope and cache agree; in
/// SpecPrefill mode the rope tracks the absolute prompt-token
/// position (during prefill of kept tokens) or the absolute
/// generation position (after prefill ends) while the cache slot
/// is the compact `loop_state_position`.
pub(crate) fn current_position(
    step: usize,
    loop_state_position: i32,
    keep_mask: Option<&Vec<bool>>,
    kept_positions: &[usize],
    effective_prompt_len: usize,
    full_prompt_len: usize,
) -> PositionPair {
    match keep_mask {
        None => PositionPair::dense(loop_state_position),
        Some(_) => {
            let rope = if step < effective_prompt_len {
                kept_positions[step] as i32
            } else {
                let gen_off = step - effective_prompt_len;
                (full_prompt_len + gen_off) as i32
            };
            PositionPair::split(rope, loop_state_position)
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen36MtpAcceptanceStats {
    mode: &'static str,
    steps: usize,
    drafted_tokens: usize,
    accepted_tokens: usize,
    emitted_tokens: usize,
    base_steps: usize,
    replay_steps: usize,
    full_accept_steps: usize,
    zero_accept_steps: usize,
    max_accept: usize,
}

impl Qwen36MtpAcceptanceStats {
    fn new(batched_spec_verify: bool) -> Self {
        Self {
            mode: if batched_spec_verify {
                "batched"
            } else {
                "sequential"
            },
            steps: 0,
            drafted_tokens: 0,
            accepted_tokens: 0,
            emitted_tokens: 0,
            base_steps: 0,
            replay_steps: 0,
            full_accept_steps: 0,
            zero_accept_steps: 0,
            max_accept: 0,
        }
    }

    fn record(&mut self, result: &SpeculativeStepResult) {
        self.steps += 1;
        self.drafted_tokens += result.n_drafted;
        self.accepted_tokens += result.n_accepted;
        self.emitted_tokens += result.emitted_tokens.len();
        self.base_steps += result.base_steps;
        self.replay_steps += result.replay_steps;
        self.max_accept = self.max_accept.max(result.n_accepted);
        if result.n_drafted > 0 && result.n_accepted == result.n_drafted {
            self.full_accept_steps += 1;
        }
        if result.n_accepted == 0 {
            self.zero_accept_steps += 1;
        }
    }

    fn print_if_requested(&self, enabled: bool) {
        if !enabled || self.steps == 0 {
            return;
        }
        let acceptance_rate = if self.drafted_tokens > 0 {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        } else {
            0.0
        };
        let emitted_per_step = self.emitted_tokens as f64 / self.steps as f64;
        let target_steps = self.base_steps + self.replay_steps;
        let target_steps_per_emitted = if self.emitted_tokens > 0 {
            target_steps as f64 / self.emitted_tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "[qwen36-mtp-acceptance] mode={} steps={} drafted_tokens={} \
             accepted_tokens={} acceptance_rate={:.6} emitted_tokens={} \
             emitted_per_step={:.6} base_steps={} replay_steps={} \
             target_steps_per_emitted={:.6} full_accept_steps={} \
             zero_accept_steps={} max_accept={}",
            self.mode,
            self.steps,
            self.drafted_tokens,
            self.accepted_tokens,
            acceptance_rate,
            self.emitted_tokens,
            emitted_per_step,
            self.base_steps,
            self.replay_steps,
            target_steps_per_emitted,
            self.full_accept_steps,
            self.zero_accept_steps,
            self.max_accept,
        );
    }
}

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
    let (context_size, context_size_source) = resolve_context_size(cli);
    validate_persistent_kv_fp8_flags(cli)?;
    validate_cuda_v1_flags(cli, entry)?;
    validate_metal_v1_flags(cli, entry)?;
    ensure_qwen36_bake(cli, entry)?;

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
    let requires_int4_bake = cli.int4 || matches!(entry.backend, Backend::Cuda | Backend::Metal);
    decode_text(
        &cli.model_dir,
        &report,
        &cli.prompt,
        cli.max_new_tokens.max(1),
        sampling,
        cli.emit_stage_timings,
        cli.speculative_decode,
        crate::bakes::effective_quant_profile(cli)?,
        requires_int4_bake,
        cli.batched_spec_verify,
        entry.backend,
        cli.device,
        // Phase 3e.4: persistent decode is now the default. The legacy
        // `--persistent-decode` flag is a hidden no-op (kept for harness
        // back-compat); `--no-persistent-decode` is the documented
        // opt-out for A/B comparison or bisecting megakernel regressions.
        entry.backend != Backend::Metal && !cli.no_persistent_decode,
        cli.kv_fp8,
        cli.dump_last_logits,
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        &cli.model,
        cli.ignore_eos,
        keep_mask,
        cli.progress_heartbeat_seconds,
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
    quant_profile: QuantProfile,
    int4_runtime: bool,
    batched_spec_verify: bool,
    backend: Backend,
    ordinal: usize,
    persistent_decode: bool,
    kv_fp8: bool,
    dump_last_logits: bool,
    profile_prefill: bool,
    profile_prefill_json: Option<&Path>,
    model_name: &str,
    ignore_eos: bool,
    keep_mask: Option<Vec<bool>>,
    progress_heartbeat_seconds: f64,
) -> Result<()> {
    validate_speculative_sampling(speculative_decode, sampling)?;

    if keep_mask.is_some() && speculative_decode {
        eprintln!(
            "[specprefill+mtp] composed run: rope on absolute prompt timeline, \
             cache on compact KV slot. See \
             docs/research/2026-05-04-qwen36-mtp-specprefill-audit.md."
        );
    }

    let decode_wall_start = std::time::Instant::now();
    let progress_interval = (progress_heartbeat_seconds > 0.0)
        .then(|| std::time::Duration::from_secs_f64(progress_heartbeat_seconds));
    let mut last_progress = decode_wall_start
        .checked_sub(progress_interval.unwrap_or(std::time::Duration::ZERO))
        .unwrap_or(decode_wall_start);
    let mut progress = |phase: &str, detail: String, force: bool| {
        let Some(interval) = progress_interval else {
            return;
        };
        let now = std::time::Instant::now();
        if force || now.duration_since(last_progress) >= interval {
            eprintln!(
                "[qwen36-moe progress] phase={phase} elapsed_ms={} {detail}",
                now.duration_since(decode_wall_start).as_millis()
            );
            last_progress = now;
        }
    };
    let weight_prefix = report.kernel_params.weight_prefix;

    progress("prompt_setup", "start".to_string(), true);
    let prompt_setup_start = std::time::Instant::now();
    let prompt_setup = prepare_prompt(model_dir, &report.config.text_config, prompt)?;
    let prompt_setup_elapsed = prompt_setup_start.elapsed();
    let tokenizer = prompt_setup.tokenizer;
    let prompt_ids = prompt_setup.prompt_ids;
    let eos_id = if ignore_eos {
        None
    } else {
        prompt_setup.eos_id
    };
    print_prompt_summary(prompt, &prompt_ids);

    progress(
        "prompt_setup",
        format!("done prompt_tokens={}", prompt_ids.len()),
        true,
    );
    progress("bake_open", "start".to_string(), true);
    let bake_open_start = std::time::Instant::now();
    let bake = select_decode_bake(model_dir, quant_profile, int4_runtime)?;
    if !bake.weight_mode.is_int4() {
        match backend {
            Backend::Cuda => anyhow::bail!(
                "Qwen3.6-35B-A3B CUDA v1 requires an INT4/q4km bake; selected {} from {}",
                bake.weight_mode.display_name(),
                bake.bake_dir.display(),
            ),
            Backend::Metal => anyhow::bail!(
                "Qwen3.6-35B-A3B Metal v1 requires an INT4-GPTQ bake; selected {} from {}",
                bake.weight_mode.display_name(),
                bake.bake_dir.display(),
            ),
            _ => {}
        }
    }
    println!(
        "  loading from bake: {} ({})",
        bake.bake_dir.display(),
        bake.weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake.bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake.bake_dir.display()))?;
    let bake_open_elapsed = bake_open_start.elapsed();
    progress(
        "bake_open",
        format!("done bake_dir={}", bake.bake_dir.display()),
        true,
    );

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
    progress(
        "layer_load",
        format!(
            "start layers={} kv_max_t={} sparse_vmm={:?}",
            geom.num_layers, kv_max_t, moe_runtime.vmm_mode
        ),
        true,
    );
    let layer_load_start = std::time::Instant::now();
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
        moe_runtime.fixed_hot_experts,
        moe_runtime.prefetch_mode,
        moe_runtime.prefetch_ranks,
        moe_runtime.transition_min_observations,
        moe_runtime.async_prefetch,
        moe_runtime.async_staging_pages,
        moe_runtime.prefetch_evict,
        moe_runtime.prefetch_evict_min_probability,
        persistent_decode,
    )?;
    let mut layers = loaded_layers.layers;
    let _ffn_prewarm_elapsed =
        prewarm_qwen36_mps_static_topn_if_requested(ordinal, backend, &geom, &mut layers)?;
    let layer_load_elapsed = layer_load_start.elapsed();
    let _moe_expert_arena = loaded_layers.moe_expert_arena;
    let mut _moe_expert_residency = loaded_layers.moe_expert_residency;
    let virtual_kv_stats = virtual_kv_stats_for_layers(&layers);
    print_virtual_kv_stats_if_active(virtual_kv_stats, kv_fp8, backend, ordinal);
    progress(
        "layer_load",
        format!("done elapsed_ms={}", layer_load_elapsed.as_millis()),
        true,
    );
    progress("session", "start".to_string(), true);
    let session_start = std::time::Instant::now();
    let max_speculative_tokens = max_speculative_tokens_for_backend(backend);
    if speculative_decode && backend == Backend::Metal && metal_mtp_experiment_enabled() {
        eprintln!(
            "[qwen36-mtp-metal-experiment] enabled=1 max_drafts={} verify=sequential status=experimental",
            max_speculative_tokens
        );
    }
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
        max_speculative_tokens,
        &mut layers,
    )?;
    let session_elapsed = session_start.elapsed();
    progress(
        "session",
        format!("done elapsed_ms={}", session_elapsed.as_millis()),
        true,
    );
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
    if kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_enabled() {
        kernel_ffi::qwen36_moe::qwen36_batched_prefill_feasibility_profile_configure(
            layers.len(),
            geom.top_k as usize,
            geom.num_experts as usize,
            crate::qwen36_moe_cli::batched_prefill::PREFILL_CHUNK_SIZE_WMMA_FULL,
            effective_prompt_len.saturating_sub(1),
        );
    }
    let backend_label = format!("{backend:?}");
    let mut prefill_profile = Some(PrefillProfileScope::new(
        profile_prefill,
        profile_prefill_json,
        "qwen3.6-moe",
        model_name,
        &backend_label,
        effective_prompt_len,
    ));

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
    let mtp_acceptance_profile =
        std::env::var_os("SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE").is_some();
    let mut mtp_acceptance_stats =
        speculative_decode.then(|| Qwen36MtpAcceptanceStats::new(batched_spec_verify));
    let mut prefill_steps = 0usize;
    let mut prefill_embed_elapsed = std::time::Duration::ZERO;
    let mut prefill_chain_elapsed = std::time::Duration::ZERO;
    let mut generation_wall_start = None;
    let mut decode_profile = None;
    let mut moe_routes = MoeRouteRuntime::new(
        geom.num_layers as usize,
        geom.top_k as usize,
        moe_runtime.sparse_requested,
        moe_runtime.prefetch_mode,
        moe_runtime.transition_min_observations,
        moe_runtime.hot_protect_min_hits,
        moe_runtime.fixed_hot_min_hits,
    );
    // Batched-Q prefill opt-in. Read once. When set the new chunked
    // host orchestrator drives the prefill range
    // `[0, effective_prompt_len - 1)` instead of the engine's main
    // per-step loop — see
    // docs/superpowers/plans/2026-05-05-qwen36-moe-batched-prefill-phase1.md.
    // M13: batched-prefill is the DEFAULT for Qwen 3.6 MoE. Bench at
    // 4K context (gfx1100, qwen3.6-35b-a3b INT4) shows 1.79x prefill
    // speedup vs the per-token persistent megakernel. Set
    // SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0 to revert to the legacy
    // per-token path (kept as a bisect/escape hatch).
    let batched_prefill_disabled = std::env::var("SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL")
        .map(|v| v == "0")
        .unwrap_or(false);

    let mut start_step = 0usize;
    let dense_prefill_token_loop =
        std::env::var_os("SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP").is_some();
    if batched_prefill_disabled
        && dense_prefill_token_loop
        && keep_mask.is_none()
        && _moe_expert_residency.is_none()
        && effective_prompt_len > 1
    {
        if let (Some(scratch), Some(embed_w)) = (persistent_scratch.as_mut(), embed_w_buf.as_ref())
        {
            let dense_prefill_count = effective_prompt_len - 1;
            let t_prefill = scratch
                .run_dense_prefill_tokens_from_device_embedding(
                    ordinal,
                    embed_w,
                    &prompt_ids[..dense_prefill_count],
                    0,
                    0,
                )
                .context("persistent dense prefill token loop")?;
            start_step = dense_prefill_count;
            prefill_steps += dense_prefill_count;
            prefill_chain_elapsed += t_prefill;
            loop_state.position += dense_prefill_count as i32;
            loop_state.current_token = prompt_ids[dense_prefill_count];
            eprintln!(
                "[qwen36-moe prefill-progress] mode=dense-token-loop variant=legacy \
                 chunks=1 tokens={} prefill_tokens={} last_context={} embed_ms={:.3} \
                 chain_ms={:.3} elapsed_ms={:.3}",
                dense_prefill_count,
                dense_prefill_count,
                dense_prefill_count,
                0.0,
                t_prefill.as_secs_f64() * 1000.0,
                t_prefill.as_secs_f64() * 1000.0,
            );
        }
    }

    if start_step == 0 && !batched_prefill_disabled && effective_prompt_len > 1 {
        let timings = crate::qwen36_moe_cli::batched_prefill::run_batched_prefill_stub(
            ordinal,
            &geom,
            &store,
            weight_prefix,
            &mut layers,
            persistent_scratch.as_mut(),
            _moe_expert_residency.as_mut(),
            &mut moe_runtime,
            &mut moe_routes,
            &mut loop_state,
            &prompt_ids,
            keep_mask.as_ref(),
            &kept_positions,
            effective_prompt_len,
            emit_stage_timings,
        )?;
        eprintln!(
            "[qwen36-moe batched-prefill] chunks={} tokens={} embed_ms={:.1} chain_ms={:.1}",
            timings.chunks,
            timings.tokens,
            timings.embed_total.as_secs_f64() * 1000.0,
            timings.chain_total.as_secs_f64() * 1000.0,
        );
        prefill_steps += timings.tokens;
        prefill_embed_elapsed += timings.embed_total;
        prefill_chain_elapsed += timings.chain_total;
        // After the orchestrator processes prefill steps
        // [0, effective_prompt_len - 1), the engine's main loop must
        // resume at the FIRST generation step (where logits are
        // computed). At that point `loop_state.position ==
        // effective_prompt_len - 1` (incremented once per processed
        // token) and `loop_state.current_token` is the LAST prompt
        // token (the one to fold into logits in the gen step).
        start_step = effective_prompt_len - 1;
    }

    for step in start_step..loop_state.total_steps {
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
            decode_profile = Some(Qwen36DecodeProfileScope::new_from_env());
        }
        // Per-step (rope, cache) pair. Dense mode: rope == cache.
        // SpecPrefill mode: rope on absolute prompt timeline, cache
        // on compact slot count. See `current_position` above.
        let position = current_position(
            step,
            loop_state.position,
            keep_mask.as_ref(),
            &kept_positions,
            effective_prompt_len,
            prompt_ids.len(),
        );
        progress(
            if is_gen_step { "generate" } else { "prefill" },
            format!(
                "step={} total_steps={} rope_position={} cache_position={} generated={} current_token={}",
                step,
                loop_state.total_steps,
                position.rope,
                position.cache,
                loop_state.generated_ids.len(),
                loop_state.current_token
            ),
            false,
        );
        if batched_prefill_disabled
            && dense_prefill_token_loop
            && !is_gen_step
            && keep_mask.is_none()
            && _moe_expert_residency.is_none()
        {
            if let (Some(scratch), Some(embed_w)) =
                (persistent_scratch.as_mut(), embed_w_buf.as_ref())
            {
                let t_chain_step = scratch
                    .run_from_device_embedding_no_download(
                        ordinal,
                        embed_w,
                        loop_state.current_token,
                        position.rope,
                        position.cache,
                    )
                    .with_context(|| {
                        format!(
                            "persistent dense prefill from device embedding \
                             (step {}, rope {}, cache {})",
                            step, position.rope, position.cache
                        )
                    })?;
                loop_state.position += 1;
                prefill_steps += 1;
                prefill_chain_elapsed += t_chain_step;
                loop_state.current_token = prompt_ids[kept_positions[step + 1]];
                continue;
            }
        }

        // Embed lookup for the current token.
        let t0 = std::time::Instant::now();
        let (initial_hidden, embed_lookup_timing) = if emit_stage_timings {
            let (row, timing) = lookup_embed_row_timed(
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
            (row, Some(timing))
        } else {
            let row = lookup_embed_row(
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
            (row, None)
        };
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
        let disable_folded_lm_head =
            std::env::var_os("SUPERSONIC_QWEN36_DISABLE_FOLDED_LM_HEAD").is_some();
        let folded_top1_enabled = matches!(logits_buf.backend(), Backend::Metal | Backend::Hip)
            && (sampling.temperature <= 0.0 || sampling.top_k == 1)
            && !dump_last_logits
            && std::env::var_os("SUPERSONIC_QWEN36_DUMP_LOGITS").is_none()
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LM_HEAD_GPU_ARGMAX").is_none();
        let fold = if is_gen_step && !disable_folded_lm_head {
            if folded_top1_enabled {
                Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                    final_norm_w: &final_norm_w_buf,
                    lm_head_w: &lm_head_w_buf,
                    logits_out: None,
                    top1_out: Some(&mut counter_buf),
                    vocab: geom.vocab,
                })
            } else {
                Some(crate::qwen36_moe_persistent_decode::LmHeadFold {
                    final_norm_w: &final_norm_w_buf,
                    lm_head_w: &lm_head_w_buf,
                    logits_out: Some(&mut logits_buf),
                    top1_out: None,
                    vocab: geom.vocab,
                })
            }
        } else {
            None
        };
        let final_hidden_observer_enabled = std::env::var_os("SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN")
            .is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_FINAL_HIDDEN_TAP").is_some()
            || std::env::var_os("SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP").is_some();
        let download_final_hidden = !is_gen_step
            || fold.is_none()
            || final_hidden_observer_enabled
            || mtp_buffers.is_some();
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
            position,
            step,
            is_gen_step,
            emit_stage_timings,
            fold,
            download_final_hidden,
        })?;
        let outputs = chain_step.outputs;
        let lm_head_folded = chain_step.lm_head_folded;
        let lm_head_folded_top1 = chain_step.lm_head_folded_top1;
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
            lm_head_folded_top1,
            dump_last_logits,
            tokenizer: tokenizer.as_ref(),
            sampling,
            t_embed_step,
            embed_lookup_timing,
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
        if let Some(profile) = prefill_profile.take() {
            profile.finish()?;
        }

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
                base_position: position,
                h_base_in: &h_base,
                first_token: next_token,
                stage_timings: &mut stage_timings,
                emit_stage_timings,
                max_drafts: max_speculative_tokens,
            })?;

            if let Some(stats) = mtp_acceptance_stats.as_mut() {
                stats.record(&result);
            }
            if loop_state.append_speculative_emissions(&result, tokenizer.as_ref(), eos_id) {
                break;
            }
        }
    }

    if let Some(profile) = prefill_profile.take() {
        profile.finish()?;
    }
    if let Some(profile) = decode_profile.take() {
        profile.finish();
    }

    print_last_logits_if_requested(dump_last_logits, &loop_state.last_logits_bytes);
    let generation_wall_ms = generation_wall_start
        .as_ref()
        .map(|start| start.elapsed().as_secs_f64() * 1000.0);
    print_generation_summary(
        &loop_state.generated_ids,
        prompt_ids.len(),
        eos_id,
        generation_wall_ms,
    );
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
            generation_wall_ms.unwrap_or(0.0),
            to_ms(decode_wall_start.elapsed()),
        );
    }
    if let Some(stats) = mtp_acceptance_stats.as_ref() {
        stats.print_if_requested(mtp_acceptance_profile || emit_stage_timings);
    }
    emit_mpp_pilot_if_requested(emit_stage_timings);
    emit_mps_expert_pilot_if_requested(emit_stage_timings);

    Ok(())
}

fn emit_mpp_pilot_if_requested(emit_stage_timings: bool) {
    if !emit_stage_timings || std::env::var_os("SUPERSONIC_METAL_QWEN36_MPP_PILOT").is_none() {
        return;
    }
    let size = std::env::var("SUPERSONIC_METAL_QWEN36_MPP_PILOT_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(2048);
    let iterations = std::env::var("SUPERSONIC_METAL_QWEN36_MPP_PILOT_ITERS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(5);
    match kernel_ffi::qwen36_moe::metal_mpp_tile_gemm_f16_tflops(size, iterations) {
        Ok(tflops) => eprintln!(
            "[qwen36-moe mpp-pilot] status=ok size={} iterations={} tile_m=64 tile_n=32 tile_k=64 tflops={:.3}",
            size, iterations, tflops
        ),
        Err(err) => eprintln!(
            "[qwen36-moe mpp-pilot] status=error size={} iterations={} tflops=0.000 error={}",
            size, iterations, err
        ),
    }
}

fn emit_mps_expert_pilot_if_requested(emit_stage_timings: bool) {
    if !emit_stage_timings || std::env::var_os("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT").is_none()
    {
        return;
    }
    let hidden = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_HIDDEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2048);
    let moe_intermediate = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_MOE_INTERMEDIATE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let top_k = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let iterations = std::env::var("SUPERSONIC_METAL_QWEN36_MPS_EXPERT_ITERS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(100);
    match kernel_ffi::qwen36_moe::metal_mps_expert_f16_probe(
        hidden,
        moe_intermediate,
        top_k,
        iterations,
    ) {
        Ok(probe) => eprintln!(
            "[qwen36-moe mps-expert-pilot] status=ok hidden={} moe_intermediate={} top_k={} iterations={} gate_up_ms={:.3} down_ms={:.3} gate_up_tflops={:.3} down_tflops={:.3}",
            hidden,
            moe_intermediate,
            top_k,
            iterations,
            probe.gate_up_ms,
            probe.down_ms,
            probe.gate_up_tflops,
            probe.down_tflops,
        ),
        Err(err) => eprintln!(
            "[qwen36-moe mps-expert-pilot] status=error hidden={} moe_intermediate={} top_k={} iterations={} gate_up_ms=0.000 down_ms=0.000 gate_up_tflops=0.000 down_tflops=0.000 error={}",
            hidden, moe_intermediate, top_k, iterations, err
        ),
    }
}
