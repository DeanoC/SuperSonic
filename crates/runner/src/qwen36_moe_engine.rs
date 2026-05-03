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

use anyhow::{anyhow, Context, Result};
use gpu_hal::{set_backend, Backend};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_decode::{
    argmax_bf16_logits, host_final_norm_lm_head, run_chained_decode, run_chained_decode_fast,
    run_chained_decode_fast_with_expert_prefetch, sample_bf16_logits, ExpertPrefetchPhase,
    ExpertRoute, MultiLayerGeom, XorshiftRng,
};
use crate::qwen36_moe_dry_run::{
    print_report, run_qwen36_moe_dry_run, ContextSizeSource, DryRunReport,
};
use crate::qwen36_moe_host::{host_load_bytes, load_lm_head_bf16, lookup_embed_row};
use crate::qwen36_moe_layers::{load_layer_buffers, Qwen36WeightMode};
use crate::qwen36_moe_prefetch::handle_moe_expert_prefetch;
use crate::qwen36_moe_session::{prepare_decode_session, Qwen36DecodeSession};
use crate::qwen36_moe_speculative::{
    run_speculative_decode_step, run_speculative_decode_step_batched,
};
use crate::qwen36_moe_state::{refresh_linear_attn_state, restore_linear_attn_state};
use crate::qwen36_moe_telemetry::{
    MoeIslandPrefetchMode, MoeRouteTelemetry, MoeSparseTelemetry, MoeSparseTelemetrySnapshot,
    MoeTransitionPredictor,
};
use crate::qwen36_moe_timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_vmm::{
    load_decode_layers_with_vmm_strategy, moe_island_cap_experts_from_env,
    moe_island_prefetch_ranks_from_env, moe_island_prefetch_transition_min_observations,
    should_use_qwen36_kv_vmm, virtual_kv_stats_for_layers, MoeExpertVmmMode,
};
use crate::registry::{Qwen36MoeKernelParams, RegistryEntry};

const MIB: f64 = (1024 * 1024) as f64;
const QWEN36_NUM_SPECULATIVE_TOKENS: usize = 3;

pub fn run(cli: &crate::Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    ensure_qwen36_bake(cli, entry)?;

    // Derive context_size + an honest source flag so the printed report can
    // tell the user which of three answers they got: explicit, prompt-derived
    // estimate, or worst-case defaults-only. The `--context-size` path is
    // verbatim; otherwise we fall back to (prompt char count) + max_new_tokens
    // when a prompt is given (chars are an upper bound on tokens for
    // English-ish text), or just max_new_tokens when the user gave neither
    // flag — that last case undercounts KV bytes for any realistic session,
    // and the report flags it.
    let max_new = cli.max_new_tokens.max(1);
    let (context_size, context_size_source) = if let Some(ctx) = cli.context_size {
        (ctx, ContextSizeSource::Explicit)
    } else if !cli.prompt.is_empty() {
        (
            cli.prompt.chars().count() + max_new,
            ContextSizeSource::EstimatedFromPrompt,
        )
    } else {
        (max_new, ContextSizeSource::MaxNewTokensOnly)
    };
    if cli.kv_fp8 && cli.no_persistent_decode {
        anyhow::bail!(
            "--kv-fp8 for Qwen3.6-35B-A3B requires the persistent megakernel; \
             remove --no-persistent-decode (persistent is on by default). The \
             back-compat step kernels stay BF16-KV."
        );
    }

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

    // The decode kernels (`kernels/qwen36_moe.hip`, the per-block step
    // launchers in `kernel-ffi`) are HIP-only. The registry currently has
    // both HIP and CUDA entries for `qwen3.6-35b-a3b` but the CUDA branches
    // of `attn_step_launch` / `linear_step_launch` / `ffn_step_launch` all
    // return `InvalidArg("CUDA backend not yet wired")`. Fail here with a
    // clear message instead of letting the engine commit to HIP buffers
    // (which would crash later inside the kernel-ffi wrappers when the
    // registry-selected backend disagrees).
    if entry.backend != Backend::Hip {
        anyhow::bail!(
            "qwen3.6-35b-a3b decode kernels are HIP-only at this stage; \
             registry-selected backend was {:?}. Re-run with --backend hip, \
             or use --dry-run for analytic accounting.",
            entry.backend,
        );
    }

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

fn ensure_qwen36_bake(cli: &crate::Cli, entry: &RegistryEntry) -> Result<()> {
    // Auto-download the requested release bake if missing or stale. 35B-A3B
    // INT4 calibration OOMs on 24 GiB hosts, so release-hosted bakes are the
    // realistic default for decode and for dry-run residency probes. Run this
    // before dry-run reporting so `SUPERSONIC_VMM_*_PROBE` can inspect a
    // freshly populated bake.
    let variant = if cli.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Int4Gptq
    };
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    // `should_fetch_exact_bake` honors --download-bake (force) and refuses to
    // fetch when an up-to-date bake is already present.
    let force_download = cli.download_bake;
    if !cli.no_download
        && crate::should_fetch_exact_bake(force_download, model_store::version_ok(&bake_dir))
    {
        let canonical_model = entry.model.to_string();
        match crate::try_download_bake(cli, variant, &canonical_model, &bake_dir) {
            Ok(true) => eprintln!(
                "[fetch] installed qwen3.6-MoE {} bake at {}",
                if cli.fp8_runtime { "FP8" } else { "INT4" },
                bake_dir.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!(
                "[fetch] qwen3.6-MoE {} bake fetch failed: {e}",
                if cli.fp8_runtime { "FP8" } else { "INT4" }
            ),
        }
    }
    Ok(())
}

/// Build the geometry the chained decoder needs from the parsed config +
/// the registry's per-family params. Mirrors what
/// `oracle/qwen36_moe_multilayer_oracle.py` puts in `config` and what
/// `MultiLayerGeom` consumes.
fn build_multi_layer_geom(
    text_config: &TextConfig,
    kernel_params: &Qwen36MoeKernelParams,
) -> MultiLayerGeom {
    MultiLayerGeom {
        hidden: text_config.hidden_size as i32,
        vocab: text_config.vocab_size as i32,
        num_layers: text_config.num_hidden_layers as i32,
        rms_norm_eps: text_config.rms_norm_eps as f32,

        num_attention_heads: text_config.num_attention_heads as i32,
        num_kv_heads: text_config.num_key_value_heads as i32,
        head_dim: text_config.head_dim as i32,
        rotary_dim: text_config.rotary_dim() as i32,
        rope_theta: text_config.rope_theta() as f32,

        num_k_heads: text_config.linear_num_key_heads as i32,
        num_v_heads: text_config.linear_num_value_heads as i32,
        head_k_dim: text_config.linear_key_head_dim as i32,
        head_v_dim: text_config.linear_value_head_dim as i32,
        conv_kernel_dim: text_config.linear_conv_kernel_dim as i32,

        num_experts: kernel_params.num_experts as i32,
        moe_intermediate: kernel_params.moe_intermediate_size as i32,
        shared_intermediate: kernel_params.shared_expert_intermediate_size as i32,
        top_k: kernel_params.top_k as i32,
    }
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
    use std::io::Write as _;

    // Greedy-only gate for speculative decode. The Phase 6.3 protocol
    // verifies MTP drafts via greedy argmax against the base model's
    // logits — extending it to non-greedy sampling needs rejection
    // sampling (Speculative Decoding §3 in vLLM's reference), which
    // hasn't been implemented. Reject up front rather than silently
    // mix `argmax` (verify) with `sample_bf16_logits` (regular sample),
    // which would emit a different distribution than plain decode and
    // break reproducibility.
    //
    // `sample_bf16_logits` falls back to argmax when `temperature <= 0`
    // (or `top_k == 1`), so any non-trivial sampling configuration —
    // any of `temperature > 0`, `top_k != 1`, `top_p < 1.0` — counts
    // as non-greedy and is rejected here.
    if speculative_decode {
        let is_greedy = sampling.temperature <= 0.0 || sampling.top_k == 1;
        if !is_greedy {
            anyhow::bail!(
                "--speculative-decode currently supports greedy sampling \
                 only (temperature ≤ 0 or top_k == 1). Got temperature={}, \
                 top_k={}, top_p={}. Phase 6.4 will add sampling-consistent \
                 verification (rejection sampling); until then, re-run with \
                 `--temperature 0` for speculative decode, or drop \
                 `--speculative-decode` for non-greedy sampling.",
                sampling.temperature,
                sampling.top_k,
                sampling.top_p
            );
        }
    }

    let weight_prefix = report.kernel_params.weight_prefix;

    // Tokenizer first — without it we can't tokenize the prompt or stream
    // decoded text. Falls back to BOS-only if the tokenizer can't load.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path).ok();

    let bos_id = report
        .config
        .text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let eos_id = report
        .config
        .text_config
        .eos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let prompt_ids: Vec<u32> = match (&tokenizer, prompt.is_empty()) {
        (Some(tok), false) => {
            let enc = tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            if ids.is_empty() {
                vec![bos_id]
            } else {
                ids
            }
        }
        _ => vec![bos_id],
    };
    println!(
        "  prompt: {prompt:?} → {} token{} ({:?}{}…)",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        &prompt_ids[..prompt_ids.len().min(8)],
        if prompt_ids.len() > 8 { ", " } else { "" },
    );

    // Pick the bake. INT4 remains the default small-VRAM path; explicit
    // --fp8-runtime selects the native FP8 bake.
    let fp8_dir = model_store::bake_dir_fp8(model_dir);
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if fp8_runtime {
        if !fp8_dir.exists() {
            return Err(anyhow!(
                "--fp8-runtime requested but no FP8-native bake exists at {}. \
                 Create one with `python3 oracle/bake_fp8.py --model-dir {}`.",
                fp8_dir.display(),
                model_dir.display()
            ));
        }
        (fp8_dir, Qwen36WeightMode::Fp8)
    } else if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if bf16_dir.exists() {
        (bf16_dir, Qwen36WeightMode::Bf16)
    } else {
        return Err(anyhow!(
            "decode requires a baked package — neither FP8-native ({}), \
             INT4-GPTQ ({}) nor BF16 ({}) exists. Create one with the standard bake pipeline \
             or re-run with --dry-run for analytic accounting only.",
            fp8_dir.display(),
            int4_dir.display(),
            bf16_dir.display()
        ));
    };
    println!(
        "  loading from bake: {} ({})",
        bake_dir.display(),
        weight_mode.display_name(),
    );
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(backend);

    // KV cache size: needs to fit prompt_len + max_new past tokens. Sized
    // generously here since per-layer KV is small (10 full-attn layers ×
    // [kv_max_t, Hkv*d=512] BF16 = 10 KiB per token of context).
    let kv_max_t = prompt_ids.len() + max_new;

    println!(
        "  loading {} layers ({} INT4 sidecar sets, KV cache cap = {} tokens)…",
        geom.num_layers,
        if weight_mode.is_int4() {
            geom.num_layers
        } else {
            0
        },
        kv_max_t,
    );

    let moe_vmm_mode = MoeExpertVmmMode::from_env()?;
    let moe_island_cap_experts = moe_island_cap_experts_from_env()?;
    if moe_island_cap_experts.is_some() && speculative_decode {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS sparse residency is not wired through speculative decode yet"
        );
    }
    if moe_island_cap_experts.is_some() && moe_vmm_mode == MoeExpertVmmMode::Disabled {
        anyhow::bail!(
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS requires VMM expert slabs; unset SUPERSONIC_VMM_MOE_ISLANDS=0"
        );
    }
    let sparse_moe_requested = moe_island_cap_experts.is_some();
    let moe_prefetch_mode = MoeIslandPrefetchMode::from_env()?;
    let moe_prefetch_ranks =
        moe_island_prefetch_ranks_from_env(moe_prefetch_mode, geom.top_k as usize)?;
    let moe_transition_min_observations =
        moe_island_prefetch_transition_min_observations(moe_prefetch_mode)?;
    if moe_prefetch_mode != MoeIslandPrefetchMode::Disabled && !sparse_moe_requested {
        anyhow::bail!("SUPERSONIC_MOE_ISLAND_PREFETCH requires SUPERSONIC_MOE_ISLAND_CAP_EXPERTS");
    }
    let mut moe_sparse_telemetry = MoeSparseTelemetry::from_env(
        sparse_moe_requested,
        persistent_decode,
        moe_prefetch_mode,
        moe_prefetch_ranks,
    )?;
    if let Some(path) = moe_sparse_telemetry
        .as_ref()
        .and_then(|telemetry| telemetry.dump_path.as_ref())
    {
        println!(
            "  [vmm] sparse MoE residency telemetry will be written to {}",
            path.display()
        );
    }
    let kv_vmm = should_use_qwen36_kv_vmm(backend, ordinal)?;
    let loaded_layers = load_decode_layers_with_vmm_strategy(
        &store,
        ordinal,
        backend,
        &geom,
        &report.config.text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        moe_vmm_mode,
        moe_island_cap_experts,
        moe_prefetch_mode,
        moe_prefetch_ranks,
        moe_transition_min_observations,
        persistent_decode,
    )?;
    let mut layers = loaded_layers.layers;
    let _moe_expert_arena = loaded_layers.moe_expert_arena;
    let mut _moe_expert_residency = loaded_layers.moe_expert_residency;
    let virtual_kv_stats = virtual_kv_stats_for_layers(&layers);
    if virtual_kv_stats.layers > 0 {
        println!(
            "  [vmm] Qwen3.6-MoE {} KV active on backend={} device {ordinal}: \
             layers={} mappings={} logical={:.2}MiB logical_resident={:.2}MiB \
             resident={:.2}MiB reserved={:.2}MiB",
            if kv_fp8 { "FP8" } else { "BF16" },
            backend,
            virtual_kv_stats.layers,
            virtual_kv_stats.mappings,
            virtual_kv_stats.logical_bytes as f64 / MIB,
            virtual_kv_stats.logical_resident_bytes as f64 / MIB,
            virtual_kv_stats.resident_bytes as f64 / MIB,
            virtual_kv_stats.reserved_bytes as f64 / MIB,
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

    println!(
        "  decoding {} prompt token{} + generating ≤{} new token{}…",
        prompt_ids.len(),
        if prompt_ids.len() == 1 { "" } else { "s" },
        max_new,
        if max_new == 1 { "" } else { "s" },
    );
    println!();
    print!("> ");
    if let Some(tok) = &tokenizer {
        if let Ok(prompt_text) = tok.decode(&prompt_ids, false) {
            print!("{prompt_text}");
        }
    }
    std::io::stdout().flush().ok();

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    // Track the BF16 logits bytes from the last decode step for --dump-last-logits.
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    let mut current_token: u32 = prompt_ids[0];
    let mut position: i32 = 0;
    // Standard prefill+generate shape: feed prompt[0..N-1] as prefill (logits
    // discarded), then prompt[N-1] is the first forward whose logits we
    // sample. Subsequent gen steps feed the just-sampled token. Total
    // forwards = (prompt_len - 1) prefill + max_new generation = prompt_len
    // + max_new - 1.
    let total_steps = prompt_ids.len() + max_new - 1;
    let mut rng = XorshiftRng::new(sampling.seed);
    println!(
        "  sampling: temp={} top_k={} top_p={} seed={}",
        sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed,
    );

    // Per-stage wall-clock accumulators. Aggregated across generation steps
    // only (prefill steps run the chain but skip the lm_head/sample stages,
    // so timing prefill mixed with gen would distort the per-token average).
    // `chain_ms` includes the GPU work + the D2H copy of `final_hidden_bytes`
    // — `run_chained_decode` syncs before returning, so the wall-clock here
    // is a real GPU+sync measurement. CPU-side stages (embed lookup, lm_head
    // GEMV, sampling, detokenize) are pure host work.
    let mut stage_timings = Qwen36StageTimingTotals::default();
    let mut previous_moe_topk_by_layer: Vec<Vec<usize>> =
        vec![Vec::new(); geom.num_layers as usize];
    let mut moe_route_telemetry =
        sparse_moe_requested.then(|| MoeRouteTelemetry::new(geom.top_k as usize));
    let mut moe_transition_predictors = moe_prefetch_mode.transition_weighted().then(|| {
        vec![
            MoeTransitionPredictor::new(geom.top_k as usize, moe_transition_min_observations,);
            geom.num_layers as usize
        ]
    });

    for step in 0..total_steps {
        // When speculative decode is on, each iteration can commit
        // multiple tokens (up to K+1), so the standard `total_steps =
        // prompt_len + max_new - 1` count over-shoots. Break here once
        // we've already committed `max_new` tokens — otherwise the
        // next regular chain call would request a cache slot beyond
        // `kv_max_t = prompt_len + max_new` (status 120). Plain decode
        // stays bit-identical because it always emits exactly one
        // token per iteration.
        if generated_ids.len() >= max_new {
            break;
        }
        // Embed lookup for the current token.
        let t0 = std::time::Instant::now();
        let initial_hidden = lookup_embed_row(
            &store,
            weight_prefix,
            current_token as usize,
            geom.hidden as usize,
        )
        .with_context(|| format!("embed lookup token {current_token} (step {step})"))?;
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
        let lm_head_folded;
        let moe_telemetry_before = _moe_expert_residency
            .as_ref()
            .map(MoeSparseTelemetrySnapshot::capture);
        let track_moe_routes =
            moe_prefetch_mode.uses_previous_token_routes() || moe_route_telemetry.is_some();
        let mut next_moe_topk_by_layer = if track_moe_routes {
            previous_moe_topk_by_layer.clone()
        } else {
            Vec::new()
        };
        let outputs = if let Some(scratch) = persistent_scratch.as_mut() {
            if let Some(manager) = _moe_expert_residency.as_mut() {
                // Sparse VMM needs a host remap point after each layer's
                // router top-k is known. The segmented persistent path keeps
                // the persistent phase bodies but skips folded lm_head; the
                // standalone lm_head launch below consumes final_hidden_bytes.
                lm_head_folded = false;
                drop(fold);
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        &store,
                        moe_prefetch_mode,
                        moe_prefetch_ranks,
                        &previous_moe_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        moe_route_telemetry.as_mut(),
                        moe_transition_predictors.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                scratch
                    .run_sparse_with_expert_prefetch(
                        ordinal,
                        &initial_hidden,
                        position,
                        &mut prefetch,
                    )
                    .with_context(|| {
                        format!(
                            "segmented persistent sparse decode (step {step}, position {position})"
                        )
                    })?
            } else {
                lm_head_folded = fold.is_some();
                scratch
                    .run(ordinal, &initial_hidden, position, fold)
                    .with_context(|| {
                        format!("persistent decode (step {step}, position {position})")
                    })?
            }
        } else {
            // Chained path doesn't support the fold; lm_head still
            // launches separately below on gen steps.
            lm_head_folded = false;
            drop(fold);
            if let Some(manager) = _moe_expert_residency.as_mut() {
                let mut prefetch = |phase: ExpertPrefetchPhase,
                                    layer_idx: usize,
                                    routes: &[ExpertRoute]|
                 -> Result<()> {
                    handle_moe_expert_prefetch(
                        manager,
                        &store,
                        moe_prefetch_mode,
                        moe_prefetch_ranks,
                        &previous_moe_topk_by_layer,
                        &mut next_moe_topk_by_layer,
                        track_moe_routes,
                        moe_route_telemetry.as_mut(),
                        moe_transition_predictors.as_deref_mut(),
                        phase,
                        layer_idx,
                        routes,
                    )
                };
                run_chained_decode_fast_with_expert_prefetch(
                    ordinal,
                    &geom,
                    &mut layers,
                    &initial_hidden,
                    position,
                    emit_stage_timings,
                    &mut prefetch,
                )
            } else {
                run_chained_decode_fast(
                    ordinal,
                    &geom,
                    &mut layers,
                    &initial_hidden,
                    position,
                    emit_stage_timings,
                )
            }
            .with_context(|| format!("chained decode (step {step}, position {position})"))?
        };
        if track_moe_routes {
            previous_moe_topk_by_layer = next_moe_topk_by_layer;
        }
        if let (Some(telemetry), Some(before), Some(manager)) = (
            moe_sparse_telemetry.as_mut(),
            moe_telemetry_before,
            _moe_expert_residency.as_ref(),
        ) {
            let after = MoeSparseTelemetrySnapshot::capture(manager);
            telemetry.record_step(step, position, is_gen_step, before, after);
        }
        let t_chain_step = t1.elapsed();
        position += 1;

        // KV-FP8 sidecar descriptors stay fixed across decode. The
        // persistent kernel computes the rolling covered range from
        // `position` and `kv_shadow_window`, so no descriptor re-upload is
        // needed when old sidecar slots roll over.

        // Prefill steps: feed the next prompt token without computing logits.
        if step + 1 < prompt_ids.len() {
            current_token = prompt_ids[step + 1];
            continue;
        }

        // Optional dump for the host-side post-chain debug harness.
        if let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN") {
            std::fs::write(&dump_path, &outputs.final_hidden_bytes)
                .with_context(|| format!("write final_hidden dump to {dump_path}"))?;
            eprintln!(
                "[debug] dumped step={step} position={position} final_hidden ({} BF16 bytes) to {dump_path}",
                outputs.final_hidden_bytes.len()
            );
        }

        // Generation step: when the megakernel didn't fold lm_head
        // (chained path), launch the standalone final RMSnorm +
        // lm_head GEMV here. When folded (persistent + gen), skip the
        // launch — `logits_buf` is already populated by the
        // megakernel.
        let t2 = std::time::Instant::now();
        if !lm_head_folded {
            gpu_hal::copy_h2d(
                ordinal,
                final_hidden_buf.as_mut_ptr(),
                outputs.final_hidden_bytes.as_ptr() as *const _,
                outputs.final_hidden_bytes.len(),
            )
            .context("h2d final_hidden -> final_hidden_buf")?;
            kernel_ffi::qwen36_moe::lm_head_launch(
                ordinal,
                geom.hidden,
                geom.vocab,
                geom.rms_norm_eps,
                &final_hidden_buf,
                &final_norm_w_buf,
                &lm_head_w_buf,
                &mut logits_buf,
                None, // base decode doesn't capture h_post — that's MTP-only
                &mut counter_buf,
            )
            .context("gpu lm_head launch")?;
        }
        let logits = logits_buf
            .to_host_bytes()
            .context("d2h logits from GPU lm_head")?;
        if dump_last_logits {
            last_logits_bytes.clone_from(&logits);
        }
        let t_lm_head_step = t2.elapsed();
        if let Ok(dump_path) = std::env::var("SUPERSONIC_QWEN36_DUMP_LOGITS") {
            std::fs::write(&dump_path, &logits)
                .with_context(|| format!("write logits dump to {dump_path}"))?;
            eprintln!(
                "[debug] dumped step={step} logits ({} BF16 bytes) to {dump_path}",
                logits.len()
            );
        }
        let t3 = std::time::Instant::now();
        let next_token = sample_bf16_logits(
            &logits,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            &mut rng,
        );
        let t_sample_step = t3.elapsed();
        generated_ids.push(next_token);

        // Stream-decode and print.
        let t4 = std::time::Instant::now();
        if let Some(tok) = &tokenizer {
            if let Ok(text) = tok.decode(&[next_token], false) {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
        }
        let t_detok_step = t4.elapsed();

        stage_timings.record_generation_step(
            t_embed_step,
            t_chain_step,
            t_lm_head_step,
            t_sample_step,
            t_detok_step,
            &outputs,
        );

        if Some(next_token) == eos_id {
            break;
        }
        current_token = next_token;

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
            if generated_ids.len() >= max_new {
                break;
            }
            // Cap K to the remaining max_new headroom so the verify
            // loop never writes cache slots beyond what we'll
            // actually commit to `generated_ids`. Spec emits up to
            // K+1 tokens (K accepted + 1 corrected/bonus), so the
            // available draft count is `headroom - 1`. If headroom <=
            // 1 we can still emit 1 token via the K=0 fallback; if
            // headroom == 0 we already broke out above.
            let headroom = max_new - generated_ids.len();
            let dynamic_k = QWEN36_NUM_SPECULATIVE_TOKENS.min(headroom.saturating_sub(1));
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
                    position,
                    dynamic_k,
                    |inputs| -> anyhow::Result<Vec<(u32, Vec<u8>)>> {
                        let n = inputs.len();
                        if n == 0 {
                            return Ok(Vec::new());
                        }
                        let hidden = geom.hidden as usize;

                        // K+1 sequential chains, accumulate final_hiddens.
                        let mut final_hiddens: Vec<Vec<u8>> = Vec::with_capacity(n);
                        for &(pos, input) in inputs {
                            let t_embed_start = std::time::Instant::now();
                            let initial_hidden = lookup_embed_row(
                                &store,
                                weight_prefix,
                                input as usize,
                                geom.hidden as usize,
                            )?;
                            stage_timings.record_embed(t_embed_start.elapsed());

                            let t_chain_start = std::time::Instant::now();
                            let chain_outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                                scratch.run(ordinal, &initial_hidden, pos, None)?
                            } else {
                                run_chained_decode_fast(
                                    ordinal,
                                    &geom,
                                    &mut layers,
                                    &initial_hidden,
                                    pos,
                                    emit_stage_timings,
                                )?
                            };
                            stage_timings.record_chain(t_chain_start.elapsed(), &chain_outputs);
                            stage_timings.count_generation_step();
                            final_hiddens.push(chain_outputs.final_hidden_bytes);
                        }

                        // ONE batched lm_head over [n, hidden].
                        let t_lm_head_start = std::time::Instant::now();
                        let mut concat = Vec::with_capacity(n * hidden * 2);
                        for fh in &final_hiddens {
                            concat.extend_from_slice(fh);
                        }
                        let fh_buf = gpu_hal::GpuBuffer::from_host_bytes(
                            ordinal,
                            gpu_hal::ScalarType::BF16,
                            &[n, hidden],
                            &concat,
                        )?;
                        let mut logits_buf_b = gpu_hal::GpuBuffer::zeros(
                            ordinal,
                            gpu_hal::ScalarType::BF16,
                            &[n, geom.vocab as usize],
                        )?;
                        kernel_ffi::qwen36_moe::lm_head_batched_launch(
                            ordinal,
                            n as i32,
                            geom.hidden,
                            geom.vocab,
                            geom.rms_norm_eps,
                            &fh_buf,
                            &final_norm_w_buf,
                            &lm_head_w_buf,
                            &mut logits_buf_b,
                            None,
                        )?;
                        let logits_bytes =
                            logits_buf_b.to_host_bytes().context("d2h batched logits")?;
                        stage_timings.record_lm_head(t_lm_head_start.elapsed());

                        let row_bytes = geom.vocab as usize * 2;
                        let mut results: Vec<(u32, Vec<u8>)> = Vec::with_capacity(n);
                        for (i, fh) in final_hiddens.into_iter().enumerate() {
                            let row = &logits_bytes[i * row_bytes..(i + 1) * row_bytes];
                            results.push((argmax_bf16_logits(row), fh));
                        }
                        Ok(results)
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
                    restore_linear_attn_state(ordinal, &mut layers, snapshot)
                        .context("restore linear-attn state after partial-accept")?;
                    // Replay (j+1) chains: first_token at `position`,
                    // then the j accepted drafts at `position+1..position+j`.
                    let mut replay: Vec<(i32, u32)> = Vec::with_capacity(r.n_accepted + 1);
                    replay.push((position, next_token));
                    for (i, &tok) in r.emitted_tokens.iter().take(r.n_accepted).enumerate() {
                        replay.push((position + 1 + i as i32, tok));
                    }
                    for &(pos, input) in &replay {
                        let t_embed_start = std::time::Instant::now();
                        let initial_hidden = lookup_embed_row(
                            &store,
                            weight_prefix,
                            input as usize,
                            geom.hidden as usize,
                        )?;
                        stage_timings.record_embed(t_embed_start.elapsed());
                        let t_chain_start = std::time::Instant::now();
                        let replay_outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                            scratch.run(ordinal, &initial_hidden, pos, None)?
                        } else {
                            run_chained_decode_fast(
                                ordinal,
                                &geom,
                                &mut layers,
                                &initial_hidden,
                                pos,
                                emit_stage_timings,
                            )?
                        };
                        stage_timings.record_chain(t_chain_start.elapsed(), &replay_outputs);
                        // Per-kernel-class breakdown for replay chains
                        // contributes to the same accumulators as the
                        // verify chains so `--emit-stage-timings` reports
                        // honest full-attn/linear-attn/ffn averages on
                        // partial-accept iters. Without this the reported
                        // chain breakdown undercounts actual work as the
                        // accept rate falls.
                        stage_timings.count_generation_step();
                    }
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
                    position,
                    dynamic_k,
                    |pos, input| -> anyhow::Result<(u32, Vec<u8>)> {
                        // Embed lookup is its own stage in the timing
                        // breakdown — bundling it into `t_chain` (as the
                        // first cut of this closure did) systematically
                        // inflates `chain_ms_avg` and deflates
                        // `embed_ms_avg` as the MTP accept rate rises.
                        let t_embed_start = std::time::Instant::now();
                        let initial_hidden = lookup_embed_row(
                            &store,
                            weight_prefix,
                            input as usize,
                            geom.hidden as usize,
                        )?;
                        stage_timings.record_embed(t_embed_start.elapsed());

                        let t_chain_start = std::time::Instant::now();
                        let outputs = if let Some(scratch) = persistent_scratch.as_mut() {
                            scratch.run(ordinal, &initial_hidden, pos, None)?
                        } else {
                            run_chained_decode_fast(
                                ordinal,
                                &geom,
                                &mut layers,
                                &initial_hidden,
                                pos,
                                emit_stage_timings,
                            )?
                        };
                        stage_timings.record_chain(t_chain_start.elapsed(), &outputs);

                        let t_lm_head_start = std::time::Instant::now();
                        gpu_hal::copy_h2d(
                            ordinal,
                            final_hidden_buf.as_mut_ptr(),
                            outputs.final_hidden_bytes.as_ptr() as *const _,
                            outputs.final_hidden_bytes.len(),
                        )?;
                        kernel_ffi::qwen36_moe::lm_head_launch(
                            ordinal,
                            geom.hidden,
                            geom.vocab,
                            geom.rms_norm_eps,
                            &final_hidden_buf,
                            &final_norm_w_buf,
                            &lm_head_w_buf,
                            &mut logits_buf,
                            None,
                            &mut counter_buf,
                        )?;
                        let logits_bytes = logits_buf
                            .to_host_bytes()
                            .context("d2h logits from spec verify lm_head")?;
                        stage_timings.record_lm_head(t_lm_head_start.elapsed());
                        // Each verify base step counts as one decode step
                        // for the per-token average — emitted_tokens.len()
                        // tokens are committed per spec call, and
                        // closure-call-count == emitted_tokens.len() in
                        // both the partial-accept and full-accept (with
                        // bonus) cases. Bumping here is equivalent to
                        // "one closure call = one decode step worth of
                        // base work."
                        stage_timings.count_generation_step();
                        Ok((
                            argmax_bf16_logits(&logits_bytes),
                            outputs.final_hidden_bytes,
                        ))
                    },
                )
                .context("speculative decode step")?
            };

            // Append emitted tokens. Honour `max_new` and EOS by
            // breaking out cleanly mid-emission.
            let mut hit_stop = false;
            for tok in result.emitted_tokens.iter().copied() {
                if generated_ids.len() >= max_new {
                    hit_stop = true;
                    break;
                }
                generated_ids.push(tok);
                if let Some(t) = &tokenizer {
                    if let Ok(text) = t.decode(&[tok], false) {
                        print!("{text}");
                    }
                }
                if Some(tok) == eos_id {
                    hit_stop = true;
                    break;
                }
            }
            std::io::stdout().flush().ok();
            position += result.emitted_tokens.len() as i32;
            if hit_stop {
                break;
            }
            current_token = *result
                .emitted_tokens
                .last()
                .expect("speculative step must emit at least one token (K=0 fallback ensured)");
        }
    }

    // Emit last-step logits for integration parity tests (--dump-last-logits).
    // Printed BEFORE any other post-loop output so the test parser can grep for
    // the first line starting with "LAST_LOGITS: ".
    if dump_last_logits && !last_logits_bytes.is_empty() {
        use std::io::Write as _;
        let logits_f32 = crate::qwen36_moe_decode::bf16_bytes_to_f32(&last_logits_bytes);
        // Lead with `\n` so the marker lands at the start of its own line —
        // the streamed-token print path uses `print!` without a trailing
        // newline, so `LAST_LOGITS:` would otherwise concatenate onto the
        // last generated text and `lines().find(...starts_with...)` in the
        // parity/smoke tests wouldn't match.
        // Format with `{}` (Display) instead of `{:.6}` to preserve full
        // f32 precision — the VMM bit-exact smoke compares via
        // `a.to_bits() == b.to_bits()`, which fails on rounded values.
        print!("\nLAST_LOGITS: ");
        for (i, x) in logits_f32.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    println!();
    println!();
    println!(
        "Generated {} token{} ({} prompt + {} new). EOS: {}.",
        generated_ids.len(),
        if generated_ids.len() == 1 { "" } else { "s" },
        prompt_ids.len(),
        generated_ids.len(),
        if eos_id
            .map(|e| generated_ids.last() == Some(&e))
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no (max_new_tokens hit)"
        },
    );
    if !generated_ids.is_empty() {
        println!("  Generated ids: {generated_ids:?}");
    }
    if let Some(manager) = _moe_expert_residency.as_ref() {
        let residency = manager.stats();
        let arena = manager.arena().stats();
        let total_resident_bytes = arena.resident_bytes + virtual_kv_stats.resident_bytes;
        let total_reserved_bytes = arena.reserved_bytes + virtual_kv_stats.reserved_bytes;
        if let Some(telemetry) = moe_sparse_telemetry.as_ref() {
            println!(
                "  [vmm] MoE island residency: resident_slices={} peak_slices={} \
                 resident_pages={} peak_pages={} page_backed_slices={} \
                 hits={} misses={} page_hits={} page_misses={} evicted_slices={} evicted_pages={} \
                 prefetch_requests={} prefetch_hits={} prefetch_misses={} \
                 prefetch_page_hits={} prefetch_page_misses={} \
                 prefetch_skipped={} prefetch_skipped_pages={} \
                 uploaded={:.2}MiB unmapped={:.2}MiB \
                 resident={:.2}MiB peak_resident={:.2}MiB reserved={:.2}MiB \
                 kv_resident={:.2}MiB total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
                residency.resident_slices,
                telemetry.peak_resident_slices,
                residency.resident_pages,
                telemetry.peak_resident_pages,
                residency.page_backed_slices,
                residency.hits,
                residency.misses,
                residency.page_hits,
                residency.page_misses,
                residency.evicted_slices,
                residency.evicted_pages,
                residency.prefetch_requests,
                residency.prefetch_hits,
                residency.prefetch_misses,
                residency.prefetch_page_hits,
                residency.prefetch_page_misses,
                residency.prefetch_skipped,
                residency.prefetch_skipped_pages,
                residency.uploaded_bytes as f64 / MIB,
                residency.unmapped_bytes as f64 / MIB,
                arena.resident_bytes as f64 / MIB,
                telemetry.peak_resident_bytes as f64 / MIB,
                arena.reserved_bytes as f64 / MIB,
                virtual_kv_stats.resident_bytes as f64 / MIB,
                total_resident_bytes as f64 / MIB,
                total_reserved_bytes as f64 / MIB,
            );
            telemetry.write_json(
                manager,
                virtual_kv_stats,
                &generated_ids,
                moe_route_telemetry.as_ref(),
            )?;
        } else {
            println!(
                "  [vmm] MoE island residency: resident_slices={} resident_pages={} \
                 page_backed_slices={} hits={} misses={} page_hits={} page_misses={} \
                 evicted_slices={} evicted_pages={} prefetch_requests={} \
                 prefetch_hits={} prefetch_misses={} prefetch_page_hits={} \
                 prefetch_page_misses={} prefetch_skipped={} prefetch_skipped_pages={} \
                 uploaded={:.2}MiB unmapped={:.2}MiB \
                 resident={:.2}MiB reserved={:.2}MiB kv_resident={:.2}MiB \
                 total_vmm_resident={:.2}MiB total_vmm_reserved={:.2}MiB",
                residency.resident_slices,
                residency.resident_pages,
                residency.page_backed_slices,
                residency.hits,
                residency.misses,
                residency.page_hits,
                residency.page_misses,
                residency.evicted_slices,
                residency.evicted_pages,
                residency.prefetch_requests,
                residency.prefetch_hits,
                residency.prefetch_misses,
                residency.prefetch_page_hits,
                residency.prefetch_page_misses,
                residency.prefetch_skipped,
                residency.prefetch_skipped_pages,
                residency.uploaded_bytes as f64 / MIB,
                residency.unmapped_bytes as f64 / MIB,
                arena.resident_bytes as f64 / MIB,
                arena.reserved_bytes as f64 / MIB,
                virtual_kv_stats.resident_bytes as f64 / MIB,
                total_resident_bytes as f64 / MIB,
                total_reserved_bytes as f64 / MIB,
            );
        }
    }
    if emit_stage_timings && stage_timings.gen_steps > 0 {
        let to_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let chain_ms = to_ms(stage_timings.chain);
        let embed_ms = to_ms(stage_timings.embed);
        let lm_head_ms = to_ms(stage_timings.lm_head);
        let sample_ms = to_ms(stage_timings.sample);
        let detok_ms = to_ms(stage_timings.detok);
        let total_ms = chain_ms + embed_ms + lm_head_ms + sample_ms + detok_ms;
        let n = stage_timings.gen_steps as f64;
        let full_attn_ms = (stage_timings.chain_full_attn_us as f64) / 1000.0;
        let linear_attn_ms = (stage_timings.chain_linear_attn_us as f64) / 1000.0;
        let ffn_ms = (stage_timings.chain_ffn_us as f64) / 1000.0;
        eprintln!(
            "[qwen36-moe stage-timings] gen_steps={} \
             embed_ms_avg={:.3} chain_ms_avg={:.3} lm_head_ms_avg={:.3} \
             sample_ms_avg={:.3} detok_ms_avg={:.3} total_ms_avg={:.3} \
             (chain_total_ms={:.1} lm_head_total_ms={:.1})",
            stage_timings.gen_steps,
            embed_ms / n,
            chain_ms / n,
            lm_head_ms / n,
            sample_ms / n,
            detok_ms / n,
            total_ms / n,
            chain_ms,
            lm_head_ms,
        );
        eprintln!(
            "[qwen36-moe chain-breakdown] gen_steps={} \
             full_attn_ms_avg={:.3} linear_attn_ms_avg={:.3} ffn_ms_avg={:.3} \
             (full_attn_total_ms={:.1} linear_attn_total_ms={:.1} ffn_total_ms={:.1})",
            stage_timings.gen_steps,
            full_attn_ms / n,
            linear_attn_ms / n,
            ffn_ms / n,
            full_attn_ms,
            linear_attn_ms,
            ffn_ms,
        );
    }

    Ok(())
}

/// Load + dequantize lm_head into a BF16 byte buffer that the host-side
/// Legacy single-token entry point — keeps the original `decode_first_token`
/// callable so the path stays exercised. Currently unused but documents the
/// minimal one-step decode shape.
#[allow(dead_code)]
fn decode_first_token(model_dir: &Path, report: &DryRunReport, kv_fp8: bool) -> Result<u32> {
    let weight_prefix = report.kernel_params.weight_prefix;

    // Pick the bake. INT4 is the realistic path on 24 GiB VRAM.
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if bf16_dir.exists() {
        (bf16_dir, Qwen36WeightMode::Bf16)
    } else {
        return Err(anyhow!(
            "decode requires a baked package — neither INT4-GPTQ ({}) nor \
             BF16 ({}) exists. Create one with the standard bake pipeline \
             or re-run with --dry-run for analytic accounting only.",
            int4_dir.display(),
            bf16_dir.display()
        ));
    };
    println!(
        "  loading from bake: {} ({})",
        bake_dir.display(),
        if weight_mode == Qwen36WeightMode::Int4 {
            "INT4 GPTQ"
        } else {
            "BF16"
        }
    );
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open BakedStore at {}", bake_dir.display()))?;

    let geom = build_multi_layer_geom(&report.config.text_config, &report.kernel_params);

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    println!(
        "  loading {} layer{} ({} INT4 sidecar set{})…",
        geom.num_layers,
        if geom.num_layers == 1 { "" } else { "s" },
        if weight_mode == Qwen36WeightMode::Int4 {
            geom.num_layers
        } else {
            0
        },
        if geom.num_layers == 1 { "" } else { "s" },
    );
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            &store,
            ordinal,
            li,
            &geom,
            &report.config.text_config,
            weight_prefix,
            weight_mode,
            0, // legacy single-token path: no KV cache, kv_len=1 fast path.
            kv_fp8,
            false,
            None,
            None,
        )
        .with_context(|| format!("load layer {li} weights"))?;
        layers.push(layer);
    }

    // BOS token: if the config exposes one, prefer it; otherwise default to
    // 0. Either way the parity criterion is "doesn't bail and emits a token",
    // and the produced token id reflects whatever embedding row we picked.
    let bos = report
        .config
        .text_config
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let initial_hidden = lookup_embed_row(&store, weight_prefix, bos, geom.hidden as usize)
        .with_context(|| format!("lookup embed row {bos}"))?;
    println!(
        "  embedding row {bos} loaded ({} BF16 bytes)",
        initial_hidden.len()
    );

    println!("  running chained decode…");
    let outputs = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, 0)
        .context("chained decode")?;
    println!(
        "  decode done; final hidden norm = {:.4}",
        crate::qwen36_moe_decode::bf16_bytes_to_f32(&outputs.final_hidden_bytes)
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    );

    let final_norm_bytes = host_load_bytes(&store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    let lm_head_bf16_bytes =
        load_lm_head_bf16(&store, &report.config.text_config, weight_prefix, &geom)
            .context("prepare lm_head BF16 buffer")?;

    println!("  computing host-side norm + lm_head GEMV…");
    let logits = host_final_norm_lm_head(
        &outputs.final_hidden_bytes,
        &final_norm_bytes,
        &lm_head_bf16_bytes,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );
    Ok(argmax_bf16_logits(&logits))
}

#[cfg(test)]
mod tests {
    use super::{ExpertRoute, MoeIslandPrefetchMode, MoeRouteTelemetry, MoeTransitionPredictor};

    #[test]
    fn moe_prefetch_mode_env_accepts_disabled_and_previous_token_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(None).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("disabled")).unwrap(),
            MoeIslandPrefetchMode::Disabled
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("prev-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousToken
        );
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_resident_only_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous-token-resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("previous_token_resident")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("resident-previous-token")).unwrap(),
            MoeIslandPrefetchMode::PreviousTokenResidentOnly
        );
        assert!(MoeIslandPrefetchMode::from_env_value(Some("resident")).is_err());
    }

    #[test]
    fn moe_prefetch_mode_env_accepts_transition_aliases() {
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(
            MoeIslandPrefetchMode::from_env_value(Some("transition-weighted")).unwrap(),
            MoeIslandPrefetchMode::Transition
        );
        assert_eq!(MoeIslandPrefetchMode::Transition.as_str(), "transition");
        assert!(MoeIslandPrefetchMode::Transition.uses_previous_token_routes());
        assert!(MoeIslandPrefetchMode::Transition.transition_weighted());
    }

    #[test]
    fn moe_transition_predictor_waits_for_warmup_and_scores_repeats() {
        let mut predictor = MoeTransitionPredictor::new(3, 2);
        let previous_routes = [10, 20, 30];
        let routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 20,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 99,
                weight: 0.25,
            },
        ];

        predictor.update(&routes, &previous_routes);
        assert!(predictor.candidates(&previous_routes, 2).is_empty());

        predictor.update(&routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20]);

        let later_routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 10,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 20,
                weight: 0.25,
            },
        ];
        predictor.update(&later_routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20, 10]);
    }

    #[test]
    fn moe_route_telemetry_records_previous_rank_transition_matrix() {
        let mut telemetry = MoeRouteTelemetry::new(3);
        let previous_routes = [7, 11, 13];
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 0,
                expert_idx: 11,
                weight: 0.5,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 1,
                expert_idx: 7,
                weight: 0.25,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 2,
                expert_idx: 99,
                weight: 0.125,
            },
            &previous_routes,
        );

        assert_eq!(telemetry.observations_by_rank, vec![1, 1, 1]);
        assert_eq!(telemetry.repeated_previous_by_rank, vec![1, 1, 0]);
        assert_eq!(
            telemetry.repeated_previous_rank_by_current_rank,
            vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]]
        );
        assert_eq!(
            telemetry
                .to_json()
                .get("repeated_previous_rank_by_current_rank")
                .unwrap(),
            &serde_json::json!([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        );
        let json = telemetry.to_json();
        assert_eq!(
            json.get("repeated_previous_probability_by_current_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("same_rank_repeat_probability_by_rank").unwrap(),
            &serde_json::json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            json.get("repeated_current_by_previous_rank").unwrap(),
            &serde_json::json!([1, 1, 0])
        );
        assert_eq!(
            json.get("repeated_current_probability_by_previous_rank")
                .unwrap(),
            &serde_json::json!([1.0, 1.0, 0.0])
        );
        assert_eq!(
            json.get("best_previous_rank_by_current_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_current_rank_by_previous_rank").unwrap(),
            &serde_json::json!([1, 0, null])
        );
        assert_eq!(
            json.get("best_transition").unwrap(),
            &serde_json::json!({
                "current_rank": 0,
                "previous_rank": 1,
                "count": 1,
                "probability_by_current_rank": 1.0,
            })
        );
    }
}
